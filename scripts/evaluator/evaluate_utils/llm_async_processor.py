import asyncio
import functools
import traceback
from typing import Any, TypeAlias, List, Tuple
from openai import OpenAI
import backoff
from langchain.schema import AIMessage
from tqdm import tqdm

from config_singleton import WandbConfigSingleton


MAX_TRIES = 100

Messages: TypeAlias = List[dict[str, str]]
Inputs: TypeAlias = List[Tuple[Messages, dict[str, Any]]]


def error_handler(func: callable) -> callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = traceback.format_exc()
            print(error_message)
            raise

    return wrapper


class LLMAsyncProcessor:
    def __init__(self, llm: object, inputs: List[Messages]):
        instance = WandbConfigSingleton.get_instance()
        cfg = instance.config
        self.llm = llm
        self.inputs = inputs
        self.api_type = cfg.api
        self.model_name = cfg.model.pretrained_model_name_or_path
        self.batch_size = cfg.get("batch_size", 256)
        self.inference_interval = cfg.inference_interval

    @error_handler
    @backoff.on_exception(backoff.expo, Exception, max_tries=MAX_TRIES)
    def _invoke(self, messages: Messages) -> Tuple[AIMessage, float]:
        if self.api_type == "google":
            for i in range(n:=10):
                response = self.llm.invoke(messages)
                if response.content.strip():
                    break
                else:
                    print(f"Retrying request due to empty content. Retry attempt {i+1} of {n}.")
        elif self.api_type == "amazon_bedrock":
            response = self.llm.invoke(messages)
        elif self.api_type == "openai":
            client = OpenAI()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages]
            )
            return AIMessage(content=response.choices[0].message.content)
        else:
            raise NotImplementedError(
                f"Synchronous invoke is not implemented for API type: {self.api_type}"
            )
        return response

    @error_handler
    @backoff.on_exception(backoff.expo, Exception, max_tries=MAX_TRIES)
    async def _ainvoke(self, messages: Messages) -> Tuple[AIMessage, float]:
        await asyncio.sleep(self.inference_interval)
        if self.api_type in ["google", "amazon_bedrock", "openai"]:
            return await asyncio.to_thread(self._invoke, messages)
        else:
            if self.model_name == "tokyotech-llm/Swallow-7b-instruct-v0.1":
                return await self.llm.ainvoke(messages, stop=["</s>"])
            else:
                return await self.llm.ainvoke(messages)

    async def _process_batch(self, batch: List[Messages]) -> List[Tuple[AIMessage, float]]:
        tasks = [
            asyncio.create_task(self._ainvoke(messages))
            for messages in batch
        ]
        return await asyncio.gather(*tasks)

    def _assert_messages_format(self, data: Messages):
        assert isinstance(data, list), "Data should be a list"
        for item in data:
            assert isinstance(item, dict), "Each item should be a dictionary"
            assert "role" in item, "'role' key is missing in an item"
            assert "content" in item, "'content' key is missing in an item"
            roles = {"system", "assistant", "user"}
            assert item["role"] in roles, f"'role' should be one of {str(roles)}"
            assert isinstance(item["content"], str), "'content' should be a string"

    async def _gather_tasks(self) -> List[Tuple[AIMessage, float]]:
        for messages in self.inputs:
            self._assert_messages_format(data=messages)
        results = []
        for i in tqdm(range(0, len(self.inputs), self.batch_size)):
            batch = self.inputs[i : i + self.batch_size]
            batch_results = await self._process_batch(batch)
            results.extend(batch_results)
        return results

    def get_results(self) -> List[Tuple[AIMessage, float]]:
        return asyncio.run(self._gather_tasks())