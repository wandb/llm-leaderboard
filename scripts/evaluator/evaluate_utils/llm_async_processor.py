import asyncio
import functools
import traceback
from typing import Any, TypeAlias, List, Tuple

import backoff
from langchain.schema import AIMessage
from tqdm import tqdm

from config_singleton import WandbConfigSingleton
from .chat_bedrock import chat_bedrock

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
    """
    LLMAsyncProcessorクラスは、指定されたLLM（大規模言語モデル）を使用して非同期にメッセージを処理するためのユーティリティクラスです。
    """

    def __init__(self, llm: object, inputs: Inputs):
        instance = WandbConfigSingleton.get_instance()
        cfg = instance.config
        self.llm = llm
        self.inputs = inputs
        self.api_type = cfg.api
        self.batch_size = cfg.get("batch_size", 256)
        self.inference_interval = cfg.inference_interval

    @backoff.on_exception(backoff.expo, Exception, max_tries=MAX_TRIES)
    @error_handler
    def _invoke(self, messages: Messages, **kwargs) -> Tuple[AIMessage, float]:
        if self.api_type == "google":
            self.llm.max_output_tokens = kwargs["max_tokens"]
            for i in range(10):
                response = self.llm.invoke(messages)
                if response.content.strip():
                    break
                else:
                    print(f"Try {i+1}")
        else:
            raise NotImplementedError(
                "Synchronous invoke is only implemented for Google API"
            )
        return response, 0

    @backoff.on_exception(backoff.expo, Exception, max_tries=MAX_TRIES)
    @error_handler
    async def _ainvoke(self, messages: Messages, **kwargs) -> Tuple[AIMessage, float]:
        await asyncio.sleep(self.inference_interval)
        if self.api_type == "google":
            # Synchronous call for Google API
            return await asyncio.to_thread(self._invoke, messages, **kwargs)
        elif self.api_type == "amazon_bedrock":
            return await asyncio.to_thread(chat_bedrock, messages, **kwargs)
        else:
            response = await self.llm.ainvoke(messages, **kwargs)
        return response, 0

    async def _process_batch(self, batch: Inputs) -> List[Tuple[AIMessage, float]]:
        tasks = [
            asyncio.create_task(self._ainvoke(messages, **kwargs))
            for messages, kwargs in batch
        ]
        return await asyncio.gather(*tasks)

    def _assert_messages_format(self, data: Messages):
        # データがリストであることを確認
        assert isinstance(data, list), "Data should be a list"
        # 各要素が辞書であることを確認
        for item in data:
            assert isinstance(item, dict), "Each item should be a dictionary"
            # 'role'キーと'content'キーが存在することを確認
            assert 'role' in item, "'role' key is missing in an item"
            assert 'content' in item, "'content' key is missing in an item"
            # 'role'の値が'system', 'assistant', 'user'のいずれかであることを確認
            assert item['role'] in ['system', 'assistant', 'user'], "'role' should be one of ['system', 'assistant', 'user']"
            # 'content'の値が文字列であることを確認
            assert isinstance(item['content'], str), "'content' should be a string"

    async def _gather_tasks(self) -> List[Tuple[AIMessage, float]]:
        for messages, _ in self.inputs:
            self._assert_messages_format(data=messages)
        results = []
        for i in tqdm(range(0, len(self.inputs), self.batch_size)):
            batch = self.inputs[i : i + self.batch_size]
            batch_results = await self._process_batch(batch)
            results.extend(batch_results)
        return results

    def get_results(self) -> List[Tuple[AIMessage, float]]:
        return asyncio.run(self._gather_tasks())


# 使用例（llmとinputsは適切なオブジェクトとデータで置き換えてください）
# llm_instance = YourLLMClass()
# inputs_data = [(messages1, kwargs1), (messages2, kwargs2), ...]
# processor = LLMAsyncProcessor(llm_instance, inputs_data, batch_size=10)
# results = processor.get_results()
