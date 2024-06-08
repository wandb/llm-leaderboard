import asyncio
import time
from typing import Any, TypeAlias, List, Tuple

import backoff
from langchain.schema import AIMessage

Messages: TypeAlias = List[dict[str, str]]
Inputs: TypeAlias = List[Tuple[Messages, dict[str, Any]]]

MAX_TRIES = 100

class LLMAsyncProcessor:
    """
    LLMAsyncProcessorクラスは、指定されたLLM（大規模言語モデル）を使用して非同期にメッセージを処理するためのユーティリティクラスです。
    """

    def __init__(self, llm: object, inputs: Inputs, batch_size: int):
        self.llm = llm
        self.inputs = inputs
        self.batch_size = batch_size  # あとで引数ではないところで定義するようにします

    @backoff.on_exception(backoff.expo, Exception, max_tries=MAX_TRIES)
    async def _ainvoke(self, messages: Messages, **kwargs) -> Tuple[AIMessage, float]:
        start = time.time()
        response = await self.llm.ainvoke(messages, **kwargs)
        end = time.time()
        latency = end - start
        return response, latency

    async def _process_batch(self, batch: Inputs) -> List[Tuple[AIMessage, float]]:
        tasks = [
            asyncio.create_task(self._ainvoke(messages, **kwargs))
            for messages, kwargs in batch
        ]
        return await asyncio.gather(*tasks)

    async def _gather_tasks(self) -> List[Tuple[AIMessage, float]]:
        results = []
        for i in range(0, len(self.inputs), self.batch_size):
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
