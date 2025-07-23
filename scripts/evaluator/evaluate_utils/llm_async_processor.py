import asyncio
import functools
import traceback
from typing import Any, TypeAlias, List, Tuple, Optional

import backoff
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
import openai
import pydantic_core

from config_singleton import WandbConfigSingleton
from llm_inference_adapter import LLMResponse


MAX_TRIES = 50  # リトライ回数を50回に削減（100回は多すぎる）

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

    def __init__(
        self,
        llm: object,
        inputs: Inputs = [],
        batch_size: Optional[int] = None,
        inference_interval: Optional[float] = None,
    ):
        instance = WandbConfigSingleton.get_instance()
        cfg = instance.config
        self.llm = llm
        self.inputs = inputs
        self.batch_size = batch_size or cfg.get("batch_size", 256)
        self.inference_interval = inference_interval or cfg.inference_interval
        self.semaphore = asyncio.Semaphore(self.batch_size)

    @error_handler
    @backoff.on_exception(
        backoff.expo, 
        (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, openai.APIError, pydantic_core.ValidationError),
        max_tries=MAX_TRIES,
        max_time=1800,  # 最大30分でタイムアウト
        jitter=backoff.full_jitter
    )
    async def _ainvoke(self, messages: Messages, **kwargs) -> Any:
        """非同期でLLMを呼び出す統一メソッド"""
        await asyncio.sleep(self.inference_interval)
        try:
            async with self.semaphore:
                return await self.llm.ainvoke(messages, **kwargs)
        except pydantic_core.ValidationError as e:
            # JSONパースエラーの場合は、エラー内容をログに出力してから再スロー
            print(f"JSON parsing error occurred: {str(e)}")
            print(f"Retrying due to JSON validation error...")
            raise  # backoffデコレータがリトライを処理

    def _assert_messages_format(self, data: Messages):
        """メッセージフォーマットの検証"""
        # データがリストであることを確認
        assert isinstance(data, list), "Data should be a list"
        # 各要素が辞書であることを確認
        for item in data:
            assert isinstance(item, dict), "Each item should be a dictionary"
            # 'role'キーと'content'キーが存在することを確認
            assert "role" in item, "'role' key is missing in an item"
            assert "content" in item, "'content' key is missing in an item"
            # 'role'の値が'system', 'assistant', 'user'のいずれかであることを確認
            roles = {"system", "assistant", "user"}
            assert item["role"] in roles, f"'role' should be one of {str(roles)}"
            # 'content'の値が文字列であることを確認
            assert isinstance(item["content"], str), "'content' should be a string"

    async def _gather_tasks(self) -> List[LLMResponse]:
        """すべてのタスクを収集して実行"""
        # 入力データの検証
        for messages, _ in self.inputs:
            self._assert_messages_format(data=messages)

        tasks = [self._ainvoke(messages, **kwargs) for messages, kwargs in self.inputs]
        return await atqdm.gather(*tasks, desc=f"Processing requests")

    def get_results(self) -> List[LLMResponse]:
        """結果を取得（同期的なエントリーポイント）"""
        return asyncio.run(self._gather_tasks())

    async def get_results_async(self) -> List[LLMResponse]:
        """結果を取得（非同期版）"""
        return await self._gather_tasks()

    def process_single(self, messages: Messages, **kwargs) -> LLMResponse:
        """単一のメッセージを処理（同期版）"""
        return asyncio.run(self.process_single_async(messages, **kwargs))

    async def process_single_async(self, messages: Messages, **kwargs) -> LLMResponse:
        """単一のメッセージを処理（非同期版）"""
        self._assert_messages_format(messages)
        return await self._ainvoke(messages, **kwargs)

    def add_input(self, messages: Messages, **kwargs):
        """新しい入力を追加"""
        self.inputs.append((messages, kwargs))

    def clear_inputs(self):
        """入力をクリア"""
        self.inputs.clear()

    def get_input_count(self) -> int:
        """入力数を取得"""
        return len(self.inputs)

    def set_batch_size(self, batch_size: int):
        """バッチサイズを設定"""
        self.batch_size = batch_size

    def set_inference_interval(self, interval: float):
        """推論間隔を設定"""
        self.inference_interval = interval

    async def process_with_callback(self, callback_func=None) -> List[Any]:
        """コールバック関数付きで処理"""
        results = []
        
        for i in tqdm(range(0, len(self.inputs), self.batch_size), desc="Processing with callback"):
            batch = self.inputs[i : i + self.batch_size]
            batch_results = await self._process_batch(batch)
            results.extend(batch_results)
            
            # コールバック関数が指定されている場合は実行
            if callback_func:
                await callback_func(batch_results, i // self.batch_size)
        
        return results

    def get_statistics(self) -> dict:
        """統計情報を取得"""
        return {
            "total_inputs": len(self.inputs),
            "batch_size": self.batch_size,
            "inference_interval": self.inference_interval,
            "estimated_batches": (len(self.inputs) + self.batch_size - 1) // self.batch_size
        }
