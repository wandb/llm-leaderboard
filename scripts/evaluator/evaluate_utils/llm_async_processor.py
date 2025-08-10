import asyncio
import functools
import traceback
import json
from typing import Any, TypeAlias, List, Tuple, Optional

import backoff
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
import openai
import pydantic_core

from config_singleton import WandbConfigSingleton
from llm_inference_adapter import LLMResponse

# Cohere例外をインポート（存在する場合）
try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False


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

    def _normalize_generation_kwargs(self, kwargs: dict) -> dict:
        """Normalize generation parameter aliases to a common schema.

        - Map alias keys like 'max_new_tokens', 'max_new_token',
          'max_output_tokens', 'max_completion_tokens' to 'max_tokens'
        - Do not overwrite an explicitly provided 'max_tokens'
        - Remove alias keys once normalized to avoid leaking unknown params
        """
        if not isinstance(kwargs, dict):
            return kwargs

        normalized = dict(kwargs)

        # If max_tokens already present, respect it
        if "max_tokens" in normalized and normalized["max_tokens"] is not None:
            # Clean up obvious aliases to reduce noise
            for alias_key in ("max_new_tokens", "max_new_token", "max_output_tokens", "max_completion_tokens"):
                if alias_key in normalized:
                    normalized.pop(alias_key)
            return normalized

        # Resolve from aliases in priority order
        alias_priority = (
            "max_tokens",
            "max_new_tokens",
            "max_new_token",
            "max_output_tokens",
            "max_completion_tokens",
        )

        chosen_value = None
        chosen_key = None
        for key in alias_priority:
            if key in normalized and normalized.get(key) is not None:
                chosen_value = normalized.get(key)
                chosen_key = key
                break

        if chosen_value is not None:
            normalized["max_tokens"] = chosen_value

        # Remove alias keys except canonical one
        for alias_key in ("max_new_tokens", "max_new_token", "max_output_tokens", "max_completion_tokens"):
            if alias_key in normalized:
                # Keep the original only if it is the canonical we selected and no max_tokens was set separately
                if alias_key != chosen_key:
                    normalized.pop(alias_key)

        return normalized

    @error_handler
    @backoff.on_exception(
        backoff.expo, 
        tuple(filter(None, [
            # OpenAI例外
            openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, 
            openai.InternalServerError, 
            # Cohere例外（利用可能な場合）
            getattr(cohere, 'TooManyRequestsError', None) if COHERE_AVAILABLE else None,
            getattr(cohere, 'APIError', None) if COHERE_AVAILABLE else None,
            getattr(cohere, 'APITimeoutError', None) if COHERE_AVAILABLE else None,
            # その他の例外
            pydantic_core.ValidationError, json.JSONDecodeError,
            # 一般的なタイムアウト例外
            TimeoutError, ConnectionError
        ])),
        max_tries=MAX_TRIES,
        max_time=1800,  # 最大30分でタイムアウト
        jitter=backoff.full_jitter
    )
    async def _ainvoke(self, messages: Messages, **kwargs) -> Any:
        """非同期でLLMを呼び出す統一メソッド"""
        await asyncio.sleep(self.inference_interval)
        try:
            async with self.semaphore:
                normalized_kwargs = self._normalize_generation_kwargs(kwargs)
                return await self.llm.ainvoke(messages, **normalized_kwargs)
        except openai.PermissionDeniedError as e:
            # コンテンツポリシー違反は即座に失敗させる（リトライしない）
            print(f"Content policy violation occurred: {str(e)}")
            raise  # backoffデコレータの対象外なので即座に例外が伝播される
        except pydantic_core.ValidationError as e:
            # JSONパースエラーの場合は、エラー内容をログに出力してから再スロー
            print(f"JSON parsing error occurred: {str(e)}")
            print(f"Retrying due to JSON validation error...")
            raise  # backoffデコレータがリトライを処理
        except json.JSONDecodeError as e:
            # JSONデコードエラーの場合は、エラー内容をログに出力してから再スロー
            print(f"JSON decode error occurred: {str(e)}")
            print(f"Retrying due to JSON decode error...")
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
            assert "content" in item or "tool_calls" in item, "'content' or 'tool_calls' key is missing in an item"
            # 'role'の値が'system', 'assistant', 'user', 'tool'のいずれかであることを確認
            roles = {"system", "assistant", "user", "tool"}
            assert item["role"] in roles, f"'role' should be one of {str(roles)}"
            # 'content'の値が文字列であることを確認
            if "content" in item:
                assert isinstance(item["content"], str), "'content' should be a string"
            if "tool_calls" in item:
                assert isinstance(item["tool_calls"], list), "'tool_calls' should be a list"
                for tool_call in item["tool_calls"]:
                    assert isinstance(tool_call, dict), "'tool_call' should be a dictionary"

    async def _gather_tasks(self) -> List[LLMResponse]:
        """すべてのタスクを収集して実行"""
        # 入力データの検証
        for messages, _ in self.inputs:
            self._assert_messages_format(data=messages)

        async def _safe_call(messages, kwargs):
            try:
                return await self._ainvoke(messages, **kwargs)
            except Exception as e:
                # 最終的に失敗した呼び出しは空レスポンスで継続
                print(f"Final failure in LLM call: {repr(e)}. Returning empty response and continuing.")
                return LLMResponse(content="")

        tasks = [_safe_call(messages, kwargs) for messages, kwargs in self.inputs]
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
