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
from evaluator.evaluate_utils.provider_rate_limiter import (
    get_provider_request_rate_limiter,
)
from llm_inference_adapter import LLMResponse

# Cohere例外をインポート（存在する場合）
try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False


MAX_TRIES = 50  # リトライ回数を50回に削減（100回は多すぎる）
MAX_TIME = 1800  # デフォルトは従来挙動を維持する
RETRYABLE_EXCEPTIONS = tuple(filter(None, [
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
]))

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


def _select_config_value(cfg: Any, path: str, default: Any = None) -> Any:
    try:
        from omegaconf import OmegaConf

        value = OmegaConf.select(cfg, path, default=default)
        return default if value is None else value
    except Exception:
        current = cfg
        for part in path.split("."):
            if isinstance(current, dict):
                current = current.get(part, default)
            else:
                current = getattr(current, part, default)
            if current is default:
                return default
        return current


def _nonnegative_float(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return default


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
        soft_fail_on_error: Optional[bool] = None,
        backoff_max_time: Optional[float] = None,
        backoff_max_tries: Optional[int] = None,
        provider_rate_limit_enabled: Optional[bool] = None,
    ):
        instance = WandbConfigSingleton.get_instance()
        cfg = instance.config
        self.llm = llm
        self.inputs = inputs
        self.batch_size = batch_size or cfg.get("batch_size", 256)
        self.inference_interval = inference_interval or cfg.inference_interval
        self.semaphore = asyncio.Semaphore(self.batch_size)
        # デフォルトはハードフェイル（従来挙動）。設定がある場合のみ上書き可能。
        try:
            # cfg.error_handling.request_failure.mode == "soft" であればソフトフェイル
            mode = getattr(cfg, "error_handling", {}).get("request_failure", {}).get("mode", "hard")
            default_soft = (mode == "soft")
        except Exception:
            default_soft = False
        self.soft_fail_on_error = default_soft if soft_fail_on_error is None else bool(soft_fail_on_error)
        self.backoff_max_time = MAX_TIME if backoff_max_time is None else backoff_max_time
        self.backoff_max_tries = MAX_TRIES if backoff_max_tries is None else backoff_max_tries
        resolved_provider_rate_limit_enabled = (
            bool(_select_config_value(cfg, "provider_rate_limit.enabled", default=False))
            if provider_rate_limit_enabled is None
            else bool(provider_rate_limit_enabled)
        )
        provider_min_interval_sec = _nonnegative_float(
            _select_config_value(cfg, "provider_rate_limit.min_request_interval_sec", default=0.0)
        )
        provider_jitter_sec = _nonnegative_float(
            _select_config_value(cfg, "provider_rate_limit.request_jitter_sec", default=0.0)
        )
        provider_rate_limit_key = str(
            _select_config_value(
                cfg,
                "provider_rate_limit.key",
                default=f"llm:{getattr(llm, 'model', 'default')}",
            )
        )
        self.provider_request_limiter = (
            get_provider_request_rate_limiter(
                provider_rate_limit_key,
                min_interval_sec=provider_min_interval_sec,
                jitter_sec=provider_jitter_sec,
            )
            if resolved_provider_rate_limit_enabled
            and (provider_min_interval_sec > 0.0 or provider_jitter_sec > 0.0)
            else None
        )
        self._ainvoke_with_backoff = backoff.on_exception(
            backoff.expo,
            RETRYABLE_EXCEPTIONS,
            max_tries=self.backoff_max_tries,
            max_time=self.backoff_max_time,
            jitter=backoff.full_jitter,
        )(self._ainvoke_impl)

    async def _ainvoke(self, messages: Messages, **kwargs) -> Any:
        """非同期でLLMを呼び出す統一メソッド（インスタンス別backoff適用）"""
        return await self._ainvoke_with_backoff(messages, **kwargs)

    @error_handler
    async def _ainvoke_impl(self, messages: Messages, **kwargs) -> Any:
        """非同期でLLMを呼び出す統一メソッド"""
        await asyncio.sleep(self.inference_interval)
        try:
            async with self.semaphore:
                if self.provider_request_limiter is not None:
                    await self.provider_request_limiter.wait_async()
                return await self.llm.ainvoke(messages, **kwargs)
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

        if self.soft_fail_on_error:
            async def _invoke_with_catch(messages: Messages, **kwargs) -> LLMResponse:
                """各リクエスト恒久失敗時に空レスポンスで継続（ソフトフェイル）"""
                try:
                    return await self._ainvoke(messages, **kwargs)
                except Exception as e:
                    print(f"Request failed permanently: {type(e).__name__}: {str(e)}")
                    return LLMResponse(content="", reasoning_content="")

            tasks = [_invoke_with_catch(messages, **kwargs) for messages, kwargs in self.inputs]
            return await atqdm.gather(*tasks, desc=f"Processing requests")
        else:
            # 従来挙動（ハードフェイル）。例外はそのまま伝播して全体を停止。
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
