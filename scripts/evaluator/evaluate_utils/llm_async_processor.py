import asyncio
import functools
import threading
import inspect
import time
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
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as exc:
            print(
                f"LLM request attempt failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
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
        inputs: Optional[Inputs] = None,
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
        self.inputs = list(inputs or [])
        self.batch_size = (
            int(batch_size)
            if batch_size is not None
            else int(cfg.get("batch_size", 256))
        )
        self.inference_interval = (
            float(inference_interval)
            if inference_interval is not None
            else float(cfg.inference_interval)
        )
        self.semaphore = asyncio.Semaphore(self.batch_size)
        # デフォルトはハードフェイル（従来挙動）。設定がある場合のみ上書き可能。
        try:
            # cfg.error_handling.request_failure.mode == "soft" であればソフトフェイル
            mode = getattr(cfg, "error_handling", {}).get("request_failure", {}).get("mode", "hard")
            default_soft = (mode == "soft")
        except Exception:
            default_soft = False
        self.soft_fail_on_error = default_soft if soft_fail_on_error is None else bool(soft_fail_on_error)
        configured_backoff_max_time = _select_config_value(
            cfg,
            "network.retry.max_time_sec",
            default=MAX_TIME,
        )
        configured_backoff_max_tries = _select_config_value(
            cfg,
            "network.retry.max_tries",
            default=MAX_TRIES,
        )
        self.backoff_max_time = (
            float(configured_backoff_max_time)
            if backoff_max_time is None
            else float(backoff_max_time)
        )
        self.backoff_max_tries = (
            int(configured_backoff_max_tries)
            if backoff_max_tries is None
            else int(backoff_max_tries)
        )
        if self.backoff_max_time <= 0:
            raise ValueError("backoff_max_time must be positive")
        if self.backoff_max_tries <= 0:
            raise ValueError("backoff_max_tries must be positive")
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
        self.progress_interval_sec = _nonnegative_float(
            _select_config_value(
                cfg,
                "network.progress_interval_sec",
                default=60.0,
            )
        )
        self._ainvoke_with_backoff = backoff.on_exception(
            backoff.expo,
            RETRYABLE_EXCEPTIONS,
            max_tries=self.backoff_max_tries,
            max_time=self.backoff_max_time,
            jitter=backoff.full_jitter,
        )(self._ainvoke_impl)
        self._sync_loop = None
        self._sync_loop_thread = None
        self._sync_loop_lock = threading.Lock()

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
            print("Retrying due to JSON validation error...")
            raise  # backoffデコレータがリトライを処理
        except json.JSONDecodeError as e:
            # JSONデコードエラーの場合は、エラー内容をログに出力してから再スロー
            print(f"JSON decode error occurred: {str(e)}")
            print("Retrying due to JSON decode error...")
            raise  # backoffデコレータがリトライを処理

    def _assert_messages_format(self, data: Messages):
        """メッセージフォーマットの検証"""
        # データがリストであることを確認
        assert isinstance(data, list), "Data should be a list"
        # 各要素が辞書であることを確認
        for item in data:
            # The OpenAI Responses API replays provider-native response items
            # (reasoning, function_call, function_call_output) alongside chat
            # messages. SDK response models are intentionally accepted here.
            if not isinstance(item, dict):
                assert hasattr(item, "type"), (
                    "Each item should be a chat dictionary or a provider-native "
                    "response item"
                )
                continue
            if "role" not in item and "type" in item:
                continue
            # 'role'キーと'content'キーが存在することを確認
            assert "role" in item, "'role' key is missing in an item"
            assert "content" in item or "tool_calls" in item, "'content' or 'tool_calls' key is missing in an item"
            # OpenAI Responses uses the developer role for application-level
            # instructions. The shared Anthropic adapter also normalizes it.
            roles = {"system", "developer", "assistant", "user", "tool"}
            assert item["role"] in roles, f"'role' should be one of {str(roles)}"
            # Provider-native chat histories may use structured content blocks
            # (Anthropic) or null content when tool_calls carry the assistant
            # output (OpenAI-compatible APIs).
            if "content" in item:
                content = item["content"]
                if content is None:
                    assert "tool_calls" in item, (
                        "null 'content' is only valid when 'tool_calls' is present"
                    )
                elif isinstance(content, list):
                    for block in content:
                        assert (
                            isinstance(block, dict) and "type" in block
                        ) or hasattr(block, "type"), (
                            "structured 'content' blocks must have a 'type'"
                        )
                else:
                    assert isinstance(content, str), (
                        "'content' should be a string, null tool-call payload, "
                        "or provider-native block list"
                    )
            if "tool_calls" in item:
                assert isinstance(item["tool_calls"], list), "'tool_calls' should be a list"
                for tool_call in item["tool_calls"]:
                    assert isinstance(tool_call, dict), "'tool_call' should be a dictionary"

    async def _gather_tasks(self, on_result=None) -> List[LLMResponse]:
        """すべてのタスクを収集して実行"""
        # 入力データの検証
        for messages, _ in self.inputs:
            self._assert_messages_format(data=messages)

        async def _invoke_with_catch(messages: Messages, **kwargs) -> LLMResponse:
                """各リクエスト恒久失敗時に空レスポンスで継続（ソフトフェイル）"""
                try:
                    return await self._ainvoke(messages, **kwargs)
                except Exception as e:
                    print(f"Request failed permanently: {type(e).__name__}: {str(e)}")
                    return LLMResponse(content="", reasoning_content="")

        async def _invoke_indexed(index, messages, kwargs):
            invoke = _invoke_with_catch if self.soft_fail_on_error else self._ainvoke
            return index, await invoke(messages, **kwargs)

        tasks = [
            asyncio.create_task(_invoke_indexed(index, messages, kwargs))
            for index, (messages, kwargs) in enumerate(self.inputs)
        ]
        results: list[Optional[LLMResponse]] = [None] * len(tasks)
        progress = {
            "completed": 0,
            "last_completed_at": time.monotonic(),
        }

        async def progress_watchdog() -> None:
            while True:
                await asyncio.sleep(self.progress_interval_sec)
                pending = len(tasks) - progress["completed"]
                idle = time.monotonic() - progress["last_completed_at"]
                print(
                    "LLM batch heartbeat: "
                    f"{progress['completed']}/{len(tasks)} completed, "
                    f"{pending} pending, {idle:.1f}s since last completion",
                    flush=True,
                )

        watchdog_task = (
            asyncio.create_task(progress_watchdog())
            if tasks and self.progress_interval_sec > 0
            else None
        )
        try:
            for completed in atqdm.as_completed(
                tasks,
                total=len(tasks),
                desc="Processing requests",
            ):
                index, response = await completed
                results[index] = response
                progress["completed"] += 1
                progress["last_completed_at"] = time.monotonic()
                if on_result is not None:
                    callback_result = on_result(index, response)
                    if inspect.isawaitable(callback_result):
                        await callback_result
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            if watchdog_task is not None:
                watchdog_task.cancel()
                await asyncio.gather(watchdog_task, return_exceptions=True)
        return [result for result in results if result is not None]

    def get_results(self, on_result=None) -> List[LLMResponse]:
        """結果を取得（同期的なエントリーポイント）"""
        return self._run_on_sync_loop(self._gather_tasks(on_result=on_result))

    async def get_results_async(self, on_result=None) -> List[LLMResponse]:
        """結果を取得（非同期版）"""
        return await self._gather_tasks(on_result=on_result)

    def process_single(self, messages: Messages, **kwargs) -> LLMResponse:
        """単一のメッセージを処理（同期版）"""
        return self._run_on_sync_loop(
            self.process_single_async(messages, **kwargs)
        )

    def _ensure_sync_loop(self) -> asyncio.AbstractEventLoop:
        with self._sync_loop_lock:
            if self._sync_loop is not None and self._sync_loop.is_running():
                return self._sync_loop

            ready = threading.Event()
            loop_holder = {}

            def run_loop() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop_holder["loop"] = loop
                ready.set()
                loop.run_forever()
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                loop.close()

            thread = threading.Thread(
                target=run_loop,
                name=f"llm-async-processor-{id(self)}",
                daemon=True,
            )
            thread.start()
            ready.wait()
            self._sync_loop = loop_holder["loop"]
            self._sync_loop_thread = thread
            return self._sync_loop

    def _run_on_sync_loop(self, coroutine):
        loop = self._ensure_sync_loop()
        return asyncio.run_coroutine_threadsafe(coroutine, loop).result()

    def close_sync_loop(self) -> None:
        with self._sync_loop_lock:
            loop = self._sync_loop
            thread = self._sync_loop_thread
            self._sync_loop = None
            self._sync_loop_thread = None
        if (
            loop is not None
            and loop.is_running()
            and thread is not threading.current_thread()
        ):
            future = asyncio.run_coroutine_threadsafe(
                self.close_async_client(),
                loop,
            )
            try:
                future.result(timeout=10)
            except Exception as exc:
                print(
                    "Warning: failed to close async LLM client cleanly: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)

    async def close_async_client(self) -> None:
        """Close an evaluator-owned async HTTP client before its event loop exits."""
        close_llm = getattr(self.llm, "aclose", None)
        if close_llm is not None:
            result = close_llm()
            if inspect.isawaitable(result):
                await result
            return
        async_client = getattr(self.llm, "async_client", None)
        close = getattr(async_client, "close", None)
        if close is None:
            close = getattr(async_client, "aclose", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result

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
        return await self._gather_tasks(on_result=callback_func)

    def get_statistics(self) -> dict:
        """統計情報を取得"""
        return {
            "total_inputs": len(self.inputs),
            "batch_size": self.batch_size,
            "inference_interval": self.inference_interval,
            "estimated_batches": (len(self.inputs) + self.batch_size - 1) // self.batch_size
        }
