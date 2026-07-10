"""Shared provider request pacing for evaluator workers.

This limiter intentionally controls request start rate, not task/case
concurrency. BFCL multi-turn cases can issue many provider requests inside a
single case, so num_threads alone cannot prevent upstream burst-limit errors.
"""

from __future__ import annotations

import asyncio
import random
import threading
import time


class ProviderRequestRateLimiter:
    def __init__(self, min_interval_sec: float = 0.0, jitter_sec: float = 0.0) -> None:
        self.min_interval_sec = max(0.0, float(min_interval_sec or 0.0))
        self.jitter_sec = max(0.0, float(jitter_sec or 0.0))
        self._lock = threading.Lock()
        self._next_request_time = 0.0

    @property
    def enabled(self) -> bool:
        return self.min_interval_sec > 0.0 or self.jitter_sec > 0.0

    def wait(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            now = time.monotonic()
            sleep_for = max(0.0, self._next_request_time - now)
            jitter = random.uniform(0.0, self.jitter_sec) if self.jitter_sec else 0.0
            self._next_request_time = max(now, self._next_request_time) + self.min_interval_sec + jitter
        if sleep_for > 0.0:
            time.sleep(sleep_for)

    async def wait_async(self) -> None:
        if not self.enabled:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.wait)


_LIMITERS: dict[str, ProviderRequestRateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def get_provider_request_rate_limiter(
    key: str,
    min_interval_sec: float = 0.0,
    jitter_sec: float = 0.0,
) -> ProviderRequestRateLimiter:
    """Return a process-wide limiter keyed by provider/model group."""

    normalized_key = key or "default"
    with _LIMITERS_LOCK:
        limiter = _LIMITERS.get(normalized_key)
        if limiter is None:
            limiter = ProviderRequestRateLimiter(
                min_interval_sec=min_interval_sec,
                jitter_sec=jitter_sec,
            )
            _LIMITERS[normalized_key] = limiter
        else:
            limiter.min_interval_sec = max(
                limiter.min_interval_sec,
                max(0.0, float(min_interval_sec or 0.0)),
            )
            limiter.jitter_sec = max(
                limiter.jitter_sec,
                max(0.0, float(jitter_sec or 0.0)),
            )
        return limiter
