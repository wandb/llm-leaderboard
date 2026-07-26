from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Mapping


AGGREGATE_BENCHMARKS = frozenset({"aggregate", "aggregate_taiwan"})
INFRASTRUCTURE_FAILURE_TYPES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "BFCLInfrastructureError",
        "ConnectError",
        "ConnectTimeout",
        "ConnectionError",
        "InternalServerError",
        "NeMoClawGatewayRestartError",
        "NeMoClawSandboxLeaseError",
        "NonScoreableOpenClawProviderTimeout",
        "ProviderRecoveryExhaustedError",
        "RateLimitError",
        "ReadError",
        "ReadTimeout",
    }
)
INFRASTRUCTURE_FAILURE_MARKERS = (
    "BFCLInfrastructureError",
    "NeMoClawGatewayRestartError",
    "NeMoClawSandboxLeaseError",
    "NonScoreableOpenClawProviderTimeout",
    "ProviderRecoveryExhaustedError",
)
NONRETRYABLE_PROVIDER_MARKERS = (
    "authentication",
    "billing",
    "credit balance",
    "insufficient balance",
    "insufficient_quota",
    "invalid api key",
    "permission denied",
)
MODEL_LIMIT_FAILURE_MARKERS = (
    "budget exceeded",
    "case timeout",
    "maximum agent turns",
    "maximum tool calls",
    "model_inference_error",
    "time limit",
    "token budget",
)


def classify_benchmark_failure(
    error_type: str | None,
    error: str | None,
) -> dict[str, Any]:
    """Classify a failed benchmark conservatively for process-level recovery."""
    normalized_type = str(error_type or "").strip()
    message = str(error or "")
    message_lower = message.lower()

    if normalized_type in {"KeyboardInterrupt", "SystemExit"}:
        category = "operator_interrupt"
        reason = "operator or process interruption is never retried automatically"
    elif any(marker in message_lower for marker in NONRETRYABLE_PROVIDER_MARKERS):
        category = "configuration_or_billing"
        reason = "provider credentials, permissions, quota, or billing require operator action"
    elif normalized_type in {"TimeoutError", "BudgetExceededError"}:
        category = "model_or_task_limit"
        reason = "model/task limits are scored outcomes, not infrastructure recovery"
    elif normalized_type in INFRASTRUCTURE_FAILURE_TYPES or any(
        marker in message for marker in INFRASTRUCTURE_FAILURE_MARKERS
    ):
        category = "infrastructure"
        reason = "known transient provider or gateway infrastructure failure"
    elif any(marker in message_lower for marker in MODEL_LIMIT_FAILURE_MARKERS):
        category = "model_or_task_limit"
        reason = "model/task limits are scored outcomes, not infrastructure recovery"
    elif normalized_type in {
        "AssertionError",
        "FileNotFoundError",
        "ImportError",
        "KeyError",
        "ModuleNotFoundError",
        "TypeError",
        "ValueError",
    }:
        category = "code_or_configuration"
        reason = "deterministic code or configuration failures require repair"
    else:
        category = "unknown"
        reason = "unclassified failures require operator review"

    return {
        "category": category,
        "infrastructure_retryable": category == "infrastructure",
        "reason": reason,
    }


def bfcl_completion_evidence_is_clean(summary: Mapping[str, Any]) -> bool:
    try:
        return int(summary.get("bfcl_inference_error_count")) == 0
    except (TypeError, ValueError):
        return False


def aggregate_requires_refresh(
    benchmark: str,
    executed_benchmarks: set[str],
) -> bool:
    return bool(
        benchmark in AGGREGATE_BENCHMARKS
        and any(
            name not in AGGREGATE_BENCHMARKS
            for name in executed_benchmarks
        )
    )


class BenchmarkCheckpointStore:
    def __init__(
        self,
        root: Path,
        *,
        run_id: str,
        resume_enabled: bool,
        config_fingerprints: Mapping[str, str] | None = None,
        code_fingerprints: Mapping[str, str] | None = None,
    ) -> None:
        self.root = root
        self.run_id = str(run_id)
        self.resume_enabled = bool(resume_enabled)
        self.config_fingerprints = dict(config_fingerprints or {})
        self.code_fingerprints = dict(code_fingerprints or {})

    def _path(self, benchmark: str) -> Path:
        return self.root / f"{benchmark}.json"

    def _write(self, benchmark: str, status: str, **extra: Any) -> None:
        path = self._path(benchmark)
        path.parent.mkdir(parents=True, exist_ok=True)
        previous = self.load_raw(benchmark)
        last_completed = None
        if isinstance(previous, dict):
            if previous.get("status") == "completed":
                last_completed = {
                    key: previous.get(key)
                    for key in (
                        "updated_at",
                        "config_fingerprint",
                        "code_fingerprint",
                    )
                    if previous.get(key) is not None
                }
            elif isinstance(previous.get("last_completed"), dict):
                last_completed = dict(previous["last_completed"])
        payload = {
            "schema_version": 2,
            "run_id": self.run_id,
            "benchmark": benchmark,
            "status": status,
            "updated_at": time.time(),
            **extra,
        }
        config_fingerprint = self.config_fingerprints.get(benchmark)
        if config_fingerprint is not None:
            payload["config_fingerprint"] = config_fingerprint
        code_fingerprint = self.code_fingerprints.get(benchmark)
        if code_fingerprint is not None:
            payload["code_fingerprint"] = code_fingerprint
        if status != "completed" and last_completed:
            payload["last_completed"] = last_completed
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def load_raw(self, benchmark: str) -> dict[str, Any] | None:
        path = self._path(benchmark)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            not isinstance(payload, dict)
            or payload.get("run_id") != self.run_id
            or payload.get("benchmark") != benchmark
        ):
            return None
        return payload

    def load(self, benchmark: str) -> dict[str, Any] | None:
        payload = self.load_raw(benchmark)
        if payload is None:
            return None
        expected_fingerprint = self.config_fingerprints.get(benchmark)
        if (
            expected_fingerprint is not None
            and payload.get("config_fingerprint") != expected_fingerprint
        ):
            return None
        return payload

    def is_completed(self, benchmark: str) -> bool:
        if not self.resume_enabled:
            return False
        payload = self.load(benchmark)
        return bool(payload and payload.get("status") == "completed")

    def has_completed_snapshot(self, benchmark: str) -> bool:
        payload = self.load_raw(benchmark)
        return bool(
            payload
            and (
                payload.get("status") == "completed"
                or isinstance(payload.get("last_completed"), dict)
            )
        )

    def completed_snapshot_matches_config(self, benchmark: str) -> bool:
        payload = self.load_raw(benchmark)
        if payload is None:
            return False
        snapshot = (
            payload
            if payload.get("status") == "completed"
            else payload.get("last_completed")
        )
        if not isinstance(snapshot, dict):
            return False
        expected = self.config_fingerprints.get(benchmark)
        return expected is None or snapshot.get("config_fingerprint") == expected

    def can_restore_completed_snapshot(
        self,
        benchmark: str,
        *,
        remote_completed: bool,
    ) -> bool:
        payload = self.load_raw(benchmark)
        return bool(
            self.resume_enabled
            and remote_completed
            and payload
            and payload.get("status") != "completed"
            and self.completed_snapshot_matches_config(benchmark)
        )

    def code_drifted(self, benchmark: str) -> bool:
        payload = self.load(benchmark)
        if payload is None:
            return False
        expected = self.code_fingerprints.get(benchmark)
        actual = payload.get("code_fingerprint")
        return bool(expected and actual and expected != actual)

    def mark_started(self, benchmark: str) -> None:
        self._write(benchmark, "started")

    def mark_completed(self, benchmark: str) -> None:
        self._write(benchmark, "completed")

    def mark_failed(self, benchmark: str, error: BaseException) -> None:
        failure = classify_benchmark_failure(type(error).__name__, str(error))
        self._write(
            benchmark,
            "failed",
            error_type=type(error).__name__,
            error=str(error),
            failure_category=failure["category"],
            infrastructure_retryable=failure["infrastructure_retryable"],
            failure_reason=failure["reason"],
        )
