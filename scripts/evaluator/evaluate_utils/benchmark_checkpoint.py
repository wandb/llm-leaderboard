from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping


AGGREGATE_BENCHMARKS = frozenset({"aggregate", "aggregate_taiwan"})


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
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

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
        self._write(
            benchmark,
            "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
