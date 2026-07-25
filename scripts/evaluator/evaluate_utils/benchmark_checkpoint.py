from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping


class BenchmarkCheckpointStore:
    def __init__(
        self,
        root: Path,
        *,
        run_id: str,
        resume_enabled: bool,
        config_fingerprints: Mapping[str, str] | None = None,
    ) -> None:
        self.root = root
        self.run_id = str(run_id)
        self.resume_enabled = bool(resume_enabled)
        self.config_fingerprints = dict(config_fingerprints or {})

    def _path(self, benchmark: str) -> Path:
        return self.root / f"{benchmark}.json"

    def _write(self, benchmark: str, status: str, **extra: Any) -> None:
        path = self._path(benchmark)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "run_id": self.run_id,
            "benchmark": benchmark,
            "status": status,
            "updated_at": time.time(),
            **extra,
        }
        config_fingerprint = self.config_fingerprints.get(benchmark)
        if config_fingerprint is not None:
            payload["config_fingerprint"] = config_fingerprint
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
