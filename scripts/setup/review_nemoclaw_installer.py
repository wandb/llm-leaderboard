#!/usr/bin/env python3
"""Review the NeMoClaw installer without executing it.

This command downloads or reads the installer bytes, computes SHA-256, and
writes review evidence for the operator. It never runs the installer, does not
query W&B, and does not launch model inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DEFAULT_INSTALLER_URL = "https://www.nvidia.com/nemoclaw.sh"
DEFAULT_INSTALL_REF = "lkg"
DEFAULT_MAX_BYTES = 128 * 1024 * 1024
SCHEMA_VERSION = 1


def utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_local(path: Path, *, max_bytes: int) -> bytes:
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"installer exceeds max bytes: {size} > {max_bytes}")
    return path.read_bytes()


def read_url(url: str, *, max_bytes: int, timeout_seconds: float) -> bytes:
    request = Request(url, headers={"User-Agent": "nejumi-taiwan-nemoclaw-review/1"})
    chunks: list[bytes] = []
    total = 0
    with urlopen(request, timeout=timeout_seconds) as response:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"installer exceeds max bytes: {total} > {max_bytes}")
            chunks.append(chunk)
    return b"".join(chunks)


def load_installer(source: str, *, max_bytes: int, timeout_seconds: float) -> bytes:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        return read_url(source, max_bytes=max_bytes, timeout_seconds=timeout_seconds)
    if parsed.scheme == "file":
        return read_local(Path(parsed.path), max_bytes=max_bytes)
    return read_local(Path(source), max_bytes=max_bytes)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("lock JSON must be an object")
    return payload


def validate_lock(
    lock: dict[str, Any],
    *,
    url: str,
    install_ref: str,
    expected_sha256: str,
) -> tuple[str, list[str]]:
    errors: list[str] = []
    if lock.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"lock schema_version must be {SCHEMA_VERSION}")
    lock_url = lock.get("installer_url")
    if lock_url != url:
        errors.append("lock installer_url does not match --url")
    lock_ref = lock.get("install_ref")
    if lock_ref != install_ref:
        errors.append("lock install_ref does not match --install-ref")
    lock_sha = str(lock.get("sha256") or "").strip().lower()
    if len(lock_sha) != 64 or any(char not in "0123456789abcdef" for char in lock_sha):
        errors.append("lock sha256 must be 64 lowercase hex characters")
    expected = str(expected_sha256 or "").strip().lower()
    if expected and lock_sha and expected != lock_sha:
        errors.append("lock sha256 does not match --expected-sha256")
    for field in (
        "will_execute_installer",
        "will_install_or_onboard",
        "will_launch_model_inference",
        "will_query_wandb",
    ):
        if lock.get(field) is not False:
            errors.append(f"lock {field} must be false")
    return lock_sha, errors


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# NeMoClaw Installer Review",
        "",
        f"- status: `{payload.get('status')}`",
        f"- ok: `{payload.get('ok')}`",
        f"- installer_url: `{payload.get('installer_url')}`",
        f"- install_ref: `{payload.get('install_ref')}`",
        f"- lock_json: `{payload.get('lock_json') or ''}`",
        f"- lock_verified: `{payload.get('lock_verified')}`",
        f"- sha256: `{payload.get('sha256') or ''}`",
        f"- size_bytes: `{payload.get('size_bytes')}`",
        f"- will_execute_installer: `{payload.get('will_execute_installer')}`",
        f"- will_install_or_onboard: `{payload.get('will_install_or_onboard')}`",
        "",
        "## Recommended Install Command",
        "",
        "```bash",
        str(payload.get("recommended_install_command") or ""),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_INSTALLER_URL, help="Installer URL or local path.")
    parser.add_argument("--install-ref", default=DEFAULT_INSTALL_REF)
    parser.add_argument("--expected-sha256", default="")
    parser.add_argument(
        "--lock-json",
        type=Path,
        help="Pinned installer provenance JSON. URL/ref/SHA must match before review passes.",
    )
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--sandbox", default="nejumi-taiwan")
    parser.add_argument("--provider", default="openai")
    parser.add_argument(
        "--gateway-port",
        default="",
        help="Optional host port for the OpenShell gateway to include in the recommended command.",
    )
    parser.add_argument("--policy-tier", default="restricted")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "ok": False,
        "status": "failed",
        "generated_at": utc_iso(),
        "installer_url": args.url,
        "install_ref": args.install_ref,
        "lock_json": str(args.lock_json) if args.lock_json else "",
        "lock_verified": False,
        "expected_sha256": args.expected_sha256,
        "sha256": "",
        "size_bytes": 0,
        "max_bytes": args.max_bytes,
        "elapsed_seconds": None,
        "will_execute_installer": False,
        "will_install_or_onboard": False,
        "will_launch_model_inference": False,
        "will_query_wandb": False,
        "recommended_install_command": "",
        "errors": [],
    }
    try:
        lock: dict[str, Any] | None = None
        expected = str(args.expected_sha256 or "").strip().lower()
        if args.lock_json:
            lock = read_json_object(args.lock_json)
            lock_sha, lock_errors = validate_lock(
                lock,
                url=args.url,
                install_ref=args.install_ref,
                expected_sha256=expected,
            )
            if lock_errors:
                payload["status"] = "lock_mismatch"
                payload["errors"] = lock_errors
                raise SystemExit
            expected = expected or lock_sha
            payload["expected_sha256"] = expected

        data = load_installer(args.url, max_bytes=args.max_bytes, timeout_seconds=args.timeout)
        digest = sha256_hex(data)
        payload["sha256"] = digest
        payload["size_bytes"] = len(data)
        if expected and expected != digest:
            payload["status"] = "sha256_mismatch"
            payload["errors"] = [f"expected {expected}, got {digest}"]
        elif lock and lock.get("size_bytes") is not None and lock.get("size_bytes") != len(data):
            payload["status"] = "lock_mismatch"
            payload["errors"] = [
                f"lock size_bytes expected {lock.get('size_bytes')}, got {len(data)}"
            ]
        else:
            payload["ok"] = True
            payload["status"] = "reviewed"
            payload["lock_verified"] = bool(lock)
        lock_arg = f"--installer-lock-json {args.lock_json} " if args.lock_json else ""
        gateway_arg = f"--gateway-port {args.gateway_port} " if args.gateway_port else ""
        payload["recommended_install_command"] = (
            "scripts/setup/install_nemoclaw.sh --install --onboard "
            f"--install-ref {args.install_ref} "
            f"--sandbox {args.sandbox} --provider {args.provider} "
            f"{gateway_arg}"
            f"--policy-tier {args.policy_tier} "
            f"{lock_arg}"
            f"--installer-sha256 {digest} "
            f"--installer-review-json {args.json} "
            "--yes-i-accept-third-party-software "
            "--json temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.json"
        )
    except SystemExit:
        pass
    except (OSError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        payload["status"] = "download_failed"
        payload["errors"] = [str(exc)]
    finally:
        payload["elapsed_seconds"] = round(time.time() - started, 3)

    write_json(args.json, payload)
    if args.markdown:
        write_markdown(args.markdown, payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
