#!/usr/bin/env python3
"""Configure OpenClaw gateway service to read provider secrets from .env.

This script writes a systemd user drop-in that references an existing env file
with EnvironmentFile=. It never copies or prints secret values.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
DEFAULT_OPENCLAW_CONFIG = Path.home() / ".openclaw" / "openclaw.json"
DEFAULT_SERVICE_NAME = "openclaw-gateway.service"
DEFAULT_DROPIN_NAME = "20-nejumi-env.conf"
VALID_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class EnvFileInfo:
    path: Path
    exists: bool
    keys_present: set[str]
    invalid_lines: list[int]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--openclaw-config", type=Path, default=DEFAULT_OPENCLAW_CONFIG)
    parser.add_argument("--service-name", default=DEFAULT_SERVICE_NAME)
    parser.add_argument("--dropin-name", default=DEFAULT_DROPIN_NAME)
    parser.add_argument(
        "--systemd-user-dir",
        type=Path,
        default=Path.home() / ".config" / "systemd" / "user",
    )
    parser.add_argument(
        "--required-key",
        action="append",
        default=[],
        help="Additional required environment key. Can be repeated.",
    )
    parser.add_argument("--write", action="store_true", help="Write the systemd user drop-in.")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Run systemctl --user daemon-reload and restart the OpenClaw gateway service after writing.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only inspect current state. This is the default when --write is not set.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    return parser.parse_args(argv)


def parse_env_file(path: Path) -> EnvFileInfo:
    expanded = path.expanduser().resolve()
    if not expanded.exists():
        return EnvFileInfo(path=expanded, exists=False, keys_present=set(), invalid_lines=[])
    keys: set[str] = set()
    invalid_lines: list[int] = []
    for line_no, raw in enumerate(expanded.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            invalid_lines.append(line_no)
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not VALID_ENV_KEY.match(key):
            invalid_lines.append(line_no)
            continue
        if value.strip().strip('"').strip("'"):
            keys.add(key)
    return EnvFileInfo(path=expanded, exists=True, keys_present=keys, invalid_lines=invalid_lines)


def find_env_secret_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, dict):
        if value.get("source") == "env" and isinstance(value.get("id"), str):
            refs.add(value["id"])
        for nested in value.values():
            refs.update(find_env_secret_refs(nested))
    elif isinstance(value, list):
        for nested in value:
            refs.update(find_env_secret_refs(nested))
    return refs


def load_openclaw_secret_refs(path: Path) -> tuple[set[str], str | None]:
    expanded = path.expanduser().resolve()
    if not expanded.exists():
        return set(), "missing"
    try:
        payload = json.loads(expanded.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return set(), f"invalid_json:{exc.lineno}"
    return find_env_secret_refs(payload), None


def dropin_path(systemd_user_dir: Path, service_name: str, dropin_name: str) -> Path:
    return systemd_user_dir.expanduser() / f"{service_name}.d" / dropin_name


def render_dropin(env_file: Path) -> str:
    return (
        "[Service]\n"
        f"EnvironmentFile={env_file}\n"
        "\n"
    )


def existing_dropin_env_files(path: Path) -> list[str]:
    if not path.exists():
        return []
    env_files: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("EnvironmentFile="):
            continue
        env_files.append(line.split("=", 1)[1].strip())
    return env_files


def run_systemctl(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["systemctl", "--user", *args],
        text=True,
        capture_output=True,
        check=False,
    )


def write_dropin(path: Path, env_file: Path) -> bool:
    content = render_dropin(env_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def build_status(args: argparse.Namespace, *, wrote: bool = False) -> dict[str, Any]:
    env_info = parse_env_file(args.env_file)
    openclaw_secret_refs, config_error = load_openclaw_secret_refs(args.openclaw_config)
    required = sorted(openclaw_secret_refs | set(args.required_key or []))
    missing = sorted(key for key in required if key not in env_info.keys_present)
    target_dropin = dropin_path(args.systemd_user_dir, args.service_name, args.dropin_name)
    configured_env_files = existing_dropin_env_files(target_dropin)
    expected_env_file = str(env_info.path)
    dropin_matches = expected_env_file in configured_env_files
    return {
        "ok": (
            env_info.exists
            and not env_info.invalid_lines
            and not missing
            and target_dropin.exists()
            and dropin_matches
        ),
        "env_file": str(env_info.path),
        "env_file_exists": env_info.exists,
        "env_file_invalid_lines": env_info.invalid_lines,
        "openclaw_config": str(args.openclaw_config.expanduser().resolve()),
        "openclaw_config_error": config_error,
        "secret_refs_required": required,
        "secret_refs_present_count": len(required) - len(missing),
        "secret_refs_missing": missing,
        "service_name": args.service_name,
        "dropin_path": str(target_dropin),
        "dropin_exists": target_dropin.exists(),
        "dropin_env_files": configured_env_files,
        "dropin_matches_env_file": dropin_matches,
        "wrote_dropin": wrote,
    }


def print_human(payload: dict[str, Any]) -> None:
    status = "OK" if payload["ok"] else "NOT READY"
    print(f"OpenClaw gateway env drop-in: {status}")
    print(f"env_file: {payload['env_file']} exists={payload['env_file_exists']}")
    print(f"dropin: {payload['dropin_path']} exists={payload['dropin_exists']}")
    print(f"dropin_matches_env_file: {payload['dropin_matches_env_file']}")
    if payload["secret_refs_required"]:
        print(
            "secret refs: "
            f"{payload['secret_refs_present_count']}/{len(payload['secret_refs_required'])} present"
        )
    if payload["secret_refs_missing"]:
        print("missing secret refs: " + ", ".join(payload["secret_refs_missing"]))
    if payload["env_file_invalid_lines"]:
        print(
            "invalid env file lines: "
            + ", ".join(str(line) for line in payload["env_file_invalid_lines"])
        )
    if payload["openclaw_config_error"]:
        print(f"openclaw config error: {payload['openclaw_config_error']}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    wrote = False
    if args.write:
        env_info = parse_env_file(args.env_file)
        wrote = write_dropin(
            dropin_path(args.systemd_user_dir, args.service_name, args.dropin_name),
            env_info.path,
        )

    daemon_reload: dict[str, Any] | None = None
    restart: dict[str, Any] | None = None
    if args.restart:
        if not args.write:
            raise SystemExit("--restart requires --write")
        reload_result = run_systemctl(["daemon-reload"])
        daemon_reload = {
            "returncode": reload_result.returncode,
            "stderr_tail": reload_result.stderr[-1000:],
        }
        if reload_result.returncode == 0:
            restart_result = run_systemctl(["restart", args.service_name])
            restart = {
                "returncode": restart_result.returncode,
                "stderr_tail": restart_result.stderr[-1000:],
            }

    payload = build_status(args, wrote=wrote)
    if daemon_reload is not None:
        payload["daemon_reload"] = daemon_reload
    if restart is not None:
        payload["restart"] = restart

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_human(payload)
        if daemon_reload is not None:
            print(f"systemctl daemon-reload returncode={daemon_reload['returncode']}")
        if restart is not None:
            print(f"systemctl restart returncode={restart['returncode']}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
