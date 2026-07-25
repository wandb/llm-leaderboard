#!/usr/bin/env python3
"""Patch the system OpenClaw runtime used by a NeMoClaw sandbox Gateway.

The NeMoClaw-managed Gateway executes the system OpenClaw package in the
sandbox container, not a user-writable npm copy under /sandbox. This helper
finds the sandbox's Docker container, copies the Nejumi OpenClaw turn-budget
patcher into it, runs the patcher as root against the system package, and
optionally restarts the NeMoClaw Gateway so the patched bundle is loaded.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "scripts" / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from nemoclaw_gateway_restart import (  # noqa: E402
    NeMoClawGatewayRestartError,
    reload_nemoclaw_gateway_process,
)


DEFAULT_PATCH_SCRIPT = REPO_ROOT / "scripts" / "setup" / "patch_openclaw_turn_budget_guard.py"
DEFAULT_OPENCLAW_PACKAGE_DIR = "/usr/local/lib/node_modules/openclaw"
DEFAULT_REMOTE_PATCH_SCRIPT = "/tmp/patch_openclaw_turn_budget_guard.py"


class CommandError(RuntimeError):
    def __init__(self, message: str, result: subprocess.CompletedProcess[str] | None = None):
        super().__init__(message)
        self.result = result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sandbox", default=os.environ.get("NEMOCLAW_SANDBOX", ""))
    parser.add_argument("--container-id", default=os.environ.get("NEMOCLAW_SANDBOX_CONTAINER_ID", ""))
    parser.add_argument("--docker-bin", default=os.environ.get("DOCKER_BIN", "docker"))
    parser.add_argument("--nemoclaw-bin", default=os.environ.get("NEMOCLAW_BIN", "nemoclaw"))
    parser.add_argument("--openshell-bin", default=os.environ.get("OPENSHELL_BIN", "openshell"))
    parser.add_argument("--patch-script", type=Path, default=DEFAULT_PATCH_SCRIPT)
    parser.add_argument("--remote-patch-script", default=DEFAULT_REMOTE_PATCH_SCRIPT)
    parser.add_argument("--openclaw-package-dir", default=DEFAULT_OPENCLAW_PACKAGE_DIR)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--no-restart-gateway", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def run_command(command: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", "replace")
        return subprocess.CompletedProcess(
            command,
            124,
            stdout=stdout,
            stderr=stderr + f"\nCommand timed out after {timeout_seconds:g} seconds",
        )


def require_success(result: subprocess.CompletedProcess[str], *, action: str) -> subprocess.CompletedProcess[str]:
    if result.returncode != 0:
        raise CommandError(
            f"{action} failed with exit code {result.returncode}",
            result,
        )
    return result


def parse_docker_ps_lines(text: str) -> list[dict[str, str]]:
    containers: list[dict[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            parts = line.split(maxsplit=1)
        if len(parts) < 2:
            continue
        containers.append({"id": parts[0].strip(), "name": parts[1].strip()})
    return containers


def sandbox_container_candidates(
    containers: list[dict[str, str]],
    sandbox: str,
) -> list[dict[str, str]]:
    sandbox_token = sandbox.lower()
    exact_prefix = f"openshell-{sandbox_token}-"
    candidates: list[dict[str, str]] = []
    for container in containers:
        name = container.get("name", "").lower()
        if name.startswith(exact_prefix):
            candidates.append(container)
    if candidates:
        return candidates
    for container in containers:
        name = container.get("name", "").lower()
        if "openshell" in name and sandbox_token in name:
            candidates.append(container)
    return candidates


def find_sandbox_container(
    *,
    docker_bin: str,
    sandbox: str,
    container_id: str,
    timeout_seconds: float,
) -> tuple[str, list[dict[str, str]]]:
    if container_id:
        inspect = run_command(
            [docker_bin, "inspect", "--format", "{{.Id}} {{.Name}}", container_id],
            timeout_seconds=timeout_seconds,
        )
        require_success(inspect, action=f"inspect Docker container {container_id}")
        return container_id, parse_docker_ps_lines(inspect.stdout.replace(" /", "\t/"))

    ps = run_command(
        [docker_bin, "ps", "--format", "{{.ID}}\t{{.Names}}"],
        timeout_seconds=timeout_seconds,
    )
    require_success(ps, action="list Docker containers")
    containers = parse_docker_ps_lines(ps.stdout)
    candidates = sandbox_container_candidates(containers, sandbox)
    if len(candidates) != 1:
        names = ", ".join(f"{item['id']}:{item['name']}" for item in candidates) or "none"
        raise CommandError(
            f"Expected exactly one Docker container for NeMoClaw sandbox {sandbox!r}; found {names}. "
            "Pass --container-id to disambiguate."
        )
    return candidates[0]["id"], candidates


def load_json_output(result: subprocess.CompletedProcess[str], *, action: str) -> dict[str, Any]:
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise CommandError(f"{action} did not return valid JSON: {exc}", result) from exc
    if not isinstance(payload, dict):
        raise CommandError(f"{action} did not return a JSON object", result)
    return payload


def patch_container_openclaw(
    *,
    docker_bin: str,
    container_id: str,
    patch_script: Path,
    remote_patch_script: str,
    openclaw_package_dir: str,
    check_only: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    local_patch = patch_script.expanduser().resolve()
    if not local_patch.is_file():
        raise CommandError(f"Patch script not found: {local_patch}")

    copy_result = run_command(
        [docker_bin, "cp", str(local_patch), f"{container_id}:{remote_patch_script}"],
        timeout_seconds=timeout_seconds,
    )
    require_success(copy_result, action="copy OpenClaw turn-budget patcher into sandbox container")

    patch_command = [
        docker_bin,
        "exec",
        "-u",
        "root",
        container_id,
        "python3",
        remote_patch_script,
        "--openclaw-package-dir",
        openclaw_package_dir,
        "--json",
    ]
    if check_only:
        patch_command.insert(-1, "--check")
    patch_result = run_command(patch_command, timeout_seconds=timeout_seconds)
    require_success(patch_result, action="patch system OpenClaw runtime in sandbox container")
    payload = load_json_output(patch_result, action="patch system OpenClaw runtime in sandbox container")
    if payload.get("ok") is not True:
        raise CommandError("system OpenClaw runtime patcher reported ok=false", patch_result)
    return payload


def restart_gateway(
    *,
    openshell_bin: str,
    docker_bin: str,
    container_id: str,
    sandbox: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    try:
        return reload_nemoclaw_gateway_process(
            docker_bin=docker_bin,
            openshell_bin=openshell_bin,
            container_id=container_id,
            sandbox=sandbox,
            timeout_seconds=timeout_seconds,
        )
    except NeMoClawGatewayRestartError as exc:
        raise CommandError(
            f"restart NeMoClaw Gateway for sandbox {sandbox} failed: {exc}; "
            f"evidence={json.dumps(exc.evidence, ensure_ascii=False)}"
        ) from exc


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    payload: dict[str, Any] = {
        "ok": False,
        "sandbox": args.sandbox,
        "container_id": args.container_id or None,
        "openclaw_package_dir": args.openclaw_package_dir,
        "check_only": args.check_only,
        "restart_gateway": not args.no_restart_gateway and not args.check_only,
    }

    try:
        if not args.sandbox:
            raise CommandError("--sandbox or NEMOCLAW_SANDBOX is required")
        container_id, candidates = find_sandbox_container(
            docker_bin=args.docker_bin,
            sandbox=args.sandbox,
            container_id=args.container_id,
            timeout_seconds=args.timeout_seconds,
        )
        payload["container_id"] = container_id
        payload["container_candidates"] = candidates
        payload["patch"] = patch_container_openclaw(
            docker_bin=args.docker_bin,
            container_id=container_id,
            patch_script=args.patch_script,
            remote_patch_script=args.remote_patch_script,
            openclaw_package_dir=args.openclaw_package_dir,
            check_only=args.check_only,
            timeout_seconds=args.timeout_seconds,
        )
        if not args.no_restart_gateway and not args.check_only:
            payload["gateway_restart"] = restart_gateway(
                openshell_bin=args.openshell_bin,
                docker_bin=args.docker_bin,
                container_id=container_id,
                sandbox=args.sandbox,
                timeout_seconds=args.timeout_seconds,
            )
        payload["ok"] = True
    except CommandError as exc:
        errors.append(str(exc))
        if exc.result is not None:
            payload["failed_command"] = exc.result.args
            payload["failed_returncode"] = exc.result.returncode
            payload["failed_stdout"] = exc.result.stdout
            payload["failed_stderr"] = exc.result.stderr
    payload["errors"] = errors

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        if payload["ok"]:
            action = "verified" if args.check_only else "patched"
            print(
                f"NeMoClaw sandbox {args.sandbox} system OpenClaw runtime {action}: "
                f"{payload['container_id']}"
            )
        else:
            for error in errors:
                print(error, file=sys.stderr)
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
