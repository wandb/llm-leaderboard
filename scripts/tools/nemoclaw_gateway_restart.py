"""Restart a NeMoClaw Gateway without retaining a foreground port forward."""

from __future__ import annotations

import os
import re
import selectors
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any


FORWARD_PORT_RE = re.compile(r"Forwarding port\s+(\d+)\s+to sandbox", re.IGNORECASE)
STALE_LOCK_RE = re.compile(
    r"shields transition lock\s+'(?P<path>[^']+)'.*?recorded owner PID\s+"
    r"(?P<pid>\d+)\s+is not running",
    re.IGNORECASE | re.DOTALL,
)


class NeMoClawGatewayRestartError(RuntimeError):
    def __init__(self, message: str, evidence: dict[str, Any]):
        super().__init__(message)
        self.evidence = evidence


def parse_forward_port(output: str) -> int | None:
    match = FORWARD_PORT_RE.search(output or "")
    return int(match.group(1)) if match else None


def remove_confirmed_stale_lock(output: str) -> str | None:
    match = STALE_LOCK_RE.search(output or "")
    if match is None:
        return None
    pid = int(match.group("pid"))
    try:
        os.kill(pid, 0)
        return None
    except PermissionError:
        return None
    except ProcessLookupError:
        pass
    state_dir = (Path.home() / ".nemoclaw" / "state").resolve(strict=False)
    lock_path = Path(match.group("path")).expanduser().resolve(strict=False)
    if lock_path.parent != state_dir or not lock_path.name.startswith(
        "shields-transition-lock-"
    ):
        return None
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass
    return str(lock_path)


def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _port_ready(port: int, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.25)
    return False


def _run(command: list[str], timeout_seconds: float) -> subprocess.CompletedProcess[str]:
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
        return subprocess.CompletedProcess(command, 124, stdout=stdout, stderr=stderr)


def _start_background_forward(command: list[str]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def recover_background_forward(
    *,
    openshell_bin: str,
    sandbox: str,
    port: int,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    stop = _run(
        [openshell_bin, "forward", "stop", str(port), sandbox],
        timeout_seconds,
    )
    start_command = [
        openshell_bin,
        "forward",
        "start",
        str(port),
        sandbox,
        "--background",
    ]
    start = _start_background_forward(start_command)
    ready = _port_ready(port, min(timeout_seconds, 30.0))
    evidence = {
        "port": port,
        "stop_returncode": stop.returncode,
        "stop_stdout": stop.stdout,
        "stop_stderr": stop.stderr,
        "start_command": start_command,
        "start_pid": start.pid,
        "start_returncode": start.poll(),
        "ready": ready,
    }
    if not ready:
        _terminate_process_group(start)
        raise NeMoClawGatewayRestartError(
            f"Failed to recover background Gateway forward on port {port}",
            evidence,
        )
    return evidence


def _sandbox_container_id(docker_bin: str, sandbox: str) -> str:
    result = _run(
        [docker_bin, "ps", "--format", "{{.ID}}\t{{.Names}}"],
        30.0,
    )
    prefix = f"openshell-{sandbox}-"
    candidates = [
        line.split("\t", 1)[0]
        for line in result.stdout.splitlines()
        if "\t" in line and line.split("\t", 1)[1].startswith(prefix)
    ]
    if result.returncode != 0 or len(candidates) != 1:
        raise NeMoClawGatewayRestartError(
            f"Expected one running sandbox container for {sandbox}; found {len(candidates)}",
            {
                "command_returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "candidates": candidates,
            },
        )
    return candidates[0]


def reload_nemoclaw_gateway_process(
    *,
    sandbox: str,
    docker_bin: str = "docker",
    openshell_bin: str = "openshell",
    container_id: str | None = None,
    gateway_port: int = 18790,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Reload Gateway config without a lifecycle restart that restores package files."""
    container_id = container_id or _sandbox_container_id(docker_bin, sandbox)
    script = r"""
set -euo pipefail
old_pid="$(pgrep -o -x openclaw)"
kill -TERM "$old_pid"
deadline=$((SECONDS + $1))
while (( SECONDS < deadline )); do
  new_pid="$(pgrep -o -x openclaw 2>/dev/null || true)"
  if [[ -n "$new_pid" && "$new_pid" != "$old_pid" ]]; then
    if nsenter -t "$new_pid" -n bash -lc "echo >/dev/tcp/127.0.0.1/$2" >/dev/null 2>&1; then
      printf '{"old_pid":%s,"new_pid":%s,"port":%s}\n' "$old_pid" "$new_pid" "$2"
      exit 0
    fi
  fi
  sleep 1
done
echo "Gateway process did not respawn healthy before timeout" >&2
exit 1
"""
    command = [
        docker_bin,
        "exec",
        "-u",
        "root",
        container_id,
        "bash",
        "-lc",
        script,
        "reload-gateway",
        str(max(1, int(timeout_seconds))),
        str(gateway_port),
    ]
    result = _run(command, timeout_seconds + 10)
    evidence = {
        "container_id": container_id,
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "gateway_port": gateway_port,
    }
    if result.returncode != 0:
        raise NeMoClawGatewayRestartError(
            "NeMoClaw Gateway process reload failed",
            evidence,
        )
    if not _port_ready(gateway_port, 10.0):
        evidence["background_forward"] = recover_background_forward(
            openshell_bin=openshell_bin,
            sandbox=sandbox,
            port=gateway_port,
        )
    evidence["ready"] = _port_ready(gateway_port, 5.0)
    if not evidence["ready"]:
        raise NeMoClawGatewayRestartError(
            "NeMoClaw Gateway reloaded but its host forward is unavailable",
            evidence,
        )
    return {"ok": True, "mode": "supervisor_process_reload", **evidence}


def restart_nemoclaw_gateway(
    *,
    nemoclaw_bin: str,
    sandbox: str,
    openshell_bin: str = "openshell",
    timeout_seconds: float = 120.0,
    _allow_stale_lock_retry: bool = True,
) -> dict[str, Any]:
    """Restart the Gateway and detach NeMoClaw's foreground forward if needed."""
    command = [nemoclaw_bin, "sandbox", "gateway", "restart", sandbox, "--quiet"]
    proc = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        start_new_session=True,
    )
    output_parts: list[str] = []
    deadline = time.monotonic() + timeout_seconds
    selector = selectors.DefaultSelector()
    assert proc.stdout is not None
    selector.register(proc.stdout, selectors.EVENT_READ)
    foreground_port: int | None = None
    try:
        while proc.poll() is None and time.monotonic() < deadline:
            for key, _ in selector.select(timeout=0.5):
                line = key.fileobj.readline()
                if not line:
                    continue
                output_parts.append(line)
                foreground_port = parse_forward_port("".join(output_parts))
                if foreground_port is not None:
                    break
            if foreground_port is not None:
                break
        if foreground_port is not None:
            _terminate_process_group(proc)
            recovery = recover_background_forward(
                openshell_bin=openshell_bin,
                sandbox=sandbox,
                port=foreground_port,
            )
            return {
                "ok": True,
                "mode": "foreground_forward_recovered",
                "command": command,
                "output": "".join(output_parts),
                "background_forward": recovery,
            }
        if proc.poll() is None:
            _terminate_process_group(proc)
            raise NeMoClawGatewayRestartError(
                f"NeMoClaw Gateway restart timed out after {timeout_seconds:g}s",
                {"command": command, "output": "".join(output_parts)},
            )
        remainder = proc.stdout.read()
        if remainder:
            output_parts.append(remainder)
        if proc.returncode != 0:
            output = "".join(output_parts)
            removed_lock = remove_confirmed_stale_lock(output)
            if removed_lock and _allow_stale_lock_retry:
                retry = restart_nemoclaw_gateway(
                    nemoclaw_bin=nemoclaw_bin,
                    sandbox=sandbox,
                    openshell_bin=openshell_bin,
                    timeout_seconds=timeout_seconds,
                    _allow_stale_lock_retry=False,
                )
                retry["removed_stale_lock"] = removed_lock
                return retry
            if (
                "gateway process restarted but health did not pass before timeout"
                in output.lower()
            ):
                status_command = [nemoclaw_bin, "sandbox", "status", sandbox]
                status_result = subprocess.run(
                    status_command,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=30,
                )
                status_output = (status_result.stdout or "") + (status_result.stderr or "")
                healthy = (
                    status_result.returncode == 0
                    and "OpenClaw: running" in status_output
                    and "Docker health: healthy" in status_output
                )
                if healthy:
                    return {
                        "ok": True,
                        "mode": "health_timeout_reconciled",
                        "command": command,
                        "returncode": proc.returncode,
                        "output": output,
                        "status_command": status_command,
                        "status_returncode": status_result.returncode,
                        "status_output": status_output,
                    }
            raise NeMoClawGatewayRestartError(
                f"NeMoClaw Gateway restart failed with exit code {proc.returncode}",
                {
                    "command": command,
                    "returncode": proc.returncode,
                    "output": output,
                },
            )
        return {
            "ok": True,
            "mode": "command_completed",
            "command": command,
            "returncode": proc.returncode,
            "output": "".join(output_parts),
        }
    finally:
        selector.close()
        _terminate_process_group(proc)
