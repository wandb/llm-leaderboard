from __future__ import annotations

import os
import fcntl
import json
import signal
import subprocess
import tempfile
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Mapping, Sequence


NEMOCLAW_SANDBOX_LEASE_ENV = "NEJUMI_NEMOCLAW_SANDBOX_LEASE"


class NeMoClawSandboxLeaseError(RuntimeError):
    pass


class NeMoClawSandboxLease:
    """Exclusive process-lifetime lease for a mutable NeMoClaw sandbox."""

    def __init__(self, sandbox: str) -> None:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in sandbox)
        self.path = Path(tempfile.gettempdir()) / f"nejumi-nemoclaw-{safe}.lock"
        self.sandbox = sandbox
        self._handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "unknown owner"
            handle.close()
            raise NeMoClawSandboxLeaseError(
                f"NeMoClaw sandbox {self.sandbox!r} is already in use: {owner}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "sandbox": self.sandbox,
                    "acquired_at_unix": time.time(),
                },
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        self._handle = handle
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        if self._handle is None:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


def nemoclaw_sandbox_from_command(command: Sequence[str]) -> str | None:
    try:
        index = list(command).index("--nemoclaw-sandbox")
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    sandbox = str(command[index + 1]).strip()
    return sandbox or None


def nemoclaw_sandbox_lease(sandbox: str | None):
    """Acquire a sandbox lease unless an owning parent passed it to this process."""
    if not sandbox or os.environ.get(NEMOCLAW_SANDBOX_LEASE_ENV) == sandbox:
        return nullcontext()
    return NeMoClawSandboxLease(sandbox)


def terminate_process_group(
    proc: subprocess.Popen[str],
    *,
    grace_seconds: float = 10.0,
) -> None:
    """Stop a child session and reap it, escalating when graceful shutdown stalls."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=grace_seconds)


class CancellableCommandRunner:
    """Track concurrent child sessions so one fatal worker can stop its peers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: set[subprocess.Popen[str]] = set()
        self._cancelled = threading.Event()

    def reset(self) -> None:
        with self._lock:
            if self._active:
                raise RuntimeError("cannot reset command runner while child processes are active")
            self._cancelled.clear()

    def cancel_all(self) -> None:
        self._cancelled.set()
        with self._lock:
            active = list(self._active)
        for proc in active:
            terminate_process_group(proc, grace_seconds=2.0)

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        if self._cancelled.is_set():
            raise RuntimeError("parallel command execution was cancelled after a peer failure")
        normalized = [str(part) for part in command]
        proc = subprocess.Popen(
            normalized,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        with self._lock:
            if self._cancelled.is_set():
                terminate_process_group(proc, grace_seconds=2.0)
                raise RuntimeError("parallel command execution was cancelled after a peer failure")
            self._active.add(proc)
        try:
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
                returncode = proc.returncode
            except subprocess.TimeoutExpired as exc:
                terminate_process_group(proc, grace_seconds=2.0)
                stdout, stderr = proc.communicate()
                stdout = stdout or (
                    exc.stdout.decode("utf-8", errors="replace")
                    if isinstance(exc.stdout, bytes)
                    else exc.stdout or ""
                )
                stderr = stderr or (
                    exc.stderr.decode("utf-8", errors="replace")
                    if isinstance(exc.stderr, bytes)
                    else exc.stderr or ""
                )
                returncode = 124
                stderr += f"\nCommand timed out after {timeout} seconds"
            return subprocess.CompletedProcess(normalized, returncode, stdout=stdout, stderr=stderr)
        finally:
            with self._lock:
                self._active.discard(proc)


def run_streaming_command(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    tail_chars: int = 4000,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command in its own session while streaming and retaining stdout."""
    normalized = [str(part) for part in command]
    print("Running:", " ".join(normalized), flush=True)
    sandbox = nemoclaw_sandbox_from_command(normalized)
    lease = nemoclaw_sandbox_lease(sandbox)
    with lease:
        child_env = os.environ.copy() if env is None else dict(env)
        if sandbox:
            child_env[NEMOCLAW_SANDBOX_LEASE_ENV] = sandbox
        proc = subprocess.Popen(
            normalized,
            cwd=str(cwd),
            env=child_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            start_new_session=True,
        )
        stdout_parts: list[str] = []
        timed_out = threading.Event()

        def stop_on_timeout() -> None:
            timed_out.set()
            terminate_process_group(proc, grace_seconds=2.0)

        timeout_timer = (
            threading.Timer(float(timeout), stop_on_timeout)
            if timeout is not None
            else None
        )
        if timeout_timer is not None:
            if timeout <= 0:
                terminate_process_group(proc, grace_seconds=2.0)
                raise ValueError("streaming command timeout must be positive")
            timeout_timer.daemon = True
            timeout_timer.start()
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                stdout_parts.append(line)
            returncode = proc.wait()
        except BaseException:
            terminate_process_group(proc)
            raise
        finally:
            if timeout_timer is not None:
                timeout_timer.cancel()

    stdout = "".join(stdout_parts)
    if timed_out.is_set():
        raise TimeoutError(
            f"Command exceeded benchmark wall timeout of {timeout} seconds: "
            f"{normalized}\n{stdout[-tail_chars:]}"
        )
    if returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {returncode}: {normalized}\n"
            f"{stdout[-tail_chars:]}"
        )
    return subprocess.CompletedProcess(normalized, returncode, stdout=stdout, stderr="")
