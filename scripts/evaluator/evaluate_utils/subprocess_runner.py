from __future__ import annotations

import os
import fcntl
import hashlib
import json
import signal
import subprocess
import tempfile
import threading
import time
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Mapping, Sequence


NEMOCLAW_SANDBOX_LEASE_ENV = "NEJUMI_NEMOCLAW_SANDBOX_LEASE"
NEMOCLAW_SANDBOX_LEASE_TIMEOUT_ENV = (
    "NEJUMI_NEMOCLAW_SANDBOX_LEASE_TIMEOUT_SECONDS"
)
DEFAULT_NEMOCLAW_SANDBOX_LEASE_TIMEOUT_SECONDS = 57_600.0


class NeMoClawSandboxLeaseError(RuntimeError):
    pass


class NeMoClawSandboxLease:
    """Exclusive process-lifetime lease for a mutable NeMoClaw sandbox."""

    def __init__(
        self,
        sandbox: str,
        *,
        timeout_seconds: float | None = None,
        poll_seconds: float = 5.0,
        report_interval_seconds: float = 60.0,
    ) -> None:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in sandbox)
        digest = hashlib.sha256(sandbox.encode("utf-8")).hexdigest()[:12]
        self.path = (
            Path(tempfile.gettempdir())
            / f"nejumi-nemoclaw-{safe[:48]}-{digest}.lock"
        )
        self.sandbox = sandbox
        configured_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else os.environ.get(NEMOCLAW_SANDBOX_LEASE_TIMEOUT_ENV)
        )
        self.timeout_seconds = float(
            configured_timeout
            if configured_timeout is not None
            else DEFAULT_NEMOCLAW_SANDBOX_LEASE_TIMEOUT_SECONDS
        )
        if self.timeout_seconds < 0:
            raise ValueError(
                "NeMoClaw sandbox lease timeout must be non-negative"
            )
        self.poll_seconds = max(0.05, float(poll_seconds))
        self.report_interval_seconds = max(
            self.poll_seconds,
            float(report_interval_seconds),
        )
        self._handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        started_at = time.monotonic()
        next_report_at = started_at
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                now = time.monotonic()
                elapsed = now - started_at
                handle.seek(0)
                owner = handle.read().strip() or "unknown owner"
                if elapsed >= self.timeout_seconds:
                    handle.close()
                    raise NeMoClawSandboxLeaseError(
                        f"Timed out waiting for NeMoClaw sandbox "
                        f"{self.sandbox!r} after {elapsed:.1f}s: {owner}"
                    ) from exc
                if now >= next_report_at:
                    print(
                        "Waiting for NeMoClaw sandbox lease: "
                        f"sandbox={self.sandbox}, elapsed={elapsed:.1f}s, "
                        f"owner={owner}",
                        flush=True,
                    )
                    next_report_at = now + self.report_interval_seconds
                time.sleep(
                    min(
                        self.poll_seconds,
                        max(0.0, self.timeout_seconds - elapsed),
                    )
                )
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "sandbox": self.sandbox,
                    "wandb_run_id": os.environ.get("WANDB_RUN_ID", ""),
                    "acquired_at_unix": time.time(),
                },
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
        self._handle = handle
        print(
            "Acquired NeMoClaw sandbox lease: "
            f"sandbox={self.sandbox}, path={self.path}",
            flush=True,
        )
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        if self._handle is None:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None
            print(
                f"Released NeMoClaw sandbox lease: sandbox={self.sandbox}",
                flush=True,
            )


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
        stdout_parts: deque[str] = deque()
        retained_stdout_chars = 0
        timed_out = threading.Event()

        def retain_stdout(line: str) -> None:
            nonlocal retained_stdout_chars
            limit = max(0, int(tail_chars))
            if limit == 0:
                return
            if len(line) >= limit:
                stdout_parts.clear()
                stdout_parts.append(line[-limit:])
                retained_stdout_chars = limit
                return
            stdout_parts.append(line)
            retained_stdout_chars += len(line)
            while stdout_parts and retained_stdout_chars > limit:
                excess = retained_stdout_chars - limit
                first = stdout_parts[0]
                if len(first) <= excess:
                    retained_stdout_chars -= len(stdout_parts.popleft())
                else:
                    stdout_parts[0] = first[excess:]
                    retained_stdout_chars -= excess

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
                retain_stdout(line)
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
