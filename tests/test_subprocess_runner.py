import importlib.util
import os
import signal
import subprocess
import sys
import time
import threading
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "scripts"
    / "evaluator"
    / "evaluate_utils"
    / "subprocess_runner.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("subprocess_runner", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_run_streaming_command_terminates_child_group_on_interrupt(tmp_path, monkeypatch):
    module = load_module()
    pid_path = tmp_path / "child.pid"
    command = [
        sys.executable,
        "-c",
        (
            "import pathlib, subprocess, sys, time; "
            "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); "
            "print('ready', flush=True); time.sleep(60)"
        ),
        str(pid_path),
    ]

    original_popen = subprocess.Popen

    class InterruptingStdout:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __iter__(self):
            for line in self.wrapped:
                yield line
                raise RuntimeError("simulated parent interruption")

    def interrupting_popen(*args, **kwargs):
        proc = original_popen(*args, **kwargs)
        proc.stdout = InterruptingStdout(proc.stdout)
        return proc

    monkeypatch.setattr(module.subprocess, "Popen", interrupting_popen)
    with pytest.raises(RuntimeError, match="simulated parent interruption"):
        module.run_streaming_command(command, cwd=tmp_path)

    child_pid = int(pid_path.read_text())
    deadline = time.monotonic() + 2
    while _pid_exists(child_pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _pid_exists(child_pid)


def test_run_streaming_command_returns_output(tmp_path):
    module = load_module()
    result = module.run_streaming_command(
        [sys.executable, "-c", "print('ok')"], cwd=tmp_path
    )
    assert result.returncode == 0
    assert result.stdout == "ok\n"


def test_run_streaming_command_retains_only_bounded_tail(tmp_path):
    module = load_module()
    result = module.run_streaming_command(
        [sys.executable, "-c", "print('123456789')"],
        cwd=tmp_path,
        tail_chars=5,
    )

    assert result.stdout == "6789\n"


def test_run_streaming_command_timeout_terminates_child_group(tmp_path):
    module = load_module()
    pid_path = tmp_path / "timeout-child.pid"
    command = [
        sys.executable,
        "-c",
        (
            "import pathlib, subprocess, sys, time; "
            "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); "
            "print('ready', flush=True); time.sleep(60)"
        ),
        str(pid_path),
    ]

    with pytest.raises(TimeoutError, match="benchmark wall timeout"):
        module.run_streaming_command(command, cwd=tmp_path, timeout=0.5)

    child_pid = int(pid_path.read_text())
    deadline = time.monotonic() + 2
    while _pid_exists(child_pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _pid_exists(child_pid)


def test_nemoclaw_sandbox_lease_rejects_concurrent_owner():
    module = load_module()
    sandbox = f"pytest-{os.getpid()}-{time.time_ns()}"
    with module.NeMoClawSandboxLease(sandbox):
        with pytest.raises(
            module.NeMoClawSandboxLeaseError,
            match="Timed out waiting",
        ):
            with module.NeMoClawSandboxLease(
                sandbox,
                timeout_seconds=0,
            ):
                pass


def test_nemoclaw_sandbox_lease_waits_for_owner_then_acquires(tmp_path):
    module = load_module()
    sandbox = f"pytest-wait-{os.getpid()}-{time.time_ns()}"
    acquired = []

    def wait_for_lease():
        with module.NeMoClawSandboxLease(
            sandbox,
            timeout_seconds=1,
            poll_seconds=0.01,
            report_interval_seconds=1,
        ):
            acquired.append(True)

    with module.NeMoClawSandboxLease(sandbox):
        thread = threading.Thread(target=wait_for_lease)
        thread.start()
        time.sleep(0.05)
        assert acquired == []
    thread.join(timeout=1)

    assert acquired == [True]


def test_nemoclaw_sandbox_is_parsed_from_runner_command():
    module = load_module()
    assert (
        module.nemoclaw_sandbox_from_command(
            ["runner", "--model", "x", "--nemoclaw-sandbox", "nejumi-taiwan"]
        )
        == "nejumi-taiwan"
    )


def test_run_streaming_command_passes_sandbox_lease_to_child(tmp_path):
    module = load_module()
    sandbox = f"pytest-inherited-{os.getpid()}-{time.time_ns()}"
    result = module.run_streaming_command(
        [
            sys.executable,
            "-c",
            (
                "import os; "
                "print(os.environ.get('NEJUMI_NEMOCLAW_SANDBOX_LEASE', ''))"
            ),
            "--nemoclaw-sandbox",
            sandbox,
        ],
        cwd=tmp_path,
    )

    assert result.stdout.strip() == sandbox


def test_inherited_sandbox_lease_does_not_reacquire(monkeypatch):
    module = load_module()
    sandbox = f"pytest-inherited-context-{os.getpid()}-{time.time_ns()}"
    monkeypatch.setenv(module.NEMOCLAW_SANDBOX_LEASE_ENV, sandbox)

    with module.NeMoClawSandboxLease(sandbox):
        with module.nemoclaw_sandbox_lease(sandbox):
            pass


def test_cancellable_command_runner_stops_active_peer(tmp_path):
    module = load_module()
    runner = module.CancellableCommandRunner()
    started = tmp_path / "started"
    outcome = {}

    def invoke():
        outcome["result"] = runner.run(
            [
                sys.executable,
                "-c",
                "import pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text('yes'); time.sleep(60)",
                str(started),
            ],
            cwd=tmp_path,
            timeout=120,
        )

    thread = threading.Thread(target=invoke)
    thread.start()
    deadline = time.monotonic() + 2
    while not started.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert started.exists()

    runner.cancel_all()
    thread.join(timeout=3)

    assert not thread.is_alive()
    assert outcome["result"].returncode < 0
