import importlib.util
import os
import signal
import sys
import threading
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script_module(path: Path):
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def assert_parent_sigint_aborts_child(command_runner):
    timer = threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGINT))
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            command_runner(
                [
                    sys.executable,
                    "-c",
                    "import time; time.sleep(30); print('child-should-not-finish')",
                ]
            )
    finally:
        timer.cancel()


def test_agentic_math_runner_stops_child_on_parent_sigint():
    module = load_script_module(REPO_ROOT / "scripts" / "evaluator" / "agentic_math.py")
    assert_parent_sigint_aborts_child(module._run_command)


def test_swebench_pro_runner_stops_child_on_parent_sigint():
    module = load_script_module(REPO_ROOT / "scripts" / "evaluator" / "swebench_pro.py")
    assert_parent_sigint_aborts_child(module._run_command)
