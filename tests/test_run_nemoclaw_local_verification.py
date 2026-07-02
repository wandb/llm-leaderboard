import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "setup" / "run_nemoclaw_local_verification.sh"
EXPECTED_TESTS = [
    "tests/test_run_nemoclaw_local_verification.py",
    "tests/test_install_nemoclaw_script.py",
    "tests/test_review_nemoclaw_installer.py",
    "tests/test_verify_nemoclaw_operator_docs.py",
    "tests/test_verify_nemoclaw_post_install.py",
    "tests/test_check_taiwan_nemoclaw_adoption.py",
    "tests/test_openclaw_agent_protocol.py",
    "tests/test_agentic_math.py",
    "tests/test_swebench_pro.py",
    "tests/test_taiwan_full_batch_runner.py",
    "tests/test_taiwan_production_readiness_gate.py",
]


def write_executable(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def fake_path(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_executable(
        bin_dir / "pytest",
        f"""#!/usr/bin/env sh
set -eu
printf '%s\\n' "$PYTEST_DISABLE_PLUGIN_AUTOLOAD" > {tmp_path / "pytest_env.txt"}
printf '%s\\n' "$@" > {tmp_path / "pytest_args.txt"}
exit 0
""",
    )
    write_executable(
        bin_dir / "python3",
        f"""#!/usr/bin/env sh
set -eu
printf '%s\\n' "$@" > {tmp_path / "python3_args.txt"}
exit 0
""",
    )
    return bin_dir


def run_script(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)
    env["PATH"] = str(fake_path(tmp_path)) + os.pathsep + env.get("PATH", "")
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_local_verification_defaults_to_offline_test_slice(tmp_path):
    result = run_script(tmp_path)

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "pytest_env.txt").read_text(encoding="utf-8").splitlines() == [
        "1"
    ]
    pytest_args = (tmp_path / "pytest_args.txt").read_text(encoding="utf-8").splitlines()
    assert pytest_args == ["-q", *EXPECTED_TESTS]
    assert not (tmp_path / "python3_args.txt").exists()
    assert "Offline NeMoClaw local verification passed." in result.stdout


def test_local_verification_release_gate_is_explicit_opt_in(tmp_path):
    result = run_script(tmp_path, "--include-release-gate")

    assert result.returncode == 0, result.stderr
    pytest_args = (tmp_path / "pytest_args.txt").read_text(encoding="utf-8").splitlines()
    assert pytest_args == ["-q", *EXPECTED_TESTS]
    assert (tmp_path / "python3_args.txt").read_text(encoding="utf-8").splitlines() == [
        "scripts/tools/run_taiwan_release_gate.py",
        "--quiet",
    ]
    assert "Running offline Taiwan release gate..." in result.stdout


def test_local_verification_rejects_unknown_arguments_without_pytest(tmp_path):
    result = run_script(tmp_path, "--execute")

    assert result.returncode == 2
    assert "Unknown argument: --execute" in result.stderr
    assert not (tmp_path / "pytest_args.txt").exists()
    assert not (tmp_path / "python3_args.txt").exists()


def test_local_verification_help_is_no_action(tmp_path):
    result = run_script(tmp_path, "--help")

    assert result.returncode == 0
    assert "Usage: scripts/setup/run_nemoclaw_local_verification.sh" in result.stdout
    assert not (tmp_path / "pytest_args.txt").exists()
    assert not (tmp_path / "python3_args.txt").exists()
