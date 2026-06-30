import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "setup" / "verify_nemoclaw_operator_docs.py"
README = REPO_ROOT / "docs" / "README_nemoclaw.md"
LOCK_JSON = REPO_ROOT / "scripts" / "setup" / "nemoclaw_installer_lock.json"


def run_verifier(
    tmp_path: Path,
    *,
    readme: Path = README,
    lock_json: Path = LOCK_JSON,
    fail_on_failed: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [
        "python3",
        str(SCRIPT),
        "--readme",
        str(readme),
        "--lock-json",
        str(lock_json),
        "--json",
        str(tmp_path / "operator_docs.json"),
        "--markdown",
        str(tmp_path / "operator_docs.md"),
    ]
    if fail_on_failed:
        command.append("--fail-on-failed")
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_verify_nemoclaw_operator_docs_accepts_repo_readme(tmp_path):
    result = run_verifier(tmp_path, fail_on_failed=True)

    assert result.returncode == 0, result.stderr
    payload = json.loads((tmp_path / "operator_docs.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["path"] == str(tmp_path / "operator_docs.json")
    assert payload["markdown_path"] == str(tmp_path / "operator_docs.md")
    assert payload["ok"] is True
    assert payload["status"] == "passed"
    assert payload["will_execute_installer"] is False
    assert payload["will_install_or_onboard"] is False
    assert payload["will_launch_model_inference"] is False
    assert payload["will_query_wandb"] is False
    assert payload["missing_requirements"] == []
    assert {check["name"] for check in payload["checks"]} >= {
        "check_only_command",
        "installer_review_command",
        "install_and_onboard_command",
        "post_install_verification_command",
        "canary_readiness_command",
        "adoption_check_command",
        "production_readiness_command",
        "operator_docs_verifier_command",
        "no_openrouter_markers",
    }
    assert "operator_docs_verifier_command" in (tmp_path / "operator_docs.md").read_text(
        encoding="utf-8"
    )


def test_verify_nemoclaw_operator_docs_rejects_missing_required_marker(tmp_path):
    readme = tmp_path / "README_nemoclaw.md"
    readme.write_text(
        README.read_text(encoding="utf-8").replace("--policy-tier restricted", ""),
        encoding="utf-8",
    )

    result = run_verifier(tmp_path, readme=readme, fail_on_failed=True)

    assert result.returncode == 1
    payload = json.loads((tmp_path / "operator_docs.json").read_text(encoding="utf-8"))
    assert payload["ok"] is False
    failing = {check["name"]: check for check in payload["checks"] if not check["ok"]}
    assert "install_and_onboard_command" in failing
    assert "--policy-tier restricted" in failing["install_and_onboard_command"][
        "missing_markers"
    ]


def test_verify_nemoclaw_operator_docs_rejects_missing_adoption_fail_fast_marker(tmp_path):
    readme = tmp_path / "README_nemoclaw.md"
    readme.write_text(
        README.read_text(encoding="utf-8").replace("  --fail-on-not-adoptable\n", ""),
        encoding="utf-8",
    )

    result = run_verifier(tmp_path, readme=readme, fail_on_failed=True)

    assert result.returncode == 1
    payload = json.loads((tmp_path / "operator_docs.json").read_text(encoding="utf-8"))
    assert payload["ok"] is False
    failing = {check["name"]: check for check in payload["checks"] if not check["ok"]}
    assert "adoption_check_command" in failing
    assert "--fail-on-not-adoptable" in failing["adoption_check_command"][
        "missing_markers"
    ]


def test_verify_nemoclaw_operator_docs_rejects_missing_production_readiness_fail_fast_marker(
    tmp_path,
):
    readme = tmp_path / "README_nemoclaw.md"
    readme.write_text(
        README.read_text(encoding="utf-8").replace("  --fail-on-not-ready\n", ""),
        encoding="utf-8",
    )

    result = run_verifier(tmp_path, readme=readme, fail_on_failed=True)

    assert result.returncode == 1
    payload = json.loads((tmp_path / "operator_docs.json").read_text(encoding="utf-8"))
    assert payload["ok"] is False
    failing = {check["name"]: check for check in payload["checks"] if not check["ok"]}
    assert "production_readiness_command" in failing
    assert "--fail-on-not-ready" in failing["production_readiness_command"][
        "missing_markers"
    ]


def test_verify_nemoclaw_operator_docs_rejects_openrouter_marker(tmp_path):
    readme = tmp_path / "README_nemoclaw.md"
    readme.write_text(
        README.read_text(encoding="utf-8") + "\nOPENROUTER_API_KEY=bad\n",
        encoding="utf-8",
    )

    result = run_verifier(tmp_path, readme=readme, fail_on_failed=True)

    assert result.returncode == 1
    payload = json.loads((tmp_path / "operator_docs.json").read_text(encoding="utf-8"))
    assert payload["ok"] is False
    failing = {check["name"]: check for check in payload["checks"] if not check["ok"]}
    assert failing["no_openrouter_markers"]["found"]
