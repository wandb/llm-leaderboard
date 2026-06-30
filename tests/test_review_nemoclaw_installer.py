import hashlib
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "setup" / "review_nemoclaw_installer.py"


def test_review_nemoclaw_installer_hashes_local_file_without_execution(tmp_path):
    installer = tmp_path / "nemoclaw.sh"
    installer.write_text("#!/usr/bin/env sh\nexit 99\n", encoding="utf-8")
    expected_sha = hashlib.sha256(installer.read_bytes()).hexdigest()
    output = tmp_path / "review.json"
    markdown = tmp_path / "review.md"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--url",
            str(installer),
            "--json",
            str(output),
            "--markdown",
            str(markdown),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["status"] == "reviewed"
    assert payload["sha256"] == expected_sha
    assert payload["size_bytes"] == len(installer.read_bytes())
    assert payload["will_execute_installer"] is False
    assert payload["will_install_or_onboard"] is False
    assert "--provider openai" in payload["recommended_install_command"]
    assert f"--installer-sha256 {expected_sha}" in payload["recommended_install_command"]
    assert f"--installer-review-json {output}" in payload["recommended_install_command"]
    assert expected_sha in markdown.read_text(encoding="utf-8")


def test_review_nemoclaw_installer_accepts_matching_lock_json(tmp_path):
    installer = tmp_path / "nemoclaw.sh"
    installer.write_text("#!/usr/bin/env sh\nexit 99\n", encoding="utf-8")
    expected_sha = hashlib.sha256(installer.read_bytes()).hexdigest()
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "installer_url": str(installer),
                "install_ref": "lkg",
                "sha256": expected_sha,
                "size_bytes": len(installer.read_bytes()),
                "will_execute_installer": False,
                "will_install_or_onboard": False,
                "will_launch_model_inference": False,
                "will_query_wandb": False,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "review.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--url",
            str(installer),
            "--lock-json",
            str(lock),
            "--json",
            str(output),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["status"] == "reviewed"
    assert payload["lock_json"] == str(lock)
    assert payload["lock_verified"] is True
    assert payload["expected_sha256"] == expected_sha
    assert payload["sha256"] == expected_sha
    assert f"--installer-lock-json {lock}" in payload["recommended_install_command"]


def test_review_nemoclaw_installer_can_record_gateway_port(tmp_path):
    installer = tmp_path / "nemoclaw.sh"
    installer.write_text("#!/usr/bin/env sh\nexit 99\n", encoding="utf-8")
    output = tmp_path / "review.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--url",
            str(installer),
            "--gateway-port",
            "18080",
            "--json",
            str(output),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "--provider openai" in payload["recommended_install_command"]
    assert "--gateway-port 18080" in payload["recommended_install_command"]


def test_review_nemoclaw_installer_rejects_lock_url_mismatch_without_execution(tmp_path):
    installer = tmp_path / "nemoclaw.sh"
    installer.write_text("#!/usr/bin/env sh\nexit 99\n", encoding="utf-8")
    expected_sha = hashlib.sha256(installer.read_bytes()).hexdigest()
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "installer_url": "https://example.invalid/nemoclaw.sh",
                "install_ref": "lkg",
                "sha256": expected_sha,
                "size_bytes": len(installer.read_bytes()),
                "will_execute_installer": False,
                "will_install_or_onboard": False,
                "will_launch_model_inference": False,
                "will_query_wandb": False,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "review.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--url",
            str(installer),
            "--lock-json",
            str(lock),
            "--json",
            str(output),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["status"] == "lock_mismatch"
    assert "lock installer_url does not match --url" in payload["errors"]
    assert payload["sha256"] == ""


def test_review_nemoclaw_installer_rejects_expected_sha_mismatch(tmp_path):
    installer = tmp_path / "nemoclaw.sh"
    installer.write_text("#!/usr/bin/env sh\nexit 0\n", encoding="utf-8")
    output = tmp_path / "review.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--url",
            str(installer),
            "--expected-sha256",
            "0" * 64,
            "--json",
            str(output),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["status"] == "sha256_mismatch"
    assert payload["errors"]
