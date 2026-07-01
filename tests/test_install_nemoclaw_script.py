import hashlib
import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "setup" / "install_nemoclaw.sh"
PROVIDER_ENV_KEYS = [
    "NVIDIA_API_KEY",
    "OPENAI_API_KEY",
    "NEMOCLAW_PROVIDER_KEY",
    "COMPATIBLE_API_KEY",
    "OPENAI_COMPATIBLE_API_KEY",
    "VLLM_API_KEY",
    "LITELLM_MASTER_KEY",
    "LITELLM_API_KEY",
    "NEMOCLAW_PROVIDER_KEY_ENV",
    "NEMOCLAW_ENDPOINT_URL",
    "OPENAI_COMPATIBLE_BASE_URL",
    "OPENAI_COMPATIBLE_API_BASE",
    "OPENAI_COMPATIBLE_ENDPOINT_URL",
    "OPENAI_COMPATIBLE_ENDPOINT",
    "VLLM_ENDPOINT_URL",
    "VLLM_BASE_URL",
    "ANTHROPIC_API_KEY",
    "CLAUDE_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
]


def scrub_provider_env(env):
    for key in PROVIDER_ENV_KEYS:
        env.pop(key, None)


def write_executable(path, text):
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def fake_bin(tmp_path, *, include_openshell):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("curl", "git", "node", "npm", "zstd"):
        write_executable(
            bin_dir / name,
            f"#!/usr/bin/env sh\n[ \"$1\" = \"--version\" ] && echo '{name} test' || echo '{name} test'\n",
        )
    write_executable(
        bin_dir / "docker",
        """#!/usr/bin/env sh
if [ "$1" = "--version" ]; then
  echo "Docker version test"
  exit 0
fi
if [ "$1" = "info" ]; then
  if [ "$2" = "--format" ]; then
    echo "test linux x86_64"
  else
    echo "docker info ok"
  fi
  exit 0
fi
echo "unexpected docker $*" >&2
exit 1
""",
    )
    write_executable(
        bin_dir / "nemoclaw",
        """#!/usr/bin/env sh
if [ "$1" = "--version" ]; then
  echo "nemoclaw test"
  exit 0
fi
if [ "$1" = "status" ] && [ "$2" = "--json" ]; then
  echo '{"status":"ok"}'
  exit 0
fi
if [ "$1" = "list" ] && [ "$2" = "--json" ]; then
  echo '{"sandboxes":[]}'
  exit 0
fi
echo "unexpected nemoclaw $*" >&2
exit 1
""",
    )
    if include_openshell:
        write_executable(
            bin_dir / "openshell",
            """#!/usr/bin/env sh
if [ "$1" = "--version" ]; then
  echo "openshell test"
  exit 0
fi
if [ "$1" = "status" ]; then
  echo "openshell status ok"
  exit 0
fi
echo "unexpected openshell $*" >&2
exit 1
""",
        )
    return bin_dir


def run_check(tmp_path, *, include_openshell):
    bin_dir = fake_bin(tmp_path, include_openshell=include_openshell)
    output = tmp_path / "nemoclaw_check.json"
    env = os.environ.copy()
    scrub_provider_env(env)
    env["PATH"] = str(bin_dir) + os.pathsep + "/usr/bin:/bin"
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--check-only",
            "--json",
            str(output),
            "--sandbox",
            "nejumi-taiwan-test",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, json.loads(output.read_text(encoding="utf-8"))


def write_installer_review(
    path: Path,
    *,
    sha256: str,
    url: str = "https://www.nvidia.com/nemoclaw.sh",
    lock_json: str = "scripts/setup/nemoclaw_installer_lock.json",
    lock_verified: bool = True,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ok": True,
                "status": "reviewed",
                "installer_url": url,
                "install_ref": "lkg",
                "lock_json": lock_json,
                "lock_verified": lock_verified,
                "sha256": sha256,
                "will_execute_installer": False,
                "will_install_or_onboard": False,
                "will_launch_model_inference": False,
                "will_query_wandb": False,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_install_nemoclaw_check_json_fails_when_openshell_missing(tmp_path):
    result, payload = run_check(tmp_path, include_openshell=False)

    assert result.returncode == 1
    assert payload["schema_version"] == 1
    assert payload["ok"] is False
    assert payload["exit_code"] == 1
    assert payload["provider"] == "openai"
    assert payload["gateway_port"] is None
    assert payload["provider_preflight"]["provider"] == "openai"
    assert payload["provider_preflight"]["normalized_provider"] == "openai"
    assert payload["provider_preflight"]["credential_envs"] == [
        "OPENAI_API_KEY",
        "NEMOCLAW_PROVIDER_KEY",
    ]
    assert payload["provider_preflight"]["credential_available"] is False
    assert payload["provider_preflight"]["nvidia_api_key_required"] is False
    assert payload["provider_preflight"]["will_launch_probe"] is False
    assert payload["sandbox_configured"] is False
    assert payload["commands"]["nemoclaw"]["available"] is True
    assert payload["commands"]["openshell"]["available"] is False
    assert payload["commands"]["openshell"]["required"] is True
    assert payload["host_prerequisites_ok"] is True
    assert payload["runtime_installed"] is False
    assert payload["missing_required_commands"] == ["openshell"]
    assert payload["third_party_software"] == {
        "name": "NVIDIA NemoClaw",
        "vendor": "NVIDIA",
        "repository_url": "https://github.com/NVIDIA/NemoClaw",
        "documentation_url": "https://docs.nvidia.com/nemoclaw/user-guide/openclaw/get-started/quickstart",
        "installer_url": "https://www.nvidia.com/nemoclaw.sh",
        "install_ref": "lkg",
            "installer_sha256": "",
            "installer_signature": "",
            "installer_lock_json": "scripts/setup/nemoclaw_installer_lock.json",
            "installer_review_json": "",
            "installer_review_verified": False,
            "installer_integrity_verified": False,
        "installer_provenance_locked": False,
        "installer_provenance_note": "Installer integrity is not verified by this script; operator review is required before install/onboard.",
        "acceptance_required": True,
        "acceptance_flag": "--yes-i-accept-third-party-software",
        "accepted": False,
        "install_or_onboard_requested": False,
        "operator_review_required_before_install": True,
    }
    assert payload["setup_plan"]["will_launch_model_inference"] is False
    assert payload["setup_plan"]["will_launch_benchmark_inference"] is False
    assert payload["setup_plan"]["onboard_may_validate_provider_endpoint"] is True
    assert payload["setup_plan"]["provider"] == "openai"
    assert payload["setup_plan"]["gateway_port"] is None
    assert payload["setup_plan"]["provider_preflight"] == payload["provider_preflight"]
    assert payload["setup_plan"]["sandbox_configured"] is False
    assert payload["setup_plan"]["sandbox_readiness_required"] is True
    assert payload["setup_plan"]["provider_notes"]["build"] == (
        "Uses NVIDIA_API_KEY for NVIDIA hosted endpoints."
    )
    assert payload["setup_plan"]["install_or_onboard_requires_explicit_acceptance"] is True
    assert payload["setup_plan"]["acceptance_flag"] == "--yes-i-accept-third-party-software"
    assert payload["setup_plan"]["third_party_software_name"] == "NVIDIA NemoClaw"
    assert payload["setup_plan"]["installer_url"] == "https://www.nvidia.com/nemoclaw.sh"
    assert payload["setup_plan"]["install_ref"] == "lkg"
    assert payload["setup_plan"]["installer_sha256"] == ""
    assert payload["setup_plan"]["installer_signature"] == ""
    assert payload["setup_plan"]["installer_lock_json"] == "scripts/setup/nemoclaw_installer_lock.json"
    assert payload["setup_plan"]["installer_review_json"] == ""
    assert payload["setup_plan"]["installer_review_verified"] is False
    assert payload["setup_plan"]["installer_integrity_verified"] is False
    assert payload["setup_plan"]["installer_provenance_locked"] is False
    assert payload["setup_plan"]["installer_provenance_note"] == (
        "Installer integrity is not verified by this script; operator review is required before install/onboard."
    )
    assert payload["policy_tier"] == "restricted"
    assert payload["policy_tier_allowed_values"] == ["restricted", "balanced", "open"]
    assert payload["policy_tier_valid"] is True
    assert payload["setup_plan"]["policy_tier"] == "restricted"
    assert payload["setup_plan"]["policy_tier_allowed_values"] == [
        "restricted",
        "balanced",
        "open",
    ]
    assert payload["setup_plan"]["policy_tier_valid"] is True
    assert payload["setup_plan"]["acceptance_ledger_fields"] == [
        "accepted_third_party_software",
        "third_party_software.name",
        "third_party_software.vendor",
        "third_party_software.installer_url",
        "third_party_software.install_ref",
        "third_party_software.installer_sha256",
        "third_party_software.installer_signature",
        "third_party_software.installer_lock_json",
        "third_party_software.installer_review_json",
        "third_party_software.installer_review_verified",
        "third_party_software.installer_integrity_verified",
        "third_party_software.installer_provenance_locked",
        "third_party_software.installer_provenance_note",
        "third_party_software.acceptance_required",
        "third_party_software.acceptance_flag",
        "third_party_software.accepted",
        "policy_tier",
        "policy_tier_allowed_values",
        "policy_tier_valid",
        "setup_plan.policy_tier",
        "setup_plan.policy_tier_allowed_values",
        "setup_plan.policy_tier_valid",
        "setup_plan.installer_sha256",
        "setup_plan.installer_signature",
        "setup_plan.installer_lock_json",
        "setup_plan.installer_review_json",
        "setup_plan.installer_review_verified",
        "setup_plan.installer_integrity_verified",
        "setup_plan.installer_provenance_locked",
        "setup_plan.installer_provenance_note",
        "operation_results.install.log_path",
        "operation_results.onboard.log_path",
    ]
    assert "review_nemoclaw_installer.py" in payload["setup_plan"]["installer_review_command"]
    assert (
        "--expected-sha256 a4ebc5710dfd8b10035968fd25562773ad56ec4c73ecf9bfe5c78c77724000e7"
        in payload["setup_plan"]["installer_review_command"]
    )
    assert "--lock-json scripts/setup/nemoclaw_installer_lock.json" in payload["setup_plan"]["installer_review_command"]
    assert "--json temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json" in payload["setup_plan"]["installer_review_command"]
    assert "--install" in payload["setup_plan"]["install_command"]
    assert "--installer-lock-json scripts/setup/nemoclaw_installer_lock.json" in payload["setup_plan"]["install_command"]
    assert "--installer-sha256 REVIEWED_INSTALLER_SHA256" in payload["setup_plan"]["install_command"]
    assert "--installer-review-json temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json" in payload["setup_plan"]["install_command"]
    assert "--json temp/nemoclaw_install_YYYYMMDDTHHMM.json" in payload["setup_plan"]["install_command"]
    assert "--onboard" in payload["setup_plan"]["onboard_command"]
    assert "--provider openai" in payload["setup_plan"]["onboard_command"]
    assert "--policy-tier restricted" in payload["setup_plan"]["onboard_command"]
    assert "--json temp/nemoclaw_onboard_YYYYMMDDTHHMM.json" in payload["setup_plan"]["onboard_command"]
    assert "--install --onboard" in payload["setup_plan"]["install_and_onboard_command"]
    assert "--provider openai" in payload["setup_plan"]["install_and_onboard_command"]
    assert "--installer-lock-json scripts/setup/nemoclaw_installer_lock.json" in payload["setup_plan"]["install_and_onboard_command"]
    assert "--installer-sha256 REVIEWED_INSTALLER_SHA256" in payload["setup_plan"]["install_and_onboard_command"]
    assert "--installer-review-json temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json" in payload["setup_plan"]["install_and_onboard_command"]
    assert "--policy-tier restricted" in payload["setup_plan"]["install_and_onboard_command"]
    assert "--json temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.json" in payload["setup_plan"]["install_and_onboard_command"]
    assert (
        payload["setup_plan"]["production_install_and_onboard_command"]
        == payload["setup_plan"]["install_and_onboard_command"]
    )
    assert "--install --onboard" in payload["setup_plan"]["production_install_and_onboard_command"]
    assert "--installer-lock-json scripts/setup/nemoclaw_installer_lock.json" in payload["setup_plan"]["production_install_and_onboard_command"]
    assert "--installer-sha256 REVIEWED_INSTALLER_SHA256" in payload["setup_plan"]["production_install_and_onboard_command"]
    assert "--installer-review-json temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json" in payload["setup_plan"]["production_install_and_onboard_command"]
    assert "--policy-tier restricted" in payload["setup_plan"]["production_install_and_onboard_command"]
    assert "--json temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.json" in payload["setup_plan"]["production_install_and_onboard_command"]
    assert "verify_nemoclaw_post_install.py" in payload["setup_plan"]["post_install_verification_command"]
    assert "--nemoclaw-openclaw-config-path /sandbox/.openclaw/openclaw.json" in payload["setup_plan"]["post_install_verification_command"]
    assert "--fail-on-failed" in payload["setup_plan"]["post_install_verification_command"]
    assert "--manifest configs/taiwan_openai_canary_models.yaml" in payload["setup_plan"]["canary_readiness_command"]
    assert "--generated-agentic-dir configs/taiwan_full/generated_openai_canary_agentic_nemoclaw" in payload["setup_plan"]["canary_readiness_command"]
    assert "--require-nemoclaw" in payload["setup_plan"]["canary_readiness_command"]
    assert "openai_canary_readiness_nemoclaw.json" in payload["setup_plan"]["canary_readiness_command"]
    assert "check_taiwan_nemoclaw_adoption.py" in payload["setup_plan"]["adoption_check_command"]
    assert "--setup-json temp/nemoclaw_setup_check_YYYYMMDDTHHMM.json" in payload["setup_plan"]["adoption_check_command"]
    assert "--readiness-json outputs/taiwan_full_eval/openai_canary_readiness_nemoclaw.json" in payload["setup_plan"]["adoption_check_command"]
    assert "--sandbox nejumi-taiwan-test" in payload["setup_plan"]["adoption_check_command"]
    assert "--agentic-config-glob configs/taiwan_full/generated_openai_canary_agentic_nemoclaw/*.yaml" in payload["setup_plan"]["adoption_check_command"]
    assert "--json temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.json" in payload["setup_plan"]["adoption_check_command"]
    assert "--markdown temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.md" in payload["setup_plan"]["adoption_check_command"]
    assert "--fail-on-not-adoptable" in payload["setup_plan"]["adoption_check_command"]
    assert [row["step"] for row in payload["setup_plan"]["operator_sequence"]] == [
        "setup_check",
        "installer_review",
        "install_and_onboard",
        "post_install_verification",
        "canary_readiness",
        "adoption_check",
        "production_readiness",
    ]
    sequence = {
        row["step"]: row
        for row in payload["setup_plan"]["operator_sequence"]
    }
    assert sequence["install_and_onboard"]["requires_external_action"] is True
    assert sequence["install_and_onboard"]["command"] == payload["setup_plan"]["production_install_and_onboard_command"]
    assert sequence["installer_review"]["requires_external_action"] is False
    assert sequence["post_install_verification"]["command"] == payload["setup_plan"]["post_install_verification_command"]
    assert sequence["adoption_check"]["command"] == payload["setup_plan"]["adoption_check_command"]
    assert payload["setup_plan"]["expected_evidence_paths"] == [
        "temp/nemoclaw_setup_check_YYYYMMDDTHHMM.json",
        "temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json",
        "temp/nemoclaw_installer_review_YYYYMMDDTHHMM.md",
        "temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.json",
        "temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.install.log",
        "temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.onboard.log",
        "temp/nemoclaw_post_install_verification_YYYYMMDDTHHMM.json",
        "temp/nemoclaw_post_install_verification_YYYYMMDDTHHMM.md",
        "outputs/taiwan_full_eval/openai_canary_readiness_nemoclaw.json",
        "temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.json",
        "temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.md",
        "outputs/taiwan_full_eval/taiwan_production_readiness_report.json",
    ]


def test_install_nemoclaw_check_json_passes_with_required_commands(tmp_path):
    result, payload = run_check(tmp_path, include_openshell=True)

    assert result.returncode == 0
    assert payload["ok"] is True
    assert payload["sandbox"] == "nejumi-taiwan-test"
    assert payload["commands"]["docker"]["info_ok"] is True
    assert payload["commands"]["openshell"]["available"] is True
    assert payload["host_prerequisites_ok"] is True
    assert payload["runtime_installed"] is True
    assert payload["missing_required_commands"] == []
    assert payload["sandbox_configured"] is False


def test_install_nemoclaw_rejects_unknown_policy_tier_and_writes_json(tmp_path):
    bin_dir = fake_bin(tmp_path, include_openshell=True)
    output = tmp_path / "nemoclaw_invalid_policy.json"
    env = os.environ.copy()
    scrub_provider_env(env)
    env["PATH"] = str(bin_dir) + os.pathsep + "/usr/bin:/bin"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--check-only",
            "--policy-tier",
            "experimental",
            "--json",
            str(output),
            "--sandbox",
            "nejumi-taiwan-test",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Invalid --policy-tier" in result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["exit_code"] == 2
    assert payload["policy_tier"] == "experimental"
    assert payload["policy_tier_allowed_values"] == ["restricted", "balanced", "open"]
    assert payload["policy_tier_valid"] is False
    assert payload["setup_plan"]["policy_tier"] == "experimental"
    assert payload["setup_plan"]["policy_tier_allowed_values"] == [
        "restricted",
        "balanced",
        "open",
    ]
    assert payload["setup_plan"]["policy_tier_valid"] is False
    assert payload["install_requested"] is False
    assert payload["onboard_requested"] is False


def test_install_nemoclaw_check_json_records_gateway_port(tmp_path):
    bin_dir = fake_bin(tmp_path, include_openshell=True)
    output = tmp_path / "nemoclaw_gateway.json"
    env = os.environ.copy()
    scrub_provider_env(env)
    env["PATH"] = str(bin_dir) + os.pathsep + "/usr/bin:/bin"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--check-only",
            "--gateway-port",
            "18080",
            "--json",
            str(output),
            "--sandbox",
            "nejumi-taiwan-test",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["gateway_port"] == "18080"
    assert payload["setup_plan"]["gateway_port"] == "18080"
    assert "--gateway-port 18080" in payload["setup_plan"]["onboard_command"]
    assert "--gateway-port 18080" in payload["setup_plan"]["install_and_onboard_command"]


def test_install_nemoclaw_provider_preflight_distinguishes_nvidia_and_custom(tmp_path):
    bin_dir = fake_bin(tmp_path, include_openshell=True)
    env = os.environ.copy()
    scrub_provider_env(env)
    env["PATH"] = str(bin_dir) + os.pathsep + "/usr/bin:/bin"

    build_output = tmp_path / "nemoclaw_build_provider.json"
    build_result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--check-only",
            "--provider",
            "build",
            "--json",
            str(build_output),
            "--sandbox",
            "nejumi-taiwan-test",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert build_result.returncode == 0
    build_payload = json.loads(build_output.read_text(encoding="utf-8"))
    assert build_payload["provider_preflight"]["normalized_provider"] == "build"
    assert build_payload["provider_preflight"]["nvidia_hosted_endpoint"] is True
    assert build_payload["provider_preflight"]["nvidia_api_key_required"] is True
    assert build_payload["provider_preflight"]["credential_envs"] == [
        "NVIDIA_API_KEY",
        "NEMOCLAW_PROVIDER_KEY",
    ]
    assert build_payload["provider_preflight"]["credential_available"] is False
    assert build_payload["provider_preflight"]["missing"] == ["credential"]

    custom_output = tmp_path / "nemoclaw_custom_provider.json"
    custom_env = env.copy()
    custom_env["COMPATIBLE_API_KEY"] = "hidden-test-key"
    custom_env["NEMOCLAW_ENDPOINT_URL"] = "http://127.0.0.1:8000/v1"
    custom_result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--check-only",
            "--provider",
            "custom",
            "--model",
            "local-model",
            "--json",
            str(custom_output),
            "--sandbox",
            "nejumi-taiwan-test",
        ],
        cwd=REPO_ROOT,
        env=custom_env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert custom_result.returncode == 0
    custom_payload = json.loads(custom_output.read_text(encoding="utf-8"))
    preflight = custom_payload["provider_preflight"]
    assert preflight["normalized_provider"] == "custom"
    assert preflight["credential_available"] is True
    assert preflight["credential_present_envs"] == ["COMPATIBLE_API_KEY"]
    assert preflight["endpoint_available"] is True
    assert preflight["endpoint_requirement_mode"] == "any"
    assert preflight["endpoint_present_envs"] == ["NEMOCLAW_ENDPOINT_URL"]
    assert preflight["model_configured"] is True
    assert preflight["ready_for_noninteractive_onboard_preflight"] is True
    assert "hidden-test-key" not in json.dumps(preflight)


def test_install_nemoclaw_custom_provider_accepts_openai_compatible_aliases(tmp_path):
    bin_dir = fake_bin(tmp_path, include_openshell=True)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_COMPATIBLE_API_KEY=hidden-compatible-key\n",
        encoding="utf-8",
    )
    output = tmp_path / "nemoclaw_custom_aliases.json"
    env = os.environ.copy()
    scrub_provider_env(env)
    env["PATH"] = str(bin_dir) + os.pathsep + "/usr/bin:/bin"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--check-only",
            "--provider",
            "custom",
            "--model",
            "chat-model",
            "--endpoint-url",
            "http://127.0.0.1:8080/v1",
            "--env-file",
            str(env_file),
            "--json",
            str(output),
            "--sandbox",
            "nejumi-taiwan-test",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    preflight = payload["provider_preflight"]
    assert payload["model"] == "chat-model"
    assert payload["setup_plan"]["model"] == "chat-model"
    assert preflight["normalized_provider"] == "custom"
    assert preflight["credential_available"] is True
    assert preflight["credential_present_envs"] == ["OPENAI_COMPATIBLE_API_KEY"]
    assert preflight["endpoint_available"] is True
    assert preflight["endpoint_present_envs"] == ["NEMOCLAW_ENDPOINT_URL"]
    assert preflight["model_configured"] is True
    assert preflight["ready_for_noninteractive_onboard_preflight"] is True
    assert "--endpoint-url http://127.0.0.1:8080/v1" in payload["setup_plan"]["onboard_command"]
    assert "--endpoint-url http://127.0.0.1:8080/v1" in payload["setup_plan"]["post_install_check_command"]
    assert "hidden-compatible-key" not in json.dumps(payload)


def test_install_nemoclaw_custom_provider_key_env_is_recorded_without_secret(tmp_path):
    bin_dir = fake_bin(tmp_path, include_openshell=True)
    env_file = tmp_path / ".env"
    env_file.write_text("LITELLM_MASTER_KEY=hidden-litellm-master-key\n", encoding="utf-8")
    output = tmp_path / "nemoclaw_custom_key_env.json"
    env = os.environ.copy()
    scrub_provider_env(env)
    env["PATH"] = str(bin_dir) + os.pathsep + "/usr/bin:/bin"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--check-only",
            "--provider",
            "custom",
            "--model",
            "chat-model",
            "--endpoint-url",
            "http://127.0.0.1:8080/v1",
            "--provider-key-env",
            "LITELLM_MASTER_KEY",
            "--env-file",
            str(env_file),
            "--json",
            str(output),
            "--sandbox",
            "nejumi-taiwan-test",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    preflight = payload["provider_preflight"]
    assert payload["model"] == "chat-model"
    assert payload["setup_plan"]["model"] == "chat-model"
    assert payload["provider_key_env"] == "LITELLM_MASTER_KEY"
    assert payload["setup_plan"]["provider_key_env"] == "LITELLM_MASTER_KEY"
    assert preflight["provider_key_env"] == "LITELLM_MASTER_KEY"
    assert preflight["credential_envs"][0] == "LITELLM_MASTER_KEY"
    assert preflight["credential_available"] is True
    assert preflight["credential_present_envs"] == ["LITELLM_MASTER_KEY"]
    assert preflight["ready_for_noninteractive_onboard_preflight"] is True
    assert "--provider-key-env LITELLM_MASTER_KEY" in payload["setup_plan"]["onboard_command"]
    assert "--provider-key-env LITELLM_MASTER_KEY" in payload["setup_plan"]["post_install_check_command"]
    assert "hidden-litellm-master-key" not in json.dumps(payload)
    assert "hidden-litellm-master-key" not in result.stdout
    assert "hidden-litellm-master-key" not in result.stderr


def test_install_nemoclaw_custom_provider_key_env_is_bridged_on_onboard(tmp_path):
    bin_dir = fake_bin(tmp_path, include_openshell=True)
    report = tmp_path / "fake_onboard_env.txt"
    write_executable(
        bin_dir / "nemoclaw",
        f"""#!/usr/bin/env sh
if [ "$1" = "--version" ]; then
  echo "nemoclaw test"
  exit 0
fi
if [ "$1" = "status" ] && [ "$2" = "--json" ]; then
  echo '{{"status":"ok"}}'
  exit 0
fi
if [ "$1" = "list" ] && [ "$2" = "--json" ]; then
  echo '{{"sandboxes":[]}}'
  exit 0
fi
if [ "$1" = "onboard" ]; then
  {{
    [ "$NEMOCLAW_PROVIDER" = "custom" ] && echo provider_ok=true || echo provider_ok=false
    [ "$NEMOCLAW_MODEL" = "chat-model" ] && echo model_ok=true || echo model_ok=false
    [ "$NEMOCLAW_PROVIDER_KEY" = "hidden-litellm-master-key" ] && echo provider_key_ok=true || echo provider_key_ok=false
    [ "$COMPATIBLE_API_KEY" = "hidden-litellm-master-key" ] && echo compatible_key_ok=true || echo compatible_key_ok=false
    [ "$NEMOCLAW_ENDPOINT_URL" = "http://127.0.0.1:8080/v1" ] && echo endpoint_ok=true || echo endpoint_ok=false
  }} > {str(report)!r}
  echo "onboard ok"
  exit 0
fi
echo "unexpected nemoclaw $*" >&2
exit 1
""",
    )
    output = tmp_path / "nemoclaw_onboard_custom_key_env.json"
    env = os.environ.copy()
    scrub_provider_env(env)
    env["PATH"] = str(bin_dir) + os.pathsep + "/usr/bin:/bin"
    env["LITELLM_MASTER_KEY"] = "hidden-litellm-master-key"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--onboard",
            "--provider",
            "custom",
            "--model",
            "chat-model",
            "--endpoint-url",
            "http://127.0.0.1:8080/v1",
            "--provider-key-env",
            "LITELLM_MASTER_KEY",
            "--yes-i-accept-third-party-software",
            "--json",
            str(output),
            "--sandbox",
            "nejumi-taiwan-test",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["provider_key_env"] == "LITELLM_MASTER_KEY"
    assert payload["operation_results"]["onboard"]["attempted"] is True
    assert "provider_key_ok=true" in report.read_text(encoding="utf-8")
    assert "compatible_key_ok=true" in report.read_text(encoding="utf-8")
    assert "endpoint_ok=true" in report.read_text(encoding="utf-8")
    assert "hidden-litellm-master-key" not in json.dumps(payload)
    assert "hidden-litellm-master-key" not in result.stdout
    assert "hidden-litellm-master-key" not in result.stderr
    assert "hidden-litellm-master-key" not in Path(
        payload["operation_results"]["onboard"]["log_path"]
    ).read_text(encoding="utf-8")


def test_install_nemoclaw_rejects_invalid_provider_key_env(tmp_path):
    bin_dir = fake_bin(tmp_path, include_openshell=True)
    output = tmp_path / "nemoclaw_invalid_provider_key_env.json"
    env = os.environ.copy()
    scrub_provider_env(env)
    env["PATH"] = str(bin_dir) + os.pathsep + "/usr/bin:/bin"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--check-only",
            "--provider",
            "custom",
            "--model",
            "chat-model",
            "--endpoint-url",
            "http://127.0.0.1:8080/v1",
            "--provider-key-env",
            "not-valid-name!",
            "--json",
            str(output),
            "--sandbox",
            "nejumi-taiwan-test",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Invalid --provider-key-env" in result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["provider_key_env"] == "not-valid-name!"


def test_install_nemoclaw_can_load_dotenv_for_provider_preflight(tmp_path):
    bin_dir = fake_bin(tmp_path, include_openshell=True)
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=hidden-openai-key\n", encoding="utf-8")
    output = tmp_path / "nemoclaw_env_file.json"
    env = os.environ.copy()
    scrub_provider_env(env)
    env["PATH"] = str(bin_dir) + os.pathsep + "/usr/bin:/bin"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--check-only",
            "--env-file",
            str(env_file),
            "--json",
            str(output),
            "--sandbox",
            "nejumi-taiwan-test",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["env_file"] == str(env_file)
    assert payload["env_file_loaded"] is True
    assert payload["provider_preflight"]["credential_available"] is True
    assert payload["provider_preflight"]["credential_present_envs"] == ["OPENAI_API_KEY"]
    assert payload["provider_preflight"]["ready_for_noninteractive_onboard_preflight"] is True
    assert "--env-file " + str(env_file) in payload["setup_plan"]["onboard_command"]
    assert "--env-file " + str(env_file) in payload["setup_plan"]["post_install_check_command"]
    assert "hidden-openai-key" not in json.dumps(payload)


def test_install_nemoclaw_rejects_missing_env_file_and_writes_json(tmp_path):
    bin_dir = fake_bin(tmp_path, include_openshell=True)
    missing_env = tmp_path / "missing.env"
    output = tmp_path / "nemoclaw_missing_env.json"
    env = os.environ.copy()
    scrub_provider_env(env)
    env["PATH"] = str(bin_dir) + os.pathsep + "/usr/bin:/bin"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--check-only",
            "--env-file",
            str(missing_env),
            "--json",
            str(output),
            "--sandbox",
            "nejumi-taiwan-test",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "env file not found" in result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["env_file"] == str(missing_env)
    assert payload["env_file_loaded"] is False
    assert "env file not found" in payload["env_file_load_error"]


def test_install_nemoclaw_rejects_invalid_gateway_port_and_writes_json(tmp_path):
    bin_dir = fake_bin(tmp_path, include_openshell=True)
    output = tmp_path / "nemoclaw_invalid_gateway.json"
    env = os.environ.copy()
    scrub_provider_env(env)
    env["PATH"] = str(bin_dir) + os.pathsep + "/usr/bin:/bin"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--check-only",
            "--gateway-port",
            "not-a-port",
            "--json",
            str(output),
            "--sandbox",
            "nejumi-taiwan-test",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Invalid --gateway-port" in result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["gateway_port"] == "not-a-port"
    assert payload["setup_plan"]["gateway_port"] == "not-a-port"


def test_install_nemoclaw_install_refusal_still_writes_json(tmp_path):
    bin_dir = fake_bin(tmp_path, include_openshell=True)
    output = tmp_path / "nemoclaw_install_refused.json"
    env = os.environ.copy()
    scrub_provider_env(env)
    env["PATH"] = str(bin_dir) + os.pathsep + "/usr/bin:/bin"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--install",
            "--json",
            str(output),
            "--sandbox",
            "nejumi-taiwan-test",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["exit_code"] == 2
    assert payload["install_requested"] is True
    assert payload["accepted_third_party_software"] is False
    assert payload["third_party_software"]["accepted"] is False
    assert payload["third_party_software"]["install_or_onboard_requested"] is True
    assert payload["operation_results"]["install"] == {
        "requested": True,
        "attempted": True,
        "returncode": 2,
        "log_path": str(tmp_path / "nemoclaw_install_refused.install.log"),
    }
    assert payload["operation_results"]["onboard"] == {
        "requested": False,
        "attempted": False,
        "skipped": False,
        "returncode": None,
        "log_path": "",
        "failure": {"failure_kind": "", "failure_detail": ""},
    }
    install_log = tmp_path / "nemoclaw_install_refused.install.log"
    assert install_log.exists()
    assert "Refusing to install NemoClaw" in install_log.read_text(encoding="utf-8")


def test_install_nemoclaw_requires_installer_sha256_for_accepted_install(tmp_path):
    bin_dir = fake_bin(tmp_path, include_openshell=True)
    output = tmp_path / "nemoclaw_install_no_sha.json"
    env = os.environ.copy()
    scrub_provider_env(env)
    env["PATH"] = str(bin_dir) + os.pathsep + "/usr/bin:/bin"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--install",
            "--yes-i-accept-third-party-software",
            "--json",
            str(output),
            "--sandbox",
            "nejumi-taiwan-test",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["third_party_software"]["installer_integrity_verified"] is False
    assert payload["third_party_software"]["installer_provenance_locked"] is False
    install_log = tmp_path / "nemoclaw_install_no_sha.install.log"
    assert "reviewed installer SHA-256" in install_log.read_text(encoding="utf-8")


def test_install_nemoclaw_requires_installer_review_json_for_accepted_install(tmp_path):
    bin_dir = fake_bin(tmp_path, include_openshell=True)
    output = tmp_path / "nemoclaw_install_no_review.json"
    env = os.environ.copy()
    scrub_provider_env(env)
    env["PATH"] = str(bin_dir) + os.pathsep + "/usr/bin:/bin"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--install",
            "--yes-i-accept-third-party-software",
            "--installer-sha256",
            "0" * 64,
            "--json",
            str(output),
            "--sandbox",
            "nejumi-taiwan-test",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["third_party_software"]["installer_review_verified"] is False
    assert payload["setup_plan"]["installer_review_verified"] is False
    install_log = tmp_path / "nemoclaw_install_no_review.install.log"
    assert "installer review evidence" in install_log.read_text(encoding="utf-8")


def test_install_nemoclaw_rejects_unlocked_installer_review_json(tmp_path):
    bin_dir = fake_bin(tmp_path, include_openshell=True)
    output = tmp_path / "nemoclaw_install_unlocked_review.json"
    review = write_installer_review(
        tmp_path / "nemoclaw_installer_review.json",
        sha256="0" * 64,
        lock_json="",
        lock_verified=False,
    )
    env = os.environ.copy()
    scrub_provider_env(env)
    env["PATH"] = str(bin_dir) + os.pathsep + "/usr/bin:/bin"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--install",
            "--yes-i-accept-third-party-software",
            "--installer-sha256",
            "0" * 64,
            "--installer-review-json",
            str(review),
            "--json",
            str(output),
            "--sandbox",
            "nejumi-taiwan-test",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["third_party_software"]["installer_review_verified"] is False
    install_log = tmp_path / "nemoclaw_install_unlocked_review.install.log"
    log_text = install_log.read_text(encoding="utf-8")
    assert "lock_verified must be true" in log_text
    assert "lock_json does not match --installer-lock-json" in log_text


def test_install_nemoclaw_failed_install_skips_onboard_and_writes_json(tmp_path):
    bin_dir = fake_bin(tmp_path, include_openshell=True)
    write_executable(
        bin_dir / "curl",
        "#!/usr/bin/env sh\nexit 42\n",
    )
    output = tmp_path / "nemoclaw_install_failed.json"
    review = write_installer_review(tmp_path / "nemoclaw_installer_review.json", sha256="0" * 64)
    env = os.environ.copy()
    scrub_provider_env(env)
    env["PATH"] = str(bin_dir) + os.pathsep + "/usr/bin:/bin"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--install",
            "--onboard",
            "--yes-i-accept-third-party-software",
            "--installer-sha256",
            "0" * 64,
            "--installer-review-json",
            str(review),
            "--json",
            str(output),
            "--sandbox",
            "nejumi-taiwan-test",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 42
    assert "skipping onboard because install failed" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["exit_code"] == 42
    assert payload["accepted_third_party_software"] is True
    assert payload["third_party_software"]["accepted"] is True
    assert payload["third_party_software"]["install_or_onboard_requested"] is True
    assert payload["operation_results"]["install"]["attempted"] is True
    assert payload["operation_results"]["install"]["returncode"] == 42
    assert payload["operation_results"]["install"]["log_path"] == str(
        tmp_path / "nemoclaw_install_failed.install.log"
    )
    assert payload["operation_results"]["onboard"] == {
        "requested": True,
        "attempted": False,
        "skipped": True,
        "returncode": None,
        "log_path": "",
        "failure": {"failure_kind": "", "failure_detail": ""},
    }
    install_log = tmp_path / "nemoclaw_install_failed.install.log"
    assert install_log.exists()
    assert "downloading NemoClaw installer" in install_log.read_text(encoding="utf-8")


def test_install_nemoclaw_classifies_onboard_provider_quota_failure(tmp_path):
    bin_dir = fake_bin(tmp_path, include_openshell=True)
    write_executable(
        bin_dir / "nemoclaw",
        """#!/usr/bin/env sh
if [ "$1" = "--version" ]; then
  echo "nemoclaw test"
  exit 0
fi
if [ "$1" = "status" ] && [ "$2" = "--json" ]; then
  echo '{"status":"ok"}'
  exit 0
fi
if [ "$1" = "list" ] && [ "$2" = "--json" ]; then
  echo '{"sandboxes":[]}'
  exit 0
fi
if [ "$1" = "onboard" ]; then
  echo "OpenAI endpoint validation failed." >&2
  echo "HTTP 429: You exceeded your current quota." >&2
  exit 9
fi
echo "unexpected nemoclaw $*" >&2
exit 1
""",
    )
    output = tmp_path / "nemoclaw_onboard_quota.json"
    env = os.environ.copy()
    scrub_provider_env(env)
    env["PATH"] = str(bin_dir) + os.pathsep + "/usr/bin:/bin"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--onboard",
            "--yes-i-accept-third-party-software",
            "--json",
            str(output),
            "--sandbox",
            "nejumi-taiwan-test",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 9
    payload = json.loads(output.read_text(encoding="utf-8"))
    onboard = payload["operation_results"]["onboard"]
    assert onboard["attempted"] is True
    assert onboard["returncode"] == 9
    assert onboard["failure"] == {
        "failure_kind": "provider_quota",
        "failure_detail": "provider validation returned HTTP 429/quota",
    }
    assert "HTTP 429" in Path(onboard["log_path"]).read_text(encoding="utf-8")


def test_install_nemoclaw_verified_installer_sets_provenance_lock(tmp_path):
    bin_dir = fake_bin(tmp_path, include_openshell=True)
    installer_body = "#!/usr/bin/env sh\nexit 0\n"
    installer_sha = hashlib.sha256(installer_body.encode("utf-8")).hexdigest()
    write_executable(
        bin_dir / "curl",
        f"""#!/usr/bin/env sh
if [ "$1" = "-fsSL" ] && [ "$3" = "-o" ]; then
  cat > "$4" <<'EOS'
{installer_body}EOS
  exit 0
fi
exit 42
""",
    )
    output = tmp_path / "nemoclaw_install_verified.json"
    review = write_installer_review(tmp_path / "nemoclaw_installer_review.json", sha256=installer_sha)
    env = os.environ.copy()
    scrub_provider_env(env)
    env["PATH"] = str(bin_dir) + os.pathsep + "/usr/bin:/bin"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--install",
            "--yes-i-accept-third-party-software",
            "--installer-sha256",
            installer_sha,
            "--installer-review-json",
            str(review),
            "--json",
            str(output),
            "--sandbox",
            "nejumi-taiwan-test",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["third_party_software"]["installer_sha256"] == installer_sha
    assert payload["third_party_software"]["installer_review_json"] == str(review)
    assert payload["third_party_software"]["installer_review_verified"] is True
    assert payload["third_party_software"]["installer_integrity_verified"] is True
    assert payload["third_party_software"]["installer_provenance_locked"] is True
    assert payload["setup_plan"]["installer_review_json"] == str(review)
    assert payload["setup_plan"]["installer_review_verified"] is True
    assert payload["setup_plan"]["installer_integrity_verified"] is True
    assert payload["setup_plan"]["installer_provenance_locked"] is True
    assert payload["operation_results"]["install"]["returncode"] == 0
    install_log = tmp_path / "nemoclaw_install_verified.install.log"
    assert f"verified installer sha256={installer_sha}" in install_log.read_text(encoding="utf-8")
