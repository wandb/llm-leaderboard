import json
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tools" / "check_taiwan_nemoclaw_adoption.py"
INSTALLER_PROVENANCE_NOTE = (
    "Installer integrity is not verified by this script; operator review is required before install/onboard."
)
INSTALLER_LOCK_JSON = "scripts/setup/nemoclaw_installer_lock.json"
INSTALLER_SHA256 = "a4ebc5710dfd8b10035968fd25562773ad56ec4c73ecf9bfe5c78c77724000e7"


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_config(
    path: Path,
    *,
    swe_nemoclaw: bool = False,
    swe_sandbox: str = "nejumi-taiwan",
    use_task_agent: bool = True,
) -> Path:
    swe_extra = f"  nemoclaw_sandbox: {swe_sandbox}\n" if swe_nemoclaw else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
run:
  agentic_math: true
  swebench_pro: true
agentic_math:
  nemoclaw_sandbox: nejumi-taiwan
  use_task_agent: {str(use_task_agent).lower()}
swebench_pro:
  output_dir: outputs/taiwan_full_eval/swebench_pro/test
{swe_extra}
""",
        encoding="utf-8",
    )
    return path


def setup_payload(*, ok: bool) -> dict:
    provider_preflight = {
        "provider": "openai",
        "normalized_provider": "openai",
        "label": "OpenAI hosted inference",
        "reads_dotenv_file": False,
        "will_launch_probe": False,
        "will_launch_benchmark_inference": False,
        "nvidia_hosted_endpoint": False,
        "nvidia_api_key_required": False,
        "credential_required": True,
        "credential_envs": ["OPENAI_API_KEY", "NEMOCLAW_PROVIDER_KEY"],
        "credential_available": False,
        "credential_present_envs": [],
        "endpoint_required": False,
        "endpoint_envs": [],
        "endpoint_available": True,
        "endpoint_present_envs": [],
        "model_required": False,
        "model_configured": False,
        "missing": ["credential"],
        "ready_for_noninteractive_onboard_preflight": False,
        "notes": [
            "This preflight only checks local environment shape and never prints secret values.",
            "It does not prove provider quota or endpoint health; NemoClaw onboarding may still run a provider smoke check.",
        ],
    }
    return {
        "ok": ok,
        "provider": "openai",
        "provider_preflight": provider_preflight,
        "sandbox_configured": ok,
        "host_prerequisites_ok": True,
        "runtime_installed": ok,
        "missing_required_commands": [] if ok else ["nemoclaw", "openshell"],
        "policy_tier": "restricted",
        "policy_tier_allowed_values": ["restricted", "balanced", "open"],
        "policy_tier_valid": True,
        "install_requested": False,
        "onboard_requested": False,
        "accepted_third_party_software": False,
        "commands": {
            "docker": {"available": True, "info_ok": True},
            "nemoclaw": {"available": ok},
            "openshell": {"available": ok},
        },
        "operation_results": {
            "install": {
                "requested": False,
                "attempted": False,
                "returncode": None,
                "log_path": "",
            },
            "onboard": {
                "requested": False,
                "attempted": False,
                "skipped": False,
                "returncode": None,
                "log_path": "",
            },
        },
        "third_party_software": {
            "name": "NVIDIA NemoClaw",
            "vendor": "NVIDIA",
            "repository_url": "https://github.com/NVIDIA/NemoClaw",
            "documentation_url": "https://docs.nvidia.com/nemoclaw/user-guide/openclaw/get-started/quickstart",
            "installer_url": "https://www.nvidia.com/nemoclaw.sh",
            "install_ref": "lkg",
                "installer_sha256": "",
                "installer_signature": "",
                "installer_lock_json": INSTALLER_LOCK_JSON,
                "installer_review_json": "",
                "installer_review_verified": False,
                "installer_integrity_verified": False,
            "installer_provenance_locked": False,
            "installer_provenance_note": INSTALLER_PROVENANCE_NOTE,
            "acceptance_required": True,
            "acceptance_flag": "--yes-i-accept-third-party-software",
            "accepted": False,
            "install_or_onboard_requested": False,
            "operator_review_required_before_install": True,
        },
        "setup_plan": {
            "will_launch_model_inference": False,
            "install_or_onboard_requires_explicit_acceptance": True,
            "acceptance_flag": "--yes-i-accept-third-party-software",
            "provider": "openai",
            "provider_preflight": provider_preflight,
            "sandbox_configured": ok,
            "sandbox_readiness_required": True,
            "third_party_software_name": "NVIDIA NemoClaw",
            "installer_url": "https://www.nvidia.com/nemoclaw.sh",
            "install_ref": "lkg",
                "installer_sha256": "",
                "installer_signature": "",
                "installer_lock_json": INSTALLER_LOCK_JSON,
                "installer_review_json": "",
                "installer_review_verified": False,
                "installer_integrity_verified": False,
            "installer_provenance_locked": False,
            "installer_provenance_note": INSTALLER_PROVENANCE_NOTE,
            "acceptance_ledger_fields": [
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
            ],
                "policy_tier": "restricted",
                "policy_tier_allowed_values": ["restricted", "balanced", "open"],
                "policy_tier_valid": True,
                "installer_review_command": (
                    "uv run python scripts/setup/review_nemoclaw_installer.py "
                    "--url https://www.nvidia.com/nemoclaw.sh "
                    "--install-ref lkg "
                    f"--expected-sha256 {INSTALLER_SHA256} "
                    f"--lock-json {INSTALLER_LOCK_JSON} "
                    "--json temp/nemoclaw_installer_review.json "
                    "--markdown temp/nemoclaw_installer_review.md"
                ),
                "install_command": "install --json temp/install.json",
            "onboard_command": "onboard --policy-tier restricted --json temp/onboard.json",
            "install_and_onboard_command": (
                "scripts/setup/install_nemoclaw.sh --install --onboard "
                f"--installer-lock-json {INSTALLER_LOCK_JSON} "
                "--installer-sha256 REVIEWED_INSTALLER_SHA256 "
                "--installer-review-json temp/nemoclaw_installer_review.json "
                "--policy-tier restricted --yes-i-accept-third-party-software "
                "--json temp/install_onboard.json"
            ),
            "production_install_and_onboard_command": (
                "scripts/setup/install_nemoclaw.sh --install --onboard "
                f"--installer-lock-json {INSTALLER_LOCK_JSON} "
                "--installer-sha256 REVIEWED_INSTALLER_SHA256 "
                "--installer-review-json temp/nemoclaw_installer_review.json "
                "--policy-tier restricted --yes-i-accept-third-party-software "
                "--json temp/install_onboard.json"
            ),
            "post_install_check_command": "check --json temp/check.json",
            "post_install_verification_command": (
                "uv run python scripts/setup/verify_nemoclaw_post_install.py "
                "--json temp/nemoclaw_post_install_verification.json "
                "--markdown temp/nemoclaw_post_install_verification.md "
                "--fail-on-failed"
            ),
            "canary_readiness_command": (
                "uv run python scripts/tools/check_taiwan_canary_readiness.py "
                "--require-nemoclaw --json outputs/taiwan_full_eval/openai_canary_readiness_nemoclaw.json"
            ),
            "adoption_check_command": (
                "uv run python scripts/tools/check_taiwan_nemoclaw_adoption.py "
                "--setup-json temp/nemoclaw_setup_check_YYYYMMDDTHHMM.json "
                "--readiness-json outputs/taiwan_full_eval/openai_canary_readiness_nemoclaw.json "
                "--sandbox nejumi-taiwan "
                "--agentic-config-glob configs/taiwan_full/generated_openai_canary_agentic_nemoclaw/*.yaml "
                "--json temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.json "
                "--markdown temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.md"
            ),
            "production_readiness_command": (
                "uv run python scripts/tools/run_taiwan_production_readiness_gate.py "
                "--report-json outputs/taiwan_full_eval/taiwan_production_readiness_report.json"
            ),
            "operator_sequence": [
                {
                    "step": "setup_check",
                    "command": "check --json temp/check.json",
                    "expected_evidence_path": "temp/nemoclaw_setup_check_YYYYMMDDTHHMM.json",
                    "requires_external_action": False,
                    "required": True,
                },
                {
                    "step": "installer_review",
                    "command": (
                        "uv run python scripts/setup/review_nemoclaw_installer.py "
                        "--url https://www.nvidia.com/nemoclaw.sh "
                        "--install-ref lkg "
                        f"--expected-sha256 {INSTALLER_SHA256} "
                        f"--lock-json {INSTALLER_LOCK_JSON} "
                        "--json temp/nemoclaw_installer_review.json "
                        "--markdown temp/nemoclaw_installer_review.md"
                    ),
                    "expected_evidence_path": "temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json",
                    "requires_external_action": False,
                    "required": True,
                },
                {
                    "step": "install_and_onboard",
                    "command": (
                        "scripts/setup/install_nemoclaw.sh --install --onboard "
                        f"--installer-lock-json {INSTALLER_LOCK_JSON} "
                        "--installer-sha256 REVIEWED_INSTALLER_SHA256 "
                        "--installer-review-json temp/nemoclaw_installer_review.json "
                        "--policy-tier restricted --yes-i-accept-third-party-software "
                        "--json temp/install_onboard.json"
                    ),
                    "expected_evidence_path": "temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.json",
                    "requires_external_action": True,
                    "required": True,
                },
                {
                    "step": "post_install_verification",
                    "command": (
                        "uv run python scripts/setup/verify_nemoclaw_post_install.py "
                        "--json temp/nemoclaw_post_install_verification.json "
                        "--markdown temp/nemoclaw_post_install_verification.md "
                        "--fail-on-failed"
                    ),
                    "expected_evidence_path": "temp/nemoclaw_post_install_verification_YYYYMMDDTHHMM.json",
                    "requires_external_action": False,
                    "required": True,
                },
                {
                    "step": "canary_readiness",
                    "command": (
                        "uv run python scripts/tools/check_taiwan_canary_readiness.py "
                        "--require-nemoclaw --json outputs/taiwan_full_eval/openai_canary_readiness_nemoclaw.json"
                    ),
                    "expected_evidence_path": "outputs/taiwan_full_eval/openai_canary_readiness_nemoclaw.json",
                    "requires_external_action": False,
                    "required": True,
                },
                {
                    "step": "adoption_check",
                    "command": (
                        "uv run python scripts/tools/check_taiwan_nemoclaw_adoption.py "
                        "--setup-json temp/nemoclaw_setup_check_YYYYMMDDTHHMM.json "
                        "--readiness-json outputs/taiwan_full_eval/openai_canary_readiness_nemoclaw.json "
                        "--sandbox nejumi-taiwan "
                        "--agentic-config-glob configs/taiwan_full/generated_openai_canary_agentic_nemoclaw/*.yaml "
                        "--json temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.json "
                        "--markdown temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.md"
                    ),
                    "expected_evidence_path": "temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.json",
                    "requires_external_action": False,
                    "required": True,
                },
                {
                    "step": "production_readiness",
                    "command": (
                        "uv run python scripts/tools/run_taiwan_production_readiness_gate.py "
                        "--report-json outputs/taiwan_full_eval/taiwan_production_readiness_report.json"
                    ),
                    "expected_evidence_path": "outputs/taiwan_full_eval/taiwan_production_readiness_report.json",
                    "requires_external_action": False,
                    "required": True,
                },
            ],
            "expected_evidence_paths": [
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
            ],
        },
    }


def readiness_payload() -> dict:
    return {
        "ok": True,
        "checks": [
            {"name": "NeMoClaw command is available", "ok": True},
            {"name": "OpenShell command is available", "ok": True},
            {"name": "NeMoClaw version command succeeds", "ok": True},
            {"name": "NeMoClaw sandbox status succeeds: nejumi-taiwan", "ok": True},
            {"name": "OpenClaw runs inside NeMoClaw sandbox: nejumi-taiwan", "ok": True},
        ],
    }


def test_nemoclaw_adoption_doctor_reports_not_installed(tmp_path):
    setup = write_json(tmp_path / "setup.json", setup_payload(ok=False))
    config = write_config(tmp_path / "config.yaml")

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--setup-json",
            str(setup),
            "--agentic-config",
            str(config),
            "--fail-on-not-adoptable",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["status"] == "not_installed"
    assert payload["adoption_decision"]["recommendation"] == "conditional_adopt_for_agentic_math"
    assert payload["adoption_decision"]["ready_for_use"] is False
    assert payload["adoption_decision"]["scope"] == "agentic_math_only"
    assert payload["adoption_decision"]["design_ready"] is True
    assert payload["adoption_decision"]["runtime_blockers"] == [
        "setup_installed",
        "sandbox_readiness",
    ]
    assert payload["adoption_decision"]["design_blockers"] == []
    assert payload["ready_for_use"] is False
    assert payload["adoption_recommendation"] == "conditional_adopt_for_agentic_math"
    assert payload["adoption_scope"] == "agentic_math_only"
    assert payload["design_ready"] is True
    assert payload["blockers"] == ["setup_installed", "sandbox_readiness"]
    assert payload["runtime_blockers"] == ["setup_installed", "sandbox_readiness"]
    assert payload["design_blockers"] == []
    assert payload["other_blockers"] == []
    assert payload["setup_runtime"] == {
        "host_prerequisites_ok": True,
        "runtime_installed": False,
        "sandbox_configured": False,
        "provider": "openai",
        "provider_preflight": setup_payload(ok=False)["provider_preflight"],
        "latest_onboard_failure": {},
        "missing_required_commands": ["nemoclaw", "openshell"],
        "missing_components": ["nemoclaw", "openshell"],
    }
    assert payload["missing_required_commands"] == ["nemoclaw", "openshell"]
    assert payload["missing_components"] == ["nemoclaw", "openshell"]
    assert payload["operator_handoff"]["available"] is True
    assert payload["operator_handoff"]["source_setup_json"] == str(setup)
    assert payload["operator_handoff"]["step_count"] == 7
    assert payload["operator_handoff"]["required_step_count"] == 7
    assert payload["operator_handoff"]["external_action_step_count"] == 1
    assert payload["operator_handoff"]["evidence_path_count"] == 12
    assert "setup_installed" in payload["summary"]["blockers"]
    assert payload["summary"]["adoption_recommendation"] == "conditional_adopt_for_agentic_math"
    assert payload["summary"]["ready_for_use"] is False
    assert payload["summary"]["setup_runtime"] == {
        "host_prerequisites_ok": True,
        "runtime_installed": False,
        "sandbox_configured": False,
        "provider": "openai",
        "provider_preflight": setup_payload(ok=False)["provider_preflight"],
        "latest_onboard_failure": {},
        "missing_required_commands": ["nemoclaw", "openshell"],
        "missing_components": ["nemoclaw", "openshell"],
    }
    assert payload["summary"]["operator_handoff"] == {
        "available": True,
        "source_setup_json": str(setup),
        "step_count": 7,
        "required_step_count": 7,
        "external_action_step_count": 1,
        "evidence_path_count": 12,
    }
    assert payload["criteria"][0]["name"] == "setup_plan_safety"
    assert payload["criteria"][0]["ok"] is True
    setup_installed = next(row for row in payload["criteria"] if row["name"] == "setup_installed")
    assert setup_installed["host_prerequisites_ok"] is True
    assert setup_installed["runtime_installed"] is False
    assert setup_installed["sandbox_configured"] is False
    assert setup_installed["provider"] == "openai"
    assert setup_installed["provider_preflight"]["nvidia_api_key_required"] is False
    assert setup_installed["missing_required_commands"] == ["nemoclaw", "openshell"]


def test_nemoclaw_adoption_doctor_rejects_installer_review_without_expected_sha(tmp_path):
    legacy = setup_payload(ok=True)
    command = legacy["setup_plan"]["installer_review_command"]
    command = command.replace(f"--expected-sha256 {INSTALLER_SHA256} ", "")
    legacy["setup_plan"]["installer_review_command"] = command
    for row in legacy["setup_plan"]["operator_sequence"]:
        if row["step"] == "installer_review":
            row["command"] = command
    setup = write_json(tmp_path / "setup.json", legacy)
    readiness = write_json(tmp_path / "readiness.json", readiness_payload())
    config = write_config(tmp_path / "config.yaml")

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--setup-json",
            str(setup),
            "--readiness-json",
            str(readiness),
            "--agentic-config",
            str(config),
            "--fail-on-not-adoptable",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    setup_plan = payload["criteria"][0]
    assert setup_plan["status"] == "invalid_setup_plan"
    assert "--expected-sha256" in setup_plan["missing_installer_review_command_markers"]
    assert (
        "installer_review_command must include --expected-sha256"
        in setup_plan["installer_review_command_errors"]
    )


def test_nemoclaw_adoption_doctor_rejects_post_install_command_without_fail_flag(
    tmp_path,
):
    legacy = setup_payload(ok=True)
    command = legacy["setup_plan"]["post_install_verification_command"].replace(
        " --fail-on-failed",
        "",
    )
    legacy["setup_plan"]["post_install_verification_command"] = command
    for row in legacy["setup_plan"]["operator_sequence"]:
        if row["step"] == "post_install_verification":
            row["command"] = command
    setup = write_json(tmp_path / "setup.json", legacy)
    readiness = write_json(tmp_path / "readiness.json", readiness_payload())
    config = write_config(tmp_path / "config.yaml")

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--setup-json",
            str(setup),
            "--readiness-json",
            str(readiness),
            "--agentic-config",
            str(config),
            "--fail-on-not-adoptable",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    setup_plan = payload["criteria"][0]
    assert setup_plan["status"] == "invalid_setup_plan"
    assert "--fail-on-failed" in setup_plan[
        "missing_post_install_verification_command_markers"
    ]


def test_nemoclaw_adoption_doctor_rejects_legacy_setup_plan(tmp_path):
    legacy = setup_payload(ok=True)
    legacy["setup_plan"].pop("post_install_verification_command")
    legacy["setup_plan"].pop("canary_readiness_command")
    legacy["setup_plan"].pop("adoption_check_command")
    legacy["setup_plan"].pop("install_and_onboard_command")
    legacy["setup_plan"].pop("production_install_and_onboard_command")
    setup = write_json(tmp_path / "setup.json", legacy)
    readiness = write_json(tmp_path / "readiness.json", readiness_payload())
    config = write_config(tmp_path / "config.yaml")

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--setup-json",
            str(setup),
            "--readiness-json",
            str(readiness),
            "--agentic-config",
            str(config),
            "--fail-on-not-adoptable",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "not_adoptable"
    assert payload["adoption_decision"]["recommendation"] == "do_not_adopt_until_remediated"
    assert payload["adoption_decision"]["ready_for_use"] is False
    assert payload["adoption_decision"]["design_blockers"] == ["setup_plan_safety"]
    assert "setup_plan_safety" in payload["summary"]["blockers"]
    setup_plan = payload["criteria"][0]
    assert setup_plan["status"] == "invalid_setup_plan"
    assert setup_plan["missing_setup_plan_fields"] == [
        "install_and_onboard_command",
        "production_install_and_onboard_command",
        "post_install_verification_command",
        "canary_readiness_command",
        "adoption_check_command",
    ]
    assert setup_plan["missing_evidence_output_fields"] == []
    assert setup_plan["missing_operation_result_fields"] == []
    assert setup_plan["missing_third_party_software_fields"] == []
    assert setup_plan["missing_acceptance_ledger_fields"] == []


def test_nemoclaw_adoption_doctor_rejects_legacy_setup_without_operation_results(tmp_path):
    legacy = setup_payload(ok=True)
    legacy.pop("operation_results")
    setup = write_json(tmp_path / "setup.json", legacy)
    readiness = write_json(tmp_path / "readiness.json", readiness_payload())
    config = write_config(tmp_path / "config.yaml")

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--setup-json",
            str(setup),
            "--readiness-json",
            str(readiness),
            "--agentic-config",
            str(config),
            "--fail-on-not-adoptable",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "not_adoptable"
    assert payload["adoption_decision"]["recommendation"] == "do_not_adopt_until_remediated"
    assert payload["adoption_decision"]["ready_for_use"] is False
    assert payload["adoption_decision"]["design_blockers"] == ["setup_plan_safety"]
    assert "setup_plan_safety" in payload["summary"]["blockers"]
    setup_plan = payload["criteria"][0]
    assert setup_plan["status"] == "invalid_setup_plan"
    assert setup_plan["missing_setup_plan_fields"] == []
    assert setup_plan["missing_evidence_output_fields"] == []
    assert setup_plan["missing_operation_result_fields"] == ["operation_results"]
    assert setup_plan["missing_third_party_software_fields"] == []
    assert setup_plan["missing_acceptance_ledger_fields"] == []


def test_nemoclaw_adoption_doctor_rejects_setup_without_provider_preflight(tmp_path):
    legacy = setup_payload(ok=True)
    legacy.pop("provider_preflight")
    legacy.pop("sandbox_configured")
    legacy["setup_plan"].pop("provider_preflight")
    legacy["setup_plan"].pop("sandbox_configured")
    legacy["setup_plan"].pop("sandbox_readiness_required")
    setup = write_json(tmp_path / "setup.json", legacy)
    readiness = write_json(tmp_path / "readiness.json", readiness_payload())
    config = write_config(tmp_path / "config.yaml")

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--setup-json",
            str(setup),
            "--readiness-json",
            str(readiness),
            "--agentic-config",
            str(config),
            "--fail-on-not-adoptable",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    setup_plan = payload["criteria"][0]
    assert setup_plan["status"] == "invalid_setup_plan"
    assert setup_plan["missing_payload_fields"] == [
        "provider_preflight",
        "sandbox_configured",
    ]
    assert setup_plan["missing_setup_plan_fields"] == [
        "provider_preflight",
        "sandbox_configured",
        "sandbox_readiness_required",
    ]


def test_nemoclaw_adoption_doctor_reports_provider_quota_failure(tmp_path):
    setup_payload_with_failure = setup_payload(ok=True)
    failure = {
        "failure_kind": "provider_quota",
        "failure_detail": "provider validation returned HTTP 429/quota",
    }
    setup_payload_with_failure["operation_results"]["onboard"] = {
        "requested": True,
        "attempted": True,
        "skipped": False,
        "returncode": 1,
        "log_path": "temp/nemoclaw_onboard.log",
        "failure": failure,
    }
    setup_payload_with_failure["onboard_requested"] = True
    setup_payload_with_failure["accepted_third_party_software"] = True
    setup_payload_with_failure["third_party_software"]["accepted"] = True
    setup_payload_with_failure["third_party_software"]["install_or_onboard_requested"] = True
    setup_payload_with_failure["third_party_software"]["installer_review_verified"] = True
    setup_payload_with_failure["third_party_software"]["installer_integrity_verified"] = True
    setup_payload_with_failure["third_party_software"]["installer_provenance_locked"] = True
    setup_payload_with_failure["third_party_software"]["installer_sha256"] = INSTALLER_SHA256
    setup_payload_with_failure["third_party_software"]["installer_review_json"] = (
        "temp/nemoclaw_installer_review.json"
    )
    setup_payload_with_failure["setup_plan"]["installer_review_verified"] = True
    setup_payload_with_failure["setup_plan"]["installer_integrity_verified"] = True
    setup_payload_with_failure["setup_plan"]["installer_provenance_locked"] = True
    setup_payload_with_failure["setup_plan"]["installer_sha256"] = INSTALLER_SHA256
    setup_payload_with_failure["setup_plan"]["installer_review_json"] = (
        "temp/nemoclaw_installer_review.json"
    )
    setup = write_json(tmp_path / "setup.json", setup_payload_with_failure)
    config = write_config(tmp_path / "config.yaml")

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--setup-json",
            str(setup),
            "--agentic-config",
            str(config),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    setup_plan = payload["criteria"][0]
    assert setup_plan["latest_onboard_failure"] == failure
    assert payload["setup_runtime"]["latest_onboard_failure"] == failure


def test_nemoclaw_adoption_doctor_rejects_setup_without_operator_sequence(tmp_path):
    legacy = setup_payload(ok=True)
    legacy["setup_plan"].pop("operator_sequence")
    legacy["setup_plan"].pop("expected_evidence_paths")
    setup = write_json(tmp_path / "setup.json", legacy)
    readiness = write_json(tmp_path / "readiness.json", readiness_payload())
    config = write_config(tmp_path / "config.yaml")

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--setup-json",
            str(setup),
            "--readiness-json",
            str(readiness),
            "--agentic-config",
            str(config),
            "--fail-on-not-adoptable",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    setup_plan = payload["criteria"][0]
    assert setup_plan["status"] == "invalid_setup_plan"
    assert setup_plan["missing_setup_plan_fields"] == [
        "operator_sequence",
        "expected_evidence_paths",
    ]
    assert setup_plan["missing_operator_sequence_steps"] == [
        "setup_check",
        "installer_review",
        "install_and_onboard",
        "post_install_verification",
        "canary_readiness",
        "adoption_check",
        "production_readiness",
    ]
    assert setup_plan["missing_operator_evidence_paths"] == [
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


def test_nemoclaw_adoption_doctor_rejects_setup_without_third_party_metadata(tmp_path):
    legacy = setup_payload(ok=True)
    legacy.pop("third_party_software")
    legacy["setup_plan"].pop("third_party_software_name")
    legacy["setup_plan"].pop("installer_url")
    legacy["setup_plan"].pop("install_ref")
    legacy["setup_plan"].pop("acceptance_ledger_fields")
    setup = write_json(tmp_path / "setup.json", legacy)
    readiness = write_json(tmp_path / "readiness.json", readiness_payload())
    config = write_config(tmp_path / "config.yaml")

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--setup-json",
            str(setup),
            "--readiness-json",
            str(readiness),
            "--agentic-config",
            str(config),
            "--fail-on-not-adoptable",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "not_adoptable"
    assert payload["adoption_decision"]["recommendation"] == "do_not_adopt_until_remediated"
    setup_plan = payload["criteria"][0]
    assert setup_plan["status"] == "invalid_setup_plan"
    assert setup_plan["missing_setup_plan_fields"] == [
        "third_party_software_name",
        "installer_url",
        "install_ref",
        "acceptance_ledger_fields",
    ]
    assert setup_plan["missing_third_party_software_fields"] == [
        "name",
        "vendor",
        "repository_url",
        "documentation_url",
        "installer_url",
            "install_ref",
            "installer_sha256",
            "installer_signature",
            "installer_lock_json",
            "installer_review_json",
            "installer_review_verified",
            "installer_integrity_verified",
        "installer_provenance_locked",
        "installer_provenance_note",
        "acceptance_required",
        "acceptance_flag",
        "accepted",
        "install_or_onboard_requested",
        "operator_review_required_before_install",
    ]
    assert setup_plan["missing_acceptance_ledger_fields"] == [
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


def test_nemoclaw_adoption_doctor_rejects_setup_without_installer_lock_json(tmp_path):
    legacy = setup_payload(ok=True)
    legacy["third_party_software"].pop("installer_lock_json")
    legacy["setup_plan"].pop("installer_lock_json")
    legacy["setup_plan"]["acceptance_ledger_fields"].remove(
        "third_party_software.installer_lock_json"
    )
    legacy["setup_plan"]["acceptance_ledger_fields"].remove("setup_plan.installer_lock_json")
    setup = write_json(tmp_path / "setup.json", legacy)
    readiness = write_json(tmp_path / "readiness.json", readiness_payload())
    config = write_config(tmp_path / "config.yaml")

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--setup-json",
            str(setup),
            "--readiness-json",
            str(readiness),
            "--agentic-config",
            str(config),
            "--fail-on-not-adoptable",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    setup_plan = payload["criteria"][0]
    assert setup_plan["status"] == "invalid_setup_plan"
    assert setup_plan["missing_setup_plan_fields"] == ["installer_lock_json"]
    assert setup_plan["missing_third_party_software_fields"] == ["installer_lock_json"]
    assert setup_plan["missing_acceptance_ledger_fields"] == [
        "third_party_software.installer_lock_json",
        "setup_plan.installer_lock_json",
    ]


def test_nemoclaw_adoption_doctor_rejects_unexpected_installer_lock_json(tmp_path):
    legacy = setup_payload(ok=True)
    legacy["third_party_software"]["installer_lock_json"] = "scripts/setup/other_lock.json"
    legacy["setup_plan"]["installer_lock_json"] = "scripts/setup/other_lock.json"
    setup = write_json(tmp_path / "setup.json", legacy)
    readiness = write_json(tmp_path / "readiness.json", readiness_payload())
    config = write_config(tmp_path / "config.yaml")

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--setup-json",
            str(setup),
            "--readiness-json",
            str(readiness),
            "--agentic-config",
            str(config),
            "--fail-on-not-adoptable",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    setup_plan = payload["criteria"][0]
    assert setup_plan["status"] == "invalid_setup_plan"
    assert setup_plan["missing_setup_plan_fields"] == []
    assert setup_plan["missing_third_party_software_fields"] == []
    assert setup_plan["missing_acceptance_ledger_fields"] == []
    assert setup_plan["installer_provenance_consistency_errors"] == [
        "third_party_software.installer_lock_json must be scripts/setup/nemoclaw_installer_lock.json",
        "setup_plan.installer_lock_json must be scripts/setup/nemoclaw_installer_lock.json",
    ]


def test_nemoclaw_adoption_doctor_rejects_requested_install_without_acceptance(tmp_path):
    unsafe = setup_payload(ok=True)
    unsafe["install_requested"] = True
    unsafe["third_party_software"]["install_or_onboard_requested"] = True
    unsafe["operation_results"]["install"] = {
        "requested": True,
        "attempted": True,
        "returncode": 2,
        "log_path": "temp/nemoclaw_install.log",
    }
    setup = write_json(tmp_path / "setup.json", unsafe)
    readiness = write_json(tmp_path / "readiness.json", readiness_payload())
    config = write_config(tmp_path / "config.yaml")

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--setup-json",
            str(setup),
            "--readiness-json",
            str(readiness),
            "--agentic-config",
            str(config),
            "--fail-on-not-adoptable",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    setup_plan = payload["criteria"][0]
    assert setup_plan["status"] == "invalid_setup_plan"
    assert setup_plan["operation_result_consistency_errors"] == []
    assert setup_plan["acceptance_consistency_errors"] == [
        "install/onboard requested or attempted without accepted_third_party_software=true",
        "install/onboard requested or attempted without third_party_software.accepted=true",
    ]


def test_nemoclaw_adoption_doctor_rejects_attempted_operation_without_log_or_returncode(tmp_path):
    unsafe = setup_payload(ok=True)
    unsafe["accepted_third_party_software"] = True
    unsafe["install_requested"] = True
    unsafe["third_party_software"]["accepted"] = True
    unsafe["third_party_software"]["install_or_onboard_requested"] = True
    unsafe["operation_results"]["install"] = {
        "requested": True,
        "attempted": True,
        "returncode": None,
        "log_path": "",
    }
    setup = write_json(tmp_path / "setup.json", unsafe)
    readiness = write_json(tmp_path / "readiness.json", readiness_payload())
    config = write_config(tmp_path / "config.yaml")

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--setup-json",
            str(setup),
            "--readiness-json",
            str(readiness),
            "--agentic-config",
            str(config),
            "--fail-on-not-adoptable",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    setup_plan = payload["criteria"][0]
    assert setup_plan["status"] == "invalid_setup_plan"
    assert setup_plan["acceptance_consistency_errors"] == []
    assert setup_plan["operation_result_consistency_errors"] == [
        "operation_results.install.attempted requires non-empty log_path",
        "operation_results.install.attempted requires integer returncode",
    ]


def test_nemoclaw_adoption_doctor_rejects_requested_install_without_locked_installer_provenance(tmp_path):
    unsafe = setup_payload(ok=True)
    unsafe["accepted_third_party_software"] = True
    unsafe["install_requested"] = True
    unsafe["third_party_software"]["accepted"] = True
    unsafe["third_party_software"]["install_or_onboard_requested"] = True
    unsafe["operation_results"]["install"] = {
        "requested": True,
        "attempted": True,
        "returncode": 0,
        "log_path": "temp/nemoclaw_install.log",
    }
    setup = write_json(tmp_path / "setup.json", unsafe)
    readiness = write_json(tmp_path / "readiness.json", readiness_payload())
    config = write_config(tmp_path / "config.yaml")

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--setup-json",
            str(setup),
            "--readiness-json",
            str(readiness),
            "--agentic-config",
            str(config),
            "--fail-on-not-adoptable",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    setup_plan = payload["criteria"][0]
    assert setup_plan["status"] == "invalid_setup_plan"
    assert setup_plan["acceptance_consistency_errors"] == []
    assert setup_plan["operation_result_consistency_errors"] == []
    assert setup_plan["installer_provenance_consistency_errors"] == [
        "install/onboard requested or attempted without third_party_software.installer_review_verified=true",
        "install/onboard requested or attempted without third_party_software.installer_integrity_verified=true",
        "install/onboard requested or attempted without third_party_software.installer_provenance_locked=true",
        "install/onboard requested or attempted without setup_plan.installer_review_verified=true",
        "install/onboard requested or attempted without setup_plan.installer_integrity_verified=true",
        "install/onboard requested or attempted without setup_plan.installer_provenance_locked=true",
    ]


def test_nemoclaw_adoption_doctor_rejects_setup_commands_without_evidence_outputs(tmp_path):
    legacy = setup_payload(ok=True)
    legacy["setup_plan"]["install_command"] = "install"
    legacy["setup_plan"]["onboard_command"] = "onboard"
    legacy["setup_plan"]["install_and_onboard_command"] = "install --onboard"
    legacy["setup_plan"]["production_install_and_onboard_command"] = "install --onboard"
    legacy["setup_plan"]["post_install_check_command"] = "check"
    legacy["setup_plan"]["post_install_verification_command"] = "verify"
    legacy["setup_plan"]["canary_readiness_command"] = "canary"
    legacy["setup_plan"]["production_readiness_command"] = "production"
    setup = write_json(tmp_path / "setup.json", legacy)
    readiness = write_json(tmp_path / "readiness.json", readiness_payload())
    config = write_config(tmp_path / "config.yaml")

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--setup-json",
            str(setup),
            "--readiness-json",
            str(readiness),
            "--agentic-config",
            str(config),
            "--fail-on-not-adoptable",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "not_adoptable"
    setup_plan = payload["criteria"][0]
    assert setup_plan["status"] == "invalid_setup_plan"
    assert setup_plan["missing_setup_plan_fields"] == []
    assert setup_plan["missing_evidence_output_fields"] == [
        "install_command",
        "onboard_command",
        "install_and_onboard_command",
        "production_install_and_onboard_command",
        "post_install_check_command",
        "post_install_verification_command",
        "canary_readiness_command",
        "production_readiness_command",
    ]
    assert setup_plan["missing_restricted_policy_command_fields"] == [
        "onboard_command",
        "install_and_onboard_command",
        "production_install_and_onboard_command",
    ]
    assert setup_plan["missing_production_install_and_onboard_markers"] == [
        "scripts/setup/install_nemoclaw.sh",
        "--install",
        "--installer-lock-json",
        "--installer-sha256",
        "--installer-review-json",
        "--yes-i-accept-third-party-software",
        "--policy-tier restricted",
        "--json",
    ]


def test_nemoclaw_adoption_doctor_rejects_mismatched_production_install_command(tmp_path):
    payload = setup_payload(ok=True)
    payload["setup_plan"][
        "production_install_and_onboard_command"
    ] = payload["setup_plan"]["production_install_and_onboard_command"].replace(
        "--json temp/install_onboard.json",
        "--json temp/other_install_onboard.json",
    )
    setup = write_json(tmp_path / "setup.json", payload)
    readiness = write_json(tmp_path / "readiness.json", readiness_payload())
    config = write_config(tmp_path / "config.yaml")

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--setup-json",
            str(setup),
            "--readiness-json",
            str(readiness),
            "--agentic-config",
            str(config),
            "--fail-on-not-adoptable",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    result_payload = json.loads(result.stdout)
    setup_plan = result_payload["criteria"][0]
    assert setup_plan["status"] == "invalid_setup_plan"
    assert setup_plan["missing_production_install_and_onboard_markers"] == []
    assert setup_plan["production_install_command_consistency_errors"] == [
        "production_install_and_onboard_command must match install_and_onboard_command"
    ]


def test_nemoclaw_adoption_doctor_rejects_non_restricted_policy_tier(tmp_path):
    setup_payload_balanced = setup_payload(ok=True)
    setup_payload_balanced["policy_tier"] = "balanced"
    setup_payload_balanced["setup_plan"]["policy_tier"] = "balanced"
    setup_payload_balanced["setup_plan"][
        "onboard_command"
    ] = "onboard --policy-tier balanced --json temp/onboard.json"
    setup_payload_balanced["setup_plan"][
        "install_and_onboard_command"
    ] = "install --onboard --policy-tier balanced --json temp/install_onboard.json"
    setup_payload_balanced["setup_plan"][
        "production_install_and_onboard_command"
    ] = "install --onboard --policy-tier balanced --json temp/install_onboard.json"
    setup = write_json(tmp_path / "setup.json", setup_payload_balanced)
    readiness = write_json(tmp_path / "readiness.json", readiness_payload())
    config = write_config(tmp_path / "config.yaml")

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--setup-json",
            str(setup),
            "--readiness-json",
            str(readiness),
            "--agentic-config",
            str(config),
            "--fail-on-not-adoptable",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "not_adoptable"
    setup_plan = payload["criteria"][0]
    assert setup_plan["status"] == "invalid_setup_plan"
    assert setup_plan["invalid_policy_fields"] == [
        "policy_tier",
        "setup_plan.policy_tier",
    ]
    assert setup_plan["missing_restricted_policy_command_fields"] == [
        "onboard_command",
        "install_and_onboard_command",
        "production_install_and_onboard_command",
    ]


def test_nemoclaw_adoption_doctor_rejects_unknown_policy_tier_metadata(tmp_path):
    setup_payload_unknown = setup_payload(ok=True)
    setup_payload_unknown["policy_tier"] = "experimental"
    setup_payload_unknown["policy_tier_valid"] = False
    setup_payload_unknown["setup_plan"]["policy_tier"] = "experimental"
    setup_payload_unknown["setup_plan"]["policy_tier_valid"] = False
    setup_payload_unknown["setup_plan"][
        "onboard_command"
    ] = "onboard --policy-tier experimental --json temp/onboard.json"
    setup_payload_unknown["setup_plan"][
        "install_and_onboard_command"
    ] = "install --onboard --policy-tier experimental --json temp/install_onboard.json"
    setup_payload_unknown["setup_plan"][
        "production_install_and_onboard_command"
    ] = "install --onboard --policy-tier experimental --json temp/install_onboard.json"
    setup = write_json(tmp_path / "setup.json", setup_payload_unknown)
    readiness = write_json(tmp_path / "readiness.json", readiness_payload())
    config = write_config(tmp_path / "config.yaml")

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--setup-json",
            str(setup),
            "--readiness-json",
            str(readiness),
            "--agentic-config",
            str(config),
            "--fail-on-not-adoptable",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    setup_plan = payload["criteria"][0]
    assert setup_plan["status"] == "invalid_setup_plan"
    for field in (
        "policy_tier",
        "policy_tier_valid",
        "setup_plan.policy_tier",
        "setup_plan.policy_tier_valid",
    ):
        assert field in setup_plan["invalid_policy_fields"]
    assert setup_plan["missing_restricted_policy_command_fields"] == [
        "onboard_command",
        "install_and_onboard_command",
        "production_install_and_onboard_command",
    ]


def test_nemoclaw_adoption_doctor_accepts_agentic_math_only_config(tmp_path):
    setup = write_json(tmp_path / "setup.json", setup_payload(ok=True))
    readiness = write_json(tmp_path / "readiness.json", readiness_payload())
    config = write_config(tmp_path / "config.yaml")
    output_json = tmp_path / "adoption.json"
    output_md = tmp_path / "adoption.md"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--setup-json",
            str(setup),
            "--readiness-json",
            str(readiness),
            "--agentic-config",
            str(config),
            "--json",
            str(output_json),
            "--markdown",
            str(output_md),
            "--fail-on-not-adoptable",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["path"] == str(output_json)
    assert payload["markdown_path"] == str(output_md)
    assert payload["ok"] is True
    assert payload["status"] == "adoptable_for_agentic_math"
    assert payload["adoption_decision"]["recommendation"] == "adopt_for_agentic_math"
    assert payload["adoption_decision"]["ready_for_use"] is True
    assert payload["adoption_decision"]["runtime_blockers"] == []
    assert payload["adoption_decision"]["design_blockers"] == []
    assert payload["ready_for_use"] is True
    assert payload["adoption_recommendation"] == "adopt_for_agentic_math"
    assert payload["adoption_scope"] == "agentic_math_only"
    assert payload["design_ready"] is True
    assert payload["blockers"] == []
    assert payload["runtime_blockers"] == []
    assert payload["design_blockers"] == []
    assert payload["other_blockers"] == []
    assert payload["setup_runtime"] == {
        "host_prerequisites_ok": True,
        "runtime_installed": True,
        "sandbox_configured": True,
        "provider": "openai",
        "provider_preflight": setup_payload(ok=True)["provider_preflight"],
        "latest_onboard_failure": {},
        "missing_required_commands": [],
        "missing_components": [],
    }
    assert payload["missing_required_commands"] == []
    assert payload["missing_components"] == []
    assert payload["operator_handoff"]["available"] is True
    assert payload["operator_handoff"]["source_setup_json"] == str(setup)
    assert payload["operator_handoff"]["step_count"] == 7
    assert payload["operator_handoff"]["required_step_count"] == 7
    assert payload["operator_handoff"]["external_action_step_count"] == 1
    assert payload["operator_handoff"]["evidence_path_count"] == 12
    assert "--install --onboard" in payload["operator_handoff"][
        "production_install_and_onboard_command"
    ]
    assert payload["summary"]["blockers"] == []
    assert payload["summary"]["adoption_recommendation"] == "adopt_for_agentic_math"
    assert payload["summary"]["ready_for_use"] is True
    assert payload["summary"]["setup_runtime"] == {
        "host_prerequisites_ok": True,
        "runtime_installed": True,
        "sandbox_configured": True,
        "provider": "openai",
        "provider_preflight": setup_payload(ok=True)["provider_preflight"],
        "latest_onboard_failure": {},
        "missing_required_commands": [],
        "missing_components": [],
    }
    assert payload["summary"]["operator_handoff"] == {
        "available": True,
        "source_setup_json": str(setup),
        "step_count": 7,
        "required_step_count": 7,
        "external_action_step_count": 1,
        "evidence_path_count": 12,
    }
    output_payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert output_payload["ok"] is True
    assert output_payload["path"] == str(output_json)
    assert output_payload["markdown_path"] == str(output_md)
    markdown = output_md.read_text(encoding="utf-8")
    assert "Operator Handoff" in markdown
    assert "swebench_pro_non_adoption_guard" in markdown


def test_nemoclaw_adoption_doctor_accepts_agentic_config_glob(tmp_path):
    setup = write_json(tmp_path / "setup.json", setup_payload(ok=True))
    readiness = write_json(tmp_path / "readiness.json", readiness_payload())
    config = write_config(tmp_path / "configs" / "config.yaml")

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--setup-json",
            str(setup),
            "--readiness-json",
            str(readiness),
            "--agentic-config-glob",
            str(tmp_path / "configs" / "*.yaml"),
            "--fail-on-not-adoptable",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["agentic_config_paths"] == [str(config)]


def test_nemoclaw_adoption_doctor_rejects_disabled_task_agent(tmp_path):
    setup = write_json(tmp_path / "setup.json", setup_payload(ok=True))
    readiness = write_json(tmp_path / "readiness.json", readiness_payload())
    config = write_config(tmp_path / "config.yaml", use_task_agent=False)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--setup-json",
            str(setup),
            "--readiness-json",
            str(readiness),
            "--agentic-config",
            str(config),
            "--fail-on-not-adoptable",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["adoption_decision"]["design_blockers"] == ["agentic_math_config"]
    agentic_gate = next(row for row in payload["criteria"] if row["name"] == "agentic_math_config")
    assert agentic_gate["status"] == "missing_nemoclaw_agentic_math_config"
    assert agentic_gate["records"][0]["agentic_math_use_task_agent"] is False


def test_nemoclaw_adoption_doctor_accepts_swebench_nemoclaw_config_from_second_glob(tmp_path):
    setup = write_json(tmp_path / "setup.json", setup_payload(ok=True))
    readiness = write_json(tmp_path / "readiness.json", readiness_payload())
    clean_agentic_config = write_config(tmp_path / "generated_agentic" / "config.yaml")
    swe_full_config = write_config(
        tmp_path / "generated" / "config.yaml",
        swe_nemoclaw=True,
    )

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--setup-json",
            str(setup),
            "--readiness-json",
            str(readiness),
            "--agentic-config-glob",
            str(tmp_path / "generated_agentic" / "*.yaml"),
            "--agentic-config-glob",
            str(tmp_path / "generated" / "*.yaml"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert clean_agentic_config.exists()
    assert swe_full_config.exists()
    assert payload["adoption_decision"]["design_blockers"] == []
    assert payload["adoption_decision"]["recommendation"] == "adopt_for_agentic_benchmarks"
    assert payload["adoption_decision"]["scope"] == "agentic_math_and_swebench_pro"
    swebench_guard = next(
        row
        for row in payload["criteria"]
        if row["name"] == "swebench_pro_non_adoption_guard"
    )
    assert swebench_guard["status"] == "passed"
    assert swebench_guard["swebench_pro_nemoclaw_ready"] is True
    assert swebench_guard["passing_records"][0]["path"] == str(swe_full_config)


def test_nemoclaw_adoption_doctor_accepts_swebench_nemoclaw_config(tmp_path):
    setup = write_json(tmp_path / "setup.json", setup_payload(ok=True))
    readiness = write_json(tmp_path / "readiness.json", readiness_payload())
    config = write_config(tmp_path / "config.yaml", swe_nemoclaw=True)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--setup-json",
            str(setup),
            "--readiness-json",
            str(readiness),
            "--agentic-config",
            str(config),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["status"] == "adoptable_for_agentic_benchmarks"
    assert payload["adoption_decision"]["recommendation"] == "adopt_for_agentic_benchmarks"
    assert payload["adoption_decision"]["scope"] == "agentic_math_and_swebench_pro"
    assert payload["adoption_decision"]["design_blockers"] == []


def test_nemoclaw_adoption_doctor_rejects_mismatched_swebench_nemoclaw_sandbox(tmp_path):
    setup = write_json(tmp_path / "setup.json", setup_payload(ok=True))
    readiness = write_json(tmp_path / "readiness.json", readiness_payload())
    config = write_config(tmp_path / "config.yaml", swe_nemoclaw=True, swe_sandbox="other-sandbox")

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--setup-json",
            str(setup),
            "--readiness-json",
            str(readiness),
            "--agentic-config",
            str(config),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["adoption_decision"]["recommendation"] == "do_not_adopt_until_remediated"
    assert payload["adoption_decision"]["design_blockers"] == ["swebench_pro_non_adoption_guard"]
    swebench_guard = next(
        row
        for row in payload["criteria"]
        if row["name"] == "swebench_pro_non_adoption_guard"
    )
    assert swebench_guard["status"] == "swebench_pro_invalid_nemoclaw_config"
    assert swebench_guard["offending_records"][0]["swebench_pro_nemoclaw_sandbox"] == "other-sandbox"
