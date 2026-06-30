import importlib.util
import hashlib
import json
import os
import sys
import time
from argparse import Namespace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PAID_REVIEW_TEMPLATE = REPO_ROOT / "docs" / "taiwan_paid_run_review_template.md"
NEMOCLAW_AGENTIC_CONFIG = (
    "taiwan_full/generated_openai_canary_agentic_nemoclaw/"
    "config-taiwan-full-gpt-4_1-mini-openai-direct-canary.yaml"
)
NON_NEMOCLAW_AGENTIC_CONFIG = (
    "taiwan_full/generated_openai_canary_agentic/"
    "config-taiwan-full-gpt-4_1-mini-openai-direct-canary.yaml"
)


def load_module():
    path = REPO_ROOT / "scripts" / "tools" / "build_taiwan_production_readiness_report.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_eval_preflight_payload(path: Path | None = None):
    if path is None:
        path = REPO_ROOT / "temp" / "test_taiwan_run_eval_preflight.json"
    return write_json(
        path,
        {
            "schema_version": 1,
            "generated_at": time.time(),
            "status": "passed",
            "ok": True,
            "config": "configs/taiwan_full/generated/config-taiwan-full-test.yaml",
            "base_config": "configs/base_config_taiwan.yaml",
            "wandb": {
                "entity": "llm-leaderboard",
                "project": "tc-leaderboard",
                "run_name": "taiwan/full/test",
            },
            "api": "openai_responses",
            "model": "gpt-4.1-mini-2025-04-14",
            "enabled_benchmarks": ["agentic_math", "swebench_pro"],
            "will_initialize_wandb": False,
            "will_log_wandb_artifacts": False,
            "will_initialize_weave": False,
            "will_start_inference_engine": False,
            "will_run_evaluators": False,
            "token_validation": {"ok": True, "has_errors": False, "results": []},
        },
    )


def add_wandb_run_metadata(payload, *, benchmark="agentic_math", run_id="run-1"):
    required = payload.setdefault("required_evidence", {})
    required["run_metadata"] = {
        "config": [
            {"key": "model.pretrained_model_name_or_path", "expected": "gpt-4.1-mini-2025-04-14"},
            {"key": f"run.{benchmark}", "expected": True},
            {"key": "wandb.run_name", "expected": f"taiwan/full/openai/gpt-4.1-mini: {run_id}"},
        ],
        "tags": ["taiwan-canary"],
        "group": "tw-canary",
        "job_type": "evaluation",
    }
    observed = payload.setdefault("observed_evidence", {})
    observed["run_metadata"] = {
        "config": [
            {
                "key": "model.pretrained_model_name_or_path",
                "present": True,
                "value": "gpt-4.1-mini-2025-04-14",
            },
            {"key": f"run.{benchmark}", "present": True, "value": True},
            {
                "key": "wandb.run_name",
                "present": True,
                "value": f"taiwan/full/openai/gpt-4.1-mini: {run_id}",
            },
        ],
        "tags": ["taiwan-canary"],
        "group": "tw-canary",
        "job_type": "evaluation",
    }
    return payload


def operator_docs_payload(*, ok: bool = True):
    check_names = [
        "readme_exists",
        "installer_lock_json_valid",
        "check_only_command",
        "installer_review_command",
        "install_and_onboard_command",
        "post_install_verification_command",
        "canary_readiness_command",
        "adoption_check_command",
        "production_readiness_command",
        "operator_docs_verifier_command",
        "no_openrouter_markers",
    ]
    checks = [{"name": name, "ok": ok} for name in check_names]
    return {
        "schema_version": 1,
        "ok": ok,
        "status": "passed" if ok else "failed",
        "generated_at": "2026-06-28T00:00:00Z",
        "readme_path": "docs/README_nemoclaw.md",
        "lock_json": "scripts/setup/nemoclaw_installer_lock.json",
        "will_execute_installer": False,
        "will_install_or_onboard": False,
        "will_launch_model_inference": False,
        "will_query_wandb": False,
        "missing_requirements": [] if ok else ["install_and_onboard_command"],
        "checks": checks,
    }


def wandb_completion_verifier_payload(
    *,
    benchmark: str = "agentic_math",
    run_id: str = "run-1",
    observed_evidence: dict | None = None,
):
    payload = {
        "ok": True,
        "benchmark": benchmark,
        "entity": "test-entity",
        "project": "test-project",
        "run_id": run_id,
        "query_source": {
            "kind": "wandb_sdk",
            "api": "wandb.Api",
            "timeout_seconds": 60,
            "entity": "test-entity",
            "project": "test-project",
            "run_id": run_id,
            "run_path": f"test-entity/test-project/{run_id}",
            "benchmark": benchmark,
            "summary_source": "run.summary_metrics",
            "artifact_source": "run.logged_artifacts",
            "history_scanned": False,
        },
        "generated_at": time.time(),
        "verification_schema_version": 1,
        "observed_evidence": observed_evidence
        or {
            "run_state": "finished",
            "summary_metrics": {
                "agentic_math/accuracy": {"ok": True, "value": 0.86},
            },
        },
    }
    return add_wandb_run_metadata(payload, benchmark=benchmark, run_id=run_id)


def weave_agents_completion_payload(
    *,
    run_id: str = "run-1",
    agent_name: str = "nejumi-taiwan-openclaw",
):
    return {
        "ok": True,
        "agent_name": agent_name,
        "generated_at": time.time(),
        "verification_schema_version": 1,
        "project_id": "test-entity/test-project",
        "agents_url": "https://wandb.ai/test-entity/test-project/weave/agents",
        "query_source": {
            "kind": "wandb_agents_api",
            "api_base_url": "https://trace.wandb.ai",
            "agents_endpoint": "/agents/query",
            "spans_endpoint": "/agents/spans/query",
            "project_id": "test-entity/test-project",
            "agent_name": agent_name,
            "conversation_id": "",
            "conversation_id_contains": run_id,
            "agents_count": 1,
            "spans_count": 2,
            "matching_span_count": 2,
            "latest_trace_span_count": 2,
        },
        "latest_trace_id": "trace-1",
        "required_evidence": {
            "input_message_required": True,
            "trace_timestamp_quality_required": True,
            "trace_final_answer_order_required": True,
            "conversation_id": "",
            "conversation_id_contains": run_id,
        },
        "content_capture_health": {
            "span_count_checked": 2,
            "message_span_count": 2,
            "message_spans_with_content": 2,
            "message_spans_with_input": 1,
            "tool_span_count": 1,
            "tool_spans_with_content": 1,
            "spans_with_valid_timestamps": 2,
            "spans_with_invalid_timestamps": 0,
        },
        "latest_trace_spans_chronological": [
            {
                "started_at": "2026-06-28T00:00:00Z",
                "ended_at": "2026-06-28T00:00:01Z",
                "span_name": "chat",
                "operation_name": "chat",
                "agent_name": agent_name,
                "trace_id": "trace-1",
                "span_id": "span-1",
                "parent_span_id": None,
                "error_type": None,
                "has_input_messages": True,
                "has_output_messages": True,
            },
            {
                "started_at": "2026-06-28T00:00:02Z",
                "ended_at": "2026-06-28T00:00:03Z",
                "span_name": "python.exec",
                "operation_name": "execute_tool",
                "agent_name": agent_name,
                "trace_id": "trace-1",
                "span_id": "span-2",
                "parent_span_id": "span-1",
                "tool_name": "python",
                "error_type": None,
            },
        ],
        "checks": [
            {"name": "agent_present", "ok": True},
            {"name": "message_content_capture", "ok": True},
            {"name": "input_message_capture", "ok": True},
            {"name": "tool_content_capture", "ok": True},
            {"name": "trace_timestamp_quality", "ok": True},
            {"name": "trace_order", "ok": True},
            {"name": "trace_user_message_order", "ok": True},
            {"name": "trace_final_answer_order", "ok": True},
        ],
    }


def weave_sync_dry_run_payload(
    *,
    review_path: Path,
    completion_path: Path,
    source_review_sha256: str,
    run_id: str = "run-1",
    agent_name: str = "nejumi-taiwan-openclaw",
):
    return {
        "ok": True,
        "status": "synced",
        "generated_at": time.time(),
        "review_path": str(review_path),
        "source_review_sha256": source_review_sha256,
        "output_path": "",
        "in_place": False,
        "dry_run": True,
        "entry_count": 1,
        "entries": [
            {
                "ok": True,
                "run_id": run_id,
                "path": str(completion_path),
                "agent_name": agent_name,
                "verification_schema_version": 1,
                "latest_trace_id": "trace-1",
                "checks_valid": True,
                "trace_present": True,
                "run_scope_proven": True,
                "conversation_id": "",
                "conversation_id_contains": run_id,
                "query_source_kind": "wandb_agents_api",
                "query_source_api_base_url": "https://trace.wandb.ai",
                "query_source_agents_endpoint": "/agents/query",
                "query_source_spans_endpoint": "/agents/spans/query",
                "query_source_project_id": "test-entity/test-project",
            }
        ],
        "change_count": 1,
        "changes": [
            {
                "target": "run",
                "action": "added",
                "run_id": run_id,
                "agent_name": agent_name,
                "config": "config.yaml",
            }
        ],
        "unmatched_count": 0,
        "unmatched_entries": [],
        "before_status": "completed",
        "after_status": "completed",
        "verify_weave_agents": True,
    }


def test_nemoclaw_operator_docs_gate_passes_latest_valid_report(tmp_path):
    module = load_module()
    docs = write_json(tmp_path / "operator_docs.json", operator_docs_payload())

    result = module.evaluate_nemoclaw_operator_docs([docs], require=True)

    assert result["ok"] is True
    assert result["blocking"] is False
    assert result["status"] == "passed"
    assert result["latest_report"]["path"] == str(docs)


def test_nemoclaw_operator_docs_gate_rejects_latest_failed_report(tmp_path):
    module = load_module()
    passed = write_json(tmp_path / "operator_docs_old.json", operator_docs_payload())
    failed = write_json(
        tmp_path / "operator_docs_new.json",
        operator_docs_payload(ok=False),
    )
    old_time = time.time() - 60
    new_time = time.time()
    os.utime(passed, (old_time, old_time))
    os.utime(failed, (new_time, new_time))

    result = module.evaluate_nemoclaw_operator_docs([passed, failed], require=True)

    assert result["ok"] is False
    assert result["blocking"] is True
    assert result["status"] == "failed"
    assert result["latest_report"]["path"] == str(failed)
    assert result["latest_report"]["failed_checks"] == [
        "readme_exists",
        "installer_lock_json_valid",
        "check_only_command",
        "installer_review_command",
        "install_and_onboard_command",
        "post_install_verification_command",
        "canary_readiness_command",
        "adoption_check_command",
        "production_readiness_command",
        "operator_docs_verifier_command",
        "no_openrouter_markers",
    ]


def test_nemoclaw_operator_docs_gate_rejects_latest_missing_required_check(tmp_path):
    module = load_module()
    payload = operator_docs_payload()
    payload["checks"] = [
        check
        for check in payload["checks"]
        if check["name"] != "production_readiness_command"
    ]
    docs = write_json(tmp_path / "operator_docs.json", payload)

    result = module.evaluate_nemoclaw_operator_docs([docs], require=True)

    assert result["ok"] is False
    assert result["blocking"] is True
    assert result["status"] == "failed"
    assert result["latest_report"]["path"] == str(docs)
    assert result["latest_report"]["missing_required_checks"] == [
        "production_readiness_command"
    ]


def test_paid_run_review_template_uses_concrete_review_paths():
    text = PAID_REVIEW_TEMPLATE.read_text(encoding="utf-8")

    assert "outputs/taiwan_full_eval/PHASE_paid_run_review.json" not in text
    assert "outputs/taiwan_full_eval/canary_agentic_paid_run_review.json" in text
    assert "outputs/taiwan_full_eval/canary_agentic_aggregate_paid_run_review.json" in text


def readiness_payload(*, nemoclaw_ok=True, report_ok=True):
    checks = [
        {"name": "manifest has exactly one canary", "ok": True},
        {"name": "NeMoClaw command is available", "ok": nemoclaw_ok},
        {"name": "OpenShell command is available", "ok": nemoclaw_ok},
        {"name": "NeMoClaw version command succeeds", "ok": nemoclaw_ok},
        {"name": "NeMoClaw sandbox status succeeds: nejumi-taiwan", "ok": nemoclaw_ok},
        {"name": "OpenClaw runs inside NeMoClaw sandbox: nejumi-taiwan", "ok": nemoclaw_ok},
    ]
    return {"ok": report_ok, "checks": checks}


def test_weave_content_gate_accepts_passed_json(tmp_path):
    module = load_module()
    gate = write_json(tmp_path / "canary.gate.json", {"ok": True, "status": "passed"})

    result = module.evaluate_weave_content_gate([gate], require=True)

    assert result["ok"] is True
    assert result["status"] == "passed"


def test_weave_content_gate_rejects_stale_passed_json(tmp_path):
    module = load_module()
    stale_generated_at = time.time() - 90_000
    gate = write_json(
        tmp_path / "canary.gate.json",
        {"ok": True, "status": "passed", "generated_at": stale_generated_at},
    )

    result = module.evaluate_weave_content_gate([gate], require=True, max_age_seconds=86_400)

    assert result["ok"] is False
    assert result["status"] == "stale"
    assert result["stale_passed_candidates"][0]["path"] == str(gate)
    assert result["stale_passed_candidates"][0]["fresh"] is False


def test_weave_content_gate_allows_disabled_freshness_check(tmp_path):
    module = load_module()
    stale_generated_at = time.time() - 90_000
    gate = write_json(
        tmp_path / "canary.gate.json",
        {"ok": True, "status": "passed", "generated_at": stale_generated_at},
    )

    result = module.evaluate_weave_content_gate([gate], require=True, max_age_seconds=None)

    assert result["ok"] is True
    assert result["status"] == "passed"


def test_nemoclaw_readiness_remediation_uses_reviewed_installer_sha(tmp_path):
    module = load_module()
    sha = "a" * 64
    lock = tmp_path / "nemoclaw_installer_lock.json"
    review = write_json(
        tmp_path / "nemoclaw_installer_review.json",
        {
            "schema_version": 1,
            "ok": True,
            "status": "reviewed",
            "generated_at": "2026-06-28T01:32:47Z",
            "installer_url": "https://www.nvidia.com/nemoclaw.sh",
            "install_ref": "lkg",
            "lock_json": str(lock),
            "lock_verified": True,
            "expected_sha256": sha,
            "sha256": sha,
            "size_bytes": 6356,
            "will_execute_installer": False,
            "will_install_or_onboard": False,
            "will_launch_model_inference": False,
            "will_query_wandb": False,
        },
    )

    result = module.evaluate_nemoclaw_readiness(
        [],
        require=True,
        installer_review_paths=[review],
    )

    assert result["status"] == "missing"
    assert result["latest_installer_review"]["path"] == str(review)
    assert result["latest_installer_review"]["lock_json"] == str(lock)
    assert result["latest_installer_review"]["lock_verified"] is True
    assert result["latest_installer_review"]["expected_sha256"] == sha
    commands = result["remediation_commands"]
    assert any(
        "review_nemoclaw_installer.py" in command
        and f"--expected-sha256 {sha}" in command
        and "--lock-json scripts/setup/nemoclaw_installer_lock.json" in command
        for command in commands
    )
    assert any(
        "install_nemoclaw.sh --install --onboard" in command
        and "--installer-lock-json scripts/setup/nemoclaw_installer_lock.json" in command
        and f"--installer-sha256 {sha}" in command
        and "--installer-review-json temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json" in command
        for command in commands
    )
    assert any(
        "check_taiwan_nemoclaw_adoption.py" in command
        and "--setup-json temp/nemoclaw_setup_check_YYYYMMDDTHHMM.json" in command
        and "--readiness-json outputs/taiwan_full_eval/openai_canary_readiness_nemoclaw.json" in command
        and "--agentic-config-glob configs/taiwan_full/generated_openai_canary_agentic_nemoclaw/*.yaml" in command
        and "--json temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.json" in command
        and "--markdown temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.md" in command
        for command in commands
    )


def test_weave_content_gate_fails_when_no_candidate_passes(tmp_path):
    module = load_module()
    gate = write_json(
        tmp_path / "canary.gate.json",
        {
            "ok": False,
            "status": "provider_failure",
            "failure_kind": "provider_quota",
            "detail": "provider returned insufficient_quota before a scoreable canary trace was produced",
            "generated_at": time.time(),
        },
    )

    result = module.evaluate_weave_content_gate([gate], require=True)

    assert result["ok"] is False
    assert result["status"] == "failed"
    assert result["candidates"][0]["failure_kind"] == "provider_quota"
    assert result["latest_candidate"]["failure_kind"] == "provider_quota"
    assert "provider_quota" in result["detail"]
    assert "available quota" in result["next_action"]
    assert any("run_weave_agents_content_canary.py --execute" in command for command in result["remediation_commands"])
    assert not any("run_openclaw_agent_protocol.py check-agents" in command for command in result["remediation_commands"])
    execute_commands = [
        command
        for command in result["remediation_commands"]
        if "run_weave_agents_content_canary.py --execute" in command
    ]
    assert execute_commands
    assert all(
        "--external-action-approval-source-packet-json "
        "outputs/taiwan_release_evidence/bundle_YYYYMMDDTHHMM/"
        "external_action_approval_packet.json"
        in command
        for command in execute_commands
    )
    assert all(
        "--external-action-approval-report-json "
        "temp/taiwan_external_action_approval_REVIEWED_YYYYMMDDTHHMM.verify.json"
        in command
        for command in execute_commands
    )


def test_weave_content_gate_points_to_verifier_regeneration_for_schema_invalid(tmp_path):
    module = load_module()
    gate = write_json(
        tmp_path / "canary.gate.json",
        {
            "ok": False,
            "status": "weave_verifier_schema_invalid",
            "detail": "required_evidence.trace_final_answer_order_required must be true",
            "generated_at": time.time(),
        },
    )

    result = module.evaluate_weave_content_gate([gate], require=True)

    assert result["ok"] is False
    assert result["status"] == "failed"
    assert result["latest_candidate"]["status"] == "weave_verifier_schema_invalid"
    assert "Regenerate the Weave Agents verifier JSON" in result["next_action"]
    assert "required_evidence.trace_timestamp_quality_required=true" in result["requirement"]
    assert "required_evidence.trace_final_answer_order_required=true" in result["requirement"]


def test_nemoclaw_readiness_requires_all_sandbox_checks(tmp_path):
    module = load_module()
    failed = write_json(tmp_path / "failed.json", readiness_payload(nemoclaw_ok=False, report_ok=False))
    passed = write_json(tmp_path / "passed.json", readiness_payload(nemoclaw_ok=True, report_ok=True))

    failed_result = module.evaluate_nemoclaw_readiness([failed], require=True)
    passed_result = module.evaluate_nemoclaw_readiness([failed, passed], require=True)

    assert failed_result["ok"] is False
    assert failed_result["status"] == "failed"
    assert passed_result["ok"] is True
    assert passed_result["status"] == "passed"


def test_nemoclaw_readiness_uses_latest_ready_evidence_path(tmp_path):
    module = load_module()
    old = write_json(tmp_path / "openai_canary_readiness_nemoclaw_A507.json", readiness_payload())
    new = write_json(tmp_path / "openai_canary_readiness_nemoclaw_A510.json", readiness_payload())
    os.utime(old, (100, 100))
    os.utime(new, (200, 200))

    result = module.evaluate_nemoclaw_readiness([old, new], require=True)

    assert result["ok"] is True
    assert result["status"] == "passed"
    assert result["evidence_paths"] == [str(new)]


def test_nemoclaw_readiness_includes_setup_check_json(tmp_path):
    module = load_module()
    provider_preflight = {
        "provider": "openai",
        "normalized_provider": "openai",
        "credential_available": True,
        "ready_for_noninteractive_onboard_preflight": True,
        "will_launch_probe": False,
        "will_launch_benchmark_inference": False,
    }
    setup = write_json(
        tmp_path / "nemoclaw_setup_check.json",
        {
            "ok": False,
            "provider": "openai",
            "model": "gpt-test",
            "provider_key_env": "OPENAI_API_KEY",
            "provider_preflight": provider_preflight,
            "sandbox_configured": False,
            "commands": {
                "nemoclaw": {"available": False},
                "openshell": {"available": False},
            },
            "setup_plan": {
                "will_launch_model_inference": False,
                "install_or_onboard_requires_explicit_acceptance": True,
                "install_command": "install",
            },
        },
    )

    result = module.evaluate_nemoclaw_readiness([], require=True, setup_paths=[setup])

    assert result["ok"] is False
    assert result["reports"][0]["checks"] == {
        "NeMoClaw command is available": False,
        "OpenShell command is available": False,
    }
    assert result["latest_setup_report"]["provider"] == "openai"
    assert result["latest_setup_report"]["model"] == "gpt-test"
    assert result["latest_setup_report"]["provider_key_env"] == "OPENAI_API_KEY"
    assert result["latest_setup_report"]["provider_preflight"] == provider_preflight
    assert result["latest_setup_report"]["sandbox_configured"] is False
    assert result["latest_setup_report"]["setup_plan"]["install_command"] == "install"
    assert result["recommended_setup_plan"]["will_launch_model_inference"] is False


def test_nemoclaw_readiness_remediation_prefers_latest_setup_plan(tmp_path):
    module = load_module()
    sha = "b" * 64
    review = write_json(
        tmp_path / "nemoclaw_installer_review.json",
        {
            "schema_version": 1,
            "ok": True,
            "status": "reviewed",
            "generated_at": "2026-06-28T01:32:47Z",
            "installer_url": "https://www.nvidia.com/nemoclaw.sh",
            "install_ref": "lkg",
            "lock_json": "scripts/setup/nemoclaw_installer_lock.json",
            "lock_verified": True,
            "sha256": sha,
            "size_bytes": 6356,
            "will_execute_installer": False,
            "will_install_or_onboard": False,
            "will_launch_model_inference": False,
            "will_query_wandb": False,
        },
    )
    setup_plan = {
        "post_install_check_command": (
            "scripts/setup/install_nemoclaw.sh --check-only --sandbox nejumi-taiwan "
            "--provider custom --gateway-port 18081 --env-file .env "
            "--endpoint-url http://127.0.0.1:8080/v1 "
            "--json temp/nemoclaw_setup_check_YYYYMMDDTHHMM.json"
        ),
        "installer_review_command": (
            "uv run python scripts/setup/review_nemoclaw_installer.py "
            "--url https://www.nvidia.com/nemoclaw.sh --install-ref lkg "
            "--lock-json scripts/setup/nemoclaw_installer_lock.json "
            "--json temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json "
            "--markdown temp/nemoclaw_installer_review_YYYYMMDDTHHMM.md"
        ),
        "production_install_and_onboard_command": (
            "scripts/setup/install_nemoclaw.sh --install --onboard --install-ref lkg "
            "--installer-lock-json scripts/setup/nemoclaw_installer_lock.json "
            "--installer-sha256 REVIEWED_INSTALLER_SHA256 "
            "--installer-review-json temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json "
            "--sandbox nejumi-taiwan --provider custom --gateway-port 18081 "
            "--env-file .env --endpoint-url http://127.0.0.1:8080/v1 "
            "--policy-tier restricted --yes-i-accept-third-party-software "
            "--json temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.json --model chat-model"
        ),
        "post_install_verification_command": "uv run python scripts/setup/verify_nemoclaw_post_install.py --sandbox nejumi-taiwan",
        "canary_readiness_command": "uv run python scripts/tools/check_taiwan_canary_readiness.py --require-nemoclaw --nemoclaw-sandbox nejumi-taiwan",
        "adoption_check_command": "uv run python scripts/tools/check_taiwan_nemoclaw_adoption.py --setup-json temp/nemoclaw_setup_check_YYYYMMDDTHHMM.json",
        "production_readiness_command": "uv run python scripts/tools/run_taiwan_production_readiness_gate.py --fail-on-not-ready",
    }
    setup = write_json(
        tmp_path / "nemoclaw_setup_check.json",
        {
            "ok": True,
            "provider": "custom",
            "sandbox_configured": False,
            "commands": {
                "nemoclaw": {"available": True},
                "openshell": {"available": True},
            },
            "setup_plan": setup_plan,
        },
    )

    result = module.evaluate_nemoclaw_readiness(
        [],
        require=True,
        setup_paths=[setup],
        installer_review_paths=[review],
    )

    commands = result["remediation_commands"]
    assert commands[0] == setup_plan["post_install_check_command"]
    assert any("--provider custom" in command for command in commands)
    assert any("--endpoint-url http://127.0.0.1:8080/v1" in command for command in commands)
    assert any("--model chat-model" in command for command in commands)
    assert any(f"--installer-sha256 {sha}" in command for command in commands)
    assert not any("--provider openai" in command for command in commands)


def test_nemoclaw_readiness_surfaces_provider_quota_failure(tmp_path):
    module = load_module()
    failure = {
        "failure_kind": "provider_quota",
        "failure_detail": "provider validation returned HTTP 429/quota",
    }
    setup = write_json(
        tmp_path / "nemoclaw_setup_check.json",
        {
            "ok": False,
            "provider": "openai",
            "provider_preflight": {
                "provider": "openai",
                "normalized_provider": "openai",
                "credential_available": True,
                "ready_for_noninteractive_onboard_preflight": True,
                "will_launch_probe": False,
                "will_launch_benchmark_inference": False,
            },
            "sandbox_configured": False,
            "commands": {
                "nemoclaw": {"available": True},
                "openshell": {"available": True},
            },
            "operation_results": {
                "onboard": {
                    "requested": True,
                    "attempted": True,
                    "skipped": False,
                    "returncode": 1,
                    "log_path": "temp/nemoclaw_onboard.log",
                    "failure": failure,
                },
            },
        },
    )

    result = module.evaluate_nemoclaw_readiness([], require=True, setup_paths=[setup])

    assert result["ok"] is False
    assert result["status"] == "failed"
    assert result["latest_onboard_failure"] == failure
    assert result["latest_setup_report"]["latest_onboard_failure"] == failure
    assert "provider validation returned HTTP 429/quota" in result["detail"]
    assert "available quota" in result["next_action"]


def test_nemoclaw_readiness_rejects_stale_pass_when_newer_setup_is_missing(tmp_path):
    module = load_module()
    passed = write_json(tmp_path / "passed.json", readiness_payload(nemoclaw_ok=True, report_ok=True))
    setup = write_json(
        tmp_path / "nemoclaw_setup_check.json",
        {
            "ok": False,
            "commands": {
                "nemoclaw": {"available": False},
                "openshell": {"available": False},
            },
        },
    )
    os.utime(passed, (100, 100))
    os.utime(setup, (200, 200))

    result = module.evaluate_nemoclaw_readiness([passed], require=True, setup_paths=[setup])

    assert result["ok"] is False
    assert result["status"] == "failed"
    assert result["latest_setup_report"]["path"] == str(setup)
    assert result["stale_ready_reports"][0]["path"] == str(passed)


def test_nemoclaw_readiness_accepts_newer_full_readiness_after_failed_setup(tmp_path):
    module = load_module()
    setup = write_json(
        tmp_path / "nemoclaw_setup_check.json",
        {
            "ok": False,
            "commands": {
                "nemoclaw": {"available": False},
                "openshell": {"available": False},
            },
        },
    )
    passed = write_json(tmp_path / "passed.json", readiness_payload(nemoclaw_ok=True, report_ok=True))
    os.utime(setup, (100, 100))
    os.utime(passed, (200, 200))

    result = module.evaluate_nemoclaw_readiness([passed], require=True, setup_paths=[setup])

    assert result["ok"] is True
    assert result["status"] == "passed"
    assert result["stale_ready_reports"] == []


def test_wandb_completion_requires_all_requested_benchmarks(tmp_path):
    module = load_module()
    math = write_json(
        tmp_path / "math.json",
        {
            "ok": True,
            "benchmark": "agentic_math",
            "run_id": "r1",
            "generated_at": time.time(),
        },
    )

    result = module.evaluate_wandb_completion(
        [math],
        required_benchmarks=["agentic_math", "agentic_swe"],
        require=True,
    )

    assert result["ok"] is False
    assert result["status"] == "missing_required_benchmarks"
    assert result["completed"] == {}
    assert result["records"][0]["schema_valid"] is False
    assert result["records"][0]["observed_evidence_valid"] is False
    assert result["legacy_schema_records"][0]["path"] == str(math)
    assert result["required_verification_schema_version"] == module.WANDB_COMPLETION_SCHEMA_VERSION
    assert result["missing_benchmarks"] == ["agentic_math", "agentic_swe"]
    assert result["remediation_commands"] == [
        (
            "uv run python scripts/tools/verify_taiwan_wandb_completion.py "
            "--run-id RUN_ID --benchmark agentic_math --expected-total 100 "
            "--env-file .env "
            "--json outputs/taiwan_full_eval/wandb_completion/agentic_math-RUN_ID.json"
        ),
        (
            "uv run python scripts/tools/verify_taiwan_wandb_completion.py "
            "--run-id RUN_ID --benchmark agentic_swe --expected-total 80 "
            "--env-file .env "
            "--json outputs/taiwan_full_eval/wandb_completion/agentic_swe-RUN_ID.json"
        )
    ]


def test_wandb_completion_can_require_specific_run_id(tmp_path):
    module = load_module()
    old_run = write_json(
        tmp_path / "old.json",
        {"ok": True, "benchmark": "agentic_math", "run_id": "old-run"},
    )

    result = module.evaluate_wandb_completion(
        [old_run],
        required_benchmarks=["agentic_math"],
        required_run_ids={"agentic_math": "new-run"},
        require=True,
    )

    assert result["ok"] is False
    assert result["status"] == "run_id_mismatch"
    assert result["missing_benchmarks"] == ["agentic_math"]
    assert result["run_id_mismatches"] == [
        {
            "path": str(old_run),
            "benchmark": "agentic_math",
            "run_id": "old-run",
            "expected_run_id": "new-run",
        }
    ]
    assert result["remediation_commands"] == [
        (
            "uv run python scripts/tools/verify_taiwan_wandb_completion.py "
            "--run-id new-run --benchmark agentic_math --expected-total 100 "
            "--env-file .env "
            "--json outputs/taiwan_full_eval/wandb_completion/agentic_math-new-run.json"
        )
    ]


def test_wandb_completion_accepts_specific_run_id_match(tmp_path):
    module = load_module()
    run = write_json(
        tmp_path / "completion.json",
        {
            "ok": True,
            "benchmark": "agentic_math",
            "entity": "test-entity",
            "project": "test-project",
            "run_id": "run-1",
            "generated_at": time.time(),
            "verification_schema_version": 1,
            "observed_evidence": {
                "run_state": "finished",
                "summary_metrics": {
                    "agentic_math/total_instances": {"ok": True, "value": 100},
                },
                "tables": [
                    {"name": "agentic_math_output_table", "ok": True, "nrows": 100},
                ],
                "artifacts": [
                    {
                        "name": "agentic-math-results:v0",
                        "type": "evaluation-results",
                        "aliases": ["latest", "production"],
                    }
                ],
            },
        },
    )

    result = module.evaluate_wandb_completion(
        [run],
        required_benchmarks=["agentic_math"],
        required_run_ids={"agentic_math": "run-1"},
        require=True,
    )

    assert result["ok"] is True
    assert result["status"] == "passed"
    assert result["completed"]["agentic_math"] == [str(run)]
    assert result["records"][0]["run_id_matches"] is True
    assert result["records"][0]["observed_evidence"]["run_state"] == "finished"
    assert (
        result["records"][0]["observed_evidence"]["summary_metrics"]
        ["agentic_math/total_instances"]["value"]
        == 100
    )
    assert result["records"][0]["observed_evidence_valid"] is True


def test_wandb_completion_rejects_legacy_schema_as_incomplete(tmp_path):
    module = load_module()
    run = write_json(
        tmp_path / "completion.json",
        {
            "ok": True,
            "benchmark": "agentic_math",
            "run_id": "run-1",
            "generated_at": time.time(),
            "observed_evidence": {
                "run_state": "finished",
                "summary_metrics": {
                    "agentic_math/accuracy": {"ok": True, "value": 0.86},
                },
            },
        },
    )

    result = module.evaluate_wandb_completion(
        [run],
        required_benchmarks=["agentic_math"],
        required_run_ids={"agentic_math": "run-1"},
        require=True,
    )

    assert result["ok"] is False
    assert result["status"] == "invalid_evidence"
    assert result["completed"] == {}
    assert result["missing_benchmarks"] == ["agentic_math"]
    assert result["legacy_schema_records"][0]["path"] == str(run)
    assert result["invalid_evidence_records"][0]["schema_valid"] is False
    assert result["invalid_evidence_records"][0]["observed_evidence_valid"] is True


def test_wandb_completion_rejects_non_finished_observed_evidence(tmp_path):
    module = load_module()
    run = write_json(
        tmp_path / "completion.json",
        {
            "ok": True,
            "benchmark": "agentic_math",
            "run_id": "run-1",
            "generated_at": time.time(),
            "verification_schema_version": 1,
            "observed_evidence": {
                "run_state": "running",
                "summary_metrics": {
                    "agentic_math/accuracy": {"ok": True, "value": 0.86},
                },
            },
        },
    )

    result = module.evaluate_wandb_completion(
        [run],
        required_benchmarks=["agentic_math"],
        required_run_ids={"agentic_math": "run-1"},
        require=True,
    )

    assert result["ok"] is False
    assert result["status"] == "invalid_evidence"
    assert result["completed"] == {}
    assert result["records"][0]["schema_valid"] is True
    assert result["records"][0]["observed_evidence_valid"] is False
    assert result["invalid_evidence_records"][0]["path"] == str(run)


def test_wandb_completion_rejects_missing_generated_at(tmp_path):
    module = load_module()
    run = write_json(
        tmp_path / "completion.json",
        {"ok": True, "benchmark": "agentic_math", "run_id": "run-1"},
    )

    result = module.evaluate_wandb_completion(
        [run],
        required_benchmarks=["agentic_math"],
        required_run_ids={"agentic_math": "run-1"},
        require=True,
    )

    assert result["ok"] is False
    assert result["status"] == "missing_generated_at"
    assert result["missing_generated_at_records"][0]["path"] == str(run)
    assert result["records"][0]["freshness_error"] == "missing generated_at"


def test_wandb_completion_rejects_stale_verifier_json(tmp_path):
    module = load_module()
    run = write_json(
        tmp_path / "completion.json",
        {
            "ok": True,
            "benchmark": "agentic_math",
            "run_id": "run-1",
            "generated_at": time.time() - 90_000,
        },
    )

    result = module.evaluate_wandb_completion(
        [run],
        required_benchmarks=["agentic_math"],
        required_run_ids={"agentic_math": "run-1"},
        require=True,
        max_age_seconds=86_400,
    )

    assert result["ok"] is False
    assert result["status"] == "stale"
    assert result["stale_completion_records"][0]["path"] == str(run)
    assert result["records"][0]["fresh"] is False


def test_one_model_canary_accepts_completed_phased_reviews(tmp_path):
    module = load_module()
    paths = [
        write_json(
            tmp_path / f"{phase}.json",
            {"status": "completed", "phase": phase, "canary": True, "model_count": 1},
        )
        for phase in ("nonagentic", "agentic", "agentic_aggregate")
    ]

    result = module.evaluate_one_model_canary(paths, require=True)

    assert result["ok"] is True
    assert result["status"] == "passed"


def test_one_model_canary_rejects_completed_agentic_without_nemoclaw_config(tmp_path):
    module = load_module()
    paths = [
        write_json(
            tmp_path / "nonagentic.json",
            {"status": "completed", "phase": "nonagentic", "canary": True, "model_count": 1},
        ),
        write_json(
            tmp_path / "agentic.json",
            {
                "status": "completed",
                "phase": "agentic",
                "canary": True,
                "model_count": 1,
                "configs": [NON_NEMOCLAW_AGENTIC_CONFIG],
            },
        ),
        write_json(
            tmp_path / "agentic_aggregate.json",
            {
                "status": "completed",
                "phase": "agentic_aggregate",
                "canary": True,
                "model_count": 1,
            },
        ),
    ]

    result = module.evaluate_one_model_canary(paths, require=True)

    assert result["ok"] is False
    assert result["status"] == "nemoclaw_agentic_config_not_proven"
    bad_record = result["bad_nemoclaw_agentic_config_records"][0]
    assert bad_record["configs"] == [NON_NEMOCLAW_AGENTIC_CONFIG]
    assert bad_record["nemoclaw_agentic_config_required"] is True
    assert bad_record["nemoclaw_agentic_config_bound"] is False


def test_one_model_canary_records_review_common_summary_fields(tmp_path):
    module = load_module()
    path = write_json(
        tmp_path / "agentic.json",
        {
            "status": "prepared",
            "phase": "agentic",
            "canary": True,
            "model_count": 1,
            "run_purpose": "one-model agentic canary",
            "expected_cost_band": "$10-$20",
            "actual_cost_estimate": "$12.34",
            "provider_bill_reference": "openai-dashboard-2026-06-28",
            "verify_wandb_completion": True,
            "verify_weave_agents": True,
            "runs": [
                {
                    "wandb_run_id": "run-1",
                    "weave_agents_completion": {
                        "ok": False,
                        "path": "outputs/taiwan_full_eval/weave_agents_completion/agentic-run-1.json",
                        "agent_name": "agentic-run-1",
                    },
                }
            ],
        },
    )

    result = module.evaluate_one_model_canary([path], require=True)

    record = result["records"][0]
    assert record["run_purpose_present"] is True
    assert record["expected_cost_band_present"] is True
    assert record["actual_cost_estimate_present"] is True
    assert record["provider_bill_reference_present"] is True
    assert record["actual_cost_estimate_placeholder"] is False
    assert record["provider_bill_reference_placeholder"] is False
    assert record["run_count"] == 1
    assert record["verify_wandb_completion"] is True
    assert record["verify_weave_agents"] is True
    assert record["weave_agents_completion_entries"][0]["agent_name"] == "agentic-run-1"
    assert record["weave_agents_completion_max_age_seconds"] == 86400


def test_one_model_canary_requires_matching_wandb_run_id_when_pinned(tmp_path):
    module = load_module()
    paths = [
        write_json(
            tmp_path / f"{phase}.json",
            {
                "status": "completed",
                "phase": phase,
                "canary": True,
                "model_count": 1,
                "runs": [{"wandb_run_id": "other-run"}],
            },
        )
        for phase in ("nonagentic", "agentic", "agentic_aggregate")
    ]

    result = module.evaluate_one_model_canary(
        paths,
        require=True,
        required_run_ids={"agentic_math": "required-run"},
    )

    assert result["ok"] is False
    assert result["status"] == "run_id_not_proven"
    assert result["missing_required_wandb_run_ids"] == ["required-run"]
    assert result["completed_wandb_run_ids"] == ["other-run"]


def test_one_model_canary_requires_wandb_completion_entries_when_benchmarks_required(tmp_path):
    module = load_module()
    paths = [
        write_json(
            tmp_path / f"{phase}.json",
            {
                "status": "completed",
                "phase": phase,
                "canary": True,
                "model_count": 1,
                "runs": [{"wandb_run_id": "required-run"}],
            },
        )
        for phase in ("nonagentic", "agentic", "agentic_aggregate")
    ]

    result = module.evaluate_one_model_canary(
        paths,
        require=True,
        required_run_ids={"agentic_math": "required-run"},
        required_benchmarks=["agentic_math"],
    )

    assert result["ok"] is False
    assert result["status"] == "wandb_completion_not_proven"
    assert result["missing_required_wandb_completion_benchmarks"] == ["agentic_math"]


def test_one_model_canary_accepts_pinned_wandb_run_id_from_completed_review(tmp_path):
    module = load_module()
    paths = [
        write_json(
            tmp_path / f"{phase}.json",
            {
                "status": "completed",
                "phase": phase,
                "canary": True,
                "model_count": 1,
                "runs": [{"wandb_run_id": "required-run"}],
            },
        )
        for phase in ("nonagentic", "agentic", "agentic_aggregate")
    ]

    result = module.evaluate_one_model_canary(
        paths,
        require=True,
        required_run_ids={"agentic_math": "required-run"},
    )

    assert result["ok"] is True
    assert result["status"] == "passed"
    assert result["completed_wandb_run_ids"] == ["required-run"]


def test_one_model_canary_accepts_required_wandb_completion_entries(tmp_path):
    module = load_module()
    math_verifier = write_json(
        tmp_path / "math.json",
        wandb_completion_verifier_payload(
            benchmark="agentic_math",
            run_id="required-run",
            observed_evidence={
                "run_state": "finished",
                "summary_metrics": {
                    "agentic_math/accuracy": {"ok": True, "value": 0.86},
                },
            },
        ),
    )
    swe_verifier = write_json(
        tmp_path / "swe.json",
        wandb_completion_verifier_payload(
            benchmark="agentic_swe",
            run_id="required-run",
            observed_evidence={
                "run_state": "finished",
                "summary_metrics": {
                    "agentic_swe/pass_at_1": {"ok": True, "value": 0.5},
                },
            },
        ),
    )
    full_verifier = write_json(
        tmp_path / "full.json",
        wandb_completion_verifier_payload(
            benchmark="taiwan_full",
            run_id="required-run",
            observed_evidence={
                "run_state": "finished",
                "aggregate_tables": [
                    {"name": "taiwan_leaderboard_table", "ok": True, "nrows": 1},
                ],
            },
        ),
    )
    weave_verifier = write_json(
        tmp_path / "weave.json",
        weave_agents_completion_payload(run_id="required-run"),
    )
    agentic_review_before_weave = write_json(
        tmp_path / "agentic.before_weave.json",
        {
            "status": "completed",
            "phase": "agentic",
            "canary": True,
            "model_count": 1,
            "runs": [{"wandb_run_id": "required-run"}],
        },
    )
    agentic_review_before_weave_sha = sha256(agentic_review_before_weave)
    weave_sync_dry_run = write_json(
        tmp_path / "weave.sync_dry_run.json",
        weave_sync_dry_run_payload(
            review_path=agentic_review_before_weave,
            completion_path=weave_verifier,
            source_review_sha256=agentic_review_before_weave_sha,
            run_id="required-run",
        ),
    )
    paths = [
        write_json(
            tmp_path / "nonagentic.json",
            {
                "status": "completed",
                "phase": "nonagentic",
                "canary": True,
                "model_count": 1,
                "runs": [{"wandb_run_id": "required-run"}],
            },
        ),
        write_json(
            tmp_path / "agentic.json",
            {
                "status": "completed",
                "phase": "agentic",
                "canary": True,
                "model_count": 1,
                "runs": [
                    {
                        "wandb_run_id": "required-run",
                        "wandb_completion": [
                            {
                                "benchmark": "agentic_math",
                                "entity": "test-entity",
                                "project": "test-project",
                                "ok": True,
                                "path": str(math_verifier),
                                "sha256": sha256(math_verifier),
                            },
                            {
                                "benchmark": "agentic_swe",
                                "entity": "test-entity",
                                "project": "test-project",
                                "ok": True,
                                "path": str(swe_verifier),
                                "sha256": sha256(swe_verifier),
                            },
                        ],
                        "weave_agents_completion": {
                            "ok": True,
                            "path": str(weave_verifier),
                            "agent_name": "nejumi-taiwan-openclaw",
                            "run_id": "required-run",
                            "sync_dry_run_report_json": str(weave_sync_dry_run),
                            "sync_dry_run_source_review_json": str(agentic_review_before_weave),
                            "sync_dry_run_source_review_sha256": agentic_review_before_weave_sha,
                        },
                    }
                ],
            },
        ),
        write_json(
            tmp_path / "agentic_aggregate.json",
            {
                "status": "completed",
                "phase": "agentic_aggregate",
                "canary": True,
                "model_count": 1,
                "runs": [
                    {
                        "wandb_run_id": "required-run",
                        "wandb_completion": [
                            {
                                "benchmark": "taiwan_full",
                                "entity": "test-entity",
                                "project": "test-project",
                                "ok": True,
                                "path": str(full_verifier),
                                "sha256": sha256(full_verifier),
                            },
                        ],
                    }
                ],
            },
        ),
    ]

    result = module.evaluate_one_model_canary(
        paths,
        require=True,
        required_run_ids={
            "agentic_math": "required-run",
            "agentic_swe": "required-run",
            "taiwan_full": "required-run",
        },
        required_benchmarks=["agentic_math", "agentic_swe", "taiwan_full"],
    )

    assert result["ok"] is True
    assert result["status"] == "passed"
    assert result["missing_required_wandb_completion_benchmarks"] == []
    assert result["weave_agents_completion_required"] is True
    assert result["missing_required_weave_agents_completion_phases"] == []
    assert sorted(result["completed_weave_agents_completion_phases"]) == ["agentic"]
    assert sorted(result["completed_wandb_completion_benchmarks"]) == [
        "agentic_math",
        "agentic_swe",
        "taiwan_full",
    ]


def test_one_model_canary_rejects_agentic_benchmark_without_weave_agents_completion(tmp_path):
    module = load_module()
    math_verifier = write_json(
        tmp_path / "math.json",
        wandb_completion_verifier_payload(
            benchmark="agentic_math",
            run_id="required-run",
            observed_evidence={
                "run_state": "finished",
                "summary_metrics": {
                    "agentic_math/accuracy": {"ok": True, "value": 0.86},
                },
            },
        ),
    )
    review = write_json(
        tmp_path / "full.json",
        {
            "status": "completed",
            "phase": "full",
            "canary": True,
            "model_count": 1,
            "runs": [
                {
                    "wandb_run_id": "required-run",
                    "wandb_completion": [
                        {
                            "benchmark": "agentic_math",
                            "entity": "test-entity",
                            "project": "test-project",
                            "ok": True,
                            "path": str(math_verifier),
                            "sha256": sha256(math_verifier),
                        },
                    ],
                }
            ],
        },
    )

    result = module.evaluate_one_model_canary(
        [review],
        require=True,
        required_run_ids={"agentic_math": "required-run"},
        required_benchmarks=["agentic_math"],
    )

    assert result["ok"] is False
    assert result["status"] == "weave_agents_completion_not_proven"
    assert result["missing_required_wandb_completion_benchmarks"] == []
    assert result["weave_agents_completion_required"] is True
    assert result["missing_required_weave_agents_completion_phases"] == ["full"]


def test_one_model_canary_rejects_wandb_completion_without_run_metadata(tmp_path):
    module = load_module()
    payload = wandb_completion_verifier_payload(
        benchmark="agentic_math",
        run_id="required-run",
        observed_evidence={
            "run_state": "finished",
            "summary_metrics": {
                "agentic_math/accuracy": {"ok": True, "value": 0.86},
            },
        },
    )
    payload.pop("required_evidence", None)
    payload["observed_evidence"].pop("run_metadata", None)
    verifier = write_json(tmp_path / "math.json", payload)
    review = write_json(
        tmp_path / "agentic.json",
        {
            "status": "completed",
            "phase": "full",
            "canary": True,
            "model_count": 1,
            "runs": [
                {
                    "wandb_run_id": "required-run",
                    "wandb_completion": [
                        {
                            "benchmark": "agentic_math",
                            "entity": "test-entity",
                            "project": "test-project",
                            "ok": True,
                            "path": str(verifier),
                            "sha256": sha256(verifier),
                        }
                    ],
                }
            ],
        },
    )

    result = module.evaluate_one_model_canary(
        [review],
        require=True,
        required_run_ids={"agentic_math": "required-run"},
        required_benchmarks=["agentic_math"],
    )

    assert result["ok"] is False
    assert result["status"] == "wandb_completion_not_proven"
    entry = result["records"][0]["wandb_completion_entries"][0]
    assert entry["run_metadata_valid"] is False
    assert "required_evidence.run_metadata is not an object" in entry["verification_error"]
    assert "observed_evidence.run_metadata is not an object" in entry["verification_error"]


def test_one_model_canary_rejects_unverified_wandb_completion_entry(tmp_path):
    module = load_module()
    paths = [
        write_json(
            tmp_path / f"{phase}.json",
            {
                "status": "completed",
                "phase": phase,
                "canary": True,
                "model_count": 1,
                "runs": [
                    {
                        "wandb_run_id": "required-run",
                        "wandb_completion": [
                            {
                                "benchmark": "agentic_math",
                                "ok": True,
                                "path": str(tmp_path / "missing-verifier.json"),
                            }
                        ],
                    }
                ],
            },
        )
        for phase in ("nonagentic", "agentic", "agentic_aggregate")
    ]

    result = module.evaluate_one_model_canary(
        paths,
        require=True,
        required_run_ids={"agentic_math": "required-run"},
        required_benchmarks=["agentic_math"],
    )

    assert result["ok"] is False
    assert result["status"] == "wandb_completion_not_proven"
    entry = result["records"][0]["wandb_completion_entries"][0]
    assert entry["entry_ok"] is True
    assert entry["verified"] is False
    assert "missing-verifier.json" in entry["verification_error"]


def test_one_model_canary_rejects_stale_wandb_completion_verifier(tmp_path):
    module = load_module()
    stale_verifier = write_json(
        tmp_path / "stale-verifier.json",
        {
            "ok": True,
            "benchmark": "agentic_math",
            "run_id": "required-run",
            "generated_at": time.time() - 90_000,
            "verification_schema_version": 1,
            "observed_evidence": {
                "run_state": "finished",
                "summary_metrics": {
                    "agentic_math/accuracy": {"ok": True, "value": 0.86},
                },
            },
        },
    )
    paths = [
        write_json(
            tmp_path / f"{phase}.json",
            {
                "status": "completed",
                "phase": phase,
                "canary": True,
                "model_count": 1,
                "runs": [
                    {
                        "wandb_run_id": "required-run",
                        "wandb_completion": [
                            {
                                "benchmark": "agentic_math",
                                "ok": True,
                                "path": str(stale_verifier),
                            }
                        ],
                    }
                ],
            },
        )
        for phase in ("nonagentic", "agentic", "agentic_aggregate")
    ]

    result = module.evaluate_one_model_canary(
        paths,
        require=True,
        required_run_ids={"agentic_math": "required-run"},
        required_benchmarks=["agentic_math"],
        wandb_completion_max_age_seconds=86_400,
    )

    assert result["ok"] is False
    assert result["status"] == "wandb_completion_not_proven"
    entry = result["records"][0]["wandb_completion_entries"][0]
    assert entry["verifier_json_ok"] is True
    assert entry["fresh"] is False
    assert entry["verified"] is False


def test_one_model_canary_rejects_legacy_wandb_completion_verifier(tmp_path):
    module = load_module()
    legacy_verifier = write_json(
        tmp_path / "legacy-verifier.json",
        {
            "ok": True,
            "benchmark": "agentic_math",
            "run_id": "required-run",
            "generated_at": time.time(),
        },
    )
    paths = [
        write_json(
            tmp_path / f"{phase}.json",
            {
                "status": "completed",
                "phase": phase,
                "canary": True,
                "model_count": 1,
                "runs": [
                    {
                        "wandb_run_id": "required-run",
                        "wandb_completion": [
                            {
                                "benchmark": "agentic_math",
                                "ok": True,
                                "path": str(legacy_verifier),
                            }
                        ],
                    }
                ],
            },
        )
        for phase in ("nonagentic", "agentic", "agentic_aggregate")
    ]

    result = module.evaluate_one_model_canary(
        paths,
        require=True,
        required_run_ids={"agentic_math": "required-run"},
        required_benchmarks=["agentic_math"],
        wandb_completion_max_age_seconds=86_400,
    )

    assert result["ok"] is False
    assert result["status"] == "wandb_completion_not_proven"
    entry = result["records"][0]["wandb_completion_entries"][0]
    assert entry["schema_valid"] is False
    assert entry["observed_evidence_valid"] is False
    assert entry["verified"] is False


def completed_review_payload(**overrides):
    preflight = write_run_eval_preflight_payload()
    payload = {
        "status": "completed",
        "phase": "full",
        "canary": True,
        "model_count": 1,
        "configs": ["configs/taiwan_full/generated/config-taiwan-full-test.yaml"],
        "run_purpose": "one-model release canary",
        "expected_cost_band": "$10-$20",
        "execution_plan_path": "outputs/taiwan_full_eval/canary_full_execution_plan.json",
        "batch_manifest_path": "outputs/taiwan_full_eval/batch_manifest.json",
        "post_run_cost_command": "uv run python scripts/analysis/estimate_agentic_usage_costs.py outputs/taiwan_full_eval",
        "run_eval_preflights": [
            {
                "config": "configs/taiwan_full/generated/config-taiwan-full-test.yaml",
                "output_json": str(preflight),
                "required_before_run_eval": True,
                "command": [
                    "python3",
                    "scripts/run_eval.py",
                    "--base-config",
                    "base_config_taiwan.yaml",
                    "--config",
                    "configs/taiwan_full/generated/config-taiwan-full-test.yaml",
                    "--preflight",
                    "--preflight-json",
                    str(preflight),
                ],
            }
        ],
        "actual_cost_estimate": "$12.34",
        "provider_bill_reference": "billing export 2026-06-27",
        "created_at": time.time() - 100,
        "started_at": time.time() - 90,
        "ended_at": time.time(),
        "verify_wandb_completion": False,
        "runs": [
            {
                "config": "configs/taiwan_full/generated/config-taiwan-full-test.yaml",
                "preflight_json": str(preflight),
                "preflight_returncode": 0,
                "preflight_ok": True,
                "preflight_status": "passed",
                "log_path": "outputs/taiwan_full_eval/logs/full-test.log",
                "wandb_run_id": "run-1",
                "wandb_entity": "test-entity",
                "wandb_project": "test-project",
                "returncode": 0,
                "started_at": time.time() - 80,
                "ended_at": time.time() - 10,
            }
        ],
    }
    payload.update(overrides)
    return payload


def wandb_sync_dry_run_payload(
    *,
    source_review: Path,
    completion: Path,
    completion_sha: str,
    source_attestation: Path,
    source_attestation_sha: str,
    source_audit: Path,
    source_audit_sha: str,
):
    return {
        "ok": True,
        "status": "synced",
        "generated_at": time.time(),
        "review_path": str(source_review),
        "source_review_sha256": sha256(source_review),
        "output_path": "",
        "in_place": False,
        "dry_run": True,
        "entry_count": 1,
        "adopted_existing_result_count": 1,
        "entries": [
            {
                "benchmark": "agentic_math",
                "ok": True,
                "entity": "test-entity",
                "project": "test-project",
                "run_id": "run-1",
                "path": str(completion),
                "sha256": completion_sha,
                "verification_schema_version": 1,
                "observed_evidence_valid": True,
                "run_metadata_valid": True,
                "adopted_existing_result": True,
                "scope_attestation": {
                    "schema_version": 1,
                    "confirmed": True,
                    "confirmed_by": "yuya",
                    "confirmed_at": "2026-06-28T01:30:00+09:00",
                    "confirmation": "This W&B run is the reviewed Agentic Math canary scope.",
                    "review_path": str(source_review),
                    "completion_path": str(completion),
                    "completion_sha256": completion_sha,
                    "benchmark": "agentic_math",
                    "entity": "test-entity",
                    "project": "test-project",
                    "run_id": "run-1",
                    "actual_cost_estimate": "$12.34",
                    "provider_bill_reference": "openai-dashboard-2026-06-28",
                    "source_attestation_json": str(source_attestation),
                    "source_attestation_sha256": source_attestation_sha,
                    "source_audit_json": str(source_audit),
                    "source_audit_sha256": source_audit_sha,
                },
            }
        ],
        "change_count": 1,
        "unmatched_count": 0,
        "changes": [
            {
                "target": "run",
                "action": "added",
                "benchmark": "agentic_math",
                "entity": "test-entity",
                "project": "test-project",
                "run_id": "run-1",
                "match_status": "matched_wandb_identity",
            }
        ],
        "unmatched_entries": [],
        "before_status": "completed",
        "after_status": "completed",
        "verify_wandb_completion": True,
    }


def wandb_source_audit_payload(*, completion: Path):
    return {
        "ok": True,
        "status": "passed",
        "formalized_records": [
            {
                "benchmark": "agentic_math",
                "wandb_completion": {
                    "path": str(completion),
                    "entity": "test-entity",
                    "project": "test-project",
                    "run_id": "run-1",
                    "verification_schema_version": 1,
                },
            }
        ],
    }


def test_paid_run_review_package_accepts_complete_review(tmp_path):
    module = load_module()
    review = write_json(tmp_path / "review.json", completed_review_payload())

    result = module.evaluate_paid_run_review_package([review], require=True)

    assert result["ok"] is True
    assert result["status"] == "passed"
    record = result["complete_records"][0]
    assert record["actual_cost_estimate_present"] is True
    assert record["configs"] == ["configs/taiwan_full/generated/config-taiwan-full-test.yaml"]
    assert record["requires_paid_model_api"] is False


def test_paid_run_review_package_rejects_completed_review_without_run_eval_preflight(tmp_path):
    module = load_module()
    payload = completed_review_payload()
    payload.pop("run_eval_preflights")
    payload["runs"][0].pop("preflight_json")
    payload["runs"][0].pop("preflight_returncode")
    payload["runs"][0].pop("preflight_ok")
    review = write_json(tmp_path / "review.json", payload)

    result = module.evaluate_paid_run_review_package([review], require=True)

    assert result["ok"] is False
    assert result["status"] == "invalid_review_package"
    record = result["records"][0]
    assert "completed review must include run_eval_preflights" in record["errors"]
    assert "run 1 missing preflight_json" in record["errors"]
    assert "run 1 preflight_returncode must be 0" in record["errors"]
    assert "run 1 preflight_ok must be true" in record["errors"]


def test_paid_run_review_package_rejects_mismatched_run_eval_preflight(tmp_path):
    module = load_module()
    payload = completed_review_payload()
    payload["run_eval_preflights"][0]["output_json"] = "temp/other_preflight.json"
    payload["run_eval_preflights"][0]["command"][-1] = "temp/other_preflight.json"
    review = write_json(tmp_path / "review.json", payload)

    result = module.evaluate_paid_run_review_package([review], require=True)

    assert result["ok"] is False
    record = result["records"][0]
    assert (
        "run 1 preflight_json does not match a top-level run_eval_preflights output_json"
        in record["errors"]
    )


def test_paid_run_review_package_rejects_completed_agentic_without_nemoclaw_config(tmp_path):
    module = load_module()
    review = write_json(
        tmp_path / "review.json",
        completed_review_payload(
            phase="agentic",
            configs=[NON_NEMOCLAW_AGENTIC_CONFIG],
            runs=[
                {
                    "config": NON_NEMOCLAW_AGENTIC_CONFIG,
                    "log_path": "outputs/taiwan_full_eval/logs/agentic-test.log",
                    "wandb_run_id": "run-1",
                    "wandb_entity": "test-entity",
                    "wandb_project": "test-project",
                    "returncode": 0,
                    "started_at": time.time() - 80,
                    "ended_at": time.time() - 10,
                }
            ],
        ),
    )

    result = module.evaluate_paid_run_review_package([review], require=True)

    assert result["ok"] is False
    assert result["status"] == "invalid_review_package"
    record = result["records"][0]
    assert record["nemoclaw_agentic_config_required"] is True
    assert record["nemoclaw_agentic_config_bound"] is False
    assert (
        "completed agentic canary review must reference the NeMoClaw agentic generated config"
        in record["errors"]
    )


def test_paid_run_review_package_verifies_wandb_completion_entry(tmp_path):
    module = load_module()
    completion = write_json(
        tmp_path / "agentic_math-run-1.json",
        add_wandb_run_metadata(
            {
                "ok": True,
                "benchmark": "agentic_math",
                "entity": "test-entity",
                "project": "test-project",
                "run_id": "run-1",
                "generated_at": time.time(),
                "verification_schema_version": 1,
                "observed_evidence": {
                    "run_state": "finished",
                    "summary_metrics": {
                        "agentic_math/accuracy": {"ok": True, "value": 0.86},
                    },
                },
            }
        ),
    )
    completion_sha = sha256(completion)
    source_review = write_json(
        tmp_path / "source_review.json",
        completed_review_payload(verify_wandb_completion=True),
    )
    source_audit = write_json(
        tmp_path / "source_audit.json",
        wandb_source_audit_payload(completion=completion),
    )
    source_audit_sha = sha256(source_audit)
    scope_attestation = write_json(
        tmp_path / "agentic_math-run-1.scope_attestation.json",
        {
            "schema_version": 1,
            "confirmed": True,
            "confirmed_by": "yuya",
            "confirmed_at": "2026-06-28T01:30:00+09:00",
            "confirmation": "This W&B run is the reviewed Agentic Math canary scope.",
            "review_path": str(source_review),
            "completion_path": str(completion),
            "completion_sha256": completion_sha,
            "benchmark": "agentic_math",
            "entity": "test-entity",
            "project": "test-project",
            "run_id": "run-1",
            "actual_cost_estimate": "$12.34",
            "provider_bill_reference": "openai-dashboard-2026-06-28",
            "source_audit_json": str(source_audit),
            "source_audit_sha256": source_audit_sha,
        },
    )
    scope_attestation_sha = sha256(scope_attestation)
    dry_run = write_json(
        tmp_path / "agentic_math-run-1.sync_dry_run.json",
        wandb_sync_dry_run_payload(
            source_review=source_review,
            completion=completion,
            completion_sha=completion_sha,
            source_attestation=scope_attestation,
            source_attestation_sha=scope_attestation_sha,
            source_audit=source_audit,
            source_audit_sha=source_audit_sha,
        ),
    )
    payload = completed_review_payload(verify_wandb_completion=True)
    payload["runs"][0]["wandb_completion"] = [
        {
            "ok": True,
            "benchmark": "agentic_math",
            "entity": "test-entity",
            "project": "test-project",
            "run_id": "run-1",
            "path": str(completion),
            "sha256": completion_sha,
            "adopted_existing_result": True,
            "sync_dry_run_report_json": str(dry_run),
            "sync_dry_run_source_review_json": str(source_review),
            "sync_dry_run_source_review_sha256": sha256(source_review),
            "scope_attestation": {
                "schema_version": 1,
                "confirmed": True,
                "confirmed_by": "yuya",
                "confirmed_at": "2026-06-28T01:30:00+09:00",
                "confirmation": "This W&B run is the reviewed Agentic Math canary scope.",
                "review_path": str(source_review),
                "completion_path": str(completion),
                "completion_sha256": completion_sha,
                "benchmark": "agentic_math",
                "entity": "test-entity",
                "project": "test-project",
                "run_id": "run-1",
                "actual_cost_estimate": "$12.34",
                "provider_bill_reference": "openai-dashboard-2026-06-28",
                "source_attestation_json": str(scope_attestation),
                "source_attestation_sha256": scope_attestation_sha,
                "source_audit_json": str(source_audit),
                "source_audit_sha256": source_audit_sha,
            },
        }
    ]
    review = write_json(tmp_path / "review.json", payload)

    result = module.evaluate_paid_run_review_package(
        [review],
        require=True,
        wandb_completion_max_age_seconds=86_400,
    )

    assert result["ok"] is True
    entry = result["complete_records"][0]["wandb_completion_entries"][0]
    assert entry["verified"] is True
    assert entry["verifier_json_ok"] is True
    assert entry["sha256"] == completion_sha
    assert entry["sha256_actual"] == completion_sha
    assert entry["sha256_matches"] is True
    assert entry["parent_run_id"] == "run-1"
    assert entry["parent_entity"] == "test-entity"
    assert entry["parent_project"] == "test-project"
    assert entry["parent_run_id_matches"] is True
    assert entry["parent_entity_matches"] is True
    assert entry["parent_project_matches"] is True
    assert entry["observed_evidence"]["run_state"] == "finished"
    assert (
        entry["observed_evidence"]["summary_metrics"]["agentic_math/accuracy"]["value"]
        == 0.86
    )
    assert entry["adopted_existing_result"] is True
    assert entry["scope_attestation_valid"] is True
    assert entry["scope_attestation"]["source_attestation_json"] == str(scope_attestation)
    assert entry["scope_attestation"]["source_attestation_sha256"] == scope_attestation_sha
    assert entry["scope_attestation"]["source_audit_json"] == str(source_audit)
    assert entry["scope_attestation"]["source_audit_sha256"] == source_audit_sha
    assert entry["scope_attestation"]["source_audit_completion_entry_matches"] is True
    assert entry["sync_dry_run_report_ok"] is True
    assert entry["sync_dry_run_source_review_sha256_matches"] is True
    assert entry["scope_attestation"]["attested_completion_sha256"] == completion_sha
    assert entry["scope_attestation"]["actual_cost_estimate"] == "$12.34"
    assert entry["scope_attestation"]["provider_bill_reference"] == "openai-dashboard-2026-06-28"


def test_scope_attestation_rejects_source_audit_without_completion_entry(tmp_path):
    module = load_module()
    completion = write_json(
        tmp_path / "agentic_math-run-1.json",
        {"ok": True, "benchmark": "agentic_math"},
    )
    completion_sha = sha256(completion)
    source_review = write_json(
        tmp_path / "source_review.json",
        completed_review_payload(verify_wandb_completion=True),
    )
    source_audit = write_json(
        tmp_path / "source_audit.json",
        {
            "ok": True,
            "status": "passed",
            "formalized_records": [
                {
                    "benchmark": "agentic_math",
                    "wandb_completion": {
                        "path": str(completion),
                        "entity": "test-entity",
                        "project": "test-project",
                        "run_id": "other-run",
                    },
                }
            ],
        },
    )
    source_audit_sha = sha256(source_audit)
    scope_payload = {
        "schema_version": 1,
        "confirmed": True,
        "confirmed_by": "yuya",
        "confirmed_at": "2026-06-28T01:30:00+09:00",
        "confirmation": "This W&B run is the reviewed Agentic Math canary scope.",
        "review_path": str(source_review),
        "completion_path": str(completion),
        "completion_sha256": completion_sha,
        "benchmark": "agentic_math",
        "entity": "test-entity",
        "project": "test-project",
        "run_id": "run-1",
        "actual_cost_estimate": "$12.34",
        "provider_bill_reference": "openai-dashboard-2026-06-28",
        "source_audit_json": str(source_audit),
        "source_audit_sha256": source_audit_sha,
    }
    source_attestation = write_json(
        tmp_path / "agentic_math-run-1.scope_attestation.json",
        scope_payload,
    )
    source_attestation_sha = sha256(source_attestation)
    scope_payload["source_attestation_json"] = str(source_attestation)
    scope_payload["source_attestation_sha256"] = source_attestation_sha

    result = module._verify_scope_attestation(
        {
            "ok": True,
            "benchmark": "agentic_math",
            "entity": "test-entity",
            "project": "test-project",
            "run_id": "run-1",
            "path": str(completion),
            "sha256": completion_sha,
            "adopted_existing_result": True,
            "sync_dry_run_source_review_json": str(source_review),
            "scope_attestation": scope_payload,
        }
    )

    assert result["verified"] is False
    assert result["source_audit_sha256_matches"] is True
    assert result["source_audit_completion_entry_matches"] is False
    assert (
        "scope_attestation source_audit_json formalized_records does not include completion entry"
        in result["errors"]
    )


def test_paid_run_review_package_rejects_adopted_wandb_completion_missing_sync_dry_run(tmp_path):
    module = load_module()
    completion = write_json(
        tmp_path / "agentic_math-run-1.json",
        add_wandb_run_metadata(
            {
                "ok": True,
                "benchmark": "agentic_math",
                "entity": "test-entity",
                "project": "test-project",
                "run_id": "run-1",
                "generated_at": time.time(),
                "verification_schema_version": 1,
                "observed_evidence": {
                    "run_state": "finished",
                    "summary_metrics": {
                        "agentic_math/accuracy": {"ok": True, "value": 0.86},
                    },
                },
            }
        ),
    )
    completion_sha = sha256(completion)
    review_path = tmp_path / "review.json"
    source_audit = write_json(
        tmp_path / "source_audit.json",
        wandb_source_audit_payload(completion=completion),
    )
    source_audit_sha = sha256(source_audit)
    scope_attestation = write_json(
        tmp_path / "agentic_math-run-1.scope_attestation.json",
        {
            "schema_version": 1,
            "confirmed": True,
            "confirmed_by": "yuya",
            "confirmed_at": "2026-06-28T01:30:00+09:00",
            "confirmation": "This W&B run is the reviewed Agentic Math canary scope.",
            "review_path": str(review_path),
            "completion_path": str(completion),
            "completion_sha256": completion_sha,
            "benchmark": "agentic_math",
            "entity": "test-entity",
            "project": "test-project",
            "run_id": "run-1",
            "actual_cost_estimate": "$12.34",
            "provider_bill_reference": "openai-dashboard-2026-06-28",
            "source_audit_json": str(source_audit),
            "source_audit_sha256": source_audit_sha,
        },
    )
    payload = completed_review_payload(verify_wandb_completion=True)
    payload["runs"][0]["wandb_completion"] = [
        {
            "ok": True,
            "benchmark": "agentic_math",
            "entity": "test-entity",
            "project": "test-project",
            "run_id": "run-1",
            "path": str(completion),
            "sha256": completion_sha,
            "adopted_existing_result": True,
            "scope_attestation": {
                "schema_version": 1,
                "confirmed": True,
                "confirmed_by": "yuya",
                "confirmed_at": "2026-06-28T01:30:00+09:00",
                "confirmation": "This W&B run is the reviewed Agentic Math canary scope.",
                "review_path": str(review_path),
                "completion_path": str(completion),
                "completion_sha256": completion_sha,
                "benchmark": "agentic_math",
                "entity": "test-entity",
                "project": "test-project",
                "run_id": "run-1",
                "actual_cost_estimate": "$12.34",
                "provider_bill_reference": "openai-dashboard-2026-06-28",
                "source_attestation_json": str(scope_attestation),
                "source_attestation_sha256": sha256(scope_attestation),
                "source_audit_json": str(source_audit),
                "source_audit_sha256": source_audit_sha,
            },
        }
    ]
    review = write_json(review_path, payload)

    result = module.evaluate_paid_run_review_package(
        [review],
        require=True,
        wandb_completion_max_age_seconds=86_400,
    )

    assert result["ok"] is False
    entry = result["records"][0]["wandb_completion_entries"][0]
    assert entry["sync_dry_run_report_ok"] is False
    assert "missing W&B completion sync dry-run report path" in entry["verification_error"]


def test_paid_run_review_package_rejects_adopted_wandb_completion_source_review_sha_mismatch(tmp_path):
    module = load_module()
    completion = write_json(
        tmp_path / "agentic_math-run-1.json",
        add_wandb_run_metadata(
            {
                "ok": True,
                "benchmark": "agentic_math",
                "entity": "test-entity",
                "project": "test-project",
                "run_id": "run-1",
                "generated_at": time.time(),
                "verification_schema_version": 1,
                "observed_evidence": {
                    "run_state": "finished",
                    "summary_metrics": {
                        "agentic_math/accuracy": {"ok": True, "value": 0.86},
                    },
                },
            }
        ),
    )
    completion_sha = sha256(completion)
    source_review = write_json(
        tmp_path / "source_review.json",
        completed_review_payload(verify_wandb_completion=True),
    )
    source_audit = write_json(
        tmp_path / "source_audit.json",
        wandb_source_audit_payload(completion=completion),
    )
    source_audit_sha = sha256(source_audit)
    scope_attestation = write_json(
        tmp_path / "agentic_math-run-1.scope_attestation.json",
        {
            "schema_version": 1,
            "confirmed": True,
            "confirmed_by": "yuya",
            "confirmed_at": "2026-06-28T01:30:00+09:00",
            "confirmation": "This W&B run is the reviewed Agentic Math canary scope.",
            "review_path": str(source_review),
            "completion_path": str(completion),
            "completion_sha256": completion_sha,
            "benchmark": "agentic_math",
            "entity": "test-entity",
            "project": "test-project",
            "run_id": "run-1",
            "actual_cost_estimate": "$12.34",
            "provider_bill_reference": "openai-dashboard-2026-06-28",
            "source_audit_json": str(source_audit),
            "source_audit_sha256": source_audit_sha,
        },
    )
    scope_attestation_sha = sha256(scope_attestation)
    dry_run = write_json(
        tmp_path / "agentic_math-run-1.sync_dry_run.json",
        wandb_sync_dry_run_payload(
            source_review=source_review,
            completion=completion,
            completion_sha=completion_sha,
            source_attestation=scope_attestation,
            source_attestation_sha=scope_attestation_sha,
            source_audit=source_audit,
            source_audit_sha=source_audit_sha,
        ),
    )
    payload = completed_review_payload(verify_wandb_completion=True)
    payload["runs"][0]["wandb_completion"] = [
        {
            "ok": True,
            "benchmark": "agentic_math",
            "entity": "test-entity",
            "project": "test-project",
            "run_id": "run-1",
            "path": str(completion),
            "sha256": completion_sha,
            "adopted_existing_result": True,
            "sync_dry_run_report_json": str(dry_run),
            "sync_dry_run_source_review_json": str(source_review),
            "sync_dry_run_source_review_sha256": "0" * 64,
            "scope_attestation": {
                "schema_version": 1,
                "confirmed": True,
                "confirmed_by": "yuya",
                "confirmed_at": "2026-06-28T01:30:00+09:00",
                "confirmation": "This W&B run is the reviewed Agentic Math canary scope.",
                "review_path": str(source_review),
                "completion_path": str(completion),
                "completion_sha256": completion_sha,
                "benchmark": "agentic_math",
                "entity": "test-entity",
                "project": "test-project",
                "run_id": "run-1",
                "actual_cost_estimate": "$12.34",
                "provider_bill_reference": "openai-dashboard-2026-06-28",
                "source_attestation_json": str(scope_attestation),
                "source_attestation_sha256": scope_attestation_sha,
                "source_audit_json": str(source_audit),
                "source_audit_sha256": source_audit_sha,
            },
        }
    ]
    review = write_json(tmp_path / "review.json", payload)

    result = module.evaluate_paid_run_review_package(
        [review],
        require=True,
        wandb_completion_max_age_seconds=86_400,
    )

    assert result["ok"] is False
    entry = result["records"][0]["wandb_completion_entries"][0]
    assert entry["sync_dry_run_source_review_sha256_matches"] is False
    assert (
        "sync dry-run report source_review_sha256 does not match review entry"
        in entry["verification_error"]
    )


def test_paid_run_review_package_rejects_wandb_completion_parent_run_mismatch(tmp_path):
    module = load_module()
    completion = write_json(
        tmp_path / "agentic_math-other-run.json",
        {
            "ok": True,
            "benchmark": "agentic_math",
            "entity": "test-entity",
            "project": "test-project",
            "run_id": "other-run",
            "generated_at": time.time(),
            "verification_schema_version": 1,
            "observed_evidence": {
                "run_state": "finished",
                "summary_metrics": {
                    "agentic_math/accuracy": {"ok": True, "value": 0.86},
                },
            },
        },
    )
    payload = completed_review_payload(verify_wandb_completion=True)
    payload["runs"][0]["wandb_completion"] = [
        {
            "ok": True,
            "benchmark": "agentic_math",
            "entity": "test-entity",
            "project": "test-project",
            "run_id": "other-run",
            "path": str(completion),
            "sha256": sha256(completion),
        }
    ]
    review = write_json(tmp_path / "review.json", payload)

    result = module.evaluate_paid_run_review_package(
        [review],
        require=True,
        wandb_completion_max_age_seconds=86_400,
    )

    assert result["ok"] is False
    entry = result["records"][0]["wandb_completion_entries"][0]
    assert entry["parent_run_id"] == "run-1"
    assert entry["parent_run_id_matches"] is False
    assert "verifier run_id does not match parent review run" in entry["verification_error"]


def test_paid_run_review_package_requires_run_wandb_identity_when_verifying_wandb(tmp_path):
    module = load_module()
    completion = write_json(
        tmp_path / "agentic_math-run-1.json",
        {
            "ok": True,
            "benchmark": "agentic_math",
            "entity": "test-entity",
            "project": "test-project",
            "run_id": "run-1",
            "generated_at": time.time(),
            "verification_schema_version": 1,
            "observed_evidence": {
                "run_state": "finished",
                "summary_metrics": {
                    "agentic_math/accuracy": {"ok": True, "value": 0.86},
                },
            },
        },
    )
    payload = completed_review_payload(verify_wandb_completion=True)
    payload["runs"][0].pop("wandb_entity")
    payload["runs"][0].pop("wandb_project")
    payload["runs"][0]["wandb_completion"] = [
        {
            "ok": True,
            "benchmark": "agentic_math",
            "entity": "test-entity",
            "project": "test-project",
            "run_id": "run-1",
            "path": str(completion),
            "sha256": sha256(completion),
        }
    ]
    review = write_json(tmp_path / "review.json", payload)

    result = module.evaluate_paid_run_review_package(
        [review],
        require=True,
        wandb_completion_max_age_seconds=86_400,
    )

    assert result["ok"] is False
    assert any(
        "missing wandb_entity" in error
        for error in result["records"][0]["errors"]
    )
    assert any(
        "missing wandb_project" in error
        for error in result["records"][0]["errors"]
    )


def test_paid_run_review_package_rejects_legacy_wandb_completion_verifier(tmp_path):
    module = load_module()
    completion = write_json(
        tmp_path / "agentic_math-run-1.json",
        {
            "ok": True,
            "benchmark": "agentic_math",
            "run_id": "run-1",
            "generated_at": time.time(),
        },
    )
    payload = completed_review_payload(verify_wandb_completion=True)
    payload["runs"][0]["wandb_completion"] = [
        {
            "ok": True,
            "benchmark": "agentic_math",
            "run_id": "run-1",
            "path": str(completion),
        }
    ]
    review = write_json(tmp_path / "review.json", payload)

    result = module.evaluate_paid_run_review_package(
        [review],
        require=True,
        wandb_completion_max_age_seconds=86_400,
    )

    assert result["ok"] is False
    entry = result["records"][0]["wandb_completion_entries"][0]
    assert entry["schema_valid"] is False
    assert entry["observed_evidence_valid"] is False
    assert entry["verified"] is False


def test_paid_run_review_package_rejects_unverified_wandb_completion_entry(tmp_path):
    module = load_module()
    payload = completed_review_payload(verify_wandb_completion=True)
    payload["runs"][0]["wandb_completion"] = [
        {
            "ok": True,
            "benchmark": "agentic_math",
            "run_id": "run-1",
            "path": str(tmp_path / "missing.json"),
        }
    ]
    review = write_json(tmp_path / "review.json", payload)

    result = module.evaluate_paid_run_review_package(
        [review],
        require=True,
        wandb_completion_max_age_seconds=86_400,
    )

    assert result["ok"] is False
    assert result["status"] == "invalid_review_package"
    entry = result["records"][0]["wandb_completion_entries"][0]
    assert entry["verified"] is False
    assert any("W&B completion entry agentic_math/run-1 is not verified" in error for error in result["records"][0]["errors"])


def test_paid_run_review_package_rejects_incomplete_accounting(tmp_path):
    module = load_module()
    review = write_json(
        tmp_path / "review.json",
        completed_review_payload(actual_cost_estimate=""),
    )

    result = module.evaluate_paid_run_review_package([review], require=True)

    assert result["ok"] is False
    assert result["status"] == "invalid_review_package"
    assert "completed review is missing actual_cost_estimate" in result["records"][0]["errors"]


def test_paid_run_review_package_rejects_budget_model_mismatch(tmp_path):
    module = load_module()
    budget = write_json(
        tmp_path / "budget.json",
        {
            "schema_version": 1,
            "target_model": "openai-direct/expensive-unreviewed-model",
            "estimated_total_usd": {"low": 1.0, "mid": 2.0, "high": 3.0},
            "pricing_source_url": "https://openai.com/index/gpt-4-1/",
        },
    )
    review = write_json(
        tmp_path / "review.json",
        completed_review_payload(
            requires_paid_model_api=True,
            pre_run_budget_estimate={
                "required_before_paid_execution": True,
                "present": True,
                "valid": True,
                "path": str(budget),
                "sha256": sha256(budget),
                "target_model": "openai-direct/expensive-unreviewed-model",
                "target_models": ["openai-direct/expensive-unreviewed-model"],
                "selected_config_model_bindings": [
                    {
                        "config": "configs/taiwan_full/generated/config-taiwan-full-test.yaml",
                        "identifiers": ["openai-direct/gpt-4.1-mini-2025-04-14"],
                    }
                ],
                "selected_model_identifiers": ["openai-direct/gpt-4.1-mini-2025-04-14"],
                "target_model_matches_selected_config": False,
                "estimated_total_usd": {"low": 1.0, "mid": 2.0, "high": 3.0},
                "pricing_source_url": "https://openai.com/index/gpt-4-1/",
                "errors": [],
            },
        ),
    )

    result = module.evaluate_paid_run_review_package([review], require=True)

    assert result["ok"] is False
    assert result["status"] == "invalid_review_package"
    assert any(
        "pre_run_budget_estimate target_model does not match selected config model identifiers"
        in error
        for error in result["records"][0]["errors"]
    )


def test_paid_run_review_package_rejects_missing_provider_bill_reference(tmp_path):
    module = load_module()
    review = write_json(
        tmp_path / "review.json",
        completed_review_payload(provider_bill_reference=""),
    )

    result = module.evaluate_paid_run_review_package([review], require=True)

    assert result["ok"] is False
    assert result["status"] == "invalid_review_package"
    assert "completed review is missing provider_bill_reference" in result["records"][0]["errors"]


def test_paid_run_review_package_rejects_prepared_review(tmp_path):
    module = load_module()
    review = write_json(
        tmp_path / "review.json",
        {
            "status": "prepared",
            "phase": "agentic",
            "canary": True,
            "model_count": 1,
            "configs": ["config.yaml"],
            "requires_paid_model_api": True,
            "run_purpose": "planned canary",
            "expected_cost_band": "$10-$20",
            "execution_plan_path": "plan.json",
            "batch_manifest_path": "manifest.json",
            "post_run_cost_command": "estimate",
            "created_at": time.time(),
            "runs": [],
        },
    )

    result = module.evaluate_paid_run_review_package([review], require=True)

    assert result["ok"] is False
    assert result["status"] == "incomplete_reviews"
    assert "required Weave Agents completion entries" in result["next_action"]
    record = result["records"][0]
    assert record["configs"] == ["config.yaml"]
    assert record["requires_paid_model_api"] is True
    assert "review status is not completed: prepared" in record["errors"]


def test_build_report_surfaces_blockers(tmp_path):
    module = load_module()
    readiness = write_json(tmp_path / "readiness.json", readiness_payload(nemoclaw_ok=False, report_ok=True))
    gate = write_json(tmp_path / "canary.gate.json", {"ok": False, "status": "content_missing"})
    review = write_json(
        tmp_path / "review.json",
        {"status": "prepared", "phase": "agentic", "canary": True, "model_count": 1},
    )
    args = Namespace(
        output_root=tmp_path,
        readiness_json=[readiness],
        nemoclaw_setup_json=[],
        nemoclaw_installer_review_json=[tmp_path / "missing_installer_review.json"],
        weave_content_canary_gate=[gate],
        wandb_completion_json=[tmp_path / "missing_wandb_completion.json"],
        batch_review_json=[review],
        required_wandb_benchmark=["agentic_math"],
        required_wandb_run_id=[],
        required_wandb_run_id_all=None,
        wandb_completion_max_age_seconds=86_400,
        weave_content_canary_max_age_seconds=86_400,
        no_require_metadata_readiness=False,
        no_require_weave_content_canary=False,
        no_require_nemoclaw=False,
        no_require_wandb_completion=False,
        no_require_paid_run_review_package=False,
        no_require_one_model_canary=False,
    )

    report = module.build_report(args)

    assert report["schema_version"] == 1
    assert report["ok"] is False
    assert report["status"] == "not_ready"
    assert set(report["summary"]["blockers"]) == {
        "weave_content_canary",
        "nemoclaw_readiness",
        "wandb_completion",
        "paid_run_review_package",
        "one_model_full_canary",
    }
    assert [item["gate"] for item in report["remediation_plan"]] == [
        "weave_content_canary",
        "nemoclaw_readiness",
        "wandb_completion",
        "paid_run_review_package",
        "one_model_full_canary",
    ]
    nemoclaw_remediation = report["remediation_plan"][1]["commands"]
    assert any(
        "review_nemoclaw_installer.py" in command
        and "--lock-json scripts/setup/nemoclaw_installer_lock.json" in command
        and "--json temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json" in command
        and "--markdown temp/nemoclaw_installer_review_YYYYMMDDTHHMM.md" in command
        for command in nemoclaw_remediation
    )
    assert any(
        "install_nemoclaw.sh --install --onboard" in command
        and "--installer-lock-json scripts/setup/nemoclaw_installer_lock.json" in command
        and "--installer-sha256 REVIEWED_INSTALLER_SHA256" in command
        and "--installer-review-json temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json" in command
        and "--json temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.json" in command
        for command in nemoclaw_remediation
    )
    assert any(
        "verify_nemoclaw_post_install.py" in command
        and "--json temp/nemoclaw_post_install_verification_YYYYMMDDTHHMM.json" in command
        and "--markdown temp/nemoclaw_post_install_verification_YYYYMMDDTHHMM.md" in command
        and "--fail-on-failed" in command
        for command in nemoclaw_remediation
    )
    assert any(
        "run_taiwan_production_readiness_gate.py" in command
        and "--report-json outputs/taiwan_full_eval/taiwan_production_readiness_report.json" in command
        and "--fail-on-not-ready" in command
        for command in nemoclaw_remediation
    )
    assert any(
        "--manifest configs/taiwan_openai_canary_models.yaml" in command
        and "--phase agentic" in command
        and "--agentic-math-nemoclaw-sandbox nejumi-taiwan" in command
        and "--swebench-pro-nemoclaw-sandbox nejumi-taiwan" in command
        and "--swebench-pro-nemoclaw-checkout-transfer-mode copy" in command
        and "--require-nemoclaw-agentic-config" in command
        for command in report["remediation_plan"][3]["commands"]
    )
    assert not any("PHASE --run-purpose" in command for command in report["remediation_plan"][3]["commands"])
    paid_review_commands = report["remediation_plan"][3]["commands"]
    paid_batch_commands = [
        command for command in paid_review_commands if "run_taiwan_full_eval_batch.py" in command
    ]
    assert paid_batch_commands
    assert all(
        "--pre-run-budget-estimate-json outputs/taiwan_full_eval/openai_canary_budget_estimate.json"
        in command
        for command in paid_batch_commands
    )
    paid_execution_batch_commands = [
        command for command in paid_batch_commands if "--prepare-only" not in command
    ]
    assert paid_execution_batch_commands
    assert all(
        "--external-action-approval-source-packet-json "
        "outputs/taiwan_release_evidence/bundle_YYYYMMDDTHHMM/"
        "external_action_approval_packet.json"
        in command
        for command in paid_execution_batch_commands
    )
    assert all(
        "--external-action-approval-report-json "
        "temp/taiwan_external_action_approval_REVIEWED_YYYYMMDDTHHMM.verify.json"
        in command
        for command in paid_execution_batch_commands
    )
    one_model_commands = report["remediation_plan"][4]["commands"]
    one_model_batch_commands = [
        command for command in one_model_commands if "run_taiwan_full_eval_batch.py" in command
    ]
    assert one_model_batch_commands
    assert all(
        "--pre-run-budget-estimate-json outputs/taiwan_full_eval/openai_canary_budget_estimate.json"
        in command
        for command in one_model_batch_commands
    )
    one_model_execution_batch_commands = [
        command for command in one_model_batch_commands if "--prepare-only" not in command
    ]
    assert one_model_execution_batch_commands
    assert all(
        "--external-action-approval-source-packet-json "
        "outputs/taiwan_release_evidence/bundle_YYYYMMDDTHHMM/"
        "external_action_approval_packet.json"
        in command
        for command in one_model_execution_batch_commands
    )
    assert all(
        "--external-action-approval-report-json "
        "temp/taiwan_external_action_approval_REVIEWED_YYYYMMDDTHHMM.verify.json"
        in command
        for command in one_model_execution_batch_commands
    )
    assert any(
        "sync_wandb_completion_to_paid_review.py" in command
        and "--report-json temp/wandb_completion_agentic_math-RUN_ID.sync_dry_run.json" in command
        and "--in-place" not in command
        for command in paid_review_commands
    )
    assert any(
        "sync_wandb_completion_to_paid_review.py" in command
        and "--in-place --set-verify-wandb-completion" in command
        and "--validated-dry-run-report-json temp/wandb_completion_agentic_math-RUN_ID.sync_dry_run.json"
        in command
        for command in paid_review_commands
    )
    assert any(
        "sync_weave_agents_completion_to_paid_review.py" in command
        and "--report-json temp/weave_agents_completion_agentic-MODEL_SLUG.sync_dry_run.json" in command
        and "--in-place" not in command
        for command in paid_review_commands
    )
    assert any(
        "sync_weave_agents_completion_to_paid_review.py" in command
        and "--in-place --set-verify-weave-agents" in command
        and "--validated-dry-run-report-json temp/weave_agents_completion_agentic-MODEL_SLUG.sync_dry_run.json" in command
        for command in paid_review_commands
    )
    paid_review_gate = next(
        gate for gate in report["gates"] if gate["name"] == "paid_run_review_package"
    )
    weave_requirements = paid_review_gate["completion_requirements"][
        "weave_agents_completion_verifier_requirements"
    ]
    assert weave_requirements["required_checks"] == [
        "trace_final_answer_order",
        "trace_order",
        "trace_timestamp_quality",
        "trace_user_message_order",
    ]
    assert weave_requirements["required_evidence.input_message_required"] is True
    assert weave_requirements["required_evidence.trace_timestamp_quality_required"] is True
    assert weave_requirements["required_evidence.trace_final_answer_order_required"] is True
    assert (
        weave_requirements["required_text_capture_when"]
        == "required_evidence.required_texts is non-empty"
    )


def test_build_report_summary_includes_benchmark_evidence_matrix(tmp_path):
    module = load_module()
    math_completion = write_json(
        tmp_path / "math-completion.json",
        {
            "ok": True,
            "benchmark": "agentic_math",
            "entity": "test-entity",
            "project": "test-project",
            "run_id": "run-math",
            "generated_at": time.time(),
            "verification_schema_version": 1,
            "observed_evidence": {
                "run_state": "finished",
                "summary_metrics": {
                    "agentic_math/accuracy": {"ok": True, "value": 0.86},
                },
            },
        },
    )
    review = write_json(
        tmp_path / "review.json",
        {
            "status": "completed",
            "phase": "full",
            "canary": True,
            "model_count": 1,
            "runs": [{"wandb_run_id": "run-math"}],
        },
    )
    args = Namespace(
        output_root=tmp_path,
        readiness_json=[],
        nemoclaw_setup_json=[],
        nemoclaw_installer_review_json=[tmp_path / "missing_installer_review.json"],
        weave_content_canary_gate=[],
        wandb_completion_json=[math_completion],
        batch_review_json=[review],
        required_wandb_benchmark=["agentic_math", "agentic_swe"],
        required_wandb_run_id=["agentic_math=run-math", "agentic_swe=run-swe"],
        required_wandb_run_id_all=None,
        wandb_completion_max_age_seconds=86_400,
        weave_content_canary_max_age_seconds=86_400,
        no_require_metadata_readiness=True,
        no_require_weave_content_canary=True,
        no_require_nemoclaw=True,
        no_require_wandb_completion=False,
        no_require_paid_run_review_package=True,
        no_require_one_model_canary=False,
    )

    report = module.build_report(args)
    assert report["schema_version"] == 1
    evidence = {
        row["benchmark"]: row
        for row in report["summary"]["benchmark_evidence"]
    }

    assert evidence["agentic_math"]["expected_run_id"] == "run-math"
    assert evidence["agentic_math"]["required"] is True
    assert evidence["agentic_math"]["standalone_wandb_completion"]["ok"] is True
    assert evidence["agentic_math"]["standalone_wandb_completion"]["status"] == "passed"
    assert evidence["agentic_math"]["review_wandb_completion"]["ok"] is False
    assert evidence["agentic_math"]["review_wandb_completion"]["status"] == "missing_review_entry"
    assert evidence["agentic_math"]["completion_proven"] is False
    assert evidence["agentic_swe"]["expected_run_id"] == "run-swe"
    assert evidence["agentic_swe"]["required"] is True
    assert evidence["agentic_swe"]["standalone_wandb_completion"]["ok"] is False
    assert evidence["agentic_swe"]["standalone_wandb_completion"]["status"] == "missing"
    assert evidence["agentic_swe"]["review_wandb_completion"]["status"] == "missing_review_entry"


def test_remediation_plan_omits_passed_gates(tmp_path):
    module = load_module()
    passing_gate = {
        "name": "wandb_completion",
        "ok": True,
        "blocking": True,
        "status": "passed",
        "remediation_commands": ["should not appear"],
    }
    failing_gate = {
        "name": "weave_content_canary",
        "ok": False,
        "blocking": True,
        "status": "failed",
        "next_action": "rerun",
        "remediation_commands": ["run canary"],
    }

    result = module.remediation_plan([passing_gate, failing_gate])

    assert result == [
        {
            "gate": "weave_content_canary",
            "status": "failed",
            "next_action": "rerun",
            "commands": ["run canary"],
        }
    ]


def test_filter_canary_readiness_paths_excludes_own_reports(tmp_path):
    module = load_module()
    paths = [
        tmp_path / "glm52_canary_readiness.json",
        tmp_path / "taiwan_production_readiness_report_20260627.json",
        tmp_path / "production_readiness_report.json",
    ]

    result = module.filter_canary_readiness_paths(paths)

    assert result == [tmp_path / "glm52_canary_readiness.json"]


def test_required_wandb_benchmark_argument_replaces_default_list():
    module = load_module()

    args = module.parse_args(["--required-wandb-benchmark", "agentic_math"])

    assert args.required_wandb_benchmark == ["agentic_math"]


def test_weave_content_canary_max_age_negative_disables_freshness():
    module = load_module()

    args = module.parse_args(["--weave-content-canary-max-age-seconds", "-1"])

    assert args.weave_content_canary_max_age_seconds is None


def test_required_wandb_run_id_argument_parses_mapping():
    module = load_module()

    args = module.parse_args(["--required-wandb-run-id", "agentic_math=f1veetyb"])
    result = module.parse_required_wandb_run_ids(args.required_wandb_run_id)

    assert result == {"agentic_math": "f1veetyb"}


def test_required_wandb_run_id_all_applies_to_selected_benchmarks():
    module = load_module()

    args = module.parse_args(
        [
            "--required-wandb-benchmark",
            "agentic_math",
            "--required-wandb-benchmark",
            "taiwan_full",
            "--required-wandb-run-id-all",
            "run-1",
        ]
    )
    result = module.parse_required_wandb_run_ids(
        args.required_wandb_run_id,
        required_benchmarks=args.required_wandb_benchmark,
        all_run_id=args.required_wandb_run_id_all,
    )

    assert result == {"agentic_math": "run-1", "taiwan_full": "run-1"}


def test_required_wandb_run_id_all_rejects_conflicting_specific_mapping():
    module = load_module()

    args = module.parse_args(
        [
            "--required-wandb-benchmark",
            "agentic_math",
            "--required-wandb-run-id-all",
            "run-1",
            "--required-wandb-run-id",
            "agentic_math=run-2",
        ]
    )

    try:
        module.parse_required_wandb_run_ids(
            args.required_wandb_run_id,
            required_benchmarks=args.required_wandb_benchmark,
            all_run_id=args.required_wandb_run_id_all,
        )
    except ValueError as exc:
        assert "conflicting W&B run id requirements" in str(exc)
    else:
        raise AssertionError("expected conflicting W&B run id requirements")


def test_existing_results_formalization_accepts_passing_audit(tmp_path):
    module = load_module()
    audit = write_json(
        tmp_path / "taiwan_existing_results_audit.json",
        {
            "ok": True,
            "status": "passed",
            "generated_at": time.time(),
            "summary": {
                "record_count": 1,
                "complete_local_count": 1,
                "formalized_wandb_complete_count": 1,
                "unformalized_complete_count": 0,
                "partial_or_probe_count": 0,
            },
            "unformalized_complete_records": [],
        },
    )

    result = module.evaluate_existing_results_formalization([audit], require=True)

    assert result["ok"] is True
    assert result["status"] == "passed"
    assert result["latest_audit"]["formalized_wandb_complete_count"] == 1


def test_existing_results_formalization_rejects_unformalized_complete_result(tmp_path):
    module = load_module()
    audit = write_json(
        tmp_path / "taiwan_existing_results_audit.json",
        {
            "ok": False,
            "status": "unformalized_complete_results",
            "generated_at": time.time(),
            "summary": {
                "record_count": 1,
                "complete_local_count": 1,
                "formalized_wandb_complete_count": 0,
                "unformalized_complete_count": 1,
                "partial_or_probe_count": 0,
            },
            "unformalized_complete_records": [
                {
                    "benchmark": "agentic_math",
                    "model_slug": "model-a",
                    "formalization_status": "local_complete_needs_wandb_relog",
                }
            ],
        },
    )

    result = module.evaluate_existing_results_formalization([audit], require=True)

    assert result["ok"] is False
    assert result["status"] == "unformalized_complete_results"
    assert result["latest_audit"]["unformalized_complete_count"] == 1
    assert any("audit_taiwan_existing_results.py" in command for command in result["remediation_commands"])
