import importlib.util
import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tools" / "build_taiwan_release_evidence_bundle.py"


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module():
    spec = importlib.util.spec_from_file_location(SCRIPT.stem, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[SCRIPT.stem] = module
    spec.loader.exec_module(module)
    return module


def test_release_evidence_bundle_copies_report_references(tmp_path):
    taxonomy = tmp_path / "nejumi45_taiwan.yaml"
    taxonomy.write_text("version: test\nunits: []\n", encoding="utf-8")
    completion = write_json(
        tmp_path / "wandb_completion" / "agentic_math-run.json",
        {
            "ok": True,
            "benchmark": "agentic_math",
            "run_id": "run-1",
            "entity": "test-entity",
            "project": "test-project",
            "generated_at": 1,
            "taxonomy_path": str(taxonomy),
        },
    )
    weave_completion = write_json(
        tmp_path / "weave_agents_completion" / "agentic-run.json",
        {
            "ok": True,
            "agent_name": "nejumi-taiwan-openclaw",
            "verification_schema_version": 1,
            "generated_at": 1,
            "latest_trace_id": "trace-1",
            "checks": [{"name": "agent_present", "ok": True}],
        },
    )
    pre_run_budget = write_json(
        tmp_path / "openai_canary_budget_estimate.json",
        {
            "schema_version": 1,
            "target_model": "openai-direct/gpt-4.1-mini-2025-04-14",
            "estimated_total_usd": {"low": 10.0, "mid": 12.0, "high": 20.0},
            "pricing_source_url": "https://openai.com/index/gpt-4-1/",
        },
    )
    review = write_json(
        tmp_path / "canary_agentic_paid_run_review.json",
        {
            "status": "completed",
            "phase": "agentic",
            "pre_run_budget_estimate": {
                "path": str(pre_run_budget),
                "valid": True,
            },
        },
    )
    confirmed_scope_attestation = write_json(
        tmp_path / "agentic_math-run-1.confirmed_scope_attestation.json",
        {
            "schema_version": 1,
            "confirmed": True,
            "confirmed_by": "yuya",
            "confirmed_at": "2026-06-28T01:30:00+09:00",
            "confirmation": "This W&B run is the reviewed Agentic Math canary scope.",
            "review_path": str(review),
            "completion_path": str(completion),
            "benchmark": "agentic_math",
            "entity": "test-entity",
            "project": "test-project",
            "run_id": "run-1",
            "actual_cost_estimate": "$12.34",
            "provider_bill_reference": "openai-dashboard-2026-06-28",
        },
    )
    nemoclaw_install_log = tmp_path / "nemoclaw_setup_check.install.log"
    nemoclaw_install_log.write_text("install failed\n", encoding="utf-8")
    nemoclaw = write_json(
        tmp_path / "nemoclaw_setup_check.json",
        {
            "ok": False,
            "commands": {"nemoclaw": {"available": False}},
            "operation_results": {
                "install": {
                    "requested": True,
                    "attempted": True,
                    "returncode": 2,
                    "log_path": str(nemoclaw_install_log),
                },
                "onboard": {
                    "requested": True,
                    "attempted": False,
                    "skipped": True,
                    "returncode": None,
                    "log_path": "",
                },
            },
        },
    )
    paid_review_check = write_json(
        tmp_path / "taiwan_paid_run_review_check.json",
        {
            "ok": False,
            "status": "not_ready",
            "summary": {
                "gate_count": 1,
                "blocker_count": 1,
                "blockers": ["paid_run_review_package"],
            },
            "requirements": {
                "required_wandb_benchmarks": ["agentic_math"],
                "wandb_completion_max_age_seconds": 86400,
                "weave_agents_completion_max_age_seconds": 86400,
            },
            "review_completion_requirements": {
                "completed_review_required_fields": [
                    "ended_at",
                    "actual_cost_estimate",
                    "provider_bill_reference",
                    "runs",
                ],
                "wandb_completion_verifier_requirements": {
                    "max_age_seconds": 86400,
                    "observed_evidence.run_state": "finished",
                },
                "weave_agents_completion_verifier_requirements": {
                    "max_age_seconds": 86400,
                    "latest_trace_id_present": True,
                },
            },
            "gates": [
                {
                    "name": "paid_run_review_package",
                    "ok": False,
                    "blocking": True,
                    "status": "incomplete_reviews",
                    "detail": "incomplete",
                    "next_action": "complete review",
                    "records": [
                        {
                            "path": str(review),
                            "ok": True,
                            "status": "completed",
                            "phase": "agentic",
                            "canary": True,
                            "model_count": 1,
                            "run_purpose_present": True,
                            "expected_cost_band_present": True,
                            "actual_cost_estimate_present": True,
                            "provider_bill_reference_present": True,
                            "run_count": 1,
                            "verify_wandb_completion": True,
                            "wandb_completion_entries": [
                                {
                                    "benchmark": "agentic_math",
                                    "entity": "test-entity",
                                    "project": "test-project",
                                    "run_id": "run-1",
                                    "ok": True,
                                    "path": str(completion),
                                    "verified": True,
                                    "adopted_existing_result": True,
                                    "scope_attestation_valid": True,
                                    "scope_attestation": {
                                        "schema_version": 1,
                                        "confirmed": True,
                                        "confirmed_by": "yuya",
                                        "confirmed_at": "2026-06-28T01:30:00+09:00",
                                        "confirmation": "This W&B run is the reviewed Agentic Math canary scope.",
                                        "review_path": str(review),
                                        "completion_path": str(completion),
                                        "benchmark": "agentic_math",
                                        "entity": "test-entity",
                                        "project": "test-project",
                                        "run_id": "run-1",
                                        "actual_cost_estimate": "$12.34",
                                        "provider_bill_reference": "openai-dashboard-2026-06-28",
                                        "source_attestation_json": str(confirmed_scope_attestation),
                                    },
                                }
                            ],
                            "verify_weave_agents": True,
                            "weave_agents_completion_entries": [],
                            "errors": [],
                        },
                        {
                            "path": str(review),
                            "ok": False,
                            "status": "prepared",
                            "phase": "agentic",
                            "canary": True,
                            "model_count": 1,
                            "run_purpose_present": True,
                            "expected_cost_band_present": True,
                            "actual_cost_estimate_present": False,
                            "provider_bill_reference_present": False,
                            "run_count": 0,
                            "verify_wandb_completion": True,
                            "wandb_completion_entries": [],
                            "verify_weave_agents": True,
                            "weave_agents_completion_entries": [],
                            "errors": ["review status is not completed: prepared"],
                        }
                    ],
                    "blocking_records": [
                        {
                            "path": str(review),
                            "ok": False,
                            "status": "prepared",
                            "phase": "agentic",
                            "canary": True,
                            "model_count": 1,
                            "run_purpose_present": True,
                            "expected_cost_band_present": True,
                            "actual_cost_estimate_present": False,
                            "provider_bill_reference_present": False,
                            "run_count": 0,
                            "verify_wandb_completion": True,
                            "wandb_completion_entries": [],
                            "verify_weave_agents": True,
                            "weave_agents_completion_entries": [],
                            "errors": ["review status is not completed: prepared"],
                        }
                    ],
                }
            ],
        },
    )
    paid_review_check_md = tmp_path / "taiwan_paid_run_review_check.md"
    paid_review_check_md.write_text("# check\n", encoding="utf-8")
    relog_plan = write_json(
        tmp_path / "agentic_math_gpt_5_5_relog_plan.json",
        {
            "schema_version": 1,
            "ok": True,
            "will_write_wandb": False,
            "benchmark": "agentic_math",
            "entity": "test-entity",
            "project": "test-project",
            "source": {
                "results_dir": "outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw",
                "summary_json": "outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw/summary.json",
                "results_jsonl": "outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw/results.jsonl",
            },
            "would_log": {
                "tables": {"agentic_math_output_table": 100},
                "artifact": {"aliases": ["latest", "production"]},
            },
            "post_log_verifier_command_template": (
                "uv run python scripts/tools/verify_taiwan_wandb_completion.py "
                "--run-id RUN_ID_AFTER_WANDB_LOG --benchmark agentic_math"
            ),
        },
    )
    existing_results_audit = write_json(
        tmp_path / "taiwan_existing_results_audit.json",
        {
            "ok": True,
            "status": "passed",
            "generated_at": 1,
            "output_root": "outputs/taiwan_full_eval",
            "completion_dir": "outputs/taiwan_full_eval/wandb_completion",
            "summary": {
                "record_count": 2,
                "complete_local_count": 1,
                "formalized_wandb_complete_count": 1,
                "unformalized_complete_count": 0,
                "partial_or_probe_count": 1,
                "wandb_completion_json_count": 1,
            },
            "remediation_commands": [],
            "formalized_records": [
                {
                    "benchmark": "agentic_math",
                    "model_slug": "gpt-5_5",
                    "model": "openai/gpt-5.5",
                    "run_kind": "final",
                    "result_dir": "outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw",
                    "summary_path": "outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw/summary.json",
                    "results_path": "outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw/results.jsonl",
                    "complete_local": True,
                    "formalization_status": "formalized_wandb_complete",
                    "expected_total": 100,
                    "row_count": 100,
                    "partial_row_count": 0,
                    "relog_dry_run_plan_json": str(relog_plan),
                    "relog_dry_run_command": (
                        "uv run python scripts/tools/log_agentic_math_results_to_wandb.py "
                        "--results-dir outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw "
                        "--model-name openai/gpt-5.5 --dry-run --plan-json "
                        f"{relog_plan}"
                    ),
                    "wandb_completion": {
                        "path": str(completion),
                        "entity": "test-entity",
                        "project": "test-project",
                        "run_id": "run-1",
                        "run_name": "agentic_math_run",
                        "generated_at": 1,
                        "verification_schema_version": 1,
                        "schema_current": True,
                        "observed_evidence_present": True,
                    },
                    "warnings": [],
                    "errors": [],
                }
            ],
            "unformalized_complete_records": [],
            "partial_records": [
                {
                    "benchmark": "agentic_swe",
                    "model_slug": "gpt-5_5",
                    "run_kind": "final",
                    "result_dir": "outputs/taiwan_full_eval/swebench_pro/gpt-5_5",
                    "complete_local": False,
                    "formalization_status": "partial_not_reloggable",
                    "expected_total": 80,
                    "row_count": 3,
                    "patch_count": 3,
                    "patch_record_count": 3,
                    "warnings": [],
                    "errors": [],
                }
            ],
            "wandb_completion_records": [
                {
                    "path": str(completion),
                    "ok": True,
                    "benchmark": "agentic_math",
                    "entity": "test-entity",
                    "project": "test-project",
                    "run_id": "run-1",
                    "run_name": "agentic_math_run",
                    "verification_schema_version": 1,
                    "schema_current": True,
                    "observed_evidence_present": True,
                    "metrics": {"total": 100},
                }
            ],
        },
    )
    existing_results_audit_md = tmp_path / "taiwan_existing_results_audit.md"
    existing_results_audit_md.write_text("# existing results\n", encoding="utf-8")
    existing_results_audit_sha = sha256(existing_results_audit)
    wandb_adoption_attestation = write_json(
        tmp_path / "agentic_math-run-1.scope_attestation.json",
        {
            "schema_version": 1,
            "confirmed": False,
            "confirmed_by": "REVIEWER",
            "confirmed_at": "YYYY-MM-DDTHH:MM:SS+09:00",
            "confirmation": "This W&B run is the reviewed canary scope for agentic_math.",
            "review_path": "outputs/taiwan_full_eval/PHASE_paid_run_review.json",
            "completion_path": str(completion),
            "benchmark": "agentic_math",
            "entity": "test-entity",
            "project": "test-project",
            "run_id": "run-1",
            "actual_cost_estimate": "$ACTUAL_OR_BILLING_ESTIMATE",
            "provider_bill_reference": "BILL_OR_DASHBOARD_REFERENCE",
        },
    )
    wandb_adoption_render_report = tmp_path / "agentic_math-run-1.scope_attestation.render.json"
    wandb_adoption_preflight = tmp_path / "agentic_math-run-1.scope_preflight.json"
    wandb_adoption_sync_dry_run = tmp_path / "agentic_math-run-1.sync_dry_run.json"
    wandb_adoption_draft = write_json(
        tmp_path / "taiwan_wandb_adoption_draft.json",
        {
            "schema_version": 1,
            "ok": True,
            "status": "candidates_pending_scope_confirmation",
            "generated_at": 1,
            "source_audit_json": str(existing_results_audit),
            "source_audit_sha256": existing_results_audit_sha,
            "review_json": "outputs/taiwan_full_eval/PHASE_paid_run_review.json",
            "candidate_count": 1,
            "scope_attestation_template_count": 1,
            "scope_attestation_templates": [
                {
                    "benchmark": "agentic_math",
                    "run_id": "run-1",
                    "path": str(wandb_adoption_attestation),
                    "preflight_report_json": str(wandb_adoption_preflight),
                }
            ],
            "requires_human_scope_confirmation": True,
            "required_human_fields": [
                "target_review_json",
                "scope_confirmed_by",
                "scope_confirmed_at",
                "scope_confirmation",
                "actual_cost_estimate",
                "provider_bill_reference",
            ],
            "candidates": [
                {
                    "benchmark": "agentic_math",
                    "model_slug": "gpt-5_5",
                    "wandb_entity": "test-entity",
                    "wandb_project": "test-project",
                    "wandb_run_id": "run-1",
                    "wandb_completion_json": str(completion),
                    "source_audit_json": str(existing_results_audit),
                    "source_audit_sha256": existing_results_audit_sha,
                    "target_review_json": "outputs/taiwan_full_eval/PHASE_paid_run_review.json",
                    "scope_attestation_template_json": str(wandb_adoption_attestation),
                    "scope_attestation_required": True,
                    "required_human_fields": [
                        "scope_attestation_json.confirmed",
                        "scope_attestation_json.confirmed_by",
                        "scope_attestation_json.confirmed_at",
                        "scope_attestation_json.confirmation",
                        "scope_attestation_json.completion_sha256",
                        "scope_attestation_json.actual_cost_estimate",
                        "scope_attestation_json.provider_bill_reference",
                    ],
                    "run_metadata_valid": True,
                    "run_metadata_errors": [],
                    "sync_ready": True,
                    "scope_attestation_render_report_json": str(wandb_adoption_render_report),
                    "scope_attestation_render_command": (
                        "uv run python scripts/tools/render_wandb_scope_attestation.py "
                        f"--template-json {wandb_adoption_attestation} "
                        f"--output-json {wandb_adoption_attestation} "
                        "--confirmed-by REVIEWER_NAME "
                        "--confirmed-at YYYY-MM-DDTHH:MM:SS+09:00 "
                        "--confirmation CONFIRM_THIS_WANDB_RUN_IS_THE_REVIEWED_SCOPE "
                        "--actual-cost-estimate ACTUAL_COST_USD "
                        "--provider-bill-reference PROVIDER_BILL_REFERENCE "
                        f"--report-json {wandb_adoption_render_report} "
                        f"--preflight-report-json {wandb_adoption_preflight} "
                        f"--sync-dry-run-report-json {wandb_adoption_sync_dry_run}"
                    ),
                    "scope_attestation_preflight_report_json": str(wandb_adoption_preflight),
                    "scope_attestation_preflight_command": (
                        "uv run python scripts/tools/verify_wandb_scope_attestation.py "
                        "--review-json outputs/taiwan_full_eval/PHASE_paid_run_review.json "
                        f"--completion-json {completion} "
                        f"--scope-attestation-json {wandb_adoption_attestation} "
                        f"--json {wandb_adoption_preflight}"
                    ),
                    "sync_dry_run_report_json": str(wandb_adoption_sync_dry_run),
                    "sync_dry_run_command": (
                        "uv run python scripts/tools/sync_wandb_completion_to_paid_review.py "
                        "--review-json outputs/taiwan_full_eval/PHASE_paid_run_review.json "
                        f"--completion-json {completion} "
                        "--set-verify-wandb-completion --top-level --adopt-existing-result "
                        f"--scope-attestation-json {wandb_adoption_attestation} "
                        f"--report-json {wandb_adoption_sync_dry_run}"
                    ),
                    "sync_apply_command": (
                        "uv run python scripts/tools/sync_wandb_completion_to_paid_review.py "
                        "--review-json outputs/taiwan_full_eval/PHASE_paid_run_review.json "
                        f"--completion-json {completion} "
                        "--in-place --set-verify-wandb-completion --top-level --adopt-existing-result "
                        f"--scope-attestation-json {wandb_adoption_attestation} "
                        f"--validated-dry-run-report-json {wandb_adoption_sync_dry_run}"
                    ),
                    "sync_command": "uv run python scripts/tools/sync_wandb_completion_to_paid_review.py --adopt-existing-result --scope-attestation-json "
                    + str(wandb_adoption_attestation)
                    + " --validated-dry-run-report-json "
                    + str(wandb_adoption_sync_dry_run),
                    "warnings": ["confirm scope"],
                }
            ],
        },
    )
    wandb_adoption_draft_md = tmp_path / "taiwan_wandb_adoption_draft.md"
    wandb_adoption_draft_md.write_text("# adoption draft\n", encoding="utf-8")
    nemoclaw_config = tmp_path / "config-taiwan-nemoclaw.yaml"
    nemoclaw_config.write_text("agentic_math:\n  nemoclaw_sandbox: nejumi-taiwan\n", encoding="utf-8")
    nemoclaw_adoption_check = write_json(
        tmp_path / "taiwan_nemoclaw_adoption_check.json",
        {
            "ok": False,
            "status": "not_installed",
            "generated_at": 1,
            "sandbox": "nejumi-taiwan",
            "adoption_decision": {
                "recommendation": "conditional_adopt_for_agentic_math",
                "ready_for_use": False,
                "scope": "agentic_math_only",
                "sandbox": "nejumi-taiwan",
                "design_ready": True,
                "runtime_blockers": ["setup_installed"],
                "design_blockers": [],
                "other_blockers": [],
                "blockers": ["setup_installed"],
                "rationale": "Design and policy criteria pass, but runtime is missing.",
                "next_action": "Install/onboard NeMoClaw, then rerun verification.",
            },
            "summary": {
                "criterion_count": 2,
                "blocker_count": 1,
                "blockers": ["setup_installed"],
                "adoption_recommendation": "conditional_adopt_for_agentic_math",
                "ready_for_use": False,
                "adoption_scope": "agentic_math_only",
                "setup_runtime": {
                    "host_prerequisites_ok": True,
                    "runtime_installed": False,
                    "missing_required_commands": ["nemoclaw", "openshell"],
                    "missing_components": ["nemoclaw", "openshell"],
                },
            },
            "setup_paths": [str(nemoclaw)],
            "agentic_config_paths": [str(nemoclaw_config)],
            "criteria": [
                {
                    "name": "setup_installed",
                    "ok": False,
                    "status": "missing_or_not_ready",
                    "detail": "missing nemoclaw",
                    "next_action": "install",
                    "evidence_paths": [str(nemoclaw)],
                },
                {
                    "name": "agentic_math_config",
                    "ok": True,
                    "status": "passed",
                    "detail": "ready",
                    "next_action": "use config",
                    "evidence_paths": [str(nemoclaw_config)],
                },
            ],
        },
    )
    nemoclaw_adoption_check_md = tmp_path / "taiwan_nemoclaw_adoption_check.md"
    nemoclaw_adoption_check_md.write_text("# adoption\n", encoding="utf-8")
    post_install_preflight = write_json(
        tmp_path / "nemoclaw_protocol_preflight.json",
        {"ok": False, "status": "missing"},
    )
    post_install_readiness = write_json(
        tmp_path / "nemoclaw_canary_readiness.json",
        {"ok": False, "status": "failed"},
    )
    post_install_check = write_json(
        tmp_path / "nemoclaw_post_install_verification.json",
        {
            "ok": False,
            "status": "failed",
            "will_launch_model_inference": False,
            "will_query_wandb": False,
            "will_install_or_onboard": False,
            "command_safety": {
                "ok": True,
                "forbidden_tokens": [
                    "--install",
                    "--onboard",
                    "--upload",
                    "--wandb",
                    "--yes-i-accept-third-party-software",
                ],
                "forbidden_prefixes": [
                    "--upload",
                    "--wandb",
                    "ANTHROPIC_API_KEY=",
                    "GEMINI_API_KEY=",
                    "GOOGLE_API_KEY=",
                    "OPENAI_API_KEY=",
                    "OPENROUTER_",
                    "OPENROUTER_API_KEY=",
                    "WANDB_",
                    "WEAVE_",
                    "XAI_API_KEY=",
                ],
                "forbidden_markers": ["openrouter", "wandb", "weave"],
                "required_step_tokens": {
                    "setup_check": ["install_nemoclaw.sh", "--check-only", "--json"],
                    "protocol_preflight": ["run_openclaw_agent_protocol.py", "preflight"],
                    "canary_readiness": [
                        "check_taiwan_canary_readiness.py",
                        "--require-nemoclaw",
                        "--json",
                    ],
                    "adoption_check": [
                        "check_taiwan_nemoclaw_adoption.py",
                        "--setup-json",
                        "--readiness-json",
                        "--json",
                        "--markdown",
                    ],
                },
                "forbidden_token_count": 0,
                "missing_required_token_count": 0,
                "missing_command_count": 0,
                "records": [
                    {
                        "name": "setup_check",
                        "ok": True,
                        "forbidden_tokens": [],
                        "missing_required_tokens": [],
                    },
                    {
                        "name": "protocol_preflight",
                        "ok": True,
                        "forbidden_tokens": [],
                        "missing_required_tokens": [],
                    },
                    {
                        "name": "canary_readiness",
                        "ok": True,
                        "forbidden_tokens": [],
                        "missing_required_tokens": [],
                    },
                    {
                        "name": "adoption_check",
                        "ok": True,
                        "forbidden_tokens": [],
                        "missing_required_tokens": [],
                    },
                ],
            },
            "outputs": {
                "setup_json": str(nemoclaw),
                "preflight_json": str(post_install_preflight),
                "readiness_json": str(post_install_readiness),
                "adoption_json": str(nemoclaw_adoption_check),
                "adoption_markdown": str(nemoclaw_adoption_check_md),
            },
            "steps": [
                {
                    "name": "setup_check",
                    "ok": False,
                    "returncode": 1,
                    "returncode_ok": False,
                    "payload_ok": False,
                    "timed_out": False,
                    "payload_status": None,
                    "command": [
                        "scripts/setup/install_nemoclaw.sh",
                        "--check-only",
                        "--json",
                        str(nemoclaw),
                    ],
                    "output_json": str(nemoclaw),
                },
                {
                    "name": "protocol_preflight",
                    "ok": False,
                    "returncode": 1,
                    "returncode_ok": False,
                    "payload_ok": False,
                    "timed_out": False,
                    "payload_status": "missing",
                    "command": [
                        "uv",
                        "run",
                        "python",
                        "scripts/tools/run_openclaw_agent_protocol.py",
                        "preflight",
                    ],
                    "output_json": str(post_install_preflight),
                },
                {
                    "name": "canary_readiness",
                    "ok": False,
                    "returncode": 1,
                    "returncode_ok": False,
                    "payload_ok": False,
                    "timed_out": False,
                    "payload_status": "failed",
                    "command": [
                        "uv",
                        "run",
                        "python",
                        "scripts/tools/check_taiwan_canary_readiness.py",
                        "--require-nemoclaw",
                        "--json",
                        str(post_install_readiness),
                    ],
                    "output_json": str(post_install_readiness),
                },
                {
                    "name": "adoption_check",
                    "ok": False,
                    "returncode": 0,
                    "returncode_ok": True,
                    "payload_ok": False,
                    "timed_out": False,
                    "payload_status": "not_installed",
                    "command": [
                        "uv",
                        "run",
                        "python",
                        "scripts/tools/check_taiwan_nemoclaw_adoption.py",
                        "--setup-json",
                        str(nemoclaw),
                        "--readiness-json",
                        str(post_install_readiness),
                        "--json",
                        str(nemoclaw_adoption_check),
                        "--markdown",
                        str(nemoclaw_adoption_check_md),
                    ],
                    "output_json": str(nemoclaw_adoption_check),
                },
            ],
        },
    )
    post_install_check_md = tmp_path / "nemoclaw_post_install_verification.md"
    post_install_check_md.write_text("# post install\n", encoding="utf-8")
    installer_sha = "a4ebc5710dfd8b10035968fd25562773ad56ec4c73ecf9bfe5c78c77724000e7"
    installer_lock = write_json(
        tmp_path / "nemoclaw_installer_lock.json",
        {
            "schema_version": 1,
            "installer_url": "https://www.nvidia.com/nemoclaw.sh",
            "install_ref": "lkg",
            "sha256": installer_sha,
            "size_bytes": 6356,
            "will_execute_installer": False,
            "will_install_or_onboard": False,
            "will_launch_model_inference": False,
            "will_query_wandb": False,
        },
    )
    installer_review = write_json(
        tmp_path / "nemoclaw_installer_review.json",
        {
            "schema_version": 1,
            "ok": True,
            "status": "reviewed",
            "generated_at": "2026-06-28T01:32:47Z",
            "installer_url": "https://www.nvidia.com/nemoclaw.sh",
            "install_ref": "lkg",
            "lock_json": str(installer_lock),
            "lock_verified": True,
            "expected_sha256": installer_sha,
            "sha256": installer_sha,
            "size_bytes": 6356,
            "will_execute_installer": False,
            "will_install_or_onboard": False,
            "will_launch_model_inference": False,
            "will_query_wandb": False,
        },
    )
    installer_review_md = tmp_path / "nemoclaw_installer_review.md"
    installer_review_md.write_text("# installer review\n", encoding="utf-8")
    nemoclaw_install_json = tmp_path / "nemoclaw_install_onboard.json"
    failed_weave_verifier = write_json(
        tmp_path / "weave_agents_verify_run_1_failed.json",
        {
            "ok": False,
            "verification_schema_version": 1,
            "agent_name": "nejumi-taiwan-openclaw",
            "project_id": "llm-leaderboard/tc-leaderboard",
            "query_source": {
                "kind": "wandb_agents_api",
                "conversation_id_contains": "run-1",
                "matching_span_count": 0,
            },
            "checks": [
                {
                    "name": "spans_present",
                    "ok": False,
                    "detail": "no matching spans returned by Agents API",
                }
            ],
        },
    )
    failed_weave_sync_report = write_json(
        tmp_path / "weave_agents_sync_run_1.validation_failed.json",
        {
            "ok": False,
            "status": "validation_failed",
            "generated_at": 1,
            "review_path": str(review),
            "source_review_sha256": sha256(review),
            "output_path": "",
            "in_place": False,
            "dry_run": True,
            "entry_count": 0,
            "entries": [],
            "change_count": 0,
            "unmatched_count": 0,
            "changes": [],
            "unmatched_entries": [],
            "completion_paths": [str(failed_weave_verifier)],
            "validation_errors": [
                f"{failed_weave_verifier} is not a passing Weave Agents verifier JSON"
            ],
            "verify_weave_agents": False,
        },
    )
    report = write_json(
        tmp_path / "readiness.json",
        {
            "schema_version": 1,
            "ok": False,
            "status": "not_ready",
            "summary": {
                "gate_count": 2,
                "blocker_count": 1,
                "blockers": ["wandb_completion"],
                "benchmark_evidence": [
                    {
                        "benchmark": "agentic_math",
                        "expected_run_id": "run-1",
                        "standalone_wandb_completion": {
                            "ok": True,
                            "status": "passed",
                            "evidence_paths": [str(completion)],
                        },
                        "review_wandb_completion": {
                            "ok": False,
                            "status": "missing_review_entry",
                            "review_paths": [str(review)],
                        },
                        "completion_proven": False,
                    }
                ],
            },
            "runner": {
                "report_json": str(tmp_path / "readiness.json"),
                "nemoclaw_check": {"path": str(nemoclaw)},
                "existing_results_audit": {
                    "path": str(existing_results_audit),
                    "markdown_path": str(existing_results_audit_md),
                    "ok": True,
                    "status": "passed",
                    "summary": {
                        "record_count": 2,
                        "complete_local_count": 1,
                        "formalized_wandb_complete_count": 1,
                        "unformalized_complete_count": 0,
                        "partial_or_probe_count": 1,
                        "wandb_completion_json_count": 1,
                    },
                },
                "wandb_adoption_draft": {
                    "path": str(wandb_adoption_draft),
                    "markdown_path": str(wandb_adoption_draft_md),
                    "ok": True,
                    "status": "candidates_pending_scope_confirmation",
                    "summary": {
                        "candidate_count": 1,
                        "requires_human_scope_confirmation": True,
                    },
                },
                "paid_run_review_check": {
                    "path": str(paid_review_check),
                    "markdown_path": str(paid_review_check_md),
                },
                "weave_agents_adoption_validation_failures": {
                    "ok": False,
                    "status": "validation_failed_reports_present",
                    "record_count": 1,
                    "glob": "temp/weave_agents_sync_*.validation_failed.json",
                    "records": [
                        {
                            "path": str(failed_weave_sync_report),
                            "ok": False,
                            "status": "validation_failed",
                            "review_path": str(review),
                            "source_review_sha256": sha256(review),
                            "completion_paths": [str(failed_weave_verifier)],
                            "validation_errors": [
                                f"{failed_weave_verifier} is not a passing Weave Agents verifier JSON"
                            ],
                            "dry_run": True,
                            "in_place": False,
                            "entry_count": 0,
                            "change_count": 0,
                        }
                    ],
                },
                "nemoclaw_adoption_check": {
                    "path": str(nemoclaw_adoption_check),
                    "markdown_path": str(nemoclaw_adoption_check_md),
                },
                "nemoclaw_post_install_verification": {
                    "path": str(post_install_check),
                    "markdown_path": str(post_install_check_md),
                },
            },
            "remediation_plan": [
                    {
                        "gate": "nemoclaw_readiness",
                        "status": "failed",
                        "next_action": "install",
                        "commands": [
                            (
                                "uv run python scripts/setup/review_nemoclaw_installer.py "
                                "--url https://www.nvidia.com/nemoclaw.sh --install-ref lkg "
                                f"--expected-sha256 {installer_sha} "
                                "--lock-json scripts/setup/nemoclaw_installer_lock.json "
                                f"--json {installer_review} "
                                f"--markdown {installer_review_md}"
                            ),
                            (
                                "scripts/setup/install_nemoclaw.sh --install --onboard "
                                "--sandbox nejumi-taiwan "
                                "--policy-tier restricted "
                                "--install-ref lkg "
                                "--installer-lock-json scripts/setup/nemoclaw_installer_lock.json "
                                f"--installer-sha256 {installer_sha} "
                                f"--installer-review-json {installer_review} "
                                "--yes-i-accept-third-party-software "
                                f"--json {nemoclaw_install_json}"
                        )
                    ],
                },
                {
                    "gate": "wandb_completion",
                    "status": "missing_required_benchmarks",
                    "next_action": "verify completion",
                    "commands": [],
                },
            ],
            "gates": [
                {
                    "name": "nemoclaw_readiness",
                    "ok": False,
                    "blocking": True,
                    "status": "failed",
                    "detail": "missing",
                    "next_action": "install",
                    "evidence_paths": [str(nemoclaw)],
                    "latest_installer_review": {
                        "path": str(installer_review),
                        "generated_at": "2026-06-28T01:32:47Z",
                        "installer_url": "https://www.nvidia.com/nemoclaw.sh",
                        "install_ref": "lkg",
                        "lock_json": str(installer_lock),
                        "lock_verified": True,
                        "expected_sha256": installer_sha,
                        "sha256": installer_sha,
                        "size_bytes": 6356,
                        "status": "reviewed",
                        "ok": True,
                    },
                },
                {
                    "name": "paid_run_review_package",
                    "ok": False,
                    "blocking": True,
                    "status": "incomplete_reviews",
                    "detail": "incomplete",
                    "next_action": "complete review",
                    "evidence_paths": [str(review)],
                    "records": [{"path": str(review), "status": "completed"}],
                },
                {
                    "name": "one_model_full_canary",
                    "ok": False,
                    "blocking": True,
                    "status": "wandb_completion_not_proven",
                    "detail": "missing completion",
                    "next_action": "complete canary",
                    "evidence_paths": [str(review)],
                    "records": [
                        {
                            "path": str(review),
                            "status": "completed",
                            "weave_agents_completion_entries": [
                                {
                                    "path": str(weave_completion),
                                    "review_path": str(review),
                                    "agent_name": "nejumi-taiwan-openclaw",
                                    "run_id": "run-1",
                                    "ok": True,
                                    "entry_ok": True,
                                    "verified": True,
                                    "schema_valid": True,
                                    "checks_valid": True,
                                    "trace_present": True,
                                    "fresh": True,
                                    "latest_trace_id": "trace-1",
                                }
                            ],
                        }
                    ],
                },
            ],
        },
    )
    output_dir = tmp_path / "bundle"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--readiness-report",
            str(report),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout = json.loads(result.stdout)
    assert stdout["ok"] is False
    assert stdout["operator_plan"] == {
        "json": "operator_plan.json",
        "markdown": "operator_plan.md",
        "schema_version": 1,
        "status": "pending",
    }
    assert stdout["external_action_approval_packet"]["json"] == "external_action_approval_packet.json"
    assert stdout["external_action_approval_packet"]["markdown"] == "external_action_approval_packet.md"
    assert stdout["external_action_approval_packet"]["status"] == "pending_approval"
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["bundle_version"] == 2
    assert manifest["schema_version"] == 1
    assert manifest["readiness_report_schema_version"] == 1
    assert manifest["status"] == "not_ready"
    assert manifest["readiness_status"] == "not_ready"
    assert manifest["current_gate"]["readiness_report_schema_version"] == 1
    assert manifest["current_gate"]["status"] == manifest["status"]
    assert manifest["operator_plan"] == {
        "json": "operator_plan.json",
        "markdown": "operator_plan.md",
        "schema_version": 1,
        "status": "pending",
    }
    assert manifest["external_action_approval_packet"]["json"] == "external_action_approval_packet.json"
    assert manifest["external_action_approval_packet"]["markdown"] == "external_action_approval_packet.md"
    assert manifest["external_action_approval_packet"]["status"] == "pending_approval"
    operator_plan_json = output_dir / "operator_plan.json"
    operator_plan_md = output_dir / "operator_plan.md"
    approval_packet_json = output_dir / "external_action_approval_packet.json"
    approval_packet_md = output_dir / "external_action_approval_packet.md"
    assert operator_plan_json.exists()
    assert operator_plan_md.exists()
    assert approval_packet_json.exists()
    assert approval_packet_md.exists()
    operator_plan = json.loads(operator_plan_json.read_text(encoding="utf-8"))
    assert operator_plan["schema_version"] == 1
    assert operator_plan["status"] == "pending"
    assert operator_plan["operator_next_steps"] == manifest["current_gate"]["operator_next_steps"]
    renderer = operator_plan["operator_execution_plan_renderer"]
    assert renderer["schema_version"] == 1
    assert renderer["status"] == "available"
    assert renderer["script"] == "scripts/tools/render_taiwan_operator_execution_plan.py"
    assert renderer["required_before_external_action"] is True
    assert renderer["safety"]["requires_command_policy_validation_for_shell_script"] is True
    assert (
        renderer["safety"][
            "requires_weave_content_canary_gate_validation_for_shell_script"
        ]
        is True
    )
    assert "operator_plan.json" in renderer["review_command_template"]
    assert renderer["approval_report_json_template"].endswith(".verify.json")
    assert "--external-action-approval-report-json" in renderer["require_ready_command_template"]
    assert renderer["approval_report_json_template"] in renderer["require_ready_command_template"]
    assert "--require-ready" in renderer["require_ready_command_template"]
    checklist = operator_plan["external_action_checklist"]
    assert checklist["schema_version"] == 1
    assert checklist["status"] == "pending"
    assert checklist["item_count"] == operator_plan["operator_next_steps"]["step_count"]
    assert "wandb_access" in checklist["requirement_counts"]
    assert checklist["items"]
    operator_plan_markdown = operator_plan_md.read_text(encoding="utf-8")
    assert "Taiwan Release Operator Plan" in operator_plan_markdown
    assert "## Operator Execution Plan Renderer" in operator_plan_markdown
    assert "render_taiwan_operator_execution_plan.py" in operator_plan_markdown
    assert "## External Action Checklist" in operator_plan_markdown
    approval_packet = json.loads(approval_packet_json.read_text(encoding="utf-8"))
    assert approval_packet["schema_version"] == 1
    assert approval_packet["status"] == "pending_approval"
    assert approval_packet["external_action_checklist"] == checklist
    assert approval_packet["external_action_checklist_sha256"] == manifest[
        "external_action_approval_packet"
    ]["external_action_checklist_sha256"]
    assert approval_packet["required_approval_count"] > 0
    assert approval_packet["all_required_approvals_granted"] is False
    assert approval_packet["approval_verifier"]["script"] == (
        "scripts/tools/verify_external_action_approval_packet.py"
    )
    assert approval_packet["approval_verifier"]["source_packet_json"] == (
        "external_action_approval_packet.json"
    )
    assert "--source-packet-json" in approval_packet["approval_verifier"]["command_template"]
    assert "--require-approved" in approval_packet["approval_verifier"]["command_template"]
    assert approval_packet["approval_template_renderer"]["script"] == (
        "scripts/tools/render_external_action_approval_template.py"
    )
    assert "--output-json" in approval_packet["approval_template_renderer"]["command_template"]
    approval_markdown = approval_packet_md.read_text(encoding="utf-8")
    assert "Taiwan External Action Approval Packet" in approval_markdown
    assert "## Approval Requirements" in approval_markdown
    assert "## Approval Template Renderer" in approval_markdown
    assert "## Approval Verifier" in approval_markdown
    assert "render_external_action_approval_template.py" in approval_markdown
    assert "verify_external_action_approval_packet.py" in approval_markdown
    current_gate = manifest["current_gate"]
    assert current_gate["readiness_report_source"] == str(report)
    assert current_gate["blocking_gates"] == ["wandb_completion"]
    assert current_gate["external_action_checklist"]["schema_version"] == 1
    assert current_gate["external_action_checklist"]["item_count"] == (
        current_gate["operator_next_steps"]["step_count"]
    )
    assert current_gate["required_next_actions"] == [
        {
            "gate": "nemoclaw_readiness",
            "status": "failed",
            "detail": "missing",
            "next_action": "install",
            "evidence_path_count": 1,
            "latest_evidence_paths": [str(nemoclaw)],
        },
        {
            "gate": "paid_run_review_package",
            "status": "incomplete_reviews",
            "detail": "incomplete",
            "next_action": "complete review",
            "evidence_path_count": 1,
            "latest_evidence_paths": [str(review)],
        },
        {
            "gate": "one_model_full_canary",
            "status": "wandb_completion_not_proven",
            "detail": "missing completion",
            "next_action": "complete canary",
            "evidence_path_count": 1,
            "latest_evidence_paths": [str(review)],
        },
    ]
    assert current_gate["benchmark_completion"] == [
        {
            "benchmark": "agentic_math",
            "required": True,
            "expected_run_id": "run-1",
            "completion_proven": False,
            "standalone_status": "passed",
            "standalone_ok": True,
            "standalone_records": [],
            "review_status": "missing_review_entry",
            "review_ok": False,
            "review_entries": [],
        }
    ]
    contract = current_gate["wandb_completion_contract"]
    assert contract["status"] == "incomplete"
    assert contract["complete"] is False
    assert contract["required_benchmarks"] == ["agentic_math"]
    assert contract["required_count"] == 1
    assert contract["release_completion_proven_count"] == 0
    assert contract["standalone_completion_ok_count"] == 1
    assert contract["formalized_existing_result_count"] == 1
    assert contract["missing_release_completion_benchmarks"] == ["agentic_math"]
    assert contract["next_action_count"] == 2
    assert contract["max_age_seconds"] == 86400
    assert contract["benchmarks"][0]["benchmark"] == "agentic_math"
    assert contract["benchmarks"][0]["required"] is True
    assert contract["benchmarks"][0]["status"] == "formalized_but_not_reviewed"
    assert contract["benchmarks"][0]["release_completion_proven"] is False
    assert contract["benchmarks"][0]["standalone_completion_ok"] is True
    assert contract["benchmarks"][0]["review_completion_ok"] is False
    assert contract["benchmarks"][0]["formalized_existing_result"] is True
    assert contract["benchmarks"][0]["formalized_existing_run_ids"] == ["run-1"]
    assert contract["benchmarks"][0]["standalone_completion_paths"] == []
    assert contract["benchmarks"][0]["formalized_existing_completion_paths"] == [str(completion)]
    assert contract["benchmarks"][0]["scope_attestation_template_paths"] == [
        str(wandb_adoption_attestation)
    ]
    assert contract["benchmarks"][0]["sync_ready_adoption_candidate_count"] == 1
    assert contract["benchmarks"][0]["adoption_sync_blocked_reasons"] == []
    assert contract["benchmarks"][0]["refresh_wandb_completion_commands"] == []
    assert contract["benchmarks"][0]["scope_attestation_render_report_paths"] == [
        str(wandb_adoption_render_report)
    ]
    assert contract["benchmarks"][0]["sync_dry_run_report_paths"] == [
        str(wandb_adoption_sync_dry_run)
    ]
    assert contract["benchmarks"][0]["scope_attestation_preflight_report_paths"] == [
        str(wandb_adoption_preflight)
    ]
    assert "paid-run review W&B completion entry is missing or failing" in contract["benchmarks"][0]["missing_reasons"]
    assert "release completion_proven is false" in contract["benchmarks"][0]["missing_reasons"]
    assert contract["benchmarks"][0]["next_actions"] == [
        "link the passing W&B completion verifier into the paid-run review",
        "rerun the release gate after completion and review evidence are updated",
    ]
    assert contract["benchmarks"][0]["recommended_commands"] == [
        (
            "uv run python scripts/tools/render_wandb_scope_attestation.py "
            f"--template-json {wandb_adoption_attestation} "
            f"--output-json {wandb_adoption_attestation} "
            "--confirmed-by REVIEWER_NAME "
            "--confirmed-at YYYY-MM-DDTHH:MM:SS+09:00 "
            "--confirmation CONFIRM_THIS_WANDB_RUN_IS_THE_REVIEWED_SCOPE "
            "--actual-cost-estimate ACTUAL_COST_USD "
            "--provider-bill-reference PROVIDER_BILL_REFERENCE "
            f"--report-json {wandb_adoption_render_report} "
            f"--preflight-report-json {wandb_adoption_preflight} "
            f"--sync-dry-run-report-json {wandb_adoption_sync_dry_run}"
        ),
        (
            "uv run python scripts/tools/verify_wandb_scope_attestation.py "
            "--review-json outputs/taiwan_full_eval/PHASE_paid_run_review.json "
            f"--completion-json {completion} "
            f"--scope-attestation-json {wandb_adoption_attestation} "
            f"--json {wandb_adoption_preflight}"
        ),
        (
            "uv run python scripts/tools/sync_wandb_completion_to_paid_review.py "
            "--review-json outputs/taiwan_full_eval/PHASE_paid_run_review.json "
            f"--completion-json {completion} "
            "--set-verify-wandb-completion --top-level --adopt-existing-result "
            f"--scope-attestation-json {wandb_adoption_attestation} "
            f"--report-json {wandb_adoption_sync_dry_run}"
        ),
        (
            "uv run python scripts/tools/sync_wandb_completion_to_paid_review.py "
            "--review-json outputs/taiwan_full_eval/PHASE_paid_run_review.json "
            f"--completion-json {completion} "
            "--in-place --set-verify-wandb-completion --top-level --adopt-existing-result "
            f"--scope-attestation-json {wandb_adoption_attestation} "
            f"--validated-dry-run-report-json {wandb_adoption_sync_dry_run}"
        ),
        "uv run python scripts/tools/run_taiwan_release_gate.py --quiet",
    ]
    assert contract["benchmarks"][0]["scope_confirmation_required"] is True
    adoption_draft = current_gate["wandb_adoption_draft"]
    assert adoption_draft["pending_scope_confirmation_candidate_count"] == 1
    assert adoption_draft["pending_human_field_count"] == 7
    assert adoption_draft["pending_human_fields"] == [
        "scope_attestation_json.confirmed",
        "scope_attestation_json.confirmed_by",
        "scope_attestation_json.confirmed_at",
        "scope_attestation_json.confirmation",
        "scope_attestation_json.completion_sha256",
        "scope_attestation_json.actual_cost_estimate",
        "scope_attestation_json.provider_bill_reference",
    ]
    adoption_candidate = adoption_draft["candidates"][0]
    assert adoption_candidate["required_human_fields"] == [
        "scope_attestation_json.confirmed",
        "scope_attestation_json.confirmed_by",
        "scope_attestation_json.confirmed_at",
        "scope_attestation_json.confirmation",
        "scope_attestation_json.completion_sha256",
        "scope_attestation_json.actual_cost_estimate",
        "scope_attestation_json.provider_bill_reference",
    ]
    assert adoption_candidate["pending_scope_confirmation"] is True
    assert adoption_candidate["pending_human_field_count"] == 7
    assert "agreed canary scope" in contract["benchmarks"][0]["scope_warning"]
    operator = current_gate["operator_next_steps"]
    assert operator["status"] == "pending"
    assert operator["step_count"] == 2
    assert operator["paid_api_step_count"] == 0
    assert operator["wandb_access_step_count"] == 1
    assert operator["wandb_write_step_count"] == 0
    assert operator["third_party_acceptance_step_count"] == 1
    assert operator["scope_confirmation_step_count"] == 1
    assert operator["nemoclaw_recommendation"] == "conditional_adopt_for_agentic_math"
    assert operator["nemoclaw_ready_for_use"] is False
    assert operator["paid_review_completed_required_fields"] == [
        "ended_at",
        "actual_cost_estimate",
        "provider_bill_reference",
        "runs",
    ]
    assert "agreed canary scope" in operator["warnings"][0]
    assert operator["steps"][0]["gate"] == "nemoclaw_readiness"
    assert operator["steps"][0]["requires_third_party_acceptance"] is True
    assert operator["steps"][0]["requires_nemoclaw_install"] is True
    assert operator["steps"][0]["evidence_to_produce"] == [
        str(installer_review),
        str(installer_review_md),
        str(nemoclaw_install_json),
        str(nemoclaw_install_json).replace(".json", ".install.log"),
        str(nemoclaw_install_json).replace(".json", ".onboard.log"),
    ]
    assert operator["steps"][1]["gate"] == "wandb_completion"
    assert operator["steps"][1]["requires_scope_confirmation"] is True
    assert operator["steps"][1]["requires_wandb_access"] is True
    assert operator["steps"][1]["related_contract_benchmarks"] == [
        {
            "benchmark": "agentic_math",
            "status": "formalized_but_not_reviewed",
            "next_actions": [
                "link the passing W&B completion verifier into the paid-run review",
                "rerun the release gate after completion and review evidence are updated",
            ],
            "scope_confirmation_required": True,
        }
        ]
    assert operator["steps"][1]["commands"] == contract["benchmarks"][0]["recommended_commands"]
    assert operator["steps"][1]["evidence_to_produce"] == [
        str(wandb_adoption_render_report),
        str(wandb_adoption_preflight),
        str(wandb_adoption_sync_dry_run),
    ]
    assert current_gate["weave_agents_completion"] == [
        {
            "gate": "one_model_full_canary",
            "review_path": str(review),
            "phase": None,
            "record_status": "completed",
            "required": False,
            "entry_count": 1,
            "verified_count": 1,
            "completion_proven": True,
            "max_age_seconds": None,
            "entries": [
                {
                    "path": str(weave_completion),
                    "review_path": str(review),
                    "agent_name": "nejumi-taiwan-openclaw",
                    "run_id": "run-1",
                    "ok": True,
                    "entry_ok": True,
                    "verified": True,
                    "schema_valid": True,
                        "checks_valid": True,
                        "trace_present": True,
                        "fresh": True,
                        "latest_trace_id": "trace-1",
                        "sync_dry_run_report_json": None,
                        "sync_dry_run_source_review_json": None,
                        "sync_dry_run_source_review_sha256": None,
                        "verification_error": None,
                    }
                ],
        }
    ]
    assert current_gate["nemoclaw_adoption"]["status"] == "not_installed"
    assert current_gate["nemoclaw_adoption"]["ok"] is False
    assert current_gate["nemoclaw_adoption"]["sandbox"] == "nejumi-taiwan"
    assert current_gate["nemoclaw_adoption"]["adoption_decision"] == {
        "recommendation": "conditional_adopt_for_agentic_math",
        "ready_for_use": False,
        "scope": "agentic_math_only",
        "sandbox": "nejumi-taiwan",
        "design_ready": True,
        "runtime_blockers": ["setup_installed"],
        "design_blockers": [],
        "other_blockers": [],
        "blockers": ["setup_installed"],
        "rationale": "Design and policy criteria pass, but runtime is missing.",
        "next_action": "Install/onboard NeMoClaw, then rerun verification.",
    }
    assert current_gate["nemoclaw_adoption"]["ready_for_use"] is False
    assert current_gate["nemoclaw_adoption"]["adoption_recommendation"] == (
        "conditional_adopt_for_agentic_math"
    )
    assert current_gate["nemoclaw_adoption"]["adoption_scope"] == "agentic_math_only"
    assert current_gate["nemoclaw_adoption"]["design_ready"] is True
    assert current_gate["nemoclaw_adoption"]["blockers"] == ["setup_installed"]
    assert current_gate["nemoclaw_adoption"]["runtime_blockers"] == ["setup_installed"]
    assert current_gate["nemoclaw_adoption"]["design_blockers"] == []
    assert current_gate["nemoclaw_adoption"]["other_blockers"] == []
    assert current_gate["nemoclaw_adoption"]["setup_runtime"] == {
        "host_prerequisites_ok": True,
        "runtime_installed": False,
        "missing_required_commands": ["nemoclaw", "openshell"],
        "missing_components": ["nemoclaw", "openshell"],
    }
    assert current_gate["nemoclaw_adoption"]["missing_required_commands"] == [
        "nemoclaw",
        "openshell",
    ]
    assert current_gate["nemoclaw_adoption"]["missing_components"] == [
        "nemoclaw",
        "openshell",
    ]
    assert current_gate["nemoclaw_adoption"]["summary"]["blockers"] == ["setup_installed"]
    assert current_gate["nemoclaw_installer_review"] == {
        "path": str(installer_review),
        "generated_at": "2026-06-28T01:32:47Z",
        "installer_url": "https://www.nvidia.com/nemoclaw.sh",
        "install_ref": "lkg",
        "lock_json": str(installer_lock),
        "lock_verified": True,
        "expected_sha256": installer_sha,
        "sha256": installer_sha,
        "size_bytes": 6356,
        "status": "reviewed",
        "ok": True,
    }
    weave_failures = current_gate["runner_evidence"][
        "weave_agents_adoption_validation_failures"
    ]
    assert weave_failures["status"] == "validation_failed_reports_present"
    assert weave_failures["record_count"] == 1
    assert weave_failures["records"][0]["path"] == str(failed_weave_sync_report)
    assert weave_failures["records"][0]["completion_paths"] == [
        str(failed_weave_verifier)
    ]
    assert "not a passing Weave Agents verifier JSON" in weave_failures["records"][0][
        "validation_errors"
    ][0]

    assert current_gate["nemoclaw_adoption"]["criteria"] == [
        {
            "name": "setup_installed",
            "ok": False,
            "status": "missing_or_not_ready",
            "detail": "missing nemoclaw",
            "next_action": "install",
            "evidence_paths": [str(nemoclaw)],
        },
        {
            "name": "agentic_math_config",
            "ok": True,
            "status": "passed",
            "detail": "ready",
            "next_action": "use config",
            "evidence_paths": [str(nemoclaw_config)],
        },
    ]
    paid_review_summary = current_gate["paid_run_review_package"]
    assert paid_review_summary["status"] == "not_ready"
    assert paid_review_summary["summary"]["blockers"] == ["paid_run_review_package"]
    assert paid_review_summary["requirements"]["required_wandb_benchmarks"] == ["agentic_math"]
    assert paid_review_summary["review_completion_requirements"]["completed_review_required_fields"] == [
        "ended_at",
        "actual_cost_estimate",
        "provider_bill_reference",
        "runs",
    ]
    assert paid_review_summary["gates"][0]["blocking_record_count"] == 1
    assert paid_review_summary["gates"][0]["blocking_records"][0]["provider_bill_reference_present"] is False
    existing = current_gate["existing_results_formalization"]
    assert existing["status"] == "passed"
    assert existing["ok"] is True
    assert existing["summary"] == {
        "record_count": 2,
        "complete_local_count": 1,
        "formalized_wandb_complete_count": 1,
        "unformalized_complete_count": 0,
        "partial_or_probe_count": 1,
        "wandb_completion_json_count": 1,
    }
    assert existing["formalized_records"][0]["formalization_status"] == "formalized_wandb_complete"
    assert existing["formalized_records"][0]["wandb_completion"]["run_id"] == "run-1"
    assert existing["partial_records"][0]["formalization_status"] == "partial_not_reloggable"
    sources = {row["source_path"]: row for row in manifest["files"]}
    for path in (
        completion,
        taxonomy,
        review,
        pre_run_budget,
        weave_completion,
        nemoclaw,
        nemoclaw_install_log,
        existing_results_audit,
        existing_results_audit_md,
        relog_plan,
        wandb_adoption_attestation,
        confirmed_scope_attestation,
        wandb_adoption_draft,
        wandb_adoption_draft_md,
        paid_review_check,
        paid_review_check_md,
        nemoclaw_config,
        nemoclaw_adoption_check,
        nemoclaw_adoption_check_md,
        post_install_preflight,
        post_install_check,
        post_install_check_md,
        installer_lock,
        installer_review,
        installer_review_md,
        failed_weave_verifier,
        failed_weave_sync_report,
        report,
        operator_plan_json,
        operator_plan_md,
        approval_packet_json,
        approval_packet_md,
    ):
        record = sources[str(path)]
        assert record["exists"] is True
        assert record["bundle_path"]
        assert (output_dir / record["bundle_path"]).exists()
        assert record["sha256"]
    assert "release_operator_plan_json" in sources[str(operator_plan_json)]["roles"]
    assert "release_operator_plan_markdown" in sources[str(operator_plan_md)]["roles"]
    assert "external_action_approval_packet_json" in sources[str(approval_packet_json)]["roles"]
    assert "external_action_approval_packet_markdown" in sources[str(approval_packet_md)]["roles"]
    verifier_script = "scripts/tools/verify_external_action_approval_packet.py"
    assert "external_action_approval_packet_verifier:script" in sources[verifier_script][
        "roles"
    ]
    renderer_script = "scripts/tools/render_external_action_approval_template.py"
    assert "external_action_approval_template_renderer:script" in sources[renderer_script][
        "roles"
    ]
    assert any(
        role.endswith(":pre_run_budget_estimate")
        for role in sources[str(pre_run_budget)]["roles"]
    )
    for script_path in (
        "scripts/tools/render_taiwan_operator_execution_plan.py",
        "scripts/tools/sync_wandb_completion_to_paid_review.py",
        "scripts/tools/run_taiwan_release_gate.py",
        "scripts/setup/review_nemoclaw_installer.py",
        "scripts/setup/install_nemoclaw.sh",
    ):
        assert script_path in sources
        assert sources[script_path]["exists"] is True
        assert sources[script_path]["bundle_path"]
        assert (
            "operator_plan:command_script" in sources[script_path]["roles"]
            or "operator_execution_plan_renderer:script" in sources[script_path]["roles"]
        )
        assert (output_dir / sources[script_path]["bundle_path"]).exists()
    for script_path in (
        "scripts/setup/review_nemoclaw_installer.py",
        "scripts/setup/install_nemoclaw.sh",
    ):
        assert "current_gate:remediation_plan:command_script" in sources[script_path]["roles"]
    expected_agentic_runner_roles = {
        "scripts/tools/run_openclaw_agent_protocol.py": "agentic_runner:protocol_script",
        "scripts/tools/run_agentic_math_openclaw.py": "agentic_runner:math_script",
        "scripts/tools/run_swebench_pro_openclaw.py": "agentic_runner:swe_script",
        "scripts/tools/log_agentic_math_results_to_wandb.py": "agentic_runner:math_relog_script",
        "scripts/tools/log_agentic_swe_results_to_wandb.py": "agentic_runner:swe_relog_script",
    }
    for script_path, role in expected_agentic_runner_roles.items():
        assert script_path in sources
        assert sources[script_path]["exists"] is True
        assert sources[script_path]["bundle_path"]
        assert "agentic_runner:script" in sources[script_path]["roles"]
        assert role in sources[script_path]["roles"]
        assert (output_dir / sources[script_path]["bundle_path"]).exists()
    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "Taiwan Release Evidence Bundle" in summary
    assert "Operator plan JSON" in summary
    assert "operator_plan.md" in summary
    assert "External action approval packet JSON" in summary
    assert "external_action_approval_packet.json" in summary
    assert "render_external_action_approval_template.py" in summary
    assert "verify_external_action_approval_packet.py" in summary
    assert "## Current Gate" in summary
    assert "## Required Next Actions" in summary
    assert "## Operator Next Steps" in summary
    assert "## External Action Checklist" in summary
    assert "## External Action Approval Packet" in summary
    assert "External action items" in summary
    assert "nemoclaw_readiness" in summary
    assert "## Benchmark W&B Completion" in summary
    assert "## W&B Completion Contract" in summary
    assert "formalized_but_not_reviewed" in summary
    assert str(wandb_adoption_attestation) in summary
    assert "sync_wandb_completion_to_paid_review.py" in summary
    assert "run_taiwan_release_gate.py --quiet" in summary
    assert "## Existing W&B Adoption Draft" in summary
    assert "candidates_pending_scope_confirmation" in summary
    assert str(existing_results_audit) in summary
    assert existing_results_audit_sha in summary
    assert str(wandb_adoption_sync_dry_run) in summary
    assert "wandb_adoption_draft" in sources[str(wandb_adoption_draft)]["roles"]
    assert "wandb_adoption_draft_markdown" in sources[str(wandb_adoption_draft_md)]["roles"]
    assert "wandb_adoption_draft:source_audit_json" in sources[str(existing_results_audit)]["roles"]
    assert (
        "existing_results_audit:record:agentic_math:gpt-5_5:relog_dry_run_plan_json"
        in sources[str(relog_plan)]["roles"]
    )
    relog_script = "scripts/tools/log_agentic_math_results_to_wandb.py"
    assert "existing_results_audit:relog_command_script" in sources[relog_script]["roles"]
    assert (
        "existing_results_audit:record:agentic_math:gpt-5_5:relog_command_script"
        in sources[relog_script]["roles"]
    )
    assert sources[relog_script]["exists"] is True
    assert sources[relog_script]["bundle_path"]
    assert (output_dir / sources[relog_script]["bundle_path"]).exists()
    relog_helper = "scripts/tools/relog_wandb_approval.py"
    assert "existing_results_audit:relog_dependency_script" in sources[relog_helper]["roles"]
    assert (
        "existing_results_audit:record:agentic_math:gpt-5_5:relog_dependency_script"
        in sources[relog_helper]["roles"]
    )
    assert sources[relog_helper]["exists"] is True
    assert sources[relog_helper]["bundle_path"]
    assert (output_dir / sources[relog_helper]["bundle_path"]).exists()
    assert (
        "wandb_adoption_draft:candidate:agentic_math:wandb_completion_json"
        in sources[str(completion)]["roles"]
    )
    assert (
        "wandb_adoption_draft:candidate:agentic_math:scope_attestation_template_json"
        in sources[str(wandb_adoption_attestation)]["roles"]
    )
    assert any(
        role.endswith("agentic_math:source_attestation_json")
        for role in sources[str(confirmed_scope_attestation)]["roles"]
    )
    assert "Attestation template count" in summary
    assert str(confirmed_scope_attestation) in summary
    assert "outputs/taiwan_full_eval/PHASE_paid_run_review.json" not in sources
    assert "## Weave Agents Completion" in summary
    assert "## Paid Run Review Package" in summary
    assert "### Paid Review W&B Completion Entries" in summary
    assert "provider_bill_reference" in summary
    assert "## Existing Results Formalization" in summary
    assert "formalized_wandb_complete" in summary
    assert "partial_not_reloggable" in summary
    assert "## NeMoClaw Adoption" in summary
    assert "conditional_adopt_for_agentic_math" in summary
    assert "setup_installed" in summary
    assert "## NeMoClaw Installer Review" in summary
    assert str(installer_lock) in summary
    assert installer_sha in summary
    assert "gate:nemoclaw_readiness:latest_installer_review_lock_json" in sources[str(installer_lock)]["roles"]
    assert "gate:nemoclaw_readiness:latest_installer_review_json" in sources[str(installer_review)]["roles"]
    assert (
        "weave_agents_adoption_validation_failure:1:report_json"
        in sources[str(failed_weave_sync_report)]["roles"]
    )
    assert (
        "weave_agents_adoption_validation_failure:1:completion_json"
        in sources[str(failed_weave_verifier)]["roles"]
    )
    assert (
        "gate:nemoclaw_readiness:latest_installer_review_markdown"
        in sources[str(installer_review_md)]["roles"]
    )
    assert "## NeMoClaw Post-Install Verification" in summary
    assert "Command safety OK" in summary
    assert "Forbidden token count" in summary
    assert "Forbidden exact tokens" in summary
    assert "Forbidden prefixes" in summary
    assert "Forbidden markers" in summary
    assert "WANDB_" in summary
    assert "openrouter" in summary
    assert "Missing required token count" in summary
    assert "setup_check" in summary
    assert "protocol_preflight" in summary
    assert "Will query W&B" in summary
    assert "wandb_completion" in summary
    assert any(
        "gate:one_model_full_canary:weave_agents_completion" in role
        for role in sources[str(weave_completion)]["roles"]
    )
    assert any(
        "operation:install:log" in role
        for role in sources[str(nemoclaw_install_log)]["roles"]
    )


def test_wandb_completion_contract_refreshes_before_sync_when_adoption_not_ready():
    module = load_module()
    contract = module.wandb_completion_contract_summary(
        benchmark_completion=[
            {
                "benchmark": "agentic_math",
                "completion_proven": False,
                "standalone_ok": True,
                "standalone_records": [
                    {
                        "ok": True,
                        "path": "outputs/taiwan_full_eval/wandb_completion/agentic_math-run-1.json",
                        "run_id": "run-1",
                    }
                ],
                "review_ok": False,
                "review_entries": [],
            }
        ],
        existing_results_formalization={
            "formalized_records": [
                {
                    "benchmark": "agentic_math",
                    "wandb_completion": {
                        "path": "outputs/taiwan_full_eval/wandb_completion/agentic_math-run-1.json",
                        "run_id": "run-1",
                    },
                }
            ]
        },
        wandb_adoption_draft={
            "candidates": [
                {
                    "benchmark": "agentic_math",
                    "wandb_run_id": "run-1",
                    "wandb_completion_json": "outputs/taiwan_full_eval/wandb_completion/agentic_math-run-1.json",
                    "target_review_json": "outputs/taiwan_full_eval/canary_agentic_paid_run_review.json",
                    "scope_attestation_template_json": "temp/agentic_math-run-1.scope_attestation.json",
                    "sync_dry_run_report_json": "temp/agentic_math-run-1.sync_dry_run.json",
                    "sync_ready": False,
                    "run_metadata_valid": False,
                    "sync_dry_run_command": "",
                    "sync_apply_command": "",
                    "sync_command": "",
                    "refresh_wandb_completion_command": (
                        "uv run python scripts/tools/verify_taiwan_wandb_completion.py "
                        "--entity llm-leaderboard --project tc-leaderboard --run-id run-1 "
                        "--benchmark agentic_math --expected-total 100 "
                        "--expected-run-tag REVIEWED_CANARY_OR_PAID_SCOPE_TAG "
                        "--json outputs/taiwan_full_eval/wandb_completion/agentic_math-run-1.json"
                    ),
                    "sync_command_blocked_reason": (
                        "W&B completion verifier JSON is not sync-ready because run metadata is missing or invalid"
                    ),
                }
            ]
        },
        paid_run_review_package={
            "requirements": {"required_wandb_benchmarks": ["agentic_math"]}
        },
    )

    row = contract["benchmarks"][0]
    assert row["status"] == "formalized_but_not_reviewed"
    assert row["sync_ready_adoption_candidate_count"] == 0
    assert row["scope_attestation_template_paths"] == [
        "temp/agentic_math-run-1.scope_attestation.json"
    ]
    assert row["sync_dry_run_report_paths"] == []
    assert row["adoption_sync_blocked_reasons"] == [
        "W&B completion verifier JSON is not sync-ready because run metadata is missing or invalid"
    ]
    assert row["next_actions"] == [
        "refresh W&B completion verifier with run metadata before paid-review adoption",
        "rerun the release gate after completion and review evidence are updated",
    ]
    assert row["recommended_commands"] == [
        (
            "uv run python scripts/tools/verify_taiwan_wandb_completion.py "
            "--entity llm-leaderboard --project tc-leaderboard --run-id run-1 "
            "--benchmark agentic_math --expected-total 100 "
            "--expected-run-tag REVIEWED_CANARY_OR_PAID_SCOPE_TAG "
            "--json outputs/taiwan_full_eval/wandb_completion/agentic_math-run-1.json"
        ),
        "uv run python scripts/tools/run_taiwan_release_gate.py --quiet",
    ]
    assert all(
        "sync_wandb_completion_to_paid_review.py" not in command
        for command in row["recommended_commands"]
    )
    assert row["scope_confirmation_required"] is False


def test_wandb_completion_contract_keeps_scope_handoff_when_verifier_is_stale():
    module = load_module()
    refresh_command = (
        "uv run python scripts/tools/verify_taiwan_wandb_completion.py "
        "--entity llm-leaderboard --project tc-leaderboard --run-id run-1 "
        "--benchmark agentic_math --expected-total 100 "
        "--expected-run-config model.pretrained_model_name_or_path=deepseek/deepseek-v4-pro "
        "--expected-run-job-type evaluation-relog "
        "--json outputs/taiwan_full_eval/wandb_completion/agentic_math-run-1.json"
    )
    render_command = (
        "uv run python scripts/tools/render_wandb_scope_attestation.py "
        "--template-json temp/agentic_math-run-1.scope_attestation.json "
        "--output-json temp/agentic_math-run-1.scope_attestation.json "
        "--confirmed-by REVIEWER_NAME --confirmed-at YYYY-MM-DDTHH:MM:SS+09:00 "
        "--confirmation CONFIRM_THIS_WANDB_RUN_IS_THE_REVIEWED_SCOPE "
        "--actual-cost-estimate ACTUAL_COST_USD "
        "--provider-bill-reference PROVIDER_BILL_REFERENCE "
        "--report-json temp/agentic_math-run-1.scope_attestation.render.json "
        "--markdown temp/agentic_math-run-1.scope_attestation.render.md "
        "--preflight-report-json temp/agentic_math-run-1.scope_preflight.json "
        "--sync-dry-run-report-json temp/agentic_math-run-1.sync_dry_run.json"
    )
    preflight_command = (
        "uv run python scripts/tools/verify_wandb_scope_attestation.py "
        "--review-json outputs/taiwan_full_eval/canary_agentic_paid_run_review.json "
        "--completion-json outputs/taiwan_full_eval/wandb_completion/agentic_math-run-1.json "
        "--scope-attestation-json temp/agentic_math-run-1.scope_attestation.json "
        "--json temp/agentic_math-run-1.scope_preflight.json"
    )
    dry_run_command = (
        "uv run python scripts/tools/sync_wandb_completion_to_paid_review.py "
        "--review-json outputs/taiwan_full_eval/canary_agentic_paid_run_review.json "
        "--completion-json outputs/taiwan_full_eval/wandb_completion/agentic_math-run-1.json "
        "--set-verify-wandb-completion --top-level --adopt-existing-result "
        "--scope-attestation-json temp/agentic_math-run-1.scope_attestation.json "
        "--report-json temp/agentic_math-run-1.sync_dry_run.json"
    )
    apply_command = (
        "uv run python scripts/tools/sync_wandb_completion_to_paid_review.py "
        "--review-json outputs/taiwan_full_eval/canary_agentic_paid_run_review.json "
        "--completion-json outputs/taiwan_full_eval/wandb_completion/agentic_math-run-1.json "
        "--in-place --set-verify-wandb-completion --top-level --adopt-existing-result "
        "--scope-attestation-json temp/agentic_math-run-1.scope_attestation.json "
        "--validated-dry-run-report-json temp/agentic_math-run-1.sync_dry_run.json"
    )
    contract = module.wandb_completion_contract_summary(
        benchmark_completion=[
            {
                "benchmark": "agentic_math",
                "completion_proven": False,
                "standalone_ok": False,
                "standalone_status": "stale",
                "standalone_records": [
                    {
                        "ok": True,
                        "path": "outputs/taiwan_full_eval/wandb_completion/agentic_math-run-1.json",
                        "run_id": "run-1",
                        "fresh": False,
                    }
                ],
                "review_ok": False,
                "review_entries": [],
            }
        ],
        existing_results_formalization={
            "formalized_records": [
                {
                    "benchmark": "agentic_math",
                    "wandb_completion": {
                        "path": "outputs/taiwan_full_eval/wandb_completion/agentic_math-run-1.json",
                        "run_id": "run-1",
                    },
                }
            ]
        },
        wandb_adoption_draft={
            "candidates": [
                {
                    "benchmark": "agentic_math",
                    "model": "deepseek/deepseek-v4-pro",
                    "metrics": {"expected_total": 100},
                    "wandb_entity": "llm-leaderboard",
                    "wandb_project": "tc-leaderboard",
                    "wandb_run_id": "run-1",
                    "wandb_run_name": "taiwan-agentic-math-relog",
                    "wandb_completion_json": "outputs/taiwan_full_eval/wandb_completion/agentic_math-run-1.json",
                    "target_review_json": "outputs/taiwan_full_eval/canary_agentic_paid_run_review.json",
                    "scope_attestation_template_json": "temp/agentic_math-run-1.scope_attestation.json",
                    "scope_attestation_preflight_report_json": "temp/agentic_math-run-1.scope_preflight.json",
                    "sync_dry_run_report_json": "temp/agentic_math-run-1.sync_dry_run.json",
                    "sync_ready": True,
                    "run_metadata_valid": True,
                    "scope_attestation_render_command": render_command,
                    "scope_attestation_preflight_command": preflight_command,
                    "sync_dry_run_command": dry_run_command,
                    "sync_apply_command": apply_command,
                }
            ]
        },
        paid_run_review_package={
            "requirements": {"required_wandb_benchmarks": ["agentic_math"]}
        },
    )

    row = contract["benchmarks"][0]
    assert row["status"] == "formalized_but_not_reviewed"
    assert row["standalone_completion_ok"] is False
    assert row["sync_ready_adoption_candidate_count"] == 1
    assert row["scope_confirmation_required"] is True
    assert row["scope_warning"]
    assert row["refresh_wandb_completion_commands"] == [refresh_command]
    assert row["next_actions"] == [
        "refresh W&B completion verifier before paid-review adoption",
        "link the passing W&B completion verifier into the paid-run review",
        "rerun the release gate after completion and review evidence are updated",
    ]
    assert row["recommended_commands"] == [
        refresh_command,
        render_command,
        preflight_command,
        dry_run_command,
        apply_command,
        "uv run python scripts/tools/run_taiwan_release_gate.py --quiet",
    ]


def test_operator_next_steps_marks_placeholder_commands_as_templates():
    module = load_module()

    operator = module.operator_next_steps_summary(
        remediation_plan=[
            {
                "gate": "wandb_completion",
                "status": "failed",
                "next_action": "verify W&B completion",
                "commands": [
                    (
                        "uv run python scripts/tools/verify_taiwan_wandb_completion.py "
                        "--run-id RUN_ID --benchmark agentic_swe "
                        "--json outputs/taiwan_full_eval/wandb_completion/agentic_swe-RUN_ID.json"
                    )
                ],
            }
        ],
        wandb_completion_contract={"benchmarks": []},
        paid_run_review_package={},
        nemoclaw_adoption={},
    )

    assert operator["command_template_step_count"] == 1
    assert operator["unresolved_placeholder_tokens"] == ["RUN_ID"]
    step = operator["steps"][0]
    assert step["command_template_count"] == 1
    assert step["evidence_template_count"] == 1
    assert step["unresolved_placeholder_tokens"] == ["RUN_ID"]
    assert step["ready_to_execute_without_placeholder"] is False


def test_release_evidence_bundle_operator_plan_lists_batch_runner_outputs(tmp_path):
    report = write_json(
        tmp_path / "readiness.json",
        {
            "schema_version": 1,
            "ok": False,
            "status": "not_ready",
            "summary": {
                "gate_count": 1,
                "blocker_count": 1,
                "blockers": ["one_model_full_canary"],
            },
            "remediation_plan": [
                {
                    "gate": "one_model_full_canary",
                    "status": "incomplete",
                    "next_action": "run canary",
                    "commands": [
                        (
                            "uv run python scripts/tools/run_taiwan_full_eval_batch.py "
                            "--manifest configs/taiwan_openai_canary_models.yaml "
                            "--canary --phase agentic "
                            "--generated-config-dir configs/taiwan_full/generated_openai_canary_agentic_nemoclaw "
                            "--output-root outputs/taiwan_full_eval "
                            "--wandb-run-id-prefix twcanary-openai-mini-YYYYMMDD "
                            "--verify-wandb-completion --verify-weave-agents "
                            "--weave-agents-require-tool-span "
                            "--weave-agents-require-tool-content "
                            "--weave-content-canary-gate WEAVE_CONTENT_CANARY_GATE "
                            "--require-weave-content-canary --yes "
                            "--run-purpose 'OpenAI-direct one-model agentic phase' "
                            "--expected-cost-band 'approved canary cap' "
                            "--external-action-approval-report-json "
                            "temp/taiwan_external_action_approval_REVIEWED_YYYYMMDDTHHMM.verify.json"
                        )
                    ],
                }
            ],
            "gates": [
                {
                    "name": "one_model_full_canary",
                    "ok": False,
                    "blocking": True,
                    "status": "incomplete",
                    "detail": "missing",
                    "next_action": "run canary",
                    "evidence_paths": [],
                }
            ],
        },
    )
    output_dir = tmp_path / "bundle"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--readiness-report",
            str(report),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    operator_plan = json.loads((output_dir / "operator_plan.json").read_text(encoding="utf-8"))
    step = operator_plan["operator_next_steps"]["steps"][0]
    assert step["evidence_to_produce"] == [
        "outputs/taiwan_full_eval/canary_agentic_execution_plan.json",
        "outputs/taiwan_full_eval/canary_agentic_paid_run_review.json",
        "outputs/taiwan_full_eval/batch_manifest.json",
        "outputs/taiwan_full_eval/wandb_completion/agentic-MODEL_SLUG-agentic_math.json",
        "outputs/taiwan_full_eval/wandb_completion/agentic-MODEL_SLUG-agentic_swe.json",
        "outputs/taiwan_full_eval/weave_agents_completion/agentic-MODEL_SLUG.json",
    ]


def test_release_evidence_bundle_operator_plan_lists_weave_content_canary_outputs(tmp_path):
    report = write_json(
        tmp_path / "readiness.json",
        {
            "schema_version": 1,
            "ok": False,
            "status": "not_ready",
            "summary": {
                "gate_count": 1,
                "blocker_count": 1,
                "blockers": ["weave_content_canary"],
            },
            "remediation_plan": [
                {
                    "gate": "weave_content_canary",
                    "status": "failed",
                    "next_action": "rerun content canary",
                    "commands": [
                        (
                            "uv run python scripts/tools/run_weave_agents_content_canary.py "
                            "--canary-id PREPARE_ONLY "
                            "--model openai-direct/gpt-4.1-nano-2025-04-14 --thinking off"
                        ),
                        (
                            "uv run python scripts/tools/run_weave_agents_content_canary.py "
                            "--execute --canary-id CONTENT_CANARY_YYYYMMDDTHHMM "
                            "--model openai-direct/gpt-4.1-nano-2025-04-14 "
                            "--thinking off --timeout 180"
                        ),
                        (
                            "uv run python scripts/tools/check_taiwan_canary_readiness.py "
                            "--weave-content-canary-gate "
                            "outputs/weave_agents_content_canary/plans/"
                            "weave_agents_content_canary_CONTENT_CANARY_YYYYMMDDTHHMM.gate.json "
                            "--require-weave-content-canary "
                            "--json outputs/taiwan_full_eval/openai_canary_readiness_weave_content.json"
                        ),
                    ],
                }
            ],
            "gates": [
                {
                    "name": "weave_content_canary",
                    "ok": False,
                    "blocking": True,
                    "status": "failed",
                    "detail": "missing",
                    "next_action": "rerun content canary",
                    "evidence_paths": [],
                }
            ],
        },
    )
    output_dir = tmp_path / "bundle"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--readiness-report",
            str(report),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    operator_plan = json.loads((output_dir / "operator_plan.json").read_text(encoding="utf-8"))
    step = operator_plan["operator_next_steps"]["steps"][0]
    assert step["evidence_to_produce"] == [
        "outputs/weave_agents_content_canary/prompts/weave_agents_content_canary_PREPARE_ONLY.md",
        "outputs/weave_agents_content_canary/plans/weave_agents_content_canary_PREPARE_ONLY.json",
        "outputs/weave_agents_content_canary/plans/weave_agents_content_canary_PREPARE_ONLY.gate.json",
        (
            "outputs/weave_agents_content_canary/prompts/"
            "weave_agents_content_canary_CONTENT_CANARY_YYYYMMDDTHHMM.md"
        ),
        (
            "outputs/weave_agents_content_canary/plans/"
            "weave_agents_content_canary_CONTENT_CANARY_YYYYMMDDTHHMM.json"
        ),
        (
            "outputs/weave_agents_content_canary/plans/"
            "weave_agents_content_canary_CONTENT_CANARY_YYYYMMDDTHHMM.gate.json"
        ),
        (
            "outputs/weave_agents_content_canary/plans/"
            "weave_agents_content_canary_CONTENT_CANARY_YYYYMMDDTHHMM.command_result.json"
        ),
        (
            "outputs/weave_agents_content_canary/agentic_math/"
            "weave_agents_content_canary_CONTENT_CANARY_YYYYMMDDTHHMM/openclaw_result.json"
        ),
        (
            "outputs/weave_agents_content_canary/verifier/"
            "weave_agents_content_canary_CONTENT_CANARY_YYYYMMDDTHHMM/attempt_*.json"
        ),
        (
            "outputs/weave_agents_content_canary/agents_diagnostics/"
            "weave_agents_content_canary_CONTENT_CANARY_YYYYMMDDTHHMM.agents.json"
        ),
        "outputs/taiwan_full_eval/openai_canary_readiness_weave_content.json",
    ]


def test_release_evidence_bundle_fail_on_not_ready_returns_nonzero(tmp_path):
    report = write_json(
        tmp_path / "readiness.json",
        {
            "schema_version": 1,
            "ok": False,
            "status": "not_ready",
            "summary": {"blockers": ["x"], "gate_count": 1, "blocker_count": 1},
            "gates": [],
        },
    )

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--readiness-report",
            str(report),
            "--output-dir",
            str(tmp_path / "bundle"),
            "--fail-on-not-ready",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert (tmp_path / "bundle" / "manifest.json").exists()
