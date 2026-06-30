#!/usr/bin/env python3
"""Build an offline Taiwan leaderboard production-readiness report.

The report consolidates existing JSON evidence. It does not query W&B, call
model providers, install NeMoClaw, or mutate runtime state.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path("outputs") / "taiwan_full_eval"
OPENAI_CANARY_MANIFEST = "configs/taiwan_openai_canary_models.yaml"
OPENAI_CANARY_FULL_DIR = "configs/taiwan_full/generated_openai_canary"
OPENAI_CANARY_NONAGENTIC_DIR = "configs/taiwan_full/generated_openai_canary_nonagentic"
OPENAI_CANARY_AGENTIC_DIR = "configs/taiwan_full/generated_openai_canary_agentic"
OPENAI_CANARY_AGENTIC_NEMOCLAW_DIR = "configs/taiwan_full/generated_openai_canary_agentic_nemoclaw"
OPENAI_CANARY_AGENTIC_AGGREGATE_DIR = "configs/taiwan_full/generated_openai_canary_agentic_aggregate"
OPENAI_CANARY_BUDGET_ESTIMATE_JSON = "outputs/taiwan_full_eval/openai_canary_budget_estimate.json"
EXTERNAL_ACTION_APPROVAL_REPORT_TEMPLATE = (
    "temp/taiwan_external_action_approval_REVIEWED_YYYYMMDDTHHMM.verify.json"
)
EXTERNAL_ACTION_APPROVAL_SOURCE_PACKET_TEMPLATE = (
    "outputs/taiwan_release_evidence/bundle_YYYYMMDDTHHMM/"
    "external_action_approval_packet.json"
)
DEFAULT_WEAVE_GATE_GLOB = "outputs/weave_agents_content_canary/plans/*.gate.json"
DEFAULT_READINESS_GLOBS = (
    "outputs/taiwan_full_eval/*readiness*.json",
    "temp/*readiness*.json",
)
DEFAULT_NEMOCLAW_SETUP_GLOB = "temp/nemoclaw_setup_check*.json"
DEFAULT_NEMOCLAW_INSTALLER_REVIEW_GLOB = "temp/nemoclaw_installer_review*.json"
DEFAULT_NEMOCLAW_OPERATOR_DOCS_GLOB = "temp/nemoclaw_operator_docs_verification*.json"
REQUIRED_NEMOCLAW_OPERATOR_DOC_CHECK_NAMES = {
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
}
NEMOCLAW_INSTALLER_LOCK_JSON = "scripts/setup/nemoclaw_installer_lock.json"
DEFAULT_REVIEW_GLOB = "outputs/taiwan_full_eval/*paid_run_review.json"
DEFAULT_WANDB_COMPLETION_GLOB = "outputs/taiwan_full_eval/wandb_completion/*.json"
DEFAULT_EXISTING_RESULTS_AUDIT_GLOB = "temp/taiwan_existing_results_audit*.json"
DEFAULT_REQUIRED_WANDB_BENCHMARKS = ("agentic_math", "agentic_swe", "taiwan_full")
AGENTIC_REQUIRED_WANDB_BENCHMARKS = {"agentic_math", "agentic_swe"}
WEAVE_AGENTS_CANARY_COMPLETION_PHASES = {"agentic", "full"}
DEFAULT_WEAVE_CONTENT_CANARY_MAX_AGE_SECONDS = 24 * 60 * 60
DEFAULT_WANDB_COMPLETION_MAX_AGE_SECONDS = 24 * 60 * 60
WANDB_COMPLETION_SCHEMA_VERSION = 1
WANDB_COMPLETION_QUERY_SOURCE_KIND = "wandb_sdk"
WANDB_COMPLETION_API_TIMEOUT_SECONDS = 60
WEAVE_AGENTS_COMPLETION_SCHEMA_VERSION = 1
REQUIRED_WEAVE_AGENTS_CHECK_NAMES = {
    "request_model",
    "trace_timestamp_quality",
    "trace_order",
    "trace_user_message_order",
    "trace_final_answer_order",
}
WEAVE_AGENTS_QUERY_SOURCE_KIND = "wandb_agents_api"
WEAVE_AGENTS_API_BASE_URL = "https://trace.wandb.ai"
WEAVE_AGENTS_QUERY_ENDPOINT = "/agents/query"
WEAVE_AGENTS_SPANS_QUERY_ENDPOINT = "/agents/spans/query"
WEAVE_AGENTS_QUERY_COUNT_FIELDS = (
    "agents_count",
    "spans_count",
    "matching_span_count",
    "latest_trace_span_count",
)
SCOPE_ATTESTATION_SCHEMA_VERSION = 1
MIN_SCOPE_CONFIRMATION_LENGTH = 20
PLACEHOLDER_ACCOUNTING_VALUES = {
    "あとで",
    "仮",
    "仮置き",
    "不明",
    "未確認",
    "未定",
    "要確認",
    "actual or billing estimate",
    "actualorbillingestimate",
    "bill or dashboard reference",
    "billordashboardreference",
    "dummy",
    "fill me",
    "fill in",
    "n a",
    "na",
    "none",
    "pending",
    "placeholder",
    "tbd",
    "todo",
    "unknown",
}
PLACEHOLDER_ACCOUNTING_PREFIXES = (
    "あとで",
    "仮",
    "不明",
    "未確認",
    "未定",
    "要確認",
    "dummy",
    "fill me",
    "fill in",
    "pending",
    "placeholder",
    "tbd",
    "todo",
    "unknown",
)
NEMOCLAW_REQUIRED_CHECKS = (
    "NeMoClaw command is available",
    "OpenShell command is available",
    "NeMoClaw version command succeeds",
    "NeMoClaw sandbox status succeeds",
    "OpenClaw runs inside NeMoClaw sandbox",
)


def weave_content_canary_commands() -> list[str]:
    approval_flag = (
        "--external-action-approval-source-packet-json "
        f"{EXTERNAL_ACTION_APPROVAL_SOURCE_PACKET_TEMPLATE} "
        "--external-action-approval-report-json "
        f"{EXTERNAL_ACTION_APPROVAL_REPORT_TEMPLATE}"
    )
    return [
        (
            "uv run python scripts/tools/run_weave_agents_content_canary.py "
            "--canary-id PREPARE_ONLY "
            "--model openai-direct/gpt-4.1-nano-2025-04-14 --thinking off "
            "--nemoclaw-sandbox nejumi-taiwan"
        ),
        (
            "uv run python scripts/tools/run_weave_agents_content_canary.py "
            "--execute --canary-id CONTENT_CANARY_YYYYMMDDTHHMM "
            "--model openai-direct/gpt-4.1-nano-2025-04-14 --thinking off --timeout 180 "
            "--nemoclaw-sandbox nejumi-taiwan "
            f"{approval_flag}"
        ),
        (
            "uv run python scripts/tools/check_taiwan_canary_readiness.py "
            f"--manifest {OPENAI_CANARY_MANIFEST} "
            f"--generated-full-dir {OPENAI_CANARY_FULL_DIR} "
            f"--generated-nonagentic-dir {OPENAI_CANARY_NONAGENTIC_DIR} "
            f"--generated-agentic-dir {OPENAI_CANARY_AGENTIC_NEMOCLAW_DIR} "
            f"--generated-agentic-aggregate-dir {OPENAI_CANARY_AGENTIC_AGGREGATE_DIR} "
            "--weave-content-canary-gate "
            "outputs/weave_agents_content_canary/plans/"
            "weave_agents_content_canary_CONTENT_CANARY_YYYYMMDDTHHMM.gate.json "
            "--require-weave-content-canary "
            "--json outputs/taiwan_full_eval/openai_canary_readiness_weave_content.json"
        ),
    ]


def _valid_nemoclaw_installer_review(payload: dict[str, Any]) -> bool:
    return (
        payload.get("schema_version") == 1
        and payload.get("ok") is True
        and payload.get("status") == "reviewed"
        and isinstance(payload.get("sha256"), str)
        and bool(re.fullmatch(r"[0-9a-fA-F]{64}", payload.get("sha256", "")))
        and payload.get("will_execute_installer") is False
        and payload.get("will_install_or_onboard") is False
        and payload.get("will_launch_model_inference") is False
        and payload.get("will_query_wandb") is False
    )


def latest_nemoclaw_installer_review(paths: list[Path]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for path in paths:
        payload, error = read_json(path)
        if payload is None or not _valid_nemoclaw_installer_review(payload):
            continue
        candidates.append(
            {
                "path": path_display(path),
                "mtime": path_mtime(path),
                "generated_at": payload.get("generated_at"),
                "installer_url": payload.get("installer_url"),
                "install_ref": payload.get("install_ref"),
                "lock_json": payload.get("lock_json"),
                "lock_verified": payload.get("lock_verified"),
                "expected_sha256": payload.get("expected_sha256"),
                "sha256": payload.get("sha256"),
                "size_bytes": payload.get("size_bytes"),
                "status": payload.get("status"),
                "ok": payload.get("ok"),
            }
        )
    return max(candidates, key=lambda item: item.get("mtime") or 0, default=None)


NEMOCLAW_SETUP_PLAN_COMMAND_KEYS = (
    "post_install_check_command",
    "installer_review_command",
    "production_install_and_onboard_command",
    "post_install_verification_command",
    "canary_readiness_command",
    "adoption_check_command",
    "production_readiness_command",
)


def _installer_review_sha(installer_review: dict[str, Any] | None = None) -> str:
    if not installer_review:
        return ""
    sha = str(installer_review.get("sha256") or "")
    return sha.lower() if re.fullmatch(r"[0-9a-fA-F]{64}", sha) else ""


def _commands_from_setup_plan(
    setup_plan: dict[str, Any] | None,
    installer_review: dict[str, Any] | None = None,
) -> list[str]:
    if not isinstance(setup_plan, dict):
        return []
    commands: list[str] = []
    for key in NEMOCLAW_SETUP_PLAN_COMMAND_KEYS:
        value = setup_plan.get(key)
        if key == "production_install_and_onboard_command" and not isinstance(value, str):
            value = setup_plan.get("install_and_onboard_command")
        if isinstance(value, str) and value.strip():
            commands.append(value.strip())
    if len(commands) != len(NEMOCLAW_SETUP_PLAN_COMMAND_KEYS):
        return []
    sha = _installer_review_sha(installer_review)
    if sha:
        commands = [
            command.replace("--installer-sha256 REVIEWED_INSTALLER_SHA256", f"--installer-sha256 {sha}")
            for command in commands
        ]
    return commands


def nemoclaw_commands(
    installer_review: dict[str, Any] | None = None,
    setup_plan: dict[str, Any] | None = None,
) -> list[str]:
    setup_plan_commands = _commands_from_setup_plan(setup_plan, installer_review)
    if setup_plan_commands:
        return setup_plan_commands
    review_json = "temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json"
    review_markdown = "temp/nemoclaw_installer_review_YYYYMMDDTHHMM.md"
    installer_url = "https://www.nvidia.com/nemoclaw.sh"
    install_ref = "lkg"
    installer_sha256 = "REVIEWED_INSTALLER_SHA256"
    expected_sha_arg = ""
    sha = _installer_review_sha(installer_review)
    if installer_review:
        installer_url = str(installer_review.get("installer_url") or installer_url)
        install_ref = str(installer_review.get("install_ref") or install_ref)
        if sha:
            installer_sha256 = sha
            expected_sha_arg = f" --expected-sha256 {installer_sha256}"
    return [
        "scripts/setup/install_nemoclaw.sh --check-only --json temp/nemoclaw_setup_check_YYYYMMDDTHHMM.json",
        (
            "uv run python scripts/setup/review_nemoclaw_installer.py "
            f"--url {installer_url} --install-ref {install_ref}"
            f"{expected_sha_arg} "
            f"--lock-json {NEMOCLAW_INSTALLER_LOCK_JSON} "
            f"--json {review_json} "
            f"--markdown {review_markdown}"
        ),
        (
            "scripts/setup/install_nemoclaw.sh --install --onboard "
            f"--install-ref {install_ref} "
            "--sandbox nejumi-taiwan --provider openai --policy-tier restricted "
            f"--installer-lock-json {NEMOCLAW_INSTALLER_LOCK_JSON} "
            f"--installer-sha256 {installer_sha256} "
            f"--installer-review-json {review_json} "
            "--yes-i-accept-third-party-software "
            "--json temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.json"
        ),
        (
            "uv run python scripts/setup/verify_nemoclaw_post_install.py "
            "--sandbox nejumi-taiwan "
            f"--canary-manifest {OPENAI_CANARY_MANIFEST} "
            f"--generated-full-dir {OPENAI_CANARY_FULL_DIR} "
            f"--generated-nonagentic-dir {OPENAI_CANARY_NONAGENTIC_DIR} "
            f"--generated-agentic-dir {OPENAI_CANARY_AGENTIC_NEMOCLAW_DIR} "
            f"--generated-agentic-aggregate-dir {OPENAI_CANARY_AGENTIC_AGGREGATE_DIR} "
            "--json temp/nemoclaw_post_install_verification_YYYYMMDDTHHMM.json "
            "--markdown temp/nemoclaw_post_install_verification_YYYYMMDDTHHMM.md "
            "--fail-on-failed"
        ),
        (
            "uv run python scripts/tools/check_taiwan_canary_readiness.py "
            f"--manifest {OPENAI_CANARY_MANIFEST} "
            f"--generated-full-dir {OPENAI_CANARY_FULL_DIR} "
            f"--generated-nonagentic-dir {OPENAI_CANARY_NONAGENTIC_DIR} "
            f"--generated-agentic-dir {OPENAI_CANARY_AGENTIC_NEMOCLAW_DIR} "
            f"--generated-agentic-aggregate-dir {OPENAI_CANARY_AGENTIC_AGGREGATE_DIR} "
            "--require-nemoclaw --nemoclaw-sandbox nejumi-taiwan "
            "--json outputs/taiwan_full_eval/openai_canary_readiness_nemoclaw.json"
        ),
        (
            "uv run python scripts/tools/check_taiwan_nemoclaw_adoption.py "
            "--setup-json temp/nemoclaw_setup_check_YYYYMMDDTHHMM.json "
            "--readiness-json outputs/taiwan_full_eval/openai_canary_readiness_nemoclaw.json "
            "--sandbox nejumi-taiwan "
            f"--agentic-config-glob {OPENAI_CANARY_AGENTIC_NEMOCLAW_DIR}/*.yaml "
            "--json temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.json "
            "--markdown temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.md "
            "--fail-on-not-adoptable"
        ),
        (
            "uv run python scripts/tools/run_taiwan_production_readiness_gate.py "
            "--report-json outputs/taiwan_full_eval/taiwan_production_readiness_report.json "
            "--fail-on-not-ready"
        ),
    ]


def wandb_completion_command(benchmark: str, *, run_id: str = "RUN_ID") -> str:
    if benchmark == "agentic_math":
        return (
            "uv run python scripts/tools/verify_taiwan_wandb_completion.py "
            f"--run-id {run_id} --benchmark agentic_math --expected-total 100 "
            "--env-file .env "
            f"--json outputs/taiwan_full_eval/wandb_completion/agentic_math-{run_id}.json"
        )
    if benchmark == "agentic_swe":
        return (
            "uv run python scripts/tools/verify_taiwan_wandb_completion.py "
            f"--run-id {run_id} --benchmark agentic_swe --expected-total 80 "
            "--env-file .env "
            f"--json outputs/taiwan_full_eval/wandb_completion/agentic_swe-{run_id}.json"
        )
    if benchmark == "taiwan_full":
        return (
            "uv run python scripts/tools/verify_taiwan_wandb_completion.py "
            f"--run-id {run_id} --benchmark taiwan_full --env-file .env "
            f"--json outputs/taiwan_full_eval/wandb_completion/taiwan_full-{run_id}.json"
        )
    return (
        "uv run python scripts/tools/verify_taiwan_wandb_completion.py "
        f"--run-id {run_id} --benchmark {benchmark} --env-file .env "
        f"--json outputs/taiwan_full_eval/wandb_completion/{benchmark}-{run_id}.json"
    )


def one_model_canary_commands() -> list[str]:
    budget_flag = f"--pre-run-budget-estimate-json {OPENAI_CANARY_BUDGET_ESTIMATE_JSON}"
    approval_flag = (
        "--external-action-approval-source-packet-json "
        f"{EXTERNAL_ACTION_APPROVAL_SOURCE_PACKET_TEMPLATE} "
        "--external-action-approval-report-json "
        f"{EXTERNAL_ACTION_APPROVAL_REPORT_TEMPLATE}"
    )
    return [
        (
            "uv run python scripts/tools/run_taiwan_full_eval_batch.py "
            f"--manifest {OPENAI_CANARY_MANIFEST} "
            "--canary --prepare-only --phase full "
            f"--generated-config-dir {OPENAI_CANARY_FULL_DIR} "
            "--output-root outputs/taiwan_full_eval "
            "--run-purpose 'Prepare OpenAI-direct gpt-4.1-mini one-model full canary metadata' "
            "--expected-cost-band 'prepare-only; no model API calls' "
            f"{budget_flag}"
        ),
        (
            "uv run python scripts/tools/run_taiwan_full_eval_batch.py "
            f"--manifest {OPENAI_CANARY_MANIFEST} "
            "--canary --phase nonagentic "
            f"--generated-config-dir {OPENAI_CANARY_NONAGENTIC_DIR} "
            "--output-root outputs/taiwan_full_eval "
            "--wandb-run-id-prefix twcanary-openai-mini-YYYYMMDD "
            "--yes --run-purpose 'OpenAI-direct gpt-4.1-mini one-model nonagentic phase' "
            "--expected-cost-band 'low-cost OpenAI-direct canary; confirm cap before execution' "
            f"{budget_flag} {approval_flag}"
        ),
        (
            "uv run python scripts/tools/run_taiwan_full_eval_batch.py "
            f"--manifest {OPENAI_CANARY_MANIFEST} "
            "--canary --phase agentic "
            f"--generated-config-dir {OPENAI_CANARY_AGENTIC_NEMOCLAW_DIR} "
            "--output-root outputs/taiwan_full_eval "
            "--agentic-math-nemoclaw-sandbox nejumi-taiwan "
            "--swebench-pro-nemoclaw-sandbox nejumi-taiwan "
            "--swebench-pro-nemoclaw-checkout-transfer-mode copy "
            "--require-nemoclaw-agentic-config "
            "--wandb-run-id-prefix twcanary-openai-mini-YYYYMMDD "
            "--verify-wandb-completion --verify-weave-agents "
            "--weave-agents-require-tool-span --weave-agents-require-tool-content "
            "--weave-agents-require-usage "
            "--weave-content-canary-gate WEAVE_CONTENT_CANARY_GATE "
            "--require-weave-content-canary "
            "--yes --run-purpose 'OpenAI-direct gpt-4.1-mini one-model agentic phase' "
            "--expected-cost-band 'low-cost OpenAI-direct canary; confirm cap before execution' "
            f"{budget_flag} {approval_flag}"
        ),
        (
            "uv run python scripts/tools/run_taiwan_full_eval_batch.py "
            f"--manifest {OPENAI_CANARY_MANIFEST} "
            "--canary --phase agentic_aggregate "
            f"--generated-config-dir {OPENAI_CANARY_AGENTIC_AGGREGATE_DIR} "
            "--output-root outputs/taiwan_full_eval "
            "--wandb-run-id-prefix twcanary-openai-mini-YYYYMMDD "
            "--verify-wandb-completion "
            "--yes --run-purpose 'OpenAI-direct gpt-4.1-mini one-model aggregate phase' "
            "--expected-cost-band 'no new OpenClaw generation expected' "
            f"{budget_flag} {approval_flag}"
        ),
    ]


def paid_run_review_commands() -> list[str]:
    return [
        "Review docs/taiwan_paid_run_review_template.md before substantial paid execution.",
        "For the current low-cost one-model canary path, run these phase commands after explicit cost approval.",
        *one_model_canary_commands(),
        (
            "For final or multi-model expansion, use the same command shape with an approved manifest, "
            "generated-config-dir, run purpose, expected cost band, and W&B run-id prefix."
        ),
        (
            "After completion, fill actual_cost_estimate/provider_bill_reference in "
            "outputs/taiwan_full_eval/*paid_run_review.json and rerun this readiness gate."
        ),
        (
            "If W&B completion verifier JSONs were produced after the run, sync them into "
            "the matching paid-run review. Use canary_agentic for agentic_math/agentic_swe "
            "and canary_agentic_aggregate for taiwan_full. First generate a dry-run report: "
            "uv run python "
            "scripts/tools/sync_wandb_completion_to_paid_review.py "
            "--review-json outputs/taiwan_full_eval/canary_agentic_paid_run_review.json "
            "--completion-json outputs/taiwan_full_eval/wandb_completion/agentic_math-RUN_ID.json "
            "--set-verify-wandb-completion "
            "--report-json temp/wandb_completion_agentic_math-RUN_ID.sync_dry_run.json"
        ),
        (
            "After reviewing that W&B completion dry-run report, apply the same sync with: "
            "uv run python scripts/tools/sync_wandb_completion_to_paid_review.py "
            "--review-json outputs/taiwan_full_eval/canary_agentic_paid_run_review.json "
            "--completion-json outputs/taiwan_full_eval/wandb_completion/agentic_math-RUN_ID.json "
            "--in-place --set-verify-wandb-completion "
            "--validated-dry-run-report-json temp/wandb_completion_agentic_math-RUN_ID.sync_dry_run.json"
        ),
        (
            "If adopting an already-finished W&B run into the review, use the "
            "W&B completion contract in the release gate. First refresh any verifier JSON "
            "that is not sync-ready, then use only the contract-provided dry-run and apply "
            "commands for sync-ready candidates."
        ),
        (
            "If Weave Agents verifier JSONs were produced after the agentic run, sync them into "
            "the matching paid-run review by first generating a dry-run report: uv run python "
            "scripts/tools/sync_weave_agents_completion_to_paid_review.py "
            "--review-json outputs/taiwan_full_eval/canary_agentic_paid_run_review.json "
            "--completion-json outputs/taiwan_full_eval/weave_agents_completion/agentic-MODEL_SLUG.json "
            "--run-id RUN_ID --set-verify-weave-agents "
            "--report-json temp/weave_agents_completion_agentic-MODEL_SLUG.sync_dry_run.json"
        ),
        (
            "After reviewing that Weave Agents dry-run report, apply the same sync with: uv run python "
            "scripts/tools/sync_weave_agents_completion_to_paid_review.py "
            "--review-json outputs/taiwan_full_eval/canary_agentic_paid_run_review.json "
            "--completion-json outputs/taiwan_full_eval/weave_agents_completion/agentic-MODEL_SLUG.json "
            "--run-id RUN_ID --in-place --set-verify-weave-agents "
            "--validated-dry-run-report-json temp/weave_agents_completion_agentic-MODEL_SLUG.sync_dry_run.json"
        ),
    ]


def paid_run_review_completion_requirements(
    *,
    wandb_completion_max_age_seconds: int | None = DEFAULT_WANDB_COMPLETION_MAX_AGE_SECONDS,
    weave_agents_completion_max_age_seconds: int | None = DEFAULT_WANDB_COMPLETION_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    return {
        "review_required_fields": [
            "status",
            "phase",
            "run_purpose",
            "expected_cost_band",
            "model_count",
            "configs",
            "run_eval_preflights",
            "created_at",
            "execution_plan_path",
            "batch_manifest_path",
            "post_run_cost_command",
            "pre_run_budget_estimate",
        ],
        "pre_run_budget_estimate_required_when": "requires_paid_model_api=true before paid execution",
        "pre_run_budget_estimate_required_fields": [
            "path",
            "sha256",
            "target_model",
            "price_per_million_tokens",
            "estimated_total_usd.low",
            "estimated_total_usd.mid",
            "estimated_total_usd.high",
            "pricing_source_url",
        ],
        "completed_review_required_fields": [
            "ended_at",
            "actual_cost_estimate",
            "provider_bill_reference",
            "runs",
        ],
        "run_required_fields": [
            "config",
            "preflight_json",
            "preflight_returncode",
            "preflight_ok",
            "log_path",
            "returncode",
            "started_at",
            "ended_at",
            "wandb_run_id for successful runs",
            "wandb_entity and wandb_project for successful W&B-verified runs",
        ],
        "run_eval_preflight_required_when": "before every run_eval.py invocation",
        "run_eval_preflight_required_fields": [
            "top-level run_eval_preflights[] with command/output_json/required_before_run_eval",
            "per-run preflight_json",
            "per-run preflight_returncode=0",
            "per-run preflight_ok=true",
            "preflight payload ok=true",
            "preflight payload status=passed",
            "preflight payload will_initialize_wandb=false",
            "preflight payload will_start_inference_engine=false",
            "preflight payload will_run_evaluators=false",
        ],
        "wandb_completion_required_when": "verify_wandb_completion=true",
        "wandb_completion_required_fields": ["benchmark", "path", "entity", "project", "run_id"],
        "wandb_completion_verifier_requirements": {
            "ok": True,
            "verification_schema_version": WANDB_COMPLETION_SCHEMA_VERSION,
            "observed_evidence_present": True,
            "observed_evidence.run_state": "finished",
            "required_evidence.run_metadata": "required for paid-review/release proof",
            "observed_evidence.run_metadata": "must match required run config/tags/group/job_type",
            "max_age_seconds": wandb_completion_max_age_seconds,
        },
        "weave_agents_completion_required_when": "verify_weave_agents=true for successful agentic runs",
        "weave_agents_completion_required_fields": ["path", "agent_name"],
        "weave_agents_completion_verifier_requirements": {
            "ok": True,
            "verification_schema_version": 1,
            "latest_trace_id_present": True,
            "all_checks_ok": True,
            "query_source.kind": WEAVE_AGENTS_QUERY_SOURCE_KIND,
            "query_source.api_base_url": WEAVE_AGENTS_API_BASE_URL,
            "query_source.agents_endpoint": WEAVE_AGENTS_QUERY_ENDPOINT,
            "query_source.spans_endpoint": WEAVE_AGENTS_SPANS_QUERY_ENDPOINT,
            "query_source.project_id": "must match verifier project_id",
            "query_source.agent_name": "must match verifier agent_name",
            "query_source.conversation_id": "must match required_evidence.conversation_id",
            "query_source.conversation_id_contains": "must match required_evidence.conversation_id_contains",
            "query_source.latest_trace_span_count": "must match latest_trace_spans_chronological",
            "required_checks": sorted(REQUIRED_WEAVE_AGENTS_CHECK_NAMES),
            "required_evidence.input_message_required": True,
            "required_evidence.trace_timestamp_quality_required": True,
            "required_evidence.trace_final_answer_order_required": True,
            "required_text_capture_when": "required_evidence.required_texts is non-empty",
            "content_capture_health.required_text_count": ">= len(required_evidence.required_texts) when required_texts is non-empty",
            "max_age_seconds": weave_agents_completion_max_age_seconds,
        },
        "accounting_required_fields": [
            "actual_cost_estimate",
            "provider_bill_reference",
        ],
        "accounting_value_requirements": {
            "actual_cost_estimate": "must be a concrete estimate, not TBD/TODO/pending/placeholder",
            "provider_bill_reference": "must be a concrete provider bill/export/reference, not TBD/TODO/pending/placeholder",
        },
    }


def existing_results_audit_commands() -> list[str]:
    return [
        (
            "uv run python scripts/tools/audit_taiwan_existing_results.py "
            "--json temp/taiwan_existing_results_audit_YYYYMMDDTHHMM.json "
            "--markdown temp/taiwan_existing_results_audit_YYYYMMDDTHHMM.md "
            "--fail-on-unformalized"
        ),
        (
            "For each agentic_math or agentic_swe local complete result reported as "
            "local_complete_needs_wandb_relog, run the benchmark-specific relogger "
            "(log_agentic_math_results_to_wandb.py or log_agentic_swe_results_to_wandb.py) "
            "and then verify_taiwan_wandb_completion.py. For taiwan_full provisional full "
            "runs, verify the source run_id directly with verify_taiwan_wandb_completion.py."
        ),
    ]


def repo_path(path: Path | str) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    return REPO_ROOT / value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"{path} does not exist"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{path} is not readable JSON: {exc}"
    if not isinstance(payload, dict):
        return None, f"{path} is not a JSON object"
    return payload, None


def discover_paths(patterns: list[str] | tuple[str, ...]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        for value in glob.glob(str(repo_path(pattern))):
            paths.add(Path(value))
    return sorted(paths, key=lambda path: str(path))


def normalize_paths(explicit: list[Path] | None, patterns: list[str] | tuple[str, ...]) -> list[Path]:
    if explicit:
        return [repo_path(path) for path in explicit]
    return discover_paths(patterns)


def parse_required_wandb_run_ids(
    values: list[str] | None,
    *,
    required_benchmarks: list[str] | None = None,
    all_run_id: str | None = None,
) -> dict[str, str]:
    result: dict[str, str] = {
        benchmark: all_run_id
        for benchmark in required_benchmarks or []
        if all_run_id
    }
    for value in values or []:
        if "=" not in value:
            raise ValueError(
                "--required-wandb-run-id must use BENCHMARK=RUN_ID format"
            )
        benchmark, run_id = value.split("=", 1)
        benchmark = benchmark.strip()
        run_id = run_id.strip()
        if benchmark not in {*DEFAULT_REQUIRED_WANDB_BENCHMARKS}:
            raise ValueError(f"unsupported W&B benchmark for run-id requirement: {benchmark}")
        if not run_id:
            raise ValueError(f"empty run id for W&B benchmark: {benchmark}")
        if benchmark in result and result[benchmark] != run_id:
            raise ValueError(
                "conflicting W&B run id requirements for "
                f"{benchmark}: {result[benchmark]} != {run_id}"
            )
        result[benchmark] = run_id
    return result


def filter_canary_readiness_paths(paths: list[Path]) -> list[Path]:
    return [
        path
        for path in paths
        if "production_readiness_report" not in path.name
        and "taiwan_production_readiness" not in path.name
    ]


def path_display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def source_path_key(path_value: Any) -> str:
    if not isinstance(path_value, str) or not path_value.strip():
        return ""
    return path_display(repo_path(path_value).resolve())


def source_audit_record_matches_wandb_entry(
    record: dict[str, Any],
    entry: dict[str, Any],
) -> bool:
    if record.get("benchmark") != entry.get("benchmark"):
        return False
    completion = record.get("wandb_completion")
    if not isinstance(completion, dict):
        return False
    if source_path_key(completion.get("path")) != source_path_key(entry.get("path")):
        return False
    for field in ("entity", "project", "run_id"):
        completion_value = completion.get(field)
        entry_value = entry.get(field)
        if not isinstance(completion_value, str) or not completion_value.strip():
            return False
        if completion_value != entry_value:
            return False
    return True


def path_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def numeric_timestamp(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def freshness_record(
    *,
    payload: dict[str, Any],
    path: Path,
    max_age_seconds: int | None,
    require_generated_at: bool = False,
) -> dict[str, Any]:
    generated_at = numeric_timestamp(payload.get("generated_at"))
    mtime = path_mtime(path)
    if max_age_seconds is not None and require_generated_at and generated_at is None:
        return {
            "generated_at": None,
            "mtime": mtime,
            "freshness_source": None,
            "age_seconds": None,
            "max_age_seconds": max_age_seconds,
            "fresh": False,
            "freshness_error": "missing generated_at",
        }
    freshness_source = "generated_at" if generated_at is not None else "mtime"
    freshness_timestamp = generated_at if generated_at is not None else mtime
    age_seconds = time.time() - freshness_timestamp if freshness_timestamp is not None else None
    fresh = max_age_seconds is None or (
        age_seconds is not None and age_seconds <= max_age_seconds
    )
    return {
        "generated_at": generated_at,
        "mtime": mtime,
        "freshness_source": freshness_source if freshness_timestamp is not None else None,
        "age_seconds": age_seconds,
        "max_age_seconds": max_age_seconds,
        "fresh": fresh,
        "freshness_error": "" if fresh else "stale",
    }


def gate_record(
    *,
    name: str,
    ok: bool,
    status: str,
    requirement: str,
    evidence_paths: list[Path],
    detail: str,
    next_action: str,
    blocking: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "name": name,
        "ok": ok,
        "blocking": blocking,
        "status": status,
        "requirement": requirement,
        "detail": detail,
        "evidence_paths": [path_display(path) for path in evidence_paths],
        "next_action": next_action,
    }
    if extra:
        record.update(extra)
    return record


def evaluate_weave_content_gate(
    paths: list[Path],
    *,
    require: bool,
    max_age_seconds: int | None = DEFAULT_WEAVE_CONTENT_CANARY_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    age_requirement = (
        f" and be no older than {max_age_seconds} seconds"
        if max_age_seconds is not None
        else ""
    )
    requirement = (
        "A fresh Weave Agents content canary gate must have ok=true and "
        "status=passed, backed by schema-v1 Weave verifier evidence with "
        "required trace_timestamp_quality, trace_order, "
        "trace_user_message_order, and trace_final_answer_order checks plus "
        "required_evidence.input_message_required=true and "
        "required_evidence.trace_timestamp_quality_required=true and "
        f"required_evidence.trace_final_answer_order_required=true{age_requirement}."
    )
    now = time.time()
    if not paths:
        return gate_record(
            name="weave_content_canary",
            ok=not require,
            blocking=require,
            status="missing" if require else "not_configured",
            requirement=requirement,
            evidence_paths=[],
            detail="No Weave content canary gate JSON was found.",
            next_action="Run run_weave_agents_content_canary.py --execute with an approved low-cost model, then pass the produced *.gate.json.",
            extra={"remediation_commands": weave_content_canary_commands()},
        )

    def latest_candidate_record(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not rows:
            return None
        return max(
            rows,
            key=lambda row: (
                row.get("generated_at")
                if isinstance(row.get("generated_at"), (int, float))
                else row.get("mtime")
                if isinstance(row.get("mtime"), (int, float))
                else -1.0,
                str(row.get("path") or ""),
            ),
        )

    def failed_next_action(latest: dict[str, Any] | None) -> str:
        if not latest:
            return "Fix the provider/runtime issue shown in the latest gate JSON and rerun the content canary."
        failure_kind = latest.get("failure_kind")
        status_value = latest.get("status")
        if failure_kind == "provider_quota":
            return "Use an approved provider/model with available quota, then rerun the live content canary."
        if failure_kind == "provider_auth":
            return "Fix provider credentials visible to OpenClaw, then rerun the live content canary."
        if failure_kind == "provider_rate_limit":
            return "Wait for rate-limit recovery or switch to an approved test model, then rerun the live content canary."
        if failure_kind == "model_not_found":
            return "Fix the OpenClaw model id/provider mapping, then rerun the live content canary."
        if status_value in {"content_missing", "tool_content_missing", "canary_text_missing"}:
            return "Fix native OpenClaw/Weave content capture so canary id, expected answer, and tool content are visible, then rerun."
        if status_value == "weave_verifier_schema_invalid":
            return "Regenerate the Weave Agents verifier JSON with the current verifier, then rerun the live content canary gate."
        return "Fix the provider/runtime issue shown in the latest gate JSON and rerun the content canary."

    candidates: list[dict[str, Any]] = []
    stale_passed_candidates: list[dict[str, Any]] = []
    for path in paths:
        payload, error = read_json(path)
        if payload is None:
            candidates.append({"path": path_display(path), "ok": False, "status": "invalid", "detail": error})
            continue
        generated_at = numeric_timestamp(payload.get("generated_at"))
        mtime = path_mtime(path)
        freshness_source = "generated_at" if generated_at is not None else "mtime"
        freshness_timestamp = generated_at if generated_at is not None else mtime
        age_seconds = now - freshness_timestamp if freshness_timestamp is not None else None
        is_fresh = (
            max_age_seconds is None
            or (age_seconds is not None and age_seconds <= max_age_seconds)
        )
        candidate = {
            "path": path_display(path),
            "ok": bool(payload.get("ok")),
            "status": payload.get("status"),
            "failure_kind": payload.get("failure_kind"),
            "detail": payload.get("detail"),
            "recommended_next_action": payload.get("recommended_next_action"),
            "model": payload.get("model"),
            "canary_id": payload.get("canary_id"),
            "generated_at": generated_at,
            "mtime": mtime,
            "freshness_source": freshness_source if freshness_timestamp is not None else None,
            "age_seconds": age_seconds,
            "max_age_seconds": max_age_seconds,
            "fresh": is_fresh,
        }
        candidates.append(candidate)
        if payload.get("ok") is True and payload.get("status") == "passed" and is_fresh:
            return gate_record(
                name="weave_content_canary",
                ok=True,
                blocking=require,
                status="passed",
                requirement=requirement,
                evidence_paths=[path],
                detail="A passing Weave content canary gate is present.",
                next_action="Use this gate path with --require-weave-content-canary for agentic/full batch execution.",
                extra={"candidates": candidates, "remediation_commands": []},
            )
        if payload.get("ok") is True and payload.get("status") == "passed" and not is_fresh:
            stale_passed_candidates.append(candidate)

    status = "stale" if stale_passed_candidates else "failed"
    latest_candidate = latest_candidate_record(candidates)
    detail = (
        "Only stale passing Weave content canary gates were supplied."
        if stale_passed_candidates
        else "No supplied Weave content canary gate passed."
    )
    if latest_candidate and not stale_passed_candidates:
        latest_status = latest_candidate.get("status")
        latest_failure = latest_candidate.get("failure_kind")
        latest_detail = latest_candidate.get("detail")
        latest_bits = [
            str(part)
            for part in (latest_status, latest_failure, latest_detail)
            if part
        ]
        if latest_bits:
            detail = f"{detail} Latest gate: {' / '.join(latest_bits)}."
    return gate_record(
        name="weave_content_canary",
        ok=not require,
        blocking=require,
        status=status,
        requirement=requirement,
        evidence_paths=paths,
        detail=detail,
        next_action=failed_next_action(latest_candidate),
        extra={
            "candidates": candidates,
            "latest_candidate": latest_candidate,
            "stale_passed_candidates": stale_passed_candidates,
            "max_age_seconds": max_age_seconds,
            "remediation_commands": weave_content_canary_commands(),
        },
    )


def _check_lookup(payload: dict[str, Any]) -> dict[str, bool]:
    result: dict[str, bool] = {}
    commands = payload.get("commands")
    if isinstance(commands, dict):
        nemoclaw = commands.get("nemoclaw")
        openshell = commands.get("openshell")
        if isinstance(nemoclaw, dict):
            result["NeMoClaw command is available"] = bool(nemoclaw.get("available"))
        if isinstance(openshell, dict):
            result["OpenShell command is available"] = bool(openshell.get("available"))
    checks = payload.get("checks")
    if not isinstance(checks, list):
        return result
    for check in checks:
        if not isinstance(check, dict):
            continue
        name = str(check.get("name") or "")
        for required in NEMOCLAW_REQUIRED_CHECKS:
            if name.startswith(required):
                result[required] = bool(check.get("ok"))
    return result


def _nemoclaw_ready(payload: dict[str, Any]) -> bool:
    lookup = _check_lookup(payload)
    return all(lookup.get(name) for name in NEMOCLAW_REQUIRED_CHECKS)


def _nemoclaw_source_type(path: Path, payload: dict[str, Any]) -> str:
    if isinstance(payload.get("commands"), dict):
        return "setup_check"
    if isinstance(payload.get("checks"), list):
        return "canary_readiness"
    return "unknown"


def _missing_nemoclaw_commands(lookup: dict[str, bool]) -> bool:
    return (
        lookup.get("NeMoClaw command is available") is False
        or lookup.get("OpenShell command is available") is False
    )


def _missing_required_nemoclaw_checks(lookup: dict[str, bool]) -> list[str]:
    return [name for name in NEMOCLAW_REQUIRED_CHECKS if not lookup.get(name)]


def _operation_failure(operation_results: Any, operation: str) -> dict[str, Any]:
    if not isinstance(operation_results, dict):
        return {}
    record = operation_results.get(operation)
    if not isinstance(record, dict):
        return {}
    failure = record.get("failure")
    if not isinstance(failure, dict):
        return {}
    return failure if str(failure.get("failure_kind") or "").strip() else {}


def evaluate_nemoclaw_readiness(
    paths: list[Path],
    *,
    require: bool,
    setup_paths: list[Path] | None = None,
    installer_review_paths: list[Path] | None = None,
) -> dict[str, Any]:
    requirement = "NeMoClaw, OpenShell, sandbox status, and sandbox OpenClaw preflight must pass."
    all_paths = [*paths, *(setup_paths or [])]
    latest_installer_review = latest_nemoclaw_installer_review(installer_review_paths or [])
    if not all_paths:
        return gate_record(
            name="nemoclaw_readiness",
            ok=not require,
            blocking=require,
            status="missing" if require else "not_configured",
            requirement=requirement,
            evidence_paths=[],
            detail="No readiness JSON was found.",
            next_action="Run check_taiwan_canary_readiness.py --require-nemoclaw after NeMoClaw onboarding.",
            extra={
                "latest_installer_review": latest_installer_review,
                "remediation_commands": nemoclaw_commands(latest_installer_review),
            },
        )

    summaries: list[dict[str, Any]] = []
    ready_candidates: list[tuple[Path, dict[str, Any]]] = []
    missing_command_setup_reports: list[dict[str, Any]] = []
    for path in all_paths:
        payload, error = read_json(path)
        if payload is None:
            summaries.append(
                {
                    "path": path_display(path),
                    "mtime": path_mtime(path),
                    "source_type": "invalid",
                    "ready": False,
                    "error": error,
                }
            )
            continue
        lookup = _check_lookup(payload)
        ready = _nemoclaw_ready(payload)
        summary = {
            "path": path_display(path),
            "mtime": path_mtime(path),
            "generated_at": payload.get("generated_at"),
            "source_type": _nemoclaw_source_type(path, payload),
            "report_ok": bool(payload.get("ok")),
            "ready": ready,
            "checks": lookup,
            "missing_required_checks": _missing_required_nemoclaw_checks(lookup),
        }
        if "host_prerequisites_ok" in payload:
            summary["host_prerequisites_ok"] = payload.get("host_prerequisites_ok")
        if "runtime_installed" in payload:
            summary["runtime_installed"] = payload.get("runtime_installed")
        if isinstance(payload.get("missing_required_commands"), list):
            summary["missing_required_commands"] = payload.get("missing_required_commands")
        if "sandbox_configured" in payload:
            summary["sandbox_configured"] = payload.get("sandbox_configured")
        if isinstance(payload.get("provider"), str):
            summary["provider"] = payload.get("provider")
        if isinstance(payload.get("model"), str):
            summary["model"] = payload.get("model")
        if isinstance(payload.get("provider_key_env"), str):
            summary["provider_key_env"] = payload.get("provider_key_env")
        if isinstance(payload.get("provider_preflight"), dict):
            summary["provider_preflight"] = payload.get("provider_preflight")
        if isinstance(payload.get("operation_results"), dict):
            summary["operation_results"] = payload.get("operation_results")
            summary["latest_onboard_failure"] = _operation_failure(
                payload.get("operation_results"),
                "onboard",
            )
        setup_plan = payload.get("setup_plan")
        if isinstance(setup_plan, dict):
            summary["setup_plan"] = setup_plan
        summaries.append(summary)
        if summary["source_type"] == "setup_check" and _missing_nemoclaw_commands(lookup):
            missing_command_setup_reports.append(summary)
        if ready:
            ready_candidates.append((path, summary))

    latest_missing_command_setup = max(
        missing_command_setup_reports,
        key=lambda item: item.get("mtime") or 0,
        default=None,
    )
    latest_missing_command_setup_mtime = (
        latest_missing_command_setup.get("mtime") if latest_missing_command_setup else None
    )
    valid_ready_candidates: list[tuple[Path, dict[str, Any]]] = []
    stale_ready_reports: list[dict[str, Any]] = []
    for path, summary in ready_candidates:
        ready_mtime = summary.get("mtime")
        if (
            latest_missing_command_setup_mtime is not None
            and ready_mtime is not None
            and ready_mtime < latest_missing_command_setup_mtime
        ):
            stale_ready_reports.append(summary)
            continue
        valid_ready_candidates.append((path, summary))

    latest_setup_report = max(
        (summary for summary in summaries if summary.get("source_type") == "setup_check"),
        key=lambda item: item.get("mtime") or 0,
        default=None,
    )
    latest_onboard_failure = (
        latest_setup_report.get("latest_onboard_failure")
        if isinstance(latest_setup_report, dict)
        and isinstance(latest_setup_report.get("latest_onboard_failure"), dict)
        else {}
    )
    latest_setup_plan = (
        latest_setup_report.get("setup_plan")
        if isinstance(latest_setup_report, dict)
        and isinstance(latest_setup_report.get("setup_plan"), dict)
        else None
    )
    provider_failure_kind = str(latest_onboard_failure.get("failure_kind") or "")
    failure_detail = str(latest_onboard_failure.get("failure_detail") or "")
    if valid_ready_candidates:
        path, _summary = max(
            valid_ready_candidates,
            key=lambda item: item[1].get("mtime") or 0,
        )
        return gate_record(
            name="nemoclaw_readiness",
            ok=True,
            blocking=require,
            status="passed",
            requirement=requirement,
            evidence_paths=[path],
            detail="NeMoClaw readiness checks passed.",
            next_action="Keep using the same sandbox name in agentic config generation and batch execution.",
            extra={
                "reports": summaries,
                "latest_setup_report": latest_setup_report,
                "recommended_setup_plan": (
                    latest_setup_report.get("setup_plan")
                    if isinstance(latest_setup_report, dict)
                    else None
                ),
                "stale_ready_reports": stale_ready_reports,
                "remediation_commands": [],
                "latest_installer_review": latest_installer_review,
                "latest_onboard_failure": latest_onboard_failure,
            },
        )

    detail = (
        "No readiness report proves NeMoClaw is installed, onboarded, and able to run OpenClaw in the sandbox."
    )
    next_action = (
        "Install/onboard NeMoClaw with explicit third-party acceptance, then rerun check_taiwan_canary_readiness.py --require-nemoclaw."
    )
    if provider_failure_kind == "provider_quota":
        detail += f" Latest onboarding failure: {failure_detail or 'provider quota'}."
        next_action = (
            "Retry NeMoClaw onboarding with a provider account that has available quota, "
            "then rerun --check-only, post-install verification, and canary readiness."
        )
    elif provider_failure_kind == "provider_auth":
        detail += f" Latest onboarding failure: {failure_detail or 'provider authentication'}."
        next_action = (
            "Fix provider credentials for the selected NeMoClaw provider, then rerun onboarding "
            "and post-install verification."
        )
    elif provider_failure_kind == "gateway_port_unavailable":
        detail += f" Latest onboarding failure: {failure_detail or 'gateway port unavailable'}."
        next_action = (
            "Choose a free --gateway-port or stop the stale OpenShell gateway, then rerun onboarding "
            "and post-install verification."
        )

    return gate_record(
        name="nemoclaw_readiness",
        ok=not require,
        blocking=require,
        status="failed",
        requirement=requirement,
        evidence_paths=all_paths,
        detail=detail,
        next_action=next_action,
        extra={
            "reports": summaries,
            "latest_setup_report": latest_setup_report,
            "recommended_setup_plan": (
                latest_setup_report.get("setup_plan")
                if isinstance(latest_setup_report, dict)
                else None
            ),
            "stale_ready_reports": stale_ready_reports,
            "latest_installer_review": latest_installer_review,
            "latest_onboard_failure": latest_onboard_failure,
            "remediation_commands": nemoclaw_commands(
                latest_installer_review,
                latest_setup_plan,
            ),
        },
    )


def evaluate_nemoclaw_operator_docs(paths: list[Path], *, require: bool) -> dict[str, Any]:
    requirement = (
        "NeMoClaw operator documentation must be verified against the release "
        "evidence contract, including locked installer review, install/onboard, "
        "post-install verification, canary readiness, fail-fast adoption check, "
        "fail-fast production readiness, restricted policy tier, explicit "
        "third-party acceptance, and no OpenRouter markers."
    )
    if not paths:
        return gate_record(
            name="nemoclaw_operator_docs",
            ok=not require,
            blocking=require,
            status="missing" if require else "not_configured",
            requirement=requirement,
            evidence_paths=[],
            detail="No NeMoClaw operator docs verification JSON was found.",
            next_action=(
                "Run uv run python scripts/setup/verify_nemoclaw_operator_docs.py "
                "--json temp/nemoclaw_operator_docs_verification_TIMESTAMP.json "
                "--markdown temp/nemoclaw_operator_docs_verification_TIMESTAMP.md "
                "--fail-on-failed."
            ),
        )

    summaries: list[dict[str, Any]] = []
    for path in paths:
        payload, error = read_json(path)
        if payload is None:
            summaries.append(
                {
                    "path": path_display(path),
                    "mtime": path_mtime(path),
                    "ok": False,
                    "status": "invalid_json",
                    "error": error,
                }
            )
            continue
        checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []
        check_names = {
            str(check.get("name") or "")
            for check in checks
            if isinstance(check, dict)
        }
        missing_required_checks = sorted(
            REQUIRED_NEMOCLAW_OPERATOR_DOC_CHECK_NAMES - check_names
        )
        failed_checks = [
            str(check.get("name") or "unknown")
            for check in checks
            if isinstance(check, dict) and not check.get("ok")
        ]
        no_external_action_flags_ok = all(
            payload.get(field) is False
            for field in (
                "will_execute_installer",
                "will_install_or_onboard",
                "will_launch_model_inference",
                "will_query_wandb",
            )
        )
        schema_ok = payload.get("schema_version") == 1
        status_ok = payload.get("status") == "passed"
        ok = (
            schema_ok
            and status_ok
            and payload.get("ok") is True
            and bool(checks)
            and not missing_required_checks
            and not failed_checks
            and no_external_action_flags_ok
        )
        summaries.append(
            {
                "path": path_display(path),
                "mtime": path_mtime(path),
                "generated_at": payload.get("generated_at"),
                "ok": ok,
                "payload_ok": payload.get("ok"),
                "status": payload.get("status"),
                "schema_version": payload.get("schema_version"),
                "failed_checks": failed_checks,
                "missing_required_checks": missing_required_checks,
                "missing_requirements": payload.get("missing_requirements"),
                "no_external_action_flags_ok": no_external_action_flags_ok,
                "readme_path": payload.get("readme_path"),
                "lock_json": payload.get("lock_json"),
                "lock_summary": payload.get("lock_summary"),
            }
        )

    latest = max(summaries, key=lambda item: item.get("mtime") or 0, default=None)
    if latest and latest.get("ok"):
        return gate_record(
            name="nemoclaw_operator_docs",
            ok=True,
            blocking=False,
            status="passed",
            requirement=requirement,
            evidence_paths=[repo_path(str(latest["path"]))],
            detail="Latest NeMoClaw operator documentation verification passed.",
            next_action="Keep README_nemoclaw.md and the setup scripts in sync.",
            extra={"latest_report": latest, "reports": summaries},
        )

    return gate_record(
        name="nemoclaw_operator_docs",
        ok=False,
        blocking=require,
        status="failed" if latest else "missing",
        requirement=requirement,
        evidence_paths=[repo_path(str(latest["path"]))] if latest else [],
        detail="Latest NeMoClaw operator documentation verification did not pass.",
        next_action=(
            "Update docs/README_nemoclaw.md to match the setup/release contract, "
            "then rerun verify_nemoclaw_operator_docs.py."
        ),
        extra={"latest_report": latest or {}, "reports": summaries},
    )


def evaluate_metadata_readiness(paths: list[Path], *, require: bool) -> dict[str, Any]:
    requirement = "Canary config, artifact, credential, and OpenClaw metadata readiness must pass."
    passing: list[Path] = []
    summaries: list[dict[str, Any]] = []
    for path in paths:
        payload, error = read_json(path)
        if payload is None:
            summaries.append({"path": path_display(path), "ok": False, "error": error})
            continue
        summaries.append({"path": path_display(path), "ok": bool(payload.get("ok"))})
        if payload.get("ok") is True:
            passing.append(path)
    if passing:
        return gate_record(
            name="canary_metadata_readiness",
            ok=True,
            blocking=require,
            status="passed",
            requirement=requirement,
            evidence_paths=passing,
            detail="At least one canary readiness report passed.",
            next_action="Use the latest passing readiness JSON in the paid-run review package.",
            extra={"reports": summaries},
        )
    return gate_record(
        name="canary_metadata_readiness",
        ok=not require,
        blocking=require,
        status="missing" if not paths else "failed",
        requirement=requirement,
        evidence_paths=paths,
        detail="No canary readiness report passed.",
        next_action="Run check_taiwan_canary_readiness.py and resolve failed config/artifact/credential checks.",
        extra={"reports": summaries},
    )


def evaluate_wandb_completion(
    paths: list[Path],
    *,
    required_benchmarks: list[str],
    required_run_ids: dict[str, str] | None = None,
    require: bool,
    max_age_seconds: int | None = DEFAULT_WANDB_COMPLETION_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    age_requirement = (
        f" and be no older than {max_age_seconds} seconds"
        if max_age_seconds is not None
        else ""
    )


    requirement = (
        "Required benchmark completion verifier JSONs must have ok=true, "
        f"include entity/project/run_id and generated_at{age_requirement}, use verification_schema_version="
        f"{WANDB_COMPLETION_SCHEMA_VERSION}, and contain observed_evidence proving "
        "a finished W&B run with logged metrics, tables, or artifacts."
    )
    required_run_ids = required_run_ids or {}
    completed: dict[str, list[str]] = {}
    records: list[dict[str, Any]] = []
    run_id_mismatches: list[dict[str, Any]] = []
    stale_completion_records: list[dict[str, Any]] = []
    legacy_schema_records: list[dict[str, Any]] = []
    invalid_evidence_records: list[dict[str, Any]] = []
    for path in paths:
        payload, error = read_json(path)
        if payload is None:
            records.append(
                {
                    "path": path_display(path),
                    "mtime": path_mtime(path),
                    "ok": False,
                    "error": error,
                }
            )
            continue
        benchmark = payload.get("benchmark")
        entity = payload.get("entity")
        project = payload.get("project")
        run_id = payload.get("run_id")
        identity_valid = (
            isinstance(entity, str)
            and bool(entity.strip())
            and isinstance(project, str)
            and bool(project.strip())
            and isinstance(run_id, str)
            and bool(run_id.strip())
        )
        expected_run_id = (
            required_run_ids.get(benchmark)
            if isinstance(benchmark, str)
            else None
        )
        run_id_matches = expected_run_id is None or run_id == expected_run_id
        freshness = freshness_record(
            payload=payload,
            path=path,
            max_age_seconds=max_age_seconds,
            require_generated_at=True,
        )
        schema_version = payload.get("verification_schema_version")
        schema_valid = schema_version == WANDB_COMPLETION_SCHEMA_VERSION
        observed_evidence = payload.get("observed_evidence")
        observed_evidence_valid = _observed_evidence_valid(observed_evidence)
        if not isinstance(observed_evidence, dict):
            observed_evidence = {}
        if payload.get("ok") is True and not run_id_matches:
            run_id_mismatches.append(
                {
                    "path": path_display(path),
                    "benchmark": benchmark,
                    "run_id": run_id,
                    "expected_run_id": expected_run_id,
                }
            )
        record = {
            "path": path_display(path),
            "benchmark": benchmark,
            "ok": bool(payload.get("ok")),
            "entity": entity,
            "project": project,
            "run_id": run_id,
            "identity_valid": identity_valid,
            "expected_run_id": expected_run_id,
            "run_id_matches": run_id_matches,
            "verification_schema_version": schema_version,
            "schema_valid": schema_valid,
            "observed_evidence": observed_evidence,
            "observed_evidence_valid": observed_evidence_valid,
            **freshness,
        }
        records.append(record)
        if payload.get("ok") is True and not schema_valid:
            legacy_schema_records.append(record)
        if payload.get("ok") is True and run_id_matches and not freshness["fresh"]:
            stale_completion_records.append(record)
        if (
            payload.get("ok") is True
            and run_id_matches
            and freshness["fresh"]
            and (not identity_valid or not schema_valid or not observed_evidence_valid)
        ):
            invalid_evidence_records.append(record)
        if (
            isinstance(benchmark, str)
            and payload.get("ok") is True
            and run_id_matches
            and freshness["fresh"]
            and identity_valid
            and schema_valid
            and observed_evidence_valid
        ):
            completed.setdefault(benchmark, []).append(path_display(path))

    missing = [benchmark for benchmark in required_benchmarks if benchmark not in completed]
    ok = not missing
    stale_benchmarks = {
        record.get("benchmark")
        for record in stale_completion_records
        if isinstance(record.get("benchmark"), str)
    }
    missing_generated_at_records = [
        record
        for record in stale_completion_records
        if record.get("freshness_error") == "missing generated_at"
    ]
    invalid_evidence_benchmarks = {
        record.get("benchmark")
        for record in invalid_evidence_records
        if isinstance(record.get("benchmark"), str)
    }
    remediation_commands = [
        wandb_completion_command(
            benchmark,
            run_id=required_run_ids.get(benchmark, "RUN_ID"),
        )
        for benchmark in missing
    ]
    return gate_record(
        name="wandb_completion",
        ok=ok or not require,
        blocking=require,
        status=(
            "passed"
            if ok
            else "missing_generated_at"
            if missing_generated_at_records and all(benchmark in stale_benchmarks for benchmark in missing)
            else "stale"
            if stale_completion_records and all(benchmark in stale_benchmarks for benchmark in missing)
            else "run_id_mismatch"
            if run_id_mismatches and all(benchmark in required_run_ids for benchmark in missing)
            else "invalid_evidence"
            if invalid_evidence_records and all(benchmark in invalid_evidence_benchmarks for benchmark in missing)
            else "missing_required_benchmarks"
        ),
        requirement=requirement,
        evidence_paths=paths,
        detail=(
            "All required W&B completion verifier JSONs are present."
            if ok
            else f"Missing generated_at in completion verifier JSON for: {', '.join(missing)}"
            if missing_generated_at_records and all(benchmark in stale_benchmarks for benchmark in missing)
            else f"Stale completion verifier JSON for: {', '.join(missing)}"
            if stale_completion_records and all(benchmark in stale_benchmarks for benchmark in missing)
            else f"Completion verifier JSON lacks current schema or valid observed evidence for: {', '.join(missing)}"
            if invalid_evidence_records and all(benchmark in invalid_evidence_benchmarks for benchmark in missing)
            else f"Missing passing completion verifier JSON for: {', '.join(missing)}"
        ),
        next_action=(
            "Include these verifier JSONs in the release package."
            if ok
            else "Run verify_taiwan_wandb_completion.py for each required benchmark after the run finishes."
        ),
        extra={
            "required_benchmarks": required_benchmarks,
            "required_run_ids": required_run_ids,
            "completed": completed,
            "records": records,
            "missing_benchmarks": missing,
            "run_id_mismatches": run_id_mismatches,
            "stale_completion_records": stale_completion_records,
            "missing_generated_at_records": missing_generated_at_records,
            "legacy_schema_records": legacy_schema_records,
            "invalid_evidence_records": invalid_evidence_records,
            "required_verification_schema_version": WANDB_COMPLETION_SCHEMA_VERSION,
            "max_age_seconds": max_age_seconds,
            "remediation_commands": remediation_commands,
        },
    )


def evaluate_existing_results_formalization(paths: list[Path], *, require: bool) -> dict[str, Any]:
    requirement = (
        "Every complete local Taiwan benchmark result must either be formalized "
        "by a passing W&B completion verifier JSON or be explicitly absent from "
        "the release evidence as partial/probe output."
    )
    if not paths:
        return gate_record(
            name="existing_results_formalization",
            ok=not require,
            blocking=require,
            status="missing" if require else "not_configured",
            requirement=requirement,
            evidence_paths=[],
            detail="No existing-results audit JSON was found.",
            next_action="Run audit_taiwan_existing_results.py before release evidence generation.",
            extra={"audits": [], "remediation_commands": existing_results_audit_commands()},
        )

    audits: list[dict[str, Any]] = []
    for path in paths:
        payload, error = read_json(path)
        if payload is None:
            audits.append(
                {
                    "path": path_display(path),
                    "ok": False,
                    "status": "invalid",
                    "generated_at": None,
                    "mtime": path_mtime(path),
                    "error": error,
                }
            )
            continue
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        audits.append(
            {
                "path": path_display(path),
                "ok": bool(payload.get("ok")),
                "status": payload.get("status"),
                "generated_at": payload.get("generated_at"),
                "mtime": path_mtime(path),
                "record_count": summary.get("record_count"),
                "complete_local_count": summary.get("complete_local_count"),
                "formalized_wandb_complete_count": summary.get("formalized_wandb_complete_count"),
                "unformalized_complete_count": summary.get("unformalized_complete_count"),
                "partial_or_probe_count": summary.get("partial_or_probe_count"),
                "unformalized_complete_records": payload.get("unformalized_complete_records", []),
            }
        )
    latest = max(
        audits,
        key=lambda item: (
            numeric_timestamp(item.get("generated_at")) or item.get("mtime") or 0
        ),
        default=None,
    )
    ok = bool(latest and latest.get("ok"))
    return gate_record(
        name="existing_results_formalization",
        ok=ok or not require,
        blocking=require,
        status=(
            "passed"
            if ok
            else "unformalized_complete_results"
            if latest and latest.get("status") == "unformalized_complete_results"
            else "invalid"
            if latest
            else "missing"
        ),
        requirement=requirement,
        evidence_paths=[repo_path(latest["path"])] if latest and isinstance(latest.get("path"), str) else paths,
        detail=(
            "Existing complete local results are formalized in W&B or classified as partial/probe."
            if ok
            else "At least one complete local result is not backed by passing W&B completion evidence."
            if latest and latest.get("status") == "unformalized_complete_results"
            else "The latest existing-results audit is invalid or not passing."
        ),
        next_action=(
            "Keep this audit JSON in the release evidence bundle."
            if ok
            else "Relog each complete local result to W&B, verify completion, and rerun audit_taiwan_existing_results.py."
        ),
        extra={
            "latest_audit": latest,
            "audits": audits,
            "remediation_commands": [] if ok else existing_results_audit_commands(),
        },
    )


def _review_run_ids(payload: dict[str, Any]) -> list[str]:
    run_ids: list[str] = []
    top_level = payload.get("wandb_run_id")
    if isinstance(top_level, str) and top_level:
        run_ids.append(top_level)
    runs = payload.get("runs")
    if isinstance(runs, list):
        for row in runs:
            if not isinstance(row, dict):
                continue
            run_id = row.get("wandb_run_id")
            if isinstance(run_id, str) and run_id:
                run_ids.append(run_id)
    return sorted(set(run_ids))


def _review_wandb_completion_entries(
    payload: dict[str, Any],
    *,
    review_path: str = "",
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    def append_entries(
        rows: Any,
        *,
        parent_run_id: str = "",
        parent_entity: str = "",
        parent_project: str = "",
    ) -> None:
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            benchmark = row.get("benchmark")
            if not isinstance(benchmark, str) or not benchmark:
                continue
            run_id = row.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                run_id = parent_run_id
            entries.append(
                {
                    "benchmark": benchmark,
                    "ok": bool(row.get("ok")),
                    "path": row.get("path") if isinstance(row.get("path"), str) else "",
                    "sha256": row.get("sha256") if isinstance(row.get("sha256"), str) else "",
                    "review_path": review_path,
                    "entity": row.get("entity") if isinstance(row.get("entity"), str) else "",
                    "project": row.get("project") if isinstance(row.get("project"), str) else "",
                    "run_id": run_id,
                    "parent_run_id": parent_run_id,
                    "parent_entity": parent_entity,
                    "parent_project": parent_project,
                    "adopted_existing_result": bool(row.get("adopted_existing_result")),
                    "scope_attestation": row.get("scope_attestation")
                    if isinstance(row.get("scope_attestation"), dict)
                    else None,
                    "sync_dry_run_report_json": (
                        row.get("sync_dry_run_report_json")
                        if isinstance(row.get("sync_dry_run_report_json"), str)
                        else ""
                    ),
                    "sync_dry_run_source_review_json": (
                        row.get("sync_dry_run_source_review_json")
                        if isinstance(row.get("sync_dry_run_source_review_json"), str)
                        else ""
                    ),
                    "sync_dry_run_source_review_sha256": (
                        row.get("sync_dry_run_source_review_sha256")
                        if isinstance(row.get("sync_dry_run_source_review_sha256"), str)
                        else ""
                    ),
                }
            )

    append_entries(payload.get("wandb_completion"))
    runs = payload.get("runs")
    if isinstance(runs, list):
        for run in runs:
            if not isinstance(run, dict):
                continue
            run_id = run.get("wandb_run_id")
            entity = run.get("wandb_entity")
            project = run.get("wandb_project")
            append_entries(
                run.get("wandb_completion"),
                parent_run_id=run_id if isinstance(run_id, str) else "",
                parent_entity=entity if isinstance(entity, str) else "",
                parent_project=project if isinstance(project, str) else "",
            )
    return entries


def _review_weave_agents_completion_entries(
    payload: dict[str, Any],
    *,
    review_path: str = "",
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    def append_entry(row: Any, *, parent_run_id: str = "") -> None:
        if not isinstance(row, dict):
            return
        path_value = row.get("path")
        agent_name = row.get("agent_name")
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            run_id = parent_run_id
        entries.append(
            {
                "ok": bool(row.get("ok")),
                "path": path_value if isinstance(path_value, str) else "",
                "agent_name": agent_name if isinstance(agent_name, str) else "",
                "run_id": run_id,
                "review_path": review_path,
                "sync_dry_run_report_json": (
                    row.get("sync_dry_run_report_json")
                    if isinstance(row.get("sync_dry_run_report_json"), str)
                    else ""
                ),
                "sync_dry_run_source_review_json": (
                    row.get("sync_dry_run_source_review_json")
                    if isinstance(row.get("sync_dry_run_source_review_json"), str)
                    else ""
                ),
                "sync_dry_run_source_review_sha256": (
                    row.get("sync_dry_run_source_review_sha256")
                    if isinstance(row.get("sync_dry_run_source_review_sha256"), str)
                    else ""
                ),
            }
        )

    append_entry(payload.get("weave_agents_completion"))
    runs = payload.get("runs")
    if isinstance(runs, list):
        for run in runs:
            if not isinstance(run, dict):
                continue
            run_id = run.get("wandb_run_id")
            append_entry(
                run.get("weave_agents_completion"),
                parent_run_id=run_id if isinstance(run_id, str) else "",
            )
    return entries


def _wandb_completion_query_source_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    query_source = payload.get("query_source")
    if not isinstance(query_source, dict):
        return ["query_source is not an object"]
    entity = payload.get("entity")
    project = payload.get("project")
    run_id = payload.get("run_id")
    expected_query_fields = {
        "kind": WANDB_COMPLETION_QUERY_SOURCE_KIND,
        "api": "wandb.Api",
        "timeout_seconds": WANDB_COMPLETION_API_TIMEOUT_SECONDS,
        "entity": entity,
        "project": project,
        "run_id": run_id,
        "benchmark": payload.get("benchmark"),
        "summary_source": "run.summary_metrics",
        "artifact_source": "run.logged_artifacts",
        "history_scanned": False,
    }
    if (
        isinstance(entity, str)
        and entity.strip()
        and isinstance(project, str)
        and project.strip()
        and isinstance(run_id, str)
        and run_id.strip()
    ):
        expected_query_fields["run_path"] = f"{entity}/{project}/{run_id}"
    for field, expected in expected_query_fields.items():
        if query_source.get(field) != expected:
            errors.append(
                f"query_source.{field} mismatch: expected {expected}, got {query_source.get(field)}"
            )
    return errors


def _observed_run_config_by_key(observed_metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = observed_metadata.get("config")
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("key")): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("key"), str) and row.get("key")
    }


def _wandb_completion_run_metadata_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = payload.get("required_evidence")
    if not isinstance(required, dict):
        errors.append("required_evidence.run_metadata is not an object")
        required = {}
    required_metadata = required.get("run_metadata")
    if not isinstance(required_metadata, dict) or not required_metadata:
        errors.append("required_evidence.run_metadata is not an object")
        required_metadata = {}

    observed = payload.get("observed_evidence")
    observed_metadata = (
        observed.get("run_metadata")
        if isinstance(observed, dict)
        else None
    )
    if not isinstance(observed_metadata, dict) or not observed_metadata:
        errors.append("observed_evidence.run_metadata is not an object")
        observed_metadata = {}

    has_requirement = False
    observed_config = _observed_run_config_by_key(observed_metadata)
    config_rows = required_metadata.get("config")
    if isinstance(config_rows, list):
        for row in config_rows:
            if not isinstance(row, dict):
                continue
            key = row.get("key")
            if not isinstance(key, str) or not key:
                continue
            if "expected" not in row:
                continue
            has_requirement = True
            expected = row.get("expected")
            observed_row = observed_config.get(key)
            if not isinstance(observed_row, dict):
                errors.append(f"observed_evidence.run_metadata.config missing {key}")
                continue
            if observed_row.get("present") is not True:
                errors.append(f"observed_evidence.run_metadata.config {key} is not present")
            if observed_row.get("value") != expected:
                errors.append(f"observed_evidence.run_metadata.config {key} value mismatch")

    observed_tags = observed_metadata.get("tags")
    if not isinstance(observed_tags, list):
        observed_tags = []
    tags = required_metadata.get("tags")
    if isinstance(tags, list):
        for tag in tags:
            if not isinstance(tag, str) or not tag:
                continue
            has_requirement = True
            if tag not in observed_tags:
                errors.append(f"observed_evidence.run_metadata.tags missing {tag}")

    for field in ("group", "job_type"):
        expected = required_metadata.get(field)
        if not isinstance(expected, str) or not expected:
            continue
        has_requirement = True
        if observed_metadata.get(field) != expected:
            errors.append(f"observed_evidence.run_metadata.{field} mismatch")

    if not has_requirement:
        errors.append(
            "required_evidence.run_metadata must specify at least one config, tag, group, or job_type requirement"
        )
    return errors


def _verify_review_wandb_completion_entry(
    entry: dict[str, Any],
    *,
    max_age_seconds: int | None,
) -> dict[str, Any]:
    result = dict(entry)
    result["entry_ok"] = bool(entry.get("ok"))
    result["verifier_json_ok"] = False
    result["verified"] = False
    path_value = entry.get("path")
    if not isinstance(path_value, str) or not path_value:
        result["verification_error"] = "missing verifier JSON path"
        return result

    verifier_path = repo_path(path_value)
    payload, error = read_json(verifier_path)
    if payload is None:
        result["verification_error"] = error or "verifier JSON could not be read"
        return result
    try:
        actual_sha256 = sha256_file(verifier_path)
    except OSError:
        actual_sha256 = ""
    declared_sha256 = entry.get("sha256") if isinstance(entry.get("sha256"), str) else ""
    sha256_matches = bool(declared_sha256) and declared_sha256 == actual_sha256

    benchmark = entry.get("benchmark")
    run_id = entry.get("run_id")
    payload_benchmark = payload.get("benchmark")
    payload_entity = payload.get("entity")
    payload_project = payload.get("project")
    payload_run_id = payload.get("run_id")
    benchmark_matches = payload_benchmark == benchmark
    entity_matches = isinstance(payload_entity, str) and bool(payload_entity.strip()) and payload_entity == entry.get("entity")
    project_matches = isinstance(payload_project, str) and bool(payload_project.strip()) and payload_project == entry.get("project")
    run_id_matches = not run_id or payload_run_id == run_id
    parent_run_id = entry.get("parent_run_id") if isinstance(entry.get("parent_run_id"), str) else ""
    parent_entity = entry.get("parent_entity") if isinstance(entry.get("parent_entity"), str) else ""
    parent_project = entry.get("parent_project") if isinstance(entry.get("parent_project"), str) else ""
    parent_run_id_matches = not parent_run_id or payload_run_id == parent_run_id
    parent_entity_matches = not parent_entity or payload_entity == parent_entity
    parent_project_matches = not parent_project or payload_project == parent_project
    payload_ok = payload.get("ok") is True
    schema_version = payload.get("verification_schema_version")
    schema_valid = schema_version == WANDB_COMPLETION_SCHEMA_VERSION
    observed_evidence = payload.get("observed_evidence")
    observed_evidence_valid = _observed_evidence_valid(observed_evidence)
    if not isinstance(observed_evidence, dict):
        observed_evidence = {}
    query_source_required = not bool(entry.get("adopted_existing_result"))
    query_source_errors = (
        _wandb_completion_query_source_errors(payload)
        if query_source_required
        else []
    )
    query_source_valid = not query_source_errors
    query_source = (
        payload.get("query_source")
        if isinstance(payload.get("query_source"), dict)
        else {}
    )
    run_metadata_errors = _wandb_completion_run_metadata_errors(payload)
    run_metadata_valid = not run_metadata_errors
    freshness = freshness_record(
        payload=payload,
        path=verifier_path,
        max_age_seconds=max_age_seconds,
        require_generated_at=True,
    )
    scope_attestation = _verify_scope_attestation(entry)
    sync_dry_run = _verify_wandb_sync_dry_run_report(entry)
    result.update(
        {
            "verifier_json_ok": payload_ok,
            "verifier_json_benchmark": payload_benchmark,
            "verifier_json_entity": payload_entity,
            "verifier_json_project": payload_project,
            "verifier_json_run_id": payload_run_id,
            "benchmark_matches": benchmark_matches,
            "entity_matches": entity_matches,
            "project_matches": project_matches,
            "run_id_matches": run_id_matches,
            "parent_run_id": parent_run_id,
            "parent_entity": parent_entity,
            "parent_project": parent_project,
            "parent_run_id_matches": parent_run_id_matches,
            "parent_entity_matches": parent_entity_matches,
            "parent_project_matches": parent_project_matches,
            "verification_schema_version": schema_version,
            "schema_valid": schema_valid,
            "observed_evidence": observed_evidence,
            "observed_evidence_valid": observed_evidence_valid,
            "query_source_required": query_source_required,
            "query_source": query_source,
            "query_source_valid": query_source_valid,
            "query_source_errors": query_source_errors,
            "run_metadata_valid": run_metadata_valid,
            "run_metadata_errors": run_metadata_errors,
            "sha256": declared_sha256,
            "sha256_actual": actual_sha256,
            "sha256_matches": sha256_matches,
            "scope_attestation": scope_attestation,
            "scope_attestation_valid": bool(scope_attestation.get("verified")),
            **sync_dry_run,
            **freshness,
            "verified": bool(entry.get("ok"))
            and payload_ok
            and benchmark_matches
            and entity_matches
            and project_matches
            and run_id_matches
            and parent_run_id_matches
            and parent_entity_matches
            and parent_project_matches
            and schema_valid
            and observed_evidence_valid
            and query_source_valid
            and run_metadata_valid
            and sha256_matches
            and bool(scope_attestation.get("verified"))
            and sync_dry_run["sync_dry_run_report_ok"]
            and freshness["fresh"],
        }
    )
    if not result["verified"]:
        issues = []
        if not bool(entry.get("ok")):
            issues.append("review entry ok is false")
        if not payload_ok:
            issues.append("verifier JSON ok is false")
        if not benchmark_matches:
            issues.append("verifier benchmark does not match review entry")
        if not entity_matches:
            issues.append("verifier entity does not match review entry")
        if not project_matches:
            issues.append("verifier project does not match review entry")
        if not run_id_matches:
            issues.append("verifier run_id does not match review entry")
        if not parent_run_id_matches:
            issues.append("verifier run_id does not match parent review run")
        if not parent_entity_matches:
            issues.append("verifier entity does not match parent review run")
        if not parent_project_matches:
            issues.append("verifier project does not match parent review run")
        if not schema_valid:
            issues.append(
                f"verifier schema version must be {WANDB_COMPLETION_SCHEMA_VERSION}"
            )
        if not observed_evidence_valid:
            issues.append("verifier JSON is missing valid observed_evidence")
        if not query_source_valid:
            issues.append(
                "verifier JSON query_source is invalid: "
                + "; ".join(query_source_errors)
            )
        if not run_metadata_valid:
            issues.append(
                "verifier JSON run_metadata is invalid: "
                + "; ".join(run_metadata_errors)
            )
        if not sha256_matches:
            issues.append("verifier JSON sha256 is missing or does not match")
        if not bool(scope_attestation.get("verified")):
            scope_errors = scope_attestation.get("errors")
            if isinstance(scope_errors, list) and scope_errors:
                issues.extend(str(error) for error in scope_errors)
            else:
                issues.append("scope_attestation is invalid")
        if not sync_dry_run["sync_dry_run_report_ok"]:
            issues.append(
                "W&B completion sync dry-run report is invalid: "
                + "; ".join(sync_dry_run["sync_dry_run_report_errors"])
            )
        if not freshness["fresh"]:
            issues.append(result.get("freshness_error") or "verifier JSON is stale")
        result["verification_error"] = "; ".join(issues) or "verifier JSON does not match the review entry"
    return result


def _verify_wandb_sync_dry_run_report(entry: dict[str, Any]) -> dict[str, Any]:
    adopted = bool(entry.get("adopted_existing_result"))
    path_value = entry.get("sync_dry_run_report_json")
    source_review_path = entry.get("sync_dry_run_source_review_json")
    source_review_sha256 = entry.get("sync_dry_run_source_review_sha256")
    result: dict[str, Any] = {
        "sync_dry_run_report_required": adopted,
        "sync_dry_run_report_json": path_value if isinstance(path_value, str) else "",
        "sync_dry_run_report_present": False,
        "sync_dry_run_report_ok": not adopted,
        "sync_dry_run_report_errors": [],
        "sync_dry_run_source_review_json": source_review_path
        if isinstance(source_review_path, str)
        else "",
        "sync_dry_run_source_review_sha256": source_review_sha256
        if isinstance(source_review_sha256, str)
        else "",
        "sync_dry_run_source_review_sha256_actual": "",
        "sync_dry_run_source_review_sha256_matches": False,
    }
    if not adopted:
        return result

    errors: list[str] = []
    if not isinstance(path_value, str) or not path_value.strip():
        result["sync_dry_run_report_errors"] = [
            "missing W&B completion sync dry-run report path"
        ]
        return result

    result["sync_dry_run_report_present"] = True
    payload, error = read_json(repo_path(path_value))
    if payload is None:
        result["sync_dry_run_report_errors"] = [
            error or "W&B completion sync dry-run report could not be read"
        ]
        return result

    if payload.get("ok") is not True:
        errors.append("sync dry-run report ok must be true")
    if payload.get("status") != "synced":
        errors.append("sync dry-run report status must be synced")
    generated_at = payload.get("generated_at")
    if (
        not isinstance(generated_at, (int, float))
        or not math.isfinite(float(generated_at))
        or generated_at <= 0
    ):
        errors.append("sync dry-run report generated_at must be a positive number")
    if payload.get("dry_run") is not True:
        errors.append("sync dry-run report dry_run must be true")
    if payload.get("in_place") is not False:
        errors.append("sync dry-run report in_place must be false")
    if payload.get("output_path") not in ("", None):
        errors.append("sync dry-run report output_path must be empty")
    if payload.get("verify_wandb_completion") is not True:
        errors.append("sync dry-run report verify_wandb_completion must be true")
    if payload.get("unmatched_count") != 0:
        errors.append("sync dry-run report unmatched_count must be 0")

    before_status = payload.get("before_status")
    after_status = payload.get("after_status")
    if not isinstance(before_status, str) or not before_status.strip():
        errors.append("sync dry-run report before_status is missing")
    if not isinstance(after_status, str) or not after_status.strip():
        errors.append("sync dry-run report after_status is missing")
    if (
        isinstance(before_status, str)
        and before_status.strip()
        and isinstance(after_status, str)
        and after_status.strip()
        and before_status != after_status
    ):
        errors.append("sync dry-run report before_status and after_status must match")

    if not isinstance(source_review_path, str) or not source_review_path.strip():
        errors.append("sync_dry_run_source_review_json is required")
    else:
        if source_path_key(payload.get("review_path")) != source_path_key(source_review_path):
            errors.append("sync dry-run report review_path does not match source review")
        try:
            actual_source_sha = sha256_file(repo_path(source_review_path))
        except OSError as exc:
            errors.append(f"sync_dry_run_source_review_json is not readable: {exc}")
        else:
            result["sync_dry_run_source_review_sha256_actual"] = actual_source_sha
            if source_review_sha256 == actual_source_sha:
                result["sync_dry_run_source_review_sha256_matches"] = True

    if not isinstance(source_review_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}",
        source_review_sha256,
    ):
        errors.append("sync_dry_run_source_review_sha256 must be a 64-character lowercase hex digest")
    elif payload.get("source_review_sha256") != source_review_sha256:
        errors.append("sync dry-run report source_review_sha256 does not match review entry")
    elif (
        result["sync_dry_run_source_review_sha256_actual"]
        and not result["sync_dry_run_source_review_sha256_matches"]
    ):
        errors.append("sync_dry_run_source_review_sha256 does not match source review JSON")

    entries = payload.get("entries")
    if not isinstance(entries, list):
        errors.append("sync dry-run report entries must be a list")
    elif payload.get("entry_count") != len(entries):
        errors.append("sync dry-run report entry_count must match entries length")
    else:
        matching_entries = [
            row
            for row in entries
            if isinstance(row, dict)
            and source_path_key(row.get("path")) == source_path_key(entry.get("path"))
            and row.get("benchmark") == entry.get("benchmark")
            and row.get("entity") == entry.get("entity")
            and row.get("project") == entry.get("project")
            and row.get("run_id") == entry.get("run_id")
            and row.get("sha256") == entry.get("sha256")
        ]
        if not matching_entries:
            errors.append("sync dry-run report entries must include this W&B completion entry")
        else:
            row = matching_entries[0]
            if row.get("ok") is not True:
                errors.append("sync dry-run report entry ok must be true")
            if row.get("observed_evidence_valid") is not True:
                errors.append("sync dry-run report entry observed_evidence_valid must be true")
            if row.get("run_metadata_valid") is not True:
                errors.append("sync dry-run report entry run_metadata_valid must be true")
            if row.get("adopted_existing_result") is not True:
                errors.append("sync dry-run report entry adopted_existing_result must be true")
            row_scope = row.get("scope_attestation")
            entry_scope = (
                entry.get("scope_attestation")
                if isinstance(entry.get("scope_attestation"), dict)
                else {}
            )
            if not isinstance(row_scope, dict):
                errors.append("sync dry-run report entry scope_attestation is missing")
            else:
                row_scope_sha = row_scope.get("source_attestation_sha256")
                entry_scope_sha = entry_scope.get("source_attestation_sha256")
                if not isinstance(row_scope_sha, str) or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    row_scope_sha,
                ):
                    errors.append(
                        "sync dry-run report entry scope_attestation source_attestation_sha256 "
                        "must be a 64-character lowercase hex digest"
                    )
                elif entry_scope_sha != row_scope_sha:
                    errors.append(
                        "sync dry-run report entry scope_attestation source_attestation_sha256 "
                        "does not match review entry"
                    )

    changes = payload.get("changes")
    if not isinstance(changes, list):
        errors.append("sync dry-run report changes must be a list")
    elif payload.get("change_count") != len(changes):
        errors.append("sync dry-run report change_count must match changes length")
    else:
        matching_changes = [
            row
            for row in changes
            if isinstance(row, dict)
            and row.get("target") in {"run", "top_level"}
            and row.get("benchmark") == entry.get("benchmark")
            and row.get("run_id") == entry.get("run_id")
            and row.get("action") in {"added", "replaced", "kept_existing"}
        ]
        if not matching_changes:
            errors.append("sync dry-run report changes must include this W&B completion entry")

    result["sync_dry_run_report_errors"] = errors
    result["sync_dry_run_report_ok"] = not errors
    return result


def _verify_scope_attestation(entry: dict[str, Any]) -> dict[str, Any]:
    adopted = bool(entry.get("adopted_existing_result"))
    result: dict[str, Any] = {
        "required": adopted,
        "verified": not adopted,
        "errors": [],
    }
    if not adopted:
        return result

    attestation = entry.get("scope_attestation")
    if not isinstance(attestation, dict):
        result["errors"].append("missing scope_attestation for adopted existing result")
        return result

    result["schema_version"] = attestation.get("schema_version")
    result["confirmed_by"] = attestation.get("confirmed_by")
    result["confirmed_at"] = attestation.get("confirmed_at")
    result["confirmation"] = attestation.get("confirmation")
    result["attested_benchmark"] = attestation.get("benchmark")
    result["attested_entity"] = attestation.get("entity")
    result["attested_project"] = attestation.get("project")
    result["attested_run_id"] = attestation.get("run_id")
    result["attested_completion_path"] = attestation.get("completion_path")
    result["attested_completion_sha256"] = attestation.get("completion_sha256")
    result["attested_review_path"] = attestation.get("review_path")
    result["actual_cost_estimate"] = attestation.get("actual_cost_estimate")
    result["provider_bill_reference"] = attestation.get("provider_bill_reference")
    result["source_attestation_json"] = attestation.get("source_attestation_json")
    result["source_attestation_sha256"] = attestation.get("source_attestation_sha256")
    result["source_attestation_sha256_actual"] = ""
    result["source_attestation_sha256_matches"] = False
    result["source_audit_json"] = attestation.get("source_audit_json")
    result["source_audit_sha256"] = attestation.get("source_audit_sha256")
    result["source_audit_sha256_actual"] = ""
    result["source_audit_sha256_matches"] = False
    result["source_audit_completion_entry_matches"] = False

    if attestation.get("schema_version") != SCOPE_ATTESTATION_SCHEMA_VERSION:
        result["errors"].append(
            f"scope_attestation schema_version must be {SCOPE_ATTESTATION_SCHEMA_VERSION}"
        )
    if attestation.get("confirmed") is not True:
        result["errors"].append("scope_attestation confirmed must be true")
    for key in (
        "confirmed_by",
        "confirmed_at",
        "confirmation",
        "review_path",
        "completion_path",
        "completion_sha256",
        "actual_cost_estimate",
        "provider_bill_reference",
        "source_attestation_json",
        "source_attestation_sha256",
        "source_audit_json",
        "source_audit_sha256",
    ):
        value = attestation.get(key)
        if not isinstance(value, str) or not value.strip():
            result["errors"].append(f"scope_attestation {key} is required")
    declared_source_sha = attestation.get("source_attestation_sha256")
    if not isinstance(declared_source_sha, str) or not re.fullmatch(
        r"[0-9a-f]{64}",
        declared_source_sha,
    ):
        result["errors"].append(
            "scope_attestation source_attestation_sha256 must be a 64-character lowercase hex digest"
        )
    declared_source_audit_sha = attestation.get("source_audit_sha256")
    if not isinstance(declared_source_audit_sha, str) or not re.fullmatch(
        r"[0-9a-f]{64}",
        declared_source_audit_sha,
    ):
        result["errors"].append(
            "scope_attestation source_audit_sha256 must be a 64-character lowercase hex digest"
        )
    for key in ("actual_cost_estimate", "provider_bill_reference"):
        value = attestation.get(key)
        if isinstance(value, str) and value.strip() and _accounting_value_placeholder(value):
            result["errors"].append(f"scope_attestation {key} must not be a placeholder")
    if not _scope_confirmed_at_valid(attestation.get("confirmed_at")):
        result["errors"].append(
            "scope_attestation confirmed_at must be a timezone-aware ISO 8601 timestamp"
        )
    if not _scope_confirmation_valid(attestation.get("confirmation")):
        result["errors"].append(
            "scope_attestation confirmation must be a concrete non-placeholder sentence "
            f"with at least {MIN_SCOPE_CONFIRMATION_LENGTH} characters"
        )
    if attestation.get("benchmark") != entry.get("benchmark"):
        result["errors"].append("scope_attestation benchmark does not match review entry")
    if attestation.get("entity") != entry.get("entity"):
        result["errors"].append("scope_attestation entity does not match review entry")
    if attestation.get("project") != entry.get("project"):
        result["errors"].append("scope_attestation project does not match review entry")
    if attestation.get("run_id") != entry.get("run_id"):
        result["errors"].append("scope_attestation run_id does not match review entry")
    if source_path_key(attestation.get("completion_path")) != source_path_key(entry.get("path")):
        result["errors"].append("scope_attestation completion_path does not match review entry")
    if attestation.get("completion_sha256") != entry.get("sha256"):
        result["errors"].append("scope_attestation completion_sha256 does not match review entry")
    source_review_path = entry.get("sync_dry_run_source_review_json")
    review_path = (
        source_review_path
        if isinstance(source_review_path, str) and source_review_path.strip()
        else entry.get("review_path")
    )
    if review_path and source_path_key(attestation.get("review_path")) != source_path_key(review_path):
        result["errors"].append("scope_attestation review_path does not match review entry")

    source_attestation_json = attestation.get("source_attestation_json")
    if isinstance(source_attestation_json, str) and source_attestation_json.strip():
        source_path = repo_path(source_attestation_json)
        try:
            actual_source_sha = sha256_file(source_path)
        except OSError as exc:
            result["errors"].append(
                f"scope_attestation source_attestation_json is not readable: {exc}"
            )
            actual_source_sha = ""
        if actual_source_sha:
            result["source_attestation_sha256_actual"] = actual_source_sha
            if declared_source_sha == actual_source_sha:
                result["source_attestation_sha256_matches"] = True
            else:
                result["errors"].append(
                    "scope_attestation source_attestation_sha256 does not match source_attestation_json"
                )
        source_payload, source_error = read_json(source_path)
        if source_payload is None:
            result["errors"].append(
                f"scope_attestation source_attestation_json is not readable: {source_error}"
            )
        else:
            if source_payload.get("schema_version") != SCOPE_ATTESTATION_SCHEMA_VERSION:
                result["errors"].append(
                    "source_attestation_json schema_version does not match"
                )
            if source_payload.get("confirmed") is not True:
                result["errors"].append("source_attestation_json confirmed must be true")
            for key in (
                "confirmed_by",
                "confirmed_at",
                "confirmation",
                "actual_cost_estimate",
                "provider_bill_reference",
                "benchmark",
                "entity",
                "project",
                "run_id",
                "completion_sha256",
                "source_audit_sha256",
            ):
                if source_payload.get(key) != attestation.get(key):
                    result["errors"].append(
                        f"source_attestation_json {key} does not match scope_attestation"
                    )
            for key in ("actual_cost_estimate", "provider_bill_reference"):
                value = source_payload.get(key)
                if isinstance(value, str) and value.strip() and _accounting_value_placeholder(value):
                    result["errors"].append(
                        f"source_attestation_json {key} must not be a placeholder"
                    )
            if not _scope_confirmed_at_valid(source_payload.get("confirmed_at")):
                result["errors"].append(
                    "source_attestation_json confirmed_at must be a timezone-aware ISO 8601 timestamp"
                )
            if not _scope_confirmation_valid(source_payload.get("confirmation")):
                result["errors"].append(
                    "source_attestation_json confirmation must be a concrete non-placeholder sentence "
                    f"with at least {MIN_SCOPE_CONFIRMATION_LENGTH} characters"
                )
            if source_path_key(source_payload.get("completion_path")) != source_path_key(entry.get("path")):
                result["errors"].append(
                    "source_attestation_json completion_path does not match review entry"
                )
            if source_payload.get("completion_sha256") != entry.get("sha256"):
                result["errors"].append(
                    "source_attestation_json completion_sha256 does not match review entry"
                )
            if review_path and source_path_key(source_payload.get("review_path")) != source_path_key(review_path):
                result["errors"].append(
                    "source_attestation_json review_path does not match review entry"
                )
            if source_path_key(source_payload.get("source_audit_json")) != source_path_key(
                attestation.get("source_audit_json")
            ):
                result["errors"].append(
                    "source_attestation_json source_audit_json does not match scope_attestation"
                )

    source_audit_json = attestation.get("source_audit_json")
    if isinstance(source_audit_json, str) and source_audit_json.strip():
        audit_path = repo_path(source_audit_json)
        try:
            actual_audit_sha = sha256_file(audit_path)
        except OSError as exc:
            result["errors"].append(
                f"scope_attestation source_audit_json is not readable: {exc}"
            )
            actual_audit_sha = ""
        if actual_audit_sha:
            result["source_audit_sha256_actual"] = actual_audit_sha
            if declared_source_audit_sha == actual_audit_sha:
                result["source_audit_sha256_matches"] = True
            else:
                result["errors"].append(
                    "scope_attestation source_audit_sha256 does not match source_audit_json"
                )
        audit_payload, audit_error = read_json(audit_path)
        if audit_payload is None:
            result["errors"].append(
                f"scope_attestation source_audit_json is not readable: {audit_error}"
            )
        else:
            formalized_records = audit_payload.get("formalized_records")
            if not isinstance(formalized_records, list):
                result["errors"].append(
                    "scope_attestation source_audit_json formalized_records must be a list"
                )
            elif any(
                isinstance(record, dict)
                and source_audit_record_matches_wandb_entry(record, entry)
                for record in formalized_records
            ):
                result["source_audit_completion_entry_matches"] = True
            else:
                result["errors"].append(
                    "scope_attestation source_audit_json formalized_records does not include completion entry"
                )

    result["verified"] = not result["errors"]
    return result


def _checks_all_ok(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    return all(isinstance(row, dict) and row.get("ok") is True for row in value)


def _missing_required_check_names(value: Any) -> list[str]:
    if not isinstance(value, list):
        return sorted(REQUIRED_WEAVE_AGENTS_CHECK_NAMES)
    names = {
        row.get("name")
        for row in value
        if isinstance(row, dict) and isinstance(row.get("name"), str)
    }
    return sorted(REQUIRED_WEAVE_AGENTS_CHECK_NAMES - names)


def _check_names(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {
        row.get("name")
        for row in value
        if isinstance(row, dict) and isinstance(row.get("name"), str)
    }


def _non_empty_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
    return result


def _check_by_name(value: Any, name: str) -> dict[str, Any] | None:
    if not isinstance(value, list):
        return None
    for row in value:
        if isinstance(row, dict) and row.get("name") == name:
            return row
    return None


def _weave_agents_request_model_evidence(
    payload: dict[str, Any],
    *,
    required_evidence: dict[str, Any],
    content_capture_health: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    expected = _non_empty_string_list(required_evidence.get("expected_request_models"))
    if not expected:
        errors.append(
            "required_evidence.expected_request_models must be a non-empty list of strings"
        )

    check = _check_by_name(payload.get("checks"), "request_model")
    check_expected: list[str] = []
    check_observed: list[str] = []
    if check is None:
        errors.append("checks missing required check: request_model")
    else:
        if check.get("ok") is not True:
            errors.append("request_model check must be ok=true")
        check_expected = _non_empty_string_list(check.get("expected_request_models"))
        check_observed = _non_empty_string_list(check.get("observed_request_models"))
        if not check_expected:
            errors.append(
                "checks.request_model.expected_request_models must be a non-empty list of strings"
            )
        elif expected and set(check_expected) != set(expected):
            errors.append(
                "checks.request_model.expected_request_models must match required_evidence.expected_request_models"
            )
        if not check_observed:
            errors.append(
                "checks.request_model.observed_request_models must be a non-empty list of strings"
            )
        elif expected and set(expected).isdisjoint(check_observed):
            errors.append(
                "checks.request_model.observed_request_models must include an expected model alias"
            )

    spans = payload.get("latest_trace_spans_chronological")
    span_rows = spans if isinstance(spans, list) else []
    span_models = sorted(
        {
            str(span.get("request_model")).strip()
            for span in span_rows
            if isinstance(span, dict)
            and isinstance(span.get("request_model"), str)
            and span.get("request_model").strip()
        }
    )
    if not span_models:
        errors.append("latest_trace_spans_chronological must expose request_model")
    elif expected and set(expected).isdisjoint(span_models):
        errors.append(
            "latest_trace_spans_chronological request_model values must include an expected model alias"
        )
    if check_observed and span_models and set(check_observed) != set(span_models):
        errors.append(
            "checks.request_model.observed_request_models must match latest_trace_spans_chronological request_model values"
        )

    request_model_count = content_capture_health.get("request_model_count")
    if not isinstance(request_model_count, int) or request_model_count <= 0:
        errors.append("content_capture_health.request_model_count must be a positive integer")
    elif span_models and request_model_count != len(span_models):
        errors.append(
            "content_capture_health.request_model_count must match unique request_model values"
        )

    return {
        "request_model_proven": not errors,
        "request_model_errors": errors,
        "expected_request_models": expected,
        "observed_request_models": check_observed,
        "span_request_models": span_models,
        "request_model_count": request_model_count,
    }


def _required_texts(value: Any) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    if value in (None, ""):
        return [], errors
    if not isinstance(value, list):
        return [], ["required_evidence.required_texts is not a list"]
    result: list[str] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, str):
            errors.append(f"required_evidence.required_texts[{index}] is not a string")
            continue
        if item:
            result.append(item)
    return result, errors


def _weave_agents_query_source_errors(
    payload: dict[str, Any],
    *,
    required_evidence: dict[str, Any],
    spans: Any,
) -> list[str]:
    errors: list[str] = []
    query_source = payload.get("query_source")
    if not isinstance(query_source, dict):
        return ["query_source is not an object"]

    expected_query_fields = {
        "kind": WEAVE_AGENTS_QUERY_SOURCE_KIND,
        "api_base_url": WEAVE_AGENTS_API_BASE_URL,
        "agents_endpoint": WEAVE_AGENTS_QUERY_ENDPOINT,
        "spans_endpoint": WEAVE_AGENTS_SPANS_QUERY_ENDPOINT,
        "project_id": payload.get("project_id"),
        "agent_name": payload.get("agent_name"),
    }
    for field, expected in expected_query_fields.items():
        if query_source.get(field) != expected:
            errors.append(
                f"query_source.{field} mismatch: expected {expected}, got {query_source.get(field)}"
            )

    for field in WEAVE_AGENTS_QUERY_COUNT_FIELDS:
        value = query_source.get(field)
        if not isinstance(value, int) or value < 0:
            errors.append(f"query_source.{field} is not a non-negative integer")

    for field in ("conversation_id", "conversation_id_contains"):
        expected = (
            required_evidence.get(field)
            if isinstance(required_evidence.get(field), str)
            else ""
        )
        if query_source.get(field) != expected:
            errors.append(f"query_source.{field} does not match required_evidence")

    if not isinstance(spans, list) or not spans:
        errors.append(
            "latest_trace_spans_chronological is required for query_source verification"
        )
    else:
        valid_span_count = sum(1 for span in spans if isinstance(span, dict))
        latest_count = query_source.get("latest_trace_span_count")
        if isinstance(latest_count, int) and latest_count != valid_span_count:
            errors.append(
                "query_source.latest_trace_span_count does not match "
                "latest_trace_spans_chronological"
            )
    return errors


def _weave_agents_run_scope(
    required_evidence: dict[str, Any],
    *,
    expected_run_id: Any,
) -> tuple[bool, dict[str, Any], str]:
    run_id = expected_run_id if isinstance(expected_run_id, str) else ""
    conversation_id = required_evidence.get("conversation_id")
    if not isinstance(conversation_id, str):
        conversation_id = ""
    conversation_id_contains = required_evidence.get("conversation_id_contains")
    if not isinstance(conversation_id_contains, str):
        conversation_id_contains = ""
    scope = {
        "run_id": run_id,
        "conversation_id": conversation_id,
        "conversation_id_contains": conversation_id_contains,
    }
    if not run_id:
        return False, scope, "review entry is missing run_id"
    if run_id in conversation_id or run_id in conversation_id_contains:
        return True, scope, ""
    return (
        False,
        scope,
        "Weave Agents verifier JSON does not prove W&B run scope: "
        "required_evidence.conversation_id or conversation_id_contains must include "
        f"{run_id}",
    )


def _verify_weave_agents_sync_dry_run_report(entry: dict[str, Any]) -> dict[str, Any]:
    path_value = entry.get("sync_dry_run_report_json")
    source_review_path = entry.get("sync_dry_run_source_review_json")
    source_review_sha256 = entry.get("sync_dry_run_source_review_sha256")
    result = {
        "sync_dry_run_report_json": path_value if isinstance(path_value, str) else "",
        "sync_dry_run_report_present": False,
        "sync_dry_run_report_ok": False,
        "sync_dry_run_report_errors": [],
        "sync_dry_run_source_review_json": source_review_path
        if isinstance(source_review_path, str)
        else "",
        "sync_dry_run_source_review_sha256": source_review_sha256
        if isinstance(source_review_sha256, str)
        else "",
        "sync_dry_run_source_review_sha256_actual": "",
        "sync_dry_run_source_review_sha256_matches": False,
    }
    errors: list[str] = []
    if not isinstance(path_value, str) or not path_value.strip():
        result["sync_dry_run_report_errors"] = [
            "missing Weave Agents sync dry-run report path"
        ]
        return result

    result["sync_dry_run_report_present"] = True
    payload, error = read_json(repo_path(path_value))
    if payload is None:
        result["sync_dry_run_report_errors"] = [
            error or "Weave Agents sync dry-run report could not be read"
        ]
        return result

    if payload.get("ok") is not True:
        errors.append("sync dry-run report ok must be true")
    if payload.get("status") != "synced":
        errors.append("sync dry-run report status must be synced")
    generated_at = payload.get("generated_at")
    if (
        not isinstance(generated_at, (int, float))
        or not math.isfinite(float(generated_at))
        or generated_at <= 0
    ):
        errors.append("sync dry-run report generated_at must be a positive number")
    if payload.get("dry_run") is not True:
        errors.append("sync dry-run report dry_run must be true")
    if payload.get("in_place") is not False:
        errors.append("sync dry-run report in_place must be false")
    if payload.get("output_path") not in ("", None):
        errors.append("sync dry-run report output_path must be empty")
    if payload.get("verify_weave_agents") is not True:
        errors.append("sync dry-run report verify_weave_agents must be true")
    if payload.get("unmatched_count") != 0:
        errors.append("sync dry-run report unmatched_count must be 0")

    before_status = payload.get("before_status")
    after_status = payload.get("after_status")
    if not isinstance(before_status, str) or not before_status.strip():
        errors.append("sync dry-run report before_status is missing")
    if not isinstance(after_status, str) or not after_status.strip():
        errors.append("sync dry-run report after_status is missing")
    if (
        isinstance(before_status, str)
        and before_status.strip()
        and isinstance(after_status, str)
        and after_status.strip()
        and before_status != after_status
    ):
        errors.append("sync dry-run report before_status and after_status must match")

    review_path = (
        source_review_path
        if isinstance(source_review_path, str) and source_review_path.strip()
        else entry.get("review_path")
    )
    if isinstance(review_path, str) and review_path:
        if source_path_key(payload.get("review_path")) != source_path_key(review_path):
            errors.append("sync dry-run report review_path does not match review entry")
    if not isinstance(source_review_path, str) or not source_review_path.strip():
        errors.append("sync_dry_run_source_review_json is required")
    else:
        try:
            actual_source_sha = sha256_file(repo_path(source_review_path))
        except OSError as exc:
            errors.append(f"sync_dry_run_source_review_json is not readable: {exc}")
        else:
            result["sync_dry_run_source_review_sha256_actual"] = actual_source_sha
            if source_review_sha256 == actual_source_sha:
                result["sync_dry_run_source_review_sha256_matches"] = True

    if not isinstance(source_review_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}",
        source_review_sha256,
    ):
        errors.append("sync_dry_run_source_review_sha256 must be a 64-character lowercase hex digest")
    elif payload.get("source_review_sha256") != source_review_sha256:
        errors.append("sync dry-run report source_review_sha256 does not match review entry")
    elif (
        result["sync_dry_run_source_review_sha256_actual"]
        and not result["sync_dry_run_source_review_sha256_matches"]
    ):
        errors.append("sync_dry_run_source_review_sha256 does not match source review JSON")

    entries = payload.get("entries")
    if not isinstance(entries, list):
        errors.append("sync dry-run report entries must be a list")
    elif payload.get("entry_count") != len(entries):
        errors.append("sync dry-run report entry_count must match entries length")
    else:
        matching_entries = [
            row
            for row in entries
            if isinstance(row, dict)
            and source_path_key(row.get("path")) == source_path_key(entry.get("path"))
            and row.get("run_id") == entry.get("run_id")
            and row.get("agent_name") == entry.get("agent_name")
        ]
        if not matching_entries:
            errors.append("sync dry-run report entries must include this Weave entry")
        else:
            row = matching_entries[0]
            if row.get("ok") is not True:
                errors.append("sync dry-run report entry ok must be true")
            if row.get("checks_valid") is not True:
                errors.append("sync dry-run report entry checks_valid must be true")
            if row.get("trace_present") is not True:
                errors.append("sync dry-run report entry trace_present must be true")
            if row.get("run_scope_proven") is not True:
                errors.append("sync dry-run report entry run_scope_proven must be true")
            if row.get("request_model_proven") is not True:
                errors.append("sync dry-run report entry request_model_proven must be true")
            latest_trace_id = entry.get("latest_trace_id")
            if isinstance(latest_trace_id, str) and latest_trace_id:
                if row.get("latest_trace_id") != latest_trace_id:
                    errors.append("sync dry-run report entry latest_trace_id does not match")
            expected_query_fields = {
                "query_source_kind": WEAVE_AGENTS_QUERY_SOURCE_KIND,
                "query_source_api_base_url": WEAVE_AGENTS_API_BASE_URL,
                "query_source_agents_endpoint": WEAVE_AGENTS_QUERY_ENDPOINT,
                "query_source_spans_endpoint": WEAVE_AGENTS_SPANS_QUERY_ENDPOINT,
            }
            for field, expected in expected_query_fields.items():
                if row.get(field) != expected:
                    errors.append(f"sync dry-run report entry {field} mismatch")

    changes = payload.get("changes")
    if not isinstance(changes, list):
        errors.append("sync dry-run report changes must be a list")
    elif payload.get("change_count") != len(changes):
        errors.append("sync dry-run report change_count must match changes length")
    else:
        matching_changes = [
            row
            for row in changes
            if isinstance(row, dict)
            and row.get("target") in {"run", "top_level"}
            and row.get("run_id") == entry.get("run_id")
            and row.get("agent_name") == entry.get("agent_name")
            and row.get("action") in {"added", "replaced", "kept_existing"}
        ]
        if not matching_changes:
            errors.append("sync dry-run report changes must include this Weave entry")

    result["sync_dry_run_report_errors"] = errors
    result["sync_dry_run_report_ok"] = not errors
    return result


def _verify_review_weave_agents_completion_entry(
    entry: dict[str, Any],
    *,
    max_age_seconds: int | None,
) -> dict[str, Any]:
    result = dict(entry)
    result["entry_ok"] = bool(entry.get("ok"))
    result["verifier_json_ok"] = False
    result["verified"] = False
    path_value = entry.get("path")
    if not isinstance(path_value, str) or not path_value:
        result.update(_verify_weave_agents_sync_dry_run_report(entry))
        result["verification_error"] = "missing Weave Agents verifier JSON path"
        return result

    payload, error = read_json(repo_path(path_value))
    if payload is None:
        result.update(_verify_weave_agents_sync_dry_run_report(entry))
        result["verification_error"] = error or "Weave Agents verifier JSON could not be read"
        return result

    payload_ok = payload.get("ok") is True
    entry_agent_name = entry.get("agent_name")
    payload_agent_name = payload.get("agent_name")
    agent_name_matches = (
        not entry_agent_name
        or payload_agent_name == entry_agent_name
    )
    schema_version = payload.get("verification_schema_version")
    schema_valid = schema_version == WEAVE_AGENTS_COMPLETION_SCHEMA_VERSION
    missing_required_checks = _missing_required_check_names(payload.get("checks"))
    check_names = _check_names(payload.get("checks"))
    required_evidence = (
        payload.get("required_evidence")
        if isinstance(payload.get("required_evidence"), dict)
        else {}
    )
    required_texts, required_text_errors = _required_texts(
        required_evidence.get("required_texts")
    )
    latest_trace_spans_chronological = payload.get("latest_trace_spans_chronological")
    query_source_errors = _weave_agents_query_source_errors(
        payload,
        required_evidence=required_evidence,
        spans=latest_trace_spans_chronological,
    )
    query_source_valid = not query_source_errors
    query_source = (
        payload.get("query_source")
        if isinstance(payload.get("query_source"), dict)
        else {}
    )
    trace_final_answer_order_required = (
        required_evidence.get("trace_final_answer_order_required") is True
    )
    trace_timestamp_quality_required = (
        required_evidence.get("trace_timestamp_quality_required") is True
    )
    input_message_required = (
        required_evidence.get("input_message_required") is True
    )
    run_scope_proven, run_scope, run_scope_error = _weave_agents_run_scope(
        required_evidence,
        expected_run_id=entry.get("run_id"),
    )
    content_capture_health = (
        payload.get("content_capture_health")
        if isinstance(payload.get("content_capture_health"), dict)
        else {}
    )
    request_model_evidence = _weave_agents_request_model_evidence(
        payload,
        required_evidence=required_evidence,
        content_capture_health=content_capture_health,
    )
    input_message_visible = (
        isinstance(content_capture_health.get("message_spans_with_input"), int)
        and content_capture_health.get("message_spans_with_input") > 0
    )
    timestamps_valid = (
        isinstance(content_capture_health.get("spans_with_invalid_timestamps"), int)
        and content_capture_health.get("spans_with_invalid_timestamps") == 0
        and isinstance(content_capture_health.get("spans_with_valid_timestamps"), int)
        and content_capture_health.get("spans_with_valid_timestamps") > 0
    )
    required_text_capture_present = (
        not required_texts or "required_text_capture" in check_names
    )
    required_text_count_ok = (
        not required_texts
        or (
            isinstance(content_capture_health.get("required_text_count"), int)
            and content_capture_health.get("required_text_count") >= len(required_texts)
        )
    )
    checks_valid = (
        _checks_all_ok(payload.get("checks"))
        and not missing_required_checks
        and not required_text_errors
        and required_text_capture_present
        and required_text_count_ok
    )
    latest_trace_id = payload.get("latest_trace_id")
    trace_present = isinstance(latest_trace_id, str) and bool(latest_trace_id)
    sync_dry_run = _verify_weave_agents_sync_dry_run_report(
        {**entry, "latest_trace_id": latest_trace_id}
    )
    freshness = freshness_record(
        payload=payload,
        path=repo_path(path_value),
        max_age_seconds=max_age_seconds,
        require_generated_at=True,
    )
    result.update(
        {
            "verifier_json_ok": payload_ok,
            "verifier_json_agent_name": payload_agent_name,
            "agent_name_matches": agent_name_matches,
            "verification_schema_version": schema_version,
            "schema_valid": schema_valid,
            "checks_valid": checks_valid,
            "missing_required_checks": missing_required_checks,
            "query_source": query_source,
            "query_source_valid": query_source_valid,
            "query_source_errors": query_source_errors,
            "required_texts": required_texts,
            "required_text_errors": required_text_errors,
            "required_text_capture_present": required_text_capture_present,
            "required_text_count_ok": required_text_count_ok,
            **request_model_evidence,
            "input_message_required": input_message_required,
            "input_message_visible": input_message_visible,
            "trace_timestamp_quality_required": trace_timestamp_quality_required,
            "timestamps_valid": timestamps_valid,
            "trace_final_answer_order_required": trace_final_answer_order_required,
            "run_scope_proven": run_scope_proven,
            "run_scope": run_scope,
            "latest_trace_id": latest_trace_id,
            "trace_present": trace_present,
            "content_capture_health": content_capture_health,
            **sync_dry_run,
            **freshness,
            "verified": bool(entry.get("ok"))
            and payload_ok
            and agent_name_matches
            and schema_valid
            and checks_valid
            and request_model_evidence["request_model_proven"]
            and query_source_valid
            and input_message_required
            and input_message_visible
            and trace_timestamp_quality_required
            and timestamps_valid
            and trace_final_answer_order_required
            and run_scope_proven
            and trace_present
            and sync_dry_run["sync_dry_run_report_ok"]
            and freshness["fresh"],
        }
    )
    if not result["verified"]:
        issues = []
        if not bool(entry.get("ok")):
            issues.append("review entry ok is false")
        if not payload_ok:
            issues.append("Weave Agents verifier JSON ok is false")
        if not agent_name_matches:
            issues.append("verifier agent_name does not match review entry")
        if not schema_valid:
            issues.append(
                f"Weave Agents verifier schema version must be {WEAVE_AGENTS_COMPLETION_SCHEMA_VERSION}"
            )
        if not checks_valid:
            if missing_required_checks:
                issues.append(
                    "Weave Agents verifier JSON is missing required checks: "
                    + ", ".join(missing_required_checks)
                )
            elif required_text_errors:
                issues.append(
                    "Weave Agents verifier JSON has invalid required_texts: "
                    + "; ".join(required_text_errors)
                )
            elif not required_text_capture_present:
                issues.append(
                    "Weave Agents verifier JSON is missing required_text_capture"
                )
            elif not required_text_count_ok:
                issues.append(
                    "Weave Agents verifier JSON does not report all required text checks"
                )
            else:
                issues.append("Weave Agents verifier JSON has missing or failed checks")
        if not request_model_evidence["request_model_proven"]:
            issues.append(
                "Weave Agents verifier JSON does not prove request_model: "
                + "; ".join(request_model_evidence["request_model_errors"])
            )
        if not query_source_valid:
            issues.append(
                "Weave Agents verifier JSON query_source is invalid: "
                + "; ".join(query_source_errors)
            )
        if not input_message_required:
            issues.append("Weave Agents verifier JSON does not require input_message")
        if not input_message_visible:
            issues.append("Weave Agents verifier JSON does not expose user/problem input")
        if not trace_timestamp_quality_required:
            issues.append(
                "Weave Agents verifier JSON does not require trace_timestamp_quality"
            )
        if not timestamps_valid:
            issues.append("Weave Agents verifier JSON does not prove valid span timestamps")
        if not trace_final_answer_order_required:
            issues.append(
                "Weave Agents verifier JSON does not require trace_final_answer_order"
            )
        if not run_scope_proven:
            issues.append(run_scope_error)
        if not trace_present:
            issues.append("Weave Agents verifier JSON is missing latest_trace_id")
        if not sync_dry_run["sync_dry_run_report_ok"]:
            issues.append(
                "Weave Agents sync dry-run report is invalid: "
                + "; ".join(sync_dry_run["sync_dry_run_report_errors"])
            )
        if not freshness["fresh"]:
            issues.append(result.get("freshness_error") or "Weave Agents verifier JSON is stale")
        result["verification_error"] = "; ".join(issues) or "Weave Agents verifier JSON does not match the review entry"
    return result


def _observed_evidence_valid(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("run_state") != "finished":
        return False
    evidence_keys = (
        "summary_metrics",
        "tables",
        "artifacts",
        "taxonomy_tables",
        "aggregate_tables",
    )
    return any(bool(value.get(key)) for key in evidence_keys)


def _review_configs(payload: dict[str, Any]) -> list[str]:
    configs = payload.get("configs")
    if not isinstance(configs, list):
        return []
    return [str(config) for config in configs if isinstance(config, str) and config.strip()]


def _path_has_nemoclaw_agentic_marker(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return "generated_openai_canary_agentic_nemoclaw/" in normalized


def _configs_include_nemoclaw_agentic_config(configs: list[str]) -> bool:
    return any(_path_has_nemoclaw_agentic_marker(config) for config in configs)


def _review_requires_nemoclaw_agentic_config(payload: dict[str, Any]) -> bool:
    return (
        str(payload.get("phase") or "") == "agentic"
        and bool(payload.get("canary"))
        and bool(_review_configs(payload))
    )


def evaluate_one_model_canary(
    paths: list[Path],
    *,
    require: bool,
    required_run_ids: dict[str, str] | None = None,
    required_benchmarks: list[str] | None = None,
    wandb_completion_max_age_seconds: int | None = DEFAULT_WANDB_COMPLETION_MAX_AGE_SECONDS,
    weave_agents_completion_max_age_seconds: int | None = DEFAULT_WANDB_COMPLETION_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    requirement = (
        "One canary model must complete either full phase or all phased canary stages, "
        "and completed review records must include passing W&B completion checks."
    )
    required_run_ids = required_run_ids or {}
    required_benchmarks = required_benchmarks or []
    required_run_id_values = sorted(set(required_run_ids.values()))
    records: list[dict[str, Any]] = []
    phase_status: dict[str, str] = {}
    completed_run_ids: set[str] = set()
    completed_wandb_benchmarks: dict[str, list[dict[str, Any]]] = {}
    completed_weave_agents_phases: dict[str, list[dict[str, Any]]] = {}
    full_completed = False
    for path in paths:
        payload, error = read_json(path)
        if payload is None:
            records.append({"path": path_display(path), "ok": False, "error": error})
            continue
        phase = str(payload.get("phase") or "")
        status = str(payload.get("status") or "")
        canary = bool(payload.get("canary"))
        run_ids = _review_run_ids(payload)
        wandb_completion_entries = _review_wandb_completion_entries(
            payload,
            review_path=path_display(path),
        )
        verified_wandb_completion_entries = [
            _verify_review_wandb_completion_entry(
                entry,
                max_age_seconds=wandb_completion_max_age_seconds,
            )
            for entry in wandb_completion_entries
        ]
        verified_weave_agents_completion_entries = [
            _verify_review_weave_agents_completion_entry(
                entry,
                max_age_seconds=weave_agents_completion_max_age_seconds,
            )
            for entry in _review_weave_agents_completion_entries(
                payload,
                review_path=path_display(path),
            )
        ]
        configs = _review_configs(payload)
        requires_nemoclaw_agentic_config = _review_requires_nemoclaw_agentic_config(payload)
        nemoclaw_agentic_config_bound = _configs_include_nemoclaw_agentic_config(configs)
        records.append(
            {
                "path": path_display(path),
                "phase": phase,
                "status": status,
                "canary": canary,
                "model_count": payload.get("model_count"),
                "configs": configs,
                "nemoclaw_agentic_config_required": requires_nemoclaw_agentic_config,
                "nemoclaw_agentic_config_bound": nemoclaw_agentic_config_bound,
                "requires_paid_model_api": bool(payload.get("requires_paid_model_api")),
                "will_execute_external_actions": bool(payload.get("will_execute_external_actions")),
                "run_purpose_present": _nonempty_string(payload.get("run_purpose")),
                "expected_cost_band_present": _nonempty_string(payload.get("expected_cost_band")),
                "pre_run_budget_estimate": _review_pre_run_budget_estimate(payload),
                "external_action_approval": _review_external_action_approval(payload),
                "actual_cost_estimate_present": _nonempty_string(payload.get("actual_cost_estimate")),
                "provider_bill_reference_present": _nonempty_string(payload.get("provider_bill_reference")),
                "actual_cost_estimate_placeholder": isinstance(payload.get("actual_cost_estimate"), str)
                and _accounting_value_placeholder(payload.get("actual_cost_estimate")),
                "provider_bill_reference_placeholder": isinstance(payload.get("provider_bill_reference"), str)
                and _accounting_value_placeholder(payload.get("provider_bill_reference")),
                "run_count": len(payload.get("runs")) if isinstance(payload.get("runs"), list) else 0,
                "verify_wandb_completion": bool(payload.get("verify_wandb_completion")),
                "wandb_run_ids": run_ids,
                "wandb_completion_entries": verified_wandb_completion_entries,
                "wandb_completion_max_age_seconds": wandb_completion_max_age_seconds,
                "verify_weave_agents": bool(payload.get("verify_weave_agents")),
                "weave_agents_completion_entries": verified_weave_agents_completion_entries,
                "weave_agents_completion_max_age_seconds": weave_agents_completion_max_age_seconds,
            }
        )
        if not canary:
            continue
        phase_status[phase] = status
        if status == "completed":
            completed_run_ids.update(run_ids)
            for entry in verified_wandb_completion_entries:
                benchmark = entry["benchmark"]
                expected_run_id = required_run_ids.get(benchmark)
                if not entry["verified"]:
                    continue
                if expected_run_id and entry["run_id"] != expected_run_id:
                    continue
                completed_wandb_benchmarks.setdefault(benchmark, []).append(entry)
            if phase in WEAVE_AGENTS_CANARY_COMPLETION_PHASES:
                for entry in verified_weave_agents_completion_entries:
                    if not entry.get("verified"):
                        continue
                    completed_weave_agents_phases.setdefault(phase, []).append(entry)
        if phase == "full" and status == "completed":
            full_completed = True
    phased_completed = all(
        phase_status.get(phase) == "completed"
        for phase in ("nonagentic", "agentic", "agentic_aggregate")
    )
    structural_ok = full_completed or phased_completed
    missing_required_run_ids = [
        run_id for run_id in required_run_id_values if run_id not in completed_run_ids
    ]
    run_id_ok = not required_run_id_values or not missing_required_run_ids
    missing_required_wandb_benchmarks = [
        benchmark
        for benchmark in required_benchmarks
        if benchmark not in completed_wandb_benchmarks
    ]
    wandb_completion_ok = not required_benchmarks or not missing_required_wandb_benchmarks
    weave_agents_completion_required = bool(
        set(required_benchmarks) & AGENTIC_REQUIRED_WANDB_BENCHMARKS
    )
    weave_agents_completion_ok = (
        not weave_agents_completion_required
        or (
            full_completed
            and bool(completed_weave_agents_phases.get("full"))
        )
        or (
            phased_completed
            and bool(completed_weave_agents_phases.get("agentic"))
        )
    )
    missing_required_weave_agents_phases: list[str] = []
    if weave_agents_completion_required and structural_ok and not weave_agents_completion_ok:
        if full_completed:
            missing_required_weave_agents_phases.append("full")
        if phased_completed:
            missing_required_weave_agents_phases.append("agentic")
        if not missing_required_weave_agents_phases:
            missing_required_weave_agents_phases.append("full_or_agentic")
    bad_nemoclaw_agentic_config_records = [
        record
        for record in records
        if record.get("status") == "completed"
        and record.get("nemoclaw_agentic_config_required")
        and not record.get("nemoclaw_agentic_config_bound")
    ]
    nemoclaw_agentic_config_ok = not bad_nemoclaw_agentic_config_records
    ok = (
        structural_ok
        and run_id_ok
        and wandb_completion_ok
        and weave_agents_completion_ok
        and nemoclaw_agentic_config_ok
    )
    return gate_record(
        name="one_model_full_canary",
        ok=ok or not require,
        blocking=require,
        status=(
            "passed"
            if ok
            else "wandb_completion_not_proven"
            if structural_ok and run_id_ok and not wandb_completion_ok
            else "weave_agents_completion_not_proven"
            if structural_ok and run_id_ok and wandb_completion_ok and not weave_agents_completion_ok
            else "nemoclaw_agentic_config_not_proven"
            if structural_ok and run_id_ok and wandb_completion_ok and weave_agents_completion_ok and not nemoclaw_agentic_config_ok
            else "run_id_not_proven"
            if structural_ok and not run_id_ok
            else "incomplete"
        ),
        requirement=requirement,
        evidence_paths=paths,
        detail=(
            "One-model canary is complete."
            if ok
            else "One-model canary review records do not prove passing W&B completion checks."
            if structural_ok and run_id_ok and not wandb_completion_ok
            else "One-model canary review records do not prove passing Weave Agents completion checks."
            if structural_ok and run_id_ok and wandb_completion_ok and not weave_agents_completion_ok
            else "Completed agentic canary review records do not prove NeMoClaw agentic config binding."
            if structural_ok and run_id_ok and wandb_completion_ok and weave_agents_completion_ok and not nemoclaw_agentic_config_ok
            else "One-model canary review records do not prove the required W&B run id."
            if structural_ok and not run_id_ok
            else "No full canary or complete phased canary review record exists."
        ),
        next_action=(
            "Review score/cost/trace quality before multi-model expansion."
            if ok
            else "Run the one-model canary with --verify-wandb-completion enabled and include the completed paid-run review JSON."
            if structural_ok and run_id_ok and not wandb_completion_ok
            else "Run the one-model agentic canary with --verify-weave-agents enabled and include the verified Weave Agents completion JSON."
            if structural_ok and run_id_ok and wandb_completion_ok and not weave_agents_completion_ok
            else "Run the one-model canary with the required W&B run id and completion verification enabled."
            if structural_ok and not run_id_ok
            else "Run and complete the agreed one-model canary phases before multi-model execution."
        ),
        extra={
            "phase_status": phase_status,
            "records": records,
            "required_wandb_benchmarks": required_benchmarks,
            "required_wandb_run_ids": required_run_ids,
            "wandb_completion_max_age_seconds": wandb_completion_max_age_seconds,
            "weave_agents_completion_max_age_seconds": weave_agents_completion_max_age_seconds,
            "completed_wandb_run_ids": sorted(completed_run_ids),
            "missing_required_wandb_run_ids": missing_required_run_ids,
            "completed_wandb_completion_benchmarks": completed_wandb_benchmarks,
            "missing_required_wandb_completion_benchmarks": missing_required_wandb_benchmarks,
            "weave_agents_completion_required": weave_agents_completion_required,
            "completed_weave_agents_completion_phases": completed_weave_agents_phases,
            "missing_required_weave_agents_completion_phases": missing_required_weave_agents_phases,
            "nemoclaw_agentic_config_ok": nemoclaw_agentic_config_ok,
            "bad_nemoclaw_agentic_config_records": bad_nemoclaw_agentic_config_records,
            "remediation_commands": [] if ok else one_model_canary_commands(),
        },
    )


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _accounting_value_placeholder(value: str) -> bool:
    normalized = value.strip().casefold()
    collapsed = re.sub(r"[\s_:/\\|.,;\-]+", " ", normalized).strip()
    compact = re.sub(r"[^0-9a-zぁ-んァ-ン一-龥]+", "", normalized).strip()
    return (
        collapsed in PLACEHOLDER_ACCOUNTING_VALUES
        or compact in PLACEHOLDER_ACCOUNTING_VALUES
        or any(
            collapsed == prefix or collapsed.startswith(prefix + " ")
            for prefix in PLACEHOLDER_ACCOUNTING_PREFIXES
        )
    )


def _scope_confirmed_at_valid(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    text = value.strip()
    if any(marker in text for marker in ("YYYY", "MM", "DD", "HH", "SS")):
        return False
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _scope_confirmation_valid(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if len(stripped) < MIN_SCOPE_CONFIRMATION_LENGTH:
        return False
    return not _accounting_value_placeholder(stripped)


def _review_common_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("status", "phase", "run_purpose", "expected_cost_band"):
        if not _nonempty_string(payload.get(field)):
            errors.append(f"missing {field}")
    model_count = payload.get("model_count")
    if not isinstance(model_count, int) or model_count <= 0:
        errors.append("model_count must be a positive integer")
    configs = payload.get("configs")
    if not isinstance(configs, list) or not configs:
        errors.append("configs must be a non-empty list")
    if numeric_timestamp(payload.get("created_at")) is None:
        errors.append("missing numeric created_at")
    for field in ("execution_plan_path", "batch_manifest_path", "post_run_cost_command"):
        if not _nonempty_string(payload.get(field)):
            errors.append(f"missing {field}")
    if payload.get("status") == "accountability_fields_missing":
        missing = payload.get("missing_fields")
        if isinstance(missing, list) and missing:
            errors.append("accountability fields missing: " + ", ".join(map(str, missing)))
        else:
            errors.append("accountability fields missing")
    pre_run_budget = _review_pre_run_budget_estimate(payload)
    if pre_run_budget["required_before_paid_execution"] and not pre_run_budget["valid"]:
        errors.extend(f"pre_run_budget_estimate {error}" for error in pre_run_budget["errors"])
    external_action_approval = _review_external_action_approval(payload)
    if (
        external_action_approval["required_before_external_action"]
        and not external_action_approval["valid"]
    ):
        errors.extend(
            f"external_action_approval {error}"
            for error in external_action_approval["errors"]
        )
    return errors


def _review_pre_run_budget_estimate(payload: dict[str, Any]) -> dict[str, Any]:
    record = payload.get("pre_run_budget_estimate")
    required = bool(payload.get("requires_paid_model_api"))
    result: dict[str, Any] = {
        "required_before_paid_execution": required,
        "present": False,
        "valid": False,
        "path": "",
        "sha256": "",
        "sha256_matches": False,
        "target_model": "",
        "target_models": [],
        "selected_model_identifiers": [],
        "target_model_matches_selected_config": False,
        "estimated_total_usd": {},
        "pricing_source_url": "",
        "errors": [],
    }
    if not isinstance(record, dict):
        result["errors"].append("is missing or not an object")
        return result

    result.update(
        {
            "present": bool(record.get("present")),
            "path": record.get("path") if isinstance(record.get("path"), str) else "",
            "sha256": record.get("sha256") if isinstance(record.get("sha256"), str) else "",
            "target_model": record.get("target_model") if isinstance(record.get("target_model"), str) else "",
            "target_models": (
                record.get("target_models") if isinstance(record.get("target_models"), list) else []
            ),
            "selected_model_identifiers": (
                record.get("selected_model_identifiers")
                if isinstance(record.get("selected_model_identifiers"), list)
                else []
            ),
            "target_model_matches_selected_config": bool(
                record.get("target_model_matches_selected_config")
            ),
            "estimated_total_usd": (
                record.get("estimated_total_usd")
                if isinstance(record.get("estimated_total_usd"), dict)
                else {}
            ),
            "pricing_source_url": (
                record.get("pricing_source_url")
                if isinstance(record.get("pricing_source_url"), str)
                else ""
            ),
        }
    )
    nested_errors = record.get("errors")
    if isinstance(nested_errors, list):
        result["errors"].extend(str(error) for error in nested_errors if error)
    if not result["path"]:
        result["errors"].append("path is missing")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", result["sha256"]):
        result["errors"].append("sha256 is missing or invalid")
    if not result["target_model"]:
        result["errors"].append("target_model is missing")
    selected_config_model_bindings = record.get("selected_config_model_bindings")
    if isinstance(selected_config_model_bindings, list) and selected_config_model_bindings:
        if record.get("target_model_matches_selected_config") is not True:
            result["errors"].append(
                "target_model does not match selected config model identifiers"
            )
    if not result["pricing_source_url"]:
        result["errors"].append("pricing_source_url is missing")
    for key in ("low", "mid", "high"):
        if not isinstance(result["estimated_total_usd"].get(key), (int, float)):
            result["errors"].append(f"estimated_total_usd.{key} is missing or not numeric")
    if result["path"]:
        budget_path = repo_path(result["path"])
        if budget_path.exists():
            try:
                actual_sha = sha256_file(budget_path)
            except OSError as exc:
                result["errors"].append(f"sha256 could not be recomputed: {exc}")
            else:
                result["sha256_matches"] = bool(result["sha256"]) and actual_sha == result["sha256"]
                if not result["sha256_matches"]:
                    result["errors"].append("sha256 does not match source file")
        elif required:
            result["errors"].append("source file does not exist")
    result["valid"] = not result["errors"] and bool(record.get("valid", True))
    if record.get("valid") is False and not result["errors"]:
        result["errors"].append("embedded valid flag is false")
        result["valid"] = False
    return result


def _review_external_action_approval(payload: dict[str, Any]) -> dict[str, Any]:
    record = payload.get("external_action_approval")
    required = bool(payload.get("will_execute_external_actions"))
    result: dict[str, Any] = {
        "required_before_external_action": required,
        "present": False,
        "valid": False,
        "path": "",
        "sha256": "",
        "sha256_matches": False,
        "status": "",
        "required_approval_count": None,
        "granted_approval_count": None,
        "all_required_approvals_granted": False,
        "source_binding_bound": False,
        "source_approval_packet_sha256": "",
        "will_execute_external_actions": None,
        "errors": [],
    }
    if not isinstance(record, dict):
        if required:
            result["errors"].append("is missing or not an object")
        return result

    source_binding = (
        record.get("source_binding")
        if isinstance(record.get("source_binding"), dict)
        else {}
    )
    result.update(
        {
            "present": bool(record.get("present")),
            "path": record.get("path") if isinstance(record.get("path"), str) else "",
            "sha256": record.get("sha256") if isinstance(record.get("sha256"), str) else "",
            "status": record.get("status") if isinstance(record.get("status"), str) else "",
            "required_approval_count": record.get("required_approval_count"),
            "granted_approval_count": record.get("granted_approval_count"),
            "all_required_approvals_granted": bool(
                record.get("all_required_approvals_granted")
            ),
            "source_binding_bound": source_binding.get("bound") is True,
            "source_approval_packet_sha256": (
                source_binding.get("source_approval_packet_sha256")
                if isinstance(source_binding.get("source_approval_packet_sha256"), str)
                else ""
            ),
            "will_execute_external_actions": record.get("will_execute_external_actions"),
        }
    )
    nested_errors = record.get("errors")
    if isinstance(nested_errors, list):
        result["errors"].extend(str(error) for error in nested_errors if error)
    if not result["path"]:
        result["errors"].append("path is missing")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", result["sha256"]):
        result["errors"].append("sha256 is missing or invalid")
    if result["status"] != "approved":
        result["errors"].append("status is not approved")
    if not isinstance(result["required_approval_count"], int) or result["required_approval_count"] <= 0:
        result["errors"].append("required_approval_count is missing or invalid")
    if result["granted_approval_count"] != result["required_approval_count"]:
        result["errors"].append("granted_approval_count does not match required_approval_count")
    if not result["all_required_approvals_granted"]:
        result["errors"].append("all_required_approvals_granted is false")
    if not result["source_binding_bound"]:
        result["errors"].append("source_binding.bound is not true")
    if not re.fullmatch(r"[0-9a-f]{64}", result["source_approval_packet_sha256"]):
        result["errors"].append("source_approval_packet_sha256 is missing or invalid")
    if result["will_execute_external_actions"] is not False:
        result["errors"].append("will_execute_external_actions is not false")
    if result["path"]:
        report_path = repo_path(result["path"])
        if report_path.exists():
            try:
                actual_sha = sha256_file(report_path)
            except OSError as exc:
                result["errors"].append(f"sha256 could not be recomputed: {exc}")
            else:
                result["sha256_matches"] = bool(result["sha256"]) and actual_sha == result["sha256"]
                if not result["sha256_matches"]:
                    result["errors"].append("sha256 does not match source file")
        elif required:
            result["errors"].append("source file does not exist")
    result["valid"] = not result["errors"] and bool(record.get("valid", True))
    if record.get("valid") is False and not result["errors"]:
        result["errors"].append("embedded valid flag is false")
        result["valid"] = False
    return result


def _review_run_eval_preflight_payload(
    path_value: Any,
    *,
    run_index: int,
) -> list[str]:
    errors: list[str] = []
    if not _nonempty_string(path_value):
        return [f"run {run_index} missing preflight_json"]
    preflight_path = repo_path(str(path_value))
    payload, error = read_json(preflight_path)
    if payload is None:
        return [f"run {run_index} preflight_json could not be read: {error or 'invalid JSON'}"]
    if payload.get("ok") is not True:
        errors.append(f"run {run_index} preflight payload ok must be true")
    if payload.get("status") != "passed":
        errors.append(f"run {run_index} preflight payload status must be passed")
    for field in (
        "will_initialize_wandb",
        "will_log_wandb_artifacts",
        "will_initialize_weave",
        "will_start_inference_engine",
        "will_run_evaluators",
    ):
        if payload.get(field) is not False:
            errors.append(f"run {run_index} preflight payload {field} must be false")
    enabled_benchmarks = payload.get("enabled_benchmarks")
    if not isinstance(enabled_benchmarks, list):
        errors.append(f"run {run_index} preflight payload enabled_benchmarks must be a list")
    return errors


def _review_run_eval_preflight_records(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    records = payload.get("run_eval_preflights")
    model_count = payload.get("model_count")
    if not isinstance(records, list) or not records:
        return ["completed review must include run_eval_preflights"]
    if isinstance(model_count, int) and model_count > 0 and len(records) != model_count:
        errors.append(
            f"completed review has {len(records)} run_eval_preflights but model_count is {model_count}"
        )
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            errors.append(f"run_eval_preflights {index} is not an object")
            continue
        for field in ("config", "output_json"):
            if not _nonempty_string(record.get(field)):
                errors.append(f"run_eval_preflights {index} missing {field}")
        command = record.get("command")
        if not isinstance(command, list) or not command:
            errors.append(f"run_eval_preflights {index} missing command")
        else:
            command_parts = [str(part) for part in command]
            if "scripts/run_eval.py" not in command_parts:
                errors.append(f"run_eval_preflights {index} command must invoke scripts/run_eval.py")
            if "--preflight" not in command_parts:
                errors.append(f"run_eval_preflights {index} command missing --preflight")
            if "--preflight-json" not in command_parts:
                errors.append(f"run_eval_preflights {index} command missing --preflight-json")
            elif _nonempty_string(record.get("output_json")):
                try:
                    output_arg = command_parts[command_parts.index("--preflight-json") + 1]
                except IndexError:
                    errors.append(
                        f"run_eval_preflights {index} command --preflight-json missing value"
                    )
                else:
                    if output_arg != record.get("output_json"):
                        errors.append(
                            f"run_eval_preflights {index} command --preflight-json "
                            "does not match output_json"
                        )
            if "--config" in command_parts and _nonempty_string(record.get("config")):
                try:
                    config_arg = command_parts[command_parts.index("--config") + 1]
                except IndexError:
                    errors.append(f"run_eval_preflights {index} command --config missing value")
                else:
                    if config_arg != record.get("config"):
                        errors.append(
                            f"run_eval_preflights {index} command --config does not match config"
                        )
            else:
                errors.append(f"run_eval_preflights {index} command missing --config")
        if record.get("required_before_run_eval") is not True:
            errors.append(
                f"run_eval_preflights {index} required_before_run_eval must be true"
            )
    return errors


def _review_completed_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if numeric_timestamp(payload.get("ended_at")) is None:
        errors.append("completed review is missing numeric ended_at")
    actual_cost_estimate = payload.get("actual_cost_estimate")
    provider_bill_reference = payload.get("provider_bill_reference")
    if not _nonempty_string(actual_cost_estimate):
        errors.append("completed review is missing actual_cost_estimate")
    elif _accounting_value_placeholder(actual_cost_estimate):
        errors.append("completed review actual_cost_estimate must not be a placeholder")
    if not _nonempty_string(provider_bill_reference):
        errors.append("completed review is missing provider_bill_reference")
    elif _accounting_value_placeholder(provider_bill_reference):
        errors.append("completed review provider_bill_reference must not be a placeholder")
    runs = payload.get("runs")
    model_count = payload.get("model_count")
    errors.extend(_review_run_eval_preflight_records(payload))
    if not isinstance(runs, list) or not runs:
        errors.append("completed review must include runs")
        return errors
    if isinstance(model_count, int) and model_count > 0 and len(runs) != model_count:
        errors.append(f"completed review has {len(runs)} runs but model_count is {model_count}")
    verify_wandb = bool(payload.get("verify_wandb_completion"))
    verify_weave = bool(payload.get("verify_weave_agents"))
    top_preflight_outputs = {
        str(record.get("output_json"))
        for record in payload.get("run_eval_preflights", [])
        if isinstance(record, dict) and _nonempty_string(record.get("output_json"))
    } if isinstance(payload.get("run_eval_preflights"), list) else set()
    for index, run in enumerate(runs, start=1):
        if not isinstance(run, dict):
            errors.append(f"run {index} is not an object")
            continue
        for field in ("config", "log_path"):
            if not _nonempty_string(run.get(field)):
                errors.append(f"run {index} missing {field}")
        if not _nonempty_string(run.get("preflight_json")):
            errors.append(f"run {index} missing preflight_json")
        elif top_preflight_outputs and str(run.get("preflight_json")) not in top_preflight_outputs:
            errors.append(
                f"run {index} preflight_json does not match a top-level run_eval_preflights output_json"
            )
        if run.get("preflight_returncode") != 0:
            errors.append(f"run {index} preflight_returncode must be 0")
        if run.get("preflight_ok") is not True:
            errors.append(f"run {index} preflight_ok must be true")
        if _nonempty_string(run.get("preflight_json")):
            errors.extend(
                _review_run_eval_preflight_payload(
                    run.get("preflight_json"),
                    run_index=index,
                )
            )
        if "returncode" not in run:
            errors.append(f"run {index} missing returncode")
        if numeric_timestamp(run.get("started_at")) is None:
            errors.append(f"run {index} missing numeric started_at")
        if numeric_timestamp(run.get("ended_at")) is None:
            errors.append(f"run {index} missing numeric ended_at")
        if run.get("returncode") == 0 and not _nonempty_string(run.get("wandb_run_id")):
            errors.append(f"run {index} succeeded but is missing wandb_run_id")
        if verify_wandb and run.get("returncode") == 0:
            for field in ("wandb_entity", "wandb_project"):
                if not _nonempty_string(run.get(field)):
                    errors.append(f"run {index} succeeded with W&B completion enabled but is missing {field}")
        if verify_wandb:
            completions = run.get("wandb_completion")
            if not isinstance(completions, list) or not completions:
                errors.append(f"run {index} missing W&B completion entries")
                continue
            for completion_index, completion in enumerate(completions, start=1):
                if not isinstance(completion, dict):
                    errors.append(f"run {index} W&B completion {completion_index} is not an object")
                    continue
                if not completion.get("ok"):
                    errors.append(f"run {index} W&B completion {completion_index} did not pass")
                for field in ("benchmark", "path", "run_id"):
                    if not _nonempty_string(completion.get(field)):
                        errors.append(f"run {index} W&B completion {completion_index} missing {field}")
        if verify_weave and run.get("returncode") == 0:
            completion = run.get("weave_agents_completion")
            if not isinstance(completion, dict):
                errors.append(f"run {index} missing Weave Agents completion entry")
                continue
            if not completion.get("ok"):
                errors.append(f"run {index} Weave Agents completion did not pass")
            for field in ("path", "agent_name"):
                if not _nonempty_string(completion.get(field)):
                    errors.append(f"run {index} Weave Agents completion missing {field}")
    return errors


def _review_wandb_completion_verification_errors(
    payload: dict[str, Any],
    *,
    max_age_seconds: int | None,
    review_path: str = "",
) -> tuple[list[str], list[dict[str, Any]]]:
    entries = [
        _verify_review_wandb_completion_entry(
            entry,
            max_age_seconds=max_age_seconds,
        )
        for entry in _review_wandb_completion_entries(payload, review_path=review_path)
    ]
    errors: list[str] = []
    if not bool(payload.get("verify_wandb_completion")):
        return errors, entries
    for entry in entries:
        if entry.get("verified"):
            continue
        benchmark = entry.get("benchmark") or "unknown_benchmark"
        run_id = entry.get("run_id") or "unknown_run_id"
        reason = (
            entry.get("verification_error")
            or entry.get("freshness_error")
            or "not verified"
        )
        errors.append(
            f"W&B completion entry {benchmark}/{run_id} is not verified: {reason}"
        )
    return errors, entries


def _review_weave_agents_completion_verification_errors(
    payload: dict[str, Any],
    *,
    review_path: str = "",
    max_age_seconds: int | None,
) -> tuple[list[str], list[dict[str, Any]]]:
    entries = [
        _verify_review_weave_agents_completion_entry(
            entry,
            max_age_seconds=max_age_seconds,
        )
        for entry in _review_weave_agents_completion_entries(
            payload,
            review_path=review_path,
        )
    ]
    errors: list[str] = []
    if not bool(payload.get("verify_weave_agents")):
        return errors, entries
    for entry in entries:
        if entry.get("verified"):
            continue
        agent_name = entry.get("agent_name") or "unknown_agent"
        run_id = entry.get("run_id") or "unknown_run_id"
        reason = (
            entry.get("verification_error")
            or entry.get("freshness_error")
            or "not verified"
        )
        errors.append(
            f"Weave Agents completion entry {agent_name}/{run_id} is not verified: {reason}"
        )
    return errors, entries


def evaluate_paid_run_review_package(
    paths: list[Path],
    *,
    require: bool,
    wandb_completion_max_age_seconds: int | None = DEFAULT_WANDB_COMPLETION_MAX_AGE_SECONDS,
    weave_agents_completion_max_age_seconds: int | None = DEFAULT_WANDB_COMPLETION_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    requirement = (
        "Paid-run review records must include purpose, model/benchmark scope, "
        "expected cost band, run evidence, and post-run cost accounting before release."
    )
    if not paths:
        return gate_record(
            name="paid_run_review_package",
            ok=not require,
            blocking=require,
            status="missing",
            requirement=requirement,
            evidence_paths=[],
            detail="No paid-run review JSON was found.",
            next_action="Create a paid-run review package before paid execution and update it after completion.",
            extra={"records": [], "remediation_commands": paid_run_review_commands()},
        )

    records: list[dict[str, Any]] = []
    complete_records: list[dict[str, Any]] = []
    blocking_records: list[dict[str, Any]] = []
    incomplete_status_records: list[dict[str, Any]] = []
    for path in paths:
        payload, error = read_json(path)
        if payload is None:
            record = {
                "path": path_display(path),
                "ok": False,
                "status": "invalid",
                "errors": [error or "invalid JSON"],
            }
            records.append(record)
            blocking_records.append(record)
            continue
        status = str(payload.get("status") or "")
        errors = _review_common_errors(payload)
        configs = _review_configs(payload)
        requires_nemoclaw_agentic_config = _review_requires_nemoclaw_agentic_config(payload)
        nemoclaw_agentic_config_bound = _configs_include_nemoclaw_agentic_config(configs)
        wandb_entry_errors: list[str] = []
        verified_wandb_completion_entries: list[dict[str, Any]] = []
        weave_entry_errors: list[str] = []
        verified_weave_agents_completion_entries: list[dict[str, Any]] = []
        if status == "completed":
            errors.extend(_review_completed_errors(payload))
            if requires_nemoclaw_agentic_config and not nemoclaw_agentic_config_bound:
                errors.append(
                    "completed agentic canary review must reference the NeMoClaw agentic generated config"
                )
            wandb_entry_errors, verified_wandb_completion_entries = (
                _review_wandb_completion_verification_errors(
                    payload,
                    review_path=path_display(path),
                    max_age_seconds=wandb_completion_max_age_seconds,
                )
            )
            errors.extend(wandb_entry_errors)
            weave_entry_errors, verified_weave_agents_completion_entries = (
                _review_weave_agents_completion_verification_errors(
                    payload,
                    review_path=path_display(path),
                    max_age_seconds=weave_agents_completion_max_age_seconds,
                )
            )
            errors.extend(weave_entry_errors)
        else:
            errors.append(f"review status is not completed: {status or 'missing'}")
            verified_wandb_completion_entries = [
                _verify_review_wandb_completion_entry(
                    entry,
                    max_age_seconds=wandb_completion_max_age_seconds,
                )
                for entry in _review_wandb_completion_entries(
                    payload,
                    review_path=path_display(path),
                )
            ]
            verified_weave_agents_completion_entries = [
                _verify_review_weave_agents_completion_entry(
                    entry,
                    max_age_seconds=weave_agents_completion_max_age_seconds,
                )
                for entry in _review_weave_agents_completion_entries(
                    payload,
                    review_path=path_display(path),
                )
            ]
        record = {
            "path": path_display(path),
            "ok": not errors,
            "status": status,
            "phase": payload.get("phase"),
            "canary": bool(payload.get("canary")),
            "model_count": payload.get("model_count"),
            "configs": configs,
            "nemoclaw_agentic_config_required": requires_nemoclaw_agentic_config,
            "nemoclaw_agentic_config_bound": nemoclaw_agentic_config_bound,
            "requires_paid_model_api": bool(payload.get("requires_paid_model_api")),
            "will_execute_external_actions": bool(payload.get("will_execute_external_actions")),
            "run_purpose_present": _nonempty_string(payload.get("run_purpose")),
            "expected_cost_band_present": _nonempty_string(payload.get("expected_cost_band")),
            "pre_run_budget_estimate": _review_pre_run_budget_estimate(payload),
            "external_action_approval": _review_external_action_approval(payload),
            "actual_cost_estimate_present": _nonempty_string(payload.get("actual_cost_estimate")),
            "provider_bill_reference_present": _nonempty_string(payload.get("provider_bill_reference")),
            "actual_cost_estimate_placeholder": isinstance(payload.get("actual_cost_estimate"), str)
            and _accounting_value_placeholder(payload.get("actual_cost_estimate")),
            "provider_bill_reference_placeholder": isinstance(payload.get("provider_bill_reference"), str)
            and _accounting_value_placeholder(payload.get("provider_bill_reference")),
            "run_count": len(payload.get("runs")) if isinstance(payload.get("runs"), list) else 0,
            "verify_wandb_completion": bool(payload.get("verify_wandb_completion")),
            "wandb_completion_entries": verified_wandb_completion_entries,
            "wandb_completion_max_age_seconds": wandb_completion_max_age_seconds,
            "verify_weave_agents": bool(payload.get("verify_weave_agents")),
            "weave_agents_completion_entries": verified_weave_agents_completion_entries,
            "weave_agents_completion_max_age_seconds": weave_agents_completion_max_age_seconds,
            "errors": errors,
        }
        records.append(record)
        if not errors and status == "completed":
            complete_records.append(record)
        else:
            blocking_records.append(record)
            if status != "completed":
                incomplete_status_records.append(record)

    ok = bool(complete_records) and not blocking_records
    status = (
        "passed"
        if ok
        else "incomplete_reviews"
        if incomplete_status_records
        else "invalid_review_package"
    )
    return gate_record(
        name="paid_run_review_package",
        ok=ok or not require,
        blocking=require,
        status=status,
        requirement=requirement,
        evidence_paths=paths,
        detail=(
            "Paid-run review package is complete."
            if ok
            else "Paid-run review package is missing completion or accounting evidence."
        ),
        next_action=(
            "Include this review package in the release evidence."
            if ok
            else "Complete the paid-run review JSONs, including actual cost estimate, W&B completion entries, and required Weave Agents completion entries, then rerun this gate."
        ),
        extra={
            "records": records,
            "complete_records": complete_records,
            "blocking_records": blocking_records,
            "wandb_completion_max_age_seconds": wandb_completion_max_age_seconds,
            "weave_agents_completion_max_age_seconds": weave_agents_completion_max_age_seconds,
            "completion_requirements": paid_run_review_completion_requirements(
                wandb_completion_max_age_seconds=wandb_completion_max_age_seconds,
                weave_agents_completion_max_age_seconds=weave_agents_completion_max_age_seconds,
            ),
            "remediation_commands": [] if ok else paid_run_review_commands(),
        },
    )


def remediation_plan(gates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for gate in gates:
        if not gate.get("blocking") or gate.get("ok"):
            continue
        commands = gate.get("remediation_commands")
        if isinstance(commands, list) and commands:
            plan.append(
                {
                    "gate": gate.get("name"),
                    "status": gate.get("status"),
                    "next_action": gate.get("next_action"),
                    "commands": commands,
                }
            )
    return plan


def _gate_by_name(gates: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for gate in gates:
        if gate.get("name") == name:
            return gate
    return {}


def _matching_wandb_records(gate: dict[str, Any], benchmark: str) -> list[dict[str, Any]]:
    records = gate.get("records")
    if not isinstance(records, list):
        return []
    return [
        record
        for record in records
        if isinstance(record, dict) and record.get("benchmark") == benchmark
    ]


def _wandb_completion_status(gate: dict[str, Any], benchmark: str) -> dict[str, Any]:
    completed = gate.get("completed")
    completed_paths = []
    if isinstance(completed, dict):
        paths = completed.get(benchmark)
        if isinstance(paths, list):
            completed_paths = [str(path) for path in paths]
    records = _matching_wandb_records(gate, benchmark)
    if completed_paths:
        return {
            "ok": True,
            "status": "passed",
            "evidence_paths": completed_paths,
            "records": records,
        }
    if not records:
        status = "missing"
    elif any(record.get("freshness_error") == "missing generated_at" for record in records):
        status = "missing_generated_at"
    elif any(record.get("ok") and record.get("run_id_matches") and not record.get("fresh") for record in records):
        status = "stale"
    elif any(record.get("ok") and not record.get("run_id_matches") for record in records):
        status = "run_id_mismatch"
    elif any(record.get("error") for record in records):
        status = "invalid_verifier_json"
    else:
        status = "not_passing"
    return {
        "ok": False,
        "status": status,
        "evidence_paths": [
            str(record["path"])
            for record in records
            if isinstance(record.get("path"), str)
        ],
        "records": records,
    }


def _review_entries_for_benchmark(gate: dict[str, Any], benchmark: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    records = gate.get("records")
    if not isinstance(records, list):
        return entries
    for record in records:
        if not isinstance(record, dict):
            continue
        review_path = record.get("path")
        raw_entries = record.get("wandb_completion_entries")
        if not isinstance(raw_entries, list):
            continue
        for entry in raw_entries:
            if not isinstance(entry, dict) or entry.get("benchmark") != benchmark:
                continue
            item = dict(entry)
            if isinstance(review_path, str):
                item["review_path"] = review_path
            entries.append(item)
    return entries


def _review_completion_status(gate: dict[str, Any], benchmark: str) -> dict[str, Any]:
    completed = gate.get("completed_wandb_completion_benchmarks")
    completed_entries = []
    if isinstance(completed, dict):
        rows = completed.get(benchmark)
        if isinstance(rows, list):
            completed_entries = [row for row in rows if isinstance(row, dict)]
    if completed_entries:
        return {
            "ok": True,
            "status": "passed",
            "entries": completed_entries,
            "review_paths": sorted(
                {
                    str(entry["review_path"])
                    for entry in completed_entries
                    if isinstance(entry.get("review_path"), str)
                }
            ),
        }

    entries = _review_entries_for_benchmark(gate, benchmark)
    if not entries:
        status = "missing_review_entry"
    elif any(entry.get("verification_error") == "missing verifier JSON path" for entry in entries):
        status = "missing_verifier_json_path"
    elif any(entry.get("freshness_error") == "missing generated_at" for entry in entries):
        status = "missing_generated_at"
    elif any(entry.get("verifier_json_ok") and not entry.get("fresh") for entry in entries):
        status = "stale"
    elif any(entry.get("verifier_json_ok") and not entry.get("run_id_matches") for entry in entries):
        status = "run_id_mismatch"
    elif any(entry.get("verifier_json_ok") and not entry.get("benchmark_matches") for entry in entries):
        status = "benchmark_mismatch"
    elif any(not entry.get("verifier_json_ok") for entry in entries):
        status = "verifier_json_not_passing"
    else:
        status = "not_verified"
    return {
        "ok": False,
        "status": status,
        "entries": entries,
        "review_paths": sorted(
            {
                str(entry["review_path"])
                for entry in entries
                if isinstance(entry.get("review_path"), str)
            }
        ),
    }


def benchmark_evidence_matrix(
    *,
    gates: list[dict[str, Any]],
    required_benchmarks: list[str],
    required_run_ids: dict[str, str],
) -> list[dict[str, Any]]:
    wandb_gate = _gate_by_name(gates, "wandb_completion")
    review_gate = _gate_by_name(gates, "one_model_full_canary")
    matrix: list[dict[str, Any]] = []
    for benchmark in required_benchmarks:
        wandb_status = _wandb_completion_status(wandb_gate, benchmark)
        review_status = _review_completion_status(review_gate, benchmark)
        matrix.append(
            {
                "benchmark": benchmark,
                "required": True,
                "expected_run_id": required_run_ids.get(benchmark),
                "standalone_wandb_completion": wandb_status,
                "review_wandb_completion": review_status,
                "completion_proven": bool(wandb_status["ok"] and review_status["ok"]),
            }
        )
    return matrix


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    readiness_paths = filter_canary_readiness_paths(
        normalize_paths(args.readiness_json, DEFAULT_READINESS_GLOBS)
    )
    nemoclaw_setup_paths = normalize_paths(args.nemoclaw_setup_json, (DEFAULT_NEMOCLAW_SETUP_GLOB,))
    nemoclaw_installer_review_paths = normalize_paths(
        getattr(args, "nemoclaw_installer_review_json", None),
        (DEFAULT_NEMOCLAW_INSTALLER_REVIEW_GLOB,),
    )
    require_nemoclaw_operator_docs = not getattr(
        args,
        "no_require_nemoclaw_operator_docs",
        True,
    )
    explicit_nemoclaw_operator_docs = getattr(args, "nemoclaw_operator_docs_json", None)
    nemoclaw_operator_docs_paths = (
        normalize_paths(
            explicit_nemoclaw_operator_docs,
            (DEFAULT_NEMOCLAW_OPERATOR_DOCS_GLOB,),
        )
        if require_nemoclaw_operator_docs or explicit_nemoclaw_operator_docs
        else []
    )
    weave_gate_paths = normalize_paths(args.weave_content_canary_gate, (DEFAULT_WEAVE_GATE_GLOB,))
    wandb_completion_paths = normalize_paths(args.wandb_completion_json, (DEFAULT_WANDB_COMPLETION_GLOB,))
    existing_results_audit_paths = normalize_paths(
        getattr(args, "existing_results_audit_json", None),
        (DEFAULT_EXISTING_RESULTS_AUDIT_GLOB,),
    )
    review_paths = normalize_paths(args.batch_review_json, (str(args.output_root / "*paid_run_review.json"),))
    required_wandb_benchmarks = args.required_wandb_benchmark or list(DEFAULT_REQUIRED_WANDB_BENCHMARKS)
    required_wandb_run_ids = parse_required_wandb_run_ids(
        args.required_wandb_run_id,
        required_benchmarks=required_wandb_benchmarks,
        all_run_id=args.required_wandb_run_id_all,
    )

    gates = [
        evaluate_metadata_readiness(readiness_paths, require=not args.no_require_metadata_readiness),
        evaluate_weave_content_gate(
            weave_gate_paths,
            require=not args.no_require_weave_content_canary,
            max_age_seconds=args.weave_content_canary_max_age_seconds,
        ),
        evaluate_nemoclaw_readiness(
            readiness_paths,
            require=not args.no_require_nemoclaw,
            setup_paths=nemoclaw_setup_paths,
            installer_review_paths=nemoclaw_installer_review_paths,
        ),
    ]
    if require_nemoclaw_operator_docs or nemoclaw_operator_docs_paths:
        gates.append(
            evaluate_nemoclaw_operator_docs(
                nemoclaw_operator_docs_paths,
                require=require_nemoclaw_operator_docs,
            )
        )
    gates.extend(
        [
            evaluate_wandb_completion(
                wandb_completion_paths,
                required_benchmarks=required_wandb_benchmarks,
                required_run_ids=required_wandb_run_ids,
                require=not args.no_require_wandb_completion,
                max_age_seconds=args.wandb_completion_max_age_seconds,
            ),
            evaluate_existing_results_formalization(
                existing_results_audit_paths,
                require=not getattr(args, "no_require_existing_results_audit", False),
            ),
            evaluate_paid_run_review_package(
                review_paths,
                require=not args.no_require_paid_run_review_package,
                wandb_completion_max_age_seconds=args.wandb_completion_max_age_seconds,
            ),
            evaluate_one_model_canary(
                review_paths,
                require=not args.no_require_one_model_canary,
                required_run_ids=required_wandb_run_ids,
                required_benchmarks=required_wandb_benchmarks,
                wandb_completion_max_age_seconds=args.wandb_completion_max_age_seconds,
            ),
        ]
    )
    blockers = [gate for gate in gates if gate["blocking"] and not gate["ok"]]
    remediation = remediation_plan(gates)
    benchmark_evidence = benchmark_evidence_matrix(
        gates=gates,
        required_benchmarks=required_wandb_benchmarks,
        required_run_ids=required_wandb_run_ids,
    )
    return {
        "schema_version": 1,
        "ok": not blockers,
        "status": "ready" if not blockers else "not_ready",
        "generated_at": time.time(),
        "requirements": {
            "metadata_readiness": not args.no_require_metadata_readiness,
            "weave_content_canary": not args.no_require_weave_content_canary,
            "weave_content_canary_max_age_seconds": args.weave_content_canary_max_age_seconds,
            "nemoclaw": not args.no_require_nemoclaw,
            "nemoclaw_operator_docs": require_nemoclaw_operator_docs,
            "wandb_completion": not args.no_require_wandb_completion,
            "wandb_completion_max_age_seconds": args.wandb_completion_max_age_seconds,
            "existing_results_audit": not getattr(args, "no_require_existing_results_audit", False),
            "paid_run_review_package": not args.no_require_paid_run_review_package,
            "one_model_canary": not args.no_require_one_model_canary,
            "required_wandb_benchmarks": required_wandb_benchmarks,
            "required_wandb_run_ids": required_wandb_run_ids,
            "nemoclaw_installer_review_json": [
                path_display(path) for path in nemoclaw_installer_review_paths
            ],
            "nemoclaw_operator_docs_json": [
                path_display(path) for path in nemoclaw_operator_docs_paths
            ],
        },
        "summary": {
            "gate_count": len(gates),
            "blocker_count": len(blockers),
            "blockers": [gate["name"] for gate in blockers],
            "benchmark_evidence": benchmark_evidence,
        },
        "remediation_plan": remediation,
        "gates": gates,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--readiness-json", type=Path, action="append")
    parser.add_argument("--nemoclaw-setup-json", type=Path, action="append")
    parser.add_argument("--nemoclaw-installer-review-json", type=Path, action="append")
    parser.add_argument("--nemoclaw-operator-docs-json", type=Path, action="append")
    parser.add_argument("--weave-content-canary-gate", type=Path, action="append")
    parser.add_argument(
        "--weave-content-canary-max-age-seconds",
        type=int,
        default=DEFAULT_WEAVE_CONTENT_CANARY_MAX_AGE_SECONDS,
        help=(
            "Maximum age for a passing Weave content canary gate. "
            "Use a negative value only to disable freshness checks deliberately."
        ),
    )
    parser.add_argument("--wandb-completion-json", type=Path, action="append")
    parser.add_argument("--existing-results-audit-json", type=Path, action="append")
    parser.add_argument(
        "--wandb-completion-max-age-seconds",
        type=int,
        default=DEFAULT_WANDB_COMPLETION_MAX_AGE_SECONDS,
        help=(
            "Maximum age for passing W&B completion verifier JSONs. "
            "Use a negative value only to disable freshness checks deliberately."
        ),
    )
    parser.add_argument("--batch-review-json", type=Path, action="append")
    parser.add_argument("--no-require-existing-results-audit", action="store_true")
    parser.add_argument("--no-require-nemoclaw-operator-docs", action="store_true")
    parser.add_argument("--no-require-paid-run-review-package", action="store_true")
    parser.add_argument(
        "--required-wandb-benchmark",
        action="append",
        default=None,
        choices=["agentic_math", "agentic_swe", "taiwan_full"],
    )
    parser.add_argument(
        "--required-wandb-run-id",
        action="append",
        default=None,
        metavar="BENCHMARK=RUN_ID",
        help="Require a passing completion JSON for a specific W&B run id.",
    )
    parser.add_argument(
        "--required-wandb-run-id-all",
        metavar="RUN_ID",
        help="Require every selected W&B completion benchmark to use the same run id.",
    )
    parser.add_argument("--no-require-metadata-readiness", action="store_true")
    parser.add_argument("--no-require-weave-content-canary", action="store_true")
    parser.add_argument("--no-require-nemoclaw", action="store_true")
    parser.add_argument("--no-require-wandb-completion", action="store_true")
    parser.add_argument("--no-require-one-model-canary", action="store_true")
    parser.add_argument("--json", type=Path, help="Optional path to write the readiness report.")
    parser.add_argument("--fail-on-not-ready", action="store_true")
    args = parser.parse_args(argv)
    if args.weave_content_canary_max_age_seconds < 0:
        args.weave_content_canary_max_age_seconds = None
    if args.wandb_completion_max_age_seconds < 0:
        args.wandb_completion_max_age_seconds = None
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_not_ready and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
