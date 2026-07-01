#!/usr/bin/env python3
"""Build a local Taiwan leaderboard release evidence bundle.

This tool is offline: it reads an existing production-readiness report and
copies referenced local evidence JSON files into a reproducible bundle
directory. It does not query W&B, call model providers, or install NeMoClaw.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path("outputs") / "taiwan_release_evidence"
COMMAND_SCRIPT_SUFFIXES = (".py", ".sh", ".mjs")
RELOG_WANDB_APPROVAL_HELPER_SCRIPT = "scripts/tools/relog_wandb_approval.py"
RELOG_COMMAND_SCRIPT_PATHS = {
    "scripts/tools/log_agentic_math_results_to_wandb.py",
    "scripts/tools/log_agentic_swe_results_to_wandb.py",
}
EXTERNAL_ACTION_APPROVAL_PACKET_VERIFIER_SCRIPT = (
    "scripts/tools/verify_external_action_approval_packet.py"
)
EXTERNAL_ACTION_APPROVAL_TEMPLATE_RENDERER_SCRIPT = (
    "scripts/tools/render_external_action_approval_template.py"
)
WEAVE_CONTENT_CANARY_GATE_CONTRACT_SCRIPT = (
    "scripts/tools/weave_content_canary_gate_contract.py"
)
NEMOCLAW_CANARY_READINESS_SCRIPT = "scripts/tools/check_taiwan_canary_readiness.py"
NEMOCLAW_ADOPTION_SCRIPT = "scripts/tools/check_taiwan_nemoclaw_adoption.py"
NEMOCLAW_POST_INSTALL_SCRIPT = "scripts/setup/verify_nemoclaw_post_install.py"
AGENTIC_RUNNER_SCRIPT_ROLES = {
    "scripts/evaluator/agentic_math.py": "agentic_runner:math_evaluator_script",
    "scripts/evaluator/swebench_pro.py": "agentic_runner:swe_evaluator_script",
    "scripts/tools/run_openclaw_agent_protocol.py": "agentic_runner:protocol_script",
    "scripts/tools/run_agentic_math_openclaw.py": "agentic_runner:math_script",
    "scripts/tools/run_swebench_pro_openclaw.py": "agentic_runner:swe_script",
    "scripts/tools/run_taiwan_full_eval_batch.py": "agentic_runner:full_batch_script",
    "scripts/tools/verify_taiwan_weave_agents.py": "agentic_runner:weave_agents_verifier_script",
    "scripts/tools/verify_taiwan_wandb_completion.py": (
        "agentic_runner:wandb_completion_verifier_script"
    ),
    WEAVE_CONTENT_CANARY_GATE_CONTRACT_SCRIPT: (
        "agentic_runner:weave_content_canary_gate_contract_script"
    ),
    "scripts/tools/log_agentic_math_results_to_wandb.py": "agentic_runner:math_relog_script",
    "scripts/tools/log_agentic_swe_results_to_wandb.py": "agentic_runner:swe_relog_script",
    "scripts/tools/audit_taiwan_existing_results.py": (
        "agentic_runner:existing_results_audit_script"
    ),
}
BENCHMARK_REVIEW_PHASES = {
    "agentic_math": "canary_agentic",
    "agentic_swe": "canary_agentic",
    "taiwan_full": "canary_agentic_aggregate",
}
ENABLED_BENCHMARK_RELEASE_ALIASES = {
    "aggregate_taiwan": "taiwan_full",
    "swebench_pro": "agentic_swe",
}


def repo_path(path: Path | str) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    return REPO_ROOT / value


def path_display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def review_json_for_benchmark(review_json: str, benchmark: str) -> str:
    phase = BENCHMARK_REVIEW_PHASES.get(benchmark, "canary_full")
    return review_json.replace("PHASE", phase)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_bundle_path(source: Path) -> Path:
    source = source.resolve()
    try:
        relative = source.relative_to(REPO_ROOT)
    except ValueError:
        relative = Path("external") / source.name
    return Path("evidence") / relative


def add_evidence(
    evidence: dict[str, dict[str, Any]],
    *,
    role: str,
    path_value: Any,
) -> None:
    if not isinstance(path_value, str) or not path_value.strip():
        return
    if "://" in path_value:
        return
    path = repo_path(path_value)
    key = path_display(path)
    entry = evidence.setdefault(
        key,
        {
            "source_path": key,
            "roles": [],
            "exists": path.exists(),
        },
    )
    if role not in entry["roles"]:
        entry["roles"].append(role)


def add_existing_evidence(
    evidence: dict[str, dict[str, Any]],
    *,
    role: str,
    path_value: Any,
) -> None:
    if not isinstance(path_value, str) or not path_value.strip():
        return
    if "://" in path_value:
        return
    if repo_path(path_value).exists():
        add_evidence(evidence, role=role, path_value=path_value)


def add_json_referenced_paths(
    evidence: dict[str, dict[str, Any]],
    *,
    role: str,
    json_path_value: Any,
) -> None:
    if not isinstance(json_path_value, str) or not json_path_value.strip():
        return
    path = repo_path(json_path_value)
    if not path.exists() or not path.is_file():
        return
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return

    for field in ("setup_paths", "readiness_paths", "agentic_config_paths", "config_paths"):
        paths = payload.get(field)
        if isinstance(paths, list):
            for item in paths:
                add_evidence(evidence, role=f"{role}:{field}", path_value=item)
    for field in ("taxonomy_path", "manifest_path"):
        add_evidence(evidence, role=f"{role}:{field}", path_value=payload.get(field))
    pre_run_budget = payload.get("pre_run_budget_estimate")
    if isinstance(pre_run_budget, dict):
        add_evidence(
            evidence,
            role=f"{role}:pre_run_budget_estimate",
            path_value=pre_run_budget.get("path"),
        )
    external_action_approval = payload.get("external_action_approval")
    if isinstance(external_action_approval, dict):
        add_evidence(
            evidence,
            role=f"{role}:external_action_approval_report",
            path_value=external_action_approval.get("path"),
        )
    if payload.get("status") == "completed":
        preflights = payload.get("run_eval_preflights")
        if isinstance(preflights, list):
            for index, record in enumerate(preflights, start=1):
                if not isinstance(record, dict):
                    continue
                add_evidence(
                    evidence,
                    role=f"{role}:run_eval_preflight",
                    path_value=record.get("output_json"),
                )
                add_evidence(
                    evidence,
                    role=f"{role}:run_eval_preflight:{index}",
                    path_value=record.get("output_json"),
                )
        runs = payload.get("runs")
        if isinstance(runs, list):
            for index, record in enumerate(runs, start=1):
                if not isinstance(record, dict):
                    continue
                add_evidence(
                    evidence,
                    role=f"{role}:run:{index}:preflight_json",
                    path_value=record.get("preflight_json"),
                )
    operation_results = payload.get("operation_results")
    if isinstance(operation_results, dict):
        for name, record in operation_results.items():
            if not isinstance(record, dict):
                continue
            add_evidence(
                evidence,
                role=f"{role}:operation:{name}:log",
                path_value=record.get("log_path"),
            )
    criteria = payload.get("criteria")
    if isinstance(criteria, list):
        for row in criteria:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "criterion")
            paths = row.get("evidence_paths")
            if isinstance(paths, list):
                for item in paths:
                    add_evidence(evidence, role=f"{role}:{name}:evidence", path_value=item)


def add_post_install_verification_paths(
    evidence: dict[str, dict[str, Any]],
    *,
    role: str,
    json_path_value: Any,
) -> None:
    if not isinstance(json_path_value, str) or not json_path_value.strip():
        return
    path = repo_path(json_path_value)
    if not path.exists() or not path.is_file():
        return
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return
    outputs = payload.get("outputs")
    if isinstance(outputs, dict):
        for name, value in outputs.items():
            add_evidence(evidence, role=f"{role}:output:{name}", path_value=value)
            if name == "adoption_json":
                add_json_referenced_paths(
                    evidence,
                    role=f"{role}:adoption_json",
                    json_path_value=value,
                )
    steps = payload.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            name = str(step.get("name") or "step")
            add_evidence(evidence, role=f"{role}:step:{name}", path_value=step.get("output_json"))


def add_weave_content_canary_gate_paths(
    evidence: dict[str, dict[str, Any]],
    *,
    role: str,
    json_path_value: Any,
) -> None:
    if not isinstance(json_path_value, str) or not json_path_value.strip():
        return
    path = repo_path(json_path_value)
    if not path.exists() or not path.is_file():
        return
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return
    if payload.get("gate") != "weave_agents_content_canary":
        return
    paths = payload.get("paths")
    if not isinstance(paths, dict):
        return
    for field in (
        "plan_file",
        "command_result_file",
        "verifier_json",
        "agents_diagnostic_json",
        "expected_sidecar",
        "prompt_file",
    ):
        add_existing_evidence(evidence, role=f"{role}:{field}", path_value=paths.get(field))


def add_wandb_adoption_draft_paths(
    evidence: dict[str, dict[str, Any]],
    *,
    role: str,
    json_path_value: Any,
) -> None:
    if not isinstance(json_path_value, str) or not json_path_value.strip():
        return
    path = repo_path(json_path_value)
    if not path.exists() or not path.is_file():
        return
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return

    source_audit = payload.get("source_audit_json")
    add_evidence(evidence, role=f"{role}:source_audit_json", path_value=source_audit)

    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        for index, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, dict):
                continue
            benchmark = str(candidate.get("benchmark") or f"candidate_{index}")
            candidate_role = f"{role}:candidate:{benchmark}"
            completion_path = candidate.get("wandb_completion_json")
            add_evidence(
                evidence,
                role=f"{candidate_role}:wandb_completion_json",
                path_value=completion_path,
            )
            add_json_referenced_paths(
                evidence,
                role=f"{candidate_role}:wandb_completion_json",
                json_path_value=completion_path,
            )
            add_existing_evidence(
                evidence,
                role=f"{candidate_role}:target_review_json",
                path_value=candidate.get("target_review_json"),
            )
            add_evidence(
                evidence,
                role=f"{candidate_role}:scope_attestation_template_json",
                path_value=candidate.get("scope_attestation_template_json"),
            )
            add_existing_evidence(
                evidence,
                role=f"{candidate_role}:scope_attestation_render_report_json",
                path_value=candidate.get("scope_attestation_render_report_json"),
            )
            add_existing_evidence(
                evidence,
                role=f"{candidate_role}:scope_attestation_render_markdown",
                path_value=candidate.get("scope_attestation_render_markdown"),
            )
            add_existing_evidence(
                evidence,
                role=f"{candidate_role}:scope_attestation_preflight_report_json",
                path_value=candidate.get("scope_attestation_preflight_report_json"),
            )
            add_existing_evidence(
                evidence,
                role=f"{candidate_role}:sync_dry_run_report_json",
                path_value=candidate.get("sync_dry_run_report_json"),
            )


def add_existing_results_audit_paths(
    evidence: dict[str, dict[str, Any]],
    *,
    role: str,
    json_path_value: Any,
) -> None:
    if not isinstance(json_path_value, str) or not json_path_value.strip():
        return
    path = repo_path(json_path_value)
    if not path.exists() or not path.is_file():
        return
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return

    add_existing_evidence(
        evidence,
        role=f"{role}:archive_manifest",
        path_value=payload.get("archive_manifest_path"),
    )
    for field in ("formalized_records", "archived_complete_records", "unformalized_complete_records", "partial_records", "records"):
        records = payload.get(field)
        if not isinstance(records, list):
            continue
        for index, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                continue
            plan_path = record.get("relog_dry_run_plan_json")
            if not isinstance(plan_path, str) or not plan_path.strip():
                continue
            benchmark = str(record.get("benchmark") or f"record_{index}")
            model_slug = str(record.get("model_slug") or f"model_{index}")
            add_existing_evidence(
                evidence,
                role=(
                    f"{role}:record:{benchmark}:{model_slug}:"
                    "relog_dry_run_plan_json"
                ),
                path_value=plan_path,
            )
            record_role_prefix = f"{role}:record:{benchmark}:{model_slug}"
            for command_field in ("relog_dry_run_command", "relog_command"):
                command = record.get(command_field)
                if not isinstance(command, str) or not command.strip():
                    continue
                for script_path in command_script_paths(command):
                    add_evidence(
                        evidence,
                        role=f"{role}:relog_command_script",
                        path_value=script_path,
                    )
                    add_evidence(
                        evidence,
                        role=f"{record_role_prefix}:relog_command_script",
                        path_value=script_path,
                    )
                    if script_path in RELOG_COMMAND_SCRIPT_PATHS:
                        add_evidence(
                            evidence,
                            role=f"{role}:relog_dependency_script",
                            path_value=RELOG_WANDB_APPROVAL_HELPER_SCRIPT,
                        )
                        add_evidence(
                            evidence,
                            role=f"{record_role_prefix}:relog_dependency_script",
                            path_value=RELOG_WANDB_APPROVAL_HELPER_SCRIPT,
                        )


def add_scope_attestation_source_path(
    evidence: dict[str, dict[str, Any]],
    *,
    role: str,
    entry: dict[str, Any],
) -> None:
    scope = entry.get("scope_attestation")
    if not isinstance(scope, dict):
        return
    benchmark = str(entry.get("benchmark") or "unknown_benchmark")
    add_evidence(
        evidence,
        role=f"{role}:{benchmark}:source_attestation_json",
        path_value=scope.get("source_attestation_json"),
    )
    add_evidence(
        evidence,
        role=f"{role}:{benchmark}:source_audit_json",
        path_value=scope.get("source_audit_json"),
    )


def add_paid_review_check_paths(
    evidence: dict[str, dict[str, Any]],
    *,
    role: str,
    json_path_value: Any,
) -> None:
    if not isinstance(json_path_value, str) or not json_path_value.strip():
        return
    path = repo_path(json_path_value)
    if not path.exists() or not path.is_file():
        return
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return

    gates = payload.get("gates")
    if not isinstance(gates, list):
        return
    for gate in gates:
        if not isinstance(gate, dict):
            continue
        gate_name = str(gate.get("name") or "unknown_gate")
        records = gate.get("records")
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            add_evidence(
                evidence,
                role=f"{role}:gate:{gate_name}:review",
                path_value=record.get("path"),
            )
            add_json_referenced_paths(
                evidence,
                role=f"{role}:gate:{gate_name}:review",
                json_path_value=record.get("path"),
            )
            budget = record.get("pre_run_budget_estimate")
            if isinstance(budget, dict):
                add_evidence(
                    evidence,
                    role=f"{role}:gate:{gate_name}:pre_run_budget_estimate",
                    path_value=budget.get("path"),
                )
            approval = record.get("external_action_approval")
            if isinstance(approval, dict):
                add_evidence(
                    evidence,
                    role=f"{role}:gate:{gate_name}:external_action_approval_report",
                    path_value=approval.get("path"),
                )
            entries = record.get("wandb_completion_entries")
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                entry_role = f"{role}:gate:{gate_name}:wandb_completion"
                add_evidence(evidence, role=entry_role, path_value=entry.get("path"))
                add_json_referenced_paths(
                    evidence,
                    role=entry_role,
                    json_path_value=entry.get("path"),
                )
                add_scope_attestation_source_path(
                    evidence,
                    role=entry_role,
                    entry=entry,
                )


def command_script_paths(command: str) -> list[str]:
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    scripts: list[str] = []
    seen: set[str] = set()
    for part in parts:
        token = part.strip().strip("`'\".,;:()[]{}")
        if token.startswith("./scripts/"):
            token = token[2:]
        if not token.startswith("scripts/"):
            continue
        if not token.endswith(COMMAND_SCRIPT_SUFFIXES):
            continue
        if token in seen:
            continue
        seen.add(token)
        scripts.append(token)
    return scripts


def add_operator_command_script_evidence(
    evidence: dict[str, dict[str, Any]],
    *,
    operator_next_steps: Any,
) -> None:
    if not isinstance(operator_next_steps, dict):
        return
    steps = operator_next_steps.get("steps")
    if not isinstance(steps, list):
        return
    for step in steps:
        if not isinstance(step, dict):
            continue
        commands = step.get("commands")
        if not isinstance(commands, list):
            continue
        for command in commands:
            if not isinstance(command, str) or not command.strip():
                continue
            for script_path in command_script_paths(command):
                add_evidence(
                    evidence,
                    role="operator_plan:command_script",
                    path_value=script_path,
                )


def add_remediation_command_script_evidence(
    evidence: dict[str, dict[str, Any]],
    *,
    remediation_plan: Any,
) -> None:
    if not isinstance(remediation_plan, list):
        return
    for row in remediation_plan:
        if not isinstance(row, dict):
            continue
        commands = row.get("commands")
        if not isinstance(commands, list):
            continue
        for command in commands:
            if not isinstance(command, str) or not command.strip():
                continue
            for script_path in command_script_paths(command):
                add_evidence(
                    evidence,
                    role="current_gate:remediation_plan:command_script",
                    path_value=script_path,
                )


def add_agentic_runner_script_evidence(
    evidence: dict[str, dict[str, Any]],
) -> None:
    for script_path, role in AGENTIC_RUNNER_SCRIPT_ROLES.items():
        add_evidence(evidence, role="agentic_runner:script", path_value=script_path)
        add_evidence(evidence, role=role, path_value=script_path)


def collect_evidence(report_path: Path, report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    add_evidence(evidence, role="production_readiness_report", path_value=str(report_path))

    runner = report.get("runner")
    if isinstance(runner, dict):
        add_evidence(evidence, role="runner_report_json", path_value=runner.get("report_json"))
        nemoclaw_check = runner.get("nemoclaw_check")
        if isinstance(nemoclaw_check, dict):
            add_evidence(evidence, role="nemoclaw_setup_check", path_value=nemoclaw_check.get("path"))
        existing_results = runner.get("existing_results_audit")
        if isinstance(existing_results, dict):
            add_evidence(evidence, role="existing_results_audit", path_value=existing_results.get("path"))
            add_evidence(evidence, role="existing_results_audit_markdown", path_value=existing_results.get("markdown_path"))
            add_existing_results_audit_paths(
                evidence,
                role="existing_results_audit",
                json_path_value=existing_results.get("path"),
            )
        adoption_draft = runner.get("wandb_adoption_draft")
        if isinstance(adoption_draft, dict):
            add_evidence(evidence, role="wandb_adoption_draft", path_value=adoption_draft.get("path"))
            add_evidence(evidence, role="wandb_adoption_draft_markdown", path_value=adoption_draft.get("markdown_path"))
            add_wandb_adoption_draft_paths(
                evidence,
                role="wandb_adoption_draft",
                json_path_value=adoption_draft.get("path"),
            )
        adoption_unconfirmed = runner.get("wandb_adoption_unconfirmed_checks")
        if isinstance(adoption_unconfirmed, dict):
            records = adoption_unconfirmed.get("records")
            if isinstance(records, list):
                for record in records:
                    if not isinstance(record, dict):
                        continue
                    benchmark = str(record.get("benchmark") or "unknown_benchmark")
                    run_id = str(record.get("run_id") or "unknown_run")
                    role_prefix = (
                        f"wandb_adoption_unconfirmed_checks:{benchmark}:{run_id}"
                    )
                    add_evidence(
                        evidence,
                        role=f"{role_prefix}:scope_attestation_template_json",
                        path_value=record.get("scope_attestation_template_json"),
                    )
                    add_evidence(
                        evidence,
                        role=f"{role_prefix}:preflight_report_json",
                        path_value=record.get("preflight_report_json"),
                    )
                    add_evidence(
                        evidence,
                        role=f"{role_prefix}:sync_dry_run_report_json",
                        path_value=record.get("sync_dry_run_report_json"),
                    )
        weave_adoption_failures = runner.get("weave_agents_adoption_validation_failures")
        if isinstance(weave_adoption_failures, dict):
            records = weave_adoption_failures.get("records")
            if isinstance(records, list):
                for index, record in enumerate(records, start=1):
                    if not isinstance(record, dict):
                        continue
                    role_prefix = f"weave_agents_adoption_validation_failure:{index}"
                    add_evidence(
                        evidence,
                        role=f"{role_prefix}:report_json",
                        path_value=record.get("path"),
                    )
                    completion_paths = record.get("completion_paths")
                    if isinstance(completion_paths, list):
                        for path in completion_paths:
                            add_evidence(
                                evidence,
                                role=f"{role_prefix}:completion_json",
                                path_value=path,
                            )
        operator_docs = runner.get("nemoclaw_operator_docs_verification")
        if isinstance(operator_docs, dict):
            add_evidence(evidence, role="nemoclaw_operator_docs_verification", path_value=operator_docs.get("path"))
            add_evidence(evidence, role="nemoclaw_operator_docs_verification_markdown", path_value=operator_docs.get("markdown_path"))
        paid_review = runner.get("paid_run_review_check")
        if isinstance(paid_review, dict):
            add_evidence(evidence, role="paid_run_review_check", path_value=paid_review.get("path"))
            add_evidence(evidence, role="paid_run_review_check_markdown", path_value=paid_review.get("markdown_path"))
            add_paid_review_check_paths(
                evidence,
                role="paid_run_review_check",
                json_path_value=paid_review.get("path"),
            )
        nemoclaw_adoption = runner.get("nemoclaw_adoption_check")
        if isinstance(nemoclaw_adoption, dict):
            add_evidence(evidence, role="nemoclaw_adoption_check", path_value=nemoclaw_adoption.get("path"))
            add_evidence(evidence, role="nemoclaw_adoption_check_markdown", path_value=nemoclaw_adoption.get("markdown_path"))
            add_evidence(
                evidence,
                role="nemoclaw_adoption_check:script",
                path_value=NEMOCLAW_ADOPTION_SCRIPT,
            )
            add_json_referenced_paths(
                evidence,
                role="nemoclaw_adoption_check",
                json_path_value=nemoclaw_adoption.get("path"),
            )
        post_install = runner.get("nemoclaw_post_install_verification")
        if isinstance(post_install, dict):
            add_evidence(evidence, role="nemoclaw_post_install_verification", path_value=post_install.get("path"))
            add_evidence(evidence, role="nemoclaw_post_install_verification_markdown", path_value=post_install.get("markdown_path"))
            add_evidence(
                evidence,
                role="nemoclaw_post_install_verification:script",
                path_value=NEMOCLAW_POST_INSTALL_SCRIPT,
            )
            add_evidence(
                evidence,
                role="nemoclaw_post_install_verification:canary_readiness_script",
                path_value=NEMOCLAW_CANARY_READINESS_SCRIPT,
            )
            add_evidence(
                evidence,
                role="nemoclaw_post_install_verification:adoption_script",
                path_value=NEMOCLAW_ADOPTION_SCRIPT,
            )
            add_post_install_verification_paths(
                evidence,
                role="nemoclaw_post_install_verification",
                json_path_value=post_install.get("path"),
            )

    gates = report.get("gates")
    if isinstance(gates, list):
        for gate in gates:
            if not isinstance(gate, dict):
                continue
            gate_name = str(gate.get("name") or "unknown_gate")
            paths = gate.get("evidence_paths")
            if isinstance(paths, list):
                for path in paths:
                    add_evidence(evidence, role=f"gate:{gate_name}:evidence", path_value=path)
                    add_json_referenced_paths(
                        evidence,
                        role=f"gate:{gate_name}:evidence",
                        json_path_value=path,
                    )
                    if gate_name == "weave_content_canary":
                        add_weave_content_canary_gate_paths(
                            evidence,
                            role=f"gate:{gate_name}:evidence",
                            json_path_value=path,
                        )
            latest_setup = gate.get("latest_setup_report")
            if isinstance(latest_setup, dict):
                add_evidence(evidence, role=f"gate:{gate_name}:latest_setup_report", path_value=latest_setup.get("path"))
            latest_installer_review = gate.get("latest_installer_review")
            if isinstance(latest_installer_review, dict):
                review_path = latest_installer_review.get("path")
                add_evidence(
                    evidence,
                    role=f"gate:{gate_name}:latest_installer_review_json",
                    path_value=review_path,
                )
                if isinstance(review_path, str) and review_path.endswith(".json"):
                    add_existing_evidence(
                        evidence,
                        role=f"gate:{gate_name}:latest_installer_review_markdown",
                        path_value=review_path[:-5] + ".md",
                    )
                add_existing_evidence(
                    evidence,
                    role=f"gate:{gate_name}:latest_installer_review_lock_json",
                    path_value=latest_installer_review.get("lock_json"),
                )
            records = gate.get("records")
            if isinstance(records, list):
                for record in records:
                    if not isinstance(record, dict):
                        continue
                    add_evidence(evidence, role=f"gate:{gate_name}:record", path_value=record.get("path"))
                    add_json_referenced_paths(
                        evidence,
                        role=f"gate:{gate_name}:record",
                        json_path_value=record.get("path"),
                    )
                    entries = record.get("wandb_completion_entries")
                    if isinstance(entries, list):
                        for entry in entries:
                            if not isinstance(entry, dict):
                                continue
                            add_evidence(evidence, role=f"gate:{gate_name}:wandb_completion", path_value=entry.get("path"))
                            add_json_referenced_paths(
                                evidence,
                                role=f"gate:{gate_name}:wandb_completion",
                                json_path_value=entry.get("path"),
                            )
                            add_evidence(evidence, role=f"gate:{gate_name}:review", path_value=entry.get("review_path"))
                            add_scope_attestation_source_path(
                                evidence,
                                role=f"gate:{gate_name}:wandb_completion",
                                entry=entry,
                            )
                    weave_entries = record.get("weave_agents_completion_entries")
                    if isinstance(weave_entries, list):
                        for entry in weave_entries:
                            if not isinstance(entry, dict):
                                continue
                            add_evidence(evidence, role=f"gate:{gate_name}:weave_agents_completion", path_value=entry.get("path"))
                            add_json_referenced_paths(
                                evidence,
                                role=f"gate:{gate_name}:weave_agents_completion",
                                json_path_value=entry.get("path"),
                            )
                            add_evidence(
                                evidence,
                                role=f"gate:{gate_name}:weave_agents_completion_sync_dry_run",
                                path_value=entry.get("sync_dry_run_report_json"),
                            )
                            add_evidence(
                                evidence,
                                role=f"gate:{gate_name}:weave_agents_completion_sync_source_review",
                                path_value=entry.get("sync_dry_run_source_review_json"),
                            )
                            add_evidence(evidence, role=f"gate:{gate_name}:review", path_value=entry.get("review_path"))

    summary = report.get("summary")
    benchmark_evidence = summary.get("benchmark_evidence") if isinstance(summary, dict) else None
    if isinstance(benchmark_evidence, list):
        for row in benchmark_evidence:
            if not isinstance(row, dict):
                continue
            benchmark = str(row.get("benchmark") or "unknown_benchmark")
            standalone = row.get("standalone_wandb_completion")
            if isinstance(standalone, dict):
                paths = standalone.get("evidence_paths")
                if isinstance(paths, list):
                    for path in paths:
                        add_evidence(evidence, role=f"benchmark:{benchmark}:standalone_wandb_completion", path_value=path)
                        add_json_referenced_paths(
                            evidence,
                            role=f"benchmark:{benchmark}:standalone_wandb_completion",
                            json_path_value=path,
                        )
            review = row.get("review_wandb_completion")
            if isinstance(review, dict):
                paths = review.get("review_paths")
                if isinstance(paths, list):
                    for path in paths:
                        add_evidence(evidence, role=f"benchmark:{benchmark}:review_wandb_completion", path_value=path)
                entries = review.get("entries")
                if isinstance(entries, list):
                    for entry in entries:
                        if not isinstance(entry, dict):
                            continue
                        add_evidence(evidence, role=f"benchmark:{benchmark}:review_wandb_completion", path_value=entry.get("path"))
                        add_json_referenced_paths(
                            evidence,
                            role=f"benchmark:{benchmark}:review_wandb_completion",
                            json_path_value=entry.get("path"),
                        )
                        add_evidence(evidence, role=f"benchmark:{benchmark}:review_wandb_completion", path_value=entry.get("review_path"))
    return evidence


def copy_evidence_files(
    evidence: dict[str, dict[str, Any]],
    *,
    output_dir: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source_display in sorted(evidence):
        source = repo_path(source_display)
        record = dict(evidence[source_display])
        record["roles"] = sorted(record["roles"])
        record["exists"] = source.exists()
        if source.exists() and source.is_file():
            bundle_relative = safe_bundle_path(source)
            destination = output_dir / bundle_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            record.update(
                {
                    "bundle_path": str(bundle_relative),
                    "size_bytes": source.stat().st_size,
                    "sha256": sha256_file(source),
                }
            )
        else:
            record.update(
                {
                    "bundle_path": "",
                    "size_bytes": None,
                    "sha256": "",
                    "missing_reason": "not found" if not source.exists() else "not a file",
                }
            )
        records.append(record)
    return records


def generated_bundle_file_record(
    path: Path,
    *,
    bundle_path: Path,
    roles: list[str],
) -> dict[str, Any]:
    return {
        "source_path": path_display(path),
        "roles": sorted(roles),
        "exists": path.exists() and path.is_file(),
        "bundle_path": str(bundle_path),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
    }


def gate_summary(report: dict[str, Any]) -> list[dict[str, Any]]:
    gates = report.get("gates")
    if not isinstance(gates, list):
        return []
    result: list[dict[str, Any]] = []
    for gate in gates:
        if not isinstance(gate, dict):
            continue
        result.append(
            {
                "name": gate.get("name"),
                "ok": bool(gate.get("ok")),
                "status": gate.get("status"),
                "blocking": bool(gate.get("blocking")),
                "detail": gate.get("detail"),
                "next_action": gate.get("next_action"),
                "evidence_paths": gate.get("evidence_paths") if isinstance(gate.get("evidence_paths"), list) else [],
                "latest_installer_review": (
                    gate.get("latest_installer_review")
                    if isinstance(gate.get("latest_installer_review"), dict)
                    else {}
                ),
            }
        )
    return result


def benchmark_completion_summary(report: dict[str, Any]) -> list[dict[str, Any]]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    evidence = summary.get("benchmark_evidence")
    if not isinstance(evidence, list):
        return []

    requirements = report.get("requirements") if isinstance(report.get("requirements"), dict) else {}
    required_benchmarks = set(
        str(item)
        for item in requirements.get("required_wandb_benchmarks", [])
        if isinstance(item, str) and item.strip()
    )
    result: list[dict[str, Any]] = []
    for row in evidence:
        if not isinstance(row, dict):
            continue
        benchmark = row.get("benchmark")
        required = row.get("required")
        if not isinstance(required, bool):
            required = (
                str(benchmark) in required_benchmarks
                if benchmark and required_benchmarks
                else bool(benchmark)
            )
        standalone = row.get("standalone_wandb_completion")
        review = row.get("review_wandb_completion")
        standalone_records = []
        if isinstance(standalone, dict) and isinstance(standalone.get("records"), list):
            for record in standalone["records"]:
                if not isinstance(record, dict):
                    continue
                standalone_records.append(
                    {
                        "path": record.get("path"),
                        "ok": bool(record.get("ok")),
                        "run_id": record.get("run_id"),
                        "schema_valid": bool(record.get("schema_valid")),
                        "observed_evidence_valid": bool(record.get("observed_evidence_valid")),
                        "fresh": bool(record.get("fresh")),
                    }
                )
        review_entries = []
        if isinstance(review, dict) and isinstance(review.get("entries"), list):
            for entry in review["entries"]:
                if not isinstance(entry, dict):
                    continue
                review_entries.append(
                    {
                        "path": entry.get("path"),
                        "review_path": entry.get("review_path"),
                        "ok": bool(entry.get("ok")),
                        "run_id": entry.get("run_id"),
                        "schema_valid": bool(entry.get("schema_valid")),
                        "observed_evidence_valid": bool(entry.get("observed_evidence_valid")),
                    }
                )
        result.append(
            {
                "benchmark": benchmark,
                "required": required,
                "expected_run_id": row.get("expected_run_id"),
                "completion_proven": bool(row.get("completion_proven")),
                "standalone_status": standalone.get("status") if isinstance(standalone, dict) else None,
                "standalone_ok": bool(standalone.get("ok")) if isinstance(standalone, dict) else False,
                "standalone_records": standalone_records,
                "review_status": review.get("status") if isinstance(review, dict) else None,
                "review_ok": bool(review.get("ok")) if isinstance(review, dict) else False,
                "review_entries": review_entries,
            }
        )
    return result


def weave_agents_completion_summary(report: dict[str, Any]) -> list[dict[str, Any]]:
    gates = report.get("gates")
    if not isinstance(gates, list):
        return []

    result: list[dict[str, Any]] = []
    for gate in gates:
        if not isinstance(gate, dict):
            continue
        gate_name = gate.get("name")
        records = gate.get("records")
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            raw_entries = record.get("weave_agents_completion_entries")
            entries: list[dict[str, Any]] = []
            if isinstance(raw_entries, list):
                for entry in raw_entries:
                    if not isinstance(entry, dict):
                        continue
                    entries.append(
                        {
                            "path": entry.get("path"),
                            "review_path": entry.get("review_path") or record.get("path"),
                            "agent_name": entry.get("agent_name"),
                            "run_id": entry.get("run_id"),
                            "ok": bool(entry.get("ok")),
                            "entry_ok": bool(entry.get("entry_ok")),
                            "verified": bool(entry.get("verified")),
                            "schema_valid": bool(entry.get("schema_valid")),
                            "checks_valid": bool(entry.get("checks_valid")),
                            "trace_present": bool(entry.get("trace_present")),
                            "fresh": bool(entry.get("fresh")),
                            "latest_trace_id": entry.get("latest_trace_id"),
                            "sync_dry_run_report_json": entry.get("sync_dry_run_report_json"),
                            "sync_dry_run_source_review_json": entry.get(
                                "sync_dry_run_source_review_json"
                            ),
                            "sync_dry_run_source_review_sha256": entry.get(
                                "sync_dry_run_source_review_sha256"
                            ),
                            "verification_error": entry.get("verification_error"),
                        }
                    )

            required = bool(record.get("verify_weave_agents"))
            if not required and not entries:
                continue
            verified_count = sum(1 for entry in entries if entry.get("verified"))
            result.append(
                {
                    "gate": gate_name,
                    "review_path": record.get("path"),
                    "phase": record.get("phase"),
                    "record_status": record.get("status"),
                    "required": required,
                    "entry_count": len(entries),
                    "verified_count": verified_count,
                    "completion_proven": bool(entries) and verified_count == len(entries),
                    "max_age_seconds": record.get("weave_agents_completion_max_age_seconds"),
                    "entries": entries,
                }
            )
    return result


def compact_existing_result_record(record: dict[str, Any]) -> dict[str, Any]:
    wandb_completion = record.get("wandb_completion")
    completion_summary = {}
    if isinstance(wandb_completion, dict):
        completion_summary = {
            "path": wandb_completion.get("path"),
            "run_id": wandb_completion.get("run_id"),
            "run_name": wandb_completion.get("run_name"),
            "generated_at": wandb_completion.get("generated_at"),
            "verification_schema_version": wandb_completion.get("verification_schema_version"),
            "schema_current": wandb_completion.get("schema_current"),
            "observed_evidence_present": wandb_completion.get("observed_evidence_present"),
        }
    return {
        "benchmark": record.get("benchmark"),
        "model_slug": record.get("model_slug"),
        "model": record.get("model"),
        "run_kind": record.get("run_kind"),
        "result_dir": record.get("result_dir"),
        "summary_path": record.get("summary_path"),
        "results_path": record.get("results_path"),
        "official_summary_path": record.get("official_summary_path"),
        "patches_path": record.get("patches_path"),
        "complete_local": bool(record.get("complete_local")),
        "formalization_status": record.get("formalization_status"),
        "expected_total": record.get("expected_total"),
        "row_count": record.get("row_count"),
        "partial_row_count": record.get("partial_row_count"),
        "patch_count": record.get("patch_count"),
        "patch_record_count": record.get("patch_record_count"),
        "wandb_completion": completion_summary,
        "warnings": record.get("warnings") if isinstance(record.get("warnings"), list) else [],
        "errors": record.get("errors") if isinstance(record.get("errors"), list) else [],
        "source_sha256s": record.get("source_sha256s") if isinstance(record.get("source_sha256s"), dict) else {},
        "archived_existing_result": bool(record.get("archived_existing_result")),
        "archive_manifest_path": record.get("archive_manifest_path"),
        "archive_manifest_entry": (
            record.get("archive_manifest_entry")
            if isinstance(record.get("archive_manifest_entry"), dict)
            else {}
        ),
        "relog_dry_run_plan_json": record.get("relog_dry_run_plan_json"),
        "relog_dry_run_command": record.get("relog_dry_run_command"),
        "relog_command": record.get("relog_command"),
        "verify_command": record.get("verify_command"),
    }


def existing_results_formalization_summary(report: dict[str, Any]) -> dict[str, Any]:
    runner = report.get("runner")
    if not isinstance(runner, dict):
        return {}
    audit = runner.get("existing_results_audit")
    if not isinstance(audit, dict):
        return {}
    path_value = audit.get("path")
    result: dict[str, Any] = {
        "path": path_value,
        "markdown_path": audit.get("markdown_path"),
        "skipped": bool(audit.get("skipped")),
        "ok": bool(audit.get("ok")),
        "status": audit.get("status"),
        "summary": audit.get("summary") if isinstance(audit.get("summary"), dict) else {},
        "formalized_records": [],
        "archived_complete_records": [],
        "unformalized_complete_records": [],
        "partial_records": [],
        "wandb_completion_records": [],
        "remediation_commands": [],
    }
    if result["skipped"] or not isinstance(path_value, str) or not path_value.strip():
        return result
    path = repo_path(path_value)
    if not path.exists() or not path.is_file():
        result["read_error"] = "missing existing-results audit JSON"
        return result
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        result["read_error"] = str(exc)
        return result

    result.update(
        {
            "ok": bool(payload.get("ok")),
            "status": payload.get("status"),
            "generated_at": payload.get("generated_at"),
            "output_root": payload.get("output_root"),
            "completion_dir": payload.get("completion_dir"),
            "summary": payload.get("summary") if isinstance(payload.get("summary"), dict) else {},
            "remediation_commands": (
                payload.get("remediation_commands")
                if isinstance(payload.get("remediation_commands"), list)
                else []
            ),
        }
    )
    for field in ("formalized_records", "archived_complete_records", "unformalized_complete_records", "partial_records"):
        records = payload.get(field)
        if isinstance(records, list):
            result[field] = [
                compact_existing_result_record(record)
                for record in records
                if isinstance(record, dict)
            ]
    completions = payload.get("wandb_completion_records")
    if isinstance(completions, list):
        result["wandb_completion_records"] = [
            {
                "path": record.get("path"),
                "ok": bool(record.get("ok")),
                "benchmark": record.get("benchmark"),
                "run_id": record.get("run_id"),
                "run_name": record.get("run_name"),
                "verification_schema_version": record.get("verification_schema_version"),
                "schema_current": record.get("schema_current"),
                "observed_evidence_present": record.get("observed_evidence_present"),
                "metrics": record.get("metrics") if isinstance(record.get("metrics"), dict) else {},
                "error": record.get("error"),
            }
            for record in completions
            if isinstance(record, dict)
        ]
    return result


def wandb_adoption_draft_summary(report: dict[str, Any]) -> dict[str, Any]:
    runner = report.get("runner")
    if not isinstance(runner, dict):
        return {}
    draft = runner.get("wandb_adoption_draft")
    if not isinstance(draft, dict):
        return {}
    path_value = draft.get("path")
    result: dict[str, Any] = {
        "path": path_value,
        "markdown_path": draft.get("markdown_path"),
        "skipped": bool(draft.get("skipped")),
        "ok": draft.get("ok"),
        "status": draft.get("status"),
        "summary": draft.get("summary") if isinstance(draft.get("summary"), dict) else {},
        "candidate_count": 0,
        "requires_human_scope_confirmation": False,
        "candidates": [],
    }
    if result["skipped"] or not isinstance(path_value, str) or not path_value.strip():
        return result
    path = repo_path(path_value)
    if not path.exists() or not path.is_file():
        result["read_error"] = "missing W&B adoption draft JSON"
        return result
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        result["read_error"] = str(exc)
        return result
    candidates = payload.get("candidates")
    result.update(
        {
            "ok": bool(payload.get("ok")),
            "status": payload.get("status"),
            "generated_at": payload.get("generated_at"),
            "source_audit_json": payload.get("source_audit_json"),
            "source_audit_sha256": payload.get("source_audit_sha256"),
            "candidate_count": payload.get("candidate_count"),
            "scope_attestation_template_count": payload.get("scope_attestation_template_count"),
            "requires_human_scope_confirmation": bool(
                payload.get("requires_human_scope_confirmation")
            ),
            "scope_attestation_templates": (
                payload.get("scope_attestation_templates")
                if isinstance(payload.get("scope_attestation_templates"), list)
                else []
            ),
            "required_human_fields": (
                payload.get("required_human_fields")
                if isinstance(payload.get("required_human_fields"), list)
                else []
            ),
            "candidates": [
                {
                    "benchmark": row.get("benchmark"),
                    "model_slug": row.get("model_slug"),
                    "model": row.get("model"),
                    "metrics": row.get("metrics") if isinstance(row.get("metrics"), dict) else {},
                    "wandb_entity": row.get("wandb_entity"),
                    "wandb_project": row.get("wandb_project"),
                    "wandb_run_id": row.get("wandb_run_id"),
                    "wandb_run_name": row.get("wandb_run_name"),
                    "wandb_completion_json": row.get("wandb_completion_json"),
                    "source_audit_json": row.get("source_audit_json"),
                    "source_audit_sha256": row.get("source_audit_sha256"),
                    "target_review_json": row.get("target_review_json"),
                    "scope_attestation_template_json": row.get("scope_attestation_template_json"),
                    "scope_attestation_required": bool(row.get("scope_attestation_required")),
                    "required_human_fields": (
                        row.get("required_human_fields")
                        if isinstance(row.get("required_human_fields"), list)
                        else []
                    ),
                    "pending_scope_confirmation": bool(
                        row.get("scope_attestation_required")
                    ),
                    "pending_human_field_count": (
                        len(row.get("required_human_fields"))
                        if isinstance(row.get("required_human_fields"), list)
                        else 0
                    ),
                    "run_metadata_valid": row.get("run_metadata_valid"),
                    "run_metadata_errors": (
                        row.get("run_metadata_errors")
                        if isinstance(row.get("run_metadata_errors"), list)
                        else []
                    ),
                    "sync_ready": row.get("sync_ready"),
                    "scope_attestation_render_report_json": row.get(
                        "scope_attestation_render_report_json"
                    ),
                    "scope_attestation_render_markdown": row.get(
                        "scope_attestation_render_markdown"
                    ),
                    "scope_attestation_render_command": row.get(
                        "scope_attestation_render_command"
                    ),
                    "scope_attestation_preflight_report_json": row.get(
                        "scope_attestation_preflight_report_json"
                    ),
                    "scope_attestation_preflight_command": row.get(
                        "scope_attestation_preflight_command"
                    ),
                    "sync_dry_run_report_json": row.get("sync_dry_run_report_json"),
                    "sync_dry_run_command": row.get("sync_dry_run_command"),
                    "sync_apply_command": row.get("sync_apply_command"),
                    "sync_command": row.get("sync_command"),
                    "refresh_wandb_completion_command": row.get(
                        "refresh_wandb_completion_command"
                    ),
                    "sync_command_blocked_reason": row.get("sync_command_blocked_reason"),
                    "operator_handoff": (
                        row.get("operator_handoff")
                        if isinstance(row.get("operator_handoff"), dict)
                        else {}
                    ),
                    "warnings": row.get("warnings") if isinstance(row.get("warnings"), list) else [],
                }
                for row in candidates
                if isinstance(row, dict)
            ]
            if isinstance(candidates, list)
            else [],
        }
    )
    summary_candidates = (
        result.get("candidates") if isinstance(result.get("candidates"), list) else []
    )
    pending_scope_candidates = [
        row
        for row in summary_candidates
        if isinstance(row, dict) and bool(row.get("pending_scope_confirmation"))
    ]
    draft_required_human_fields = (
        result.get("required_human_fields")
        if isinstance(result.get("required_human_fields"), list)
        else []
    )
    pending_human_fields: list[str] = []
    pending_human_field_count = 0
    for row in pending_scope_candidates:
        row_fields = (
            row.get("required_human_fields")
            if isinstance(row.get("required_human_fields"), list)
            else []
        )
        fields = row_fields if row_fields else draft_required_human_fields
        normalized_fields = [
            field for field in fields if isinstance(field, str) and field.strip()
        ]
        pending_human_field_count += len(normalized_fields)
        for field in normalized_fields:
            if field not in pending_human_fields:
                pending_human_fields.append(field)
    result["pending_scope_confirmation_candidate_count"] = len(
        pending_scope_candidates
    )
    result["pending_human_field_count"] = pending_human_field_count
    result["pending_human_fields"] = pending_human_fields
    operator_handoff = (
        payload.get("operator_handoff")
        if isinstance(payload.get("operator_handoff"), dict)
        else {}
    )
    if operator_handoff:
        result["operator_handoff"] = operator_handoff
        result["operator_handoff_candidate_count"] = operator_handoff.get("candidate_count")
        result["operator_handoff_available_candidate_count"] = operator_handoff.get(
            "available_candidate_count"
        )
        result["operator_handoff_step_count"] = operator_handoff.get("step_count")
        result["operator_handoff_command_count"] = operator_handoff.get("command_count")
        result["operator_handoff_external_action_step_count"] = operator_handoff.get(
            "external_action_step_count"
        )
        result["operator_handoff_scope_confirmation_step_count"] = operator_handoff.get(
            "scope_confirmation_step_count"
        )
        result["operator_handoff_review_mutation_step_count"] = operator_handoff.get(
            "review_mutation_step_count"
        )
        result["operator_handoff_evidence_path_count"] = operator_handoff.get(
            "evidence_path_count"
        )
    return result


def compact_paid_review_record(record: dict[str, Any]) -> dict[str, Any]:
    wandb_entries = (
        record.get("wandb_completion_entries")
        if isinstance(record.get("wandb_completion_entries"), list)
        else []
    )
    return {
        "path": record.get("path"),
        "ok": bool(record.get("ok")),
        "status": record.get("status"),
        "phase": record.get("phase"),
        "canary": bool(record.get("canary")),
        "model_count": record.get("model_count"),
        "configs": record.get("configs") if isinstance(record.get("configs"), list) else [],
        "config_count": len(record.get("configs") or []) if isinstance(record.get("configs"), list) else 0,
        "requires_paid_model_api": bool(record.get("requires_paid_model_api")),
        "run_purpose_present": bool(record.get("run_purpose_present")),
        "expected_cost_band_present": bool(record.get("expected_cost_band_present")),
        "actual_cost_estimate_present": bool(record.get("actual_cost_estimate_present")),
        "provider_bill_reference_present": bool(record.get("provider_bill_reference_present")),
        "actual_cost_estimate_placeholder": bool(record.get("actual_cost_estimate_placeholder")),
        "provider_bill_reference_placeholder": bool(record.get("provider_bill_reference_placeholder")),
        "pre_run_budget_estimate": (
            record.get("pre_run_budget_estimate")
            if isinstance(record.get("pre_run_budget_estimate"), dict)
            else None
        ),
        "run_count": record.get("run_count"),
        "verify_wandb_completion": bool(record.get("verify_wandb_completion")),
        "wandb_completion_entry_count": len(wandb_entries),
        "wandb_completion_entries": [
            entry for entry in wandb_entries if isinstance(entry, dict)
        ],
        "verify_weave_agents": bool(record.get("verify_weave_agents")),
        "weave_agents_completion_entry_count": len(record.get("weave_agents_completion_entries") or []),
        "errors": record.get("errors") if isinstance(record.get("errors"), list) else [],
    }


def paid_run_review_package_summary(report: dict[str, Any]) -> dict[str, Any]:
    runner = report.get("runner")
    if not isinstance(runner, dict):
        return {}
    check = runner.get("paid_run_review_check")
    if not isinstance(check, dict):
        return {}
    path_value = check.get("path")
    result: dict[str, Any] = {
        "path": path_value,
        "markdown_path": check.get("markdown_path"),
        "ok": bool(check.get("ok")),
        "status": check.get("status"),
        "summary": check.get("summary") if isinstance(check.get("summary"), dict) else {},
        "requirements": {},
        "review_completion_requirements": {},
        "gates": [],
    }
    if not isinstance(path_value, str) or not path_value.strip():
        return result
    path = repo_path(path_value)
    if not path.exists() or not path.is_file():
        result["read_error"] = "missing paid-run review check JSON"
        return result
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        result["read_error"] = str(exc)
        return result
    result.update(
        {
            "ok": bool(payload.get("ok")),
            "status": payload.get("status"),
            "generated_at": payload.get("generated_at"),
            "review_paths": payload.get("review_paths") if isinstance(payload.get("review_paths"), list) else [],
            "summary": payload.get("summary") if isinstance(payload.get("summary"), dict) else {},
            "requirements": payload.get("requirements") if isinstance(payload.get("requirements"), dict) else {},
            "review_completion_requirements": (
                payload.get("review_completion_requirements")
                if isinstance(payload.get("review_completion_requirements"), dict)
                else {}
            ),
        }
    )
    gates = payload.get("gates")
    if isinstance(gates, list):
        compact_gates: list[dict[str, Any]] = []
        for gate in gates:
            if not isinstance(gate, dict):
                continue
            records = gate.get("records")
            blocking_records = gate.get("blocking_records")
            compact_gates.append(
                {
                    "name": gate.get("name"),
                    "ok": bool(gate.get("ok")),
                    "status": gate.get("status"),
                    "blocking": bool(gate.get("blocking")),
                    "detail": gate.get("detail"),
                    "next_action": gate.get("next_action"),
                    "record_count": len(records) if isinstance(records, list) else 0,
                    "blocking_record_count": (
                        len(blocking_records)
                        if isinstance(blocking_records, list)
                        else 0
                    ),
                    "records": [
                        compact_paid_review_record(record)
                        for record in (records or [])
                        if isinstance(record, dict)
                    ],
                    "blocking_records": [
                        compact_paid_review_record(record)
                        for record in (blocking_records or [])
                        if isinstance(record, dict)
                    ],
                }
            )
        result["gates"] = compact_gates
    return result


def _unique_strings(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def resolve_review_config_path(path_value: Any) -> Path | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    direct = repo_path(path_value)
    if direct.exists():
        return direct
    under_configs = REPO_ROOT / "configs" / path_value
    if under_configs.exists():
        return under_configs
    return direct


def enabled_benchmarks_from_config(path_value: Any) -> dict[str, Any]:
    path = resolve_review_config_path(path_value)
    path_text = path_display(path) if path is not None else str(path_value or "")
    result: dict[str, Any] = {
        "config_path": path_text,
        "enabled_benchmarks": [],
        "errors": [],
    }
    if path is None:
        result["errors"].append("config path is missing")
        return result
    if not path.exists():
        result["errors"].append("config path does not exist")
        return result
    try:
        payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception as exc:  # pragma: no cover - exact parser exception is version-specific
        result["errors"].append(f"config could not be read: {exc}")
        return result
    if not isinstance(payload, dict):
        result["errors"].append("config payload is not an object")
        return result
    run = payload.get("run")
    if not isinstance(run, dict):
        result["errors"].append("config run section is missing")
        return result
    result["enabled_benchmarks"] = sorted(
        str(name)
        for name, enabled in run.items()
        if isinstance(name, str) and bool(enabled)
    )
    return result


def benchmark_progress_matrix_summary(
    *,
    wandb_completion_contract: dict[str, Any],
    paid_run_review_package: dict[str, Any],
    weave_agents_completion: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return a compact phase/benchmark status matrix for operator review.

    The existing W&B completion contract is intentionally strict and focused on
    release completion. This matrix adds the adjacent canary phase state so a
    reviewer can see what is merely prepared, what has W&B proof, and what is
    still waiting on Weave/paid-run accounting without opening each evidence
    file.
    """

    rows: list[dict[str, Any]] = []
    review_records_by_path: dict[str, dict[str, Any]] = {}
    gates = (
        paid_run_review_package.get("gates")
        if isinstance(paid_run_review_package, dict)
        else []
    )
    if isinstance(gates, list):
        for gate in gates:
            if not isinstance(gate, dict):
                continue
            for key in ("records", "blocking_records"):
                records = gate.get(key)
                if not isinstance(records, list):
                    continue
                for record in records:
                    if not isinstance(record, dict):
                        continue
                    path = record.get("path")
                    if not isinstance(path, str) or not path:
                        path = f"{record.get('phase') or 'unknown'}:{len(review_records_by_path)}"
                    review_records_by_path[path] = record

    contract_rows = (
        wandb_completion_contract.get("benchmarks")
        if isinstance(wandb_completion_contract, dict)
        else []
    )
    contract_by_benchmark = {
        str(row.get("benchmark")): row
        for row in contract_rows
        if isinstance(row, dict)
        and isinstance(row.get("benchmark"), str)
        and row.get("benchmark")
    } if isinstance(contract_rows, list) else {}

    for path, record in sorted(review_records_by_path.items()):
        phase = record.get("phase")
        rows.append(
            {
                "scope": "canary_phase",
                "name": phase,
                "phase": phase,
                "status": record.get("status"),
                "ok": bool(record.get("ok")),
                "prepared": record.get("status") == "prepared" or bool(record.get("ok")),
                "review_path": path,
                "model_count": record.get("model_count"),
                "config_count": record.get("config_count"),
                "requires_paid_model_api": bool(record.get("requires_paid_model_api")),
                "run_count": record.get("run_count"),
                "verify_wandb_completion": bool(record.get("verify_wandb_completion")),
                "wandb_completion_entry_count": record.get("wandb_completion_entry_count"),
                "verify_weave_agents": bool(record.get("verify_weave_agents")),
                "weave_agents_completion_entry_count": record.get(
                    "weave_agents_completion_entry_count"
                ),
                "pre_run_budget_estimate_present": isinstance(
                    record.get("pre_run_budget_estimate"),
                    dict,
                ),
                "accounting_complete": bool(record.get("actual_cost_estimate_present"))
                and bool(record.get("provider_bill_reference_present"))
                and not bool(record.get("actual_cost_estimate_placeholder"))
                and not bool(record.get("provider_bill_reference_placeholder")),
                "errors": record.get("errors") if isinstance(record.get("errors"), list) else [],
            }
        )
        configs = record.get("configs")
        if not isinstance(configs, list):
            configs = []
        for config in configs:
            config_summary = enabled_benchmarks_from_config(config)
            config_errors = (
                config_summary.get("errors")
                if isinstance(config_summary.get("errors"), list)
                else []
            )
            enabled = config_summary.get("enabled_benchmarks")
            if not isinstance(enabled, list):
                enabled = []
            for benchmark in enabled:
                release_benchmark = ENABLED_BENCHMARK_RELEASE_ALIASES.get(
                    benchmark,
                    benchmark,
                )
                release_row = contract_by_benchmark.get(release_benchmark) or {}
                wandb_required = bool(release_row.get("required"))
                rows.append(
                    {
                        "scope": "enabled_benchmark",
                        "name": benchmark,
                        "benchmark": benchmark,
                        "release_benchmark": release_benchmark
                        if release_benchmark in contract_by_benchmark
                        else "",
                        "phase": phase,
                        "status": "enabled_not_run"
                        if not record.get("run_count")
                        else record.get("status"),
                        "phase_review_status": record.get("status"),
                        "ok": bool(record.get("ok")),
                        "review_path": path,
                        "config_path": config_summary.get("config_path"),
                        "config_errors": config_errors,
                        "requires_paid_model_api": bool(record.get("requires_paid_model_api")),
                        "run_count": record.get("run_count"),
                        "wandb_completion_required": wandb_required,
                        "wandb_completion_status": (
                            release_row.get("status")
                            if wandb_required
                            else "not_release_gated"
                        ),
                        "weave_agents_required": release_benchmark
                        in {"agentic_math", "agentic_swe"},
                    }
                )

    weave_by_phase: dict[str, dict[str, Any]] = {}
    for row in weave_agents_completion:
        if not isinstance(row, dict):
            continue
        phase = row.get("phase")
        if isinstance(phase, str) and phase:
            existing = weave_by_phase.get(phase)
            if existing is None or (
                int(row.get("verified_count") or 0)
                > int(existing.get("verified_count") or 0)
            ):
                weave_by_phase[phase] = row

    if isinstance(contract_rows, list):
        for row in contract_rows:
            if not isinstance(row, dict):
                continue
            benchmark = row.get("benchmark")
            if not isinstance(benchmark, str) or not benchmark:
                continue
            review_phase = BENCHMARK_REVIEW_PHASES.get(benchmark, "canary_full")
            phase = review_phase.removeprefix("canary_")
            weave = weave_by_phase.get(phase) or {}
            weave_required = benchmark in {"agentic_math", "agentic_swe"}
            review_run_ids = row.get("review_run_ids") if isinstance(row.get("review_run_ids"), list) else []
            standalone_run_ids = (
                row.get("standalone_run_ids")
                if isinstance(row.get("standalone_run_ids"), list)
                else []
            )
            formalized_run_ids = (
                row.get("formalized_existing_run_ids")
                if isinstance(row.get("formalized_existing_run_ids"), list)
                else []
            )
            rows.append(
                {
                    "scope": "release_benchmark",
                    "name": benchmark,
                    "benchmark": benchmark,
                    "phase": phase,
                    "status": row.get("status"),
                    "ok": bool(row.get("release_completion_proven")),
                    "required": bool(row.get("required")),
                    "release_completion_proven": bool(row.get("release_completion_proven")),
                    "standalone_wandb_status": row.get("standalone_status"),
                    "standalone_wandb_ok": bool(row.get("standalone_completion_ok")),
                    "review_wandb_status": row.get("review_status"),
                    "review_wandb_ok": bool(row.get("review_completion_ok")),
                    "existing_formalized": bool(row.get("formalized_existing_result")),
                    "run_ids": _unique_strings(
                        [*review_run_ids, *standalone_run_ids, *formalized_run_ids]
                    ),
                    "scope_confirmation_required": bool(
                        row.get("scope_confirmation_required")
                    ),
                    "weave_agents_required": weave_required,
                    "weave_agents_verified_count": weave.get("verified_count", 0)
                    if weave_required
                    else 0,
                    "weave_agents_entry_count": weave.get("entry_count", 0)
                    if weave_required
                    else 0,
                    "missing_reasons": (
                        row.get("missing_reasons")
                        if isinstance(row.get("missing_reasons"), list)
                        else []
                    ),
                    "next_actions": (
                        row.get("next_actions")
                        if isinstance(row.get("next_actions"), list)
                        else []
                    ),
                }
            )
    return rows


def _completion_verify_command(benchmark: str) -> str:
    parts = [
        "uv run python scripts/tools/verify_taiwan_wandb_completion.py",
        "--run-id RUN_ID",
        f"--benchmark {benchmark}",
    ]
    if benchmark == "agentic_math":
        parts.append("--expected-total 100")
    if benchmark == "agentic_swe":
        parts.append("--expected-total 80")
    if benchmark in {"agentic_math", "agentic_swe"}:
        parts.append("--require-nemoclaw-session-audit")
    parts.extend(
        [
            "--env-file .env",
            f"--json outputs/taiwan_full_eval/wandb_completion/{benchmark}-RUN_ID.json",
        ]
    )
    return " ".join(parts)


def _quote_shell_value(value: Any) -> str:
    return shlex.quote(str(value))


def _adoption_refresh_wandb_completion_command(
    candidate: dict[str, Any],
    *,
    benchmark: str,
) -> str:
    parts = [
        "uv run python scripts/tools/verify_taiwan_wandb_completion.py",
        "--entity",
        _quote_shell_value(candidate.get("wandb_entity") or "WANDB_ENTITY"),
        "--project",
        _quote_shell_value(candidate.get("wandb_project") or "WANDB_PROJECT"),
        "--run-id",
        _quote_shell_value(candidate.get("wandb_run_id") or "RUN_ID"),
        "--benchmark",
        _quote_shell_value(candidate.get("benchmark") or benchmark or "BENCHMARK"),
    ]
    metrics = candidate.get("metrics")
    expected_total = metrics.get("expected_total") if isinstance(metrics, dict) else None
    if isinstance(expected_total, int):
        parts.extend(["--expected-total", str(expected_total)])
    elif benchmark == "agentic_math":
        parts.extend(["--expected-total", "100"])
    elif benchmark == "agentic_swe":
        parts.extend(["--expected-total", "80"])
    if benchmark in {"agentic_math", "agentic_swe"}:
        parts.append("--require-nemoclaw-session-audit")
    model = candidate.get("model")
    if isinstance(model, str) and model:
        parts.extend(
            [
                "--expected-run-config",
                _quote_shell_value(f"model.pretrained_model_name_or_path={model}"),
            ]
        )
    run_name = candidate.get("wandb_run_name")
    if isinstance(run_name, str) and "relog" in run_name:
        parts.extend(["--expected-run-job-type", "evaluation-relog"])
    if not any(part.startswith("--expected-run-") for part in parts):
        parts.extend(["--expected-run-tag", "REVIEWER_SELECTED_SCOPE_TAG"])
    parts.extend(
        [
            "--json",
            _quote_shell_value(
                candidate.get("wandb_completion_json") or "VERIFIER_JSON"
            ),
        ]
    )
    return " ".join(parts)


def _completion_sync_command(
    completion_path: str,
    *,
    benchmark: str,
    adopt_existing_result: bool = False,
    review_json: str | None = None,
    scope_attestation_template: str | None = None,
    in_place: bool = True,
    report_json: str | None = None,
    validated_dry_run_report_json: str | None = None,
) -> str:
    review_path = review_json or review_json_for_benchmark(
        "outputs/taiwan_full_eval/PHASE_paid_run_review.json",
        benchmark,
    )
    command = (
        "uv run python scripts/tools/sync_wandb_completion_to_paid_review.py "
        f"--review-json {review_path} "
        f"--completion-json {completion_path}"
    )
    if in_place:
        command += " --in-place"
    command += " --set-verify-wandb-completion"
    if adopt_existing_result:
        command += " --top-level --adopt-existing-result"
        command += (
            f" --scope-attestation-json {scope_attestation_template}"
            if scope_attestation_template
            else " --scope-attestation-json REQUIRED_SCOPE_ATTESTATION_JSON"
        )
    if report_json:
        command += f" --report-json {report_json}"
    if validated_dry_run_report_json:
        command += f" --validated-dry-run-report-json {validated_dry_run_report_json}"
    return command


def _wandb_contract_next_actions(
    *,
    benchmark: str,
    required: bool,
    release_completion_proven: bool,
    standalone_ok: bool,
    review_ok: bool,
    completion_paths: list[str],
    review_json: str | None = None,
    scope_attestation_template_path: str | None = None,
    sync_dry_run_report_json: str | None = None,
    scope_attestation_preflight_report_json: str | None = None,
    adoption_sync_ready: bool = False,
    adoption_scope_render_command: str | None = None,
    adoption_scope_preflight_command: str | None = None,
    adoption_sync_dry_run_command: str | None = None,
    adoption_sync_apply_command: str | None = None,
    refresh_wandb_completion_command: str | None = None,
    adoption_sync_blocked_reason: str | None = None,
) -> tuple[list[str], list[str], bool, str | None]:
    if not required or release_completion_proven:
        return [], [], False, None

    actions: list[str] = []
    commands: list[str] = []
    scope_confirmation_required = False
    scope_warning: str | None = None
    if not standalone_ok:
        if adoption_sync_ready:
            actions.append("refresh W&B completion verifier before paid-review adoption")
            commands.append(
                refresh_wandb_completion_command
                or _completion_verify_command(benchmark)
            )
        else:
            actions.append("finish benchmark run and create a passing W&B completion verifier")
            commands.append(
                refresh_wandb_completion_command
                or _completion_verify_command(benchmark)
            )
    if (standalone_ok or adoption_sync_ready) and not review_ok:
        if not adoption_sync_ready:
            actions.append(
                "refresh W&B completion verifier with run metadata before paid-review adoption"
            )
            if refresh_wandb_completion_command:
                commands.append(refresh_wandb_completion_command)
        else:
            actions.append("link the passing W&B completion verifier into the paid-run review")
        completion_path = (
            completion_paths[0]
            if completion_paths
            else f"outputs/taiwan_full_eval/wandb_completion/{benchmark}-RUN_ID.json"
        )
        dry_run_report_json = sync_dry_run_report_json or "REQUIRED_SYNC_DRY_RUN_REPORT_JSON"
        if adoption_sync_ready:
            if adoption_scope_render_command:
                commands.append(adoption_scope_render_command)
            if adoption_scope_preflight_command:
                commands.append(adoption_scope_preflight_command)
            commands.append(
                adoption_sync_dry_run_command
                or _completion_sync_command(
                    completion_path,
                    benchmark=benchmark,
                    review_json=review_json,
                    adopt_existing_result=True,
                    scope_attestation_template=scope_attestation_template_path,
                    in_place=False,
                    report_json=dry_run_report_json,
                )
            )
            commands.append(
                adoption_sync_apply_command
                or _completion_sync_command(
                    completion_path,
                    benchmark=benchmark,
                    review_json=review_json,
                    adopt_existing_result=True,
                    scope_attestation_template=scope_attestation_template_path,
                    validated_dry_run_report_json=dry_run_report_json,
                )
            )
            scope_confirmation_required = True
            scope_warning = (
                "Use the sync command only when the W&B run belongs to the reviewed "
                "paid run or agreed canary scope; otherwise rerun the benchmark in "
                "the agreed one-model canary."
            )
    if not release_completion_proven:
        actions.append("rerun the release gate after completion and review evidence are updated")
        commands.append("uv run python scripts/tools/run_taiwan_release_gate.py --quiet")
    return (
        _unique_strings(actions),
        _unique_strings(commands),
        scope_confirmation_required,
        scope_warning,
    )


def wandb_completion_contract_summary(
    *,
    benchmark_completion: list[dict[str, Any]],
    existing_results_formalization: dict[str, Any],
    wandb_adoption_draft: dict[str, Any],
    paid_run_review_package: dict[str, Any],
) -> dict[str, Any]:
    """Summarize required W&B completion proof across release gates.

    This deliberately separates existing-result formalization from release
    completion. A historical/local complete result can be properly logged to
    W&B, but the release contract is only satisfied when the required
    benchmark completion is also linked through the paid-run review evidence.
    """

    requirements = (
        paid_run_review_package.get("requirements")
        if isinstance(paid_run_review_package, dict)
        else {}
    )
    if not isinstance(requirements, dict):
        requirements = {}
    required = [
        str(item)
        for item in requirements.get("required_wandb_benchmarks", [])
        if isinstance(item, str) and item.strip()
    ]
    required_run_ids = requirements.get("required_wandb_run_ids")
    if not isinstance(required_run_ids, dict):
        required_run_ids = {}

    by_benchmark = {
        str(row.get("benchmark")): row
        for row in benchmark_completion
        if isinstance(row, dict) and row.get("benchmark")
    }
    formalized_records = (
        existing_results_formalization.get("formalized_records")
        if isinstance(existing_results_formalization, dict)
        else []
    )
    formalized_by_benchmark: dict[str, list[dict[str, Any]]] = {}
    if isinstance(formalized_records, list):
        for record in formalized_records:
            if not isinstance(record, dict):
                continue
            benchmark = record.get("benchmark")
            if not benchmark:
                continue
            formalized_by_benchmark.setdefault(str(benchmark), []).append(record)
    adoption_candidates = (
        wandb_adoption_draft.get("candidates")
        if isinstance(wandb_adoption_draft, dict)
        else []
    )
    adoption_templates_by_benchmark: dict[str, list[dict[str, Any]]] = {}
    if isinstance(adoption_candidates, list):
        for candidate in adoption_candidates:
            if not isinstance(candidate, dict):
                continue
            benchmark = candidate.get("benchmark")
            template = candidate.get("scope_attestation_template_json")
            if not isinstance(benchmark, str) or not benchmark:
                continue
            if not isinstance(template, str) or not template.strip():
                continue
            adoption_templates_by_benchmark.setdefault(benchmark, []).append(candidate)

    all_benchmarks = _unique_strings(
        [
            *required,
            *list(by_benchmark.keys()),
            *list(formalized_by_benchmark.keys()),
        ]
    )
    benchmark_rows: list[dict[str, Any]] = []
    for benchmark in all_benchmarks:
        row = by_benchmark.get(benchmark, {})
        standalone_records = row.get("standalone_records") if isinstance(row, dict) else []
        if not isinstance(standalone_records, list):
            standalone_records = []
        review_entries = row.get("review_entries") if isinstance(row, dict) else []
        if not isinstance(review_entries, list):
            review_entries = []
        formalized = formalized_by_benchmark.get(benchmark, [])
        formalized_run_ids = _unique_strings(
            [
                ((record.get("wandb_completion") or {}).get("run_id"))
                for record in formalized
                if isinstance(record, dict)
                and isinstance(record.get("wandb_completion"), dict)
            ]
        )
        standalone_run_ids = _unique_strings(
            [
                record.get("run_id")
                for record in standalone_records
                if isinstance(record, dict) and record.get("ok")
            ]
        )
        review_run_ids = _unique_strings(
            [
                entry.get("run_id")
                for entry in review_entries
                if isinstance(entry, dict) and entry.get("ok")
            ]
        )
        standalone_paths = _unique_strings(
            [
                record.get("path")
                for record in standalone_records
                if isinstance(record, dict) and record.get("path")
            ]
        )
        review_paths = _unique_strings(
            [
                entry.get("path")
                for entry in review_entries
                if isinstance(entry, dict) and entry.get("path")
            ]
        )
        formalized_paths = _unique_strings(
            [
                ((record.get("wandb_completion") or {}).get("path"))
                for record in formalized
                if isinstance(record, dict)
                and isinstance(record.get("wandb_completion"), dict)
            ]
        )
        required_for_release = benchmark in required
        standalone_ok = bool(row.get("standalone_ok")) if isinstance(row, dict) else False
        review_ok = bool(row.get("review_ok")) if isinstance(row, dict) else False
        release_completion_proven = bool(row.get("completion_proven")) if isinstance(row, dict) else False
        formalized_existing_result = bool(formalized)
        missing_reasons: list[str] = []
        if required_for_release and benchmark not in by_benchmark:
            missing_reasons.append("benchmark evidence row is missing")
        if required_for_release and not standalone_ok:
            missing_reasons.append("standalone W&B completion verifier is missing or failing")
        if required_for_release and not review_ok:
            missing_reasons.append("paid-run review W&B completion entry is missing or failing")
        if required_for_release and not release_completion_proven:
            missing_reasons.append("release completion_proven is false")

        if release_completion_proven:
            status = "release_complete"
        elif standalone_ok or formalized_existing_result:
            status = "formalized_but_not_reviewed"
        elif required_for_release:
            status = "missing_release_completion"
        else:
            status = "not_required"

        completion_paths = _unique_strings([*standalone_paths, *formalized_paths])
        scope_template_candidates = adoption_templates_by_benchmark.get(benchmark, [])
        matching_scope_candidates = [
            candidate
            for candidate in scope_template_candidates
            if isinstance(candidate, dict)
            and (
                not completion_paths
                or candidate.get("wandb_completion_json") in completion_paths
            )
        ]
        sync_ready_scope_candidates = [
            candidate
            for candidate in matching_scope_candidates
            if candidate.get("sync_ready") is True
        ]
        scope_attestation_template_paths = _unique_strings(
            [
                candidate.get("scope_attestation_template_json")
                for candidate in matching_scope_candidates
                if isinstance(candidate.get("scope_attestation_template_json"), str)
                and candidate.get("scope_attestation_template_json")
            ]
        )
        sync_ready_scope_attestation_template_paths = _unique_strings(
            [
                candidate.get("scope_attestation_template_json")
                for candidate in sync_ready_scope_candidates
                if isinstance(candidate.get("scope_attestation_template_json"), str)
                and candidate.get("scope_attestation_template_json")
            ]
        )
        sync_ready_candidate = (
            sync_ready_scope_candidates[0]
            if sync_ready_scope_candidates
            else None
        )
        scope_attestation_template_path = (
            sync_ready_scope_attestation_template_paths[0]
            if sync_ready_scope_attestation_template_paths
            else None
        )
        scope_attestation_review_paths = _unique_strings(
            [
                candidate.get("target_review_json")
                for candidate in sync_ready_scope_candidates
                if isinstance(candidate.get("target_review_json"), str)
                and candidate.get("target_review_json")
            ]
        )
        scope_attestation_review_path = (
            scope_attestation_review_paths[0]
            if scope_attestation_review_paths
            else None
        )
        sync_dry_run_report_paths = _unique_strings(
            [
                candidate.get("sync_dry_run_report_json")
                for candidate in sync_ready_scope_candidates
                if isinstance(candidate.get("sync_dry_run_report_json"), str)
                and candidate.get("sync_dry_run_report_json")
            ]
        )
        sync_dry_run_report_json = (
            sync_dry_run_report_paths[0]
            if sync_dry_run_report_paths
            else None
        )
        scope_attestation_preflight_report_paths = _unique_strings(
            [
                candidate.get("scope_attestation_preflight_report_json")
                for candidate in sync_ready_scope_candidates
                if isinstance(candidate.get("scope_attestation_preflight_report_json"), str)
                and candidate.get("scope_attestation_preflight_report_json")
            ]
        )
        scope_attestation_preflight_report_json = (
            scope_attestation_preflight_report_paths[0]
            if scope_attestation_preflight_report_paths
            else None
        )
        refresh_wandb_completion_command = None
        if matching_scope_candidates and isinstance(matching_scope_candidates[0], dict):
            first_matching_scope_candidate = matching_scope_candidates[0]
            refresh_wandb_completion_command = first_matching_scope_candidate.get(
                "refresh_wandb_completion_command"
            )
            if not isinstance(refresh_wandb_completion_command, str) or not refresh_wandb_completion_command.strip():
                refresh_wandb_completion_command = (
                    _adoption_refresh_wandb_completion_command(
                        first_matching_scope_candidate,
                        benchmark=benchmark,
                    )
                )
        next_actions, recommended_commands, scope_confirmation_required, scope_warning = (
            _wandb_contract_next_actions(
                benchmark=benchmark,
                required=required_for_release,
                release_completion_proven=release_completion_proven,
                standalone_ok=standalone_ok,
                review_ok=review_ok,
                completion_paths=completion_paths,
                review_json=scope_attestation_review_path,
                scope_attestation_template_path=scope_attestation_template_path,
                sync_dry_run_report_json=sync_dry_run_report_json,
                scope_attestation_preflight_report_json=scope_attestation_preflight_report_json,
                adoption_sync_ready=sync_ready_candidate is not None,
                adoption_scope_render_command=(
                    sync_ready_candidate.get("scope_attestation_render_command")
                    if isinstance(sync_ready_candidate, dict)
                    else None
                ),
                adoption_scope_preflight_command=(
                    sync_ready_candidate.get("scope_attestation_preflight_command")
                    if isinstance(sync_ready_candidate, dict)
                    else None
                ),
                adoption_sync_dry_run_command=(
                    sync_ready_candidate.get("sync_dry_run_command")
                    if isinstance(sync_ready_candidate, dict)
                    else None
                ),
                adoption_sync_apply_command=(
                    (
                        sync_ready_candidate.get("sync_apply_command")
                        or sync_ready_candidate.get("sync_command")
                    )
                    if isinstance(sync_ready_candidate, dict)
                    else None
                ),
                refresh_wandb_completion_command=refresh_wandb_completion_command,
                adoption_sync_blocked_reason=(
                    matching_scope_candidates[0].get("sync_command_blocked_reason")
                    if matching_scope_candidates
                    and isinstance(matching_scope_candidates[0], dict)
                    else None
                ),
            )
        )
        benchmark_rows.append(
            {
                "benchmark": benchmark,
                "required": required_for_release,
                "expected_run_id": row.get("expected_run_id") if isinstance(row, dict) else required_run_ids.get(benchmark),
                "release_completion_proven": release_completion_proven,
                "standalone_completion_ok": standalone_ok,
                "standalone_status": row.get("standalone_status") if isinstance(row, dict) else None,
                "standalone_run_ids": standalone_run_ids,
                "standalone_completion_paths": standalone_paths,
                "review_completion_ok": review_ok,
                "review_status": row.get("review_status") if isinstance(row, dict) else None,
                "review_run_ids": review_run_ids,
                "review_completion_paths": review_paths,
                "formalized_existing_result": formalized_existing_result,
                "formalized_existing_run_ids": formalized_run_ids,
                "formalized_existing_completion_paths": formalized_paths,
                "formalized_existing_count": len(formalized),
                "scope_attestation_template_paths": scope_attestation_template_paths,
                "sync_ready_adoption_candidate_count": len(sync_ready_scope_candidates),
                "adoption_sync_blocked_reasons": _unique_strings(
                    [
                        candidate.get("sync_command_blocked_reason")
                        for candidate in matching_scope_candidates
                        if isinstance(candidate, dict)
                        and isinstance(candidate.get("sync_command_blocked_reason"), str)
                        and candidate.get("sync_command_blocked_reason")
                    ]
                ),
                "refresh_wandb_completion_commands": _unique_strings(
                    [
                        candidate.get("refresh_wandb_completion_command")
                        if isinstance(candidate.get("refresh_wandb_completion_command"), str)
                        and candidate.get("refresh_wandb_completion_command").strip()
                        else (
                            _adoption_refresh_wandb_completion_command(
                                candidate,
                                benchmark=benchmark,
                            )
                            if not standalone_ok
                            else None
                        )
                        for candidate in matching_scope_candidates
                        if isinstance(candidate, dict)
                    ]
                ),
                "scope_attestation_render_report_paths": _unique_strings(
                    [
                        candidate.get("scope_attestation_render_report_json")
                        for candidate in sync_ready_scope_candidates
                        if isinstance(candidate.get("scope_attestation_render_report_json"), str)
                        and candidate.get("scope_attestation_render_report_json")
                    ]
                ),
                "scope_attestation_preflight_report_paths": scope_attestation_preflight_report_paths,
                "sync_dry_run_report_paths": sync_dry_run_report_paths,
                "status": status,
                "missing_reasons": missing_reasons,
                "next_actions": next_actions,
                "recommended_commands": recommended_commands,
                "scope_confirmation_required": scope_confirmation_required,
                "scope_warning": scope_warning,
            }
        )

    required_rows = [row for row in benchmark_rows if row.get("required")]
    missing_release = [
        row["benchmark"]
        for row in required_rows
        if not row.get("release_completion_proven")
    ]
    complete = bool(required_rows) and not missing_release
    return {
        "status": "passed" if complete else ("no_required_benchmarks" if not required_rows else "incomplete"),
        "complete": complete,
        "required_benchmarks": required,
        "required_count": len(required_rows),
        "release_completion_proven_count": sum(
            1 for row in required_rows if row.get("release_completion_proven")
        ),
        "standalone_completion_ok_count": sum(
            1 for row in required_rows if row.get("standalone_completion_ok")
        ),
        "formalized_existing_result_count": sum(
            1 for row in required_rows if row.get("formalized_existing_result")
        ),
        "missing_release_completion_benchmarks": missing_release,
        "max_age_seconds": requirements.get("wandb_completion_max_age_seconds"),
        "next_action_count": sum(len(row.get("next_actions") or []) for row in required_rows),
        "benchmarks": benchmark_rows,
    }


def nemoclaw_adoption_summary(report: dict[str, Any]) -> dict[str, Any]:
    runner = report.get("runner")
    if not isinstance(runner, dict):
        return {}
    adoption = runner.get("nemoclaw_adoption_check")
    if not isinstance(adoption, dict):
        return {}
    path_value = adoption.get("path")
    result: dict[str, Any] = {
        "path": path_value,
        "markdown_path": adoption.get("markdown_path"),
        "ok": bool(adoption.get("ok")),
        "status": adoption.get("status"),
        "adoption_decision": {},
        "summary": adoption.get("summary") if isinstance(adoption.get("summary"), dict) else {},
        "criteria": [],
    }
    if not isinstance(path_value, str) or not path_value.strip():
        return result
    path = repo_path(path_value)
    if not path.exists() or not path.is_file():
        result["read_error"] = "missing adoption JSON"
        return result
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        result["read_error"] = str(exc)
        return result
    result.update(
        {
            "ok": bool(payload.get("ok")),
            "status": payload.get("status"),
            "generated_at": payload.get("generated_at"),
            "sandbox": payload.get("sandbox"),
            "adoption_decision": (
                payload.get("adoption_decision")
                if isinstance(payload.get("adoption_decision"), dict)
                else {}
            ),
            "summary": payload.get("summary") if isinstance(payload.get("summary"), dict) else {},
            "setup_paths": payload.get("setup_paths") if isinstance(payload.get("setup_paths"), list) else [],
            "readiness_paths": payload.get("readiness_paths") if isinstance(payload.get("readiness_paths"), list) else [],
            "agentic_config_paths": payload.get("agentic_config_paths") if isinstance(payload.get("agentic_config_paths"), list) else [],
        }
    )
    decision = (
        payload.get("adoption_decision")
        if isinstance(payload.get("adoption_decision"), dict)
        else {}
    )
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    result["ready_for_use"] = (
        payload.get("ready_for_use")
        if isinstance(payload.get("ready_for_use"), bool)
        else bool(decision.get("ready_for_use"))
    )
    result["adoption_recommendation"] = payload.get(
        "adoption_recommendation",
        decision.get("recommendation"),
    )
    result["adoption_scope"] = payload.get("adoption_scope", decision.get("scope"))
    result["design_ready"] = (
        payload.get("design_ready")
        if isinstance(payload.get("design_ready"), bool)
        else decision.get("design_ready")
    )
    result["blockers"] = (
        payload.get("blockers")
        if isinstance(payload.get("blockers"), list)
        else
        summary.get("blockers")
        if isinstance(summary.get("blockers"), list)
        else decision.get("blockers")
        if isinstance(decision.get("blockers"), list)
        else []
    )
    result["runtime_blockers"] = (
        payload.get("runtime_blockers")
        if isinstance(payload.get("runtime_blockers"), list)
        else decision.get("runtime_blockers")
        if isinstance(decision.get("runtime_blockers"), list)
        else []
    )
    result["design_blockers"] = (
        payload.get("design_blockers")
        if isinstance(payload.get("design_blockers"), list)
        else decision.get("design_blockers")
        if isinstance(decision.get("design_blockers"), list)
        else []
    )
    result["other_blockers"] = (
        payload.get("other_blockers")
        if isinstance(payload.get("other_blockers"), list)
        else decision.get("other_blockers")
        if isinstance(decision.get("other_blockers"), list)
        else []
    )
    setup_runtime = (
        payload.get("setup_runtime")
        if isinstance(payload.get("setup_runtime"), dict)
        else summary.get("setup_runtime")
        if isinstance(summary.get("setup_runtime"), dict)
        else None
    )
    if isinstance(setup_runtime, dict):
        result["setup_runtime"] = setup_runtime
    operator_handoff = (
        payload.get("operator_handoff")
        if isinstance(payload.get("operator_handoff"), dict)
        else {}
    )
    if operator_handoff:
        result["operator_handoff"] = operator_handoff
        result["operator_handoff_step_count"] = operator_handoff.get("step_count")
        result["operator_handoff_external_action_step_count"] = operator_handoff.get(
            "external_action_step_count"
        )
        result["operator_handoff_evidence_path_count"] = operator_handoff.get(
            "evidence_path_count"
        )
    if isinstance(payload.get("missing_required_commands"), list):
        result["missing_required_commands"] = payload["missing_required_commands"]
    elif isinstance(setup_runtime, dict) and isinstance(
        setup_runtime.get("missing_required_commands"),
        list,
    ):
        result["missing_required_commands"] = setup_runtime["missing_required_commands"]
    if isinstance(payload.get("missing_components"), list):
        result["missing_components"] = payload["missing_components"]
    elif isinstance(setup_runtime, dict) and isinstance(
        setup_runtime.get("missing_components"),
        list,
    ):
        result["missing_components"] = setup_runtime["missing_components"]
    criteria = payload.get("criteria")
    if isinstance(criteria, list):
        result["criteria"] = []
        for row in criteria:
            if not isinstance(row, dict):
                continue
            criterion = {
                "name": row.get("name"),
                "ok": bool(row.get("ok")),
                "status": row.get("status"),
                "detail": row.get("detail"),
                "next_action": row.get("next_action"),
                "evidence_paths": row.get("evidence_paths") if isinstance(row.get("evidence_paths"), list) else [],
            }
            if "host_prerequisites_ok" in row:
                criterion["host_prerequisites_ok"] = row.get("host_prerequisites_ok")
            if "runtime_installed" in row:
                criterion["runtime_installed"] = row.get("runtime_installed")
            if isinstance(row.get("missing_required_commands"), list):
                criterion["missing_required_commands"] = row.get("missing_required_commands")
            if isinstance(row.get("missing_components"), list):
                criterion["missing_components"] = row.get("missing_components")
            for key in (
                "wandb_weave_policy_present",
                "runtime_network_policy_allowlist_ok",
                "unknown_runtime_network_policies",
                "allowed_runtime_network_policies",
                "detailed_status_network_policy_count",
                "detailed_status_network_policies",
                "non_wandb_network_policies",
            ):
                value = row.get(key)
                if isinstance(value, (bool, int, str)) or isinstance(value, list):
                    criterion[key] = value
            result["criteria"].append(criterion)
    return result


def nemoclaw_post_install_runner_summary(value: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": value.get("path"),
        "markdown_path": value.get("markdown_path"),
        "ok": value.get("ok"),
        "status": value.get("status"),
        "returncode": value.get("returncode"),
        "summary": value.get("summary"),
    }
    path_value = value.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        return result
    try:
        payload = read_json(repo_path(path_value))
    except Exception as exc:
        result["read_error"] = str(exc)
        return result
    result.update(
        {
            "ok": payload.get("ok"),
            "status": payload.get("status"),
            "generated_at": payload.get("generated_at"),
            "summary": payload.get("summary") if isinstance(payload.get("summary"), dict) else {},
            "will_launch_model_inference": payload.get("will_launch_model_inference"),
            "will_query_wandb": payload.get("will_query_wandb"),
            "will_install_or_onboard": payload.get("will_install_or_onboard"),
            "command_safety": (
                payload.get("command_safety")
                if isinstance(payload.get("command_safety"), dict)
                else {}
            ),
            "outputs": payload.get("outputs") if isinstance(payload.get("outputs"), dict) else {},
        }
    )
    steps = payload.get("steps")
    if isinstance(steps, list):
        result["steps"] = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            command = step.get("command")
            command_display = ""
            if isinstance(command, list):
                command_display = shlex.join(str(token) for token in command)
            elif isinstance(command, str):
                command_display = command
            result["steps"].append(
                {
                    "name": step.get("name"),
                    "ok": step.get("ok"),
                    "returncode": step.get("returncode"),
                    "returncode_ok": step.get("returncode_ok"),
                    "payload_ok": step.get("payload_ok"),
                    "payload_status": step.get("payload_status"),
                    "timed_out": step.get("timed_out"),
                    "output_json": step.get("output_json"),
                    "command": command_display,
                }
            )
    return result


def nemoclaw_installer_review_summary(gates: list[dict[str, Any]]) -> dict[str, Any]:
    for gate in gates:
        if not isinstance(gate, dict) or gate.get("name") != "nemoclaw_readiness":
            continue
        review = gate.get("latest_installer_review")
        if not isinstance(review, dict) or not review:
            return {}
        return {
            "path": review.get("path"),
            "generated_at": review.get("generated_at"),
            "installer_url": review.get("installer_url"),
            "install_ref": review.get("install_ref"),
            "lock_json": review.get("lock_json"),
            "lock_verified": review.get("lock_verified"),
            "expected_sha256": review.get("expected_sha256"),
            "sha256": review.get("sha256"),
            "size_bytes": review.get("size_bytes"),
            "status": review.get("status"),
            "ok": review.get("ok"),
        }
    return {}


def runner_evidence_summary(report: dict[str, Any]) -> dict[str, Any]:
    runner = report.get("runner")
    if not isinstance(runner, dict):
        return {}
    result: dict[str, Any] = {
        "name": runner.get("name"),
        "generated_at": runner.get("generated_at"),
        "report_json": runner.get("report_json"),
        "forwarded_report_args": runner.get("forwarded_report_args", []),
    }
    for key in (
        "nemoclaw_check",
        "existing_results_audit",
        "wandb_adoption_draft",
        "wandb_adoption_unconfirmed_checks",
        "nemoclaw_operator_docs_verification",
        "paid_run_review_check",
        "weave_agents_adoption_validation_failures",
        "nemoclaw_adoption_check",
        "nemoclaw_post_install_verification",
    ):
        value = runner.get(key)
        if not isinstance(value, dict):
            continue
        if key == "nemoclaw_post_install_verification":
            result[key] = nemoclaw_post_install_runner_summary(value)
            continue
        if key == "wandb_adoption_unconfirmed_checks":
            records = value.get("records")
            result[key] = {
                "ok": value.get("ok"),
                "status": value.get("status"),
                "record_count": value.get("record_count"),
                "output_dir": value.get("output_dir"),
                "records": [
                    {
                        "benchmark": record.get("benchmark"),
                        "run_id": record.get("run_id"),
                        "scope_attestation_template_json": record.get(
                            "scope_attestation_template_json"
                        ),
                        "preflight_report_json": record.get("preflight_report_json"),
                        "preflight_returncode": record.get("preflight_returncode"),
                        "preflight_failed_as_expected": record.get(
                            "preflight_failed_as_expected"
                        ),
                        "preflight_status": record.get("preflight_status"),
                        "sync_dry_run_report_json": record.get(
                            "sync_dry_run_report_json"
                        ),
                        "sync_dry_run_returncode": record.get("sync_dry_run_returncode"),
                        "sync_dry_run_failed_as_expected": record.get(
                            "sync_dry_run_failed_as_expected"
                        ),
                        "sync_dry_run_status": record.get("sync_dry_run_status"),
                        "will_query_wandb": record.get("will_query_wandb"),
                        "will_write_wandb": record.get("will_write_wandb"),
                        "will_launch_model_inference": record.get(
                            "will_launch_model_inference"
                        ),
                        "will_mutate_review_json": record.get(
                            "will_mutate_review_json"
                        ),
                    }
                    for record in records
                    if isinstance(record, dict)
                ]
                if isinstance(records, list)
                else [],
            }
            continue
        if key == "weave_agents_adoption_validation_failures":
            records = value.get("records")
            result[key] = {
                "ok": value.get("ok"),
                "status": value.get("status"),
                "record_count": value.get("record_count"),
                "glob": value.get("glob"),
                "records": [
                    {
                        "path": record.get("path"),
                        "ok": record.get("ok"),
                        "status": record.get("status"),
                        "review_path": record.get("review_path"),
                        "completion_paths": record.get("completion_paths"),
                        "validation_errors": record.get("validation_errors"),
                        "dry_run": record.get("dry_run"),
                        "in_place": record.get("in_place"),
                        "entry_count": record.get("entry_count"),
                        "change_count": record.get("change_count"),
                    }
                    for record in records
                    if isinstance(record, dict)
                ]
                if isinstance(records, list)
                else [],
            }
            continue
        result[key] = {
            "path": value.get("path"),
            "markdown_path": value.get("markdown_path"),
            "ok": value.get("ok"),
            "status": value.get("status"),
            "returncode": value.get("returncode"),
            "summary": value.get("summary"),
        }
    return result


def compact_evidence_paths(paths: Any, *, limit: int = 5) -> list[str]:
    if not isinstance(paths, list):
        return []
    values = [str(path) for path in paths if isinstance(path, str) and path.strip()]
    values.sort(
        key=lambda value: (
            repo_path(value).stat().st_mtime if repo_path(value).exists() else -1.0,
            value,
        )
    )
    return values[-limit:]


def required_next_actions(gates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for gate in gates:
        if gate.get("ok") or not gate.get("blocking"):
            continue
        evidence_paths = gate.get("evidence_paths", [])
        actions.append(
            {
                "gate": gate.get("name"),
                "status": gate.get("status"),
                "detail": gate.get("detail"),
                "next_action": gate.get("next_action"),
                "evidence_path_count": len(evidence_paths) if isinstance(evidence_paths, list) else 0,
                "latest_evidence_paths": compact_evidence_paths(evidence_paths),
            }
        )
    return actions


def _command_flag_value(parts: list[str], flag: str, default: str | None = None) -> str | None:
    try:
        index = parts.index(flag)
    except ValueError:
        return default
    if index + 1 >= len(parts):
        return default
    return parts[index + 1]


def _command_flag_values(parts: list[str], flag: str) -> list[str]:
    values: list[str] = []
    for index, part in enumerate(parts[:-1]):
        if part == flag:
            values.append(parts[index + 1])
    return values


def _taiwan_full_batch_implicit_output_paths(parts: list[str]) -> list[str]:
    if not any(part.endswith("run_taiwan_full_eval_batch.py") for part in parts):
        return []
    output_root = _command_flag_value(parts, "--output-root", "outputs/taiwan_full_eval")
    phase = _command_flag_value(parts, "--phase", "full")
    if not output_root or not phase:
        return []
    plan_prefix = "canary_" if "--canary" in parts else ""
    root = Path(output_root)
    outputs = [
        str(root / f"{plan_prefix}{phase}_execution_plan.json"),
        str(root / f"{plan_prefix}{phase}_paid_run_review.json"),
    ]
    if "--prepare-only" not in parts:
        outputs.append(str(root / "batch_manifest.json"))
    if "--verify-wandb-completion" in parts:
        benchmarks = _command_flag_values(parts, "--wandb-verify-benchmark")
        if not benchmarks:
            benchmarks = {
                "full": ["taiwan_full"],
                "agentic": ["agentic_math", "agentic_swe"],
                "agentic_aggregate": ["taiwan_full"],
            }.get(phase, [])
        outputs.extend(
            str(root / "wandb_completion" / f"{phase}-MODEL_SLUG-{benchmark}.json")
            for benchmark in benchmarks
        )
    if "--verify-weave-agents" in parts:
        outputs.append(str(root / "weave_agents_completion" / f"{phase}-MODEL_SLUG.json"))
    return _unique_strings(outputs)


def _weave_content_canary_implicit_output_paths(parts: list[str]) -> list[str]:
    if not any(part.endswith("run_weave_agents_content_canary.py") for part in parts):
        return []
    output_dir = _command_flag_value(parts, "--output-dir", "outputs/weave_agents_content_canary")
    canary_id = _command_flag_value(parts, "--canary-id", "CANARY_ID")
    if not output_dir or not canary_id:
        return []
    safe_id = "".join(char if char.isalnum() else "_" for char in canary_id)
    task_id = f"weave_agents_content_canary_{safe_id}"
    root = Path(output_dir)
    outputs = [
        str(root / "prompts" / f"{task_id}.md"),
        str(root / "plans" / f"{task_id}.json"),
        str(root / "plans" / f"{task_id}.gate.json"),
    ]
    if "--execute" in parts:
        outputs.extend(
            [
                str(root / "plans" / f"{task_id}.command_result.json"),
                str(root / "agentic_math" / task_id / "openclaw_result.json"),
                str(root / "verifier" / task_id / "attempt_*.json"),
                str(root / "agents_diagnostics" / f"{task_id}.agents.json"),
            ]
        )
    return _unique_strings(outputs)


def _command_output_paths(command: str) -> list[str]:
    try:
        parts = shlex.split(command)
    except ValueError:
        return []
    output_flags = {
        "--json",
        "--markdown",
        "--report-json",
        "--release-gate-json",
        "--bundle-verification-json",
        "--readiness-report-json",
    }
    outputs: list[str] = []
    for index, part in enumerate(parts[:-1]):
        if part in output_flags:
            outputs.append(parts[index + 1])
    outputs.extend(_taiwan_full_batch_implicit_output_paths(parts))
    outputs.extend(_weave_content_canary_implicit_output_paths(parts))
    outputs.extend(_nemoclaw_install_operation_log_paths(parts))
    return _unique_strings(outputs)


def _nemoclaw_install_operation_log_paths(parts: list[str]) -> list[str]:
    if not any(part.endswith("install_nemoclaw.sh") for part in parts):
        return []
    if "--install" not in parts and "--onboard" not in parts:
        return []
    output_json = ""
    for index, part in enumerate(parts[:-1]):
        if part == "--json":
            output_json = parts[index + 1]
    if not output_json:
        return []
    stem = output_json[:-5] if output_json.endswith(".json") else output_json
    outputs: list[str] = []
    if "--install" in parts:
        outputs.append(f"{stem}.install.log")
    if "--onboard" in parts:
        outputs.append(f"{stem}.onboard.log")
    return outputs


def _command_requires_paid_api(command: str) -> bool:
    if "--prepare-only" in command:
        return False
    return any(
        marker in command
        for marker in (
            "run_taiwan_full_eval_batch.py",
            "run_weave_agents_content_canary.py --execute",
            "run_agentic_math_openclaw.py",
            "run_swebench_pro_openclaw.py",
        )
    )


def _command_requires_wandb_access(command: str) -> bool:
    return any(
        marker in command
        for marker in (
            "verify_taiwan_wandb_completion.py",
            "run_taiwan_full_eval_batch.py",
            "run_weave_agents_content_canary.py",
            "run_openclaw_agent_protocol.py check-agents",
            "verify_taiwan_weave_agents.py",
            "sync_wandb_completion_to_paid_review.py",
            "sync_weave_agents_completion_to_paid_review.py",
        )
    )


def _command_requires_wandb_write(command: str) -> bool:
    if "--prepare-only" in command:
        return False
    return any(
        marker in command
        for marker in (
            "run_taiwan_full_eval_batch.py",
            "log_agentic_math_results_to_wandb.py",
            "log_agentic_swe_results_to_wandb.py",
            "run_weave_agents_content_canary.py --execute",
        )
    )


OPERATOR_PLACEHOLDER_TOKENS = (
    "PHASE",
    "RUN_ID",
    "MODEL_SLUG",
    "YYYYMMDD",
    "YYYYMMDDTHHMM",
    "YYYYMMDDTHHMMSS",
    "CONTENT_CANARY_YYYYMMDDTHHMM",
    "WEAVE_CONTENT_CANARY_GATE",
    "REQUIRED_SCOPE_ATTESTATION_JSON",
    "REQUIRED_SYNC_DRY_RUN_REPORT_JSON",
)


def unresolved_placeholder_tokens(values: list[str]) -> list[str]:
    found: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        for token in OPERATOR_PLACEHOLDER_TOKENS:
            if token in value and token not in found:
                found.append(token)
    return found


def operator_next_steps_summary(
    *,
    remediation_plan: list[dict[str, Any]],
    wandb_completion_contract: dict[str, Any],
    paid_run_review_package: dict[str, Any],
    nemoclaw_adoption: dict[str, Any],
) -> dict[str, Any]:
    contract_rows = (
        wandb_completion_contract.get("benchmarks")
        if isinstance(wandb_completion_contract, dict)
        else []
    )
    if not isinstance(contract_rows, list):
        contract_rows = []
    contract_pending_rows = [
        row
        for row in contract_rows
        if isinstance(row, dict)
        and row.get("required")
        and not row.get("release_completion_proven")
    ]
    required_review_fields = (
        (paid_run_review_package.get("review_completion_requirements") or {}).get(
            "completed_review_required_fields"
        )
        if isinstance(paid_run_review_package, dict)
        else []
    )
    if not isinstance(required_review_fields, list):
        required_review_fields = []
    adoption_decision = (
        nemoclaw_adoption.get("adoption_decision")
        if isinstance(nemoclaw_adoption, dict)
        else {}
    )
    if not isinstance(adoption_decision, dict):
        adoption_decision = {}

    steps: list[dict[str, Any]] = []
    for raw_step in remediation_plan:
        if not isinstance(raw_step, dict):
            continue
        gate = str(raw_step.get("gate") or "unknown_gate")
        commands = [
            str(command)
            for command in raw_step.get("commands", [])
            if isinstance(command, str) and command.strip()
        ]
        related_contract_rows: list[dict[str, Any]] = []
        if gate == "wandb_completion":
            related_contract_rows = contract_pending_rows
        elif gate == "paid_run_review_package":
            related_contract_rows = [
                row for row in contract_pending_rows if row.get("scope_confirmation_required")
            ]

        contract_commands: list[str] = []
        contract_warnings: list[str] = []
        contract_benchmarks: list[dict[str, Any]] = []
        for row in related_contract_rows:
            contract_commands.extend(
                command
                for command in row.get("recommended_commands", [])
                if isinstance(command, str) and command.strip()
            )
            warning = row.get("scope_warning")
            if isinstance(warning, str) and warning.strip():
                contract_warnings.append(warning)
            contract_benchmarks.append(
                {
                    "benchmark": row.get("benchmark"),
                    "status": row.get("status"),
                    "next_actions": row.get("next_actions") if isinstance(row.get("next_actions"), list) else [],
                    "scope_confirmation_required": bool(row.get("scope_confirmation_required")),
                }
            )
        all_commands = _unique_strings([*commands, *contract_commands])
        requires_scope_confirmation = any(
            row.get("scope_confirmation_required") for row in related_contract_rows
        )
        requires_third_party_acceptance = any(
            "--yes-i-accept-third-party-software" in command for command in all_commands
        )
        requires_nemoclaw_install = gate == "nemoclaw_readiness" or any(
            "install_nemoclaw.sh --install" in command
            or "install_nemoclaw.sh --install --onboard" in command
            for command in all_commands
        )
        evidence_to_produce = _unique_strings(
            [
                output
                for command in all_commands
                for output in _command_output_paths(command)
            ]
        )
        placeholder_tokens = unresolved_placeholder_tokens([*all_commands, *evidence_to_produce])
        command_template_count = sum(
            1 for command in all_commands if unresolved_placeholder_tokens([command])
        )
        evidence_template_count = sum(
            1 for output in evidence_to_produce if unresolved_placeholder_tokens([output])
        )
        steps.append(
            {
                "order": len(steps) + 1,
                "gate": gate,
                "status": raw_step.get("status"),
                "next_action": raw_step.get("next_action"),
                "requires_paid_api": any(_command_requires_paid_api(command) for command in all_commands),
                "requires_wandb_access": any(_command_requires_wandb_access(command) for command in all_commands),
                "requires_wandb_write": any(_command_requires_wandb_write(command) for command in all_commands),
                "requires_third_party_acceptance": requires_third_party_acceptance,
                "requires_nemoclaw_install": requires_nemoclaw_install,
                "requires_scope_confirmation": requires_scope_confirmation,
                "commands": all_commands,
                "evidence_to_produce": evidence_to_produce,
                "command_count": len(all_commands),
                "evidence_path_count": len(evidence_to_produce),
                "unresolved_placeholder_tokens": placeholder_tokens,
                "command_template_count": command_template_count,
                "evidence_template_count": evidence_template_count,
                "ready_to_execute_without_placeholder": not placeholder_tokens,
                "related_contract_benchmarks": contract_benchmarks,
                "warnings": _unique_strings(contract_warnings),
            }
        )

    warnings = _unique_strings(
        [
            warning
            for step in steps
            for warning in (step.get("warnings") or [])
            if isinstance(warning, str)
        ]
    )
    return {
        "status": "complete" if not steps else "pending",
        "step_count": len(steps),
        "paid_api_step_count": sum(1 for step in steps if step.get("requires_paid_api")),
        "wandb_access_step_count": sum(1 for step in steps if step.get("requires_wandb_access")),
        "wandb_write_step_count": sum(1 for step in steps if step.get("requires_wandb_write")),
        "third_party_acceptance_step_count": sum(
            1 for step in steps if step.get("requires_third_party_acceptance")
        ),
        "scope_confirmation_step_count": sum(
            1 for step in steps if step.get("requires_scope_confirmation")
        ),
        "command_template_step_count": sum(
            1 for step in steps if step.get("command_template_count", 0) > 0
        ),
        "unresolved_placeholder_tokens": _unique_strings(
            [
                token
                for step in steps
                for token in (
                    step.get("unresolved_placeholder_tokens")
                    if isinstance(step.get("unresolved_placeholder_tokens"), list)
                    else []
                )
                if isinstance(token, str) and token
            ]
        ),
        "nemoclaw_recommendation": adoption_decision.get("recommendation"),
        "nemoclaw_ready_for_use": adoption_decision.get("ready_for_use"),
        "paid_review_completed_required_fields": required_review_fields,
        "warnings": warnings,
        "steps": steps,
    }


def build_current_gate_summary(
    *,
    report_path: Path,
    report: dict[str, Any],
    gates: list[dict[str, Any]],
    benchmark_evidence: list[dict[str, Any]],
    weave_agents_evidence: list[dict[str, Any]],
    existing_results_formalization: dict[str, Any],
    wandb_adoption_draft: dict[str, Any],
    paid_run_review_package: dict[str, Any],
    wandb_completion_contract: dict[str, Any],
    benchmark_progress_matrix: list[dict[str, Any]],
    nemoclaw_adoption: dict[str, Any],
    nemoclaw_installer_review: dict[str, Any],
    operator_next_steps: dict[str, Any],
    remediation_plan: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    blocking_gates = summary.get("blockers", [])
    external_action_checklist = build_external_action_checklist(
        operator_steps=operator_next_steps,
        blocking_gates=blocking_gates if isinstance(blocking_gates, list) else [],
    )
    return {
        "readiness_report_source": path_display(report_path),
        "readiness_report_schema_version": report.get("schema_version"),
        "status": report.get("status"),
        "readiness_status": report.get("status"),
        "readiness_ok": bool(report.get("ok")),
        "gate_count": summary.get("gate_count"),
        "blocker_count": summary.get("blocker_count"),
        "blocking_gates": blocking_gates,
        "required_next_actions": required_next_actions(gates),
        "benchmark_completion": benchmark_evidence,
        "weave_agents_completion": weave_agents_evidence,
        "existing_results_formalization": existing_results_formalization,
        "wandb_adoption_draft": wandb_adoption_draft,
        "paid_run_review_package": paid_run_review_package,
        "wandb_completion_contract": wandb_completion_contract,
        "benchmark_progress_matrix": benchmark_progress_matrix,
        "nemoclaw_adoption": nemoclaw_adoption,
        "nemoclaw_installer_review": nemoclaw_installer_review,
        "operator_next_steps": operator_next_steps,
        "external_action_checklist": external_action_checklist,
        "remediation_plan": remediation_plan,
        "runner_evidence": runner_evidence_summary(report),
    }


def build_manifest(
    *,
    report_path: Path,
    report: dict[str, Any],
    evidence_files: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    gates = gate_summary(report)
    benchmark_evidence = benchmark_completion_summary(report)
    weave_agents_evidence = weave_agents_completion_summary(report)
    existing_results_formalization = existing_results_formalization_summary(report)
    wandb_adoption_draft = wandb_adoption_draft_summary(report)
    paid_run_review_package = paid_run_review_package_summary(report)
    wandb_completion_contract = wandb_completion_contract_summary(
        benchmark_completion=benchmark_evidence,
        existing_results_formalization=existing_results_formalization,
        wandb_adoption_draft=wandb_adoption_draft,
        paid_run_review_package=paid_run_review_package,
    )
    benchmark_progress_matrix = benchmark_progress_matrix_summary(
        wandb_completion_contract=wandb_completion_contract,
        paid_run_review_package=paid_run_review_package,
        weave_agents_completion=weave_agents_evidence,
    )
    nemoclaw_adoption = nemoclaw_adoption_summary(report)
    nemoclaw_installer_review = nemoclaw_installer_review_summary(gates)
    remediation_plan = report.get("remediation_plan")
    if not isinstance(remediation_plan, list):
        remediation_plan = []
    operator_next_steps = operator_next_steps_summary(
        remediation_plan=remediation_plan,
        wandb_completion_contract=wandb_completion_contract,
        paid_run_review_package=paid_run_review_package,
        nemoclaw_adoption=nemoclaw_adoption,
    )
    current_gate = build_current_gate_summary(
        report_path=report_path,
        report=report,
        gates=gates,
        benchmark_evidence=benchmark_evidence,
        weave_agents_evidence=weave_agents_evidence,
        existing_results_formalization=existing_results_formalization,
        wandb_adoption_draft=wandb_adoption_draft,
        paid_run_review_package=paid_run_review_package,
        wandb_completion_contract=wandb_completion_contract,
        benchmark_progress_matrix=benchmark_progress_matrix,
        nemoclaw_adoption=nemoclaw_adoption,
        nemoclaw_installer_review=nemoclaw_installer_review,
        operator_next_steps=operator_next_steps,
        remediation_plan=remediation_plan,
    )
    return {
        "schema_version": 1,
        "bundle_version": 2,
        "generated_at": time.time(),
        "readiness_report_source": path_display(report_path),
        "readiness_report_schema_version": report.get("schema_version"),
        "status": report.get("status"),
        "readiness_status": report.get("status"),
        "readiness_ok": bool(report.get("ok")),
        "blocking_gates": summary.get("blockers", []),
        "gate_count": summary.get("gate_count"),
        "blocker_count": summary.get("blocker_count"),
        "current_gate": current_gate,
        "gates": gates,
        "benchmark_evidence": summary.get("benchmark_evidence", []),
        "files": evidence_files,
    }


def add_computed_operator_command_script_evidence(
    evidence: dict[str, dict[str, Any]],
    *,
    report_path: Path,
    report: dict[str, Any],
) -> None:
    preliminary_manifest = build_manifest(
        report_path=report_path,
        report=report,
        evidence_files=[],
    )
    current_gate = (
        preliminary_manifest.get("current_gate")
        if isinstance(preliminary_manifest.get("current_gate"), dict)
        else {}
    )
    add_operator_command_script_evidence(
        evidence,
        operator_next_steps=current_gate.get("operator_next_steps"),
    )
    add_remediation_command_script_evidence(
        evidence,
        remediation_plan=current_gate.get("remediation_plan"),
    )
    add_evidence(
        evidence,
        role="operator_execution_plan_renderer:script",
        path_value="scripts/tools/render_taiwan_operator_execution_plan.py",
    )
    add_evidence(
        evidence,
        role="operator_execution_plan_renderer:dependency_script",
        path_value=WEAVE_CONTENT_CANARY_GATE_CONTRACT_SCRIPT,
    )
    add_agentic_runner_script_evidence(evidence)


def format_bool(value: object) -> str:
    return "true" if bool(value) else "false"


def operator_execution_plan_renderer_summary(
    *,
    operator_plan_json: str,
    operator_steps: dict[str, Any],
    output_dir: str = "temp",
    approval_source_packet_json: str = "external_action_approval_packet.json",
    release_gate_json: str | None = None,
) -> dict[str, Any]:
    unresolved = operator_steps.get("unresolved_placeholder_tokens")
    if not isinstance(unresolved, list):
        unresolved = []
    tokens = [str(token) for token in unresolved if isinstance(token, str) and token]
    timestamp = "YYYYMMDDTHHMM"
    output_json = f"{output_dir}/taiwan_operator_execution_plan_{timestamp}.json"
    output_markdown = f"{output_dir}/taiwan_operator_execution_plan_{timestamp}.md"
    output_shell = f"{output_dir}/taiwan_operator_execution_plan_{timestamp}.sh"
    approval_report = (
        f"{output_dir}/taiwan_external_action_approval_REVIEWED_{timestamp}.verify.json"
    )
    command_parts = [
        "uv run python scripts/tools/render_taiwan_operator_execution_plan.py",
        f"--operator-plan-json {operator_plan_json}",
        f"--timestamp {timestamp}",
    ]
    if release_gate_json:
        command_parts.append(f"--release-gate-json {release_gate_json}")
    if "RUN_ID" in tokens:
        command_parts.append("--run-id RUN_ID")
    if "MODEL_SLUG" in tokens:
        command_parts.append("--model-slug MODEL_SLUG")
    if "WEAVE_CONTENT_CANARY_GATE" in tokens:
        command_parts.append("--weave-content-canary-gate WEAVE_CONTENT_CANARY_GATE")
    command_parts.extend(
        [
            f"--output-json {output_json}",
            f"--markdown {output_markdown}",
        ]
    )
    review_command = " ".join(command_parts)
    ready_command = " ".join(
        [
            *command_parts,
            f"--external-action-approval-source-packet-json {approval_source_packet_json}",
            f"--external-action-approval-report-json {approval_report}",
            f"--shell-script {output_shell}",
            "--require-ready",
        ]
    )
    return {
        "schema_version": 1,
        "status": "available",
        "script": "scripts/tools/render_taiwan_operator_execution_plan.py",
        "required_before_external_action": True,
        "purpose": (
            "Render operator-plan command templates with reviewed values before "
            "paid API, W&B write, NeMoClaw install/onboard, or scope-confirmation actions."
        ),
        "placeholder_tokens": tokens,
        "approval_report_json_template": approval_report,
        "approval_source_packet_json_template": approval_source_packet_json,
        "release_gate_json_template": release_gate_json or "",
        "review_command_template": review_command,
        "require_ready_command_template": ready_command,
        "expected_outputs": {
            "json": output_json,
            "markdown": output_markdown,
            "shell_script": output_shell,
        },
        "safety": {
            "executes_external_action": False,
            "writes_shell_script_only_when_placeholders_resolved": True,
            "requires_valid_external_action_approval_for_shell_script": True,
            "requires_source_packet_match_for_shell_script": True,
            "requires_command_approval_paths_match_for_shell_script": True,
            "requires_release_gate_match_for_shell_script": True,
            "requires_command_policy_validation_for_shell_script": True,
            "requires_weave_content_canary_gate_validation_for_shell_script": True,
            "requires_canary_approval_scope_match_for_shell_script": True,
        },
    }


def benchmark_progress_matrix_markdown(progress_rows: object) -> list[str]:
    lines = [
        "",
        "## Benchmark Progress Matrix",
        "",
        "| Scope | Name | Phase | Status | W&B | Weave | Runs | Review | Next |",
        "|---|---|---|---|---|---|---:|---|---|",
    ]
    if not isinstance(progress_rows, list) or not progress_rows:
        lines.append("| none |  |  |  |  |  |  |  |  |")
        return lines
    for row in progress_rows:
        if not isinstance(row, dict):
            continue
        if row.get("scope") == "release_benchmark":
            wandb_state = (
                "release"
                if row.get("release_completion_proven")
                else (
                    "standalone="
                    f"{row.get('standalone_wandb_status') or 'missing'}; "
                    "review="
                    f"{row.get('review_wandb_status') or 'missing'}"
                )
            )
            weave_state = (
                f"{row.get('weave_agents_verified_count', 0)}/"
                f"{row.get('weave_agents_entry_count', 0)}"
                if row.get("weave_agents_required")
                else "n/a"
            )
            review_state = row.get("review_wandb_status") or ""
            next_state = row.get("missing_reasons") or row.get("next_actions") or []
        elif row.get("scope") == "enabled_benchmark":
            wandb_state = row.get("wandb_completion_status") or "not_release_gated"
            weave_state = "required" if row.get("weave_agents_required") else "n/a"
            review_state = row.get("config_path") or row.get("review_path") or ""
            next_state = (
                row.get("config_errors")
                if isinstance(row.get("config_errors"), list)
                and row.get("config_errors")
                else []
            )
        else:
            wandb_state = (
                f"{row.get('wandb_completion_entry_count', 0)} entries"
                if row.get("verify_wandb_completion")
                else "n/a"
            )
            weave_state = (
                f"{row.get('weave_agents_completion_entry_count', 0)} entries"
                if row.get("verify_weave_agents")
                else "n/a"
            )
            review_state = row.get("review_path") or ""
            next_state = row.get("errors") or []
        lines.append(
            "| "
            f"{md_cell(row.get('scope'))} | "
            f"{md_cell(row.get('name'))} | "
            f"{md_cell(row.get('phase'))} | "
            f"{md_cell(row.get('status'))} | "
            f"{md_cell(wandb_state)} | "
            f"{md_cell(weave_state)} | "
            f"{md_cell(row.get('run_count'))} | "
            f"{md_cell(review_state)} | "
            f"{md_cell(next_state)} |"
        )
    return lines


EXTERNAL_ACTION_REQUIREMENTS = (
    ("requires_paid_api", "paid_api", "Paid API"),
    ("requires_wandb_access", "wandb_access", "W&B access"),
    ("requires_wandb_write", "wandb_write", "W&B write"),
    (
        "requires_third_party_acceptance",
        "third_party_acceptance",
        "Third-party acceptance",
    ),
    ("requires_nemoclaw_install", "nemoclaw_install", "NeMoClaw install"),
    ("requires_scope_confirmation", "scope_confirmation", "Scope confirmation"),
)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def build_external_action_checklist(
    *,
    operator_steps: dict[str, Any],
    blocking_gates: list[Any],
) -> dict[str, Any]:
    steps = operator_steps.get("steps")
    if not isinstance(steps, list):
        steps = []
    blocking_gate_names = [item for item in blocking_gates if isinstance(item, str)]

    items: list[dict[str, Any]] = []
    requirement_counts = {name: 0 for _, name, _ in EXTERNAL_ACTION_REQUIREMENTS}
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        requirements = [
            name
            for field, name, _ in EXTERNAL_ACTION_REQUIREMENTS
            if step.get(field) is True
        ]
        for name in requirements:
            requirement_counts[name] += 1
        commands = _string_list(step.get("commands"))
        evidence_to_produce = _string_list(step.get("evidence_to_produce"))
        warnings = _string_list(step.get("warnings"))
        gate = step.get("gate")
        items.append(
            {
                "order": step.get("order") if isinstance(step.get("order"), int) else index,
                "gate": gate,
                "status": step.get("status"),
                "blocking": isinstance(gate, str) and gate in blocking_gate_names,
                "external_action_required": bool(requirements),
                "requirements": requirements,
                "command_count": len(commands),
                "evidence_path_count": len(evidence_to_produce),
                "next_action": step.get("next_action"),
                "commands": commands,
                "evidence_to_produce": evidence_to_produce,
                "warnings": warnings,
            }
        )

    return {
        "schema_version": 1,
        "status": operator_steps.get("status", "unknown"),
        "blocking_gate_count": len(blocking_gate_names),
        "item_count": len(items),
        "external_action_item_count": sum(
            1 for item in items if item.get("external_action_required")
        ),
        "requirement_counts": requirement_counts,
        "items": items,
    }


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def approval_requirement_template(
    *,
    requirement: str,
    label: str,
    count: int,
    checklist: dict[str, Any],
) -> dict[str, Any]:
    gates = [
        item.get("gate")
        for item in checklist.get("items", [])
        if isinstance(item, dict)
        and requirement in (item.get("requirements") if isinstance(item.get("requirements"), list) else [])
    ]
    reviewer_fields = ["approved_by", "approved_at", "approval_reference"]
    if requirement == "paid_api":
        reviewer_fields.extend(["approved_budget_usd", "approved_model_scope"])
    elif requirement == "wandb_write":
        reviewer_fields.extend(["approved_wandb_entity", "approved_wandb_project"])
    elif requirement == "third_party_acceptance":
        reviewer_fields.append("third_party_terms_reviewed")
    elif requirement == "nemoclaw_install":
        reviewer_fields.extend(["installer_lock_json", "installer_sha256", "sandbox"])
    elif requirement == "scope_confirmation":
        reviewer_fields.append("scope_attestation_json")
    return {
        "requirement": requirement,
        "label": label,
        "count": count,
        "required": count > 0,
        "approval_status": "not_granted" if count > 0 else "not_required",
        "required_before_gates": gates,
        "reviewer_fields": reviewer_fields if count > 0 else [],
    }


def build_external_action_approval_packet(
    manifest: dict[str, Any],
    *,
    json_bundle_path: Path,
    markdown_bundle_path: Path,
) -> dict[str, Any]:
    current_gate = (
        manifest.get("current_gate")
        if isinstance(manifest.get("current_gate"), dict)
        else {}
    )
    checklist = (
        current_gate.get("external_action_checklist")
        if isinstance(current_gate.get("external_action_checklist"), dict)
        else {}
    )
    requirement_counts = (
        checklist.get("requirement_counts")
        if isinstance(checklist.get("requirement_counts"), dict)
        else {}
    )
    approval_requirements = [
        approval_requirement_template(
            requirement=name,
            label=label,
            count=int(requirement_counts.get(name) or 0),
            checklist=checklist,
        )
        for _, name, label in EXTERNAL_ACTION_REQUIREMENTS
    ]
    required_count = sum(1 for item in approval_requirements if item["required"])
    external_item_count = int(checklist.get("external_action_item_count") or 0)
    reviewed_packet_template = (
        "temp/taiwan_external_action_approval_REVIEWED_YYYYMMDDTHHMM.json"
    )
    reviewed_markdown_template = (
        "temp/taiwan_external_action_approval_REVIEWED_YYYYMMDDTHHMM.md"
    )
    verifier_report_template = (
        "temp/taiwan_external_action_approval_REVIEWED_YYYYMMDDTHHMM.verify.json"
    )
    return {
        "schema_version": 1,
        "generated_at": time.time(),
        "status": "pending_approval" if external_item_count else "no_external_action_required",
        "readiness_status": manifest.get("readiness_status"),
        "readiness_ok": bool(manifest.get("readiness_ok")),
        "blocking_gates": manifest.get("blocking_gates") or [],
        "source": {
            "manifest": "manifest.json",
            "operator_plan_json": (manifest.get("operator_plan") or {}).get("json"),
            "operator_plan_markdown": (manifest.get("operator_plan") or {}).get("markdown"),
            "release_gate_pointer": manifest.get("release_gate_pointer") or {},
        },
        "external_action_checklist_sha256": canonical_json_sha256(checklist),
        "external_action_checklist": checklist,
        "approval_requirement_count": len(approval_requirements),
        "required_approval_count": required_count,
        "all_required_approvals_granted": required_count == 0,
        "approval_requirements": approval_requirements,
        "approval_verifier": {
            "schema_version": 1,
            "status": "available",
            "script": EXTERNAL_ACTION_APPROVAL_PACKET_VERIFIER_SCRIPT,
            "required_before_external_action": external_item_count > 0,
            "source_packet_json": str(json_bundle_path),
            "reviewed_packet_json_template": reviewed_packet_template,
            "report_json_template": verifier_report_template,
            "command_template": (
                "uv run python "
                f"{EXTERNAL_ACTION_APPROVAL_PACKET_VERIFIER_SCRIPT} "
                f"--approval-packet-json {reviewed_packet_template} "
                f"--source-packet-json {json_bundle_path} "
                "--require-approved "
                f"--json {verifier_report_template}"
            ),
            "safety": {
                "executes_external_action": False,
                "queries_wandb": False,
                "writes_wandb": False,
                "installs_third_party": False,
                "launches_model_inference": False,
            },
        },
        "approval_template_renderer": {
            "schema_version": 1,
            "status": "available",
            "script": EXTERNAL_ACTION_APPROVAL_TEMPLATE_RENDERER_SCRIPT,
            "required_before_external_action": external_item_count > 0,
            "reviewed_packet_json_template": reviewed_packet_template,
            "reviewed_packet_markdown_template": reviewed_markdown_template,
            "command_template": (
                "uv run python "
                f"{EXTERNAL_ACTION_APPROVAL_TEMPLATE_RENDERER_SCRIPT} "
                f"--approval-packet-json {json_bundle_path} "
                f"--output-json {reviewed_packet_template} "
                f"--markdown {reviewed_markdown_template}"
            ),
            "safety": {
                "executes_external_action": False,
                "queries_wandb": False,
                "writes_wandb": False,
                "installs_third_party": False,
                "launches_model_inference": False,
            },
        },
        "execution_policy": {
            "this_packet_launches_external_actions": False,
            "requires_human_review_before_paid_api": requirement_counts.get("paid_api", 0) > 0,
            "requires_human_review_before_wandb_write": requirement_counts.get("wandb_write", 0) > 0,
            "requires_human_review_before_third_party_install": (
                requirement_counts.get("third_party_acceptance", 0) > 0
                or requirement_counts.get("nemoclaw_install", 0) > 0
            ),
            "requires_scope_attestation_before_adopting_existing_results": (
                requirement_counts.get("scope_confirmation", 0) > 0
            ),
        },
        "outputs": {
            "json": str(json_bundle_path),
            "markdown": str(markdown_bundle_path),
        },
    }


def external_action_approval_packet_markdown(packet: dict[str, Any]) -> str:
    lines = [
        "# Taiwan External Action Approval Packet",
        "",
        f"Status: `{packet.get('status')}`",
        f"Readiness status: `{packet.get('readiness_status')}`",
        f"Readiness OK: `{format_bool(packet.get('readiness_ok'))}`",
        f"All required approvals granted: `{format_bool(packet.get('all_required_approvals_granted'))}`",
        f"External action checklist SHA-256: `{packet.get('external_action_checklist_sha256')}`",
        "",
        "## Approval Requirements",
        "",
        "| Requirement | Required | Count | Approval status | Required before gates | Reviewer fields |",
        "|---|---|---:|---|---|---|",
    ]
    requirements = packet.get("approval_requirements")
    if isinstance(requirements, list) and requirements:
        for item in requirements:
            if not isinstance(item, dict):
                continue
            lines.append(
                "| "
                f"{md_cell(item.get('label'))} | "
                f"{md_cell(item.get('required'))} | "
                f"{md_cell(item.get('count'))} | "
                f"{md_cell(item.get('approval_status'))} | "
                f"{md_cell(item.get('required_before_gates'))} | "
                f"{md_cell(item.get('reviewer_fields'))} |"
            )
    else:
        lines.append("| none | False | 0 | not_required |  |  |")

    verifier = packet.get("approval_verifier")
    if isinstance(verifier, dict):
        lines.extend(
            [
                "",
                "## Approval Verifier",
                "",
                f"- Script: `{verifier.get('script') or ''}`",
                f"- Required before external action: `{format_bool(verifier.get('required_before_external_action'))}`",
                f"- Source packet: `{verifier.get('source_packet_json') or ''}`",
                f"- Reviewed packet template: `{verifier.get('reviewed_packet_json_template') or ''}`",
                f"- Report template: `{verifier.get('report_json_template') or ''}`",
                "",
                "```bash",
                str(verifier.get("command_template") or ""),
                "```",
            ]
        )

    renderer = packet.get("approval_template_renderer")
    if isinstance(renderer, dict):
        lines.extend(
            [
                "",
                "## Approval Template Renderer",
                "",
                f"- Script: `{renderer.get('script') or ''}`",
                f"- Required before external action: `{format_bool(renderer.get('required_before_external_action'))}`",
                f"- Reviewed packet template: `{renderer.get('reviewed_packet_json_template') or ''}`",
                f"- Reviewed markdown template: `{renderer.get('reviewed_packet_markdown_template') or ''}`",
                "",
                "```bash",
                str(renderer.get("command_template") or ""),
                "```",
            ]
        )

    policy = packet.get("execution_policy") if isinstance(packet.get("execution_policy"), dict) else {}
    lines.extend(
        [
            "",
            "## Execution Policy",
            "",
            "| Field | Value |",
            "|---|---|",
        ]
    )
    for key in (
        "this_packet_launches_external_actions",
        "requires_human_review_before_paid_api",
        "requires_human_review_before_wandb_write",
        "requires_human_review_before_third_party_install",
        "requires_scope_attestation_before_adopting_existing_results",
    ):
        lines.append(f"| {key} | {md_cell(policy.get(key))} |")

    lines.extend(external_action_checklist_markdown(packet.get("external_action_checklist")))
    lines.append("")
    return "\n".join(lines)


def build_operator_plan(
    manifest: dict[str, Any],
    *,
    json_bundle_path: Path,
    markdown_bundle_path: Path,
) -> dict[str, Any]:
    current_gate = (
        manifest.get("current_gate")
        if isinstance(manifest.get("current_gate"), dict)
        else {}
    )
    operator_steps = (
        current_gate.get("operator_next_steps")
        if isinstance(current_gate.get("operator_next_steps"), dict)
        else {}
    )
    release_gate_pointer = (
        manifest.get("release_gate_pointer")
        if isinstance(manifest.get("release_gate_pointer"), dict)
        else {}
    )
    blocking_gates = (
        manifest.get("blocking_gates")
        if isinstance(manifest.get("blocking_gates"), list)
        else []
    )
    external_action_checklist = build_external_action_checklist(
        operator_steps=operator_steps,
        blocking_gates=blocking_gates,
    )
    return {
        "schema_version": 1,
        "generated_at": time.time(),
        "timestamp": manifest.get("timestamp"),
        "status": operator_steps.get("status", "unknown"),
        "source_release_gate_json": release_gate_pointer.get("release_gate_json"),
        "release_gate_json": release_gate_pointer.get("release_gate_json"),
        "release_gate_status": current_gate.get("status"),
        "readiness_status": manifest.get("readiness_status"),
        "readiness_ok": bool(manifest.get("readiness_ok")),
        "readiness_report_source": manifest.get("readiness_report_source"),
        "blocking_gates": blocking_gates,
        "operator_next_steps": operator_steps,
        "operator_execution_plan_renderer": operator_execution_plan_renderer_summary(
            operator_plan_json=str(json_bundle_path),
            operator_steps=operator_steps,
            approval_source_packet_json="external_action_approval_packet.json",
            release_gate_json=str(release_gate_pointer.get("release_gate_json") or ""),
        ),
        "external_action_checklist": external_action_checklist,
        "wandb_completion_contract": current_gate.get("wandb_completion_contract") or {},
        "benchmark_progress_matrix": current_gate.get("benchmark_progress_matrix") or [],
        "wandb_adoption_draft": current_gate.get("wandb_adoption_draft") or {},
        "paid_run_review_package": current_gate.get("paid_run_review_package") or {},
        "nemoclaw_adoption": current_gate.get("nemoclaw_adoption") or {},
        "release_gate_pointer": release_gate_pointer,
        "outputs": {
            "json": str(json_bundle_path),
            "markdown": str(markdown_bundle_path),
            "manifest": "manifest.json",
            "summary": "summary.md",
            "readiness_report_source": manifest.get("readiness_report_source"),
            "release_gate_json": release_gate_pointer.get("release_gate_json"),
            "latest_pointer_json": release_gate_pointer.get("latest_pointer_json"),
            "latest_pointer_verification_json": release_gate_pointer.get(
                "latest_pointer_verification_json"
            ),
        },
    }


def operator_plan_markdown(plan: dict[str, Any]) -> str:
    operator_steps = (
        plan.get("operator_next_steps")
        if isinstance(plan.get("operator_next_steps"), dict)
        else {}
    )
    lines = [
        "# Taiwan Release Operator Plan",
        "",
        f"Status: `{plan.get('status', 'unknown')}`",
        f"Timestamp: `{plan.get('timestamp') or ''}`",
        f"Source release gate JSON: `{plan.get('source_release_gate_json') or ''}`",
        f"Release gate: `{plan.get('release_gate_status', 'unknown')}`",
        f"Readiness status: `{plan.get('readiness_status', 'unknown')}`",
        f"Readiness OK: `{format_bool(plan.get('readiness_ok'))}`",
        f"Readiness report: `{plan.get('readiness_report_source') or ''}`",
        "",
        "## Blocking Gates",
        "",
    ]
    blockers = plan.get("blocking_gates")
    if isinstance(blockers, list) and blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Requirement Counts",
            "",
            "| Requirement | Count |",
            "|---|---:|",
        ]
    )
    for label, key in [
        ("Paid API", "paid_api_step_count"),
        ("W&B access", "wandb_access_step_count"),
        ("W&B write", "wandb_write_step_count"),
        ("Third-party acceptance", "third_party_acceptance_step_count"),
        ("Scope confirmation", "scope_confirmation_step_count"),
    ]:
        value = operator_steps.get(key)
        lines.append(f"| {label} | {value if isinstance(value, int) else 0} |")

    warnings = operator_steps.get("warnings")
    if isinstance(warnings, list) and warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)

    renderer = plan.get("operator_execution_plan_renderer")
    if isinstance(renderer, dict):
        lines.extend(
            [
                "",
                "## Operator Execution Plan Renderer",
                "",
                f"Status: `{renderer.get('status', 'unknown')}`",
                f"Required before external action: `{format_bool(renderer.get('required_before_external_action'))}`",
                f"Script: `{renderer.get('script', '')}`",
                "",
                "Placeholder tokens:",
                "",
            ]
        )
        tokens = renderer.get("placeholder_tokens")
        if isinstance(tokens, list) and tokens:
            lines.extend(f"- `{token}`" for token in tokens)
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "Commands:",
                "",
                f"- Review: `{renderer.get('review_command_template', '')}`",
                f"- Placeholder-ready shell: `{renderer.get('require_ready_command_template', '')}`",
                f"- External approval source packet: `{renderer.get('approval_source_packet_json_template', '')}`",
                f"- External approval report: `{renderer.get('approval_report_json_template', '')}`",
                "",
                "Safety flags:",
                "",
            ]
        )
        safety = renderer.get("safety")
        if isinstance(safety, dict):
            for key in sorted(safety):
                lines.append(f"- `{key}`: `{format_bool(safety.get(key))}`")
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "Expected outputs:",
                "",
            ]
        )
        outputs = renderer.get("expected_outputs")
        if isinstance(outputs, dict):
            for key in ("json", "markdown", "shell_script"):
                value = outputs.get(key)
                if value:
                    lines.append(f"- {key}: `{value}`")

    lines.extend(benchmark_progress_matrix_markdown(plan.get("benchmark_progress_matrix")))

    lines.extend(external_action_checklist_markdown(plan.get("external_action_checklist")))

    lines.extend(["", "## Steps", ""])
    steps = operator_steps.get("steps")
    if not isinstance(steps, list) or not steps:
        lines.append("- none")
    else:
        for step in steps:
            if not isinstance(step, dict):
                continue
            required_labels = [
                label
                for label, key in [
                    ("paid API", "requires_paid_api"),
                    ("W&B access", "requires_wandb_access"),
                    ("W&B write", "requires_wandb_write"),
                    (
                        "third-party acceptance",
                        "requires_third_party_acceptance",
                    ),
                    ("NeMoClaw install", "requires_nemoclaw_install"),
                    ("scope confirmation", "requires_scope_confirmation"),
                ]
                if step.get(key)
            ]
            lines.extend(
                [
                    f"### {step.get('order', '?')}. {step.get('gate', 'unknown')}",
                    "",
                    f"- Status: `{step.get('status', 'unknown')}`",
                    f"- Next action: {step.get('next_action', '')}",
                    "- Requires: " + (", ".join(required_labels) or "none"),
                    "",
                    "Commands:",
                    "",
                ]
            )
            commands = step.get("commands")
            if isinstance(commands, list) and commands:
                lines.extend(f"- `{command}`" for command in commands)
            else:
                lines.append("- none")
            evidence = step.get("evidence_to_produce")
            if isinstance(evidence, list) and evidence:
                lines.extend(["", "Evidence to produce:", ""])
                lines.extend(f"- `{path}`" for path in evidence)
            step_warnings = step.get("warnings")
            if isinstance(step_warnings, list) and step_warnings:
                lines.extend(["", "Step warnings:", ""])
                lines.extend(f"- {warning}" for warning in step_warnings)
            lines.append("")

    outputs = plan.get("outputs")
    if isinstance(outputs, dict):
        lines.extend(
            [
                "## Evidence Files",
                "",
                "| Type | Path |",
                "|---|---|",
            ]
        )
        for key in [
            "json",
            "markdown",
            "manifest",
            "summary",
            "readiness_report_source",
            "release_gate_json",
            "latest_pointer_json",
            "latest_pointer_verification_json",
        ]:
            value = outputs.get(key)
            if value:
                lines.append(f"| {key} | `{value}` |")
    lines.append("")
    return "\n".join(lines)


def external_action_checklist_markdown(checklist: object) -> list[str]:
    if not isinstance(checklist, dict):
        return ["", "## External Action Checklist", "", "- none"]

    lines = [
        "",
        "## External Action Checklist",
        "",
        f"Status: `{checklist.get('status', 'unknown')}`",
        f"External action items: `{checklist.get('external_action_item_count', 0)}`",
        "",
        "| Gate | Status | Required external actions | Commands | Evidence paths |",
        "|---|---|---|---:|---:|",
    ]
    requirement_labels = {
        name: label for _, name, label in EXTERNAL_ACTION_REQUIREMENTS
    }
    items = checklist.get("items")
    if not isinstance(items, list) or not items:
        lines.append("| none |  | none | 0 | 0 |")
        return lines
    for item in items:
        if not isinstance(item, dict):
            continue
        requirements = item.get("requirements")
        requirement_text = "none"
        if isinstance(requirements, list) and requirements:
            requirement_text = ", ".join(
                requirement_labels.get(str(name), str(name)) for name in requirements
            )
        gate = item.get("gate") or "unknown"
        status = item.get("status") or "unknown"
        lines.append(
            f"| `{md_cell(gate)}` | `{md_cell(status)}` | "
            f"{md_cell(requirement_text)} | "
            f"{md_cell(item.get('command_count'))} | "
            f"{md_cell(item.get('evidence_path_count'))} |"
        )
    return lines


def add_operator_plan_files(output_dir: Path, manifest: dict[str, Any]) -> None:
    json_relative = Path("operator_plan.json")
    markdown_relative = Path("operator_plan.md")
    json_path = output_dir / json_relative
    markdown_path = output_dir / markdown_relative
    plan = build_operator_plan(
        manifest,
        json_bundle_path=json_relative,
        markdown_bundle_path=markdown_relative,
    )
    write_json(json_path, plan)
    markdown_path.write_text(operator_plan_markdown(plan), encoding="utf-8")
    manifest["operator_plan"] = {
        "json": str(json_relative),
        "markdown": str(markdown_relative),
        "schema_version": plan["schema_version"],
        "status": plan["status"],
    }
    files = manifest.get("files")
    if not isinstance(files, list):
        files = []
        manifest["files"] = files
    files.extend(
        [
            generated_bundle_file_record(
                json_path,
                bundle_path=json_relative,
                roles=["release_operator_plan", "release_operator_plan_json"],
            ),
            generated_bundle_file_record(
                markdown_path,
                bundle_path=markdown_relative,
                roles=["release_operator_plan", "release_operator_plan_markdown"],
            ),
        ]
    )


def add_external_action_approval_packet_files(output_dir: Path, manifest: dict[str, Any]) -> None:
    json_relative = Path("external_action_approval_packet.json")
    markdown_relative = Path("external_action_approval_packet.md")
    json_path = output_dir / json_relative
    markdown_path = output_dir / markdown_relative
    verifier_source = repo_path(EXTERNAL_ACTION_APPROVAL_PACKET_VERIFIER_SCRIPT)
    verifier_bundle_path = safe_bundle_path(verifier_source)
    verifier_destination = output_dir / verifier_bundle_path
    renderer_source = repo_path(EXTERNAL_ACTION_APPROVAL_TEMPLATE_RENDERER_SCRIPT)
    renderer_bundle_path = safe_bundle_path(renderer_source)
    renderer_destination = output_dir / renderer_bundle_path
    packet = build_external_action_approval_packet(
        manifest,
        json_bundle_path=json_relative,
        markdown_bundle_path=markdown_relative,
    )
    write_json(json_path, packet)
    markdown_path.write_text(
        external_action_approval_packet_markdown(packet),
        encoding="utf-8",
    )
    if verifier_source.exists() and verifier_source.is_file():
        verifier_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(verifier_source, verifier_destination)
    if renderer_source.exists() and renderer_source.is_file():
        renderer_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(renderer_source, renderer_destination)
    manifest["external_action_approval_packet"] = {
        "json": str(json_relative),
        "markdown": str(markdown_relative),
        "schema_version": packet["schema_version"],
        "status": packet["status"],
        "external_action_checklist_sha256": packet["external_action_checklist_sha256"],
        "required_approval_count": packet["required_approval_count"],
        "all_required_approvals_granted": packet["all_required_approvals_granted"],
        "approval_verifier": packet["approval_verifier"],
        "approval_template_renderer": packet["approval_template_renderer"],
    }
    files = manifest.get("files")
    if not isinstance(files, list):
        files = []
        manifest["files"] = files
    files.extend(
        [
            generated_bundle_file_record(
                json_path,
                bundle_path=json_relative,
                roles=[
                    "external_action_approval_packet",
                    "external_action_approval_packet_json",
                ],
            ),
            generated_bundle_file_record(
                markdown_path,
                bundle_path=markdown_relative,
                roles=[
                    "external_action_approval_packet",
                    "external_action_approval_packet_markdown",
                ],
            ),
            generated_bundle_file_record(
                verifier_source,
                bundle_path=verifier_bundle_path,
                roles=[
                    "external_action_approval_packet",
                    "external_action_approval_packet_verifier",
                    "external_action_approval_packet_verifier:script",
                ],
            ),
            generated_bundle_file_record(
                renderer_source,
                bundle_path=renderer_bundle_path,
                roles=[
                    "external_action_approval_packet",
                    "external_action_approval_template_renderer",
                    "external_action_approval_template_renderer:script",
                ],
            ),
        ]
    )


def md_cell(value: Any) -> str:
    if value is None:
        text = ""
    elif isinstance(value, bool):
        text = str(value)
    elif isinstance(value, (list, tuple)):
        text = ", ".join(str(item) for item in value)
    else:
        text = str(value)
    return text.replace("\n", "<br>").replace("|", "\\|")


def paid_review_wandb_entry_rows(paid_run_review_package: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gates = paid_run_review_package.get("gates")
    if not isinstance(gates, list):
        return rows
    for gate in gates:
        if not isinstance(gate, dict):
            continue
        records = gate.get("records")
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            entries = record.get("wandb_completion_entries")
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                scope = (
                    entry.get("scope_attestation")
                    if isinstance(entry.get("scope_attestation"), dict)
                    else {}
                )
                rows.append(
                    {
                        "gate": gate.get("name"),
                        "review_path": record.get("path"),
                        "phase": record.get("phase"),
                        "benchmark": entry.get("benchmark"),
                        "run_id": entry.get("run_id"),
                        "verified": entry.get("verified"),
                        "adopted_existing_result": entry.get("adopted_existing_result"),
                        "scope_attestation_valid": entry.get("scope_attestation_valid"),
                        "scope_confirmed_by": scope.get("confirmed_by"),
                        "scope_confirmed_at": scope.get("confirmed_at"),
                        "source_attestation_json": scope.get("source_attestation_json"),
                        "path": entry.get("path"),
                        "verification_error": entry.get("verification_error"),
                    }
                )
    return rows


def summary_markdown(manifest: dict[str, Any]) -> str:
    current_gate = manifest.get("current_gate") if isinstance(manifest.get("current_gate"), dict) else {}
    release_gate_pointer = (
        manifest.get("release_gate_pointer")
        if isinstance(manifest.get("release_gate_pointer"), dict)
        else {}
    )
    runner_evidence = (
        current_gate.get("runner_evidence")
        if isinstance(current_gate.get("runner_evidence"), dict)
        else {}
    )
    next_actions = current_gate.get("required_next_actions") if isinstance(current_gate, dict) else []
    benchmark_completion = current_gate.get("benchmark_completion") if isinstance(current_gate, dict) else []
    weave_agents_completion = current_gate.get("weave_agents_completion") if isinstance(current_gate, dict) else []
    existing_results_formalization = (
        current_gate.get("existing_results_formalization")
        if isinstance(current_gate, dict)
        else {}
    )
    wandb_adoption_draft = (
        current_gate.get("wandb_adoption_draft")
        if isinstance(current_gate, dict)
        else {}
    )
    paid_run_review_package = (
        current_gate.get("paid_run_review_package")
        if isinstance(current_gate, dict)
        else {}
    )
    wandb_completion_contract = (
        current_gate.get("wandb_completion_contract")
        if isinstance(current_gate, dict)
        else {}
    )
    benchmark_progress_matrix = (
        current_gate.get("benchmark_progress_matrix")
        if isinstance(current_gate, dict)
        else []
    )
    nemoclaw_adoption = current_gate.get("nemoclaw_adoption") if isinstance(current_gate, dict) else {}
    nemoclaw_installer_review = (
        current_gate.get("nemoclaw_installer_review")
        if isinstance(current_gate.get("nemoclaw_installer_review"), dict)
        else {}
    )
    nemoclaw_post_install = (
        runner_evidence.get("nemoclaw_post_install_verification")
        if isinstance(runner_evidence.get("nemoclaw_post_install_verification"), dict)
        else {}
    )
    nemoclaw_operator_docs = (
        runner_evidence.get("nemoclaw_operator_docs_verification")
        if isinstance(runner_evidence.get("nemoclaw_operator_docs_verification"), dict)
        else {}
    )
    nemoclaw_operator_docs_summary = (
        nemoclaw_operator_docs.get("summary")
        if isinstance(nemoclaw_operator_docs.get("summary"), dict)
        else {}
    )
    operator_next_steps = current_gate.get("operator_next_steps") if isinstance(current_gate, dict) else {}
    external_action_checklist = (
        current_gate.get("external_action_checklist")
        if isinstance(current_gate.get("external_action_checklist"), dict)
        else {}
    )
    operator_plan = manifest.get("operator_plan") if isinstance(manifest.get("operator_plan"), dict) else {}
    approval_packet = (
        manifest.get("external_action_approval_packet")
        if isinstance(manifest.get("external_action_approval_packet"), dict)
        else {}
    )
    lines = [
        "# Taiwan Release Evidence Bundle",
        "",
        "## Current Gate",
        "",
        f"- Readiness status: `{manifest.get('readiness_status')}`",
        f"- Readiness ok: `{manifest.get('readiness_ok')}`",
        f"- Source report: `{manifest.get('readiness_report_source')}`",
        f"- Blocking gates: `{', '.join(map(str, manifest.get('blocking_gates') or [])) or 'none'}`",
        f"- Operator plan JSON: `{operator_plan.get('json') or ''}`",
        f"- Operator plan Markdown: `{operator_plan.get('markdown') or ''}`",
        f"- External action approval packet JSON: `{approval_packet.get('json') or ''}`",
        f"- External action approval packet Markdown: `{approval_packet.get('markdown') or ''}`",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Gate count | {md_cell(current_gate.get('gate_count'))} |",
        f"| Blocker count | {md_cell(current_gate.get('blocker_count'))} |",
        f"| Runner | `{md_cell(runner_evidence.get('name'))}` |",
        f"| Runner report | `{md_cell(runner_evidence.get('report_json'))}` |",
    ]
    if release_gate_pointer:
        lines.extend(
            [
                "",
                "## Release Gate Pointer",
                "",
                "| Field | Value |",
                "| --- | --- |",
                f"| Release gate JSON | `{md_cell(release_gate_pointer.get('release_gate_json'))}` |",
                f"| Latest pointer JSON | `{md_cell(release_gate_pointer.get('latest_pointer_json'))}` |",
                f"| Pointer verification JSON | `{md_cell(release_gate_pointer.get('latest_pointer_verification_json'))}` |",
                f"| Pointer verification OK | {md_cell(release_gate_pointer.get('latest_pointer_verification_ok'))} |",
                f"| Pointer verification status | {md_cell(release_gate_pointer.get('latest_pointer_verification_status'))} |",
                f"| Pointer verification issue count | {md_cell(release_gate_pointer.get('latest_pointer_verification_issue_count'))} |",
            ]
        )
    if nemoclaw_installer_review:
        lines.extend(
            [
                "",
                "## NeMoClaw Installer Review",
                "",
                "| Field | Value |",
                "| --- | --- |",
                f"| Status | {md_cell(nemoclaw_installer_review.get('status'))} |",
                f"| OK | {md_cell(nemoclaw_installer_review.get('ok'))} |",
                f"| JSON | `{md_cell(nemoclaw_installer_review.get('path'))}` |",
                f"| Lock JSON | `{md_cell(nemoclaw_installer_review.get('lock_json'))}` |",
                f"| Lock verified | {md_cell(nemoclaw_installer_review.get('lock_verified'))} |",
                f"| Installer URL | `{md_cell(nemoclaw_installer_review.get('installer_url'))}` |",
                f"| Install ref | `{md_cell(nemoclaw_installer_review.get('install_ref'))}` |",
                f"| Expected SHA-256 | `{md_cell(nemoclaw_installer_review.get('expected_sha256'))}` |",
                f"| SHA-256 | `{md_cell(nemoclaw_installer_review.get('sha256'))}` |",
                f"| Size bytes | {md_cell(nemoclaw_installer_review.get('size_bytes'))} |",
                f"| Generated at | {md_cell(nemoclaw_installer_review.get('generated_at'))} |",
            ]
        )
    lines.extend(
        [
            "",
            "## Required Next Actions",
            "",
            "| Gate | Status | Detail | Next action |",
            "| --- | --- | --- | --- |",
        ]
    )
    if isinstance(next_actions, list) and next_actions:
        for action in next_actions:
            if not isinstance(action, dict):
                continue
            lines.append(
                "| "
                f"{md_cell(action.get('gate'))} | "
                f"{md_cell(action.get('status'))} | "
                f"{md_cell(action.get('detail'))} | "
                f"{md_cell(action.get('next_action'))} |"
            )
    else:
        lines.append("| none |  |  |  |")
    lines.extend(
        [
            "",
            "## Operator Next Steps",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Status | {md_cell((operator_next_steps or {}).get('status'))} |",
            f"| Step count | {md_cell((operator_next_steps or {}).get('step_count'))} |",
            f"| Paid API steps | {md_cell((operator_next_steps or {}).get('paid_api_step_count'))} |",
            f"| W&B access steps | {md_cell((operator_next_steps or {}).get('wandb_access_step_count'))} |",
            f"| W&B write steps | {md_cell((operator_next_steps or {}).get('wandb_write_step_count'))} |",
            f"| Third-party acceptance steps | {md_cell((operator_next_steps or {}).get('third_party_acceptance_step_count'))} |",
            f"| Scope-confirmation steps | {md_cell((operator_next_steps or {}).get('scope_confirmation_step_count'))} |",
            f"| Command-template steps | {md_cell((operator_next_steps or {}).get('command_template_step_count'))} |",
            f"| Unresolved placeholder tokens | {md_cell((operator_next_steps or {}).get('unresolved_placeholder_tokens'))} |",
            f"| NeMoClaw recommendation | {md_cell((operator_next_steps or {}).get('nemoclaw_recommendation'))} |",
            f"| NeMoClaw ready | {md_cell((operator_next_steps or {}).get('nemoclaw_ready_for_use'))} |",
            f"| Paid review completed fields | {md_cell((operator_next_steps or {}).get('paid_review_completed_required_fields'))} |",
            f"| Warnings | {md_cell((operator_next_steps or {}).get('warnings'))} |",
            "",
            "| Order | Gate | Status | Paid API | W&B access | W&B write | Third-party acceptance | Scope confirmation | Template commands | Evidence paths | Placeholder tokens | Ready without edit | Commands | Evidence outputs | Next action |",
            "| ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | --- | --- |",
        ]
    )
    operator_steps = (
        operator_next_steps.get("steps")
        if isinstance(operator_next_steps, dict)
        else []
    )
    if isinstance(operator_steps, list) and operator_steps:
        for row in operator_steps:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| "
                f"{md_cell(row.get('order'))} | "
                f"{md_cell(row.get('gate'))} | "
                f"{md_cell(row.get('status'))} | "
                f"{md_cell(row.get('requires_paid_api'))} | "
                f"{md_cell(row.get('requires_wandb_access'))} | "
                f"{md_cell(row.get('requires_wandb_write'))} | "
                f"{md_cell(row.get('requires_third_party_acceptance'))} | "
                f"{md_cell(row.get('requires_scope_confirmation'))} | "
                f"{md_cell(row.get('command_template_count'))} | "
                f"{md_cell(row.get('evidence_path_count'))} | "
                f"{md_cell(row.get('unresolved_placeholder_tokens'))} | "
                f"{md_cell(row.get('ready_to_execute_without_placeholder'))} | "
                f"{md_cell(row.get('command_count'))} | "
                f"{md_cell(row.get('evidence_to_produce'))} | "
                f"{md_cell(row.get('next_action'))} |"
            )
    else:
        lines.append("| none |  |  |  |  |  |  |  |  |  |  |")
    lines.extend(benchmark_progress_matrix_markdown(benchmark_progress_matrix))
    lines.extend(external_action_checklist_markdown(external_action_checklist))
    if approval_packet:
        approval_verifier = (
            approval_packet.get("approval_verifier")
            if isinstance(approval_packet.get("approval_verifier"), dict)
            else {}
        )
        approval_template_renderer = (
            approval_packet.get("approval_template_renderer")
            if isinstance(approval_packet.get("approval_template_renderer"), dict)
            else {}
        )
        lines.extend(
            [
                "",
                "## External Action Approval Packet",
                "",
                "| Field | Value |",
                "| --- | --- |",
                f"| Status | {md_cell(approval_packet.get('status'))} |",
                f"| JSON | `{md_cell(approval_packet.get('json'))}` |",
                f"| Markdown | `{md_cell(approval_packet.get('markdown'))}` |",
                f"| Checklist SHA-256 | `{md_cell(approval_packet.get('external_action_checklist_sha256'))}` |",
                f"| Required approval count | {md_cell(approval_packet.get('required_approval_count'))} |",
                f"| All approvals granted | {md_cell(approval_packet.get('all_required_approvals_granted'))} |",
                f"| Approval verifier script | `{md_cell(approval_verifier.get('script'))}` |",
                f"| Approval verifier command | `{md_cell(approval_verifier.get('command_template'))}` |",
                f"| Approval template renderer script | `{md_cell(approval_template_renderer.get('script'))}` |",
                f"| Approval template renderer command | `{md_cell(approval_template_renderer.get('command_template'))}` |",
            ]
        )
    lines.extend(
        [
            "",
            "## Benchmark W&B Completion",
            "",
            "| Benchmark | Required | Proven | Standalone | Review | Evidence |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    if isinstance(benchmark_completion, list) and benchmark_completion:
        for row in benchmark_completion:
            if not isinstance(row, dict):
                continue
            records = row.get("standalone_records")
            entries = row.get("review_entries")
            paths: list[str] = []
            if isinstance(records, list):
                paths.extend(str(record.get("path")) for record in records if isinstance(record, dict) and record.get("path"))
            if isinstance(entries, list):
                paths.extend(str(entry.get("path")) for entry in entries if isinstance(entry, dict) and entry.get("path"))
            lines.append(
                "| "
                f"{md_cell(row.get('benchmark'))} | "
                f"{md_cell(row.get('required'))} | "
                f"{md_cell(row.get('completion_proven'))} | "
                f"{md_cell(row.get('standalone_status'))} | "
                f"{md_cell(row.get('review_status'))} | "
                f"{md_cell(paths)} |"
            )
    else:
        lines.append("| none |  |  |  |  |  |")
    lines.extend(
        [
            "",
            "## W&B Completion Contract",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Status | {md_cell((wandb_completion_contract or {}).get('status'))} |",
            f"| Complete | {md_cell((wandb_completion_contract or {}).get('complete'))} |",
            f"| Required benchmarks | {md_cell((wandb_completion_contract or {}).get('required_benchmarks'))} |",
            f"| Required count | {md_cell((wandb_completion_contract or {}).get('required_count'))} |",
            f"| Release-proven count | {md_cell((wandb_completion_contract or {}).get('release_completion_proven_count'))} |",
            f"| Standalone W&B OK count | {md_cell((wandb_completion_contract or {}).get('standalone_completion_ok_count'))} |",
            f"| Existing formalized count | {md_cell((wandb_completion_contract or {}).get('formalized_existing_result_count'))} |",
            f"| Missing release completion | {md_cell((wandb_completion_contract or {}).get('missing_release_completion_benchmarks'))} |",
            f"| Next action count | {md_cell((wandb_completion_contract or {}).get('next_action_count'))} |",
            f"| Max age seconds | {md_cell((wandb_completion_contract or {}).get('max_age_seconds'))} |",
            "",
            "| Benchmark | Required | Status | Release proven | Standalone | Review | Existing formalized | Run IDs | Attestation templates | Preflight reports | Dry-run reports | Missing reasons | Next actions | Commands |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    contract_rows = (
        wandb_completion_contract.get("benchmarks")
        if isinstance(wandb_completion_contract, dict)
        else []
    )
    if isinstance(contract_rows, list) and contract_rows:
        for row in contract_rows:
            if not isinstance(row, dict):
                continue
            run_ids = _unique_strings(
                [
                    *(row.get("review_run_ids") if isinstance(row.get("review_run_ids"), list) else []),
                    *(row.get("standalone_run_ids") if isinstance(row.get("standalone_run_ids"), list) else []),
                    *(
                        row.get("formalized_existing_run_ids")
                        if isinstance(row.get("formalized_existing_run_ids"), list)
                        else []
                    ),
                ]
            )
            lines.append(
                "| "
                f"{md_cell(row.get('benchmark'))} | "
                f"{md_cell(row.get('required'))} | "
                f"{md_cell(row.get('status'))} | "
                f"{md_cell(row.get('release_completion_proven'))} | "
                f"{md_cell(row.get('standalone_status'))} | "
                f"{md_cell(row.get('review_status'))} | "
                f"{md_cell(row.get('formalized_existing_result'))} | "
                f"{md_cell(run_ids)} | "
                f"{md_cell(row.get('scope_attestation_template_paths'))} | "
                f"{md_cell(row.get('scope_attestation_preflight_report_paths'))} | "
                f"{md_cell(row.get('sync_dry_run_report_paths'))} | "
                f"{md_cell(row.get('missing_reasons'))} | "
                f"{md_cell(row.get('next_actions'))} | "
                f"{md_cell(row.get('recommended_commands'))} |"
            )
    else:
        lines.append("| none |  |  |  |  |  |  |  |  |  |  |  |  |  |")
    lines.extend(
        [
            "",
            "## Existing W&B Adoption Draft",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Status | {md_cell((wandb_adoption_draft or {}).get('status'))} |",
            f"| OK | {md_cell((wandb_adoption_draft or {}).get('ok'))} |",
            f"| JSON | `{md_cell((wandb_adoption_draft or {}).get('path'))}` |",
            f"| Markdown | `{md_cell((wandb_adoption_draft or {}).get('markdown_path'))}` |",
            f"| Candidate count | {md_cell((wandb_adoption_draft or {}).get('candidate_count'))} |",
            f"| Source audit JSON | `{md_cell((wandb_adoption_draft or {}).get('source_audit_json'))}` |",
            f"| Source audit SHA-256 | {md_cell((wandb_adoption_draft or {}).get('source_audit_sha256'))} |",
            f"| Attestation template count | {md_cell((wandb_adoption_draft or {}).get('scope_attestation_template_count'))} |",
            f"| Requires scope confirmation | {md_cell((wandb_adoption_draft or {}).get('requires_human_scope_confirmation'))} |",
            f"| Required human fields | {md_cell((wandb_adoption_draft or {}).get('required_human_fields'))} |",
            f"| Pending scope-confirmation candidates | {md_cell((wandb_adoption_draft or {}).get('pending_scope_confirmation_candidate_count'))} |",
            f"| Pending human field count | {md_cell((wandb_adoption_draft or {}).get('pending_human_field_count'))} |",
            f"| Pending human fields | {md_cell((wandb_adoption_draft or {}).get('pending_human_fields'))} |",
            f"| Handoff candidate count | {md_cell((wandb_adoption_draft or {}).get('operator_handoff_candidate_count'))} |",
            f"| Handoff available candidates | {md_cell((wandb_adoption_draft or {}).get('operator_handoff_available_candidate_count'))} |",
            f"| Handoff steps | {md_cell((wandb_adoption_draft or {}).get('operator_handoff_step_count'))} |",
            f"| Handoff commands | {md_cell((wandb_adoption_draft or {}).get('operator_handoff_command_count'))} |",
            f"| Handoff external-action steps | {md_cell((wandb_adoption_draft or {}).get('operator_handoff_external_action_step_count'))} |",
            f"| Handoff scope-confirmation steps | {md_cell((wandb_adoption_draft or {}).get('operator_handoff_scope_confirmation_step_count'))} |",
            f"| Handoff review-mutation steps | {md_cell((wandb_adoption_draft or {}).get('operator_handoff_review_mutation_step_count'))} |",
            f"| Handoff evidence paths | {md_cell((wandb_adoption_draft or {}).get('operator_handoff_evidence_path_count'))} |",
            f"| Handoff expected evidence paths | {md_cell(((wandb_adoption_draft or {}).get('operator_handoff') or {}).get('expected_evidence_paths'))} |",
            "",
            "| Benchmark | Model | Run ID | Source audit JSON | Source audit SHA-256 | Target review | Completion JSON | Attestation template | Render report | Preflight report | Dry-run report | Scope required | Handoff available | Handoff steps | Handoff commands | Handoff evidence paths | Handoff expected evidence paths | Render command | Preflight command | Dry-run command | Apply command |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    adoption_candidates = (
        wandb_adoption_draft.get("candidates")
        if isinstance(wandb_adoption_draft, dict)
        else []
    )
    if isinstance(adoption_candidates, list) and adoption_candidates:
        for row in adoption_candidates:
            if not isinstance(row, dict):
                continue
            handoff = (
                row.get("operator_handoff")
                if isinstance(row.get("operator_handoff"), dict)
                else {}
            )
            lines.append(
                "| "
                f"{md_cell(row.get('benchmark'))} | "
                f"{md_cell(row.get('model_slug'))} | "
                f"{md_cell(row.get('wandb_run_id'))} | "
                f"{md_cell(row.get('source_audit_json'))} | "
                f"{md_cell(row.get('source_audit_sha256'))} | "
                f"{md_cell(row.get('target_review_json'))} | "
                f"{md_cell(row.get('wandb_completion_json'))} | "
                f"{md_cell(row.get('scope_attestation_template_json'))} | "
                f"{md_cell(row.get('scope_attestation_render_report_json'))} | "
                f"{md_cell(row.get('scope_attestation_preflight_report_json'))} | "
                f"{md_cell(row.get('sync_dry_run_report_json'))} | "
                f"{md_cell(row.get('scope_attestation_required'))} | "
                f"{md_cell(handoff.get('available'))} | "
                f"{md_cell(handoff.get('step_count'))} | "
                f"{md_cell(handoff.get('command_count'))} | "
                f"{md_cell(handoff.get('evidence_path_count'))} | "
                f"{md_cell(handoff.get('expected_evidence_paths'))} | "
                f"{md_cell(row.get('scope_attestation_render_command'))} | "
                f"{md_cell(row.get('scope_attestation_preflight_command'))} | "
                f"{md_cell(row.get('sync_dry_run_command'))} | "
                f"{md_cell(row.get('sync_apply_command') or row.get('sync_command'))} |"
            )
    else:
        lines.append("| none |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |")
    lines.extend(
        [
            "",
            "### W&B Adoption Handoff Steps",
            "",
            "| Benchmark | Model | Run ID | Step | Required | External action | Scope confirmation | Review mutation | Command | Expected evidence paths |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    handoff_step_row_count = 0
    if isinstance(adoption_candidates, list) and adoption_candidates:
        for row in adoption_candidates:
            if not isinstance(row, dict):
                continue
            handoff = (
                row.get("operator_handoff")
                if isinstance(row.get("operator_handoff"), dict)
                else {}
            )
            steps = handoff.get("steps") if isinstance(handoff.get("steps"), list) else []
            for step in steps:
                if not isinstance(step, dict):
                    continue
                handoff_step_row_count += 1
                lines.append(
                    "| "
                    f"{md_cell(row.get('benchmark'))} | "
                    f"{md_cell(row.get('model_slug'))} | "
                    f"{md_cell(row.get('wandb_run_id'))} | "
                    f"{md_cell(step.get('step'))} | "
                    f"{md_cell(step.get('required'))} | "
                    f"{md_cell(step.get('requires_external_action'))} | "
                    f"{md_cell(step.get('requires_scope_confirmation'))} | "
                    f"{md_cell(step.get('mutates_review_json'))} | "
                    f"{md_cell(step.get('command'))} | "
                    f"{md_cell(step.get('expected_evidence_paths'))} |"
                )
    if handoff_step_row_count == 0:
        lines.append("| none |  |  |  |  |  |  |  |  |  |")
    adoption_unconfirmed = (
        runner_evidence.get("wandb_adoption_unconfirmed_checks")
        if isinstance(runner_evidence.get("wandb_adoption_unconfirmed_checks"), dict)
        else {}
    )
    lines.extend(
        [
            "",
            "## Existing W&B Adoption Unconfirmed-Template Checks",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Status | {md_cell(adoption_unconfirmed.get('status'))} |",
            f"| OK | {md_cell(adoption_unconfirmed.get('ok'))} |",
            f"| Record count | {md_cell(adoption_unconfirmed.get('record_count'))} |",
            f"| Output dir | `{md_cell(adoption_unconfirmed.get('output_dir'))}` |",
            "",
            "| Benchmark | Run ID | Attestation template | Preflight report | Preflight failed as expected | Sync dry-run report | Sync dry-run failed as expected | Review mutation | W&B write | Model inference |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    unconfirmed_records = adoption_unconfirmed.get("records")
    if isinstance(unconfirmed_records, list) and unconfirmed_records:
        for record in unconfirmed_records:
            if not isinstance(record, dict):
                continue
            lines.append(
                "| "
                f"{md_cell(record.get('benchmark'))} | "
                f"{md_cell(record.get('run_id'))} | "
                f"{md_cell(record.get('scope_attestation_template_json'))} | "
                f"{md_cell(record.get('preflight_report_json'))} | "
                f"{md_cell(record.get('preflight_failed_as_expected'))} | "
                f"{md_cell(record.get('sync_dry_run_report_json'))} | "
                f"{md_cell(record.get('sync_dry_run_failed_as_expected'))} | "
                f"{md_cell(record.get('will_mutate_review_json'))} | "
                f"{md_cell(record.get('will_write_wandb'))} | "
                f"{md_cell(record.get('will_launch_model_inference'))} |"
            )
    else:
        lines.append("| none |  |  |  |  |  |  |  |  |  |")
    lines.extend(
        [
            "",
            "## Weave Agents Completion",
            "",
            "| Gate | Review | Phase | Required | Entries | Verified | Proven | Evidence |",
            "| --- | --- | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    if isinstance(weave_agents_completion, list) and weave_agents_completion:
        for row in weave_agents_completion:
            if not isinstance(row, dict):
                continue
            entries = row.get("entries")
            paths: list[str] = []
            if isinstance(entries, list):
                paths.extend(str(entry.get("path")) for entry in entries if isinstance(entry, dict) and entry.get("path"))
            lines.append(
                "| "
                f"{md_cell(row.get('gate'))} | "
                f"{md_cell(row.get('review_path'))} | "
                f"{md_cell(row.get('phase'))} | "
                f"{md_cell(row.get('required'))} | "
                f"{md_cell(row.get('entry_count'))} | "
                f"{md_cell(row.get('verified_count'))} | "
                f"{md_cell(row.get('completion_proven'))} | "
                f"{md_cell(paths)} |"
            )
    else:
        lines.append("| none |  |  |  |  |  |  |  |")
    lines.extend(
        [
            "",
            "## Paid Run Review Package",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Status | {md_cell((paid_run_review_package or {}).get('status'))} |",
            f"| OK | {md_cell((paid_run_review_package or {}).get('ok'))} |",
            f"| JSON | `{md_cell((paid_run_review_package or {}).get('path'))}` |",
            f"| Markdown | `{md_cell((paid_run_review_package or {}).get('markdown_path'))}` |",
            f"| Blockers | {md_cell(((paid_run_review_package or {}).get('summary') or {}).get('blockers'))} |",
            f"| Required W&B benchmarks | {md_cell(((paid_run_review_package or {}).get('requirements') or {}).get('required_wandb_benchmarks'))} |",
            f"| Completed review fields | {md_cell(((paid_run_review_package or {}).get('review_completion_requirements') or {}).get('completed_review_required_fields'))} |",
            f"| Run fields | {md_cell(((paid_run_review_package or {}).get('review_completion_requirements') or {}).get('run_required_fields'))} |",
            f"| W&B completion max age | {md_cell(((paid_run_review_package or {}).get('review_completion_requirements') or {}).get('wandb_completion_verifier_requirements', {}).get('max_age_seconds'))} |",
            f"| Weave Agents max age | {md_cell(((paid_run_review_package or {}).get('review_completion_requirements') or {}).get('weave_agents_completion_verifier_requirements', {}).get('max_age_seconds'))} |",
            "",
            "| Gate | Status | OK | Records | Blocking records |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    review_gates = (
        paid_run_review_package.get("gates")
        if isinstance(paid_run_review_package, dict)
        else []
    )
    if isinstance(review_gates, list) and review_gates:
        for gate in review_gates:
            if not isinstance(gate, dict):
                continue
            lines.append(
                "| "
                f"{md_cell(gate.get('name'))} | "
                f"{md_cell(gate.get('status'))} | "
                f"{md_cell(gate.get('ok'))} | "
                f"{md_cell(gate.get('record_count'))} | "
                f"{md_cell(gate.get('blocking_record_count'))} |"
            )
    else:
        lines.append("| none |  |  |  |  |")
    lines.extend(
        [
            "",
            "### Paid Review W&B Completion Entries",
            "",
            "| Gate | Review | Phase | Benchmark | Run ID | Verified | Adopted existing | Scope attestation | Confirmed by | Confirmed at | Verifier JSON | Source attestation | Error |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    paid_wandb_entries = paid_review_wandb_entry_rows(
        paid_run_review_package if isinstance(paid_run_review_package, dict) else {}
    )
    if paid_wandb_entries:
        for entry in paid_wandb_entries:
            lines.append(
                "| "
                f"{md_cell(entry.get('gate'))} | "
                f"{md_cell(entry.get('review_path'))} | "
                f"{md_cell(entry.get('phase'))} | "
                f"{md_cell(entry.get('benchmark'))} | "
                f"{md_cell(entry.get('run_id'))} | "
                f"{md_cell(entry.get('verified'))} | "
                f"{md_cell(entry.get('adopted_existing_result'))} | "
                f"{md_cell(entry.get('scope_attestation_valid'))} | "
                f"{md_cell(entry.get('scope_confirmed_by'))} | "
                f"{md_cell(entry.get('scope_confirmed_at'))} | "
                f"{md_cell(entry.get('path'))} | "
                f"{md_cell(entry.get('source_attestation_json'))} | "
                f"{md_cell(entry.get('verification_error'))} |"
            )
    else:
        lines.append("| none |  |  |  |  |  |  |  |  |  |  |  |  |")
    lines.extend(
        [
            "",
            "## Existing Results Formalization",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Status | {md_cell((existing_results_formalization or {}).get('status'))} |",
            f"| OK | {md_cell((existing_results_formalization or {}).get('ok'))} |",
            f"| JSON | `{md_cell((existing_results_formalization or {}).get('path'))}` |",
            f"| Markdown | `{md_cell((existing_results_formalization or {}).get('markdown_path'))}` |",
            f"| Record count | {md_cell(((existing_results_formalization or {}).get('summary') or {}).get('record_count'))} |",
            f"| Complete local | {md_cell(((existing_results_formalization or {}).get('summary') or {}).get('complete_local_count'))} |",
            f"| Formalized in W&B | {md_cell(((existing_results_formalization or {}).get('summary') or {}).get('formalized_wandb_complete_count'))} |",
            f"| Archived non-release candidates | {md_cell(((existing_results_formalization or {}).get('summary') or {}).get('archived_complete_count'))} |",
            f"| Unformalized complete | {md_cell(((existing_results_formalization or {}).get('summary') or {}).get('unformalized_complete_count'))} |",
            f"| Partial/probe | {md_cell(((existing_results_formalization or {}).get('summary') or {}).get('partial_or_probe_count'))} |",
            "",
            "| Benchmark | Model | Status | Rows | Partial/checkpoint rows | W&B run | Result dir |",
            "| --- | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    existing_rows: list[dict[str, Any]] = []
    if isinstance(existing_results_formalization, dict):
        for field in ("formalized_records", "archived_complete_records", "unformalized_complete_records", "partial_records"):
            records = existing_results_formalization.get(field)
            if isinstance(records, list):
                existing_rows.extend(record for record in records if isinstance(record, dict))
    if existing_rows:
        for row in existing_rows:
            completion = row.get("wandb_completion") if isinstance(row.get("wandb_completion"), dict) else {}
            lines.append(
                "| "
                f"{md_cell(row.get('benchmark'))} | "
                f"{md_cell(row.get('model_slug'))} | "
                f"{md_cell(row.get('formalization_status'))} | "
                f"{md_cell(row.get('row_count'))} | "
                f"{md_cell(row.get('partial_row_count'))} | "
                f"{md_cell(completion.get('run_id'))} | "
                f"`{md_cell(row.get('result_dir'))}` |"
            )
    else:
        lines.append("| none |  |  |  |  |  |  |")
    lines.extend(
        [
            "",
            "## NeMoClaw Adoption",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Status | {md_cell((nemoclaw_adoption or {}).get('status'))} |",
            f"| OK | {md_cell((nemoclaw_adoption or {}).get('ok'))} |",
            f"| Recommendation | {md_cell(((nemoclaw_adoption or {}).get('adoption_decision') or {}).get('recommendation'))} |",
            f"| Ready for use | {md_cell(((nemoclaw_adoption or {}).get('adoption_decision') or {}).get('ready_for_use'))} |",
            f"| Scope | {md_cell(((nemoclaw_adoption or {}).get('adoption_decision') or {}).get('scope'))} |",
            f"| Sandbox | {md_cell((nemoclaw_adoption or {}).get('sandbox'))} |",
            f"| JSON | `{md_cell((nemoclaw_adoption or {}).get('path'))}` |",
            f"| Blockers | {md_cell(((nemoclaw_adoption or {}).get('summary') or {}).get('blockers'))} |",
            f"| Host prerequisites OK | {md_cell((((nemoclaw_adoption or {}).get('summary') or {}).get('setup_runtime') or {}).get('host_prerequisites_ok'))} |",
            f"| Runtime installed | {md_cell((((nemoclaw_adoption or {}).get('summary') or {}).get('setup_runtime') or {}).get('runtime_installed'))} |",
            f"| Missing required commands | {md_cell((((nemoclaw_adoption or {}).get('summary') or {}).get('setup_runtime') or {}).get('missing_required_commands'))} |",
            f"| Decision next action | {md_cell(((nemoclaw_adoption or {}).get('adoption_decision') or {}).get('next_action'))} |",
            "",
            "| Criterion | OK | Status | Detail | Next action |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    adoption_criteria = (
        nemoclaw_adoption.get("criteria")
        if isinstance(nemoclaw_adoption, dict)
        else []
    )
    if isinstance(adoption_criteria, list) and adoption_criteria:
        for row in adoption_criteria:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| "
                f"{md_cell(row.get('name'))} | "
                f"{md_cell(row.get('ok'))} | "
                f"{md_cell(row.get('status'))} | "
                f"{md_cell(row.get('detail'))} | "
                f"{md_cell(row.get('next_action'))} |"
            )
    else:
        lines.append("| none |  |  |  |  |")
    lines.extend(
        [
            "",
            "## NeMoClaw Operator Docs Verification",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Status | {md_cell(nemoclaw_operator_docs.get('status'))} |",
            f"| OK | {md_cell(nemoclaw_operator_docs.get('ok'))} |",
            f"| JSON | `{md_cell(nemoclaw_operator_docs.get('path'))}` |",
            f"| Markdown | `{md_cell(nemoclaw_operator_docs.get('markdown_path'))}` |",
            f"| Missing requirements | {md_cell(nemoclaw_operator_docs_summary.get('missing_requirements'))} |",
        ]
    )
    command_safety = (
        nemoclaw_post_install.get("command_safety")
        if isinstance(nemoclaw_post_install.get("command_safety"), dict)
        else {}
    )
    lines.extend(
        [
            "",
            "## NeMoClaw Post-Install Verification",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Status | {md_cell(nemoclaw_post_install.get('status'))} |",
            f"| OK | {md_cell(nemoclaw_post_install.get('ok'))} |",
            f"| JSON | `{md_cell(nemoclaw_post_install.get('path'))}` |",
            f"| Markdown | `{md_cell(nemoclaw_post_install.get('markdown_path'))}` |",
            f"| Generated at | {md_cell(nemoclaw_post_install.get('generated_at'))} |",
            f"| Will launch model inference | {md_cell(nemoclaw_post_install.get('will_launch_model_inference'))} |",
            f"| Will query W&B | {md_cell(nemoclaw_post_install.get('will_query_wandb'))} |",
            f"| Will install/onboard | {md_cell(nemoclaw_post_install.get('will_install_or_onboard'))} |",
            f"| Command safety OK | {md_cell(command_safety.get('ok'))} |",
            f"| Forbidden token count | {md_cell(command_safety.get('forbidden_token_count'))} |",
            f"| Forbidden exact tokens | {md_cell(command_safety.get('forbidden_tokens'))} |",
            f"| Forbidden prefixes | {md_cell(command_safety.get('forbidden_prefixes'))} |",
            f"| Forbidden markers | {md_cell(command_safety.get('forbidden_markers'))} |",
            f"| Missing required token count | {md_cell(command_safety.get('missing_required_token_count'))} |",
            f"| Missing command count | {md_cell(command_safety.get('missing_command_count'))} |",
            "",
            "### NeMoClaw Post-Install Command Safety",
            "",
            "| Step | OK | Forbidden tokens | Missing required tokens |",
            "| --- | --- | --- | --- |",
        ]
    )
    command_safety_records = command_safety.get("records")
    if isinstance(command_safety_records, list) and command_safety_records:
        for row in command_safety_records:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| "
                f"{md_cell(row.get('name'))} | "
                f"{md_cell(row.get('ok'))} | "
                f"{md_cell(row.get('forbidden_tokens'))} | "
                f"{md_cell(row.get('missing_required_tokens'))} |"
            )
    else:
        lines.append("| none |  |  |  |")
    lines.extend(
        [
            "",
            "### NeMoClaw Post-Install Steps",
            "",
            "| Step | OK | Return code | Returncode OK | Payload OK | Payload status | Timed out | Output JSON |",
            "| --- | --- | ---: | --- | --- | --- | --- | --- |",
        ]
    )
    post_install_steps = nemoclaw_post_install.get("steps")
    if isinstance(post_install_steps, list) and post_install_steps:
        for row in post_install_steps:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| "
                f"{md_cell(row.get('name'))} | "
                f"{md_cell(row.get('ok'))} | "
                f"{md_cell(row.get('returncode'))} | "
                f"{md_cell(row.get('returncode_ok'))} | "
                f"{md_cell(row.get('payload_ok'))} | "
                f"{md_cell(row.get('payload_status'))} | "
                f"{md_cell(row.get('timed_out'))} | "
                f"`{md_cell(row.get('output_json'))}` |"
            )
    else:
        lines.append("| none |  |  |  |  |  |  |  |")
    lines.extend(
        [
            "",
        "## Gates",
        "",
            "| Gate | Status | OK | Blocking | Next action |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for gate in manifest.get("gates") or []:
        lines.append(
            "| "
            f"{md_cell(gate.get('name'))} | "
            f"{md_cell(gate.get('status'))} | "
            f"{md_cell(gate.get('ok'))} | "
            f"{md_cell(gate.get('blocking'))} | "
            f"{md_cell(gate.get('next_action'))} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "| Source | Bundle path | Present | Roles |",
            "| --- | --- | --- | --- |",
        ]
    )
    for record in manifest.get("files") or []:
        roles = ", ".join(record.get("roles") or [])
        lines.append(
            f"| `{record.get('source_path')}` | `{record.get('bundle_path')}` | {record.get('exists')} | {roles} |"
        )
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fail-on-not-ready", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report_path = repo_path(args.readiness_report)
    report = read_json(report_path)
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    output_dir = repo_path(args.output_dir or DEFAULT_OUTPUT_ROOT / f"bundle_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence = collect_evidence(report_path, report)
    add_computed_operator_command_script_evidence(
        evidence,
        report_path=report_path,
        report=report,
    )
    evidence_files = copy_evidence_files(evidence, output_dir=output_dir)
    manifest = build_manifest(
        report_path=report_path,
        report=report,
        evidence_files=evidence_files,
    )
    add_operator_plan_files(output_dir, manifest)
    add_external_action_approval_packet_files(output_dir, manifest)
    summary_path = output_dir / "summary.md"
    summary_path.write_text(summary_markdown(manifest), encoding="utf-8")
    files = manifest.get("files")
    if not isinstance(files, list):
        files = []
        manifest["files"] = files
    files.append(
        generated_bundle_file_record(
            summary_path,
            bundle_path=Path("summary.md"),
            roles=["release_summary", "release_summary_markdown"],
        )
    )
    write_json(output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "ok": bool(manifest["readiness_ok"]),
                "output_dir": str(output_dir),
                "manifest": str(output_dir / "manifest.json"),
                "operator_plan": manifest.get("operator_plan"),
                "external_action_approval_packet": manifest.get(
                    "external_action_approval_packet"
                ),
            },
            ensure_ascii=False,
        )
    )
    if args.fail_on_not_ready and not manifest["readiness_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
