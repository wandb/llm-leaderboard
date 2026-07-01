#!/usr/bin/env python3
"""
Run Taiwan full evaluations sequentially from the generated model configs.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

from external_action_approval_checks import (
    approval_results as normalized_approval_results,
    extract_paid_api_approval,
    validate_approval_results,
)
from prepare_taiwan_full_eval_configs import (
    AGENTIC_DENIED_ARGUMENT_PATTERNS,
    AGENTIC_DENIED_TOOLS,
    CONFIG_DIR,
    DEFAULT_MANIFEST,
    PHASE_CHOICES,
    generate_configs,
)
from weave_content_canary_gate_contract import (
    weave_content_canary_gate_contract_issues,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
WANDB_COMPLETION_VERIFY_RUNNER = REPO_ROOT / "scripts" / "tools" / "verify_taiwan_wandb_completion.py"
WEAVE_AGENTS_VERIFY_RUNNER = REPO_ROOT / "scripts" / "tools" / "verify_taiwan_weave_agents.py"
DEFAULT_WEAVE_AGENT_NAME = "nejumi-taiwan-openclaw"
DEFAULT_WANDB_VERIFY_BENCHMARKS = {
    "full": ["agentic_math", "agentic_swe", "taiwan_full"],
    "agentic": ["agentic_math", "agentic_swe"],
    "agentic_aggregate": ["taiwan_full"],
}
DEFAULT_EXPECTED_TOTALS = {
    "agentic_math": 100,
    "agentic_swe": 80,
}
CANONICAL_NEMOCLAW_OPENCLAW_CONFIG_PATH = "/sandbox/.openclaw/openclaw.json"
BENCHMARK_RUN_FLAG_EXPECTATIONS = {
    "agentic_math": "run.agentic_math",
    "agentic_swe": "run.swebench_pro",
    "taiwan_full": "run.aggregate_taiwan",
}
BENCHMARK_MODEL_CONFIG_EXPECTATIONS = {
    "agentic_math": "agentic_math.openclaw_model",
    "agentic_swe": "swebench_pro.openclaw_model",
}
BENCHMARK_NEMOCLAW_CONFIG_EXPECTATIONS = {
    "agentic_math": [
        "agentic_math.nemoclaw_sandbox",
        "agentic_math.nemoclaw_openclaw_config_path",
        "agentic_math.use_task_agent",
        "agentic_math.deny_tool",
        "agentic_math.deny_argument_pattern",
    ],
    "agentic_swe": [
        "swebench_pro.nemoclaw_sandbox",
        "swebench_pro.nemoclaw_openclaw_config_path",
        "swebench_pro.nemoclaw_checkout_transfer_mode",
        "swebench_pro.nemoclaw_checkout_sandbox_root",
        "swebench_pro.deny_tool",
        "swebench_pro.deny_argument_pattern",
    ],
}
AGENTIC_GENERATION_PHASES = {"agentic", "full"}
DEFAULT_WEAVE_CONTENT_CANARY_MAX_AGE_SECONDS = 24 * 60 * 60
WEAVE_AGENTS_COMPLETION_SCHEMA_VERSION = 1
REQUIRED_NEMOCLAW_AGENTIC_DENIED_TOOLS = set(AGENTIC_DENIED_TOOLS)
REQUIRED_NEMOCLAW_AGENTIC_DENIED_ARGUMENT_PATTERNS = set(
    AGENTIC_DENIED_ARGUMENT_PATTERNS
)
REQUIRED_NEMOCLAW_AGENTIC_ALLOWED_LOCAL_TOOLS = {"exec"}
AGENTIC_PRODUCTION_EVIDENCE_REQUIREMENTS = {
    "--require-nemoclaw-agentic-config": "Agentic Math/SWE generated configs must route through NeMoClaw with deny-policy guards.",
    "--agentic-math-nemoclaw-openclaw-config-path": "Agentic Math must bind NeMoClaw to the reviewed OpenClaw config inside the sandbox.",
    "--swebench-pro-nemoclaw-openclaw-config-path": "SWE-Bench Pro must bind NeMoClaw to the reviewed OpenClaw config inside the sandbox.",
    "--require-weave-content-canary": "A fresh native Weave content canary must pass before paid agentic execution.",
    "--weave-content-canary-gate": "The paid run must be bound to the reviewed content-canary gate JSON.",
    "--verify-wandb-completion": "W&B scalar/table/artifact completion verification is mandatory.",
    "--verify-weave-agents": "Native W&B Weave Agents trace verification is mandatory.",
    "--wandb-run-id-prefix": "W&B run IDs must be explicit so W&B and Weave evidence can be scoped to this run.",
    "--weave-agents-require-content": "Weave Agents verification must require visible conversation content.",
    "--weave-agents-require-tool-span": "Weave Agents verification must require tool spans.",
    "--weave-agents-require-tool-content": "Weave Agents verification must require tool content.",
    "--weave-agents-require-usage": "Weave Agents verification must require usage metadata for cost/accountability review.",
}


def nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def canonical_nemoclaw_openclaw_config_path(value: object) -> bool:
    return value == CANONICAL_NEMOCLAW_OPENCLAW_CONFIG_PATH


def load_env_file(env: dict[str, str], path: Path) -> dict[str, str]:
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in env:
            env[key] = value.strip().strip('"').strip("'")
    return env


def config_arg(path: Path) -> str:
    try:
        return str(path.relative_to(CONFIG_DIR))
    except ValueError:
        return str(path.resolve())


def resolve_weave_conversation_id_contains(
    template: str | None,
    *,
    wandb_run_id: str,
) -> str:
    if not wandb_run_id:
        raise ValueError("wandb_run_id is required for Weave Agents verification")
    if not template:
        return wandb_run_id
    resolved = template.replace("{wandb_run_id}", wandb_run_id)
    if wandb_run_id not in resolved:
        raise ValueError(
            "--weave-agents-conversation-id-contains must include the W&B run id "
            "or the {wandb_run_id} placeholder when --verify-weave-agents is enabled"
        )
    return resolved


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> tuple[dict | None, str | None]:
    if not path.exists():
        return None, f"{path} does not exist"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{path} is not readable JSON: {exc}"
    if not isinstance(payload, dict):
        return None, f"{path} is not a JSON object"
    return payload, None


def numeric_timestamp(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def path_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def model_identifier_values(config: dict, *, phase: str) -> list[str]:
    keys = ["model.pretrained_model_name_or_path"]
    if phase in AGENTIC_GENERATION_PHASES:
        keys.extend(
            [
                "agentic_math.openclaw_model",
                "swebench_pro.openclaw_model",
            ]
        )
    values: list[str] = []
    api = config.get("api")
    for key in keys:
        present, value = config_lookup(config, key)
        if present and isinstance(value, str) and value.strip():
            model_id = value.strip()
            values.append(model_id)
            if api == "openai_responses" and "/" not in model_id:
                values.append(f"openai-direct/{model_id}")
            if model_id.startswith("openai-direct/"):
                values.append(model_id.removeprefix("openai-direct/"))
    return list(dict.fromkeys(values))


def collect_selected_config_model_bindings(
    config_paths: list[Path],
    *,
    phase: str,
) -> list[dict[str, object]]:
    bindings: list[dict[str, object]] = []
    for path in config_paths:
        try:
            loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        except Exception as exc:  # pragma: no cover - defensive path for operator evidence
            bindings.append(
                {
                    "config": config_arg(path),
                    "identifiers": [],
                    "error": f"config model identifiers could not be loaded: {exc}",
                }
            )
            continue
        if not isinstance(loaded, dict):
            bindings.append(
                {
                    "config": config_arg(path),
                    "identifiers": [],
                    "error": "config is not an object",
                }
            )
            continue
        bindings.append(
            {
                "config": config_arg(path),
                "identifiers": model_identifier_values(loaded, phase=phase),
            }
        )
    return bindings


def build_nemoclaw_agentic_config_guard(
    config_paths: list[Path],
    *,
    phase: str,
    required: bool,
) -> dict[str, object]:
    enforced = bool(required and phase in AGENTIC_GENERATION_PHASES)
    records: list[dict[str, object]] = []
    errors: list[str] = []
    for path in config_paths:
        record: dict[str, object] = {
            "config": config_arg(path),
            "ok": True,
            "errors": [],
        }
        try:
            loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        except Exception as exc:  # pragma: no cover - defensive operator evidence
            issue = f"config could not be loaded: {exc}"
            record["ok"] = False
            record["errors"] = [issue]
            records.append(record)
            errors.append(f"{config_arg(path)}: {issue}")
            continue
        if not isinstance(loaded, dict):
            issue = "config is not an object"
            record["ok"] = False
            record["errors"] = [issue]
            records.append(record)
            errors.append(f"{config_arg(path)}: {issue}")
            continue

        def lookup(key: str) -> object:
            _, value = config_lookup(loaded, key)
            return value

        def string_list(value: object) -> list[str] | None:
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                return None
            return value

        def require_string_superset(
            *,
            section: str,
            field: str,
            observed_value: object,
            required_values: set[str],
        ) -> None:
            observed = string_list(observed_value)
            if observed is None:
                issues.append(f"{section}.{field} must be a string list")
                return
            missing = sorted(required_values - set(observed))
            if missing:
                issues.append(
                    f"{section}.{field} missing required values: {', '.join(missing)}"
                )

        def tool_name_matches_pattern(tool_name: str, pattern: str) -> bool:
            tool_name_norm = tool_name.lower()
            pattern_norm = pattern.lower()
            if pattern_norm.startswith("re:"):
                return re.search(pattern_norm[3:], tool_name_norm) is not None
            if "*" in pattern_norm or "?" in pattern_norm:
                return fnmatch.fnmatch(tool_name_norm, pattern_norm)
            return tool_name_norm == pattern_norm

        def local_exec_blocking_patterns(observed_value: object) -> list[str]:
            observed = string_list(observed_value)
            if observed is None:
                return []
            return sorted(
                pattern
                for pattern in observed
                if any(
                    tool_name_matches_pattern(tool_name, pattern)
                    for tool_name in REQUIRED_NEMOCLAW_AGENTIC_ALLOWED_LOCAL_TOOLS
                )
            )

        def require_local_exec_allowed(section: str, observed_value: object) -> list[str]:
            conflicts = local_exec_blocking_patterns(observed_value)
            if conflicts:
                issues.append(
                    f"{section}.deny_tool must not block local OpenClaw exec tool: "
                    f"{', '.join(conflicts)}"
                )
            return conflicts

        issues: list[str] = []
        run_agentic_math = lookup("run.agentic_math") is True
        run_swebench_pro = lookup("run.swebench_pro") is True
        math_sandbox = lookup("agentic_math.nemoclaw_sandbox")
        math_config_path = lookup("agentic_math.nemoclaw_openclaw_config_path")
        math_use_task_agent = lookup("agentic_math.use_task_agent")
        math_deny_tools = lookup("agentic_math.deny_tool")
        math_deny_argument_patterns = lookup("agentic_math.deny_argument_pattern")
        swe_sandbox = lookup("swebench_pro.nemoclaw_sandbox")
        swe_config_path = lookup("swebench_pro.nemoclaw_openclaw_config_path")
        swe_transfer_mode = lookup("swebench_pro.nemoclaw_checkout_transfer_mode")
        swe_checkout_root = lookup("swebench_pro.nemoclaw_checkout_sandbox_root")
        swe_deny_tools = lookup("swebench_pro.deny_tool")
        swe_deny_argument_patterns = lookup("swebench_pro.deny_argument_pattern")

        record.update(
            {
                "run_agentic_math": run_agentic_math,
                "run_swebench_pro": run_swebench_pro,
                "agentic_math_nemoclaw_sandbox": math_sandbox if isinstance(math_sandbox, str) else "",
                "agentic_math_nemoclaw_openclaw_config_path": (
                    math_config_path if isinstance(math_config_path, str) else ""
                ),
                "agentic_math_use_task_agent": math_use_task_agent,
                "agentic_math_deny_tool": string_list(math_deny_tools) or [],
                "agentic_math_local_exec_blocking_patterns": local_exec_blocking_patterns(
                    math_deny_tools
                ),
                "agentic_math_local_exec_allowed": not local_exec_blocking_patterns(
                    math_deny_tools
                ),
                "agentic_math_deny_argument_pattern": (
                    string_list(math_deny_argument_patterns) or []
                ),
                "swebench_pro_nemoclaw_sandbox": swe_sandbox if isinstance(swe_sandbox, str) else "",
                "swebench_pro_nemoclaw_openclaw_config_path": (
                    swe_config_path if isinstance(swe_config_path, str) else ""
                ),
                "swebench_pro_nemoclaw_checkout_transfer_mode": (
                    swe_transfer_mode if isinstance(swe_transfer_mode, str) else ""
                ),
                "swebench_pro_nemoclaw_checkout_sandbox_root": (
                    swe_checkout_root if isinstance(swe_checkout_root, str) else ""
                ),
                "swebench_pro_deny_tool": string_list(swe_deny_tools) or [],
                "swebench_pro_local_exec_blocking_patterns": local_exec_blocking_patterns(
                    swe_deny_tools
                ),
                "swebench_pro_local_exec_allowed": not local_exec_blocking_patterns(
                    swe_deny_tools
                ),
                "swebench_pro_deny_argument_pattern": (
                    string_list(swe_deny_argument_patterns) or []
                ),
            }
        )
        if enforced:
            if not run_agentic_math:
                issues.append("run.agentic_math must be true")
            if not run_swebench_pro:
                issues.append("run.swebench_pro must be true")
            if not nonempty_string(math_sandbox):
                issues.append("agentic_math.nemoclaw_sandbox must be set")
            if not canonical_nemoclaw_openclaw_config_path(math_config_path):
                issues.append(
                    "agentic_math.nemoclaw_openclaw_config_path must be "
                    f"{CANONICAL_NEMOCLAW_OPENCLAW_CONFIG_PATH}"
                )
            if math_use_task_agent is False:
                issues.append("agentic_math.use_task_agent must not be false")
            require_string_superset(
                section="agentic_math",
                field="deny_tool",
                observed_value=math_deny_tools,
                required_values=REQUIRED_NEMOCLAW_AGENTIC_DENIED_TOOLS,
            )
            require_local_exec_allowed("agentic_math", math_deny_tools)
            require_string_superset(
                section="agentic_math",
                field="deny_argument_pattern",
                observed_value=math_deny_argument_patterns,
                required_values=REQUIRED_NEMOCLAW_AGENTIC_DENIED_ARGUMENT_PATTERNS,
            )
            if not nonempty_string(swe_sandbox):
                issues.append("swebench_pro.nemoclaw_sandbox must be set")
            if not canonical_nemoclaw_openclaw_config_path(swe_config_path):
                issues.append(
                    "swebench_pro.nemoclaw_openclaw_config_path must be "
                    f"{CANONICAL_NEMOCLAW_OPENCLAW_CONFIG_PATH}"
                )
            if not (
                swe_transfer_mode == "copy"
                or nonempty_string(swe_checkout_root)
            ):
                issues.append(
                    "swebench_pro must set nemoclaw_checkout_transfer_mode=copy "
                    "or nemoclaw_checkout_sandbox_root"
                )
            require_string_superset(
                section="swebench_pro",
                field="deny_tool",
                observed_value=swe_deny_tools,
                required_values=REQUIRED_NEMOCLAW_AGENTIC_DENIED_TOOLS,
            )
            require_local_exec_allowed("swebench_pro", swe_deny_tools)
            require_string_superset(
                section="swebench_pro",
                field="deny_argument_pattern",
                observed_value=swe_deny_argument_patterns,
                required_values=REQUIRED_NEMOCLAW_AGENTIC_DENIED_ARGUMENT_PATTERNS,
            )
        record["ok"] = not issues
        record["errors"] = issues
        records.append(record)
        for issue in issues:
            errors.append(f"{config_arg(path)}: {issue}")

    return {
        "required": bool(required),
        "enforced": enforced,
        "phase": phase,
        "ok": not errors,
        "errors": errors,
        "records": records,
    }


def build_agentic_production_evidence_guard(
    args: argparse.Namespace,
    *,
    phase: str,
    will_call_paid_model_api: bool,
) -> dict[str, object]:
    enforced = bool(will_call_paid_model_api and phase in AGENTIC_GENERATION_PHASES)
    math_config_path = getattr(args, "agentic_math_nemoclaw_openclaw_config_path", None)
    swe_config_path = getattr(args, "swebench_pro_nemoclaw_openclaw_config_path", None)
    observations = {
        "--require-nemoclaw-agentic-config": bool(args.require_nemoclaw_agentic_config),
        "--agentic-math-nemoclaw-openclaw-config-path": nonempty_string(
            math_config_path
        ),
        "--swebench-pro-nemoclaw-openclaw-config-path": nonempty_string(
            swe_config_path
        ),
        "--require-weave-content-canary": bool(args.require_weave_content_canary),
        "--weave-content-canary-gate": bool(args.weave_content_canary_gate),
        "--verify-wandb-completion": bool(args.verify_wandb_completion),
        "--verify-weave-agents": bool(args.verify_weave_agents),
        "--wandb-run-id-prefix": nonempty_string(args.wandb_run_id_prefix),
        "--weave-agents-require-content": not bool(args.weave_agents_no_require_content),
        "--weave-agents-require-tool-span": bool(args.weave_agents_require_tool_span),
        "--weave-agents-require-tool-content": bool(args.weave_agents_require_tool_content),
        "--weave-agents-require-usage": bool(args.weave_agents_require_usage),
    }
    missing = [flag for flag, present in observations.items() if not present]
    invalid_values = {
        flag: value
        for flag, value in {
            "--agentic-math-nemoclaw-openclaw-config-path": math_config_path,
            "--swebench-pro-nemoclaw-openclaw-config-path": swe_config_path,
        }.items()
        if nonempty_string(value)
        and not canonical_nemoclaw_openclaw_config_path(value)
    }
    errors = [
        f"{flag} is required for paid {phase} execution: {AGENTIC_PRODUCTION_EVIDENCE_REQUIREMENTS[flag]}"
        for flag in missing
    ] if enforced else []
    if enforced:
        errors.extend(
            f"{flag} must be {CANONICAL_NEMOCLAW_OPENCLAW_CONFIG_PATH} for paid {phase} execution"
            for flag in sorted(invalid_values)
        )
    return {
        "required": phase in AGENTIC_GENERATION_PHASES,
        "enforced": enforced,
        "phase": phase,
        "ok": not errors,
        "missing_flags": missing if enforced else [],
        "invalid_values": invalid_values if enforced else {},
        "errors": errors,
        "observations": observations,
        "requirements": AGENTIC_PRODUCTION_EVIDENCE_REQUIREMENTS,
    }


def budget_target_models(payload: dict) -> list[str]:
    values: list[str] = []
    target_model = payload.get("target_model")
    if isinstance(target_model, str) and target_model.strip():
        values.append(target_model.strip())
    target_models = payload.get("target_models")
    if isinstance(target_models, list):
        for item in target_models:
            if isinstance(item, str) and item.strip():
                values.append(item.strip())
    return list(dict.fromkeys(values))


def build_pre_run_budget_estimate_record(
    budget_path: Path | None,
    *,
    required_before_paid_execution: bool,
    selected_config_model_bindings: list[dict[str, object]] | None = None,
) -> dict:
    selected_config_model_bindings = selected_config_model_bindings or []
    record = {
        "required_before_paid_execution": bool(required_before_paid_execution),
        "path": str(budget_path) if budget_path else "",
        "present": False,
        "valid": False,
        "sha256": "",
        "schema_version": None,
        "generated_at": "",
        "target_model": "",
        "target_models": [],
        "selected_config_model_bindings": selected_config_model_bindings,
        "selected_model_identifiers": [],
        "target_model_matches_selected_config": False,
        "price_per_million_tokens": {},
        "estimated_total_usd": {},
        "pricing_source_url": "",
        "pricing_note": "",
        "provider_dashboard_authoritative": True,
        "errors": [],
    }
    if budget_path is None:
        if required_before_paid_execution:
            record["errors"].append("pre-run budget estimate path is required before paid execution")
        return record

    payload, error = load_json_object(budget_path)
    if payload is None:
        record["errors"].append(error or "pre-run budget estimate JSON could not be loaded")
        return record

    record["present"] = True
    try:
        record["sha256"] = sha256_file(budget_path)
    except OSError as exc:
        record["errors"].append(f"pre-run budget estimate sha256 failed: {exc}")

    total = payload.get("estimated_total_usd")
    price = payload.get("price_per_million_tokens")
    target_model = payload.get("target_model")
    if not isinstance(total, dict):
        record["errors"].append("estimated_total_usd is missing or not an object")
        total = {}
    else:
        for key in ("low", "mid", "high"):
            if not isinstance(total.get(key), (int, float)):
                record["errors"].append(f"estimated_total_usd.{key} is missing or not numeric")
    if not isinstance(price, dict):
        record["errors"].append("price_per_million_tokens is missing or not an object")
        price = {}
    if not isinstance(target_model, str) or not target_model:
        record["errors"].append("target_model is missing or not a string")
        target_model = ""
    for category in ("agentic_math", "swebench_pro"):
        category_payload = payload.get(category)
        if category_payload is None:
            continue
        if not isinstance(category_payload, dict):
            record["errors"].append(f"{category} is present but not an object")
            continue
        historical_records = category_payload.get("historical_records")
        if not isinstance(historical_records, int) or historical_records <= 0:
            record["errors"].append(
                f"{category}.historical_records must be positive when {category} evidence is present"
            )
        estimate = category_payload.get("estimate_usd")
        if not isinstance(estimate, dict):
            record["errors"].append(f"{category}.estimate_usd is missing or not an object")
            continue
        for key in ("low", "mid", "high"):
            if not isinstance(estimate.get(key), (int, float)):
                record["errors"].append(f"{category}.estimate_usd.{key} is missing or not numeric")
    target_models = budget_target_models(payload)
    selected_identifiers: list[str] = []
    unmatched_configs: list[str] = []
    for binding in selected_config_model_bindings:
        identifiers = [
            item
            for item in binding.get("identifiers", [])
            if isinstance(item, str) and item
        ] if isinstance(binding, dict) else []
        selected_identifiers.extend(identifiers)
        if identifiers and target_models and set(identifiers).isdisjoint(target_models):
            unmatched_configs.append(str(binding.get("config") or ""))
        elif identifiers and not target_models:
            unmatched_configs.append(str(binding.get("config") or ""))
    selected_identifiers = list(dict.fromkeys(selected_identifiers))
    if selected_config_model_bindings:
        if not target_models:
            record["errors"].append("target_model or target_models must contain at least one model identifier")
        if unmatched_configs:
            record["errors"].append(
                "budget target_model(s) do not match selected config model identifiers: "
                + ", ".join(item for item in unmatched_configs if item)
            )

    record.update(
        {
            "schema_version": payload.get("schema_version"),
            "generated_at": str(payload.get("generated_at") or ""),
            "target_model": target_model,
            "target_models": target_models,
            "selected_model_identifiers": selected_identifiers,
            "target_model_matches_selected_config": (
                bool(selected_config_model_bindings) and not unmatched_configs and bool(target_models)
            ),
            "price_per_million_tokens": price,
            "estimated_total_usd": total,
            "pricing_source_url": str(payload.get("pricing_source_url") or ""),
            "pricing_note": str(payload.get("pricing_note") or ""),
        }
    )
    record["valid"] = not record["errors"]
    return record


def build_external_action_approval_record(
    approval_report_path: Path | None,
    *,
    required_before_external_action: bool,
    expected_source_packet_path: Path | None = None,
) -> dict:
    record = {
        "required_before_external_action": bool(required_before_external_action),
        "path": str(approval_report_path) if approval_report_path else "",
        "present": False,
        "valid": False,
        "sha256": "",
        "schema_version": None,
        "status": "",
        "generated_at": "",
        "approval_packet_json": "",
        "external_action_checklist_sha256": "",
        "required_approval_count": None,
        "granted_approval_count": None,
        "all_required_approvals_granted": False,
        "source_binding": {},
        "expected_source_packet_json": str(expected_source_packet_path)
        if expected_source_packet_path
        else "",
        "expected_source_packet_sha256": "",
        "source_packet_path_matches_expected": False,
        "source_packet_sha256_matches_expected": False,
        "will_execute_external_actions": None,
        "approval_results": [],
        "approval_results_validation": {},
        "paid_api_approved_budget_usd": None,
        "paid_api_minimum_approved_budget_usd": None,
        "paid_api_minimum_approved_budget_source": "",
        "paid_api_approved_model_scope": "",
        "errors": [],
    }
    if required_before_external_action and expected_source_packet_path is None:
        record["errors"].append(
            "external-action approval source packet is required before external execution"
        )
    if expected_source_packet_path is not None:
        if not expected_source_packet_path.exists():
            record["errors"].append(
                "external-action approval source packet does not exist: "
                f"{expected_source_packet_path}"
            )
        else:
            try:
                record["expected_source_packet_sha256"] = sha256_file(
                    expected_source_packet_path
                )
            except OSError as exc:
                record["errors"].append(
                    f"external-action approval source packet sha256 failed: {exc}"
                )
    if approval_report_path is None:
        if required_before_external_action:
            record["errors"].append(
                "external-action approval verifier report is required before external execution"
            )
        return record

    payload, error = load_json_object(approval_report_path)
    if payload is None:
        record["errors"].append(error or "external-action approval report JSON could not be loaded")
        return record

    record["present"] = True
    try:
        record["sha256"] = sha256_file(approval_report_path)
    except OSError as exc:
        record["errors"].append(f"external-action approval report sha256 failed: {exc}")

    source_binding = (
        payload.get("source_binding")
        if isinstance(payload.get("source_binding"), dict)
        else {}
    )
    source_errors = source_binding.get("errors")
    if not isinstance(source_errors, list):
        source_errors = ["source_binding.errors is missing or not a list"]

    required_count = payload.get("required_approval_count")
    granted_count = payload.get("granted_approval_count")
    approval_results = normalized_approval_results(payload)
    approval_results_validation = validate_approval_results(payload)
    paid_api = extract_paid_api_approval(payload)
    record.update(
        {
            "schema_version": payload.get("schema_version"),
            "status": str(payload.get("status") or ""),
            "generated_at": payload.get("generated_at") or "",
            "approval_packet_json": str(payload.get("approval_packet_json") or ""),
            "external_action_checklist_sha256": str(
                payload.get("external_action_checklist_sha256") or ""
            ),
            "required_approval_count": required_count,
            "granted_approval_count": granted_count,
            "all_required_approvals_granted": bool(
                payload.get("all_required_approvals_granted")
            ),
            "source_binding": source_binding,
            "will_execute_external_actions": payload.get("will_execute_external_actions"),
            "approval_results": approval_results,
            "approval_results_validation": approval_results_validation,
            "paid_api_approved_budget_usd": paid_api.get("approved_budget_usd"),
            "paid_api_minimum_approved_budget_usd": paid_api.get(
                "minimum_approved_budget_usd"
            ),
            "paid_api_minimum_approved_budget_source": (
                paid_api.get("minimum_approved_budget_source") or ""
            ),
            "paid_api_approved_model_scope": paid_api.get("approved_model_scope") or "",
        }
    )
    record["errors"].extend(
        str(error)
        for error in approval_results_validation.get("errors", [])
        if isinstance(error, str)
    )

    if payload.get("schema_version") != 1:
        record["errors"].append("schema_version must be 1")
    if payload.get("ok") is not True:
        record["errors"].append("ok must be true")
    if payload.get("status") != "approved":
        record["errors"].append("status must be approved")
    if not isinstance(required_count, int) or required_count <= 0:
        record["errors"].append("required_approval_count must be a positive integer")
    if not isinstance(granted_count, int):
        record["errors"].append("granted_approval_count must be an integer")
    elif isinstance(required_count, int) and granted_count != required_count:
        record["errors"].append("granted_approval_count must equal required_approval_count")
    if payload.get("all_required_approvals_granted") is not True:
        record["errors"].append("all_required_approvals_granted must be true")
    if payload.get("will_execute_external_actions") is not False:
        record["errors"].append("will_execute_external_actions must be false")
    if source_binding.get("bound") is not True:
        record["errors"].append("source_binding.bound must be true")
    if source_binding.get("source_packet_readable") is not True:
        record["errors"].append("source_binding.source_packet_readable must be true")
    source_sha = str(source_binding.get("source_approval_packet_sha256") or "")
    if len(source_sha) != 64 or any(char not in "0123456789abcdef" for char in source_sha):
        record["errors"].append(
            "source_binding.source_approval_packet_sha256 must be 64 lowercase hex characters"
        )
    if expected_source_packet_path is not None:
        source_packet_value = str(source_binding.get("source_packet_json") or "")
        if not source_packet_value:
            record["errors"].append("source_binding.source_packet_json is missing")
        else:
            source_packet_path = Path(source_packet_value)
            if not source_packet_path.is_absolute():
                source_packet_path = (approval_report_path.parent / source_packet_path).resolve()
            expected_resolved = expected_source_packet_path.resolve()
            source_resolved = source_packet_path.resolve()
            record["source_packet_path_matches_expected"] = source_resolved == expected_resolved
            if source_resolved != expected_resolved:
                record["errors"].append(
                    "source_binding.source_packet_json does not match "
                    "--external-action-approval-source-packet-json"
                )
        expected_sha = str(record.get("expected_source_packet_sha256") or "")
        if expected_sha and source_sha:
            record["source_packet_sha256_matches_expected"] = source_sha == expected_sha
            if source_sha != expected_sha:
                record["errors"].append(
                    "source_binding.source_approval_packet_sha256 does not match "
                    "--external-action-approval-source-packet-json"
                )
    if source_errors:
        record["errors"].extend(f"source_binding: {item}" for item in source_errors)

    record["valid"] = not record["errors"]
    return record


def build_budget_approval_alignment_record(
    *,
    pre_run_budget_estimate: dict,
    external_action_approval: dict,
    required_before_paid_execution: bool,
) -> dict:
    estimated_total = (
        pre_run_budget_estimate.get("estimated_total_usd")
        if isinstance(pre_run_budget_estimate.get("estimated_total_usd"), dict)
        else {}
    )
    estimated_high = estimated_total.get("high")
    if isinstance(estimated_high, bool) or not isinstance(estimated_high, (int, float)):
        estimated_high = None
    approved_budget = external_action_approval.get("paid_api_approved_budget_usd")
    if isinstance(approved_budget, bool) or not isinstance(approved_budget, (int, float)):
        approved_budget = None

    record = {
        "required_before_paid_execution": bool(required_before_paid_execution),
        "valid": False,
        "pre_run_budget_estimate_valid": bool(pre_run_budget_estimate.get("valid")),
        "external_action_approval_valid": bool(external_action_approval.get("valid")),
        "estimated_total_high_usd": estimated_high,
        "approved_budget_usd": approved_budget,
        "approved_model_scope": str(
            external_action_approval.get("paid_api_approved_model_scope") or ""
        ),
        "approved_budget_covers_estimate_high": False,
        "errors": [],
    }

    if not required_before_paid_execution:
        record["valid"] = True
        return record

    if pre_run_budget_estimate.get("valid") is not True:
        record["errors"].append("pre_run_budget_estimate must be valid before budget approval alignment")
    if external_action_approval.get("valid") is not True:
        record["errors"].append("external_action_approval must be valid before budget approval alignment")
    if estimated_high is None:
        record["errors"].append("pre_run_budget_estimate.estimated_total_usd.high is missing or not numeric")
    if approved_budget is None:
        record["errors"].append("external_action_approval paid_api.approved_budget_usd is missing or not numeric")
    if estimated_high is not None and approved_budget is not None:
        record["approved_budget_covers_estimate_high"] = approved_budget >= estimated_high
        if approved_budget < estimated_high:
            record["errors"].append(
                "external_action_approval paid_api.approved_budget_usd is lower than "
                "pre_run_budget_estimate.estimated_total_usd.high"
            )

    record["valid"] = not record["errors"]
    return record


def build_weave_content_canary_gate_record(
    *,
    gate_path: Path | None,
    require_gate: bool,
    phase: str,
    will_call_paid_model_api: bool,
    max_age_seconds: int | None = DEFAULT_WEAVE_CONTENT_CANARY_MAX_AGE_SECONDS,
) -> dict:
    enforced = bool(require_gate and will_call_paid_model_api and phase in AGENTIC_GENERATION_PHASES)
    record = {
        "required": bool(require_gate),
        "enforced": enforced,
        "path": str(gate_path) if gate_path else "",
        "passed": False,
        "blocking_ok": not enforced,
        "status": "not_configured",
        "failure_kind": "",
        "detail": "",
        "fresh": False,
        "age_seconds": None,
        "max_age_seconds": max_age_seconds,
        "freshness_source": "",
    }
    if gate_path is None:
        record["detail"] = "No Weave content canary gate path was supplied."
        return record

    payload, error = load_json_object(gate_path)
    if payload is None:
        record["status"] = "missing_or_invalid"
        record["detail"] = error or "Gate JSON could not be loaded."
        return record

    contract_issues = weave_content_canary_gate_contract_issues(payload)
    generated_at = numeric_timestamp(payload.get("generated_at"))
    mtime = path_mtime(gate_path)
    freshness_source = "generated_at" if generated_at is not None else "mtime"
    freshness_timestamp = generated_at if generated_at is not None else mtime
    age_seconds = time.time() - freshness_timestamp if freshness_timestamp is not None else None
    fresh = max_age_seconds is None or (
        age_seconds is not None and age_seconds <= max_age_seconds
    )
    passed = bool(payload.get("ok")) and payload.get("status") == "passed"
    contract_ok = not contract_issues
    record.update(
        {
            "passed": passed,
            "native_weave_contract_ok": contract_ok,
            "native_weave_contract_issues": contract_issues,
            "blocking_ok": (passed and fresh and contract_ok) or not enforced,
            "status": str(payload.get("status") or ""),
            "failure_kind": str(payload.get("failure_kind") or ""),
            "detail": str(payload.get("detail") or ""),
            "model": payload.get("model"),
            "canary_id": payload.get("canary_id"),
            "task_id": payload.get("task_id"),
            "generated_at": generated_at,
            "mtime": mtime,
            "fresh": fresh,
            "age_seconds": age_seconds,
            "max_age_seconds": max_age_seconds,
            "freshness_source": freshness_source if freshness_timestamp is not None else "",
        }
    )
    if passed and not fresh:
        record["status"] = "stale"
        record["detail"] = "Passing Weave content canary gate is older than the accepted freshness window."
    elif passed and contract_issues:
        record["status"] = "weave_gate_contract_invalid"
        record["detail"] = (
            "Passing Weave content canary gate is missing native Weave verifier "
            "contract evidence: " + "; ".join(contract_issues)
        )
    return record


def stream_run(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        proc = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()


def default_wandb_verify_benchmarks(phase: str) -> list[str]:
    return list(DEFAULT_WANDB_VERIFY_BENCHMARKS.get(phase, []))


def config_lookup(config: dict, key: str) -> tuple[bool, object]:
    if key in config:
        return True, config[key]
    current: object = config
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def expected_config_cli_arg(key: str, value: object) -> str:
    return f"{key}={json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"


def wandb_verify_config_expectations(config_path: Path, *, benchmark: str) -> dict[str, object]:
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(config, dict):
        return {}
    keys = [
        "wandb.run_name",
        "model.pretrained_model_name_or_path",
    ]
    flag_key = BENCHMARK_RUN_FLAG_EXPECTATIONS.get(benchmark)
    if flag_key:
        keys.append(flag_key)
    model_key = BENCHMARK_MODEL_CONFIG_EXPECTATIONS.get(benchmark)
    if model_key:
        keys.append(model_key)
    keys.extend(BENCHMARK_NEMOCLAW_CONFIG_EXPECTATIONS.get(benchmark, []))
    expectations: dict[str, object] = {}
    for key in keys:
        present, value = config_lookup(config, key)
        if present:
            expectations[key] = value
    return expectations


def request_model_aliases(model_id: str) -> list[str]:
    value = model_id.strip()
    if not value:
        return []
    aliases = [value]
    if value.startswith("openai-direct/"):
        aliases.append(value.removeprefix("openai-direct/"))
    if "/" in value:
        aliases.append(value.rsplit("/", 1)[-1])
    return list(dict.fromkeys(alias for alias in aliases if alias))


def weave_expected_request_models(config_path: Path, *, phase: str) -> list[str]:
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(config, dict):
        return []
    models: list[str] = []
    for model_id in model_identifier_values(config, phase=phase):
        models.extend(request_model_aliases(model_id))
    return sorted(dict.fromkeys(models))


def build_wandb_verify_command(
    *,
    python: str,
    run_id: str,
    benchmark: str,
    num_few_shots: int,
    include_pending: bool,
    require_aggregate: bool,
    expected_run_config: dict[str, object] | None = None,
    expected_run_tags: list[str] | None = None,
    expected_run_group: str | None = None,
    expected_run_job_type: str | None = None,
    require_nemoclaw_session_audit: bool = False,
    env_file: Path | None = None,
    json_path: Path | None = None,
) -> list[str]:
    command = [
        python,
        str(WANDB_COMPLETION_VERIFY_RUNNER),
        "--run-id",
        run_id,
        "--benchmark",
        benchmark,
    ]
    expected_total = DEFAULT_EXPECTED_TOTALS.get(benchmark)
    if expected_total is not None:
        command.extend(["--expected-total", str(expected_total)])
    if benchmark == "taiwan_full":
        command.extend(["--num-few-shots", str(num_few_shots)])
        if include_pending:
            command.append("--include-pending")
        if not require_aggregate:
            command.append("--no-require-aggregate")
    for key, value in sorted((expected_run_config or {}).items()):
        command.extend(["--expected-run-config", expected_config_cli_arg(key, value)])
    for tag in sorted(expected_run_tags or []):
        command.extend(["--expected-run-tag", tag])
    if expected_run_group:
        command.extend(["--expected-run-group", expected_run_group])
    if expected_run_job_type:
        command.extend(["--expected-run-job-type", expected_run_job_type])
    if require_nemoclaw_session_audit and benchmark in {"agentic_math", "agentic_swe"}:
        command.append("--require-nemoclaw-session-audit")
    if env_file:
        command.extend(["--env-file", str(env_file)])
    if json_path:
        command.extend(["--json", str(json_path)])
    return command


def run_wandb_completion_verification(
    *,
    python: str,
    run_id: str,
    benchmark: str,
    output_path: Path,
    env: dict[str, str],
    num_few_shots: int,
    include_pending: bool,
    require_aggregate: bool,
    expected_run_config: dict[str, object] | None = None,
    expected_run_tags: list[str] | None = None,
    expected_run_group: str | None = None,
    expected_run_job_type: str | None = None,
    require_nemoclaw_session_audit: bool = False,
    env_file: Path | None = None,
) -> dict:
    command = build_wandb_verify_command(
        python=python,
        run_id=run_id,
        benchmark=benchmark,
        num_few_shots=num_few_shots,
        include_pending=include_pending,
        require_aggregate=require_aggregate,
        expected_run_config=expected_run_config,
        expected_run_tags=expected_run_tags,
        expected_run_group=expected_run_group,
        expected_run_job_type=expected_run_job_type,
        require_nemoclaw_session_audit=require_nemoclaw_session_audit,
        env_file=env_file,
        json_path=output_path,
    )
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        payload = {
            "ok": False,
            "benchmark": benchmark,
            "run_id": run_id,
            "checks": [],
            "parse_error": "Verifier stdout was not JSON",
        }
    payload_ok = bool(payload.get("ok"))
    returncode_ok = result.returncode == 0
    payload["payload_ok"] = payload_ok
    payload["returncode_ok"] = returncode_ok
    payload["ok"] = payload_ok and returncode_ok
    payload["command"] = command
    payload["returncode"] = result.returncode
    payload["stderr"] = result.stderr
    write_json(output_path, payload)
    return payload


def build_weave_agents_verify_command(
    *,
    python: str,
    agent_name: str,
    limit: int,
    require_content: bool,
    require_tool_span: bool,
    require_tool_content: bool,
    require_usage: bool,
    conversation_id_contains: str | None = None,
    expected_request_models: list[str] | None = None,
) -> list[str]:
    command = [
        python,
        str(WEAVE_AGENTS_VERIFY_RUNNER),
        "--agent-name",
        agent_name,
        "--limit",
        str(limit),
    ]
    if not require_content:
        command.append("--no-require-content")
    if require_tool_span:
        command.append("--require-tool-span")
    if require_tool_content:
        command.append("--require-tool-content")
    if require_usage:
        command.append("--require-usage")
    if conversation_id_contains:
        command.extend(["--conversation-id-contains", conversation_id_contains])
    for model in sorted(dict.fromkeys(expected_request_models or [])):
        command.extend(["--expected-request-model", model])
    return command


def build_run_eval_command(
    *,
    python: str,
    base_config: str,
    config: str,
    yes: bool,
) -> list[str]:
    command = [
        python,
        "scripts/run_eval.py",
        "--base-config",
        base_config,
        "--config",
        config,
    ]
    if yes:
        command.append("--yes")
    return command


def build_run_eval_preflight_command(
    *,
    python: str,
    base_config: str,
    config: str,
    output_json: Path,
) -> list[str]:
    return [
        python,
        "scripts/run_eval.py",
        "--base-config",
        base_config,
        "--config",
        config,
        "--preflight",
        "--preflight-json",
        str(output_json),
    ]


def build_run_eval_preflight_records(
    config_paths: list[Path],
    *,
    phase: str,
    python: str,
    base_config: str,
    output_root: Path,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for config_path in config_paths:
        rel_config = config_arg(config_path)
        slug = config_path.stem.replace("config-taiwan-full-", "")
        output_json = output_root / "run_eval_preflight" / f"{phase}-{slug}.json"
        records.append(
            {
                "config": rel_config,
                "output_json": str(output_json),
                "required_before_run_eval": True,
                "command": build_run_eval_preflight_command(
                    python=python,
                    base_config=base_config,
                    config=rel_config,
                    output_json=output_json,
                ),
            }
        )
    return records


def run_weave_agents_verification(
    *,
    python: str,
    agent_name: str,
    limit: int,
    output_path: Path,
    env: dict[str, str],
    require_content: bool,
    require_tool_span: bool,
    require_tool_content: bool,
    require_usage: bool,
    conversation_id_contains: str | None = None,
    expected_request_models: list[str] | None = None,
) -> dict:
    command = build_weave_agents_verify_command(
        python=python,
        agent_name=agent_name,
        limit=limit,
        require_content=require_content,
        require_tool_span=require_tool_span,
        require_tool_content=require_tool_content,
        require_usage=require_usage,
        conversation_id_contains=conversation_id_contains,
        expected_request_models=expected_request_models,
    )
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        payload = {
            "ok": False,
            "agent_name": agent_name,
            "checks": [],
            "parse_error": "Verifier stdout was not JSON",
        }
    payload_ok = bool(payload.get("ok"))
    returncode_ok = result.returncode == 0
    payload.setdefault("generated_at", time.time())
    payload.setdefault("verification_schema_version", WEAVE_AGENTS_COMPLETION_SCHEMA_VERSION)
    payload.setdefault(
        "required_evidence",
        {
            "content_required": require_content,
            "tool_span_required": require_tool_span,
            "tool_content_required": require_tool_content,
            "usage_required": require_usage,
            "conversation_id_contains": conversation_id_contains or "",
            "expected_request_models": list(expected_request_models or []),
        },
    )
    payload["payload_ok"] = payload_ok
    payload["returncode_ok"] = returncode_ok
    payload["ok"] = payload_ok and returncode_ok
    payload["command"] = command
    payload["returncode"] = result.returncode
    payload["stderr"] = result.stderr
    write_json(output_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", action="append", help="Model slug to run. Repeatable.")
    parser.add_argument(
        "--canary",
        action="store_true",
        help="Run the single manifest model marked canary=true for one-model full review.",
    )
    parser.add_argument(
        "--include-final-only",
        action="store_true",
        help="Allow models marked final_only=true, such as very high-cost final release candidates.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--base-config", default="base_config_taiwan.yaml")
    parser.add_argument("--phase", choices=PHASE_CHOICES, default="full")
    parser.add_argument(
        "--run-purpose",
        help="Human-readable reason for this paid evaluation run, recorded in the execution plan.",
    )
    parser.add_argument(
        "--expected-cost-band",
        help="Expected cost range or cap, e.g. '$50-$100' or '<$200', recorded in the execution plan.",
    )
    parser.add_argument(
        "--pre-run-budget-estimate-json",
        type=Path,
        help=(
            "Optional machine-readable budget estimate JSON to bind into the "
            "execution plan and paid-run review before launching paid model calls."
        ),
    )
    parser.add_argument(
        "--external-action-approval-report-json",
        type=Path,
        help=(
            "Verifier report JSON produced by verify_external_action_approval_packet.py "
            "with --source-packet-json. Required before non-prepare external execution."
        ),
    )
    parser.add_argument(
        "--external-action-approval-source-packet-json",
        type=Path,
        help=(
            "Source external_action_approval_packet.json that the verifier report "
            "must be bound to. Required before non-prepare external execution so "
            "an approval report from another release bundle cannot authorize this run."
        ),
    )
    parser.add_argument(
        "--wandb-run-id-prefix",
        help=(
            "Set WANDB_RUN_ID to '<prefix>-<model-slug>' for each model. "
            "Use the same prefix across phases to append all benchmark tables to one W&B run."
        ),
    )
    parser.add_argument(
        "--verify-wandb-completion",
        action="store_true",
        help=(
            "After each successful run_eval.py process, verify that the expected W&B "
            "leaderboard tables/scalars/artifacts are present. Requires --wandb-run-id-prefix."
        ),
    )
    parser.add_argument(
        "--wandb-verify-benchmark",
        action="append",
        choices=["agentic_math", "agentic_swe", "taiwan_full"],
        help=(
            "Benchmark completion gate to run. Repeatable. Defaults by phase: "
            "full=taiwan_full, agentic=agentic_math+agentic_swe, "
            "agentic_aggregate=taiwan_full."
        ),
    )
    parser.add_argument("--wandb-verify-num-few-shots", type=int, default=2)
    parser.add_argument("--wandb-verify-include-pending", action="store_true")
    parser.add_argument("--wandb-verify-no-require-aggregate", action="store_true")
    parser.add_argument(
        "--verify-weave-agents",
        action="store_true",
        help=(
            "After each successful run_eval.py process, verify that W&B Weave "
            "Agents traces exist and expose conversation content in the Agents API."
        ),
    )
    parser.add_argument("--weave-agent-name", default=DEFAULT_WEAVE_AGENT_NAME)
    parser.add_argument("--weave-agents-limit", type=int, default=20)
    parser.add_argument(
        "--weave-agents-no-require-content",
        action="store_true",
        help="Allow structure-only Agents traces. Production Taiwan agentic runs should normally keep content required.",
    )
    parser.add_argument("--weave-agents-require-tool-span", action="store_true")
    parser.add_argument("--weave-agents-require-tool-content", action="store_true")
    parser.add_argument("--weave-agents-require-usage", action="store_true")
    parser.add_argument(
        "--weave-agents-conversation-id-contains",
        help=(
            "Optional Weave conversation-id substring filter. When "
            "--verify-weave-agents is enabled, this must include the W&B run id "
            "or the {wandb_run_id} placeholder; default is the per-run WANDB_RUN_ID."
        ),
    )
    parser.add_argument(
        "--weave-content-canary-gate",
        type=Path,
        help="Gate JSON produced by verify_weave_agents_content_canary_result.py.",
    )
    parser.add_argument(
        "--require-weave-content-canary",
        action="store_true",
        help=(
            "For paid agentic/full phases, fail before run_eval.py unless the "
            "supplied content canary gate has ok=true, status=passed, and "
            "is within the accepted freshness window."
        ),
    )
    parser.add_argument(
        "--weave-content-canary-max-age-seconds",
        type=int,
        default=DEFAULT_WEAVE_CONTENT_CANARY_MAX_AGE_SECONDS,
        help=(
            "Maximum age for a passing Weave content canary gate. "
            "Use a negative value only to disable freshness checks deliberately."
        ),
    )
    parser.add_argument(
        "--require-nemoclaw-agentic-config",
        action="store_true",
        help=(
            "Fail before run_eval.py when phase=agentic/full selected configs do not "
            "route Agentic Math and SWE-Bench Pro through NeMoClaw."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs") / "taiwan_full_eval",
    )
    parser.add_argument(
        "--generated-config-dir",
        type=Path,
        default=CONFIG_DIR / "taiwan_full" / "generated",
    )
    parser.add_argument("--agentic-math-nemoclaw-sandbox")
    parser.add_argument("--agentic-math-nemoclaw-bin", default="nemoclaw")
    parser.add_argument("--agentic-math-nemoclaw-workdir", default="/sandbox")
    parser.add_argument("--agentic-math-nemoclaw-openclaw-config-path")
    parser.add_argument("--swebench-pro-nemoclaw-sandbox")
    parser.add_argument("--swebench-pro-nemoclaw-bin", default="nemoclaw")
    parser.add_argument("--swebench-pro-nemoclaw-checkout-sandbox-root")
    parser.add_argument("--swebench-pro-nemoclaw-workdir")
    parser.add_argument(
        "--swebench-pro-nemoclaw-checkout-transfer-mode",
        choices=["visible", "copy"],
    )
    parser.add_argument("--swebench-pro-nemoclaw-openclaw-config-path")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--yes", action="store_true", help="Pass --yes to scripts/run_eval.py.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.weave_content_canary_max_age_seconds < 0:
        args.weave_content_canary_max_age_seconds = None
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.output_dir = args.generated_config_dir
    configs = generate_configs(args)
    configs = configs[args.start_index :]
    if args.limit is not None:
        configs = configs[: args.limit]

    phase_requires_paid_model_api = args.phase in {"full", "nonagentic", "agentic"}
    will_call_paid_model_api = not args.prepare_only and phase_requires_paid_model_api
    will_execute_external_actions = not args.prepare_only and (
        phase_requires_paid_model_api
        or bool(args.verify_wandb_completion)
        or bool(args.verify_weave_agents)
    )
    selected_config_model_bindings = collect_selected_config_model_bindings(
        configs,
        phase=args.phase,
    )
    agentic_production_evidence_guard = build_agentic_production_evidence_guard(
        args,
        phase=args.phase,
        will_call_paid_model_api=will_call_paid_model_api,
    )
    nemoclaw_agentic_config_guard = build_nemoclaw_agentic_config_guard(
        configs,
        phase=args.phase,
        required=bool(args.require_nemoclaw_agentic_config),
    )
    weave_content_canary_gate = build_weave_content_canary_gate_record(
        gate_path=args.weave_content_canary_gate,
        require_gate=bool(args.require_weave_content_canary),
        phase=args.phase,
        will_call_paid_model_api=will_call_paid_model_api,
        max_age_seconds=args.weave_content_canary_max_age_seconds,
    )
    pre_run_budget_estimate = build_pre_run_budget_estimate_record(
        args.pre_run_budget_estimate_json,
        required_before_paid_execution=phase_requires_paid_model_api,
        selected_config_model_bindings=selected_config_model_bindings,
    )
    external_action_approval = build_external_action_approval_record(
        args.external_action_approval_report_json,
        required_before_external_action=will_execute_external_actions,
        expected_source_packet_path=args.external_action_approval_source_packet_json,
    )
    budget_approval_alignment = build_budget_approval_alignment_record(
        pre_run_budget_estimate=pre_run_budget_estimate,
        external_action_approval=external_action_approval,
        required_before_paid_execution=will_call_paid_model_api,
    )
    run_eval_preflights = build_run_eval_preflight_records(
        configs,
        phase=args.phase,
        python=args.python,
        base_config=args.base_config,
        output_root=args.output_root,
    )
    execution_plan = {
        "phase": args.phase,
        "prepare_only": bool(args.prepare_only),
        "canary": bool(args.canary),
        "include_final_only": bool(args.include_final_only),
        "run_purpose": args.run_purpose or "",
        "expected_cost_band": args.expected_cost_band or "",
        "requires_paid_model_api": phase_requires_paid_model_api,
        "will_call_paid_model_api": will_call_paid_model_api,
        "will_execute_external_actions": will_execute_external_actions,
        "model_count": len(configs),
        "configs": [config_arg(path) for path in configs],
        "selected_config_model_bindings": selected_config_model_bindings,
        "agentic_production_evidence_guard": agentic_production_evidence_guard,
        "nemoclaw_agentic_config_guard": nemoclaw_agentic_config_guard,
        "wandb_run_id_prefix": args.wandb_run_id_prefix or "",
        "verify_wandb_completion": bool(args.verify_wandb_completion),
        "wandb_verify_benchmarks": args.wandb_verify_benchmark
        or default_wandb_verify_benchmarks(args.phase),
        "verify_weave_agents": bool(args.verify_weave_agents),
        "weave_agent_name": args.weave_agent_name,
        "weave_agents_require_content": not bool(args.weave_agents_no_require_content),
        "weave_agents_require_tool_span": bool(args.weave_agents_require_tool_span),
        "weave_agents_require_tool_content": bool(args.weave_agents_require_tool_content),
        "weave_agents_require_usage": bool(args.weave_agents_require_usage),
        "weave_agents_conversation_id_contains": args.weave_agents_conversation_id_contains or "",
        "weave_content_canary_gate": weave_content_canary_gate,
        "run_eval_preflights": run_eval_preflights,
        "pre_run_budget_estimate": pre_run_budget_estimate,
        "external_action_approval": external_action_approval,
        "budget_approval_alignment": budget_approval_alignment,
        "created_at": time.time(),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan_prefix = "canary_" if args.canary else ""
    plan_path = args.output_root / f"{plan_prefix}{args.phase}_execution_plan.json"
    review_path = args.output_root / f"{plan_prefix}{args.phase}_paid_run_review.json"
    batch_manifest_path = args.output_root / "batch_manifest.json"
    write_json(plan_path, execution_plan)

    review_record = {
        "status": "prepared" if args.prepare_only else "pending",
        "phase": args.phase,
        "prepare_only": bool(args.prepare_only),
        "canary": bool(args.canary),
        "include_final_only": bool(args.include_final_only),
        "requires_paid_model_api": phase_requires_paid_model_api,
        "will_call_paid_model_api": will_call_paid_model_api,
        "will_execute_external_actions": will_execute_external_actions,
        "run_purpose": args.run_purpose or "",
        "expected_cost_band": args.expected_cost_band or "",
        "model_count": len(configs),
        "configs": execution_plan["configs"],
        "selected_config_model_bindings": selected_config_model_bindings,
        "agentic_production_evidence_guard": agentic_production_evidence_guard,
        "nemoclaw_agentic_config_guard": nemoclaw_agentic_config_guard,
        "wandb_run_id_prefix": args.wandb_run_id_prefix or "",
        "verify_wandb_completion": bool(args.verify_wandb_completion),
        "wandb_verify_benchmarks": args.wandb_verify_benchmark
        or default_wandb_verify_benchmarks(args.phase),
        "verify_weave_agents": bool(args.verify_weave_agents),
        "weave_agent_name": args.weave_agent_name,
        "weave_agents_require_content": not bool(args.weave_agents_no_require_content),
        "weave_agents_require_tool_span": bool(args.weave_agents_require_tool_span),
        "weave_agents_require_tool_content": bool(args.weave_agents_require_tool_content),
        "weave_agents_require_usage": bool(args.weave_agents_require_usage),
        "weave_agents_conversation_id_contains": args.weave_agents_conversation_id_contains or "",
        "weave_content_canary_gate": weave_content_canary_gate,
        "run_eval_preflights": run_eval_preflights,
        "pre_run_budget_estimate": pre_run_budget_estimate,
        "external_action_approval": external_action_approval,
        "budget_approval_alignment": budget_approval_alignment,
        "completion_requirements": {
            "status": "completed",
            "pre_run_budget_estimate": {
                "required_before_paid_execution": phase_requires_paid_model_api,
                "required_fields": [
                    "path",
                    "sha256",
                    "target_model",
                    "target_model or target_models matching selected config model identifiers",
                    "price_per_million_tokens",
                    "estimated_total_usd.low",
                    "estimated_total_usd.mid",
                    "estimated_total_usd.high",
                    "pricing_source_url",
                ],
                "provider_dashboard_authoritative": True,
            },
            "external_action_approval": {
                "required_before_external_action": will_execute_external_actions,
                "required_fields": [
                    "path",
                    "sha256",
                    "schema_version=1",
                    "status=approved",
                    "required_approval_count",
                    "granted_approval_count",
                    "all_required_approvals_granted=true",
                    "source_binding.bound=true",
                    "source_binding.source_packet_json",
                    "source_binding.source_approval_packet_sha256",
                    "expected_source_packet_json",
                    "expected_source_packet_sha256",
                    "source_packet_path_matches_expected=true",
                    "source_packet_sha256_matches_expected=true",
                    "will_execute_external_actions=false",
                    "paid_api.approved_budget_usd",
                ],
                "source_bound_review_copy_required": True,
            },
            "budget_approval_alignment": {
                "required_before_paid_execution": will_call_paid_model_api,
                "required_fields": [
                    "pre_run_budget_estimate.estimated_total_usd.high",
                    "external_action_approval paid_api.approved_budget_usd",
                    "approved_budget_usd >= estimated_total_usd.high",
                ],
            },
            "actual_cost_estimate": "required after execution",
            "provider_bill_reference": "required after execution",
            "runs": {
                "count_must_equal_model_count": True,
                "required_fields": [
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
            },
            "run_eval_preflight": {
                "required": True,
                "required_before_run_eval": True,
                "expected_record_count": len(configs),
                "required_fields": [
                    "config",
                    "command",
                    "output_json",
                    "ok=true",
                    "status=passed",
                    "will_initialize_wandb=false",
                    "will_start_inference_engine=false",
                    "will_run_evaluators=false",
                ],
            },
            "wandb_completion": {
                "required": bool(args.verify_wandb_completion),
                "benchmarks": args.wandb_verify_benchmark
                or default_wandb_verify_benchmarks(args.phase),
                "schema_version": 1,
                "observed_evidence_required": True,
                "nemoclaw_session_audit_required_for_agentic_benchmarks": bool(
                    args.require_nemoclaw_agentic_config
                ),
            },
            "weave_agents_completion": {
                "required": bool(args.verify_weave_agents),
                "agent_name": args.weave_agent_name if args.verify_weave_agents else "",
                "content_required": bool(args.verify_weave_agents)
                and not bool(args.weave_agents_no_require_content),
                "tool_span_required": bool(args.weave_agents_require_tool_span),
                "tool_content_required": bool(args.weave_agents_require_tool_content),
                "usage_required": bool(args.weave_agents_require_usage),
                "run_scope_required": bool(args.verify_weave_agents),
                "conversation_id_contains_default": (
                    "wandb_run_id when --weave-agents-conversation-id-contains is not set"
                ),
            },
            "agentic_production_evidence": {
                "required_for_paid_agentic_or_full": True,
                "required_flags": list(AGENTIC_PRODUCTION_EVIDENCE_REQUIREMENTS),
                "purpose": (
                    "Prevent paid Agentic Math/SWE execution from being reviewed "
                    "as complete unless NeMoClaw routing, W&B completion, and "
                    "native Weave Agents evidence are all required up front."
                ),
            },
        },
        "execution_plan_path": str(plan_path),
        "batch_manifest_path": str(batch_manifest_path),
        "post_run_cost_command": (
            "uv run python scripts/analysis/estimate_agentic_usage_costs.py "
            "outputs/taiwan_full_eval --csv outputs/taiwan_full_eval/canary_usage_cost_summary.csv"
        ),
        "actual_cost_estimate": "",
        "provider_bill_reference": "",
        "runs": [],
        "created_at": time.time(),
        "started_at": None,
        "ended_at": None,
    }
    write_json(review_path, review_record)

    if not agentic_production_evidence_guard["ok"]:
        review_record["status"] = "agentic_production_evidence_required"
        review_record["blocking_reason"] = agentic_production_evidence_guard
        write_json(review_path, review_record)
        raise SystemExit(
            "Paid Agentic Math/SWE execution requires production evidence flags: "
            + ", ".join(str(item) for item in agentic_production_evidence_guard["missing_flags"])
        )

    if not nemoclaw_agentic_config_guard["ok"]:
        review_record["status"] = "nemoclaw_agentic_config_failed"
        review_record["blocking_reason"] = nemoclaw_agentic_config_guard
        write_json(review_path, review_record)
        raise SystemExit(
            "NeMoClaw agentic config requirement failed: "
            + "; ".join(str(item) for item in nemoclaw_agentic_config_guard["errors"])
        )

    if will_call_paid_model_api:
        missing = [
            flag
            for flag, value in {
                "--run-purpose": args.run_purpose,
                "--expected-cost-band": args.expected_cost_band,
                "--pre-run-budget-estimate-json": (
                    args.pre_run_budget_estimate_json
                    if pre_run_budget_estimate.get("valid")
                    else None
                ),
                "--external-action-approval-report-json": (
                    args.external_action_approval_report_json
                    if external_action_approval.get("valid")
                    else None
                ),
                "--external-action-approval-source-packet-json": (
                    args.external_action_approval_source_packet_json
                    if external_action_approval.get("source_packet_path_matches_expected")
                    and external_action_approval.get("source_packet_sha256_matches_expected")
                    else None
                ),
            }.items()
            if not value
        ]
        if missing:
            review_record["status"] = "accountability_fields_missing"
            review_record["missing_fields"] = missing
            write_json(review_path, review_record)
            raise SystemExit(
                "Paid model evaluation requires accountability fields: "
                + ", ".join(missing)
            )

    if will_execute_external_actions and not external_action_approval.get("valid"):
        review_record["status"] = "external_action_approval_missing"
        review_record["missing_fields"] = [
            flag
            for flag, value in {
                "--external-action-approval-report-json": (
                    args.external_action_approval_report_json
                    if external_action_approval.get("present")
                    else None
                ),
                "--external-action-approval-source-packet-json": (
                    args.external_action_approval_source_packet_json
                    if external_action_approval.get("source_packet_path_matches_expected")
                    and external_action_approval.get("source_packet_sha256_matches_expected")
                    else None
                ),
            }.items()
            if not value
        ]
        write_json(review_path, review_record)
        raise SystemExit(
            "External execution requires a valid source-bound approval verifier report "
            "and matching source packet: --external-action-approval-report-json, "
            "--external-action-approval-source-packet-json"
        )

    if will_call_paid_model_api and not budget_approval_alignment.get("valid"):
        review_record["status"] = "budget_approval_alignment_failed"
        review_record["blocking_reason"] = budget_approval_alignment
        write_json(review_path, review_record)
        raise SystemExit(
            "Paid model evaluation requires approved budget to cover the pre-run "
            "high estimate: "
            + "; ".join(str(item) for item in budget_approval_alignment["errors"])
        )

    if not weave_content_canary_gate["blocking_ok"]:
        review_record["status"] = "weave_content_canary_gate_failed"
        review_record["blocking_reason"] = weave_content_canary_gate
        write_json(review_path, review_record)
        raise SystemExit(
            "Weave content canary gate failed before paid agentic execution: "
            f"{weave_content_canary_gate['status']} "
            f"{weave_content_canary_gate['failure_kind']}".strip()
        )

    if args.verify_wandb_completion:
        if not args.wandb_run_id_prefix:
            raise SystemExit("--verify-wandb-completion requires --wandb-run-id-prefix")
        if not (args.wandb_verify_benchmark or default_wandb_verify_benchmarks(args.phase)):
            raise SystemExit(
                "--verify-wandb-completion has no default benchmark for this phase; "
                "pass --wandb-verify-benchmark explicitly."
            )
    if args.verify_weave_agents and not args.wandb_run_id_prefix:
        raise SystemExit("--verify-weave-agents requires --wandb-run-id-prefix")

    manifest_rows = []
    if args.prepare_only:
        for path in configs:
            print(config_arg(path))
        return

    env = load_env_file(os.environ.copy(), args.env_file)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("WANDB_CONSOLE", "wrap")

    for index, config_path in enumerate(configs, start=1):
        rel_config = config_arg(config_path)
        slug = config_path.stem.replace("config-taiwan-full-", "")
        log_path = args.output_root / "logs" / f"{args.phase}-{slug}.log"
        preflight_json = args.output_root / "run_eval_preflight" / f"{args.phase}-{slug}.json"
        preflight_command = build_run_eval_preflight_command(
            python=args.python,
            base_config=args.base_config,
            config=rel_config,
            output_json=preflight_json,
        )
        command = build_run_eval_command(
            python=args.python,
            base_config=args.base_config,
            config=rel_config,
            yes=bool(args.yes),
        )
        started_at = time.time()
        print(f"\n[{index}/{len(configs)}] {rel_config}")
        run_env = env.copy()
        if args.wandb_run_id_prefix:
            run_env["WANDB_RUN_ID"] = f"{args.wandb_run_id_prefix}-{slug}"
            run_env["WANDB_RESUME"] = "allow"
        if review_record["started_at"] is None:
            review_record["started_at"] = started_at
        review_record["status"] = "running"
        write_json(review_path, review_record)
        preflight_log_path = args.output_root / "logs" / f"{args.phase}-{slug}.preflight.log"
        preflight_returncode = stream_run(preflight_command, preflight_log_path, run_env)
        preflight_ok = preflight_returncode == 0
        preflight_status = ""
        if preflight_json.exists():
            preflight_payload, preflight_error = load_json_object(preflight_json)
            if preflight_payload is not None:
                preflight_ok = preflight_ok and preflight_payload.get("ok") is True
                preflight_status = str(preflight_payload.get("status") or "")
            else:
                preflight_status = preflight_error or "preflight JSON could not be read"
                preflight_ok = False
        else:
            preflight_status = "preflight JSON was not written"
            preflight_ok = False
        if not preflight_ok:
            row = {
                "config": rel_config,
                "preflight_command": preflight_command,
                "preflight_log_path": str(preflight_log_path),
                "preflight_json": str(preflight_json),
                "preflight_returncode": preflight_returncode,
                "preflight_ok": False,
                "preflight_status": preflight_status,
                "phase": args.phase,
                "wandb_run_id": run_env.get("WANDB_RUN_ID", ""),
                "wandb_entity": run_env.get("WANDB_ENTITY", ""),
                "wandb_project": run_env.get("WANDB_PROJECT", ""),
                "returncode": preflight_returncode,
                "started_at": started_at,
                "ended_at": time.time(),
            }
            manifest_rows.append(row)
            review_record["runs"].append(row)
            review_record["status"] = "run_eval_preflight_failed"
            review_record["ended_at"] = time.time()
            write_json(batch_manifest_path, manifest_rows)
            write_json(review_path, review_record)
            raise SystemExit(
                f"run_eval preflight failed for {rel_config}: {preflight_status}"
            )
        returncode = stream_run(command, log_path, run_env)
        row = {
            "config": rel_config,
            "preflight_command": preflight_command,
            "preflight_log_path": str(preflight_log_path),
            "preflight_json": str(preflight_json),
            "preflight_returncode": preflight_returncode,
            "preflight_ok": preflight_ok,
            "preflight_status": preflight_status,
            "log_path": str(log_path),
            "phase": args.phase,
            "wandb_run_id": run_env.get("WANDB_RUN_ID", ""),
            "wandb_entity": run_env.get("WANDB_ENTITY", ""),
            "wandb_project": run_env.get("WANDB_PROJECT", ""),
            "returncode": returncode,
            "started_at": started_at,
            "ended_at": time.time(),
        }
        if returncode == 0 and args.verify_wandb_completion:
            verify_benchmarks = args.wandb_verify_benchmark or default_wandb_verify_benchmarks(args.phase)
            verification_results = []
            for benchmark in verify_benchmarks:
                verify_output_path = (
                    args.output_root
                    / "wandb_completion"
                    / f"{args.phase}-{slug}-{benchmark}.json"
                )
                verification_results.append(
                    run_wandb_completion_verification(
                        python=args.python,
                        run_id=row["wandb_run_id"],
                        benchmark=benchmark,
                        output_path=verify_output_path,
                        env=run_env,
                        num_few_shots=args.wandb_verify_num_few_shots,
                        include_pending=bool(args.wandb_verify_include_pending),
                        require_aggregate=not bool(args.wandb_verify_no_require_aggregate),
                        expected_run_config=wandb_verify_config_expectations(
                            config_path,
                            benchmark=benchmark,
                        ),
                        expected_run_job_type="evaluation",
                        require_nemoclaw_session_audit=bool(
                            args.require_nemoclaw_agentic_config
                            and benchmark in {"agentic_math", "agentic_swe"}
                        ),
                        env_file=args.env_file,
                    )
                )
            row["wandb_completion"] = [
                {
                    "benchmark": result.get("benchmark"),
                    "ok": bool(result.get("ok")),
                    "entity": result.get("entity") or row["wandb_entity"],
                    "project": result.get("project") or row["wandb_project"],
                    "run_id": result.get("run_id") or row["wandb_run_id"],
                    "path": str(
                        args.output_root
                        / "wandb_completion"
                        / f"{args.phase}-{slug}-{result.get('benchmark')}.json"
                    ),
                }
                for result in verification_results
            ]
            if not all(result.get("ok") for result in verification_results):
                row["returncode"] = 1
                row["wandb_completion_failed"] = True
                returncode = 1
        if returncode == 0 and args.verify_weave_agents:
            weave_output_path = (
                args.output_root
                / "weave_agents_completion"
                / f"{args.phase}-{slug}.json"
            )
            try:
                weave_conversation_id_contains = resolve_weave_conversation_id_contains(
                    args.weave_agents_conversation_id_contains,
                    wandb_run_id=row["wandb_run_id"],
                )
            except ValueError as exc:
                row["returncode"] = 1
                row["weave_agents_completion_failed"] = True
                row["weave_agents_completion_error"] = str(exc)
                returncode = 1
                manifest_rows.append(row)
                write_json(batch_manifest_path, manifest_rows)
                review_record["runs"] = manifest_rows
                review_record["ended_at"] = row["ended_at"]
                review_record["status"] = "failed"
                write_json(review_path, review_record)
                raise SystemExit(str(exc)) from exc
            weave_result = run_weave_agents_verification(
                python=args.python,
                agent_name=args.weave_agent_name,
                limit=args.weave_agents_limit,
                output_path=weave_output_path,
                env=run_env,
                require_content=not bool(args.weave_agents_no_require_content),
                require_tool_span=bool(args.weave_agents_require_tool_span),
                require_tool_content=bool(args.weave_agents_require_tool_content),
                require_usage=bool(args.weave_agents_require_usage),
                conversation_id_contains=weave_conversation_id_contains,
                expected_request_models=weave_expected_request_models(
                    config_path,
                    phase=args.phase,
                ),
            )
            row["weave_agents_completion"] = {
                "ok": bool(weave_result.get("ok")),
                "path": str(weave_output_path),
                "agent_name": args.weave_agent_name,
                "run_id": row["wandb_run_id"],
                "conversation_id_contains": weave_conversation_id_contains,
                "expected_request_models": weave_expected_request_models(
                    config_path,
                    phase=args.phase,
                ),
            }
            if not weave_result.get("ok"):
                row["returncode"] = 1
                row["weave_agents_completion_failed"] = True
                returncode = 1
        manifest_rows.append(row)
        write_json(batch_manifest_path, manifest_rows)
        review_record["runs"] = manifest_rows
        review_record["ended_at"] = row["ended_at"]
        review_record["status"] = "failed" if returncode != 0 else "running"
        write_json(review_path, review_record)
        if returncode != 0:
            raise SystemExit(returncode)
    review_record["status"] = "completed"
    review_record["ended_at"] = time.time()
    review_record["runs"] = manifest_rows
    write_json(review_path, review_record)


if __name__ == "__main__":
    main()
