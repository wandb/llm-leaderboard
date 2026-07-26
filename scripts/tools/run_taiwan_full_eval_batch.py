#!/usr/bin/env python3
"""
Run Taiwan full evaluations sequentially from the generated model configs.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"

from external_action_approval_checks import (
    approval_results as normalized_approval_results,
    extract_paid_api_approval,
    validate_approval_results,
)
from prepare_taiwan_full_eval_configs import (
    AGENTIC_DENIED_ARGUMENT_PATTERNS,
    AGENTIC_SWE_ASSORTED_DENIED_ARGUMENT_PATTERNS,
    AGENTIC_SWE_ASSORTED_DENIED_TOOLS,
    AGENTIC_DENIED_TOOLS,
    CONFIG_DIR,
    DEFAULT_MANIFEST,
    PHASE_CHOICES,
    generate_configs,
)
from weave_content_canary_gate_contract import (
    weave_content_canary_gate_contract_issues,
)

_BENCHMARK_CHECKPOINT_SPEC = importlib.util.spec_from_file_location(
    "_taiwan_benchmark_checkpoint",
    SCRIPTS_ROOT / "evaluator" / "evaluate_utils" / "benchmark_checkpoint.py",
)
assert _BENCHMARK_CHECKPOINT_SPEC and _BENCHMARK_CHECKPOINT_SPEC.loader
_BENCHMARK_CHECKPOINT_MODULE = importlib.util.module_from_spec(
    _BENCHMARK_CHECKPOINT_SPEC
)
_BENCHMARK_CHECKPOINT_SPEC.loader.exec_module(_BENCHMARK_CHECKPOINT_MODULE)
classify_benchmark_failure = (
    _BENCHMARK_CHECKPOINT_MODULE.classify_benchmark_failure
)


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
NEMOCLAW_OPENCLAW_CONFIG_SOURCE_TRACE_TEXT = (
    f"openclaw_config_source: {CANONICAL_NEMOCLAW_OPENCLAW_CONFIG_PATH}"
)
BENCHMARK_RUN_FLAG_EXPECTATIONS = {
    "agentic_math": "run.agentic_math",
    "agentic_swe": "run.agentic_swe_assorted",
    "taiwan_full": "run.aggregate_taiwan",
}
BENCHMARK_MODEL_CONFIG_EXPECTATIONS = {
    "agentic_math": "agentic_math.openclaw_model",
    "agentic_swe": "agentic_swe_assorted.openclaw_model",
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
        "agentic_swe_assorted.nemoclaw_sandbox",
        "agentic_swe_assorted.nemoclaw_openclaw_config_path",
        "agentic_swe_assorted.nemoclaw_checkout_transfer_mode",
        "agentic_swe_assorted.deny_tool",
        "agentic_swe_assorted.deny_argument_pattern",
    ],
}
AGENTIC_GENERATION_PHASES = {"agentic", "full"}
DEFAULT_WEAVE_CONTENT_CANARY_MAX_AGE_SECONDS = 24 * 60 * 60
WEAVE_AGENTS_COMPLETION_SCHEMA_VERSION = 1
REQUIRED_NEMOCLAW_AGENTIC_DENIED_TOOLS = set(AGENTIC_DENIED_TOOLS)
REQUIRED_NEMOCLAW_AGENTIC_DENIED_ARGUMENT_PATTERNS = set(
    AGENTIC_DENIED_ARGUMENT_PATTERNS
)
REQUIRED_NEMOCLAW_AGENTIC_SWE_DENIED_TOOLS = set(AGENTIC_SWE_ASSORTED_DENIED_TOOLS)
REQUIRED_NEMOCLAW_AGENTIC_SWE_DENIED_ARGUMENT_PATTERNS = set(
    AGENTIC_SWE_ASSORTED_DENIED_ARGUMENT_PATTERNS
)
REQUIRED_NEMOCLAW_AGENTIC_ALLOWED_LOCAL_TOOLS = {"exec"}
AGENTIC_PRODUCTION_EVIDENCE_REQUIREMENTS = {
    "--require-nemoclaw-agentic-config": "Agentic Math/SWE generated configs must route through NeMoClaw with deny-policy guards.",
    "--agentic-math-nemoclaw-openclaw-config-path": "Agentic Math must bind NeMoClaw to the reviewed OpenClaw config inside the sandbox.",
    "--agentic-swe-assorted-nemoclaw-openclaw-config-path": "Agentic SWE-Assorted must bind NeMoClaw to the reviewed OpenClaw config inside the sandbox.",
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

PHASE_EXPECTED_SCHEDULED_EVALUATORS = {
    "full": [
        "bfcl",
        "agentic_math",
        "agentic_swe_assorted",
        "mtbench",
        "script_adherence",
        "hle",
        "hallulens_zh_tw",
        "arc_agi",
        "ifeval_zh_tw",
        "ts_bench",
        "tceval_v2",
        "jaster",
        "aggregate_taiwan",
    ],
    "nonagentic": [
        "bfcl",
        "mtbench",
        "script_adherence",
        "hle",
        "hallulens_zh_tw",
        "arc_agi",
        "ifeval_zh_tw",
        "ts_bench",
        "tceval_v2",
        "jaster",
    ],
    "agentic": [
        "agentic_math",
        "agentic_swe_assorted",
    ],
    "agentic_aggregate": [
        "agentic_math",
        "agentic_swe_assorted",
        "aggregate_taiwan",
    ],
}


def nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def valid_nemoclaw_sandbox_name(value: object) -> bool:
    return nonempty_string(value) and str(value).strip().lower() not in {
        "none",
        "null",
        "false",
    }


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


def expected_scheduled_evaluators_for_phase(phase: str) -> list[str]:
    return list(PHASE_EXPECTED_SCHEDULED_EVALUATORS.get(phase, []))


def validate_run_eval_preflight_phase(payload: dict, phase: str) -> dict[str, object]:
    expected = expected_scheduled_evaluators_for_phase(phase)
    observed = payload.get("scheduled_evaluators", payload.get("enabled_benchmarks", []))
    if not isinstance(observed, list):
        observed = []
    observed = [str(item) for item in observed]
    missing = [item for item in expected if item not in observed]
    unexpected = [item for item in observed if item not in expected]
    order_matches = observed == expected
    ok = not missing and not unexpected and order_matches
    return {
        "ok": ok,
        "phase": phase,
        "expected_scheduled_evaluators": expected,
        "observed_scheduled_evaluators": observed,
        "missing_scheduled_evaluators": missing,
        "unexpected_scheduled_evaluators": unexpected,
        "order_matches": order_matches,
    }


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
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def subprocess_output_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


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
                "agentic_swe_assorted.openclaw_model",
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
                "cash_cost_exempt": bool(
                    config_lookup(loaded, "execution.cash_cost_exempt")[1]
                ),
                "cash_cost_exempt_reason": str(
                    config_lookup(
                        loaded,
                        "execution.cash_cost_exempt_reason",
                    )[1]
                    or ""
                ),
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
        run_agentic_swe_assorted = lookup("run.agentic_swe_assorted") is True
        math_sandbox = lookup("agentic_math.nemoclaw_sandbox")
        math_config_path = lookup("agentic_math.nemoclaw_openclaw_config_path")
        math_use_task_agent = lookup("agentic_math.use_task_agent")
        math_no_local = lookup("agentic_math.no_local")
        math_weave_sidecar = lookup("agentic_math.weave_sidecar")
        math_weave_sidecar_strict = lookup("agentic_math.weave_sidecar_strict")
        math_deny_tools = lookup("agentic_math.deny_tool")
        math_deny_argument_patterns = lookup("agentic_math.deny_argument_pattern")
        swe_sandbox = lookup("agentic_swe_assorted.nemoclaw_sandbox")
        swe_config_path = lookup("agentic_swe_assorted.nemoclaw_openclaw_config_path")
        swe_transfer_mode = lookup("agentic_swe_assorted.nemoclaw_checkout_transfer_mode")
        swe_no_local = lookup("agentic_swe_assorted.no_local")
        swe_weave_sidecar = lookup("agentic_swe_assorted.weave_sidecar")
        swe_weave_sidecar_strict = lookup("agentic_swe_assorted.weave_sidecar_strict")
        swe_deny_tools = lookup("agentic_swe_assorted.deny_tool")
        swe_deny_argument_patterns = lookup("agentic_swe_assorted.deny_argument_pattern")

        record.update(
            {
                "run_agentic_math": run_agentic_math,
                "run_agentic_swe_assorted": run_agentic_swe_assorted,
                "agentic_math_nemoclaw_sandbox": math_sandbox if isinstance(math_sandbox, str) else "",
                "agentic_math_nemoclaw_openclaw_config_path": (
                    math_config_path if isinstance(math_config_path, str) else ""
                ),
                "agentic_math_use_task_agent": math_use_task_agent,
                "agentic_math_no_local": math_no_local,
                "agentic_math_weave_sidecar": math_weave_sidecar,
                "agentic_math_weave_sidecar_strict": math_weave_sidecar_strict,
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
                "agentic_swe_assorted_nemoclaw_sandbox": swe_sandbox if isinstance(swe_sandbox, str) else "",
                "agentic_swe_assorted_nemoclaw_openclaw_config_path": (
                    swe_config_path if isinstance(swe_config_path, str) else ""
                ),
                "agentic_swe_assorted_nemoclaw_checkout_transfer_mode": (
                    swe_transfer_mode if isinstance(swe_transfer_mode, str) else ""
                ),
                "agentic_swe_assorted_no_local": swe_no_local,
                "agentic_swe_assorted_weave_sidecar": swe_weave_sidecar,
                "agentic_swe_assorted_weave_sidecar_strict": swe_weave_sidecar_strict,
                "agentic_swe_assorted_deny_tool": string_list(swe_deny_tools) or [],
                "agentic_swe_assorted_local_exec_blocking_patterns": local_exec_blocking_patterns(
                    swe_deny_tools
                ),
                "agentic_swe_assorted_local_exec_allowed": not local_exec_blocking_patterns(
                    swe_deny_tools
                ),
                "agentic_swe_assorted_deny_argument_pattern": (
                    string_list(swe_deny_argument_patterns) or []
                ),
            }
        )
        if enforced:
            if not run_agentic_math:
                issues.append("run.agentic_math must be true")
            if not run_agentic_swe_assorted:
                issues.append("run.agentic_swe_assorted must be true")
            if not valid_nemoclaw_sandbox_name(math_sandbox):
                issues.append(
                    "agentic_math.nemoclaw_sandbox must name an isolated sandbox"
                )
            if not canonical_nemoclaw_openclaw_config_path(math_config_path):
                issues.append(
                    "agentic_math.nemoclaw_openclaw_config_path must be "
                    f"{CANONICAL_NEMOCLAW_OPENCLAW_CONFIG_PATH}"
                )
            if math_use_task_agent is False:
                issues.append("agentic_math.use_task_agent must not be false")
            if math_no_local is not True:
                issues.append(
                    "agentic_math.no_local must be true for production native "
                    "weave-openclaw tracing; local OpenClaw execution is not "
                    "accepted as production trace evidence"
                )
            if math_weave_sidecar is True or math_weave_sidecar_strict is True:
                issues.append(
                    "agentic_math must use native weave-openclaw tracing only; "
                    "weave_sidecar/weave_sidecar_strict are diagnostic paths and "
                    "are not valid production evidence"
                )
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
            if not valid_nemoclaw_sandbox_name(swe_sandbox):
                issues.append(
                    "agentic_swe_assorted.nemoclaw_sandbox must name an isolated sandbox"
                )
            if not canonical_nemoclaw_openclaw_config_path(swe_config_path):
                issues.append(
                    "agentic_swe_assorted.nemoclaw_openclaw_config_path must be "
                    f"{CANONICAL_NEMOCLAW_OPENCLAW_CONFIG_PATH}"
                )
            if swe_transfer_mode != "copy":
                issues.append(
                    "agentic_swe_assorted.nemoclaw_checkout_transfer_mode must be copy"
                )
            if swe_no_local is not True:
                issues.append(
                    "agentic_swe_assorted.no_local must be true for production native "
                    "weave-openclaw tracing; local OpenClaw execution is not "
                    "accepted as production trace evidence"
                )
            if swe_weave_sidecar is True or swe_weave_sidecar_strict is True:
                issues.append(
                    "agentic_swe_assorted must use native weave-openclaw tracing only; "
                    "weave_sidecar/weave_sidecar_strict are diagnostic paths and "
                    "are not valid production evidence"
                )
            require_string_superset(
                section="agentic_swe_assorted",
                field="deny_tool",
                observed_value=swe_deny_tools,
                required_values=REQUIRED_NEMOCLAW_AGENTIC_SWE_DENIED_TOOLS,
            )
            require_local_exec_allowed("agentic_swe_assorted", swe_deny_tools)
            require_string_superset(
                section="agentic_swe_assorted",
                field="deny_argument_pattern",
                observed_value=swe_deny_argument_patterns,
                required_values=REQUIRED_NEMOCLAW_AGENTIC_SWE_DENIED_ARGUMENT_PATTERNS,
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
    swe_config_path = getattr(
        args, "agentic_swe_assorted_nemoclaw_openclaw_config_path", None
    )
    observations = {
        "--require-nemoclaw-agentic-config": bool(args.require_nemoclaw_agentic_config),
        "--agentic-math-nemoclaw-openclaw-config-path": nonempty_string(
            math_config_path
        ),
        "--agentic-swe-assorted-nemoclaw-openclaw-config-path": nonempty_string(
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
            "--agentic-swe-assorted-nemoclaw-openclaw-config-path": swe_config_path,
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
    for category in ("agentic_math", "agentic_swe_assorted"):
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


def build_wandb_resume_policy_record(args: argparse.Namespace) -> dict:
    allow_requested = bool(getattr(args, "allow_wandb_resume", False))
    infrastructure_recovery_attempts = int(
        getattr(args, "infrastructure_resume_attempts", 0) or 0
    )
    config_path = getattr(args, "wandb_resume_config_json", None)
    record = {
        "allow_wandb_resume": allow_requested,
        "infrastructure_recovery_attempts": infrastructure_recovery_attempts,
        "resume_config_json": str(config_path) if config_path else "",
        "valid": not allow_requested,
        "status": "resume_disabled" if not allow_requested else "resume_config_required",
        "errors": [],
        "expected": {
            "allow_wandb_resume": True,
            "wandb_run_id_prefix": str(getattr(args, "wandb_run_id_prefix", "") or ""),
            "phase": str(getattr(args, "phase", "") or ""),
            "explicit_user_instruction": True,
            "infrastructure_recovery_attempts": infrastructure_recovery_attempts,
        },
        "observed": {},
    }
    if infrastructure_recovery_attempts < 0 or infrastructure_recovery_attempts > 3:
        record["errors"].append(
            "--infrastructure-resume-attempts must be between 0 and 3"
        )
    if infrastructure_recovery_attempts and not allow_requested:
        record["errors"].append(
            "--infrastructure-resume-attempts requires --allow-wandb-resume"
        )
    if allow_requested and not nonempty_string(
        getattr(args, "wandb_run_id_prefix", None)
    ):
        record["errors"].append(
            "--allow-wandb-resume requires a non-empty --wandb-run-id-prefix"
        )
    base_seconds = float(
        getattr(args, "infrastructure_resume_base_seconds", 30.0) or 0.0
    )
    if not math.isfinite(base_seconds) or base_seconds < 0:
        record["errors"].append(
            "--infrastructure-resume-base-seconds must be a finite non-negative number"
        )
    if not allow_requested:
        record["valid"] = not record["errors"]
        record["status"] = "resume_disabled" if record["valid"] else "invalid"
        return record
    if not config_path:
        record["errors"].append(
            "--allow-wandb-resume requires --wandb-resume-config-json"
        )
        return record

    payload, error = load_json_object(config_path)
    if payload is None:
        record["errors"].append(error or "resume config JSON could not be read")
        return record
    record["observed"] = {
        "allow_wandb_resume": payload.get("allow_wandb_resume"),
        "wandb_run_id_prefix": payload.get("wandb_run_id_prefix"),
        "phase": payload.get("phase"),
        "explicit_user_instruction": payload.get("explicit_user_instruction"),
        "infrastructure_recovery_attempts": payload.get(
            "infrastructure_recovery_attempts"
        ),
        "purpose": payload.get("purpose"),
    }
    if payload.get("allow_wandb_resume") is not True:
        record["errors"].append(
            "wandb resume config must set allow_wandb_resume=true"
        )
    expected_prefix = str(getattr(args, "wandb_run_id_prefix", "") or "")
    if payload.get("wandb_run_id_prefix") != expected_prefix:
        record["errors"].append(
            "wandb resume config wandb_run_id_prefix must match "
            "--wandb-run-id-prefix"
        )
    expected_phase = str(getattr(args, "phase", "") or "")
    if payload.get("phase") != expected_phase:
        record["errors"].append(
            "wandb resume config phase must match --phase"
        )
    if payload.get("explicit_user_instruction") is not True:
        record["errors"].append(
            "wandb resume config must set explicit_user_instruction=true"
        )
    if not nonempty_string(payload.get("purpose")):
        record["errors"].append("wandb resume config must include a non-empty purpose")
    if infrastructure_recovery_attempts > 0 and payload.get(
        "infrastructure_recovery_attempts"
    ) != infrastructure_recovery_attempts:
        record["errors"].append(
            "wandb resume config infrastructure_recovery_attempts must match "
            "--infrastructure-resume-attempts"
        )
    record["valid"] = not record["errors"]
    record["status"] = "valid" if record["valid"] else "invalid"
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
    with log_path.open("a", encoding="utf-8") as log:
        if log.tell() > 0:
            log.write("\n")
        log.write(
            f"===== attempt started at {time.strftime('%Y-%m-%dT%H:%M:%S%z')} =====\n"
        )
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
            start_new_session=True,
        )
        assert proc.stdout is not None
        previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

        def interrupt_on_sigterm(_signum, _frame):
            raise KeyboardInterrupt("Taiwan full evaluation batch received SIGTERM")

        signal.signal(signal.SIGTERM, interrupt_on_sigterm)
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            returncode = proc.wait()
            log.write(
                "===== attempt finished at "
                f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} "
                f"(returncode={returncode}) =====\n"
            )
            log.flush()
            return returncode
        except BaseException:
            log.write(
                "===== attempt interrupted at "
                f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} =====\n"
            )
            log.flush()
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait(timeout=10)
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)


def _resolve_config_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return path.resolve()
    candidate = CONFIG_DIR / path
    return candidate.resolve()


def benchmark_checkpoint_root(
    *,
    base_config: str,
    model_config: Path,
    wandb_run_id: str,
) -> Path:
    merged = OmegaConf.merge(
        OmegaConf.load(_resolve_config_path(base_config)),
        OmegaConf.load(model_config),
    )
    configured = OmegaConf.select(merged, "output.resolved_run_root", default=None)
    if configured:
        run_root = Path(str(configured)).expanduser()
        if not run_root.is_absolute():
            run_root = REPO_ROOT / run_root
    else:
        run_root = REPO_ROOT / "outputs" / "taiwan_full_eval_runs" / wandb_run_id
    return run_root / "benchmark_checkpoints"


def latest_failed_benchmark_checkpoint(
    checkpoint_root: Path,
    *,
    wandb_run_id: str,
    not_before: float | None = None,
) -> dict[str, object] | None:
    candidates: list[dict[str, object]] = []
    for path in checkpoint_root.glob("*.json"):
        payload, _ = load_json_object(path)
        if payload is None:
            continue
        try:
            updated_at = float(payload.get("updated_at") or 0.0)
        except (TypeError, ValueError):
            continue
        if (
            payload.get("status") != "failed"
            or payload.get("run_id") != wandb_run_id
            or (not_before is not None and updated_at < not_before)
        ):
            continue
        failure = classify_benchmark_failure(
            str(payload.get("error_type") or ""),
            str(payload.get("error") or ""),
        )
        candidates.append(
            {
                "path": str(path),
                "benchmark": str(payload.get("benchmark") or path.stem),
                "updated_at": updated_at,
                "error_type": str(payload.get("error_type") or ""),
                "error": str(payload.get("error") or ""),
                "failure_category": failure["category"],
                "infrastructure_retryable": failure[
                    "infrastructure_retryable"
                ],
                "failure_reason": failure["reason"],
            }
        )
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item["updated_at"]))


def run_eval_with_infrastructure_recovery(
    command: list[str],
    *,
    log_path: Path,
    env: dict[str, str],
    checkpoint_root: Path,
    wandb_run_id: str,
    max_resume_attempts: int,
    base_delay_seconds: float,
) -> tuple[int, list[dict[str, object]]]:
    attempts: list[dict[str, object]] = []
    maximum = max(0, int(max_resume_attempts))
    for attempt_index in range(maximum + 1):
        attempt_started_at = time.time()
        returncode = stream_run(command, log_path, env)
        record: dict[str, object] = {
            "attempt": attempt_index + 1,
            "returncode": returncode,
            "started_as_resume": attempt_index > 0,
            "started_at": attempt_started_at,
            "ended_at": time.time(),
        }
        if returncode == 0:
            record["action"] = "completed"
            attempts.append(record)
            return returncode, attempts

        failure = latest_failed_benchmark_checkpoint(
            checkpoint_root,
            wandb_run_id=wandb_run_id,
            not_before=attempt_started_at - 1.0,
        )
        record["failure"] = failure
        retries_remaining = maximum - attempt_index
        if not failure:
            record["action"] = "stopped_no_failed_checkpoint"
        elif not failure["infrastructure_retryable"]:
            record["action"] = "stopped_non_infrastructure_failure"
        elif retries_remaining <= 0:
            record["action"] = "stopped_recovery_exhausted"
        else:
            delay = max(0.0, float(base_delay_seconds)) * (2**attempt_index)
            record["action"] = "resume_after_cooldown"
            record["cooldown_seconds"] = delay
            attempts.append(record)
            print(
                "Infrastructure failure detected in "
                f"{failure['benchmark']} ({failure['error_type']}). "
                f"Resuming the same W&B run after {delay:.1f}s; "
                f"{retries_remaining} authorized attempt(s) remain.",
                flush=True,
            )
            if delay:
                time.sleep(delay)
            continue
        attempts.append(record)
        return returncode, attempts
    raise AssertionError("infrastructure recovery loop ended unexpectedly")


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
        provider, remainder = value.split("/", 1)
        if (
            provider.endswith("-direct")
            or provider in {"openrouter", "wandb-inference"}
        ) and remainder:
            aliases.append(remainder)
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


def weave_required_texts_for_phase(
    *,
    phase: str,
    require_nemoclaw_agentic_config: bool,
) -> list[str]:
    if phase in AGENTIC_GENERATION_PHASES and require_nemoclaw_agentic_config:
        return [NEMOCLAW_OPENCLAW_CONFIG_SOURCE_TRACE_TEXT]
    return []


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
    require_agentic_swe_assorted: bool = False,
    expected_total_override: int | None = None,
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
    expected_total = (
        expected_total_override
        if expected_total_override is not None
        else DEFAULT_EXPECTED_TOTALS.get(benchmark)
    )
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
    if require_agentic_swe_assorted and benchmark == "agentic_swe":
        command.append("--require-agentic-swe-assorted")
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
    require_agentic_swe_assorted: bool = False,
    expected_total_override: int | None = None,
    env_file: Path | None = None,
    timeout_seconds: float = 600.0,
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
        require_agentic_swe_assorted=require_agentic_swe_assorted,
        expected_total_override=expected_total_override,
        env_file=env_file,
        json_path=output_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        payload = {
            "ok": False,
            "payload_ok": False,
            "returncode_ok": False,
            "benchmark": benchmark,
            "run_id": run_id,
            "checks": [],
            "timeout": True,
            "timeout_seconds": timeout_seconds,
            "command": command,
            "returncode": 124,
            "stdout": subprocess_output_text(exc.stdout),
            "stderr": subprocess_output_text(exc.stderr),
        }
        write_json(output_path, payload)
        return payload
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


def expected_total_for_config(config_path: Path, benchmark: str) -> int | None:
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(config, dict):
        return DEFAULT_EXPECTED_TOTALS.get(benchmark)
    if benchmark == "agentic_math":
        section = config.get("agentic_math")
        if isinstance(section, dict) and section.get("limit") is not None:
            return int(section["limit"])
    if benchmark == "agentic_swe":
        run = config.get("run")
        assorted_enabled = isinstance(run, dict) and bool(run.get("agentic_swe_assorted"))
        section = config.get("agentic_swe_assorted")
        if assorted_enabled and isinstance(section, dict):
            low_middle = 0
            if not bool(section.get("skip_low_middle", False)):
                if section.get("low_middle_limit") is not None:
                    low_middle = int(section["low_middle_limit"])
                else:
                    low_middle = int(section.get("low_limit", 20) or 0) + int(
                        section.get("middle_limit", 20) or 0
                    )
            high = 0
            if not bool(section.get("skip_high", False)):
                high = int(section.get("high_limit", 10) or 0)
            return low_middle + high
    return DEFAULT_EXPECTED_TOTALS.get(benchmark)


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
    required_texts: list[str] | None = None,
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
    for text in sorted(dict.fromkeys(required_texts or [])):
        command.extend(["--require-text", text])
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


def normalize_run_eval_base_config_arg(value: str) -> str:
    """Accept either config-dir-relative or repository-relative base config paths."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    if path.exists():
        return str(path.resolve())
    return value


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
                "expected_scheduled_evaluators": expected_scheduled_evaluators_for_phase(phase),
                "command": build_run_eval_preflight_command(
                    python=python,
                    base_config=base_config,
                    config=rel_config,
                    output_json=output_json,
                ),
            }
        )
    return records


def execute_run_eval_preflight(
    *,
    command: list[str],
    output_json: Path,
    log_path: Path,
    env: dict[str, str],
    phase: str,
) -> dict[str, object]:
    """Run one static preflight and reject stale or phase-incomplete evidence."""
    output_json.unlink(missing_ok=True)
    returncode = stream_run(command, log_path, env)
    payload, payload_error = load_json_object(output_json)
    status = ""
    phase_validation = {
        "ok": False,
        "phase": phase,
        "expected_scheduled_evaluators": expected_scheduled_evaluators_for_phase(phase),
        "observed_scheduled_evaluators": [],
        "missing_scheduled_evaluators": expected_scheduled_evaluators_for_phase(phase),
        "unexpected_scheduled_evaluators": [],
        "order_matches": False,
    }
    ok = returncode == 0
    if payload is None:
        status = payload_error or "preflight JSON could not be read"
        ok = False
    else:
        ok = ok and payload.get("ok") is True
        status = str(payload.get("status") or "")
        if payload.get("ok") is True:
            phase_validation = validate_run_eval_preflight_phase(payload, phase)
            ok = ok and bool(phase_validation["ok"])
            if not phase_validation["ok"]:
                status = (
                    "preflight scheduled evaluator mismatch: "
                    + json.dumps(
                        phase_validation,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
    return {
        "ok": ok,
        "returncode": returncode,
        "status": status,
        "phase_validation": phase_validation,
        "output_json": str(output_json),
        "log_path": str(log_path),
    }


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
    required_texts: list[str] | None = None,
    timeout_seconds: float = 600.0,
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
        required_texts=required_texts,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        payload = {
            "ok": False,
            "payload_ok": False,
            "returncode_ok": False,
            "agent_name": agent_name,
            "checks": [],
            "timeout": True,
            "timeout_seconds": timeout_seconds,
            "command": command,
            "returncode": 124,
            "stdout": subprocess_output_text(exc.stdout),
            "stderr": subprocess_output_text(exc.stderr),
        }
        write_json(output_path, payload)
        return payload
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
            "required_texts": list(required_texts or []),
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
    parser.add_argument(
        "--include-suspended",
        action="store_true",
        help=(
            "Explicitly allow models marked suspended=true. Use only after resolving "
            "the suspension reason recorded in the manifest."
        ),
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
            "Use a fresh prefix for each paid run unless --allow-wandb-resume is "
            "explicitly requested."
        ),
    )
    parser.add_argument(
        "--allow-wandb-resume",
        action="store_true",
        help=(
            "Explicitly allow W&B run resume for the supplied --wandb-run-id-prefix. "
            "By default, the batch runner sets WANDB_RESUME=never so stale/crashed "
            "runs are not silently reused."
        ),
    )
    parser.add_argument(
        "--wandb-resume-config-json",
        type=Path,
        help=(
            "Required with --allow-wandb-resume. JSON must explicitly bind the "
            "resume request to the phase, W&B run id prefix, purpose, and user "
            "instruction acknowledgement."
        ),
    )
    parser.add_argument(
        "--infrastructure-resume-attempts",
        type=int,
        default=0,
        help=(
            "Automatically resume the same W&B run only after a checkpointed, "
            "known infrastructure failure. Requires --allow-wandb-resume and an "
            "exact matching value in the resume config. Maximum 3."
        ),
    )
    parser.add_argument(
        "--infrastructure-resume-base-seconds",
        type=float,
        default=30.0,
        help="Cooldown before the first authorized infrastructure-only resume.",
    )
    parser.add_argument(
        "--allow-cash-cost-exempt-execution",
        action="store_true",
        help=(
            "Authorize external execution without paid-API budget packets only "
            "when every selected generated config declares "
            "execution.cash_cost_exempt=true with a non-empty reason."
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
        "--verification-timeout-seconds",
        type=float,
        default=600.0,
        help="Independent timeout for each post-run W&B or Weave verifier.",
    )
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
            "route Agentic Math and Agentic SWE-Assorted through NeMoClaw."
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
    parser.add_argument("--agentic-swe-assorted-nemoclaw-sandbox")
    parser.add_argument("--agentic-swe-assorted-nemoclaw-bin", default="nemoclaw")
    parser.add_argument("--agentic-swe-assorted-nemoclaw-workdir", default="/sandbox")
    parser.add_argument(
        "--agentic-swe-assorted-nemoclaw-checkout-transfer-mode",
        choices=["visible", "copy"],
    )
    parser.add_argument("--agentic-swe-assorted-nemoclaw-openclaw-config-path")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--yes", action="store_true", help="Pass --yes to scripts/run_eval.py.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.base_config = normalize_run_eval_base_config_arg(args.base_config)
    if args.verification_timeout_seconds <= 0:
        raise SystemExit("--verification-timeout-seconds must be positive")
    if args.weave_content_canary_max_age_seconds < 0:
        args.weave_content_canary_max_age_seconds = None
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.output_dir = args.generated_config_dir
    configs = generate_configs(args)
    configs = configs[args.start_index :]
    if args.limit is not None:
        configs = configs[: args.limit]

    selected_config_model_bindings = collect_selected_config_model_bindings(
        configs,
        phase=args.phase,
    )
    phase_executes_model_api = args.phase in {"full", "nonagentic", "agentic"}
    selected_configs_cash_cost_exempt = bool(selected_config_model_bindings) and all(
        binding.get("cash_cost_exempt") is True
        and nonempty_string(binding.get("cash_cost_exempt_reason"))
        and not binding.get("error")
        for binding in selected_config_model_bindings
    )
    cash_cost_exempt_execution = bool(
        args.allow_cash_cost_exempt_execution
        and selected_configs_cash_cost_exempt
    )
    phase_requires_cash_budget = bool(
        phase_executes_model_api and not cash_cost_exempt_execution
    )
    will_call_model_api = not args.prepare_only and phase_executes_model_api
    will_call_paid_model_api = (
        not args.prepare_only and phase_requires_cash_budget
    )
    will_execute_external_actions = not args.prepare_only and (
        phase_executes_model_api
        or bool(args.verify_wandb_completion)
        or bool(args.verify_weave_agents)
    )
    external_action_approval_required = bool(
        will_execute_external_actions and not cash_cost_exempt_execution
    )
    weave_agents_required_texts = weave_required_texts_for_phase(
        phase=args.phase,
        require_nemoclaw_agentic_config=bool(args.require_nemoclaw_agentic_config),
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
        required_before_paid_execution=phase_requires_cash_budget,
        selected_config_model_bindings=selected_config_model_bindings,
    )
    external_action_approval = build_external_action_approval_record(
        args.external_action_approval_report_json,
        required_before_external_action=external_action_approval_required,
        expected_source_packet_path=args.external_action_approval_source_packet_json,
    )
    budget_approval_alignment = build_budget_approval_alignment_record(
        pre_run_budget_estimate=pre_run_budget_estimate,
        external_action_approval=external_action_approval,
        required_before_paid_execution=will_call_paid_model_api,
    )
    wandb_resume_policy = build_wandb_resume_policy_record(args)
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
        "include_suspended": bool(getattr(args, "include_suspended", False)),
        "run_purpose": args.run_purpose or "",
        "expected_cost_band": args.expected_cost_band or "",
        "executes_model_api": phase_executes_model_api,
        "requires_paid_model_api": phase_requires_cash_budget,
        "will_call_model_api": will_call_model_api,
        "will_call_paid_model_api": will_call_paid_model_api,
        "will_execute_external_actions": will_execute_external_actions,
        "external_action_approval_required": external_action_approval_required,
        "allow_cash_cost_exempt_execution": bool(
            args.allow_cash_cost_exempt_execution
        ),
        "selected_configs_cash_cost_exempt": selected_configs_cash_cost_exempt,
        "cash_cost_exempt_execution": cash_cost_exempt_execution,
        "model_count": len(configs),
        "configs": [config_arg(path) for path in configs],
        "selected_config_model_bindings": selected_config_model_bindings,
        "agentic_production_evidence_guard": agentic_production_evidence_guard,
        "nemoclaw_agentic_config_guard": nemoclaw_agentic_config_guard,
        "wandb_run_id_prefix": args.wandb_run_id_prefix or "",
        "allow_wandb_resume": bool(args.allow_wandb_resume),
        "infrastructure_resume_attempts": args.infrastructure_resume_attempts,
        "infrastructure_resume_base_seconds": args.infrastructure_resume_base_seconds,
        "wandb_resume_policy": wandb_resume_policy,
        "verify_wandb_completion": bool(args.verify_wandb_completion),
        "wandb_verify_benchmarks": args.wandb_verify_benchmark
        or default_wandb_verify_benchmarks(args.phase),
        "verify_weave_agents": bool(args.verify_weave_agents),
        "weave_agent_name": args.weave_agent_name,
        "weave_agents_require_content": not bool(args.weave_agents_no_require_content),
        "weave_agents_require_tool_span": bool(args.weave_agents_require_tool_span),
        "weave_agents_require_tool_content": bool(args.weave_agents_require_tool_content),
        "weave_agents_require_usage": bool(args.weave_agents_require_usage),
        "weave_agents_required_texts": weave_agents_required_texts,
        "weave_agents_conversation_id_contains": args.weave_agents_conversation_id_contains or "",
        "weave_content_canary_gate": weave_content_canary_gate,
        "run_eval_preflights": run_eval_preflights,
        "pre_run_budget_estimate": pre_run_budget_estimate,
        "external_action_approval": external_action_approval,
        "budget_approval_alignment": budget_approval_alignment,
        "wandb_resume_policy": wandb_resume_policy,
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
        "include_suspended": bool(getattr(args, "include_suspended", False)),
        "executes_model_api": phase_executes_model_api,
        "requires_paid_model_api": phase_requires_cash_budget,
        "will_call_model_api": will_call_model_api,
        "will_call_paid_model_api": will_call_paid_model_api,
        "will_execute_external_actions": will_execute_external_actions,
        "external_action_approval_required": external_action_approval_required,
        "allow_cash_cost_exempt_execution": bool(
            args.allow_cash_cost_exempt_execution
        ),
        "selected_configs_cash_cost_exempt": selected_configs_cash_cost_exempt,
        "cash_cost_exempt_execution": cash_cost_exempt_execution,
        "run_purpose": args.run_purpose or "",
        "expected_cost_band": args.expected_cost_band or "",
        "model_count": len(configs),
        "configs": execution_plan["configs"],
        "selected_config_model_bindings": selected_config_model_bindings,
        "agentic_production_evidence_guard": agentic_production_evidence_guard,
        "nemoclaw_agentic_config_guard": nemoclaw_agentic_config_guard,
        "wandb_run_id_prefix": args.wandb_run_id_prefix or "",
        "allow_wandb_resume": bool(args.allow_wandb_resume),
        "infrastructure_resume_attempts": args.infrastructure_resume_attempts,
        "infrastructure_resume_base_seconds": args.infrastructure_resume_base_seconds,
        "wandb_resume_policy": wandb_resume_policy,
        "verify_wandb_completion": bool(args.verify_wandb_completion),
        "wandb_verify_benchmarks": args.wandb_verify_benchmark
        or default_wandb_verify_benchmarks(args.phase),
        "verify_weave_agents": bool(args.verify_weave_agents),
        "weave_agent_name": args.weave_agent_name,
        "weave_agents_require_content": not bool(args.weave_agents_no_require_content),
        "weave_agents_require_tool_span": bool(args.weave_agents_require_tool_span),
        "weave_agents_require_tool_content": bool(args.weave_agents_require_tool_content),
        "weave_agents_require_usage": bool(args.weave_agents_require_usage),
        "weave_agents_required_texts": weave_agents_required_texts,
        "weave_agents_conversation_id_contains": args.weave_agents_conversation_id_contains or "",
        "weave_content_canary_gate": weave_content_canary_gate,
        "run_eval_preflights": run_eval_preflights,
        "pre_run_budget_estimate": pre_run_budget_estimate,
        "external_action_approval": external_action_approval,
        "budget_approval_alignment": budget_approval_alignment,
        "completion_requirements": {
            "status": "completed",
            "pre_run_budget_estimate": {
                "required_before_paid_execution": phase_requires_cash_budget,
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
                "required_before_external_action": external_action_approval_required,
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
            "wandb_resume_policy": {
                "default": "resume disabled",
                "required_when_allow_wandb_resume": True,
                "required_fields": [
                    "allow_wandb_resume=true",
                    "wandb_run_id_prefix matches --wandb-run-id-prefix",
                    "phase matches --phase",
                    "explicit_user_instruction=true",
                    "purpose is non-empty",
                    "infrastructure_recovery_attempts matches the CLI when enabled",
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
                "required_texts": weave_agents_required_texts,
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

    if (
        args.allow_cash_cost_exempt_execution
        and not selected_configs_cash_cost_exempt
    ):
        review_record["status"] = "cash_cost_exemption_invalid"
        review_record["blocking_reason"] = (
            "Every selected generated config must declare "
            "execution.cash_cost_exempt=true and a non-empty "
            "execution.cash_cost_exempt_reason."
        )
        write_json(review_path, review_record)
        raise SystemExit(str(review_record["blocking_reason"]))

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

    if (
        external_action_approval_required
        and not external_action_approval.get("valid")
    ):
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

    if not wandb_resume_policy.get("valid"):
        review_record["status"] = "wandb_resume_policy_failed"
        review_record["blocking_reason"] = wandb_resume_policy
        write_json(review_path, review_record)
        raise SystemExit(
            "W&B resume requires an explicit resume config and user instruction: "
            + "; ".join(str(item) for item in wandb_resume_policy["errors"])
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

    env = load_env_file(os.environ.copy(), args.env_file)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("WANDB_CONSOLE", "wrap")

    manifest_rows = []
    if args.prepare_only:
        for index, record in enumerate(run_eval_preflights, start=1):
            command = list(record["command"])
            output_json = Path(str(record["output_json"]))
            config_name = str(record["config"])
            slug = Path(config_name).stem.replace("config-taiwan-full-", "")
            log_path = args.output_root / "logs" / (
                f"{args.phase}-{slug}.prepare-preflight.log"
            )
            print(
                f"\n[{index}/{len(run_eval_preflights)}] static preflight: "
                f"{config_name}",
                flush=True,
            )
            result = execute_run_eval_preflight(
                command=command,
                output_json=output_json,
                log_path=log_path,
                env=env,
                phase=args.phase,
            )
            record.update(
                {
                    "executed": True,
                    "returncode": result["returncode"],
                    "ok": result["ok"],
                    "status": result["status"],
                    "phase_validation": result["phase_validation"],
                    "log_path": result["log_path"],
                }
            )
            write_json(review_path, review_record)
            if not result["ok"]:
                review_record["status"] = "prepare_preflight_failed"
                review_record["blocking_reason"] = record
                write_json(review_path, review_record)
                raise SystemExit(
                    f"prepare-only preflight failed for {config_name}: "
                    f"{result['status']}"
                )
        review_record["status"] = "prepared"
        review_record["ended_at"] = time.time()
        write_json(review_path, review_record)
        for path in configs:
            print(config_arg(path))
        return

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
            if args.allow_wandb_resume:
                run_env["WANDB_RESUME"] = "allow"
                run_env["NEJUMI_ALLOW_WANDB_RESUME"] = "1"
            else:
                run_env["WANDB_RESUME"] = "never"
                run_env.pop("NEJUMI_ALLOW_WANDB_RESUME", None)
        if review_record["started_at"] is None:
            review_record["started_at"] = started_at
        review_record["status"] = "running"
        write_json(review_path, review_record)
        preflight_log_path = args.output_root / "logs" / f"{args.phase}-{slug}.preflight.log"
        preflight_result = execute_run_eval_preflight(
            command=preflight_command,
            output_json=preflight_json,
            log_path=preflight_log_path,
            env=run_env,
            phase=args.phase,
        )
        preflight_returncode = int(preflight_result["returncode"])
        preflight_ok = bool(preflight_result["ok"])
        preflight_status = str(preflight_result["status"])
        preflight_phase_validation = preflight_result["phase_validation"]
        if not preflight_ok:
            row = {
                "config": rel_config,
                "preflight_command": preflight_command,
                "preflight_log_path": str(preflight_log_path),
                "preflight_json": str(preflight_json),
                "preflight_returncode": preflight_returncode,
                "preflight_ok": False,
                "preflight_status": preflight_status,
                "preflight_phase_validation": preflight_phase_validation,
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
        checkpoint_root = benchmark_checkpoint_root(
            base_config=args.base_config,
            model_config=config_path,
            wandb_run_id=run_env.get("WANDB_RUN_ID", ""),
        )
        returncode, process_attempts = run_eval_with_infrastructure_recovery(
            command,
            log_path=log_path,
            env=run_env,
            checkpoint_root=checkpoint_root,
            wandb_run_id=run_env.get("WANDB_RUN_ID", ""),
            max_resume_attempts=args.infrastructure_resume_attempts,
            base_delay_seconds=args.infrastructure_resume_base_seconds,
        )
        row = {
            "config": rel_config,
            "preflight_command": preflight_command,
            "preflight_log_path": str(preflight_log_path),
            "preflight_json": str(preflight_json),
            "preflight_returncode": preflight_returncode,
            "preflight_ok": preflight_ok,
            "preflight_status": preflight_status,
            "preflight_phase_validation": preflight_phase_validation,
            "log_path": str(log_path),
            "phase": args.phase,
            "wandb_run_id": run_env.get("WANDB_RUN_ID", ""),
            "wandb_entity": run_env.get("WANDB_ENTITY", ""),
            "wandb_project": run_env.get("WANDB_PROJECT", ""),
            "returncode": returncode,
            "process_attempts": process_attempts,
            "benchmark_checkpoint_root": str(checkpoint_root),
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
                        require_agentic_swe_assorted=(benchmark == "agentic_swe"),
                        expected_total_override=expected_total_for_config(
                            config_path, benchmark
                        ),
                        env_file=args.env_file,
                        timeout_seconds=args.verification_timeout_seconds,
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
                required_texts=weave_agents_required_texts,
                timeout_seconds=args.verification_timeout_seconds,
            )
            row["weave_agents_completion"] = {
                "ok": bool(weave_result.get("ok")),
                "path": str(weave_output_path),
                "agent_name": args.weave_agent_name,
                "run_id": row["wandb_run_id"],
                "conversation_id_contains": weave_conversation_id_contains,
                "required_texts": weave_agents_required_texts,
                "expected_request_models": weave_expected_request_models(
                    config_path,
                    phase=args.phase,
                ),
            }
            if not weave_result.get("ok"):
                row["returncode"] = 1
                row["weave_agents_completion_failed"] = True
                returncode = 1
        row["ended_at"] = time.time()
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
