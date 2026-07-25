#!/usr/bin/env python3
"""
Generate Taiwan full-evaluation configs from existing model configs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_MANIFEST = CONFIG_DIR / "taiwan_openai_canary_models.yaml"
DEFAULT_OUTPUT_DIR = CONFIG_DIR / "taiwan_full" / "generated"
DEFAULT_NEMOCLAW_OPENCLAW_CONFIG_PATH = "/sandbox/.openclaw/openclaw.json"
TAIWAN_WANDB_ENTITY = "llm-leaderboard"
TAIWAN_WANDB_PROJECT = "tc-leaderboard"


RUN_FLAGS: dict[str, bool] = {
    "agentic_math": True,
    "bfcl": True,
    "swebench": False,
    "swebench_pro": False,
    "deepswe": False,
    "agentic_swe_assorted": True,
    "mtbench": True,
    "jbbq": False,
    "toxicity": False,
    "jtruthfulqa": False,
    "hle": True,
    "hallulens": False,
    "hallulens_zh_tw": True,
    "arc_agi": True,
    "m_ifeval": False,
    "ifeval_zh_tw": True,
    "ts_bench": True,
    "twbias": False,
    "tceval_v2": True,
    "script_adherence": True,
    "jaster": True,
    "jmmlu_robustness": False,
    "tmmluplus_robustness": True,
    "aggregate": False,
    "aggregate_taiwan": True,
}

PHASE_CHOICES = ("full", "nonagentic", "agentic", "agentic_aggregate")

AGENTIC_DENIED_TOOLS = [
    "code_execution",
    "process",
    "process_*",
    "web_search",
    "web_fetch",
    "browser",
    "browser_*",
]
AGENTIC_DENIED_ARGUMENT_PATTERNS = [
    r"https?://",
    r"\b(curl|wget)\b",
    r"\b(?:python(?:3)?\s+-m\s+)?pip(?:3)?\s+install\b",
    r"\b(requests|urllib|httpx)\.",
]
AGENTIC_SWE_DENIED_ARGUMENT_PATTERNS = [
    r"https?://",
    r"\b(?:ftp|sftp|ssh)://",
    r"\bgit\+",
    r"\bgit\s+(?:clone|fetch|pull|ls-remote)\b",
    r"\b(curl|wget)\b",
]
AGENTIC_SWE_ASSORTED_DENIED_TOOLS = [
    "code_execution",
    "web_search",
    "web_fetch",
    "browser",
    "browser_*",
]
AGENTIC_SWE_ASSORTED_DENIED_ARGUMENT_PATTERNS = [
    *AGENTIC_SWE_DENIED_ARGUMENT_PATTERNS,
]
MANIFEST_JUDGE_OVERRIDE_KEYS = (
    "judge_model",
    "judge_parallel",
    "judge_params",
)


def _plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _optional_plain_dict(value: Any, *, field_name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    plain = _plain(value)
    if not isinstance(plain, dict):
        raise ValueError(f"{field_name} must be a mapping/object")
    return plain


def _openclaw_model_params(model: dict[str, Any], section: str) -> dict[str, Any] | None:
    section_key = f"{section}_openclaw_model_params"
    if section_key in model:
        return _optional_plain_dict(model.get(section_key), field_name=section_key)
    return _optional_plain_dict(model.get("openclaw_model_params"), field_name="openclaw_model_params")


def _openclaw_model_overrides(model: dict[str, Any], section: str) -> dict[str, Any] | None:
    section_key = f"{section}_openclaw_model_overrides"
    if section_key in model:
        return _optional_plain_dict(model.get(section_key), field_name=section_key)
    return _optional_plain_dict(
        model.get("openclaw_model_overrides"),
        field_name="openclaw_model_overrides",
    )


def _reject_manifest_judge_overrides(model: dict[str, Any]) -> None:
    present = [key for key in MANIFEST_JUDGE_OVERRIDE_KEYS if key in model]
    if not present:
        return
    slug = str(model.get("slug") or "<unknown>")
    raise ValueError(
        "Manifest-level judge overrides are forbidden for Taiwan production-spec "
        "full evaluation configs. Change judge settings only in "
        "configs/base_config_taiwan.yaml after explicit user approval. "
        f"model={slug}, keys={', '.join(present)}"
    )


def _run_flags_for_phase(phase: str) -> dict[str, bool]:
    flags = dict(RUN_FLAGS)
    if phase == "full":
        return flags
    if phase == "nonagentic":
        flags["agentic_math"] = False
        flags["swebench_pro"] = False
        flags["deepswe"] = False
        flags["agentic_swe_assorted"] = False
        flags["aggregate_taiwan"] = False
        return flags
    if phase == "agentic":
        return {key: key in {"agentic_math", "agentic_swe_assorted"} for key in flags}
    if phase == "agentic_aggregate":
        return {
            key: key in {"agentic_math", "agentic_swe_assorted", "aggregate_taiwan"}
            for key in flags
        }
    raise ValueError(f"Unsupported Taiwan eval phase: {phase}")


def _with_cli_nemoclaw_overrides(
    model: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    updated = dict(model)
    sandbox = getattr(args, "agentic_math_nemoclaw_sandbox", None)
    if sandbox:
        updated["agentic_math_nemoclaw_sandbox"] = str(sandbox)
        updated["agentic_math_nemoclaw_bin"] = str(
            getattr(args, "agentic_math_nemoclaw_bin", None) or "nemoclaw"
        )
        updated["agentic_math_nemoclaw_workdir"] = str(
            getattr(args, "agentic_math_nemoclaw_workdir", None) or "/sandbox"
        )
    swe_sandbox = getattr(args, "swebench_pro_nemoclaw_sandbox", None)
    if swe_sandbox:
        updated["swebench_pro_nemoclaw_sandbox"] = str(swe_sandbox)
        updated["swebench_pro_nemoclaw_bin"] = str(
            getattr(args, "swebench_pro_nemoclaw_bin", None) or "nemoclaw"
        )
        checkout_sandbox_root = getattr(args, "swebench_pro_nemoclaw_checkout_sandbox_root", None)
        if checkout_sandbox_root:
            updated["swebench_pro_nemoclaw_checkout_sandbox_root"] = str(checkout_sandbox_root)
        swe_workdir = getattr(args, "swebench_pro_nemoclaw_workdir", None)
        if swe_workdir:
            updated["swebench_pro_nemoclaw_workdir"] = str(swe_workdir)
        swe_transfer_mode = getattr(args, "swebench_pro_nemoclaw_checkout_transfer_mode", None)
        if swe_transfer_mode:
            updated["swebench_pro_nemoclaw_checkout_transfer_mode"] = str(swe_transfer_mode)
        swe_config_path = getattr(args, "swebench_pro_nemoclaw_openclaw_config_path", None)
        if swe_config_path:
            updated["swebench_pro_nemoclaw_openclaw_config_path"] = str(swe_config_path)
    deepswe_sandbox = getattr(args, "deepswe_nemoclaw_sandbox", None)
    if deepswe_sandbox:
        updated["deepswe_nemoclaw_sandbox"] = str(deepswe_sandbox)
        updated["deepswe_nemoclaw_bin"] = str(
            getattr(args, "deepswe_nemoclaw_bin", None) or "nemoclaw"
        )
        deepswe_workdir = getattr(args, "deepswe_nemoclaw_workdir", None)
        if deepswe_workdir:
            updated["deepswe_nemoclaw_workdir"] = str(deepswe_workdir)
        deepswe_config_path = getattr(args, "deepswe_nemoclaw_openclaw_config_path", None)
        if deepswe_config_path:
            updated["deepswe_nemoclaw_openclaw_config_path"] = str(deepswe_config_path)
    assorted_sandbox = getattr(args, "agentic_swe_assorted_nemoclaw_sandbox", None)
    if assorted_sandbox:
        updated["agentic_swe_assorted_nemoclaw_sandbox"] = str(assorted_sandbox)
        updated["agentic_swe_assorted_nemoclaw_bin"] = str(
            getattr(args, "agentic_swe_assorted_nemoclaw_bin", None) or "nemoclaw"
        )
        assorted_workdir = getattr(args, "agentic_swe_assorted_nemoclaw_workdir", None)
        if assorted_workdir:
            updated["agentic_swe_assorted_nemoclaw_workdir"] = str(assorted_workdir)
        assorted_transfer_mode = getattr(
            args, "agentic_swe_assorted_nemoclaw_checkout_transfer_mode", None
        )
        if assorted_transfer_mode:
            updated["agentic_swe_assorted_nemoclaw_checkout_transfer_mode"] = str(
                assorted_transfer_mode
            )
        assorted_config_path = getattr(
            args, "agentic_swe_assorted_nemoclaw_openclaw_config_path", None
        )
        if assorted_config_path:
            updated["agentic_swe_assorted_nemoclaw_openclaw_config_path"] = str(
                assorted_config_path
            )
    math_config_path = getattr(args, "agentic_math_nemoclaw_openclaw_config_path", None)
    if sandbox:
        updated["agentic_math_nemoclaw_openclaw_config_path"] = str(
            math_config_path or DEFAULT_NEMOCLAW_OPENCLAW_CONFIG_PATH
        )
    return updated


def _apply_agentic_math_nemoclaw_config(
    agentic_math: dict[str, Any],
    model: dict[str, Any],
) -> None:
    sandbox = model.get("agentic_math_nemoclaw_sandbox") or model.get("nemoclaw_sandbox")
    if not sandbox:
        return
    agentic_math["nemoclaw_sandbox"] = str(sandbox)
    agentic_math["nemoclaw_bin"] = str(
        model.get("agentic_math_nemoclaw_bin") or model.get("nemoclaw_bin") or "nemoclaw"
    )
    agentic_math["nemoclaw_workdir"] = str(
        model.get("agentic_math_nemoclaw_workdir") or model.get("nemoclaw_workdir") or "/sandbox"
    )
    agentic_math["use_task_agent"] = bool(model.get("agentic_math_use_task_agent", True))
    config_path = (
        model.get("agentic_math_nemoclaw_openclaw_config_path")
        or model.get("nemoclaw_openclaw_config_path")
        or DEFAULT_NEMOCLAW_OPENCLAW_CONFIG_PATH
    )
    agentic_math["nemoclaw_openclaw_config_path"] = str(config_path)


def _apply_swebench_pro_nemoclaw_config(
    swebench_pro: dict[str, Any],
    model: dict[str, Any],
) -> None:
    sandbox = model.get("swebench_pro_nemoclaw_sandbox")
    if not sandbox:
        return
    swebench_pro["nemoclaw_sandbox"] = str(sandbox)
    swebench_pro["nemoclaw_bin"] = str(
        model.get("swebench_pro_nemoclaw_bin") or model.get("nemoclaw_bin") or "nemoclaw"
    )
    checkout_sandbox_root = model.get("swebench_pro_nemoclaw_checkout_sandbox_root")
    if checkout_sandbox_root:
        swebench_pro["nemoclaw_checkout_sandbox_root"] = str(checkout_sandbox_root)
    workdir = model.get("swebench_pro_nemoclaw_workdir")
    if workdir:
        swebench_pro["nemoclaw_workdir"] = str(workdir)
    transfer_mode = model.get("swebench_pro_nemoclaw_checkout_transfer_mode")
    if transfer_mode:
        swebench_pro["nemoclaw_checkout_transfer_mode"] = str(transfer_mode)
    transfer_timeout = model.get("swebench_pro_nemoclaw_checkout_transfer_timeout")
    if transfer_timeout:
        swebench_pro["nemoclaw_checkout_transfer_timeout"] = int(transfer_timeout)
    config_path = (
        model.get("swebench_pro_nemoclaw_openclaw_config_path")
        or model.get("nemoclaw_openclaw_config_path")
        or DEFAULT_NEMOCLAW_OPENCLAW_CONFIG_PATH
    )
    swebench_pro["nemoclaw_openclaw_config_path"] = str(config_path)


def _apply_deepswe_nemoclaw_config(
    deepswe: dict[str, Any],
    model: dict[str, Any],
) -> None:
    sandbox = model.get("deepswe_nemoclaw_sandbox") or model.get("nemoclaw_sandbox")
    if not sandbox:
        return
    deepswe["nemoclaw_sandbox"] = str(sandbox)
    deepswe["nemoclaw_bin"] = str(
        model.get("deepswe_nemoclaw_bin") or model.get("nemoclaw_bin") or "nemoclaw"
    )
    deepswe["nemoclaw_workdir"] = str(
        model.get("deepswe_nemoclaw_workdir") or model.get("nemoclaw_workdir") or "/sandbox"
    )
    config_path = (
        model.get("deepswe_nemoclaw_openclaw_config_path")
        or model.get("nemoclaw_openclaw_config_path")
        or DEFAULT_NEMOCLAW_OPENCLAW_CONFIG_PATH
    )
    deepswe["nemoclaw_openclaw_config_path"] = str(config_path)


def _apply_agentic_swe_assorted_nemoclaw_config(
    agentic_swe_assorted: dict[str, Any],
    model: dict[str, Any],
) -> None:
    sandbox = model.get("agentic_swe_assorted_nemoclaw_sandbox") or model.get("nemoclaw_sandbox")
    if not sandbox:
        return
    agentic_swe_assorted["nemoclaw_sandbox"] = str(sandbox)
    agentic_swe_assorted["nemoclaw_bin"] = str(
        model.get("agentic_swe_assorted_nemoclaw_bin")
        or model.get("nemoclaw_bin")
        or "nemoclaw"
    )
    agentic_swe_assorted["nemoclaw_workdir"] = str(
        model.get("agentic_swe_assorted_nemoclaw_workdir")
        or model.get("nemoclaw_workdir")
        or "/sandbox"
    )
    transfer_mode = (
        model.get("agentic_swe_assorted_nemoclaw_checkout_transfer_mode")
        or model.get("nemoclaw_checkout_transfer_mode")
        or "copy"
    )
    agentic_swe_assorted["nemoclaw_checkout_transfer_mode"] = str(transfer_mode)
    transfer_timeout = model.get("agentic_swe_assorted_nemoclaw_checkout_transfer_timeout")
    if transfer_timeout:
        agentic_swe_assorted["nemoclaw_checkout_transfer_timeout"] = int(transfer_timeout)
    config_path = (
        model.get("agentic_swe_assorted_nemoclaw_openclaw_config_path")
        or model.get("nemoclaw_openclaw_config_path")
        or DEFAULT_NEMOCLAW_OPENCLAW_CONFIG_PATH
    )
    agentic_swe_assorted["nemoclaw_openclaw_config_path"] = str(config_path)


def build_override(
    model: dict[str, Any],
    output_root: Path,
    phase: str = "full",
) -> dict[str, Any]:
    _reject_manifest_judge_overrides(model)
    slug = str(model["slug"])
    openclaw_model = str(model["openclaw_model"])
    reasoning_effort = model.get("reasoning_effort")
    agentic_math_model_params = _openclaw_model_params(model, "agentic_math")
    swebench_pro_model_params = _openclaw_model_params(model, "swebench_pro")
    deepswe_model_params = _openclaw_model_params(model, "deepswe")
    agentic_swe_assorted_model_params = _openclaw_model_params(
        model, "agentic_swe_assorted"
    )
    agentic_math_model_overrides = _openclaw_model_overrides(model, "agentic_math")
    swebench_pro_model_overrides = _openclaw_model_overrides(model, "swebench_pro")
    deepswe_model_overrides = _openclaw_model_overrides(model, "deepswe")
    agentic_swe_assorted_model_overrides = _openclaw_model_overrides(
        model, "agentic_swe_assorted"
    )
    math_limit = model.get("math_limit", 50)
    if math_limit is None:
        math_limit = 50
    math_run_openclaw = bool(
        model.get(
            "agentic_math_run_openclaw",
            phase != "agentic_aggregate",
        )
    )
    math_results_dir = model.get("agentic_math_results_dir")
    if not math_run_openclaw and not math_results_dir:
        math_results_dir = output_root / "agentic_math" / slug / "openclaw"
    bfcl_run_generation = bool(model.get("bfcl_run_generation", True))
    bfcl_result_dir = model.get("bfcl_result_dir")
    if not bfcl_run_generation and not bfcl_result_dir:
        raise ValueError(
            f"Model {slug!r} sets bfcl_run_generation=false but has no "
            "bfcl_result_dir"
        )

    override: dict[str, Any] = {
        "testmode": False,
        "wandb": {
            "entity": TAIWAN_WANDB_ENTITY,
            "project": TAIWAN_WANDB_PROJECT,
            "expected_entity": TAIWAN_WANDB_ENTITY,
            "expected_project": TAIWAN_WANDB_PROJECT,
            "run_name": str(model["run_name"]),
        },
        "run": _run_flags_for_phase(phase),
        "provider_rate_limit": {
            "enabled": bool(model.get("provider_rate_limit_enabled", True)),
            "key": str(model.get("provider_rate_limit_key", f"llm:{slug}")),
            "min_request_interval_sec": float(
                model.get("provider_min_request_interval_sec", 1.0)
            ),
            "request_jitter_sec": float(model.get("provider_request_jitter_sec", 0.25)),
        },
        "agentic_math": {
            "subset": str(model.get("math_subset", "leaderboard")),
            "limit": int(math_limit),
            "output_dir": str(output_root / "agentic_math" / slug),
            "prefix": f"taiwan-math-{slug}",
            "task_agent_prefix": f"tw-math-{slug}",
            "openclaw_model": openclaw_model,
            **({"openclaw_model_params": agentic_math_model_params} if agentic_math_model_params else {}),
            **(
                {"openclaw_model_overrides": agentic_math_model_overrides}
                if agentic_math_model_overrides
                else {}
            ),
            "thinking": str(model.get("agentic_thinking", "high")),
            "num_workers": int(model.get("math_num_workers", 8)),
            "task_start_min_interval_seconds": float(
                model.get("math_task_start_min_interval_seconds", 5.0)
            ),
            "openclaw_timeout": int(model.get("math_openclaw_timeout", 900)),
            "openclaw_max_attempts": int(model.get("openclaw_max_attempts", 3)),
            "openclaw_retry_base_seconds": int(model.get("openclaw_retry_base_seconds", 15)),
            "max_input_tokens": int(model.get("math_max_input_tokens", 500_000)),
            "max_cumulative_input_tokens": int(
                model.get(
                    "math_max_cumulative_input_tokens",
                    model.get("math_max_input_tokens", 500_000),
                )
            ),
            "max_cumulative_output_tokens": int(
                model.get(
                    "math_max_cumulative_output_tokens",
                    model.get("math_max_input_tokens", 500_000),
                )
            ),
            "require_actual_token_usage": bool(model.get("require_actual_token_usage", True)),
            "max_tool_calls": int(model.get("math_max_tool_calls", 40)),
            "max_agent_turns": int(model.get("math_max_agent_turns", 40)),
            "max_tool_wall_seconds": int(model.get("math_max_tool_wall_seconds", 120)),
            "verify_weave_agents": bool(model.get("verify_weave_agents", True)),
            "weave_agents_entity": str(model.get("weave_agents_entity", "llm-leaderboard")),
            "weave_agents_project": str(model.get("weave_agents_project", "tc-leaderboard")),
            "weave_agents_agent_name": str(
                model.get("weave_agents_agent_name", "nejumi-taiwan-openclaw")
            ),
            "weave_agents_limit": int(model.get("weave_agents_limit", 50)),
            "weave_agents_verification_timeout": int(
                model.get("weave_agents_verification_timeout", 120)
            ),
            "weave_agents_poll_seconds": int(model.get("weave_agents_poll_seconds", 5)),
            "dry_run": False,
            "run_openclaw": math_run_openclaw,
            "redo": bool(model.get("agentic_math_redo", False)),
            "results_dir": str(math_results_dir) if math_results_dir else None,
            "no_local": True,
            "weave_sidecar": False,
            "weave_sidecar_strict": False,
            "openclaw_tool_profile": "coding",
            "deny_tool": AGENTIC_DENIED_TOOLS,
            "deny_argument_pattern": AGENTIC_DENIED_ARGUMENT_PATTERNS,
        },
        "swebench_pro": {
            "subset": str(model.get("swe_subset", "leaderboard_compact_80")),
            "output_dir": str(output_root / "swebench_pro" / slug),
            "checkout_root": str(output_root / "swebench_pro_checkouts" / slug),
            "prefix": f"taiwan-swe-{slug}",
            "task_agent_prefix": f"tw-swe-{slug}",
            "openclaw_model": openclaw_model,
            **({"openclaw_model_params": swebench_pro_model_params} if swebench_pro_model_params else {}),
            **(
                {"openclaw_model_overrides": swebench_pro_model_overrides}
                if swebench_pro_model_overrides
                else {}
            ),
            "thinking": str(model.get("swe_thinking", "medium")),
            "openclaw_num_workers": int(model.get("swe_openclaw_num_workers", 8)),
            "openclaw_task_start_min_interval_seconds": float(
                model.get("swe_openclaw_task_start_min_interval_seconds", 15.0)
            ),
            "openclaw_timeout": int(model.get("swe_openclaw_timeout", 3600)),
            "openclaw_max_attempts": int(model.get("openclaw_max_attempts", 3)),
            "openclaw_retry_base_seconds": int(model.get("openclaw_retry_base_seconds", 15)),
            "dry_run": False,
            "run_openclaw": phase != "agentic_aggregate",
            "redo": bool(model.get("swebench_pro_redo", False)),
            "patch_path": str(output_root / "swebench_pro" / slug / "openclaw" / "patches.json")
            if phase == "agentic_aggregate"
            else None,
            "evaluate": True,
            "no_local": True,
            "weave_sidecar": False,
            "weave_sidecar_strict": False,
            "openclaw_tool_profile": "coding",
            "max_input_tokens": int(model.get("swe_max_input_tokens", 1_000_000)),
            "max_cumulative_input_tokens": int(
                model.get(
                    "swe_max_cumulative_input_tokens",
                    model.get("swe_max_input_tokens", 1_000_000),
                )
            ),
            "max_cumulative_output_tokens": int(
                model.get(
                    "swe_max_cumulative_output_tokens",
                    model.get("swe_max_input_tokens", 1_000_000),
                )
            ),
            "require_actual_token_usage": bool(model.get("require_actual_token_usage", True)),
            "max_tool_calls": int(model.get("swe_max_tool_calls", 40)),
            "max_agent_turns": int(model.get("swe_max_agent_turns", 40)),
            "max_tool_wall_seconds": int(model.get("swe_max_tool_wall_seconds", 300)),
            "verify_weave_agents": bool(model.get("verify_weave_agents", True)),
            "weave_agents_entity": str(model.get("weave_agents_entity", "llm-leaderboard")),
            "weave_agents_project": str(model.get("weave_agents_project", "tc-leaderboard")),
            "weave_agents_agent_name": str(
                model.get("weave_agents_agent_name", "nejumi-taiwan-openclaw")
            ),
            "weave_agents_limit": int(model.get("weave_agents_limit", 50)),
            "weave_agents_verification_timeout": int(
                model.get("weave_agents_verification_timeout", 120)
            ),
            "weave_agents_poll_seconds": int(model.get("weave_agents_poll_seconds", 5)),
            "deny_tool": AGENTIC_DENIED_TOOLS,
            "deny_argument_pattern": AGENTIC_SWE_DENIED_ARGUMENT_PATTERNS,
        },
        "deepswe": {
            "subset": str(model.get("deepswe_subset", "pilot_16")),
            "local_dataset_dir": str(model.get("deepswe_local_dataset_dir", "data/taiwan/deepswe")),
            "tasks_root": str(model.get("deepswe_tasks_root", "external/deep-swe/tasks")),
            "output_dir": str(output_root / "deepswe" / slug),
            "job_name": f"deepswe-{slug}",
            "prefix": f"taiwan-deepswe-{slug}",
            "task_agent_prefix": f"tw-deepswe-{slug}",
            "session_prefix": f"deepswe-{slug}",
            "openclaw_model": openclaw_model,
            **({"openclaw_model_params": deepswe_model_params} if deepswe_model_params else {}),
            **(
                {"openclaw_model_overrides": deepswe_model_overrides}
                if deepswe_model_overrides
                else {}
            ),
            "thinking": str(model.get("deepswe_thinking", model.get("swe_thinking", "high"))),
            "n_concurrent": int(model.get("deepswe_n_concurrent", 1)),
            "openclaw_timeout": int(model.get("deepswe_openclaw_timeout", 3600)),
            "openclaw_max_attempts": int(model.get("deepswe_openclaw_max_attempts", 2)),
            "openclaw_retry_base_seconds": int(model.get("openclaw_retry_base_seconds", 15)),
            "dry_run": False,
            "run_openclaw": phase != "agentic_aggregate",
            "results_dir": str(output_root / "deepswe" / slug / "runner")
            if phase == "agentic_aggregate"
            else None,
            "no_local": True,
            "use_task_agent": bool(model.get("deepswe_use_task_agent", True)),
            "restart_gateway_before_run": bool(
                model.get("deepswe_restart_gateway_before_run", True)
            ),
            "delete": bool(model.get("deepswe_delete_environment", True)),
            "disable_verification": bool(model.get("deepswe_disable_verification", False)),
            "quiet": bool(model.get("deepswe_quiet", False)),
            "openclaw_tool_profile": "coding",
            "max_input_tokens": int(model.get("deepswe_max_input_tokens", 1_000_000)),
            "max_cumulative_input_tokens": int(
                model.get(
                    "deepswe_max_cumulative_input_tokens",
                    model.get("deepswe_max_input_tokens", 1_000_000),
                )
            ),
            "max_cumulative_output_tokens": int(
                model.get("deepswe_max_cumulative_output_tokens", 500_000)
            ),
            "require_actual_token_usage": bool(model.get("require_actual_token_usage", True)),
            "max_tool_calls": int(model.get("deepswe_max_tool_calls", 40)),
            "max_agent_turns": int(model.get("deepswe_max_agent_turns", 40)),
            "max_tool_wall_seconds": int(model.get("deepswe_max_tool_wall_seconds", 300)),
            "verify_weave_agents": bool(model.get("verify_weave_agents", True)),
            "weave_agents_entity": str(model.get("weave_agents_entity", "llm-leaderboard")),
            "weave_agents_project": str(model.get("weave_agents_project", "tc-leaderboard")),
            "weave_agents_agent_name": str(
                model.get("weave_agents_agent_name", "nejumi-taiwan-openclaw")
            ),
            "weave_agents_limit": int(model.get("weave_agents_limit", 50)),
            "weave_agents_verification_timeout": int(
                model.get("weave_agents_verification_timeout", 120)
            ),
            "weave_agents_poll_seconds": int(model.get("weave_agents_poll_seconds", 5)),
            "deny_tool": AGENTIC_DENIED_TOOLS,
            "deny_argument_pattern": AGENTIC_SWE_DENIED_ARGUMENT_PATTERNS,
        },
        "agentic_swe_assorted": {
            "output_dir": str(output_root / "agentic_swe_assorted" / slug),
            "results_dir": str(output_root / "agentic_swe_assorted" / slug / "runner")
            if phase == "agentic_aggregate"
            else None,
            "run_openclaw": phase != "agentic_aggregate",
            "dry_run": False,
            "prefix": f"taiwan-swe-assorted-{slug}",
            "task_agent_prefix": f"tw-swe-assorted-{slug}",
            "session_prefix": f"agentic-swe-assorted-{slug}",
            "openclaw_model": openclaw_model,
            **(
                {"openclaw_model_params": agentic_swe_assorted_model_params}
                if agentic_swe_assorted_model_params
                else {}
            ),
            **(
                {"openclaw_model_overrides": agentic_swe_assorted_model_overrides}
                if agentic_swe_assorted_model_overrides
                else {}
            ),
            **(
                {
                    "high_openclaw_model_params": _optional_plain_dict(
                        model.get("agentic_swe_assorted_high_openclaw_model_params"),
                        field_name="agentic_swe_assorted_high_openclaw_model_params",
                    )
                }
                if model.get("agentic_swe_assorted_high_openclaw_model_params") is not None
                else {}
            ),
            **(
                {
                    "high_openclaw_model_overrides": _optional_plain_dict(
                        model.get("agentic_swe_assorted_high_openclaw_model_overrides"),
                        field_name="agentic_swe_assorted_high_openclaw_model_overrides",
                    )
                }
                if model.get("agentic_swe_assorted_high_openclaw_model_overrides") is not None
                else {}
            ),
            "thinking": str(
                model.get("agentic_swe_assorted_thinking", model.get("swe_thinking", "high"))
            ),
            "tier_weights": str(model.get("agentic_swe_assorted_tier_weights", "low=1,middle=1,high=1")),
            "low_middle_jsonl": str(
                model.get(
                    "agentic_swe_assorted_low_middle_jsonl",
                    "data/taiwan/swebench_lite_assorted/subsets/low_middle_v3_40.jsonl",
                )
            ),
            "low_middle_instance_ids_json": str(
                model.get(
                    "agentic_swe_assorted_low_middle_instance_ids_json",
                    "data/taiwan/swebench_lite_assorted/subsets/low_middle_v3_40_instance_ids.json",
                )
            ),
            "low_limit": int(model.get("agentic_swe_assorted_low_limit", 20)),
            "middle_limit": int(model.get("agentic_swe_assorted_middle_limit", 20)),
            "high_limit": int(model.get("agentic_swe_assorted_high_limit", 10)),
            "deepswe_metadata_jsonl": str(
                model.get(
                    "agentic_swe_assorted_deepswe_metadata_jsonl",
                    "data/taiwan/deepswe/subsets/essential_anchored_high_10_model_fidelity_cost_balanced.jsonl",
                )
            ),
            "deepswe_task_names_file": str(
                model.get(
                    "agentic_swe_assorted_deepswe_task_names_file",
                    "data/taiwan/deepswe/subsets/essential_anchored_high_10_model_fidelity_cost_balanced_task_names.json",
                )
            ),
            "deepswe_tasks_root": str(
                model.get("agentic_swe_assorted_deepswe_tasks_root", "external/deep-swe/tasks")
            ),
            "deepswe_public_trials_json": str(
                model.get(
                    "agentic_swe_assorted_deepswe_public_trials_json",
                    "outputs/deepswe_subset_analysis/deepswe_v1_1_trials.json",
                )
            ),
            "deepswe_public_model": model.get("agentic_swe_assorted_deepswe_public_model"),
            "deepswe_public_effort": model.get("agentic_swe_assorted_deepswe_public_effort"),
            "reuse_high_results_jsonl": list(
                model.get("agentic_swe_assorted_reuse_high_results_jsonl", [])
            ),
            "reuse_low_middle_patches_json": model.get(
                "agentic_swe_assorted_reuse_low_middle_patches_json"
            ),
            "deepswe_budget_preflight": str(
                model.get("agentic_swe_assorted_deepswe_budget_preflight", "error")
            ),
            "deepswe_preflight_hard_stat": str(
                model.get("agentic_swe_assorted_deepswe_preflight_hard_stat", "p90")
            ),
            "allow_deepswe_budget_mismatch": bool(
                model.get("agentic_swe_assorted_allow_deepswe_budget_mismatch", False)
            ),
            "official_swebench_repo": str(
                model.get("agentic_swe_assorted_official_swebench_repo", "external/SWE-bench")
            ),
            "checkout_root": str(output_root / "swebench_lite_checkouts" / slug),
            "skip_low_middle": bool(model.get("agentic_swe_assorted_skip_low_middle", False)),
            "skip_high": bool(model.get("agentic_swe_assorted_skip_high", False)),
            "no_docker_check": bool(model.get("agentic_swe_assorted_no_docker_check", False)),
            "swe_workers": int(model.get("agentic_swe_assorted_swe_workers", 4)),
            "high_workers": int(model.get("agentic_swe_assorted_high_workers", 2)),
            "eval_workers": int(model.get("agentic_swe_assorted_eval_workers", 4)),
            "swe_task_start_min_interval_seconds": float(
                model.get("agentic_swe_assorted_swe_task_start_min_interval_seconds", 5.0)
            ),
            "swe_openclaw_timeout": int(
                model.get("agentic_swe_assorted_swe_openclaw_timeout", 900)
            ),
            "high_openclaw_timeout": int(
                model.get("agentic_swe_assorted_high_openclaw_timeout", 1800)
            ),
            "eval_timeout": int(model.get("agentic_swe_assorted_eval_timeout", 1800)),
            "openclaw_max_attempts": int(
                model.get("agentic_swe_assorted_openclaw_max_attempts", 2)
            ),
            "openclaw_retry_base_seconds": int(model.get("openclaw_retry_base_seconds", 15)),
            "max_input_tokens": int(model.get("agentic_swe_assorted_max_input_tokens", 1_000_000)),
            "max_cumulative_input_tokens": int(
                model.get("agentic_swe_assorted_max_cumulative_input_tokens", 1_000_000)
            ),
            "high_max_cumulative_input_tokens": int(
                model.get("agentic_swe_assorted_high_max_cumulative_input_tokens", 14_000_000)
            ),
            "max_cumulative_output_tokens": int(
                model.get("agentic_swe_assorted_max_cumulative_output_tokens", 500_000)
            ),
            "require_actual_token_usage": bool(model.get("require_actual_token_usage", True)),
            "max_tool_calls": int(model.get("agentic_swe_assorted_max_tool_calls", 40)),
            "high_max_tool_calls": int(
                model.get("agentic_swe_assorted_high_max_tool_calls", 200)
            ),
            "max_agent_turns": int(model.get("agentic_swe_assorted_max_agent_turns", 40)),
            "high_max_agent_turns": int(
                model.get("agentic_swe_assorted_high_max_agent_turns", 150)
            ),
            "max_tool_wall_seconds": int(
                model.get("agentic_swe_assorted_max_tool_wall_seconds", 120)
            ),
            "llm_response_idle_timeout_seconds": int(
                model.get("agentic_swe_assorted_llm_response_idle_timeout_seconds", 900)
            ),
            "no_local": True,
            "use_task_agent": bool(model.get("agentic_swe_assorted_use_task_agent", True)),
            "openclaw_tool_profile": "coding",
            "verify_weave_agents": bool(model.get("verify_weave_agents", True)),
            "weave_agents_entity": str(model.get("weave_agents_entity", "llm-leaderboard")),
            "weave_agents_project": str(model.get("weave_agents_project", "tc-leaderboard")),
            "weave_agents_agent_name": str(
                model.get("weave_agents_agent_name", "nejumi-taiwan-openclaw")
            ),
            "weave_agents_limit": int(model.get("weave_agents_limit", 100)),
            "weave_agents_verification_timeout": int(
                model.get("weave_agents_verification_timeout", 120)
            ),
            "weave_agents_poll_seconds": int(model.get("weave_agents_poll_seconds", 5)),
            "swebench_namespace": str(
                model.get("agentic_swe_assorted_swebench_namespace", "swebench")
            ),
            "swebench_cache_level": str(
                model.get("agentic_swe_assorted_swebench_cache_level", "env")
            ),
            "deny_tool": AGENTIC_SWE_ASSORTED_DENIED_TOOLS,
            "deny_argument_pattern": AGENTIC_SWE_ASSORTED_DENIED_ARGUMENT_PATTERNS,
        },
        "bfcl": {
            "allow_overwrite": False,
            "run_generation": bfcl_run_generation,
            **(
                {"result_dir": str(bfcl_result_dir)}
                if bfcl_result_dir
                else {}
            ),
            "num_threads": int(model.get("bfcl_num_threads", 4)),
            "provider_min_request_interval_sec": float(
                model.get("bfcl_provider_min_request_interval_sec", 2.0)
            ),
            "provider_request_jitter_sec": float(
                model.get("bfcl_provider_request_jitter_sec", 0.5)
            ),
            "provider_rate_limit_key": str(
                model.get("bfcl_provider_rate_limit_key", f"bfcl:{slug}")
            ),
            "consecutive_failure_fail_fast": int(
                model.get("bfcl_consecutive_failure_fail_fast", 5)
            ),
        },
    }
    _apply_agentic_math_nemoclaw_config(override["agentic_math"], model)
    _apply_swebench_pro_nemoclaw_config(override["swebench_pro"], model)
    _apply_deepswe_nemoclaw_config(override["deepswe"], model)
    _apply_agentic_swe_assorted_nemoclaw_config(
        override["agentic_swe_assorted"], model
    )
    if reasoning_effort:
        override.setdefault("generator", {}).setdefault("extra_body", {}).setdefault(
            "reasoning", {}
        )["effort"] = str(reasoning_effort)
    return override


def _is_final_only(model: dict[str, Any]) -> bool:
    return bool(model.get("final_only"))


def _is_suspended(model: dict[str, Any]) -> bool:
    return bool(model.get("suspended"))


def select_models(
    models: list[dict[str, Any]],
    selected: list[str] | None,
    *,
    canary: bool = False,
    include_final_only: bool = False,
    include_suspended: bool = False,
) -> list[dict[str, Any]]:
    if canary and selected:
        raise SystemExit("--canary cannot be combined with --model")
    if canary:
        chosen = [model for model in models if bool(model.get("canary"))]
        if len(chosen) != 1:
            raise SystemExit(
                f"Expected exactly one canary model in the manifest; found {len(chosen)}."
            )
        if _is_suspended(chosen[0]):
            raise SystemExit(
                f"Canary model {chosen[0].get('slug')} is suspended; choose an active canary."
            )
        return chosen
    if include_suspended and not selected:
        raise SystemExit("--include-suspended requires at least one explicit --model slug")
    if not selected:
        return [
            model
            for model in models
            if (include_final_only or not _is_final_only(model))
            and not _is_suspended(model)
        ]
    wanted = set(selected)
    chosen = [model for model in models if str(model.get("slug")) in wanted]
    missing = wanted - {str(model.get("slug")) for model in chosen}
    if missing:
        raise SystemExit(f"Unknown model slug(s): {', '.join(sorted(missing))}")
    final_only = [
        str(model.get("slug"))
        for model in chosen
        if _is_final_only(model)
    ]
    if final_only and not include_final_only:
        raise SystemExit(
            "Final-only model(s) require --include-final-only: "
            + ", ".join(sorted(final_only))
        )
    suspended = [
        str(model.get("slug"))
        for model in chosen
        if _is_suspended(model)
    ]
    if suspended and not include_suspended:
        raise SystemExit(
            "Suspended model(s) require --include-suspended: "
            + ", ".join(sorted(suspended))
        )
    return chosen


def generate_configs(args: argparse.Namespace) -> list[Path]:
    phase = getattr(args, "phase", "full")
    if phase not in PHASE_CHOICES:
        raise SystemExit(f"Unsupported phase {phase!r}; choose from {', '.join(PHASE_CHOICES)}")
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    manifest = OmegaConf.load(args.manifest)
    models = select_models(
        _plain(manifest).get("models", []),
        args.model,
        canary=bool(getattr(args, "canary", False)),
        include_final_only=bool(getattr(args, "include_final_only", False)),
        include_suspended=bool(getattr(args, "include_suspended", False)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for model in models:
        model = _with_cli_nemoclaw_overrides(model, args)
        source_path = CONFIG_DIR / str(model["source_config"])
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        source_cfg = OmegaConf.load(source_path)
        override = OmegaConf.create(
            build_override(
                model,
                args.output_root,
                phase=phase,
            )
        )
        generated = OmegaConf.merge(source_cfg, override)
        output_path = output_dir / f"config-taiwan-full-{model['slug']}.yaml"
        OmegaConf.save(config=generated, f=output_path)
        output_paths.append(output_path)
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs") / "taiwan_full_eval",
        help="Root for benchmark output directories inside generated configs.",
    )
    parser.add_argument("--model", action="append", help="Model slug to generate. Repeatable.")
    parser.add_argument(
        "--canary",
        action="store_true",
        help="Generate the single manifest model marked canary=true for one-model full review.",
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
    parser.add_argument(
        "--phase",
        choices=PHASE_CHOICES,
        default="full",
        help=(
            "full runs every Taiwan benchmark; nonagentic skips Agentic Math, "
            "SWE-Bench Pro, and aggregate; agentic_aggregate reuses completed "
            "agentic artifacts and then logs aggregate_taiwan."
        ),
    )
    parser.add_argument(
        "--agentic-math-nemoclaw-sandbox",
        help=(
            "Opt selected Agentic Math configs into NeMoClaw sandbox execution. "
            "Use --swebench-pro-nemoclaw-sandbox separately for SWE-Bench Pro."
        ),
    )
    parser.add_argument(
        "--agentic-math-nemoclaw-bin",
        default="nemoclaw",
        help="NeMoClaw executable to use when --agentic-math-nemoclaw-sandbox is set.",
    )
    parser.add_argument(
        "--agentic-math-nemoclaw-workdir",
        default="/sandbox",
        help="Working directory inside the NeMoClaw sandbox for Agentic Math.",
    )
    parser.add_argument(
        "--agentic-math-nemoclaw-openclaw-config-path",
        help=(
            "Sandbox OpenClaw config path to read as the Agentic Math per-task "
            "template. Defaults to the runner's NeMoClaw config path."
        ),
    )
    parser.add_argument(
        "--swebench-pro-nemoclaw-sandbox",
        help=(
            "Opt selected SWE-Bench Pro configs into NeMoClaw sandbox execution. "
            "The SWE runner will generate a sandbox-visible task OpenClaw config "
            "inside each checkout and exclude it from patch capture."
        ),
    )
    parser.add_argument(
        "--swebench-pro-nemoclaw-bin",
        default="nemoclaw",
        help="NeMoClaw executable to use when --swebench-pro-nemoclaw-sandbox is set.",
    )
    parser.add_argument(
        "--swebench-pro-nemoclaw-checkout-sandbox-root",
        help=(
            "Sandbox-visible root corresponding to the generated SWE-Bench Pro "
            "checkout_root. Omit when host checkout paths are mounted unchanged."
        ),
    )
    parser.add_argument(
        "--swebench-pro-nemoclaw-workdir",
        help=(
            "Working directory inside the NeMoClaw sandbox for SWE-Bench Pro. "
            "Omit to use the sandbox-visible per-instance checkout path."
        ),
    )
    parser.add_argument(
        "--swebench-pro-nemoclaw-checkout-transfer-mode",
        choices=["visible", "copy"],
        help=(
            "SWE-Bench Pro NeMoClaw checkout access mode. 'visible' requires "
            "a mounted checkout path; 'copy' uploads each checkout to the sandbox."
        ),
    )
    parser.add_argument(
        "--swebench-pro-nemoclaw-openclaw-config-path",
        help=(
            "Sandbox OpenClaw config path to read as the SWE-Bench Pro per-task "
            "template. Defaults to the runner's NeMoClaw config path."
        ),
    )
    parser.add_argument(
        "--deepswe-nemoclaw-sandbox",
        help=(
            "Opt selected DeepSWE configs into NeMoClaw sandbox execution. "
            "DeepSWE runs through Pier and uses the same native OpenClaw/Weave path."
        ),
    )
    parser.add_argument(
        "--deepswe-nemoclaw-bin",
        default="nemoclaw",
        help="NeMoClaw executable to use when --deepswe-nemoclaw-sandbox is set.",
    )
    parser.add_argument(
        "--deepswe-nemoclaw-workdir",
        default="/sandbox",
        help="Working directory inside the NeMoClaw sandbox for DeepSWE.",
    )
    parser.add_argument(
        "--deepswe-nemoclaw-openclaw-config-path",
        help=(
            "Sandbox OpenClaw config path to read as the DeepSWE per-task "
            "template. Defaults to the runner's NeMoClaw config path."
        ),
    )
    parser.add_argument(
        "--agentic-swe-assorted-nemoclaw-sandbox",
        help=(
            "Opt selected Agentic SWE-Assorted configs into NeMoClaw sandbox "
            "execution."
        ),
    )
    parser.add_argument(
        "--agentic-swe-assorted-nemoclaw-bin",
        default="nemoclaw",
        help="NeMoClaw executable to use for Agentic SWE-Assorted.",
    )
    parser.add_argument(
        "--agentic-swe-assorted-nemoclaw-workdir",
        default="/sandbox",
        help="Working directory inside the NeMoClaw sandbox for Agentic SWE-Assorted.",
    )
    parser.add_argument(
        "--agentic-swe-assorted-nemoclaw-checkout-transfer-mode",
        choices=["visible", "copy"],
        help="Agentic SWE-Assorted NeMoClaw checkout access mode.",
    )
    parser.add_argument(
        "--agentic-swe-assorted-nemoclaw-openclaw-config-path",
        help=(
            "Sandbox OpenClaw config path to read as the Agentic SWE-Assorted "
            "per-task template. Defaults to the runner's NeMoClaw config path."
        ),
    )
    return parser.parse_args()


def main() -> None:
    paths = generate_configs(parse_args())
    for path in paths:
        try:
            display_path = path.relative_to(CONFIG_DIR)
        except ValueError:
            display_path = path
        print(display_path)


if __name__ == "__main__":
    main()
