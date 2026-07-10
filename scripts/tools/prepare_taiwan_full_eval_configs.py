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


RUN_FLAGS: dict[str, bool] = {
    "agentic_math": True,
    "bfcl": True,
    "swebench": False,
    "swebench_pro": True,
    "deepswe": False,
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
MANIFEST_JUDGE_OVERRIDE_KEYS = (
    "judge_model",
    "judge_parallel",
    "judge_params",
)


def _plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


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
        flags["aggregate_taiwan"] = False
        return flags
    if phase == "agentic":
        return {key: key in {"agentic_math", "swebench_pro", "deepswe"} for key in flags}
    if phase == "agentic_aggregate":
        return {
            key: key in {"agentic_math", "swebench_pro", "deepswe", "aggregate_taiwan"}
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


def build_override(
    model: dict[str, Any],
    output_root: Path,
    phase: str = "full",
) -> dict[str, Any]:
    _reject_manifest_judge_overrides(model)
    slug = str(model["slug"])
    openclaw_model = str(model["openclaw_model"])
    reasoning_effort = model.get("reasoning_effort")
    math_limit = model.get("math_limit", 50)
    if math_limit is None:
        math_limit = 50

    override: dict[str, Any] = {
        "testmode": False,
        "wandb": {"run_name": str(model["run_name"])},
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
            "run_openclaw": phase != "agentic_aggregate",
            "redo": bool(model.get("agentic_math_redo", False)),
            "results_dir": str(output_root / "agentic_math" / slug / "openclaw")
            if phase == "agentic_aggregate"
            else None,
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
            "deny_argument_pattern": AGENTIC_DENIED_ARGUMENT_PATTERNS,
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
            "thinking": str(model.get("deepswe_thinking", model.get("swe_thinking", "high"))),
            "n_concurrent": int(model.get("deepswe_n_concurrent", 1)),
            "openclaw_timeout": int(model.get("deepswe_openclaw_timeout", 3600)),
            "openclaw_max_attempts": int(model.get("deepswe_openclaw_max_attempts", 1)),
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
            "deny_argument_pattern": AGENTIC_DENIED_ARGUMENT_PATTERNS,
        },
        "bfcl": {
            "allow_overwrite": False,
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
    if reasoning_effort:
        override.setdefault("generator", {}).setdefault("extra_body", {}).setdefault(
            "reasoning", {}
        )["effort"] = str(reasoning_effort)
    return override


def _is_final_only(model: dict[str, Any]) -> bool:
    return bool(model.get("final_only"))


def select_models(
    models: list[dict[str, Any]],
    selected: list[str] | None,
    *,
    canary: bool = False,
    include_final_only: bool = False,
) -> list[dict[str, Any]]:
    if canary and selected:
        raise SystemExit("--canary cannot be combined with --model")
    if canary:
        chosen = [model for model in models if bool(model.get("canary"))]
        if len(chosen) != 1:
            raise SystemExit(
                f"Expected exactly one canary model in the manifest; found {len(chosen)}."
            )
        return chosen
    if not selected:
        return [
            model
            for model in models
            if include_final_only or not _is_final_only(model)
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
