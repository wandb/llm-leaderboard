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


RUN_FLAGS: dict[str, bool] = {
    "agentic_math": True,
    "bfcl": True,
    "swebench": False,
    "swebench_pro": True,
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

AGENTIC_DENIED_TOOLS = ["code_execution", "web_search", "web_fetch", "browser", "browser_*", "*search*"]
AGENTIC_DENIED_ARGUMENT_PATTERNS = [
    r"https?://",
    r"\b(curl|wget)\b",
    r"\b(requests|urllib|httpx)\.",
]


def _plain(value: Any) -> Any:
    return OmegaConf.to_container(value, resolve=True)


def _run_flags_for_phase(phase: str) -> dict[str, bool]:
    flags = dict(RUN_FLAGS)
    if phase == "full":
        return flags
    if phase == "nonagentic":
        flags["agentic_math"] = False
        flags["swebench_pro"] = False
        flags["aggregate_taiwan"] = False
        return flags
    if phase == "agentic":
        return {key: key in {"agentic_math", "swebench_pro"} for key in flags}
    if phase == "agentic_aggregate":
        return {key: key in {"agentic_math", "swebench_pro", "aggregate_taiwan"} for key in flags}
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
    math_config_path = getattr(args, "agentic_math_nemoclaw_openclaw_config_path", None)
    if sandbox and math_config_path:
        updated["agentic_math_nemoclaw_openclaw_config_path"] = str(math_config_path)
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
    config_path = model.get("agentic_math_nemoclaw_openclaw_config_path")
    if config_path:
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
    config_path = model.get("swebench_pro_nemoclaw_openclaw_config_path")
    if config_path:
        swebench_pro["nemoclaw_openclaw_config_path"] = str(config_path)


def build_override(model: dict[str, Any], output_root: Path, phase: str = "full") -> dict[str, Any]:
    slug = str(model["slug"])
    openclaw_model = str(model["openclaw_model"])
    reasoning_effort = model.get("reasoning_effort")

    override: dict[str, Any] = {
        "testmode": False,
        "wandb": {"run_name": str(model["run_name"])},
        "run": _run_flags_for_phase(phase),
        "agentic_math": {
            "subset": "leaderboard",
            "output_dir": str(output_root / "agentic_math" / slug),
            "prefix": f"taiwan-math-{slug}",
            "task_agent_prefix": f"tw-math-{slug}",
            "openclaw_model": openclaw_model,
            "thinking": str(model.get("agentic_thinking", "high")),
            "openclaw_max_attempts": int(model.get("openclaw_max_attempts", 3)),
            "openclaw_retry_base_seconds": int(model.get("openclaw_retry_base_seconds", 15)),
            "dry_run": False,
            "run_openclaw": phase != "agentic_aggregate",
            "results_dir": str(output_root / "agentic_math" / slug / "openclaw")
            if phase == "agentic_aggregate"
            else None,
            "weave_sidecar": False,
            "weave_sidecar_strict": False,
            "openclaw_tool_profile": "coding",
            "deny_tool": AGENTIC_DENIED_TOOLS,
            "deny_argument_pattern": AGENTIC_DENIED_ARGUMENT_PATTERNS,
        },
        "swebench_pro": {
            "subset": "leaderboard_compact_80",
            "output_dir": str(output_root / "swebench_pro" / slug),
            "checkout_root": str(output_root / "swebench_pro_checkouts" / slug),
            "prefix": f"taiwan-swe-{slug}",
            "task_agent_prefix": f"tw-swe-{slug}",
            "openclaw_model": openclaw_model,
            "thinking": str(model.get("swe_thinking", "medium")),
            "openclaw_max_attempts": int(model.get("openclaw_max_attempts", 3)),
            "openclaw_retry_base_seconds": int(model.get("openclaw_retry_base_seconds", 15)),
            "dry_run": False,
            "run_openclaw": phase != "agentic_aggregate",
            "patch_path": str(output_root / "swebench_pro" / slug / "openclaw" / "patches.json")
            if phase == "agentic_aggregate"
            else None,
            "evaluate": True,
            "weave_sidecar": False,
            "weave_sidecar_strict": False,
            "openclaw_tool_profile": "coding",
            "max_input_tokens": int(model.get("swe_max_input_tokens", 1_000_000)),
            "max_tool_calls": int(model.get("swe_max_tool_calls", 60)),
            "deny_tool": AGENTIC_DENIED_TOOLS,
            "deny_argument_pattern": AGENTIC_DENIED_ARGUMENT_PATTERNS,
        },
        "bfcl": {
            "allow_overwrite": False,
        },
    }
    _apply_agentic_math_nemoclaw_config(override["agentic_math"], model)
    _apply_swebench_pro_nemoclaw_config(override["swebench_pro"], model)
    judge_model = model.get("judge_model")
    if judge_model:
        for task_name in ("mtbench", "hle", "hallulens_zh_tw"):
            override.setdefault(task_name, {}).setdefault("judge", {})["model"] = str(
                judge_model
            )
            override[task_name]["judge"]["parallel"] = int(model.get("judge_parallel", 8))
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
        override = OmegaConf.create(build_override(model, args.output_root, phase=phase))
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
