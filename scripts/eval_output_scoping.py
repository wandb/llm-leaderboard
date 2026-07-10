from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _section(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    value = cfg.setdefault(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"Config section {key!r} must be a mapping")
    return value


def _safe_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return safe or "run"


def _scoped_root(output_cfg: dict[str, Any], run_id: str) -> Path:
    template = str(
        output_cfg.get("root")
        or output_cfg.get("run_root")
        or "outputs/taiwan_full_eval_runs"
    )
    safe_run_id = _safe_component(run_id)
    if "{run_id}" in template or "{wandb_run_id}" in template:
        return Path(template.format(run_id=safe_run_id, wandb_run_id=safe_run_id))
    return Path(template) / safe_run_id


def should_apply_run_scoped_outputs(cfg: dict[str, Any]) -> bool:
    output_cfg = cfg.get("output") if isinstance(cfg.get("output"), dict) else {}
    if "run_scoped" in output_cfg:
        return _as_bool(output_cfg.get("run_scoped"))

    # Taiwan full runs should be safe to rerun with the exact same command.
    # Japanese/other configs are unchanged unless they opt in explicitly.
    run_cfg = cfg.get("run") if isinstance(cfg.get("run"), dict) else {}
    wandb_cfg = cfg.get("wandb") if isinstance(cfg.get("wandb"), dict) else {}
    return (
        _as_bool(run_cfg.get("aggregate_taiwan"))
        and str(wandb_cfg.get("project") or "") == "tc-leaderboard"
    )


def apply_run_scoped_outputs(
    cfg: dict[str, Any],
    *,
    run_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a config whose local outputs are scoped to the current W&B run.

    This keeps the human-facing command stable: rerunning the same config creates
    a fresh W&B run and fresh local outputs without requiring a new config file.
    """

    resolved = deepcopy(cfg)
    if not run_id or not should_apply_run_scoped_outputs(resolved):
        return resolved, {"applied": False, "run_id": run_id or "", "run_root": ""}

    run_cfg = resolved.get("run") if isinstance(resolved.get("run"), dict) else {}
    output_cfg = _section(resolved, "output")
    root = _scoped_root(output_cfg, run_id)
    output_cfg["run_scoped"] = True
    output_cfg["resolved_run_id"] = run_id
    output_cfg["resolved_run_root"] = str(root)

    if _as_bool(run_cfg.get("agentic_math")):
        agentic_math = _section(resolved, "agentic_math")
        agentic_math["output_dir"] = str(root / "agentic_math")
        if _as_bool(agentic_math.get("run_openclaw", True)):
            agentic_math["results_dir"] = None

    if _as_bool(run_cfg.get("swebench_pro")):
        swebench_pro = _section(resolved, "swebench_pro")
        swebench_pro["output_dir"] = str(root / "swebench_pro")
        swebench_pro["checkout_root"] = str(root / "swebench_pro_checkouts")
        if _as_bool(swebench_pro.get("run_openclaw", True)):
            swebench_pro["patch_path"] = None

    if _as_bool(run_cfg.get("deepswe")):
        deepswe = _section(resolved, "deepswe")
        deepswe["output_dir"] = str(root / "deepswe")
        if _as_bool(deepswe.get("run_openclaw", True)):
            deepswe["results_dir"] = None

    if _as_bool(run_cfg.get("bfcl")):
        bfcl = _section(resolved, "bfcl")
        bfcl["result_dir"] = str(root / "bfcl" / "result")
        bfcl["score_dir"] = str(root / "bfcl" / "score")
        bfcl["allow_overwrite"] = True

    return resolved, {"applied": True, "run_id": run_id, "run_root": str(root)}
