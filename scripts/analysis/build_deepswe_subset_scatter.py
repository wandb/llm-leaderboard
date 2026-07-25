#!/usr/bin/env python3
"""Build scatter plots for a DeepSWE task subset against full DeepSWE scores."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


REPO_ROOT = Path(__file__).resolve().parents[2]
ESSENTIAL_SELECTOR_PATH = REPO_ROOT / "scripts" / "analysis" / "select_deepswe_essential_subset.py"
_spec = importlib.util.spec_from_file_location("deepswe_essential_selector", ESSENTIAL_SELECTOR_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Failed to load {ESSENTIAL_SELECTOR_PATH}")
_essential = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_essential)

DEFAULT_TASKS_ROOT = _essential.DEFAULT_TASKS_ROOT
DEFAULT_ANALYSIS_DIR = _essential.DEFAULT_ANALYSIS_DIR
PUBLIC_TASKS_URL = _essential.PUBLIC_TASKS_URL
PUBLIC_TRIALS_URL = _essential.PUBLIC_TRIALS_URL
EssentialSelector = _essential.EssentialSelector
_fetch_or_read = _essential._fetch_or_read
_read_json = _essential._read_json


DEFAULT_SUBSET_NAME = "essential_anchored_high_8_wandb_glm52_cap100_10m_lang_balanced"
DEFAULT_SUBSET_TASKS = (
    REPO_ROOT
    / "data"
    / "taiwan"
    / "deepswe"
    / "subsets"
    / f"{DEFAULT_SUBSET_NAME}_task_names.json"
)


def _metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "n": float(len(x)),
        "mae": float(mean_absolute_error(y, x)),
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
        "kendall": float(kendalltau(x, y).statistic),
        "r2": float(r2_score(y, x)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _plot(path: Path, rows: list[dict[str, Any]], *, title: str, subtitle: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.array([float(row["full_score"]) for row in rows])
    y = np.array([float(row["subset_score"]) for row in rows])
    metrics = _metrics(y, x)

    fig, ax = plt.subplots(figsize=(8.8, 6.6), dpi=180)
    ax.scatter(
        x,
        y,
        s=58,
        color="#2563eb",
        edgecolor="#0f172a",
        linewidth=0.55,
        alpha=0.86,
    )

    lo = max(0.0, min(float(x.min()), float(y.min())) - 0.04)
    hi = min(1.0, max(float(x.max()), float(y.max())) + 0.04)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#64748b", linewidth=1.0, label="y = x")
    if len(rows) >= 2:
        coef = np.polyfit(x, y, 1)
        xx = np.linspace(lo, hi, 120)
        ax.plot(xx, coef[0] * xx + coef[1], color="#ef4444", linewidth=1.3, alpha=0.80)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Full DeepSWE public score")
    ax.set_ylabel("Subset score")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.55)
    fig.suptitle(title, x=0.08, y=0.98, ha="left", fontsize=13, weight="bold")
    ax.set_title(
        (
            f"{subtitle}\n"
            f"n={int(metrics['n'])}, Pearson={metrics['pearson']:.3f}, "
            f"Spearman={metrics['spearman']:.3f}, MAE={metrics['mae']:.3f}"
        ),
        loc="left",
        fontsize=9,
        color="#334155",
        pad=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path)
    plt.close(fig)


def build_in_sample_rows(selector: EssentialSelector, task_names: list[str]) -> list[dict[str, Any]]:
    name_to_idx = {name: idx for idx, name in enumerate(selector.task_names)}
    selected = [name_to_idx[name] for name in task_names]
    subset_scores = selector.X[:, selected].mean(axis=1)
    rows: list[dict[str, Any]] = []
    meta = selector.meta.reset_index()
    for idx, subset_score in enumerate(subset_scores):
        rows.append(
            {
                "effort_key": str(meta.iloc[idx]["effort_key"]),
                "model": str(meta.iloc[idx]["model"]),
                "reasoning_effort": str(meta.iloc[idx]["reasoning_effort"]),
                "provider": str(meta.iloc[idx]["provider"]),
                "full_score": float(selector.y[idx]),
                "subset_score": float(subset_score),
            }
        )
    return rows


def build_leave_family_out_rows(selector: EssentialSelector, task_names: list[str]) -> list[dict[str, Any]]:
    name_to_idx = {name: idx for idx, name in enumerate(selector.task_names)}
    base_selected = [name_to_idx[name] for name in task_names]
    meta = selector.meta.reset_index()
    rows: list[dict[str, Any]] = []

    for family in sorted(set(selector.families)):
        test_idx = np.where(selector.families == family)[0]
        train_idx = np.where(selector.families != family)[0]
        if len(train_idx) < 5 or len(test_idx) < 1:
            continue
        selected = selector.greedy_select(len(base_selected), train_idx)
        subset_scores = selector.X[np.ix_(test_idx, selected)].mean(axis=1)
        selected_names = [selector.task_names[idx] for idx in selected]
        for local_offset, row_idx in enumerate(test_idx):
            rows.append(
                {
                    "held_out_family": str(family),
                    "effort_key": str(meta.iloc[row_idx]["effort_key"]),
                    "model": str(meta.iloc[row_idx]["model"]),
                    "reasoning_effort": str(meta.iloc[row_idx]["reasoning_effort"]),
                    "provider": str(meta.iloc[row_idx]["provider"]),
                    "full_score": float(selector.y[row_idx]),
                    "subset_score": float(subset_scores[local_offset]),
                    "selected_task_names": "|".join(selected_names),
                }
            )
    return rows


def build_constrained_leave_family_out_rows(
    selector: EssentialSelector,
    summary_path: Path,
) -> list[dict[str, Any]]:
    summary = _read_json(summary_path)
    family_to_selected = {
        str(row["family"]): [str(task_name) for task_name in row["selected"]]
        for row in summary.get("leave_family_out", {}).get("rows", [])
    }
    if not family_to_selected:
        raise ValueError(f"no leave_family_out rows in {summary_path}")
    name_to_idx = {name: idx for idx, name in enumerate(selector.task_names)}
    meta = selector.meta.reset_index()
    rows: list[dict[str, Any]] = []
    for family, selected_names in family_to_selected.items():
        test_idx = np.where(selector.families == family)[0]
        if len(test_idx) < 1:
            continue
        selected = [name_to_idx[name] for name in selected_names if name in name_to_idx]
        if not selected:
            continue
        subset_scores = selector.X[np.ix_(test_idx, selected)].mean(axis=1)
        for local_offset, row_idx in enumerate(test_idx):
            rows.append(
                {
                    "held_out_family": str(family),
                    "effort_key": str(meta.iloc[row_idx]["effort_key"]),
                    "model": str(meta.iloc[row_idx]["model"]),
                    "reasoning_effort": str(meta.iloc[row_idx]["reasoning_effort"]),
                    "provider": str(meta.iloc[row_idx]["provider"]),
                    "full_score": float(selector.y[row_idx]),
                    "subset_score": float(subset_scores[local_offset]),
                    "selected_task_names": "|".join(selected_names),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-name", default=DEFAULT_SUBSET_NAME)
    parser.add_argument("--task-names-json", type=Path, default=DEFAULT_SUBSET_TASKS)
    parser.add_argument("--tasks-root", type=Path, default=DEFAULT_TASKS_ROOT)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument(
        "--selection-summary-json",
        type=Path,
        help=(
            "Selection summary with constrained leave_family_out rows. Defaults to "
            "<analysis-dir>/<subset-name>_selection_summary.json when present."
        ),
    )
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    task_names = [str(item) for item in _read_json(args.task_names_json)]
    selector = EssentialSelector(
        trials_payload=_fetch_or_read(
            PUBLIC_TRIALS_URL,
            analysis_dir / "deepswe_v1_1_trials.json",
            refresh=args.refresh,
        ),
        tasks_payload=_fetch_or_read(
            PUBLIC_TASKS_URL,
            analysis_dir / "deepswe_v1_1_tasks.json",
            refresh=args.refresh,
        ),
        tasks_root=args.tasks_root.resolve(),
    )

    in_sample_rows = build_in_sample_rows(selector, task_names)
    summary_path = (
        args.selection_summary_json
        if args.selection_summary_json is not None
        else analysis_dir / f"{args.subset_name}_selection_summary.json"
    )
    if summary_path.exists():
        oof_rows = build_constrained_leave_family_out_rows(selector, summary_path)
        oof_label = "leave-one-model-family-out constrained reselection"
    else:
        oof_rows = build_leave_family_out_rows(selector, task_names)
        oof_label = "leave-one-model-family-out unconstrained reselection"
    prefix = analysis_dir / args.subset_name

    _write_csv(prefix.with_name(prefix.name + "_scatter_in_sample.csv"), in_sample_rows)
    _write_csv(prefix.with_name(prefix.name + "_scatter_leave_family_out.csv"), oof_rows)
    _plot(
        prefix.with_name(prefix.name + "_scatter_in_sample.png"),
        in_sample_rows,
        title="DeepSWE subset vs full score",
        subtitle=f"{args.subset_name} / in-sample fixed subset",
    )
    _plot(
        prefix.with_name(prefix.name + "_scatter_leave_family_out.png"),
        oof_rows,
        title="DeepSWE subset vs full score",
        subtitle=f"{args.subset_name} / {oof_label}",
    )
    payload = {
        "subset_name": args.subset_name,
        "task_names": task_names,
        "in_sample": _metrics(
            np.array([row["subset_score"] for row in in_sample_rows], dtype=float),
            np.array([row["full_score"] for row in in_sample_rows], dtype=float),
        ),
        "leave_family_out": _metrics(
            np.array([row["subset_score"] for row in oof_rows], dtype=float),
            np.array([row["full_score"] for row in oof_rows], dtype=float),
        ),
        "files": {
            "in_sample_png": str(prefix.with_name(prefix.name + "_scatter_in_sample.png")),
            "leave_family_out_png": str(
                prefix.with_name(prefix.name + "_scatter_leave_family_out.png")
            ),
            "in_sample_csv": str(prefix.with_name(prefix.name + "_scatter_in_sample.csv")),
            "leave_family_out_csv": str(
                prefix.with_name(prefix.name + "_scatter_leave_family_out.csv")
            ),
            "selection_summary_json": str(summary_path) if summary_path.exists() else None,
        },
    }
    (prefix.with_name(prefix.name + "_scatter_summary.json")).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
