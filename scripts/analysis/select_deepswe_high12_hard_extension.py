#!/usr/bin/env python3
"""Select a hard, budget-aware four-task extension to the frozen High-8.

This review selector keeps the active High-8 unchanged. It adds one Go, two
Python, and one TypeScript task while explicitly constraining public difficulty
and top-end headroom. The output is review-only and does not activate a subset.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_SELECTOR_PATH = REPO_ROOT / "scripts" / "analysis" / "select_deepswe_high_extension.py"
_SPEC = importlib.util.spec_from_file_location("deepswe_high_extension", BASE_SELECTOR_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load {BASE_SELECTOR_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

DEFAULT_OUTPUT_PREFIX = (
    REPO_ROOT
    / "outputs"
    / "deepswe_subset_analysis"
    / "essential_anchored_high_12_hard_extension_review_20260723"
)
DEFAULT_LANGUAGE_COUNTS = {"go": 1, "python": 2, "typescript": 1}
PUBLIC_PROFILE_KEYS = (
    "mini_swe_agent_gpt_5_6_luna_high",
    "mini_swe_agent_gpt_5_6_luna_max",
    "mini_swe_agent_gpt_5_6_sol_max",
    "mini_swe_agent_claude_sonnet_4_6_high",
    "mini_swe_agent_claude_fable_5_max",
    "mini_swe_agent_glm_5_2_max",
)


def _candidate_extensions(
    candidates: dict[str, list[str]],
    language_counts: dict[str, int],
) -> list[list[str]]:
    per_language = [
        list(itertools.combinations(candidates[language], count))
        for language, count in language_counts.items()
    ]
    return [
        [task_name for group in combination for task_name in group]
        for combination in itertools.product(*per_language)
    ]


def _evaluate(
    selector: Any,
    *,
    anchor_indices: list[int],
    extension_indices: list[int],
    rows: np.ndarray,
) -> dict[str, float]:
    selected_indices = anchor_indices + extension_indices
    predicted = selector.X[np.ix_(rows, selected_indices)].mean(axis=1)
    target = selector.y[rows]
    metrics = _BASE._metrics(predicted, target)
    top_cutoff = float(np.quantile(target, 0.75))
    top_rows = rows[target >= top_cutoff]
    extension_scores = selector.X[np.ix_(rows, extension_indices)].mean(axis=1)
    metrics.update(
        {
            "extension_matrix_pass_rate": float(extension_scores.mean()),
            "top_quartile_extension_pass_rate": float(
                selector.X[np.ix_(top_rows, extension_indices)].mean()
            ),
            "maximum_subset_score": float(predicted.max()),
        }
    )
    return metrics


def _objective(metrics: dict[str, float], average_cost_usd: float) -> float:
    return (
        0.50 * metrics["spearman"]
        + 0.35 * metrics["pearson"]
        - 0.10 * metrics["mae"]
        - 0.02 * average_cost_usd
    )


def _rank_extensions(
    selector: Any,
    *,
    anchors: list[str],
    extensions: list[list[str]],
    rows: np.ndarray,
    max_extension_task_pass_rate: float,
    max_top_quartile_extension_pass_rate: float,
    max_subset_score: float,
) -> list[dict[str, Any]]:
    name_to_index = {name: index for index, name in enumerate(selector.task_names)}
    anchor_indices = [name_to_index[name] for name in anchors]
    ranked = []
    for extension in extensions:
        repositories = list(selector.stats.loc[extension, "repository"].astype(str))
        if len(set(repositories)) != len(repositories):
            continue
        extension_task_pass_rate = float(
            selector.stats.loc[extension, "task_pass_rate"].mean()
        )
        if extension_task_pass_rate > max_extension_task_pass_rate:
            continue
        extension_indices = [name_to_index[name] for name in extension]
        metrics = _evaluate(
            selector,
            anchor_indices=anchor_indices,
            extension_indices=extension_indices,
            rows=rows,
        )
        if (
            metrics["top_quartile_extension_pass_rate"]
            > max_top_quartile_extension_pass_rate
            or metrics["maximum_subset_score"] > max_subset_score
        ):
            continue
        average_cost = float(selector.stats.loc[extension, "avg_cost_usd"].mean())
        ranked.append(
            {
                "objective": _objective(metrics, average_cost),
                "task_names": extension,
                "repositories": repositories,
                "extension_task_pass_rate": extension_task_pass_rate,
                "average_public_cost_usd": average_cost,
                "metrics": metrics,
            }
        )
    ranked.sort(key=lambda row: row["objective"], reverse=True)
    return ranked


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors-json", type=Path, default=_BASE.DEFAULT_ANCHORS)
    parser.add_argument("--tasks-root", type=Path, default=_BASE.DEFAULT_TASKS_ROOT)
    parser.add_argument("--analysis-dir", type=Path, default=_BASE.DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--min-task-pass-rate", type=float, default=0.10)
    parser.add_argument("--max-task-pass-rate", type=float, default=0.50)
    parser.add_argument("--max-extension-task-pass-rate", type=float, default=0.30)
    parser.add_argument(
        "--max-top-quartile-extension-pass-rate", type=float, default=0.45
    )
    parser.add_argument("--max-subset-score", type=float, default=0.75)
    parser.add_argument("--max-p90-steps", type=float, default=150.0)
    parser.add_argument("--max-p90-input-tokens", type=float, default=13_000_000.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    analysis_dir = args.analysis_dir.resolve()
    anchors = [str(value) for value in _BASE._read_json(args.anchors_json.resolve())]
    selector = _BASE.EssentialSelector(
        trials_payload=_BASE._read_json(analysis_dir / "deepswe_v1_1_trials.json"),
        tasks_payload=_BASE._read_json(analysis_dir / "deepswe_v1_1_tasks.json"),
        tasks_root=args.tasks_root.resolve(),
    )
    runtime_stats = _BASE._runtime_stats(selector.trial_df)
    candidates = _BASE._eligible_candidates(
        selector,
        runtime_stats,
        anchors=anchors,
        languages=tuple(DEFAULT_LANGUAGE_COUNTS),
        excluded_tasks=set(_BASE.DEFAULT_EXCLUDED_TASKS),
        runtime_profiles=_BASE.DEFAULT_RUNTIME_PROFILES,
        cost_profiles=_BASE.DEFAULT_COST_PROFILES,
        max_p90_steps=args.max_p90_steps,
        max_p90_input_tokens=args.max_p90_input_tokens,
    )
    for language in candidates:
        candidates[language] = [
            task_name
            for task_name in candidates[language]
            if args.min_task_pass_rate
            <= float(selector.stats.loc[task_name, "task_pass_rate"])
            <= args.max_task_pass_rate
        ]
    extensions = _candidate_extensions(candidates, DEFAULT_LANGUAGE_COUNTS)
    all_rows = np.arange(len(selector.y))
    ranked = _rank_extensions(
        selector,
        anchors=anchors,
        extensions=extensions,
        rows=all_rows,
        max_extension_task_pass_rate=args.max_extension_task_pass_rate,
        max_top_quartile_extension_pass_rate=(
            args.max_top_quartile_extension_pass_rate
        ),
        max_subset_score=args.max_subset_score,
    )
    if not ranked:
        raise RuntimeError("No extension satisfies the hard-extension constraints")
    selected_extension = ranked[0]["task_names"]
    selected_names = anchors + selected_extension
    name_to_index = {name: index for index, name in enumerate(selector.task_names)}
    selected_indices = [name_to_index[name] for name in selected_names]
    fixed_metrics = _BASE._metrics(
        selector.X[:, selected_indices].mean(axis=1),
        selector.y,
    )

    model_meta = selector.meta.reset_index()
    oof_rows = []
    fold_selections = []
    for family in sorted(set(selector.families)):
        train_rows = np.where(selector.families != family)[0]
        test_rows = np.where(selector.families == family)[0]
        fold_ranked = _rank_extensions(
            selector,
            anchors=anchors,
            extensions=extensions,
            rows=train_rows,
            max_extension_task_pass_rate=args.max_extension_task_pass_rate,
            max_top_quartile_extension_pass_rate=(
                args.max_top_quartile_extension_pass_rate
            ),
            max_subset_score=args.max_subset_score,
        )
        if not fold_ranked:
            raise RuntimeError(f"No valid extension for held-out family {family}")
        fold_extension = fold_ranked[0]["task_names"]
        fold_indices = [
            name_to_index[name] for name in anchors + fold_extension
        ]
        predictions = selector.X[np.ix_(test_rows, fold_indices)].mean(axis=1)
        fold_selections.append(
            {"family": family, "selected_extension": fold_extension}
        )
        for offset, row_index in enumerate(test_rows):
            oof_rows.append(
                {
                    "held_out_family": family,
                    "effort_key": str(model_meta.iloc[row_index]["effort_key"]),
                    "full_score": float(selector.y[row_index]),
                    "subset_score": float(predictions[offset]),
                    "selected_extension": "|".join(fold_extension),
                }
            )
    oof_target = np.array([row["full_score"] for row in oof_rows], dtype=float)
    oof_predicted = np.array([row["subset_score"] for row in oof_rows], dtype=float)
    oof_metrics = _BASE._metrics(oof_predicted, oof_target)

    profile_rows = []
    for task_name in selected_extension:
        for effort_key in PUBLIC_PROFILE_KEYS:
            profile_rows.append(
                {
                    "task_name": task_name,
                    "effort_key": effort_key,
                    "pass_rate": _BASE._runtime_value(
                        runtime_stats, effort_key, task_name, "pass_rate"
                    ),
                    "average_cost_usd": _BASE._runtime_value(
                        runtime_stats, effort_key, task_name, "average_cost_usd"
                    ),
                    "p90_steps": _BASE._runtime_value(
                        runtime_stats, effort_key, task_name, "p90_steps"
                    ),
                    "p90_input_tokens": _BASE._runtime_value(
                        runtime_stats, effort_key, task_name, "p90_input_tokens"
                    ),
                }
            )

    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ranked).to_json(
        output_prefix.with_name(output_prefix.name + "_candidate_extensions.jsonl"),
        orient="records",
        lines=True,
        force_ascii=False,
    )
    pd.DataFrame(oof_rows).to_csv(
        output_prefix.with_name(output_prefix.name + "_selection_aware_oof.csv"),
        index=False,
    )
    _BASE._write_json(
        output_prefix.with_name(output_prefix.name + "_task_names.json"),
        selected_names,
    )
    _BASE._write_jsonl(
        output_prefix.with_suffix(".jsonl"),
        [
            selector.task_record("high", rank, name_to_index[task_name])
            for rank, task_name in enumerate(selected_names, start=1)
        ],
    )
    summary = {
        "status": "review_only_not_active",
        "anchors": anchors,
        "selected_extension": selected_extension,
        "selected_task_names": selected_names,
        "language_distribution": (
            selector.stats.loc[selected_names, "language"].value_counts().to_dict()
        ),
        "constraints": {
            "language_counts": DEFAULT_LANGUAGE_COUNTS,
            "min_task_pass_rate": args.min_task_pass_rate,
            "max_task_pass_rate": args.max_task_pass_rate,
            "max_extension_task_pass_rate": args.max_extension_task_pass_rate,
            "max_top_quartile_extension_pass_rate": (
                args.max_top_quartile_extension_pass_rate
            ),
            "max_subset_score": args.max_subset_score,
            "max_p90_steps": args.max_p90_steps,
            "max_p90_input_tokens": args.max_p90_input_tokens,
            "runtime_profiles": _BASE.DEFAULT_RUNTIME_PROFILES,
            "valid_cost_profiles": _BASE.DEFAULT_COST_PROFILES,
        },
        "fixed_subset_metrics": fixed_metrics,
        "selection_aware_leave_one_family_out_metrics": oof_metrics,
        "fold_selections": fold_selections,
        "selected_extension_public_profiles": profile_rows,
        "top_candidate_extensions": ranked[:10],
        "activation_note": (
            "Review only. Run environment preflight and the four-task reference "
            "pilot before changing the active Agentic SWE-Assorted manifest."
        ),
    }
    _BASE._write_json(
        output_prefix.with_name(output_prefix.name + "_summary.json"),
        summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_BASE._json_value))


if __name__ == "__main__":
    main()
