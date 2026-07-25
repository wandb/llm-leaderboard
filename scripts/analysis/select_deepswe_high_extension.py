#!/usr/bin/env python3
"""Select a small, budget-safe extension to the frozen DeepSWE High subset.

The existing High tasks remain anchors. Candidate additions are selected from
the public DeepSWE rollout matrix using rank fidelity, upper-end headroom,
runtime caps, cost, language balance, and repository diversity. The reported
leave-one-model-family-out result reselects only the extension on each training
fold; it does not pretend that the already-frozen anchors were selected OOF.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


REPO_ROOT = Path(__file__).resolve().parents[2]
SELECTOR_PATH = REPO_ROOT / "scripts" / "analysis" / "select_deepswe_essential_subset.py"
_SPEC = importlib.util.spec_from_file_location("deepswe_essential_selector", SELECTOR_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load {SELECTOR_PATH}")
_SELECTOR_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SELECTOR_MODULE)

EssentialSelector = _SELECTOR_MODULE.EssentialSelector
_read_json = _SELECTOR_MODULE._read_json

DEFAULT_ANALYSIS_DIR = REPO_ROOT / "outputs" / "deepswe_subset_analysis"
DEFAULT_TASKS_ROOT = REPO_ROOT / "external" / "deep-swe" / "tasks"
DEFAULT_ANCHORS = (
    REPO_ROOT
    / "data"
    / "taiwan"
    / "deepswe"
    / "subsets"
    / "essential_anchored_high_8_wandb_glm52_cap100_10m_lang_balanced_task_names.json"
)
DEFAULT_OUTPUT_PREFIX = (
    DEFAULT_ANALYSIS_DIR / "essential_anchored_high_10_extension_review_20260723"
)
DEFAULT_RUNTIME_PROFILES = (
    "mini_swe_agent_glm_5_2_max",
    "mini_swe_agent_gpt_5_6_luna_high",
    "mini_swe_agent_claude_sonnet_4_6_high",
    "mini_swe_agent_gpt_5_6_sol_max",
)
DEFAULT_COST_PROFILES = ("mini_swe_agent_claude_fable_5_max",)
DEFAULT_EXCLUDED_TASKS = (
    "termenv-preserve-ansi-resets",
    "superjson-error-stack-serialization",
    "ts-pattern-match-each",
    "kcp-go-multiplexed-kcp-streams",
)


def _safe_correlation(fn, x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or len(set(np.round(x, 10))) < 2 or len(set(np.round(y, 10))) < 2:
        return float("nan")
    value = fn(x, y).statistic
    return float(value) if not np.isnan(value) else float("nan")


def _metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    # Mean accumulation order can otherwise perturb exact ties by ~1e-16 and
    # make rank correlations depend on task ordering.
    rank_predicted = np.round(predicted, 12)
    rank_target = np.round(target, 12)
    return {
        "n": float(len(target)),
        "mae": float(mean_absolute_error(target, predicted)),
        "pearson": _safe_correlation(pearsonr, rank_predicted, rank_target),
        "spearman": _safe_correlation(spearmanr, rank_predicted, rank_target),
        "kendall": _safe_correlation(kendalltau, rank_predicted, rank_target),
        "r2": (
            float(r2_score(target, predicted))
            if len(target) >= 2 and len(set(np.round(target, 10))) >= 2
            else float("nan")
        ),
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_value) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_value) + "\n")


def _runtime_stats(trial_df: pd.DataFrame) -> pd.DataFrame:
    return trial_df.groupby(["effort_key", "task_name"]).agg(
        pass_rate=("score_value", "mean"),
        average_cost_usd=("cost_usd", "mean"),
        p90_steps=("n_agent_steps", lambda values: values.quantile(0.9)),
        p90_input_tokens=("n_input_tokens", lambda values: values.quantile(0.9)),
    )


def _runtime_value(
    runtime_stats: pd.DataFrame,
    effort_key: str,
    task_name: str,
    column: str,
) -> float:
    try:
        return float(runtime_stats.loc[(effort_key, task_name), column])
    except KeyError:
        return float("nan")


def _eligible_candidates(
    selector: EssentialSelector,
    runtime_stats: pd.DataFrame,
    *,
    anchors: list[str],
    languages: tuple[str, ...],
    excluded_tasks: set[str],
    runtime_profiles: tuple[str, ...],
    cost_profiles: tuple[str, ...],
    max_p90_steps: float,
    max_p90_input_tokens: float,
) -> dict[str, list[str]]:
    anchor_repositories = set(selector.stats.loc[anchors, "repository"].astype(str))
    candidates = {language: [] for language in languages}
    for task_name in selector.task_names:
        language = str(selector.stats.loc[task_name, "language"]).lower()
        repository = str(selector.stats.loc[task_name, "repository"])
        if language not in candidates:
            continue
        if task_name in anchors or task_name in excluded_tasks:
            continue
        if repository in anchor_repositories:
            continue
        runtime_ok = True
        for effort_key in runtime_profiles:
            p90_steps = _runtime_value(runtime_stats, effort_key, task_name, "p90_steps")
            p90_input = _runtime_value(
                runtime_stats, effort_key, task_name, "p90_input_tokens"
            )
            if (
                np.isnan(p90_steps)
                or np.isnan(p90_input)
                or p90_steps > max_p90_steps
                or p90_input > max_p90_input_tokens
            ):
                runtime_ok = False
                break
        for effort_key in cost_profiles:
            profile_cost = _runtime_value(
                runtime_stats, effort_key, task_name, "average_cost_usd"
            )
            profile_steps = _runtime_value(runtime_stats, effort_key, task_name, "p90_steps")
            if np.isnan(profile_cost) or np.isnan(profile_steps) or profile_steps <= 0:
                runtime_ok = False
                break
        if runtime_ok:
            candidates[language].append(task_name)
    return candidates


def _pair_evaluation(
    selector: EssentialSelector,
    *,
    anchor_indices: list[int],
    pair_indices: list[int],
    rows: np.ndarray,
) -> dict[str, float]:
    selected = anchor_indices + pair_indices
    predicted = selector.X[np.ix_(rows, selected)].mean(axis=1)
    target = selector.y[rows]
    result = _metrics(predicted, target)
    top_cutoff = float(np.quantile(target, 0.75))
    top_rows = rows[target >= top_cutoff]
    pair_scores = selector.X[np.ix_(rows, pair_indices)].mean(axis=1)
    result.update(
        {
            "pair_pass_rate": float(pair_scores.mean()),
            "top_quartile_pair_pass_rate": float(
                selector.X[np.ix_(top_rows, pair_indices)].mean()
            ),
            "maximum_subset_score": float(predicted.max()),
        }
    )
    return result


def _objective(metrics: dict[str, float], average_cost_usd: float) -> float:
    """Balance rank fidelity with useful top-end difficulty and practical cost."""
    return (
        0.48 * metrics["spearman"]
        + 0.30 * metrics["pearson"]
        - 0.08 * metrics["mae"]
        - 0.03 * average_cost_usd
        - 0.06 * abs(metrics["top_quartile_pair_pass_rate"] - 0.60)
        - 0.04 * abs(metrics["pair_pass_rate"] - 0.40)
        - 0.05 * max(0.0, metrics["maximum_subset_score"] - 0.90)
    )


def _rank_pairs(
    selector: EssentialSelector,
    *,
    anchors: list[str],
    candidates: dict[str, list[str]],
    languages: tuple[str, str],
    rows: np.ndarray,
) -> list[dict[str, Any]]:
    name_to_index = {name: idx for idx, name in enumerate(selector.task_names)}
    anchor_indices = [name_to_index[name] for name in anchors]
    ranked: list[dict[str, Any]] = []
    for first in candidates[languages[0]]:
        for second in candidates[languages[1]]:
            first_repository = str(selector.stats.loc[first, "repository"])
            second_repository = str(selector.stats.loc[second, "repository"])
            if first_repository == second_repository:
                continue
            pair = [first, second]
            pair_indices = [name_to_index[name] for name in pair]
            metrics = _pair_evaluation(
                selector,
                anchor_indices=anchor_indices,
                pair_indices=pair_indices,
                rows=rows,
            )
            average_cost = float(selector.stats.loc[pair, "avg_cost_usd"].mean())
            ranked.append(
                {
                    "objective": _objective(metrics, average_cost),
                    "task_names": pair,
                    "languages": list(languages),
                    "repositories": [first_repository, second_repository],
                    "average_public_cost_usd": average_cost,
                    "metrics": metrics,
                }
            )
    ranked.sort(key=lambda row: row["objective"], reverse=True)
    return ranked


def _selection_aware_oof(
    selector: EssentialSelector,
    *,
    anchors: list[str],
    candidates: dict[str, list[str]],
    languages: tuple[str, str],
) -> tuple[list[dict[str, Any]], dict[str, float], list[dict[str, Any]]]:
    name_to_index = {name: idx for idx, name in enumerate(selector.task_names)}
    meta = selector.meta.reset_index()
    oof_rows: list[dict[str, Any]] = []
    fold_selections: list[dict[str, Any]] = []
    for family in sorted(set(selector.families)):
        test_rows = np.where(selector.families == family)[0]
        train_rows = np.where(selector.families != family)[0]
        best = _rank_pairs(
            selector,
            anchors=anchors,
            candidates=candidates,
            languages=languages,
            rows=train_rows,
        )[0]
        selected_names = anchors + best["task_names"]
        fold_selections.append(
            {
                "family": family,
                "selected": selected_names,
                "selected_extension": best["task_names"],
                "objective": best["objective"],
            }
        )
        selected_indices = [name_to_index[name] for name in selected_names]
        predictions = selector.X[np.ix_(test_rows, selected_indices)].mean(axis=1)
        for offset, row_index in enumerate(test_rows):
            oof_rows.append(
                {
                    "held_out_family": family,
                    "effort_key": str(meta.iloc[row_index]["effort_key"]),
                    "model": str(meta.iloc[row_index]["model"]),
                    "reasoning_effort": str(meta.iloc[row_index]["reasoning_effort"]),
                    "full_score": float(selector.y[row_index]),
                    "subset_score": float(predictions[offset]),
                    "selected_extension": "|".join(best["task_names"]),
                }
            )
    predicted = np.array([row["subset_score"] for row in oof_rows], dtype=float)
    target = np.array([row["full_score"] for row in oof_rows], dtype=float)
    return oof_rows, _metrics(predicted, target), fold_selections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors-json", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--tasks-root", type=Path, default=DEFAULT_TASKS_ROOT)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--languages", nargs=2, default=("go", "python"))
    parser.add_argument(
        "--runtime-profile",
        action="append",
        dest="runtime_profiles",
        default=None,
        help="Public DeepSWE effort_key used for runtime-cap eligibility.",
    )
    parser.add_argument(
        "--cost-profile",
        action="append",
        dest="cost_profiles",
        default=None,
        help=(
            "Public DeepSWE effort_key that must have valid nonzero execution and "
            "cost data, even when it is not used as a runtime-cap profile."
        ),
    )
    parser.add_argument("--max-p90-steps", type=float, default=150.0)
    parser.add_argument("--max-p90-input-tokens", type=float, default=13_000_000.0)
    parser.add_argument(
        "--exclude-task",
        action="append",
        dest="excluded_tasks",
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    anchors = [str(value) for value in _read_json(args.anchors_json.resolve())]
    runtime_profiles = tuple(args.runtime_profiles or DEFAULT_RUNTIME_PROFILES)
    cost_profiles = tuple(args.cost_profiles or DEFAULT_COST_PROFILES)
    excluded_tasks = set(args.excluded_tasks or DEFAULT_EXCLUDED_TASKS)
    languages = tuple(str(value).lower() for value in args.languages)
    if len(languages) != 2 or languages[0] == languages[1]:
        raise ValueError("--languages must contain two distinct languages")

    selector = EssentialSelector(
        trials_payload=_read_json(analysis_dir / "deepswe_v1_1_trials.json"),
        tasks_payload=_read_json(analysis_dir / "deepswe_v1_1_tasks.json"),
        tasks_root=args.tasks_root.resolve(),
    )
    runtime_stats = _runtime_stats(selector.trial_df)
    candidates = _eligible_candidates(
        selector,
        runtime_stats,
        anchors=anchors,
        languages=languages,
        excluded_tasks=excluded_tasks,
        runtime_profiles=runtime_profiles,
        cost_profiles=cost_profiles,
        max_p90_steps=args.max_p90_steps,
        max_p90_input_tokens=args.max_p90_input_tokens,
    )
    if any(not candidates[language] for language in languages):
        raise RuntimeError(f"No eligible candidates for one or more languages: {candidates}")

    all_rows = np.arange(len(selector.y))
    ranked = _rank_pairs(
        selector,
        anchors=anchors,
        candidates=candidates,
        languages=languages,
        rows=all_rows,
    )
    selected_pair = ranked[0]["task_names"]
    selected_names = anchors + selected_pair
    name_to_index = {name: idx for idx, name in enumerate(selector.task_names)}
    selected_indices = [name_to_index[name] for name in selected_names]
    in_sample_scores = selector.X[:, selected_indices].mean(axis=1)
    in_sample_metrics = _metrics(in_sample_scores, selector.y)
    oof_rows, oof_metrics, fold_selections = _selection_aware_oof(
        selector,
        anchors=anchors,
        candidates=candidates,
        languages=languages,
    )

    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ranked).to_json(
        output_prefix.with_name(output_prefix.name + "_candidate_pairs.jsonl"),
        orient="records",
        lines=True,
        force_ascii=False,
    )
    pd.DataFrame(oof_rows).to_csv(
        output_prefix.with_name(output_prefix.name + "_selection_aware_oof.csv"),
        index=False,
    )
    _write_json(
        output_prefix.with_name(output_prefix.name + "_task_names.json"),
        selected_names,
    )
    records = [
        selector.task_record("high", rank, name_to_index[task_name])
        for rank, task_name in enumerate(selected_names, start=1)
    ]
    _write_jsonl(output_prefix.with_suffix(".jsonl"), records)

    runtime_profile_rows = []
    profile_keys = tuple(dict.fromkeys((*runtime_profiles, *cost_profiles)))
    for task_name in selected_pair:
        for effort_key in profile_keys:
            runtime_profile_rows.append(
                {
                    "task_name": task_name,
                    "effort_key": effort_key,
                    "pass_rate": _runtime_value(
                        runtime_stats, effort_key, task_name, "pass_rate"
                    ),
                    "average_cost_usd": _runtime_value(
                        runtime_stats, effort_key, task_name, "average_cost_usd"
                    ),
                    "p90_steps": _runtime_value(
                        runtime_stats, effort_key, task_name, "p90_steps"
                    ),
                    "p90_input_tokens": _runtime_value(
                        runtime_stats, effort_key, task_name, "p90_input_tokens"
                    ),
                }
            )
    summary = {
        "status": "review_only_not_active",
        "method": (
            "Frozen High-8 anchors plus one Go and one Python task. The extension "
            "maximizes full-score rank fidelity while penalizing cost, excessive "
            "top-end pass rate, and runtime-cap violations."
        ),
        "anchors": anchors,
        "selected_extension": selected_pair,
        "selected_task_names": selected_names,
        "eligible_candidate_counts": {
            language: len(candidates[language]) for language in languages
        },
        "runtime_constraints": {
            "effort_keys": runtime_profiles,
            "valid_cost_effort_keys": cost_profiles,
            "max_p90_steps": args.max_p90_steps,
            "max_p90_input_tokens": args.max_p90_input_tokens,
        },
        "in_sample_fixed_subset": in_sample_metrics,
        "selection_aware_leave_one_model_family_out_extension": oof_metrics,
        "leave_family_out": {
            "method": (
                "Frozen anchors with the two-task extension reselected using only "
                "non-held-out model families."
            ),
            "summary": oof_metrics,
            "rows": fold_selections,
        },
        "selected_pair_public_profiles": runtime_profile_rows,
        "top_candidate_pairs": ranked[:10],
        "excluded_tasks": sorted(excluded_tasks),
        "activation_note": (
            "This artifact does not change the active Agentic SWE-Assorted subset. "
            "Activate only after review."
        ),
    }
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")
    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_value))


if __name__ == "__main__":
    main()
