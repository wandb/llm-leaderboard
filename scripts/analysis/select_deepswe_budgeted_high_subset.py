#!/usr/bin/env python3
"""Select a budget-conscious, language-balanced DeepSWE High subset.

This is intentionally separate from ``select_deepswe_essential_subset.py``.
The Essential subsets optimize sparse approximation of the full DeepSWE matrix;
this selector starts from that idea but adds a hard budget/cap proxy so the
subset is usable inside the Taiwan Agentic SWE-Assorted harness.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[2]
ESSENTIAL_SELECTOR_PATH = REPO_ROOT / "scripts" / "analysis" / "select_deepswe_essential_subset.py"
_spec = importlib.util.spec_from_file_location("deepswe_essential_selector", ESSENTIAL_SELECTOR_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Failed to load {ESSENTIAL_SELECTOR_PATH}")
_essential = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_essential)

DEFAULT_ANALYSIS_DIR = _essential.DEFAULT_ANALYSIS_DIR
DEFAULT_OUTPUT_DIR = _essential.DEFAULT_OUTPUT_DIR
DEFAULT_TASKS_ROOT = _essential.DEFAULT_TASKS_ROOT
PUBLIC_TASKS_URL = _essential.PUBLIC_TASKS_URL
PUBLIC_TRIALS_URL = _essential.PUBLIC_TRIALS_URL
EssentialSelector = _essential.EssentialSelector
_fetch_or_read = _essential._fetch_or_read
_read_json = _essential._read_json
_write_json = _essential._write_json
_write_jsonl = _essential._write_jsonl


DEFAULT_LANGUAGE_COUNTS = ("go=3", "python=2", "typescript=3")
DEFAULT_MUST_INCLUDE = ("psd-tools-blend-range-api",)


def parse_key_values(values: list[str] | tuple[str, ...]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for raw in values:
        key, sep, value = raw.partition("=")
        if sep != "=":
            raise ValueError(f"expected key=value, got: {raw}")
        count = int(value)
        if count < 0:
            raise ValueError(f"language count must be non-negative: {raw}")
        parsed[key.strip().lower()] = count
    if sum(parsed.values()) <= 0:
        raise ValueError("at least one language count must be positive")
    return parsed


def read_task_names(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(item) for item in payload}


def corr_metrics(selector: EssentialSelector, selected: list[int]) -> dict[str, float]:
    subset_score = selector.X[:, selected].mean(axis=1)
    target = selector.y
    return {
        "n": float(len(target)),
        "mae": float(mean_absolute_error(target, subset_score)),
        "pearson": float(pearsonr(subset_score, target).statistic),
        "spearman": float(spearmanr(subset_score, target).statistic),
        "kendall": float(kendalltau(subset_score, target).statistic),
        "r2": float(r2_score(target, subset_score)),
    }


def fixed_subset_summary(selector: EssentialSelector, task_names: list[str]) -> dict[str, Any]:
    name_to_idx = {name: idx for idx, name in enumerate(selector.task_names)}
    selected = [name_to_idx[name] for name in task_names if name in name_to_idx]
    missing = [name for name in task_names if name not in name_to_idx]
    stats = selector.stats.iloc[selected] if selected else selector.stats.iloc[[]]
    return {
        "task_names": [selector.task_names[idx] for idx in selected],
        "missing_task_names": missing,
        "language_distribution": dict(Counter(selector.languages[idx] for idx in selected)),
        "sum_public_avg_cost_usd": float(stats["avg_cost_usd"].sum()) if selected else 0.0,
        "mean_public_avg_cost_usd": float(stats["avg_cost_usd"].mean()) if selected else 0.0,
        "mean_public_avg_steps": float(stats["avg_steps"].mean()) if selected else 0.0,
        "mean_public_avg_input_tokens": float(stats["avg_input_tokens"].mean()) if selected else 0.0,
        "mean_public_avg_duration_seconds": float(stats["avg_duration_seconds"].mean()) if selected else 0.0,
        "metrics": corr_metrics(selector, selected) if selected else {},
    }


def summarize_leave_family_out(rows: list[dict[str, Any]]) -> dict[str, float]:
    summary = {}
    for metric in ("pearson", "spearman", "kendall", "mae", "r2"):
        values = [
            float(row[metric])
            for row in rows
            if metric in row and not np.isnan(float(row[metric]))
        ]
        summary[f"lo_family_mean_{metric}"] = float(np.mean(values)) if values else float("nan")
        summary[f"lo_family_min_{metric}"] = float(np.min(values)) if values else float("nan")
    return summary


class BudgetedHighSelector:
    def __init__(
        self,
        *,
        selector: EssentialSelector,
        language_counts: dict[str, int],
        base_tasks: set[str],
        budgeted_tasks: set[str],
        must_include: set[str],
        exclude_tasks: set[str],
        max_avg_cost_usd: float,
        max_avg_steps: float,
        max_avg_input_tokens: float | None,
        budget_model: str | None,
        budget_effort: str | None,
        max_model_avg_steps: float | None,
        max_model_avg_input_tokens: float | None,
        min_pass_rate: float,
        max_pass_rate: float,
        max_candidates_per_language: int,
    ) -> None:
        self.selector = selector
        self.language_counts = language_counts
        self.base_tasks = base_tasks
        self.budgeted_tasks = budgeted_tasks
        self.must_include = must_include
        self.exclude_tasks = exclude_tasks
        self.max_avg_cost_usd = max_avg_cost_usd
        self.max_avg_steps = max_avg_steps
        self.max_avg_input_tokens = max_avg_input_tokens
        self.budget_model = budget_model
        self.budget_effort = budget_effort
        self.max_model_avg_steps = max_model_avg_steps
        self.max_model_avg_input_tokens = max_model_avg_input_tokens
        self.min_pass_rate = min_pass_rate
        self.max_pass_rate = max_pass_rate
        self.max_candidates_per_language = max_candidates_per_language

        self.names = np.array(selector.task_names)
        self.languages = np.array(selector.languages)
        self.repositories = np.array(selector.repositories)
        self.cost = selector.stats["avg_cost_usd"].to_numpy(float)
        self.steps = selector.stats["avg_steps"].to_numpy(float)
        self.input_tokens = selector.stats["avg_input_tokens"].to_numpy(float)
        self.duration = selector.stats["avg_duration_seconds"].to_numpy(float)
        self.pass_rate = selector.stats["task_pass_rate"].to_numpy(float)
        self.single_spearman = np.array(selector.single_spearman, dtype=float)
        self.base_mask = np.array([name in base_tasks for name in selector.task_names])
        self.budgeted_mask = np.array([name in budgeted_tasks for name in selector.task_names])
        self.target_rank = rankdata(selector.y)
        self.model_steps, self.model_input_tokens = self._model_budget_arrays()

    def _model_budget_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.budget_model:
            nan_values = np.full(len(self.selector.task_names), np.nan, dtype=float)
            return nan_values, nan_values.copy()
        model_df = self.selector.trial_df[self.selector.trial_df["model"] == self.budget_model]
        if self.budget_effort:
            model_df = model_df[model_df["reasoning_effort"] == self.budget_effort]
        grouped = model_df.groupby("task_name").agg(
            model_avg_steps=("n_agent_steps", "mean"),
            model_avg_input_tokens=("n_input_tokens", "mean"),
        )
        steps = []
        input_tokens = []
        for task_name in self.selector.task_names:
            if task_name not in grouped.index:
                steps.append(float("nan"))
                input_tokens.append(float("nan"))
                continue
            steps.append(float(grouped.loc[task_name, "model_avg_steps"]))
            input_tokens.append(float(grouped.loc[task_name, "model_avg_input_tokens"]))
        return np.array(steps, dtype=float), np.array(input_tokens, dtype=float)

    def candidate_score(self, idx: int) -> float:
        return float(
            self.single_spearman[idx]
            - 0.025 * self.cost[idx]
            - 0.0015 * self.steps[idx]
            - 0.0040 * (self.input_tokens[idx] / 1_000_000.0)
            - (
                0.0020 * (self.model_input_tokens[idx] / 1_000_000.0)
                if not np.isnan(self.model_input_tokens[idx])
                else 0.0
            )
            + (0.020 if self.base_mask[idx] else 0.0)
            + (0.006 if self.budgeted_mask[idx] else 0.0)
        )

    def candidates_by_language(self) -> dict[str, list[int]]:
        result: dict[str, list[int]] = {}
        for language, _count in self.language_counts.items():
            candidates: list[int] = []
            for idx, name in enumerate(self.selector.task_names):
                if name in self.exclude_tasks:
                    continue
                if self.languages[idx] != language:
                    continue
                if self.cost[idx] > self.max_avg_cost_usd and name not in self.must_include:
                    continue
                if self.steps[idx] > self.max_avg_steps and name not in self.must_include:
                    continue
                if (
                    self.max_avg_input_tokens is not None
                    and self.input_tokens[idx] > self.max_avg_input_tokens
                    and name not in self.must_include
                ):
                    continue
                if self.max_model_avg_steps is not None:
                    if np.isnan(self.model_steps[idx]) or self.model_steps[idx] > self.max_model_avg_steps:
                        continue
                if self.max_model_avg_input_tokens is not None:
                    if (
                        np.isnan(self.model_input_tokens[idx])
                        or self.model_input_tokens[idx] > self.max_model_avg_input_tokens
                    ):
                        continue
                if self.pass_rate[idx] < self.min_pass_rate or self.pass_rate[idx] > self.max_pass_rate:
                    continue
                candidates.append(idx)
            candidates.sort(key=self.candidate_score, reverse=True)
            limited = candidates[: self.max_candidates_per_language]
            for idx in candidates:
                if self.selector.task_names[idx] in self.must_include and idx not in limited:
                    limited.append(idx)
            result[language] = limited
        return result

    def fast_metrics(
        self,
        selected: list[int],
        rows: np.ndarray | None = None,
    ) -> tuple[float, float]:
        row_idx = np.arange(len(self.selector.y)) if rows is None else np.asarray(rows)
        subset_score = self.selector.X[np.ix_(row_idx, selected)].mean(axis=1)
        target = self.selector.y[row_idx]
        target_rank = rankdata(target)
        rank_score = rankdata(subset_score)
        centered_score = rank_score - rank_score.mean()
        centered_target = target_rank - target_rank.mean()
        spearman = float(
            (centered_score @ centered_target)
            / np.sqrt((centered_score @ centered_score) * (centered_target @ centered_target))
        )
        mae = float(np.mean(np.abs(target - subset_score)))
        return spearman, mae

    def objective(self, selected: list[int], rows: np.ndarray | None = None) -> float:
        spearman, mae = self.fast_metrics(selected, rows)
        selected_arr = np.array(selected, dtype=int)
        return float(
            spearman
            - 0.50 * mae
            - 0.025 * float(self.cost[selected_arr].mean())
            - 0.0015 * float(self.steps[selected_arr].mean())
            - 0.0040 * float(self.input_tokens[selected_arr].mean() / 1_000_000.0)
            + 0.008 * float(self.base_mask[selected_arr].sum())
            + 0.003 * float(self.budgeted_mask[selected_arr].sum())
        )

    def select(self, rows: np.ndarray | None = None) -> tuple[list[int], dict[str, Any]]:
        candidates = self.candidates_by_language()
        missing_languages = [
            language
            for language, count in self.language_counts.items()
            if len(candidates.get(language, [])) < count
        ]
        if missing_languages:
            raise RuntimeError(f"not enough candidates for languages: {missing_languages}")

        language_combinations = [
            list(itertools.combinations(candidates[language], count))
            for language, count in self.language_counts.items()
        ]
        best_score = -999.0
        best_selected: list[int] | None = None
        evaluated = 0
        for combo in itertools.product(*language_combinations):
            selected = [idx for part in combo for idx in part]
            selected_names = {self.selector.task_names[idx] for idx in selected}
            if not self.must_include <= selected_names:
                continue
            if len(set(self.repositories[selected])) < len(selected):
                continue
            evaluated += 1
            score = self.objective(selected, rows)
            if score > best_score:
                best_score = score
                best_selected = selected
        if best_selected is None:
            raise RuntimeError("failed to select a budgeted high subset")
        diagnostics = {
            "candidate_counts": {
                language: len(indices) for language, indices in candidates.items()
            },
            "candidate_task_names": {
                language: [self.selector.task_names[idx] for idx in indices]
                for language, indices in candidates.items()
            },
            "evaluated_combinations": evaluated,
            "objective": best_score,
        }
        return best_selected, diagnostics

    def leave_family_out(self) -> list[dict[str, Any]]:
        rows = []
        for family in sorted(set(self.selector.families)):
            test_idx = np.where(self.selector.families == family)[0]
            train_idx = np.where(self.selector.families != family)[0]
            if len(train_idx) < 5 or len(test_idx) < 1:
                continue
            selected, diagnostics = self.select(rows=train_idx)
            metrics = self.selector._metrics(selected, test_idx)
            rows.append(
                {
                    "family": family,
                    **metrics,
                    "selected": [self.selector.task_names[idx] for idx in selected],
                    "objective": diagnostics["objective"],
                }
            )
        return rows


def task_record(
    selector: EssentialSelector,
    subset_name: str,
    rank: int,
    idx: int,
    high_selector: BudgetedHighSelector | None = None,
) -> dict[str, Any]:
    record = selector.task_record(subset_name, rank, idx)
    task_stats = selector.stats.loc[selector.task_names[idx]]
    record["source"] = "DeepSWE v1.1 public rollout budgeted language-balanced selection"
    record["selection_stats"]["public_avg_steps"] = float(task_stats["avg_steps"])
    record["selection_stats"]["public_avg_input_tokens"] = float(task_stats["avg_input_tokens"])
    if high_selector is not None:
        model_steps = high_selector.model_steps[idx]
        model_input_tokens = high_selector.model_input_tokens[idx]
        if not np.isnan(model_steps):
            record["selection_stats"]["public_budget_model_avg_steps"] = float(model_steps)
        if not np.isnan(model_input_tokens):
            record["selection_stats"]["public_budget_model_avg_input_tokens"] = float(
                model_input_tokens
            )
    return record


def display_name_for_subset(subset_name: str) -> str:
    if subset_name == "essential_anchored_high_8_wandb_glm52_cap100_10m_lang_balanced":
        return "DeepSWE-Essential-Anchored-High-8-WandB-GLM52-Cap100-10M-Lang-Balanced"
    if subset_name == "essential_anchored_high_8_glm52max_cap100_10m_lang_balanced":
        return "DeepSWE-Essential-Anchored-High-8-GLM52Max-Cap100-10M-Lang-Balanced"
    if subset_name == "essential_anchored_high_8_essential3_cost_trimmed_lang_balanced":
        return "DeepSWE-Essential-Anchored-High-8-Essential3-Cost-Trimmed-Lang-Balanced"
    if subset_name == "essential_anchored_high_8_cost_trimmed_cap_safe_lang_balanced":
        return "DeepSWE-Essential-Anchored-High-8-Cost-Trimmed-Cap-Safe-Lang-Balanced"
    if subset_name == "essential_anchored_high_8_cost_trimmed_no_rust_lang_balanced":
        return "DeepSWE-Essential-Anchored-High-8-Cost-Trimmed-No-Rust-Lang-Balanced"
    if subset_name == "essential_anchored_high_8_safe_go_local_lang_balanced":
        return "DeepSWE-Essential-Anchored-High-8-Safe-Go-Local-Lang-Balanced"
    if subset_name == "essential_anchored_high_8_safe_local_image_lang_balanced":
        return "DeepSWE-Essential-Anchored-High-8-Safe-Local-Image-Lang-Balanced"
    if subset_name == "essential_anchored_high_8_safe_local_lang_balanced":
        return "DeepSWE-Essential-Anchored-High-8-Safe-Local-Lang-Balanced"
    if subset_name == "essential_anchored_high_8_safe_lang_balanced":
        return "DeepSWE-Essential-Anchored-High-8-Safe-Lang-Balanced"
    if subset_name == "essential_anchored_high_8_lang_balanced":
        return "DeepSWE-Essential-Anchored-High-8-Lang-Balanced"
    if "cap50" in subset_name:
        return "DeepSWE-Budgeted-High-8-Cap50-Lang-Balanced"
    if "lang_balanced" in subset_name:
        return "DeepSWE-Budgeted-High-8-Lang-Balanced"
    return "DeepSWE-Budgeted-High-8"


def update_manifest(
    output_dir: Path,
    *,
    subset_name: str,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    manifest_path = output_dir / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    manifest.setdefault("benchmark", "DeepSWE")
    manifest.setdefault("subsets", {})
    language_distribution = dict(
        sorted(Counter(record["language"] for record in records).items())
    )
    manifest["subsets"][subset_name] = {
        "display_name": display_name_for_subset(subset_name),
        "count": len(records),
        "language_distribution": language_distribution,
        "task_names_path": f"subsets/{subset_name}_task_names.json",
        "metadata_jsonl_path": f"subsets/{subset_name}.jsonl",
        "selection_metrics": summary["selected_subset"]["metrics"],
        "selection_summary_path": f"../../../outputs/deepswe_subset_analysis/{subset_name}_selection_summary.json",
        "selection_constraints": summary["constraints"],
    }
    manifest["budgeted_high_selection"] = {
        "name_prefix": "DeepSWE-Budgeted-High",
        "public_trials_url": PUBLIC_TRIALS_URL,
        "public_tasks_url": PUBLIC_TASKS_URL,
        "method": (
            "Budget-conscious fixed-size selection over public DeepSWE v1.1 rollout "
            "matrix. The default High-8 profile is anchored on DeepSWE-Essential-8, "
            "keeps three Essential tasks, excludes empirically cap-unsafe pilots, "
            "and optimizes Spearman-to-full under cost, step, and language-balance "
            "constraints."
        ),
    }
    _write_json(manifest_path, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", type=Path, default=DEFAULT_TASKS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--subset-name", default="budgeted_high_8_lang_balanced")
    parser.add_argument("--language-count", action="append", default=list(DEFAULT_LANGUAGE_COUNTS))
    parser.add_argument("--base-subset", default="essential_8")
    parser.add_argument("--budgeted-subset", default="budgeted_high_8")
    parser.add_argument("--must-include", action="append", default=None)
    parser.add_argument(
        "--exclude-task-name",
        action="append",
        default=None,
        help="Task name to exclude from the candidate pool. May be repeated.",
    )
    parser.add_argument(
        "--no-default-must-include",
        action="store_true",
        help="Do not automatically keep the default Essential-derived must-include tasks.",
    )
    parser.add_argument("--max-avg-cost-usd", type=float, default=4.8)
    parser.add_argument("--max-avg-steps", type=float, default=70.0)
    parser.add_argument(
        "--max-avg-input-tokens",
        type=float,
        default=None,
        help="Optional all-model public average input-token ceiling per candidate task.",
    )
    parser.add_argument(
        "--budget-model",
        help="Optional public DeepSWE model key, e.g. glm-5-2, for model-specific cap filtering.",
    )
    parser.add_argument(
        "--budget-effort",
        help="Optional public DeepSWE reasoning_effort, e.g. max, for model-specific cap filtering.",
    )
    parser.add_argument(
        "--max-model-avg-steps",
        type=float,
        default=None,
        help="Optional model-specific public average step ceiling per candidate task.",
    )
    parser.add_argument(
        "--max-model-avg-input-tokens",
        type=float,
        default=None,
        help="Optional model-specific public average input-token ceiling per candidate task.",
    )
    parser.add_argument("--min-pass-rate", type=float, default=0.12)
    parser.add_argument("--max-pass-rate", type=float, default=0.90)
    parser.add_argument("--max-candidates-per-language", type=int, default=8)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    output_dir = args.output_dir.resolve()
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
    language_counts = parse_key_values(args.language_count)
    base_tasks = read_task_names(output_dir / "subsets" / f"{args.base_subset}_task_names.json")
    budgeted_tasks = read_task_names(output_dir / "subsets" / f"{args.budgeted_subset}_task_names.json")
    must_include_values = list(args.must_include or [])
    if not args.no_default_must_include:
        must_include_values = list(DEFAULT_MUST_INCLUDE) + must_include_values
    must_include = {str(item) for item in must_include_values if str(item)}
    exclude_tasks = {str(item) for item in (args.exclude_task_name or []) if str(item)}
    overlap = must_include & exclude_tasks
    if overlap:
        raise ValueError(f"tasks cannot be both must-include and excluded: {sorted(overlap)}")
    high_selector = BudgetedHighSelector(
        selector=selector,
        language_counts=language_counts,
        base_tasks=base_tasks,
        budgeted_tasks=budgeted_tasks,
        must_include=must_include,
        exclude_tasks=exclude_tasks,
        max_avg_cost_usd=args.max_avg_cost_usd,
        max_avg_steps=args.max_avg_steps,
        max_avg_input_tokens=args.max_avg_input_tokens,
        budget_model=args.budget_model,
        budget_effort=args.budget_effort,
        max_model_avg_steps=args.max_model_avg_steps,
        max_model_avg_input_tokens=args.max_model_avg_input_tokens,
        min_pass_rate=args.min_pass_rate,
        max_pass_rate=args.max_pass_rate,
        max_candidates_per_language=max(1, int(args.max_candidates_per_language)),
    )
    selected, diagnostics = high_selector.select()
    records = [
        task_record(selector, args.subset_name, rank, idx, high_selector=high_selector)
        for rank, idx in enumerate(selected, start=1)
    ]
    selected_task_names = [record["task_name"] for record in records]
    leave_family_out = high_selector.leave_family_out()

    summary = {
        "subset_name": args.subset_name,
        "constraints": {
            "language_counts": language_counts,
            "base_subset": args.base_subset,
            "budgeted_subset": args.budgeted_subset,
            "must_include": sorted(must_include),
            "exclude_task_names": sorted(exclude_tasks),
            "max_avg_cost_usd": args.max_avg_cost_usd,
            "max_avg_steps": args.max_avg_steps,
            "max_avg_input_tokens": args.max_avg_input_tokens,
            "budget_model": args.budget_model,
            "budget_effort": args.budget_effort,
            "max_model_avg_steps": args.max_model_avg_steps,
            "max_model_avg_input_tokens": args.max_model_avg_input_tokens,
            "min_pass_rate": args.min_pass_rate,
            "max_pass_rate": args.max_pass_rate,
            "max_candidates_per_language": args.max_candidates_per_language,
        },
        "diagnostics": diagnostics,
        "selected_subset": fixed_subset_summary(selector, selected_task_names),
        "leave_family_out": {
            "summary": summarize_leave_family_out(leave_family_out),
            "rows": leave_family_out,
            "note": (
                "Each held-out model family is evaluated on tasks selected from the "
                "remaining families under the same budget/language constraints."
            ),
        },
        "comparisons": {
            args.base_subset: fixed_subset_summary(
                selector,
                list(read_task_names(output_dir / "subsets" / f"{args.base_subset}_task_names.json")),
            ),
            args.budgeted_subset: fixed_subset_summary(
                selector,
                list(read_task_names(output_dir / "subsets" / f"{args.budgeted_subset}_task_names.json")),
            ),
        },
    }
    _write_jsonl(output_dir / "subsets" / f"{args.subset_name}.jsonl", records)
    _write_json(
        output_dir / "subsets" / f"{args.subset_name}_task_names.json",
        selected_task_names,
    )
    _write_json(analysis_dir / f"{args.subset_name}_selection_summary.json", summary)
    pd.DataFrame(
        [
            {
                "task_name": record["task_name"],
                "language": record["language"],
                "repository": record["repository"],
                "problem_title": record["problem_title"],
                **record["selection_stats"],
            }
            for record in records
        ]
    ).to_csv(analysis_dir / f"{args.subset_name}_tasks.csv", index=False)
    update_manifest(output_dir, subset_name=args.subset_name, records=records, summary=summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
