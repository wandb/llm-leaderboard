#!/usr/bin/env python3
"""Select DeepSWE-Essential subsets from public DeepSWE rollout outcomes.

The selected subsets are ordinary unweighted task-name subsets.  Public
DeepSWE outcomes are used only for task selection; the leaderboard score for a
selected subset remains pass@1 over the selected tasks.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS_ROOT = REPO_ROOT / "external" / "deep-swe" / "tasks"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "taiwan" / "deepswe"
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "outputs" / "deepswe_subset_analysis"
PUBLIC_TRIALS_URL = "https://deepswe.datacurve.ai/artifacts/v1.1/trials.json"
PUBLIC_TASKS_URL = "https://deepswe.datacurve.ai/artifacts/v1.1/tasks.json"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _fetch_or_read(url: str, cache_path: Path, *, refresh: bool) -> Any:
    if cache_path.exists() and not refresh:
        return _read_json(cache_path)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    payload = response.json()
    _write_json(cache_path, payload)
    return payload


def _safe_corr(fn, x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or len(set(np.round(x, 10))) < 2 or len(set(np.round(y, 10))) < 2:
        return float("nan")
    value = fn(x, y).statistic
    return float(value) if not np.isnan(value) else float("nan")


class EssentialSelector:
    def __init__(
        self,
        *,
        trials_payload: dict[str, Any],
        tasks_payload: dict[str, Any],
        tasks_root: Path,
    ) -> None:
        trial_df = pd.DataFrame(trials_payload["rows"])
        self.task_df = pd.DataFrame(tasks_payload["rows"]).set_index("id")
        trial_df = trial_df[
            (trial_df["source"] == "deep-swe")
            & (trial_df["eval_scope"] == "full")
            & (trial_df["included_in_score"] == True)  # noqa: E712
        ].copy()
        trial_df["score_value"] = pd.to_numeric(trial_df["score_value"], errors="coerce")
        trial_df["effort_key"] = trial_df["config"]
        local_tasks = {path.name for path in tasks_root.iterdir() if path.is_dir()}
        pivot = trial_df.groupby(["effort_key", "task_name"])["score_value"].mean().unstack()
        columns = [column for column in pivot.columns if column in local_tasks]
        pivot = pivot[columns]
        target = trial_df.groupby("effort_key")["score_value"].mean().reindex(pivot.index)
        mask = pivot.notna().all(axis=1) & target.notna()

        self.trial_df = trial_df
        self.meta = (
            trial_df.groupby("effort_key")[["model", "reasoning_effort", "provider", "harness"]]
            .agg(lambda series: series.dropna().iloc[0] if series.dropna().size else None)
            .loc[pivot.loc[mask].index]
        )
        self.task_names = list(pivot.loc[mask].columns)
        self.X = pivot.loc[mask].to_numpy(float)
        self.y = target.loc[mask].to_numpy(float)
        self.families = self.meta["model"].astype(str).to_numpy()

        stats = trial_df.groupby("task_name").agg(
            task_pass_rate=("score_value", "mean"),
            avg_cost_usd=("cost_usd", "mean"),
            avg_duration_seconds=("agent_duration_seconds", "mean"),
            avg_steps=("n_agent_steps", "mean"),
            n=("score_value", "size"),
        )
        for column in ("language", "repository", "repository_url", "problem_title", "prompt_characters"):
            stats[column] = self.task_df[column]
        self.stats = stats.loc[self.task_names]
        self.cost_rank = self.stats["avg_cost_usd"].rank(pct=True).to_numpy(float)
        self.duration_rank = self.stats["avg_duration_seconds"].rank(pct=True).to_numpy(float)
        self.pass_rate = self.stats["task_pass_rate"].to_numpy(float)
        self.languages = self.stats["language"].astype(str).to_numpy()
        self.repositories = self.stats["repository"].astype(str).to_numpy()
        self.single_spearman = self._single_task_spearman()
        self.seed_order = list(
            np.argsort(
                -(
                    self.single_spearman
                    - 0.04 * self.cost_rank
                    - 0.02 * self.duration_rank
                    - 0.01 * np.abs(self.pass_rate - 0.5)
                )
            )
        )

    def _single_task_spearman(self) -> np.ndarray:
        values = []
        for idx in range(self.X.shape[1]):
            value = spearmanr(self.X[:, idx], self.y).statistic
            values.append(float(value) if not np.isnan(value) else -1.0)
        return np.array(values)

    def _metrics(self, selected: list[int], rows: np.ndarray | None = None) -> dict[str, float]:
        row_idx = np.arange(len(self.y)) if rows is None else np.asarray(rows)
        subset_score = self.X[np.ix_(row_idx, selected)].mean(axis=1)
        target = self.y[row_idx]
        return {
            "n": float(len(row_idx)),
            "mae": float(mean_absolute_error(target, subset_score)),
            "pearson": _safe_corr(pearsonr, subset_score, target),
            "spearman": _safe_corr(spearmanr, subset_score, target),
            "kendall": _safe_corr(kendalltau, subset_score, target),
            "r2": float(r2_score(target, subset_score))
            if len(row_idx) >= 2 and len(set(np.round(target, 10))) >= 2
            else float("nan"),
        }

    def _objective(self, selected: list[int], rows: np.ndarray) -> float:
        metrics = self._metrics(selected, rows)
        spearman = -1.0 if np.isnan(metrics["spearman"]) else metrics["spearman"]
        return spearman - 0.60 * metrics["mae"]

    def _language_caps(self, size: int) -> dict[str, int]:
        frequencies = pd.Series(self.languages).value_counts(normalize=True).to_dict()
        return {language: max(1, math.ceil(size * fraction) + 1) for language, fraction in frequencies.items()}

    def greedy_select(self, size: int, rows: np.ndarray | None = None, *, diversity: bool = True) -> list[int]:
        row_idx = np.arange(len(self.y)) if rows is None else np.asarray(rows)
        selected: list[int] = []
        language_count: dict[str, int] = {}
        repository_count: dict[str, int] = {}
        language_caps = self._language_caps(size)
        for _step in range(size):
            best_idx: int | None = None
            best_score = -999.0
            for idx in self.seed_order:
                if idx in selected:
                    continue
                if diversity:
                    if language_count.get(self.languages[idx], 0) >= language_caps.get(self.languages[idx], size):
                        continue
                    if repository_count.get(self.repositories[idx], 0) >= 2:
                        continue
                candidate = selected + [idx]
                score = self._objective(candidate, row_idx)
                score -= 0.012 * self.cost_rank[idx] / max(size, 1)
                score -= 0.006 * self.duration_rank[idx] / max(size, 1)
                score -= 0.003 * abs(self.pass_rate[idx] - 0.5)
                if score > best_score:
                    best_idx = idx
                    best_score = score
            if best_idx is None:
                for idx in self.seed_order:
                    if idx in selected:
                        continue
                    score = self._objective(selected + [idx], row_idx)
                    if score > best_score:
                        best_idx = idx
                        best_score = score
            if best_idx is None:
                raise RuntimeError("Failed to select a DeepSWE task")
            selected.append(best_idx)
            language_count[self.languages[best_idx]] = language_count.get(self.languages[best_idx], 0) + 1
            repository_count[self.repositories[best_idx]] = repository_count.get(self.repositories[best_idx], 0) + 1
        return selected

    def leave_family_out(self, size: int) -> list[dict[str, Any]]:
        rows = []
        for family in sorted(set(self.families)):
            test_idx = np.where(self.families == family)[0]
            train_idx = np.where(self.families != family)[0]
            if len(train_idx) < 5 or len(test_idx) < 1:
                continue
            selected = self.greedy_select(size, train_idx, diversity=True)
            metrics = self._metrics(selected, test_idx)
            rows.append(
                {
                    "family": family,
                    **metrics,
                    "selected": [self.task_names[idx] for idx in selected],
                }
            )
        return rows

    def task_record(self, subset: str, rank: int, idx: int) -> dict[str, Any]:
        task_name = self.task_names[idx]
        task_meta = self.task_df.loc[task_name]
        task_stats = self.stats.loc[task_name]
        prompt_chars = task_meta.get("prompt_characters")
        return {
            "task_id": task_name,
            "task_name": task_name,
            "subset": subset,
            "subset_rank": rank,
            "source": "DeepSWE v1.1 public rollout correlation selection",
            "language": task_meta.get("language"),
            "repository": task_meta.get("repository"),
            "repository_url": task_meta.get("repository_url"),
            "problem_title": task_meta.get("problem_title"),
            "prompt_characters": int(prompt_chars) if pd.notna(prompt_chars) else None,
            "selection_stats": {
                "public_task_pass_rate": float(task_stats["task_pass_rate"]),
                "public_avg_cost_usd": float(task_stats["avg_cost_usd"]),
                "public_avg_duration_seconds": float(task_stats["avg_duration_seconds"]),
                "single_task_spearman_to_full": float(self.single_spearman[idx]),
            },
        }


def _summarize_cv(cv_rows: list[dict[str, Any]]) -> dict[str, float]:
    summary = {}
    for metric in ("pearson", "spearman", "kendall", "mae", "r2"):
        values = [float(row[metric]) for row in cv_rows if not np.isnan(float(row[metric]))]
        summary[f"lo_family_mean_{metric}"] = float(np.mean(values)) if values else float("nan")
        summary[f"lo_family_min_{metric}"] = float(np.min(values)) if values else float("nan")
    return summary


def _update_manifest(output_dir: Path, subset_rows: list[dict[str, Any]]) -> None:
    manifest_path = output_dir / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    manifest.setdefault("benchmark", "DeepSWE")
    manifest.setdefault("subsets", {})
    manifest["essential_selection"] = {
        "name_prefix": "DeepSWE-Essential",
        "public_trials_url": PUBLIC_TRIALS_URL,
        "public_tasks_url": PUBLIC_TASKS_URL,
        "method": (
            "Supervised sparse greedy task selection from public DeepSWE v1.1 rollout "
            "matrix. Objective maximizes Spearman correlation to the full 113-task "
            "score with MAE, cost, duration, language, and repository penalties."
        ),
    }
    for row in subset_rows:
        name = f"essential_{int(row['k'])}"
        records = [
            json.loads(line)
            for line in (output_dir / "subsets" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        language_distribution = dict(sorted(pd.Series([r["language"] for r in records]).value_counts().to_dict().items()))
        manifest["subsets"][name] = {
            "display_name": f"DeepSWE-Essential-{int(row['k'])}",
            "count": len(records),
            "language_distribution": language_distribution,
            "task_names_path": f"subsets/{name}_task_names.json",
            "metadata_jsonl_path": f"subsets/{name}.jsonl",
            "selection_metrics": row,
        }
    _write_json(manifest_path, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", type=Path, default=DEFAULT_TASKS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 16, 20, 30])
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    output_dir = args.output_dir.resolve()
    trials_payload = _fetch_or_read(
        PUBLIC_TRIALS_URL,
        analysis_dir / "deepswe_v1_1_trials.json",
        refresh=args.refresh,
    )
    tasks_payload = _fetch_or_read(
        PUBLIC_TASKS_URL,
        analysis_dir / "deepswe_v1_1_tasks.json",
        refresh=args.refresh,
    )
    selector = EssentialSelector(
        trials_payload=trials_payload,
        tasks_payload=tasks_payload,
        tasks_root=args.tasks_root.resolve(),
    )
    print(
        f"DeepSWE public matrix: {selector.X.shape[0]} model-efforts x "
        f"{selector.X.shape[1]} tasks"
    )

    subset_summaries: list[dict[str, Any]] = []
    for size in args.sizes:
        selected = selector.greedy_select(size, diversity=True)
        in_sample = selector._metrics(selected)
        cv_rows = selector.leave_family_out(size)
        summary = {
            "k": size,
            **{f"in_{key}": value for key, value in in_sample.items() if key != "n"},
            **_summarize_cv(cv_rows),
        }
        subset_summaries.append(summary)
        subset = f"essential_{size}"
        records = [
            selector.task_record(subset, rank, idx)
            for rank, idx in enumerate(selected, start=1)
        ]
        _write_jsonl(output_dir / "subsets" / f"{subset}.jsonl", records)
        _write_json(
            output_dir / "subsets" / f"{subset}_task_names.json",
            [record["task_name"] for record in records],
        )
        _write_json(analysis_dir / f"{subset}_leave_family_out.json", cv_rows)
        print(
            f"{subset}: in_spearman={summary['in_spearman']:.3f}, "
            f"lo_family_mean_spearman={summary['lo_family_mean_spearman']:.3f}, "
            f"lo_family_mean_mae={summary['lo_family_mean_mae']:.3f}"
        )

    pd.DataFrame(subset_summaries).to_csv(
        analysis_dir / "subset_size_metrics.csv",
        index=False,
    )
    task_scores = selector.stats.copy()
    task_scores["single_spearman_to_full"] = selector.single_spearman
    task_scores["cost_rank_pct"] = selector.cost_rank
    task_scores["duration_rank_pct"] = selector.duration_rank
    task_scores.to_csv(analysis_dir / "task_selection_scores.csv")
    _write_json(analysis_dir / "selected_subsets.json", subset_summaries)
    _update_manifest(output_dir, subset_summaries)


if __name__ == "__main__":
    main()
