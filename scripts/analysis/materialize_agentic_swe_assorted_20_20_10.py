#!/usr/bin/env python3
"""Materialize the frozen Agentic SWE-Assorted 20/20/10 release subset.

Low and Middle are nested selections from the v2 36-task pools. The selected
tasks preserve public SWE-bench Lite difficulty strata and the observed score
profile of GPT-4.1 mini, GPT-5.6 Luna, and Claude Sonnet 4.6 while capping each
repository at four tasks. High is the reviewed DeepSWE High-8 anchor plus the
two approved harder extension tasks.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LITE_DIR = ROOT / "data" / "taiwan" / "swebench_lite_assorted" / "subsets"
DEEPSWE_DIR = ROOT / "data" / "taiwan" / "deepswe" / "subsets"
HIGH_REVIEW_DIR = ROOT / "outputs" / "deepswe_subset_analysis"

LOW_SOURCE = LITE_DIR / "low_v2_36.jsonl"
MIDDLE_SOURCE = LITE_DIR / "middle_v2_36.jsonl"
HIGH_SOURCE = (
    HIGH_REVIEW_DIR / "essential_anchored_high_10_extension_review_20260723.jsonl"
)

LOW_SUBSET = "low_v3_20"
MIDDLE_SUBSET = "middle_v3_20"
COMBINED_SUBSET = "low_middle_v3_40"
HIGH_SUBSET = "essential_anchored_high_10_model_fidelity_cost_balanced"
HIGH_ANCHOR_SUBSET = (
    "essential_anchored_high_8_wandb_glm52_cap100_10m_lang_balanced"
)
SELECTION_VERSION = "v3_stratified_model_fidelity_cost_20260724"

LOW_INSTANCE_IDS = (
    "matplotlib__matplotlib-26020",
    "mwaskom__seaborn-3190",
    "sphinx-doc__sphinx-8721",
    "pytest-dev__pytest-7432",
    "scikit-learn__scikit-learn-12471",
    "scikit-learn__scikit-learn-13584",
    "sympy__sympy-21847",
    "psf__requests-2317",
    "astropy__astropy-14995",
    "scikit-learn__scikit-learn-14894",
    "sympy__sympy-13647",
    "scikit-learn__scikit-learn-13439",
    "matplotlib__matplotlib-23964",
    "sphinx-doc__sphinx-8713",
    "sympy__sympy-14774",
    "mwaskom__seaborn-3010",
    "pytest-dev__pytest-5227",
    "django__django-14752",
    "django__django-11099",
    "django__django-16527",
)

MIDDLE_INSTANCE_IDS = (
    "sphinx-doc__sphinx-8435",
    "sphinx-doc__sphinx-11445",
    "pytest-dev__pytest-7168",
    "pytest-dev__pytest-11148",
    "scikit-learn__scikit-learn-14983",
    "pylint-dev__pylint-6506",
    "sympy__sympy-14396",
    "scikit-learn__scikit-learn-13497",
    "sympy__sympy-22005",
    "sympy__sympy-16988",
    "django__django-11964",
    "django__django-14580",
    "sphinx-doc__sphinx-7975",
    "django__django-13551",
    "django__django-12184",
    "matplotlib__matplotlib-23562",
    "psf__requests-1963",
    "pylint-dev__pylint-5859",
    "scikit-learn__scikit-learn-11281",
    "pylint-dev__pylint-7993",
)

HIGH_TASK_NAMES = (
    "go-genai-streamed-function-args",
    "etree-xml-diff-patch",
    "ytt-jsonpath-query-api",
    "mnamer-daemon-watch-lifecycle",
    "langchain-request-coalescing",
    "ofetch-per-origin-circuit-breaker",
    "kea-atomic-signal-selectors",
    "happy-dom-deterministic-intersectionobserver",
    "participle-grammar-conflict-analysis",
    "bandit-incremental-cache-control",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def select_lite_rows(
    source_path: Path,
    instance_ids: tuple[str, ...],
    *,
    subset: str,
    tier: str,
) -> list[dict[str, Any]]:
    source_rows = read_jsonl(source_path)
    by_id = {str(row["instance_id"]): row for row in source_rows}
    missing = [instance_id for instance_id in instance_ids if instance_id not in by_id]
    if missing:
        raise ValueError(f"{source_path} is missing selected IDs: {missing}")
    rows = []
    for rank, instance_id in enumerate(instance_ids, start=1):
        row = dict(by_id[instance_id])
        row.update(
            {
                "agentic_swe_selection_version": SELECTION_VERSION,
                "agentic_swe_selection_basis": (
                    "Nested stratified selection from the v2 36-task tier. "
                    "Preserves public difficulty and observed model-score profiles, "
                    "penalizes cost/runtime, and caps repositories at four tasks."
                ),
                "agentic_swe_subset_rank": rank,
                "agentic_swe_tier": tier,
                "source_subset": subset,
            }
        )
        rows.append(row)
    return rows


def lite_stats(source_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    source = read_jsonl(source_path)

    def values(items: list[dict[str, Any]], key: str) -> list[float]:
        return [float(row[key]) for row in items]

    source_public = values(source, "public_lite_resolve_rate")
    selected_public = values(rows, "public_lite_resolve_rate")
    source_static = values(source, "static_difficulty_score")
    selected_static = values(rows, "static_difficulty_score")
    return {
        "source_count": len(source),
        "selected_count": len(rows),
        "public_resolve_rate": {
            "source_mean": mean(source_public),
            "source_population_sd": pstdev(source_public),
            "selected_mean": mean(selected_public),
            "selected_population_sd": pstdev(selected_public),
        },
        "static_difficulty": {
            "source_mean": mean(source_static),
            "selected_mean": mean(selected_static),
        },
        "repo_distribution": dict(
            sorted(Counter(str(row["repo"]) for row in rows).items())
        ),
    }


def write_lite_subset(subset: str, rows: list[dict[str, Any]]) -> None:
    write_jsonl(LITE_DIR / f"{subset}.jsonl", rows)
    write_csv(LITE_DIR / f"{subset}.csv", rows)
    (LITE_DIR / f"{subset}_instance_ids.json").write_text(
        json.dumps([row["instance_id"] for row in rows], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def materialize_high() -> list[dict[str, Any]]:
    source_rows = read_jsonl(HIGH_SOURCE)
    by_name = {str(row["task_name"]): row for row in source_rows}
    missing = [name for name in HIGH_TASK_NAMES if name not in by_name]
    if missing:
        raise ValueError(f"{HIGH_SOURCE} is missing selected tasks: {missing}")
    rows = []
    for rank, task_name in enumerate(HIGH_TASK_NAMES, start=1):
        row = dict(by_name[task_name])
        row.update(
            {
                "subset": HIGH_SUBSET,
                "subset_rank": rank,
                "selection_version": "high10_extension_review_20260723",
            }
        )
        rows.append(row)
    write_jsonl(DEEPSWE_DIR / f"{HIGH_SUBSET}.jsonl", rows)
    (DEEPSWE_DIR / f"{HIGH_SUBSET}_task_names.json").write_text(
        json.dumps(list(HIGH_TASK_NAMES), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return rows


def manifest_subset_entry(
    subset: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    static_values = [float(row["static_difficulty_score"]) for row in rows]
    return {
        "count": len(rows),
        "jsonl_path": f"subsets/{subset}.jsonl",
        "csv_path": f"subsets/{subset}.csv",
        "instance_ids_path": f"subsets/{subset}_instance_ids.json",
        "repo_distribution": dict(
            sorted(Counter(str(row["repo"]) for row in rows).items())
        ),
        "static_difficulty": {
            "min": min(static_values),
            "mean": mean(static_values),
            "max": max(static_values),
        },
        "public_lite_resolve_rate_mean": mean(
            float(row["public_lite_resolve_rate"]) for row in rows
        ),
        "gold_patch_changed_lines_mean": mean(
            float(row["gold_patch_changed_lines"]) for row in rows
        ),
        "fail_to_pass_count_mean": mean(
            float(row["fail_to_pass_count"]) for row in rows
        ),
        "pass_to_pass_count_mean": mean(
            float(row["pass_to_pass_count"]) for row in rows
        ),
    }


def update_manifests(
    low: list[dict[str, Any]],
    middle: list[dict[str, Any]],
    high: list[dict[str, Any]],
) -> None:
    lite_path = LITE_DIR.parent / "manifest.json"
    lite = json.loads(lite_path.read_text(encoding="utf-8"))
    lite["selection_versions"][SELECTION_VERSION] = {
        "subsets": [LOW_SUBSET, MIDDLE_SUBSET, COMBINED_SUBSET],
        "source_subsets": ["low_v2_36", "middle_v2_36"],
        "method": (
            "Five public-difficulty strata per tier with four tasks selected "
            "per stratum. The objective preserves three observed model score "
            "profiles and public/static difficulty while penalizing cost, wall "
            "time, and repository concentration."
        ),
        "reference_models": [
            "gpt-4.1-mini",
            "gpt-5.6-luna-high",
            "claude-sonnet-4.6-high",
        ],
        "repo_cap": 4,
        "selection_audit_path": (
            "../../../outputs/agentic_swe_subset_analysis/"
            "assorted_20_20_10_selection_20260724.json"
        ),
    }
    lite["subsets"][LOW_SUBSET] = manifest_subset_entry(LOW_SUBSET, low)
    lite["subsets"][MIDDLE_SUBSET] = manifest_subset_entry(MIDDLE_SUBSET, middle)
    lite["subsets"][COMBINED_SUBSET] = manifest_subset_entry(
        COMBINED_SUBSET, low + middle
    )
    lite["assorted_80_plan"]["status"] = "historical_v2"
    lite["assorted_50_plan"] = {
        "status": "frozen_default",
        "frozen_at": "2026-07-24",
        "low": 20,
        "middle": 20,
        "high": 10,
        "low_middle_jsonl_path": f"subsets/{COMBINED_SUBSET}.jsonl",
        "high_subset": (
            f"data/taiwan/deepswe/subsets/{HIGH_SUBSET}.jsonl"
        ),
        "score_weights": {"low": 1 / 3, "middle": 1 / 3, "high": 1 / 3},
        "score_definition": (
            "Score = (Low mean task score + Middle mean task score + High "
            "mean task score) / 3. Per task: resolved=1; otherwise an applied "
            "scoreable patch receives up to 0.3 * F2P^2 * P2P."
        ),
    }
    lite_path.write_text(
        json.dumps(lite, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    deepswe_path = DEEPSWE_DIR.parent / "manifest.json"
    deepswe = json.loads(deepswe_path.read_text(encoding="utf-8"))
    if HIGH_ANCHOR_SUBSET in deepswe["subsets"]:
        deepswe["subsets"][HIGH_ANCHOR_SUBSET][
            "default_for_agentic_swe_assorted"
        ] = False
        deepswe["subsets"][HIGH_ANCHOR_SUBSET]["status"] = "historical_frozen"
    deepswe["agentic_swe_assorted_default_high_subset"] = HIGH_SUBSET
    deepswe["agentic_swe_assorted_default_high_status"] = {
        "status": "frozen",
        "frozen_at": "2026-07-24",
        "change_policy": (
            "Do not modify task membership in place. Create a new subset name "
            "and switch defaults only after explicit approval."
        ),
    }
    deepswe["subsets"][HIGH_SUBSET] = {
        "display_name": (
            "DeepSWE-Essential-Anchored-High-10-Model-Fidelity-Cost-Balanced"
        ),
        "status": "frozen_default",
        "frozen_at": "2026-07-24",
        "default_for_agentic_swe_assorted": True,
        "change_policy": (
            "Do not modify this subset's task list in place. Create a new "
            "subset name for any future alternative."
        ),
        "count": len(high),
        "language_distribution": dict(
            sorted(Counter(str(row["language"]) for row in high).items())
        ),
        "task_names_path": f"subsets/{HIGH_SUBSET}_task_names.json",
        "metadata_jsonl_path": f"subsets/{HIGH_SUBSET}.jsonl",
        "selection_metrics": {
            "n": 40.0,
            "mae": 0.06004394943094129,
            "pearson": 0.9470468963713088,
            "spearman": 0.9320565914635588,
            "kendall": 0.8013928976732341,
            "selection_aware_leave_one_model_family_out": {
                "n": 40.0,
                "mae": 0.07719296208734212,
                "pearson": 0.9211417078954856,
                "spearman": 0.8965393865487309,
                "kendall": 0.7532982115358966,
            },
        },
        "selection_summary_path": (
            "../../../outputs/deepswe_subset_analysis/"
            "essential_anchored_high_10_extension_review_20260723_summary.json"
        ),
        "selection_constraints": {
            "base_subset": HIGH_ANCHOR_SUBSET,
            "anchored_task_count": 8,
            "extension_task_count": 2,
            "language_counts": {"go": 4, "python": 3, "typescript": 3},
            "max_p90_steps": 150.0,
            "max_p90_input_tokens": 13_000_000.0,
        },
    }
    deepswe_path.write_text(
        json.dumps(deepswe, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    low = select_lite_rows(
        LOW_SOURCE,
        LOW_INSTANCE_IDS,
        subset=LOW_SUBSET,
        tier="low",
    )
    middle = select_lite_rows(
        MIDDLE_SOURCE,
        MIDDLE_INSTANCE_IDS,
        subset=MIDDLE_SUBSET,
        tier="middle",
    )
    write_lite_subset(LOW_SUBSET, low)
    write_lite_subset(MIDDLE_SUBSET, middle)
    write_lite_subset(COMBINED_SUBSET, low + middle)
    high = materialize_high()
    update_manifests(low, middle, high)

    audit = {
        "selection_version": SELECTION_VERSION,
        "status": "frozen_default",
        "task_counts": {"low": len(low), "middle": len(middle), "high": len(high)},
        "score_weights": {"low": 1 / 3, "middle": 1 / 3, "high": 1 / 3},
        "low": lite_stats(LOW_SOURCE, low),
        "middle": lite_stats(MIDDLE_SOURCE, middle),
        "reference_run_fidelity": {
            "score_formula": "resolved=1 else 0.3 * F2P^2 * P2P",
            "rows": [
                {
                    "model": "gpt-4.1-mini",
                    "tier": "low",
                    "binary_source": 0.7778,
                    "binary_selected": 0.8,
                    "official_source": 0.8008,
                    "official_selected": 0.8053,
                },
                {
                    "model": "gpt-4.1-mini",
                    "tier": "middle",
                    "binary_source": 0.5278,
                    "binary_selected": 0.55,
                    "official_source": 0.543,
                    "official_selected": 0.55,
                },
                {
                    "model": "gpt-5.6-luna-high",
                    "tier": "low",
                    "binary_source": 0.9444,
                    "binary_selected": 0.95,
                    "official_source": 0.9465,
                    "official_selected": 0.95,
                },
                {
                    "model": "gpt-5.6-luna-high",
                    "tier": "middle",
                    "binary_source": 0.75,
                    "binary_selected": 0.75,
                    "official_source": 0.7665,
                    "official_selected": 0.765,
                },
                {
                    "model": "claude-sonnet-4.6-high",
                    "tier": "low",
                    "binary_source": 0.9444,
                    "binary_selected": 0.95,
                    "official_source": 0.9444,
                    "official_selected": 0.95,
                },
                {
                    "model": "claude-sonnet-4.6-high",
                    "tier": "middle",
                    "binary_source": 0.75,
                    "binary_selected": 0.75,
                    "official_source": 0.75,
                    "official_selected": 0.75,
                },
            ],
        },
        "high": {
            "subset": HIGH_SUBSET,
            "language_distribution": dict(
                sorted(Counter(str(row["language"]) for row in high).items())
            ),
            "in_sample_full_deepswe_fidelity": {
                "pearson": 0.9470468963713088,
                "spearman": 0.9320565914635588,
                "kendall": 0.8013928976732341,
            },
            "selection_aware_leave_one_model_family_out": {
                "pearson": 0.9211417078954856,
                "spearman": 0.8965393865487309,
                "kendall": 0.7532982115358966,
            },
        },
        "selection_method": (
            "Low/Middle: five public-difficulty strata with four tasks selected "
            "per stratum; objective preserves three observed model score profiles "
            "and public/static difficulty while penalizing cost, wall time, and "
            "repository concentration. High: reviewed frozen High-8 anchors plus "
            "one Go and one Python extension selected from public DeepSWE v1.1."
        ),
    }
    audit_path = (
        ROOT
        / "outputs"
        / "agentic_swe_subset_analysis"
        / "assorted_20_20_10_selection_20260724.json"
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
