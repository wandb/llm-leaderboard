import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEEPSWE_DIR = ROOT / "data" / "taiwan" / "deepswe"
TASKS_ROOT = ROOT / "external" / "deep-swe" / "tasks"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_deepswe_essential_subsets_are_manifested_and_known_tasks():
    manifest = _read_json(DEEPSWE_DIR / "manifest.json")
    known_tasks = {path.name for path in TASKS_ROOT.iterdir() if path.is_dir()}

    expected_thresholds = {
        "essential_8": 0.75,
        "essential_10": 0.65,
        "essential_16": 0.75,
        "essential_20": 0.80,
        "essential_30": 0.90,
    }
    for subset, min_lo_family_spearman in expected_thresholds.items():
        entry = manifest["subsets"][subset]
        task_names = _read_json(DEEPSWE_DIR / entry["task_names_path"])
        records = _read_jsonl(DEEPSWE_DIR / entry["metadata_jsonl_path"])

        assert entry["display_name"] == f"DeepSWE-Essential-{entry['count']}"
        assert len(task_names) == entry["count"]
        assert len(records) == entry["count"]
        assert len(set(task_names)) == len(task_names)
        assert set(task_names) <= known_tasks
        assert [record["task_name"] for record in records] == task_names
        assert all(record["selection_stats"]["public_avg_cost_usd"] > 0 for record in records)

        metrics = entry["selection_metrics"]
        assert metrics["lo_family_mean_spearman"] >= min_lo_family_spearman
        assert 0 <= metrics["lo_family_mean_mae"] <= 0.06


def test_deepswe_essential_selection_source_is_recorded():
    manifest = _read_json(DEEPSWE_DIR / "manifest.json")
    selection = manifest["essential_selection"]

    assert selection["name_prefix"] == "DeepSWE-Essential"
    assert selection["public_trials_url"].endswith("/artifacts/v1.1/trials.json")
    assert selection["public_tasks_url"].endswith("/artifacts/v1.1/tasks.json")
    assert "Spearman" in selection["method"]


def test_deepswe_budgeted_high_lang_balanced_subset_is_manifested():
    manifest = _read_json(DEEPSWE_DIR / "manifest.json")
    subset = manifest["subsets"]["budgeted_high_8_lang_balanced"]
    task_names = _read_json(DEEPSWE_DIR / subset["task_names_path"])
    records = _read_jsonl(DEEPSWE_DIR / subset["metadata_jsonl_path"])

    assert subset["display_name"] == "DeepSWE-Budgeted-High-8-Lang-Balanced"
    assert subset["count"] == 8
    assert len(task_names) == 8
    assert [record["task_name"] for record in records] == task_names
    assert subset["language_distribution"] == {"go": 3, "python": 2, "typescript": 3}
    assert {record["language"] for record in records} == {"go", "python", "typescript"}

    total_public_cost = sum(
        record["selection_stats"]["public_avg_cost_usd"] for record in records
    )
    mean_public_steps = sum(
        record["selection_stats"]["public_avg_steps"] for record in records
    ) / len(records)

    assert total_public_cost < 25.0
    assert mean_public_steps < 50.0
    assert subset["selection_metrics"]["spearman"] >= 0.90
    assert manifest["budgeted_high_selection"]["name_prefix"] == "DeepSWE-Budgeted-High"


def test_deepswe_budgeted_high_cap50_lang_balanced_subset_is_manifested():
    manifest = _read_json(DEEPSWE_DIR / "manifest.json")
    subset = manifest["subsets"]["budgeted_high_8_cap50_lang_balanced"]
    task_names = _read_json(DEEPSWE_DIR / subset["task_names_path"])
    records = _read_jsonl(DEEPSWE_DIR / subset["metadata_jsonl_path"])

    assert subset["display_name"] == "DeepSWE-Budgeted-High-8-Cap50-Lang-Balanced"
    assert subset["count"] == 8
    assert len(task_names) == 8
    assert [record["task_name"] for record in records] == task_names
    assert subset["language_distribution"] == {"go": 3, "python": 2, "typescript": 3}
    assert {record["language"] for record in records} == {"go", "python", "typescript"}

    total_public_cost = sum(
        record["selection_stats"]["public_avg_cost_usd"] for record in records
    )
    mean_public_steps = sum(
        record["selection_stats"]["public_avg_steps"] for record in records
    ) / len(records)

    assert total_public_cost < 21.0
    assert mean_public_steps < 40.0
    assert subset["selection_metrics"]["spearman"] >= 0.80


def test_deepswe_selection_summary_paths_exist():
    manifest = _read_json(DEEPSWE_DIR / "manifest.json")
    for subset_name, subset in manifest["subsets"].items():
        summary_path = subset.get("selection_summary_path")
        if not summary_path:
            continue
        assert (DEEPSWE_DIR / summary_path).resolve().exists(), subset_name


def test_deepswe_essential3_high_subset_is_manifested():
    manifest = _read_json(DEEPSWE_DIR / "manifest.json")
    subset = manifest["subsets"]["essential_anchored_high_8_essential3_cost_trimmed_lang_balanced"]
    task_names = _read_json(DEEPSWE_DIR / subset["task_names_path"])
    records = _read_jsonl(DEEPSWE_DIR / subset["metadata_jsonl_path"])

    assert (
        subset["display_name"]
        == "DeepSWE-Essential-Anchored-High-8-Essential3-Cost-Trimmed-Lang-Balanced"
    )
    assert subset["count"] == 8
    assert len(task_names) == 8
    assert [record["task_name"] for record in records] == task_names
    assert subset["language_distribution"] == {"go": 3, "python": 2, "typescript": 3}
    assert {record["language"] for record in records} == {"go", "python", "typescript"}

    assert {
        "expr-try-catch-errors",
        "psd-tools-blend-range-api",
        "true-myth-iterable-collection-combinators",
    } <= set(task_names)
    assert subset["selection_metrics"]["pearson"] >= 0.90
    assert subset["selection_metrics"]["spearman"] >= 0.90


def test_deepswe_wandb_glm52_cap_aware_high_subset_is_manifested():
    manifest = _read_json(DEEPSWE_DIR / "manifest.json")
    name = "essential_anchored_high_8_wandb_glm52_cap100_10m_lang_balanced"
    subset = manifest["subsets"][name]
    task_names = _read_json(DEEPSWE_DIR / subset["task_names_path"])
    records = _read_jsonl(DEEPSWE_DIR / subset["metadata_jsonl_path"])
    expected_task_names = [
        "go-genai-streamed-function-args",
        "etree-xml-diff-patch",
        "ytt-jsonpath-query-api",
        "mnamer-daemon-watch-lifecycle",
        "langchain-request-coalescing",
        "ofetch-per-origin-circuit-breaker",
        "kea-atomic-signal-selectors",
        "happy-dom-deterministic-intersectionobserver",
    ]

    assert (
        manifest["agentic_swe_assorted_default_high_subset"]
        == "essential_anchored_high_10_model_fidelity_cost_balanced"
    )
    assert manifest["agentic_swe_assorted_default_high_status"]["status"] == "frozen"
    assert (
        subset["display_name"]
        == "DeepSWE-Essential-Anchored-High-8-WandB-GLM52-Cap100-10M-Lang-Balanced"
    )
    assert subset["status"] == "historical_frozen"
    assert subset["default_for_agentic_swe_assorted"] is False
    assert subset["count"] == 8
    assert task_names == expected_task_names
    assert len(task_names) == 8
    assert [record["task_name"] for record in records] == task_names
    assert subset["language_distribution"] == {"go": 3, "python": 2, "typescript": 3}
    assert subset["selection_constraints"]["budget_model"] == "glm-5-2"
    assert subset["selection_constraints"]["budget_effort"] == "max"
    assert subset["selection_constraints"]["max_model_avg_steps"] == 100.0
    assert subset["selection_constraints"]["max_model_avg_input_tokens"] == 10_000_000.0

    for record in records:
        stats = record["selection_stats"]
        assert stats["public_budget_model_avg_steps"] <= 100.0
        assert stats["public_budget_model_avg_input_tokens"] <= 10_000_000.0
    assert subset["selection_metrics"]["pearson"] >= 0.90
    assert subset["selection_metrics"]["spearman"] >= 0.90


def test_deepswe_high10_is_the_frozen_agentic_swe_default():
    manifest = _read_json(DEEPSWE_DIR / "manifest.json")
    name = "essential_anchored_high_10_model_fidelity_cost_balanced"
    subset = manifest["subsets"][name]
    task_names = _read_json(DEEPSWE_DIR / subset["task_names_path"])
    records = _read_jsonl(DEEPSWE_DIR / subset["metadata_jsonl_path"])

    assert manifest["agentic_swe_assorted_default_high_subset"] == name
    assert subset["status"] == "frozen_default"
    assert subset["default_for_agentic_swe_assorted"] is True
    assert subset["count"] == 10
    assert len(task_names) == 10
    assert [record["task_name"] for record in records] == task_names
    assert subset["language_distribution"] == {
        "go": 4,
        "python": 3,
        "typescript": 3,
    }
    assert {
        "participle-grammar-conflict-analysis",
        "bandit-incremental-cache-control",
    } <= set(task_names)
    assert subset["selection_metrics"]["pearson"] >= 0.94
    assert subset["selection_metrics"]["spearman"] >= 0.93
    assert (
        subset["selection_metrics"]["selection_aware_leave_one_model_family_out"][
            "spearman"
        ]
        >= 0.89
    )
