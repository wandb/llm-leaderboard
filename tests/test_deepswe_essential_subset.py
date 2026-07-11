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
