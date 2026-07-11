import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSORTED_DIR = ROOT / "data" / "taiwan" / "swebench_lite_assorted"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_swebench_lite_low_middle_subsets_are_manifested():
    manifest = _read_json(ASSORTED_DIR / "manifest.json")

    assert manifest["source_benchmark"] == "SWE-bench Lite"
    assert manifest["source_dataset"] == "princeton-nlp/SWE-bench_Lite"
    assert manifest["source_split"] == "test"
    assert manifest["subsets"]["low_36"]["count"] == 36
    assert manifest["subsets"]["middle_36"]["count"] == 36
    assert manifest["subsets"]["low_middle_72"]["count"] == 72
    assert manifest["subsets"]["low_v2_36"]["count"] == 36
    assert manifest["subsets"]["middle_v2_36"]["count"] == 36
    assert manifest["subsets"]["low_middle_v2_72"]["count"] == 72
    assert manifest["assorted_80_plan"] == {
        "low": 36,
        "middle": 36,
        "high": 8,
        "low_middle_jsonl_path": "subsets/low_middle_v2_72.jsonl",
        "legacy_low_middle_jsonl_path": "subsets/low_middle_72.jsonl",
        "high_subset": "data/taiwan/deepswe/subsets/essential_8.jsonl",
    }


def test_swebench_lite_low_middle_rows_have_source_and_runner_fields():
    required_fields = {
        "instance_id",
        "repo",
        "base_commit",
        "problem_statement",
        "patch",
        "test_patch",
        "benchmark_id",
        "benchmark_name",
        "agentic_swe_tier",
        "source_benchmark",
        "source_dataset",
        "source_config",
        "source_split",
        "source_subset",
        "source_instance_id",
        "selected_test_files_to_run",
        "static_difficulty_score",
        "static_difficulty_percentile",
        "public_lite_seen_count",
        "public_lite_resolved_count",
        "public_lite_resolve_rate",
    }

    rows = _read_jsonl(ASSORTED_DIR / "subsets" / "low_middle_v2_72.jsonl")

    assert len(rows) == 72
    assert all(required_fields <= row.keys() for row in rows)
    assert all(row["benchmark_id"] == "swebench_lite" for row in rows)
    assert all(row["source_benchmark"] == "SWE-bench Lite" for row in rows)
    assert all(row["source_dataset"] == "princeton-nlp/SWE-bench_Lite" for row in rows)
    assert all(row["source_instance_id"] == row["instance_id"] for row in rows)
    assert {row["agentic_swe_tier"] for row in rows} == {"low", "middle"}
    assert {row["source_subset"] for row in rows} == {"low_v2_36", "middle_v2_36"}
    assert all(row["public_lite_seen_count"] >= 20 for row in rows)


def test_swebench_lite_low_middle_difficulty_and_overlap():
    low = _read_jsonl(ASSORTED_DIR / "subsets" / "low_36.jsonl")
    middle = _read_jsonl(ASSORTED_DIR / "subsets" / "middle_36.jsonl")

    low_ids = {row["instance_id"] for row in low}
    middle_ids = {row["instance_id"] for row in middle}
    low_mean = sum(row["static_difficulty_score"] for row in low) / len(low)
    middle_mean = sum(row["static_difficulty_score"] for row in middle) / len(middle)

    assert len(low_ids) == 36
    assert len(middle_ids) == 36
    assert low_ids.isdisjoint(middle_ids)
    assert low_mean < middle_mean
    assert max(Counter(row["repo"] for row in low).values()) <= 6
    assert max(Counter(row["repo"] for row in middle).values()) <= 6


def test_swebench_lite_v2_uses_public_prior_and_excludes_pilot_cost_risk():
    low = _read_jsonl(ASSORTED_DIR / "subsets" / "low_v2_36.jsonl")
    middle = _read_jsonl(ASSORTED_DIR / "subsets" / "middle_v2_36.jsonl")

    low_ids = {row["instance_id"] for row in low}
    middle_ids = {row["instance_id"] for row in middle}
    low_rates = [row["public_lite_resolve_rate"] for row in low]
    middle_rates = [row["public_lite_resolve_rate"] for row in middle]

    assert len(low_ids) == 36
    assert len(middle_ids) == 36
    assert low_ids.isdisjoint(middle_ids)
    assert min(low_rates) >= 0.65
    assert max(middle_rates) <= 0.72
    assert sum(low_rates) / len(low_rates) > sum(middle_rates) / len(middle_rates)
    assert max(Counter(row["repo"] for row in low).values()) <= 6
    assert max(Counter(row["repo"] for row in middle).values()) <= 6
    assert not any(row.get("pilot_glm52_lm12m12_runtime_budget_exceeded") for row in low)
    assert not any(row.get("pilot_glm52_lm12m12_runtime_budget_exceeded") for row in middle)
