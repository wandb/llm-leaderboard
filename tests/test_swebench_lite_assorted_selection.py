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
    assert manifest["subsets"]["low_v3_20"]["count"] == 20
    assert manifest["subsets"]["middle_v3_20"]["count"] == 20
    assert manifest["subsets"]["low_middle_v3_40"]["count"] == 40
    expected_plan_fields = {
        "low": 20,
        "middle": 20,
        "high": 10,
        "low_middle_jsonl_path": "subsets/low_middle_v3_40.jsonl",
        "high_subset": (
            "data/taiwan/deepswe/subsets/"
            "essential_anchored_high_10_model_fidelity_cost_balanced.jsonl"
        ),
    }
    for key, value in expected_plan_fields.items():
        assert manifest["assorted_50_plan"][key] == value
    assert manifest["assorted_50_plan"]["score_weights"] == {
        "low": 1 / 3,
        "middle": 1 / 3,
        "high": 1 / 3,
    }
    assert "F2P^2 * P2P" in manifest["assorted_50_plan"]["score_definition"]
    assert manifest["assorted_80_plan"]["status"] == "historical_v2"


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
    assert max(Counter(row["repo"] for row in middle).values()) <= 7
    assert "sympy__sympy-14817" not in middle_ids
    assert max(len(row.get("pass_to_pass") or []) for row in middle) <= 150
    assert not any(row.get("pilot_glm52_lm12m12_runtime_budget_exceeded") for row in low)
    assert not any(row.get("pilot_glm52_lm12m12_runtime_budget_exceeded") for row in middle)


def test_swebench_lite_v3_preserves_tier_difficulty_with_repo_cap_four():
    low_v2 = _read_jsonl(ASSORTED_DIR / "subsets" / "low_v2_36.jsonl")
    middle_v2 = _read_jsonl(ASSORTED_DIR / "subsets" / "middle_v2_36.jsonl")
    low = _read_jsonl(ASSORTED_DIR / "subsets" / "low_v3_20.jsonl")
    middle = _read_jsonl(ASSORTED_DIR / "subsets" / "middle_v3_20.jsonl")
    combined = _read_jsonl(ASSORTED_DIR / "subsets" / "low_middle_v3_40.jsonl")

    assert len(low) == len(middle) == 20
    assert combined == low + middle
    assert {row["source_subset"] for row in low} == {"low_v3_20"}
    assert {row["source_subset"] for row in middle} == {"middle_v3_20"}
    assert max(Counter(row["repo"] for row in low).values()) <= 4
    assert max(Counter(row["repo"] for row in middle).values()) <= 4

    for source, selected in ((low_v2, low), (middle_v2, middle)):
        source_mean = sum(row["public_lite_resolve_rate"] for row in source) / len(source)
        selected_mean = sum(row["public_lite_resolve_rate"] for row in selected) / len(
            selected
        )
        assert abs(source_mean - selected_mean) < 0.001
