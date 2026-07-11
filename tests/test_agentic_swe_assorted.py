import importlib.util
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tools" / "run_agentic_swe_assorted.py"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_agentic_swe_assorted_keeps_tier_and_source_fields():
    module = load_module(SCRIPT)
    rows = module.lite_rows(
        source_rows=[
            {
                "instance_id": "django__django-1",
                "repo": "django/django",
                "agentic_swe_tier": "low",
                "source_benchmark": "SWE-bench Lite",
                "source_dataset": "princeton-nlp/SWE-bench_Lite",
                "source_subset": "low_36",
                "source_instance_id": "django__django-1",
            }
        ],
        patch_rows=[
            {
                "instance_id": "django__django-1",
                "patch": "diff --git a/a.py b/a.py\n",
                "openclaw_tool_call_count": 3,
                "openclaw_usage": {"inputTokens": 100, "outputTokens": 20},
            }
        ],
        eval_results={"django__django-1": True},
    )

    module.validate_output_rows(rows)
    [row] = rows
    assert row["agentic_swe_tier"] == "low"
    assert row["source_benchmark"] == "SWE-bench Lite"
    assert row["source_dataset"] == "princeton-nlp/SWE-bench_Lite"
    assert row["source_subset"] == "low_36"
    assert row["source_instance_id"] == "django__django-1"
    assert row["resolved"] is True


def test_agentic_swe_assorted_summary_groups_by_tier_and_source():
    module = load_module(SCRIPT)
    args = SimpleNamespace(
        model="dummy/local",
        dry_run=False,
        max_input_tokens=1_000_000,
        max_cumulative_input_tokens=1_000_000,
        max_cumulative_output_tokens=500_000,
        max_tool_calls=40,
        max_agent_turns=40,
        max_tool_wall_seconds=120,
        swe_workers=4,
        high_workers=2,
    )
    rows = [
        {
            "source_benchmark": "SWE-bench Lite",
            "source_dataset": "princeton-nlp/SWE-bench_Lite",
            "source_subset": "low_36",
            "source_instance_id": "a",
            "agentic_swe_tier": "low",
            "instance_id": "a",
            "resolved": True,
            "score": 1.0,
            "weave_agents_conversation_url": "https://example.com/a",
            "openclaw_tool_call_count": 2,
            "openclaw_usage": {"inputTokens": 100, "outputTokens": 50},
        },
        {
            "source_benchmark": "DeepSWE",
            "source_dataset": "DataCurve DeepSWE v1.1",
            "source_subset": "essential_8",
            "source_instance_id": "b",
            "agentic_swe_tier": "high",
            "instance_id": "b",
            "resolved": False,
            "score": 0.0,
            "weave_agents_conversation_url": "https://example.com/b",
            "openclaw_tool_call_count": 40,
            "openclaw_usage": {"inputTokens": 200, "outputTokens": 100},
        },
    ]

    summary = module.build_summary(rows, args=args, elapsed=12.5)

    assert summary["total"]["total_instances"] == 2
    assert summary["total"]["resolved_instances"] == 1
    assert summary["by_tier"]["low"]["pass_at_1"] == 1.0
    assert summary["by_tier"]["high"]["pass_at_1"] == 0.0
    assert summary["by_source"]["SWE-bench Lite"]["total_instances"] == 1
    assert summary["total"]["usage"]["input_tokens"] == 300
    assert summary["total"]["usage"]["output_tokens"] == 150


def test_agentic_swe_assorted_usage_tracks_cache_tokens_separately():
    module = load_module(SCRIPT)

    usage = module.usage_numbers(
        {
            "inputTokens": 100,
            "outputTokens": 20,
            "cacheReadInputTokens": 300,
            "cacheWriteInputTokens": 40,
        }
    )

    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 20
    assert usage["cache_read_input_tokens"] == 300
    assert usage["cache_write_input_tokens"] == 40
    assert usage["total_tokens"] == 460
