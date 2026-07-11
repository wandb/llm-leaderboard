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


def test_agentic_swe_assorted_low_middle_limit_keeps_legacy_order():
    module = load_module(SCRIPT)
    rows = [
        {"instance_id": "low-1", "agentic_swe_tier": "low"},
        {"instance_id": "low-2", "agentic_swe_tier": "low"},
        {"instance_id": "middle-1", "agentic_swe_tier": "middle"},
    ]
    args = SimpleNamespace(low_middle_limit=2, low_limit=None, middle_limit=None)

    selected = module.selected_low_middle_rows(rows, args)

    assert [row["instance_id"] for row in selected] == ["low-1", "low-2"]


def test_agentic_swe_assorted_can_select_balanced_low_middle_subset():
    module = load_module(SCRIPT)
    rows = [
        {"instance_id": "low-1", "agentic_swe_tier": "low"},
        {"instance_id": "low-2", "agentic_swe_tier": "low"},
        {"instance_id": "low-3", "agentic_swe_tier": "low"},
        {"instance_id": "middle-1", "agentic_swe_tier": "middle"},
        {"instance_id": "middle-2", "agentic_swe_tier": "middle"},
        {"instance_id": "middle-3", "agentic_swe_tier": "middle"},
    ]
    args = SimpleNamespace(low_middle_limit=None, low_limit=2, middle_limit=2)

    selected = module.selected_low_middle_rows(rows, args)

    assert [row["instance_id"] for row in selected] == [
        "low-1",
        "middle-1",
        "low-2",
        "middle-2",
    ]


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


def test_agentic_swe_assorted_summary_estimates_model_cost():
    module = load_module(SCRIPT)
    args = SimpleNamespace(
        model="openai-direct/gpt-4.1-mini-2025-04-14",
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
            "agentic_swe_tier": "low",
            "resolved": False,
            "patch_empty": True,
            "weave_agents_ok": True,
            "openclaw_usage": {
                "inputTokens": 1_000_000,
                "outputTokens": 100_000,
                "cacheReadInputTokens": 2_000_000,
            },
        }
    ]

    summary = module.build_summary(rows, args=args, elapsed=1.0)

    assert summary["total"]["usage"]["cost_usd"] == 0.76


def test_deepswe_usage_falls_back_to_weave_agents_trace_usage(tmp_path):
    module = load_module(SCRIPT)
    verifier = tmp_path / "weave_agents.json"
    verifier.write_text(
        """{
  "checks": [
    {
      "name": "usage",
      "ok": true,
      "agent_input_tokens": 10,
      "agent_output_tokens": 2,
      "trace_input_tokens": 1000,
      "trace_output_tokens": 50
    }
  ]
}
""",
        encoding="utf-8",
    )
    rows = module.deepswe_rows(
        metadata_rows=[
            {
                "task_name": "example-task",
                "repository": "example/repo",
                "subset": "essential_8",
            }
        ],
        result_rows=[
            {
                "task_name": "datacurve/example-task",
                "resolved": False,
                "score": 0.0,
                "openclaw_usage": {},
                "weave_agents_verifier_json": str(verifier),
            }
        ],
    )

    assert rows[0]["openclaw_usage"] == {
        "inputTokens": 1010,
        "outputTokens": 52,
        "cacheReadInputTokens": 0,
        "usageSource": "weave_agents_trace",
        "usageApproximate": True,
    }


def test_lite_usage_falls_back_to_weave_agents_trace_usage(tmp_path):
    module = load_module(SCRIPT)
    verifier = tmp_path / "weave_agents.json"
    verifier.write_text(
        """{
  "checks": [
    {
      "name": "usage",
      "ok": true,
      "trace_input_tokens": 1234,
      "trace_output_tokens": 56
    }
  ]
}
""",
        encoding="utf-8",
    )
    rows = module.lite_rows(
        source_rows=[
            {
                "instance_id": "django__django-1",
                "repo": "django/django",
                "agentic_swe_tier": "low",
            }
        ],
        patch_rows=[
            {
                "instance_id": "django__django-1",
                "patch": "",
                "openclaw_usage": {},
                "weave_agents_verifier_json": str(verifier),
            }
        ],
        eval_results={"django__django-1": False},
    )

    assert rows[0]["openclaw_usage"] == {
        "inputTokens": 1234,
        "outputTokens": 56,
        "cacheReadInputTokens": 0,
        "usageSource": "weave_agents_trace",
        "usageApproximate": True,
    }
