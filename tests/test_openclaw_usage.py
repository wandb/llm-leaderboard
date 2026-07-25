import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = REPO_ROOT / "scripts" / "tools" / "openclaw_usage.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_aggregate_openclaw_usage_handles_sdk_shapes_and_cost_once():
    module = load_module()

    usage = module.aggregate_openclaw_usage(
        [
            {
                "input": 10,
                "output": 2,
                "cacheRead": 30,
                "reasoningTokens": 1,
                "totalTokens": 42,
                "cost": {"input": 0.1, "output": 0.2, "total": 0.3},
            },
            {
                "inputTokens": 11,
                "outputTokens": 3,
                "cacheReadInputTokens": 31,
                "cacheWriteInputTokens": 4,
                "costUsd": 0.4,
            },
        ]
    )

    assert usage == {
        "inputTokens": 21.0,
        "outputTokens": 5.0,
        "cacheReadInputTokens": 61.0,
        "cacheWriteInputTokens": 4.0,
        "reasoningTokens": 1.0,
        "totalTokens": 91.0,
        "costUsd": 0.7,
    }


def test_attach_billable_usage_keeps_final_usage_separate_from_retries():
    module = load_module()
    final_usage = {"inputTokens": 20, "outputTokens": 4}
    record = {"openclaw_usage": final_usage}

    updated = module.attach_billable_openclaw_usage(
        record,
        [
            {
                "attempt_id": "a1",
                "attempt_number": 1,
                "returncode": 1,
                "wall_clock_seconds": 2.5,
                "usage": {"inputTokens": 10, "outputTokens": 2},
            },
            {
                "attempt_id": "a2",
                "attempt_number": 2,
                "returncode": 0,
                "wall_clock_seconds": 3.5,
                "usage": final_usage,
            },
        ],
    )

    assert updated["openclaw_usage"] is final_usage
    assert updated["billable_openclaw_attempt_count"] == 2
    assert updated["billable_openclaw_wall_seconds"] == 6.0
    assert updated["billable_openclaw_usage"]["inputTokens"] == 30
    assert updated["billable_openclaw_usage"]["outputTokens"] == 6


def test_summarize_billable_records_reports_retries_and_cost():
    module = load_module()

    summary = module.summarize_billable_openclaw_records(
        [
            {
                "billable_openclaw_attempt_count": 2,
                "billable_openclaw_wall_seconds": 10,
                "billable_openclaw_usage": {
                    "inputTokens": 100,
                    "outputTokens": 20,
                    "costUsd": 0.25,
                },
            },
            {
                "billable_openclaw_attempt_count": 1,
                "billable_openclaw_wall_seconds": 4,
                "billable_openclaw_usage": {
                    "inputTokens": 30,
                    "outputTokens": 5,
                    "costUsd": 0.05,
                },
            },
        ]
    )

    assert summary["attempt_count"] == 3
    assert summary["retry_count"] == 1
    assert summary["wall_seconds"] == 14
    assert summary["usage"]["inputTokens"] == 130
    assert summary["usage"]["costUsd"] == 0.3


def test_merge_prior_billable_usage_preserves_provider_recovery_cost():
    module = load_module()
    prior = module.attach_billable_openclaw_usage(
        {"openclaw_usage": {"inputTokens": 10, "outputTokens": 2}},
        [
            {
                "attempt_id": "provider-failed",
                "attempt_number": 1,
                "returncode": 1,
                "wall_clock_seconds": 4,
                "usage": {"inputTokens": 10, "outputTokens": 2},
            }
        ],
    )
    resumed = module.attach_billable_openclaw_usage(
        {"openclaw_usage": {"inputTokens": 20, "outputTokens": 3}},
        [
            {
                "attempt_id": "provider-recovered",
                "attempt_number": 1,
                "returncode": 0,
                "wall_clock_seconds": 6,
                "usage": {"inputTokens": 20, "outputTokens": 3},
            }
        ],
    )

    merged = module.merge_prior_billable_openclaw_usage(resumed, prior)

    assert merged["openclaw_usage"] == {"inputTokens": 20, "outputTokens": 3}
    assert merged["billable_openclaw_attempt_count"] == 2
    assert merged["billable_openclaw_wall_seconds"] == 10
    assert merged["billable_openclaw_usage"]["inputTokens"] == 30
    assert merged["billable_openclaw_usage"]["outputTokens"] == 5
    assert [
        attempt["attempt_id"] for attempt in merged["billable_openclaw_attempts"]
    ] == ["provider-failed", "provider-recovered"]
