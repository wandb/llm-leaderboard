from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


from evaluator.evaluate_utils.benchmark_checkpoint import (
    BenchmarkCheckpointStore,
    aggregate_requires_refresh,
    bfcl_completion_evidence_is_clean,
    classify_benchmark_failure,
)


def test_bfcl_completion_requires_zero_infrastructure_errors():
    assert bfcl_completion_evidence_is_clean(
        {"bfcl_inference_error_count": 0}
    )
    assert not bfcl_completion_evidence_is_clean(
        {"bfcl_inference_error_count": 7}
    )
    assert not bfcl_completion_evidence_is_clean({})


def test_benchmark_failure_classification_is_conservative():
    assert classify_benchmark_failure(
        "BFCLInfrastructureError",
        "provider read timed out",
    )["infrastructure_retryable"]
    assert classify_benchmark_failure(
        "RuntimeError",
        "child failed with ProviderRecoveryExhaustedError: 504",
    )["infrastructure_retryable"]
    assert classify_benchmark_failure(
        "RuntimeError",
        "ProviderRecoveryExhaustedError after a task mentioned token budget",
    )["infrastructure_retryable"]
    assert not classify_benchmark_failure(
        "TimeoutError",
        "task exceeded benchmark wall timeout",
    )["infrastructure_retryable"]
    assert not classify_benchmark_failure(
        "RateLimitError",
        "insufficient_quota: billing limit reached",
    )["infrastructure_retryable"]
    assert not classify_benchmark_failure(
        "RuntimeError",
        "unknown failure",
    )["infrastructure_retryable"]


def test_aggregate_refreshes_after_upstream_execution_only():
    assert aggregate_requires_refresh("aggregate_taiwan", {"bfcl"})
    assert aggregate_requires_refresh("aggregate", {"agentic_math"})
    assert not aggregate_requires_refresh("aggregate", set())
    assert not aggregate_requires_refresh(
        "aggregate_taiwan",
        {"aggregate"},
    )
    assert not aggregate_requires_refresh("bfcl", {"agentic_math"})


def test_completed_benchmark_is_skipped_only_for_explicit_resume(tmp_path):
    writer = BenchmarkCheckpointStore(
        tmp_path,
        run_id="run-1",
        resume_enabled=True,
    )
    writer.mark_completed("mtbench")

    assert writer.is_completed("mtbench") is True
    assert (
        BenchmarkCheckpointStore(
            tmp_path,
            run_id="run-1",
            resume_enabled=False,
        ).is_completed("mtbench")
        is False
    )
    assert (
        BenchmarkCheckpointStore(
            tmp_path,
            run_id="run-2",
            resume_enabled=True,
        ).is_completed("mtbench")
        is False
    )


def test_failure_replaces_started_state_and_is_not_skipped(tmp_path):
    store = BenchmarkCheckpointStore(
        tmp_path,
        run_id="run-1",
        resume_enabled=True,
    )
    store.mark_started("agentic_swe_assorted")
    store.mark_failed("agentic_swe_assorted", RuntimeError("grading failed"))

    payload = store.load("agentic_swe_assorted")
    assert payload["status"] == "failed"
    assert payload["error_type"] == "RuntimeError"
    assert payload["failure_category"] == "unknown"
    assert payload["infrastructure_retryable"] is False
    assert store.is_completed("agentic_swe_assorted") is False


def test_completed_checkpoint_is_invalidated_when_config_changes(tmp_path):
    writer = BenchmarkCheckpointStore(
        tmp_path,
        run_id="run-1",
        resume_enabled=True,
        config_fingerprints={"mtbench": "config-a"},
    )
    writer.mark_completed("mtbench")

    assert writer.is_completed("mtbench") is True
    changed = BenchmarkCheckpointStore(
        tmp_path,
        run_id="run-1",
        resume_enabled=True,
        config_fingerprints={"mtbench": "config-b"},
    )
    assert changed.is_completed("mtbench") is False


def test_legacy_checkpoint_is_available_only_for_explicit_migration(tmp_path):
    legacy = BenchmarkCheckpointStore(
        tmp_path,
        run_id="run-1",
        resume_enabled=True,
    )
    legacy.mark_completed("mtbench")

    fingerprinted = BenchmarkCheckpointStore(
        tmp_path,
        run_id="run-1",
        resume_enabled=True,
        config_fingerprints={"mtbench": "config-a"},
    )

    assert fingerprinted.is_completed("mtbench") is False
    assert fingerprinted.load("mtbench") is None
    assert fingerprinted.load_raw("mtbench")["status"] == "completed"


def test_interrupted_rerun_preserves_matching_completed_snapshot(tmp_path):
    store = BenchmarkCheckpointStore(
        tmp_path,
        run_id="run-1",
        resume_enabled=True,
        config_fingerprints={"agentic_math": "config-a"},
        code_fingerprints={"agentic_math": "code-a"},
    )
    store.mark_completed("agentic_math")
    store.mark_started("agentic_math")
    store.mark_failed("agentic_math", SystemExit(130))

    payload = store.load_raw("agentic_math")
    assert payload["status"] == "failed"
    assert payload["last_completed"]["config_fingerprint"] == "config-a"
    assert payload["last_completed"]["code_fingerprint"] == "code-a"
    assert store.completed_snapshot_matches_config("agentic_math") is True
    assert (
        store.can_restore_completed_snapshot(
            "agentic_math",
            remote_completed=False,
        )
        is False
    )
    assert (
        store.can_restore_completed_snapshot(
            "agentic_math",
            remote_completed=True,
        )
        is True
    )


def test_completed_checkpoint_allows_code_drift_without_config_drift(tmp_path):
    writer = BenchmarkCheckpointStore(
        tmp_path,
        run_id="run-1",
        resume_enabled=True,
        config_fingerprints={"agentic_math": "config-a"},
        code_fingerprints={"agentic_math": "code-a"},
    )
    writer.mark_completed("agentic_math")

    reader = BenchmarkCheckpointStore(
        tmp_path,
        run_id="run-1",
        resume_enabled=True,
        config_fingerprints={"agentic_math": "config-a"},
        code_fingerprints={"agentic_math": "code-b"},
    )
    assert reader.is_completed("agentic_math") is True
    assert reader.code_drifted("agentic_math") is True


def test_completed_snapshot_does_not_match_changed_config(tmp_path):
    writer = BenchmarkCheckpointStore(
        tmp_path,
        run_id="run-1",
        resume_enabled=True,
        config_fingerprints={"agentic_math": "config-a"},
    )
    writer.mark_completed("agentic_math")
    writer.mark_failed("agentic_math", RuntimeError("interrupted"))

    reader = BenchmarkCheckpointStore(
        tmp_path,
        run_id="run-1",
        resume_enabled=True,
        config_fingerprints={"agentic_math": "config-b"},
    )
    assert reader.completed_snapshot_matches_config("agentic_math") is False
