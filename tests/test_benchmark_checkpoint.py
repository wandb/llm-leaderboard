from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


from evaluator.evaluate_utils.benchmark_checkpoint import BenchmarkCheckpointStore


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
