import importlib.util
import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tools" / "verify_wandb_scope_attestation.py"


def load_module():
    spec = importlib.util.spec_from_file_location(SCRIPT.stem, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[SCRIPT.stem] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def completion_payload():
    return {
        "ok": True,
        "benchmark": "agentic_math",
        "entity": "test-entity",
        "project": "test-project",
        "run_id": "run-1",
        "query_source": {
            "kind": "wandb_sdk",
            "api": "wandb.Api",
            "timeout_seconds": 60,
            "entity": "test-entity",
            "project": "test-project",
            "run_id": "run-1",
            "run_path": "test-entity/test-project/run-1",
            "benchmark": "agentic_math",
            "summary_source": "run.summary_metrics",
            "artifact_source": "run.logged_artifacts",
            "history_scanned": False,
        },
        "generated_at": 1,
        "verification_schema_version": 1,
        "required_evidence": {
            "expected_total": 1,
            "nemoclaw_session_audit": {"required": True},
            "run_metadata": {
                "config": [
                    {
                        "key": "model.pretrained_model_name_or_path",
                        "expected": "gpt-4.1-mini-2025-04-14",
                    }
                ],
                "tags": ["taiwan-canary"],
                "job_type": "evaluation",
            }
        },
        "observed_evidence": {
            "run_state": "finished",
            "summary_metrics": {"agentic_math/accuracy": {"ok": True, "value": 0.86}},
            "nemoclaw_session_audit": {
                "ok": True,
                "required": 1,
                "passed": 1,
                "failed": 0,
                "expected_total": 1,
            },
            "tables": [
                {
                    "name": "agentic_math_output_table",
                    "columns_ok": True,
                    "missing_columns": [],
                    "required_columns": [
                        "nemoclaw_session_copy_source",
                        "nemoclaw_session_copied_bytes",
                    ],
                    "row_observability_ok": True,
                    "row_observability_invalid_row_count": 0,
                    "row_observability_checked_rows": 1,
                    "row_observability_expected_rows": 1,
                    "row_observability_required_copy_source_columns": [
                        "nemoclaw_session_copy_source"
                    ],
                    "row_observability_required_positive_int_columns": [
                        "nemoclaw_session_copied_bytes"
                    ],
                    "row_observability_allowed_copy_sources": [
                        "stdout_agent_meta",
                        "live_runtime_budget",
                    ],
                }
            ],
            "run_metadata": {
                "config": [
                    {
                        "key": "model.pretrained_model_name_or_path",
                        "present": True,
                        "value": "gpt-4.1-mini-2025-04-14",
                    }
                ],
                "tags": ["taiwan-canary"],
                "job_type": "evaluation",
            },
        },
    }


def review_payload():
    return {
        "status": "completed",
        "phase": "agentic",
        "verify_wandb_completion": False,
        "runs": [{"wandb_run_id": "run-1", "returncode": 0}],
    }


def attestation_payload(review: Path, completion: Path):
    return {
        "schema_version": 1,
        "confirmed": True,
        "confirmed_by": "yuya",
        "confirmed_at": "2026-06-28T00:45:00+09:00",
        "confirmation": "This W&B run is the reviewed Agentic Math canary scope.",
        "review_path": str(review),
        "completion_path": str(completion),
        "completion_sha256": sha256(completion),
        "benchmark": "agentic_math",
        "entity": "test-entity",
        "project": "test-project",
        "run_id": "run-1",
        "actual_cost_estimate": "$12.34",
        "provider_bill_reference": "openai-dashboard-2026-06-28",
    }


def test_verify_scope_attestation_passes_with_exact_completion_binding(tmp_path):
    module = load_module()
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "completion.json", completion_payload())
    attestation = write_json(
        tmp_path / "attestation.json",
        attestation_payload(review, completion),
    )

    report = module.verify_scope_attestation(
        review_path=review,
        completion_path=completion,
        attestation_path=attestation,
        require_query_source=True,
    )

    assert report["ok"] is True
    assert report["status"] == "passed"
    assert report["entry"]["benchmark"] == "agentic_math"
    assert report["entry"]["sha256"] == sha256(completion)
    assert report["scope_attestation"]["completion_sha256"] == sha256(completion)
    assert report["source_files"]["review_json"]["sha256"] == sha256(review)
    assert report["source_files"]["completion_json"]["sha256"] == sha256(completion)
    assert report["source_files"]["scope_attestation_json"]["sha256"] == sha256(attestation)
    assert report["source_files"]["review_json"]["readable"] is True
    assert report["errors"] == []


def test_cli_writes_failure_report_for_unconfirmed_template(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "completion.json", completion_payload())
    payload = attestation_payload(review, completion)
    payload["confirmed"] = False
    payload["actual_cost_estimate"] = "$ACTUAL_OR_BILLING_ESTIMATE"
    attestation = write_json(tmp_path / "attestation.json", payload)
    report_json = tmp_path / "preflight.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--scope-attestation-json",
            str(attestation),
            "--json",
            str(report_json),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["status"] == "validation_failed"
    assert report["source_files"]["review_json"]["sha256"] == sha256(review)
    assert report["source_files"]["completion_json"]["sha256"] == sha256(completion)
    assert report["source_files"]["scope_attestation_json"]["sha256"] == sha256(attestation)
    assert "confirmed must be true" in report["errors"][0]
    assert "actual_cost_estimate must not be a placeholder" in report["errors"][0]
