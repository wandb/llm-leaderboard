import importlib.util
import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tools" / "sync_wandb_completion_to_paid_review.py"


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


def add_wandb_run_metadata(payload, *, benchmark="agentic_math", run_id="run-1"):
    required = payload.setdefault("required_evidence", {})
    required["run_metadata"] = {
        "config": [
            {"key": "model.pretrained_model_name_or_path", "expected": "gpt-4.1-mini-2025-04-14"},
            {"key": f"run.{benchmark}", "expected": True},
            {"key": "wandb.run_name", "expected": f"taiwan/full/openai/gpt-4.1-mini: {run_id}"},
        ],
        "tags": ["taiwan-canary"],
        "group": "tw-canary",
        "job_type": "evaluation",
    }
    observed = payload.setdefault("observed_evidence", {})
    observed["run_metadata"] = {
        "config": [
            {
                "key": "model.pretrained_model_name_or_path",
                "present": True,
                "value": "gpt-4.1-mini-2025-04-14",
            },
            {"key": f"run.{benchmark}", "present": True, "value": True},
            {
                "key": "wandb.run_name",
                "present": True,
                "value": f"taiwan/full/openai/gpt-4.1-mini: {run_id}",
            },
        ],
        "tags": ["taiwan-canary"],
        "group": "tw-canary",
        "job_type": "evaluation",
    }
    return payload


def completion_payload(*, benchmark="agentic_math", run_id="run-1", ok=True):
    expected_total = 100 if benchmark == "agentic_math" else 80 if benchmark == "agentic_swe" else None
    payload = {
        "ok": ok,
        "benchmark": benchmark,
        "entity": "test-entity",
        "project": "test-project",
        "run_id": run_id,
        "query_source": {
            "kind": "wandb_sdk",
            "api": "wandb.Api",
            "timeout_seconds": 60,
            "entity": "test-entity",
            "project": "test-project",
            "run_id": run_id,
            "run_path": f"test-entity/test-project/{run_id}",
            "benchmark": benchmark,
            "summary_source": "run.summary_metrics",
            "artifact_source": "run.logged_artifacts",
            "history_scanned": False,
        },
        "generated_at": 1,
        "verification_schema_version": 1,
        "observed_evidence": {
            "run_state": "finished",
            "summary_metrics": {
                "agentic_math/accuracy": {"ok": True, "value": 0.86},
            },
        },
        "checks": [],
    }
    if expected_total is not None:
        payload["required_evidence"] = {
            "expected_total": expected_total,
            "nemoclaw_session_audit": {
                "required": True,
                "required_metric": f"{benchmark}/nemoclaw_session_audit_required_instances",
                "passed_metric": f"{benchmark}/nemoclaw_session_audit_passed_instances",
                "failed_metric": f"{benchmark}/nemoclaw_session_audit_failed_instances",
            },
        }
        payload["observed_evidence"]["nemoclaw_session_audit"] = {
            "ok": True,
            "required": expected_total,
            "passed": expected_total,
            "failed": 0,
            "expected_total": expected_total,
        }
    return add_wandb_run_metadata(payload, benchmark=benchmark, run_id=run_id)


def review_payload():
    return {
        "status": "completed",
        "phase": "agentic",
        "verify_wandb_completion": False,
        "actual_cost_estimate": "",
        "provider_bill_reference": "",
        "runs": [
            {
                "config": "config-a.yaml",
                "wandb_run_id": "run-1",
                "wandb_entity": "test-entity",
                "wandb_project": "test-project",
                "returncode": 0,
            }
        ],
    }


def source_audit_payload(completion: Path, *, run_id: str = "run-1"):
    return {
        "ok": True,
        "status": "passed",
        "formalized_records": [
            {
                "benchmark": "agentic_math",
                "model_slug": "gpt-4_1-mini",
                "model": "gpt-4.1-mini-2025-04-14",
                "run_kind": "canary",
                "result_dir": "outputs/taiwan_full_eval/agentic_math/openai/gpt-4.1-mini",
                "wandb_completion": {
                    "path": str(completion),
                    "entity": "test-entity",
                    "project": "test-project",
                    "run_id": run_id,
                    "verification_schema_version": 1,
                    "schema_current": True,
                    "observed_evidence_present": True,
                },
            }
        ],
    }


def scope_attestation_payload(
    review: Path,
    completion: Path,
    *,
    run_id: str = "run-1",
    source_audit: Path | None = None,
):
    payload = {
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
        "run_id": run_id,
        "actual_cost_estimate": "$12.34",
        "provider_bill_reference": "openai-dashboard-2026-06-28",
    }
    if source_audit is not None:
        payload["source_audit_json"] = str(source_audit)
        payload["source_audit_sha256"] = sha256(source_audit)
    return payload


def test_sync_review_adds_completion_to_matching_run():
    module = load_module()
    entry = {
        "benchmark": "agentic_math",
        "ok": True,
        "entity": "test-entity",
        "project": "test-project",
        "run_id": "run-1",
        "path": "outputs/wandb_completion/agentic_math-run-1.json",
    }

    updated, changes, unmatched = module.sync_review(
        review_payload(),
        [entry],
        top_level=False,
        replace=True,
        set_verify_wandb_completion=True,
        actual_cost_estimate="$12.34",
        provider_bill_reference="bill-1",
    )

    assert unmatched == []
    assert changes[0]["action"] == "added"
    assert updated["verify_wandb_completion"] is True
    assert updated["actual_cost_estimate"] == "$12.34"
    assert updated["provider_bill_reference"] == "bill-1"
    assert updated["runs"][0]["wandb_completion"] == [entry]


def test_sync_review_replaces_existing_completion_entry():
    module = load_module()
    review = review_payload()
    review["runs"][0]["wandb_completion"] = [
        {
            "benchmark": "agentic_math",
            "ok": False,
            "entity": "test-entity",
            "project": "test-project",
            "run_id": "run-1",
            "path": "old.json",
        }
    ]
    entry = {
        "benchmark": "agentic_math",
        "ok": True,
        "entity": "test-entity",
        "project": "test-project",
        "run_id": "run-1",
        "path": "new.json",
    }

    updated, changes, unmatched = module.sync_review(
        review,
        [entry],
        top_level=False,
        replace=True,
        set_verify_wandb_completion=False,
        actual_cost_estimate=None,
        provider_bill_reference=None,
    )

    assert unmatched == []
    assert changes[0]["action"] == "replaced"
    assert updated["runs"][0]["wandb_completion"][0]["path"] == "new.json"


def test_sync_review_keeps_same_run_id_from_different_wandb_project_separate():
    module = load_module()
    review = review_payload()
    review["runs"][0].pop("wandb_entity")
    review["runs"][0].pop("wandb_project")
    review["runs"][0]["wandb_completion"] = [
        {
            "benchmark": "agentic_math",
            "ok": True,
            "entity": "entity-a",
            "project": "project-a",
            "run_id": "run-1",
            "path": "entity-a/project-a/run-1.json",
        }
    ]
    entry = {
        "benchmark": "agentic_math",
        "ok": True,
        "entity": "entity-b",
        "project": "project-b",
        "run_id": "run-1",
        "path": "entity-b/project-b/run-1.json",
    }

    updated, changes, unmatched = module.sync_review(
        review,
        [entry],
        top_level=False,
        replace=True,
        set_verify_wandb_completion=False,
        actual_cost_estimate=None,
        provider_bill_reference=None,
    )

    assert unmatched == []
    assert changes[0]["action"] == "added"
    assert [row["path"] for row in updated["runs"][0]["wandb_completion"]] == [
        "entity-a/project-a/run-1.json",
        "entity-b/project-b/run-1.json",
    ]


def test_sync_review_rejects_run_identity_mismatch():
    module = load_module()
    review = review_payload()
    review["runs"][0]["wandb_entity"] = "entity-a"
    review["runs"][0]["wandb_project"] = "project-a"
    entry = {
        "benchmark": "agentic_math",
        "ok": True,
        "entity": "entity-b",
        "project": "project-b",
        "run_id": "run-1",
        "path": "entity-b/project-b/run-1.json",
    }

    updated, changes, unmatched = module.sync_review(
        review,
        [entry],
        top_level=False,
        replace=True,
        set_verify_wandb_completion=False,
        actual_cost_estimate=None,
        provider_bill_reference=None,
    )

    assert changes == []
    assert unmatched[0]["match_status"] == "unmatched_run_identity"
    assert "wandb_completion" not in updated["runs"][0]


def test_sync_review_rejects_ambiguous_duplicate_run_id_without_wandb_identity():
    module = load_module()
    review = review_payload()
    review["runs"] = [
        {"config": "a.yaml", "wandb_run_id": "run-1", "returncode": 0},
        {"config": "b.yaml", "wandb_run_id": "run-1", "returncode": 0},
    ]
    entry = {
        "benchmark": "agentic_math",
        "ok": True,
        "entity": "test-entity",
        "project": "test-project",
        "run_id": "run-1",
        "path": "test-entity/test-project/run-1.json",
    }

    updated, changes, unmatched = module.sync_review(
        review,
        [entry],
        top_level=False,
        replace=True,
        set_verify_wandb_completion=False,
        actual_cost_estimate=None,
        provider_bill_reference=None,
    )

    assert changes == []
    assert unmatched[0]["match_status"] == "ambiguous_run_id_without_wandb_identity"
    assert all("wandb_completion" not in run for run in updated["runs"])


def test_sync_review_can_write_top_level_completion():
    module = load_module()
    entry = {
        "benchmark": "taiwan_full",
        "ok": True,
        "run_id": "run-2",
        "path": "taiwan_full-run-2.json",
    }

    updated, changes, unmatched = module.sync_review(
        review_payload(),
        [entry],
        top_level=True,
        replace=True,
        set_verify_wandb_completion=False,
        actual_cost_estimate=None,
        provider_bill_reference=None,
    )

    assert unmatched == []
    assert changes[0]["target"] == "top_level"
    assert updated["wandb_completion"] == [entry]


def test_cli_fails_on_unmatched_run_id(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(
        tmp_path / "completion.json",
        completion_payload(run_id="other-run"),
    )
    report = tmp_path / "report.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--report-json",
            str(report),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "unmatched_run_id"
    assert payload["unmatched_count"] == 1
    assert json.loads(report.read_text(encoding="utf-8"))["ok"] is False


def test_cli_writes_updated_review_to_output(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "completion.json", completion_payload())
    output = tmp_path / "updated_review.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--output-json",
            str(output),
            "--set-verify-wandb-completion",
            "--actual-cost-estimate",
            "$12.34",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    updated = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert updated["verify_wandb_completion"] is True
    assert updated["actual_cost_estimate"] == "$12.34"
    assert updated["runs"][0]["wandb_completion"][0]["path"].endswith("completion.json")
    assert updated["runs"][0]["wandb_completion"][0]["verification_schema_version"] == 1
    assert updated["runs"][0]["wandb_completion"][0]["observed_evidence_valid"] is True
    assert updated["runs"][0]["wandb_completion"][0]["run_metadata_valid"] is True
    assert updated["runs"][0]["wandb_completion"][0]["nemoclaw_session_audit_valid"] is True
    assert updated["runs"][0]["wandb_completion"][0]["query_source_kind"] == "wandb_sdk"
    assert (
        updated["runs"][0]["wandb_completion"][0]["query_source_run_path"]
        == "test-entity/test-project/run-1"
    )


def test_cli_rejects_non_adopted_completion_without_query_source(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    payload = completion_payload()
    payload.pop("query_source")
    completion = write_json(tmp_path / "completion.json", payload)
    output = tmp_path / "updated_review.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--output-json",
            str(output),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "query_source must be an object" in result.stderr
    assert not output.exists()


def test_cli_rejects_completion_without_run_metadata(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    payload = completion_payload()
    payload.pop("required_evidence", None)
    payload["observed_evidence"].pop("run_metadata", None)
    completion = write_json(tmp_path / "completion.json", payload)
    output = tmp_path / "updated_review.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--output-json",
            str(output),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "required_evidence.run_metadata must be an object" in result.stderr
    assert "observed_evidence.run_metadata must be an object" in result.stderr
    assert not output.exists()


def test_cli_rejects_agentic_completion_without_nemoclaw_audit(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    payload = completion_payload()
    payload["required_evidence"].pop("nemoclaw_session_audit")
    payload["observed_evidence"].pop("nemoclaw_session_audit")
    completion = write_json(tmp_path / "completion.json", payload)
    output = tmp_path / "updated_review.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--output-json",
            str(output),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "required_evidence.nemoclaw_session_audit must be an object" in result.stderr
    assert "observed_evidence.nemoclaw_session_audit must be an object" in result.stderr
    assert not output.exists()


def test_cli_requires_scope_attestation_for_existing_result_adoption(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "completion.json", completion_payload())

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--adopt-existing-result",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--scope-attestation-json" in result.stderr


def test_cli_rejects_manual_scope_attestation_args_for_existing_result_adoption(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "completion.json", completion_payload())
    output = tmp_path / "updated_review.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--output-json",
            str(output),
            "--adopt-existing-result",
            "--scope-confirmed-by",
            "yuya",
            "--scope-confirmed-at",
            "2026-06-27T23:30:00+09:00",
            "--scope-confirmation",
            "This existing W&B run is the reviewed Agentic Math canary scope.",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Manual scope-attestation args are no longer supported" in result.stderr
    assert "--scope-attestation-json" in result.stderr
    assert not output.exists()


def test_cli_accepts_scope_attestation_json_for_existing_result_adoption(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "completion.json", completion_payload())
    attestation = write_json(
        tmp_path / "attestation.json",
        scope_attestation_payload(review, completion),
    )
    output = tmp_path / "updated_review.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--output-json",
            str(output),
            "--set-verify-wandb-completion",
            "--adopt-existing-result",
            "--scope-attestation-json",
            str(attestation),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["adopted_existing_result_count"] == 1
    updated = json.loads(output.read_text(encoding="utf-8"))
    assert updated["actual_cost_estimate"] == "$12.34"
    assert updated["provider_bill_reference"] == "openai-dashboard-2026-06-28"
    entry = updated["runs"][0]["wandb_completion"][0]
    assert entry["adopted_existing_result"] is True
    assert entry["entity"] == "test-entity"
    assert entry["project"] == "test-project"
    assert entry["sha256"] == sha256(completion)
    assert entry["scope_attestation"]["entity"] == "test-entity"
    assert entry["scope_attestation"]["project"] == "test-project"
    assert entry["scope_attestation"]["completion_sha256"] == sha256(completion)
    assert entry["scope_attestation"]["source_attestation_json"].endswith("attestation.json")
    assert entry["scope_attestation"]["source_attestation_sha256"] == sha256(attestation)
    assert entry["scope_attestation"]["completion_path"].endswith("completion.json")


def test_cli_accepts_scope_attestation_json_with_source_audit_binding(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "completion.json", completion_payload())
    source_audit = write_json(tmp_path / "existing_results_audit.json", source_audit_payload(completion))
    attestation = write_json(
        tmp_path / "attestation.json",
        scope_attestation_payload(review, completion, source_audit=source_audit),
    )
    output = tmp_path / "updated_review.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--output-json",
            str(output),
            "--set-verify-wandb-completion",
            "--adopt-existing-result",
            "--scope-attestation-json",
            str(attestation),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    updated = json.loads(output.read_text(encoding="utf-8"))
    entry = updated["runs"][0]["wandb_completion"][0]
    assert entry["scope_attestation"]["source_audit_json"] == str(source_audit)
    assert entry["scope_attestation"]["source_audit_sha256"] == sha256(source_audit)
    report = json.loads(result.stdout)
    assert report["entries"][0]["scope_attestation"]["source_audit_json"] == str(source_audit)
    assert report["entries"][0]["scope_attestation"]["source_audit_sha256"] == sha256(source_audit)


def test_cli_rejects_scope_attestation_source_audit_sha_mismatch(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "completion.json", completion_payload())
    source_audit = write_json(tmp_path / "existing_results_audit.json", source_audit_payload(completion))
    payload = scope_attestation_payload(review, completion, source_audit=source_audit)
    payload["source_audit_sha256"] = "0" * 64
    attestation = write_json(tmp_path / "attestation.json", payload)
    output = tmp_path / "updated_review.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--output-json",
            str(output),
            "--set-verify-wandb-completion",
            "--adopt-existing-result",
            "--scope-attestation-json",
            str(attestation),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "source_audit_sha256 does not match source_audit_json" in result.stderr
    assert not output.exists()


def test_cli_rejects_scope_attestation_source_audit_without_matching_record(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "completion.json", completion_payload())
    other_completion = tmp_path / "other_completion.json"
    source_audit = write_json(tmp_path / "existing_results_audit.json", source_audit_payload(other_completion))
    attestation = write_json(
        tmp_path / "attestation.json",
        scope_attestation_payload(review, completion, source_audit=source_audit),
    )
    output = tmp_path / "updated_review.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--output-json",
            str(output),
            "--set-verify-wandb-completion",
            "--adopt-existing-result",
            "--scope-attestation-json",
            str(attestation),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "source_audit_json formalized_records does not include completion entry" in result.stderr
    assert not output.exists()


def test_cli_rejects_placeholder_accounting_args(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "completion.json", completion_payload())
    output = tmp_path / "updated_review.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--output-json",
            str(output),
            "--actual-cost-estimate",
            "$ACTUAL_OR_BILLING_ESTIMATE",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--actual-cost-estimate must not be a placeholder" in result.stderr
    assert not output.exists()


def test_cli_rejects_placeholder_scope_attestation_accounting(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "completion.json", completion_payload())
    attestation = write_json(
        tmp_path / "attestation.json",
        {
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
            "provider_bill_reference": "BILL_OR_DASHBOARD_REFERENCE",
        },
    )

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--adopt-existing-result",
            "--scope-attestation-json",
            str(attestation),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "provider_bill_reference must not be a placeholder" in result.stderr


def test_cli_writes_validation_report_for_unconfirmed_scope_attestation(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "completion.json", completion_payload())
    payload = scope_attestation_payload(review, completion)
    payload["confirmed"] = False
    payload["confirmed_at"] = "YYYY-MM-DDTHH:MM:SS+09:00"
    payload["actual_cost_estimate"] = "$ACTUAL_OR_BILLING_ESTIMATE"
    payload["provider_bill_reference"] = "BILL_OR_DASHBOARD_REFERENCE"
    attestation = write_json(tmp_path / "attestation.json", payload)
    report = tmp_path / "sync_dry_run.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--adopt-existing-result",
            "--scope-attestation-json",
            str(attestation),
            "--report-json",
            str(report),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert "confirmed must be true" in result.stderr
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["status"] == "validation_failed"
    assert payload["dry_run"] is True
    assert payload["entry_count"] == 0
    assert "confirmed must be true" in payload["errors"][0]


def test_cli_rejects_scope_attestation_template_confirmed_at(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "completion.json", completion_payload())
    payload = scope_attestation_payload(review, completion)
    payload["confirmed_at"] = "YYYY-MM-DDTHH:MM:SS+09:00"
    attestation = write_json(tmp_path / "attestation.json", payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--adopt-existing-result",
            "--scope-attestation-json",
            str(attestation),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "confirmed_at must be a timezone-aware ISO 8601 timestamp" in result.stderr


def test_cli_rejects_scope_attestation_short_confirmation(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "completion.json", completion_payload())
    payload = scope_attestation_payload(review, completion)
    payload["confirmation"] = "ok"
    attestation = write_json(tmp_path / "attestation.json", payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--adopt-existing-result",
            "--scope-attestation-json",
            str(attestation),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "confirmation must be a concrete non-placeholder sentence" in result.stderr


def test_cli_accepts_scope_attestation_json_for_top_level_existing_result_adoption(tmp_path):
    review_payload_without_runs = review_payload()
    review_payload_without_runs["runs"] = []
    review = write_json(tmp_path / "review.json", review_payload_without_runs)
    completion = write_json(tmp_path / "completion.json", completion_payload(run_id="existing-run"))
    attestation = write_json(
        tmp_path / "attestation.json",
        {
            "schema_version": 1,
            "confirmed": True,
            "confirmed_by": "yuya",
            "confirmed_at": "2026-06-28T01:00:00+09:00",
            "confirmation": "This W&B run is the reviewed Agentic Math canary scope.",
            "review_path": str(review),
            "completion_path": str(completion),
            "completion_sha256": sha256(completion),
            "benchmark": "agentic_math",
            "entity": "test-entity",
            "project": "test-project",
            "run_id": "existing-run",
            "actual_cost_estimate": "$12.34",
            "provider_bill_reference": "openai-dashboard-2026-06-28",
        },
    )
    output = tmp_path / "updated_review.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--output-json",
            str(output),
            "--top-level",
            "--set-verify-wandb-completion",
            "--adopt-existing-result",
            "--scope-attestation-json",
            str(attestation),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "synced"
    assert report["unmatched_count"] == 0
    updated = json.loads(output.read_text(encoding="utf-8"))
    assert updated["runs"] == []
    assert updated["actual_cost_estimate"] == "$12.34"
    assert updated["provider_bill_reference"] == "openai-dashboard-2026-06-28"
    entry = updated["wandb_completion"][0]
    assert entry["run_id"] == "existing-run"
    assert entry["entity"] == "test-entity"
    assert entry["project"] == "test-project"
    assert entry["sha256"] == sha256(completion)
    assert entry["adopted_existing_result"] is True
    assert entry["scope_attestation"]["completion_sha256"] == sha256(completion)
    assert entry["scope_attestation"]["source_attestation_json"].endswith("attestation.json")


def test_cli_validates_matching_dry_run_report_before_apply(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "completion.json", completion_payload())
    attestation = write_json(
        tmp_path / "attestation.json",
        scope_attestation_payload(review, completion),
    )
    dry_run_report = tmp_path / "sync_dry_run.json"
    output = tmp_path / "updated_review.json"
    base_args = [
        "python3",
        str(SCRIPT),
        "--review-json",
        str(review),
        "--completion-json",
        str(completion),
        "--set-verify-wandb-completion",
        "--adopt-existing-result",
        "--scope-attestation-json",
        str(attestation),
    ]

    dry_run = subprocess.run(
        [*base_args, "--report-json", str(dry_run_report)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    dry_run_payload = json.loads(dry_run_report.read_text(encoding="utf-8"))
    assert dry_run_payload["dry_run"] is True
    assert dry_run_payload["in_place"] is False
    assert dry_run_payload["output_path"] == ""
    assert dry_run_payload["source_review_sha256"] == sha256(review)
    assert dry_run_payload["entry_count"] == 1
    assert dry_run_payload["entries"][0]["run_id"] == "run-1"
    assert dry_run_payload["entries"][0]["scope_attestation"]["source_attestation_sha256"] == sha256(attestation)

    apply_result = subprocess.run(
        [
            *base_args,
            "--output-json",
            str(output),
            "--validated-dry-run-report-json",
            str(dry_run_report),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert apply_result.returncode == 0, apply_result.stderr
    report = json.loads(apply_result.stdout)
    assert report["dry_run"] is False
    assert report["entry_count"] == dry_run_payload["entry_count"]
    assert report["entries"] == dry_run_payload["entries"]
    assert report["validated_dry_run_report_json"] == str(dry_run_report)
    assert output.exists()
    updated = json.loads(output.read_text(encoding="utf-8"))
    entry = updated["runs"][0]["wandb_completion"][0]
    assert entry["sync_dry_run_report_json"] == str(dry_run_report)
    assert entry["sync_dry_run_source_review_json"] == str(review)
    assert entry["sync_dry_run_source_review_sha256"] == sha256(review)


def test_cli_rejects_mismatched_validated_dry_run_report(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "completion.json", completion_payload())
    attestation = write_json(
        tmp_path / "attestation.json",
        scope_attestation_payload(review, completion),
    )
    dry_run_report = tmp_path / "sync_dry_run.json"
    output = tmp_path / "updated_review.json"
    base_args = [
        "python3",
        str(SCRIPT),
        "--review-json",
        str(review),
        "--completion-json",
        str(completion),
        "--set-verify-wandb-completion",
        "--adopt-existing-result",
        "--scope-attestation-json",
        str(attestation),
    ]

    dry_run = subprocess.run(
        [*base_args, "--report-json", str(dry_run_report)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    payload = json.loads(dry_run_report.read_text(encoding="utf-8"))
    payload["entries"][0]["run_id"] = "other-run"
    dry_run_report.write_text(json.dumps(payload), encoding="utf-8")

    apply_result = subprocess.run(
        [
            *base_args,
            "--output-json",
            str(output),
            "--validated-dry-run-report-json",
            str(dry_run_report),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert apply_result.returncode != 0
    assert "not a valid matching dry-run report" in apply_result.stderr
    assert "entries does not match current sync" in apply_result.stderr
    assert not output.exists()


def test_cli_rejects_validated_dry_run_report_without_generated_at(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "completion.json", completion_payload())
    attestation = write_json(
        tmp_path / "attestation.json",
        scope_attestation_payload(review, completion),
    )
    dry_run_report = tmp_path / "sync_dry_run.json"
    output = tmp_path / "updated_review.json"
    base_args = [
        "python3",
        str(SCRIPT),
        "--review-json",
        str(review),
        "--completion-json",
        str(completion),
        "--set-verify-wandb-completion",
        "--adopt-existing-result",
        "--scope-attestation-json",
        str(attestation),
    ]

    dry_run = subprocess.run(
        [*base_args, "--report-json", str(dry_run_report)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    payload = json.loads(dry_run_report.read_text(encoding="utf-8"))
    payload.pop("generated_at", None)
    dry_run_report.write_text(json.dumps(payload), encoding="utf-8")

    apply_result = subprocess.run(
        [
            *base_args,
            "--output-json",
            str(output),
            "--validated-dry-run-report-json",
            str(dry_run_report),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert apply_result.returncode != 0
    assert "not a valid matching dry-run report" in apply_result.stderr
    assert "generated_at must be a positive number" in apply_result.stderr
    assert not output.exists()


def test_cli_rejects_validated_dry_run_report_when_review_status_changed(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "completion.json", completion_payload())
    attestation = write_json(
        tmp_path / "attestation.json",
        scope_attestation_payload(review, completion),
    )
    dry_run_report = tmp_path / "sync_dry_run.json"
    output = tmp_path / "updated_review.json"
    base_args = [
        "python3",
        str(SCRIPT),
        "--review-json",
        str(review),
        "--completion-json",
        str(completion),
        "--set-verify-wandb-completion",
        "--adopt-existing-result",
        "--scope-attestation-json",
        str(attestation),
    ]

    dry_run = subprocess.run(
        [*base_args, "--report-json", str(dry_run_report)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    review_payload_changed = json.loads(review.read_text(encoding="utf-8"))
    review_payload_changed["status"] = "reviewed_after_dry_run"
    review.write_text(json.dumps(review_payload_changed), encoding="utf-8")

    apply_result = subprocess.run(
        [
            *base_args,
            "--output-json",
            str(output),
            "--validated-dry-run-report-json",
            str(dry_run_report),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert apply_result.returncode != 0
    assert "not a valid matching dry-run report" in apply_result.stderr
    assert "source_review_sha256 does not match current sync" in apply_result.stderr
    assert "before_status does not match current sync" in apply_result.stderr
    assert "after_status does not match current sync" in apply_result.stderr
    assert not output.exists()


def test_cli_rejects_validated_dry_run_report_when_review_content_changed(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "completion.json", completion_payload())
    attestation = write_json(
        tmp_path / "attestation.json",
        scope_attestation_payload(review, completion),
    )
    dry_run_report = tmp_path / "sync_dry_run.json"
    output = tmp_path / "updated_review.json"
    base_args = [
        "python3",
        str(SCRIPT),
        "--review-json",
        str(review),
        "--completion-json",
        str(completion),
        "--set-verify-wandb-completion",
        "--adopt-existing-result",
        "--scope-attestation-json",
        str(attestation),
    ]

    dry_run = subprocess.run(
        [*base_args, "--report-json", str(dry_run_report)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    review_payload_changed = json.loads(review.read_text(encoding="utf-8"))
    review_payload_changed["operator_note"] = "changed after dry-run"
    review.write_text(json.dumps(review_payload_changed), encoding="utf-8")

    apply_result = subprocess.run(
        [
            *base_args,
            "--output-json",
            str(output),
            "--validated-dry-run-report-json",
            str(dry_run_report),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert apply_result.returncode != 0
    assert "not a valid matching dry-run report" in apply_result.stderr
    assert "source_review_sha256 does not match current sync" in apply_result.stderr
    assert not output.exists()


def test_cli_rejects_validated_dry_run_report_without_apply_target(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "completion.json", completion_payload())
    attestation = write_json(
        tmp_path / "attestation.json",
        scope_attestation_payload(review, completion),
    )

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--set-verify-wandb-completion",
            "--adopt-existing-result",
            "--scope-attestation-json",
            str(attestation),
            "--validated-dry-run-report-json",
            str(tmp_path / "sync_dry_run.json"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--validated-dry-run-report-json requires --in-place or --output-json" in result.stderr


def test_cli_rejects_in_place_without_validated_dry_run_report(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "completion.json", completion_payload())

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--in-place",
            "--set-verify-wandb-completion",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--in-place requires --validated-dry-run-report-json" in result.stderr


def test_cli_rejects_scope_attestation_json_completion_sha_mismatch(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "completion.json", completion_payload())
    attestation = write_json(
        tmp_path / "attestation.json",
        {
            "schema_version": 1,
            "confirmed": True,
            "confirmed_by": "yuya",
            "confirmed_at": "2026-06-28T00:45:00+09:00",
            "confirmation": "This W&B run is the reviewed Agentic Math canary scope.",
            "review_path": str(review),
            "completion_path": str(completion),
            "completion_sha256": "0" * 64,
            "benchmark": "agentic_math",
            "entity": "test-entity",
            "project": "test-project",
            "run_id": "run-1",
            "actual_cost_estimate": "$12.34",
            "provider_bill_reference": "openai-dashboard-2026-06-28",
        },
    )

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--adopt-existing-result",
            "--scope-attestation-json",
            str(attestation),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "completion_sha256 does not match completion entry" in result.stderr


def test_cli_rejects_scope_attestation_json_mismatch(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "completion.json", completion_payload())
    attestation = write_json(
        tmp_path / "attestation.json",
        {
            "schema_version": 1,
            "confirmed": True,
            "confirmed_by": "yuya",
            "confirmed_at": "2026-06-28T00:45:00+09:00",
            "confirmation": "This W&B run is the reviewed Agentic Math canary scope.",
            "review_path": str(review),
            "completion_path": str(completion),
            "completion_sha256": sha256(completion),
            "benchmark": "agentic_swe",
            "entity": "test-entity",
            "project": "test-project",
            "run_id": "run-1",
            "actual_cost_estimate": "$12.34",
            "provider_bill_reference": "openai-dashboard-2026-06-28",
        },
    )

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--adopt-existing-result",
            "--scope-attestation-json",
            str(attestation),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "not a valid scope attestation JSON" in result.stderr
    assert "benchmark does not match completion entry" in result.stderr


def test_cli_rejects_scope_attestation_json_identity_mismatch(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "completion.json", completion_payload())
    attestation = write_json(
        tmp_path / "attestation.json",
        {
            "schema_version": 1,
            "confirmed": True,
            "confirmed_by": "yuya",
            "confirmed_at": "2026-06-28T00:45:00+09:00",
            "confirmation": "This W&B run is the reviewed Agentic Math canary scope.",
            "review_path": str(review),
            "completion_path": str(completion),
            "completion_sha256": sha256(completion),
            "benchmark": "agentic_math",
            "entity": "other-entity",
            "project": "other-project",
            "run_id": "run-1",
            "actual_cost_estimate": "$12.34",
            "provider_bill_reference": "openai-dashboard-2026-06-28",
        },
    )

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--adopt-existing-result",
            "--scope-attestation-json",
            str(attestation),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "entity does not match completion entry" in result.stderr
    assert "project does not match completion entry" in result.stderr


def test_cli_rejects_legacy_completion_json_by_default(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    legacy_payload = completion_payload()
    legacy_payload.pop("verification_schema_version")
    legacy_payload.pop("observed_evidence")
    completion = write_json(tmp_path / "completion.json", legacy_payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "not a current W&B completion verifier JSON" in result.stderr
    assert "verification_schema_version" in result.stderr
    assert "observed_evidence" in result.stderr


def test_cli_rejects_completion_json_without_entity_project(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    payload = completion_payload()
    payload.pop("entity")
    payload.pop("project")
    completion = write_json(tmp_path / "completion.json", payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "missing entity" in result.stderr


def test_cli_rejects_non_finished_completion_json_by_default(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    payload = completion_payload()
    payload["observed_evidence"]["run_state"] = "running"
    completion = write_json(tmp_path / "completion.json", payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "not a current W&B completion verifier JSON" in result.stderr
    assert "finished W&B run" in result.stderr


def test_cli_rejects_completion_json_with_failed_verifier_returncode(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    payload = completion_payload()
    payload["returncode_ok"] = False
    payload["returncode"] = 7
    completion = write_json(tmp_path / "completion.json", payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "verifier subprocess returncode was nonzero" in result.stderr
