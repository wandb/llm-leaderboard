import hashlib
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tools" / "render_wandb_scope_attestation.py"
PREFLIGHT_SCRIPT = REPO_ROOT / "scripts" / "tools" / "verify_wandb_scope_attestation.py"


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


def source_audit_payload(completion: Path):
    return {
        "ok": True,
        "status": "passed",
        "formalized_records": [
            {
                "benchmark": "agentic_math",
                "model_slug": "gpt-4_1-mini",
                "model": "gpt-4.1-mini-2025-04-14",
                "run_kind": "canary",
                "result_dir": "outputs/taiwan_full_eval/agentic_math/gpt-4_1-mini",
                "wandb_completion": {
                    "path": str(completion),
                    "entity": "test-entity",
                    "project": "test-project",
                    "run_id": "run-1",
                    "verification_schema_version": 1,
                    "schema_current": True,
                    "observed_evidence_present": True,
                },
            }
        ],
    }


def attestation_template(review: Path, completion: Path, source_audit: Path):
    return {
        "schema_version": 1,
        "confirmed": False,
        "confirmed_by": "REVIEWER",
        "confirmed_at": "YYYY-MM-DDTHH:MM:SS+09:00",
        "confirmation": "This W&B run is the reviewed canary scope for agentic_math.",
        "review_path": str(review),
        "completion_path": str(completion),
        "completion_sha256": sha256(completion),
        "source_audit_json": str(source_audit),
        "source_audit_sha256": sha256(source_audit),
        "benchmark": "agentic_math",
        "entity": "test-entity",
        "project": "test-project",
        "run_id": "run-1",
        "actual_cost_estimate": "$ACTUAL_OR_BILLING_ESTIMATE",
        "provider_bill_reference": "BILL_OR_DASHBOARD_REFERENCE",
    }


def test_rendered_scope_attestation_passes_preflight(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "completion.json", completion_payload())
    source_audit = write_json(tmp_path / "source_audit.json", source_audit_payload(completion))
    template = write_json(
        tmp_path / "scope_attestation.template.json",
        attestation_template(review, completion, source_audit),
    )
    output = tmp_path / "scope_attestation.json"
    report = tmp_path / "scope_attestation.render.json"
    markdown = tmp_path / "scope_attestation.render.md"
    preflight_json = tmp_path / "preflight.json"
    sync_dry_run_json = tmp_path / "sync_dry_run.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--template-json",
            str(template),
            "--output-json",
            str(output),
            "--confirmed-by",
            "yuya",
            "--confirmed-at",
            "2026-06-28T00:45:00+09:00",
            "--confirmation",
            "This W&B run is the reviewed Agentic Math canary scope.",
            "--actual-cost-estimate",
            "$12.34",
            "--provider-bill-reference",
            "openai-dashboard-2026-06-28",
            "--report-json",
            str(report),
            "--markdown",
            str(markdown),
            "--preflight-report-json",
            str(preflight_json),
            "--sync-dry-run-report-json",
            str(sync_dry_run_json),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    rendered = json.loads(output.read_text(encoding="utf-8"))
    render_report = json.loads(report.read_text(encoding="utf-8"))
    assert rendered["confirmed"] is True
    assert rendered["completion_sha256"] == sha256(completion)
    assert rendered["source_audit_sha256"] == sha256(source_audit)
    assert rendered["rendered_scope_attestation"]["source_template_sha256"] == sha256(template)
    assert render_report["ok"] is True
    assert render_report["output_sha256"] == sha256(output)
    assert render_report["will_execute_external_actions"] is False
    assert "--json" in render_report["next_commands"]["preflight"]
    assert str(preflight_json) in render_report["next_commands"]["preflight"]
    assert "--report-json" in render_report["next_commands"]["sync_dry_run"]
    assert str(sync_dry_run_json) in render_report["next_commands"]["sync_dry_run"]
    markdown_text = markdown.read_text(encoding="utf-8")
    assert "verify_wandb_scope_attestation.py" in markdown_text
    assert str(preflight_json) in markdown_text
    assert str(sync_dry_run_json) in markdown_text

    preflight = subprocess.run(
        [
            "python3",
            str(PREFLIGHT_SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--scope-attestation-json",
            str(output),
            "--json",
            str(preflight_json),
            "--require-query-source",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert preflight.returncode == 0, preflight.stderr
    preflight_report = json.loads(preflight_json.read_text(encoding="utf-8"))
    assert preflight_report["ok"] is True
    assert preflight_report["source_files"]["scope_attestation_json"]["sha256"] == sha256(output)


def test_scope_attestation_renderer_rejects_placeholders(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "completion.json", completion_payload())
    source_audit = write_json(tmp_path / "source_audit.json", source_audit_payload(completion))
    template = write_json(
        tmp_path / "scope_attestation.template.json",
        attestation_template(review, completion, source_audit),
    )
    report = tmp_path / "scope_attestation.render.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--template-json",
            str(template),
            "--output-json",
            str(tmp_path / "scope_attestation.json"),
            "--confirmed-by",
            "REVIEWER",
            "--confirmed-at",
            "YYYY-MM-DDTHH:MM:SS+09:00",
            "--confirmation",
            "TODO",
            "--actual-cost-estimate",
            "$ACTUAL_OR_BILLING_ESTIMATE",
            "--provider-bill-reference",
            "BILL_OR_DASHBOARD_REFERENCE",
            "--report-json",
            str(report),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["status"] == "validation_failed"
    assert "confirmed_by must be a concrete non-placeholder string" in payload["errors"]
    assert "confirmed_at must be a timezone-aware ISO 8601 timestamp" in payload["errors"]
    assert "actual_cost_estimate must be a concrete non-placeholder string" in payload["errors"]
