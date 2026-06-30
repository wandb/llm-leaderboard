import importlib.util
import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tools" / "draft_existing_wandb_adoption_review.py"


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


def completion_payload(*, with_run_metadata=True):
    payload = {
        "ok": True,
        "verification_schema_version": 1,
        "generated_at": 123.0,
        "benchmark": "agentic_math",
        "entity": "llm-leaderboard",
        "project": "tc-leaderboard",
        "run_id": "f1veetyb",
        "required_evidence": {
            "run_state": "finished",
            "expected_total": 100,
            "summary_metrics": [
                "agentic_math/total_instances",
                "agentic_math/answered_instances",
                "agentic_math/correct_instances",
                "agentic_math/accuracy",
            ],
            "tables": [
                {"name": "agentic_math_leaderboard_table", "row_count": ">=1"},
                {"name": "agentic_math_output_table", "row_count": "must equal total metric"},
            ],
            "artifacts": [{"type": "evaluation-results", "required_aliases": ["production"]}],
        },
        "observed_evidence": {
            "run_state": "finished",
            "expected_total": 100,
            "summary_metrics": {
                "agentic_math/total_instances": {"ok": True, "value": 100},
            },
            "tables": [
                {"name": "agentic_math_leaderboard_table", "ok": True, "nrows": 1},
                {"name": "agentic_math_output_table", "ok": True, "nrows": 100},
            ],
            "artifacts": [
                {
                    "name": "agentic-math-results:v0",
                    "type": "evaluation-results",
                    "aliases": ["latest", "production"],
                }
            ],
        },
    }
    if with_run_metadata:
        payload["required_evidence"]["run_metadata"] = {
            "config": [{"key": "benchmark", "expected": "agentic_math"}],
            "tags": ["taiwan-canary"],
            "group": "taiwan-one-model-canary",
            "job_type": "evaluation",
        }
        payload["observed_evidence"]["run_metadata"] = {
            "config": [{"key": "benchmark", "present": True, "value": "agentic_math"}],
            "tags": ["taiwan-canary"],
            "group": "taiwan-one-model-canary",
            "job_type": "evaluation",
        }
    return payload


def audit_payload(*, completion_path: str = "outputs/taiwan_full_eval/wandb_completion/agentic_math-f1veetyb.json"):
    return {
        "ok": True,
        "status": "passed",
        "formalized_records": [
            {
                "benchmark": "agentic_math",
                "model_slug": "deepseek-v4-pro-thinking-max",
                "model": "deepseek/deepseek-v4-pro",
                "run_kind": "final",
                "result_dir": "outputs/taiwan_full_eval/agentic_math/deepseek/openclaw",
                "expected_total": 100,
                "row_count": 100,
                "answered_instances": 99,
                "correct_instances": 86,
                "incorrect_instances": 14,
                "accuracy": 0.86,
                "wandb_completion": {
                    "path": completion_path,
                    "entity": "llm-leaderboard",
                    "project": "tc-leaderboard",
                    "run_id": "f1veetyb",
                    "run_name": "taiwan-agentic-math-deepseek-v4-pro-relog-20260627",
                    "verification_schema_version": 1,
                    "schema_current": True,
                    "observed_evidence_present": True,
                },
            }
        ],
    }


def test_build_draft_creates_scope_attested_sync_command(tmp_path):
    module = load_module()
    completion = write_json(tmp_path / "completion.json", completion_payload())
    audit = write_json(tmp_path / "audit.json", audit_payload(completion_path=str(completion)))

    draft = module.build_draft(
        audit_json=audit,
        review_json="outputs/taiwan_full_eval/canary_agentic_paid_run_review.json",
    )

    assert draft["status"] == "candidates_pending_scope_confirmation"
    assert draft["candidate_count"] == 1
    assert draft["source_audit_json"] == str(audit)
    assert draft["source_audit_sha256"] == sha256(audit)
    assert "scope_attestation_json.completion_sha256" in draft["required_human_fields"]
    assert "scope_attestation_json.provider_bill_reference" in draft["required_human_fields"]
    candidate = draft["candidates"][0]
    assert candidate["benchmark"] == "agentic_math"
    assert candidate["wandb_entity"] == "llm-leaderboard"
    assert candidate["wandb_project"] == "tc-leaderboard"
    assert candidate["wandb_run_id"] == "f1veetyb"
    assert candidate["wandb_completion_json"] == str(completion)
    assert candidate["wandb_completion_sha256"] == sha256(completion)
    assert candidate["source_audit_json"] == str(audit)
    assert candidate["source_audit_sha256"] == sha256(audit)
    assert candidate["scope_attestation_required"] is True
    assert candidate["scope_attestation_schema_version"] == 1
    assert candidate["run_metadata_valid"] is True
    assert candidate["run_metadata_errors"] == []
    assert candidate["sync_ready"] is True
    assert candidate["sync_command"] == ""
    assert candidate["sync_dry_run_command"] == ""
    assert candidate["sync_apply_command"] == ""
    assert candidate["sync_dry_run_report_json"] == ""
    assert candidate["scope_attestation_render_command"] == ""
    assert candidate["scope_attestation_render_report_json"] == ""
    assert candidate["scope_attestation_preflight_command"] == ""
    assert candidate["scope_attestation_preflight_report_json"] == ""
    assert "attestation-template-dir" in candidate["sync_command_blocked_reason"]
    assert "scope_attestation_json.completion_sha256" in candidate["required_human_fields"]
    assert "scope_attestation_json.provider_bill_reference" in candidate["required_human_fields"]


def test_build_draft_resolves_phase_placeholder_by_benchmark(tmp_path):
    module = load_module()
    completion = write_json(tmp_path / "completion.json", completion_payload())
    audit = write_json(tmp_path / "audit.json", audit_payload(completion_path=str(completion)))

    draft = module.build_draft(
        audit_json=audit,
        review_json="outputs/taiwan_full_eval/PHASE_paid_run_review.json",
    )

    candidate = draft["candidates"][0]
    assert candidate["review_json_template"] == "outputs/taiwan_full_eval/PHASE_paid_run_review.json"
    assert candidate["target_review_json"] == "outputs/taiwan_full_eval/canary_agentic_paid_run_review.json"
    assert candidate["scope_attestation_template"]["review_path"] == (
        "outputs/taiwan_full_eval/canary_agentic_paid_run_review.json"
    )
    assert candidate["scope_attestation_template"]["entity"] == "llm-leaderboard"
    assert candidate["scope_attestation_template"]["project"] == "tc-leaderboard"
    assert candidate["scope_attestation_template"]["completion_sha256"] == sha256(completion)
    assert candidate["scope_attestation_template"]["source_audit_json"] == str(audit)
    assert candidate["scope_attestation_template"]["source_audit_sha256"] == sha256(audit)


def test_cli_writes_json_and_markdown(tmp_path):
    completion = write_json(tmp_path / "completion.json", completion_payload())
    audit = write_json(tmp_path / "audit.json", audit_payload(completion_path=str(completion)))
    output_json = tmp_path / "draft.json"
    output_md = tmp_path / "draft.md"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--audit-json",
            str(audit),
            "--review-json",
            "outputs/taiwan_full_eval/canary_agentic_paid_run_review.json",
            "--json",
            str(output_json),
            "--markdown",
            str(output_md),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["path"] == str(output_json)
    assert payload["markdown_path"] == str(output_md)
    assert payload["candidate_count"] == 1
    markdown = output_md.read_text(encoding="utf-8")
    assert "Existing W&B Adoption Review Draft" in markdown
    assert "f1veetyb" in markdown
    assert "Run metadata" in markdown
    assert "Sync command is withheld until a scope-attestation template is generated." in markdown
    assert "attestation-template-dir" in markdown


def test_cli_writes_scope_attestation_templates(tmp_path):
    completion = write_json(tmp_path / "completion.json", completion_payload())
    audit = write_json(tmp_path / "audit.json", audit_payload(completion_path=str(completion)))
    output_json = tmp_path / "draft.json"
    output_md = tmp_path / "draft.md"
    attestation_dir = tmp_path / "attestations"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--audit-json",
            str(audit),
            "--review-json",
            "outputs/taiwan_full_eval/canary_agentic_paid_run_review.json",
            "--json",
            str(output_json),
            "--markdown",
            str(output_md),
            "--attestation-template-dir",
            str(attestation_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["path"] == str(output_json)
    assert payload["markdown_path"] == str(output_md)
    candidate = payload["candidates"][0]
    assert candidate["sync_ready"] is True
    assert candidate["run_metadata_valid"] is True
    template_path = REPO_ROOT / candidate["scope_attestation_template_json"]
    assert template_path.exists()
    template = json.loads(template_path.read_text(encoding="utf-8"))
    assert template["schema_version"] == 1
    assert template["confirmed"] is False
    assert template["benchmark"] == "agentic_math"
    assert template["entity"] == "llm-leaderboard"
    assert template["project"] == "tc-leaderboard"
    assert template["run_id"] == "f1veetyb"
    assert template["completion_path"] == str(completion)
    assert template["completion_sha256"] == sha256(completion)
    assert template["source_audit_json"] == str(audit)
    assert template["source_audit_sha256"] == sha256(audit)
    assert template["review_path"] == "outputs/taiwan_full_eval/canary_agentic_paid_run_review.json"
    assert "actual_cost_estimate" in template
    assert "provider_bill_reference" in template
    assert candidate["sync_dry_run_report_json"].endswith(
        "agentic_math-f1veetyb.sync_dry_run.json"
    )
    assert candidate["scope_attestation_render_report_json"].endswith(
        "agentic_math-f1veetyb.scope_attestation.render.json"
    )
    assert candidate["scope_attestation_render_markdown"].endswith(
        "agentic_math-f1veetyb.scope_attestation.render.md"
    )
    assert candidate["scope_attestation_preflight_report_json"].endswith(
        "agentic_math-f1veetyb.scope_preflight.json"
    )
    assert "render_wandb_scope_attestation.py" in candidate[
        "scope_attestation_render_command"
    ]
    assert candidate["scope_attestation_template_json"] in candidate[
        "scope_attestation_render_command"
    ]
    assert candidate["scope_attestation_render_report_json"] in candidate[
        "scope_attestation_render_command"
    ]
    assert "--preflight-report-json" in candidate["scope_attestation_render_command"]
    assert (
        candidate["scope_attestation_preflight_report_json"]
        in candidate["scope_attestation_render_command"]
    )
    assert "--sync-dry-run-report-json" in candidate["scope_attestation_render_command"]
    assert candidate["sync_dry_run_report_json"] in candidate[
        "scope_attestation_render_command"
    ]
    assert "verify_wandb_scope_attestation.py" in candidate[
        "scope_attestation_preflight_command"
    ]
    assert "--json" in candidate["scope_attestation_preflight_command"]
    assert (
        candidate["scope_attestation_preflight_report_json"]
        in candidate["scope_attestation_preflight_command"]
    )
    assert "--report-json" in candidate["sync_dry_run_command"]
    assert "--in-place" not in candidate["sync_dry_run_command"]
    assert candidate["sync_dry_run_report_json"] in candidate["sync_dry_run_command"]
    assert "--in-place" in candidate["sync_apply_command"]
    assert "--validated-dry-run-report-json" in candidate["sync_apply_command"]
    assert candidate["sync_dry_run_report_json"] in candidate["sync_apply_command"]
    assert "--scope-attestation-json" in candidate["sync_command"]
    assert "--top-level" in candidate["sync_command"]
    markdown = output_md.read_text(encoding="utf-8")
    assert candidate["scope_attestation_template_json"] in markdown
    assert "#### Candidate 1 Handoff Steps" in markdown
    assert "confirm_scope_attestation" in markdown
    assert "render_confirmed_attestation" in markdown
    assert "preflight_scope_attestation" in markdown
    assert "sync_paid_review_dry_run" in markdown
    assert "sync_paid_review_apply" in markdown
    assert "| confirm_scope_attestation | true | true | true | false |  |" in markdown
    assert "| sync_paid_review_apply | true | false | false | true |" in markdown
    assert "Preflight:" in markdown
    assert "Dry-run:" in markdown
    assert "Apply:" in markdown


def test_attestation_template_does_not_emit_sync_commands_without_run_metadata(tmp_path):
    completion = write_json(
        tmp_path / "completion.json",
        completion_payload(with_run_metadata=False),
    )
    audit = write_json(tmp_path / "audit.json", audit_payload(completion_path=str(completion)))
    output_json = tmp_path / "draft.json"
    output_md = tmp_path / "draft.md"
    attestation_dir = tmp_path / "attestations"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--audit-json",
            str(audit),
            "--review-json",
            "outputs/taiwan_full_eval/canary_agentic_paid_run_review.json",
            "--json",
            str(output_json),
            "--markdown",
            str(output_md),
            "--attestation-template-dir",
            str(attestation_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    candidate = payload["candidates"][0]
    assert candidate["run_metadata_valid"] is False
    assert candidate["sync_ready"] is False
    assert "required_evidence.run_metadata must be an object" in candidate["run_metadata_errors"]
    assert "observed_evidence.run_metadata must be an object" in candidate["run_metadata_errors"]
    assert candidate["sync_command"] == ""
    assert candidate["sync_dry_run_command"] == ""
    assert candidate["sync_apply_command"] == ""
    assert candidate["scope_attestation_render_command"] == ""
    assert candidate["scope_attestation_preflight_command"] == ""
    assert "refresh_wandb_completion_command" in candidate
    assert "verify_taiwan_wandb_completion.py" in candidate["refresh_wandb_completion_command"]
    assert (
        "--expected-run-config model.pretrained_model_name_or_path=deepseek/deepseek-v4-pro"
        in candidate["refresh_wandb_completion_command"]
    )
    assert "--expected-run-job-type evaluation-relog" in candidate[
        "refresh_wandb_completion_command"
    ]
    template_path = REPO_ROOT / candidate["scope_attestation_template_json"]
    assert template_path.exists()
    markdown = output_md.read_text(encoding="utf-8")
    assert "Refresh verifier:" in markdown
    assert "required run metadata" in markdown
