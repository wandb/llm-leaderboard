import hashlib
import json
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tools" / "check_taiwan_paid_run_review_package.py"


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_eval_preflight_payload(path: Path | None = None):
    if path is None:
        path = REPO_ROOT / "temp" / "test_taiwan_paid_review_run_eval_preflight.json"
    return write_json(
        path,
        {
            "schema_version": 1,
            "generated_at": time.time(),
            "status": "passed",
            "ok": True,
            "config": "config.yaml",
            "base_config": "configs/base_config_taiwan.yaml",
            "wandb": {
                "entity": "test-entity",
                "project": "test-project",
                "run_name": "taiwan/full/test",
            },
            "api": "openai_responses",
            "model": "gpt-4.1-mini-2025-04-14",
            "enabled_benchmarks": ["agentic_math", "swebench_pro"],
            "will_initialize_wandb": False,
            "will_log_wandb_artifacts": False,
            "will_initialize_weave": False,
            "will_start_inference_engine": False,
            "will_run_evaluators": False,
            "token_validation": {"ok": True, "has_errors": False, "results": []},
        },
    )


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


def wandb_completion_payload(*, run_id: str = "run-1"):
    payload = {
        "ok": True,
        "benchmark": "agentic_math",
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
            "benchmark": "agentic_math",
            "summary_source": "run.summary_metrics",
            "artifact_source": "run.logged_artifacts",
            "history_scanned": False,
        },
        "generated_at": time.time(),
        "verification_schema_version": 1,
        "observed_evidence": {
            "run_state": "finished",
            "summary_metrics": {
                "agentic_math/accuracy": {"ok": True, "value": 0.86},
            },
        },
    }
    return add_wandb_run_metadata(payload, run_id=run_id)


def completed_review_payload(completion_path: Path):
    completion_sha256 = sha256_file(completion_path)
    preflight = write_run_eval_preflight_payload()
    return {
        "status": "completed",
        "phase": "full",
        "canary": True,
        "model_count": 1,
        "configs": ["config.yaml"],
        "run_purpose": "one-model release canary",
        "expected_cost_band": "$10-$20",
        "execution_plan_path": "plan.json",
        "batch_manifest_path": "manifest.json",
        "post_run_cost_command": "estimate",
        "run_eval_preflights": [
            {
                "config": "config.yaml",
                "output_json": str(preflight),
                "required_before_run_eval": True,
                "command": [
                    "python3",
                    "scripts/run_eval.py",
                    "--base-config",
                    "base_config_taiwan.yaml",
                    "--config",
                    "config.yaml",
                    "--preflight",
                    "--preflight-json",
                    str(preflight),
                ],
            }
        ],
        "actual_cost_estimate": "$12.34",
        "provider_bill_reference": "billing export",
        "created_at": time.time() - 100,
        "ended_at": time.time(),
        "verify_wandb_completion": True,
        "runs": [
            {
                "config": "config.yaml",
                "preflight_json": str(preflight),
                "preflight_returncode": 0,
                "preflight_ok": True,
                "preflight_status": "passed",
                "log_path": "run.log",
                "wandb_entity": "test-entity",
                "wandb_project": "test-project",
                "wandb_run_id": "run-1",
                "returncode": 0,
                "started_at": time.time() - 90,
                "ended_at": time.time() - 10,
                "wandb_completion": [
                    {
                        "ok": True,
                        "benchmark": "agentic_math",
                        "entity": "test-entity",
                        "project": "test-project",
                        "run_id": "run-1",
                        "path": str(completion_path),
                        "sha256": completion_sha256,
                    }
                ],
            }
        ],
    }


def write_weave_completion(path: Path, *, ok: bool = True, agent_name: str = "nejumi-taiwan-openclaw"):
    return write_json(
        path,
        {
            "ok": ok,
            "agent_name": agent_name,
            "generated_at": time.time(),
            "verification_schema_version": 1,
            "project_id": "test-entity/test-project",
            "agents_url": "https://wandb.ai/test-entity/test-project/weave/agents",
            "query_source": {
                "kind": "wandb_agents_api",
                "api_base_url": "https://trace.wandb.ai",
                "agents_endpoint": "/agents/query",
                "spans_endpoint": "/agents/spans/query",
                "project_id": "test-entity/test-project",
                "agent_name": agent_name,
                "conversation_id": "",
                "conversation_id_contains": "run-1",
                "agents_count": 1,
                "spans_count": 2,
                "matching_span_count": 2,
                "latest_trace_span_count": 2,
            },
            "latest_trace_id": "trace-1" if ok else "",
            "required_evidence": {
                "input_message_required": True,
                "trace_timestamp_quality_required": True,
                "trace_final_answer_order_required": True,
                "conversation_id": "",
                "conversation_id_contains": "run-1",
            },
            "content_capture_health": {
                "span_count_checked": 2,
                "message_span_count": 2,
                "message_spans_with_content": 2 if ok else 0,
                "message_spans_with_input": 1 if ok else 0,
                "tool_span_count": 1,
                "tool_spans_with_content": 1 if ok else 0,
                "spans_with_valid_timestamps": 2,
                "spans_with_invalid_timestamps": 0,
            },
            "latest_trace_spans_chronological": [
                {
                    "started_at": "2026-06-28T00:00:00Z",
                    "ended_at": "2026-06-28T00:00:01Z",
                    "span_name": "chat",
                    "operation_name": "chat",
                    "agent_name": agent_name,
                    "trace_id": "trace-1",
                    "span_id": "span-1",
                    "parent_span_id": None,
                    "error_type": None,
                    "has_input_messages": ok,
                    "has_output_messages": ok,
                },
                {
                    "started_at": "2026-06-28T00:00:02Z",
                    "ended_at": "2026-06-28T00:00:03Z",
                    "span_name": "python.exec",
                    "operation_name": "execute_tool",
                    "agent_name": agent_name,
                    "trace_id": "trace-1",
                    "span_id": "span-2",
                    "parent_span_id": "span-1",
                    "tool_name": "python",
                    "error_type": None,
                },
            ],
            "checks": [
                {"name": "agent_present", "ok": ok},
                {"name": "message_content_capture", "ok": ok},
                {"name": "input_message_capture", "ok": ok},
                {"name": "tool_content_capture", "ok": ok},
                {"name": "trace_timestamp_quality", "ok": True},
                {"name": "trace_order", "ok": ok},
                {"name": "trace_user_message_order", "ok": ok},
                {"name": "trace_final_answer_order", "ok": ok},
            ],
        },
    )


def write_weave_sync_dry_run(
    path: Path,
    *,
    review_path: Path,
    completion_path: Path,
    run_id: str = "run-1",
    agent_name: str = "nejumi-taiwan-openclaw",
    source_review_sha256: str | None = None,
):
    if source_review_sha256 is None:
        source_review_sha256 = sha256_file(review_path) if review_path.exists() else "f" * 64
    return write_json(
        path,
        {
            "ok": True,
            "status": "synced",
            "generated_at": time.time(),
            "review_path": str(review_path),
            "source_review_sha256": source_review_sha256,
            "output_path": "",
            "in_place": False,
            "dry_run": True,
            "entry_count": 1,
            "entries": [
                {
                    "ok": True,
                    "run_id": run_id,
                    "path": str(completion_path),
                    "agent_name": agent_name,
                    "verification_schema_version": 1,
                    "latest_trace_id": "trace-1",
                    "checks_valid": True,
                    "trace_present": True,
                    "run_scope_proven": True,
                    "conversation_id": "",
                    "conversation_id_contains": run_id,
                    "query_source_kind": "wandb_agents_api",
                    "query_source_api_base_url": "https://trace.wandb.ai",
                    "query_source_agents_endpoint": "/agents/query",
                    "query_source_spans_endpoint": "/agents/spans/query",
                    "query_source_project_id": "test-entity/test-project",
                }
            ],
            "change_count": 1,
            "changes": [
                {
                    "target": "run",
                    "action": "added",
                    "run_id": run_id,
                    "agent_name": agent_name,
                    "config": "config.yaml",
                }
            ],
            "unmatched_count": 0,
            "unmatched_entries": [],
            "before_status": "completed",
            "after_status": "completed",
            "verify_weave_agents": True,
        },
    )


def write_wandb_sync_dry_run(
    path: Path,
    *,
    source_review: Path,
    completion_path: Path,
    source_attestation: Path,
    source_attestation_sha: str,
    source_audit: Path,
    source_audit_sha: str,
):
    completion_sha = sha256_file(completion_path)
    return write_json(
        path,
        {
            "ok": True,
            "status": "synced",
            "generated_at": time.time(),
            "review_path": str(source_review),
            "source_review_sha256": sha256_file(source_review),
            "output_path": "",
            "in_place": False,
            "dry_run": True,
            "entry_count": 1,
            "adopted_existing_result_count": 1,
            "entries": [
                {
                    "benchmark": "agentic_math",
                    "ok": True,
                    "entity": "test-entity",
                    "project": "test-project",
                    "run_id": "run-1",
                    "path": str(completion_path),
                    "sha256": completion_sha,
                    "verification_schema_version": 1,
                    "observed_evidence_valid": True,
                    "run_metadata_valid": True,
                    "adopted_existing_result": True,
                    "scope_attestation": {
                        "schema_version": 1,
                        "confirmed": True,
                        "confirmed_by": "yuya",
                        "confirmed_at": "2026-06-27T23:30:00+09:00",
                        "confirmation": "This existing run is within the reviewed one-model canary scope.",
                        "review_path": str(source_review),
                        "completion_path": str(completion_path),
                        "benchmark": "agentic_math",
                        "entity": "test-entity",
                        "project": "test-project",
                        "run_id": "run-1",
                        "completion_sha256": completion_sha,
                        "actual_cost_estimate": "$12.34",
                        "provider_bill_reference": "billing export",
                        "source_attestation_json": str(source_attestation),
                        "source_attestation_sha256": source_attestation_sha,
                        "source_audit_json": str(source_audit),
                        "source_audit_sha256": source_audit_sha,
                    },
                }
            ],
            "change_count": 1,
            "unmatched_count": 0,
            "changes": [
                {
                    "target": "run",
                    "action": "added",
                    "benchmark": "agentic_math",
                    "entity": "test-entity",
                    "project": "test-project",
                    "run_id": "run-1",
                    "match_status": "matched_wandb_identity",
                }
            ],
            "unmatched_entries": [],
            "before_status": "completed",
            "after_status": "completed",
            "verify_wandb_completion": True,
        },
    )


def wandb_source_audit_payload(*, completion_path: Path):
    return {
        "ok": True,
        "status": "passed",
        "formalized_records": [
            {
                "benchmark": "agentic_math",
                "wandb_completion": {
                    "path": str(completion_path),
                    "entity": "test-entity",
                    "project": "test-project",
                    "run_id": "run-1",
                    "verification_schema_version": 1,
                },
            }
        ],
    }


def test_paid_review_doctor_accepts_verified_one_model_review(tmp_path):
    completion = write_json(
        tmp_path / "agentic_math-run-1.json",
        wandb_completion_payload(),
    )
    source_review = write_json(
        tmp_path / "review.before_weave_sync.json",
        completed_review_payload(completion),
    )
    source_review_sha256 = sha256_file(source_review)
    weave_completion = write_weave_completion(tmp_path / "weave-agents-run-1.json")
    weave_sync_dry_run = write_weave_sync_dry_run(
        tmp_path / "weave-agents-run-1.sync_dry_run.json",
        review_path=source_review,
        completion_path=weave_completion,
        source_review_sha256=source_review_sha256,
    )
    payload = completed_review_payload(completion)
    payload["verify_weave_agents"] = True
    payload["runs"][0]["weave_agents_completion"] = {
        "ok": True,
        "path": str(weave_completion),
        "agent_name": "nejumi-taiwan-openclaw",
        "run_id": "run-1",
        "sync_dry_run_report_json": str(weave_sync_dry_run),
        "sync_dry_run_source_review_json": str(source_review),
        "sync_dry_run_source_review_sha256": source_review_sha256,
    }
    review = write_json(tmp_path / "review.json", payload)
    output_json = tmp_path / "doctor.json"
    output_md = tmp_path / "doctor.md"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--required-wandb-benchmark",
            "agentic_math",
            "--required-wandb-run-id",
            "agentic_math=run-1",
            "--require-one-model-canary",
            "--json",
            str(output_json),
            "--markdown",
            str(output_md),
            "--fail-on-invalid",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["summary"]["blockers"] == []
    assert payload["review_completion_requirements"]["completed_review_required_fields"] == [
        "ended_at",
        "actual_cost_estimate",
        "provider_bill_reference",
        "runs",
    ]
    assert "run_eval_preflights" in payload["review_completion_requirements"]["review_required_fields"]
    assert payload["review_completion_requirements"]["run_eval_preflight_required_when"] == (
        "before every run_eval.py invocation"
    )
    assert "preflight_json" in payload["review_completion_requirements"]["run_required_fields"]
    assert "preflight_returncode" in payload["review_completion_requirements"]["run_required_fields"]
    assert "preflight_ok" in payload["review_completion_requirements"]["run_required_fields"]
    assert (
        payload["review_completion_requirements"]["wandb_completion_verifier_requirements"][
            "observed_evidence.run_state"
        ]
        == "finished"
    )
    assert json.loads(output_json.read_text(encoding="utf-8"))["status"] == "passed"
    markdown = output_md.read_text(encoding="utf-8")
    assert "paid_run_review_package" in markdown
    assert "Completion Requirements" in markdown
    assert "provider_bill_reference" in markdown


def test_paid_review_doctor_rejects_non_adopted_wandb_completion_missing_query_source(tmp_path):
    payload = wandb_completion_payload()
    payload.pop("query_source")
    completion = write_json(tmp_path / "agentic_math-run-1.json", payload)
    review = write_json(tmp_path / "review.json", completed_review_payload(completion))

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--required-wandb-benchmark",
            "agentic_math",
            "--required-wandb-run-id",
            "agentic_math=run-1",
            "--require-one-model-canary",
            "--fail-on-invalid",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    entry = report["gates"][0]["records"][0]["wandb_completion_entries"][0]
    assert entry["verified"] is False
    assert entry["query_source_required"] is True
    assert entry["query_source_valid"] is False
    assert "query_source is not an object" in entry["verification_error"]


def test_paid_review_doctor_rejects_wandb_completion_missing_run_metadata(tmp_path):
    payload = wandb_completion_payload()
    payload.pop("required_evidence", None)
    payload["observed_evidence"].pop("run_metadata", None)
    completion = write_json(tmp_path / "agentic_math-run-1.json", payload)
    review = write_json(tmp_path / "review.json", completed_review_payload(completion))

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--required-wandb-benchmark",
            "agentic_math",
            "--required-wandb-run-id",
            "agentic_math=run-1",
            "--require-one-model-canary",
            "--fail-on-invalid",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    entry = report["gates"][0]["records"][0]["wandb_completion_entries"][0]
    assert entry["verified"] is False
    assert entry["run_metadata_valid"] is False
    assert "required_evidence.run_metadata is not an object" in entry["verification_error"]
    assert "observed_evidence.run_metadata is not an object" in entry["verification_error"]


def test_paid_review_doctor_rejects_placeholder_accounting_fields(tmp_path):
    completion = write_json(
        tmp_path / "agentic_math-run-1.json",
        add_wandb_run_metadata(
            {
                "ok": True,
                "benchmark": "agentic_math",
                "entity": "test-entity",
                "project": "test-project",
                "run_id": "run-1",
                "generated_at": time.time(),
                "verification_schema_version": 1,
                "observed_evidence": {
                    "run_state": "finished",
                    "summary_metrics": {
                        "agentic_math/accuracy": {"ok": True, "value": 0.86},
                    },
                },
            }
        ),
    )
    payload = completed_review_payload(completion)
    payload["actual_cost_estimate"] = "$ACTUAL_OR_BILLING_ESTIMATE"
    payload["provider_bill_reference"] = "BILL_OR_DASHBOARD_REFERENCE"
    review = write_json(tmp_path / "review.json", payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--required-wandb-benchmark",
            "agentic_math",
            "--required-wandb-run-id",
            "agentic_math=run-1",
            "--require-one-model-canary",
            "--fail-on-invalid",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    record = report["gates"][0]["records"][0]
    assert record["actual_cost_estimate_placeholder"] is True
    assert record["provider_bill_reference_placeholder"] is True
    assert "completed review actual_cost_estimate must not be a placeholder" in record["errors"]
    assert "completed review provider_bill_reference must not be a placeholder" in record["errors"]


def test_paid_review_doctor_rejects_adopted_existing_result_without_attestation(tmp_path):
    completion = write_json(
        tmp_path / "agentic_math-run-1.json",
        add_wandb_run_metadata(
            {
                "ok": True,
                "benchmark": "agentic_math",
                "entity": "test-entity",
                "project": "test-project",
                "run_id": "run-1",
                "generated_at": time.time(),
                "verification_schema_version": 1,
                "observed_evidence": {
                    "run_state": "finished",
                    "summary_metrics": {
                        "agentic_math/accuracy": {"ok": True, "value": 0.86},
                    },
                },
            }
        ),
    )
    payload = completed_review_payload(completion)
    payload["runs"][0]["wandb_completion"][0]["adopted_existing_result"] = True
    review = write_json(tmp_path / "review.json", payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--required-wandb-benchmark",
            "agentic_math",
            "--required-wandb-run-id",
            "agentic_math=run-1",
            "--require-one-model-canary",
            "--fail-on-invalid",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    entry = report["gates"][0]["records"][0]["wandb_completion_entries"][0]
    assert entry["verified"] is False
    assert entry["scope_attestation_valid"] is False
    assert "missing scope_attestation" in entry["verification_error"]


def test_paid_review_doctor_accepts_adopted_existing_result_with_attestation(tmp_path):
    completion = write_json(
        tmp_path / "agentic_math-run-1.json",
        add_wandb_run_metadata(
            {
                "ok": True,
                "benchmark": "agentic_math",
                "entity": "test-entity",
                "project": "test-project",
                "run_id": "run-1",
                "generated_at": time.time(),
                "verification_schema_version": 1,
                "observed_evidence": {
                    "run_state": "finished",
                    "summary_metrics": {
                        "agentic_math/accuracy": {"ok": True, "value": 0.86},
                    },
                },
            }
        ),
    )
    source_review = write_json(
        tmp_path / "source_review.json",
        completed_review_payload(completion),
    )
    payload = completed_review_payload(completion)
    review = tmp_path / "review.json"
    source_audit = write_json(
        tmp_path / "source_audit.json",
        wandb_source_audit_payload(completion_path=completion),
    )
    source_audit_sha = sha256_file(source_audit)
    source_attestation = write_json(
        tmp_path / "agentic_math-run-1.scope_attestation.json",
        {
            "schema_version": 1,
            "confirmed": True,
            "confirmed_by": "yuya",
            "confirmed_at": "2026-06-27T23:30:00+09:00",
            "confirmation": "This existing run is within the reviewed one-model canary scope.",
            "review_path": str(source_review),
            "completion_path": str(completion),
            "benchmark": "agentic_math",
            "entity": "test-entity",
            "project": "test-project",
            "run_id": "run-1",
            "completion_sha256": sha256_file(completion),
            "actual_cost_estimate": "$12.34",
            "provider_bill_reference": "billing export",
            "source_audit_json": str(source_audit),
            "source_audit_sha256": source_audit_sha,
        },
    )
    source_attestation_sha = sha256_file(source_attestation)
    dry_run = write_wandb_sync_dry_run(
        tmp_path / "agentic_math-run-1.sync_dry_run.json",
        source_review=source_review,
        completion_path=completion,
        source_attestation=source_attestation,
        source_attestation_sha=source_attestation_sha,
        source_audit=source_audit,
        source_audit_sha=source_audit_sha,
    )
    entry = payload["runs"][0]["wandb_completion"][0]
    entry["adopted_existing_result"] = True
    entry["sync_dry_run_report_json"] = str(dry_run)
    entry["sync_dry_run_source_review_json"] = str(source_review)
    entry["sync_dry_run_source_review_sha256"] = sha256_file(source_review)
    entry["scope_attestation"] = {
        "schema_version": 1,
        "confirmed": True,
        "confirmed_by": "yuya",
        "confirmed_at": "2026-06-27T23:30:00+09:00",
        "confirmation": "This existing run is within the reviewed one-model canary scope.",
        "review_path": str(source_review),
        "completion_path": str(completion),
        "benchmark": "agentic_math",
        "entity": "test-entity",
        "project": "test-project",
        "run_id": "run-1",
        "completion_sha256": sha256_file(completion),
        "actual_cost_estimate": "$12.34",
        "provider_bill_reference": "billing export",
        "source_attestation_json": str(source_attestation),
        "source_attestation_sha256": source_attestation_sha,
        "source_audit_json": str(source_audit),
        "source_audit_sha256": source_audit_sha,
    }
    weave_completion = write_weave_completion(tmp_path / "weave-agents-run-1.json")
    weave_sync_dry_run = write_weave_sync_dry_run(
        tmp_path / "weave-agents-run-1.sync_dry_run.json",
        review_path=source_review,
        completion_path=weave_completion,
        source_review_sha256=sha256_file(source_review),
    )
    payload["verify_weave_agents"] = True
    payload["runs"][0]["weave_agents_completion"] = {
        "ok": True,
        "path": str(weave_completion),
        "agent_name": "nejumi-taiwan-openclaw",
        "run_id": "run-1",
        "sync_dry_run_report_json": str(weave_sync_dry_run),
        "sync_dry_run_source_review_json": str(source_review),
        "sync_dry_run_source_review_sha256": sha256_file(source_review),
    }
    write_json(review, payload)
    output_md = tmp_path / "doctor.md"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--required-wandb-benchmark",
            "agentic_math",
            "--required-wandb-run-id",
            "agentic_math=run-1",
            "--require-one-model-canary",
            "--markdown",
            str(output_md),
            "--fail-on-invalid",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    entry = report["gates"][0]["records"][0]["wandb_completion_entries"][0]
    assert entry["verified"] is True
    assert entry["scope_attestation_valid"] is True
    assert entry["sync_dry_run_report_ok"] is True
    markdown = output_md.read_text(encoding="utf-8")
    assert "## W&B Completion Entries" in markdown
    assert "agentic_math" in markdown
    assert "run-1" in markdown
    assert "yuya" in markdown
    assert "2026-06-27T23:30:00+09:00" in markdown
    assert str(source_attestation) in markdown
    assert str(source_audit) in markdown
    assert str(dry_run) in markdown
    assert source_attestation_sha in markdown
    assert source_audit_sha in markdown


def test_paid_review_doctor_rejects_adopted_existing_result_placeholder_attestation_accounting(tmp_path):
    completion = write_json(
        tmp_path / "agentic_math-run-1.json",
        {
            "ok": True,
            "benchmark": "agentic_math",
            "entity": "test-entity",
            "project": "test-project",
            "run_id": "run-1",
            "generated_at": time.time(),
            "verification_schema_version": 1,
            "observed_evidence": {
                "run_state": "finished",
                "summary_metrics": {
                    "agentic_math/accuracy": {"ok": True, "value": 0.86},
                },
            },
        },
    )
    payload = completed_review_payload(completion)
    review = tmp_path / "review.json"
    source_attestation = write_json(
        tmp_path / "agentic_math-run-1.scope_attestation.json",
        {
            "schema_version": 1,
            "confirmed": True,
            "confirmed_by": "yuya",
            "confirmed_at": "2026-06-27T23:30:00+09:00",
            "confirmation": "This existing run is within the reviewed one-model canary scope.",
            "review_path": str(review),
            "completion_path": str(completion),
            "benchmark": "agentic_math",
            "run_id": "run-1",
            "actual_cost_estimate": "$ACTUAL_OR_BILLING_ESTIMATE",
            "provider_bill_reference": "BILL_OR_DASHBOARD_REFERENCE",
        },
    )
    entry = payload["runs"][0]["wandb_completion"][0]
    entry["adopted_existing_result"] = True
    entry["scope_attestation"] = {
        "schema_version": 1,
        "confirmed": True,
        "confirmed_by": "yuya",
        "confirmed_at": "2026-06-27T23:30:00+09:00",
        "confirmation": "This existing run is within the reviewed one-model canary scope.",
        "review_path": str(review),
        "completion_path": str(completion),
        "benchmark": "agentic_math",
        "run_id": "run-1",
        "actual_cost_estimate": "$ACTUAL_OR_BILLING_ESTIMATE",
        "provider_bill_reference": "BILL_OR_DASHBOARD_REFERENCE",
        "source_attestation_json": str(source_attestation),
    }
    write_json(review, payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--required-wandb-benchmark",
            "agentic_math",
            "--required-wandb-run-id",
            "agentic_math=run-1",
            "--require-one-model-canary",
            "--fail-on-invalid",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    entry = report["gates"][0]["records"][0]["wandb_completion_entries"][0]
    assert entry["verified"] is False
    assert entry["scope_attestation_valid"] is False
    assert "scope_attestation actual_cost_estimate must not be a placeholder" in entry["verification_error"]
    assert "scope_attestation provider_bill_reference must not be a placeholder" in entry["verification_error"]
    assert "source_attestation_json actual_cost_estimate must not be a placeholder" in entry["verification_error"]
    assert "source_attestation_json provider_bill_reference must not be a placeholder" in entry["verification_error"]


def test_paid_review_doctor_rejects_adopted_existing_result_invalid_confirmed_at(tmp_path):
    completion = write_json(
        tmp_path / "agentic_math-run-1.json",
        {
            "ok": True,
            "benchmark": "agentic_math",
            "entity": "test-entity",
            "project": "test-project",
            "run_id": "run-1",
            "generated_at": time.time(),
            "verification_schema_version": 1,
            "observed_evidence": {
                "run_state": "finished",
                "summary_metrics": {
                    "agentic_math/accuracy": {"ok": True, "value": 0.86},
                },
            },
        },
    )
    payload = completed_review_payload(completion)
    review = tmp_path / "review.json"
    source_attestation = write_json(
        tmp_path / "agentic_math-run-1.scope_attestation.json",
        {
            "schema_version": 1,
            "confirmed": True,
            "confirmed_by": "yuya",
            "confirmed_at": "2026-06-27T23:30:00",
            "confirmation": "This existing run is within the reviewed one-model canary scope.",
            "review_path": str(review),
            "completion_path": str(completion),
            "benchmark": "agentic_math",
            "entity": "test-entity",
            "project": "test-project",
            "run_id": "run-1",
            "completion_sha256": sha256_file(completion),
            "actual_cost_estimate": "$12.34",
            "provider_bill_reference": "billing export",
        },
    )
    entry = payload["runs"][0]["wandb_completion"][0]
    entry["adopted_existing_result"] = True
    entry["scope_attestation"] = {
        "schema_version": 1,
        "confirmed": True,
        "confirmed_by": "yuya",
        "confirmed_at": "2026-06-27T23:30:00",
        "confirmation": "This existing run is within the reviewed one-model canary scope.",
        "review_path": str(review),
        "completion_path": str(completion),
        "benchmark": "agentic_math",
        "entity": "test-entity",
        "project": "test-project",
        "run_id": "run-1",
        "completion_sha256": sha256_file(completion),
        "actual_cost_estimate": "$12.34",
        "provider_bill_reference": "billing export",
        "source_attestation_json": str(source_attestation),
    }
    write_json(review, payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--required-wandb-benchmark",
            "agentic_math",
            "--required-wandb-run-id",
            "agentic_math=run-1",
            "--require-one-model-canary",
            "--fail-on-invalid",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    entry = report["gates"][0]["records"][0]["wandb_completion_entries"][0]
    assert entry["verified"] is False
    assert entry["scope_attestation_valid"] is False
    assert "scope_attestation confirmed_at must be a timezone-aware ISO 8601 timestamp" in entry["verification_error"]
    assert "source_attestation_json confirmed_at must be a timezone-aware ISO 8601 timestamp" in entry["verification_error"]


def test_paid_review_doctor_rejects_adopted_existing_result_without_source_attestation_json(tmp_path):
    completion = write_json(
        tmp_path / "agentic_math-run-1.json",
        {
            "ok": True,
            "benchmark": "agentic_math",
            "entity": "test-entity",
            "project": "test-project",
            "run_id": "run-1",
            "generated_at": time.time(),
            "verification_schema_version": 1,
            "observed_evidence": {
                "run_state": "finished",
                "summary_metrics": {
                    "agentic_math/accuracy": {"ok": True, "value": 0.86},
                },
            },
        },
    )
    payload = completed_review_payload(completion)
    review = tmp_path / "review.json"
    entry = payload["runs"][0]["wandb_completion"][0]
    entry["adopted_existing_result"] = True
    entry["scope_attestation"] = {
        "schema_version": 1,
        "confirmed": True,
        "confirmed_by": "yuya",
        "confirmed_at": "2026-06-27T23:30:00+09:00",
        "confirmation": "This existing run is within the reviewed one-model canary scope.",
        "review_path": str(review),
        "completion_path": str(completion),
        "benchmark": "agentic_math",
        "run_id": "run-1",
        "actual_cost_estimate": "$12.34",
        "provider_bill_reference": "billing export",
    }
    write_json(review, payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--required-wandb-benchmark",
            "agentic_math",
            "--required-wandb-run-id",
            "agentic_math=run-1",
            "--require-one-model-canary",
            "--fail-on-invalid",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    entry = report["gates"][0]["records"][0]["wandb_completion_entries"][0]
    assert entry["verified"] is False
    assert entry["scope_attestation_valid"] is False
    assert "scope_attestation source_attestation_json is required" in entry["verification_error"]


def test_paid_review_doctor_accepts_verified_weave_agents_review(tmp_path):
    completion = write_json(
        tmp_path / "agentic_math-run-1.json",
        wandb_completion_payload(),
    )
    weave_completion = write_weave_completion(tmp_path / "weave-agents-run-1.json")
    payload = completed_review_payload(completion)
    source_review = write_json(
        tmp_path / "review.before_weave_sync.json",
        completed_review_payload(completion),
    )
    source_review_sha256 = sha256_file(source_review)
    review = tmp_path / "review.json"
    weave_sync_dry_run = write_weave_sync_dry_run(
        tmp_path / "weave-agents-run-1.sync_dry_run.json",
        review_path=source_review,
        completion_path=weave_completion,
        source_review_sha256=source_review_sha256,
    )
    payload["verify_weave_agents"] = True
    payload["runs"][0]["weave_agents_completion"] = {
        "ok": True,
        "path": str(weave_completion),
        "agent_name": "nejumi-taiwan-openclaw",
        "run_id": "run-1",
        "sync_dry_run_report_json": str(weave_sync_dry_run),
        "sync_dry_run_source_review_json": str(source_review),
        "sync_dry_run_source_review_sha256": source_review_sha256,
    }
    write_json(review, payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--required-wandb-benchmark",
            "agentic_math",
            "--required-wandb-run-id",
            "agentic_math=run-1",
            "--require-one-model-canary",
            "--fail-on-invalid",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert (
        report["review_completion_requirements"]["weave_agents_completion_required_when"]
        == "verify_weave_agents=true for successful agentic runs"
    )
    gate = report["gates"][0]
    entry = gate["records"][0]["weave_agents_completion_entries"][0]
    assert entry["verified"] is True
    assert entry["schema_valid"] is True
    assert entry["checks_valid"] is True
    assert entry["query_source_valid"] is True
    assert entry["query_source"]["kind"] == "wandb_agents_api"
    assert entry["run_scope_proven"] is True
    assert entry["sync_dry_run_report_ok"] is True
    assert entry["sync_dry_run_report_present"] is True


def test_paid_review_doctor_rejects_weave_agents_missing_sync_dry_run(tmp_path):
    completion = write_json(
        tmp_path / "agentic_math-run-1.json",
        wandb_completion_payload(),
    )
    weave_completion = write_weave_completion(tmp_path / "weave-agents-run-1.json")
    payload = completed_review_payload(completion)
    payload["verify_weave_agents"] = True
    payload["runs"][0]["weave_agents_completion"] = {
        "ok": True,
        "path": str(weave_completion),
        "agent_name": "nejumi-taiwan-openclaw",
        "run_id": "run-1",
    }
    review = write_json(tmp_path / "review.json", payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--required-wandb-benchmark",
            "agentic_math",
            "--required-wandb-run-id",
            "agentic_math=run-1",
            "--require-one-model-canary",
            "--fail-on-invalid",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    entry = report["gates"][0]["records"][0]["weave_agents_completion_entries"][0]
    assert entry["verified"] is False
    assert entry["sync_dry_run_report_ok"] is False
    assert "missing Weave Agents sync dry-run report path" in entry["verification_error"]


def test_paid_review_doctor_rejects_weave_agents_sync_dry_run_mismatch(tmp_path):
    completion = write_json(
        tmp_path / "agentic_math-run-1.json",
        wandb_completion_payload(),
    )
    weave_completion = write_weave_completion(tmp_path / "weave-agents-run-1.json")
    review = tmp_path / "review.json"
    weave_sync_dry_run = write_weave_sync_dry_run(
        tmp_path / "weave-agents-run-1.sync_dry_run.json",
        review_path=review,
        completion_path=weave_completion,
    )
    dry_run_payload = json.loads(weave_sync_dry_run.read_text(encoding="utf-8"))
    dry_run_payload["entries"][0]["run_id"] = "other-run"
    dry_run_payload["changes"][0]["run_id"] = "other-run"
    weave_sync_dry_run.write_text(json.dumps(dry_run_payload), encoding="utf-8")
    payload = completed_review_payload(completion)
    payload["verify_weave_agents"] = True
    payload["runs"][0]["weave_agents_completion"] = {
        "ok": True,
        "path": str(weave_completion),
        "agent_name": "nejumi-taiwan-openclaw",
        "run_id": "run-1",
        "sync_dry_run_report_json": str(weave_sync_dry_run),
    }
    write_json(review, payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--required-wandb-benchmark",
            "agentic_math",
            "--required-wandb-run-id",
            "agentic_math=run-1",
            "--require-one-model-canary",
            "--fail-on-invalid",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    entry = report["gates"][0]["records"][0]["weave_agents_completion_entries"][0]
    assert entry["verified"] is False
    assert entry["sync_dry_run_report_ok"] is False
    assert "sync dry-run report entries must include this Weave entry" in entry["verification_error"]
    assert "sync dry-run report changes must include this Weave entry" in entry["verification_error"]


def test_paid_review_doctor_rejects_weave_agents_sync_dry_run_trace_mismatch(tmp_path):
    completion = write_json(
        tmp_path / "agentic_math-run-1.json",
        wandb_completion_payload(),
    )
    weave_completion = write_weave_completion(tmp_path / "weave-agents-run-1.json")
    review = tmp_path / "review.json"
    weave_sync_dry_run = write_weave_sync_dry_run(
        tmp_path / "weave-agents-run-1.sync_dry_run.json",
        review_path=review,
        completion_path=weave_completion,
    )
    dry_run_payload = json.loads(weave_sync_dry_run.read_text(encoding="utf-8"))
    dry_run_payload["entries"][0]["latest_trace_id"] = "other-trace"
    weave_sync_dry_run.write_text(json.dumps(dry_run_payload), encoding="utf-8")
    payload = completed_review_payload(completion)
    payload["verify_weave_agents"] = True
    payload["runs"][0]["weave_agents_completion"] = {
        "ok": True,
        "path": str(weave_completion),
        "agent_name": "nejumi-taiwan-openclaw",
        "run_id": "run-1",
        "sync_dry_run_report_json": str(weave_sync_dry_run),
    }
    write_json(review, payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--required-wandb-benchmark",
            "agentic_math",
            "--required-wandb-run-id",
            "agentic_math=run-1",
            "--require-one-model-canary",
            "--fail-on-invalid",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    entry = report["gates"][0]["records"][0]["weave_agents_completion_entries"][0]
    assert entry["verified"] is False
    assert entry["sync_dry_run_report_ok"] is False
    assert "sync dry-run report entry latest_trace_id does not match" in entry["verification_error"]


def test_paid_review_doctor_rejects_weave_agents_missing_query_source(tmp_path):
    completion = write_json(
        tmp_path / "agentic_math-run-1.json",
        wandb_completion_payload(),
    )
    weave_completion = write_weave_completion(tmp_path / "weave-agents-run-1.json")
    weave_payload = json.loads(weave_completion.read_text(encoding="utf-8"))
    weave_payload.pop("query_source")
    weave_completion.write_text(json.dumps(weave_payload), encoding="utf-8")
    payload = completed_review_payload(completion)
    payload["verify_weave_agents"] = True
    payload["runs"][0]["weave_agents_completion"] = {
        "ok": True,
        "path": str(weave_completion),
        "agent_name": "nejumi-taiwan-openclaw",
    }
    review = write_json(tmp_path / "review.json", payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--required-wandb-benchmark",
            "agentic_math",
            "--required-wandb-run-id",
            "agentic_math=run-1",
            "--require-one-model-canary",
            "--fail-on-invalid",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    entry = report["gates"][0]["records"][0]["weave_agents_completion_entries"][0]
    assert entry["verified"] is False
    assert entry["query_source_valid"] is False
    assert "query_source is not an object" in entry["verification_error"]


def test_paid_review_doctor_rejects_weave_agents_query_source_mismatch(tmp_path):
    completion = write_json(
        tmp_path / "agentic_math-run-1.json",
        {
            "ok": True,
            "benchmark": "agentic_math",
            "run_id": "run-1",
            "generated_at": time.time(),
            "verification_schema_version": 1,
            "observed_evidence": {
                "run_state": "finished",
                "summary_metrics": {
                    "agentic_math/accuracy": {"ok": True, "value": 0.86},
                },
            },
        },
    )
    weave_completion = write_weave_completion(tmp_path / "weave-agents-run-1.json")
    weave_payload = json.loads(weave_completion.read_text(encoding="utf-8"))
    weave_payload["query_source"]["kind"] = "manual_json"
    weave_payload["query_source"]["conversation_id_contains"] = "other-run"
    weave_payload["query_source"]["latest_trace_span_count"] = 99
    weave_completion.write_text(json.dumps(weave_payload), encoding="utf-8")
    payload = completed_review_payload(completion)
    payload["verify_weave_agents"] = True
    payload["runs"][0]["weave_agents_completion"] = {
        "ok": True,
        "path": str(weave_completion),
        "agent_name": "nejumi-taiwan-openclaw",
    }
    review = write_json(tmp_path / "review.json", payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--required-wandb-benchmark",
            "agentic_math",
            "--required-wandb-run-id",
            "agentic_math=run-1",
            "--require-one-model-canary",
            "--fail-on-invalid",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    entry = report["gates"][0]["records"][0]["weave_agents_completion_entries"][0]
    assert entry["verified"] is False
    assert entry["query_source_valid"] is False
    assert "query_source.kind mismatch" in entry["verification_error"]
    assert "query_source.conversation_id_contains does not match required_evidence" in entry["verification_error"]
    assert (
        "query_source.latest_trace_span_count does not match latest_trace_spans_chronological"
        in entry["verification_error"]
    )


def test_paid_review_doctor_rejects_weave_agents_missing_final_order_check(tmp_path):
    completion = write_json(
        tmp_path / "agentic_math-run-1.json",
        {
            "ok": True,
            "benchmark": "agentic_math",
            "run_id": "run-1",
            "generated_at": time.time(),
            "verification_schema_version": 1,
            "observed_evidence": {
                "run_state": "finished",
                "summary_metrics": {
                    "agentic_math/accuracy": {"ok": True, "value": 0.86},
                },
            },
        },
    )
    weave_completion = write_weave_completion(tmp_path / "weave-agents-run-1.json")
    weave_payload = json.loads(weave_completion.read_text(encoding="utf-8"))
    weave_payload["checks"] = [
        check
        for check in weave_payload["checks"]
        if check.get("name") != "trace_final_answer_order"
    ]
    weave_completion.write_text(json.dumps(weave_payload), encoding="utf-8")
    payload = completed_review_payload(completion)
    payload["verify_weave_agents"] = True
    payload["runs"][0]["weave_agents_completion"] = {
        "ok": True,
        "path": str(weave_completion),
        "agent_name": "nejumi-taiwan-openclaw",
    }
    review = write_json(tmp_path / "review.json", payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--required-wandb-benchmark",
            "agentic_math",
            "--required-wandb-run-id",
            "agentic_math=run-1",
            "--require-one-model-canary",
            "--fail-on-invalid",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    entry = report["gates"][0]["records"][0]["weave_agents_completion_entries"][0]
    assert entry["verified"] is False
    assert entry["checks_valid"] is False
    assert entry["missing_required_checks"] == ["trace_final_answer_order"]


def test_paid_review_doctor_rejects_weave_agents_missing_run_scope(tmp_path):
    completion = write_json(
        tmp_path / "agentic_math-run-1.json",
        {
            "ok": True,
            "benchmark": "agentic_math",
            "run_id": "run-1",
            "generated_at": time.time(),
            "verification_schema_version": 1,
            "observed_evidence": {
                "run_state": "finished",
                "summary_metrics": {
                    "agentic_math/accuracy": {"ok": True, "value": 0.86},
                },
            },
        },
    )
    weave_completion = write_weave_completion(tmp_path / "weave-agents-run-1.json")
    weave_payload = json.loads(weave_completion.read_text(encoding="utf-8"))
    weave_payload["required_evidence"]["conversation_id_contains"] = ""
    weave_completion.write_text(json.dumps(weave_payload), encoding="utf-8")
    payload = completed_review_payload(completion)
    payload["verify_weave_agents"] = True
    payload["runs"][0]["weave_agents_completion"] = {
        "ok": True,
        "path": str(weave_completion),
        "agent_name": "nejumi-taiwan-openclaw",
    }
    review = write_json(tmp_path / "review.json", payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--required-wandb-benchmark",
            "agentic_math",
            "--required-wandb-run-id",
            "agentic_math=run-1",
            "--require-one-model-canary",
            "--fail-on-invalid",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    entry = report["gates"][0]["records"][0]["weave_agents_completion_entries"][0]
    assert entry["verified"] is False
    assert entry["run_scope_proven"] is False
    assert "does not prove W&B run scope" in entry["verification_error"]


def test_paid_review_doctor_rejects_weave_agents_missing_final_order_requirement(tmp_path):
    completion = write_json(
        tmp_path / "agentic_math-run-1.json",
        {
            "ok": True,
            "benchmark": "agentic_math",
            "run_id": "run-1",
            "generated_at": time.time(),
            "verification_schema_version": 1,
            "observed_evidence": {
                "run_state": "finished",
                "summary_metrics": {
                    "agentic_math/accuracy": {"ok": True, "value": 0.86},
                },
            },
        },
    )
    weave_completion = write_weave_completion(tmp_path / "weave-agents-run-1.json")
    weave_payload = json.loads(weave_completion.read_text(encoding="utf-8"))
    weave_payload["required_evidence"].pop("trace_final_answer_order_required")
    weave_completion.write_text(json.dumps(weave_payload), encoding="utf-8")
    payload = completed_review_payload(completion)
    payload["verify_weave_agents"] = True
    payload["runs"][0]["weave_agents_completion"] = {
        "ok": True,
        "path": str(weave_completion),
        "agent_name": "nejumi-taiwan-openclaw",
    }
    review = write_json(tmp_path / "review.json", payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--required-wandb-benchmark",
            "agentic_math",
            "--required-wandb-run-id",
            "agentic_math=run-1",
            "--require-one-model-canary",
            "--fail-on-invalid",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    entry = report["gates"][0]["records"][0]["weave_agents_completion_entries"][0]
    assert entry["verified"] is False
    assert entry["trace_final_answer_order_required"] is False
    assert (
        "Weave Agents verifier JSON does not require trace_final_answer_order"
        in entry["verification_error"]
    )


def test_paid_review_doctor_rejects_weave_agents_missing_timestamp_quality_requirement(tmp_path):
    completion = write_json(
        tmp_path / "agentic_math-run-1.json",
        {
            "ok": True,
            "benchmark": "agentic_math",
            "run_id": "run-1",
            "generated_at": time.time(),
            "verification_schema_version": 1,
            "observed_evidence": {
                "run_state": "finished",
                "summary_metrics": {
                    "agentic_math/accuracy": {"ok": True, "value": 0.86},
                },
            },
        },
    )
    weave_completion = write_weave_completion(tmp_path / "weave-agents-run-1.json")
    weave_payload = json.loads(weave_completion.read_text(encoding="utf-8"))
    weave_payload["required_evidence"].pop("trace_timestamp_quality_required")
    weave_completion.write_text(json.dumps(weave_payload), encoding="utf-8")
    payload = completed_review_payload(completion)
    payload["verify_weave_agents"] = True
    payload["runs"][0]["weave_agents_completion"] = {
        "ok": True,
        "path": str(weave_completion),
        "agent_name": "nejumi-taiwan-openclaw",
    }
    review = write_json(tmp_path / "review.json", payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--required-wandb-benchmark",
            "agentic_math",
            "--required-wandb-run-id",
            "agentic_math=run-1",
            "--require-one-model-canary",
            "--fail-on-invalid",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    entry = report["gates"][0]["records"][0]["weave_agents_completion_entries"][0]
    assert entry["verified"] is False
    assert entry["trace_timestamp_quality_required"] is False
    assert (
        "Weave Agents verifier JSON does not require trace_timestamp_quality"
        in entry["verification_error"]
    )


def test_paid_review_doctor_rejects_weave_agents_missing_required_text_capture(tmp_path):
    completion = write_json(
        tmp_path / "agentic_math-run-1.json",
        {
            "ok": True,
            "benchmark": "agentic_math",
            "run_id": "run-1",
            "generated_at": time.time(),
            "verification_schema_version": 1,
            "observed_evidence": {
                "run_state": "finished",
                "summary_metrics": {
                    "agentic_math/accuracy": {"ok": True, "value": 0.86},
                },
            },
        },
    )
    weave_completion = write_weave_completion(tmp_path / "weave-agents-run-1.json")
    weave_payload = json.loads(weave_completion.read_text(encoding="utf-8"))
    weave_payload["required_evidence"]["required_texts"] = [
        "CANARY_ID",
        "CANARY_RESULT CANARY_ID 91",
    ]
    weave_payload["content_capture_health"]["required_text_count"] = 2
    weave_completion.write_text(json.dumps(weave_payload), encoding="utf-8")
    payload = completed_review_payload(completion)
    payload["verify_weave_agents"] = True
    payload["runs"][0]["weave_agents_completion"] = {
        "ok": True,
        "path": str(weave_completion),
        "agent_name": "nejumi-taiwan-openclaw",
    }
    review = write_json(tmp_path / "review.json", payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--required-wandb-benchmark",
            "agentic_math",
            "--required-wandb-run-id",
            "agentic_math=run-1",
            "--require-one-model-canary",
            "--fail-on-invalid",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    entry = report["gates"][0]["records"][0]["weave_agents_completion_entries"][0]
    assert entry["verified"] is False
    assert entry["required_text_capture_present"] is False
    assert "missing required_text_capture" in entry["verification_error"]


def test_paid_review_doctor_fails_for_missing_completion_verifier(tmp_path):
    review = write_json(
        tmp_path / "review.json",
        completed_review_payload(tmp_path / "missing.json"),
    )

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--required-wandb-benchmark",
            "agentic_math",
            "--require-one-model-canary",
            "--fail-on-invalid",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "paid_run_review_package" in payload["summary"]["blockers"]


def test_paid_review_doctor_fails_for_missing_weave_agents_verifier(tmp_path):
    completion = write_json(
        tmp_path / "agentic_math-run-1.json",
        {
            "ok": True,
            "benchmark": "agentic_math",
            "run_id": "run-1",
            "generated_at": time.time(),
            "verification_schema_version": 1,
            "observed_evidence": {
                "run_state": "finished",
                "summary_metrics": {
                    "agentic_math/accuracy": {"ok": True, "value": 0.86},
                },
            },
        },
    )
    payload = completed_review_payload(completion)
    payload["verify_weave_agents"] = True
    payload["runs"][0]["weave_agents_completion"] = {
        "ok": True,
        "path": str(tmp_path / "missing-weave.json"),
        "agent_name": "nejumi-taiwan-openclaw",
    }
    review = write_json(tmp_path / "review.json", payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--required-wandb-benchmark",
            "agentic_math",
            "--require-one-model-canary",
            "--fail-on-invalid",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    gate = report["gates"][0]
    assert any(
        "Weave Agents completion entry" in error
        for error in gate["records"][0]["errors"]
    )
