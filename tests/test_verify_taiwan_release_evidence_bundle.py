import importlib.util
import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "tools" / "build_taiwan_release_evidence_bundle.py"
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "tools" / "verify_taiwan_release_evidence_bundle.py"
INSTALLER_LOCK_JSON = "scripts/setup/nemoclaw_installer_lock.json"
INSTALLER_SHA256 = "a4ebc5710dfd8b10035968fd25562773ad56ec4c73ecf9bfe5c78c77724000e7"
AGENTIC_MATH_OUTPUT_COLUMNS = [
    "nemoclaw_session_audit_ok",
    "nemoclaw_session_audit",
    "conversation_order_ok",
    "conversation_order",
    "tool_policy_ok",
    "tool_policy_violations",
    "weave_sidecar_ok",
    "weave_sidecar",
]
WANDB_SCOPE_REQUIRED_HUMAN_FIELDS = [
    "scope_attestation_json.confirmed",
    "scope_attestation_json.confirmed_by",
    "scope_attestation_json.confirmed_at",
    "scope_attestation_json.confirmation",
    "scope_attestation_json.completion_sha256",
    "scope_attestation_json.actual_cost_estimate",
    "scope_attestation_json.provider_bill_reference",
]


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_verify_module():
    spec = importlib.util.spec_from_file_location(VERIFY_SCRIPT.stem, VERIFY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[VERIFY_SCRIPT.stem] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wandb_write_approval_requirement(entity: str = "test-entity", project: str = "test-project"):
    return {
        "required_before_wandb_write": True,
        "required_report_option": "--external-action-approval-report-json",
        "required_source_packet_option": "--external-action-approval-source-packet-json",
        "required_report_status": "approved",
        "required_source_bound": True,
        "required_source_packet_sha256_match": True,
        "required_requirements": ["wandb_access", "wandb_write"],
        "target_entity": entity,
        "target_project": project,
    }


def refresh_manifest_record_hash(bundle: Path, bundle_path: str) -> None:
    target = bundle / bundle_path
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["files"]:
        if record.get("bundle_path") == bundle_path:
            record["sha256"] = sha256(target)
            record["size_bytes"] = target.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def add_manifest_file_record(manifest: dict, bundle: Path, source: Path, role: str) -> None:
    bundle_path = Path("evidence") / source.name
    destination = bundle / bundle_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
    manifest.setdefault("files", []).append(
        {
            "source_path": str(source),
            "roles": ["release_gate_pointer", role],
            "exists": True,
            "bundle_path": str(bundle_path),
            "size_bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }
    )


def attach_valid_release_gate_pointer(bundle: Path, tmp_path: Path) -> dict:
    timestamp = "20260628T040000Z"
    release_gate_json = tmp_path / f"taiwan_release_gate_{timestamp}.json"
    latest_pointer_json = tmp_path / "latest_taiwan_release_gate.json"
    verification_json = tmp_path / f"latest_taiwan_release_gate_verify_{timestamp}.json"
    standalone_operator_plan_json = tmp_path / f"taiwan_release_operator_plan_{timestamp}.json"
    standalone_operator_plan_markdown = tmp_path / f"taiwan_release_operator_plan_{timestamp}.md"
    standalone_operator_plan = {
        "json": str(standalone_operator_plan_json),
        "markdown": str(standalone_operator_plan_markdown),
        "schema_version": 1,
        "status": "pending",
    }
    write_json(
        release_gate_json,
        {
            "schema_version": 1,
            "timestamp": timestamp,
            "release_gate_json": str(release_gate_json),
            "latest_pointer_json": str(latest_pointer_json),
            "latest_pointer_verification_json": str(verification_json),
            "operator_plan": standalone_operator_plan,
        },
    )
    write_json(
        latest_pointer_json,
        {
            "schema_version": 1,
            "timestamp": timestamp,
            "release_gate_json": str(release_gate_json),
            "latest_pointer_json": str(latest_pointer_json),
            "latest_pointer_verification_json": str(verification_json),
            "latest_pointer_verification_ok": True,
            "latest_pointer_verification_status": "passed",
            "latest_pointer_verification_issue_count": 0,
            "operator_plan": standalone_operator_plan,
        },
    )
    write_json(
        verification_json,
        {
            "schema_version": 1,
            "ok": True,
            "status": "passed",
            "issue_count": 0,
            "issues": [],
            "release_gate_json": str(release_gate_json),
            "pointer_json": str(latest_pointer_json),
        },
    )

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["timestamp"] = timestamp
    manifest["release_gate_pointer"] = {
        "release_gate_json": str(release_gate_json),
        "latest_pointer_json": str(latest_pointer_json),
        "latest_pointer_verification_json": str(verification_json),
        "latest_pointer_verification_ok": True,
        "latest_pointer_verification_status": "passed",
        "latest_pointer_verification_issue_count": 0,
        "operator_plan": standalone_operator_plan,
    }
    add_manifest_file_record(
        manifest,
        bundle,
        release_gate_json,
        "release_gate_pointer:release_gate_json",
    )
    add_manifest_file_record(
        manifest,
        bundle,
        latest_pointer_json,
        "release_gate_pointer:latest_pointer_json",
    )
    add_manifest_file_record(
        manifest,
        bundle,
        verification_json,
        "release_gate_pointer:latest_pointer_verification_json",
    )
    proof_path = bundle / "release_gate_pointer_proof.json"
    write_json(
        proof_path,
        {
            "schema_version": 1,
            "kind": "release_gate_pointer_proof",
            "ok": True,
            "status": "passed",
            "release_gate_pointer": manifest["release_gate_pointer"],
            "latest_pointer_verification": {
                "json": str(verification_json),
                "ok": True,
                "status": "passed",
                "issue_count": 0,
            },
        },
    )
    manifest["release_gate_pointer_proof"] = {
        "json": "release_gate_pointer_proof.json",
        "schema_version": 1,
        "status": "passed",
        "ok": True,
    }
    manifest.setdefault("files", []).append(
        {
            "source_path": str(proof_path),
            "roles": ["release_gate_pointer", "release_gate_pointer:proof_json"],
            "exists": True,
            "bundle_path": "release_gate_pointer_proof.json",
            "size_bytes": proof_path.stat().st_size,
            "sha256": sha256(proof_path),
        }
    )

    operator_plan_path = bundle / manifest["operator_plan"]["json"]
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    operator_plan["timestamp"] = timestamp
    operator_plan["source_release_gate_json"] = str(release_gate_json)
    operator_plan["release_gate_json"] = str(release_gate_json)
    operator_plan["release_gate_status"] = manifest["current_gate"]["status"]
    operator_plan["readiness_report_source"] = manifest["readiness_report_source"]
    operator_plan["release_gate_pointer"] = manifest["release_gate_pointer"]
    operator_plan["outputs"]["release_gate_json"] = str(release_gate_json)
    operator_plan["outputs"]["latest_pointer_json"] = str(latest_pointer_json)
    operator_plan["outputs"]["latest_pointer_verification_json"] = str(verification_json)
    renderer = operator_plan["operator_execution_plan_renderer"]
    renderer["release_gate_json_template"] = str(release_gate_json)
    renderer["safety"]["requires_release_gate_match_for_shell_script"] = True
    for key in ("review_command_template", "require_ready_command_template"):
        if "--release-gate-json" not in renderer[key]:
            renderer[key] = f"{renderer[key]} --release-gate-json {release_gate_json}"
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")

    for record in manifest["files"]:
        if record.get("bundle_path") == manifest["operator_plan"]["json"]:
            record["sha256"] = sha256(operator_plan_path)
            record["size_bytes"] = operator_plan_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "timestamp": timestamp,
        "release_gate_json": str(release_gate_json),
        "latest_pointer_json": str(latest_pointer_json),
        "verification_json": str(verification_json),
        "operator_plan": standalone_operator_plan,
    }


def agentic_math_wandb_completion_payload():
    return {
        "ok": True,
        "schema_version": 1,
        "verification_schema_version": 1,
        "status": "passed",
        "benchmark": "agentic_math",
        "run_id": "run-1",
        "entity": "test-entity",
        "project": "test-project",
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
        "required_evidence": {
            "run_state": "finished",
            "expected_total": 100,
            "nemoclaw_session_audit": {
                "required": True,
                "required_metric": "agentic_math/nemoclaw_session_audit_required_instances",
                "passed_metric": "agentic_math/nemoclaw_session_audit_passed_instances",
                "failed_metric": "agentic_math/nemoclaw_session_audit_failed_instances",
            },
            "summary_metrics": [
                "agentic_math/total_instances",
                "agentic_math/answered_instances",
                "agentic_math/correct_instances",
                "agentic_math/accuracy",
            ],
            "tables": [
                {
                    "name": "agentic_math_leaderboard_table",
                    "row_count": ">=1",
                },
                {
                    "name": "agentic_math_output_table",
                    "row_count": "must equal total metric",
                    "required_columns": AGENTIC_MATH_OUTPUT_COLUMNS,
                },
            ],
            "artifacts": [
                {
                    "type": "evaluation-results",
                    "required_aliases": ["production"],
                }
            ],
        },
        "observed_evidence": {
            "run_state": "finished",
            "expected_total": 100,
            "nemoclaw_session_audit": {
                "ok": True,
                "required": 100,
                "passed": 100,
                "failed": 0,
                "expected_total": 100,
            },
            "summary_metrics": {
                "agentic_math/total_instances": {
                    "ok": True,
                    "value": 100,
                },
                "agentic_math/answered_instances": {
                    "ok": True,
                    "value": 99,
                },
                "agentic_math/correct_instances": {
                    "ok": True,
                    "value": 86,
                },
                "agentic_math/accuracy": {
                    "ok": True,
                    "value": 0.86,
                },
            },
            "tables": [
                {
                    "name": "agentic_math_leaderboard_table",
                    "ok": True,
                    "nrows": 1,
                },
                {
                    "name": "agentic_math_output_table",
                    "ok": True,
                    "nrows": 100,
                    "columns_ok": True,
                    "columns": AGENTIC_MATH_OUTPUT_COLUMNS,
                    "required_columns": AGENTIC_MATH_OUTPUT_COLUMNS,
                    "missing_columns": [],
                    "columns_source": "summary",
                },
            ],
            "artifacts": [
                {
                    "name": "agentic-math-results:v0",
                    "type": "evaluation-results",
                    "aliases": ["latest", "production"],
                }
            ],
        },
        "checks": [
            {
                "name": "run_state",
                "ok": True,
                "detail": "run state is finished",
                "state": "finished",
            },
            {
                "name": "total_metric",
                "ok": True,
                "detail": "agentic_math/total_instances is present",
                "value": 100,
            },
            {
                "name": "leaderboard_table",
                "ok": True,
                "detail": "agentic_math_leaderboard_table has rows",
                "nrows": 1,
            },
            {
                "name": "output_table",
                "ok": True,
                "detail": "agentic_math_output_table row count matches total metric",
                "nrows": 100,
            },
            {
                "name": "output_table_columns",
                "ok": True,
                "detail": "agentic_math_output_table contains required observability columns",
                "table_name": "agentic_math_output_table",
                "required_columns": AGENTIC_MATH_OUTPUT_COLUMNS,
                "columns": AGENTIC_MATH_OUTPUT_COLUMNS,
                "missing_columns": [],
                "source": "summary",
            },
            {
                "name": "answered_metric",
                "ok": True,
                "detail": "agentic_math/answered_instances is present",
                "value": 99,
            },
            {
                "name": "correct_metric",
                "ok": True,
                "detail": "agentic_math/correct_instances is present",
                "value": 86,
            },
            {
                "name": "accuracy_metric",
                "ok": True,
                "detail": "agentic_math/accuracy equals correct/total",
                "value": 0.86,
            },
            {
                "name": "nemoclaw_session_audit",
                "ok": True,
                "required": 100,
                "passed": 100,
                "failed": 0,
                "expected_total": 100,
            },
            {
                "name": "result_artifact",
                "ok": True,
                "detail": "result artifact is logged",
                "artifacts": [
                    {
                        "name": "agentic-math-results:v0",
                        "type": "evaluation-results",
                        "aliases": ["latest", "production"],
                    }
                ],
            },
        ],
    }


def add_wandb_completion_run_metadata(payload):
    payload["required_evidence"]["run_metadata"] = {
        "config": [
            {
                "key": "model.pretrained_model_name_or_path",
                "expected": "gpt-4.1-mini-2025-04-14",
            },
            {"key": "run.agentic_math", "expected": True},
            {
                "key": "wandb.run_name",
                "expected": "taiwan/full/openai/gpt-4.1-mini: canary",
            },
        ],
        "tags": ["taiwan-canary"],
        "group": "tw-canary",
        "job_type": "evaluation",
    }
    payload["observed_evidence"]["run_metadata"] = {
        "config": [
            {
                "key": "model.pretrained_model_name_or_path",
                "present": True,
                "value": "gpt-4.1-mini-2025-04-14",
            },
            {"key": "run.agentic_math", "present": True, "value": True},
            {
                "key": "wandb.run_name",
                "present": True,
                "value": "taiwan/full/openai/gpt-4.1-mini: canary",
            },
        ],
        "tags": ["taiwan-canary"],
        "group": "tw-canary",
        "job_type": "evaluation",
    }
    payload["checks"].extend(
        [
            {
                "name": "run_config",
                "ok": True,
                "detail": "run.config model.pretrained_model_name_or_path matches expected value",
                "key": "model.pretrained_model_name_or_path",
                "expected": "gpt-4.1-mini-2025-04-14",
                "value": "gpt-4.1-mini-2025-04-14",
                "present": True,
            },
            {
                "name": "run_config",
                "ok": True,
                "detail": "run.config run.agentic_math matches expected value",
                "key": "run.agentic_math",
                "expected": True,
                "value": True,
                "present": True,
            },
            {
                "name": "run_config",
                "ok": True,
                "detail": "run.config wandb.run_name matches expected value",
                "key": "wandb.run_name",
                "expected": "taiwan/full/openai/gpt-4.1-mini: canary",
                "value": "taiwan/full/openai/gpt-4.1-mini: canary",
                "present": True,
            },
            {
                "name": "run_tag",
                "ok": True,
                "detail": "run tag is present",
                "tag": "taiwan-canary",
            },
            {
                "name": "run_group",
                "ok": True,
                "detail": "run group matches expected value",
                "value": "tw-canary",
            },
            {
                "name": "run_job_type",
                "ok": True,
                "detail": "run job_type matches expected value",
                "value": "evaluation",
            },
        ]
    )
    return payload


def build_bundle(tmp_path, *, readiness_ok=False):
    evidence = write_json(tmp_path / "evidence.json", {"ok": True})
    report = write_json(
        tmp_path / "readiness.json",
        {
            "schema_version": 1,
            "ok": readiness_ok,
            "status": "ready" if readiness_ok else "not_ready",
            "summary": {
                "gate_count": 1,
                "blocker_count": 0 if readiness_ok else 1,
                "blockers": [] if readiness_ok else ["example_gate"],
            },
            "gates": [
                {
                    "name": "example_gate",
                    "ok": readiness_ok,
                    "blocking": True,
                    "status": "passed" if readiness_ok else "failed",
                    "evidence_paths": [str(evidence)],
                }
            ],
        },
    )
    output_dir = tmp_path / "bundle"
    result = subprocess.run(
        [
            "python3",
            str(BUILD_SCRIPT),
            "--readiness-report",
            str(report),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return output_dir


def test_collect_wandb_completion_proofs_ignores_stale_or_blocked_candidates():
    module = load_verify_module()
    stale_path = "outputs/taiwan_full_eval/wandb_completion/agentic_math-stale.json"
    blocked_candidate_path = (
        "outputs/taiwan_full_eval/wandb_completion/agentic_math-blocked.json"
    )
    current_path = "outputs/taiwan_full_eval/wandb_completion/agentic_math-current.json"
    manifest = {
        "current_gate": {
            "benchmark_completion": [
                {
                    "benchmark": "agentic_math",
                    "standalone_ok": False,
                    "standalone_records": [
                        {"path": stale_path, "ok": True, "schema_valid": True}
                    ],
                    "review_ok": False,
                    "review_entries": [],
                },
                {
                    "benchmark": "agentic_math",
                    "standalone_ok": True,
                    "standalone_records": [
                        {"path": current_path, "ok": True, "schema_valid": True}
                    ],
                    "review_ok": False,
                    "review_entries": [],
                },
            ],
            "wandb_completion_contract": {
                "benchmarks": [
                    {
                        "benchmark": "agentic_math",
                        "standalone_completion_ok": False,
                        "standalone_completion_paths": [stale_path],
                        "review_completion_ok": False,
                        "review_completion_paths": [],
                        "formalized_existing_result": False,
                        "formalized_existing_completion_paths": [],
                    },
                    {
                        "benchmark": "agentic_math",
                        "standalone_completion_ok": True,
                        "standalone_completion_paths": [current_path],
                        "review_completion_ok": False,
                        "review_completion_paths": [],
                        "formalized_existing_result": False,
                        "formalized_existing_completion_paths": [],
                    },
                ]
            },
            "wandb_adoption_draft": {
                "candidates": [
                    {
                        "benchmark": "agentic_math",
                        "wandb_completion_json": blocked_candidate_path,
                        "sync_ready": False,
                    },
                    {
                        "benchmark": "agentic_math",
                        "wandb_completion_json": current_path,
                        "sync_ready": True,
                    },
                ]
            },
        }
    }

    proofs = module.collect_wandb_completion_proofs(manifest)

    assert stale_path not in proofs
    assert blocked_candidate_path not in proofs
    assert current_path in proofs


def build_bundle_with_operator_command_script(tmp_path):
    evidence = write_json(tmp_path / "evidence.json", {"ok": True})
    report = write_json(
        tmp_path / "readiness.json",
        {
            "schema_version": 1,
            "ok": False,
            "status": "not_ready",
            "summary": {
                "gate_count": 1,
                "blocker_count": 1,
                "blockers": ["example_gate"],
            },
            "gates": [
                {
                    "name": "example_gate",
                    "ok": False,
                    "blocking": True,
                    "status": "failed",
                    "evidence_paths": [str(evidence)],
                }
            ],
            "remediation_plan": [
                {
                    "gate": "example_gate",
                    "status": "failed",
                    "next_action": "rerun release gate",
                    "commands": [
                        "uv run python scripts/tools/run_taiwan_release_gate.py --quiet",
                        (
                            "uv run python scripts/tools/sync_weave_agents_completion_to_paid_review.py "
                            "--review-json outputs/taiwan_full_eval/PHASE_paid_run_review.json "
                            "--completion-json outputs/taiwan_full_eval/weave_agents_completion/PHASE-MODEL_SLUG.json "
                            "--run-id RUN_ID --in-place --set-verify-weave-agents "
                            "--validated-dry-run-report-json "
                            "temp/weave_agents_completion_PHASE_MODEL_SLUG.sync_dry_run.json"
                        ),
                    ],
                }
            ],
        },
    )
    output_dir = tmp_path / "bundle_with_operator_command_script"
    result = subprocess.run(
        [
            "python3",
            str(BUILD_SCRIPT),
            "--readiness-report",
            str(report),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return output_dir


def build_bundle_with_existing_results_relog_command_scripts(tmp_path):
    result_dir = tmp_path / "outputs" / "taiwan_full_eval" / "agentic_math" / "gpt-5_5"
    summary_json = write_json(result_dir / "summary.json", {"score": 1.0, "total": 2})
    results_jsonl = result_dir / "results.jsonl"
    results_jsonl.write_text('{"id": 1, "correct": true}\n{"id": 2, "correct": true}\n', encoding="utf-8")
    source_sha256 = {
        "summary_json": sha256(summary_json),
        "results_jsonl": sha256(results_jsonl),
    }
    relog_plan = write_json(
        tmp_path / "agentic_math_relog_plan.json",
        {
            "schema_version": 1,
            "ok": True,
            "will_write_wandb": False,
            "benchmark": "agentic_math",
            "entity": "test-entity",
            "project": "test-project",
            "source": {
                "results_dir": str(result_dir),
                "summary_json": str(summary_json),
                "results_jsonl": str(results_jsonl),
                "source_sha256": source_sha256,
            },
            "config": {"relog": {"source_sha256": source_sha256}},
            "would_log": {
                "tables": {"agentic_math_output_table": 2},
                "artifact": {"aliases": ["latest", "production"]},
            },
            "external_action_approval": wandb_write_approval_requirement(),
            "post_log_verifier_command_template": (
                "uv run python scripts/tools/verify_taiwan_wandb_completion.py "
                "--run-id RUN_ID_AFTER_WANDB_LOG --benchmark agentic_math "
                "--expected-total 100 --require-nemoclaw-session-audit "
                f"--expected-run-config relog.source_sha256.summary_json={source_sha256['summary_json']} "
                f"--expected-run-config relog.source_sha256.results_jsonl={source_sha256['results_jsonl']}"
            ),
        },
    )
    existing_results_audit = write_json(
        tmp_path / "existing_results_audit.json",
        {
            "ok": True,
            "status": "passed",
            "formalized_records": [
                {
                    "benchmark": "agentic_math",
                    "model_slug": "gpt-5_5",
                    "model": "openai/gpt-5.5",
                    "run_kind": "final",
                    "result_dir": str(result_dir),
                    "summary_path": str(summary_json),
                    "results_path": str(results_jsonl),
                    "complete_local": True,
                    "formalization_status": "formalized_wandb_complete",
                    "expected_total": 2,
                    "row_count": 2,
                    "relog_dry_run_plan_json": str(relog_plan),
                    "relog_dry_run_command": (
                        "uv run python scripts/tools/log_agentic_math_results_to_wandb.py "
                        f"--results-dir {result_dir} --model-name openai/gpt-5.5 "
                        f"--dry-run --plan-json {relog_plan}"
                    ),
                    "relog_command": (
                        "uv run python scripts/tools/log_agentic_math_results_to_wandb.py "
                        f"--results-dir {result_dir} --model-name openai/gpt-5.5 "
                        f"--validated-dry-run-plan-json {relog_plan} "
                        "--external-action-approval-source-packet-json "
                        "outputs/taiwan_release_evidence/bundle/external_action_approval_packet.json "
                        "--external-action-approval-report-json "
                        "temp/taiwan_external_action_approval_REVIEWED.verify.json"
                    ),
                    "warnings": [],
                    "errors": [],
                }
            ],
            "unformalized_complete_records": [],
            "partial_records": [],
        },
    )
    report = write_json(
        tmp_path / "readiness.json",
        {
            "schema_version": 1,
            "ok": False,
            "status": "not_ready",
            "summary": {
                "gate_count": 1,
                "blocker_count": 1,
                "blockers": ["wandb_completion"],
            },
            "runner": {
                "existing_results_audit": {
                    "path": str(existing_results_audit),
                    "ok": True,
                    "status": "passed",
                }
            },
            "gates": [
                {
                    "name": "wandb_completion",
                    "ok": False,
                    "blocking": True,
                    "status": "incomplete",
                    "evidence_paths": [],
                }
            ],
        },
    )
    output_dir = tmp_path / "bundle_with_existing_results_relog_command_scripts"
    result = subprocess.run(
        [
            "python3",
            str(BUILD_SCRIPT),
            "--readiness-report",
            str(report),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return output_dir


def build_bundle_with_operator_batch_command(tmp_path):
    report = write_json(
        tmp_path / "readiness.json",
        {
            "schema_version": 1,
            "ok": False,
            "status": "not_ready",
            "summary": {
                "gate_count": 1,
                "blocker_count": 1,
                "blockers": ["one_model_full_canary"],
            },
            "remediation_plan": [
                {
                    "gate": "one_model_full_canary",
                    "status": "incomplete",
                    "next_action": "run canary",
                    "commands": [
                        (
                            "uv run python scripts/tools/run_taiwan_full_eval_batch.py "
                            "--manifest configs/taiwan_openai_canary_models.yaml "
                            "--canary --phase agentic "
                            "--generated-config-dir configs/taiwan_full/generated_openai_canary_agentic_nemoclaw "
                            "--output-root outputs/taiwan_full_eval "
                            "--wandb-run-id-prefix twcanary-openai-mini-YYYYMMDD "
                            "--require-nemoclaw-agentic-config "
                            "--agentic-math-nemoclaw-sandbox nejumi-taiwan "
                            "--swebench-pro-nemoclaw-sandbox nejumi-taiwan "
                            "--swebench-pro-nemoclaw-checkout-transfer-mode copy "
                            "--verify-wandb-completion --verify-weave-agents "
                            "--weave-agents-require-tool-span "
                            "--weave-agents-require-tool-content "
                            "--weave-agents-require-usage "
                            "--weave-content-canary-gate WEAVE_CONTENT_CANARY_GATE "
                            "--require-weave-content-canary --yes "
                            "--agentic-math-nemoclaw-openclaw-config-path "
                            "/sandbox/.openclaw/openclaw.json "
                            "--swebench-pro-nemoclaw-openclaw-config-path "
                            "/sandbox/.openclaw/openclaw.json "
                            "--run-purpose 'OpenAI-direct one-model agentic phase' "
                            "--expected-cost-band 'approved canary cap' "
                            "--external-action-approval-source-packet-json "
                            "outputs/taiwan_release_evidence/bundle_YYYYMMDDTHHMM/"
                            "external_action_approval_packet.json "
                            "--external-action-approval-report-json "
                            "temp/taiwan_external_action_approval_REVIEWED_YYYYMMDDTHHMM.verify.json"
                        )
                    ],
                }
            ],
            "gates": [
                {
                    "name": "one_model_full_canary",
                    "ok": False,
                    "blocking": True,
                    "status": "incomplete",
                    "evidence_paths": [],
                    "required_wandb_benchmarks": ["agentic_math"],
                    "weave_agents_completion_required": True,
                    "completed_weave_agents_completion_phases": {},
                    "missing_required_weave_agents_completion_phases": [],
                }
            ],
        },
    )
    output_dir = tmp_path / "bundle_with_operator_batch_command"
    result = subprocess.run(
        [
            "python3",
            str(BUILD_SCRIPT),
            "--readiness-report",
            str(report),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return output_dir


def build_bundle_with_operator_weave_content_canary_command(tmp_path):
    report = write_json(
        tmp_path / "readiness.json",
        {
            "schema_version": 1,
            "ok": False,
            "status": "not_ready",
            "summary": {
                "gate_count": 1,
                "blocker_count": 1,
                "blockers": ["weave_content_canary"],
            },
            "remediation_plan": [
                {
                    "gate": "weave_content_canary",
                    "status": "failed",
                    "next_action": "rerun content canary",
                    "commands": [
                        (
                            "uv run python scripts/tools/run_weave_agents_content_canary.py "
                            "--canary-id PREPARE_ONLY "
                            "--model openai-direct/gpt-4.1-nano-2025-04-14 --thinking off "
                            "--nemoclaw-sandbox nejumi-taiwan "
                            "--nemoclaw-openclaw-config-path /sandbox/.openclaw/openclaw.json"
                        ),
                        (
                            "uv run python scripts/tools/run_weave_agents_content_canary.py "
                            "--execute --canary-id CONTENT_CANARY_YYYYMMDDTHHMM "
                            "--model openai-direct/gpt-4.1-nano-2025-04-14 "
                            "--thinking off --timeout 180 "
                            "--nemoclaw-sandbox nejumi-taiwan "
                            "--nemoclaw-openclaw-config-path /sandbox/.openclaw/openclaw.json "
                            "--external-action-approval-source-packet-json "
                            "outputs/taiwan_release_evidence/bundle_YYYYMMDDTHHMM/"
                            "external_action_approval_packet.json "
                            "--external-action-approval-report-json "
                            "temp/taiwan_external_action_approval_REVIEWED_YYYYMMDDTHHMM.verify.json"
                        ),
                        (
                            "uv run python scripts/tools/check_taiwan_canary_readiness.py "
                            "--weave-content-canary-gate "
                            "outputs/weave_agents_content_canary/plans/"
                            "weave_agents_content_canary_CONTENT_CANARY_YYYYMMDDTHHMM.gate.json "
                            "--require-weave-content-canary "
                            "--json outputs/taiwan_full_eval/openai_canary_readiness_weave_content.json"
                        ),
                    ],
                }
            ],
            "gates": [
                {
                    "name": "weave_content_canary",
                    "ok": False,
                    "blocking": True,
                    "status": "failed",
                    "evidence_paths": [],
                }
            ],
        },
    )
    output_dir = tmp_path / "bundle_with_operator_weave_content_canary_command"
    result = subprocess.run(
        [
            "python3",
            str(BUILD_SCRIPT),
            "--readiness-report",
            str(report),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return output_dir


def wandb_adoption_candidate_operator_handoff_payload(candidate):
    required_human_fields = (
        candidate.get("required_human_fields")
        if isinstance(candidate.get("required_human_fields"), list)
        else []
    )
    if candidate.get("sync_ready") is not True:
        steps = []
        refresh_command = candidate.get("refresh_wandb_completion_command")
        if isinstance(refresh_command, str) and refresh_command:
            steps.append(
                {
                    "step": "refresh_wandb_completion_verifier",
                    "command": refresh_command,
                    "expected_evidence_paths": [candidate.get("wandb_completion_json")],
                    "requires_external_action": False,
                    "requires_scope_confirmation": False,
                    "mutates_review_json": False,
                    "required": True,
                }
            )
        evidence_paths = [
            path
            for step in steps
            for path in step["expected_evidence_paths"]
            if isinstance(path, str) and path
        ]
        return {
            "available": False,
            "blocked_reason": str(candidate.get("sync_command_blocked_reason") or ""),
            "requires_scope_confirmation": bool(candidate.get("scope_attestation_required")),
            "required_human_fields": required_human_fields,
            "pending_human_field_count": len(required_human_fields),
            "step_count": len(steps),
            "required_step_count": sum(1 for step in steps if step["required"]),
            "command_count": sum(1 for step in steps if step.get("command")),
            "external_action_step_count": sum(
                1 for step in steps if step["requires_external_action"]
            ),
            "scope_confirmation_step_count": sum(
                1 for step in steps if step["requires_scope_confirmation"]
            ),
            "review_mutation_step_count": sum(
                1 for step in steps if step["mutates_review_json"]
            ),
            "evidence_path_count": len(evidence_paths),
            "expected_evidence_paths": evidence_paths,
            "steps": steps,
        }
    steps = [
        {
            "step": "confirm_scope_attestation",
            "command": None,
            "expected_evidence_paths": [candidate.get("scope_attestation_template_json")],
            "requires_external_action": True,
            "requires_scope_confirmation": True,
            "mutates_review_json": False,
            "required": True,
        },
        {
            "step": "render_confirmed_attestation",
            "command": candidate.get("scope_attestation_render_command"),
            "expected_evidence_paths": [
                candidate.get("scope_attestation_template_json"),
                candidate.get("scope_attestation_render_report_json"),
                candidate.get("scope_attestation_render_markdown"),
            ],
            "requires_external_action": False,
            "requires_scope_confirmation": False,
            "mutates_review_json": False,
            "required": True,
        },
        {
            "step": "preflight_scope_attestation",
            "command": candidate.get("scope_attestation_preflight_command"),
            "expected_evidence_paths": [
                candidate.get("scope_attestation_preflight_report_json")
            ],
            "requires_external_action": False,
            "requires_scope_confirmation": False,
            "mutates_review_json": False,
            "required": True,
        },
        {
            "step": "sync_paid_review_dry_run",
            "command": candidate.get("sync_dry_run_command"),
            "expected_evidence_paths": [candidate.get("sync_dry_run_report_json")],
            "requires_external_action": False,
            "requires_scope_confirmation": False,
            "mutates_review_json": False,
            "required": True,
        },
        {
            "step": "sync_paid_review_apply",
            "command": candidate.get("sync_apply_command") or candidate.get("sync_command"),
            "expected_evidence_paths": [candidate.get("target_review_json")],
            "requires_external_action": False,
            "requires_scope_confirmation": False,
            "mutates_review_json": True,
            "required": True,
        },
    ]
    normalized = []
    for step in steps:
        normalized.append(
            {
                **step,
                "expected_evidence_paths": [
                    path
                    for path in step["expected_evidence_paths"]
                    if isinstance(path, str) and path
                ],
            }
        )
    evidence_paths = []
    for step in normalized:
        for path in step["expected_evidence_paths"]:
            if path not in evidence_paths:
                evidence_paths.append(path)
    return {
        "available": all(
            step.get("command") or step["step"] == "confirm_scope_attestation"
            for step in normalized
        ),
        "blocked_reason": "",
        "requires_scope_confirmation": bool(candidate.get("scope_attestation_required")),
        "required_human_fields": required_human_fields,
        "pending_human_field_count": len(required_human_fields),
        "step_count": len(normalized),
        "required_step_count": sum(1 for step in normalized if step["required"]),
        "command_count": sum(1 for step in normalized if step.get("command")),
        "external_action_step_count": sum(
            1 for step in normalized if step["requires_external_action"]
        ),
        "scope_confirmation_step_count": sum(
            1 for step in normalized if step["requires_scope_confirmation"]
        ),
        "review_mutation_step_count": sum(
            1 for step in normalized if step["mutates_review_json"]
        ),
        "evidence_path_count": len(evidence_paths),
        "expected_evidence_paths": evidence_paths,
        "steps": normalized,
    }


def wandb_adoption_operator_handoff_payload(candidates):
    aggregate = {
        "candidate_count": 0,
        "available_candidate_count": 0,
        "step_count": 0,
        "command_count": 0,
        "external_action_step_count": 0,
        "scope_confirmation_step_count": 0,
        "review_mutation_step_count": 0,
        "evidence_path_count": 0,
    }
    evidence_paths = []
    for candidate in candidates:
        handoff = candidate["operator_handoff"]
        aggregate["candidate_count"] += 1
        if handoff["available"]:
            aggregate["available_candidate_count"] += 1
        for field in (
            "step_count",
            "command_count",
            "external_action_step_count",
            "scope_confirmation_step_count",
            "review_mutation_step_count",
        ):
            aggregate[field] += handoff[field]
        for path in handoff["expected_evidence_paths"]:
            if path not in evidence_paths:
                evidence_paths.append(path)
    aggregate["evidence_path_count"] = len(evidence_paths)
    aggregate["expected_evidence_paths"] = evidence_paths
    return aggregate


def wandb_adoption_draft_markdown_cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    return text.replace("\n", " ").replace("|", "\\|")


def wandb_adoption_draft_markdown_payload(draft):
    lines = [
        "# Existing W&B Adoption Review Draft",
        "",
        f"Status: `{draft.get('status')}`",
        f"Candidates: `{draft.get('candidate_count')}`",
        f"Source audit: `{draft.get('source_audit_json')}`",
        f"Source audit SHA256: `{draft.get('source_audit_sha256')}`",
        "",
        "## Required Human Fields",
        "",
    ]
    fields = draft.get("required_human_fields")
    if isinstance(fields, list) and fields:
        lines.extend(f"- `{field}`" for field in fields)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Candidates",
            "",
            "| Benchmark | Model | Run ID | Schema | Observed evidence | Run metadata | Sync-ready | Target review | Completion JSON | Attestation template | Metrics |",
            "|---|---|---|---:|---:|---:|---:|---|---|---|---|",
        ]
    )
    candidates = draft.get("candidates")
    if isinstance(candidates, list) and candidates:
        for row in candidates:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        wandb_adoption_draft_markdown_cell(row.get("benchmark")),
                        wandb_adoption_draft_markdown_cell(row.get("model_slug")),
                        wandb_adoption_draft_markdown_cell(row.get("wandb_run_id")),
                        wandb_adoption_draft_markdown_cell(
                            row.get("verification_schema_version")
                        ),
                        wandb_adoption_draft_markdown_cell(
                            row.get("observed_evidence_present")
                        ),
                        wandb_adoption_draft_markdown_cell(
                            row.get("run_metadata_valid")
                        ),
                        wandb_adoption_draft_markdown_cell(row.get("sync_ready")),
                        wandb_adoption_draft_markdown_cell(
                            row.get("target_review_json")
                        ),
                        wandb_adoption_draft_markdown_cell(
                            row.get("wandb_completion_json")
                        ),
                        wandb_adoption_draft_markdown_cell(
                            row.get("scope_attestation_template_json")
                        ),
                        wandb_adoption_draft_markdown_cell(row.get("metrics")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("| none |  |  |  |  |  |  |  |  |  |  |")
    lines.extend(["", "## Commands", ""])
    if isinstance(candidates, list) and candidates:
        for index, row in enumerate(candidates, start=1):
            if not isinstance(row, dict):
                continue
            handoff = row.get("operator_handoff") if isinstance(row.get("operator_handoff"), dict) else {}
            lines.extend(
                [
                    f"### Candidate {index}",
                    "",
                    (
                        "- Operator handoff available: "
                        f"`{wandb_adoption_draft_markdown_cell(handoff.get('available'))}`"
                    ),
                    (
                        "- Handoff steps: "
                        f"`{wandb_adoption_draft_markdown_cell(handoff.get('step_count'))}`"
                    ),
                    (
                        "- Handoff commands: "
                        f"`{wandb_adoption_draft_markdown_cell(handoff.get('command_count'))}`"
                    ),
                    (
                        "- Scope confirmation steps: "
                        f"`{wandb_adoption_draft_markdown_cell(handoff.get('scope_confirmation_step_count'))}`"
                    ),
                    (
                        "- Evidence paths: "
                        f"`{wandb_adoption_draft_markdown_cell(handoff.get('evidence_path_count'))}`"
                    ),
                    "",
                    f"#### Candidate {index} Handoff Steps",
                    "",
                    "| Step | Required | External action | Scope confirmation | Review mutation | Command | Expected evidence paths |",
                    "|---|---:|---:|---:|---:|---|---|",
                ]
            )
            steps = handoff.get("steps") if isinstance(handoff.get("steps"), list) else []
            if steps:
                for step in steps:
                    if not isinstance(step, dict):
                        continue
                    lines.append(
                        "| "
                        + " | ".join(
                            [
                                wandb_adoption_draft_markdown_cell(step.get("step")),
                                wandb_adoption_draft_markdown_cell(step.get("required")),
                                wandb_adoption_draft_markdown_cell(
                                    step.get("requires_external_action")
                                ),
                                wandb_adoption_draft_markdown_cell(
                                    step.get("requires_scope_confirmation")
                                ),
                                wandb_adoption_draft_markdown_cell(
                                    step.get("mutates_review_json")
                                ),
                                wandb_adoption_draft_markdown_cell(step.get("command")),
                                wandb_adoption_draft_markdown_cell(
                                    step.get("expected_evidence_paths")
                                ),
                            ]
                        )
                        + " |"
                    )
            else:
                lines.append("| none |  |  |  |  |  |  |")
            lines.append("")
    else:
        lines.append("- none")
    return "\n".join(lines)


def build_bundle_with_wandb_adoption_draft(
    tmp_path,
    *,
    include_scope_render=False,
    include_scope_preflight=False,
    include_sync_dry_run=False,
    include_unconfirmed_checks=False,
    include_weave_agents_adoption_validation_failures=False,
):
    taxonomy = tmp_path / "taxonomy.yaml"
    taxonomy.write_text("version: test\n", encoding="utf-8")
    completion = write_json(
        tmp_path / "wandb_completion.json",
        {**agentic_math_wandb_completion_payload(), "taxonomy_path": str(taxonomy)},
    )
    completion_sha = sha256(completion)
    target_review = write_json(
        tmp_path / "canary_agentic_paid_run_review.json",
        {"status": "prepared", "phase": "agentic", "runs": []},
    )
    target_review_json = str(target_review)
    target_review_sha = sha256(target_review)
    source_audit = write_json(
        tmp_path / "existing_results_audit.json",
        {
            "ok": True,
            "status": "passed",
            "formalized_records": [
                {
                    "benchmark": "agentic_math",
                    "model_slug": "gpt-5_5",
                    "model": "gpt-5.5",
                    "run_kind": "canary",
                    "result_dir": "outputs/taiwan_full_eval/agentic_math/gpt-5_5",
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
            "wandb_completion_records": [
                {
                    "path": str(completion),
                    "ok": True,
                    "benchmark": "agentic_math",
                    "entity": "test-entity",
                    "project": "test-project",
                    "run_id": "run-1",
                    "verification_schema_version": 1,
                    "schema_current": True,
                    "schema_current_issues": [],
                    "observed_evidence_present": True,
                }
            ],
        },
    )
    source_audit_sha = sha256(source_audit)
    attestation_payload = {
        "schema_version": 1,
        "confirmed": False,
        "confirmed_by": "REVIEWER",
        "confirmed_at": "YYYY-MM-DDTHH:MM:SS+09:00",
        "confirmation": "This W&B run is the reviewed canary scope for agentic_math.",
        "review_path": target_review_json,
        "completion_path": str(completion),
        "completion_sha256": completion_sha,
        "source_audit_json": str(source_audit),
        "source_audit_sha256": source_audit_sha,
        "benchmark": "agentic_math",
        "entity": "test-entity",
        "project": "test-project",
        "run_id": "run-1",
        "actual_cost_estimate": "$ACTUAL_OR_BILLING_ESTIMATE",
        "provider_bill_reference": "BILL_OR_DASHBOARD_REFERENCE",
    }
    source_template_sha = "1" * 64
    if include_scope_render or include_scope_preflight or include_sync_dry_run:
        attestation_payload.update(
            {
                "confirmed": True,
                "confirmed_by": "yuya",
                "confirmed_at": "2026-06-28T01:30:00+09:00",
                "confirmation": "This W&B run is the reviewed Agentic Math canary scope.",
                "actual_cost_estimate": "$12.34",
                "provider_bill_reference": "openai-dashboard-2026-06-28",
            }
        )
    if include_scope_render:
        attestation_payload["rendered_scope_attestation"] = {
            "schema_version": 1,
            "generated_at": 1,
            "source_template_json": str(tmp_path / "agentic_math-run-1.scope_attestation.json"),
            "source_template_sha256": source_template_sha,
            "output_json": str(tmp_path / "agentic_math-run-1.scope_attestation.json"),
            "will_execute_external_actions": False,
            "safety": {
                "executes_external_action": False,
                "queries_wandb": False,
                "writes_wandb": False,
                "installs_third_party": False,
                "launches_model_inference": False,
                "mutates_paid_review": False,
            },
        }
    attestation_template = write_json(
        tmp_path / "agentic_math-run-1.scope_attestation.json",
        attestation_payload,
    )
    scope_preflight_report_path = tmp_path / "agentic_math-run-1.scope_preflight.json"
    sync_dry_run_report_path = tmp_path / "agentic_math-run-1.sync_dry_run.json"
    scope_render_report_path = tmp_path / "agentic_math-run-1.scope_attestation.render.json"
    scope_render_markdown_path = tmp_path / "agentic_math-run-1.scope_attestation.render.md"
    scope_render_report = None
    scope_render_markdown = None
    if include_scope_render:
        preflight_command = (
            "uv run python scripts/tools/verify_wandb_scope_attestation.py "
            f"--review-json {target_review_json} "
            f"--completion-json {completion} "
            f"--scope-attestation-json {attestation_template} "
            f"--json {scope_preflight_report_path}"
        )
        sync_dry_run_command = (
            "uv run python scripts/tools/sync_wandb_completion_to_paid_review.py "
            f"--review-json {target_review_json} "
            f"--completion-json {completion} "
            "--set-verify-wandb-completion --top-level --adopt-existing-result "
            f"--scope-attestation-json {attestation_template} "
            f"--report-json {sync_dry_run_report_path}"
        )
        scope_render_payload = {
            "schema_version": 1,
            "ok": True,
            "status": "rendered",
            "generated_at": 1,
            "template_json": str(attestation_template),
            "template_sha256": source_template_sha,
            "output_json": str(attestation_template),
            "output_sha256": sha256(attestation_template),
            "review_path": target_review_json,
            "completion_path": str(completion),
            "completion_sha256": completion_sha,
            "source_audit_json": str(source_audit),
            "source_audit_sha256": source_audit_sha,
            "benchmark": "agentic_math",
            "entity": "test-entity",
            "project": "test-project",
            "run_id": "run-1",
            "will_execute_external_actions": False,
            "safety": {
                "executes_external_action": False,
                "queries_wandb": False,
                "writes_wandb": False,
                "installs_third_party": False,
                "launches_model_inference": False,
                "mutates_paid_review": False,
            },
            "next_commands": {
                "preflight": preflight_command,
                "sync_dry_run": sync_dry_run_command,
            },
            "errors": [],
        }
        scope_render_report = write_json(
            scope_render_report_path,
            scope_render_payload,
        )
        scope_render_markdown = scope_render_markdown_path
        scope_render_markdown.write_text(
            "\n".join(
                [
                    "# W&B Scope Attestation Render Report",
                    "",
                    "Status: `rendered`",
                    "OK: `True`",
                    "Will execute external actions: `False`",
                    "",
                    "| Field | Value |",
                    "|---|---|",
                    f"| template_json | {attestation_template} |",
                    f"| template_sha256 | {source_template_sha} |",
                    f"| output_json | {attestation_template} |",
                    f"| output_sha256 | {sha256(attestation_template)} |",
                    f"| review_path | {target_review_json} |",
                    f"| completion_path | {completion} |",
                    f"| completion_sha256 | {completion_sha} |",
                    f"| source_audit_json | {source_audit} |",
                    f"| source_audit_sha256 | {source_audit_sha} |",
                    "| benchmark | agentic_math |",
                    "| entity | test-entity |",
                    "| project | test-project |",
                    "| run_id | run-1 |",
                    "",
                    "## Next Commands",
                    "",
                    "### preflight",
                    "",
                    "```bash",
                    preflight_command,
                    "```",
                    "",
                    "### sync_dry_run",
                    "",
                    "```bash",
                    sync_dry_run_command,
                    "```",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    scope_preflight_report = None
    if include_scope_preflight:
        scope_preflight_report = write_json(
            scope_preflight_report_path,
            {
                "ok": True,
                "status": "passed",
                "generated_at": 1,
                "review_path": target_review_json,
                "completion_json": str(completion),
                "scope_attestation_json": str(attestation_template),
                "source_files": {
                    "review_json": {
                        "path": target_review_json,
                        "readable": True,
                        "sha256": target_review_sha,
                    },
                    "completion_json": {
                        "path": str(completion),
                        "readable": True,
                        "sha256": completion_sha,
                    },
                    "scope_attestation_json": {
                        "path": str(attestation_template),
                        "readable": True,
                        "sha256": sha256(attestation_template),
                    },
                },
                "entry": {
                    "benchmark": "agentic_math",
                    "ok": True,
                    "entity": "test-entity",
                    "project": "test-project",
                    "run_id": "run-1",
                    "path": str(completion),
                    "sha256": completion_sha,
                    "verification_schema_version": 1,
                    "observed_evidence_valid": True,
                    "run_metadata_valid": True,
                    "adopted_existing_result": False,
                    "query_source_kind": "wandb_sdk",
                    "query_source_run_path": "test-entity/test-project/run-1",
                },
                "scope_attestation": {
                    **attestation_payload,
                    "source_attestation_json": str(attestation_template),
                    "source_attestation_sha256": sha256(attestation_template),
                },
                "errors": [],
            },
        )
    sync_dry_run_report = None
    if include_sync_dry_run:
        sync_dry_run_report = write_json(
            sync_dry_run_report_path,
            {
                "ok": True,
                "status": "synced",
                "generated_at": 1,
                "review_path": target_review_json,
                "source_review_sha256": target_review_sha,
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
                        "path": str(completion),
                        "sha256": completion_sha,
                        "verification_schema_version": 1,
                        "observed_evidence_valid": True,
                        "adopted_existing_result": True,
                        "scope_attestation": {
                            **attestation_payload,
                            "source_attestation_json": str(attestation_template),
                            "source_attestation_sha256": sha256(attestation_template),
                        },
                    }
                ],
                "change_count": 1,
                "unmatched_count": 0,
                "changes": [
                    {
                        "target": "top_level",
                        "action": "added",
                        "benchmark": "agentic_math",
                        "run_id": "run-1",
                    }
                ],
                "unmatched_entries": [],
                "before_status": "prepared",
                "after_status": "prepared",
                "verify_wandb_completion": True,
            },
        )
    candidate = {
        "benchmark": "agentic_math",
        "model_slug": "gpt-5_5",
        "wandb_entity": "test-entity",
        "wandb_project": "test-project",
        "wandb_run_id": "run-1",
        "wandb_completion_json": str(completion),
        "wandb_completion_sha256": completion_sha,
        "source_audit_json": str(source_audit),
        "source_audit_sha256": source_audit_sha,
        "target_review_json": target_review_json,
        "scope_attestation_template_json": str(attestation_template),
        "scope_attestation_required": True,
        "required_human_fields": list(WANDB_SCOPE_REQUIRED_HUMAN_FIELDS),
    }
    if include_scope_render or include_unconfirmed_checks or scope_preflight_report is not None:
        candidate["scope_attestation_preflight_report_json"] = str(
            scope_preflight_report_path
        )
    if include_scope_render or include_unconfirmed_checks:
        candidate["scope_attestation_render_report_json"] = str(scope_render_report_path)
        candidate["scope_attestation_render_markdown"] = str(scope_render_markdown_path)
    if include_scope_render or include_unconfirmed_checks or sync_dry_run_report is not None:
        candidate["sync_dry_run_report_json"] = str(sync_dry_run_report_path)
    if include_scope_render or include_unconfirmed_checks:
        candidate["scope_attestation_render_command"] = (
            "uv run python scripts/tools/render_wandb_scope_attestation.py "
            f"--template-json {attestation_template} "
            f"--output-json {attestation_template} "
            "--confirmed-by REVIEWER_NAME "
            "--confirmed-at YYYY-MM-DDTHH:MM:SS+09:00 "
            "--confirmation CONFIRM_THIS_WANDB_RUN_IS_THE_REVIEWED_SCOPE "
            "--actual-cost-estimate ACTUAL_COST_USD "
            "--provider-bill-reference PROVIDER_BILL_REFERENCE "
            f"--report-json {scope_render_report_path} "
            f"--markdown {scope_render_markdown_path} "
            f"--preflight-report-json {scope_preflight_report_path} "
            f"--sync-dry-run-report-json {sync_dry_run_report_path}"
        )
    if include_unconfirmed_checks:
        candidate["sync_ready"] = True
    candidate["operator_handoff"] = wandb_adoption_candidate_operator_handoff_payload(
        candidate
    )
    operator_handoff = wandb_adoption_operator_handoff_payload([candidate])
    draft_path = tmp_path / "wandb_adoption_draft.json"
    draft_md = tmp_path / "wandb_adoption_draft.md"
    draft_payload = {
        "schema_version": 1,
        "path": str(draft_path),
        "markdown_path": str(draft_md),
        "ok": True,
        "status": "candidates_pending_scope_confirmation",
        "source_audit_json": str(source_audit),
        "source_audit_sha256": source_audit_sha,
        "candidate_count": 1,
        "scope_attestation_template_count": 1,
        "scope_attestation_templates": [
            {
                "benchmark": "agentic_math",
                "run_id": "run-1",
                "path": str(attestation_template),
            }
        ],
        "operator_handoff": operator_handoff,
        "requires_human_scope_confirmation": True,
        "required_human_fields": list(WANDB_SCOPE_REQUIRED_HUMAN_FIELDS),
        "candidates": [candidate],
    }
    draft = write_json(
        draft_path,
        draft_payload,
    )
    draft_md.write_text(
        wandb_adoption_draft_markdown_payload(draft_payload) + "\n",
        encoding="utf-8",
    )
    unconfirmed_checks = None
    if include_unconfirmed_checks:
        preflight_report = write_json(
            tmp_path / "agentic_math-run-1.unconfirmed_scope_preflight.validation_failed.json",
            {
                "ok": False,
                "status": "validation_failed",
                "generated_at": 1,
                "review_path": target_review_json,
                "completion_json": str(completion),
                "scope_attestation_json": str(attestation_template),
                "source_files": {
                    "review_json": {
                        "path": target_review_json,
                        "readable": True,
                        "sha256": target_review_sha,
                    },
                    "completion_json": {
                        "path": str(completion),
                        "readable": True,
                        "sha256": completion_sha,
                    },
                    "scope_attestation_json": {
                        "path": str(attestation_template),
                        "readable": True,
                        "sha256": sha256(attestation_template),
                    },
                },
                "entry": {
                    "benchmark": "agentic_math",
                    "ok": True,
                    "entity": "test-entity",
                    "project": "test-project",
                    "run_id": "run-1",
                    "path": str(completion),
                    "sha256": completion_sha,
                    "verification_schema_version": 1,
                    "observed_evidence_valid": True,
                    "run_metadata_valid": True,
                    "adopted_existing_result": False,
                    "query_source_kind": "wandb_sdk",
                    "query_source_run_path": "test-entity/test-project/run-1",
                },
                "scope_attestation": None,
                "errors": [
                    "scope attestation is not valid: confirmed must be true; "
                    "actual_cost_estimate must not be a placeholder; "
                    "provider_bill_reference must not be a placeholder"
                ],
            },
        )
        unconfirmed_sync_report = write_json(
            tmp_path / "agentic_math-run-1.unconfirmed_sync_dry_run.validation_failed.json",
            {
                "ok": False,
                "status": "validation_failed",
                "generated_at": 1,
                "review_path": target_review_json,
                "output_path": "",
                "in_place": False,
                "dry_run": True,
                "entry_count": 0,
                "adopted_existing_result_count": 0,
                "entries": [],
                "change_count": 0,
                "unmatched_count": 0,
                "changes": [],
                "unmatched_entries": [],
                "verify_wandb_completion": False,
                "errors": [
                    "scope attestation is not valid: confirmed must be true; "
                    "actual_cost_estimate must not be a placeholder; "
                    "provider_bill_reference must not be a placeholder"
                ],
            },
        )
        unconfirmed_checks = {
            "ok": True,
            "status": "passed",
            "record_count": 1,
            "output_dir": str(tmp_path),
            "records": [
                {
                    "benchmark": "agentic_math",
                    "run_id": "run-1",
                    "scope_attestation_template_json": str(attestation_template),
                    "preflight_report_json": str(preflight_report),
                    "preflight_returncode": 1,
                    "preflight_failed_as_expected": True,
                    "preflight_status": "validation_failed",
                    "sync_dry_run_report_json": str(unconfirmed_sync_report),
                    "sync_dry_run_returncode": 1,
                    "sync_dry_run_failed_as_expected": True,
                    "sync_dry_run_status": "validation_failed",
                    "will_query_wandb": False,
                    "will_write_wandb": False,
                    "will_launch_model_inference": False,
                    "will_mutate_review_json": False,
                }
            ],
        }
    weave_agents_adoption_validation_failures = None
    if include_weave_agents_adoption_validation_failures:
        failed_weave_verifier = write_json(
            tmp_path / "weave_agents_verify_run_1_failed.json",
            {
                "ok": False,
                "verification_schema_version": 1,
                "agent_name": "nejumi-taiwan-openclaw",
                "project_id": "llm-leaderboard/tc-leaderboard",
                "query_source": {
                    "kind": "wandb_agents_api",
                    "conversation_id_contains": "run-1",
                    "matching_span_count": 0,
                },
                "checks": [
                    {
                        "name": "spans_present",
                        "ok": False,
                        "detail": "no matching spans returned by Agents API",
                    }
                ],
            },
        )
        failed_weave_sync_report = write_json(
            tmp_path / "weave_agents_sync_run_1.validation_failed.json",
            {
                "ok": False,
                "status": "validation_failed",
                "generated_at": 1,
                "review_path": target_review_json,
                "source_review_sha256": target_review_sha,
                "output_path": "",
                "in_place": False,
                "dry_run": True,
                "entry_count": 0,
                "entries": [],
                "change_count": 0,
                "unmatched_count": 0,
                "changes": [],
                "unmatched_entries": [],
                "completion_paths": [str(failed_weave_verifier)],
                "validation_errors": [
                    f"{failed_weave_verifier} is not a passing Weave Agents verifier JSON"
                ],
                "verify_weave_agents": False,
            },
        )
        weave_agents_adoption_validation_failures = {
            "ok": False,
            "status": "validation_failed_reports_present",
            "record_count": 1,
            "glob": "temp/weave_agents_sync_*.validation_failed.json",
            "records": [
                {
                    "path": str(failed_weave_sync_report),
                    "ok": False,
                    "status": "validation_failed",
                    "review_path": target_review_json,
                    "completion_paths": [str(failed_weave_verifier)],
                    "validation_errors": [
                        f"{failed_weave_verifier} is not a passing Weave Agents verifier JSON"
                    ],
                    "dry_run": True,
                    "in_place": False,
                    "entry_count": 0,
                    "change_count": 0,
                }
            ],
        }
    report = write_json(
        tmp_path / "readiness.json",
        {
            "schema_version": 1,
            "ok": False,
            "status": "not_ready",
            "summary": {
                "gate_count": 1,
                "blocker_count": 1,
                "blockers": ["wandb_completion"],
            },
            "runner": {
                "report_json": str(tmp_path / "readiness.json"),
                "wandb_adoption_draft": {
                    "path": str(draft),
                    "markdown_path": str(draft_md),
                    "ok": True,
                    "status": "candidates_pending_scope_confirmation",
                    "summary": {
                        "candidate_count": 1,
                        "requires_human_scope_confirmation": True,
                    },
                },
                **(
                    {"wandb_adoption_unconfirmed_checks": unconfirmed_checks}
                    if isinstance(unconfirmed_checks, dict)
                    else {}
                ),
                **(
                    {
                        "weave_agents_adoption_validation_failures": (
                            weave_agents_adoption_validation_failures
                        )
                    }
                    if isinstance(weave_agents_adoption_validation_failures, dict)
                    else {}
                ),
            },
            "gates": [
                {
                    "name": "wandb_completion",
                    "ok": False,
                    "blocking": True,
                    "status": "missing_required_benchmarks",
                    "evidence_paths": [],
                }
            ],
        },
    )
    output_dir = tmp_path / "bundle_with_adoption"
    result = subprocess.run(
        [
            "python3",
            str(BUILD_SCRIPT),
            "--readiness-report",
            str(report),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return output_dir, source_audit, completion


def build_bundle_with_paid_review_scope_attestation(tmp_path):
    taxonomy = tmp_path / "taxonomy.yaml"
    taxonomy.write_text("version: test\n", encoding="utf-8")
    completion_payload = add_wandb_completion_run_metadata(
        {**agentic_math_wandb_completion_payload(), "taxonomy_path": str(taxonomy)}
    )
    completion = write_json(
        tmp_path / "wandb_completion.json",
        completion_payload,
    )
    completion_sha = sha256(completion)
    pre_run_budget = write_json(
        tmp_path / "openai_canary_budget_estimate.json",
        {
            "schema_version": 1,
            "target_model": "openai-direct/gpt-4.1-mini-2025-04-14",
            "estimated_total_usd": {"low": 10.0, "mid": 12.0, "high": 20.0},
            "pricing_source_url": "https://openai.com/index/gpt-4-1/",
        },
    )
    pre_run_budget_sha = sha256(pre_run_budget)
    run_eval_preflight = write_json(
        tmp_path / "run_eval_preflight.json",
        {
            "schema_version": 1,
            "generated_at": 1,
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
    review_path = tmp_path / "canary_agentic_paid_run_review.json"
    attestation_path = tmp_path / "agentic_math-run-1.confirmed_scope_attestation.json"
    review = write_json(
        review_path,
        {
            "status": "completed",
            "phase": "agentic",
            "canary": True,
            "model_count": 1,
            "run_purpose": "one-model agentic canary",
            "expected_cost_band": "$10-$20",
            "actual_cost_estimate": "$12.34",
            "provider_bill_reference": "openai-dashboard-2026-06-28",
            "pre_run_budget_estimate": {
                "path": str(pre_run_budget),
                "present": True,
                "valid": True,
                "sha256": pre_run_budget_sha,
                "target_model": "openai-direct/gpt-4.1-mini-2025-04-14",
                "estimated_total_usd": {"low": 10.0, "mid": 12.0, "high": 20.0},
            },
            "verify_wandb_completion": True,
            "verify_weave_agents": False,
            "run_eval_preflights": [
                {
                    "config": "config.yaml",
                    "output_json": str(run_eval_preflight),
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
                        str(run_eval_preflight),
                    ],
                }
            ],
            "runs": [
                {
                    "config": "config.yaml",
                    "preflight_json": str(run_eval_preflight),
                    "preflight_returncode": 0,
                    "preflight_ok": True,
                    "preflight_status": "passed",
                    "log_path": "run.log",
                    "wandb_entity": "test-entity",
                    "wandb_project": "test-project",
                    "wandb_run_id": "run-1",
                    "returncode": 0,
                    "started_at": 1,
                    "ended_at": 2,
                    "wandb_completion": [
                        {
                            "benchmark": "agentic_math",
                            "entity": "test-entity",
                            "project": "test-project",
                            "run_id": "run-1",
                            "ok": True,
                            "path": str(completion),
                            "sha256": completion_sha,
                            "verified": True,
                            "adopted_existing_result": True,
                            "scope_attestation": {
                                "schema_version": 1,
                                "confirmed": True,
                                "confirmed_by": "yuya",
                                "confirmed_at": "2026-06-28T01:30:00+09:00",
                                "confirmation": "This W&B run is the reviewed Agentic Math canary scope.",
                                "review_path": str(review_path),
                                "completion_path": str(completion),
                                "completion_sha256": completion_sha,
                                "benchmark": "agentic_math",
                                "entity": "test-entity",
                                "project": "test-project",
                                "run_id": "run-1",
                                "actual_cost_estimate": "$12.34",
                                "provider_bill_reference": "openai-dashboard-2026-06-28",
                                "source_attestation_json": str(attestation_path),
                            },
                        }
                    ],
                }
            ],
        },
    )
    attestation = write_json(
        attestation_path,
        {
            "schema_version": 1,
            "confirmed": True,
            "confirmed_by": "yuya",
            "confirmed_at": "2026-06-28T01:30:00+09:00",
            "confirmation": "This W&B run is the reviewed Agentic Math canary scope.",
            "review_path": str(review),
            "completion_path": str(completion),
            "completion_sha256": completion_sha,
            "benchmark": "agentic_math",
            "entity": "test-entity",
            "project": "test-project",
            "run_id": "run-1",
            "actual_cost_estimate": "$12.34",
            "provider_bill_reference": "openai-dashboard-2026-06-28",
        },
    )
    paid_review_check = write_json(
        tmp_path / "taiwan_paid_run_review_check.json",
        {
            "ok": True,
            "status": "passed",
            "summary": {"blockers": []},
            "requirements": {"required_wandb_benchmarks": ["agentic_math"]},
            "review_completion_requirements": {},
            "gates": [
                {
                    "name": "paid_run_review_package",
                    "ok": True,
                    "blocking": True,
                    "status": "passed",
                    "records": [
                        {
                            "path": str(review),
                            "ok": True,
                            "status": "completed",
                            "phase": "agentic",
                            "canary": True,
                            "model_count": 1,
                            "run_purpose_present": True,
                            "expected_cost_band_present": True,
                            "actual_cost_estimate_present": True,
                            "provider_bill_reference_present": True,
                            "actual_cost_estimate_placeholder": False,
                            "provider_bill_reference_placeholder": False,
                            "pre_run_budget_estimate": {
                                "path": str(pre_run_budget),
                                "present": True,
                                "valid": True,
                                "sha256": pre_run_budget_sha,
                                "target_model": "openai-direct/gpt-4.1-mini-2025-04-14",
                                "estimated_total_usd": {"low": 10.0, "mid": 12.0, "high": 20.0},
                            },
                            "run_count": 1,
                            "verify_wandb_completion": True,
                            "wandb_completion_entries": [
                                {
                                    "benchmark": "agentic_math",
                                    "entity": "test-entity",
                                    "project": "test-project",
                                    "run_id": "run-1",
                                    "parent_run_id": "run-1",
                                    "parent_entity": "test-entity",
                                    "parent_project": "test-project",
                                    "parent_run_id_matches": True,
                                    "parent_entity_matches": True,
                                    "parent_project_matches": True,
                                    "ok": True,
                                    "path": str(completion),
                                    "sha256": completion_sha,
                                    "verified": True,
                                    "adopted_existing_result": True,
                                    "scope_attestation_valid": True,
                                    "scope_attestation": {
                                        "schema_version": 1,
                                        "confirmed": True,
                                        "confirmed_by": "yuya",
                                        "confirmed_at": "2026-06-28T01:30:00+09:00",
                                        "confirmation": "This W&B run is the reviewed Agentic Math canary scope.",
                                        "review_path": str(review),
                                        "completion_path": str(completion),
                                        "completion_sha256": completion_sha,
                                        "benchmark": "agentic_math",
                                        "entity": "test-entity",
                                        "project": "test-project",
                                        "run_id": "run-1",
                                        "actual_cost_estimate": "$12.34",
                                        "provider_bill_reference": "openai-dashboard-2026-06-28",
                                        "source_attestation_json": str(attestation),
                                    },
                                }
                            ],
                            "verify_weave_agents": False,
                            "weave_agents_completion_entries": [],
                            "errors": [],
                        }
                    ],
                    "blocking_records": [],
                }
            ],
        },
    )
    report = write_json(
        tmp_path / "readiness.json",
        {
            "schema_version": 1,
            "ok": False,
            "status": "not_ready",
            "summary": {
                "gate_count": 1,
                "blocker_count": 1,
                "blockers": ["one_model_full_canary"],
            },
            "runner": {
                "report_json": str(tmp_path / "readiness.json"),
                "paid_run_review_check": {
                    "path": str(paid_review_check),
                    "markdown_path": "",
                    "ok": True,
                    "status": "passed",
                },
            },
            "gates": [
                {
                    "name": "one_model_full_canary",
                    "ok": False,
                    "blocking": True,
                    "status": "incomplete",
                    "evidence_paths": [],
                    "required_wandb_benchmarks": ["agentic_math"],
                    "weave_agents_completion_required": True,
                    "completed_weave_agents_completion_phases": {},
                    "missing_required_weave_agents_completion_phases": [],
                }
            ],
        },
    )
    output_dir = tmp_path / "bundle_with_paid_review_scope"
    result = subprocess.run(
        [
            "python3",
            str(BUILD_SCRIPT),
            "--readiness-report",
            str(report),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return output_dir, attestation


def build_bundle_with_prepared_one_model_review(tmp_path):
    review = write_json(
        tmp_path / "canary_agentic_paid_run_review.json",
        {
            "status": "prepared",
            "phase": "agentic",
            "canary": True,
            "model_count": 1,
            "run_purpose": "one-model agentic canary",
            "expected_cost_band": "$10-$20",
            "verify_wandb_completion": True,
            "verify_weave_agents": True,
            "runs": [],
        },
    )
    paid_review_check = write_json(
        tmp_path / "taiwan_paid_run_review_check.json",
        {
            "ok": False,
            "status": "not_ready",
            "summary": {"blockers": ["one_model_full_canary"]},
            "requirements": {"required_wandb_benchmarks": ["agentic_math"]},
            "review_completion_requirements": {},
            "gates": [
                {
                    "name": "one_model_full_canary",
                    "ok": False,
                    "blocking": True,
                    "status": "incomplete",
                    "records": [
                        {
                            "path": str(review),
                            "ok": False,
                            "status": "prepared",
                            "phase": "agentic",
                            "canary": True,
                            "model_count": 1,
                            "run_purpose_present": True,
                            "expected_cost_band_present": True,
                            "actual_cost_estimate_present": False,
                            "provider_bill_reference_present": False,
                            "actual_cost_estimate_placeholder": False,
                            "provider_bill_reference_placeholder": False,
                            "run_count": 0,
                            "verify_wandb_completion": True,
                            "wandb_completion_entries": [],
                            "verify_weave_agents": True,
                            "weave_agents_completion_entries": [],
                            "errors": [],
                        }
                    ],
                    "blocking_records": [],
                }
            ],
        },
    )
    report = write_json(
        tmp_path / "readiness.json",
        {
            "schema_version": 1,
            "ok": False,
            "status": "not_ready",
            "summary": {
                "gate_count": 1,
                "blocker_count": 1,
                "blockers": ["one_model_full_canary"],
            },
            "runner": {
                "report_json": str(tmp_path / "readiness.json"),
                "paid_run_review_check": {
                    "path": str(paid_review_check),
                    "markdown_path": "",
                    "ok": False,
                    "status": "not_ready",
                },
            },
            "gates": [
                {
                    "name": "one_model_full_canary",
                    "ok": False,
                    "blocking": True,
                    "status": "incomplete",
                    "evidence_paths": [],
                    "required_wandb_benchmarks": ["agentic_math"],
                    "weave_agents_completion_required": True,
                    "completed_weave_agents_completion_phases": {},
                    "missing_required_weave_agents_completion_phases": [],
                }
            ],
        },
    )
    output_dir = tmp_path / "bundle_with_prepared_one_model_review"
    result = subprocess.run(
        [
            "python3",
            str(BUILD_SCRIPT),
            "--readiness-report",
            str(report),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return output_dir


def nemoclaw_setup_payload():
    return {
        "schema_version": 1,
        "ok": False,
        "host_prerequisites_ok": True,
        "runtime_installed": False,
        "missing_required_commands": ["nemoclaw", "openshell"],
        "commands": {
            "docker": {"available": True, "path": "/usr/bin/docker", "required": True, "info_ok": True},
            "nemoclaw": {"available": False, "path": "", "required": True},
            "openshell": {"available": False, "path": "", "required": True},
        },
        "policy_tier": "restricted",
        "policy_tier_allowed_values": ["restricted", "balanced", "open"],
        "policy_tier_valid": True,
        "third_party_software": {
            "name": "NVIDIA NemoClaw",
            "vendor": "NVIDIA",
            "repository_url": "https://github.com/NVIDIA/NemoClaw",
            "documentation_url": "https://docs.nvidia.com/nemoclaw/user-guide/openclaw/get-started/quickstart",
            "installer_url": "https://www.nvidia.com/nemoclaw.sh",
            "install_ref": "lkg",
                "installer_sha256": "",
                "installer_signature": "",
                "installer_lock_json": INSTALLER_LOCK_JSON,
                "installer_review_json": "",
                "installer_review_verified": False,
                "installer_integrity_verified": False,
            "installer_provenance_locked": False,
            "installer_provenance_note": "Installer integrity is not verified by this script; operator review is required before install/onboard.",
            "acceptance_required": True,
            "acceptance_flag": "--yes-i-accept-third-party-software",
            "accepted": False,
            "install_or_onboard_requested": False,
            "operator_review_required_before_install": True,
        },
        "setup_plan": {
            "third_party_software_name": "NVIDIA NemoClaw",
            "installer_url": "https://www.nvidia.com/nemoclaw.sh",
            "install_ref": "lkg",
                "installer_sha256": "",
                "installer_signature": "",
                "installer_lock_json": INSTALLER_LOCK_JSON,
                "installer_review_json": "",
                "installer_review_verified": False,
                "installer_integrity_verified": False,
            "installer_provenance_locked": False,
            "installer_provenance_note": "Installer integrity is not verified by this script; operator review is required before install/onboard.",
            "acceptance_ledger_fields": [
                "accepted_third_party_software",
                "third_party_software.name",
                "third_party_software.vendor",
                "third_party_software.installer_url",
                "third_party_software.install_ref",
                    "third_party_software.installer_sha256",
                    "third_party_software.installer_signature",
                    "third_party_software.installer_lock_json",
                    "third_party_software.installer_review_json",
                    "third_party_software.installer_review_verified",
                    "third_party_software.installer_integrity_verified",
                "third_party_software.installer_provenance_locked",
                "third_party_software.installer_provenance_note",
                "third_party_software.acceptance_required",
                "third_party_software.acceptance_flag",
                "third_party_software.accepted",
                "policy_tier",
                "policy_tier_allowed_values",
                "policy_tier_valid",
                "setup_plan.policy_tier",
                "setup_plan.policy_tier_allowed_values",
                "setup_plan.policy_tier_valid",
                    "setup_plan.installer_sha256",
                    "setup_plan.installer_signature",
                    "setup_plan.installer_lock_json",
                    "setup_plan.installer_review_json",
                    "setup_plan.installer_review_verified",
                    "setup_plan.installer_integrity_verified",
                "setup_plan.installer_provenance_locked",
                "setup_plan.installer_provenance_note",
                "operation_results.install.log_path",
                "operation_results.onboard.log_path",
            ],
            "policy_tier": "restricted",
            "policy_tier_allowed_values": ["restricted", "balanced", "open"],
            "policy_tier_valid": True,
            "post_install_check_command": (
                "scripts/setup/install_nemoclaw.sh --check-only "
                "--sandbox nejumi-taiwan "
                "--json temp/nemoclaw_setup_check_YYYYMMDDTHHMM.json"
            ),
            "installer_review_command": (
                "uv run python scripts/setup/review_nemoclaw_installer.py "
                "--url https://www.nvidia.com/nemoclaw.sh "
                "--install-ref lkg "
                f"--expected-sha256 {INSTALLER_SHA256} "
                f"--lock-json {INSTALLER_LOCK_JSON} "
                "--json temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json "
                "--markdown temp/nemoclaw_installer_review_YYYYMMDDTHHMM.md"
            ),
            "install_and_onboard_command": (
                "scripts/setup/install_nemoclaw.sh --install --onboard "
                "--sandbox nejumi-taiwan "
                "--provider openai "
                "--policy-tier restricted "
                "--install-ref lkg "
                f"--installer-lock-json {INSTALLER_LOCK_JSON} "
                f"--installer-sha256 {INSTALLER_SHA256} "
                "--installer-review-json temp/nemoclaw_installer_review.json "
                "--yes-i-accept-third-party-software "
                "--json temp/nemoclaw_install_onboard.json"
            ),
            "post_install_verification_command": (
                "uv run python scripts/setup/verify_nemoclaw_post_install.py "
                "--sandbox nejumi-taiwan "
                "--nemoclaw-openclaw-config-path /sandbox/.openclaw/openclaw.json "
                "--canary-manifest configs/taiwan_openai_canary_models.yaml "
                "--generated-full-dir configs/taiwan_full/generated_openai_canary "
                "--generated-nonagentic-dir configs/taiwan_full/generated_openai_canary_nonagentic "
                "--generated-agentic-dir configs/taiwan_full/generated_openai_canary_agentic_nemoclaw "
                "--generated-agentic-aggregate-dir configs/taiwan_full/generated_openai_canary_agentic_aggregate "
                "--json temp/nemoclaw_post_install_verification_YYYYMMDDTHHMM.json "
                "--markdown temp/nemoclaw_post_install_verification_YYYYMMDDTHHMM.md "
                "--fail-on-failed"
            ),
            "canary_readiness_command": (
                "uv run python scripts/tools/check_taiwan_canary_readiness.py "
                "--manifest configs/taiwan_openai_canary_models.yaml "
                "--generated-full-dir configs/taiwan_full/generated_openai_canary "
                "--generated-nonagentic-dir configs/taiwan_full/generated_openai_canary_nonagentic "
                "--generated-agentic-dir configs/taiwan_full/generated_openai_canary_agentic_nemoclaw "
                "--generated-agentic-aggregate-dir configs/taiwan_full/generated_openai_canary_agentic_aggregate "
                "--require-nemoclaw "
                "--nemoclaw-sandbox nejumi-taiwan "
                "--json outputs/taiwan_full_eval/openai_canary_readiness_nemoclaw.json"
            ),
            "adoption_check_command": (
                "uv run python scripts/tools/check_taiwan_nemoclaw_adoption.py "
                "--setup-json temp/nemoclaw_setup_check_YYYYMMDDTHHMM.json "
                "--readiness-json outputs/taiwan_full_eval/openai_canary_readiness_nemoclaw.json "
                "--sandbox nejumi-taiwan "
                "--agentic-config-glob configs/taiwan_full/generated_openai_canary_agentic_nemoclaw/*.yaml "
                "--json temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.json "
                "--markdown temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.md "
                "--fail-on-not-adoptable"
            ),
            "production_readiness_command": (
                "uv run python scripts/tools/run_taiwan_production_readiness_gate.py "
                "--report-json outputs/taiwan_full_eval/taiwan_production_readiness_report.json "
                "--fail-on-not-ready"
            ),
            "production_install_and_onboard_command": (
                "scripts/setup/install_nemoclaw.sh --install --onboard "
                "--sandbox nejumi-taiwan "
                "--provider openai "
                "--policy-tier restricted "
                "--install-ref lkg "
                f"--installer-lock-json {INSTALLER_LOCK_JSON} "
                f"--installer-sha256 {INSTALLER_SHA256} "
                "--installer-review-json temp/nemoclaw_installer_review.json "
                "--yes-i-accept-third-party-software "
                "--json temp/nemoclaw_install_onboard.json"
            ),
            "operator_sequence": [
                {
                    "step": "setup_check",
                    "command": (
                        "scripts/setup/install_nemoclaw.sh --check-only "
                        "--sandbox nejumi-taiwan "
                        "--json temp/nemoclaw_setup_check_YYYYMMDDTHHMM.json"
                    ),
                    "expected_evidence_path": "temp/nemoclaw_setup_check_YYYYMMDDTHHMM.json",
                    "requires_external_action": False,
                    "required": True,
                },
                {
                    "step": "installer_review",
                    "command": (
                        "uv run python scripts/setup/review_nemoclaw_installer.py "
                        "--url https://www.nvidia.com/nemoclaw.sh "
                        "--install-ref lkg "
                        f"--expected-sha256 {INSTALLER_SHA256} "
                        f"--lock-json {INSTALLER_LOCK_JSON} "
                        "--json temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json "
                        "--markdown temp/nemoclaw_installer_review_YYYYMMDDTHHMM.md"
                    ),
                    "expected_evidence_path": "temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json",
                    "requires_external_action": False,
                    "required": True,
                },
                {
                    "step": "install_and_onboard",
                    "command": (
                        "scripts/setup/install_nemoclaw.sh --install --onboard "
                        "--sandbox nejumi-taiwan "
                        "--provider openai "
                        "--policy-tier restricted "
                        "--install-ref lkg "
                        f"--installer-lock-json {INSTALLER_LOCK_JSON} "
                        f"--installer-sha256 {INSTALLER_SHA256} "
                        "--installer-review-json temp/nemoclaw_installer_review.json "
                        "--yes-i-accept-third-party-software "
                        "--json temp/nemoclaw_install_onboard.json"
                    ),
                    "expected_evidence_path": "temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.json",
                    "requires_external_action": True,
                    "required": True,
                },
                {
                    "step": "post_install_verification",
                    "command": (
                        "uv run python scripts/setup/verify_nemoclaw_post_install.py "
                        "--sandbox nejumi-taiwan "
                        "--nemoclaw-openclaw-config-path /sandbox/.openclaw/openclaw.json "
                        "--canary-manifest configs/taiwan_openai_canary_models.yaml "
                        "--generated-full-dir configs/taiwan_full/generated_openai_canary "
                        "--generated-nonagentic-dir configs/taiwan_full/generated_openai_canary_nonagentic "
                        "--generated-agentic-dir configs/taiwan_full/generated_openai_canary_agentic_nemoclaw "
                        "--generated-agentic-aggregate-dir configs/taiwan_full/generated_openai_canary_agentic_aggregate "
                        "--json temp/nemoclaw_post_install_verification_YYYYMMDDTHHMM.json "
                        "--markdown temp/nemoclaw_post_install_verification_YYYYMMDDTHHMM.md "
                        "--fail-on-failed"
                    ),
                    "expected_evidence_path": "temp/nemoclaw_post_install_verification_YYYYMMDDTHHMM.json",
                    "requires_external_action": False,
                    "required": True,
                },
                {
                    "step": "canary_readiness",
                    "command": (
                        "uv run python scripts/tools/check_taiwan_canary_readiness.py "
                        "--manifest configs/taiwan_openai_canary_models.yaml "
                        "--generated-full-dir configs/taiwan_full/generated_openai_canary "
                        "--generated-nonagentic-dir configs/taiwan_full/generated_openai_canary_nonagentic "
                        "--generated-agentic-dir configs/taiwan_full/generated_openai_canary_agentic_nemoclaw "
                        "--generated-agentic-aggregate-dir configs/taiwan_full/generated_openai_canary_agentic_aggregate "
                        "--require-nemoclaw "
                        "--nemoclaw-sandbox nejumi-taiwan "
                        "--json outputs/taiwan_full_eval/openai_canary_readiness_nemoclaw.json"
                    ),
                    "expected_evidence_path": "outputs/taiwan_full_eval/openai_canary_readiness_nemoclaw.json",
                    "requires_external_action": False,
                    "required": True,
                },
                {
                    "step": "adoption_check",
                    "command": (
                        "uv run python scripts/tools/check_taiwan_nemoclaw_adoption.py "
                        "--setup-json temp/nemoclaw_setup_check_YYYYMMDDTHHMM.json "
                        "--readiness-json outputs/taiwan_full_eval/openai_canary_readiness_nemoclaw.json "
                        "--sandbox nejumi-taiwan "
                        "--agentic-config-glob configs/taiwan_full/generated_openai_canary_agentic_nemoclaw/*.yaml "
                        "--json temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.json "
                        "--markdown temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.md "
                        "--fail-on-not-adoptable"
                    ),
                    "expected_evidence_path": "temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.json",
                    "requires_external_action": False,
                    "required": True,
                },
                {
                    "step": "production_readiness",
                    "command": (
                        "uv run python scripts/tools/run_taiwan_production_readiness_gate.py "
                        "--report-json outputs/taiwan_full_eval/taiwan_production_readiness_report.json "
                        "--fail-on-not-ready"
                    ),
                    "expected_evidence_path": "outputs/taiwan_full_eval/taiwan_production_readiness_report.json",
                    "requires_external_action": False,
                    "required": True,
                },
            ],
            "expected_evidence_paths": [
                "temp/nemoclaw_setup_check_YYYYMMDDTHHMM.json",
                "temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json",
                "temp/nemoclaw_installer_review_YYYYMMDDTHHMM.md",
                "temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.json",
                "temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.install.log",
                "temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.onboard.log",
                "temp/nemoclaw_post_install_verification_YYYYMMDDTHHMM.json",
                "temp/nemoclaw_post_install_verification_YYYYMMDDTHHMM.md",
                "outputs/taiwan_full_eval/openai_canary_readiness_nemoclaw.json",
                "temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.json",
                "temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.md",
                "outputs/taiwan_full_eval/taiwan_production_readiness_report.json",
            ],
        },
    }


def test_validate_wandb_completion_payload_rejects_bad_schema_version():
    module = load_verify_module()
    payload = agentic_math_wandb_completion_payload()
    payload["schema_version"] = 2

    errors = module.validate_wandb_completion_payload(
        payload,
        label="proof",
        expected_benchmark="agentic_math",
    )

    assert "proof schema_version is not 1" in errors


def test_validate_wandb_completion_payload_rejects_agentic_without_nemoclaw_audit():
    module = load_verify_module()
    payload = agentic_math_wandb_completion_payload()
    payload["required_evidence"].pop("nemoclaw_session_audit")
    payload["observed_evidence"].pop("nemoclaw_session_audit")

    errors = module.validate_wandb_completion_payload(
        payload,
        label="proof",
        expected_benchmark="agentic_math",
    )

    assert "proof required_evidence.nemoclaw_session_audit is not an object" in errors
    assert "proof observed_evidence.nemoclaw_session_audit is not an object" in errors


def test_validate_wandb_completion_payload_rejects_agentic_missing_required_output_columns():
    module = load_verify_module()
    payload = agentic_math_wandb_completion_payload()
    output_table = payload["required_evidence"]["tables"][1]
    output_table["required_columns"] = [
        column
        for column in AGENTIC_MATH_OUTPUT_COLUMNS
        if column != "weave_sidecar_ok"
    ]

    errors = module.validate_wandb_completion_payload(
        payload,
        label="proof",
        expected_benchmark="agentic_math",
    )

    assert (
        "proof required_evidence.tables agentic_math_output_table "
        "required_columns mismatch"
    ) in errors
    assert "proof checks output_table_columns required_columns mismatch" in errors


def test_validate_wandb_completion_payload_rejects_agentic_missing_observed_output_column():
    module = load_verify_module()
    payload = agentic_math_wandb_completion_payload()
    output_table = payload["observed_evidence"]["tables"][1]
    output_table["columns"] = [
        column
        for column in AGENTIC_MATH_OUTPUT_COLUMNS
        if column != "tool_policy_violations"
    ]
    output_table["missing_columns"] = ["tool_policy_violations"]
    output_table["columns_ok"] = False
    output_check = next(
        check for check in payload["checks"] if check["name"] == "output_table_columns"
    )
    output_check["columns"] = output_table["columns"]
    output_check["missing_columns"] = ["tool_policy_violations"]
    output_check["ok"] = False

    errors = module.validate_wandb_completion_payload(
        payload,
        label="proof",
        expected_benchmark="agentic_math",
    )

    assert "proof observed_evidence.tables agentic_math_output_table missing tool_policy_violations" in errors
    assert "proof observed_evidence.tables agentic_math_output_table columns_ok is not true" in errors
    assert "proof checks contains failing check: output_table_columns" in errors


def nemoclaw_operator_handoff_payload(setup_path, setup_payload):
    setup_plan = setup_payload["setup_plan"]
    steps = [
        {
            "step": row.get("step"),
            "command": row.get("command"),
            "expected_evidence_path": row.get("expected_evidence_path"),
            "requires_external_action": row.get("requires_external_action") is True,
            "required": row.get("required") is True,
        }
        for row in setup_plan["operator_sequence"]
    ]
    expected_evidence_paths = [
        path
        for path in setup_plan["expected_evidence_paths"]
        if isinstance(path, str) and path.strip()
    ]
    handoff = {
        "available": bool(steps),
        "source_setup_json": str(setup_path),
        "step_count": len(steps),
        "required_step_count": sum(1 for row in steps if row["required"]),
        "external_action_step_count": sum(
            1 for row in steps if row["requires_external_action"]
        ),
        "evidence_path_count": len(expected_evidence_paths),
        "steps": steps,
        "expected_evidence_paths": expected_evidence_paths,
    }
    for field in (
        "post_install_check_command",
        "installer_review_command",
        "install_and_onboard_command",
        "production_install_and_onboard_command",
        "post_install_verification_command",
        "canary_readiness_command",
        "adoption_check_command",
        "production_readiness_command",
    ):
        handoff[field] = setup_plan[field]
    return handoff


def build_bundle_with_nemoclaw_adoption(tmp_path):
    setup_payload = nemoclaw_setup_payload()
    setup = write_json(tmp_path / "nemoclaw_setup.json", setup_payload)
    operator_handoff = nemoclaw_operator_handoff_payload(setup, setup_payload)
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "run:",
                "  agentic_math: true",
                "  swebench_pro: true",
                "agentic_math:",
                "  nemoclaw_sandbox: nejumi-taiwan",
                "  use_task_agent: true",
                "  deny_tool:",
                "    - code_execution",
                "    - web_search",
                "    - web_fetch",
                "    - browser",
                "    - browser_*",
                "    - '*search*'",
                "  deny_argument_pattern:",
                "    - https?://",
                r"    - \b(curl|wget)\b",
                r"    - \b(requests|urllib|httpx)\.",
                "swebench_pro: {}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    installer_review_path = tmp_path / "nemoclaw_installer_review.json"
    install_onboard_json = tmp_path / "nemoclaw_install_onboard.json"
    installer_review = write_json(
        installer_review_path,
        {
            "schema_version": 1,
            "ok": True,
            "status": "reviewed",
            "generated_at": "2026-06-28T01:32:47Z",
            "installer_url": "https://www.nvidia.com/nemoclaw.sh",
            "install_ref": "lkg",
            "lock_json": INSTALLER_LOCK_JSON,
            "lock_verified": True,
            "expected_sha256": INSTALLER_SHA256,
            "sha256": INSTALLER_SHA256,
            "size_bytes": 6356,
            "will_execute_installer": False,
            "will_install_or_onboard": False,
            "will_launch_model_inference": False,
            "will_query_wandb": False,
            "recommended_install_command": (
                "scripts/setup/install_nemoclaw.sh --install --onboard "
                "--sandbox nejumi-taiwan "
                "--provider openai "
                "--policy-tier restricted "
                "--install-ref lkg "
                f"--installer-lock-json {INSTALLER_LOCK_JSON} "
                f"--installer-sha256 {INSTALLER_SHA256} "
                f"--installer-review-json {installer_review_path} "
                "--yes-i-accept-third-party-software "
                f"--json {install_onboard_json}"
            ),
        },
    )
    installer_review_md = tmp_path / "nemoclaw_installer_review.md"
    installer_review_md.write_text("# installer review\n", encoding="utf-8")
    adoption_path = tmp_path / "nemoclaw_adoption.json"
    adoption_md = tmp_path / "nemoclaw_adoption.md"
    adoption = write_json(
        adoption_path,
        {
            "schema_version": 1,
            "path": str(adoption_path),
            "markdown_path": str(adoption_md),
            "ok": False,
            "status": "not_installed",
            "ready_for_use": False,
            "adoption_recommendation": "conditional_adopt_for_agentic_math",
            "adoption_scope": "agentic_math_only",
            "design_ready": True,
            "blockers": ["setup_installed"],
            "runtime_blockers": ["setup_installed"],
            "design_blockers": [],
            "other_blockers": [],
            "setup_runtime": {
                "host_prerequisites_ok": True,
                "runtime_installed": False,
                "missing_required_commands": ["nemoclaw", "openshell"],
                "missing_components": ["nemoclaw", "openshell"],
            },
            "missing_required_commands": ["nemoclaw", "openshell"],
            "missing_components": ["nemoclaw", "openshell"],
            "operator_handoff": operator_handoff,
            "operator_handoff_step_count": operator_handoff["step_count"],
            "operator_handoff_external_action_step_count": operator_handoff[
                "external_action_step_count"
            ],
            "operator_handoff_evidence_path_count": operator_handoff[
                "evidence_path_count"
            ],
            "summary": {
                "criterion_count": 1,
                "blocker_count": 1,
                "blockers": ["setup_installed"],
                "adoption_recommendation": "conditional_adopt_for_agentic_math",
                "ready_for_use": False,
                "adoption_scope": "agentic_math_only",
                "setup_runtime": {
                    "host_prerequisites_ok": True,
                    "runtime_installed": False,
                    "missing_required_commands": ["nemoclaw", "openshell"],
                    "missing_components": ["nemoclaw", "openshell"],
                },
                "operator_handoff": {
                    "available": True,
                    "source_setup_json": str(setup),
                    "step_count": operator_handoff["step_count"],
                    "required_step_count": operator_handoff["required_step_count"],
                    "external_action_step_count": operator_handoff[
                        "external_action_step_count"
                    ],
                    "evidence_path_count": operator_handoff["evidence_path_count"],
                },
            },
            "adoption_decision": {
                "recommendation": "conditional_adopt_for_agentic_math",
                "ready_for_use": False,
                "scope": "agentic_math_only",
                "sandbox": "nejumi-taiwan",
                "design_ready": True,
                "runtime_blockers": ["setup_installed"],
                "design_blockers": [],
                "other_blockers": [],
                "blockers": ["setup_installed"],
            },
            "setup_paths": [str(setup)],
            "criteria": [
                {
                    "name": "setup_plan_safety",
                    "ok": True,
                    "status": "passed",
                    "detail": "safe",
                    "next_action": "review",
                    "evidence_paths": [str(setup)],
                },
                {
                    "name": "setup_installed",
                    "ok": False,
                    "status": "missing_or_not_ready",
                    "detail": "Missing or not-ready setup components: nemoclaw, openshell",
                    "next_action": "Install/onboard NeMoClaw and OpenShell, then rerun --check-only.",
                    "evidence_paths": [str(setup)],
                    "host_prerequisites_ok": True,
                    "runtime_installed": False,
                    "missing_required_commands": ["nemoclaw", "openshell"],
                    "missing_components": ["nemoclaw", "openshell"],
                },
                {
                    "name": "agentic_math_config",
                    "ok": True,
                    "status": "passed",
                    "detail": "Agentic Math has a NeMoClaw-ready generated config.",
                    "next_action": "Use the passing config for the Agentic Math canary.",
                    "evidence_paths": [str(config)],
                },
                {
                    "name": "swebench_pro_non_adoption_guard",
                    "ok": True,
                    "status": "passed",
                    "detail": "No checked SWE-Bench Pro config contains NeMoClaw fields; SWE remains on host OpenClaw.",
                    "next_action": "Optionally generate SWE configs with --swebench-pro-nemoclaw-sandbox nejumi-taiwan.",
                    "evidence_paths": [str(config)],
                    "records": [
                        {
                            "path": str(config),
                            "valid": True,
                            "swebench_pro_enabled": True,
                            "swebench_pro_nemoclaw_keys": [],
                        }
                    ],
                    "offending_records": [],
                }
            ],
        },
    )
    adoption_md.write_text("# adoption\n", encoding="utf-8")
    report = write_json(
        tmp_path / "readiness.json",
        {
            "schema_version": 1,
            "ok": False,
            "status": "not_ready",
            "summary": {
                "gate_count": 1,
                "blocker_count": 1,
                "blockers": ["nemoclaw_readiness"],
            },
            "runner": {
                "report_json": str(tmp_path / "readiness.json"),
                "nemoclaw_adoption_check": {
                    "path": str(adoption),
                    "markdown_path": str(adoption_md),
                    "ok": False,
                    "status": "not_installed",
                    "summary": {
                        "criterion_count": 1,
                        "blocker_count": 1,
                        "blockers": ["setup_installed"],
                    },
                },
            },
            "gates": [
                {
                    "name": "nemoclaw_readiness",
                    "ok": False,
                    "blocking": True,
                    "status": "failed",
                    "evidence_paths": [str(setup)],
                    "latest_installer_review": {
                        "path": str(installer_review),
                        "generated_at": "2026-06-28T01:32:47Z",
                        "installer_url": "https://www.nvidia.com/nemoclaw.sh",
                        "install_ref": "lkg",
                        "lock_json": INSTALLER_LOCK_JSON,
                        "lock_verified": True,
                        "expected_sha256": INSTALLER_SHA256,
                        "sha256": INSTALLER_SHA256,
                        "size_bytes": 6356,
                        "status": "reviewed",
                        "ok": True,
                    },
                }
            ],
        },
    )
    output_dir = tmp_path / "bundle_with_nemoclaw"
    result = subprocess.run(
        [
            "python3",
            str(BUILD_SCRIPT),
            "--readiness-report",
            str(report),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return output_dir, setup


def build_bundle_with_nemoclaw_post_install(tmp_path):
    setup = write_json(tmp_path / "nemoclaw_setup.json", nemoclaw_setup_payload())
    preflight = write_json(
        tmp_path / "nemoclaw_protocol_preflight.json",
        {"ok": False, "status": "missing"},
    )
    readiness = write_json(
        tmp_path / "nemoclaw_canary_readiness.json",
        {
            "ok": False,
            "status": "failed",
            "checks": [
                {
                    "name": "NeMoClaw command is available",
                    "ok": False,
                    "detail": "missing",
                },
                {
                    "name": "agentic Math denies remote lookup via deny_tool",
                    "ok": True,
                    "detail": json.dumps(
                        {
                            "section": "agentic_math",
                            "missing": [],
                            "observed": [
                                "*search*",
                                "browser",
                                "browser_*",
                                "code_execution",
                                "web_fetch",
                                "web_search",
                            ],
                        }
                    ),
                },
                {
                    "name": "agentic SWE denies remote lookup via deny_tool",
                    "ok": True,
                    "detail": json.dumps(
                        {
                            "section": "swebench_pro",
                            "missing": [],
                            "observed": [
                                "*search*",
                                "browser",
                                "browser_*",
                                "code_execution",
                                "web_fetch",
                                "web_search",
                            ],
                        }
                    ),
                },
                {
                    "name": "agentic Math denies remote lookup via deny_argument_pattern",
                    "ok": True,
                    "detail": json.dumps(
                        {
                            "section": "agentic_math",
                            "missing": [],
                            "observed": [
                                r"\b(curl|wget)\b",
                                r"\b(requests|urllib|httpx)\.",
                                "https?://",
                            ],
                        }
                    ),
                },
                {
                    "name": "agentic SWE denies remote lookup via deny_argument_pattern",
                    "ok": True,
                    "detail": json.dumps(
                        {
                            "section": "swebench_pro",
                            "missing": [],
                            "observed": [
                                r"\b(curl|wget)\b",
                                r"\b(requests|urllib|httpx)\.",
                                "https?://",
                            ],
                        }
                    ),
                },
            ],
        },
    )
    adoption = write_json(
        tmp_path / "taiwan_nemoclaw_adoption_check.json",
        {"schema_version": 1, "ok": False, "status": "not_installed"},
    )
    adoption_md = tmp_path / "taiwan_nemoclaw_adoption_check.md"
    adoption_md.write_text("# adoption\n", encoding="utf-8")
    output_sha256 = {
        "setup_json": sha256(setup),
        "preflight_json": sha256(preflight),
        "readiness_json": sha256(readiness),
        "adoption_json": sha256(adoption),
        "adoption_markdown": sha256(adoption_md),
    }
    post_install_path = tmp_path / "nemoclaw_post_install_verification.json"
    post_install_md = tmp_path / "nemoclaw_post_install_verification.md"
    post_install = write_json(
        post_install_path,
        {
            "schema_version": 1,
            "path": str(post_install_path),
            "markdown_path": str(post_install_md),
            "ok": False,
            "status": "failed",
            "generated_at": 1.0,
            "sandbox": "nejumi-taiwan",
            "will_launch_model_inference": False,
            "will_query_wandb": False,
            "will_install_or_onboard": False,
            "command_safety": {
                "ok": True,
                "forbidden_tokens": [
                    "--install",
                    "--onboard",
                    "--upload",
                    "--wandb",
                    "--yes-i-accept-third-party-software",
                ],
                "forbidden_prefixes": [
                    "--upload",
                    "--wandb",
                    "ANTHROPIC_API_KEY=",
                    "GEMINI_API_KEY=",
                    "GOOGLE_API_KEY=",
                    "OPENAI_API_KEY=",
                    "OPENROUTER_",
                    "OPENROUTER_API_KEY=",
                    "WANDB_",
                    "WEAVE_",
                    "XAI_API_KEY=",
                ],
                "forbidden_markers": ["openrouter", "wandb", "weave"],
                "required_step_tokens": {
                    "setup_check": ["install_nemoclaw.sh", "--check-only", "--json"],
                    "protocol_preflight": ["run_openclaw_agent_protocol.py", "preflight"],
                    "canary_readiness": [
                        "check_taiwan_canary_readiness.py",
                        "--require-nemoclaw",
                        "--json",
                    ],
                    "adoption_check": [
                        "check_taiwan_nemoclaw_adoption.py",
                        "--setup-json",
                        "--readiness-json",
                        "--json",
                        "--markdown",
                    ],
                },
                "forbidden_token_count": 0,
                "missing_required_token_count": 0,
                "missing_command_count": 0,
                "records": [
                    {
                        "name": "setup_check",
                        "ok": True,
                        "forbidden_tokens": [],
                        "missing_required_tokens": [],
                    },
                    {
                        "name": "protocol_preflight",
                        "ok": True,
                        "forbidden_tokens": [],
                        "missing_required_tokens": [],
                    },
                    {
                        "name": "canary_readiness",
                        "ok": True,
                        "forbidden_tokens": [],
                        "missing_required_tokens": [],
                    },
                    {
                        "name": "adoption_check",
                        "ok": True,
                        "forbidden_tokens": [],
                        "missing_required_tokens": [],
                    },
                ],
            },
            "outputs": {
                "setup_json": str(setup),
                "preflight_json": str(preflight),
                "readiness_json": str(readiness),
                "adoption_json": str(adoption),
                "adoption_markdown": str(adoption_md),
            },
            "outputs_sha256": output_sha256,
            "steps": [
                {
                    "name": "setup_check",
                    "ok": False,
                    "returncode": 1,
                    "returncode_ok": False,
                    "payload_ok": False,
                    "payload_contract_ok": False,
                    "payload_contract_errors": ["runtime_installed must be true"],
                    "timed_out": False,
                    "payload_status": None,
                    "command": [
                        "scripts/setup/install_nemoclaw.sh",
                        "--check-only",
                        "--json",
                        str(setup),
                    ],
                    "output_json": str(setup),
                    "output_json_sha256": output_sha256["setup_json"],
                },
                {
                    "name": "protocol_preflight",
                    "ok": False,
                    "returncode": 1,
                    "returncode_ok": False,
                    "payload_ok": False,
                    "payload_contract_ok": False,
                    "payload_contract_errors": ["nemoclaw.installed must be true"],
                    "timed_out": False,
                    "payload_status": "missing",
                    "command": [
                        "uv",
                        "run",
                        "python",
                        "scripts/tools/run_openclaw_agent_protocol.py",
                        "preflight",
                    ],
                    "output_json": str(preflight),
                    "output_json_sha256": output_sha256["preflight_json"],
                },
                {
                    "name": "canary_readiness",
                    "ok": False,
                    "returncode": 1,
                    "returncode_ok": False,
                    "payload_ok": False,
                    "payload_contract_ok": False,
                    "payload_contract_errors": ["NeMoClaw command is available must be true"],
                    "timed_out": False,
                    "payload_status": "failed",
                    "command": [
                        "uv",
                        "run",
                        "python",
                        "scripts/tools/check_taiwan_canary_readiness.py",
                        "--require-nemoclaw",
                        "--json",
                        str(readiness),
                    ],
                    "output_json": str(readiness),
                    "output_json_sha256": output_sha256["readiness_json"],
                },
                {
                    "name": "adoption_check",
                    "ok": False,
                    "returncode": 0,
                    "returncode_ok": True,
                    "payload_ok": False,
                    "payload_contract_ok": False,
                    "payload_contract_errors": ["status must be adoptable_for_agentic_math"],
                    "timed_out": False,
                    "payload_status": "not_installed",
                    "command": [
                        "uv",
                        "run",
                        "python",
                        "scripts/tools/check_taiwan_nemoclaw_adoption.py",
                        "--setup-json",
                        str(setup),
                        "--readiness-json",
                        str(readiness),
                        "--json",
                        str(adoption),
                        "--markdown",
                        str(adoption_md),
                    ],
                    "output_json": str(adoption),
                    "output_json_sha256": output_sha256["adoption_json"],
                },
            ],
        },
    )
    post_install_md.write_text("# post install\n", encoding="utf-8")
    operator_docs_path = tmp_path / "nemoclaw_operator_docs_verification.json"
    operator_docs_md = tmp_path / "nemoclaw_operator_docs_verification.md"
    operator_docs = write_json(
        operator_docs_path,
        {
            "schema_version": 1,
            "path": str(operator_docs_path),
            "markdown_path": str(operator_docs_md),
            "ok": True,
            "status": "passed",
            "generated_at": "2026-06-28T00:00:00Z",
            "readme_path": "docs/README_nemoclaw.md",
            "lock_json": "scripts/setup/nemoclaw_installer_lock.json",
            "will_execute_installer": False,
            "will_install_or_onboard": False,
            "will_launch_model_inference": False,
            "will_query_wandb": False,
            "missing_requirements": [],
            "checks": [
                {"name": "readme_exists", "ok": True},
                {"name": "installer_lock_json_valid", "ok": True},
                {"name": "check_only_command", "ok": True},
                {"name": "installer_review_command", "ok": True},
                {"name": "install_and_onboard_command", "ok": True},
                {"name": "post_install_verification_command", "ok": True},
                {"name": "canary_readiness_command", "ok": True},
                {"name": "adoption_check_command", "ok": True},
                {"name": "production_readiness_command", "ok": True},
                {"name": "operator_docs_verifier_command", "ok": True},
                {"name": "no_openrouter_markers", "ok": True},
            ],
        },
    )
    operator_docs_md.write_text("# operator docs\n", encoding="utf-8")
    report = write_json(
        tmp_path / "readiness.json",
        {
            "schema_version": 1,
            "ok": False,
            "status": "not_ready",
            "summary": {
                "gate_count": 1,
                "blocker_count": 1,
                "blockers": ["nemoclaw_readiness"],
            },
            "runner": {
                "report_json": str(tmp_path / "readiness.json"),
                "nemoclaw_post_install_verification": {
                    "path": str(post_install),
                    "markdown_path": str(post_install_md),
                    "ok": False,
                    "status": "failed",
                },
                "nemoclaw_operator_docs_verification": {
                    "path": str(operator_docs),
                    "markdown_path": str(operator_docs_md),
                    "ok": True,
                    "status": "passed",
                    "summary": {"missing_requirements": []},
                },
            },
            "gates": [
                {
                    "name": "nemoclaw_readiness",
                    "ok": False,
                    "blocking": True,
                    "status": "failed",
                    "evidence_paths": [str(setup)],
                }
            ],
        },
    )
    output_dir = tmp_path / "bundle_with_nemoclaw_post_install"
    result = subprocess.run(
        [
            "python3",
            str(BUILD_SCRIPT),
            "--readiness-report",
            str(report),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return output_dir, post_install


def build_bundle_with_nemoclaw_installer_review(tmp_path):
    installer_sha = "a4ebc5710dfd8b10035968fd25562773ad56ec4c73ecf9bfe5c78c77724000e7"
    lock = write_json(
        tmp_path / "nemoclaw_installer_lock.json",
        {
            "schema_version": 1,
            "installer_url": "https://www.nvidia.com/nemoclaw.sh",
            "install_ref": "lkg",
            "sha256": installer_sha,
            "size_bytes": 6356,
            "will_execute_installer": False,
            "will_install_or_onboard": False,
            "will_launch_model_inference": False,
            "will_query_wandb": False,
        },
    )
    install_json = tmp_path / "nemoclaw_install_onboard.json"
    recommended_install_command = (
        "scripts/setup/install_nemoclaw.sh --install --onboard "
        "--sandbox nejumi-taiwan "
        "--provider openai "
        "--policy-tier restricted "
        "--install-ref lkg "
        f"--installer-lock-json {lock} "
        f"--installer-sha256 {installer_sha} "
        f"--installer-review-json {tmp_path / 'nemoclaw_installer_review.json'} "
        "--yes-i-accept-third-party-software "
        f"--json {install_json}"
    )
    review = write_json(
        tmp_path / "nemoclaw_installer_review.json",
        {
            "schema_version": 1,
            "ok": True,
            "status": "reviewed",
            "generated_at": "2026-06-28T01:32:47Z",
            "installer_url": "https://www.nvidia.com/nemoclaw.sh",
            "install_ref": "lkg",
            "lock_json": str(lock),
            "lock_verified": True,
            "expected_sha256": installer_sha,
            "sha256": installer_sha,
            "size_bytes": 6356,
            "will_execute_installer": False,
            "will_install_or_onboard": False,
            "will_launch_model_inference": False,
            "will_query_wandb": False,
            "recommended_install_command": recommended_install_command,
        },
    )
    review_md = tmp_path / "nemoclaw_installer_review.md"
    review_md.write_text("# installer review\n", encoding="utf-8")
    report = write_json(
        tmp_path / "readiness.json",
        {
            "schema_version": 1,
            "ok": False,
            "status": "not_ready",
            "summary": {
                "gate_count": 1,
                "blocker_count": 1,
                "blockers": ["nemoclaw_readiness"],
            },
            "remediation_plan": [
                {
                    "gate": "nemoclaw_readiness",
                    "status": "failed",
                    "next_action": "install",
                    "commands": [
                        (
                            "uv run python scripts/setup/review_nemoclaw_installer.py "
                            "--url https://www.nvidia.com/nemoclaw.sh "
                            "--install-ref lkg "
                            f"--expected-sha256 {installer_sha} "
                            f"--lock-json {lock} "
                            f"--json {review} "
                            f"--markdown {review_md}"
                        ),
                        (
                            "scripts/setup/install_nemoclaw.sh --install --onboard "
                            "--sandbox nejumi-taiwan "
                            "--policy-tier restricted "
                            "--install-ref lkg "
                            f"--installer-lock-json {lock} "
                            f"--installer-sha256 {installer_sha} "
                            f"--installer-review-json {review} "
                            "--yes-i-accept-third-party-software "
                            f"--json {install_json}"
                        ),
                    ],
                }
            ],
            "gates": [
                {
                    "name": "nemoclaw_readiness",
                    "ok": False,
                    "blocking": True,
                    "status": "failed",
                    "evidence_paths": [],
                    "latest_installer_review": {
                        "path": str(review),
                        "generated_at": "2026-06-28T01:32:47Z",
                        "installer_url": "https://www.nvidia.com/nemoclaw.sh",
                        "install_ref": "lkg",
                        "lock_json": str(lock),
                        "lock_verified": True,
                        "expected_sha256": installer_sha,
                        "sha256": installer_sha,
                        "size_bytes": 6356,
                        "status": "reviewed",
                        "ok": True,
                    },
                }
            ],
        },
    )
    output_dir = tmp_path / "bundle_with_nemoclaw_installer_review"
    result = subprocess.run(
        [
            "python3",
            str(BUILD_SCRIPT),
            "--readiness-report",
            str(report),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return output_dir, review, lock, installer_sha


def weave_agents_completion_payload():
    return {
        "ok": True,
        "schema_version": 1,
        "verification_schema_version": 1,
        "status": "passed",
        "generated_at": 1,
        "project_id": "llm-leaderboard/tc-leaderboard",
        "agent_name": "nejumi-taiwan-openclaw",
        "agents_url": "https://wandb.ai/llm-leaderboard/tc-leaderboard/weave/agents",
        "query_source": {
            "kind": "wandb_agents_api",
            "api_base_url": "https://trace.wandb.ai",
            "agents_endpoint": "/agents/query",
            "spans_endpoint": "/agents/spans/query",
            "project_id": "llm-leaderboard/tc-leaderboard",
            "agent_name": "nejumi-taiwan-openclaw",
            "conversation_id": "",
            "conversation_id_contains": "run-1",
            "agents_count": 1,
            "spans_count": 2,
            "matching_span_count": 2,
            "latest_trace_span_count": 2,
        },
        "latest_trace_id": "trace-1",
        "required_evidence": {
            "min_agent_invocations": 1,
            "min_trace_spans": 2,
            "min_message_spans": 1,
            "content_required": True,
            "input_message_required": True,
            "tool_span_required": True,
            "tool_content_required": True,
            "trace_timestamp_quality_required": True,
            "trace_final_answer_order_required": True,
            "usage_required": True,
            "no_error_spans_required": True,
            "required_texts": [],
            "conversation_id": "",
            "conversation_id_contains": "run-1",
            "expected_request_models": ["gpt-4.1-mini-2025-04-14"],
        },
        "content_capture_health": {
            "span_count_checked": 2,
            "message_span_count": 1,
            "message_spans_with_content": 1,
            "message_spans_with_input": 1,
            "tool_span_count": 1,
            "tool_spans_with_content": 1,
            "spans_with_valid_timestamps": 2,
            "spans_with_invalid_timestamps": 0,
            "trace_input_tokens": 10,
            "trace_output_tokens": 5,
            "required_text_count": 0,
            "request_model_count": 1,
        },
        "latest_trace_spans_chronological": [
            {
                "started_at": "2026-06-28T00:00:00Z",
                "ended_at": "2026-06-28T00:00:01Z",
                "span_name": "chat",
                "operation_name": "chat",
                "agent_name": "nejumi-taiwan-openclaw",
                "trace_id": "trace-1",
                "span_id": "span-1",
                "parent_span_id": None,
                "error_type": None,
                "has_input_messages": True,
                "has_output_messages": True,
                "request_model": "gpt-4.1-mini-2025-04-14",
            },
            {
                "started_at": "2026-06-28T00:00:02Z",
                "ended_at": "2026-06-28T00:00:03Z",
                "span_name": "python.exec",
                "operation_name": "execute_tool",
                "agent_name": "nejumi-taiwan-openclaw",
                "trace_id": "trace-1",
                "span_id": "span-2",
                "parent_span_id": "span-1",
                "tool_name": "python",
                "error_type": None,
                "has_tool_call_arguments": True,
                "has_tool_call_result": True,
                "request_model": "gpt-4.1-mini-2025-04-14",
            },
        ],
        "checks": [
            {"name": "agent_present", "ok": True, "detail": "agent is present"},
            {"name": "latest_trace", "ok": True, "detail": "latest trace id is present"},
            {"name": "trace_span_count", "ok": True, "detail": "latest trace has enough spans"},
            {
                "name": "request_model",
                "ok": True,
                "detail": "latest trace request_model matches an expected model alias",
                "expected_request_models": ["gpt-4.1-mini-2025-04-14"],
                "observed_request_models": ["gpt-4.1-mini-2025-04-14"],
            },
            {"name": "message_content_capture", "ok": True, "detail": "message content is visible"},
            {"name": "input_message_capture", "ok": True, "detail": "user/problem input is visible"},
            {"name": "tool_span_count", "ok": True, "detail": "tool spans are present"},
            {"name": "tool_content_capture", "ok": True, "detail": "tool content is visible"},
            {
                "name": "usage",
                "ok": True,
                "agent_input_tokens": 0,
                "agent_output_tokens": 0,
                "trace_input_tokens": 10,
                "trace_output_tokens": 5,
            },
            {
                "name": "trace_timestamp_quality",
                "ok": True,
                "detail": "all latest trace spans have parseable start/end timestamps",
            },
            {"name": "trace_order", "ok": True, "detail": "tool spans do not start before messages"},
            {
                "name": "trace_user_message_order",
                "ok": True,
                "detail": "tool spans do not start before visible user/problem input",
            },
            {
                "name": "trace_final_answer_order",
                "ok": True,
                "detail": "no final-answer marker and tool-order conflict was detected",
            },
            {"name": "trace_errors", "ok": True, "detail": "latest trace has no error spans"},
        ],
    }


def test_validate_weave_agents_completion_payload_rejects_bad_schema_version():
    module = load_verify_module()
    payload = weave_agents_completion_payload()
    payload["schema_version"] = 2

    errors = module.validate_weave_agents_completion_payload(
        payload,
        label="weave proof",
        expected_agent_name="nejumi-taiwan-openclaw",
        expected_latest_trace_id="trace-1",
        expected_run_id="run-1",
    )

    assert "weave proof schema_version is not 1" in errors


def test_validate_weave_agents_completion_payload_rejects_missing_tool_usage_requirements():
    module = load_verify_module()
    payload = weave_agents_completion_payload()
    payload["required_evidence"]["tool_content_required"] = False
    payload["required_evidence"]["usage_required"] = False
    payload["checks"] = [
        check
        for check in payload["checks"]
        if check.get("name") not in {"tool_content_capture", "usage"}
    ]

    errors = module.validate_weave_agents_completion_payload(
        payload,
        label="weave proof",
        expected_agent_name="nejumi-taiwan-openclaw",
        expected_latest_trace_id="trace-1",
        expected_run_id="run-1",
    )

    assert "weave proof required_evidence.tool_content_required is not true" in errors
    assert "weave proof required_evidence.usage_required is not true" in errors
    assert "weave proof checks missing required check: tool_content_capture" in errors
    assert "weave proof checks missing required check: usage" in errors


def build_bundle_with_weave_agents_completion(tmp_path):
    completion = write_json(
        tmp_path / "weave_agents_completion.json",
        weave_agents_completion_payload(),
    )
    source_review = write_json(
        tmp_path / "canary_agentic_paid_run_review.before_weave_sync.json",
        {
            "status": "completed",
            "phase": "agentic",
            "verify_weave_agents": True,
            "runs": [
                {
                    "config": "config.yaml",
                    "log_path": "run.log",
                    "wandb_run_id": "run-1",
                    "returncode": 0,
                    "started_at": 1,
                    "ended_at": 2,
                }
            ],
        },
    )
    source_review_sha256 = sha256(source_review)
    sync_dry_run_report = write_json(
        tmp_path / "weave_agents_completion.sync_dry_run.json",
        {
            "ok": True,
            "status": "synced",
            "generated_at": 1,
            "review_path": str(source_review),
            "source_review_sha256": source_review_sha256,
            "output_path": "",
            "in_place": False,
            "dry_run": True,
            "entry_count": 1,
            "entries": [
                {
                    "ok": True,
                    "run_id": "run-1",
                    "path": str(completion),
                    "agent_name": "nejumi-taiwan-openclaw",
                    "verification_schema_version": 1,
                    "latest_trace_id": "trace-1",
                    "checks_valid": True,
                    "trace_present": True,
                    "run_scope_proven": True,
                    "request_model_proven": True,
                    "expected_request_models": ["gpt-4.1-mini-2025-04-14"],
                    "observed_request_models": ["gpt-4.1-mini-2025-04-14"],
                    "span_request_models": ["gpt-4.1-mini-2025-04-14"],
                    "conversation_id": "",
                    "conversation_id_contains": "run-1",
                    "query_source_kind": "wandb_agents_api",
                    "query_source_api_base_url": "https://trace.wandb.ai",
                    "query_source_agents_endpoint": "/agents/query",
                    "query_source_spans_endpoint": "/agents/spans/query",
                    "query_source_project_id": "llm-leaderboard/tc-leaderboard",
                }
            ],
            "change_count": 1,
            "changes": [
                {
                    "target": "run",
                    "action": "added",
                    "run_id": "run-1",
                    "agent_name": "nejumi-taiwan-openclaw",
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
    review = write_json(
        tmp_path / "canary_agentic_paid_run_review.json",
        {
            "status": "completed",
            "phase": "agentic",
            "verify_weave_agents": True,
            "runs": [
                {
                    "config": "config.yaml",
                    "log_path": "run.log",
                    "wandb_run_id": "run-1",
                    "returncode": 0,
                    "started_at": 1,
                    "ended_at": 2,
                    "weave_agents_completion": {
                        "path": str(completion),
                        "agent_name": "nejumi-taiwan-openclaw",
                        "run_id": "run-1",
                        "ok": True,
                        "latest_trace_id": "trace-1",
                        "sync_dry_run_report_json": str(sync_dry_run_report),
                        "sync_dry_run_source_review_json": str(source_review),
                        "sync_dry_run_source_review_sha256": source_review_sha256,
                    },
                }
            ],
        },
    )
    report = write_json(
        tmp_path / "readiness.json",
        {
            "schema_version": 1,
            "ok": False,
            "status": "not_ready",
            "summary": {
                "gate_count": 1,
                "blocker_count": 1,
                "blockers": ["one_model_full_canary"],
            },
            "gates": [
                {
                    "name": "one_model_full_canary",
                    "ok": False,
                    "blocking": True,
                    "status": "weave_agents_completion_not_proven",
                    "evidence_paths": [str(review)],
                    "records": [
                        {
                            "path": str(review),
                            "phase": "agentic",
                            "status": "completed",
                            "verify_weave_agents": True,
                            "weave_agents_completion_max_age_seconds": 86400,
                            "weave_agents_completion_entries": [
                                {
                                    "path": str(completion),
                                    "review_path": str(review),
                                    "agent_name": "nejumi-taiwan-openclaw",
                                    "run_id": "run-1",
                                    "ok": True,
                                    "entry_ok": True,
                                    "verified": True,
                                    "schema_valid": True,
                                    "checks_valid": True,
                                    "trace_present": True,
                                    "fresh": True,
                                    "latest_trace_id": "trace-1",
                                    "sync_dry_run_report_json": str(sync_dry_run_report),
                                    "sync_dry_run_source_review_json": str(source_review),
                                    "sync_dry_run_source_review_sha256": source_review_sha256,
                                }
                            ],
                        }
                    ],
                }
            ],
        },
    )
    output_dir = tmp_path / "bundle_with_weave_agents"
    result = subprocess.run(
        [
            "python3",
            str(BUILD_SCRIPT),
            "--readiness-report",
            str(report),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return output_dir, completion


def build_bundle_with_weave_content_canary(tmp_path):
    canary_id = "TEST_CANARY"
    task_id = "weave_agents_content_canary_TEST_CANARY"
    verifier_payload = weave_agents_completion_payload()
    verifier_payload["query_source"]["conversation_id_contains"] = task_id
    verifier_payload["required_evidence"]["conversation_id_contains"] = task_id
    verifier_payload["required_evidence"]["required_texts"] = [
        canary_id,
        f"CANARY_RESULT {canary_id} 91",
    ]
    verifier_payload["required_evidence"]["expected_request_models"] = [
        "gpt-4.1-mini-2025-04-14"
    ]
    verifier_payload["content_capture_health"]["required_text_count"] = 2
    verifier_payload["checks"].append(
        {
            "name": "required_text_capture",
            "ok": True,
            "detail": "required canary text is visible",
        }
    )
    for span in verifier_payload["latest_trace_spans_chronological"]:
        span["conversation_id"] = task_id
    verifier = write_json(
        tmp_path / "weave_content_canary_verifier.json",
        verifier_payload,
    )
    plan = write_json(
        tmp_path / "weave_content_canary_plan.json",
        {
            "canary_id": canary_id,
            "task_id": task_id,
            "will_call_paid_model_api": True,
            "will_execute_external_actions": True,
            "model": "openai-direct/test-mini",
            "thinking": "low",
            "nemoclaw": {
                "required": True,
                "enabled": True,
                "bin": "nemoclaw",
                "sandbox": "nejumi-taiwan",
                "workdir": "/sandbox",
            },
            "nemoclaw_openclaw_config_preflight": {
                "required_before_openclaw": True,
                "ran": True,
                "ok": True,
                "model": "openai-direct/test-mini",
                "provider": "openai-direct",
                "model_id": "test-mini",
                "config_path": "/sandbox/.openclaw/openclaw.json",
                "command": [
                    "nemoclaw",
                    "sandbox",
                    "exec",
                    "nejumi-taiwan",
                    "--no-tty",
                    "--timeout",
                    "30",
                    "--",
                    "cat",
                    "/sandbox/.openclaw/openclaw.json",
                ],
                "returncode": 0,
                "checks": [
                    {
                        "name": (
                            "NeMoClaw sandbox OpenClaw config is readable: "
                            "/sandbox/.openclaw/openclaw.json"
                        ),
                        "ok": True,
                        "detail": "bytes=1234",
                    },
                    {
                        "name": "NeMoClaw sandbox OpenClaw openai-direct provider exists",
                        "ok": True,
                        "detail": "present",
                    },
                    {
                        "name": (
                            "NeMoClaw sandbox OpenClaw model is registered: "
                            "openai-direct/test-mini"
                        ),
                        "ok": True,
                        "detail": '["test-mini"]',
                    },
                    {
                        "name": "NeMoClaw sandbox OpenClaw Weave plugin is enabled",
                        "ok": True,
                        "detail": "True",
                    },
                ],
                "errors": [],
            },
            "agent_name": "nejumi-taiwan-openclaw",
            "entity": "llm-leaderboard",
            "project": "tc-leaderboard",
            "agents_diagnostic_file": str(tmp_path / "weave_content_canary.agents.json"),
            "verification_requirements": {
                "require_content": True,
                "require_tool_span": True,
                "require_tool_content": True,
                "require_usage": False,
                "expected_request_models": ["gpt-4.1-mini-2025-04-14"],
                "required_texts": [
                    canary_id,
                    f"CANARY_RESULT {canary_id} 91",
                ],
            },
        },
    )
    command_result = write_json(
        tmp_path / "weave_content_canary.command_result.json",
        {"ok": True, "returncode": 0},
    )
    sidecar = write_json(
        tmp_path / "weave_content_canary_openclaw_result.json",
        {"ok": True, "stdout_json": {"meta": {"agentMeta": {"sessionKey": task_id}}}},
    )
    agents_diagnostic = write_json(
        tmp_path / "weave_content_canary.agents.json",
        {
            "diagnostic_schema_version": 1,
            "generated_at": 1,
            "project_id": "llm-leaderboard/tc-leaderboard",
            "agent_name_filter": "nejumi-taiwan-openclaw",
            "agents_url": "https://wandb.ai/llm-leaderboard/tc-leaderboard/weave/agents",
            "query_source": {
                "kind": "wandb_agents_api",
                "api_base_url": "https://trace.wandb.ai",
                "agents_endpoint": "/agents/query",
                "spans_endpoint": "/agents/spans/query",
                "project_id": "llm-leaderboard/tc-leaderboard",
                "agent_name": "nejumi-taiwan-openclaw",
                "conversation_id": "",
                "conversation_id_contains": task_id,
                "limit": 30,
                "span_limit": 120,
                "agents_count": 1,
                "spans_count": 3,
                "matching_span_count": 3,
                "latest_trace_span_count": 3,
            },
            "agents": [{"agent_name": "nejumi-taiwan-openclaw"}],
            "total_count": 1,
            "latest_trace_id": "trace-1",
            "latest_trace_spans_chronological": [
                {
                    "started_at": "2026-06-28T00:00:00Z",
                    "ended_at": "2026-06-28T00:00:01Z",
                    "span_name": "chat",
                    "operation_name": "chat",
                    "agent_name": "nejumi-taiwan-openclaw",
                    "conversation_id": task_id,
                    "trace_id": "trace-1",
                    "span_id": "span-1",
                    "parent_span_id": None,
                    "tool_name": None,
                    "error_type": None,
                    "has_input_messages": True,
                    "has_output_messages": False,
                    "input_message_count": 1,
                    "output_message_count": 0,
                    "has_tool_call_arguments": False,
                    "has_tool_call_result": False,
                    "has_final_answer_marker": False,
                },
                {
                    "started_at": "2026-06-28T00:00:02Z",
                    "ended_at": "2026-06-28T00:00:03Z",
                    "span_name": "python.exec",
                    "operation_name": "execute_tool",
                    "agent_name": "nejumi-taiwan-openclaw",
                    "conversation_id": task_id,
                    "trace_id": "trace-1",
                    "span_id": "span-2",
                    "parent_span_id": "span-1",
                    "tool_name": "python",
                    "error_type": None,
                    "has_input_messages": False,
                    "has_output_messages": False,
                    "input_message_count": 0,
                    "output_message_count": 0,
                    "has_tool_call_arguments": True,
                    "has_tool_call_result": True,
                    "has_final_answer_marker": False,
                },
                {
                    "started_at": "2026-06-28T00:00:04Z",
                    "ended_at": "2026-06-28T00:00:05Z",
                    "span_name": "chat",
                    "operation_name": "chat",
                    "agent_name": "nejumi-taiwan-openclaw",
                    "conversation_id": task_id,
                    "trace_id": "trace-1",
                    "span_id": "span-3",
                    "parent_span_id": "span-1",
                    "tool_name": None,
                    "error_type": None,
                    "has_input_messages": False,
                    "has_output_messages": True,
                    "input_message_count": 0,
                    "output_message_count": 1,
                    "has_tool_call_arguments": False,
                    "has_tool_call_result": False,
                    "has_final_answer_marker": True,
                },
            ],
            "content_capture_health": {
                "span_count_checked": 3,
                "message_span_count": 2,
                "message_spans_with_content": 2,
                "message_spans_with_input": 1,
                "tool_span_count": 1,
                "tool_spans_with_content": 1,
                "final_answer_span_count": 1,
                "spans_with_valid_timestamps": 3,
                "spans_with_invalid_timestamps": 0,
                "trace_timestamp_quality_ok": True,
                "trace_order_ok": True,
                "trace_user_message_order_ok": True,
                "trace_final_answer_order_ok": True,
            },
            "trace_order_health": {
                "timestamp_quality_ok": True,
                "timestamp_issue_count": 0,
                "timestamp_issues": [],
                "message_span_count": 2,
                "message_spans_with_input": 1,
                "tool_span_count": 1,
                "final_answer_span_count": 1,
                "trace_order_ok": True,
                "trace_user_message_order_ok": True,
                "trace_final_answer_order_ok": True,
                "order_issues": [],
            },
            "latest_spans_api_order": [],
        },
    )
    prompt = tmp_path / "weave_content_canary_prompt.md"
    prompt.write_text(
        "\n".join(
            [
                "# W&B Agents Content Capture Canary",
                f"Canary ID: {canary_id}",
                "Use the Python execution tool exactly once to compute `7 * 13`.",
                f"CANARY_RESULT {canary_id} 91",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    gate = write_json(
        tmp_path / "weave_content_canary.gate.json",
        {
            "ok": True,
            "gate": "weave_agents_content_canary",
            "status": "passed",
            "detail": "fresh Weave Agents canary has trace, content, tool content, and usage",
            "generated_at": 1,
            "canary_id": canary_id,
            "task_id": task_id,
            "model": "openai-direct/test-mini",
            "thinking": "low",
            "agent_name": "nejumi-taiwan-openclaw",
            "entity": "llm-leaderboard",
            "project": "tc-leaderboard",
            "expected_request_models": ["gpt-4.1-mini-2025-04-14"],
            "observed_request_models": ["gpt-4.1-mini-2025-04-14"],
            "span_request_models": ["gpt-4.1-mini-2025-04-14"],
            "request_model_proven": True,
            "nemoclaw": {
                "required": True,
                "enabled": True,
                "bin": "nemoclaw",
                "sandbox": "nejumi-taiwan",
                "workdir": "/sandbox",
            },
            "nemoclaw_openclaw_config_preflight": {
                "required_before_openclaw": True,
                "ran": True,
                "ok": True,
                "model": "openai-direct/test-mini",
                "provider": "openai-direct",
                "model_id": "test-mini",
                "config_path": "/sandbox/.openclaw/openclaw.json",
                "command": [
                    "nemoclaw",
                    "sandbox",
                    "exec",
                    "nejumi-taiwan",
                    "--no-tty",
                    "--timeout",
                    "30",
                    "--",
                    "cat",
                    "/sandbox/.openclaw/openclaw.json",
                ],
                "returncode": 0,
                "checks": [
                    {
                        "name": (
                            "NeMoClaw sandbox OpenClaw config is readable: "
                            "/sandbox/.openclaw/openclaw.json"
                        ),
                        "ok": True,
                        "detail": "bytes=1234",
                    },
                    {
                        "name": "NeMoClaw sandbox OpenClaw openai-direct provider exists",
                        "ok": True,
                        "detail": "present",
                    },
                    {
                        "name": (
                            "NeMoClaw sandbox OpenClaw model is registered: "
                            "openai-direct/test-mini"
                        ),
                        "ok": True,
                        "detail": '["test-mini"]',
                    },
                    {
                        "name": "NeMoClaw sandbox OpenClaw Weave plugin is enabled",
                        "ok": True,
                        "detail": "True",
                    },
                ],
                "errors": [],
            },
            "weave_verifier_ok": True,
            "weave_verifier_schema_version": 1,
            "weave_verifier_latest_trace_id": "trace-1",
            "weave_verifier_validation_issues": [],
            "agents_diagnostic_ok": True,
            "agents_diagnostic_schema_version": 1,
            "agents_diagnostic_latest_trace_id": "trace-1",
            "agents_diagnostic_validation_issues": [],
            "content_capture_health": {
                "message_spans_with_input": 1,
                "tool_spans_with_content": 1,
                "spans_with_valid_timestamps": 2,
                "spans_with_invalid_timestamps": 0,
                "request_model_count": 1,
            },
            "paths": {
                "plan_file": str(plan),
                "command_result_file": str(command_result),
                "verifier_json": str(verifier),
                "agents_diagnostic_json": str(agents_diagnostic),
                "expected_sidecar": str(sidecar),
                "prompt_file": str(prompt),
            },
        },
    )
    report = write_json(
        tmp_path / "readiness.json",
        {
            "schema_version": 1,
            "ok": True,
            "status": "ready",
            "summary": {
                "gate_count": 1,
                "blocker_count": 0,
                "blockers": [],
            },
            "gates": [
                {
                    "name": "weave_content_canary",
                    "ok": True,
                    "blocking": True,
                    "status": "passed",
                    "detail": "passed",
                    "evidence_paths": [str(gate)],
                }
            ],
        },
    )
    output_dir = tmp_path / "bundle_with_weave_content_canary"
    result = subprocess.run(
        [
            "python3",
            str(BUILD_SCRIPT),
            "--readiness-report",
            str(report),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return output_dir, gate, verifier


def test_verify_release_evidence_bundle_accepts_integrity_for_not_ready_bundle(tmp_path):
    bundle = build_bundle(tmp_path, readiness_ok=False)

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["ok"] is True
    assert payload["status"] == "passed"
    assert payload["integrity_ok"] is True
    assert payload["readiness_ok"] is False
    assert payload["error_count"] == 0


def test_verify_release_evidence_bundle_accepts_weave_content_canary_proof(tmp_path):
    bundle, gate, verifier = build_bundle_with_weave_content_canary(tmp_path)

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle), "--require-ready"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is True
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    sources = {record["source_path"]: record for record in manifest["files"]}
    assert "gate:weave_content_canary:evidence" in sources[str(gate)]["roles"]
    assert (
        "gate:weave_content_canary:evidence:verifier_json"
        in sources[str(verifier)]["roles"]
    )
    assert any(
        "gate:weave_content_canary:evidence:agents_diagnostic_json"
        in record.get("roles", [])
        for record in sources.values()
    )


def test_verify_release_evidence_bundle_rejects_weave_content_canary_without_nemoclaw_openclaw_config_preflight(
    tmp_path,
):
    bundle, gate, _verifier = build_bundle_with_weave_content_canary(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    gate_record = next(
        record for record in manifest["files"] if record["source_path"] == str(gate)
    )
    bundled_gate = bundle / gate_record["bundle_path"]
    payload = json.loads(bundled_gate.read_text(encoding="utf-8"))
    payload.pop("nemoclaw_openclaw_config_preflight")
    bundled_gate.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, gate_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "nemoclaw_openclaw_config_preflight is not an object" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_weave_content_canary_failed_nemoclaw_openclaw_config_preflight(
    tmp_path,
):
    bundle, gate, _verifier = build_bundle_with_weave_content_canary(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    gate_record = next(
        record for record in manifest["files"] if record["source_path"] == str(gate)
    )
    bundled_gate = bundle / gate_record["bundle_path"]
    payload = json.loads(bundled_gate.read_text(encoding="utf-8"))
    payload["nemoclaw_openclaw_config_preflight"]["ok"] = False
    payload["nemoclaw_openclaw_config_preflight"]["errors"] = [
        "NeMoClaw sandbox OpenClaw model is registered: openai-direct/test-mini",
    ]
    bundled_gate.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, gate_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "nemoclaw_openclaw_config_preflight.ok is not true" in error
        for error in payload["errors"]
    )
    assert any(
        "nemoclaw_openclaw_config_preflight.errors is not empty" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_legacy_weave_content_canary_proof(tmp_path):
    bundle, gate, _verifier = build_bundle_with_weave_content_canary(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    gate_record = next(
        record for record in manifest["files"] if record["source_path"] == str(gate)
    )
    bundled_gate = bundle / gate_record["bundle_path"]
    payload = json.loads(bundled_gate.read_text(encoding="utf-8"))
    payload.pop("weave_verifier_schema_version")
    bundled_gate.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, gate_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "weave_verifier_schema_version is not 1" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_weave_content_canary_without_request_model_proof(tmp_path):
    bundle, gate, _verifier = build_bundle_with_weave_content_canary(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    gate_record = next(
        record for record in manifest["files"] if record["source_path"] == str(gate)
    )
    bundled_gate = bundle / gate_record["bundle_path"]
    payload = json.loads(bundled_gate.read_text(encoding="utf-8"))
    payload.pop("expected_request_models")
    payload.pop("observed_request_models")
    payload.pop("span_request_models")
    payload["request_model_proven"] = False
    payload["content_capture_health"].pop("request_model_count")
    bundled_gate.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, gate_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any("expected_request_models is not a non-empty list" in error for error in payload["errors"])
    assert any("request_model_proven is not true" in error for error in payload["errors"])


def test_verify_release_evidence_bundle_rejects_weave_content_canary_without_verifier_json(tmp_path):
    bundle, gate, _verifier = build_bundle_with_weave_content_canary(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    gate_record = next(
        record for record in manifest["files"] if record["source_path"] == str(gate)
    )
    bundled_gate = bundle / gate_record["bundle_path"]
    payload = json.loads(bundled_gate.read_text(encoding="utf-8"))
    payload["paths"].pop("verifier_json")
    bundled_gate.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, gate_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "paths.verifier_json is missing for a passed gate" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_weave_content_canary_without_plan_file(tmp_path):
    bundle, gate, _verifier = build_bundle_with_weave_content_canary(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    gate_record = next(
        record for record in manifest["files"] if record["source_path"] == str(gate)
    )
    bundled_gate = bundle / gate_record["bundle_path"]
    payload = json.loads(bundled_gate.read_text(encoding="utf-8"))
    payload["paths"].pop("plan_file")
    bundled_gate.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, gate_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "paths.plan_file is missing for a passed gate" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_weave_content_canary_verifier_scope_mismatch(tmp_path):
    bundle, _gate, verifier = build_bundle_with_weave_content_canary(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    verifier_record = next(
        record for record in manifest["files"] if record["source_path"] == str(verifier)
    )
    bundled_verifier = bundle / verifier_record["bundle_path"]
    payload = json.loads(bundled_verifier.read_text(encoding="utf-8"))
    payload["required_evidence"]["conversation_id_contains"] = "other-task"
    payload["query_source"]["conversation_id_contains"] = "other-task"
    for span in payload["latest_trace_spans_chronological"]:
        span["conversation_id"] = "other-task"
    bundled_verifier.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, verifier_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "verifier conversation scope does not include task_id" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_weave_content_canary_required_text_mismatch(tmp_path):
    bundle, _gate, verifier = build_bundle_with_weave_content_canary(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    verifier_record = next(
        record for record in manifest["files"] if record["source_path"] == str(verifier)
    )
    bundled_verifier = bundle / verifier_record["bundle_path"]
    payload = json.loads(bundled_verifier.read_text(encoding="utf-8"))
    payload["required_evidence"]["required_texts"] = ["TEST_CANARY"]
    payload["content_capture_health"]["required_text_count"] = 1
    bundled_verifier.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, verifier_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "verifier.required_evidence.required_texts does not include 'CANARY_RESULT TEST_CANARY 91'"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_require_ready_fails_for_not_ready_bundle(tmp_path):
    bundle = build_bundle(tmp_path, readiness_ok=False)

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle), "--require-ready"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["ok"] is False
    assert payload["status"] == "failed"
    assert payload["integrity_ok"] is True
    assert payload["readiness_ok"] is False
    assert payload["error_count"] == len(payload["errors"])
    assert "readiness_ok is false" in payload["errors"]


def test_verify_release_evidence_bundle_detects_hash_mismatch(tmp_path):
    bundle = build_bundle(tmp_path, readiness_ok=True)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    first_file = bundle / manifest["files"][0]["bundle_path"]
    first_file.write_text("tampered", encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle), "--require-ready"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any("sha256 mismatch" in error for error in payload["errors"])


def test_verify_release_evidence_bundle_rejects_manifest_missing_schema_version(tmp_path):
    bundle = build_bundle(tmp_path, readiness_ok=True)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("schema_version")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert "manifest schema_version must be 1" in payload["errors"]


def test_verify_release_evidence_bundle_rejects_readiness_report_missing_schema_version(tmp_path):
    bundle = build_bundle(tmp_path, readiness_ok=True)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    report_record = next(
        record
        for record in manifest["files"]
        if "production_readiness_report" in record.get("roles", [])
    )
    bundled_report = bundle / report_record["bundle_path"]
    payload = json.loads(bundled_report.read_text(encoding="utf-8"))
    payload.pop("schema_version")
    bundled_report.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, report_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert "production readiness report schema_version must be 1" in payload["errors"]


def test_verify_release_evidence_bundle_rejects_manifest_without_current_gate(tmp_path):
    bundle = build_bundle(tmp_path, readiness_ok=True)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("current_gate")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert "manifest current_gate is not an object" in payload["errors"]


def test_verify_release_evidence_bundle_rejects_manifest_current_gate_blocker_mismatch(tmp_path):
    bundle = build_bundle(tmp_path, readiness_ok=False)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["current_gate"]["blocking_gates"] = []
    manifest["current_gate"]["blocker_count"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "manifest blocker_count does not match current_gate blocker_count"
        in payload["errors"]
    )
    assert (
        "manifest blocking_gates does not match current_gate blocking_gates"
        in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_manifest_gate_count_mismatch(tmp_path):
    bundle = build_bundle(tmp_path, readiness_ok=True)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["gate_count"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert "manifest gate_count does not match current_gate gate_count" in payload["errors"]
    assert "manifest gate_count does not match gates length" in payload["errors"]


def test_verify_release_evidence_bundle_rejects_manifest_current_gate_status_mismatch(tmp_path):
    bundle = build_bundle(tmp_path, readiness_ok=False)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["current_gate"]["status"] = "stale"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert "manifest status does not match current_gate status" in payload["errors"]


def test_verify_release_evidence_bundle_rejects_manifest_without_completion_contract(tmp_path):
    bundle = build_bundle(tmp_path, readiness_ok=True)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["current_gate"].pop("wandb_completion_contract")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert "manifest current_gate missing wandb_completion_contract" in payload["errors"]
    assert "manifest current_gate wandb_completion_contract is not an object" in payload["errors"]


def test_verify_release_evidence_bundle_rejects_missing_wandb_completion_contract_summary_section(
    tmp_path,
):
    bundle, _attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    summary_path = bundle / "summary.md"
    summary_text = summary_path.read_text(encoding="utf-8")
    summary_path.write_text(
        summary_text.replace(
            "## W&B Completion Contract",
            "## W&B Completion Summary",
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, "summary.md")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert "summary.md missing required section: ## W&B Completion Contract" in payload["errors"]


def test_verify_release_evidence_bundle_rejects_wandb_completion_contract_summary_row_mismatch(
    tmp_path,
):
    bundle, _attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    summary_path = bundle / "summary.md"
    summary_text = summary_path.read_text(encoding="utf-8")
    expected_row = next(
        line
        for line in summary_text.splitlines()
        if line.startswith("| agentic_math | True |")
    )
    summary_path.write_text(
        summary_text.replace(
            expected_row,
            expected_row.replace("| True |", "| False |", 1),
            1,
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, "summary.md")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "summary.md missing W&B completion contract content "
        f"from current_gate: {expected_row}"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_manifest_without_operator_steps(tmp_path):
    bundle = build_bundle(tmp_path, readiness_ok=True)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["current_gate"].pop("operator_next_steps")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert "manifest current_gate missing operator_next_steps" in payload["errors"]
    assert "manifest current_gate operator_next_steps is not an object" in payload["errors"]


def test_verify_release_evidence_bundle_rejects_manifest_without_remediation_plan(tmp_path):
    bundle = build_bundle(tmp_path, readiness_ok=True)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["current_gate"].pop("remediation_plan")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert "manifest current_gate missing remediation_plan" in payload["errors"]
    assert "manifest current_gate remediation_plan is not a list" in payload["errors"]


def test_verify_release_evidence_bundle_rejects_manifest_without_operator_plan(tmp_path):
    bundle = build_bundle(tmp_path, readiness_ok=True)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("operator_plan")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert "manifest operator_plan is not an object" in payload["errors"]


def test_verify_release_evidence_bundle_rejects_manifest_without_external_action_approval_packet(tmp_path):
    bundle = build_bundle(tmp_path, readiness_ok=True)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("external_action_approval_packet")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert "manifest external_action_approval_packet is not an object" in payload["errors"]


def test_verify_release_evidence_bundle_rejects_operator_plan_not_listed_in_files(tmp_path):
    bundle = build_bundle(tmp_path, readiness_ok=True)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [
        record
        for record in manifest["files"]
        if record.get("bundle_path") != manifest["operator_plan"]["json"]
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert "manifest operator_plan json is not listed in files" in payload["errors"]


def test_verify_release_evidence_bundle_rejects_operator_plan_payload_status_mismatch(tmp_path):
    bundle = build_bundle(tmp_path, readiness_ok=True)
    operator_plan_path = bundle / "operator_plan.json"
    payload = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    payload["status"] = "stale"
    operator_plan_path.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert "operator_plan status does not match manifest" in payload["errors"]


def test_verify_release_evidence_bundle_rejects_operator_steps_payload_mismatch(tmp_path):
    bundle = build_bundle(tmp_path, readiness_ok=True)
    operator_plan_path = bundle / "operator_plan.json"
    payload = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    payload["operator_next_steps"]["step_count"] = 999
    operator_plan_path.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "operator_plan operator_next_steps does not match manifest current_gate"
        in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_operator_external_action_checklist_mismatch(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    operator_plan_path = bundle / "operator_plan.json"
    payload = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    payload["external_action_checklist"]["external_action_item_count"] = 0
    operator_plan_path.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "operator_plan external_action_checklist does not match operator_next_steps"
        in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_missing_operator_renderer(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    operator_plan_path = bundle / "operator_plan.json"
    payload = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    payload.pop("operator_execution_plan_renderer")
    operator_plan_path.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert "operator_plan operator_execution_plan_renderer is not an object" in payload["errors"]


def test_verify_release_evidence_bundle_rejects_operator_renderer_command_mismatch(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    operator_plan_path = bundle / "operator_plan.json"
    payload = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    renderer = payload["operator_execution_plan_renderer"]
    renderer["require_ready_command_template"] = renderer[
        "require_ready_command_template"
    ].replace(" --require-ready", "")
    operator_plan_path.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "operator_plan renderer require_ready command missing --require-ready"
        in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_operator_renderer_missing_approval_report(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    operator_plan_path = bundle / "operator_plan.json"
    payload = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    renderer = payload["operator_execution_plan_renderer"]
    approval_template = renderer["approval_report_json_template"]
    renderer["require_ready_command_template"] = renderer[
        "require_ready_command_template"
    ].replace(f" --external-action-approval-report-json {approval_template}", "")
    operator_plan_path.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "operator_plan renderer require_ready command missing --external-action-approval-report-json"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_operator_renderer_missing_approval_source_packet(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    operator_plan_path = bundle / "operator_plan.json"
    payload = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    renderer = payload["operator_execution_plan_renderer"]
    source_packet_template = renderer["approval_source_packet_json_template"]
    renderer.pop("approval_source_packet_json_template")
    renderer["require_ready_command_template"] = renderer[
        "require_ready_command_template"
    ].replace(
        f" --external-action-approval-source-packet-json {source_packet_template}",
        "",
    )
    operator_plan_path.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "operator_plan renderer approval_source_packet_json_template is missing"
        in error
        for error in payload["errors"]
    )
    assert any(
        "operator_plan renderer require_ready command missing --external-action-approval-source-packet-json"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_operator_plan_release_gate_mismatch(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    attach_valid_release_gate_pointer(bundle, tmp_path)
    operator_plan_path = bundle / "operator_plan.json"
    payload = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    payload["source_release_gate_json"] = str(
        tmp_path / "taiwan_release_gate_20260628T050000Z.json"
    )
    operator_plan_path.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "operator_plan source_release_gate_json does not match manifest release_gate_pointer"
        in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_operator_renderer_missing_release_gate_binding(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    pointer = attach_valid_release_gate_pointer(bundle, tmp_path)
    operator_plan_path = bundle / "operator_plan.json"
    payload = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    renderer = payload["operator_execution_plan_renderer"]
    renderer.pop("release_gate_json_template")
    renderer["safety"].pop("requires_release_gate_match_for_shell_script")
    for key in ("review_command_template", "require_ready_command_template"):
        renderer[key] = renderer[key].replace(
            f" --release-gate-json {pointer['release_gate_json']}",
            "",
        )
    operator_plan_path.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert "operator_plan renderer release_gate_json_template is missing" in payload["errors"]
    assert (
        "operator_plan renderer safety must require release gate match for shell script"
        in payload["errors"]
    )
    assert any(
        "operator_plan renderer review command missing --release-gate-json" in error
        for error in payload["errors"]
    )
    assert any(
        "operator_plan renderer require_ready command missing --release-gate-json"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_operator_renderer_missing_command_policy_safety(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    operator_plan_path = bundle / "operator_plan.json"
    payload = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    renderer = payload["operator_execution_plan_renderer"]
    renderer["safety"].pop("requires_command_policy_validation_for_shell_script")
    operator_plan_path.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "operator_plan renderer safety must require command policy validation for shell script"
        in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_operator_renderer_missing_command_approval_path_safety(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    operator_plan_path = bundle / "operator_plan.json"
    payload = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    renderer = payload["operator_execution_plan_renderer"]
    renderer["safety"].pop(
        "requires_command_approval_paths_match_for_shell_script",
        None,
    )
    operator_plan_path.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "operator_plan renderer safety must require command approval path match for shell script"
        in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_operator_renderer_missing_weave_gate_safety(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    operator_plan_path = bundle / "operator_plan.json"
    payload = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    renderer = payload["operator_execution_plan_renderer"]
    renderer["safety"].pop(
        "requires_weave_content_canary_gate_validation_for_shell_script",
        None,
    )
    operator_plan_path.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "operator_plan renderer safety must require Weave content canary gate validation for shell script"
        in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_operator_renderer_missing_canary_approval_scope_safety(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    operator_plan_path = bundle / "operator_plan.json"
    payload = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    renderer = payload["operator_execution_plan_renderer"]
    renderer["safety"].pop(
        "requires_canary_approval_scope_match_for_shell_script",
        None,
    )
    operator_plan_path.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "operator_plan renderer safety must require canary approval scope match for shell script"
        in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_operator_renderer_script_missing_weave_gate_contract(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path")
        == "scripts/tools/render_taiwan_operator_execution_plan.py"
    )
    renderer_script = bundle / record["bundle_path"]
    renderer_script.write_text(
        renderer_script.read_text(encoding="utf-8").replace(
            "def validate_weave_content_canary_gate_option",
            "def removed_weave_content_canary_gate_option",
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "operator execution plan renderer script missing source contract "
        "Weave content canary gate validator function: "
        "def validate_weave_content_canary_gate_option"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_operator_renderer_script_missing_command_approval_contract(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path")
        == "scripts/tools/render_taiwan_operator_execution_plan.py"
    )
    renderer_script = bundle / record["bundle_path"]
    renderer_script.write_text(
        renderer_script.read_text(encoding="utf-8").replace(
            "def validate_external_action_source_packet_option(",
            "def removed_external_action_source_packet_option(",
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "operator execution plan renderer script missing source contract "
        "command approval source packet path validator: "
        "def validate_external_action_source_packet_option("
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_operator_renderer_missing_weave_contract_helper(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path")
        == "scripts/tools/render_taiwan_operator_execution_plan.py"
    )
    renderer_script = bundle / record["bundle_path"]
    renderer_script.write_text(
        renderer_script.read_text(encoding="utf-8").replace(
            "from weave_content_canary_gate_contract import",
            "from removed_weave_content_canary_gate_contract import",
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "operator execution plan renderer script missing source contract "
        "native Weave content canary contract helper import: "
        "from weave_content_canary_gate_contract import"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_operator_renderer_helper_missing_dependency_role(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path")
        == "scripts/tools/weave_content_canary_gate_contract.py"
    )
    record["roles"] = [
        role
        for role in record.get("roles", [])
        if role != "operator_execution_plan_renderer:dependency_script"
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "bundled evidence for operator execution plan renderer dependency script "
        "missing role operator_execution_plan_renderer:dependency_script: "
        "scripts/tools/weave_content_canary_gate_contract.py"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_external_action_approval_packet_hash_mismatch(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    packet_path = bundle / "external_action_approval_packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["external_action_checklist_sha256"] = "0" * 64
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "external_action_approval_packet.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert "external_action_approval_packet checklist sha256 mismatch" in payload["errors"]


def test_verify_release_evidence_bundle_rejects_external_action_approval_verifier_command_mismatch(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    packet_path = bundle / "external_action_approval_packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["approval_verifier"]["command_template"] = packet["approval_verifier"][
        "command_template"
    ].replace(" --require-approved", "")
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["external_action_approval_packet"]["approval_verifier"] = packet[
        "approval_verifier"
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "external_action_approval_packet.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "external_action_approval_packet approval_verifier command missing --require-approved"
        in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_external_action_approval_verifier_missing_source_packet(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    packet_path = bundle / "external_action_approval_packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["approval_verifier"].pop("source_packet_json")
    packet["approval_verifier"]["command_template"] = packet["approval_verifier"][
        "command_template"
    ].replace(" --source-packet-json external_action_approval_packet.json", "")
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["external_action_approval_packet"]["approval_verifier"] = packet[
        "approval_verifier"
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "external_action_approval_packet.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "external_action_approval_packet approval_verifier source_packet_json is missing"
        in payload["errors"]
    )
    assert (
        "external_action_approval_packet approval_verifier command missing --source-packet-json"
        in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_external_action_approval_renderer_command_mismatch(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    packet_path = bundle / "external_action_approval_packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["approval_template_renderer"]["command_template"] = packet[
        "approval_template_renderer"
    ]["command_template"].replace(" --output-json", " --missing-output-json")
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["external_action_approval_packet"]["approval_template_renderer"] = packet[
        "approval_template_renderer"
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "external_action_approval_packet.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "external_action_approval_packet approval_template_renderer command missing --output-json"
        in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_current_gate_external_action_checklist_mismatch(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["current_gate"]["external_action_checklist"]["item_count"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "manifest current_gate external_action_checklist does not match operator_next_steps"
        in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_operator_wandb_contract_mismatch(tmp_path):
    bundle = build_bundle(tmp_path, readiness_ok=True)
    operator_plan_path = bundle / "operator_plan.json"
    payload = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    payload["wandb_completion_contract"]["status"] = "stale"
    operator_plan_path.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "operator_plan wandb_completion_contract does not match manifest current_gate"
        in payload["errors"]
    )


def mark_wandb_contract_row_sync_ready(
    bundle: Path,
    *,
    scope_confirmation_required: bool = True,
    refresh_command: str | None = None,
) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = manifest["current_gate"]["wandb_completion_contract"]["benchmarks"][0]
    if refresh_command is None:
        refresh_command = (
            "uv run python scripts/tools/verify_taiwan_wandb_completion.py "
            "--entity test-entity --project test-project --run-id run-1 "
            "--benchmark agentic_math --expected-total 100 "
            "--require-nemoclaw-session-audit "
            "--json outputs/taiwan_full_eval/wandb_completion/agentic_math-run-1.json"
        )
    render_command = (
        "uv run python scripts/tools/render_wandb_scope_attestation.py "
        "--template-json temp/agentic_math-run-1.scope_attestation.json "
        "--output-json temp/agentic_math-run-1.scope_attestation.json "
        "--confirmed-by REVIEWER_NAME "
        "--confirmed-at YYYY-MM-DDTHH:MM:SS+09:00 "
        "--confirmation CONFIRM_THIS_WANDB_RUN_IS_THE_REVIEWED_SCOPE "
        "--actual-cost-estimate ACTUAL_COST_USD "
        "--provider-bill-reference PROVIDER_BILL_REFERENCE "
        "--report-json temp/agentic_math-run-1.scope_attestation.render.json "
        "--markdown temp/agentic_math-run-1.scope_attestation.render.md "
        "--preflight-report-json temp/agentic_math-run-1.scope_preflight.json "
        "--sync-dry-run-report-json temp/agentic_math-run-1.sync_dry_run.json"
    )
    preflight_command = (
        "uv run python scripts/tools/verify_wandb_scope_attestation.py "
        "--review-json outputs/taiwan_full_eval/canary_agentic_paid_run_review.json "
        "--completion-json outputs/taiwan_full_eval/wandb_completion/agentic_math-run-1.json "
        "--scope-attestation-json temp/agentic_math-run-1.scope_attestation.json "
        "--json temp/agentic_math-run-1.scope_preflight.json"
    )
    dry_run_sync_command = (
        "uv run python scripts/tools/sync_wandb_completion_to_paid_review.py "
        "--review-json outputs/taiwan_full_eval/canary_agentic_paid_run_review.json "
        "--completion-json outputs/taiwan_full_eval/wandb_completion/agentic_math-run-1.json "
        "--set-verify-wandb-completion --top-level --adopt-existing-result "
        "--scope-attestation-json temp/agentic_math-run-1.scope_attestation.json "
        "--report-json temp/agentic_math-run-1.sync_dry_run.json"
    )
    apply_sync_command = (
        "uv run python scripts/tools/sync_wandb_completion_to_paid_review.py "
        "--review-json outputs/taiwan_full_eval/canary_agentic_paid_run_review.json "
        "--completion-json outputs/taiwan_full_eval/wandb_completion/agentic_math-run-1.json "
        "--in-place --set-verify-wandb-completion --top-level --adopt-existing-result "
        "--scope-attestation-json temp/agentic_math-run-1.scope_attestation.json "
        "--validated-dry-run-report-json temp/agentic_math-run-1.sync_dry_run.json"
    )
    row.update(
        {
            "sync_ready_adoption_candidate_count": 1,
            "scope_attestation_template_paths": [
                "temp/agentic_math-run-1.scope_attestation.json"
            ],
            "scope_confirmation_required": scope_confirmation_required,
            "scope_warning": (
                "Use the sync command only when the W&B run belongs to the "
                "reviewed paid run or agreed canary scope."
            ),
            "refresh_wandb_completion_commands": [refresh_command]
            if refresh_command
            else [],
            "next_actions": [
                "refresh W&B completion verifier before paid-review adoption",
                "link the passing W&B completion verifier into the paid-run review",
                "rerun the release gate after completion and review evidence are updated",
            ],
            "recommended_commands": [
                command
                for command in [
                    refresh_command,
                    render_command,
                    preflight_command,
                    dry_run_sync_command,
                    apply_sync_command,
                    "uv run python scripts/tools/run_taiwan_release_gate.py --quiet",
                ]
                if command
            ],
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    operator_plan_path = bundle / "operator_plan.json"
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    operator_plan["wandb_completion_contract"] = manifest["current_gate"][
        "wandb_completion_contract"
    ]
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")


def wandb_contract_sync_ready_manifest() -> dict:
    completion_path = "outputs/taiwan_full_eval/wandb_completion/agentic_math-run-1.json"
    template_path = "temp/agentic_math-run-1.scope_attestation.json"
    render_report_path = "temp/agentic_math-run-1.scope_attestation.render.json"
    preflight_path = "temp/agentic_math-run-1.scope_preflight.json"
    sync_dry_run_path = "temp/agentic_math-run-1.sync_dry_run.json"
    target_review_path = "outputs/taiwan_full_eval/canary_agentic_paid_run_review.json"
    refresh_command = (
        "uv run python scripts/tools/verify_taiwan_wandb_completion.py "
        "--entity test-entity --project test-project --run-id run-1 "
        "--benchmark agentic_math --expected-total 100 "
        "--require-nemoclaw-session-audit "
        f"--json {completion_path}"
    )
    render_command = (
        "uv run python scripts/tools/render_wandb_scope_attestation.py "
        f"--template-json {template_path} "
        f"--output-json {template_path} "
        "--confirmed-by REVIEWER_NAME "
        "--confirmed-at YYYY-MM-DDTHH:MM:SS+09:00 "
        "--confirmation CONFIRM_THIS_WANDB_RUN_IS_THE_REVIEWED_SCOPE "
        "--actual-cost-estimate ACTUAL_COST_USD "
        "--provider-bill-reference PROVIDER_BILL_REFERENCE "
        f"--report-json {render_report_path} "
        " --markdown temp/agentic_math-run-1.scope_attestation.render.md "
        f"--preflight-report-json {preflight_path} "
        f"--sync-dry-run-report-json {sync_dry_run_path}"
    )
    preflight_command = (
        "uv run python scripts/tools/verify_wandb_scope_attestation.py "
        f"--review-json {target_review_path} "
        f"--completion-json {completion_path} "
        f"--scope-attestation-json {template_path} "
        f"--json {preflight_path}"
    )
    dry_run_sync_command = (
        "uv run python scripts/tools/sync_wandb_completion_to_paid_review.py "
        f"--review-json {target_review_path} "
        f"--completion-json {completion_path} "
        "--set-verify-wandb-completion --top-level --adopt-existing-result "
        f"--scope-attestation-json {template_path} "
        f"--report-json {sync_dry_run_path}"
    )
    apply_sync_command = (
        "uv run python scripts/tools/sync_wandb_completion_to_paid_review.py "
        f"--review-json {target_review_path} "
        f"--completion-json {completion_path} "
        "--in-place --set-verify-wandb-completion --top-level "
        "--adopt-existing-result "
        f"--scope-attestation-json {template_path} "
        f"--validated-dry-run-report-json {sync_dry_run_path}"
    )
    return {
        "current_gate": {
            "benchmark_completion": [
                {
                    "benchmark": "agentic_math",
                    "required": True,
                    "completion_proven": False,
                    "standalone_ok": False,
                    "standalone_status": "stale",
                    "standalone_records": [
                        {
                            "path": completion_path,
                            "ok": True,
                            "run_id": "run-1",
                        }
                    ],
                    "review_ok": False,
                    "review_status": "missing_review_entry",
                    "review_entries": [],
                }
            ],
            "existing_results_formalization": {
                "formalized_records": [
                    {
                        "benchmark": "agentic_math",
                        "wandb_completion": {
                            "path": completion_path,
                            "run_id": "run-1",
                        },
                    }
                ]
            },
            "wandb_adoption_draft": {
                "skipped": False,
                "candidates": [
                    {
                        "benchmark": "agentic_math",
                        "wandb_entity": "test-entity",
                        "wandb_project": "test-project",
                        "wandb_run_id": "run-1",
                        "wandb_completion_json": completion_path,
                        "scope_attestation_template_json": template_path,
                        "scope_attestation_render_report_json": render_report_path,
                        "scope_attestation_preflight_report_json": preflight_path,
                        "sync_dry_run_report_json": sync_dry_run_path,
                        "sync_ready": True,
                    }
                ],
            },
            "wandb_completion_contract": {
                "status": "incomplete",
                "complete": False,
                "required_benchmarks": ["agentic_math"],
                "required_count": 1,
                "release_completion_proven_count": 0,
                "standalone_completion_ok_count": 0,
                "formalized_existing_result_count": 1,
                "missing_release_completion_benchmarks": ["agentic_math"],
                "max_age_seconds": None,
                "next_action_count": 3,
                "benchmarks": [
                    {
                        "benchmark": "agentic_math",
                        "required": True,
                        "release_completion_proven": False,
                        "standalone_completion_ok": False,
                        "standalone_status": "stale",
                        "standalone_run_ids": ["run-1"],
                        "standalone_completion_paths": [completion_path],
                        "review_completion_ok": False,
                        "review_status": "missing_review_entry",
                        "review_run_ids": [],
                        "review_completion_paths": [],
                        "formalized_existing_result": True,
                        "formalized_existing_run_ids": ["run-1"],
                        "formalized_existing_completion_paths": [completion_path],
                        "formalized_existing_count": 1,
                        "scope_attestation_template_paths": [template_path],
                        "scope_attestation_render_report_paths": [render_report_path],
                        "scope_attestation_preflight_report_paths": [preflight_path],
                        "sync_dry_run_report_paths": [sync_dry_run_path],
                        "sync_ready_adoption_candidate_count": 1,
                        "adoption_sync_blocked_reasons": [],
                        "refresh_wandb_completion_commands": [refresh_command],
                        "status": "formalized_but_not_reviewed",
                        "next_actions": [
                            "refresh W&B completion verifier before paid-review adoption",
                            "link the passing W&B completion verifier into the paid-run review",
                            "rerun the release gate after completion and review evidence are updated",
                        ],
                        "recommended_commands": [
                            refresh_command,
                            render_command,
                            preflight_command,
                            dry_run_sync_command,
                            apply_sync_command,
                            "uv run python scripts/tools/run_taiwan_release_gate.py --quiet",
                        ],
                        "scope_confirmation_required": True,
                        "scope_warning": (
                            "Use the sync command only when the W&B run belongs "
                            "to the reviewed paid run or agreed canary scope."
                        ),
                    }
                ],
            },
        }
    }


def test_verify_release_evidence_bundle_accepts_wandb_contract_draft_sync_ready_match():
    module = load_verify_module()
    errors = module.validate_wandb_completion_contract_consistency(
        wandb_contract_sync_ready_manifest()
    )

    assert errors == []


def test_verify_release_evidence_bundle_rejects_wandb_contract_draft_sync_ready_count_mismatch():
    module = load_verify_module()
    manifest = wandb_contract_sync_ready_manifest()
    row = manifest["current_gate"]["wandb_completion_contract"]["benchmarks"][0]
    row["sync_ready_adoption_candidate_count"] = 0

    errors = module.validate_wandb_completion_contract_consistency(manifest)

    assert (
        "wandb_completion_contract benchmark agentic_math "
        "sync_ready_adoption_candidate_count does not match W&B adoption draft"
    ) in errors


def test_verify_release_evidence_bundle_rejects_wandb_contract_draft_path_mismatch():
    module = load_verify_module()
    manifest = wandb_contract_sync_ready_manifest()
    row = manifest["current_gate"]["wandb_completion_contract"]["benchmarks"][0]
    row["scope_attestation_template_paths"] = ["temp/other.scope_attestation.json"]

    errors = module.validate_wandb_completion_contract_consistency(manifest)

    assert (
        "wandb_completion_contract benchmark agentic_math "
        "scope_attestation_template_paths does not match W&B adoption draft"
    ) in errors


def test_verify_release_evidence_bundle_rejects_wandb_contract_draft_refresh_mismatch():
    module = load_verify_module()
    manifest = wandb_contract_sync_ready_manifest()
    row = manifest["current_gate"]["wandb_completion_contract"]["benchmarks"][0]
    row["refresh_wandb_completion_commands"] = [
        (
            "uv run python scripts/tools/verify_taiwan_wandb_completion.py "
            "--entity test-entity --project test-project --run-id other-run "
            "--benchmark agentic_math --expected-total 100 "
            "--require-nemoclaw-session-audit "
            "--json outputs/taiwan_full_eval/wandb_completion/agentic_math-run-1.json"
        )
    ]
    row["recommended_commands"][0] = row["refresh_wandb_completion_commands"][0]

    errors = module.validate_wandb_completion_contract_consistency(manifest)

    assert (
        "wandb_completion_contract benchmark agentic_math stale sync-ready "
        "adoption refresh commands do not match W&B adoption draft candidate"
    ) in errors


def test_verify_release_evidence_bundle_rejects_wandb_contract_agentic_refresh_without_audit_flag():
    module = load_verify_module()
    manifest = wandb_contract_sync_ready_manifest()
    row = manifest["current_gate"]["wandb_completion_contract"]["benchmarks"][0]
    command = row["refresh_wandb_completion_commands"][0].replace(
        " --require-nemoclaw-session-audit",
        "",
    )
    row["refresh_wandb_completion_commands"] = [command]
    row["recommended_commands"][0] = command

    errors = module.validate_wandb_completion_contract_consistency(manifest)

    assert (
        "wandb_completion_contract benchmark agentic_math "
        "verify command missing --require-nemoclaw-session-audit"
    ) in errors
    assert (
        "wandb_completion_contract benchmark agentic_math stale sync-ready "
        "adoption refresh command 1 missing --require-nemoclaw-session-audit"
    ) in errors


def test_verify_release_evidence_bundle_rejects_wandb_contract_sync_ready_without_scope_confirmation(
    tmp_path,
):
    bundle, _attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    mark_wandb_contract_row_sync_ready(
        bundle,
        scope_confirmation_required=False,
    )

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "wandb_completion_contract benchmark agentic_math sync-ready adoption "
        "requires scope_confirmation_required=true" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_contract_stale_sync_ready_without_refresh_command(
    tmp_path,
):
    bundle, _attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    mark_wandb_contract_row_sync_ready(bundle, refresh_command="")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "wandb_completion_contract benchmark agentic_math stale sync-ready "
        "adoption requires refresh_wandb_completion_commands" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_contract_placeholder_refresh_command(
    tmp_path,
):
    bundle, _attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    mark_wandb_contract_row_sync_ready(
        bundle,
        refresh_command=(
            "uv run python scripts/tools/verify_taiwan_wandb_completion.py "
            "--entity test-entity --project test-project --run-id RUN_ID "
            "--benchmark agentic_math --json outputs/taiwan_full_eval/wandb_completion/"
            "agentic_math-RUN_ID.json"
        ),
    )

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "wandb_completion_contract benchmark agentic_math stale sync-ready adoption "
        "refresh command 1 is not a concrete verify_taiwan_wandb_completion.py command"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_contract_count_mismatch(tmp_path):
    bundle, _attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["current_gate"]["wandb_completion_contract"]["release_completion_proven_count"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "wandb_completion_contract release_completion_proven_count does not match benchmarks"
        in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_contract_missing_list_mismatch(tmp_path):
    bundle, _attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["current_gate"]["wandb_completion_contract"][
        "missing_release_completion_benchmarks"
    ] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "wandb_completion_contract missing_release_completion_benchmarks does not match benchmarks"
        in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_contract_row_status_mismatch(tmp_path):
    bundle, _attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["current_gate"]["wandb_completion_contract"]["benchmarks"][0][
        "status"
    ] = "release_complete"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "wandb_completion_contract benchmark agentic_math status does not match evidence flags"
        in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_contract_fake_benchmark_row(tmp_path):
    bundle = build_bundle(tmp_path, readiness_ok=True)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["current_gate"]["wandb_completion_contract"]["benchmarks"].append(
        {
            "benchmark": "unbacked_benchmark",
            "required": False,
            "release_completion_proven": False,
            "standalone_completion_ok": False,
            "standalone_status": None,
            "standalone_run_ids": [],
            "standalone_completion_paths": [],
            "review_completion_ok": False,
            "review_status": None,
            "review_run_ids": [],
            "review_completion_paths": [],
            "formalized_existing_result": False,
            "formalized_existing_run_ids": [],
            "formalized_existing_completion_paths": [],
            "formalized_existing_count": 0,
            "status": "not_required",
            "next_actions": [],
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "wandb_completion_contract benchmarks do not match source benchmark set"
        in payload["errors"]
    )


def test_verify_release_evidence_bundle_accepts_wandb_contract_non_required_source_row(tmp_path):
    bundle = build_bundle(tmp_path, readiness_ok=True)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["current_gate"]["benchmark_completion"] = [
        {
            "benchmark": "optional_benchmark",
            "required": False,
            "expected_run_id": None,
            "completion_proven": False,
            "standalone_status": "missing",
            "standalone_ok": False,
            "standalone_records": [],
            "review_status": "missing_review_entry",
            "review_ok": False,
            "review_entries": [],
        }
    ]
    manifest["current_gate"]["wandb_completion_contract"] = {
        "status": "no_required_benchmarks",
        "complete": False,
        "required_benchmarks": [],
        "required_count": 0,
        "release_completion_proven_count": 0,
        "standalone_completion_ok_count": 0,
        "formalized_existing_result_count": 0,
        "missing_release_completion_benchmarks": [],
        "max_age_seconds": None,
        "next_action_count": 0,
        "benchmarks": [
            {
                "benchmark": "optional_benchmark",
                "required": False,
                "expected_run_id": None,
                "release_completion_proven": False,
                "standalone_completion_ok": False,
                "standalone_status": "missing",
                "standalone_run_ids": [],
                "standalone_completion_paths": [],
                "review_completion_ok": False,
                "review_status": "missing_review_entry",
                "review_run_ids": [],
                "review_completion_paths": [],
                "formalized_existing_result": False,
                "formalized_existing_run_ids": [],
                "formalized_existing_completion_paths": [],
                "formalized_existing_count": 0,
                "status": "not_required",
                "next_actions": [],
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    operator_plan_path = bundle / "operator_plan.json"
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    operator_plan["wandb_completion_contract"] = manifest["current_gate"][
        "wandb_completion_contract"
    ]
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")
    summary_path = bundle / "summary.md"
    summary_text = summary_path.read_text(encoding="utf-8")
    contract_section, rest = summary_text.split("## W&B Completion Contract", 1)
    rest = rest.replace(
        "| none |  |  |  |  |  |  |  |  |  |  |  |  |  |",
        "| optional_benchmark | False | not_required | False | missing | missing_review_entry | False |  |  |  |  |  |  |  |",
        1,
    )
    summary_path.write_text(
        f"{contract_section}## W&B Completion Contract{rest}",
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, "summary.md")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is True


def test_verify_release_evidence_bundle_rejects_wandb_contract_required_source_mismatch(tmp_path):
    bundle = build_bundle(tmp_path, readiness_ok=True)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["current_gate"]["benchmark_completion"] = [
        {
            "benchmark": "agentic_math",
            "required": False,
            "expected_run_id": None,
            "completion_proven": False,
            "standalone_status": "missing",
            "standalone_ok": False,
            "standalone_records": [],
            "review_status": "missing_review_entry",
            "review_ok": False,
            "review_entries": [],
        }
    ]
    manifest["current_gate"]["wandb_completion_contract"] = {
        "status": "incomplete",
        "complete": False,
        "required_benchmarks": ["agentic_math"],
        "required_count": 1,
        "release_completion_proven_count": 0,
        "standalone_completion_ok_count": 0,
        "formalized_existing_result_count": 0,
        "missing_release_completion_benchmarks": ["agentic_math"],
        "max_age_seconds": None,
        "next_action_count": 0,
        "benchmarks": [
            {
                "benchmark": "agentic_math",
                "required": True,
                "expected_run_id": None,
                "release_completion_proven": False,
                "standalone_completion_ok": False,
                "standalone_status": "missing",
                "standalone_run_ids": [],
                "standalone_completion_paths": [],
                "review_completion_ok": False,
                "review_status": "missing_review_entry",
                "review_run_ids": [],
                "review_completion_paths": [],
                "formalized_existing_result": False,
                "formalized_existing_run_ids": [],
                "formalized_existing_completion_paths": [],
                "formalized_existing_count": 0,
                "status": "missing_release_completion",
                "next_actions": [],
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "wandb_completion_contract benchmark agentic_math required does not match source evidence"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_contract_review_source_mismatch(tmp_path):
    bundle, _attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = manifest["current_gate"]["wandb_completion_contract"]["benchmarks"][0]
    row["review_completion_ok"] = True
    row["review_status"] = "passed"
    row["review_run_ids"] = ["run-1"]
    row["review_completion_paths"] = ["wandb_completion.json"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "wandb_completion_contract benchmark agentic_math review_completion_ok does not match source evidence"
        in error
        for error in payload["errors"]
    )
    assert any(
        "wandb_completion_contract benchmark agentic_math review_run_ids does not match source evidence"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_contract_formalized_source_mismatch(tmp_path):
    bundle, _attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = manifest["current_gate"]["wandb_completion_contract"]["benchmarks"][0]
    row["formalized_existing_result"] = True
    row["formalized_existing_run_ids"] = ["run-1"]
    row["formalized_existing_completion_paths"] = ["wandb_completion.json"]
    row["formalized_existing_count"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "wandb_completion_contract benchmark agentic_math formalized_existing_result does not match source evidence"
        in error
        for error in payload["errors"]
    )
    assert any(
        "wandb_completion_contract benchmark agentic_math formalized_existing_count does not match source evidence"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_operator_wandb_adoption_draft_mismatch(tmp_path):
    bundle = build_bundle(tmp_path, readiness_ok=True)
    operator_plan_path = bundle / "operator_plan.json"
    payload = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    payload["wandb_adoption_draft"]["status"] = "stale"
    operator_plan_path.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "operator_plan wandb_adoption_draft does not match manifest current_gate"
        in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_operator_nemoclaw_summary_mismatch(tmp_path):
    bundle = build_bundle(tmp_path, readiness_ok=True)
    operator_plan_path = bundle / "operator_plan.json"
    payload = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    payload["nemoclaw_adoption"]["status"] = "stale"
    operator_plan_path.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "operator_plan nemoclaw_adoption does not match manifest current_gate"
        in payload["errors"]
    )


def test_verify_release_evidence_bundle_requires_wandb_adoption_source_audit(tmp_path):
    bundle, source_audit, _completion = build_bundle_with_wandb_adoption_draft(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [
        record
        for record in manifest["files"]
        if record.get("source_path") != str(source_audit)
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "missing bundled evidence for W&B adoption draft source_audit_json" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_adoption_source_audit_sha_mismatch(tmp_path):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    draft_record = next(
        record for record in manifest["files"] if "wandb_adoption_draft" in record.get("roles", [])
    )
    bundled_draft = bundle / draft_record["bundle_path"]
    payload = json.loads(bundled_draft.read_text(encoding="utf-8"))
    payload["source_audit_sha256"] = "0" * 64
    bundled_draft.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, draft_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "W&B adoption draft source_audit_sha256 does not match bundled source_audit_json"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_adoption_summary_source_audit_mismatch(tmp_path):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["current_gate"]["wandb_adoption_draft"]["source_audit_sha256"] = "0" * 64
    manifest["current_gate"]["wandb_adoption_draft"]["candidates"][0][
        "source_audit_sha256"
    ] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "manifest current_gate wandb_adoption_draft source_audit_sha256 "
        "does not match bundled draft JSON" in error
        for error in payload["errors"]
    )
    assert any(
        "manifest current_gate wandb_adoption_draft candidate agentic_math "
        "source_audit_sha256 does not match bundled draft JSON" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_adoption_required_fields_mismatch(tmp_path):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["current_gate"]["wandb_adoption_draft"]["candidates"][0][
        "required_human_fields"
    ] = ["tampered_field"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "manifest current_gate wandb_adoption_draft candidate agentic_math "
        "required_human_fields does not match bundled draft JSON" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_adoption_top_level_required_fields_mismatch(tmp_path):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["current_gate"]["wandb_adoption_draft"]["required_human_fields"] = [
        "scope_attestation_json.confirmed",
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "manifest current_gate wandb_adoption_draft "
        "required_human_fields does not match bundled draft JSON" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_adoption_source_required_fields_mismatch(tmp_path):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    draft_record = next(
        record
        for record in manifest["files"]
        if record.get("source_path", "").endswith("wandb_adoption_draft.json")
    )
    draft_path = bundle / draft_record["bundle_path"]
    draft_payload = json.loads(draft_path.read_text(encoding="utf-8"))
    draft_payload["required_human_fields"] = ["scope_attestation_json.confirmed"]
    draft_path.write_text(json.dumps(draft_payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, draft_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "W&B adoption draft JSON required_human_fields does not match "
        "candidate required_human_fields" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_adoption_pending_human_field_summary_mismatch(tmp_path):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    draft = manifest["current_gate"]["wandb_adoption_draft"]
    draft["pending_human_field_count"] = 0
    draft["pending_human_fields"] = ["tampered_field"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "manifest current_gate wandb_adoption_draft "
        "pending_human_field_count does not match bundled draft JSON" in error
        for error in payload["errors"]
    )
    assert any(
        "manifest current_gate wandb_adoption_draft "
        "pending_human_fields does not match bundled draft JSON" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_adoption_operator_handoff_mismatch(tmp_path):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_unconfirmed_checks=True,
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    draft_record = next(
        record
        for record in manifest["files"]
        if "wandb_adoption_draft" in record.get("roles", [])
    )
    bundled_draft = bundle / draft_record["bundle_path"]
    payload = json.loads(bundled_draft.read_text(encoding="utf-8"))
    payload["candidates"][0]["operator_handoff"]["steps"][0][
        "expected_evidence_paths"
    ] = ["tampered.json"]
    bundled_draft.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, draft_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "W&B adoption draft candidate agentic_math operator_handoff does not "
        "match candidate" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_adoption_draft_markdown_handoff_mismatch(tmp_path):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_unconfirmed_checks=True,
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    markdown_record = next(
        record
        for record in manifest["files"]
        if "wandb_adoption_draft_markdown" in record.get("roles", [])
    )
    bundled_markdown = bundle / markdown_record["bundle_path"]
    markdown_text = bundled_markdown.read_text(encoding="utf-8")
    assert "confirm_scope_attestation" in markdown_text
    bundled_markdown.write_text(
        markdown_text.replace(
            "confirm_scope_attestation",
            "confirm_scope_attestation_removed",
            1,
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, markdown_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "W&B adoption draft Markdown missing operator handoff content "
        "from bundled draft JSON" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_requires_wandb_adoption_draft_markdown_path(tmp_path):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_unconfirmed_checks=True,
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["current_gate"]["wandb_adoption_draft"].pop("markdown_path")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "manifest current_gate wandb_adoption_draft markdown_path is required "
        "when adoption candidates are present" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_adoption_draft_markdown_path_mismatch(tmp_path):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_unconfirmed_checks=True,
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    draft_record = next(
        record
        for record in manifest["files"]
        if "wandb_adoption_draft" in record.get("roles", [])
    )
    bundled_draft = bundle / draft_record["bundle_path"]
    payload = json.loads(bundled_draft.read_text(encoding="utf-8"))
    payload["markdown_path"] = str(bundle / "other_wandb_adoption_draft.md")
    bundled_draft.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, draft_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "manifest current_gate wandb_adoption_draft markdown_path does not "
        "match bundled draft JSON" in error
        for error in payload["errors"]
    )


def mutate_bundled_wandb_adoption_source_audit(
    bundle: Path,
    source_audit: Path,
    mutator,
) -> None:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))

    source_record = next(
        record
        for record in manifest["files"]
        if record.get("source_path") == str(source_audit)
    )
    bundled_source = bundle / source_record["bundle_path"]
    source_payload = json.loads(bundled_source.read_text(encoding="utf-8"))
    mutator(source_payload)
    bundled_source.write_text(json.dumps(source_payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, source_record["bundle_path"])
    new_source_sha = sha256(bundled_source)

    draft_record = next(
        record
        for record in manifest["files"]
        if "wandb_adoption_draft" in record.get("roles", [])
    )
    bundled_draft = bundle / draft_record["bundle_path"]
    draft_payload = json.loads(bundled_draft.read_text(encoding="utf-8"))
    draft_payload["source_audit_sha256"] = new_source_sha
    for candidate in draft_payload.get("candidates", []):
        if isinstance(candidate, dict):
            candidate["source_audit_sha256"] = new_source_sha
    bundled_draft.write_text(json.dumps(draft_payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, draft_record["bundle_path"])

    template_role = "wandb_adoption_draft:candidate:agentic_math:scope_attestation_template_json"
    template_record = next(
        (
            record
            for record in manifest["files"]
            if template_role in record.get("roles", [])
        ),
        None,
    )
    if isinstance(template_record, dict):
        bundled_template = bundle / template_record["bundle_path"]
        template_payload = json.loads(bundled_template.read_text(encoding="utf-8"))
        template_payload["source_audit_sha256"] = new_source_sha
        bundled_template.write_text(json.dumps(template_payload), encoding="utf-8")
        refresh_manifest_record_hash(bundle, template_record["bundle_path"])


def test_verify_release_evidence_bundle_rejects_wandb_adoption_source_audit_missing_candidate(tmp_path):
    bundle, source_audit, _completion = build_bundle_with_wandb_adoption_draft(tmp_path)

    def change_formalized_run_id(source_payload):
        source_payload["formalized_records"][0]["wandb_completion"]["run_id"] = "other-run"

    mutate_bundled_wandb_adoption_source_audit(
        bundle,
        source_audit,
        change_formalized_run_id,
    )

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "W&B adoption draft candidate agentic_math source_audit_json "
        "formalized_records does not include candidate W&B completion" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_adoption_source_audit_missing_completion_records(tmp_path):
    bundle, source_audit, _completion = build_bundle_with_wandb_adoption_draft(tmp_path)
    mutate_bundled_wandb_adoption_source_audit(
        bundle,
        source_audit,
        lambda source_payload: source_payload.pop("wandb_completion_records", None),
    )

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "W&B adoption draft source_audit_json wandb_completion_records "
        "must be a list" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_adoption_source_audit_schema_issues(tmp_path):
    bundle, source_audit, _completion = build_bundle_with_wandb_adoption_draft(tmp_path)

    def add_schema_issue(source_payload):
        source_payload["wandb_completion_records"][0]["schema_current"] = False
        source_payload["wandb_completion_records"][0]["schema_current_issues"] = [
            "status must be passed"
        ]

    mutate_bundled_wandb_adoption_source_audit(
        bundle,
        source_audit,
        add_schema_issue,
    )

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "W&B adoption draft candidate agentic_math source_audit_json "
        "wandb_completion_records candidate schema_current is not true" in error
        for error in payload["errors"]
    )
    assert any(
        "W&B adoption draft candidate agentic_math source_audit_json "
        "wandb_completion_records candidate schema_current_issues must be empty" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_adoption_template_source_audit_mismatch(tmp_path):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    role = "wandb_adoption_draft:candidate:agentic_math:scope_attestation_template_json"
    template_record = next(
        record for record in manifest["files"] if role in record.get("roles", [])
    )
    bundled_template = bundle / template_record["bundle_path"]
    payload = json.loads(bundled_template.read_text(encoding="utf-8"))
    payload["source_audit_sha256"] = "0" * 64
    bundled_template.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, template_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "W&B adoption draft scope attestation template for agentic_math "
        "source_audit_sha256 does not match draft" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_requires_wandb_adoption_candidate_completion(tmp_path):
    bundle, _source_audit, completion = build_bundle_with_wandb_adoption_draft(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [
        record
        for record in manifest["files"]
        if record.get("source_path") != str(completion)
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "missing bundled evidence for W&B adoption draft candidate agentic_math completion JSON"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_requires_wandb_adoption_attestation_template(tmp_path):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    role = "wandb_adoption_draft:candidate:agentic_math:scope_attestation_template_json"
    manifest["files"] = [
        record
        for record in manifest["files"]
        if role not in record.get("roles", [])
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "missing bundled evidence for W&B adoption draft candidate agentic_math scope attestation template"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_adoption_attestation_mismatch(tmp_path):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    role = "wandb_adoption_draft:candidate:agentic_math:scope_attestation_template_json"
    template_record = next(
        record for record in manifest["files"] if role in record.get("roles", [])
    )
    bundled_template = bundle / template_record["bundle_path"]
    payload = json.loads(bundled_template.read_text(encoding="utf-8"))
    payload["run_id"] = "other-run"
    bundled_template.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, template_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "scope attestation template for agentic_math run_id does not match candidate"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_missing_wandb_adoption_draft_summary_section(
    tmp_path,
):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(tmp_path)
    summary_path = bundle / "summary.md"
    summary_text = summary_path.read_text(encoding="utf-8")
    summary_path.write_text(
        summary_text.replace(
            "## Existing W&B Adoption Draft",
            "## Existing W&B Adoption Candidates",
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, "summary.md")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert "summary.md missing required section: ## Existing W&B Adoption Draft" in payload["errors"]


def test_verify_release_evidence_bundle_rejects_wandb_adoption_draft_summary_row_mismatch(
    tmp_path,
):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(tmp_path)
    summary_path = bundle / "summary.md"
    summary_text = summary_path.read_text(encoding="utf-8")
    expected_row = next(
        line
        for line in summary_text.splitlines()
        if line.startswith("| agentic_math | gpt-5_5 | run-1 |")
    )
    summary_path.write_text(
        summary_text.replace(
            expected_row,
            expected_row.replace("| run-1 |", "| run-2 |", 1),
            1,
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, "summary.md")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "summary.md missing W&B adoption draft content "
        f"from current_gate: {expected_row}"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_accepts_wandb_adoption_scope_preflight_report(tmp_path):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_scope_preflight=True,
    )

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is True
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert any(
        "wandb_adoption_draft:candidate:agentic_math:scope_attestation_preflight_report_json"
        in record.get("roles", [])
        for record in manifest["files"]
    )


def test_verify_release_evidence_bundle_accepts_wandb_adoption_scope_render_report(tmp_path):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_scope_render=True,
    )

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is True
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert any(
        "wandb_adoption_draft:candidate:agentic_math:scope_attestation_render_report_json"
        in record.get("roles", [])
        for record in manifest["files"]
    )
    assert any(
        "wandb_adoption_draft:candidate:agentic_math:scope_attestation_render_markdown"
        in record.get("roles", [])
        for record in manifest["files"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_adoption_scope_render_command_missing_report_target(
    tmp_path,
):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_scope_render=True,
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    draft_record = next(
        record for record in manifest["files"] if "wandb_adoption_draft" in record.get("roles", [])
    )
    bundled_draft = bundle / draft_record["bundle_path"]
    payload = json.loads(bundled_draft.read_text(encoding="utf-8"))
    command = payload["candidates"][0]["scope_attestation_render_command"]
    payload["candidates"][0]["scope_attestation_render_command"] = command.split(
        " --preflight-report-json ",
        1,
    )[0]
    bundled_draft.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, draft_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "W&B adoption draft candidate agentic_math scope_attestation_render_command "
        "requires --preflight-report-json"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_wandb_adoption_scope_render_sha_mismatch(
    tmp_path,
):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_scope_render=True,
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    role = "wandb_adoption_draft:candidate:agentic_math:scope_attestation_render_report_json"
    report_record = next(
        record for record in manifest["files"] if role in record.get("roles", [])
    )
    bundled_report = bundle / report_record["bundle_path"]
    payload = json.loads(bundled_report.read_text(encoding="utf-8"))
    payload["output_sha256"] = "0" * 64
    bundled_report.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, report_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "W&B adoption draft candidate agentic_math scope render report "
        "output_sha256 does not match bundled output JSON"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_wandb_adoption_scope_render_unsafe_report(
    tmp_path,
):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_scope_render=True,
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    role = "wandb_adoption_draft:candidate:agentic_math:scope_attestation_render_report_json"
    report_record = next(
        record for record in manifest["files"] if role in record.get("roles", [])
    )
    bundled_report = bundle / report_record["bundle_path"]
    payload = json.loads(bundled_report.read_text(encoding="utf-8"))
    payload["safety"]["writes_wandb"] = True
    bundled_report.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, report_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "W&B adoption draft candidate agentic_math scope render report "
        "safety writes_wandb must be false"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_wandb_adoption_scope_render_incomplete_next_command(
    tmp_path,
):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_scope_render=True,
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    role = "wandb_adoption_draft:candidate:agentic_math:scope_attestation_render_report_json"
    report_record = next(
        record for record in manifest["files"] if role in record.get("roles", [])
    )
    bundled_report = bundle / report_record["bundle_path"]
    payload = json.loads(bundled_report.read_text(encoding="utf-8"))
    payload["next_commands"]["preflight"] = payload["next_commands"]["preflight"].split(
        " --json ",
        1,
    )[0]
    bundled_report.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, report_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "W&B adoption draft candidate agentic_math scope render report next_commands."
        "preflight requires --json"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_wandb_adoption_scope_render_markdown_mismatch(
    tmp_path,
):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_scope_render=True,
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    role = "wandb_adoption_draft:candidate:agentic_math:scope_attestation_render_markdown"
    markdown_record = next(
        record for record in manifest["files"] if role in record.get("roles", [])
    )
    bundled_markdown = bundle / markdown_record["bundle_path"]
    markdown = bundled_markdown.read_text(encoding="utf-8")
    bundled_markdown.write_text(markdown.replace("run-1", "run-2"), encoding="utf-8")
    refresh_manifest_record_hash(bundle, markdown_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "W&B adoption draft candidate agentic_math scope render Markdown "
        "missing expected snippet: | run_id | run-1 |"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_wandb_adoption_scope_preflight_missing_source_files(
    tmp_path,
):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_scope_preflight=True,
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    role = "wandb_adoption_draft:candidate:agentic_math:scope_attestation_preflight_report_json"
    report_record = next(
        record for record in manifest["files"] if role in record.get("roles", [])
    )
    bundled_report = bundle / report_record["bundle_path"]
    payload = json.loads(bundled_report.read_text(encoding="utf-8"))
    payload.pop("source_files", None)
    bundled_report.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, report_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "W&B adoption draft candidate agentic_math scope preflight report "
        "source_files must be an object"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_wandb_adoption_scope_preflight_sha_mismatch(
    tmp_path,
):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_scope_preflight=True,
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    role = "wandb_adoption_draft:candidate:agentic_math:scope_attestation_preflight_report_json"
    report_record = next(
        record for record in manifest["files"] if role in record.get("roles", [])
    )
    bundled_report = bundle / report_record["bundle_path"]
    payload = json.loads(bundled_report.read_text(encoding="utf-8"))
    payload["source_files"]["completion_json"]["sha256"] = "0" * 64
    bundled_report.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, report_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "W&B adoption draft candidate agentic_math scope preflight report "
        "completion_json source file sha256 does not match expected sha256"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_wandb_adoption_scope_preflight_source_attestation_sha_mismatch(
    tmp_path,
):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_scope_preflight=True,
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    role = "wandb_adoption_draft:candidate:agentic_math:scope_attestation_preflight_report_json"
    report_record = next(
        record for record in manifest["files"] if role in record.get("roles", [])
    )
    bundled_report = bundle / report_record["bundle_path"]
    payload = json.loads(bundled_report.read_text(encoding="utf-8"))
    payload["scope_attestation"]["source_attestation_sha256"] = "0" * 64
    bundled_report.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, report_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "W&B adoption draft candidate agentic_math scope preflight report "
        "scope_attestation source_attestation_sha256 does not match bundled source attestation JSON"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_accepts_wandb_adoption_sync_dry_run_report(tmp_path):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_sync_dry_run=True,
    )

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is True
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert any(
        "wandb_adoption_draft:candidate:agentic_math:sync_dry_run_report_json"
        in record.get("roles", [])
        for record in manifest["files"]
    )


def test_verify_release_evidence_bundle_accepts_wandb_adoption_unconfirmed_checks(tmp_path):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_unconfirmed_checks=True,
    )

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is True
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert any(
        "wandb_adoption_unconfirmed_checks:agentic_math:run-1:preflight_report_json"
        in record.get("roles", [])
        for record in manifest["files"]
    )
    assert any(
        "wandb_adoption_unconfirmed_checks:agentic_math:run-1:sync_dry_run_report_json"
        in record.get("roles", [])
        for record in manifest["files"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_adoption_unconfirmed_preflight_missing_source_files(
    tmp_path,
):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_unconfirmed_checks=True,
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    role = "wandb_adoption_unconfirmed_checks:agentic_math:run-1:preflight_report_json"
    report_record = next(
        record for record in manifest["files"] if role in record.get("roles", [])
    )
    bundled_report = bundle / report_record["bundle_path"]
    payload = json.loads(bundled_report.read_text(encoding="utf-8"))
    payload.pop("source_files", None)
    bundled_report.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, report_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "W&B adoption unconfirmed preflight report agentic_math/run-1 "
        "source_files must be an object"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_wandb_adoption_unconfirmed_preflight_sha_mismatch(
    tmp_path,
):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_unconfirmed_checks=True,
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    role = "wandb_adoption_unconfirmed_checks:agentic_math:run-1:preflight_report_json"
    report_record = next(
        record for record in manifest["files"] if role in record.get("roles", [])
    )
    bundled_report = bundle / report_record["bundle_path"]
    payload = json.loads(bundled_report.read_text(encoding="utf-8"))
    payload["source_files"]["completion_json"]["sha256"] = "0" * 64
    bundled_report.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, report_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "W&B adoption unconfirmed preflight report agentic_math/run-1 "
        "completion_json source file sha256 does not match expected sha256"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_missing_wandb_adoption_unconfirmed_summary_section(
    tmp_path,
):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_unconfirmed_checks=True,
    )
    summary_path = bundle / "summary.md"
    summary_text = summary_path.read_text(encoding="utf-8")
    summary_path.write_text(
        summary_text.replace(
            "## Existing W&B Adoption Unconfirmed-Template Checks",
            "## Existing W&B Adoption Unconfirmed Checks",
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, "summary.md")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "summary.md missing required section: "
        "## Existing W&B Adoption Unconfirmed-Template Checks"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_wandb_adoption_unconfirmed_summary_row_mismatch(
    tmp_path,
):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_unconfirmed_checks=True,
    )
    summary_path = bundle / "summary.md"
    summary_text = summary_path.read_text(encoding="utf-8")
    expected_row = next(
        line
        for line in summary_text.splitlines()
        if line.startswith("| agentic_math | run-1 |")
    )
    summary_path.write_text(
        summary_text.replace(
            expected_row,
            expected_row.replace("| run-1 |", "| run-2 |", 1),
            1,
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, "summary.md")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "summary.md missing W&B adoption unconfirmed-template content "
        f"from current_gate: {expected_row}"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_wandb_adoption_unconfirmed_sync_success(
    tmp_path,
):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_unconfirmed_checks=True,
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    role = "wandb_adoption_unconfirmed_checks:agentic_math:run-1:sync_dry_run_report_json"
    report_record = next(
        record for record in manifest["files"] if role in record.get("roles", [])
    )
    bundled_report = bundle / report_record["bundle_path"]
    payload = json.loads(bundled_report.read_text(encoding="utf-8"))
    payload["ok"] = True
    payload["status"] = "synced"
    payload["entry_count"] = 1
    payload["change_count"] = 1
    bundled_report.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, report_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "W&B adoption unconfirmed sync dry-run report agentic_math/run-1 ok must be false"
        in error
        for error in payload["errors"]
    )
    assert any(
        "W&B adoption unconfirmed sync dry-run report agentic_math/run-1 "
        "status must be validation_failed" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_accepts_weave_agents_adoption_validation_failures(
    tmp_path,
):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_weave_agents_adoption_validation_failures=True,
    )

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is True
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert any(
        "weave_agents_adoption_validation_failure:1:report_json"
        in record.get("roles", [])
        for record in manifest["files"]
    )
    assert any(
        "weave_agents_adoption_validation_failure:1:completion_json"
        in record.get("roles", [])
        for record in manifest["files"]
    )


def test_verify_release_evidence_bundle_rejects_weave_agents_adoption_failure_report_success(
    tmp_path,
):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_weave_agents_adoption_validation_failures=True,
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    role = "weave_agents_adoption_validation_failure:1:report_json"
    report_record = next(
        record for record in manifest["files"] if role in record.get("roles", [])
    )
    bundled_report = bundle / report_record["bundle_path"]
    payload = json.loads(bundled_report.read_text(encoding="utf-8"))
    payload["ok"] = True
    payload["status"] = "synced"
    payload["entry_count"] = 1
    bundled_report.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, report_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert "Weave Agents adoption validation failure #1 report ok must be false" in payload[
        "errors"
    ]
    assert (
        "Weave Agents adoption validation failure #1 report status must be validation_failed"
        in payload["errors"]
    )
    assert (
        "Weave Agents adoption validation failure #1 report entry_count must be 0"
        in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_weave_agents_adoption_completion_success(
    tmp_path,
):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_weave_agents_adoption_validation_failures=True,
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    role = "weave_agents_adoption_validation_failure:1:completion_json"
    completion_record = next(
        record for record in manifest["files"] if role in record.get("roles", [])
    )
    bundled_completion = bundle / completion_record["bundle_path"]
    payload = json.loads(bundled_completion.read_text(encoding="utf-8"))
    payload["ok"] = True
    payload["query_source"]["matching_span_count"] = 1
    payload["checks"] = [{"name": "spans_present", "ok": True}]
    bundled_completion.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, completion_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        error.endswith("ok must be false")
        and "Weave Agents adoption validation failure #1 rejected completion" in error
        for error in payload["errors"]
    )
    assert any(
        error.endswith("query_source.matching_span_count must be 0")
        and "Weave Agents adoption validation failure #1 rejected completion" in error
        for error in payload["errors"]
    )
    assert any(
        error.endswith("checks must contain a failing check")
        and "Weave Agents adoption validation failure #1 rejected completion" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_accepts_existing_results_relog_dry_run_plan_payload():
    module = load_verify_module()
    source_sha256 = {
        "summary_json": "a" * 64,
        "results_jsonl": "b" * 64,
    }
    record = {
        "benchmark": "agentic_math",
        "model_slug": "gpt-5_5",
        "result_dir": "outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw",
        "summary_path": "outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw/summary.json",
        "results_path": "outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw/results.jsonl",
        "row_count": 100,
    }
    payload = {
        "schema_version": 1,
        "ok": True,
        "will_write_wandb": False,
        "benchmark": "agentic_math",
        "entity": "test-entity",
        "project": "test-project",
        "source": {
            "results_dir": "outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw",
            "summary_json": "outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw/summary.json",
            "results_jsonl": "outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw/results.jsonl",
            "source_sha256": source_sha256,
        },
        "config": {
            "relog": {
                "source_sha256": source_sha256,
            },
        },
        "would_log": {
            "tables": {"agentic_math_output_table": 100},
            "artifact": {"aliases": ["latest", "production"]},
        },
        "external_action_approval": wandb_write_approval_requirement(),
        "post_log_verifier_command_template": (
            "uv run python scripts/tools/verify_taiwan_wandb_completion.py "
            "--run-id RUN_ID_AFTER_WANDB_LOG --benchmark agentic_math "
            "--expected-total 100 --require-nemoclaw-session-audit "
            f"--expected-run-config relog.source_sha256.summary_json={source_sha256['summary_json']} "
            f"--expected-run-config relog.source_sha256.results_jsonl={source_sha256['results_jsonl']}"
        ),
    }

    errors = module.validate_existing_results_relog_dry_run_plan_payload(
        payload,
        record=record,
        label="plan",
    )

    assert errors == []


def test_verify_release_evidence_bundle_rejects_existing_results_relog_plan_without_run_config_sha_binding():
    module = load_verify_module()
    source_sha256 = {
        "summary_json": "a" * 64,
        "results_jsonl": "b" * 64,
    }
    record = {
        "benchmark": "agentic_math",
        "model_slug": "gpt-5_5",
        "result_dir": "outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw",
        "summary_path": "outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw/summary.json",
        "results_path": "outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw/results.jsonl",
        "row_count": 100,
    }
    payload = {
        "schema_version": 1,
        "ok": True,
        "will_write_wandb": False,
        "benchmark": "agentic_math",
        "entity": "test-entity",
        "project": "test-project",
        "source": {
            "results_dir": "outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw",
            "summary_json": "outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw/summary.json",
            "results_jsonl": "outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw/results.jsonl",
            "source_sha256": source_sha256,
        },
        "config": {
            "relog": {
                "source_sha256": {
                    "summary_json": source_sha256["summary_json"],
                },
            },
        },
        "would_log": {
            "tables": {"agentic_math_output_table": 100},
            "artifact": {"aliases": ["latest", "production"]},
        },
        "post_log_verifier_command_template": (
            "uv run python scripts/tools/verify_taiwan_wandb_completion.py "
            "--run-id RUN_ID_AFTER_WANDB_LOG --benchmark agentic_math "
            "--expected-total 100 --require-nemoclaw-session-audit "
            f"--expected-run-config relog.source_sha256.summary_json={source_sha256['summary_json']}"
        ),
    }

    errors = module.validate_existing_results_relog_dry_run_plan_payload(
        payload,
        record=record,
        label="plan",
    )

    assert "plan config.relog.source_sha256 must match source.source_sha256" in errors
    assert (
        "plan verifier command missing relog source sha expected-run-config for results_jsonl"
        in errors
    )


def test_verify_release_evidence_bundle_accepts_existing_results_validation_failed_relog_plan_payload():
    module = load_verify_module()
    summary_path = (
        "outputs/taiwan_full_eval/swebench_pro/deepseek-v4-pro-thinking-max/"
        "official_eval_first_patch/summary.json"
    )
    eval_results_path = (
        "outputs/taiwan_full_eval/swebench_pro/deepseek-v4-pro-thinking-max/"
        "official_eval_first_patch/eval_results.json"
    )
    patch_path = (
        "outputs/taiwan_full_eval/swebench_pro/deepseek-v4-pro-thinking-max/"
        "openclaw/patches.json"
    )
    record = {
        "benchmark": "agentic_swe",
        "model_slug": "deepseek-v4-pro-thinking-max",
        "official_summary_path": summary_path,
        "patches_path": patch_path,
        "row_count": 1,
    }
    payload = {
        "schema_version": 1,
        "ok": False,
        "status": "validation_failed",
        "will_write_wandb": False,
        "benchmark": "agentic_swe",
        "entity": "test-entity",
        "project": "test-project",
        "source": {
            "official_eval_dir": (
                "outputs/taiwan_full_eval/swebench_pro/deepseek-v4-pro-thinking-max/"
                "official_eval_first_patch"
            ),
                "summary_json": summary_path,
                "eval_results_json": eval_results_path,
                "patch_path": patch_path,
                "source_sha256": {
                    "summary_json": module.sha256_file(module.repo_path(summary_path)),
                    "eval_results_json": module.sha256_file(module.repo_path(eval_results_path)),
                    "patch_path": module.sha256_file(module.repo_path(patch_path)),
                },
            },
        "external_action_approval": wandb_write_approval_requirement(),
        "errors": ["total_instances=1 but expected_total=80"],
    }

    errors = module.validate_existing_results_relog_dry_run_plan_payload(
        payload,
        record=record,
        label="plan",
    )

    assert errors == []


def test_verify_release_evidence_bundle_rejects_existing_results_relog_plan_that_writes_wandb():
    module = load_verify_module()
    record = {
        "benchmark": "agentic_math",
        "model_slug": "gpt-5_5",
        "result_dir": "outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw",
        "summary_path": "outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw/summary.json",
        "results_path": "outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw/results.jsonl",
        "row_count": 100,
    }
    payload = {
        "schema_version": 1,
        "ok": True,
        "will_write_wandb": True,
        "benchmark": "agentic_math",
        "entity": "test-entity",
        "project": "test-project",
        "source": {
            "results_dir": "outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw",
            "summary_json": "outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw/summary.json",
            "results_jsonl": "outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw/results.jsonl",
            "source_sha256": {
                "summary_json": "a" * 64,
                "results_jsonl": "b" * 64,
            },
        },
        "would_log": {
            "tables": {"agentic_math_output_table": 99},
            "artifact": {"aliases": ["latest"]},
        },
        "post_log_verifier_command_template": (
            "uv run python scripts/tools/verify_taiwan_wandb_completion.py "
            "--run-id RUN_ID_AFTER_WANDB_LOG --benchmark agentic_swe"
        ),
    }

    errors = module.validate_existing_results_relog_dry_run_plan_payload(
        payload,
        record=record,
        label="plan",
    )

    assert "plan will_write_wandb must be false" in errors
    assert "plan artifact aliases must include production" in errors
    assert "plan verifier command benchmark does not match plan" in errors
    assert "plan output table row count does not match audit record" in errors


def test_verify_release_evidence_bundle_rejects_existing_results_relog_plan_missing_source_sha256():
    module = load_verify_module()
    record = {
        "benchmark": "agentic_math",
        "model_slug": "gpt-5_5",
        "result_dir": "outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw",
        "summary_path": "outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw/summary.json",
        "results_path": "outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw/results.jsonl",
        "row_count": 100,
    }
    payload = {
        "schema_version": 1,
        "ok": True,
        "will_write_wandb": False,
        "benchmark": "agentic_math",
        "entity": "test-entity",
        "project": "test-project",
        "source": {
            "results_dir": "outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw",
            "summary_json": "outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw/summary.json",
            "results_jsonl": "outputs/taiwan_full_eval/agentic_math/gpt-5_5/openclaw/results.jsonl",
        },
        "would_log": {
            "tables": {"agentic_math_output_table": 100},
            "artifact": {"aliases": ["latest", "production"]},
        },
        "post_log_verifier_command_template": (
            "uv run python scripts/tools/verify_taiwan_wandb_completion.py "
            "--run-id RUN_ID_AFTER_WANDB_LOG --benchmark agentic_math"
        ),
    }

    errors = module.validate_existing_results_relog_dry_run_plan_payload(
        payload,
        record=record,
        label="plan",
    )

    assert "plan source.source_sha256 must be an object" in errors
    assert "plan source.source_sha256.summary_json is missing" in errors
    assert "plan source.source_sha256.results_jsonl is missing" in errors


def test_verify_release_evidence_bundle_rejects_existing_results_relog_plan_source_sha256_mismatch(tmp_path):
    module = load_verify_module()
    result_dir = tmp_path / "math"
    result_dir.mkdir()
    summary_path = result_dir / "summary.json"
    results_path = result_dir / "results.jsonl"
    summary_path.write_text("{}", encoding="utf-8")
    results_path.write_text("{}\n", encoding="utf-8")
    record = {
        "benchmark": "agentic_math",
        "model_slug": "gpt-5_5",
        "result_dir": str(result_dir),
        "summary_path": str(summary_path),
        "results_path": str(results_path),
        "row_count": 1,
    }
    payload = {
        "schema_version": 1,
        "ok": True,
        "will_write_wandb": False,
        "benchmark": "agentic_math",
        "entity": "test-entity",
        "project": "test-project",
        "source": {
            "results_dir": str(result_dir),
            "summary_json": str(summary_path),
            "results_jsonl": str(results_path),
            "source_sha256": {
                "summary_json": "a" * 64,
                "results_jsonl": "b" * 64,
            },
        },
        "would_log": {
            "tables": {"agentic_math_output_table": 1},
            "artifact": {"aliases": ["latest", "production"]},
        },
        "post_log_verifier_command_template": (
            "uv run python scripts/tools/verify_taiwan_wandb_completion.py "
            "--run-id RUN_ID_AFTER_WANDB_LOG --benchmark agentic_math"
        ),
    }

    errors = module.validate_existing_results_relog_dry_run_plan_payload(
        payload,
        record=record,
        label="plan",
    )

    assert "plan source.source_sha256.summary_json does not match current source file" in errors
    assert "plan source.source_sha256.results_jsonl does not match current source file" in errors


def test_verify_release_evidence_bundle_accepts_existing_results_relog_command_contract():
    module = load_verify_module()
    record = {
        "benchmark": "agentic_math",
        "model_slug": "gpt-5_5",
        "relog_dry_run_plan_json": "temp/wandb_relog_plans/agentic-math-gpt-5-5.plan.json",
        "relog_dry_run_command": (
            "uv run python scripts/tools/log_agentic_math_results_to_wandb.py "
            "--results-dir outputs/math --model-name gpt-5.5 --dry-run "
            "--plan-json temp/wandb_relog_plans/agentic-math-gpt-5-5.plan.json"
        ),
        "relog_command": (
            "uv run python scripts/tools/log_agentic_math_results_to_wandb.py "
            "--results-dir outputs/math --model-name gpt-5.5 "
            "--validated-dry-run-plan-json "
            "temp/wandb_relog_plans/agentic-math-gpt-5-5.plan.json "
            "--external-action-approval-source-packet-json "
            "outputs/taiwan_release_evidence/bundle/external_action_approval_packet.json "
            "--external-action-approval-report-json "
            "temp/taiwan_external_action_approval_REVIEWED.verify.json"
        ),
    }

    errors = module.validate_existing_results_relog_command_contract(
        record,
        label="plan",
    )

    assert errors == []


def test_verify_release_evidence_bundle_rejects_existing_results_relog_command_without_validated_plan():
    module = load_verify_module()
    record = {
        "benchmark": "agentic_swe",
        "model_slug": "deepseek",
        "relog_dry_run_plan_json": "temp/wandb_relog_plans/agentic-swe-deepseek.plan.json",
        "relog_dry_run_command": (
            "uv run python scripts/tools/log_agentic_swe_results_to_wandb.py "
            "--official-eval-dir outputs/swe --patch-path outputs/patches.json "
            "--model-name deepseek --dry-run --plan-json "
            "temp/wandb_relog_plans/agentic-swe-deepseek.plan.json "
            "--validated-dry-run-plan-json "
            "temp/wandb_relog_plans/agentic-swe-deepseek.plan.json"
        ),
        "relog_command": (
            "uv run python scripts/tools/log_agentic_swe_results_to_wandb.py "
            "--official-eval-dir outputs/swe --patch-path outputs/patches.json "
            "--model-name deepseek"
        ),
    }

    errors = module.validate_existing_results_relog_command_contract(
        record,
        label="plan",
    )

    assert "plan dry-run command must not require an existing validated plan" in errors
    assert "plan relog command must include --validated-dry-run-plan-json" in errors
    assert "plan relog command must include --external-action-approval-source-packet-json" in errors
    assert "plan relog command must include --external-action-approval-report-json" in errors


def test_verify_release_evidence_bundle_rejects_wandb_adoption_sync_dry_run_report_mismatch(tmp_path):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_sync_dry_run=True,
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    role = "wandb_adoption_draft:candidate:agentic_math:sync_dry_run_report_json"
    report_record = next(
        record for record in manifest["files"] if role in record.get("roles", [])
    )
    bundled_report = bundle / report_record["bundle_path"]
    payload = json.loads(bundled_report.read_text(encoding="utf-8"))
    payload["dry_run"] = False
    payload["in_place"] = True
    payload["review_path"] = "outputs/taiwan_full_eval/other_paid_run_review.json"
    payload["unmatched_count"] = 1
    payload["changes"][0]["run_id"] = "other-run"
    bundled_report.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, report_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "W&B adoption draft candidate agentic_math sync dry-run report dry_run must be true"
        in error
        for error in payload["errors"]
    )
    assert any(
        "W&B adoption draft candidate agentic_math sync dry-run report review_path does not match candidate"
        in error
        for error in payload["errors"]
    )
    assert any(
        "W&B adoption draft candidate agentic_math sync dry-run report changes must include top_level action for candidate run"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_adoption_sync_dry_run_missing_entries(tmp_path):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_sync_dry_run=True,
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    role = "wandb_adoption_draft:candidate:agentic_math:sync_dry_run_report_json"
    report_record = next(
        record for record in manifest["files"] if role in record.get("roles", [])
    )
    bundled_report = bundle / report_record["bundle_path"]
    payload = json.loads(bundled_report.read_text(encoding="utf-8"))
    payload.pop("entries", None)
    bundled_report.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, report_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "W&B adoption draft candidate agentic_math sync dry-run report entries must be a list"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_adoption_sync_dry_run_missing_generated_at(tmp_path):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_sync_dry_run=True,
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    role = "wandb_adoption_draft:candidate:agentic_math:sync_dry_run_report_json"
    report_record = next(
        record for record in manifest["files"] if role in record.get("roles", [])
    )
    bundled_report = bundle / report_record["bundle_path"]
    payload = json.loads(bundled_report.read_text(encoding="utf-8"))
    payload.pop("generated_at", None)
    bundled_report.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, report_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "W&B adoption draft candidate agentic_math sync dry-run report "
        "generated_at must be a positive number" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_adoption_sync_dry_run_missing_source_review_sha(
    tmp_path,
):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_sync_dry_run=True,
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    role = "wandb_adoption_draft:candidate:agentic_math:sync_dry_run_report_json"
    report_record = next(
        record for record in manifest["files"] if role in record.get("roles", [])
    )
    bundled_report = bundle / report_record["bundle_path"]
    payload = json.loads(bundled_report.read_text(encoding="utf-8"))
    payload.pop("source_review_sha256", None)
    bundled_report.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, report_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "W&B adoption draft candidate agentic_math sync dry-run report "
        "source_review_sha256 must be a 64-character lowercase hex digest" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_adoption_sync_dry_run_source_attestation_sha_mismatch(
    tmp_path,
):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_sync_dry_run=True,
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    role = "wandb_adoption_draft:candidate:agentic_math:sync_dry_run_report_json"
    report_record = next(
        record for record in manifest["files"] if role in record.get("roles", [])
    )
    bundled_report = bundle / report_record["bundle_path"]
    payload = json.loads(bundled_report.read_text(encoding="utf-8"))
    payload["entries"][0]["scope_attestation"]["source_attestation_sha256"] = "0" * 64
    bundled_report.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, report_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "W&B adoption draft candidate agentic_math sync dry-run report "
        "candidate entry scope_attestation source_attestation_sha256 does not match bundled source attestation JSON"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_adoption_sync_dry_run_status_mismatch(tmp_path):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_sync_dry_run=True,
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    role = "wandb_adoption_draft:candidate:agentic_math:sync_dry_run_report_json"
    report_record = next(
        record for record in manifest["files"] if role in record.get("roles", [])
    )
    bundled_report = bundle / report_record["bundle_path"]
    payload = json.loads(bundled_report.read_text(encoding="utf-8"))
    payload["before_status"] = "prepared"
    payload["after_status"] = "completed"
    bundled_report.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, report_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "W&B adoption draft candidate agentic_math sync dry-run report "
        "before_status and after_status must match" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_adoption_sync_dry_run_unconfirmed_source(tmp_path):
    bundle, _source_audit, _completion = build_bundle_with_wandb_adoption_draft(
        tmp_path,
        include_sync_dry_run=True,
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    role = "wandb_adoption_draft:candidate:agentic_math:scope_attestation_template_json"
    template_record = next(
        record for record in manifest["files"] if role in record.get("roles", [])
    )
    bundled_template = bundle / template_record["bundle_path"]
    payload = json.loads(bundled_template.read_text(encoding="utf-8"))
    payload["confirmed"] = False
    payload["confirmed_at"] = "YYYY-MM-DDTHH:MM:SS+09:00"
    bundled_template.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, template_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "W&B adoption draft candidate agentic_math sync dry-run report "
        "source scope_attestation JSON confirmed is not true" in error
        for error in payload["errors"]
    )
    assert any(
        "W&B adoption draft candidate agentic_math sync dry-run report "
        "source scope_attestation JSON confirmed_at must be a timezone-aware ISO 8601 timestamp"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_accepts_paid_review_scope_attestation_source(tmp_path):
    bundle, attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    records = {record["source_path"]: record for record in manifest["files"]}
    assert any(
        role.endswith("agentic_math:source_attestation_json")
        for role in records[str(attestation)]["roles"]
    )
    budget_records = [
        record
        for record in manifest["files"]
        if any(str(role).endswith(":pre_run_budget_estimate") for role in record.get("roles", []))
    ]
    assert len(budget_records) == 1


def test_verify_release_evidence_bundle_requires_paid_review_wandb_completion_run_metadata(tmp_path):
    bundle, _attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    completion_record = next(
        record
        for record in manifest["files"]
        if any(
            str(role).endswith("paid_run_review_package:wandb_completion")
            for role in record.get("roles", [])
        )
    )
    bundled_completion = bundle / completion_record["bundle_path"]
    payload = json.loads(bundled_completion.read_text(encoding="utf-8"))
    payload["required_evidence"].pop("run_metadata", None)
    payload["observed_evidence"].pop("run_metadata", None)
    bundled_completion.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, completion_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "required_evidence.run_metadata is required for paid-review/release proof"
        in error
        for error in payload["errors"]
    )
    assert any(
        "observed_evidence.run_metadata is required for paid-review/release proof"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_missing_pre_run_budget_source(tmp_path):
    bundle, _attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [
        record
        for record in manifest["files"]
        if not any(str(role).endswith(":pre_run_budget_estimate") for role in record.get("roles", []))
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert any(
        "missing bundled evidence for paid review pre-run budget estimate" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_includes_paid_review_run_eval_preflight(tmp_path):
    bundle, _attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))

    preflight_records = [
        record
        for record in manifest["files"]
        if any(str(role).endswith(":run_eval_preflight") for role in record.get("roles", []))
    ]

    assert preflight_records
    payload = json.loads((bundle / preflight_records[0]["bundle_path"]).read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["status"] == "passed"
    assert payload["will_initialize_wandb"] is False
    assert payload["will_start_inference_engine"] is False
    assert payload["will_run_evaluators"] is False


def test_verify_release_evidence_bundle_rejects_missing_paid_review_run_eval_preflight(
    tmp_path,
):
    bundle, _attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [
        record
        for record in manifest["files"]
        if not any(str(role).endswith(":run_eval_preflight") for role in record.get("roles", []))
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert any(
        "missing bundled evidence for paid review run_eval preflight" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_mutated_paid_review_run_eval_preflight(
    tmp_path,
):
    bundle, _attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    preflight_record = next(
        record
        for record in manifest["files"]
        if any(str(role).endswith(":run_eval_preflight") for role in record.get("roles", []))
    )
    bundled_preflight = bundle / preflight_record["bundle_path"]
    payload = json.loads(bundled_preflight.read_text(encoding="utf-8"))
    payload["will_initialize_wandb"] = True
    bundled_preflight.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, preflight_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert any(
        "will_initialize_wandb must be false" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_pre_run_budget_model_binding_mismatch(
    tmp_path,
):
    bundle, _attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    budget = manifest["current_gate"]["paid_run_review_package"]["gates"][0]["records"][0][
        "pre_run_budget_estimate"
    ]
    budget["target_models"] = ["openai-direct/gpt-4.1-mini-2025-04-14"]
    budget["selected_model_identifiers"] = [
        "gpt-4.1-mini-2025-04-14",
        "openai-direct/gpt-4.1-mini-2025-04-14",
    ]
    budget["target_model_matches_selected_config"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "paid review pre-run budget estimate paid_run_review_package#1 "
        "target_model does not match selected config model identifiers" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_bundled_review_pre_run_budget_model_binding_mismatch(
    tmp_path,
):
    bundle, _attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    review_record = next(
        record
        for record in manifest["files"]
        if record.get("source_path", "").endswith("canary_agentic_paid_run_review.json")
    )
    bundled_review = bundle / review_record["bundle_path"]
    review_payload = json.loads(bundled_review.read_text(encoding="utf-8"))
    budget = review_payload["pre_run_budget_estimate"]
    budget["target_models"] = ["openai-direct/gpt-4.1-mini-2025-04-14"]
    budget["selected_model_identifiers"] = ["openai-direct/not-the-reviewed-model"]
    budget["target_model_matches_selected_config"] = False
    bundled_review.write_text(json.dumps(review_payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, review_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "paid review pre-run budget estimate paid_run_review_package#1 "
        "bundled paid review JSON target_model does not match selected config model identifiers"
        in error
        for error in payload["errors"]
    )
    assert any(
        "paid review pre-run budget estimate paid_run_review_package#1 "
        "bundled paid review JSON target_model(s) do not intersect selected config model identifiers"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_paid_review_raw_missing_scope_attestation(
    tmp_path,
):
    bundle, _attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    review_record = next(
        record
        for record in manifest["files"]
        if record.get("source_path", "").endswith("canary_agentic_paid_run_review.json")
    )
    bundled_review = bundle / review_record["bundle_path"]
    review_payload = json.loads(bundled_review.read_text(encoding="utf-8"))
    review_payload["runs"][0]["wandb_completion"][0].pop("scope_attestation", None)
    bundled_review.write_text(json.dumps(review_payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, review_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "paid review W&B completion entry paid_run_review_package#1.1 "
        "agentic_math/run-1 adopted existing result is missing scope_attestation "
        "in bundled paid review JSON" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_paid_review_raw_scope_attestation_mismatch(
    tmp_path,
):
    bundle, _attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    review_record = next(
        record
        for record in manifest["files"]
        if record.get("source_path", "").endswith("canary_agentic_paid_run_review.json")
    )
    bundled_review = bundle / review_record["bundle_path"]
    review_payload = json.loads(bundled_review.read_text(encoding="utf-8"))
    review_payload["runs"][0]["wandb_completion"][0]["scope_attestation"][
        "provider_bill_reference"
    ] = "other-bill-reference"
    bundled_review.write_text(json.dumps(review_payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, review_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "paid review W&B completion entry paid_run_review_package#1.1 "
        "agentic_math/run-1 raw paid review scope_attestation "
        "provider_bill_reference does not match current_gate" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_paid_review_completion_payload_identity_mismatch(tmp_path):
    bundle, _attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    role_suffix = "paid_run_review_package:wandb_completion"
    completion_record = next(
        record
        for record in manifest["files"]
        if any(str(role).endswith(role_suffix) for role in record.get("roles", []))
    )
    bundled_completion = bundle / completion_record["bundle_path"]
    payload = json.loads(bundled_completion.read_text(encoding="utf-8"))
    payload["entity"] = "other-entity"
    payload["project"] = "other-project"
    bundled_completion.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, completion_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "paid review W&B completion entry paid_run_review_package#1.1 "
        "agentic_math/run-1 entity does not match W&B completion payload" in error
        for error in payload["errors"]
    )
    assert any(
        "paid review W&B completion entry paid_run_review_package#1.1 "
        "agentic_math/run-1 project does not match W&B completion payload" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_completion_query_source_mismatch(tmp_path):
    bundle, _attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    role_suffix = "paid_run_review_package:wandb_completion"
    completion_record = next(
        record
        for record in manifest["files"]
        if any(str(role).endswith(role_suffix) for role in record.get("roles", []))
    )
    bundled_completion = bundle / completion_record["bundle_path"]
    payload = json.loads(bundled_completion.read_text(encoding="utf-8"))
    payload["query_source"]["kind"] = "manual_json"
    payload["query_source"]["run_path"] = "test-entity/test-project/other-run"
    bundled_completion.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, completion_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "query_source.kind mismatch: expected wandb_sdk, got manual_json" in error
        for error in payload["errors"]
    )
    assert any(
        "query_source.run_path mismatch: expected test-entity/test-project/run-1"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_non_adopted_wandb_completion_missing_query_source(tmp_path):
    bundle, _attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["current_gate"]["paid_run_review_package"]["gates"][0]["records"][0][
        "wandb_completion_entries"
    ][0]
    entry["adopted_existing_result"] = False
    entry.pop("scope_attestation", None)
    entry["scope_attestation_valid"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    completion_record = next(
        record
        for record in manifest["files"]
        if any(
            str(role).endswith("paid_run_review_package:wandb_completion")
            for role in record.get("roles", [])
        )
    )
    bundled_completion = bundle / completion_record["bundle_path"]
    completion_payload = json.loads(bundled_completion.read_text(encoding="utf-8"))
    completion_payload.pop("query_source")
    bundled_completion.write_text(json.dumps(completion_payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, completion_record["bundle_path"])

    review_record = next(
        record
        for record in manifest["files"]
        if record.get("source_path", "").endswith("canary_agentic_paid_run_review.json")
    )
    bundled_review = bundle / review_record["bundle_path"]
    review_payload = json.loads(bundled_review.read_text(encoding="utf-8"))
    raw_entry = review_payload["runs"][0]["wandb_completion"][0]
    raw_entry["adopted_existing_result"] = False
    raw_entry.pop("scope_attestation", None)
    bundled_review.write_text(json.dumps(review_payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, review_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "paid review W&B completion entry paid_run_review_package#1.1 "
        "agentic_math/run-1 query_source is not an object" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_paid_review_completion_parent_identity_mismatch(tmp_path):
    bundle, _attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["current_gate"]["paid_run_review_package"]["gates"][0]["records"][0][
        "wandb_completion_entries"
    ][0]
    entry["parent_run_id"] = "other-run"
    entry["parent_run_id_matches"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "paid review W&B completion entry paid_run_review_package#1.1 "
        "agentic_math/run-1 run_id does not match parent review run" in error
        for error in payload["errors"]
    )
    assert any(
        "paid review W&B completion entry paid_run_review_package#1.1 "
        "agentic_math/run-1 parent_run_id_matches is not true" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_paid_review_completion_missing_from_review_json(tmp_path):
    bundle, _attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    review_record = next(
        record
        for record in manifest["files"]
        if record.get("source_path", "").endswith("canary_agentic_paid_run_review.json")
    )
    bundled_review = bundle / review_record["bundle_path"]
    review_payload = json.loads(bundled_review.read_text(encoding="utf-8"))
    review_payload["runs"][0]["wandb_completion"] = []
    bundled_review.write_text(json.dumps(review_payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, review_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "paid review W&B completion entry paid_run_review_package#1.1 "
        "agentic_math/run-1 is not present in bundled paid review JSON" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_paid_review_summary_review_json_mismatch(tmp_path):
    bundle, _attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    review_record = next(
        record
        for record in manifest["files"]
        if record.get("source_path", "").endswith("canary_agentic_paid_run_review.json")
    )
    bundled_review = bundle / review_record["bundle_path"]
    review_payload = json.loads(bundled_review.read_text(encoding="utf-8"))
    review_payload["status"] = "prepared"
    bundled_review.write_text(json.dumps(review_payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, review_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "paid review W&B completion entry paid_run_review_package#1.1 "
        "agentic_math/run-1 paid review status does not match bundled review JSON" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_accepts_prepared_one_model_review_source(tmp_path):
    bundle = build_bundle_with_prepared_one_model_review(tmp_path)

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is True
    assert not payload["errors"]


def test_verify_release_evidence_bundle_rejects_prepared_one_model_review_summary_mismatch(
    tmp_path,
):
    bundle = build_bundle_with_prepared_one_model_review(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["current_gate"]["paid_run_review_package"]["gates"][0]["records"][0]
    record["run_purpose_present"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "paid review record one_model_full_canary#1 "
        "run_purpose_present does not match bundled review JSON" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_paid_review_completion_entry_sha_mismatch(tmp_path):
    bundle, _attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["current_gate"]["paid_run_review_package"]["gates"][0]["records"][0][
        "wandb_completion_entries"
    ][0]
    entry["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "paid review W&B completion entry paid_run_review_package#1.1 "
        "agentic_math/run-1 sha256 does not match bundled W&B completion JSON" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_requires_paid_review_scope_attestation_source(tmp_path):
    bundle, attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [
        record
        for record in manifest["files"]
        if record.get("source_path") != str(attestation)
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "missing bundled evidence for paid review scope attestation source JSON for agentic_math"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_paid_review_scope_attestation_mismatch(tmp_path):
    bundle, _attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    role_suffix = "agentic_math:source_attestation_json"
    attestation_record = next(
        record
        for record in manifest["files"]
        if any(str(role).endswith(role_suffix) for role in record.get("roles", []))
    )
    bundled_attestation = bundle / attestation_record["bundle_path"]
    payload = json.loads(bundled_attestation.read_text(encoding="utf-8"))
    payload["run_id"] = "other-run"
    bundled_attestation.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, attestation_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "paid review scope attestation source JSON for agentic_math run_id does not match paid review entry"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_paid_review_scope_attestation_placeholder_accounting(tmp_path):
    bundle, _attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    role_suffix = "agentic_math:source_attestation_json"
    attestation_record = next(
        record
        for record in manifest["files"]
        if any(str(role).endswith(role_suffix) for role in record.get("roles", []))
    )
    bundled_attestation = bundle / attestation_record["bundle_path"]
    payload = json.loads(bundled_attestation.read_text(encoding="utf-8"))
    payload["actual_cost_estimate"] = "$ACTUAL_OR_BILLING_ESTIMATE"
    payload["provider_bill_reference"] = "BILL_OR_DASHBOARD_REFERENCE"
    bundled_attestation.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, attestation_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "paid review scope attestation source JSON for agentic_math actual_cost_estimate must not be a placeholder"
        in error
        for error in payload["errors"]
    )
    assert any(
        "paid review scope attestation source JSON for agentic_math provider_bill_reference must not be a placeholder"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_paid_review_scope_attestation_template_confirmed_at(tmp_path):
    bundle, _attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    role_suffix = "agentic_math:source_attestation_json"
    attestation_record = next(
        record
        for record in manifest["files"]
        if any(str(role).endswith(role_suffix) for role in record.get("roles", []))
    )
    bundled_attestation = bundle / attestation_record["bundle_path"]
    payload = json.loads(bundled_attestation.read_text(encoding="utf-8"))
    payload["confirmed_at"] = "YYYY-MM-DDTHH:MM:SS+09:00"
    bundled_attestation.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, attestation_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "paid review scope attestation source JSON for agentic_math confirmed_at "
        "must be a timezone-aware ISO 8601 timestamp" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_paid_review_record_placeholder_accounting_flags(tmp_path):
    bundle, _attestation = build_bundle_with_paid_review_scope_attestation(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["current_gate"]["paid_run_review_package"]["gates"][0]["records"][0]
    record["actual_cost_estimate_placeholder"] = True
    record["provider_bill_reference_placeholder"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "paid review record paid_run_review_package#1 actual_cost_estimate must not be a placeholder"
        in error
        for error in payload["errors"]
    )
    assert any(
        "paid review record paid_run_review_package#1 provider_bill_reference must not be a placeholder"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_requires_wandb_completion_observed_evidence(tmp_path):
    bundle, _source_audit, completion = build_bundle_with_wandb_adoption_draft(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    completion_record = next(record for record in manifest["files"] if record["source_path"] == str(completion))
    bundled_completion = bundle / completion_record["bundle_path"]
    payload = json.loads(bundled_completion.read_text(encoding="utf-8"))
    payload.pop("observed_evidence")
    bundled_completion.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, completion_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "observed_evidence is not an object" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_requires_wandb_completion_entity_and_project(tmp_path):
    bundle, _source_audit, completion = build_bundle_with_wandb_adoption_draft(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    completion_record = next(record for record in manifest["files"] if record["source_path"] == str(completion))
    bundled_completion = bundle / completion_record["bundle_path"]
    payload = json.loads(bundled_completion.read_text(encoding="utf-8"))
    payload.pop("entity")
    payload["project"] = ""
    bundled_completion.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, completion_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any("entity is missing" in error for error in payload["errors"])
    assert any("project is missing" in error for error in payload["errors"])


def test_verify_release_evidence_bundle_requires_wandb_completion_observed_metrics_tables_and_artifacts(tmp_path):
    bundle, _source_audit, completion = build_bundle_with_wandb_adoption_draft(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    completion_record = next(record for record in manifest["files"] if record["source_path"] == str(completion))
    bundled_completion = bundle / completion_record["bundle_path"]
    payload = json.loads(bundled_completion.read_text(encoding="utf-8"))
    payload["observed_evidence"]["summary_metrics"].pop("agentic_math/accuracy")
    payload["observed_evidence"]["tables"][1]["nrows"] = 99
    payload["observed_evidence"]["artifacts"][0]["aliases"] = ["latest"]
    bundled_completion.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, completion_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "observed_evidence.summary_metrics missing agentic_math/accuracy" in error
        for error in payload["errors"]
    )
    assert any(
        "agentic_math_output_table nrows must equal expected_total 100, got 99" in error
        for error in payload["errors"]
    )
    assert any(
        "observed_evidence.artifacts type evaluation-results missing required alias production"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rechecks_wandb_completion_metric_consistency(tmp_path):
    bundle, _source_audit, completion = build_bundle_with_wandb_adoption_draft(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    completion_record = next(record for record in manifest["files"] if record["source_path"] == str(completion))
    bundled_completion = bundle / completion_record["bundle_path"]
    payload = json.loads(bundled_completion.read_text(encoding="utf-8"))
    metrics = payload["observed_evidence"]["summary_metrics"]
    metrics["agentic_math/total_instances"]["value"] = 99
    metrics["agentic_math/answered_instances"]["value"] = 101
    metrics["agentic_math/accuracy"]["value"] = 0.86
    bundled_completion.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, completion_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "observed_evidence.summary_metrics agentic_math/total_instances must equal expected_total 100, got 99"
        in error
        for error in payload["errors"]
    )
    assert any(
        "observed_evidence.summary_metrics agentic_math/answered_instances must be <= agentic_math/total_instances 99, got 101"
        in error
        for error in payload["errors"]
    )
    assert any(
        "observed_evidence.summary_metrics agentic_math/accuracy must equal numerator/total"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_accepts_wandb_completion_run_metadata(tmp_path):
    bundle, _source_audit, completion = build_bundle_with_wandb_adoption_draft(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    completion_record = next(record for record in manifest["files"] if record["source_path"] == str(completion))
    bundled_completion = bundle / completion_record["bundle_path"]
    payload = add_wandb_completion_run_metadata(
        json.loads(bundled_completion.read_text(encoding="utf-8"))
    )
    bundled_completion.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, completion_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is True


def test_verify_release_evidence_bundle_rejects_wandb_completion_run_metadata_mismatch(tmp_path):
    bundle, _source_audit, completion = build_bundle_with_wandb_adoption_draft(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    completion_record = next(record for record in manifest["files"] if record["source_path"] == str(completion))
    bundled_completion = bundle / completion_record["bundle_path"]
    payload = add_wandb_completion_run_metadata(
        json.loads(bundled_completion.read_text(encoding="utf-8"))
    )
    payload["observed_evidence"]["run_metadata"]["config"][0]["value"] = "wrong-model"
    payload["observed_evidence"]["run_metadata"]["tags"] = []
    payload["checks"] = [
        check
        for check in payload["checks"]
        if check.get("name") != "run_group"
    ]
    bundled_completion.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, completion_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "observed_evidence.run_metadata.config model.pretrained_model_name_or_path value mismatch"
        in error
        for error in payload["errors"]
    )
    assert any(
        "observed_evidence.run_metadata.tags missing taiwan-canary" in error
        for error in payload["errors"]
    )
    assert any("checks run_group missing" in error for error in payload["errors"])


def test_verify_release_evidence_bundle_requires_wandb_completion_checks(tmp_path):
    bundle, _source_audit, completion = build_bundle_with_wandb_adoption_draft(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    completion_record = next(record for record in manifest["files"] if record["source_path"] == str(completion))
    bundled_completion = bundle / completion_record["bundle_path"]
    payload = json.loads(bundled_completion.read_text(encoding="utf-8"))
    payload["checks"] = [
        check
        for check in payload["checks"]
        if check.get("name") != "accuracy_metric"
    ]
    payload["checks"][0]["ok"] = False
    bundled_completion.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, completion_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "checks contains failing check: run_state" in error
        for error in payload["errors"]
    )
    assert any(
        "checks missing required check: accuracy_metric" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rechecks_wandb_completion_checks_against_observed_evidence(tmp_path):
    bundle, _source_audit, completion = build_bundle_with_wandb_adoption_draft(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    completion_record = next(record for record in manifest["files"] if record["source_path"] == str(completion))
    bundled_completion = bundle / completion_record["bundle_path"]
    payload = json.loads(bundled_completion.read_text(encoding="utf-8"))
    checks = {check["name"]: check for check in payload["checks"]}
    checks["run_state"]["state"] = "running"
    checks["total_metric"]["value"] = 99
    checks["output_table"]["nrows"] = 99
    checks["nemoclaw_session_audit"]["passed"] = 99
    checks["result_artifact"]["artifacts"][0]["aliases"] = ["latest"]
    bundled_completion.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, completion_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "checks run_state state does not match observed_evidence.run_state" in error
        for error in payload["errors"]
    )
    assert any(
        "checks total_metric value does not match observed_evidence.summary_metrics agentic_math/total_instances" in error
        for error in payload["errors"]
    )
    assert any(
        "checks output_table nrows does not match observed_evidence table agentic_math_output_table" in error
        for error in payload["errors"]
    )
    assert any(
        "checks result_artifact type evaluation-results missing required alias production"
        in error
        for error in payload["errors"]
    )
    assert any(
        "checks nemoclaw_session_audit passed does not match observed_evidence.nemoclaw_session_audit"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_completion_unknown_or_duplicate_checks(tmp_path):
    bundle, _source_audit, completion = build_bundle_with_wandb_adoption_draft(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    completion_record = next(record for record in manifest["files"] if record["source_path"] == str(completion))
    bundled_completion = bundle / completion_record["bundle_path"]
    payload = json.loads(bundled_completion.read_text(encoding="utf-8"))
    payload["checks"].append(
        {
            "name": "run_state",
            "ok": True,
            "detail": "duplicate should not be accepted",
            "state": "finished",
        }
    )
    payload["checks"].append(
        {
            "name": "manual_override",
            "ok": True,
            "detail": "unknown checks must not become release evidence",
        }
    )
    bundled_completion.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, completion_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "checks contains duplicate singleton check: run_state" in error
        for error in payload["errors"]
    )
    assert any(
        "checks contains unknown check: manual_override" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_accepts_nemoclaw_setup_acceptance_metadata(tmp_path):
    bundle, _setup = build_bundle_with_nemoclaw_adoption(tmp_path)

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is True


def test_verify_release_evidence_bundle_rejects_nemoclaw_missing_operator_sequence(tmp_path):
    bundle, setup = build_bundle_with_nemoclaw_adoption(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    setup_record = next(record for record in manifest["files"] if record["source_path"] == str(setup))
    bundled_setup = bundle / setup_record["bundle_path"]
    payload = json.loads(bundled_setup.read_text(encoding="utf-8"))
    payload["setup_plan"].pop("operator_sequence")
    payload["setup_plan"].pop("expected_evidence_paths")
    bundled_setup.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, setup_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "setup_plan.operator_sequence is missing or invalid" in error
        for error in payload["errors"]
    )
    assert any(
        "setup_plan.expected_evidence_paths is missing or invalid" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_nemoclaw_missing_production_install_command_flag(tmp_path):
    bundle, setup = build_bundle_with_nemoclaw_adoption(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    setup_record = next(record for record in manifest["files"] if record["source_path"] == str(setup))
    bundled_setup = bundle / setup_record["bundle_path"]
    payload = json.loads(bundled_setup.read_text(encoding="utf-8"))
    payload["setup_plan"]["production_install_and_onboard_command"] = payload["setup_plan"][
        "production_install_and_onboard_command"
    ].replace(
        "--installer-review-json temp/nemoclaw_installer_review.json ",
        "",
    )
    bundled_setup.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, setup_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "setup_plan.production_install_and_onboard_command missing --installer-review-json"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_nemoclaw_setup_review_command_missing_expected_sha(
    tmp_path,
):
    bundle, setup = build_bundle_with_nemoclaw_adoption(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    setup_record = next(record for record in manifest["files"] if record["source_path"] == str(setup))
    bundled_setup = bundle / setup_record["bundle_path"]
    payload = json.loads(bundled_setup.read_text(encoding="utf-8"))
    command = payload["setup_plan"]["installer_review_command"].replace(
        f"--expected-sha256 {INSTALLER_SHA256} ",
        "",
    )
    payload["setup_plan"]["installer_review_command"] = command
    for row in payload["setup_plan"]["operator_sequence"]:
        if row["step"] == "installer_review":
            row["command"] = command
    bundled_setup.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, setup_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "setup_plan.installer_review_command missing --expected-sha256" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_nemoclaw_setup_post_install_command_missing_fail_flag(
    tmp_path,
):
    bundle, setup = build_bundle_with_nemoclaw_adoption(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    setup_record = next(record for record in manifest["files"] if record["source_path"] == str(setup))
    bundled_setup = bundle / setup_record["bundle_path"]
    payload = json.loads(bundled_setup.read_text(encoding="utf-8"))
    command = payload["setup_plan"]["post_install_verification_command"].replace(
        " --fail-on-failed",
        "",
    )
    payload["setup_plan"]["post_install_verification_command"] = command
    for row in payload["setup_plan"]["operator_sequence"]:
        if row["step"] == "post_install_verification":
            row["command"] = command
    bundled_setup.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, setup_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "setup_plan.post_install_verification_command missing --fail-on-failed" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_nemoclaw_setup_post_install_command_missing_openclaw_config_path(
    tmp_path,
):
    bundle, setup = build_bundle_with_nemoclaw_adoption(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    setup_record = next(record for record in manifest["files"] if record["source_path"] == str(setup))
    bundled_setup = bundle / setup_record["bundle_path"]
    payload = json.loads(bundled_setup.read_text(encoding="utf-8"))
    command = payload["setup_plan"]["post_install_verification_command"].replace(
        " --nemoclaw-openclaw-config-path /sandbox/.openclaw/openclaw.json",
        "",
    )
    payload["setup_plan"]["post_install_verification_command"] = command
    for row in payload["setup_plan"]["operator_sequence"]:
        if row["step"] == "post_install_verification":
            row["command"] = command
    bundled_setup.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, setup_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "setup_plan.post_install_verification_command missing --nemoclaw-openclaw-config-path"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_nemoclaw_setup_installed_evidence_mismatch(tmp_path):
    bundle, setup = build_bundle_with_nemoclaw_adoption(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    setup_record = next(record for record in manifest["files"] if record["source_path"] == str(setup))
    bundled_setup = bundle / setup_record["bundle_path"]
    payload = json.loads(bundled_setup.read_text(encoding="utf-8"))
    payload["missing_required_commands"] = []
    bundled_setup.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, setup_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "NeMoClaw setup_installed missing_required_commands does not match setup evidence"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_requires_nemoclaw_third_party_metadata(tmp_path):
    bundle, setup = build_bundle_with_nemoclaw_adoption(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    setup_record = next(record for record in manifest["files"] if record["source_path"] == str(setup))
    bundled_setup = bundle / setup_record["bundle_path"]
    payload = json.loads(bundled_setup.read_text(encoding="utf-8"))
    payload.pop("third_party_software")
    bundled_setup.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, setup_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "third_party_software missing/invalid vendor" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_requires_nemoclaw_installer_provenance_metadata(tmp_path):
    bundle, setup = build_bundle_with_nemoclaw_adoption(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    setup_record = next(record for record in manifest["files"] if record["source_path"] == str(setup))
    bundled_setup = bundle / setup_record["bundle_path"]
    payload = json.loads(bundled_setup.read_text(encoding="utf-8"))
    payload["third_party_software"].pop("installer_provenance_locked")
    payload["setup_plan"].pop("installer_provenance_locked")
    bundled_setup.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, setup_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "third_party_software missing/invalid installer_provenance_locked" in error
        for error in payload["errors"]
    )
    assert any(
        "setup_plan.installer_provenance_locked is missing or invalid" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_requires_nemoclaw_installer_lock_json(tmp_path):
    bundle, setup = build_bundle_with_nemoclaw_adoption(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    setup_record = next(record for record in manifest["files"] if record["source_path"] == str(setup))
    bundled_setup = bundle / setup_record["bundle_path"]
    payload = json.loads(bundled_setup.read_text(encoding="utf-8"))
    payload["third_party_software"].pop("installer_lock_json")
    payload["setup_plan"].pop("installer_lock_json")
    payload["setup_plan"]["acceptance_ledger_fields"].remove(
        "third_party_software.installer_lock_json"
    )
    payload["setup_plan"]["acceptance_ledger_fields"].remove("setup_plan.installer_lock_json")
    bundled_setup.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, setup_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "third_party_software missing/invalid installer_lock_json" in error
        for error in payload["errors"]
    )
    assert any(
        "setup_plan.installer_lock_json is missing or invalid" in error
        for error in payload["errors"]
    )
    assert any(
        "setup_plan.acceptance_ledger_fields missing third_party_software.installer_lock_json"
        in error
        for error in payload["errors"]
    )
    assert any(
        "setup_plan.acceptance_ledger_fields missing setup_plan.installer_lock_json" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_nemoclaw_installer_lock_json_mismatch(tmp_path):
    bundle, setup = build_bundle_with_nemoclaw_adoption(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    setup_record = next(record for record in manifest["files"] if record["source_path"] == str(setup))
    bundled_setup = bundle / setup_record["bundle_path"]
    payload = json.loads(bundled_setup.read_text(encoding="utf-8"))
    payload["third_party_software"]["installer_lock_json"] = "scripts/setup/other_lock.json"
    payload["setup_plan"]["installer_lock_json"] = "scripts/setup/other_lock.json"
    bundled_setup.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, setup_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "third_party_software.installer_lock_json does not match current_gate.nemoclaw_installer_review.lock_json"
        in error
        for error in payload["errors"]
    )
    assert any(
        "setup_plan.installer_lock_json does not match current_gate.nemoclaw_installer_review.lock_json"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_nemoclaw_install_without_locked_installer_provenance(tmp_path):
    bundle, setup = build_bundle_with_nemoclaw_adoption(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    setup_record = next(record for record in manifest["files"] if record["source_path"] == str(setup))
    bundled_setup = bundle / setup_record["bundle_path"]
    payload = json.loads(bundled_setup.read_text(encoding="utf-8"))
    payload["install_requested"] = True
    payload["operation_results"] = {
        "install": {
            "requested": True,
            "attempted": True,
            "returncode": 0,
            "log_path": "temp/nemoclaw_install.log",
        },
        "onboard": {
            "requested": False,
            "attempted": False,
            "skipped": False,
            "returncode": None,
            "log_path": "",
        },
    }
    payload["third_party_software"]["install_or_onboard_requested"] = True
    bundled_setup.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, setup_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "install/onboard requested or attempted without third_party_software.installer_provenance_locked=true"
        in error
        for error in payload["errors"]
    )
    assert any(
        "install/onboard requested or attempted without setup_plan.installer_integrity_verified=true"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_nemoclaw_attempted_install_without_bundled_log(tmp_path):
    bundle, setup = build_bundle_with_nemoclaw_adoption(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    setup_record = next(record for record in manifest["files"] if record["source_path"] == str(setup))
    bundled_setup = bundle / setup_record["bundle_path"]
    payload = json.loads(bundled_setup.read_text(encoding="utf-8"))
    payload["install_requested"] = True
    payload["accepted_third_party_software"] = True
    payload["operation_results"] = {
        "install": {
            "requested": True,
            "attempted": True,
            "returncode": 0,
            "log_path": "temp/nemoclaw_install.log",
        },
        "onboard": {
            "requested": False,
            "attempted": False,
            "skipped": False,
            "returncode": None,
            "log_path": "",
        },
    }
    for field in ("third_party_software", "setup_plan"):
        payload[field]["install_or_onboard_requested"] = True
        payload[field]["accepted"] = True
        payload[field]["installer_sha256"] = INSTALLER_SHA256
        payload[field]["installer_lock_json"] = INSTALLER_LOCK_JSON
        payload[field]["installer_review_json"] = "temp/nemoclaw_installer_review.json"
        payload[field]["installer_review_verified"] = True
        payload[field]["installer_integrity_verified"] = True
        payload[field]["installer_provenance_locked"] = True
    bundled_setup.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, setup_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "missing bundled evidence for NeMoClaw setup evidence" in error
        and "operation_results.install.log_path" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_requires_nemoclaw_restricted_policy_tier(tmp_path):
    bundle, setup = build_bundle_with_nemoclaw_adoption(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    setup_record = next(record for record in manifest["files"] if record["source_path"] == str(setup))
    bundled_setup = bundle / setup_record["bundle_path"]
    payload = json.loads(bundled_setup.read_text(encoding="utf-8"))
    payload["policy_tier"] = "balanced"
    payload["setup_plan"]["policy_tier"] = "balanced"
    bundled_setup.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, setup_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any("policy_tier must be restricted" in error for error in payload["errors"])
    assert any("setup_plan.policy_tier must be restricted" in error for error in payload["errors"])


def test_verify_release_evidence_bundle_requires_nemoclaw_policy_tier_metadata(tmp_path):
    bundle, setup = build_bundle_with_nemoclaw_adoption(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    setup_record = next(record for record in manifest["files"] if record["source_path"] == str(setup))
    bundled_setup = bundle / setup_record["bundle_path"]
    payload = json.loads(bundled_setup.read_text(encoding="utf-8"))
    payload["policy_tier_valid"] = False
    payload["setup_plan"].pop("policy_tier_allowed_values")
    payload["setup_plan"]["policy_tier_valid"] = False
    bundled_setup.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, setup_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any("policy_tier_valid must be true" in error for error in payload["errors"])
    assert any(
        "setup_plan.policy_tier_allowed_values are missing or invalid" in error
        for error in payload["errors"]
    )
    assert any(
        "setup_plan.policy_tier_valid must be true" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_nemoclaw_adoption_summary_mismatch(tmp_path):
    bundle, _setup = build_bundle_with_nemoclaw_adoption(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    adoption_record = next(
        record
        for record in manifest["files"]
        if "nemoclaw_adoption_check" in record.get("roles", [])
    )
    bundled_adoption = bundle / adoption_record["bundle_path"]
    payload = json.loads(bundled_adoption.read_text(encoding="utf-8"))
    payload["adoption_decision"]["recommendation"] = "adopt_for_agentic_math"
    bundled_adoption.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, adoption_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "NeMoClaw adoption JSON adoption_decision does not match manifest current_gate"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_nemoclaw_adoption_top_level_summary_mismatch(tmp_path):
    bundle, _setup = build_bundle_with_nemoclaw_adoption(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    adoption_record = next(
        record
        for record in manifest["files"]
        if "nemoclaw_adoption_check" in record.get("roles", [])
    )
    bundled_adoption = bundle / adoption_record["bundle_path"]
    payload = json.loads(bundled_adoption.read_text(encoding="utf-8"))
    payload["ready_for_use"] = True
    payload["missing_required_commands"] = []
    bundled_adoption.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, adoption_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "NeMoClaw adoption JSON ready_for_use does not match adoption_decision"
        in error
        for error in payload["errors"]
    )
    assert any(
        "NeMoClaw adoption JSON missing_required_commands does not match "
        "setup_runtime" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_nemoclaw_operator_handoff_mismatch(tmp_path):
    bundle, _setup = build_bundle_with_nemoclaw_adoption(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    adoption_record = next(
        record
        for record in manifest["files"]
        if "nemoclaw_adoption_check" in record.get("roles", [])
    )
    bundled_adoption = bundle / adoption_record["bundle_path"]
    payload = json.loads(bundled_adoption.read_text(encoding="utf-8"))
    payload["operator_handoff"]["steps"][0]["command"] = "tampered"
    bundled_adoption.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, adoption_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "NeMoClaw adoption JSON operator_handoff does not match setup evidence"
        in error
        for error in payload["errors"]
    )
    assert any(
        "NeMoClaw adoption JSON operator_handoff does not match manifest current_gate"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_nemoclaw_adoption_markdown_path_mismatch(tmp_path):
    bundle, _setup = build_bundle_with_nemoclaw_adoption(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    adoption_record = next(
        record
        for record in manifest["files"]
        if "nemoclaw_adoption_check" in record.get("roles", [])
    )
    bundled_adoption = bundle / adoption_record["bundle_path"]
    payload = json.loads(bundled_adoption.read_text(encoding="utf-8"))
    payload["markdown_path"] = str(bundle / "other_nemoclaw_adoption.md")
    bundled_adoption.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, adoption_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "NeMoClaw adoption JSON markdown_path does not match manifest current_gate"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_passed_swebench_nemoclaw_guard_with_offenders(tmp_path):
    bundle, _setup = build_bundle_with_nemoclaw_adoption(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    adoption_record = next(
        record
        for record in manifest["files"]
        if "nemoclaw_adoption_check" in record.get("roles", [])
    )
    bundled_adoption = bundle / adoption_record["bundle_path"]
    payload = json.loads(bundled_adoption.read_text(encoding="utf-8"))
    swebench_guard = next(
        row
        for row in payload["criteria"]
        if row["name"] == "swebench_pro_non_adoption_guard"
    )
    swebench_guard["offending_records"] = [
        {
            "path": "configs/taiwan_full/generated/config.yaml",
            "swebench_pro_nemoclaw_keys": ["nemoclaw_sandbox"],
        }
    ]
    swebench_guard["records"][0]["swebench_pro_nemoclaw_keys"] = ["nemoclaw_sandbox"]
    bundled_adoption.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, adoption_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "offending_records is non-empty" in error
        for error in payload["errors"]
    )
    assert any(
        "contains NeMoClaw key(s) without swebench_pro_nemoclaw_sandbox" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_requires_swebench_nemoclaw_config_evidence(tmp_path):
    bundle, _setup = build_bundle_with_nemoclaw_adoption(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    adoption_record = next(
        record
        for record in manifest["files"]
        if "nemoclaw_adoption_check" in record.get("roles", [])
    )
    bundled_adoption = bundle / adoption_record["bundle_path"]
    payload = json.loads(bundled_adoption.read_text(encoding="utf-8"))
    swebench_guard = next(
        row
        for row in payload["criteria"]
        if row["name"] == "swebench_pro_non_adoption_guard"
    )
    swebench_guard["evidence_paths"] = []
    bundled_adoption.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, adoption_record["bundle_path"])

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current_gate_guard = next(
        row
        for row in manifest["current_gate"]["nemoclaw_adoption"]["criteria"]
        if row["name"] == "swebench_pro_non_adoption_guard"
    )
    current_gate_guard["evidence_paths"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "SWE-Bench Pro migration guard passed but evidence_paths is empty"
        in error
        for error in payload["errors"]
    )
    assert any(
        "is not listed in evidence_paths" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_agentic_math_config_missing_deny_policy(
    tmp_path,
):
    bundle, _setup = build_bundle_with_nemoclaw_adoption(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    config_record = next(
        record
        for record in manifest["files"]
        if any(
            str(role).endswith("agentic_math_config:evidence")
            for role in record.get("roles", [])
        )
    )
    bundled_config = bundle / config_record["bundle_path"]
    text = bundled_config.read_text(encoding="utf-8")
    bundled_config.write_text(text.replace("    - web_search\n", ""), encoding="utf-8")
    refresh_manifest_record_hash(bundle, config_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "agentic_math deny_tool missing required value web_search" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_agentic_math_config_denying_local_exec(
    tmp_path,
):
    bundle, _setup = build_bundle_with_nemoclaw_adoption(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    config_record = next(
        record
        for record in manifest["files"]
        if any(
            str(role).endswith("agentic_math_config:evidence")
            for role in record.get("roles", [])
        )
    )
    bundled_config = bundle / config_record["bundle_path"]
    text = bundled_config.read_text(encoding="utf-8")
    bundled_config.write_text(
        text.replace("    - code_execution\n", "    - code_execution\n    - exec\n", 1),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, config_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "agentic_math deny_tool must not block local OpenClaw exec tool: exec" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_swebench_nemoclaw_config_without_checkout_transfer(
    tmp_path,
):
    bundle, _setup = build_bundle_with_nemoclaw_adoption(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    config_record = next(
        record
        for record in manifest["files"]
        if any(
            str(role).endswith("swebench_pro_non_adoption_guard:evidence")
            for role in record.get("roles", [])
        )
    )
    bundled_config = bundle / config_record["bundle_path"]
    text = bundled_config.read_text(encoding="utf-8")
    swe_section = "\n".join(
        [
            "swebench_pro:",
            "  nemoclaw_sandbox: nejumi-taiwan",
            "  deny_tool:",
            "    - code_execution",
            "    - web_search",
            "    - web_fetch",
            "    - browser",
            "    - browser_*",
            "    - '*search*'",
            "  deny_argument_pattern:",
            "    - https?://",
            r"    - \b(curl|wget)\b",
            r"    - \b(requests|urllib|httpx)\.",
            "",
        ]
    )
    bundled_config.write_text(text.replace("swebench_pro: {}\n", swe_section), encoding="utf-8")
    refresh_manifest_record_hash(bundle, config_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "swebench_pro must set nemoclaw_checkout_transfer_mode='copy' "
        "or nemoclaw_checkout_sandbox_root" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_accepts_nemoclaw_post_install_verification(tmp_path):
    bundle, _post_install = build_bundle_with_nemoclaw_post_install(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    summary_record = next(
        record for record in manifest["files"] if record.get("bundle_path") == "summary.md"
    )
    assert "release_summary_markdown" in summary_record["roles"]

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is True


def test_verify_release_evidence_bundle_rejects_nemoclaw_post_install_path_mismatch(tmp_path):
    bundle, _post_install = build_bundle_with_nemoclaw_post_install(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    record = next(
        record
        for record in manifest["files"]
        if "nemoclaw_post_install_verification" in record.get("roles", [])
    )
    bundled_post_install = bundle / record["bundle_path"]
    payload = json.loads(bundled_post_install.read_text(encoding="utf-8"))
    payload["path"] = str(bundle / "other_post_install.json")
    bundled_post_install.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["integrity_ok"] is False
    assert (
        "NeMoClaw post-install verification JSON path does not match runner_evidence"
        in report["errors"]
    )


def test_verify_release_evidence_bundle_rejects_nemoclaw_post_install_markdown_path_mismatch(
    tmp_path,
):
    bundle, _post_install = build_bundle_with_nemoclaw_post_install(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    record = next(
        record
        for record in manifest["files"]
        if "nemoclaw_post_install_verification" in record.get("roles", [])
    )
    bundled_post_install = bundle / record["bundle_path"]
    payload = json.loads(bundled_post_install.read_text(encoding="utf-8"))
    payload["markdown_path"] = str(bundle / "other_post_install.md")
    bundled_post_install.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["integrity_ok"] is False
    assert (
        "NeMoClaw post-install verification JSON markdown_path does not match "
        "runner_evidence"
    ) in report["errors"]


def test_verify_release_evidence_bundle_rejects_nemoclaw_setup_missing_schema_version(tmp_path):
    bundle, _setup = build_bundle_with_nemoclaw_adoption(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    setup_record = next(
        record
        for record in manifest["files"]
        if record["source_path"].endswith("nemoclaw_setup.json")
    )
    bundled_setup = bundle / setup_record["bundle_path"]
    payload = json.loads(bundled_setup.read_text(encoding="utf-8"))
    payload.pop("schema_version")
    bundled_setup.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, setup_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["integrity_ok"] is False
    assert any("schema_version must be 1" in error for error in report["errors"])


def test_verify_release_evidence_bundle_rejects_nemoclaw_adoption_missing_schema_version(tmp_path):
    bundle, _setup = build_bundle_with_nemoclaw_adoption(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    adoption_record = next(
        record
        for record in manifest["files"]
        if record["source_path"].endswith("nemoclaw_adoption.json")
    )
    bundled_adoption = bundle / adoption_record["bundle_path"]
    payload = json.loads(bundled_adoption.read_text(encoding="utf-8"))
    payload.pop("schema_version")
    bundled_adoption.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, adoption_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["integrity_ok"] is False
    assert "NeMoClaw adoption JSON schema_version must be 1" in report["errors"]


def test_verify_release_evidence_bundle_rejects_nemoclaw_post_install_missing_schema_version(
    tmp_path,
):
    bundle, post_install = build_bundle_with_nemoclaw_post_install(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    post_install_record = next(
        record for record in manifest["files"] if record["source_path"] == str(post_install)
    )
    bundled_post_install = bundle / post_install_record["bundle_path"]
    payload = json.loads(bundled_post_install.read_text(encoding="utf-8"))
    payload.pop("schema_version")
    bundled_post_install.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, post_install_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["integrity_ok"] is False
    assert "NeMoClaw post-install verification schema_version must be 1" in report["errors"]


def test_verify_release_evidence_bundle_accepts_nemoclaw_installer_review_evidence(tmp_path):
    bundle, review, lock, installer_sha = build_bundle_with_nemoclaw_installer_review(tmp_path)

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is True
    assert not payload["errors"]
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["current_gate"]["nemoclaw_installer_review"]["sha256"] == installer_sha
    assert any(
        record["source_path"] == str(review)
        and "gate:nemoclaw_readiness:latest_installer_review_json" in record["roles"]
        for record in manifest["files"]
    )
    assert any(
        record["source_path"] == str(lock)
        and "gate:nemoclaw_readiness:latest_installer_review_lock_json" in record["roles"]
        for record in manifest["files"]
    )


def test_verify_release_evidence_bundle_rejects_nemoclaw_installer_review_sha_mismatch(tmp_path):
    bundle, review, _lock, _installer_sha = build_bundle_with_nemoclaw_installer_review(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    review_record = next(record for record in manifest["files"] if record["source_path"] == str(review))
    bundled_review = bundle / review_record["bundle_path"]
    payload = json.loads(bundled_review.read_text(encoding="utf-8"))
    payload["sha256"] = "0" * 64
    bundled_review.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, review_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert "NeMoClaw installer review JSON sha256 does not match current_gate summary" in payload["errors"]


def test_verify_release_evidence_bundle_rejects_nemoclaw_installer_lock_mismatch(tmp_path):
    bundle, _review, lock, _installer_sha = build_bundle_with_nemoclaw_installer_review(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    lock_record = next(record for record in manifest["files"] if record["source_path"] == str(lock))
    bundled_lock = bundle / lock_record["bundle_path"]
    payload = json.loads(bundled_lock.read_text(encoding="utf-8"))
    payload["size_bytes"] = 999
    bundled_lock.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, lock_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "NeMoClaw installer review lock JSON size_bytes does not match current_gate installer review summary"
        in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_nemoclaw_installer_review_command_sha_mismatch(
    tmp_path,
):
    bundle, _review, _lock, _installer_sha = build_bundle_with_nemoclaw_installer_review(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operator_plan_path = bundle / "operator_plan.json"
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    bad_sha = "1" * 64
    expected_sha = manifest["current_gate"]["nemoclaw_installer_review"]["sha256"]
    operator_plan["operator_next_steps"]["steps"][0]["commands"][0] = (
        operator_plan["operator_next_steps"]["steps"][0]["commands"][0].replace(
            expected_sha,
            bad_sha,
        )
    )
    manifest["current_gate"]["operator_next_steps"] = operator_plan["operator_next_steps"]
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "operator_plan command 1 --expected-sha256 does not match bundled installer review summary"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_nemoclaw_review_recommended_command_sha_mismatch(
    tmp_path,
):
    bundle, review, _lock, installer_sha = build_bundle_with_nemoclaw_installer_review(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    review_record = next(record for record in manifest["files"] if record["source_path"] == str(review))
    bundled_review = bundle / review_record["bundle_path"]
    payload = json.loads(bundled_review.read_text(encoding="utf-8"))
    payload["recommended_install_command"] = payload["recommended_install_command"].replace(
        installer_sha,
        "1" * 64,
    )
    bundled_review.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, review_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    output = json.loads(result.stdout)
    assert output["integrity_ok"] is False
    assert (
        "NeMoClaw installer review JSON recommended_install_command "
        "--installer-sha256 does not match bundled installer review summary"
        in output["errors"]
    )


def test_verify_release_evidence_bundle_rejects_nemoclaw_review_recommended_command_broad_policy(
    tmp_path,
):
    bundle, review, _lock, _installer_sha = build_bundle_with_nemoclaw_installer_review(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    review_record = next(record for record in manifest["files"] if record["source_path"] == str(review))
    bundled_review = bundle / review_record["bundle_path"]
    payload = json.loads(bundled_review.read_text(encoding="utf-8"))
    payload["recommended_install_command"] = payload["recommended_install_command"].replace(
        "--policy-tier restricted",
        "--policy-tier open",
    )
    bundled_review.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, review_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    output = json.loads(result.stdout)
    assert output["integrity_ok"] is False
    assert (
        "NeMoClaw installer review JSON recommended_install_command "
        "--policy-tier must be restricted"
        in output["errors"]
    )


def test_verify_release_evidence_bundle_rejects_operator_nemoclaw_review_without_lock_json(
    tmp_path,
):
    bundle, _review, lock, _installer_sha = build_bundle_with_nemoclaw_installer_review(tmp_path)
    manifest_path = bundle / "manifest.json"
    operator_plan_path = bundle / "operator_plan.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    operator_plan["operator_next_steps"]["steps"][0]["commands"][0] = (
        operator_plan["operator_next_steps"]["steps"][0]["commands"][0].replace(
            f"--lock-json {lock} ",
            "",
        )
    )
    manifest["current_gate"]["operator_next_steps"] = operator_plan["operator_next_steps"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "operator_plan command 1 invokes NeMoClaw installer review without --lock-json"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_operator_nemoclaw_install_without_lock_json(
    tmp_path,
):
    bundle, _review, lock, _installer_sha = build_bundle_with_nemoclaw_installer_review(tmp_path)
    manifest_path = bundle / "manifest.json"
    operator_plan_path = bundle / "operator_plan.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    operator_plan["operator_next_steps"]["steps"][0]["commands"][1] = (
        operator_plan["operator_next_steps"]["steps"][0]["commands"][1].replace(
            f"--installer-lock-json {lock} ",
            "",
        )
    )
    manifest["current_gate"]["operator_next_steps"] = operator_plan["operator_next_steps"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "operator_plan command 2 installs NeMoClaw without --installer-lock-json" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_current_gate_nemoclaw_review_without_lock_json(
    tmp_path,
):
    bundle, _review, lock, _installer_sha = build_bundle_with_nemoclaw_installer_review(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["current_gate"]["remediation_plan"][0]["commands"][0] = (
        manifest["current_gate"]["remediation_plan"][0]["commands"][0].replace(
            f"--lock-json {lock} ",
            "",
        )
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        (
            "manifest current_gate remediation_plan row 1 command 1 invokes "
            "NeMoClaw installer review without --lock-json"
        )
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_current_gate_nemoclaw_install_without_lock_json(
    tmp_path,
):
    bundle, _review, lock, _installer_sha = build_bundle_with_nemoclaw_installer_review(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["current_gate"]["remediation_plan"][0]["commands"][1] = (
        manifest["current_gate"]["remediation_plan"][0]["commands"][1].replace(
            f"--installer-lock-json {lock} ",
            "",
        )
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        (
            "manifest current_gate remediation_plan row 1 command 2 installs "
            "NeMoClaw without --installer-lock-json"
        )
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_current_gate_nemoclaw_install_sha_mismatch(
    tmp_path,
):
    bundle, _review, _lock, _installer_sha = build_bundle_with_nemoclaw_installer_review(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_sha = manifest["current_gate"]["nemoclaw_installer_review"]["sha256"]
    bad_sha = "2" * 64
    manifest["current_gate"]["remediation_plan"][0]["commands"][1] = (
        manifest["current_gate"]["remediation_plan"][0]["commands"][1].replace(
            expected_sha,
            bad_sha,
        )
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        (
            "manifest current_gate remediation_plan row 1 command 2 --installer-sha256 "
            "does not match bundled installer review summary"
        )
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_missing_external_action_summary_content(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    summary_path = bundle / "summary.md"
    summary_text = summary_path.read_text(encoding="utf-8")
    summary_text = summary_text.replace(
        "## External Action Checklist",
        "## External Actions",
    )
    summary_path.write_text(summary_text, encoding="utf-8")
    refresh_manifest_record_hash(bundle, "summary.md")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert "summary.md missing required section: ## External Action Checklist" in payload["errors"]


def test_verify_release_evidence_bundle_rejects_external_action_summary_count_mismatch(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    checklist = manifest["current_gate"]["external_action_checklist"]
    expected_line = f"External action items: `{checklist['external_action_item_count']}`"
    summary_path = bundle / "summary.md"
    summary_text = summary_path.read_text(encoding="utf-8")
    assert expected_line in summary_text
    summary_path.write_text(
        summary_text.replace(expected_line, "External action items: `0`"),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, "summary.md")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "summary.md missing external action checklist content from current_gate: "
        f"{expected_line}"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_external_action_summary_row_mismatch(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    summary_path = bundle / "summary.md"
    summary_text = summary_path.read_text(encoding="utf-8")
    checklist_start = summary_text.index("| Gate | Status | Required external actions |")
    checklist_lines = summary_text[checklist_start:].splitlines()
    expected_row = next(line for line in checklist_lines if line.startswith("| `"))
    summary_path.write_text(
        summary_text.replace(
            expected_row,
            expected_row.replace("| `", "| `wrong-", 1),
            1,
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, "summary.md")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "summary.md missing external action checklist content from current_gate: "
        f"{expected_row}"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_missing_nemoclaw_post_install_summary_content(tmp_path):
    bundle, _post_install = build_bundle_with_nemoclaw_post_install(tmp_path)
    summary_path = bundle / "summary.md"
    summary_text = summary_path.read_text(encoding="utf-8")
    summary_text = summary_text.replace(
        "## NeMoClaw Post-Install Verification",
        "## NeMoClaw Post Install Verification",
    )
    summary_path.write_text(summary_text, encoding="utf-8")
    refresh_manifest_record_hash(bundle, "summary.md")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "summary.md missing NeMoClaw post-install content: "
        "## NeMoClaw Post-Install Verification" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_missing_nemoclaw_operator_docs_summary_content(tmp_path):
    bundle, _post_install = build_bundle_with_nemoclaw_post_install(tmp_path)
    summary_path = bundle / "summary.md"
    summary_text = summary_path.read_text(encoding="utf-8")
    summary_text = summary_text.replace(
        "## NeMoClaw Operator Docs Verification",
        "## NeMoClaw Operator Documentation",
    )
    summary_path.write_text(summary_text, encoding="utf-8")
    refresh_manifest_record_hash(bundle, "summary.md")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "summary.md missing NeMoClaw operator docs content: "
        "## NeMoClaw Operator Docs Verification" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_tampered_nemoclaw_operator_docs_json(tmp_path):
    bundle, _post_install = build_bundle_with_nemoclaw_post_install(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    record = next(
        record
        for record in manifest["files"]
        if "nemoclaw_operator_docs_verification" in record.get("roles", [])
    )
    docs_json_path = bundle / record["bundle_path"]
    docs_payload = json.loads(docs_json_path.read_text(encoding="utf-8"))
    docs_payload["checks"][0]["ok"] = False
    docs_json_path.write_text(json.dumps(docs_payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "NeMoClaw operator docs verification has ok=true but a check failed: readme_exists"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_nemoclaw_operator_docs_path_mismatch(tmp_path):
    bundle, _post_install = build_bundle_with_nemoclaw_post_install(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    record = next(
        record
        for record in manifest["files"]
        if "nemoclaw_operator_docs_verification" in record.get("roles", [])
    )
    docs_json_path = bundle / record["bundle_path"]
    docs_payload = json.loads(docs_json_path.read_text(encoding="utf-8"))
    docs_payload["path"] = str(bundle / "other_operator_docs.json")
    docs_json_path.write_text(json.dumps(docs_payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "NeMoClaw operator docs verification JSON path does not match runner_evidence"
        in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_nemoclaw_operator_docs_markdown_path_mismatch(
    tmp_path,
):
    bundle, _post_install = build_bundle_with_nemoclaw_post_install(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    record = next(
        record
        for record in manifest["files"]
        if "nemoclaw_operator_docs_verification" in record.get("roles", [])
    )
    docs_json_path = bundle / record["bundle_path"]
    docs_payload = json.loads(docs_json_path.read_text(encoding="utf-8"))
    docs_payload["markdown_path"] = str(bundle / "other_operator_docs.md")
    docs_json_path.write_text(json.dumps(docs_payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "NeMoClaw operator docs verification JSON markdown_path does not match "
        "runner_evidence"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_legacy_nemoclaw_operator_docs_json(
    tmp_path,
):
    bundle, _post_install = build_bundle_with_nemoclaw_post_install(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    record = next(
        record
        for record in manifest["files"]
        if "nemoclaw_operator_docs_verification" in record.get("roles", [])
    )
    docs_json_path = bundle / record["bundle_path"]
    docs_payload = json.loads(docs_json_path.read_text(encoding="utf-8"))
    docs_payload["checks"] = [
        check
        for check in docs_payload["checks"]
        if check.get("name") != "production_readiness_command"
    ]
    docs_json_path.write_text(json.dumps(docs_payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "NeMoClaw operator docs verification missing required check: "
        "production_readiness_command"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_missing_nemoclaw_post_install_summary_forbidden_prefixes(tmp_path):
    bundle, _post_install = build_bundle_with_nemoclaw_post_install(tmp_path)
    summary_path = bundle / "summary.md"
    summary_text = summary_path.read_text(encoding="utf-8")
    summary_text = summary_text.replace("Forbidden prefixes", "Forbidden prefix list")
    summary_path.write_text(summary_text, encoding="utf-8")
    refresh_manifest_record_hash(bundle, "summary.md")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "summary.md missing NeMoClaw post-install content: Forbidden prefixes" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_nemoclaw_post_install_safety_flags(tmp_path):
    bundle, post_install = build_bundle_with_nemoclaw_post_install(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    post_install_record = next(
        record for record in manifest["files"] if record["source_path"] == str(post_install)
    )
    bundled_post_install = bundle / post_install_record["bundle_path"]
    payload = json.loads(bundled_post_install.read_text(encoding="utf-8"))
    payload["will_query_wandb"] = True
    bundled_post_install.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, post_install_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "NeMoClaw post-install verification will_query_wandb must be false" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_nemoclaw_post_install_missing_output(tmp_path):
    bundle, post_install = build_bundle_with_nemoclaw_post_install(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    post_install_record = next(
        record for record in manifest["files"] if record["source_path"] == str(post_install)
    )
    bundled_post_install = bundle / post_install_record["bundle_path"]
    payload = json.loads(bundled_post_install.read_text(encoding="utf-8"))
    payload["outputs"].pop("readiness_json")
    bundled_post_install.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, post_install_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "NeMoClaw post-install verification outputs missing: readiness_json" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_nemoclaw_post_install_missing_output_sha(
    tmp_path,
):
    bundle, post_install = build_bundle_with_nemoclaw_post_install(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    post_install_record = next(
        record for record in manifest["files"] if record["source_path"] == str(post_install)
    )
    bundled_post_install = bundle / post_install_record["bundle_path"]
    payload = json.loads(bundled_post_install.read_text(encoding="utf-8"))
    payload["outputs_sha256"].pop("readiness_json")
    bundled_post_install.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, post_install_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "NeMoClaw post-install verification outputs_sha256 missing or invalid: readiness_json"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_nemoclaw_post_install_output_sha_mismatch(
    tmp_path,
):
    bundle, post_install = build_bundle_with_nemoclaw_post_install(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    post_install_record = next(
        record for record in manifest["files"] if record["source_path"] == str(post_install)
    )
    bundled_post_install = bundle / post_install_record["bundle_path"]
    payload = json.loads(bundled_post_install.read_text(encoding="utf-8"))
    payload["outputs_sha256"]["readiness_json"] = "0" * 64
    payload["steps"][2]["output_json_sha256"] = "0" * 64
    bundled_post_install.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, post_install_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "NeMoClaw post-install verification output readiness_json sha256 does not match outputs_sha256"
        in error
        for error in payload["errors"]
    )
    assert any(
        "NeMoClaw post-install verification step canary_readiness output_json_sha256 does not match bundled output"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_nemoclaw_post_install_missing_command_safety(tmp_path):
    bundle, post_install = build_bundle_with_nemoclaw_post_install(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    post_install_record = next(
        record for record in manifest["files"] if record["source_path"] == str(post_install)
    )
    bundled_post_install = bundle / post_install_record["bundle_path"]
    payload = json.loads(bundled_post_install.read_text(encoding="utf-8"))
    payload.pop("command_safety")
    bundled_post_install.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, post_install_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "NeMoClaw post-install verification command_safety is not an object" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_nemoclaw_post_install_missing_policy_metadata(tmp_path):
    bundle, post_install = build_bundle_with_nemoclaw_post_install(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    post_install_record = next(
        record for record in manifest["files"] if record["source_path"] == str(post_install)
    )
    bundled_post_install = bundle / post_install_record["bundle_path"]
    payload = json.loads(bundled_post_install.read_text(encoding="utf-8"))
    command_safety = payload["command_safety"]
    command_safety["forbidden_tokens"].remove("--wandb")
    command_safety["forbidden_prefixes"].remove("WANDB_")
    command_safety["forbidden_markers"].remove("openrouter")
    command_safety["required_step_tokens"]["adoption_check"].remove("--markdown")
    bundled_post_install.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, post_install_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "NeMoClaw post-install verification command_safety "
        "forbidden_tokens missing required token --wandb" in error
        for error in payload["errors"]
    )
    assert any(
        "NeMoClaw post-install verification command_safety "
        "forbidden_prefixes missing required prefix WANDB_" in error
        for error in payload["errors"]
    )
    assert any(
        "NeMoClaw post-install verification command_safety "
        "forbidden_markers missing required marker openrouter" in error
        for error in payload["errors"]
    )
    assert any(
        "NeMoClaw post-install verification command_safety "
        "required_step_tokens adoption_check missing required token --markdown" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_nemoclaw_post_install_step_payload_mismatch(tmp_path):
    bundle, post_install = build_bundle_with_nemoclaw_post_install(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    post_install_record = next(
        record for record in manifest["files"] if record["source_path"] == str(post_install)
    )
    bundled_post_install = bundle / post_install_record["bundle_path"]
    post_install_payload = json.loads(bundled_post_install.read_text(encoding="utf-8"))
    readiness_source = post_install_payload["outputs"]["readiness_json"]
    readiness_record = next(
        record for record in manifest["files"] if record["source_path"] == readiness_source
    )
    bundled_readiness = bundle / readiness_record["bundle_path"]
    readiness_payload = json.loads(bundled_readiness.read_text(encoding="utf-8"))
    readiness_payload["ok"] = True
    bundled_readiness.write_text(json.dumps(readiness_payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, readiness_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "NeMoClaw post-install verification step canary_readiness payload_ok "
        "does not match output readiness_json.ok" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_nemoclaw_readiness_missing_remote_lookup_check(
    tmp_path,
):
    bundle, post_install = build_bundle_with_nemoclaw_post_install(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    post_install_record = next(
        record for record in manifest["files"] if record["source_path"] == str(post_install)
    )
    bundled_post_install = bundle / post_install_record["bundle_path"]
    post_install_payload = json.loads(bundled_post_install.read_text(encoding="utf-8"))
    readiness_source = post_install_payload["outputs"]["readiness_json"]
    readiness_record = next(
        record for record in manifest["files"] if record["source_path"] == readiness_source
    )
    bundled_readiness = bundle / readiness_record["bundle_path"]
    readiness_payload = json.loads(bundled_readiness.read_text(encoding="utf-8"))
    readiness_payload["checks"] = [
        check
        for check in readiness_payload["checks"]
        if check.get("name") != "agentic SWE denies remote lookup via deny_tool"
    ]
    bundled_readiness.write_text(json.dumps(readiness_payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, readiness_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "NeMoClaw canary readiness missing required remote lookup check: "
        "agentic SWE denies remote lookup via deny_tool"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_nemoclaw_readiness_missing_sandbox_config_check(
    tmp_path,
):
    bundle, post_install = build_bundle_with_nemoclaw_post_install(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    post_install_record = next(
        record for record in manifest["files"] if record["source_path"] == str(post_install)
    )
    post_install_payload = json.loads(
        (bundle / post_install_record["bundle_path"]).read_text(encoding="utf-8")
    )
    readiness_source = post_install_payload["outputs"]["readiness_json"]
    readiness_record = next(
        record for record in manifest["files"] if record["source_path"] == readiness_source
    )
    bundled_readiness = bundle / readiness_record["bundle_path"]
    readiness_payload = json.loads(bundled_readiness.read_text(encoding="utf-8"))
    policy_detail = {
        "sandbox": "nejumi-taiwan",
        "sandbox_found": True,
        "policy_count": 7,
        "policies": [
            "clawhub",
            "managed_inference",
            "npm_registry",
            "nvidia",
            "openclaw_api",
            "openclaw_docs",
            "wandb-weave",
        ],
        "policy_configured": True,
        "summary_policy_count": 0,
        "summary_policies": [],
        "detailed_status_network_policy_count": 7,
        "detailed_status_network_policies": [
            "clawhub",
            "managed_inference",
            "npm_registry",
            "nvidia",
            "openclaw_api",
            "openclaw_docs",
            "wandb-weave",
        ],
        "allowed_runtime_network_policies": [
            "clawhub",
            "managed_inference",
            "npm_registry",
            "nvidia",
            "openclaw_api",
            "openclaw_docs",
            "wandb-weave",
        ],
        "runtime_network_policy_allowlist_ok": True,
        "unknown_runtime_network_policies": [],
        "wandb_weave_policy_present": True,
        "non_wandb_network_policies": [
            "clawhub",
            "managed_inference",
            "npm_registry",
            "nvidia",
            "openclaw_api",
            "openclaw_docs",
        ],
    }
    readiness_payload["ok"] = True
    readiness_payload["checks"].extend(
        [
            {
                "name": "NeMoClaw sandbox runtime policy is introspectable: nejumi-taiwan",
                "ok": True,
                "detail": json.dumps(policy_detail),
            },
            {
                "name": "NeMoClaw W&B/Weave runtime policy is present: nejumi-taiwan",
                "ok": True,
                "detail": json.dumps(policy_detail),
            },
            {
                "name": "NeMoClaw runtime network policies are allowlisted: nejumi-taiwan",
                "ok": True,
                "detail": json.dumps(policy_detail),
            },
            {
                "name": (
                    "NeMoClaw sandbox OpenClaw config is readable: "
                    "/sandbox/.openclaw/openclaw.json"
                ),
                "ok": True,
                "detail": "bytes=7465",
            },
            {
                "name": (
                    "NeMoClaw sandbox OpenClaw model is registered: "
                    "openai-direct/gpt-4.1-mini-2025-04-14"
                ),
                "ok": True,
            },
            {
                "name": "NeMoClaw sandbox OpenClaw Weave plugin is enabled",
                "ok": True,
            },
        ]
    )
    bundled_readiness.write_text(json.dumps(readiness_payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, readiness_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "NeMoClaw canary readiness missing sandbox OpenClaw config check: "
        "NeMoClaw sandbox OpenClaw openai-direct provider exists"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_nemoclaw_post_install_missing_payload_contract(
    tmp_path,
):
    bundle, post_install = build_bundle_with_nemoclaw_post_install(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    post_install_record = next(
        record for record in manifest["files"] if record["source_path"] == str(post_install)
    )
    bundled_post_install = bundle / post_install_record["bundle_path"]
    payload = json.loads(bundled_post_install.read_text(encoding="utf-8"))
    payload["steps"][0].pop("payload_contract_ok")
    payload["steps"][0].pop("payload_contract_errors")
    bundled_post_install.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, post_install_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "NeMoClaw post-install verification step setup_check "
        "payload_contract_ok is not a bool" in error
        for error in payload["errors"]
    )
    assert any(
        "NeMoClaw post-install verification step setup_check "
        "payload_contract_errors is not a string list" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_nemoclaw_post_install_forbidden_command(tmp_path):
    bundle, post_install = build_bundle_with_nemoclaw_post_install(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    post_install_record = next(
        record for record in manifest["files"] if record["source_path"] == str(post_install)
    )
    bundled_post_install = bundle / post_install_record["bundle_path"]
    payload = json.loads(bundled_post_install.read_text(encoding="utf-8"))
    payload["steps"][0]["command"].append("--install")
    bundled_post_install.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, post_install_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "NeMoClaw post-install verification step setup_check "
        "contains forbidden command token --install" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_nemoclaw_post_install_wandb_prefixed_command(tmp_path):
    bundle, post_install = build_bundle_with_nemoclaw_post_install(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    post_install_record = next(
        record for record in manifest["files"] if record["source_path"] == str(post_install)
    )
    bundled_post_install = bundle / post_install_record["bundle_path"]
    payload = json.loads(bundled_post_install.read_text(encoding="utf-8"))
    payload["steps"][2]["command"].extend(["--wandb-project=taiwan", "WANDB_API_KEY=test"])
    bundled_post_install.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, post_install_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "NeMoClaw post-install verification step canary_readiness "
        "contains forbidden command token --wandb-project=taiwan" in error
        for error in payload["errors"]
    )
    assert any(
        "NeMoClaw post-install verification step canary_readiness "
        "contains forbidden command token WANDB_API_KEY=test" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_nemoclaw_post_install_missing_command_token(tmp_path):
    bundle, post_install = build_bundle_with_nemoclaw_post_install(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    post_install_record = next(
        record for record in manifest["files"] if record["source_path"] == str(post_install)
    )
    bundled_post_install = bundle / post_install_record["bundle_path"]
    payload = json.loads(bundled_post_install.read_text(encoding="utf-8"))
    protocol = next(step for step in payload["steps"] if step["name"] == "protocol_preflight")
    protocol["command"] = [
        token for token in protocol["command"] if token != "preflight"
    ]
    bundled_post_install.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, post_install_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "NeMoClaw post-install verification step protocol_preflight "
        "command missing required token preflight" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_accepts_weave_agents_completion_proof(tmp_path):
    bundle, _completion = build_bundle_with_weave_agents_completion(tmp_path)

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is True
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert any(
        "gate:one_model_full_canary:weave_agents_completion_sync_dry_run"
        in record.get("roles", [])
        for record in manifest["files"]
    )


def production_readiness_record(manifest: dict) -> dict:
    return next(
        record
        for record in manifest["files"]
        if "production_readiness_report" in record.get("roles", [])
    )


def bundled_one_model_gate(bundle: Path, report_record: dict) -> tuple[Path, dict]:
    bundled_report = bundle / report_record["bundle_path"]
    payload = json.loads(bundled_report.read_text(encoding="utf-8"))
    gate = next(
        row for row in payload["gates"] if row.get("name") == "one_model_full_canary"
    )
    return bundled_report, gate


def test_verify_release_evidence_bundle_accepts_one_model_agentic_weave_contract(tmp_path):
    bundle, _completion = build_bundle_with_weave_agents_completion(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    report_record = production_readiness_record(manifest)
    bundled_report, gate = bundled_one_model_gate(bundle, report_record)
    entry = gate["records"][0]["weave_agents_completion_entries"][0]
    gate["ok"] = True
    gate["status"] = "passed"
    gate["required_wandb_benchmarks"] = ["agentic_math"]
    gate["weave_agents_completion_required"] = True
    gate["completed_weave_agents_completion_phases"] = {"agentic": [entry]}
    gate["missing_required_weave_agents_completion_phases"] = []
    payload = json.loads(bundled_report.read_text(encoding="utf-8"))
    for index, row in enumerate(payload["gates"]):
        if row.get("name") == "one_model_full_canary":
            payload["gates"][index] = gate
    bundled_report.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, report_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is True


def test_verify_release_evidence_bundle_rejects_one_model_agentic_without_weave_required_flag(tmp_path):
    bundle, _completion = build_bundle_with_weave_agents_completion(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    report_record = production_readiness_record(manifest)
    bundled_report, gate = bundled_one_model_gate(bundle, report_record)
    gate["required_wandb_benchmarks"] = ["agentic_math"]
    gate["weave_agents_completion_required"] = False
    gate["completed_weave_agents_completion_phases"] = {}
    gate["missing_required_weave_agents_completion_phases"] = []
    payload = json.loads(bundled_report.read_text(encoding="utf-8"))
    for index, row in enumerate(payload["gates"]):
        if row.get("name") == "one_model_full_canary":
            payload["gates"][index] = gate
    bundled_report.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, report_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "one_model_full_canary weave_agents_completion_required must be true "
        "when agentic W&B benchmarks are required" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_passed_one_model_without_completed_weave_phase(tmp_path):
    bundle, _completion = build_bundle_with_weave_agents_completion(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    report_record = production_readiness_record(manifest)
    bundled_report, gate = bundled_one_model_gate(bundle, report_record)
    gate["ok"] = True
    gate["status"] = "passed"
    gate["required_wandb_benchmarks"] = ["agentic_swe"]
    gate["weave_agents_completion_required"] = True
    gate["completed_weave_agents_completion_phases"] = {}
    gate["missing_required_weave_agents_completion_phases"] = []
    payload = json.loads(bundled_report.read_text(encoding="utf-8"))
    for index, row in enumerate(payload["gates"]):
        if row.get("name") == "one_model_full_canary":
            payload["gates"][index] = gate
    bundled_report.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, report_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "one_model_full_canary passed but no completed full/agentic "
        "Weave Agents phase is recorded" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_weave_agents_sync_dry_run_mismatch(tmp_path):
    bundle, _completion = build_bundle_with_weave_agents_completion(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    role = "gate:one_model_full_canary:weave_agents_completion_sync_dry_run"
    report_record = next(
        record for record in manifest["files"] if role in record.get("roles", [])
    )
    bundled_report = bundle / report_record["bundle_path"]
    payload = json.loads(bundled_report.read_text(encoding="utf-8"))
    payload["dry_run"] = False
    payload["in_place"] = True
    payload["review_path"] = "outputs/taiwan_full_eval/other_paid_run_review.json"
    payload["entries"][0]["run_id"] = "other-run"
    payload["entries"][0]["query_source_kind"] = "manual_json"
    payload["changes"][0]["run_id"] = "other-run"
    bundled_report.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, report_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "Weave Agents completion entry row #1.1 sync dry-run report dry_run must be true"
        in error
        for error in payload["errors"]
    )
    assert any(
        "Weave Agents completion entry row #1.1 sync dry-run report "
        "entries must include the claimed Weave completion" in error
        for error in payload["errors"]
    )
    assert any(
        "Weave Agents completion entry row #1.1 sync dry-run report "
        "changes must include the claimed Weave completion" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_weave_agents_sync_dry_run_missing_from_review_json(tmp_path):
    bundle, _completion = build_bundle_with_weave_agents_completion(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    review_record = next(
        record
        for record in manifest["files"]
        if record.get("source_path", "").endswith("canary_agentic_paid_run_review.json")
    )
    bundled_review = bundle / review_record["bundle_path"]
    review_payload = json.loads(bundled_review.read_text(encoding="utf-8"))
    review_payload["runs"][0]["weave_agents_completion"].pop(
        "sync_dry_run_report_json"
    )
    bundled_review.write_text(json.dumps(review_payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, review_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "Weave Agents completion entry row #1.1 sync_dry_run_report_json "
        "is not present in bundled paid review JSON" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_weave_agents_sync_dry_run_sha_missing_from_review_json(tmp_path):
    bundle, _completion = build_bundle_with_weave_agents_completion(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    review_record = next(
        record
        for record in manifest["files"]
        if record.get("source_path", "").endswith("canary_agentic_paid_run_review.json")
    )
    bundled_review = bundle / review_record["bundle_path"]
    review_payload = json.loads(bundled_review.read_text(encoding="utf-8"))
    review_payload["runs"][0]["weave_agents_completion"].pop(
        "sync_dry_run_source_review_sha256"
    )
    bundled_review.write_text(json.dumps(review_payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, review_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "Weave Agents completion entry row #1.1 sync_dry_run_source_review_sha256 "
        "is not present in bundled paid review JSON" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_weave_agents_sync_dry_run_sha_mismatch(tmp_path):
    bundle, _completion = build_bundle_with_weave_agents_completion(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    role = "gate:one_model_full_canary:weave_agents_completion_sync_dry_run"
    report_record = next(
        record for record in manifest["files"] if role in record.get("roles", [])
    )
    bundled_report = bundle / report_record["bundle_path"]
    payload = json.loads(bundled_report.read_text(encoding="utf-8"))
    payload["source_review_sha256"] = "0" * 64
    bundled_report.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, report_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "Weave Agents completion entry row #1.1 sync dry-run report "
        "source_review_sha256 does not match review entry" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_weave_agents_sync_source_review_sha_mismatch(tmp_path):
    bundle, _completion = build_bundle_with_weave_agents_completion(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    source_review_record = next(
        record
        for record in manifest["files"]
        if record.get("source_path", "").endswith(
            "canary_agentic_paid_run_review.before_weave_sync.json"
        )
    )
    bundled_source_review = bundle / source_review_record["bundle_path"]
    source_payload = json.loads(bundled_source_review.read_text(encoding="utf-8"))
    source_payload["provider_bill_reference"] = "changed-after-sync"
    bundled_source_review.write_text(json.dumps(source_payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, source_review_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "Weave Agents completion entry row #1.1 sync dry-run report "
        "source_review_sha256 does not match bundled source review JSON" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_weave_agents_completion_missing_from_review_json(tmp_path):
    bundle, _completion = build_bundle_with_weave_agents_completion(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    review_record = next(
        record
        for record in manifest["files"]
        if record.get("source_path", "").endswith("canary_agentic_paid_run_review.json")
    )
    bundled_review = bundle / review_record["bundle_path"]
    review_payload = json.loads(bundled_review.read_text(encoding="utf-8"))
    review_payload["runs"][0]["weave_agents_completion"] = {}
    bundled_review.write_text(json.dumps(review_payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, review_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "Weave Agents completion entry row #1.1 is not present in bundled paid review JSON"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_weave_agents_review_json_status_mismatch(tmp_path):
    bundle, _completion = build_bundle_with_weave_agents_completion(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    review_record = next(
        record
        for record in manifest["files"]
        if record.get("source_path", "").endswith("canary_agentic_paid_run_review.json")
    )
    bundled_review = bundle / review_record["bundle_path"]
    review_payload = json.loads(bundled_review.read_text(encoding="utf-8"))
    review_payload["status"] = "prepared"
    bundled_review.write_text(json.dumps(review_payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, review_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "Weave Agents completion row #1 paid review status does not match bundled review JSON"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_weave_agents_completion_without_run_scope(tmp_path):
    bundle, completion = build_bundle_with_weave_agents_completion(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    completion_record = next(
        record for record in manifest["files"] if record["source_path"] == str(completion)
    )
    bundled_completion = bundle / completion_record["bundle_path"]
    payload = json.loads(bundled_completion.read_text(encoding="utf-8"))
    payload["required_evidence"]["conversation_id_contains"] = ""
    bundled_completion.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, completion_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "Weave Agents completion proof" in error
        and "conversation scope does not include run_id run-1" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_weave_agents_completion_missing_query_source(tmp_path):
    bundle, completion = build_bundle_with_weave_agents_completion(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    completion_record = next(
        record for record in manifest["files"] if record["source_path"] == str(completion)
    )
    bundled_completion = bundle / completion_record["bundle_path"]
    payload = json.loads(bundled_completion.read_text(encoding="utf-8"))
    payload.pop("query_source")
    bundled_completion.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, completion_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "Weave Agents completion proof" in error
        and "query_source is not an object" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_weave_agents_completion_query_source_mismatch(tmp_path):
    bundle, completion = build_bundle_with_weave_agents_completion(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    completion_record = next(
        record for record in manifest["files"] if record["source_path"] == str(completion)
    )
    bundled_completion = bundle / completion_record["bundle_path"]
    payload = json.loads(bundled_completion.read_text(encoding="utf-8"))
    payload["query_source"]["kind"] = "manual_json"
    payload["query_source"]["conversation_id_contains"] = "other-run"
    payload["query_source"]["latest_trace_span_count"] = 99
    bundled_completion.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, completion_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "Weave Agents completion proof" in error
        and "query_source.kind mismatch" in error
        for error in payload["errors"]
    )
    assert any(
        "query_source.conversation_id_contains does not match required_evidence" in error
        for error in payload["errors"]
    )
    assert any(
        "query_source.latest_trace_span_count does not match latest_trace_spans_chronological"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_failing_weave_agents_check(tmp_path):
    bundle, completion = build_bundle_with_weave_agents_completion(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    completion_record = next(
        record for record in manifest["files"] if record["source_path"] == str(completion)
    )
    bundled_completion = bundle / completion_record["bundle_path"]
    payload = json.loads(bundled_completion.read_text(encoding="utf-8"))
    payload["checks"][0]["ok"] = False
    bundled_completion.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, completion_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "Weave Agents completion proof" in error
        and "checks contains failing check: agent_present" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_missing_final_answer_order_check(tmp_path):
    bundle, completion = build_bundle_with_weave_agents_completion(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    completion_record = next(
        record for record in manifest["files"] if record["source_path"] == str(completion)
    )
    bundled_completion = bundle / completion_record["bundle_path"]
    payload = json.loads(bundled_completion.read_text(encoding="utf-8"))
    payload["checks"] = [
        check
        for check in payload["checks"]
        if check.get("name") != "trace_final_answer_order"
    ]
    bundled_completion.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, completion_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "checks missing required check: trace_final_answer_order" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_missing_timestamp_quality_check(tmp_path):
    bundle, completion = build_bundle_with_weave_agents_completion(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    completion_record = next(
        record for record in manifest["files"] if record["source_path"] == str(completion)
    )
    bundled_completion = bundle / completion_record["bundle_path"]
    payload = json.loads(bundled_completion.read_text(encoding="utf-8"))
    payload["checks"] = [
        check
        for check in payload["checks"]
        if check.get("name") != "trace_timestamp_quality"
    ]
    bundled_completion.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, completion_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "checks missing required check: trace_timestamp_quality" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_weave_agents_completion_missing_request_model_evidence(tmp_path):
    bundle, completion = build_bundle_with_weave_agents_completion(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    completion_record = next(
        record for record in manifest["files"] if record["source_path"] == str(completion)
    )
    bundled_completion = bundle / completion_record["bundle_path"]
    payload = json.loads(bundled_completion.read_text(encoding="utf-8"))
    payload["required_evidence"].pop("expected_request_models")
    payload["checks"] = [
        check for check in payload["checks"] if check.get("name") != "request_model"
    ]
    payload["content_capture_health"].pop("request_model_count")
    for span in payload["latest_trace_spans_chronological"]:
        span.pop("request_model", None)
    bundled_completion.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, completion_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "Weave Agents completion proof" in error
        and "expected_request_models" in error
        for error in payload["errors"]
    )
    assert any(
        "checks missing required check: request_model" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_missing_final_answer_order_requirement(tmp_path):
    bundle, completion = build_bundle_with_weave_agents_completion(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    completion_record = next(
        record for record in manifest["files"] if record["source_path"] == str(completion)
    )
    bundled_completion = bundle / completion_record["bundle_path"]
    payload = json.loads(bundled_completion.read_text(encoding="utf-8"))
    payload["required_evidence"].pop("trace_final_answer_order_required")
    bundled_completion.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, completion_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "required_evidence.trace_final_answer_order_required is not true" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_missing_timestamp_quality_requirement(tmp_path):
    bundle, completion = build_bundle_with_weave_agents_completion(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    completion_record = next(
        record for record in manifest["files"] if record["source_path"] == str(completion)
    )
    bundled_completion = bundle / completion_record["bundle_path"]
    payload = json.loads(bundled_completion.read_text(encoding="utf-8"))
    payload["required_evidence"].pop("trace_timestamp_quality_required")
    bundled_completion.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, completion_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "required_evidence.trace_timestamp_quality_required is not true" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_missing_input_message_requirement(tmp_path):
    bundle, completion = build_bundle_with_weave_agents_completion(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    completion_record = next(
        record for record in manifest["files"] if record["source_path"] == str(completion)
    )
    bundled_completion = bundle / completion_record["bundle_path"]
    payload = json.loads(bundled_completion.read_text(encoding="utf-8"))
    payload["required_evidence"].pop("input_message_required")
    bundled_completion.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, completion_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "required_evidence.input_message_required is not true" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_missing_required_text_capture(tmp_path):
    bundle, completion = build_bundle_with_weave_agents_completion(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    completion_record = next(
        record for record in manifest["files"] if record["source_path"] == str(completion)
    )
    bundled_completion = bundle / completion_record["bundle_path"]
    payload = json.loads(bundled_completion.read_text(encoding="utf-8"))
    payload["required_evidence"]["required_texts"] = [
        "CANARY_ID",
        "CANARY_RESULT CANARY_ID 91",
    ]
    payload["content_capture_health"]["required_text_count"] = 2
    bundled_completion.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, completion_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "checks missing required check: required_text_capture" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_required_text_count_below_required_texts(tmp_path):
    bundle, completion = build_bundle_with_weave_agents_completion(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    completion_record = next(
        record for record in manifest["files"] if record["source_path"] == str(completion)
    )
    bundled_completion = bundle / completion_record["bundle_path"]
    payload = json.loads(bundled_completion.read_text(encoding="utf-8"))
    payload["required_evidence"]["required_texts"] = [
        "CANARY_ID",
        "CANARY_RESULT CANARY_ID 91",
    ]
    payload["checks"].append({"name": "required_text_capture", "ok": True})
    payload["content_capture_health"]["required_text_count"] = 1
    bundled_completion.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, completion_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "content_capture_health.required_text_count below required minimum" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_weave_agents_trace_order_payload(tmp_path):
    bundle, completion = build_bundle_with_weave_agents_completion(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    completion_record = next(
        record for record in manifest["files"] if record["source_path"] == str(completion)
    )
    bundled_completion = bundle / completion_record["bundle_path"]
    payload = json.loads(bundled_completion.read_text(encoding="utf-8"))
    payload["latest_trace_spans_chronological"] = list(
        reversed(payload["latest_trace_spans_chronological"])
    )
    bundled_completion.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, completion_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "latest_trace_spans_chronological is not sorted by span time" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_invalid_weave_agents_timestamp_payload(tmp_path):
    bundle, completion = build_bundle_with_weave_agents_completion(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    completion_record = next(
        record for record in manifest["files"] if record["source_path"] == str(completion)
    )
    bundled_completion = bundle / completion_record["bundle_path"]
    payload = json.loads(bundled_completion.read_text(encoding="utf-8"))
    payload["latest_trace_spans_chronological"][1]["started_at"] = ""
    payload["content_capture_health"]["spans_with_valid_timestamps"] = 1
    payload["content_capture_health"]["spans_with_invalid_timestamps"] = 1
    bundled_completion.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, completion_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "latest_trace_spans_chronological[2] has missing or invalid timestamps" in error
        for error in payload["errors"]
    )
    assert any(
        "content_capture_health.spans_with_valid_timestamps does not match span count"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_weave_agents_tool_before_message(tmp_path):
    bundle, completion = build_bundle_with_weave_agents_completion(tmp_path)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    completion_record = next(
        record for record in manifest["files"] if record["source_path"] == str(completion)
    )
    bundled_completion = bundle / completion_record["bundle_path"]
    payload = json.loads(bundled_completion.read_text(encoding="utf-8"))
    spans = payload["latest_trace_spans_chronological"]
    spans[1]["started_at"] = "2026-06-27T23:59:59Z"
    spans[1]["ended_at"] = "2026-06-28T00:00:00Z"
    payload["latest_trace_spans_chronological"] = [spans[1], spans[0]]
    bundled_completion.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, completion_record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "tool span starts before the first message span" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_accepts_operator_command_scripts(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is True


def test_verify_release_evidence_bundle_rejects_operator_step_semantic_mismatch(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    operator_plan_path = bundle / "operator_plan.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    operator_plan["operator_next_steps"]["steps"][0]["requires_wandb_access"] = False
    operator_plan["operator_next_steps"]["wandb_access_step_count"] = 0
    manifest["current_gate"]["operator_next_steps"] = operator_plan["operator_next_steps"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "operator_plan step 1 requires_wandb_access does not match command semantics"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_operator_placeholder_mismatch(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    operator_plan_path = bundle / "operator_plan.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    step = operator_plan["operator_next_steps"]["steps"][0]
    assert step["unresolved_placeholder_tokens"] == ["PHASE", "RUN_ID", "MODEL_SLUG"]
    assert step["command_template_count"] == 1
    assert step["ready_to_execute_without_placeholder"] is False
    step["unresolved_placeholder_tokens"] = []
    step["command_template_count"] = 0
    operator_plan["operator_next_steps"]["unresolved_placeholder_tokens"] = []
    operator_plan["operator_next_steps"]["command_template_step_count"] = 0
    manifest["current_gate"]["operator_next_steps"] = operator_plan["operator_next_steps"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "operator_plan step 1 unresolved_placeholder_tokens does not match command/evidence placeholders"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_operator_command_count_mismatch(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    operator_plan_path = bundle / "operator_plan.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    step = operator_plan["operator_next_steps"]["steps"][0]
    assert step["command_count"] == len(step["commands"])
    step["command_count"] = 0
    manifest["current_gate"]["operator_next_steps"] = operator_plan["operator_next_steps"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "operator_plan step 1 command_count does not match commands" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_operator_evidence_path_count_mismatch(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    operator_plan_path = bundle / "operator_plan.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    step = operator_plan["operator_next_steps"]["steps"][0]
    assert step["evidence_path_count"] == len(step["evidence_to_produce"])
    step["evidence_path_count"] = len(step["evidence_to_produce"]) + 1
    manifest["current_gate"]["operator_next_steps"] = operator_plan["operator_next_steps"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "operator_plan step 1 evidence_path_count does not match evidence_to_produce"
        in error
        for error in payload["errors"]
    )


def test_operator_command_output_paths_include_nemoclaw_install_operation_logs():
    module = load_verify_module()

    outputs = module.operator_command_output_paths(
        "scripts/setup/install_nemoclaw.sh --install --onboard "
        "--json temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.json"
    )

    assert outputs == [
        "temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.json",
        "temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.install.log",
        "temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.onboard.log",
    ]


def test_verify_release_evidence_bundle_rejects_missing_nemoclaw_operator_log_outputs(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    operator_plan_path = bundle / "operator_plan.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    step = operator_plan["operator_next_steps"]["steps"][0]
    step["commands"] = [
        "scripts/setup/install_nemoclaw.sh --install --onboard "
        "--json temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.json"
    ]
    step["evidence_to_produce"] = [
        "temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.json"
    ]
    step["command_count"] = 1
    step["evidence_path_count"] = 1
    step["requires_third_party_acceptance"] = False
    step["requires_nemoclaw_install"] = True
    step["unresolved_placeholder_tokens"] = ["YYYYMMDDTHHMM"]
    step["command_template_count"] = 1
    step["evidence_template_count"] = 1
    step["ready_to_execute_without_placeholder"] = False
    manifest["current_gate"]["operator_next_steps"] = operator_plan["operator_next_steps"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "operator_plan step 1 evidence_to_produce does not match command outputs"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_missing_batch_runner_outputs(tmp_path):
    bundle = build_bundle_with_operator_batch_command(tmp_path)
    manifest_path = bundle / "manifest.json"
    operator_plan_path = bundle / "operator_plan.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    operator_plan["operator_next_steps"]["steps"][0]["evidence_to_produce"] = [
        "outputs/taiwan_full_eval/canary_agentic_execution_plan.json",
        "outputs/taiwan_full_eval/canary_agentic_paid_run_review.json",
    ]
    manifest["current_gate"]["operator_next_steps"] = operator_plan["operator_next_steps"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "operator_plan step 1 evidence_to_produce does not match command outputs"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_missing_weave_content_canary_outputs(tmp_path):
    bundle = build_bundle_with_operator_weave_content_canary_command(tmp_path)
    manifest_path = bundle / "manifest.json"
    operator_plan_path = bundle / "operator_plan.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    operator_plan["operator_next_steps"]["steps"][0]["evidence_to_produce"] = [
        "outputs/taiwan_full_eval/openai_canary_readiness_weave_content.json"
    ]
    manifest["current_gate"]["operator_next_steps"] = operator_plan["operator_next_steps"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "operator_plan step 1 evidence_to_produce does not match command outputs"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_requires_operator_command_scripts(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [
        record
        for record in manifest["files"]
        if record.get("source_path") != "scripts/tools/run_taiwan_release_gate.py"
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert "operator_plan command script is not bundled: scripts/tools/run_taiwan_release_gate.py" in payload["errors"]


def test_verify_release_evidence_bundle_requires_current_gate_remediation_command_scripts(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["current_gate"]["remediation_plan"][0]["commands"] = [
        "uv run python scripts/tools/check_taiwan_canary_readiness.py --json temp/check.json"
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "current_gate remediation_plan command script is not bundled: "
        "scripts/tools/check_taiwan_canary_readiness.py"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_canary_readiness_script_missing_deny_policy_contract(
    tmp_path,
):
    bundle = build_bundle_with_operator_weave_content_canary_command(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/check_taiwan_canary_readiness.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            "AGENTIC_REQUIRED_DENIED_TOOLS",
            "REMOTE_DENIED_TOOLS_REMOVED",
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "NeMoClaw canary readiness script missing source contract "
        "remote lookup deny tools constant: "
        "scripts/tools/check_taiwan_canary_readiness.py: AGENTIC_REQUIRED_DENIED_TOOLS"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_canary_readiness_script_missing_runtime_policy_contract(
    tmp_path,
):
    bundle = build_bundle_with_operator_weave_content_canary_command(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/check_taiwan_canary_readiness.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace("def _run_json_status(", "def _removed_json_status("),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "NeMoClaw canary readiness script missing source contract "
        "NeMoClaw status JSON introspection: "
        "scripts/tools/check_taiwan_canary_readiness.py: def _run_json_status("
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_canary_readiness_script_missing_sandbox_config_contract(
    tmp_path,
):
    bundle = build_bundle_with_operator_weave_content_canary_command(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/check_taiwan_canary_readiness.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            "def openclaw_config_data_checks(",
            "def removed_openclaw_config_data_checks(",
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "NeMoClaw canary readiness script missing source contract "
        "NeMoClaw sandbox OpenClaw config reusable parser: "
        "scripts/tools/check_taiwan_canary_readiness.py: "
        "def openclaw_config_data_checks("
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_adoption_script_missing_runtime_policy_contract(
    tmp_path,
):
    bundle = build_bundle_with_operator_weave_content_canary_command(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = REPO_ROOT / "scripts" / "tools" / "check_taiwan_nemoclaw_adoption.py"
    bundle_path = Path("scripts/tools/check_taiwan_nemoclaw_adoption.py")
    target = bundle / bundle_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    manifest["files"].append(
        {
            "source_path": "scripts/tools/check_taiwan_nemoclaw_adoption.py",
            "roles": ["operator_plan:command_script"],
            "exists": True,
            "bundle_path": str(bundle_path),
            "size_bytes": target.stat().st_size,
            "sha256": sha256(target),
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/check_taiwan_nemoclaw_adoption.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            "def runtime_wandb_weave_policy(",
            "def removed_runtime_wandb_weave_policy(",
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "NeMoClaw adoption script missing source contract "
        "W&B/Weave runtime policy criterion: "
        "scripts/tools/check_taiwan_nemoclaw_adoption.py: "
        "def runtime_wandb_weave_policy("
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_adoption_script_missing_policy_allowlist_contract(
    tmp_path,
):
    bundle = build_bundle_with_operator_weave_content_canary_command(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = REPO_ROOT / "scripts" / "tools" / "check_taiwan_nemoclaw_adoption.py"
    bundle_path = Path("scripts/tools/check_taiwan_nemoclaw_adoption.py")
    target = bundle / bundle_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    manifest["files"].append(
        {
            "source_path": "scripts/tools/check_taiwan_nemoclaw_adoption.py",
            "roles": ["operator_plan:command_script"],
            "exists": True,
            "bundle_path": str(bundle_path),
            "size_bytes": target.stat().st_size,
            "sha256": sha256(target),
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/check_taiwan_nemoclaw_adoption.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            "def runtime_network_policy_allowlist(",
            "def removed_runtime_network_policy_allowlist(",
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "NeMoClaw adoption script missing source contract "
        "runtime network policy allowlist criterion: "
        "scripts/tools/check_taiwan_nemoclaw_adoption.py: "
        "def runtime_network_policy_allowlist("
    ) in payload["errors"]


def test_verify_release_evidence_bundle_accepts_existing_results_relog_command_scripts(tmp_path):
    bundle = build_bundle_with_existing_results_relog_command_scripts(tmp_path)

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is True


def test_verify_release_evidence_bundle_requires_existing_results_relog_command_script(tmp_path):
    bundle = build_bundle_with_existing_results_relog_command_scripts(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [
        record
        for record in manifest["files"]
        if record.get("source_path") != "scripts/tools/log_agentic_math_results_to_wandb.py"
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "existing_results relog command script is not bundled: "
        "scripts/tools/log_agentic_math_results_to_wandb.py"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_requires_existing_results_relog_dependency_script(tmp_path):
    bundle = build_bundle_with_existing_results_relog_command_scripts(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [
        record
        for record in manifest["files"]
        if record.get("source_path") != "scripts/tools/relog_wandb_approval.py"
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "existing_results relog dependency script is not bundled: "
        "scripts/tools/relog_wandb_approval.py"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_requires_agentic_runner_scripts(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [
        record
        for record in manifest["files"]
        if record.get("source_path") != "scripts/tools/run_openclaw_agent_protocol.py"
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "agentic runner script is not bundled: "
        "scripts/tools/run_openclaw_agent_protocol.py"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_agentic_runner_missing_source_contract(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/run_openclaw_agent_protocol.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace("def conversation_order_status(", "def removed_conversation_order_status("),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "agentic runner script missing source contract conversation-order validator: "
        "scripts/tools/run_openclaw_agent_protocol.py: def conversation_order_status("
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_agentic_runner_missing_nemoclaw_session_audit_contract(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/run_openclaw_agent_protocol.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace("def nemoclaw_session_audit_status(", "def removed_nemoclaw_session_audit_status("),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "agentic runner script missing source contract NeMoClaw session audit validator: "
        "scripts/tools/run_openclaw_agent_protocol.py: def nemoclaw_session_audit_status("
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_agentic_runner_missing_session_scope_contract(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/run_agentic_math_openclaw.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace("def resolve_session_prefix(", "def removed_session_prefix("),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "agentic runner script missing source contract W&B session-scope resolver: "
        "scripts/tools/run_agentic_math_openclaw.py: def resolve_session_prefix("
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_wandb_completion_verifier_missing_output_column_contract(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/verify_taiwan_wandb_completion.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            "AGENTIC_MATH_OUTPUT_TABLE_REQUIRED_COLUMNS = (",
            "AGENTIC_MATH_OUTPUT_TABLE_COLUMNS = (",
            1,
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "agentic runner script missing source contract "
        "Agentic Math W&B output required columns: "
        "scripts/tools/verify_taiwan_wandb_completion.py: "
        "AGENTIC_MATH_OUTPUT_TABLE_REQUIRED_COLUMNS = ("
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_agentic_evaluator_missing_session_prefix_contract(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/evaluator/agentic_math.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            'command.extend(["--session-prefix", str(session_prefix)])',
            'command.extend(["--removed-session-prefix", str(session_prefix)])',
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "agentic runner script missing source contract Agentic Math session-prefix pass-through: "
        'scripts/evaluator/agentic_math.py: command.extend(["--session-prefix", str(session_prefix)])'
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_agentic_math_missing_remote_lookup_disable_contract(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/run_agentic_math_openclaw.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            'tools["toolSearch"] = False',
            'tools["toolSearch"] = True',
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "agentic runner script missing source contract OpenClaw tool search disabled: "
        'scripts/tools/run_agentic_math_openclaw.py: tools["toolSearch"] = False'
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_math_relog_missing_observability_validation_call(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/log_agentic_math_results_to_wandb.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            "validate_observability_acceptance(rows)",
            "observability_acceptance_issues(rows)",
            1,
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "agentic runner script missing source contract "
        "Agentic Math relog observability validation call: "
        "scripts/tools/log_agentic_math_results_to_wandb.py: "
        "validate_observability_acceptance(rows)"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_swe_relog_missing_observability_validation_call(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/log_agentic_swe_results_to_wandb.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            "validate_observability_acceptance(rows_for_audit)",
            "observability_acceptance_issues(rows_for_audit)",
            1,
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "agentic runner script missing source contract "
        "SWE relog observability validation call: "
        "scripts/tools/log_agentic_swe_results_to_wandb.py: "
        "validate_observability_acceptance(rows_for_audit)"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_math_relog_missing_weave_output_column(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/log_agentic_math_results_to_wandb.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            '    "weave_sidecar_ok",\n',
            "",
            1,
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "agentic runner script missing source contract "
        "Agentic Math relog output Weave-sidecar column: "
        "scripts/tools/log_agentic_math_results_to_wandb.py: "
        '"weave_sidecar_ok",'
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_swe_relog_missing_weave_output_column(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/log_agentic_swe_results_to_wandb.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            '                "weave_sidecar_ok": patch_row.get("weave_sidecar_ok"),\n',
            "",
            1,
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "agentic runner script missing source contract "
        "SWE relog output Weave-sidecar field: "
        "scripts/tools/log_agentic_swe_results_to_wandb.py: "
        '"weave_sidecar_ok": patch_row.get("weave_sidecar_ok"),'
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_existing_results_audit_missing_math_observability_call(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/audit_taiwan_existing_results.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            'issues.extend(agentic_observability_acceptance_issues(rows, row_label="result row"))',
            "return issues",
            1,
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "agentic runner script missing source contract "
        "Agentic Math existing-results observability validation call: "
        "scripts/tools/audit_taiwan_existing_results.py: "
        'agentic_observability_acceptance_issues(rows, row_label="result row")'
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_existing_results_audit_missing_swe_observability_call(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/audit_taiwan_existing_results.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            'issues.extend(agentic_observability_acceptance_issues(rows_for_audit, row_label="patch row"))',
            "return issues",
            1,
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "agentic runner script missing source contract "
        "SWE existing-results observability validation call: "
        "scripts/tools/audit_taiwan_existing_results.py: "
        'agentic_observability_acceptance_issues(rows_for_audit, row_label="patch row")'
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_agentic_math_missing_cached_weave_guard_call(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/run_agentic_math_openclaw.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            "and record_weave_sidecar_allows_reuse(record)",
            "and True",
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "agentic runner script missing source contract "
        "Weave sidecar cached result reuse call: "
        "scripts/tools/run_agentic_math_openclaw.py: "
        "and record_weave_sidecar_allows_reuse(record)"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_agentic_math_missing_cached_audit_guard_call(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/run_agentic_math_openclaw.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            "and record_nemoclaw_session_audit_matches_cache(record, cache_key)",
            "and True",
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "agentic runner script missing source contract "
        "NeMoClaw session audit cached result reuse call: "
        "scripts/tools/run_agentic_math_openclaw.py: "
        "and record_nemoclaw_session_audit_matches_cache(record, cache_key)"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_agentic_math_missing_fresh_sidecar_identity_guard_call(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/run_agentic_math_openclaw.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            "if not sidecar_matches_cache(sidecar, cache_key):\n        raise RuntimeError(",
            "if False:\n        raise RuntimeError(",
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "OpenClaw fresh sidecar identity adoption call" in error
        and "scripts/tools/run_agentic_math_openclaw.py" in error
        and "if not sidecar_matches_cache(sidecar, cache_key):\n        raise RuntimeError(" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_agentic_math_missing_relogged_sidecar_identity_guard_call(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/run_agentic_math_openclaw.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            "sidecar = relog_existing_sidecar(sidecar_path, args)\n"
            "            if not sidecar_matches_cache(sidecar, cache_key):",
            "sidecar = relog_existing_sidecar(sidecar_path, args)\n"
            "            if False:",
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "OpenClaw relogged sidecar identity revalidation call" in error
        and "scripts/tools/run_agentic_math_openclaw.py" in error
        and "sidecar = relog_existing_sidecar(sidecar_path, args)\n"
        "            if not sidecar_matches_cache(sidecar, cache_key):" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_agentic_math_missing_fresh_sidecar_audit_guard_call(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/run_agentic_math_openclaw.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            "if not sidecar_nemoclaw_session_audit_matches_cache(sidecar, cache_key):\n"
            "        raise RuntimeError(",
            "if False:\n        raise RuntimeError(",
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "NeMoClaw session audit fresh sidecar adoption call" in error
        and "scripts/tools/run_agentic_math_openclaw.py" in error
        and "if not sidecar_nemoclaw_session_audit_matches_cache(sidecar, cache_key):\n"
        "        raise RuntimeError(" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_swe_missing_cached_weave_guard_call(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/run_swebench_pro_openclaw.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            "if not patch_record_weave_sidecar_allows_reuse(record):",
            "if False:",
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "agentic runner script missing source contract "
        "Weave sidecar cached patch reuse call: "
        "scripts/tools/run_swebench_pro_openclaw.py: "
        "if not patch_record_weave_sidecar_allows_reuse(record):"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_swe_missing_fresh_sidecar_identity_guard_call(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/run_swebench_pro_openclaw.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            "if not sidecar_identity_matches_cache(sidecar, cache_key):\n        raise RuntimeError(",
            "if False:\n        raise RuntimeError(",
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "OpenClaw fresh sidecar identity adoption call" in error
        and "scripts/tools/run_swebench_pro_openclaw.py" in error
        and "if not sidecar_identity_matches_cache(sidecar, cache_key):\n        raise RuntimeError(" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_swe_missing_fresh_sidecar_audit_guard_call(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/run_swebench_pro_openclaw.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            "if not sidecar_nemoclaw_session_audit_matches_cache(sidecar, cache_key):\n"
            "        raise RuntimeError(",
            "if False:\n        raise RuntimeError(",
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "NeMoClaw session audit fresh sidecar adoption call" in error
        and "scripts/tools/run_swebench_pro_openclaw.py" in error
        and "if not sidecar_nemoclaw_session_audit_matches_cache(sidecar, cache_key):\n"
        "        raise RuntimeError(" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_swe_missing_cached_audit_guard_call(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/run_swebench_pro_openclaw.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            "if not patch_record_nemoclaw_session_audit_matches_cache(record, cache_key):",
            "if False:",
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "agentic runner script missing source contract "
        "NeMoClaw session audit cached patch reuse call: "
        "scripts/tools/run_swebench_pro_openclaw.py: "
        "if not patch_record_nemoclaw_session_audit_matches_cache(record, cache_key):"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_swe_missing_remote_lookup_disable_contract(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/run_swebench_pro_openclaw.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            'browser["enabled"] = False',
            'browser["enabled"] = True',
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "agentic runner script missing source contract OpenClaw browser disabled: "
        'scripts/tools/run_swebench_pro_openclaw.py: browser["enabled"] = False'
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_full_batch_missing_weave_gate_contract(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/run_taiwan_full_eval_batch.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            "from weave_content_canary_gate_contract import",
            "from removed_weave_content_canary_gate_contract import",
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "agentic runner script missing source contract "
        "native Weave content canary contract helper import: "
        "scripts/tools/run_taiwan_full_eval_batch.py: "
        "from weave_content_canary_gate_contract import"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_full_batch_missing_nemoclaw_deny_guard(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/run_taiwan_full_eval_batch.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            "REQUIRED_NEMOCLAW_AGENTIC_DENIED_TOOLS",
            "REMOVED_NEMOCLAW_AGENTIC_DENIED_TOOLS",
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "agentic runner script missing source contract "
        "NeMoClaw batch deny tools constant: "
        "scripts/tools/run_taiwan_full_eval_batch.py: "
        "REQUIRED_NEMOCLAW_AGENTIC_DENIED_TOOLS"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_full_batch_missing_nemoclaw_wandb_config_expectations(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/run_taiwan_full_eval_batch.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            "BENCHMARK_NEMOCLAW_CONFIG_EXPECTATIONS",
            "REMOVED_AGENTIC_WANDB_CONFIG_EXPECTATIONS",
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "agentic runner script missing source contract "
        "NeMoClaw W&B verifier config expectations constant: "
        "scripts/tools/run_taiwan_full_eval_batch.py: "
        "BENCHMARK_NEMOCLAW_CONFIG_EXPECTATIONS"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_full_batch_missing_agentic_production_evidence_guard(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/run_taiwan_full_eval_batch.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            "def build_agentic_production_evidence_guard(",
            "def removed_agentic_production_evidence_guard(",
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "agentic runner script missing source contract "
        "Agentic production evidence guard function: "
        "scripts/tools/run_taiwan_full_eval_batch.py: "
        "def build_agentic_production_evidence_guard("
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_full_batch_missing_weave_request_model_binding(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/run_taiwan_full_eval_batch.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            "def weave_expected_request_models(",
            "def removed_request_models(",
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "agentic runner script missing source contract "
        "Weave request-model alias resolver: "
        "scripts/tools/run_taiwan_full_eval_batch.py: "
        "def weave_expected_request_models("
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_full_batch_missing_nemoclaw_wandb_audit_flag(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/run_taiwan_full_eval_batch.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            "--require-nemoclaw-session-audit",
            "--removed-nemoclaw-session-audit",
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "agentic runner script missing source contract "
        "W&B completion requires NeMoClaw session audit: "
        "scripts/tools/run_taiwan_full_eval_batch.py: "
        "--require-nemoclaw-session-audit"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_weave_agents_verifier_missing_request_model_contract(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path") == "scripts/tools/verify_taiwan_weave_agents.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            "--expected-request-model",
            "--removed-request-model",
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "agentic runner script missing source contract "
        "Weave request-model CLI option: "
        "scripts/tools/verify_taiwan_weave_agents.py: "
        "--expected-request-model"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_weave_content_canary_contract_helper_tamper(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        item
        for item in manifest["files"]
        if item.get("source_path")
        == "scripts/tools/weave_content_canary_gate_contract.py"
    )
    script_path = bundle / record["bundle_path"]
    script_text = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script_text.replace(
            "def weave_content_canary_gate_contract_issues(",
            "def removed_weave_content_canary_gate_contract_issues(",
        ),
        encoding="utf-8",
    )
    refresh_manifest_record_hash(bundle, record["bundle_path"])

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "agentic runner script missing source contract "
        "native Weave content canary contract validator: "
        "scripts/tools/weave_content_canary_gate_contract.py: "
        "def weave_content_canary_gate_contract_issues("
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_current_gate_remediation_script_missing_role(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["files"]:
        if record.get("source_path") == "scripts/tools/run_taiwan_release_gate.py":
            record["roles"] = [
                role
                for role in record.get("roles", [])
                if role != "current_gate:remediation_plan:command_script"
            ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "current_gate remediation_plan command script missing role "
        "current_gate:remediation_plan:command_script: scripts/tools/run_taiwan_release_gate.py"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_release_gate_pointer_missing_paths(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_gate_pointer"] = {
        "release_gate_json": None,
        "latest_pointer_json": "",
        "latest_pointer_verification_json": None,
        "latest_pointer_verification_ok": True,
        "latest_pointer_verification_status": "passed",
        "latest_pointer_verification_issue_count": 0,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "manifest release_gate_pointer release_gate_json is not a non-empty string"
        in payload["errors"]
    )
    assert (
        "manifest release_gate_pointer latest_pointer_json is not a non-empty string"
        in payload["errors"]
    )
    assert (
        "manifest release_gate_pointer latest_pointer_verification_json is not a non-empty string"
        in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_failed_release_gate_pointer_summary(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_gate_pointer"] = {
        "release_gate_json": "temp/taiwan_release_gate_20260628T040000Z.json",
        "latest_pointer_json": "temp/latest_taiwan_release_gate.json",
        "latest_pointer_verification_json": "temp/latest_taiwan_release_gate_verify_20260628T040000Z.json",
        "latest_pointer_verification_ok": False,
        "latest_pointer_verification_status": "failed",
        "latest_pointer_verification_issue_count": 2,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "manifest release_gate_pointer latest_pointer_verification_ok must be true"
        in payload["errors"]
    )
    assert (
        "manifest release_gate_pointer latest_pointer_verification_status must be passed"
        in payload["errors"]
    )
    assert (
        "manifest release_gate_pointer latest_pointer_verification_issue_count must be 0"
        in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_release_gate_pointer_missing_operator_plan(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    attach_valid_release_gate_pointer(bundle, tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_gate_pointer"].pop("operator_plan")
    proof_path = bundle / "release_gate_pointer_proof.json"
    proof_payload = json.loads(proof_path.read_text(encoding="utf-8"))
    proof_payload["release_gate_pointer"] = manifest["release_gate_pointer"]
    write_json(proof_path, proof_payload)
    refresh_manifest_record_hash(bundle, "release_gate_pointer_proof.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_gate_pointer"] = proof_payload["release_gate_pointer"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "manifest release_gate_pointer operator_plan is not an object"
        in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_release_gate_pointer_operator_plan_mismatch(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    attach_valid_release_gate_pointer(bundle, tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_gate_pointer"]["operator_plan"]["status"] = "stale"
    proof_path = bundle / "release_gate_pointer_proof.json"
    proof_payload = json.loads(proof_path.read_text(encoding="utf-8"))
    proof_payload["release_gate_pointer"] = manifest["release_gate_pointer"]
    write_json(proof_path, proof_payload)
    refresh_manifest_record_hash(bundle, "release_gate_pointer_proof.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_gate_pointer"] = proof_payload["release_gate_pointer"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "release gate pointer release gate JSON operator_plan does not match "
        "manifest release_gate_pointer"
    ) in payload["errors"]
    assert (
        "release gate pointer latest pointer JSON operator_plan does not match "
        "manifest release_gate_pointer"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_release_gate_pointer_operator_plan_bad_name(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    attach_valid_release_gate_pointer(bundle, tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_gate_pointer"]["operator_plan"]["json"] = (
        str(tmp_path / "operator_plan.json")
    )
    manifest["release_gate_pointer"]["operator_plan"]["markdown"] = (
        str(tmp_path / "operator_plan.md")
    )
    proof_path = bundle / "release_gate_pointer_proof.json"
    proof_payload = json.loads(proof_path.read_text(encoding="utf-8"))
    proof_payload["release_gate_pointer"] = manifest["release_gate_pointer"]
    write_json(proof_path, proof_payload)
    refresh_manifest_record_hash(bundle, "release_gate_pointer_proof.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_gate_pointer"] = proof_payload["release_gate_pointer"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "manifest release_gate_pointer operator_plan json must point to "
        "taiwan_release_operator_plan_YYYYMMDDTHHMMSSZ.json"
    ) in payload["errors"]
    assert (
        "manifest release_gate_pointer operator_plan markdown must point to "
        "taiwan_release_operator_plan_YYYYMMDDTHHMMSSZ.md"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_release_gate_pointer_nonformal_paths(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_gate_pointer"] = {
        "release_gate_json": "temp/release_gate.json",
        "latest_pointer_json": "temp/latest.json",
        "latest_pointer_verification_json": "temp/latest_pointer_check.json",
        "latest_pointer_verification_ok": True,
        "latest_pointer_verification_status": "passed",
        "latest_pointer_verification_issue_count": 0,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "manifest release_gate_pointer release_gate_json must point to "
        "taiwan_release_gate_YYYYMMDDTHHMMSSZ.json"
    ) in payload["errors"]
    assert (
        "manifest release_gate_pointer latest_pointer_json must point to "
        "latest_taiwan_release_gate.json"
    ) in payload["errors"]
    assert (
        "manifest release_gate_pointer latest_pointer_verification_json must point to "
        "latest_taiwan_release_gate_verify_YYYYMMDDTHHMMSSZ.json"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_release_gate_pointer_timestamp_mismatch(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_gate_pointer"] = {
        "release_gate_json": "temp/taiwan_release_gate_20260628T040000Z.json",
        "latest_pointer_json": "temp/latest_taiwan_release_gate.json",
        "latest_pointer_verification_json": "temp/latest_taiwan_release_gate_verify_20260628T041000Z.json",
        "latest_pointer_verification_ok": True,
        "latest_pointer_verification_status": "passed",
        "latest_pointer_verification_issue_count": 0,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "manifest release_gate_pointer release gate timestamp does not match "
        "latest pointer verification timestamp"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_release_gate_pointer_verification_json_mismatch(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    release_gate_json = tmp_path / "taiwan_release_gate_20260628T040000Z.json"
    latest_pointer_json = tmp_path / "latest_taiwan_release_gate.json"
    verification_json = tmp_path / "latest_taiwan_release_gate_verify_20260628T040000Z.json"
    release_gate_json.write_text("{}", encoding="utf-8")
    latest_pointer_json.write_text("{}", encoding="utf-8")
    write_json(
        verification_json,
        {
            "schema_version": 1,
            "ok": True,
            "status": "passed",
            "issue_count": 0,
            "issues": [],
            "release_gate_json": str(tmp_path / "taiwan_release_gate_20260628T041000Z.json"),
            "pointer_json": str(latest_pointer_json),
        },
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_gate_pointer"] = {
        "release_gate_json": str(release_gate_json),
        "latest_pointer_json": str(latest_pointer_json),
        "latest_pointer_verification_json": str(verification_json),
        "latest_pointer_verification_ok": True,
        "latest_pointer_verification_status": "passed",
        "latest_pointer_verification_issue_count": 0,
    }
    add_manifest_file_record(
        manifest,
        bundle,
        release_gate_json,
        "release_gate_pointer:release_gate_json",
    )
    add_manifest_file_record(
        manifest,
        bundle,
        latest_pointer_json,
        "release_gate_pointer:latest_pointer_json",
    )
    add_manifest_file_record(
        manifest,
        bundle,
        verification_json,
        "release_gate_pointer:latest_pointer_verification_json",
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "latest pointer verification JSON release_gate_json does not match "
        "manifest release_gate_pointer"
    ) in payload["errors"]


def test_verify_release_evidence_bundle_rejects_release_gate_pointer_verification_missing_schema(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    release_gate_json = tmp_path / "taiwan_release_gate_20260628T040000Z.json"
    latest_pointer_json = tmp_path / "latest_taiwan_release_gate.json"
    verification_json = tmp_path / "latest_taiwan_release_gate_verify_20260628T040000Z.json"
    release_gate_json.write_text("{}", encoding="utf-8")
    latest_pointer_json.write_text("{}", encoding="utf-8")
    write_json(
        verification_json,
        {
            "ok": True,
            "status": "passed",
            "issue_count": 0,
            "issues": [],
            "release_gate_json": str(release_gate_json),
            "pointer_json": str(latest_pointer_json),
        },
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_gate_pointer"] = {
        "release_gate_json": str(release_gate_json),
        "latest_pointer_json": str(latest_pointer_json),
        "latest_pointer_verification_json": str(verification_json),
        "latest_pointer_verification_ok": True,
        "latest_pointer_verification_status": "passed",
        "latest_pointer_verification_issue_count": 0,
    }
    add_manifest_file_record(
        manifest,
        bundle,
        release_gate_json,
        "release_gate_pointer:release_gate_json",
    )
    add_manifest_file_record(
        manifest,
        bundle,
        latest_pointer_json,
        "release_gate_pointer:latest_pointer_json",
    )
    add_manifest_file_record(
        manifest,
        bundle,
        verification_json,
        "release_gate_pointer:latest_pointer_verification_json",
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert "latest pointer verification JSON schema_version must be 1" in payload["errors"]


def test_verify_release_evidence_bundle_rejects_stale_release_gate_pointer_proof(
    tmp_path,
):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    timestamp = "20260628T040000Z"
    release_gate_json = tmp_path / f"taiwan_release_gate_{timestamp}.json"
    latest_pointer_json = tmp_path / "latest_taiwan_release_gate.json"
    verification_json = tmp_path / f"latest_taiwan_release_gate_verify_{timestamp}.json"
    write_json(
        release_gate_json,
        {
            "schema_version": 1,
            "timestamp": timestamp,
            "release_gate_json": str(release_gate_json),
            "latest_pointer_json": str(latest_pointer_json),
            "latest_pointer_verification_json": str(verification_json),
            "latest_pointer_verification_status": "failed",
            "latest_pointer_verification_issue_count": 1,
            "verification_error_count": 7,
            "bundle_integrity_ok": False,
            "bundle_verification": {
                "integrity_ok": False,
                "errors": ["stale pointer copy"],
            },
            "latest_pointer_verification": {
                "ok": False,
                "status": "failed",
                "issue_count": 1,
            },
        },
    )
    write_json(
        latest_pointer_json,
        {
            "schema_version": 1,
            "timestamp": timestamp,
            "release_gate_json": str(release_gate_json),
            "latest_pointer_json": str(latest_pointer_json),
            "latest_pointer_verification_json": str(verification_json),
            "latest_pointer_verification_ok": True,
            "latest_pointer_verification_status": "passed",
            "latest_pointer_verification_issue_count": 0,
            "verification_error_count": 3,
        },
    )
    write_json(
        verification_json,
        {
            "schema_version": 1,
            "ok": True,
            "status": "passed",
            "issue_count": 0,
            "issues": [],
            "release_gate_json": str(release_gate_json),
            "pointer_json": str(latest_pointer_json),
        },
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_gate_pointer"] = {
        "release_gate_json": str(release_gate_json),
        "latest_pointer_json": str(latest_pointer_json),
        "latest_pointer_verification_json": str(verification_json),
        "latest_pointer_verification_ok": True,
        "latest_pointer_verification_status": "passed",
        "latest_pointer_verification_issue_count": 0,
    }
    add_manifest_file_record(
        manifest,
        bundle,
        release_gate_json,
        "release_gate_pointer:release_gate_json",
    )
    add_manifest_file_record(
        manifest,
        bundle,
        latest_pointer_json,
        "release_gate_pointer:latest_pointer_json",
    )
    add_manifest_file_record(
        manifest,
        bundle,
        verification_json,
        "release_gate_pointer:latest_pointer_verification_json",
    )
    proof_path = bundle / "release_gate_pointer_proof.json"
    write_json(
        proof_path,
        {
            "schema_version": 1,
            "kind": "release_gate_pointer_proof",
            "ok": False,
            "status": "failed",
            "release_gate_pointer": manifest["release_gate_pointer"],
            "latest_pointer_verification": {
                "json": str(verification_json),
                "ok": False,
                "status": "failed",
                "issue_count": 1,
            },
        },
    )
    manifest["release_gate_pointer_proof"] = {
        "json": "release_gate_pointer_proof.json",
        "schema_version": 1,
        "status": "failed",
        "ok": False,
    }
    manifest.setdefault("files", []).append(
        {
            "source_path": str(proof_path),
            "roles": ["release_gate_pointer", "release_gate_pointer:proof_json"],
            "exists": True,
            "bundle_path": "release_gate_pointer_proof.json",
            "size_bytes": proof_path.stat().st_size,
            "sha256": sha256(proof_path),
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert (
        "manifest release_gate_pointer_proof status must be passed"
        in payload["errors"]
    )
    assert (
        "release_gate_pointer_proof JSON latest_pointer_verification issue_count must be 0"
        in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_deprecated_wandb_adoption_flags(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    operator_plan_path = bundle / "operator_plan.json"
    payload = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    payload["operator_next_steps"]["steps"][0]["commands"][0] += (
        " --scope-confirmed-by REVIEWER"
    )
    operator_plan_path.write_text(json.dumps(payload), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "uses deprecated W&B adoption flag(s): --scope-confirmed-by" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_batch_command_without_external_approval(tmp_path):
    bundle = build_bundle_with_operator_batch_command(tmp_path)
    manifest_path = bundle / "manifest.json"
    operator_plan_path = bundle / "operator_plan.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    approval_flag = (
        " --external-action-approval-report-json "
        "temp/taiwan_external_action_approval_REVIEWED_YYYYMMDDTHHMM.verify.json"
    )
    operator_plan["operator_next_steps"]["steps"][0]["commands"][0] = operator_plan[
        "operator_next_steps"
    ]["steps"][0]["commands"][0].replace(approval_flag, "")
    manifest["current_gate"]["operator_next_steps"] = operator_plan["operator_next_steps"]
    manifest["current_gate"]["remediation_plan"][0]["commands"][0] = manifest[
        "current_gate"
    ]["remediation_plan"][0]["commands"][0].replace(approval_flag, "")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "runs run_taiwan_full_eval_batch.py without --external-action-approval-report-json"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_batch_command_without_approval_source_packet(tmp_path):
    bundle = build_bundle_with_operator_batch_command(tmp_path)
    manifest_path = bundle / "manifest.json"
    operator_plan_path = bundle / "operator_plan.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    source_packet_flag = (
        " --external-action-approval-source-packet-json "
        "outputs/taiwan_release_evidence/bundle_YYYYMMDDTHHMM/"
        "external_action_approval_packet.json"
    )
    operator_plan["operator_next_steps"]["steps"][0]["commands"][0] = operator_plan[
        "operator_next_steps"
    ]["steps"][0]["commands"][0].replace(source_packet_flag, "")
    manifest["current_gate"]["operator_next_steps"] = operator_plan["operator_next_steps"]
    manifest["current_gate"]["remediation_plan"][0]["commands"][0] = manifest[
        "current_gate"
    ]["remediation_plan"][0]["commands"][0].replace(source_packet_flag, "")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "runs run_taiwan_full_eval_batch.py without "
        "--external-action-approval-source-packet-json"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_batch_command_without_weave_usage_guard(tmp_path):
    bundle = build_bundle_with_operator_batch_command(tmp_path)
    manifest_path = bundle / "manifest.json"
    operator_plan_path = bundle / "operator_plan.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    usage_flag = " --weave-agents-require-usage"
    operator_plan["operator_next_steps"]["steps"][0]["commands"][0] = operator_plan[
        "operator_next_steps"
    ]["steps"][0]["commands"][0].replace(usage_flag, "")
    manifest["current_gate"]["operator_next_steps"] = operator_plan["operator_next_steps"]
    manifest["current_gate"]["remediation_plan"][0]["commands"][0] = manifest[
        "current_gate"
    ]["remediation_plan"][0]["commands"][0].replace(usage_flag, "")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "runs paid agentic run_taiwan_full_eval_batch.py without "
        "--weave-agents-require-usage"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_batch_command_without_agentic_math_nemoclaw_config_path(
    tmp_path,
):
    bundle = build_bundle_with_operator_batch_command(tmp_path)
    manifest_path = bundle / "manifest.json"
    operator_plan_path = bundle / "operator_plan.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    config_flag = (
        " --agentic-math-nemoclaw-openclaw-config-path "
        "/sandbox/.openclaw/openclaw.json"
    )
    operator_plan["operator_next_steps"]["steps"][0]["commands"][0] = operator_plan[
        "operator_next_steps"
    ]["steps"][0]["commands"][0].replace(config_flag, "")
    manifest["current_gate"]["operator_next_steps"] = operator_plan["operator_next_steps"]
    manifest["current_gate"]["remediation_plan"][0]["commands"][0] = manifest[
        "current_gate"
    ]["remediation_plan"][0]["commands"][0].replace(config_flag, "")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "runs paid agentic run_taiwan_full_eval_batch.py without "
        "--agentic-math-nemoclaw-openclaw-config-path"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_batch_command_with_wrong_swe_nemoclaw_config_path(
    tmp_path,
):
    bundle = build_bundle_with_operator_batch_command(tmp_path)
    manifest_path = bundle / "manifest.json"
    operator_plan_path = bundle / "operator_plan.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    expected_flag = (
        "--swebench-pro-nemoclaw-openclaw-config-path "
        "/sandbox/.openclaw/openclaw.json"
    )
    wrong_flag = (
        "--swebench-pro-nemoclaw-openclaw-config-path "
        "/tmp/openclaw.json"
    )
    operator_plan["operator_next_steps"]["steps"][0]["commands"][0] = operator_plan[
        "operator_next_steps"
    ]["steps"][0]["commands"][0].replace(expected_flag, wrong_flag)
    manifest["current_gate"]["operator_next_steps"] = operator_plan["operator_next_steps"]
    manifest["current_gate"]["remediation_plan"][0]["commands"][0] = manifest[
        "current_gate"
    ]["remediation_plan"][0]["commands"][0].replace(expected_flag, wrong_flag)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "runs paid agentic run_taiwan_full_eval_batch.py with "
        "--swebench-pro-nemoclaw-openclaw-config-path other than "
        "/sandbox/.openclaw/openclaw.json"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_weave_canary_without_external_approval(tmp_path):
    bundle = build_bundle_with_operator_weave_content_canary_command(tmp_path)
    manifest_path = bundle / "manifest.json"
    operator_plan_path = bundle / "operator_plan.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    approval_flag = (
        " --external-action-approval-report-json "
        "temp/taiwan_external_action_approval_REVIEWED_YYYYMMDDTHHMM.verify.json"
    )
    operator_plan["operator_next_steps"]["steps"][0]["commands"][1] = operator_plan[
        "operator_next_steps"
    ]["steps"][0]["commands"][1].replace(approval_flag, "")
    manifest["current_gate"]["operator_next_steps"] = operator_plan["operator_next_steps"]
    manifest["current_gate"]["remediation_plan"][0]["commands"][1] = manifest[
        "current_gate"
    ]["remediation_plan"][0]["commands"][1].replace(approval_flag, "")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "runs run_weave_agents_content_canary.py --execute without "
        "--external-action-approval-report-json"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_weave_canary_without_approval_source_packet(tmp_path):
    bundle = build_bundle_with_operator_weave_content_canary_command(tmp_path)
    manifest_path = bundle / "manifest.json"
    operator_plan_path = bundle / "operator_plan.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    source_packet_flag = (
        " --external-action-approval-source-packet-json "
        "outputs/taiwan_release_evidence/bundle_YYYYMMDDTHHMM/"
        "external_action_approval_packet.json"
    )
    operator_plan["operator_next_steps"]["steps"][0]["commands"][1] = operator_plan[
        "operator_next_steps"
    ]["steps"][0]["commands"][1].replace(source_packet_flag, "")
    manifest["current_gate"]["operator_next_steps"] = operator_plan["operator_next_steps"]
    manifest["current_gate"]["remediation_plan"][0]["commands"][1] = manifest[
        "current_gate"
    ]["remediation_plan"][0]["commands"][1].replace(source_packet_flag, "")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "runs run_weave_agents_content_canary.py --execute without "
        "--external-action-approval-source-packet-json"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_weave_canary_without_nemoclaw_sandbox(tmp_path):
    bundle = build_bundle_with_operator_weave_content_canary_command(tmp_path)
    manifest_path = bundle / "manifest.json"
    operator_plan_path = bundle / "operator_plan.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    nemoclaw_flag = " --nemoclaw-sandbox nejumi-taiwan"
    operator_plan["operator_next_steps"]["steps"][0]["commands"][1] = operator_plan[
        "operator_next_steps"
    ]["steps"][0]["commands"][1].replace(nemoclaw_flag, "")
    manifest["current_gate"]["operator_next_steps"] = operator_plan["operator_next_steps"]
    manifest["current_gate"]["remediation_plan"][0]["commands"][1] = manifest[
        "current_gate"
    ]["remediation_plan"][0]["commands"][1].replace(nemoclaw_flag, "")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "runs run_weave_agents_content_canary.py --execute without --nemoclaw-sandbox"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_weave_canary_without_nemoclaw_openclaw_config_path(
    tmp_path,
):
    bundle = build_bundle_with_operator_weave_content_canary_command(tmp_path)
    manifest_path = bundle / "manifest.json"
    operator_plan_path = bundle / "operator_plan.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    config_flag = " --nemoclaw-openclaw-config-path /sandbox/.openclaw/openclaw.json"
    operator_plan["operator_next_steps"]["steps"][0]["commands"][1] = operator_plan[
        "operator_next_steps"
    ]["steps"][0]["commands"][1].replace(config_flag, "")
    manifest["current_gate"]["operator_next_steps"] = operator_plan["operator_next_steps"]
    manifest["current_gate"]["remediation_plan"][0]["commands"][1] = manifest[
        "current_gate"
    ]["remediation_plan"][0]["commands"][1].replace(config_flag, "")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "runs run_weave_agents_content_canary.py --execute without "
        "--nemoclaw-openclaw-config-path /sandbox/.openclaw/openclaw.json"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_agents_check_without_json_output(tmp_path):
    bundle = build_bundle_with_operator_weave_content_canary_command(tmp_path)
    manifest_path = bundle / "manifest.json"
    operator_plan_path = bundle / "operator_plan.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    command = (
        "uv run python scripts/tools/run_openclaw_agent_protocol.py check-agents "
        "--entity llm-leaderboard --project tc-leaderboard "
        "--agent-name nejumi-taiwan-openclaw --limit 20"
    )
    operator_plan["operator_next_steps"]["steps"][0]["commands"].append(command)
    manifest["current_gate"]["operator_next_steps"] = operator_plan["operator_next_steps"]
    manifest["current_gate"]["remediation_plan"][0]["commands"].append(command)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "runs run_openclaw_agent_protocol.py check-agents without --json" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_agents_check_noncanonical_json_output(tmp_path):
    bundle = build_bundle_with_operator_weave_content_canary_command(tmp_path)
    manifest_path = bundle / "manifest.json"
    operator_plan_path = bundle / "operator_plan.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    command = (
        "uv run python scripts/tools/run_openclaw_agent_protocol.py check-agents "
        "--entity llm-leaderboard --project tc-leaderboard "
        "--agent-name nejumi-taiwan-openclaw --limit 20 "
        "--json outputs/weave_agents_content_canary/agents_diagnostics/check.json"
    )
    output_path = "outputs/weave_agents_content_canary/agents_diagnostics/check.json"
    operator_plan["operator_next_steps"]["steps"][0]["commands"].append(command)
    operator_plan["operator_next_steps"]["steps"][0]["evidence_to_produce"].append(output_path)
    manifest["current_gate"]["operator_next_steps"] = operator_plan["operator_next_steps"]
    manifest["current_gate"]["remediation_plan"][0]["commands"].append(command)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "check-agents --json output must end with .agents.json" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_wandb_sync_apply_without_dry_run(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    operator_plan_path = bundle / "operator_plan.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    command = (
        "uv run python scripts/tools/sync_wandb_completion_to_paid_review.py "
        "--review-json outputs/taiwan_full_eval/canary_agentic_paid_run_review.json "
        "--completion-json outputs/taiwan_full_eval/wandb_completion/agentic_math-RUN_ID.json "
        "--in-place --set-verify-wandb-completion"
    )
    operator_plan["operator_next_steps"]["steps"][0]["commands"][1] = command
    manifest["current_gate"]["operator_next_steps"] = operator_plan["operator_next_steps"]
    manifest["current_gate"]["remediation_plan"][0]["commands"][1] = command
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "runs sync_wandb_completion_to_paid_review.py --in-place without "
        "--validated-dry-run-report-json"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_weave_agents_sync_apply_without_dry_run(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    operator_plan_path = bundle / "operator_plan.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    command = (
        "uv run python scripts/tools/sync_weave_agents_completion_to_paid_review.py "
        "--review-json outputs/taiwan_full_eval/canary_agentic_paid_run_review.json "
        "--completion-json outputs/taiwan_full_eval/weave_agents_completion/agentic-MODEL_SLUG.json "
        "--run-id RUN_ID --in-place --set-verify-weave-agents"
    )
    operator_plan["operator_next_steps"]["steps"][0]["commands"][1] = command
    manifest["current_gate"]["operator_next_steps"] = operator_plan["operator_next_steps"]
    manifest["current_gate"]["remediation_plan"][0]["commands"][1] = command
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "runs sync_weave_agents_completion_to_paid_review.py --in-place without "
        "--validated-dry-run-report-json"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_operator_openrouter_command(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    operator_plan_path = bundle / "operator_plan.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    command = (
        "uv run python scripts/tools/run_taiwan_full_eval_batch.py "
        "--manifest configs/taiwan_full_eval_models.yaml "
        "--openclaw-model openrouter-direct/z-ai/glm-5.2 "
        "--output-root outputs/taiwan_full_eval"
    )
    operator_plan["operator_next_steps"]["steps"][0]["commands"][0] = command
    operator_plan["operator_next_steps"]["steps"][0]["requires_paid_api"] = True
    operator_plan["operator_next_steps"]["steps"][0]["requires_wandb_access"] = True
    operator_plan["operator_next_steps"]["steps"][0]["requires_wandb_write"] = True
    operator_plan["operator_next_steps"]["steps"][0]["evidence_to_produce"] = []
    operator_plan["operator_next_steps"]["paid_api_step_count"] = 1
    operator_plan["operator_next_steps"]["wandb_access_step_count"] = 1
    operator_plan["operator_next_steps"]["wandb_write_step_count"] = 1
    manifest["current_gate"]["operator_next_steps"] = operator_plan["operator_next_steps"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "uses forbidden release operator marker(s): openrouter" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_current_gate_remediation_openrouter_command(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["current_gate"]["remediation_plan"][0]["commands"][0] = (
        "uv run python scripts/tools/run_taiwan_full_eval_batch.py "
        "--openclaw-model openrouter-direct/z-ai/glm-5.2"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "current_gate remediation_plan row 1 command 1 uses forbidden release operator marker(s): openrouter"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_operator_nemoclaw_broader_policy_tier(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    operator_plan_path = bundle / "operator_plan.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    command = (
        "scripts/setup/install_nemoclaw.sh --install --onboard "
        "--sandbox nejumi-taiwan --provider openai --policy-tier open "
        f"--installer-lock-json {INSTALLER_LOCK_JSON} "
        "--installer-sha256 REVIEWED_INSTALLER_SHA256 "
        "--yes-i-accept-third-party-software "
        "--json temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.json"
    )
    operator_plan["operator_next_steps"]["steps"][0]["commands"][0] = command
    operator_plan["operator_next_steps"]["steps"][0]["requires_third_party_acceptance"] = True
    operator_plan["operator_next_steps"]["steps"][0]["requires_nemoclaw_install"] = True
    operator_plan["operator_next_steps"]["steps"][0]["evidence_to_produce"] = [
        "temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.json"
    ]
    operator_plan["operator_next_steps"]["third_party_acceptance_step_count"] = 1
    manifest["current_gate"]["operator_next_steps"] = operator_plan["operator_next_steps"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "uses NeMoClaw --policy-tier open; use --policy-tier restricted" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_operator_nemoclaw_install_without_review(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    operator_plan_path = bundle / "operator_plan.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    command = (
        "scripts/setup/install_nemoclaw.sh --install --onboard "
        "--sandbox nejumi-taiwan --provider openai --policy-tier restricted "
        f"--installer-lock-json {INSTALLER_LOCK_JSON} "
        "--installer-sha256 REVIEWED_INSTALLER_SHA256 "
        "--yes-i-accept-third-party-software "
        "--json temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.json"
    )
    operator_plan["operator_next_steps"]["steps"][0]["commands"][0] = command
    operator_plan["operator_next_steps"]["steps"][0]["requires_third_party_acceptance"] = True
    operator_plan["operator_next_steps"]["steps"][0]["requires_nemoclaw_install"] = True
    operator_plan["operator_next_steps"]["steps"][0]["evidence_to_produce"] = [
        "temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.json"
    ]
    operator_plan["operator_next_steps"]["third_party_acceptance_step_count"] = 1
    manifest["current_gate"]["operator_next_steps"] = operator_plan["operator_next_steps"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "operator_plan installs NeMoClaw but has no review_nemoclaw_installer.py command"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_operator_nemoclaw_install_without_installer_sha(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    operator_plan_path = bundle / "operator_plan.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    command = (
        "scripts/setup/install_nemoclaw.sh --install --onboard "
        "--sandbox nejumi-taiwan --provider openai --policy-tier restricted "
        f"--installer-lock-json {INSTALLER_LOCK_JSON} "
        "--yes-i-accept-third-party-software "
        "--json temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.json"
    )
    operator_plan["operator_next_steps"]["steps"][0]["commands"][0] = command
    operator_plan["operator_next_steps"]["steps"][0]["requires_third_party_acceptance"] = True
    operator_plan["operator_next_steps"]["steps"][0]["requires_nemoclaw_install"] = True
    operator_plan["operator_next_steps"]["steps"][0]["evidence_to_produce"] = [
        "temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.json"
    ]
    operator_plan["operator_next_steps"]["third_party_acceptance_step_count"] = 1
    manifest["current_gate"]["operator_next_steps"] = operator_plan["operator_next_steps"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "operator_plan command 1 installs NeMoClaw without --installer-sha256" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_current_gate_nemoclaw_install_without_review(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["current_gate"]["remediation_plan"][0]["commands"][0] = (
        "scripts/setup/install_nemoclaw.sh --install --onboard "
        "--sandbox nejumi-taiwan --provider openai --policy-tier restricted "
        f"--installer-lock-json {INSTALLER_LOCK_JSON} "
        "--installer-sha256 REVIEWED_INSTALLER_SHA256 "
        "--yes-i-accept-third-party-software "
        "--json temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.json"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "manifest current_gate remediation_plan installs NeMoClaw but has no "
        "review_nemoclaw_installer.py command" in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_current_gate_nemoclaw_install_without_installer_sha(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["current_gate"]["remediation_plan"][0]["commands"][0] = (
        "scripts/setup/install_nemoclaw.sh --install --onboard "
        "--sandbox nejumi-taiwan --provider openai --policy-tier restricted "
        f"--installer-lock-json {INSTALLER_LOCK_JSON} "
        "--yes-i-accept-third-party-software "
        "--json temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.json"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "manifest current_gate remediation_plan row 1 command 1 installs NeMoClaw without --installer-sha256"
        in error
        for error in payload["errors"]
    )


def test_verify_release_evidence_bundle_rejects_operator_nemoclaw_onboard_without_policy_tier(tmp_path):
    bundle = build_bundle_with_operator_command_script(tmp_path)
    manifest_path = bundle / "manifest.json"
    operator_plan_path = bundle / "operator_plan.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operator_plan = json.loads(operator_plan_path.read_text(encoding="utf-8"))
    command = (
        "scripts/setup/install_nemoclaw.sh --onboard "
        "--sandbox nejumi-taiwan --provider openai "
        "--yes-i-accept-third-party-software "
        "--json temp/nemoclaw_onboard_YYYYMMDDTHHMM.json"
    )
    operator_plan["operator_next_steps"]["steps"][0]["commands"][0] = command
    operator_plan["operator_next_steps"]["steps"][0]["requires_third_party_acceptance"] = True
    operator_plan["operator_next_steps"]["steps"][0]["requires_nemoclaw_install"] = False
    operator_plan["operator_next_steps"]["steps"][0]["evidence_to_produce"] = [
        "temp/nemoclaw_onboard_YYYYMMDDTHHMM.json"
    ]
    operator_plan["operator_next_steps"]["third_party_acceptance_step_count"] = 1
    manifest["current_gate"]["operator_next_steps"] = operator_plan["operator_next_steps"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    operator_plan_path.write_text(json.dumps(operator_plan), encoding="utf-8")
    refresh_manifest_record_hash(bundle, "operator_plan.json")

    result = subprocess.run(
        ["python3", str(VERIFY_SCRIPT), "--bundle-dir", str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["integrity_ok"] is False
    assert any(
        "onboards NeMoClaw without --policy-tier restricted" in error
        for error in payload["errors"]
    )
