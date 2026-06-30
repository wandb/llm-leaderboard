import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tools" / "verify_taiwan_release_gate_pointer.py"


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


def write_pointer_fixture(tmp_path: Path):
    timestamp = "20260627T211341Z"
    gate_path = tmp_path / f"taiwan_release_gate_{timestamp}.json"
    readiness_path = tmp_path / "readiness.json"
    operator_plan_json = tmp_path / f"taiwan_release_operator_plan_{timestamp}.json"
    operator_plan_markdown = tmp_path / f"taiwan_release_operator_plan_{timestamp}.md"
    manifest = write_json(
        tmp_path / "bundle" / "manifest.json",
        {"schema_version": 1, "bundle_version": 2},
    )
    verify_json = tmp_path / "bundle_verify.json"
    latest_pointer_verify_json = (
        tmp_path / f"latest_taiwan_release_gate_verify_{timestamp}.json"
    )
    external_action_checklist = {
        "schema_version": 1,
        "status": "pending",
        "blocking_gate_count": 1,
        "item_count": 1,
        "external_action_item_count": 1,
        "requirement_counts": {
            "paid_api": 1,
            "wandb_access": 1,
            "wandb_write": 1,
            "third_party_acceptance": 0,
            "nemoclaw_install": 0,
            "scope_confirmation": 0,
        },
        "items": [
            {
                "order": 1,
                "gate": "weave_content_canary",
                "status": "failed",
                "blocking": True,
                "external_action_required": True,
                "requirements": ["paid_api", "wandb_access", "wandb_write"],
                "command_count": 1,
                "evidence_path_count": 1,
                "next_action": "rerun canary",
                "commands": ["uv run python scripts/tools/run_weave_agents_content_canary.py --execute"],
                "evidence_to_produce": ["outputs/weave_agents_content_canary/plans/canary.gate.json"],
                "warnings": [],
            }
        ],
    }
    external_action_approval_packet = {
        "schema_version": 1,
        "status": "pending_approval",
        "json": "external_action_approval_packet.json",
        "markdown": "external_action_approval_packet.md",
        "external_action_checklist_sha256": "abc123",
        "required_approval_count": 1,
        "all_required_approvals_granted": False,
    }
    operator_plan = {
        "json": str(operator_plan_json),
        "markdown": str(operator_plan_markdown),
        "schema_version": 1,
        "status": "pending",
    }
    gate = write_json(
        gate_path,
        {
            "schema_version": 1,
            "timestamp": timestamp,
            "ok": False,
            "status": "not_ready",
            "release_ready": False,
            "readiness_ok": False,
            "readiness_report_schema_version": 1,
            "gate_count": 3,
            "blocker_count": 1,
            "readiness_status": "not_ready",
            "blocking_gates": ["weave_content_canary"],
            "blockers": ["weave_content_canary"],
            "required_next_actions": [
                {
                    "gate": "weave_content_canary",
                    "status": "failed",
                    "detail": "missing canary",
                    "next_action": "rerun canary",
                }
            ],
            "benchmark_completion": [{"benchmark": "agentic_math", "completion_proven": False}],
            "weave_agents_completion": [{"gate": "paid_run_review_package", "entry_count": 0}],
            "existing_results_formalization": {"status": "passed"},
            "wandb_adoption_draft": {"status": "pending"},
            "paid_run_review_package": {"status": "not_ready"},
            "wandb_completion_contract": {"status": "incomplete"},
            "nemoclaw_adoption": {"status": "not_installed"},
            "operator_next_steps": {"status": "pending", "step_count": 1},
            "operator_plan": operator_plan,
            "external_action_checklist": external_action_checklist,
            "external_action_approval_packet": external_action_approval_packet,
            "bundle_file_count": 12,
            "bundle_missing_file_count": 0,
            "readiness_report": str(readiness_path),
            "readiness_report_source": str(readiness_path),
            "latest_pointer_json": str(tmp_path / "latest_taiwan_release_gate.json"),
            "latest_pointer_verification_json": str(latest_pointer_verify_json),
            "latest_pointer_verification_ok": True,
            "latest_pointer_verification_status": "passed",
            "latest_pointer_verification_issue_count": 0,
            "latest_pointer_verification": {
                "json": str(latest_pointer_verify_json),
                "ok": True,
                "status": "passed",
                "issue_count": 0,
            },
            "bundle_integrity_ok": True,
            "bundle": {
                "manifest": str(manifest),
                "output_dir": str(manifest.parent),
            },
            "bundle_verification": {
                "json": str(verify_json),
                "schema_version": 1,
                "ok": True,
                "status": "passed",
                "integrity_ok": True,
                "readiness_ok": False,
                "readiness_status": "not_ready",
                "manifest": str(manifest),
                "bundle_dir": str(manifest.parent),
                "checked_file_count": 12,
                "error_count": 0,
                "errors": [],
                "require_ready": False,
            },
        },
    )
    write_json(
        verify_json,
        {
            "schema_version": 1,
            "ok": True,
            "status": "passed",
            "integrity_ok": True,
            "readiness_ok": False,
            "readiness_status": "not_ready",
            "manifest": str(manifest),
            "bundle_dir": str(manifest.parent),
            "checked_file_count": 12,
            "error_count": 0,
            "errors": [],
            "require_ready": False,
        },
    )
    gate_payload = json.loads(gate.read_text(encoding="utf-8"))
    operator_plan_payload = {
        "schema_version": 1,
        "timestamp": timestamp,
        "generated_at": 1,
        "status": "pending",
        "source_release_gate_json": str(gate),
        "release_gate_json": str(gate),
        "release_gate_status": gate_payload["status"],
        "release_ready": gate_payload["release_ready"],
        "readiness_status": gate_payload["readiness_status"],
        "readiness_ok": gate_payload["readiness_ok"],
        "readiness_report_source": gate_payload["readiness_report_source"],
        "blocking_gates": gate_payload["blocking_gates"],
        "operator_next_steps": gate_payload["operator_next_steps"],
        "external_action_checklist": gate_payload["external_action_checklist"],
        "wandb_adoption_draft": gate_payload["wandb_adoption_draft"],
        "paid_run_review_package": gate_payload["paid_run_review_package"],
        "wandb_completion_contract": gate_payload["wandb_completion_contract"],
        "benchmark_progress_matrix": gate_payload.get("benchmark_progress_matrix"),
        "nemoclaw_adoption": gate_payload["nemoclaw_adoption"],
        "outputs": {
            "json": str(operator_plan_json),
            "markdown": str(operator_plan_markdown),
        },
    }
    write_json(operator_plan_json, operator_plan_payload)
    operator_plan_markdown.write_text(
        "\n".join(
            [
                "# Taiwan Release Operator Plan",
                "",
                "Status: `pending`",
                f"Timestamp: `{timestamp}`",
                f"Source release gate JSON: `{gate}`",
                "Release gate: `not_ready`",
                "Release ready: `false`",
                "Readiness status: `not_ready`",
                "Readiness OK: `false`",
                f"Readiness report: `{readiness_path}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    current_gate_fields = [
        "readiness_report_schema_version",
        "status",
        "readiness_status",
        "readiness_ok",
        "gate_count",
        "blocker_count",
        "blocking_gates",
        "required_next_actions",
        "benchmark_completion",
        "weave_agents_completion",
        "existing_results_formalization",
        "wandb_adoption_draft",
        "paid_run_review_package",
        "wandb_completion_contract",
        "benchmark_progress_matrix",
        "nemoclaw_adoption",
        "operator_next_steps",
        "external_action_checklist",
    ]
    current_gate = {field: gate_payload.get(field) for field in current_gate_fields}
    current_gate["readiness_report_source"] = gate_payload["readiness_report_source"]
    current_gate["runner_evidence"] = {
        "name": "run_taiwan_release_gate.py",
        "report_json": gate_payload["readiness_report_source"],
    }
    write_json(
        manifest,
        {
            "schema_version": 1,
            "bundle_version": 2,
            "status": gate_payload["status"],
            "readiness_status": gate_payload["readiness_status"],
            "readiness_ok": gate_payload["readiness_ok"],
            "gate_count": gate_payload["gate_count"],
            "blocker_count": gate_payload["blocker_count"],
            "blocking_gates": gate_payload["blocking_gates"],
            "current_gate": current_gate,
        },
    )
    pointer = write_json(
        tmp_path / "latest_taiwan_release_gate.json",
        {
            "schema_version": 1,
            "generated_at": 1,
            "timestamp": timestamp,
            "ok": False,
            "status": "not_ready",
            "release_ready": False,
            "readiness_ok": False,
            "readiness_report_schema_version": 1,
            "gate_count": 3,
            "blocker_count": 1,
            "readiness_status": "not_ready",
            "blocking_gates": ["weave_content_canary"],
            "blockers": ["weave_content_canary"],
            "required_next_actions": [
                {
                    "gate": "weave_content_canary",
                    "status": "failed",
                    "detail": "missing canary",
                    "next_action": "rerun canary",
                }
            ],
            "benchmark_completion": [{"benchmark": "agentic_math", "completion_proven": False}],
            "weave_agents_completion": [{"gate": "paid_run_review_package", "entry_count": 0}],
            "existing_results_formalization": {"status": "passed"},
            "wandb_adoption_draft": {"status": "pending"},
            "paid_run_review_package": {"status": "not_ready"},
            "wandb_completion_contract": {"status": "incomplete"},
            "nemoclaw_adoption": {"status": "not_installed"},
            "operator_next_steps": {"status": "pending", "step_count": 1},
            "operator_plan": operator_plan,
            "external_action_checklist": external_action_checklist,
            "external_action_approval_packet": external_action_approval_packet,
            "release_gate_json": str(gate),
            "latest_pointer_json": str(tmp_path / "latest_taiwan_release_gate.json"),
            "latest_pointer_verification_json": str(latest_pointer_verify_json),
            "latest_pointer_verification_ok": True,
            "latest_pointer_verification_status": "passed",
            "latest_pointer_verification_issue_count": 0,
            "latest_pointer_verification": {
                "json": str(latest_pointer_verify_json),
                "ok": True,
                "status": "passed",
                "issue_count": 0,
            },
            "readiness_report": str(tmp_path / "readiness.json"),
            "readiness_report_source": str(tmp_path / "readiness.json"),
            "bundle_manifest": str(manifest),
            "bundle_output_dir": str(manifest.parent),
            "bundle_verification_json": str(verify_json),
            "bundle_integrity_ok": True,
            "bundle_file_count": 12,
            "bundle_missing_file_count": 0,
            "checked_file_count": 12,
            "verification_error_count": 0,
        },
    )
    write_json(
        latest_pointer_verify_json,
        {
            "schema_version": 1,
            "ok": True,
            "status": "passed",
            "issue_count": 0,
            "issues": [],
            "pointer_json": str(pointer),
            "release_gate_json": str(gate),
        },
    )
    release_gate_pointer = {
        "release_gate_json": str(gate),
        "latest_pointer_json": str(pointer),
        "latest_pointer_verification_json": str(latest_pointer_verify_json),
        "latest_pointer_verification_ok": True,
        "latest_pointer_verification_status": "passed",
        "latest_pointer_verification_issue_count": 0,
        "operator_plan": operator_plan,
    }
    proof = write_json(
        manifest.parent / "release_gate_pointer_proof.json",
        {
            "schema_version": 1,
            "kind": "release_gate_pointer_proof",
            "generated_at": 1,
            "ok": True,
            "status": "passed",
            "timestamp": timestamp,
            "release_gate_status": "not_ready",
            "release_ready": False,
            "readiness_status": "not_ready",
            "readiness_ok": False,
            "blocking_gates": ["weave_content_canary"],
            "release_gate_pointer": release_gate_pointer,
            "latest_pointer_verification": {
                "json": str(latest_pointer_verify_json),
                "ok": True,
                "status": "passed",
                "issue_count": 0,
            },
        },
    )
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["release_gate_pointer"] = release_gate_pointer
    manifest_payload["release_gate_pointer_proof"] = {
        "json": "release_gate_pointer_proof.json",
        "schema_version": 1,
        "status": "passed",
        "ok": True,
    }
    manifest_payload["files"] = [
        {
            "source_path": str(proof),
            "roles": ["release_gate_pointer", "release_gate_pointer:proof_json"],
            "exists": True,
            "bundle_path": "release_gate_pointer_proof.json",
            "size_bytes": proof.stat().st_size,
            "sha256": "0" * 64,
        }
    ]
    write_json(manifest, manifest_payload)
    return pointer, gate


def test_validate_pointer_accepts_matching_gate(tmp_path):
    module = load_module()
    pointer, _ = write_pointer_fixture(tmp_path)

    result = module.validate_pointer(pointer)

    assert result["ok"] is True
    assert result["schema_version"] == 1
    assert result["status"] == "passed"
    assert result["issues"] == []
    assert result["pointer_json"].endswith("latest_taiwan_release_gate.json")


def test_validate_pointer_rejects_self_path_mismatch(tmp_path):
    module = load_module()
    pointer, _ = write_pointer_fixture(tmp_path)
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    payload["latest_pointer_json"] = str(tmp_path / "other_latest.json")
    write_json(pointer, payload)

    result = module.validate_pointer(pointer)

    assert result["ok"] is False
    assert "pointer latest_pointer_json does not point to this pointer" in result["issues"]


def test_validate_pointer_rejects_non_timestamped_gate(tmp_path):
    module = load_module()
    pointer, gate = write_pointer_fixture(tmp_path)
    bad_gate = tmp_path / "release_gate.json"
    bad_gate.write_text(gate.read_text(encoding="utf-8"), encoding="utf-8")
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    payload["release_gate_json"] = str(bad_gate)
    write_json(pointer, payload)

    result = module.validate_pointer(pointer)

    assert result["ok"] is False
    assert any("formal taiwan_release_gate" in issue for issue in result["issues"])


def test_validate_pointer_rejects_count_mismatch(tmp_path):
    module = load_module()
    pointer, _ = write_pointer_fixture(tmp_path)
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    payload["checked_file_count"] = 999
    write_json(pointer, payload)

    result = module.validate_pointer(pointer)

    assert result["ok"] is False
    assert "pointer checked_file_count does not match release gate verification" in result["issues"]


def test_validate_pointer_rejects_bundle_verification_json_mismatch(tmp_path):
    module = load_module()
    pointer, _ = write_pointer_fixture(tmp_path)
    pointer_payload = json.loads(pointer.read_text(encoding="utf-8"))
    verification_json = Path(pointer_payload["bundle_verification_json"])
    verification_payload = json.loads(verification_json.read_text(encoding="utf-8"))
    verification_payload["checked_file_count"] = 999
    write_json(verification_json, verification_payload)

    result = module.validate_pointer(pointer)

    assert result["ok"] is False
    assert (
        "bundle verification JSON checked_file_count does not match release gate payload"
        in result["issues"]
    )
    assert (
        "pointer checked_file_count does not match bundle verification JSON"
        in result["issues"]
    )


def test_validate_pointer_rejects_bundle_verification_json_manifest_mismatch(tmp_path):
    module = load_module()
    pointer, _ = write_pointer_fixture(tmp_path)
    pointer_payload = json.loads(pointer.read_text(encoding="utf-8"))
    verification_json = Path(pointer_payload["bundle_verification_json"])
    verification_payload = json.loads(verification_json.read_text(encoding="utf-8"))
    verification_payload["manifest"] = str(tmp_path / "other_bundle" / "manifest.json")
    write_json(verification_json, verification_payload)

    result = module.validate_pointer(pointer)

    assert result["ok"] is False
    assert (
        "bundle verification JSON manifest does not match release gate bundle manifest"
        in result["issues"]
    )
    assert "bundle verification JSON manifest does not match pointer" in result["issues"]


def test_validate_pointer_rejects_bundle_verification_json_schema_status_and_error_count(
    tmp_path,
):
    module = load_module()
    pointer, _ = write_pointer_fixture(tmp_path)
    pointer_payload = json.loads(pointer.read_text(encoding="utf-8"))
    verification_json = Path(pointer_payload["bundle_verification_json"])
    verification_payload = json.loads(verification_json.read_text(encoding="utf-8"))
    verification_payload.pop("schema_version")
    verification_payload["status"] = "failed"
    verification_payload["error_count"] = 1
    write_json(verification_json, verification_payload)

    result = module.validate_pointer(pointer)

    assert result["ok"] is False
    assert "bundle verification JSON schema_version must be 1" in result["issues"]
    assert "bundle verification JSON status does not match ok" in result["issues"]
    assert (
        "bundle verification JSON error_count does not match errors"
        in result["issues"]
    )
    assert (
        "bundle verification JSON error_count does not match release gate payload"
        in result["issues"]
    )


def test_validate_pointer_rejects_latest_pointer_verification_summary_mismatch(tmp_path):
    module = load_module()
    pointer, gate = write_pointer_fixture(tmp_path)
    pointer_payload = json.loads(pointer.read_text(encoding="utf-8"))
    verification_json = tmp_path / "latest_pointer_verify.json"
    write_json(
        verification_json,
        {
            "schema_version": 1,
            "ok": True,
            "status": "passed",
            "issue_count": 0,
            "pointer_json": str(pointer),
            "release_gate_json": str(gate),
        },
    )
    summary = {
        "json": str(verification_json),
        "ok": True,
        "status": "passed",
        "issue_count": 0,
    }
    gate_payload = json.loads(gate.read_text(encoding="utf-8"))
    gate_payload["latest_pointer_verification_json"] = str(verification_json)
    gate_payload["latest_pointer_verification_ok"] = True
    gate_payload["latest_pointer_verification_status"] = "passed"
    gate_payload["latest_pointer_verification_issue_count"] = 0
    gate_payload["latest_pointer_verification"] = summary
    write_json(gate, gate_payload)
    pointer_payload["latest_pointer_verification_json"] = str(verification_json)
    pointer_payload["latest_pointer_verification_ok"] = False
    pointer_payload["latest_pointer_verification_status"] = "failed"
    pointer_payload["latest_pointer_verification_issue_count"] = 1
    pointer_payload["latest_pointer_verification"] = {
        **summary,
        "ok": False,
        "status": "failed",
        "issue_count": 1,
    }
    write_json(pointer, pointer_payload)

    result = module.validate_pointer(pointer)

    assert result["ok"] is False
    assert (
        "pointer latest_pointer_verification_ok does not match release gate payload"
        in result["issues"]
    )
    assert (
        "pointer latest_pointer_verification_ok does not match latest pointer verification JSON"
        in result["issues"]
    )


def test_validate_pointer_rejects_manifest_current_gate_mismatch(tmp_path):
    module = load_module()
    pointer, _ = write_pointer_fixture(tmp_path)
    pointer_payload = json.loads(pointer.read_text(encoding="utf-8"))
    manifest = Path(pointer_payload["bundle_manifest"])
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["current_gate"]["nemoclaw_adoption"]["status"] = "ready"
    write_json(manifest, manifest_payload)

    result = module.validate_pointer(pointer)

    assert result["ok"] is False
    assert (
        "bundle manifest current_gate nemoclaw_adoption does not match release gate payload"
        in result["issues"]
    )


def test_validate_pointer_rejects_missing_manifest_runner_evidence(tmp_path):
    module = load_module()
    pointer, _ = write_pointer_fixture(tmp_path)
    pointer_payload = json.loads(pointer.read_text(encoding="utf-8"))
    manifest = Path(pointer_payload["bundle_manifest"])
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["current_gate"].pop("runner_evidence")
    write_json(manifest, manifest_payload)

    result = module.validate_pointer(pointer)

    assert result["ok"] is False
    assert "bundle manifest current_gate runner_evidence is required" in result["issues"]


def test_validate_pointer_rejects_missing_release_gate_pointer_proof(tmp_path):
    module = load_module()
    pointer, _ = write_pointer_fixture(tmp_path)
    pointer_payload = json.loads(pointer.read_text(encoding="utf-8"))
    manifest = Path(pointer_payload["bundle_manifest"])
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload.pop("release_gate_pointer_proof")
    write_json(manifest, manifest_payload)

    result = module.validate_pointer(pointer)

    assert result["ok"] is False
    assert "bundle manifest release_gate_pointer_proof is required" in result["issues"]


def test_validate_pointer_rejects_release_gate_pointer_proof_mismatch(tmp_path):
    module = load_module()
    pointer, _ = write_pointer_fixture(tmp_path)
    pointer_payload = json.loads(pointer.read_text(encoding="utf-8"))
    manifest = Path(pointer_payload["bundle_manifest"])
    proof = manifest.parent / "release_gate_pointer_proof.json"
    proof_payload = json.loads(proof.read_text(encoding="utf-8"))
    proof_payload["latest_pointer_verification"]["issue_count"] = 1
    proof_payload["latest_pointer_verification"]["status"] = "failed"
    write_json(proof, proof_payload)

    result = module.validate_pointer(pointer)

    assert result["ok"] is False
    assert (
        "release gate pointer proof JSON latest_pointer_verification status must be passed"
        in result["issues"]
    )
    assert (
        "release gate pointer proof JSON latest_pointer_verification issue_count must be 0"
        in result["issues"]
    )


def test_validate_pointer_rejects_legacy_release_gate_schema(tmp_path):
    module = load_module()
    pointer, gate = write_pointer_fixture(tmp_path)
    payload = json.loads(gate.read_text(encoding="utf-8"))
    payload.pop("schema_version")
    write_json(gate, payload)

    result = module.validate_pointer(pointer)

    assert result["ok"] is False
    assert "release gate schema_version must be 1" in result["issues"]


def test_validate_pointer_rejects_summary_mismatch(tmp_path):
    module = load_module()
    pointer, _ = write_pointer_fixture(tmp_path)
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    payload["operator_next_steps"]["step_count"] = 99
    payload["operator_plan"]["status"] = "stale"
    payload["external_action_checklist"]["item_count"] = 99
    payload["external_action_approval_packet"]["required_approval_count"] = 99
    payload["required_next_actions"] = []
    write_json(pointer, payload)

    result = module.validate_pointer(pointer)

    assert result["ok"] is False
    assert "pointer operator_next_steps does not match release gate payload" in result["issues"]
    assert "pointer operator_plan does not match release gate payload" in result["issues"]
    assert "pointer external_action_checklist does not match release gate payload" in result["issues"]
    assert "pointer external_action_approval_packet does not match release gate payload" in result["issues"]
    assert "pointer required_next_actions does not match release gate payload" in result["issues"]


def test_validate_pointer_rejects_missing_operator_plan_file(tmp_path):
    module = load_module()
    pointer, _ = write_pointer_fixture(tmp_path)
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    operator_plan_json = Path(payload["operator_plan"]["json"])
    operator_plan_json.unlink()

    result = module.validate_pointer(pointer)

    assert result["ok"] is False
    assert any("operator plan JSON does not exist" in issue for issue in result["issues"])


def test_validate_pointer_rejects_operator_plan_release_gate_mismatch(tmp_path):
    module = load_module()
    pointer, _ = write_pointer_fixture(tmp_path)
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    operator_plan_json = Path(payload["operator_plan"]["json"])
    operator_payload = json.loads(operator_plan_json.read_text(encoding="utf-8"))
    operator_payload["source_release_gate_json"] = str(tmp_path / "other_gate.json")
    operator_payload["operator_next_steps"] = {"status": "pending", "step_count": 99}
    write_json(operator_plan_json, operator_payload)

    result = module.validate_pointer(pointer)

    assert result["ok"] is False
    assert (
        "operator plan JSON source_release_gate_json does not match release gate"
        in result["issues"]
    )
    assert (
        "operator plan JSON operator_next_steps does not match release gate payload"
        in result["issues"]
    )


def test_cli_writes_json_and_exits_nonzero_when_invalid_requested(tmp_path):
    pointer, _ = write_pointer_fixture(tmp_path)
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    payload["verification_error_count"] = 1
    write_json(pointer, payload)
    output = tmp_path / "pointer_check.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--pointer-json",
            str(pointer),
            "--json",
            str(output),
            "--fail-on-invalid",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert "pointer verification_error_count does not match release gate verification" in report["issues"]
