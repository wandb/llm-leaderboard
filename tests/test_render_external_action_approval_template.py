import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RENDER_SCRIPT = REPO_ROOT / "scripts" / "tools" / "render_external_action_approval_template.py"
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "tools" / "verify_external_action_approval_packet.py"


def load_verifier_module():
    spec = importlib.util.spec_from_file_location(VERIFY_SCRIPT.stem, VERIFY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[VERIFY_SCRIPT.stem] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def approval_packet() -> dict:
    verifier = load_verifier_module()
    requirements = [
        "paid_api",
        "wandb_access",
        "wandb_write",
        "third_party_acceptance",
        "nemoclaw_install",
        "scope_confirmation",
    ]
    checklist = {
        "schema_version": 1,
        "status": "pending",
        "external_action_item_count": 1,
        "requirement_counts": {name: 1 for name in requirements},
        "approval_requirement_constraints": {
            "paid_api": {
                "minimum_approved_budget_usd": 30.0,
                "minimum_approved_budget_source": "max_pre_run_budget_estimate_high",
            }
        },
        "items": [{"gate": "all", "requirements": requirements}],
    }
    reviewed_packet_template = "temp/taiwan_external_action_approval_REVIEWED.json"
    verifier_report_template = "temp/taiwan_external_action_approval_REVIEWED.verify.json"
    return {
        "schema_version": 1,
        "status": "pending_approval",
        "external_action_checklist_sha256": verifier.canonical_json_sha256(checklist),
        "external_action_checklist": checklist,
        "approval_requirement_count": 6,
        "required_approval_count": 6,
        "all_required_approvals_granted": False,
        "approval_requirements": [],
        "approval_verifier": {
            "schema_version": 1,
            "status": "available",
            "script": "scripts/tools/verify_external_action_approval_packet.py",
            "required_before_external_action": True,
            "source_packet_json": "external_action_approval_packet.json",
            "reviewed_packet_json_template": reviewed_packet_template,
            "report_json_template": verifier_report_template,
            "command_template": (
                "uv run python scripts/tools/verify_external_action_approval_packet.py "
                f"--approval-packet-json {reviewed_packet_template} "
                "--source-packet-json external_action_approval_packet.json "
                "--require-approved "
                f"--json {verifier_report_template}"
            ),
            "safety": {
                "executes_external_action": False,
                "queries_wandb": False,
                "writes_wandb": False,
                "installs_third_party": False,
                "launches_model_inference": False,
            },
        },
    }


def test_render_external_action_approval_template_does_not_grant_approvals(tmp_path):
    packet_path = write_json(tmp_path / "external_action_approval_packet.json", approval_packet())
    output_json = tmp_path / "reviewed_template.json"
    output_md = tmp_path / "reviewed_template.md"

    result = subprocess.run(
        [
            "python3",
            str(RENDER_SCRIPT),
            "--approval-packet-json",
            str(packet_path),
            "--output-json",
            str(output_json),
            "--markdown",
            str(output_md),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert report["will_execute_external_actions"] is False
    template = json.loads(output_json.read_text(encoding="utf-8"))
    assert template["status"] == "pending_approval"
    assert template["all_required_approvals_granted"] is False
    paid_api = next(
        item for item in template["approval_requirements"] if item["requirement"] == "paid_api"
    )
    assert paid_api["approval_status"] == "not_granted"
    assert paid_api["minimum_approved_budget_usd"] == 30.0
    assert paid_api["minimum_approved_budget_source"] == "max_pre_run_budget_estimate_high"
    assert paid_api["approved_budget_usd"] == "APPROVED_BUDGET_USD"
    assert paid_api["approved_model_scope"] == "APPROVED_MODEL_SCOPE"
    third_party = next(
        item
        for item in template["approval_requirements"]
        if item["requirement"] == "third_party_acceptance"
    )
    assert third_party["third_party_terms_reviewed"] is False
    assert template["approval_template"]["will_execute_external_actions"] is False
    assert len(template["approval_template"]["source_approval_packet_sha256"]) == 64
    markdown = output_md.read_text(encoding="utf-8")
    assert "Taiwan External Action Approval Reviewed Template" in markdown
    assert "Source approval packet SHA-256" in markdown
    assert "Minimum approved budget USD" in markdown
    assert "verify_external_action_approval_packet.py" in markdown
    assert f"--source-packet-json {packet_path}" in markdown
    assert f"--json {output_json.with_suffix('.verify.json')}" in markdown

    verify_result = subprocess.run(
        [
            "python3",
            str(VERIFY_SCRIPT),
            "--approval-packet-json",
            str(output_json),
            "--require-approved",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert verify_result.returncode == 1
    verify_report = json.loads(verify_result.stdout)
    assert verify_report["ok"] is False
    assert "paid_api.approval_status must be granted" in verify_report["errors"]


def test_render_external_action_approval_template_rejects_checklist_hash_mismatch(tmp_path):
    packet = approval_packet()
    packet["external_action_checklist"]["external_action_item_count"] = 99
    packet_path = write_json(tmp_path / "external_action_approval_packet.json", packet)

    result = subprocess.run(
        [
            "python3",
            str(RENDER_SCRIPT),
            "--approval-packet-json",
            str(packet_path),
            "--output-json",
            str(tmp_path / "reviewed_template.json"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "external_action_checklist_sha256 mismatch" in result.stderr
