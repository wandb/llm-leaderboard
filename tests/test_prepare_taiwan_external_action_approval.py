import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tools" / "prepare_taiwan_external_action_approval.py"
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
    counts = {
        "paid_api": 1,
        "wandb_access": 1,
        "wandb_write": 1,
        "third_party_acceptance": 0,
        "nemoclaw_install": 0,
        "scope_confirmation": 0,
    }
    checklist = {
        "schema_version": 1,
        "status": "pending",
        "external_action_item_count": 1,
        "requirement_counts": counts,
        "items": [
            {
                "gate": "weave_content_canary",
                "requirements": ["paid_api", "wandb_access", "wandb_write"],
            }
        ],
    }
    requirements = []
    for name, label in verifier.EXTERNAL_ACTION_REQUIREMENTS:
        base = verifier.expected_requirement_base(
            requirement=name,
            label=label,
            count=counts[name],
            checklist=checklist,
        )
        base["approval_status"] = "not_granted" if base["required"] else "not_required"
        requirements.append(base)
    return {
        "schema_version": 1,
        "generated_at": 1782918000.0,
        "status": "pending_approval",
        "readiness_status": "not_ready",
        "readiness_ok": False,
        "blocking_gates": ["weave_content_canary"],
        "source": {"manifest": "manifest.json"},
        "external_action_checklist_sha256": verifier.canonical_json_sha256(checklist),
        "external_action_checklist": checklist,
        "approval_requirement_count": len(requirements),
        "required_approval_count": 3,
        "all_required_approvals_granted": False,
        "approval_requirements": requirements,
        "approval_verifier": {
            "schema_version": 1,
            "status": "available",
            "script": "scripts/tools/verify_external_action_approval_packet.py",
            "required_before_external_action": True,
            "source_packet_json": "external_action_approval_packet.json",
            "reviewed_packet_json_template": (
                "temp/taiwan_external_action_approval_REVIEWED_YYYYMMDDTHHMM.json"
            ),
            "report_json_template": (
                "temp/taiwan_external_action_approval_REVIEWED_YYYYMMDDTHHMM.verify.json"
            ),
            "command_template": (
                "uv run python scripts/tools/verify_external_action_approval_packet.py "
                "--approval-packet-json "
                "temp/taiwan_external_action_approval_REVIEWED_YYYYMMDDTHHMM.json "
                "--source-packet-json external_action_approval_packet.json "
                "--require-approved "
                "--json temp/taiwan_external_action_approval_REVIEWED_YYYYMMDDTHHMM.verify.json"
            ),
            "safety": {
                "executes_external_action": False,
                "queries_wandb": False,
                "writes_wandb": False,
                "installs_third_party": False,
                "launches_model_inference": False,
            },
        },
        "approval_template_renderer": {
            "schema_version": 1,
            "status": "available",
            "script": "scripts/tools/render_external_action_approval_template.py",
            "required_before_external_action": True,
            "reviewed_packet_json_template": (
                "temp/taiwan_external_action_approval_REVIEWED_YYYYMMDDTHHMM.json"
            ),
            "reviewed_packet_markdown_template": (
                "temp/taiwan_external_action_approval_REVIEWED_YYYYMMDDTHHMM.md"
            ),
            "command_template": (
                "uv run python scripts/tools/render_external_action_approval_template.py "
                "--approval-packet-json external_action_approval_packet.json "
                "--output-json "
                "temp/taiwan_external_action_approval_REVIEWED_YYYYMMDDTHHMM.json "
                "--markdown temp/taiwan_external_action_approval_REVIEWED_YYYYMMDDTHHMM.md"
            ),
            "safety": {
                "executes_external_action": False,
                "queries_wandb": False,
                "writes_wandb": False,
                "installs_third_party": False,
                "launches_model_inference": False,
            },
        },
        "execution_policy": {
            "this_packet_launches_external_actions": False,
            "reviewed_copy_required_before_external_action": True,
        },
        "outputs": {
            "json": "external_action_approval_packet.json",
            "markdown": "external_action_approval_packet.md",
        },
    }


def test_prepare_external_action_approval_handoff_keeps_unapproved_template_pending(tmp_path):
    bundle_dir = tmp_path / "bundle"
    packet_path = write_json(
        bundle_dir / "external_action_approval_packet.json",
        approval_packet(),
    )
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--bundle-dir",
            str(bundle_dir),
            "--timestamp",
            "TESTSTAMP",
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    handoff = json.loads(result.stdout)
    assert handoff["ok"] is True
    assert handoff["status"] == "pending_human_approval"
    assert handoff["will_execute_external_actions"] is False
    assert handoff["source"]["approval_packet_json"] == str(packet_path)
    assert len(handoff["source"]["approval_packet_sha256"]) == 64
    assert handoff["required_approval_count"] == 3
    assert handoff["granted_approval_count"] == 0
    assert handoff["all_required_approvals_granted"] is False
    assert all(value is False for value in handoff["safety"].values())

    reviewed_json = Path(handoff["paths"]["reviewed_json"])
    verify_report_json = Path(handoff["paths"]["verify_report_json"])
    handoff_json = Path(handoff["paths"]["handoff_json"])
    handoff_markdown = Path(handoff["paths"]["handoff_markdown"])
    assert reviewed_json.exists()
    assert verify_report_json.exists()
    assert handoff_json.exists()
    assert handoff_markdown.exists()
    verify_report = json.loads(verify_report_json.read_text(encoding="utf-8"))
    assert verify_report["ok"] is False
    assert verify_report["status"] == "validation_failed"
    assert verify_report["source_binding"]["bound"] is True
    assert "paid_api.approval_status must be granted" in verify_report["errors"]

    reviewed = json.loads(reviewed_json.read_text(encoding="utf-8"))
    assert reviewed["status"] == "pending_approval"
    assert reviewed["all_required_approvals_granted"] is False
    assert reviewed["approval_template"]["source_approval_packet_sha256"] == handoff[
        "source"
    ]["approval_packet_sha256"]
    assert "Verifier ok: `False`" in handoff_markdown.read_text(encoding="utf-8")


def test_prepare_external_action_approval_handoff_resolves_latest_pointer(tmp_path):
    bundle_dir = tmp_path / "bundle"
    write_json(bundle_dir / "external_action_approval_packet.json", approval_packet())
    latest_pointer = write_json(
        tmp_path / "latest_taiwan_release_gate.json",
        {
            "schema_version": 1,
            "timestamp": "20260701T145711Z",
            "status": "not_ready",
            "bundle_output_dir": str(bundle_dir),
        },
    )

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--latest-pointer-json",
            str(latest_pointer),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    handoff = json.loads(result.stdout)
    assert handoff["source"]["latest_pointer_timestamp"] == "20260701T145711Z"
    assert handoff["source"]["bundle_dir"] == str(bundle_dir)
    assert handoff["status"] == "pending_human_approval"
