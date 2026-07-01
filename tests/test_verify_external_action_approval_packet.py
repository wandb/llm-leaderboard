import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tools" / "verify_external_action_approval_packet.py"


def load_module():
    spec = importlib.util.spec_from_file_location(SCRIPT.stem, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[SCRIPT.stem] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def approved_item(
    *,
    requirement: str,
    label: str,
    count: int,
    reviewer_fields: list[str],
    extra: dict,
) -> dict:
    return {
        "requirement": requirement,
        "label": label,
        "count": count,
        "required": count > 0,
        "approval_status": "granted" if count > 0 else "not_required",
        "required_before_gates": ["all"],
        "reviewer_fields": reviewer_fields,
        **(
            {
                "approved_by": "yuya",
                "approved_at": "2026-06-29T01:30:00+09:00",
                "approval_reference": "internal-approval-2026-06-29",
            }
            if count > 0
            else {}
        ),
        **extra,
    }


def approval_packet(tmp_path: Path, *, approved: bool = True) -> dict:
    module = load_module()
    installer_sha = "a" * 64
    lock_json = write_json(
        tmp_path / "nemoclaw_installer_lock.json",
        {
            "schema_version": 1,
            "installer_url": "https://www.nvidia.com/nemoclaw.sh",
            "install_ref": "lkg",
            "sha256": installer_sha,
            "will_execute_installer": False,
            "will_install_or_onboard": False,
            "will_launch_model_inference": False,
            "will_query_wandb": False,
        },
    )
    review_json = write_json(
        tmp_path / "paid_run_review.json",
        {"schema_version": 1, "status": "reviewed"},
    )
    completion_json = write_json(
        tmp_path / "wandb_completion.json",
        {
            "verification_schema_version": 1,
            "ok": True,
            "benchmark": "agentic_math",
            "entity": "llm-leaderboard",
            "project": "tc-leaderboard",
            "run_id": "f1veetyb",
        },
    )
    scope_attestation = write_json(
        tmp_path / "scope_attestation.json",
        {
            "schema_version": 1,
            "confirmed": True,
            "confirmed_by": "yuya",
            "confirmed_at": "2026-06-29T01:30:00+09:00",
            "confirmation": "Reviewed existing W&B run f1veetyb for the Taiwan canary scope.",
            "actual_cost_estimate": "$12.34",
            "provider_bill_reference": "openai-dashboard-2026-06-29",
            "benchmark": "agentic_math",
            "entity": "llm-leaderboard",
            "project": "tc-leaderboard",
            "run_id": "f1veetyb",
            "completion_path": str(completion_json),
            "completion_sha256": module.sha256_file(completion_json),
            "review_path": str(review_json),
        },
    )
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
        "items": [{"gate": "all", "requirements": requirements}],
    }
    approval_requirements = [
        approved_item(
            requirement="paid_api",
            label="Paid API",
            count=1,
            reviewer_fields=[
                "approved_by",
                "approved_at",
                "approval_reference",
                "approved_budget_usd",
                "approved_model_scope",
            ],
            extra={"approved_budget_usd": 25.0, "approved_model_scope": "OpenAI mini canary"},
        ),
        approved_item(
            requirement="wandb_access",
            label="W&B access",
            count=1,
            reviewer_fields=["approved_by", "approved_at", "approval_reference"],
            extra={},
        ),
        approved_item(
            requirement="wandb_write",
            label="W&B write",
            count=1,
            reviewer_fields=[
                "approved_by",
                "approved_at",
                "approval_reference",
                "approved_wandb_entity",
                "approved_wandb_project",
            ],
            extra={"approved_wandb_entity": "llm-leaderboard", "approved_wandb_project": "tc-leaderboard"},
        ),
        approved_item(
            requirement="third_party_acceptance",
            label="Third-party acceptance",
            count=1,
            reviewer_fields=[
                "approved_by",
                "approved_at",
                "approval_reference",
                "third_party_terms_reviewed",
            ],
            extra={"third_party_terms_reviewed": True},
        ),
        approved_item(
            requirement="nemoclaw_install",
            label="NeMoClaw install",
            count=1,
            reviewer_fields=[
                "approved_by",
                "approved_at",
                "approval_reference",
                "installer_lock_json",
                "installer_sha256",
                "sandbox",
            ],
            extra={
                "installer_lock_json": str(lock_json),
                "installer_sha256": installer_sha,
                "sandbox": "nejumi-taiwan",
            },
        ),
        approved_item(
            requirement="scope_confirmation",
            label="Scope confirmation",
            count=1,
            reviewer_fields=[
                "approved_by",
                "approved_at",
                "approval_reference",
                "scope_attestation_json",
            ],
            extra={"scope_attestation_json": str(scope_attestation)},
        ),
    ]
    if not approved:
        approval_requirements[0]["approval_status"] = "not_granted"
        approval_requirements[0]["approved_budget_usd"] = "$APPROVED_BUDGET_USD"
    all_granted = approved
    return {
        "schema_version": 1,
        "status": "approved" if all_granted else "pending_approval",
        "external_action_checklist_sha256": module.canonical_json_sha256(checklist),
        "external_action_checklist": checklist,
        "approval_requirement_count": len(approval_requirements),
        "required_approval_count": len(requirements),
        "all_required_approvals_granted": all_granted,
        "approval_requirements": approval_requirements,
    }


def test_verify_external_action_approval_packet_passes_for_reviewed_packet(tmp_path):
    module = load_module()
    packet_path = write_json(tmp_path / "approval_packet.json", approval_packet(tmp_path))

    report = module.verify_approval_packet(packet_path)

    assert report["ok"] is True
    assert report["status"] == "approved"
    assert report["granted_approval_count"] == report["required_approval_count"]
    assert report["will_execute_external_actions"] is False
    assert report["errors"] == []
    paid_api = next(
        item for item in report["approval_results"] if item["requirement"] == "paid_api"
    )
    assert paid_api["approved_budget_usd"] == 25.0
    assert paid_api["approved_model_scope"] == "OpenAI mini canary"


def test_verify_external_action_approval_packet_passes_with_source_binding(tmp_path):
    module = load_module()
    source_packet = approval_packet(tmp_path, approved=False)
    source_packet["blocking_gates"] = ["wandb_completion"]
    source_path = write_json(tmp_path / "source_packet.json", source_packet)
    reviewed_packet = approval_packet(tmp_path, approved=True)
    reviewed_packet["blocking_gates"] = source_packet["blocking_gates"]
    reviewed_packet["approval_template"] = {
        "schema_version": 1,
        "source_approval_packet_json": str(source_path),
        "source_approval_packet_sha256": module.sha256_file(source_path),
        "source_external_action_checklist_sha256": source_packet[
            "external_action_checklist_sha256"
        ],
        "will_execute_external_actions": False,
    }
    reviewed_path = write_json(tmp_path / "reviewed_packet.json", reviewed_packet)

    report = module.verify_approval_packet(
        reviewed_path,
        source_packet_path=source_path,
    )

    assert report["ok"] is True
    assert report["source_binding"]["bound"] is True
    assert report["source_binding"]["source_approval_packet_sha256"] == module.sha256_file(
        source_path
    )


def test_verify_external_action_approval_packet_rejects_source_binding_mismatch(tmp_path):
    module = load_module()
    source_packet = approval_packet(tmp_path, approved=False)
    source_packet["blocking_gates"] = ["wandb_completion"]
    source_path = write_json(tmp_path / "source_packet.json", source_packet)
    reviewed_packet = approval_packet(tmp_path, approved=True)
    reviewed_packet["blocking_gates"] = ["one_model_full_canary"]
    reviewed_packet["approval_template"] = {
        "schema_version": 1,
        "source_approval_packet_json": str(source_path),
        "source_approval_packet_sha256": module.sha256_file(source_path),
        "source_external_action_checklist_sha256": source_packet[
            "external_action_checklist_sha256"
        ],
        "will_execute_external_actions": False,
    }
    reviewed_path = write_json(tmp_path / "reviewed_packet.json", reviewed_packet)

    report = module.verify_approval_packet(
        reviewed_path,
        source_packet_path=source_path,
    )

    assert report["ok"] is False
    assert report["source_binding"]["bound"] is False
    assert "blocking_gates does not match source approval packet" in report["errors"]


def test_verify_external_action_approval_packet_rejects_unbound_scope_attestation(tmp_path):
    module = load_module()
    packet = approval_packet(tmp_path)
    scope_path = Path(
        next(
            item["scope_attestation_json"]
            for item in packet["approval_requirements"]
            if item["requirement"] == "scope_confirmation"
        )
    )
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    scope["completion_sha256"] = "b" * 64
    scope_path.write_text(json.dumps(scope), encoding="utf-8")
    packet_path = write_json(tmp_path / "approval_packet.json", packet)

    report = module.verify_approval_packet(packet_path)

    assert report["ok"] is False
    assert (
        "scope_confirmation scope_attestation_json completion_sha256 "
        "does not match completion_path"
    ) in report["errors"]


def test_cli_rejects_unapproved_placeholder_packet(tmp_path):
    packet_path = write_json(
        tmp_path / "approval_packet.json",
        approval_packet(tmp_path, approved=False),
    )
    report_path = tmp_path / "approval_report.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--approval-packet-json",
            str(packet_path),
            "--require-approved",
            "--json",
            str(report_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert "paid_api.approval_status must be granted" in report["errors"]
    assert "paid_api.approved_budget_usd must be a positive USD number" in report["errors"]
