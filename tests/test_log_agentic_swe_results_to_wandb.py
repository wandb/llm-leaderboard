import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = REPO_ROOT / "scripts" / "tools" / "log_agentic_swe_results_to_wandb.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def summary_payload():
    return {
        "total_instances": 3,
        "resolved_instances": 2,
        "unresolved_instances": 1,
        "pass_at_1": 2 / 3,
        "resolved_ids": ["i1", "i3"],
        "unresolved_ids": ["i2"],
    }


def patch_rows_with_nemoclaw_audit(*, tool_policy_ok: bool = True):
    return [
        {
            "instance_id": "i1",
            "tool_policy_ok": tool_policy_ok,
            "nemoclaw_session_audit_ok": True,
            "nemoclaw_session_audit": {"required": True, "ok": True},
        },
        {
            "instance_id": "i2",
            "tool_policy_ok": True,
            "nemoclaw_session_audit_ok": True,
            "nemoclaw_session_audit": {"required": True, "ok": True},
        },
        {
            "instance_id": "i3",
            "tool_policy_ok": True,
            "nemoclaw_session_audit_ok": True,
            "nemoclaw_session_audit": {"required": True, "ok": True},
        },
    ]


def test_validate_summary_accepts_consistent_eval_results():
    module = load_module()

    module.validate_summary(
        summary_payload(),
        eval_results={"i1": True, "i2": False, "i3": True},
        expected_total=3,
    )


def test_validate_summary_rejects_expected_total_mismatch():
    module = load_module()

    try:
        module.validate_summary(
            summary_payload(),
            eval_results={"i1": True, "i2": False, "i3": True},
            expected_total=80,
        )
    except ValueError as exc:
        assert "expected_total=80" in str(exc)
    else:
        raise AssertionError("expected summary validation failure")


def test_validate_summary_rejects_eval_result_id_mismatch():
    module = load_module()

    try:
        module.validate_summary(
            summary_payload(),
            eval_results={"i1": True, "i2": False, "other": True},
            expected_total=3,
        )
    except ValueError as exc:
        assert "eval_results.json instance ids do not match summary ids" in str(exc)
    else:
        raise AssertionError("expected summary validation failure")


def test_build_output_table_preserves_patch_metadata():
    module = load_module()

    table = module.build_output_table(
        summary_payload(),
        [
            {
                "instance_id": "i1",
                "openclaw_returncode": 0,
                "tool_policy_ok": True,
                "nemoclaw_session_audit_ok": True,
                "nemoclaw_session_audit": {"required": True, "ok": True},
                "openclaw_tool_call_count": 12,
                "openclaw_result_path": "path/to/result.json",
            }
        ],
    )

    rows = table.to_dict(orient="records")
    assert len(rows) == 3
    i1 = next(row for row in rows if row["instance_id"] == "i1")
    i2 = next(row for row in rows if row["instance_id"] == "i2")
    assert i1["resolved"] is True
    assert i1["has_patch_record"] is True
    assert i1["openclaw_tool_call_count"] == 12
    assert i1["nemoclaw_session_audit_ok"] is True
    assert i1["nemoclaw_session_audit_required"] is True
    assert i2["resolved"] is False
    assert i2["has_patch_record"] is False


def test_read_patch_rows_rejects_non_list(tmp_path):
    module = load_module()
    patch_path = tmp_path / "patches.json"
    patch_path.write_text(json.dumps({"instance_id": "i1"}), encoding="utf-8")

    try:
        module.read_patch_rows(patch_path)
    except ValueError as exc:
        assert "must contain a JSON list" in str(exc)
    else:
        raise AssertionError("expected patch row validation failure")


def test_main_dry_run_writes_plan_without_wandb_login(tmp_path, monkeypatch, capsys):
    module = load_module()
    official_eval_dir = tmp_path / "official"
    official_eval_dir.mkdir()
    patch_path = tmp_path / "patches.json"
    (official_eval_dir / "summary.json").write_text(
        json.dumps(summary_payload()),
        encoding="utf-8",
    )
    (official_eval_dir / "eval_results.json").write_text(
        json.dumps({"i1": True, "i2": False, "i3": True}),
        encoding="utf-8",
    )
    patch_path.write_text(
        json.dumps(
            patch_rows_with_nemoclaw_audit()
        ),
        encoding="utf-8",
    )
    plan_json = tmp_path / "plan.json"

    def fail_login():
        raise AssertionError("dry-run must not login to W&B")

    monkeypatch.setattr(module.wandb, "login", fail_login)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "log_agentic_swe_results_to_wandb.py",
            "--official-eval-dir",
            str(official_eval_dir),
            "--patch-path",
            str(patch_path),
            "--model-name",
            "test/model",
            "--expected-total",
            "3",
            "--dry-run",
            "--plan-json",
            str(plan_json),
        ],
    )

    module.main()
    stdout = json.loads(capsys.readouterr().out)
    plan = json.loads(plan_json.read_text(encoding="utf-8"))
    assert stdout == plan
    assert plan["will_write_wandb"] is False
    assert set(plan["source"]["source_sha256"]) == {
        "summary_json",
        "eval_results_json",
        "patch_path",
    }
    assert plan["config"]["relog"]["source_sha256"] == plan["source"]["source_sha256"]
    assert plan["would_log"]["tables"]["agentic_swe_output_table"] == 3
    assert (
        plan["would_log"]["summary_metrics"][
            "agentic_swe/nemoclaw_session_audit_required_patches"
        ]
        == 3
    )
    assert plan["would_log"]["artifact"]["aliases"] == ["latest", "production"]
    assert plan["external_action_approval"] == {
        "required_before_wandb_write": True,
        "required_report_option": "--external-action-approval-report-json",
        "required_source_packet_option": "--external-action-approval-source-packet-json",
        "required_report_status": "approved",
        "required_source_bound": True,
        "required_source_packet_sha256_match": True,
        "required_requirements": ["wandb_access", "wandb_write"],
        "target_entity": "llm-leaderboard",
        "target_project": "tc-leaderboard",
    }
    assert "--benchmark agentic_swe" in plan["post_log_verifier_command_template"]
    assert "--expected-total 3" in plan["post_log_verifier_command_template"]
    assert (
        "--expected-run-config relog.source_sha256.summary_json="
        in plan["post_log_verifier_command_template"]
    )
    assert (
        "--expected-run-config relog.source_sha256.eval_results_json="
        in plan["post_log_verifier_command_template"]
    )
    assert (
        "--expected-run-config relog.source_sha256.patch_path="
        in plan["post_log_verifier_command_template"]
    )
    assert "--require-nemoclaw-session-audit" in plan["post_log_verifier_command_template"]


def test_main_dry_run_validation_failure_writes_machine_readable_plan(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = load_module()
    official_eval_dir = tmp_path / "official"
    official_eval_dir.mkdir()
    patch_path = tmp_path / "patches.json"
    (official_eval_dir / "summary.json").write_text(
        json.dumps(summary_payload()),
        encoding="utf-8",
    )
    (official_eval_dir / "eval_results.json").write_text(
        json.dumps({"i1": True, "i2": False, "i3": True}),
        encoding="utf-8",
    )
    patch_path.write_text(json.dumps(patch_rows_with_nemoclaw_audit()), encoding="utf-8")
    plan_json = tmp_path / "validation_failed.json"

    def fail_login():
        raise AssertionError("validation-failed dry-run must not login to W&B")

    monkeypatch.setattr(module.wandb, "login", fail_login)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "log_agentic_swe_results_to_wandb.py",
            "--official-eval-dir",
            str(official_eval_dir),
            "--patch-path",
            str(patch_path),
            "--model-name",
            "test/model",
            "--expected-total",
            "80",
            "--dry-run",
            "--plan-json",
            str(plan_json),
        ],
    )

    try:
        module.main()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("expected dry-run validation failure")
    stdout = json.loads(capsys.readouterr().out)
    plan = json.loads(plan_json.read_text(encoding="utf-8"))
    assert stdout == plan
    assert plan["ok"] is False
    assert plan["status"] == "validation_failed"
    assert plan["will_write_wandb"] is False
    assert plan["expected_total"] == 80
    assert plan["external_action_approval"] == {
        "required_before_wandb_write": True,
        "required_report_option": "--external-action-approval-report-json",
        "required_source_packet_option": "--external-action-approval-source-packet-json",
        "required_report_status": "approved",
        "required_source_bound": True,
        "required_source_packet_sha256_match": True,
        "required_requirements": ["wandb_access", "wandb_write"],
        "target_entity": "llm-leaderboard",
        "target_project": "tc-leaderboard",
    }
    assert "expected_total=80" in plan["errors"][0]


def test_main_write_requires_validated_dry_run_plan_before_wandb_login(
    tmp_path,
    monkeypatch,
):
    module = load_module()
    official_eval_dir = tmp_path / "official"
    official_eval_dir.mkdir()
    patch_path = tmp_path / "patches.json"
    (official_eval_dir / "summary.json").write_text(
        json.dumps(summary_payload()),
        encoding="utf-8",
    )
    (official_eval_dir / "eval_results.json").write_text(
        json.dumps({"i1": True, "i2": False, "i3": True}),
        encoding="utf-8",
    )
    patch_path.write_text(json.dumps(patch_rows_with_nemoclaw_audit()), encoding="utf-8")

    def fail_login():
        raise AssertionError("must reject before W&B login")

    monkeypatch.setattr(module.wandb, "login", fail_login)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "log_agentic_swe_results_to_wandb.py",
            "--official-eval-dir",
            str(official_eval_dir),
            "--patch-path",
            str(patch_path),
            "--model-name",
            "test/model",
            "--expected-total",
            "3",
        ],
    )

    with pytest.raises(SystemExit, match="validated-dry-run-plan-json"):
        module.main()


def test_main_write_rejects_mismatched_validated_plan_before_wandb_login(
    tmp_path,
    monkeypatch,
):
    module = load_module()
    official_eval_dir = tmp_path / "official"
    official_eval_dir.mkdir()
    patch_path = tmp_path / "patches.json"
    (official_eval_dir / "summary.json").write_text(
        json.dumps(summary_payload()),
        encoding="utf-8",
    )
    (official_eval_dir / "eval_results.json").write_text(
        json.dumps({"i1": True, "i2": False, "i3": True}),
        encoding="utf-8",
    )
    patch_path.write_text(json.dumps(patch_rows_with_nemoclaw_audit()), encoding="utf-8")
    plan_json = tmp_path / "plan.json"
    plan_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ok": True,
                "will_write_wandb": False,
                "benchmark": "agentic_swe",
                "entity": "llm-leaderboard",
                "project": "tc-leaderboard",
                "run_name": "wrong-run-name",
                "job_type": "evaluation-relog",
                "tags": [],
                "config": {},
                "source": {},
                "would_log": {},
                "post_log_verifier_command_template": "",
            }
        ),
        encoding="utf-8",
    )

    def fail_login():
        raise AssertionError("must reject before W&B login")

    monkeypatch.setattr(module.wandb, "login", fail_login)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "log_agentic_swe_results_to_wandb.py",
            "--official-eval-dir",
            str(official_eval_dir),
            "--patch-path",
            str(patch_path),
            "--model-name",
            "test/model",
            "--expected-total",
            "3",
            "--validated-dry-run-plan-json",
            str(plan_json),
        ],
    )

    with pytest.raises(ValueError, match="validated dry-run plan"):
        module.main()


def test_main_write_rejects_source_file_drift_after_validated_plan(
    tmp_path,
    monkeypatch,
):
    module = load_module()
    official_eval_dir = tmp_path / "official"
    official_eval_dir.mkdir()
    patch_path = tmp_path / "patches.json"
    (official_eval_dir / "summary.json").write_text(
        json.dumps(summary_payload()),
        encoding="utf-8",
    )
    (official_eval_dir / "eval_results.json").write_text(
        json.dumps({"i1": True, "i2": False, "i3": True}),
        encoding="utf-8",
    )
    patch_path.write_text(
        json.dumps(
            patch_rows_with_nemoclaw_audit()
        ),
        encoding="utf-8",
    )
    plan_json = tmp_path / "plan.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "log_agentic_swe_results_to_wandb.py",
            "--official-eval-dir",
            str(official_eval_dir),
            "--patch-path",
            str(patch_path),
            "--model-name",
            "test/model",
            "--expected-total",
            "3",
            "--dry-run",
            "--plan-json",
            str(plan_json),
        ],
    )
    module.main()

    patch_path.write_text(
        json.dumps(
            patch_rows_with_nemoclaw_audit(tool_policy_ok=False)
        ),
        encoding="utf-8",
    )

    def fail_login():
        raise AssertionError("must reject before W&B login")

    monkeypatch.setattr(module.wandb, "login", fail_login)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "log_agentic_swe_results_to_wandb.py",
            "--official-eval-dir",
            str(official_eval_dir),
            "--patch-path",
            str(patch_path),
            "--model-name",
            "test/model",
            "--expected-total",
            "3",
            "--validated-dry-run-plan-json",
            str(plan_json),
        ],
    )

    with pytest.raises(ValueError, match="source"):
        module.main()


def test_main_write_requires_external_action_approval_before_wandb_login(
    tmp_path,
    monkeypatch,
):
    module = load_module()
    official_eval_dir = tmp_path / "official"
    official_eval_dir.mkdir()
    patch_path = tmp_path / "patches.json"
    (official_eval_dir / "summary.json").write_text(
        json.dumps(summary_payload()),
        encoding="utf-8",
    )
    (official_eval_dir / "eval_results.json").write_text(
        json.dumps({"i1": True, "i2": False, "i3": True}),
        encoding="utf-8",
    )
    patch_path.write_text(
        json.dumps(
            patch_rows_with_nemoclaw_audit()
        ),
        encoding="utf-8",
    )
    plan_json = tmp_path / "plan.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "log_agentic_swe_results_to_wandb.py",
            "--official-eval-dir",
            str(official_eval_dir),
            "--patch-path",
            str(patch_path),
            "--model-name",
            "test/model",
            "--expected-total",
            "3",
            "--dry-run",
            "--plan-json",
            str(plan_json),
        ],
    )
    module.main()

    def fail_login():
        raise AssertionError("must reject before W&B login")

    monkeypatch.setattr(module.wandb, "login", fail_login)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "log_agentic_swe_results_to_wandb.py",
            "--official-eval-dir",
            str(official_eval_dir),
            "--patch-path",
            str(patch_path),
            "--model-name",
            "test/model",
            "--expected-total",
            "3",
            "--validated-dry-run-plan-json",
            str(plan_json),
        ],
    )

    with pytest.raises(SystemExit, match="external-action-approval-report-json"):
        module.main()


def test_main_write_requires_external_action_approval_source_packet_before_wandb_login(
    tmp_path,
    monkeypatch,
):
    module = load_module()
    official_eval_dir = tmp_path / "official"
    official_eval_dir.mkdir()
    patch_path = tmp_path / "patches.json"
    (official_eval_dir / "summary.json").write_text(
        json.dumps(summary_payload()),
        encoding="utf-8",
    )
    (official_eval_dir / "eval_results.json").write_text(
        json.dumps({"i1": True, "i2": False, "i3": True}),
        encoding="utf-8",
    )
    patch_path.write_text(
        json.dumps(
            patch_rows_with_nemoclaw_audit()
        ),
        encoding="utf-8",
    )
    plan_json = tmp_path / "plan.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "log_agentic_swe_results_to_wandb.py",
            "--official-eval-dir",
            str(official_eval_dir),
            "--patch-path",
            str(patch_path),
            "--model-name",
            "test/model",
            "--expected-total",
            "3",
            "--dry-run",
            "--plan-json",
            str(plan_json),
        ],
    )
    module.main()
    report_json = tmp_path / "approval_report.json"
    report_json.write_text("{}", encoding="utf-8")

    def fail_login():
        raise AssertionError("must reject before W&B login")

    monkeypatch.setattr(module.wandb, "login", fail_login)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "log_agentic_swe_results_to_wandb.py",
            "--official-eval-dir",
            str(official_eval_dir),
            "--patch-path",
            str(patch_path),
            "--model-name",
            "test/model",
            "--expected-total",
            "3",
            "--validated-dry-run-plan-json",
            str(plan_json),
            "--external-action-approval-report-json",
            str(report_json),
        ],
    )

    with pytest.raises(SystemExit, match="external-action-approval-source-packet-json"):
        module.main()
