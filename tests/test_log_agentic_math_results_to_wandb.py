import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = REPO_ROOT / "scripts" / "tools" / "log_agentic_math_results_to_wandb.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_binding(source_packet: Path, reviewed_packet: Path) -> dict:
    return {
        "source_packet_json": str(source_packet),
        "source_packet_readable": True,
        "source_approval_packet_sha256": sha256_file(source_packet),
        "reviewed_packet_json": str(reviewed_packet),
        "bound": True,
        "errors": [],
    }


def test_validate_summary_accepts_consistent_rows():
    module = load_module()
    summary = {
        "total_instances": 3,
        "answered_instances": 2,
        "correct_instances": 2,
        "incorrect_instances": 1,
        "accuracy": 2 / 3,
        "correctness": 2 / 3,
    }
    rows = [
        {"correct": True, "predicted_answer": "1"},
        {"correct": True, "predicted_answer": "2"},
        {"correct": False, "predicted_answer": None},
    ]

    module.validate_summary(summary, rows)


def test_validate_summary_rejects_mismatched_counts():
    module = load_module()
    summary = {
        "total_instances": 2,
        "answered_instances": 2,
        "correct_instances": 2,
        "incorrect_instances": 0,
        "accuracy": 1.0,
        "correctness": 1.0,
    }
    rows = [
        {"correct": True, "predicted_answer": "1"},
        {"correct": False, "predicted_answer": "2"},
    ]

    with pytest.raises(ValueError, match="correct_instances=2"):
        module.validate_summary(summary, rows)


def test_build_leaderboard_uses_existing_agentic_math_schema():
    module = load_module()
    summary = {
        "total_instances": 100,
        "answered_instances": 99,
        "correct_instances": 86,
        "incorrect_instances": 14,
        "accuracy": 0.86,
        "correctness": 0.86,
    }

    table = module.build_leaderboard(
        model_name="deepseek/deepseek-v4-pro",
        summary=summary,
        benchmark_name="OlymMATH-HARD zh-TW",
        result_source="/tmp/results",
    )

    assert table.to_dict(orient="records") == [
        {
            "model_name": "deepseek/deepseek-v4-pro",
            "benchmark": "OlymMATH-HARD zh-TW",
            "total_samples": 100,
            "answered_samples": 99,
            "correct_count": 86,
            "accuracy": 0.86,
            "correctness": 0.86,
            "result_source": "/tmp/results",
        }
    ]


def test_main_dry_run_writes_plan_without_wandb_login(tmp_path, monkeypatch, capsys):
    module = load_module()
    results_dir = tmp_path / "math"
    results_dir.mkdir()
    summary = {
        "total_instances": 2,
        "answered_instances": 2,
        "correct_instances": 1,
        "incorrect_instances": 1,
        "accuracy": 0.5,
        "correctness": 0.5,
        "runner_version": "test",
        "model": "test/model",
        "thinking": "enabled",
    }
    rows = [
        {"id": "m1", "correct": True, "predicted_answer": "1"},
        {"id": "m2", "correct": False, "predicted_answer": "2"},
    ]
    (results_dir / "summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    (results_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
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
            "log_agentic_math_results_to_wandb.py",
            "--results-dir",
            str(results_dir),
            "--model-name",
            "test/model",
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
    assert set(plan["source"]["source_sha256"]) == {"summary_json", "results_jsonl"}
    assert plan["config"]["relog"]["source_sha256"] == plan["source"]["source_sha256"]
    assert plan["would_log"]["tables"]["agentic_math_output_table"] == 2
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
    assert "--benchmark agentic_math" in plan["post_log_verifier_command_template"]
    assert "--expected-total 2" in plan["post_log_verifier_command_template"]
    assert (
        "--expected-run-config relog.source_sha256.summary_json="
        in plan["post_log_verifier_command_template"]
    )
    assert (
        "--expected-run-config relog.source_sha256.results_jsonl="
        in plan["post_log_verifier_command_template"]
    )


def test_main_write_requires_validated_dry_run_plan_before_wandb_login(
    tmp_path,
    monkeypatch,
):
    module = load_module()
    results_dir = tmp_path / "math"
    results_dir.mkdir()
    summary = {
        "total_instances": 1,
        "answered_instances": 1,
        "correct_instances": 1,
        "incorrect_instances": 0,
        "accuracy": 1.0,
        "correctness": 1.0,
    }
    (results_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (results_dir / "results.jsonl").write_text(
        json.dumps({"correct": True, "predicted_answer": "1"}) + "\n",
        encoding="utf-8",
    )

    def fail_login():
        raise AssertionError("must reject before W&B login")

    monkeypatch.setattr(module.wandb, "login", fail_login)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "log_agentic_math_results_to_wandb.py",
            "--results-dir",
            str(results_dir),
            "--model-name",
            "test/model",
        ],
    )

    with pytest.raises(SystemExit, match="validated-dry-run-plan-json"):
        module.main()


def test_main_write_rejects_mismatched_validated_plan_before_wandb_login(
    tmp_path,
    monkeypatch,
):
    module = load_module()
    results_dir = tmp_path / "math"
    results_dir.mkdir()
    summary = {
        "total_instances": 1,
        "answered_instances": 1,
        "correct_instances": 1,
        "incorrect_instances": 0,
        "accuracy": 1.0,
        "correctness": 1.0,
    }
    (results_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (results_dir / "results.jsonl").write_text(
        json.dumps({"correct": True, "predicted_answer": "1"}) + "\n",
        encoding="utf-8",
    )
    plan_json = tmp_path / "plan.json"
    plan_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ok": True,
                "will_write_wandb": False,
                "benchmark": "agentic_math",
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
            "log_agentic_math_results_to_wandb.py",
            "--results-dir",
            str(results_dir),
            "--model-name",
            "test/model",
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
    results_dir = tmp_path / "math"
    results_dir.mkdir()
    summary = {
        "total_instances": 1,
        "answered_instances": 1,
        "correct_instances": 1,
        "incorrect_instances": 0,
        "accuracy": 1.0,
        "correctness": 1.0,
    }
    (results_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    results_path = results_dir / "results.jsonl"
    results_path.write_text(
        json.dumps({"correct": True, "predicted_answer": "1"}) + "\n",
        encoding="utf-8",
    )
    plan_json = tmp_path / "plan.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "log_agentic_math_results_to_wandb.py",
            "--results-dir",
            str(results_dir),
            "--model-name",
            "test/model",
            "--dry-run",
            "--plan-json",
            str(plan_json),
        ],
    )
    module.main()

    results_path.write_text(
        json.dumps({"correct": True, "predicted_answer": "2"}) + "\n",
        encoding="utf-8",
    )

    def fail_login():
        raise AssertionError("must reject before W&B login")

    monkeypatch.setattr(module.wandb, "login", fail_login)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "log_agentic_math_results_to_wandb.py",
            "--results-dir",
            str(results_dir),
            "--model-name",
            "test/model",
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
    results_dir = tmp_path / "math"
    results_dir.mkdir()
    summary = {
        "total_instances": 1,
        "answered_instances": 1,
        "correct_instances": 1,
        "incorrect_instances": 0,
        "accuracy": 1.0,
        "correctness": 1.0,
    }
    (results_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (results_dir / "results.jsonl").write_text(
        json.dumps({"correct": True, "predicted_answer": "1"}) + "\n",
        encoding="utf-8",
    )
    plan_json = tmp_path / "plan.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "log_agentic_math_results_to_wandb.py",
            "--results-dir",
            str(results_dir),
            "--model-name",
            "test/model",
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
            "log_agentic_math_results_to_wandb.py",
            "--results-dir",
            str(results_dir),
            "--model-name",
            "test/model",
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
    results_dir = tmp_path / "math"
    results_dir.mkdir()
    summary = {
        "total_instances": 1,
        "answered_instances": 1,
        "correct_instances": 1,
        "incorrect_instances": 0,
        "accuracy": 1.0,
        "correctness": 1.0,
    }
    (results_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (results_dir / "results.jsonl").write_text(
        json.dumps({"correct": True, "predicted_answer": "1"}) + "\n",
        encoding="utf-8",
    )
    plan_json = tmp_path / "plan.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "log_agentic_math_results_to_wandb.py",
            "--results-dir",
            str(results_dir),
            "--model-name",
            "test/model",
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
            "log_agentic_math_results_to_wandb.py",
            "--results-dir",
            str(results_dir),
            "--model-name",
            "test/model",
            "--validated-dry-run-plan-json",
            str(plan_json),
            "--external-action-approval-report-json",
            str(report_json),
        ],
    )

    with pytest.raises(SystemExit, match="external-action-approval-source-packet-json"):
        module.main()


def test_external_action_approval_must_match_wandb_scope(tmp_path):
    module = load_module()
    source_packet = tmp_path / "source_approval_packet.json"
    source_packet.write_text(json.dumps({"source": "release-bundle"}), encoding="utf-8")
    packet = tmp_path / "approval_packet.json"
    packet.write_text(
        json.dumps(
            {
                "approval_requirements": [
                    {
                        "requirement": "wandb_write",
                        "approved_wandb_entity": "other-entity",
                        "approved_wandb_project": "tc-leaderboard",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "approval_report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ok": True,
                "status": "approved",
                "required_approval_count": 2,
                "granted_approval_count": 2,
                "all_required_approvals_granted": True,
                "approval_packet_json": str(packet),
                "source_binding": source_binding(source_packet, packet),
                "approval_results": [
                    {"requirement": "wandb_access", "required": True, "approved": True},
                    {"requirement": "wandb_write", "required": True, "approved": True},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="approved_wandb_entity"):
        module.validate_external_action_approval_for_wandb_write(
            report,
            entity="llm-leaderboard",
            project="tc-leaderboard",
        )


def test_external_action_approval_resolves_packet_path_relative_to_report(tmp_path):
    module = load_module()
    report_dir = tmp_path / "review"
    report_dir.mkdir()
    source_packet = report_dir / "source_approval_packet.json"
    source_packet.write_text(json.dumps({"source": "release-bundle"}), encoding="utf-8")
    packet = report_dir / "approval_packet.json"
    packet.write_text(
        json.dumps(
            {
                "approval_requirements": [
                    {
                        "requirement": "wandb_write",
                        "approved_wandb_entity": "llm-leaderboard",
                        "approved_wandb_project": "tc-leaderboard",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    report = report_dir / "approval_report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ok": True,
                "status": "approved",
                "required_approval_count": 2,
                "granted_approval_count": 2,
                "all_required_approvals_granted": True,
                "approval_packet_json": "approval_packet.json",
                "source_binding": {
                    "source_packet_json": "source_approval_packet.json",
                    "source_packet_readable": True,
                    "source_approval_packet_sha256": sha256_file(source_packet),
                    "reviewed_packet_json": "approval_packet.json",
                    "bound": True,
                    "errors": [],
                },
                "approval_results": [
                    {"requirement": "wandb_access", "required": True, "approved": True},
                    {"requirement": "wandb_write", "required": True, "approved": True},
                ],
            }
        ),
        encoding="utf-8",
    )

    module.validate_external_action_approval_for_wandb_write(
        report,
        entity="llm-leaderboard",
        project="tc-leaderboard",
        expected_source_packet_path=source_packet,
    )


def test_external_action_approval_must_match_expected_source_packet_path(tmp_path):
    module = load_module()
    source_packet = tmp_path / "source_approval_packet.json"
    source_packet.write_text(json.dumps({"source": "release-bundle"}), encoding="utf-8")
    other_source_packet = tmp_path / "other_source_approval_packet.json"
    other_source_packet.write_text(
        json.dumps({"source": "other-release-bundle"}),
        encoding="utf-8",
    )
    packet = tmp_path / "approval_packet.json"
    packet.write_text(
        json.dumps(
            {
                "approval_requirements": [
                    {
                        "requirement": "wandb_write",
                        "approved_wandb_entity": "llm-leaderboard",
                        "approved_wandb_project": "tc-leaderboard",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "approval_report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ok": True,
                "status": "approved",
                "required_approval_count": 2,
                "granted_approval_count": 2,
                "all_required_approvals_granted": True,
                "approval_packet_json": str(packet),
                "source_binding": source_binding(source_packet, packet),
                "approval_results": [
                    {"requirement": "wandb_access", "required": True, "approved": True},
                    {"requirement": "wandb_write", "required": True, "approved": True},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source packet path"):
        module.validate_external_action_approval_for_wandb_write(
            report,
            entity="llm-leaderboard",
            project="tc-leaderboard",
            expected_source_packet_path=other_source_packet,
        )


def test_external_action_approval_requires_source_packet_readable(tmp_path):
    module = load_module()
    source_packet = tmp_path / "source_approval_packet.json"
    source_packet.write_text(json.dumps({"source": "release-bundle"}), encoding="utf-8")
    packet = tmp_path / "approval_packet.json"
    packet.write_text(
        json.dumps(
            {
                "approval_requirements": [
                    {
                        "requirement": "wandb_write",
                        "approved_wandb_entity": "llm-leaderboard",
                        "approved_wandb_project": "tc-leaderboard",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    binding = source_binding(source_packet, packet)
    binding["source_packet_readable"] = False
    report = tmp_path / "approval_report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ok": True,
                "status": "approved",
                "required_approval_count": 2,
                "granted_approval_count": 2,
                "all_required_approvals_granted": True,
                "approval_packet_json": str(packet),
                "source_binding": binding,
                "approval_results": [
                    {"requirement": "wandb_access", "required": True, "approved": True},
                    {"requirement": "wandb_write", "required": True, "approved": True},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source packet must be readable"):
        module.validate_external_action_approval_for_wandb_write(
            report,
            entity="llm-leaderboard",
            project="tc-leaderboard",
        )
