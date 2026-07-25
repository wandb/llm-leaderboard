import importlib.util
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PARTIAL_MODULE = ROOT / "scripts" / "tools" / "agentic_swe_partial_credit.py"


_partial = None


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_partial = load_module(PARTIAL_MODULE)
DIAGNOSTIC_PARTIAL_CREDIT_VERSION = _partial.DIAGNOSTIC_PARTIAL_CREDIT_VERSION
build_diagnostic_partial_credit = _partial.build_diagnostic_partial_credit
from_deepswe_verifier = _partial.from_deepswe_verifier
from_swebench_report = _partial.from_swebench_report


def test_resolved_task_keeps_binary_one_without_partial_inflation():
    result = build_diagnostic_partial_credit(
        resolved=True,
        f2p_passed=1,
        f2p_total=1,
        p2p_passed=10,
        p2p_total=10,
        patch_applied=True,
        evidence_source="test",
    )

    assert result["binary_score"] == 1.0
    assert result["diagnostic_partial_credit"] == 0.0
    assert result["diagnostic_score_with_partial"] == 1.0


def test_unresolved_official_score_requires_f2p_and_penalizes_regressions():
    result = build_diagnostic_partial_credit(
        resolved=False,
        f2p_passed=3,
        f2p_total=4,
        p2p_passed=9,
        p2p_total=10,
        patch_applied=True,
        evidence_source="test",
    )

    assert result["binary_score"] == 0.0
    assert result["official_score"] == pytest.approx(0.3 * 0.75**2 * 0.9)
    assert result["diagnostic_partial_credit"] == result["official_score"]
    assert result["diagnostic_score_with_partial"] == result["diagnostic_partial_credit"]
    assert result["diagnostic_partial_credit_version"] == DIAGNOSTIC_PARTIAL_CREDIT_VERSION


@pytest.mark.parametrize("patch_applied,scoreable", [(False, True), (True, False)])
def test_unapplied_or_unscoreable_patch_gets_no_partial_credit(patch_applied, scoreable):
    result = build_diagnostic_partial_credit(
        resolved=False,
        f2p_passed=9,
        f2p_total=10,
        p2p_passed=100,
        p2p_total=100,
        patch_applied=patch_applied,
        scoreable=scoreable,
        evidence_source="test",
    )

    assert result["diagnostic_score_with_partial"] == 0.0


def test_p2p_success_cannot_create_points_when_no_f2p_test_passes():
    result = build_diagnostic_partial_credit(
        resolved=False,
        f2p_passed=0,
        f2p_total=4,
        p2p_passed=1000,
        p2p_total=1000,
        patch_applied=True,
        evidence_source="test",
    )

    assert result["diagnostic_score_with_partial"] == 0.0


def test_swebench_and_deepswe_adapters_use_same_formula():
    swebench = from_swebench_report(
        "repo__issue-1",
        {
            "resolved": False,
            "patch_successfully_applied": True,
            "tests_status": {
                "FAIL_TO_PASS": {"success": ["a"], "failure": ["b"]},
                "PASS_TO_PASS": {"success": ["c", "d"], "failure": []},
            },
        },
    )
    deepswe = from_deepswe_verifier(
        {
            "rewards": {
                "reward": 0,
                "f2p_passed": 1,
                "f2p_total": 2,
                "p2p_passed": 2,
                "p2p_total": 2,
            }
        },
        resolved=False,
        patch_applied=True,
    )

    assert swebench["diagnostic_score_with_partial"] == pytest.approx(0.075)
    assert deepswe["diagnostic_score_with_partial"] == swebench[
        "diagnostic_score_with_partial"
    ]


def test_lite_collector_reads_only_current_run_reports(tmp_path):
    module = load_module(ROOT / "scripts" / "tools" / "evaluate_swebench_lite_patches.py")
    report_dir = tmp_path / "logs" / "run_evaluation" / "current" / "model" / "repo__issue-1"
    report_dir.mkdir(parents=True)
    (report_dir / "report.json").write_text(
        json.dumps(
            {
                "repo__issue-1": {
                    "resolved": False,
                    "patch_successfully_applied": True,
                    "tests_status": {
                        "FAIL_TO_PASS": {"success": ["a"], "failure": ["b"]},
                        "PASS_TO_PASS": {"success": ["c"], "failure": []},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    results = module.collect_partial_eval_results(
        tmp_path,
        instance_ids=["repo__issue-1", "repo__missing-2"],
        run_id="current",
    )

    assert results["repo__issue-1"]["diagnostic_score_with_partial"] == pytest.approx(0.075)
    assert results["repo__missing-2"]["diagnostic_scoreable"] is False
    assert (tmp_path / "partial_eval_results.json").is_file()


def test_lite_collector_follows_isolated_child_run_ids(tmp_path):
    module = load_module(ROOT / "scripts" / "tools" / "evaluate_swebench_lite_patches.py")
    instance_id = "repo__issue-1"
    child_run_id = "parent-child"
    result_dir = tmp_path / "isolated" / instance_id
    result_dir.mkdir(parents=True)
    (result_dir / "result.json").write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "run_id": child_run_id,
                "ok": True,
            }
        ),
        encoding="utf-8",
    )
    report_dir = (
        tmp_path
        / "logs"
        / "run_evaluation"
        / child_run_id
        / "model"
        / instance_id
    )
    report_dir.mkdir(parents=True)
    (report_dir / "report.json").write_text(
        json.dumps(
            {
                instance_id: {
                    "resolved": False,
                    "patch_successfully_applied": True,
                    "tests_status": {
                        "FAIL_TO_PASS": {"success": ["a"], "failure": ["b"]},
                        "PASS_TO_PASS": {"success": ["c"], "failure": []},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    results = module.collect_partial_eval_results(
        tmp_path,
        instance_ids=[instance_id],
        run_id="parent",
    )

    assert results[instance_id]["diagnostic_score_with_partial"] == pytest.approx(0.075)


def test_official_summary_never_downgrades_resolved_result(tmp_path):
    module = load_module(ROOT / "scripts" / "tools" / "evaluate_swebench_lite_patches.py")

    summary = module.write_summary(
        tmp_path,
        report={"submitted_instances": 1, "completed_instances": 1},
        eval_results={"repo__issue-1": True},
        partial_eval_results={
            "repo__issue-1": {"diagnostic_score_with_partial": 0.0}
        },
        command=["test"],
    )

    assert summary["pass_at_1"] == 1.0
    assert summary["diagnostic_score_with_partial"] == 1.0


def test_assorted_summary_uses_partial_credit_as_the_single_official_score():
    module = load_module(ROOT / "scripts" / "tools" / "run_agentic_swe_assorted.py")
    rows = []
    for tier, binary, diagnostic in (
        ("low", 1.0, 1.0),
        ("middle", 0.0, 0.12),
        ("high", 0.0, 0.24),
    ):
        rows.append(
            {
                "agentic_swe_tier": tier,
                "resolved": bool(binary),
                "score": diagnostic,
                "diagnostic_score_with_partial": diagnostic,
                "diagnostic_partial_credit": diagnostic if not binary else 0.0,
            }
        )
    by_tier = {tier: [row] for tier, row in zip(("low", "middle", "high"), rows)}
    scoring = module.build_scoring_summary(
        by_tier,
        args=SimpleNamespace(model="test/model", tier_weights="low=1,middle=1,high=1"),
    )

    assert scoring["weighted_binary_pass_at_1"] == pytest.approx(1 / 3)
    assert scoring["weighted_pass_at_1"] == pytest.approx(
        (1.0 + 0.12 + 0.24) / 3
    )
    assert scoring["weighted_official_score"] == scoring["weighted_pass_at_1"]


def test_evaluator_rejects_null_nemoclaw_sandbox_before_launch(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    module = importlib.import_module("evaluator.agentic_swe_assorted")
    cfg = SimpleNamespace(
        agentic_swe_assorted={"nemoclaw_sandbox": None},
        model={"pretrained_model_name_or_path": "test/model"},
    )
    launched = []
    monkeypatch.setattr(module, "_run_command", lambda command: launched.append(command))

    with pytest.raises(ValueError, match="nemoclaw_sandbox must name an isolated"):
        module._run_assorted(cfg, tmp_path)

    assert launched == []
