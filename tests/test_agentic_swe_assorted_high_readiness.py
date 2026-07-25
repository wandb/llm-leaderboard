import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "tools" / "check_agentic_swe_assorted_high_readiness.py"
spec = importlib.util.spec_from_file_location("check_agentic_swe_assorted_high_readiness", MODULE_PATH)
assert spec is not None and spec.loader is not None
readiness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readiness)


def _args(
    findings: Path,
    model: str = "openai-direct/gpt-4.1-mini-2025-04-14",
    subset: str = "essential_anchored_high_10_model_fidelity_cost_balanced",
):
    return argparse.Namespace(
        findings=findings,
        model=model,
        subset=subset,
        manifest=ROOT / "data" / "taiwan" / "deepswe" / "manifest.json",
        pilot_run_dir=[],
    )


def _write_findings(path: Path, entries: list[dict]):
    path.write_text(
        json.dumps({"findings": entries}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_skipped_runtime_or_sandbox_check_never_authorizes_paid_run():
    decision, recommendation = readiness.build_readiness_decision(
        {"ok": True},
        {"ok": True},
        {"ok": None, "skipped": True},
        {"ok": None, "skipped": True},
        {"ok": True},
    )

    assert decision["status"] == "unverified"
    assert decision["sandbox_ready"] is False
    assert decision["runtime_ready"] is False
    assert decision["full_run_recommended"] is False
    assert "skipped" in recommendation


def _runtime_args(tmp_path: Path):
    return argparse.Namespace(
        skip_openclaw_runtime_check=False,
        openclaw_runtime_patch_script=tmp_path / "patch_openclaw_turn_budget_guard.py",
        nemoclaw_runtime_patch_script=tmp_path / "patch_nemoclaw_openclaw_runtime.py",
        openclaw_package_dir=None,
        nemoclaw_openclaw_package_dir="/usr/local/lib/node_modules/openclaw",
        openclaw_runtime_check_timeout=120,
        sandbox="nejumi-taiwan",
        nemoclaw_bin="nemoclaw",
    )


def test_scoreable_model_side_budget_failure_counts_as_provider_ready(tmp_path):
    findings = tmp_path / "findings.json"
    _write_findings(
        findings,
        [
            {
                "task_name": "etree-xml-diff-patch",
                "run_dir": (
                    "outputs/agentic_swe_assorted_runs/"
                    "essential_anchored_high_10_model_fidelity_cost_balanced_"
                    "gpt41mini_thinking_off_pilot_20260713"
                ),
                "subset": "essential_anchored_high_10_model_fidelity_cost_balanced",
                "model": "openai-direct/gpt-4.1-mini-2025-04-14",
                "status": "scoreable_model_side_budget_failure",
                "reason": "agent_turn_limit_exceeded_scored_as_incorrect",
                "deepswe_reward": 0.0,
                "openclaw_disqualified_reason": "runtime_budget_exceeded",
                "weave_agents_ok": True,
            }
        ],
    )

    report = readiness.check_provider_findings(_args(findings))

    assert report["ok"] is True
    assert report["full_run_recommended"] is True
    scoreable_check = report["checks"][0]
    assert scoreable_check["ok"] is True
    assert scoreable_check["detail"]["runs"][0]["status"] == "scoreable_model_side_budget_failure"
    assert scoreable_check["detail"]["runs"][0]["openclaw_disqualified_reason"] == (
        "runtime_budget_exceeded"
    )


def test_provider_timeout_blocks_full_run_for_current_subset_entry(tmp_path):
    findings = tmp_path / "findings.json"
    _write_findings(
        findings,
        [
            {
                "task_name": "etree-xml-diff-patch",
                "run_dir": (
                    "outputs/agentic_swe_assorted_runs/"
                    "essential_anchored_high_10_model_fidelity_cost_balanced_"
                    "etree_pilot_20260713"
                ),
                "subset": "essential_anchored_high_10_model_fidelity_cost_balanced",
                "model": "openrouter-direct/z-ai/glm-5.2",
                "status": "provider_blocked_not_task_excluded",
                "reason": "provider_timeout_504",
                "openclaw_disqualified_reason": "provider_transient_exhausted",
            }
        ],
    )

    report = readiness.check_provider_findings(
        _args(findings, model="openrouter-direct/z-ai/glm-5.2")
    )

    assert report["ok"] is False
    assert report["full_run_recommended"] is False
    assert report["checks"][1]["ok"] is False
    assert report["checks"][1]["detail"][0]["reason"] == "provider_timeout_504"


def test_legacy_essential3_provider_timeout_does_not_block_new_default_subset(tmp_path):
    findings = tmp_path / "findings.json"
    _write_findings(
        findings,
        [
            {
                "task_name": "etree-xml-diff-patch",
                "run_dir": "outputs/agentic_swe_assorted_runs/high_essential3_etree_pilot_20260713",
                "model": "openrouter-direct/z-ai/glm-5.2",
                "status": "provider_blocked_not_task_excluded",
                "reason": "provider_timeout_504",
                "openclaw_disqualified_reason": "provider_transient_exhausted",
            }
        ],
    )

    report = readiness.check_provider_findings(
        _args(findings, model="openrouter-direct/z-ai/glm-5.2")
    )

    assert report["checks"][1]["ok"] is True
    assert report["checks"][1]["detail"] == []


def test_scoreable_path_verified_unresolved_counts_as_provider_ready(tmp_path):
    findings = tmp_path / "findings.json"
    _write_findings(
        findings,
        [
            {
                "task_name": "etree-xml-diff-patch",
                "run_dir": (
                    "outputs/agentic_swe_assorted_runs/"
                    "essential_anchored_high_10_model_fidelity_cost_balanced_"
                    "luna_max_probe_20260713"
                ),
                "subset": "essential_anchored_high_10_model_fidelity_cost_balanced",
                "model": "openai-direct/gpt-5.6-luna",
                "status": "scoreable_path_verified_unresolved",
                "reason": "model_side_unresolved_after_budget",
                "deepswe_reward": 0.0,
                "openclaw_disqualified_reason": None,
            }
        ],
    )

    report = readiness.check_provider_findings(
        _args(findings, model="openai-direct/gpt-5.6-luna")
    )

    assert report["ok"] is True
    assert report["full_run_recommended"] is True
    assert report["checks"][0]["detail"]["runs"][0]["status"] == (
        "scoreable_path_verified_unresolved"
    )


def test_idle_timeout_scoreable_history_does_not_count_as_provider_ready(tmp_path):
    findings = tmp_path / "findings.json"
    _write_findings(
        findings,
        [
            {
                "task_name": "ts-pattern-match-each",
                "subset": "essential_anchored_high_10_model_fidelity_cost_balanced",
                "run_dir": "outputs/agentic_swe_assorted_runs/glm52_idle_timeout_pilot",
                "model": "openrouter-direct/z-ai/glm-5.2",
                "status": "scoreable_model_side_budget_failure",
                "reason": "llm_response_idle_timeout_scored_as_incorrect",
                "scoreable_failure_reason": "llm_response_idle_timeout",
                "deepswe_reward": 0.0,
                "openclaw_disqualified_reason": "runtime_budget_exceeded",
                "weave_agents_ok": False,
            }
        ],
    )

    report = readiness.check_provider_findings(
        _args(findings, model="openrouter-direct/z-ai/glm-5.2")
    )

    assert report["ok"] is False
    assert report["full_run_recommended"] is False
    assert report["checks"][0]["ok"] is False
    assert report["checks"][2]["ok"] is False
    assert report["checks"][2]["detail"][0]["scoreable_failure_reason"] == (
        "llm_response_idle_timeout"
    )


def test_completed_scoreable_pilot_run_counts_as_provider_ready(tmp_path):
    findings = tmp_path / "findings.json"
    _write_findings(findings, [])
    run_dir = tmp_path / "pilot"
    deepswe_dir = run_dir / "high" / "deepswe"
    deepswe_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"model": "openrouter/z-ai/glm-5.2"}), encoding="utf-8"
    )
    (deepswe_dir / "summary.json").write_text(
        json.dumps(
            {
                "total_trials": 1,
                "scored_trials": 1,
                "exceptions": 0,
                "non_scoreable_provider_timeouts": 0,
                "weave_agents_ok": 1,
            }
        ),
        encoding="utf-8",
    )
    result = {
        "task_name": "datacurve/etree-xml-diff-patch",
        "resolved": False,
        "score": 0,
        "exception": None,
        "openclaw_disqualified_reason": "",
        "weave_agents_ok": True,
        "weave_agents_conversation_url": "https://example.test/conversation",
    }
    (deepswe_dir / "results.jsonl").write_text(
        json.dumps(result) + "\n", encoding="utf-8"
    )
    args = _args(findings, model="openrouter/z-ai/glm-5.2")
    args.pilot_run_dir = [run_dir]

    report = readiness.check_provider_findings(args)

    assert report["ok"] is True
    assert report["checks"][0]["detail"]["scoreable_pilot_count"] == 1
    assert report["checks"][0]["detail"]["runs"][0]["status"] == "scoreable_incorrect"
    assert report["evidence"]["pilot_runs"][0]["scored_trials"] == 1


def test_completed_pilot_provider_timeout_blocks_readiness(tmp_path):
    findings = tmp_path / "findings.json"
    _write_findings(findings, [])
    run_dir = tmp_path / "pilot"
    deepswe_dir = run_dir / "high" / "deepswe"
    deepswe_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"model": "openrouter/z-ai/glm-5.2"}), encoding="utf-8"
    )
    (deepswe_dir / "summary.json").write_text(
        json.dumps(
            {
                "total_trials": 1,
                "scored_trials": 0,
                "exceptions": 1,
                "non_scoreable_provider_timeouts": 1,
                "weave_agents_ok": 0,
            }
        ),
        encoding="utf-8",
    )
    (deepswe_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "task_name": "datacurve/etree-xml-diff-patch",
                "resolved": False,
                "score": None,
                "exception": "provider timeout",
                "openclaw_disqualified_reason": "provider_transient_exhausted",
                "weave_agents_ok": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = _args(findings, model="openrouter/z-ai/glm-5.2")
    args.pilot_run_dir = [run_dir]

    report = readiness.check_provider_findings(args)

    assert report["ok"] is False
    assert report["checks"][1]["ok"] is False
    assert report["evidence"]["pilot_runs"][0]["non_scoreable_provider_timeouts"] == 1


def test_openclaw_runtime_check_requires_extended_thinking_patch(tmp_path, monkeypatch):
    args = _runtime_args(tmp_path)

    def fake_run_command(command, timeout):
        if "--check-only" in command:
            payload = {
                "ok": True,
                "patch": {
                    "ok": True,
                    "verified_extended_thinking": ["/usr/local/lib/node_modules/openclaw/dist/shared.js"],
                },
            }
        else:
            payload = {
                "ok": True,
                "verified_extended_thinking": ["/opt/openclaw/dist/shared.js"],
            }
        return {
            "ok": True,
            "returncode": 0,
            "stdout": json.dumps(payload),
            "stderr": "",
        }

    monkeypatch.setattr(readiness, "run_command", fake_run_command)

    report = readiness.check_openclaw_runtime(args)

    assert report["ok"] is True
    assert [check["ok"] for check in report["checks"]] == [True, True]


def test_openclaw_runtime_check_reports_setup_remediation_when_missing(tmp_path, monkeypatch):
    args = _runtime_args(tmp_path)

    def fake_run_command(command, timeout):
        payload = {"ok": True, "verified_exec_timeout": ["/opt/openclaw/dist/shared.js"]}
        return {
            "ok": True,
            "returncode": 0,
            "stdout": json.dumps(payload),
            "stderr": "",
        }

    monkeypatch.setattr(readiness, "run_command", fake_run_command)

    report = readiness.check_openclaw_runtime(args)

    assert report["ok"] is False
    assert "install_openclaw_budget_guard.sh" in report["remediation"]
    assert report["checks"][0]["ok"] is False
