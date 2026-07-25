import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


from evaluator.evaluate_utils.benchmark_completion_evidence import (
    validate_agentic_math_completion,
)


def _write_valid_math_output(root: Path, count: int = 2) -> None:
    runner = root / "openclaw"
    runner.mkdir(parents=True)
    summary = {
        "total_instances": count,
        "weave_agents_required_instances": count,
        "weave_agents_passed_instances": count,
        "weave_agents_failed_instances": 0,
        "nemoclaw_session_audit_required_instances": count,
        "nemoclaw_session_audit_passed_instances": count,
        "nemoclaw_session_audit_failed_instances": 0,
    }
    (runner / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    rows = [
        {
            "task_id": f"math-{index}",
            "cache_key": {"model": "openai-direct/test"},
            "weave_agents_ok": True,
            "nemoclaw_session_audit_ok": True,
        }
        for index in range(count)
    ]
    (runner / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_validated_math_completion_accepts_full_audited_output(tmp_path):
    _write_valid_math_output(tmp_path)

    evidence = validate_agentic_math_completion(
        tmp_path,
        expected_count=2,
        expected_model="openai-direct/test",
        require_weave_agents=True,
        require_nemoclaw_session_audit=True,
    )

    assert evidence["ok"] is True
    assert evidence["result_count"] == 2


def test_validated_math_completion_rejects_model_or_trace_mismatch(tmp_path):
    _write_valid_math_output(tmp_path)
    results_path = tmp_path / "openclaw" / "results.jsonl"
    rows = [json.loads(line) for line in results_path.read_text().splitlines()]
    rows[0]["cache_key"]["model"] = "openai-direct/other"
    rows[1]["weave_agents_ok"] = False
    results_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    evidence = validate_agentic_math_completion(
        tmp_path,
        expected_count=2,
        expected_model="openai-direct/test",
        require_weave_agents=True,
        require_nemoclaw_session_audit=True,
    )

    assert evidence["ok"] is False
    assert any("model mismatch" in error for error in evidence["errors"])
    assert any("Weave evidence" in error for error in evidence["errors"])
