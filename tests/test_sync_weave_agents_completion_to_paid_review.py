import importlib.util
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tools" / "sync_weave_agents_completion_to_paid_review.py"
PAID_REVIEW_SCRIPT = REPO_ROOT / "scripts" / "tools" / "check_taiwan_paid_run_review_package.py"


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_wandb_run_metadata(payload, *, benchmark="agentic_math", run_id="run-1"):
    required = payload.setdefault("required_evidence", {})
    required["run_metadata"] = {
        "config": [
            {"key": "model.pretrained_model_name_or_path", "expected": "gpt-4.1-mini-2025-04-14"},
            {"key": f"run.{benchmark}", "expected": True},
            {"key": "wandb.run_name", "expected": f"taiwan/full/openai/gpt-4.1-mini: {run_id}"},
        ],
        "tags": ["taiwan-canary"],
        "group": "tw-canary",
        "job_type": "evaluation",
    }
    observed = payload.setdefault("observed_evidence", {})
    observed["run_metadata"] = {
        "config": [
            {
                "key": "model.pretrained_model_name_or_path",
                "present": True,
                "value": "gpt-4.1-mini-2025-04-14",
            },
            {"key": f"run.{benchmark}", "present": True, "value": True},
            {
                "key": "wandb.run_name",
                "present": True,
                "value": f"taiwan/full/openai/gpt-4.1-mini: {run_id}",
            },
        ],
        "tags": ["taiwan-canary"],
        "group": "tw-canary",
        "job_type": "evaluation",
    }
    return payload


def weave_completion_payload(*, ok=True, agent_name="nejumi-taiwan-openclaw"):
    return {
        "ok": ok,
        "verification_schema_version": 1,
        "generated_at": time.time(),
        "project_id": "llm-leaderboard/tc-leaderboard",
        "agent_name": agent_name,
        "agents_url": "https://wandb.ai/llm-leaderboard/tc-leaderboard/weave/agents",
        "query_source": {
            "kind": "wandb_agents_api",
            "api_base_url": "https://trace.wandb.ai",
            "agents_endpoint": "/agents/query",
            "spans_endpoint": "/agents/spans/query",
            "project_id": "llm-leaderboard/tc-leaderboard",
            "agent_name": agent_name,
            "conversation_id": "",
            "conversation_id_contains": "run-1",
            "agents_count": 1,
            "spans_count": 2,
            "matching_span_count": 2,
            "latest_trace_span_count": 2,
        },
        "latest_trace_id": "trace-1" if ok else "",
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
            "usage_required": False,
            "no_error_spans_required": True,
            "required_texts": [],
            "conversation_id": "",
            "conversation_id_contains": "run-1",
        },
        "content_capture_health": {
            "span_count_checked": 2,
            "message_span_count": 1,
            "message_spans_with_content": 1 if ok else 0,
            "message_spans_with_input": 1 if ok else 0,
            "tool_span_count": 1,
            "tool_spans_with_content": 1 if ok else 0,
            "spans_with_valid_timestamps": 2,
            "spans_with_invalid_timestamps": 0,
            "trace_input_tokens": 10,
            "trace_output_tokens": 5,
            "required_text_count": 0,
        },
        "latest_trace_spans_chronological": [
            {
                "started_at": "2026-06-28T00:00:00Z",
                "ended_at": "2026-06-28T00:00:01Z",
                "span_name": "chat",
                "operation_name": "chat",
                "agent_name": agent_name,
                "trace_id": "trace-1",
                "span_id": "span-1",
                "parent_span_id": None,
                "error_type": None,
                "has_input_messages": ok,
                "has_output_messages": ok,
            },
            {
                "started_at": "2026-06-28T00:00:02Z",
                "ended_at": "2026-06-28T00:00:03Z",
                "span_name": "python.exec",
                "operation_name": "execute_tool",
                "agent_name": agent_name,
                "trace_id": "trace-1",
                "span_id": "span-2",
                "parent_span_id": "span-1",
                "tool_name": "python",
                "error_type": None,
            },
        ],
        "checks": [
            {"name": "agent_present", "ok": ok},
            {"name": "latest_trace", "ok": ok},
            {"name": "message_content_capture", "ok": ok},
            {"name": "input_message_capture", "ok": ok},
            {"name": "tool_content_capture", "ok": ok},
            {"name": "trace_timestamp_quality", "ok": True},
            {"name": "trace_order", "ok": ok},
            {"name": "trace_user_message_order", "ok": ok},
            {"name": "trace_final_answer_order", "ok": ok},
        ],
    }


def wandb_completion_payload():
    payload = {
        "ok": True,
        "benchmark": "agentic_math",
        "entity": "llm-leaderboard",
        "project": "tc-leaderboard",
        "run_id": "run-1",
        "query_source": {
            "kind": "wandb_sdk",
            "api": "wandb.Api",
            "timeout_seconds": 60,
            "entity": "llm-leaderboard",
            "project": "tc-leaderboard",
            "run_id": "run-1",
            "run_path": "llm-leaderboard/tc-leaderboard/run-1",
            "benchmark": "agentic_math",
            "summary_source": "run.summary_metrics",
            "artifact_source": "run.logged_artifacts",
            "history_scanned": False,
        },
        "generated_at": time.time(),
        "verification_schema_version": 1,
        "observed_evidence": {
            "run_state": "finished",
            "summary_metrics": {
                "agentic_math/accuracy": {"ok": True, "value": 0.86},
            },
        },
    }
    return add_wandb_run_metadata(payload)


def review_payload(wandb_completion_path: Path | None = None):
    run = {
        "config": "config-a.yaml",
        "log_path": "run.log",
        "wandb_entity": "llm-leaderboard",
        "wandb_project": "tc-leaderboard",
        "wandb_run_id": "run-1",
        "returncode": 0,
        "started_at": time.time() - 90,
        "ended_at": time.time() - 10,
    }
    if wandb_completion_path is not None:
        run["wandb_completion"] = [
            {
                "ok": True,
                "benchmark": "agentic_math",
                "entity": "llm-leaderboard",
                "project": "tc-leaderboard",
                "run_id": "run-1",
                "path": str(wandb_completion_path),
                "sha256": sha256(wandb_completion_path),
            }
        ]
    return {
        "status": "completed",
        "phase": "full",
        "canary": True,
        "model_count": 1,
        "configs": ["config-a.yaml"],
        "run_purpose": "one-model release canary",
        "expected_cost_band": "$10-$20",
        "execution_plan_path": "plan.json",
        "batch_manifest_path": "manifest.json",
        "post_run_cost_command": "estimate",
        "actual_cost_estimate": "$12.34",
        "provider_bill_reference": "billing export",
        "created_at": time.time() - 100,
        "ended_at": time.time(),
        "verify_wandb_completion": wandb_completion_path is not None,
        "verify_weave_agents": False,
        "runs": [run],
    }


def test_sync_review_adds_weave_completion_to_matching_run():
    module = load_module()
    entry = {
        "ok": True,
        "run_id": "run-1",
        "path": "outputs/weave_agents_completion/run-1.json",
        "agent_name": "nejumi-taiwan-openclaw",
    }

    updated, changes, unmatched = module.sync_review(
        review_payload(),
        [entry],
        top_level=False,
        replace=True,
        set_verify_weave_agents=True,
    )

    assert unmatched == []
    assert changes[0]["action"] == "added"
    assert updated["verify_weave_agents"] is True
    assert updated["runs"][0]["weave_agents_completion"] == entry


def test_cli_writes_updated_review_to_output(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "weave.json", weave_completion_payload())
    output = tmp_path / "updated_review.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--run-id",
            "run-1",
            "--output-json",
            str(output),
            "--set-verify-weave-agents",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    updated = json.loads(output.read_text(encoding="utf-8"))
    entry = updated["runs"][0]["weave_agents_completion"]
    assert payload["ok"] is True
    assert updated["verify_weave_agents"] is True
    assert entry["path"].endswith("weave.json")
    assert entry["agent_name"] == "nejumi-taiwan-openclaw"
    assert entry["latest_trace_id"] == "trace-1"
    assert entry["checks_valid"] is True
    assert entry["run_scope_proven"] is True
    assert entry["conversation_id_contains"] == "run-1"
    assert entry["query_source_kind"] == "wandb_agents_api"
    assert entry["query_source_api_base_url"] == "https://trace.wandb.ai"
    assert entry["query_source_agents_endpoint"] == "/agents/query"
    assert entry["query_source_spans_endpoint"] == "/agents/spans/query"
    assert entry["query_source_project_id"] == "llm-leaderboard/tc-leaderboard"


def test_cli_rejects_weave_agents_verifier_missing_query_source(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    payload = weave_completion_payload()
    payload.pop("query_source")
    completion = write_json(tmp_path / "weave.json", payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--run-id",
            "run-1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "query_source must be an object" in result.stderr


def test_cli_rejects_weave_agents_verifier_query_source_mismatch(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    payload = weave_completion_payload()
    payload["query_source"]["kind"] = "manual_json"
    payload["query_source"]["conversation_id_contains"] = "other-run"
    payload["query_source"]["latest_trace_span_count"] = 99
    completion = write_json(tmp_path / "weave.json", payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--run-id",
            "run-1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "query_source.kind must be wandb_agents_api" in result.stderr
    assert "query_source.conversation_id_contains must match required_evidence" in result.stderr
    assert (
        "query_source.latest_trace_span_count must match latest_trace_spans_chronological"
        in result.stderr
    )


def test_cli_rejects_failing_weave_agents_verifier_by_default(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "weave.json", weave_completion_payload(ok=False))

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--run-id",
            "run-1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "not a passing Weave Agents verifier JSON" in result.stderr


def test_cli_writes_validation_failed_report_for_failing_weave_verifier(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "weave.json", weave_completion_payload(ok=False))
    report = tmp_path / "sync_report.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--run-id",
            "run-1",
            "--report-json",
            str(report),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["status"] == "validation_failed"
    assert payload["dry_run"] is True
    assert payload["entry_count"] == 0
    assert payload["change_count"] == 0
    assert payload["completion_paths"] == [str(completion)]
    assert "not a passing Weave Agents verifier JSON" in payload["validation_errors"][0]
    assert json.loads(report.read_text(encoding="utf-8")) == payload


def test_cli_rejects_weave_agents_verifier_without_run_scope(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    payload = weave_completion_payload()
    payload["required_evidence"]["conversation_id_contains"] = ""
    payload["query_source"]["conversation_id_contains"] = ""
    completion = write_json(tmp_path / "weave.json", payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--run-id",
            "run-1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "does not prove W&B run scope" in result.stderr


def test_cli_rejects_tool_before_message_payload(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    payload = weave_completion_payload()
    spans = payload["latest_trace_spans_chronological"]
    spans[1]["started_at"] = "2026-06-27T23:59:59Z"
    spans[1]["ended_at"] = "2026-06-28T00:00:00Z"
    payload["latest_trace_spans_chronological"] = [spans[1], spans[0]]
    completion = write_json(tmp_path / "weave.json", payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--run-id",
            "run-1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "tool span starts before the first message span" in result.stderr


def test_cli_rejects_missing_final_answer_order_check(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    payload = weave_completion_payload()
    payload["checks"] = [
        check
        for check in payload["checks"]
        if check.get("name") != "trace_final_answer_order"
    ]
    completion = write_json(tmp_path / "weave.json", payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--run-id",
            "run-1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "checks missing required check(s): trace_final_answer_order" in result.stderr


def test_cli_rejects_missing_timestamp_quality_check(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    payload = weave_completion_payload()
    payload["checks"] = [
        check
        for check in payload["checks"]
        if check.get("name") != "trace_timestamp_quality"
    ]
    completion = write_json(tmp_path / "weave.json", payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--run-id",
            "run-1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "checks missing required check(s): trace_timestamp_quality" in result.stderr


def test_cli_rejects_missing_final_answer_order_requirement(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    payload = weave_completion_payload()
    payload["required_evidence"].pop("trace_final_answer_order_required")
    completion = write_json(tmp_path / "weave.json", payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--run-id",
            "run-1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert (
        "required_evidence.trace_final_answer_order_required must be true"
        in result.stderr
    )


def test_cli_rejects_missing_timestamp_quality_requirement(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    payload = weave_completion_payload()
    payload["required_evidence"].pop("trace_timestamp_quality_required")
    completion = write_json(tmp_path / "weave.json", payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--run-id",
            "run-1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "required_evidence.trace_timestamp_quality_required must be true" in result.stderr


def test_cli_rejects_invalid_span_timestamp_payload(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    payload = weave_completion_payload()
    payload["latest_trace_spans_chronological"][1]["started_at"] = ""
    payload["content_capture_health"]["spans_with_valid_timestamps"] = 1
    payload["content_capture_health"]["spans_with_invalid_timestamps"] = 1
    completion = write_json(tmp_path / "weave.json", payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--run-id",
            "run-1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "has missing or invalid timestamps" in result.stderr
    assert "latest trace has invalid timestamps" in result.stderr


def test_cli_rejects_missing_input_message_requirement(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    payload = weave_completion_payload()
    payload["required_evidence"].pop("input_message_required")
    completion = write_json(tmp_path / "weave.json", payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--run-id",
            "run-1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "required_evidence.input_message_required must be true" in result.stderr


def test_cli_rejects_missing_input_message_visibility(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    payload = weave_completion_payload()
    payload["content_capture_health"]["message_spans_with_input"] = 0
    completion = write_json(tmp_path / "weave.json", payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--run-id",
            "run-1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "required user/problem input message is not visible" in result.stderr


def test_cli_rejects_missing_required_text_capture_check(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    payload = weave_completion_payload()
    payload["required_evidence"]["required_texts"] = [
        "CANARY_ID",
        "CANARY_RESULT CANARY_ID 91",
    ]
    payload["content_capture_health"]["required_text_count"] = 2
    completion = write_json(tmp_path / "weave.json", payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--run-id",
            "run-1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "checks missing required check(s): required_text_capture" in result.stderr


def test_cli_rejects_required_text_count_below_required_texts(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    payload = weave_completion_payload()
    payload["required_evidence"]["required_texts"] = [
        "CANARY_ID",
        "CANARY_RESULT CANARY_ID 91",
    ]
    payload["checks"].append({"name": "required_text_capture", "ok": True})
    payload["content_capture_health"]["required_text_count"] = 1
    completion = write_json(tmp_path / "weave.json", payload)

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--run-id",
            "run-1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "content_capture_health.required_text_count must be at least 2" in result.stderr


def test_cli_fails_on_unmatched_run_id(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    payload = weave_completion_payload()
    payload["required_evidence"]["conversation_id_contains"] = "other-run"
    payload["query_source"]["conversation_id_contains"] = "other-run"
    completion = write_json(tmp_path / "weave.json", payload)
    report = tmp_path / "report.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--run-id",
            "other-run",
            "--report-json",
            str(report),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "unmatched_run_id"
    assert payload["unmatched_count"] == 1
    assert json.loads(report.read_text(encoding="utf-8"))["ok"] is False


def test_cli_validates_matching_dry_run_report_before_apply(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "weave.json", weave_completion_payload())
    dry_run_report = tmp_path / "sync_dry_run.json"
    output = tmp_path / "updated_review.json"
    base_args = [
        "python3",
        str(SCRIPT),
        "--review-json",
        str(review),
        "--completion-json",
        str(completion),
        "--run-id",
        "run-1",
        "--set-verify-weave-agents",
    ]

    dry_run = subprocess.run(
        [*base_args, "--report-json", str(dry_run_report)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    dry_run_payload = json.loads(dry_run_report.read_text(encoding="utf-8"))
    assert dry_run_payload["dry_run"] is True
    assert dry_run_payload["in_place"] is False
    assert dry_run_payload["output_path"] == ""
    assert dry_run_payload["entry_count"] == 1
    assert dry_run_payload["entries"][0]["run_id"] == "run-1"
    assert dry_run_payload["source_review_sha256"] == sha256(review)

    apply_result = subprocess.run(
        [
            *base_args,
            "--output-json",
            str(output),
            "--validated-dry-run-report-json",
            str(dry_run_report),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert apply_result.returncode == 0, apply_result.stderr
    report = json.loads(apply_result.stdout)
    assert report["dry_run"] is False
    assert report["entries"] == dry_run_payload["entries"]
    assert report["changes"] == dry_run_payload["changes"]
    assert report["validated_dry_run_report_json"] == str(dry_run_report)
    assert output.exists()
    updated = json.loads(output.read_text(encoding="utf-8"))
    assert (
        updated["runs"][0]["weave_agents_completion"]["sync_dry_run_report_json"]
        == str(dry_run_report)
    )
    assert (
        updated["runs"][0]["weave_agents_completion"]["sync_dry_run_source_review_json"]
        == str(review)
    )
    assert (
        updated["runs"][0]["weave_agents_completion"]["sync_dry_run_source_review_sha256"]
        == sha256(review)
    )


def test_cli_rejects_mismatched_validated_dry_run_report(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "weave.json", weave_completion_payload())
    dry_run_report = tmp_path / "sync_dry_run.json"
    output = tmp_path / "updated_review.json"
    base_args = [
        "python3",
        str(SCRIPT),
        "--review-json",
        str(review),
        "--completion-json",
        str(completion),
        "--run-id",
        "run-1",
        "--set-verify-weave-agents",
    ]

    dry_run = subprocess.run(
        [*base_args, "--report-json", str(dry_run_report)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    payload = json.loads(dry_run_report.read_text(encoding="utf-8"))
    payload["entries"][0]["latest_trace_id"] = "other-trace"
    dry_run_report.write_text(json.dumps(payload), encoding="utf-8")

    apply_result = subprocess.run(
        [
            *base_args,
            "--output-json",
            str(output),
            "--validated-dry-run-report-json",
            str(dry_run_report),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert apply_result.returncode != 0
    assert "not a valid matching dry-run report" in apply_result.stderr
    assert "entries does not match current sync" in apply_result.stderr
    assert not output.exists()


def test_cli_rejects_validated_dry_run_report_when_review_status_changed(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "weave.json", weave_completion_payload())
    dry_run_report = tmp_path / "sync_dry_run.json"
    output = tmp_path / "updated_review.json"
    base_args = [
        "python3",
        str(SCRIPT),
        "--review-json",
        str(review),
        "--completion-json",
        str(completion),
        "--run-id",
        "run-1",
        "--set-verify-weave-agents",
    ]

    dry_run = subprocess.run(
        [*base_args, "--report-json", str(dry_run_report)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    review_payload_changed = json.loads(review.read_text(encoding="utf-8"))
    review_payload_changed["status"] = "reviewed_after_dry_run"
    review.write_text(json.dumps(review_payload_changed), encoding="utf-8")

    apply_result = subprocess.run(
        [
            *base_args,
            "--output-json",
            str(output),
            "--validated-dry-run-report-json",
            str(dry_run_report),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert apply_result.returncode != 0
    assert "not a valid matching dry-run report" in apply_result.stderr
    assert "before_status does not match current sync" in apply_result.stderr
    assert "after_status does not match current sync" in apply_result.stderr
    assert not output.exists()


def test_cli_rejects_validated_dry_run_report_when_review_content_changed(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "weave.json", weave_completion_payload())
    dry_run_report = tmp_path / "sync_dry_run.json"
    output = tmp_path / "updated_review.json"
    base_args = [
        "python3",
        str(SCRIPT),
        "--review-json",
        str(review),
        "--completion-json",
        str(completion),
        "--run-id",
        "run-1",
        "--set-verify-weave-agents",
    ]

    dry_run = subprocess.run(
        [*base_args, "--report-json", str(dry_run_report)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    review_payload_changed = json.loads(review.read_text(encoding="utf-8"))
    review_payload_changed["provider_bill_reference"] = "invoice-after-dry-run"
    review.write_text(json.dumps(review_payload_changed), encoding="utf-8")

    apply_result = subprocess.run(
        [
            *base_args,
            "--output-json",
            str(output),
            "--validated-dry-run-report-json",
            str(dry_run_report),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert apply_result.returncode != 0
    assert "not a valid matching dry-run report" in apply_result.stderr
    assert "source_review_sha256 does not match current sync" in apply_result.stderr
    assert not output.exists()


def test_cli_rejects_validated_dry_run_report_without_apply_target(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "weave.json", weave_completion_payload())

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--run-id",
            "run-1",
            "--set-verify-weave-agents",
            "--validated-dry-run-report-json",
            str(tmp_path / "sync_dry_run.json"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--validated-dry-run-report-json requires --in-place or --output-json" in result.stderr


def test_cli_rejects_in_place_without_validated_dry_run_report(tmp_path):
    review = write_json(tmp_path / "review.json", review_payload())
    completion = write_json(tmp_path / "weave.json", weave_completion_payload())

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--review-json",
            str(review),
            "--completion-json",
            str(completion),
            "--run-id",
            "run-1",
            "--set-verify-weave-agents",
            "--in-place",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--in-place requires --validated-dry-run-report-json" in result.stderr


def test_sync_output_passes_paid_review_doctor(tmp_path):
    wandb_completion = write_json(
        tmp_path / "wandb_completion.json",
        wandb_completion_payload(),
    )
    review = write_json(tmp_path / "review.json", review_payload(wandb_completion))
    weave_completion = write_json(tmp_path / "weave.json", weave_completion_payload())
    dry_run_report = tmp_path / "sync_dry_run.json"
    output = tmp_path / "updated_review.json"
    base_args = [
        "python3",
        str(SCRIPT),
        "--review-json",
        str(review),
        "--completion-json",
        str(weave_completion),
        "--run-id",
        "run-1",
        "--set-verify-weave-agents",
    ]

    dry_run = subprocess.run(
        [*base_args, "--report-json", str(dry_run_report)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert dry_run.returncode == 0, dry_run.stderr

    apply_result = subprocess.run(
        [
            *base_args,
            "--output-json",
            str(output),
            "--validated-dry-run-report-json",
            str(dry_run_report),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert apply_result.returncode == 0, apply_result.stderr

    doctor_result = subprocess.run(
        [
            "python3",
            str(PAID_REVIEW_SCRIPT),
            "--review-json",
            str(output),
            "--required-wandb-benchmark",
            "agentic_math",
            "--required-wandb-run-id",
            "agentic_math=run-1",
            "--require-one-model-canary",
            "--weave-agents-completion-max-age-seconds",
            "-1",
            "--fail-on-invalid",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert doctor_result.returncode == 0, doctor_result.stderr
    report = json.loads(doctor_result.stdout)
    entry = report["gates"][0]["records"][0]["weave_agents_completion_entries"][0]
    assert entry["verified"] is True
    assert entry["agent_name_matches"] is True
    assert entry["checks_valid"] is True
