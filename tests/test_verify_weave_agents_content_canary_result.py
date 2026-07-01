import importlib.util
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = REPO_ROOT / "scripts" / "tools" / "verify_weave_agents_content_canary_result.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def load_gate_contract_module():
    path = REPO_ROOT / "scripts" / "tools" / "weave_content_canary_gate_contract.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def write_plan(tmp_path, *, will_call_paid_model_api=False, task_id="weave_agents_content_canary_TEST"):
    plan_dir = tmp_path / "plans"
    plan_dir.mkdir(parents=True)
    plan_file = plan_dir / f"{task_id}.json"
    run_command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py"),
        "run",
        "--benchmark-id",
        "agentic_math",
        "--task-id",
        task_id,
        "--model",
        "openai-direct/test-mini",
    ]
    run_command_sha256 = hashlib.sha256(
        json.dumps(run_command, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    plan_file.write_text(
        json.dumps(
            {
                "canary_id": "TEST",
                "task_id": task_id,
                "will_call_paid_model_api": will_call_paid_model_api,
                "model": "openai-direct/test-mini",
                "thinking": "low",
                "expected_sidecar": str(tmp_path / "agentic_math" / task_id / "openclaw_result.json"),
                "agents_diagnostic_file": str(
                    tmp_path / "agents_diagnostics" / f"{task_id}.agents.json"
                ),
                "prompt_file": str(tmp_path / "prompts" / f"{task_id}.md"),
                "agent_name": "nejumi-taiwan-openclaw",
                "entity": "llm-leaderboard",
                "project": "tc-leaderboard",
                "verification_requirements": {
                    "expected_request_models": ["openai-direct/test-mini", "test-mini"],
                },
                "run_command": run_command,
                "run_command_sha256": run_command_sha256,
                "nemoclaw": {
                    "required": True,
                    "enabled": True,
                    "bin": "nemoclaw",
                    "sandbox": "nejumi-taiwan",
                    "workdir": "/sandbox",
                },
                "nemoclaw_openclaw_config_preflight": {
                    "required_before_openclaw": bool(will_call_paid_model_api),
                    "ran": bool(will_call_paid_model_api),
                    "ok": True,
                    "model": "openai-direct/test-mini",
                    "provider": "openai-direct",
                    "model_id": "test-mini",
                    "config_path": "/sandbox/.openclaw/openclaw.json",
                    "command": [
                        "nemoclaw",
                        "sandbox",
                        "exec",
                        "nejumi-taiwan",
                        "--no-tty",
                        "--timeout",
                        "30",
                        "--",
                        "cat",
                        "/sandbox/.openclaw/openclaw.json",
                    ],
                    "returncode": 0 if will_call_paid_model_api else None,
                    "checks": [
                        {
                            "name": (
                                "NeMoClaw sandbox OpenClaw config is readable: "
                                "/sandbox/.openclaw/openclaw.json"
                            ),
                            "ok": True,
                            "detail": "bytes=1234",
                        },
                        {
                            "name": "NeMoClaw sandbox OpenClaw openai-direct provider exists",
                            "ok": True,
                            "detail": "present",
                        },
                        {
                            "name": (
                                "NeMoClaw sandbox OpenClaw model is registered: "
                                "openai-direct/test-mini"
                            ),
                            "ok": True,
                            "detail": '["test-mini"]',
                        },
                        {
                            "name": "NeMoClaw sandbox OpenClaw Weave plugin is enabled",
                            "ok": True,
                            "detail": "True",
                        },
                    ]
                    if will_call_paid_model_api
                    else [],
                    "errors": [],
                },
            }
        ),
        encoding="utf-8",
    )
    return plan_file


def write_command_result(plan_file, task_id, payload):
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    payload = {
        **{
            "canary_id": plan.get("canary_id"),
            "task_id": plan.get("task_id"),
            "model": plan.get("model"),
            "thinking": plan.get("thinking"),
            "plan_file": str(plan_file),
            "prompt_file": plan.get("prompt_file"),
            "expected_sidecar": plan.get("expected_sidecar"),
            "run_command": plan.get("run_command"),
            "run_command_sha256": plan.get("run_command_sha256"),
        },
        **payload,
    }
    path = plan_file.with_name(f"{task_id}.command_result.json")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_verifier(tmp_path, task_id, payload):
    path = tmp_path / "verifier" / task_id / "attempt_01.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_agents_diagnostic(tmp_path, task_id, payload):
    path = tmp_path / "agents_diagnostics" / f"{task_id}.agents.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def passing_verifier_payload(task_id="weave_agents_content_canary_PASS"):
    return {
        "ok": True,
        "verification_schema_version": 1,
        "generated_at": 1,
        "latest_trace_id": "trace-1",
        "project_id": "llm-leaderboard/tc-leaderboard",
        "agent_name": "nejumi-taiwan-openclaw",
        "query_source": {
            "kind": "wandb_agents_api",
            "api_base_url": "https://trace.wandb.ai",
            "agents_endpoint": "/agents/query",
            "spans_endpoint": "/agents/spans/query",
            "project_id": "llm-leaderboard/tc-leaderboard",
            "agent_name": "nejumi-taiwan-openclaw",
            "conversation_id": "",
            "conversation_id_contains": task_id,
            "agents_count": 1,
            "spans_count": 2,
            "matching_span_count": 2,
            "latest_trace_span_count": 2,
        },
        "required_evidence": {
            "input_message_required": True,
            "trace_timestamp_quality_required": True,
            "trace_final_answer_order_required": True,
            "expected_request_models": ["openai-direct/test-mini", "test-mini"],
            "conversation_id": "",
            "conversation_id_contains": task_id,
        },
        "content_capture_health": {
            "message_spans_with_content": 1,
            "message_spans_with_input": 1,
            "tool_spans_with_content": 1,
            "spans_with_valid_timestamps": 2,
            "spans_with_invalid_timestamps": 0,
            "trace_input_tokens": 10,
            "trace_output_tokens": 2,
            "request_model_count": 1,
        },
        "checks": [
            {"name": "message_content_capture", "ok": True},
            {"name": "input_message_capture", "ok": True},
            {"name": "tool_content_capture", "ok": True},
            {"name": "usage", "ok": True},
            {
                "name": "request_model",
                "ok": True,
                "expected_request_models": ["openai-direct/test-mini", "test-mini"],
                "observed_request_models": ["test-mini"],
            },
            {"name": "trace_timestamp_quality", "ok": True},
            {"name": "trace_order", "ok": True},
            {"name": "trace_user_message_order", "ok": True},
            {"name": "trace_final_answer_order", "ok": True},
        ],
        "latest_trace_spans_chronological": [
            {
                "started_at": "2026-06-27T00:00:01.000000",
                "ended_at": "2026-06-27T00:00:02.000000",
                "span_name": "chat",
                "operation_name": "chat",
                "agent_name": "nejumi-taiwan-openclaw",
                "provider_name": "openai-direct",
                "request_model": "test-mini",
                "conversation_id": task_id,
                "trace_id": "trace-1",
                "span_id": "chat-1",
                "parent_span_id": "root",
                "tool_name": None,
                "error_type": "",
                "has_input_messages": True,
                "has_output_messages": True,
                "input_message_count": 1,
                "output_message_count": 1,
                "has_tool_call_arguments": False,
                "has_tool_call_result": False,
            },
            {
                "started_at": "2026-06-27T00:00:03.000000",
                "ended_at": "2026-06-27T00:00:04.000000",
                "span_name": "execute_tool",
                "operation_name": "execute_tool",
                "agent_name": "nejumi-taiwan-openclaw",
                "provider_name": "openai-direct",
                "request_model": "test-mini",
                "conversation_id": task_id,
                "trace_id": "trace-1",
                "span_id": "tool-1",
                "parent_span_id": "root",
                "tool_name": "exec",
                "error_type": "",
                "has_input_messages": False,
                "has_output_messages": False,
                "input_message_count": 0,
                "output_message_count": 0,
                "has_tool_call_arguments": True,
                "has_tool_call_result": True,
            },
        ],
    }


def passing_agents_diagnostic_payload(task_id="weave_agents_content_canary_PASS"):
    return {
        "diagnostic_schema_version": 1,
        "generated_at": 1,
        "project_id": "llm-leaderboard/tc-leaderboard",
        "agent_name_filter": "nejumi-taiwan-openclaw",
        "agents_url": "https://wandb.ai/llm-leaderboard/tc-leaderboard/weave/agents",
        "query_source": {
            "kind": "wandb_agents_api",
            "api_base_url": "https://trace.wandb.ai",
            "agents_endpoint": "/agents/query",
            "spans_endpoint": "/agents/spans/query",
            "project_id": "llm-leaderboard/tc-leaderboard",
            "agent_name": "nejumi-taiwan-openclaw",
            "conversation_id": "",
            "conversation_id_contains": task_id,
            "limit": 30,
            "span_limit": 120,
            "agents_count": 1,
            "spans_count": 3,
            "matching_span_count": 3,
            "latest_trace_span_count": 3,
        },
        "agents": [{"agent_name": "nejumi-taiwan-openclaw"}],
        "total_count": 1,
        "latest_trace_id": "trace-1",
        "latest_trace_spans_chronological": [
            {
                "started_at": "2026-06-27T00:00:01.000000",
                "ended_at": "2026-06-27T00:00:02.000000",
                "span_name": "chat",
                "operation_name": "chat",
                "agent_name": "nejumi-taiwan-openclaw",
                "conversation_id": task_id,
                "trace_id": "trace-1",
                "span_id": "chat-1",
                "parent_span_id": "root",
                "tool_name": None,
                "error_type": "",
                "has_input_messages": True,
                "has_output_messages": False,
                "input_message_count": 1,
                "output_message_count": 0,
                "has_tool_call_arguments": False,
                "has_tool_call_result": False,
                "has_final_answer_marker": False,
            },
            {
                "started_at": "2026-06-27T00:00:03.000000",
                "ended_at": "2026-06-27T00:00:04.000000",
                "span_name": "execute_tool",
                "operation_name": "execute_tool",
                "agent_name": "nejumi-taiwan-openclaw",
                "conversation_id": task_id,
                "trace_id": "trace-1",
                "span_id": "tool-1",
                "parent_span_id": "root",
                "tool_name": "exec",
                "error_type": "",
                "has_input_messages": False,
                "has_output_messages": False,
                "input_message_count": 0,
                "output_message_count": 0,
                "has_tool_call_arguments": True,
                "has_tool_call_result": True,
                "has_final_answer_marker": False,
            },
            {
                "started_at": "2026-06-27T00:00:05.000000",
                "ended_at": "2026-06-27T00:00:06.000000",
                "span_name": "chat",
                "operation_name": "chat",
                "agent_name": "nejumi-taiwan-openclaw",
                "conversation_id": task_id,
                "trace_id": "trace-1",
                "span_id": "answer-1",
                "parent_span_id": "root",
                "tool_name": None,
                "error_type": "",
                "has_input_messages": False,
                "has_output_messages": True,
                "input_message_count": 0,
                "output_message_count": 1,
                "has_tool_call_arguments": False,
                "has_tool_call_result": False,
                "has_final_answer_marker": True,
            },
        ],
        "content_capture_health": {
            "span_count_checked": 3,
            "message_span_count": 2,
            "message_spans_with_content": 2,
            "message_spans_with_input": 1,
            "tool_span_count": 1,
            "tool_spans_with_content": 1,
            "final_answer_span_count": 1,
            "spans_with_valid_timestamps": 3,
            "spans_with_invalid_timestamps": 0,
            "trace_timestamp_quality_ok": True,
            "trace_order_ok": True,
            "trace_user_message_order_ok": True,
            "trace_final_answer_order_ok": True,
        },
        "trace_order_health": {
            "timestamp_quality_ok": True,
            "timestamp_issue_count": 0,
            "timestamp_issues": [],
            "message_span_count": 2,
            "message_spans_with_input": 1,
            "tool_span_count": 1,
            "final_answer_span_count": 1,
            "trace_order_ok": True,
            "trace_user_message_order_ok": True,
            "trace_final_answer_order_ok": True,
            "order_issues": [],
        },
        "latest_spans_api_order": [],
    }


def test_prepare_only_canary_is_not_run_gate(tmp_path):
    module = load_module()
    plan_file = write_plan(tmp_path, will_call_paid_model_api=False)

    summary = module.build_gate_summary(plan_file=plan_file)

    assert summary["ok"] is False
    assert summary["status"] == "not_run"
    assert summary["will_call_paid_model_api"] is False
    assert summary["paid_api_attempted"] is False


def test_provider_quota_failure_is_explicit_gate_status(tmp_path):
    module = load_module()
    task_id = "weave_agents_content_canary_QUOTA"
    plan_file = write_plan(tmp_path, will_call_paid_model_api=True, task_id=task_id)
    write_command_result(
        plan_file,
        task_id,
        {
            "ok": False,
            "returncode": 1,
            "failure": {
                "kind": "provider_quota",
                "detail": "provider returned insufficient_quota before a scoreable canary trace was produced",
            },
        },
    )

    summary = module.build_gate_summary(plan_file=plan_file)

    assert summary["ok"] is False
    assert summary["status"] == "provider_failure"
    assert summary["failure_kind"] == "provider_quota"
    assert "quota" in summary["recommended_next_action"]


def test_external_action_approval_block_is_not_paid_api_attempt(tmp_path):
    module = load_module()
    task_id = "weave_agents_content_canary_APPROVAL_BLOCK"
    plan_file = write_plan(tmp_path, will_call_paid_model_api=True, task_id=task_id)
    write_command_result(
        plan_file,
        task_id,
        {
            "ok": False,
            "returncode": 2,
            "blocked_before_openclaw": True,
            "paid_api_attempted": False,
            "failure": {
                "kind": "external_action_approval_missing",
                "detail": (
                    "external-action approval was missing or invalid; OpenClaw "
                    "and provider execution were not started"
                ),
            },
        },
    )

    summary = module.build_gate_summary(plan_file=plan_file)

    assert summary["ok"] is False
    assert summary["status"] == "external_action_approval_missing"
    assert summary["failure_kind"] == "external_action_approval_missing"
    assert summary["paid_api_attempted"] is False
    assert "approve" in summary["recommended_next_action"].lower()


def test_provider_quota_can_be_inferred_from_legacy_sidecar(tmp_path):
    module = load_module()
    task_id = "weave_agents_content_canary_LEGACY_QUOTA"
    plan_file = write_plan(tmp_path, will_call_paid_model_api=True, task_id=task_id)
    sidecar = tmp_path / "agentic_math" / task_id / "openclaw_result.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(
        json.dumps({"stderr": "code=insufficient_quota message=You exceeded your current quota"}),
        encoding="utf-8",
    )
    write_command_result(plan_file, task_id, {"ok": False, "returncode": 1})

    summary = module.build_gate_summary(plan_file=plan_file)

    assert summary["status"] == "provider_failure"
    assert summary["failure_kind"] == "provider_quota"


def test_successful_verifier_passes_gate(tmp_path):
    module = load_module()
    task_id = "weave_agents_content_canary_PASS"
    plan_file = write_plan(tmp_path, will_call_paid_model_api=True, task_id=task_id)
    write_command_result(plan_file, task_id, {"ok": True, "returncode": 0})
    write_verifier(tmp_path, task_id, passing_verifier_payload(task_id))
    write_agents_diagnostic(tmp_path, task_id, passing_agents_diagnostic_payload(task_id))

    summary = module.build_gate_summary(plan_file=plan_file)

    assert summary["ok"] is True
    assert summary["status"] == "passed"
    assert summary["weave_verifier_ok"] is True
    assert summary["weave_verifier_validation_issues"] == []
    assert summary["agents_diagnostic_ok"] is True
    assert summary["agents_diagnostic_validation_issues"] == []
    assert summary["expected_request_models"] == ["openai-direct/test-mini", "test-mini"]
    assert summary["observed_request_models"] == ["test-mini"]
    assert summary["span_request_models"] == ["test-mini"]
    assert summary["request_model_proven"] is True
    assert summary["nemoclaw_openclaw_config_preflight"]["ok"] is True
    assert summary["command_result_contract_issues"] == []


def test_successful_command_result_must_match_plan_identity(tmp_path):
    module = load_module()
    task_id = "weave_agents_content_canary_COMMAND_MISMATCH"
    plan_file = write_plan(tmp_path, will_call_paid_model_api=True, task_id=task_id)
    write_command_result(
        plan_file,
        task_id,
        {
            "ok": True,
            "returncode": 0,
            "task_id": "weave_agents_content_canary_OTHER",
            "run_command_sha256": "0" * 64,
        },
    )
    write_verifier(tmp_path, task_id, passing_verifier_payload(task_id))
    write_agents_diagnostic(tmp_path, task_id, passing_agents_diagnostic_payload(task_id))

    summary = module.build_gate_summary(plan_file=plan_file)

    assert summary["ok"] is False
    assert summary["status"] == "command_result_contract_invalid"
    assert "command_result.task_id must match plan.task_id" in summary["detail"]
    assert "command_result.run_command_sha256 must match" in summary["detail"]


def test_successful_verifier_summary_satisfies_shared_gate_contract(tmp_path):
    module = load_module()
    contract_module = load_gate_contract_module()
    task_id = "weave_agents_content_canary_PASS"
    plan_file = write_plan(tmp_path, will_call_paid_model_api=True, task_id=task_id)
    write_command_result(plan_file, task_id, {"ok": True, "returncode": 0})
    write_verifier(tmp_path, task_id, passing_verifier_payload(task_id))
    write_agents_diagnostic(tmp_path, task_id, passing_agents_diagnostic_payload(task_id))

    summary = module.build_gate_summary(plan_file=plan_file)

    assert contract_module.weave_content_canary_gate_contract_issues(summary) == []


def test_successful_verifier_without_nemoclaw_openclaw_config_preflight_does_not_pass_gate(
    tmp_path,
):
    module = load_module()
    task_id = "weave_agents_content_canary_PREFLIGHT_MISSING"
    plan_file = write_plan(tmp_path, will_call_paid_model_api=True, task_id=task_id)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    plan.pop("nemoclaw_openclaw_config_preflight")
    plan_file.write_text(json.dumps(plan), encoding="utf-8")
    write_command_result(plan_file, task_id, {"ok": True, "returncode": 0})
    write_verifier(tmp_path, task_id, passing_verifier_payload(task_id))
    write_agents_diagnostic(tmp_path, task_id, passing_agents_diagnostic_payload(task_id))

    summary = module.build_gate_summary(plan_file=plan_file)

    assert summary["ok"] is False
    assert summary["status"] == "nemoclaw_config_preflight_invalid"
    assert "nemoclaw_openclaw_config_preflight must be an object" in summary["detail"]


def test_successful_verifier_with_failed_nemoclaw_openclaw_config_preflight_does_not_pass_gate(
    tmp_path,
):
    module = load_module()
    task_id = "weave_agents_content_canary_PREFLIGHT_FAILED"
    plan_file = write_plan(tmp_path, will_call_paid_model_api=True, task_id=task_id)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    plan["nemoclaw_openclaw_config_preflight"]["ok"] = False
    plan["nemoclaw_openclaw_config_preflight"]["errors"] = [
        "NeMoClaw sandbox OpenClaw model is registered: openai-direct/test-mini"
    ]
    plan_file.write_text(json.dumps(plan), encoding="utf-8")
    write_command_result(plan_file, task_id, {"ok": True, "returncode": 0})
    write_verifier(tmp_path, task_id, passing_verifier_payload(task_id))
    write_agents_diagnostic(tmp_path, task_id, passing_agents_diagnostic_payload(task_id))

    summary = module.build_gate_summary(plan_file=plan_file)

    assert summary["ok"] is False
    assert summary["status"] == "nemoclaw_config_preflight_invalid"
    assert "nemoclaw_openclaw_config_preflight.ok must be true" in summary["detail"]


def test_successful_verifier_summary_requires_nemoclaw_metadata(tmp_path):
    module = load_module()
    contract_module = load_gate_contract_module()
    task_id = "weave_agents_content_canary_PASS"
    plan_file = write_plan(tmp_path, will_call_paid_model_api=True, task_id=task_id)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    plan.pop("nemoclaw")
    plan_file.write_text(json.dumps(plan), encoding="utf-8")
    write_command_result(plan_file, task_id, {"ok": True, "returncode": 0})
    write_verifier(tmp_path, task_id, passing_verifier_payload(task_id))
    write_agents_diagnostic(tmp_path, task_id, passing_agents_diagnostic_payload(task_id))

    summary = module.build_gate_summary(plan_file=plan_file)

    assert summary["ok"] is True
    issues = contract_module.weave_content_canary_gate_contract_issues(summary)
    assert "nemoclaw.required must be true" in issues or "nemoclaw must be an object" in issues


def test_successful_verifier_without_agents_diagnostic_does_not_pass_gate(tmp_path):
    module = load_module()
    task_id = "weave_agents_content_canary_DIAGNOSTIC_MISSING"
    plan_file = write_plan(tmp_path, will_call_paid_model_api=True, task_id=task_id)
    write_command_result(plan_file, task_id, {"ok": True, "returncode": 0})
    write_verifier(tmp_path, task_id, passing_verifier_payload(task_id))

    summary = module.build_gate_summary(plan_file=plan_file)

    assert summary["ok"] is False
    assert summary["status"] == "agents_diagnostic_invalid"
    assert "Agents diagnostic JSON is missing" in summary["detail"]


def test_agents_diagnostic_with_wrong_scope_does_not_pass_gate(tmp_path):
    module = load_module()
    task_id = "weave_agents_content_canary_DIAGNOSTIC_SCOPE"
    plan_file = write_plan(tmp_path, will_call_paid_model_api=True, task_id=task_id)
    write_command_result(plan_file, task_id, {"ok": True, "returncode": 0})
    write_verifier(tmp_path, task_id, passing_verifier_payload(task_id))
    payload = passing_agents_diagnostic_payload(task_id)
    payload["query_source"]["conversation_id_contains"] = "other-task"
    for span in payload["latest_trace_spans_chronological"]:
        span["conversation_id"] = "other-task"
    write_agents_diagnostic(tmp_path, task_id, payload)

    summary = module.build_gate_summary(plan_file=plan_file)

    assert summary["ok"] is False
    assert summary["status"] == "agents_diagnostic_invalid"
    assert "conversation_id_contains must scope" in summary["detail"]


def test_ok_verifier_without_native_agents_query_source_does_not_pass_gate(tmp_path):
    module = load_module()
    task_id = "weave_agents_content_canary_QUERY_SOURCE_MISSING"
    plan_file = write_plan(tmp_path, will_call_paid_model_api=True, task_id=task_id)
    write_command_result(plan_file, task_id, {"ok": True, "returncode": 0})
    payload = passing_verifier_payload(task_id)
    payload.pop("query_source")
    write_verifier(tmp_path, task_id, payload)

    summary = module.build_gate_summary(plan_file=plan_file)

    assert summary["ok"] is False
    assert summary["status"] == "weave_verifier_schema_invalid"
    assert "query_source must be an object from the native W&B Agents API verifier" in summary["detail"]


def test_ok_verifier_with_wrong_canary_scope_does_not_pass_gate(tmp_path):
    module = load_module()
    task_id = "weave_agents_content_canary_SCOPE"
    plan_file = write_plan(tmp_path, will_call_paid_model_api=True, task_id=task_id)
    write_command_result(plan_file, task_id, {"ok": True, "returncode": 0})
    payload = passing_verifier_payload(task_id)
    payload["query_source"]["conversation_id_contains"] = "other-task"
    payload["required_evidence"]["conversation_id_contains"] = "other-task"
    for span in payload["latest_trace_spans_chronological"]:
        span["conversation_id"] = "other-task"
    write_verifier(tmp_path, task_id, payload)

    summary = module.build_gate_summary(plan_file=plan_file)

    assert summary["ok"] is False
    assert summary["status"] == "weave_verifier_schema_invalid"
    assert "must scope the query to the canary task_id" in summary["detail"]
    assert "conversation_id must contain" in summary["detail"]


def test_ok_verifier_without_current_schema_does_not_pass_gate(tmp_path):
    module = load_module()
    task_id = "weave_agents_content_canary_LEGACY_OK"
    plan_file = write_plan(tmp_path, will_call_paid_model_api=True, task_id=task_id)
    write_command_result(plan_file, task_id, {"ok": True, "returncode": 0})
    payload = passing_verifier_payload(task_id)
    payload.pop("verification_schema_version")
    write_verifier(tmp_path, task_id, payload)

    summary = module.build_gate_summary(plan_file=plan_file)

    assert summary["ok"] is False
    assert summary["status"] == "weave_verifier_schema_invalid"
    assert "verification_schema_version must be 1" in summary["detail"]
    assert "verification_schema_version must be 1" in summary["weave_verifier_validation_issues"]


def test_ok_verifier_without_final_answer_requirement_does_not_pass_gate(tmp_path):
    module = load_module()
    task_id = "weave_agents_content_canary_FINAL_REQUIREMENT_MISSING"
    plan_file = write_plan(tmp_path, will_call_paid_model_api=True, task_id=task_id)
    write_command_result(plan_file, task_id, {"ok": True, "returncode": 0})
    payload = passing_verifier_payload(task_id)
    payload["required_evidence"].pop("trace_final_answer_order_required")
    write_verifier(tmp_path, task_id, payload)

    summary = module.build_gate_summary(plan_file=plan_file)

    assert summary["ok"] is False
    assert summary["status"] == "weave_verifier_schema_invalid"
    assert "required_evidence.trace_final_answer_order_required must be true" in summary["detail"]


def test_ok_verifier_without_timestamp_quality_requirement_does_not_pass_gate(tmp_path):
    module = load_module()
    task_id = "weave_agents_content_canary_TIMESTAMP_REQUIREMENT_MISSING"
    plan_file = write_plan(tmp_path, will_call_paid_model_api=True, task_id=task_id)
    write_command_result(plan_file, task_id, {"ok": True, "returncode": 0})
    payload = passing_verifier_payload(task_id)
    payload["required_evidence"].pop("trace_timestamp_quality_required")
    write_verifier(tmp_path, task_id, payload)

    summary = module.build_gate_summary(plan_file=plan_file)

    assert summary["ok"] is False
    assert summary["status"] == "weave_verifier_schema_invalid"
    assert "required_evidence.trace_timestamp_quality_required must be true" in summary["detail"]


def test_ok_verifier_without_input_message_requirement_does_not_pass_gate(tmp_path):
    module = load_module()
    task_id = "weave_agents_content_canary_INPUT_REQUIREMENT_MISSING"
    plan_file = write_plan(tmp_path, will_call_paid_model_api=True, task_id=task_id)
    write_command_result(plan_file, task_id, {"ok": True, "returncode": 0})
    payload = passing_verifier_payload(task_id)
    payload["required_evidence"].pop("input_message_required")
    write_verifier(tmp_path, task_id, payload)

    summary = module.build_gate_summary(plan_file=plan_file)

    assert summary["ok"] is False
    assert summary["status"] == "weave_verifier_schema_invalid"
    assert "required_evidence.input_message_required must be true" in summary["detail"]


def test_ok_verifier_without_request_model_evidence_does_not_pass_gate(tmp_path):
    module = load_module()
    task_id = "weave_agents_content_canary_REQUEST_MODEL_MISSING"
    plan_file = write_plan(tmp_path, will_call_paid_model_api=True, task_id=task_id)
    write_command_result(plan_file, task_id, {"ok": True, "returncode": 0})
    payload = passing_verifier_payload(task_id)
    payload["required_evidence"].pop("expected_request_models")
    payload["checks"] = [
        check for check in payload["checks"] if check.get("name") != "request_model"
    ]
    payload["content_capture_health"].pop("request_model_count")
    for span in payload["latest_trace_spans_chronological"]:
        span.pop("request_model", None)
    write_verifier(tmp_path, task_id, payload)

    summary = module.build_gate_summary(plan_file=plan_file)

    assert summary["ok"] is False
    assert summary["status"] == "weave_verifier_schema_invalid"
    assert summary["request_model_proven"] is False
    assert "required_evidence.expected_request_models" in summary["detail"]
    assert "checks must include exactly one request_model check" in summary["detail"]


def test_ok_verifier_without_required_text_capture_does_not_pass_gate(tmp_path):
    module = load_module()
    task_id = "weave_agents_content_canary_REQUIRED_TEXT_MISSING"
    plan_file = write_plan(tmp_path, will_call_paid_model_api=True, task_id=task_id)
    write_command_result(plan_file, task_id, {"ok": True, "returncode": 0})
    payload = passing_verifier_payload(task_id)
    payload["required_evidence"]["required_texts"] = [
        "TEST",
        "CANARY_RESULT TEST 91",
    ]
    payload["content_capture_health"]["required_text_count"] = 2
    write_verifier(tmp_path, task_id, payload)

    summary = module.build_gate_summary(plan_file=plan_file)

    assert summary["ok"] is False
    assert summary["status"] == "weave_verifier_schema_invalid"
    assert "checks missing required check(s): required_text_capture" in summary["detail"]


def test_ok_verifier_with_low_required_text_count_does_not_pass_gate(tmp_path):
    module = load_module()
    task_id = "weave_agents_content_canary_REQUIRED_TEXT_COUNT_LOW"
    plan_file = write_plan(tmp_path, will_call_paid_model_api=True, task_id=task_id)
    write_command_result(plan_file, task_id, {"ok": True, "returncode": 0})
    payload = passing_verifier_payload(task_id)
    payload["required_evidence"]["required_texts"] = [
        "TEST",
        "CANARY_RESULT TEST 91",
    ]
    payload["checks"].append({"name": "required_text_capture", "ok": True})
    payload["content_capture_health"]["required_text_count"] = 1
    write_verifier(tmp_path, task_id, payload)

    summary = module.build_gate_summary(plan_file=plan_file)

    assert summary["ok"] is False
    assert summary["status"] == "weave_verifier_schema_invalid"
    assert "content_capture_health.required_text_count must be at least 2" in summary["detail"]


def test_ok_verifier_without_final_answer_check_does_not_pass_gate(tmp_path):
    module = load_module()
    task_id = "weave_agents_content_canary_FINAL_CHECK_MISSING"
    plan_file = write_plan(tmp_path, will_call_paid_model_api=True, task_id=task_id)
    write_command_result(plan_file, task_id, {"ok": True, "returncode": 0})
    payload = passing_verifier_payload(task_id)
    payload["checks"] = [
        check
        for check in payload["checks"]
        if check.get("name") != "trace_final_answer_order"
    ]
    write_verifier(tmp_path, task_id, payload)

    summary = module.build_gate_summary(plan_file=plan_file)

    assert summary["ok"] is False
    assert summary["status"] == "weave_verifier_schema_invalid"
    assert "checks missing required check(s): trace_final_answer_order" in summary["detail"]


def test_message_content_failure_is_classified(tmp_path):
    module = load_module()
    task_id = "weave_agents_content_canary_EMPTY_CONTENT"
    plan_file = write_plan(tmp_path, will_call_paid_model_api=True, task_id=task_id)
    write_command_result(plan_file, task_id, {"ok": True, "returncode": 0})
    write_verifier(
        tmp_path,
        task_id,
        {
            "ok": False,
            "content_capture_health": {
                "message_span_count": 1,
                "message_spans_with_content": 0,
                "tool_span_count": 1,
                "tool_spans_with_content": 1,
            },
            "checks": [
                {"name": "message_content_capture", "ok": False},
                {"name": "tool_content_capture", "ok": True},
            ],
        },
    )

    summary = module.build_gate_summary(plan_file=plan_file)

    assert summary["ok"] is False
    assert summary["status"] == "content_missing"
    assert summary["failed_checks"][0]["name"] == "message_content_capture"


def test_required_canary_text_failure_is_classified(tmp_path):
    module = load_module()
    task_id = "weave_agents_content_canary_TEXT_MISSING"
    plan_file = write_plan(tmp_path, will_call_paid_model_api=True, task_id=task_id)
    write_command_result(plan_file, task_id, {"ok": True, "returncode": 0})
    write_verifier(
        tmp_path,
        task_id,
        {
            "ok": False,
            "content_capture_health": {
                "message_span_count": 1,
                "message_spans_with_content": 1,
                "tool_span_count": 1,
                "tool_spans_with_content": 1,
                "required_text_count": 2,
            },
            "checks": [
                {"name": "message_content_capture", "ok": True},
                {"name": "tool_content_capture", "ok": True},
                {
                    "name": "required_text_capture",
                    "ok": False,
                    "missing_required_texts": ["CANARY_RESULT TEST 91"],
                },
            ],
        },
    )

    summary = module.build_gate_summary(plan_file=plan_file)

    assert summary["ok"] is False
    assert summary["status"] == "canary_text_missing"
    assert summary["failed_checks"][-1]["name"] == "required_text_capture"


def test_final_answer_order_failure_is_classified(tmp_path):
    module = load_module()
    task_id = "weave_agents_content_canary_FINAL_AFTER_TOOL"
    plan_file = write_plan(tmp_path, will_call_paid_model_api=True, task_id=task_id)
    write_command_result(plan_file, task_id, {"ok": True, "returncode": 0})
    write_verifier(
        tmp_path,
        task_id,
        {
            "ok": False,
            "content_capture_health": {
                "message_span_count": 2,
                "message_spans_with_content": 2,
                "tool_span_count": 1,
                "tool_spans_with_content": 1,
                "final_answer_span_count": 1,
            },
            "checks": [
                {"name": "message_content_capture", "ok": True},
                {"name": "tool_content_capture", "ok": True},
                {
                    "name": "trace_final_answer_order",
                    "ok": False,
                    "detail": "a tool span starts after a final-answer message span",
                },
            ],
        },
    )

    summary = module.build_gate_summary(plan_file=plan_file)

    assert summary["ok"] is False
    assert summary["status"] == "trace_order_invalid"
    assert "final-answer" in summary["detail"]
