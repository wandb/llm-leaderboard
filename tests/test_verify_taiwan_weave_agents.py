import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = REPO_ROOT / "scripts" / "tools" / "verify_taiwan_weave_agents.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def agents_payload():
    return {
        "agents": [
            {
                "agent_name": "nejumi-taiwan-openclaw",
                "invocation_count": 3,
                "span_count": 9,
                "total_input_tokens": 123,
                "total_output_tokens": 45,
            }
        ],
        "total_count": 1,
    }


def spans_payload(
    *,
    content=True,
    tool_before_message=False,
    conversation_id="agentic_math:task-1",
    output_content="working draft",
    final_after_tool=False,
    final_output_content="ANSWER: 1",
    request_model="gpt-4.1-mini-2025-04-14",
):
    first_message_started = "2026-06-27T00:00:01.000000"
    tool_started = "2026-06-27T00:00:02.000000"
    if tool_before_message:
        first_message_started = "2026-06-27T00:00:03.000000"
        tool_started = "2026-06-27T00:00:01.000000"
    input_messages = [{"role": "user", "content": "problem"}] if content else []
    output_messages = [{"role": "assistant", "content": output_content}] if content else []
    tool_args = {"cmd": "python check.py"} if content else None
    tool_result = "ok" if content else None
    spans = [
        {
                "agent_name": "nejumi-taiwan-openclaw",
                "trace_id": "trace-1",
                "span_id": "chat-1",
                "parent_span_id": "root",
                "conversation_id": conversation_id,
                "operation_name": "chat",
                "span_name": "chat",
                "request_model": request_model,
                "started_at": first_message_started,
                "ended_at": "2026-06-27T00:00:04.000000",
                "input_messages": input_messages,
                "output_messages": output_messages,
                "input_tokens": 10,
                "output_tokens": 3,
                "error_type": "",
        },
        {
                "agent_name": "nejumi-taiwan-openclaw",
                "trace_id": "trace-1",
                "span_id": "tool-1",
                "parent_span_id": "root",
                "conversation_id": conversation_id,
                "operation_name": "execute_tool",
                "span_name": "execute_tool",
                "request_model": request_model,
                "tool_name": "exec",
                "started_at": tool_started,
                "ended_at": "2026-06-27T00:00:02.500000",
                "tool_call_arguments": tool_args,
                "tool_call_result": tool_result,
                "input_tokens": 0,
                "output_tokens": 0,
                "error_type": "",
        },
    ]
    if final_after_tool:
        spans.append(
            {
                "agent_name": "nejumi-taiwan-openclaw",
                "trace_id": "trace-1",
                "span_id": "chat-2",
                "parent_span_id": "root",
                "conversation_id": conversation_id,
                "operation_name": "chat",
                "span_name": "chat",
                "request_model": request_model,
                "started_at": "2026-06-27T00:00:03.000000",
                "ended_at": "2026-06-27T00:00:04.000000",
                "input_messages": [],
                "output_messages": [{"role": "assistant", "content": final_output_content}] if content else [],
                "input_tokens": 2,
                "output_tokens": 5,
                "error_type": "",
            }
        )
    return {"spans": spans}


def trace_chat_payload(
    *,
    canary_id="TEST_CANARY",
    conversation_id="agentic_math:task-1",
    request_model="gpt-4.1-mini-2025-04-14",
):
    return {
        "trace_id": "trace-1",
        "conversation_id": conversation_id,
        "messages": [
            {
                "type": "user_message",
                "started_at": "2026-06-27T00:00:01.000000",
                "user_message": {
                    "text": f"problem\nCanary ID: {canary_id}",
                },
            },
            {
                "type": "agent_start",
                "started_at": "2026-06-27T00:00:01.100000",
                "agent_start": {
                    "model": request_model,
                    "system_instructions": "solve the problem",
                },
            },
            {
                "type": "tool_call",
                "started_at": "2026-06-27T00:00:02.000000",
                "tool_call": {
                    "tool_name": "exec",
                    "tool_arguments": "{\"code\":\"return 7 * 13\"}",
                    "tool_result": "{\"value\":91}",
                },
            },
            {
                "type": "assistant_message",
                "started_at": "2026-06-27T00:00:03.000000",
                "assistant_message": {
                    "text": f"CANARY_RESULT {canary_id} 91",
                    "model": request_model,
                },
            },
        ],
    }


def test_verify_weave_agents_accepts_contentful_ordered_trace():
    module = load_module()

    result = module.verify_agents_payload(
        agents_payload(),
        spans_payload(),
        entity="llm-leaderboard",
        project="tc-leaderboard",
        agent_name="nejumi-taiwan-openclaw",
        require_content=True,
        require_tool_span=True,
        require_tool_content=True,
        require_usage=True,
    )

    assert result["ok"] is True
    assert result["schema_version"] == module.VERIFICATION_SCHEMA_VERSION
    assert result["verification_schema_version"] == module.VERIFICATION_SCHEMA_VERSION
    assert result["status"] == "passed"
    assert result["query_source"] == {
        "kind": "wandb_agents_api",
        "api_base_url": "https://trace.wandb.ai",
        "agents_endpoint": "/agents/query",
        "spans_endpoint": "/agents/spans/query",
        "project_id": "llm-leaderboard/tc-leaderboard",
        "agent_name": "nejumi-taiwan-openclaw",
        "conversation_id": "",
        "conversation_id_contains": "",
        "agents_count": 1,
        "spans_count": 2,
        "matching_span_count": 2,
        "latest_trace_span_count": 2,
    }
    assert result["content_capture_health"] == {
        "span_count_checked": 2,
        "message_span_count": 1,
        "message_spans_with_content": 1,
        "message_spans_with_input": 1,
        "tool_span_count": 1,
        "tool_spans_with_content": 1,
        "final_answer_span_count": 0,
        "spans_with_valid_timestamps": 2,
        "spans_with_invalid_timestamps": 0,
        "trace_input_tokens": 10,
        "trace_output_tokens": 3,
        "required_text_count": 0,
        "request_model_count": 1,
        "trace_final_answer_after_tool_warning": False,
    }


def test_verify_weave_agents_accepts_content_from_trace_chat_when_span_rows_are_scalar_only():
    module = load_module()

    result = module.verify_agents_payload(
        agents_payload(),
        spans_payload(content=False),
        trace_chat_payload=trace_chat_payload(canary_id="TEST_CANARY"),
        entity="llm-leaderboard",
        project="tc-leaderboard",
        agent_name="nejumi-taiwan-openclaw",
        require_content=True,
        require_tool_span=True,
        require_tool_content=True,
        require_usage=True,
        required_texts=["problem", "CANARY_RESULT TEST_CANARY 91"],
        expected_request_models=[
            "openai-direct/gpt-4.1-mini-2025-04-14",
            "gpt-4.1-mini-2025-04-14",
        ],
    )

    assert result["ok"] is True
    assert result["query_source"]["trace_chat_endpoint"] == module.AGENTS_TRACES_CHAT_ENDPOINT
    assert result["content_capture_health"]["message_spans_with_content"] == 2
    assert result["content_capture_health"]["message_spans_with_input"] == 1
    assert result["content_capture_health"]["tool_spans_with_content"] == 1
    assert result["content_capture_health"]["final_answer_span_count"] == 1
    assert result["content_capture_health"]["chat_tool_calls_with_content"] == 1
    assert any(
        check["name"] == "required_text_capture" and check["ok"]
        for check in result["checks"]
    )
    assert any(
        check["name"] == "trace_final_answer_order" and check["ok"]
        for check in result["checks"]
    )


def test_verify_weave_agents_rejects_empty_message_content_when_required():
    module = load_module()

    result = module.verify_agents_payload(
        agents_payload(),
        spans_payload(content=False),
        entity="llm-leaderboard",
        project="tc-leaderboard",
        agent_name="nejumi-taiwan-openclaw",
        require_content=True,
    )

    assert result["ok"] is False
    assert result["schema_version"] == module.VERIFICATION_SCHEMA_VERSION
    assert result["verification_schema_version"] == module.VERIFICATION_SCHEMA_VERSION
    assert result["status"] == "failed"
    assert any(
        check["name"] == "message_content_capture" and not check["ok"]
        for check in result["checks"]
    )


def test_verify_weave_agents_allows_structure_only_when_content_not_required():
    module = load_module()

    result = module.verify_agents_payload(
        agents_payload(),
        spans_payload(content=False),
        entity="llm-leaderboard",
        project="tc-leaderboard",
        agent_name="nejumi-taiwan-openclaw",
        require_content=False,
    )

    assert result["ok"] is True


def test_verify_weave_agents_rejects_tool_before_first_message():
    module = load_module()

    result = module.verify_agents_payload(
        agents_payload(),
        spans_payload(tool_before_message=True),
        entity="llm-leaderboard",
        project="tc-leaderboard",
        agent_name="nejumi-taiwan-openclaw",
        require_content=True,
    )

    assert result["ok"] is False
    assert any(check["name"] == "trace_order" and not check["ok"] for check in result["checks"])


def test_verify_weave_agents_rejects_tool_without_visible_user_input():
    module = load_module()
    payload = spans_payload()
    payload["spans"][0]["input_messages"] = []
    payload["spans"][0]["output_messages"] = [
        {"role": "assistant", "content": "working draft"}
    ]

    result = module.verify_agents_payload(
        agents_payload(),
        payload,
        entity="llm-leaderboard",
        project="tc-leaderboard",
        agent_name="nejumi-taiwan-openclaw",
        require_content=True,
        require_tool_span=True,
    )

    assert result["ok"] is False
    assert any(
        check["name"] == "input_message_capture" and not check["ok"]
        for check in result["checks"]
    )
    assert any(
        check["name"] == "trace_user_message_order" and not check["ok"]
        for check in result["checks"]
    )


def test_verify_weave_agents_warns_tool_after_final_answer_message():
    module = load_module()
    payload = spans_payload(output_content="ANSWER: 1")
    payload["spans"][0]["ended_at"] = "2026-06-27T00:00:01.500000"

    result = module.verify_agents_payload(
        agents_payload(),
        payload,
        entity="llm-leaderboard",
        project="tc-leaderboard",
        agent_name="nejumi-taiwan-openclaw",
        require_content=True,
        require_tool_span=True,
    )

    assert result["ok"] is True
    assert any(
        check["name"] == "trace_final_answer_order"
        and check["ok"]
        and check.get("tool_after_final_answer_warning") is True
        for check in result["checks"]
    )


def test_verify_weave_agents_does_not_treat_boxed_scratch_as_final_answer():
    module = load_module()

    result = module.verify_agents_payload(
        agents_payload(),
        spans_payload(output_content="途中式として \\boxed{x+1} を検討する。まだ検算する。"),
        entity="llm-leaderboard",
        project="tc-leaderboard",
        agent_name="nejumi-taiwan-openclaw",
        require_content=True,
        require_tool_span=True,
    )

    assert result["ok"] is True
    health = result["content_capture_health"]
    assert health["final_answer_span_count"] == 0
    assert any(
        check["name"] == "trace_final_answer_order" and check["ok"]
        for check in result["checks"]
    )


def test_verify_weave_agents_does_not_treat_tool_use_message_as_final_answer():
    module = load_module()
    payload = spans_payload(output_content="working")
    payload["spans"][0]["output_messages"] = [
        {
            "role": "assistant",
            "content": "ANSWER: \\boxed{11} と見えるが、まだexecで検算する。",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "exec", "arguments": "{\"cmd\":\"python3 check.py\"}"},
                }
            ],
        }
    ]

    result = module.verify_agents_payload(
        agents_payload(),
        payload,
        entity="llm-leaderboard",
        project="tc-leaderboard",
        agent_name="nejumi-taiwan-openclaw",
        require_content=True,
        require_tool_span=True,
    )

    assert result["ok"] is True
    assert result["content_capture_health"]["final_answer_span_count"] == 0
    assert any(
        check["name"] == "trace_final_answer_order" and check["ok"]
        for check in result["checks"]
    )


def test_verify_weave_agents_does_not_treat_chat_tool_use_message_as_final_answer():
    module = load_module()
    chat_payload = trace_chat_payload()
    chat_payload["messages"] = [
        {
            "type": "user_message",
            "started_at": "2026-06-27T00:00:01.000000",
            "user_message": {"text": "problem"},
        },
        {
            "type": "assistant_message",
            "started_at": "2026-06-27T00:00:01.500000",
            "assistant_message": {
                "text": "ANSWER: \\boxed{11} と見えるが、toolUse中なので最終回答ではない。",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "exec", "arguments": "{}"},
                    }
                ],
            },
        },
        {
            "type": "tool_call",
            "started_at": "2026-06-27T00:00:02.000000",
            "tool_call": {
                "tool_name": "exec",
                "tool_arguments": "{}",
                "tool_result": "ok",
            },
        },
    ]

    result = module.verify_agents_payload(
        agents_payload(),
        spans_payload(output_content="working"),
        trace_chat_payload=chat_payload,
        entity="llm-leaderboard",
        project="tc-leaderboard",
        agent_name="nejumi-taiwan-openclaw",
        require_content=True,
        require_tool_span=True,
    )

    assert result["ok"] is True
    assert result["content_capture_health"]["final_answer_span_count"] == 0
    assert any(
        message["has_assistant_message"] and not message["is_final_answer"]
        for message in result["latest_trace_chat_messages_chronological"]
    )


def test_verify_weave_agents_allows_tool_inside_final_answer_parent_span():
    module = load_module()

    result = module.verify_agents_payload(
        agents_payload(),
        spans_payload(output_content="ANSWER: 1"),
        entity="llm-leaderboard",
        project="tc-leaderboard",
        agent_name="nejumi-taiwan-openclaw",
        require_content=True,
        require_tool_span=True,
    )

    assert result["ok"] is True
    assert any(
        check["name"] == "trace_final_answer_order" and check["ok"]
        for check in result["checks"]
    )


def test_verify_weave_agents_rejects_invalid_span_timestamps():
    module = load_module()
    payload = spans_payload()
    payload["spans"][1]["started_at"] = ""

    result = module.verify_agents_payload(
        agents_payload(),
        payload,
        entity="llm-leaderboard",
        project="tc-leaderboard",
        agent_name="nejumi-taiwan-openclaw",
        require_content=True,
        require_tool_span=True,
    )

    assert result["ok"] is False
    assert result["content_capture_health"]["spans_with_invalid_timestamps"] == 1
    assert any(
        check["name"] == "trace_timestamp_quality" and not check["ok"]
        for check in result["checks"]
    )
    assert any(
        check["name"] == "trace_user_message_order" and not check["ok"]
        for check in result["checks"]
    )


def test_verify_weave_agents_rejects_same_time_tool_and_user_input():
    module = load_module()
    payload = spans_payload()
    payload["spans"][1]["started_at"] = payload["spans"][0]["started_at"]

    result = module.verify_agents_payload(
        agents_payload(),
        payload,
        entity="llm-leaderboard",
        project="tc-leaderboard",
        agent_name="nejumi-taiwan-openclaw",
        require_content=True,
        require_tool_span=True,
    )

    assert result["ok"] is False
    assert any(
        check["name"] == "trace_user_message_order" and not check["ok"]
        for check in result["checks"]
    )


def test_verify_weave_agents_filters_conversation_id_contains():
    module = load_module()

    result = module.verify_agents_payload(
        agents_payload(),
        spans_payload(conversation_id="agentic_math:task-1"),
        entity="llm-leaderboard",
        project="tc-leaderboard",
        agent_name="nejumi-taiwan-openclaw",
        conversation_id_contains="swebench",
    )

    assert result["ok"] is False
    assert any(check["name"] == "spans_present" and not check["ok"] for check in result["checks"])


def test_verify_weave_agents_accepts_dynamic_agent_blank_name_when_conversation_matches():
    module = load_module()
    payload = spans_payload(conversation_id="agent:dynamic-agent:run:agentic-math:task-1")
    for span in payload["spans"]:
        span["agent_name"] = ""

    result = module.verify_agents_payload(
        {"agents": [], "total_count": 0},
        payload,
        entity="llm-leaderboard",
        project="tc-leaderboard",
        agent_name="nejumi-taiwan-openclaw",
        conversation_id_contains="agent:dynamic-agent:run:agentic-math:task-1",
        require_content=True,
        require_tool_span=True,
        require_tool_content=True,
        require_usage=True,
    )

    assert result["ok"] is True
    assert result["query_source"]["matching_span_count"] == 2
    assert any(
        check["name"] == "agent_present"
        and check["ok"]
        and check.get("agent_name_missing_on_spans") is True
        for check in result["checks"]
    )


def test_verify_weave_agents_rejects_dynamic_agent_tool_only_conversation():
    module = load_module()
    conversation_id = "agent:dynamic-agent:run:agentic-math:task-1"
    payload = {
        "spans": [
            {
                "agent_name": "",
                "trace_id": "trace-tool-only",
                "span_id": "tool-1",
                "parent_span_id": "root",
                "conversation_id": conversation_id,
                "operation_name": "execute_tool",
                "span_name": "execute_tool exec",
                "request_model": "",
                "tool_name": "exec",
                "started_at": "2026-06-27T00:00:02.000000",
                "ended_at": "2026-06-27T00:00:02.500000",
                "tool_call_arguments": {"command": "python3 check.py"},
                "tool_call_result": "ok",
                "input_tokens": 0,
                "output_tokens": 0,
                "error_type": "",
            }
        ]
    }

    result = module.verify_agents_payload(
        {"agents": [], "total_count": 0},
        payload,
        entity="llm-leaderboard",
        project="tc-leaderboard",
        agent_name="nejumi-taiwan-openclaw",
        conversation_id_contains=conversation_id,
        require_content=True,
        require_tool_span=True,
        require_tool_content=True,
    )

    assert result["ok"] is False
    assert any(check["name"] == "spans_present" and check["ok"] for check in result["checks"])
    assert any(
        check["name"] == "input_message_capture" and not check["ok"]
        for check in result["checks"]
    )
    assert any(
        check["name"] == "trace_user_message_order" and not check["ok"]
        for check in result["checks"]
    )


def test_verify_weave_agents_chooses_latest_trace_by_span_time_not_api_order():
    module = load_module()
    payload = spans_payload()
    old_spans = payload["spans"]
    for span in old_spans:
        span["trace_id"] = "trace-old"
        span["started_at"] = span["started_at"].replace("2026-06-27", "2026-06-26")
        span["ended_at"] = span["ended_at"].replace("2026-06-27", "2026-06-26")
    new_payload = spans_payload(conversation_id="agentic_math:task-2")
    for span in new_payload["spans"]:
        span["trace_id"] = "trace-new"
    payload["spans"] = [*old_spans, *new_payload["spans"]]

    result = module.verify_agents_payload(
        agents_payload(),
        payload,
        entity="llm-leaderboard",
        project="tc-leaderboard",
        agent_name="nejumi-taiwan-openclaw",
        require_content=True,
        require_tool_span=True,
        require_tool_content=True,
    )

    assert result["ok"] is True
    assert result["latest_trace_id"] == "trace-new"
    assert {
        span["trace_id"] for span in result["latest_trace_spans_chronological"]
    } == {"trace-new"}


def test_verify_weave_agents_usage_accepts_span_tokens_when_agent_totals_are_zero():
    module = load_module()
    payload = agents_payload()
    payload["agents"][0]["total_input_tokens"] = 0
    payload["agents"][0]["total_output_tokens"] = 0

    result = module.verify_agents_payload(
        payload,
        spans_payload(),
        entity="llm-leaderboard",
        project="tc-leaderboard",
        agent_name="nejumi-taiwan-openclaw",
        require_usage=True,
    )

    assert result["ok"] is True
    assert any(
        check["name"] == "usage"
        and check["trace_input_tokens"] == 10
        and check["trace_output_tokens"] == 3
        for check in result["checks"]
    )


def test_verify_weave_agents_accepts_required_texts_in_visible_content():
    module = load_module()

    result = module.verify_agents_payload(
        agents_payload(),
        spans_payload(
            final_after_tool=True,
            final_output_content="CANARY_RESULT TEST_CANARY 91",
        ),
        entity="llm-leaderboard",
        project="tc-leaderboard",
        agent_name="nejumi-taiwan-openclaw",
        require_content=True,
        required_texts=["problem", "CANARY_RESULT TEST_CANARY 91"],
    )

    assert result["ok"] is True
    assert result["content_capture_health"]["required_text_count"] == 2
    assert any(
        check["name"] == "required_text_capture" and check["ok"]
        for check in result["checks"]
    )


def test_verify_weave_agents_rejects_missing_required_texts():
    module = load_module()

    result = module.verify_agents_payload(
        agents_payload(),
        spans_payload(output_content="ANSWER: 1"),
        entity="llm-leaderboard",
        project="tc-leaderboard",
        agent_name="nejumi-taiwan-openclaw",
        require_content=True,
        required_texts=["CANARY_RESULT TEST_CANARY 91"],
    )

    assert result["ok"] is False
    assert any(
        check["name"] == "required_text_capture"
        and not check["ok"]
        and check["missing_required_texts"] == ["CANARY_RESULT TEST_CANARY 91"]
        for check in result["checks"]
    )


def test_verify_weave_agents_accepts_expected_request_model_alias():
    module = load_module()

    result = module.verify_agents_payload(
        agents_payload(),
        spans_payload(request_model="gpt-4.1-mini-2025-04-14"),
        entity="llm-leaderboard",
        project="tc-leaderboard",
        agent_name="nejumi-taiwan-openclaw",
        require_content=True,
        expected_request_models=[
            "openai-direct/gpt-4.1-mini-2025-04-14",
            "gpt-4.1-mini-2025-04-14",
        ],
    )

    assert result["ok"] is True
    assert result["content_capture_health"]["request_model_count"] == 1
    assert any(
        check["name"] == "request_model"
        and check["ok"]
        and "gpt-4.1-mini-2025-04-14" in check["observed_request_models"]
        for check in result["checks"]
    )


def test_verify_weave_agents_rejects_unexpected_request_model():
    module = load_module()

    result = module.verify_agents_payload(
        agents_payload(),
        spans_payload(request_model="wrong-model"),
        entity="llm-leaderboard",
        project="tc-leaderboard",
        agent_name="nejumi-taiwan-openclaw",
        require_content=True,
        expected_request_models=["gpt-4.1-mini-2025-04-14"],
    )

    assert result["ok"] is False
    assert any(
        check["name"] == "request_model"
        and not check["ok"]
        and check["observed_request_models"] == ["wrong-model"]
        for check in result["checks"]
    )
