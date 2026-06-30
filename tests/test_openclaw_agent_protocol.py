import importlib.util
import json
import subprocess
import time
from argparse import Namespace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_last_json_line_skips_weave_banner():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py")
    stdout = 'Initializing project: llm-leaderboard/tc-leaderboard\n{"ok":true,"trace":"abc"}\n'
    assert module.parse_last_json_line(stdout) == {"ok": True, "trace": "abc"}


def test_concurrent_export_limit_is_transient_weave_sidecar_error():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py")

    assert module.is_transient_weave_sidecar_error(
        "Initializing project: llm-leaderboard/tc-leaderboard\n",
        "Error: Concurrent export limit reached",
    )
    assert not module.is_transient_weave_sidecar_error("", "TypeError: Cannot read properties of undefined")


def test_extract_openclaw_text_prefers_meta_visible_text():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py")
    sidecar = {
        "stdout_json": {
            "payloads": [{"text": "payload answer"}],
            "meta": {"finalAssistantVisibleText": "visible answer"},
        }
    }
    assert module.extract_assistant_text(sidecar) == "visible answer"


def test_extract_gateway_openclaw_result_shape(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py")
    session = tmp_path / "session.jsonl"
    session.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "message": {
                            "role": "assistant",
                            "timestamp": 123,
                            "content": [
                                {
                                    "type": "toolCall",
                                    "id": "call_1",
                                    "name": "exec",
                                    "arguments": {"command": "python3 -c 'print(4)'"},
                                }
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "message": {
                            "role": "toolResult",
                            "timestamp": 124,
                            "toolCallId": "call_1",
                            "toolName": "exec",
                            "content": [{"type": "text", "text": "4"}],
                            "isError": False,
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    sidecar = {
        "stdout_json": {
            "status": "ok",
            "result": {
                "payloads": [{"text": "ANSWER: \\boxed{4}"}],
                "meta": {"agentMeta": {"sessionFile": str(session)}},
                "finalAssistantVisibleText": "ANSWER: \\boxed{4}",
            },
        }
    }

    assert module.extract_assistant_text(sidecar) == "ANSWER: \\boxed{4}"
    assert module.extract_agent_meta(sidecar)["sessionFile"] == str(session)
    assert module.extract_tool_events(sidecar)[0]["toolName"] == "exec"


def test_normalize_usage_maps_openclaw_usage_to_weave_usage():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py")
    assert module.normalize_usage(
        {
            "usage": {
                "input": 10,
                "output": 3,
                "reasoningTokens": 2,
                "cacheRead": 7,
                "cacheWrite": 1,
            }
        }
    ) == {
        "inputTokens": 10,
        "outputTokens": 3,
        "reasoningTokens": 2,
        "cacheReadInputTokens": 7,
        "cacheCreationInputTokens": 1,
    }


def test_extract_reasoning_text_from_openclaw_session(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py")
    session = tmp_path / "session.jsonl"
    session.write_text(
        "\n".join(
            [
                json.dumps({"type": "session"}),
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "thinking",
                                    "thinking": "first reasoning chunk",
                                    "thinkingSignature": "reasoning_content",
                                },
                                {"type": "text", "text": "ANSWER: 1"},
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "thinking", "thinking": "second chunk"}],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    sidecar = {"stdout_json": {"meta": {"agentMeta": {"sessionFile": str(session)}}}}
    assert module.extract_reasoning_text(sidecar) == "first reasoning chunk\n\n---\n\nsecond chunk"


def test_extract_tool_events_from_openclaw_session(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py")
    session = tmp_path / "session.jsonl"
    session.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "message": {
                            "role": "assistant",
                            "timestamp": 123,
                            "content": [
                                {
                                    "type": "toolCall",
                                    "id": "call_1",
                                    "name": "code_execution",
                                    "arguments": {"task": "print(1 + 1)"},
                                }
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "message": {
                            "role": "toolResult",
                            "timestamp": 456,
                            "toolCallId": "call_1",
                            "toolName": "code_execution",
                            "content": [{"type": "text", "text": "2"}],
                            "details": {"tookMs": 10},
                            "isError": False,
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    sidecar = {"stdout_json": {"meta": {"agentMeta": {"sessionFile": str(session)}}}}

    events = module.extract_tool_events(sidecar)

    assert events == [
        {
            "type": "tool_call",
            "toolCallId": "call_1",
            "toolName": "code_execution",
            "arguments": {"task": "print(1 + 1)"},
            "timestamp": 123,
            "index": 0,
        },
        {
            "type": "tool_result",
            "toolCallId": "call_1",
            "toolName": "code_execution",
            "content": "2",
            "details": {"tookMs": 10},
            "isError": False,
            "timestamp": 456,
            "index": 1,
        },
    ]


def test_extract_tool_events_from_live_session_file_fallback(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py")
    session = tmp_path / "session.jsonl"
    session.write_text(
        json.dumps(
            {
                "message": {
                    "role": "assistant",
                    "timestamp": 123,
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "call_1",
                            "name": "exec",
                            "arguments": {"cmd": "python3 check.py"},
                        }
                    ],
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    events = module.extract_tool_events({"live_session_file": str(session)})

    assert len(events) == 1
    assert events[0]["toolName"] == "exec"


def test_extract_tool_events_prefers_copied_nemoclaw_session(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py")
    host_session = tmp_path / "copied.jsonl"
    host_session.write_text(
        json.dumps(
            {
                "message": {
                    "role": "assistant",
                    "timestamp": 123,
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "call_1",
                            "name": "exec",
                            "arguments": {"cmd": "python3 check.py"},
                        }
                    ],
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sidecar = {
        "stdout_json": {
            "meta": {"agentMeta": {"sessionFile": "/sandbox/.openclaw/agents/main/sessions/s1.jsonl"}}
        },
        "copied_session_file": str(host_session),
    }

    meta = module.extract_agent_meta(sidecar)
    events = module.extract_tool_events(sidecar)

    assert meta["sandboxSessionFile"] == "/sandbox/.openclaw/agents/main/sessions/s1.jsonl"
    assert meta["sessionFile"] == str(host_session)
    assert events[0]["toolName"] == "exec"


def test_copy_nemoclaw_session_file_writes_host_audit_copy(tmp_path, monkeypatch):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py")
    sidecar = {
        "stdout_json": {
            "meta": {"agentMeta": {"sessionFile": "/sandbox/.openclaw/agents/main/sessions/s1.jsonl"}}
        }
    }
    args = Namespace(nemoclaw_bin="nemoclaw", nemoclaw_sandbox="nejumi-taiwan")
    captured = {}

    def fake_run(command, text, capture_output, check, env):
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout='{"message":{"role":"assistant","content":[]}}\n', stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    status = module.copy_nemoclaw_session_file(sidecar, args, tmp_path, {"PATH": "/bin"})

    assert status["ok"] is True
    assert status["sandbox_session_file"] == "/sandbox/.openclaw/agents/main/sessions/s1.jsonl"
    assert Path(status["copied_session_file"]).read_text(encoding="utf-8").startswith('{"message"')
    assert captured["command"][:4] == ["nemoclaw", "sandbox", "exec", "nejumi-taiwan"]
    assert "/sandbox/.openclaw/agents/main/sessions/s1.jsonl" in captured["command"]


def test_live_tool_budget_status_detects_agent_session_overage(tmp_path, monkeypatch):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py")
    monkeypatch.setenv("OPENCLAW_STATE_DIR", str(tmp_path / "openclaw-state"))
    session_dir = tmp_path / "openclaw-state" / "agents" / "agent-a" / "sessions"
    session_dir.mkdir(parents=True)
    session = session_dir / "session-1.jsonl"
    session.write_text(
        "\n".join(
            json.dumps(
                {
                    "message": {
                        "role": "assistant",
                        "timestamp": index,
                        "content": [
                            {
                                "type": "toolCall",
                                "id": f"call_{index}",
                                "name": "exec",
                                "arguments": {"cmd": "true"},
                            }
                        ],
                    }
                }
            )
            for index in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    args = Namespace(agent="agent-a", profile=None, max_tool_calls=1)

    status = module.live_tool_budget_status(args, time.time() - 1)

    assert status["enabled"] is True
    assert status["exceeded"] is True
    assert status["tool_call_count"] == 2
    assert status["session_file"] == str(session)


def test_extract_timeline_events_preserves_openclaw_session_order(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py")
    session = tmp_path / "session.jsonl"
    session.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "message": {
                            "role": "user",
                            "timestamp": 100,
                            "content": [{"type": "text", "text": "problem"}],
                        }
                    }
                ),
                json.dumps(
                    {
                        "message": {
                            "role": "assistant",
                            "timestamp": 200,
                            "content": [
                                {"type": "thinking", "thinking": "answer seems 11"},
                                {
                                    "type": "toolCall",
                                    "id": "call_1",
                                    "name": "exec",
                                    "arguments": {"cmd": "python3 check.py"},
                                },
                            ],
                        }
                    }
                ),
                json.dumps(
                    {
                        "message": {
                            "role": "toolResult",
                            "timestamp": 300,
                            "toolCallId": "call_1",
                            "toolName": "exec",
                            "content": [{"type": "text", "text": "verified"}],
                            "isError": False,
                        }
                    }
                ),
                json.dumps(
                    {
                        "message": {
                            "role": "assistant",
                            "timestamp": 400,
                            "content": [{"type": "text", "text": "ANSWER: \\boxed{11}"}],
                        }
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    sidecar = {"stdout_json": {"meta": {"agentMeta": {"sessionFile": str(session)}}}}

    events = module.extract_timeline_events(sidecar)

    assert [event["type"] for event in events] == [
        "user_message",
        "assistant_reasoning",
        "tool_call",
        "tool_result",
        "assistant_message",
    ]
    assert [event["timelineIndex"] for event in events] == [0, 1, 2, 3, 4]
    assert events[4]["content"] == "ANSWER: \\boxed{11}"


def test_conversation_order_status_accepts_user_tool_answer_sequence():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py")
    events = [
        {"type": "user_message", "timelineIndex": 0, "content": "problem"},
        {"type": "tool_call", "timelineIndex": 1, "toolCallId": "call_1", "toolName": "exec"},
        {"type": "tool_result", "timelineIndex": 2, "toolCallId": "call_1", "toolName": "exec"},
        {"type": "assistant_message", "timelineIndex": 3, "content": "ANSWER: \\boxed{11}"},
    ]

    status = module.conversation_order_status(events)

    assert status["ok"] is True
    assert status["checked"] is True
    assert status["source"] == "openclaw_session_jsonl"
    assert status["issues"] == []
    assert status["first_user_index"] == 0
    assert status["first_tool_call_index"] == 1
    assert status["first_final_answer_index"] == 3


def test_conversation_order_status_rejects_tool_before_problem():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py")
    events = [
        {"type": "tool_call", "timelineIndex": 0, "toolCallId": "call_1", "toolName": "exec"},
        {"type": "user_message", "timelineIndex": 1, "content": "problem"},
        {"type": "tool_result", "timelineIndex": 2, "toolCallId": "call_1", "toolName": "exec"},
    ]

    status = module.conversation_order_status(events)

    assert status["ok"] is False
    assert any(issue["type"] == "tool_before_or_at_first_user_message" for issue in status["issues"])


def test_conversation_order_status_rejects_tool_after_final_answer():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py")
    events = [
        {"type": "user_message", "timelineIndex": 0, "content": "problem"},
        {"type": "assistant_message", "timelineIndex": 1, "content": "ANSWER: \\boxed{11}"},
        {"type": "tool_call", "timelineIndex": 2, "toolCallId": "call_1", "toolName": "exec"},
    ]

    status = module.conversation_order_status(events)

    assert status["ok"] is False
    assert any(issue["type"] == "tool_after_final_answer" for issue in status["issues"])


def test_build_agents_check_summary_reports_timestamp_and_order_health():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py")
    agents = {"agents": [{"agent_name": "nejumi-taiwan-openclaw"}], "total_count": 1}
    spans = {
        "spans": [
            {
                "started_at": "2026-06-29T00:00:00Z",
                "ended_at": "2026-06-29T00:00:01Z",
                "operation_name": "chat",
                "agent_name": "nejumi-taiwan-openclaw",
                "trace_id": "trace-ok",
                "span_id": "span-user",
                "conversation_id": "run-123",
                "input_messages": [{"role": "user", "content": "problem"}],
            },
            {
                "started_at": "2026-06-29T00:00:02Z",
                "ended_at": "2026-06-29T00:00:03Z",
                "operation_name": "execute_tool",
                "agent_name": "nejumi-taiwan-openclaw",
                "trace_id": "trace-ok",
                "span_id": "span-tool",
                "conversation_id": "run-123",
                "tool_call_arguments": {"cmd": "python3 check.py"},
                "tool_call_result": "verified",
            },
            {
                "started_at": "2026-06-29T00:00:04Z",
                "ended_at": "2026-06-29T00:00:05Z",
                "operation_name": "chat",
                "agent_name": "nejumi-taiwan-openclaw",
                "trace_id": "trace-ok",
                "span_id": "span-answer",
                "conversation_id": "run-123",
                "output_messages": [{"role": "assistant", "content": "ANSWER: \\boxed{11}"}],
            },
        ]
    }

    summary = module.build_agents_check_summary(
        agents,
        spans,
        entity="llm-leaderboard",
        project="tc-leaderboard",
        agent_name="nejumi-taiwan-openclaw",
        limit=10,
        span_limit=40,
    )

    assert summary["diagnostic_schema_version"] == 1
    assert isinstance(summary["generated_at"], float)
    assert summary["query_source"] == {
        "kind": "wandb_agents_api",
        "api_base_url": "https://trace.wandb.ai",
        "agents_endpoint": "/agents/query",
        "spans_endpoint": "/agents/spans/query",
        "project_id": "llm-leaderboard/tc-leaderboard",
        "agent_name": "nejumi-taiwan-openclaw",
        "conversation_id": "",
        "conversation_id_contains": "",
        "limit": 10,
        "span_limit": 40,
        "agents_count": 1,
        "spans_count": 3,
        "matching_span_count": 3,
        "latest_trace_span_count": 3,
    }
    health = summary["content_capture_health"]
    assert health["spans_with_valid_timestamps"] == 3
    assert health["spans_with_invalid_timestamps"] == 0
    assert health["final_answer_span_count"] == 1
    assert health["trace_order_ok"] is True
    assert health["trace_user_message_order_ok"] is True
    assert health["trace_final_answer_order_ok"] is True
    assert summary["trace_order_health"]["order_issues"] == []
    assert summary["latest_trace_spans_chronological"][2]["has_final_answer_marker"] is True


def test_check_agents_writes_json_diagnostic(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py")
    output = tmp_path / "agents_diagnostic.json"
    agents = {"agents": [{"agent_name": "nejumi-taiwan-openclaw"}], "total_count": 1}
    spans = {
        "spans": [
            {
                "started_at": "2026-06-29T00:00:00Z",
                "ended_at": "2026-06-29T00:00:01Z",
                "operation_name": "chat",
                "agent_name": "nejumi-taiwan-openclaw",
                "trace_id": "trace-json",
                "span_id": "span-user",
                "input_messages": [{"role": "user", "content": "problem"}],
            }
        ]
    }

    def fake_agents_api_post(_env, path, _payload):
        if path == "/agents/query":
            return agents
        if path == "/agents/spans/query":
            return spans
        raise AssertionError(path)

    original = module.agents_api_post
    module.agents_api_post = fake_agents_api_post
    try:
        module.check_agents(
            Namespace(
                entity="llm-leaderboard",
                project="tc-leaderboard",
                agent_name="nejumi-taiwan-openclaw",
                env_file=None,
                limit=5,
                conversation_id=None,
                conversation_id_contains=None,
                json=output,
            )
        )
    finally:
        module.agents_api_post = original

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["diagnostic_schema_version"] == 1
    assert isinstance(payload["generated_at"], float)
    assert payload["query_source"]["kind"] == "wandb_agents_api"
    assert payload["query_source"]["api_base_url"] == "https://trace.wandb.ai"
    assert payload["query_source"]["agents_endpoint"] == "/agents/query"
    assert payload["query_source"]["spans_endpoint"] == "/agents/spans/query"
    assert payload["query_source"]["project_id"] == "llm-leaderboard/tc-leaderboard"
    assert payload["query_source"]["agent_name"] == "nejumi-taiwan-openclaw"
    assert payload["query_source"]["conversation_id"] == ""
    assert payload["query_source"]["conversation_id_contains"] == ""
    assert payload["query_source"]["limit"] == 5
    assert payload["query_source"]["span_limit"] == 20
    assert payload["query_source"]["agents_count"] == 1
    assert payload["query_source"]["spans_count"] == 1
    assert payload["query_source"]["matching_span_count"] == 1
    assert payload["query_source"]["latest_trace_span_count"] == 1
    assert payload["latest_trace_id"] == "trace-json"
    assert payload["content_capture_health"]["spans_with_invalid_timestamps"] == 0
    assert payload["trace_order_health"]["timestamp_quality_ok"] is True


def test_build_agents_check_summary_filters_by_conversation_scope():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py")
    agents = {"agents": [{"agent_name": "nejumi-taiwan-openclaw"}], "total_count": 1}
    spans = {
        "spans": [
            {
                "started_at": "2026-06-29T00:00:00Z",
                "ended_at": "2026-06-29T00:00:01Z",
                "operation_name": "chat",
                "agent_name": "nejumi-taiwan-openclaw",
                "trace_id": "trace-old",
                "span_id": "span-old",
                "conversation_id": "other-run",
                "input_messages": [{"role": "user", "content": "old"}],
            },
            {
                "started_at": "2026-06-29T00:00:02Z",
                "ended_at": "2026-06-29T00:00:03Z",
                "operation_name": "chat",
                "agent_name": "nejumi-taiwan-openclaw",
                "trace_id": "trace-canary",
                "span_id": "span-canary",
                "conversation_id": "weave_agents_content_canary_TEST",
                "input_messages": [{"role": "user", "content": "problem"}],
            },
        ]
    }

    summary = module.build_agents_check_summary(
        agents,
        spans,
        entity="llm-leaderboard",
        project="tc-leaderboard",
        agent_name="nejumi-taiwan-openclaw",
        limit=10,
        span_limit=40,
        conversation_id_contains="content_canary_TEST",
    )

    assert summary["latest_trace_id"] == "trace-canary"
    assert summary["query_source"]["conversation_id_contains"] == "content_canary_TEST"
    assert summary["query_source"]["spans_count"] == 2
    assert summary["query_source"]["matching_span_count"] == 1
    assert summary["query_source"]["latest_trace_span_count"] == 1
    assert summary["latest_trace_spans_chronological"][0]["span_id"] == "span-canary"


def test_build_agents_check_summary_flags_tool_before_visible_input():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py")
    agents = {"agents": [{"agent_name": "nejumi-taiwan-openclaw"}], "total_count": 1}
    spans = {
        "spans": [
            {
                "started_at": "2026-06-29T00:00:00Z",
                "ended_at": "2026-06-29T00:00:01Z",
                "operation_name": "execute_tool",
                "agent_name": "nejumi-taiwan-openclaw",
                "trace_id": "trace-bad",
                "span_id": "span-tool",
                "tool_call_arguments": {"cmd": "python3 check.py"},
                "tool_call_result": "verified",
            },
            {
                "started_at": "2026-06-29T00:00:00Z",
                "ended_at": "2026-06-29T00:00:02Z",
                "operation_name": "chat",
                "agent_name": "nejumi-taiwan-openclaw",
                "trace_id": "trace-bad",
                "span_id": "span-user",
                "input_messages": [{"role": "user", "content": "problem"}],
            },
        ]
    }

    summary = module.build_agents_check_summary(
        agents,
        spans,
        entity="llm-leaderboard",
        project="tc-leaderboard",
        agent_name="nejumi-taiwan-openclaw",
        limit=10,
    )

    order = summary["trace_order_health"]
    assert order["trace_order_ok"] is False
    assert order["trace_user_message_order_ok"] is False
    assert "tool_started_before_or_at_first_message" in order["order_issues"]
    assert "tool_started_before_or_at_visible_user_input" in order["order_issues"]


def test_build_agents_check_summary_flags_tool_after_final_answer_end():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py")
    agents = {"agents": [{"agent_name": "nejumi-taiwan-openclaw"}], "total_count": 1}
    spans = {
        "spans": [
            {
                "started_at": "2026-06-29T00:00:00Z",
                "ended_at": "2026-06-29T00:00:01Z",
                "operation_name": "chat",
                "agent_name": "nejumi-taiwan-openclaw",
                "trace_id": "trace-final-order",
                "span_id": "span-user",
                "input_messages": [{"role": "user", "content": "problem"}],
            },
            {
                "started_at": "2026-06-29T00:00:02Z",
                "ended_at": "2026-06-29T00:00:03Z",
                "operation_name": "chat",
                "agent_name": "nejumi-taiwan-openclaw",
                "trace_id": "trace-final-order",
                "span_id": "span-answer",
                "output_messages": [{"role": "assistant", "content": "ANSWER: \\boxed{11}"}],
            },
            {
                "started_at": "2026-06-29T00:00:04Z",
                "ended_at": "2026-06-29T00:00:05Z",
                "operation_name": "execute_tool",
                "agent_name": "nejumi-taiwan-openclaw",
                "trace_id": "trace-final-order",
                "span_id": "span-tool",
                "tool_call_arguments": {"cmd": "python3 late.py"},
                "tool_call_result": "late",
            },
        ]
    }

    summary = module.build_agents_check_summary(
        agents,
        spans,
        entity="llm-leaderboard",
        project="tc-leaderboard",
        agent_name="nejumi-taiwan-openclaw",
        limit=10,
    )

    order = summary["trace_order_health"]
    assert order["trace_user_message_order_ok"] is True
    assert order["trace_final_answer_order_ok"] is False
    assert "tool_started_after_or_at_final_answer_end" in order["order_issues"]
    assert order["first_final_answer_ended_at"] == "2026-06-29T00:00:03Z"
    assert order["last_tool_started_at"] == "2026-06-29T00:00:04Z"


def test_build_agents_check_summary_flags_invalid_timestamps():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py")
    agents = {"agents": [{"agent_name": "nejumi-taiwan-openclaw"}], "total_count": 1}
    spans = {
        "spans": [
            {
                "started_at": "2026-06-29T00:00:03Z",
                "ended_at": "2026-06-29T00:00:02Z",
                "operation_name": "chat",
                "agent_name": "nejumi-taiwan-openclaw",
                "trace_id": "trace-invalid",
                "span_id": "span-reversed",
                "input_messages": [{"role": "user", "content": "problem"}],
            },
            {
                "started_at": "not-a-time",
                "ended_at": "2026-06-29T00:00:05Z",
                "operation_name": "execute_tool",
                "agent_name": "nejumi-taiwan-openclaw",
                "trace_id": "trace-invalid",
                "span_id": "span-invalid",
                "tool_call_arguments": {"cmd": "python3 check.py"},
            },
        ]
    }

    summary = module.build_agents_check_summary(
        agents,
        spans,
        entity="llm-leaderboard",
        project="tc-leaderboard",
        agent_name="nejumi-taiwan-openclaw",
        limit=10,
    )

    health = summary["content_capture_health"]
    assert health["spans_with_invalid_timestamps"] == 2
    assert health["trace_timestamp_quality_ok"] is False
    assert health["trace_order_ok"] is False
    assert health["trace_user_message_order_ok"] is False
    assert summary["trace_order_health"]["timestamp_issue_count"] == 2


def test_tool_policy_violations_catch_web_search_and_http_arguments():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py")
    events = [
        {
            "type": "tool_call",
            "toolCallId": "call_1",
            "toolName": "web_search",
            "arguments": {"query": "exact benchmark problem"},
            "index": 0,
        },
        {
            "type": "tool_call",
            "toolCallId": "call_2",
            "toolName": "code_execution",
            "arguments": {"task": "import requests\nrequests.get('https://example.com')"},
            "index": 1,
        },
        {
            "type": "tool_call",
            "toolCallId": "call_3",
            "toolName": "code_execution",
            "arguments": {"task": "print(2 + 2)"},
            "index": 2,
        },
    ]
    policy = {
        "deny_tools": ["code_execution", "web_*", "*search*"],
        "deny_argument_patterns": [r"https?://", r"\b(requests|urllib|httpx)\."],
    }

    violations = module.tool_policy_violations(events, policy)

    assert any(
        violation["type"] == "denied_tool" and violation["toolName"] == "web_search"
        for violation in violations
    )
    assert any(
        violation["type"] == "denied_argument_pattern" and violation["toolName"] == "code_execution"
        for violation in violations
    )
    assert any(
        violation["type"] == "denied_tool" and violation["toolName"] == "code_execution"
        for violation in violations
    )


def test_tool_policy_argument_patterns_do_not_block_static_file_writes():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py")
    events = [
        {
            "type": "tool_call",
            "toolCallId": "call_write",
            "toolName": "write",
            "arguments": {
                "path": "lib/example.py",
                "content": "DOCUMENTATION = 'https://docs.example.invalid/reference'",
            },
            "index": 0,
        },
        {
            "type": "tool_call",
            "toolCallId": "call_exec",
            "toolName": "exec",
            "arguments": {"cmd": "python - <<'PY'\nimport requests\nrequests.get('https://example.com')\nPY"},
            "index": 1,
        },
    ]
    policy = {
        "deny_tools": ["web_*", "*search*"],
        "deny_argument_patterns": [r"https?://", r"\b(requests|urllib|httpx)\."],
    }

    violations = module.tool_policy_violations(events, policy)

    assert not any(violation["toolName"] == "write" for violation in violations)
    assert any(
        violation["type"] == "denied_argument_pattern" and violation["toolName"] == "exec"
        for violation in violations
    )


def test_tool_policy_bare_url_pattern_does_not_block_local_url_literals_in_exec():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py")
    events = [
        {
            "type": "tool_call",
            "toolCallId": "call_exec_literal",
            "toolName": "exec",
            "arguments": {
                "command": "python - <<'PY'\npattern = 'https://www.example.org/'\nprint(pattern)\nPY"
            },
            "index": 0,
        },
        {
            "type": "tool_call",
            "toolCallId": "call_exec_network",
            "toolName": "exec",
            "arguments": {"command": "python - <<'PY'\nimport urllib.request\nurllib.request.urlopen('https://example.com')\nPY"},
            "index": 1,
        },
    ]
    policy = {
        "deny_tools": ["web_*", "*search*"],
        "deny_argument_patterns": [r"https?://", r"\b(requests|urllib|httpx)\."],
    }

    violations = module.tool_policy_violations(events, policy)

    assert not any(violation["toolCallId"] == "call_exec_literal" for violation in violations)
    assert any(
        violation["toolCallId"] == "call_exec_network"
        and violation["type"] == "denied_argument_pattern"
        for violation in violations
    )


def test_build_openclaw_command_can_wrap_nemoclaw_sandbox_exec():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py")
    args = Namespace(
        openclaw_bin="openclaw",
        nemoclaw_bin="nemoclaw",
        nemoclaw_sandbox="nejumi-taiwan",
        nemoclaw_workdir="/sandbox",
        profile=None,
        agent="main",
        session_key="bench:task",
        benchmark_id="agentic_math",
        task_id="task",
        timeout=120,
        local=True,
        model="deepseek/deepseek-v4-pro",
        thinking="max",
        openclaw_config_path=None,
    )

    command = module.build_openclaw_command(args, "hello", None)

    assert command[:9] == [
        "nemoclaw",
        "sandbox",
        "exec",
        "nejumi-taiwan",
        "--workdir",
        "/sandbox",
        "--no-tty",
        "--timeout",
        "180",
    ]
    assert command[9:11] == ["--", "openclaw"]
    assert command[11:14] == ["agent", "--agent", "main"]
    assert "--message" in command
    assert "hello" in command


def test_build_openclaw_command_passes_sandbox_visible_config_path_for_nemoclaw():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py")
    args = Namespace(
        openclaw_bin="openclaw",
        nemoclaw_bin="nemoclaw",
        nemoclaw_sandbox="nejumi-taiwan",
        nemoclaw_workdir="/sandbox",
        profile=None,
        agent="main",
        session_key="bench:task",
        benchmark_id="agentic_math",
        task_id="task",
        timeout=120,
        local=True,
        model=None,
        thinking="high",
        openclaw_config_path=Path("/sandbox/repo/.nejumi_openclaw/openclaw_config.json"),
    )

    command = module.build_openclaw_command(args, "hello", None)

    assert command[9:12] == [
        "--",
        "env",
        "OPENCLAW_CONFIG_PATH=/sandbox/repo/.nejumi_openclaw/openclaw_config.json",
    ]
    assert command[12:14] == ["openclaw", "agent"]
