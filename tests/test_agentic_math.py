import json
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_script_module(path: Path):
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        return load_module(path)
    finally:
        sys.path.pop(0)


def test_agentic_math_main_rejects_weave_sidecar_before_dataset_read(tmp_path, monkeypatch):
    module = load_script_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_agentic_math_openclaw.py",
            "--dataset-jsonl",
            str(tmp_path / "missing.jsonl"),
            "--weave-sidecar",
        ],
    )

    with pytest.raises(SystemExit, match="Weave sidecar logging is disabled"):
        module.main()


def test_agentic_math_weave_agents_verifier_failure_is_per_instance_evidence(
    tmp_path, monkeypatch
):
    module = load_script_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")

    def raise_verifier_failure(**kwargs):
        raise RuntimeError("trace content was incomplete")

    monkeypatch.setattr(module, "verify_native_weave_agents_trace", raise_verifier_failure)
    args = SimpleNamespace(
        verify_weave_agents=True,
        dry_run=False,
        weave_agents_entity="llm-leaderboard",
        weave_agents_project="tc-leaderboard",
        weave_agents_agent_name="nejumi-taiwan-openclaw",
        weave_agents_env_file=tmp_path / ".env",
        weave_agents_limit=50,
        weave_agents_verification_timeout=0,
        weave_agents_poll_seconds=1,
        model="openrouter-direct/z-ai/glm-5.2",
    )

    evidence = module.verify_weave_agents_for_attempt(
        {"task_id": "olymmath_hard_1_zh"},
        tmp_path,
        args,
        session_key="run:math:olymmath_hard_1_zh:attempt-1",
        agent_id="tw-math-test-agent",
        sidecar={"tool_call_count": 2},
    )

    assert evidence["weave_agents_required"] is True
    assert evidence["weave_agents_ok"] is False
    assert "trace content was incomplete" in evidence["weave_agents_error"]
    assert "agent:tw-math-test-agent:run:math:olymmath_hard_1_zh:attempt-1" in evidence[
        "weave_agents_conversation_id"
    ]
    assert evidence["weave_agents_verifier_json"].endswith(".json")


def test_extract_answer_prefers_answer_line():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    text = "I checked 12 cases.\nANSWER: 070\n"
    assert module.extract_answer(text) == "70"


def test_extract_answer_accepts_boxed_fallback():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    text = "Therefore the final value is \\boxed{336}."
    assert module.extract_answer(text) == "336"


def test_extract_answer_accepts_symbolic_answer_line():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    text = "Reasoning omitted.\nANSWER: \\frac{\\sqrt{51}}{6}\n"
    assert module.extract_answer(text) == "\\frac{\\sqrt{51}}{6}"


def test_extract_text_from_openclaw_accepts_gateway_result_payloads():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    sidecar = {
        "stdout_json": {
            "status": "ok",
            "result": {
                "payloads": [
                    {
                        "text": "The answer follows from the blocked-budget probe.\n\nANSWER: 333"
                    }
                ],
                "meta": {
                    "finalAssistantVisibleText": "ANSWER: 111",
                },
            },
        }
    }

    text = module.extract_text_from_openclaw(sidecar)

    assert text.endswith("ANSWER: 333")
    assert module.extract_answer(text) == "333"


def test_extract_text_from_openclaw_accepts_gateway_result_meta_fallback():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    sidecar = {
        "stdout_json": {
            "status": "ok",
            "result": {
                "meta": {
                    "finalAssistantVisibleText": "No payloads were emitted.\n\nANSWER: 222",
                },
            },
        }
    }

    text = module.extract_text_from_openclaw(sidecar)

    assert text.endswith("ANSWER: 222")
    assert module.extract_answer(text) == "222"


def test_math_equivalence_accepts_latex_radicals_and_fractions():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    score = module.answers_equivalent("241+44sqrt(30)", "241 + 44\\sqrt{30}")
    assert score["equivalent"] is True
    score = module.answers_equivalent("sqrt(51)/6", "\\frac{\\sqrt{51}}{6}")
    assert score["equivalent"] is True


def test_math_equivalence_accepts_intervals():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    score = module.answers_equivalent("[3, 193 - 132 sqrt(2)]", "[3,193-132\\sqrt{2}]")
    assert score["equivalent"] is True


def test_math_equivalence_accepts_chained_interval_inequality():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    score = module.answers_equivalent(
        "3 \\le m \\le 193 - 132\\sqrt{2}",
        "[3,193-132\\sqrt{2}]",
    )
    assert score["equivalent"] is True
    assert score["method"] == "interval_sympy"


def test_openclaw_error_record_marks_incorrect(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    sidecar_path = tmp_path / "openclaw_result.json"
    sidecar_path.write_text(
        '{"returncode": 1, "stderr": "LLM request timed out."}\n',
        encoding="utf-8",
    )
    row = {"task_id": "task_1", "answer": "42", "question": "q"}
    completed = subprocess.CompletedProcess(["cmd"], 1, stdout=str(sidecar_path), stderr="")
    cache_key = {
        "runner_version": module.RUNNER_VERSION,
        "task_id": "task_1",
        "prompt_hash": "abc",
        "model": "model",
        "thinking": "high",
        "answer_format": "math_expression",
    }

    record = module.build_openclaw_error_record(row, completed, sidecar_path, cache_key)

    assert record["correct"] is False
    assert record["predicted_answer"] is None
    assert record["scoring_method"] == "openclaw_error"
    assert "timed out" in record["scoring_error"]
    assert record["openclaw_returncode"] == 1
    assert record["cache_key"] == cache_key


def test_openclaw_error_record_marks_hard_conversation_order_violation_disqualified(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    sidecar_path = tmp_path / "openclaw_result.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "returncode": 0,
                "stderr": "Conversation order violation",
                "conversation_order": {
                    "ok": False,
                    "issues": [{"type": "tool_before_or_at_first_user_message"}],
                },
            }
        ),
        encoding="utf-8",
    )
    row = {"task_id": "task_1", "answer": "42", "question": "q"}
    completed = subprocess.CompletedProcess(
        ["cmd"],
        1,
        stdout=str(sidecar_path),
        stderr="Conversation order violation",
    )
    cache_key = {
        "runner_version": module.RUNNER_VERSION,
        "task_id": "task_1",
        "prompt_hash": "abc",
        "model": "model",
        "thinking": "high",
        "answer_format": "math_expression",
    }

    record = module.build_openclaw_error_record(row, completed, sidecar_path, cache_key)

    assert record["correct"] is False
    assert record["scoring_method"] == "openclaw_error"
    assert record["conversation_order_ok"] is False
    assert record["openclaw_disqualified_reason"] == "conversation_order_violation"


def test_transient_openclaw_failure_detects_provider_timeout():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    completed = subprocess.CompletedProcess(
        ["cmd"],
        1,
        stdout="",
        stderr="FailoverError: LLM request timed out. rawError=terminated",
    )

    assert module.is_transient_openclaw_failure(completed, None)


def test_outer_openclaw_timeout_is_scoreable_time_up_not_retry():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    completed = subprocess.CompletedProcess(
        ["cmd"],
        124,
        stdout="",
        stderr="Command timed out after 960 seconds",
    )

    assert module.is_outer_openclaw_timeout(completed, None)
    assert not module.is_transient_openclaw_failure(completed, None)
    assert module.non_scoreable_openclaw_failure_reason(completed, None) is None


def test_returncode_zero_provider_timeout_sidecar_is_transient():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    completed = subprocess.CompletedProcess(["cmd"], 0, stdout="", stderr="")
    sidecar = {
        "returncode": 0,
        "stderr": "(node:123) [UNDICI-EHPA] Warning: proxy warning",
        "stdout_json": {
            "status": "timeout",
            "timeoutPhase": "provider",
            "result": {
                "payloads": [
                    {
                        "text": (
                            "LLM request failed.\n\n"
                            "Request timed out before a response was generated."
                        )
                    }
                ]
            },
        },
        "runtime_budget": {"ok": True, "violations": []},
        "tool_policy_ok": True,
        "tool_policy_violations": [],
    }

    assert module.is_transient_openclaw_failure(completed, sidecar)
    assert "Request timed out before a response was generated" in module.sidecar_error_text(
        sidecar, ""
    )


def test_transient_openclaw_failure_detects_provider_sse_rate_limit():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    completed = subprocess.CompletedProcess(
        ["cmd"],
        1,
        stdout="",
        stderr="",
    )
    sidecar = {
        "stderr": (
            "FailoverError: JSON error injected into SSE stream "
            "stage=assistant decision=surface_error reason=rate_limit"
        ),
        "tool_policy_ok": True,
        "tool_policy_violations": [],
    }

    assert module.is_transient_openclaw_failure(completed, sidecar)


def test_runtime_budget_openclaw_failure_is_not_transient():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    completed = subprocess.CompletedProcess(
        ["cmd"],
        125,
        stdout="",
        stderr="Live runtime budget exceeded: reason=timeout rawError=terminated",
    )
    sidecar = {
        "runtime_budget": {
            "ok": False,
            "violations": [{"type": "budget_guard_blocked", "source": "live_runtime_budget"}],
        },
        "tool_policy_ok": True,
        "tool_policy_violations": [],
    }

    assert not module.is_transient_openclaw_failure(completed, sidecar)


def test_provider_timeout_interrupt_is_transient():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    completed = subprocess.CompletedProcess(
        ["cmd"],
        125,
        stdout="",
        stderr=(
            'Live OpenClaw interrupt: provider_timeout_count=1 '
            'provider_timeouts=[{"errorCode":"504","errorMessage":"Upstream idle timeout exceeded"}]'
        ),
    )

    assert module.is_transient_openclaw_failure(completed, None)


def test_cached_transient_openclaw_error_is_not_reusable():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    record = {
        "scoring_method": "openclaw_error",
        "scoring_error": "FailoverError: JSON error injected into SSE stream reason=rate_limit",
    }

    assert not module.cached_result_is_reusable(record)


def test_cached_non_transient_openclaw_error_is_reusable():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    record = {
        "scoring_method": "openclaw_error",
        "scoring_error": "agent returned no answer without provider failure",
    }

    assert module.cached_result_is_reusable(record)


def test_cached_nemoclaw_result_requires_session_audit(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    cache_key = {
        "task_id": "task_1",
        "prompt_hash": "prompt-hash",
        "agent_runtime": "nemoclaw",
        "nemoclaw_sandbox": "nejumi-taiwan",
    }
    command = ["python3", "run_openclaw_agent_protocol.py", "run"]
    invocation = {
        "cache_key": cache_key,
        "command": command,
        "command_sha256": module.command_sha256(command),
        "expected_openclaw_result_path": str(tmp_path / "openclaw_result.json"),
    }
    invocation_path = tmp_path / "openclaw_invocation.json"
    invocation_path.write_text(json.dumps(invocation, ensure_ascii=False), encoding="utf-8")
    record = {
        "task_id": "task_1",
        "cache_key": cache_key,
        "correct": True,
        "openclaw_result_path": invocation["expected_openclaw_result_path"],
        "openclaw_invocation_path": str(invocation_path),
        "openclaw_invocation_sha256": module.sha256_file(invocation_path),
        "openclaw_command_sha256": invocation["command_sha256"],
    }

    assert not module.cached_result_matches_cache(record, cache_key)
    record["nemoclaw_session_audit"] = {"required": True, "ok": False}
    record["nemoclaw_session_audit_ok"] = False
    assert not module.cached_result_matches_cache(record, cache_key)
    record["nemoclaw_session_audit"] = {"required": True, "ok": True}
    record["nemoclaw_session_audit_ok"] = True
    assert module.cached_result_matches_cache(record, cache_key)
    record["openclaw_invocation_sha256"] = "0" * 64
    assert not module.cached_result_matches_cache(record, cache_key)
    record["openclaw_invocation_sha256"] = module.sha256_file(invocation_path)
    record["conversation_order_ok"] = False
    assert not module.cached_result_matches_cache(record, cache_key)
    record["conversation_order_ok"] = True
    record["conversation_order"] = {"ok": False}
    assert not module.cached_result_matches_cache(record, cache_key)
    record["conversation_order"] = {"ok": True}
    assert module.cached_result_matches_cache(record, cache_key)
    record["tool_policy_ok"] = False
    assert not module.cached_result_matches_cache(record, cache_key)
    record["tool_policy_ok"] = True
    record["tool_policy_violations"] = [{"type": "denied_tool", "toolName": "web_search"}]
    assert not module.cached_result_matches_cache(record, cache_key)
    record["tool_policy_violations"] = []
    assert module.cached_result_matches_cache(record, cache_key)
    record["weave_sidecar"] = {"ok": False}
    assert not module.cached_result_matches_cache(record, cache_key)
    record["weave_sidecar"] = {"ok": True}
    assert module.cached_result_matches_cache(record, cache_key)
    record["weave_sidecar_ok"] = False
    assert not module.cached_result_matches_cache(record, cache_key)
    record["weave_sidecar_ok"] = True
    assert module.cached_result_matches_cache(record, cache_key)


def test_cached_math_scoreable_disqualification_allows_failed_session_audit(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    cache_key = {
        "task_id": "task_1",
        "prompt_hash": "prompt-hash",
        "agent_runtime": "nemoclaw",
        "nemoclaw_sandbox": "nejumi-taiwan",
    }
    sidecar_path = tmp_path / "openclaw_result.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "runtime_budget": {
                    "observed": {
                        "actual_usage": {
                            "inputTokens": 100,
                            "outputTokens": 20,
                            "cacheReadInputTokens": 300,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    command = ["python3", "run_openclaw_agent_protocol.py", "run"]
    invocation = {
        "cache_key": cache_key,
        "command": command,
        "command_sha256": module.command_sha256(command),
        "expected_openclaw_result_path": str(sidecar_path),
    }
    invocation_path = tmp_path / "openclaw_invocation.json"
    invocation_path.write_text(json.dumps(invocation, ensure_ascii=False), encoding="utf-8")
    record_path = tmp_path / "result.json"
    record = {
        "task_id": "task_1",
        "cache_key": cache_key,
        "correct": False,
        "scoring_method": "openclaw_error",
        "scoring_error": "Runtime budget exceeded",
        "openclaw_disqualified_reason": "runtime_budget_exceeded",
        "openclaw_result_path": str(sidecar_path),
        "openclaw_invocation_path": str(invocation_path),
        "openclaw_invocation_sha256": module.sha256_file(invocation_path),
        "openclaw_command_sha256": invocation["command_sha256"],
        "conversation_order_ok": False,
        "conversation_order": {"ok": False},
        "tool_policy_ok": True,
        "tool_policy_violations": [],
        "nemoclaw_session_audit": {"required": True, "ok": False},
        "nemoclaw_session_audit_ok": False,
    }
    record_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    assert module.cached_result_matches_cache(record, cache_key)
    cached = module.backfill_record_usage(record, record_path)

    assert cached["openclaw_usage"] == {
        "inputTokens": 100,
        "outputTokens": 20,
        "cacheReadInputTokens": 300,
    }


def test_math_sidecar_usage_reads_result_meta_and_runtime_budget():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")

    assert module.sidecar_usage(
        {
            "stdout_json": {
                "result": {
                    "meta": {
                        "agentMeta": {
                            "usage": {"input": 10, "output": 2, "cacheRead": 30}
                        }
                    }
                }
            }
        }
    ) == {"input": 10, "output": 2, "cacheRead": 30}
    assert module.sidecar_usage(
        {
            "runtime_budget": {
                "observed": {
                    "actual_usage": {
                        "inputTokens": 11,
                        "outputTokens": 3,
                        "cacheReadInputTokens": 31,
                    }
                }
            }
        }
    ) == {"inputTokens": 11, "outputTokens": 3, "cacheReadInputTokens": 31}


def test_transient_openclaw_failure_ignores_tool_policy_violation():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    completed = subprocess.CompletedProcess(
        ["cmd"],
        1,
        stdout="",
        stderr="FailoverError: LLM request timed out.",
    )

    assert not module.is_transient_openclaw_failure(
        completed,
        {"tool_policy_ok": False, "tool_policy_violations": [{"type": "denied_tool"}]},
    )


def test_non_scoreable_openclaw_failure_detects_configuration_and_quota_errors():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")

    unsupported = subprocess.CompletedProcess(
        ["cmd"],
        1,
        stdout="",
        stderr='Error: Thinking level "xhigh" is not supported for provider/model.',
    )
    assert module.non_scoreable_openclaw_failure_reason(unsupported, None) == "unsupported_thinking"

    quota = subprocess.CompletedProcess(
        ["cmd"],
        1,
        stdout="",
        stderr="code=insufficient_quota message=You exceeded your current quota",
    )
    assert module.non_scoreable_openclaw_failure_reason(quota, None) == "insufficient_quota"

    workspace = subprocess.CompletedProcess(
        ["cmd"],
        1,
        stdout="",
        stderr="WorkspaceVanishedError: OpenClaw workspace appears to have disappeared",
    )
    assert module.non_scoreable_openclaw_failure_reason(workspace, None) == "workspace_vanished"

    order = subprocess.CompletedProcess(
        ["cmd"],
        1,
        stdout="",
        stderr="Conversation order violation",
    )
    assert module.non_scoreable_openclaw_failure_reason(order, None) is None

    session_audit = subprocess.CompletedProcess(
        ["cmd"],
        1,
        stdout="",
        stderr="NeMoClaw session audit failed",
    )
    assert (
        module.non_scoreable_openclaw_failure_reason(session_audit, None)
        == "nemoclaw_session_audit_failed"
    )


def test_non_scoreable_openclaw_failure_does_not_catch_transient_timeout():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    completed = subprocess.CompletedProcess(
        ["cmd"],
        1,
        stdout="",
        stderr="FailoverError: LLM request timed out. rawError=terminated",
    )

    assert module.non_scoreable_openclaw_failure_reason(completed, None) is None
    assert module.is_transient_openclaw_failure(completed, None)


def test_recover_workspace_vanished_attestation(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    attestation_root = tmp_path / ".openclaw" / "workspace-attestations"
    attestation_root.mkdir(parents=True)
    attestation = attestation_root / "abc.attested"
    attestation.write_text("openclaw-workspace-attestation:v1\n", encoding="utf-8")
    message = (
        "WorkspaceVanishedError: OpenClaw workspace appears to have disappeared after "
        f"a recent initialization: {workspace}. Refusing to reseed BOOTSTRAP.md over "
        f"a recently attested workspace. Restore the workspace or remove {attestation} "
        "if this reset was intentional."
    )
    completed = subprocess.CompletedProcess(["cmd"], 1, stdout="", stderr=message)
    task_dir = tmp_path / "task"

    recovery = module.recover_workspace_vanished_failure(task_dir, completed, None)

    assert recovery is not None
    assert recovery["workspace"] == str(workspace.resolve())
    assert not attestation.exists()
    moved = Path(recovery["moved_to"])
    assert moved.exists()
    assert moved.parent == task_dir / "recovered_workspace_attestations"


def test_cache_key_requires_prompt_model_thinking_and_runner_version():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    args = type(
        "Args",
        (),
        {
            "model": "deepseek/deepseek-v4-pro",
            "thinking": "max",
            "deny_tool": ["web_search"],
            "deny_argument_pattern": [r"https?://"],
        },
    )()
    row = {"task_id": "task_1", "answer_format": "math_expression"}
    key = module.build_cache_key(row, "prompt", args)

    assert module.cache_key_matches({"cache_key": key}, key)
    assert key["deny_tools"] == ["web_search"]
    assert key["deny_argument_patterns"] == [r"https?://"]

    stale = dict(key)
    stale["prompt_hash"] = "different"
    assert not module.cache_key_matches({"cache_key": stale}, key)


def test_cache_key_includes_nemoclaw_openclaw_config_source():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    args = type(
        "Args",
        (),
        {
            "model": "openai-direct/example-model",
            "thinking": "high",
            "deny_tool": None,
            "deny_argument_pattern": None,
            "nemoclaw_sandbox": "nejumi-taiwan",
            "nemoclaw_openclaw_config_path": "/sandbox/.openclaw/openclaw.json",
        },
    )()
    row = {"task_id": "task_1", "answer_format": "math_expression"}

    key = module.build_cache_key(row, "prompt", args)
    changed_args = type(
        "Args",
        (),
        {
            "model": "openai-direct/example-model",
            "thinking": "high",
            "deny_tool": None,
            "deny_argument_pattern": None,
            "nemoclaw_sandbox": "nejumi-taiwan",
            "nemoclaw_openclaw_config_path": "/sandbox/other-openclaw.json",
        },
    )()
    changed_key = module.build_cache_key(row, "prompt", changed_args)

    assert key["openclaw_config_source"] == "/sandbox/.openclaw/openclaw.json"
    assert not module.cache_key_matches({"cache_key": changed_key}, key)


def test_agentic_math_session_prefix_is_bound_to_wandb_run_id(monkeypatch):
    monkeypatch.setenv("WANDB_RUN_ID", "twcanary-run-1")
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    args = type(
        "Args",
        (),
        {
            "session_prefix": None,
            "model": "openai-direct/example-model",
            "thinking": "high",
            "deny_tool": None,
            "deny_argument_pattern": None,
        },
    )()

    assert module.resolve_session_prefix(args) == "twcanary-run-1:agentic-math"
    key = module.build_cache_key(
        {"task_id": "task_1", "answer_format": "math_expression"},
        "prompt",
        args,
    )
    assert key["session_prefix"] == "twcanary-run-1:agentic-math"


def test_agentic_math_session_prefix_expands_wandb_placeholder(monkeypatch):
    monkeypatch.setenv("WANDB_RUN_ID", "twcanary-run-2")
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    args = type("Args", (), {"session_prefix": "math/{wandb_run_id}"})()

    assert module.resolve_session_prefix(args) == "math/twcanary-run-2"


def test_agentic_math_openclaw_context_tokens_updates_existing_model_entry():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    config = {
        "models": {
            "providers": {
                "openai-direct": {
                    "models": [
                        {
                            "id": "gpt-4.1-mini-2025-04-14",
                            "contextWindow": 1_047_576,
                        }
                    ]
                }
            }
        }
    }
    args = SimpleNamespace(
        model="openai-direct/gpt-4.1-mini-2025-04-14",
        max_input_tokens=500_000,
    )

    result = module.configure_openclaw_context_tokens(config, args)

    assert result == {
        "provider": "openai-direct",
        "model": "gpt-4.1-mini-2025-04-14",
        "contextTokens": 500_000,
    }
    [entry] = config["models"]["providers"]["openai-direct"]["models"]
    assert entry["contextTokens"] == 500_000


def test_agentic_math_openclaw_context_tokens_keeps_stricter_existing_cap():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    config = {
        "models": {
            "providers": {
                "openai-direct": {
                    "models": [
                        {
                            "id": "gpt-4.1-mini-2025-04-14",
                            "contextWindow": 1_047_576,
                            "contextTokens": 200_000,
                        }
                    ]
                }
            }
        }
    }
    args = SimpleNamespace(
        model="openai-direct/gpt-4.1-mini-2025-04-14",
        max_input_tokens=500_000,
    )

    result = module.configure_openclaw_context_tokens(config, args)

    assert result["contextTokens"] == 200_000
    [entry] = config["models"]["providers"]["openai-direct"]["models"]
    assert entry["contextTokens"] == 200_000


def test_agentic_math_run_passes_wandb_scoped_session_key(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_RUN_ID", "twcanary-run-3")
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    captured_command = []
    row = {
        "task_id": "math_1",
        "answer": "2",
        "question": "1+1?",
        "subject": "algebra",
        "answer_format": "math_expression",
    }

    def fake_run_command(command, cwd=None, timeout=None, check=True):
        captured_command[:] = command
        output_dir = Path(command[command.index("--output-dir") + 1])
        prompt_text = Path(command[command.index("--prompt-file") + 1]).read_text(encoding="utf-8")
        sidecar_path = module.task_sidecar_path(output_dir, row["task_id"])
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(
            json.dumps(
                {
                    "returncode": 0,
                    "metadata": {
                        "task_id": row["task_id"],
                        "prompt_hash": module.sha256_text(prompt_text),
                        "model_id": command[command.index("--model") + 1],
                        "openclaw_config_source": command[command.index("--openclaw-config-source") + 1],
                    },
                    "tool_policy": {
                        "deny_tools": [
                            command[index + 1]
                            for index, token in enumerate(command[:-1])
                            if token == "--deny-tool"
                        ],
                        "deny_argument_patterns": [
                            command[index + 1]
                            for index, token in enumerate(command[:-1])
                            if token == "--deny-argument-pattern"
                        ],
                    },
                    "stdout": "ANSWER: 2\n",
                    "tool_policy_ok": True,
                    "tool_policy_violations": [],
                    "tool_call_count": 0,
                    "tool_error_count": 0,
                    "conversation_order": {"ok": True},
                    "weave_sidecar": {"ok": True},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout=str(sidecar_path), stderr="")

    monkeypatch.setattr(module, "run_command", fake_run_command)
    args = type(
        "Args",
        (),
        {
            "agent": "main",
            "allow_failed_preflight": False,
            "deny_argument_pattern": None,
            "deny_tool": None,
            "dry_run": False,
            "fail_fast": False,
            "model": "openai-direct/example-model",
            "nemoclaw_sandbox": None,
            "no_local": False,
            "openclaw_max_attempts": 1,
            "openclaw_retry_base_seconds": 0,
            "openclaw_timeout": 30,
            "max_input_tokens": 12345,
            "max_tool_calls": 7,
            "max_agent_turns": 8,
            "profile": None,
            "redo": False,
            "session_prefix": None,
            "thinking": "high",
            "use_task_agent": False,
            "weave_sidecar": False,
            "weave_sidecar_strict": False,
        },
    )()

    record = module.run_openclaw_for_task(row, tmp_path / "task", args)

    session_key = captured_command[captured_command.index("--session-key") + 1]
    config_source = captured_command[captured_command.index("--openclaw-config-source") + 1]
    assert session_key.startswith("twcanary-run-3:agentic-math:math_1:")
    assert config_source == record["cache_key"]["openclaw_config_source"]
    assert captured_command[captured_command.index("--max-input-tokens") + 1] == "12345"
    assert captured_command[captured_command.index("--max-tool-calls") + 1] == "7"
    assert captured_command[captured_command.index("--max-agent-turns") + 1] == "8"
    assert record["correct"] is True
    assert record["cache_key"]["session_prefix"] == "twcanary-run-3:agentic-math"
    assert record["cache_key"]["max_input_tokens"] == 12345
    assert record["cache_key"]["max_tool_calls"] == 7
    assert record["cache_key"]["max_agent_turns"] == 8


def test_agentic_math_returncode_zero_provider_timeout_is_disqualified(tmp_path, monkeypatch):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    row = {
        "task_id": "math_1",
        "answer": "2",
        "question": "1+1?",
        "subject": "algebra",
        "answer_format": "math_expression",
    }

    def fake_run_command(command, cwd=None, timeout=None, check=True):
        output_dir = Path(command[command.index("--output-dir") + 1])
        prompt_text = Path(command[command.index("--prompt-file") + 1]).read_text(encoding="utf-8")
        sidecar_path = module.task_sidecar_path(output_dir, row["task_id"])
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(
            json.dumps(
                {
                    "returncode": 0,
                    "metadata": {
                        "task_id": row["task_id"],
                        "prompt_hash": module.sha256_text(prompt_text),
                        "model_id": command[command.index("--model") + 1],
                        "openclaw_config_source": command[command.index("--openclaw-config-source") + 1],
                    },
                    "stderr": "(node:123) [UNDICI-EHPA] Warning: proxy warning",
                    "stdout_json": {
                        "status": "timeout",
                        "timeoutPhase": "provider",
                        "stopReason": "rpc",
                        "result": {
                            "payloads": [
                                {
                                    "text": (
                                        "LLM request failed.\n\n"
                                        "Request timed out before a response was generated."
                                    )
                                }
                            ],
                            "meta": {"agentMeta": {"usage": {"totalTokens": 123}}},
                        },
                    },
                    "tool_policy_ok": True,
                    "tool_policy_violations": [],
                    "runtime_budget": {"ok": True, "violations": []},
                    "tool_call_count": 4,
                    "tool_error_count": 1,
                    "conversation_order": {"ok": True},
                    "weave_sidecar": {"ok": True},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout=str(sidecar_path), stderr="")

    monkeypatch.setattr(module, "run_command", fake_run_command)
    args = type(
        "Args",
        (),
        {
            "agent": "main",
            "allow_failed_preflight": False,
            "deny_argument_pattern": None,
            "deny_tool": None,
            "dry_run": False,
            "fail_fast": False,
            "model": "openai-direct/example-model",
            "nemoclaw_sandbox": None,
            "no_local": False,
            "openclaw_max_attempts": 1,
            "openclaw_retry_base_seconds": 0,
            "openclaw_timeout": 30,
            "max_input_tokens": 12345,
            "max_tool_calls": 7,
            "max_agent_turns": 8,
            "max_tool_wall_seconds": 60,
            "profile": None,
            "redo": False,
            "session_prefix": None,
            "thinking": "high",
            "use_task_agent": False,
            "weave_sidecar": False,
            "weave_sidecar_strict": False,
        },
    )()

    record = module.run_openclaw_for_task(row, tmp_path / "task", args)

    assert record["correct"] is False
    assert record["scoring_method"] == "openclaw_error"
    assert record["openclaw_disqualified_reason"] == "provider_transient_exhausted"
    assert record["openclaw_returncode"] == 0
    assert "Request timed out before a response was generated" in record["scoring_error"]
    failures = (tmp_path / "task" / "openclaw_transient_failures.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"exhausted": true' in failures


def test_agentic_math_fresh_success_rejects_sidecar_config_source_mismatch(tmp_path, monkeypatch):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    row = {
        "task_id": "math_1",
        "answer": "2",
        "question": "1+1?",
        "subject": "algebra",
        "answer_format": "math_expression",
    }

    def fake_run_command(command, cwd=None, timeout=None, check=True):
        output_dir = Path(command[command.index("--output-dir") + 1])
        prompt_text = Path(command[command.index("--prompt-file") + 1]).read_text(encoding="utf-8")
        sidecar_path = module.task_sidecar_path(output_dir, row["task_id"])
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(
            json.dumps(
                {
                    "returncode": 0,
                    "metadata": {
                        "task_id": row["task_id"],
                        "prompt_hash": module.sha256_text(prompt_text),
                        "model_id": command[command.index("--model") + 1],
                        "openclaw_config_source": "/sandbox/other-openclaw.json",
                    },
                    "tool_policy": {
                        "deny_tools": [
                            command[index + 1]
                            for index, token in enumerate(command[:-1])
                            if token == "--deny-tool"
                        ],
                        "deny_argument_patterns": [
                            command[index + 1]
                            for index, token in enumerate(command[:-1])
                            if token == "--deny-argument-pattern"
                        ],
                    },
                    "stdout": "ANSWER: 2\n",
                    "tool_policy_ok": True,
                    "tool_policy_violations": [],
                    "conversation_order": {"ok": True},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout=str(sidecar_path), stderr="")

    monkeypatch.setattr(module, "run_command", fake_run_command)
    args = type(
        "Args",
        (),
        {
            "agent": "main",
            "allow_failed_preflight": False,
            "deny_argument_pattern": None,
            "deny_tool": None,
            "dry_run": False,
            "fail_fast": False,
            "model": "openai-direct/example-model",
            "nemoclaw_sandbox": None,
            "no_local": False,
            "openclaw_max_attempts": 1,
            "openclaw_retry_base_seconds": 0,
            "openclaw_timeout": 30,
            "profile": None,
            "redo": False,
            "session_prefix": None,
            "thinking": "high",
            "use_task_agent": False,
            "weave_sidecar": False,
            "weave_sidecar_strict": False,
        },
    )()

    with pytest.raises(RuntimeError, match="OpenClaw sidecar metadata mismatch"):
        module.run_openclaw_for_task(row, tmp_path / "task", args)


def test_agentic_math_fresh_success_rejects_required_nemoclaw_audit_failure(tmp_path, monkeypatch):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    row = {
        "task_id": "math_1",
        "answer": "2",
        "question": "1+1?",
        "subject": "algebra",
        "answer_format": "math_expression",
    }

    def fake_run_command(command, cwd=None, timeout=None, check=True):
        output_dir = Path(command[command.index("--output-dir") + 1])
        prompt_text = Path(command[command.index("--prompt-file") + 1]).read_text(encoding="utf-8")
        sidecar_path = module.task_sidecar_path(output_dir, row["task_id"])
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(
            json.dumps(
                {
                    "returncode": 0,
                    "metadata": {
                        "task_id": row["task_id"],
                        "prompt_hash": module.sha256_text(prompt_text),
                        "model_id": command[command.index("--model") + 1],
                        "openclaw_config_source": command[command.index("--openclaw-config-source") + 1],
                    },
                    "tool_policy": {
                        "deny_tools": [
                            command[index + 1]
                            for index, token in enumerate(command[:-1])
                            if token == "--deny-tool"
                        ],
                        "deny_argument_patterns": [
                            command[index + 1]
                            for index, token in enumerate(command[:-1])
                            if token == "--deny-argument-pattern"
                        ],
                    },
                    "stdout": "ANSWER: 2\n",
                    "tool_policy_ok": True,
                    "tool_policy_violations": [],
                    "conversation_order": {"ok": True, "checked": True},
                    "nemoclaw_session_audit": {"required": True, "ok": False},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout=str(sidecar_path), stderr="")

    monkeypatch.setattr(module, "run_command", fake_run_command)
    args = type(
        "Args",
        (),
        {
            "agent": "main",
            "allow_failed_preflight": False,
            "deny_argument_pattern": None,
            "deny_tool": None,
            "dry_run": False,
            "fail_fast": False,
            "model": "openai-direct/example-model",
            "nemoclaw_bin": "nemoclaw",
            "nemoclaw_openclaw_config_path": "/sandbox/.openclaw/openclaw.json",
            "nemoclaw_sandbox": "nejumi-taiwan",
            "nemoclaw_workdir": "/sandbox",
            "no_local": False,
            "openclaw_max_attempts": 1,
            "openclaw_retry_base_seconds": 0,
            "openclaw_timeout": 30,
            "profile": None,
            "redo": False,
            "session_prefix": None,
            "thinking": "high",
            "use_task_agent": False,
            "weave_sidecar": False,
            "weave_sidecar_strict": False,
        },
    )()

    with pytest.raises(RuntimeError, match="OpenClaw NeMoClaw session audit mismatch"):
        module.run_openclaw_for_task(row, tmp_path / "task", args)


def test_archive_stale_result_moves_active_result_out_of_collection_path(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    result_path = tmp_path / "result.json"
    result_path.write_text('{"task_id": "task_1", "prompt_hash": "oldhash", "cache_key": {"runner_version": "old"}}\n')
    stale_invocation = tmp_path / "openclaw_invocation.json"
    stale_invocation.write_text('{"latest_attempt_id": "old"}\n', encoding="utf-8")
    expected_cache_key = {
        "runner_version": module.RUNNER_VERSION,
        "task_id": "task_1",
        "prompt_hash": "newhash",
        "model": "model",
        "thinking": "high",
        "answer_format": "math_expression",
    }

    archive_path = module.archive_stale_result(tmp_path, result_path, expected_cache_key)

    assert archive_path is not None
    assert archive_path.parent == tmp_path / "stale_results"
    payload = json.loads(archive_path.read_text(encoding="utf-8"))
    assert payload["archive_reason"] == "cache_key_mismatch"
    assert payload["expected_cache_key"] == expected_cache_key
    assert payload["archived_result"]["prompt_hash"] == "oldhash"
    assert not result_path.exists()
    assert not stale_invocation.exists()


def test_task_sidecar_path_uses_attempt_output_dir(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    attempt_dir = tmp_path / "openclaw_attempts" / "attempt-1"

    path = module.task_sidecar_path(attempt_dir, "task_1")

    assert path == attempt_dir / "agentic_math" / "task_1" / "openclaw_result.json"


def test_success_sidecar_matches_cache_and_recovers_attempt_metadata(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    cache_key = {
        "runner_version": module.RUNNER_VERSION,
        "task_id": "task_1",
        "prompt_hash": "prompt-hash",
        "model": "provider/model",
        "thinking": "high",
        "answer_format": "math_expression",
        "deny_tools": ["code_execution", "web_search"],
        "deny_argument_patterns": [r"https?://"],
        "openclaw_config_source": "/sandbox/.openclaw/openclaw.json",
    }
    sidecar = {
        "returncode": 0,
        "metadata": {
            "task_id": "task_1",
            "prompt_hash": "prompt-hash",
            "model_id": "provider/model",
            "openclaw_config_source": "/sandbox/.openclaw/openclaw.json",
        },
        "tool_policy": {"deny_tools": ["web_search", "code_execution"], "deny_argument_patterns": [r"https?://"]},
        "tool_policy_ok": True,
        "tool_policy_violations": [],
    }
    task_dir = tmp_path / "task"
    sidecar_path = task_dir / "openclaw_attempts" / "attempt-1" / "agentic_math" / "task_1" / "openclaw_result.json"

    assert module.sidecar_matches_cache(sidecar, cache_key)
    metadata = module.attempt_metadata_from_sidecar_path(task_dir, sidecar_path)
    assert metadata["openclaw_attempt_id"] == "attempt-1"
    assert metadata["openclaw_attempt_output_dir"].endswith("openclaw_attempts/attempt-1")


def test_success_sidecar_recovery_requires_invocation_cache_key_match(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    cache_key = {
        "runner_version": module.RUNNER_VERSION,
        "task_id": "task_1",
        "prompt_hash": "prompt-hash",
        "model": "provider/model",
        "thinking": "high",
        "answer_format": "math_expression",
        "deny_tools": ["web_search"],
        "deny_argument_patterns": [r"https?://"],
        "openclaw_config_source": "/sandbox/.openclaw/openclaw.json",
    }
    sidecar_path = (
        tmp_path
        / "task"
        / "openclaw_attempts"
        / "attempt-1"
        / "agentic_math"
        / "task_1"
        / "openclaw_result.json"
    )
    sidecar_path.parent.mkdir(parents=True)
    invocation_dir = tmp_path / "task" / "openclaw_invocations"
    invocation_dir.mkdir(parents=True)
    invocation_path = invocation_dir / "attempt-1.json"
    invocation_path.write_text(
        json.dumps({"cache_key": {**cache_key, "runner_version": "old-runner"}}),
        encoding="utf-8",
    )

    assert not module.sidecar_invocation_matches_cache(tmp_path / "task", sidecar_path, cache_key)

    invocation_path.write_text(json.dumps({"cache_key": cache_key}), encoding="utf-8")
    assert module.sidecar_invocation_matches_cache(tmp_path / "task", sidecar_path, cache_key)


def test_success_sidecar_recovery_rejects_config_source_mismatch():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    cache_key = {
        "task_id": "task_1",
        "prompt_hash": "prompt-hash",
        "model": "provider/model",
        "deny_tools": ["web_search"],
        "deny_argument_patterns": [r"https?://"],
        "openclaw_config_source": "/sandbox/.openclaw/openclaw.json",
    }
    sidecar = {
        "returncode": 0,
        "metadata": {
            "task_id": "task_1",
            "prompt_hash": "prompt-hash",
            "model_id": "provider/model",
            "openclaw_config_source": "/sandbox/other-openclaw.json",
        },
        "tool_policy": {"deny_tools": ["web_search"], "deny_argument_patterns": [r"https?://"]},
        "tool_policy_ok": True,
        "tool_policy_violations": [],
    }

    assert not module.sidecar_matches_cache(sidecar, cache_key)


def test_success_sidecar_recovery_requires_nemoclaw_session_audit_when_cache_is_nemoclaw():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    cache_key = {
        "task_id": "task_1",
        "prompt_hash": "prompt-hash",
        "model": "provider/model",
        "deny_tools": ["web_search"],
        "deny_argument_patterns": [r"https?://"],
        "openclaw_config_source": "/sandbox/.openclaw/openclaw.json",
        "agent_runtime": "nemoclaw",
        "nemoclaw_sandbox": "nejumi-taiwan",
    }
    sidecar = {
        "returncode": 0,
        "metadata": {
            "task_id": "task_1",
            "prompt_hash": "prompt-hash",
            "model_id": "provider/model",
            "openclaw_config_source": "/sandbox/.openclaw/openclaw.json",
        },
        "tool_policy": {"deny_tools": ["web_search"], "deny_argument_patterns": [r"https?://"]},
        "tool_policy_ok": True,
        "tool_policy_violations": [],
        "conversation_order": {"ok": True, "checked": True},
    }

    assert not module.sidecar_matches_cache(sidecar, cache_key)
    sidecar["nemoclaw_session_audit"] = {"required": True, "ok": False}
    assert not module.sidecar_matches_cache(sidecar, cache_key)
    sidecar["nemoclaw_session_audit"] = {"required": True, "ok": True}
    assert module.sidecar_matches_cache(sidecar, cache_key)


def test_success_sidecar_recovery_rejects_relogged_cache_mismatch(tmp_path, monkeypatch):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    cache_key = {
        "task_id": "task_1",
        "prompt_hash": "prompt-hash",
        "model": "provider/model",
        "deny_tools": ["web_search"],
        "deny_argument_patterns": [r"https?://"],
        "openclaw_config_source": "/sandbox/.openclaw/openclaw.json",
    }
    sidecar = {
        "returncode": 0,
        "stdout_json": {"finalAssistantVisibleText": "ANSWER: \\boxed{2}"},
        "metadata": {
            "task_id": "task_1",
            "prompt_hash": "prompt-hash",
            "model_id": "provider/model",
            "openclaw_config_source": "/sandbox/.openclaw/openclaw.json",
        },
        "tool_policy": {"deny_tools": ["web_search"], "deny_argument_patterns": [r"https?://"]},
        "tool_policy_ok": True,
        "tool_policy_violations": [],
        "conversation_order": {"ok": True},
    }
    task_dir = tmp_path / "task"
    sidecar_path = task_dir / "openclaw_attempts" / "attempt-1" / "agentic_math" / "task_1" / "openclaw_result.json"
    sidecar_path.parent.mkdir(parents=True)
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    def fake_relog_existing_sidecar(_sidecar_path, _args):
        relogged = dict(sidecar)
        relogged["metadata"] = dict(sidecar["metadata"], prompt_hash="other-prompt-hash")
        relogged["weave_sidecar"] = {"ok": True}
        return relogged

    monkeypatch.setattr(module, "relog_existing_sidecar", fake_relog_existing_sidecar)
    args = type(
        "Args",
        (),
        {
            "weave_sidecar": True,
            "weave_sidecar_strict": True,
        },
    )()

    recovered = module.recover_existing_success_sidecar(
        {"task_id": "task_1", "answer": "2"},
        task_dir,
        cache_key,
        args,
    )

    assert recovered is None


def test_scored_record_preserves_nemoclaw_session_audit(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    sidecar_path = tmp_path / "openclaw_result.json"
    sidecar = {
        "returncode": 0,
        "stdout_json": {"finalAssistantVisibleText": "ANSWER: \\boxed{2}"},
        "tool_policy_ok": True,
        "tool_policy_violations": [],
        "conversation_order": {"ok": True},
        "nemoclaw_session_audit": {
            "required": True,
            "ok": True,
            "copied_session_file": "outputs/task/nemoclaw_session.jsonl",
        },
    }

    record = module.build_scored_record_from_sidecar(
        {"task_id": "task_1", "answer": "2"},
        sidecar_path,
        sidecar,
        {"prompt_hash": "prompt-hash"},
        {"openclaw_attempt_number": 1},
    )

    assert record["correct"] is True
    assert record["nemoclaw_session_audit_ok"] is True
    assert record["nemoclaw_session_audit"]["required"] is True
    assert record["nemoclaw_session_audit"]["copied_session_file"].endswith("nemoclaw_session.jsonl")


def test_scored_record_credits_answer_with_conversation_order_warning(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    sidecar_path = tmp_path / "openclaw_result.json"
    sidecar = {
        "returncode": 0,
        "stdout_json": {"finalAssistantVisibleText": "ANSWER: \\boxed{2}"},
        "conversation_order": {
            "ok": True,
            "issues": [],
            "warnings": [{"type": "tool_after_final_answer"}],
        },
    }

    record = module.build_scored_record_from_sidecar(
        {"task_id": "task_1", "answer": "2"},
        sidecar_path,
        sidecar,
        {"prompt_hash": "prompt-hash"},
        {"openclaw_attempt_number": 1},
    )

    assert record["predicted_answer"] == "2"
    assert record["correct"] is True
    assert record["conversation_order_ok"] is True
    assert record["openclaw_disqualified_reason"] == ""


def test_weave_sidecar_failure_is_observability_failure_not_model_error():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    assert module.is_weave_sidecar_failure({"returncode": 0, "weave_sidecar": {"ok": False}})
    assert not module.is_weave_sidecar_failure({"returncode": 1, "weave_sidecar": {"ok": False}})


def test_write_summary_counts_tool_usage(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    args = type(
        "Args",
        (),
        {
            "model": "model",
            "thinking": "high",
            "dry_run": False,
            "weave_sidecar": False,
            "weave_sidecar_strict": True,
        },
    )()
    results = [
        {
            "task_id": "a",
            "subject": "algebra",
            "correct": True,
            "predicted_answer": "1",
            "openclaw_tool_call_count": 2,
            "nemoclaw_session_audit_ok": True,
            "nemoclaw_session_audit": {"required": True, "ok": True},
        },
        {
            "task_id": "b",
            "subject": "algebra",
            "correct": False,
            "predicted_answer": None,
            "openclaw_tool_call_count": 0,
            "openclaw_tool_error_count": 1,
            "tool_policy_violations": [{"type": "denied_tool"}],
            "nemoclaw_session_audit_ok": False,
            "nemoclaw_session_audit": {"required": True, "ok": False},
        },
    ]

    summary = module.write_summary(tmp_path, results, args)

    assert summary["runner_version"] == module.RUNNER_VERSION
    assert summary["weave_sidecar"] is False
    assert summary["weave_sidecar_strict"] is True
    assert summary["tool_called_instances"] == 1
    assert summary["tool_error_instances"] == 1
    assert summary["tool_policy_violation_instances"] == 1
    assert summary["nemoclaw_session_audit_required_instances"] == 2
    assert summary["nemoclaw_session_audit_passed_instances"] == 1
    assert summary["nemoclaw_session_audit_failed_instances"] == 1


def test_build_prompt_includes_python_tool_guidance():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    prompt = module.build_prompt(
        {
            "task_id": "task_1",
            "benchmark": "OlymMATH-HARD",
            "answer_format": "math_expression",
            "source_config": "zh-hard",
            "problem_index": 1,
            "subject": "代數",
            "question": "求 1+1.",
        },
        max_tool_wall_seconds=45,
    )

    assert "## Python Tool Guidance" in prompt
    assert "sympy" in prompt
    assert "Do not rely on unbounded brute force" in prompt
    assert "wall-clock limit of 45 seconds" in prompt
    assert "Do not run broad random searches" in prompt
    assert "If a computation times out" in prompt
    assert "local shell/Python execution" in prompt
    assert "Do not start an interactive shell" in prompt
    assert "Never call `exec` with `pty=true`" in prompt
    assert "python3 - <<" in prompt
    assert "Do not use web search" in prompt
    assert "call the tool before writing any `ANSWER:` line" in prompt
    assert "Do not write provisional" in prompt


def test_task_openclaw_config_includes_tool_deny_policy(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    template = tmp_path / "openclaw.json"
    template.write_text('{"agents": {"list": []}, "tools": {"profile": "coding"}}\n', encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "use_task_agent": True,
            "no_local": False,
            "openclaw_config_template": template,
            "task_agent_prefix": "tw-math",
            "openclaw_tool_profile": "coding",
            "deny_tool": ["web_search"],
            "max_tool_wall_seconds": 45,
        },
    )()
    task_dir = tmp_path / "task"
    task_dir.mkdir()

    agent_id, config_path = module.write_task_openclaw_config(
        {"task_id": "task_1"},
        tmp_path / "workspace",
        task_dir,
        args,
    )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    [agent_entry] = [entry for entry in config["agents"]["list"] if entry["id"] == agent_id]
    assert agent_entry["tools"]["profile"] == "coding"
    assert agent_entry["tools"]["deny"] == ["web_search"]
    assert config["tools"]["exec"]["timeoutSec"] == 45
    assert config["tools"]["toolSearch"] is False
    assert config["tools"]["web"]["fetch"]["enabled"] is False
    assert "browser" not in config["tools"]


def test_task_openclaw_config_supports_nemoclaw_task_workspace(tmp_path, monkeypatch):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    template = tmp_path / "openclaw.json"
    template.write_text(
        json.dumps(
            {
                "agents": {"list": []},
                "plugins": {"entries": {"weave": {"enabled": True}}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    writes = {}

    def fake_write(args, path, text):
        writes["path"] = path
        writes["config"] = json.loads(text)

    monkeypatch.setattr(module, "write_nemoclaw_text_file", fake_write)
    args = type(
        "Args",
        (),
        {
            "agent": "main",
            "model": "openai-direct/gpt-4.1-mini-2025-04-14",
            "max_input_tokens": 12345,
            "max_agent_turns": 8,
            "max_tool_wall_seconds": 90,
            "use_task_agent": True,
            "no_local": False,
            "nemoclaw_sandbox": "nejumi-taiwan",
            "nemoclaw_bin": "nemoclaw",
            "nemoclaw_workdir": "/sandbox/tasks",
            "nemoclaw_openclaw_config_path": "/sandbox/.openclaw/openclaw.json",
            "openclaw_config_template": template,
            "task_agent_prefix": "tw-math",
            "openclaw_tool_profile": "coding",
            "deny_tool": ["web_search"],
            "deny_argument_pattern": None,
        },
    )()
    task_dir = tmp_path / "task"
    task_dir.mkdir()

    agent_id, config_path = module.write_task_openclaw_config(
        {"task_id": "math/task 1"},
        tmp_path / "workspace",
        task_dir,
        args,
    )

    assert str(config_path) == writes["path"]
    assert str(config_path).startswith("/sandbox/tasks/agentic_math/task-")
    [agent_entry] = [entry for entry in writes["config"]["agents"]["list"] if entry["id"] == agent_id]
    assert agent_entry["workspace"].startswith("/sandbox/tasks/agentic_math/task-")
    assert agent_entry["agentDir"].endswith("/openclaw_agent_state")
    assert agent_entry["contextTokens"] == 12345
    assert writes["config"]["tools"]["exec"]["timeoutSec"] == 90
    assert agent_entry["runRetries"] == {
        "base": 8,
        "perProfile": 0,
        "min": 8,
        "max": 8,
    }
    assert writes["config"]["tools"]["toolSearch"] is False
    assert writes["config"]["tools"]["web"]["fetch"]["enabled"] is False
    assert "browser" not in writes["config"]["tools"]
    metadata = json.loads((task_dir / "openclaw_task_agent.json").read_text(encoding="utf-8"))
    assert metadata["nemoclaw_sandbox"] == "nejumi-taiwan"
    assert metadata["config_path"] == str(config_path)
    assert metadata["context_cap"]["contextTokens"] == 12345
    assert metadata["exec_timeout"] == {"timeoutSec": 90}
    assert metadata["run_retries"] == {
        "base": 8,
        "perProfile": 0,
        "min": 8,
        "max": 8,
    }


def test_task_openclaw_config_registers_nemoclaw_gateway_agent_when_no_local(tmp_path, monkeypatch):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    calls = []

    def fake_run_nemoclaw_text_command(args, command, input_text=None, timeout=60, check=True):
        calls.append(
            {
                "command": command,
                "input_text": input_text,
                "timeout": timeout,
                "check": check,
            }
        )
        return subprocess.CompletedProcess(command, 0, stdout='{"ok": true}\n', stderr="")

    monkeypatch.setattr(module, "run_nemoclaw_text_command", fake_run_nemoclaw_text_command)
    args = type(
        "Args",
        (),
        {
            "agent": "main",
            "dry_run": False,
            "model": "openai-direct/gpt-4.1-mini-2025-04-14",
            "max_input_tokens": 50000,
            "max_tool_calls": 40,
            "max_agent_turns": 40,
            "max_tool_wall_seconds": 75,
            "use_task_agent": True,
            "no_local": True,
            "nemoclaw_sandbox": "nejumi-taiwan",
            "nemoclaw_bin": "nemoclaw",
            "nemoclaw_workdir": "/sandbox/tasks",
            "nemoclaw_openclaw_config_path": "/sandbox/.openclaw/openclaw.json",
            "openclaw_config_template": None,
            "task_agent_prefix": "tw-math",
            "openclaw_tool_profile": "coding",
            "deny_tool": ["web_search"],
            "deny_argument_pattern": None,
        },
    )()
    task_dir = tmp_path / "task"
    task_dir.mkdir()

    agent_id, config_path = module.write_task_openclaw_config(
        {"task_id": "math/task 1"},
        tmp_path / "workspace",
        task_dir,
        args,
    )

    assert config_path is None
    assert calls
    command = calls[0]["command"]
    assert command[0] == "env"
    script_b64 = command[1].split("=", 1)[1]
    script = module.base64.b64decode(script_b64).decode("utf-8")
    assert "openclaw agents add" in script
    assert 'entry["contextTokens"] = context_cap["contextTokens"]' in script
    assert 'entry["runRetries"] = turn_run_retries' in script
    assert 'exec_config["timeoutSec"] = max_tool_wall_seconds' in script
    assert "min(existing_timeout, max_tool_wall_seconds)" not in script
    assert 'target["contextTokens"]' not in script
    assert command[2:5] == [
        "bash",
        "-lc",
        'printf %s "$OPENCLAW_REGISTER_SCRIPT_B64" | base64 -d | bash -s -- "$@"',
    ]
    assert command[5] == "register-task-agent"
    assert command[6] == agent_id
    assert command[9] == "openai-direct/gpt-4.1-mini-2025-04-14"
    budget = json.loads(command[13])
    assert budget["max_input_tokens"] == 50000
    assert budget["max_tool_wall_seconds"] == 75
    assert budget["max_tool_calls"] == 40
    assert budget["max_agent_turns"] == 40
    assert budget["max_cumulative_input_tokens"] == 50000
    assert budget["max_cumulative_output_tokens"] == 0
    assert budget["require_actual_token_usage"] is False
    metadata = json.loads((task_dir / "openclaw_task_agent.json").read_text(encoding="utf-8"))
    assert metadata["config_path"] == "/sandbox/.openclaw/openclaw.json"
    assert metadata["exec_timeout"] == {"timeoutSec": 75}
    assert metadata["gateway_registered"]["ok"] is True
    assert (
        module.task_live_sandbox_session_dir({"task_id": "math/task 1"}, args, agent_id)
        == f"/sandbox/.openclaw/agents/{agent_id}/sessions"
    )
    module._REGISTERED_NEMOCLAW_GATEWAY_AGENTS.clear()


def test_configure_openclaw_exec_timeout_overrides_short_template_default():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    config = {"tools": {"exec": {"timeoutSec": 10}}}
    args = type("Args", (), {"max_tool_wall_seconds": 120})()

    metadata = module.configure_openclaw_exec_timeout(config, args)

    assert metadata == {"timeoutSec": 120}
    assert config["tools"]["exec"]["timeoutSec"] == 120


def test_task_python_sitecustomize_is_written_for_nemoclaw(tmp_path, monkeypatch):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    writes = []

    def fake_write(args, path, text):
        writes.append({"path": path, "text": text})

    monkeypatch.setattr(module, "write_nemoclaw_text_file", fake_write)
    args = type(
        "Args",
        (),
        {
            "nemoclaw_sandbox": "nejumi-taiwan",
            "nemoclaw_workdir": "/sandbox/tasks",
        },
    )()

    module.write_task_python_sitecustomize(
        {"task_id": "math/task 1"},
        tmp_path / "workspace",
        args,
        "tw-math-task-1",
    )

    paths = [write["path"] for write in writes]
    assert "/sandbox/sitecustomize.py" in paths
    assert any(path.endswith("/sitecustomize.py") and path != "/sandbox/sitecustomize.py" for path in paths)
    assert all("/tmp/.local/lib" in write["text"] for write in writes)
    assert all("sys.path.insert" in write["text"] for write in writes)


def test_nemoclaw_gateway_cleanup_uses_short_bounded_timeouts(monkeypatch, capsys):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    calls = []

    def fake_unregister(args, agent_id, *, timeout=60):
        calls.append((agent_id, timeout))
        return {"ok": True, "agent_id": agent_id}

    monkeypatch.setattr(module, "unregister_nemoclaw_gateway_task_agent", fake_unregister)
    module._REGISTERED_NEMOCLAW_GATEWAY_AGENTS[:] = [(SimpleNamespace(), "agent-a")]

    module.cleanup_registered_nemoclaw_gateway_agents()

    assert calls == [("agent-a", module.NEMOCLAW_GATEWAY_CLEANUP_PER_AGENT_TIMEOUT_SEC)]
    assert module._REGISTERED_NEMOCLAW_GATEWAY_AGENTS == []
    assert capsys.readouterr().err == ""


def test_nemoclaw_gateway_cleanup_skips_remaining_after_total_budget(monkeypatch, capsys):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    calls = []

    def fake_unregister(args, agent_id, *, timeout=60):
        calls.append((agent_id, timeout))
        return {"ok": True, "agent_id": agent_id}

    monkeypatch.setattr(module, "unregister_nemoclaw_gateway_task_agent", fake_unregister)
    monkeypatch.setattr(module, "NEMOCLAW_GATEWAY_CLEANUP_TOTAL_TIMEOUT_SEC", 0.0)
    module._REGISTERED_NEMOCLAW_GATEWAY_AGENTS[:] = [
        (SimpleNamespace(), "agent-a"),
        (SimpleNamespace(), "agent-b"),
    ]

    module.cleanup_registered_nemoclaw_gateway_agents()

    assert calls == []
    assert module._REGISTERED_NEMOCLAW_GATEWAY_AGENTS == []
    assert "skipped 2 NeMoClaw task-agent cleanup calls" in capsys.readouterr().err


def test_task_python_sitecustomize_is_written_for_local_workspace(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    args = type("Args", (), {"nemoclaw_sandbox": None})()
    workspace = tmp_path / "workspace"

    module.write_task_python_sitecustomize({"task_id": "task_1"}, workspace, args, "agent")

    text = (workspace / "sitecustomize.py").read_text(encoding="utf-8")
    assert "/tmp/.local/lib" in text
    assert "sys.path.insert" in text


def test_task_live_session_dir_uses_local_task_agent_state(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    task_dir = tmp_path / "task"
    row = {"task_id": "math_1"}

    local_args = type(
        "Args",
        (),
        {
            "use_task_agent": True,
            "nemoclaw_sandbox": None,
        },
    )()
    nemoclaw_args = type(
        "Args",
        (),
        {
            "use_task_agent": True,
            "nemoclaw_sandbox": "nejumi-taiwan",
        },
    )()

    assert module.task_live_session_dir(task_dir, local_args) == (
        task_dir / "openclaw_agent_state" / "sessions"
    )
    assert module.task_live_session_dir(task_dir, nemoclaw_args) is None
    safe_task = module.safe_agent_id("math_1", "task")
    assert (
        module.task_live_sandbox_session_dir(row, nemoclaw_args, "tw-math")
        == f"/sandbox/agentic_math/{safe_task}/openclaw_agent_state/sessions"
    )


def test_agentic_math_nemoclaw_forwards_live_sandbox_session_dir(tmp_path, monkeypatch):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    row = {
        "task_id": "math_1",
        "answer": "2",
        "question": "1+1?",
        "subject": "algebra",
        "answer_format": "math_expression",
    }
    captured_command = []

    monkeypatch.setattr(
        module,
        "read_openclaw_config_template",
        lambda args: ({"agents": {"list": []}}, "/sandbox/.openclaw/openclaw.json"),
    )
    monkeypatch.setattr(module, "write_nemoclaw_text_file", lambda args, path, text: None)

    def fake_run_command(command, cwd=None, timeout=None, check=True):
        captured_command[:] = command
        output_dir = Path(command[command.index("--output-dir") + 1])
        prompt_text = Path(command[command.index("--prompt-file") + 1]).read_text(encoding="utf-8")
        sidecar_path = module.task_sidecar_path(output_dir, row["task_id"])
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(
            json.dumps(
                {
                    "returncode": 0,
                    "metadata": {
                        "task_id": row["task_id"],
                        "prompt_hash": module.sha256_text(prompt_text),
                        "model_id": command[command.index("--model") + 1],
                        "openclaw_config_source": command[command.index("--openclaw-config-source") + 1],
                    },
                    "tool_policy": {
                        "deny_tools": [
                            command[index + 1]
                            for index, token in enumerate(command[:-1])
                            if token == "--deny-tool"
                        ],
                        "deny_argument_patterns": [
                            command[index + 1]
                            for index, token in enumerate(command[:-1])
                            if token == "--deny-argument-pattern"
                        ],
                    },
                    "stdout": "ANSWER: 2\n",
                    "tool_policy_ok": True,
                    "tool_policy_violations": [],
                    "tool_call_count": 1,
                    "tool_error_count": 0,
                    "runtime_budget": {"ok": True},
                    "conversation_order": {"ok": True, "checked": True},
                    "nemoclaw_session_audit": {"required": True, "ok": True},
                    "weave_sidecar": {"ok": True},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout=str(sidecar_path), stderr="")

    monkeypatch.setattr(module, "run_command", fake_run_command)
    args = type(
        "Args",
        (),
        {
            "agent": "main",
            "allow_failed_preflight": False,
            "deny_argument_pattern": None,
            "deny_tool": None,
            "dry_run": False,
            "fail_fast": False,
            "model": "openai-direct/example-model",
            "nemoclaw_bin": "nemoclaw",
            "nemoclaw_openclaw_config_path": "/sandbox/.openclaw/openclaw.json",
            "nemoclaw_sandbox": "nejumi-taiwan",
            "nemoclaw_workdir": "/sandbox/tasks",
            "no_local": False,
            "openclaw_max_attempts": 1,
            "openclaw_retry_base_seconds": 0,
            "openclaw_timeout": 30,
            "openclaw_tool_profile": "coding",
            "profile": None,
            "redo": False,
            "session_prefix": None,
            "task_agent_prefix": "tw-math",
            "thinking": "high",
            "use_task_agent": True,
            "max_input_tokens": 500_000,
            "max_tool_calls": 60,
            "weave_sidecar": False,
            "weave_sidecar_strict": False,
        },
    )()

    record = module.run_openclaw_for_task(row, tmp_path / "task", args)

    assert record["correct"] is True
    assert captured_command[captured_command.index("--nemoclaw-sandbox") + 1] == "nejumi-taiwan"
    live_index = captured_command.index("--live-sandbox-session-dir")
    safe_task = module.safe_agent_id("math_1", "task")
    assert (
        captured_command[live_index + 1]
        == f"/sandbox/tasks/agentic_math/{safe_task}/openclaw_agent_state/sessions"
    )


def test_main_dry_run_uses_protocol_path(tmp_path, monkeypatch):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    calls = []
    output_dir = tmp_path / "out"
    row = {
        "task_id": "math_1",
        "answer": "2",
        "question": "1+1?",
        "subject": "algebra",
        "answer_format": "math_expression",
    }

    args = type(
        "Args",
        (),
        {
            "dataset_jsonl": tmp_path / "dataset.jsonl",
            "output_dir": output_dir,
            "limit": None,
            "dry_run": True,
            "model": "inference/example",
            "thinking": "high",
            "weave_sidecar": False,
            "weave_sidecar_strict": False,
        },
    )()

    def fake_run_openclaw_for_task(task_row, task_dir, parsed_args):
        calls.append((task_row, task_dir, parsed_args.dry_run))
        return {
            **task_row,
            "response": "",
            "predicted_answer": None,
            "gold_answer": "2",
            "correct": False,
            "dry_run": True,
        }

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "read_jsonl", lambda path: [row])
    monkeypatch.setattr(module, "run_openclaw_for_task", fake_run_openclaw_for_task)

    module.main()

    assert calls == [(row, output_dir / "math_1", True)]
    assert (output_dir / "results.jsonl").exists()


def test_evaluator_as_list_expands_omegaconf_listconfig():
    from omegaconf import OmegaConf

    agentic_module = load_script_module(REPO_ROOT / "scripts" / "evaluator" / "agentic_math.py")
    swe_module = load_script_module(REPO_ROOT / "scripts" / "evaluator" / "swebench_pro.py")
    value = OmegaConf.create(["web_search", "browser_*"])

    assert agentic_module._as_list(value) == ["web_search", "browser_*"]
    assert swe_module._as_list(value) == ["web_search", "browser_*"]


def test_evaluator_passes_nemoclaw_args_to_agentic_math_runner(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    agentic_module = load_script_module(REPO_ROOT / "scripts" / "evaluator" / "agentic_math.py")
    commands = []

    def fake_run_command(command):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(agentic_module, "_run_command", fake_run_command)
    cfg = OmegaConf.create(
        {
            "testmode": False,
            "model": {"pretrained_model_name_or_path": "provider/model"},
            "agentic_math": {
                "prefix": "tw-math",
                "thinking": "high",
                "agent": "main",
                "openclaw_timeout": 60,
                "openclaw_max_attempts": 1,
                "openclaw_retry_base_seconds": 1,
                "max_input_tokens": 222222,
                "max_tool_calls": 9,
                "max_agent_turns": 10,
                "use_task_agent": True,
                "nemoclaw_sandbox": "nejumi-taiwan",
                "nemoclaw_bin": "/usr/local/bin/nemoclaw",
                "nemoclaw_workdir": "/sandbox/work",
                "nemoclaw_openclaw_config_path": "/sandbox/.openclaw/openclaw.json",
                "session_prefix": "{wandb_run_id}:agentic-math",
                "weave_sidecar": False,
            },
        }
    )

    agentic_module._run_openclaw(cfg, tmp_path / "dataset.jsonl", tmp_path / "outputs")

    [command] = commands
    assert "--no-use-task-agent" not in command
    assert command[command.index("--nemoclaw-sandbox") + 1] == "nejumi-taiwan"
    assert command[command.index("--nemoclaw-bin") + 1] == "/usr/local/bin/nemoclaw"
    assert command[command.index("--nemoclaw-workdir") + 1] == "/sandbox/work"
    assert command[command.index("--nemoclaw-openclaw-config-path") + 1] == "/sandbox/.openclaw/openclaw.json"
    assert command[command.index("--session-prefix") + 1] == "{wandb_run_id}:agentic-math"
    assert command[command.index("--max-input-tokens") + 1] == "222222"
    assert command[command.index("--max-tool-calls") + 1] == "9"
    assert command[command.index("--max-agent-turns") + 1] == "10"


def test_evaluator_passes_no_use_task_agent_when_explicitly_disabled(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    agentic_module = load_script_module(REPO_ROOT / "scripts" / "evaluator" / "agentic_math.py")
    commands = []

    def fake_run_command(command):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(agentic_module, "_run_command", fake_run_command)
    cfg = OmegaConf.create(
        {
            "testmode": False,
            "model": {"pretrained_model_name_or_path": "provider/model"},
            "agentic_math": {
                "use_task_agent": False,
                "nemoclaw_sandbox": "nejumi-taiwan",
            },
        }
    )

    agentic_module._run_openclaw(cfg, tmp_path / "dataset.jsonl", tmp_path / "outputs")

    assert "--no-use-task-agent" in commands[0]
    assert (
        commands[0][commands[0].index("--nemoclaw-openclaw-config-path") + 1]
        == "/sandbox/.openclaw/openclaw.json"
    )


def test_prepare_agentic_math_normalizes_unit_answer():
    module = load_module(REPO_ROOT / "scripts" / "data_uploader" / "prepare_agentic_math.py")
    assert module.normalize_answer("336^\\circ") == ("336", 336)


def test_prepare_olymmath_task_id_is_stable():
    module = load_module(REPO_ROOT / "scripts" / "data_uploader" / "prepare_olymmath_zh_tw.py")
    assert module.to_task_id("OlymMATH-HARD-24-ZH") == "olymmath_hard_24_zh"
