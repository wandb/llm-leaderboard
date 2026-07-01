import json
import importlib.util
import subprocess
import sys
from pathlib import Path

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


def test_transient_openclaw_failure_detects_provider_timeout():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_agentic_math_openclaw.py")
    completed = subprocess.CompletedProcess(
        ["cmd"],
        1,
        stdout="",
        stderr="FailoverError: LLM request timed out. rawError=terminated",
    )

    assert module.is_transient_openclaw_failure(completed, None)


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
    assert module.non_scoreable_openclaw_failure_reason(order, None) == "conversation_order_violation"

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
        sidecar_path = module.task_sidecar_path(output_dir, row["task_id"])
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(
            json.dumps(
                {
                    "returncode": 0,
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
    assert session_key.startswith("twcanary-run-3:agentic-math:math_1:")
    assert record["correct"] is True
    assert record["cache_key"]["session_prefix"] == "twcanary-run-3:agentic-math"


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
    }
    sidecar = {
        "returncode": 0,
        "metadata": {"task_id": "task_1", "prompt_hash": "prompt-hash", "model_id": "provider/model"},
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
        }
    )

    assert "## Python Tool Guidance" in prompt
    assert "sympy" in prompt
    assert "Do not rely on unbounded brute force" in prompt
    assert "local shell/Python execution" in prompt
    assert "Do not use web search" in prompt


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
    assert config["tools"]["toolSearch"] is False
    assert config["tools"]["web"]["fetch"]["enabled"] is False


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
    assert writes["config"]["tools"]["toolSearch"] is False
    assert writes["config"]["tools"]["web"]["fetch"]["enabled"] is False
    metadata = json.loads((task_dir / "openclaw_task_agent.json").read_text(encoding="utf-8"))
    assert metadata["nemoclaw_sandbox"] == "nejumi-taiwan"
    assert metadata["config_path"] == str(config_path)


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


def test_prepare_agentic_math_normalizes_unit_answer():
    module = load_module(REPO_ROOT / "scripts" / "data_uploader" / "prepare_agentic_math.py")
    assert module.normalize_answer("336^\\circ") == ("336", 336)


def test_prepare_olymmath_task_id_is_stable():
    module = load_module(REPO_ROOT / "scripts" / "data_uploader" / "prepare_olymmath_zh_tw.py")
    assert module.to_task_id("OlymMATH-HARD-24-ZH") == "olymmath_hard_24_zh"
