import argparse
import importlib.util
import json
import subprocess
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "tools" / "run_deepswe_openclaw.py"
spec = importlib.util.spec_from_file_location("run_deepswe_openclaw", MODULE_PATH)
assert spec is not None and spec.loader is not None
run_deepswe_openclaw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_deepswe_openclaw)


def load_deepswe_evaluator():
    path = ROOT / "scripts" / "evaluator" / "deepswe.py"
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        evaluator_spec = importlib.util.spec_from_file_location("deepswe_evaluator", path)
        assert evaluator_spec is not None and evaluator_spec.loader is not None
        module = importlib.util.module_from_spec(evaluator_spec)
        evaluator_spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def _deepswe_evaluator_cfg(*, no_local: bool, nemoclaw_sandbox):
    return OmegaConf.create(
        {
            "model": {"pretrained_model_name_or_path": "test-model"},
            "deepswe": {
                "no_local": no_local,
                "nemoclaw_sandbox": nemoclaw_sandbox,
                "verify_weave_agents": False,
                "disable_verification": True,
                "delete": False,
            },
        }
    )


def test_deepswe_evaluator_rejects_no_local_without_named_sandbox(tmp_path):
    evaluator = load_deepswe_evaluator()
    cfg = _deepswe_evaluator_cfg(no_local=True, nemoclaw_sandbox=None)

    with pytest.raises(ValueError, match="must name an isolated NeMoClaw sandbox"):
        evaluator._run_openclaw(cfg, tmp_path / "tasks.json", tmp_path / "output")


def test_deepswe_evaluator_local_mode_omits_absent_nemoclaw_sandbox(
    monkeypatch,
    tmp_path,
):
    evaluator = load_deepswe_evaluator()
    cfg = _deepswe_evaluator_cfg(no_local=False, nemoclaw_sandbox=None)
    commands = []
    def fake_run_command(command, **kwargs):
        commands.append(command)
        assert kwargs["timeout"] == 129_600

    monkeypatch.setattr(evaluator, "_run_command", fake_run_command)

    runner_dir = evaluator._run_openclaw(
        cfg,
        tmp_path / "tasks.json",
        tmp_path / "output",
    )

    assert runner_dir == tmp_path / "output" / "runner"
    assert len(commands) == 1
    assert "--no-no-local" in commands[0]
    assert "--nemoclaw-sandbox" not in commands[0]
    assert "None" not in commands[0]


def _args(tmp_path: Path, task_names_file: Path) -> argparse.Namespace:
    return argparse.Namespace(
        tasks_root=tmp_path / "tasks",
        output_dir=tmp_path / "out",
        jobs_dir=tmp_path / "jobs",
        task_names_file=task_names_file,
        include_task_name=None,
        n_tasks=16,
        sample_seed=0,
        n_concurrent=2,
        model="openrouter-direct/z-ai/glm-5.2",
        thinking="high",
        agent="main",
        prefix="deepswe-test",
        openclaw_timeout=3600,
        openclaw_max_attempts=1,
        openclaw_retry_base_seconds=15.0,
        max_input_tokens=1_000_000,
        max_cumulative_input_tokens=1_000_000,
        max_cumulative_output_tokens=500_000,
        max_tool_calls=40,
        max_agent_turns=40,
        max_tool_wall_seconds=300,
        final_assistant_idle_salvage_seconds=60.0,
        final_assistant_shutdown_grace_seconds=30.0,
        llm_response_idle_timeout_seconds=900.0,
        require_actual_token_usage=True,
        nemoclaw_bin="nemoclaw",
        nemoclaw_sandbox="nejumi-taiwan",
        nemoclaw_workdir="/sandbox",
        nemoclaw_openclaw_config_path="/sandbox/.openclaw/openclaw.json",
        nemoclaw_checkout_transfer_mode="copy",
        nemoclaw_checkout_transfer_timeout=600,
        openclaw_tool_profile="coding",
        task_agent_prefix="tw-deepswe-test",
        session_prefix="deepswe-test",
        no_local=True,
        use_task_agent=True,
        restart_gateway_before_run=True,
        verify_weave_agents=True,
        weave_agents_entity="llm-leaderboard",
        weave_agents_project="tc-leaderboard",
        weave_agents_agent_name="nejumi-taiwan-openclaw",
        weave_agents_limit=50,
        weave_agents_verification_timeout=120.0,
        weave_agents_poll_seconds=5.0,
        deny_tool=["web_search", "browser"],
        deny_argument_pattern=[r"https?://"],
        allow_failed_preflight=False,
        preflight_docker_images=True,
        preflight_openclaw_sandbox_tools=True,
        prepull_missing_docker_images=True,
        docker_pull_retries=3,
        docker_pull_retry_seconds=30.0,
        fail_fast_environment_setup=True,
        fail_fast_non_scoreable_policy=True,
        fail_fast_trace_evidence=True,
        environment_failfast_poll_seconds=2.0,
        agent_timeout_multiplier=None,
        disable_verification=False,
        delete=True,
        quiet=False,
    )


def _load_deepswe_pier_agent_module(monkeypatch):
    pier_module = types.ModuleType("pier")
    agents_module = types.ModuleType("pier.agents")
    agents_base_module = types.ModuleType("pier.agents.base")
    environments_module = types.ModuleType("pier.environments")
    environments_base_module = types.ModuleType("pier.environments.base")
    models_module = types.ModuleType("pier.models")
    agent_models_module = types.ModuleType("pier.models.agent")
    context_module = types.ModuleType("pier.models.agent.context")

    class BaseAgent:
        def __init__(self, logs_dir, model_name=None, **kwargs):
            self.logs_dir = Path(logs_dir)
            self.model_name = model_name

    class BaseEnvironment:
        pass

    class AgentContext:
        pass

    agents_base_module.BaseAgent = BaseAgent
    environments_base_module.BaseEnvironment = BaseEnvironment
    context_module.AgentContext = AgentContext

    for name, module in {
        "pier": pier_module,
        "pier.agents": agents_module,
        "pier.agents.base": agents_base_module,
        "pier.environments": environments_module,
        "pier.environments.base": environments_base_module,
        "pier.models": models_module,
        "pier.models.agent": agent_models_module,
        "pier.models.agent.context": context_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_path = ROOT / "scripts" / "tools" / "deepswe_openclaw_pier_agent.py"
    spec = importlib.util.spec_from_file_location("deepswe_openclaw_pier_agent_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deepswe_completion_requires_explicit_submit_marker(monkeypatch):
    module = _load_deepswe_pier_agent_module(monkeypatch)

    assert module.DEEPSWE_SUBMIT_MARKER in module.DEEPSWE_COMPLETION_REQUIREMENTS
    assert "final shell action" in module.DEEPSWE_COMPLETION_REQUIREMENTS


def test_deepswe_time_up_preserves_existing_patch(monkeypatch):
    module = _load_deepswe_pier_agent_module(monkeypatch)

    assert module._deepswe_should_force_empty_patch(
        {"openclaw_disqualified_reason": "time_up"}
    ) is False


def test_deepswe_provider_failure_still_forces_empty_patch(monkeypatch):
    module = _load_deepswe_pier_agent_module(monkeypatch)

    assert module._deepswe_should_force_empty_patch(
        {"openclaw_disqualified_reason": "provider_transient_exhausted"}
    ) is True


def test_submission_trace_detects_marker_and_usage_only_reasoning(monkeypatch):
    module = _load_deepswe_pier_agent_module(monkeypatch)
    sidecar = {
        "timeline_events": [
            {"type": "assistant_message", "content": "Implemented and tested."},
            {
                "type": "tool_call",
                "arguments": {"command": f"echo {module.DEEPSWE_SUBMIT_MARKER}"},
            },
        ]
    }

    trace = module._submission_trace(
        sidecar,
        {"openclaw_usage": {"reasoningTokens": 42}},
    )

    assert trace["explicit_submit_marker_seen"] is True
    assert trace["final_assistant_text"] == "Implemented and tested."
    assert trace["final_assistant_character_count"] == len("Implemented and tested.")
    assert trace["reasoning_trace_present"] is False
    assert trace["reasoning_trace_source"] == "provider_usage_only"


def test_submission_trace_records_visible_reasoning(monkeypatch):
    module = _load_deepswe_pier_agent_module(monkeypatch)

    trace = module._submission_trace(
        {
            "timeline_events": [
                {"type": "assistant_reasoning", "content": "Inspect the failing path."},
                {"type": "assistant_message", "content": "Done."},
            ]
        },
        {"openclaw_usage": {}},
    )

    assert trace["reasoning_trace_present"] is True
    assert trace["reasoning_trace_source"] == "openclaw_session_jsonl"
    assert trace["reasoning_event_count"] == 1


def test_submission_evidence_accepts_applicable_explicit_patch(monkeypatch, tmp_path):
    module = _load_deepswe_pier_agent_module(monkeypatch)
    checkout = tmp_path / "checkout"
    task_dir = tmp_path / "task"
    checkout.mkdir()
    task_dir.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=checkout, check=True)
    (checkout / "file.txt").write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "add", "file.txt"], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=checkout, check=True)
    (checkout / "file.txt").write_text("after\n", encoding="utf-8")
    patch = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=checkout,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    subprocess.run(["git", "restore", "file.txt"], cwd=checkout, check=True)
    sidecar_path = task_dir / "openclaw_result.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "timeline_events": [
                    {
                        "type": "tool_call",
                        "arguments": {"command": f"echo {module.DEEPSWE_SUBMIT_MARKER}"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    evidence = module._deepswe_submission_evidence(
        "task-1",
        task_dir,
        checkout,
        {"openclaw_result_path": str(sidecar_path), "openclaw_usage": {}},
        patch,
    )

    assert evidence["mode"] == "explicit_submit"
    assert evidence["accepted"] is True
    assert evidence["patch_apply_check_ok"] is True
    assert "file.txt" in evidence["patch_stat"]


def test_configured_task_names_filters_task_names_file_when_include_is_set(tmp_path):
    task_names_file = tmp_path / "tasks.json"
    task_names_file.write_text(
        json.dumps(["task-a", "task-b", "task-c"]) + "\n",
        encoding="utf-8",
    )
    args = _args(tmp_path, task_names_file)
    args.include_task_name = ["task-b"]

    assert run_deepswe_openclaw.configured_task_names(args) == ["task-b"]


def test_deepswe_agent_python_runtime_paths_use_task_image_hash(monkeypatch):
    module = _load_deepswe_pier_agent_module(monkeypatch)
    task_toml = {
        "metadata": {"language": "python"},
        "environment": {
            "docker_image": "public.ecr.aws/example/python-task:kh77-v1.1"
        },
    }
    image_hash = module.hashlib.sha256(
        b"public.ecr.aws/example/python-task:kh77-v1.1"
    ).hexdigest()[:12]

    path_entries, pythonpath_entries = module._deepswe_python_runtime_paths(task_toml)

    assert path_entries == [f"/sandbox/.deepswe-tools/python-bin/{image_hash}"]
    assert pythonpath_entries == [
        f"/sandbox/.deepswe-tools/python-site/{image_hash}/site-packages"
    ]


@pytest.mark.parametrize(
    ("language", "expected_fragment"),
    [
        ("go", "GOPROXY=off"),
        ("python", "import pytest, setuptools"),
        ("typescript", "test -d node_modules"),
    ],
)
def test_deepswe_checkout_preflight_commands_are_offline(
    monkeypatch,
    language,
    expected_fragment,
):
    module = _load_deepswe_pier_agent_module(monkeypatch)

    resolved_language, command = module._deepswe_checkout_preflight_command(
        {"metadata": {"language": language}}
    )

    assert resolved_language == language
    assert expected_fragment in command[-1]
    assert "curl" not in command[-1]
    assert "wget" not in command[-1]


def test_deepswe_python_preflight_imports_target_package(monkeypatch):
    module = _load_deepswe_pier_agent_module(monkeypatch)

    _, command = module._deepswe_checkout_preflight_command(
        {
            "metadata": {
                "language": "python",
                "repository_url": "https://github.com/langchain-ai/langchain.git",
            }
        }
    )

    assert "langchain_core" in command[-1]
    assert "$PWD/libs/core" in command[-1]
    assert "--collect-only" not in command[-1]


def test_deepswe_failed_checkout_preflight_stops_before_model(monkeypatch, tmp_path):
    module = _load_deepswe_pier_agent_module(monkeypatch)
    model_calls = []

    monkeypatch.setattr(module.swe_runner, "ensure_nemoclaw_openclaw_permissions", lambda args: None)
    monkeypatch.setattr(
        module.swe_runner,
        "ensure_nemoclaw_checkout_ready",
        lambda checkout_dir, task_dir, args: {"mode": "copy"},
    )
    monkeypatch.setattr(module.swe_runner, "uses_nemoclaw_gateway_task_agent", lambda args: False)

    def fail_preflight(*args, **kwargs):
        raise module.NonScoreableOpenClawConfigurationError("offline compile failed")

    monkeypatch.setattr(module, "_preflight_deepswe_agent_checkout", fail_preflight)
    monkeypatch.setattr(
        module.swe_runner,
        "run_openclaw_for_task",
        lambda *args, **kwargs: model_calls.append(True),
    )
    agent = module.NejumiDeepSWEOpenClawAgent(
        logs_dir=tmp_path / "logs",
        output_root=tmp_path / "openclaw",
        model_name="openrouter-direct/z-ai/glm-5.2",
        verify_weave_agents=False,
        use_task_agent=False,
        restart_gateway_before_run=False,
    )
    checkout_dir = tmp_path / "checkout"
    checkout_dir.mkdir()

    with pytest.raises(module.NonScoreableOpenClawConfigurationError, match="offline compile failed"):
        agent._run_openclaw_host(
            "go-genai-streamed-function-args",
            {"metadata": {"language": "go"}, "environment": {}},
            "Implement the feature.",
            checkout_dir,
        )

    assert model_calls == []


def test_deepswe_checkout_preflight_injects_task_python_overlay(monkeypatch, tmp_path):
    module = _load_deepswe_pier_agent_module(monkeypatch)
    observed = {}

    def run_command(args, command, **kwargs):
        observed["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout="collected", stderr="")

    monkeypatch.setattr(module.swe_runner, "run_nemoclaw_text_command", run_command)
    monkeypatch.setattr(
        module,
        "_docker_image_identity",
        lambda task_toml: ("python-image", "sha256:python"),
    )
    args = types.SimpleNamespace(
        max_tool_wall_seconds=300,
        nemoclaw_checkout_sandbox_root="/sandbox/checkouts/test",
        nemoclaw_checkout_transfer_mode="copy",
        nemoclaw_sandbox="nejumi-taiwan",
        nemoclaw_extra_path=["/sandbox/.deepswe-tools/python-bin/hash"],
        nemoclaw_extra_pythonpath=["/sandbox/.deepswe-tools/python-site/hash/site-packages"],
    )

    evidence = module._preflight_deepswe_agent_checkout(
        "python-task",
        {"metadata": {"language": "python"}},
        tmp_path / "checkout",
        tmp_path,
        args,
    )

    script = observed["command"][-1]
    assert "export PATH=" in script
    assert "export PYTHONPATH=" in script
    assert evidence["extra_pythonpath"] == args.nemoclaw_extra_pythonpath


def test_deepswe_runtime_budget_stop_keeps_patch_scoreable(monkeypatch):
    module = _load_deepswe_pier_agent_module(monkeypatch)
    metadata = {
        "openclaw_disqualified_reason": "runtime_budget_exceeded",
        "runtime_budget": {
            "live": {
                "reason": "llm_response_idle_timeout",
                "interrupt_reason": "llm_response_idle_timeout",
                "exceeded_limits": ["llm_response_idle_timeout"],
            }
        },
    }

    assert module._metadata_has_scoreable_runtime_budget_failure(metadata) is True
    assert module._deepswe_should_force_empty_patch(metadata) is False


def test_deepswe_provider_timeout_still_forces_empty_patch(monkeypatch):
    module = _load_deepswe_pier_agent_module(monkeypatch)
    metadata = {
        "openclaw_disqualified_reason": "runtime_budget_exceeded",
        "runtime_budget": {
            "live": {
                "live_provider_timeout_count": 1,
                "reason": "live_provider_timeout",
                "exceeded_limits": ["live_provider_timeout"],
            }
        },
    }

    assert module._metadata_has_non_scoreable_provider_timeout(metadata) is True
    assert module._metadata_has_scoreable_runtime_budget_failure(metadata) is False
    assert module._deepswe_should_force_empty_patch(metadata) is True


def test_deepswe_protocol_disqualification_still_forces_empty_patch(monkeypatch):
    module = _load_deepswe_pier_agent_module(monkeypatch)
    metadata = {"openclaw_disqualified_reason": "conversation_order_violation"}

    assert module._deepswe_should_force_empty_patch(metadata) is True


def test_deepswe_runtime_budget_stop_captures_patch(monkeypatch, tmp_path):
    module = _load_deepswe_pier_agent_module(monkeypatch)
    calls = []

    monkeypatch.setattr(module.swe_runner, "ensure_nemoclaw_openclaw_permissions", lambda args: None)
    monkeypatch.setattr(
        module.swe_runner,
        "ensure_nemoclaw_checkout_ready",
        lambda checkout_dir, task_dir, args: {"mode": "copy"},
    )
    monkeypatch.setattr(module.swe_runner, "uses_nemoclaw_gateway_task_agent", lambda args: False)
    monkeypatch.setattr(
        module,
        "_preflight_deepswe_agent_checkout",
        lambda *args, **kwargs: {"ok": True, "environment_mode": "test"},
    )
    monkeypatch.setattr(
        module.swe_runner,
        "run_openclaw_for_task",
        lambda row, checkout_dir, task_dir, args: {
            "openclaw_disqualified_reason": "runtime_budget_exceeded",
            "runtime_budget": {
                "live": {
                    "reason": "llm_response_idle_timeout",
                    "interrupt_reason": "llm_response_idle_timeout",
                    "exceeded_limits": ["llm_response_idle_timeout"],
                }
            },
        },
    )

    def fake_capture_patch_nemoclaw(checkout_dir, args, excluded_paths):
        calls.append(("capture_patch_nemoclaw", checkout_dir, excluded_paths))
        return "diff --git a/file.ts b/file.ts\n"

    monkeypatch.setattr(module.swe_runner, "capture_patch_nemoclaw", fake_capture_patch_nemoclaw)
    monkeypatch.setattr(module.swe_runner, "capture_patch", lambda checkout_dir, excluded_paths: "")

    agent = module.NejumiDeepSWEOpenClawAgent(
        logs_dir=tmp_path / "logs",
        output_root=tmp_path / "openclaw",
        model_name="openrouter-direct/z-ai/glm-5.2",
        verify_weave_agents=False,
        use_task_agent=False,
        restart_gateway_before_run=False,
    )
    checkout_dir = tmp_path / "checkout"
    checkout_dir.mkdir()

    metadata, patch = agent._run_openclaw_host(
        "superjson-error-stack-serialization",
        {"metadata": {"language": "typescript"}, "environment": {}},
        "Implement the feature.",
        checkout_dir,
    )

    assert calls == [("capture_patch_nemoclaw", checkout_dir, [])]
    assert patch == "diff --git a/file.ts b/file.ts\n"
    assert metadata["deepswe_scoreable_failure_reason"] == "llm_response_idle_timeout"
    assert metadata["deepswe_checkout_preflight"]["ok"] is True
    assert metadata["deepswe_patch_bytes"] == len(patch.encode("utf-8"))
    assert metadata["deepswe_submission"]["mode"] == "invalid_patch_submission"
    assert metadata["deepswe_submission"]["accepted"] is False
    assert (tmp_path / "openclaw" / "superjson-error-stack-serialization" / "model.patch").read_text(
        encoding="utf-8"
    ) == patch


def test_deepswe_provider_timeout_recovers_without_aborting_other_work(
    monkeypatch,
    tmp_path,
):
    module = _load_deepswe_pier_agent_module(monkeypatch)
    model_results = [
        {
            "openclaw_disqualified_reason": "provider_transient_exhausted",
            "openclaw_usage": {"inputTokens": 10, "outputTokens": 2},
            "runtime_budget": {
                "live": {
                    "live_provider_timeout_count": 1,
                    "reason": "live_provider_timeout",
                }
            },
        },
        {
            "openclaw_disqualified_reason": "",
            "openclaw_usage": {"inputTokens": 20, "outputTokens": 3},
        },
    ]
    monkeypatch.setattr(module.swe_runner, "ensure_nemoclaw_openclaw_permissions", lambda args: None)
    monkeypatch.setattr(
        module.swe_runner,
        "ensure_nemoclaw_checkout_ready",
        lambda checkout_dir, task_dir, args: {"mode": "copy"},
    )
    monkeypatch.setattr(module.swe_runner, "uses_nemoclaw_gateway_task_agent", lambda args: False)
    monkeypatch.setattr(
        module,
        "_preflight_deepswe_agent_checkout",
        lambda *args, **kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        module.swe_runner,
        "run_openclaw_for_task",
        lambda *args, **kwargs: model_results.pop(0),
    )
    monkeypatch.setattr(
        module.swe_runner,
        "capture_patch_nemoclaw",
        lambda *args, **kwargs: "diff --git a/file.ts b/file.ts\n",
    )

    agent = module.NejumiDeepSWEOpenClawAgent(
        logs_dir=tmp_path / "logs",
        output_root=tmp_path / "openclaw",
        model_name="wandb-inference/zai-org/GLM-5.2",
        verify_weave_agents=False,
        use_task_agent=False,
        restart_gateway_before_run=False,
        provider_recovery_rounds=1,
        provider_recovery_base_seconds=0,
    )
    checkout_dir = tmp_path / "checkout"
    checkout_dir.mkdir()

    metadata, patch = agent._run_openclaw_host(
        "provider-recovery-task",
        {"metadata": {"language": "typescript"}, "environment": {}},
        "Implement the feature.",
        checkout_dir,
    )

    assert model_results == []
    assert patch == "diff --git a/file.ts b/file.ts\n"
    assert len(metadata["provider_recovery"]) == 1
    assert metadata["billable_openclaw_attempt_count"] == 2
    assert metadata["billable_openclaw_usage"]["inputTokens"] == 30
    assert metadata["billable_openclaw_usage"]["outputTokens"] == 5
    recovery_path = tmp_path / "openclaw" / "provider-recovery-task" / "provider_recovery.json"
    assert json.loads(recovery_path.read_text(encoding="utf-8"))[0]["recovery_round"] == 1


def test_deepswe_isolated_native_trace_failure_is_recovered_task_locally(
    monkeypatch,
    tmp_path,
):
    module = _load_deepswe_pier_agent_module(monkeypatch)
    model_results = [
        {
            "weave_agents_required": True,
            "weave_agents_ok": False,
            "openclaw_usage": {"inputTokens": 10, "outputTokens": 2},
        },
        {
            "weave_agents_required": True,
            "weave_agents_ok": True,
            "openclaw_usage": {"inputTokens": 20, "outputTokens": 3},
        },
    ]
    monkeypatch.setattr(module.swe_runner, "ensure_nemoclaw_openclaw_permissions", lambda args: None)
    monkeypatch.setattr(
        module.swe_runner,
        "ensure_nemoclaw_checkout_ready",
        lambda checkout_dir, task_dir, args: {"mode": "copy"},
    )
    monkeypatch.setattr(module.swe_runner, "uses_nemoclaw_gateway_task_agent", lambda args: False)
    monkeypatch.setattr(
        module,
        "_preflight_deepswe_agent_checkout",
        lambda *args, **kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        module.swe_runner,
        "run_openclaw_for_task",
        lambda *args, **kwargs: model_results.pop(0),
    )
    monkeypatch.setattr(
        module.swe_runner,
        "capture_patch_nemoclaw",
        lambda *args, **kwargs: "diff --git a/file.ts b/file.ts\n",
    )

    agent = module.NejumiDeepSWEOpenClawAgent(
        logs_dir=tmp_path / "logs",
        output_root=tmp_path / "openclaw",
        model_name="openai-direct/gpt-5.6-luna",
        verify_weave_agents=True,
        use_task_agent=False,
        restart_gateway_before_run=False,
        provider_recovery_rounds=0,
        native_trace_recovery_attempts=1,
        native_trace_recovery_base_seconds=0,
    )
    checkout_dir = tmp_path / "checkout"
    checkout_dir.mkdir()

    metadata, patch = agent._run_openclaw_host(
        "trace-recovery-task",
        {"metadata": {"language": "typescript"}, "environment": {}},
        "Implement the feature.",
        checkout_dir,
    )

    assert model_results == []
    assert patch == "diff --git a/file.ts b/file.ts\n"
    assert metadata["weave_agents_ok"] is True
    assert metadata["billable_openclaw_attempt_count"] == 2
    assert metadata["billable_openclaw_usage"]["inputTokens"] == 30
    recovery_path = (
        tmp_path
        / "openclaw"
        / "trace-recovery-task"
        / "native_trace_recovery.json"
    )
    assert json.loads(recovery_path.read_text(encoding="utf-8"))[0][
        "recovery_round"
    ] == 1


def test_deepswe_model_execution_is_not_serialized_by_setup_lock(monkeypatch, tmp_path):
    module = _load_deepswe_pier_agent_module(monkeypatch)
    model_execution_barrier = threading.Barrier(2, timeout=2.0)
    entered: list[str] = []
    entered_lock = threading.Lock()

    monkeypatch.setattr(module.swe_runner, "ensure_nemoclaw_openclaw_permissions", lambda args: None)
    monkeypatch.setattr(
        module.swe_runner,
        "ensure_nemoclaw_checkout_ready",
        lambda checkout_dir, task_dir, args: {"mode": "copy"},
    )
    monkeypatch.setattr(module.swe_runner, "uses_nemoclaw_gateway_task_agent", lambda args: False)
    monkeypatch.setattr(
        module,
        "_preflight_deepswe_agent_checkout",
        lambda *args, **kwargs: {"ok": True, "environment_mode": "test"},
    )

    def fake_run_openclaw_for_task(row, checkout_dir, task_dir, args):
        with entered_lock:
            entered.append(row["instance_id"])
        model_execution_barrier.wait()
        return {"status": "ok"}

    monkeypatch.setattr(
        module.swe_runner,
        "run_openclaw_for_task",
        fake_run_openclaw_for_task,
    )
    monkeypatch.setattr(
        module.swe_runner,
        "capture_patch_nemoclaw",
        lambda checkout_dir, args, excluded_paths: "",
    )
    monkeypatch.setattr(module.swe_runner, "capture_patch", lambda checkout_dir, excluded_paths: "")

    def run_task(task_id: str):
        agent = module.NejumiDeepSWEOpenClawAgent(
            logs_dir=tmp_path / f"logs-{task_id}",
            output_root=tmp_path / "openclaw",
            model_name="wandb-inference/zai-org/GLM-5.2",
            verify_weave_agents=False,
            use_task_agent=False,
            restart_gateway_before_run=False,
        )
        checkout_dir = tmp_path / f"checkout-{task_id}"
        checkout_dir.mkdir()
        return agent._run_openclaw_host(
            task_id,
            {"metadata": {"language": "typescript"}, "environment": {}},
            "Implement the feature.",
            checkout_dir,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run_task, ["task-a", "task-b"]))

    assert sorted(entered) == ["task-a", "task-b"]
    assert [patch for _, patch in results] == ["", ""]


def test_build_job_config_pins_deepswe_budget_and_native_agent(tmp_path):
    task_names_file = tmp_path / "pilot_2_task_names.json"
    task_names_file.write_text(json.dumps(["task-a", "task-b"]) + "\n", encoding="utf-8")
    args = _args(tmp_path, task_names_file)

    config = run_deepswe_openclaw.build_job_config(args, "job-a")

    assert config["job_name"] == "job-a"
    assert config["n_concurrent_trials"] == 2
    assert config["datasets"][0]["task_names"] == ["task-a", "task-b"]
    assert config["agents"][0]["import_path"] == run_deepswe_openclaw.AGENT_IMPORT_PATH
    kwargs = config["agents"][0]["kwargs"]
    assert kwargs["openclaw_model"] == "openrouter-direct/z-ai/glm-5.2"
    assert kwargs["max_input_tokens"] == 1_000_000
    assert kwargs["max_tool_calls"] == 40
    assert kwargs["max_agent_turns"] == 40
    assert kwargs["final_assistant_idle_salvage_seconds"] == 60.0
    assert kwargs["final_assistant_shutdown_grace_seconds"] == 30.0
    assert kwargs["llm_response_idle_timeout_seconds"] == 900.0
    assert kwargs["no_local"] == "true"
    assert kwargs["use_task_agent"] == "true"
    assert kwargs["verify_weave_agents"] == "true"
    assert kwargs["deny_tool"] == "web_search,browser"
    assert kwargs["deny_argument_pattern"] == r"https?://"


def test_build_job_config_forwards_openclaw_model_params_json(tmp_path):
    task_names_file = tmp_path / "pilot_2_task_names.json"
    task_names_file.write_text(json.dumps(["task-a"]) + "\n", encoding="utf-8")
    args = _args(tmp_path, task_names_file)
    args.openclaw_model_params_json = json.dumps(
        {"provider": {"only": ["z-ai/fp8"], "allow_fallbacks": False}}
    )
    args.openclaw_model_overrides_json = json.dumps({"maxTokens": 4096})

    config = run_deepswe_openclaw.build_job_config(args, "job-a")

    kwargs = config["agents"][0]["kwargs"]
    assert kwargs["openclaw_model_params_json"] == args.openclaw_model_params_json
    assert kwargs["openclaw_model_overrides_json"] == args.openclaw_model_overrides_json


def test_sandbox_tool_preflight_retries_after_stale_nemoclaw_lock(monkeypatch, tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    lock_path = state_dir / "shields-transition-lock-nejumi-taiwan.json"
    lock_path.write_text(
        json.dumps(
            {
                "version": 1,
                "sandboxName": "nejumi-taiwan",
                "pid": 999999999,
                "command": "inspect mutable config permissions",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(run_deepswe_openclaw, "NEMOCLAW_STATE_DIR", state_dir)

    calls = []

    def fake_kill(pid, sig):
        assert sig == 0
        if pid == 999999999:
            raise ProcessLookupError
        raise AssertionError(f"unexpected pid: {pid}")

    def fake_run(command, text, stdout, stderr, check):
        calls.append(command)
        if len(calls) == 1:
            return run_deepswe_openclaw.subprocess.CompletedProcess(
                command,
                1,
                stdout="node\tOK\t/usr/local/bin/node\n",
                stderr=(
                    "permission inspection failed: Timed out after 30000ms waiting "
                    f"for shields transition lock '{lock_path}': recorded owner PID "
                    "999999999 is not running (inspect mutable config permissions)."
                ),
            )
        return run_deepswe_openclaw.subprocess.CompletedProcess(
            command,
            0,
            stdout="node\tOK\t/usr/local/bin/node\n",
            stderr="",
        )

    monkeypatch.setattr(run_deepswe_openclaw.os, "kill", fake_kill)
    monkeypatch.setattr(run_deepswe_openclaw.subprocess, "run", fake_run)

    availability = run_deepswe_openclaw._check_tools_with_command(["nemoclaw", "sandbox"])

    assert availability == {"node": {"available": True, "path": "/usr/local/bin/node"}}
    assert len(calls) == 2
    assert not lock_path.exists()


def test_parse_no_local_flag_keeps_gateway_mode(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_deepswe_openclaw.py", "--model", "dummy/model"])
    args = run_deepswe_openclaw.parse_args()
    assert args.no_local is True

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_deepswe_openclaw.py", "--model", "dummy/model", "--no-local"],
    )
    args = run_deepswe_openclaw.parse_args()
    assert args.no_local is True

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_deepswe_openclaw.py",
            "--model",
            "dummy/model",
            "--local",
            "--no-verify-weave-agents",
        ],
    )
    args = run_deepswe_openclaw.parse_args()
    assert args.no_local is False


def test_prepare_openclaw_sandbox_runtime_runs_setup_before_model(tmp_path, monkeypatch):
    task_names_file = tmp_path / "tasks.json"
    task_names_file.write_text(json.dumps(["python-task"]) + "\n", encoding="utf-8")
    setup_script = tmp_path / "install_deepswe_sandbox_deps.sh"
    setup_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    args = _args(tmp_path, task_names_file)
    args.prepare_sandbox_runtime = True
    calls = []

    class Result:
        returncode = 0
        stdout = "python-overlay\tOK\timage\t/sandbox/.deepswe-tools/python-site/hash/site-packages\n"

    def fake_run(command, **_kwargs):
        calls.append(command)
        return Result()

    monkeypatch.setattr(run_deepswe_openclaw, "DEEPSWE_SANDBOX_DEPS_SCRIPT", setup_script)
    monkeypatch.setattr(run_deepswe_openclaw.subprocess, "run", fake_run)

    payload = run_deepswe_openclaw.prepare_openclaw_sandbox_runtime(args)

    assert payload["enabled"] is True
    assert payload["returncode"] == 0
    assert calls == [
        [
            str(setup_script),
            "--sandbox",
            "nejumi-taiwan",
            "--nemoclaw-bin",
            "nemoclaw",
            "--tasks-root",
            str(args.tasks_root),
            "--task-names-file",
            str(task_names_file),
            "--no-pull",
        ]
    ]


def test_prepare_openclaw_sandbox_runtime_uses_filtered_task_names_for_include(
    tmp_path, monkeypatch
):
    task_names_file = tmp_path / "tasks.json"
    task_names_file.write_text(
        json.dumps(["python-task", "typescript-task"]) + "\n",
        encoding="utf-8",
    )
    setup_script = tmp_path / "install_deepswe_sandbox_deps.sh"
    setup_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    args = _args(tmp_path, task_names_file)
    args.prepare_sandbox_runtime = True
    args.include_task_name = ["typescript-task"]
    calls = []

    class Result:
        returncode = 0
        stdout = "node\tOK\t/usr/local/bin/node\n"

    def fake_run(command, **_kwargs):
        calls.append(command)
        return Result()

    monkeypatch.setattr(run_deepswe_openclaw, "DEEPSWE_SANDBOX_DEPS_SCRIPT", setup_script)
    monkeypatch.setattr(run_deepswe_openclaw.subprocess, "run", fake_run)

    payload = run_deepswe_openclaw.prepare_openclaw_sandbox_runtime(args)

    assert payload["enabled"] is True
    generated_task_names_file = args.output_dir / "sandbox_runtime_task_names.json"
    assert json.loads(generated_task_names_file.read_text(encoding="utf-8")) == [
        "typescript-task"
    ]
    assert calls[0][calls[0].index("--task-names-file") + 1] == str(
        generated_task_names_file
    )


def test_prepare_openclaw_sandbox_runtime_fails_before_model(tmp_path, monkeypatch):
    task_names_file = tmp_path / "tasks.json"
    task_names_file.write_text(json.dumps(["python-task"]) + "\n", encoding="utf-8")
    setup_script = tmp_path / "install_deepswe_sandbox_deps.sh"
    setup_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    args = _args(tmp_path, task_names_file)

    class Result:
        returncode = 1
        stdout = "python-overlay\tMISSING\timage\t/sandbox/.deepswe-tools/python-site/hash/site-packages\n"

    monkeypatch.setattr(run_deepswe_openclaw, "DEEPSWE_SANDBOX_DEPS_SCRIPT", setup_script)
    monkeypatch.setattr(run_deepswe_openclaw.subprocess, "run", lambda *a, **k: Result())

    try:
        run_deepswe_openclaw.prepare_openclaw_sandbox_runtime(args)
    except RuntimeError as exc:
        text = str(exc)
        assert "before model execution" in text
        assert "python-overlay" in text
    else:
        raise AssertionError("Expected sandbox runtime preflight to fail")


def test_deepswe_setup_tracks_go_cache_per_immutable_task_image():
    script = (
        ROOT / "scripts" / "setup" / "install_deepswe_sandbox_deps.sh"
    ).read_text(encoding="utf-8")

    assert "go-mod-cache-markers" in script
    assert "docker image inspect --format '{{.Id}}'" in script
    assert "write_go_cache_marker \"$selected_go_image\"" in script
    assert 'find /sandbox/go/pkg/mod -type d -name "*@v*"' not in script


def test_build_job_config_rejects_native_trace_without_gateway_mode(tmp_path):
    task_names_file = tmp_path / "pilot_2_task_names.json"
    task_names_file.write_text(json.dumps(["task-a", "task-b"]) + "\n", encoding="utf-8")
    args = _args(tmp_path, task_names_file)
    args.no_local = False

    try:
        run_deepswe_openclaw.build_job_config(args, "job-a")
    except ValueError as exc:
        assert "native Weave Agents verification requires --no-local" in str(exc)
    else:
        raise AssertionError("Expected build_job_config to reject local native trace mode")


def test_dry_run_results_are_traceably_incomplete(tmp_path):
    task_names_file = tmp_path / "pilot_2_task_names.json"
    task_names_file.write_text(json.dumps(["task-a", "task-b"]) + "\n", encoding="utf-8")
    args = _args(tmp_path, task_names_file)
    config_path = tmp_path / "out" / "pier_job_config.json"
    run_deepswe_openclaw.write_json(config_path, run_deepswe_openclaw.build_job_config(args, "job-a"))

    run_deepswe_openclaw.write_dry_run_results(
        tmp_path / "out",
        command=["pier", "run", "--config", str(config_path), "--yes"],
        config_path=config_path,
        args=args,
    )

    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert summary["dry_run"] is True
    assert summary["total_trials"] == 2
    assert summary["native_trace_missing"] == 2
    assert [row["task_name"] for row in rows] == ["datacurve/task-a", "datacurve/task-b"]
    assert all(row["dry_run"] is True for row in rows)


def test_collect_results_uses_openclaw_metadata_fallback_for_exception_rows(tmp_path):
    job_dir = tmp_path / "jobs" / "job-a"
    trial_dir = job_dir / "trial-1"
    trial_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text(json.dumps({"ok": True}) + "\n", encoding="utf-8")
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "datacurve/expr-try-catch-errors",
                "trial_name": "trial-1",
                "exception_info": {"type": "RuntimeError"},
                "agent_result": {
                    "metadata": {"patch_apply": {"patch_applied": True}}
                },
                "verifier_result": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    metadata_dir = tmp_path / "out" / "openclaw" / "expr-try-catch-errors"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "deepswe_openclaw_metadata.json").write_text(
        json.dumps(
            {
                "openclaw_result_path": "openclaw-result.json",
                "openclaw_tool_call_count": 12,
                "openclaw_usage": {"inputTokens": 100, "outputTokens": 20},
                "weave_agents_ok": True,
                "weave_agents_conversation_url": "https://wandb.ai/example/conversation",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    run_deepswe_openclaw.collect_results(
        job_dir,
        tmp_path / "out",
        command=["pier", "run"],
        elapsed=1.0,
    )

    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert summary["weave_agents_ok"] == 1
    assert summary["native_trace_missing"] == 0
    assert rows[0]["weave_agents_ok"] is True
    assert rows[0]["openclaw_tool_call_count"] == 12
    assert rows[0]["agent_result"]["metadata"]["openclaw"]["openclaw_usage"]["inputTokens"] == 100


def test_collect_results_falls_back_to_weave_agents_trace_usage(tmp_path):
    job_dir = tmp_path / "jobs" / "job-a"
    trial_dir = job_dir / "trial-1"
    trial_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text(json.dumps({"ok": True}) + "\n", encoding="utf-8")
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "datacurve/psd-tools-blend-range-api",
                "trial_name": "trial-1",
                "agent_result": {"metadata": {}},
                "verifier_result": {"rewards": {"reward": 0}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    metadata_dir = tmp_path / "out" / "openclaw" / "psd-tools-blend-range-api"
    verifier_dir = metadata_dir / "weave_agents_verifications"
    verifier_dir.mkdir(parents=True)
    verifier_path = verifier_dir / "agent.json"
    verifier_path.write_text(
        json.dumps(
            {
                "content_capture_health": {
                    "trace_input_tokens": 4321501,
                    "trace_output_tokens": 17563,
                    "conversation_input_tokens": 4321501,
                    "conversation_output_tokens": 17563,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (metadata_dir / "deepswe_openclaw_metadata.json").write_text(
        json.dumps(
            {
                "openclaw_result_path": "openclaw-result.json",
                "openclaw_tool_call_count": 47,
                "openclaw_usage": {},
                "weave_agents_ok": True,
                "weave_agents_verifier_json": str(verifier_path),
                "weave_agents_conversation_url": "https://wandb.ai/example/conversation",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    run_deepswe_openclaw.collect_results(
        job_dir,
        tmp_path / "out",
        command=["pier", "run"],
        elapsed=1.0,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert rows[0]["weave_agents_verifier_json"] == str(verifier_path)
    assert rows[0]["openclaw_usage"] == {
        "inputTokens": 4321501,
        "outputTokens": 17563,
        "cacheReadInputTokens": 0,
        "usageSource": "weave_agents_trace",
        "usageApproximate": True,
    }


def test_collect_results_disqualifies_trace_usage_over_runtime_cap(tmp_path):
    job_dir = tmp_path / "jobs" / "job-a"
    trial_dir = job_dir / "trial-1"
    trial_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text(json.dumps({"ok": True}) + "\n", encoding="utf-8")
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "datacurve/psd-tools-blend-range-api",
                "trial_name": "trial-1",
                "agent_result": {"metadata": {}},
                "verifier_result": {"rewards": {"reward": 1}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    metadata_dir = tmp_path / "out" / "openclaw" / "psd-tools-blend-range-api"
    verifier_dir = metadata_dir / "weave_agents_verifications"
    verifier_dir.mkdir(parents=True)
    verifier_path = verifier_dir / "agent.json"
    verifier_path.write_text(
        json.dumps(
            {
                "content_capture_health": {
                    "trace_input_tokens": 4321501,
                    "trace_output_tokens": 17563,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (metadata_dir / "deepswe_openclaw_metadata.json").write_text(
        json.dumps(
            {
                "openclaw_usage": {},
                "weave_agents_ok": True,
                "weave_agents_verifier_json": str(verifier_path),
                "runtime_budget": {
                    "limits": {
                        "max_cumulative_input_tokens": 3000000,
                        "max_cumulative_output_tokens": 500000,
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    run_deepswe_openclaw.collect_results(
        job_dir,
        tmp_path / "out",
        command=["pier", "run"],
        elapsed=1.0,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert rows[0]["score"] == 0.0
    assert rows[0]["resolved"] is False
    assert rows[0]["openclaw_disqualified_reason"] == "runtime_budget_exceeded"
    assert rows[0]["runtime_budget_posthoc_trace_violations"] == [
        {
            "type": "trace_cumulative_input_tokens_exceeded",
            "observed": 4321501,
            "limit": 3000000,
            "source": "weave_agents_trace",
        }
    ]


def test_collect_results_reads_pier_rewards_score(tmp_path):
    job_dir = tmp_path / "jobs" / "job-a"
    trial_dir = job_dir / "trial-1"
    trial_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text(json.dumps({"ok": True}) + "\n", encoding="utf-8")
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "datacurve/termenv-preserve-ansi-resets",
                "trial_name": "trial-1",
                "agent_result": {"metadata": {}},
                "verifier_result": {
                    "rewards": {
                        "reward": 0,
                        "partial": 0.71,
                        "f2p_passed": 0,
                        "f2p_total": 35,
                        "p2p_passed": 13,
                        "p2p_total": 13,
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    run_deepswe_openclaw.collect_results(
        job_dir,
        tmp_path / "out",
        command=["pier", "run"],
        elapsed=1.0,
    )

    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert rows[0]["score"] == 0.0
    assert rows[0]["resolved"] is False
    assert rows[0]["diagnostic_score_with_partial"] == 0.0
    assert rows[0]["f2p_total"] == 35
    assert rows[0]["p2p_total"] == 13
    assert rows[0]["verifier_rewards"]["partial"] == 0.71
    assert summary["scored_trials"] == 1
    assert summary["pass_at_1"] == 0.0
    assert summary["diagnostic_score_with_partial"] == 0.0
    assert summary["diagnostic_evidence_trials"] == 1


def test_selected_docker_images_reads_task_toml_in_task_order(tmp_path):
    task_names_file = tmp_path / "tasks.json"
    task_names_file.write_text(json.dumps(["task-a", "task-b", "task-c"]) + "\n", encoding="utf-8")
    for name, image in {
        "task-a": "public.ecr.aws/example/image-a:v1",
        "task-b": "public.ecr.aws/example/image-b:v1",
        "task-c": "public.ecr.aws/example/image-a:v1",
    }.items():
        task_dir = tmp_path / "tasks" / name
        task_dir.mkdir(parents=True)
        (task_dir / "task.toml").write_text(f'docker_image = "{image}"\n', encoding="utf-8")
    args = _args(tmp_path, task_names_file)

    assert run_deepswe_openclaw.selected_docker_images(args) == [
        "public.ecr.aws/example/image-a:v1",
        "public.ecr.aws/example/image-b:v1",
    ]


def test_preflight_docker_images_fails_before_model_when_pull_fails(tmp_path, monkeypatch):
    task_names_file = tmp_path / "tasks.json"
    task_names_file.write_text(json.dumps(["task-a"]) + "\n", encoding="utf-8")
    task_dir = tmp_path / "tasks" / "task-a"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text(
        'docker_image = "public.ecr.aws/example/missing:v1"\n',
        encoding="utf-8",
    )
    args = _args(tmp_path, task_names_file)
    args.docker_pull_retries = 1
    args.docker_pull_retry_seconds = 0.0

    monkeypatch.setattr(run_deepswe_openclaw, "docker_image_present", lambda image: False)

    class Result:
        returncode = 1
        stdout = "toomanyrequests: Rate exceeded"

    monkeypatch.setattr(run_deepswe_openclaw.subprocess, "run", lambda *a, **k: Result())

    try:
        run_deepswe_openclaw.preflight_docker_images(args)
    except RuntimeError as exc:
        text = str(exc)
        assert "before model execution" in text
        assert "toomanyrequests" in text
    else:
        raise AssertionError("Expected preflight to fail before model execution")


def test_preflight_openclaw_sandbox_tools_fails_before_model_when_tool_missing(
    tmp_path, monkeypatch
):
    task_names_file = tmp_path / "tasks.json"
    task_names_file.write_text(json.dumps(["go-task"]) + "\n", encoding="utf-8")
    task_dir = tmp_path / "tasks" / "go-task"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text(
        "\n".join(
            [
                "[metadata]",
                'language = "go"',
                "[environment]",
                'docker_image = "public.ecr.aws/example/go-task:v1"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    args = _args(tmp_path, task_names_file)

    def fake_availability(_args, required_tools):
        assert required_tools == ["go"]
        return {"go": {"available": False, "path": None}}

    monkeypatch.setattr(
        run_deepswe_openclaw,
        "openclaw_execution_tool_availability",
        fake_availability,
    )

    try:
        run_deepswe_openclaw.preflight_openclaw_sandbox_tools(args)
    except run_deepswe_openclaw.DeepSWEPreflightError as exc:
        text = str(exc)
        assert "before model execution" in text
        assert "evaluation setup mismatch" in text
        assert "not a model incorrect answer" in text
        assert "go" in text
        assert exc.payload["missing_tools"] == ["go"]
    else:
        raise AssertionError("Expected missing sandbox tool preflight to fail")


def test_preflight_openclaw_sandbox_tools_passes_when_required_tools_available(
    tmp_path, monkeypatch
):
    task_names_file = tmp_path / "tasks.json"
    task_names_file.write_text(json.dumps(["ts-task"]) + "\n", encoding="utf-8")
    task_dir = tmp_path / "tasks" / "ts-task"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text(
        "\n".join(
            [
                "[metadata]",
                'language = "typescript"',
                "[environment]",
                'docker_image = "public.ecr.aws/example/ts-task:v1"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    args = _args(tmp_path, task_names_file)

    monkeypatch.setattr(
        run_deepswe_openclaw,
        "openclaw_execution_tool_availability",
        lambda _args, tools: {"node": {"available": True, "path": "/usr/bin/node"}},
    )

    payload = run_deepswe_openclaw.preflight_openclaw_sandbox_tools(args)

    assert payload["missing_tools"] == []
    assert payload["required_tools_by_language"] == {"typescript": ["node"]}
    assert payload["availability"]["node"]["available"] is True


def test_openclaw_tool_preflight_uses_model_visible_default_path(tmp_path, monkeypatch):
    task_names_file = tmp_path / "tasks.json"
    task_names_file.write_text(json.dumps(["go-task"]) + "\n", encoding="utf-8")
    args = _args(tmp_path, task_names_file)
    captured = {}

    class Result:
        returncode = 0
        stdout = "go\tOK\t/usr/local/bin/go\n"
        stderr = ""

    def fake_run(command, **_kwargs):
        captured["command"] = command
        return Result()

    monkeypatch.setattr(run_deepswe_openclaw.subprocess, "run", fake_run)

    availability = run_deepswe_openclaw.openclaw_execution_tool_availability(args, ["go"])

    command_text = " ".join(captured["command"])
    assert "/sandbox/.deepswe-tools" not in command_text
    assert "/sandbox/.npm-global" not in command_text
    assert availability["go"]["available"] is True
    assert availability["go"]["path"] == "/usr/local/bin/go"


def test_environment_setup_failure_detection_matches_pier_docker_pull_error():
    result = {
        "task_name": "datacurve/ofetch-per-origin-circuit-breaker",
        "trial_name": "ofetch",
        "agent_result": None,
        "agent_execution": None,
        "environment_setup": {
            "started_at": "2026-07-12T11:13:59Z",
            "finished_at": "2026-07-12T11:14:01Z",
        },
        "exception_info": {
            "exception_type": "RuntimeError",
            "exception_message": "Docker compose command failed. Error toomanyrequests: Rate exceeded",
        },
    }

    assert run_deepswe_openclaw.is_environment_setup_failure(result) is True


def test_environment_setup_failure_detection_ignores_agent_result_exception():
    result = {
        "agent_result": {"metadata": {}},
        "agent_execution": {"started_at": "2026-07-12T11:13:59Z"},
        "environment_setup": {
            "started_at": "2026-07-12T11:13:00Z",
            "finished_at": "2026-07-12T11:13:10Z",
        },
        "exception_info": {
            "exception_type": "RuntimeError",
            "exception_message": "OpenClaw failed",
        },
    }

    assert run_deepswe_openclaw.is_environment_setup_failure(result) is False


def test_policy_block_exception_is_scoreable():
    result = {
        "task_name": "datacurve/kcp-go-multiplexed-kcp-streams",
        "trial_name": "kcp-go",
        "agent_result": None,
        "exception_info": {
            "exception_type": "NonScoreableOpenClawPolicyBlock",
            "exception_message": "non_scoreable_policy_block: OpenClaw tool policy blocked this DeepSWE task",
        },
    }

    assert run_deepswe_openclaw.is_non_scoreable_policy_failure(result) is False


def test_policy_block_metadata_is_scoreable():
    result = {
        "agent_result": {
            "metadata": {
                "openclaw": {
                    "openclaw_disqualified_reason": "runtime_budget_exceeded",
                    "runtime_budget": {
                        "violations": [
                            {
                                "type": "live_tool_policy_violation",
                                "source": "live_runtime_budget",
                            }
                        ]
                    },
                }
            }
        },
        "exception_info": None,
    }

    assert run_deepswe_openclaw.is_non_scoreable_policy_failure(result) is False


def test_non_scoreable_provider_timeout_detection_matches_openclaw_metadata():
    result = {
        "agent_result": {
            "metadata": {
                "openclaw": {
                    "openclaw_disqualified_reason": "runtime_budget_exceeded",
                    "runtime_budget": {
                        "live": {
                            "live_provider_timeout_count": 1,
                            "live_provider_timeouts": [
                                {
                                    "type": "provider_timeout",
                                    "errorCode": "504",
                                    "errorMessage": "Upstream idle timeout exceeded",
                                }
                            ],
                            "reason": "live_provider_timeout",
                        },
                    },
                }
            }
        },
        "exception_info": None,
    }

    assert run_deepswe_openclaw.is_non_scoreable_provider_timeout(result) is True

    idle_result = {
        "agent_result": {
            "metadata": {
                "openclaw": {
                    "openclaw_disqualified_reason": "runtime_budget_exceeded",
                    "runtime_budget": {
                        "live": {
                            "reason": "llm_response_idle_timeout",
                            "interrupt_reason": "llm_response_idle_timeout",
                            "exceeded_limits": ["llm_response_idle_timeout"],
                            "llm_response_idle_seconds": 901.0,
                        },
                    },
                }
            }
        },
        "exception_info": None,
    }

    assert run_deepswe_openclaw.is_non_scoreable_provider_timeout(idle_result) is False
    assert run_deepswe_openclaw.is_non_scoreable_llm_response_idle_timeout(idle_result) is False
    assert run_deepswe_openclaw.is_llm_response_idle_timeout(idle_result) is True


def test_non_scoreable_provider_timeout_detection_matches_agent_exception():
    result = {
        "task_name": "datacurve/etree-xml-diff-patch",
        "trial_name": "etree",
        "agent_result": None,
        "exception_info": {
            "exception_type": "NonScoreableOpenClawProviderTimeout",
            "exception_message": "non_scoreable_provider_timeout: OpenClaw provider timed out",
        },
    }

    assert run_deepswe_openclaw.is_non_scoreable_provider_timeout(result) is True


def test_llm_response_idle_timeout_detection_matches_agent_exception_without_non_scoreable():
    result = {
        "task_name": "datacurve/etree-xml-diff-patch",
        "trial_name": "etree",
        "agent_result": None,
        "exception_info": {
            "exception_type": "NonScoreableOpenClawLLMResponseIdleTimeout",
            "exception_message": (
                "non_scoreable_llm_response_idle_timeout: OpenClaw did not receive "
                "a model response after a tool result"
            ),
        },
    }

    assert run_deepswe_openclaw.is_non_scoreable_provider_timeout(result) is False
    assert run_deepswe_openclaw.is_non_scoreable_llm_response_idle_timeout(result) is False
    assert run_deepswe_openclaw.is_llm_response_idle_timeout(result) is True


def test_non_scoreable_configuration_error_detection_matches_agent_exception():
    result = {
        "task_name": "datacurve/etree-xml-diff-patch",
        "trial_name": "etree",
        "agent_result": None,
        "exception_info": {
            "exception_type": "NonScoreableOpenClawConfigurationError",
            "exception_message": (
                "non_scoreable_configuration_error: requested OpenClaw thinking level "
                "is unsupported by the model"
            ),
        },
    }

    assert run_deepswe_openclaw.is_non_scoreable_configuration_error(result) is True


def test_required_trace_failure_is_fail_fast_for_completed_model_result():
    result = {
        "agent_result": {
            "metadata": {
                "openclaw": {
                    "weave_agents_required": True,
                    "weave_agents_ok": False,
                    "openclaw_disqualified_reason": "",
                }
            }
        }
    }

    assert run_deepswe_openclaw.is_required_trace_evidence_failure(result) is True


def test_required_trace_exception_is_fail_fast_before_scoring():
    result = {
        "agent_result": None,
        "exception_info": {
            "exception_type": "RequiredWeaveAgentsTraceError",
            "exception_message": (
                "required_trace_evidence_failure: native trace was missing"
            ),
        },
    }

    assert run_deepswe_openclaw.is_required_trace_evidence_failure(result) is True


def test_cleanup_deepswe_run_resources_is_scoped_to_run_metadata_and_job_labels(
    monkeypatch, tmp_path
):
    args = argparse.Namespace(
        output_dir=tmp_path / "output",
        nemoclaw_sandbox="nejumi-taiwan",
        nemoclaw_bin="nemoclaw",
    )
    task_dir = args.output_dir / "openclaw" / "task-a"
    task_dir.mkdir(parents=True)
    (task_dir / "openclaw_task_agent.json").write_text(
        json.dumps(
            {
                "agent_id": "tw-run-task-a",
                "nemoclaw_sandbox": "nejumi-taiwan",
                "workspace": "/sandbox/checkouts/deepswe/task-a-run",
            }
        ),
        encoding="utf-8",
    )
    unrelated_dir = args.output_dir / "openclaw" / "unrelated"
    unrelated_dir.mkdir(parents=True)
    (unrelated_dir / "openclaw_task_agent.json").write_text(
        json.dumps(
            {
                "agent_id": "japanese-production-agent",
                "nemoclaw_sandbox": "japanese-production",
            }
        ),
        encoding="utf-8",
    )
    job_dir = tmp_path / "jobs" / "this-run"
    removed_agents = []

    def run_nemoclaw_text_command(cleanup_args, command, timeout, check):
        removed_agents.append(
            {
                "sandbox": cleanup_args.nemoclaw_sandbox,
                "command": command,
                "timeout": timeout,
                "check": check,
            }
        )
        return types.SimpleNamespace(returncode=0, stdout="deleted\n", stderr="")

    fake_runner = types.SimpleNamespace(
        run_nemoclaw_text_command=run_nemoclaw_text_command,
    )
    monkeypatch.setitem(sys.modules, "run_swebench_pro_openclaw", fake_runner)
    monkeypatch.setattr(
        run_deepswe_openclaw,
        "_pier_containers_for_job",
        lambda observed_job_dir: [
            {
                "id": "container-this-run",
                "name": "task-a-main-1",
                "compose_project": "task-a",
                "config_files": str(observed_job_dir / "docker-compose-mounts.json"),
            }
        ],
    )
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return types.SimpleNamespace(returncode=0, stdout="removed\n", stderr="")

    monkeypatch.setattr(run_deepswe_openclaw.subprocess, "run", fake_run)

    report = run_deepswe_openclaw.cleanup_deepswe_run_resources(
        args,
        job_dir=job_dir,
        reason="test_failfast",
    )

    assert len(removed_agents) == 2
    assert removed_agents[0]["sandbox"] == "nejumi-taiwan"
    assert removed_agents[0]["command"][:2] == ["python3", "-c"]
    assert "base64.b64decode" in removed_agents[0]["command"][2]
    assert "\n" not in removed_agents[0]["command"][2]
    assert "\n" not in removed_agents[0]["command"][3]
    assert json.loads(removed_agents[0]["command"][-1]) == [
        "/sandbox/checkouts/deepswe/task-a-run"
    ]
    assert removed_agents[0]["timeout"] == 15
    assert removed_agents[0]["check"] is False
    assert json.loads(removed_agents[1]["command"][-1]) == ["tw-run-task-a"]
    assert removed_agents[1]["timeout"] == 60
    assert commands == [["docker", "rm", "-f", "container-this-run"]]
    assert [entry["id"] for entry in report["pier_containers"]] == [
        "container-this-run"
    ]
    assert report["sandbox_processes"]["ok"] is True
    assert report["ok"] is True
    saved = json.loads((args.output_dir / "resource_cleanup.json").read_text())
    assert saved["reason"] == "test_failfast"


def test_completed_run_cleanup_requires_success(monkeypatch, tmp_path):
    args = argparse.Namespace(output_dir=tmp_path / "output")
    job_dir = tmp_path / "jobs" / "this-run"
    calls = []

    def fake_cleanup(cleanup_args, *, job_dir, reason):
        calls.append((cleanup_args, job_dir, reason))
        return {"ok": False}

    monkeypatch.setattr(
        run_deepswe_openclaw,
        "cleanup_deepswe_run_resources",
        fake_cleanup,
    )

    with pytest.raises(RuntimeError, match="resource cleanup failed"):
        run_deepswe_openclaw.cleanup_completed_deepswe_run_resources(
            args,
            job_dir=job_dir,
        )

    assert calls == [(args, job_dir, "pier_completed")]


def test_pier_container_discovery_requires_exact_job_directory_in_compose_labels(
    monkeypatch, tmp_path
):
    job_dir = tmp_path / "jobs" / "this-run"
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[:3] == ["docker", "ps", "-aq"]:
            return types.SimpleNamespace(returncode=0, stdout="own\nother\n", stderr="")
        assert command[:2] == ["docker", "inspect"]
        return types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                [
                    {
                        "Id": "own",
                        "Name": "/own-task",
                        "Config": {
                            "Labels": {
                                "com.docker.compose.project": "own-task",
                                "com.docker.compose.project.config_files": str(
                                    job_dir / "trial" / "docker-compose-mounts.json"
                                ),
                            }
                        },
                    },
                    {
                        "Id": "other",
                        "Name": "/japanese-production",
                        "Config": {
                            "Labels": {
                                "com.docker.compose.project": "llm-leaderboard",
                                "com.docker.compose.project.config_files": (
                                    "/home/yuya/qwen3-next/llm-leaderboard/docker-compose.yaml"
                                ),
                            }
                        },
                    },
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr(run_deepswe_openclaw.subprocess, "run", fake_run)

    matches = run_deepswe_openclaw._pier_containers_for_job(job_dir)

    assert [item["id"] for item in matches] == ["own"]
    assert calls[1] == ["docker", "inspect", "own", "other"]


def test_model_truncation_trace_gap_remains_scoreable_model_failure():
    result = {
        "agent_result": {
            "metadata": {
                "openclaw": {
                    "weave_agents_required": True,
                    "weave_agents_ok": False,
                    "openclaw_disqualified_reason": "model_output_truncated",
                    "model_completion": {
                        "failure_category": "model",
                        "retryable": False,
                    },
                }
            }
        }
    }

    assert run_deepswe_openclaw.is_required_trace_evidence_failure(result) is False
    assert run_deepswe_openclaw.is_non_scoreable_provider_timeout(result) is False


def test_non_scoreable_configuration_error_detection_matches_openclaw_stderr():
    result = {
        "agent_result": {
            "metadata": {
                "openclaw": {
                    "stderr": (
                        'GatewayClientRequestError: Error: Thinking level "high" is not '
                        "supported for openai-direct/gpt-4.1-mini-2025-04-14. Use one of: off."
                    )
                }
            }
        },
        "exception_info": None,
    }

    assert run_deepswe_openclaw.is_non_scoreable_configuration_error(result) is True


def test_deepswe_pier_agent_detects_unsupported_model_thinking(monkeypatch):
    module = _load_deepswe_pier_agent_module(monkeypatch)

    assert module._metadata_has_unsupported_model_thinking(
        {
            "stderr": (
                'GatewayClientRequestError: Error: Thinking level "high" is not supported '
                "for openai-direct/gpt-4.1-mini-2025-04-14. Use one of: off."
            )
        }
    )


def test_collect_results_counts_non_scoreable_configuration_error(tmp_path):
    job_dir = tmp_path / "jobs" / "job-a"
    trial_dir = job_dir / "trial-1"
    trial_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text(json.dumps({"ok": True}) + "\n", encoding="utf-8")
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "datacurve/etree-xml-diff-patch",
                "trial_name": "trial-1",
                "agent_result": None,
                "verifier_result": None,
                "exception_info": {
                    "exception_type": "NonScoreableOpenClawConfigurationError",
                    "exception_message": (
                        "non_scoreable_configuration_error: requested OpenClaw thinking "
                        "level is unsupported by the model"
                    ),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    run_deepswe_openclaw.collect_results(
        job_dir,
        tmp_path / "out",
        command=["pier", "run"],
        elapsed=1.0,
    )

    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["exceptions"] == 1
    assert summary["non_scoreable_configuration_errors"] == 1
    assert summary["non_scoreable_provider_timeouts"] == 0
    assert summary["non_scoreable_llm_response_idle_timeouts"] == 0
    assert summary["non_scoreable_policy_blocks"] == 0


def test_collect_results_counts_non_scoreable_provider_timeout(tmp_path):
    job_dir = tmp_path / "jobs" / "job-a"
    trial_dir = job_dir / "trial-1"
    trial_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text(json.dumps({"ok": True}) + "\n", encoding="utf-8")
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "datacurve/kcp-go-multiplexed-kcp-streams",
                "trial_name": "trial-1",
                "agent_result": {
                    "metadata": {
                        "openclaw": {
                            "runtime_budget": {
                                "live": {"live_provider_timeout_count": 1}
                            }
                        }
                    }
                },
                "verifier_result": None,
                "exception_info": {
                    "exception_type": "NonScoreableOpenClawProviderTimeout",
                    "exception_message": "non_scoreable_provider_timeout: OpenClaw provider timed out",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    run_deepswe_openclaw.collect_results(
        job_dir,
        tmp_path / "out",
        command=["pier", "run"],
        elapsed=1.0,
    )

    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["exceptions"] == 1
    assert summary["non_scoreable_provider_timeouts"] == 1
    assert summary["non_scoreable_llm_response_idle_timeouts"] == 0
    assert summary["non_scoreable_policy_blocks"] == 0


def test_collect_results_counts_scoreable_llm_response_idle_timeout(tmp_path):
    job_dir = tmp_path / "jobs" / "job-a"
    trial_dir = job_dir / "trial-1"
    trial_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text(json.dumps({"ok": True}) + "\n", encoding="utf-8")
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "datacurve/kcp-go-multiplexed-kcp-streams",
                "trial_name": "trial-1",
                "agent_result": {
                    "metadata": {
                        "openclaw": {
                            "runtime_budget": {
                                "live": {
                                    "reason": "llm_response_idle_timeout",
                                    "interrupt_reason": "llm_response_idle_timeout",
                                    "exceeded_limits": ["llm_response_idle_timeout"],
                                }
                            }
                        }
                    }
                },
                "verifier_result": None,
                "exception_info": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    run_deepswe_openclaw.collect_results(
        job_dir,
        tmp_path / "out",
        command=["pier", "run"],
        elapsed=1.0,
    )

    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["exceptions"] == 0
    assert summary["non_scoreable_provider_timeouts"] == 0
    assert summary["non_scoreable_llm_response_idle_timeouts"] == 0
    assert summary["llm_response_idle_timeouts"] == 1
    assert summary["non_scoreable_policy_blocks"] == 0


def test_deepswe_patch_apply_falls_back_when_index_hashes_differ():
    source = (ROOT / "scripts" / "tools" / "deepswe_openclaw_pier_agent.py").read_text(
        encoding="utf-8"
    )

    assert "git apply --binary --index /tmp/nejumi_deepswe_model.patch" in source
    assert "git apply --binary /tmp/nejumi_deepswe_model.patch" in source
    assert '\\"apply_method\\": \\"$apply_method\\"' in source


def test_deepswe_pier_agent_uses_trial_scoped_sandbox_checkout_root(monkeypatch, tmp_path):
    module = _load_deepswe_pier_agent_module(monkeypatch)

    root_a = module._deepswe_sandbox_checkout_root(
        "expr-try-catch-errors",
        tmp_path / "trial-a" / "agent",
    )
    root_b = module._deepswe_sandbox_checkout_root(
        "expr-try-catch-errors",
        tmp_path / "trial-b" / "agent",
    )
    root_c = module._deepswe_sandbox_checkout_root(
        "boa-hierarchical-evaluation-cancellation",
        tmp_path / "trial-a" / "agent",
    )

    assert root_a.startswith("/sandbox/checkouts/deepswe/expr-try-catch-errors-")
    assert root_a != root_b
    assert root_a != root_c


def test_deepswe_preregisters_all_gateway_agents_and_restarts_once(monkeypatch, tmp_path):
    task_names_file = tmp_path / "tasks.json"
    task_names_file.write_text(json.dumps(["task-a", "task-b"]), encoding="utf-8")
    args = _args(tmp_path, task_names_file)
    calls: list[tuple[str, object]] = []

    def write_task_openclaw_config(row, checkout_dir, task_dir, registration_args):
        calls.append(
            (
                "register",
                {
                    "task": row["instance_id"],
                    "checkout": str(checkout_dir),
                    "task_dir": str(task_dir),
                    "sandbox_root": registration_args.nemoclaw_checkout_sandbox_root,
                },
            )
        )
        return f"agent-{row['instance_id']}", None

    registered_ids: list[str] = []

    def list_gateway_agents(registration_args, command, timeout):
        calls.append(
            (
                "gateway_readiness",
                {
                    "restart_enabled": registration_args.restart_gateway_after_task_agent_registration,
                    "command": command,
                    "timeout": timeout,
                },
            )
        )
        return types.SimpleNamespace(
            stdout=json.dumps({"agents": [{"id": agent_id} for agent_id in registered_ids]}),
            returncode=0,
        )

    def record_registration(row, checkout_dir, task_dir, registration_args):
        result = write_task_openclaw_config(row, checkout_dir, task_dir, registration_args)
        registered_ids.append(result[0])
        return result

    fake_runner = types.SimpleNamespace(
        safe_id=lambda value: value,
        ensure_nemoclaw_openclaw_permissions=lambda registration_args: calls.append(
            ("permissions", registration_args.nemoclaw_sandbox)
        ),
        gateway_task_agent_registration_is_reusable=lambda *args, **kwargs: False,
        write_task_openclaw_config=record_registration,
        restart_nemoclaw_gateway_after_task_agent_registration=lambda registration_args, label: calls.append(
            (
                "restart",
                {
                    "label": label,
                    "enabled": registration_args.restart_gateway_after_task_agent_registration,
                },
            )
        ),
        run_nemoclaw_text_command=list_gateway_agents,
    )
    monkeypatch.setitem(sys.modules, "run_swebench_pro_openclaw", fake_runner)

    status = run_deepswe_openclaw.preregister_gateway_task_agents(args)

    registrations = [value for kind, value in calls if kind == "register"]
    assert status["ok"] is True
    assert status["task_count"] == 2
    assert status["registration_changes"] == 2
    assert status["gateway_restart_count"] == 1
    assert status["gateway_readiness_verified"] is True
    assert len(registrations) == 2
    assert registrations[0]["sandbox_root"] != registrations[1]["sandbox_root"]
    assert [kind for kind, _ in calls].count("restart") == 1
    restart = next(value for kind, value in calls if kind == "restart")
    assert restart == {"label": "DeepSWE batch", "enabled": True}
    readiness = next(value for kind, value in calls if kind == "gateway_readiness")
    assert readiness["restart_enabled"] is True
    assert readiness["command"][:4] == ["openclaw", "gateway", "call", "agents.list"]
    assert "--timeout" not in readiness["command"]
    assert args.restart_gateway_before_run is False


def test_deepswe_skips_gateway_restart_when_all_task_agents_are_reusable(
    monkeypatch,
    tmp_path,
):
    task_names_file = tmp_path / "tasks.json"
    task_names_file.write_text(json.dumps(["task-a", "task-b"]), encoding="utf-8")
    args = _args(tmp_path, task_names_file)
    calls: list[str] = []

    fake_runner = types.SimpleNamespace(
        safe_id=lambda value: value,
        ensure_nemoclaw_openclaw_permissions=lambda registration_args: calls.append(
            "permissions"
        ),
        gateway_task_agent_registration_is_reusable=lambda *args, **kwargs: True,
        write_task_openclaw_config=lambda row, checkout_dir, task_dir, registration_args: (
            f"agent-{row['instance_id']}",
            None,
        ),
        restart_nemoclaw_gateway_after_task_agent_registration=lambda *args, **kwargs: calls.append(
            "restart"
        ),
        run_nemoclaw_text_command=lambda *args, **kwargs: calls.append("readiness"),
    )
    monkeypatch.setitem(sys.modules, "run_swebench_pro_openclaw", fake_runner)

    status = run_deepswe_openclaw.preregister_gateway_task_agents(args)

    assert status["ok"] is True
    assert status["performed"] is False
    assert status["reason"] == "all_task_agent_registrations_reusable"
    assert status["registration_changes"] == 0
    assert status["gateway_restart_count"] == 0
    assert "restart" not in calls
    assert "readiness" not in calls
    assert args.restart_gateway_before_run is False


def test_deepswe_pier_agent_forwards_openclaw_model_params_json(monkeypatch, tmp_path):
    module = _load_deepswe_pier_agent_module(monkeypatch)
    params_json = json.dumps({"provider": {"only": ["z-ai/fp8"], "allow_fallbacks": False}})
    overrides_json = json.dumps({"maxTokens": 4096})
    agent = module.NejumiDeepSWEOpenClawAgent(
        logs_dir=tmp_path / "logs",
        model_name="openrouter-direct/z-ai/glm-5.2",
        openclaw_model_params_json=params_json,
        openclaw_model_overrides_json=overrides_json,
    )

    args = agent._args()

    assert args.openclaw_model_params_json == params_json
    assert args.openclaw_model_overrides_json == overrides_json


def test_deepswe_pier_agent_sanitizes_branch_commit_runner_directive(monkeypatch):
    module = _load_deepswe_pier_agent_module(monkeypatch)
    instruction = "\n".join(
        [
            "Implement the feature.",
            "IMPORTANT: Please work on this in a new branch from main and commit everything when you are done.",
            "Keep the public API stable.",
        ]
    )

    sanitized = module._sanitize_deepswe_instruction(instruction)

    assert "Implement the feature." in sanitized
    assert "Keep the public API stable." in sanitized
    assert "new branch" not in sanitized
    assert "commit everything" not in sanitized


def test_deepswe_pier_agent_requires_nonempty_working_tree_diff(monkeypatch, tmp_path):
    module = _load_deepswe_pier_agent_module(monkeypatch)
    agent = module.NejumiDeepSWEOpenClawAgent(
        logs_dir=tmp_path / "logs",
        model_name="openrouter-direct/z-ai/glm-5.2",
    )

    row = agent._row(
        "example-task",
        {"metadata": {"language": "python"}, "environment": {}},
        "Implement the feature.",
    )

    requirements = row["requirements"]
    assert "Do not stop after only describing a plan" in requirements
    assert "`git diff HEAD --check`" in requirements
    assert "`git diff HEAD --stat`" in requirements
    assert "continue editing instead of finishing" in requirements
    assert "harness scores all changes relative to HEAD" in requirements
