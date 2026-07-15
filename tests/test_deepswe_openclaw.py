import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "tools" / "run_deepswe_openclaw.py"
spec = importlib.util.spec_from_file_location("run_deepswe_openclaw", MODULE_PATH)
assert spec is not None and spec.loader is not None
run_deepswe_openclaw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_deepswe_openclaw)


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
    assert metadata["deepswe_patch_bytes"] == len(patch.encode("utf-8"))
    assert (tmp_path / "openclaw" / "superjson-error-stack-serialization" / "model.patch").read_text(
        encoding="utf-8"
    ) == patch


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
                "agent_result": {"metadata": {}},
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
    assert summary["scored_trials"] == 1
    assert summary["pass_at_1"] == 0.0


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
