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
        agent_timeout_multiplier=None,
        disable_verification=False,
        delete=True,
        quiet=False,
    )


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
    assert kwargs["no_local"] == "true"
    assert kwargs["use_task_agent"] == "true"
    assert kwargs["verify_weave_agents"] == "true"
    assert kwargs["deny_tool"] == "web_search,browser"
    assert kwargs["deny_argument_pattern"] == r"https?://"


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


def test_deepswe_pier_agent_uses_trial_scoped_sandbox_checkout_root(monkeypatch, tmp_path):
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
