import importlib.util
import json
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


def test_swebench_main_rejects_weave_sidecar_before_dataset_read(tmp_path, monkeypatch):
    module = load_script_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_swebench_pro_openclaw.py",
            "--dataset-jsonl",
            str(tmp_path / "missing.jsonl"),
            "--weave-sidecar",
        ],
    )

    with pytest.raises(SystemExit, match="Weave sidecar logging is disabled"):
        module.main()


def sample_row() -> dict:
    return {
        "repo": "example/repo",
        "instance_id": "example__repo-1",
        "base_commit": "0" * 40,
        "problem_statement": "Fix the bug.",
        "requirements": "Keep the public API stable.",
        "interface": "Function: fix_bug",
        "repo_language": "Python",
        "issue_specificity": "specific",
        "issue_categories": ["bug"],
        "patch": (
            "diff --git a/src/example.py b/src/example.py\n"
            "--- a/src/example.py\n"
            "+++ b/src/example.py\n"
        ),
    }


def command_flag_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1] if flag in command else ""


def command_flag_values(command: list[str], flag: str) -> list[str]:
    values = []
    for index, token in enumerate(command[:-1]):
        if token == flag:
            values.append(command[index + 1])
    return values


def protocol_sidecar_identity(module, command: list[str], instance_id: str) -> dict:
    prompt_text = Path(command_flag_value(command, "--prompt-file")).read_text(encoding="utf-8")
    sidecar = {
        "metadata": {
            "task_id": instance_id,
            "prompt_hash": module.sha256_text(prompt_text),
            "model_id": command_flag_value(command, "--model"),
            "openclaw_config_source": command_flag_value(command, "--openclaw-config-source"),
        },
        "tool_policy": {
            "deny_tools": command_flag_values(command, "--deny-tool"),
            "deny_argument_patterns": command_flag_values(command, "--deny-argument-pattern"),
        },
    }
    if "--nemoclaw-sandbox" in command:
        sidecar["conversation_order"] = {"ok": True, "checked": True}
        sidecar["nemoclaw_session_audit"] = {"required": True, "ok": True}
    return sidecar


def test_agentic_prompt_does_not_embed_code_context():
    module = load_module(REPO_ROOT / "scripts" / "data_uploader" / "prepare_swebench_pro.py")
    row = module.enrich_row(sample_row(), object())
    assert "agentic_prompt" in row
    assert row["text"] == row["agentic_prompt"]
    assert "allowed_repository_context" not in row["text"]
    assert "<file path=" not in row["text"]
    assert "repo_files" not in row


def test_swebench_list_normalization_accepts_python_literal_strings():
    module = load_module(REPO_ROOT / "scripts" / "data_uploader" / "prepare_swebench_pro.py")

    assert module.as_list("['Test_a', 'Test_b/subcase']") == ["Test_a", "Test_b/subcase"]
    assert module.as_list('["Test_a", "Test_b/subcase"]') == ["Test_a", "Test_b/subcase"]


def test_swebench_list_normalization_flattens_stringified_singleton_list():
    module = load_module(REPO_ROOT / "scripts" / "data_uploader" / "prepare_swebench_pro.py")

    row = module.normalize_row(
        {
            "FAIL_TO_PASS": "['Test_shouldDownload', 'Test_shouldDownload/schema_version_mismatch']",
            "PASS_TO_PASS": ["Test_shouldDownload/no_db_file"],
            "selected_test_files_to_run": [
                "['Test_shouldDownload', 'Test_shouldDownload/schema_version_mismatch']"
            ],
        }
    )

    assert row["fail_to_pass"] == [
        "Test_shouldDownload",
        "Test_shouldDownload/schema_version_mismatch",
    ]
    assert row["pass_to_pass"] == ["Test_shouldDownload/no_db_file"]
    assert row["selected_test_files_to_run"] == [
        "Test_shouldDownload",
        "Test_shouldDownload/schema_version_mismatch",
    ]


def test_swebench_repo_tree_metrics_compute_static_cost_proxy():
    module = load_module(REPO_ROOT / "scripts" / "data_uploader" / "prepare_swebench_pro.py")
    row = sample_row()
    row["selected_test_files_to_run"] = ["src/example.py::test_bug", "missing.py"]
    tree_files = module.parse_ls_tree(
        "100644 blob abc 100\tsrc/example.py\0"
        "100644 blob def 200\tREADME.md\0"
        "100644 blob ghi 300\tassets/logo.png\0"
    )

    metrics = module.repo_tree_metrics(row, tree_files)

    assert metrics["metadata_schema_version"] == 1
    assert metrics["measurement_method"] == "git-ls-tree-v1"
    assert metrics["tracked_file_count"] == 3
    assert metrics["tracked_total_bytes"] == 600
    assert metrics["source_file_count"] == 2
    assert metrics["source_total_bytes"] == 300
    assert metrics["selected_test_file_count"] == 1
    assert metrics["selected_test_total_bytes"] == 100
    assert metrics["static_cost_proxy_score"] > 0


def test_swebench_attach_repo_metadata_rejects_mismatched_base_commit():
    module = load_module(REPO_ROOT / "scripts" / "data_uploader" / "prepare_swebench_pro.py")
    row = sample_row()
    metadata = module.repo_tree_metrics(row, [])
    metadata["base_commit"] = "1" * 40

    try:
        module.attach_repo_metadata([row], {row["instance_id"]: metadata})
    except ValueError as exc:
        assert "Repo metadata mismatch" in str(exc)
    else:
        raise AssertionError("metadata mismatch was not rejected")


def test_swebench_compact_sample_filters_high_cost_rows_and_caps_repo():
    module = load_module(REPO_ROOT / "scripts" / "data_uploader" / "prepare_swebench_pro.py")
    rows = []
    for idx in range(12):
        repo = "large/repo" if idx >= 8 else f"small/repo{idx % 4}"
        row = {
            **sample_row(),
            "repo": repo,
            "instance_id": f"task-{idx}",
            "repo_language": "python" if idx % 2 else "go",
            "issue_specificity": "specific",
            "issue_categories": ["bug"],
            "static_cost_proxy_score": float(idx + 1),
            "repo_static_cost": {
                "tracked_file_count": idx + 1,
                "tracked_total_bytes": (idx + 1) * 100,
                "static_cost_proxy_score": float(idx + 1),
            },
        }
        rows.append(row)
    rows = module.add_cost_percentiles(rows)

    sampled, metadata = module.compact_sample(
        rows,
        size=4,
        seed=45,
        max_cost_percentile=0.70,
        max_per_repo=1,
    )

    assert metadata["candidate_pool_size"] == 8
    assert metadata["max_per_repo"] == 1
    assert len(sampled) == 4
    assert all(float(row["static_cost_proxy_percentile"]) <= 0.70 for row in sampled)
    assert len({row["repo"] for row in sampled}) == 4
    assert all(row["repo"] != "large/repo" for row in sampled)


def test_swebench_materialize_writes_compact_subsets(tmp_path, monkeypatch):
    module = load_module(REPO_ROOT / "scripts" / "data_uploader" / "prepare_swebench_pro.py")
    rows = []
    records = []
    for idx in range(8):
        row = {
            **sample_row(),
            "repo": f"example/repo{idx % 4}",
            "instance_id": f"task-{idx}",
            "base_commit": f"{idx:040d}",
            "repo_language": "python" if idx % 2 else "go",
            "issue_specificity": "specific",
            "issue_categories": ["bug"],
            "selected_test_files_to_run": [f"src/file{idx}.py::test_case"],
        }
        rows.append(row)
        records.append(
            module.repo_tree_metrics(
                row,
                module.parse_ls_tree(
                    f"100644 blob abc {100 + idx}\tsrc/file{idx}.py\0"
                    f"100644 blob def {200 + idx}\tREADME.md\0"
                ),
            )
        )
    metadata_path = tmp_path / "metadata.jsonl"
    module.write_jsonl(metadata_path, records)
    monkeypatch.setattr(module, "load_rows", lambda dataset, split, limit: rows)

    args = SimpleNamespace(
        dataset="dummy",
        split="test",
        limit=None,
        output_dir=tmp_path / "out",
        dataset_dir_name="swebench_pro_public",
        repo_metadata_jsonl=metadata_path,
        collect_repo_metadata=False,
        metadata_mirror_root=tmp_path / "mirrors",
        metadata_refresh=False,
        compact_leaderboard_size=4,
        compact_pilot_size=2,
        compact_max_cost_percentile=0.75,
        compact_max_per_repo=2,
        compact_max_tracked_total_bytes=None,
        compact_max_tracked_file_count=None,
        leaderboard_size=4,
        smoke_size=2,
        seed=45,
    )

    artifact_root = module.materialize(args)
    manifest = json.loads((artifact_root / "manifest.json").read_text(encoding="utf-8"))

    assert (artifact_root / "subsets" / "leaderboard_compact_4.jsonl").exists()
    assert (artifact_root / "subsets" / "leaderboard_compact_2.jsonl").exists()
    assert (artifact_root / "repo_metadata.jsonl").exists()
    assert manifest["compact_subset_sizes"] == {
        "leaderboard_compact_4": 4,
        "leaderboard_compact_2": 2,
    }
    assert manifest["repo_static_cost_metadata"]["included"] is True
    assert manifest["repo_static_cost_metadata"]["record_count"] == 8


def test_openclaw_prompt_ignores_embedded_text_fields():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    row = sample_row()
    row["text"] = "<file path=\"src/example.py\">def bug(): pass</file>"
    row["allow_text"] = "<allowed_repository_context>secret code</allowed_repository_context>"

    prompt = module.build_prompt(row)
    assert "Fix the bug." in prompt
    assert "<file path=" not in prompt
    assert "allowed_repository_context" not in prompt
    assert "secret code" not in prompt
    assert "local shell execution" in prompt
    assert "Do not use web search" in prompt


def test_swebench_sidecar_path_is_attempt_scoped(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    attempt_dir = tmp_path / "openclaw_attempts" / "attempt-1"

    path = module.task_sidecar_path(attempt_dir, "example__repo-1")

    assert path == attempt_dir / "agentic_swe" / "example__repo-1" / "openclaw_result.json"


def test_swebench_nemoclaw_task_agent_config_is_sandbox_visible(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    row = sample_row()
    checkout_dir = tmp_path / "checkout-example"
    checkout_dir.mkdir()
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    template = tmp_path / "openclaw_template.json"
    template.write_text('{"agents": {"list": []}}\n', encoding="utf-8")
    args = SimpleNamespace(
        agent="fallback-agent",
        deny_argument_pattern=None,
        deny_tool=None,
        no_local=False,
        nemoclaw_checkout_sandbox_root="/sandbox/checkouts",
        nemoclaw_openclaw_config_path="/sandbox/.openclaw/openclaw.json",
        nemoclaw_sandbox="nejumi-taiwan",
        openclaw_config_template=template,
        openclaw_tool_profile="coding",
        task_agent_prefix="tw-swe",
        use_task_agent=True,
    )

    agent_id, config_path = module.write_task_openclaw_config(row, checkout_dir, task_dir, args)

    expected_sandbox_config = (
        Path("/sandbox/checkouts")
        / checkout_dir.name
        / ".nejumi_openclaw"
        / "openclaw_config.json"
    )
    host_config = checkout_dir / ".nejumi_openclaw" / "openclaw_config.json"
    assert config_path == expected_sandbox_config
    assert host_config.exists()
    config = json.loads(host_config.read_text(encoding="utf-8"))
    [agent] = config["agents"]["list"]
    assert agent["id"] == agent_id
    assert agent["workspace"] == f"/sandbox/checkouts/{checkout_dir.name}"
    assert agent["agentDir"] == f"/sandbox/checkouts/{checkout_dir.name}/.nejumi_openclaw/agent_state"
    assert config["tools"]["toolSearch"] is False
    assert config["tools"]["web"]["fetch"]["enabled"] is False
    assert "browser" not in config["tools"]
    metadata = json.loads((task_dir / "openclaw_task_agent.json").read_text(encoding="utf-8"))
    assert metadata["host_config_path"] == str(host_config)
    assert metadata["config_path"] == str(expected_sandbox_config)
    assert metadata["nemoclaw_sandbox"] == "nejumi-taiwan"


def test_swebench_registers_nemoclaw_gateway_task_agent_when_no_local(tmp_path, monkeypatch):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    row = sample_row()
    checkout_dir = tmp_path / "checkout-example"
    checkout_dir.mkdir()
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    calls = []

    def fake_run_nemoclaw_text_command(
        args,
        command,
        input_text=None,
        timeout=60,
        check=True,
        workdir="/sandbox",
    ):
        calls.append(
            {
                "command": command,
                "input_text": input_text,
                "timeout": timeout,
                "check": check,
                "workdir": workdir,
            }
        )
        return subprocess.CompletedProcess(command, 0, stdout='{"ok": true}\n', stderr="")

    monkeypatch.setattr(module, "run_nemoclaw_text_command", fake_run_nemoclaw_text_command)
    args = SimpleNamespace(
        agent="fallback-agent",
        deny_argument_pattern=None,
        deny_tool=["web_search"],
        dry_run=False,
        model="openai-direct/gpt-4.1-mini-2025-04-14",
        no_local=True,
        nemoclaw_checkout_sandbox_root="/sandbox/checkouts",
        nemoclaw_openclaw_config_path="/sandbox/.openclaw/openclaw.json",
        nemoclaw_sandbox="nejumi-taiwan",
        openclaw_config_template=None,
        openclaw_tool_profile="coding",
        task_agent_prefix="tw-swe",
        use_task_agent=True,
    )

    agent_id, config_path = module.write_task_openclaw_config(row, checkout_dir, task_dir, args)

    assert config_path is None
    assert calls
    command = calls[0]["command"]
    assert command[0] == "env"
    script_b64 = command[1].split("=", 1)[1]
    script = module.base64.b64decode(script_b64).decode("utf-8")
    assert "openclaw agents add" in script
    assert command[2:5] == [
        "bash",
        "-lc",
        'printf %s "$OPENCLAW_REGISTER_SCRIPT_B64" | base64 -d | bash -s -- "$@"',
    ]
    assert command[5] == "register-task-agent"
    assert command[6] == agent_id
    assert command[9] == "openai-direct/gpt-4.1-mini-2025-04-14"
    metadata = json.loads((task_dir / "openclaw_task_agent.json").read_text(encoding="utf-8"))
    assert metadata["config_path"] == "/sandbox/.openclaw/openclaw.json"
    assert metadata["gateway_registered"]["ok"] is True
    assert module.task_live_session_dir(checkout_dir, task_dir, args) is None
    assert (
        module.task_live_sandbox_session_dir(checkout_dir, args, agent_id)
        == f"/sandbox/.openclaw/agents/{agent_id}/sessions"
    )
    module._REGISTERED_NEMOCLAW_GATEWAY_AGENTS.clear()


def test_swebench_host_task_agent_config_keeps_task_dir_state(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    row = sample_row()
    checkout_dir = tmp_path / "checkout"
    checkout_dir.mkdir()
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    template = tmp_path / "openclaw_template.json"
    template.write_text('{"agents": {"list": []}}\n', encoding="utf-8")
    args = SimpleNamespace(
        agent="fallback-agent",
        deny_argument_pattern=None,
        deny_tool=None,
        no_local=False,
        nemoclaw_sandbox=None,
        openclaw_config_template=template,
        openclaw_tool_profile="coding",
        task_agent_prefix="tw-swe",
        use_task_agent=True,
    )

    _, config_path = module.write_task_openclaw_config(row, checkout_dir, task_dir, args)

    assert config_path == task_dir / "openclaw_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    [agent] = config["agents"]["list"]
    assert agent["workspace"] == str(checkout_dir.resolve())
    assert agent["agentDir"] == str(task_dir / "openclaw_agent_state")
    assert not (checkout_dir / ".nejumi_openclaw").exists()


def test_swebench_weave_sidecar_failure_is_not_patchable_success():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")

    assert module.is_weave_sidecar_failure({"returncode": 0, "weave_sidecar": {"ok": False}})
    assert not module.is_weave_sidecar_failure({"returncode": 1, "weave_sidecar": {"ok": False}})


def test_openclaw_protocol_runtime_budget_detects_overages():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py")
    sidecar = {
        "stdout_json": {
            "meta": {
                "agentMeta": {
                    "usage": {"input": 1_000_001, "output": 10, "cacheRead": 0}
                }
            }
        },
        "tool_call_count": 61,
    }
    args = SimpleNamespace(max_input_tokens=1_000_000, max_tool_calls=60)

    status = module.runtime_budget_status(sidecar, args)

    assert status["ok"] is False
    assert status["limits"] == {
        "max_input_tokens": 1_000_000,
        "max_tool_calls": 60,
        "max_agent_turns": None,
    }
    assert {violation["type"] for violation in status["violations"]} == {
        "max_input_tokens_exceeded",
        "max_tool_calls_exceeded",
    }


def test_swebench_runtime_budget_exceeded_returns_disqualified_metadata(tmp_path, monkeypatch):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    row = sample_row()
    checkout_dir = tmp_path / "checkout"
    checkout_dir.mkdir()
    task_dir = tmp_path / "task"

    def fake_run_command(command, cwd=None, timeout=None, check=True):
        output_dir = Path(command[command.index("--output-dir") + 1])
        sidecar_path = module.task_sidecar_path(output_dir, row["instance_id"])
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(
            json.dumps(
                {
                    **protocol_sidecar_identity(module, command, row["instance_id"]),
                    "returncode": 0,
                    "stderr": "Runtime budget exceeded",
                    "tool_policy_ok": True,
                    "tool_policy_violations": [],
                    "tool_call_count": 61,
                    "tool_error_count": 0,
                    "runtime_budget": {
                        "ok": False,
                        "enforced": True,
                        "limits": {"max_input_tokens": 1_000_000, "max_tool_calls": 60},
                        "observed": {"input_tokens": 1_000_001, "tool_call_count": 61},
                        "violations": [
                            {
                                "type": "max_input_tokens_exceeded",
                                "observed": 1_000_001,
                                "limit": 1_000_000,
                            },
                            {
                                "type": "max_tool_calls_exceeded",
                                "observed": 61,
                                "limit": 60,
                            },
                        ],
                    },
                    "weave_sidecar": {"ok": True},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 1, stdout=str(sidecar_path), stderr="Runtime budget exceeded")

    monkeypatch.setattr(module, "run_command", fake_run_command)
    monkeypatch.setattr(module, "ensure_nemoclaw_checkout_ready", lambda checkout_dir, task_dir, args: None)
    args = SimpleNamespace(
        agent="test-agent",
        allow_failed_preflight=False,
        deny_argument_pattern=None,
        deny_tool=None,
        dry_run=False,
        max_input_tokens=1_000_000,
        max_tool_calls=60,
        model="openai-direct/example-model",
        no_local=False,
        openclaw_max_attempts=1,
        openclaw_retry_base_seconds=0,
        openclaw_timeout=30,
        profile=None,
        session_prefix=None,
        thinking="high",
        use_task_agent=False,
        weave_sidecar=True,
        weave_sidecar_strict=False,
    )

    metadata = module.run_openclaw_for_task(row, checkout_dir, task_dir, args)

    assert metadata["openclaw_disqualified_reason"] == "runtime_budget_exceeded"
    assert metadata["runtime_budget"]["ok"] is False
    assert module.should_force_empty_patch(metadata)


def test_swebench_nemoclaw_run_forwards_sandbox_command_args(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_RUN_ID", "twcanary-swe-run")
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    row = sample_row()
    checkout_dir = tmp_path / "checkout-example"
    checkout_dir.mkdir()
    task_dir = tmp_path / "task"
    template = tmp_path / "openclaw_template.json"
    template.write_text('{"agents": {"list": []}}\n', encoding="utf-8")
    captured_command = []

    def fake_run_command(command, cwd=None, timeout=None, check=True):
        captured_command[:] = command
        output_dir = Path(command[command.index("--output-dir") + 1])
        sidecar_path = module.task_sidecar_path(output_dir, row["instance_id"])
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(
            json.dumps(
                {
                    **protocol_sidecar_identity(module, command, row["instance_id"]),
                    "returncode": 0,
                    "tool_policy_ok": True,
                    "tool_policy_violations": [],
                    "tool_call_count": 2,
                    "tool_error_count": 0,
                    "runtime_budget": {"ok": True},
                    "weave_sidecar": {"ok": True},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout=str(sidecar_path), stderr="")

    monkeypatch.setattr(module, "run_command", fake_run_command)
    monkeypatch.setattr(module, "ensure_nemoclaw_checkout_ready", lambda checkout_dir, task_dir, args: None)
    args = SimpleNamespace(
        agent="fallback-agent",
        allow_failed_preflight=False,
        deny_argument_pattern=None,
        deny_tool=None,
        dry_run=False,
        max_input_tokens=1_000_000,
        max_tool_calls=60,
        model="openai-direct/gpt-4.1-mini-2025-04-14",
        no_local=False,
        nemoclaw_bin="nemoclaw",
        nemoclaw_checkout_sandbox_root="/sandbox/checkouts",
        nemoclaw_openclaw_config_path="/sandbox/.openclaw/openclaw.json",
        nemoclaw_sandbox="nejumi-taiwan",
        nemoclaw_workdir=None,
        openclaw_config_template=template,
        openclaw_max_attempts=1,
        openclaw_retry_base_seconds=0,
        openclaw_timeout=30,
        openclaw_tool_profile="coding",
        profile=None,
        session_prefix=None,
        task_agent_prefix="tw-swe",
        thinking="off",
        use_task_agent=True,
        weave_sidecar=False,
        weave_sidecar_strict=False,
    )

    metadata = module.run_openclaw_for_task(row, checkout_dir, task_dir, args)

    expected_checkout = f"/sandbox/checkouts/{checkout_dir.name}"
    expected_config = f"{expected_checkout}/.nejumi_openclaw/openclaw_config.json"
    assert metadata["nemoclaw_sandbox"] == "nejumi-taiwan"
    assert metadata["nemoclaw_workdir"] == expected_checkout
    assert captured_command[captured_command.index("--nemoclaw-sandbox") + 1] == "nejumi-taiwan"
    assert captured_command[captured_command.index("--nemoclaw-workdir") + 1] == expected_checkout
    assert captured_command[captured_command.index("--openclaw-config-path") + 1] == expected_config
    assert captured_command[captured_command.index("--openclaw-config-source") + 1] == str(template)
    assert captured_command[captured_command.index("--live-session-dir") + 1] == str(
        checkout_dir / ".nejumi_openclaw" / "agent_state" / "sessions"
    )
    session_key = captured_command[captured_command.index("--session-key") + 1]
    assert session_key.startswith(f"twcanary-swe-run:swebench-pro:{row['instance_id']}:")


def test_swebench_nemoclaw_copy_mode_forwards_live_sandbox_session_dir(tmp_path, monkeypatch):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    row = sample_row()
    checkout_dir = tmp_path / "checkout-copy"
    checkout_dir.mkdir()
    task_dir = tmp_path / "task"
    template = tmp_path / "openclaw_template.json"
    template.write_text('{"agents": {"list": []}}\n', encoding="utf-8")
    captured_command = []

    def fake_run_command(command, cwd=None, timeout=None, check=True):
        captured_command[:] = command
        output_dir = Path(command[command.index("--output-dir") + 1])
        sidecar_path = module.task_sidecar_path(output_dir, row["instance_id"])
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(
            json.dumps(
                {
                    **protocol_sidecar_identity(module, command, row["instance_id"]),
                    "returncode": 0,
                    "tool_policy_ok": True,
                    "tool_policy_violations": [],
                    "tool_call_count": 2,
                    "tool_error_count": 0,
                    "runtime_budget": {"ok": True},
                    "weave_sidecar": {"ok": True},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout=str(sidecar_path), stderr="")

    monkeypatch.setattr(module, "run_command", fake_run_command)
    monkeypatch.setattr(module, "ensure_nemoclaw_checkout_ready", lambda checkout_dir, task_dir, args: None)
    monkeypatch.setattr(
        module,
        "run_nemoclaw_text_command",
        lambda args, command, input_text=None, timeout=60, check=True, workdir="/sandbox": subprocess.CompletedProcess(
            command, 0, stdout="", stderr=""
        ),
    )
    args = SimpleNamespace(
        agent="fallback-agent",
        allow_failed_preflight=False,
        deny_argument_pattern=None,
        deny_tool=None,
        dry_run=False,
        max_input_tokens=1_000_000,
        max_tool_calls=60,
        model="openai-direct/gpt-4.1-mini-2025-04-14",
        no_local=False,
        nemoclaw_bin="nemoclaw",
        nemoclaw_checkout_sandbox_root="/sandbox/checkouts",
        nemoclaw_checkout_transfer_mode="copy",
        nemoclaw_openclaw_config_path="/sandbox/.openclaw/openclaw.json",
        nemoclaw_sandbox="nejumi-taiwan",
        nemoclaw_workdir=None,
        openclaw_config_template=template,
        openclaw_max_attempts=1,
        openclaw_retry_base_seconds=0,
        openclaw_timeout=30,
        openclaw_tool_profile="coding",
        profile=None,
        session_prefix=None,
        task_agent_prefix="tw-swe",
        thinking="off",
        use_task_agent=True,
        weave_sidecar=False,
        weave_sidecar_strict=False,
    )

    metadata = module.run_openclaw_for_task(row, checkout_dir, task_dir, args)

    expected_checkout = f"/sandbox/checkouts/{checkout_dir.name}"
    assert metadata["nemoclaw_sandbox"] == "nejumi-taiwan"
    assert "--live-session-dir" not in captured_command
    live_index = captured_command.index("--live-sandbox-session-dir")
    assert (
        captured_command[live_index + 1]
        == f"{expected_checkout}/.nejumi_openclaw/agent_state/sessions"
    )


def test_swebench_rejects_sidecar_config_source_mismatch(tmp_path, monkeypatch):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    row = sample_row()
    checkout_dir = tmp_path / "checkout"
    checkout_dir.mkdir()
    task_dir = tmp_path / "task"

    def fake_run_command(command, cwd=None, timeout=None, check=True):
        output_dir = Path(command[command.index("--output-dir") + 1])
        sidecar_path = module.task_sidecar_path(output_dir, row["instance_id"])
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **protocol_sidecar_identity(module, command, row["instance_id"]),
            "returncode": 0,
            "tool_policy_ok": True,
            "tool_policy_violations": [],
            "tool_call_count": 1,
            "tool_error_count": 0,
            "runtime_budget": {"ok": True},
            "weave_sidecar": {"ok": True},
        }
        payload["metadata"]["openclaw_config_source"] = "/sandbox/other-openclaw.json"
        sidecar_path.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout=str(sidecar_path), stderr="")

    monkeypatch.setattr(module, "run_command", fake_run_command)
    monkeypatch.setattr(module, "ensure_nemoclaw_checkout_ready", lambda checkout_dir, task_dir, args: None)
    args = SimpleNamespace(
        agent="test-agent",
        allow_failed_preflight=False,
        deny_argument_pattern=None,
        deny_tool=None,
        dry_run=False,
        max_input_tokens=1_000_000,
        max_tool_calls=60,
        model="openai-direct/example-model",
        no_local=False,
        openclaw_max_attempts=1,
        openclaw_retry_base_seconds=0,
        openclaw_timeout=30,
        profile=None,
        session_prefix=None,
        thinking="high",
        use_task_agent=False,
        weave_sidecar=False,
        weave_sidecar_strict=False,
    )

    with pytest.raises(RuntimeError, match="OpenClaw sidecar metadata mismatch"):
        module.run_openclaw_for_task(row, checkout_dir, task_dir, args)


def test_swebench_rejects_required_nemoclaw_audit_failure(tmp_path, monkeypatch):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    row = sample_row()
    checkout_dir = tmp_path / "checkout"
    checkout_dir.mkdir()
    task_dir = tmp_path / "task"

    def fake_run_command(command, cwd=None, timeout=None, check=True):
        output_dir = Path(command[command.index("--output-dir") + 1])
        sidecar_path = module.task_sidecar_path(output_dir, row["instance_id"])
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(
            json.dumps(
                {
                    **protocol_sidecar_identity(module, command, row["instance_id"]),
                    "returncode": 0,
                    "tool_policy_ok": True,
                    "tool_policy_violations": [],
                    "tool_call_count": 1,
                    "tool_error_count": 0,
                    "runtime_budget": {"ok": True},
                    "weave_sidecar": {"ok": True},
                    "nemoclaw_session_audit": {"required": True, "ok": False},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout=str(sidecar_path), stderr="")

    monkeypatch.setattr(module, "run_command", fake_run_command)
    monkeypatch.setattr(module, "ensure_nemoclaw_checkout_ready", lambda checkout_dir, task_dir, args: None)
    args = SimpleNamespace(
        agent="test-agent",
        allow_failed_preflight=False,
        deny_argument_pattern=None,
        deny_tool=None,
        dry_run=False,
        max_input_tokens=1_000_000,
        max_tool_calls=60,
        model="openai-direct/example-model",
        no_local=False,
        nemoclaw_bin="nemoclaw",
        nemoclaw_checkout_sandbox_root="/sandbox/checkouts",
        nemoclaw_checkout_transfer_mode="copy",
        nemoclaw_sandbox="nejumi-taiwan",
        nemoclaw_workdir=None,
        openclaw_max_attempts=1,
        openclaw_retry_base_seconds=0,
        openclaw_timeout=30,
        profile=None,
        session_prefix=None,
        thinking="high",
        use_task_agent=False,
        weave_sidecar=False,
        weave_sidecar_strict=False,
    )

    with pytest.raises(RuntimeError, match="OpenClaw NeMoClaw session audit mismatch"):
        module.run_openclaw_for_task(row, checkout_dir, task_dir, args)


def test_swebench_nemoclaw_copy_mode_syncs_when_checkout_not_visible(tmp_path, monkeypatch):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    checkout_dir = tmp_path / "checkout-example"
    checkout_dir.mkdir()
    task_dir = tmp_path / "task"
    args = SimpleNamespace(
        nemoclaw_sandbox="nejumi-taiwan",
        nemoclaw_checkout_sandbox_root="/sandbox/checkouts",
        nemoclaw_checkout_transfer_mode="copy",
    )

    monkeypatch.setattr(module, "nemoclaw_checkout_visible", lambda checkout, parsed_args: False)
    monkeypatch.setattr(
        module,
        "sync_checkout_to_nemoclaw_copy",
        lambda checkout, task, parsed_args: {
            "mode": "copy",
            "sandbox_checkout_dir": str(module.sandbox_checkout_dir(checkout, parsed_args)),
        },
    )

    metadata = module.ensure_nemoclaw_checkout_ready(checkout_dir, task_dir, args)

    assert metadata == {
        "mode": "copy",
        "sandbox_checkout_dir": f"/sandbox/checkouts/{checkout_dir.name}",
    }


def test_swebench_nemoclaw_copy_mode_writes_config_to_sandbox(tmp_path, monkeypatch):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    row = sample_row()
    checkout_dir = tmp_path / "checkout-example"
    checkout_dir.mkdir()
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    template = tmp_path / "openclaw_template.json"
    template.write_text('{"agents": {"list": []}}\n', encoding="utf-8")
    writes = []

    def fake_run_nemoclaw_text_command(args, command, input_text=None, timeout=60, check=True, workdir="/sandbox"):
        writes.append({"command": command, "input_text": input_text, "workdir": workdir})
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(module, "run_nemoclaw_text_command", fake_run_nemoclaw_text_command)
    args = SimpleNamespace(
        agent="fallback-agent",
        deny_argument_pattern=None,
        deny_tool=None,
        no_local=False,
        nemoclaw_checkout_sandbox_root="/sandbox/checkouts",
        nemoclaw_checkout_transfer_mode="copy",
        nemoclaw_sandbox="nejumi-taiwan",
        openclaw_config_template=template,
        openclaw_tool_profile="coding",
        task_agent_prefix="tw-swe",
        use_task_agent=True,
    )

    _, config_path = module.write_task_openclaw_config(row, checkout_dir, task_dir, args)

    assert str(config_path) == f"/sandbox/checkouts/{checkout_dir.name}/.nejumi_openclaw/openclaw_config.json"
    assert writes
    sandbox_write = writes[-1]
    assert sandbox_write["command"][-1] == str(config_path)
    assert '"toolSearch": false' in sandbox_write["input_text"]
    sandbox_config = json.loads(sandbox_write["input_text"])
    assert "browser" not in sandbox_config["tools"]


def test_swebench_nemoclaw_copy_mode_defaults_to_sandbox_checkout_root(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    checkout_dir = tmp_path / "checkout-example"
    checkout_dir.mkdir()
    args = SimpleNamespace(
        nemoclaw_checkout_sandbox_root=None,
        nemoclaw_checkout_transfer_mode="copy",
        nemoclaw_sandbox="nejumi-taiwan",
    )

    assert (
        str(module.sandbox_checkout_dir(checkout_dir, args))
        == f"/sandbox/checkouts/{checkout_dir.name}"
    )


def test_swebench_main_copy_mode_captures_patch_in_sandbox(tmp_path, monkeypatch):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    row = sample_row()
    row["instance_id"] = "copy-mode-task"
    captured = {"sandbox_patch": False}
    args = SimpleNamespace(
        dataset_jsonl=tmp_path / "dataset.jsonl",
        output_dir=tmp_path / "out",
        checkout_root=tmp_path / "checkouts",
        prefix="openclaw",
        instance_id=None,
        limit=None,
        redo=True,
        dry_run=False,
        no_reset=False,
        skip_agent=False,
        model="inference/example",
        thinking="off",
        deny_tool=None,
        deny_argument_pattern=None,
        max_input_tokens=1_000_000,
        max_tool_calls=60,
        nemoclaw_sandbox="nejumi-taiwan",
        nemoclaw_checkout_sandbox_root="/sandbox/checkouts",
        nemoclaw_checkout_transfer_mode="copy",
    )

    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "read_jsonl", lambda path: [row])
    monkeypatch.setattr(module, "prepare_checkout", lambda task_row, checkout_root, reset: tmp_path / "checkout")
    (tmp_path / "checkout").mkdir()
    monkeypatch.setattr(
        module,
        "run_openclaw_for_task",
        lambda task_row, checkout_dir, task_dir, parsed_args: {
            "nemoclaw_checkout_transfer": {"mode": "copy"},
            "tool_policy_ok": True,
        },
    )

    def fake_capture_patch_nemoclaw(checkout_dir, parsed_args, excluded_paths=None):
        captured["sandbox_patch"] = True
        return "diff --git a/file b/file\n"

    monkeypatch.setattr(module, "capture_patch_nemoclaw", fake_capture_patch_nemoclaw)
    monkeypatch.setattr(module, "capture_patch", lambda checkout_dir, excluded_paths=None: "")

    module.main()

    assert captured["sandbox_patch"] is True
    patches = json.loads((args.output_dir / "patches.json").read_text(encoding="utf-8"))
    assert patches[0]["patch"].startswith("diff --git")


def test_swebench_runtime_budget_summary_uses_configured_caps_for_dry_run():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    args = SimpleNamespace(max_input_tokens=1_000_000, max_tool_calls=60)

    summary = module.runtime_budget_summary([], args)

    assert summary == {
        "max_input_tokens": 1_000_000,
        "max_tool_calls": 60,
        "max_agent_turns": None,
    }


def test_swebench_transient_openclaw_failure_detects_provider_timeout():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    completed = subprocess.CompletedProcess(
        ["cmd"],
        1,
        stdout="",
        stderr="FailoverError: LLM request timed out. rawError=terminated",
    )

    assert module.is_transient_openclaw_failure(completed, None)


def test_swebench_transient_openclaw_failure_detects_provider_sse_rate_limit():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    completed = subprocess.CompletedProcess(["cmd"], 1, stdout="", stderr="")
    sidecar = {
        "stderr": (
            "FailoverError: JSON error injected into SSE stream "
            "stage=assistant decision=surface_error reason=rate_limit"
        ),
        "tool_policy_ok": True,
        "tool_policy_violations": [],
    }

    assert module.is_transient_openclaw_failure(completed, sidecar)


def test_swebench_non_scoreable_openclaw_failure_detects_setup_and_provider_errors():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")

    unsupported = subprocess.CompletedProcess(
        ["cmd"],
        1,
        stdout='Thinking level "xhigh" is not supported for openai-direct/example-model.',
        stderr="",
    )
    quota = subprocess.CompletedProcess(
        ["cmd"],
        1,
        stdout="",
        stderr="Error code: 429 - insufficient_quota: You exceeded your current quota.",
    )
    auth_sidecar = {
        "stderr": "HTTP 401 Unauthorized: User not found",
        "tool_policy_ok": True,
    }

    assert module.non_scoreable_openclaw_failure_reason(unsupported, None) == "unsupported_thinking"
    assert module.non_scoreable_openclaw_failure_reason(quota, None) == "insufficient_quota"
    assert module.non_scoreable_openclaw_failure_reason(quota, auth_sidecar) == "insufficient_quota"
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


def test_swebench_non_scoreable_openclaw_failure_does_not_catch_transient_timeout():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    timeout = subprocess.CompletedProcess(
        ["cmd"],
        1,
        stdout="",
        stderr="FailoverError: LLM request timed out. rawError=terminated",
    )

    assert module.is_transient_openclaw_failure(timeout, None)
    assert module.non_scoreable_openclaw_failure_reason(timeout, None) is None


def test_swebench_policy_violation_returns_disqualified_metadata(tmp_path, monkeypatch):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    row = sample_row()
    checkout_dir = tmp_path / "checkout"
    checkout_dir.mkdir()
    task_dir = tmp_path / "task"

    def fake_run_command(command, cwd=None, timeout=None, check=True):
        output_dir = Path(command[command.index("--output-dir") + 1])
        sidecar_path = module.task_sidecar_path(output_dir, row["instance_id"])
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(
            json.dumps(
                {
                    **protocol_sidecar_identity(module, command, row["instance_id"]),
                    "returncode": 0,
                    "stderr": "[agent] run ended with stopReason=stop",
                    "tool_policy_ok": False,
                    "tool_policy_violations": [
                        {
                            "type": "denied_argument_pattern",
                            "toolName": "exec",
                            "pattern": "\\b(curl|wget)\\b",
                            "toolCallId": "tool_exec_test",
                            "index": 1,
                        }
                    ],
                    "tool_call_count": 3,
                    "tool_error_count": 0,
                    "weave_sidecar": {"ok": True},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="tool policy violation")

    monkeypatch.setattr(module, "run_command", fake_run_command)
    monkeypatch.setattr(module, "ensure_nemoclaw_checkout_ready", lambda checkout_dir, task_dir, args: None)
    args = SimpleNamespace(
        agent="test-agent",
        allow_failed_preflight=False,
        deny_argument_pattern=None,
        deny_tool=None,
        dry_run=False,
        model="openai-direct/example-model",
        no_local=False,
        openclaw_max_attempts=1,
        openclaw_retry_base_seconds=0,
        openclaw_timeout=30,
        profile=None,
        session_prefix=None,
        thinking="high",
        use_task_agent=False,
        weave_sidecar=True,
        weave_sidecar_strict=False,
    )

    metadata = module.run_openclaw_for_task(row, checkout_dir, task_dir, args)

    assert metadata["openclaw_disqualified_reason"] == "tool_policy_violation"
    assert metadata["tool_policy_ok"] is False
    assert metadata["tool_policy_violations"][0]["toolName"] == "exec"
    assert module.should_force_empty_patch(metadata)


def test_swebench_transient_exhaustion_returns_disqualified_metadata(tmp_path, monkeypatch):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    row = sample_row()
    checkout_dir = tmp_path / "checkout"
    checkout_dir.mkdir()
    task_dir = tmp_path / "task"

    def fake_run_command(command, cwd=None, timeout=None, check=True):
        output_dir = Path(command[command.index("--output-dir") + 1])
        sidecar_path = module.task_sidecar_path(output_dir, row["instance_id"])
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(
            json.dumps(
                {
                    **protocol_sidecar_identity(module, command, row["instance_id"]),
                    "returncode": 1,
                    "stderr": "FailoverError: LLM request timed out. rawError=terminated",
                    "tool_policy_ok": True,
                    "tool_policy_violations": [],
                    "tool_call_count": 0,
                    "tool_error_count": 0,
                    "weave_sidecar": {"ok": True},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            command,
            1,
            stdout=str(sidecar_path),
            stderr="FailoverError: LLM request timed out. rawError=terminated",
        )

    monkeypatch.setattr(module, "run_command", fake_run_command)
    args = SimpleNamespace(
        agent="test-agent",
        allow_failed_preflight=False,
        deny_argument_pattern=None,
        deny_tool=None,
        dry_run=False,
        model="openai-direct/example-model",
        no_local=False,
        openclaw_max_attempts=1,
        openclaw_retry_base_seconds=0,
        openclaw_timeout=30,
        profile=None,
        session_prefix=None,
        thinking="high",
        use_task_agent=False,
        weave_sidecar=True,
        weave_sidecar_strict=False,
    )

    metadata = module.run_openclaw_for_task(row, checkout_dir, task_dir, args)

    assert metadata["openclaw_disqualified_reason"] == "provider_transient_exhausted"
    assert metadata["openclaw_returncode"] == 1
    assert metadata["tool_policy_ok"] is True
    assert module.should_force_empty_patch(metadata)
    failures = (task_dir / "openclaw_transient_failures.jsonl").read_text(encoding="utf-8")
    assert '"exhausted": true' in failures


def test_swebench_recovers_workspace_vanished_attestation(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
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


def test_swebench_patch_cache_reuses_only_matching_cache_key(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    row = sample_row()
    args = SimpleNamespace(
        model="openai-direct/example-model",
        thinking="high",
        deny_tool=None,
        deny_argument_pattern=None,
    )
    prompt = module.build_prompt(row)
    cache_key = module.build_cache_key(row, prompt, args)
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    record = {
        "instance_id": row["instance_id"],
        "patch": "diff --git a/x b/x\n",
        "prefix": "old-prefix",
        "cache_key": cache_key,
    }
    (task_dir / "patch_record.json").write_text(
        json.dumps(record, ensure_ascii=False), encoding="utf-8"
    )

    cached = module.load_cached_patch_record(task_dir, cache_key, "new-prefix")
    assert cached is not None
    assert cached["patch"] == record["patch"]
    assert cached["prefix"] == "new-prefix"

    changed_args = SimpleNamespace(
        model="openai-direct/example-model",
        thinking="medium",
        deny_tool=None,
        deny_argument_pattern=None,
    )
    changed_key = module.build_cache_key(row, prompt, changed_args)
    assert module.load_cached_patch_record(task_dir, changed_key, "new-prefix") is None


def test_swebench_patch_cache_rejects_nemoclaw_config_source_change(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    row = sample_row()
    args = SimpleNamespace(
        model="openai-direct/example-model",
        thinking="high",
        deny_tool=None,
        deny_argument_pattern=None,
        nemoclaw_sandbox="nejumi-taiwan",
        nemoclaw_openclaw_config_path="/sandbox/.openclaw/openclaw.json",
    )
    prompt = module.build_prompt(row)
    cache_key = module.build_cache_key(row, prompt, args)
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "patch_record.json").write_text(
        json.dumps(
            {
                "instance_id": row["instance_id"],
                "patch": "diff --git a/x b/x\n",
                "cache_key": cache_key,
                "patch_capture_version": module.PATCH_CAPTURE_VERSION,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    changed_args = SimpleNamespace(
        model="openai-direct/example-model",
        thinking="high",
        deny_tool=None,
        deny_argument_pattern=None,
        nemoclaw_sandbox="nejumi-taiwan",
        nemoclaw_openclaw_config_path="/sandbox/other-openclaw.json",
    )
    changed_key = module.build_cache_key(row, prompt, changed_args)

    assert cache_key["openclaw_config_source"] == "/sandbox/.openclaw/openclaw.json"
    assert module.load_cached_patch_record(task_dir, changed_key, "new-prefix") is None


def test_swebench_patch_cache_rejects_nemoclaw_record_without_session_audit(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    row = sample_row()
    args = SimpleNamespace(
        model="openai-direct/example-model",
        thinking="high",
        deny_tool=None,
        deny_argument_pattern=None,
        max_input_tokens=1_000_000,
        max_tool_calls=60,
        nemoclaw_sandbox="nejumi-taiwan",
        nemoclaw_openclaw_config_path="/sandbox/.openclaw/openclaw.json",
        nemoclaw_checkout_sandbox_root="/sandbox/checkouts",
        nemoclaw_checkout_transfer_mode="copy",
        session_prefix=None,
    )
    prompt = module.build_prompt(row)
    cache_key = module.build_cache_key(row, prompt, args)
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    command = ["python3", "run_openclaw_agent_protocol.py", "run"]
    invocation = {
        "cache_key": cache_key,
        "command": command,
        "command_sha256": module.command_sha256(command),
        "expected_openclaw_result_path": str(task_dir / "openclaw_result.json"),
    }
    invocation_path = task_dir / "openclaw_invocation.json"
    invocation_path.write_text(json.dumps(invocation, ensure_ascii=False), encoding="utf-8")
    record = {
        "instance_id": row["instance_id"],
        "patch": "diff --git a/x b/x\n",
        "cache_key": cache_key,
        "patch_capture_version": module.PATCH_CAPTURE_VERSION,
        "openclaw_result_path": invocation["expected_openclaw_result_path"],
        "openclaw_invocation_path": str(invocation_path),
        "openclaw_invocation_sha256": module.sha256_file(invocation_path),
        "openclaw_command_sha256": invocation["command_sha256"],
    }
    record_path = task_dir / "patch_record.json"
    record_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    assert module.load_cached_patch_record(task_dir, cache_key, "new-prefix") is None
    record["nemoclaw_session_audit"] = {"required": True, "ok": False}
    record["nemoclaw_session_audit_ok"] = False
    record_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    assert module.load_cached_patch_record(task_dir, cache_key, "new-prefix") is None
    record["nemoclaw_session_audit"] = {"required": True, "ok": True}
    record["nemoclaw_session_audit_ok"] = True
    record_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    cached = module.load_cached_patch_record(task_dir, cache_key, "new-prefix")
    assert cached is not None
    assert cached["prefix"] == "new-prefix"
    record["openclaw_invocation_sha256"] = "0" * 64
    record_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    assert module.load_cached_patch_record(task_dir, cache_key, "new-prefix") is None
    record["openclaw_invocation_sha256"] = module.sha256_file(invocation_path)
    record_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    record["tool_policy_ok"] = False
    record_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    assert module.load_cached_patch_record(task_dir, cache_key, "new-prefix") is None
    record["tool_policy_ok"] = True
    record["tool_policy_violations"] = [{"type": "denied_tool", "toolName": "web_search"}]
    record_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    assert module.load_cached_patch_record(task_dir, cache_key, "new-prefix") is None
    record["tool_policy_violations"] = []
    record_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    cached = module.load_cached_patch_record(task_dir, cache_key, "new-prefix")
    assert cached is not None
    assert cached["prefix"] == "new-prefix"
    record["weave_sidecar"] = {"ok": False}
    record_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    assert module.load_cached_patch_record(task_dir, cache_key, "new-prefix") is None
    record["weave_sidecar"] = {"ok": True}
    record_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    cached = module.load_cached_patch_record(task_dir, cache_key, "new-prefix")
    assert cached is not None
    assert cached["prefix"] == "new-prefix"
    record["weave_sidecar_ok"] = False
    record_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    assert module.load_cached_patch_record(task_dir, cache_key, "new-prefix") is None
    record["weave_sidecar_ok"] = True
    record_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    cached = module.load_cached_patch_record(task_dir, cache_key, "new-prefix")
    assert cached is not None
    assert cached["prefix"] == "new-prefix"
    record["conversation_order_ok"] = False
    record_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    assert module.load_cached_patch_record(task_dir, cache_key, "new-prefix") is None
    record["conversation_order_ok"] = True
    record["conversation_order"] = {"ok": False}
    record_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    assert module.load_cached_patch_record(task_dir, cache_key, "new-prefix") is None
    record["conversation_order"] = {"ok": True}
    record_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    cached = module.load_cached_patch_record(task_dir, cache_key, "new-prefix")
    assert cached is not None
    assert cached["prefix"] == "new-prefix"


def test_swebench_session_prefix_is_bound_to_wandb_run_id(monkeypatch):
    monkeypatch.setenv("WANDB_RUN_ID", "twcanary-run-1")
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    row = sample_row()
    args = SimpleNamespace(
        model="openai-direct/example-model",
        thinking="high",
        deny_tool=None,
        deny_argument_pattern=None,
        max_input_tokens=1_000_000,
        max_tool_calls=60,
        nemoclaw_sandbox=None,
        nemoclaw_checkout_sandbox_root=None,
        nemoclaw_checkout_transfer_mode="visible",
        session_prefix=None,
    )

    assert module.resolve_session_prefix(args) == "twcanary-run-1:swebench-pro"
    key = module.build_cache_key(row, module.build_prompt(row), args)
    assert key["session_prefix"] == "twcanary-run-1:swebench-pro"


def test_swebench_session_prefix_expands_wandb_placeholder(monkeypatch):
    monkeypatch.setenv("WANDB_RUN_ID", "twcanary-run-2")
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    args = SimpleNamespace(session_prefix="swe/{wandb_run_id}")

    assert module.resolve_session_prefix(args) == "swe/twcanary-run-2"


def test_swebench_patch_cache_rejects_legacy_empty_patch(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    row = sample_row()
    args = SimpleNamespace(
        model="openai-direct/example-model",
        thinking="high",
        deny_tool=None,
        deny_argument_pattern=None,
    )
    prompt = module.build_prompt(row)
    cache_key = module.build_cache_key(row, prompt, args)
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    record = {
        "instance_id": row["instance_id"],
        "patch": "",
        "prefix": "old-prefix",
        "cache_key": cache_key,
    }
    (task_dir / "patch_record.json").write_text(
        json.dumps(record, ensure_ascii=False), encoding="utf-8"
    )

    assert module.load_cached_patch_record(task_dir, cache_key, "new-prefix") is None


def test_swebench_patch_cache_rejects_legacy_patch_with_selected_tests(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    row = sample_row()
    args = SimpleNamespace(
        model="openai-direct/example-model",
        thinking="high",
        deny_tool=None,
        deny_argument_pattern=None,
    )
    prompt = module.build_prompt(row)
    cache_key = module.build_cache_key(row, prompt, args)
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    record = {
        "instance_id": row["instance_id"],
        "patch": (
            "diff --git a/src/fix.py b/src/fix.py\n"
            "diff --git a/test/units/test_fix.py b/test/units/test_fix.py\n"
        ),
        "prefix": "old-prefix",
        "cache_key": cache_key,
        "patch_capture_version": "git-diff-with-untracked-v1",
    }
    (task_dir / "patch_record.json").write_text(
        json.dumps(record, ensure_ascii=False), encoding="utf-8"
    )

    cached = module.load_cached_patch_record(
        task_dir,
        cache_key,
        "new-prefix",
        ["test/units/test_fix.py"],
    )

    assert cached is None


def test_capture_patch_includes_untracked_files(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    (tmp_path / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)

    (tmp_path / "tracked.py").write_text("VALUE = 2\n", encoding="utf-8")
    (tmp_path / "new_module.py").write_text("NEW_VALUE = 3\n", encoding="utf-8")

    patch = module.capture_patch(tmp_path)

    assert "diff --git a/tracked.py b/tracked.py" in patch
    assert "diff --git a/new_module.py b/new_module.py" in patch
    assert "new file mode" in patch
    assert "+NEW_VALUE = 3" in patch


def test_capture_patch_excludes_selected_test_files(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    (tmp_path / "base.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "base.py"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)

    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "new_module.py").write_text("NEW_VALUE = 3\n", encoding="utf-8")
    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "test_new_module.py").write_text("def test_new(): pass\n", encoding="utf-8")

    patch = module.capture_patch(tmp_path, ["test/test_new_module.py"])

    assert "diff --git a/lib/new_module.py b/lib/new_module.py" in patch
    assert "+NEW_VALUE = 3" in patch
    assert "test/test_new_module.py" not in patch


def test_capture_patch_excludes_openclaw_runtime_dir(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    (tmp_path / "base.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "base.py"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)

    (tmp_path / "base.py").write_text("VALUE = 2\n", encoding="utf-8")
    runtime_dir = tmp_path / ".nejumi_openclaw"
    runtime_dir.mkdir()
    (runtime_dir / "openclaw_config.json").write_text("{}\n", encoding="utf-8")

    patch = module.capture_patch(tmp_path)

    assert "diff --git a/base.py b/base.py" in patch
    assert ".nejumi_openclaw" not in patch


def test_swebench_eval_wandb_artifact_includes_summary_and_patch(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "evaluate_swebench_pro_patches.py")
    patch_path = tmp_path / "patches.json"
    patch_path.write_text("[]\n", encoding="utf-8")
    for filename in (
        "summary.json",
        "eval_results.json",
        "official_invocation.json",
        "official_stdout.log",
        "official_stderr.log",
    ):
        (tmp_path / filename).write_text("{}\n", encoding="utf-8")
    args = SimpleNamespace(
        model_name="provider/model",
        output_dir=tmp_path,
        patch_path=patch_path,
    )
    summary = {
        "total_instances": 2,
        "resolved_instances": 1,
        "unresolved_instances": 1,
        "pass_at_1": 0.5,
    }

    artifact = module.make_result_artifact(args, summary)

    assert artifact.type == "evaluation-results"
    manifest_names = sorted(entry.path for entry in artifact.manifest.entries.values())
    assert manifest_names == [
        "official_eval/eval_results.json",
        "official_eval/official_invocation.json",
        "official_eval/official_stderr.log",
        "official_eval/official_stdout.log",
        "official_eval/summary.json",
        "patches.json",
    ]


def test_evaluator_passes_session_prefix_to_swebench_runner(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    module = load_script_module(REPO_ROOT / "scripts" / "evaluator" / "swebench_pro.py")
    commands = []

    def fake_run_command(command):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    cfg = OmegaConf.create(
        {
            "testmode": False,
            "model": {"pretrained_model_name_or_path": "provider/model"},
            "swebench_pro": {
                "checkout_root": str(tmp_path / "checkouts"),
                "prefix": "tw-swe",
                "thinking": "high",
                "agent": "nejumi-taiwan",
                "openclaw_timeout": 60,
                "openclaw_max_attempts": 1,
                "openclaw_retry_base_seconds": 1,
                "max_input_tokens": 1_000_000,
                "max_tool_calls": 60,
                "max_agent_turns": 55,
                "session_prefix": "{wandb_run_id}:swebench-pro",
                "weave_sidecar": False,
            },
        }
    )

    module._run_openclaw(cfg, tmp_path / "dataset.jsonl", tmp_path / "outputs")

    [command] = commands
    assert command[command.index("--session-prefix") + 1] == "{wandb_run_id}:swebench-pro"
    assert command[command.index("--max-agent-turns") + 1] == "55"


def test_evaluator_defaults_nemoclaw_openclaw_config_path_to_swebench_runner(
    tmp_path,
    monkeypatch,
):
    from omegaconf import OmegaConf

    module = load_script_module(REPO_ROOT / "scripts" / "evaluator" / "swebench_pro.py")
    commands = []

    def fake_run_command(command):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    cfg = OmegaConf.create(
        {
            "testmode": False,
            "model": {"pretrained_model_name_or_path": "provider/model"},
            "swebench_pro": {
                "checkout_root": str(tmp_path / "checkouts"),
                "nemoclaw_sandbox": "nejumi-taiwan",
            },
        }
    )

    module._run_openclaw(cfg, tmp_path / "dataset.jsonl", tmp_path / "outputs")

    [command] = commands
    assert command[command.index("--nemoclaw-sandbox") + 1] == "nejumi-taiwan"
    assert (
        command[command.index("--nemoclaw-openclaw-config-path") + 1]
        == "/sandbox/.openclaw/openclaw.json"
    )
