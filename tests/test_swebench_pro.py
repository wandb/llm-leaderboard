import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tarfile
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


def test_swebench_default_policy_allows_pip_and_blocks_external_fetches():
    module = load_script_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    patterns = [re.compile(pattern) for pattern in module.DEFAULT_DENIED_ARGUMENT_PATTERNS]

    def denied(command: str) -> bool:
        return any(pattern.search(command) for pattern in patterns)

    assert not denied("pip install -e .")
    assert not denied('python -m pip install --break-system-packages -e ".[dev]"')
    assert not denied("cd /sandbox/checkouts/example && .venv/bin/pip install -e .")
    assert not denied("pip install setuptools")
    assert not denied("pip install --break-system-packages asgiref sqlparse")
    assert denied("pip install git+https://example.com/project.git")
    assert denied("pip install https://example.com/package.whl")
    assert denied("pip install git+ssh://example.com/project.git")
    assert denied("git clone git@example.com:project/repo.git")
    assert denied("curl https://example.com/data")
    assert not denied("python - <<'PY'\nimport requests\nprint(requests.Session)\nPY")
    assert denied("python - <<'PY'\nimport requests\nrequests.get('https://example.com')\nPY")


def test_swebench_summary_separates_scoreable_stops_from_unscoreable_failures(tmp_path):
    module = load_script_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    args = SimpleNamespace(
        model="provider/model",
        openclaw_num_workers=2,
        openclaw_task_start_min_interval_seconds=1.0,
        max_input_tokens=1_000_000,
        max_cumulative_input_tokens=1_000_000,
        max_cumulative_output_tokens=500_000,
        max_tool_calls=40,
        max_agent_turns=40,
        max_tool_wall_seconds=120,
    )
    patches = [
        {
            "instance_id": "scoreable-stop",
            "patch": "diff --git a/a b/a\n",
            "openclaw_disqualified_reason": "runtime_budget_exceeded",
        },
        {
            "instance_id": "unscoreable-failure",
            "patch": "",
            "openclaw_disqualified_reason": "provider_transient_exhausted",
        },
    ]

    module.write_outputs(
        tmp_path,
        patches,
        [{"instance_id": "scoreable-stop"}, {"instance_id": "unscoreable-failure"}],
        args,
    )

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["runtime_budget_exceeded_patches"] == 1
    assert summary["scoreable_stop_patches"] == 1
    assert summary["unscoreable_failure_patches"] == 1
    assert summary["disqualified_patches"] == 2


def test_swebench_weave_agents_verifier_failure_is_per_instance_evidence(
    tmp_path, monkeypatch
):
    module = load_script_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")

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
        sample_row(),
        tmp_path,
        args,
        session_key="run:swe:example__repo-1:attempt-1",
        agent_id="tw-swe-test-agent",
        sidecar={"tool_call_count": 3},
    )

    assert evidence["weave_agents_required"] is True
    assert evidence["weave_agents_ok"] is False
    assert "trace content was incomplete" in evidence["weave_agents_error"]
    assert "agent:tw-swe-test-agent:run:swe:example__repo-1:attempt-1" in evidence[
        "weave_agents_conversation_id"
    ]
    assert evidence["weave_agents_verifier_json"].endswith(".json")


def test_swebench_weave_agents_verifier_failure_raises_when_fail_fast(
    tmp_path, monkeypatch
):
    module = load_script_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")

    monkeypatch.setattr(
        module,
        "verify_native_weave_agents_trace",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("trace was missing")),
    )
    args = SimpleNamespace(
        verify_weave_agents=True,
        dry_run=False,
        fail_fast_trace_evidence=True,
        weave_agents_entity="llm-leaderboard",
        weave_agents_project="tc-leaderboard",
        weave_agents_agent_name="nejumi-taiwan-openclaw",
        weave_agents_env_file=tmp_path / ".env",
        weave_agents_limit=50,
        weave_agents_verification_timeout=0,
        weave_agents_poll_seconds=1,
        model="wandb-inference/zai-org/GLM-5.2",
    )

    with pytest.raises(module.RequiredWeaveAgentsTraceError, match="trace was missing"):
        module.verify_weave_agents_for_attempt(
            sample_row(),
            tmp_path,
            args,
            session_key="run:swe:example__repo-1:attempt-1",
            agent_id="tw-swe-test-agent",
            sidecar={"tool_call_count": 3},
        )


def test_swebench_reverifies_cached_native_trace_without_model_rerun(
    tmp_path, monkeypatch
):
    module = load_script_module(
        REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py"
    )
    row = sample_row()
    task_dir = tmp_path / row["instance_id"]
    task_dir.mkdir()
    record = {
        "instance_id": row["instance_id"],
        "weave_agents_required": True,
        "weave_agents_ok": False,
        "weave_agents_conversation_id_contains": "agent:test:conversation",
        "openclaw_tool_call_count": 3,
    }
    (task_dir / "patch_record.json").write_text(json.dumps(record), encoding="utf-8")
    observed = {}

    def verify(**kwargs):
        observed.update(kwargs)
        return {
            "weave_agents_required": True,
            "weave_agents_ok": True,
            "weave_agents_trace_id": "trace-123",
            "weave_agents_conversation_id": "agent:test:conversation",
            "weave_agents_conversation_id_contains": "agent:test:conversation",
            "weave_agents_conversation_url": "https://wandb.example/conversation",
            "weave_agents_url": "https://wandb.example/agents",
        }

    monkeypatch.setattr(module, "verify_native_weave_agents_trace", verify)
    args = SimpleNamespace(
        weave_agents_entity="llm-leaderboard",
        weave_agents_project="tc-leaderboard",
        weave_agents_agent_name="nejumi-taiwan-openclaw",
        weave_agents_env_file=tmp_path / ".env",
        weave_agents_limit=50,
        weave_agents_poll_seconds=1,
        native_trace_cached_reverification_timeout=0,
        model="anthropic/claude-sonnet-4-6",
    )

    refreshed = module.reverify_cached_native_trace(row, task_dir, args, record)

    assert refreshed["weave_agents_ok"] is True
    assert refreshed["weave_agents_trace_id"] == "trace-123"
    assert observed["conversation_id_contains"] == "agent:test:conversation"
    assert observed["required_texts"] == [f"instance_id: {row['instance_id']}"]
    assert observed["require_tool_trace"] is True
    saved = json.loads((task_dir / "patch_record.json").read_text(encoding="utf-8"))
    assert saved["weave_agents_ok"] is True


def test_swebench_model_truncation_is_scoreable_non_retryable_and_preserves_patch():
    module = load_script_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    sidecar = {
        "model_completion": {
            "ok": False,
            "failure_category": "model",
            "reason": "model_output_truncated",
            "scoreable": True,
            "retryable": False,
        }
    }

    reason = module.sidecar_model_failure_reason(sidecar)

    assert reason == "model_output_truncated"
    assert reason in module.SCOREABLE_OPENCLAW_DISQUALIFIED_REASONS
    assert module.should_force_empty_patch({"openclaw_disqualified_reason": reason}) is False


def test_swebench_provider_failure_still_forces_empty_patch():
    module = load_script_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")

    assert module.sidecar_model_failure_reason({"model_completion": {"ok": True}}) is None
    assert module.should_force_empty_patch(
        {"openclaw_disqualified_reason": "provider_transient_exhausted"}
    ) is True


def test_swebench_runtime_budget_stop_preserves_current_patch():
    module = load_script_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")

    assert module.should_force_empty_patch(
        {"openclaw_disqualified_reason": "runtime_budget_exceeded"}
    ) is False
    assert module.should_force_empty_patch(
        {"openclaw_disqualified_reason": "time_up"}
    ) is False


def test_swebench_openclaw_context_tokens_updates_existing_model_entry():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
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
        max_input_tokens=1_000_000,
    )

    result = module.configure_openclaw_context_tokens(config, args)

    assert result == {
        "provider": "openai-direct",
        "model": "gpt-4.1-mini-2025-04-14",
        "contextTokens": 1_000_000,
    }
    [entry] = config["models"]["providers"]["openai-direct"]["models"]
    assert entry["contextTokens"] == 1_000_000


def test_swebench_openclaw_context_tokens_appends_missing_model_entry():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    config = {"models": {"providers": {"openai-direct": {"models": []}}}}
    args = SimpleNamespace(
        model="openai-direct/gpt-4.1-mini-2025-04-14",
        max_input_tokens=1_000_000,
    )

    result = module.configure_openclaw_context_tokens(config, args)

    assert result["contextTokens"] == 1_000_000
    assert config["models"]["providers"]["openai-direct"]["models"] == [
        {
            "id": "gpt-4.1-mini-2025-04-14",
            "name": "gpt-4.1-mini-2025-04-14",
            "contextTokens": 1_000_000,
        }
    ]


def test_swebench_openclaw_model_params_updates_model_entry():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    config = {"models": {"providers": {"openrouter-direct": {"models": []}}}}
    args = SimpleNamespace(
        model="openrouter-direct/z-ai/glm-5.2",
        max_input_tokens=0,
        openclaw_model_overrides_json=json.dumps({"maxTokens": 4096}),
        openclaw_model_params_json=json.dumps(
            {
                "provider": {
                    "order": ["z-ai/fp8"],
                    "only": ["z-ai/fp8"],
                    "allow_fallbacks": False,
                }
            }
        ),
    )

    result = module.configure_openclaw_context_tokens(config, args)

    assert result["overrides"]["maxTokens"] == 4096
    assert result["params"]["provider"]["only"] == ["z-ai/fp8"]
    [entry] = config["models"]["providers"]["openrouter-direct"]["models"]
    assert entry["maxTokens"] == 4096
    assert entry["params"]["provider"]["order"] == ["z-ai/fp8"]
    assert "compat" not in entry


def test_swebench_copy_archive_excludes_git_history(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").mkdir()
    (checkout / ".git" / "objects").mkdir()
    (checkout / ".git" / "objects" / "large").write_text("history", encoding="utf-8")
    (checkout / "src.py").write_text("print('ok')\n", encoding="utf-8")
    archive = tmp_path / "checkout.tgz"

    module.create_checkout_archive(checkout, archive)

    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()
    assert "./src.py" in names
    assert not any(name == "./.git" or name.startswith("./.git/") for name in names)


def test_swebench_copy_archive_dereferences_hard_links(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    checkout = tmp_path / "checkout"
    package_dir = checkout / "node_modules" / ".pnpm" / "package"
    package_dir.mkdir(parents=True)
    package_file = package_dir / "index.js"
    package_file.write_text("module.exports = true;\n", encoding="utf-8")
    linked_file = checkout / "node_modules" / "linked-index.js"
    linked_file.parent.mkdir(parents=True, exist_ok=True)
    linked_file.hardlink_to(package_file)
    archive = tmp_path / "checkout.tgz"

    module.create_checkout_archive(checkout, archive)

    with tarfile.open(archive, "r:gz") as tar:
        package_member = tar.getmember("./node_modules/.pnpm/package/index.js")
        linked_member = tar.getmember("./node_modules/linked-index.js")
    assert package_member.isfile()
    assert linked_member.isfile()


def test_swebench_tracked_archive_materializes_skip_worktree_files(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=checkout,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=checkout,
        check=True,
    )
    tracked = checkout / ".editorconfig"
    tracked.write_text("root = true\n", encoding="utf-8")
    subprocess.run(["git", "add", ".editorconfig"], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "baseline"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "update-index", "--skip-worktree", ".editorconfig"],
        cwd=checkout,
        check=True,
    )
    tracked.unlink()
    assert (
        subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )
    archive = tmp_path / "tracked.tgz"

    module.create_checkout_tracked_archive(checkout, archive)

    with tarfile.open(archive, "r:gz") as tar:
        member = tar.extractfile("./.editorconfig")
        assert member is not None
        assert member.read() == b"root = true\n"


def test_swebench_rebuilds_exact_head_without_tracking_ignored_dependencies(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=checkout,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=checkout,
        check=True,
    )
    (checkout / ".gitignore").write_text("node_modules/\ngenerated.py\n", encoding="utf-8")
    (checkout / "src.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore", "src.py"], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "baseline"], cwd=checkout, check=True)
    dependency = checkout / "node_modules" / "package" / "index.js"
    dependency.parent.mkdir(parents=True)
    dependency.write_text("module.exports = true;\n", encoding="utf-8")
    (checkout / "generated.py").write_text("VERSION = 'runtime'\n", encoding="utf-8")

    expected_tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = module.checkout_head_tree_manifest(checkout)
    assert {entry["path"] for entry in manifest} == {".gitignore", "src.py"}

    reconstructed = tmp_path / "reconstructed"
    shutil.copytree(checkout, reconstructed, ignore=shutil.ignore_patterns(".git"))
    subprocess.run(["git", "init", "-q"], cwd=reconstructed, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=reconstructed,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=reconstructed,
        check=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            module.CHECKOUT_BASELINE_REBUILD_SCRIPT,
            str(reconstructed),
            expected_tree,
        ],
        input=json.dumps(manifest),
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == expected_tree
    assert dependency.relative_to(checkout).joinpath().as_posix() not in {
        entry["path"] for entry in module.checkout_head_tree_manifest(reconstructed)
    }
    assert (reconstructed / "node_modules" / "package" / "index.js").is_file()
    assert (reconstructed / "generated.py").is_file()
    assert (
        subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=reconstructed,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )


def test_nemoclaw_checkout_transfer_uses_sandbox_upload(tmp_path, monkeypatch):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    local_file = tmp_path / "checkout.tgz"
    local_file.write_bytes(b"archive")
    calls = []

    def fake_text_command(args, command, **kwargs):
        calls.append(("text", command))
        if command[0] == "sha256sum":
            return subprocess.CompletedProcess(command, 0, stdout="abc123 /sandbox/tmp/checkout.tgz\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def fake_run_command(command, **kwargs):
        calls.append(("run", command))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(module, "sha256_file", lambda path: "abc123")
    monkeypatch.setattr(module, "run_nemoclaw_text_command", fake_text_command)
    monkeypatch.setattr(module, "run_command", fake_run_command)

    result = module.upload_file_to_nemoclaw(
        local_file,
        "/sandbox/tmp/checkout.tgz",
        SimpleNamespace(
            nemoclaw_bin="nemoclaw",
            nemoclaw_sandbox="nejumi-taiwan",
            nemoclaw_checkout_transfer_timeout=600,
        ),
    )

    assert ("run", ["nemoclaw", "sandbox", "upload", "nejumi-taiwan", str(local_file), "/sandbox/tmp/checkout.tgz"]) in calls
    assert result["transport"] == "nemoclaw_sandbox_upload"
    assert result["chunk_count"] is None


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

    prompt = module.build_prompt(row, max_tool_wall_seconds=240)
    assert "Fix the bug." in prompt
    assert "<file path=" not in prompt
    assert "allowed_repository_context" not in prompt
    assert "secret code" not in prompt
    assert "local shell execution" in prompt
    assert "Use repository search before reading specific files" in prompt
    assert "stop guessing paths and search the checkout" in prompt
    assert "wall-clock limit of 240 seconds" in prompt
    assert "Keep commands targeted" in prompt
    assert "If a command times out" in prompt
    assert "Do not use web search" in prompt
    assert "Do not write `FINAL ANSWER`, `ANSWER:`" in prompt
    assert "Do not include a final-answer marker in the same assistant turn as a tool call" in prompt


def test_openclaw_prompt_includes_runtime_budget_when_configured():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    row = sample_row()

    prompt = module.build_prompt(
        row,
        max_tool_wall_seconds=120,
        max_tool_calls=40,
        max_agent_turns=40,
        max_input_tokens=1_000_000,
        max_cumulative_input_tokens=1_000_000,
        max_cumulative_output_tokens=500_000,
        max_output_tokens_per_response=65_536,
    )

    assert "## Runtime Budget" in prompt
    assert "Maximum tool calls: 40" in prompt
    assert "Maximum agent turns: 40" in prompt
    assert "Per-turn input-token cap: 1000000" in prompt
    assert "Per-response output-token cap: 65536" in prompt
    assert "Cumulative input-token cap: 1000000" in prompt
    assert "Cumulative output-token cap: 500000" in prompt
    assert "hard evaluation limits" in prompt
    assert "finish repository orientation within about 13 tool calls" in prompt
    assert "reserve the remaining calls for patching and verification" in prompt


def test_openclaw_prompt_omits_runtime_budget_when_unconfigured():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    row = sample_row()

    prompt = module.build_prompt(row, max_tool_wall_seconds=120)

    assert "## Runtime Budget" not in prompt
    assert "Maximum tool calls" not in prompt


def test_openclaw_prompt_normalizes_crlf_text_fields():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    row = sample_row()
    row["problem_statement"] = "Line 1\r\nLine 2\rLine 3"
    row["requirements"] = "Req 1\r\nReq 2"
    row["interface"] = "Iface 1\rIface 2"

    prompt = module.build_prompt(row, max_tool_wall_seconds=240)
    normalized_row = sample_row()
    normalized_row["problem_statement"] = "Line 1\nLine 2\nLine 3"
    normalized_row["requirements"] = "Req 1\nReq 2"
    normalized_row["interface"] = "Iface 1\nIface 2"

    assert "\r" not in prompt
    assert prompt == module.build_prompt(normalized_row, max_tool_wall_seconds=240)


def test_swebench_sidecar_path_is_attempt_scoped(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    attempt_dir = tmp_path / "openclaw_attempts" / "attempt-1"

    path = module.task_sidecar_path(attempt_dir, "example__repo-1")

    assert path == attempt_dir / "agentic_swe" / "example__repo-1" / "openclaw_result.json"


def test_swebench_protocol_benchmark_id_normalizes_swebench_sources():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")

    assert module.protocol_benchmark_id({"benchmark_id": "swebench_lite"}) == "agentic_swe"
    assert module.protocol_benchmark_id({"benchmark_id": "swebench_pro"}) == "agentic_swe"
    assert module.protocol_benchmark_id({}) == "agentic_swe"
    assert module.protocol_benchmark_id({"benchmark_id": "deepswe"}) == "deepswe"


def test_swebench_effective_deny_tools_allows_process_control_tool():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    args = SimpleNamespace(
        deny_tool=[
            "web_search",
            "process",
            "process_log",
            "process_*",
            "browser",
        ]
    )

    assert module.effective_deny_tools(args) == ["browser", "web_search"]


def test_swebench_safe_agent_id_preserves_digest_within_openclaw_limit():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    instance_id = "matplotlib__matplotlib-26020"
    prefix = "tw-swe-assorted-gpt-5-6-luna-openai-direct-high-lite"

    agent_id = module.safe_agent_id(instance_id, prefix)

    expected_digest = module.hashlib.sha1(instance_id.encode("utf-8")).hexdigest()[
        : module.TASK_AGENT_DIGEST_LENGTH
    ]
    assert len(agent_id) <= module.OPENCLAW_MAX_AGENT_ID_LENGTH
    assert agent_id.endswith(f"-{expected_digest}")
    assert agent_id == "tw-swe-assorted-gpt-5-6-luna-openai-direct-high-lit-f2a5a25f3109"


def test_swebench_registered_agent_id_from_pretty_json_output():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    stdout = (
        '{\n  "agentId": "normalized-agent",\n  "name": "requested-agent"\n}\n'
        '{"ok": true, "agent_id": "requested-agent"}\n'
    )

    assert module.registered_agent_id_from_stdout(stdout) == "normalized-agent"


def test_swebench_rejects_openclaw_agent_id_rewrite(monkeypatch):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")

    def fake_run_nemoclaw_text_command(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            [],
            0,
            stdout='{"agentId": "normalized-agent"}\n',
            stderr="",
        )

    monkeypatch.setattr(
        module,
        "run_nemoclaw_text_command",
        fake_run_nemoclaw_text_command,
    )
    args = SimpleNamespace(
        deny_argument_pattern=None,
        deny_tool=None,
        dry_run=False,
        max_agent_turns=40,
        max_input_tokens=1_000_000,
        max_tool_calls=40,
        max_tool_wall_seconds=120,
        model="openai-direct/gpt-5.6-luna",
        nemoclaw_openclaw_config_path="/sandbox/.openclaw/openclaw.json",
        openclaw_tool_profile="coding",
        require_actual_token_usage=True,
        session_prefix="test-session",
    )

    with pytest.raises(RuntimeError, match="trace identity mismatch"):
        module.register_nemoclaw_gateway_task_agent(
            args,
            agent_id="requested-agent",
            workspace="/sandbox/checkouts/task",
            agent_dir="/sandbox/checkouts/task/.agent",
        )


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
        max_input_tokens=12345,
        max_agent_turns=55,
        max_tool_wall_seconds=180,
        model="openai-direct/gpt-4.1-mini-2025-04-14",
        no_local=False,
        nemoclaw_checkout_sandbox_root="/sandbox/checkouts",
        nemoclaw_extra_path=["/sandbox/.deepswe-tools/python-bin/abc123"],
        nemoclaw_extra_pythonpath=[
            "/sandbox/.deepswe-tools/python-site/abc123/site-packages"
        ],
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
    assert agent["contextTokens"] == 12345
    assert config["tools"]["exec"]["timeoutSec"] == 180
    assert agent["runRetries"] == {
        "base": 55,
        "perProfile": 0,
        "min": 55,
        "max": 55,
    }
    assert config["tools"]["toolSearch"] is False
    assert config["tools"]["web"]["fetch"]["enabled"] is False
    assert "browser" not in config["tools"]
    metadata = json.loads((task_dir / "openclaw_task_agent.json").read_text(encoding="utf-8"))
    assert metadata["host_config_path"] == str(host_config)
    assert metadata["config_path"] == str(expected_sandbox_config)
    assert metadata["nemoclaw_sandbox"] == "nejumi-taiwan"
    assert metadata["context_cap"]["contextTokens"] == 12345
    assert metadata["exec_timeout"] == {"timeoutSec": 180}
    assert metadata["run_retries"] == {
        "base": 55,
        "perProfile": 0,
        "min": 55,
        "max": 55,
    }


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
        deny_tool=["web_search", "process", "process_*"],
        dry_run=False,
        max_input_tokens=50000,
        max_tool_calls=40,
        max_agent_turns=40,
        max_tool_wall_seconds=240,
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
    assert 'entry["contextTokens"] = context_cap["contextTokens"]' in script
    assert 'entry["runRetries"] = turn_run_retries' in script
    assert 'exec_config["timeoutSec"] = max_tool_wall_seconds' in script
    assert "min(existing_timeout, max_tool_wall_seconds)" not in script
    assert 'target["contextTokens"]' not in script
    assert "is_process_control_tool" in script
    assert 'existing_budget_config.get("denyArgumentPatterns"' not in script
    assert '"denyTools": current_deny_tools' in script
    assert '"denyArgumentPatterns": current_deny_argument_patterns' in script
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
    assert budget["max_tool_calls"] == 40
    assert budget["max_agent_turns"] == 40
    assert budget["max_tool_wall_seconds"] == 240
    assert budget["max_cumulative_input_tokens"] == 50000
    assert budget["max_cumulative_output_tokens"] == 0
    assert budget["deny_tools"] == ["web_search"]
    assert all("requests" not in pattern for pattern in budget["deny_argument_patterns"])
    assert budget["require_actual_token_usage"] is False
    metadata = json.loads((task_dir / "openclaw_task_agent.json").read_text(encoding="utf-8"))
    assert metadata["config_path"] == "/sandbox/.openclaw/openclaw.json"
    assert metadata["exec_timeout"] == {"timeoutSec": 240}
    assert metadata["registration_spec"]["model"] == args.model
    assert metadata["registration_spec"]["max_agent_turns"] == 40
    assert metadata["registration_key"] == module.task_agent_registration_key(
        metadata["registration_spec"]
    )
    assert metadata["gateway_registered"]["ok"] is True
    first_registration_key = metadata["registration_key"]

    module.write_task_openclaw_config(row, checkout_dir, task_dir, args)
    assert len(calls) == 1

    args.max_agent_turns = 41
    module.write_task_openclaw_config(row, checkout_dir, task_dir, args)
    assert len(calls) == 2
    updated_metadata = json.loads(
        (task_dir / "openclaw_task_agent.json").read_text(encoding="utf-8")
    )
    assert updated_metadata["registration_key"] != first_registration_key
    assert updated_metadata["registration_spec"]["max_agent_turns"] == 41
    assert module.task_live_session_dir(checkout_dir, task_dir, args) is None
    assert (
        module.task_live_sandbox_session_dir(checkout_dir, args, agent_id)
        == f"/sandbox/.openclaw/agents/{agent_id}/sessions"
    )
    module._REGISTERED_NEMOCLAW_GATEWAY_AGENTS.clear()
    module.write_task_openclaw_config(row, checkout_dir, task_dir, args)
    assert len(calls) == 3
    module._REGISTERED_NEMOCLAW_GATEWAY_AGENTS.clear()


def test_swebench_configure_openclaw_exec_timeout_overrides_short_template_default():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    config = {"tools": {"exec": {"timeoutSec": 10}}}
    args = SimpleNamespace(max_tool_wall_seconds=240)

    metadata = module.configure_openclaw_exec_timeout(config, args)

    assert metadata == {"timeoutSec": 240}
    assert config["tools"]["exec"]["timeoutSec"] == 240


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


def test_nemoclaw_gateway_cleanup_uses_short_bounded_timeouts(monkeypatch, capsys):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
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
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
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


def test_nemoclaw_gateway_cleanup_swallows_keyboard_interrupt(monkeypatch, capsys):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")

    def fake_unregister(args, agent_id, *, timeout=60):
        raise KeyboardInterrupt("pier shutdown")

    monkeypatch.setattr(module, "unregister_nemoclaw_gateway_task_agent", fake_unregister)
    module._REGISTERED_NEMOCLAW_GATEWAY_AGENTS[:] = [(SimpleNamespace(), "agent-a")]

    module.cleanup_registered_nemoclaw_gateway_agents()

    assert module._REGISTERED_NEMOCLAW_GATEWAY_AGENTS == []
    assert "ignored NeMoClaw task-agent cleanup failure for agent-a" in capsys.readouterr().err


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
        "max_cumulative_input_tokens": None,
        "max_cumulative_output_tokens": None,
        "require_actual_token_usage": False,
        "max_tool_calls": 60,
        "max_agent_turns": None,
        "max_tool_wall_seconds": None,
    }
    assert {violation["type"] for violation in status["violations"]} == {
        "max_input_tokens_exceeded",
        "max_tool_calls_exceeded",
    }


def test_swebench_runtime_budget_exceeded_returns_scoreable_metadata(tmp_path, monkeypatch):
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
    assert module.should_force_empty_patch(metadata) is False


def test_swebench_hard_conversation_order_violation_returns_disqualified_metadata(tmp_path, monkeypatch):
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
                    "stderr": "Conversation order violation",
                    "tool_policy_ok": True,
                    "tool_policy_violations": [],
                    "conversation_order": {
                        "ok": False,
                        "checked": True,
                        "issues": [{"type": "tool_before_or_at_first_user_message"}],
                    },
                    "weave_sidecar": {"ok": True},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            command,
            1,
            stdout=str(sidecar_path),
            stderr="Conversation order violation",
        )

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

    assert metadata["conversation_order_ok"] is False
    assert metadata["openclaw_disqualified_reason"] == "conversation_order_violation"
    assert module.should_force_empty_patch(metadata)


def test_swebench_tool_after_answer_warning_does_not_force_empty_patch(tmp_path, monkeypatch):
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
                    "conversation_order": {
                        "ok": True,
                        "checked": True,
                        "issues": [],
                        "warnings": [{"type": "tool_after_final_answer"}],
                    },
                    "weave_sidecar": {"ok": True},
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

    assert metadata["conversation_order_ok"] is True
    assert metadata["conversation_order"]["warnings"] == [{"type": "tool_after_final_answer"}]
    assert metadata["openclaw_disqualified_reason"] == ""
    assert not module.should_force_empty_patch(metadata)


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
        nemoclaw_extra_path=["/sandbox/.deepswe-tools/python-bin/abc123"],
        nemoclaw_extra_pythonpath=[
            "/sandbox/.deepswe-tools/python-site/abc123/site-packages"
        ],
        nemoclaw_openclaw_config_path="/sandbox/.openclaw/openclaw.json",
        nemoclaw_sandbox="nejumi-taiwan",
        nemoclaw_workdir="/sandbox",
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
    assert (
        captured_command[captured_command.index("--nemoclaw-extra-path") + 1]
        == "/sandbox/.deepswe-tools/python-bin/abc123"
    )
    assert (
        captured_command[captured_command.index("--nemoclaw-extra-pythonpath") + 1]
        == "/sandbox/.deepswe-tools/python-site/abc123/site-packages"
    )
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


def test_swebench_budget_disqualification_allows_nemoclaw_audit_failure(
    tmp_path, monkeypatch
):
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
                    "returncode": 125,
                    "stderr": "Runtime budget exceeded",
                    "tool_policy_ok": True,
                    "tool_policy_violations": [],
                    "tool_call_count": 41,
                    "tool_error_count": 0,
                    "conversation_order": {"ok": False, "checked": True},
                    "runtime_budget": {
                        "ok": False,
                        "enforced": True,
                        "limits": {"max_tool_calls": 40, "max_agent_turns": 40},
                        "observed": {"tool_call_count": 41, "agent_turn_count": 41},
                        "violations": [
                            {
                                "type": "max_tool_calls_exceeded",
                                "observed": 41,
                                "limit": 40,
                            }
                        ],
                    },
                    "weave_sidecar": {"ok": True},
                    "nemoclaw_session_audit": {
                        "required": True,
                        "ok": False,
                        "errors": ["conversation_order_not_ok"],
                    },
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            command,
            125,
            stdout=str(sidecar_path),
            stderr="Runtime budget exceeded",
        )

    monkeypatch.setattr(module, "run_command", fake_run_command)
    monkeypatch.setattr(module, "ensure_nemoclaw_checkout_ready", lambda checkout_dir, task_dir, args: None)
    args = SimpleNamespace(
        agent="test-agent",
        allow_failed_preflight=False,
        deny_argument_pattern=None,
        deny_tool=None,
        dry_run=False,
        max_input_tokens=1_000_000,
        max_tool_calls=40,
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
        thinking="off",
        use_task_agent=False,
        weave_sidecar=False,
        weave_sidecar_strict=False,
    )

    metadata = module.run_openclaw_for_task(row, checkout_dir, task_dir, args)

    assert metadata["openclaw_disqualified_reason"] == "runtime_budget_exceeded"
    assert metadata["runtime_budget"]["ok"] is False
    assert metadata["nemoclaw_session_audit_ok"] is False
    assert module.should_force_empty_patch(metadata) is False


def test_swebench_cached_scoreable_disqualification_allows_failed_session_audit(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    cache_key = {
        "instance_id": "example__repo-1",
        "nemoclaw_sandbox": "nejumi-taiwan",
        "verify_weave_agents": False,
    }
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    sidecar_path = task_dir / "openclaw_result.json"
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
    invocation_path = task_dir / "openclaw_invocation.json"
    invocation_path.write_text(
        json.dumps(
            {
                "cache_key": cache_key,
                "expected_openclaw_result_path": str(sidecar_path),
            }
        ),
        encoding="utf-8",
    )
    record = {
        "instance_id": "example__repo-1",
        "patch": "",
        "cache_key": cache_key,
        "patch_capture_version": module.PATCH_CAPTURE_VERSION,
        "openclaw_disqualified_reason": "runtime_budget_exceeded",
        "conversation_order_ok": False,
        "conversation_order": {"ok": False},
        "tool_policy_ok": True,
        "tool_policy_violations": [],
        "nemoclaw_session_audit_ok": False,
        "nemoclaw_session_audit": {"required": True, "ok": False},
        "openclaw_result_path": str(sidecar_path),
        "openclaw_invocation_path": str(invocation_path),
        "openclaw_invocation_sha256": module.sha256_file(invocation_path),
    }
    (task_dir / "patch_record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    cached = module.load_cached_patch_record(task_dir, cache_key, "repair")

    assert cached is not None
    assert cached["openclaw_disqualified_reason"] == "runtime_budget_exceeded"
    assert cached["openclaw_usage"] == {
        "inputTokens": 100,
        "outputTokens": 20,
        "cacheReadInputTokens": 300,
    }


def test_swebench_cached_provider_failure_is_retried(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    cache_key = {
        "instance_id": "example__repo-1",
        "nemoclaw_sandbox": "nejumi-taiwan",
        "verify_weave_agents": False,
    }
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "patch_record.json").write_text(
        json.dumps(
            {
                "instance_id": "example__repo-1",
                "patch": "",
                "cache_key": cache_key,
                "patch_capture_version": module.PATCH_CAPTURE_VERSION,
                "openclaw_disqualified_reason": "provider_transient_exhausted",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert module.load_cached_patch_record(task_dir, cache_key, "repair") is None


def test_swebench_sidecar_usage_reads_result_meta_and_runtime_budget():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")

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


def test_swebench_nemoclaw_copy_mode_always_resyncs_existing_checkout(tmp_path, monkeypatch):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    checkout_dir = tmp_path / "checkout-example"
    checkout_dir.mkdir()
    task_dir = tmp_path / "task"
    args = SimpleNamespace(
        nemoclaw_sandbox="nejumi-taiwan",
        nemoclaw_checkout_sandbox_root="/sandbox/checkouts",
        nemoclaw_checkout_transfer_mode="copy",
    )

    monkeypatch.setattr(
        module,
        "nemoclaw_checkout_visible",
        lambda checkout, parsed_args: pytest.fail(
            "copy mode must not trust an existing sandbox checkout"
        ),
    )
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


def test_swebench_copy_checkout_path_is_scoped_to_output_run(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    checkout_dir = tmp_path / "checkout-example"
    checkout_dir.mkdir()

    def args_for(output_dir):
        return SimpleNamespace(
            output_dir=output_dir,
            nemoclaw_sandbox="nejumi-taiwan",
            nemoclaw_checkout_sandbox_root="/sandbox/checkouts",
            nemoclaw_checkout_transfer_mode="copy",
        )

    first = module.sandbox_checkout_dir(checkout_dir, args_for(tmp_path / "run-a"))
    first_again = module.sandbox_checkout_dir(checkout_dir, args_for(tmp_path / "run-a"))
    second = module.sandbox_checkout_dir(checkout_dir, args_for(tmp_path / "run-b"))

    assert first == first_again
    assert first != second
    assert first.parent == Path("/sandbox/checkouts")
    assert first.name.startswith(f"{checkout_dir.name}-")


def test_swebench_copy_archives_are_scoped_to_parallel_sandbox_checkout(tmp_path):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    checkout_dir = tmp_path / "openclaw_checkout"

    first = module.sandbox_checkout_archive_paths(
        checkout_dir,
        "/sandbox/checkouts/deepswe/task-a/openclaw_checkout",
    )
    first_again = module.sandbox_checkout_archive_paths(
        checkout_dir,
        "/sandbox/checkouts/deepswe/task-a/openclaw_checkout",
    )
    second = module.sandbox_checkout_archive_paths(
        checkout_dir,
        "/sandbox/checkouts/deepswe/task-b/openclaw_checkout",
    )

    assert first == first_again
    assert first != second
    assert first[0].endswith(".tgz")
    assert first[1].endswith("-tracked.tgz")


def test_swebench_copy_rejects_dirty_host_checkout(tmp_path):
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
    tracked = tmp_path / "tracked.py"
    tracked.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "tracked.py"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "baseline"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    baseline = module.assert_clean_checkout_for_sandbox_copy(tmp_path)
    assert baseline["host_status"] == "clean"
    assert baseline["host_head_commit"]
    assert baseline["host_head_tree"]
    tracked.write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="dirty host checkout"):
        module.assert_clean_checkout_for_sandbox_copy(tmp_path)


def test_swebench_nemoclaw_copy_mode_requires_git_checkout(tmp_path, monkeypatch):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    checkout_dir = tmp_path / "checkout-example"
    checkout_dir.mkdir()
    args = SimpleNamespace(
        nemoclaw_sandbox="nejumi-taiwan",
        nemoclaw_checkout_sandbox_root="/sandbox/checkouts",
        nemoclaw_checkout_transfer_mode="copy",
    )
    calls = []

    def fake_run_nemoclaw_text_command(args, command, input_text=None, timeout=60, check=True, workdir="/sandbox"):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="")

    monkeypatch.setattr(module, "run_nemoclaw_text_command", fake_run_nemoclaw_text_command)

    assert module.nemoclaw_checkout_visible(checkout_dir, args) is False
    assert "git -C" in calls[0][2]
    assert "rev-parse --is-inside-work-tree" in calls[0][2]


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
    monkeypatch.setattr(
        module,
        "ensure_nemoclaw_openclaw_permissions",
        lambda parsed_args: None,
    )
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


def test_swebench_main_defers_and_recovers_provider_transient_task(tmp_path, monkeypatch):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    row = sample_row()
    row["instance_id"] = "provider-recovery-task"
    attempts = []
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
        model="wandb-inference/zai-org/GLM-5.2",
        thinking="high",
        deny_tool=None,
        deny_argument_pattern=None,
        max_input_tokens=1_000_000,
        max_tool_calls=40,
        nemoclaw_sandbox="nejumi-taiwan",
        nemoclaw_checkout_sandbox_root="/sandbox/checkouts",
        nemoclaw_checkout_transfer_mode="copy",
        openclaw_num_workers=1,
        provider_recovery_rounds=1,
        provider_recovery_base_seconds=0,
    )
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "ensure_nemoclaw_openclaw_permissions", lambda parsed_args: None)
    monkeypatch.setattr(module, "read_jsonl", lambda path: [row])
    checkout_dir = tmp_path / "checkout"
    checkout_dir.mkdir()
    monkeypatch.setattr(
        module,
        "prepare_checkout",
        lambda task_row, checkout_root, reset: checkout_dir,
    )

    def run_openclaw_for_task(task_row, checkout, task_dir, parsed_args):
        attempts.append(task_row["instance_id"])
        if len(attempts) == 1:
            return {
                "openclaw_disqualified_reason": "provider_transient_exhausted",
                "openclaw_usage": {"inputTokens": 10, "outputTokens": 2},
            }
        return {
            "openclaw_disqualified_reason": "",
            "tool_policy_ok": True,
            "openclaw_usage": {"inputTokens": 20, "outputTokens": 3},
        }

    monkeypatch.setattr(module, "run_openclaw_for_task", run_openclaw_for_task)
    monkeypatch.setattr(
        module,
        "capture_patch_nemoclaw",
        lambda *args, **kwargs: "diff --git a/file b/file\n",
    )

    module.main()

    assert attempts == ["provider-recovery-task", "provider-recovery-task"]
    patches = json.loads((args.output_dir / "patches.json").read_text(encoding="utf-8"))
    assert len(patches) == 1
    assert patches[0]["openclaw_disqualified_reason"] == ""
    assert patches[0]["billable_openclaw_attempt_count"] == 2
    assert patches[0]["billable_openclaw_usage"]["inputTokens"] == 30
    assert patches[0]["billable_openclaw_usage"]["outputTokens"] == 5
    recovery = json.loads(
        (args.output_dir / "provider_recovery_state.json").read_text(encoding="utf-8")
    )
    assert recovery["recovery_round"] == 1
    assert recovery["exhausted"] is False
    assert recovery["pending_instance_ids"] == []


def test_swebench_main_recovers_one_isolated_native_trace_failure(
    tmp_path, monkeypatch
):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    row = sample_row()
    row["instance_id"] = "trace-recovery-task"
    attempts = []
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
        model="openai-direct/gpt-4.1-mini-2025-04-14",
        thinking="off",
        deny_tool=None,
        deny_argument_pattern=None,
        max_input_tokens=1_000_000,
        max_tool_calls=40,
        nemoclaw_sandbox="nejumi-taiwan",
        nemoclaw_checkout_sandbox_root="/sandbox/checkouts",
        nemoclaw_checkout_transfer_mode="copy",
        openclaw_num_workers=1,
        provider_recovery_rounds=0,
        native_trace_recovery_attempts=1,
        native_trace_recovery_max_tasks=2,
        native_trace_recovery_base_seconds=0,
    )
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "ensure_nemoclaw_openclaw_permissions", lambda parsed_args: None)
    monkeypatch.setattr(module, "read_jsonl", lambda path: [row])
    checkout_dir = tmp_path / "checkout"
    checkout_dir.mkdir()
    monkeypatch.setattr(
        module,
        "prepare_checkout",
        lambda task_row, checkout_root, reset: checkout_dir,
    )

    def run_openclaw_for_task(task_row, checkout, task_dir, parsed_args):
        attempts.append(task_row["instance_id"])
        return {
            "tool_policy_ok": True,
            "weave_agents_required": True,
            "weave_agents_ok": len(attempts) > 1,
            "openclaw_usage": {"inputTokens": 10, "outputTokens": 2},
        }

    monkeypatch.setattr(module, "run_openclaw_for_task", run_openclaw_for_task)
    monkeypatch.setattr(
        module,
        "capture_patch_nemoclaw",
        lambda *args, **kwargs: "diff --git a/file b/file\n",
    )

    module.main()

    assert attempts == ["trace-recovery-task", "trace-recovery-task"]
    patches = json.loads((args.output_dir / "patches.json").read_text(encoding="utf-8"))
    assert patches[0]["weave_agents_ok"] is True
    assert patches[0]["billable_openclaw_attempt_count"] == 2
    assert patches[0]["billable_openclaw_usage"]["inputTokens"] == 20
    recovery = json.loads(
        (args.output_dir / "native_trace_recovery_state.json").read_text(encoding="utf-8")
    )
    assert recovery["pending_instance_ids"] == []
    assert recovery["attempted_instance_ids"] == ["trace-recovery-task"]
    assert recovery["broad_failure"] is False


def test_swebench_native_trace_broad_failure_stops_before_paid_recovery(
    tmp_path, monkeypatch
):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    rows = []
    for index in range(3):
        row = sample_row()
        row["instance_id"] = f"trace-failure-{index}"
        rows.append(row)
    attempts = []
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
        model="openai-direct/gpt-4.1-mini-2025-04-14",
        thinking="off",
        deny_tool=None,
        deny_argument_pattern=None,
        max_input_tokens=1_000_000,
        max_tool_calls=40,
        nemoclaw_sandbox="nejumi-taiwan",
        nemoclaw_checkout_sandbox_root="/sandbox/checkouts",
        nemoclaw_checkout_transfer_mode="copy",
        openclaw_num_workers=1,
        provider_recovery_rounds=0,
        native_trace_recovery_attempts=1,
        native_trace_recovery_max_tasks=2,
        native_trace_recovery_base_seconds=0,
    )
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "ensure_nemoclaw_openclaw_permissions", lambda parsed_args: None)
    monkeypatch.setattr(module, "read_jsonl", lambda path: rows)
    checkout_dir = tmp_path / "checkout"
    checkout_dir.mkdir()
    monkeypatch.setattr(
        module,
        "prepare_checkout",
        lambda task_row, checkout_root, reset: checkout_dir,
    )

    def run_openclaw_for_task(task_row, checkout, task_dir, parsed_args):
        attempts.append(task_row["instance_id"])
        return {
            "tool_policy_ok": True,
            "weave_agents_required": True,
            "weave_agents_ok": False,
            "openclaw_usage": {"inputTokens": 10, "outputTokens": 2},
        }

    monkeypatch.setattr(module, "run_openclaw_for_task", run_openclaw_for_task)
    monkeypatch.setattr(
        module,
        "capture_patch_nemoclaw",
        lambda *args, **kwargs: "diff --git a/file b/file\n",
    )

    with pytest.raises(module.NativeTraceRecoveryError, match="No paid"):
        module.main()

    assert attempts == [row["instance_id"] for row in rows]
    recovery = json.loads(
        (args.output_dir / "native_trace_recovery_state.json").read_text(encoding="utf-8")
    )
    assert recovery["broad_failure"] is True
    assert recovery["attempted_instance_ids"] == []


def test_swebench_runtime_budget_summary_uses_configured_caps_for_dry_run():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    args = SimpleNamespace(max_input_tokens=1_000_000, max_tool_calls=60)

    summary = module.runtime_budget_summary([], args)

    assert summary == {
        "max_input_tokens": 1_000_000,
        "max_tool_calls": 60,
        "max_agent_turns": None,
        "max_tool_wall_seconds": None,
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


def test_swebench_outer_openclaw_timeout_is_scoreable_time_up_not_retry():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    completed = subprocess.CompletedProcess(
        ["cmd"],
        124,
        stdout="",
        stderr="Command timed out after 3660 seconds",
    )

    assert module.is_outer_openclaw_timeout(completed, None)
    assert not module.is_transient_openclaw_failure(completed, None)
    assert module.non_scoreable_openclaw_failure_reason(completed, None) is None


def test_swebench_outer_timeout_without_sidecar_still_verifies_native_trace(
    tmp_path, monkeypatch
):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    row = sample_row()
    checkout_dir = tmp_path / "checkout"
    checkout_dir.mkdir()
    task_dir = tmp_path / "task"
    observed = {}

    def fake_run_command(command, cwd=None, timeout=None, check=True):
        return subprocess.CompletedProcess(
            command,
            124,
            stdout="",
            stderr="Command timed out after 40 seconds",
        )

    def fake_verify(row_arg, task_dir_arg, args_arg, **kwargs):
        observed.update(kwargs)
        return {
            **module.default_weave_agents_evidence(
                args_arg,
                required=True,
                session_key=kwargs["session_key"],
                agent_id=kwargs["agent_id"],
            ),
            "weave_agents_ok": True,
            "weave_agents_required": True,
            "weave_agents_trace_id": "trace-time-up",
        }

    monkeypatch.setattr(module, "run_command", fake_run_command)
    monkeypatch.setattr(module, "verify_weave_agents_for_attempt", fake_verify)
    monkeypatch.setattr(
        module,
        "ensure_nemoclaw_checkout_ready",
        lambda checkout_dir, task_dir, args: None,
    )
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
        openclaw_max_attempts=2,
        openclaw_retry_base_seconds=0,
        openclaw_timeout=30,
        profile=None,
        session_prefix=None,
        thinking="high",
        use_task_agent=False,
        verify_weave_agents=True,
        weave_sidecar=False,
    )

    metadata = module.run_openclaw_for_task(row, checkout_dir, task_dir, args)

    assert metadata["openclaw_disqualified_reason"] == "time_up"
    assert metadata["weave_agents_ok"] is True
    assert metadata["weave_agents_required"] is True
    assert metadata["weave_agents_trace_id"] == "trace-time-up"
    assert observed["sidecar"] == {"tool_call_count": 1}


def test_swebench_returncode_zero_provider_timeout_sidecar_is_transient():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
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


def test_swebench_success_sidecar_does_not_treat_prompt_timeout_text_as_provider_failure():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    completed = subprocess.CompletedProcess(["cmd"], 0, stdout="", stderr="")
    sidecar = {
        "returncode": 0,
        "stdout_json": {
            "status": "ok",
            "summary": "completed",
            "result": {
                "finalPromptText": "If a command times out, narrow the command.",
                "finalAssistantVisibleText": "A unit test timed out before the fix.",
                "executionTrace": {
                    "attempts": [{"result": "success", "stage": "assistant"}],
                },
            },
        },
        "runtime_budget": {
            "ok": True,
            "violations": [],
            "live": {"live_provider_timeout_count": 0, "exceeded_limits": []},
        },
        "tool_policy_ok": True,
        "tool_policy_violations": [],
    }

    assert not module.sidecar_provider_timeout(sidecar)
    assert not module.is_transient_openclaw_failure(completed, sidecar)


def test_swebench_provider_timeout_accepts_explicit_stdout_json_error_field():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    sidecar = {
        "stdout_json": {
            "status": "error",
            "errorMessage": "FailoverError: LLM request timed out",
        },
        "runtime_budget": {"ok": True, "violations": []},
    }

    assert module.sidecar_provider_timeout(sidecar)


def test_swebench_nonzero_failure_does_not_scan_prompt_or_assistant_payload():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    completed = subprocess.CompletedProcess(["cmd"], 1, stdout="", stderr="")
    sidecar = {
        "stdout_json": {
            "status": "ok",
            "prompt": "If a command times out, inspect the failure.",
            "messages": [{"role": "assistant", "content": "A unit test timed out earlier."}],
        },
        "runtime_budget": {"ok": True, "violations": []},
        "tool_policy_ok": True,
        "tool_policy_violations": [],
    }

    assert not module.is_transient_openclaw_failure(completed, sidecar)


def test_swebench_live_provider_timeout_overrides_runtime_budget_violation():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    completed = subprocess.CompletedProcess(["cmd"], 125, stdout="", stderr="")
    sidecar = {
        "returncode": 125,
        "stderr": "Live OpenClaw interrupt: reason=live_provider_timeout",
        "runtime_budget": {
            "ok": False,
            "violations": [
                {
                    "type": "missing_actual_token_usage",
                    "source": "live_runtime_budget",
                }
            ],
            "live": {
                "reason": "live_provider_timeout",
                "live_provider_timeout_count": 1,
                "provider_timeouts": [
                    {
                        "errorCode": "504",
                        "errorMessage": "Upstream idle timeout exceeded",
                    }
                ],
            },
        },
        "tool_policy_ok": True,
        "tool_policy_violations": [],
    }

    assert module.sidecar_provider_timeout(sidecar)
    assert module.is_transient_openclaw_failure(completed, sidecar)

    idle_sidecar = {
        "runtime_budget": {
            "ok": False,
            "violations": [{"type": "missing_actual_token_usage"}],
            "live": {
                "reason": "llm_response_idle_timeout",
                "interrupt_reason": "llm_response_idle_timeout",
                "exceeded_limits": ["llm_response_idle_timeout"],
            },
        },
        "tool_policy_ok": True,
        "tool_policy_violations": [],
    }
    assert not module.sidecar_provider_timeout(idle_sidecar)
    assert module.sidecar_llm_response_idle_timeout(idle_sidecar)
    assert not module.is_transient_openclaw_failure(completed, idle_sidecar)


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


def test_swebench_runtime_budget_openclaw_failure_is_not_transient():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
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


def test_swebench_provider_timeout_interrupt_is_transient():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
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


def test_swebench_policy_violation_is_recorded_without_disqualification(tmp_path, monkeypatch):
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

    assert metadata["openclaw_disqualified_reason"] == ""
    assert metadata["tool_policy_ok"] is False
    assert metadata["tool_policy_violations"][0]["toolName"] == "exec"
    assert not module.should_force_empty_patch(metadata)


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


def test_swebench_provider_timeout_runtime_budget_sidecar_retries_before_disqualifying(
    tmp_path, monkeypatch
):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    row = sample_row()
    checkout_dir = tmp_path / "checkout"
    checkout_dir.mkdir()
    task_dir = tmp_path / "task"
    calls = []

    def fake_run_command(command, cwd=None, timeout=None, check=True):
        calls.append(command)
        output_dir = Path(command[command.index("--output-dir") + 1])
        sidecar_path = module.task_sidecar_path(output_dir, row["instance_id"])
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        identity = protocol_sidecar_identity(module, command, row["instance_id"])
        if len(calls) == 1:
            sidecar_path.write_text(
                json.dumps(
                    {
                        **identity,
                        "returncode": 125,
                        "stderr": "Live OpenClaw interrupt: reason=live_provider_timeout",
                        "tool_policy_ok": True,
                        "tool_policy_violations": [],
                        "tool_call_count": 12,
                        "tool_error_count": 0,
                        "runtime_budget": {
                            "ok": False,
                            "violations": [
                                {
                                    "type": "missing_actual_token_usage",
                                    "source": "live_runtime_budget",
                                }
                            ],
                            "live": {
                                "reason": "live_provider_timeout",
                                "live_provider_timeout_count": 1,
                            },
                        },
                        "stdout_json": {
                            "result": {"meta": {"agentMeta": {"usage": {"input": 10, "output": 2}}}}
                        },
                        "weave_sidecar": {"ok": True},
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(
                command,
                125,
                stdout=str(sidecar_path),
                stderr="Live OpenClaw interrupt: reason=live_provider_timeout",
            )
        sidecar_path.write_text(
            json.dumps(
                {
                    **identity,
                    "returncode": 0,
                    "stderr": "[agent] run ended with stopReason=stop",
                    "tool_policy_ok": True,
                    "tool_policy_violations": [],
                    "tool_call_count": 10,
                    "tool_error_count": 0,
                    "runtime_budget": {"ok": True, "violations": []},
                    "stdout_json": {
                        "result": {"meta": {"agentMeta": {"usage": {"input": 20, "output": 4}}}}
                    },
                    "weave_sidecar": {"ok": True},
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
        max_agent_turns=40,
        max_cumulative_input_tokens=1_000_000,
        max_cumulative_output_tokens=500_000,
        max_tool_wall_seconds=120,
        model="openai-direct/example-model",
        no_local=False,
        openclaw_max_attempts=2,
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

    assert len(calls) == 2
    assert metadata["attempt_number"] == 2
    assert metadata["openclaw_disqualified_reason"] == ""
    assert metadata["openclaw_returncode"] == 0
    assert metadata["openclaw_usage"] == {"input": 20, "output": 4}
    assert metadata["billable_openclaw_attempt_count"] == 2
    assert metadata["billable_openclaw_usage"]["inputTokens"] == 30
    assert metadata["billable_openclaw_usage"]["outputTokens"] == 6
    failures = [
        json.loads(line)
        for line in (task_dir / "openclaw_transient_failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert failures[0]["failure_reason"] == "provider_timeout"
    assert failures[0]["exhausted"] is False


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


def test_capture_patch_includes_model_staged_changes(tmp_path):
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
    (tmp_path / "staged_new.py").write_text("STAGED = True\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "tracked.py", "staged_new.py"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    patch = module.capture_patch(tmp_path)

    assert "diff --git a/tracked.py b/tracked.py" in patch
    assert "+VALUE = 2" in patch
    assert "diff --git a/staged_new.py b/staged_new.py" in patch
    assert "new file mode" in patch
    assert "+STAGED = True" in patch


def test_capture_patch_nemoclaw_diffs_against_head(tmp_path, monkeypatch):
    module = load_module(REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py")
    calls = []

    def fake_run_nemoclaw_text_command(args, command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(module, "run_nemoclaw_text_command", fake_run_nemoclaw_text_command)
    args = SimpleNamespace(
        nemoclaw_checkout_sandbox_root="/sandbox/checkouts",
        nemoclaw_checkout_transfer_mode="copy",
        nemoclaw_sandbox="nejumi-taiwan",
    )

    module.capture_patch_nemoclaw(tmp_path / "checkout", args)

    diff_call = next(command for command in calls if "git diff" in command[2])
    assert diff_call[2] == 'git diff --binary HEAD -- "$@"'


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
    subprocess.run(
        ["git", "add", "lib/new_module.py", "test/test_new_module.py"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

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

    def fake_run_command(command, **kwargs):
        commands.append(command)
        assert kwargs["timeout"] == 28_800
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


def test_evaluator_passes_non_dry_run_limit_to_swebench_runner(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    module = load_script_module(REPO_ROOT / "scripts" / "evaluator" / "swebench_pro.py")
    commands = []

    def fake_run_command(command, **kwargs):
        commands.append(command)
        assert kwargs["timeout"] == 28_800
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    cfg = OmegaConf.create(
        {
            "testmode": False,
            "model": {"pretrained_model_name_or_path": "provider/model"},
            "swebench_pro": {
                "checkout_root": str(tmp_path / "checkouts"),
                "prefix": "tw-swe",
                "limit": 1,
                "dry_run": False,
                "weave_sidecar": False,
            },
        }
    )

    module._run_openclaw(cfg, tmp_path / "dataset.jsonl", tmp_path / "outputs")

    [command] = commands
    assert command[command.index("--limit") + 1] == "1"
    assert "--dry-run" not in command


def test_evaluator_defaults_nemoclaw_openclaw_config_path_to_swebench_runner(
    tmp_path,
    monkeypatch,
):
    from omegaconf import OmegaConf

    module = load_script_module(REPO_ROOT / "scripts" / "evaluator" / "swebench_pro.py")
    commands = []

    def fake_run_command(command, **kwargs):
        commands.append(command)
        assert kwargs["timeout"] == 28_800
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


def test_evaluator_rejects_no_local_swebench_without_named_sandbox(tmp_path):
    from omegaconf import OmegaConf

    module = load_script_module(REPO_ROOT / "scripts" / "evaluator" / "swebench_pro.py")
    cfg = OmegaConf.create(
        {
            "model": {"pretrained_model_name_or_path": "provider/model"},
            "swebench_pro": {
                "checkout_root": str(tmp_path / "checkouts"),
                "no_local": True,
                "nemoclaw_sandbox": None,
            },
        }
    )

    with pytest.raises(ValueError, match="swebench_pro.nemoclaw_sandbox"):
        module._run_openclaw(cfg, tmp_path / "dataset.jsonl", tmp_path / "outputs")
