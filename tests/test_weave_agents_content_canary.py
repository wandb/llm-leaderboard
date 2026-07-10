import importlib.util
import hashlib
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "scripts" / "tools"


def load_module():
    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))
    path = TOOLS_DIR / "run_weave_agents_content_canary.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def make_args(tmp_path, **overrides):
    values = {
        "execute": False,
        "model": None,
        "thinking": "low",
        "timeout": 180,
        "canary_id": "TEST_CANARY_001",
        "output_dir": tmp_path,
        "entity": "llm-leaderboard",
        "project": "tc-leaderboard",
        "agent_name": "nejumi-taiwan-openclaw",
        "env_file": REPO_ROOT / ".env",
        "openclaw_bin": "openclaw",
        "openclaw_config_path": None,
        "cwd": REPO_ROOT,
        "agent": "main",
        "profile": None,
        "nemoclaw_bin": "nemoclaw",
        "nemoclaw_sandbox": None,
        "nemoclaw_workdir": "/sandbox",
        "nemoclaw_openclaw_config_path": "/sandbox/.openclaw/openclaw.json",
        "allow_failed_preflight": False,
        "verify_attempts": 8,
        "verify_sleep_seconds": 0.0,
        "agents_limit": 30,
        "no_require_tool": False,
        "no_require_usage": False,
        "weave_sidecar": False,
        "external_action_approval_report_json": None,
        "external_action_approval_source_packet_json": None,
    }
    values.update(overrides)
    return Namespace(**values)


def expected_nemoclaw_metadata() -> dict:
    return {
        "required": True,
        "enabled": True,
        "bin": "nemoclaw",
        "sandbox": "nejumi-taiwan",
        "workdir": "/sandbox",
    }


def write_external_action_approval_report(
    path: Path,
    *,
    source_packet: Path | None = None,
    approved_budget_usd: float = 25.0,
    minimum_approved_budget_usd: float = 20.0,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if source_packet is None:
        source_packet = path.parent / "external_action_approval_packet.json"
    source_packet.parent.mkdir(parents=True, exist_ok=True)
    source_packet.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "external_action_approval_packet",
                "external_action_checklist_sha256": "b" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source_packet_sha256 = hashlib.sha256(source_packet.read_bytes()).hexdigest()
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ok": True,
                "status": "approved",
                "generated_at": 123.0,
                "approval_packet_json": "temp/reviewed_external_action_approval.json",
                "external_action_checklist_sha256": "b" * 64,
                "required_approval_count": 3,
                "granted_approval_count": 3,
                "all_required_approvals_granted": True,
                "approval_results": [
                    {
                        "requirement": "paid_api",
                        "required": True,
                        "approved": True,
                        "approved_budget_usd": approved_budget_usd,
                        "minimum_approved_budget_usd": minimum_approved_budget_usd,
                        "minimum_approved_budget_source": (
                            "max_pre_run_budget_estimate_high"
                        ),
                        "approved_model_scope": "OpenAI content canary",
                        "errors": [],
                    },
                    {
                        "requirement": "wandb_access",
                        "required": True,
                        "approved": True,
                        "errors": [],
                    },
                    {
                        "requirement": "wandb_write",
                        "required": True,
                        "approved": True,
                        "errors": [],
                    },
                ],
                "source_binding": {
                    "source_packet_json": str(source_packet),
                    "source_packet_readable": True,
                    "source_approval_packet_sha256": source_packet_sha256,
                    "reviewed_packet_json": "temp/reviewed_external_action_approval.json",
                    "bound": True,
                    "errors": [],
                },
                "will_execute_external_actions": False,
                "errors": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return source_packet


def test_prepare_only_writes_plan_without_paid_execution(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "tools" / "run_weave_agents_content_canary.py"),
            "--canary-id",
            "TEST_CANARY_001",
            "--output-dir",
            str(tmp_path),
            "--model",
            "gpt-5.4-mini-2026-03-17",
            "--nemoclaw-sandbox",
            "nejumi-taiwan",
            "--verify-sleep-seconds",
            "0",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["executed"] is False
    plan = payload["plan"]
    assert plan["will_call_paid_model_api"] is False
    assert plan["model"] == "gpt-5.4-mini-2026-03-17"
    assert plan["nemoclaw"] == expected_nemoclaw_metadata()
    assert plan["verification_requirements"]["required_texts"] == [
        "TEST_CANARY_001",
        "CANARY_RESULT TEST_CANARY_001 91",
        "openclaw_config_source: /sandbox/.openclaw/openclaw.json",
    ]
    assert plan["verification_requirements"]["expected_request_models"] == [
        "gpt-5.4-mini-2026-03-17"
    ]
    assert plan["run_command"]
    assert "--openclaw-config-source" in plan["run_command"]
    assert "/sandbox/.openclaw/openclaw.json" in plan["run_command"]
    assert len(plan["run_command_sha256"]) == 64
    assert plan["nemoclaw_openclaw_config_preflight"] == {
        "required_before_openclaw": False,
        "ran": False,
        "ok": True,
        "model": "gpt-5.4-mini-2026-03-17",
        "provider": "",
        "model_id": "",
        "config_path": "/sandbox/.openclaw/openclaw.json",
        "command": [],
        "returncode": None,
        "checks": [],
        "errors": [],
    }
    assert plan["verification_requirements"]["require_tool_content"] is True
    assert Path(plan["prompt_file"]).exists()
    assert Path(plan["expected_sidecar"]).name == "openclaw_result.json"
    assert Path(plan["agents_diagnostic_file"]).name.endswith(".agents.json")
    assert payload["gate_result"]["status"] == "not_run"
    assert Path(plan["gate_result_file"]).exists()


def test_execute_requires_explicit_model():
    module = load_module()

    try:
        module.main(["--execute", "--verify-sleep-seconds", "0"])
    except SystemExit as exc:
        assert "--model is required" in str(exc)
    else:
        raise AssertionError("main should require --model with --execute")


def test_execute_requires_nemoclaw_sandbox():
    module = load_module()

    try:
        module.main(
            [
                "--execute",
                "--model",
                "openai-direct/gpt-4.1-nano-2025-04-14",
                "--verify-sleep-seconds",
                "0",
            ]
        )
    except SystemExit as exc:
        assert "--nemoclaw-sandbox is required" in str(exc)
    else:
        raise AssertionError("main should require --nemoclaw-sandbox with --execute")


def test_execute_requires_external_action_approval_before_openclaw(tmp_path, capsys):
    module = load_module()

    try:
        module.main(
            [
                "--execute",
                "--model",
                "openai-direct/gpt-4.1-nano-2025-04-14",
                "--nemoclaw-sandbox",
                "nejumi-taiwan",
                "--canary-id",
                "TEST_CANARY_APPROVAL",
                "--output-dir",
                str(tmp_path),
                "--verify-sleep-seconds",
                "0",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("main should require external action approval with --execute")

    payload = json.loads(capsys.readouterr().out)
    assert payload["executed"] is False
    assert payload["blocked_before_openclaw"] is True
    assert "--external-action-approval-source-packet-json" in payload["missing_fields"]
    assert "--external-action-approval-report-json" in payload["missing_fields"]
    plan = json.loads(Path(payload["plan_file"]).read_text(encoding="utf-8"))
    assert plan["will_execute_external_actions"] is True
    assert plan["nemoclaw"] == expected_nemoclaw_metadata()
    assert plan["verification_requirements"]["expected_request_models"] == [
        "openai-direct/gpt-4.1-nano-2025-04-14",
        "gpt-4.1-nano-2025-04-14",
    ]
    assert plan["external_action_approval"]["valid"] is False
    assert plan["nemoclaw_openclaw_config_preflight"]["required_before_openclaw"] is True
    assert plan["nemoclaw_openclaw_config_preflight"]["ran"] is False
    command_result_path = (
        tmp_path
        / "plans"
        / "weave_agents_content_canary_TEST_CANARY_APPROVAL.command_result.json"
    )
    assert command_result_path.exists()
    command_result = json.loads(command_result_path.read_text(encoding="utf-8"))
    assert command_result["ok"] is False
    assert command_result["blocked_before_openclaw"] is True
    assert command_result["paid_api_attempted"] is False
    assert command_result["failure"]["kind"] == "external_action_approval_missing"
    assert command_result["task_id"] == "weave_agents_content_canary_TEST_CANARY_APPROVAL"
    assert command_result["model"] == "openai-direct/gpt-4.1-nano-2025-04-14"
    assert command_result["run_command"] == plan["run_command"]
    assert command_result["run_command_sha256"] == plan["run_command_sha256"]
    gate = json.loads(Path(payload["gate_result_file"]).read_text(encoding="utf-8"))
    assert gate["status"] == "external_action_approval_missing"
    assert gate["paid_api_attempted"] is False


def test_external_action_approval_record_requires_matching_source_packet(tmp_path):
    module = load_module()
    report = tmp_path / "external_action_approval.verify.json"
    source_packet = write_external_action_approval_report(report)
    other_source_packet = tmp_path / "other" / "external_action_approval_packet.json"
    other_source_packet.parent.mkdir(parents=True, exist_ok=True)
    other_source_packet.write_text(
        json.dumps({"schema_version": 1, "kind": "other"}) + "\n",
        encoding="utf-8",
    )

    ok_record = module.build_external_action_approval_record(
        report,
        required_before_external_action=True,
        expected_source_packet_path=source_packet,
    )
    bad_record = module.build_external_action_approval_record(
        report,
        required_before_external_action=True,
        expected_source_packet_path=other_source_packet,
    )

    assert ok_record["valid"] is True
    assert ok_record["source_packet_path_matches_expected"] is True
    assert ok_record["source_packet_sha256_matches_expected"] is True
    assert bad_record["valid"] is False
    assert bad_record["source_packet_path_matches_expected"] is False
    assert bad_record["source_packet_sha256_matches_expected"] is False
    assert any("source_packet_json does not match" in item for item in bad_record["errors"])


def test_external_action_approval_record_rejects_paid_budget_below_floor(tmp_path):
    module = load_module()
    report = tmp_path / "external_action_approval.verify.json"
    source_packet = write_external_action_approval_report(
        report,
        approved_budget_usd=19.0,
        minimum_approved_budget_usd=20.0,
    )

    record = module.build_external_action_approval_record(
        report,
        required_before_external_action=True,
        expected_source_packet_path=source_packet,
    )

    assert record["valid"] is False
    assert record["paid_api_approval"]["approved_budget_usd"] == 19.0
    assert record["paid_api_approval"]["minimum_approved_budget_usd"] == 20.0
    assert record["approval_results_validation"]["valid"] is False
    assert any(
        "greater than or equal to minimum_approved_budget_usd" in error
        for error in record["errors"]
    )


def test_request_model_aliases_include_direct_provider_remainder():
    module = load_module()

    assert module.request_model_aliases("openrouter-direct/z-ai/glm-5.2") == [
        "openrouter-direct/z-ai/glm-5.2",
        "z-ai/glm-5.2",
        "glm-5.2",
    ]


def test_verify_command_uses_sidecar_conversation_or_task_id(tmp_path):
    module = load_module()
    args = make_args(
        tmp_path,
        model="openai-direct/gpt-4.1-nano-2025-04-14",
        nemoclaw_sandbox="nejumi-taiwan",
    )
    paths = module.canary_paths(tmp_path, "TEST_CANARY_001")
    sidecar_dir = paths.expected_sidecar.parent
    sidecar_dir.mkdir(parents=True)
    paths.expected_sidecar.write_text(
        json.dumps({"stdout_json": {"meta": {"agentMeta": {"sessionId": "session-abc"}}}}),
        encoding="utf-8",
    )

    conversation_id = module.extract_conversation_id(paths.expected_sidecar)
    assert conversation_id == "session-abc"

    command = module.build_verify_command(
        args,
        paths,
        tmp_path / "verify.json",
        conversation_id=conversation_id,
    )
    assert "--conversation-id" in command
    assert "session-abc" in command

    fallback = module.build_verify_command(args, paths, tmp_path / "verify.json")
    assert "--conversation-id-contains" in fallback
    assert paths.task_id.lower() in fallback
    assert fallback.count("--require-text") == 3
    assert "TEST_CANARY_001" in fallback
    assert "CANARY_RESULT TEST_CANARY_001 91" in fallback
    assert "openclaw_config_source: /sandbox/.openclaw/openclaw.json" in fallback
    assert fallback.count("--expected-request-model") == 2
    assert "openai-direct/gpt-4.1-nano-2025-04-14" in fallback
    assert "gpt-4.1-nano-2025-04-14" in fallback

    diagnostic = module.build_agents_diagnostic_command(
        args,
        paths,
        conversation_id=conversation_id,
    )
    assert "--conversation-id" in diagnostic
    assert "session-abc" in diagnostic
    assert "--json" in diagnostic
    assert str(paths.agents_diagnostic_file) in diagnostic

    diagnostic_fallback = module.build_agents_diagnostic_command(args, paths)
    assert "--conversation-id-contains" in diagnostic_fallback
    assert paths.task_id.lower() in diagnostic_fallback


def test_extract_conversation_id_prefers_openclaw_session_key(tmp_path):
    module = load_module()
    sidecar = tmp_path / "openclaw_result.json"
    sidecar.write_text(
        json.dumps(
            {
                "stdout_json": {
                    "result": {
                        "meta": {
                            "agentMeta": {
                                "sessionId": "uuid-session-id",
                            },
                            "systemPromptReport": {
                                "sessionKey": "agent:main:weave_agents_content_canary_test",
                            },
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert (
        module.extract_conversation_id(sidecar)
        == "agent:main:weave_agents_content_canary_test"
    )


def test_run_command_routes_through_nemoclaw_when_sandbox_is_set(tmp_path):
    module = load_module()
    args = make_args(tmp_path, model="openai-direct/test", nemoclaw_sandbox="nejumi-taiwan")
    paths = module.canary_paths(tmp_path, "TEST_CANARY_NEMOCLAW")

    command = module.build_run_command(args, paths)

    assert "--nemoclaw-bin" in command
    assert "nemoclaw" in command
    assert "--nemoclaw-sandbox" in command
    assert "nejumi-taiwan" in command
    assert "--nemoclaw-workdir" in command
    assert "/sandbox" in command
    assert module.build_nemoclaw_metadata(args) == expected_nemoclaw_metadata()


def test_nemoclaw_openclaw_config_preflight_accepts_expected_model(tmp_path, monkeypatch):
    module = load_module()
    args = make_args(
        tmp_path,
        execute=True,
        model="openai-direct/test-mini",
        nemoclaw_sandbox="nejumi-taiwan",
    )

    def fake_run_subprocess(command, *, cwd):
        assert command[:4] == ["nemoclaw", "sandbox", "exec", "nejumi-taiwan"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "models": {
                        "providers": {
                            "openai-direct": {
                                "models": [{"id": "test-mini"}],
                            }
                        }
                    },
                    "plugins": {"entries": {"weave": {"enabled": True}}},
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(module, "run_subprocess", fake_run_subprocess)

    record = module.build_nemoclaw_openclaw_config_preflight(args, run=True)

    assert record["ok"] is True
    assert record["ran"] is True
    assert record["returncode"] == 0
    assert record["model"] == "openai-direct/test-mini"
    assert record["provider"] == "openai-direct"
    assert record["model_id"] == "test-mini"
    assert record["errors"] == []
    assert all(check["ok"] for check in record["checks"])


def test_execute_blocks_before_openclaw_when_nemoclaw_openclaw_config_is_stale(
    tmp_path,
    capsys,
    monkeypatch,
):
    module = load_module()
    report = tmp_path / "external_action_approval.verify.json"
    source_packet = write_external_action_approval_report(report)
    real_run_subprocess = module.run_subprocess

    def fake_run_subprocess(command, *, cwd):
        if command[:4] != ["nemoclaw", "sandbox", "exec", "nejumi-taiwan"]:
            return real_run_subprocess(command, cwd=cwd)
        assert command[-2:] == ["cat", "/sandbox/.openclaw/openclaw.json"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "models": {
                        "providers": {
                            "openai-direct": {
                                "models": [{"id": "other-model"}],
                            }
                        }
                    },
                    "plugins": {"entries": {"weave": {"enabled": True}}},
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(module, "run_subprocess", fake_run_subprocess)

    try:
        module.main(
            [
                "--execute",
                "--model",
                "openai-direct/gpt-4.1-nano-2025-04-14",
                "--nemoclaw-sandbox",
                "nejumi-taiwan",
                "--canary-id",
                "TEST_CANARY_STALE_CONFIG",
                "--output-dir",
                str(tmp_path),
                "--verify-sleep-seconds",
                "0",
                "--external-action-approval-source-packet-json",
                str(source_packet),
                "--external-action-approval-report-json",
                str(report),
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("main should block before OpenClaw when sandbox config is stale")

    payload = json.loads(capsys.readouterr().out)
    assert payload["executed"] is False
    assert payload["blocked_before_openclaw"] is True
    preflight = payload["nemoclaw_openclaw_config_preflight"]
    assert preflight["ok"] is False
    assert preflight["ran"] is True
    assert any(
        check["name"]
        == "NeMoClaw sandbox OpenClaw model is registered: openai-direct/gpt-4.1-nano-2025-04-14"
        and check["ok"] is False
        for check in preflight["checks"]
    )
    command_result_path = (
        tmp_path
        / "plans"
        / "weave_agents_content_canary_TEST_CANARY_STALE_CONFIG.command_result.json"
    )
    assert command_result_path.exists()
    command_result = json.loads(command_result_path.read_text(encoding="utf-8"))
    assert command_result["ok"] is False
    assert command_result["blocked_before_openclaw"] is True
    assert command_result["paid_api_attempted"] is False
    assert command_result["failure"]["kind"] == "nemoclaw_config_preflight_failed"
    plan = json.loads(Path(payload["plan_file"]).read_text(encoding="utf-8"))
    assert command_result["task_id"] == "weave_agents_content_canary_TEST_CANARY_STALE_CONFIG"
    assert command_result["model"] == "openai-direct/gpt-4.1-nano-2025-04-14"
    assert command_result["run_command"] == plan["run_command"]
    assert command_result["run_command_sha256"] == plan["run_command_sha256"]
    gate = json.loads(Path(payload["gate_result_file"]).read_text(encoding="utf-8"))
    assert gate["status"] == "nemoclaw_config_preflight_failed"
    assert gate["paid_api_attempted"] is False


def test_classify_openclaw_failure_detects_provider_quota(tmp_path):
    module = load_module()
    sidecar = tmp_path / "openclaw_result.json"
    sidecar.write_text(
        json.dumps(
            {
                "stderr": (
                    "[openai-transport] status=undefined code=insufficient_quota "
                    "message=You exceeded your current quota"
                )
            }
        ),
        encoding="utf-8",
    )

    failure = module.classify_openclaw_failure(sidecar, "")

    assert failure == {
        "kind": "provider_quota",
        "detail": "provider returned insufficient_quota before a scoreable canary trace was produced",
    }


def test_classify_openclaw_failure_detects_unknown_model(tmp_path):
    module = load_module()

    failure = module.classify_openclaw_failure(
        tmp_path / "missing.json",
        "FailoverError: Unknown model: openai/gpt-5.4-mini-2026-03-17",
    )

    assert failure["kind"] == "model_not_found"
