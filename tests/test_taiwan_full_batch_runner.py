import importlib.util
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "scripts" / "tools"


def load_module():
    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))
    path = TOOLS_DIR / "run_taiwan_full_eval_batch.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_run_eval_base_config_arg_accepts_repository_relative_path(
    tmp_path, monkeypatch
):
    module = load_module()
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    base_config = config_dir / "base.yaml"
    base_config.write_text("run: {}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    normalized = module.normalize_run_eval_base_config_arg("configs/base.yaml")

    assert normalized == str(base_config.resolve())


def test_normalize_run_eval_base_config_arg_preserves_config_dir_relative_name(
    tmp_path, monkeypatch
):
    module = load_module()
    monkeypatch.chdir(tmp_path)

    assert (
        module.normalize_run_eval_base_config_arg("base_config_taiwan.yaml")
        == "base_config_taiwan.yaml"
    )


def test_selected_config_binding_records_cash_cost_exemption(tmp_path):
    module = load_module()
    config = tmp_path / "wandb.yaml"
    config.write_text(
        "\n".join(
            [
                "model:",
                "  pretrained_model_name_or_path: provider/model",
                "execution:",
                "  cash_cost_exempt: true",
                "  cash_cost_exempt_reason: employee inference account",
                "agentic_math:",
                "  openclaw_model: wandb-inference/provider/model",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    [binding] = module.collect_selected_config_model_bindings(
        [config],
        phase="agentic",
    )

    assert binding["cash_cost_exempt"] is True
    assert binding["cash_cost_exempt_reason"] == "employee inference account"
    assert "wandb-inference/provider/model" in binding["identifiers"]


def test_cash_cost_exemption_flag_rejects_non_exempt_model(
    tmp_path, monkeypatch
):
    module = load_module()
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        "\n".join(
            [
                "models:",
                "  - slug: paid-model",
                "    source_config: config-gpt-4.1-mini-2025-04-14.yaml",
                "    run_name: paid-model",
                "    openclaw_model: openai-direct/gpt-4.1-mini-2025-04-14",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "outputs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_taiwan_full_eval_batch.py",
            "--manifest",
            str(manifest),
            "--model",
            "paid-model",
            "--prepare-only",
            "--phase",
            "agentic",
            "--generated-config-dir",
            str(tmp_path / "generated"),
            "--output-root",
            str(output_root),
            "--allow-cash-cost-exempt-execution",
        ],
    )

    with pytest.raises(SystemExit, match="cash_cost_exempt"):
        module.main()

    review = json.loads(
        (output_root / "agentic_paid_run_review.json").read_text(
            encoding="utf-8"
        )
    )
    assert review["status"] == "cash_cost_exemption_invalid"
    assert review["selected_configs_cash_cost_exempt"] is False


def test_cash_cost_exempt_prepare_only_needs_no_budget_or_approval(
    tmp_path, monkeypatch
):
    module = load_module()
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        "\n".join(
            [
                "models:",
                "  - slug: glm-wandb",
                "    source_config: config-zai-glm-5_2-wandb-inference.yaml",
                "    run_name: glm-wandb",
                "    openclaw_model: wandb-inference/zai-org/GLM-5.2",
                "    cash_cost_exempt: true",
                "    cash_cost_exempt_reason: employee inference account",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "outputs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_taiwan_full_eval_batch.py",
            "--manifest",
            str(manifest),
            "--model",
            "glm-wandb",
            "--prepare-only",
            "--phase",
            "agentic",
            "--generated-config-dir",
            str(tmp_path / "generated"),
            "--output-root",
            str(output_root),
            "--allow-cash-cost-exempt-execution",
        ],
    )

    module.main()

    review = json.loads(
        (output_root / "agentic_paid_run_review.json").read_text(
            encoding="utf-8"
        )
    )
    assert review["status"] == "prepared"
    assert review["selected_configs_cash_cost_exempt"] is True
    assert review["cash_cost_exempt_execution"] is True
    assert review["requires_paid_model_api"] is False
    assert review["will_call_model_api"] is False
    assert review["will_call_paid_model_api"] is False
    requirements = review["completion_requirements"]
    assert (
        requirements["pre_run_budget_estimate"][
            "required_before_paid_execution"
        ]
        is False
    )
    assert (
        requirements["external_action_approval"][
            "required_before_external_action"
        ]
        is False
    )


def write_budget_estimate(
    path: Path,
    *,
    target_model: str = "openai-direct/gpt-4.1-mini-2025-04-14",
    target_models: list[str] | None = None,
    estimated_total_usd: dict[str, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": "2026-06-28T00:00:00+00:00",
        "target_model": target_model,
        "price_per_million_tokens": {
            "input": 0.40,
            "output": 1.60,
            "cacheRead": 0.10,
            "cacheWrite": 0.0,
        },
        "estimated_total_usd": estimated_total_usd or {"low": 1.0, "mid": 2.0, "high": 3.0},
        "pricing_source_url": "https://openai.com/index/gpt-4-1/",
        "pricing_note": "provider dashboards are authoritative",
    }
    if target_models is not None:
        payload["target_models"] = target_models
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


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
                "generated_at": time.time(),
                "approval_packet_json": "temp/reviewed_external_action_approval.json",
                "external_action_checklist_sha256": "b" * 64,
                "required_approval_count": 1,
                "granted_approval_count": 1,
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
                        "approved_model_scope": "OpenAI mini canary",
                        "errors": [],
                    }
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


def passing_weave_content_canary_gate_payload() -> dict:
    return {
        "ok": True,
        "gate": "weave_agents_content_canary",
        "status": "passed",
        "generated_at": time.time(),
        "model": "openai-direct/test-mini",
        "canary_id": "CANARY",
        "task_id": "weave_agents_content_canary_CANARY",
        "agent_name": "nejumi-taiwan-openclaw",
        "entity": "llm-leaderboard",
        "project": "tc-leaderboard",
        "expected_request_models": ["openai-direct/test-mini", "test-mini"],
        "observed_request_models": ["test-mini"],
        "span_request_models": ["test-mini"],
        "request_model_proven": True,
        "nemoclaw": {
            "required": True,
            "enabled": True,
            "bin": "nemoclaw",
            "sandbox": "nejumi-taiwan",
            "workdir": "/sandbox",
        },
        "nemoclaw_openclaw_config_preflight": {
            "required_before_openclaw": True,
            "ran": True,
            "ok": True,
            "model": "openai-direct/test-mini",
            "provider": "openai-direct",
            "model_id": "test-mini",
            "config_path": "/sandbox/.openclaw/openclaw.json",
            "command": [
                "nemoclaw",
                "sandbox",
                "exec",
                "nejumi-taiwan",
                "--no-tty",
                "--timeout",
                "30",
                "--",
                "cat",
                "/sandbox/.openclaw/openclaw.json",
            ],
            "returncode": 0,
            "checks": [
                {
                    "name": "NeMoClaw sandbox OpenClaw config is readable",
                    "ok": True,
                    "detail": "bytes=1234",
                }
            ],
            "errors": [],
        },
        "will_call_paid_model_api": True,
        "paid_api_attempted": True,
        "command_ok": True,
        "command_returncode": 0,
        "command_result_contract_issues": [],
        "expected_required_texts": [
            "CANARY",
            "CANARY_RESULT CANARY 91",
            "openclaw_config_source: /sandbox/.openclaw/openclaw.json",
        ],
        "plan_required_text_validation_issues": [],
        "weave_verifier_ok": True,
        "weave_verifier_schema_version": 1,
        "weave_verifier_latest_trace_id": "trace-1",
        "weave_verifier_validation_issues": [],
        "agents_diagnostic_ok": True,
        "agents_diagnostic_schema_version": 1,
        "agents_diagnostic_latest_trace_id": "trace-1",
        "agents_diagnostic_validation_issues": [],
        "content_capture_health": {
            "message_spans_with_input": 1,
            "tool_spans_with_content": 1,
            "spans_with_valid_timestamps": 3,
            "spans_with_invalid_timestamps": 0,
            "request_model_count": 1,
        },
        "failed_checks": [],
        "paths": {
            "plan_file": "outputs/weave_agents_content_canary/plans/canary.json",
            "command_result_file": "outputs/weave_agents_content_canary/plans/canary.command_result.json",
            "command_result_exists": True,
            "verifier_json": "outputs/weave_agents_content_canary/verifier/canary/attempt_001.json",
            "verifier_json_exists": True,
            "agents_diagnostic_json": "outputs/weave_agents_content_canary/agents_diagnostics/canary.agents.json",
            "agents_diagnostic_json_exists": True,
            "expected_sidecar": "outputs/weave_agents_content_canary/agentic_math/canary/openclaw_result.json",
            "prompt_file": "outputs/weave_agents_content_canary/prompts/canary.md",
        },
    }


def write_passing_weave_content_canary_gate(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(passing_weave_content_canary_gate_payload()) + "\n",
        encoding="utf-8",
    )
    return path


def test_default_wandb_verify_benchmarks_by_phase():
    module = load_module()

    assert module.default_wandb_verify_benchmarks("full") == [
        "agentic_math",
        "agentic_swe",
        "taiwan_full",
    ]
    assert module.default_wandb_verify_benchmarks("agentic") == [
        "agentic_math",
        "agentic_swe",
    ]
    assert module.default_wandb_verify_benchmarks("nonagentic") == []


def test_stream_run_appends_attempt_history(tmp_path):
    module = load_module()
    log_path = tmp_path / "run.log"

    assert module.stream_run(
        [sys.executable, "-c", "print('first')"],
        log_path,
        dict(os.environ),
    ) == 0
    assert module.stream_run(
        [sys.executable, "-c", "print('second')"],
        log_path,
        dict(os.environ),
    ) == 0

    log_text = log_path.read_text(encoding="utf-8")
    assert log_text.count("attempt started at") == 2
    assert log_text.count("attempt finished at") == 2
    assert "first" in log_text
    assert "second" in log_text
    assert log_text.index("first") < log_text.index("second")


def test_build_wandb_verify_command_adds_expected_totals_and_full_options():
    module = load_module()

    agentic = module.build_wandb_verify_command(
        python="python3",
        run_id="run-1",
        benchmark="agentic_swe",
        num_few_shots=2,
        include_pending=False,
        require_aggregate=True,
    )
    full = module.build_wandb_verify_command(
        python="python3",
        run_id="run-1",
        benchmark="taiwan_full",
        num_few_shots=2,
        include_pending=True,
        require_aggregate=False,
    )

    assert agentic[-2:] == ["--expected-total", "80"]
    assert "--num-few-shots" not in agentic
    assert "--benchmark" in full
    assert "taiwan_full" in full
    assert ["--num-few-shots", "2"] == full[-4:-2]
    assert full[-2:] == ["--include-pending", "--no-require-aggregate"]


def test_build_wandb_verify_command_can_require_nemoclaw_session_audit():
    module = load_module()

    command = module.build_wandb_verify_command(
        python="python3",
        run_id="run-1",
        benchmark="agentic_math",
        num_few_shots=2,
        include_pending=False,
        require_aggregate=True,
        require_nemoclaw_session_audit=True,
    )

    assert "--require-nemoclaw-session-audit" in command


def test_build_wandb_verify_command_can_require_assorted_contract():
    module = load_module()
    command = module.build_wandb_verify_command(
        python="python3",
        run_id="run-1",
        benchmark="agentic_swe",
        num_few_shots=2,
        include_pending=False,
        require_aggregate=True,
        require_agentic_swe_assorted=True,
        expected_total_override=79,
    )
    assert "--require-agentic-swe-assorted" in command
    assert command[command.index("--expected-total") + 1] == "79"


def test_expected_total_for_config_uses_math_limit_and_assorted_tiers(tmp_path):
    module = load_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
run:
  agentic_swe_assorted: true
agentic_math:
  limit: 50
agentic_swe_assorted:
  low_limit: 36
  middle_limit: 35
  high_limit: 8
""".strip()
        + "\n",
        encoding="utf-8",
    )
    assert module.expected_total_for_config(config_path, "agentic_math") == 50
    assert module.expected_total_for_config(config_path, "agentic_swe") == 79


def test_build_wandb_verify_command_can_write_json_with_env_file(tmp_path):
    module = load_module()
    output = tmp_path / "completion.json"
    env_file = tmp_path / ".env"

    command = module.build_wandb_verify_command(
        python="python3",
        run_id="run-1",
        benchmark="agentic_math",
        num_few_shots=2,
        include_pending=False,
        require_aggregate=True,
        env_file=env_file,
        json_path=output,
    )

    assert command[-4:] == ["--env-file", str(env_file), "--json", str(output)]


def test_build_run_eval_preflight_records_include_nonexecuting_command(tmp_path):
    module = load_module()
    config = tmp_path / "config-taiwan-full-gpt-4_1-mini.yaml"
    config.write_text("run:\n  agentic_math: true\n", encoding="utf-8")

    records = module.build_run_eval_preflight_records(
        [config],
        phase="agentic",
        python="python3",
        base_config="base_config_taiwan.yaml",
        output_root=tmp_path / "outputs",
    )

    assert len(records) == 1
    record = records[0]
    assert record["config"] == str(config.resolve())
    assert record["required_before_run_eval"] is True
    assert record["expected_scheduled_evaluators"] == [
        "agentic_math",
        "agentic_swe_assorted",
    ]
    assert record["output_json"].endswith(
        "run_eval_preflight/agentic-gpt-4_1-mini.json"
    )
    command = record["command"]
    assert command[:2] == ["python3", "scripts/run_eval.py"]
    assert "--preflight" in command
    assert "--preflight-json" in command
    assert "base_config_taiwan.yaml" in command


def test_run_eval_preflight_phase_validation_rejects_missing_taiwan_evaluators():
    module = load_module()
    validation = module.validate_run_eval_preflight_phase(
        {
            "ok": True,
            "scheduled_evaluators": ["agentic_math", "agentic_swe_assorted"],
        },
        "full",
    )

    assert validation["ok"] is False
    assert "hallulens_zh_tw" in validation["missing_scheduled_evaluators"]
    assert "ifeval_zh_tw" in validation["missing_scheduled_evaluators"]
    assert "ts_bench" in validation["missing_scheduled_evaluators"]
    assert "tceval_v2" in validation["missing_scheduled_evaluators"]
    assert "script_adherence" in validation["missing_scheduled_evaluators"]
    assert "aggregate_taiwan" in validation["missing_scheduled_evaluators"]


def test_build_wandb_verify_command_adds_expected_run_metadata():
    module = load_module()

    command = module.build_wandb_verify_command(
        python="python3",
        run_id="run-1",
        benchmark="agentic_math",
        num_few_shots=2,
        include_pending=False,
        require_aggregate=True,
        expected_run_config={
            "model.pretrained_model_name_or_path": "gpt-4.1-mini-2025-04-14",
            "run.agentic_math": True,
        },
        expected_run_tags=["taiwan-canary"],
        expected_run_group="tw-canary",
        expected_run_job_type="evaluation",
    )

    assert "--expected-run-config" in command
    assert "model.pretrained_model_name_or_path=\"gpt-4.1-mini-2025-04-14\"" in command
    assert "run.agentic_math=true" in command
    assert ["--expected-run-tag", "taiwan-canary"] == command[-6:-4]
    assert ["--expected-run-group", "tw-canary"] == command[-4:-2]
    assert ["--expected-run-job-type", "evaluation"] == command[-2:]


def test_wandb_verify_config_expectations_read_generated_config(tmp_path):
    module = load_module()
    config = tmp_path / "config-taiwan-full-gpt-mini.yaml"
    config.write_text(
        "\n".join(
            [
                "wandb:",
                "  run_name: 'taiwan/full/openai/gpt-4.1-mini: canary'",
                "model:",
                "  pretrained_model_name_or_path: gpt-4.1-mini-2025-04-14",
                "run:",
                "  agentic_math: true",
                "  agentic_swe_assorted: true",
                "  aggregate_taiwan: false",
                "agentic_math:",
                "  openclaw_model: openai-direct/gpt-4.1-mini-2025-04-14",
                "  nemoclaw_sandbox: nejumi-taiwan",
                "  nemoclaw_openclaw_config_path: /sandbox/.openclaw/openclaw.json",
                "  use_task_agent: true",
                "  deny_tool:",
                "    - web_search",
                "    - web_fetch",
                "  deny_argument_pattern:",
                "    - https?://",
                "agentic_swe_assorted:",
                "  openclaw_model: openai-direct/gpt-4.1-mini-2025-04-14",
                "  nemoclaw_sandbox: nejumi-taiwan",
                "  nemoclaw_openclaw_config_path: /sandbox/.openclaw/openclaw.json",
                "  nemoclaw_checkout_transfer_mode: copy",
                "  deny_tool:",
                "    - web_search",
                "    - web_fetch",
                "  deny_argument_pattern:",
                "    - https?://",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    math_expectations = module.wandb_verify_config_expectations(
        config,
        benchmark="agentic_math",
    )
    swe_expectations = module.wandb_verify_config_expectations(
        config,
        benchmark="agentic_swe",
    )

    assert math_expectations == {
        "agentic_math.deny_argument_pattern": ["https?://"],
        "agentic_math.deny_tool": ["web_search", "web_fetch"],
        "agentic_math.nemoclaw_openclaw_config_path": "/sandbox/.openclaw/openclaw.json",
        "agentic_math.nemoclaw_sandbox": "nejumi-taiwan",
        "agentic_math.openclaw_model": "openai-direct/gpt-4.1-mini-2025-04-14",
        "agentic_math.use_task_agent": True,
        "model.pretrained_model_name_or_path": "gpt-4.1-mini-2025-04-14",
        "run.agentic_math": True,
        "wandb.run_name": "taiwan/full/openai/gpt-4.1-mini: canary",
    }
    assert swe_expectations == {
        "model.pretrained_model_name_or_path": "gpt-4.1-mini-2025-04-14",
        "run.agentic_swe_assorted": True,
        "agentic_swe_assorted.deny_argument_pattern": ["https?://"],
        "agentic_swe_assorted.deny_tool": ["web_search", "web_fetch"],
        "agentic_swe_assorted.nemoclaw_checkout_transfer_mode": "copy",
        "agentic_swe_assorted.nemoclaw_openclaw_config_path": "/sandbox/.openclaw/openclaw.json",
        "agentic_swe_assorted.nemoclaw_sandbox": "nejumi-taiwan",
        "agentic_swe_assorted.openclaw_model": "openai-direct/gpt-4.1-mini-2025-04-14",
        "wandb.run_name": "taiwan/full/openai/gpt-4.1-mini: canary",
    }


def test_run_wandb_completion_verification_writes_payload(tmp_path, monkeypatch):
    module = load_module()

    def fake_run(command, cwd, env, text, capture_output, check, timeout):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"ok": True, "benchmark": "agentic_math", "checks": []}),
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    output_path = tmp_path / "completion.json"

    payload = module.run_wandb_completion_verification(
        python="python3",
        run_id="run-1",
        benchmark="agentic_math",
        output_path=output_path,
        env={"WANDB_API_KEY": "test"},
        num_few_shots=2,
        include_pending=False,
        require_aggregate=True,
    )

    assert payload["ok"] is True
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["benchmark"] == "agentic_math"
    assert written["returncode"] == 0
    assert written["command"][0] == "python3"
    assert written["payload_ok"] is True
    assert written["returncode_ok"] is True


def test_run_wandb_completion_verification_requires_zero_returncode(tmp_path, monkeypatch):
    module = load_module()

    def fake_run(command, cwd, env, text, capture_output, check, timeout):
        return SimpleNamespace(
            returncode=7,
            stdout=json.dumps({"ok": True, "benchmark": "agentic_math", "checks": []}),
            stderr="process failed after writing ok",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    output_path = tmp_path / "completion.json"

    payload = module.run_wandb_completion_verification(
        python="python3",
        run_id="run-1",
        benchmark="agentic_math",
        output_path=output_path,
        env={"WANDB_API_KEY": "test"},
        num_few_shots=2,
        include_pending=False,
        require_aggregate=True,
    )

    assert payload["ok"] is False
    assert payload["payload_ok"] is True
    assert payload["returncode_ok"] is False
    assert payload["returncode"] == 7
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["ok"] is False


def test_run_wandb_completion_verification_records_timeout(
    tmp_path, monkeypatch
):
    module = load_module()

    def fake_run(command, **_kwargs):
        raise module.subprocess.TimeoutExpired(
            command,
            timeout=12,
            output=b"partial verifier output",
            stderr=b"provider did not respond",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    output_path = tmp_path / "completion.json"

    payload = module.run_wandb_completion_verification(
        python="python3",
        run_id="run-1",
        benchmark="agentic_math",
        output_path=output_path,
        env={"WANDB_API_KEY": "test"},
        num_few_shots=2,
        include_pending=False,
        require_aggregate=True,
        timeout_seconds=12,
    )

    assert payload["ok"] is False
    assert payload["timeout"] is True
    assert payload["returncode"] == 124
    assert payload["stdout"] == "partial verifier output"
    assert payload["stderr"] == "provider did not respond"
    assert json.loads(output_path.read_text(encoding="utf-8"))["timeout"] is True


def test_build_weave_agents_verify_command_adds_strict_trace_options():
    module = load_module()

    command = module.build_weave_agents_verify_command(
        python="python3",
        agent_name="nejumi-taiwan-openclaw",
        limit=50,
        require_content=True,
        require_tool_span=True,
        require_tool_content=True,
        require_usage=True,
        conversation_id_contains="agentic_math",
        expected_request_models=[
            "openai-direct/gpt-4.1-mini-2025-04-14",
            "gpt-4.1-mini-2025-04-14",
        ],
        required_texts=[
            "openclaw_config_source: /sandbox/.openclaw/openclaw.json",
        ],
    )

    assert command[:2] == ["python3", str(module.WEAVE_AGENTS_VERIFY_RUNNER)]
    assert ["--agent-name", "nejumi-taiwan-openclaw"] == command[2:4]
    assert ["--limit", "50"] == command[4:6]
    assert "--no-require-content" not in command
    assert "--require-tool-span" in command
    assert "--require-tool-content" in command
    assert "--require-usage" in command
    required_text_index = command.index("--require-text")
    assert [
        "--require-text",
        "openclaw_config_source: /sandbox/.openclaw/openclaw.json",
    ] == command[required_text_index : required_text_index + 2]
    conversation_filter_index = command.index("--conversation-id-contains")
    assert ["--conversation-id-contains", "agentic_math"] == command[
        conversation_filter_index : conversation_filter_index + 2
    ]
    assert command.count("--expected-request-model") == 2
    assert "gpt-4.1-mini-2025-04-14" in command
    assert "openai-direct/gpt-4.1-mini-2025-04-14" in command


def test_weave_required_texts_for_phase_requires_nemoclaw_config_source():
    module = load_module()

    assert module.weave_required_texts_for_phase(
        phase="agentic",
        require_nemoclaw_agentic_config=True,
    ) == ["openclaw_config_source: /sandbox/.openclaw/openclaw.json"]
    assert module.weave_required_texts_for_phase(
        phase="full",
        require_nemoclaw_agentic_config=True,
    ) == ["openclaw_config_source: /sandbox/.openclaw/openclaw.json"]
    assert module.weave_required_texts_for_phase(
        phase="agentic",
        require_nemoclaw_agentic_config=False,
    ) == []


def test_weave_expected_request_models_reads_generated_config(tmp_path):
    module = load_module()
    config = tmp_path / "config-taiwan-full-gpt-mini.yaml"
    config.write_text(
        "\n".join(
            [
                "api: openai_responses",
                "model:",
                "  pretrained_model_name_or_path: gpt-4.1-mini-2025-04-14",
                "run:",
                "  agentic_math: true",
                "  agentic_swe_assorted: true",
                "agentic_math:",
                "  openclaw_model: openai-direct/gpt-4.1-mini-2025-04-14",
                "agentic_swe_assorted:",
                "  openclaw_model: openai-direct/gpt-4.1-mini-2025-04-14",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert module.weave_expected_request_models(config, phase="agentic") == [
        "gpt-4.1-mini-2025-04-14",
        "openai-direct/gpt-4.1-mini-2025-04-14",
    ]


def test_weave_expected_request_models_include_wandb_inference_remainder(tmp_path):
    module = load_module()
    config = tmp_path / "config-taiwan-full-glm52-wandb.yaml"
    config.write_text(
        "\n".join(
            [
                "api: openai_compatible",
                "model:",
                "  pretrained_model_name_or_path: zai-org/GLM-5.2",
                "run:",
                "  agentic_math: true",
                "  agentic_swe_assorted: true",
                "agentic_math:",
                "  openclaw_model: wandb-inference/zai-org/GLM-5.2",
                "agentic_swe_assorted:",
                "  openclaw_model: wandb-inference/zai-org/GLM-5.2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert module.weave_expected_request_models(config, phase="agentic") == [
        "GLM-5.2",
        "wandb-inference/zai-org/GLM-5.2",
        "zai-org/GLM-5.2",
    ]


def test_resolve_weave_conversation_id_contains_defaults_to_wandb_run_id():
    module = load_module()

    assert (
        module.resolve_weave_conversation_id_contains(None, wandb_run_id="run-1")
        == "run-1"
    )


def test_resolve_weave_conversation_id_contains_expands_run_id_placeholder():
    module = load_module()

    assert (
        module.resolve_weave_conversation_id_contains(
            "agentic_math/{wandb_run_id}",
            wandb_run_id="run-1",
        )
        == "agentic_math/run-1"
    )


def test_resolve_weave_conversation_id_contains_rejects_unscoped_template():
    module = load_module()

    try:
        module.resolve_weave_conversation_id_contains(
            "agentic_math",
            wandb_run_id="run-1",
        )
    except ValueError as exc:
        assert "{wandb_run_id}" in str(exc)
    else:
        raise AssertionError("expected unscoped Weave conversation filter to fail")


def test_run_weave_agents_verification_writes_payload(tmp_path, monkeypatch):
    module = load_module()

    def fake_run(command, cwd, env, text, capture_output, check, timeout):
        return SimpleNamespace(
            returncode=1,
            stdout=json.dumps({"ok": False, "agent_name": "nejumi-taiwan-openclaw", "checks": []}),
            stderr="missing content",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    output_path = tmp_path / "weave-agents.json"

    payload = module.run_weave_agents_verification(
        python="python3",
        agent_name="nejumi-taiwan-openclaw",
        limit=20,
        output_path=output_path,
        env={"WANDB_API_KEY": "test"},
        require_content=True,
        require_tool_span=False,
        require_tool_content=False,
        require_usage=False,
    )

    assert payload["ok"] is False
    assert payload["returncode"] == 1
    assert payload["payload_ok"] is False
    assert payload["returncode_ok"] is False
    assert payload["verification_schema_version"] == 1
    assert isinstance(payload["generated_at"], float)
    assert payload["required_evidence"]["content_required"] is True
    assert payload["stderr"] == "missing content"
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["agent_name"] == "nejumi-taiwan-openclaw"
    assert written["verification_schema_version"] == 1


def test_run_weave_agents_verification_requires_zero_returncode(tmp_path, monkeypatch):
    module = load_module()

    def fake_run(command, cwd, env, text, capture_output, check, timeout):
        return SimpleNamespace(
            returncode=9,
            stdout=json.dumps({"ok": True, "agent_name": "nejumi-taiwan-openclaw", "checks": []}),
            stderr="trace verifier failed after ok payload",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    output_path = tmp_path / "weave-agents.json"

    payload = module.run_weave_agents_verification(
        python="python3",
        agent_name="nejumi-taiwan-openclaw",
        limit=20,
        output_path=output_path,
        env={"WANDB_API_KEY": "test"},
        require_content=True,
        require_tool_span=False,
        require_tool_content=False,
        require_usage=False,
    )

    assert payload["ok"] is False
    assert payload["payload_ok"] is True
    assert payload["returncode_ok"] is False
    assert payload["returncode"] == 9
    assert payload["verification_schema_version"] == 1
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["ok"] is False


def test_weave_content_canary_gate_blocks_paid_agentic_when_failed(tmp_path):
    module = load_module()
    gate = tmp_path / "content_canary.gate.json"
    gate.write_text(
        json.dumps(
            {
                "ok": False,
                "status": "provider_failure",
                "failure_kind": "provider_quota",
                "detail": "quota exhausted",
            }
        ),
        encoding="utf-8",
    )

    record = module.build_weave_content_canary_gate_record(
        gate_path=gate,
        require_gate=True,
        phase="agentic",
        will_call_paid_model_api=True,
    )

    assert record["enforced"] is True
    assert record["passed"] is False
    assert record["blocking_ok"] is False
    assert record["status"] == "provider_failure"
    assert record["failure_kind"] == "provider_quota"


def test_weave_content_canary_gate_does_not_block_prepare_only(tmp_path):
    module = load_module()
    gate = tmp_path / "content_canary.gate.json"
    gate.write_text(
        json.dumps({"ok": False, "status": "content_missing"}),
        encoding="utf-8",
    )

    record = module.build_weave_content_canary_gate_record(
        gate_path=gate,
        require_gate=True,
        phase="agentic",
        will_call_paid_model_api=False,
    )

    assert record["required"] is True
    assert record["enforced"] is False
    assert record["passed"] is False
    assert record["blocking_ok"] is True


def test_weave_content_canary_gate_accepts_passed_json(tmp_path):
    module = load_module()
    gate = write_passing_weave_content_canary_gate(tmp_path / "content_canary.gate.json")

    record = module.build_weave_content_canary_gate_record(
        gate_path=gate,
        require_gate=True,
        phase="full",
        will_call_paid_model_api=True,
    )

    assert record["enforced"] is True
    assert record["passed"] is True
    assert record["blocking_ok"] is True
    assert record["status"] == "passed"
    assert record["fresh"] is True
    assert record["native_weave_contract_ok"] is True
    assert record["native_weave_contract_issues"] == []


def test_weave_content_canary_gate_rejects_hand_edited_passed_json(tmp_path):
    module = load_module()
    gate = tmp_path / "content_canary.gate.json"
    gate.write_text(
        json.dumps(
            {
                "ok": True,
                "status": "passed",
                "generated_at": time.time(),
                "model": "openai-direct/test-mini",
                "canary_id": "CANARY",
            }
        ),
        encoding="utf-8",
    )

    record = module.build_weave_content_canary_gate_record(
        gate_path=gate,
        require_gate=True,
        phase="agentic",
        will_call_paid_model_api=True,
    )

    assert record["enforced"] is True
    assert record["passed"] is True
    assert record["blocking_ok"] is False
    assert record["status"] == "weave_gate_contract_invalid"
    assert record["native_weave_contract_ok"] is False
    assert "weave_verifier_ok must be true" in record["native_weave_contract_issues"]
    assert "paths must be an object" in record["native_weave_contract_issues"]


def test_weave_content_canary_gate_blocks_stale_passed_json(tmp_path):
    module = load_module()
    gate = tmp_path / "content_canary.gate.json"
    gate.write_text(
        json.dumps(
            {
                **passing_weave_content_canary_gate_payload(),
                "generated_at": time.time() - 90_000,
            }
        ),
        encoding="utf-8",
    )

    record = module.build_weave_content_canary_gate_record(
        gate_path=gate,
        require_gate=True,
        phase="agentic",
        will_call_paid_model_api=True,
        max_age_seconds=86_400,
    )

    assert record["enforced"] is True
    assert record["passed"] is True
    assert record["fresh"] is False
    assert record["blocking_ok"] is False
    assert record["status"] == "stale"


def test_weave_content_canary_gate_can_disable_freshness(tmp_path):
    module = load_module()
    gate = tmp_path / "content_canary.gate.json"
    gate.write_text(
        json.dumps(
            {
                **passing_weave_content_canary_gate_payload(),
                "generated_at": time.time() - 90_000,
            }
        ),
        encoding="utf-8",
    )

    record = module.build_weave_content_canary_gate_record(
        gate_path=gate,
        require_gate=True,
        phase="agentic",
        will_call_paid_model_api=True,
        max_age_seconds=None,
    )

    assert record["fresh"] is True
    assert record["blocking_ok"] is True


def test_paid_agentic_run_rejects_hand_edited_weave_gate_before_run_eval(
    tmp_path,
    monkeypatch,
):
    module = load_module()
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        "\n".join(
            [
                "models:",
                "  - slug: gpt-4_1-mini-openai-direct-canary",
                "    source_config: config-gpt-4.1-mini-2025-04-14.yaml",
                "    run_name: 'taiwan/full/openai/gpt-4.1-mini: canary'",
                "    openclaw_model: 'openai-direct/gpt-4.1-mini-2025-04-14'",
                "    agentic_thinking: 'off'",
                "    swe_thinking: 'off'",
                "    canary: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "outputs"
    budget = output_root / "openai_canary_budget_estimate.json"
    write_budget_estimate(budget)
    approval = output_root / "external_action_approval.verify.json"
    source_packet = write_external_action_approval_report(approval)
    gate = output_root / "hand_edited_content_canary.gate.json"
    gate.write_text(
        json.dumps(
            {
                "ok": True,
                "status": "passed",
                "generated_at": time.time(),
                "model": "openai-direct/test-mini",
                "canary_id": "CANARY",
            }
        ),
        encoding="utf-8",
    )

    def unexpected_stream_run(command, log_path, env):  # pragma: no cover - assertion path
        raise AssertionError(f"run_eval should not be reached: {command}")

    monkeypatch.setattr(module, "stream_run", unexpected_stream_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_taiwan_full_eval_batch.py",
            "--manifest",
            str(manifest),
            "--canary",
            "--phase",
            "agentic",
            "--generated-config-dir",
            str(tmp_path / "generated"),
            "--output-root",
            str(output_root),
            "--agentic-math-nemoclaw-sandbox",
            "nejumi-taiwan",
            "--agentic-math-nemoclaw-openclaw-config-path",
            "/sandbox/.openclaw/openclaw.json",
            "--agentic-swe-assorted-nemoclaw-sandbox",
            "nejumi-taiwan",
            "--agentic-swe-assorted-nemoclaw-openclaw-config-path",
            "/sandbox/.openclaw/openclaw.json",
            "--agentic-swe-assorted-nemoclaw-checkout-transfer-mode",
            "copy",
            "--require-nemoclaw-agentic-config",
            "--wandb-run-id-prefix",
            "twcanary-test",
            "--verify-wandb-completion",
            "--verify-weave-agents",
            "--weave-agents-require-tool-span",
            "--weave-agents-require-tool-content",
            "--weave-agents-require-usage",
            "--run-purpose",
            "paid content canary guard test",
            "--expected-cost-band",
            "$1-$3",
            "--pre-run-budget-estimate-json",
            str(budget),
            "--external-action-approval-report-json",
            str(approval),
            "--external-action-approval-source-packet-json",
            str(source_packet),
            "--weave-content-canary-gate",
            str(gate),
            "--require-weave-content-canary",
        ],
    )

    try:
        module.main()
    except SystemExit as exc:
        assert "Weave content canary gate failed" in str(exc)
    else:
        raise AssertionError("expected hand-edited content canary gate to block paid run")

    review = json.loads(
        (output_root / "canary_agentic_paid_run_review.json").read_text(
            encoding="utf-8"
        )
    )
    assert review["status"] == "weave_content_canary_gate_failed"
    assert review["blocking_reason"]["status"] == "weave_gate_contract_invalid"
    assert review["blocking_reason"]["native_weave_contract_ok"] is False


def test_pre_run_budget_estimate_record_validates_schema_and_hash(tmp_path):
    module = load_module()
    budget = tmp_path / "budget.json"
    write_budget_estimate(budget, target_model="openai-direct/gpt-4.1-mini-2025-04-14")

    record = module.build_pre_run_budget_estimate_record(
        budget,
        required_before_paid_execution=True,
    )

    assert record["required_before_paid_execution"] is True
    assert record["present"] is True
    assert record["valid"] is True
    assert record["target_model"] == "openai-direct/gpt-4.1-mini-2025-04-14"
    assert record["target_models"] == ["openai-direct/gpt-4.1-mini-2025-04-14"]
    assert record["estimated_total_usd"] == {"low": 1.0, "mid": 2.0, "high": 3.0}
    assert len(record["sha256"]) == 64
    assert record["errors"] == []


def test_pre_run_budget_estimate_record_binds_to_selected_config_models(tmp_path):
    module = load_module()
    budget = tmp_path / "budget.json"
    write_budget_estimate(
        budget,
        target_model="openai-direct/gpt-4.1-mini-2025-04-14",
    )

    record = module.build_pre_run_budget_estimate_record(
        budget,
        required_before_paid_execution=True,
        selected_config_model_bindings=[
            {
                "config": "config-taiwan-full-gpt-mini.yaml",
                "identifiers": [
                    "gpt-4.1-mini-2025-04-14",
                    "openai-direct/gpt-4.1-mini-2025-04-14",
                ],
            }
        ],
    )

    assert record["valid"] is True
    assert record["target_model_matches_selected_config"] is True
    assert record["selected_model_identifiers"] == [
        "gpt-4.1-mini-2025-04-14",
        "openai-direct/gpt-4.1-mini-2025-04-14",
    ]


def test_pre_run_budget_estimate_record_rejects_unmatched_selected_config(tmp_path):
    module = load_module()
    budget = tmp_path / "budget.json"
    write_budget_estimate(
        budget,
        target_model="openai-direct/expensive-unreviewed-model",
    )

    record = module.build_pre_run_budget_estimate_record(
        budget,
        required_before_paid_execution=True,
        selected_config_model_bindings=[
            {
                "config": "config-taiwan-full-gpt-mini.yaml",
                "identifiers": [
                    "openai-direct/gpt-4.1-mini-2025-04-14",
                ],
            }
        ],
    )

    assert record["valid"] is False
    assert record["target_model_matches_selected_config"] is False
    assert any("do not match selected config" in item for item in record["errors"])


def test_pre_run_budget_estimate_record_reports_missing_path():
    module = load_module()

    record = module.build_pre_run_budget_estimate_record(
        None,
        required_before_paid_execution=True,
    )

    assert record["present"] is False
    assert record["valid"] is False
    assert "required before paid execution" in record["errors"][0]


def test_pre_run_budget_estimate_record_rejects_incomplete_agentic_breakdown(tmp_path):
    module = load_module()
    budget = tmp_path / "budget.json"
    write_budget_estimate(budget)
    payload = json.loads(budget.read_text(encoding="utf-8"))
    payload["agentic_math"] = {
        "historical_records": 0,
        "estimate_usd": {"method": "no_local_evidence", "low": None, "mid": None, "high": None},
    }
    payload["agentic_swe_assorted"] = {
        "historical_records": 0,
        "estimate_usd": {"method": "no_local_evidence", "low": None, "mid": None, "high": None},
    }
    budget.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    record = module.build_pre_run_budget_estimate_record(
        budget,
        required_before_paid_execution=True,
    )

    assert record["valid"] is False
    assert any("agentic_math.historical_records must be positive" in item for item in record["errors"])
    assert any("agentic_swe_assorted.estimate_usd.low is missing" in item for item in record["errors"])


def test_external_action_approval_record_accepts_source_bound_verifier_report(tmp_path):
    module = load_module()
    report = tmp_path / "external_action_approval.verify.json"
    source_packet = write_external_action_approval_report(report)

    record = module.build_external_action_approval_record(
        report,
        required_before_external_action=True,
        expected_source_packet_path=source_packet,
    )

    assert record["required_before_external_action"] is True
    assert record["present"] is True
    assert record["valid"] is True
    assert record["status"] == "approved"
    assert record["required_approval_count"] == 1
    assert record["granted_approval_count"] == 1
    assert record["source_binding"]["bound"] is True
    assert record["source_packet_path_matches_expected"] is True
    assert record["source_packet_sha256_matches_expected"] is True
    assert record["expected_source_packet_json"] == str(source_packet)
    assert record["paid_api_approved_budget_usd"] == 25.0
    assert record["paid_api_minimum_approved_budget_usd"] == 20.0
    assert (
        record["paid_api_minimum_approved_budget_source"]
        == "max_pre_run_budget_estimate_high"
    )
    assert record["paid_api_approved_model_scope"] == "OpenAI mini canary"
    assert record["approval_results_validation"]["valid"] is True
    assert len(record["sha256"]) == 64
    assert record["errors"] == []


def test_external_action_approval_record_rejects_budget_below_minimum(tmp_path):
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
    assert record["paid_api_approved_budget_usd"] == 19.0
    assert record["paid_api_minimum_approved_budget_usd"] == 20.0
    assert record["approval_results_validation"]["valid"] is False
    assert any(
        "greater than or equal to minimum_approved_budget_usd" in error
        for error in record["errors"]
    )


def test_budget_approval_alignment_rejects_approval_below_high_estimate(tmp_path):
    module = load_module()
    budget = tmp_path / "budget.json"
    write_budget_estimate(
        budget,
        estimated_total_usd={"low": 1.0, "mid": 2.0, "high": 12.5},
    )
    approval = tmp_path / "external_action_approval.verify.json"
    source_packet = write_external_action_approval_report(
        approval,
        approved_budget_usd=10.0,
    )
    budget_record = module.build_pre_run_budget_estimate_record(
        budget,
        required_before_paid_execution=True,
    )
    approval_record = module.build_external_action_approval_record(
        approval,
        required_before_external_action=True,
        expected_source_packet_path=source_packet,
    )

    record = module.build_budget_approval_alignment_record(
        pre_run_budget_estimate=budget_record,
        external_action_approval=approval_record,
        required_before_paid_execution=True,
    )

    assert record["valid"] is False
    assert record["estimated_total_high_usd"] == 12.5
    assert record["approved_budget_usd"] == 10.0
    assert record["approved_budget_covers_estimate_high"] is False
    assert any("lower than pre_run_budget_estimate" in item for item in record["errors"])


def test_external_action_approval_record_rejects_unbound_report(tmp_path):
    module = load_module()
    report = tmp_path / "external_action_approval.verify.json"
    source_packet = write_external_action_approval_report(report)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["source_binding"]["bound"] = False
    payload["source_binding"]["errors"] = ["approval_template.source_approval_packet_sha256 mismatch"]
    report.write_text(json.dumps(payload), encoding="utf-8")

    record = module.build_external_action_approval_record(
        report,
        required_before_external_action=True,
        expected_source_packet_path=source_packet,
    )

    assert record["valid"] is False
    assert "source_binding.bound must be true" in record["errors"]
    assert any("source_approval_packet_sha256 mismatch" in item for item in record["errors"])


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

    record = module.build_external_action_approval_record(
        report,
        required_before_external_action=True,
        expected_source_packet_path=other_source_packet,
    )

    assert source_packet != other_source_packet
    assert record["valid"] is False
    assert record["source_packet_path_matches_expected"] is False
    assert record["source_packet_sha256_matches_expected"] is False
    assert any("source_packet_json does not match" in item for item in record["errors"])
    assert any("source_approval_packet_sha256 does not match" in item for item in record["errors"])


def test_external_action_approval_record_reports_missing_path_when_required():
    module = load_module()

    record = module.build_external_action_approval_record(
        None,
        required_before_external_action=True,
    )

    assert record["present"] is False
    assert record["valid"] is False
    assert "required before external execution" in record["errors"][0]


def test_prepare_only_review_records_completion_requirements(tmp_path, monkeypatch):
    module = load_module()
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        "\n".join(
            [
                "models:",
                "  - slug: gpt-4_1-mini-openai-direct-canary",
                "    source_config: config-gpt-4.1-mini-2025-04-14.yaml",
                "    run_name: 'taiwan/full/openai/gpt-4.1-mini: canary'",
                "    openclaw_model: 'openai-direct/gpt-4.1-mini-2025-04-14'",
                "    agentic_thinking: 'off'",
                "    swe_thinking: 'off'",
                "    canary: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "outputs"
    generated_dir = tmp_path / "generated"
    budget = output_root / "openai_canary_budget_estimate.json"
    write_budget_estimate(
        budget,
        target_model="openai-direct/gpt-4.1-mini-2025-04-14",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_taiwan_full_eval_batch.py",
            "--manifest",
            str(manifest),
            "--canary",
            "--prepare-only",
            "--phase",
            "agentic",
            "--generated-config-dir",
            str(generated_dir),
            "--output-root",
            str(output_root),
            "--agentic-math-nemoclaw-sandbox",
            "nejumi-taiwan",
            "--agentic-math-nemoclaw-openclaw-config-path",
            "/sandbox/.openclaw/openclaw.json",
            "--agentic-swe-assorted-nemoclaw-sandbox",
            "nejumi-taiwan",
            "--agentic-swe-assorted-nemoclaw-openclaw-config-path",
            "/sandbox/.openclaw/openclaw.json",
            "--agentic-swe-assorted-nemoclaw-checkout-transfer-mode",
            "copy",
            "--require-nemoclaw-agentic-config",
            "--wandb-run-id-prefix",
            "twcanary-test",
            "--verify-wandb-completion",
            "--verify-weave-agents",
            "--weave-agents-require-tool-span",
            "--weave-agents-require-tool-content",
            "--run-purpose",
            "prepare-only schema test",
            "--expected-cost-band",
            "prepare-only",
            "--pre-run-budget-estimate-json",
            str(budget),
        ],
    )

    module.main()

    review = json.loads(
        (output_root / "canary_agentic_paid_run_review.json").read_text(
            encoding="utf-8"
        )
    )
    requirements = review["completion_requirements"]
    assert review["status"] == "prepared"
    assert requirements["status"] == "completed"
    assert requirements["actual_cost_estimate"] == "required after execution"
    assert requirements["provider_bill_reference"] == "required after execution"
    assert requirements["pre_run_budget_estimate"]["required_before_paid_execution"] is True
    assert requirements["external_action_approval"]["required_before_external_action"] is False
    assert (
        "source_binding.bound=true"
        in requirements["external_action_approval"]["required_fields"]
    )
    assert (
        "wandb_entity and wandb_project for successful W&B-verified runs"
        in requirements["runs"]["required_fields"]
    )
    assert "preflight_json" in requirements["runs"]["required_fields"]
    assert "preflight_returncode" in requirements["runs"]["required_fields"]
    assert "preflight_ok" in requirements["runs"]["required_fields"]
    assert requirements["run_eval_preflight"]["required"] is True
    assert requirements["run_eval_preflight"]["required_before_run_eval"] is True
    assert requirements["run_eval_preflight"]["expected_record_count"] == 1
    assert (
        "will_initialize_wandb=false"
        in requirements["run_eval_preflight"]["required_fields"]
    )
    assert (
        "will_start_inference_engine=false"
        in requirements["run_eval_preflight"]["required_fields"]
    )
    assert (
        "will_run_evaluators=false"
        in requirements["run_eval_preflight"]["required_fields"]
    )
    assert requirements["wandb_completion"]["required"] is True
    assert requirements["wandb_completion"]["benchmarks"] == [
        "agentic_math",
        "agentic_swe",
    ]
    assert requirements["wandb_completion"]["schema_version"] == 1
    assert requirements["wandb_completion"]["observed_evidence_required"] is True
    assert (
        requirements["wandb_completion"][
            "nemoclaw_session_audit_required_for_agentic_benchmarks"
        ]
        is True
    )
    assert requirements["weave_agents_completion"]["required"] is True
    assert requirements["weave_agents_completion"]["content_required"] is True
    assert requirements["weave_agents_completion"]["tool_span_required"] is True
    assert requirements["weave_agents_completion"]["tool_content_required"] is True
    assert requirements["weave_agents_completion"]["run_scope_required"] is True
    assert review["pre_run_budget_estimate"]["valid"] is True
    assert review["pre_run_budget_estimate"]["path"] == str(budget)
    assert len(review["pre_run_budget_estimate"]["sha256"]) == 64
    assert (
        review["pre_run_budget_estimate"]["target_model"]
        == "openai-direct/gpt-4.1-mini-2025-04-14"
    )
    assert review["pre_run_budget_estimate"]["estimated_total_usd"] == {
        "low": 1.0,
        "mid": 2.0,
        "high": 3.0,
    }
    assert review["external_action_approval"]["required_before_external_action"] is False
    assert review["external_action_approval"]["present"] is False
    assert "canary_usage_cost_summary.csv" in review["post_run_cost_command"]
    assert "glm52" not in review["post_run_cost_command"]
    guard = review["nemoclaw_agentic_config_guard"]
    assert guard["required"] is True
    assert guard["enforced"] is True
    assert guard["ok"] is True
    assert guard["records"][0]["agentic_math_nemoclaw_sandbox"] == "nejumi-taiwan"
    assert (
        guard["records"][0]["agentic_math_nemoclaw_openclaw_config_path"]
        == "/sandbox/.openclaw/openclaw.json"
    )
    assert guard["records"][0]["agentic_math_use_task_agent"] is True
    assert guard["records"][0]["agentic_math_no_local"] is True
    assert "web_search" in guard["records"][0]["agentic_math_deny_tool"]
    assert "https?://" in guard["records"][0]["agentic_math_deny_argument_pattern"]
    assert guard["records"][0]["agentic_math_local_exec_allowed"] is True
    assert guard["records"][0]["agentic_math_local_exec_blocking_patterns"] == []
    assert guard["records"][0]["agentic_swe_assorted_nemoclaw_sandbox"] == "nejumi-taiwan"
    assert (
        guard["records"][0]["agentic_swe_assorted_nemoclaw_openclaw_config_path"]
        == "/sandbox/.openclaw/openclaw.json"
    )
    assert guard["records"][0]["agentic_swe_assorted_nemoclaw_checkout_transfer_mode"] == "copy"
    assert guard["records"][0]["agentic_swe_assorted_no_local"] is True
    assert "web_search" in guard["records"][0]["agentic_swe_assorted_deny_tool"]
    assert "https?://" in guard["records"][0]["agentic_swe_assorted_deny_argument_pattern"]
    assert guard["records"][0]["agentic_swe_assorted_local_exec_allowed"] is True
    assert guard["records"][0]["agentic_swe_assorted_local_exec_blocking_patterns"] == []
    preflights = review["run_eval_preflights"]
    assert len(preflights) == 1
    assert preflights[0]["required_before_run_eval"] is True
    assert preflights[0]["executed"] is True
    assert preflights[0]["returncode"] == 0
    assert preflights[0]["ok"] is True
    assert preflights[0]["status"] == "passed"
    assert preflights[0]["phase_validation"]["ok"] is True
    assert preflights[0]["output_json"].endswith(
        "run_eval_preflight/agentic-gpt-4_1-mini-openai-direct-canary.json"
    )
    assert "--preflight" in preflights[0]["command"]
    assert "--preflight-json" in preflights[0]["command"]


def test_run_eval_preflight_rejects_stale_success_json(tmp_path, monkeypatch):
    module = load_module()
    output_json = tmp_path / "preflight.json"
    output_json.write_text(
        json.dumps({"ok": True, "status": "passed"}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "stream_run", lambda *_args, **_kwargs: 0)

    result = module.execute_run_eval_preflight(
        command=["python", "scripts/run_eval.py", "--preflight"],
        output_json=output_json,
        log_path=tmp_path / "preflight.log",
        env={},
        phase="agentic",
    )

    assert result["ok"] is False
    assert result["returncode"] == 0
    assert "does not exist" in result["status"]
    assert output_json.exists() is False


def test_prepare_only_can_require_nemoclaw_agentic_config(tmp_path, monkeypatch):
    module = load_module()
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        "\n".join(
            [
                "models:",
                "  - slug: gpt-4_1-mini-openai-direct-canary",
                "    source_config: config-gpt-4.1-mini-2025-04-14.yaml",
                "    run_name: 'taiwan/full/openai/gpt-4.1-mini: canary'",
                "    openclaw_model: 'openai-direct/gpt-4.1-mini-2025-04-14'",
                "    agentic_thinking: 'off'",
                "    swe_thinking: 'off'",
                "    nemoclaw_sandbox: 'None'",
                "    canary: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "outputs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_taiwan_full_eval_batch.py",
            "--manifest",
            str(manifest),
            "--canary",
            "--prepare-only",
            "--phase",
            "agentic",
            "--generated-config-dir",
            str(tmp_path / "generated"),
            "--output-root",
            str(output_root),
            "--require-nemoclaw-agentic-config",
            "--run-purpose",
            "prepare-only negative test",
            "--expected-cost-band",
            "prepare-only",
        ],
    )

    try:
        module.main()
    except SystemExit as exc:
        assert "NeMoClaw agentic config requirement failed" in str(exc)
    else:  # pragma: no cover - regression guard
        raise AssertionError("expected NeMoClaw agentic config guard to fail")

    review = json.loads(
        (output_root / "canary_agentic_paid_run_review.json").read_text(
            encoding="utf-8"
        )
    )
    assert review["status"] == "nemoclaw_agentic_config_failed"
    guard = review["blocking_reason"]
    assert guard["required"] is True
    assert guard["enforced"] is True
    assert guard["ok"] is False
    assert (
        "agentic_math.nemoclaw_sandbox must name an isolated sandbox"
        in guard["errors"][0]
    )


def test_nemoclaw_agentic_config_guard_requires_remote_lookup_deny_policy(tmp_path):
    module = load_module()
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "run:",
                "  agentic_math: true",
                "  agentic_swe_assorted: true",
                "agentic_math:",
                "  nemoclaw_sandbox: nejumi-taiwan",
                "  use_task_agent: true",
                "  deny_tool:",
                "    - code_execution",
                "  deny_argument_pattern:",
                r"    - \b(curl|wget)\b",
                "agentic_swe_assorted:",
                "  nemoclaw_sandbox: nejumi-taiwan",
                "  nemoclaw_checkout_transfer_mode: copy",
                "  deny_tool:",
                "    - code_execution",
                "    - web_search",
                "    - web_fetch",
                "    - browser",
                "    - browser_*",
                "  deny_argument_pattern:",
                "    - https?://",
                r"    - \b(curl|wget)\b",
                r"    - \b(requests|urllib|httpx)\.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    guard = module.build_nemoclaw_agentic_config_guard(
        [config],
        phase="agentic",
        required=True,
    )

    assert guard["ok"] is False
    assert any(
        "agentic_math.deny_tool missing required values" in error
        for error in guard["errors"]
    )
    assert any(
        "agentic_math.deny_argument_pattern missing required values" in error
        for error in guard["errors"]
    )


def test_nemoclaw_agentic_config_guard_rejects_local_exec_denied(tmp_path):
    module = load_module()
    config = tmp_path / "bad_exec.yaml"
    config.write_text(
        "\n".join(
            [
                "run:",
                "  agentic_math: true",
                "  agentic_swe_assorted: true",
                "agentic_math:",
                "  nemoclaw_sandbox: nejumi-taiwan",
                "  use_task_agent: true",
                "  deny_tool:",
                "    - code_execution",
                "    - exec",
                "    - web_search",
                "    - web_fetch",
                "    - browser",
                "    - browser_*",
                "  deny_argument_pattern:",
                "    - https?://",
                r"    - \b(curl|wget)\b",
                r"    - \b(requests|urllib|httpx)\.",
                "agentic_swe_assorted:",
                "  nemoclaw_sandbox: nejumi-taiwan",
                "  nemoclaw_checkout_transfer_mode: copy",
                "  deny_tool:",
                "    - code_execution",
                "    - '*exec*'",
                "    - web_search",
                "    - web_fetch",
                "    - browser",
                "    - browser_*",
                "  deny_argument_pattern:",
                "    - https?://",
                r"    - \b(curl|wget)\b",
                r"    - \b(requests|urllib|httpx)\.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    guard = module.build_nemoclaw_agentic_config_guard(
        [config],
        phase="agentic",
        required=True,
    )

    assert guard["ok"] is False
    assert any(
        "agentic_math.deny_tool must not block local OpenClaw exec tool: exec" in error
        for error in guard["errors"]
    )
    assert any(
        "agentic_swe_assorted.deny_tool must not block local OpenClaw exec tool: *exec*" in error
        for error in guard["errors"]
    )
    assert guard["records"][0]["agentic_math_local_exec_allowed"] is False
    assert guard["records"][0]["agentic_math_local_exec_blocking_patterns"] == ["exec"]
    assert guard["records"][0]["agentic_swe_assorted_local_exec_allowed"] is False
    assert guard["records"][0]["agentic_swe_assorted_local_exec_blocking_patterns"] == ["*exec*"]


def test_nemoclaw_agentic_config_guard_rejects_weave_sidecar_for_production(tmp_path):
    module = load_module()
    config = tmp_path / "sidecar.yaml"
    config.write_text(
        "\n".join(
            [
                "run:",
                "  agentic_math: true",
                "  agentic_swe_assorted: true",
                "agentic_math:",
                "  nemoclaw_sandbox: nejumi-taiwan",
                "  nemoclaw_openclaw_config_path: /sandbox/.openclaw/openclaw.json",
                "  use_task_agent: true",
                "  no_local: true",
                "  weave_sidecar: true",
                "  deny_tool:",
                "    - code_execution",
                "    - web_search",
                "    - web_fetch",
                "    - browser",
                "    - browser_*",
                "  deny_argument_pattern:",
                "    - https?://",
                r"    - \b(curl|wget)\b",
                r"    - \b(requests|urllib|httpx)\.",
                "agentic_swe_assorted:",
                "  nemoclaw_sandbox: nejumi-taiwan",
                "  nemoclaw_openclaw_config_path: /sandbox/.openclaw/openclaw.json",
                "  nemoclaw_checkout_transfer_mode: copy",
                "  no_local: true",
                "  weave_sidecar_strict: true",
                "  deny_tool:",
                "    - code_execution",
                "    - web_search",
                "    - web_fetch",
                "    - browser",
                "    - browser_*",
                "  deny_argument_pattern:",
                "    - https?://",
                r"    - \b(curl|wget)\b",
                r"    - \b(requests|urllib|httpx)\.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    guard = module.build_nemoclaw_agentic_config_guard(
        [config],
        phase="agentic",
        required=True,
    )

    assert guard["ok"] is False
    assert any(
        "agentic_math must use native weave-openclaw tracing only" in error
        for error in guard["errors"]
    )
    assert any(
        "agentic_swe_assorted must use native weave-openclaw tracing only" in error
        for error in guard["errors"]
    )
    record = guard["records"][0]
    assert record["agentic_math_weave_sidecar"] is True
    assert record["agentic_swe_assorted_weave_sidecar_strict"] is True


def test_agentic_production_evidence_guard_requires_wandb_weave_and_nemoclaw(tmp_path):
    module = load_module()
    args = SimpleNamespace(
        require_nemoclaw_agentic_config=False,
        agentic_math_nemoclaw_openclaw_config_path=None,
        agentic_swe_assorted_nemoclaw_openclaw_config_path=None,
        require_weave_content_canary=False,
        weave_content_canary_gate=None,
        verify_wandb_completion=False,
        verify_weave_agents=False,
        wandb_run_id_prefix="",
        weave_agents_no_require_content=False,
        weave_agents_require_tool_span=False,
        weave_agents_require_tool_content=False,
        weave_agents_require_usage=False,
    )

    blocked = module.build_agentic_production_evidence_guard(
        args,
        phase="agentic",
        will_call_paid_model_api=True,
    )
    prepare_only = module.build_agentic_production_evidence_guard(
        args,
        phase="agentic",
        will_call_paid_model_api=False,
    )
    nonagentic = module.build_agentic_production_evidence_guard(
        args,
        phase="nonagentic",
        will_call_paid_model_api=True,
    )

    assert blocked["enforced"] is True
    assert blocked["ok"] is False
    assert "--verify-wandb-completion" in blocked["missing_flags"]
    assert "--verify-weave-agents" in blocked["missing_flags"]
    assert "--require-nemoclaw-agentic-config" in blocked["missing_flags"]
    assert "--agentic-math-nemoclaw-openclaw-config-path" in blocked["missing_flags"]
    assert "--agentic-swe-assorted-nemoclaw-openclaw-config-path" in blocked["missing_flags"]
    assert "--weave-content-canary-gate" in blocked["missing_flags"]
    assert prepare_only["enforced"] is False
    assert prepare_only["ok"] is True
    assert nonagentic["enforced"] is False
    assert nonagentic["ok"] is True


def test_agentic_production_evidence_guard_rejects_wrong_nemoclaw_openclaw_config_path(
    tmp_path,
):
    module = load_module()
    args = SimpleNamespace(
        require_nemoclaw_agentic_config=True,
        agentic_math_nemoclaw_openclaw_config_path="/tmp/openclaw.json",
        agentic_swe_assorted_nemoclaw_openclaw_config_path="/sandbox/.openclaw/openclaw.json",
        require_weave_content_canary=True,
        weave_content_canary_gate=tmp_path / "gate.json",
        verify_wandb_completion=True,
        verify_weave_agents=True,
        wandb_run_id_prefix="twcanary-test",
        weave_agents_no_require_content=False,
        weave_agents_require_tool_span=True,
        weave_agents_require_tool_content=True,
        weave_agents_require_usage=True,
    )

    guard = module.build_agentic_production_evidence_guard(
        args,
        phase="agentic",
        will_call_paid_model_api=True,
    )

    assert guard["ok"] is False
    assert guard["missing_flags"] == []
    assert guard["invalid_values"] == {
        "--agentic-math-nemoclaw-openclaw-config-path": "/tmp/openclaw.json"
    }
    assert any(
        "--agentic-math-nemoclaw-openclaw-config-path must be "
        "/sandbox/.openclaw/openclaw.json"
        in error
        for error in guard["errors"]
    )


def test_wandb_resume_policy_requires_explicit_matching_config(tmp_path):
    module = load_module()
    args = SimpleNamespace(
        allow_wandb_resume=True,
        wandb_resume_config_json=None,
        wandb_run_id_prefix="twcanary-test",
        phase="agentic",
        infrastructure_resume_attempts=0,
        infrastructure_resume_base_seconds=30,
    )

    missing = module.build_wandb_resume_policy_record(args)

    assert missing["valid"] is False
    assert "--allow-wandb-resume requires --wandb-resume-config-json" in missing["errors"]

    resume_config = tmp_path / "resume.json"
    resume_config.write_text(
        json.dumps(
            {
                "allow_wandb_resume": True,
                "wandb_run_id_prefix": "twcanary-test",
                "phase": "agentic",
                "explicit_user_instruction": True,
                "purpose": "operator requested append-only resume for a reviewed run",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args.wandb_resume_config_json = resume_config

    valid = module.build_wandb_resume_policy_record(args)

    assert valid["valid"] is True
    assert valid["status"] == "valid"


def test_wandb_resume_policy_binds_infrastructure_attempts(tmp_path):
    module = load_module()
    resume_config = tmp_path / "resume.json"
    resume_config.write_text(
        json.dumps(
            {
                "allow_wandb_resume": True,
                "wandb_run_id_prefix": "twfull-test",
                "phase": "full",
                "explicit_user_instruction": True,
                "purpose": "recover reviewed transient infrastructure failures",
                "infrastructure_recovery_attempts": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        allow_wandb_resume=True,
        wandb_resume_config_json=resume_config,
        wandb_run_id_prefix="twfull-test",
        phase="full",
        infrastructure_resume_attempts=2,
        infrastructure_resume_base_seconds=30,
    )

    valid = module.build_wandb_resume_policy_record(args)
    args.infrastructure_resume_attempts = 1
    invalid = module.build_wandb_resume_policy_record(args)

    assert valid["valid"] is True
    assert invalid["valid"] is False
    assert any(
        "infrastructure_recovery_attempts must match" in error
        for error in invalid["errors"]
    )

    args.allow_wandb_resume = False
    without_resume = module.build_wandb_resume_policy_record(args)
    assert without_resume["valid"] is False
    assert any(
        "requires --allow-wandb-resume" in error
        for error in without_resume["errors"]
    )


def test_process_recovery_resumes_only_checkpointed_infrastructure_failure(
    tmp_path,
    monkeypatch,
):
    module = load_module()
    checkpoint_root = tmp_path / "benchmark_checkpoints"
    checkpoint_root.mkdir()
    responses = [9, 0]
    sleeps = []

    def fake_stream_run(command, log_path, env):
        returncode = responses.pop(0)
        if returncode:
            (checkpoint_root / "bfcl.json").write_text(
                json.dumps(
                    {
                        "run_id": "run-1",
                        "benchmark": "bfcl",
                        "status": "failed",
                        "updated_at": module.time.time(),
                        "error_type": "BFCLInfrastructureError",
                        "error": "provider read timeout",
                    }
                ),
                encoding="utf-8",
            )
        return returncode

    monkeypatch.setattr(module, "stream_run", fake_stream_run)
    monkeypatch.setattr(module.time, "sleep", sleeps.append)

    returncode, attempts = module.run_eval_with_infrastructure_recovery(
        ["python", "scripts/run_eval.py"],
        log_path=tmp_path / "run.log",
        env={},
        checkpoint_root=checkpoint_root,
        wandb_run_id="run-1",
        max_resume_attempts=2,
        base_delay_seconds=3,
    )

    assert returncode == 0
    assert [attempt["action"] for attempt in attempts] == [
        "resume_after_cooldown",
        "completed",
    ]
    assert sleeps == [3.0]


def test_process_recovery_stops_on_model_timeout(tmp_path, monkeypatch):
    module = load_module()
    checkpoint_root = tmp_path / "benchmark_checkpoints"
    checkpoint_root.mkdir()
    (checkpoint_root / "agentic_math.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "benchmark": "agentic_math",
                "status": "failed",
                "updated_at": module.time.time(),
                "error_type": "TimeoutError",
                "error": "task exceeded benchmark wall timeout",
            }
        ),
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(
        module,
        "stream_run",
        lambda command, log_path, env: calls.append(command) or 9,
    )

    returncode, attempts = module.run_eval_with_infrastructure_recovery(
        ["python", "scripts/run_eval.py"],
        log_path=tmp_path / "run.log",
        env={},
        checkpoint_root=checkpoint_root,
        wandb_run_id="run-1",
        max_resume_attempts=2,
        base_delay_seconds=0,
    )

    assert returncode == 9
    assert len(calls) == 1
    assert attempts[0]["action"] == "stopped_non_infrastructure_failure"


def test_process_recovery_ignores_stale_failed_checkpoint(tmp_path, monkeypatch):
    module = load_module()
    checkpoint_root = tmp_path / "benchmark_checkpoints"
    checkpoint_root.mkdir()
    (checkpoint_root / "bfcl.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "benchmark": "bfcl",
                "status": "failed",
                "updated_at": 1,
                "error_type": "BFCLInfrastructureError",
                "error": "old provider timeout",
            }
        ),
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(
        module,
        "stream_run",
        lambda command, log_path, env: calls.append(command) or 9,
    )

    returncode, attempts = module.run_eval_with_infrastructure_recovery(
        ["python", "scripts/run_eval.py"],
        log_path=tmp_path / "run.log",
        env={},
        checkpoint_root=checkpoint_root,
        wandb_run_id="run-1",
        max_resume_attempts=2,
        base_delay_seconds=0,
    )

    assert returncode == 9
    assert len(calls) == 1
    assert attempts[0]["action"] == "stopped_no_failed_checkpoint"


def test_paid_run_executes_run_eval_preflight_before_run_eval(tmp_path, monkeypatch):
    module = load_module()
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        "\n".join(
            [
                "models:",
                "  - slug: gpt-4_1-mini-openai-direct-canary",
                "    source_config: config-gpt-4.1-mini-2025-04-14.yaml",
                "    run_name: 'taiwan/full/openai/gpt-4.1-mini: canary'",
                "    openclaw_model: 'openai-direct/gpt-4.1-mini-2025-04-14'",
                "    agentic_thinking: 'off'",
                "    swe_thinking: 'off'",
                "    canary: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "outputs"
    budget = output_root / "openai_canary_budget_estimate.json"
    write_budget_estimate(
        budget,
        target_model="openai-direct/gpt-4.1-mini-2025-04-14",
    )
    approval = output_root / "external_action_approval.verify.json"
    source_packet = write_external_action_approval_report(approval)
    gate = write_passing_weave_content_canary_gate(
        output_root / "content_canary.gate.json"
    )
    calls: list[list[str]] = []
    envs: list[dict[str, str]] = []

    def fake_stream_run(command, log_path, env):
        calls.append(command)
        envs.append(dict(env))
        if "--preflight" in command:
            output_path = Path(command[command.index("--preflight-json") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "ok": True,
                        "status": "passed",
                        "enabled_benchmarks": ["agentic_math", "agentic_swe_assorted"],
                        "scheduled_evaluators": ["agentic_math", "agentic_swe_assorted"],
                        "will_initialize_wandb": False,
                        "will_log_wandb_artifacts": False,
                        "will_initialize_weave": False,
                        "will_start_inference_engine": False,
                        "will_run_evaluators": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return 0
        return 7

    monkeypatch.setattr(module, "stream_run", fake_stream_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_taiwan_full_eval_batch.py",
            "--manifest",
            str(manifest),
            "--canary",
            "--phase",
            "agentic",
            "--generated-config-dir",
            str(tmp_path / "generated"),
            "--output-root",
            str(output_root),
            "--agentic-math-nemoclaw-sandbox",
            "nejumi-taiwan",
            "--agentic-math-nemoclaw-openclaw-config-path",
            "/sandbox/.openclaw/openclaw.json",
            "--agentic-swe-assorted-nemoclaw-sandbox",
            "nejumi-taiwan",
            "--agentic-swe-assorted-nemoclaw-openclaw-config-path",
            "/sandbox/.openclaw/openclaw.json",
            "--agentic-swe-assorted-nemoclaw-checkout-transfer-mode",
            "copy",
            "--require-nemoclaw-agentic-config",
            "--wandb-run-id-prefix",
            "twcanary-test",
            "--verify-wandb-completion",
            "--verify-weave-agents",
            "--weave-agents-require-tool-span",
            "--weave-agents-require-tool-content",
            "--weave-agents-require-usage",
            "--weave-content-canary-gate",
            str(gate),
            "--require-weave-content-canary",
            "--run-purpose",
            "paid preflight order test",
            "--expected-cost-band",
            "$1-$3",
            "--pre-run-budget-estimate-json",
            str(budget),
            "--external-action-approval-report-json",
            str(approval),
            "--external-action-approval-source-packet-json",
            str(source_packet),
        ],
    )

    try:
        module.main()
    except SystemExit as exc:
        assert exc.code == 7
    else:  # pragma: no cover - regression guard
        raise AssertionError("expected mocked run_eval failure")

    assert len(calls) == 2
    assert "--preflight" in calls[0]
    assert "--preflight" not in calls[1]
    assert envs[0]["WANDB_RESUME"] == "never"
    assert envs[1]["WANDB_RESUME"] == "never"
    assert "NEJUMI_ALLOW_WANDB_RESUME" not in envs[0]
    assert "NEJUMI_ALLOW_WANDB_RESUME" not in envs[1]
    review = json.loads(
        (output_root / "canary_agentic_paid_run_review.json").read_text(
            encoding="utf-8"
        )
    )
    assert review["status"] == "failed"
    run = review["runs"][0]
    assert run["preflight_ok"] is True
    assert run["preflight_returncode"] == 0
    assert run["preflight_status"] == "passed"
    assert run["preflight_phase_validation"]["ok"] is True
    assert run["returncode"] == 7


def test_paid_run_stops_before_run_eval_when_preflight_fails(tmp_path, monkeypatch):
    module = load_module()
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        "\n".join(
            [
                "models:",
                "  - slug: gpt-4_1-mini-openai-direct-canary",
                "    source_config: config-gpt-4.1-mini-2025-04-14.yaml",
                "    run_name: 'taiwan/full/openai/gpt-4.1-mini: canary'",
                "    openclaw_model: 'openai-direct/gpt-4.1-mini-2025-04-14'",
                "    agentic_thinking: 'off'",
                "    swe_thinking: 'off'",
                "    canary: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "outputs"
    budget = output_root / "openai_canary_budget_estimate.json"
    write_budget_estimate(
        budget,
        target_model="openai-direct/gpt-4.1-mini-2025-04-14",
    )
    approval = output_root / "external_action_approval.verify.json"
    source_packet = write_external_action_approval_report(approval)
    gate = write_passing_weave_content_canary_gate(
        output_root / "content_canary.gate.json"
    )
    calls: list[list[str]] = []

    def fake_stream_run(command, log_path, env):
        calls.append(command)
        assert "--preflight" in command
        output_path = Path(command[command.index("--preflight-json") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "ok": False,
                    "status": "failed",
                    "enabled_benchmarks": [],
                    "will_initialize_wandb": False,
                    "will_log_wandb_artifacts": False,
                    "will_initialize_weave": False,
                    "will_start_inference_engine": False,
                    "will_run_evaluators": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return 2

    monkeypatch.setattr(module, "stream_run", fake_stream_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_taiwan_full_eval_batch.py",
            "--manifest",
            str(manifest),
            "--canary",
            "--phase",
            "agentic",
            "--generated-config-dir",
            str(tmp_path / "generated"),
            "--output-root",
            str(output_root),
            "--agentic-math-nemoclaw-sandbox",
            "nejumi-taiwan",
            "--agentic-math-nemoclaw-openclaw-config-path",
            "/sandbox/.openclaw/openclaw.json",
            "--agentic-swe-assorted-nemoclaw-sandbox",
            "nejumi-taiwan",
            "--agentic-swe-assorted-nemoclaw-openclaw-config-path",
            "/sandbox/.openclaw/openclaw.json",
            "--agentic-swe-assorted-nemoclaw-checkout-transfer-mode",
            "copy",
            "--require-nemoclaw-agentic-config",
            "--wandb-run-id-prefix",
            "twcanary-test",
            "--verify-wandb-completion",
            "--verify-weave-agents",
            "--weave-agents-require-tool-span",
            "--weave-agents-require-tool-content",
            "--weave-agents-require-usage",
            "--weave-content-canary-gate",
            str(gate),
            "--require-weave-content-canary",
            "--run-purpose",
            "paid preflight failure test",
            "--expected-cost-band",
            "$1-$3",
            "--pre-run-budget-estimate-json",
            str(budget),
            "--external-action-approval-report-json",
            str(approval),
            "--external-action-approval-source-packet-json",
            str(source_packet),
        ],
    )

    try:
        module.main()
    except SystemExit as exc:
        assert "run_eval preflight failed" in str(exc)
    else:  # pragma: no cover - regression guard
        raise AssertionError("expected preflight failure")

    assert len(calls) == 1
    assert "--preflight" in calls[0]
    review = json.loads(
        (output_root / "canary_agentic_paid_run_review.json").read_text(
            encoding="utf-8"
        )
    )
    assert review["status"] == "run_eval_preflight_failed"
    run = review["runs"][0]
    assert run["preflight_ok"] is False
    assert run["preflight_returncode"] == 2
    assert run["preflight_status"] == "failed"
    assert run["returncode"] == 2


def test_paid_run_requires_pre_run_budget_estimate(tmp_path, monkeypatch):
    module = load_module()
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        "\n".join(
            [
                "models:",
                "  - slug: gpt-4_1-mini-openai-direct-canary",
                "    source_config: config-gpt-4.1-mini-2025-04-14.yaml",
                "    run_name: 'taiwan/full/openai/gpt-4.1-mini: canary'",
                "    openclaw_model: 'openai-direct/gpt-4.1-mini-2025-04-14'",
                "    agentic_thinking: 'off'",
                "    swe_thinking: 'off'",
                "    canary: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "outputs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_taiwan_full_eval_batch.py",
            "--manifest",
            str(manifest),
            "--canary",
            "--phase",
            "nonagentic",
            "--generated-config-dir",
            str(tmp_path / "generated"),
            "--output-root",
            str(output_root),
            "--run-purpose",
            "paid schema test",
            "--expected-cost-band",
            "$1-$3",
        ],
    )

    try:
        module.main()
    except SystemExit as exc:
        assert "--pre-run-budget-estimate-json" in str(exc)
    else:
        raise AssertionError("expected missing budget estimate to block paid execution")

    review = json.loads(
        (output_root / "canary_nonagentic_paid_run_review.json").read_text(
            encoding="utf-8"
        )
    )
    assert review["status"] == "accountability_fields_missing"
    assert "--pre-run-budget-estimate-json" in review["missing_fields"]


def test_paid_run_rejects_budget_for_different_model(tmp_path, monkeypatch):
    module = load_module()
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        "\n".join(
            [
                "models:",
                "  - slug: gpt-4_1-mini-openai-direct-canary",
                "    source_config: config-gpt-4.1-mini-2025-04-14.yaml",
                "    run_name: 'taiwan/full/openai/gpt-4.1-mini: canary'",
                "    openclaw_model: 'openai-direct/gpt-4.1-mini-2025-04-14'",
                "    agentic_thinking: 'off'",
                "    swe_thinking: 'off'",
                "    canary: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "outputs"
    budget = output_root / "wrong_model_budget_estimate.json"
    write_budget_estimate(
        budget,
        target_model="openai-direct/expensive-unreviewed-model",
    )
    approval = output_root / "external_action_approval.verify.json"
    write_external_action_approval_report(approval)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_taiwan_full_eval_batch.py",
            "--manifest",
            str(manifest),
            "--canary",
            "--phase",
            "nonagentic",
            "--generated-config-dir",
            str(tmp_path / "generated"),
            "--output-root",
            str(output_root),
            "--run-purpose",
            "paid schema test",
            "--expected-cost-band",
            "$1-$3",
            "--pre-run-budget-estimate-json",
            str(budget),
            "--external-action-approval-report-json",
            str(approval),
        ],
    )

    try:
        module.main()
    except SystemExit as exc:
        assert "--pre-run-budget-estimate-json" in str(exc)
    else:
        raise AssertionError("expected mismatched budget estimate to block paid execution")

    review = json.loads(
        (output_root / "canary_nonagentic_paid_run_review.json").read_text(
            encoding="utf-8"
        )
    )
    assert review["status"] == "accountability_fields_missing"
    assert "--pre-run-budget-estimate-json" in review["missing_fields"]
    budget_record = review["pre_run_budget_estimate"]
    assert budget_record["valid"] is False
    assert budget_record["target_model_matches_selected_config"] is False
    assert any("do not match selected config" in item for item in budget_record["errors"])


def test_paid_run_rejects_approval_budget_below_pre_run_high(tmp_path, monkeypatch):
    module = load_module()
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        "\n".join(
            [
                "models:",
                "  - slug: gpt-4_1-mini-openai-direct-canary",
                "    source_config: config-gpt-4.1-mini-2025-04-14.yaml",
                "    run_name: 'taiwan/full/openai/gpt-4.1-mini: canary'",
                "    openclaw_model: 'openai-direct/gpt-4.1-mini-2025-04-14'",
                "    agentic_thinking: 'off'",
                "    swe_thinking: 'off'",
                "    canary: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "outputs"
    budget = output_root / "openai_canary_budget_estimate.json"
    write_budget_estimate(
        budget,
        estimated_total_usd={"low": 1.0, "mid": 2.0, "high": 3.0},
    )
    approval = output_root / "external_action_approval.verify.json"
    source_packet = write_external_action_approval_report(
        approval,
        approved_budget_usd=2.5,
        minimum_approved_budget_usd=1.0,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_taiwan_full_eval_batch.py",
            "--manifest",
            str(manifest),
            "--canary",
            "--phase",
            "nonagentic",
            "--generated-config-dir",
            str(tmp_path / "generated"),
            "--output-root",
            str(output_root),
            "--run-purpose",
            "paid schema test",
            "--expected-cost-band",
            "$1-$3",
            "--pre-run-budget-estimate-json",
            str(budget),
            "--external-action-approval-report-json",
            str(approval),
            "--external-action-approval-source-packet-json",
            str(source_packet),
        ],
    )

    try:
        module.main()
    except SystemExit as exc:
        assert "approved budget" in str(exc)
    else:
        raise AssertionError("expected approval budget cap to block paid execution")

    review = json.loads(
        (output_root / "canary_nonagentic_paid_run_review.json").read_text(
            encoding="utf-8"
        )
    )
    assert review["status"] == "budget_approval_alignment_failed"
    alignment = review["budget_approval_alignment"]
    assert alignment["estimated_total_high_usd"] == 3.0
    assert alignment["approved_budget_usd"] == 2.5
    assert alignment["approved_budget_covers_estimate_high"] is False
    assert review["external_action_approval"]["valid"] is True
    assert review["pre_run_budget_estimate"]["valid"] is True


def test_paid_run_requires_external_action_approval_report(tmp_path, monkeypatch):
    module = load_module()
    manifest = tmp_path / "models.yaml"
    manifest.write_text(
        "\n".join(
            [
                "models:",
                "  - slug: gpt-4_1-mini-openai-direct-canary",
                "    source_config: config-gpt-4.1-mini-2025-04-14.yaml",
                "    run_name: 'taiwan/full/openai/gpt-4.1-mini: canary'",
                "    openclaw_model: 'openai-direct/gpt-4.1-mini-2025-04-14'",
                "    agentic_thinking: 'off'",
                "    swe_thinking: 'off'",
                "    canary: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "outputs"
    budget = output_root / "openai_canary_budget_estimate.json"
    write_budget_estimate(budget)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_taiwan_full_eval_batch.py",
            "--manifest",
            str(manifest),
            "--canary",
            "--phase",
            "nonagentic",
            "--generated-config-dir",
            str(tmp_path / "generated"),
            "--output-root",
            str(output_root),
            "--run-purpose",
            "paid schema test",
            "--expected-cost-band",
            "$1-$3",
            "--pre-run-budget-estimate-json",
            str(budget),
        ],
    )

    try:
        module.main()
    except SystemExit as exc:
        assert "--external-action-approval-report-json" in str(exc)
    else:
        raise AssertionError("expected missing external action approval to block paid execution")

    review = json.loads(
        (output_root / "canary_nonagentic_paid_run_review.json").read_text(
            encoding="utf-8"
        )
    )
    assert review["status"] == "accountability_fields_missing"
    assert "--external-action-approval-report-json" in review["missing_fields"]
    assert "--external-action-approval-source-packet-json" in review["missing_fields"]
