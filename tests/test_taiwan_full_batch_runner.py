import importlib.util
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace


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


def write_budget_estimate(
    path: Path,
    *,
    target_model: str = "openai-direct/gpt-4.1-mini-2025-04-14",
    target_models: list[str] | None = None,
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
        "estimated_total_usd": {"low": 1.0, "mid": 2.0, "high": 3.0},
        "pricing_source_url": "https://openai.com/index/gpt-4-1/",
        "pricing_note": "provider dashboards are authoritative",
    }
    if target_models is not None:
        payload["target_models"] = target_models
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def write_external_action_approval_report(path: Path, *, source_packet: Path | None = None) -> Path:
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
                "required_approval_count": 6,
                "granted_approval_count": 6,
                "all_required_approvals_granted": True,
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


def test_default_wandb_verify_benchmarks_by_phase():
    module = load_module()

    assert module.default_wandb_verify_benchmarks("full") == ["taiwan_full"]
    assert module.default_wandb_verify_benchmarks("agentic") == [
        "agentic_math",
        "agentic_swe",
    ]
    assert module.default_wandb_verify_benchmarks("nonagentic") == []


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
    assert record["output_json"].endswith(
        "run_eval_preflight/agentic-gpt-4_1-mini.json"
    )
    command = record["command"]
    assert command[:2] == ["python3", "scripts/run_eval.py"]
    assert "--preflight" in command
    assert "--preflight-json" in command
    assert "base_config_taiwan.yaml" in command


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
                "  swebench_pro: false",
                "  aggregate_taiwan: false",
                "agentic_math:",
                "  openclaw_model: openai-direct/gpt-4.1-mini-2025-04-14",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    expectations = module.wandb_verify_config_expectations(
        config,
        benchmark="agentic_math",
    )

    assert expectations == {
        "agentic_math.openclaw_model": "openai-direct/gpt-4.1-mini-2025-04-14",
        "model.pretrained_model_name_or_path": "gpt-4.1-mini-2025-04-14",
        "run.agentic_math": True,
        "wandb.run_name": "taiwan/full/openai/gpt-4.1-mini: canary",
    }


def test_run_wandb_completion_verification_writes_payload(tmp_path, monkeypatch):
    module = load_module()

    def fake_run(command, cwd, env, text, capture_output, check):
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

    def fake_run(command, cwd, env, text, capture_output, check):
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
    )

    assert command[:2] == ["python3", str(module.WEAVE_AGENTS_VERIFY_RUNNER)]
    assert ["--agent-name", "nejumi-taiwan-openclaw"] == command[2:4]
    assert ["--limit", "50"] == command[4:6]
    assert "--no-require-content" not in command
    assert "--require-tool-span" in command
    assert "--require-tool-content" in command
    assert "--require-usage" in command
    assert command[-2:] == ["--conversation-id-contains", "agentic_math"]


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

    def fake_run(command, cwd, env, text, capture_output, check):
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

    def fake_run(command, cwd, env, text, capture_output, check):
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
    gate = tmp_path / "content_canary.gate.json"
    gate.write_text(
        json.dumps(
            {
                "ok": True,
                "status": "passed",
                "model": "openai-direct/test-mini",
                "canary_id": "CANARY",
            }
        ),
        encoding="utf-8",
    )

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


def test_weave_content_canary_gate_blocks_stale_passed_json(tmp_path):
    module = load_module()
    gate = tmp_path / "content_canary.gate.json"
    gate.write_text(
        json.dumps(
            {
                "ok": True,
                "status": "passed",
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
                "ok": True,
                "status": "passed",
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
    assert record["required_approval_count"] == 6
    assert record["granted_approval_count"] == 6
    assert record["source_binding"]["bound"] is True
    assert record["source_packet_path_matches_expected"] is True
    assert record["source_packet_sha256_matches_expected"] is True
    assert record["expected_source_packet_json"] == str(source_packet)
    assert len(record["sha256"]) == 64
    assert record["errors"] == []


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
                "    judge_model: 'gpt-4.1-mini-2025-04-14'",
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
            "--swebench-pro-nemoclaw-sandbox",
            "nejumi-taiwan",
            "--swebench-pro-nemoclaw-checkout-transfer-mode",
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
    assert guard["records"][0]["agentic_math_use_task_agent"] is True
    assert guard["records"][0]["swebench_pro_nemoclaw_sandbox"] == "nejumi-taiwan"
    assert guard["records"][0]["swebench_pro_nemoclaw_checkout_transfer_mode"] == "copy"
    preflights = review["run_eval_preflights"]
    assert len(preflights) == 1
    assert preflights[0]["required_before_run_eval"] is True
    assert preflights[0]["output_json"].endswith(
        "run_eval_preflight/agentic-gpt-4_1-mini-openai-direct-canary.json"
    )
    assert "--preflight" in preflights[0]["command"]
    assert "--preflight-json" in preflights[0]["command"]


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
                "    judge_model: 'gpt-4.1-mini-2025-04-14'",
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
    assert "agentic_math.nemoclaw_sandbox must be set" in guard["errors"][0]


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
                "    judge_model: 'gpt-4.1-mini-2025-04-14'",
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
                "    judge_model: 'gpt-4.1-mini-2025-04-14'",
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
                "    judge_model: 'gpt-4.1-mini-2025-04-14'",
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
