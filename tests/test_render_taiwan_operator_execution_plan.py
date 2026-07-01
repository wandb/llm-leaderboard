import hashlib
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tools" / "render_taiwan_operator_execution_plan.py"


def write_operator_plan(path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "status": "pending",
        "operator_next_steps": {
            "steps": [
                {
                    "order": 1,
                    "gate": "wandb_completion",
                    "status": "missing_required_benchmarks",
                    "next_action": "Verify W&B completion.",
                    "requires_paid_api": False,
                    "requires_wandb_access": True,
                    "requires_wandb_write": False,
                    "requires_third_party_acceptance": False,
                    "requires_nemoclaw_install": False,
                    "requires_scope_confirmation": True,
                    "commands": [
                        "Review docs/taiwan_paid_run_review_template.md first.",
                        "uv run python scripts/tools/verify_taiwan_wandb_completion.py --run-id RUN_ID --benchmark agentic_swe --json outputs/taiwan_full_eval/wandb_completion/agentic_swe-RUN_ID.json",
                    ],
                    "evidence_to_produce": [
                        "outputs/taiwan_full_eval/wandb_completion/agentic_swe-RUN_ID.json"
                    ],
                    "warnings": ["Use the reviewed run scope."],
                },
                {
                    "order": 2,
                    "gate": "weave_content_canary",
                    "status": "failed",
                    "next_action": "Run content canary.",
                    "requires_paid_api": True,
                    "requires_wandb_access": True,
                    "requires_wandb_write": True,
                    "requires_third_party_acceptance": False,
                    "requires_nemoclaw_install": False,
                    "requires_scope_confirmation": False,
                    "commands": [
                        "uv run python scripts/tools/run_weave_agents_content_canary.py --execute --canary-id CONTENT_CANARY_YYYYMMDDTHHMM --nemoclaw-sandbox nejumi-taiwan --external-action-approval-source-packet-json temp/external_action_approval_packet.json --external-action-approval-report-json temp/external_action_approval.verify.json"
                    ],
                    "evidence_to_produce": [
                        "outputs/weave_agents_content_canary/plans/weave_agents_content_canary_CONTENT_CANARY_YYYYMMDDTHHMM.gate.json"
                    ],
                    "warnings": [],
                },
            ]
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_source_packet(path: Path, marker: str = "source") -> Path:
    path.write_text(
        json.dumps({"schema_version": 1, "marker": marker}, sort_keys=True),
        encoding="utf-8",
    )
    return path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_external_action_approval_report(
    path: Path,
    source_packet: Path,
    *,
    approved_model_scope: str = "openai-direct/gpt-4.1-mini-2025-04-14 canary",
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ok": True,
                "status": "approved",
                "approval_packet_json": "temp/reviewed_approval.json",
                "required_approval_count": 6,
                "granted_approval_count": 6,
                "all_required_approvals_granted": True,
                "source_binding": {
                    "source_packet_json": str(source_packet),
                    "source_packet_readable": True,
                    "source_approval_packet_sha256": sha256_file(source_packet),
                    "bound": True,
                    "errors": [],
                },
                "approval_results": [
                    {
                        "requirement": "paid_api",
                        "required": True,
                        "approved": True,
                        "approved_budget_usd": 25.0,
                        "approved_model_scope": approved_model_scope,
                        "errors": [],
                    }
                ],
                "will_execute_external_actions": False,
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def write_weave_content_canary_gate(
    path: Path,
    *,
    ok: bool = True,
    status: str = "passed",
    generated_at: float = 9_999_999_999.0,
    failure_kind: str = "",
    paid_api_attempted: bool | None = None,
    recommended_next_action: str = "",
) -> Path:
    payload = {
        "schema_version": 1,
        "ok": ok,
        "status": status,
        "generated_at": generated_at,
        "failure_kind": failure_kind,
        "canary_id": "CONTENT_CANARY_TEST",
    }
    if paid_api_attempted is not None:
        payload["paid_api_attempted"] = paid_api_attempted
    if recommended_next_action:
        payload["recommended_next_action"] = recommended_next_action
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def native_weave_content_canary_gate_payload() -> dict:
    return {
        "schema_version": 1,
        "ok": True,
        "gate": "weave_agents_content_canary",
        "status": "passed",
        "generated_at": 9_999_999_999.0,
        "model": "openai-direct/test-mini",
        "canary_id": "CONTENT_CANARY_TEST",
        "task_id": "weave_agents_content_canary_CONTENT_CANARY_TEST",
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
        "paid_api_attempted": True,
        "command_ok": True,
        "command_returncode": 0,
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


def write_native_weave_content_canary_gate(path: Path) -> Path:
    path.write_text(
        json.dumps(native_weave_content_canary_gate_payload()),
        encoding="utf-8",
    )
    return path


def write_agentic_batch_operator_plan(
    path: Path,
    *,
    source_packet: Path,
    approval_report: Path,
    gate_json: Path,
) -> Path:
    payload = {
        "schema_version": 1,
        "status": "pending",
        "operator_next_steps": {
            "steps": [
                {
                    "order": 1,
                    "gate": "paid_run_review_package",
                    "status": "incomplete_reviews",
                    "next_action": "Run agentic command.",
                    "requires_paid_api": True,
                    "requires_wandb_access": True,
                    "requires_wandb_write": True,
                    "requires_third_party_acceptance": False,
                    "requires_nemoclaw_install": False,
                    "requires_scope_confirmation": False,
                    "commands": [
                        (
                            "uv run python scripts/tools/run_taiwan_full_eval_batch.py "
                            "--phase agentic --yes "
                            "--external-action-approval-source-packet-json "
                            f"{source_packet} "
                            "--external-action-approval-report-json "
                            f"{approval_report} "
                            "--require-nemoclaw-agentic-config "
                            "--agentic-math-nemoclaw-sandbox nejumi-taiwan "
                            "--agentic-math-nemoclaw-openclaw-config-path "
                            "/sandbox/.openclaw/openclaw.json "
                            "--swebench-pro-nemoclaw-sandbox nejumi-taiwan "
                            "--swebench-pro-nemoclaw-openclaw-config-path "
                            "/sandbox/.openclaw/openclaw.json "
                            "--swebench-pro-nemoclaw-checkout-transfer-mode copy "
                            "--verify-wandb-completion "
                            "--verify-weave-agents "
                            "--wandb-run-id-prefix twcanary-test "
                            "--weave-agents-require-tool-span "
                            "--weave-agents-require-tool-content "
                            "--weave-agents-require-usage "
                            f"--weave-content-canary-gate {gate_json} "
                            "--require-weave-content-canary"
                        )
                    ],
                    "evidence_to_produce": [
                        "outputs/taiwan_full_eval/canary_agentic_paid_run_review.json"
                    ],
                    "warnings": [],
                }
            ]
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_render_operator_execution_plan_resolves_placeholders_and_shell(tmp_path):
    operator_plan = write_operator_plan(tmp_path / "operator_plan.json")
    source_packet = write_source_packet(tmp_path / "external_action_approval_packet.json")
    approval_report = write_external_action_approval_report(
        tmp_path / "approval.verify.json",
        source_packet,
    )
    output_json = tmp_path / "execution_plan.json"
    output_md = tmp_path / "execution_plan.md"
    output_sh = tmp_path / "execution_plan.sh"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--operator-plan-json",
            str(operator_plan),
            "--gate",
            "wandb_completion",
            "--run-id",
            "abc123",
            "--external-action-approval-source-packet-json",
            str(source_packet),
            "--external-action-approval-report-json",
            str(approval_report),
            "--output-json",
            str(output_json),
            "--markdown",
            str(output_md),
            "--shell-script",
            str(output_sh),
            "--require-ready",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["status"] == "placeholder_ready"
    assert payload["source_operator_plan"] == str(operator_plan)
    assert payload["source_operator_plan_sha256"] == sha256_file(operator_plan)
    assert payload["all_ready_to_execute_without_placeholder"] is True
    assert payload["all_ready_for_external_execution"] is True
    assert payload["external_action_approval"]["valid"] is True
    assert payload["command_policy"]["valid"] is True
    assert payload["command_policy"]["error_count"] == 0
    assert payload["external_action_approval"]["source_packet_path_matches_expected"] is True
    assert payload["external_action_approval"]["source_packet_sha256_matches_expected"] is True
    assert payload["unresolved_placeholder_tokens"] == []
    assert payload["requirement_counts"]["wandb_access_steps"] == 1
    assert payload["requirement_counts"]["scope_confirmation_steps"] == 1
    assert payload["total_command_count"] == 2
    assert payload["total_executable_command_count"] == 1
    assert payload["total_evidence_path_count"] == 1
    assert payload["steps"][0]["command_count"] == 2
    assert payload["steps"][0]["executable_command_count"] == 1
    assert payload["steps"][0]["non_executable_note_count"] == 1
    assert payload["steps"][0]["evidence_path_count"] == 1
    command = payload["steps"][0]["executable_commands"][0]
    assert "RUN_ID" not in command
    assert "abc123" in command
    assert payload["steps"][0]["non_executable_notes"] == [
        "Review docs/taiwan_paid_run_review_template.md first."
    ]

    markdown = output_md.read_text(encoding="utf-8")
    assert "Status: `placeholder_ready`" in markdown
    assert f"Source operator plan SHA-256: `{sha256_file(operator_plan)}`" in markdown
    assert "- Total commands: `2`" in markdown
    assert "- Total executable commands: `1`" in markdown
    assert "- Total evidence paths: `1`" in markdown
    assert "## Command Policy" in markdown
    assert "- Valid: `true`" in markdown
    assert "- Commands: `2` total, `1` executable, `1` notes" in markdown
    assert "- Evidence paths: `1`" in markdown
    assert "abc123" in markdown
    assert "RUN_ID" not in markdown

    shell = output_sh.read_text(encoding="utf-8")
    assert "# NOTE: Review docs/taiwan_paid_run_review_template.md first." in shell
    assert f"# Source operator plan: {operator_plan}" in shell
    assert f"# Source operator plan SHA-256: {sha256_file(operator_plan)}" in shell
    assert "# Commands: 2 total, 1 executable, 1 notes" in shell
    assert "# Evidence paths: 1" in shell
    assert f"# External action approval report: {approval_report}" in shell
    assert f"# External action approval source packet: {source_packet}" in shell
    assert "# Command policy valid: true" in shell
    assert "abc123" in shell
    assert "RUN_ID" not in shell
    assert output_sh.stat().st_mode & 0o100


def test_render_operator_execution_plan_refuses_shell_when_unresolved(tmp_path):
    operator_plan = write_operator_plan(tmp_path / "operator_plan.json")
    output_json = tmp_path / "execution_plan.json"
    output_sh = tmp_path / "execution_plan.sh"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--operator-plan-json",
            str(operator_plan),
            "--gate",
            "wandb_completion",
            "--output-json",
            str(output_json),
            "--shell-script",
            str(output_sh),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "refusing to write shell script while placeholders remain" in result.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["status"] == "templates_pending_values"
    assert payload["unresolved_placeholder_tokens"] == ["RUN_ID"]
    assert not output_sh.exists()


def test_render_operator_execution_plan_refuses_shell_without_external_approval(tmp_path):
    operator_plan = write_operator_plan(tmp_path / "operator_plan.json")
    output_json = tmp_path / "execution_plan.json"
    output_sh = tmp_path / "execution_plan.sh"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--operator-plan-json",
            str(operator_plan),
            "--gate",
            "wandb_completion",
            "--run-id",
            "abc123",
            "--output-json",
            str(output_json),
            "--shell-script",
            str(output_sh),
            "--require-ready",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "external-action approval" in result.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["all_ready_to_execute_without_placeholder"] is True
    assert payload["all_ready_for_external_execution"] is False
    assert payload["external_action_approval"]["valid"] is False
    assert not output_sh.exists()


def test_render_operator_execution_plan_refuses_shell_without_source_packet(tmp_path):
    operator_plan = write_operator_plan(tmp_path / "operator_plan.json")
    source_packet = write_source_packet(tmp_path / "external_action_approval_packet.json")
    approval_report = write_external_action_approval_report(
        tmp_path / "approval.verify.json",
        source_packet,
    )
    output_json = tmp_path / "execution_plan.json"
    output_sh = tmp_path / "execution_plan.sh"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--operator-plan-json",
            str(operator_plan),
            "--gate",
            "wandb_completion",
            "--run-id",
            "abc123",
            "--external-action-approval-report-json",
            str(approval_report),
            "--output-json",
            str(output_json),
            "--shell-script",
            str(output_sh),
            "--require-ready",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["all_ready_to_execute_without_placeholder"] is True
    assert payload["all_ready_for_external_execution"] is False
    assert any(
        "source packet is required" in error
        for error in payload["external_action_approval"]["errors"]
    )
    assert not output_sh.exists()


def test_render_operator_execution_plan_refuses_shell_when_source_packet_mismatches(tmp_path):
    operator_plan = write_operator_plan(tmp_path / "operator_plan.json")
    source_packet = write_source_packet(tmp_path / "external_action_approval_packet.json")
    other_packet = write_source_packet(
        tmp_path / "other_external_action_approval_packet.json",
        marker="other",
    )
    approval_report = write_external_action_approval_report(
        tmp_path / "approval.verify.json",
        source_packet,
    )
    output_json = tmp_path / "execution_plan.json"
    output_sh = tmp_path / "execution_plan.sh"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--operator-plan-json",
            str(operator_plan),
            "--gate",
            "wandb_completion",
            "--run-id",
            "abc123",
            "--external-action-approval-source-packet-json",
            str(other_packet),
            "--external-action-approval-report-json",
            str(approval_report),
            "--output-json",
            str(output_json),
            "--shell-script",
            str(output_sh),
            "--require-ready",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["all_ready_to_execute_without_placeholder"] is True
    assert payload["all_ready_for_external_execution"] is False
    assert payload["external_action_approval"]["source_packet_path_matches_expected"] is False
    assert payload["external_action_approval"]["source_packet_sha256_matches_expected"] is False
    assert any(
        "source_binding.source_packet_json does not match" in error
        for error in payload["external_action_approval"]["errors"]
    )
    assert any(
        "source_binding.source_approval_packet_sha256 does not match" in error
        for error in payload["external_action_approval"]["errors"]
    )
    assert not output_sh.exists()


def test_render_operator_execution_plan_resolves_timestamp_canary_tokens(tmp_path):
    operator_plan = write_operator_plan(tmp_path / "operator_plan.json")
    source_packet = write_source_packet(tmp_path / "external_action_approval_packet.json")
    approval_report = write_external_action_approval_report(
        tmp_path / "approval.verify.json",
        source_packet,
    )
    payload = json.loads(operator_plan.read_text(encoding="utf-8"))
    payload["operator_next_steps"]["steps"][1]["commands"][0] = payload[
        "operator_next_steps"
    ]["steps"][1]["commands"][0].replace(
        "temp/external_action_approval_packet.json",
        str(source_packet),
    ).replace(
        "temp/external_action_approval.verify.json",
        str(approval_report),
    )
    operator_plan.write_text(json.dumps(payload), encoding="utf-8")
    output_json = tmp_path / "execution_plan.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--operator-plan-json",
            str(operator_plan),
            "--gate",
            "weave_content_canary",
            "--timestamp",
            "20260629T0105",
            "--external-action-approval-source-packet-json",
            str(source_packet),
            "--external-action-approval-report-json",
            str(approval_report),
            "--output-json",
            str(output_json),
            "--require-ready",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["status"] == "placeholder_ready"
    rendered = "\n".join(
        payload["steps"][0]["commands"] + payload["steps"][0]["evidence_to_produce"]
    )
    assert "CONTENT_CANARY_20260629T0105" in rendered
    assert "CONTENT_CANARY_YYYYMMDDTHHMM" not in rendered


def test_render_operator_execution_plan_binds_placeholder_approval_paths_to_reviewed_paths(tmp_path):
    operator_plan = write_operator_plan(tmp_path / "operator_plan.json")
    source_packet = write_source_packet(tmp_path / "external_action_approval_packet.json")
    approval_report = write_external_action_approval_report(
        tmp_path / "approval.verify.json",
        source_packet,
    )
    payload = json.loads(operator_plan.read_text(encoding="utf-8"))
    payload["operator_next_steps"]["steps"][1]["commands"][0] = (
        "uv run python scripts/tools/run_weave_agents_content_canary.py "
        "--execute --canary-id CONTENT_CANARY_YYYYMMDDTHHMM "
        "--nemoclaw-sandbox nejumi-taiwan "
        "--external-action-approval-source-packet-json "
        "outputs/taiwan_release_evidence/bundle_YYYYMMDDTHHMM/"
        "external_action_approval_packet.json "
        "--external-action-approval-report-json "
        "temp/taiwan_external_action_approval_REVIEWED_YYYYMMDDTHHMM.verify.json"
    )
    operator_plan.write_text(json.dumps(payload), encoding="utf-8")
    output_json = tmp_path / "execution_plan.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--operator-plan-json",
            str(operator_plan),
            "--gate",
            "weave_content_canary",
            "--timestamp",
            "20260701T123456",
            "--external-action-approval-source-packet-json",
            str(source_packet),
            "--external-action-approval-report-json",
            str(approval_report),
            "--output-json",
            str(output_json),
            "--require-ready",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["command_policy"]["valid"] is True
    command = payload["steps"][0]["commands"][0]
    assert str(source_packet) in command
    assert str(approval_report) in command
    assert "bundle_20260701T123456/external_action_approval_packet.json" not in command
    assert "taiwan_external_action_approval_REVIEWED_20260701T123456.verify.json" not in command


def test_render_operator_execution_plan_rejects_command_source_packet_mismatch(tmp_path):
    source_packet = write_source_packet(tmp_path / "external_action_approval_packet.json")
    other_packet = write_source_packet(
        tmp_path / "other_external_action_approval_packet.json",
        marker="other",
    )
    approval_report = write_external_action_approval_report(
        tmp_path / "approval.verify.json",
        source_packet,
    )
    gate_json = write_native_weave_content_canary_gate(
        tmp_path / "native_content_canary.gate.json",
    )
    operator_plan = write_agentic_batch_operator_plan(
        tmp_path / "operator_plan.json",
        source_packet=other_packet,
        approval_report=approval_report,
        gate_json=gate_json,
    )
    output_json = tmp_path / "execution_plan.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--operator-plan-json",
            str(operator_plan),
            "--gate",
            "paid_run_review_package",
            "--external-action-approval-source-packet-json",
            str(source_packet),
            "--external-action-approval-report-json",
            str(approval_report),
            "--output-json",
            str(output_json),
            "--require-ready",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    rendered = json.loads(output_json.read_text(encoding="utf-8"))
    assert rendered["external_action_approval"]["valid"] is True
    assert rendered["command_policy"]["valid"] is False
    assert any(
        "--external-action-approval-source-packet-json command value does not match"
        in error
        for error in rendered["command_policy"]["errors"]
    )


def test_render_operator_execution_plan_rejects_command_approval_report_mismatch(tmp_path):
    source_packet = write_source_packet(tmp_path / "external_action_approval_packet.json")
    approval_report = write_external_action_approval_report(
        tmp_path / "approval.verify.json",
        source_packet,
    )
    other_report = write_external_action_approval_report(
        tmp_path / "other_approval.verify.json",
        source_packet,
    )
    gate_json = write_native_weave_content_canary_gate(
        tmp_path / "native_content_canary.gate.json",
    )
    operator_plan = write_agentic_batch_operator_plan(
        tmp_path / "operator_plan.json",
        source_packet=source_packet,
        approval_report=other_report,
        gate_json=gate_json,
    )
    output_json = tmp_path / "execution_plan.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--operator-plan-json",
            str(operator_plan),
            "--gate",
            "paid_run_review_package",
            "--external-action-approval-source-packet-json",
            str(source_packet),
            "--external-action-approval-report-json",
            str(approval_report),
            "--output-json",
            str(output_json),
            "--require-ready",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    rendered = json.loads(output_json.read_text(encoding="utf-8"))
    assert rendered["external_action_approval"]["valid"] is True
    assert rendered["command_policy"]["valid"] is False
    assert any(
        "--external-action-approval-report-json command value does not match"
        in error
        for error in rendered["command_policy"]["errors"]
    )


def test_render_operator_execution_plan_binds_to_source_release_gate(tmp_path):
    operator_plan = write_operator_plan(tmp_path / "operator_plan.json")
    release_gate = tmp_path / "taiwan_release_gate_RELEASE.json"
    release_gate.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    payload = json.loads(operator_plan.read_text(encoding="utf-8"))
    payload["source_release_gate_json"] = str(release_gate)
    payload["release_gate_json"] = str(release_gate)
    operator_plan.write_text(json.dumps(payload), encoding="utf-8")
    source_packet = write_source_packet(tmp_path / "external_action_approval_packet.json")
    approval_report = write_external_action_approval_report(
        tmp_path / "approval.verify.json",
        source_packet,
    )
    output_json = tmp_path / "execution_plan.json"
    output_md = tmp_path / "execution_plan.md"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--operator-plan-json",
            str(operator_plan),
            "--gate",
            "wandb_completion",
            "--run-id",
            "abc123",
            "--release-gate-json",
            str(release_gate),
            "--external-action-approval-source-packet-json",
            str(source_packet),
            "--external-action-approval-report-json",
            str(approval_report),
            "--output-json",
            str(output_json),
            "--markdown",
            str(output_md),
            "--require-ready",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    rendered = json.loads(output_json.read_text(encoding="utf-8"))
    binding = rendered["release_gate_binding"]
    assert binding["required"] is True
    assert binding["valid"] is True
    assert binding["source_path_matches_expected"] is True
    assert binding["source_exists"] is True
    assert binding["expected_exists"] is True
    markdown = output_md.read_text(encoding="utf-8")
    assert "## Release Gate Binding" in markdown
    assert f"- Expected release gate JSON: `{release_gate}`" in markdown


def test_render_operator_execution_plan_rejects_wrong_release_gate(tmp_path):
    operator_plan = write_operator_plan(tmp_path / "operator_plan.json")
    release_gate = tmp_path / "taiwan_release_gate_RELEASE.json"
    release_gate.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    other_release_gate = tmp_path / "taiwan_release_gate_OTHER.json"
    other_release_gate.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    payload = json.loads(operator_plan.read_text(encoding="utf-8"))
    payload["source_release_gate_json"] = str(release_gate)
    payload["release_gate_json"] = str(release_gate)
    operator_plan.write_text(json.dumps(payload), encoding="utf-8")
    source_packet = write_source_packet(tmp_path / "external_action_approval_packet.json")
    approval_report = write_external_action_approval_report(
        tmp_path / "approval.verify.json",
        source_packet,
    )
    output_json = tmp_path / "execution_plan.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--operator-plan-json",
            str(operator_plan),
            "--gate",
            "wandb_completion",
            "--run-id",
            "abc123",
            "--release-gate-json",
            str(other_release_gate),
            "--external-action-approval-source-packet-json",
            str(source_packet),
            "--external-action-approval-report-json",
            str(approval_report),
            "--output-json",
            str(output_json),
            "--require-ready",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "source release gate JSON does not match" in result.stderr
    rendered = json.loads(output_json.read_text(encoding="utf-8"))
    assert rendered["all_ready_to_execute_without_placeholder"] is True
    assert rendered["all_ready_for_external_execution"] is False
    binding = rendered["release_gate_binding"]
    assert binding["valid"] is False
    assert binding["source_path_matches_expected"] is False


def test_render_operator_execution_plan_rejects_stale_agentic_batch_command(tmp_path):
    operator_plan = tmp_path / "operator_plan.json"
    payload = {
        "schema_version": 1,
        "status": "pending",
        "operator_next_steps": {
            "steps": [
                {
                    "order": 1,
                    "gate": "paid_run_review_package",
                    "status": "incomplete_reviews",
                    "next_action": "Run stale agentic command.",
                    "requires_paid_api": True,
                    "requires_wandb_access": True,
                    "requires_wandb_write": True,
                    "requires_third_party_acceptance": False,
                    "requires_nemoclaw_install": False,
                    "requires_scope_confirmation": False,
                    "commands": [
                        "uv run python scripts/tools/run_taiwan_full_eval_batch.py --phase agentic --yes"
                    ],
                    "evidence_to_produce": [
                        "outputs/taiwan_full_eval/canary_agentic_paid_run_review.json"
                    ],
                    "warnings": [],
                }
            ]
        },
    }
    operator_plan.write_text(json.dumps(payload), encoding="utf-8")
    source_packet = write_source_packet(tmp_path / "external_action_approval_packet.json")
    approval_report = write_external_action_approval_report(
        tmp_path / "approval.verify.json",
        source_packet,
    )
    output_json = tmp_path / "execution_plan.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--operator-plan-json",
            str(operator_plan),
            "--gate",
            "paid_run_review_package",
            "--external-action-approval-source-packet-json",
            str(source_packet),
            "--external-action-approval-report-json",
            str(approval_report),
            "--output-json",
            str(output_json),
            "--require-ready",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    rendered = json.loads(output_json.read_text(encoding="utf-8"))
    assert rendered["all_ready_to_execute_without_placeholder"] is True
    assert rendered["all_ready_for_external_execution"] is False
    policy = rendered["command_policy"]
    assert policy["valid"] is False
    assert any("--require-nemoclaw-agentic-config" in error for error in policy["errors"])
    assert any("--weave-content-canary-gate" in error for error in policy["errors"])
    assert any("--weave-agents-require-usage" in error for error in policy["errors"])
    assert any(
        "--external-action-approval-report-json" in error
        for error in policy["errors"]
    )


def test_render_operator_execution_plan_allows_openai_direct_canary_command(tmp_path):
    source_packet = write_source_packet(tmp_path / "external_action_approval_packet.json")
    approval_report = write_external_action_approval_report(
        tmp_path / "approval.verify.json",
        source_packet,
    )
    operator_plan = tmp_path / "operator_plan.json"
    payload = {
        "schema_version": 1,
        "status": "pending",
        "operator_next_steps": {
            "steps": [
                {
                    "order": 1,
                    "gate": "one_model_full_canary",
                    "status": "incomplete",
                    "next_action": "Run OpenAI-direct nonagentic canary.",
                    "requires_paid_api": True,
                    "requires_wandb_access": True,
                    "requires_wandb_write": True,
                    "requires_third_party_acceptance": False,
                    "requires_nemoclaw_install": False,
                    "requires_scope_confirmation": False,
                    "commands": [
                        (
                            "uv run python scripts/tools/run_taiwan_full_eval_batch.py "
                            "--manifest configs/taiwan_openai_canary_models.yaml "
                            "--canary --phase nonagentic "
                            "--generated-config-dir configs/taiwan_full/generated_openai_canary_nonagentic "
                            "--output-root outputs/taiwan_full_eval "
                            "--wandb-run-id-prefix twcanary-openai-mini-20260701 "
                            "--yes --run-purpose 'OpenAI-direct gpt-4.1-mini one-model nonagentic phase' "
                            "--expected-cost-band 'low-cost OpenAI-direct canary; confirm cap before execution' "
                            "--pre-run-budget-estimate-json outputs/taiwan_full_eval/openai_canary_budget_estimate.json "
                            f"--external-action-approval-source-packet-json {source_packet} "
                            f"--external-action-approval-report-json {approval_report}"
                        )
                    ],
                    "evidence_to_produce": [
                        "outputs/taiwan_full_eval/canary_nonagentic_paid_run_review.json"
                    ],
                    "warnings": [],
                }
            ]
        },
    }
    operator_plan.write_text(json.dumps(payload), encoding="utf-8")
    output_json = tmp_path / "execution_plan.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--operator-plan-json",
            str(operator_plan),
            "--gate",
            "one_model_full_canary",
            "--external-action-approval-source-packet-json",
            str(source_packet),
            "--external-action-approval-report-json",
            str(approval_report),
            "--output-json",
            str(output_json),
            "--require-ready",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    rendered = json.loads(output_json.read_text(encoding="utf-8"))
    assert rendered["command_policy"]["valid"] is True
    assert rendered["approval_scope_policy"]["valid"] is True
    assert rendered["all_ready_for_external_execution"] is True


def test_render_operator_execution_plan_rejects_canary_scope_mismatch(tmp_path):
    source_packet = write_source_packet(tmp_path / "external_action_approval_packet.json")
    approval_report = write_external_action_approval_report(
        tmp_path / "approval.verify.json",
        source_packet,
        approved_model_scope="OpenAI canary without exact model scope",
    )
    operator_plan = tmp_path / "operator_plan.json"
    payload = {
        "schema_version": 1,
        "status": "pending",
        "operator_next_steps": {
            "steps": [
                {
                    "order": 1,
                    "gate": "one_model_full_canary",
                    "status": "incomplete",
                    "next_action": "Run OpenAI-direct nonagentic canary.",
                    "requires_paid_api": True,
                    "requires_wandb_access": True,
                    "requires_wandb_write": True,
                    "requires_third_party_acceptance": False,
                    "requires_nemoclaw_install": False,
                    "requires_scope_confirmation": False,
                    "commands": [
                        (
                            "uv run python scripts/tools/run_taiwan_full_eval_batch.py "
                            "--manifest configs/taiwan_openai_canary_models.yaml "
                            "--canary --phase nonagentic "
                            "--generated-config-dir configs/taiwan_full/generated_openai_canary_nonagentic "
                            "--output-root outputs/taiwan_full_eval "
                            "--wandb-run-id-prefix twcanary-openai-mini-20260701 "
                            "--yes --run-purpose 'OpenAI-direct gpt-4.1-mini one-model nonagentic phase' "
                            "--expected-cost-band 'low-cost OpenAI-direct canary; confirm cap before execution' "
                            "--pre-run-budget-estimate-json outputs/taiwan_full_eval/openai_canary_budget_estimate.json "
                            f"--external-action-approval-source-packet-json {source_packet} "
                            f"--external-action-approval-report-json {approval_report}"
                        )
                    ],
                    "evidence_to_produce": [
                        "outputs/taiwan_full_eval/canary_nonagentic_paid_run_review.json"
                    ],
                    "warnings": [],
                }
            ]
        },
    }
    operator_plan.write_text(json.dumps(payload), encoding="utf-8")
    output_json = tmp_path / "execution_plan.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--operator-plan-json",
            str(operator_plan),
            "--gate",
            "one_model_full_canary",
            "--external-action-approval-source-packet-json",
            str(source_packet),
            "--external-action-approval-report-json",
            str(approval_report),
            "--output-json",
            str(output_json),
            "--require-ready",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    rendered = json.loads(output_json.read_text(encoding="utf-8"))
    scope_policy = rendered["approval_scope_policy"]
    assert scope_policy["valid"] is False
    assert scope_policy["required"] is True
    assert any("openai-direct/gpt-4.1-mini" in error for error in scope_policy["errors"])


def test_render_operator_execution_plan_rejects_non_openai_canary_command(tmp_path):
    source_packet = write_source_packet(tmp_path / "external_action_approval_packet.json")
    approval_report = write_external_action_approval_report(
        tmp_path / "approval.verify.json",
        source_packet,
    )
    operator_plan = tmp_path / "operator_plan.json"
    payload = {
        "schema_version": 1,
        "status": "pending",
        "operator_next_steps": {
            "steps": [
                {
                    "order": 1,
                    "gate": "one_model_full_canary",
                    "status": "incomplete",
                    "next_action": "Run stale Gemini canary.",
                    "requires_paid_api": True,
                    "requires_wandb_access": True,
                    "requires_wandb_write": True,
                    "requires_third_party_acceptance": False,
                    "requires_nemoclaw_install": False,
                    "requires_scope_confirmation": False,
                    "commands": [
                        (
                            "uv run python scripts/tools/run_taiwan_full_eval_batch.py "
                            "--manifest configs/config-gemini-3_1-pro-preview.yaml "
                            "--canary --phase nonagentic "
                            "--generated-config-dir configs/taiwan_full/generated_openai_canary_nonagentic "
                            "--output-root outputs/taiwan_full_eval "
                            "--wandb-run-id-prefix twcanary-gemini-20260701 "
                            "--yes --run-purpose 'Gemini one-model nonagentic phase' "
                            "--expected-cost-band 'not approved for current canary' "
                            "--pre-run-budget-estimate-json outputs/taiwan_full_eval/openai_canary_budget_estimate.json "
                            f"--external-action-approval-source-packet-json {source_packet} "
                            f"--external-action-approval-report-json {approval_report}"
                        )
                    ],
                    "evidence_to_produce": [
                        "outputs/taiwan_full_eval/canary_nonagentic_paid_run_review.json"
                    ],
                    "warnings": [],
                }
            ]
        },
    }
    operator_plan.write_text(json.dumps(payload), encoding="utf-8")
    output_json = tmp_path / "execution_plan.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--operator-plan-json",
            str(operator_plan),
            "--gate",
            "one_model_full_canary",
            "--external-action-approval-source-packet-json",
            str(source_packet),
            "--external-action-approval-report-json",
            str(approval_report),
            "--output-json",
            str(output_json),
            "--require-ready",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    rendered = json.loads(output_json.read_text(encoding="utf-8"))
    policy = rendered["command_policy"]
    assert policy["valid"] is False
    assert any(
        "OpenAI-direct canary command uses forbidden provider marker(s): gemini"
        in error
        for error in policy["errors"]
    )
    assert any(
        "--manifest configs/taiwan_openai_canary_models.yaml" in error
        for error in policy["errors"]
    )


def test_render_operator_execution_plan_rejects_failed_weave_content_canary_gate(tmp_path):
    source_packet = write_source_packet(tmp_path / "external_action_approval_packet.json")
    approval_report = write_external_action_approval_report(
        tmp_path / "approval.verify.json",
        source_packet,
    )
    gate_json = write_weave_content_canary_gate(
        tmp_path / "failed_content_canary.gate.json",
        ok=False,
        status="provider_failure",
        failure_kind="provider_quota",
    )
    operator_plan = write_agentic_batch_operator_plan(
        tmp_path / "operator_plan.json",
        source_packet=source_packet,
        approval_report=approval_report,
        gate_json=gate_json,
    )
    output_json = tmp_path / "execution_plan.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--operator-plan-json",
            str(operator_plan),
            "--gate",
            "paid_run_review_package",
            "--external-action-approval-source-packet-json",
            str(source_packet),
            "--external-action-approval-report-json",
            str(approval_report),
            "--output-json",
            str(output_json),
            "--require-ready",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    rendered = json.loads(output_json.read_text(encoding="utf-8"))
    assert rendered["all_ready_to_execute_without_placeholder"] is True
    assert rendered["all_ready_for_external_execution"] is False
    policy = rendered["command_policy"]
    assert policy["valid"] is False
    assert any(
        "Weave content canary gate must have ok=true and status=passed" in error
        and "provider_quota" in error
        for error in policy["errors"]
    )


def test_render_operator_execution_plan_explains_external_action_blocked_weave_gate(tmp_path):
    source_packet = write_source_packet(tmp_path / "external_action_approval_packet.json")
    approval_report = write_external_action_approval_report(
        tmp_path / "approval.verify.json",
        source_packet,
    )
    gate_json = write_weave_content_canary_gate(
        tmp_path / "blocked_content_canary.gate.json",
        ok=False,
        status="external_action_approval_missing",
        failure_kind="external_action_approval_missing",
        paid_api_attempted=False,
    )
    operator_plan = write_agentic_batch_operator_plan(
        tmp_path / "operator_plan.json",
        source_packet=source_packet,
        approval_report=approval_report,
        gate_json=gate_json,
    )
    output_json = tmp_path / "execution_plan.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--operator-plan-json",
            str(operator_plan),
            "--gate",
            "paid_run_review_package",
            "--external-action-approval-source-packet-json",
            str(source_packet),
            "--external-action-approval-report-json",
            str(approval_report),
            "--output-json",
            str(output_json),
            "--require-ready",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    rendered = json.loads(output_json.read_text(encoding="utf-8"))
    policy = rendered["command_policy"]
    assert policy["valid"] is False
    assert any(
        "external_action_approval_missing" in error
        and "paid_api_attempted=False" in error
        and "approve the source-bound external-action packet" in error
        and "before any paid API attempt" in error
        for error in policy["errors"]
    )


def test_render_operator_execution_plan_explains_nemoclaw_preflight_blocked_weave_gate(tmp_path):
    source_packet = write_source_packet(tmp_path / "external_action_approval_packet.json")
    approval_report = write_external_action_approval_report(
        tmp_path / "approval.verify.json",
        source_packet,
    )
    gate_json = write_weave_content_canary_gate(
        tmp_path / "blocked_content_canary.gate.json",
        ok=False,
        status="nemoclaw_config_preflight_failed",
        failure_kind="nemoclaw_config_preflight_failed",
        paid_api_attempted=False,
    )
    operator_plan = write_agentic_batch_operator_plan(
        tmp_path / "operator_plan.json",
        source_packet=source_packet,
        approval_report=approval_report,
        gate_json=gate_json,
    )
    output_json = tmp_path / "execution_plan.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--operator-plan-json",
            str(operator_plan),
            "--gate",
            "paid_run_review_package",
            "--external-action-approval-source-packet-json",
            str(source_packet),
            "--external-action-approval-report-json",
            str(approval_report),
            "--output-json",
            str(output_json),
            "--require-ready",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    rendered = json.loads(output_json.read_text(encoding="utf-8"))
    policy = rendered["command_policy"]
    assert policy["valid"] is False
    assert any(
        "nemoclaw_config_preflight_failed" in error
        and "paid_api_attempted=False" in error
        and "NeMoClaw sandbox OpenClaw config" in error
        and "before any paid API attempt" in error
        for error in policy["errors"]
    )


def test_render_operator_execution_plan_rejects_hand_edited_passed_weave_content_canary_gate(tmp_path):
    source_packet = write_source_packet(tmp_path / "external_action_approval_packet.json")
    approval_report = write_external_action_approval_report(
        tmp_path / "approval.verify.json",
        source_packet,
    )
    gate_json = write_weave_content_canary_gate(
        tmp_path / "hand_edited_content_canary.gate.json",
    )
    operator_plan = write_agentic_batch_operator_plan(
        tmp_path / "operator_plan.json",
        source_packet=source_packet,
        approval_report=approval_report,
        gate_json=gate_json,
    )
    output_json = tmp_path / "execution_plan.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--operator-plan-json",
            str(operator_plan),
            "--gate",
            "paid_run_review_package",
            "--external-action-approval-source-packet-json",
            str(source_packet),
            "--external-action-approval-report-json",
            str(approval_report),
            "--output-json",
            str(output_json),
            "--require-ready",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    rendered = json.loads(output_json.read_text(encoding="utf-8"))
    assert rendered["all_ready_to_execute_without_placeholder"] is True
    assert rendered["all_ready_for_external_execution"] is False
    policy = rendered["command_policy"]
    assert policy["valid"] is False
    assert any(
        "missing native Weave verifier contract evidence" in error
        and "weave_verifier_ok must be true" in error
        and "paths must be an object" in error
        for error in policy["errors"]
    )


def test_render_operator_execution_plan_accepts_native_weave_content_canary_gate(tmp_path):
    source_packet = write_source_packet(tmp_path / "external_action_approval_packet.json")
    approval_report = write_external_action_approval_report(
        tmp_path / "approval.verify.json",
        source_packet,
    )
    gate_json = write_native_weave_content_canary_gate(
        tmp_path / "native_content_canary.gate.json",
    )
    operator_plan = write_agentic_batch_operator_plan(
        tmp_path / "operator_plan.json",
        source_packet=source_packet,
        approval_report=approval_report,
        gate_json=gate_json,
    )
    output_json = tmp_path / "execution_plan.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--operator-plan-json",
            str(operator_plan),
            "--gate",
            "paid_run_review_package",
            "--external-action-approval-source-packet-json",
            str(source_packet),
            "--external-action-approval-report-json",
            str(approval_report),
            "--output-json",
            str(output_json),
            "--require-ready",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    rendered = json.loads(output_json.read_text(encoding="utf-8"))
    assert rendered["all_ready_to_execute_without_placeholder"] is True
    assert rendered["all_ready_for_external_execution"] is True
    assert rendered["command_policy"]["valid"] is True
    assert rendered["command_policy"]["error_count"] == 0


def test_render_operator_execution_plan_rejects_agentic_batch_missing_nemoclaw_config_path(
    tmp_path,
):
    source_packet = write_source_packet(tmp_path / "external_action_approval_packet.json")
    approval_report = write_external_action_approval_report(
        tmp_path / "approval.verify.json",
        source_packet,
    )
    gate_json = write_native_weave_content_canary_gate(
        tmp_path / "native_content_canary.gate.json",
    )
    operator_plan = write_agentic_batch_operator_plan(
        tmp_path / "operator_plan.json",
        source_packet=source_packet,
        approval_report=approval_report,
        gate_json=gate_json,
    )
    payload = json.loads(operator_plan.read_text(encoding="utf-8"))
    command = payload["operator_next_steps"]["steps"][0]["commands"][0]
    payload["operator_next_steps"]["steps"][0]["commands"][0] = command.replace(
        " --agentic-math-nemoclaw-openclaw-config-path /sandbox/.openclaw/openclaw.json",
        "",
    )
    operator_plan.write_text(json.dumps(payload), encoding="utf-8")
    output_json = tmp_path / "execution_plan.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--operator-plan-json",
            str(operator_plan),
            "--gate",
            "paid_run_review_package",
            "--external-action-approval-source-packet-json",
            str(source_packet),
            "--external-action-approval-report-json",
            str(approval_report),
            "--output-json",
            str(output_json),
            "--require-ready",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    rendered = json.loads(output_json.read_text(encoding="utf-8"))
    policy = rendered["command_policy"]
    assert policy["valid"] is False
    assert any(
        "run_taiwan_full_eval_batch.py agentic command is missing "
        "--agentic-math-nemoclaw-openclaw-config-path"
        in error
        for error in policy["errors"]
    )


def test_render_operator_execution_plan_rejects_agentic_batch_wrong_nemoclaw_config_path(
    tmp_path,
):
    source_packet = write_source_packet(tmp_path / "external_action_approval_packet.json")
    approval_report = write_external_action_approval_report(
        tmp_path / "approval.verify.json",
        source_packet,
    )
    gate_json = write_native_weave_content_canary_gate(
        tmp_path / "native_content_canary.gate.json",
    )
    operator_plan = write_agentic_batch_operator_plan(
        tmp_path / "operator_plan.json",
        source_packet=source_packet,
        approval_report=approval_report,
        gate_json=gate_json,
    )
    payload = json.loads(operator_plan.read_text(encoding="utf-8"))
    command = payload["operator_next_steps"]["steps"][0]["commands"][0]
    payload["operator_next_steps"]["steps"][0]["commands"][0] = command.replace(
        "--swebench-pro-nemoclaw-openclaw-config-path /sandbox/.openclaw/openclaw.json",
        "--swebench-pro-nemoclaw-openclaw-config-path /tmp/openclaw.json",
    )
    operator_plan.write_text(json.dumps(payload), encoding="utf-8")
    output_json = tmp_path / "execution_plan.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--operator-plan-json",
            str(operator_plan),
            "--gate",
            "paid_run_review_package",
            "--external-action-approval-source-packet-json",
            str(source_packet),
            "--external-action-approval-report-json",
            str(approval_report),
            "--output-json",
            str(output_json),
            "--require-ready",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    rendered = json.loads(output_json.read_text(encoding="utf-8"))
    policy = rendered["command_policy"]
    assert policy["valid"] is False
    assert any(
        "run_taiwan_full_eval_batch.py agentic command "
        "--swebench-pro-nemoclaw-openclaw-config-path must be "
        "/sandbox/.openclaw/openclaw.json"
        in error
        for error in policy["errors"]
    )


def test_render_operator_execution_plan_rejects_weave_canary_without_nemoclaw(tmp_path):
    source_packet = write_source_packet(tmp_path / "external_action_approval_packet.json")
    approval_report = write_external_action_approval_report(
        tmp_path / "approval.verify.json",
        source_packet,
    )
    gate_json = write_native_weave_content_canary_gate(
        tmp_path / "native_content_canary.gate.json",
    )
    operator_plan = write_agentic_batch_operator_plan(
        tmp_path / "operator_plan.json",
        source_packet=source_packet,
        approval_report=approval_report,
        gate_json=gate_json,
    )
    payload = json.loads(operator_plan.read_text(encoding="utf-8"))
    payload["operator_next_steps"]["steps"].append(
        {
            "order": 99,
            "gate": "weave_content_canary",
            "status": "failed",
            "commands": [
                (
                    "uv run python scripts/tools/run_weave_agents_content_canary.py "
                    "--execute --canary-id CONTENT_CANARY_TEST "
                    "--model openai-direct/gpt-4.1-nano-2025-04-14 "
                    "--thinking off --timeout 180 "
                    "--nemoclaw-sandbox nejumi-taiwan "
                    f"--external-action-approval-source-packet-json {source_packet} "
                    f"--external-action-approval-report-json {approval_report}"
                )
            ],
            "evidence_to_produce": [str(gate_json)],
        }
    )
    replaced = False
    for step in payload["operator_next_steps"]["steps"]:
        for index, command in enumerate(step.get("commands") or []):
            if "run_weave_agents_content_canary.py --execute" not in command:
                continue
            step["commands"][index] = command.replace(
                " --nemoclaw-sandbox nejumi-taiwan",
                "",
            )
            replaced = True
    assert replaced
    operator_plan.write_text(json.dumps(payload), encoding="utf-8")
    output_json = tmp_path / "execution_plan.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--operator-plan-json",
            str(operator_plan),
            "--gate",
            "weave_content_canary",
            "--external-action-approval-source-packet-json",
            str(source_packet),
            "--external-action-approval-report-json",
            str(approval_report),
            "--weave-content-canary-gate",
            str(gate_json),
            "--output-json",
            str(output_json),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    rendered = json.loads(output_json.read_text(encoding="utf-8"))
    assert rendered["command_policy"]["valid"] is False
    assert any(
        "run_weave_agents_content_canary.py --execute is missing --nemoclaw-sandbox"
        in error
        for error in rendered["command_policy"]["errors"]
    )
