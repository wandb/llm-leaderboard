import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_SHA = "a" * 64
AGENTIC_MATH_OUTPUT_COLUMNS = [
    "nemoclaw_session_audit_ok",
    "nemoclaw_session_audit",
    "nemoclaw_session_copy_source",
    "nemoclaw_session_copied_bytes",
    "conversation_order_ok",
    "conversation_order",
    "tool_policy_ok",
    "tool_policy_violations",
    "weave_agents_ok",
    "weave_agents_required",
    "weave_agents_agent_name",
    "weave_agents_conversation_id",
    "weave_agents_conversation_id_contains",
    "weave_agents_conversation_url",
    "weave_agents_trace_id",
    "weave_agents_url",
    "weave_agents_trace_url",
    "weave_agents_verifier_json",
    "weave_agents_error",
    "openclaw_result_path",
    "openclaw_invocation_path",
    "openclaw_invocation_sha256",
    "openclaw_command_sha256",
    "openclaw_config_source",
]
AGENTIC_SWE_OUTPUT_COLUMNS = [
    "nemoclaw_session_audit_ok",
    "nemoclaw_session_audit_required",
    "nemoclaw_session_copy_source",
    "nemoclaw_session_copied_bytes",
    "conversation_order_ok",
    "conversation_order",
    "tool_policy_ok",
    "tool_policy_violations",
    "weave_agents_ok",
    "weave_agents_required",
    "weave_agents_agent_name",
    "weave_agents_conversation_id",
    "weave_agents_conversation_id_contains",
    "weave_agents_conversation_url",
    "weave_agents_trace_id",
    "weave_agents_url",
    "weave_agents_trace_url",
    "weave_agents_verifier_json",
    "weave_agents_error",
    "openclaw_result_path",
    "openclaw_invocation_path",
    "openclaw_invocation_sha256",
    "openclaw_command_sha256",
    "openclaw_config_source",
]
AGENTIC_SWE_ASSORTED_OUTPUT_COLUMNS = [
    "source_benchmark",
    "source_dataset",
    "source_subset",
    "source_instance_id",
    "agentic_swe_tier",
    "instance_id",
    "resolved",
    "score",
    "weave_agents_conversation_url",
    "openclaw_tool_call_count",
    "openclaw_usage",
]


def table_value(column, row_index):
    if column.endswith("_ok"):
        return True
    if column in {"tool_policy_violations"}:
        return []
    if column in {"conversation_order", "nemoclaw_session_audit"}:
        return {"ok": True, "required": True}
    if column == "weave_agents_required":
        return True
    if column == "weave_agents_agent_name":
        return "nejumi-taiwan-openclaw"
    if column == "weave_agents_conversation_id":
        return f"agent:main:run123:agentic:{row_index}"
    if column == "weave_agents_conversation_id_contains":
        return f"run123:agentic:{row_index}"
    if column == "weave_agents_conversation_url":
        return (
            "https://wandb.ai/llm-leaderboard/tc-leaderboard/weave/agents/"
            f"conversations/agent%3Amain%3Arun123%3Aagentic%3A{row_index}"
        )
    if column == "weave_agents_trace_id":
        return f"trace-{row_index}"
    if column == "weave_agents_url":
        return "https://wandb.ai/llm-leaderboard/tc-leaderboard/weave/agents"
    if column == "weave_agents_trace_url":
        return f"https://wandb.ai/llm-leaderboard/tc-leaderboard/weave/agents?trace_id=trace-{row_index}"
    if column == "weave_agents_verifier_json":
        return f"/tmp/task-{row_index}/weave_agents_verification.json"
    if column == "weave_agents_error":
        return ""
    if column == "nemoclaw_session_audit_required":
        return True
    if column == "nemoclaw_session_copy_source":
        return "stdout_agent_meta"
    if column == "nemoclaw_session_copied_bytes":
        return 1024
    if column == "openclaw_result_path":
        return f"/tmp/task-{row_index}/openclaw_result.json"
    if column == "openclaw_invocation_path":
        return f"/tmp/task-{row_index}/openclaw_invocation.json"
    if column == "openclaw_invocation_sha256":
        return TEST_SHA
    if column == "openclaw_command_sha256":
        return "b" * 64
    if column == "openclaw_config_source":
        return "/sandbox/.openclaw/openclaw.json"
    return f"value-{row_index}"


def table_payload(columns, nrows):
    return {
        "columns": list(columns),
        "data": [
            [table_value(column, row_index) for column in columns]
            for row_index in range(1, nrows + 1)
        ],
    }


def assorted_table_payload(tier_counts):
    columns = AGENTIC_SWE_OUTPUT_COLUMNS + AGENTIC_SWE_ASSORTED_OUTPUT_COLUMNS
    rows = []
    row_index = 1
    for tier, count in tier_counts.items():
        for tier_index in range(1, count + 1):
            row = {column: table_value(column, row_index) for column in columns}
            row["agentic_swe_tier"] = tier
            row["source_benchmark"] = "DeepSWE" if tier == "high" else "SWE-bench Lite"
            row["source_dataset"] = "deep-swe" if tier == "high" else "swe-bench-lite"
            row["source_subset"] = "high-8" if tier == "high" else tier
            row["source_instance_id"] = f"{tier}-{tier_index}"
            row["instance_id"] = f"{tier}-{tier_index}"
            row["resolved"] = tier_index % 2 == 0
            row["score"] = 1 if row["resolved"] else 0
            row["openclaw_tool_call_count"] = tier_index
            row["openclaw_usage"] = {"input_tokens": 10, "output_tokens": 5}
            rows.append([row[column] for column in columns])
            row_index += 1
    return {"columns": columns, "data": rows}


class FakeArtifact:
    def __init__(self, name, type_, aliases, manifest_entries=None):
        self.name = name
        self.type = type_
        self.aliases = aliases
        if manifest_entries is not None:
            class Manifest:
                pass

            self.manifest = Manifest()
            self.manifest.entries = {entry: object() for entry in manifest_entries}


class FakeWandbFile:
    def __init__(self, path, payload):
        self.path = path
        self.payload = payload

    def download(self, root, replace=True):
        target = Path(root) / self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.payload), encoding="utf-8")

        class Downloaded:
            pass

        downloaded = Downloaded()
        downloaded.name = str(target)
        return downloaded


class FakeRun:
    id = "run123"
    name = "fake-run"

    def __init__(
        self,
        *,
        summary,
        state="finished",
        artifacts=None,
        config=None,
        tags=None,
        group="",
        job_type="evaluation",
        table_files=None,
    ):
        self.summary_metrics = summary
        self.state = state
        self._artifacts = artifacts or []
        self.config = config or {}
        self.tags = tags or []
        self.group = group
        self.job_type = job_type
        self._table_files = table_files if table_files is not None else self._default_table_files()

    def _default_table_files(self):
        table_files = {}
        for value in self.summary_metrics.values():
            if not isinstance(value, dict):
                continue
            path = value.get("path")
            columns = value.get("columns")
            nrows = value.get("nrows")
            if isinstance(path, str) and isinstance(columns, list) and isinstance(nrows, int):
                table_files[path] = table_payload(columns, nrows)
        return table_files

    def logged_artifacts(self):
        return self._artifacts

    def file(self, path):
        if path not in self._table_files:
            raise FileNotFoundError(path)
        return FakeWandbFile(path, self._table_files[path])


def load_module():
    path = REPO_ROOT / "scripts" / "tools" / "verify_taiwan_wandb_completion.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def complete_agentic_math_summary():
    return {
        "agentic_math/total_instances": 100,
        "agentic_math/answered_instances": 99,
        "agentic_math/correct_instances": 86,
        "agentic_math/accuracy": 0.86,
        "agentic_math/nemoclaw_session_audit_required_instances": 100,
        "agentic_math/nemoclaw_session_audit_passed_instances": 100,
        "agentic_math/nemoclaw_session_audit_failed_instances": 0,
        "agentic_math_leaderboard_table": {"_type": "table-file", "nrows": 1},
        "agentic_math_output_table": {
            "_type": "table-file",
            "nrows": 100,
            "path": "media/table/agentic_math_output_table_0.table.json",
            "columns": AGENTIC_MATH_OUTPUT_COLUMNS,
        },
    }


def complete_result_artifact():
    return FakeArtifact(
        "agentic-math-olymmath-hard-zh-tw-model-results:v0",
        "evaluation-results",
        ["latest", "production"],
    )


def complete_agentic_swe_summary():
    return {
        "agentic_swe/total_instances": 80,
        "agentic_swe/resolved_instances": 24,
        "agentic_swe/unresolved_instances": 56,
        "agentic_swe/pass_at_1": 0.3,
        "agentic_swe/nemoclaw_session_audit_required_patches": 80,
        "agentic_swe/nemoclaw_session_audit_passed_patches": 80,
        "agentic_swe/nemoclaw_session_audit_failed_patches": 0,
        "agentic_swe_leaderboard_table": {"_type": "table-file", "nrows": 1},
        "agentic_swe_output_table": {
            "_type": "table-file",
            "nrows": 80,
            "path": "media/table/agentic_swe_output_table_0.table.json",
            "columns": AGENTIC_SWE_OUTPUT_COLUMNS,
        },
    }


def complete_swe_result_artifact():
    return FakeArtifact(
        "agentic-swe-swebench-pro-model-results:v0",
        "evaluation-results",
        ["latest", "production"],
    )


def complete_swe_assorted_result_artifact():
    return FakeArtifact(
        "agentic-swe-assorted-model-results:v0",
        "evaluation-results",
        ["latest", "production"],
        manifest_entries=[
            "summary.json",
            "output_table.jsonl",
            "leaderboard_table.json",
            "report.md",
        ],
    )


def complete_agentic_swe_assorted_summary():
    summary = complete_agentic_swe_summary()
    summary.update(
        {
            "agentic_swe/resolved_instances": 40,
            "agentic_swe/unresolved_instances": 40,
            "agentic_swe/pass_at_1": 0.7,
            "agentic_swe/weighted_pass_at_1": 0.7,
            "agentic_swe/weighted_present_pass_at_1": 0.7,
            "agentic_swe/weighted_pass_at_1_complete": 1,
            "agentic_swe/low/pass_at_1": 0.9,
            "agentic_swe/middle/pass_at_1": 0.8,
            "agentic_swe/high/pass_at_1": 0.4,
            "agentic_swe_output_table": {
                "_type": "table-file",
                "nrows": 80,
                "path": "media/table/agentic_swe_output_table_0.table.json",
                "columns": AGENTIC_SWE_OUTPUT_COLUMNS + AGENTIC_SWE_ASSORTED_OUTPUT_COLUMNS,
            },
        }
    )
    return summary


def complete_taiwan_full_summary():
    return {
        "GLP": 34.5,
        "ALT": 67.0,
        "Overall": 48.4285714286,
        "taiwan_missing_required_count": 0,
        "taiwan_pending_count": 1,
        "mtbench_leaderboard_table": {"_type": "table-file", "nrows": 1},
        "arc_agi_leaderboard_table": {"_type": "table-file", "nrows": 1},
        "agentic_math_leaderboard_table": {"_type": "table-file", "nrows": 1},
        "jaster_0shot_leaderboard_table": {"_type": "table-file", "nrows": 1},
        "jaster_2shot_leaderboard_table": {"_type": "table-file", "nrows": 1},
        "hle_test_leaderboard_table": {"_type": "table-file", "nrows": 1},
        "tceval_v2_selected_leaderboard_table": {"_type": "table-file", "nrows": 1},
        "agentic_swe_leaderboard_table": {"_type": "table-file", "nrows": 1},
        "bfcl_leaderboard_table": {"_type": "table-file", "nrows": 1},
        "ifeval_zh_tw_leaderboard_table": {"_type": "table-file", "nrows": 1},
        "ts_bench_leaderboard_table": {"_type": "table-file", "nrows": 1},
        "hallulens_zh_tw_leaderboard_table": {"_type": "table-file", "nrows": 1},
        "tmmluplus_robust_2shot_leaderboard_table": {"_type": "table-file", "nrows": 1},
        "traditional_chinese_script_adherence_leaderboard_table": {"_type": "table-file", "nrows": 1},
        "taiwan_leaderboard_table": {"_type": "table-file", "nrows": 1},
        "taiwan_unit_scores_table": {"_type": "table-file", "nrows": 14},
        "taiwan_glp_radar_table": {"_type": "table-file", "nrows": 8},
        "taiwan_alt_radar_table": {"_type": "table-file", "nrows": 6},
        "bfcl_timeout_count": 0,
        "bfcl_inference_error_count": 0,
        "bfcl_profile": "full",
        "bfcl_profile_case_count": 496,
    }


def complete_bfcl_summary():
    return {
        "bfcl_runtime_case_count": 12,
        "bfcl_profile_case_count": 12,
        "bfcl_timeout_count": 0,
        "bfcl_inference_error_count": 0,
        "bfcl_version": "v4",
        "bfcl_profile": "core",
        "bfcl_upstream_commit": "a" * 40,
        "bfcl_leaderboard_table": {
            "_type": "table-file",
            "nrows": 1,
        },
        "bfcl_output_table": {
            "_type": "table-file",
            "nrows": 12,
            "path": "media/table/bfcl_output.table.json",
            "columns": [
                "model",
                "id",
                "category",
                "prompt",
                "output",
                "accuracy",
                "possible_answer",
                "reasoning_content",
                "input_token_count",
                "output_token_count",
                "timeout",
                "error",
            ],
        },
    }


def test_verify_bfcl_v4_canary_accepts_complete_zero_error_run():
    module = load_module()
    run = FakeRun(summary=complete_bfcl_summary())

    result = module.verify_run(
        run,
        module.BENCHMARK_SPECS["bfcl"],
        expected_total=12,
    )

    assert result["ok"] is True
    output_table_evidence = result["required_evidence"]["tables"][1]
    assert "row_observability" not in output_table_evidence


def test_verify_bfcl_v4_accepts_runtime_prerequisite_rows_beyond_logical_total():
    module = load_module()
    summary = complete_bfcl_summary()
    summary["bfcl_profile_case_count"] = 12
    summary["bfcl_runtime_case_count"] = 15
    summary["bfcl_output_table"]["nrows"] = 15
    run = FakeRun(summary=summary)

    result = module.verify_run(
        run,
        module.BENCHMARK_SPECS["bfcl"],
        expected_total=12,
    )

    assert result["ok"] is True
    assert result["required_evidence"]["summary_metrics"] == [
        "bfcl_profile_case_count",
        "bfcl_runtime_case_count",
    ]
    assert any(
        check["name"] == "output_total_metric"
        and check["value"] == 15
        and check["ok"]
        for check in result["checks"]
    )
    assert any(
        check["name"] == "output_table"
        and check["nrows"] == 15
        and check["ok"]
        for check in result["checks"]
    )


def test_verify_bfcl_v4_canary_rejects_runtime_errors():
    module = load_module()
    summary = complete_bfcl_summary()
    summary["bfcl_inference_error_count"] = 1
    run = FakeRun(summary=summary)

    result = module.verify_run(
        run,
        module.BENCHMARK_SPECS["bfcl"],
        expected_total=12,
    )

    assert result["ok"] is False
    assert any(
        check["name"] == "bfcl_runtime_error_metric"
        and check.get("metric") == "bfcl_inference_error_count"
        and not check["ok"]
        for check in result["checks"]
    )


def test_verify_bfcl_v4_allows_timeouts_as_scored_model_outcomes():
    module = load_module()
    summary = complete_bfcl_summary()
    summary["bfcl_timeout_count"] = 3
    run = FakeRun(summary=summary)

    result = module.verify_run(
        run,
        module.BENCHMARK_SPECS["bfcl"],
        expected_total=12,
    )

    assert result["ok"] is True
    assert any(
        check["name"] == "bfcl_runtime_error_metric"
        and check.get("metric") == "bfcl_timeout_count"
        and check.get("value") == 3
        and check["ok"]
        for check in result["checks"]
    )


def test_verify_agentic_math_wandb_completion_accepts_complete_run():
    module = load_module()
    run = FakeRun(
        summary=complete_agentic_math_summary(),
        artifacts=[complete_result_artifact()],
    )

    result = module.verify_run(
        run,
        module.BENCHMARK_SPECS["agentic_math"],
        expected_total=100,
    )

    assert result["ok"] is True
    assert result["schema_version"] == module.VERIFICATION_SCHEMA_VERSION
    assert result["verification_schema_version"] == module.VERIFICATION_SCHEMA_VERSION
    assert result["status"] == "passed"
    assert result["required_evidence"]["expected_total"] == 100
    assert "agentic_math/accuracy" in result["required_evidence"]["summary_metrics"]
    assert result["required_evidence"]["artifacts"][0]["required_aliases"] == ["production"]
    assert result["observed_evidence"]["run_state"] == "finished"
    assert result["observed_evidence"]["summary_metrics"]["agentic_math/total_instances"]["value"] == 100
    assert result["observed_evidence"]["summary_metrics"]["agentic_math/accuracy"]["value"] == 0.86
    assert result["observed_evidence"]["tables"] == [
        {
            "name": "agentic_math_leaderboard_table",
            "ok": True,
            "nrows": 1,
            "expected": None,
        },
        {
            "name": "agentic_math_output_table",
            "ok": True,
            "nrows": 100,
            "expected": None,
            "columns_ok": True,
            "columns": AGENTIC_MATH_OUTPUT_COLUMNS,
            "required_columns": AGENTIC_MATH_OUTPUT_COLUMNS,
            "missing_columns": [],
            "columns_source": "summary",
            "invocation_evidence_ok": True,
            "invocation_evidence_source": "wandb_file",
            "invocation_checked_rows": 100,
            "invocation_expected_rows": 100,
            "invocation_invalid_row_count": 0,
            "invocation_invalid_examples": [],
            "openclaw_config_source_ok": True,
            "openclaw_config_source_source": "wandb_file",
            "openclaw_config_source_checked_rows": 100,
            "openclaw_config_source_expected_rows": 100,
            "openclaw_config_source_expected": "/sandbox/.openclaw/openclaw.json",
            "openclaw_config_source_invalid_row_count": 0,
            "openclaw_config_source_invalid_examples": [],
            "row_observability_ok": True,
            "row_observability_source": "wandb_file",
            "row_observability_checked_rows": 100,
            "row_observability_expected_rows": 100,
            "row_observability_invalid_row_count": 0,
            "row_observability_invalid_examples": [],
            "row_observability_required_true_columns": [
                "nemoclaw_session_audit_ok",
                "conversation_order_ok",
                "tool_policy_ok",
                "weave_agents_ok",
                "weave_agents_required",
            ],
            "row_observability_required_empty_list_columns": [
                "tool_policy_violations",
            ],
            "row_observability_required_dict_ok_columns": [
                "nemoclaw_session_audit",
                "conversation_order",
            ],
            "row_observability_required_nonempty_string_columns": [
                "weave_agents_agent_name",
                "weave_agents_conversation_id",
                "weave_agents_conversation_id_contains",
                "weave_agents_conversation_url",
                "weave_agents_trace_id",
                "weave_agents_url",
                "weave_agents_verifier_json",
            ],
            "row_observability_required_copy_source_columns": [
                "nemoclaw_session_copy_source",
            ],
            "row_observability_required_positive_int_columns": [
                "nemoclaw_session_copied_bytes",
            ],
            "row_observability_allowed_copy_sources": [
                "stdout_agent_meta",
                "live_runtime_budget",
            ],
        },
    ]
    assert result["observed_evidence"]["artifacts"][0]["aliases"] == ["latest", "production"]
    assert {check["name"] for check in result["checks"]} >= {
        "run_state",
        "total_metric",
        "leaderboard_table",
        "output_table",
        "output_table_columns",
        "output_table_invocation_evidence",
        "output_table_openclaw_config_source",
        "output_table_row_observability",
        "answered_metric",
        "correct_metric",
        "accuracy_metric",
        "result_artifact",
    }


def test_verify_agentic_math_wandb_completion_rejects_missing_output_table():
    module = load_module()
    summary = complete_agentic_math_summary()
    del summary["agentic_math_output_table"]
    run = FakeRun(summary=summary, artifacts=[complete_result_artifact()])

    result = module.verify_run(run, module.BENCHMARK_SPECS["agentic_math"])

    assert result["ok"] is False
    assert result["schema_version"] == module.VERIFICATION_SCHEMA_VERSION
    assert result["verification_schema_version"] == module.VERIFICATION_SCHEMA_VERSION
    assert result["status"] == "failed"
    assert any(check["name"] == "output_table" and not check["ok"] for check in result["checks"])


def test_verify_agentic_math_wandb_completion_rejects_missing_output_observability_column():
    module = load_module()
    summary = complete_agentic_math_summary()
    summary["agentic_math_output_table"]["columns"] = [
        column
        for column in AGENTIC_MATH_OUTPUT_COLUMNS
        if column != "weave_agents_trace_id"
    ]
    run = FakeRun(summary=summary, artifacts=[complete_result_artifact()])

    result = module.verify_run(run, module.BENCHMARK_SPECS["agentic_math"])

    assert result["ok"] is False
    column_check = next(
        check for check in result["checks"] if check["name"] == "output_table_columns"
    )
    assert column_check["ok"] is False
    assert column_check["missing_columns"] == ["weave_agents_trace_id"]
    output_table = next(
        table
        for table in result["observed_evidence"]["tables"]
        if table["name"] == "agentic_math_output_table"
    )
    assert output_table["columns_ok"] is False
    assert output_table["missing_columns"] == ["weave_agents_trace_id"]


def test_verify_agentic_math_wandb_completion_loads_output_columns_from_table_file():
    module = load_module()
    summary = complete_agentic_math_summary()
    summary["agentic_math_output_table"] = {
        "_type": "table-file",
        "nrows": 100,
        "path": "media/table/agentic_math_output_table_0.table.json",
    }
    run = FakeRun(
        summary=summary,
        artifacts=[complete_result_artifact()],
        table_files={
            "media/table/agentic_math_output_table_0.table.json": table_payload(
                AGENTIC_MATH_OUTPUT_COLUMNS,
                100,
            )
        },
    )

    result = module.verify_run(run, module.BENCHMARK_SPECS["agentic_math"])

    assert result["ok"] is True
    column_check = next(
        check for check in result["checks"] if check["name"] == "output_table_columns"
    )
    assert column_check["source"] == "wandb_file"
    output_table = next(
        table
        for table in result["observed_evidence"]["tables"]
        if table["name"] == "agentic_math_output_table"
    )
    assert output_table["columns_source"] == "wandb_file"


def test_verify_agentic_math_wandb_completion_rejects_missing_invocation_row_evidence():
    module = load_module()
    summary = complete_agentic_math_summary()
    payload = table_payload(AGENTIC_MATH_OUTPUT_COLUMNS, 100)
    column_index = AGENTIC_MATH_OUTPUT_COLUMNS.index("openclaw_invocation_path")
    payload["data"][0][column_index] = ""
    run = FakeRun(
        summary=summary,
        artifacts=[complete_result_artifact()],
        table_files={
            "media/table/agentic_math_output_table_0.table.json": payload,
        },
    )

    result = module.verify_run(run, module.BENCHMARK_SPECS["agentic_math"])

    assert result["ok"] is False
    check = next(
        check
        for check in result["checks"]
        if check["name"] == "output_table_invocation_evidence"
    )
    assert check["ok"] is False
    assert check["invalid_row_count"] == 1
    assert check["invalid_examples"][0]["missing_or_empty"] == [
        "openclaw_invocation_path"
    ]


def test_verify_agentic_swe_wandb_completion_rejects_invalid_invocation_hash():
    module = load_module()
    summary = complete_agentic_swe_summary()
    payload = table_payload(AGENTIC_SWE_OUTPUT_COLUMNS, 80)
    column_index = AGENTIC_SWE_OUTPUT_COLUMNS.index("openclaw_command_sha256")
    payload["data"][0][column_index] = "not-a-sha"
    run = FakeRun(
        summary=summary,
        artifacts=[complete_swe_result_artifact()],
        table_files={
            "media/table/agentic_swe_output_table_0.table.json": payload,
        },
    )

    result = module.verify_run(run, module.BENCHMARK_SPECS["agentic_swe"])

    assert result["ok"] is False
    check = next(
        check
        for check in result["checks"]
        if check["name"] == "output_table_invocation_evidence"
    )
    assert check["ok"] is False
    assert check["invalid_row_count"] == 1
    assert check["invalid_examples"][0]["invalid_hashes"] == [
        "openclaw_command_sha256"
    ]


def test_verify_agentic_math_wandb_completion_rejects_wrong_openclaw_config_source():
    module = load_module()
    summary = complete_agentic_math_summary()
    payload = table_payload(AGENTIC_MATH_OUTPUT_COLUMNS, 100)
    column_index = AGENTIC_MATH_OUTPUT_COLUMNS.index("openclaw_config_source")
    payload["data"][0][column_index] = "/sandbox/other-openclaw.json"
    run = FakeRun(
        summary=summary,
        artifacts=[complete_result_artifact()],
        table_files={
            "media/table/agentic_math_output_table_0.table.json": payload,
        },
    )

    result = module.verify_run(run, module.BENCHMARK_SPECS["agentic_math"])

    assert result["ok"] is False
    check = next(
        check
        for check in result["checks"]
        if check["name"] == "output_table_openclaw_config_source"
    )
    assert check["ok"] is False
    assert check["expected_config_source"] == "/sandbox/.openclaw/openclaw.json"
    assert check["invalid_row_count"] == 1
    assert check["invalid_examples"][0]["openclaw_config_source"] == "/sandbox/other-openclaw.json"


def test_verify_agentic_math_wandb_completion_rejects_failed_row_observability():
    module = load_module()
    summary = complete_agentic_math_summary()
    payload = table_payload(AGENTIC_MATH_OUTPUT_COLUMNS, 100)
    tool_ok_index = AGENTIC_MATH_OUTPUT_COLUMNS.index("tool_policy_ok")
    violations_index = AGENTIC_MATH_OUTPUT_COLUMNS.index("tool_policy_violations")
    payload["data"][0][tool_ok_index] = False
    payload["data"][0][violations_index] = [{"type": "denied_tool"}]
    run = FakeRun(
        summary=summary,
        artifacts=[complete_result_artifact()],
        table_files={
            "media/table/agentic_math_output_table_0.table.json": payload,
        },
    )

    result = module.verify_run(run, module.BENCHMARK_SPECS["agentic_math"])

    assert result["ok"] is False
    check = next(
        check
        for check in result["checks"]
        if check["name"] == "output_table_row_observability"
    )
    assert check["ok"] is False
    assert check["invalid_row_count"] == 1
    assert check["invalid_examples"][0]["not_true"] == ["tool_policy_ok"]
    assert check["invalid_examples"][0]["non_empty_lists"] == [
        "tool_policy_violations"
    ]
    output_table = next(
        table
        for table in result["observed_evidence"]["tables"]
        if table["name"] == "agentic_math_output_table"
    )
    assert output_table["row_observability_ok"] is False
    assert output_table["row_observability_invalid_row_count"] == 1


def test_verify_agentic_swe_wandb_completion_rejects_failed_nested_observability():
    module = load_module()
    summary = complete_agentic_swe_summary()
    payload = table_payload(AGENTIC_SWE_OUTPUT_COLUMNS, 80)
    conversation_index = AGENTIC_SWE_OUTPUT_COLUMNS.index("conversation_order")
    payload["data"][0][conversation_index] = {"ok": False, "reason": "bad order"}
    run = FakeRun(
        summary=summary,
        artifacts=[complete_swe_result_artifact()],
        table_files={
            "media/table/agentic_swe_output_table_0.table.json": payload,
        },
    )

    result = module.verify_run(run, module.BENCHMARK_SPECS["agentic_swe"])

    assert result["ok"] is False
    check = next(
        check
        for check in result["checks"]
        if check["name"] == "output_table_row_observability"
    )
    assert check["ok"] is False
    assert check["invalid_row_count"] == 1
    assert check["invalid_examples"][0]["dict_not_ok"] == ["conversation_order"]


def test_verify_agentic_math_wandb_completion_accepts_json_string_observability():
    module = load_module()
    summary = complete_agentic_math_summary()
    payload = table_payload(AGENTIC_MATH_OUTPUT_COLUMNS, 100)
    conversation_index = AGENTIC_MATH_OUTPUT_COLUMNS.index("conversation_order")
    audit_index = AGENTIC_MATH_OUTPUT_COLUMNS.index("nemoclaw_session_audit")
    violations_index = AGENTIC_MATH_OUTPUT_COLUMNS.index("tool_policy_violations")
    payload["data"][0][conversation_index] = json.dumps(
        {"ok": True, "first_final_answer_index": 6}
    )
    payload["data"][0][audit_index] = json.dumps({"ok": True, "required": True})
    payload["data"][0][violations_index] = json.dumps([])
    run = FakeRun(
        summary=summary,
        artifacts=[complete_result_artifact()],
        table_files={
            "media/table/agentic_math_output_table_0.table.json": payload,
        },
    )

    result = module.verify_run(run, module.BENCHMARK_SPECS["agentic_math"])

    check = next(
        check
        for check in result["checks"]
        if check["name"] == "output_table_row_observability"
    )
    assert check["ok"] is True


def test_verify_agentic_math_wandb_completion_rejects_invalid_session_copy_evidence():
    module = load_module()
    summary = complete_agentic_math_summary()
    payload = table_payload(AGENTIC_MATH_OUTPUT_COLUMNS, 100)
    source_index = AGENTIC_MATH_OUTPUT_COLUMNS.index("nemoclaw_session_copy_source")
    bytes_index = AGENTIC_MATH_OUTPUT_COLUMNS.index("nemoclaw_session_copied_bytes")
    payload["data"][0][source_index] = "manual_json_transform"
    payload["data"][0][bytes_index] = 0
    run = FakeRun(
        summary=summary,
        artifacts=[complete_result_artifact()],
        table_files={
            "media/table/agentic_math_output_table_0.table.json": payload,
        },
    )

    result = module.verify_run(run, module.BENCHMARK_SPECS["agentic_math"])

    assert result["ok"] is False
    check = next(
        check
        for check in result["checks"]
        if check["name"] == "output_table_row_observability"
    )
    assert check["ok"] is False
    assert check["invalid_row_count"] == 1
    assert check["invalid_examples"][0]["invalid_copy_sources"] == [
        "nemoclaw_session_copy_source"
    ]
    assert check["invalid_examples"][0]["invalid_positive_ints"] == [
        "nemoclaw_session_copied_bytes"
    ]


def test_verify_agentic_math_wandb_completion_requires_nemoclaw_session_audit():
    module = load_module()
    run = FakeRun(
        summary=complete_agentic_math_summary(),
        artifacts=[complete_result_artifact()],
    )

    result = module.verify_run(
        run,
        module.BENCHMARK_SPECS["agentic_math"],
        expected_total=100,
        require_nemoclaw_session_audit=True,
    )

    assert result["ok"] is True
    assert result["required_evidence"]["nemoclaw_session_audit"]["required"] is True
    assert result["observed_evidence"]["nemoclaw_session_audit"] == {
        "ok": True,
        "required": 100,
        "passed": 100,
        "failed": 0,
        "expected_total": 100,
        "required_metric": "agentic_math/nemoclaw_session_audit_required_instances",
        "passed_metric": "agentic_math/nemoclaw_session_audit_passed_instances",
        "failed_metric": "agentic_math/nemoclaw_session_audit_failed_instances",
    }


def test_verify_agentic_math_wandb_completion_rejects_failed_nemoclaw_session_audit():
    module = load_module()
    summary = complete_agentic_math_summary()
    summary["agentic_math/nemoclaw_session_audit_passed_instances"] = 99
    summary["agentic_math/nemoclaw_session_audit_failed_instances"] = 1
    run = FakeRun(summary=summary, artifacts=[complete_result_artifact()])

    result = module.verify_run(
        run,
        module.BENCHMARK_SPECS["agentic_math"],
        expected_total=100,
        require_nemoclaw_session_audit=True,
    )

    assert result["ok"] is False
    assert any(
        check["name"] == "nemoclaw_session_audit" and not check["ok"]
        for check in result["checks"]
    )


def test_verify_agentic_math_wandb_completion_rejects_accuracy_mismatch():
    module = load_module()
    summary = complete_agentic_math_summary()
    summary["agentic_math/accuracy"] = 0.87
    run = FakeRun(summary=summary, artifacts=[complete_result_artifact()])

    result = module.verify_run(run, module.BENCHMARK_SPECS["agentic_math"])

    assert result["ok"] is False
    assert any(
        check["name"] == "accuracy_metric" and not check["ok"]
        for check in result["checks"]
    )


def test_verify_agentic_math_wandb_completion_rejects_missing_artifact():
    module = load_module()
    run = FakeRun(summary=complete_agentic_math_summary(), artifacts=[])

    result = module.verify_run(run, module.BENCHMARK_SPECS["agentic_math"])

    assert result["ok"] is False
    assert any(
        check["name"] == "result_artifact" and not check["ok"]
        for check in result["checks"]
    )


def test_verify_taiwan_full_completion_accepts_taxonomy_tables_and_aggregate():
    module = load_module()
    run = FakeRun(summary=complete_taiwan_full_summary())

    result = module.verify_full_taiwan_run(run, taxonomy_path=REPO_ROOT / "taxonomies" / "nejumi45_taiwan.yaml")

    assert result["ok"] is True
    assert result["benchmark"] == "taiwan_full"
    assert result["schema_version"] == module.VERIFICATION_SCHEMA_VERSION
    assert result["verification_schema_version"] == module.VERIFICATION_SCHEMA_VERSION
    assert result["status"] == "passed"
    assert result["required_evidence"]["taxonomy_path"].endswith("nejumi45_taiwan.yaml")
    assert result["required_evidence"]["aggregate_tables"]
    assert any(
        table["unit_id"] == "agentic_swe"
        for table in result["required_evidence"]["taxonomy_tables"]
    )
    assert any(
        check["name"] == "taxonomy_unit_pending_skipped"
        and check["unit_id"] == "twbias"
        for check in result["checks"]
    )
    assert any(
        check["name"] == "aggregate_table"
        and check["table_name"] == "taiwan_leaderboard_table"
        for check in result["checks"]
    )
    assert result["observed_evidence"]["run_state"] == "finished"
    assert any(
        table["unit_id"] == "agentic_swe"
        and table["table_name"] == "agentic_swe_leaderboard_table"
        and table["nrows"] == 1
        for table in result["observed_evidence"]["taxonomy_tables"]
    )
    assert any(
        table["name"] == "taiwan_leaderboard_table" and table["nrows"] == 1
        for table in result["observed_evidence"]["aggregate_tables"]
    )
    assert any(
        unit["unit_id"] == "twbias"
        for unit in result["observed_evidence"]["skipped_pending_units"]
    )


def test_verify_taiwan_full_completion_rejects_missing_required_table():
    module = load_module()
    summary = complete_taiwan_full_summary()
    del summary["agentic_swe_leaderboard_table"]
    run = FakeRun(summary=summary)

    result = module.verify_full_taiwan_run(run, taxonomy_path=REPO_ROOT / "taxonomies" / "nejumi45_taiwan.yaml")

    assert result["ok"] is False
    assert any(
        check["name"] == "taxonomy_table"
        and check["unit_id"] == "agentic_swe"
        and not check["ok"]
        for check in result["checks"]
    )


def test_verify_taiwan_full_completion_rejects_missing_aggregate_scalar():
    module = load_module()
    summary = complete_taiwan_full_summary()
    del summary["GLP"]
    run = FakeRun(summary=summary)

    result = module.verify_full_taiwan_run(
        run,
        taxonomy_path=REPO_ROOT / "taxonomies" / "nejumi45_taiwan.yaml",
    )

    assert result["ok"] is False
    assert any(
        check["name"] == "aggregate_score_metric"
        and check["metric"] == "GLP"
        and not check["ok"]
        for check in result["checks"]
    )


def test_verify_taiwan_full_completion_rejects_bfcl_core_profile():
    module = load_module()
    summary = complete_taiwan_full_summary()
    summary["bfcl_profile"] = "core"
    summary["bfcl_profile_case_count"] = 346
    run = FakeRun(summary=summary)

    result = module.verify_full_taiwan_run(
        run,
        taxonomy_path=REPO_ROOT / "taxonomies" / "nejumi45_taiwan.yaml",
    )

    assert result["ok"] is False
    assert any(
        check["name"] == "bfcl_release_profile" and not check["ok"]
        for check in result["checks"]
    )
    assert any(
        check["name"] == "bfcl_release_case_count" and not check["ok"]
        for check in result["checks"]
    )


def test_verify_taiwan_full_completion_allows_bfcl_timeout_count_as_scored_incorrect():
    module = load_module()
    summary = complete_taiwan_full_summary()
    summary["bfcl_timeout_count"] = 1
    run = FakeRun(summary=summary)

    result = module.verify_full_taiwan_run(
        run,
        taxonomy_path=REPO_ROOT / "taxonomies" / "nejumi45_taiwan.yaml",
    )

    assert result["ok"] is True
    assert any(
        check["name"] == "bfcl_runtime_error_metric"
        and check.get("metric") == "bfcl_timeout_count"
        and check["ok"]
        for check in result["checks"]
    )


def test_verify_taiwan_full_completion_requires_aggregate_by_default():
    module = load_module()
    summary = complete_taiwan_full_summary()
    del summary["taiwan_leaderboard_table"]
    run = FakeRun(summary=summary)

    result = module.verify_full_taiwan_run(run, taxonomy_path=REPO_ROOT / "taxonomies" / "nejumi45_taiwan.yaml")
    relaxed = module.verify_full_taiwan_run(
        run,
        taxonomy_path=REPO_ROOT / "taxonomies" / "nejumi45_taiwan.yaml",
        require_aggregate=False,
    )

    assert result["ok"] is False
    assert any(
        check["name"] == "aggregate_table"
        and check["table_name"] == "taiwan_leaderboard_table"
        and not check["ok"]
        for check in result["checks"]
    )
    assert relaxed["ok"] is True


def test_verify_agentic_swe_wandb_completion_accepts_complete_run():
    module = load_module()
    run = FakeRun(
        summary=complete_agentic_swe_summary(),
        artifacts=[complete_swe_result_artifact()],
    )

    result = module.verify_run(
        run,
        module.BENCHMARK_SPECS["agentic_swe"],
        expected_total=80,
    )

    assert result["ok"] is True
    assert any(
        check["name"] == "accuracy_metric" and check["value"] == 0.3
        for check in result["checks"]
    )


def test_verify_agentic_swe_assorted_completion_accepts_complete_weighted_run():
    module = load_module()
    tier_counts = {"low": 36, "middle": 36, "high": 8}
    run = FakeRun(
        summary=complete_agentic_swe_assorted_summary(),
        artifacts=[complete_swe_assorted_result_artifact()],
        table_files={
            "media/table/agentic_swe_output_table_0.table.json": assorted_table_payload(
                tier_counts
            )
        },
    )

    result = module.verify_run(
        run,
        module.BENCHMARK_SPECS["agentic_swe"],
        expected_total=80,
        require_agentic_swe_assorted=True,
        agentic_swe_assorted_tier_counts=tier_counts,
    )

    assert result["ok"] is True
    assert result["required_evidence"]["agentic_swe_assorted"] == {
        "required": True,
        "expected_tier_counts": tier_counts,
        "weighted_score_complete_metric": "agentic_swe/weighted_pass_at_1_complete",
        "weighted_score_metric": "agentic_swe/weighted_pass_at_1",
        "pass_at_1_metric": "agentic_swe/pass_at_1",
        "output_table_required_columns": module.AGENTIC_SWE_ASSORTED_OUTPUT_TABLE_REQUIRED_COLUMNS
        if isinstance(module.AGENTIC_SWE_ASSORTED_OUTPUT_TABLE_REQUIRED_COLUMNS, list)
        else list(module.AGENTIC_SWE_ASSORTED_OUTPUT_TABLE_REQUIRED_COLUMNS),
        "required_artifact_files": list(module.AGENTIC_SWE_ASSORTED_REQUIRED_ARTIFACT_FILES),
    }
    assert any(
        check["name"] == "accuracy_metric"
        and check["expected"] == "validated by Agentic SWE-Assorted checks"
        for check in result["checks"]
    )
    assert any(
        check["name"] == "agentic_swe_assorted_tier_counts" and check["ok"]
        for check in result["checks"]
    )
    assert any(
        check["name"] == "result_artifact_files" and check["ok"]
        for check in result["checks"]
    )


def test_parse_agentic_swe_assorted_tier_counts_is_case_insensitive():
    module = load_module()

    assert module._parse_agentic_swe_assorted_tier_counts(
        "Low=20,MIDDLE=20, high=10"
    ) == {"low": 20, "middle": 20, "high": 10}


def test_verify_agentic_swe_assorted_tier_counts_is_case_insensitive():
    module = load_module()
    observed_tier_counts = {"Low": 20, "MIDDLE": 20, "high": 10}
    summary = complete_agentic_swe_assorted_summary()
    summary.update(
        {
            "agentic_swe/total_instances": 50,
            "agentic_swe/resolved_instances": 12,
            "agentic_swe_output_table": {
                "_type": "table-file",
                "nrows": 50,
                "path": "media/table/agentic_swe_output_table_0.table.json",
                "columns": AGENTIC_SWE_OUTPUT_COLUMNS
                + AGENTIC_SWE_ASSORTED_OUTPUT_COLUMNS,
            },
        }
    )
    run = FakeRun(
        summary=summary,
        artifacts=[complete_swe_assorted_result_artifact()],
        table_files={
            "media/table/agentic_swe_output_table_0.table.json": assorted_table_payload(
                observed_tier_counts
            )
        },
    )

    result = module.verify_run(
        run,
        module.BENCHMARK_SPECS["agentic_swe"],
        expected_total=50,
        require_agentic_swe_assorted=True,
        agentic_swe_assorted_tier_counts={"Low": 20, "Middle": 20, "High": 10},
    )

    assert result["ok"] is True
    assert any(
        check["name"] == "agentic_swe_assorted_tier_counts" and check["ok"]
        for check in result["checks"]
    )


def test_verify_agentic_swe_assorted_completion_rejects_incomplete_tier_counts():
    module = load_module()
    expected_tier_counts = {"low": 36, "middle": 36, "high": 8}
    observed_tier_counts = {"low": 36, "middle": 36, "high": 7}
    summary = complete_agentic_swe_assorted_summary()
    summary["agentic_swe/total_instances"] = 79
    summary["agentic_swe_output_table"]["nrows"] = 79
    run = FakeRun(
        summary=summary,
        artifacts=[complete_swe_assorted_result_artifact()],
        table_files={
            "media/table/agentic_swe_output_table_0.table.json": assorted_table_payload(
                observed_tier_counts
            )
        },
    )

    result = module.verify_run(
        run,
        module.BENCHMARK_SPECS["agentic_swe"],
        require_agentic_swe_assorted=True,
        agentic_swe_assorted_tier_counts=expected_tier_counts,
    )

    assert result["ok"] is False
    assert any(
        check["name"] == "agentic_swe_assorted_total" and not check["ok"]
        for check in result["checks"]
    )
    assert any(
        check["name"] == "agentic_swe_assorted_tier_counts" and not check["ok"]
        for check in result["checks"]
    )


def test_verify_agentic_swe_assorted_completion_rejects_missing_report_artifact_file():
    module = load_module()
    tier_counts = {"low": 36, "middle": 36, "high": 8}
    artifact = FakeArtifact(
        "agentic-swe-assorted-model-results:v0",
        "evaluation-results",
        ["latest", "production"],
        manifest_entries=["summary.json", "output_table.jsonl", "leaderboard_table.json"],
    )
    run = FakeRun(
        summary=complete_agentic_swe_assorted_summary(),
        artifacts=[artifact],
        table_files={
            "media/table/agentic_swe_output_table_0.table.json": assorted_table_payload(
                tier_counts
            )
        },
    )

    result = module.verify_run(
        run,
        module.BENCHMARK_SPECS["agentic_swe"],
        require_agentic_swe_assorted=True,
        agentic_swe_assorted_tier_counts=tier_counts,
    )

    assert result["ok"] is False
    check = next(
        check for check in result["checks"] if check["name"] == "result_artifact_files"
    )
    assert check["ok"] is False
    assert check["missing_by_artifact"][0]["missing_files"] == ["report.md"]


def test_verify_agentic_swe_wandb_completion_requires_nemoclaw_session_audit():
    module = load_module()
    run = FakeRun(
        summary=complete_agentic_swe_summary(),
        artifacts=[complete_swe_result_artifact()],
    )

    result = module.verify_run(
        run,
        module.BENCHMARK_SPECS["agentic_swe"],
        expected_total=80,
        require_nemoclaw_session_audit=True,
    )

    assert result["ok"] is True
    assert result["observed_evidence"]["nemoclaw_session_audit"]["required"] == 80


def test_verify_agentic_swe_wandb_completion_rejects_output_row_mismatch():
    module = load_module()
    summary = complete_agentic_swe_summary()
    summary["agentic_swe_output_table"] = {
        "_type": "table-file",
        "nrows": 79,
        "columns": AGENTIC_SWE_OUTPUT_COLUMNS,
    }
    run = FakeRun(summary=summary, artifacts=[complete_swe_result_artifact()])

    result = module.verify_run(run, module.BENCHMARK_SPECS["agentic_swe"])

    assert result["ok"] is False
    assert any(
        check["name"] == "output_table" and not check["ok"]
        for check in result["checks"]
    )


def test_verify_agentic_swe_wandb_completion_rejects_missing_output_observability_column():
    module = load_module()
    summary = complete_agentic_swe_summary()
    summary["agentic_swe_output_table"]["columns"] = [
        column
        for column in AGENTIC_SWE_OUTPUT_COLUMNS
        if column != "nemoclaw_session_audit_required"
    ]
    run = FakeRun(summary=summary, artifacts=[complete_swe_result_artifact()])

    result = module.verify_run(run, module.BENCHMARK_SPECS["agentic_swe"])

    assert result["ok"] is False
    assert any(
        check["name"] == "output_table_columns"
        and check["missing_columns"] == ["nemoclaw_session_audit_required"]
        for check in result["checks"]
    )


def test_verify_agentic_swe_wandb_completion_rejects_pass_at_1_mismatch():
    module = load_module()
    summary = complete_agentic_swe_summary()
    summary["agentic_swe/pass_at_1"] = 0.31
    run = FakeRun(summary=summary, artifacts=[complete_swe_result_artifact()])

    result = module.verify_run(run, module.BENCHMARK_SPECS["agentic_swe"])

    assert result["ok"] is False
    assert any(
        check["name"] == "accuracy_metric" and not check["ok"]
        for check in result["checks"]
    )


def test_verify_agentic_math_wandb_completion_rejects_artifact_without_production_alias():
    module = load_module()
    run = FakeRun(
        summary=complete_agentic_math_summary(),
        artifacts=[
            FakeArtifact(
                "agentic-math-olymmath-hard-zh-tw-model-results:v0",
                "evaluation-results",
                ["latest"],
            )
        ],
    )

    result = module.verify_run(run, module.BENCHMARK_SPECS["agentic_math"])

    assert result["ok"] is False
    assert any(
        check["name"] == "result_artifact" and not check["ok"]
        for check in result["checks"]
    )


def test_verify_wandb_completion_accepts_expected_run_metadata():
    module = load_module()
    run = FakeRun(
        summary=complete_agentic_math_summary(),
        artifacts=[complete_result_artifact()],
        config={
            "wandb": {"run_name": "taiwan/full/openai/gpt-4.1-mini: canary"},
            "model": {"pretrained_model_name_or_path": "gpt-4.1-mini-2025-04-14"},
            "run": {"agentic_math": True},
        },
        tags=["taiwan-canary"],
        group="tw-canary",
        job_type="evaluation",
    )

    result = module.verify_run(
        run,
        module.BENCHMARK_SPECS["agentic_math"],
        expected_total=100,
        expected_config={
            "wandb.run_name": "taiwan/full/openai/gpt-4.1-mini: canary",
            "model.pretrained_model_name_or_path": "gpt-4.1-mini-2025-04-14",
            "run.agentic_math": True,
        },
        expected_tags=["taiwan-canary"],
        expected_group="tw-canary",
        expected_job_type="evaluation",
    )

    assert result["ok"] is True
    assert result["required_evidence"]["run_metadata"]["config"] == [
        {
            "key": "model.pretrained_model_name_or_path",
            "expected": "gpt-4.1-mini-2025-04-14",
        },
        {"key": "run.agentic_math", "expected": True},
        {
            "key": "wandb.run_name",
            "expected": "taiwan/full/openai/gpt-4.1-mini: canary",
        },
    ]
    assert result["observed_evidence"]["run_metadata"]["group"] == "tw-canary"
    assert result["observed_evidence"]["run_metadata"]["job_type"] == "evaluation"
    assert {check["name"] for check in result["checks"]} >= {
        "run_config",
        "run_tag",
        "run_group",
        "run_job_type",
    }


def test_verify_wandb_completion_rejects_expected_run_metadata_mismatch():
    module = load_module()
    run = FakeRun(
        summary=complete_agentic_math_summary(),
        artifacts=[complete_result_artifact()],
        config={
            "wandb": {"run_name": "wrong-run"},
            "model": {"pretrained_model_name_or_path": "wrong-model"},
        },
        tags=[],
        group="wrong-group",
        job_type="wrong-job",
    )

    result = module.verify_run(
        run,
        module.BENCHMARK_SPECS["agentic_math"],
        expected_total=100,
        expected_config={
            "wandb.run_name": "taiwan/full/openai/gpt-4.1-mini: canary",
            "model.pretrained_model_name_or_path": "gpt-4.1-mini-2025-04-14",
            "run.agentic_math": True,
        },
        expected_tags=["taiwan-canary"],
        expected_group="tw-canary",
        expected_job_type="evaluation",
    )

    assert result["ok"] is False
    failing = {check["name"] for check in result["checks"] if not check["ok"]}
    assert failing >= {"run_config", "run_tag", "run_group", "run_job_type"}


def test_parse_expected_config_parses_json_values():
    module = load_module()

    parsed = module.parse_expected_config(
        [
            "run.agentic_math=true",
            "generator.max_tokens=2048",
            "model.name=gpt-4.1-mini-2025-04-14",
        ]
    )

    assert parsed == {
        "run.agentic_math": True,
        "generator.max_tokens": 2048,
        "model.name": "gpt-4.1-mini-2025-04-14",
    }


def test_wandb_completion_cli_loads_env_file_and_writes_json(tmp_path, monkeypatch, capsys):
    module = load_module()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "WANDB_ENTITY=test-entity\nWANDB_PROJECT=test-project\nWANDB_API_KEY=hidden\n",
        encoding="utf-8",
    )
    output = tmp_path / "completion.json"
    calls = {}

    def fake_load_run(entity, project, run_id):
        calls["entity"] = entity
        calls["project"] = project
        calls["run_id"] = run_id
        return FakeRun(
            summary=complete_agentic_math_summary(),
            artifacts=[complete_result_artifact()],
            config={
                "wandb": {"run_name": "taiwan/full/openai/gpt-4.1-mini: canary"},
                "model": {"pretrained_model_name_or_path": "gpt-4.1-mini-2025-04-14"},
                "run": {"agentic_math": True},
            },
            tags=["taiwan-canary"],
        )

    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.setattr(module, "load_run", fake_load_run)

    module.main(
        [
            "--run-id",
            "run-1",
            "--benchmark",
            "agentic_math",
            "--expected-total",
            "100",
            "--expected-run-config",
            "run.agentic_math=true",
            "--expected-run-config",
            "model.pretrained_model_name_or_path=\"gpt-4.1-mini-2025-04-14\"",
            "--expected-run-tag",
            "taiwan-canary",
            "--expected-run-job-type",
            "evaluation",
            "--env-file",
            str(env_file),
            "--json",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    printed = json.loads(capsys.readouterr().out)
    assert calls == {"entity": "test-entity", "project": "test-project", "run_id": "run-1"}
    assert payload["ok"] is True
    assert payload["entity"] == "test-entity"
    assert payload["project"] == "test-project"
    assert payload["query_source"] == {
        "kind": "wandb_sdk",
        "api": "wandb.Api",
        "timeout_seconds": 60,
        "entity": "test-entity",
        "project": "test-project",
        "run_id": "run-1",
        "run_path": "test-entity/test-project/run-1",
        "benchmark": "agentic_math",
        "summary_source": "run.summary_metrics",
        "artifact_source": "run.logged_artifacts",
        "history_scanned": False,
    }
    assert payload["env_file_loaded"] is True
    assert isinstance(payload["generated_at"], float)
    assert payload["schema_version"] == module.VERIFICATION_SCHEMA_VERSION
    assert payload["verification_schema_version"] == module.VERIFICATION_SCHEMA_VERSION
    assert payload["status"] == "passed"
    assert payload["required_evidence"]["run_metadata"]["job_type"] == "evaluation"
    assert printed["run_id"] == "run123"
