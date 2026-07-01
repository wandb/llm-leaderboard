import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeArtifact:
    def __init__(self, name, type_, aliases):
        self.name = name
        self.type = type_
        self.aliases = aliases


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
    ):
        self.summary_metrics = summary
        self.state = state
        self._artifacts = artifacts or []
        self.config = config or {}
        self.tags = tags or []
        self.group = group
        self.job_type = job_type

    def logged_artifacts(self):
        return self._artifacts


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
        "agentic_math_output_table": {"_type": "table-file", "nrows": 100},
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
        "agentic_swe_output_table": {"_type": "table-file", "nrows": 80},
    }


def complete_swe_result_artifact():
    return FakeArtifact(
        "agentic-swe-swebench-pro-model-results:v0",
        "evaluation-results",
        ["latest", "production"],
    )


def complete_taiwan_full_summary():
    return {
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
    }


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
        },
    ]
    assert result["observed_evidence"]["artifacts"][0]["aliases"] == ["latest", "production"]
    assert {check["name"] for check in result["checks"]} >= {
        "run_state",
        "total_metric",
        "leaderboard_table",
        "output_table",
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
    summary["agentic_swe_output_table"] = {"_type": "table-file", "nrows": 79}
    run = FakeRun(summary=summary, artifacts=[complete_swe_result_artifact()])

    result = module.verify_run(run, module.BENCHMARK_SPECS["agentic_swe"])

    assert result["ok"] is False
    assert any(
        check["name"] == "output_table" and not check["ok"]
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
