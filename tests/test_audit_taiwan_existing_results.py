import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = REPO_ROOT / "scripts" / "tools" / "audit_taiwan_existing_results.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def write_csv(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_agentic_math_result(root: Path, *, model_slug="model-a", total=3):
    result_dir = root / "agentic_math" / model_slug / "openclaw"
    rows = [
        {
            "task_id": f"task-{index}",
            "predicted_answer": str(index),
            "correct": index < 2,
        }
        for index in range(total)
    ]
    write_jsonl(result_dir / "results.jsonl", rows)
    write_jsonl(result_dir / "results.partial.jsonl", rows)
    write_json(
        result_dir / "summary.json",
        {
            "total_instances": total,
            "answered_instances": total,
            "correct_instances": 2,
            "incorrect_instances": total - 2,
            "accuracy": 2 / total,
            "correctness": 2 / total,
            "model": "provider/model-a",
            "thinking": "low",
            "dry_run": False,
        },
    )
    return result_dir


def write_agentic_math_completion(root: Path, *, run_id="run-1", total=3, current=True):
    payload = {
        "ok": True,
        "benchmark": "agentic_math",
        "entity": "test-entity",
        "project": "test-project",
        "run_id": run_id,
        "run_name": "taiwan-agentic-math-model-a-relog",
        "generated_at": 1,
        "checks": [
            {"name": "total_metric", "ok": True, "value": total},
            {"name": "answered_metric", "ok": True, "value": total},
            {"name": "correct_metric", "ok": True, "value": 2},
            {"name": "accuracy_metric", "ok": True, "value": 2 / total},
            {"name": "output_table", "ok": True, "nrows": total},
            {
                "name": "result_artifact",
                "ok": True,
                "artifacts": [
                    {
                        "name": "agentic-math-model-a-results:v0",
                        "type": "evaluation-results",
                        "aliases": ["production"],
                    }
                ],
            },
        ],
    }
    if current:
        payload.update(
            {
                "schema_version": 1,
                "verification_schema_version": 1,
                "status": "passed",
                "query_source": {
                    "kind": "wandb_sdk",
                    "api": "wandb.Api",
                    "timeout_seconds": 60,
                    "entity": "test-entity",
                    "project": "test-project",
                    "run_id": run_id,
                    "run_path": f"test-entity/test-project/{run_id}",
                    "benchmark": "agentic_math",
                    "summary_source": "run.summary_metrics",
                    "artifact_source": "run.logged_artifacts",
                    "history_scanned": False,
                },
                "required_evidence": {
                    "run_state": "finished",
                    "expected_total": total,
                    "summary_metrics": [
                        "agentic_math/total_instances",
                        "agentic_math/answered_instances",
                        "agentic_math/correct_instances",
                        "agentic_math/accuracy",
                    ],
                    "tables": [
                        {
                            "name": "agentic_math_leaderboard_table",
                            "row_count": ">=1",
                        },
                        {
                            "name": "agentic_math_output_table",
                            "row_count": "must equal total metric",
                        },
                    ],
                    "artifacts": [
                        {
                            "type": "evaluation-results",
                            "required_aliases": ["production"],
                        }
                    ],
                },
                "observed_evidence": {
                    "run_state": "finished",
                    "expected_total": total,
                    "summary_metrics": {
                        "agentic_math/total_instances": {"ok": True, "value": total},
                        "agentic_math/answered_instances": {"ok": True, "value": total},
                        "agentic_math/correct_instances": {"ok": True, "value": 2},
                        "agentic_math/accuracy": {"ok": True, "value": 2 / total},
                    },
                    "tables": [
                        {
                            "name": "agentic_math_leaderboard_table",
                            "ok": True,
                            "nrows": 1,
                        },
                        {
                            "name": "agentic_math_output_table",
                            "ok": True,
                            "nrows": total,
                        },
                    ],
                    "artifacts": [
                        {
                            "name": "agentic-math-model-a-results:v0",
                            "type": "evaluation-results",
                            "aliases": ["production"],
                        }
                    ],
                },
            }
        )
    return write_json(root / "wandb_completion" / f"agentic_math-{run_id}.json", payload)


def write_taiwan_full_provisional(root: Path, *, run_id="full-run-1"):
    provisional = root / "provisional_leaderboard"
    write_csv(
        provisional / "leaderboard.csv",
        (
            "slug,model_name,Overall,GLP,ALT,source_run_id,source_run_name,source_run_url\n"
            f"model-full,Model Full,87.5,88,86,{run_id},taiwan/full/model,"
            "https://wandb.ai/test-entity/test-project/runs/full-run-1\n"
        ),
    )
    write_csv(
        provisional / "unit_scores.csv",
        "slug,unit_id,score,source_run_id\nmodel-full,agentic_math,0.86,full-run-1\n",
    )
    write_csv(
        provisional / "run_status.csv",
        (
            "slug,run_name,run_id,state,status,url\n"
            f"model-full,taiwan/full/model,{run_id},finished,ok,"
            "https://wandb.ai/test-entity/test-project/runs/full-run-1\n"
        ),
    )
    return provisional


def write_taiwan_full_completion(root: Path, *, run_id="full-run-1", current=True):
    payload = {
        "ok": True,
        "benchmark": "taiwan_full",
        "entity": "test-entity",
        "project": "test-project",
        "run_id": run_id,
        "run_name": "taiwan/full/model",
        "generated_at": 1,
        "checks": [
            {"name": "run_state", "ok": True, "state": "finished"},
            {
                "name": "taxonomy_table",
                "ok": True,
                "unit_id": "agentic_math",
                "display_name": "Agentic Math",
                "table_name": "agentic_math_leaderboard_table",
                "nrows": 1,
            },
            {
                "name": "aggregate_table",
                "ok": True,
                "table_name": "taiwan_leaderboard_table",
                "nrows": 1,
            },
        ],
    }
    if current:
        payload.update(
            {
                "schema_version": 1,
                "verification_schema_version": 1,
                "status": "passed",
                "query_source": {
                    "kind": "wandb_sdk",
                    "api": "wandb.Api",
                    "timeout_seconds": 60,
                    "entity": "test-entity",
                    "project": "test-project",
                    "run_id": run_id,
                    "run_path": f"test-entity/test-project/{run_id}",
                    "benchmark": "taiwan_full",
                    "summary_source": "run.summary_metrics",
                    "artifact_source": "run.logged_artifacts",
                    "history_scanned": False,
                },
                "required_evidence": {
                    "run_state": "finished",
                    "taxonomy_path": "taxonomies/nejumi45_taiwan.yaml",
                    "taxonomy_version": "test",
                    "num_few_shots": 0,
                    "include_pending": False,
                    "skipped_pending_units": [],
                    "taxonomy_tables": [
                        {
                            "unit_id": "agentic_math",
                            "display_name": "Agentic Math",
                            "table_name": "agentic_math_leaderboard_table",
                            "row_count": ">=1",
                        }
                    ],
                    "aggregate_tables": [
                        {"name": "taiwan_leaderboard_table", "row_count": ">=1"}
                    ],
                },
                "observed_evidence": {
                    "run_state": "finished",
                    "taxonomy_tables": [
                        {
                            "unit_id": "agentic_math",
                            "display_name": "Agentic Math",
                            "table_name": "agentic_math_leaderboard_table",
                            "ok": True,
                            "nrows": 1,
                        }
                    ],
                    "aggregate_tables": [
                        {"name": "taiwan_leaderboard_table", "ok": True, "nrows": 1}
                    ],
                    "skipped_pending_units": [],
                },
            }
        )
    return write_json(root / "wandb_completion" / f"taiwan_full-{run_id}.json", payload)


def test_existing_result_audit_accepts_formalized_agentic_math(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "AGENTIC_MATH_EXPECTED_TOTAL", 3)
    write_agentic_math_result(tmp_path, total=3)
    write_agentic_math_completion(tmp_path, total=3)

    audit = module.build_audit(
        output_root=tmp_path,
        completion_dir=tmp_path / "wandb_completion",
    )

    assert audit["ok"] is True
    assert audit["summary"]["complete_local_count"] == 1
    assert audit["summary"]["formalized_wandb_complete_count"] == 1
    assert audit["summary"]["unformalized_complete_count"] == 0
    record = audit["formalized_records"][0]
    assert record["formalization_status"] == "formalized_wandb_complete"
    assert record["wandb_completion"]["entity"] == "test-entity"
    assert record["wandb_completion"]["project"] == "test-project"
    assert record["wandb_completion"]["run_id"] == "run-1"
    assert record["wandb_completion"]["schema_current"] is True
    assert record["wandb_completion"]["observed_evidence_present"] is True


def test_existing_result_audit_accepts_duplicate_completion_for_same_run_id(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "AGENTIC_MATH_EXPECTED_TOTAL", 3)
    write_agentic_math_result(tmp_path, total=3)
    first = write_agentic_math_completion(tmp_path, run_id="run-1", total=3)
    refreshed = tmp_path / "wandb_completion" / "agentic_math-run-1-refresh.json"
    payload = json.loads(first.read_text(encoding="utf-8"))
    payload["generated_at"] = 2
    write_json(refreshed, payload)

    audit = module.build_audit(
        output_root=tmp_path,
        completion_dir=tmp_path / "wandb_completion",
    )

    assert audit["ok"] is True
    assert audit["summary"]["complete_local_count"] == 1
    assert audit["summary"]["formalized_wandb_complete_count"] == 1
    assert audit["summary"]["unformalized_complete_count"] == 0
    record = audit["formalized_records"][0]
    assert record["formalization_status"] == "formalized_wandb_complete"
    assert record["wandb_completion"]["run_id"] == "run-1"
    assert record["wandb_completion"]["path"].endswith("agentic_math-run-1-refresh.json")
    assert sorted(
        Path(path).name for path in record["matching_wandb_completion_paths"]
    ) == ["agentic_math-run-1-refresh.json", "agentic_math-run-1.json"]
    assert [Path(path).name for path in record["deduped_wandb_completion_paths"]] == [
        "agentic_math-run-1.json"
    ]
    assert record["warnings"] == [
        "multiple W&B completion JSONs matched the same W&B entity/project/run_id; selected newest generated_at"
    ]


def test_existing_result_audit_rejects_duplicate_completion_for_same_run_id_different_entity(
    tmp_path, monkeypatch
):
    module = load_module()
    monkeypatch.setattr(module, "AGENTIC_MATH_EXPECTED_TOTAL", 3)
    write_agentic_math_result(tmp_path, total=3)
    first = write_agentic_math_completion(tmp_path, run_id="run-1", total=3)
    other_entity = tmp_path / "wandb_completion" / "agentic_math-run-1-other-entity.json"
    payload = json.loads(first.read_text(encoding="utf-8"))
    payload["entity"] = "other-entity"
    payload["query_source"]["entity"] = "other-entity"
    payload["query_source"]["run_path"] = "other-entity/test-project/run-1"
    payload["generated_at"] = 2
    write_json(other_entity, payload)

    audit = module.build_audit(
        output_root=tmp_path,
        completion_dir=tmp_path / "wandb_completion",
    )

    assert audit["ok"] is False
    assert audit["summary"]["complete_local_count"] == 1
    assert audit["summary"]["formalized_wandb_complete_count"] == 0
    assert audit["summary"]["unformalized_complete_count"] == 1
    record = audit["unformalized_complete_records"][0]
    assert record["formalization_status"] == "ambiguous_wandb_completion"
    assert record["errors"] == ["multiple W&B completion JSONs match this local result"]
    assert sorted(
        Path(path).name for path in record["matching_wandb_completion_paths"]
    ) == ["agentic_math-run-1-other-entity.json", "agentic_math-run-1.json"]


def test_existing_result_audit_rejects_duplicate_completion_for_different_run_ids(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "AGENTIC_MATH_EXPECTED_TOTAL", 3)
    write_agentic_math_result(tmp_path, total=3)
    write_agentic_math_completion(tmp_path, run_id="run-1", total=3)
    write_agentic_math_completion(tmp_path, run_id="run-2", total=3)

    audit = module.build_audit(
        output_root=tmp_path,
        completion_dir=tmp_path / "wandb_completion",
    )

    assert audit["ok"] is False
    assert audit["summary"]["complete_local_count"] == 1
    assert audit["summary"]["formalized_wandb_complete_count"] == 0
    assert audit["summary"]["unformalized_complete_count"] == 1
    record = audit["unformalized_complete_records"][0]
    assert record["formalization_status"] == "ambiguous_wandb_completion"
    assert record["errors"] == ["multiple W&B completion JSONs match this local result"]
    assert sorted(
        Path(path).name for path in record["matching_wandb_completion_paths"]
    ) == ["agentic_math-run-1.json", "agentic_math-run-2.json"]


def test_existing_result_audit_rejects_legacy_completion_without_current_schema(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "AGENTIC_MATH_EXPECTED_TOTAL", 3)
    write_agentic_math_result(tmp_path, total=3)
    write_agentic_math_completion(tmp_path, total=3, current=False)

    audit = module.build_audit(
        output_root=tmp_path,
        completion_dir=tmp_path / "wandb_completion",
    )

    assert audit["ok"] is False
    assert audit["summary"]["complete_local_count"] == 1
    assert audit["summary"]["formalized_wandb_complete_count"] == 0
    assert audit["summary"]["unformalized_complete_count"] == 1
    assert audit["wandb_completion_records"][0]["schema_current"] is False
    assert audit["wandb_completion_records"][0]["observed_evidence_present"] is False


def test_existing_result_audit_rejects_completion_without_entity_project(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "AGENTIC_MATH_EXPECTED_TOTAL", 3)
    write_agentic_math_result(tmp_path, total=3)
    completion = write_agentic_math_completion(tmp_path, total=3, current=True)
    payload = json.loads(completion.read_text(encoding="utf-8"))
    payload.pop("entity")
    payload.pop("project")
    completion.write_text(json.dumps(payload), encoding="utf-8")

    audit = module.build_audit(
        output_root=tmp_path,
        completion_dir=tmp_path / "wandb_completion",
    )

    assert audit["ok"] is False
    assert audit["summary"]["formalized_wandb_complete_count"] == 0
    assert audit["summary"]["unformalized_complete_count"] == 1
    assert audit["wandb_completion_records"][0]["schema_current"] is False


def test_existing_result_audit_rejects_non_finished_completion(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "AGENTIC_MATH_EXPECTED_TOTAL", 3)
    write_agentic_math_result(tmp_path, total=3)
    completion = write_agentic_math_completion(tmp_path, total=3, current=True)
    payload = json.loads(completion.read_text(encoding="utf-8"))
    payload["observed_evidence"]["run_state"] = "running"
    completion.write_text(json.dumps(payload), encoding="utf-8")

    audit = module.build_audit(
        output_root=tmp_path,
        completion_dir=tmp_path / "wandb_completion",
    )

    assert audit["ok"] is False
    assert audit["summary"]["formalized_wandb_complete_count"] == 0
    assert audit["summary"]["unformalized_complete_count"] == 1
    assert audit["wandb_completion_records"][0]["schema_current"] is False


def test_existing_result_audit_rejects_completion_with_failed_status_even_when_ok_true(
    tmp_path, monkeypatch
):
    module = load_module()
    monkeypatch.setattr(module, "AGENTIC_MATH_EXPECTED_TOTAL", 3)
    write_agentic_math_result(tmp_path, total=3)
    completion = write_agentic_math_completion(tmp_path, total=3, current=True)
    payload = json.loads(completion.read_text(encoding="utf-8"))
    payload["status"] = "failed"
    completion.write_text(json.dumps(payload), encoding="utf-8")

    audit = module.build_audit(
        output_root=tmp_path,
        completion_dir=tmp_path / "wandb_completion",
    )

    assert audit["ok"] is False
    assert audit["summary"]["formalized_wandb_complete_count"] == 0
    assert audit["summary"]["unformalized_complete_count"] == 1
    record = audit["wandb_completion_records"][0]
    assert record["schema_current"] is False
    assert "status must be passed" in record["schema_current_issues"]


def test_existing_result_audit_rejects_completion_with_failed_check_even_when_ok_true(
    tmp_path, monkeypatch
):
    module = load_module()
    monkeypatch.setattr(module, "AGENTIC_MATH_EXPECTED_TOTAL", 3)
    write_agentic_math_result(tmp_path, total=3)
    completion = write_agentic_math_completion(tmp_path, total=3, current=True)
    payload = json.loads(completion.read_text(encoding="utf-8"))
    payload["checks"][0]["ok"] = False
    completion.write_text(json.dumps(payload), encoding="utf-8")

    audit = module.build_audit(
        output_root=tmp_path,
        completion_dir=tmp_path / "wandb_completion",
    )

    assert audit["ok"] is False
    assert audit["summary"]["formalized_wandb_complete_count"] == 0
    assert audit["summary"]["unformalized_complete_count"] == 1
    record = audit["wandb_completion_records"][0]
    assert record["schema_current"] is False
    assert any(
        issue.startswith("all checks must be ok=true")
        for issue in record["schema_current_issues"]
    )


def test_existing_result_audit_rejects_completion_with_query_source_mismatch(
    tmp_path, monkeypatch
):
    module = load_module()
    monkeypatch.setattr(module, "AGENTIC_MATH_EXPECTED_TOTAL", 3)
    write_agentic_math_result(tmp_path, total=3)
    completion = write_agentic_math_completion(tmp_path, total=3, current=True)
    payload = json.loads(completion.read_text(encoding="utf-8"))
    payload["query_source"]["run_id"] = "different-run"
    completion.write_text(json.dumps(payload), encoding="utf-8")

    audit = module.build_audit(
        output_root=tmp_path,
        completion_dir=tmp_path / "wandb_completion",
    )

    assert audit["ok"] is False
    assert audit["summary"]["formalized_wandb_complete_count"] == 0
    assert audit["summary"]["unformalized_complete_count"] == 1
    record = audit["wandb_completion_records"][0]
    assert record["schema_current"] is False
    assert "query_source.run_id must match top-level run_id" in record["schema_current_issues"]


def test_existing_result_audit_rejects_complete_local_without_wandb(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "AGENTIC_MATH_EXPECTED_TOTAL", 3)
    write_agentic_math_result(tmp_path, total=3)

    audit = module.build_audit(
        output_root=tmp_path,
        completion_dir=tmp_path / "wandb_completion",
    )

    assert audit["ok"] is False
    assert audit["status"] == "unformalized_complete_results"
    assert audit["summary"]["unformalized_complete_count"] == 1
    assert audit["unformalized_complete_records"][0]["formalization_status"] == "local_complete_needs_wandb_relog"
    record = audit["unformalized_complete_records"][0]
    assert "log_agentic_math_results_to_wandb.py" in record["relog_dry_run_command"]
    assert "--dry-run" in record["relog_dry_run_command"]
    assert "--plan-json" in record["relog_dry_run_command"]
    assert "--validated-dry-run-plan-json" not in record["relog_dry_run_command"]
    assert record["relog_dry_run_plan_json"].endswith(
        "agentic-math-model-a.plan.json"
    )
    assert "log_agentic_math_results_to_wandb.py" in record["relog_command"]
    assert "--validated-dry-run-plan-json" in record["relog_command"]
    assert "--external-action-approval-source-packet-json" in record["relog_command"]
    assert "EXTERNAL_ACTION_APPROVAL_SOURCE_PACKET_JSON" in record["relog_command"]
    assert "--external-action-approval-report-json" in record["relog_command"]
    assert "EXTERNAL_ACTION_APPROVAL_REPORT_JSON" in record["relog_command"]
    assert record["relog_dry_run_plan_json"] in record["relog_command"]
    assert "--external-action-approval-source-packet-json" not in record["relog_dry_run_command"]
    assert "--external-action-approval-report-json" not in record["relog_dry_run_command"]
    assert audit["remediation_commands"][0] == record["relog_dry_run_command"]
    verify_command = record["verify_command"]
    assert "verify_taiwan_wandb_completion.py" in verify_command
    assert verify_command.count("uv run python scripts/tools/verify_taiwan_wandb_completion.py") == 1


def test_existing_result_audit_accepts_formalized_taiwan_full_provisional_run(tmp_path):
    module = load_module()
    write_taiwan_full_provisional(tmp_path)
    write_taiwan_full_completion(tmp_path)

    audit = module.build_audit(
        output_root=tmp_path,
        completion_dir=tmp_path / "wandb_completion",
    )

    assert audit["ok"] is True
    assert audit["summary"]["complete_local_count"] == 1
    assert audit["summary"]["formalized_wandb_complete_count"] == 1
    assert audit["summary"]["unformalized_complete_count"] == 0
    record = audit["formalized_records"][0]
    assert record["benchmark"] == "taiwan_full"
    assert record["model_slug"] == "model-full"
    assert record["formalization_status"] == "formalized_wandb_complete"
    assert record["wandb_completion"]["run_id"] == "full-run-1"
    assert record["wandb_completion"]["schema_current"] is True
    assert record["wandb_completion"]["observed_evidence_present"] is True


def test_existing_result_audit_rejects_taiwan_full_without_completion(tmp_path):
    module = load_module()
    write_taiwan_full_provisional(tmp_path)

    audit = module.build_audit(
        output_root=tmp_path,
        completion_dir=tmp_path / "wandb_completion",
    )

    assert audit["ok"] is False
    assert audit["status"] == "unformalized_complete_results"
    assert audit["summary"]["complete_local_count"] == 1
    assert audit["summary"]["unformalized_complete_count"] == 1
    record = audit["unformalized_complete_records"][0]
    assert record["benchmark"] == "taiwan_full"
    assert record["relog_command"] == ""
    assert "--benchmark taiwan_full" in record["verify_command"]
    assert "--run-id full-run-1" in record["verify_command"]


def test_existing_result_audit_treats_empty_taiwan_full_provisional_as_partial(tmp_path):
    module = load_module()
    provisional = tmp_path / "provisional_leaderboard"
    write_csv(
        provisional / "leaderboard.csv",
        "slug,model_name,Overall,GLP,ALT,source_run_id,source_run_name,source_run_url\n",
    )
    write_csv(
        provisional / "run_status.csv",
        "slug,run_name,run_id,state,status,url\nmodel-full,taiwan/full/model,kqbj1qh4,running,running,\n",
    )

    audit = module.build_audit(
        output_root=tmp_path,
        completion_dir=tmp_path / "wandb_completion",
    )

    assert audit["ok"] is True
    assert audit["summary"]["complete_local_count"] == 0
    assert audit["summary"]["partial_or_probe_count"] == 1
    record = audit["partial_records"][0]
    assert record["benchmark"] == "taiwan_full"
    assert record["model_slug"] == "provisional_leaderboard"
    assert record["formalization_status"] == "partial_not_reloggable"
    assert record["run_status_count"] == 1
    assert record["warnings"] == ["no completed Taiwan full W&B run rows in provisional leaderboard"]


def test_existing_result_audit_does_not_block_partial_outputs(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "AGENTIC_MATH_EXPECTED_TOTAL", 3)
    result_dir = tmp_path / "agentic_math" / "model-b" / "openclaw"
    write_jsonl(result_dir / "results.partial.jsonl", [{"task_id": "task-1"}])

    audit = module.build_audit(
        output_root=tmp_path,
        completion_dir=tmp_path / "wandb_completion",
    )

    assert audit["ok"] is True
    assert audit["summary"]["complete_local_count"] == 0
    assert audit["summary"]["partial_or_probe_count"] == 1
    assert audit["partial_records"][0]["formalization_status"] == "partial_not_reloggable"


def test_existing_result_audit_uses_unique_probe_relog_plan_paths(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "AGENTIC_MATH_EXPECTED_TOTAL", 100)
    model_dir = tmp_path / "agentic_math" / "model-d"
    for probe_name in ("probe-a", "probe-b"):
        result_dir = model_dir / probe_name
        write_jsonl(
            result_dir / "results.jsonl",
            [{"task_id": probe_name, "predicted_answer": "", "correct": False}],
        )
        write_jsonl(
            result_dir / "results.partial.jsonl",
            [{"task_id": probe_name, "predicted_answer": "", "correct": False}],
        )
        write_json(
            result_dir / "summary.json",
            {
                "total_instances": 1,
                "answered_instances": 0,
                "correct_instances": 0,
                "incorrect_instances": 1,
                "accuracy": 0.0,
                "model": "provider/model-d",
                "thinking": "low",
                "dry_run": False,
            },
        )

    audit = module.build_audit(
        output_root=tmp_path,
        completion_dir=tmp_path / "wandb_completion",
    )

    probe_records = [
        record for record in audit["partial_records"] if record["run_kind"] == "probe"
    ]
    assert len(probe_records) == 2
    plan_paths = {record["relog_dry_run_plan_json"] for record in probe_records}
    assert plan_paths == {
        "temp/wandb_relog_plans/agentic-math-model-d-probe-probe-a.plan.json",
        "temp/wandb_relog_plans/agentic-math-model-d-probe-probe-b.plan.json",
    }


def test_existing_result_audit_marks_swe_patches_as_partial_without_full_official_eval(tmp_path):
    module = load_module()
    patches = tmp_path / "swebench_pro" / "model-c" / "openclaw" / "patches.json"
    write_json(patches, [{"instance_id": "i1", "patch": "diff --git a/x b/x"}])

    audit = module.build_audit(
        output_root=tmp_path,
        completion_dir=tmp_path / "wandb_completion",
    )

    assert audit["ok"] is True
    assert audit["summary"]["partial_or_probe_count"] == 1
    record = audit["partial_records"][0]
    assert record["benchmark"] == "agentic_swe"
    assert record["patch_count"] == 1


def test_existing_result_audit_reports_swe_relog_command_for_complete_local_eval(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "AGENTIC_SWE_EXPECTED_TOTAL", 1)
    model_dir = tmp_path / "swebench_pro" / "model-c"
    write_json(
        model_dir / "openclaw" / "patches.json",
        [{"instance_id": "i1", "patch": "diff --git a/x b/x"}],
    )
    write_json(
        model_dir / "official_eval" / "summary.json",
        {
            "total_instances": 1,
            "resolved_instances": 1,
            "unresolved_instances": 0,
            "pass_at_1": 1.0,
            "resolved_ids": ["i1"],
            "unresolved_ids": [],
        },
    )

    audit = module.build_audit(
        output_root=tmp_path,
        completion_dir=tmp_path / "wandb_completion",
    )

    assert audit["ok"] is False
    record = audit["unformalized_complete_records"][0]
    assert record["benchmark"] == "agentic_swe"
    assert "log_agentic_swe_results_to_wandb.py" in record["relog_command"]
    assert "log_agentic_swe_results_to_wandb.py" in record["relog_dry_run_command"]
    assert "--official-eval-dir" in record["relog_command"]
    assert "--validated-dry-run-plan-json" in record["relog_command"]
    assert "--external-action-approval-source-packet-json" in record["relog_command"]
    assert "EXTERNAL_ACTION_APPROVAL_SOURCE_PACKET_JSON" in record["relog_command"]
    assert "--external-action-approval-report-json" in record["relog_command"]
    assert "EXTERNAL_ACTION_APPROVAL_REPORT_JSON" in record["relog_command"]
    assert record["relog_dry_run_plan_json"] in record["relog_command"]
    assert "--dry-run" in record["relog_dry_run_command"]
    assert "--validated-dry-run-plan-json" not in record["relog_dry_run_command"]
    assert "--external-action-approval-source-packet-json" not in record["relog_dry_run_command"]
    assert "--external-action-approval-report-json" not in record["relog_dry_run_command"]
    assert record["relog_dry_run_plan_json"].endswith(
        "agentic-swe-model-c.plan.json"
    )
    assert "verify_taiwan_wandb_completion.py" in record["verify_command"]
