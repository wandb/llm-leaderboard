import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tools" / "evaluate_swebench_lite_patches.py"
OFFICIAL_REPO = REPO_ROOT / "external" / "SWE-bench"


def load_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("evaluate_swebench_lite_patches", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_round_robin_instance_ids_by_repo_preserves_coverage_and_spreads_repos():
    module = load_module()
    instance_ids = [
        "django__django-1",
        "django__django-2",
        "django__django-3",
        "sympy__sympy-1",
        "sympy__sympy-2",
        "pytest-dev__pytest-1",
    ]

    scheduled = module.round_robin_instance_ids_by_repo(instance_ids)

    assert scheduled == [
        "django__django-1",
        "sympy__sympy-1",
        "pytest-dev__pytest-1",
        "django__django-2",
        "sympy__sympy-2",
        "django__django-3",
    ]
    assert sorted(scheduled) == sorted(instance_ids)


def test_resolve_executable_survives_child_working_directory_change(tmp_path):
    module = load_module()
    executable = tmp_path / "bin" / "python"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\n", encoding="utf-8")

    relative = os.path.relpath(executable, Path.cwd())
    resolved = module.resolve_executable(relative)

    assert resolved == str(executable.resolve())


def test_resolve_executable_accepts_path_lookup():
    module = load_module()

    assert Path(module.resolve_executable("python3")).is_absolute()


def test_resolve_executable_preserves_virtualenv_symlink(tmp_path):
    module = load_module()
    executable = tmp_path / "venv" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)

    assert module.resolve_executable(str(executable)) == str(executable.absolute())


def test_official_runtime_check_rejects_missing_python_dependency(
    tmp_path, monkeypatch
):
    module = load_module()

    def fake_run_command(command, *, cwd, env, check, timeout):
        return subprocess.CompletedProcess(
            command,
            1,
            "",
            "ModuleNotFoundError: No module named 'ghapi'",
        )

    monkeypatch.setattr(module, "run_command", fake_run_command)

    with pytest.raises(RuntimeError, match="Python runtime is incomplete"):
        module.check_official_harness_runtime(
            python=sys.executable,
            official_repo=tmp_path,
        )


def test_incomplete_official_report_is_not_scored():
    module = load_module()

    with pytest.raises(RuntimeError, match="cannot be converted into model scores"):
        module.validate_complete_report(
            {
                "submitted_ids": ["a", "b"],
                "completed_ids": [],
                "error_ids": ["a", "b"],
                "incomplete_ids": ["a", "b"],
                "completed_instances": 0,
                "error_instances": 2,
            },
            expected_instance_ids=["a", "b"],
        )


def test_complete_official_report_is_accepted():
    module = load_module()
    module.validate_complete_report(
        {
            "submitted_ids": ["a", "b"],
            "completed_ids": ["a", "b"],
            "error_ids": [],
            "incomplete_ids": [],
            "completed_instances": 2,
            "error_instances": 0,
        },
        expected_instance_ids=["a", "b"],
    )


def test_isolated_evaluation_accepts_official_summary_written_to_cwd(
    tmp_path, monkeypatch
):
    module = load_module()
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "instance_id": "django__django-1",
                "model_patch": "diff --git a/a b/a\n",
                "model_name_or_path": "test-model",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        output_dir=tmp_path,
        python=sys.executable,
        dataset_name="dataset",
        split="test",
        max_workers=1,
        run_id="isolated-test",
        timeout=30,
        cache_level="env",
        clean=False,
        force_rebuild=False,
        namespace="swebench",
        rewrite_reports=False,
        model_name="test-model",
        isolated_cleanup_grace_seconds=5,
        isolated_infrastructure_retries=1,
    )

    def fake_run_command(command, *, cwd, env, check, timeout):
        run_id = command[command.index("--run_id") + 1]
        report_path = cwd / module.report_filename(args.model_name, run_id)
        report_path.write_text(
            json.dumps(
                {
                    "submitted_ids": ["django__django-1"],
                    "completed_ids": ["django__django-1"],
                    "resolved_ids": ["django__django-1"],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module, "run_command", fake_run_command)

    report, commands = module.evaluate_isolated_instances(
        args,
        predictions_path=predictions_path,
        instance_ids=["django__django-1"],
        cwd=tmp_path,
        env={},
    )

    assert len(commands) == 1
    assert report["completed_ids"] == ["django__django-1"]
    assert report["resolved_ids"] == ["django__django-1"]
    cached = json.loads(
        (tmp_path / "isolated" / "django__django-1" / "result.json").read_text(
            encoding="utf-8"
        )
    )
    assert cached["ok"] is True


def test_isolated_evaluation_accepts_detail_after_cleanup_failure(
    tmp_path, monkeypatch
):
    module = load_module()
    instance_id = "sphinx-doc__sphinx-8713"
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "model_patch": "diff --git a/a b/a\n",
                "model_name_or_path": "test/model",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        output_dir=tmp_path,
        python=sys.executable,
        dataset_name="dataset",
        split="test",
        max_workers=1,
        run_id="cleanup-test",
        timeout=30,
        cache_level="env",
        clean=False,
        force_rebuild=False,
        namespace="swebench",
        rewrite_reports=False,
        model_name="test/model",
        isolated_cleanup_grace_seconds=5,
        isolated_infrastructure_retries=1,
    )

    def fake_run_command(command, *, cwd, env, check, timeout):
        run_id = command[command.index("--run_id") + 1]
        detail_dir = (
            cwd
            / "logs"
            / "run_evaluation"
            / run_id
            / "test__model"
            / instance_id
        )
        detail_dir.mkdir(parents=True)
        (detail_dir / "report.json").write_text(
            json.dumps(
                {
                    instance_id: {
                        "patch_exists": True,
                        "patch_successfully_applied": True,
                        "resolved": True,
                    }
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 1, "All instances run.", "cleanup failed")

    monkeypatch.setattr(module, "run_command", fake_run_command)

    report, commands = module.evaluate_isolated_instances(
        args,
        predictions_path=predictions_path,
        instance_ids=[instance_id],
        cwd=tmp_path,
        env={},
    )

    assert len(commands) == 1
    assert report["completed_ids"] == [instance_id]
    assert report["resolved_ids"] == [instance_id]
    assert report["error_ids"] == []
    cached = json.loads(
        (tmp_path / "isolated" / instance_id / "result.json").read_text(
            encoding="utf-8"
        )
    )
    assert cached["ok"] is True
    assert cached["cleanup_warning"] == "official harness return code 1"
    assert cached["attempts"][0]["returncode"] == 1


def test_isolated_evaluation_retries_only_missing_infrastructure_result(
    tmp_path, monkeypatch
):
    module = load_module()
    instance_id = "django__django-1"
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "model_patch": "diff --git a/a b/a\n",
                "model_name_or_path": "test-model",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        output_dir=tmp_path,
        python=sys.executable,
        dataset_name="dataset",
        split="test",
        max_workers=1,
        run_id="retry-test",
        timeout=30,
        cache_level="env",
        clean=False,
        force_rebuild=False,
        namespace="swebench",
        rewrite_reports=False,
        model_name="test-model",
        isolated_cleanup_grace_seconds=5,
        isolated_infrastructure_retries=1,
    )
    calls = 0

    def fake_run_command(command, *, cwd, env, check, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(command, 1, "", "docker unavailable")
        run_id = command[command.index("--run_id") + 1]
        report_path = cwd / module.report_filename(args.model_name, run_id)
        report_path.write_text(
            json.dumps(
                {
                    "submitted_ids": [instance_id],
                    "completed_ids": [instance_id],
                    "resolved_ids": [],
                    "unresolved_ids": [instance_id],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module, "run_command", fake_run_command)

    report, commands = module.evaluate_isolated_instances(
        args,
        predictions_path=predictions_path,
        instance_ids=[instance_id],
        cwd=tmp_path,
        env={},
    )

    assert calls == 2
    assert len(commands) == 2
    assert report["completed_ids"] == [instance_id]
    assert report["resolved_ids"] == []
    assert report["error_ids"] == []
    cached = json.loads(
        (tmp_path / "isolated" / instance_id / "result.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(cached["attempts"]) == 2


@pytest.mark.skipif(
    not (OFFICIAL_REPO / "swebench" / "harness" / "run_evaluation.py").exists(),
    reason="official SWE-bench checkout is not installed",
)
def test_evaluate_swebench_lite_prepare_only_writes_official_predictions(tmp_path):
    patch_path = tmp_path / "patches.json"
    patch_path.write_text(
        json.dumps(
            [
                {
                    "instance_id": "django__django-13447",
                    "patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n",
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ids_path = tmp_path / "ids.json"
    ids_path.write_text(json.dumps(["django__django-13447"]) + "\n", encoding="utf-8")
    output_dir = tmp_path / "eval"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--official-repo",
            str(OFFICIAL_REPO),
            "--patch-path",
            str(patch_path),
            "--instance-ids-json",
            str(ids_path),
            "--output-dir",
            str(output_dir),
            "--model-name",
            "openai-direct/gpt-4.1-mini-2025-04-14",
            "--run-id",
            "prepare-test",
            "--prepare-only",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=True,
    )

    predictions = [
        json.loads(line)
        for line in (output_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    invocation = json.loads((output_dir / "official_invocation.json").read_text(encoding="utf-8"))

    assert result.stdout
    assert predictions == [
        {
            "instance_id": "django__django-13447",
            "model_patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n",
            "model_name_or_path": "openai-direct/gpt-4.1-mini-2025-04-14",
        }
    ]
    assert "--dataset_name" in invocation["command"]
    assert "princeton-nlp/SWE-bench_Lite" in invocation["command"]
    assert invocation["instance_ids"] == ["django__django-13447"]
