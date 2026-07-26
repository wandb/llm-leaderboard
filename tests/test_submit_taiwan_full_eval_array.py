import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "scripts" / "tools"


def load_module():
    path = TOOLS_DIR / "submit_taiwan_full_eval_array.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def make_submit_args(module, tmp_path, *extra):
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    (repo / "scripts" / "tools").mkdir(parents=True)
    (repo / "scripts" / "slurm").mkdir(parents=True)
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / "configs" / "models.yaml").write_text("models: []\n", encoding="utf-8")
    (repo / "configs" / "base.yaml").write_text("run: {}\n", encoding="utf-8")
    (repo / "python-real").write_text("", encoding="utf-8")
    (repo / ".venv" / "bin" / "python").symlink_to(repo / "python-real")
    return module.parse_args(
        [
            "submit",
            "--repo-root",
            str(repo),
            "--manifest",
            "configs/models.yaml",
            "--base-config",
            "configs/base.yaml",
            "--python",
            ".venv/bin/python",
            "--bundle-root",
            "outputs/jobs",
            "--batch-id",
            "test-array",
            "--model",
            "model-a",
            "--model",
            "model-b",
            *extra,
        ]
    )


def test_build_sbatch_command_is_portable_array_with_isolated_tasks(tmp_path, monkeypatch):
    module = load_module()
    args = make_submit_args(module, tmp_path, "--max-parallel", "2")
    repo = args.repo_root
    monkeypatch.setattr(module, "SBATCH_SCRIPT", repo / "scripts/slurm/job.sbatch")
    monkeypatch.setattr(module, "BATCH_RUNNER", repo / "scripts/tools/runner.py")
    module.SBATCH_SCRIPT.write_text("#!/bin/bash\n", encoding="utf-8")
    module.BATCH_RUNNER.write_text("", encoding="utf-8")

    manifest_path, payload = module.build_job_manifest(args)
    command = module.build_sbatch_command(args, manifest_path, payload)

    assert payload["scheduler_compatibility"] == ["slotd", "slurm"]
    assert payload["prepare_only"] is True
    assert payload["max_parallel"] == 2
    assert payload["jobs"][0]["output_root"] != payload["jobs"][1]["output_root"]
    assert "--prepare-only" in payload["jobs"][0]["command"]
    assert payload["jobs"][0]["command"][0] == str(repo / ".venv" / "bin" / "python")
    assert payload["launcher_python"] == str(repo / ".venv" / "bin" / "python")
    assert command[command.index("--array") + 1] == "0-1%2"
    assert command[command.index("--partition") + 1] == "cpu"
    assert "%A_%a.out" in command[command.index("--output") + 1]
    exported = command[command.index("--export") + 1]
    assert "TAIWAN_LB_ROOT=" in exported
    assert "TAIWAN_EVAL_LAUNCHER_PYTHON=" in exported
    assert "TAIWAN_EVAL_JOB_MANIFEST=" in exported


def test_execute_requires_approval_and_budget_evidence(tmp_path, monkeypatch):
    module = load_module()
    args = make_submit_args(module, tmp_path, "--execute")
    repo = args.repo_root
    monkeypatch.setattr(module, "SBATCH_SCRIPT", repo / "scripts/slurm/job.sbatch")
    monkeypatch.setattr(module, "BATCH_RUNNER", repo / "scripts/tools/runner.py")
    module.SBATCH_SCRIPT.write_text("#!/bin/bash\n", encoding="utf-8")
    module.BATCH_RUNNER.write_text("", encoding="utf-8")

    try:
        module.build_job_manifest(args)
    except ValueError as exc:
        assert "accountability fields" in str(exc)
    else:
        raise AssertionError("external execution should require explicit approval evidence")


def test_cash_cost_exempt_execution_skips_cash_budget_artifacts(
    tmp_path, monkeypatch
):
    module = load_module()
    args = make_submit_args(
        module,
        tmp_path,
        "--execute",
        "--run-purpose",
        "W&B employee inference validation",
        "--wandb-run-id-prefix",
        "tw-wandb-free",
        "--allow-cash-cost-exempt-execution",
    )
    repo = args.repo_root
    (repo / ".env").write_text("WANDB_API_KEY=test\n", encoding="utf-8")
    monkeypatch.setattr(module, "SBATCH_SCRIPT", repo / "scripts/slurm/job.sbatch")
    monkeypatch.setattr(module, "BATCH_RUNNER", repo / "scripts/tools/runner.py")
    module.SBATCH_SCRIPT.write_text("#!/bin/bash\n", encoding="utf-8")
    module.BATCH_RUNNER.write_text("", encoding="utf-8")

    _, payload = module.build_job_manifest(args)

    command = payload["jobs"][0]["command"]
    assert "--allow-cash-cost-exempt-execution" in command
    assert "--pre-run-budget-estimate-json" not in command
    assert "--external-action-approval-report-json" not in command


def test_runner_arg_cannot_override_array_isolation():
    module = load_module()

    try:
        module.validate_runner_args(["--output-root=/tmp/shared"])
    except ValueError as exc:
        assert "--output-root" in str(exc)
    else:
        raise AssertionError("protected runner argument should be rejected")

    try:
        module.validate_runner_args(["--allow-cash-cost-exempt-execution"])
    except ValueError as exc:
        assert "--allow-cash-cost-exempt-execution" in str(exc)
    else:
        raise AssertionError("cash exemption must be managed by the launcher")


def test_task_command_selects_exact_array_entry(tmp_path):
    module = load_module()
    manifest = tmp_path / "job_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repo_root": str(tmp_path),
                "jobs": [
                    {
                        "model_slug": "first",
                        "output_root": str(tmp_path / "first"),
                        "command": ["/bin/echo", "first"],
                    },
                    {
                        "model_slug": "second",
                        "output_root": str(tmp_path / "second"),
                        "command": ["/bin/echo", "second"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    _, job, command = module.task_command(manifest, 1)

    assert job["model_slug"] == "second"
    assert command == ["/bin/echo", "second"]


def test_run_task_records_terminal_status_and_returncode(tmp_path, monkeypatch):
    module = load_module()
    output_root = tmp_path / "runner"
    manifest = tmp_path / "job_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repo_root": str(tmp_path),
                "jobs": [
                    {
                        "model_slug": "model-a",
                        "output_root": str(output_root),
                        "command": ["/bin/false"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    args = module.parse_args(
        [
            "run-task",
            "--job-manifest",
            str(manifest),
            "--index",
            "0",
        ]
    )

    returncode = module.run_task(args)

    assert returncode == 7
    assert calls[0][0] == ["/bin/false"]
    record = json.loads(
        (output_root / "slurm_task.json").read_text(encoding="utf-8")
    )
    assert record["status"] == "failed"
    assert record["returncode"] == 7
    assert record["ended_at"] >= record["started_at"]
