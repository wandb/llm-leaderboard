import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_output_scoping import apply_run_scoped_outputs


def test_taiwan_full_outputs_are_scoped_to_wandb_run_id():
    cfg = {
        "wandb": {"project": "tc-leaderboard"},
        "run": {
            "aggregate_taiwan": True,
            "agentic_math": True,
            "swebench_pro": True,
            "bfcl": True,
        },
        "agentic_math": {
            "output_dir": "outputs/old/agentic_math",
            "results_dir": "outputs/old/agentic_math/openclaw",
            "run_openclaw": True,
        },
        "swebench_pro": {
            "output_dir": "outputs/old/swebench_pro",
            "checkout_root": "outputs/old/swebench_pro_checkouts",
            "patch_path": "outputs/old/swebench_pro/openclaw/patches.json",
            "run_openclaw": True,
        },
        "bfcl": {
            "allow_overwrite": False,
        },
    }

    resolved, record = apply_run_scoped_outputs(cfg, run_id="abc123")

    assert record == {
        "applied": True,
        "run_id": "abc123",
        "run_root": "outputs/taiwan_full_eval_runs/abc123",
    }
    assert resolved["output"]["run_scoped"] is True
    assert resolved["output"]["resolved_run_root"] == "outputs/taiwan_full_eval_runs/abc123"
    assert resolved["agentic_math"]["output_dir"] == "outputs/taiwan_full_eval_runs/abc123/agentic_math"
    assert resolved["agentic_math"]["results_dir"] is None
    assert resolved["swebench_pro"]["output_dir"] == "outputs/taiwan_full_eval_runs/abc123/swebench_pro"
    assert (
        resolved["swebench_pro"]["checkout_root"]
        == "outputs/taiwan_full_eval_runs/abc123/swebench_pro_checkouts"
    )
    assert resolved["swebench_pro"]["patch_path"] is None
    assert resolved["bfcl"]["result_dir"] == "outputs/taiwan_full_eval_runs/abc123/bfcl/result"
    assert resolved["bfcl"]["score_dir"] == "outputs/taiwan_full_eval_runs/abc123/bfcl/score"
    assert resolved["bfcl"]["allow_overwrite"] is True


def test_non_taiwan_config_is_not_scoped_by_default():
    cfg = {
        "wandb": {"project": "nejumi-leaderboard4"},
        "run": {"aggregate_taiwan": False, "agentic_math": True},
        "agentic_math": {"output_dir": "outputs/agentic_math"},
    }

    resolved, record = apply_run_scoped_outputs(cfg, run_id="abc123")

    assert record["applied"] is False
    assert resolved == cfg


def test_explicit_output_root_template_is_supported():
    cfg = {
        "wandb": {"project": "tc-leaderboard"},
        "output": {"run_scoped": True, "root": "outputs/custom/{run_id}"},
        "run": {"agentic_math": True},
        "agentic_math": {"output_dir": "outputs/old"},
    }

    resolved, record = apply_run_scoped_outputs(cfg, run_id="run id/with spaces")

    assert record["run_root"] == "outputs/custom/run-id-with-spaces"
    assert resolved["agentic_math"]["output_dir"] == "outputs/custom/run-id-with-spaces/agentic_math"
