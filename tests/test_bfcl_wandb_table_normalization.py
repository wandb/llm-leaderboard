import importlib.util
import sys
from pathlib import Path

import pandas as pd
import wandb


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
BFCL_ROOT = REPO_ROOT / "scripts" / "evaluator" / "evaluate_utils" / "bfcl_pkg"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(BFCL_ROOT))


def load_bfcl_evaluator_module():
    module_path = SCRIPTS_ROOT / "evaluator" / "bfcl.py"
    spec = importlib.util.spec_from_file_location("leaderboard_bfcl_evaluator", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_bfcl_nested_reasoning_content_is_wandb_table_safe():
    module = load_bfcl_evaluator_module()
    rows = [
        {
            "id": "simple_1",
            "reasoning_content": module.normalize_wandb_table_text("plain"),
        },
        {
            "id": "multi_turn_1",
            "reasoning_content": module.normalize_wandb_table_text(
                [["turn 1 reasoning"], ["turn 2 reasoning", {"tool": "find"}]]
            ),
        },
    ]

    table = wandb.Table(dataframe=pd.DataFrame(rows))

    assert table.columns == ["id", "reasoning_content"]
    assert table.data[1][1].startswith("[[")
    assert "turn 2 reasoning" in table.data[1][1]
