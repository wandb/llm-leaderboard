import importlib.util
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
MODULE_PATH = REPO_ROOT / "scripts" / "evaluator" / "twbias.py"
SPEC = importlib.util.spec_from_file_location("twbias", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_default_task_specs_match_upstream_scripts():
    tasks = MODULE.default_task_specs()
    assert len(tasks) == 18
    assert {task["domain"] for task in tasks} == {"gender", "ethnicity"}
    assert any(task["task_id"] == "gender_male_to_female" for task in tasks)
    assert any(task["task_id"] == "ethnicity_B_T1_to_T2" for task in tasks)


def test_build_target_dict_without_combination():
    df = pd.DataFrame({"T1": ["男人", "男性"], "T2": ["女人", "女性"]})
    target_dict = MODULE.build_target_dict(df, "T1", "T2", False)
    assert target_dict == {"男人": ["女人"], "男性": ["女性"]}


def test_build_target_dict_with_combination():
    df = pd.DataFrame({"T1": ["本省", "本省人"], "T2": ["外省", "外省人"]})
    target_dict = MODULE.build_target_dict(df, "T1", "T2", True)
    assert target_dict["本省"] == ["外省", "外省人"]
    assert target_dict["本省人"] == ["外省", "外省人"]


def test_replace_target_words_deduplicates():
    replacements = MODULE.replace_target_words("男人和男性都在這裡", {"男人": ["女人"], "男性": ["女性", "女性"]})
    assert replacements == ["女人和男性都在這裡", "男人和女性都在這裡"]


def test_summarize_outputs_unbiased_response_rate():
    output_df = pd.DataFrame(
        [
            {
                "domain": "gender",
                "task_id": "a",
                "prompt_id": "0",
                "status": "ok",
                "origin_ppl": 10.0,
                "replace_ppl": 20.0,
                "delta_ppl": 10.0,
                "delta_ratio": 1.0,
                "biased_preference": True,
            },
            {
                "domain": "gender",
                "task_id": "a",
                "prompt_id": "0",
                "status": "ok",
                "origin_ppl": 20.0,
                "replace_ppl": 10.0,
                "delta_ppl": -10.0,
                "delta_ratio": -0.5,
                "biased_preference": False,
            },
            {
                "domain": "gender",
                "task_id": "a",
                "prompt_id": "0",
                "status": "no_replacement",
                "origin_ppl": None,
                "replace_ppl": None,
                "delta_ppl": None,
                "delta_ratio": None,
                "biased_preference": None,
            },
        ]
    )
    leaderboard, task_table = MODULE.summarize_outputs(output_df, "model-x")
    assert float(leaderboard["unbiased_response_rate"].iloc[0]) == 0.5
    assert float(leaderboard["bias_preference_rate"].iloc[0]) == 0.5
    assert int(leaderboard["valid_count"].iloc[0]) == 2
    assert int(leaderboard["skipped_count"].iloc[0]) == 1
    assert float(task_table["unbiased_response_rate"].iloc[0]) == 0.5
