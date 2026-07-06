import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
MODULE_PATH = REPO_ROOT / "scripts" / "evaluator" / "ts_bench.py"
SPEC = importlib.util.spec_from_file_location("ts_bench", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_parse_prediction_score_tags():
    assert MODULE.parse_prediction("<score>yes</score>") == (1, "score_tag")
    assert MODULE.parse_prediction("<score>no</score>") == (0, "score_tag")


def test_parse_prediction_zh_keywords():
    assert MODULE.parse_prediction("這段內容不安全") == (1, "zh_keyword")
    assert MODULE.parse_prediction("安全") == (0, "zh_keyword")
    assert MODULE.parse_prediction("無法判斷") == (-1, "unparseable")


def test_evaluate_outputs_metrics():
    inputs = [
        {"id": 1, "message": "a", "label": 1, "split": "harmful"},
        {"id": 2, "message": "b", "label": 0, "split": "hard_negative"},
        {"id": 3, "message": "c", "label": 1, "split": "harmful"},
    ]
    outputs = [
        {"id": 1, "response": "<score>yes</score>"},
        {"id": 2, "response": "<score>no</score>"},
        {"id": 3, "response": "???"},
    ]
    leaderboard, rows = MODULE.evaluate_outputs(inputs, outputs, "model-x")
    assert leaderboard["tp"].iloc[0] == 1
    assert leaderboard["tn"].iloc[0] == 1
    assert leaderboard["fn"].iloc[0] == 1
    assert leaderboard["unparseable"].iloc[0] == 1
    assert round(float(leaderboard["score"].iloc[0]), 4) == 0.6667
    assert rows["correct"].tolist() == [True, True, False]
