import importlib.util
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

PREPARE_PATH = REPO_ROOT / "scripts" / "data_uploader" / "prepare_hallulens_zh_tw.py"
PREPARE_SPEC = importlib.util.spec_from_file_location("prepare_hallulens_zh_tw", PREPARE_PATH)
PREPARE = importlib.util.module_from_spec(PREPARE_SPEC)
assert PREPARE_SPEC.loader is not None
PREPARE_SPEC.loader.exec_module(PREPARE)

_existing_evaluator = sys.modules.get("evaluator")
_existing_evaluate_utils = sys.modules.get("evaluator.evaluate_utils")
if _existing_evaluator is None:
    evaluator_pkg = types.ModuleType("evaluator")
    evaluator_pkg.__path__ = [str(REPO_ROOT / "scripts" / "evaluator")]
    sys.modules["evaluator"] = evaluator_pkg
if _existing_evaluate_utils is None:
    evaluate_utils_stub = types.ModuleType("evaluator.evaluate_utils")
    evaluate_utils_stub.__path__ = [
        str(REPO_ROOT / "scripts" / "evaluator" / "evaluate_utils")
    ]
    evaluate_utils_stub.LLMAsyncProcessor = object
    evaluate_utils_stub.get_openai_judge_client = lambda *args, **kwargs: None
    sys.modules["evaluator.evaluate_utils"] = evaluate_utils_stub

HALLULENS_PATH = REPO_ROOT / "scripts" / "evaluator" / "hallulens.py"
HALLULENS_SPEC = importlib.util.spec_from_file_location("evaluator.hallulens", HALLULENS_PATH)
HALLULENS = importlib.util.module_from_spec(HALLULENS_SPEC)
assert HALLULENS_SPEC.loader is not None
HALLULENS_SPEC.loader.exec_module(HALLULENS)

# The stub is an import aid for this module only. Leaving it in sys.modules
# makes unrelated evaluator tests depend on collection order.
if _existing_evaluate_utils is None:
    sys.modules.pop("evaluator.evaluate_utils", None)
else:
    sys.modules["evaluator.evaluate_utils"] = _existing_evaluate_utils
if _existing_evaluator is None:
    sys.modules.pop("evaluator", None)
else:
    sys.modules["evaluator"] = _existing_evaluator


def test_quoted_name_from_prompt_prefers_display_name():
    prompt = "フリーゼンハイムにある「HopfenLounge」というバーについて詳しく教えていただけますか？"
    assert PREPARE.quoted_name_from_prompt(prompt) == "HopfenLounge"


def test_prompt_for_row_uses_traditional_chinese_template():
    assert PREPARE.prompt_for_row("台北", "酒吧", "HopfenLounge") == "可以詳細介紹位於台北的「HopfenLounge」這間酒吧嗎？"


def test_transform_row_preserves_originals_and_maps_type():
    row = {
        "place": "Taipei",
        "type_": "バー",
        "name": "ホップフェンラウンジ",
        "prompt": "Taipeiにある「HopfenLounge」というバーについて詳しく教えていただけますか？",
    }
    transformed = PREPARE.transform_row(row, {}, "model-x")
    assert transformed["type_"] == "酒吧"
    assert transformed["name"] == "HopfenLounge"
    assert transformed["type_original"] == "バー"
    assert transformed["prompt_original"] == row["prompt"]


def test_hallulens_template_keeps_json_braces_literal():
    rendered = HALLULENS.render_judge_prompt_template(
        'Return JSON: {{"does_believe": true}} name={name} text={generation}',
        name="測試店",
        generation="沒有資料",
    )
    assert rendered == 'Return JSON: {"does_believe": true} name=測試店 text=沒有資料'
