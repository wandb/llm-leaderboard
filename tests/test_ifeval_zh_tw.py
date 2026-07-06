import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "evaluator" / "evaluate_utils" / "ifeval_zh_tw_utils.py"
SPEC = importlib.util.spec_from_file_location("ifeval_zh_tw_utils", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def score(instruction_id, kwargs, response):
    inp = MODULE.InputExample(
        key=1,
        instruction_id_list=[instruction_id],
        prompt="prompt",
        kwargs=[kwargs],
    )
    return MODULE.test_instruction_following_strict(inp, {"prompt": response}).follow_all_instructions


def test_zh_tw_keyword_and_forbidden_checks():
    assert score("zh_tw:keywords:existence", {"keywords": ["捷運", "悠遊卡"]}, "捷運可以使用悠遊卡。")
    assert not score("zh_tw:keywords:forbidden_words", {"forbidden_words": ["汽車"]}, "汽車很方便。")


def test_zh_tw_format_checks():
    assert score("zh_tw:detectable_format:number_bullet_lists", {"num_bullets": 2}, "- 第一\n- 第二")
    assert score("zh_tw:startend:quotation", {}, "「保持穩定前進。」")
    assert score("zh_tw:detectable_format:json_format", {}, '{"name":"小明"}')


def test_zh_tw_language_rejects_simplified_markers():
    assert score("zh_tw:language:response_language", {"language": "zh_tw"}, "這是一段使用繁體中文撰寫的自然回答。")
    assert not score("zh_tw:language:response_language", {"language": "zh_tw"}, "这是一段使用简体中文书写的回答。")
