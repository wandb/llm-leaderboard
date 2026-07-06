import dataclasses
import json
import re
from typing import Any


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[dict[str, Any]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]
    score: float


SIMPLIFIED_MARKERS = set("汉语国会为这时来对说书买门问开关发后无东车长风电云广乐")
HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def read_prompt_list(json_rows):
    return [
        InputExample(
            key=example["key"],
            instruction_id_list=example["instruction_id_list"],
            prompt=example["prompt"],
            kwargs=example["kwargs"],
        )
        for example in json_rows
    ]


def read_prompt_to_response_dict(json_rows):
    return {example["prompt"]: example["response"] for example in json_rows}


def relation_ok(actual: int, threshold: int, relation: str | None) -> bool:
    relation = relation or "at least"
    if relation in {"at least", "以上", ">=", "不少於"}:
        return actual >= threshold
    if relation in {"less than", "未満", "<", "少於"}:
        return actual < threshold
    if relation in {"exactly", "剛好", "等於", "=="}:
        return actual == threshold
    raise ValueError(f"Unsupported relation: {relation}")


def count_sentences(text: str) -> int:
    return len([part for part in re.split(r"[。！？!?]+", text) if part.strip()])


def count_han(text: str) -> int:
    return len(HAN_RE.findall(text or ""))


def check_number_placeholders(response: str, args: dict[str, Any], _: InputExample) -> bool:
    return len(re.findall(r"\[[^\[\]]+\]", response)) >= int(args["num_placeholders"])


def check_bullet_lists(response: str, args: dict[str, Any], _: InputExample) -> bool:
    bullets = re.findall(r"^\s*[-*•]\s+\S.*$", response, flags=re.MULTILINE)
    return len(bullets) == int(args["num_bullets"])


def check_numbered_lists(response: str, args: dict[str, Any], _: InputExample) -> bool:
    items = re.findall(r"^\s*\d+\.\s+\S.*$", response, flags=re.MULTILINE)
    return len(items) == int(args["num_items"])


def check_highlights(response: str, args: dict[str, Any], _: InputExample) -> bool:
    highlights = [item for item in re.findall(r"《[^《》\n]+》", response) if item.strip("《》").strip()]
    return len(highlights) >= int(args["num_highlights"])


def check_sections(response: str, args: dict[str, Any], _: InputExample) -> bool:
    splitter = re.escape(str(args.get("section_spliter") or "節"))
    sections = re.findall(rf"第[\d一二三四五六七八九十]+{splitter}", response)
    return len(sections) >= int(args["num_sections"])


def check_paragraphs(response: str, args: dict[str, Any], _: InputExample) -> bool:
    paragraphs = [part.strip() for part in re.split(r"\s?\*\*\*\s?", response) if part.strip()]
    return len(paragraphs) == int(args["num_paragraphs"])


def check_postscript(response: str, args: dict[str, Any], _: InputExample) -> bool:
    marker = str(args["postscript_marker"])
    return marker in response and response.rstrip().splitlines()[-1].lstrip().startswith(marker)


def check_keywords(response: str, args: dict[str, Any], _: InputExample) -> bool:
    return all(str(keyword) in response for keyword in args["keywords"])


def check_keyword_frequency(response: str, args: dict[str, Any], _: InputExample) -> bool:
    actual = response.count(str(args["keyword"]))
    return relation_ok(actual, int(args["frequency"]), args.get("relation"))


def check_forbidden_words(response: str, args: dict[str, Any], _: InputExample) -> bool:
    return all(str(word) not in response for word in args["forbidden_words"])


def check_number_sentences(response: str, args: dict[str, Any], _: InputExample) -> bool:
    return relation_ok(count_sentences(response), int(args["num_sentences"]), args.get("relation"))


def check_number_letters(response: str, args: dict[str, Any], _: InputExample) -> bool:
    return relation_ok(count_han(response), int(args["num_letters"]), args.get("relation"))


def check_json_format(response: str, _args: dict[str, Any], _inp: InputExample) -> bool:
    try:
        json.loads(response.strip())
    except Exception:
        return False
    return True


def check_constrained_response(response: str, args: dict[str, Any], _: InputExample) -> bool:
    return response.strip() in {str(option) for option in args["options"]}


def check_end(response: str, args: dict[str, Any], _: InputExample) -> bool:
    return response.strip().endswith(str(args["end_phrase"]))


def check_title(response: str, _args: dict[str, Any], _inp: InputExample) -> bool:
    return bool(re.search(r"《[^《》\n]+》", response))


def check_no_period(response: str, _args: dict[str, Any], _inp: InputExample) -> bool:
    return not re.search(r"[。．.]", response)


def check_no_comma(response: str, _args: dict[str, Any], _inp: InputExample) -> bool:
    return not re.search(r"[，、,]", response)


def check_quotation(response: str, _args: dict[str, Any], _inp: InputExample) -> bool:
    stripped = response.strip()
    return len(stripped) >= 2 and (
        (stripped[0] == "「" and stripped[-1] == "」")
        or (stripped[0] == "『" and stripped[-1] == "』")
    )


def check_language(response: str, _args: dict[str, Any], _inp: InputExample) -> bool:
    han_count = count_han(response)
    if han_count < 5:
        return False
    if re.search(r"[\u3040-\u30ff]", response):
        return False
    simplified = sum(1 for char in response if char in SIMPLIFIED_MARKERS)
    return simplified / han_count <= 0.05


def check_two_responses(response: str, args: dict[str, Any], _: InputExample) -> bool:
    first = str(args.get("first_marker") or "回覆一：")
    second = str(args.get("second_marker") or "回覆二：")
    return first in response and second in response and response.index(first) < response.index(second)


def check_repeat_prompt(response: str, args: dict[str, Any], _: InputExample) -> bool:
    repeated = str(args["prompt_to_repeat"])
    return response.lstrip().startswith(repeated)


CHECKERS = {
    "zh_tw:detectable_content:number_placeholders": check_number_placeholders,
    "zh_tw:detectable_format:number_bullet_lists": check_bullet_lists,
    "zh_tw:detectable_format:number_numbered_lists": check_numbered_lists,
    "zh_tw:detectable_format:number_highlighted_sections": check_highlights,
    "zh_tw:detectable_format:multiple_sections": check_sections,
    "zh_tw:length_constraints:number_paragraphs": check_paragraphs,
    "zh_tw:detectable_content:postscript": check_postscript,
    "zh_tw:keywords:existence": check_keywords,
    "zh_tw:keywords:frequency": check_keyword_frequency,
    "zh_tw:keywords:forbidden_words": check_forbidden_words,
    "zh_tw:length_constraints:number_sentences": check_number_sentences,
    "zh_tw:length_constraints:number_letters": check_number_letters,
    "zh_tw:detectable_format:json_format": check_json_format,
    "zh_tw:detectable_format:constrained_response": check_constrained_response,
    "zh_tw:startend:end_checker": check_end,
    "zh_tw:detectable_format:title": check_title,
    "zh_tw:punctuation:no_period": check_no_period,
    "zh_tw:punctuation:no_comma": check_no_comma,
    "zh_tw:startend:quotation": check_quotation,
    "zh_tw:language:response_language": check_language,
    "zh_tw:combination:two_responses": check_two_responses,
    "zh_tw:combination:repeat_prompt": check_repeat_prompt,
}


def test_instruction_following_strict(inp: InputExample, prompt_to_response: dict[str, str]) -> OutputExample:
    response = prompt_to_response[inp.prompt]
    is_following_list = []
    for index, instruction_id in enumerate(inp.instruction_id_list):
        checker = CHECKERS[instruction_id]
        is_following_list.append(bool(checker(response, inp.kwargs[index], inp)))
    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
        score=sum(is_following_list) / len(is_following_list),
    )
