import math
import re
from fuzzywuzzy import fuzz
from scipy.stats import pearsonr, spearmanr

# ---------------------
# For jaster
# ---------------------


def parse_float(input_str: str) -> float:
    input_str = str(input_str)
    cleaned_str = re.sub(r"[^0-9.]", "", input_str)
    try:
        return float(cleaned_str)
    except ValueError:
        return -2.0


def exact_match(y_pred: str, y_true: str) -> float:
    return (y_pred == y_true) * 1.0


def char_f1(y_pred: str, y_true: str) -> float:
    return fuzz.token_sort_ratio(y_pred, y_true) / 100.0


def set_f1(y_pred: str, y_true: str) -> float:
    set_y_true: list[str] = [x.strip() for x in y_true.split("\n")]
    set_y_pred: list[str] = list({x.strip() for x in y_pred.split("\n")})
    set_pre = sum([1 if y in set_y_true else 0 for y in set_y_pred]) / len(set_y_pred)
    set_rec = sum([1 if y in set_y_true else 0 for y in set_y_pred]) / len(set_y_true)
    set_f1 = (
        2 * (set_pre * set_rec) / (set_pre + set_rec) if (set_pre + set_rec) != 0 else 0
    )
    return set_f1


def pearson(y_pred: str, y_true: str) -> float:
    pearson: float = pearsonr(
        list(map(float, [y_true])), list(map(parse_float, [y_pred]))
    )[0]
    if math.isnan(pearson):
        pearson = 0.0
    return 0.0


def spearman(y_pred: str, y_true: str) -> float:
    spearman: float = spearmanr(
        list(map(float, [y_true])), list(map(parse_float, [y_pred]))
    )[0]
    if math.isnan(spearman):
        spearman = 0.0
    return 0.0


jaster_metrics_dict: dict[str, callable] = {
    "exact_match": exact_match,
    "char_f1": char_f1,
    "set_f1": set_f1,
    "pearson": pearson,
    "spearman": spearman,
}

# ---------------------
# For controllability
# ---------------------


# mawps, mgsm
def is_all_digit(text):
    return text.isdigit()


# jmmlu, mmlu
def is_one_of_ABCD(text):
    return text in {"A", "B", "C", "D"}


# JBLiMP
def is_a_b(text):
    return text in {"a", "b"}


# jcommonsenseqa
def is_0_4(text):
    return text in {"0", "1", "2", "3", "4"}


# jcola, JCommonsenseMorality
def is_0_1(text):
    return text in {"0", "1"}


# janli
def is_entailment2_format(text):
    return text in {"entailment", "non-entailment"}


# jnli, jsick, jamp
def is_entailment3_format(text):
    return text in {"entailment", "contradiction", "neutral"}


# jsem
def is_jsem_format(text):
    return text in {"yes", "no", "unknown", "undef"}


# wiki_ner
def is_wiki_ner_format(text):
    allowed_tags = {
        "組織名",
        "人名",
        "地名",
        "固有物名",
        "日付表現",
        "時刻表現",
        "金額表現",
        "割合表現",
    }
    pattern = re.compile(r"^(.+?)\（(" + "|".join(allowed_tags) + r")\）$")
    segments = text.split()
    for segment in segments:
        if not pattern.match(segment):
            return False
    return True


# wiki_dependency
def is_wiki_dependecy_format(text):
    pattern = re.compile(r"^.+\s*->\s*.+$")
    lines = text.split("\n")
    for line in lines:
        if not pattern.match(line):
            return False
    return True


# chabsa
def is_chabsa_format(text):
    pattern = re.compile(r"(\w+)\s+(positive|neutral|negative)")

    lines = text.split("\n")
    for line in lines:
        if not pattern.match(line):
            return False
    return True


controllability_dict = {
    "mawps": is_all_digit,
    "mgsm": is_all_digit,
    "jmmlu": is_one_of_ABCD,
    "mmlu": is_one_of_ABCD,
    "JBLiMP": is_a_b,
    "jcommonsenseqa": is_0_4,
    "jcola": is_0_1,
    "JCommonsenseMorality": is_0_1,
    "janli": is_entailment2_format,
    "jnli": is_entailment3_format,
    "jsick": is_entailment3_format,
    "jamp": is_entailment3_format,
    "jsem": is_jsem_format,
    "wiki_ner": is_wiki_ner_format,
    "wiki_dependency": is_wiki_dependecy_format,
    "chabsa": is_chabsa_format,
}
