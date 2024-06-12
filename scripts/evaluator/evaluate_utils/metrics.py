import math
import re
from fuzzywuzzy import fuzz
from scipy.stats import pearsonr, spearmanr
from sacrebleu import BLEU
import bert_score

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

def blue_en(y_pred: str, y_true: str) -> float:
    y_pred = y_pred.strip()
    y_true = y_true.strip()
    
    if not y_true:
        raise ValueError("The reference text (y_true) is empty.")    
    if not y_pred:
        return 0.0
    bleu_config = {"effective_order": True, "trg_lang": "en"}

    bleu_score = BLEU(**bleu_config).corpus_score([y_pred], [[y_true]]).score
    return bleu_score/100

def blue_ja(y_pred: str, y_true: str) -> float:
    y_pred = y_pred.strip()
    y_true = y_true.strip()
    
    if not y_true:
        raise ValueError("The reference text (y_true) is empty.")    
    if not y_pred:
        return 0.0
    bleu_config = {"effective_order": True, "trg_lang": "ja"}

    bleu_score = BLEU(**bleu_config).corpus_score([y_pred], [[y_true]]).score
    return bleu_score/100

def bert_score_en_f1(y_pred:str, y_true:str) -> float:
    return bert_score.score([y_pred], [y_true], lang="en")[2].tolist()[0] #[2]=f1

def bert_score_ja_f1(y_pred:str, y_true:str) -> float:
    return bert_score.score([y_pred], [y_true], lang="ja")[2].tolist()[0] #[2]=f1


jaster_metrics_dict: dict[str, callable] = {
    "exact_match": exact_match,
    "char_f1": char_f1,
    "set_f1": set_f1,
    "pearson": pearson,
    "spearman": spearman,
    "bleu_ja": blue_ja,
    "bleu_en": blue_en,
    "bert_score_en_f1": bert_score_en_f1,
    "bert_score_ja_f1": bert_score_ja_f1
}

jmmlu_dict = {
    'jmmlu_abstract_algebra': 'jmmlu_stem',
    'jmmlu_anatomy': 'jmmlu_stem',
    'jmmlu_astronomy': 'jmmlu_stem',
    'jmmlu_business_ethics': 'jmmlu_other',
    'jmmlu_clinical_knowledge': 'jmmlu_other',
    'jmmlu_college_biology': 'jmmlu_stem',
    'jmmlu_college_chemistry': 'jmmlu_stem',
    'jmmlu_college_computer_science': 'jmmlu_stem',
    'jmmlu_college_mathematics': 'jmmlu_stem',
    'jmmlu_college_medicine': 'jmmlu_other',
    'jmmlu_college_physics': 'jmmlu_stem',
    'jmmlu_computer_security': 'jmmlu_stem',
    'jmmlu_conceptual_physics': 'jmmlu_stem',
    'jmmlu_econometrics': 'jmmlu_social_sciences',
    'jmmlu_electrical_engineering': 'jmmlu_stem',
    'jmmlu_elementary_mathematics': 'jmmlu_stem',
    'jmmlu_formal_logic': 'jmmlu_humanities',
    'jmmlu_global_facts': 'jmmlu_other',
    'jmmlu_high_school_biology': 'jmmlu_stem',
    'jmmlu_high_school_chemistry': 'jmmlu_stem',
    'jmmlu_high_school_computer_science': 'jmmlu_stem',
    'jmmlu_high_school_european_history': 'jmmlu_humanities',
    'jmmlu_high_school_geography': 'jmmlu_social_sciences',
    'jmmlu_high_school_macroeconomics': 'jmmlu_social_sciences',
    'jmmlu_high_school_mathematics': 'jmmlu_stem',
    'jmmlu_high_school_microeconomics': 'jmmlu_social_sciences',
    'jmmlu_high_school_physics': 'jmmlu_stem',
    'jmmlu_high_school_psychology': 'jmmlu_social_sciences',
    'jmmlu_high_school_statistics': 'jmmlu_stem',
    'jmmlu_human_aging': 'jmmlu_other',
    'jmmlu_human_sexuality': 'jmmlu_social_sciences',
    'jmmlu_international_law': 'jmmlu_humanities',
    'jmmlu_japanese_history': 'jmmlu_humanities',
    'jmmlu_jurisprudence': 'jmmlu_humanities',
    'jmmlu_logical_fallacies': 'jmmlu_humanities',
    'jmmlu_machine_learning': 'jmmlu_stem',
    'jmmlu_management': 'jmmlu_other',
    'jmmlu_marketing': 'jmmlu_other',
    'jmmlu_medical_genetics': 'jmmlu_other',
    'jmmlu_miscellaneous': 'jmmlu_other',
    'jmmlu_moral_disputes': 'jmmlu_humanities',
    'jmmlu_nutrition': 'jmmlu_other',
    'jmmlu_philosophy': 'jmmlu_humanities',
    'jmmlu_prehistory': 'jmmlu_humanities',
    'jmmlu_professional_accounting': 'jmmlu_other',
    'jmmlu_professional_medicine': 'jmmlu_other',
    'jmmlu_professional_psychology': 'jmmlu_social_sciences',
    'jmmlu_public_relations': 'jmmlu_social_sciences',
    'jmmlu_security_studies': 'jmmlu_social_sciences',
    'jmmlu_sociology': 'jmmlu_social_sciences',
    'jmmlu_virology': 'jmmlu_other',
    'jmmlu_world_history': 'jmmlu_humanities',
    'jmmlu_world_religions': 'jmmlu_humanities'
}


# ---------------------
# For controllability
# ---------------------
# mawps, mgsm
def is_all_digit(text: str) -> int:
    return 1 if text.isdigit() else 0

# jmmlu, mmlu
def is_one_of_ABCD(text: str) -> int:
    return 1 if text in {"A", "B", "C", "D"} else 0

# JBLiMP
def is_a_b(text: str) -> int:
    return 1 if text in {"a", "b"} else 0

# jcommonsenseqa
def is_0_4(text: str) -> int:
    return 1 if text in {"0", "1", "2", "3", "4"} else 0

# kuci
def is_0_3(text: str) -> int:
    return 1 if text in {"0", "1", "2", "3"} else 0

# jcola, JCommonsenseMorality
def is_0_1(text: str) -> int:
    return 1 if text in {"0", "1"} else 0

# janli
def is_entailment2_format(text: str) -> int:
    return 1 if text in {"entailment", "non-entailment"} else 0

# jnli, jsick, jamp
def is_entailment3_format(text: str) -> int:
    return 1 if text in {"entailment", "contradiction", "neutral"} else 0

# jsem
def is_jsem_format(text: str) -> int:
    return 1 if text in {"yes", "no", "unknown", "undef"} else 0

# wiki_ner
def is_wiki_ner_format(text: str) -> int:
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
            return 0
    return 1

# wiki_dependency
def is_wiki_dependecy_format(text: str) -> int:
    pattern = re.compile(r"^.+\s*->\s*.+$")
    lines = text.split("\n")
    for line in lines:
        if not pattern.match(line):
            return 0
    return 1

# chabsa
def is_chabsa_format(text: str) -> int:
    pattern = re.compile(r"(\w+)\s+(positive|neutral|negative)")

    lines = text.split("\n")
    for line in lines:
        if not pattern.match(line):
            return 0
    return 1

# no_check
def no_check(text: str):
    return None

controllability_dict = {
    "aio": no_check,
    "alt-e-to-j": no_check,
    "alt-j-to-e": no_check,
    "chabsa": is_chabsa_format,
    "commonsensemoralja": is_0_1,
    "jamp": is_entailment3_format,
    "janli": is_entailment2_format,
    "jblimp": is_a_b,
    "jcola-in-domain": is_0_1,
    "jcola-out-of-domain": is_0_1,
    "jcommonsenseqa": is_0_4,
    "jemhopqa": no_check,
    "jnli": is_entailment3_format,
    "jsem": is_jsem_format,
    "jsick": is_entailment3_format,
    "jsquad": no_check,
    "jmmlu": is_one_of_ABCD,
    "mmlu_en": is_one_of_ABCD,
    "kuci": is_0_3,
    "mawps": is_all_digit,
    "mgsm": is_all_digit,
    "niilc": no_check,
    "wiki_coreference": no_check,
    "wiki_dependency": is_wiki_dependecy_format,
    "wiki_ner": is_wiki_ner_format,
    "wiki_pas": no_check,
    "wiki_reading": no_check,
    "wikicorpus-e-to-j": no_check,
    "wikicorpus-j-to-e": no_check,
}
