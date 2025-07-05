import math
import re
from fuzzywuzzy import fuzz
from scipy.stats import pearsonr, spearmanr
from statistics import mean
from sacrebleu import BLEU
import textwrap
from jinja2 import Template
import ast
import subprocess
from pathlib import Path
from .sandbox_client import CodeExecutor, is_sandbox_running
import logging

logger = logging.getLogger(__name__)

#import bert_score
import shutil
from comet import download_model, load_from_checkpoint
from abc import ABC, abstractmethod

# ---------------------
# For jaster
# ---------------------


class BaseMetric(ABC):
    name: str
    
    @abstractmethod
    def __call__(self, y_pred: str, y_true: str, **kwargs) -> float:
        pass


def parse_float(input_str: str) -> float:
    input_str = str(input_str)
    cleaned_str = re.sub(r"[^0-9.]", "", input_str)
    try:
        return float(cleaned_str)
    except ValueError:
        return -2.0


def exact_match(y_pred: str, y_true: str) -> float:
    return (y_pred == y_true) * 1.0


class ExactMatchMetric(BaseMetric):
    name = "exact_match"

    def __call__(self, y_pred: str, y_true: str, **kwargs) -> float:
        from sklearn.metrics import accuracy_score
        return accuracy_score([y_true], [y_pred])


def exact_match_figure(y_pred: str, y_true: str) -> float:
    try:
        return (float(y_pred) == float(y_true)) * 1.0
    except ValueError:
        return 0.0


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

"""
def bert_score_en_f1(y_pred:str, y_true:str) -> float:
    return bert_score.score([y_pred], [y_true], lang="en")[2].tolist()[0] #[2]=f1

def bert_score_ja_f1(y_pred:str, y_true:str) -> float:
    return bert_score.score([y_pred], [y_true], lang="ja")[2].tolist()[0] #[2]=f1
"""

def comet_wmt22(): #this is fake func
    pass

def commet_score(commet_srcs, commet_mt, commet_ref):
    print("--------downloading comet model to evaluate translation task--------")
    comet_model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_model_path)
    comet_data = []
    for i in range(len(commet_srcs)):
        comet_instance = {}
        comet_instance["src"] = commet_srcs[i]
        comet_instance["mt"] = commet_mt[i]
        comet_instance["ref"] = commet_ref[i]
        comet_data.append(comet_instance)
    scores = comet_model.predict(comet_data, batch_size=8, gpus=1, progress_bar=False).scores
    #delete_model_directory(comet_model_path)
    return scores
    

class CodeExecMetricWithSandbox(BaseMetric):
    name = "code_exec_sandbox"
    code_template = textwrap.dedent("""
    {{ code}}
    {% for testcase in testcases %}
    {{ testcase }}
    {% endfor %}
    """)

    def __call__(self, y_preds: list[str], y_trues: list[str], **kwargs) -> float:
        if not is_sandbox_running():
            logger.warning("Sandbox is not running. Skipping code execution.")
            return 0.0

        scores = []
        for pred, true in zip(y_preds, y_trues, strict=False):
            testcases = ast.literal_eval(true)
            template = Template(self.code_template)
            scores.append(CodeExecutor.check_all_passed(code=template.render(code=pred, testcases=testcases)) * 1.0)
        return mean(scores)


class PylintCheckMetric(BaseMetric):
    name = "pylint_check"

    def __call__(self, y_preds: list[str], y_trues: list[str], **kwargs) -> float:
        scores = []
        tmp_path = Path("pylint.test.tmp.py")

        for pred in y_preds:
            try:
                tmp_path.write_text(pred)
                result = (
                    subprocess.run(
                        f"pylint --rc-file=pylintrc {tmp_path}",
                        stdout=subprocess.PIPE,
                        shell=True,
                    )
                    .stdout.decode("utf-8")
                    .split("\n")[1:-1]
                )

                is_valid = not any("E" in line.split()[1] for line in result if line and line.split())
                scores.append(1.0 if is_valid else 0.0)

            except Exception:
                scores.append(0.0)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

        return mean(scores)



def delete_model_directory(directory):
    """Deletes the specified directory."""
    try:
        shutil.rmtree(directory)
        print(f"The directory containing the model at {directory} has been successfully deleted.")
    except Exception as e:
        print(f"An error occurred while trying to delete the directory: {e}")


jaster_metrics_dict: dict[str, callable] = {
    "exact_match": ExactMatchMetric(),
    "exact_match_figure": exact_match_figure,
    "char_f1": char_f1,
    "set_f1": set_f1,
    "pearson": pearson,
    "spearman": spearman,
    "bleu_ja": blue_ja,
    "bleu_en": blue_en,
    #"bert_score_en_f1": bert_score_en_f1,
    #"bert_score_ja_f1": bert_score_ja_f1,
    "comet_wmt22": comet_wmt22,
    "code_exec_sandbox": CodeExecMetricWithSandbox(),
    "pylint_check": PylintCheckMetric(),
}

task_to_sub_category = {
    "alt-e-to-j": "GLP_translation",
    "alt-j-to-e": "GLP_translation",
    "jsquad": "GLP_information_extraction",
    "mawps": "GLP_mathematical_reasoning",
    "jcommonsenseqa": "GLP_knowledge_QA",
    "jemhopqa": "GLP_knowledge_QA",
    "jmmlu": "GLP_knowledge_QA",
    "niilc": "GLP_knowledge_QA",
    "aio": "GLP_knowledge_QA",
    "jnli": "GLP_semantic_analysis",
    "janli": "GLP_semantic_analysis",
    "jsem": "GLP_semantic_analysis",
    "jsick": "GLP_semantic_analysis",
    "jamp": "GLP_semantic_analysis",
    "jcola-in-domain": "GLP_syntactic_analysis",
    "jcola-out-of-domain": "GLP_syntactic_analysis",
    "jblimp": "GLP_syntactic_analysis",
    "mmlu_prox_ja": "GLP_knowledge_QA",
    "jmmlu": "GLP_knowledge_QA",
    "commonsensemoralja": "ALT_ethics_moral",
    "toxicity": "ALT_toxicity",
    "humanities": "GLP_expression",
    "roleplay": "GLP_expression",
    "writing": "GLP_expression",
    "reasoning": "GLP_reasoning",
    "math": "GLP_mathematical_reasoning",
    "mgsm": "GLP_mathematical_reasoning",
    "extraction": "GLP_entity_extraction",
    "stem": "GLP_knowledge_QA",
    "coding": "ADVANCED_programing",
    "jhumaneval": "ADVANCED_programing",
}

# ---------------------
# For controllability
# ---------------------
# mawps, mgsm
def is_all_digit(text: str) -> int:
    try:
        float(text)
        return 1
    except ValueError:
        return 0

# jmmlu, mmlu_prox_ja
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

# no_check
def no_check(text: str):
    return None

controllability_dict = {
    "aio": no_check,
    "alt-e-to-j": no_check,
    "alt-j-to-e": no_check,
    "commonsensemoralja": is_0_1,
    "jamp": is_entailment3_format,
    "janli": is_entailment2_format,
    "jblimp": is_a_b,
    "jcola-in-domain": is_0_1,
    "jcola-out-of-domain": is_0_1,
    "jcommonsenseqa": is_0_4,
    "jemhopqa": no_check,
    "jhumaneval": no_check,
    "jnli": is_entailment3_format,
    "jsem": is_jsem_format,
    "jsick": is_entailment3_format,
    "jsquad": no_check,
    "jmmlu": is_one_of_ABCD,
    "mmlu_prox_ja": is_one_of_ABCD,
    "mmlu_en": is_one_of_ABCD,
    "kuci": is_0_3,
    "mawps": is_all_digit,
    "mgsm": is_all_digit,
    "niilc": no_check,
}
