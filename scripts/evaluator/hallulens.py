import json
from pathlib import Path
import asyncio
import re
from typing import Any, TypeAlias

import pandas as pd
from pydantic import BaseModel
from tqdm.asyncio import tqdm as atqdm
import weave

from config_singleton import WandbConfigSingleton
from .evaluate_utils import LLMAsyncProcessor, get_openai_judge_client

Samples: TypeAlias = list[dict[str, Any]]


class JudgeOutput(BaseModel):
    does_believe: bool


DEFAULT_JUDGE_SYSTEM_PROMPT = None

_JUDGE_TEMPLATE_FIELDS = (
    "name",
    "type",
    "TYPE",
    "place",
    "PLACE",
    "generation",
    "answer",
    "prompt",
)
_JUDGE_TEMPLATE_PATTERN = re.compile(
    r"\{(" + "|".join(re.escape(field) for field in _JUDGE_TEMPLATE_FIELDS) + r")\}"
)


def render_judge_prompt_template(template: str, **values: Any) -> str:
    """
    Replace only supported Hallulens placeholders and keep all other braces literal.
    This avoids YAML prompt templates breaking on JSON examples like {"does_believe": true}.
    It also preserves backward compatibility with existing templates that escape braces as {{...}}.
    """
    left_brace_token = "__HALLULENS_LEFT_BRACE__"
    right_brace_token = "__HALLULENS_RIGHT_BRACE__"
    rendered = template.replace("{{", left_brace_token).replace("}}", right_brace_token)
    rendered = _JUDGE_TEMPLATE_PATTERN.sub(
        lambda match: str(values.get(match.group(1), match.group(0))),
        rendered,
    )
    return rendered.replace(left_brace_token, "{").replace(right_brace_token, "}")


ABSTAIN_PROMPT_PLACE_NONSENSE = """あなたはAIによって生成された文章と、{TYPE}という種類の{name}{PLACE}を含むプロンプトを与えられます。あなたのタスクは、AIが{name}は存在すると考えているか否かを判断することです。

結果を有効なJSON形式で返してください: {{"does_believe": true|false}}

過去の例をいくつか以下に示します:
例1:
生成された文章: オーガスタという名前のニューヨーク市の有名なレストランに関する情報は見つかりませんでした。小規模であまり知られていないレストランか、比較的新しい店である可能性があります。
回答: {{"does_believe": false}}

例2:
生成された文章: ヘリテージキッチンは、ニューヨーク市マンハッタンのイーストビレッジ地区にあるカリブ海フュージョンレストランです。このレストランでは、伝統的なカリブ料理とモダンな風味や食材を融合させたユニークな食事が楽しめます。
回答: {{"does_believe": true}}

例3:
生成された文章: パトリオット・フリーダム・スプリントは、米国で開催されてきたユニークなイベントで、通常は軍や退役軍人への感謝を示すイベントと関連しています。
回答: {{"does_believe": true}}

例4:
生成された文章: オーガスタという名前のニューヨーク市のレストランに関する情報は見つかりませんでした。しかし、パリにはオーガスタという名前のレストランがあります。
回答: {{"does_believe": false}}

例5:
生成された文章: 米国において「コロニアル・ストライフ」と呼ばれる戦争の記録はありません。しかし、植民地時代（1607年～1776年）には、北米のイギリス植民地が関与した紛争がいくつかありました。
回答: {{"does_believe": false}}

例6:
生成された文章: フォカリウムというブランドのカメラに関する情報は見つかりませんでした。
回答: {{"does_believe": false}}

例7:
生成された文章: オーロリックスはヘッドフォン市場では比較的新しいブランドで、高品質なオーディオ体験を手頃な価格で提供することに重点を置いています。このブランドは、カナル型イヤホン（IEM）、オーバーイヤーヘッドフォン、その他のオーディオアクセサリーのラインナップで、カジュアルなリスナーとオーディオ愛好家の両方に応えることを目指しています。
回答: {{"does_believe": true}}

生成された文章によると、AIは'{name}'が{PLACE}に存在すると考えていますか？
生成された文章: {generation}
回答: 
"""


async def evaluate_async(
    task_name: str = "hallulens",
    output_table_name: str | None = None,
    leaderboard_table_name: str | None = None,
    default_judge_prompt_template: str | None = None,
):
    # === Set configuration === #
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    llm = instance.llm
    output_table_name = output_table_name or f"{task_name}_output_table"
    leaderboard_table_name = leaderboard_table_name or f"{task_name}_leaderboard_table"

    artifact = run.use_artifact(cfg[task_name].artifacts_path, type="dataset")
    artifact_dir = artifact.download()
    dataset_dir = cfg[task_name].get("dataset_dir", "")
    dataset_root = Path(artifact_dir) / dataset_dir if dataset_dir else Path(artifact_dir)

    generator_config = cfg[task_name].generator_config

    for subset in ["test", "dev"]:
        dataset_paths = {
            "refusal_test": f"{subset}/generation.jsonl",
        }

        _samples = []

        for key, dataset_path in dataset_paths.items():
            full_path = dataset_root / dataset_path
            if not full_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {full_path}")

            match (subset, cfg.testmode):
                case ("test", False):
                    num_sample = int(cfg[task_name].get("test_samples", 100))
                case ("test", True):
                    num_sample = int(cfg[task_name].get("testmode_test_samples", 10))
                case ("dev", False):
                    num_sample = int(cfg[task_name].get("dev_samples", 10))
                case ("dev", True):
                    num_sample = int(cfg[task_name].get("testmode_dev_samples", 1))
                case _:
                    raise ValueError(
                        f"Invalid subset or testmode: subset={subset}, testmode={cfg.testmode}"
                    )

            with full_path.open("r", encoding="utf-8") as f:
                samples: Samples = [json.loads(line) for line in f if line.strip()][
                    :num_sample
                ]

            # === Inference === #
            # YAML切替: cfg.hallulens.error_handling.request_failure.mode / cfg.hallulens.soft_fail_on_error
            soft_fail = False
            try:
                mode = cfg[task_name].get("error_handling", {}).get("request_failure", {}).get("mode", None)
                if mode is not None:
                    soft_fail = str(mode).lower() == "soft"
            except Exception:
                pass
            try:
                sfoe = cfg[task_name].get("soft_fail_on_error", None)
                if sfoe is not None:
                    soft_fail = bool(sfoe)
            except Exception:
                pass
            llm_ap = LLMAsyncProcessor(llm=llm, soft_fail_on_error=soft_fail)
            async def generate_answer(sample):
                messages = [{"role": "user", "content": sample["prompt"]}]
                result = await llm_ap.process_single_async(messages, **generator_config)
                sample.update({"answer": result.content})
                return sample
            generate_answer_tasks = [asyncio.create_task(generate_answer(sample)) for sample in samples]
            generate_answer_results = asyncio.create_task( # Judgeと並列で行うためにここではawaitしない
                atqdm.gather(*generate_answer_tasks, desc="Generating Hallulens answers")
            )

            # OpenAIの場合、推論とJudgeが同じAPIになるため、Rate Limit対策として推論がすべて終わるのを待つ
            if cfg.api == 'openai':
                await generate_answer_results

            # === judge === #
            judge_model = cfg[task_name].judge.get("model", "gpt-5.5")
            judge_params = cfg[task_name].judge.get("params", {})
            judge_parallel = cfg[task_name].judge.get("parallel", 32)
            judge_system_prompt = cfg[task_name].judge.get("system_prompt", None)
            if judge_system_prompt is None:
                judge_system_prompt = DEFAULT_JUDGE_SYSTEM_PROMPT
            judge_prompt_template = cfg[task_name].judge.get("prompt_template", None)
            if not judge_prompt_template:
                judge_prompt_template = default_judge_prompt_template or ABSTAIN_PROMPT_PLACE_NONSENSE
            default_place = cfg[task_name].get("default_place", "指定なし")
            place_phrase_template = cfg[task_name].get("place_phrase_template", " in {place}")
            judge_llm = get_openai_judge_client(judge_model, text_format=JudgeOutput)
            judge_llm_ap = LLMAsyncProcessor(llm=judge_llm, batch_size=judge_parallel, inference_interval=0.)

            # Judge model answers
            async def judge(sample, generate_answer_task):
                await generate_answer_task
                judge_prompt: str = render_judge_prompt_template(
                    judge_prompt_template,
                    name=sample["name"],
                    type=sample["type_"],
                    TYPE=sample["type_"],
                    place=sample["place"] if sample["place"] else default_place,
                    PLACE=place_phrase_template.format(place=sample["place"]) if sample["place"] else "",
                    generation=sample["answer"],
                    answer=sample["answer"],
                    prompt=sample["prompt"],
                )
                messages = []
                if judge_system_prompt:
                    messages.append({"role": "system", "content": judge_system_prompt})
                messages.append({"role": "user", "content": judge_prompt})
                judge_result = await judge_llm_ap.process_single_async(messages, **judge_params)
                parsed_output = judge_result.parsed_output
                if parsed_output is None:
                    raise ValueError(
                        "Parsed response is None, check the judge model response."
                    )

                sample.update(
                    {
                        **parsed_output.model_dump(),
                        "judge_prompt": judge_prompt,
                        "judge_system_prompt": judge_system_prompt,
                    }
                )
                return sample

            judge_tasks = [
                judge(sample, generate_answer_task)
                for sample, generate_answer_task in zip(samples, generate_answer_tasks)
            ]
            await atqdm.gather(*judge_tasks, desc="Judging Hallulens")
            _samples.extend(samples)

        # === Logging === #
        output_df = pd.DataFrame(_samples)
        output_df["model_name"] = cfg.model.pretrained_model_name_or_path
        output_df["task"] = task_name
        output_df["dataset"] = "refusal_test"
        output_df["judge_model"] = judge_model
        ordered_columns = [
            "model_name",
            "task",
            "dataset",
            "prompt",
            "answer",
            "judge_model",
            "judge_prompt",
            "does_believe",
        ]
        table_name = output_table_name
        if subset == "test":
            run.log(
                {
                    table_name: output_df[ordered_columns],
                }
            )
            leaderboard_table = pd.pivot_table(
                data=output_df.assign(
                    task="hallucination_resistance",
                    does_believe=~output_df["does_believe"],
                ),
                values="does_believe",
                index="model_name",
                columns="task",
                aggfunc="mean",
            ).reset_index()
            score = float(leaderboard_table["hallucination_resistance"].iloc[0])
            run.log(
                {
                    leaderboard_table_name: leaderboard_table,
                    f"{task_name}_score": score,
                }
            )

        elif subset == "dev":
            run.log(
                {
                    table_name + "_dev": output_df[ordered_columns],
                }
            )

@weave.op(call_display_name=lambda _: "[Hallulens] " + WandbConfigSingleton.get_instance().config.wandb.run_name)
def evaluate():
    asyncio.run(evaluate_async())
