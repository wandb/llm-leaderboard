import json
from pathlib import Path

import pandas as pd
import wandb
import weave
from tqdm import tqdm

from config_singleton import WandbConfigSingleton
from .evaluate_utils import LLMAsyncProcessor
from .evaluate_utils.ifeval_zh_tw_utils import (
    read_prompt_list,
    read_prompt_to_response_dict,
    test_instruction_following_strict,
)


def _to_plain_dict(value) -> dict:
    if value is None:
        return {}
    try:
        from omegaconf import OmegaConf

        if not isinstance(value, dict):
            return OmegaConf.to_container(value, resolve=True) or {}
    except Exception:
        pass
    return dict(value)


def evaluate_outputs(input_data, output_data):
    instance = WandbConfigSingleton.get_instance()
    cfg = instance.config

    inputs = read_prompt_list(input_data)
    prompt_to_response = read_prompt_to_response_dict(output_data)
    outputs = [
        test_instruction_following_strict(inp, prompt_to_response)
        for inp in tqdm(inputs, desc="Evaluating IFEval zh-TW")
    ]

    instruction_total = sum(len(output.instruction_id_list) for output in outputs)
    instruction_correct = sum(sum(output.follow_instruction_list) for output in outputs)
    prompt_correct = sum(1 for output in outputs if output.follow_all_instructions)
    pass_rate = instruction_correct / instruction_total if instruction_total else 0.0
    prompt_pass_rate = prompt_correct / len(outputs) if outputs else 0.0

    output_rows = []
    for data, output in zip(output_data, outputs):
        output_rows.append(
            {
                "model_name": cfg.model.pretrained_model_name_or_path,
                "key": data["key"],
                "prompt": data["prompt"],
                "response": data["response"],
                "instruction_id_list": data["instruction_id_list"],
                "kwargs": json.dumps(data["kwargs"], ensure_ascii=False, sort_keys=True),
                "follow_instruction_list": output.follow_instruction_list,
                "follow_all_instructions": output.follow_all_instructions,
                "score": output.score,
            }
        )
    leaderboard_df = pd.DataFrame(
        [
            {
                "model_name": cfg.model.pretrained_model_name_or_path,
                "pass_rate": pass_rate,
                "prompt_pass_rate": prompt_pass_rate,
                "instruction_total": instruction_total,
                "instruction_correct": instruction_correct,
                "prompt_total": len(outputs),
                "prompt_correct": prompt_correct,
            }
        ]
    )
    return leaderboard_df, pd.DataFrame(output_rows)


@weave.op(call_display_name=lambda _: "[IFEval zh-TW] " + WandbConfigSingleton.get_instance().config.wandb.run_name)
def evaluate():
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    llm = instance.llm

    dataset_name = "ifeval_zh_tw"
    artifact = run.use_artifact(cfg[dataset_name].artifacts_path, type="dataset")
    artifact_dir = Path(artifact.download())
    input_data_path = artifact_dir / cfg[dataset_name].dataset_dir
    if not input_data_path.exists():
        raise FileNotFoundError(input_data_path)

    input_data = []
    with input_data_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                input_data.append(json.loads(line))

    testmode_max_samples = int(cfg[dataset_name].get("testmode_max_samples", 5))
    max_samples = cfg[dataset_name].get("max_samples", None)
    if cfg.get("testmode", False):
        input_data = input_data[:testmode_max_samples]
    elif max_samples is not None:
        input_data = input_data[: int(max_samples)]

    all_inputs = []
    evaluation_results = []
    for data in input_data:
        generator_config = _to_plain_dict(getattr(cfg, "generator", {}))
        generator_config.update(_to_plain_dict(getattr(cfg[dataset_name], "generator_config", {})))
        generator_config["max_tokens"] = int(generator_config.get("max_tokens", 2048))
        messages = [{"role": "user", "content": data["prompt"]}]
        all_inputs.append([messages, generator_config])
        evaluation_results.append(
            {
                "key": data["key"],
                "prompt": data["prompt"],
                "instruction_id_list": data["instruction_id_list"],
                "kwargs": data["kwargs"],
                "response": "",
            }
        )

    responses = LLMAsyncProcessor(llm=llm, inputs=all_inputs).get_results()
    for response, row in zip(responses, evaluation_results):
        row["response"] = getattr(response, "content", str(response))

    leaderboard_df, output_df = evaluate_outputs(input_data, evaluation_results)
    run.log(
        {
            "ifeval_zh_tw_leaderboard_table": wandb.Table(dataframe=leaderboard_df),
            "ifeval_zh_tw_output_table": wandb.Table(dataframe=output_df),
            "ifeval_zh_tw_score": float(leaderboard_df["pass_rate"].iloc[0]),
        }
    )
    return float(leaderboard_df["pass_rate"].iloc[0])
