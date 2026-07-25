import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
import weave
from toolz import pipe
from tqdm import tqdm

from config_singleton import WandbConfigSingleton
from .evaluate_utils import (
    jaster_metrics_dict,
    normalize,
    text_formatter,
)
from .evaluate_utils.llm_response_checkpoint import (
    LLMResponseCheckpointStore,
    default_checkpoint_root,
    run_checkpointed_batch,
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


def _extract_answer(raw_output: str, expected_output: str, task: str) -> str:
    formatted = pipe(
        raw_output,
        lambda x: text_formatter(x, task),
        lambda x: x.split("\n\n")[0],
        lambda x: x.strip(),
        lambda x: x.strip("'").strip('"'),
        lambda x: x.strip(),
        normalize,
    )
    if task == "penguin_table" and expected_output in {"A", "B", "C", "D", "E"}:
        match = re.search(r"\b([A-E])\b", formatted)
        if match:
            return match.group(1)
    return formatted


@weave.op(call_display_name=lambda _: "[TCEval-v2] " + WandbConfigSingleton.get_instance().config.wandb.run_name)
def evaluate():
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    llm = instance.llm

    dataset_name = "tceval_v2"
    artifact = run.use_artifact(cfg[dataset_name].artifacts_path, type="dataset")
    artifact_dir = Path(artifact.download())
    dataset_dir = artifact_dir / cfg[dataset_name].dataset_dir
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

    tasks = list(cfg[dataset_name].get("tasks", ["drcd", "penguin_table"]))
    max_samples = cfg[dataset_name].get("max_samples", None)
    testmode_max_samples = cfg[dataset_name].get("testmode_max_samples", 1)
    generator_override = _to_plain_dict(cfg[dataset_name].get("generator_config", {}))

    evaluation_results = []
    all_inputs = []
    request_keys = []

    for task in tasks:
        for subset in ("test", "dev"):
            task_data_path = dataset_dir / subset / f"{task}.json"
            if not task_data_path.exists():
                raise FileNotFoundError(
                    f"TCEval-v2 required task file not found: {task_data_path}"
                )

            with task_data_path.open(encoding="utf-8") as f:
                task_data = json.load(f)

            samples = task_data["samples"]
            if cfg.testmode:
                samples = samples[:testmode_max_samples]
            elif max_samples is not None:
                samples = samples[:max_samples]

            for idx, sample in enumerate(samples):
                generator_config = _to_plain_dict(getattr(cfg, "generator", {}))
                generator_config.update(generator_override)
                override_max_tokens = cfg[dataset_name].get("override_max_tokens", None)
                generator_config["max_tokens"] = override_max_tokens or task_data[
                    "output_length"
                ]

                user_content = "\n".join([task_data["instruction"], "", sample["input"]])
                messages = [{"role": "user", "content": user_content}]
                all_inputs.append([messages, generator_config])
                request_keys.append(f"{task}:{subset}:{idx}")
                evaluation_results.append(
                    {
                        "model_name": cfg.model.pretrained_model_name_or_path,
                        "dataset": dataset_name,
                        "task": task,
                        "subset": subset,
                        "index": idx,
                        "input": sample["input"],
                        "expected_output": pipe(sample["output"], normalize),
                        "prompt": user_content,
                        "metrics": task_data["metrics"][0],
                        "task_output_length": task_data["output_length"],
                        "requested_max_tokens": generator_config["max_tokens"],
                    }
                )

    if not all_inputs:
        raise RuntimeError("TCEval-v2 produced no evaluation requests")

    configured_checkpoint_dir = cfg[dataset_name].get("checkpoint_dir", None)
    checkpoint_store = LLMResponseCheckpointStore(
        Path(configured_checkpoint_dir)
        if configured_checkpoint_dir
        else default_checkpoint_root(run, dataset_name),
        model_name=cfg.model.pretrained_model_name_or_path,
    )
    responses = run_checkpointed_batch(
        llm=llm,
        inputs=all_inputs,
        keys=request_keys,
        checkpoint_store=checkpoint_store,
        label="TCEval-v2",
    )
    if len(responses) != len(evaluation_results):
        raise RuntimeError(
            "TCEval-v2 response count mismatch: "
            f"{len(responses)}/{len(evaluation_results)}"
        )

    for response, row in tqdm(
        zip(responses, evaluation_results),
        total=len(evaluation_results),
        desc="Evaluating TCEval-v2 selected",
    ):
        raw_output = response.content
        y_pred = _extract_answer(raw_output, row["expected_output"], row["task"])
        metrics_func = jaster_metrics_dict[row["metrics"]]
        row["raw_output"] = raw_output
        row["output"] = y_pred
        row["score"] = metrics_func(y_pred, row["expected_output"])
        row["prompt_tokens"] = response.prompt_tokens
        row["completion_tokens"] = response.completion_tokens
        row["finish_reason"] = response.finish_reason
        row["content_was_none"] = response.content_was_none
        row["reasoning_content_len"] = len(response.reasoning_content or "")
        row["raw_output_is_empty"] = raw_output == ""

    output_df = pd.DataFrame(evaluation_results)
    dev_table = output_df.query("subset == 'dev'")
    test_table = output_df.query("subset == 'test'")

    leaderboard_table = pd.pivot_table(
        data=test_table,
        values="score",
        index="model_name",
        columns="task",
        aggfunc="mean",
    ).reset_index()
    task_cols = [col for col in tasks if col in leaderboard_table.columns]
    leaderboard_table.insert(1, "AVG", leaderboard_table[task_cols].mean(axis=1))
    leaderboard_table["drcd_num_samples"] = int((test_table["task"] == "drcd").sum())
    leaderboard_table["penguin_table_num_samples"] = int(
        (test_table["task"] == "penguin_table").sum()
    )

    run.log(
        {
            "tceval_v2_selected_output_table_dev": wandb.Table(dataframe=dev_table),
            "tceval_v2_selected_output_table": wandb.Table(dataframe=test_table),
            "tceval_v2_selected_leaderboard_table": wandb.Table(
                dataframe=leaderboard_table
            ),
            "tceval_v2_selected_score": float(
                leaderboard_table["AVG"].iloc[0]
                if len(leaderboard_table) > 0
                else np.nan
            ),
        }
    )
