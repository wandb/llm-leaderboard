import json
import time
from pathlib import Path

import wandb
from langchain.prompts import BasePromptTemplate
from tqdm import tqdm
import pandas as pd
from toolz import pipe

from config_singleton import WandbConfigSingleton
from .evaluate_utils import (
    apply_chat_template,
    get_few_shot_messages,
    get_system_message,
    jaster_metrics_dict,
    normalize,
    text_formatter,
)


def evaluate_n_shot(few_shots: bool):
    # Retrieve the instance from WandbConfigSingleton and load the W&B run and configuration
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    llm = instance.llm
    api_type = cfg.api

    # download dataset
    dataset_name = "mmlu"
    artifact = run.use_artifact(cfg[dataset_name].artifacts_path, type="dataset")
    artifact_dir = artifact.download()
    dataset_dir = Path(artifact_dir) / cfg[dataset_name].dataset_dir
    if not dataset_dir.exists():
        print(f"skip {dataset_name} because it is not found in {artifact_dir}")
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")
    tasks = sorted({p.stem for p in dataset_dir.glob("**/mmlu_en_*.json")})

    if few_shots:
        num_few_shots = cfg.get("num_few_shots", None)
        if (num_few_shots is None) or (num_few_shots == 0):
            return
    else:
        num_few_shots = 0

    evaluation_results = []
    for task in tasks:
        # execute evaluation
        language = cfg[dataset_name].language
        for subset in ("test", "dev"):
            eval_matainfo = {
                "run_name": run.name,
                "model_name": cfg.model.pretrained_model_name_or_path,
                "dataset": dataset_name,
                "task": task[len("mmlu_en_"):],
                "num_few_shots": cfg.num_few_shots,
                "subset": subset,
            }

            # read task data
            task_data_path = (
                dataset_dir
                / subset
                / f"{task}.json"
            )
            if not task_data_path.exists():
                print(f"skip {task} because it is not found in {artifact_dir}")
                continue
            with task_data_path.open(encoding="utf-8") as f:
                task_data = json.load(f)

            # number of evaluation samples
            if cfg.testmode:
                test_max_num_samples = 1
                val_max_num_samples = 1
            else:
                test_max_num_samples = 5
                val_max_num_samples = 1

            if subset == "test":
                num_samples = test_max_num_samples
            elif subset == "dev":
                num_samples = val_max_num_samples
            samples = task_data["samples"][:num_samples]

            for idx, sample in tqdm(enumerate(samples)):
                # compose messages
                messages = []

                # system message
                system_message = get_system_message(
                    system_message_intro=cfg[dataset_name].system_message,
                    instruction=task_data["instruction"],
                )
                messages.append({"role": "system", "content": system_message})

                # add fewshots samples
                if few_shots:
                    few_shot_messages = get_few_shot_messages(
                        target_dataset_path=task_data_path,
                        num_few_shots=num_few_shots,
                    )
                    messages.extend(few_shot_messages)

                # user input
                messages.append({"role": "user", "content": sample["input"]})

                # generate output
                start_time = time.time()
                prompt = apply_chat_template(messages=messages)
                output = llm.invoke(messages, max_tokens=task_data["output_length"]).content
                end_time = time.time()
                latency = end_time - start_time

                # score
                y_pred: str = pipe(
                    output,
                    lambda x: text_formatter(x, task),
                    lambda x: x.split("\n\n")[0],
                    normalize,
                )
                y_true: str = pipe(sample["output"], normalize)
                metrics: list[str] = task_data["metrics"][0]
                metrics_func: callable = jaster_metrics_dict[metrics]
                score = metrics_func(y_pred, y_true)

                # collect data
                evaluation_results.append(
                    {
                        **eval_matainfo,
                        "index": idx,
                        "input": sample["input"],
                        'raw_output': output,
                        "output": y_pred,
                        "expected_output": y_true,
                        "prompt": prompt,
                        "metrics": metrics,
                        "score": score,
                        "latency": latency,
                    }
                )

    # log table
    output_df = pd.DataFrame(evaluation_results)
    dev_table = output_df.query("subset == 'dev'")
    test_table = output_df.query("subset == 'test'")
    leaderboard_table = pd.pivot_table(
        data=test_table, values="score", index=['run_name', "model_name"], columns="dataset", aggfunc="mean"
    ).reset_index()
    wandb.log(
        {
            f"{dataset_name}_{num_few_shots}shot_output_table_dev": dev_table,
            f"{dataset_name}_{num_few_shots}shot_output_table": test_table,
            f"{dataset_name}_{num_few_shots}shot_leaderboard_table": leaderboard_table,
        }
    )

def evaluate():
    evaluate_n_shot(few_shots=False)
    evaluate_n_shot(few_shots=True)