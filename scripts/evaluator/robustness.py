import wandb
import pandas as pd

from config_singleton import WandbConfigSingleton
from utils import read_wandb_table
from .evaluate_utils import symbol_to_ABCD, incorrect_to_ABCD


def eval_robustness(row):
    score = 0.0
    if not str(row["expected_output"]) in {"A", "B", "C", "D"}:
        return score
    else:
        if row["expected_output"] == row["output_IncorrectChoice"]:
            score += 0.5
        if row["expected_output"] == row["output_SymbolChoice"]:
            score += 0.5
        return score


def evaluate_n_shot(few_shots: bool):
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    dataset_name = "jmmlu_robustness"

    num_few_shots = cfg.get("num_few_shots", None) if few_shots else 0
    if num_few_shots is None:
        return

    if few_shots:
        num_few_shots = cfg.get("num_few_shots", None)
        if (num_few_shots is None) or (num_few_shots == 0):
            return
    else:
        num_few_shots = 0

    table_name_format = "jmmlu{task_suffix}_{n}shot_output_table{table_suffix}"
    for table_suffix in ("", "_dev"):
        kwargs = {
            "n": num_few_shots,
            "table_suffix": table_suffix,
        }
        task_defs = [
            {"task_suffix": "", **kwargs},
            {"task_suffix": "_SymbolChoice", **kwargs},
            {"task_suffix": "_IncorrectChoice", **kwargs},
        ]

        df_list = []
        for task_def in task_defs:
            task_suffix = task_def["task_suffix"]
            table_name = table_name_format.format(**task_def)
            output_df = read_wandb_table(run=run, table_name=table_name)

            if task_suffix == "":
                use_cols = [
                    "run_name",
                    "model_name",
                    "dataset",
                    "task",
                    "num_few_shots",
                    "subset",
                    "index",
                    "output",
                ]
                df_list.append(
                    output_df[use_cols].rename(columns={"output": "expected_output"})
                )
                continue
            else:
                if task_suffix == "_SymbolChoice":
                    output_df["output"] = output_df["output"].apply(symbol_to_ABCD)
                elif task_suffix == "_IncorrectChoice":
                    output_df["output"] = output_df["output"].apply(incorrect_to_ABCD)
                df_list.append(
                    output_df[["output"]].rename(
                        columns={"output": f"output{task_suffix}"}
                    )
                )

        output_df = pd.concat(df_list, axis=1)
        output_df["dataset"] = dataset_name
        output_df["metrics"] = "accuracy"
        output_df["score"] = output_df.apply(eval_robustness, axis=1)
        table_dict = {
            f"{dataset_name}_{num_few_shots}shot_output_table{table_suffix}": output_df,
        }

        if table_suffix == "":
            leaderboard_table = pd.pivot_table(
                data=output_df,
                values="score",
                index=["run_name", "model_name"],
                columns="dataset",
                aggfunc="mean",
            ).reset_index()
            table_dict.update(
                {
                    f"{dataset_name}_{num_few_shots}shot_leaderboard_table{table_suffix}": leaderboard_table,
                }
            )
        wandb.log(table_dict)


def evaluate():
    evaluate_n_shot(few_shots=False)
    evaluate_n_shot(few_shots=True)
