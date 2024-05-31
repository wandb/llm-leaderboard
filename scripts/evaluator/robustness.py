import wandb
import pandas as pd

from config_singleton import WandbConfigSingleton
from utils import read_wandb_table
from .evaluate_utils import symbol_to_ABCD, incorrect_to_ABCD


def evaluate():
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    config = instance.config

    # jmmlu_0shot_output_table
    # jmmlu_SymbolChoice_0shot_output_table
    # jmmlu_IncorrectChoice_0shot_output_table

    # jmmlu_0shot_output_table_dev
    # jmmlu_SymbolChoice_0shot_output_table_dev
    # jmmlu_IncorrectChoice_0shot_output_table_dev

    # jmmlu_4shot_output_table
    # jmmlu_SymbolChoice_4shot_output_table
    # jmmlu_IncorrectChoice_4shot_output_table

    # jmmlu_4shot_output_table_dev
    # jmmlu_SymbolChoice_4shot_output_table_dev
    # jmmlu_IncorrectChoice_4shot_output_table_dev

    table_name_format = "jmmlu{task_suffix}_{n}shot_output_table{table_suffix}"
    task_defs = [
        {
            "task_suffix": "",
            "table_suffix": "",
            "n": "0",
        },
        {
            "task_suffix": "_SymbolChoice",
            "table_suffix": "",
            "n": "0",
        },
        {
            "task_suffix": "_IncorrectChoice",
            "table_suffix": "",
            "n": "0",
        },
    ]
    for task_def in task_defs:
        task_suffix = task_def['task_suffix']
        table_name = table_name_format.format(**task_def)

        output_df = read_wandb_table(run=run, table_name=table_name).reset_index()
        if task_suffix == "_SymbolChoice":
            output_df['output'] = output_df['output'].apply(symbol_to_ABCD)
        elif task_suffix == "_IncorrectChoice":
            output_df['output'] = output_df['output'].apply(incorrect_to_ABCD)
        output_df.rename(columns={"output": f"output{task_suffix}"}, inplace=True)
        print(output_df)
    # task_suffixes = ["", "_dev"]
    # for subset in ("test", "dev"):
    #     # get output table
    #     suffix = "" if subset == "test" else "_dev"
    #     table_name = f"{input_task}_output_table{suffix}"
    #     output_df = read_wandb_table(run=run, table_name=table_name)
    #     # evaluate controllability
    #     output_df["metrics"] = output_df["task"].map(
    #         {k: v.__name__ for k, v in format_check_dict.items()}
    #     )
    #     output_df.dropna(subset=["metrics"], axis=0, inplace=True)
    #     output_df["score"] = output_df.apply(
    #         lambda x: format_check_dict[x["task"]](x["output"]) * 1, axis=1
    #     )
    #     # log tables
    #     table_dict = {
    #         f"{output_task}_output_table{suffix}": output_df,
    #     }
    #     if subset == "test":
    #         leaderboard_df = pd.pivot_table(
    #             data=output_df,
    #             values="score",
    #             index=["run_name", "model_name"],
    #             columns="dataset",
    #             aggfunc="mean",
    #         ).reset_index()
    #         leaderboard_df.columns = [
    #             "run_name",
    #             "model_name",
    #             output_task,
    #         ]
    #         table_dict.update(
    #             {
    #                 f"{output_task}_leaderboard_table": leaderboard_df,
    #             }
    #         )
    #     wandb.log(table_dict)
