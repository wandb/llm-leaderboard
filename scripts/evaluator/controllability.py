import wandb
import pandas as pd

from config_singleton import WandbConfigSingleton
from utils import read_wandb_table
from .evaluate_utils import controllability_dict

"""
TODO
chatとcompletionの使い分け
"""

def evaluate():
    instance = WandbConfigSingleton.get_instance()
    run = instance.run

    input_task = "jaster_0shot"
    output_task = input_task + "_controllability"
    for table_suffix in ("", "_dev"):
        # get output table
        table_name = f"{input_task}_output_table{table_suffix}"
        output_df = read_wandb_table(run=run, table_name=table_name)
        # evaluate controllability
        output_df["metrics"] = output_df["task"].map(
            {k: v.__name__ for k, v in controllability_dict.items()}
        )
        output_df.dropna(subset=["metrics"], axis=0, inplace=True)
        output_df["score"] = output_df.apply(
            lambda x: controllability_dict[x["task"]](x["output"]) * 1, axis=1
        )
        # log tables
        table_dict = {
            f"{output_task}_output_table{table_suffix}": output_df,
        }
        if table_suffix == "":
            leaderboard_df = pd.pivot_table(
                data=output_df,
                values="score",
                index=["run_name", "model_name"],
                columns="dataset",
                aggfunc="mean",
            ).reset_index()
            leaderboard_df.columns = [
                "run_name",
                "model_name",
                output_task,
            ]
            table_dict.update(
                {
                    f"{output_task}_leaderboard_table": leaderboard_df,
                }
            )
        wandb.log(table_dict)
