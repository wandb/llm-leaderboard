import wandb
import json
import pandas as pd
from config_singleton import WandbConfigSingleton
from .format_checker import format_check_dict


def evaluate():
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config

    entity = cfg.wandb.entity
    project = cfg.wandb.project
    run_id = run.id

    input_task = "jaster_0shot"
    output_task = input_task + "_controllability"
    for subset in ("test", "dev"):
        # get output table
        suffix = "" if subset == "test" else "_dev"
        table_name = f"{input_task}_output_table{suffix}"
        artifact_path = f"{entity}/{project}/run-{run_id}-{table_name}:latest"
        artifact = run.use_artifact(artifact_path)
        artifact_dir = artifact.download()
        with open(f"{artifact_dir}/{table_name}.table.json") as f:
            tjs = json.load(f)
        output_table = wandb.Table.from_json(json_obj=tjs, source_artifact=artifact)
        output_df = pd.DataFrame(
            data=output_table.data, columns=output_table.columns
        )
        # evaluate controllability
        output_df["metrics"] = output_df["task"].map(
            {k: v.__name__ for k, v in format_check_dict.items()}
        )
        output_df.dropna(subset=["metrics"], axis=0, inplace=True)
        output_df["score"] = output_df.apply(
            lambda x: format_check_dict[x["task"]](x["output"]) * 1, axis=1
        )
        # log tables
        table_dict = {
            f"{output_task}_output_table{suffix}": output_df,
        }
        if subset == "test":
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
