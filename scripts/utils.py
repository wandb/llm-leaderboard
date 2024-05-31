import json
import wandb
import pandas as pd


def read_wandb_table(
    run: object, table_name: str, version: str = "latest"
) -> pd.DataFrame:
    artifact_path = f"{run.entity}/{run.project}/run-{run.id}-{table_name}:{version}"
    artifact = run.use_artifact(artifact_path)
    artifact_dir = artifact.download()
    with open(f"{artifact_dir}/{table_name}.table.json") as f:
        tjs = json.load(f)
    output_table = wandb.Table.from_json(json_obj=tjs, source_artifact=artifact)
    output_df = pd.DataFrame(data=output_table.data, columns=output_table.columns)
    return output_df
