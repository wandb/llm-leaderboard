import os
from pathlib import Path
import json

from omegaconf import OmegaConf
import pandas as pd
import wandb
import yaml


def get_output_table(
    run, entity: str, project: str, run_id: str, table_name: str
) -> pd.DataFrame:
    art_path = f"{entity}/{project}/run-{run_id}-{table_name}:latest"
    artifact = run.use_artifact(art_path)
    artifact_dir = artifact.download()
    with open(f"{artifact_dir}/{table_name}.table.json") as f:
        tjs = json.load(f)
    old_table = wandb.Table.from_json(json_obj=tjs, source_artifact=artifact)
    df = pd.DataFrame(data=old_table.data, columns=old_table.columns)
    return df


def main():
    config_path = Path(
        "llm-leaderboard/integration/integrate_config.yaml"
    )
    with config_path.open() as f:
        config = OmegaConf.create(yaml.safe_load(f))

    new_run: dict[str, str] = config.new_run
    old_runs: list[object] = config.old_runs

    wandb.login()
    with wandb.init(
        entity=new_run.entity,
        project=new_run.project,
        name=new_run.run_name,
        job_type="evaluation",
    ) as run:
        # log config
        artifact = wandb.Artifact(config_path.stem, type="config")
        artifact.add_file(config_path)
        run.log_artifact(artifact)
        # get tables
        output_dict = {} # TODO classでリファクタ
        for old_run in old_runs:
            for dataset_name in old_run.datasets:
                if dataset_name.startswith("jaster"):
                    prefix, language, num_shots, _ = dataset_name.split("_")
                    table_names = [
                        f"{prefix}_leaderboard_table_{language}_{num_shots}shot",
                        f"{prefix}_output_table_dev_{language}_{num_shots}shot",
                        f"{prefix}_output_table_{language}_{num_shots}shot",
                    ]
                elif dataset_name.startswith("mtbench"):
                    prefix, language = dataset_name.split("_")
                    table_names = [
                        f"{prefix}_leaderboard_table_{language}",
                        f"{prefix}_output_table_{language}",
                        f"{prefix}_radar_table_{language}",
                    ]
                for table_name in table_names:
                    output_table = get_output_table(
                        run=run,
                        entity=old_run.entity,
                        project=old_run.project,
                        run_id=old_run.run_id,
                        table_name=table_name,
                    )
                    output_dict[table_name] = output_table
        for table_name, output_table in output_dict.items():
            run.log({table_name: output_table})

if __name__ == "__main__":
    main()

# output_artifact = wandb.Artifact(name="output", type="dataset")
# output_artifact.add(jaster_leaderboard_table, "leaderboard")
# output_artifact.add(wandb_outputs_table, "output_test")
# output_artifact.add(wandb_outputs_table_dev, "output_dev")  # add for leaderboard
