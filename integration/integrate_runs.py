"""TODO
- lineage
- history
"""

from pathlib import Path
import json

from omegaconf import OmegaConf
import pandas as pd
import wandb
import yaml


def get_output_table(
    table_name: str,
    run,
    entity: str,
    project: str,
    run_id: str,
) -> pd.DataFrame:
    art_path = f"{entity}/{project}/run-{run_id}-{table_name}:latest"
    artifact = run.use_artifact(art_path)
    artifact_dir = artifact.download()
    with open(f"{artifact_dir}/{table_name}.table.json") as f:
        tjs = json.load(f)
    old_table = wandb.Table.from_json(json_obj=tjs, source_artifact=artifact)
    df = pd.DataFrame(data=old_table.data, columns=old_table.columns)
    return df


def process_jaster_dataset(dataset_name, run, old_run, leaderboard_tables):
    prefix, language, num_fewshots, _ = dataset_name.split("_")
    table_names = [
        f"{prefix}_leaderboard_table_{language}_{num_fewshots}shot",
        f"{prefix}_output_table_dev_{language}_{num_fewshots}shot",
        f"{prefix}_output_table_{language}_{num_fewshots}shot",
    ]
    for table_name in table_names:
        output_table = get_output_table(
            table_name=table_name,
            run=run,
            entity=old_run.entity,
            project=old_run.project,
            run_id=old_run.run_id,
        )
        run.log({table_name: output_table})
        if "leaderboard" in table_name:
            new_cols = [
                f"{col}_jaster_{language}_{num_fewshots}shot"
                for col in output_table.columns
            ]
            output_table.columns = new_cols
            leaderboard_tables.append(output_table)


def process_mtbench_dataset(dataset_name, run, old_run, leaderboard_tables):
    prefix, language = dataset_name.split("_")
    table_names = [
        f"{prefix}_leaderboard_table_{language}",
        f"{prefix}_output_table_{language}",
        f"{prefix}_radar_table_{language}",
    ]
    for table_name in table_names:
        output_table = get_output_table(
            table_name=table_name,
            run=run,
            entity=old_run.entity,
            project=old_run.project,
            run_id=old_run.run_id,
        )
        run.log({table_name: output_table})
        if "leaderboard" in table_name:
            new_cols = [f"{col}_MTbench_{language}" for col in output_table.columns]
            output_table.columns = new_cols
            leaderboard_tables.append(output_table)


def main():
    config_path = Path("integration/integrate_config.yaml")
    with config_path.open() as f:
        config = OmegaConf.create(yaml.safe_load(f))

    new_run: dict[str, str] = config.new_run
    old_runs: list[object] = config.old_runs

    # test dataset names
    all_datasets = []
    for old_run in old_runs:
        all_datasets += old_run.datasets
    assert len(all_datasets) == len(set(all_datasets)), "Dataset names must be unique"

    wandb.login()
    run = wandb.init(
        entity=new_run.entity,
        project=new_run.project,
        name=new_run.run_name,
        job_type="evaluation",
    )
    # log config
    artifact = wandb.Artifact(config_path.stem, type="config")
    artifact.add_file(config_path)
    run.log_artifact(artifact)

    # log output tables
    leaderboard_tables = []
    for old_run in old_runs:
        for dataset_name in old_run.datasets:
            if dataset_name.startswith("jaster"):
                process_jaster_dataset(
                    dataset_name=dataset_name,
                    run=run,
                    old_run=old_run,
                    leaderboard_tables=leaderboard_tables,
                )
            elif dataset_name.startswith("mtbench"):
                process_mtbench_dataset(
                    dataset_name=dataset_name,
                    run=run,
                    old_run=old_run,
                    leaderboard_tables=leaderboard_tables,
                )

    # log leaderboard table
    leaderboard_table = pd.concat(leaderboard_tables, axis=1)
    run.log({"leaderboard_table": leaderboard_table})
    run.finish()


if __name__ == "__main__":
    main()
