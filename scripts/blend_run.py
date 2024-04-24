from pathlib import Path
import json
from typing import Union

from omegaconf import OmegaConf
import pandas as pd
import wandb
from wandb.sdk.wandb_run import Run
import yaml

from config_singleton import WandbConfigSingleton

DATASET_DICT = {
    "run_llm_jp_eval_ja_0_shot": "jaster_ja_0_shot",
    "run_llm_jp_eval_ja_few_shots": "jaster_ja_4_shot",
    "run_llm_jp_eval_en_0_shot": "jaster_en_0_shot",
    "run_llm_jp_eval_en_few_shots": "jaster_en_4_shot",
    "run_mt_bench_ja": "mtbench_ja",
    "run_mt_bench_en": "mtbench_en",
}


def get_output_table(
    table_name: str,
    entity: str,
    project: str,
    run_id: str,
    run: Run,
) -> pd.DataFrame:
    art_path = f"{entity}/{project}/run-{run_id}-{table_name}:latest"
    artifact = run.use_artifact(art_path)
    artifact_dir = artifact.download()
    with open(f"{artifact_dir}/{table_name}.table.json") as f:
        tjs = json.load(f)
    old_table = wandb.Table.from_json(json_obj=tjs, source_artifact=artifact)
    df = pd.DataFrame(data=old_table.data, columns=old_table.columns)
    return df


def process_jaster_dataset(
    dataset_name: str,
    old_run: dict[str, Union[str, list[str]]],
    leaderboard_tables: list[pd.DataFrame],
    run: Run,
):
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


def process_mtbench_dataset(
    dataset_name: str,
    run: Run,
    old_run: dict[str, Union[str, list[str]]],
    leaderboard_tables: list[pd.DataFrame],
):
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


def test_dataset_names(old_runs: dict[str, Union[str, list[str]]], all_datasets: list[str]):
    for old_run in old_runs:
        all_datasets += old_run.datasets
    assert len(all_datasets) == len(set(all_datasets)), "Dataset names must be unique"


def log_tables(
    leaderboard_tables: list[pd.DataFrame],
    old_runs:  dict[str, Union[str, list[str]]],
    run: Run,
    blend_cfg_path: Path,
) -> list[pd.DataFrame]:
    artifact = wandb.Artifact(blend_cfg_path.stem, type="config")
    artifact.add_file(blend_cfg_path)
    run.log_artifact(artifact)
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
    leaderboard_table = pd.concat(leaderboard_tables, axis=1)
    return leaderboard_table


def blend_run(run_chain: bool):
    blend_cfg_path = Path("blend_run_configs/config.yaml")
    with blend_cfg_path.open() as f:
        blend_cfg = OmegaConf.create(yaml.safe_load(f))
    old_runs: list[object] = blend_cfg.old_runs

    if run_chain:
        instance = WandbConfigSingleton.get_instance()
        run = instance.run
        cfg = instance.config
        old_leaderboard_table = instance.table

        # test dataset names
        all_datasets = []
        for k, v in DATASET_DICT.items():
            if cfg[k]:
                all_datasets.append(v)
        test_dataset_names(old_runs=old_runs, all_datasets=all_datasets)

        # log tables and update tables
        leaderboard_tables = [old_leaderboard_table.get_dataframe()]
        leaderboard_table = log_tables(
            leaderboard_tables=leaderboard_tables,
            old_runs=old_runs,
            run=run,
            blend_cfg_path=blend_cfg_path,
        )
        instance.table = wandb.Table(dataframe=leaderboard_table)

    else:
        # test dataset names
        all_datasets = []
        test_dataset_names(old_runs=old_runs, all_datasets=all_datasets)

        wandb.login()
        run = wandb.init(
            entity=blend_cfg.new_run.entity,
            project=blend_cfg.new_run.project,
            name=blend_cfg.new_run.run_name,
            job_type="evaluation",
        )

        # log output tables
        leaderboard_tables = []
        leaderboard_table = log_tables(
            leaderboard_tables=leaderboard_tables,
            old_runs=old_runs,
            run=run,
            blend_cfg_path=blend_cfg_path,
        )
        run.log({"leaderboard_table": leaderboard_table})
        run.finish()


if __name__ == "__main__":
    blend_run(run_chain=False)
