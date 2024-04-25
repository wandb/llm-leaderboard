from pathlib import Path
import json
from typing import Union

from omegaconf import OmegaConf
import pandas as pd
import wandb
from wandb.sdk.wandb_run import Run
import yaml

from config_singleton import WandbConfigSingleton

TASK_DICT = {
    "run_llm_jp_eval_ja_0_shot": "jaster_ja_0_shot",
    "run_llm_jp_eval_ja_few_shots": "jaster_ja_4_shot",
    "run_llm_jp_eval_en_0_shot": "jaster_en_0_shot",
    "run_llm_jp_eval_en_few_shots": "jaster_en_4_shot",
    "run_mt_bench_ja": "mtbench_ja",
    "run_mt_bench_en": "mtbench_en",
}


def test_task_name(
    old_runs: dict[str, Union[str, list[str]]], all_tasks: list[str]
) -> None:
    for old_run in old_runs:
        all_tasks += old_run.tasks
    assert len(all_tasks) == len(set(all_tasks)), "Dataset names must be unique"

    return None


def get_output_table(
    entity: str,
    project: str,
    run_id: str,
    table_name: str,
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


def process_task(
    dataset_name: str,
    task_name: str,
    old_run: dict[str, Union[str, list[str]]],
    leaderboard_tables: list[pd.DataFrame],
    run: Run,
) -> list[pd.DataFrame]:
    if dataset_name == "jaster":
        prefix, language, num_fewshots, _ = task_name.split("_")
        table_names = [
            f"{prefix}_leaderboard_table_{language}_{num_fewshots}shot",
            f"{prefix}_output_table_dev_{language}_{num_fewshots}shot",
            f"{prefix}_output_table_{language}_{num_fewshots}shot",
        ]
    elif dataset_name == "mtbench":
        prefix, language = task_name.split("_")
        table_names = [
            f"{prefix}_leaderboard_table_{language}",
            f"{prefix}_output_table_{language}",
            f"{prefix}_radar_table_{language}",
        ]

    for table_name in table_names:
        output_table = get_output_table(
            entity=old_run.entity,
            project=old_run.project,
            run_id=old_run.run_id,
            table_name=table_name,
            run=run,
        )
        run.log({table_name: output_table})
        if "leaderboard" in table_name:
            if dataset_name == "jaster":
                new_cols = [
                    f"{col}_jaster_{language}_{num_fewshots}shot"
                    for col in output_table.columns
                ]
            elif dataset_name == "mtbench":
                new_cols = [f"{col}_MTbench_{language}" for col in output_table.columns]
            output_table.columns = new_cols
            leaderboard_tables.append(output_table)

    return leaderboard_tables


def blend_tables(
    old_runs: dict[str, Union[str, list[str]]],
    leaderboard_tables: list[pd.DataFrame],
    run: Run,
) -> list[pd.DataFrame]:
    for old_run in old_runs:
        for task_name in old_run.tasks:
            if task_name.startswith("jaster"):
                dataset_name = "jaster"
            elif task_name.startswith("mtbench"):
                dataset_name = "mtbench"
            else:
                raise ValueError(f"Invalid task name: {task_name}")
            leaderboard_tables = process_task(
                dataset_name=dataset_name,
                task_name=task_name,
                old_run=old_run,
                leaderboard_tables=leaderboard_tables,
                run=run,
            )
    leaderboard_table = pd.concat(leaderboard_tables, axis=1)

    return leaderboard_table


def blend_run(run_chain: bool):
    blend_cfg_path = Path("blend_run_configs/config.yaml")
    with blend_cfg_path.open() as f:
        blend_cfg = OmegaConf.create(yaml.safe_load(f))
    old_runs: list[dict[str, Union[str, list[str]]]] = blend_cfg.old_runs

    # get run
    if run_chain:
        instance = WandbConfigSingleton.get_instance()
        run = instance.run
        cfg = instance.config
        old_leaderboard_table = instance.table
    else:
        wandb.login()
        run = wandb.init(
            entity=blend_cfg.new_run.entity,
            project=blend_cfg.new_run.project,
            name=blend_cfg.new_run.run_name,
            job_type="evaluation",
        )

    # log config
    artifact = wandb.Artifact(blend_cfg_path.stem, type="config")
    artifact.add_file(blend_cfg_path)
    run.log_artifact(artifact)

    # test
    all_tasks = []
    if run_chain:
        for k, v in TASK_DICT.items():
            if cfg[k]:
                all_tasks.append(v)
    test_task_name(old_runs=old_runs, all_tasks=all_tasks)

    # log tables and update tables
    if run_chain:
        leaderboard_tables = [old_leaderboard_table.get_dataframe()]
    else:
        leaderboard_tables = []
    leaderboard_table = blend_tables(
        old_runs=old_runs,
        leaderboard_tables=leaderboard_tables,
        run=run,
    )

    # finish
    if run_chain:
        instance.table = wandb.Table(dataframe=leaderboard_table)
    else:
        run.log({"leaderboard_table": leaderboard_table})
        run.finish()


if __name__ == "__main__":
    blend_run(run_chain=False)
