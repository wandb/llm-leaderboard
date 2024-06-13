from pathlib import Path
import json
from typing import Union

from omegaconf import OmegaConf
import pandas as pd
import wandb
from wandb.sdk.wandb_run import Run
import yaml
from aggregate import aggregate
from utils import read_wandb_table

from config_singleton import WandbConfigSingleton

# TODO: configでtaskを選択できるようになったら、blendしたtaskの総数チェック機能を追加
# TASK_DICT = {
#     "run_llm_jp_eval_ja_0_shot": "jaster_0_shot",
#     "run_llm_jp_eval_ja_few_shots": "jaster_4_shot",
#     "run_llm_jp_eval_ja_0_shot": "jaster_control_0_shot",
#     "run_llm_jp_eval_ja_few_shots": "jaster_control_4_shot",
#     "run_mt_bench_ja": "mtbench",
# }


# def test_task_name(
#     old_runs: dict[str, Union[str, list[str]]], all_tasks: list[str]
# ) -> None:
#     for old_run in old_runs:
#         all_tasks += old_run.tasks
#     assert len(all_tasks) == len(set(all_tasks)), "Tasks must be unique"

#     return None


def process_task(
    dataset_name: str,
    task_name: str,
    old_run: dict[str, Union[str, list[str]]],
    run: Run,
) -> None:
    if dataset_name == "jaster":
        if "control" in task_name:
            prefix, _, num_fewshots, _ = task_name.split("_")
        else:
            prefix, num_fewshots, _ = task_name.split("_")
        table_names = [
            f"{prefix}_{num_fewshots}shot_leaderboard_table",
            f"{prefix}_{num_fewshots}shot_output_table_dev",
            f"{prefix}_{num_fewshots}shot_output_table",
            f"{prefix}_control_{num_fewshots}shot_leaderboard_table",
        ]
    elif dataset_name == "jmmlu":
        prefix, _, num_fewshots, _ = task_name.split("_")
        table_names = [
            f"{prefix}_robust_{num_fewshots}shot_leaderboard_table",
            f"{prefix}_robust_{num_fewshots}shot_output_table_dev",
            f"{prefix}_robust_{num_fewshots}shot_output_table",
        ]
    elif dataset_name == "mtbench":
        prefix = task_name
        table_names = [
            f"{prefix}_leaderboard_table",
            f"{prefix}_output_table",
            f"{prefix}_radar_table",
        ]
    elif dataset_name == "jbbq":
        prefix, num_fewshots, _ = task_name.split("_")
        table_names = [
            f"{prefix}_{num_fewshots}shot_leaderboard_table",
            f"{prefix}_{num_fewshots}shot_output_table_dev",
            f"{prefix}_{num_fewshots}shot_output_table",
        ]
    elif dataset_name == "lctg":
        prefix = task_name
        table_names = [
            f"{prefix}_summary_leaderboard_table",
            f"{prefix}_ad_text_leaderboard_table",
            f"{prefix}_pros_and_cons_leaderboard_table",
            f"{prefix}_overall_leaderboard_table",
            f"{prefix}_output_table",
        ]
    elif dataset_name == "toxicity":
        prefix = task_name
        table_names = [
            f"{prefix}_leaderboard_table",
            f"{prefix}_output_table",
            f"{prefix}_radar_table",
        ]
    else:
        raise ValueError(f"Invalid task name: {task_name}")

    for table_name in table_names:
        output_table = read_wandb_table(
            run_path=old_run.run_path,
            table_name=table_name,
            run=run,
        )
        run.log({table_name: output_table})
        if "leaderboard" in table_name:
            if dataset_name == "jaster":
                if "control" in table_name:
                    new_cols = [f"{col}_{prefix}_control_{num_fewshots}shot" for col in output_table.columns]
                else:
                    new_cols = [f"{col}_{prefix}_{num_fewshots}shot" for col in output_table.columns]
            elif dataset_name == "jmmlu":
                new_cols = [f"{col}_{prefix}_robust_{num_fewshots}shot" for col in output_table.columns]
            elif dataset_name == "mtbench":
                new_cols = [f"{col}_{prefix}" for col in output_table.columns]
            elif dataset_name == "jbbq":
                new_cols = [f"{col}_{prefix}_{num_fewshots}shot" for col in output_table.columns]
            elif dataset_name == "lctg":
                if "summary" in table_name:
                    new_cols = [f"{col}_{prefix}_summary" for col in output_table.columns]
                else:
                    new_cols = [f"{col}_{prefix}_ad_text" for col in output_table.columns]
            elif dataset_name == "toxicity":
                new_cols = [f"{col}_{prefix}" for col in output_table.columns]
            else:
                raise ValueError(f"Invalid dataset name: {dataset_name}")
            
            output_table.columns = new_cols


def blend_tables(
    old_runs: dict[str, Union[str, list[str]]],
    run: Run,
) -> None:
    for old_run in old_runs:
        for task_name in old_run.tasks:
            if task_name.startswith("jaster"):
                dataset_name = "jaster"
            elif task_name.startswith("jmmlu"):
                dataset_name = "jmmlu"
            elif task_name.startswith("mtbench"):
                dataset_name = "mtbench"
            elif task_name.startswith("jbbq"):
                dataset_name = "jbbq"
            elif task_name.startswith("lctg"):
                dataset_name = "lctg"
            elif task_name.startswith("toxicity"):
                dataset_name = "toxicity"
            else:
                raise ValueError(f"Invalid task name: {task_name}")
            process_task(
                dataset_name=dataset_name,
                task_name=task_name,
                old_run=old_run,
                run=run,
            )


def blend_run(run_chain: bool) -> None:
    blend_cfg_path = Path("blend_run_configs/config.yaml")

    # config check
    if not blend_cfg_path.exists():
        print("Blend run skipped.")
        return None
    else:
        with blend_cfg_path.open() as f:
            blend_cfg = OmegaConf.create(yaml.safe_load(f))
        blend_cfg_dict = OmegaConf.to_container(blend_cfg, resolve=True)
        old_runs: list[dict[str, Union[str, list[str]]]] = blend_cfg.old_runs
    
    # check mode
    if not run_chain:
        pass
    elif not blend_cfg.run_chain:
        print("Blend run skipped.")
        return None

    # get run
    if run_chain:
        instance = WandbConfigSingleton.get_instance()
        run = instance.run
        cfg = instance.config
    else:
        wandb.login()
        run = wandb.init(
            entity=blend_cfg_dict["new_run"]["entity"],
            project=blend_cfg_dict["new_run"]["project"],
            name=blend_cfg_dict["new_run"]["run_name"],
            config=blend_cfg_dict,
            job_type="evaluation",
        )
        WandbConfigSingleton.initialize(run, llm=None)

    # log config
    artifact = wandb.Artifact(blend_cfg_path.stem, type="config")
    artifact.add_file(blend_cfg_path)
    run.log_artifact(artifact)

    # task_check
    # all_tasks = []
    # if run_chain:
    #     for k, v in TASK_DICT.items():
    #         if cfg[k]:
    #             all_tasks.append(v)
    # test_task_name(old_runs=old_runs, all_tasks=all_tasks)

    blend_tables(old_runs=old_runs, run=run)

    # finish
    if not run_chain:
        aggregate()


if __name__ == "__main__":
    blend_run(run_chain=False)