from pathlib import Path
from typing import Union
from wandb.sdk.wandb_run import Run
from omegaconf import OmegaConf
from evaluator.aggregate import evaluate
from utils import read_wandb_table
from config_singleton import WandbConfigSingleton
import wandb
import yaml


def process_task(
    dataset: str,
    old_run: dict[str, Union[str, list[str]]],
    run: Run,
    num_few_shots: int,
) -> None:
    if dataset == "mtbench":
        table_names = [
            "mtbench_leaderboard_table",
            "mtbench_output_table",
            "mtbench_radar_table",
        ]
    elif dataset == "jbbq":
        table_names = [
            f"jbbq_{num_few_shots}shot_leaderboard_table",
            f"jbbq_{num_few_shots}shot_output_table_dev",
            f"jbbq_{num_few_shots}shot_output_table",
        ]
    elif dataset == "lctg":
        table_names = [
            "lctg_summary_leaderboard_table",
            "lctg_ad_text_leaderboard_table",
            "lctg_pros_and_cons_leaderboard_table",
            "lctg_overall_leaderboard_table",
            "lctg_output_table",
            "lctg_task_radar_table",
            "lctg_subtask_radar_table",
        ]
    elif dataset == "toxicity":
        table_names = [
            "toxicity_leaderboard_table",
            "toxicity_output_table",
            "toxicity_radar_table",
        ]
    elif dataset == "jaster":
        table_names = [
            f"jaster_0shot_output_table_dev",
            f"jaster_0shot_output_table",
            f"jaster_control_0shot_leaderboard_table",
            f"jaster_0shot_leaderboard_table",
            f"jaster_{num_few_shots}shot_output_table_dev",
            f"jaster_{num_few_shots}shot_output_table",
            f"jaster_control_{num_few_shots}shot_leaderboard_table",
            f"jaster_{num_few_shots}shot_leaderboard_table",
            f"jmmlu_robust_{num_few_shots}shot_output_table_dev",
            f"jmmlu_robust_{num_few_shots}shot_output_table",
            f"jmmlu_robust_{num_few_shots}shot_leaderboard_table",
        ]
    else:
        raise ValueError(f"Invalid dataset name: {dataset}")

    for table_name in table_names:
        output_table = read_wandb_table(
            run_path=old_run.run_path,
            table_name=table_name,
            run=run,
        )
        run.log({table_name: output_table})
        if "leaderboard" in table_name:
            if dataset == "mtbench":
                new_cols = [f"{col}_{dataset}" for col in output_table.columns]
            elif dataset == "jbbq":
                new_cols = [f"{col}_{dataset}_{num_few_shots}shot" for col in output_table.columns]
            elif dataset == "lctg":
                if "summary" in table_name:
                    new_cols = [f"{col}_{dataset}_summary" for col in output_table.columns]
                else:
                    new_cols = [f"{col}_{dataset}_ad_text" for col in output_table.columns]
            elif dataset == "toxicity":
                new_cols = [f"{col}_{dataset}" for col in output_table.columns]
            elif dataset == "jaster":
                if "control" in table_name:
                    new_cols = [f"{col}_{dataset}_control_{num_few_shots}shot" for col in output_table.columns]
                else:
                    new_cols = [f"{col}_{dataset}_{num_few_shots}shot" for col in output_table.columns]
            elif dataset == "jmmlu":
                new_cols = [f"{col}_{dataset}_robust_{num_few_shots}shot" for col in output_table.columns]
            else:
                raise ValueError(f"Invalid dataset name: {dataset}")
            
            output_table.columns = new_cols


def blend_tables(
    old_runs: dict[str, Union[str, list[str]]],
    run: Run,
    num_few_shots: int,
) -> None:
    for old_run in old_runs:
        for dataset in old_run.dataset:
            # if task_name.startswith("mtbench"):
            #     dataset = "mtbench"
            # elif task_name.startswith("jbbq"):
            #     dataset = "jbbq"
            # elif task_name.startswith("lctg"):
            #     dataset = "lctg"
            # elif task_name.startswith("toxicity"):
            #     dataset = "toxicity"
            # elif task_name.startswith("jaster"):
            #     dataset = "jaster"
            # else:
            #     raise ValueError(f"Invalid task name: {task_name}")
            process_task(
                dataset=dataset,
                old_run=old_run,
                run=run,
                num_few_shots=num_few_shots,
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
        instance = WandbConfigSingleton.get_instance()
        cfg = instance.config

    # log config
    artifact = wandb.Artifact(blend_cfg_path.stem, type="config")
    artifact.add_file(blend_cfg_path)
    run.log_artifact(artifact)

    blend_tables(old_runs=old_runs, run=run, num_few_shots=cfg.num_few_shots)

    # finish
    if not run_chain:
        evaluate()


if __name__ == "__main__":
    blend_run(run_chain=False)