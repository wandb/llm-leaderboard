import json
import math
import re
import time
import unicodedata
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

import wandb
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import BasePromptTemplate
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from config_singleton import WandbConfigSingleton
from llm_inference_adapter import get_llm_inference_engine

from utils import (
    get_evaluation_prompt,
    get_evaluation_result,
    get_few_shot_samples,
    post_process_score_results,
)

def sample_evaluate():
    # Retrieve the instance from WandbConfigSingleton and load the W&B run and configuration
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    pipe = get_llm_inference_engine()
    run_name = run.name

    ## データセットを取得しましょう
    # 効率的な運用のために、少量データでのテストモードを別途作るようにしてください
    # 実装はSaaSだけでなく、dedicated cloudでも動くように、OpenAIだけでなく、Azure OpenAIでも動くように心がけてください
    if cfg.testmode:
        artifact_dir = run.use_artifact(
            cfg.sample_dataset.test_artifacts_path, type="dataset"
        ).download()
    else:
        artifact_dir = run.use_artifact(
            cfg.sample_dataset.artifacts_path, type="dataset"
        ).download()

    score_results: dict[str, float] = {}
    output_results: dict[str, list[dict[str, str]]] = {}

    target_datasets = ["sapmle_dataset"]
    for target_dataset in target_datasets:
        target_dataset_path = Path(artifact_dir / "test" / f"{target_dataset}.json")
        if not target_dataset_path.exists():
            print(f"skip {target_dataset} because it is not found in {artifact_dir}")
            continue
        with target_dataset.open(encoding="utf-8") as f:
            target_data = json.load(f)

        # if "custom_prompt_template" in cfg:
        #     custom_prompt_template = cfg.custom_prompt_template
        # else:
        #     custom_prompt_template = None

        # if "custom_fewshots_template" in cfg:
        #     custom_fewshots_template = cfg.custom_fewshots_template
        # else:
        #     custom_fewshots_template = None

        few_shots = get_few_shot_samples(
            target_dataset_path=target_dataset_path, num_fewshots=4 # FIXME
        )

        prompt: BasePromptTemplate = get_evaluation_prompt(
            instruction=target_data["instruction"],
            few_shots=few_shots,
            custom_prompt_template=None,
            custom_fewshots_template=None,
        )
        llm_chain = LLMChain(
            llm=pipe,
            prompt=prompt,
            output_key="output",
        )
        overall_chain = SequentialChain(
            chains=[llm_chain],
            input_variables=llm_chain.input_keys,
            output_variables=llm_chain.output_keys,
            verbose=False,
        )

        test_max_num_samples = 100
        val_max_num_samples = 10
        if cfg.testmode:
            test_max_num_samples = 2
            val_max_num_samples = 2
            if target_dataset.startswith("mmlu_en"):
                test_max_num_samples = 1
                val_max_num_samples = 1
        elif "wiki" in target_dataset:
            test_max_num_samples = 20
            val_max_num_samples = 5
        elif target_dataset.startswith("mmlu_en"):
            test_max_num_samples = 5
            val_max_num_samples = 1

        test_metainfo = {"model_name": cfg.model_name, "data_type": "test", "num_few_shots": num_fewshots}
        score_result, output_result = get_evaluation_result(
            run_name=run_name,
            chain=overall_chain,
            samples=target_data["samples"],
            max_num_samples=test_max_num_samples,
            target_dataset_name=target_dataset,
            metrics=target_data["metrics"],
            metainfo=test_metainfo,
            target_dataset=target_dataset,
            wandb_outputs_table=None # wandb_outputs_table,
        )
        # added for leaderboard
        dev_metainfo = {"model_name": cfg.model_name, "data_type": "dev", "num_few_shots": num_fewshots}
        _, _ = get_evaluation_result(
            run_name=run_name,
            chain=overall_chain,
            samples=target_data["samples"],
            max_num_samples=val_max_num_samples,
            target_dataset_name=target_dataset,
            metrics=target_data["metrics"],
            metainfo=dev_metainfo,
            target_dataset=target_dataset,
            wandb_outputs_table=None # wandb_outputs_table_dev,
        )

        score_results.update(score_result)
        output_results[target_dataset] = [{"prompt": prompt.template, **x} for x in output_result]

    post_processed_score_results, task_category_to_task_names = post_process_score_results(
        score_results,
        add_avg_score=cfg.llm_jp_eval.target_dataset == "all",
    )

    post_processed_score_results = {
        k: (0 if isinstance(v, (int, float)) and np.isnan(v) else v) for k, v in post_processed_score_results.items()
    }

    radar_columns = list(task_category_to_task_names.keys())
    jaster_leaderboard_table = wandb.Table(
        columns=list(post_processed_score_results.keys()),
        data=[[float(value) for value in post_processed_score_results.values()]],
    )

    # Radar table
    if cfg.llm_jp_eval.target_dataset == "all":
        radar_T = jaster_leaderboard_table.get_dataframe()[radar_columns]
        radar = radar_T.transpose().reset_index()
        radar.columns = ["category", "score"]
        wandb_radar_table = wandb.Table(dataframe=radar)

    wandb_artifact = wandb.Artifact(name="output", type="dataset")
    wandb_artifact.add(jaster_leaderboard_table, "leaderboard")

    new_leaderboard_df = jaster_leaderboard_table.get_dataframe()

    print(new_leaderboard_df)


    # metainfo = OmegaConf.to_container(cfg.metainfo)
    # assert isinstance(metainfo, dict)
    # for m in list(metainfo.keys()):
    #     if isinstance(metainfo[m], list):
    #         metainfo[m] = ",".join(sorted(metainfo[m]))
    #     leaderboard_table.add_column(name=m, data=[metainfo[m]])

    # genhps = OmegaConf.to_container(cfg.generator)
    # assert isinstance(genhps, dict)
    # for hp in list(genhps.keys()):
    #     leaderboard_table.add_column(name=hp, data=[genhps[hp]])
