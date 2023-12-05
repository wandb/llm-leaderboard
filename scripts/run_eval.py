import wandb
import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
sys.path.append('llm-jp-eval/src') # このimportの方法をより洗練させる必要あり
import llm_jp_eval
sys.path.append('llm-jp-eval/scripts') # このimportの方法をより洗練させる必要あり
from evaluate_llm import main

config_path = "../configs"
config_name = "config"
hydra.initialize(config_path=config_path, version_base=None)
cfg = hydra.compose(config_name=config_name)
cfg_dict = OmegaConf.to_container(cfg, resolve=True)

#llm-jp-eval
with wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            config=cfg_dict,
            job_type="evaluate",
        ) as run:
    # download data
    artifact = run.use_artifact('wandb-japan/llm-leaderboard/jaster:v0', type='dataset')
    artifact_dir = artifact.download()
    print(artifact_dir)

    cfg.dataset_dir = artifact_dir+cfg.dataset_dir
    run = main(cfg)

    




