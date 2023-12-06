import wandb
import sys
import os
from omegaconf import DictConfig, OmegaConf
sys.path.append('llm-jp-eval/src') # このimportの方法をより洗練させる必要あり
from llm_jp_eval.evaluator import evaluate


# evaluation to jaster data

# 2023/12/05: update from the original llm-jp-eval/dev
# 1. return runに変更をする
# 2. use artifactsを追加する
#  artifact = run.use_artifact('wandb-japan/llm-leaderboard/jaster:v0', type='dataset')
#  artifact_dir = artifact.download()
#  cfg.dataset_dir = artifact_dir+cfg.dataset_dir

cfg = OmegaConf.load("configs/config.yaml")
run = evaluate(cfg)







