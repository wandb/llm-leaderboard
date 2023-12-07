import wandb
import sys
import os
from omegaconf import DictConfig, OmegaConf
sys.path.append('llm-jp-eval/src') 
from llm_jp_eval.evaluator import evaluate
sys.path.append('FastChat') 
from mtbench_eval import mtbench_evaluate

cfg = OmegaConf.load("configs/config.yaml")

# llm-jp-eval
# 2023/12/7: update from the original llm-jp-eval/dev
# use artifactsを追加する
#   artifact = run.use_artifact('wandb-japan/llm-leaderboard/jaster:v0', type='dataset')
#   artifact_dir = artifact.download()
#   cfg.dataset_dir = artifact_dir+cfg.dataset_dir
# change of table in run.log (add "jaster_" as prefix)
# add a script to generate leaderboard_score
# change the variables returned from the original to run, leaderboard_score
run, leaderboard_score = evaluate(cfg)

# mt-bench evaluation
# 2023/12/7: create mtbench_eval.py by using functions of FastChat
mtbench_evaluate(run_id=run.id, cfg=cfg,leaderboard_score=leaderboard_score)