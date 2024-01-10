import wandb
from wandb.sdk.wandb_run import Run
import sys
from omegaconf import OmegaConf
import pandas as pd
sys.path.append('llm-jp-eval/src') 
sys.path.append('FastChat')
from llm_jp_eval.evaluator import evaluate
from mtbench_eval import mtbench_evaluate
from config_singleton import WandbConfigSingleton

# Configuration loading
cfg = OmegaConf.load("configs/config.yaml")

# W&B setup and artifact handling
if cfg.wandb.log:
    wandb.login()
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        config=cfg_dict,
        job_type="evaluation",
    )
    
    # Save configuration as artifact
    artifact = wandb.Artifact('config', type='config')
    artifact.add_file("configs/config.yaml")
    run.log_artifact(artifact)
else:
    run = None
    run_name = "Nan"

# Initialize the WandbConfigSingleton
WandbConfigSingleton.initialize(run, wandb.Table(dataframe=pd.DataFrame()))

# Evaluation phase
# 1. llm-jp-eval evaluation
evaluate()

# 2. mt-bench evaluation
mtbench_evaluate()

# Logging results to W&B
if cfg.wandb.log and run is not None:
    instance = WandbConfigSingleton.get_instance()
    run.log({
        "leaderboard_table": instance.table
    })
    run.finish()