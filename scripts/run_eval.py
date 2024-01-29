import wandb
from wandb.sdk.wandb_run import Run
import os
import sys
from omegaconf import DictConfig, OmegaConf
import pandas as pd
sys.path.append('llm-jp-eval/src') 
sys.path.append('FastChat')
from llm_jp_eval.evaluator import evaluate
from mtbench_eval import mtbench_evaluate
from config_singleton import WandbConfigSingleton
from cleanup import cleanup_gpu

# Configuration loading
if os.path.exists("configs/config.yaml"):
    cfg = OmegaConf.load("configs/config.yaml")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)
else:
    # Provide default settings in case config.yaml does not exist
    cfg_dict = {
        'wandb': {
            'entity': 'default_entity',
            'project': 'default_project',
            'run_name': 'default_run_name'
        }
    }

# W&B setup and artifact handling
wandb.login()
run = wandb.init(
    entity=cfg_dict['wandb']['entity'],
    project=cfg_dict['wandb']['project'],
    name=cfg_dict['wandb']['run_name'],
    config=cfg_dict,
    job_type="evaluation",
)

# Initialize the WandbConfigSingleton
WandbConfigSingleton.initialize(run, wandb.Table(dataframe=pd.DataFrame()))
cfg = WandbConfigSingleton.get_instance().config

# Save configuration as artifact
if cfg.wandb.log:
    if os.path.exists("configs/config.yaml"):
        artifact_config_path = "configs/config.yaml"
    else:
        # If "configs/config.yaml" does not exist, write the contents of run.config as a YAML configuration string
        instance = WandbConfigSingleton.get_instance()
        assert isinstance(instance.config, DictConfig), "instance.config must be a DictConfig"
        with open("configs/config.yaml", 'w') as f:
            f.write(OmegaConf.to_yaml(instance.config))
        artifact_config_path = "configs/config.yaml"

    artifact = wandb.Artifact('config', type='config')
    artifact.add_file(artifact_config_path)
    run.log_artifact(artifact)

# Evaluation phase
# 1. Llm-jp-eval-jp 0 shot evaluation
if cfg.run_llm_jp_eval_jp_0_shot:
    evaluate(num_fewshots=0, target="all_jp")
    cleanup_gpu()

# 2. Llm-jp-eval-jp few shots evaluation
if cfg.run_llm_jp_eval_jp_few_shots:
    evaluate(num_fewshots=cfg.metainfo.num_few_shots, target="all_jp")
    cleanup_gpu()

# 3. Llm-jp-eval-en 0 shot evaluation
if cfg.run_llm_jp_eval_en_0_shot:
    evaluate(num_fewshots=0, target="all_en")
    cleanup_gpu()

# 4. Llm-jp-eval-en few shots evaluation
if cfg.run_llm_jp_eval_en_few_shots:
    evaluate(num_fewshots=cfg.metainfo.num_few_shots, target="all_en")
    cleanup_gpu()

# 5. mt-bench evaluation
if cfg.run_mt_bench_jp:
    mtbench_evaluate(language="jp")
    cleanup_gpu()

# 6. mt-bench evaluation
if cfg.run_mt_bench_en:
    mtbench_evaluate(language="en")
    cleanup_gpu()

# Logging results to W&B
if cfg.wandb.log and run is not None:
    instance = WandbConfigSingleton.get_instance()
    run.log({
        "leaderboard_table": instance.table
    })
    run.finish()