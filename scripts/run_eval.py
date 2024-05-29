import wandb
from wandb.sdk.wandb_run import Run
import os
import sys
from omegaconf import DictConfig, OmegaConf
import pandas as pd
sys.path.append('FastChat')
from mtbench_eval import mtbench_evaluate
from config_singleton import WandbConfigSingleton
from llm_inference_adapter import get_llm_inference_engine

from evaluator import (
    jaster,
    jmmlu,
    mmlu,
    controllability,
)

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
WandbConfigSingleton.initialize(run, llm=None)
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


# 0. Start inference server
llm = get_llm_inference_engine()
instance = WandbConfigSingleton.get_instance()
instance.llm = llm

# Evaluation phase
# 1. llm-jp-eval evaluation (jmmlu含む)
jaster.evaluate()

jmmlu.evaluate()
controllability.evaluate()

mmlu.evaluate()


# 2. mt-bench evaluation
#mtbench_evaluate()

# 3. bbq, jbbq
#bbq_eval

# 4. lctg-bench
#lctgbench_evaluate()

# 5. ly-toxicity
#ly_toxicity_evaluate()

# Sample
# sample_evaluate()

# 6. Aggregation
#aggregate()