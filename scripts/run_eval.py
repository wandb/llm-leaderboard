import wandb
from pathlib import Path
from argparse import ArgumentParser
from omegaconf import OmegaConf
import questionary

from mtbench_eval import mtbench_evaluate
from toxicity_eval import toxicity_evaluate
from config_singleton import WandbConfigSingleton
from llm_inference_adapter import get_llm_inference_engine
from evaluator import (
    jaster,
    jmmlu,
    mmlu,
    controllability,
    robustness,
    lctg
)

# set config path
parser = ArgumentParser()
parser.add_argument("--config", "-c", type=str, default="config.yaml")
parser.add_argument("--select-config", "-s", action="store_true", default=False)
args = parser.parse_args()

config_dir = Path("configs")
if args.select_config:
    selected_config = questionary.select(
        "Select config",
        choices=[p.name for p in config_dir.iterdir() if p.suffix == ".yaml"],
        use_shortcuts=True,
    ).ask()
    cfg_path = config_dir / selected_config
elif args.config is not None:
    cfg_path = config_dir / args.config

if cfg_path.suffix != ".yaml":
    cfg_path = cfg_path.with_suffix(".yaml")
assert cfg_path.exists(), f"Config file {cfg_path} does not exist"


# Configuration loading
_cfg = OmegaConf.load(cfg_path)
cfg_dict = OmegaConf.to_container(_cfg, resolve=True)
assert isinstance(cfg_dict, dict), "instance.config must be a DictConfig"


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
instance = WandbConfigSingleton.get_instance()

artifact = wandb.Artifact('config', type='config')
artifact.add_file(cfg_path)
run.log_artifact(artifact)

# 0. Start inference server
llm = get_llm_inference_engine()
instance = WandbConfigSingleton.get_instance()
instance.llm = llm

# Evaluation phase
# 1. llm-jp-eval evaluation (jmmlu含む)
jaster.evaluate()
controllability.evaluate()

jmmlu.evaluate()
robustness.evaluate()
mmlu.evaluate()

# 2. mt-bench evaluation
mtbench_evaluate()

# 3. bbq, jbbq
# bbq_eval

# 4. lctg-bench
lctg.evaluate()

# 5. toxicity
toxicity_evaluate()

# Sample
# sample_evaluate()

# 6. Aggregation
# aggregate()
