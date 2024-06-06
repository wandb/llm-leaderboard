import wandb
from wandb.sdk.wandb_run import Run
import os
import sys
import argparse
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from config_singleton import WandbConfigSingleton
from cleanup import cleanup_gpu

def parse_args():
    parser = argparse.ArgumentParser(
        prog="nejumi eval", description="Run nejumi evaluation"
    )
    parser.add_argument("-c", "--config", default="configs/config_template.yaml", help="Config file to use")
    parser.add_argument("-m", "--model", help="Model path or name to use")
    parser.add_argument("-t", "--test", help="Test mode - use smaller data for quick iteration", action="store_true")
    parser.add_argument("--llm-jp-eval", help="Run only llm-jp-eval", action="store_true")
    parser.add_argument("--mtbench", help="Run only mt-bench", action="store_true")

    args = parser.parse_args()
    return args

def load_config(conf_file, model_path):
    # Configuration loading
    if os.path.exists(conf_file):
        cfg = OmegaConf.load(conf_file)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        # XXX why is this assert here? When would it be false?
        assert isinstance(cfg_dict, dict)
        if model_path:
            # Note this assumes the tokenizer is included with the model.
            cfg_dict["model"]["pretrained_model_name_or_path"] = model_path
            cfg_dict["tokenizer"]["pretrained_model_name_or_path"] = model_path
            cfg_dict["wandb"]["run_name"] = f"nejumi eval {model_path}"
        # if the model path is actually an HF name, use that as the ID
        if not os.path.exists(model_path):
            cfg_dict["mtbench"]["model_id"] = model_path.replace("/", "--")
    else:
        # XXX Would it not be better to just fail?
        print(f"Config file {conf_file} not found, using default values") 
        # Provide default settings in case the config file does not exist
        cfg_dict = {
            'wandb': {
                'entity': 'default_entity',
                'project': 'default_project',
                'run_name': 'default_run_name'
            }
        }
    return cfg_dict


def wandb_setup(cfg_dict, conf_file):
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
        if os.path.exists(conf_file):
            artifact_config_path = conf_file
        else:
            # If the config file does not exist, write the contents of run.config as a YAML configuration string
            instance = WandbConfigSingleton.get_instance()
            assert isinstance(instance.config, DictConfig), "instance.config must be a DictConfig"
            with open("configs/config.yaml", 'w') as f:
                f.write(OmegaConf.to_yaml(instance.config))
            artifact_config_path = "configs/config.yaml"

        artifact = wandb.Artifact('config', type='config')
        artifact.add_file(artifact_config_path)
        run.log_artifact(artifact)

    return run, cfg, table

def run_llm_jp(run, config, table):
    from llm_jp_eval.evaluator import evaluate
    # TODO this function should take and return values instead of using the singleton
    evaluate()
    cleanup_gpu()


def run_mt_bench(run, config, table):
    from mtbench_eval import mtbench_evaluate
    # If API keys are not set, fail fast
    try:
        os.environ["AZURE_OPENAI_KEY"]
    except KeyError:
        print("You need to load your OpenAI keys!")
        sys.exit(1)

    # TODO this function should take and return values instead of using the singleton
    mtbench_evaluate()
    cleanup_gpu()

def finish(run, cfg, table):
    # Logging results to W&B
    if cfg.wandb.log and run is not None:
        instance = WandbConfigSingleton.get_instance()
        run.log({
            "leaderboard_table": instance.table
        })
        run.finish()

def main():
    args = parse_args()
    cfg_dict = load_config(args.config, args.model)
    run, cfg, table = wandb_setup(cfg_dict, args.config)
    cfg.testmode = args.test

    if args.llm_jp_eval:
        if not args.mtbench:
            run.name = run.name + " (llm-jp-eval only)"
        run_llm_jp(run, cfg, table)
    
    if args.mtbench:
        if not args.llm_jp_eval:
            run.name = run.name + " (mtbench only)"
        run_mt_bench(run, cfg, table)

    finish(run, cfg, table)

if __name__ == "__main__":
    main()
