import os
if os.environ.get("NEJUMI_MAIN_STARTED") == "1":
    print("Duplicate invocation detected; skipping.")
    raise SystemExit(0)
os.environ["NEJUMI_MAIN_STARTED"] = "1"

import wandb
import logging
from pathlib import Path
from argparse import ArgumentParser
from omegaconf import OmegaConf
import questionary

from config_singleton import WandbConfigSingleton
from llm_inference_adapter import get_llm_inference_engine
from vllm_server import shutdown_vllm_server
from docker_vllm_manager import stop_vllm_container_if_needed, start_vllm_container_if_needed
from blend_run import blend_run
from evaluator import (
    jaster,
    jbbq,
    mtbench,
    jaster_translation,
    toxicity,
    bfcl,
    jtruthfulqa,
    hle,
    hallulens,
    m_ifeval,
    aggregate,
    swe_bench,
    arc_agi,
)
from utils import paginate_choices
import weave

# Suppress noisy Gemini SDK warnings about non-text parts
logging.getLogger("google_genai.types").setLevel(logging.ERROR)

# Set config path
config_dir = Path("configs")
base_cfg_name = "base_config.yaml"
parser = ArgumentParser()
parser.add_argument("--config", "-c", type=str)
parser.add_argument("--select-config", "-s", action="store_true", default=False)
args = parser.parse_args()

if args.select_config:
    choices = sorted([p.name for p in config_dir.iterdir() if p.suffix == ".yaml"])
    if len(choices) > 36:
        custom_cfg_name = paginate_choices(choices)
    else:
        custom_cfg_name = questionary.select(
            "Select config",
            choices=choices,
            use_shortcuts=True,
        ).ask()
    custom_cfg_path = config_dir / custom_cfg_name
elif args.config:
    custom_cfg_path = config_dir / args.config
else:
    raise ValueError("No arguments found. Please specify either --config or --select-config.")

if custom_cfg_path.suffix != ".yaml":
    custom_cfg_path = custom_cfg_path.with_suffix(".yaml")
assert custom_cfg_path.exists(), f"Config file {custom_cfg_path.resolve()} does not exist"

# Configuration loading
custom_cfg = OmegaConf.load(custom_cfg_path)
base_cfg_path = config_dir / base_cfg_name
base_cfg = OmegaConf.load(base_cfg_path)

# vLLM利用時にbase_urlが未指定の場合、Composeサービス名 'vllm' をデフォルト設定
if "api" in custom_cfg and custom_cfg.api in ["vllm", "vllm-docker"]:
    if "base_url" not in custom_cfg:
        print("INFO: api is vllm/vllm-docker and base_url is not set. Defaulting to http://vllm:8000/v1")
        custom_cfg.base_url = "http://vllm:8000/v1"

custom_cfg = OmegaConf.merge(base_cfg, custom_cfg)
cfg_dict = OmegaConf.to_container(custom_cfg, resolve=True)
assert isinstance(cfg_dict, dict), "instance.config must be a DictConfig"

# 環境変数からAPIキーを取得
def get_api_key_from_env(service_name):
    """環境変数からAPIキーを取得"""
    env_var_name = f"{service_name.upper()}_API_KEY"
    return os.getenv(env_var_name)

# W&B APIキーの設定
wandb_api_key = get_api_key_from_env("wandb")
if wandb_api_key:
    os.environ["WANDB_API_KEY"] = wandb_api_key
    print(f"Wandb API key loaded from environment variable")
else:
    print("Warning: WANDB_API_KEY not found in environment variables")

"""Safeguard W&B init in multi-run environments"""
wandb_run = os.environ.get("WANDB_RUN_ID")
try:
    if os.environ.get("NEJUMI_WANDB_INIT_DONE") == "1":
        print("W&B already initialized; reusing existing run context.")
        run = wandb.run  # may be None if not active
    else:
        wandb.login()
        run = wandb.init(
            entity=cfg_dict["wandb"]["entity"],
            project=cfg_dict["wandb"]["project"],
            name=cfg_dict["wandb"]["run_name"],
            config=cfg_dict,
            job_type="evaluation",
            resume="allow" if wandb_run else None,
        )
        os.environ["NEJUMI_WANDB_INIT_DONE"] = "1"
        weave.init(cfg_dict["wandb"]["entity"]+"/"+cfg_dict["wandb"]["project"])
except Exception as e:
    print(f"Warning: Failed to initialize Wandb: {e}")
    print("Continuing without Wandb logging...")
    run = None

# Initialize the WandbConfigSingleton
if run:
    WandbConfigSingleton.initialize(run, llm=None)
    cfg = WandbConfigSingleton.get_instance().config

    # Save configuration as artifact
    artifact = wandb.Artifact("config", type="config")
    artifact.add_file(custom_cfg_path)
    run.log_artifact(artifact)

    # Inherit old runs
    blend_run(run_chain=True)
else:
    # Wandbが利用できない場合の代替設定
    cfg = OmegaConf.create(cfg_dict)

# vLLMコンテナの起動処理を追加
# Start vLLM container if needed (for vllm/vllm-docker API types)
# # Start vLLM container if needed
# if cfg.api in ["vllm-docker"]:
#     print(f"Starting vLLM container for model: {cfg.model.pretrained_model_name_or_path}")
#     
#     # 環境変数を設定（コンテナが正しいモデルを使用するために必要）
#     os.environ["EVAL_CONFIG_PATH"] = str(custom_cfg_path.name)
#     
#     # vLLMコンテナを起動（モデル名を明示的に渡す）
#     success = start_vllm_container_if_needed(model_name=cfg.model.pretrained_model_name_or_path)
#     if success is False:
#         print("Failed to start vLLM container, aborting evaluation")
#         if run:
#             wandb.finish()
#         exit(1)
#     elif success is True:
#         print("vLLM container started successfully")

# Start inference server
llm = get_llm_inference_engine()
if run:
    instance = WandbConfigSingleton.get_instance()
    instance.llm = llm

# BFCL
if cfg.run.bfcl:
    bfcl.evaluate()

# SWE-Bench Verified evaluation
if cfg.run.swebench:
    if cfg.swebench.background_eval:
        # 評価プロセスの実行時間が長いため、他のベンチと並行でバックグラウンド実行する
        # evaluate() はコールバック（wait_and_log_metrics）を返す実装に統一
        swebench_postprocess = swe_bench.evaluate()
    else:
        swe_bench.evaluate()

# mt-bench evaluation
if cfg.run.mtbench:
    mtbench.evaluate()

# jbbq
if cfg.run.jbbq:
    jbbq.evaluate()

# toxicity
if cfg.run.toxicity:
    toxicity.evaluate()

# JTruthfulQA
if cfg.run.jtruthfulqa:
    jtruthfulqa.evaluate()

# hle
if cfg.run.hle:
    hle.evaluate()

# HalluLens
if cfg.run.hallulens:
    hallulens.evaluate()

# ARC-AGI
if cfg.run.arc_agi:
    arc_agi.evaluate()

# M-IFEval
if cfg.run.m_ifeval:
    m_ifeval.evaluate()

# Evaluation phase
if cfg.run.jaster:
    # llm-jp-eval evaluation
    jaster.evaluate()

    #### open weight model base evaluation
    # 1. evaluation for translation task in jaster with comet
    # APIタイプに応じてvLLMサーバー/コンテナをシャットダウン
    lifecycle_mode = cfg.vllm.get("lifecycle", "auto")
    
    # jaster_translation (COMET) や jtruthfulqa (RoBERTa) はGPUメモリを大きく消費するため、
    # vLLMを一時停止する必要がある。
    # lifecycle: 'always_on' が指定されている場合を除く
    if lifecycle_mode != 'always_on':
        if cfg.api == "vllm-local":
            shutdown_vllm_server()
        elif cfg.api in ["vllm", "vllm-docker"]:
            stop_vllm_container_if_needed()
    
    # COMET評価を実行
    jaster_translation.evaluate()
    
    # APIタイプに応じてvLLMサーバー/コンテナを再起動
    if lifecycle_mode != 'always_on':
        if cfg.api == "vllm-local":
            llm = get_llm_inference_engine()
            if run:
                instance.llm = llm
        elif cfg.api in ["vllm", "vllm-docker"]:
            start_vllm_container_if_needed(model_name=cfg.model.pretrained_model_name_or_path)
            # llmインスタンスは同じものを使い続ける
            pass

if cfg.run.swebench and cfg.swebench.background_eval:
    # SWE-Bench評価完了を待ってから集計・W&Bロギングを確実に実施
    if callable(swebench_postprocess):
        swebench_postprocess()
    else:
        print("SWE-Bench background eval returned no callback; skipping explicit wait.")

# Aggregation
if cfg.run.aggregate:
    aggregate.evaluate()

# 評価完了後、vLLMコンテナを停止
if cfg.api in ["vllm", "vllm-docker"]:
    print("Stopping vLLM container...")
    # stop_vllm_container_if_needed()
    pass

# Finish
if run:
    run.finish()
