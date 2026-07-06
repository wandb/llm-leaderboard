import os
if os.environ.get("NEJUMI_MAIN_STARTED") == "1":
    print("Duplicate invocation detected; skipping.")
    raise SystemExit(0)
os.environ["NEJUMI_MAIN_STARTED"] = "1"

import json
import time
from pathlib import Path
from argparse import ArgumentParser
from omegaconf import OmegaConf
import questionary
import importlib
import importlib.util

from utils import paginate_choices


class LazyEvaluatorModule:
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.module = None

    def __getattr__(self, item):
        if self.module is None:
            self.module = importlib.import_module(f"evaluator.{self.module_name}")
        return getattr(self.module, item)


jaster = LazyEvaluatorModule("jaster")
jbbq = LazyEvaluatorModule("jbbq")
mtbench = LazyEvaluatorModule("mtbench")
jaster_translation = LazyEvaluatorModule("jaster_translation")
toxicity = LazyEvaluatorModule("toxicity")
bfcl = LazyEvaluatorModule("bfcl")
jtruthfulqa = LazyEvaluatorModule("jtruthfulqa")
hle = LazyEvaluatorModule("hle")
hallulens = LazyEvaluatorModule("hallulens")
hallulens_zh_tw = LazyEvaluatorModule("hallulens_zh_tw")
m_ifeval = LazyEvaluatorModule("m_ifeval")
aggregate = LazyEvaluatorModule("aggregate")
aggregate_taiwan = LazyEvaluatorModule("aggregate_taiwan")
swe_bench = LazyEvaluatorModule("swe_bench")
swebench_pro = LazyEvaluatorModule("swebench_pro")
agentic_math = LazyEvaluatorModule("agentic_math")
arc_agi = LazyEvaluatorModule("arc_agi")
ifeval_zh_tw = LazyEvaluatorModule("ifeval_zh_tw")
ts_bench = LazyEvaluatorModule("ts_bench")
twbias = LazyEvaluatorModule("twbias")
tceval_v2 = LazyEvaluatorModule("tceval_v2")
script_adherence = LazyEvaluatorModule("script_adherence")

BENCHMARK_MAP = {
    'bfcl': 'bfcl',
    'agentic_math': 'agentic_math',
    'swebench': 'swebench',
    'swebench_pro': 'swebench_pro',
    'mtbench': 'mtbench',
    'script_adherence': 'script_adherence',
    'jbbq': 'jbbq',
    'toxicity': 'toxicity',
    'jtruthfulqa': 'jtruthfulqa',
    'hle': 'hle',
    'hallulens': 'hallulens',
    'hallulens_zh_tw': 'hallulens_zh_tw',
    'arc_agi': 'arc_agi',
    'm_ifeval': 'm_ifeval',
    'ifeval_zh_tw': 'ifeval_zh_tw',
    'ts_bench': 'ts_bench',
    'twbias': 'twbias',
    'tceval_v2': 'tceval_v2',
    'jaster': 'jaster',
    'aggregate': 'aggregate',
    'aggregate_taiwan': 'aggregate_taiwan',
}

AUXILIARY_RUN_FLAGS = {
    # Consumed inside jaster.evaluate() rather than dispatched as a standalone evaluator.
    "jmmlu_robustness",
    "tmmluplus_robustness",
}

KNOWN_RUN_FLAGS = set(BENCHMARK_MAP) | AUXILIARY_RUN_FLAGS


def load_validate_all_benchmarks():
    validation_helpers_path = (
        Path(__file__).resolve().parent
        / "evaluator"
        / "evaluate_utils"
        / "validation_helpers.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_nejumi_validation_helpers",
        validation_helpers_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load validation helper: {validation_helpers_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_all_benchmarks


validate_all_benchmarks = load_validate_all_benchmarks()


def _run_flags_dict(cfg) -> dict:
    run_cfg = OmegaConf.select(cfg, "run", default={})
    if run_cfg is None:
        return {}
    return OmegaConf.to_container(run_cfg, resolve=True) or {}


def is_run_flag_enabled(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def enabled_benchmarks_from_config(cfg):
    enabled = []
    for bench_key, bench_name in BENCHMARK_MAP.items():
        if is_run_flag_enabled(OmegaConf.select(cfg, f"run.{bench_key}", default=False)):
            enabled.append(bench_name)
    return enabled


def dispatch_validation_from_config(cfg):
    run_flags = _run_flags_dict(cfg)
    unsupported_truthy = sorted(
        key
        for key, value in run_flags.items()
        if is_run_flag_enabled(value) and key not in KNOWN_RUN_FLAGS
    )
    return {
        "ok": not unsupported_truthy,
        "known_run_flags": sorted(KNOWN_RUN_FLAGS),
        "dispatched_run_flags": [
            key
            for key in BENCHMARK_MAP
            if is_run_flag_enabled(run_flags.get(key, False))
        ],
        "auxiliary_run_flags": [
            key
            for key in sorted(AUXILIARY_RUN_FLAGS)
            if is_run_flag_enabled(run_flags.get(key, False))
        ],
        "unsupported_truthy_run_flags": unsupported_truthy,
    }


def summarize_token_validation(cfg, enabled_benchmarks):
    validation_results = validate_all_benchmarks(cfg)
    rows = []
    has_warnings = False
    has_errors = False

    for benchmark, (is_valid, message) in validation_results.items():
        if benchmark not in enabled_benchmarks:
            continue
        severity = "ok"
        if not is_valid:
            if "❌" in message:
                has_errors = True
                severity = "error"
            else:
                has_warnings = True
                severity = "warning"
        rows.append(
            {
                "benchmark": benchmark,
                "ok": bool(is_valid),
                "severity": severity,
                "message": message,
            }
        )

    return {
        "ok": not has_errors,
        "has_errors": has_errors,
        "has_warnings": has_warnings,
        "results": rows,
    }


def print_token_validation_summary(validation_summary):
    print("\n" + "=" * 80)
    print("🔍 GLOBAL TOKEN ALLOCATION VALIDATION")
    print("=" * 80)
    for row in validation_summary["results"]:
        print(row["message"])
    print("=" * 80)

    if validation_summary["has_errors"]:
        print("\n❌ CRITICAL: Some benchmarks have insufficient output tokens!")
        print("   This will likely cause empty responses and unfairly low scores.")
    elif validation_summary["has_warnings"]:
        print("\n⚠️  WARNING: Some benchmarks have suboptimal token allocation.")
    else:
        print("\n✅ All token allocations look good!")


def build_preflight_payload(custom_cfg_path, base_cfg_path, cfg, enabled_benchmarks):
    validation_summary = summarize_token_validation(cfg, enabled_benchmarks)
    dispatch_validation = dispatch_validation_from_config(cfg)
    ok = bool(validation_summary["ok"]) and bool(dispatch_validation["ok"])
    return {
        "schema_version": 1,
        "generated_at": time.time(),
        "status": "passed" if ok else "failed",
        "ok": ok,
        "config": str(custom_cfg_path),
        "base_config": str(base_cfg_path),
        "wandb": {
            "entity": OmegaConf.select(cfg, "wandb.entity"),
            "project": OmegaConf.select(cfg, "wandb.project"),
            "run_name": OmegaConf.select(cfg, "wandb.run_name"),
        },
        "api": OmegaConf.select(cfg, "api"),
        "model": OmegaConf.select(cfg, "model.pretrained_model_name_or_path"),
        "enabled_benchmarks": enabled_benchmarks,
        "scheduled_evaluators": enabled_benchmarks,
        "dispatch_validation": dispatch_validation,
        "will_initialize_wandb": False,
        "will_log_wandb_artifacts": False,
        "will_initialize_weave": False,
        "will_start_inference_engine": False,
        "will_run_evaluators": False,
        "token_validation": validation_summary,
    }

# Set config path
config_dir = Path("configs")
base_cfg_name = "base_config.yaml"
parser = ArgumentParser()
parser.add_argument("--config", "-c", type=str)
parser.add_argument("--select-config", "-s", action="store_true", default=False)
parser.add_argument("--base-config", type=str, default=base_cfg_name)
parser.add_argument("--yes", "-y", action="store_true", default=False)
parser.add_argument(
    "--preflight",
    action="store_true",
    default=False,
    help="Load and validate the merged config, then exit before W&B, Weave, model, or evaluator execution.",
)
parser.add_argument(
    "--preflight-json",
    type=str,
    default=None,
    help="Optional JSON output path for --preflight.",
)
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
base_cfg_path = config_dir / args.base_config
base_cfg = OmegaConf.load(base_cfg_path)

# vLLM利用時にbase_urlが未指定の場合、Composeサービス名 'vllm' をデフォルト設定
if "api" in custom_cfg and custom_cfg.api in ["vllm", "vllm-docker"]:
    if "base_url" not in custom_cfg:
        print("INFO: api is vllm/vllm-docker and base_url is not set. Defaulting to http://vllm:8000/v1")
        custom_cfg.base_url = "http://vllm:8000/v1"

custom_cfg = OmegaConf.merge(base_cfg, custom_cfg)
cfg_dict = OmegaConf.to_container(custom_cfg, resolve=True)
assert isinstance(cfg_dict, dict), "instance.config must be a DictConfig"
enabled_benchmarks = enabled_benchmarks_from_config(custom_cfg)
dispatch_validation = dispatch_validation_from_config(custom_cfg)

if args.preflight:
    payload = build_preflight_payload(
        custom_cfg_path=custom_cfg_path,
        base_cfg_path=base_cfg_path,
        cfg=custom_cfg,
        enabled_benchmarks=enabled_benchmarks,
    )
    print_token_validation_summary(payload["token_validation"])
    print("\nPreflight summary:")
    print(f"  config: {payload['config']}")
    print(f"  base_config: {payload['base_config']}")
    print(f"  model: {payload['model']}")
    print(f"  enabled_benchmarks: {', '.join(enabled_benchmarks) if enabled_benchmarks else '(none)'}")
    print("  W&B/Weave/model/evaluator execution: skipped")

    if args.preflight_json:
        preflight_json_path = Path(args.preflight_json)
        preflight_json_path.parent.mkdir(parents=True, exist_ok=True)
        preflight_json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"  preflight_json: {preflight_json_path}")

    raise SystemExit(0 if payload["ok"] else 2)

if not dispatch_validation["ok"]:
    raise SystemExit(
        "Unsupported truthy run flag(s): "
        + ", ".join(dispatch_validation["unsupported_truthy_run_flags"])
        + ". Add an evaluator dispatch or mark the flag as auxiliary before running."
    )

import wandb
import weave
from blend_run import blend_run
from config_singleton import WandbConfigSingleton
from docker_vllm_manager import stop_vllm_container_if_needed, start_vllm_container_if_needed
from llm_inference_adapter import get_llm_inference_engine
from vllm_server import shutdown_vllm_server

# プログレストラッカーをインポート
from evaluator.evaluate_utils.progress_tracker import (
    initialize_progress_tracker, start_benchmark_tracking,
    complete_benchmark_tracking, finish_progress_tracking
)

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
            settings=wandb.Settings(init_timeout=300),
        )
        os.environ["NEJUMI_WANDB_INIT_DONE"] = "1"
except Exception as e:
    raise SystemExit(
        "Failed to initialize W&B. W&B is required for this evaluation because "
        "artifacts, run metadata, and result logging must all be tracked there.\n"
        f"Original error: {e}\n"
        "Check your WANDB_API_KEY / login state and verify the target entity/project."
    ) from e

# Initialize Weave separately so Weave failures don't disable W&B
if run:
    try:
        weave.init(cfg_dict["wandb"]["entity"]+"/"+cfg_dict["wandb"]["project"])
    except Exception as e:
        print(f"Warning: Failed to initialize Weave: {e}")
        print("Continuing without Weave...")

WandbConfigSingleton.initialize(run, llm=None)
cfg = WandbConfigSingleton.get_instance().config

# Save configuration as artifact
artifact = wandb.Artifact("config", type="config")
artifact.add_file(custom_cfg_path)
run.log_artifact(artifact)

# Inherit old runs
blend_run(run_chain=True)

# グローバルトークンバリデーション実行
try:
    validation_summary = summarize_token_validation(cfg, enabled_benchmarks)
    print_token_validation_summary(validation_summary)

    if validation_summary["has_errors"]:
        response = "y" if args.yes else input("\nContinue anyway? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Evaluation aborted by user.")
            if run:
                run.finish()
            exit(1)
    elif validation_summary["has_warnings"]:
        response = "y" if args.yes else input("\nContinue? (Y/n): ").strip().lower()
        if response in ['n', 'no']:
            print("Evaluation aborted by user.")
            if run:
                run.finish()
            exit(1)
except Exception as e:
    print(f"⚠️  Token validation failed: {e}")
    print("Proceeding with evaluation...")

# プログレストラッカーを初期化
tracker = initialize_progress_tracker(enabled_benchmarks)
tracker.start_tracking()

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
    start_benchmark_tracking('bfcl')
    bfcl.evaluate()
    complete_benchmark_tracking('bfcl')

# Agentic Math evaluation
if cfg.run.get('agentic_math', False):
    start_benchmark_tracking('agentic_math')
    agentic_math.evaluate()
    complete_benchmark_tracking('agentic_math')

# SWE-Bench Verified evaluation
if cfg.run.swebench:
    start_benchmark_tracking('swebench')
    if cfg.swebench.background_eval:
        # 評価プロセスの実行時間が長いため、他のベンチと並行でバックグラウンド実行する
        # evaluate() はコールバック（wait_and_log_metrics）を返す実装に統一
        swebench_postprocess = swe_bench.evaluate()
    else:
        swe_bench.evaluate()
    complete_benchmark_tracking('swebench')

# SWE-bench Pro agentic evaluation
if cfg.run.get('swebench_pro', False):
    start_benchmark_tracking('swebench_pro')
    swebench_pro.evaluate()
    complete_benchmark_tracking('swebench_pro')

# mt-bench evaluation
if is_run_flag_enabled(cfg.run.get("mtbench", False)):
    start_benchmark_tracking('mtbench')
    mtbench.evaluate()
    complete_benchmark_tracking('mtbench')

# Traditional Chinese script adherence, derived from mtbench_output_table.
if is_run_flag_enabled(cfg.run.get("script_adherence", False)):
    start_benchmark_tracking('script_adherence')
    script_adherence.evaluate()
    complete_benchmark_tracking('script_adherence')

# jbbq
if cfg.run.jbbq:
    start_benchmark_tracking('jbbq')
    jbbq.evaluate()
    complete_benchmark_tracking('jbbq')

# toxicity
if cfg.run.toxicity:
    start_benchmark_tracking('toxicity')
    toxicity.evaluate()
    complete_benchmark_tracking('toxicity')

# JTruthfulQA
if cfg.run.jtruthfulqa:
    start_benchmark_tracking('jtruthfulqa')
    jtruthfulqa.evaluate()
    complete_benchmark_tracking('jtruthfulqa')

# hle
if cfg.run.hle:
    start_benchmark_tracking('hle')
    hle.evaluate()
    complete_benchmark_tracking('hle')

# HalluLens
if cfg.run.hallulens:
    start_benchmark_tracking('hallulens')
    hallulens.evaluate()
    complete_benchmark_tracking('hallulens')

# HalluLens zh-TW
if is_run_flag_enabled(cfg.run.get("hallulens_zh_tw", False)):
    start_benchmark_tracking('hallulens_zh_tw')
    hallulens_zh_tw.evaluate()
    complete_benchmark_tracking('hallulens_zh_tw')

# ARC-AGI
if cfg.run.arc_agi:
    start_benchmark_tracking('arc_agi')
    arc_agi.evaluate()
    complete_benchmark_tracking('arc_agi')

# M-IFEval
if cfg.run.m_ifeval:
    start_benchmark_tracking('m_ifeval')
    m_ifeval.evaluate()
    complete_benchmark_tracking('m_ifeval')

# IFEval zh-TW
if is_run_flag_enabled(cfg.run.get("ifeval_zh_tw", False)):
    start_benchmark_tracking('ifeval_zh_tw')
    ifeval_zh_tw.evaluate()
    complete_benchmark_tracking('ifeval_zh_tw')

# TS-Bench
if is_run_flag_enabled(cfg.run.get("ts_bench", False)):
    start_benchmark_tracking('ts_bench')
    ts_bench.evaluate()
    complete_benchmark_tracking('ts_bench')

# TWBias
if is_run_flag_enabled(cfg.run.get("twbias", False)):
    start_benchmark_tracking('twbias')
    twbias.evaluate()
    complete_benchmark_tracking('twbias')

# TCEval-v2 selected
if is_run_flag_enabled(cfg.run.get("tceval_v2", False)):
    start_benchmark_tracking('tceval_v2')
    tceval_v2.evaluate()
    complete_benchmark_tracking('tceval_v2')

# Evaluation phase
if cfg.run.jaster:
    start_benchmark_tracking('jaster')
    # llm-jp-eval evaluation
    jaster.evaluate()
    complete_benchmark_tracking('jaster')

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
    start_benchmark_tracking('aggregate')
    aggregate.evaluate()
    complete_benchmark_tracking('aggregate')

# Taiwan leaderboard aggregation
if is_run_flag_enabled(cfg.run.get("aggregate_taiwan", False)):
    start_benchmark_tracking('aggregate_taiwan')
    aggregate_taiwan.evaluate()
    complete_benchmark_tracking('aggregate_taiwan')

# プログレストラッキング終了
finish_progress_tracking()

# 評価完了後、vLLMコンテナを停止
if cfg.api in ["vllm", "vllm-docker"]:
    print("Stopping vLLM container...")
    # stop_vllm_container_if_needed()
    pass

# Finish
if run:
    run.finish()
