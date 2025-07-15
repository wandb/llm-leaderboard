"""
BFCL evaluation module
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple, Any
import wandb
import pandas as pd
from pathlib import Path
import datetime
import hashlib
from config_singleton import WandbConfigSingleton
from types import SimpleNamespace

# Import directly from bfcl_pkg
from evaluator.evaluate_utils.bfcl_pkg import generation_main, evaluation_main
from evaluator.evaluate_utils.bfcl_pkg.bfcl.constants.eval_config import RESULT_PATH, SCORE_PATH
from evaluator.evaluate_utils.bfcl_pkg.bfcl.utils import extract_test_category, load_file
from evaluator.evaluate_utils.bfcl_pkg.bfcl.constants.eval_config import PROMPT_PATH, POSSIBLE_ANSWER_PATH

def get_default_config() -> Dict[str, Any]:
    """BFCLのデフォルト設定を返す"""
    return {
        "model_name": None,  # モデル名
        "test_category": "all",  # テストするカテゴリのリスト
        "temperature": 0.001,  # 生成時の温度パラメータ
        "num_gpus": 1,  # 使用するGPUの数
        "num_threads": 1,  # 使用するスレッド数
        "gpu_memory_utilization": 0.9,  # GPU メモリ使用率
        "backend": "vllm",  # 使用するバックエンド
        "skip_server_setup": True,  # サーバーセットアップをスキップするかどうか（既存のvLLMサービスを使用）
        "local_model_path": None,  # ローカルモデルのパス（動的に設定される）
        "allow_overwrite": False,  # 既存の結果を上書きするかどうか
        "include_input_log": False,  # 推論ログに入力ログを含めるかどうか
        "exclude_state_log": False,  # 推論ログから状態ログを除外するかどうか
        "result_dir": RESULT_PATH,
        "score_dir": SCORE_PATH,
        "run_ids": False,  # テストエントリーIDを実行するかどうか
        "samples_per_category": None,  # 各カテゴリから取得するサンプル数
        "artifacts_path": None,  # WandB Artifactのパス
    }


def merge_config(default_config: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """デフォルト設定とユーザー設定をマージする"""
    merged = default_config.copy()
    if config:
        for key, value in config.items():
            if value is not None:  # Noneの場合はデフォルト値を使用
                merged[key] = value
    return merged

def evaluate():
    """BFCLの評価を実行する"""
    print("BFCLの評価を開始します")
    # WandbConfigSingletonからインスタンスを取得
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
     
    # デフォルト設定を取得
    default_config = get_default_config()
    
    # configからbfcl設定を取得し、デフォルト値とマージ
    user_config = cfg.get('bfcl', {})
    bfcl_cfg = merge_config(default_config, user_config)
    
    # BFCL対応モデル名がある場合はそれを使用、なければ従来の方法を使用
    if hasattr(cfg.model, 'bfcl_model_name') and cfg.model.bfcl_model_name:
        bfcl_cfg['model_name'] = cfg.model.bfcl_model_name
    else:
        bfcl_cfg['model_name'] = cfg.model.pretrained_model_name_or_path

    bfcl_cfg['backend'] = cfg.api

    # testmodeの場合はサンプル数を1に設定
    if cfg.testmode:
        bfcl_cfg['samples_per_category'] = 1
    
    # 結果保存用のディレクトリを作成
    result_path = bfcl_cfg['result_dir']
    os.makedirs(result_path, exist_ok=True)

    # 必須パラメータのチェック
    if not bfcl_cfg['model_name']:
        raise ValueError("model_name must be specified in the configuration")
    
    # download artifacts
    artifact = run.use_artifact(bfcl_cfg['artifacts_path'])
    artifact_dir = artifact.download()
    
    # generate arguments
    gen_args = SimpleNamespace(
        model=bfcl_cfg['model_name'],
        test_category=bfcl_cfg['test_category'],
        temperature=bfcl_cfg['temperature'],
        include_input_log=bfcl_cfg['include_input_log'],
        exclude_state_log=bfcl_cfg['exclude_state_log'],
        num_gpus=bfcl_cfg['num_gpus'],
        num_threads=bfcl_cfg['num_threads'],
        gpu_memory_utilization=bfcl_cfg['gpu_memory_utilization'],
        backend=bfcl_cfg['backend'],
        skip_server_setup=bfcl_cfg['skip_server_setup'],
        local_model_path=bfcl_cfg['local_model_path'],
        result_dir=bfcl_cfg['result_dir'],
        allow_overwrite=bfcl_cfg['allow_overwrite'],
        run_ids=bfcl_cfg['run_ids'],
        samples_per_category=bfcl_cfg['samples_per_category'],
        artifacts_path=artifact_dir,
    )
    # Prediction 
    generation_main(gen_args)

    # Evaluation
    _, _, _, overall_df = evaluation_main(
        model=[bfcl_cfg['model_name']],
        test_categories=bfcl_cfg['test_category'],
        result_dir=bfcl_cfg['result_dir'],
        score_dir=bfcl_cfg['score_dir'],
        samples_per_category=bfcl_cfg['samples_per_category'],
        artifacts_path=artifact_dir
    )
    

    # Leaderboard Table
    print(f"DEBUG: Overall DataFrame columns: {list(overall_df.columns)}")
    columns_to_drop = ['Organization','License','Model Link','Cost ($ Per 1k Function Calls)','Latency Mean (s)','Latency Standard Deviation (s)','Latency 95th Percentile (s)']

    # Only drop columns that exist
    existing_columns_to_drop = [col for col in columns_to_drop if col in overall_df.columns]
    print(f"DEBUG: Dropping columns: {existing_columns_to_drop}")
    overall_df.drop(columns=existing_columns_to_drop, inplace=True)
    overall_df.rename(columns={'Model': 'BFCL Model Name'}, inplace=True)
    overall_df["model_name"] = cfg.model.pretrained_model_name_or_path
    table_metric = wandb.Table(dataframe=overall_df)

    # Radar chart Table 
    radar_data = []
    first_row = overall_df.iloc[0]  
    radar_data.append(['Non-Live Acc', first_row['Non-Live Acc']])
    radar_data.append(['Live Acc', first_row['Live Acc']])
    radar_data.append(['Multi Turn Acc', first_row['Multi Turn Acc']])
    radar_data.append(['Irrelevance Detection', first_row['Irrelevance Detection']])
    radar_data.append(['Overall Acc', first_row['Overall Acc']]) 
    radar_df = pd.DataFrame(radar_data, columns=['category', 'score'])
    table_radar = wandb.Table(dataframe=radar_df)

    # Prediction Log
    def flatten_and_sum(value):
        if isinstance(value, (int, float)):
            return value
        elif isinstance(value, list):
            return sum(flatten_and_sum(item) for item in value)
        else:
            return 0
    
    print("DEBUG: Starting bfcl_output_table generation...")
    model_dir = Path(bfcl_cfg['score_dir']) / bfcl_cfg['model_name'].replace("/", "_")
    results = []
    
    print(f"DEBUG: Looking for score files in: {model_dir}")
    print(f"DEBUG: Model directory exists: {model_dir.exists()}")
    
    # Extract information from score files (contains all test case details)
    score_files = list(model_dir.glob("BFCL_v3_*_score.json"))
    print(f"DEBUG: Found {len(score_files)} score files: {[f.name for f in score_files]}")
    
    for json_file in score_files:
        test_category = extract_test_category(json_file.name)
        print(f"DEBUG: Processing file: {json_file.name}, category: {test_category}")
        
        with open(json_file, 'r') as f:
            lines = f.readlines()
            print(f"DEBUG: File {json_file.name} has {len(lines)} lines")
            # Skip the first line (overall accuracy summary)
            for line_num, line in enumerate(lines[1:], 2):
                if line.strip():
                    entry = json.loads(line)
                    entry_id = entry.get('id', '')
                    print(f"DEBUG: Processing entry {entry_id} from line {line_num}")
                    
                    # Extract prompt text
                    prompt_text = ""
                    if 'prompt' in entry:
                        prompt_entry = entry['prompt']
                        if 'question' in prompt_entry and prompt_entry['question']:
                            question_data = prompt_entry['question']
                            if isinstance(question_data, list) and len(question_data) > 0:
                                # Check if this is a multi-turn conversation
                                if len(question_data) > 1:
                                    # Multi-turn: show all instructions
                                    prompt_parts = []
                                    for i, turn in enumerate(question_data):
                                        if isinstance(turn, list) and len(turn) > 0:
                                            first_message = turn[0]
                                            if isinstance(first_message, dict) and 'content' in first_message:
                                                prompt_parts.append(f"Turn {i+1}: {first_message['content']}")
                                    prompt_text = " | ".join(prompt_parts)
                                else:
                                    # Single turn: show first instruction
                                    first_question = question_data[0]
                                    if isinstance(first_question, list) and len(first_question) > 0:
                                        first_message = first_question[0]
                                        if isinstance(first_message, dict) and 'content' in first_message:
                                            prompt_text = first_message['content']
                    
                    # Prepare result entry
                    # Handle different field names used by different evaluation functions
                    output_raw = entry.get('model_result_raw', entry.get('model_result', ''))
                    possible_answer_raw = entry.get('possible_answer', '')
                    
                    result_entry = {
                        'id': entry_id,
                        'category': test_category,
                        'prompt': prompt_text,
                        'output': str(output_raw),  # Check both model_result_raw and model_result
                        'accuracy': entry.get('success', 0),  # Use success field (1=success, 0=failure)
                        'possible_answer': str(possible_answer_raw),
                    }
                    
                    # Add error information only if it exists
                    if entry.get('error') is not None:
                        result_entry['error'] = str(entry.get('error', ''))
                    
                    results.append(result_entry)
                    print(f"DEBUG: Added result entry for {entry_id}")
    
    print(f"DEBUG: Total results collected: {len(results)}")
    
    if len(results) > 0:
        print(f"DEBUG: Sample result: {results[0]}")
    else:
        print("DEBUG: No results found!")
    
    print(f"DEBUG: Creating wandb table with {len(results)} results")
    table_log = wandb.Table(dataframe=pd.DataFrame(results))
    print(f"DEBUG: Wandb table created successfully")

    # Log
    print("DEBUG: Starting wandb logging...")
    run.log({
            "bfcl_output_table": table_log,
            "bfcl_leaderboard_table": table_metric,
            "bfcl_radar_table": table_radar,
        })
    print("DEBUG: Wandb logging completed successfully")

    print("BFCLの評価が正常に完了しました！")