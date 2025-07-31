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
        "temperature": 0.01,  # 生成時の温度パラメータ
        "num_gpus": None,  # 使用するGPUの数 (vllmを使うことになるので、Nejumi Leaderboardでは、これは利用しない。)
        "batch_size": 256, # 使用するGPUの数 (Nejumi Leaderboardでは、これは利用しない。)
        "num_threads": 1,  # 使用するスレッド数
        "gpu_memory_utilization": 0.9,  # GPU メモリ使用率
        "backend": "vllm",  # 使用するバックエンド
        "skip_server_setup": True,  # サーバーセットアップをスキップするかどうか（既存のvLLMサービスを使用）
        "local_model_path": None,  # ローカルモデルのパス（動的に設定される）
        "allow_overwrite": True,  # 既存の結果を上書きするかどうか
        "include_input_log": False,  # 推論ログに入力ログを含めるかどうか
        "exclude_state_log": False,  # 推論ログから状態ログを除外するかどうか
        "result_dir": RESULT_PATH,
        "score_dir": SCORE_PATH,
        "run_ids": False,  # テストエントリーIDを実行するかどうか
        "samples_per_category": 50,  # 各カテゴリから取得するサンプル数
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
    
    bfcl_cfg['model_name'] = cfg.model.bfcl_model_id


    # testmodeの場合はサンプル数を1に設定
    if cfg.testmode:
        bfcl_cfg['samples_per_category'] = 2
    
    # 結果保存用のディレクトリを作成
    result_path = bfcl_cfg['result_dir']
    os.makedirs(result_path, exist_ok=True)

    # download artifacts
    artifact = run.use_artifact(bfcl_cfg['artifacts_path'])
    artifact_dir = artifact.download()
    

    
    # generate arguments
    gen_args = SimpleNamespace(
        model_name=bfcl_cfg['model_name'],
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
        model_names=[bfcl_cfg['model_name'].replace("/", "_")],
        test_categories=bfcl_cfg['test_category'],  # This now contains only available categories
        result_dir=bfcl_cfg['result_dir'],
        score_dir=bfcl_cfg['score_dir'],
        samples_per_category=bfcl_cfg['samples_per_category'],
        artifacts_path=artifact_dir
    )
    
    # Leaderboard Table
    columns_to_drop = ['Organization','License','Model Link','Cost ($ Per 1k Function Calls)','Latency Mean (s)','Latency Standard Deviation (s)','Latency 95th Percentile (s)']

    # Only drop columns that exist
    existing_columns_to_drop = [col for col in columns_to_drop if col in overall_df.columns]
    overall_df.drop(columns=existing_columns_to_drop, inplace=True)
    overall_df.rename(columns={'Model': 'BFCL Model Name'}, inplace=True)
    overall_df["model_name"] = cfg.model.pretrained_model_name_or_path
    table_metric = wandb.Table(dataframe=overall_df)

    # Radar chart Table 
    radar_data = []
    first_row = overall_df.iloc[0]  
    print(first_row)
    
    # Helper function to convert empty strings to 0.0 for numeric display
    def convert_to_numeric(value):
        if pd.isna(value) or value == '' or value == 'N/A':
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    radar_data.append(['Non-Live Ast Acc', convert_to_numeric(first_row['Non-Live AST Acc'])])
    radar_data.append(['Live Ast Acc', convert_to_numeric(first_row['Live AST Acc'])])
    radar_data.append(['Multi Turn Acc', convert_to_numeric(first_row['Multi Turn Acc'])])
    radar_data.append(['Irrelevance Detection', convert_to_numeric(first_row['Irrelevance Detection'])])
    radar_data.append(['Overall Acc', convert_to_numeric(first_row['Overall Acc'])]) 
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
    
    model_dir = Path(bfcl_cfg['score_dir']) / bfcl_cfg['model_name'].replace("/", "_")
    results = []
    processed_ids = set()  # Track which IDs we've already processed
    
    # Extract information from score files (contains all test case details)
    score_files = list(model_dir.glob("BFCL_v3_*_score.json"))
    
    # Also check result files for comprehensive data
    result_model_dir = Path(bfcl_cfg['result_dir']) / bfcl_cfg['model_name'].replace("/", "_")
    result_files = list(result_model_dir.glob("BFCL_v3_*.json")) if result_model_dir.exists() else []
    
    # Create a map of entry_id to result data for easier lookup
    result_data_map = {}
    for result_file in result_files:
        test_category = extract_test_category(result_file.name)
        try:
            with open(result_file, 'r') as f:
                result_data = []
                for line in f:
                    if line.strip():  # Skip empty lines
                        entry = json.loads(line)
                        result_data.append(entry)
                
                for entry in result_data:
                    if 'id' in entry:
                        result_data_map[entry['id']] = {
                            'category': test_category,
                            'raw_data': entry
                        }
        except Exception as e:
            print(f"Warning: Could not read result file {result_file}: {e}")
    
    # Process score files first (these have detailed evaluation results)
    for json_file in score_files:
        test_category = extract_test_category(json_file.name)
        
        with open(json_file, 'r') as f:
            lines = f.readlines()
            # Skip the first line (overall accuracy summary)
            for line_num, line in enumerate(lines[1:], 2):
                if line.strip():
                    entry = json.loads(line)
                    entry_id = entry.get('id', '')
                    processed_ids.add(entry_id)  # Mark as processed
                    
                    # Extract prompt text (prioritize score file, fallback to result file)
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
                    
                    # Fallback to result file data if prompt is empty
                    if not prompt_text and entry_id in result_data_map:
                        result_entry_data = result_data_map[entry_id]['raw_data']
                        if 'question' in result_entry_data:
                            question_data = result_entry_data['question']
                            if isinstance(question_data, list) and len(question_data) > 0:
                                if len(question_data) > 1:
                                    prompt_parts = []
                                    for i, turn in enumerate(question_data):
                                        if isinstance(turn, list) and len(turn) > 0:
                                            first_message = turn[0]
                                            if isinstance(first_message, dict) and 'content' in first_message:
                                                prompt_parts.append(f"Turn {i+1}: {first_message['content']}")
                                    prompt_text = " | ".join(prompt_parts)
                                else:
                                    first_question = question_data[0]
                                    if isinstance(first_question, list) and len(first_question) > 0:
                                        first_message = first_question[0]
                                        if isinstance(first_message, dict) and 'content' in first_message:
                                            prompt_text = first_message['content']
                    
                    # Prepare result entry
                    # Handle different field names used by different evaluation functions
                    output_raw = entry.get('model_result_raw', entry.get('model_result', ''))
                    possible_answer_raw = entry.get('possible_answer', '')
                    
                    # Helper function to get max value from potentially nested arrays
                    def get_max_token_count(token_data):
                        if isinstance(token_data, (list, tuple)):
                            # Flatten nested arrays and get maximum
                            flat_list = []
                            for item in token_data:
                                if isinstance(item, (list, tuple)):
                                    flat_list.extend(item)
                                else:
                                    flat_list.append(item)
                            return max(flat_list) if flat_list else 0
                        elif isinstance(token_data, (int, float)):
                            return token_data
                        else:
                            return 0
                    
                    result_entry = {
                        'id': entry_id,
                        'category': test_category,
                        'prompt': prompt_text,
                        'output': str(output_raw),
                        'accuracy': entry.get('success', 0),  # Use success field (1=success, 0=failure)
                        'possible_answer': str(possible_answer_raw),
                        'input_token_count': get_max_token_count(entry.get('input_token_count', 0)),
                        'output_token_count': get_max_token_count(entry.get('output_token_count', 0)),
                    }   
                    
                    # Add error information only if it exists
                    if entry.get('error') is not None:
                        result_entry['error'] = str(entry.get('error', ''))
                    
                    results.append(result_entry)
    
    # Now process any result entries that weren't found in score files
    # These might be successful cases that weren't included in score files
    for entry_id, result_info in result_data_map.items():
        if entry_id not in processed_ids:  # Only process if not already processed
            result_entry_data = result_info['raw_data']
            test_category = result_info['category']
            
            # Extract prompt text from result file
            prompt_text = ""
            if 'question' in result_entry_data:
                question_data = result_entry_data['question']
                if isinstance(question_data, list) and len(question_data) > 0:
                    if len(question_data) > 1:
                        prompt_parts = []
                        for i, turn in enumerate(question_data):
                            if isinstance(turn, list) and len(turn) > 0:
                                first_message = turn[0]
                                if isinstance(first_message, dict) and 'content' in first_message:
                                    prompt_parts.append(f"Turn {i+1}: {first_message['content']}")
                        prompt_text = " | ".join(prompt_parts)
                    else:
                        first_question = question_data[0]
                        if isinstance(first_question, list) and len(first_question) > 0:
                            first_message = first_question[0]
                            if isinstance(first_message, dict) and 'content' in first_message:
                                prompt_text = first_message['content']
            
            # Get model output from result file
            output_raw = result_entry_data.get('model_result', '')
            
            result_entry = {
                'id': entry_id,
                'category': test_category,
                'prompt': prompt_text,
                'output': str(output_raw),
                'accuracy': 1,  # Assume success if only in result file (might need adjustment)
                'possible_answer': '',  # Not available in result files
                'input_token_count': result_entry_data.get('input_token_count', 0),
                'output_token_count': result_entry_data.get('output_token_count', 0),
            }
            
            results.append(result_entry)
            
    table_log = wandb.Table(dataframe=pd.DataFrame(results))

    # Log
    run.log({
            "bfcl_output_table": table_log,
            "bfcl_leaderboard_table": table_metric,
            "bfcl_radar_table": table_radar,
        })

    print("BFCLの評価が正常に完了しました！")