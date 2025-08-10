import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from toolz import pipe
from tqdm import tqdm
import wandb
import sys
import os

from config_singleton import WandbConfigSingleton


from .evaluate_utils.m_ifeval_utils import (
    read_prompt_list,
    read_prompt_to_response_dict,
    test_instruction_following_strict,
)
from .evaluate_utils import (
    apply_chat_template,
    LLMAsyncProcessor,
)

def evaluate_m_ifeval(input_data, output_data):
    """M-IFEval結果を評価します。"""
    
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    
    # モデル名抽出
    model_name_safe = cfg.model.pretrained_model_name_or_path.replace('/', '__')

    inputs = read_prompt_list(input_data)
    prompt_to_response = read_prompt_to_response_dict(output_data)
    
    print(f"\n=== STRICT 評価開始 ===")
    outputs = []
    for inp in tqdm(inputs, desc=f"strict 評価実行"):
        outputs.append(test_instruction_following_strict(inp, prompt_to_response))
    
    # メトリクス計算 (W&B ログ用)
    instruction_total = sum(len(o.instruction_id_list) for o in outputs)
    instruction_correct = sum(sum(o.follow_instruction_list) for o in outputs)
    run.log({
        "m_ifeval_leaderboard_table": wandb.Table(
            columns=["model_name", "m_ifeval_score"],
            data=[[model_name_safe, instruction_correct / instruction_total]]
        )
    })
    scores = [o.score for o in outputs]

    # log output_data as wandb table
    table_data = [[model_name_safe, data['key'], data['prompt'], data['response'], str(data['instruction_id_list']), str(data['kwargs']), score] for data, score in zip(output_data, scores)]
    run.log({
        "m_ifeval_output_table": wandb.Table(
            columns=["model_name", "key", "prompt", "response", "instruction_id_list", "kwargs", "score"],
            data=table_data
        )
    })
    
    return instruction_correct / instruction_total

def evaluate():
    """M-IFEval評価のためのresponse生成"""
    # Retrieve the instance from WandbConfigSingleton and load the W&B run and configuration
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    llm = instance.llm

    # M-IFEvalデータセットパス設定

    dataset_name = "m_ifeval"
    artifact = run.use_artifact(cfg[dataset_name].artifacts_path, type="dataset")
    artifact_dir = artifact.download()
    input_data_path = Path(artifact_dir) / cfg[dataset_name].dataset_dir
    print(f"input_data_path: {input_data_path}")
    # input_data_path = Path("../M-IFEval/data/ja_input_data.jsonl")
    if not input_data_path.exists():
        print(f"Input data not found: {input_data_path}")
        return

    # 入力データロード
    input_data = []
    with open(input_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            input_data.append(data)

    # テストモードの場合は少数のサンプルのみ処理
    if cfg.get('testmode', False):
        input_data = input_data[:5]
        print(f"テストモード: {len(input_data)}個のサンプルのみ処理します。")

    # 各プロンプトに対してメッセージ形式に変換
    evaluation_results = []
    for idx, data in enumerate(input_data):
        prompt = data['prompt']
        
        # メッセージ形式に変換
        messages = [{"role": "user", "content": prompt}]
        
        # generator config設定（YAMLがあれば優先。なければ既定2048）
        gcfg = cfg.m_ifeval.get("generator_config", {}) if hasattr(cfg, 'm_ifeval') else {}
        # 後方互換: 既存の max_completion_tokens などを取り込む
        alias = (
            gcfg.get("max_tokens")
            or gcfg.get("max_new_token")
            or gcfg.get("max_new_tokens")
            or gcfg.get("max_output_tokens")
            or gcfg.get("max_completion_tokens")
        )
        generator_config = {"max_tokens": alias if alias is not None else 2048}
        
        inputs = [messages, generator_config]
        
        # 評価結果情報保存
        evaluation_results.append({
            "key": data['key'],
            "prompt": prompt,
            "instruction_id_list": data['instruction_id_list'],
            "kwargs": data['kwargs'],
            "inputs": inputs,
            "response": None,  # 後で埋められる予定
            "score": None,  # 後で埋められる予定
        })

    # 全ての入力を準備
    all_inputs = [er["inputs"] for er in evaluation_results]
    
    print("LLMで応答生成中...")
    
    # LLMで応答生成
    llm_ap = LLMAsyncProcessor(
        llm=llm,
        inputs=all_inputs,
    )
    results = llm_ap.get_results()
    
    # 結果をevaluation_resultsに保存
    for response, evaluation_result in tqdm(zip(results, evaluation_results), 
                                          desc="応答結果処理"):
        evaluation_result["response"] = response

    # 出力ファイル保存
    output_data = []
    for er in evaluation_results:
        # AIMessageオブジェクトからテキスト内容抽出
        response_text = ""
        if er["response"]:
            if hasattr(er["response"], 'content'):
                response_text = er["response"].content
            elif isinstance(er["response"], str):
                response_text = er["response"]
            else:
                response_text = str(er["response"])
        
        output_data.append({
            "key": er["key"],
            "prompt": er["prompt"],
            "response": response_text,
            "instruction_id_list": er["instruction_id_list"],
            "kwargs": er["kwargs"]
        })

    return evaluate_m_ifeval(input_data, output_data)