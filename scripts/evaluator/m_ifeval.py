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
    """M-IFEval 결과를 평가합니다."""
    
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    
    # 모델명 추출
    model_name_safe = cfg.model.pretrained_model_name_or_path.replace('/', '__')

    inputs = read_prompt_list(input_data)
    prompt_to_response = read_prompt_to_response_dict(output_data)
    
    for eval_type, eval_func in [("strict", test_instruction_following_strict)]:
        print(f"\n=== {eval_type.upper()} 평가 시작 ===")
        outputs = []
        for inp in tqdm(inputs, desc=f"{eval_type} 평가 진행"):
            outputs.append(eval_func(inp, prompt_to_response))
        
        # 메트릭 계산 (W&B 로깅용)
        instruction_total = sum(len(o.instruction_id_list) for o in outputs)
        instruction_correct = sum(sum(o.follow_instruction_list) for o in outputs)
        run.log({
            "m_ifeval_leaderboard_table": wandb.Table(
                columns=["model_name", "m_ifeval_score"],
                data=[[model_name_safe, instruction_correct / instruction_total]]
            )
        })
    
    return instruction_correct / instruction_total

def evaluate():
    """M-IFEval 평가를 위한 response 생성"""
    # Retrieve the instance from WandbConfigSingleton and load the W&B run and configuration
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    llm = instance.llm

    model_name_safe = cfg.model.pretrained_model_name_or_path.replace('/', '__')

    # M-IFEval 데이터셋 경로 설정
    input_data_path = Path("../M-IFEval/data/ja_input_data.jsonl")
    if not input_data_path.exists():
        print(f"Input data not found: {input_data_path}")
        return

    # 입력 데이터 로드
    input_data = []
    with open(input_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            input_data.append(data)

    # 테스트 모드인 경우 적은 수의 샘플만 처리
    if cfg.get('testmode', False):
        input_data = input_data[:5]
        print(f"테스트 모드: {len(input_data)}개의 샘플만 처리합니다.")

    # 각 프롬프트에 대해 메시지 형식으로 변환
    evaluation_results = []
    for idx, data in enumerate(input_data):
        prompt = data['prompt']
        
        # 메시지 형식으로 변환
        messages = [{"role": "user", "content": prompt}]
        
        # generator config 설정
        generator_config = {"max_tokens": 2048}
        
        inputs = [messages, generator_config]
        
        # 평가 결과 정보 저장
        evaluation_results.append({
            "key": data['key'],
            "prompt": prompt,
            "instruction_id_list": data['instruction_id_list'],
            "kwargs": data['kwargs'],
            "inputs": inputs,
            "response": None,  # 나중에 채워질 예정
        })

    # 모든 입력을 준비
    all_inputs = [er["inputs"] for er in evaluation_results]
    
    print("LLM으로 응답 생성 중...")
    
    # LLM으로 응답 생성
    llm_ap = LLMAsyncProcessor(
        llm=llm,
        inputs=all_inputs,
    )
    results = llm_ap.get_results()
    
    # 결과를 evaluation_results에 저장
    for response, evaluation_result in tqdm(zip(results, evaluation_results), 
                                          desc="응답 결과 처리"):
        evaluation_result["response"] = response

    # 출력 파일 저장
    output_data = []
    for er in evaluation_results:
        # AIMessage 객체에서 텍스트 내용 추출
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

    table_data = [[model_name_safe, data['key'], data['prompt'], data['response'], str(data['instruction_id_list']), str(data['kwargs'])] for data in output_data]

    # log output_data as wandb table
    run.log({
        "m_ifeval_output_table": wandb.Table(
            columns=["model_name", "key", "prompt", "response", "instruction_id_list", "kwargs"],
            data=table_data
        )
    })

    return evaluate_m_ifeval(input_data, output_data)