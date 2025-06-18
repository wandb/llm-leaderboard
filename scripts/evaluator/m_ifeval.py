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
import collections
import dataclasses
from typing import Dict, Optional, Sequence, Union

from config_singleton import WandbConfigSingleton
from .evaluate_utils import (
    apply_chat_template,
    LLMAsyncProcessor,
)

# M-IFEval 라이브러리 경로 추가
sys.path.append(str(Path("../M-IFEval").resolve()))

try:
    from evaluation_main import (
        read_prompt_list,
        read_prompt_to_response_dict,
        test_instruction_following_strict,
        test_instruction_following_loose,
        print_report,
        write_outputs
    )
    print("M-IFEval 평가 모듈을 성공적으로 불러왔습니다.")
except ImportError as e:
    print(f"M-IFEval 평가 모듈을 불러올 수 없습니다: {e}")
    print("M-IFEval 디렉토리가 올바른 위치에 있는지 확인하세요.")
    read_prompt_list = None
    read_prompt_to_response_dict = None
    test_instruction_following_strict = None
    test_instruction_following_loose = None
    print_report = None
    write_outputs = None

@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[Dict[str, Optional[Union[str, int]]]]

@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]

def evaluate_m_ifeval():
    """M-IFEval 결과를 평가합니다."""
    if read_prompt_list is None:
        print("M-IFEval 모듈을 불러올 수 없어 평가를 수행할 수 없습니다.")
        return None
    
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    
    # 모델명 추출
    model_name_safe = cfg.model.pretrained_model_name_or_path.replace('/', '__')
    
    # 입력 데이터 및 응답 데이터 경로
    input_data_path = Path("../M-IFEval/data/ja_input_data.jsonl")
    response_data_path = Path(f"../M-IFEval/data/ja_input_response_data_{model_name_safe}.jsonl")
    
    print(f"입력 데이터: {input_data_path}")
    print(f"응답 데이터: {response_data_path}")
    
    # 파일 존재 확인
    if not input_data_path.exists():
        print(f"입력 데이터를 찾을 수 없습니다: {input_data_path}")
        return None
    
    if not response_data_path.exists():
        print(f"응답 데이터를 찾을 수 없습니다: {response_data_path}")
        print("먼저 evaluate() 함수를 실행하여 응답을 생성하세요.")
        return None
    
    # M-IFEval의 기존 함수들을 사용하여 데이터 로드
    inputs = read_prompt_list(str(input_data_path))
    prompt_to_response = read_prompt_to_response_dict(str(response_data_path))
    
    print(f"총 {len(inputs)}개의 프롬프트를 평가합니다.")
    
    # 출력 디렉토리 설정
    output_dir = Path(f"../M-IFEval/evaluations/ja_input_response_data_{model_name_safe}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # strict 및 loose 평가 수행
    evaluation_results = {}
    
    for eval_type, eval_func in [
        ("strict", test_instruction_following_strict),
        ("loose", test_instruction_following_loose),
    ]:
        print(f"\n=== {eval_type.upper()} 평가 시작 ===")
        
        outputs = []
        for inp in tqdm(inputs, desc=f"{eval_type} 평가 진행"):
            outputs.append(eval_func(inp, prompt_to_response))
        
        # 정확도 계산
        follow_all_instructions = [o.follow_all_instructions for o in outputs]
        accuracy = sum(follow_all_instructions) / len(outputs)
        print(f"{eval_type} 평가 정확도: {accuracy:.4f}")
        
        # M-IFEval의 기존 함수를 사용하여 결과 저장
        output_file = output_dir / f"eval_results_{eval_type}.jsonl"
        write_outputs(str(output_file), outputs)
        
        print(f"결과 저장됨: {output_file}")
        
        # 상세 리포트 출력
        print(f"\n=== {eval_type.upper()} 평가 상세 리포트 ===")
        
        # M-IFEval의 기존 print_report 함수 사용
        print_report(outputs)
        
        # 메트릭 계산 (W&B 로깅용)
        prompt_total = len(outputs)
        prompt_correct = sum(follow_all_instructions)
        instruction_total = sum(len(o.instruction_id_list) for o in outputs)
        instruction_correct = sum(sum(o.follow_instruction_list) for o in outputs)
        
        metrics = {
            "prompt_level_accuracy": prompt_correct / prompt_total,
            "instruction_level_accuracy": instruction_correct / instruction_total,
        }
        
        # 평가 결과 저장
        evaluation_results[eval_type] = {
            "accuracy": accuracy,
            "metrics": metrics,
            "output_file": str(output_file)
        }
        
        # W&B 로깅
        run.log({
            f"m_ifeval_{eval_type}_accuracy": accuracy,
            f"m_ifeval_{eval_type}_prompt_level_accuracy": metrics["prompt_level_accuracy"],
            f"m_ifeval_{eval_type}_instruction_level_accuracy": metrics["instruction_level_accuracy"],
        })
    
    print("\n=== M-IFEval 평가 완료 ===")
    print(f"Strict 정확도: {evaluation_results['strict']['accuracy']:.4f}")
    print(f"Loose 정확도: {evaluation_results['loose']['accuracy']:.4f}")
    
    return evaluation_results

def evaluate():
    """M-IFEval 평가를 위한 response 생성"""
    # Retrieve the instance from WandbConfigSingleton and load the W&B run and configuration
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    llm = instance.llm

    # M-IFEval 데이터셋 경로 설정
    input_data_path = Path("../M-IFEval/data/ja_input_data.jsonl")
    if not input_data_path.exists():
        print(f"Input data not found: {input_data_path}")
        return

    # 출력 파일 경로 설정
    model_name_safe = cfg.model.pretrained_model_name_or_path.replace('/', '__')
    output_dir = Path("../M-IFEval/data")
    output_file = output_dir / f"ja_input_response_data_{model_name_safe}.jsonl"
    
    print(f"입력 파일: {input_data_path}")
    print(f"출력 파일: {output_file}")

    # 입력 데이터 로드
    input_data = []
    with open(input_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            input_data.append(data)

    print(f"총 {len(input_data)}개의 프롬프트를 처리합니다.")

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

    # 출력 디렉토리가 없으면 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSONL 파일로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for data in output_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"Response 파일이 저장되었습니다: {output_file}")
    
    # W&B에 결과 로깅
    run.log({
        "m_ifeval_total_samples": len(output_data),
        "m_ifeval_output_file": str(output_file),
    })

    # 간단한 통계 출력
    response_lengths = [len(data["response"]) if data["response"] else 0 for data in output_data]
    print(f"응답 길이 통계:")
    print(f"  평균: {np.mean(response_lengths):.1f} 문자")
    print(f"  최소: {np.min(response_lengths)} 문자")
    print(f"  최대: {np.max(response_lengths)} 문자")
    
    # 빈 응답 개수 체크
    empty_responses = sum(1 for data in output_data if not data["response"] or not data["response"].strip())
    if empty_responses > 0:
        print(f"경고: {empty_responses}개의 빈 응답이 있습니다.")
    
    print("M-IFEval response 생성이 완료되었습니다!")
    
    # 응답 생성 후 자동으로 평가 수행
    print("\n응답 생성이 완료되었습니다. 이제 평가를 시작합니다...")
    return evaluate_m_ifeval()
