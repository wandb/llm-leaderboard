"""Generate answers with GPT-4

Usage:
python3 gen_api_answer.py --model gpt-3.5-turbo
"""
import argparse
import json
import os
import time
import concurrent.futures
from omegaconf import OmegaConf

import openai
import shortuuid
import tqdm
import wandb
from config_singleton import WandbConfigSingleton

from fastchat.llm_judge.common import (
    load_questions,
    temperature_config,
    chat_completion_openai,
    chat_completion_anthropic,
    chat_completion_cohere,
    chat_completion_palm,
    chat_completion_gemini,
    chat_completion_bedrock,
    chat_completion_mistral,
    chat_completion_vllm,
    chat_completion_upstage,
    chat_completion_openai_azure,
    chat_completion_xai,
)
from fastchat.llm_judge.gen_model_answer import reorg_answer_file
from fastchat.model.model_adapter import get_conversation_template, ANTHROPIC_MODEL_LIST


def get_api_answer(question_file, answer_file, num_worker = 1):
    cfg = WandbConfigSingleton.get_instance().config
    questions = load_questions(question_file, None, None)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_worker) as executor:
        futures = []
        for question in questions:
            future = executor.submit(
                get_answer,
                question,
                cfg.model.pretrained_model_name_or_path,
                cfg.mtbench.num_choices,
                cfg.mtbench.max_new_token,
                answer_file,
            )
            futures.append(future)

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            try:
                future.result()
            except Exception as e:
                import traceback
                print(f"Function raised an exception: {e}")
                print(f"Exception type: {type(e).__name__}")
                print(f"Traceback: {traceback.format_exc()}")

    reorg_answer_file(answer_file)


def get_answer(
    question: dict, model: str, num_choices: int, max_tokens: int, answer_file: str
):

    cfg = WandbConfigSingleton.get_instance().config
    # Debug: check cfg.api type and value
    # print(f"Debug: cfg.api = {cfg.api}, type = {type(cfg.api)}")
    
    # cfgからtemperature_overrideを取得
    temperature_override = cfg.mtbench.temperature_override

    # temperature_configをtemperature_overrideで更新
    temperature_config.update(temperature_override)

    if question["category"] in temperature_config:
        temperature = temperature_config[question["category"]]
    else:
        temperature = 0.7

    choices = []
    chat_state = None  # for palm-2, gemini and bedrock-claude model
    for i in range(num_choices):
        conv = get_conversation_template(model)

        turns = []
        for j in range(len(question["turns"])):
            conv.append_message(conv.roles[0], question["turns"][j])
            conv.append_message(conv.roles[1], None)

            api_type = cfg.get("api", cfg.api) if hasattr(cfg, "get") else cfg.api
            
            if api_type in ["vllm", "vllm-external"]:
                output = chat_completion_vllm(
                    model, conv, temperature, max_tokens
                )
            elif api_type == "anthropic":
                output = chat_completion_anthropic(
                    model, conv, temperature, max_tokens
                )
            elif model == "palm-2-chat-bison-001":
                chat_state, output = chat_completion_palm(
                    chat_state, model, conv, temperature, max_tokens
                )
            elif api_type == "cohere":
                output = chat_completion_cohere(
                    model, conv, temperature, max_tokens
                ) 
            elif api_type == "google":
                chat_state, output = chat_completion_gemini(
                    chat_state, model, conv, temperature, max_tokens
                ) 
            elif api_type == "amazon_bedrock":
                chat_state, output = chat_completion_bedrock(
                    chat_state, model, conv, temperature, max_tokens
                ) 
            elif api_type == "mistral":
                chat_state, output = chat_completion_mistral(
                    chat_state, model, conv, temperature, max_tokens
                )  
            elif api_type == "upstage":
                output = chat_completion_upstage(
                    model, conv, temperature, max_tokens
                )
            elif api_type == "azure-openai":
                output = chat_completion_openai_azure(
                    model, conv, temperature, max_tokens
                )
            elif api_type == "xai":
                output = chat_completion_xai(
                    model, conv, temperature, max_tokens
                )
            elif api_type == "openai_responses":
                from fastchat.llm_judge.common import chat_completion_openai_responses
                output = chat_completion_openai_responses(
                    model, conv, temperature, max_tokens
                )
            else:
                output = chat_completion_openai(
                    model, conv, temperature, max_tokens
                )

            conv.update_last_message(output)
            turns.append(output)

        choices.append({"index": i, "turns": turns})

    # Dump answers
    ans = {
        "question_id": question["question_id"],
        "answer_id": shortuuid.uuid(),
        "model_id": model,
        "choices": choices,
        "tstamp": time.time(),
    }

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "a") as fout:
        fout.write(json.dumps(ans, ensure_ascii=False) + "\n")
