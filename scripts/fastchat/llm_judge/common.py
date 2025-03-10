"""
Common data structures and utilities.
"""

import ast
import dataclasses
import glob
import json
import os
import re
import time
from typing import Optional

import openai
import anthropic
import cohere
import google.generativeai as genai

from fastchat.model.model_adapter import (
    get_conversation_template,
    ANTHROPIC_MODEL_LIST,
    OPENAI_MODEL_LIST,
)

from config_singleton import WandbConfigSingleton
from tenacity import retry, stop_after_attempt, wait_fixed


# API setting constants
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"

TIE_DELTA = 0.1

# Categories that need reference answers
NEED_REF_CATS = ["math", "reasoning", "coding", "arena-hard-200"]

# Extract scores from judgments
two_score_pattern = re.compile("\[\[(\d+\.?\d*),\s?(\d+\.?\d*)\]\]")
two_score_pattern_backup = re.compile("\[(\d+\.?\d*),\s?(\d+\.?\d*)\]")
one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")

# TODO: (meng) thinking about changing setting for japanese llm usage
# Sampling temperature configs for
temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
    "arena-hard-200": 0.0,
}

reverse_model_map = {
    "model_1": "model_2",
    "model_2": "model_1",
}


@dataclasses.dataclass
class Judge:
    model_name: str
    prompt_template: dict
    ref_based: bool = False
    multi_turn: bool = False


@dataclasses.dataclass
class MatchSingle:
    question: dict
    model: str
    answer: dict
    judge: Judge
    ref_answer: dict = None
    multi_turn: bool = False


@dataclasses.dataclass
class MatchPair:
    question: dict
    model_1: str
    model_2: str
    answer_1: dict
    answer_2: dict
    judge: Judge
    ref_answer: dict = None
    multi_turn: bool = False


def load_questions(question_file: str, begin: Optional[int], end: Optional[int]):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                q = json.loads(line)
                q['question_id'] = int(q['question_id'])
                questions.append(q)
    questions = questions[begin:end]
    return questions


def load_model_answers(answer_dir: str):
    """Load model answers.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[question_id: int -> answer: dict]]
    """
    filenames = glob.glob(os.path.join(answer_dir, "*.jsonl"))
    filenames.sort()
    model_answers = {}

    for filename in filenames:
        model_name = os.path.basename(filename)[:-6]
        answer = {}
        with open(filename) as fin:
            for line in fin:
                line = json.loads(line)
                answer[int(line["question_id"])] = line
        model_answers[model_name] = answer

    return model_answers


def load_judge_prompts(prompt_file: str):
    """Load judge prompts.

    The return value is a python dict of type:
    Dict[judge_name: str -> dict]
    """
    prompts = {}
    with open(prompt_file) as fin:
        for line in fin:
            line = json.loads(line)
            prompts[line["name"]] = line
    return prompts


def run_judge_single(question, answer, judge, ref_answer, multi_turn=False):
    kwargs = {}
    model = judge.model_name
    if ref_answer is not None:
        kwargs["ref_answer_1"] = ref_answer["choices"][0]["turns"][0]
        if multi_turn:
            kwargs["ref_answer_2"] = ref_answer["choices"][0]["turns"][1]

    if multi_turn:
        user_prompt = judge.prompt_template["prompt_template"].format(
            question_1=question["turns"][0],
            question_2=question["turns"][1],
            answer_1=answer["choices"][0]["turns"][0],
            answer_2=answer["choices"][0]["turns"][1],
            **kwargs,
        )
    else:
        user_prompt = judge.prompt_template["prompt_template"].format(
            question=question["turns"][0],
            answer=answer["choices"][0]["turns"][0],
            **kwargs,
        )

    rating = -1

    system_prompt = judge.prompt_template["system_prompt"]
    conv = get_conversation_template(model)
    conv.set_system_message(system_prompt)
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)

    instance = WandbConfigSingleton.get_instance()
    cfg = instance.config
    api_type = cfg.toxicity.get('judge_api_type', 'openai')
    if api_type=="azure-opanai" or "openai":
        print("Selecting API_Type (OpenAI or Azure OpenAI)")
        judgment = chat_completion_azure_fallback(model, conv, temperature=0, max_tokens=2048)
    elif model in ANTHROPIC_MODEL_LIST:
        judgment = chat_completion_anthropic(
            model, conv, temperature=0, max_tokens=1024
        )
    else:
        raise ValueError(f"Invalid judge model name: {model}")

    if judge.prompt_template["output_format"] in ["[[rating]]", "[[評価]]", "[[평가]]"]:
        match = re.search(one_score_pattern, judgment)
        if not match:
            match = re.search(one_score_pattern_backup, judgment)

        if match:
            rating = ast.literal_eval(match.groups()[0])
        else:
            rating = -1
    else:
        raise ValueError(
            f"invalid output format: {judge.prompt_template['output_format']}"
        )

    return rating, user_prompt, judgment


def play_a_match_single(match: MatchSingle, output_file: str):
    question, model, answer, judge, ref_answer, multi_turn = (
        match.question,
        match.model,
        match.answer,
        match.judge,
        match.ref_answer,
        match.multi_turn,
    )

    if judge.prompt_template["type"] == "single":
        score, user_prompt, judgment = run_judge_single(
            question, answer, judge, ref_answer, multi_turn=multi_turn
        )

        question_id = question["question_id"]
        turn = 1 if not multi_turn else 2
        result = {
            "question_id": question_id,
            "model": model,
            "judge": (judge.model_name, judge.prompt_template["name"]),
            "user_prompt": user_prompt,
            "judgment": judgment,
            "score": score,
            "turn": turn,
            "tstamp": time.time(),
        }
    else:
        raise ValueError(f"invalid judge type: {judge['type']}")

    if output_file:
        output_file = os.path.join(
            output_file.replace(".jsonl", ""),
            f"{model}__{turn}turn.jsonl"
        )
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "a") as fout:
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    return result


def run_judge_pair(question, answer_a, answer_b, judge, ref_answer, multi_turn=False):
    kwargs = {}
    model = judge.model_name
    if ref_answer is not None:
        kwargs["ref_answer_1"] = ref_answer["choices"][0]["turns"][0]
        if multi_turn:
            kwargs["ref_answer_2"] = ref_answer["choices"][0]["turns"][1]

    if multi_turn:
        system_prompt = judge.prompt_template["system_prompt"]
        user_prompt = judge.prompt_template["prompt_template"].format(
            question_1=question["turns"][0],
            question_2=question["turns"][1],
            answer_a_1=answer_a["choices"][0]["turns"][0],
            answer_b_1=answer_b["choices"][0]["turns"][0],
            answer_a_2=answer_a["choices"][0]["turns"][1],
            answer_b_2=answer_b["choices"][0]["turns"][1],
            **kwargs,
        )
    else:
        system_prompt = judge.prompt_template["system_prompt"]
        user_prompt = judge.prompt_template["prompt_template"].format(
            question=question["turns"][0],
            answer_a=answer_a["choices"][0]["turns"][0],
            answer_b=answer_b["choices"][0]["turns"][0],
            **kwargs,
        )

    winner = "error"

    conv = get_conversation_template(model)
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)

    if model in OPENAI_MODEL_LIST:
        conv.set_system_message(system_prompt)
        judgment = chat_completion_openai(model, conv, temperature=0, max_tokens=2048)
    elif model in ANTHROPIC_MODEL_LIST:
        if system_prompt != "You are a helpful assistant.":
            user_prompt = "[Instruction]\n" + system_prompt + "\n\n" + user_prompt
            conv.messages[0][1] = user_prompt
        judgment = chat_completion_anthropic(
            model, conv, temperature=0, max_tokens=1024
        )
    else:
        raise ValueError(f"Invalid judge model name: {model}")

    if judge.prompt_template["output_format"] == "[[A]]":
        if "[[A]]" in judgment:
            winner = "A"
        elif "[[B]]" in judgment:
            winner = "B"
        elif "[[C]]" in judgment:
            winner = "tie"
        else:
            winner = "error"
    elif judge.prompt_template["output_format"] == "[[rating_a,rating_b]]":
        match = re.search(two_score_pattern, judgment)
        if not match:
            match = re.search(two_score_pattern_backup, judgment)
        if match:
            scores = [ast.literal_eval(s.strip()) for s in match.groups()]
            if abs(scores[0] - scores[1]) <= TIE_DELTA:
                winner = "tie"
            elif scores[0] > scores[1]:
                winner = "A"
            else:
                winner = "B"
        else:
            winner = "error"
    else:
        raise ValueError(
            f"invalid output format: {judge.prompt_template['output_format']}"
        )

    return winner, user_prompt, judgment


def play_a_match_pair(match: MatchPair, output_file: str):
    question, model_1, model_2, answer_1, answer_2, judge, ref_answer, multi_turn = (
        match.question,
        match.model_1,
        match.model_2,
        match.answer_1,
        match.answer_2,
        match.judge,
        match.ref_answer,
        match.multi_turn,
    )

    if judge.prompt_template["type"] == "pairwise":
        g1_winner, g1_user_prompt, g1_judgment = run_judge_pair(
            question, answer_1, answer_2, judge, ref_answer, multi_turn=multi_turn
        )
        g2_winner, g2_user_prompt, g2_judgment = run_judge_pair(
            question, answer_2, answer_1, judge, ref_answer, multi_turn=multi_turn
        )

        g1_map = {"A": "model_1", "B": "model_2"}
        g2_map = {"A": "model_2", "B": "model_1"}
        g1_winner = g1_map.get(g1_winner, g1_winner)
        g2_winner = g2_map.get(g2_winner, g2_winner)
        question_id = question["question_id"]
        turn = 1 if not multi_turn else 2

        result = {
            "question_id": question_id,
            "model_1": model_1,
            "model_2": model_2,
            "g1_winner": g1_winner,
            "g2_winner": g2_winner,
            "judge": (judge.model_name, judge.prompt_template["name"]),
            "g1_user_prompt": g1_user_prompt,
            "g1_judgment": g1_judgment,
            "g2_user_prompt": g2_user_prompt,
            "g2_judgment": g2_judgment,
            "turn": turn,
            "tstamp": time.time(),
        }

        print(
            f"question: {question_id}, turn: {turn}, model_1: {model_1}, model_2: {model_2}, "
            f"g1_winner: {g1_winner}, g2_winner: {g2_winner}, "
            f"judge: {(judge.model_name, judge.prompt_template['name'])}"
        )
    elif judge.prompt_template["type"] == "single":
        m1_score, m1_user_prompt, m1_judgment = run_judge_single(
            question, answer_1, judge
        )
        m2_score, m2_user_prompt, m2_judgment = run_judge_single(
            question, answer_2, judge
        )

        if abs(m1_score - m2_score) <= TIE_DELTA:
            winner = "tie"
        elif m1_score > m2_score:
            winner = "model_1"
        else:
            winner = "model_2"

        question_id = question["question_id"]
        result = {
            "question_id": question_id,
            "model_1": model_1,
            "model_2": model_2,
            "g1_winner": winner,
            "g2_winner": winner,
            "judge": (judge.model_name, judge.prompt_template["name"]),
            "g1_user_prompt": m1_user_prompt,
            "g1_judgment": m1_judgment,
            "g2_user_prompt": m2_user_prompt,
            "g2_judgment": m2_judgment,
            "m1_score": m1_score,
            "m2_score": m2_score,
            "tstamp": time.time(),
        }
        print(
            f"question: {question_id}, model_1: {model_1}, model_2: {model_2}, "
            f"winner: {winner}, m1_score: {m1_score}, m2_score: {m2_score}, "
            f"judge: {(judge.model_name, judge.prompt_template['name'])}"
        )
    else:
        raise ValueError(f"invalid judge type: {judge['type']}")

    if output_file:
        output_file = os.path.join(
            output_file.replace(".jsonl", ""),
            f"{model_1}__{model_2}__{turn}turn.jsonl"
        )
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "a") as fout:
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    return result
    
def setup_openai_api(model: str, use_azure=False):
    from functools import partial

    if model == "gpt-3.5-turbo":
        deployment_id = "misc-35"
    elif model == "gpt-4":
        deployment_id = "misc-4"
    else:
        raise NotImplementedError(f"{model=}")

    if use_azure:
        openai.api_type = "azure"
        openai.api_key = os.environ['OPENAI_AZURE_API_KEY']
        openai.api_base = os.environ['OPENAI_AZURE_API_BASE']
        openai.api_version = "2023-05-15"  # subject to change
        return partial(openai.ChatCompletion.create, deployment_id=deployment_id)
    else:
        openai.api_key = os.environ['OPENAI_API_KEY']
        return openai.ChatCompletion.create

def chat_completion_azure_fallback(model, conv, temperature, max_tokens):
    api_type = os.environ.get('OPENAI_API_TYPE', 'openai')
    print(f"API Type: {api_type}")
    if api_type == "azure":
        print('Azure OpenAI is being used.')
        return chat_completion_openai_azure(model, conv, temperature, max_tokens)
    else:
        print('OpenAI is being used.')
        return chat_completion_openai(model, conv, temperature, max_tokens)

def chat_completion_upstage(model, conv, temperature, max_tokens):
    #openai_chat_completion_func = setup_openai_api(model)
    client = openai.OpenAI(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    base_url="https://api.upstage.ai/v1/solar"
    )
    output = API_ERROR_OUTPUT
    # TODO: allow additional params for toggling between azure api
    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response.choices[0].message.content
            break
        except openai.OpenAIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

    return output

def chat_completion_xai(model, conv, temperature, max_tokens):
    #openai_chat_completion_func = setup_openai_api(model)
    client = openai.OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1"
    )
    output = API_ERROR_OUTPUT
    # TODO: allow additional params for toggling between azure api
    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response.choices[0].message.content
            break
        except openai.OpenAIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

    return output

def chat_completion_openai(model, conv, temperature, max_tokens):
    #openai_chat_completion_func = setup_openai_api(model)
    output = API_ERROR_OUTPUT
    # TODO: allow additional params for toggling between azure api
    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response.choices[0].message.content
            break
        except openai.OpenAIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

    return output

@retry(stop=stop_after_attempt(5), wait=wait_fixed(30))
def chat_completion_vllm(model, conv, temperature, max_tokens):
    from openai import OpenAI
    from config_singleton import WandbConfigSingleton

    instance = WandbConfigSingleton.get_instance()
    cfg = instance.config

    openai_api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    openai_api_base = cfg.get("base_url", "http://localhost:8000/v1")

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    # LoRAの設定を確認
    lora_config = cfg.model.get("lora", None)
    if lora_config and lora_config.get("enable", False):
        # LoRAが有効な場合、LoRAアダプター名をモデル名として使用
        model = lora_config.get("adapter_name", model)

    print(f"Using model: {model}")
    print(client.models.list())

    messages = conv.to_openai_api_messages()
    print(messages)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        n=1,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


from openai import AzureOpenAI
import os
import time

def chat_completion_openai_azure(model, conv, temperature, max_tokens, api_dict=None):
    if api_dict is not None:
        client = AzureOpenAI(
            azure_endpoint=api_dict["api_base"],
            api_key=api_dict["api_key"],
            api_version="2023-07-01-preview"
        )
    else:
        client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version="2023-07-01-preview"
        )

    if "azure-" in model:
        model = model[6:]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response.choices[0].message.content
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return output


def chat_completion_anthropic(model, conv, temperature, max_tokens, api_dict=None):
    if api_dict is not None and "api_key" in api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["ANTHROPIC_API_KEY"]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        if model in {"claude-3-opus-20240229", "claude-3-5-sonnet"}:
            llm = ChatAnthropic(model_name=model)
            max_retries = 3
            retry_count = 0
            prompt = conv.get_prompt()
            print(prompt)
            while retry_count < max_retries:
                try:
                    output = llm.invoke(prompt).content
                    # print(output)
                    time.sleep(60)
                    break 
                except Exception as e:
                    print(f"Error happened!!! : {e}")
                    retry_count += 1
                    time.sleep(1)
        else:
            try:
                c = anthropic.Anthropic(api_key=api_key)
                prompt = conv.get_prompt()
                response = c.completions.create(
                    model=model,
                    prompt=prompt,
                    stop_sequences=[anthropic.HUMAN_PROMPT],
                    max_tokens_to_sample=max_tokens,
                    temperature=temperature,
                )
                output = response.completion
                break
            except anthropic.APIError as e:
                print(type(e), e)
                time.sleep(API_RETRY_SLEEP)
    return output.strip()


def chat_completion_cohere(model, conv, temperature, max_tokens):
    import cohere
    import os
    import time

    output = API_ERROR_OUTPUT
    cohere_api_key = os.environ.get("COHERE_API_KEY")
    if not cohere_api_key:
        print("COHERE_API_KEY is not set in the environment variables.")
        return API_ERROR_OUTPUT

    for attempt in range(API_MAX_RETRY):
        try:
            # Use Cohere's ClientV2
            co = cohere.ClientV2(api_key=cohere_api_key)

            # Use OpenAI-style messages directly
            messages = conv.to_openai_api_messages()

            response = co.chat(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Get the generated message from the response
            output = response.message.content[0].text.strip()
            break  # Exit the loop if successful

        except Exception as e:
            print(f"An error occurred (attempt {attempt + 1}/{API_MAX_RETRY}): {e}")
            wait_time = API_RETRY_SLEEP
            print(f"Retrying in {wait_time} seconds.")
            time.sleep(wait_time)

    return output if output else API_ERROR_OUTPUT


def chat_completion_palm(chat_state, model, conv, temperature, max_tokens):
    from fastchat.serve.api_provider import init_palm_chat

    assert model == "palm-2-chat-bison-001"

    if chat_state is None:
        chat_state = init_palm_chat("chat-bison@001")

    parameters = {
        "temperature": temperature,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": max_tokens,
    }
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = chat_state.send_message(conv.messages[-2][1], **parameters)
            output = response.text
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return chat_state, output


def chat_completion_gemini(chat_state, model, conv, temperature, max_tokens):
    safety_settings_NONE=[
                            { "category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE" },
                            { "category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE" },
                            { "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE" },
                            { "category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                        ]

    # assert model == "gemini-pro"

    if chat_state is None:
        gemini = genai.GenerativeModel(
            model_name=model, safety_settings=safety_settings_NONE)
        chat_state = gemini.start_chat()

    parameters = {
        "temperature": temperature,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": max_tokens,
    }
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = chat_state.send_message(conv.messages[-2][1], generation_config=parameters)
            output = response.text
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return chat_state, output


'''def chat_completion_bedrock(chat_state, model, conv, temperature, max_tokens):
    from langchain_community.chat_models import BedrockChat
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        MessagesPlaceholder,
    )

    if chat_state is None:
        llm = BedrockChat(
            model_id=model,
            model_kwargs={"temperature":temperature},
        )

        memory = ConversationBufferMemory(return_messages=True)
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("""{input}""")
        ])

        chat_state = ConversationChain(llm=llm, prompt=prompt, memory=memory)

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = chat_state.run(conv.messages[-2][1])
            output = response
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return chat_state, output'''
import boto3
import json
import time
import os
from dataclasses import dataclass
from botocore.exceptions import ClientError
from langchain_community.chat_models import BedrockChat
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

@dataclass
class BedrockResponse:
    content: str

def chat_completion_bedrock(chat_state, model, conv, temperature, max_tokens):
    if chat_state is None:
        if "anthropic" in model.lower():
            llm = BedrockChat(
                model_id=model,
                model_kwargs={"temperature": temperature, "max_tokens": max_tokens},
            )

            memory = ConversationBufferMemory(return_messages=True)
            prompt = ChatPromptTemplate.from_messages([
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}")
            ])

            chat_state = ConversationChain(llm=llm, prompt=prompt, memory=memory)
        else:
            chat_state = ChatBedrock(model, temperature)

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            if isinstance(chat_state, ConversationChain):
                response = chat_state.run(conv.messages[-2][1])
                output = response
            else:
                messages = conv.to_openai_api_messages()
                response = chat_state.invoke(messages, max_tokens)
                output = response.content
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return chat_state, output

class ChatBedrock:
    def __init__(self, model_id, temperature) -> None:
        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
        )
        self.model_id = model_id
        self.generator_config = {"temperature": temperature}

    def _invoke(self, messages: list[dict[str, str]], max_tokens: int):
        # モデルタイプの判定
        is_llama = "llama" in self.model_id.lower()
        is_nova = "nova" in self.model_id.lower()

        if is_nova:
            # Novaモデル用の処理
            for message in messages:
                if isinstance(message['content'], str):
                    message['content'] = [{"text": message['content']}]

            inference_config = {
                "temperature": self.generator_config.get("temperature", 0.0),
                "maxTokens": max_tokens
            }

            body_dict = {
                "messages": messages,
                "inferenceConfig": inference_config
            }
        else:
            # 既存のLlama用処理
            prompt = self._format_llama_prompt(messages)
            body_dict = {
                "prompt": prompt,
                "max_gen_len": max_tokens,
                **self.generator_config,
            }

        try:
            if is_nova:
                response = self.bedrock_runtime.converse(
                    modelId=self.model_id,
                    **body_dict
                )
                response_body = response
            else:
                response = self.bedrock_runtime.invoke_model(
                    body=json.dumps(body_dict),
                    modelId=self.model_id
                )
                response_body = json.loads(response.get("body").read())
        except ClientError as e:
            print(f"ERROR: Can't invoke '{self.model_id}'. Reason: {e}")
            raise

        return response_body

    def invoke(self, messages, max_tokens: int):
        response = self._invoke(messages=messages, max_tokens=max_tokens)
        if "nova" in self.model_id.lower():
            content = response.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")
        else:
            content = response.get("generation", "")
        return BedrockResponse(content=content)

    def _format_llama_prompt(self, messages):
        formatted_messages = []
        for message in messages:
            if message["role"] == "system":
                formatted_messages.append(f"<|system|>\n{message['content']}\n")
            elif message["role"] == "user":
                formatted_messages.append(f"<|user|>\n{message['content']}\n")
            elif message["role"] == "assistant":
                formatted_messages.append(f"<|assistant|>\n{message['content']}\n")
        formatted_messages.append("<|assistant|>\n")  # Add for the model to continue
        return "<|begin_of_text|>\n" + "".join(formatted_messages)


def chat_completion_mistral(chat_state, model, conv, temperature, max_tokens):
    from langchain_mistralai.chat_models import ChatMistralAI
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        MessagesPlaceholder,
    )

    if chat_state is None:
        llm = ChatMistralAI(
            model=model,
            mistral_api_key=os.environ.get("MISTRAL_API_KEY"),
            temperature=temperature, 
            max_tokens=max_tokens,
        )

        memory = ConversationBufferMemory(return_messages=True)
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("""{input}""")
        ])

        chat_state = ConversationChain(llm=llm, prompt=prompt, memory=memory)

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = chat_state.run(conv.messages[-2][1])
            output = response
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return chat_state, output


def normalize_game_key_single(gamekey, result):
    """Make the model names sorted in a game key."""
    qid, model_1, model_2 = gamekey
    if model_1 < model_2:
        return gamekey, result
    else:
        new_gamekey = (qid, model_2, model_1)
        new_result = {
            "winners": tuple(reverse_model_map.get(x, x) for x in result["winners"]),
            "g1_judgment": result["g2_judgment"],
            "g2_judgment": result["g1_judgment"],
        }
        return new_gamekey, new_result


def normalize_game_key_dict(judgment_dict):
    """Make the model names sorted in the game keys."""
    ret = {}
    for key, value in judgment_dict.items():
        new_key, new_value = normalize_game_key_single(key, value)
        ret[new_key] = new_value
    return ret


def load_pairwise_model_judgments(filename: str):
    """Load model judgments.

    The return value is a dict of type:
    Dict[judge: Tuple -> Dict[game_key: tuple -> game_result: dict]
    """
    judge_dict = {}

    if not os.path.exists(filename):
        filenames = glob.glob(os.path.join(filename.replace(".jsonl", ""), "*.jsonl"))
    else:
        filenames = [filename]

    for filename in filenames:
        for line in open(filename):
            obj = json.loads(line)
            obj["question_id"] = int(obj["question_id"])
            judge = tuple(obj["judge"])
            qid, model_1, model_2 = obj["question_id"], obj["model_1"], obj["model_2"]

            if judge not in judge_dict:
                judge_dict[judge] = {}

            if "winner" in obj:
                winner = obj["winner"]
            elif "g1_winner" in obj and "g2_winner" in obj:
                g1_winner, g2_winner = obj["g1_winner"], obj["g2_winner"]
                if g1_winner == g2_winner:
                    winner = g1_winner
                else:
                    winner = "inconsistent"
            else:
                raise ValueError(f"Invalid keys: {list(obj.keys())}")

            gamekey = (qid, model_1, model_2)
            winners = (winner,)

            judge_dict[judge][gamekey] = {
                "winners": winners,
                "g1_judgment": obj["g1_judgment"],
                "g2_judgment": obj["g2_judgment"],
            }

    # Make the model names sorted in the game keys
    normalized = {}
    for judge, value in judge_dict.items():
        normalized[judge] = normalize_game_key_dict(value)
    return normalized


def load_single_model_judgments(filename: str):
    """Load model judgments.

    The return value is a dict of type:
    Dict[judge: Tuple -> Dict[game_key: tuple -> game_result: dict]
    """
    judge_dict = {}

    if not os.path.exists(filename):
        filenames = glob.glob(os.path.join(filename.replace(".jsonl", ""), "*.jsonl"))
    else:
        filenames = [filename]

    for filename in filenames:
        for line in open(filename):
            obj = json.loads(line)
            obj["question_id"] = int(obj["question_id"])
            judge = tuple(obj["judge"])
            qid, model = obj["question_id"], obj["model"]

            if judge not in judge_dict:
                judge_dict[judge] = {}

            gamekey = (qid, model)

            judge_dict[judge][gamekey] = {
                "score": obj["score"],
                "judgment": obj["judgment"],
            }
    return judge_dict


def resolve_pairwise_judgment_dict(
    question, model_judgments_normal, model_judgments_math, multi_turn=False
):
    """Return the correct pairwise judge."""
    if multi_turn:
        if question["category"] in NEED_REF_CATS:
            return model_judgments_math[("gpt-4", "pair-math-v1-multi-turn")]
        return model_judgments_normal[("gpt-4", "pair-v2-multi-turn")]

    if question["category"] in NEED_REF_CATS:
        return model_judgments_math[("gpt-4", "pair-math-v1")]
    else:
        return model_judgments_normal[("gpt-4", "pair-v2")]


def resolve_single_judgment_dict(
    question, model_judgments_normal, model_judgments_math, multi_turn=False
):
    """Return the correct single answer grading judge."""
    if multi_turn:
        if question["category"] in NEED_REF_CATS:
            return model_judgments_math[("gpt-4", "single-math-v1-multi-turn")]
        return model_judgments_normal[("gpt-4", "single-v1-multi-turn")]

    if question["category"] in NEED_REF_CATS:
        return model_judgments_math[("gpt-4", "single-math-v1")]
    else:
        return model_judgments_normal[("gpt-4", "single-v1")]


def get_pairwise_judge_explanation(gamekey, judgment_dict):
    """Get model judge explanation."""
    try:
        qid, model_1, model_2 = gamekey
        if model_1 < model_2:
            res = judgment_dict[gamekey]
            g1_judgment, g2_judgment = res["g1_judgment"], res["g2_judgment"]
        else:
            new_gamekey = (qid, model_2, model_1)
            res = judgment_dict[new_gamekey]

            model_1, model_2 = model_1, model_2
            g1_judgment, g2_judgment = res["g2_judgment"], res["g1_judgment"]

        return (
            f"**Game 1**. **A**: {model_1}, **B**: {model_2}\n\n"
            f"**Judgment**: {g1_judgment}"
            + f"\n\n`--------------------------`\n\n"
            + f"**Game 2**. **A**: {model_2}, **B**: {model_1}\n\n"
            f"**Judgment**: {g2_judgment}"
        )
    except KeyError:
        return "N/A"


def get_single_judge_explanation(gamekey, judgment_dict):
    """Get model judge explanation."""
    try:
        qid, model = gamekey

        res = judgment_dict[gamekey]

        g1_judgment = res["judgment"]
        g1_score = res["score"]

        return (
            f"**Game 1**. **A**: {model}, **Score**: {g1_score}\n\n"
            f"**Judgment**: {g1_judgment}"
        )
    except KeyError:
        return "N/A"


def check_data(questions, model_answers, ref_answers, models, judges):
    # check model answers
    for m in models:
        assert m in model_answers, f"Missing model answer for {m}"
        m_answer = model_answers[m]
        for q in questions:
            assert (
                q["question_id"] in m_answer
            ), f"Missing model {m}'s answer to Question {q['question_id']}"
    # check ref answers
    for jg in judges.values():
        if not jg.ref_based:
            continue
        for q in questions:
            if q["category"] not in NEED_REF_CATS:
                continue
            assert (
                q["question_id"] in list(ref_answers.values())[0].keys()
            ), f"Missing reference answer to Question {q['question_id']} for judge {list(ref_answers.keys())[0]}"


def get_model_list(answer_dir):
    file_paths = glob.glob(f"{answer_dir}/*.jsonl")
    file_names = [os.path.splitext(os.path.basename(f))[0] for f in file_paths]
    return file_names