import datetime
import hashlib
import json
import os
import asyncio
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import shortuuid
import time
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from config_singleton import WandbConfigSingleton
from .evaluate_utils.llm_async_processor import LLMAsyncProcessor
from llm_inference_adapter import get_llm_inference_engine

# カテゴリが参照回答を必要とするかどうかのリスト
NEED_REF_CATS = ["math", "reasoning", "coding", "arena-hard-200"]

# default settings for temperature
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

@dataclass
class Question:
    question_id: int
    category: str
    turns: List[str]

@dataclass
class Answer:
    question_id: int
    model_id: str
    choices: List[Dict[str, List[str]]]
    
@dataclass
class Judge:
    model_name: str
    prompt_template: Dict[str, Any]
    ref_based: bool = False
    multi_turn: bool = False

@dataclass
class MatchSingle:
    question: Question
    model: str
    answer: Answer
    judge: Judge
    ref_answer: Optional[Answer] = None
    multi_turn: bool = False

def load_questions(question_file: str, begin: Optional[int] = None, end: Optional[int] = None) -> List[Question]:
    """質問をファイルから読み込む"""
    questions = []
    with open(question_file, "r") as f:
        for line in f:
            if line:
                q = json.loads(line)
                q['question_id'] = int(q['question_id'])
                questions.append(Question(
                    question_id=q['question_id'],
                    category=q['category'],
                    turns=q['turns']
                ))
    if begin is not None and end is not None:
        questions = questions[begin:end]
    return questions

def load_model_answers(answer_dir: str) -> Dict[str, Dict[int, Answer]]:
    """モデル回答を読み込む"""
    import glob
    filenames = glob.glob(os.path.join(answer_dir, "*.jsonl"))
    filenames.sort()
    model_answers = {}

    for filename in filenames:
        model_name = os.path.basename(filename)[:-6]
        answer = {}
        with open(filename) as f:
            for line in f:
                line_data = json.loads(line)
                answer[int(line_data["question_id"])] = Answer(
                    question_id=int(line_data["question_id"]),
                    model_id=line_data["model_id"],
                    choices=line_data["choices"]
                )
        model_answers[model_name] = answer

    return model_answers

def load_judge_prompts(prompt_file: str) -> Dict[str, Dict[str, Any]]:
    """審査官プロンプトを読み込む"""
    prompts = {}
    with open(prompt_file) as f:
        for line in f:
            line_data = json.loads(line)
            prompts[line_data["name"]] = line_data
    return prompts

def make_judge_single(judge_model: str, judge_prompts: Dict[str, Dict[str, Any]]) -> Dict[str, Judge]:
    """単一モデル評価用のジャッジを作成"""
    judges = {}
    judges["default"] = Judge(
        model_name=judge_model,
        prompt_template=judge_prompts["single-v1"],
        ref_based=True,
    )
    judges["math"] = Judge(
        model_name=judge_model,
        prompt_template=judge_prompts["single-math-v1"],
        ref_based=True,
    )
    judges["default-mt"] = Judge(
        model_name=judge_model,
        prompt_template=judge_prompts["single-v1-multi-turn"],
        ref_based=True,
        multi_turn=True,
    )
    judges["math-mt"] = Judge(
        model_name=judge_model,
        prompt_template=judge_prompts["single-math-v1-multi-turn"],
        ref_based=True,
        multi_turn=True,
    )
    return judges

def make_match_single(
    questions: List[Question], 
    models: List[str], 
    model_answers: Dict[str, Dict[int, Answer]], 
    judge: Judge, 
    baseline_model: Optional[str] = None,
    ref_answers: Optional[Dict[str, Dict[int, Answer]]] = None, 
    multi_turn: bool = False
) -> List[MatchSingle]:
    """単一モデル評価のマッチを作成"""
    matches = []
    for question in questions:
        for model in models:
            if question.question_id not in model_answers[model]:
                continue
            
            ref_answer = None
            if ref_answers is not None and judge.ref_based:
                # 参照回答が必要なカテゴリかどうかを確認
                if question.category in NEED_REF_CATS:
                    ref_model = list(ref_answers.keys())[0]
                    if question.question_id in ref_answers[ref_model]:
                        ref_answer = ref_answers[ref_model][question.question_id]
            
            match = MatchSingle(
                question=question,
                model=model,
                answer=model_answers[model][question.question_id],
                judge=judge,
                ref_answer=ref_answer,
                multi_turn=multi_turn,
            )
            matches.append(match)
    return matches

async def generate_model_answer(question: Question, llm, model_id: str, max_tokens: int) -> Tuple[str, str]:
    """モデルの回答を生成する（非同期）"""
    # カテゴリに基づいてtemperatureを取得
    temperature = temperature_config.get(question.category, 0.7) # デフォルトは0.7
    
    # 一回目の質問に回答
    messages_turn1 = [{"role": "user", "content": question.turns[0]}]
    response_turn1 = await asyncio.to_thread(
        llm.invoke, 
        messages_turn1,
        max_tokens=max_tokens, # max_tokensを設定
        temperature=temperature # temperatureを設定
    )
    answer_turn1 = response_turn1.content if hasattr(response_turn1, 'content') else str(response_turn1) # 文字列に変換
    
    # 二回目の質問に回答（マルチターン）
    messages_turn2 = [
        {"role": "user", "content": question.turns[0]},
        {"role": "assistant", "content": answer_turn1},
        {"role": "user", "content": question.turns[1]}
    ]
    
    response_turn2 = await asyncio.to_thread(
        llm.invoke, 
        messages_turn2,
        max_tokens=max_tokens, # max_tokensを設定
        temperature=temperature # temperatureを設定
    )
    answer_turn2 = response_turn2.content if hasattr(response_turn2, 'content') else str(response_turn2) # 文字列に変換
    
    return answer_turn1, answer_turn2

async def generate_model_answers(questions: List[Question], output_file: str):
    """すべての質問に対するモデル回答を生成（LLMAsyncProcessorを使用）"""
    instance = WandbConfigSingleton.get_instance()
    cfg = instance.config
    llm = instance.llm
    model_id = cfg.mtbench.model_id
    # mtbenchの設定からmax_tokensを取得、なければデフォルト1024
    max_tokens = cfg.mtbench.get("max_new_token", 1024) # 修正: cfg.mtbench から取得
    # グローバルのtemperatureをデフォルト値として取得
    default_temperature = cfg.generator.get("temperature", 0.7) 
    # mtbenchのtemperatureオーバーライド設定を取得
    temperature_overrides = cfg.mtbench.get("temperature_override", {}) 

    # 出力ディレクトリを作成
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # LLMAsyncProcessor に渡す inputs を作成
    inputs_for_processor = []
    for question in questions:
        # カテゴリに基づいてtemperatureを取得 (オーバーライドがあれば優先、なければデフォルト)
        temperature = temperature_overrides.get(question.category, default_temperature) # 修正: cfg から取得
        
        # ターン1のメッセージとパラメータ
        messages_turn1 = [{"role": "user", "content": question.turns[0]}]
        params_turn1 = {"max_tokens": max_tokens, "temperature": temperature}
        inputs_for_processor.append((messages_turn1, params_turn1))
        
        # ターン2のメッセージとパラメータ（プレースホルダとして、ターン1の結果が必要）
        # まずはターン1の結果を取得してから、ターン2の inputs を作成する

    print(f"Generating answers for turn 1... (using LLMAsyncProcessor)")
    llm_ap_turn1 = LLMAsyncProcessor(llm=llm, inputs=[item for item in inputs_for_processor])
    # get_results を asyncio.to_thread で呼び出す
    results_turn1 = await asyncio.to_thread(llm_ap_turn1.get_results)

    answers_turn1 = [res.content if hasattr(res, 'content') else str(res) for res in results_turn1]

    # ターン2の inputs を作成
    inputs_for_processor_turn2 = []
    for i, question in enumerate(questions):
         # カテゴリに基づいてtemperatureを取得 (ターン2も同じ設定を使う)
        temperature = temperature_overrides.get(question.category, default_temperature) # 修正: cfg から取得
        # ターン2のメッセージを作成
        answer_turn1 = answers_turn1[i]
        messages_turn2 = [
            {"role": "user", "content": question.turns[0]},
            {"role": "assistant", "content": answer_turn1},
            {"role": "user", "content": question.turns[1]}
        ]
        params_turn2 = {"max_tokens": max_tokens, "temperature": temperature}
        inputs_for_processor_turn2.append((messages_turn2, params_turn2))

    print(f"Generating answers for turn 2... (using LLMAsyncProcessor)")
    llm_ap_turn2 = LLMAsyncProcessor(llm=llm, inputs=inputs_for_processor_turn2)
    # get_results を asyncio.to_thread で呼び出す
    results_turn2 = await asyncio.to_thread(llm_ap_turn2.get_results)
    answers_turn2 = [res.content if hasattr(res, 'content') else str(res) for res in results_turn2]

    # 結果をファイルに書き込み
    print(f"Saving model answers to {output_file}...")
    with open(output_file, "w") as f:
        for i, question in enumerate(questions):
            answer = {
                "question_id": question.question_id,
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": [{"turns": [answers_turn1[i], answers_turn2[i]]}],
                "tstamp": time.time(),
            }
            f.write(json.dumps(answer, ensure_ascii=False) + "\n")

    print("Model answer generation finished.")
    return

async def run_judge_single_async(question: Question, answer: Answer, judge: Judge, ref_answer: Optional[Answer], multi_turn: bool = False) -> Tuple[float, str, str]:
    """回答を評価する（非同期）"""
    instance = WandbConfigSingleton.get_instance()
    llm = instance.llm
    
    # ジャッジ用のパラメータ設定
    judge_max_tokens = 2048 # 元のcommon.pyに合わせた値
    judge_temperature = 0.0 # 評価なので0
    
    kwargs = {}
    if ref_answer is not None:
        kwargs["ref_answer_1"] = ref_answer.choices[0]["turns"][0]
        if multi_turn:
            kwargs["ref_answer_2"] = ref_answer.choices[0]["turns"][1]
    
    if multi_turn:
        user_prompt = judge.prompt_template["prompt_template"].format(
            question_1=question.turns[0],
            question_2=question.turns[1],
            answer_1=answer.choices[0]["turns"][0],
            answer_2=answer.choices[0]["turns"][1],
            **kwargs,
        )
    else:
        user_prompt = judge.prompt_template["prompt_template"].format(
            question=question.turns[0],
            answer=answer.choices[0]["turns"][0],
            **kwargs,
        )
    
    system_prompt = judge.prompt_template["system_prompt"]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # 評価実行（非同期）
    response = await asyncio.to_thread(
        llm.invoke, 
        messages,
        max_tokens=judge_max_tokens, # max_tokensを設定
        temperature=judge_temperature # temperatureを設定
    )
    judgment = response.content if hasattr(response, 'content') else str(response) # 文字列に変換
    
    # スコア抽出
    import re
    rating = -1
    if judge.prompt_template["output_format"] in ["[[rating]]", "[[評価]]", "[[평가]]"]:
        one_score_pattern = re.compile(r"\[\[(\d+\.?\d*)\]\]")
        one_score_pattern_backup = re.compile(r"\[(\d+\.?\d*)\]")
        score_match = re.search(one_score_pattern, judgment)
        if not score_match:
            score_match = re.search(one_score_pattern_backup, judgment)
        if score_match and score_match.groups():
            try:
                rating = float(score_match.groups()[0])
            except ValueError:
                print(f"Warning: Could not convert score '{score_match.groups()[0]}' to float for question {question.question_id}")
                rating = -1
        else:
            print(f"[WARN] Judgment output did not match expected pattern: {judgment}")
            rating = -1
    
    return rating, user_prompt, judgment

async def play_a_match_single_async(match: MatchSingle, output_file: str) -> Dict[str, Any]:
    """単一モデルの評価を実行（非同期）"""
    question, model, answer, judge, ref_answer, multi_turn = (
        match.question,
        match.model,
        match.answer,
        match.judge,
        match.ref_answer,
        match.multi_turn,
    )
    
    if judge.prompt_template["type"] == "single":
        score, user_prompt, judgment = await run_judge_single_async(
            question, answer, judge, ref_answer, multi_turn=multi_turn
        )
        
        question_id = question.question_id
        turn = 1 if not multi_turn else 2
        result = {
            "question_id": question_id,
            "model": model,
            "judge": (judge.model_name, judge.prompt_template["name"]),
            "user_prompt": user_prompt,
            "judgment": judgment,
            "score": score,
            "turn": turn,
            "tstamp": datetime.datetime.now().timestamp(),
        }
    else:
        raise ValueError(f"無効なジャッジタイプ: {judge.prompt_template['type']}")
    
    if output_file:
        output_file_path = os.path.join(
            output_file.replace(".jsonl", ""),
            f"{model}__{turn}turn.jsonl"
        )
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    return result

def task_to_sub_category(category):
    """カテゴリをサブカテゴリにマッピング"""
    sub_category_map = {
        "writing": "creation",
        "roleplay": "creation",
        "extraction": "stem",
        "math": "stem",
        "coding": "stem",
        "reasoning": "stem",
        "humanities": "humanities",
        "stem": "stem",
        "arena-hard-200": "arena-hard-200"
    }
    return sub_category_map.get(category, "others")

def check_data(questions: List[Question], model_answers: Dict[str, Dict[int, Answer]], ref_answers: Dict[str, Dict[int, Answer]], models: List[str], judges: Dict[str, Judge]):
    """データが完全かチェック"""
    # モデル回答のチェック
    for m in models:
        assert m in model_answers, f"モデル {m} の回答がありません"
        m_answer = model_answers[m]
        for q in questions:
            assert q.question_id in m_answer, f"質問 {q.question_id} に対するモデル {m} の回答がありません"
    
    # 参照回答のチェック（NEED_REF_CATSのカテゴリのみ）
    for jg in judges.values():
        if not jg.ref_based:
            continue
        for q in questions:
            if q.category in NEED_REF_CATS:  # このカテゴリのみチェック
                ref_model = list(ref_answers.keys())[0]
                assert q.question_id in ref_answers[ref_model], f"質問 {q.question_id} の参照回答がありません（ジャッジ {ref_model}）"

async def async_evaluate():
    """MT-Benchの評価を実行する非同期関数"""
    # WandbConfigSingletonからインスタンスを取得
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    
    # ハッシュを作成してmodel_idに追加（重複を避けるため）
    mnaum_data = str(datetime.datetime.now())
    encoded_data = mnaum_data.encode()
    hash_object = hashlib.sha256(encoded_data)
    hashed_string = hash_object.hexdigest()
    if cfg.mtbench.model_id is None:
        cfg.mtbench.model_id = f'{cfg.model.pretrained_model_name_or_path.replace("/", "--")}_hash_{hashed_string}' 
    
    # ファイルパス
    # 質問
    if cfg.testmode:
        artifact_dir = run.use_artifact(cfg.mtbench.question_artifacts_path_test, type='dataset').download()
    else:
        artifact_dir = run.use_artifact(cfg.mtbench.question_artifacts_path, type='dataset').download()
    question_file = artifact_dir + "/question.jsonl"
    
    # 回答ファイルと回答ディレクトリ
    answer_file = f"data/{cfg.mtbench.bench_name}/model_answer/{cfg.mtbench.model_id}.jsonl"
    answer_dir = f"data/{cfg.mtbench.bench_name}/model_answer"
    
    # 参照回答
    if cfg.testmode:
        ref_answer_dir = run.use_artifact(cfg.mtbench.referenceanswer_artifacts_path_test, type='dataset').download()
    else:
        ref_answer_dir = run.use_artifact(cfg.mtbench.referenceanswer_artifacts_path, type='dataset').download()

    # ***** 質問データの読み込みをここで行う *****
    print("1. 質問データを読み込み中...")
    questions = load_questions(question_file)
    
    # first_nが指定されていれば、最初のn個の質問だけを使用
    if cfg.mtbench.first_n:
        questions = questions[:cfg.mtbench.first_n]
    # *****************************************

    # 1. モデル回答の生成 (修正済み関数を呼び出す)
    print(f"2. モデル回答を生成中... (質問数: {len(questions)}) (using LLMAsyncProcessor)")
    await generate_model_answers(
        questions=questions,
        output_file=answer_file,
        # parallel パラメータは LLMAsyncProcessor が内部で管理するので不要
    )
    
    # 2. 評価
    print("3. モデル回答とリファレンス回答を読み込み中...")
    # モデル回答とリファレンス回答を読み込む
    model_answers = load_model_answers(answer_dir)
    model_answers = {cfg.mtbench.model_id: model_answers[cfg.mtbench.model_id]}
    ref_answers = load_model_answers(ref_answer_dir)
    
    # ジャッジプロンプトを読み込む
    artifact_dir = run.use_artifact(cfg.mtbench.judge_prompt_artifacts_path, type='dataset').download()
    judge_prompts = load_judge_prompts(artifact_dir + "/judge_prompts.jsonl")

    models = [cfg.mtbench.model_id]
 
    # ジャッジを作成
    judges = make_judge_single(cfg.mtbench.judge_model, judge_prompts)
    output_file = f"data/{cfg.mtbench.bench_name}/model_judgment/{cfg.mtbench.judge_model}_single"

    # データチェック
    check_data(questions, model_answers, ref_answers, models, judges)

    # 質問を数学/非数学に分ける
    question_math = [q for q in questions if q.category in NEED_REF_CATS]
    question_default = [q for q in questions if q.category not in NEED_REF_CATS]

    # マッチを作成
    print("4. 評価マッチを作成中...")
    matches = []
    matches += make_match_single(
        question_default, models, model_answers, judges["default"], 
        ref_answers=ref_answers  # 参照回答を渡す
    )
    matches += make_match_single(
        question_math,
        models,
        model_answers,
        judges["math"],
        ref_answers=ref_answers,
    )
    matches += make_match_single(
        question_default,
        models,
        model_answers,
        judges["default-mt"],
        ref_answers=ref_answers,  # 参照回答を渡す
        multi_turn=True,
    )
    matches += make_match_single(
        question_math,
        models,
        model_answers,
        judges["math-mt"],
        ref_answers=ref_answers,
        multi_turn=True,
    )

    judge_count = cfg.mtbench.get('judge_count', 1)
    matches *= judge_count

    # マッチ統計
    match_stat = {
        "bench_name": cfg.mtbench.bench_name,
        "mode": "single",
        "judge": cfg.mtbench.judge_model,
        "baseline": None,
        "model_list": models,
        "total_num_questions": len(questions),
        "total_num_matches": len(matches),
        "output_path": output_file,
    }
    
    print("統計情報:")
    print(json.dumps(match_stat, indent=4))
    
    # ジャッジ用のパラメータ
    judge_max_tokens = 2048
    judge_temperature = 0.0
    judge_model_name = cfg.mtbench.judge_model
    judge_parallel = getattr(cfg.mtbench, 'parallel', 80)
    judge_api_key = os.environ.get("OPENAI_API_KEY")

    client = openai.OpenAI(api_key=judge_api_key)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=1, max=10))
    async def openai_judge_async(messages, max_tokens, temperature, model):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ).choices[0].message.content
        )

    # LLMAsyncProcessor に渡す inputs を作成
    inputs_for_judge_processor = []
    for match in matches:
        question = match.question
        answer = match.answer
        judge = match.judge
        ref_answer = match.ref_answer
        multi_turn = match.multi_turn

        kwargs = {}
        if ref_answer is not None:
            kwargs["ref_answer_1"] = ref_answer.choices[0]["turns"][0]
            if multi_turn:
                kwargs["ref_answer_2"] = ref_answer.choices[0]["turns"][1]

        if multi_turn:
            user_prompt = judge.prompt_template["prompt_template"].format(
                question_1=question.turns[0],
                question_2=question.turns[1],
                answer_1=answer.choices[0]["turns"][0],
                answer_2=answer.choices[0]["turns"][1],
                **kwargs,
            )
        else:
            user_prompt = judge.prompt_template["prompt_template"].format(
                question=question.turns[0],
                answer=answer.choices[0]["turns"][0],
                **kwargs,
            )

        system_prompt = judge.prompt_template["system_prompt"]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        inputs_for_judge_processor.append((messages, judge_max_tokens, judge_temperature, judge_model_name, match, user_prompt))

    print(f"Processing judge results (OpenAI API parallel, batch={judge_parallel})...")
    results = []
    sem = asyncio.Semaphore(judge_parallel)

    async def judge_task(args):
        messages, max_tokens, temperature, model, match, user_prompt = args
        async with sem:
            try:
                judgment = await openai_judge_async(messages, max_tokens, temperature, model)
            except Exception as e:
                print(f"OpenAI judge error: {e}")
                judgment = "$ERROR$"
            # スコア抽出
            import re
            rating = -1
            if match.judge.prompt_template["output_format"] in ["[[rating]]", "[[評価]]", "[[평가]]"]:
                one_score_pattern = re.compile(r"\[\[(\d+\.?\d*)\]\]")
                one_score_pattern_backup = re.compile(r"\[(\d+\.?\d*)\]")
                score_match = re.search(one_score_pattern, judgment)
                if not score_match:
                    score_match = re.search(one_score_pattern_backup, judgment)
                if score_match and score_match.groups():
                    try:
                        rating = float(score_match.groups()[0])
                    except ValueError:
                        print(f"Warning: Could not convert score '{score_match.groups()[0]}' to float for question {match.question.question_id}")
                        rating = -1
                else:
                    print(f"[WARN] Judgment output did not match expected pattern: {judgment}")
                    rating = -1
            question_id = match.question.question_id
            turn = 1 if not match.multi_turn else 2
            result = {
                "question_id": question_id,
                "model": match.model,
                "judge": (match.judge.model_name, match.judge.prompt_template["name"]),
                "user_prompt": user_prompt,
                "judgment": judgment,
                "score": rating,
                "turn": turn,
                "tstamp": datetime.datetime.now().timestamp(),
            }
            # ファイル出力
            if output_file:
                output_file_path = os.path.join(
                    output_file.replace(".jsonl", ""),
                    f"{match.model}__{turn}turn.jsonl"
                )
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                with open(output_file_path, "a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            return result

    # 並列バッチで実行
    tasks = [judge_task(args) for args in inputs_for_judge_processor]
    for i in range(0, len(tasks), judge_parallel):
        batch = tasks[i:i+judge_parallel]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)

    print("6. 結果を集計中...")
    # 3. 結果を集計してwandb.Tableとしてログに記録
    # 質問を読み込む
    df_question = pd.DataFrame([{
        'question_id': q.question_id,
        'category': q.category,
        'turns': q.turns
    } for q in questions])
    
    # df_questionをjudge_count回繰り返す
    df_question_repeat = df_question.loc[df_question.index.repeat(judge_count)].reset_index(drop=True)
    
    # 回答を読み込む
    df_answer = pd.read_json(answer_file, lines=True)
    df_answer = df_answer[(df_answer.model_id == cfg.mtbench.model_id)|(df_answer.model_id == cfg.model.pretrained_model_name_or_path)]
    df_answer = df_answer.sort_values(['question_id'])
    df_answer_repeat = df_answer.loc[df_answer.index.repeat(judge_count)].reset_index(drop=True)

    # ジャッジ結果を読み込む
    output_file_turn1 = output_file + "/" + cfg.mtbench.model_id + "__1turn.jsonl"
    output_file_turn2 = output_file + "/" + cfg.mtbench.model_id + "__2turn.jsonl"
    df_judge1 = pd.read_json(output_file_turn1, lines=True)
    df_judge2 = pd.read_json(output_file_turn2, lines=True)
    df_judge = pd.concat([df_judge1, df_judge2], ignore_index=True)

    df_judge = df_judge[df_judge.model == cfg.mtbench.model_id]
    df_judge.model = df_judge.model.str.replace("--", "/")
    df_judge['hash'] = df_judge.model.apply(lambda x: x.split('_hash_')[-1])
    df_judge['model'] = df_judge.model.apply(lambda x: x.split('_hash_')[0])
    df_judge = df_judge.sort_values(['question_id', 'turn'])

    # テーブルをマージ
    df_judge["question"] = pd.Series(np.nan, dtype='object')
    df_judge.loc[df_judge.turn == 1, 'question'] = df_question_repeat.turns.apply(lambda x: x[0]).values
    df_judge.loc[df_judge.turn == 2, 'question'] = df_question_repeat.turns.apply(lambda x: x[1]).values

    df_judge['answer'] = pd.Series(np.nan, dtype='object')
    df_judge.loc[df_judge.turn == 1, 'answer'] = df_answer_repeat.choices.apply(lambda x: x[0]['turns'][0]).values
    df_judge.loc[df_judge.turn == 2, 'answer'] = df_answer_repeat.choices.apply(lambda x: x[0]['turns'][1]).values
    
    # 元のコードと同じように単純にマージ
    df_judge = df_judge.merge(df_answer[['question_id', 'answer_id']], on='question_id', how='left')
    df_judge = df_judge.merge(df_question[['question_id', 'category']], on='question_id', how='left')

    # 元のコードと同じ固定の列リスト
    use_col = [
        'model', 'question_id', 'category', 'question', 
        'answer', 'judge', 'user_prompt', 'judgment', 
        'score', 'turn', 'tstamp'
    ]
    df_judge = df_judge[use_col]
    df_judge.rename(columns={'model': 'model_name'}, inplace=True)
    df_judge['sub_category'] = df_judge['category'].map(task_to_sub_category)

    # テーブルをログに記録
    print("7. 結果をWandBにログ記録中...")
    table_log = wandb.Table(dataframe=df_judge)
    
    # レーダーチャート用のテーブル
    _df_judge = df_judge.query('score != -1').groupby(['question_id', 'turn', 'category'], as_index=False).score.mean()
    df_summary = _df_judge.groupby(['category'], as_index=False).score.mean()
    table_radar = wandb.Table(dataframe=df_summary)

    # リーダーボード用のテーブル
    mtbench_df = pd.DataFrame([df_summary.score.values.tolist()], columns=df_summary.category.values.tolist())
    mtbench_df.insert(0, "AVG_mtbench", mtbench_df.mean(axis=1, numeric_only=True))
    mtbench_df.insert(0, "model_name", cfg.model.pretrained_model_name_or_path)
    table_metric = wandb.Table(dataframe=mtbench_df)

    run.log({
        "mtbench_output_table": table_log,
        "mtbench_leaderboard_table": table_metric,
        "mtbench_radar_table": table_radar,
    })
    
    print("MT-Bench評価が正常に完了しました！")
    return

def evaluate():
    """非同期評価関数を実行するエントリーポイント"""
    asyncio.run(async_evaluate()) 