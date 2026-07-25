import datetime
import json
import os
import asyncio
import re
import numpy as np
import pandas as pd
import wandb
import weave
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal, Awaitable
import shortuuid
import time
from pydantic import BaseModel
from tqdm.asyncio import tqdm as atqdm

from config_singleton import WandbConfigSingleton
from .evaluate_utils import LLMAsyncProcessor, get_openai_judge_client

# カテゴリが参照回答を必要とするかどうかのリスト
NEED_REF_CATS = ["math", "reasoning", "coding", "arena-hard-200"]


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
    tstamp: Optional[float] = None
    answer_id: str = field(default_factory=shortuuid.uuid)
    
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

class JudgeSingle(BaseModel):
    explanation: str
    rating: Literal["[[0]]", "[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]", "[[6]]", "[[7]]", "[[8]]", "[[9]]", "[[10]]"]


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "item"


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _answer_payload(answer: Answer) -> Dict[str, Any]:
    return {
        "question_id": answer.question_id,
        "model_id": answer.model_id,
        "choices": answer.choices,
        "tstamp": answer.tstamp,
        "answer_id": answer.answer_id,
    }


def _load_answer_checkpoint(
    path: Path,
    *,
    question_id: int,
    model_id: str,
) -> Optional[Answer]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        payload.get("question_id") != question_id
        or payload.get("model_id") != model_id
        or not isinstance(payload.get("choices"), list)
    ):
        return None
    return Answer(
        question_id=question_id,
        model_id=model_id,
        choices=payload["choices"],
        tstamp=payload.get("tstamp"),
        answer_id=str(payload.get("answer_id") or shortuuid.uuid()),
    )


def _load_judge_checkpoint(
    path: Path,
    *,
    question_id: int,
    turn: int,
    judge_index: int,
) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        payload.get("question_id") != question_id
        or payload.get("turn") != turn
        or payload.get("judge_index") != judge_index
        or not isinstance(payload.get("score"), int)
    ):
        return None
    return payload

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

async def generate_model_answer(
    question: Question,
    llm_ap: LLMAsyncProcessor,
    answer: Answer,
    default_generator_config: Dict[str, Any],
    temperature_overrides: Dict[str, float],
    turn1_task: Optional[Awaitable] = None,
    checkpoint_path: Optional[Path] = None,
) -> Answer:
    """モデルの回答を生成する（非同期）"""
    default_temperature = default_generator_config.get("temperature", 0.0)
    generator_config = {
        **default_generator_config,
        "temperature": temperature_overrides.get(question.category, default_temperature),
    }

    messages = [{"role": "user", "content": question.turns[0]}]
    # turn1_taskがあれば、turn1の完了を待ったあとにturn2の質問を生成
    if turn1_task is not None:
        await asyncio.wait([turn1_task])
        messages += [
            {"role": "assistant", "content": answer.choices[0]["turns"][0]},
            {"role": "user", "content": question.turns[1]}
        ]

    response = await llm_ap.process_single_async(messages, **generator_config)
    content = response.content if hasattr(response, 'content') else str(response)
    answer.choices[0]["turns"].append(content)
    answer.tstamp = time.time()
    if checkpoint_path is not None:
        _atomic_write_json(checkpoint_path, _answer_payload(answer))
    return answer

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
    llm = instance.llm
    cfg = instance.config
    
    if cfg.mtbench.model_id is None:
        stable_run_id = str(getattr(run, "id", "") or "local")
        cfg.mtbench.model_id = (
            f'{cfg.model.pretrained_model_name_or_path.replace("/", "--")}'
            f'__run_{_safe_component(stable_run_id)}'
        )
    configured_checkpoint_dir = cfg.mtbench.get("checkpoint_dir", None)
    checkpoint_root = Path(
        configured_checkpoint_dir
        or f"outputs/mtbench_checkpoints/{_safe_component(str(getattr(run, 'id', 'local')))}"
    )
    answer_checkpoint_dir = checkpoint_root / "answers"
    judge_checkpoint_dir = checkpoint_root / "judgments"
    answer_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    judge_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # ファイルパス
    # 質問
    if cfg.testmode:
        artifact_dir = run.use_artifact(cfg.mtbench.question_artifacts_path_test, type='dataset').download()
    else:
        artifact_dir = run.use_artifact(cfg.mtbench.question_artifacts_path, type='dataset').download()
    question_file = artifact_dir + "/question.jsonl"
    
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
    print(f"2. モデル回答タスクを生成中... (質問数: {len(questions)})")
    # mtbenchのgenerator_configを取得（max_new_tokenとgenerator_config.max_tokensの両方に対応）
    default_generator_config = cfg.mtbench.get("generator_config", {})
    
    # max_new_tokenが設定されていて、generator_configにmax_tokensが設定されていない場合はmax_new_tokenを使用
    if hasattr(cfg.mtbench, 'max_new_token') and cfg.mtbench.max_new_token and 'max_tokens' not in default_generator_config:
        default_generator_config['max_tokens'] = cfg.mtbench.max_new_token
    # generator_configにmax_tokensが設定されていない場合のフォールバック処理
    elif 'max_tokens' not in default_generator_config:
        # generator.max_tokensをフォールバックとして使用
        if hasattr(cfg, 'generator') and hasattr(cfg.generator, 'max_tokens'):
            default_generator_config['max_tokens'] = cfg.generator.max_tokens
        else:
            default_generator_config['max_tokens'] = 1024  # デフォルト値
    # mtbenchのtemperatureオーバーライド設定を取得
    temperature_overrides = cfg.mtbench.get("temperature_override", {}) 

    llm_ap = LLMAsyncProcessor(llm=llm, inputs=[])
    model_answers = {cfg.mtbench.model_id: {}}
    for question in questions:
        checkpoint_path = answer_checkpoint_dir / f"{question.question_id}.json"
        answer = _load_answer_checkpoint(
            checkpoint_path,
            question_id=question.question_id,
            model_id=cfg.mtbench.model_id,
        )
        model_answers[cfg.mtbench.model_id][question.question_id] = answer or Answer(
            question_id=question.question_id,
            model_id=cfg.mtbench.model_id,
            choices=[{"turns": []}],
        )

    async def completed_answer(answer: Answer) -> Answer:
        return answer

    turn1_tasks = {}
    turn2_tasks = {}
    for question in questions:
        answer = model_answers[cfg.mtbench.model_id][question.question_id]
        checkpoint_path = answer_checkpoint_dir / f"{question.question_id}.json"
        turns = answer.choices[0].get("turns", [])
        turn1_tasks[question.question_id] = asyncio.create_task(
            completed_answer(answer)
            if len(turns) >= 1
            else generate_model_answer(
                question,
                llm_ap,
                answer,
                default_generator_config,
                temperature_overrides,
                checkpoint_path=checkpoint_path,
            )
        )
        turn2_tasks[question.question_id] = asyncio.create_task(
            completed_answer(answer)
            if len(turns) >= 2
            else generate_model_answer(
                question,
                llm_ap,
                answer,
                default_generator_config,
                temperature_overrides,
                turn1_task=turn1_tasks[question.question_id],
                checkpoint_path=checkpoint_path,
            )
        )

    # 回答生成タスクをProgressbar付きで並列実行
    try:
        await atqdm.gather(
            *turn1_tasks.values(),
            *turn2_tasks.values(),
            desc="Generating MT-Bench answers",
        )
    finally:
        await llm_ap.close_async_client()

    # 2. 評価
    print("3. リファレンス回答を読み込み中...")
    # リファレンス回答を読み込む
    ref_answers = load_model_answers(ref_answer_dir)
    
    # ジャッジプロンプトを読み込む
    artifact_dir = run.use_artifact(cfg.mtbench.judge_prompt_artifacts_path, type='dataset').download()
    judge_prompts = load_judge_prompts(artifact_dir + "/judge_prompts.jsonl")

    models = [cfg.mtbench.model_id]

    # ジャッジを作成
    judges = make_judge_single(cfg.mtbench.judge.model, judge_prompts)
    output_file = str(
        checkpoint_root
        / "model_judgment"
        / f"{_safe_component(cfg.mtbench.judge.model)}_single"
    )

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
    indexed_matches = [
        (match, judge_index)
        for judge_index in range(judge_count)
        for match in matches
    ]

    # マッチ統計
    match_stat = {
        "bench_name": cfg.mtbench.bench_name,
        "mode": "single",
        "judge": cfg.mtbench.judge.model,
        "baseline": None,
        "model_list": models,
        "total_num_questions": len(questions),
        "total_num_matches": len(indexed_matches),
        "output_path": output_file,
    }
    
    print("統計情報:")
    print(json.dumps(match_stat, indent=4))
    
    print("5. 回答生成とジャッジを実行中...(using LLMAsyncProcessor)")
    # ジャッジ用のパラメータ
    judge_model_name = cfg.mtbench.judge.model
    judge_parallel = cfg.mtbench.judge.parallel
    judge_params = cfg.mtbench.judge.params
    client = get_openai_judge_client(judge_model_name, text_format=JudgeSingle)
    # YAML切替: cfg.mtbench.error_handling.request_failure.mode / cfg.mtbench.soft_fail_on_error（Judge側）
    soft_fail_judge = False
    try:
        mode = cfg.mtbench.get("error_handling", {}).get("request_failure", {}).get("mode", None)
        if mode is not None:
            soft_fail_judge = str(mode).lower() == "soft"
    except Exception:
        pass
    try:
        sfoe = cfg.mtbench.get("soft_fail_on_error", None)
        if sfoe is not None:
            soft_fail_judge = bool(sfoe)
    except Exception:
        pass
    judge_llm_ap = LLMAsyncProcessor(llm=client, batch_size=judge_parallel, inference_interval=0., soft_fail_on_error=soft_fail_judge)

    # LLMAsyncProcessor に渡す inputs を作成
    async def wait_answer_and_judge(
        match,
        judge_index,
        judge_llm_ap,
        generate_answer_task,
    ):
        question = match.question
        answer = await generate_answer_task
        judge = match.judge
        ref_answer = match.ref_answer
        multi_turn = match.multi_turn
        turn = 2 if multi_turn else 1
        checkpoint_path = (
            judge_checkpoint_dir
            / f"q{question.question_id}-turn{turn}-judge{judge_index}.json"
        )
        cached = _load_judge_checkpoint(
            checkpoint_path,
            question_id=question.question_id,
            turn=turn,
            judge_index=judge_index,
        )
        if cached is not None:
            return cached

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
        judge_result = await judge_llm_ap.process_single_async(messages, **judge_params)
        if (
            not getattr(judge_result, "parsed_output", None)
            or not getattr(judge_result.parsed_output, "rating", None)
        ):
            raise RuntimeError(
                f"MT-Bench judge returned no score for question {question.question_id}, "
                f"turn {turn}, judge_index {judge_index}"
            )
        result = {
            "question_id": question.question_id,
            "model": match.model,
            "judge_model": match.judge.model_name,
            "judge_prompt_template": match.judge.prompt_template["name"],
            "judge_prompt": user_prompt,
            "judgment": judge_result.parsed_output.explanation,
            "score": int(
                judge_result.parsed_output.rating.replace("[[", "").replace("]]", "")
            ),
            "turn": turn,
            "judge_index": judge_index,
            "tstamp": datetime.datetime.now().timestamp(),
        }
        _atomic_write_json(checkpoint_path, result)
        return result

    judge_tasks = [
        wait_answer_and_judge(
            match,
            judge_index,
            judge_llm_ap,
            turn2_tasks[match.question.question_id]
            if match.multi_turn
            else turn1_tasks[match.question.question_id],
        )
        for match, judge_index in indexed_matches
    ]

    # JudgeをProgressbar付きで並列実行
    print(f"Processing judge results (OpenAI API parallel, batch={judge_parallel})...")
    try:
        results = await atqdm.gather(*judge_tasks, desc="Judging MT-Bench")
    finally:
        await judge_llm_ap.close_async_client()
    expected_results = len(questions) * 2 * judge_count
    if len(results) != expected_results:
        raise RuntimeError(
            f"MT-Bench checkpoint set is incomplete: {len(results)}/{expected_results}"
        )
    for turn in (1, 2):
        output_file_path = os.path.join(
            output_file,
            f"{cfg.mtbench.model_id}__{turn}turn.jsonl",
        )
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, "w", encoding="utf-8") as handle:
            for result in sorted(
                (row for row in results if row["turn"] == turn),
                key=lambda row: (row["question_id"], row["judge_index"]),
            ):
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")

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
    df_answer = pd.DataFrame(list(model_answers[cfg.mtbench.model_id].values()))
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
    df_judge["model"] = cfg.model.pretrained_model_name_or_path
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
        'answer', 'judge_model', 'judge_prompt_template', 'judge_prompt', 'judgment', 
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
    
    # スコアを0-1スケールに正規化
    df_summary['score'] = df_summary['score'] / 10.0
    
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

@weave.op(call_display_name=lambda _: "[MT-Bench] " + WandbConfigSingleton.get_instance().config.wandb.run_name)
def evaluate():
    """非同期評価関数を実行するエントリーポイント"""
    asyncio.run(async_evaluate())
