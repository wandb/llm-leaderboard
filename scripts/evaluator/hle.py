import json
import asyncio
import copy
import math
import numpy as np
from pathlib import Path
from typing import Literal, Dict, Any, List
from pydantic import BaseModel
from tqdm.asyncio import tqdm as atqdm
import pandas as pd

import wandb
import weave

from config_singleton import WandbConfigSingleton
from .evaluate_utils import LLMAsyncProcessor, get_openai_judge_client


SYSTEM_PROMPT = """回答は以下の形式で行ってください：
説明: {選択した答えに対する説明}
答え: {選択した答え}
信頼度: {答えに対する0%から100%の信頼度スコア}"""

JUDGE_PROMPT = """以下の[質問]に対する[回答]が、明確で曖昧さのない[正解]に基づいて正しいかどうかを判定してください。

[質問]: {question}

[回答]: {response}

判定は以下の形式と基準に従って行ってください：

extracted_final_answer: [回答]から抽出した最終的な正確な答え。回答から正確で最終的な答えを抽出できない場合は'None'と記載してください。

[正解]: {correct_answer}

reasoning: [正解]に基づいて、extracted_final_answerが正しいか間違っているかを説明してください。[正解]とextracted_final_answerの間に意味のある違いがあるかどうかのみに焦点を当ててください。問題の背景についてコメントしたり、問題を解こうとしたり、[正解]と異なる答えを主張したりせず、答えが一致するかどうかのみに焦点を当ててください。

correct: extracted_final_answerが上記の[正解]と一致する場合、または数値問題で小さな誤差の範囲内にある場合は'yes'と答えてください。そうでない場合、つまり不一致、曖昧さ、非等価性がある場合、または抽出された答えが間違っている場合は'no'と答えてください。

confidence: [回答]から抽出した0%から100%の間の信頼度スコア。信頼度スコアが利用できない場合は100を記載してください。"""

class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True] = True  # 100% reliability

def load_questions(file_path: Path) -> List[Dict[str, Any]]:
    """Load questions from a JSONL file."""
    questions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    
    return questions

def format_message_for_llm_processor(question: Dict[str, Any], model_name: str) -> List[Dict[str, str]]:
    """Format message for LLMAsyncProcessor (simplified format)"""
    question_text = question['question']
    
    # 画像存在チェック
    try:
        image_value = question.get('image', '')
        # 文字列に変換してから判定
        image_str = str(image_value) if image_value is not None else ''
        has_image = bool(image_str.strip() and image_str.lower() not in ['nan', 'none', ''])
    except:
        has_image = False
    
    if has_image:
        combined_content = f"{SYSTEM_PROMPT}\n\n{question_text}"
        print(f"Warning: Question {question.get('id', 'unknown')} contains image but LLMAsyncProcessor may not support it")
    else:
        combined_content = f"{SYSTEM_PROMPT}\n\n{question_text}"
    
    return [{"role": "user", "content": combined_content}]

def calib_err(confidence: np.ndarray, correct: np.ndarray, p: str = '2', beta: int = 100) -> float:
    """Calculate calibration error"""
    # source: https://github.com/hendrycks/outlier-exposure/blob/master/utils/calibration_tools.py
    if len(confidence) == 0:
        return 0.0
    
    # データサイズに応じてbetaを自動調整
    n = len(confidence)
    if n < 200:
        beta = max(10, n // 10)  # 最低10個のビンを作る
    elif n < 1000:
        beta = 50
    else:
        beta = 100
    
    print(f"Data size: {n}, Using beta: {beta}, Expected bins: {n // beta}")
    
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    if bins:  # ビンが存在する場合のみ調整
        bins[-1] = [bins[-1][0], len(confidence)]
    else:
        # 全データを1つのビンとして扱う
        bins = [[0, len(confidence)]]

    cerr = 0
    total_examples = len(confidence)
    
    for i in range(len(bins)):
        bin_confidence = confidence[bins[i][0]:bins[i][1]]
        bin_correct = correct[bins[i][0]:bins[i][1]]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))

            if p == '2':
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == '1':
                cerr += num_examples_in_bin / total_examples * difference
            elif p == 'infty' or p == 'infinity' or p == 'max':
                cerr = np.maximum(cerr, difference)

    if p == '2':
        cerr = np.sqrt(cerr)

    return cerr

def calculate_metrics(predictions: Dict[str, Dict[str, Any]], total_questions: int) -> Dict[str, float]:
    """Calculate accuracy and calibration metrics"""
    correct = []
    confidence = []
    
    for k, v in predictions.items():
        if "judge_response" in v:
            judge_response = v["judge_response"]
            correct.append("yes" in judge_response["correct"])
            confidence.append(judge_response["confidence"])
        else:
            print(f"Missing judge response for {k}")

    correct = np.array(correct)
    confidence = np.array(confidence) / 100

    if len(correct) != total_questions:
        print(f"Available predictions: {len(correct)} | Total questions: {total_questions}")

    accuracy = round(100 * sum(correct) / total_questions, 2) if total_questions > 0 else 0.0
    # Wald estimator, 95% confidence interval
    confidence_half_width = round(1.96 * math.sqrt(accuracy * (100 - accuracy) / total_questions), 2) if total_questions > 0 else 0.0
    calibration_error = 100 * round(calib_err(confidence, correct, p='2', beta=100), 2)

    return {
        "accuracy": accuracy / 100.0, # 0-1スケールに正規化
        "accuracy_percent": accuracy, # 元のパーセント表記
        "confidence_half_width": confidence_half_width,
        "calibration_error": calibration_error,
        "total_questions": total_questions,
        "answered_questions": len(correct)
    }

async def evaluate_async():
    """Main evaluation function for HLE benchmark"""
    # Get configuration
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    llm = instance.llm
    
    # Configuration parameters
    generator_config = cfg.hle.get("generator_config", {})
    judge_model = cfg.hle.judge.get("model", "o3-mini-2025-01-31")
    judge_parallel = cfg.hle.judge.get("parallel", 32)
    judge_params = cfg.hle.judge.get("params", {})
    max_samples = cfg.hle.get("max_samples", None)

    try:
        artifact_path = cfg.hle.artifact_path
        artifact = run.use_artifact(artifact_path, type="dataset")
        artifact_dir = artifact.download()
    except Exception as e:
        print(f"Error downloading artifact: {e}")
        return

    for subset in ("test", "dev"):
        if cfg.testmode and subset == "test":
            print("Skipping test set in test mode.")
            continue

        data_file = Path(artifact_dir) / "hle-ja" / f"{subset}.jsonl"
        if not data_file.exists():
            print(f"Dataset file not found for subset '{subset}': {data_file}")
            continue
            
        print(f"--- Evaluating subset: {subset} --- ")
        
        try:
            questions = load_questions(data_file)
            
            # Handle sample size
            if cfg.testmode:
                # In test mode, limit samples for both dev and test (though test is usually skipped)
                num_samples = max_samples if max_samples is not None else 5
                questions = questions[:num_samples]
            else:
                # In normal mode, only limit samples for the 'dev' set
                if subset == "dev":
                    if max_samples is not None:
                        questions = questions[:max_samples]
                # For the 'test' set, we do nothing, so all samples are used.
                
            print(f"Loaded {len(questions)} questions from {data_file.name}")
            
            multimodal_count = sum(1 for q in questions if q.get('image'))
            if multimodal_count > 0:
                print(f"Warning: {multimodal_count} questions contain images. Ensure your LLM supports multimodal input.")
            
        except Exception as e:
            print(f"Error loading dataset for subset '{subset}': {e}")
            continue
        
        print("Generating model responses...")
        predictions = {}
        judged_predictions = {}
        llm_ap = LLMAsyncProcessor(llm=llm)
        judge_llm = get_openai_judge_client(judge_model, text_format=ExtractedAnswer)
        judge_llm_ap = LLMAsyncProcessor(llm=judge_llm, batch_size=judge_parallel, inference_interval=0.)

        # Inference and judge in parallel
        # Generate model responses
        async def generate_answer(q):
            messages = format_message_for_llm_processor(q, cfg.model.pretrained_model_name_or_path)
            try:
                result = await llm_ap.process_single_async(messages, **generator_config)
                predictions[q["id"]] = {
                    "model": cfg.model.pretrained_model_name_or_path,
                    "response": result.content,
                    "usage": {
                        "completion_tokens": len(result.content.split()),
                        "prompt_tokens": len(q["question"].split())
                    }
                }
            except Exception as e:
                print(f"Error generating model responses for subset '{subset}': {e}")
            return q

        generate_answer_tasks = [asyncio.create_task(generate_answer(q)) for q in questions]
        generate_answer_results = asyncio.create_task(
            atqdm.gather(*generate_answer_tasks, desc="Generating HLE answers")
        )

        # OpenAIの場合、推論とJudgeが同じAPIになるため、Rate Limit対策として推論がすべて終わるのを待つ
        if cfg.api == 'openai':
            await generate_answer_results
            print(f"Generated {len(predictions)} model responses for {subset}")

        # Judge model responses
        async def judge(q, generate_answer_task):
            await generate_answer_task
            if q["id"] not in predictions:
                return

            if "judge_response" in predictions[q["id"]]:
                judged_predictions[q["id"]] = predictions[q["id"]]
                return

            prompt = JUDGE_PROMPT.format(
                question=q["question"],
                correct_answer=q["answer"],
                response=predictions[q["id"]]["response"],
            )
            messages = [{"role": "user", "content": prompt}]

            try:
                result = await judge_llm_ap.process_single_async(messages, **judge_params)
                prediction = copy.deepcopy(predictions[q["id"]])
                prediction["judge_response"] = {
                    "correct_answer": q["answer"],
                    "model_answer": result.parsed_output.extracted_final_answer,
                    "reasoning": result.parsed_output.reasoning,
                    "correct": result.parsed_output.correct,
                    "confidence": result.parsed_output.confidence,
                }
                judged_predictions[q["id"]] = prediction
            except Exception as e:
                print(f"Error judging model responses for subset '{subset}': {e}")

        judge_tasks = [judge(q, task) for q, task in zip(questions, generate_answer_tasks)]
        await atqdm.gather(*judge_tasks, desc="Judging HLE")
        print(f"Judged {len(judged_predictions)} responses for {subset}")
        
        metrics = calculate_metrics(judged_predictions, len(questions))
        
        print(f"*** HLE Evaluation Results for {subset} ***")
        print(f"Accuracy: {metrics['accuracy_percent']}% +/- {metrics['confidence_half_width']}% | n = {metrics['total_questions']}")
        print(f"Calibration Error: {metrics['calibration_error']}")
        
        results_data = []
        for q in questions:
            if q["id"] in judged_predictions:
                pred = judged_predictions[q["id"]]
                judge_resp = pred.get("judge_response", {})
                results_data.append({
                    "id": q["id"],
                    "question": q["question"],
                    "correct_answer": q["answer"],
                    "model_response": pred["response"],
                    "model_answer": judge_resp.get("model_answer", ""),
                    "correct": judge_resp.get("correct", ""),
                    "confidence": judge_resp.get("confidence", 0),
                    "reasoning": judge_resp.get("reasoning", ""),
                })
        
        output_df = pd.DataFrame(results_data)
        output_df.insert(0, 'model_name', cfg.model.pretrained_model_name_or_path)
        
        leaderboard_data = {
            "model_name": cfg.model.pretrained_model_name_or_path,
            "accuracy": metrics["accuracy"],
            "accuracy_percent": metrics["accuracy_percent"],
            "calibration_error": metrics["calibration_error"],
            "confidence_half_width": metrics["confidence_half_width"],
            "total_questions": metrics["total_questions"],
            "answered_questions": metrics["answered_questions"]
        }
        leaderboard_df = pd.DataFrame([leaderboard_data])
        
        run.log({
            f"hle_{subset}_output_table": wandb.Table(dataframe=output_df),
            f"hle_{subset}_leaderboard_table": wandb.Table(dataframe=leaderboard_df),
            f"hle_{subset}_accuracy": metrics["accuracy_percent"],
            f"hle_{subset}_calibration_error": metrics["calibration_error"]
        })
        
        print(f"HLE evaluation for subset '{subset}' completed and logged to wandb")

@weave.op(call_display_name=lambda _: "[HLE] " + WandbConfigSingleton.get_instance().config.wandb.run_name)
def evaluate():
    asyncio.run(evaluate_async())
