import json
import asyncio
import copy
import math
import numpy as np
from pathlib import Path
from typing import Literal, Dict, Any, List, Optional
from pydantic import BaseModel
from tqdm import tqdm
import pandas as pd

import wandb
from openai import AsyncOpenAI, OpenAI, AzureOpenAI
import os

from config_singleton import WandbConfigSingleton
from .evaluate_utils import LLMAsyncProcessor

# OpenAI client setup
def get_openai_client(async_client=True):
    """Get OpenAI client based on environment configuration"""
    api_type = os.environ.get('OPENAI_API_TYPE', 'openai')
    
    if api_type == "azure":
        if async_client:
            return AsyncOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version="2023-07-01-preview",
                timeout=600.0,
                max_retries=1
            )
        else:
            return AzureOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version="2023-07-01-preview"
            )
    else:
        if async_client:
            return AsyncOpenAI(timeout=600.0, max_retries=1)
        else:
            return OpenAI()

# Judge prompt and response format
JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available."""

SYSTEM_PROMPT = "Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"

class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True] = True  # 100% reliability

def load_questions(artifact_dir: str) -> List[Dict[str, Any]]:
    """Load questions from JSONL file"""
    questions = []
    hle_file_path = Path(artifact_dir) / "hle.jsonl"
    
    if not hle_file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {hle_file_path}")
    
    with open(hle_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    
    return questions

def format_message(question: Dict[str, Any], model_name: str) -> List[Dict[str, Any]]:
    """Format question into message format"""
    question_text = question['question']
    
    text_content = {"type": "text", "text": question_text}
    if question.get('image'):  # "" if not multi-modal
        image_content = {"type": "image_url", "image_url": {"url": question['image']}}
        content = [text_content, image_content]
    else:
        content = [text_content]
    
    system_role = "user" if "o1" in model_name else "system"  # o1 no sys prompt
    messages = [
        {"role": system_role, "content": SYSTEM_PROMPT}, 
        {"role": "user", "content": content}
    ]
    return messages

async def extract_answer(question: str, correct_answer: str, response: str, judge_model: str) -> Optional[Dict[str, Any]]:
    """Extract and judge answer using OpenAI API"""
    client = get_openai_client(async_client=True)
    prompt = JUDGE_PROMPT.format(question=question, correct_answer=correct_answer, response=response)
    
    # Handle Azure model naming
    if judge_model.startswith("azure-"):
        judge_model = judge_model[6:]
    
    try:
        response_obj = await client.beta.chat.completions.parse(
            model=judge_model,
            max_completion_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            response_format=ExtractedAnswer,
        )
        content = response_obj.choices[0].message.parsed
        return {
            "correct_answer": correct_answer,
            "model_answer": content.extracted_final_answer,
            "reasoning": content.reasoning,
            "correct": content.correct,
            "confidence": content.confidence
        }
    except Exception as e:
        print(f"Error in judge evaluation: {e}")
        return None

async def judge_response(question: Dict[str, Any], prediction: Dict[str, Any], judge_model: str) -> Optional[Dict[str, Any]]:
    """Judge a single response"""
    if "judge_response" in prediction:  # already judged
        return question["id"], prediction
    
    question_text = question["question"]
    correct_answer = question["answer"]
    response = prediction["response"]
    
    judge_result = await extract_answer(question_text, correct_answer, response, judge_model)
    
    if judge_result is not None:
        prediction_copy = copy.deepcopy(prediction)
        prediction_copy["judge_response"] = judge_result
        return question["id"], prediction_copy
    else:
        return None, None

def calib_err(confidence: np.ndarray, correct: np.ndarray, p: str = '2', beta: int = 100) -> float:
    """Calculate calibration error"""
    # source: https://github.com/hendrycks/outlier-exposure/blob/master/utils/calibration_tools.py
    if len(confidence) == 0:
        return 0.0
        
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    
    if len(confidence) < beta:
        beta = max(1, len(confidence) // 2)
    
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    if bins:
        bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)
    for i in range(len(bins) - 1):
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
            else:
                assert False, "p must be '1', '2', or 'infty'"

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
        "accuracy": accuracy,
        "confidence_half_width": confidence_half_width,
        "calibration_error": calibration_error,
        "total_questions": total_questions,
        "answered_questions": len(correct)
    }

async def evaluate_batch(questions: List[Dict[str, Any]], predictions: Dict[str, Dict[str, Any]], 
                        judge_model: str, max_workers: int = 10) -> Dict[str, Dict[str, Any]]:
    """Evaluate a batch of questions with judge"""
    semaphore = asyncio.Semaphore(max_workers)
    
    async def bound_judge(question):
        async with semaphore:
            return await judge_response(question, predictions[question["id"]], judge_model)
    
    # Only judge questions that have predictions but no judge response
    questions_to_judge = [q for q in questions if q["id"] in predictions]
    
    tasks = [bound_judge(q) for q in questions_to_judge]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    judged_predictions = {}
    for result in results:
        if isinstance(result, Exception):
            print(f"Error in batch evaluation: {result}")
            continue
        unique_id, prediction = result
        if unique_id is not None:
            judged_predictions[unique_id] = prediction
    
    return judged_predictions

def evaluate():
    """Main evaluation function for HLE benchmark"""
    # Get configuration
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    llm = instance.llm
    
    # Configuration parameters
    max_completion_tokens = cfg.hle.get("max_completion_tokens", 8192)
    max_workers = cfg.hle.get("max_workers", 10)
    judge_model = cfg.hle.get("judge_model", "o3-mini-2025-01-31")
    max_samples = cfg.hle.get("max_samples", None)
    
    if cfg.testmode:
        max_samples = 10
    
    print(f"Starting HLE evaluation with max_completion_tokens={max_completion_tokens}, max_workers={max_workers}")
    
    # Load dataset from wandb artifact
    try:
        artifact_path = cfg.hle.artifact_path
        artifact = run.use_artifact(artifact_path, type="dataset")
        artifact_dir = artifact.download()
        
        questions = load_questions(artifact_dir)
        
        if max_samples:
            questions = questions[:max_samples]
            
        print(f"Loaded {len(questions)} questions from HLE dataset")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Generate model predictions
    print("Generating model responses...")
    predictions = {}
    
    # Use LLMAsyncProcessor for batch processing
    generator_config = {"max_tokens": max_completion_tokens}
    inputs = [
        ([{"role": "user", "content": q["question"]}], generator_config)
        for q in questions
    ]
    
    try:
        llm_ap = LLMAsyncProcessor(llm=llm, inputs=inputs)
        results = llm_ap.get_results()
        
        for q, result in zip(questions, results):
            if result and result.content:
                predictions[q["id"]] = {
                    "model": cfg.model.pretrained_model_name_or_path,
                    "response": result.content,
                    "usage": {
                        "completion_tokens": len(result.content.split()),
                        "prompt_tokens": len(q["question"].split())
                    }
                }
        
        print(f"Generated {len(predictions)} model responses")
        
    except Exception as e:
        print(f"Error generating model responses: {e}")
        return
    
    # Judge responses
    print("Judging responses...")
    try:
        judged_predictions = asyncio.run(
            evaluate_batch(questions, predictions, judge_model, max_workers)
        )
        print(f"Judged {len(judged_predictions)} responses")
        
    except Exception as e:
        print(f"Error in judging responses: {e}")
        return
    
    # Calculate metrics
    metrics = calculate_metrics(judged_predictions, len(questions))
    
    print("*** HLE Evaluation Results ***")
    print(f"Accuracy: {metrics['accuracy']}% +/- {metrics['confidence_half_width']}% | n = {metrics['total_questions']}")
    print(f"Calibration Error: {metrics['calibration_error']}")
    
    # Prepare data for logging
    results_data = []
    for q in questions:
        if q["id"] in judged_predictions:
            pred = judged_predictions[q["id"]]
            judge_resp = pred.get("judge_response", {})
            results_data.append({
                "id": q["id"],
                "question": q["question"][:100] + "..." if len(q["question"]) > 100 else q["question"],
                "correct_answer": q["answer"],
                "model_response": pred["response"][:200] + "..." if len(pred["response"]) > 200 else pred["response"],
                "model_answer": judge_resp.get("model_answer", ""),
                "correct": judge_resp.get("correct", ""),
                "confidence": judge_resp.get("confidence", 0),
                "reasoning": judge_resp.get("reasoning", "")[:100] + "..." if len(judge_resp.get("reasoning", "")) > 100 else judge_resp.get("reasoning", "")
            })
    
    # Create output table
    output_df = pd.DataFrame(results_data)
    output_df.insert(0, 'model_name', cfg.model.pretrained_model_name_or_path)
    hle_output_table = wandb.Table(dataframe=output_df)
    
    # Create leaderboard table
    leaderboard_data = {
        "model_name": cfg.model.pretrained_model_name_or_path,
        "accuracy": metrics["accuracy"],
        "calibration_error": metrics["calibration_error"],
        "confidence_half_width": metrics["confidence_half_width"],
        "total_questions": metrics["total_questions"],
        "answered_questions": metrics["answered_questions"]
    }
    leaderboard_df = pd.DataFrame([leaderboard_data])
    hle_leaderboard_table = wandb.Table(dataframe=leaderboard_df)
    
    # Log to wandb
    run.log({
        "hle_output_table": hle_output_table,
        "hle_leaderboard_table": hle_leaderboard_table,
        "hle_accuracy": metrics["accuracy"],
        "hle_calibration_error": metrics["calibration_error"],
    })
    
    print("HLE evaluation completed and logged to wandb")