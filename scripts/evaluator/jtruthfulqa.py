import json
import wandb
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config_singleton import WandbConfigSingleton
from .evaluate_utils import LLMAsyncProcessor
from llm_inference_adapter import get_llm_inference_engine
from vllm_server import shutdown_vllm_server, start_vllm_server
from docker_vllm_manager import stop_vllm_container_if_needed, start_vllm_container_if_needed
import asyncio

def load_questions(artifact_dir):
    questions = []
    with open(artifact_dir + "/JTruthfulQA_all.jsonl", "r") as f:
        for line in f:
            questions.append(json.loads(line))
    return questions

def generate_answers(questions, llm):
    # 質問をメッセージ形式に変換
    inputs = []
    for q in questions:
        messages = [{"role": "user", "content": q["Question"]}]
        generator_config = {"max_tokens": 256}
        inputs.append((messages, generator_config))
    
    # 非同期処理で回答を生成
    llm_ap = LLMAsyncProcessor(llm=llm, inputs=inputs)
    results = llm_ap.get_results()
    answers = [r.content for r in results]
    return answers



class RoBERTaEvaluator:
    def __init__(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)

    def evaluate(self, question, answer):
        inputs = self.tokenizer(question + " " + answer, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probabilities[0][1].item()

def evaluate():
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    llm = instance.llm
    api_type = cfg.api

    # データセットのロード
    artifact_path = cfg.jtruthfulqa.artifact_path
    artifact_dir = run.use_artifact(artifact_path, type='dataset').download()
    questions = load_questions(artifact_dir)
    if cfg.testmode:
        questions = questions[:10]

    # vllmサーバーの起動と回答の生成
    answers = generate_answers(questions, llm)
    for q, a in zip(questions, answers):
        q.update({"answer": a})
    
        # APIタイプに応じてvLLMサーバー/コンテナをシャットダウン
    lifecycle_mode = cfg.vllm.get("lifecycle", "auto")
    
    # RoBERTaの推論中はGPUメモリを専有するため、vLLMを一時停止する
    # lifecycle: 'always_on' が指定されている場合を除く
    if lifecycle_mode != 'always_on':
        if api_type == "vllm-local":
            shutdown_vllm_server()
        elif api_type in ["vllm", "vllm-docker"]:
            stop_vllm_container_if_needed()

    # RoBERTa評価器の初期化
    roberta_evaluator = RoBERTaEvaluator(cfg.jtruthfulqa.roberta_model_name)
    
    results = []
    for q in tqdm(questions, total=len(questions), desc="Evaluating"):
        try:
            score = roberta_evaluator.evaluate(q["Question"], q["answer"])
            q["score"] = score
            q["Label"] = 1 if score > 0.5 else 0
            results.append(q)
        except Exception as e:
            print(f"Error processing question: {e}")
            continue

    # RoBERTaモデルの削除
    del roberta_evaluator
    torch.cuda.empty_cache()
    try:
        torch.cuda.synchronize()
    except RuntimeError as e:
        if "No CUDA GPUs are available" not in str(e):
            raise  # 他のエラーの場合は再度発生させる
        # GPUが利用できない場合は無視する

    # 結果の集計
    df_results = pd.DataFrame(results)
    category_list = df_results['Category'].unique().tolist()
    category_scores = df_results.groupby('Category')['Label'].mean().loc[category_list]
    type_list = df_results['Type'].unique().tolist()
    type_scores = df_results.groupby('Type')['Label'].mean().loc[type_list]
    overall_score = df_results['Label'].mean()
    
    # jtruthfulqa_output_table
    jtruthfulqa_output_table = wandb.Table(dataframe=df_results)

    # jtruthfulqa_radar_table
    radar_data = type_scores.reset_index()
    radar_data.columns = ['Type', 'score']
    jtruthfulqa_radar_table = wandb.Table(dataframe=radar_data)

    # jtruthfulqa_leaderboard_table
    leaderboard_data = {
        "model_name": cfg.model.pretrained_model_name_or_path,
        "overall_score": overall_score,
    }
    leaderboard_data.update(type_scores.to_dict())
    leaderboard_data.update(category_scores.to_dict())
    columns = list(leaderboard_data.keys())
    data = [list(leaderboard_data.values())]
    jtruthfulqa_leaderboard_table = wandb.Table(data=data, columns=columns)

    # 結果のログ
    run.log({
        "jtruthfulqa_output_table": jtruthfulqa_output_table,
        "jtruthfulqa_radar_table": jtruthfulqa_radar_table,
        "jtruthfulqa_leaderboard_table": jtruthfulqa_leaderboard_table,
    })

    print(f"JTruthfulQA Evaluation Complete. Overall Score: {overall_score:.4f}")

    # 評価後にvLLMを再起動
    if lifecycle_mode != 'always_on':
        if api_type == "vllm-local":
            llm = get_llm_inference_engine()
            instance.llm = llm
        elif api_type in ["vllm", "vllm-docker"]:
            start_vllm_container_if_needed()
            # llmインスタンスは同じものを使い続ける（コンテナが再起動しても接続情報は同じ）
            pass

if __name__ == "__main__":
    evaluate()