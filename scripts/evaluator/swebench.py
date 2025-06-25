import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from toolz import pipe
from tqdm import tqdm
import wandb
import re
import subprocess
import tempfile
import os
import shutil
import logging
from typing import Dict, List, Any, Optional
import yaml
import hashlib

from config_singleton import WandbConfigSingleton
from .evaluate_utils import (
    apply_chat_template,
    get_few_shot_messages,
    LLMAsyncProcessor,
    normalize,
    text_formatter,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SWEBenchEvaluator:
    """公式SWE-Bench Verifiedコードベースを使用した評価クラス"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.temp_dir = Path(tempfile.mkdtemp(prefix="swebench_eval_"))
        self.predictions_file = self.temp_dir / "predictions.jsonl"
        
        # 公式SWE-Benchコードベースのセットアップ
        self.setup_swebench_repo()
        
    def setup_swebench_repo(self):
        """公式SWE-Benchリポジトリをクローン・セットアップ"""
        self.swebench_dir = self.temp_dir / "SWE-bench"
        
        if not self.swebench_dir.exists():
            logger.info("Cloning official SWE-bench repository...")
            subprocess.run([
                "git", "clone", 
                "https://github.com/princeton-nlp/SWE-bench.git",
                str(self.swebench_dir)
            ], check=True)
            
            # SWE-benchパッケージをインストール
            logger.info("Installing SWE-bench package...")
            subprocess.run([
                "pip", "install", "-e", str(self.swebench_dir)
            ], check=True)
        
    def cleanup(self):
        """一時ディレクトリをクリーンアップ"""
        try:
            shutil.rmtree(self.temp_dir)
            logger.info("Cleaned up temporary directory")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {e}")
            
    def extract_patch_from_response(self, response: str) -> str:
        """レスポンスからパッチを抽出（公式と同様の方法）"""
        # 公式SWE-Benchと同じパッチ抽出ロジック
        patch_patterns = [
            # diff --git形式
            r'```(?:diff|patch)?\s*\n(diff --git.*?)\n```',
            r'(diff --git.*?)(?=\n\n(?![\+\-\s@])|$)',
            # @@マーカーを含むdiff
            r'```(?:diff|patch)?\s*\n(.*?@@.*?)\n```',
            r'(@@.*?)(?=\n\n(?![\+\-\s@])|$)',
            # その他のコードブロック
            r'```\s*\n(.*?)\n```',
        ]
        
        for pattern in patch_patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.MULTILINE)
            if matches:
                patch = matches[0].strip()
                # 基本的な妥当性チェック
                if any(marker in patch for marker in ['diff --git', '@@', '---', '+++']):
                    return patch
        
        # パッチパターンが見つからない場合、全体をパッチとして扱う
        return response.strip()
    
    def format_problem_statement(self, instance: Dict) -> str:
        """公式SWE-Benchスタイルのプロンプト生成"""
        repo = instance["repo"]
        instance_id = instance["instance_id"]
        problem_statement = instance["problem_statement"]
        hints_text = instance.get("hints_text", "")
        
        # 公式SWE-Benchのプロンプトテンプレートに従う
        prompt = f"""You will be provided with a partial code base and an issue statement explaining a problem to resolve.

<issue>
{problem_statement}
</issue>

{f"<hints>{hints_text}</hints>" if hints_text.strip() else ""}

Please generate a patch in unified diff format to resolve this issue. The patch should:
1. Correctly address the issue described
2. Not break any existing functionality
3. Follow the existing code style and patterns

Generate only the patch content in diff format without any additional explanations."""
        
        return prompt
    
    def save_predictions(self, evaluation_results: List[Dict], predictions: List[str]):
        """予測結果を公式フォーマットで保存"""
        predictions_data = []
        
        for eval_result, prediction in zip(evaluation_results, predictions):
            # 公式SWE-Benchの予測フォーマット
            pred_data = {
                "instance_id": eval_result["instance_id"],
                "model_patch": prediction,
                "model_name_or_path": self.cfg.model.pretrained_model_name_or_path,
            }
            predictions_data.append(pred_data)
        
        # JSONLファイルとして保存
        with open(self.predictions_file, 'w') as f:
            for pred in predictions_data:
                f.write(json.dumps(pred) + '\n')
                
        logger.info(f"Saved predictions to {self.predictions_file}")
        return self.predictions_file
    
    def run_official_evaluation(self, dataset_path: Path, predictions_path: Path) -> Dict:
        """公式SWE-Benchの評価スクリプトを実行"""
        logger.info("Running official SWE-bench evaluation...")
        
        # 公式評価スクリプトのパス
        eval_script = self.swebench_dir / "swebench" / "harness" / "run_evaluation.py"
        
        # 評価結果の出力ディレクトリ
        output_dir = self.temp_dir / "evaluation_results"
        output_dir.mkdir(exist_ok=True)
        
        # 公式評価コマンドを実行
        cmd = [
            "python", str(eval_script),
            "--dataset_name", "princeton-nlp/SWE-bench_Verified",
            "--predictions_path", str(predictions_path),
            "--max_workers", str(self.cfg.swebench.get("max_workers", 4)),
            "--run_id", f"nejumi_eval_{int(time.time())}",
            "--output_dir", str(output_dir)
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=7200,  # 2時間タイムアウト
                cwd=str(self.swebench_dir)
            )
            
            if result.returncode != 0:
                logger.error(f"Evaluation failed: {result.stderr}")
                raise RuntimeError(f"Official evaluation failed: {result.stderr}")
            
            logger.info("Official evaluation completed successfully")
            
            # 評価結果を読み込み
            results_file = output_dir / "results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                return results
            else:
                # 結果ファイルを検索
                result_files = list(output_dir.glob("*.json"))
                if result_files:
                    with open(result_files[0], 'r') as f:
                        results = json.load(f)
                    return results
                else:
                    raise RuntimeError("No evaluation results found")
                    
        except subprocess.TimeoutExpired:
            logger.error("Evaluation timed out")
            raise RuntimeError("Evaluation timed out after 2 hours")
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            raise

def evaluate():
    """公式SWE-Bench Verifiedを使用した完全評価"""
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    llm = instance.llm

    # データセットをダウンロード
    dataset_name = "swebench"
    artifact = run.use_artifact(cfg[dataset_name].artifacts_path, type="dataset")
    artifact_dir = artifact.download()
    dataset_dir = Path(artifact_dir) / cfg[dataset_name].dataset_dir
    
    if not dataset_dir.exists():
        print(f"skip {dataset_name} because it is not found in {artifact_dir}")
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

    # SWE-Bench Verifiedデータの読み込み
    dataset_file = dataset_dir / "swe_bench_verified.json"
    if not dataset_file.exists():
        dataset_file = dataset_dir / "swe_bench_verified.jsonl"
        
    if not dataset_file.exists():
        print(f"SWE-Bench Verified dataset not found in {dataset_dir}")
        return

    # データセットを読み込み
    with dataset_file.open(encoding="utf-8") as f:
        if dataset_file.suffix == ".jsonl":
            task_data = [json.loads(line) for line in f]
        else:
            task_data = json.load(f)

    # サンプル数を設定
    if cfg.testmode:
        max_num_samples = 2
    else:
        max_num_samples = cfg.swebench.get("max_samples", 500)

    samples = task_data[:max_num_samples]
    print(f"Evaluating {len(samples)} samples from SWE-Bench Verified")

    # 評価器を初期化
    evaluator = SWEBenchEvaluator(cfg)
    
    try:
        # 評価データを準備
        evaluation_results = []
        for idx, sample in enumerate(samples):
            instance_id = sample["instance_id"]
            
            # プロンプトを構築（公式スタイル）
            formatted_prompt = evaluator.format_problem_statement(sample)
            messages = [{"role": "user", "content": formatted_prompt}]
            
            # 生成パラメータ
            generator_config = {"max_tokens": cfg.swebench.get("max_tokens", 4096)}
            inputs = [messages, generator_config]
            
            eval_metadata = {
                "model_name": cfg.model.pretrained_model_name_or_path,
                "dataset": dataset_name,
                "task": "swe_bench_verified", 
                "subset": "test",
                "index": idx,
                "instance_id": instance_id,
                "repo": sample["repo"],
                "base_commit": sample["base_commit"],
                "problem_statement": sample["problem_statement"],
                "hints_text": sample.get("hints_text", ""),
                "prompt": apply_chat_template(messages=messages),
                "inputs": inputs,
            }
            
            evaluation_results.append(eval_metadata)

        # 推論を実行
        print("Generating predictions with LLM...")
        all_inputs = [er["inputs"] for er in evaluation_results]
        llm_ap = LLMAsyncProcessor(
            llm=llm,
            inputs=all_inputs,
        )
        responses = llm_ap.get_results()

        # パッチを抽出
        predictions = []
        for response in responses:
            raw_output = response.content if hasattr(response, 'content') else str(response)
            prediction = evaluator.extract_patch_from_response(raw_output)
            predictions.append(prediction)
            
        # 予測結果を保存（公式フォーマット）
        predictions_file = evaluator.save_predictions(evaluation_results, predictions)
        
        # 公式評価を実行
        print("Running official SWE-bench evaluation...")
        official_results = evaluator.run_official_evaluation(dataset_file, predictions_file)
        
        # 結果を解析・集計
        total_samples = len(evaluation_results)
        resolved_count = official_results.get("resolved", 0)
        
        resolution_rate = resolved_count / total_samples if total_samples > 0 else 0
        
        print(f"=== Official SWE-Bench Verified Results ===")
        print(f"Total samples: {total_samples}")
        print(f"Issues resolved: {resolved_count}")
        print(f"Resolution rate: {resolution_rate:.3f}")
        
        # 詳細統計
        if "stats" in official_results:
            stats = official_results["stats"]
            print(f"Additional stats: {stats}")

        # リーダーボードテーブルを作成
        leaderboard_data = {
            "model_name": cfg.model.pretrained_model_name_or_path,
            "total_samples": total_samples,
            "issues_resolved": resolved_count,
            "resolution_rate": resolution_rate,
            "official_evaluation": True,
        }
        
        # 公式結果に追加統計があれば含める
        if "stats" in official_results:
            leaderboard_data.update(official_results["stats"])
        
        leaderboard_table = pd.DataFrame([leaderboard_data])
        run.log({"swebench_leaderboard_table": wandb.Table(dataframe=leaderboard_table)})

        # 詳細な出力テーブルを作成
        output_data = []
        for i, (er, prediction) in enumerate(zip(evaluation_results, predictions)):
            # 公式結果から個別のresultを取得
            instance_result = None
            if "results" in official_results:
                instance_result = official_results["results"].get(er["instance_id"], {})
            
            output_data.append({
                "instance_id": er["instance_id"],
                "repo": er["repo"],
                "problem_statement": er["problem_statement"][:500] + "..." if len(er["problem_statement"]) > 500 else er["problem_statement"],
                "generated_patch": prediction[:1000] + "..." if prediction and len(prediction) > 1000 else prediction,
                "resolved": instance_result.get("resolved", False) if instance_result else False,
                "error_message": instance_result.get("error", "") if instance_result else "",
            })
        
        output_table = pd.DataFrame(output_data)
        run.log({"swebench_output_table": wandb.Table(dataframe=output_table)})

        # リポジトリ別集計
        repo_stats = {}
        for i, er in enumerate(evaluation_results):
            repo = er["repo"]
            if repo not in repo_stats:
                repo_stats[repo] = {"total": 0, "resolved": 0}
            repo_stats[repo]["total"] += 1
            
            # 公式結果から解決状況を取得
            if "results" in official_results:
                instance_result = official_results["results"].get(er["instance_id"], {})
                if instance_result.get("resolved", False):
                    repo_stats[repo]["resolved"] += 1
        
        repo_data = []
        for repo, stats in repo_stats.items():
            repo_data.append({
                "repository": repo,
                "total_samples": stats["total"],
                "issues_resolved": stats["resolved"],
                "resolution_rate": stats["resolved"] / stats["total"] if stats["total"] > 0 else 0,
            })
        
        repo_table = pd.DataFrame(repo_data)
        run.log({"swebench_repo_breakdown_table": wandb.Table(dataframe=repo_table)})

        # 公式結果の詳細をWandBにログ
        if official_results:
            run.log({"swebench_official_results": official_results})

        print("SWE-Bench Verified evaluation completed successfully using official codebase!")
        
    finally:
        # クリーンアップ
        evaluator.cleanup() 