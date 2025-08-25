#!/usr/bin/env python3
"""
SWE-Bench Verifiedデータセットをダウンロードして、WandBアーティファクトとしてアップロードするスクリプト

Usage:
    python scripts/data_uploader/upload_swebench_verified.py --entity your-entity --project your-project
"""

import os
import json
import argparse
import logging
from pathlib import Path
import tempfile
import shutil
import zipfile
from typing import Dict, List

import wandb
import requests
from datasets import load_dataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SWEBenchDataUploader:
    """SWE-Bench Verifiedデータセットアップローダー"""
    
    def __init__(self, entity: str, project: str):
        self.entity = entity
        self.project = project
        self.temp_dir = Path(tempfile.mkdtemp(prefix="swebench_data_"))
        self.dataset_dir = self.temp_dir / "swebench"
        self.dataset_dir.mkdir(exist_ok=True, parents=True)
        
    def cleanup(self):
        """一時ディレクトリをクリーンアップ"""
        try:
            shutil.rmtree(self.temp_dir)
            logger.info("Cleaned up temporary directory")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {e}")
            
    def download_swebench_verified(self) -> Path:
        """SWE-Bench Verifiedデータセットをダウンロード"""
        logger.info("Downloading SWE-Bench Verified dataset from HuggingFace...")
        
        try:
            # HuggingFace Datasetsライブラリを使用してダウンロード
            dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
            
            # JSONファイルとして保存
            output_file = self.dataset_dir / "swe_bench_verified.json"
            
            # データセットをJSON形式に変換
            data = []
            for item in tqdm(dataset, desc="Processing dataset"):
                # 公式SWE-benchの全フィールドを含める
                data_item = {
                    "repo": item["repo"],
                    "instance_id": item["instance_id"],
                    "base_commit": item["base_commit"],
                    "patch": item["patch"],
                    "test_patch": item["test_patch"],
                    "problem_statement": item["problem_statement"],
                    "hints_text": item.get("hints_text", ""),
                    "created_at": item["created_at"],
                    "version": item["version"],
                    "FAIL_TO_PASS": item["FAIL_TO_PASS"],
                    "PASS_TO_PASS": item["PASS_TO_PASS"],
                    # 追加フィールド（必要に応じて）
                    "environment_setup_commit": item.get("environment_setup_commit", ""),
                    # --- add synthesized text field (identical to make_dataset pipeline) ---
                    "text": self._build_text_field(item),
                }
                data.append(data_item)
            
            # JSONファイルに保存
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Downloaded {len(data)} instances to {output_file}")
            
            # 統計情報を出力
            repos = set(item["repo"] for item in data)
            logger.info(f"Dataset contains {len(repos)} unique repositories:")
            for repo in sorted(repos):
                count = sum(1 for item in data if item["repo"] == repo)
                logger.info(f"  {repo}: {count} instances")
            
            # JSONLファイルも作成（公式SWE-benchの形式）
            jsonl_file = self.dataset_dir / "swe_bench_verified.jsonl"
            with open(jsonl_file, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            logger.info(f"Also saved as JSONL: {jsonl_file}")
                
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to download SWE-Bench Verified dataset: {e}")
            raise
    
    def download_additional_resources(self) -> List[Path]:
        """追加リソースをダウンロード（必要に応じて）"""
        additional_files = []
        
        # メタデータファイルを作成
        metadata_file = self.dataset_dir / "metadata.json"
        metadata = {
            "dataset_name": "SWE-Bench Verified",
            "version": "1.0",
            "description": "Expert-verified subset of 500 solvable GitHub issues from SWE-bench",
            "source": "princeton-nlp/SWE-bench_Verified",
            "license": "MIT",
            "total_instances": 500,
            "format": "JSON",
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        additional_files.append(metadata_file)
        
        # READMEファイルを作成
        readme_file = self.dataset_dir / "README.md"
        readme_content = """# SWE-Bench Verified Dataset

## Overview
SWE-Bench Verified is an expert-verified subset of 500 solvable problems from the original SWE-Bench dataset. Each instance has been manually reviewed by experienced Python developers to ensure:

- The issue description is clear and well-specified
- The unit tests are appropriate and related to the GitHub issue
- The development environment can be reliably set up

## Dataset Structure
- `swe_bench_verified.json`: Main dataset file containing all 500 instances
- `metadata.json`: Dataset metadata and statistics

## Instance Format
Each instance contains:
- `repo`: GitHub repository (owner/name)
- `instance_id`: Unique identifier (repo-issue_number)
- `base_commit`: Git commit hash for the base state
- `patch`: Gold solution patch (for reference only)
- `test_patch`: Test patch containing unit tests
- `problem_statement`: Issue description from GitHub
- `hints_text`: Additional hints or comments
- `created_at`: Timestamp of issue creation
- `version`: Repository package version
- `FAIL_TO_PASS`: Test cases that should pass after fix
- `PASS_TO_PASS`: Test cases that should continue passing

## Usage
This dataset is designed for evaluating code generation models on real-world software engineering tasks. Models should generate patches to resolve the GitHub issues, which are then evaluated by applying the patch and running the test suite.

## License
MIT License - See original dataset for details.

## Citation
```
@article{jimenez2024swebench,
  title={SWE-bench: Can Language Models Resolve Real-World GitHub Issues?},
  author={Jimenez, Carlos E and Yang, John and Wettig, Alexander and Yao, Shunyu and Pei, Kexin and Press, Ofir and Narasimhan, Karthik},
  journal={arXiv preprint arXiv:2310.06770},
  year={2024}
}
```
"""
        
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        additional_files.append(readme_file)
        
        return additional_files
    
    def create_validation_script(self) -> Path:
        """データセット検証スクリプトを作成"""
        validation_script = self.dataset_dir / "validate_dataset.py"
        validation_code = '''#!/usr/bin/env python3
"""
SWE-Bench Verified dataset validation script
"""

import json
import sys
from pathlib import Path

def validate_dataset(dataset_path):
    """データセットの整合性を検証"""
    print("Validating SWE-Bench Verified dataset...")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total instances: {len(data)}")
    
    # 必須フィールドの確認
    required_fields = [
        "repo", "instance_id", "base_commit", "patch", "test_patch",
        "problem_statement", "created_at", "version", "FAIL_TO_PASS", "PASS_TO_PASS"
    ]
    
    valid_count = 0
    for i, instance in enumerate(data):
        missing_fields = [field for field in required_fields if field not in instance]
        if missing_fields:
            print(f"Instance {i} missing fields: {missing_fields}")
        else:
            valid_count += 1
    
    print(f"Valid instances: {valid_count}/{len(data)}")
    
    # リポジトリ統計
    repos = {}
    for instance in data:
        repo = instance["repo"]
        repos[repo] = repos.get(repo, 0) + 1
    
    print(f"\\nRepository distribution:")
    for repo, count in sorted(repos.items()):
        print(f"  {repo}: {count}")
    
    return valid_count == len(data)

if __name__ == "__main__":
    dataset_path = Path(__file__).parent / "swe_bench_verified.json"
    if validate_dataset(dataset_path):
        print("\\nDataset validation successful!")
        sys.exit(0)
    else:
        print("\\nDataset validation failed!")
        sys.exit(1)
'''
        
        with open(validation_script, 'w', encoding='utf-8') as f:
            f.write(validation_code)
        
        # 実行権限を付与
        validation_script.chmod(0o755)
        
        return validation_script
    
    def upload_to_wandb(self) -> str:
        """データセットをWandBにアップロード"""
        logger.info("Uploading SWE-Bench Verified dataset to WandB...")
        
        # WandBを初期化
        wandb.login()
        run = wandb.init(
            entity=self.entity,
            project=self.project,
            job_type="data-upload",
            name="swebench-verified-upload"
        )
        
        try:
            # アーティファクトを作成
            artifact = wandb.Artifact(
                name="swebench_verified",
                type="dataset",
                description="SWE-Bench Verified dataset: Expert-verified subset of 500 solvable GitHub issues",
                metadata={
                    "dataset_name": "SWE-Bench Verified",
                    "total_instances": 500,
                    "source": "princeton-nlp/SWE-bench_Verified",
                    "license": "MIT",
                    "format": "JSON"
                }
            )
            
            # データセットディレクトリ全体を追加
            artifact.add_dir(str(self.dataset_dir), name="swebench")
            
            # アーティファクトをログ
            run.log_artifact(artifact)
            
            # アーティファクトパスを取得
            artifact_path = f"{self.entity}/{self.project}/swebench_verified:latest"
            
            logger.info(f"Successfully uploaded to: {artifact_path}")
            
            return artifact_path
            
        finally:
            run.finish()
    
    def run(self) -> str:
        """完全なアップロードプロセスを実行"""
        try:
            logger.info("Starting SWE-Bench Verified dataset upload process...")
            
            # データセットをダウンロード
            dataset_file = self.download_swebench_verified()
            
            # 追加リソースをダウンロード
            additional_files = self.download_additional_resources()
            
            # 検証スクリプトを作成
            validation_script = self.create_validation_script()
            
            # データセットを検証
            logger.info("Validating dataset...")
            import subprocess
            result = subprocess.run([str(validation_script)], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Dataset validation failed: {result.stdout}\n{result.stderr}")
                raise RuntimeError("Dataset validation failed")
            
            logger.info("Dataset validation successful")
            
            # WandBにアップロード
            artifact_path = self.upload_to_wandb()
            
            logger.info(f"Upload process completed successfully!")
            logger.info(f"Artifact path: {artifact_path}")
            logger.info(f"Use this path in your config file:")
            logger.info(f"  swebench:")
            logger.info(f"    artifacts_path: \"{artifact_path}\"")
            
            return artifact_path
            
        finally:
            self.cleanup()

    def _build_text_field(self, item: Dict) -> str:
        """Replicate swebench make_dataset text formatting (system + user)."""
        system_header = (
            "You are ChatGPT, a large language model trained by OpenAI. "
            "Follow the user's instructions carefully. "
            "Respond in valid unified diff format."
        )
        issue_block = f"""<issue>\n{item['problem_statement']}\n</issue>"""
        hints_block = f"\n<hints>\n{item.get('hints_text','')}\n</hints>" if item.get("hints_text","" ).strip() else ""

        # Minimal code context: gold patch の変更前ファイルを取得して 120 行ずつ抜粋
        try:
            from swebench.harness.utils import get_modified_files, get_repo_file
            files = get_modified_files(item["patch"])[:2]  # up to 2 files for prompt size
            context_blocks = []
            for fp in files:
                code = get_repo_file(item["repo"], item["base_commit"], fp)
                if code:
                    # 先頭 120 行だけ
                    snippet = "\n".join(code.splitlines()[:120])
                    context_blocks.append(f"""<file path=\"{fp}\">\n{snippet}\n</file>""")
            code_block = "\n\n".join(context_blocks)
        except Exception:
            code_block = ""

        user_prompt = (
            "You will be provided with a GitHub issue and partial repository context. "
            "Generate a patch that fixes the bug. The patch must be a unified diff."
        )

        return f"{system_header}\n\n{user_prompt}\n\n{issue_block}{hints_block}\n\n{code_block}"

def main():
    parser = argparse.ArgumentParser(
        description="Upload SWE-Bench Verified dataset to WandB"
    )
    parser.add_argument(
        "--entity", 
        required=True,
        help="WandB entity name"
    )
    parser.add_argument(
        "--project", 
        required=True,
        help="WandB project name"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download and validate dataset without uploading to WandB"
    )
    
    args = parser.parse_args()
    
    uploader = SWEBenchDataUploader(args.entity, args.project)
    
    if args.dry_run:
        logger.info("Running in dry-run mode...")
        try:
            # データセットをダウンロードして検証のみ
            dataset_file = uploader.download_swebench_verified()
            additional_files = uploader.download_additional_resources()
            validation_script = uploader.create_validation_script()
            
            # 検証実行
            import subprocess
            result = subprocess.run([str(validation_script)], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Dry run completed successfully! Dataset is ready for upload.")
            else:
                logger.error(f"Dry run failed: {result.stdout}\n{result.stderr}")
        finally:
            uploader.cleanup()
    else:
        artifact_path = uploader.run()
        print(f"\n✅ Success! Artifact path: {artifact_path}")

if __name__ == "__main__":
    main() 