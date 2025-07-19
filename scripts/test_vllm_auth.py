#!/usr/bin/env python3
"""vLLMコンテナの認証問題をテストするスクリプト"""

import subprocess
import time
import os
import json

def run_command(cmd, capture=True):
    """コマンドを実行して結果を返す"""
    print(f"\n実行: {' '.join(cmd)}")
    if capture:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            print(f"出力: {result.stdout[:200]}...")
        if result.stderr:
            print(f"エラー: {result.stderr[:200]}...")
        return result
    else:
        return subprocess.run(cmd)

def test_token_in_container():
    """コンテナ内でトークンが正しく設定されているかテスト"""
    print("\n=== テスト1: トークンの確認 ===")
    
    # llm-leaderboardコンテナのトークンを取得
    result = run_command([
        "docker", "exec", "llm-leaderboard", 
        "bash", "-c", "echo $HUGGINGFACE_HUB_TOKEN"
    ])
    token = result.stdout.strip()
    print(f"llm-leaderboardのトークン: {token[:20]}...")
    
    # 環境変数のトークンを確認
    env_token = os.environ.get("HUGGINGFACE_HUB_TOKEN", "")
    print(f"環境変数のトークン: {env_token[:20] if env_token else 'Not set'}...")
    
    return token

def test_direct_access():
    """直接Pythonでアクセステスト"""
    print("\n=== テスト2: 直接アクセステスト ===")
    
    # llm-leaderboardコンテナ内でテスト
    cmd = [
        "docker", "exec", "llm-leaderboard",
        "bash", "-c", 
        "cd /workspace && source .venv/bin/activate && "
        "python3 -c \"from huggingface_hub import HfApi; "
        "api = HfApi(); "
        "print(api.model_info('meta-llama/Meta-Llama-3-8B-Instruct').modelId)\""
    ]
    result = run_command(cmd)
    return result.returncode == 0

def test_vllm_container_env():
    """vLLMコンテナの環境変数を確認"""
    print("\n=== テスト3: vLLMコンテナ環境変数テスト ===")
    
    # 既存のコンテナを削除
    run_command(["docker", "rm", "-f", "test-vllm"], capture=False)
    
    # トークンを取得
    result = run_command([
        "docker", "exec", "llm-leaderboard", 
        "bash", "-c", "echo $HUGGINGFACE_HUB_TOKEN"
    ])
    token = result.stdout.strip()
    
    # テストコンテナを起動（entrypointをオーバーライド）
    cmd = [
        "docker", "run", "--rm", "--name", "test-vllm",
        "-e", f"HUGGINGFACE_HUB_TOKEN={token}",
        "--entrypoint", "python3",
        "vllm/vllm-openai:v0.5.5",
        "-c", 
        "import os; "
        "print(f'Token: {os.environ.get(\"HUGGINGFACE_HUB_TOKEN\", \"Not set\")[:20]}...'); "
        "print(f'HF_TOKEN: {os.environ.get(\"HF_TOKEN\", \"Not set\")[:20]}...')"
    ]
    result = run_command(cmd)
    
def test_vllm_huggingface_access():
    """vLLMコンテナからHugging Faceへのアクセステスト"""
    print("\n=== テスト4: vLLMコンテナからのHFアクセステスト ===")
    
    # トークンを取得
    result = run_command([
        "docker", "exec", "llm-leaderboard", 
        "bash", "-c", "echo $HUGGINGFACE_HUB_TOKEN"
    ])
    token = result.stdout.strip()
    
    # アクセステスト
    python_code = """
import os
from huggingface_hub import HfApi
print(f'Using token: {os.environ.get("HUGGINGFACE_HUB_TOKEN", "Not set")[:20]}...')
api = HfApi()
try:
    info = api.model_info('meta-llama/Meta-Llama-3-8B-Instruct')
    print(f'Success: {info.modelId}')
except Exception as e:
    print(f'Error: {e}')
"""
    
    cmd = [
        "docker", "run", "--rm",
        "-e", f"HUGGINGFACE_HUB_TOKEN={token}",
        "--entrypoint", "python3",
        "vllm/vllm-openai:v0.5.5",
        "-c", python_code.strip()
    ]
    result = run_command(cmd)

def test_vllm_with_hf_token_env():
    """HF_TOKENも試してみる"""
    print("\n=== テスト5: HF_TOKEN環境変数テスト ===")
    
    # トークンを取得
    result = run_command([
        "docker", "exec", "llm-leaderboard", 
        "bash", "-c", "echo $HUGGINGFACE_HUB_TOKEN"
    ])
    token = result.stdout.strip()
    
    # HF_TOKENとHUGGINGFACE_HUB_TOKEN両方を設定
    python_code = """
import os
from huggingface_hub import HfApi
print(f'HUGGINGFACE_HUB_TOKEN: {os.environ.get("HUGGINGFACE_HUB_TOKEN", "Not set")[:20]}...')
print(f'HF_TOKEN: {os.environ.get("HF_TOKEN", "Not set")[:20]}...')
api = HfApi()
try:
    info = api.model_info('meta-llama/Meta-Llama-3-8B-Instruct')
    print(f'Success: {info.modelId}')
except Exception as e:
    print(f'Error: {e}')
"""
    
    cmd = [
        "docker", "run", "--rm",
        "-e", f"HUGGINGFACE_HUB_TOKEN={token}",
        "-e", f"HF_TOKEN={token}",
        "--entrypoint", "python3",
        "vllm/vllm-openai:v0.5.5",
        "-c", python_code.strip()
    ]
    result = run_command(cmd)

def test_actual_vllm_startup():
    """実際のvLLM起動をテスト"""
    print("\n=== テスト6: 実際のvLLM起動テスト ===")
    
    # 既存のコンテナを削除
    run_command(["docker", "rm", "-f", "test-vllm-real"], capture=False)
    
    # トークンを取得
    result = run_command([
        "docker", "exec", "llm-leaderboard", 
        "bash", "-c", "echo $HUGGINGFACE_HUB_TOKEN"
    ])
    token = result.stdout.strip()
    
    # vLLMを起動（デバッグモード）
    cmd = [
        "docker", "run", "-d", "--name", "test-vllm-real",
        "--network", "llm-stack-network",
        "--gpus", "all",
        "-e", f"HUGGINGFACE_HUB_TOKEN={token}",
        "-e", f"HF_TOKEN={token}",
        "-e", "VLLM_USAGE_SOURCE=llm-leaderboard",
        "vllm/vllm-openai:v0.5.5",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--model", "meta-llama/Meta-Llama-3-8B-Instruct",
        "--dtype", "half"
    ]
    result = run_command(cmd)
    
    if result.returncode == 0:
        print("コンテナ起動成功、ログを確認中...")
        time.sleep(10)
        
        # ログを確認
        log_result = run_command(["docker", "logs", "test-vllm-real", "--tail", "50"])
        
        # クリーンアップ
        run_command(["docker", "rm", "-f", "test-vllm-real"], capture=False)

if __name__ == "__main__":
    print("vLLM認証問題のテストを開始します...")
    
    # 各テストを実行
    test_token_in_container()
    test_direct_access()
    test_vllm_container_env()
    test_vllm_huggingface_access()
    test_vllm_with_hf_token_env()
    test_actual_vllm_startup()
    
    print("\n\nテスト完了") 