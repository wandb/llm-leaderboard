#!/usr/bin/env python3
"""vLLM起動の詳細なテスト"""

import subprocess
import time
import requests
import json

def run_command(cmd, capture=True):
    """コマンドを実行して結果を返す"""
    print(f"\n実行: {' '.join(cmd)}")
    if capture:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            print(f"出力: {result.stdout}")
        if result.stderr:
            print(f"エラー: {result.stderr}")
        return result
    else:
        return subprocess.run(cmd)

def monitor_vllm_startup():
    """vLLMの起動を詳細にモニタリング"""
    print("\n=== vLLM起動の詳細モニタリング ===")
    
    # 既存のコンテナを削除
    run_command(["docker", "rm", "-f", "test-vllm-monitor"], capture=False)
    
    # トークンを取得
    result = run_command([
        "docker", "exec", "llm-leaderboard", 
        "bash", "-c", "echo $HUGGINGFACE_HUB_TOKEN"
    ])
    token = result.stdout.strip()
    print(f"トークン: {token[:20]}...")
    
    # vLLMを起動
    cmd = [
        "docker", "run", "-d", "--name", "test-vllm-monitor",
        "--network", "llm-stack-network",
        "--gpus", "all",
        "-p", "18000:8000",  # 別のポートを使用
        "-e", f"HUGGINGFACE_HUB_TOKEN={token}",
        "-e", f"HF_TOKEN={token}",
        "-e", "VLLM_USAGE_SOURCE=llm-leaderboard",
        "-e", "CUDA_VISIBLE_DEVICES=0",
        "vllm/vllm-openai:v0.5.5",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--model", "meta-llama/Meta-Llama-3-8B-Instruct",
        "--dtype", "half"
    ]
    
    result = run_command(cmd)
    if result.returncode != 0:
        print("コンテナ起動に失敗しました")
        return
    
    container_id = result.stdout.strip()
    print(f"コンテナID: {container_id}")
    
    # 起動プロセスをモニタリング
    start_time = time.time()
    max_wait = 60  # 60秒待つ
    
    while time.time() - start_time < max_wait:
        # コンテナの状態を確認
        status_result = run_command([
            "docker", "inspect", "test-vllm-monitor",
            "--format", "{{.State.Status}}"
        ])
        status = status_result.stdout.strip()
        
        print(f"\n時間: {int(time.time() - start_time)}秒")
        print(f"コンテナ状態: {status}")
        
        if status != "running":
            print("コンテナが停止しました")
            # 最後のログを表示
            run_command(["docker", "logs", "test-vllm-monitor", "--tail", "100"])
            break
        
        # 最新のログを表示（最後の5行）
        log_result = run_command([
            "docker", "logs", "test-vllm-monitor", "--tail", "5"
        ])
        
        # ヘルスチェック
        try:
            response = requests.get("http://localhost:18000/health", timeout=2)
            print(f"ヘルスチェック: {response.status_code}")
            if response.status_code == 200:
                print("vLLMが正常に起動しました！")
                
                # モデル情報を確認
                models_response = requests.get("http://localhost:18000/v1/models", timeout=2)
                if models_response.status_code == 200:
                    print(f"モデル情報: {models_response.json()}")
                break
        except requests.exceptions.RequestException:
            print("ヘルスチェック: 接続できません")
        
        time.sleep(5)
    
    # クリーンアップ
    print("\nクリーンアップ中...")
    run_command(["docker", "rm", "-f", "test-vllm-monitor"], capture=False)

def test_with_different_model():
    """別のモデルでテスト"""
    print("\n=== 別のモデルでのテスト ===")
    
    # 既存のコンテナを削除
    run_command(["docker", "rm", "-f", "test-vllm-alt"], capture=False)
    
    # トークンを取得
    result = run_command([
        "docker", "exec", "llm-leaderboard", 
        "bash", "-c", "echo $HUGGINGFACE_HUB_TOKEN"
    ])
    token = result.stdout.strip()
    
    # 公開モデルで試す
    cmd = [
        "docker", "run", "-d", "--name", "test-vllm-alt",
        "--network", "llm-stack-network",
        "--gpus", "all",
        "-e", f"HUGGINGFACE_HUB_TOKEN={token}",
        "vllm/vllm-openai:v0.5.5",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--model", "microsoft/phi-2",  # 公開モデル
        "--dtype", "half"
    ]
    
    result = run_command(cmd)
    if result.returncode == 0:
        print("コンテナ起動成功")
        time.sleep(10)
        
        # ログを確認
        log_result = run_command(["docker", "logs", "test-vllm-alt", "--tail", "20"])
        
        # 状態を確認
        status_result = run_command([
            "docker", "inspect", "test-vllm-alt",
            "--format", "{{.State.Status}}"
        ])
        print(f"コンテナ状態: {status_result.stdout.strip()}")
    
    # クリーンアップ
    run_command(["docker", "rm", "-f", "test-vllm-alt"], capture=False)

if __name__ == "__main__":
    print("vLLM起動の詳細テストを開始します...")
    
    monitor_vllm_startup()
    test_with_different_model()
    
    print("\n\nテスト完了") 