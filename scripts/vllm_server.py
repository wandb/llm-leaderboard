from config_singleton import WandbConfigSingleton
import subprocess
import time
import requests
import atexit
import tempfile
import os
import signal
import psutil
import torch
from pathlib import Path

from utils import get_tokenizer_config


def start_vllm_server():
    instance = WandbConfigSingleton.get_instance()
    cfg = instance.config
    run = instance.run

    model_artifact_path = cfg.model.get("artifact_path", None)
    if model_artifact_path is not None:
        artifact = run.use_artifact(model_artifact_path, type='model')
        artifact = Path(artifact.download())
        cfg.model.update({"local_path": artifact / artifact.name.split(":")[0]})

    def run_vllm_server():
        # set tokenizer_config
        tokenizer_config = get_tokenizer_config()
        cfg.update({"tokenizer_config": tokenizer_config})
        chat_template: str = cfg.tokenizer_config.get("chat_template", None)
        if chat_template is None:
            raise ValueError("chat_template is None. Please provide a valid chat_template in the configuration.")

        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_file:
            # chat_templateをファイルに書き込んでパスを取得
            temp_file.write(chat_template)
            chat_template_path = temp_file.name
            model_id = cfg.model.pretrained_model_name_or_path
            model_path = cfg.model.get("local_path", model_id)

            # サーバーを起動するためのコマンド
            command = [
                "python3", "-m", "vllm.entrypoints.openai.api_server",
                "--model", str(model_path),
                "--served-model-name", model_id,
                # "--tokenizer", str(model_path),
                "--dtype", cfg.model.dtype, 
                "--chat-template", chat_template_path,
                "--max-model-len", str(cfg.model.max_model_len),
                "--max-num-seqs", str(cfg.batch_size),
                "--tensor-parallel-size", str(cfg.get("num_gpus", 1)),
                "--seed", "42",
                "--uvicorn-log-level", "warning",
                "--disable-log-stats",
                "--disable-log-requests",
            ]

            # subprocessでサーバーをバックグラウンドで実行
            process = subprocess.Popen(command)

        # プロセスIDをファイルに保存
        with open('vllm_server.pid', 'w') as pid_file:
            pid_file.write(str(process.pid))
        # サーバーが起動するのを待つ
        time.sleep(10)

        # サーバーのプロセスを返す
        return process

    def health_check():
        url = "http://localhost:8000/health"
        while True:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    print("Health check passed!")
                    break
                else:
                    print(f"Health check failed with status code: {response.status_code}")
            except requests.ConnectionError:
                print("Failed to connect to the server. Retrying...")
            time.sleep(10)  # 待機してから再試行

    # サーバーを起動
    server_process = run_vllm_server()
    print("vLLM server is starting...")

    # スクリプト終了時にサーバーを終了する
    def cleanup():
        print("Terminating vLLM server...")
        server_process.terminate()
        server_process.wait()

    atexit.register(cleanup)

    # SIGTERMシグナルをキャッチしてサーバーを終了する
    def handle_sigterm(signal, frame):
        print("SIGTERM received. Shutting down vLLM server gracefully...")
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)

    # サーバーが完全に起動するのを待つ
    health_check()

import asyncio
import psutil
import time
import torch
import os
import signal

def force_cuda_memory_cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        # すべての CUDA デバイスに対してメモリをクリア
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

def wait_for_gpu_memory_release(timeout=60):
    start_time = time.time()
    while torch.cuda.memory_allocated() > 0:
        if time.time() - start_time > timeout:
            print(f"Warning: GPU memory not fully released after {timeout} seconds.")
            break
        print(f"Waiting for GPU memory to be released. {torch.cuda.memory_allocated()} bytes still allocated.")
        force_cuda_memory_cleanup()
        time.sleep(1)

def shutdown_vllm_server():
    try:
        with open('vllm_server.pid', 'r') as pid_file:
            pid = int(pid_file.read().strip())
        
        process = psutil.Process(pid)
        
        # 同期的にプロセスを終了
        process.terminate()
        try:
            process.wait(timeout=30)
        except psutil.TimeoutExpired:
            print("Termination timed out. Killing the process.")
            process.kill()
        
        print(f"vLLM server with PID {pid} has been terminated.")
        
        # PIDファイルを削除
        os.remove('vllm_server.pid')
        
    except psutil.NoSuchProcess:
        print(f"Process with PID {pid} not found. It may have already been terminated.")
    except FileNotFoundError:
        print("PID file not found. vLLM server may not be running.")
    except Exception as e:
        print(f"An error occurred while shutting down vLLM server: {e}")
    finally:
        # GPU メモリのクリーンアップ
        print("Cleaning up GPU memory...")
        force_cuda_memory_cleanup()
        wait_for_gpu_memory_release()
        print("GPU memory cleanup completed.")

    # 少し待機してリソースが確実に解放されるのを待つ
    time.sleep(10)