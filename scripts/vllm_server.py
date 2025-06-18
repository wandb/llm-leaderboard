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
import sys
import socket
from pathlib import Path
import logging
import json
from utils import get_tokenizer_config

disable_logger_apis =[
    "httpx",
    "openai",
    "openai._base_client", 
    "openai._client",
    "anthropic",
    "anthropic._base_client",
    "mistralai", 
    "google.generativeai",
    "cohere",
    "boto3",
    "botocore"
]
    
logging.basicConfig(level=logging.INFO)
for logger_name in disable_logger_apis:
    logging.getLogger(logger_name).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def find_and_kill_process_on_port(port):
    try:
        # ポートを使用しているプロセスを見つける
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                for conns in proc.connections(kind='inet'):
                    if conns.laddr.port == port:
                        logger.info(f"Found process using port {port}: PID {proc.pid}")
                        # プロセスツリー全体を終了
                        parent = psutil.Process(proc.pid)
                        children = parent.children(recursive=True)
                        for child in children:
                            try:
                                child.terminate()
                                child.wait(timeout=3)
                            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                                try:
                                    child.kill()
                                except psutil.NoSuchProcess:
                                    pass
                        
                        try:
                            parent.terminate()
                            parent.wait(timeout=3)
                        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                            try:
                                parent.kill()
                            except psutil.NoSuchProcess:
                                pass
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as e:
        logger.error(f"Error while finding and killing process: {e}")
    return False

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            # SO_REUSEADDRを設定
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('localhost', port))
            return False
        except socket.error:
            return True

def wait_for_port_release(port, timeout=30):
    start_time = time.time()
    while is_port_in_use(port):
        if time.time() - start_time > timeout:
            # タイムアウト時に強制的にプロセスを終了
            if find_and_kill_process_on_port(port):
                logger.info(f"Forcefully killed process using port {port}")
                time.sleep(2)  # プロセス終了待機
                if not is_port_in_use(port):
                    return
            raise TimeoutError(f"Port {port} was not released after {timeout} seconds")
        time.sleep(1)

def force_cuda_memory_cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

def wait_for_gpu_memory_release(timeout=60):
    start_time = time.time()
    while torch.cuda.memory_allocated() > 0:
        if time.time() - start_time > timeout:
            logger.warning(f"GPU memory not fully released after {timeout} seconds")
            break
        logger.info(f"Waiting for GPU memory to be released. {torch.cuda.memory_allocated()} bytes still allocated")
        force_cuda_memory_cleanup()
        time.sleep(1)

def force_cleanup(server_process):
    try:
        # プロセスツリー全体を終了
        parent = psutil.Process(server_process.pid)
        children = parent.children(recursive=True)
        
        # 子プロセスを終了
        for child in children:
            try:
                child.terminate()
                child.wait(timeout=3)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass

        # 親プロセスを終了
        try:
            parent.terminate()
            parent.wait(timeout=3)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            try:
                parent.kill()
            except psutil.NoSuchProcess:
                pass

    except psutil.NoSuchProcess:
        pass
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def start_vllm_server():
    instance = WandbConfigSingleton.get_instance()
    cfg = instance.config
    run = instance.run

    port = cfg.get("port", 8000)
    
    # 起動前のクリーンアップ
    if is_port_in_use(port):
        logger.info(f"Port {port} is in use. Attempting to clean up...")
        find_and_kill_process_on_port(port)
        wait_for_port_release(port)
    
    force_cuda_memory_cleanup()
    wait_for_gpu_memory_release()

    model_artifact_path = cfg.model.get("artifacts_path", None)
    if model_artifact_path is not None:
        artifact = run.use_artifact(model_artifact_path, type='model')
        artifact = Path(artifact.download())
        cfg.model.update({"local_path": artifact / artifact.name.split(":")[0]})

    def run_vllm_server():
        tokenizer_config = get_tokenizer_config()
        cfg.update({"tokenizer_config": tokenizer_config})
        chat_template: str = cfg.tokenizer_config.get("chat_template", None)
        if chat_template is None:
            raise ValueError("chat_template is None. Please provide a valid chat_template in the configuration.")

        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as temp_file:
            temp_file.write(chat_template)
            chat_template_path = temp_file.name
            model_id = cfg.model.pretrained_model_name_or_path
            model_path = cfg.model.get("local_path", model_id)

            command = [
                "python3", "-m", "vllm.entrypoints.openai.api_server",
                "--model", str(model_path),
                "--served-model-name", model_id,
                "--dtype", cfg.model.dtype, 
                "--chat-template", chat_template_path,
                "--max-model-len", str(cfg.model.max_model_len),
                "--max-num-seqs", str(cfg.batch_size),
                "--tensor-parallel-size", str(cfg.get("num_gpus", 4)),
                "--device", cfg.model.device_map,
                "--seed", "42",
                "--uvicorn-log-level", "warning",
                "--disable-log-stats",
                "--disable-log-requests",
                "--revision", str(cfg.get("revision", 'main')),
                "--gpu-memory-utilization", str(cfg.get("gpu_memory_utilization", 0.9)),
                "--port", str(port),
            ]

            # quantizationが指定されている場合のみ追加
            quantization = cfg.get("quantization", None)
            if quantization is not None and str(quantization).lower() != 'none':
                command.extend(["--quantization", str(quantization)])

            # LoRAの設定を追加
            lora_config = cfg.model.get("lora", None)
            if lora_config and lora_config.get("enable", False):
                command.append("--enable-lora")
                
                # LoRAモジュールの設定を追加
                adapter_name = lora_config.get("adapter_name", "default-lora")
                adapter_path = lora_config.get("adapter_path")
                base_model_name = cfg.model.get("base_model_name_or_path", cfg.model.pretrained_model_name_or_path)
                if adapter_name and adapter_path:
                    lora_config_json = {
                        "name": adapter_name,
                        "path": adapter_path,
                        "base_model_name": base_model_name
                    }
                    command.extend(["--lora-modules", json.dumps(lora_config_json)])
                
                # その他のLoRA関連のオプションを追加
                if "max_lora_rank" in lora_config:
                    command.extend(["--max-lora-rank", str(lora_config["max_lora_rank"])])
                if "max_loras" in lora_config:
                    command.extend(["--max-loras", str(lora_config["max_loras"])])
                if "max_cpu_loras" in lora_config:
                    command.extend(["--max-cpu-loras", str(lora_config["max_cpu_loras"])])
                if lora_config.fully_sharded_loras:
                    command.append("--fully-sharded-loras")
                    
            if cfg.model.trust_remote_code:
                command.append("--trust-remote-code")

            print(command)
            process = subprocess.Popen(command)

        with open('vllm_server.pid', 'w') as pid_file:
            pid_file.write(str(process.pid))
        
        time.sleep(15)
        return process

    def health_check():
        url = f"http://localhost:{port}/health"
        max_retries = 30
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    logger.info("Health check passed!")
                    return True
                else:
                    logger.warning(f"Health check failed with status code: {response.status_code}")
            except requests.ConnectionError:
                logger.info(f"Failed to connect to the server. Retry {retry_count + 1}/{max_retries}")
            retry_count += 1
            time.sleep(10)
        
        raise TimeoutError("Server failed to start after maximum retries")

    server_process = run_vllm_server()
    logger.info("vLLM server is starting...")

    def cleanup():
        logger.info("Terminating vLLM server...")
        force_cleanup(server_process)
        # ポートの解放を確認
        try:
            wait_for_port_release(port)
        except TimeoutError:
            logger.warning(f"Port {port} could not be released")
        force_cuda_memory_cleanup()
        wait_for_gpu_memory_release()

    atexit.register(cleanup)

    def handle_sigterm(signum, frame):
        logger.info("SIGTERM received. Shutting down vLLM server gracefully...")
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)

    health_check()

def shutdown_vllm_server():
    instance = WandbConfigSingleton.get_instance()
    port = instance.config.get("port", 8000)
    
    try:
        with open('vllm_server.pid', 'r') as pid_file:
            pid = int(pid_file.read().strip())
        
        process = psutil.Process(pid)
        children = process.children(recursive=True)
        
        # 子プロセスを終了
        for child in children:
            try:
                child.terminate()
                child.wait(timeout=3)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
        
        # 親プロセスを終了
        try:
            process.terminate()
            process.wait(timeout=3)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        
        logger.info(f"vLLM server with PID {pid} has been terminated.")
        
        os.remove('vllm_server.pid')
        
    except (psutil.NoSuchProcess, FileNotFoundError) as e:
        logger.warning(f"Process or PID file not found: {e}")
    except Exception as e:
        logger.error(f"An error occurred while shutting down vLLM server: {e}")
    finally:
        # 使用中のポートを強制的に解放
        if is_port_in_use(port):
            find_and_kill_process_on_port(port)
        
        logger.info("Cleaning up GPU memory...")
        force_cuda_memory_cleanup()
        wait_for_gpu_memory_release()
        logger.info("GPU memory cleanup completed.")

        try:
            wait_for_port_release(port)
        except TimeoutError:
            logger.warning(f"Port {port} could not be released")

        time.sleep(15)