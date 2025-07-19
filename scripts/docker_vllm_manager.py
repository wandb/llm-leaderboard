import subprocess
import time
import requests
import logging
from typing import Optional
from config_singleton import WandbConfigSingleton
import os

logger = logging.getLogger(__name__)


class DockerVLLMManager:
    """Docker vLLMコンテナの起動・停止・ヘルスチェックを管理"""
    
    def __init__(self):
        self.container_name = "llm-stack-vllm-1"
        self.service_name = "vllm"
        # llm-leaderboardコンテナ内から実行する場合は、コンテナ名を使用
        self.health_check_url = "http://llm-stack-vllm-1:8000/health"
        self.models_check_url = "http://llm-stack-vllm-1:8000/v1/models"
        
    def is_container_running(self) -> bool:
        """コンテナが実行中かチェック"""
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                check=True
            )
            return self.container_name in result.stdout
        except subprocess.CalledProcessError:
            return False
    
    def get_container_status(self) -> str:
        """コンテナの詳細な状態を取得"""
        try:
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name={self.container_name}", "--format", "{{.Status}}"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return "Unknown"
    
    def stop_container(self) -> bool:
        """Dockerコンテナを停止"""
        if not self.is_container_running():
            logger.info(f"Container {self.container_name} is not running")
            return True
            
        try:
            logger.info(f"Stopping Docker container: {self.container_name}")
            subprocess.run(
                ["docker", "stop", self.container_name],
                check=True,
                capture_output=True,
                text=True
            )
            
            # 停止を待つ
            max_wait = 30
            for i in range(max_wait):
                if not self.is_container_running():
                    logger.info(f"Container {self.container_name} stopped successfully")
                    return True
                time.sleep(1)
                
            logger.warning(f"Container {self.container_name} did not stop within {max_wait} seconds")
            return False
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to stop container: {e}")
            return False
    
    def remove_container_if_exists(self) -> bool:
        """コンテナが存在する場合は削除"""
        try:
            # コンテナが存在するかチェック
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                check=True
            )
            
            if self.container_name in result.stdout:
                logger.info(f"Removing existing container: {self.container_name}")
                subprocess.run(
                    ["docker", "rm", "-f", self.container_name],
                    check=True,
                    capture_output=True,
                    text=True
                )
                time.sleep(2)  # 削除完了を待つ
                return True
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to remove container: {e}")
            return False
    
    def start_container(self, wait_for_ready: bool = True) -> bool:
        """Dockerコンテナを起動"""
        if self.is_container_running():
            logger.info(f"Container {self.container_name} is already running")
            if wait_for_ready:
                return self.wait_for_service_ready()
            return True
            
        try:
            logger.info(f"Starting Docker container with service: {self.service_name}")
            
            # 既存のコンテナを削除（クリーンな状態から開始）
            if not self.remove_container_if_exists():
                logger.warning("Failed to remove existing container, continuing anyway")
            
            # docker compose downでクリーンアップ（スキップ可能）
            logger.info("Trying to clean up with docker compose...")
            try:
                # docker composeまたはdocker-composeを試す
                compose_cmd = None
                for cmd in ["docker", "compose"], ["docker-compose"]:
                    try:
                        subprocess.run(cmd + ["--version"], capture_output=True, check=True)
                        compose_cmd = cmd
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
                
                if compose_cmd:
                    subprocess.run(
                        compose_cmd + ["--profile", "vllm-docker", "down"],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    time.sleep(3)  # クリーンアップ完了を待つ
                else:
                    logger.warning("docker compose not available, skipping cleanup")
            except subprocess.CalledProcessError as e:
                logger.warning(f"docker compose down failed (this is usually OK): {e}")
            
            # docker runで直接起動を試みる
            logger.info("Starting vLLM container with docker run...")
            return self._start_with_docker_run()
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start container: {e}")
            return False
    
    def _start_with_docker_run(self) -> bool:
        """docker runを使用してコンテナを起動（フォールバック）"""
        try:
            logger.info("Attempting to start vLLM container with docker run...")
            
            # 環境変数を取得
            env_vars = []
            
            # HUGGINGFACE_HUB_TOKENは特別扱い（重複を避ける）
            hf_token = None
            
            # まずllm-leaderboardコンテナから取得を試みる
            try:
                result = subprocess.run(
                    ["docker", "exec", "llm-leaderboard", "bash", "-c", "echo $HUGGINGFACE_HUB_TOKEN"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                hf_token = result.stdout.strip()
                if hf_token and hf_token != "$HUGGINGFACE_HUB_TOKEN":
                    logger.info("Hugging Face token retrieved from llm-leaderboard container")
                else:
                    hf_token = None
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to get Hugging Face token from llm-leaderboard container: {e}")
            
            # コンテナから取得できなかった場合は環境変数から取得
            if not hf_token:
                hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
                if hf_token:
                    logger.info("Hugging Face token retrieved from environment variable")
            
            # トークンが取得できた場合のみ追加
            if hf_token:
                env_vars.extend(["-e", f"HUGGINGFACE_HUB_TOKEN={hf_token}"])
                # vLLM v0.5.5ではHF_TOKENも必要な場合がある
                env_vars.extend(["-e", f"HF_TOKEN={hf_token}"])
            else:
                logger.error("No Hugging Face token found! Model download may fail.")
            
            # その他の環境変数を追加
            for key in ["LOCAL_MODEL_PATH", "EVAL_CONFIG_PATH", "VLLM_PORT"]:
                value = os.environ.get(key)
                if value:
                    env_vars.extend(["-e", f"{key}={value}"])
            
            # モデル名を環境変数から取得、なければWandbConfigSingletonから取得
            model_name = os.environ.get("VLLM_MODEL_NAME")
            if not model_name:
                try:
                    from config_singleton import WandbConfigSingleton
                    instance = WandbConfigSingleton.get_instance()
                    if instance and instance.config:
                        model_name = instance.config.model.pretrained_model_name_or_path
                        logger.info(f"Using model from config: {model_name}")
                    else:
                        logger.warning("WandbConfigSingleton not initialized, using default model")
                        model_name = "tokyotech-llm/Swallow-7b-instruct-v0.1"
                except Exception as e:
                    logger.warning(f"Failed to get model from config: {e}")
                    model_name = "tokyotech-llm/Swallow-7b-instruct-v0.1"
            else:
                logger.info(f"Using model from environment variable: {model_name}")
            
            # 絶対パスを取得
            import pathlib
            # llm-leaderboardコンテナ内から実行されている場合は、コンテナ内のパスを使用
            if os.path.exists("/workspace"):
                config_path = "/workspace/configs"
                cache_path = "/root/.cache/huggingface"
            else:
                # ホストから実行されている場合
                config_path = str(pathlib.Path("./configs").resolve())
                cache_path = str(pathlib.Path("~/.cache/huggingface").expanduser().resolve())
            
            # docker runコマンドを構築
            cmd = [
                "docker", "run", "-d",
                "--name", self.container_name,
                "--network", "llm-stack-network",
                "--gpus", "all",
                "-p", "8000:8000",
                "-v", f"{config_path}:/workspace/configs:ro",
                "-v", f"{cache_path}:/root/.cache/huggingface",
            ]
            
            if os.environ.get("LOCAL_MODEL_PATH"):
                local_model_path = str(pathlib.Path(os.environ['LOCAL_MODEL_PATH']).resolve())
                cmd.extend(["-v", f"{local_model_path}:/workspace/models:ro"])
            
            cmd.extend(env_vars)
            
            # vLLMのバージョンを決定（V100 GPU対応）
            # TODO: GPUタイプを自動検出して適切なバージョンを選択
            vllm_image = "vllm/vllm-openai:v0.5.5"  # V100対応バージョン
            
            cmd.append(vllm_image)
            
            # vLLMの起動コマンドを追加（CMDとして渡される引数のみ）
            cmd.extend([
                "--host", "0.0.0.0",
                "--port", "8000",
                "--model", model_name,
                "--dtype", "half"  # V100ではfloat16を使用
            ])
            
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("docker run completed successfully")
                # wait_for_readyがTrueの場合は、実際にサービスが準備完了するまで待つ
                time.sleep(2)  # コンテナ起動直後の待機
                return self.wait_for_service_ready()
            else:
                logger.error(f"docker run failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start with docker run: {e}")
            return False
    
    def wait_for_service_ready(self, timeout: int = 600) -> bool:
        """サービスが準備完了するまで待機（タイムアウトを10分に延長）"""
        logger.info("Waiting for vLLM service to be ready...")
        start_time = time.time()
        last_log_check = 0
        
        # コンテナが起動するまで待つ
        logger.info("Waiting for container to start...")
        while not self.is_container_running():
            if time.time() - start_time > timeout:
                logger.error("Container failed to start within timeout")
                status = self.get_container_status()
                logger.error(f"Container status: {status}")
                return False
            time.sleep(2)
        
        # コンテナの起動直後の状態を確認
        time.sleep(5)
        if not self.is_container_running():
            logger.error("Container stopped immediately after starting")
            # ログを出力
            try:
                result = subprocess.run(
                    ["docker", "logs", self.container_name, "--tail", "50"],
                    capture_output=True,
                    text=True
                )
                logger.error(f"Container logs:\n{result.stdout}\n{result.stderr}")
            except subprocess.CalledProcessError:
                pass
            return False
        
        logger.info("Container started, waiting for service initialization...")
        # 少し待ってからヘルスチェック開始
        time.sleep(10)
        
        # ヘルスチェック
        while time.time() - start_time < timeout:
            elapsed = int(time.time() - start_time)
            
            # 30秒ごとにログを確認してダウンロード進捗を表示
            if elapsed - last_log_check >= 30:
                last_log_check = elapsed
                try:
                    result = subprocess.run(
                        ["docker", "logs", self.container_name, "--tail", "10"],
                        capture_output=True,
                        text=True
                    )
                    if result.stdout:
                        latest_logs = result.stdout.strip().split('\n')
                        for log in latest_logs:
                            if "Downloading" in log or "Loading" in log or "model" in log.lower():
                                logger.info(f"Progress ({elapsed}s): {log[:100]}...")
                except subprocess.CalledProcessError:
                    pass
            
            try:
                # まずヘルスエンドポイントをチェック
                logger.debug(f"Checking health endpoint... ({elapsed}s elapsed)")
                response = requests.get(self.health_check_url, timeout=10)
                if response.status_code == 200:
                    logger.debug("Health endpoint OK, checking models...")
                    # 次にモデルがロードされているか確認
                    models_response = requests.get(self.models_check_url, timeout=10)
                    if models_response.status_code == 200:
                        models_data = models_response.json()
                        if models_data.get("data", []):
                            logger.info(f"vLLM service is ready with loaded models (took {elapsed}s)")
                            return True
                        else:
                            logger.debug("vLLM service is healthy but models not loaded yet")
                    else:
                        logger.debug(f"Models endpoint returned {models_response.status_code}")
                else:
                    logger.debug(f"Health endpoint returned {response.status_code}")
            except requests.exceptions.RequestException as e:
                logger.debug(f"Service not ready yet: {e}")
            
            time.sleep(5)
        
        logger.error(f"vLLM service did not become ready within {timeout} seconds")
        # タイムアウト時の詳細なログを出力
        try:
            result = subprocess.run(
                ["docker", "logs", self.container_name, "--tail", "100"],
                capture_output=True,
                text=True
            )
            logger.error(f"Final container logs:\n{result.stdout}")
        except subprocess.CalledProcessError:
            pass
        return False
    
    def restart_container(self) -> bool:
        """コンテナを再起動"""
        logger.info("Restarting vLLM container...")
        if not self.stop_container():
            logger.error("Failed to stop container for restart")
            return False
            
        # GPUメモリが解放されるのを待つ
        time.sleep(5)
        
        if not self.start_container(wait_for_ready=True):
            logger.error("Failed to start container after restart")
            return False
            
        return True


# シングルトンインスタンス
_docker_vllm_manager = None


def get_docker_vllm_manager() -> DockerVLLMManager:
    """DockerVLLMManagerのシングルトンインスタンスを取得"""
    global _docker_vllm_manager
    if _docker_vllm_manager is None:
        _docker_vllm_manager = DockerVLLMManager()
    return _docker_vllm_manager


def stop_vllm_container_if_needed() -> Optional[bool]:
    """必要に応じてvLLMコンテナを停止"""
    instance = WandbConfigSingleton.get_instance()
    cfg = instance.config
    api_type = cfg.api
    
    if api_type in ["vllm", "vllm-docker"]:
        manager = get_docker_vllm_manager()
        return manager.stop_container()
    return None


def start_vllm_container_if_needed(model_name: Optional[str] = None) -> Optional[bool]:
    """必要に応じてvLLMコンテナを起動
    
    Args:
        model_name: 起動するモデル名。指定しない場合は設定から取得
    """
    instance = WandbConfigSingleton.get_instance()
    cfg = instance.config
    api_type = cfg.api
    
    if api_type in ["vllm", "vllm-docker"]:
        manager = get_docker_vllm_manager()
        # モデル名が指定されていない場合は設定から取得
        if model_name is None:
            model_name = cfg.model.pretrained_model_name_or_path
        # 環境変数に設定
        os.environ["VLLM_MODEL_NAME"] = model_name
        return manager.start_container(wait_for_ready=True)
    return None 