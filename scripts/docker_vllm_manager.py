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
        # Docker networkでサービス名を使用
        if os.path.exists("/workspace"):
            # コンテナ内から実行されている場合
            # vLLMコンテナ名で通信（docker runで起動した場合）
            self.health_check_url = "http://llm-stack-vllm-1:8000/health"
            self.models_check_url = "http://llm-stack-vllm-1:8000/v1/models"
        else:
            # ホストから実行されている場合
            self.health_check_url = "http://localhost:8000/health"
            self.models_check_url = "http://localhost:8000/v1/models"
        
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
    
    def container_exists(self) -> bool:
        """コンテナの存在チェック（実行中・停止中を問わず）"""
        try:
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                check=True
            )
            return self.container_name in result.stdout
        except subprocess.CalledProcessError:
            return False

    def start_container(self, wait_for_ready: bool = True) -> bool:
        """Dockerコンテナを起動（compose優先）。既に存在する場合は docker start を使用"""
        # すでに実行中ならそのまま
        if self.is_container_running():
            logger.info(f"Container {self.container_name} is already running")
            return self.wait_for_service_ready() if wait_for_ready else True

        # 停止状態で存在していれば docker start
        if self.container_exists():
            logger.info(f"Container {self.container_name} exists (stopped). Starting it...")
            try:
                subprocess.run(["docker", "start", self.container_name], check=True)
                return self.wait_for_service_ready() if wait_for_ready else True
            except subprocess.CalledProcessError as e:
                logger.warning(f"docker start failed: {e}. Will attempt fresh start via compose.")

        # compose でサービス名が登録されている場合は up
        try:
            logger.info("Starting vLLM container via docker compose ...")
            subprocess.run([
                "docker", "compose", "--profile", "vllm-docker", "up", "-d", self.container_name
            ], check=True)
            return self.wait_for_service_ready() if wait_for_ready else True
        except subprocess.CalledProcessError as e:
            logger.warning(f"docker compose up failed: {e}. Falling back to docker run …")

        # 最後のフォールバック（従来の docker run）
        return self._start_with_docker_run()
    
    def _start_with_docker_run(self) -> bool:
        """docker runを使用してコンテナを起動（フォールバック）"""
        try:
            logger.info("Attempting to start vLLM container with docker run...")
            
            # 環境変数を取得
            env_vars = []
            
            # HUGGINGFACE_HUB_TOKENは特別扱い（重複を避ける）
            hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
            if not hf_token:
                hf_token = os.environ.get("HF_TOKEN") # Fallback for alternative name

            # トークンが取得できた場合のみ追加
            if hf_token:
                env_vars.extend(["-e", f"HUGGINGFACE_HUB_TOKEN={hf_token}"])
                # vLLM v0.5.5ではHF_TOKENも必要な場合がある
                env_vars.extend(["-e", f"HF_TOKEN={hf_token}"])
                logger.info("Hugging Face token retrieved from environment variables.")
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
            inside_container = os.path.exists("/workspace")

            cmd = [
                "docker", "run", "-d",
                "--name", self.container_name,
                "--network", "llm-stack-network",
                "--gpus", "all",
                "-p", "8000:8000",
            ]

            if not inside_container:
                # ホストから実行している場合のみ、ホストパスをマウント
                config_path = str(pathlib.Path("./configs").resolve())
                cache_path = str(pathlib.Path("~/.cache/huggingface").expanduser().resolve())
                cmd.extend(["-v", f"{config_path}:/workspace/configs:ro"])
                cmd.extend(["-v", f"{cache_path}:/root/.cache/huggingface"])
            else:
                # コンテナ内から実行している場合は、ボリュームソースパスの不一致を避けるためマウントをスキップ
                logger.warning("Running inside a container; skipping host config mount")

                # ただし、キャッシュを再利用するためにホスト側 HuggingFace キャッシュを指定された場合はマウントする
                host_cache_path = os.environ.get("HOST_HF_CACHE_PATH")
                if host_cache_path:
                    logger.info(f"Mounting host HuggingFace cache from {host_cache_path}")
                    cmd.extend(["-v", f"{host_cache_path}:/root/.cache/huggingface"])
                else:
                    logger.warning("HOST_HF_CACHE_PATH not provided; model will be re-downloaded if not cached in image")

            if os.environ.get("LOCAL_MODEL_PATH"):
                local_model_path = str(pathlib.Path(os.environ['LOCAL_MODEL_PATH']).resolve())
                cmd.extend(["-v", f"{local_model_path}:/workspace/models:ro"])
            
            # disable_triton_mma フラグを反映
            try:
                from config_singleton import WandbConfigSingleton
                instance = WandbConfigSingleton.get_instance()
                if instance and instance.config:
                    if instance.config.vllm.get("disable_triton_mma", False):
                        env_vars.extend(["-e", "VLLM_DISABLE_TRITON_MMA=1"])
                        logger.info("VLLM_DISABLE_TRITON_MMA=1 is set due to config")
            except Exception as e:
                logger.warning(f"Failed to get disable_triton_mma from config: {e}")

            cmd.extend(env_vars)
            
            # vLLMのバージョンを決定
            vllm_image = "vllm/vllm-openai:v0.5.5"  # デフォルト
            try:
                from config_singleton import WandbConfigSingleton
                instance = WandbConfigSingleton.get_instance()
                if instance and instance.config:
                    tag = instance.config.vllm.get("vllm_tag")
                    image = instance.config.vllm.get("vllm_docker_image")
                    if image:
                        vllm_image = image
                    elif tag:
                        vllm_image = f"vllm/vllm-openai:{tag}"
            except Exception as e:
                logger.warning(f"Failed to get vLLM image from config: {e}")
            
            cmd.append(vllm_image)
            
            # vLLMの起動コマンドを追加（CMDとして渡される引数のみ）

            # dtypeをYAMLから取得
            dtype = "bfloat16" # デフォルト
            try:
                from config_singleton import WandbConfigSingleton
                instance = WandbConfigSingleton.get_instance()
                if instance and instance.config:
                    dtype = instance.config.vllm.get("dtype", dtype)
            except Exception as e:
                logger.warning(f"Failed to get dtype from config: {e}")

            cmd.extend([
                "--host", "0.0.0.0",
                "--port", "8000",
                "--model", model_name,
                "--dtype", dtype
            ])

            # YAMLからの追加引数を反映
            try:
                from config_singleton import WandbConfigSingleton
                instance = WandbConfigSingleton.get_instance()
                if instance and instance.config:
                    extra_args = instance.config.vllm.get("extra_args", [])
                    if extra_args:
                        # extra_argsがリスト形式であることを確認
                        if isinstance(extra_args, list) or hasattr(extra_args, '__iter__'):
                            cmd.extend(extra_args)
                            logger.info(f"Added extra vLLM args from config: {extra_args}")
                        else:
                            logger.warning(f"vllm.extra_args is not a list, skipping: {extra_args}")
            except Exception as e:
                logger.warning(f"Failed to get extra_args from config: {e}")
            
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
        """コンテナを再起動 (compose 管理優先)"""
        logger.info("Restarting vLLM container…")
        if self.container_exists():
            try:
                subprocess.run(["docker", "stop", self.container_name], check=True)
            except subprocess.CalledProcessError as e:
                logger.warning(f"docker stop failed: {e}")

        # GPUメモリ解放待ち
        time.sleep(5)
        return self.start_container(wait_for_ready=True)


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