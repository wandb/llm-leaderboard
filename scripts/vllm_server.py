from pathlib import Path
from config_singleton import WandbConfigSingleton
import subprocess
import time
import requests
import atexit
import tempfile

from utils import download_tokenizer_config


def start_vllm_server():

    instance = WandbConfigSingleton.get_instance()
    cfg = instance.config
    model_id = cfg.model.pretrained_model_name_or_path
    dtype = cfg.model.dtype
    max_model_len = cfg.model.max_model_len

    def run_vllm_server(model_id, dtype, max_model_len=2048):
        # set tokenizer_config
        tokenizer_config = download_tokenizer_config(repo_id=model_id)
        cfg.update({"tokenizer_config": tokenizer_config})

        # set chat_template
        chat_template_name = cfg.get('chat_template')
        if not isinstance(chat_template_name, str):
            raise ValueError("Chat template is not set in the config file")

        # chat_template from local
        local_chat_template_path = Path(f"chat_templates/{chat_template_name}.jinja")
        if local_chat_template_path.exists():
            with local_chat_template_path.open(encoding="utf-8") as f:
                chat_template = f.read()

        # chat_template from hf
        else:
            chat_template = download_tokenizer_config(repo_id=chat_template_name).get("chat_template")
            if chat_template is None:
                raise ValueError(f"Chat template {chat_template_name} is not found")

        # serve chat_template
        cfg.tokenizer_config.update({"chat_template": chat_template})
        with tempfile.NamedTemporaryFile(suffix='.jinja', encoding="utf-8", mode="w+") as f:
            f.write(chat_template)
            chat_template_path = Path(f.name)

        # サーバーを起動するためのコマンド
        command = [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_id, 
            "--dtype", dtype, 
            "--max-model-len", str(max_model_len),
            "--chat-template", chat_template_path.resolve(),
            "--seed", "42",
            "--disable-log-stats",
            "--disable-log-requests",
        ]

        # subprocessでサーバーをバックグラウンドで実行
        process = subprocess.Popen(command)

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
            time.sleep(5)  # 5秒待機してから再試行

    # サーバーを起動
    server_process = run_vllm_server(model_id, dtype, max_model_len)
    print("vLLM server is starting...")

    # スクリプト終了時にサーバーを終了する
    def cleanup():
        print("Terminating vLLM server...")
        server_process.terminate()
        server_process.wait()

    atexit.register(cleanup)

    # サーバーが完全に起動するのを待つ
    health_check()
