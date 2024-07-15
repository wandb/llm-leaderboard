import os
from dataclasses import dataclass
import json
from config_singleton import WandbConfigSingleton
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from langchain_anthropic import ChatAnthropic
import boto3
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

@dataclass
class BedrockResponse:
    content: str

class ChatBedrock:
    def __init__(self, cfg) -> None:
        self.bedrock_runtime = boto3.client(service_name="bedrock-runtime")
        self.model_id = cfg.model.pretrained_model_name_or_path
        self.ignore_keys = ["max_tokens"]
        self.generator_config = {
            k: v for k, v in cfg.generator.items() if not k in self.ignore_keys
        }

    def _invoke(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
    ):
        body_dict = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            **self.generator_config,
        }
        if messages[0]["role"] == "system":
            body_dict.update(
                {"messages": messages[1:], "system": messages[0]["content"]}
            )
        else:
            body_dict.update({"messages": messages})

        response = self.bedrock_runtime.invoke_model(
            body=json.dumps(body_dict), modelId=self.model_id
        )
        response_body = json.loads(response.get("body").read())

        return response_body

    def invoke(self, messages, max_tokens: int):
        response = self._invoke(messages=messages, max_tokens=max_tokens)
        if response["content"]:
            content = response["content"][0]["text"]
        else:
            content = ""

        return BedrockResponse(content=content)

def get_model_path(cfg, run):
    if cfg.model.get("source") == "wandb":
        model_artifact_path = cfg.model.artifacts_path
        if model_artifact_path:
            artifact = run.use_artifact(model_artifact_path, type='model')
            model_dir = Path(artifact.download())
            return str(model_dir)  # モデルディレクトリのパスを返す
    else:  # デフォルトはHugging Face
        repo_id = cfg.model.get("repo_id") or cfg.model.pretrained_model_name_or_path
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        try:
            return snapshot_download(repo_id=repo_id, cache_dir=cache_dir)
        except Exception as e:
            print(f"Error downloading from Hugging Face: {e}")
            print("Falling back to pretrained_model_name_or_path")
            return cfg.model.pretrained_model_name_or_path

def get_llm_inference_engine():
    instance = WandbConfigSingleton.get_instance()
    cfg = instance.config
    run = instance.run
    api_type = cfg.api

    if api_type in ["vllm", "fastchat"]:
        from inference_server import start_inference_server
        model_path = get_model_path(cfg, run)
        start_inference_server(api_type, model_path)

        llm = ChatOpenAI(
            openai_api_key="EMPTY",
            openai_api_base="http://localhost:8000/v1",
            model_name=cfg.model.pretrained_model_name_or_path,
            **cfg.generator,
        )

    elif api_type == "openai":
        llm = ChatOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            model=cfg.model.pretrained_model_name_or_path,
            **cfg.generator,
        )

    elif api_type == "mistral":
        llm = ChatMistralAI(
            model=cfg.model.pretrained_model_name_or_path, 
            api_key=os.environ["MISTRAL_API_KEY"],
            **cfg.generator,
        )

    elif api_type == "google":
        categories = [
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            HarmCategory.HARM_CATEGORY_HARASSMENT,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        ]
        safety_settings = {cat: HarmBlockThreshold.BLOCK_NONE for cat in categories}
        
        llm = ChatGoogleGenerativeAI(
            model=cfg.model.pretrained_model_name_or_path,
            api_key=os.environ["GOOGLE_API_KEY"],
            safety_settings=safety_settings,
            **cfg.generator,
        )

    elif api_type == "amazon_bedrock":
        llm = ChatBedrock(cfg=cfg)

    elif api_type == "anthropic":
        llm = ChatAnthropic(
            model=cfg.model.pretrained_model_name_or_path, 
            api_key=os.environ["ANTHROPIC_API_KEY"],
            **cfg.generator,
        )
    
    elif api_type == "upstage":
        llm = ChatOpenAI(
            api_key=os.environ["UPSTAGE_API_KEY"],
            model=cfg.model.pretrained_model_name_or_path,
            base_url="https://api.upstage.ai/v1/solar",
            **cfg.generator,
        )

    else:
        raise ValueError(f"Unsupported API type: {api_type}")
    
    return llm