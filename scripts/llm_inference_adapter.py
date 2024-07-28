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
from transformers import AutoModel, AutoTokenizer
import torch
# from langchain_aws import ChatBedrock
from langchain_anthropic import ChatAnthropic
from botocore.exceptions import ClientError
import boto3

# from langchain_cohere import Cohere


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
        # create body
        body_dict = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            **self.generator_config,
        }
        # handle system message
        if messages[0]["role"] == "system":
            body_dict.update(
                {"messages": messages[1:], "system": messages[0]["content"]}
            )
        else:
            body_dict.update({"messages": messages})

        # inference
        response = self.bedrock_runtime.invoke_model(
            body=json.dumps(body_dict), modelId=self.model_id
        )
        response_body = json.loads(response.get("body").read())

        return response_body

    def invoke(self, messages, max_tokens: int):
        response = self._invoke(messages=messages, max_tokens=max_tokens)
        if response["content"]:
            content = content = response["content"][0]["text"]
        else:
            content = ""

        return BedrockResponse(content=content)


def get_llm_inference_engine():
    instance = WandbConfigSingleton.get_instance()
    cfg = instance.config
    api_type = cfg.api

    if api_type == "vllm":
        # vLLMサーバーを起動
        from vllm_server import start_vllm_server
        start_vllm_server()

        #model = AutoModel.from_pretrained(model_name)
        #num_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
        #wandb.config.update({"model.size": num_params})

        # LangChainのVLLMインテグレーションを使用
        llm = ChatOpenAI(
            openai_api_key="EMPTY",
            openai_api_base="http://localhost:8000/v1",
            model_name=cfg.model.pretrained_model_name_or_path,
            **cfg.generator,
        )

    elif api_type == "openai":
        # LangChainのOpenAIインテグレーションを使用
        llm = ChatOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            model=cfg.model.pretrained_model_name_or_path,
            **cfg.generator,
        )

    elif api_type == "mistral":
        # LangChainのMistralAIインテグレーションを使用
        llm = ChatMistralAI(
            model=cfg.model.pretrained_model_name_or_path, 
            api_key=os.environ["MISTRAL_API_KEY"],
            **cfg.generator,
        )

    elif api_type == "google":
        # LangChainのGoogleGenerativeAIインテグレーションを使用
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
        # LangChainのBedrockインテグレーションを使用
        # llm = ChatBedrock(
        #     region_name=os.environ["AWS_DEFAULT_REGION"],
        #     model_id=cfg.model.pretrained_model_name_or_path,
        #     model_kwargs=cfg.generator,
        # )

    elif api_type == "anthropic":
        # LangChainのAnthropicインテグレーションを使用
        llm = ChatAnthropic(
            model=cfg.model.pretrained_model_name_or_path, 
            api_key=os.environ["ANTHROPIC_API_KEY"],
            **cfg.generator,
        )
    
    elif api_type == "upstage":
        # LangChainのOpenAIインテグレーションを使用
        llm = ChatOpenAI(
            api_key=os.environ["UPSTAGE_API_KEY"],
            model=cfg.model.pretrained_model_name_or_path,
            base_url="https://api.upstage.ai/v1/solar",
            **cfg.generator,
        )

    # elif api_type == "azure-openai":
    #     llm = AzureChatOpenAI(
    #         api_key=os.environ["OPENAI_API_KEY"],
    #         # api_base=os.environ["OPENAI_API_BASE"],
    #         # api_version=os.environ["OPENAI_API_VERSION"]
    #         api_version="2024-05-01-preview",
    #         model=cfg.model.pretrained_model_name_or_path,
    #         **cfg.generator,
    #     )

    # elif api_type == "cohere":
    #     llm = Cohere(
    #         model=cfg.model.pretrained_model_name_or_path,
    #         cohere_api_key=os.environ["COHERE_API_KEY"],
    #         **cfg.generator,
    #     )

    else:
        raise ValueError(f"Unsupported API type: {api_type}")
    
    return llm
