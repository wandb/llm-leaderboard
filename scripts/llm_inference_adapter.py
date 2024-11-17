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
# from langchain_aws import ChatBedrock
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from botocore.exceptions import ClientError
import boto3
from botocore.config import Config


import json
import boto3
from dataclasses import dataclass

@dataclass
class BedrockResponse:
    content: str


class ChatBedrock:
    def __init__(self, cfg) -> None:
        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"),
            config=Config(read_timeout=1000),
        )
        self.model_id = cfg.model.pretrained_model_name_or_path
        self.ignore_keys = ["max_tokens"]
        self.generator_config = {
            k: v for k, v in cfg.generator.items() if k not in self.ignore_keys
        }

    def _invoke(self, messages: list[dict[str, str]], max_tokens: int):
        # Determine the model type (Anthropic Claude or Meta Llama 3)
        is_claude = "anthropic" in self.model_id.lower()
        is_llama = "llama" in self.model_id.lower()

        if is_claude:
            body_dict = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                **self.generator_config,
            }
            if messages[0]["role"] == "system":
                body_dict.update({"messages": messages[1:], "system": messages[0]["content"]})
            else:
                body_dict.update({"messages": messages})
        elif is_llama:
            prompt = self._format_llama_prompt(messages)
            body_dict = {
                "prompt": prompt,
                "max_gen_len": max_tokens,
                **self.generator_config,
            }
        else:
            raise ValueError(f"Unsupported model: {self.model_id}")

        try:
            response = self.bedrock_runtime.invoke_model(
                body=json.dumps(body_dict),
                modelId=self.model_id
            )
            response_body = json.loads(response.get("body").read())
        except ClientError as e:
            print(f"ERROR: Can't invoke '{self.model_id}'. Reason: {e}")
            raise

        return response_body

    def _format_llama_prompt(self, messages):
        formatted_messages = ["<|begin_of_text|>"]
        for message in messages:
            if message["role"] == "system":
                formatted_messages.append(f"<|start_header_id|>system<|end_header_id|>\n{message['content']}\n<|eot_id|>")
            elif message["role"] == "user":
                formatted_messages.append(f"<|start_header_id|>user<|end_header_id|>\n{message['content']}\n<|eot_id|>")
            elif message["role"] == "assistant":
                formatted_messages.append(f"<|start_header_id|>assistant<|end_header_id|>\n{message['content']}\n<|eot_id|>")
        return "".join(formatted_messages)

    def invoke(self, messages, max_tokens: int):
        response = self._invoke(messages=messages, max_tokens=max_tokens)
        if "anthropic" in self.model_id.lower():
            content = response.get("content", [{"text": ""}])[0]["text"]
        elif "llama" in self.model_id.lower():
            content = response.get("generation", "")
            # ヘッダーとフッターを除去
            content = content.replace("<|start_header_id|>assistant<|end_header_id|>\n", "").replace("\n<|eot_id|>", "") 
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

        # LoRAの設定を確認
        lora_config = cfg.model.get("lora", None)
        base_url = "http://localhost:8000/v1"
        model_name = cfg.model.pretrained_model_name_or_path

        if lora_config and lora_config.get("enable", False):
            # LoRAが有効な場合、LoRAアダプター名をモデル名として使用
            model_name = lora_config.get("adapter_name", model_name)

        # LangChainのVLLMインテグレーションを使用
        llm = ChatOpenAI(
            openai_api_key="EMPTY",
            openai_api_base=base_url,
            model_name=model_name,
            **cfg.generator,
        )

    elif api_type == "openai":
        # LangChainのOpenAIインテグレーションを使用
        llm = ChatOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            model=cfg.model.pretrained_model_name_or_path,
            **cfg.generator,
        )

    elif api_type == "xai":
        # LangChainのOpenAIインテグレーションを使用
        llm = ChatOpenAI(
            base_url="https://api.x.ai/v1",
            api_key=os.environ["XAI_API_KEY"],
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

    elif api_type == "azure-openai":
        # LangChainのAzure OpenAIインテグレーションを使用
        llm = AzureChatOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=cfg.model.pretrained_model_name_or_path,
            api_version=cfg.model.get("api_version", "2024-05-01-preview"),
            **cfg.generator,
        )

    elif api_type == "cohere":
        llm = ChatCohere(
            model=cfg.model.pretrained_model_name_or_path,
            cohere_api_key=os.environ["COHERE_API_KEY"],
            **cfg.generator,
        )

    else:
        raise ValueError(f"Unsupported API type: {api_type}")
    
    return llm
