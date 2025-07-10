import os
import asyncio
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import json
from config_singleton import WandbConfigSingleton
from omegaconf import OmegaConf, DictConfig
import openai
from openai.types.responses import Response as OpenAIResponse
from openai.types.chat import ChatCompletion as OpenAIChatCompletion
from mistralai import Mistral
import google.generativeai as genai
from anthropic import Anthropic
import cohere
from botocore.exceptions import ClientError
import boto3
from botocore.config import Config
from openai import AzureOpenAI


@dataclass
class LLMResponse:
    content: str
    reasoning_content: str = ""


def filter_params(params, allowed_params):
    """許可されたパラメータのみをフィルタリング"""
    return {k: v for k, v in params.items() if k in allowed_params}


def map_common_params(params, param_mapping):
    """共通パラメータを各プロバイダ固有のパラメータ名にマッピング"""
    mapped_params = {}
    for key, value in params.items():
        mapped_key = param_mapping.get(key, key)
        if isinstance(value, DictConfig):
            value = OmegaConf.to_container(value)
        mapped_params[mapped_key] = value
    return mapped_params


class BaseLLMClient(ABC):
    def invoke(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, **kwargs) -> LLMResponse:
        """同期版のinvoke（後方互換性のため）"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.ainvoke(messages, max_tokens, **kwargs))
        finally:
            loop.close()

    @abstractmethod
    async def ainvoke(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, **kwargs) -> LLMResponse:
        raise NotImplementedError


class ChatBedrock(BaseLLMClient):
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

    async def _invoke_async(self, messages: list[dict[str, str]], max_tokens: int):
        """非同期でBedrockを呼び出し"""
        is_claude = "anthropic" in self.model_id.lower()
        is_llama = "llama" in self.model_id.lower()
        is_nova = "nova" in self.model_id.lower()

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
        elif is_nova:
            for message in messages:
                if isinstance(message['content'], str):
                    message['content'] = [{"text": message['content']}]

            parameter_mapping = {
                "top_p": "topP",
                "max_tokens": "maxTokens",
            }

            inference_config = {
                parameter_mapping.get(k, k): v for k, v in {
                    "temperature": self.generator_config.get("temperature", 0.0),
                    "maxTokens": max_tokens,
                    **{k: v for k, v in self.generator_config.items() if k not in ["temperature"]}
                }.items()
            }

            body_dict = {
                "messages": messages,
                "inferenceConfig": inference_config
            }
        else:
            raise ValueError(f"Unsupported model: {self.model_id}")

        try:
            if is_nova:
                # 非同期でconverseを実行
                response = await asyncio.to_thread(
                    self.bedrock_runtime.converse,
                    modelId=self.model_id,
                    **body_dict
                )
                response_body = response
            else:
                # 非同期でinvoke_modelを実行
                response = await asyncio.to_thread(
                    self.bedrock_runtime.invoke_model,
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

    def invoke(self, messages, max_tokens=None, **kwargs):
        """同期版のinvoke（後方互換性のため）"""
        if max_tokens is None:
            max_tokens = 1024
        return super().invoke(messages, max_tokens, **kwargs)

    async def ainvoke(self, messages, max_tokens=None, **kwargs):
        """非同期版のinvoke"""
        if max_tokens is None:
            max_tokens = 1024
        
        response = await self._invoke_async(messages=messages, max_tokens=max_tokens)
        
        if "anthropic" in self.model_id.lower():
            content = response.get("content", [{"text": ""}])[0]["text"]
        elif "llama" in self.model_id.lower():
            content = response.get("generation", "")
            content = content.replace("<|start_header_id|>assistant<|end_header_id|>\n", "").replace("\n<|eot_id|>", "") 
        elif "nova" in self.model_id.lower():
            content = response.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")
        else:
            content = ""
        return LLMResponse(content=content)


class OpenAIClient:
    def __init__(self, api_key=None, base_url=None, model=None, **kwargs):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.async_client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.kwargs = kwargs
        
        self.allowed_params = {
            'temperature', 'top_p', 'n', 'stream', 'stop', 
            'max_tokens', 'presence_penalty', 'frequency_penalty',
            'logit_bias', 'user', 'response_format', 'seed',
            'tools', 'tool_choice', 'parallel_tool_calls', 'extra_body',
        }
        
        self.param_mapping = {}

    async def ainvoke(self, messages, max_tokens=None, **kwargs):
        """非同期版のinvoke"""
        all_kwargs = {**self.kwargs, **kwargs}
        mapped_params = map_common_params(all_kwargs, self.param_mapping)
        filtered_params = filter_params(mapped_params, self.allowed_params)
        
        params = {
            "model": self.model,
            "messages": messages,
            **filtered_params
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        response: OpenAIChatCompletion = await self.async_client.chat.completions.create(**params)
        content = response.choices[0].message.content
        # vLLM reasoning parser output
        # https://docs.vllm.ai/en/latest/features/reasoning_outputs.html#quickstart
        reasoning_content = getattr(response.choices[0].message, 'reasoning_content', '')
        return LLMResponse(content=content, reasoning_content=reasoning_content)


class OpenAIResponsesClient(BaseLLMClient):
    """
    OpenAIのResponses APIを使用するクライアント(Reasoning対応)
    """
    def __init__(self, api_key=None, base_url=None, model=None, **kwargs):
        self.async_client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.kwargs = kwargs
        
        self.allowed_params = {
            'instructions', 'max_output_tokens', 'max_tool_calls', 'metadata',
            'parallel_tool_calls', 'previous_response_id', 'reasoning', 
            'service_tier', 'store', 'stream', 'temperature', 'text',
            'tool_choice', 'tools', 'top_logprobs', 'top_p', 'truncation', 'user',
        }
        
        self.param_mapping = {'max_tokens': 'max_output_tokens', 'top_k': 'top_logprobs'}

    async def ainvoke(self, messages, max_tokens=None, **kwargs):
        """非同期版のinvoke"""
        all_kwargs = {**self.kwargs, **kwargs}
        mapped_params = map_common_params(all_kwargs, self.param_mapping)
        filtered_params = filter_params(mapped_params, self.allowed_params)
        
        params = {
            "model": self.model,
            "input": messages,
            **filtered_params
        }
        
        if max_tokens:
            params["max_output_tokens"] = max_tokens
        
        response: OpenAIResponse = await self.async_client.responses.create(**params)
        content, reasoning_content = "", ""
        for output in response.output:
            if output.type == "message":
                content = output.content[0].text
            elif output.type == "reasoning" and len(output.summary) > 0: # reasoning contentは長さ0の場合がある
                reasoning_content = output.summary[0].text
        return LLMResponse(content=content, reasoning_content=reasoning_content)


class MistralClient(BaseLLMClient):
    def __init__(self, api_key, model, **kwargs):
        self.client = Mistral(api_key=api_key)
        self.model = model
        self.kwargs = kwargs
        
        self.allowed_params = {
            'temperature', 'top_p', 'max_tokens', 'stream', 'stop',
            'random_seed', 'response_format', 'tools', 'tool_choice'
        }
        
        self.param_mapping = {
            'seed': 'random_seed',
        }

    async def ainvoke(self, messages, max_tokens=None, **kwargs):
        """非同期版のinvoke"""
        all_kwargs = {**self.kwargs, **kwargs}
        mapped_params = map_common_params(all_kwargs, self.param_mapping)
        filtered_params = filter_params(mapped_params, self.allowed_params)
        
        params = {
            "model": self.model,
            "messages": messages,
            **filtered_params
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        # Mistral APIを非同期で実行
        response = await asyncio.to_thread(self.client.chat.complete, **params)
        return LLMResponse(content=response.choices[0].message.content)


class GoogleClient(BaseLLMClient):
    def __init__(self, api_key, model, **kwargs):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.kwargs = kwargs
        
        self.allowed_params = {
            'temperature', 'top_p', 'top_k', 'max_output_tokens',
            'candidate_count', 'stop_sequences'
        }
        
        self.param_mapping = {
            'max_tokens': 'max_output_tokens',
            'stop': 'stop_sequences',
        }

    async def ainvoke(self, messages, max_tokens=None, **kwargs):
        """非同期版のinvoke"""
        # メッセージ処理
        chat_history = []
        system_instruction = None
        
        for i, msg in enumerate(messages):
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                if i < len(messages) - 1:
                    chat_history.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                chat_history.append({"role": "model", "parts": [msg["content"]]})
        
        if system_instruction:
            model = genai.GenerativeModel(
                model_name=self.model.model_name,
                system_instruction=system_instruction
            )
        else:
            model = self.model
        
        chat = model.start_chat(history=chat_history)
        
        last_user_message = None
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break
        
        if not last_user_message:
            raise ValueError("No user message found")
        
        all_kwargs = {**self.kwargs, **kwargs}
        mapped_params = map_common_params(all_kwargs, self.param_mapping)
        generation_config = filter_params(mapped_params, self.allowed_params)
        
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens
        
        # Google APIを非同期で実行
        response = await asyncio.to_thread(
            chat.send_message, 
            last_user_message, 
            generation_config=generation_config
        )
        return LLMResponse(content=response.text)


class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key, model, **kwargs):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.kwargs = kwargs
        
        self.allowed_params = {
            'temperature', 'top_p', 'top_k', 'max_tokens',
            'stop_sequences', 'stream'
        }
        
        self.param_mapping = {
            'stop': 'stop_sequences',
        }

    async def ainvoke(self, messages, max_tokens=None, **kwargs):
        """非同期版のinvoke"""
        system_message = None
        filtered_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                filtered_messages.append(msg)
        
        all_kwargs = {**self.kwargs, **kwargs}
        mapped_params = map_common_params(all_kwargs, self.param_mapping)
        filtered_params = filter_params(mapped_params, self.allowed_params)
        
        params = {
            "model": self.model,
            "messages": filtered_messages,
            **filtered_params
        }
        
        if system_message:
            params["system"] = system_message
            
        if max_tokens:
            params["max_tokens"] = max_tokens
        elif "max_tokens" not in params:
            params["max_tokens"] = 1024
        
        # Anthropic APIを非同期で実行
        response = await asyncio.to_thread(self.client.messages.create, **params)
        return LLMResponse(content=response.content[0].text)


class AzureOpenAIClient(BaseLLMClient):
    def __init__(self, api_key, azure_endpoint, azure_deployment, api_version, **kwargs):
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version
        )
        self.async_client = openai.AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version
        )
        self.azure_deployment = azure_deployment
        self.kwargs = kwargs
        
        self.allowed_params = {
            'temperature', 'top_p', 'n', 'stream', 'stop', 
            'max_tokens', 'presence_penalty', 'frequency_penalty',
            'logit_bias', 'user', 'response_format', 'seed',
            'tools', 'tool_choice', 'parallel_tool_calls'
        }
        
        self.param_mapping = {}

    async def ainvoke(self, messages, max_tokens=None, **kwargs):
        """非同期版のinvoke"""
        all_kwargs = {**self.kwargs, **kwargs}
        mapped_params = map_common_params(all_kwargs, self.param_mapping)
        filtered_params = filter_params(mapped_params, self.allowed_params)
        
        params = {
            "model": self.azure_deployment,
            "messages": messages,
            **filtered_params
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        response = await self.async_client.chat.completions.create(**params)
        return LLMResponse(content=response.choices[0].message.content)


class CohereClient(BaseLLMClient):
    def __init__(self, api_key, model, **kwargs):
        self.client = cohere.Client(api_key)
        self.model = model
        self.kwargs = kwargs
        
        self.allowed_params = {
            'temperature', 'p', 'k', 'max_tokens', 'stop_sequences',
            'frequency_penalty', 'presence_penalty', 'seed'
        }
        
        self.param_mapping = {
            'top_p': 'p',
            'top_k': 'k',
            'stop': 'stop_sequences',
        }

    async def ainvoke(self, messages, max_tokens=None, **kwargs):
        """非同期版のinvoke"""
        chat_history = []
        message = ""
        preamble = None
        
        for i, msg in enumerate(messages):
            if msg["role"] == "system":
                preamble = msg["content"]
            elif msg["role"] == "user":
                if i == len(messages) - 1:
                    message = msg["content"]
                else:
                    chat_history.append({"role": "USER", "message": msg["content"]})
            elif msg["role"] == "assistant":
                chat_history.append({"role": "CHATBOT", "message": msg["content"]})
        
        all_kwargs = {**self.kwargs, **kwargs}
        mapped_params = map_common_params(all_kwargs, self.param_mapping)
        filtered_params = filter_params(mapped_params, self.allowed_params)
        
        params = {
            "model": self.model,
            "message": message,
            "chat_history": chat_history,
            **filtered_params
        }
        
        if preamble:
            params["preamble"] = preamble
            
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        # Cohere APIを非同期で実行
        response = await asyncio.to_thread(self.client.chat, **params)
        return LLMResponse(content=response.text)


def get_llm_inference_engine() -> BaseLLMClient:
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

        llm = OpenAIClient(
            api_key="EMPTY",
            base_url=base_url,
            model_name=model_name,
            **cfg.generator,
        )

    elif api_type == "vllm-external":
        # vLLMサーバーは起動済みのものを用いるので、ここでは起動しない
        #from vllm_server import start_vllm_server
        #start_vllm_server()

        base_url = cfg.get("base_url", "http://localhost:8000/v1") #"http://localhost:8000/v1"
        model_name = cfg.model.pretrained_model_name_or_path

        llm = OpenAIClient(
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            base_url=base_url,
            model_name=model_name,
            **cfg.generator,
        )

    elif api_type == "openai_responses":
        llm = OpenAIResponsesClient(
            api_key=os.environ["OPENAI_API_KEY"],
            model=cfg.model.pretrained_model_name_or_path,
            **cfg.generator,
        )

    elif api_type in ["openai", "openai_chat"]: # "openai" は後方互換性のため
        llm = OpenAIClient(
            api_key=os.environ["OPENAI_API_KEY"],
            model=cfg.model.pretrained_model_name_or_path,
            **cfg.generator,
        )

    elif api_type == "xai":
        llm = OpenAIClient(
            api_key=os.environ["XAI_API_KEY"],
            base_url="https://api.x.ai/v1",
            model=cfg.model.pretrained_model_name_or_path,
            **cfg.generator,
        )

    elif api_type == "mistral":
        llm = MistralClient(
            api_key=os.environ["MISTRAL_API_KEY"],
            model=cfg.model.pretrained_model_name_or_path,
            **cfg.generator,
        )

    elif api_type == "google":
        llm = GoogleClient(
            api_key=os.environ["GOOGLE_API_KEY"],
            model=cfg.model.pretrained_model_name_or_path,
            **cfg.generator,
        )

    elif api_type == "amazon_bedrock":
        llm = ChatBedrock(cfg=cfg)

    elif api_type == "anthropic":
        llm = AnthropicClient(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model=cfg.model.pretrained_model_name_or_path,
            **cfg.generator,
        )
    
    elif api_type == "upstage":
        llm = OpenAIClient(
            api_key=os.environ["UPSTAGE_API_KEY"],
            base_url="https://api.upstage.ai/v1/solar",
            model=cfg.model.pretrained_model_name_or_path,
            **cfg.generator,
        )

    elif api_type == "azure-openai":
        llm = AzureOpenAIClient(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=cfg.model.pretrained_model_name_or_path,
            api_version=cfg.model.get("api_version", "2024-05-01-preview"),
            **cfg.generator,
        )

    elif api_type == "cohere":
        llm = CohereClient(
            api_key=os.environ["COHERE_API_KEY"],
            model=cfg.model.pretrained_model_name_or_path,
            **cfg.generator,
        )

    else:
        raise ValueError(f"Unsupported API type: {api_type}")
    
    return llm