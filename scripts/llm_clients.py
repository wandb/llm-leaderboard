import os
import asyncio
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any
import uuid
import json
import warnings
import logging
from config_singleton import WandbConfigSingleton
from omegaconf import OmegaConf, DictConfig
import openai
from openai.types.responses import Response as OpenAIResponse, ParsedResponse as OpenAIParsedResponse
from openai.types.chat import ChatCompletion as OpenAIChatCompletion
import httpx
from pydantic import BaseModel


logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    type: str = 'function'
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class LLMResponse:
    content: str
    reasoning_content: str = ""
    parsed_output: Optional[Any] = None
    tool_calls: Optional[List[ToolCall]] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


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


class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key=None, base_url=None, model=None, **kwargs):
        # タイムアウト設定を改善
        timeout_config = httpx.Timeout(
            connect=30.0,  # 接続タイムアウトを30秒に設定
            read=300.0,    # 読み取りタイムアウトを5分に設定
            write=300.0,   # 書き込みタイムアウトを5分に設定
            pool=30.0      # プールタイムアウトを30秒に設定
        )
        
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_config,
            max_retries=3  # リトライ回数を3回に設定
        )
        self.async_client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_config,
            max_retries=3  # リトライ回数を3回に設定
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

    def invoke(self, messages, max_tokens=None, **kwargs):
        """同期版のinvoke（後方互換性のため）"""
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
        
        # extra_bodyを直接渡す（OpenRouterのprovider、reasoningなどに対応）
        if "extra_body" in all_kwargs:
            # DictConfigを通常の辞書に変換
            try:
                import omegaconf
                if isinstance(all_kwargs["extra_body"], omegaconf.DictConfig):
                    params["extra_body"] = omegaconf.OmegaConf.to_container(all_kwargs["extra_body"], resolve=True)
                else:
                    params["extra_body"] = all_kwargs["extra_body"]
            except (ImportError, AttributeError):
                # omegaconfが利用できない場合やDictConfigでない場合はそのまま使用
                params["extra_body"] = all_kwargs["extra_body"]
        
        response = self.client.chat.completions.create(**params)
        content = response.choices[0].message.content
        if content is None:
            print("Warning: Content is None in sync invoke, converting to empty string")
            content = ""
        # OpenRouter reasoning field support
        reasoning_content = getattr(response.choices[0].message, 'reasoning', '')
        return LLMResponse(content=content, reasoning_content=reasoning_content)

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
        
        # extra_bodyを直接渡す（OpenRouterのprovider、reasoningなどに対応）
        if "extra_body" in all_kwargs:
            # DictConfigを通常の辞書に変換
            try:
                import omegaconf
                if isinstance(all_kwargs["extra_body"], omegaconf.DictConfig):
                    params["extra_body"] = omegaconf.OmegaConf.to_container(all_kwargs["extra_body"], resolve=True)
                else:
                    params["extra_body"] = all_kwargs["extra_body"]
            except (ImportError, AttributeError):
                # omegaconfが利用できない場合やDictConfigでない場合はそのまま使用
                params["extra_body"] = all_kwargs["extra_body"]

        response = await self.async_client.chat.completions.create(**params)
        
        content = response.choices[0].message.content
        if content is None:
            print("Warning: Content is None, converting to empty string")
            content = ""
        # OpenRouter reasoning field support
        reasoning_content = getattr(response.choices[0].message, 'reasoning', '')

        return LLMResponse(
            content=content,
            reasoning_content=reasoning_content,
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=response.usage.completion_tokens if response.usage else None
        )


class OpenAIResponsesClient(BaseLLMClient):
    """
    OpenAIのResponses APIを使用するクライアント(Reasoning対応)
    """
    def __init__(self, api_key=None, base_url=None, model=None, structured=False, **kwargs):
        # タイムアウト設定を改善
        timeout = httpx.Timeout(30.0, connect=10.0)
        self.async_client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=3
        )
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=3
        )
        self.model = model
        self.kwargs = kwargs
        self.structured = structured
        
        self.allowed_params = {
            'instructions', 'max_output_tokens', 'max_tool_calls', 'metadata',
            'parallel_tool_calls', 'previous_response_id', 'reasoning', 
            'service_tier', 'store', 'stream', 'text', 'text_format',
            'tool_choice', 'tools', 'top_logprobs', 'truncation', 'user',
        }

        self.param_mapping = {'max_tokens': 'max_output_tokens', 'top_k': 'top_logprobs'}

    def invoke(self, messages, max_tokens=None, **kwargs):
        """同期版のinvoke（後方互換性のため）"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.ainvoke(messages, max_tokens, **kwargs))
        finally:
            loop.close()

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
            params["max_output_tokens"] = max(max_tokens, 16)
        
        if 'text_format' in filtered_params:
            response: OpenAIParsedResponse = await self.async_client.responses.parse(**params)
            parsed_output = response.output_parsed
        else:
            response: OpenAIResponse = await self.async_client.responses.create(**params)
            parsed_output = None

        content, reasoning_content = "", ""
        for output in response.output:
            if output.type == "message":
                content = output.content[0].text
            elif output.type == "reasoning" and len(output.summary) > 0: # reasoning contentは長さ0の場合がある
                reasoning_content = output.summary[0].text
        return LLMResponse(content=content, reasoning_content=reasoning_content, parsed_output=parsed_output)


def get_llm_inference_engine() -> BaseLLMClient:
    """LLMクライアントを取得する"""
    instance = WandbConfigSingleton.get_instance()
    cfg = instance.config
    api_type = cfg.api

    if api_type == "openai_responses":
        llm = OpenAIResponsesClient(
            api_key=os.environ["OPENAI_API_KEY"],
            model=cfg.model.pretrained_model_name_or_path,
            **cfg.generator,
        )
        return llm
    
    elif api_type == "openai" or api_type == "openai_chat":
        llm = OpenAIClient(
            api_key=os.environ["OPENAI_API_KEY"],
            model=cfg.model.pretrained_model_name_or_path,
            **cfg.generator,
        )
        return llm
    
    elif api_type == "vllm-external" or api_type == "openai-compatible":
        # 外部のOpenAI互換APIに接続（OpenRouterを含む）
        base_url = cfg.get("base_url", "http://localhost:8000/v1")
        model_name = cfg.model.pretrained_model_name_or_path

        llm = OpenAIClient(
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            base_url=base_url,
            model=model_name,
            **cfg.generator,
        )
        return llm
    else:
        raise ValueError(f"Unsupported API type: {api_type}")