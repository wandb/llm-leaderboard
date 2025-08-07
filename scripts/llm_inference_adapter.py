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
from mistralai import Mistral
import google.generativeai as genai
from anthropic import Anthropic
import cohere
from botocore.exceptions import ClientError
import boto3
from botocore.config import Config
from openai import AzureOpenAI
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
    parsed_output: Optional[BaseModel] = None
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


class ChatBedrock(BaseLLMClient):
    def __init__(self, cfg) -> None:
        # 接続プールとタイムアウト設定を改善
        config = Config(
            read_timeout=1000,
            connect_timeout=60,
            max_pool_connections=50,  # 接続プールサイズを増加
        )
        
        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-west-1"),
            config=config,
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
        except Exception as e:
            print(f"ERROR: Unexpected error during invocation of '{self.model_id}'. Reason: {e}")
            # エラーの場合は空のレスポンスを返す
            return {"content": []}

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
            content_list = response.get("content", [])
            if content_list and len(content_list) > 0:
                content = content_list[0].get("text", "")
            else:
                content = ""
        elif "llama" in self.model_id.lower():
            content = response.get("generation", "")
            content = content.replace("<|start_header_id|>assistant<|end_header_id|>\n", "").replace("\n<|eot_id|>", "") 
        elif "nova" in self.model_id.lower():
            content_list = response.get("output", {}).get("message", {}).get("content", [])
            if content_list and len(content_list) > 0:
                content = content_list[0].get("text", "")
            else:
                content = ""
        else:
            content = ""
        
        return LLMResponse(content=content)


class OpenAIClient:
    def __init__(self, api_key=None, base_url=None, model=None, **kwargs):
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
        
        self.allowed_params = {
            'frequency_penalty', 'logit_bias', 'logprobs', 'max_tokens',
            'n', 'presence_penalty', 'response_format', 'seed', 'stop',
            'stream', 'temperature', 'top_p', 'tools', 'tool_choice',
            'user', 'extra_body', 'functions', 'function_call'
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
                    params["extra_body"] = omegaconf.OmegaConf.to_container(all_kwargs["extra_body"])
                else:
                    params["extra_body"] = all_kwargs["extra_body"]
            except (ImportError, AttributeError):
                # omegaconfが利用できない場合やDictConfigでない場合はそのまま使用
                params["extra_body"] = all_kwargs["extra_body"]
        
        # 後方互換性のためのレガシーパラメータサポート
        if "include_reasoning" in all_kwargs:
            if "extra_body" not in params:
                params["extra_body"] = {}
            if all_kwargs["include_reasoning"] is True:
                params["extra_body"]["reasoning"] = {}
            elif all_kwargs["include_reasoning"] is False:
                params["extra_body"]["reasoning"] = {"exclude": True}
        
        response = self.client.chat.completions.create(**params)
        content = response.choices[0].message.content
        # vLLM reasoning parser output
        reasoning_content = getattr(response.choices[0].message, 'reasoning_content', '')
        # OpenRouter reasoning field support
        if not reasoning_content and hasattr(response.choices[0].message, 'reasoning'):
            reasoning_content = getattr(response.choices[0].message, 'reasoning', '')
        return LLMResponse(content=content, reasoning_content=reasoning_content)

    async def ainvoke(self, messages, max_tokens=None, **kwargs):
        """非同期版のinvoke"""
        all_kwargs = {**self.kwargs, **kwargs}
        mapped_params = map_common_params(all_kwargs, self.param_mapping)
        filtered_params = filter_params(mapped_params, self.allowed_params)
        
        # OmegaConfオブジェクトを通常のPythonオブジェクトに変換
        def convert_omegaconf(obj):
            import omegaconf
            if hasattr(obj, '_content') and hasattr(obj, '_metadata'):  # DictConfig
                return omegaconf.OmegaConf.to_container(obj)
            elif hasattr(obj, '_content') and not hasattr(obj, '_metadata'):  # ListConfig
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_omegaconf(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_omegaconf(item) for item in obj]
            else:
                return obj
        
        # すべてのパラメータを変換
        filtered_params = convert_omegaconf(filtered_params)
        
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
                    params["extra_body"] = omegaconf.OmegaConf.to_container(all_kwargs["extra_body"])
                elif isinstance(all_kwargs["extra_body"], omegaconf.ListConfig):
                    params["extra_body"] = list(all_kwargs["extra_body"])
                else:
                    params["extra_body"] = all_kwargs["extra_body"]
            except (ImportError, AttributeError):
                # omegaconfが利用できない場合やDictConfigでない場合はそのまま使用
                params["extra_body"] = all_kwargs["extra_body"]
        
        # 後方互換性のためのレガシーパラメータサポート
        if "include_reasoning" in all_kwargs:
            if "extra_body" not in params:
                params["extra_body"] = {}
            if all_kwargs["include_reasoning"] is True:
                params["extra_body"]["reasoning"] = {}
            elif all_kwargs["include_reasoning"] is False:
                params["extra_body"]["reasoning"] = {"exclude": True}

        # Structured output
        if "response_format" in params:
            response: OpenAIChatCompletion = await self.async_client.beta.chat.completions.parse(**params)
            parsed_output = response.choices[0].message.parsed
        else:
            response: OpenAIChatCompletion = await self.async_client.chat.completions.create(**params)
            parsed_output = None

        content = response.choices[0].message.content
        # vLLM reasoning parser output
        # https://docs.vllm.ai/en/latest/features/reasoning_outputs.html#quickstart
        reasoning_content = getattr(response.choices[0].message, 'reasoning_content', '')
        # OpenRouter reasoning field support
        if not reasoning_content and hasattr(response.choices[0].message, 'reasoning'):
            reasoning_content = getattr(response.choices[0].message, 'reasoning', '')

        llm_response = LLMResponse(
            content=content,
            reasoning_content=reasoning_content,
            parsed_output=parsed_output,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens
        )

        def parse_arguments(arguments: str):
            """
            ArgumentsのJSONをパースする
            OpenRouterなどの一部の互換実装ではValid JSONではなく空文字や最後の}がないケースがあるのでカバーする
            """
            if arguments is None or arguments == "":
                return {}
            try:
                return json.loads(arguments)
            except json.JSONDecodeError as e1:
                try:
                    return json.loads(arguments + '}')
                except json.JSONDecodeError:
                    raise e1

        if tool_calls := response.choices[0].message.tool_calls:
            llm_response.tool_calls = [ToolCall(
                name=tool_call.function.name,
                arguments=parse_arguments(tool_call.function.arguments),
                id=tool_call.id,
                type=tool_call.type
            ) for tool_call in tool_calls]

        return llm_response


class OpenAIResponsesClient(BaseLLMClient):
    """
    OpenAIのResponses APIを使用するクライアント(Reasoning対応)
    """
    def __init__(self, api_key=None, base_url=None, model=None, structured=False, **kwargs):
        self.async_client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
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


class AzureOpenAIResponsesClient(OpenAIResponsesClient):
    def __init__(self, api_key, azure_endpoint, azure_deployment, api_version, structured=False, **kwargs):
        self.async_client = openai.AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version
        )
        self.model = azure_deployment
        self.kwargs = kwargs
        self.structured = structured

        self.allowed_params = {
            'n', 'stream', 'stop', 
            'max_tokens', 'presence_penalty', 'frequency_penalty',
            'logit_bias', 'user', 'response_format', 'seed',
            'tools', 'tool_choice', 'parallel_tool_calls', 'extra_body',
        }
        
        self.param_mapping = {}


# ================================ Client factory function ===========================


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
        for i in range(10):
            response = await asyncio.to_thread(
                chat.send_message,
                last_user_message,
                generation_config=generation_config
            )
            # Googleの場合は空のレスポンスに対してリトライ処理
            if response.text.strip():
                break
            print(f"Retrying request due to empty content. Retry attempt {i+1} of 10.")
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
        
        # Anthropic純正API用のthinking対応
        params, reasoning_content = self._configure_thinking(params, max_tokens, all_kwargs)
        
        # Anthropic APIを非同期で実行
        try:
            response = await asyncio.to_thread(self.client.messages.create, **params)
        except Exception as e:
            # AnthropicエラーをOpenAI互換エラーに変換してLLMAsyncProcessorのbackoffに対応
            import openai
            import anthropic
            
            if isinstance(e, anthropic.APITimeoutError):
                raise openai.APITimeoutError(f"Anthropic API timeout: {str(e)}")
            elif isinstance(e, anthropic.RateLimitError):
                raise openai.RateLimitError(f"Anthropic rate limit: {str(e)}")
            elif isinstance(e, anthropic.InternalServerError):
                raise openai.InternalServerError(f"Anthropic internal server error: {str(e)}")
            elif isinstance(e, anthropic.APIConnectionError):
                raise openai.APIConnectionError(f"Anthropic connection error: {str(e)}")
            elif "timeout" in str(e).lower() or "timed out" in str(e).lower():
                raise openai.APITimeoutError(f"Anthropic API timeout: {str(e)}")
            elif "rate limit" in str(e).lower():
                raise openai.RateLimitError(f"Anthropic rate limit: {str(e)}")
            else:
                # その他のエラーはそのまま再発生
                raise
        
        # レスポンス処理（thinking対応）
        content = ""
        thinking_content = ""
        
        for block in response.content:
            if hasattr(block, 'type'):
                if block.type == "text":
                    content = block.text
                elif block.type == "thinking":
                    thinking_content = getattr(block, 'thinking', '')
        
        return LLMResponse(content=content, reasoning_content=thinking_content)
    
    def _configure_thinking(self, params, max_tokens, all_kwargs):
        """Anthropic純正API用のthinking設定"""
        try:
            # Docker環境での正しいインポートパス
            instance = WandbConfigSingleton.get_instance()
            cfg = instance.config
            
            # thinking設定を取得
            thinking_config = getattr(cfg.model, 'thinking', None)
            
            if thinking_config:
                budget_tokens = thinking_config.get('budget_tokens', 16384)
                
                # max_tokensを設定（回答用 + 推論用）
                answer_tokens = max_tokens or 2048
                total_max_tokens = answer_tokens + budget_tokens
                
                params["max_tokens"] = total_max_tokens
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget_tokens
                }
                
                # Anthropic thinking有効時はtemperatureを1に設定する必要がある
                params["temperature"] = 1.0
                
                print(f"Anthropic thinking configuration: total={total_max_tokens} "
                      f"(answer={answer_tokens}, budget={budget_tokens}), temperature=1.0")
                
                return params, True
            else:
                # thinking設定がない場合は通常の処理
                if max_tokens:
                    params["max_tokens"] = max_tokens
                elif "max_tokens" not in params:
                    params["max_tokens"] = 1024
                
                return params, False
                
        except Exception as e:
            print(f"Failed to configure Anthropic thinking: {e}")
            # エラーの場合は通常の処理にフォールバック
            if max_tokens:
                params["max_tokens"] = max_tokens
            elif "max_tokens" not in params:
                params["max_tokens"] = 1024
            
            return params, False


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
        # タイムアウト設定
        import httpx
        timeout_config = httpx.Timeout(
            connect=30.0,   # 接続タイムアウト30秒
            read=300.0,     # 読み取りタイムアウト5分
            write=300.0,    # 書き込みタイムアウト5分
            pool=30.0       # プールタイムアウト30秒
        )
        
        # モデル名に基づいてクライアントバージョンを選択
        # command-a-* や新しいモデルはv2を使用
        self.use_v2 = "command-a" in model.lower() or "2025" in model
        
        try:
            if self.use_v2:
                print(f"Using Cohere v2 client for model: {model}")
                self.client = cohere.ClientV2(
                    api_key=api_key,
                    timeout=timeout_config,
                    max_retries=3
                )
            else:
                print(f"Using Cohere v1 client for model: {model}")
                self.client = cohere.Client(
                    api_key=api_key,
                    timeout=timeout_config,
                    max_retries=3
                )
        except Exception as e:
            print(f"Failed to initialize Cohere client: {e}")
            raise
            
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
        """非同期版のinvoke - v1/v2両対応"""
        if self.use_v2:
            return await self._ainvoke_v2(messages, max_tokens, **kwargs)
        else:
            return await self._ainvoke_v1(messages, max_tokens, **kwargs)
    
    async def _ainvoke_v2(self, messages, max_tokens=None, **kwargs):
        """v2 API用の非同期実装"""
        # v2 API用のメッセージ形式に変換
        v2_messages = []
        system_message = None
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            elif msg["role"] == "user":
                v2_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                v2_messages.append({"role": "assistant", "content": msg["content"]})
        
        all_kwargs = {**self.kwargs, **kwargs}
        mapped_params = map_common_params(all_kwargs, self.param_mapping)
        filtered_params = filter_params(mapped_params, self.allowed_params)
        
        params = {
            "model": self.model,
            "messages": v2_messages,
            **filtered_params
        }
        
        if system_message:
            params["system"] = system_message
            
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        try:
            # Cohere v2 APIを非同期で実行
            response = await asyncio.to_thread(self.client.chat, **params)
            # v2 APIのレスポンス形式に対応
            content = response.message.content[0].text if response.message.content else ""
            return LLMResponse(content=content)
        except Exception as e:
            print(f"Cohere v2 API error: {type(e).__name__}: {e}")
            if "timeout" in str(e).lower():
                print("Connection timeout - check network connectivity and API status")
            raise
    
    async def _ainvoke_v1(self, messages, max_tokens=None, **kwargs):
        """v1 API用の非同期実装（既存のコード）"""
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
        
        try:
            # Cohere v1 APIを非同期で実行
            response = await asyncio.to_thread(self.client.chat, **params)
            return LLMResponse(content=response.text)
        except Exception as e:
            print(f"Cohere v1 API error: {type(e).__name__}: {e}")
            if "timeout" in str(e).lower():
                print("Connection timeout - check network connectivity and API status")
            raise



def get_llm_inference_engine() -> BaseLLMClient:
    instance = WandbConfigSingleton.get_instance()
    cfg = instance.config
    api_type = cfg.api

    # vllm-localのみ非推奨警告を表示
    if api_type == "vllm-local":
        warnings.warn(
            "API type 'vllm-local' is deprecated and will be removed in a future version. "
            "Please use 'vllm' (recommended) or 'vllm-docker' for Docker-based vLLM.",
            DeprecationWarning,
            stacklevel=2
        )
        # 後方互換性のため、既存の処理を継続（コンテナ内でvLLM起動）
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
        return llm

    elif api_type in ["vllm", "vllm-docker"]:
        # 推奨: DockerコンテナのvLLMサービスに接続
        # "vllm"と"vllm-docker"は同じ動作
        # llm-leaderboardコンテナ内から実行する場合は、コンテナ名を使用
        base_url = cfg.get("base_url", "http://llm-stack-vllm-1:8000/v1")
        model_name = cfg.model.pretrained_model_name_or_path

        llm = OpenAIClient(
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            base_url=base_url,
            model=model_name,
            **cfg.generator,
        )
        return llm

    elif api_type == "vllm-external":
        warnings.warn(
            "API type 'vllm-external' is deprecated. "
            "Please use 'openai-compatible' for external OpenAI-compatible APIs.",
            DeprecationWarning,
            stacklevel=2
        )
        # 後方互換性のため、openai-compatibleと同じ処理
        api_type = "openai-compatible"
        # fall through to openai-compatible handling

    if api_type == "openai-compatible":
        # 外部のOpenAI互換APIに接続
        base_url = cfg.get("base_url", "http://localhost:8000/v1")
        model_name = cfg.model.pretrained_model_name_or_path

        llm = OpenAIClient(
            api_key=os.environ.get("OPENAI_COMPATIBLE_API_KEY", 
                                  os.environ.get("VLLM_API_KEY", "EMPTY")),
            base_url=base_url,
            model=model_name,  # model_name -> model に修正
            **cfg.generator,
        )
        return llm

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