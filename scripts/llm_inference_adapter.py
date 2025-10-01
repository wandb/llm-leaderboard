import os
import asyncio
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any
import uuid
import json
import warnings
import logging
import re
from config_singleton import WandbConfigSingleton
from omegaconf import OmegaConf, DictConfig
import openai
from openai.types.responses import Response as OpenAIResponse, ParsedResponse as OpenAIParsedResponse
from openai.types.chat import ChatCompletion as OpenAIChatCompletion
from mistralai import Mistral
# import google.generativeai as genai  # Old import - no longer needed
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


def _resolve_http_timeout_from_cfg(cfg, primary_key: str = "openai") -> httpx.Timeout:
    """cfg から HTTP タイムアウト設定を読み出して httpx.Timeout を返す。

    優先順位:
    1. {primary_key}.http_timeout
    2. network.http_timeout
    3. デフォルト（connect=10.0, read=300.0, write=300.0, pool=30.0）
    """
    # デフォルトの緩め設定
    default_connect = 10.0
    default_read = 300.0
    default_write = 300.0
    default_pool = 30.0

    timeout_cfg = None
    try:
        # OmegaConf.select は存在しない場合 None を返すので安全
        timeout_cfg = OmegaConf.select(cfg, f"{primary_key}.http_timeout")
        if timeout_cfg is None:
            timeout_cfg = OmegaConf.select(cfg, "network.http_timeout")
    except Exception:
        # cfg 形式に依存せず安全にフォールバック
        timeout_cfg = None

    def _get_number(dct, key, default_value):
        try:
            value = dct.get(key, default_value)
        except Exception:
            value = default_value
        try:
            return float(value)
        except Exception:
            return default_value

    if timeout_cfg is not None:
        connect = _get_number(timeout_cfg, "connect", default_connect)
        read = _get_number(timeout_cfg, "read", default_read)
        write = _get_number(timeout_cfg, "write", default_write)
        pool = _get_number(timeout_cfg, "pool", default_pool)
    else:
        connect, read, write, pool = default_connect, default_read, default_write, default_pool

    return httpx.Timeout(connect=connect, read=read, write=write, pool=pool)


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
            # Prefer explicit AWS_BEDROCK_REGION, then AWS_DEFAULT_REGION, fallback to us-east-1
            region_name=os.environ.get("AWS_BEDROCK_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")),
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
                # 一部プロバイダが非JSONを返す場合のフォールバック
                try:
                    response_body = json.loads(response.get("body").read())
                except Exception:
                    response_body = {"content": []}
        except ClientError as e:
            print(f"ERROR: Can't invoke '{self.model_id}'. Reason: {e}")
            # フォールバック空応答
            return {"content": []}
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
        # YAMLから（なければデフォルトで）HTTPタイムアウトを解決
        instance = WandbConfigSingleton.get_instance()
        cfg = instance.config if instance else None
        timeout = _resolve_http_timeout_from_cfg(cfg, primary_key="openai") if cfg else httpx.Timeout(connect=10.0, read=300.0, write=300.0, pool=30.0)

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
        
        try:
            response = self.client.chat.completions.create(**params)
        except openai.BadRequestError as e:
            # DashScopeのコンテンツフィルタリングエラーに対する処理
            if "data_inspection_failed" in str(e) or "inappropriate content" in str(e).lower():
                warnings.warn(f"Content filtering detected by DashScope API: {e}")
                # コンテンツフィルタリングエラーの場合、空のレスポンスを返して処理を継続
                return LLMResponse(
                    content="[Content filtered by API provider]",
                    prompt_tokens=0,
                    completion_tokens=0,
                    reasoning_content=""
                )
            else:
                # その他のBadRequestErrorはそのまま再発生
                raise e
        
        # reasoningでtokenを使い切るとcontentがNoneになる対策
        content = '' if response.choices[0].message.content is None else response.choices[0].message.content
        reasoning_content = ''
        # vLLM reasoning parser output
        if hasattr(response.choices[0].message, 'reasoning_content'):
            reasoning_content = getattr(response.choices[0].message, 'reasoning_content', '')
            content = content.lstrip('\n') # vLLM reasoning parserは</think>の後の改行を除去しないためここで除去
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

        try:
            # Structured output
            if "response_format" in params:
                response: OpenAIChatCompletion = await self.async_client.beta.chat.completions.parse(**params)
                parsed_output = response.choices[0].message.parsed
            else:
                response: OpenAIChatCompletion = await self.async_client.chat.completions.create(**params)
                parsed_output = None
        except openai.BadRequestError as e:
            # DashScopeのコンテンツフィルタリングエラーに対する処理
            if "data_inspection_failed" in str(e) or "inappropriate content" in str(e).lower():
                warnings.warn(f"Content filtering detected by DashScope API: {e}")
                # コンテンツフィルタリングエラーの場合、空のレスポンスを返して処理を継続
                return LLMResponse(
                    content="[Content filtered by API provider]",
                    prompt_tokens=0,
                    completion_tokens=0,
                    reasoning_content="",
                    parsed_output=None
                )
            
            # リクエスト時点でpromt+max_tokensがコンテキスト長を超える場合、max_tokensを調整してmax_context_lengthギリギリまで可能な範囲で生成する
            # (OpenAI, vLLMのエラーメッセージに対応)
            if match := re.search("maximum context length is ([0-9]+) tokens. However, you requested [0-9]+ tokens \(([0-9]+) in the messages, ([0-9]+) in the completion\)", e.message):
                max_context_length = int(match.group(1))
                prompt_tokens = int(match.group(2))
                completion_tokens = int(match.group(3))
                shrinked_completion_tokens = max_context_length - prompt_tokens
                if shrinked_completion_tokens > 0:
                    # トークン数が残っている場合はリトライ
                    warnings.warn(
                        f"Shrinking max_tokens {completion_tokens} -> {shrinked_completion_tokens}"
                        f"(max_context_length:{max_context_length}, prompt_tokens:{prompt_tokens}, max_output_tokens:{completion_tokens})"
                    )
                    params["max_tokens"] = shrinked_completion_tokens
                    if "response_format" in params:
                        response: OpenAIChatCompletion = await self.async_client.beta.chat.completions.parse(**params)
                        parsed_output = response.choices[0].message.parsed
                    else:
                        response: OpenAIChatCompletion = await self.async_client.chat.completions.create(**params)
                        parsed_output = None
                else:
                    # promptでトークン数を使い切っている場合はエラー
                    raise e
            else:
                # Token長以外のBadRequestの場合はそのままエラー
                raise e

        # reasoningでtokenを使い切るとcontentがNoneになる対策
        content = '' if response.choices[0].message.content is None else response.choices[0].message.content
        reasoning_content = ''
        # vLLM reasoning parser output
        # https://docs.vllm.ai/en/latest/features/reasoning_outputs.html#quickstart
        if hasattr(response.choices[0].message, 'reasoning_content'):
            reasoning_content = getattr(response.choices[0].message, 'reasoning_content', '')
            content = content.lstrip('\n') # vLLM reasoning parserは</think>の後の改行を除去しないためここで除去
        # OpenRouter reasoning field support
        if not reasoning_content and hasattr(response.choices[0].message, 'reasoning'):
            reasoning_content = getattr(response.choices[0].message, 'reasoning', '')

        # usage が欠落する実装に対しても安全にデフォルト0で継続
        usage = getattr(response, 'usage', None)
        prompt_tokens = getattr(usage, 'prompt_tokens', 0) if usage is not None else 0
        completion_tokens = getattr(usage, 'completion_tokens', 0) if usage is not None else 0

        llm_response = LLMResponse(
            content=content,
            reasoning_content=reasoning_content,
            parsed_output=parsed_output,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
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
        # YAMLから（なければデフォルトで）HTTPタイムアウトを解決
        instance = WandbConfigSingleton.get_instance()
        cfg = instance.config if instance else None
        timeout = _resolve_http_timeout_from_cfg(cfg, primary_key="openai") if cfg else httpx.Timeout(connect=10.0, read=300.0, write=300.0, pool=30.0)

        self.async_client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout
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
        # generator.extra_body.reasoning をトップレベルreasoningに昇格
        try:
            extra_body = all_kwargs.get('extra_body')
            if extra_body and isinstance(extra_body, dict) and 'reasoning' in extra_body:
                all_kwargs['reasoning'] = extra_body['reasoning']
        except Exception:
            pass
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
        collected_tool_calls: List[ToolCall] = []
        for output in response.output:
            if output.type == "message":
                # Restore previous behavior: take the first text part of the message
                try:
                    if output.content and output.content[0].text:
                        content = output.content[0].text
                except Exception:
                    pass
            elif output.type == "reasoning" and len(getattr(output, "summary", [])) > 0:
                # Restore previous behavior: take one summary text if present
                try:
                    reasoning_content = output.summary[0].text
                except Exception:
                    pass
            elif output.type == "function_call":
                # Responses API function calling item
                try:
                    name = getattr(output, "name", None)
                    arguments_raw = getattr(output, "arguments", None)
                    call_id = getattr(output, "call_id", None)
                    parsed_args = {}
                    if isinstance(arguments_raw, str) and len(arguments_raw) > 0:
                        try:
                            parsed_args = json.loads(arguments_raw)
                        except json.JSONDecodeError:
                            # Try to recover from missing closing brace
                            try:
                                parsed_args = json.loads(arguments_raw + "}")
                            except Exception:
                                parsed_args = {"_raw": arguments_raw}
                    collected_tool_calls.append(ToolCall(name=name or "", arguments=parsed_args, id=str(call_id) if call_id else str(uuid.uuid4()), type="function"))
                except Exception:
                    pass

        return LLMResponse(content=content, reasoning_content=reasoning_content, parsed_output=parsed_output, tool_calls=collected_tool_calls if collected_tool_calls else None)


class AzureOpenAIResponsesClient(OpenAIResponsesClient):
    def __init__(self, api_key, azure_endpoint, azure_deployment, api_version, structured=False, **kwargs):
        # YAMLから（なければデフォルトで）HTTPタイムアウトを解決
        instance = WandbConfigSingleton.get_instance()
        cfg = instance.config if instance else None
        timeout = _resolve_http_timeout_from_cfg(cfg, primary_key="azure_openai") if cfg else httpx.Timeout(connect=10.0, read=300.0, write=300.0, pool=30.0)

        self.async_client = openai.AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            timeout=timeout
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
    """
    Google Gemini API client using the new Google GenAI SDK.
    Updated to support the latest Gemini API features including thinking for Gemini 2.5 models.
    """
    def __init__(self, api_key, model, **kwargs):
        # Use the new Google GenAI SDK
        from google import genai
        from google.genai.types import HttpOptions
        
        # Configure the client with API key
        http_options = HttpOptions(api_version="v1beta")
        self.client = genai.Client(api_key=api_key, http_options=http_options)
        self.model = model
        self.kwargs = kwargs
        
        # Completely disable Google GenAI logging and warnings
        import logging
        import warnings
        
        # Disable Google GenAI loggers
        for logger_name in ['google.genai.types', 'google.genai.models', 'google.genai', 'google']:
            logger = logging.getLogger(logger_name)
            logger.disabled = True
            logger.propagate = False
            # Remove all handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
        
        # Set environment variables to suppress warnings
        os.environ['GOOGLE_GENAI_SUPPRESS_WARNINGS'] = 'true'
        os.environ['GOOGLE_GENAI_QUIET'] = 'true'
        os.environ['GOOGLE_GENAI_SILENT'] = 'true'
        
        # Suppress all warnings from Google libraries
        warnings.filterwarnings("ignore", category=UserWarning, module="google")
        warnings.filterwarnings("ignore", message=".*AFC.*")
        warnings.filterwarnings("ignore", message=".*function_call.*")
        warnings.filterwarnings("ignore", message=".*thought_signature.*")
        
        # Suppress all warnings globally
        warnings.filterwarnings("ignore")
        
        # Set logging level for root logger
        logging.getLogger().setLevel(logging.ERROR)
        
        self.allowed_params = {
            'temperature', 'top_p', 'top_k', 'max_output_tokens',
            'candidate_count', 'stop_sequences', 'thinking_config',
            'safety_settings'
        }
        
        # Note: thinking_config supports the new Gemini 2.5 thinking feature
        # Example: thinking_config={"thinking_budget": 0} to disable thinking
        # safety_settings: Configure content safety filters
        # Example: safety_settings=[
        #     SafetySetting(category=SafetyCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.BLOCK_NONE),
        #     SafetySetting(category=SafetyCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.BLOCK_NONE),
        #     SafetySetting(category=SafetyCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_NONE),
        #     SafetySetting(category=SafetyCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.BLOCK_NONE),
        #     SafetySetting(category=SafetyCategory.HARM_CATEGORY_CIVIC_INTEGRITY, threshold=HarmBlockThreshold.BLOCK_NONE),
        # ]
        # Note: Function calling (tools) is only available in BFCL evaluation, not in llm_inference_adapter

        # generation_config: Advanced generation configuration options

        
        self.param_mapping = {
            'max_tokens': 'max_output_tokens',
            'stop': 'stop_sequences',
        }

    async def ainvoke(self, messages, max_tokens=None, **kwargs):
        """非同期版のinvoke using the new Google GenAI SDK"""
        # Convert messages to the format expected by the new SDK
        contents = []
        
        for msg in messages:
            if msg["role"] == "system":
                # System messages are handled differently in the new SDK
                # We'll prepend it to the first user message
                continue
            elif msg["role"] == "user":
                contents.append({
                    "role": "user",
                    "parts": [{"text": msg["content"]}]
                })
            elif msg["role"] == "assistant":
                contents.append({
                    "role": "model",
                    "parts": [{"text": msg["content"]}]
                })
        
        # Handle system message by prepending to first user message if exists
        system_message = None
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
                break
        
        if system_message and contents:
            # Prepend system instruction to the first user message
            if contents[0]["role"] == "user":
                contents[0]["parts"][0]["text"] = f"{system_message}\n\n{contents[0]['parts'][0]['text']}"
        
        # Prepare generation config
        all_kwargs = {**self.kwargs, **kwargs}
        mapped_params = map_common_params(all_kwargs, self.param_mapping)
        generation_config = filter_params(mapped_params, self.allowed_params)
        
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens
        
        # Prepare the request parameters
        params = {
            "model": self.model,
            "contents": contents,
        }
        
        # Prepare config object for generation parameters
        config_params = {}
        
        # All parameters go directly in config according to Gemini API docs
        for key, value in generation_config.items():
            config_params[key] = value
        
        # Disable function calling for SWE-bench evaluation
        # This prevents the 'NoneType' object has no attribute 'strip' error
        if "tools" in config_params:
            del config_params["tools"]
        if "tool_choice" in config_params:
            del config_params["tool_choice"]
        
        # Create config object if needed
        if config_params:
            try:
                from google.genai import types
                
                # Create config with all parameters according to Gemini API docs
                config_kwargs = {}
                
                # Handle thinking configuration for Gemini 2.5 models
                # Get thinking_budget from config file, not from config_params
                try:
                    instance = WandbConfigSingleton.get_instance()
                    cfg = instance.config
                    thinking_budget = getattr(cfg.model, 'thinking_budget', 0)
                    
                    # Handle automatic thinking budget allocation
                    if thinking_budget == -1:
                        # Use automatic allocation (API will choose optimal value)
                        config_kwargs["thinking_config"] = types.ThinkingConfig()
                        print(f"Gemini thinking enabled with automatic budget allocation")
                    elif thinking_budget >= 0:
                        # Use specific budget value
                        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
                        print(f"Gemini thinking enabled with budget: {thinking_budget}")
                    else:
                        # Disable thinking
                        print(f"Gemini thinking disabled (budget: {thinking_budget})")
                except Exception as e:
                    print(f"Warning: Failed to configure thinking: {e}")
                    # Fallback: disable thinking
                    pass
                
                # Handle safety settings if provided
                if 'safety_settings' in config_params:
                    safety_settings = config_params['safety_settings']
                    # Convert to proper format if needed
                    if isinstance(safety_settings, list) and safety_settings:
                        # Check if we need to convert from dict format to SafetySetting objects
                        if isinstance(safety_settings[0], dict):
                            try:
                                converted_safety_settings = []
                                for setting in safety_settings:
                                    if 'category' in setting and 'threshold' in setting:
                                        converted_safety_settings.append(
                                            types.SafetySetting(
                                                category=setting['category'],
                                                threshold=setting['threshold']
                                            )
                                        )
                                    else:
                                        converted_safety_settings.append(types.SafetySetting(**setting))
                                config_kwargs["safety_settings"] = converted_safety_settings
                                print(f"Converted {len(converted_safety_settings)} safety settings from dict format")
                            except Exception as e:
                                print(f"Warning: Failed to convert safety settings: {e}")
                        else:
                            config_kwargs["safety_settings"] = safety_settings
                            print(f"Safety settings configured: {len(safety_settings)} categories")
                    else:
                        config_kwargs["safety_settings"] = safety_settings
                        print("Safety settings configured")
                
                # Add all other parameters directly to config
                for key, value in config_params.items():
                    if key not in ['thinking_config', 'safety_settings']:
                        config_kwargs[key] = value
                
                # Create the main config
                params["config"] = types.GenerateContentConfig(**config_kwargs)
                
            except ImportError:
                print("Warning: google.genai.types not available, config ignored")
                # Fallback: add all parameters directly to params if types not available
                for key, value in config_params.items():
                    if key == 'thinking_config':
                        # Skip thinking_config as it requires types
                        print(f"Warning: Skipping {key} as types not available")
                    else:
                        params[key] = value
        
        try:
            # Use the new Google GenAI SDK to generate content
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                **params
            )
            
            # Extract the response text with function calling support
            content = ""
            try:
                if hasattr(response, 'text') and response.text is not None:
                    content = response.text
                elif hasattr(response, 'candidates') and response.candidates:
                    # Handle function calling responses properly
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content:
                        if hasattr(candidate.content, 'parts') and candidate.content.parts:
                            for part in candidate.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    content += part.text
                                elif hasattr(part, 'function_call') and part.function_call:
                                    # Skip function calls in SWE-bench evaluation
                                    continue
            except Exception as parse_error:
                print(f"Warning: Failed to parse response content: {parse_error}")
                content = ""
            
            # Handle empty responses with retry logic
            if not content.strip():
                # Retry up to 3 times for empty responses
                for i in range(3):
                    print(f"Retrying request due to empty content. Retry attempt {i+1} of 3.")
                    try:
                        response = await asyncio.to_thread(
                            self.client.models.generate_content,
                            **params
                        )
                        # Parse response again
                        content = ""
                        if hasattr(response, 'text') and response.text is not None:
                            content = response.text
                        if content.strip():
                            break
                    except Exception as retry_error:
                        print(f"Retry {i+1} failed: {retry_error}")
                        continue
            
            return LLMResponse(content=content)
            
        except Exception as e:
            print(f"Google GenAI API error: {type(e).__name__}: {e}")
            # Log additional error details for debugging
            if hasattr(e, 'status_code'):
                print(f"HTTP Status: {e.status_code}")
            if hasattr(e, 'response'):
                print(f"Response: {e.response}")
            
            # Handle specific error types
            if "500" in str(e) or "INTERNAL" in str(e):
                print("Google GenAI internal error - this is a server-side issue")
            elif "ConnectError" in str(e):
                print("Network connection error - check internet connectivity")
            
            # Return empty response on error
            return LLMResponse(content="")


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
        # YAMLから（なければデフォルトで）HTTPタイムアウトを解決
        instance = WandbConfigSingleton.get_instance()
        cfg = instance.config if instance else None
        timeout = _resolve_http_timeout_from_cfg(cfg, primary_key="azure_openai") if cfg else httpx.Timeout(connect=10.0, read=300.0, write=300.0, pool=30.0)

        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            timeout=timeout
        )
        self.async_client = openai.AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            timeout=timeout
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
                )
            else:
                print(f"Using Cohere v1 client for model: {model}")
                self.client = cohere.Client(
                    api_key=api_key,
                    timeout=timeout_config,
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
            base_url="https://api.upstage.ai/v1/",
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


