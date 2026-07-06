import os
from llm_inference_adapter import (
    AzureOpenAIResponsesClient,
    OpenAIClient,
    OpenAIResponsesClient,
)


def _as_chat_structured_kwargs(kwargs: dict) -> dict:
    """Map Responses-style structured output args to chat-compatible clients."""
    mapped = dict(kwargs)
    text_format = mapped.pop("text_format", None)
    if text_format is not None:
        mapped["response_format"] = text_format
        mapped["manual_response_format_parse"] = True
    return mapped


def _openrouter_api_key() -> str:
    api_key_env = os.environ.get("NEJUMI_OPENROUTER_API_KEY_ENV")
    if api_key_env:
        return os.environ[api_key_env]
    return os.environ["OPENROUTER_API_KEY"]


def _openrouter_judge_client(model: str, **kwargs):
    return OpenAIClient(
        api_key=_openrouter_api_key(),
        base_url="https://openrouter.ai/api/v1",
        model=model,
        timeout_primary_key="openrouter",
        **_as_chat_structured_kwargs(kwargs),
    )


def _deepseek_judge_client(model: str, **kwargs):
    return OpenAIClient(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        model=model,
        timeout_primary_key="deepseek",
        **_as_chat_structured_kwargs(kwargs),
    )


def _xai_judge_client(model: str, **kwargs):
    return OpenAIClient(
        api_key=os.environ["XAI_API_KEY"],
        base_url="https://api.x.ai/v1",
        model=model,
        timeout_primary_key="xai",
        **_as_chat_structured_kwargs(kwargs),
    )


def get_openai_judge_client(model: str, **kwargs):
    """Get judge client based on environment configuration.

    Historical callers use this OpenAI-named function for all LLM judges.  Keep
    the API stable, but allow explicit model prefixes for OpenAI-compatible
    fallback judges when direct OpenAI quota is unavailable.
    """
    if model.startswith("openrouter/"):
        return _openrouter_judge_client(model.removeprefix("openrouter/"), **kwargs)
    if model.startswith("deepseek/"):
        return _deepseek_judge_client(model.removeprefix("deepseek/"), **kwargs)
    if model.startswith("xai/"):
        return _xai_judge_client(model.removeprefix("xai/"), **kwargs)

    forced_provider = os.environ.get("NEJUMI_JUDGE_PROVIDER", "").strip().lower()
    if forced_provider == "openrouter":
        return _openrouter_judge_client(model, **kwargs)
    if forced_provider == "deepseek":
        return _deepseek_judge_client(model, **kwargs)
    if forced_provider == "xai":
        return _xai_judge_client(model, **kwargs)

    api_type = os.environ.get('OPENAI_API_TYPE', 'openai')
    
    if api_type == "azure":
        if model.startswith("azure-"):
            model = model[6:]
        return AzureOpenAIResponsesClient(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version="2023-07-01-preview",
            azure_deployment=model,
            **kwargs,
        )
    else:
        return OpenAIResponsesClient(
            model=model,
            **kwargs,
        )
