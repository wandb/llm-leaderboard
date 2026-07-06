import sys
import types
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


if "mistralai" not in sys.modules:
    mistralai_stub = types.ModuleType("mistralai")
    mistralai_stub.Mistral = object
    sys.modules["mistralai"] = mistralai_stub

from config_singleton import WandbConfigSingleton
from evaluator.evaluate_utils.llm_judge_client import get_openai_judge_client
from llm_inference_adapter import _resolve_openai_compatible_api_key


class DummyJudgeOutput(BaseModel):
    answer: str


def test_openrouter_judge_prefix_uses_chat_client(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setattr(
        WandbConfigSingleton,
        "_instance",
        SimpleNamespace(
            run=None,
            llm=None,
            config=OmegaConf.create(
                {"inference_interval": 0, "batch_size": 1, "network": {}}
            ),
        ),
    )

    client = get_openai_judge_client(
        "openrouter/anthropic/claude-sonnet-4.6",
        text_format=DummyJudgeOutput,
    )

    assert client.model == "anthropic/claude-sonnet-4.6"
    assert client.base_url == "https://openrouter.ai/api/v1"
    assert client.kwargs["response_format"] is DummyJudgeOutput
    assert client.kwargs["manual_response_format_parse"] is True


def test_openrouter_gpt55_judge_prefix_uses_openrouter_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setattr(
        WandbConfigSingleton,
        "_instance",
        SimpleNamespace(
            run=None,
            llm=None,
            config=OmegaConf.create(
                {"inference_interval": 0, "batch_size": 1, "network": {}}
            ),
        ),
    )

    client = get_openai_judge_client(
        "openrouter/openai/gpt-5.5",
        text_format=DummyJudgeOutput,
    )

    assert client.model == "openai/gpt-5.5"
    assert client.base_url == "https://openrouter.ai/api/v1"
    assert client.kwargs["response_format"] is DummyJudgeOutput
    assert client.kwargs["manual_response_format_parse"] is True


def test_openai_compatible_openrouter_key_precedence(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "generic-key")

    assert (
        _resolve_openai_compatible_api_key("https://openrouter.ai/api/v1")
        == "openrouter-key"
    )


def test_openai_compatible_openrouter_key_env_override(monkeypatch):
    monkeypatch.setenv("NEJUMI_OPENROUTER_API_KEY_ENV", "OPENAI_COMPATIBLE_API_KEY")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "generic-key")

    assert (
        _resolve_openai_compatible_api_key("https://openrouter.ai/api/v1")
        == "generic-key"
    )


def test_openai_compatible_explicit_key_env_wins(monkeypatch):
    cfg = OmegaConf.create({"api_key_env": "CUSTOM_COMPAT_KEY"})
    monkeypatch.setenv("CUSTOM_COMPAT_KEY", "custom-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")

    assert (
        _resolve_openai_compatible_api_key("https://openrouter.ai/api/v1", cfg)
        == "custom-key"
    )
