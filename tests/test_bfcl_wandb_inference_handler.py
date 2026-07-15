from pathlib import Path
from types import SimpleNamespace
import sys

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
BFCL_ROOT = REPO_ROOT / "scripts" / "evaluator" / "evaluate_utils" / "bfcl_pkg"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(BFCL_ROOT))


def test_wandb_inference_bfcl_model_is_registered():
    from bfcl.constants.model_config import MODEL_CONFIG_MAPPING
    from bfcl.model_handler.api_inference.wandb_inference import WandBInferenceHandler

    config = MODEL_CONFIG_MAPPING["WandBInference-FC"]

    assert config.model_handler is WandBInferenceHandler
    assert config.is_fc_model is True
    assert config.underscore_to_dot is True


def test_wandb_inference_handler_uses_async_openai_compatible_path(monkeypatch):
    from bfcl import _llm_response_generation as generation
    from bfcl.model_handler.api_inference.wandb_inference import WandBInferenceHandler
    from bfcl.model_handler.model_style import ModelStyle

    called = {}

    async def fake_async_generate_results(args, handler, test_cases_total):
        called["args"] = args
        called["handler"] = handler
        called["cases"] = test_cases_total

    handler = object.__new__(WandBInferenceHandler)
    handler.model_style = ModelStyle.OpenAI_Completions

    monkeypatch.setattr(generation, "async_generate_results", fake_async_generate_results)

    args = SimpleNamespace(temperature=0.0, local_model_path=None)
    cases = [{"id": "case_0", "function": []}]
    generation.generate_results(args, "WandBInference-FC", cases, handler=handler)

    assert called["handler"] is handler
    assert called["cases"] == cases


def test_bfcl_model_alias_does_not_override_configured_openai_compatible_model(monkeypatch):
    import llm_inference_adapter as adapter_module
    from llm_inference_adapter import OpenAIClient

    cfg = OmegaConf.create(
        {
            "openai": {
                "http_timeout": {
                    "connect": 1,
                    "read": 10,
                    "write": 10,
                    "pool": 1,
                }
            }
        }
    )
    monkeypatch.setattr(
        adapter_module.WandbConfigSingleton,
        "get_instance",
        staticmethod(lambda: SimpleNamespace(config=cfg)),
    )

    client = OpenAIClient(
        api_key="test-key",
        base_url="https://api.inference.wandb.ai/v1",
        model="zai-org/GLM-5.2",
    )
    captured = {}

    class FakeCompletions:
        def create(self, **params):
            captured.update(params)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="ok"),
                        finish_reason="stop",
                    )
                ]
            )

    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )

    response = client.invoke(
        [{"role": "user", "content": "hello"}],
        model="WandBInference",
        max_tokens=16,
    )

    assert response.content == "ok"
    assert captured["model"] == "zai-org/GLM-5.2"
