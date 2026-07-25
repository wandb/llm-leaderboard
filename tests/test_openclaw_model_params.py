import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "tools" / "openclaw_model_params.py"


def load_module():
    spec = importlib.util.spec_from_file_location("openclaw_model_params", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_top_level_max_tokens_is_mirrored_to_request_params():
    module = load_module()
    args = SimpleNamespace(
        openclaw_model_params=None,
        openclaw_model_params_json=json.dumps({"provider": {"only": ["z-ai/fp8"]}}),
        openclaw_model_overrides=None,
        openclaw_model_overrides_json=json.dumps({"maxTokens": 65536}),
    )

    params = module.openclaw_model_params_from_args(args)

    assert params == {
        "provider": {"only": ["z-ai/fp8"]},
        "maxTokens": 65536,
    }


def test_explicit_request_max_tokens_is_not_replaced_by_model_cap():
    module = load_module()
    args = SimpleNamespace(
        openclaw_model_params={"maxTokens": 8192},
        openclaw_model_params_json=None,
        openclaw_model_overrides={"maxTokens": 65536},
        openclaw_model_overrides_json=None,
    )

    assert module.openclaw_model_params_from_args(args)["maxTokens"] == 8192


def test_resolves_effective_per_response_output_cap():
    module = load_module()
    args = SimpleNamespace(
        openclaw_model_params={"maxTokens": 32768},
        openclaw_model_params_json='{"maxTokens": 65536}',
        openclaw_model_overrides=None,
        openclaw_model_overrides_json=None,
    )

    assert module.openclaw_max_output_tokens_from_args(args) == 65536
