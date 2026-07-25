from dataclasses import dataclass
from typing import Optional

from bfcl_eval.model_handler.configured_llm import ConfiguredLLMHandler


@dataclass
class ModelConfig:
    model_name: str
    display_name: str
    url: str
    org: str
    license: str
    model_handler: type
    input_price: Optional[float] = None
    output_price: Optional[float] = None
    is_fc_model: bool = True
    underscore_to_dot: bool = False


def _is_fc_registry_name(name: str) -> bool:
    normalized = name.lower()
    if "jsonschema" in normalized or normalized.endswith("-prompt"):
        return False
    return (
        normalized.endswith("-fc")
        or normalized in {
            "openrouter-fc",
            "wandbinference-fc",
            "openairesponseshandler-fc",
            "claude-fc",
            "unified-oss-fc",
        }
    )


class DynamicModelConfigMapping(dict):
    """Resolve leaderboard YAML model IDs without importing optional SDKs."""

    def _resolve(self, key):
        if not isinstance(key, str) or not key.strip():
            raise KeyError(key)
        is_fc_model = _is_fc_registry_name(key)
        return ModelConfig(
            model_name=key.removesuffix("-FC"),
            display_name=key,
            url="",
            org="Configured provider",
            license="Provider-specific",
            model_handler=ConfiguredLLMHandler,
            input_price=None,
            output_price=None,
            is_fc_model=is_fc_model,
            underscore_to_dot=is_fc_model,
        )

    def __missing__(self, key):
        value = self._resolve(key)
        self[key] = value
        return value

    def __contains__(self, key):
        try:
            self._resolve(key)
        except KeyError:
            return False
        return True


MODEL_CONFIG_MAPPING = DynamicModelConfigMapping()
