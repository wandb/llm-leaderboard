"""Helpers for applying per-model OpenClaw configuration.

OpenClaw stores provider-specific request extensions on a model entry's
``params`` object. Recent OpenClaw versions also derive stream options such as
``maxTokens`` from that object, while the top-level model ``maxTokens`` remains
the model metadata cap. The helpers therefore mirror a top-level maxTokens
override into request params when no explicit request value was supplied.
"""

from __future__ import annotations

import copy
import json
from typing import Any


def deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_json_object(value: Any, *, label: str) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return copy.deepcopy(value)
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be a JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def openclaw_model_params_from_args(args: Any) -> dict[str, Any]:
    params = parse_json_object(
        getattr(args, "openclaw_model_params", None),
        label="openclaw_model_params",
    )
    params_json = parse_json_object(
        getattr(args, "openclaw_model_params_json", None),
        label="--openclaw-model-params-json",
    )
    merged = deep_merge_dict(params, params_json)
    overrides = openclaw_model_overrides_from_args(args)
    if "maxTokens" not in merged and "max_tokens" not in merged:
        max_tokens = overrides.get("maxTokens")
        if max_tokens is not None:
            merged["maxTokens"] = copy.deepcopy(max_tokens)
    return merged


def openclaw_model_overrides_from_args(args: Any) -> dict[str, Any]:
    overrides = parse_json_object(
        getattr(args, "openclaw_model_overrides", None),
        label="openclaw_model_overrides",
    )
    overrides_json = parse_json_object(
        getattr(args, "openclaw_model_overrides_json", None),
        label="--openclaw-model-overrides-json",
    )
    return deep_merge_dict(overrides, overrides_json)


def openclaw_max_output_tokens_from_args(args: Any) -> int | None:
    """Resolve the effective per-response output cap communicated to the model."""
    params = openclaw_model_params_from_args(args)
    raw_value = params.get("maxTokens", params.get("max_tokens"))
    try:
        value = int(raw_value or 0)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def apply_openclaw_model_overrides(
    model_entry: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any] | None:
    if not overrides:
        return None
    merged = deep_merge_dict(model_entry, overrides)
    model_entry.clear()
    model_entry.update(merged)
    return copy.deepcopy(overrides)


def apply_openclaw_model_params(
    model_entry: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any] | None:
    if not params:
        return None
    existing = model_entry.get("params")
    if not isinstance(existing, dict):
        existing = {}
    merged = deep_merge_dict(existing, params)
    model_entry["params"] = merged
    return merged
