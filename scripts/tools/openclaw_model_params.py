"""Helpers for applying per-model OpenClaw configuration.

OpenClaw stores provider-specific request extensions on a model entry's
``params`` object. For OpenRouter, ``params.provider`` is injected into the
request body as the OpenRouter provider-routing object.

Some OpenClaw settings are model-entry fields instead of request ``params``.
For example, OpenClaw derives the provider ``max_tokens`` request field from
the model entry's top-level ``maxTokens`` value. Keep these two override
surfaces separate so provider routing does not get mixed with model metadata.
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
    return deep_merge_dict(params, params_json)


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
