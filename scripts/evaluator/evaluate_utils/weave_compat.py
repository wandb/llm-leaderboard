from __future__ import annotations

from typing import Any


_PATCH_MARKER = "_nejumi_openai_responses_compatible"


class _ProbeClient:
    _base_url = "https://example.invalid/v1"
    _version = "probe"


class _ResponsesResourceProbe:
    """Match OpenAI Responses resources, which do not expose `.messages`."""

    _client = _ProbeClient()


def configure_openai_responses_input_sanitization() -> dict[str, Any]:
    """Prevent older Weave releases from retaining OpenAI Responses clients.

    Weave's OpenAI input handler replaces resource `self` with client metadata
    before tracing. Older releases only recognize Chat Completions resources
    because they require a `.messages` attribute. Responses resources lack that
    attribute, so their HTTP client can be retained by every trace call.

    The behavior probe makes this compatibility patch unnecessary once upstream
    Weave recognizes Responses resources itself.
    """
    try:
        from weave.integrations.openai import openai_sdk
    except (ImportError, ModuleNotFoundError) as exc:
        return {
            "available": False,
            "applied": False,
            "status": "weave_openai_integration_unavailable",
            "detail": f"{type(exc).__name__}: {exc}",
        }

    checker = getattr(openai_sdk, "completion_instance_check", None)
    converter = getattr(openai_sdk, "convert_completion_to_dict", None)
    if not callable(checker) or not callable(converter):
        return {
            "available": True,
            "applied": False,
            "status": "unsupported_weave_openai_integration",
        }

    if bool(checker(_ResponsesResourceProbe())):
        return {
            "available": True,
            "applied": bool(getattr(checker, _PATCH_MARKER, False)),
            "status": (
                "compatibility_patch_already_applied"
                if getattr(checker, _PATCH_MARKER, False)
                else "upstream_behavior_compatible"
            ),
        }

    original_checker = checker

    def responses_compatible_checker(obj: Any) -> bool:
        if original_checker(obj):
            return True
        client = getattr(obj, "_client", None)
        return (
            client is not None
            and hasattr(client, "_base_url")
            and hasattr(client, "_version")
        )

    setattr(responses_compatible_checker, _PATCH_MARKER, True)
    setattr(responses_compatible_checker, "_nejumi_original", original_checker)
    openai_sdk.completion_instance_check = responses_compatible_checker

    if not bool(openai_sdk.completion_instance_check(_ResponsesResourceProbe())):
        raise RuntimeError(
            "Failed to enable Weave OpenAI Responses input sanitization"
        )

    return {
        "available": True,
        "applied": True,
        "status": "compatibility_patch_applied",
    }
