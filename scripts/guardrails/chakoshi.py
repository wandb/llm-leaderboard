import os
from typing import Any, Dict, Optional

import requests


DEFAULT_API_BASE = "https://api.beta.chakoshi.ntt.com"


def judge_text(
    text: str,
    *,
    api_key: Optional[str] = None,
    model: str = "chakoshi-moderation-241223",
    category_set_id: Optional[str] = None,
    timeout_seconds: float = 10.0,
    api_base: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Call Chakoshi moderation API to judge the given text.

    Returns a dict with keys: flagged, reason, unsafe_score, unsafe_category, label_str.
    Errors and non-200 responses are treated as not-flagged with a reason message.
    """
    api_key = api_key or os.getenv("CHAKOSHI_API_KEY")
    if not api_key:
        return {
            "flagged": False,
            "reason": "CHAKOSHI_API_KEY is not set",
            "unsafe_score": None,
            "unsafe_category": None,
            "label_str": None,
        }

    url = f"{(api_base or DEFAULT_API_BASE).rstrip('/')}/v1/judge/text"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "accept": "application/json",
    }
    payload: Dict[str, Any] = {
        "input": str(text),
        "model": model,
    }
    csid = category_set_id or os.getenv("CHAKOSHI_CATEGORY_SET_ID")
    if csid:
        payload["category_set_id"] = csid

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
    except Exception as e:
        return {
            "flagged": False,
            "reason": f"Chakoshi API error: {e}",
            "unsafe_score": None,
            "unsafe_category": None,
            "label_str": None,
        }

    if resp.status_code != 200:
        return {
            "flagged": False,
            "reason": f"Chakoshi API non-200: {resp.status_code}",
            "unsafe_score": None,
            "unsafe_category": None,
            "label_str": None,
        }

    try:
        data: Dict[str, Any] = resp.json() if resp.content else {}
    except Exception:
        data = {}

    results: Dict[str, Any] = data.get("results", {}) if isinstance(data, dict) else {}
    unsafe_flag = bool(results.get("unsafe_flag", False))
    label_str = results.get("label_str")
    unsafe_score = results.get("unsafe_score")
    unsafe_category = results.get("unsafe_category")

    return {
        "flagged": unsafe_flag,
        "reason": (f"Chakoshi flagged: {unsafe_category}, score: {unsafe_score} ({label_str})" if unsafe_flag else None),
        "unsafe_score": unsafe_score,
        "unsafe_category": unsafe_category,
        "label_str": label_str,
    }


def apply_guardrail(
    text: str,
    *,
    enabled: bool,
    model: str = "chakoshi-moderation-241223",
    category_set_id: Optional[str] = None,
    timeout_seconds: float = 10.0,
    on_flag: str = "block",  # block | replace | allow
    block_message: str = "[BLOCKED] Unsafe content detected",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> str:
    """
    Apply Chakoshi moderation guardrail to text when enabled.

    - When flagged and on_flag == 'block': returns block_message
    - When flagged and on_flag == 'replace': returns block_message
    - When flagged and on_flag == 'allow': returns original text
    - When not flagged or disabled: returns original text
    """
    if not enabled:
        return text

    result = judge_text(
        text,
        api_key=api_key,
        model=model,
        category_set_id=category_set_id,
        timeout_seconds=timeout_seconds,
        api_base=api_base,
    )
    if not result.get("flagged"):
        return text

    if on_flag in {"block", "replace"}:
        reason = result.get("reason")
        if reason:
            return f"{block_message} ({reason})"
        return block_message

    # allow
    return text

