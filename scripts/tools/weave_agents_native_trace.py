#!/usr/bin/env python3
"""Helpers for validating native weave-openclaw Agents traces."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote


REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFY_WEAVE_AGENTS = REPO_ROOT / "scripts" / "tools" / "verify_taiwan_weave_agents.py"
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
DEFAULT_AGENT_NAME = "nejumi-taiwan-openclaw"
DEFAULT_ENTITY = "llm-leaderboard"
DEFAULT_PROJECT = "tc-leaderboard"

WEAVE_AGENTS_OUTPUT_COLUMNS = (
    "weave_agents_ok",
    "weave_agents_required",
    "weave_agents_agent_name",
    "weave_agents_conversation_id",
    "weave_agents_conversation_id_contains",
    "weave_agents_conversation_url",
    "weave_agents_conversation_link_html",
    "weave_agents_trace_id",
    "weave_agents_url",
    "weave_agents_trace_url",
    "weave_agents_verifier_json",
    "weave_agents_error",
)


def agents_url(entity: str, project: str) -> str:
    return f"https://wandb.ai/{entity}/{project}/weave/agents"


def trace_url(entity: str, project: str, trace_id: str | None) -> str:
    if not trace_id:
        return ""
    return f"{agents_url(entity, project)}?trace_id={quote(trace_id)}"


def conversation_url(entity: str, project: str, conversation_id: str | None) -> str:
    if not conversation_id:
        return ""
    return f"{agents_url(entity, project)}/conversations/{quote(conversation_id, safe='')}"


def conversation_link_html(url: str, *, label: str = "Agents conversation") -> str:
    if not url:
        return ""
    return f'<a href="{escape(url, quote=True)}" target="_blank">{escape(label)}</a>'


def model_aliases(model: str | None) -> list[str]:
    raw = str(model or "").strip()
    if not raw:
        return []
    aliases = [raw]
    if "/" in raw:
        aliases.append(raw.rsplit("/", 1)[-1])
    return list(dict.fromkeys(aliases))


def empty_weave_agents_evidence(
    *,
    required: bool,
    entity: str,
    project: str,
    agent_name: str,
    conversation_id: str = "",
    conversation_id_contains: str = "",
    verifier_json: str = "",
    error: str = "",
) -> dict[str, Any]:
    return {
        "weave_agents_ok": None if required else False,
        "weave_agents_required": required,
        "weave_agents_agent_name": agent_name,
        "weave_agents_conversation_id": conversation_id,
        "weave_agents_conversation_id_contains": conversation_id_contains,
        "weave_agents_conversation_url": conversation_url(entity, project, conversation_id),
        "weave_agents_conversation_link_html": conversation_link_html(
            conversation_url(entity, project, conversation_id)
        ),
        "weave_agents_trace_id": "",
        "weave_agents_url": agents_url(entity, project),
        "weave_agents_trace_url": "",
        "weave_agents_verifier_json": verifier_json,
        "weave_agents_error": error,
    }


def summarize_weave_agents_payload(
    payload: dict[str, Any],
    *,
    entity: str,
    project: str,
    agent_name: str,
    conversation_id: str,
    conversation_id_contains: str,
    verifier_json: Path,
    error: str = "",
) -> dict[str, Any]:
    trace_id = str(payload.get("latest_trace_id") or "")
    query_source = payload.get("query_source") if isinstance(payload.get("query_source"), dict) else {}
    observed_id = observed_conversation_id(payload) or conversation_id or conversation_id_contains
    observed_url = conversation_url(entity, project, observed_id)
    return {
        "weave_agents_ok": bool(payload.get("ok")),
        "weave_agents_required": True,
        "weave_agents_agent_name": str(payload.get("agent_name") or agent_name),
        "weave_agents_conversation_id": observed_id,
        "weave_agents_conversation_id_contains": str(
            query_source.get("conversation_id_contains") or conversation_id_contains or ""
        ),
        "weave_agents_conversation_url": observed_url,
        "weave_agents_conversation_link_html": conversation_link_html(observed_url),
        "weave_agents_trace_id": trace_id,
        "weave_agents_url": str(payload.get("agents_url") or agents_url(entity, project)),
        "weave_agents_trace_url": trace_url(entity, project, trace_id),
        "weave_agents_verifier_json": str(verifier_json),
        "weave_agents_error": error,
    }


def load_verifier_payload(path: Path, stdout: str) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(stdout)


def observed_conversation_id(payload: dict[str, Any]) -> str:
    for key in ("latest_trace_spans_chronological", "latest_trace_chat_messages_chronological"):
        rows = payload.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            value = row.get("conversation_id")
            if isinstance(value, str) and value.strip():
                return value.strip()
    query_source = payload.get("query_source") if isinstance(payload.get("query_source"), dict) else {}
    for key in ("conversation_id", "conversation_id_contains"):
        value = query_source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def verify_native_weave_agents_trace(
    *,
    entity: str,
    project: str,
    agent_name: str,
    conversation_id_contains: str,
    verifier_json: Path,
    env_file: Path | None = DEFAULT_ENV_FILE,
    expected_model: str | None = None,
    required_texts: list[str] | None = None,
    require_tool_trace: bool = False,
    require_usage: bool = True,
    limit: int = 50,
    timeout_seconds: float = 120.0,
    poll_seconds: float = 5.0,
) -> dict[str, Any]:
    verifier_json.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(VERIFY_WEAVE_AGENTS),
        "--entity",
        entity,
        "--project",
        project,
        "--agent-name",
        agent_name,
        "--conversation-id-contains",
        conversation_id_contains,
        "--limit",
        str(limit),
        "--json",
        str(verifier_json),
    ]
    if env_file is not None:
        command.extend(["--env-file", str(env_file)])
    if require_usage:
        command.append("--require-usage")
    if require_tool_trace:
        command.extend(["--require-tool-span", "--require-tool-content"])
    for text in required_texts or []:
        if text:
            command.extend(["--require-text", text])
    for alias in model_aliases(expected_model):
        command.extend(["--expected-request-model", alias])

    deadline = time.monotonic() + max(0.0, timeout_seconds)
    last_result: subprocess.CompletedProcess[str] | None = None
    last_payload: dict[str, Any] | None = None
    while True:
        last_result = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
        try:
            last_payload = load_verifier_payload(verifier_json, last_result.stdout)
        except (json.JSONDecodeError, OSError):
            last_payload = None
        if last_result.returncode == 0 and isinstance(last_payload, dict) and last_payload.get("ok"):
            return summarize_weave_agents_payload(
                last_payload,
                entity=entity,
                project=project,
                agent_name=agent_name,
                conversation_id=conversation_id_contains,
                conversation_id_contains=conversation_id_contains,
                verifier_json=verifier_json,
            )
        if time.monotonic() >= deadline:
            break
        time.sleep(max(0.1, poll_seconds))

    detail = ""
    if last_payload is not None:
        failed = [
            f"{check.get('name')}: {check.get('detail')}"
            for check in last_payload.get("checks", [])
            if isinstance(check, dict) and check.get("ok") is False
        ]
        detail = "; ".join(failed[:5])
    if not detail and last_result is not None:
        detail = (last_result.stderr or last_result.stdout or "").strip()[-4000:]
    raise RuntimeError(
        "Native weave-openclaw Agents trace verification failed for "
        f"{conversation_id_contains}: {detail}"
    )


def env_default_entity() -> str:
    return os.environ.get("WANDB_ENTITY", DEFAULT_ENTITY)


def env_default_project() -> str:
    return os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT)
