import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = REPO_ROOT / "scripts" / "tools" / "weave_agents_native_trace.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def test_conversation_url_uses_agents_conversations_route():
    module = load_module()

    url = module.conversation_url(
        "llm-leaderboard",
        "tc-leaderboard",
        "agent:main:61dexmrl:agentic-math:olymmath_hard_0_zh:1783391972-811044-1",
    )

    assert url == (
        "https://wandb.ai/llm-leaderboard/tc-leaderboard/weave/agents/"
        "conversations/"
        "agent%3Amain%3A61dexmrl%3Aagentic-math%3Aolymmath_hard_0_zh"
        "%3A1783391972-811044-1"
    )


def test_summary_prefers_observed_conversation_id_from_latest_trace():
    module = load_module()
    payload = {
        "ok": True,
        "agent_name": "nejumi-taiwan-openclaw",
        "latest_trace_id": "trace-1",
        "agents_url": "https://wandb.ai/llm-leaderboard/tc-leaderboard/weave/agents",
        "query_source": {
            "conversation_id": "",
            "conversation_id_contains": "61dexmrl:agentic-math:task-1",
        },
        "latest_trace_spans_chronological": [
            {
                "conversation_id": "agent:main:61dexmrl:agentic-math:task-1",
            }
        ],
    }

    summary = module.summarize_weave_agents_payload(
        payload,
        entity="llm-leaderboard",
        project="tc-leaderboard",
        agent_name="nejumi-taiwan-openclaw",
        conversation_id="61dexmrl:agentic-math:task-1",
        conversation_id_contains="61dexmrl:agentic-math:task-1",
        verifier_json=Path("verify.json"),
    )

    assert summary["weave_agents_conversation_id"] == (
        "agent:main:61dexmrl:agentic-math:task-1"
    )
    assert summary["weave_agents_conversation_url"].endswith(
        "/weave/agents/conversations/agent%3Amain%3A61dexmrl%3Aagentic-math%3Atask-1"
    )
    assert 'href="https://wandb.ai/llm-leaderboard/tc-leaderboard/weave/agents/' in (
        summary["weave_agents_conversation_link_html"]
    )
