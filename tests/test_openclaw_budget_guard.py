from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ENTRY = REPO_ROOT / "openclaw-plugins" / "nejumi-budget-guard" / "dist" / "index.js"


def run_node_probe(source: str) -> dict:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", source],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_budget_guard_blocks_41st_tool_call() -> None:
    source = f"""
import plugin from {json.dumps(str(PLUGIN_ENTRY))};
const handlers = {{}};
const api = {{
  pluginConfig: {{
    enabled: true,
    maxToolCalls: 40,
    maxAgentTurns: 0,
    agentIds: ["agent-a"],
    sessionKeyPrefixes: ["swebench-pro"],
    blockReasonPrefix: "NEJUMI_BUDGET_GUARD_BLOCKED",
  }},
  on(name, handler) {{ handlers[name] = handler; }},
}};
plugin.register(api);
let blocked = null;
for (let i = 1; i <= 41; i++) {{
  const result = await handlers.before_tool_call(
    {{ toolName: "read", toolCallId: `call-${{i}}` }},
    {{ agentId: "agent-a", sessionKey: "swebench-pro:task:attempt", runId: "run-tool" }},
  );
  if (result?.block) blocked = {{ i, result }};
}}
console.log(JSON.stringify(blocked));
"""
    blocked = run_node_probe(source)
    assert blocked["i"] == 41
    assert blocked["result"]["block"] is True
    assert "tool_call_limit_exceeded" in blocked["result"]["blockReason"]
    assert "observed=41" in blocked["result"]["blockReason"]
    assert "limit=40" in blocked["result"]["blockReason"]


def test_budget_guard_blocks_41st_agent_turn() -> None:
    source = f"""
import plugin from {json.dumps(str(PLUGIN_ENTRY))};
const handlers = {{}};
const api = {{
  pluginConfig: {{
    enabled: true,
    maxToolCalls: 0,
    maxAgentTurns: 40,
    agentIds: ["agent-a"],
    sessionKeyPrefixes: ["agentic-math"],
    blockReasonPrefix: "NEJUMI_BUDGET_GUARD_BLOCKED",
  }},
  on(name, handler) {{ handlers[name] = handler; }},
}};
plugin.register(api);
let blocked = null;
for (let i = 1; i <= 41; i++) {{
  const result = await handlers.before_agent_run(
    {{}},
    {{ agentId: "agent-a", sessionKey: "agentic-math:task:attempt", runId: "run-turn" }},
  );
  if (result?.block) blocked = {{ i, result }};
}}
console.log(JSON.stringify(blocked));
"""
    blocked = run_node_probe(source)
    assert blocked["i"] == 41
    assert blocked["result"]["block"] is True
    assert "agent_turn_limit_exceeded" in blocked["result"]["blockReason"]
    assert "observed=41" in blocked["result"]["blockReason"]
    assert "limit=40" in blocked["result"]["blockReason"]


def test_budget_guard_scope_does_not_count_other_agents() -> None:
    source = f"""
import plugin from {json.dumps(str(PLUGIN_ENTRY))};
const handlers = {{}};
const api = {{
  pluginConfig: {{
    enabled: true,
    maxToolCalls: 1,
    maxAgentTurns: 0,
    agentIds: ["agent-a"],
    sessionKeyPrefixes: ["swebench-pro"],
  }},
  on(name, handler) {{ handlers[name] = handler; }},
}};
plugin.register(api);
const other = await handlers.before_tool_call(
  {{ toolName: "read", toolCallId: "other-1" }},
  {{ agentId: "agent-b", sessionKey: "swebench-pro:task:attempt", runId: "run-other" }},
);
const first = await handlers.before_tool_call(
  {{ toolName: "read", toolCallId: "main-1" }},
  {{ agentId: "agent-a", sessionKey: "swebench-pro:task:attempt", runId: "run-main" }},
);
const second = await handlers.before_tool_call(
  {{ toolName: "read", toolCallId: "main-2" }},
  {{ agentId: "agent-a", sessionKey: "swebench-pro:task:attempt", runId: "run-main" }},
);
console.log(JSON.stringify({{ other: other ?? null, first: first ?? null, second }}));
"""
    result = run_node_probe(source)
    assert result["other"] is None
    assert result["first"] is None
    assert result["second"]["block"] is True


def test_budget_guard_counts_by_session_key_before_run_id() -> None:
    source = f"""
import plugin from {json.dumps(str(PLUGIN_ENTRY))};
const handlers = {{}};
const api = {{
  pluginConfig: {{
    enabled: true,
    maxToolCalls: 1,
    maxAgentTurns: 0,
    agentIds: ["agent-a"],
    sessionKeyPrefixes: ["swebench-pro"],
  }},
  on(name, handler) {{ handlers[name] = handler; }},
}};
plugin.register(api);
const firstSessionFirst = await handlers.before_tool_call(
  {{ toolName: "read", toolCallId: "s1-1" }},
  {{ agentId: "agent-a", sessionKey: "swebench-pro:task-1", runId: "shared-wandb-run" }},
);
const secondSessionFirst = await handlers.before_tool_call(
  {{ toolName: "read", toolCallId: "s2-1" }},
  {{ agentId: "agent-a", sessionKey: "swebench-pro:task-2", runId: "shared-wandb-run" }},
);
const firstSessionSecond = await handlers.before_tool_call(
  {{ toolName: "read", toolCallId: "s1-2" }},
  {{ agentId: "agent-a", sessionKey: "swebench-pro:task-1", runId: "shared-wandb-run" }},
);
console.log(JSON.stringify({{
  firstSessionFirst: firstSessionFirst ?? null,
  secondSessionFirst: secondSessionFirst ?? null,
  firstSessionSecond,
}}));
"""
    result = run_node_probe(source)
    assert result["firstSessionFirst"] is None
    assert result["secondSessionFirst"] is None
    assert result["firstSessionSecond"]["block"] is True
