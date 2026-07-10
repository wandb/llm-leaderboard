from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ENTRY = REPO_ROOT / "openclaw-plugins" / "nejumi-budget-guard" / "dist" / "index.js"


def run_node_probe(source: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="nejumi-budget-guard-test-") as state_dir:
        result = subprocess.run(
            ["node", "--input-type=module", "-e", source],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
            env={
                **dict(os.environ),
                "OPENCLAW_NEJUMI_BUDGET_GUARD_STATE_DIR": state_dir,
            },
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


def test_budget_guard_registers_trusted_tool_policy() -> None:
    source = f"""
import plugin from {json.dumps(str(PLUGIN_ENTRY))};
const handlers = {{}};
const policies = {{}};
const api = {{
  pluginConfig: {{
    enabled: true,
    maxToolCalls: 1,
    maxAgentTurns: 0,
    agentIds: ["agent-a"],
    sessionKeyPrefixes: ["swebench-pro"],
    blockReasonPrefix: "NEJUMI_BUDGET_GUARD_BLOCKED",
  }},
  on(name, handler) {{ handlers[name] = handler; }},
  registerTrustedToolPolicy(policy) {{ policies[policy.id] = policy; }},
}};
plugin.register(api);
const first = await policies["budget-guard"].evaluate(
  {{ toolName: "read", toolCallId: "call-1" }},
  {{ agentId: "agent-a", sessionKey: "swebench-pro:task:attempt", runId: "run-tool" }},
);
const second = await policies["budget-guard"].evaluate(
  {{ toolName: "read", toolCallId: "call-2" }},
  {{ agentId: "agent-a", sessionKey: "swebench-pro:task:attempt", runId: "run-tool" }},
);
console.log(JSON.stringify({{
  registered: Object.keys(policies),
  first: first ?? null,
  second,
}}));
"""
    result = run_node_probe(source)
    assert result["registered"] == ["budget-guard"]
    assert result["first"] is None
    assert result["second"]["block"] is True
    assert "tool_call_limit_exceeded" in result["second"]["blockReason"]


def test_budget_guard_deduplicates_trusted_policy_and_before_tool_hook() -> None:
    source = f"""
import plugin from {json.dumps(str(PLUGIN_ENTRY))};
const handlers = {{}};
const policies = {{}};
const api = {{
  pluginConfig: {{
    enabled: true,
    maxToolCalls: 1,
    maxAgentTurns: 0,
    agentIds: ["agent-a"],
    sessionKeyPrefixes: ["swebench-pro"],
    blockReasonPrefix: "NEJUMI_BUDGET_GUARD_BLOCKED",
  }},
  on(name, handler) {{ handlers[name] = handler; }},
  registerTrustedToolPolicy(policy) {{ policies[policy.id] = policy; }},
}};
plugin.register(api);
const ctx = {{ agentId: "agent-a", sessionKey: "swebench-pro:task:attempt", runId: "run-tool" }};
const event = {{ toolName: "read", toolCallId: "call-1" }};
const trusted = await policies["budget-guard"].evaluate(event, ctx);
const hook = await handlers.before_tool_call(event, ctx);
const second = await policies["budget-guard"].evaluate(
  {{ toolName: "read", toolCallId: "call-2" }},
  ctx,
);
console.log(JSON.stringify({{
  trusted: trusted ?? null,
  hook: hook ?? null,
  second,
}}));
"""
    result = run_node_probe(source)
    assert result["trusted"] is None
    assert result["hook"] is None
    assert result["second"]["block"] is True
    assert "observed=2" in result["second"]["blockReason"]


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
  if (result?.outcome === "block") blocked = {{ i, result }};
}}
console.log(JSON.stringify(blocked));
"""
    blocked = run_node_probe(source)
    assert blocked["i"] == 41
    assert blocked["result"]["outcome"] == "block"
    assert "agent_turn_limit_exceeded" in blocked["result"]["reason"]
    assert "observed=41" in blocked["result"]["reason"]
    assert "limit=40" in blocked["result"]["reason"]


def test_budget_guard_audits_before_agent_reply_hook_coverage() -> None:
    source = f"""
import fs from "node:fs";
import path from "node:path";
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
const result = await handlers.before_agent_reply(
  {{ cleanedBody: "solve this" }},
  {{ agentId: "agent-a", sessionKey: "agentic-math:task:attempt", runId: "run-reply" }},
);
const auditPath = path.join(process.env.OPENCLAW_NEJUMI_BUDGET_GUARD_STATE_DIR, "audit.jsonl");
const audit = fs.readFileSync(auditPath, "utf8").trim().split("\\n").map((line) => JSON.parse(line));
console.log(JSON.stringify({{ result: result ?? null, audit }}));
"""
    result = run_node_probe(source)
    assert result["result"] is None
    assert result["audit"][-1]["phase"] == "before_agent_reply_seen"
    assert result["audit"][-1]["scoped"] is True
    assert result["audit"][-1]["cleanedBodyLength"] == len("solve this")


def test_budget_guard_blocks_next_agent_run_after_tool_cap() -> None:
    source = f"""
import plugin from {json.dumps(str(PLUGIN_ENTRY))};
const handlers = {{}};
const api = {{
  pluginConfig: {{
    enabled: true,
    maxToolCalls: 1,
    maxAgentTurns: 40,
    agentIds: ["agent-a"],
    sessionKeyPrefixes: ["swebench-pro"],
    blockReasonPrefix: "NEJUMI_BUDGET_GUARD_BLOCKED",
  }},
  on(name, handler) {{ handlers[name] = handler; }},
}};
plugin.register(api);
await handlers.before_agent_run(
  {{}},
  {{ agentId: "agent-a", sessionKey: "agent:agent-a:swebench-pro:task:attempt", runId: "run-main" }},
);
const firstTool = await handlers.before_tool_call(
  {{ toolName: "read", toolCallId: "tool-1" }},
  {{ agentId: "agent-a", sessionKey: "agent:agent-a:swebench-pro:task:attempt", runId: "run-main" }},
);
const blockedTool = await handlers.before_tool_call(
  {{ toolName: "read", toolCallId: "tool-2" }},
  {{ agentId: "agent-a", sessionKey: "agent:agent-a:swebench-pro:task:attempt", runId: "run-main" }},
);
const blockedAgentRun = await handlers.before_agent_run(
  {{}},
  {{ agentId: "agent-a", sessionKey: "agent:agent-a:swebench-pro:task:attempt", runId: "run-main" }},
);
console.log(JSON.stringify({{
  firstTool: firstTool ?? null,
  blockedTool,
  blockedAgentRun,
}}));
"""
    result = run_node_probe(source)
    assert result["firstTool"] is None
    assert result["blockedTool"]["block"] is True
    assert "tool_call_limit_exceeded" in result["blockedTool"]["blockReason"]
    assert result["blockedAgentRun"]["outcome"] == "block"
    assert "tool_call_limit_exceeded" in result["blockedAgentRun"]["reason"]


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


def test_budget_guard_scopes_by_session_key_when_agent_id_is_missing() -> None:
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
const first = await handlers.before_tool_call(
  {{ toolName: "read", toolCallId: "main-1" }},
  {{ sessionKey: "swebench-pro:task:attempt", runId: "run-main" }},
);
const second = await handlers.before_tool_call(
  {{ toolName: "read", toolCallId: "main-2" }},
  {{ sessionKey: "swebench-pro:task:attempt", runId: "run-main" }},
);
const other = await handlers.before_tool_call(
  {{ toolName: "read", toolCallId: "other-1" }},
  {{ sessionKey: "other-benchmark:task:attempt", runId: "run-other" }},
);
console.log(JSON.stringify({{ first: first ?? null, second, other: other ?? null }}));
"""
    result = run_node_probe(source)
    assert result["first"] is None
    assert result["second"]["block"] is True
    assert result["other"] is None


def test_budget_guard_matches_openclaw_agent_prefixed_session_key() -> None:
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
const first = await handlers.before_tool_call(
  {{ toolName: "read", toolCallId: "main-1" }},
  {{ sessionKey: "agent:agent-a:swebench-pro:task:attempt", runId: "run-main" }},
);
const second = await handlers.before_tool_call(
  {{ toolName: "read", toolCallId: "main-2" }},
  {{ sessionKey: "agent:agent-a:swebench-pro:task:attempt", runId: "run-main" }},
);
console.log(JSON.stringify({{ first: first ?? null, second }}));
"""
    result = run_node_probe(source)
    assert result["first"] is None
    assert result["second"]["block"] is True


def test_budget_guard_matches_openclaw_lowercased_session_key_prefix() -> None:
    source = f"""
import plugin from {json.dumps(str(PLUGIN_ENTRY))};
const handlers = {{}};
const policies = {{}};
const api = {{
  pluginConfig: {{
    enabled: true,
    maxToolCalls: 1,
    maxAgentTurns: 0,
    agentIds: ["agent-a"],
    sessionKeyPrefixes: ["budget-guard-probe-20260707T200549"],
  }},
  on(name, handler) {{ handlers[name] = handler; }},
  registerTrustedToolPolicy(policy) {{ policies[policy.id] = policy; }},
}};
plugin.register(api);
const ctx = {{
  agentId: "agent-a",
  sessionKey: "agent:agent-a:budget-guard-probe-20260707t200549:task:attempt",
  runId: "run-main",
}};
const first = await policies["budget-guard"].evaluate(
  {{ toolName: "exec", toolCallId: "call-1" }},
  ctx,
);
const second = await policies["budget-guard"].evaluate(
  {{ toolName: "exec", toolCallId: "call-2" }},
  ctx,
);
console.log(JSON.stringify({{ first: first ?? null, second }}));
"""
    result = run_node_probe(source)
    assert result["first"] is None
    assert result["second"]["block"] is True
    assert "tool_call_limit_exceeded" in result["second"]["blockReason"]


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


def test_budget_guard_persists_tool_count_across_processes(tmp_path: Path) -> None:
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
const result = await handlers.before_tool_call(
  {{ toolName: "read", toolCallId: process.env.TOOL_CALL_ID }},
  {{ agentId: "agent-a", sessionKey: "swebench-pro:process-persistence", runId: "run-main" }},
);
console.log(JSON.stringify(result ?? null));
"""
    env = dict(os.environ)
    env["TOOL_CALL_ID"] = "call-1"
    first = subprocess.run(
        ["node", "--input-type=module", "-e", source],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={**env, "OPENCLAW_NEJUMI_BUDGET_GUARD_STATE_DIR": str(tmp_path)},
    )
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout) is None

    env["TOOL_CALL_ID"] = "call-2"
    second = subprocess.run(
        ["node", "--input-type=module", "-e", source],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={**env, "OPENCLAW_NEJUMI_BUDGET_GUARD_STATE_DIR": str(tmp_path)},
    )
    assert second.returncode == 0, second.stderr
    blocked = json.loads(second.stdout)
    assert blocked["block"] is True
    assert "tool_call_limit_exceeded" in blocked["blockReason"]


def test_budget_guard_prefers_live_openclaw_config_over_static_plugin_config(tmp_path: Path) -> None:
    config_path = tmp_path / "openclaw.json"
    config_path.write_text(
        json.dumps(
            {
                "plugins": {
                    "entries": {
                        "nejumi-budget-guard": {
                            "config": {
                                "enabled": True,
                                "maxToolCalls": 1,
                                "maxAgentTurns": 0,
                                "agentIds": ["agent-current"],
                                "sessionKeyPrefixes": ["swebench-pro-current"],
                            }
                        }
                    }
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source = f"""
import plugin from {json.dumps(str(PLUGIN_ENTRY))};
const handlers = {{}};
const api = {{
  pluginConfig: {{
    enabled: true,
    maxToolCalls: 1,
    maxAgentTurns: 0,
    agentIds: ["agent-stale"],
    sessionKeyPrefixes: ["stale-session"],
  }},
  on(name, handler) {{ handlers[name] = handler; }},
}};
plugin.register(api);
await handlers.before_tool_call(
  {{ toolName: "read", toolCallId: "call-1" }},
  {{ agentId: "agent-current", sessionKey: "swebench-pro-current:task", runId: "run-main" }},
);
const second = await handlers.before_tool_call(
  {{ toolName: "read", toolCallId: "call-2" }},
  {{ agentId: "agent-current", sessionKey: "swebench-pro-current:task", runId: "run-main" }},
);
console.log(JSON.stringify(second ?? null));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", source],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={
            **dict(os.environ),
            "OPENCLAW_CONFIG_PATH": str(config_path),
            "OPENCLAW_NEJUMI_BUDGET_GUARD_STATE_DIR": str(tmp_path / "state"),
        },
    )
    assert result.returncode == 0, result.stderr
    blocked = json.loads(result.stdout)
    assert blocked["block"] is True
    assert "tool_call_limit_exceeded" in blocked["blockReason"]
