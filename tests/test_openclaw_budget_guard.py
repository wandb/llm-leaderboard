from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ENTRY = REPO_ROOT / "openclaw-plugins" / "nejumi-budget-guard" / "dist" / "index.js"
LANDLOCK_LAUNCHER = (
    REPO_ROOT / "openclaw-plugins" / "nejumi-budget-guard" / "landlock_exec.py"
)


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


def test_budget_guard_blocks_denied_exec_argument_before_execution() -> None:
    source = f"""
import plugin from {json.dumps(str(PLUGIN_ENTRY))};
const handlers = {{}};
const policies = {{}};
const api = {{
  pluginConfig: {{
    enabled: true,
    maxToolCalls: 60,
    maxAgentTurns: 60,
    agentIds: ["agent-a"],
    sessionKeyPrefixes: ["deepswe"],
    denyTools: ["web_search"],
    denyArgumentPatterns: [String.raw`\\b(curl|wget)\\b`, String.raw`https?://`],
    blockReasonPrefix: "NEJUMI_BUDGET_GUARD_BLOCKED",
  }},
  on(name, handler) {{ handlers[name] = handler; }},
  registerTrustedToolPolicy(policy) {{ policies[policy.id] = policy; }},
}};
plugin.register(api);
const result = await policies["budget-guard"].evaluate(
  {{
    toolName: "exec",
    toolCallId: "call-remote",
    arguments: {{ command: "curl -sL https://example.com/archive.tar.gz -o /tmp/archive.tar.gz" }},
  }},
  {{ agentId: "agent-a", sessionKey: "deepswe:task:attempt", runId: "run-policy" }},
);
console.log(JSON.stringify(result));
"""
    result = run_node_probe(source)
    assert result["block"] is True
    assert "tool_policy_violation" in result["blockReason"]
    assert "type=denied_argument_pattern" in result["blockReason"]
    assert "toolName=exec" in result["blockReason"]


def test_budget_guard_does_not_block_url_text_in_non_executable_tool() -> None:
    source = f"""
import plugin from {json.dumps(str(PLUGIN_ENTRY))};
const policies = {{}};
const api = {{
  pluginConfig: {{
    enabled: true,
    maxToolCalls: 60,
    maxAgentTurns: 0,
    agentIds: ["agent-a"],
    sessionKeyPrefixes: ["deepswe"],
    denyArgumentPatterns: [String.raw`https?://`],
    blockReasonPrefix: "NEJUMI_BUDGET_GUARD_BLOCKED",
  }},
  on() {{}},
  registerTrustedToolPolicy(policy) {{ policies[policy.id] = policy; }},
}};
plugin.register(api);
const result = await policies["budget-guard"].evaluate(
  {{
    toolName: "write_file",
    toolCallId: "call-write-url",
    arguments: {{ path: "README.md", content: "See https://example.com/docs" }},
  }},
  {{ agentId: "agent-a", sessionKey: "deepswe:task:attempt", runId: "run-policy" }},
);
console.log(JSON.stringify({{ result: result ?? null }}));
"""
    result = run_node_probe(source)
    assert result["result"] is None


def test_budget_guard_blocks_denied_tool_by_wildcard() -> None:
    source = f"""
import plugin from {json.dumps(str(PLUGIN_ENTRY))};
const handlers = {{}};
const api = {{
  pluginConfig: {{
    enabled: true,
    maxToolCalls: 0,
    maxAgentTurns: 0,
    agentIds: ["agent-a"],
    sessionKeyPrefixes: ["deepswe"],
    denyTools: ["browser_*"],
    blockReasonPrefix: "NEJUMI_BUDGET_GUARD_BLOCKED",
  }},
  on(name, handler) {{ handlers[name] = handler; }},
}};
plugin.register(api);
const result = await handlers.before_tool_call(
  {{ toolName: "browser_fetch", toolCallId: "call-browser" }},
  {{ agentId: "agent-a", sessionKey: "deepswe:task:attempt", runId: "run-policy" }},
);
const blockedAgentRun = await handlers.before_agent_run(
  {{}},
  {{ agentId: "agent-a", sessionKey: "deepswe:task:attempt", runId: "run-policy" }},
);
console.log(JSON.stringify({{ result, blockedAgentRun }}));
"""
    result = run_node_probe(source)
    assert result["result"]["block"] is True
    assert "type=denied_tool" in result["result"]["blockReason"]
    assert result["blockedAgentRun"]["outcome"] == "block"
    assert "tool_policy_limit_exceeded" in result["blockedAgentRun"]["reason"]


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


def test_budget_guard_injects_staged_english_warnings_into_tool_results() -> None:
    source = f"""
import plugin from {json.dumps(str(PLUGIN_ENTRY))};
const handlers = {{}};
const api = {{
  pluginConfig: {{
    enabled: true,
    maxToolCalls: 40,
    maxAgentTurns: 40,
    agentIds: ["agent-a"],
    sessionKeyPrefixes: ["swebench-lite"],
    blockReasonPrefix: "NEJUMI_BUDGET_GUARD_BLOCKED",
  }},
  on(name, handler) {{ handlers[name] = handler; }},
}};
plugin.register(api);
const ctx = {{ agentId: "agent-a", sessionKey: "swebench-lite:task:attempt", runId: "run-main" }};
const warnings = [];
for (let i = 1; i <= 38; i++) {{
  await handlers.before_tool_call(
    {{ toolName: "exec", toolCallId: `tool-${{i}}` }},
    ctx,
  );
  const result = handlers.tool_result_persist(
    {{
      toolName: "exec",
      toolCallId: `tool-${{i}}`,
      message: {{ role: "toolResult", content: [{{ type: "text", text: "ok" }}] }},
    }},
    ctx,
  );
  if (result instanceof Promise) throw new Error("tool_result_persist must be synchronous");
  if (result?.message) warnings.push({{ i, text: result.message.content.at(-1).text }});
}}
console.log(JSON.stringify(warnings));
"""
    warnings = run_node_probe(source)
    assert [item["i"] for item in warnings] == [20, 30, 35, 38]
    assert all("NEJUMI RUNTIME BUDGET WARNING" in item["text"] for item in warnings)
    assert "20 remain" in warnings[0]["text"]
    assert "2 remain" in warnings[-1]["text"]
    assert "current git diff is submitted" in warnings[-1]["text"]
    assert "No extra cleanup turn is guaranteed" in warnings[-1]["text"]


def test_budget_guard_injects_staged_agent_turn_warnings() -> None:
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
  }},
  on(name, handler) {{ handlers[name] = handler; }},
}};
plugin.register(api);
const ctx = {{ agentId: "agent-a", sessionKey: "agentic-math:task:attempt", runId: "run-main" }};
const warnings = [];
for (let i = 1; i <= 38; i++) {{
  const prepared = await handlers.agent_turn_prepare({{ prompt: "solve", messages: [] }}, ctx);
  if (prepared?.appendContext) warnings.push({{ i, text: prepared.appendContext }});
  await handlers.before_agent_run({{}}, ctx);
}}
console.log(JSON.stringify(warnings));
"""
    warnings = run_node_probe(source)
    assert [item["i"] for item in warnings] == [20, 30, 35, 38]
    assert "20 turns remain after this one" in warnings[0]["text"]
    assert "2 turns remain after this one" in warnings[-1]["text"]
    assert "current answer is submitted" in warnings[-1]["text"]


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


def test_budget_guard_reads_live_config_from_openclaw_state_dir(tmp_path: Path) -> None:
    state_dir = tmp_path / ".openclaw"
    state_dir.mkdir()
    config_path = state_dir / "openclaw.json"
    config_path.write_text(
        json.dumps(
            {
                "plugins": {
                    "entries": {
                        "nejumi-budget-guard": {
                            "enabled": True,
                            "config": {
                                "enabled": True,
                                "maxToolCalls": 1,
                                "maxAgentTurns": 0,
                                "agentIds": ["agent-current"],
                            },
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
const staleConfig = {{
  enabled: true,
  maxToolCalls: 10,
  maxAgentTurns: 0,
  agentIds: ["agent-stale"],
}};
const api = {{
  pluginConfig: staleConfig,
  on(name, handler) {{ handlers[name] = handler; }},
}};
plugin.register(api);
const ctx = {{ agentId: "agent-current", sessionKey: "agent:agent-current:test", runId: "run-main" }};
await handlers.before_tool_call(
  {{ toolName: "read", toolCallId: "call-1" }},
  {{ ...ctx, config: {{ plugins: {{ entries: {{ "nejumi-budget-guard": {{ config: staleConfig }} }} }} }} }},
);
const second = await handlers.before_tool_call(
  {{ toolName: "read", toolCallId: "call-2" }},
  {{ ...ctx, config: {{ plugins: {{ entries: {{ "nejumi-budget-guard": {{ config: staleConfig }} }} }} }} }},
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
            "OPENCLAW_HOME": str(tmp_path / "different-home"),
            "OPENCLAW_STATE_DIR": str(state_dir),
            "OPENCLAW_NEJUMI_BUDGET_GUARD_STATE_DIR": str(tmp_path / "guard-state"),
        },
    )
    assert result.returncode == 0, result.stderr
    blocked = json.loads(result.stdout)
    assert blocked["block"] is True
    assert "tool_call_limit_exceeded" in blocked["blockReason"]


def test_budget_guard_reads_explicit_live_config_path_from_stale_config(tmp_path: Path) -> None:
    live_dir = tmp_path / "sandbox" / ".openclaw"
    live_dir.mkdir(parents=True)
    config_path = live_dir / "openclaw.json"
    config_path.write_text(
        json.dumps(
            {
                "plugins": {
                    "entries": {
                        "nejumi-budget-guard": {
                            "config": {
                                "enabled": True,
                                "liveConfigPath": str(config_path),
                                "maxToolCalls": 1,
                                "agentIds": ["agent-current"],
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
const staleConfig = {{
  enabled: true,
  liveConfigPath: {json.dumps(str(config_path))},
  maxToolCalls: 10,
  agentIds: ["agent-stale"],
}};
const api = {{
  pluginConfig: staleConfig,
  on(name, handler) {{ handlers[name] = handler; }},
}};
plugin.register(api);
const ctx = {{ agentId: "agent-current", sessionKey: "agent:agent-current:test", runId: "run-main" }};
await handlers.before_tool_call({{ toolName: "read", toolCallId: "call-1" }}, ctx);
const second = await handlers.before_tool_call({{ toolName: "read", toolCallId: "call-2" }}, ctx);
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
            "OPENCLAW_HOME": str(tmp_path / "wrong-home"),
            "OPENCLAW_NEJUMI_BUDGET_GUARD_STATE_DIR": str(tmp_path / "guard-state"),
        },
    )
    assert result.returncode == 0, result.stderr
    blocked = json.loads(result.stdout)
    assert blocked["block"] is True
    assert "tool_call_limit_exceeded" in blocked["blockReason"]


def test_workspace_guard_blocks_sibling_path_but_allows_next_workspace_read(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "task-a"
    sibling = tmp_path / "task-b"
    private_tmp = tmp_path / "task-a-tmp"
    private_home = tmp_path / "task-a-home"
    for directory in (workspace, sibling, private_tmp, private_home):
        directory.mkdir()
    (workspace / "visible.txt").write_text("own task\n", encoding="utf-8")
    (sibling / "secret.txt").write_text("other task\n", encoding="utf-8")
    source = f"""
import plugin from {json.dumps(str(PLUGIN_ENTRY))};
const handlers = {{}};
const api = {{
  pluginConfig: {{
    enabled: true,
    maxToolCalls: 10,
    maxAgentTurns: 0,
    agentIds: ["agent-a"],
    workspaceIsolationEnabled: true,
    agentWorkspaces: {{
      "agent-a": {{
        workspace: {json.dumps(str(workspace))},
        tmp: {json.dumps(str(private_tmp))},
        home: {json.dumps(str(private_home))},
        readOnlyRoots: [],
      }},
    }},
  }},
  on(name, handler) {{ handlers[name] = handler; }},
}};
plugin.register(api);
const ctx = {{ agentId: "agent-a", sessionKey: "swe:task-a", runId: "run-a" }};
const blocked = await handlers.before_tool_call(
  {{ toolName: "read", toolCallId: "read-other", params: {{ path: {json.dumps(str(sibling / "secret.txt"))} }} }},
  ctx,
);
const allowed = await handlers.before_tool_call(
  {{ toolName: "read", toolCallId: "read-own", params: {{ path: "visible.txt" }} }},
  ctx,
);
console.log(JSON.stringify({{ blocked, allowed: allowed ?? null }}));
"""
    result = run_node_probe(source)
    assert result["blocked"]["block"] is True
    assert "cross_task_filesystem_access" in result["blocked"]["blockReason"]
    assert result["allowed"] is None


def test_workspace_guard_rewrites_exec_through_landlock_launcher(tmp_path: Path) -> None:
    workspace = tmp_path / "task-a"
    private_tmp = tmp_path / "task-a-tmp"
    private_home = tmp_path / "task-a-home"
    workspace.mkdir()
    source = f"""
import plugin from {json.dumps(str(PLUGIN_ENTRY))};
const handlers = {{}};
const api = {{
  pluginConfig: {{
    enabled: true,
    maxToolCalls: 10,
    maxAgentTurns: 0,
    agentIds: ["agent-a"],
    workspaceIsolationEnabled: true,
    agentWorkspaces: {{
      "agent-a": {{
        workspace: {json.dumps(str(workspace))},
        tmp: {json.dumps(str(private_tmp))},
        home: {json.dumps(str(private_home))},
        readOnlyRoots: [],
      }},
    }},
  }},
  on(name, handler) {{ handlers[name] = handler; }},
}};
plugin.register(api);
const result = await handlers.before_tool_call(
  {{ toolName: "exec", toolCallId: "exec-own", params: {{ command: "pwd && git status --short" }} }},
  {{ agentId: "agent-a", sessionKey: "swe:task-a", runId: "run-a" }},
);
console.log(JSON.stringify(result));
"""
    result = run_node_probe(source)
    command = result["params"]["command"]
    assert str(LANDLOCK_LAUNCHER) in command
    assert f"--workspace {workspace!s}" not in command  # shell-quoted arguments
    assert str(workspace) in command
    assert str(private_tmp) in command
    assert "pwd && git status --short" in command


def test_landlock_launcher_denies_sibling_and_shared_tmp_reads(tmp_path: Path) -> None:
    workspace = tmp_path / "task-a"
    sibling = tmp_path / "task-b"
    private_tmp = tmp_path / "private-tmp"
    private_home = tmp_path / "private-home"
    for directory in (workspace, sibling, private_tmp, private_home):
        directory.mkdir()
    own_file = workspace / "own.txt"
    sibling_file = sibling / "secret.txt"
    shared_tmp_file = Path("/tmp") / f"nejumi-cross-task-{os.getpid()}.txt"
    own_file.write_text("own\n", encoding="utf-8")
    sibling_file.write_text("other\n", encoding="utf-8")
    shared_tmp_file.write_text("shared\n", encoding="utf-8")
    try:
        allowed = subprocess.run(
            [
                "python3",
                str(LANDLOCK_LAUNCHER),
                "--workspace",
                str(workspace),
                "--tmp",
                str(private_tmp),
                "--home",
                str(private_home),
                "--",
                "/bin/bash",
                "-lc",
                "cat own.txt; printf private > \"$TMPDIR/result.txt\"",
            ],
            text=True,
            capture_output=True,
            check=False,
            env={
                **os.environ,
                "HOME": str(private_home),
                "TMPDIR": str(private_tmp),
            },
        )
        assert allowed.returncode == 0, allowed.stderr
        assert allowed.stdout == "own\n"
        assert (private_tmp / "result.txt").read_text(encoding="utf-8") == "private"

        for forbidden in (sibling_file, shared_tmp_file):
            denied = subprocess.run(
                [
                    "python3",
                    str(LANDLOCK_LAUNCHER),
                    "--workspace",
                    str(workspace),
                    "--tmp",
                    str(private_tmp),
                    "--home",
                    str(private_home),
                    "--",
                    "/bin/bash",
                    "-lc",
                    f"cat {forbidden}",
                ],
                text=True,
                capture_output=True,
                check=False,
                env={
                    **os.environ,
                    "HOME": str(private_home),
                    "TMPDIR": str(private_tmp),
                },
            )
            assert denied.returncode != 0
            assert "Permission denied" in denied.stderr
            assert "other" not in denied.stdout
            assert "shared" not in denied.stdout
    finally:
        shared_tmp_file.unlink(missing_ok=True)
