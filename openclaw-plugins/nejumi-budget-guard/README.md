# Nejumi Budget Guard

OpenClaw plugin for Nejumi Taiwan agentic benchmarks.

It enforces hard pre-execution limits through OpenClaw plugin hooks:

- `before_tool_call`: allows calls `1..maxToolCalls` and blocks the next call.
- `before_agent_run`: allows turns `1..maxAgentTurns` and blocks the next model turn.

The block reason starts with `NEJUMI_BUDGET_GUARD_BLOCKED` so the harness can
detect a budget stop from OpenClaw session logs.

Install locally:

```bash
openclaw plugins install ./openclaw-plugins/nejumi-budget-guard --force
```

For NeMoClaw, install inside the target sandbox too:

```bash
NEMOCLAW_SANDBOX=nejumi-taiwan scripts/setup/install_openclaw_budget_guard.sh
```
