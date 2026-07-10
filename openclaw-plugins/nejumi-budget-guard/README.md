# Nejumi Budget Guard

OpenClaw plugin for Nejumi Taiwan agentic benchmarks.

It enforces hard pre-execution limits through OpenClaw plugin hooks:

- `before_tool_call`: allows calls `1..maxToolCalls` and blocks the next call.
- `before_agent_run`: records the outer run and blocks a later run after a
  previous tool budget block. OpenClaw 2026.6.10 does not call this hook for
  every internal tool-continuation turn, so `maxAgentTurns` still needs live
  runner enforcement.
- `before_agent_reply`: audit-only coverage probe for whether the OpenClaw
  runtime exposes a pre-model reply hook on a given path.

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
