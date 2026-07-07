const PLUGIN_ID = "nejumi-budget-guard";
const GLOBAL_KEY = Symbol.for("nejumi.openclaw.budgetGuard.v1");

function sharedState() {
  const root = globalThis;
  if (!root[GLOBAL_KEY]) {
    Object.defineProperty(root, GLOBAL_KEY, {
      value: {
        runs: new Map(),
      },
      writable: false,
      enumerable: false,
      configurable: true,
    });
  }
  return root[GLOBAL_KEY];
}

function asPositiveInteger(value) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.max(0, Math.trunc(value));
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return Math.max(0, Math.trunc(parsed));
    }
  }
  return 0;
}

function asStringArray(value) {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((item) => String(item)).filter((item) => item.length > 0);
}

function normalizeConfig(config) {
  const raw = config && typeof config === "object" ? config : {};
  return {
    enabled: raw.enabled !== false,
    maxToolCalls: asPositiveInteger(raw.maxToolCalls),
    maxAgentTurns: asPositiveInteger(raw.maxAgentTurns),
    agentIds: asStringArray(raw.agentIds),
    sessionKeyPrefixes: asStringArray(raw.sessionKeyPrefixes),
    blockReasonPrefix: String(raw.blockReasonPrefix || "NEJUMI_BUDGET_GUARD_BLOCKED"),
  };
}

function contextValue(event, ctx, key) {
  if (ctx && typeof ctx === "object" && ctx[key] != null) {
    return String(ctx[key]);
  }
  if (event && typeof event === "object" && event[key] != null) {
    return String(event[key]);
  }
  const context = event && typeof event === "object" ? event.context : null;
  if (context && typeof context === "object" && context[key] != null) {
    return String(context[key]);
  }
  return "";
}

function runKey(event, ctx) {
  return (
    contextValue(event, ctx, "sessionKey") ||
    contextValue(event, ctx, "sessionId") ||
    contextValue(event, ctx, "runId") ||
    contextValue(event, ctx, "agentId") ||
    "global"
  );
}

function sessionKey(event, ctx) {
  return contextValue(event, ctx, "sessionKey");
}

function agentId(event, ctx) {
  return contextValue(event, ctx, "agentId");
}

function scoped(config, event, ctx) {
  if (!config.enabled) {
    return false;
  }
  const currentAgentId = agentId(event, ctx);
  if (config.agentIds.length > 0 && !config.agentIds.includes(currentAgentId)) {
    return false;
  }
  const currentSessionKey = sessionKey(event, ctx);
  if (
    config.sessionKeyPrefixes.length > 0 &&
    !config.sessionKeyPrefixes.some((prefix) => currentSessionKey.startsWith(prefix))
  ) {
    return false;
  }
  return true;
}

function runState(key) {
  const state = sharedState();
  let value = state.runs.get(key);
  if (!value) {
    value = {
      toolCalls: 0,
      agentTurns: 0,
      seenToolCallIds: new Set(),
      blocked: false,
    };
    state.runs.set(key, value);
  }
  return value;
}

function toolCallId(event, ctx) {
  return (
    contextValue(event, ctx, "toolCallId") ||
    contextValue(event, ctx, "tool_call_id") ||
    ""
  );
}

function toolName(event) {
  if (!event || typeof event !== "object") {
    return "";
  }
  return String(event.toolName || event.name || "");
}

function blockResult(config, kind, observed, limit, event, ctx) {
  const reason = [
    config.blockReasonPrefix,
    `${kind}_limit_exceeded`,
    `observed=${observed}`,
    `limit=${limit}`,
    `agentId=${agentId(event, ctx) || "unknown"}`,
    `sessionKey=${sessionKey(event, ctx) || "unknown"}`,
    `toolName=${toolName(event) || "n/a"}`,
  ].join(" ");
  return {
    block: true,
    blockReason: reason,
  };
}

function pluginConfig(api, event) {
  const eventConfig =
    event && typeof event === "object" && event.context && typeof event.context === "object"
      ? event.context.pluginConfig
      : undefined;
  return normalizeConfig(eventConfig || api.pluginConfig || {});
}

export default {
  id: PLUGIN_ID,
  name: "Nejumi Budget Guard",
  description: "Blocks OpenClaw agent runs before tool or model-turn budgets are exceeded.",
  register(api) {
    api.on(
      "before_agent_run",
      async (event, ctx) => {
        const config = pluginConfig(api, event);
        if (!scoped(config, event, ctx) || config.maxAgentTurns <= 0) {
          return;
        }
        const state = runState(runKey(event, ctx));
        const nextAgentTurn = state.agentTurns + 1;
        if (nextAgentTurn > config.maxAgentTurns) {
          state.blocked = true;
          return blockResult(config, "agent_turn", nextAgentTurn, config.maxAgentTurns, event, ctx);
        }
        state.agentTurns = nextAgentTurn;
      },
      { priority: 10000, timeoutMs: 1000 },
    );

    api.on(
      "before_tool_call",
      async (event, ctx) => {
        const config = pluginConfig(api, event);
        if (!scoped(config, event, ctx) || config.maxToolCalls <= 0) {
          return;
        }
        const state = runState(runKey(event, ctx));
        const id = toolCallId(event, ctx);
        if (id && state.seenToolCallIds.has(id)) {
          return;
        }
        const nextToolCall = state.toolCalls + 1;
        if (nextToolCall > config.maxToolCalls) {
          state.blocked = true;
          return blockResult(config, "tool_call", nextToolCall, config.maxToolCalls, event, ctx);
        }
        state.toolCalls = nextToolCall;
        if (id) {
          state.seenToolCallIds.add(id);
        }
      },
      { priority: 10000, timeoutMs: 1000 },
    );

    api.on("session_end", (event, ctx) => {
      const key = runKey(event, ctx);
      sharedState().runs.delete(key);
    });
  },
};
