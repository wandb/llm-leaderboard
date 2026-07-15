import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const PLUGIN_ID = "nejumi-budget-guard";
const TRUSTED_TOOL_POLICY_ID = "budget-guard";
const GLOBAL_KEY = Symbol.for("nejumi.openclaw.budgetGuard.v1");

const CONFIG_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    enabled: {
      type: "boolean",
      default: true,
    },
    maxToolCalls: {
      type: "integer",
      minimum: 0,
      default: 0,
    },
    maxAgentTurns: {
      type: "integer",
      minimum: 0,
      default: 0,
    },
    maxCumulativeInputTokens: {
      type: "integer",
      minimum: 0,
      default: 0,
    },
    maxCumulativeOutputTokens: {
      type: "integer",
      minimum: 0,
      default: 0,
    },
    requireActualTokenUsage: {
      type: "boolean",
      default: false,
    },
    agentIds: {
      type: "array",
      items: {
        type: "string",
      },
      default: [],
    },
    sessionKeyPrefixes: {
      type: "array",
      items: {
        type: "string",
      },
      default: [],
    },
    denyTools: {
      type: "array",
      items: {
        type: "string",
      },
      default: [],
    },
    denyArgumentPatterns: {
      type: "array",
      items: {
        type: "string",
      },
      default: [],
    },
    blockReasonPrefix: {
      type: "string",
      default: "NEJUMI_BUDGET_GUARD_BLOCKED",
    },
  },
};

function definePluginEntry(definition) {
  return definition;
}

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

function asString(value) {
  return typeof value === "string" ? value : "";
}

function normalizeConfig(config) {
  const raw = config && typeof config === "object" ? config : {};
  return {
    enabled: raw.enabled !== false,
    maxToolCalls: asPositiveInteger(raw.maxToolCalls),
    maxAgentTurns: asPositiveInteger(raw.maxAgentTurns),
    agentIds: asStringArray(raw.agentIds),
    sessionKeyPrefixes: asStringArray(raw.sessionKeyPrefixes),
    denyTools: asStringArray(raw.denyTools || raw.deny_tools),
    denyArgumentPatterns: asStringArray(raw.denyArgumentPatterns || raw.deny_argument_patterns),
    blockReasonPrefix: String(raw.blockReasonPrefix || "NEJUMI_BUDGET_GUARD_BLOCKED"),
    stateRoot: asString(raw.stateRoot),
    auditFile: asString(raw.auditFile),
  };
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function getPath(root, segments) {
  let current = root;
  for (const segment of segments) {
    if (!isRecord(current) || !(segment in current)) {
      return undefined;
    }
    current = current[segment];
  }
  return current;
}

function hasBudgetKeys(value) {
  if (!isRecord(value)) {
    return false;
  }
  return (
    "enabled" in value ||
    "maxToolCalls" in value ||
    "maxAgentTurns" in value ||
    "maxCumulativeInputTokens" in value ||
    "maxCumulativeOutputTokens" in value ||
    "requireActualTokenUsage" in value ||
    "agentIds" in value ||
    "sessionKeyPrefixes" in value ||
    "denyTools" in value ||
    "deny_tools" in value ||
    "denyArgumentPatterns" in value ||
    "deny_argument_patterns" in value ||
    "blockReasonPrefix" in value ||
    "stateRoot" in value ||
    "auditFile" in value
  );
}

function extractBudgetConfig(value) {
  if (!isRecord(value)) {
    return undefined;
  }
  if (hasBudgetKeys(value)) {
    return value;
  }
  if (isRecord(value.config) && hasBudgetKeys(value.config)) {
    return value.config;
  }
  return undefined;
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
  const hasAgentFilter = config.agentIds.length > 0;
  const agentMatches = currentAgentId && config.agentIds.includes(currentAgentId);
  if (hasAgentFilter && currentAgentId && !agentMatches) {
    return false;
  }
  const currentSessionKey = sessionKey(event, ctx);
  const hasSessionFilter = config.sessionKeyPrefixes.length > 0;
  const currentSessionKeyLower = currentSessionKey.toLowerCase();
  const sessionMatches =
    currentSessionKey &&
    config.sessionKeyPrefixes.some((prefix) => {
      const normalizedPrefix = prefix.toLowerCase();
      return (
        currentSessionKeyLower.startsWith(normalizedPrefix) ||
        currentSessionKeyLower.includes(`:${normalizedPrefix}:`)
      );
    });
  if (
    hasSessionFilter &&
    currentSessionKey &&
    !sessionMatches
  ) {
    return false;
  }
  if (!hasAgentFilter && !hasSessionFilter) {
    return true;
  }
  return Boolean(agentMatches || sessionMatches);
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
      blockedKind: "",
      blockedObserved: 0,
      blockedLimit: 0,
    };
    state.runs.set(key, value);
  }
  return value;
}

function initialRunState() {
  return {
    toolCalls: 0,
    agentTurns: 0,
    seenToolCallIds: new Set(),
    blocked: false,
    blockedKind: "",
    blockedObserved: 0,
    blockedLimit: 0,
  };
}

function normalizeRunState(raw) {
  if (!isRecord(raw)) {
    return initialRunState();
  }
  return {
    toolCalls: asPositiveInteger(raw.toolCalls),
    agentTurns: asPositiveInteger(raw.agentTurns),
    seenToolCallIds: new Set(asStringArray(raw.seenToolCallIds)),
    blocked: raw.blocked === true,
    blockedKind: asString(raw.blockedKind),
    blockedObserved: asPositiveInteger(raw.blockedObserved),
    blockedLimit: asPositiveInteger(raw.blockedLimit),
  };
}

function serializeRunState(value) {
  return {
    toolCalls: asPositiveInteger(value.toolCalls),
    agentTurns: asPositiveInteger(value.agentTurns),
    seenToolCallIds: Array.from(value.seenToolCallIds || []),
    blocked: value.blocked === true,
    blockedKind: asString(value.blockedKind),
    blockedObserved: asPositiveInteger(value.blockedObserved),
    blockedLimit: asPositiveInteger(value.blockedLimit),
  };
}

function defaultOpenClawHome() {
  if (process.env.OPENCLAW_HOME) {
    return process.env.OPENCLAW_HOME;
  }
  const home = os.homedir();
  if (home) {
    return path.join(home, ".openclaw");
  }
  return path.join(os.tmpdir(), "openclaw");
}

function defaultOpenClawConfigPath() {
  return path.join(defaultOpenClawHome(), "openclaw.json");
}

function readLiveBudgetConfig() {
  const configPath = process.env.OPENCLAW_CONFIG_PATH || defaultOpenClawConfigPath();
  try {
    const raw = JSON.parse(fs.readFileSync(configPath, "utf8"));
    return getPath(raw, ["plugins", "entries", PLUGIN_ID, "config"]);
  } catch {
    return undefined;
  }
}

function resolveStateRoot(config) {
  return (
    config.stateRoot ||
    process.env.OPENCLAW_NEJUMI_BUDGET_GUARD_STATE_DIR ||
    path.join(defaultOpenClawHome(), "state", PLUGIN_ID)
  );
}

function runStatePath(config, key) {
  const digest = createHash("sha256").update(String(key)).digest("hex").slice(0, 32);
  return path.join(resolveStateRoot(config), `${digest}.json`);
}

function readPersistentRunState(filePath) {
  try {
    return normalizeRunState(JSON.parse(fs.readFileSync(filePath, "utf8")));
  } catch (error) {
    if (error && error.code === "ENOENT") {
      return initialRunState();
    }
    throw error;
  }
}

function writePersistentRunState(filePath, state) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const tmpPath = `${filePath}.${process.pid}.${Date.now()}.${Math.random().toString(16).slice(2)}.tmp`;
  fs.writeFileSync(tmpPath, `${JSON.stringify(serializeRunState(state))}\n`, "utf8");
  fs.renameSync(tmpPath, filePath);
}

function removePersistentRunState(config, key) {
  try {
    fs.rmSync(runStatePath(config, key), { force: true });
  } catch {
    // Best effort cleanup only.
  }
}

function waitForLock() {
  const buffer = new SharedArrayBuffer(4);
  const view = new Int32Array(buffer);
  Atomics.wait(view, 0, 0, 10);
}

function withPersistentRunState(config, key, mutate) {
  const filePath = runStatePath(config, key);
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const lockPath = `${filePath}.lock`;
  const started = Date.now();
  let locked = false;
  while (!locked) {
    try {
      fs.mkdirSync(lockPath);
      locked = true;
    } catch (error) {
      if (!error || error.code !== "EEXIST") {
        throw error;
      }
      try {
        const stat = fs.statSync(lockPath);
        if (Date.now() - stat.mtimeMs > 5000) {
          fs.rmSync(lockPath, { recursive: true, force: true });
          continue;
        }
      } catch {
        continue;
      }
      if (Date.now() - started > 750) {
        throw new Error(`timed out waiting for ${PLUGIN_ID} state lock`);
      }
      waitForLock();
    }
  }
  try {
    const state = readPersistentRunState(filePath);
    const result = mutate(state);
    writePersistentRunState(filePath, state);
    return result;
  } finally {
    fs.rmSync(lockPath, { recursive: true, force: true });
  }
}

function mutateRunState(config, key, mutate) {
  if (resolveStateRoot(config)) {
    return withPersistentRunState(config, key, mutate);
  }
  const state = runState(key);
  return mutate(state);
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

function wildcardToRegExp(pattern) {
  const escaped = String(pattern).replace(/[|\\{}()[\]^$+?.]/g, "\\$&");
  return new RegExp(`^${escaped.replace(/\*/g, ".*").replace(/\?/g, ".")}$`, "i");
}

function toolNameMatches(value, pattern) {
  const tool = String(value || "").toLowerCase();
  const rawPattern = String(pattern || "");
  const normalizedPattern = rawPattern.toLowerCase();
  if (!tool || !normalizedPattern) {
    return false;
  }
  if (normalizedPattern.startsWith("re:")) {
    try {
      return new RegExp(rawPattern.slice(3), "i").test(tool);
    } catch {
      return false;
    }
  }
  if (normalizedPattern.includes("*") || normalizedPattern.includes("?")) {
    return wildcardToRegExp(rawPattern).test(tool);
  }
  return tool === normalizedPattern;
}

function toolArgumentsText(value) {
  if (value == null) {
    return "";
  }
  if (typeof value === "string") {
    return value;
  }
  if (Array.isArray(value)) {
    return value.map((item) => toolArgumentsText(item)).join("\n");
  }
  if (typeof value === "object") {
    return Object.values(value).map((item) => toolArgumentsText(item)).join("\n");
  }
  return String(value);
}

function toolArgumentsMayExecute(name) {
  const normalized = String(name || "").toLowerCase();
  const executableNames = new Set([
    "bash",
    "code_execution",
    "exec",
    "python",
    "python_exec",
    "shell",
    "terminal",
  ]);
  if (executableNames.has(normalized)) {
    return true;
  }
  return ["exec", "shell", "terminal", "bash"].some((token) => normalized.includes(token));
}

function policyViolation(config, event) {
  const name = toolName(event);
  for (const pattern of config.denyTools) {
    if (toolNameMatches(name, pattern)) {
      return {
        type: "denied_tool",
        pattern,
      };
    }
  }
  if (!toolArgumentsMayExecute(name)) {
    return null;
  }
  const argsText = toolArgumentsText(
    event?.arguments ?? event?.args ?? event?.input ?? event?.params ?? event?.parameters,
  );
  for (const pattern of config.denyArgumentPatterns) {
    try {
      if (new RegExp(pattern, "i").test(argsText)) {
        return {
          type: "denied_argument_pattern",
          pattern,
        };
      }
    } catch {
      // Invalid regexes are ignored by the hook rather than weakening unrelated
      // budget enforcement. The post-run verifier still records configured policy.
    }
  }
  return null;
}

function policyBlockResult(config, violation, event, ctx) {
  const reason = [
    config.blockReasonPrefix,
    "tool_policy_violation",
    `type=${violation.type}`,
    `pattern=${encodeURIComponent(String(violation.pattern || ""))}`,
    `agentId=${agentId(event, ctx) || "unknown"}`,
    `sessionKey=${sessionKey(event, ctx) || "unknown"}`,
    `toolName=${toolName(event) || "n/a"}`,
  ].join(" ");
  return {
    block: true,
    blockReason: reason,
  };
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

function agentRunBlockResult(config, kind, observed, limit, event, ctx) {
  const reason = blockResult(config, kind, observed, limit, event, ctx).blockReason;
  return {
    outcome: "block",
    reason,
    message: reason,
    category: "budget_limit",
    metadata: {
      kind,
      observed,
      limit,
      agentId: agentId(event, ctx) || "",
      sessionKey: sessionKey(event, ctx) || "",
    },
  };
}

function writeAudit(config, phase, event, ctx, details) {
  const auditFile =
    config.auditFile ||
    process.env.OPENCLAW_NEJUMI_BUDGET_GUARD_AUDIT_FILE ||
    path.join(resolveStateRoot(config), "audit.jsonl");
  if (!auditFile) {
    return;
  }
  const row = {
    ts: new Date().toISOString(),
    phase,
    agentId: agentId(event, ctx) || "",
    sessionKey: sessionKey(event, ctx) || "",
    runKey: runKey(event, ctx),
    toolName: toolName(event) || "",
    toolCallId: toolCallId(event, ctx) || "",
    maxToolCalls: config.maxToolCalls,
    maxAgentTurns: config.maxAgentTurns,
    scoped: scoped(config, event, ctx),
    ...details,
  };
  fs.mkdirSync(path.dirname(auditFile), { recursive: true });
  fs.appendFileSync(auditFile, `${JSON.stringify(row)}\n`, "utf8");
}

function pluginConfig(api, event, ctx) {
  const candidates = [
    getPath(event, ["context", "pluginConfig"]),
    getPath(event, ["pluginConfig"]),
    getPath(ctx, ["pluginConfig"]),
    getPath(ctx, ["config", "plugins", "entries", PLUGIN_ID, "config"]),
    getPath(ctx, ["config", "plugins", "entries", PLUGIN_ID]),
    readLiveBudgetConfig(),
    api.pluginConfig,
    getPath(api, ["config", "plugins", "entries", PLUGIN_ID, "config"]),
    getPath(api, ["config", "plugins", "entries", PLUGIN_ID]),
  ];
  for (const candidate of candidates) {
    const extracted = extractBudgetConfig(candidate);
    if (extracted) {
      return normalizeConfig(extracted);
    }
  }
  return normalizeConfig({});
}

function handleBeforeToolCall(api, event, ctx, phase) {
  const config = pluginConfig(api, event, ctx);
  const inScope = scoped(config, event, ctx);
  const hasPolicy = config.denyTools.length > 0 || config.denyArgumentPatterns.length > 0;
  if (!inScope || (config.maxToolCalls <= 0 && !hasPolicy)) {
    if (config.maxToolCalls > 0 || config.maxAgentTurns > 0 || hasPolicy) {
      writeAudit(config, `${phase}_skip`, event, ctx, { inScope });
    }
    return undefined;
  }
  return mutateRunState(config, runKey(event, ctx), (state) => {
    const id = toolCallId(event, ctx);
    if (id && state.seenToolCallIds.has(id)) {
      writeAudit(config, `${phase}_seen`, event, ctx, {
        state: serializeRunState(state),
      });
      return undefined;
    }
    const violation = policyViolation(config, event);
    if (violation) {
      state.blocked = true;
      state.blockedKind = "tool_policy";
      state.blockedObserved = 1;
      state.blockedLimit = 0;
      if (id) {
        state.seenToolCallIds.add(id);
      }
      const result = policyBlockResult(config, violation, event, ctx);
      writeAudit(config, `${phase}_policy_block`, event, ctx, {
        result,
        violation,
        state: serializeRunState(state),
      });
      return result;
    }
    if (config.maxToolCalls <= 0) {
      if (id) {
        state.seenToolCallIds.add(id);
      }
      writeAudit(config, `${phase}_allow`, event, ctx, {
        observed: state.toolCalls,
        state: serializeRunState(state),
      });
      return undefined;
    }
    const nextToolCall = state.toolCalls + 1;
    if (nextToolCall > config.maxToolCalls) {
      state.blocked = true;
      state.blockedKind = "tool_call";
      state.blockedObserved = nextToolCall;
      state.blockedLimit = config.maxToolCalls;
      const result = blockResult(config, "tool_call", nextToolCall, config.maxToolCalls, event, ctx);
      writeAudit(config, `${phase}_block`, event, ctx, {
        result,
        state: serializeRunState(state),
      });
      return result;
    }
    state.toolCalls = nextToolCall;
    if (id) {
      state.seenToolCallIds.add(id);
    }
    writeAudit(config, `${phase}_allow`, event, ctx, {
      observed: nextToolCall,
      state: serializeRunState(state),
    });
    return undefined;
  });
}

function handleBeforeAgentReplyAudit(api, event, ctx) {
  const config = pluginConfig(api, event, ctx);
  const inScope = scoped(config, event, ctx);
  const hasPolicy = config.denyTools.length > 0 || config.denyArgumentPatterns.length > 0;
  if (config.maxAgentTurns > 0 || config.maxToolCalls > 0 || hasPolicy) {
    writeAudit(config, inScope ? "before_agent_reply_seen" : "before_agent_reply_skip", event, ctx, {
      inScope,
      cleanedBodyLength: String(event?.cleanedBody || "").length,
    });
  }
  return undefined;
}

export default definePluginEntry({
  id: PLUGIN_ID,
  name: "Nejumi Budget Guard",
  description: "Blocks OpenClaw tool calls before execution and audits agent-turn hook coverage for Nejumi budgets.",
  configSchema: CONFIG_SCHEMA,
  register(api) {
    if (typeof api.registerTrustedToolPolicy === "function") {
      api.registerTrustedToolPolicy({
        id: TRUSTED_TOOL_POLICY_ID,
        description: "Blocks Nejumi benchmark tool calls before the configured per-session budget is exceeded.",
        evaluate: async (event, ctx) => handleBeforeToolCall(api, event, ctx, "trusted_tool_policy"),
      });
    }

    api.on(
      "before_agent_run",
      async (event, ctx) => {
	        const config = pluginConfig(api, event, ctx);
	        const inScope = scoped(config, event, ctx);
	        const hasPolicy = config.denyTools.length > 0 || config.denyArgumentPatterns.length > 0;
	        if (!inScope || (config.maxAgentTurns <= 0 && config.maxToolCalls <= 0 && !hasPolicy)) {
	          if (config.maxAgentTurns > 0 || config.maxToolCalls > 0 || hasPolicy) {
	            writeAudit(config, "before_agent_run_skip", event, ctx, { inScope });
	          }
	          return;
	        }
        return mutateRunState(config, runKey(event, ctx), (state) => {
          if (state.blocked) {
            const result = agentRunBlockResult(
              config,
              state.blockedKind || "budget",
              state.blockedObserved || 1,
              state.blockedLimit || 0,
              event,
              ctx,
            );
            writeAudit(config, "before_agent_run_blocked_previous", event, ctx, {
              result,
              state: serializeRunState(state),
            });
            return result;
          }
          if (config.maxAgentTurns <= 0) {
            writeAudit(config, "before_agent_run_allow", event, ctx, {
              state: serializeRunState(state),
            });
            return;
          }
          const nextAgentTurn = state.agentTurns + 1;
          if (nextAgentTurn > config.maxAgentTurns) {
            state.blocked = true;
            state.blockedKind = "agent_turn";
            state.blockedObserved = nextAgentTurn;
            state.blockedLimit = config.maxAgentTurns;
            const result = agentRunBlockResult(config, "agent_turn", nextAgentTurn, config.maxAgentTurns, event, ctx);
            writeAudit(config, "before_agent_run_block", event, ctx, {
              result,
              state: serializeRunState(state),
            });
            return result;
          }
          state.agentTurns = nextAgentTurn;
          writeAudit(config, "before_agent_run_allow", event, ctx, {
            observed: nextAgentTurn,
            state: serializeRunState(state),
          });
        });
      },
      { priority: 10000, timeoutMs: 1000 },
    );

    api.on(
      "before_tool_call",
      async (event, ctx) => handleBeforeToolCall(api, event, ctx, "before_tool_call"),
      { priority: 10000, timeoutMs: 1000 },
    );

    api.on(
      "before_agent_reply",
      async (event, ctx) => handleBeforeAgentReplyAudit(api, event, ctx),
      { priority: 10000, timeoutMs: 1000 },
    );

    api.on("session_end", (event, ctx) => {
      const key = runKey(event, ctx);
      sharedState().runs.delete(key);
      removePersistentRunState(pluginConfig(api, event, ctx), key);
    });
  },
});
