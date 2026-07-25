import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const PLUGIN_ID = "nejumi-budget-guard";
const TRUSTED_TOOL_POLICY_ID = "budget-guard";
const GLOBAL_KEY = Symbol.for("nejumi.openclaw.budgetGuard.v1");
const LANDLOCK_LAUNCHER = fileURLToPath(new URL("../landlock_exec.py", import.meta.url));

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
    workspaceIsolationEnabled: {
      type: "boolean",
      default: false,
    },
    liveConfigPath: {
      type: "string",
      default: "",
    },
    agentWorkspaces: {
      type: "object",
      additionalProperties: {
        type: "object",
        additionalProperties: false,
        required: ["workspace", "tmp", "home"],
        properties: {
          workspace: { type: "string" },
          tmp: { type: "string" },
          home: { type: "string" },
          readOnlyRoots: {
            type: "array",
            items: { type: "string" },
            default: [],
          },
        },
      },
      default: {},
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
  const rawWorkspaces =
    raw.agentWorkspaces && typeof raw.agentWorkspaces === "object"
      ? raw.agentWorkspaces
      : {};
  const agentWorkspaces = {};
  for (const [id, value] of Object.entries(rawWorkspaces)) {
    if (!value || typeof value !== "object") {
      continue;
    }
    const workspace = asString(value.workspace);
    const tmp = asString(value.tmp);
    const home = asString(value.home);
    if (!workspace || !tmp || !home) {
      continue;
    }
    agentWorkspaces[String(id)] = {
      workspace,
      tmp,
      home,
      readOnlyRoots: asStringArray(value.readOnlyRoots),
    };
  }
  return {
    enabled: raw.enabled !== false,
    maxToolCalls: asPositiveInteger(raw.maxToolCalls),
    maxAgentTurns: asPositiveInteger(raw.maxAgentTurns),
    agentIds: asStringArray(raw.agentIds),
    sessionKeyPrefixes: asStringArray(raw.sessionKeyPrefixes),
    denyTools: asStringArray(raw.denyTools || raw.deny_tools),
    denyArgumentPatterns: asStringArray(raw.denyArgumentPatterns || raw.deny_argument_patterns),
    blockReasonPrefix: String(raw.blockReasonPrefix || "NEJUMI_BUDGET_GUARD_BLOCKED"),
    workspaceIsolationEnabled: raw.workspaceIsolationEnabled === true,
    agentWorkspaces,
    liveConfigPath: asString(raw.liveConfigPath),
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
    "workspaceIsolationEnabled" in value ||
    "agentWorkspaces" in value ||
    "liveConfigPath" in value ||
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
      warningStages: [],
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
    warningStages: [],
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
    warningStages: asStringArray(raw.warningStages),
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
    warningStages: asStringArray(value.warningStages),
  };
}

const BUDGET_WARNING_STAGES = [
  { id: "half", ratio: 0.5 },
  { id: "converge", ratio: 0.75 },
  { id: "finalize", ratio: 0.875 },
  { id: "urgent", ratio: 0.95 },
];

function pendingBudgetWarning(config, state) {
  if (config.maxToolCalls <= 0) return null;
  const ratio = state.toolCalls / config.maxToolCalls;
  let selected = null;
  for (const stage of BUDGET_WARNING_STAGES) {
    if (ratio >= stage.ratio && !state.warningStages.includes(stage.id)) selected = stage;
  }
  if (!selected) return null;
  state.warningStages.push(selected.id);
  const remaining = Math.max(0, config.maxToolCalls - state.toolCalls);
  const action = selected.id === "urgent"
    ? "Stop all investigation. Preserve the best working patch now and use another tool only if it is essential to complete or verify that patch."
    : selected.id === "finalize"
      ? "Finish the minimal patch now. Avoid new exploratory work and run only focused verification."
      : selected.id === "converge"
        ? "Conclude exploration and prioritize implementation, targeted tests, and a clean final diff."
        : "Checkpoint your progress and ensure a viable minimal patch exists before spending the remaining budget.";
  const turnLimit = config.maxAgentTurns > 0
    ? ` The independent hard agent-turn limit is ${config.maxAgentTurns} and may stop execution sooner.`
    : "";
  return [
    "[NEJUMI RUNTIME BUDGET WARNING]",
    `Tool calls used: ${state.toolCalls}/${config.maxToolCalls}; ${remaining} remain.${turnLimit}`,
    "When any hard runtime budget is exhausted, execution stops immediately: the current git diff is submitted for coding tasks, or the current answer is submitted for answer tasks, even if incomplete. No extra cleanup turn is guaranteed.",
    `ACTION REQUIRED: ${action}`,
  ].join("\n");
}

function pendingAgentTurnWarning(config, state) {
  if (config.maxAgentTurns <= 0) return null;
  const nextTurn = state.agentTurns + 1;
  const ratio = nextTurn / config.maxAgentTurns;
  let selected = null;
  for (const stage of BUDGET_WARNING_STAGES) {
    const id = `turn:${stage.id}`;
    if (ratio >= stage.ratio && !state.warningStages.includes(id)) selected = { ...stage, id };
  }
  if (!selected) return null;
  state.warningStages.push(selected.id);
  const remaining = Math.max(0, config.maxAgentTurns - nextTurn);
  const action = selected.id === "turn:urgent"
    ? "Stop exploring. Complete the best viable patch or exact final answer in this turn."
    : selected.id === "turn:finalize"
      ? "Finalize the solution now and avoid starting any new line of investigation."
      : selected.id === "turn:converge"
        ? "Converge on the current approach; prioritize implementation or a complete final derivation."
        : "Checkpoint progress now and ensure a viable submission exists before using more turns.";
  return [
    "[NEJUMI RUNTIME BUDGET WARNING]",
    `Agent turn starting: ${nextTurn}/${config.maxAgentTurns}; ${remaining} turns remain after this one.`,
    "When any hard runtime budget is exhausted, execution stops immediately: the current git diff is submitted for coding tasks, or the current answer is submitted for answer tasks, even if incomplete. No extra cleanup turn is guaranteed.",
    `ACTION REQUIRED: ${action}`,
  ].join("\n");
}

function appendWarningToToolMessage(message, warning) {
  if (!isRecord(message)) return message;
  const updated = { ...message };
  if (Array.isArray(message.content)) {
    updated.content = [...message.content, { type: "text", text: warning }];
  } else if (typeof message.content === "string") {
    updated.content = `${message.content}\n\n${warning}`;
  } else {
    updated.content = [{ type: "text", text: warning }];
  }
  return updated;
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
  if (process.env.OPENCLAW_STATE_DIR) {
    return path.join(process.env.OPENCLAW_STATE_DIR, "openclaw.json");
  }
  return path.join(defaultOpenClawHome(), "openclaw.json");
}

function readLiveBudgetConfig(explicitConfigPath = "") {
  const configPath =
    process.env.OPENCLAW_CONFIG_PATH ||
    (process.env.OPENCLAW_STATE_DIR ? defaultOpenClawConfigPath() : "") ||
    explicitConfigPath;
  if (!configPath) {
    return undefined;
  }
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

function shellQuote(value) {
  return `'${String(value).replaceAll("'", `'\"'\"'`)}'`;
}

function workspaceSpec(config, event, ctx) {
  if (!config.workspaceIsolationEnabled) {
    return null;
  }
  const id = agentId(event, ctx);
  return id ? config.agentWorkspaces[id] || null : null;
}

function pathInside(candidate, root) {
  const relative = path.relative(root, candidate);
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

function canonicalizeTarget(rawPath, workspace) {
  const absolute = path.resolve(workspace, String(rawPath));
  let cursor = absolute;
  const suffix = [];
  while (!fs.existsSync(cursor)) {
    const parent = path.dirname(cursor);
    if (parent === cursor) {
      break;
    }
    suffix.unshift(path.basename(cursor));
    cursor = parent;
  }
  const canonicalParent = fs.realpathSync(cursor);
  return path.resolve(canonicalParent, ...suffix);
}

function collectDerivedPaths(event) {
  const values = [];
  const derived = event?.derivedPaths;
  if (Array.isArray(derived)) {
    values.push(...derived);
  } else if (derived && typeof derived === "object") {
    for (const value of Object.values(derived)) {
      if (Array.isArray(value)) {
        values.push(...value);
      } else if (typeof value === "string") {
        values.push(value);
      }
    }
  }
  return values.filter((value) => typeof value === "string" && value.length > 0);
}

function collectDirectPathParams(params) {
  if (!params || typeof params !== "object") {
    return [];
  }
  const keys = new Set([
    "path",
    "file",
    "file_path",
    "filePath",
    "target_path",
    "targetPath",
    "directory",
    "dir",
    "cwd",
    "workdir",
  ]);
  const values = [];
  for (const [key, value] of Object.entries(params)) {
    if (!keys.has(key)) {
      continue;
    }
    if (typeof value === "string" && value.length > 0) {
      values.push(value);
    } else if (Array.isArray(value)) {
      values.push(...value.filter((item) => typeof item === "string" && item.length > 0));
    }
  }
  return values;
}

function collectPatchPaths(params) {
  if (!params || typeof params !== "object") {
    return [];
  }
  const patch = [params.patch, params.input, params.content]
    .find((value) => typeof value === "string");
  if (!patch) {
    return [];
  }
  const paths = [];
  const pattern = /^\*\*\* (?:Add File|Update File|Delete File|Move to): (.+)$/gm;
  for (const match of patch.matchAll(pattern)) {
    if (match[1]) {
      paths.push(match[1].trim());
    }
  }
  return paths;
}

function toolMutatesFilesystem(name) {
  const normalized = String(name || "").toLowerCase();
  return ["write", "edit", "apply_patch", "patch", "delete", "move"].some(
    (token) => normalized === token || normalized.includes(token),
  );
}

function toolUsesDirectFilesystem(name) {
  const normalized = String(name || "").toLowerCase();
  return [
    "read",
    "write",
    "edit",
    "apply_patch",
    "patch",
    "glob",
    "grep",
    "find",
    "list",
  ].some((token) => normalized === token || normalized.includes(token));
}

function workspaceBlockResult(event, ctx, target, workspace) {
  return {
    block: true,
    blockReason: [
      "NEJUMI_WORKSPACE_GUARD_BLOCKED",
      "cross_task_filesystem_access",
      `agentId=${agentId(event, ctx) || "unknown"}`,
      `toolName=${toolName(event) || "n/a"}`,
      `target=${encodeURIComponent(String(target || ""))}`,
      `workspace=${encodeURIComponent(workspace)}`,
      "Continue using only the current task workspace; do not inspect sibling tasks or shared temporary files.",
    ].join(" "),
  };
}

function wrapExecParams(params, spec) {
  if (!params || typeof params !== "object") {
    return null;
  }
  const commandKey =
    typeof params.command === "string"
      ? "command"
      : typeof params.cmd === "string"
        ? "cmd"
        : null;
  if (!commandKey) {
    return null;
  }
  const workspace = path.resolve(spec.workspace);
  const privateTmp = path.resolve(spec.tmp);
  const privateHome = path.resolve(spec.home);
  fs.mkdirSync(privateTmp, { recursive: true, mode: 0o700 });
  fs.mkdirSync(privateHome, { recursive: true, mode: 0o700 });
  const readOnlyArgs = spec.readOnlyRoots
    .filter((root) => typeof root === "string" && root.length > 0 && fs.existsSync(root))
    .flatMap((root) => ["--read-only", root]);
  const launcherArgs = [
    LANDLOCK_LAUNCHER,
    "--workspace",
    workspace,
    "--tmp",
    privateTmp,
    "--home",
    privateHome,
    ...readOnlyArgs,
    "--",
    "/bin/bash",
    "-lc",
    params[commandKey],
  ];
  const cacheRoot = path.join(privateHome, ".cache");
  const envValues = {
    HOME: privateHome,
    TMPDIR: privateTmp,
    TMP: privateTmp,
    TEMP: privateTmp,
    XDG_CACHE_HOME: cacheRoot,
    PIP_CACHE_DIR: path.join(cacheRoot, "pip"),
    npm_config_cache: path.join(cacheRoot, "npm"),
  };
  const wrapped = [
    "exec",
    "env",
    ...Object.entries(envValues).map(([key, value]) => `${key}=${shellQuote(value)}`),
    "/usr/bin/python3",
  ];
  wrapped.push(...launcherArgs.map(shellQuote));
  return {
    ...params,
    [commandKey]: wrapped.join(" "),
  };
}

function workspaceGuardDecision(config, event, ctx) {
  const spec = workspaceSpec(config, event, ctx);
  if (!spec) {
    return undefined;
  }
  const name = toolName(event);
  const normalized = String(name || "").toLowerCase();
  const params = event?.params ?? event?.arguments ?? event?.args ?? event?.input;
  if (toolArgumentsMayExecute(name)) {
    const rewritten = wrapExecParams(params, spec);
    if (!rewritten) {
      return workspaceBlockResult(event, ctx, "missing_exec_command", spec.workspace);
    }
    return { params: rewritten };
  }
  if (!toolUsesDirectFilesystem(normalized)) {
    return undefined;
  }

  const rawTargets = [
    ...collectDerivedPaths(event),
    ...collectDirectPathParams(params),
    ...collectPatchPaths(params),
  ];
  const workspace = fs.realpathSync(spec.workspace);
  const writableRoots = [workspace, spec.tmp, spec.home]
    .filter((value) => fs.existsSync(value))
    .map((value) => fs.realpathSync(value));
  const readableRoots = [
    ...writableRoots,
    ...spec.readOnlyRoots
      .filter((value) => fs.existsSync(value))
      .map((value) => fs.realpathSync(value)),
  ];
  const allowedRoots = toolMutatesFilesystem(normalized) ? writableRoots : readableRoots;
  for (const rawTarget of rawTargets) {
    let target;
    try {
      target = canonicalizeTarget(rawTarget, workspace);
    } catch {
      return workspaceBlockResult(event, ctx, rawTarget, workspace);
    }
    if (!allowedRoots.some((root) => pathInside(target, root))) {
      return workspaceBlockResult(event, ctx, target, workspace);
    }
  }
  return undefined;
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
  const staticCandidates = [
    getPath(event, ["context", "pluginConfig"]),
    getPath(event, ["pluginConfig"]),
    getPath(ctx, ["pluginConfig"]),
    getPath(ctx, ["config", "plugins", "entries", PLUGIN_ID, "config"]),
    getPath(ctx, ["config", "plugins", "entries", PLUGIN_ID]),
    api.pluginConfig,
    getPath(api, ["config", "plugins", "entries", PLUGIN_ID, "config"]),
    getPath(api, ["config", "plugins", "entries", PLUGIN_ID]),
  ];
  let liveConfigPath = "";
  for (const candidate of staticCandidates) {
    const extracted = extractBudgetConfig(candidate);
    if (extracted && typeof extracted.liveConfigPath === "string" && extracted.liveConfigPath) {
      liveConfigPath = extracted.liveConfigPath;
      break;
    }
  }
  const candidates = [
    // Task agents are registered dynamically while the Gateway is running.
    // OpenClaw hook contexts can retain the startup config, so the on-disk
    // config must be the source of truth whenever it is available.
    readLiveBudgetConfig(liveConfigPath),
    ...staticCandidates,
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
  const workspaceDecision = inScope ? workspaceGuardDecision(config, event, ctx) : undefined;
  const hasWorkspaceGuard = Boolean(workspaceSpec(config, event, ctx));
  if (workspaceDecision?.block) {
    writeAudit(config, `${phase}_workspace_block`, event, ctx, {
      result: workspaceDecision,
    });
    return workspaceDecision;
  }
  if (!inScope || (config.maxToolCalls <= 0 && !hasPolicy && !hasWorkspaceGuard)) {
    if (config.maxToolCalls > 0 || config.maxAgentTurns > 0 || hasPolicy || hasWorkspaceGuard) {
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
      return workspaceDecision;
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
      return workspaceDecision;
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
    return workspaceDecision;
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

function handleAgentTurnPrepare(api, event, ctx) {
  const config = pluginConfig(api, event, ctx);
  if (!scoped(config, event, ctx) || config.maxAgentTurns <= 0) return undefined;
  return mutateRunState(config, runKey(event, ctx), (state) => {
    const warning = pendingAgentTurnWarning(config, state);
    if (!warning) return undefined;
    writeAudit(config, "agent_turn_budget_warning", event, ctx, {
      warning,
      state: serializeRunState(state),
    });
    return { appendContext: warning };
  });
}

function handleToolResultPersist(api, event, ctx) {
  const config = pluginConfig(api, event, ctx);
  const inScope = scoped(config, event, ctx);
  if (!inScope || config.maxToolCalls <= 0) return undefined;
  return mutateRunState(config, runKey(event, ctx), (state) => {
    const warning = pendingBudgetWarning(config, state);
    if (!warning) return undefined;
    const result = { message: appendWarningToToolMessage(event?.message, warning) };
    writeAudit(config, "tool_result_budget_warning", event, ctx, {
      warning,
      state: serializeRunState(state),
    });
    return result;
  });
}

export default definePluginEntry({
  id: PLUGIN_ID,
  name: "Nejumi Budget Guard",
  description: "Warns models as runtime budgets run low and blocks tool calls before hard limits are exceeded.",
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
      "agent_turn_prepare",
      async (event, ctx) => handleAgentTurnPrepare(api, event, ctx),
      { priority: 10000, timeoutMs: 1000 },
    );

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
      "tool_result_persist",
      (event, ctx) => handleToolResultPersist(api, event, ctx),
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
