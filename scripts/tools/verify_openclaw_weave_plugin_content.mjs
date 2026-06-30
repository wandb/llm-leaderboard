#!/usr/bin/env node
/*
 * Verify that the installed weave-openclaw plugin can preserve message/tool
 * content when OpenClaw-style events are replayed into the plugin handlers.
 *
 * This is local-only: it points the Weave GenAI exporter at an in-process OTLP
 * HTTP collector and checks for unique content markers in the captured payload.
 * It does not run a model and does not send traces to W&B.
 */

import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { spawnSync } from "node:child_process";

const defaultOpenClawStateDir = path.join(os.homedir(), ".openclaw");
const defaultPluginProjectDir = path.join(
  defaultOpenClawStateDir,
  "npm",
  "projects",
  "weave-openclaw",
);

function parseArgs(argv) {
  const args = {
    pluginProjectDir: process.env.OPENCLAW_WEAVE_PLUGIN_PROJECT_DIR || defaultPluginProjectDir,
    openclawPackageDir:
      process.env.OPENCLAW_PACKAGE_DIR ||
      path.join(
        os.homedir(),
        ".nvm",
        "versions",
        "node",
        "v24.17.0",
        "lib",
        "node_modules",
        "openclaw",
      ),
    json: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (value === "--plugin-project-dir") {
      args.pluginProjectDir = argv[++index];
    } else if (value === "--openclaw-package-dir") {
      args.openclawPackageDir = argv[++index];
    } else if (value === "--json") {
      args.json = true;
    } else if (value === "-h" || value === "--help") {
      console.log(`Usage: node scripts/tools/verify_openclaw_weave_plugin_content.mjs [options]

Options:
  --plugin-project-dir PATH   OpenClaw-managed weave-openclaw project dir.
  --openclaw-package-dir PATH Global OpenClaw package dir used for plugin-sdk imports.
  --json                      Print only the final JSON result.
`);
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${value}`);
    }
  }
  return args;
}

function ensureDir(label, dir) {
  const stat = fs.existsSync(dir) ? fs.statSync(dir) : null;
  if (!stat?.isDirectory()) {
    throw new Error(`${label} does not exist or is not a directory: ${dir}`);
  }
}

function symlinkNodeModules(sourceNodeModules, tempNodeModules) {
  fs.mkdirSync(tempNodeModules, { recursive: true });
  for (const entry of fs.readdirSync(sourceNodeModules)) {
    const source = path.join(sourceNodeModules, entry);
    const target = path.join(tempNodeModules, entry);
    if (!fs.existsSync(target)) {
      fs.symlinkSync(source, target);
    }
  }
}

function replayScript() {
  return String.raw`
import http from "node:http";
import { flushOTel } from "weave";
import { createWeavePlugin } from "./node_modules/weave-openclaw/dist/src/plugin.js";
import { createWeaveHookState } from "./node_modules/weave-openclaw/dist/src/state/hook-state.js";

function readBody(req) {
  return new Promise((resolve) => {
    const chunks = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end", () => resolve(Buffer.concat(chunks)));
  });
}

const chunks = [];
const server = http.createServer(async (req, res) => {
  const body = await readBody(req);
  if (req.url === "/agents/otel/v1/traces") {
    chunks.push(body);
  }
  res.writeHead(200, {"content-type": "application/json"});
  res.end("{}");
});

await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const {port} = server.address();
process.env.WF_TRACE_SERVER_URL = ` + "`http://127.0.0.1:${port}`" + `;
process.env.WANDB_API_KEY ||= "nejumi-local-plugin-content-check";

const marker = ` + "`NEJUMI_PLUGIN_CONTENT_${Date.now()}`" + `;
const plugin = createWeavePlugin({
  pluginConfig: {
    entity: "llm-leaderboard",
    project: "tc-leaderboard",
    agentName: "nejumi-plugin-local-check",
    captureContent: true,
    flushIntervalMs: 1000,
  },
  hookState: createWeaveHookState(),
});
const logger = {
  info() {},
  warn() {},
  error(...args) {
    console.error(...args);
  },
};
await plugin.service.start({logger, config: {}});
const trusted = {trusted: true};

plugin.handlers.diagnostic({
  type: "run.started",
  runId: "run-1",
  sessionKey: "session-1",
  model: "local-model",
}, trusted);
plugin.handlers.diagnostic({
  type: "model.call.started",
  runId: "run-1",
  callId: "call-1",
  model: "local-model",
  provider: "local",
}, trusted);
plugin.handlers.hook.llm_input?.({
  runId: "run-1",
  systemPrompt: "system",
  historyMessages: [],
  prompt: ` + "`${marker}_INPUT`" + `,
}, {});
plugin.handlers.hook.llm_output?.({
  runId: "run-1",
  assistantTexts: [` + "`${marker}_OUTPUT`" + `],
  usage: {input: 7, output: 3},
}, {});
plugin.handlers.diagnostic({
  type: "model.call.completed",
  runId: "run-1",
  callId: "call-1",
}, trusted);
plugin.handlers.hook.before_tool_call?.({
  toolCallId: "tool-1",
  toolName: "exec",
  params: {cmd: ` + "`${marker}_TOOL_ARGS`" + `},
}, {});
plugin.handlers.diagnostic({
  type: "tool.execution.started",
  runId: "run-1",
  toolCallId: "tool-1",
  toolName: "exec",
  paramsSummary: {cmd: "summary"},
}, trusted);
plugin.handlers.hook.after_tool_call?.({
  toolCallId: "tool-1",
  result: ` + "`${marker}_TOOL_RESULT`" + `,
}, {});
plugin.handlers.diagnostic({
  type: "tool.execution.completed",
  runId: "run-1",
  toolCallId: "tool-1",
  toolName: "exec",
}, trusted);
plugin.handlers.diagnostic({
  type: "run.completed",
  runId: "run-1",
  sessionKey: "session-1",
  outcome: "completed",
}, trusted);

await plugin.service.stop({logger});
await flushOTel();
await new Promise((resolve) => setTimeout(resolve, 100));
server.close();

const payload = Buffer.concat(chunks);
const text = payload.toString("utf8");
const result = {
  ok: true,
  bytes: payload.length,
  has_input: text.includes(` + "`${marker}_INPUT`" + `),
  has_output: text.includes(` + "`${marker}_OUTPUT`" + `),
  has_tool_args: text.includes(` + "`${marker}_TOOL_ARGS`" + `),
  has_tool_result: text.includes(` + "`${marker}_TOOL_RESULT`" + `),
};
result.ok = result.bytes > 0
  && result.has_input
  && result.has_output
  && result.has_tool_args
  && result.has_tool_result;
console.log(JSON.stringify(result));
`;
}

function runReplay(args) {
  const pluginProjectDir = path.resolve(args.pluginProjectDir);
  const openclawPackageDir = path.resolve(args.openclawPackageDir);
  const sourceNodeModules = path.join(pluginProjectDir, "node_modules");
  ensureDir("plugin project dir", pluginProjectDir);
  ensureDir("plugin node_modules", sourceNodeModules);
  ensureDir("OpenClaw package dir", openclawPackageDir);
  ensureDir(
    "weave-openclaw package",
    path.join(sourceNodeModules, "weave-openclaw"),
  );
  ensureDir("weave package", path.join(sourceNodeModules, "weave"));

  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "nejumi-weave-plugin-"));
  try {
    const tempNodeModules = path.join(tempDir, "node_modules");
    symlinkNodeModules(sourceNodeModules, tempNodeModules);
    const openclawTarget = path.join(tempNodeModules, "openclaw");
    if (!fs.existsSync(openclawTarget)) {
      fs.symlinkSync(openclawPackageDir, openclawTarget);
    }
    const child = spawnSync(
      process.execPath,
      ["--preserve-symlinks", "--input-type=module"],
      {
        cwd: tempDir,
        input: replayScript(),
        text: true,
        encoding: "utf8",
        env: {
          ...process.env,
          WANDB_API_KEY: "nejumi-local-plugin-content-check",
        },
      },
    );
    if (child.error) {
      throw child.error;
    }
    const stdoutLines = child.stdout.trim().split(/\r?\n/).filter(Boolean);
    let payload = {};
    try {
      payload = JSON.parse(stdoutLines.at(-1) || "{}");
    } catch {
      payload = {};
    }
    return {
      ok: child.status === 0 && payload.ok === true,
      returncode: child.status,
      plugin_project_dir: pluginProjectDir,
      openclaw_package_dir: openclawPackageDir,
      stdout_tail: child.stdout.slice(-2000),
      stderr_tail: child.stderr.slice(-2000),
      payload,
    };
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
}

try {
  const args = parseArgs(process.argv.slice(2));
  const result = runReplay(args);
  if (args.json) {
    console.log(JSON.stringify(result, null, 2));
  } else if (result.ok) {
    console.log(
      `weave-openclaw plugin local content replay OK: ${result.payload.bytes} OTLP bytes`,
    );
  } else {
    console.log(JSON.stringify(result, null, 2));
  }
  if (!result.ok) {
    process.exit(1);
  }
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
