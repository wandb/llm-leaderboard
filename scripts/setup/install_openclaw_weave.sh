#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

MIN_NODE_VERSION="${MIN_NODE_VERSION:-22.19.0}"
MIN_OPENCLAW_VERSION="${MIN_OPENCLAW_VERSION:-2026.4.25}"
WANDB_ENTITY="${WANDB_ENTITY:-llm-leaderboard}"
WANDB_PROJECT="${WANDB_PROJECT:-tc-leaderboard}"
OPENCLAW_CONFIG="${OPENCLAW_CONFIG:-$HOME/.openclaw/openclaw.json}"
OPENCLAW_ENV_FILE="${OPENCLAW_ENV_FILE:-$REPO_ROOT/.env}"
OPENCLAW_AGENT_NAME="${OPENCLAW_AGENT_NAME:-nejumi-taiwan-openclaw}"
OPENCLAW_AGENT_VERSION="${OPENCLAW_AGENT_VERSION:-nejumi-agent-protocol-2026.04}"
OPENCLAW_SERVICE_NAME="${OPENCLAW_SERVICE_NAME:-openclaw-agent}"
CAPTURE_CONTENT="${CAPTURE_CONTENT:-true}"
WEAVE_OTEL_VERSION="${WEAVE_OTEL_VERSION:-${WEAVE_OTEL_CORE_VERSION:-1.26.0}}"

INSTALL_NODE=0
INSTALL_OPENCLAW=0
INSTALL_PLUGIN=0
INSTALL_PROVIDER_PLUGINS=0
WRITE_CONFIG=0
WRITE_MODEL_CONFIG=0
CONFIGURE_GATEWAY_ENV=0
RESTART_GATEWAY=0
RUN_ONBOARD=0
CHECK_ONLY=0

usage() {
  cat <<'USAGE'
Usage:
  scripts/setup/install_openclaw_weave.sh --all
  scripts/setup/install_openclaw_weave.sh --check-only

Options:
  --all                 Install/update Node via nvm when needed, OpenClaw, weave-openclaw, config,
                        model providers, and the OpenClaw gateway .env systemd drop-in.
  --check-only          Print current versions and exit non-zero when requirements are missing.
  --install-node        Install Node 22.19+ using nvm if available.
  --install-openclaw    Install/update OpenClaw with npm.
  --install-plugin      Install/update weave-openclaw via OpenClaw plugin manager.
  --install-provider-plugins
                        Install provider plugins used by the Taiwan agentic evals.
  --write-config        Merge W&B Weave plugin config into ~/.openclaw/openclaw.json.
  --write-model-config  Merge Taiwan eval model providers into OpenClaw config.
  --configure-gateway-env
                        Write a systemd user drop-in so openclaw-gateway.service reads .env.
  --restart-gateway     Restart openclaw-gateway.service after writing the env drop-in.
  --run-onboard         Run `openclaw onboard --install-daemon` after OpenClaw install.
  --capture-content     Send prompt/reply/tool content to W&B Weave. This is the default.
  --no-capture-content  Keep Weave Agents structure-only when content logging is not desired.
  --entity VALUE        W&B entity. Default: llm-leaderboard.
  --project VALUE       W&B project. Default: tc-leaderboard.
  --config PATH         OpenClaw config path. Default: ~/.openclaw/openclaw.json.
  --env-file PATH       Env file for OpenClaw gateway SecretRefs. Default: repo .env.
  -h, --help            Show this help.

Environment:
  WANDB_API_KEY         Used by weave-openclaw. The script does not print or persist the key.
  WANDB_ENTITY          Same as --entity.
  WANDB_PROJECT         Same as --project.
  OPENCLAW_CONFIG       Same as --config.
  OPENCLAW_ENV_FILE     Same as --env-file.
  WEAVE_OTEL_VERSION    @opentelemetry/core/resources/sdk-trace-base version pinned
                        inside weave-openclaw. Default: 1.26.0.
USAGE
}

log() {
  printf '[openclaw-setup] %s\n' "$*"
}

version_ge() {
  local current="$1"
  local minimum="$2"
  python3 - "$current" "$minimum" <<'PY'
import re
import sys

def parse(value):
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", value or "")
    if not match:
        return None
    return tuple(int(part) for part in match.groups())

current = parse(sys.argv[1])
minimum = parse(sys.argv[2])
sys.exit(0 if current is not None and current >= minimum else 1)
PY
}

node_version() {
  if command -v node >/dev/null 2>&1; then
    node --version | sed 's/^v//'
  fi
}

prepend_best_nvm_node() {
  local root="$HOME/.nvm/versions/node"
  local candidate
  if [ ! -d "$root" ]; then
    return 0
  fi
  while IFS= read -r candidate; do
    if [ -x "$candidate/node" ]; then
      local version
      version="$("$candidate/node" --version 2>/dev/null | sed 's/^v//' || true)"
      if [ -n "$version" ] && version_ge "$version" "$MIN_NODE_VERSION"; then
        export PATH="$candidate:$PATH"
        log "Using nvm Node: $version"
        return 0
      fi
    fi
  done < <(find "$root" -mindepth 2 -maxdepth 2 -type d -name bin | sort -Vr)
}

openclaw_version() {
  if command -v openclaw >/dev/null 2>&1; then
    openclaw --version 2>&1 | grep -Eo '[0-9]+\.[0-9]+\.[0-9]+' | head -n 1 || true
  fi
}

load_nvm() {
  if command -v nvm >/dev/null 2>&1; then
    return 0
  fi
  if [ -s "$HOME/.nvm/nvm.sh" ]; then
    # shellcheck disable=SC1091
    . "$HOME/.nvm/nvm.sh"
  fi
}

ensure_node() {
  prepend_best_nvm_node
  local current
  current="$(node_version || true)"
  if [ -n "$current" ] && version_ge "$current" "$MIN_NODE_VERSION"; then
    log "Node OK: $current"
    return 0
  fi

  log "Node ${current:-missing} does not satisfy >= $MIN_NODE_VERSION"
  if [ "$INSTALL_NODE" -eq 0 ]; then
    return 1
  fi

  load_nvm
  if ! command -v nvm >/dev/null 2>&1; then
    log "nvm is not available. Install Node 24 or Node >= $MIN_NODE_VERSION, then rerun."
    return 1
  fi

  log "Installing Node 24 with nvm"
  nvm install 24
  nvm use 24

  current="$(node_version || true)"
  if [ -n "$current" ] && version_ge "$current" "$MIN_NODE_VERSION"; then
    log "Node OK after install: $current"
    return 0
  fi
  return 1
}

ensure_openclaw() {
  local current
  current="$(openclaw_version || true)"
  if [ -n "$current" ] && version_ge "$current" "$MIN_OPENCLAW_VERSION"; then
    log "OpenClaw OK: $current"
    return 0
  fi

  log "OpenClaw ${current:-missing} does not satisfy >= $MIN_OPENCLAW_VERSION"
  if [ "$INSTALL_OPENCLAW" -eq 0 ]; then
    return 1
  fi

  if ! command -v npm >/dev/null 2>&1; then
    log "npm is not available. Install Node/npm first."
    return 1
  fi

  log "Installing OpenClaw via npm"
  npm install -g openclaw@latest

  current="$(openclaw_version || true)"
  if [ -n "$current" ] && version_ge "$current" "$MIN_OPENCLAW_VERSION"; then
    log "OpenClaw OK after install: $current"
    return 0
  fi
  return 1
}

install_weave_plugin() {
  if [ "$INSTALL_PLUGIN" -eq 0 ]; then
    return 0
  fi
  log "Installing weave-openclaw plugin"
  openclaw plugins install weave-openclaw --force
}

install_provider_plugins() {
  if [ "$INSTALL_PROVIDER_PLUGINS" -eq 0 ]; then
    return 0
  fi
  log "Installing Taiwan eval provider plugins"
  # Anthropic, Google, Mistral, Cohere, and xAI ship as bundled OpenClaw
  # provider plugins in current OpenClaw releases. DeepSeek is installed
  # separately for the Taiwan eval harness.
  if openclaw plugins inspect deepseek --runtime --json >/tmp/nejumi-openclaw-deepseek-inspect.json 2>/tmp/nejumi-openclaw-deepseek-inspect.err; then
    if python3 - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("/tmp/nejumi-openclaw-deepseek-inspect.json").read_text(encoding="utf-8"))
status = payload.get("plugin", {}).get("status")
raise SystemExit(0 if status == "loaded" else 1)
PY
    then
      log "DeepSeek provider plugin already loaded"
      return 0
    fi
  fi
  openclaw plugins install @openclaw/deepseek-provider
}

repair_weave_otel_dependencies() {
  local project_dir="${OPENCLAW_STATE_DIR:-$HOME/.openclaw}/npm/projects/weave-openclaw"
  if [ ! -d "$project_dir" ]; then
    return 0
  fi
  if ! command -v npm >/dev/null 2>&1; then
    log "npm is not available; skipping weave-openclaw OpenTelemetry dependency repair."
    return 0
  fi
  log "Pinning weave-openclaw OpenTelemetry packages to $WEAVE_OTEL_VERSION"
  (
    cd "$project_dir"
    npm pkg set \
      "overrides.@opentelemetry/core=$WEAVE_OTEL_VERSION" \
      "overrides.@opentelemetry/resources=$WEAVE_OTEL_VERSION" \
      "overrides.@opentelemetry/sdk-trace-base=$WEAVE_OTEL_VERSION" >/dev/null
    npm install --omit=dev
  )
  patch_weave_genai_provider "$project_dir"
  patch_weave_openclaw_content_hooks "$project_dir"
  patch_weave_openclaw_private_content_and_flush "$project_dir"
  verify_weave_genai_exporter_local "$project_dir"
  verify_weave_openclaw_plugin_content "$project_dir"
}

patch_weave_genai_provider() {
  local project_dir="$1"
  python3 - "$project_dir" <<'PY'
import sys
from pathlib import Path

project_dir = Path(sys.argv[1])
targets = [
    project_dir / "node_modules" / "weave" / "dist" / "genai" / "provider.js",
    project_dir / "node_modules" / "weave" / "dist" / "genai" / "provider.mjs",
]
patched = []
for path in targets:
    if not path.exists():
        continue
    text = path.read_text(encoding="utf-8")
    if "addSpanProcessor(spanProcessor)" in text:
        continue
    if "sdk_trace_base_1.BasicTracerProvider" in text:
        old = """    _providerHolder.provider = new sdk_trace_base_1.BasicTracerProvider({
        resource,
        spanProcessors: [buildSpanProcessor(client)],
    });
"""
        new = """    const spanProcessor = buildSpanProcessor(client);
    _providerHolder.provider = new sdk_trace_base_1.BasicTracerProvider({
        resource,
        spanProcessors: [spanProcessor],
    });
    // OpenTelemetry JS 1.x ignores the constructor-level spanProcessors option
    // and requires addSpanProcessor(). OpenTelemetry JS 2.x removed that legacy
    // method and uses the constructor option. Support both shapes so the Weave
    // GenAI exporter actually receives ended spans.
    if (typeof _providerHolder.provider.addSpanProcessor === "function") {
        _providerHolder.provider.addSpanProcessor(spanProcessor);
    }
"""
    else:
        old = """    _providerHolder.provider = new BasicTracerProvider({
        resource,
        spanProcessors: [buildSpanProcessor(client)],
    });
"""
        new = """    const spanProcessor = buildSpanProcessor(client);
    _providerHolder.provider = new BasicTracerProvider({
        resource,
        spanProcessors: [spanProcessor],
    });
    // OpenTelemetry JS 1.x ignores the constructor-level spanProcessors option
    // and requires addSpanProcessor(). OpenTelemetry JS 2.x removed that legacy
    // method and uses the constructor option. Support both shapes so the Weave
    // GenAI exporter actually receives ended spans.
    if (typeof _providerHolder.provider.addSpanProcessor === "function") {
        _providerHolder.provider.addSpanProcessor(spanProcessor);
    }
"""
    if old not in text:
        raise SystemExit(f"Could not find expected provider construction block in {path}")
    path.write_text(text.replace(old, new), encoding="utf-8")
    patched.append(str(path))
if patched:
    print("patched weave provider:", ", ".join(patched))
PY
}

patch_weave_openclaw_content_hooks() {
  local project_dir="$1"
  python3 - "$project_dir" <<'PY'
import sys
from pathlib import Path

project_dir = Path(sys.argv[1])
plugin_dir = project_dir / "node_modules" / "weave-openclaw"
if not plugin_dir.exists():
    raise SystemExit(f"weave-openclaw package not found under {project_dir}")

patched: list[str] = []


def write_if_changed(path: Path, text: str, original: str) -> None:
    if text == original:
        return
    path.write_text(text, encoding="utf-8")
    patched.append(str(path))


chat_path = plugin_dir / "dist" / "src" / "handlers" / "diagnostic" / "chat.js"
chat_text = chat_path.read_text(encoding="utf-8")
original_chat = chat_text
if 'import { beginModelCall } from "../../state/hook-state.js";' not in chat_text:
    chat_text = chat_text.replace(
        'import { finalizeChatSpan } from "../llm-state.js";\n',
        'import { finalizeChatSpan } from "../llm-state.js";\n'
        'import { beginModelCall } from "../../state/hook-state.js";\n',
    )
if "Nejumi patch: correlate diagnostic chat spans" not in chat_text:
    old = """            const turn = deps.registries.turns.get(event.runId);
            if (!turn)
                return;
            const llm = runIsolated(() => turn.startLLM({
"""
    new = """            const turn = deps.registries.turns.get(event.runId);
            if (!turn)
                return;
            // Nejumi patch: correlate diagnostic chat spans with typed content hooks.
            beginModelCall(deps.hookState, event.runId, event.callId);
            const llm = runIsolated(() => turn.startLLM({
"""
    if old not in chat_text:
        raise SystemExit(f"Could not find diagnostic chat start block in {chat_path}")
    chat_text = chat_text.replace(old, new)
if (
    "Nejumi patch: defer chat span closure until run completion" not in chat_text
    and "captured by llm_output is flushed before the span ends" not in chat_text
    and "onChatFinalize(event, status, errorType, privateData = {})" not in chat_text
):
    old = """        onChatFinalize(event, status, errorType) {
            finalizeChatSpan(deps, event.runId, event.callId, status, errorType);
        },
"""
    new = """        onChatFinalize(_event, _status, _errorType) {
            // Nejumi patch: OpenClaw 2026.6 emits llm_output after model.call.completed.
            // Defer closing chat spans to the run.completed backstop so content/usage
            // captured by llm_output is flushed before the span ends.
        },
"""
    if old not in chat_text:
        raise SystemExit(f"Could not find diagnostic chat finalize block in {chat_path}")
    chat_text = chat_text.replace(old, new)
write_if_changed(chat_path, chat_text, original_chat)

llm_path = plugin_dir / "dist" / "src" / "handlers" / "hooks" / "llm.js"
llm_text = llm_path.read_text(encoding="utf-8")
original_llm = llm_text
if "llm_output(event)" not in llm_text:
    old = """        llm_input(event) {
            const capture = {
                systemPrompt: event.systemPrompt,
                prompt: event.prompt,
                historyMessages: event.historyMessages,
            };
            const callId = resolveCurrentCallId(deps.hookState, event.runId);
            if (callId) {
                captureLlmInput(deps.hookState, callId, capture);
            }
            else {
                bufferPendingLlmInputForRun(deps.hookState, event.runId, capture);
            }
        },
        // Assistant message fires just before this call's model.call.completed; capture
"""
    new = """        llm_input(event) {
            const capture = {
                systemPrompt: event.systemPrompt,
                prompt: event.prompt,
                historyMessages: event.historyMessages,
            };
            const callId = resolveCurrentCallId(deps.hookState, event.runId);
            if (callId) {
                captureLlmInput(deps.hookState, callId, capture);
            }
            else {
                bufferPendingLlmInputForRun(deps.hookState, event.runId, capture);
            }
        },
        llm_output(event) {
            const callId = resolveCurrentCallId(deps.hookState, event.runId);
            if (!callId)
                return;
            const assistantText = extractAssistantTexts(event.assistantTexts)
                ?? extractAssistantText(event.lastAssistant?.content);
            captureAssistantOutput(deps.hookState, callId, {
                text: assistantText,
                usage: event.usage,
            });
            if (event.prompt && !deps.hookState.llmInputs.get(callId)) {
                captureLlmInput(deps.hookState, callId, {
                    prompt: event.prompt,
                    historyMessages: [],
                });
            }
        },
        // Assistant message fires just before this call's model.call.completed; capture
"""
    if old not in llm_text:
        raise SystemExit(f"Could not find llm_input block in {llm_path}")
    llm_text = llm_text.replace(old, new)
if "function extractAssistantTexts" not in llm_text:
    old = """// Assistant text from the message content blocks. Tool calls get their own
// execute_tool spans, so only text lands on the chat span.
function extractAssistantText(content) {
    const text = content
        .filter((b) => b.type === "text")
        .map((b) => b.text)
        .join("");
    return text || undefined;
}
//# sourceMappingURL=llm.js.map"""
    new = """function extractAssistantTexts(texts) {
    if (!Array.isArray(texts))
        return undefined;
    const text = texts.filter((part) => typeof part === "string" && part.trim()).join("\\n\\n");
    return text || undefined;
}
// Assistant text from the message content blocks. Tool calls get their own
// execute_tool spans, so only text lands on the chat span.
function extractAssistantText(content) {
    if (typeof content === "string")
        return content || undefined;
    if (!Array.isArray(content))
        return undefined;
    const text = content
        .filter((b) => b && typeof b === "object" && b.type === "text")
        .map((b) => b.text)
        .filter((text) => typeof text === "string")
        .join("");
    return text || undefined;
}
//# sourceMappingURL=llm.js.map"""
    if old not in llm_text:
        raise SystemExit(f"Could not find assistant text helper block in {llm_path}")
    llm_text = llm_text.replace(old, new)
write_if_changed(llm_path, llm_text, original_llm)

index_path = plugin_dir / "dist" / "index.js"
index_text = index_path.read_text(encoding="utf-8")
original_index = index_text
if 'api.on("llm_output"' not in index_text:
    old = """        api.on("model_call_started", (event, ctx) => hooks.model_call_started?.(event, ctx));
        api.on("llm_input", (event, ctx) => hooks.llm_input?.(event, ctx));
        api.on("before_message_write", (event, ctx) => hooks.before_message_write?.(event, ctx));
"""
    new = """        api.on("model_call_started", (event, ctx) => hooks.model_call_started?.(event, ctx));
        api.on("llm_input", (event, ctx) => hooks.llm_input?.(event, ctx));
        api.on("llm_output", (event, ctx) => hooks.llm_output?.(event, ctx));
        api.on("before_message_write", (event, ctx) => hooks.before_message_write?.(event, ctx));
"""
    if old not in index_text:
        raise SystemExit(f"Could not find typed hook registration block in {index_path}")
    index_text = index_text.replace(old, new)
write_if_changed(index_path, index_text, original_index)

for path, needle in (
    (chat_path, "beginModelCall(deps.hookState, event.runId, event.callId)"),
    (llm_path, "llm_output(event)"),
    (index_path, 'api.on("llm_output"'),
):
    if needle not in path.read_text(encoding="utf-8"):
        raise SystemExit(f"Missing expected weave-openclaw content hook patch in {path}: {needle}")

if patched:
    print("patched weave-openclaw content hooks:", ", ".join(patched))
else:
    print("weave-openclaw content hooks already patched")
PY
}

patch_weave_openclaw_private_content_and_flush() {
  local project_dir="$1"
  python3 - "$project_dir" <<'PY'
import sys
from pathlib import Path

project_dir = Path(sys.argv[1])
plugin_dir = project_dir / "node_modules" / "weave-openclaw"
if not plugin_dir.exists():
    raise SystemExit(f"weave-openclaw package not found under {project_dir}")

patched: list[str] = []


def write_if_changed(path: Path, text: str, original: str) -> None:
    if text == original:
        return
    path.write_text(text, encoding="utf-8")
    patched.append(str(path))


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old in text:
        return text.replace(old, new, 1)
    if new in text:
        return text
    raise SystemExit(f"Could not find expected block for {label}")


chat_path = plugin_dir / "dist" / "src" / "handlers" / "diagnostic" / "chat.js"
chat_text = chat_path.read_text(encoding="utf-8")
original_chat = chat_text
if "recordModelContent(deps, llm, privateData?.modelContent" not in chat_text:
    chat_text = chat_text.replace("onChatStart(event) {", "onChatStart(event, privateData = {}) {", 1)
    chat_text = replace_once(
        chat_text,
        """            const llm = runIsolated(() => turn.startLLM({
                model: event.model,
                providerName: event.provider,
            }));
            // Default ok; a call with no model.call.completed/error before run.completed closes ok.
""",
        """            const llm = runIsolated(() => turn.startLLM({
                model: event.model,
                providerName: event.provider,
            }));
            if (llm.span?.updateName && event.model)
                llm.span.updateName(`chat ${event.model}`);
            recordModelContent(deps, llm, privateData?.modelContent, { input: true, output: false });
            // Default ok; a call with no model.call.completed/error before run.completed closes ok.
""",
        f"{chat_path}: private model start content",
    )
if "onChatFinalize(event, status, errorType, privateData = {})" not in chat_text:
    chat_text = replace_once(
        chat_text,
        """        onChatFinalize(_event, _status, _errorType) {
            // Nejumi patch: OpenClaw 2026.6 emits llm_output after model.call.completed.
            // Defer closing chat spans to the run.completed backstop so content/usage
            // captured by llm_output is flushed before the span ends.
        },
""",
        """        onChatFinalize(event, status, errorType, privateData = {}) {
            const handle = deps.registries.calls.get(event.callId);
            if (handle) {
                recordModelContent(deps, handle.llm, privateData?.modelContent, { input: true, output: true });
                handle.status = status;
                handle.errorType = errorType;
            }
            // Nejumi patch: OpenClaw 2026.6 emits llm_output after model.call.completed.
            // Keep the chat span open until run.completed so typed llm_output hooks
            // can attach provider usage and final assistant text before export.
        },
""",
        f"{chat_path}: private model finalize content",
    )
if "function recordModelContent" not in chat_text:
    chat_text = replace_once(
        chat_text,
        """}
//# sourceMappingURL=chat.js.map""",
        """}
function setJsonAttr(span, key, value) {
    if (!span?.setAttribute || value === undefined)
        return;
    try {
        span.setAttribute(key, JSON.stringify(value));
    }
    catch {}
}
function recordModelContent(deps, llm, modelContent, opts) {
    if (!deps.getResolved()?.captureContent || !modelContent)
        return;
    const record = {};
    if (opts.input) {
        const input = normalizeMessages(modelContent.inputMessages);
        if (typeof modelContent.systemPrompt === "string" && modelContent.systemPrompt.trim()) {
            input.unshift({ role: "system", content: modelContent.systemPrompt });
        }
        if (input.length)
            record.inputMessages = input;
    }
    if (opts.output) {
        const output = normalizeMessages(modelContent.outputMessages);
        if (output.length)
            record.outputMessages = output;
    }
    if (Object.keys(record).length) {
        llm.record(record);
        if (record.inputMessages)
            setJsonAttr(llm.span, "gen_ai.input.messages", record.inputMessages);
        if (record.outputMessages)
            setJsonAttr(llm.span, "gen_ai.output.messages", record.outputMessages);
    }
}
function normalizeMessages(value) {
    if (!Array.isArray(value))
        return [];
    return value.map(normalizeMessage).filter(Boolean);
}
function normalizeMessage(message) {
    if (typeof message === "string")
        return { role: "user", content: message };
    if (!message || typeof message !== "object")
        return undefined;
    const rawRole = typeof message.role === "string" ? message.role : "user";
    const role = rawRole === "toolResult" ? "tool" : ["system", "user", "assistant", "tool"].includes(rawRole) ? rawRole : "user";
    const out = { role };
    const contentText = contentToText(message.content);
    const parts = contentToParts(message.content);
    if (contentText)
        out.content = contentText;
    if (parts.length)
        out.parts = parts;
    if (typeof message.toolCallId === "string")
        out.toolCallId = message.toolCallId;
    if (typeof message.toolName === "string")
        out.toolName = message.toolName;
    return out.content || out.parts || out.toolCallId ? out : { role, content: "" };
}
function contentToText(content) {
    if (typeof content === "string")
        return content;
    if (!Array.isArray(content))
        return undefined;
    const text = content
        .filter((part) => part && typeof part === "object" && part.type === "text" && typeof part.text === "string")
        .map((part) => part.text)
        .join("");
    return text || undefined;
}
function contentToParts(content) {
    if (typeof content === "string")
        return content ? [{ type: "text", content }] : [];
    if (!Array.isArray(content))
        return [];
    const parts = [];
    for (const part of content) {
        if (!part || typeof part !== "object")
            continue;
        if (part.type === "text" && typeof part.text === "string") {
            parts.push({ type: "text", content: part.text });
        }
        else if (part.type === "toolCall") {
            parts.push({
                type: "tool_call",
                toolCallId: String(part.id ?? ""),
                toolName: String(part.name ?? "unknown"),
                arguments: safeJsonString(part.arguments ?? part.partialJson),
            });
        }
        else if (part.type === "toolResult") {
            parts.push({
                type: "tool_result",
                toolCallId: String(part.toolCallId ?? ""),
                result: safeJsonString(part.result ?? part.text ?? part.content),
            });
        }
    }
    return parts;
}
function safeJsonString(value) {
    if (value === undefined)
        return undefined;
    if (typeof value === "string")
        return value;
    try {
        return JSON.stringify(value);
    }
    catch {
        return String(value);
    }
}
//# sourceMappingURL=chat.js.map""",
        f"{chat_path}: private model helpers",
    )
write_if_changed(chat_path, chat_text, original_chat)

llm_state_path = plugin_dir / "dist" / "src" / "handlers" / "llm-state.js"
llm_state_text = llm_state_path.read_text(encoding="utf-8")
original_llm_state = llm_state_text
if "Nejumi patch: mirror messages and usage onto OTEL attributes" not in llm_state_text:
    llm_state_text = replace_once(
        llm_state_text,
        """import { totalPromptTokens } from "./util.js";
// Close this chat span at model.call.completed/error (not run end) so its input
""",
        """import { totalPromptTokens } from "./util.js";
function setJsonAttr(span, key, value) {
    if (!span?.setAttribute || value === undefined)
        return;
    try {
        span.setAttribute(key, JSON.stringify(value));
    }
    catch {}
}
function setIntAttr(span, key, value) {
    if (!span?.setAttribute || typeof value !== "number" || !Number.isFinite(value) || value < 0)
        return;
    span.setAttribute(key, Math.trunc(value));
}
// Nejumi patch: mirror messages and usage onto OTEL attributes.
// Close this chat span at model.call.completed/error (not run end) so its input
""",
        f"{llm_state_path}: attribute helpers",
    )
    llm_state_text = replace_once(
        llm_state_text,
        """    for (const callId of callIds ?? [])
        closeChatSpan(deps, callId, "ok", undefined);
}
""",
        """    for (const callId of callIds ?? []) {
        const handle = deps.registries.calls.get(callId);
        closeChatSpan(deps, callId, handle?.status ?? "ok", handle?.errorType);
    }
}
""",
        f"{llm_state_path}: preserve terminal status",
    )
    llm_state_text = replace_once(
        llm_state_text,
        """    if (shaped.input.length || shaped.output.length || usage) {
        handle.llm.record({
            inputMessages: shaped.input,
            outputMessages: shaped.output,
            ...(usage ? { usage } : {}),
        });
    }
    handle.llm.end(status === "error"
""",
        """    if (shaped.input.length || shaped.output.length || usage) {
        handle.llm.record({
            inputMessages: shaped.input,
            outputMessages: shaped.output,
            ...(usage ? { usage } : {}),
        });
    }
    setJsonAttr(handle.llm.span, "gen_ai.input.messages", handle.llm.inputMessages);
    setJsonAttr(handle.llm.span, "gen_ai.output.messages", handle.llm.outputMessages);
    setIntAttr(handle.llm.span, "gen_ai.usage.input_tokens", handle.llm.usage?.inputTokens);
    setIntAttr(handle.llm.span, "gen_ai.usage.output_tokens", handle.llm.usage?.outputTokens);
    setIntAttr(handle.llm.span, "gen_ai.usage.total_tokens", handle.llm.usage?.totalTokens);
    setIntAttr(handle.llm.span, "gen_ai.usage.cache_read.input_tokens", handle.llm.usage?.cacheReadInputTokens);
    setIntAttr(handle.llm.span, "gen_ai.usage.cache_creation.input_tokens", handle.llm.usage?.cacheCreationInputTokens);
    handle.llm.end(status === "error"
""",
        f"{llm_state_path}: OTEL attributes",
    )
    llm_state_text = llm_state_text.replace(
        'out.input.push({ role: "system", content: capture.input.systemPrompt });',
        'out.input.push(textMessage("system", capture.input.systemPrompt));',
    )
    llm_state_text = llm_state_text.replace("out.input.push(m);", "out.input.push(normalizeMessageForOtel(m));")
    llm_state_text = llm_state_text.replace(
        'out.input.push({ role: "user", content: capture.input.prompt });',
        'out.input.push(textMessage("user", capture.input.prompt));',
    )
    llm_state_text = llm_state_text.replace(
        'out.output.push({ role: "assistant", content: capture.text });',
        'out.output.push(textMessage("assistant", capture.text));',
    )
    llm_state_text = replace_once(
        llm_state_text,
        """    return out;
}
function toUsage(raw) {
""",
        """    return out;
}
function textMessage(role, content) {
    const text = typeof content === "string" ? content : String(content ?? "");
    return { role, content: text, parts: text ? [{ type: "text", content: text }] : [] };
}
function normalizeMessageForOtel(message) {
    const role = typeof message.role === "string" ? message.role : "user";
    if (typeof message.content === "string")
        return textMessage(role, message.content);
    return message;
}
function toUsage(raw) {
""",
        f"{llm_state_path}: OTEL message shaping",
    )
if "existingInputMessages" not in llm_state_text:
    llm_state_text = replace_once(
        llm_state_text,
        """    if (shaped.input.length || shaped.output.length || usage) {
        handle.llm.record({
            inputMessages: shaped.input,
            outputMessages: shaped.output,
            ...(usage ? { usage } : {}),
        });
    }
    setJsonAttr(handle.llm.span, "gen_ai.input.messages", handle.llm.inputMessages);
""",
        """    const existingInputMessages = Array.isArray(handle.llm.inputMessages) ? handle.llm.inputMessages : [];
    const existingOutputMessages = Array.isArray(handle.llm.outputMessages) ? handle.llm.outputMessages : [];
    if (shaped.input.length || shaped.output.length || usage) {
        handle.llm.record({
            inputMessages: [...existingInputMessages, ...shaped.input],
            outputMessages: [...existingOutputMessages, ...shaped.output],
            ...(usage ? { usage } : {}),
        });
    }
    setJsonAttr(handle.llm.span, "gen_ai.input.messages", handle.llm.inputMessages);
""",
        f"{llm_state_path}: merge private and typed model content",
    )
write_if_changed(llm_state_path, llm_state_text, original_llm_state)

tool_path = plugin_dir / "dist" / "src" / "handlers" / "diagnostic" / "tool.js"
tool_text = tool_path.read_text(encoding="utf-8")
original_tool = tool_text
if "Nejumi patch: record private diagnostic tool content" not in tool_text:
    tool_text = replace_once(
        tool_text,
        "export function createToolDiagnosticHandlers(deps) {",
        "// Nejumi patch: record private diagnostic tool content.\nexport function createToolDiagnosticHandlers(deps) {",
        f"{tool_path}: private tool marker",
    )
    tool_text = replace_once(
        tool_text,
        """            const captured = lookupToolCall(deps.hookState, event.toolCallId).args;
            const args = resolved.captureContent
                ? safeJson(captured?.params ?? event.paramsSummary)
                : undefined;
            const tool = runIsolated(() => turn.startTool({
                name: event.toolName ?? captured?.toolName ?? "unknown",
                toolCallId: event.toolCallId,
                args,
            }));
            deps.registries.tools.set(event.toolCallId, tool);
        },
        onToolFinalize(event, status, errorType) {
""",
        """            const captured = lookupToolCall(deps.hookState, event.toolCallId).args;
            const toolName = event.toolName ?? captured?.toolName ?? "unknown";
            const args = resolved.captureContent
                ? safeJson(captured?.params ?? event.paramsSummary)
                : undefined;
            const tool = runIsolated(() => turn.startTool({
                name: toolName,
                toolCallId: event.toolCallId,
                args,
            }));
            if (tool.span?.updateName)
                tool.span.updateName(`execute_tool ${toolName}`);
            deps.registries.tools.set(event.toolCallId, tool);
        },
        onToolFinalize(event, status, errorType, privateData = {}) {
""",
        f"{tool_path}: private tool start/finalize",
    )
    tool_text = replace_once(
        tool_text,
        "            finalizeTool(deps, event.toolCallId);\n",
        "            finalizeTool(deps, event.toolCallId, { toolContent: privateData?.toolContent });\n",
        f"{tool_path}: private tool content dispatch",
    )
    tool_text = replace_once(
        tool_text,
        """    const captured = lookupToolCall(deps.hookState, toolCallId).result;
    if (!opts.force && captured === undefined)
        return; // after_tool_call not in yet
    if (deps.getResolved()?.captureContent) {
        const result = safeJson(captured?.result);
        if (result !== undefined)
            tool.result = result;
    }
""",
        """    const lookup = lookupToolCall(deps.hookState, toolCallId);
    const captured = lookup.result;
    const toolContent = opts.toolContent;
    if (!opts.force && captured === undefined && toolContent === undefined)
        return; // after_tool_call/privateData not in yet
    if (deps.getResolved()?.captureContent) {
        const args = safeJson(toolContent?.toolInput ?? lookup.args?.params);
        if (args !== undefined && tool.span?.setAttribute)
            tool.span.setAttribute("gen_ai.tool.call.arguments", args);
        const result = safeJson(toolContent?.toolOutput ?? captured?.result);
        if (result !== undefined) {
            if (tool.span?.setAttribute)
                tool.span.setAttribute("gen_ai.tool.call.result", result);
            tool.result = result;
        }
    }
""",
        f"{tool_path}: private tool attributes",
    )
write_if_changed(tool_path, tool_text, original_tool)

plugin_path = plugin_dir / "dist" / "src" / "plugin.js"
plugin_text = plugin_path.read_text(encoding="utf-8")
original_plugin = plugin_text
if "Nejumi patch: preserve active traces across duplicate service starts" not in plugin_text:
    plugin_text = replace_once(
        plugin_text,
        """        async start(ctx) {
            logger = ctx.logger;
            if (lifecycle === "running")
                resetTransientState();
            resolved = undefined;
""",
        """        async start(ctx) {
            logger = ctx.logger;
            // Nejumi patch: preserve active traces across duplicate service starts.
            // OpenClaw re-registers this shared service when another task agent is
            // added. Clearing process-wide registries here drops concurrent runs.
            if (lifecycle === "running") {
                ctx.logger.info("weave: shared service already running; preserving active trace state");
                return;
            }
            resolved = undefined;
""",
        f"{plugin_path}: idempotent shared service start",
    )
if "Nejumi patch: expose explicit OTel flush" not in plugin_text:
    plugin_text = replace_once(
        plugin_text,
        """    const service = {
        id: "weave",
""",
        """    async function flush(reason, ctx) {
        try {
            await flushOTel();
        }
        catch (err) {
            const targetLogger = ctx?.logger ?? logger;
            targetLogger?.warn?.(`weave: flushOTel failed during ${reason}: ${err instanceof Error ? err.message : String(err)}`);
        }
    }
    // Nejumi patch: expose explicit OTel flush for short-lived OpenClaw agent commands.
    const service = {
        id: "weave",
""",
        f"{plugin_path}: explicit flush helper",
    )
    plugin_text = replace_once(
        plugin_text,
        """        async stop(ctx) {
            lifecycle = "stopped";
            try {
                await flushOTel();
            }
            catch (err) {
                ctx.logger.warn(`weave: flushOTel failed during stop: ${err instanceof Error ? err.message : String(err)}`);
            }
            resetTransientState();
        },
""",
        """        async stop(ctx) {
            lifecycle = "stopped";
            await flush("stop", ctx);
            resetTransientState();
        },
""",
        f"{plugin_path}: service.stop flush",
    )
if "diagnostic(event, meta, privateData = {})" not in plugin_text:
    plugin_text = replace_once(
        plugin_text,
        "diagnostic(event, meta) {",
        "diagnostic(event, meta, privateData = {}) {",
        f"{plugin_path}: private diagnostic signature",
    )
    for old, new in {
        "return onChatStart(event);": "return onChatStart(event, privateData);",
        'return onChatFinalize(event, "ok", undefined);': 'return onChatFinalize(event, "ok", undefined, privateData);',
        'return onChatFinalize(event, "error", event.errorCategory);': 'return onChatFinalize(event, "error", event.errorCategory, privateData);',
        'return onToolFinalize(event, "ok", undefined);': 'return onToolFinalize(event, "ok", undefined, privateData);',
        'return onToolFinalize(event, "error", event.errorCategory);': 'return onToolFinalize(event, "error", event.errorCategory, privateData);',
        'return onToolFinalize(event, "error", "blocked");': 'return onToolFinalize(event, "error", "blocked", privateData);',
    }.items():
        plugin_text = replace_once(plugin_text, old, new, f"{plugin_path}: {old}")
if "return { service, registries, getStatus, flush, handlers };" not in plugin_text:
    plugin_text = replace_once(
        plugin_text,
        "    return { service, registries, getStatus, handlers };\n",
        "    return { service, registries, getStatus, flush, handlers };\n",
        f"{plugin_path}: return flush",
    )
write_if_changed(plugin_path, plugin_text, original_plugin)

index_path = plugin_dir / "dist" / "index.js"
index_text = index_path.read_text(encoding="utf-8")
original_index = index_text
if "Nejumi patch: prefer trusted OpenClaw diagnostic subscription" not in index_text:
    index_text = replace_once(
        index_text,
        """import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { onInternalDiagnosticEvent } from "openclaw/plugin-sdk/diagnostic-runtime";
""",
        """import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { onInternalDiagnosticEvent } from "openclaw/plugin-sdk/diagnostic-runtime";
""",
        f"{index_path}: trusted diagnostic imports",
    )
    index_text = replace_once(
        index_text,
        """const PLUGIN_GLOBAL_KEY = Symbol.for("weave-openclaw.plugin");
const DIAGNOSTIC_SUBSCRIBED_KEY = Symbol.for("weave-openclaw.diagnosticSubscribed");
function getOrCreateSharedPlugin(pluginConfig) {
""",
        """const PLUGIN_GLOBAL_KEY = Symbol.for("weave-openclaw.plugin");
const DIAGNOSTIC_SUBSCRIBED_KEY = Symbol.for("weave-openclaw.diagnosticSubscribed");
// Nejumi patch: prefer trusted OpenClaw diagnostic subscription so private
// model/tool content reaches W&B Agents traces. File names are build-hashed,
// so resolve the OpenClaw package at runtime instead of hard-coding a path.
const onTrustedInternalDiagnosticEvent = resolveTrustedInternalDiagnosticEvent();
function candidateOpenClawDistDirs() {
    const dirs = [];
    try {
        const require = createRequire(import.meta.url);
        dirs.push(path.join(path.dirname(require.resolve("openclaw/package.json")), "dist"));
    }
    catch {}
    const stateDir = process.env.OPENCLAW_STATE_DIR || path.join(os.homedir(), ".openclaw");
    const npmDir = path.join(stateDir, "npm");
    try {
        for (const entry of fs.readdirSync(npmDir, { withFileTypes: true })) {
            if (entry.isDirectory() && entry.name.startsWith("openclaw-"))
                dirs.push(path.join(npmDir, entry.name, "node_modules", "openclaw", "dist"));
        }
    }
    catch {}
    dirs.push(path.resolve(path.dirname(process.execPath), "..", "lib", "node_modules", "openclaw", "dist"));
    return [...new Set(dirs)].filter((dir) => fs.existsSync(dir));
}
function resolveTrustedInternalDiagnosticEvent() {
    const require = createRequire(import.meta.url);
    for (const dir of candidateOpenClawDistDirs()) {
        let files = [];
        try {
            files = fs.readdirSync(dir).filter((name) => /^diagnostic-events-.*\\.js$/.test(name)).sort().reverse();
        }
        catch {
            continue;
        }
        for (const file of files) {
            try {
                const mod = require(path.join(dir, file));
                const trusted = mod.onTrustedInternalDiagnosticEvent ?? mod.h;
                if (typeof trusted === "function")
                    return trusted;
            }
            catch {}
        }
    }
    return (handler) => onInternalDiagnosticEvent((event, meta) => handler(event, meta, {}));
}
function getOrCreateSharedPlugin(pluginConfig) {
""",
        f"{index_path}: trusted diagnostic resolver",
    )
    index_text = replace_once(
        index_text,
        "        onInternalDiagnosticEvent(plugin.handlers.diagnostic);\n",
        "        onTrustedInternalDiagnosticEvent((event, meta, privateData) => plugin.handlers.diagnostic(event, meta, privateData));\n",
        f"{index_path}: trusted diagnostic subscription",
    )
if "await resolveTrustedInternalDiagnosticEvent" in index_text:
    index_text = index_text.replace(
        "const onTrustedInternalDiagnosticEvent = await resolveTrustedInternalDiagnosticEvent();",
        "const onTrustedInternalDiagnosticEvent = resolveTrustedInternalDiagnosticEvent();",
    )
    index_text = index_text.replace(
        "async function resolveTrustedInternalDiagnosticEvent() {\n    for (const dir of candidateOpenClawDistDirs()) {",
        "function resolveTrustedInternalDiagnosticEvent() {\n    const require = createRequire(import.meta.url);\n    for (const dir of candidateOpenClawDistDirs()) {",
    )
    index_text = index_text.replace(
        "const mod = await import(pathToFileURL(path.join(dir, file)).href);",
        "const mod = require(path.join(dir, file));",
    )
if 'plugin.flush?.("agent_end"' not in index_text:
    index_text = replace_once(
        index_text,
        """        api.on("session_start", (event, ctx) => hooks.session_start?.(event, ctx));
        api.on("session_end", (event, ctx) => hooks.session_end?.(event, ctx));
""",
        """        api.on("session_start", (event, ctx) => hooks.session_start?.(event, ctx));
        api.on("session_end", async (event, ctx) => {
            const result = await hooks.session_end?.(event, ctx);
            await plugin.flush?.("session_end", ctx);
            return result;
        });
""",
        f"{index_path}: session_end flush",
    )
    index_text = replace_once(
        index_text,
        """        api.on("agent_end", (event, ctx) => hooks.agent_end?.(event, ctx));
        api.on("message_received", (event, ctx) => hooks.message_received?.(event, ctx));
""",
        """        api.on("agent_end", async (event, ctx) => {
            const result = await hooks.agent_end?.(event, ctx);
            await plugin.flush?.("agent_end", ctx);
            return result;
        });
        api.on("message_received", (event, ctx) => hooks.message_received?.(event, ctx));
""",
        f"{index_path}: agent_end flush",
    )
write_if_changed(index_path, index_text, original_index)

for path, needle in (
    (chat_path, "recordModelContent(deps, llm, privateData?.modelContent"),
    (llm_state_path, 'setJsonAttr(handle.llm.span, "gen_ai.input.messages"'),
    (tool_path, "toolContent?.toolOutput ?? captured?.result"),
    (plugin_path, "async function flush(reason, ctx)"),
    (plugin_path, "preserve active traces across duplicate service starts"),
    (plugin_path, "return { service, registries, getStatus, flush, handlers };"),
    (index_path, "resolveTrustedInternalDiagnosticEvent"),
    (index_path, 'plugin.flush?.("agent_end"'),
):
    if needle not in path.read_text(encoding="utf-8"):
        raise SystemExit(f"Missing expected private-content/flush patch in {path}: {needle}")

if patched:
    print("patched weave-openclaw private content/flush:", ", ".join(patched))
else:
    print("weave-openclaw private content/flush already patched")
PY
}

verify_weave_genai_exporter_local() {
  local project_dir="$1"
  if [ ! -d "$project_dir/node_modules/weave" ]; then
    return 0
  fi
  if ! command -v node >/dev/null 2>&1; then
    return 0
  fi
  log "Verifying weave GenAI OTLP exporter and content capture against a local collector"
  (
    cd "$project_dir"
    WANDB_API_KEY="${WANDB_API_KEY:-nejumi-local-otel-test}" node --input-type=module <<'NODE'
import http from "node:http";
import { init, startTurn, flushOTel } from "weave";

function readBody(req) {
  return new Promise((resolve) => {
    const chunks = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end", () => resolve(Buffer.concat(chunks)));
  });
}

let otlpBytes = 0;
const otlpChunks = [];
const server = http.createServer(async (req, res) => {
  const body = await readBody(req);
  if (req.url === "/agents/otel/v1/traces") {
    otlpBytes += body.length;
    otlpChunks.push(body);
  }
  res.writeHead(200, {"content-type": "application/json"});
  res.end("{}");
});

await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const {port} = server.address();
process.env.WF_TRACE_SERVER_URL = `http://127.0.0.1:${port}`;
process.env.WANDB_API_KEY ||= "nejumi-local-otel-test";

await init("llm-leaderboard/tc-leaderboard", {
  genai: {batchOptions: {scheduledDelayMillis: 20, maxExportBatchSize: 1}},
});
const marker = `NEJUMI_LOCAL_CONTENT_${Date.now()}`;
const turn = startTurn({agentName: "nejumi-openclaw-local-check", model: "local-check"});
const llm = turn.startLLM({model: "local-model", providerName: "local"});
llm.record({
  inputMessages: [{role: "user", content: `${marker}_INPUT`}],
  outputMessages: [{role: "assistant", content: `${marker}_OUTPUT`}],
  usage: {inputTokens: 1, outputTokens: 2},
});
llm.end();
const tool = turn.startTool({name: "exec", toolCallId: "nejumi-local-tool", args: `${marker}_TOOL_ARGS`});
tool.result = `${marker}_TOOL_RESULT`;
tool.end();
turn.end();
await flushOTel();
await new Promise((resolve) => setTimeout(resolve, 50));
server.close();

if (otlpBytes <= 0) {
  throw new Error("weave GenAI exporter produced no OTLP payload");
}
const payloadText = Buffer.concat(otlpChunks).toString("utf8");
const requiredMarkers = [
  `${marker}_INPUT`,
  `${marker}_OUTPUT`,
  `${marker}_TOOL_ARGS`,
  `${marker}_TOOL_RESULT`,
];
const missing = requiredMarkers.filter((value) => !payloadText.includes(value));
if (missing.length) {
  throw new Error(`weave GenAI local OTLP payload omitted content markers: ${missing.join(", ")}`);
}
console.log(`local OTLP payload bytes: ${otlpBytes}; content markers present`);
NODE
  )
}

verify_weave_openclaw_plugin_content() {
  local project_dir="$1"
  if [ ! -d "$project_dir/node_modules/weave-openclaw" ]; then
    return 0
  fi
  if ! command -v node >/dev/null 2>&1; then
    return 0
  fi
  log "Verifying weave-openclaw plugin event content replay against a local collector"
  node scripts/tools/verify_openclaw_weave_plugin_content.mjs \
    --plugin-project-dir "$project_dir" \
    --json >/tmp/nejumi-weave-openclaw-plugin-content.json
  python3 - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("/tmp/nejumi-weave-openclaw-plugin-content.json").read_text(encoding="utf-8"))
inner = payload.get("payload", {})
if not payload.get("ok"):
    raise SystemExit("weave-openclaw plugin content replay failed")
print(
    "weave-openclaw plugin content replay OK: "
    f"{inner.get('bytes')} OTLP bytes; "
    "input/output/tool markers present"
)
PY
}

verify_openclaw_runtime_content_hooks() {
  if ! command -v python3 >/dev/null 2>&1; then
    return 0
  fi
  log "Verifying OpenClaw runtime hook/content contract"
  python3 scripts/tools/verify_openclaw_runtime_content_hooks.py \
    --json >/tmp/nejumi-openclaw-runtime-content-hooks.json
  python3 - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("/tmp/nejumi-openclaw-runtime-content-hooks.json").read_text(encoding="utf-8"))
if not payload.get("ok"):
    failed = [
        f"{check.get('hook')} missing={','.join(check.get('missing_terms') or [])}"
        for check in payload.get("checks", [])
        if not check.get("ok")
    ]
    raise SystemExit("OpenClaw runtime hook/content contract failed: " + "; ".join(failed))
print("OpenClaw runtime hook/content contract OK")
PY
}

check_gateway_env_dropin() {
  log "Checking OpenClaw gateway .env systemd drop-in"
  python3 scripts/setup/configure_openclaw_gateway_env.py \
    --env-file "$OPENCLAW_ENV_FILE" \
    --openclaw-config "$OPENCLAW_CONFIG" \
    --check-only \
    --json >/tmp/nejumi-openclaw-gateway-env.json
  python3 - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("/tmp/nejumi-openclaw-gateway-env.json").read_text(encoding="utf-8"))
if not payload.get("ok"):
    missing = payload.get("secret_refs_missing") or []
    dropin = payload.get("dropin_path")
    raise SystemExit(
        "OpenClaw gateway .env drop-in is not ready: "
        f"dropin={dropin} missing_secret_refs={','.join(missing)}"
    )
print("OpenClaw gateway .env drop-in OK")
PY
}

configure_gateway_env_dropin() {
  if [ "$CONFIGURE_GATEWAY_ENV" -eq 0 ]; then
    return 0
  fi
  log "Configuring OpenClaw gateway .env systemd drop-in"
  local args=(
    --env-file "$OPENCLAW_ENV_FILE"
    --openclaw-config "$OPENCLAW_CONFIG"
    --write
  )
  if [ "$RESTART_GATEWAY" -eq 1 ]; then
    args+=(--restart)
  fi
  python3 scripts/setup/configure_openclaw_gateway_env.py "${args[@]}"
}

check_weave_agents_api() {
  python3 - "$WANDB_ENTITY" "$WANDB_PROJECT" "$OPENCLAW_AGENT_NAME" <<'PY'
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

entity, project, agent_name = sys.argv[1:4]
env = os.environ.copy()
dotenv = Path(".env")
if dotenv.exists():
    for raw in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in env:
            env[key] = value.strip().strip('"').strip("'")

api_key = env.get("WANDB_API_KEY")
if not api_key:
    print("[openclaw-setup] WANDB_API_KEY not set; skipping live Agents API probe.")
    raise SystemExit(0)

payload = json.dumps({
    "project_id": f"{entity}/{project}",
    "filters": {"agent_name": agent_name},
    "limit": 20,
    "offset": 0,
}).encode("utf-8")
token = base64.b64encode(f"api:{api_key}".encode("utf-8")).decode("ascii")
req = urllib.request.Request(
    "https://trace.wandb.ai/agents/query",
    data=payload,
    method="POST",
    headers={
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json",
    },
)
try:
    with urllib.request.urlopen(req, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
except urllib.error.HTTPError as exc:
    print(f"[openclaw-setup] Agents API probe failed: HTTP {exc.code}")
    raise SystemExit(1)

count = data.get("total_count", 0)
print(f"[openclaw-setup] Agents API reachable: {entity}/{project}, agent={agent_name}, matches={count}")
PY
}

check_weave_plugin() {
  if ! command -v openclaw >/dev/null 2>&1; then
    return 1
  fi
  openclaw plugins inspect weave --runtime --json >/tmp/nejumi-openclaw-weave-inspect.json 2>/tmp/nejumi-openclaw-weave-inspect.err || return 1
  python3 - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("/tmp/nejumi-openclaw-weave-inspect.json").read_text(encoding="utf-8"))
status = payload.get("plugin", {}).get("status")
raise SystemExit(0 if status == "loaded" else 1)
PY
}

write_openclaw_config() {
  if [ "$WRITE_CONFIG" -eq 0 ]; then
    return 0
  fi

  if [ -z "${WANDB_API_KEY:-}" ]; then
    log "WANDB_API_KEY is not set. weave-openclaw can still use wandb login/.netrc, but WANDB_API_KEY is recommended for reproducible setup."
  fi
  log "Merging Weave plugin config into $OPENCLAW_CONFIG"
  mkdir -p "$(dirname "$OPENCLAW_CONFIG")"
  python3 - "$OPENCLAW_CONFIG" "$WANDB_ENTITY" "$WANDB_PROJECT" "$OPENCLAW_SERVICE_NAME" "$OPENCLAW_AGENT_NAME" "$OPENCLAW_AGENT_VERSION" "$CAPTURE_CONTENT" <<'PY'
import json
import os
import sys
from pathlib import Path

config_path = Path(sys.argv[1]).expanduser()
entity, project, service_name, agent_name, agent_version, capture_content = sys.argv[2:8]
capture_content_bool = capture_content.lower() in {"1", "true", "yes", "on"}

if config_path.exists():
    data = json.loads(config_path.read_text(encoding="utf-8"))
    backup = config_path.with_suffix(config_path.suffix + ".bak")
    backup.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
else:
    data = {}

data.setdefault("diagnostics", {})
data["diagnostics"]["enabled"] = True
plugins = data.setdefault("plugins", {})
allow = plugins.setdefault("allow", [])
if "weave" not in allow:
    allow.append("weave")
entries = plugins.setdefault("entries", {})
entries["weave"] = {
    "enabled": True,
    "config": {
        "entity": entity,
        "project": project,
        "apiKey": {"source": "env", "provider": "default", "id": "WANDB_API_KEY"},
        "serviceName": service_name,
        "agentName": agent_name,
        "agentVersion": agent_version,
        "agentDescription": "Nejumi 4.5 Taiwan agentic evaluation",
        "captureContent": capture_content_bool,
        "flushIntervalMs": 1000,
    },
    "hooks": {"allowConversationAccess": True},
}

config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(config_path)
PY
}

write_openclaw_model_config() {
  if [ "$WRITE_MODEL_CONFIG" -eq 0 ]; then
    return 0
  fi

  log "Merging Taiwan eval model providers into $OPENCLAW_CONFIG"
  mkdir -p "$(dirname "$OPENCLAW_CONFIG")"
  python3 - "$OPENCLAW_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1]).expanduser()
if config_path.exists():
    data = json.loads(config_path.read_text(encoding="utf-8"))
    backup = config_path.with_suffix(config_path.suffix + ".bak")
    backup.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
else:
    data = {}


def secret_ref(env_name: str) -> dict:
    return {"source": "env", "provider": "default", "id": env_name}


def merge_models(existing: list | None, additions: list[dict]) -> list[dict]:
    by_id = {}
    for model in existing or []:
        if isinstance(model, dict) and model.get("id"):
            by_id[str(model["id"])] = model
    for model in additions:
        merged = dict(by_id.get(model["id"], {}))
        merged.update(model)
        by_id[model["id"]] = merged
    return list(by_id.values())


plugins = data.setdefault("plugins", {})
allow = plugins.setdefault("allow", [])
for plugin_id in ("weave", "deepseek", "anthropic", "mistral", "google", "cohere", "xai"):
    if plugin_id not in allow:
        allow.append(plugin_id)
entries = plugins.setdefault("entries", {})
for plugin_id in ("deepseek", "anthropic", "mistral", "google", "cohere", "xai"):
    entries.setdefault(plugin_id, {})["enabled"] = True

providers = data.setdefault("models", {}).setdefault("providers", {})
runtime = {"id": "openclaw"}

openai_direct = providers.setdefault("openai-direct", {})
openai_direct.update(
    {
        "baseUrl": "https://api.openai.com/v1",
        "apiKey": secret_ref("OPENAI_API_KEY"),
        "auth": "api-key",
        "api": "openai-responses",
        "agentRuntime": runtime,
    }
)
openai_direct["models"] = merge_models(
    openai_direct.get("models"),
    [
        {
            "id": "gpt-5.5",
            "name": "gpt-5.5",
            "api": "openai-responses",
            "reasoning": True,
            "input": ["text"],
            "contextWindow": 400000,
            "contextTokens": 200000,
            "maxTokens": 32768,
            "agentRuntime": runtime,
        },
        {
            "id": "gpt-5.5-2026-04-23",
            "name": "gpt-5.5-2026-04-23",
            "api": "openai-responses",
            "reasoning": True,
            "input": ["text"],
            "contextWindow": 400000,
            "contextTokens": 200000,
            "maxTokens": 32768,
            "agentRuntime": runtime,
        },
        {
            "id": "gpt-5.4-2026-03-05",
            "name": "gpt-5.4-2026-03-05",
            "api": "openai-responses",
            "reasoning": True,
            "input": ["text"],
            "contextWindow": 272000,
            "maxTokens": 32768,
            "agentRuntime": runtime,
        },
        {
            "id": "gpt-5.4-mini-2026-03-17",
            "name": "gpt-5.4-mini-2026-03-17",
            "api": "openai-responses",
            "reasoning": True,
            "input": ["text"],
            "contextWindow": 400000,
            "maxTokens": 32768,
            "agentRuntime": runtime,
        },
        {
            "id": "gpt-5.4-nano-2026-03-17",
            "name": "gpt-5.4-nano-2026-03-17",
            "api": "openai-responses",
            "reasoning": True,
            "input": ["text"],
            "contextWindow": 400000,
            "maxTokens": 32768,
            "agentRuntime": runtime,
        },
        {
            "id": "gpt-5-nano-2025-08-07",
            "name": "gpt-5-nano-2025-08-07",
            "api": "openai-responses",
            "reasoning": True,
            "input": ["text"],
            "contextWindow": 200000,
            "maxTokens": 32768,
            "agentRuntime": runtime,
        },
        {
            "id": "gpt-4.1-nano-2025-04-14",
            "name": "gpt-4.1-nano-2025-04-14",
            "api": "openai-responses",
            "reasoning": False,
            "input": ["text"],
            "contextWindow": 1047576,
            "maxTokens": 32768,
            "agentRuntime": runtime,
        },
        {
            "id": "gpt-4.1-mini-2025-04-14",
            "name": "gpt-4.1-mini-2025-04-14",
            "api": "openai-responses",
            "reasoning": False,
            "input": ["text"],
            "contextWindow": 1047576,
            "maxTokens": 32768,
            "agentRuntime": runtime,
        },
        {
            "id": "gpt-5.1-2025-11-13",
            "name": "gpt-5.1-2025-11-13",
            "api": "openai-responses",
            "reasoning": True,
            "input": ["text"],
            "contextWindow": 200000,
            "maxTokens": 32768,
            "agentRuntime": runtime,
        },
        {
            "id": "gpt-5-2025-08-07",
            "name": "gpt-5-2025-08-07",
            "api": "openai-responses",
            "reasoning": True,
            "input": ["text"],
            "contextWindow": 200000,
            "maxTokens": 32768,
            "agentRuntime": runtime,
        },
        {
            "id": "o3-2025-04-16",
            "name": "o3-2025-04-16",
            "api": "openai-responses",
            "reasoning": True,
            "input": ["text"],
            "contextWindow": 200000,
            "maxTokens": 32768,
            "agentRuntime": runtime,
        },
    ],
)

deepseek = providers.setdefault("deepseek", {})
deepseek.pop("apiKey", None)
deepseek.update(
    {
        "baseUrl": "https://api.deepseek.com",
        "auth": "api-key",
        "api": "openai-completions",
        "agentRuntime": runtime,
    }
)
deepseek["models"] = merge_models(
    deepseek.get("models"),
    [
        {
            "id": "deepseek-v4-pro",
            "name": "DeepSeek V4 Pro",
            "reasoning": True,
            "input": ["text"],
            "contextWindow": 1000000,
            "maxTokens": 384000,
            "cost": {
                "input": 1.74,
                "output": 3.48,
                "cacheRead": 0.145,
                "cacheWrite": 0,
            },
            "compat": {
                "supportsUsageInStreaming": True,
                "supportsReasoningEffort": True,
                "maxTokensField": "max_tokens",
            },
        },
        {
            "id": "deepseek-v4-flash",
            "name": "DeepSeek V4 Flash",
            "reasoning": True,
            "input": ["text"],
            "contextWindow": 1000000,
            "maxTokens": 384000,
            "cost": {
                "input": 0.14,
                "output": 0.28,
                "cacheRead": 0.028,
                "cacheWrite": 0,
            },
            "compat": {
                "supportsUsageInStreaming": True,
                "supportsReasoningEffort": True,
                "maxTokensField": "max_tokens",
            },
        },
    ],
)

config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(config_path)
PY
}

run_onboard() {
  if [ "$RUN_ONBOARD" -eq 0 ]; then
    return 0
  fi
  log "Running OpenClaw onboard"
  openclaw onboard --install-daemon
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --all)
      INSTALL_NODE=1
      INSTALL_OPENCLAW=1
      INSTALL_PLUGIN=1
      INSTALL_PROVIDER_PLUGINS=1
      WRITE_CONFIG=1
      WRITE_MODEL_CONFIG=1
      CONFIGURE_GATEWAY_ENV=1
      ;;
    --check-only)
      CHECK_ONLY=1
      ;;
    --install-node)
      INSTALL_NODE=1
      ;;
    --install-openclaw)
      INSTALL_OPENCLAW=1
      ;;
    --install-plugin)
      INSTALL_PLUGIN=1
      ;;
    --install-provider-plugins)
      INSTALL_PROVIDER_PLUGINS=1
      ;;
    --write-config)
      WRITE_CONFIG=1
      ;;
    --write-model-config)
      WRITE_MODEL_CONFIG=1
      ;;
    --configure-gateway-env)
      CONFIGURE_GATEWAY_ENV=1
      ;;
    --restart-gateway)
      CONFIGURE_GATEWAY_ENV=1
      RESTART_GATEWAY=1
      ;;
    --run-onboard)
      RUN_ONBOARD=1
      ;;
    --capture-content)
      CAPTURE_CONTENT=true
      ;;
    --no-capture-content)
      CAPTURE_CONTENT=false
      ;;
    --entity)
      WANDB_ENTITY="$2"
      shift
      ;;
    --project)
      WANDB_PROJECT="$2"
      shift
      ;;
    --config)
      OPENCLAW_CONFIG="$2"
      shift
      ;;
    --env-file)
      OPENCLAW_ENV_FILE="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
  shift
done

if [ "$CHECK_ONLY" -eq 1 ]; then
  set +e
  ensure_node
  node_status=$?
  ensure_openclaw
  openclaw_status=$?
  python3 scripts/tools/run_openclaw_agent_protocol.py preflight
  preflight_status=$?
  check_weave_plugin
  weave_status=$?
  verify_weave_genai_exporter_local "${OPENCLAW_STATE_DIR:-$HOME/.openclaw}/npm/projects/weave-openclaw"
  exporter_status=$?
  verify_weave_openclaw_plugin_content "${OPENCLAW_STATE_DIR:-$HOME/.openclaw}/npm/projects/weave-openclaw"
  plugin_content_status=$?
  verify_openclaw_runtime_content_hooks
  runtime_content_status=$?
  check_gateway_env_dropin
  gateway_env_status=$?
  check_weave_agents_api
  agents_api_status=$?
  set -e
  if [ "$node_status" -eq 0 ] && [ "$openclaw_status" -eq 0 ] && [ "$preflight_status" -eq 0 ] && [ "$weave_status" -eq 0 ] && [ "$exporter_status" -eq 0 ] && [ "$plugin_content_status" -eq 0 ] && [ "$runtime_content_status" -eq 0 ] && [ "$gateway_env_status" -eq 0 ] && [ "$agents_api_status" -eq 0 ]; then
    exit 0
  fi
  exit 1
fi

ensure_node
ensure_openclaw
verify_openclaw_runtime_content_hooks
install_weave_plugin
install_provider_plugins
if [ "$INSTALL_PLUGIN" -eq 1 ] || [ "$WRITE_CONFIG" -eq 1 ]; then
  repair_weave_otel_dependencies
fi
write_openclaw_config
write_openclaw_model_config
configure_gateway_env_dropin
if [ "$INSTALL_PLUGIN" -eq 1 ] || [ "$WRITE_CONFIG" -eq 1 ]; then
  check_weave_agents_api
fi
run_onboard
python3 scripts/tools/run_openclaw_agent_protocol.py preflight

if [ "$CONFIGURE_GATEWAY_ENV" -eq 1 ] && [ "$RESTART_GATEWAY" -eq 1 ]; then
  log "Gateway env drop-in configured and openclaw-gateway.service restart requested. Open https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT/weave/agents after running a canary."
elif [ "$CONFIGURE_GATEWAY_ENV" -eq 1 ]; then
  log "Gateway env drop-in configured. Restart the OpenClaw gateway to apply it, or rerun with --restart-gateway."
else
  log "Done. Restart the OpenClaw gateway if it is already running, then open https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT/weave/agents."
fi
