#!/usr/bin/env node
/*
 * Log a completed OpenClaw agent run to W&B Weave Agents.
 *
 * The native weave-openclaw plugin is the production integration. This script
 * is a diagnostic fallback only: it turns the persisted openclaw_result.json
 * content into a GenAI invoke_agent -> chat trace and flushes it explicitly.
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {pathToFileURL} from "node:url";

function readPayload() {
  const payloadPath = process.argv[2];
  if (!payloadPath) {
    throw new Error("usage: node log_openclaw_result_to_weave.mjs PAYLOAD_JSON");
  }
  return JSON.parse(fs.readFileSync(payloadPath, "utf8"));
}

function weaveModulePath() {
  const stateDir = process.env.OPENCLAW_STATE_DIR || path.join(os.homedir(), ".openclaw");
  const candidates = [
    path.join(stateDir, "npm", "projects", "weave-openclaw", "node_modules", "weave", "dist", "index.mjs"),
    path.join(process.cwd(), "node_modules", "weave", "dist", "index.mjs"),
  ];
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  throw new Error(
    "Could not find the Weave JS SDK. Run scripts/setup/install_openclaw_weave.sh --all first."
  );
}

function cleanString(value) {
  if (typeof value !== "string") {
    return "";
  }
  return value;
}

function cleanAttrValue(value) {
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return value;
  }
  return JSON.stringify(value);
}

function setAttrs(spanLike, attrs) {
  for (const [key, value] of Object.entries(attrs)) {
    const clean = cleanAttrValue(value);
    if (clean !== undefined) {
      spanLike.setAttribute(key, clean);
    }
  }
}

function usage(value) {
  if (!value || typeof value !== "object") {
    return {};
  }
  const output = {};
  for (const [key, tokenValue] of Object.entries(value)) {
    if (Number.isFinite(tokenValue)) {
      output[key] = Math.trunc(tokenValue);
    }
  }
  return output;
}

function positiveIntegerEnv(name, fallback) {
  const raw = process.env[name];
  if (!raw) {
    return fallback;
  }
  const value = Number.parseInt(raw, 10);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function weaveGenaiSettings() {
  return {
    spanProcessor: "batch",
    batchOptions: {
      maxExportBatchSize: positiveIntegerEnv("NEJUMI_WEAVE_MAX_EXPORT_BATCH_SIZE", 128),
      maxQueueSize: positiveIntegerEnv("NEJUMI_WEAVE_MAX_QUEUE_SIZE", 4096),
      scheduledDelayMillis: positiveIntegerEnv("NEJUMI_WEAVE_SCHEDULED_DELAY_MS", 1000),
      exportTimeoutMillis: positiveIntegerEnv("NEJUMI_WEAVE_EXPORT_TIMEOUT_MS", 120000),
    },
  };
}

function genAiUsageAttrs(value) {
  const normalized = usage(value);
  const attrs = {};
  if (Number.isFinite(normalized.inputTokens)) {
    attrs["gen_ai.usage.input_tokens"] = normalized.inputTokens;
  }
  if (Number.isFinite(normalized.outputTokens)) {
    attrs["gen_ai.usage.output_tokens"] = normalized.outputTokens;
  }
  if (Number.isFinite(normalized.reasoningTokens)) {
    attrs["gen_ai.usage.reasoning.output_tokens"] = normalized.reasoningTokens;
  }
  if (Number.isFinite(normalized.cacheReadInputTokens)) {
    attrs["gen_ai.usage.cache_read.input_tokens"] = normalized.cacheReadInputTokens;
  }
  if (Number.isFinite(normalized.cacheCreationInputTokens)) {
    attrs["gen_ai.usage.cache_creation.input_tokens"] = normalized.cacheCreationInputTokens;
  }
  const total =
    (normalized.inputTokens || 0) +
    (normalized.outputTokens || 0) +
    (normalized.reasoningTokens || 0);
  if (total > 0) {
    attrs["gen_ai.usage.total_tokens"] = total;
  }
  return attrs;
}

function addMessageEvents(spanLike, inputMessages, outputMessages) {
  if (!spanLike || typeof spanLike.addEvent !== "function") {
    return;
  }
  for (const message of inputMessages) {
    spanLike.addEvent("gen_ai.user.message", {
      "gen_ai.event.content": JSON.stringify(message),
    });
  }
  for (const message of outputMessages) {
    spanLike.addEvent("gen_ai.assistant.message", {
      "gen_ai.event.content": JSON.stringify(message),
    });
  }
}

function addReasoningEvent(spanLike, reasoning) {
  if (!spanLike || typeof spanLike.addEvent !== "function" || !reasoning) {
    return;
  }
  spanLike.addEvent("gen_ai.assistant.reasoning", {
    "gen_ai.event.content": reasoning,
  });
}

function toolValueToString(value) {
  if (value === undefined || value === null) {
    return "";
  }
  if (typeof value === "string") {
    return value;
  }
  return JSON.stringify(value);
}

function toolResultText(event) {
  if (!event || typeof event !== "object") {
    return "";
  }
  if (event.content) {
    return toolValueToString(event.content);
  }
  if (event.details && typeof event.details === "object" && event.details.content) {
    return toolValueToString(event.details.content);
  }
  return toolValueToString(event.details);
}

function addToolEvents(spanLike, toolEvents) {
  if (!spanLike || typeof spanLike.addEvent !== "function" || !Array.isArray(toolEvents)) {
    return;
  }
  for (const event of toolEvents) {
    if (!event || typeof event !== "object") {
      continue;
    }
    const eventName =
      event.type === "tool_result" ? "gen_ai.tool.result" : "gen_ai.tool.call";
    const attrs = {
      "gen_ai.event.content": JSON.stringify(event),
      "gen_ai.tool.call.id": cleanString(event.toolCallId),
      "gen_ai.tool.name": cleanString(event.toolName) || "unknown",
    };
    if (Number.isFinite(event.index)) {
      attrs["nejumi.tool.event.index"] = event.index;
    }
    spanLike.addEvent(eventName, attrs);
  }
}

function addTimelineEvents(spanLike, timelineEvents) {
  if (!spanLike || typeof spanLike.addEvent !== "function" || !Array.isArray(timelineEvents)) {
    return false;
  }
  for (const event of timelineEvents) {
    if (!event || typeof event !== "object") {
      continue;
    }
    let eventName = "nejumi.openclaw.event";
    const attrs = {
      "gen_ai.event.content": JSON.stringify(event),
      "nejumi.timeline.index": Number.isFinite(event.timelineIndex)
        ? event.timelineIndex
        : undefined,
      "nejumi.openclaw.timestamp_unix_ms": event.timestamp,
    };
    if (event.type === "user_message") {
      eventName = "gen_ai.user.message";
    } else if (event.type === "assistant_message") {
      eventName = "gen_ai.assistant.message";
    } else if (event.type === "assistant_reasoning") {
      eventName = "gen_ai.assistant.reasoning";
    } else if (event.type === "tool_result") {
      eventName = "gen_ai.tool.result";
      attrs["gen_ai.tool.call.id"] = cleanString(event.toolCallId);
      attrs["gen_ai.tool.name"] = cleanString(event.toolName) || "unknown";
      attrs["nejumi.tool.is_error"] = Boolean(event.isError);
    } else if (event.type === "tool_call") {
      eventName = "gen_ai.tool.call";
      attrs["gen_ai.tool.call.id"] = cleanString(event.toolCallId);
      attrs["gen_ai.tool.name"] = cleanString(event.toolName) || "unknown";
    }
    spanLike.addEvent(eventName, attrs);
  }
  return timelineEvents.length > 0;
}

function recordToolSpans(turn, toolEvents) {
  if (!turn || typeof turn.startTool !== "function" || !Array.isArray(toolEvents)) {
    return;
  }
  const pending = new Map();
  for (const event of toolEvents) {
    if (!event || typeof event !== "object") {
      continue;
    }
    const callId = cleanString(event.toolCallId) || `tool-${event.index ?? pending.size}`;
    if (event.type === "tool_call") {
      const tool = turn.startTool({
        name: cleanString(event.toolName) || "unknown",
        toolCallId: callId,
        args: toolValueToString(event.arguments),
      });
      if (tool.span && typeof tool.span.setAttribute === "function") {
        setAttrs(tool.span, {
          "nejumi.tool.event.index": event.index,
          "nejumi.tool.timestamp_unix_ms": event.timestamp,
        });
      }
      pending.set(callId, tool);
      continue;
    }
    if (event.type === "tool_result") {
      let tool = pending.get(callId);
      if (!tool) {
        tool = turn.startTool({
          name: cleanString(event.toolName) || "unknown",
          toolCallId: callId,
          args: "",
        });
      }
      tool.result = toolResultText(event);
      const error = event.isError ? new Error("tool.execution.error") : undefined;
      tool.end(error ? {error} : undefined);
      pending.delete(callId);
    }
  }
  for (const tool of pending.values()) {
    tool.end();
  }
}

async function main() {
  const payload = readPayload();
  const weave = await import(pathToFileURL(weaveModulePath()).href);
  const project = `${payload.entity}/${payload.project}`;
  const agentName = cleanString(payload.agentName) || "nejumi-taiwan-sidecar-diagnostic";
  const model = cleanString(payload.model) || "unknown";
  const providerName = cleanString(payload.providerName) || "openclaw";
  const conversationId =
    cleanString(payload.conversationId) ||
    `${cleanString(payload.benchmarkId)}:${cleanString(payload.taskId)}`;

  await weave.init(project, {
    genai: weaveGenaiSettings(),
  });

  const inputMessages = [{role: "user", content: cleanString(payload.prompt)}];
  const reasoning = cleanString(payload.reasoning);
  const assistantParts = [];
  if (reasoning) {
    assistantParts.push({type: "reasoning", content: reasoning});
  }
  assistantParts.push({type: "text", content: cleanString(payload.assistant)});
  const outputMessages = [
    {
      role: "assistant",
      parts: assistantParts,
      finish_reason: cleanString(payload.stopReason) || undefined,
    },
  ];
  const messageAttrs = {
    "gen_ai.input.messages": JSON.stringify(inputMessages),
    "gen_ai.output.messages": JSON.stringify(outputMessages),
  };
  const usageAttrs = genAiUsageAttrs(payload.usage);
  const toolEvents = Array.isArray(payload.toolEvents) ? payload.toolEvents : [];
  const timelineEvents = Array.isArray(payload.timelineEvents) ? payload.timelineEvents : [];
  const toolPolicyViolations = Array.isArray(payload.toolPolicyViolations)
    ? payload.toolPolicyViolations
    : [];

  const session = weave.startSession({agentName, model, sessionId: conversationId});
  const turn = session.startTurn({agentName, model});
  setAttrs(turn, {
    ...messageAttrs,
    ...usageAttrs,
    "nejumi.protocol.version": payload.protocolVersion,
    "nejumi.benchmark_id": payload.benchmarkId,
    "nejumi.task_id": payload.taskId,
    "nejumi.prompt_hash": payload.promptHash,
    "nejumi.tool_policy_hash": payload.toolPolicyHash,
    "nejumi.tool_policy_ok": payload.toolPolicyOk,
    "nejumi.tool_policy_violations": JSON.stringify(toolPolicyViolations),
    "nejumi.verifier_hash": payload.verifierHash,
    "nejumi.agent_runtime": "openclaw",
    "nejumi.trace_source": "openclaw_result_sidecar",
    "nejumi.native_plugin_expected": payload.nativePluginExpected,
    "nejumi.openclaw_result_path": payload.openclawResultPath,
    "nejumi.openclaw_config_path": payload.openclawConfigPath,
    "nejumi.openclaw_session_file": payload.sessionFile,
    "nejumi.cwd": payload.cwd,
    "nejumi.started_at_unix": payload.startedAt,
    "nejumi.ended_at_unix": payload.endedAt,
    "nejumi.duration_ms": payload.durationMs,
    "nejumi.returncode": payload.returncode,
    "nejumi.stop_reason": payload.stopReason,
    "nejumi.thinking": payload.thinking,
    "nejumi.tool_call_count": payload.toolCallCount,
    "nejumi.tool_error_count": payload.toolErrorCount,
    "nejumi.timeline_event_count": timelineEvents.length,
    "weave.agent.version": payload.agentVersion,
    "weave.agent.description": payload.agentDescription,
  });

  const llm = turn.startLLM({model, providerName});
  if (llm.span && typeof llm.span.setAttribute === "function") {
    if (!addTimelineEvents(llm.span, timelineEvents)) {
      addMessageEvents(llm.span, inputMessages, outputMessages);
      addReasoningEvent(llm.span, reasoning);
      addToolEvents(llm.span, toolEvents);
    }
    setAttrs(llm.span, {
      ...messageAttrs,
      ...usageAttrs,
      "gen_ai.response.model": model,
      "gen_ai.response.finish_reasons": payload.stopReason
        ? JSON.stringify([payload.stopReason])
        : undefined,
    });
  }
  llm.record({
    inputMessages,
    outputMessages,
    usage: usage(payload.usage),
  });
  recordToolSpans(turn, toolEvents);

  const error =
    payload.toolPolicyOk === false
      ? new Error("OpenClaw tool policy violation")
      : payload.returncode && payload.returncode !== 0
        ? new Error(`OpenClaw returned ${payload.returncode}`)
        : undefined;
  llm.end(error ? {error} : undefined);
  turn.end(error ? {error} : undefined);
  session.end();
  await weave.flushOTel();

  console.log(
    JSON.stringify({
      ok: true,
      project,
      agentName,
      model,
      providerName,
      conversationId,
      benchmarkId: payload.benchmarkId,
      taskId: payload.taskId,
    })
  );
}

main().catch((error) => {
  console.error(error && error.stack ? error.stack : String(error));
  process.exitCode = 1;
});
