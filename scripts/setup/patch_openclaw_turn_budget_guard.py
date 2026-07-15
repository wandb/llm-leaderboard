#!/usr/bin/env python3
"""Patch installed OpenClaw to enforce Nejumi agentic runtime budgets.

OpenClaw 2026.6.x exposes plugin hooks for tool calls, but not a blocking
per-model-call hook on the Gateway task-agent path. The benchmark budget guard
therefore needs a small runtime patch:

* wrap the embedded agent stream function and throw before provider dispatch
  once the configured `nejumi-budget-guard` maxAgentTurns limit is reached;
* expose OpenClaw's actual model-call token usage from the diagnostic wrapper
  and throw immediately after a completed call if the configured cumulative
  provider-token budget is exceeded or usage is missing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


PATCH_MARKER = "Nejumi patch v4: enforce budget guard turns and cumulative actual tokens"
DIAGNOSTIC_PATCH_MARKER = "Nejumi patch v4: expose model call usage to budget guard"
EXEC_TIMEOUT_PATCH_MARKER = "Nejumi patch v1: clamp exec tool timeout to configured max"
EXTENDED_THINKING_PATCH_MARKER = (
    "Nejumi patch v2: pass through explicit thinking levels to provider validation"
)
OLD_EXTENDED_THINKING_PATCH_MARKERS = (
    "Nejumi patch v1: permit xhigh/max thinking levels through provider validation",
)
OLD_PATCH_MARKERS = (
    "Nejumi patch v3: enforce budget guard turns and cumulative actual tokens",
    "Nejumi patch v2: enforce budget guard maxAgentTurns before model stream dispatch",
    "Nejumi patch: enforce budget guard maxAgentTurns before model stream dispatch",
)
OLD_DIAGNOSTIC_PATCH_MARKERS = (
    "Nejumi patch v3: expose model call usage to budget guard",
)

HELPER_BLOCK = r"""
			// Nejumi patch v4: enforce budget guard turns and cumulative actual tokens.
			const extractNejumiBudgetGuardConfig = (runtimeConfig) => {
				const entry = runtimeConfig?.plugins?.entries?.["nejumi-budget-guard"];
				if (!entry || typeof entry !== "object" || entry.enabled === false) return null;
				const rawConfig = entry.config && typeof entry.config === "object" ? entry.config : entry;
				if (rawConfig.enabled === false) return null;
				return rawConfig;
			};
			const parseNejumiBudgetGuardLimit = (value) => {
				const parsed = typeof value === "number" && Number.isFinite(value) ? value : Number.parseInt(String(value ?? ""), 10);
				if (!Number.isFinite(parsed) || parsed <= 0) return null;
				return Math.max(0, Math.trunc(parsed));
			};
			const parseNejumiBudgetGuardBoolean = (value) => {
				if (value === true) return true;
				if (value === false || value == null) return false;
				const normalized = String(value).trim().toLowerCase();
				return normalized === "1" || normalized === "true" || normalized === "yes";
			};
			const resolveNejumiBudgetGuardConfig = () => {
				let runtimeConfig = null;
				try {
					runtimeConfig = typeof getRuntimeConfig === "function" ? getRuntimeConfig() : null;
				} catch {
					runtimeConfig = null;
				}
				const rawConfig = extractNejumiBudgetGuardConfig(params.config) ?? extractNejumiBudgetGuardConfig(runtimeConfig);
				if (!rawConfig) return null;
				const maxAgentTurns = parseNejumiBudgetGuardLimit(rawConfig.maxAgentTurns);
				const maxCumulativeInputTokens = parseNejumiBudgetGuardLimit(rawConfig.maxCumulativeInputTokens);
				const maxCumulativeOutputTokens = parseNejumiBudgetGuardLimit(rawConfig.maxCumulativeOutputTokens);
				const requireActualTokenUsage = parseNejumiBudgetGuardBoolean(rawConfig.requireActualTokenUsage);
				if (!maxAgentTurns && !maxCumulativeInputTokens && !maxCumulativeOutputTokens && !requireActualTokenUsage) return null;
				const agentIds = Array.isArray(rawConfig.agentIds) ? rawConfig.agentIds.map((value) => String(value)).filter((value) => value.length > 0) : [];
				const sessionKeyPrefixes = Array.isArray(rawConfig.sessionKeyPrefixes) ? rawConfig.sessionKeyPrefixes.map((value) => String(value)).filter((value) => value.length > 0) : [];
				const sessionKey = String(params.sessionKey ?? params.sessionId ?? "");
				const keyAgentMatch = sessionKey.match(/^agent:([^:]+):/);
				const configuredAgentId = String(sessionAgentId ?? params.agentId ?? "");
				const currentAgentId = configuredAgentId || (keyAgentMatch ? keyAgentMatch[1] : "");
				const agentMatches = Boolean(currentAgentId && agentIds.includes(currentAgentId)) || agentIds.some((id) => sessionKey.includes(`agent:${id}:`));
				const sessionKeyLower = sessionKey.toLowerCase();
				const sessionMatches = Boolean(sessionKey) && sessionKeyPrefixes.some((prefix) => {
					const normalizedPrefix = prefix.toLowerCase();
					return sessionKeyLower.startsWith(normalizedPrefix) || sessionKeyLower.includes(`:${normalizedPrefix}:`);
				});
				if (agentIds.length > 0 && currentAgentId && !agentMatches) return null;
				if (sessionKeyPrefixes.length > 0 && sessionKey && !sessionMatches) return null;
				if ((agentIds.length > 0 || sessionKeyPrefixes.length > 0) && !agentMatches && !sessionMatches) return null;
				return {
					maxAgentTurns,
					maxCumulativeInputTokens,
					maxCumulativeOutputTokens,
					requireActualTokenUsage,
					sessionKey,
					agentId: currentAgentId,
					blockReasonPrefix: String(rawConfig.blockReasonPrefix || "NEJUMI_BUDGET_GUARD_BLOCKED")
				};
			};
			const nejumiBudgetGuardTurnConfig = resolveNejumiBudgetGuardConfig();
			const makeNejumiBudgetGuardError = (kind, fields = {}) => {
				const message = [
					nejumiBudgetGuardTurnConfig.blockReasonPrefix,
					kind,
					...Object.entries(fields).map(([key, value]) => `${key}=${value ?? "unknown"}`),
					`agentId=${nejumiBudgetGuardTurnConfig.agentId || "unknown"}`,
					`sessionKey=${nejumiBudgetGuardTurnConfig.sessionKey || "unknown"}`
				].join(" ");
				const error = new Error(message);
				error.name = "NejumiBudgetGuardRuntimeLimitError";
				error.code = `NEJUMI_BUDGET_GUARD_${kind.toUpperCase()}`;
				return error;
			};
			const nejumiFiniteToken = (value) => typeof value === "number" && Number.isFinite(value) ? Math.max(0, Math.trunc(value)) : void 0;
			const nejumiUsageInputTokens = (usage) => {
				if (!usage || typeof usage !== "object") return void 0;
				const input = nejumiFiniteToken(usage.input);
				const cacheRead = nejumiFiniteToken(usage.cacheRead) ?? 0;
				const cacheWrite = nejumiFiniteToken(usage.cacheWrite) ?? 0;
				const output = nejumiFiniteToken(usage.output);
				const total = nejumiFiniteToken(usage.total ?? usage.totalTokens);
				if (input !== void 0) return input + cacheRead + cacheWrite;
				if (total !== void 0 && output !== void 0) return Math.max(0, total - output);
				return void 0;
			};
			const nejumiUsageOutputTokens = (usage) => {
				if (!usage || typeof usage !== "object") return void 0;
				const output = nejumiFiniteToken(usage.output);
				if (output !== void 0) return output;
				const input = nejumiUsageInputTokens(usage);
				const total = nejumiFiniteToken(usage.total ?? usage.totalTokens);
				if (total !== void 0 && input !== void 0) return Math.max(0, total - input);
				return void 0;
			};
			const nejumiUsageHasNonzeroActualTokens = (usage) => {
				if (!usage || typeof usage !== "object") return false;
				return ["input", "output", "cacheRead", "cacheWrite", "total", "totalTokens", "reasoningTokens"].some((key) => {
					const value = nejumiFiniteToken(usage[key]);
					return value !== void 0 && value > 0;
				});
			};
			const nejumiNormalizeUsage = (usage) => {
				try {
					return typeof normalizeUsage === "function" ? normalizeUsage(usage ?? void 0) : usage;
				} catch {
					return usage;
				}
			};
			const nejumiMessageUsage = (message) => {
				if (!message || typeof message !== "object") return null;
				if (message.stopReason === "error") return null;
				const usage = nejumiNormalizeUsage(message.usage ?? message.message?.usage);
				return nejumiUsageHasNonzeroActualTokens(usage) ? usage : null;
			};
			const latestNejumiAssistantUsageFromMessages = () => {
				const messages = Array.isArray(activeSession?.messages) ? activeSession.messages : [];
				for (let index = messages.length - 1; index >= 0; index -= 1) {
					const message = messages[index];
					if (message?.role !== "assistant") continue;
					const usage = nejumiMessageUsage(message);
					if (usage) return usage;
				}
				return null;
			};
			const latestNejumiAssistantUsageFromSessionFile = () => {
				const sessionFile = params.sessionFile;
				if (!sessionFile) return null;
				let text = "";
				try {
					text = readFileSync(sessionFile, "utf8");
				} catch {
					return null;
				}
				const lines = text.split(/\r?\n/);
				for (let index = lines.length - 1; index >= 0; index -= 1) {
					const line = lines[index]?.trim();
					if (!line) continue;
					let entry;
					try {
						entry = JSON.parse(line);
					} catch {
						continue;
					}
					const message = entry?.message;
					if (message?.role !== "assistant") continue;
					const usage = nejumiMessageUsage(message);
					if (usage) return usage;
				}
				return null;
			};
			let nejumiObservedAgentTurns = 0;
			let nejumiCumulativeInputTokens = 0;
			let nejumiCumulativeOutputTokens = 0;
			let nejumiUsageObservationCount = 0;
			let nejumiPendingUsageViolation = null;
			const applyNejumiBudgetGuardModelCallUsage = (usage, callId, observedCalls) => {
				if (!nejumiBudgetGuardTurnConfig) return;
				const hasUsage = nejumiUsageHasNonzeroActualTokens(usage);
				const inputTokens = nejumiUsageInputTokens(usage);
				const outputTokens = nejumiUsageOutputTokens(usage);
				if (nejumiBudgetGuardTurnConfig.requireActualTokenUsage && !hasUsage) {
					nejumiPendingUsageViolation ??= {
						kind: "missing_actual_token_usage",
						fields: { callId, observedCalls }
					};
					return;
				}
				if (nejumiBudgetGuardTurnConfig.maxCumulativeInputTokens && inputTokens === void 0) {
					nejumiPendingUsageViolation ??= {
						kind: "missing_actual_input_tokens",
						fields: { callId, limit: nejumiBudgetGuardTurnConfig.maxCumulativeInputTokens }
					};
					return;
				}
				if (nejumiBudgetGuardTurnConfig.maxCumulativeOutputTokens && outputTokens === void 0) {
					nejumiPendingUsageViolation ??= {
						kind: "missing_actual_output_tokens",
						fields: { callId, limit: nejumiBudgetGuardTurnConfig.maxCumulativeOutputTokens }
					};
					return;
				}
				if (inputTokens !== void 0) nejumiCumulativeInputTokens += inputTokens;
				if (outputTokens !== void 0) nejumiCumulativeOutputTokens += outputTokens;
				if (nejumiBudgetGuardTurnConfig.maxCumulativeInputTokens && nejumiCumulativeInputTokens > nejumiBudgetGuardTurnConfig.maxCumulativeInputTokens) {
					nejumiPendingUsageViolation ??= {
						kind: "cumulative_input_tokens_limit_exceeded",
						fields: {
							callId,
							observed: nejumiCumulativeInputTokens,
							limit: nejumiBudgetGuardTurnConfig.maxCumulativeInputTokens,
							observedCalls
						}
					};
					return;
				}
				if (nejumiBudgetGuardTurnConfig.maxCumulativeOutputTokens && nejumiCumulativeOutputTokens > nejumiBudgetGuardTurnConfig.maxCumulativeOutputTokens) {
					nejumiPendingUsageViolation ??= {
						kind: "cumulative_output_tokens_limit_exceeded",
						fields: {
							callId,
							observed: nejumiCumulativeOutputTokens,
							limit: nejumiBudgetGuardTurnConfig.maxCumulativeOutputTokens,
							observedCalls
						}
					};
				}
			};
			const observeNejumiBudgetGuardModelCallUsage = (event) => {
				if (!nejumiBudgetGuardTurnConfig) return;
				const usage = event?.usage;
				const callId = event?.callId || "unknown";
				nejumiUsageObservationCount += 1;
				applyNejumiBudgetGuardModelCallUsage(usage, callId, nejumiUsageObservationCount);
			};
			const recoverNejumiPendingUsageViolation = () => {
				if (!nejumiPendingUsageViolation) return;
				if (!String(nejumiPendingUsageViolation.kind || "").startsWith("missing_actual_")) return;
				const usage = latestNejumiAssistantUsageFromMessages() ?? latestNejumiAssistantUsageFromSessionFile();
				if (!nejumiUsageHasNonzeroActualTokens(usage)) return;
				const pending = nejumiPendingUsageViolation;
				nejumiPendingUsageViolation = null;
				applyNejumiBudgetGuardModelCallUsage(
					usage,
					pending.fields?.callId || "recovered",
					pending.fields?.observedCalls || nejumiUsageObservationCount || 1
				);
			};
			const applyNejumiBudgetGuardTurnAndUsageCap = (streamFn) => {
				if (!nejumiBudgetGuardTurnConfig) return streamFn;
				return (model, context, options) => {
					if (nejumiPendingUsageViolation) {
						recoverNejumiPendingUsageViolation();
					}
					if (nejumiPendingUsageViolation) {
						throw makeNejumiBudgetGuardError(
							nejumiPendingUsageViolation.kind,
							nejumiPendingUsageViolation.fields
						);
					}
					const nextAgentTurn = nejumiObservedAgentTurns + 1;
					if (nejumiBudgetGuardTurnConfig.maxAgentTurns && nextAgentTurn > nejumiBudgetGuardTurnConfig.maxAgentTurns) {
						throw makeNejumiBudgetGuardError("agent_turn_limit_exceeded", {
							observed: nextAgentTurn,
							limit: nejumiBudgetGuardTurnConfig.maxAgentTurns
						});
					}
					nejumiObservedAgentTurns = nextAgentTurn;
					return streamFn(model, context, options);
				};
			};
""".rstrip()

DIAGNOSTIC_HELPER_BLOCK = r"""
// Nejumi patch v4: expose model call usage to budget guard.
function collectNejumiBudgetGuardUsageCandidates(value, candidates = [], seen = /* @__PURE__ */ new Set(), depth = 0) {
	if (value == null || depth > 5) return candidates;
	if (Array.isArray(value)) {
		for (const item of value) collectNejumiBudgetGuardUsageCandidates(item, candidates, seen, depth + 1);
		return candidates;
	}
	if (typeof value !== "object") return candidates;
	if (seen.has(value)) return candidates;
	seen.add(value);
	for (const key of ["usage", "lastCallUsage", "tokenUsage", "modelUsage"]) {
		if (value[key] !== void 0) candidates.push(value[key]);
	}
	for (const key of [
		"message",
		"partial",
		"delta",
		"response",
		"result",
		"meta",
		"output",
		"data",
		"assistantMessage"
	]) {
		if (value[key] !== void 0) collectNejumiBudgetGuardUsageCandidates(value[key], candidates, seen, depth + 1);
	}
	return candidates;
}
function firstNejumiBudgetGuardNormalizedUsage(...values) {
	for (const value of values) {
		const candidates = collectNejumiBudgetGuardUsageCandidates(value);
		for (const candidate of candidates) {
			const usage = normalizeUsage(candidate ?? void 0);
			if (usage) return usage;
		}
	}
	return null;
}
function observeNejumiBudgetGuardDiagnosticUsageCandidate(state, value) {
	const usage = firstNejumiBudgetGuardNormalizedUsage(value);
	if (usage) state.nejumiBudgetGuardLastUsage = usage;
}
function extractNejumiBudgetGuardDiagnosticUsage(state) {
	const usage = firstNejumiBudgetGuardNormalizedUsage(
		state?.nejumiBudgetGuardLastUsage,
		state?.lastResult,
		state?.outputMessages
	);
	return usage ?? null;
}
function notifyNejumiBudgetGuardModelCallCompleted(eventBase, state) {
	const onCompleted = state?.nejumiBudgetGuardOnCompleted;
	if (typeof onCompleted !== "function") return;
	onCompleted({
		callId: eventBase.callId,
		provider: eventBase.provider,
		model: eventBase.model,
		usage: extractNejumiBudgetGuardDiagnosticUsage(state)
	});
}
""".rstrip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openclaw-package-dir", type=Path, default=None)
    parser.add_argument("--dist-dir", type=Path, default=None)
    parser.add_argument("--openclaw-bin", default="openclaw")
    parser.add_argument("--check", action="store_true", help="Only verify that the patch is present.")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def npm_global_openclaw_dir(env: dict[str, str]) -> Path | None:
    npm = shutil.which("npm", path=env.get("PATH"))
    if not npm:
        return None
    result = subprocess.run(
        [npm, "root", "-g"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    root = result.stdout.strip()
    if result.returncode != 0 or not root:
        return None
    return Path(root) / "openclaw"


def package_dir_from_binary(openclaw_bin: str, env: dict[str, str]) -> Path | None:
    binary = shutil.which(openclaw_bin, path=env.get("PATH"))
    if not binary:
        return None
    path = Path(binary).resolve()
    for parent in [path, *path.parents]:
        package_json = parent / "package.json"
        if not package_json.exists():
            continue
        try:
            payload = json.loads(package_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("name") == "openclaw":
            return parent
    for parent in path.parents:
        candidate = parent / "lib" / "node_modules" / "openclaw"
        if (candidate / "package.json").exists():
            return candidate
    return None


def resolve_dist_dir(args: argparse.Namespace, env: dict[str, str]) -> tuple[Path | None, Path]:
    if args.dist_dir is not None:
        dist_dir = args.dist_dir.expanduser().resolve()
        return dist_dir.parent, dist_dir

    candidates: list[Path] = []
    if args.openclaw_package_dir is not None:
        candidates.append(args.openclaw_package_dir)
    if env.get("OPENCLAW_PACKAGE_DIR"):
        candidates.append(Path(env["OPENCLAW_PACKAGE_DIR"]))
    npm_candidate = npm_global_openclaw_dir(env)
    if npm_candidate is not None:
        candidates.append(npm_candidate)
    binary_candidate = package_dir_from_binary(args.openclaw_bin, env)
    if binary_candidate is not None:
        candidates.append(binary_candidate)

    seen: set[Path] = set()
    for candidate in candidates:
        package_dir = candidate.expanduser().resolve()
        if package_dir in seen:
            continue
        seen.add(package_dir)
        dist_dir = package_dir / "dist"
        if dist_dir.is_dir():
            return package_dir, dist_dir

    raise SystemExit("Could not locate OpenClaw dist directory.")


def candidate_files(dist_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in dist_dir.glob("*.js"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if (
            has_any_selection_patch_marker(text)
            or has_diagnostic_patch_marker(text)
            or EXTENDED_THINKING_PATCH_MARKER in text
            or any(marker in text for marker in OLD_EXTENDED_THINKING_PATCH_MARKERS)
            or (
                "let diagnosticModelCallSeq = 0;" in text
                and "wrapStreamFnWithDiagnosticModelCallEvents(activeSession.agent.streamFn" in text
            )
            or (
                "function emitModelCallCompleted(eventBase, startedAt, state)" in text
                and "function wrapStreamFnWithDiagnosticModelCallEvents(streamFn, ctx)" in text
            )
            or (
                "function createExecTool(defaults)" in text
                and "const defaultTimeoutSec = typeof defaults?.timeoutSec" in text
                and (
                    "const effectiveTimeout = (typeof params.timeout === \"number\" ? params.timeout : null) ?? defaultTimeoutSec;"
                    in text
                    or EXEC_TIMEOUT_PATCH_MARKER in text
                    or "const effectiveTimeout = nejumiEffectiveRequestedTimeoutSec ?? defaultTimeoutSec;"
                    in text
                )
            )
            or (
                "function resolveFoundryReasoningEfforts(value)" in text
                and "function buildFoundryThinkingLevelMap(efforts)" in text
            )
            or (
                "const GPT_52_REASONING_EFFORTS = [" in text
                and "function resolveOpenAIReasoningEffortForModel(params)" in text
            )
            or (
                "function resolveThinkingProfile(params)" in text
                and "function appendProfileLevel(profile, id)" in text
            )
            or (
                "function getSupportedThinkingLevels(model)" in text
                and "const EXTENDED_THINKING_LEVELS = [" in text
            )
        ):
            files.append(path)
    return files


def has_any_selection_patch_marker(text: str) -> bool:
    return PATCH_MARKER in text or any(marker in text for marker in OLD_PATCH_MARKERS)


def has_diagnostic_patch_marker(text: str) -> bool:
    return DIAGNOSTIC_PATCH_MARKER in text or any(
        marker in text for marker in OLD_DIAGNOSTIC_PATCH_MARKERS
    )


def replace_existing_helper(text: str) -> tuple[str, bool]:
    positions = [
        text.find(marker)
        for marker in (PATCH_MARKER, *OLD_PATCH_MARKERS)
        if text.find(marker) >= 0
    ]
    if not positions:
        return text, False
    marker_index = min(positions)
    line_start = text.rfind("\n", 0, marker_index)
    if line_start < 0:
        raise ValueError("existing patch marker is not on a normal line")
    helper_start = line_start + 1
    wrapper_anchor = "activeSession.agent.streamFn = wrapStreamFnWithDiagnosticModelCallEvents(activeSession.agent.streamFn"
    helper_end = text.find(wrapper_anchor, marker_index)
    if helper_end < 0:
        raise ValueError("existing patch marker is not followed by the diagnostic wrapper anchor")
    return text[:helper_start] + HELPER_BLOCK + "\n" + text[helper_end:], True


def replace_existing_diagnostic_helper(text: str) -> tuple[str, bool]:
    positions = [
        text.find(marker)
        for marker in (DIAGNOSTIC_PATCH_MARKER, *OLD_DIAGNOSTIC_PATCH_MARKERS)
        if text.find(marker) >= 0
    ]
    if not positions:
        return text, False
    marker_index = min(positions)
    line_start = text.rfind("\n", 0, marker_index)
    if line_start < 0:
        raise ValueError("existing diagnostic patch marker is not on a normal line")
    helper_start = line_start + 1
    helper_end = text.find(
        "function emitModelCallCompleted(eventBase, startedAt, state) {",
        marker_index,
    )
    if helper_end < 0:
        raise ValueError("existing diagnostic patch marker is not followed by emitModelCallCompleted")
    return text[:helper_start] + DIAGNOSTIC_HELPER_BLOCK + "\n" + text[helper_end:], True


def add_selection_on_completed_callback(text: str) -> tuple[str, bool]:
    """Wire the diagnostic wrapper's per-call completion event into the guard."""
    if "observeNejumiBudgetGuardModelCallUsage(event)" in text:
        old_call = "applyNejumiBudgetGuardTurnCap(activeSession.agent.streamFn)"
        changed = old_call in text
        text = text.replace(
            old_call,
            "applyNejumiBudgetGuardTurnAndUsageCap(activeSession.agent.streamFn)",
        )
        return text, changed

    wrapper_anchor = "activeSession.agent.streamFn = wrapStreamFnWithDiagnosticModelCallEvents(activeSession.agent.streamFn"
    wrapper_start = text.find(wrapper_anchor)
    if wrapper_start < 0:
        raise ValueError("target bundle does not contain the expected diagnostic wrapper anchor")
    guard_call = "activeSession.agent.streamFn = applyNejumiBudgetGuardTurnCap(activeSession.agent.streamFn);"
    guard_index = text.find(guard_call, wrapper_start)
    replacement_guard_call = (
        "activeSession.agent.streamFn = applyNejumiBudgetGuardTurnAndUsageCap(activeSession.agent.streamFn);"
    )
    if guard_index < 0:
        guard_call = replacement_guard_call
        guard_index = text.find(guard_call, wrapper_start)
        if guard_index < 0:
            raise ValueError("target bundle does not contain the expected budget guard call")

    call_text = text[wrapper_start:guard_index]
    pattern = re.compile(r"(\n\s*onStarted:\s*\(\)\s*=>\s*\{.*?\})(\n\s*\}\s*\);\n\s*)", re.DOTALL)
    match = pattern.search(call_text)
    if not match:
        raise ValueError("target bundle does not contain the expected onStarted callback")
    replacement = (
        match.group(1)
        + ",\n				onCompleted: (event) => {\n"
        + "					observeNejumiBudgetGuardModelCallUsage(event);\n"
        + "				}"
        + match.group(2)
    )
    patched_call_text = call_text[: match.start()] + replacement + call_text[match.end() :]
    text = text[:wrapper_start] + patched_call_text + text[guard_index:]
    text = text.replace(guard_call, replacement_guard_call, 1)
    return text, True


def patch_selection_text(text: str) -> tuple[str, bool]:
    changed = False
    selection_helper_current = (
        PATCH_MARKER in text
        and "latestNejumiAssistantUsageFromMessages" in text
        and "latestNejumiAssistantUsageFromSessionFile" in text
        and "recoverNejumiPendingUsageViolation();" in text
    )
    if not selection_helper_current:
        if has_any_selection_patch_marker(text):
            text, helper_changed = replace_existing_helper(text)
            changed = changed or helper_changed
        else:
            helper_anchor = "			let diagnosticModelCallSeq = 0;"
            wrapper_anchor = "activeSession.agent.streamFn = wrapStreamFnWithDiagnosticModelCallEvents(activeSession.agent.streamFn"
            if helper_anchor not in text or wrapper_anchor not in text:
                raise ValueError("target bundle does not contain the expected stream wrapper anchors")

            text = text.replace(helper_anchor, helper_anchor + "\n" + HELPER_BLOCK, 1)
            wrapper_start = text.index(wrapper_anchor)
            tail_needle = "\n			});\n			try {\n				if (isRawModelRun) {"
            tail_index = text.find(tail_needle, wrapper_start)
            if tail_index < 0:
                raise ValueError("target bundle does not contain the expected diagnostic wrapper tail")
            replacement = (
                "\n			});\n"
                "			activeSession.agent.streamFn = applyNejumiBudgetGuardTurnAndUsageCap(activeSession.agent.streamFn);\n"
                "			try {\n"
                "				if (isRawModelRun) {"
            )
            text = text[:tail_index] + replacement + text[tail_index + len(tail_needle) :]
            changed = True
    text, callback_changed = add_selection_on_completed_callback(text)
    changed = changed or callback_changed
    return text, changed


def patch_diagnostic_text(text: str) -> tuple[str, bool]:
    changed = False
    if DIAGNOSTIC_PATCH_MARKER not in text:
        if has_diagnostic_patch_marker(text):
            text, helper_changed = replace_existing_diagnostic_helper(text)
            changed = changed or helper_changed
        else:
            helper_anchor = "function emitModelCallCompleted(eventBase, startedAt, state) {"
            if helper_anchor not in text:
                raise ValueError("diagnostic bundle does not contain emitModelCallCompleted")
            text = text.replace(helper_anchor, DIAGNOSTIC_HELPER_BLOCK + "\n" + helper_anchor, 1)
            changed = True

    if "observeNejumiBudgetGuardDiagnosticUsageCandidate(state, chunk);" not in text:
        chunk_anchor = "observeOutputMessageContent(state, chunk);\n\tconst bytes = responseStreamChunkByteLength(chunk);"
        chunk_replacement = (
            "observeOutputMessageContent(state, chunk);\n"
            "\tobserveNejumiBudgetGuardDiagnosticUsageCandidate(state, chunk);\n"
            "\tconst bytes = responseStreamChunkByteLength(chunk);"
        )
        if chunk_anchor not in text:
            raise ValueError("diagnostic bundle does not contain the expected response chunk observer")
        text = text.replace(chunk_anchor, chunk_replacement, 1)
        changed = True

    if "observeNejumiBudgetGuardDiagnosticUsageCandidate(state, result);" not in text:
        result_anchor = "state.timeToFirstByteMs ??= Math.max(0, Date.now() - startedAt);\n\tif (state.contentCapture?.outputMessages"
        result_replacement = (
            "state.timeToFirstByteMs ??= Math.max(0, Date.now() - startedAt);\n"
            "\tobserveNejumiBudgetGuardDiagnosticUsageCandidate(state, result);\n"
            "\tif (state.contentCapture?.outputMessages"
        )
        if result_anchor not in text:
            raise ValueError("diagnostic bundle does not contain the expected result observer")
        text = text.replace(result_anchor, result_replacement, 1)
        changed = True

    if "nejumiBudgetGuardOnCompleted: ctx.onCompleted" not in text:
        state_anchor = (
            "const state = {\n"
            "\t\t\tresponseStreamBytes: 0,\n"
            "\t\t\tmodelContent,\n"
            "\t\t\tcontentCapture: ctx.contentCapture\n"
            "\t\t};"
        )
        state_replacement = (
            "const state = {\n"
            "\t\t\tresponseStreamBytes: 0,\n"
            "\t\t\tmodelContent,\n"
            "\t\t\tcontentCapture: ctx.contentCapture,\n"
            "\t\t\tnejumiBudgetGuardOnCompleted: ctx.onCompleted\n"
            "\t\t};"
        )
        if state_anchor not in text:
            raise ValueError("diagnostic bundle does not contain the expected state initializer")
        text = text.replace(state_anchor, state_replacement, 1)
        changed = True

    if "notifyNejumiBudgetGuardModelCallCompleted(eventBase, state);" not in text:
        hook_anchor = (
            "dispatchModelCallEndedHook(eventBase, {\n"
            "\t\tdurationMs,\n"
            "\t\toutcome: \"completed\",\n"
            "\t\t...sizeTimingFields\n"
            "\t});"
        )
        hook_replacement = hook_anchor + "\n\tnotifyNejumiBudgetGuardModelCallCompleted(eventBase, state);"
        if hook_anchor not in text:
            raise ValueError("diagnostic bundle does not contain the expected completed hook dispatch")
        text = text.replace(hook_anchor, hook_replacement, 1)
        changed = True
    return text, changed


def patch_exec_timeout_text(text: str) -> tuple[str, bool]:
    """Clamp model-supplied exec timeout values to tools.exec.timeoutSec.

    OpenClaw treats tools.exec.timeoutSec as the default timeout, while a model
    can still pass a larger `timeout` argument on an individual exec call. For
    benchmark runs this must be a hard cap, otherwise one pathological tool call
    can block the whole evaluation despite token/turn/tool-count budgets.
    """
    if EXEC_TIMEOUT_PATCH_MARKER in text:
        return text, False
    if "function createExecTool(defaults)" not in text:
        return text, False
    if "const defaultTimeoutSec = typeof defaults?.timeoutSec" not in text:
        return text, False

    params_anchor = "if (!params.command) throw new Error(\"Provide a command to start.\");"
    timeout_helper = (
        params_anchor
        + "\n"
        + "\t\t\t// "
        + EXEC_TIMEOUT_PATCH_MARKER
        + ".\n"
        + "\t\t\tconst nejumiEffectiveRequestedTimeoutSec = (() => {\n"
        + "\t\t\t\tconst requestedTimeoutSec = typeof params.timeout === \"number\" && Number.isFinite(params.timeout) && params.timeout > 0 ? params.timeout : void 0;\n"
        + "\t\t\t\treturn requestedTimeoutSec === void 0 ? void 0 : Math.min(requestedTimeoutSec, defaultTimeoutSec);\n"
        + "\t\t\t})();"
    )
    if params_anchor not in text:
        raise ValueError("exec bundle does not contain the expected command-validation anchor")
    text = text.replace(params_anchor, timeout_helper, 1)

    old_effective = "const effectiveTimeout = (typeof params.timeout === \"number\" ? params.timeout : null) ?? defaultTimeoutSec;"
    new_effective = "const effectiveTimeout = nejumiEffectiveRequestedTimeoutSec ?? defaultTimeoutSec;"
    if old_effective not in text:
        raise ValueError("exec bundle does not contain the expected effective timeout expression")
    text = text.replace(old_effective, new_effective, 1)

    replaced_timeout_params = text.count("timeoutSec: params.timeout")
    if replaced_timeout_params <= 0:
        raise ValueError("exec bundle does not contain model-supplied timeout passthroughs")
    text = text.replace("timeoutSec: params.timeout", "timeoutSec: nejumiEffectiveRequestedTimeoutSec")
    return text, True


def has_extended_thinking_support(text: str) -> bool:
    """Return whether this bundle already passes explicit thinking levels through.

    This intentionally accepts either the Nejumi runtime patch or an upstream
    OpenClaw implementation with equivalent capability. Future OpenClaw builds
    should not be forced through a local patch solely because the marker is
    absent.
    """
    if EXTENDED_THINKING_PATCH_MARKER in text:
        return True
    if "function resolveFoundryReasoningEfforts(value)" in text:
        return (
            '"xhigh"' in text
            and '"max"' in text
            and (
                'max: supported.has("max") ? "max" : null' in text
                or 'max: "max"' in text
            )
        )
    if "const GPT_52_REASONING_EFFORTS = [" in text:
        return (
            "const GENERIC_REASONING_EFFORTS = [" in text
            and '"xhigh"' in text
            and '"max"' in text
        )
    if (
        "function resolveThinkingProfile(params)" in text
        and "function appendProfileLevel(profile, id)" in text
    ):
        return (
            "function applyNejumiThinkingProfilePassthrough(profile)" in text
            and 'appendProfileLevel(profile, "high");' in text
            and 'appendProfileLevel(profile, "xhigh");' in text
            and 'appendProfileLevel(profile, "max");' in text
        )
    if "function getSupportedThinkingLevels(model)" in text:
        return 'if (level === "xhigh" || level === "max") return mapped !== null;' in text
    return False


def patch_extended_thinking_text(text: str) -> tuple[str, bool]:
    """Keep OpenClaw from rejecting new high-effort model levels locally.

    The benchmark should send an explicit xhigh/max request to the provider and
    let the provider/API be the source of truth. A stale OpenClaw allowlist
    should not fail before the model is called.
    """
    changed = False

    if has_extended_thinking_support(text):
        return text, False

    if "function resolveFoundryReasoningEfforts(value)" in text:
        old_gpt52_block = """if (/^gpt-5\\.[2-9](?:\\.|-|$)/u.test(normalized)) return [
\t\t\"none\",
\t\t\"low\",
\t\t\"medium\",
\t\t\"high\"
\t];"""
        new_gpt52_block = """if (/^gpt-5\\.[2-9](?:\\.|-|$)/u.test(normalized)) return [
\t\t\"none\",
\t\t\"low\",
\t\t\"medium\",
\t\t\"high\",
\t\t\"xhigh\",
\t\t\"max\"
\t];"""
        if old_gpt52_block in text:
            text = text.replace(old_gpt52_block, new_gpt52_block, 1)
            changed = True

        old_map = """\t\thigh: supported.has(\"high\") ? \"high\" : null,
\t\txhigh: supported.has(\"xhigh\") ? \"xhigh\" : null,
\t\tmax: null"""
        new_map = """\t\thigh: supported.has(\"high\") ? \"high\" : null,
\t\txhigh: supported.has(\"xhigh\") ? \"xhigh\" : null,
\t\tmax: supported.has(\"max\") ? \"max\" : null"""
        if old_map in text:
            text = text.replace(old_map, new_map, 1)
            changed = True

    if "const GPT_52_REASONING_EFFORTS = [" in text:
        old_gpt52_efforts = """const GPT_52_REASONING_EFFORTS = [
\t\"none\",
\t\"low\",
\t\"medium\",
\t\"high\",
\t\"xhigh\"
];"""
        new_gpt52_efforts = """const GPT_52_REASONING_EFFORTS = [
\t\"none\",
\t\"low\",
\t\"medium\",
\t\"high\",
\t\"xhigh\",
\t\"max\"
];"""
        if old_gpt52_efforts in text:
            text = text.replace(old_gpt52_efforts, new_gpt52_efforts, 1)
            changed = True

        old_generic_efforts = """const GENERIC_REASONING_EFFORTS = [
\t\"low\",
\t\"medium\",
\t\"high\"
];"""
        new_generic_efforts = """const GENERIC_REASONING_EFFORTS = [
\t\"low\",
\t\"medium\",
\t\"high\",
\t\"xhigh\",
\t\"max\"
];"""
        if old_generic_efforts in text:
            text = text.replace(old_generic_efforts, new_generic_efforts, 1)
            changed = True

    if (
        "function resolveThinkingProfile(params)" in text
        and "function appendProfileLevel(profile, id)" in text
    ):
        helper_anchor = "function appendProfileLevel(profile, id) {"
        helper_end_anchor = "\n/** Resolve supported thinking levels and default for a provider/model pair. */"
        helper_block = """function applyNejumiThinkingProfilePassthrough(profile) {
\tif (!profile || !Array.isArray(profile.levels)) return profile;
\tappendProfileLevel(profile, \"minimal\");
\tappendProfileLevel(profile, \"low\");
\tappendProfileLevel(profile, \"medium\");
\tappendProfileLevel(profile, \"high\");
\tappendProfileLevel(profile, \"xhigh\");
\tappendProfileLevel(profile, \"max\");
\treturn profile;
}
"""
        text, old_helper_replacements = re.subn(
            r"\n?function applyNejumiPermissiveExtendedThinkingProfile\(profile\) \{.*?\n\}\n",
            "\n" + helper_block,
            text,
            count=1,
            flags=re.DOTALL,
        )
        if old_helper_replacements:
            changed = True
        if "function applyNejumiThinkingProfilePassthrough(profile)" not in text:
            anchor_index = text.find(helper_end_anchor)
            if helper_anchor in text and anchor_index >= 0:
                text = text[:anchor_index] + "\n" + helper_block + text[anchor_index:]
                changed = True

        if "applyNejumiPermissiveExtendedThinkingProfile" in text:
            text = text.replace(
                "applyNejumiPermissiveExtendedThinkingProfile",
                "applyNejumiThinkingProfilePassthrough",
            )
            changed = True

        old_plugin_return = (
            "if (normalized.levels.length > 0 && (context.reasoning !== false || "
            "pluginProfile.preserveWhenCatalogReasoningFalse === true)) return normalized;"
        )
        new_plugin_return = (
            "if (normalized.levels.length > 0 && (context.reasoning !== false || "
            "pluginProfile.preserveWhenCatalogReasoningFalse === true)) return "
            "applyNejumiThinkingProfilePassthrough(normalized);"
        )
        if old_plugin_return in text:
            text = text.replace(old_plugin_return, new_plugin_return, 1)
            changed = True

        old_fallback_return = "\treturn profile;\n}\nfunction supportsThinkingLevel"
        new_fallback_return = (
            "\treturn applyNejumiThinkingProfilePassthrough(profile);\n"
            "}\nfunction supportsThinkingLevel"
        )
        if old_fallback_return in text:
            text = text.replace(old_fallback_return, new_fallback_return, 1)
            changed = True

    if "function getSupportedThinkingLevels(model)" in text:
        old_extended_filter = """\t\tif (level === \"xhigh\" || level === \"max\") return mapped !== void 0;
\t\treturn true;"""
        new_extended_filter = """\t\tif (level === \"xhigh\" || level === \"max\") return mapped !== null;
\t\treturn true;"""
        if old_extended_filter in text:
            text = text.replace(old_extended_filter, new_extended_filter, 1)
            changed = True

    if changed:
        marker_anchor = "//#region"
        marker = f"// {EXTENDED_THINKING_PATCH_MARKER}.\n"
        if marker_anchor in text:
            text = text.replace(marker_anchor, marker + marker_anchor, 1)
        else:
            text = marker + text
    return text, changed


def patch_text(text: str) -> tuple[str, bool]:
    changed = False
    text, extended_thinking_changed = patch_extended_thinking_text(text)
    changed = changed or extended_thinking_changed
    selection_needs_patch = (
        (
            "let diagnosticModelCallSeq = 0;" in text
            and "wrapStreamFnWithDiagnosticModelCallEvents(activeSession.agent.streamFn" in text
        )
        or has_any_selection_patch_marker(text)
        or (
            PATCH_MARKER in text
            and "latestNejumiAssistantUsageFromMessages" not in text
        )
    )
    if selection_needs_patch:
        text, selection_changed = patch_selection_text(text)
        changed = changed or selection_changed
    if (
        "function emitModelCallCompleted(eventBase, startedAt, state)" in text
        and "function wrapStreamFnWithDiagnosticModelCallEvents(streamFn, ctx)" in text
    ) or has_diagnostic_patch_marker(text):
        text, diagnostic_changed = patch_diagnostic_text(text)
        changed = changed or diagnostic_changed
    if (
        "function createExecTool(defaults)" in text
        and "const defaultTimeoutSec = typeof defaults?.timeoutSec" in text
    ) or EXEC_TIMEOUT_PATCH_MARKER in text:
        text, exec_timeout_changed = patch_exec_timeout_text(text)
        changed = changed or exec_timeout_changed
    if changed:
        return text, True
    if (
        PATCH_MARKER in text
        or DIAGNOSTIC_PATCH_MARKER in text
        or EXEC_TIMEOUT_PATCH_MARKER in text
        or has_extended_thinking_support(text)
    ):
        return text, False
    raise ValueError("target bundle does not contain supported OpenClaw patch anchors")


def patch_text_legacy(text: str) -> tuple[str, bool]:
    if PATCH_MARKER in text:
        return text, False
    if has_any_selection_patch_marker(text):
        return replace_existing_helper(text)
    helper_anchor = "			let diagnosticModelCallSeq = 0;"
    wrapper_anchor = "activeSession.agent.streamFn = wrapStreamFnWithDiagnosticModelCallEvents(activeSession.agent.streamFn"
    if helper_anchor not in text or wrapper_anchor not in text:
        raise ValueError("target bundle does not contain the expected stream wrapper anchors")

    text = text.replace(helper_anchor, helper_anchor + "\n" + HELPER_BLOCK, 1)
    wrapper_start = text.index(wrapper_anchor)
    tail_needle = "\n			});\n			try {\n				if (isRawModelRun) {"
    tail_index = text.find(tail_needle, wrapper_start)
    if tail_index < 0:
        raise ValueError("target bundle does not contain the expected diagnostic wrapper tail")
    replacement = (
        "\n			});\n"
        "			activeSession.agent.streamFn = applyNejumiBudgetGuardTurnAndUsageCap(activeSession.agent.streamFn);\n"
        "			try {\n"
        "				if (isRawModelRun) {"
    )
    text = text[:tail_index] + replacement + text[tail_index + len(tail_needle) :]
    return text, True


def main() -> None:
    args = parse_args()
    env = dict(os.environ)
    package_dir, dist_dir = resolve_dist_dir(args, env)
    files = candidate_files(dist_dir)
    if not files:
        raise SystemExit(f"No OpenClaw bundle candidate found under {dist_dir}")

    patched: list[str] = []
    already: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        if args.check:
            already.append(str(path))
            continue
        try:
            new_text, changed = patch_text(text)
        except ValueError:
            continue
        if changed:
            path.write_text(new_text, encoding="utf-8")
            patched.append(str(path))
        else:
            already.append(str(path))

    patched_files = patched or already
    verified_selection = []
    verified_diagnostic = []
    verified_exec_timeout = []
    verified_extended_thinking = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        if (
            PATCH_MARKER in text
            and "applyNejumiBudgetGuardTurnAndUsageCap(activeSession.agent.streamFn)" in text
            and "observeNejumiBudgetGuardModelCallUsage(event)" in text
        ):
            verified_selection.append(str(path))
        if (
            DIAGNOSTIC_PATCH_MARKER in text
            and "observeNejumiBudgetGuardDiagnosticUsageCandidate(state, chunk);" in text
            and "observeNejumiBudgetGuardDiagnosticUsageCandidate(state, result);" in text
            and "nejumiBudgetGuardOnCompleted: ctx.onCompleted" in text
            and "notifyNejumiBudgetGuardModelCallCompleted(eventBase, state);" in text
        ):
            verified_diagnostic.append(str(path))
        if (
            EXEC_TIMEOUT_PATCH_MARKER in text
            and "const nejumiEffectiveRequestedTimeoutSec = (() => {" in text
            and "const effectiveTimeout = nejumiEffectiveRequestedTimeoutSec ?? defaultTimeoutSec;" in text
            and "timeoutSec: nejumiEffectiveRequestedTimeoutSec" in text
        ):
            verified_exec_timeout.append(str(path))
        if has_extended_thinking_support(text):
            verified_extended_thinking.append(str(path))
    ok = bool(
        verified_selection
        and verified_diagnostic
        and verified_exec_timeout
        and verified_extended_thinking
    )
    patched_files = sorted(
        set(
            patched_files
            + verified_selection
            + verified_diagnostic
            + verified_exec_timeout
            + verified_extended_thinking
        )
    )
    payload = {
        "ok": ok,
        "package_dir": str(package_dir) if package_dir else None,
        "dist_dir": str(dist_dir),
        "patched": patched,
        "already_patched": already,
        "verified_selection": verified_selection,
        "verified_diagnostic": verified_diagnostic,
        "verified_exec_timeout": verified_exec_timeout,
        "verified_extended_thinking": verified_extended_thinking,
        "check": args.check,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        action = "already patched" if already and not patched else "patched"
        print(f"OpenClaw turn budget guard {action}: {', '.join(patched_files)}")
    if not ok:
        raise SystemExit("OpenClaw turn budget guard patch verification failed")


if __name__ == "__main__":
    main()
