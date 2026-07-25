import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_patch_module():
    path = REPO_ROOT / "scripts" / "setup" / "patch_openclaw_turn_budget_guard.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_nemoclaw_patch_module():
    path = REPO_ROOT / "scripts" / "setup" / "patch_nemoclaw_openclaw_runtime.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_patch_text_wraps_diagnostic_stream_once():
    module = load_patch_module()
    source = (
        "\t\t\tlet diagnosticModelCallSeq = 0;\n"
        "\t\t\tactiveSession.agent.streamFn = wrapStreamFnWithDiagnosticModelCallEvents(activeSession.agent.streamFn, {\n"
        "\t\t\t\tnextCallId: () => `${params.runId}:model:${diagnosticModelCallSeq += 1}`,\n"
        "\t\t\t\tonStarted: () => {}\n"
        "\t\t\t});\n"
        "\t\t\ttry {\n"
        "\t\t\t\tif (isRawModelRun) {\n"
        "\t\t\t\t\tactiveSession.agent.reset();\n"
        "\t\t\t\t}\n"
    )

    patched, changed = module.patch_text(source)
    repatched, changed_again = module.patch_text(patched)

    assert changed is True
    assert changed_again is False
    assert patched == repatched
    assert module.PATCH_MARKER in patched
    assert patched.count("applyNejumiBudgetGuardTurnAndUsageCap(activeSession.agent.streamFn)") == 1
    assert "observeNejumiBudgetGuardModelCallUsage(event)" in patched
    assert "latestNejumiAssistantUsageFromMessages" in patched
    assert "latestNejumiAssistantUsageFromSessionFile" in patched
    assert "allNejumiAssistantUsagesFromSessionFile" in patched
    assert "reconcileNejumiBudgetGuardUsageFromSession();" in patched
    assert 'source: "session_reconciliation"' in patched
    assert "recoverNejumiPendingUsageViolation();" in patched
    assert "cumulative_input_tokens_limit_exceeded" in patched


def diagnostic_bundle_source(*, old_helper: bool = False) -> str:
    helper = ""
    if old_helper:
        helper = (
            "// Nejumi patch v3: expose model call usage to budget guard.\n"
            "function extractNejumiBudgetGuardDiagnosticUsage(state) {\n"
            "\tconst outputMessages = Array.isArray(state?.outputMessages) ? state.outputMessages : [];\n"
            "\tfor (const message of outputMessages) {\n"
            "\t\tconst usage = normalizeUsage(message?.usage ?? void 0);\n"
            "\t\tif (usage) return usage;\n"
            "\t}\n"
            "\treturn null;\n"
            "}\n"
            "function notifyNejumiBudgetGuardModelCallCompleted(eventBase, state) {\n"
            "\tconst onCompleted = state?.nejumiBudgetGuardOnCompleted;\n"
            "\tif (typeof onCompleted !== \"function\") return;\n"
            "\tonCompleted({ callId: eventBase.callId, usage: extractNejumiBudgetGuardDiagnosticUsage(state) });\n"
            "}\n"
        )
    return (
        "function observeResultMessageContent(state, startedAt, result) {\n"
        "\tstate.timeToFirstByteMs ??= Math.max(0, Date.now() - startedAt);\n"
        "\tif (state.contentCapture?.outputMessages && state.outputMessages === void 0) state.outputMessages = [cloneDiagnosticContentValue(result)];\n"
        "\tif (state.responseStreamBytes === 0) {}\n"
        "}\n"
        "function observeResponseChunk(state, startedAt, chunk) {\n"
        "\tstate.timeToFirstByteMs ??= Math.max(0, Date.now() - startedAt);\n"
        "\tobserveOutputMessageContent(state, chunk);\n"
        "\tconst bytes = responseStreamChunkByteLength(chunk);\n"
        "\tif (bytes !== void 0) state.responseStreamBytes += bytes;\n"
        "}\n"
        f"{helper}"
        "function emitModelCallCompleted(eventBase, startedAt, state) {\n"
        "\tconst durationMs = Date.now() - startedAt;\n"
        "\tconst sizeTimingFields = modelCallSizeTimingFields(state);\n"
        "\tdispatchModelCallEndedHook(eventBase, {\n"
        "\t\tdurationMs,\n"
        "\t\toutcome: \"completed\",\n"
        "\t\t...sizeTimingFields\n"
        "\t});\n"
        "}\n"
        "function wrapStreamFnWithDiagnosticModelCallEvents(streamFn, ctx) {\n"
        "\treturn ((model, streamContext, options) => {\n"
        "\t\tconst state = {\n"
        "\t\t\tresponseStreamBytes: 0,\n"
        "\t\t\tmodelContent,\n"
        "\t\t\tcontentCapture: ctx.contentCapture\n"
        "\t\t};\n"
        "\t});\n"
        "}\n"
    )


def test_patch_text_wires_diagnostic_usage_from_stream_chunks_and_results():
    module = load_patch_module()

    patched, changed = module.patch_text(diagnostic_bundle_source())
    repatched, changed_again = module.patch_text(patched)

    assert changed is True
    assert changed_again is False
    assert patched == repatched
    assert module.DIAGNOSTIC_PATCH_MARKER in patched
    assert "collectNejumiBudgetGuardUsageCandidates" in patched
    assert "observeNejumiBudgetGuardDiagnosticUsageCandidate(state, chunk);" in patched
    assert "observeNejumiBudgetGuardDiagnosticUsageCandidate(state, result);" in patched
    assert "nejumiBudgetGuardOnCompleted: ctx.onCompleted" in patched
    assert "notifyNejumiBudgetGuardModelCallCompleted(eventBase, state);" in patched


def test_patch_text_replaces_old_diagnostic_usage_helper():
    module = load_patch_module()

    patched, changed = module.patch_text(diagnostic_bundle_source(old_helper=True))

    assert changed is True
    assert module.DIAGNOSTIC_PATCH_MARKER in patched
    assert "Nejumi patch v3: expose model call usage to budget guard" not in patched
    assert patched.count("function extractNejumiBudgetGuardDiagnosticUsage(state)") == 1
    assert "state?.nejumiBudgetGuardLastUsage" in patched


def exec_bundle_source() -> str:
    return (
        "function createExecTool(defaults) {\n"
        "\tconst defaultTimeoutSec = typeof defaults?.timeoutSec === \"number\" && defaults.timeoutSec > 0 ? defaults.timeoutSec : 1800;\n"
        "\treturn {\n"
        "\t\texecute: async (_toolCallId, args, signal, onUpdate) => {\n"
        "\t\t\tconst params = args;\n"
        "\t\t\tif (!params.command) throw new Error(\"Provide a command to start.\");\n"
        "\t\t\tif (host === \"node\") return executeNodeHostCommand({\n"
        "\t\t\t\ttimeoutSec: params.timeout,\n"
        "\t\t\t\tdefaultTimeoutSec,\n"
        "\t\t\t});\n"
        "\t\t\tconst gatewayResult = await processGatewayAllowlist({\n"
        "\t\t\t\ttimeoutSec: params.timeout,\n"
        "\t\t\t\tdefaultTimeoutSec,\n"
        "\t\t\t});\n"
        "\t\t\tconst effectiveTimeout = (typeof params.timeout === \"number\" ? params.timeout : null) ?? defaultTimeoutSec;\n"
        "\t\t\tawait runExecProcess({ timeoutSec: effectiveTimeout });\n"
        "\t\t}\n"
        "\t};\n"
        "}\n"
    )


def test_patch_text_clamps_exec_tool_timeout_once():
    module = load_patch_module()

    patched, changed = module.patch_text(exec_bundle_source())
    repatched, changed_again = module.patch_text(patched)

    assert changed is True
    assert changed_again is False
    assert patched == repatched
    assert module.EXEC_TIMEOUT_PATCH_MARKER in patched
    assert "const nejumiEffectiveRequestedTimeoutSec = (() => {" in patched
    assert "Math.min(requestedTimeoutSec, defaultTimeoutSec)" in patched
    assert "const effectiveTimeout = nejumiEffectiveRequestedTimeoutSec ?? defaultTimeoutSec;" in patched
    assert "timeoutSec: params.timeout" not in patched
    assert patched.count("timeoutSec: nejumiEffectiveRequestedTimeoutSec") == 2


def test_patch_text_permits_extended_openai_foundry_thinking_levels():
    module = load_patch_module()
    source = (
        "function resolveFoundryReasoningEfforts(value) {\n"
        "\tconst normalized = normalizeFoundryModelName(value);\n"
        "\tif (/^gpt-5\\.[2-9](?:\\.|-|$)/u.test(normalized)) return [\n"
        "\t\t\"none\",\n"
        "\t\t\"low\",\n"
        "\t\t\"medium\",\n"
        "\t\t\"high\"\n"
        "\t];\n"
        "}\n"
        "function buildFoundryThinkingLevelMap(efforts) {\n"
        "\tconst supported = new Set(efforts);\n"
        "\treturn {\n"
        "\t\toff: supported.has(\"none\") ? \"none\" : null,\n"
        "\t\tminimal: supported.has(\"minimal\") ? \"minimal\" : null,\n"
        "\t\tlow: supported.has(\"low\") ? \"low\" : null,\n"
        "\t\tmedium: supported.has(\"medium\") ? \"medium\" : null,\n"
        "\t\thigh: supported.has(\"high\") ? \"high\" : null,\n"
        "\t\txhigh: supported.has(\"xhigh\") ? \"xhigh\" : null,\n"
        "\t\tmax: null\n"
        "\t};\n"
        "}\n"
    )

    patched, changed = module.patch_text(source)
    repatched, changed_again = module.patch_text(patched)

    assert changed is True
    assert changed_again is False
    assert patched == repatched
    assert module.EXTENDED_THINKING_PATCH_MARKER in patched
    assert '"xhigh",' in patched
    assert '"max"' in patched
    assert 'max: supported.has("max") ? "max" : null' in patched


def test_patch_text_accepts_native_extended_openai_foundry_thinking_levels():
    module = load_patch_module()
    source = (
        "function resolveFoundryReasoningEfforts(value) {\n"
        "\tconst normalized = normalizeFoundryModelName(value);\n"
        "\tif (/^gpt-5\\.[2-9](?:\\.|-|$)/u.test(normalized)) return [\n"
        "\t\t\"none\",\n"
        "\t\t\"low\",\n"
        "\t\t\"medium\",\n"
        "\t\t\"high\",\n"
        "\t\t\"xhigh\",\n"
        "\t\t\"max\"\n"
        "\t];\n"
        "}\n"
        "function buildFoundryThinkingLevelMap(efforts) {\n"
        "\tconst supported = new Set(efforts);\n"
        "\treturn {\n"
        "\t\toff: supported.has(\"none\") ? \"none\" : null,\n"
        "\t\tminimal: supported.has(\"minimal\") ? \"minimal\" : null,\n"
        "\t\tlow: supported.has(\"low\") ? \"low\" : null,\n"
        "\t\tmedium: supported.has(\"medium\") ? \"medium\" : null,\n"
        "\t\thigh: supported.has(\"high\") ? \"high\" : null,\n"
        "\t\txhigh: supported.has(\"xhigh\") ? \"xhigh\" : null,\n"
        "\t\tmax: supported.has(\"max\") ? \"max\" : null\n"
        "\t};\n"
        "}\n"
    )

    patched, changed = module.patch_text(source)

    assert changed is False
    assert patched == source
    assert module.EXTENDED_THINKING_PATCH_MARKER not in patched
    assert module.has_extended_thinking_support(patched) is True


def test_patch_text_permits_extended_generic_openai_compatible_efforts():
    module = load_patch_module()
    source = (
        "const GPT_52_REASONING_EFFORTS = [\n"
        "\t\"none\",\n"
        "\t\"low\",\n"
        "\t\"medium\",\n"
        "\t\"high\",\n"
        "\t\"xhigh\"\n"
        "];\n"
        "const GENERIC_REASONING_EFFORTS = [\n"
        "\t\"low\",\n"
        "\t\"medium\",\n"
        "\t\"high\"\n"
        "];\n"
        "function resolveOpenAIReasoningEffortForModel(params) {\n"
        "\treturn params.effort;\n"
        "}\n"
    )

    patched, changed = module.patch_text(source)

    assert changed is True
    assert module.EXTENDED_THINKING_PATCH_MARKER in patched
    assert 'const GPT_52_REASONING_EFFORTS = [\n\t"none",\n\t"low",\n\t"medium",\n\t"high",\n\t"xhigh",\n\t"max"\n];' in patched
    assert 'const GENERIC_REASONING_EFFORTS = [\n\t"low",\n\t"medium",\n\t"high",\n\t"xhigh",\n\t"max"\n];' in patched


def test_patch_text_accepts_native_extended_generic_openai_compatible_efforts():
    module = load_patch_module()
    source = (
        "const GPT_52_REASONING_EFFORTS = [\n"
        "\t\"none\",\n"
        "\t\"low\",\n"
        "\t\"medium\",\n"
        "\t\"high\",\n"
        "\t\"xhigh\",\n"
        "\t\"max\"\n"
        "];\n"
        "const GENERIC_REASONING_EFFORTS = [\n"
        "\t\"low\",\n"
        "\t\"medium\",\n"
        "\t\"high\",\n"
        "\t\"xhigh\",\n"
        "\t\"max\"\n"
        "];\n"
        "function resolveOpenAIReasoningEffortForModel(params) {\n"
        "\treturn params.effort;\n"
        "}\n"
    )

    patched, changed = module.patch_text(source)

    assert changed is False
    assert patched == source
    assert module.EXTENDED_THINKING_PATCH_MARKER not in patched
    assert module.has_extended_thinking_support(patched) is True


def test_patch_text_passes_explicit_thinking_levels_through_off_only_profiles():
    module = load_patch_module()
    source = (
        "function appendProfileLevel(profile, id) {\n"
        "\tif (profile.levels.some((level) => level.id === id)) return;\n"
        "\tprofile.levels.push({ id });\n"
        "}\n"
        "/** Resolve supported thinking levels and default for a provider/model pair. */\n"
        "function resolveThinkingProfile(params) {\n"
        "\tconst pluginProfile = params.pluginProfile;\n"
        "\tif (pluginProfile) {\n"
        "\t\tconst normalized = normalizeThinkingProfile(pluginProfile);\n"
        "\t\tif (normalized.levels.length > 0 && (context.reasoning !== false || pluginProfile.preserveWhenCatalogReasoningFalse === true)) return normalized;\n"
        "\t}\n"
        "\tconst profile = buildBaseThinkingProfile();\n"
        "\treturn profile;\n"
        "}\n"
        "function supportsThinkingLevel(provider, model, level, catalog) {\n"
        "\treturn true;\n"
        "}\n"
    )

    patched, changed = module.patch_text(source)

    assert changed is True
    assert module.EXTENDED_THINKING_PATCH_MARKER in patched
    assert "function applyNejumiThinkingProfilePassthrough(profile)" in patched
    assert 'appendProfileLevel(profile, "minimal");' in patched
    assert 'appendProfileLevel(profile, "low");' in patched
    assert 'appendProfileLevel(profile, "medium");' in patched
    assert 'appendProfileLevel(profile, "high");' in patched
    assert 'appendProfileLevel(profile, "xhigh");' in patched
    assert 'appendProfileLevel(profile, "max");' in patched
    assert "return applyNejumiThinkingProfilePassthrough(normalized);" in patched
    assert "return applyNejumiThinkingProfilePassthrough(profile);" in patched


def test_patch_text_upgrades_old_extended_thinking_profile_patch():
    module = load_patch_module()
    source = (
        "// Nejumi patch v1: permit xhigh/max thinking levels through provider validation.\n"
        "function appendProfileLevel(profile, id) {\n"
        "\tif (profile.levels.some((level) => level.id === id)) return;\n"
        "\tprofile.levels.push({ id });\n"
        "}\n"
        "function applyNejumiPermissiveExtendedThinkingProfile(profile) {\n"
        "\tif (!profile || !Array.isArray(profile.levels)) return profile;\n"
        "\tconst hasNonOffLevel = profile.levels.some((level) => level?.id && level.id !== \"off\");\n"
        "\tif (!hasNonOffLevel) return profile;\n"
        "\tappendProfileLevel(profile, \"xhigh\");\n"
        "\tappendProfileLevel(profile, \"max\");\n"
        "\treturn profile;\n"
        "}\n"
        "/** Resolve supported thinking levels and default for a provider/model pair. */\n"
        "function resolveThinkingProfile(params) {\n"
        "\tconst profile = buildBaseThinkingProfile();\n"
        "\treturn applyNejumiPermissiveExtendedThinkingProfile(profile);\n"
        "}\n"
        "function supportsThinkingLevel(provider, model, level, catalog) {\n"
        "\treturn true;\n"
        "}\n"
    )

    patched, changed = module.patch_text(source)

    assert changed is True
    assert module.EXTENDED_THINKING_PATCH_MARKER in patched
    assert module.OLD_EXTENDED_THINKING_PATCH_MARKERS[0] in patched
    assert "applyNejumiPermissiveExtendedThinkingProfile" not in patched
    assert "function applyNejumiThinkingProfilePassthrough(profile)" in patched
    assert 'appendProfileLevel(profile, "high");' in patched
    assert "return applyNejumiThinkingProfilePassthrough(profile);" in patched


def test_patch_text_treats_absent_extended_level_map_as_provider_authoritative():
    module = load_patch_module()
    source = (
        "const EXTENDED_THINKING_LEVELS = [\"off\", \"minimal\", \"low\", \"medium\", \"high\", \"xhigh\", \"max\"];\n"
        "function getSupportedThinkingLevels(model) {\n"
        "\treturn EXTENDED_THINKING_LEVELS.filter((level) => {\n"
        "\t\tconst mapped = thinkingLevelMap?.[level];\n"
        "\t\tif (mapped === null) return false;\n"
        "\t\tif (level === \"xhigh\" || level === \"max\") return mapped !== void 0;\n"
        "\t\treturn true;\n"
        "\t});\n"
        "}\n"
    )

    patched, changed = module.patch_text(source)

    assert changed is True
    assert module.EXTENDED_THINKING_PATCH_MARKER in patched
    assert 'if (level === "xhigh" || level === "max") return mapped !== null;' in patched


def test_nemoclaw_runtime_patch_selects_openshell_sandbox_container():
    module = load_nemoclaw_patch_module()
    containers = module.parse_docker_ps_lines(
        "abc123\topenshell-nejumi-taiwan-b7a88e90-70c2-4478-982e-e7fdc9a91225\n"
        "def456\tunrelated\n"
    )

    candidates = module.sandbox_container_candidates(containers, "nejumi-taiwan")

    assert candidates == [
        {
            "id": "abc123",
            "name": "openshell-nejumi-taiwan-b7a88e90-70c2-4478-982e-e7fdc9a91225",
        }
    ]


def test_nemoclaw_runtime_patch_runs_docker_exec_as_root(tmp_path, monkeypatch):
    module = load_nemoclaw_patch_module()
    patch_script = tmp_path / "patch.py"
    patch_script.write_text("print('patch')\n", encoding="utf-8")
    commands = []

    def fake_run_command(command, *, timeout_seconds):
        commands.append(command)
        if command[:2] == ["docker", "cp"]:
            return module.subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:5] == ["docker", "exec", "-u", "root", "cid123"]:
            return module.subprocess.CompletedProcess(
                command,
                0,
                stdout='{"ok": true, "patched": ["/usr/local/lib/node_modules/openclaw/dist/selection.js"], "already_patched": []}\n',
                stderr="",
            )
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(module, "run_command", fake_run_command)

    payload = module.patch_container_openclaw(
        docker_bin="docker",
        container_id="cid123",
        patch_script=patch_script,
        remote_patch_script="/tmp/patch.py",
        openclaw_package_dir="/usr/local/lib/node_modules/openclaw",
        check_only=False,
        timeout_seconds=10,
    )

    assert payload["ok"] is True
    assert commands[0] == ["docker", "cp", str(patch_script.resolve()), "cid123:/tmp/patch.py"]
    assert commands[1] == [
        "docker",
        "exec",
        "-u",
        "root",
        "cid123",
        "python3",
        "/tmp/patch.py",
        "--openclaw-package-dir",
        "/usr/local/lib/node_modules/openclaw",
        "--json",
    ]
