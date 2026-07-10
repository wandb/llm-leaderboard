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
