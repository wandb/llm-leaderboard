import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = REPO_ROOT / "scripts" / "tools" / "verify_openclaw_runtime_content_hooks.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def write_runtime(dist_dir: Path, text: str) -> None:
    dist_dir.mkdir(parents=True)
    (dist_dir / "runtime.js").write_text(text, encoding="utf-8")


def complete_runtime_text() -> str:
    return """
function modelCallHookEventBase(eventBase) {
  return {callId: eventBase.callId, provider: eventBase.provider, model: eventBase.model};
}
function dispatchModelCallStartedHook(eventBase) {
  if (hookRunner.hasHooks("model_call_started")) {
    hookRunner.runModelCallStarted(modelCallHookEventBase(eventBase), {});
  }
}
function submitPrompt(params) {
  if (hookRunner.hasHooks("llm_input")) {
    hookRunner.runLlmInput({
      systemPrompt: params.systemPrompt,
      prompt: params.prompt,
      historyMessages: params.historyMessages,
      tools: params.tools,
    }, {});
  }
}
function finishPrompt(params) {
  if (hookRunner.hasHooks("llm_output")) {
    hookRunner.runLlmOutput({
      assistantTexts: params.assistantTexts,
      lastAssistant: params.lastAssistant,
      usage: params.usage,
    }, {});
  }
}
async function beforeTool(toolName, params, toolCallId) {
  if (hookRunner.hasHooks("before_tool_call")) {
    return hookRunner.runBeforeToolCall({toolName, params, toolCallId}, {});
  }
}
async function afterTool(toolName, params, result, toolCallId) {
  if (hookRunner.hasHooks("after_tool_call")) {
    return hookRunner.runAfterToolCall({toolName, params, result, toolCallId}, {});
  }
}
function beforeMessage(message) {
  if (hookRunner.hasHooks("before_message_write")) {
    return hookRunner.runBeforeMessageWrite({message}, {});
  }
}
"""


def test_verify_runtime_accepts_content_bearing_hooks(tmp_path):
    module = load_module()
    dist_dir = tmp_path / "openclaw" / "dist"
    write_runtime(dist_dir, complete_runtime_text())

    payload = module.verify_runtime(dist_dir, tmp_path / "openclaw")

    assert payload["ok"] is True
    assert {check["hook"] for check in payload["checks"]} == {
        "model_call_started",
        "llm_input",
        "llm_output",
        "before_tool_call",
        "after_tool_call",
        "before_message_write",
    }


def test_verify_runtime_rejects_tool_hook_without_result_content(tmp_path):
    module = load_module()
    dist_dir = tmp_path / "openclaw" / "dist"
    broken = (
        complete_runtime_text()
        .replace("params, result, toolCallId", "params, toolCallId")
        .replace("params, result, toolCallId}", "params, toolCallId}")
    )
    write_runtime(dist_dir, broken)

    payload = module.verify_runtime(dist_dir, tmp_path / "openclaw")

    assert payload["ok"] is False
    after_tool = next(check for check in payload["checks"] if check["hook"] == "after_tool_call")
    assert "result" in after_tool["missing_terms"]


def test_verify_runtime_rejects_missing_hook(tmp_path):
    module = load_module()
    dist_dir = tmp_path / "openclaw" / "dist"
    write_runtime(dist_dir, complete_runtime_text().replace('"llm_output"', '"llm_output_missing"'))

    payload = module.verify_runtime(dist_dir, tmp_path / "openclaw")

    assert payload["ok"] is False
    llm_output = next(check for check in payload["checks"] if check["hook"] == "llm_output")
    assert llm_output["occurrences"] == 0
