import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
EVALUATOR = SCRIPTS / "evaluator"
BFCL_V4 = EVALUATOR / "evaluate_utils" / "bfcl_v4_pkg"
for path in (SCRIPTS, EVALUATOR, BFCL_V4):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.configured_llm import (
    ConfiguredLLMHandler,
    _model_style_for_api,
)


def test_configured_handler_selects_provider_native_model_style():
    assert _model_style_for_api("openai_responses") == ModelStyle.OPENAI_RESPONSES
    assert _model_style_for_api("xai_responses") == ModelStyle.OPENAI_RESPONSES
    assert _model_style_for_api("anthropic") == ModelStyle.ANTHROPIC
    assert _model_style_for_api("openai-compatible") == ModelStyle.OPENAI_COMPLETIONS


def test_responses_history_replays_native_items_and_tool_results():
    handler = object.__new__(ConfiguredLLMHandler)
    handler.model_style = ModelStyle.OPENAI_RESPONSES
    inference_data = {"message": []}

    handler.add_first_turn_message_FC(
        inference_data,
        [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "hello"},
        ],
    )
    handler._add_assistant_message_FC(
        inference_data,
        {
            "model_responses_message_for_chat_history": [
                {"type": "reasoning", "encrypted_content": "opaque"},
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "weather",
                    "arguments": "{}",
                },
            ]
        },
    )
    handler._add_execution_results_FC(
        inference_data,
        ["sunny"],
        {"tool_call_ids": ["call-1"]},
    )

    assert inference_data["message"][0]["role"] == "developer"
    assert inference_data["message"][-1] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": "sunny",
    }
