from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
BFCL_ROOT = ROOT / "scripts" / "evaluator" / "evaluate_utils" / "bfcl_pkg"
SCRIPTS_ROOT = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(BFCL_ROOT))

from bfcl.model_handler.api_inference.openai_response import OpenAIResponsesHandler


def _handler():
    handler = object.__new__(OpenAIResponsesHandler)
    handler.model_name = "test-model-FC"
    handler.is_fc_model = False
    return handler


def test_fc_decode_treats_text_response_as_no_tool_call():
    handler = _handler()

    assert handler.decode_ast("需要更多資訊才能繼續。") == []
    assert handler.decode_execute("需要更多資訊才能繼續。") == []


def test_fc_decode_keeps_native_tool_calls():
    handler = _handler()

    raw = [{"search": '{"query":"台北"}'}]

    assert handler.decode_ast(raw) == [{"search": {"query": "台北"}}]
    assert handler.decode_execute(raw) == ["search(query='台北')"]
