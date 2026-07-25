import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
BFCL_ROOT = REPO_ROOT / "scripts" / "evaluator" / "evaluate_utils" / "bfcl_pkg"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(BFCL_ROOT))


def test_claude_direct_bfcl_model_is_registered():
    from bfcl.constants.model_config import MODEL_CONFIG_MAPPING
    from bfcl.model_handler.api_inference.claude import ClaudeHandler

    config = MODEL_CONFIG_MAPPING["Claude-FC"]

    assert config.model_handler is ClaudeHandler
    assert config.is_fc_model is True
    assert config.underscore_to_dot is True


def test_fable_bfcl_request_uses_effort_without_manual_thinking_or_temperature():
    from bfcl.model_handler.api_inference.claude import ClaudeHandler

    captured = {}
    handler = object.__new__(ClaudeHandler)
    handler.model_name = "Claude-FC"
    handler.actual_model_name = "claude-fable-5"
    handler.effort = "high"
    handler.temperature = 0.01

    def fake_generate_with_backoff(**kwargs):
        captured.update(kwargs)
        return object(), 0.1

    handler.generate_with_backoff = fake_generate_with_backoff
    handler._query_FC(
        {
            "caching_enabled": False,
            "message": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "tools": [],
        }
    )

    assert captured["model"] == "claude-fable-5"
    assert captured["max_tokens"] == 128_000
    assert captured["output_config"] == {"effort": "high"}
    assert "thinking" not in captured
    assert "temperature" not in captured
