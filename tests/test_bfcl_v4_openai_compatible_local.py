import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
EVALUATOR = SCRIPTS / "evaluator"
BFCL_V4 = EVALUATOR / "evaluate_utils" / "bfcl_v4_pkg"
for path in (SCRIPTS, EVALUATOR, BFCL_V4):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from bfcl_eval.model_handler.configured_llm import ConfiguredLLMHandler
from llm_inference_adapter import OpenAIClient


def test_openai_compatible_handler_round_trip_over_local_http(monkeypatch):
    captured = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            captured["path"] = self.path
            captured["request"] = json.loads(body)
            response = {
                "id": "chatcmpl-local",
                "object": "chat.completion",
                "created": 1,
                "model": "local-oss-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-local-1",
                                    "type": "function",
                                    "function": {
                                        "name": "weather",
                                        "arguments": '{"city":"Taipei"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 5,
                    "total_tokens": 14,
                },
            }
            encoded = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    cfg = OmegaConf.create(
        {
            "api": "openai-compatible",
            "base_url": f"http://127.0.0.1:{server.server_port}/v1",
            "batch_size": 1,
            "inference_interval": 0,
            "provider_rate_limit": {"enabled": False},
            "error_handling": {"request_failure": {"mode": "hard"}},
            "model": {"pretrained_model_name_or_path": "local-oss-model"},
            "generator": {"temperature": 0.01, "max_tokens": 64},
            "bfcl": {
                "generator_config": {"temperature": 0.01, "max_tokens": 64},
                "request_timeout_sec": 5,
                "request_max_retries": 0,
                "backoff_max_time_sec": 1,
                "backoff_max_tries": 1,
            },
        }
    )
    singleton = SimpleNamespace(config=cfg, llm=None)
    monkeypatch.setattr(
        "config_singleton.WandbConfigSingleton.get_instance",
        lambda: singleton,
    )

    client = OpenAIClient(
        api_key="test",
        base_url=cfg.base_url,
        model=cfg.model.pretrained_model_name_or_path,
        **cfg.generator,
    )
    singleton.llm = client
    bfcl_handler = ConfiguredLLMHandler(
        model_name="local-oss-model",
        temperature=0.01,
        registry_name="unified-oss-fc",
        is_fc_model=True,
    )

    try:
        response, _latency = bfcl_handler._query_FC(
            {
                "message": [{"role": "user", "content": "台北天氣如何？"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "description": "取得天氣",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {"type": "string"},
                                },
                                "required": ["city"],
                            },
                        },
                    }
                ],
            }
        )
        parsed = bfcl_handler._parse_query_response_FC(response)
    finally:
        bfcl_handler.llm_processor.close_sync_loop()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert captured["path"] == "/v1/chat/completions"
    assert captured["request"]["model"] == "local-oss-model"
    assert captured["request"]["tools"][0]["function"]["name"] == "weather"
    assert parsed["model_responses"] == [
        {"weather": '{"city": "Taipei"}'}
    ]
    assert parsed["tool_call_ids"] == ["call-local-1"]
    assert parsed["input_token"] == 9
    assert parsed["output_token"] == 5
