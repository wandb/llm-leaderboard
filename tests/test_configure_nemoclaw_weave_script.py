import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "setup" / "configure_nemoclaw_weave.sh"
POLICY = REPO_ROOT / "configs" / "nemoclaw" / "policies" / "wandb_weave.yaml"


def write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def test_configure_nemoclaw_weave_shell_syntax():
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_wandb_weave_policy_allows_direct_model_apis_without_broad_network():
    text = POLICY.read_text(encoding="utf-8")

    assert "host: api.inference.wandb.ai" in text
    assert "path: /v1/models" in text
    assert "path: /v1/models/**" in text
    assert "path: /v1/chat/completions" in text
    assert "host: api.openai.com" in text
    assert "path: /v1/responses" in text
    assert "host: api.anthropic.com" in text
    assert "path: /v1/messages" in text


def test_configure_nemoclaw_weave_registers_wandb_inference_model(tmp_path):
    fake_nemoclaw = tmp_path / "nemoclaw"
    write_executable(
        fake_nemoclaw,
        """#!/usr/bin/env python3
import os
import subprocess
import sys

args = sys.argv[1:]
if args == ["sandbox", "status", "nejumi-test"]:
    print("host: api.wandb.ai")
    print("host: api.inference.wandb.ai")
    print("host: api.openai.com")
    print("host: api.anthropic.com")
    raise SystemExit(0)
if len(args) >= 3 and args[:2] == ["sandbox", "exec"]:
    try:
        command_index = args.index("--")
    except ValueError:
        print(f"missing command separator: {args}", file=sys.stderr)
        raise SystemExit(1)
    command = args[command_index + 1 :]
    result = subprocess.run(command, stdin=sys.stdin.buffer)
    raise SystemExit(result.returncode)
print(f"unexpected nemoclaw args: {args}", file=sys.stderr)
raise SystemExit(1)
""",
    )
    env_file = tmp_path / ".env"
    env_file.write_text(
        "WANDB_API_KEY=wandb-test\nOPENAI_API_KEY=openai-test\nANTHROPIC_API_KEY=anthropic-test\n",
        encoding="utf-8",
    )
    secret_file = tmp_path / "secrets.json"
    config_file = tmp_path / "openclaw.json"
    report_file = tmp_path / "report.json"
    model_params = {
        "extra_body": {"provider": {"order": ["Z.AI"]}},
        "temperature": 0.0,
    }

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--sandbox",
            "nejumi-test",
            "--nemoclaw-bin",
            str(fake_nemoclaw),
            "--env-file",
            str(env_file),
            "--secret-file",
            str(secret_file),
            "--openclaw-config",
            str(config_file),
            "--skip-policy",
            "--skip-plugin-install",
            "--wandb-inference-model-id",
            "z-ai/glm-5.2",
            "--wandb-inference-max-tokens",
            "8192",
            "--wandb-inference-context-window",
            "131072",
            "--wandb-inference-reasoning",
            "true",
            "--wandb-inference-model-params-json",
            json.dumps(model_params),
            "--json",
            str(report_file),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["wandb_inference_enabled"] is True
    assert report["wandb_inference_config_ok"] is True
    assert report["wandb_inference_model_params"] == model_params
    assert report["secret_value_in_report"] is False
    assert report["openai_secret_value_in_report"] is False
    assert report["anthropic_secret_value_in_report"] is False
    assert report["anthropic_direct_config_ok"] is True

    config = json.loads(config_file.read_text(encoding="utf-8"))
    provider = config["models"]["providers"]["wandb-inference"]
    assert provider["baseUrl"] == "https://api.inference.wandb.ai/v1"
    assert provider["api"] == "openai-completions"
    assert provider["apiKey"] == {
        "source": "file",
        "provider": "nejumi-wandb",
        "id": "/wandb/apiKey",
    }
    model = next(item for item in provider["models"] if item["id"] == "z-ai/glm-5.2")
    assert model["maxTokens"] == 8192
    assert model["contextWindow"] == 131072
    assert model["reasoning"] is True
    assert model["params"] == model_params

    secrets = json.loads(secret_file.read_text(encoding="utf-8"))
    assert secrets["wandb"]["apiKey"] == "wandb-test"
    assert secrets["openai"]["apiKey"] == "openai-test"
    assert secrets["anthropic"]["apiKey"] == "anthropic-test"

    anthropic_provider = config["models"]["providers"]["anthropic"]
    assert anthropic_provider["baseUrl"] == "https://api.anthropic.com"
    assert anthropic_provider["api"] == "anthropic-messages"
    assert anthropic_provider["apiKey"] == {
        "source": "file",
        "provider": "nejumi-anthropic",
        "id": "/anthropic/apiKey",
    }
    fable = next(
        item for item in anthropic_provider["models"] if item["id"] == "claude-fable-5"
    )
    assert fable["contextWindow"] == 1_000_000
    assert fable["maxTokens"] == 128_000
    sonnet = next(
        item for item in anthropic_provider["models"] if item["id"] == "claude-sonnet-4-6"
    )
    assert sonnet["api"] == "anthropic-messages"
    assert sonnet["reasoning"] is True
    assert sonnet["contextWindow"] == 1_000_000
    assert sonnet["maxTokens"] == 64_000
