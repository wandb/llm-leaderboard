import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "setup" / "configure_nemoclaw_openrouter.sh"
POLICY = REPO_ROOT / "configs" / "nemoclaw" / "policies" / "openrouter_inference.yaml"


def write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def fake_nemoclaw(tmp_path: Path) -> Path:
    path = tmp_path / "nemoclaw"
    write_executable(
        path,
        """#!/usr/bin/env python3
import subprocess
import sys

args = sys.argv[1:]
if args == ["sandbox", "status", "nejumi-test"]:
    print("host: openrouter.ai")
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
    return path


def test_configure_nemoclaw_openrouter_shell_syntax():
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_openrouter_policy_allows_only_openrouter_host():
    text = POLICY.read_text(encoding="utf-8")

    assert "host: openrouter.ai" in text
    assert "path: /**" in text
    assert "api.inference.wandb.ai" not in text


def test_configure_nemoclaw_openrouter_registers_pinned_glm_model(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=or-test\n", encoding="utf-8")
    secret_file = tmp_path / "secrets.json"
    config_file = tmp_path / "openclaw.json"
    report_file = tmp_path / "report.json"
    model_params = {
        "provider": {
            "order": ["z-ai/fp8"],
            "only": ["z-ai/fp8"],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
    }

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--sandbox",
            "nejumi-test",
            "--nemoclaw-bin",
            str(fake_nemoclaw(tmp_path)),
            "--env-file",
            str(env_file),
            "--secret-file",
            str(secret_file),
            "--openclaw-config",
            str(config_file),
            "--skip-policy",
            "--openclaw-model-params-json",
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
    assert report["config_ok"] is True
    assert report["openclaw_model_params"] == model_params
    assert report["secret_value_in_report"] is False

    config = json.loads(config_file.read_text(encoding="utf-8"))
    provider = config["models"]["providers"]["openrouter-direct"]
    assert provider["baseUrl"] == "https://openrouter.ai/api/v1"
    assert provider["api"] == "openai-completions"
    assert provider["apiKey"] == {
        "source": "file",
        "provider": "nejumi-openrouter",
        "id": "/openrouter/apiKey",
    }
    glm = next(item for item in provider["models"] if item["id"] == "z-ai/glm-5.2")
    assert glm["params"] == model_params
    assert glm["maxTokens"] == 32768

    secrets = json.loads(secret_file.read_text(encoding="utf-8"))
    assert secrets["openrouter"]["apiKey"] == "or-test"


def test_configure_nemoclaw_openrouter_rejects_non_object_model_params(tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--nemoclaw-bin",
            str(fake_nemoclaw(tmp_path)),
            "--openclaw-model-params-json",
            "[]",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "--openclaw-model-params-json must be a JSON object" in result.stderr
