import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = REPO_ROOT / "scripts" / "setup" / "configure_openclaw_gateway_env.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def make_args(tmp_path, env_file, config_file):
    return Namespace(
        env_file=env_file,
        openclaw_config=config_file,
        service_name="openclaw-gateway.service",
        dropin_name="20-nejumi-env.conf",
        systemd_user_dir=tmp_path / "systemd" / "user",
        required_key=[],
        write=False,
        restart=False,
        check_only=True,
        json=True,
    )


def test_find_env_secret_refs_recurses_without_values():
    module = load_module()
    payload = {
        "plugins": {
            "entries": {
                "weave": {"config": {"apiKey": {"source": "env", "id": "WANDB_API_KEY"}}}
            }
        },
        "models": {
            "providers": {
                "openai-direct": {"apiKey": {"source": "env", "id": "OPENAI_API_KEY"}},
                "literal": {"apiKey": {"source": "literal", "value": "do-not-report"}},
            }
        },
    }

    assert module.find_env_secret_refs(payload) == {"WANDB_API_KEY", "OPENAI_API_KEY"}


def test_parse_env_file_reports_present_keys_and_invalid_lines(tmp_path):
    module = load_module()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "WANDB_API_KEY=abc",
                "export OPENAI_API_KEY='sk-test'",
                "EMPTY_KEY=",
                "not a valid line",
                "1BAD=value",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    info = module.parse_env_file(env_file)

    assert info.keys_present == {"WANDB_API_KEY", "OPENAI_API_KEY"}
    assert info.invalid_lines == [4, 5]


def test_write_dropin_and_status_without_secret_values(tmp_path):
    module = load_module()
    env_file = tmp_path / ".env"
    env_file.write_text("WANDB_API_KEY=abc\nOPENAI_API_KEY=sk-test\n", encoding="utf-8")
    config_file = tmp_path / "openclaw.json"
    config_file.write_text(
        json.dumps(
            {
                "plugins": {
                    "entries": {
                        "weave": {
                            "config": {"apiKey": {"source": "env", "id": "WANDB_API_KEY"}}
                        }
                    }
                },
                "models": {
                    "providers": {
                        "openai-direct": {
                            "apiKey": {"source": "env", "id": "OPENAI_API_KEY"}
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    args = make_args(tmp_path, env_file, config_file)
    dropin = module.dropin_path(args.systemd_user_dir, args.service_name, args.dropin_name)

    wrote = module.write_dropin(dropin, env_file.resolve())
    payload = module.build_status(args, wrote=wrote)

    assert wrote is True
    assert payload["ok"] is True
    assert payload["secret_refs_required"] == ["OPENAI_API_KEY", "WANDB_API_KEY"]
    assert payload["secret_refs_missing"] == []
    assert "sk-test" not in json.dumps(payload)
    assert dropin.read_text(encoding="utf-8") == f"[Service]\nEnvironmentFile={env_file.resolve()}\n\n"


def test_status_fails_when_required_secret_missing(tmp_path):
    module = load_module()
    env_file = tmp_path / ".env"
    env_file.write_text("WANDB_API_KEY=abc\n", encoding="utf-8")
    config_file = tmp_path / "openclaw.json"
    config_file.write_text(
        json.dumps({"models": {"providers": {"openai-direct": {"apiKey": {"source": "env", "id": "OPENAI_API_KEY"}}}}}),
        encoding="utf-8",
    )
    args = make_args(tmp_path, env_file, config_file)
    module.write_dropin(
        module.dropin_path(args.systemd_user_dir, args.service_name, args.dropin_name),
        env_file.resolve(),
    )

    payload = module.build_status(args)

    assert payload["ok"] is False
    assert payload["secret_refs_missing"] == ["OPENAI_API_KEY"]
