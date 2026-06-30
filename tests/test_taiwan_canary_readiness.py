import importlib.util
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = REPO_ROOT / "scripts" / "tools" / "check_taiwan_canary_readiness.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def test_nemoclaw_optional_readiness_does_not_fail_when_missing():
    module = load_module()

    checks = module.check_nemoclaw(
        {"PATH": ""},
        nemoclaw_bin="nemoclaw",
        sandbox="nejumi-taiwan",
        require=False,
    )

    assert all(check.ok for check in checks)
    assert any("missing" in check.detail for check in checks)


def test_nemoclaw_required_readiness_fails_when_missing():
    module = load_module()

    checks = module.check_nemoclaw(
        {"PATH": ""},
        nemoclaw_bin="nemoclaw",
        sandbox="nejumi-taiwan",
        require=True,
    )

    assert not all(check.ok for check in checks)
    assert any(check.name == "NeMoClaw command is available" and not check.ok for check in checks)


def test_nemoclaw_required_readiness_passes_with_sandbox_preflight(tmp_path):
    module = load_module()
    nemoclaw = tmp_path / "nemoclaw"
    openshell = tmp_path / "openshell"
    nemoclaw.write_text(
        """#!/usr/bin/env sh
if [ "$1" = "--version" ]; then
  echo "nemoclaw 0.0.test"
  exit 0
fi
if [ "$1" = "status" ] && [ "$2" = "--json" ]; then
  cat <<'JSON'
{"sandboxes":[{"name":"nejumi-taiwan","provider":"compatible-endpoint","model":"test-model","connected":false,"policies":["wandb-weave"]}]}
JSON
  exit 0
fi
if [ "$1" = "sandbox" ] && [ "$2" = "status" ]; then
  echo "sandbox ok"
  exit 0
fi
if [ "$1" = "sandbox" ] && [ "$2" = "exec" ]; then
  echo "openclaw 2026.6.9"
  exit 0
fi
echo "unexpected $*" >&2
exit 1
""",
        encoding="utf-8",
    )
    openshell.write_text("#!/usr/bin/env sh\necho openshell 0.0.test\n", encoding="utf-8")
    nemoclaw.chmod(0o755)
    openshell.chmod(0o755)
    env = {
        "PATH": str(tmp_path) + os.pathsep + os.environ.get("PATH", ""),
        **{key: value for key, value in os.environ.items() if key.startswith("HOME")},
    }

    checks = module.check_nemoclaw(
        env,
        nemoclaw_bin="nemoclaw",
        sandbox="nejumi-taiwan",
        require=True,
    )

    assert all(check.ok for check in checks)
    assert any(
        check.name == "OpenClaw runs inside NeMoClaw sandbox: nejumi-taiwan"
        and "openclaw" in check.detail
        for check in checks
    )
    policy_check = next(
        check
        for check in checks
        if check.name == "NeMoClaw sandbox runtime policy is introspectable: nejumi-taiwan"
    )
    assert '"policy_count": 1' in policy_check.detail
    assert "wandb-weave" in policy_check.detail
    weave_check = next(
        check
        for check in checks
        if check.name == "NeMoClaw W&B/Weave runtime policy is present: nejumi-taiwan"
    )
    assert weave_check.ok


def test_nemoclaw_required_readiness_accepts_detailed_status_runtime_policy(tmp_path):
    module = load_module()
    nemoclaw = tmp_path / "nemoclaw"
    openshell = tmp_path / "openshell"
    nemoclaw.write_text(
        """#!/usr/bin/env sh
if [ "$1" = "--version" ]; then
  echo "nemoclaw 0.0.test"
  exit 0
fi
if [ "$1" = "status" ] && [ "$2" = "--json" ]; then
  cat <<'JSON'
{"sandboxes":[{"name":"nejumi-taiwan","provider":"compatible-endpoint","model":"test-model","connected":false,"policies":[]}]}
JSON
  exit 0
fi
if [ "$1" = "sandbox" ] && [ "$2" = "status" ]; then
  cat <<'TEXT'
Sandbox: nejumi-taiwan
Policy:
  network_policies:
    clawhub:
      name: clawhub
    wandb-weave:
      name: wandb-weave
TEXT
  exit 0
fi
if [ "$1" = "sandbox" ] && [ "$2" = "exec" ]; then
  echo "openclaw 2026.6.9"
  exit 0
fi
echo "unexpected $*" >&2
exit 1
""",
        encoding="utf-8",
    )
    openshell.write_text("#!/usr/bin/env sh\necho openshell 0.0.test\n", encoding="utf-8")
    nemoclaw.chmod(0o755)
    openshell.chmod(0o755)
    env = {
        "PATH": str(tmp_path) + os.pathsep + os.environ.get("PATH", ""),
        **{key: value for key, value in os.environ.items() if key.startswith("HOME")},
    }

    checks = module.check_nemoclaw(
        env,
        nemoclaw_bin="nemoclaw",
        sandbox="nejumi-taiwan",
        require=True,
    )

    assert all(check.ok for check in checks)
    policy_check = next(
        check
        for check in checks
        if check.name == "NeMoClaw sandbox runtime policy is introspectable: nejumi-taiwan"
    )
    assert '"summary_policy_count": 0' in policy_check.detail
    assert '"detailed_status_network_policy_count": 2' in policy_check.detail
    assert '"wandb_weave_policy_present": true' in policy_check.detail
    weave_check = next(
        check
        for check in checks
        if check.name == "NeMoClaw W&B/Weave runtime policy is present: nejumi-taiwan"
    )
    assert weave_check.ok


def test_nemoclaw_required_readiness_fails_without_wandb_weave_runtime_policy(tmp_path):
    module = load_module()
    nemoclaw = tmp_path / "nemoclaw"
    openshell = tmp_path / "openshell"
    nemoclaw.write_text(
        """#!/usr/bin/env sh
if [ "$1" = "--version" ]; then
  echo "nemoclaw 0.0.test"
  exit 0
fi
if [ "$1" = "status" ] && [ "$2" = "--json" ]; then
  cat <<'JSON'
{"sandboxes":[{"name":"nejumi-taiwan","provider":"compatible-endpoint","model":"test-model","connected":false,"policies":[]}]}
JSON
  exit 0
fi
if [ "$1" = "sandbox" ] && [ "$2" = "status" ]; then
  echo "sandbox ok"
  exit 0
fi
if [ "$1" = "sandbox" ] && [ "$2" = "exec" ]; then
  echo "openclaw 2026.6.9"
  exit 0
fi
echo "unexpected $*" >&2
exit 1
""",
        encoding="utf-8",
    )
    openshell.write_text("#!/usr/bin/env sh\necho openshell 0.0.test\n", encoding="utf-8")
    nemoclaw.chmod(0o755)
    openshell.chmod(0o755)
    env = {
        "PATH": str(tmp_path) + os.pathsep + os.environ.get("PATH", ""),
        **{key: value for key, value in os.environ.items() if key.startswith("HOME")},
    }

    checks = module.check_nemoclaw(
        env,
        nemoclaw_bin="nemoclaw",
        sandbox="nejumi-taiwan",
        require=True,
    )

    assert not all(check.ok for check in checks)
    policy_check = next(
        check
        for check in checks
        if check.name == "NeMoClaw sandbox runtime policy is introspectable: nejumi-taiwan"
    )
    assert '"policy_count": 0' in policy_check.detail
    assert '"policy_configured": false' in policy_check.detail
    assert "OpenClaw deny_tool" in policy_check.detail
    weave_check = next(
        check
        for check in checks
        if check.name == "NeMoClaw W&B/Weave runtime policy is present: nejumi-taiwan"
    )
    assert not weave_check.ok


def test_weave_content_canary_gate_passes_only_with_passed_json(tmp_path):
    module = load_module()
    gate = tmp_path / "canary.gate.json"
    gate.write_text('{"ok": true, "status": "passed"}', encoding="utf-8")

    checks = module.check_weave_content_canary_gate(gate, require=True)

    assert all(check.ok for check in checks)


def test_weave_content_canary_gate_fails_for_provider_failure(tmp_path):
    module = load_module()
    gate = tmp_path / "canary.gate.json"
    gate.write_text(
        '{"ok": false, "status": "provider_failure", "failure_kind": "provider_quota"}',
        encoding="utf-8",
    )

    checks = module.check_weave_content_canary_gate(gate, require=True)

    assert not all(check.ok for check in checks)
    assert "provider_quota" in checks[0].detail


def test_weave_content_canary_gate_required_fails_when_unconfigured():
    module = load_module()

    checks = module.check_weave_content_canary_gate(None, require=True)

    assert not all(check.ok for check in checks)
    assert "not configured" in checks[0].detail


def test_manifest_check_accepts_openai_canary_without_opus(tmp_path):
    module = load_module()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """
models:
  - slug: gpt-4_1-mini-openai-direct-canary
    source_config: config-gpt-4.1-mini-2025-04-14.yaml
    openclaw_model: openai-direct/gpt-4.1-mini-2025-04-14
    canary: true
""",
        encoding="utf-8",
    )

    checks = module.check_manifest(
        manifest,
        expected_slug="gpt-4_1-mini-openai-direct-canary",
    )

    assert all(check.ok for check in checks)


def test_openclaw_config_check_is_provider_generic(tmp_path):
    module = load_module()
    config = tmp_path / "openclaw.json"
    config.write_text(
        """
{
  "models": {
    "providers": {
      "openai-direct": {
        "models": [
          {"id": "gpt-4.1-mini-2025-04-14"}
        ]
      }
    }
  },
  "plugins": {
    "entries": {
      "weave": {"enabled": true}
    }
  }
}
""",
        encoding="utf-8",
    )

    checks = module.check_openclaw_config(
        config,
        openclaw_model="openai-direct/gpt-4.1-mini-2025-04-14",
    )

    assert all(check.ok for check in checks)
    assert checks[0].name == "OpenClaw openai-direct provider exists"


def test_openai_direct_env_check_uses_openai_api_key():
    module = load_module()

    checks = module.check_env(
        {"WANDB_API_KEY": "wandb", "OPENAI_API_KEY": "openai"},
        openclaw_model="openai-direct/gpt-4.1-mini-2025-04-14",
    )
    missing = module.check_env(
        {"WANDB_API_KEY": "wandb"},
        openclaw_model="openai-direct/gpt-4.1-mini-2025-04-14",
    )

    assert all(check.ok for check in checks)
    assert any(check.name.startswith("OPENAI_API_KEY") and not check.ok for check in missing)


def test_default_canary_readiness_targets_openai_direct_canary():
    module = load_module()

    assert module.DEFAULT_MANIFEST.name == "taiwan_openai_canary_models.yaml"
    assert module.DEFAULT_CANARY_SLUG == "gpt-4_1-mini-openai-direct-canary"
    assert module.DEFAULT_OPENCLAW_MODEL == "openai-direct/gpt-4.1-mini-2025-04-14"
    assert all("openrouter" not in str(path).lower() for path in module.DEFAULT_CANARY_CONFIG_DIRS.values())
    assert all("generated_openai_canary" in str(path) for path in module.DEFAULT_CANARY_CONFIG_DIRS.values())


def test_default_agentic_config_dir_switches_only_when_nemoclaw_required():
    module = load_module()

    assert module.default_generated_agentic_dir(require_nemoclaw=False) == (
        REPO_ROOT / "configs/taiwan_full/generated_openai_canary_agentic"
    )
    assert module.default_generated_agentic_dir(require_nemoclaw=True) == (
        REPO_ROOT / "configs/taiwan_full/generated_openai_canary_agentic_nemoclaw"
    )


def test_generated_agentic_config_requires_nemoclaw_routing_when_requested(tmp_path):
    module = load_module()
    base = tmp_path / "base.yaml"
    generated = tmp_path / "agentic.yaml"
    base.write_text(
        """
model:
  pretrained_model_name_or_path: gpt-4.1-mini-2025-04-14
""",
        encoding="utf-8",
    )
    generated.write_text(
        """
model:
  pretrained_model_name_or_path: gpt-4.1-mini-2025-04-14
agentic_math:
  openclaw_model: openai-direct/gpt-4.1-mini-2025-04-14
  deny_tool: [code_execution, web_search, web_fetch, browser, browser_*, '*search*']
  deny_argument_pattern: ['https?://', '\\b(curl|wget)\\b', '\\b(requests|urllib|httpx)\\.']
  nemoclaw_sandbox: nejumi-taiwan
  use_task_agent: true
swebench_pro:
  openclaw_model: openai-direct/gpt-4.1-mini-2025-04-14
  deny_tool: [code_execution, web_search, web_fetch, browser, browser_*, '*search*']
  deny_argument_pattern: ['https?://', '\\b(curl|wget)\\b', '\\b(requests|urllib|httpx)\\.']
  nemoclaw_sandbox: nejumi-taiwan
  nemoclaw_checkout_transfer_mode: copy
""",
        encoding="utf-8",
    )

    checks = module.check_generated_configs(
        base,
        {"agentic": generated},
        expected_pretrained_model="gpt-4.1-mini-2025-04-14",
        openclaw_model="openai-direct/gpt-4.1-mini-2025-04-14",
        require_nemoclaw=True,
        nemoclaw_sandbox="nejumi-taiwan",
    )

    assert all(check.ok for check in checks)
    assert any(check.name == "agentic config uses NeMoClaw sandbox" for check in checks)
    assert any(check.name == "agentic config keeps task-agent enabled" for check in checks)
    assert any(check.name == "agentic SWE checkout is sandbox-accessible" for check in checks)
    assert any(check.name == "agentic Math denies remote lookup via deny_tool" for check in checks)
    assert any(check.name == "agentic SWE denies remote lookup via deny_tool" for check in checks)
    assert any(
        check.name == "agentic Math denies remote lookup via deny_argument_pattern"
        for check in checks
    )
    assert any(
        check.name == "agentic SWE denies remote lookup via deny_argument_pattern"
        for check in checks
    )


def test_generated_agentic_config_rejects_non_nemoclaw_routing_when_required(tmp_path):
    module = load_module()
    base = tmp_path / "base.yaml"
    generated = tmp_path / "agentic.yaml"
    base.write_text(
        """
model:
  pretrained_model_name_or_path: gpt-4.1-mini-2025-04-14
""",
        encoding="utf-8",
    )
    generated.write_text(
        """
model:
  pretrained_model_name_or_path: gpt-4.1-mini-2025-04-14
agentic_math:
  openclaw_model: openai-direct/gpt-4.1-mini-2025-04-14
  use_task_agent: false
swebench_pro:
  openclaw_model: openai-direct/gpt-4.1-mini-2025-04-14
""",
        encoding="utf-8",
    )

    checks = module.check_generated_configs(
        base,
        {"agentic": generated},
        expected_pretrained_model="gpt-4.1-mini-2025-04-14",
        openclaw_model="openai-direct/gpt-4.1-mini-2025-04-14",
        require_nemoclaw=True,
        nemoclaw_sandbox="nejumi-taiwan",
    )

    failed = {check.name for check in checks if not check.ok}
    assert "agentic config uses NeMoClaw sandbox" in failed
    assert "agentic config keeps task-agent enabled" in failed
    assert "agentic SWE checkout is sandbox-accessible" in failed
    assert "agentic Math denies remote lookup via deny_tool" in failed
    assert "agentic SWE denies remote lookup via deny_tool" in failed
    assert "agentic Math denies remote lookup via deny_argument_pattern" in failed
    assert "agentic SWE denies remote lookup via deny_argument_pattern" in failed
