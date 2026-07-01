import json
import subprocess
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "setup" / "verify_nemoclaw_post_install.py"


def write_executable(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def write_fake_scripts(tmp_path: Path, *, ok: bool) -> dict[str, Path]:
    py_bool = "True" if ok else "False"
    json_bool = str(ok).lower()
    install = write_executable(
        tmp_path / "install_check.sh",
        f"""#!/usr/bin/env sh
set -eu
out=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --json)
      out="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
mkdir -p "$(dirname "$out")"
cat > "$out" <<'JSON'
{{
  "ok": {json_bool},
  "check_only": true,
  "install_requested": false,
  "onboard_requested": false,
  "host_prerequisites_ok": {json_bool},
  "runtime_installed": {json_bool},
  "sandbox_configured": {json_bool},
  "missing_required_commands": [],
  "commands": {{
    "docker": {{"available": {json_bool}, "info_ok": {json_bool}}},
    "nemoclaw": {{"available": {json_bool}}},
    "openshell": {{"available": {json_bool}}}
  }},
  "setup_plan": {{
    "will_launch_model_inference": false,
    "sandbox_configured": {json_bool},
    "sandbox_readiness_required": true,
    "install_or_onboard_requires_explicit_acceptance": true,
    "acceptance_flag": "--yes-i-accept-third-party-software",
    "install_command": "install",
    "onboard_command": "onboard",
    "post_install_check_command": "check"
  }}
}}
JSON
exit {0 if ok else 1}
""",
    )
    protocol = write_executable(
        tmp_path / "protocol.py",
        f"""#!/usr/bin/env python3
import json
payload = {{
  "ok": {py_bool},
  "node_ok": {py_bool},
  "nemoclaw": {{
    "sandbox": "nejumi-taiwan",
    "installed": {py_bool},
    "sandbox_status_ok": {py_bool},
    "sandbox_openclaw_ok": {py_bool}
  }}
}}
print(json.dumps(payload))
""",
    )
    canary = write_executable(
        tmp_path / "canary.py",
        f"""#!/usr/bin/env python3
import argparse, json
p = argparse.ArgumentParser()
p.add_argument('--json')
p.add_argument('--require-nemoclaw', action='store_true')
p.add_argument('--nemoclaw-bin')
p.add_argument('--nemoclaw-sandbox')
args, _unknown = p.parse_known_args()
policy_detail = {{
  "sandbox": "nejumi-taiwan",
  "sandbox_found": True,
  "policy_count": 7,
  "policies": ["clawhub", "managed_inference", "npm_registry", "nvidia", "openclaw_api", "openclaw_docs", "wandb-weave"],
  "policy_configured": True,
  "summary_policy_count": 0,
  "summary_policies": [],
  "detailed_status_network_policy_count": 7,
  "detailed_status_network_policies": ["clawhub", "managed_inference", "npm_registry", "nvidia", "openclaw_api", "openclaw_docs", "wandb-weave"],
  "allowed_runtime_network_policies": ["clawhub", "managed_inference", "npm_registry", "nvidia", "openclaw_api", "openclaw_docs", "wandb-weave"],
  "runtime_network_policy_allowlist_ok": True,
  "unknown_runtime_network_policies": [],
  "wandb_weave_policy_present": True,
  "non_wandb_network_policies": ["clawhub", "managed_inference", "npm_registry", "nvidia", "openclaw_api", "openclaw_docs"]
}}
payload = {{
  "ok": {py_bool},
  "checks": [
    {{"name": "NeMoClaw command is available", "ok": {py_bool}}},
    {{"name": "OpenShell command is available", "ok": {py_bool}}},
    {{"name": "NeMoClaw version command succeeds", "ok": {py_bool}}},
    {{"name": "NeMoClaw sandbox status succeeds: nejumi-taiwan", "ok": {py_bool}}},
    {{"name": "OpenClaw runs inside NeMoClaw sandbox: nejumi-taiwan", "ok": {py_bool}}},
    {{"name": "NeMoClaw sandbox runtime policy is introspectable: nejumi-taiwan", "ok": {py_bool}, "detail": json.dumps(policy_detail)}},
    {{"name": "NeMoClaw W&B/Weave runtime policy is present: nejumi-taiwan", "ok": {py_bool}, "detail": json.dumps(policy_detail)}},
    {{"name": "NeMoClaw runtime network policies are allowlisted: nejumi-taiwan", "ok": {py_bool}, "detail": json.dumps(policy_detail)}},
    {{"name": "NeMoClaw sandbox OpenClaw config is readable: /sandbox/.openclaw/openclaw.json", "ok": {py_bool}, "detail": "bytes=7465"}},
    {{"name": "NeMoClaw sandbox OpenClaw openai-direct provider exists", "ok": {py_bool}}},
    {{"name": "NeMoClaw sandbox OpenClaw model is registered: openai-direct/gpt-4.1-mini-2025-04-14", "ok": {py_bool}}},
    {{"name": "NeMoClaw sandbox OpenClaw Weave plugin is enabled", "ok": {py_bool}}}
  ]
}}
open(args.json, 'w', encoding='utf-8').write(json.dumps(payload) + '\\n')
raise SystemExit(0 if {py_bool} else 1)
""",
    )
    adoption = write_executable(
        tmp_path / "adoption.py",
        f"""#!/usr/bin/env python3
import argparse, json
p = argparse.ArgumentParser()
p.add_argument('--setup-json')
p.add_argument('--readiness-json')
p.add_argument('--sandbox')
p.add_argument('--json')
p.add_argument('--markdown')
p.add_argument('--fail-on-not-adoptable', action='store_true')
args, _unknown = p.parse_known_args()
payload = {{
  "ok": {py_bool},
  "status": "{'adoptable_for_agentic_math' if ok else 'not_installed'}",
  "summary": {{
    "ready_for_use": {py_bool},
    "blockers": []
  }},
  "adoption_decision": {{
    "ready_for_use": {py_bool},
    "design_ready": {py_bool},
    "scope": "agentic_math_only",
    "blockers": []
  }},
  "criteria": [
    {{"name": "runtime_wandb_weave_policy", "ok": {py_bool}}},
    {{"name": "runtime_network_policy_allowlist", "ok": {py_bool}}}
  ]
}}
open(args.json, 'w', encoding='utf-8').write(json.dumps(payload) + '\\n')
open(args.markdown, 'w', encoding='utf-8').write('# adoption\\n')
raise SystemExit(0 if {py_bool} else 1)
""",
    )
    return {
        "install": install,
        "protocol": protocol,
        "canary": canary,
        "adoption": adoption,
    }


def run_verifier(
    tmp_path: Path,
    *,
    ok: bool,
    scripts: dict[str, Path] | None = None,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    scripts = scripts or write_fake_scripts(tmp_path, ok=ok)
    extra_args = extra_args or []
    return subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--output-dir",
            str(tmp_path / "out"),
            "--timestamp",
            "TEST",
            "--python-command",
            "python3",
            "--install-check-script",
            str(scripts["install"]),
            "--protocol-script",
            str(scripts["protocol"]),
            "--canary-readiness-script",
            str(scripts["canary"]),
            "--adoption-script",
            str(scripts["adoption"]),
            "--json",
            str(tmp_path / "summary.json"),
            "--markdown",
            str(tmp_path / "summary.md"),
            "--fail-on-failed",
            *extra_args,
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_verify_nemoclaw_post_install_passes_with_all_steps_ok(tmp_path):
    result = run_verifier(tmp_path, ok=True)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["path"] == str(tmp_path / "summary.json")
    assert payload["markdown_path"] == str(tmp_path / "summary.md")
    assert payload["ok"] is True
    assert payload["status"] == "passed"
    assert payload["will_launch_model_inference"] is False
    assert payload["will_query_wandb"] is False
    assert payload["will_install_or_onboard"] is False
    assert payload["command_safety"]["ok"] is True
    assert payload["command_safety"]["forbidden_token_count"] == 0
    assert payload["command_safety"]["missing_required_token_count"] == 0
    assert set(payload["outputs_sha256"]) == {
        "setup_json",
        "preflight_json",
        "readiness_json",
        "adoption_json",
        "adoption_markdown",
    }
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", value)
        for value in payload["outputs_sha256"].values()
    )
    assert all(step["payload_contract_ok"] for step in payload["steps"])
    assert all(re.fullmatch(r"[0-9a-f]{64}", step["output_json_sha256"]) for step in payload["steps"])
    assert [step["name"] for step in payload["steps"]] == [
        "setup_check",
        "protocol_preflight",
        "canary_readiness",
        "adoption_check",
    ]
    written_payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert written_payload["ok"] is True
    assert written_payload["path"] == str(tmp_path / "summary.json")
    assert written_payload["markdown_path"] == str(tmp_path / "summary.md")
    assert "protocol_preflight" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_verify_nemoclaw_post_install_accepts_agentic_benchmarks_scope(tmp_path):
    scripts = write_fake_scripts(tmp_path, ok=True)
    scripts["adoption"] = write_executable(
        tmp_path / "adoption_benchmarks.py",
        """#!/usr/bin/env python3
import argparse, json
p = argparse.ArgumentParser()
p.add_argument('--setup-json')
p.add_argument('--readiness-json')
p.add_argument('--sandbox')
p.add_argument('--json')
p.add_argument('--markdown')
p.add_argument('--fail-on-not-adoptable', action='store_true')
args, _unknown = p.parse_known_args()
payload = {
  "ok": True,
  "status": "adoptable_for_agentic_benchmarks",
  "summary": {
    "ready_for_use": True,
    "blockers": []
  },
  "adoption_decision": {
    "ready_for_use": True,
    "design_ready": True,
    "scope": "agentic_math_and_swebench_pro",
    "blockers": []
  },
  "criteria": [
    {"name": "runtime_wandb_weave_policy", "ok": True},
    {"name": "runtime_network_policy_allowlist", "ok": True}
  ]
}
open(args.json, 'w', encoding='utf-8').write(json.dumps(payload) + '\\n')
open(args.markdown, 'w', encoding='utf-8').write('# adoption\\n')
""",
    )

    result = run_verifier(tmp_path, ok=True, scripts=scripts)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    adoption = payload["steps"][-1]
    assert adoption["name"] == "adoption_check"
    assert adoption["payload_status"] == "adoptable_for_agentic_benchmarks"
    assert adoption["payload_contract_ok"] is True


def test_verify_nemoclaw_post_install_defaults_to_openai_nemoclaw_canary(tmp_path):
    result = run_verifier(tmp_path, ok=True)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    readiness_command = payload["steps"][2]["command"]
    assert "--manifest" in readiness_command
    assert str((REPO_ROOT / "configs/taiwan_openai_canary_models.yaml").resolve()) in readiness_command
    assert "--canary-slug" in readiness_command
    assert "gpt-4_1-mini-openai-direct-canary" in readiness_command
    assert "--openclaw-model" in readiness_command
    assert "openai-direct/gpt-4.1-mini-2025-04-14" in readiness_command
    assert "--nemoclaw-openclaw-config-path" in readiness_command
    assert "/sandbox/.openclaw/openclaw.json" in readiness_command
    assert "--generated-full-dir" in readiness_command
    assert str((REPO_ROOT / "configs/taiwan_full/generated_openai_canary").resolve()) in readiness_command
    assert "--generated-nonagentic-dir" in readiness_command
    assert (
        str((REPO_ROOT / "configs/taiwan_full/generated_openai_canary_nonagentic").resolve())
        in readiness_command
    )
    assert "--generated-agentic-dir" in readiness_command
    assert (
        str((REPO_ROOT / "configs/taiwan_full/generated_openai_canary_agentic_nemoclaw").resolve())
        in readiness_command
    )
    assert "--generated-agentic-aggregate-dir" in readiness_command
    assert (
        str((REPO_ROOT / "configs/taiwan_full/generated_openai_canary_agentic_aggregate").resolve())
        in readiness_command
    )


def test_verify_nemoclaw_post_install_fails_when_any_step_fails(tmp_path):
    result = run_verifier(tmp_path, ok=False)

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["status"] == "failed"
    assert any(not step["ok"] for step in payload["steps"])


def test_verify_nemoclaw_post_install_rejects_shallow_ok_payload(tmp_path):
    scripts = write_fake_scripts(tmp_path, ok=True)
    scripts["canary"] = write_executable(
        tmp_path / "canary_shallow.py",
        """#!/usr/bin/env python3
import argparse, json
p = argparse.ArgumentParser()
p.add_argument('--json')
args, _unknown = p.parse_known_args()
open(args.json, 'w', encoding='utf-8').write(json.dumps({"ok": True}) + '\\n')
raise SystemExit(0)
""",
    )

    result = run_verifier(tmp_path, ok=True, scripts=scripts)

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    canary = next(step for step in payload["steps"] if step["name"] == "canary_readiness")
    assert canary["payload_ok"] is True
    assert canary["payload_contract_ok"] is False
    assert "checks must be a list" in canary["payload_contract_errors"]


def test_verify_nemoclaw_post_install_rejects_missing_sandbox_openai_config_proof(
    tmp_path,
):
    scripts = write_fake_scripts(tmp_path, ok=True)
    scripts["canary"] = write_executable(
        tmp_path / "canary_missing_sandbox_openai_config.py",
        """#!/usr/bin/env python3
import argparse, json
p = argparse.ArgumentParser()
p.add_argument('--json')
args, _unknown = p.parse_known_args()
policy_detail = {
  "sandbox": "nejumi-taiwan",
  "sandbox_found": True,
  "policy_count": 7,
  "policies": ["clawhub", "managed_inference", "npm_registry", "nvidia", "openclaw_api", "openclaw_docs", "wandb-weave"],
  "policy_configured": True,
  "summary_policy_count": 0,
  "summary_policies": [],
  "detailed_status_network_policy_count": 7,
  "detailed_status_network_policies": ["clawhub", "managed_inference", "npm_registry", "nvidia", "openclaw_api", "openclaw_docs", "wandb-weave"],
  "allowed_runtime_network_policies": ["clawhub", "managed_inference", "npm_registry", "nvidia", "openclaw_api", "openclaw_docs", "wandb-weave"],
  "runtime_network_policy_allowlist_ok": True,
  "unknown_runtime_network_policies": [],
  "wandb_weave_policy_present": True,
  "non_wandb_network_policies": ["clawhub", "managed_inference", "npm_registry", "nvidia", "openclaw_api", "openclaw_docs"]
}
payload = {
  "ok": True,
  "checks": [
    {"name": "NeMoClaw command is available", "ok": True},
    {"name": "OpenShell command is available", "ok": True},
    {"name": "NeMoClaw version command succeeds", "ok": True},
    {"name": "NeMoClaw sandbox status succeeds: nejumi-taiwan", "ok": True},
    {"name": "OpenClaw runs inside NeMoClaw sandbox: nejumi-taiwan", "ok": True},
    {"name": "NeMoClaw sandbox runtime policy is introspectable: nejumi-taiwan", "ok": True, "detail": json.dumps(policy_detail)},
    {"name": "NeMoClaw W&B/Weave runtime policy is present: nejumi-taiwan", "ok": True, "detail": json.dumps(policy_detail)}
  ]
}
open(args.json, 'w', encoding='utf-8').write(json.dumps(payload) + '\\n')
raise SystemExit(0)
""",
    )

    result = run_verifier(tmp_path, ok=True, scripts=scripts)

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    canary = next(step for step in payload["steps"] if step["name"] == "canary_readiness")
    assert canary["payload_ok"] is True
    assert canary["payload_contract_ok"] is False
    assert (
        "missing required check 'NeMoClaw sandbox OpenClaw openai-direct provider exists'"
        in canary["payload_contract_errors"]
    )


def test_verify_nemoclaw_post_install_rejects_runtime_without_sandbox(tmp_path):
    scripts = write_fake_scripts(tmp_path, ok=True)
    scripts["install"] = write_executable(
        tmp_path / "install_check_runtime_only.sh",
        """#!/usr/bin/env sh
set -eu
out=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --json)
      out="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
mkdir -p "$(dirname "$out")"
cat > "$out" <<'JSON'
{
  "ok": true,
  "check_only": true,
  "install_requested": false,
  "onboard_requested": false,
  "host_prerequisites_ok": true,
  "runtime_installed": true,
  "sandbox_configured": false,
  "missing_required_commands": [],
  "commands": {
    "docker": {"available": true, "info_ok": true},
    "nemoclaw": {"available": true},
    "openshell": {"available": true}
  },
  "setup_plan": {
    "will_launch_model_inference": false,
    "sandbox_configured": false,
    "sandbox_readiness_required": true,
    "install_or_onboard_requires_explicit_acceptance": true
  }
}
JSON
exit 0
""",
    )

    result = run_verifier(tmp_path, ok=True, scripts=scripts)

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    setup = next(step for step in payload["steps"] if step["name"] == "setup_check")
    assert setup["returncode_ok"] is True
    assert setup["payload_ok"] is True
    assert setup["payload_contract_ok"] is False
    assert "sandbox_configured must be true" in setup["payload_contract_errors"]
    assert "setup_plan.sandbox_configured must be true" in setup["payload_contract_errors"]


def test_verify_nemoclaw_post_install_forwards_canary_and_adoption_args(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("models: []\n", encoding="utf-8")
    generated_agentic = tmp_path / "generated_agentic"
    generated_agentic.mkdir()

    result = run_verifier(
        tmp_path,
        ok=True,
        extra_args=[
            "--canary-manifest",
            str(manifest),
            "--canary-slug",
            "gpt-4_1-mini-openai-direct-canary",
            "--canary-openclaw-model",
            "openai-direct/gpt-4.1-mini-2025-04-14",
            "--generated-agentic-dir",
            str(generated_agentic),
            "--adoption-agentic-config-glob",
            "configs/taiwan_full/generated_openai_canary_agentic_nemoclaw/*.yaml",
        ],
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    readiness_command = payload["steps"][2]["command"]
    adoption_command = payload["steps"][3]["command"]
    assert "--manifest" in readiness_command
    assert str(manifest.resolve()) in readiness_command
    assert "--canary-slug" in readiness_command
    assert "gpt-4_1-mini-openai-direct-canary" in readiness_command
    assert "--openclaw-model" in readiness_command
    assert "openai-direct/gpt-4.1-mini-2025-04-14" in readiness_command
    assert "--generated-agentic-dir" in readiness_command
    assert str(generated_agentic.resolve()) in readiness_command
    assert "--agentic-config-glob" in adoption_command
    assert "configs/taiwan_full/generated_openai_canary_agentic_nemoclaw/*.yaml" in adoption_command


def test_verify_nemoclaw_post_install_requires_zero_returncode_even_when_json_ok(tmp_path):
    scripts = write_fake_scripts(tmp_path, ok=True)
    scripts["install"] = write_executable(
        tmp_path / "install_check_nonzero.sh",
        """#!/usr/bin/env sh
set -eu
out=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --json)
      out="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
mkdir -p "$(dirname "$out")"
cat > "$out" <<'JSON'
{"ok": true, "status": "looks_ok_but_process_failed"}
JSON
exit 7
""",
    )

    result = run_verifier(tmp_path, ok=True, scripts=scripts)

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    setup = payload["steps"][0]
    assert setup["name"] == "setup_check"
    assert setup["returncode"] == 7
    assert setup["payload_ok"] is True
    assert setup["returncode_ok"] is False
    assert setup["ok"] is False


def test_verify_nemoclaw_post_install_times_out_individual_steps(tmp_path):
    scripts = write_fake_scripts(tmp_path, ok=True)
    scripts["protocol"] = write_executable(
        tmp_path / "protocol_sleep.py",
        """#!/usr/bin/env python3
import time
time.sleep(10)
""",
    )

    result = run_verifier(
        tmp_path,
        ok=True,
        scripts=scripts,
        extra_args=["--step-timeout-seconds", "0.01"],
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    preflight = payload["steps"][1]
    assert preflight["name"] == "protocol_preflight"
    assert preflight["returncode"] == 124
    assert preflight["returncode_ok"] is False
    assert preflight["timed_out"] is True
    assert preflight["ok"] is False


def test_verify_nemoclaw_post_install_marks_forbidden_command_tokens_unsafe(tmp_path):
    result = run_verifier(
        tmp_path,
        ok=True,
        extra_args=["--python-command", "python3 --wandb"],
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["command_safety"]["ok"] is False
    assert payload["command_safety"]["forbidden_token_count"] >= 1
    assert any(
        "--wandb" in record["forbidden_tokens"]
        for record in payload["command_safety"]["records"]
    )


def test_verify_nemoclaw_post_install_marks_wandb_prefix_and_api_key_tokens_unsafe(tmp_path):
    result = run_verifier(
        tmp_path,
        ok=True,
        extra_args=["--python-command", "python3 --wandb-project=taiwan WANDB_API_KEY=test"],
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["command_safety"]["ok"] is False
    flattened = [
        token
        for record in payload["command_safety"]["records"]
        for token in record["forbidden_tokens"]
    ]
    assert "--wandb-project=taiwan" in flattened
    assert "WANDB_API_KEY=test" in flattened
