import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tools" / "run_taiwan_release_gate.py"


def write_fake_nemoclaw_check(path: Path) -> Path:
    path.write_text(
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
  "schema_version": 1,
  "ok": false,
  "commands": {
    "nemoclaw": {"available": false},
    "openshell": {"available": false}
  },
  "setup_plan": {
    "will_launch_model_inference": false
  }
}
JSON
exit 1
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def write_fake_post_install(path: Path) -> Path:
    path.write_text(
        r'''#!/usr/bin/env python3
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


args = sys.argv[1:]
output_dir = None
timestamp = None
json_path = None
markdown_path = None
index = 0
while index < len(args):
    arg = args[index]
    if arg == "--output-dir":
        output_dir = Path(args[index + 1])
        index += 2
    elif arg == "--timestamp":
        timestamp = args[index + 1]
        index += 2
    elif arg == "--json":
        json_path = Path(args[index + 1])
        index += 2
    elif arg == "--markdown":
        markdown_path = Path(args[index + 1])
        index += 2
    else:
        index += 1

assert output_dir is not None
assert timestamp is not None
assert json_path is not None
assert markdown_path is not None
output_dir.mkdir(parents=True, exist_ok=True)

setup_json = output_dir / f"fake_post_install_setup_{timestamp}.json"
preflight_json = output_dir / f"fake_post_install_preflight_{timestamp}.json"
readiness_json = output_dir / f"fake_post_install_readiness_{timestamp}.json"
adoption_json = output_dir / f"fake_post_install_adoption_{timestamp}.json"
adoption_markdown = output_dir / f"fake_post_install_adoption_{timestamp}.md"
policy_detail = {
    "sandbox": "nejumi-taiwan",
    "sandbox_found": True,
    "policy_count": 7,
    "policies": [
        "clawhub",
        "managed_inference",
        "npm_registry",
        "nvidia",
        "openclaw_api",
        "openclaw_docs",
        "wandb-weave",
    ],
    "policy_configured": True,
    "summary_policy_count": 0,
    "summary_policies": [],
    "detailed_status_network_policy_count": 7,
    "detailed_status_network_policies": [
        "clawhub",
        "managed_inference",
        "npm_registry",
        "nvidia",
        "openclaw_api",
        "openclaw_docs",
        "wandb-weave",
    ],
    "allowed_runtime_network_policies": [
        "clawhub",
        "managed_inference",
        "npm_registry",
        "nvidia",
        "openclaw_api",
        "openclaw_docs",
        "wandb-weave",
    ],
    "runtime_network_policy_allowlist_ok": True,
    "unknown_runtime_network_policies": [],
    "wandb_weave_policy_present": True,
    "non_wandb_network_policies": [
        "clawhub",
        "managed_inference",
        "npm_registry",
        "nvidia",
        "openclaw_api",
        "openclaw_docs",
    ],
}

payloads = {
    setup_json: {"ok": True, "status": "passed"},
    preflight_json: {"ok": True, "status": "passed"},
    readiness_json: {
        "ok": True,
        "status": "passed",
        "checks": [
            {"name": "manifest has exactly one canary", "ok": True},
            {"name": "NeMoClaw command is available", "ok": True},
            {"name": "OpenShell command is available", "ok": True},
            {"name": "NeMoClaw version command succeeds", "ok": True},
            {"name": "NeMoClaw sandbox status succeeds: nejumi-taiwan", "ok": True},
            {"name": "OpenClaw runs inside NeMoClaw sandbox: nejumi-taiwan", "ok": True},
            {
                "name": "NeMoClaw sandbox runtime policy is introspectable: nejumi-taiwan",
                "ok": True,
                "detail": json.dumps(policy_detail),
            },
            {
                "name": "NeMoClaw W&B/Weave runtime policy is present: nejumi-taiwan",
                "ok": True,
                "detail": json.dumps(policy_detail),
            },
            {
                "name": "NeMoClaw runtime network policies are allowlisted: nejumi-taiwan",
                "ok": True,
                "detail": json.dumps(policy_detail),
            },
            {
                "name": "agentic Math denies remote lookup via deny_tool",
                "ok": True,
                "detail": json.dumps(
                    {
                        "section": "agentic_math",
                        "missing": [],
                        "observed": [
                            "*search*",
                            "browser",
                            "browser_*",
                            "code_execution",
                            "web_fetch",
                            "web_search",
                        ],
                    }
                ),
            },
            {
                "name": "agentic SWE denies remote lookup via deny_tool",
                "ok": True,
                "detail": json.dumps(
                    {
                        "section": "swebench_pro",
                        "missing": [],
                        "observed": [
                            "*search*",
                            "browser",
                            "browser_*",
                            "code_execution",
                            "web_fetch",
                            "web_search",
                        ],
                    }
                ),
            },
            {
                "name": "agentic Math denies remote lookup via deny_argument_pattern",
                "ok": True,
                "detail": json.dumps(
                    {
                        "section": "agentic_math",
                        "missing": [],
                        "observed": [
                            r"\b(curl|wget)\b",
                            r"\b(requests|urllib|httpx)\.",
                            "https?://",
                        ],
                    }
                ),
            },
            {
                "name": "agentic SWE denies remote lookup via deny_argument_pattern",
                "ok": True,
                "detail": json.dumps(
                    {
                        "section": "swebench_pro",
                        "missing": [],
                        "observed": [
                            r"\b(curl|wget)\b",
                            r"\b(requests|urllib|httpx)\.",
                            "https?://",
                        ],
                    }
                ),
            },
        ],
    },
    adoption_json: {
        "ok": True,
        "status": "adoptable_for_agentic_benchmarks",
        "path": str(adoption_json),
        "markdown_path": str(adoption_markdown),
        "criteria": [
            {"name": "runtime_wandb_weave_policy", "ok": True, "wandb_weave_policy_present": True},
            {
                "name": "runtime_network_policy_allowlist",
                "ok": True,
                "runtime_network_policy_allowlist_ok": True,
                "unknown_runtime_network_policies": [],
                "allowed_runtime_network_policies": [
                    "clawhub",
                    "managed_inference",
                    "npm_registry",
                    "nvidia",
                    "openclaw_api",
                    "openclaw_docs",
                    "wandb-weave",
                ],
                "detailed_status_network_policy_count": 7,
                "detailed_status_network_policies": [
                    "clawhub",
                    "managed_inference",
                    "npm_registry",
                    "nvidia",
                    "openclaw_api",
                    "openclaw_docs",
                    "wandb-weave",
                ],
                "non_wandb_network_policies": [
                    "clawhub",
                    "managed_inference",
                    "npm_registry",
                    "nvidia",
                    "openclaw_api",
                    "openclaw_docs",
                ],
            },
        ],
    },
}
for file_path, payload in payloads.items():
    file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
adoption_markdown.write_text("# adoption passed\n", encoding="utf-8")

outputs = {
    "setup_json": str(setup_json),
    "preflight_json": str(preflight_json),
    "readiness_json": str(readiness_json),
    "adoption_json": str(adoption_json),
    "adoption_markdown": str(adoption_markdown),
}
outputs_sha256 = {name: sha256(Path(value)) for name, value in outputs.items()}
steps = [
    (
        "setup_check",
        "setup_json",
        ["scripts/setup/install_nemoclaw.sh", "--check-only", "--json", str(setup_json)],
    ),
    (
        "protocol_preflight",
        "preflight_json",
        ["uv", "run", "python", "scripts/tools/run_openclaw_agent_protocol.py", "preflight"],
    ),
    (
        "canary_readiness",
        "readiness_json",
        ["uv", "run", "python", "scripts/tools/check_taiwan_canary_readiness.py", "--require-nemoclaw", "--json", str(readiness_json)],
    ),
    (
        "adoption_check",
        "adoption_json",
        [
            "uv",
            "run",
            "python",
            "scripts/tools/check_taiwan_nemoclaw_adoption.py",
            "--setup-json",
            str(setup_json),
            "--readiness-json",
            str(readiness_json),
            "--json",
            str(adoption_json),
            "--markdown",
            str(adoption_markdown),
        ],
    ),
]
post_install = {
    "schema_version": 1,
    "ok": True,
    "status": "passed",
    "will_launch_model_inference": False,
    "will_query_wandb": False,
    "will_install_or_onboard": False,
    "command_safety": {
        "ok": True,
        "forbidden_tokens": ["--install", "--onboard", "--upload", "--wandb", "--yes-i-accept-third-party-software"],
        "forbidden_prefixes": ["--upload", "--wandb", "ANTHROPIC_API_KEY=", "GEMINI_API_KEY=", "GOOGLE_API_KEY=", "OPENAI_API_KEY=", "OPENROUTER_", "OPENROUTER_API_KEY=", "WANDB_", "WEAVE_", "XAI_API_KEY="],
        "forbidden_markers": ["openrouter", "wandb", "weave"],
        "required_step_tokens": {
            "setup_check": ["--check-only", "--json"],
            "protocol_preflight": ["preflight"],
            "canary_readiness": ["--require-nemoclaw", "--json"],
            "adoption_check": ["--setup-json", "--readiness-json", "--json", "--markdown"],
        },
        "forbidden_token_count": 0,
        "missing_required_token_count": 0,
        "missing_command_count": 0,
        "records": [
            {"name": name, "ok": True, "forbidden_tokens": [], "missing_required_tokens": []}
            for name, _, _ in steps
        ],
    },
    "outputs": outputs,
    "outputs_sha256": outputs_sha256,
    "steps": [
        {
            "name": name,
            "ok": True,
            "returncode": 0,
            "returncode_ok": True,
            "payload_ok": True,
            "payload_contract_ok": True,
            "payload_contract_errors": [],
            "timed_out": False,
            "command": command,
            "output_json": outputs[output_name],
            "output_json_sha256": outputs_sha256[output_name],
            "payload_status": payloads[Path(outputs[output_name])]["status"],
        }
        for name, output_name, command in steps
    ],
    "path": str(json_path),
    "markdown_path": str(markdown_path),
}
json_path.parent.mkdir(parents=True, exist_ok=True)
json_path.write_text(json.dumps(post_install, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
markdown_path.write_text("# post install passed\n\nprotocol_preflight\ncanary_readiness\nadoption_check\n", encoding="utf-8")
''',
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def test_release_gate_writes_report_bundle_and_verification(tmp_path):
    fake_check = write_fake_nemoclaw_check(tmp_path / "fake_install_nemoclaw.sh")
    fake_post_install = write_fake_post_install(tmp_path / "fake_post_install.py")
    output_dir = tmp_path / "out"
    report_json = output_dir / "readiness.json"
    bundle_dir = tmp_path / "bundle"
    verify_json = output_dir / "bundle_verify.json"
    release_json = output_dir / "release_gate.json"
    latest_pointer_json = output_dir / "latest_taiwan_release_gate.json"
    operator_json = output_dir / "taiwan_release_operator_plan_RELEASE.json"
    operator_md = output_dir / "taiwan_release_operator_plan_RELEASE.md"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--output-dir",
            str(output_dir),
            "--timestamp",
            "RELEASE",
            "--readiness-report-json",
            str(report_json),
            "--bundle-output-dir",
            str(bundle_dir),
            "--bundle-verification-json",
            str(verify_json),
            "--release-gate-json",
            str(release_json),
            "--quiet",
            "--nemoclaw-check-script",
            str(fake_check),
            "--nemoclaw-post-install-verify-script",
            str(fake_post_install),
            "--no-require-metadata-readiness",
            "--no-require-weave-content-canary",
            "--no-require-nemoclaw",
            "--no-require-wandb-completion",
            "--no-require-one-model-canary",
            "--no-require-paid-run-review-package",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert report_json.exists()
    assert (bundle_dir / "manifest.json").exists()
    assert (bundle_dir / "summary.md").exists()
    assert verify_json.exists()
    assert release_json.exists()
    assert latest_pointer_json.exists()
    assert operator_json.exists()
    assert operator_md.exists()
    release = json.loads(release_json.read_text(encoding="utf-8"))
    verification = json.loads(verify_json.read_text(encoding="utf-8"))
    latest_pointer = json.loads(latest_pointer_json.read_text(encoding="utf-8"))
    operator_plan = json.loads(operator_json.read_text(encoding="utf-8"))
    assert release["schema_version"] == 1
    assert release["readiness_report_schema_version"] == 1
    assert release["bundle_integrity_ok"] is True
    assert release["ok"] is release["release_ready"]
    assert release["bundle_verification_ok"] is True
    assert release["readiness_status"] in {"ready", "not_ready"}
    assert isinstance(release["checked_file_count"], int)
    assert release["checked_file_count"] == verification["checked_file_count"]
    assert release["bundle_file_count"] == release["bundle"]["file_count"]
    assert release["bundle_missing_file_count"] == 0
    assert release["verification_error_count"] == len(verification["errors"])
    assert release["blocking_gates"] == release["blockers"]
    assert release["latest_pointer_json"] == str(latest_pointer_json)
    assert latest_pointer["schema_version"] == 1
    assert latest_pointer["readiness_report_schema_version"] == 1
    assert latest_pointer["timestamp"] == "RELEASE"
    assert latest_pointer["latest_pointer_json"] == str(latest_pointer_json)
    assert latest_pointer["release_gate_json"] == str(release_json)
    assert latest_pointer["bundle_manifest"] == str(bundle_dir / "manifest.json")
    assert latest_pointer["status"] == release["status"]
    assert latest_pointer["readiness_report_source"] == release["readiness_report"]
    assert latest_pointer["readiness_ok"] == release["readiness_ok"]
    assert latest_pointer["gate_count"] == release["gate_count"]
    assert latest_pointer["blocker_count"] == release["blocker_count"]
    assert latest_pointer["blocking_gates"] == release["blocking_gates"]
    assert latest_pointer["blockers"] == release["blockers"]
    assert latest_pointer["required_next_actions"] == release["required_next_actions"]
    assert latest_pointer["benchmark_completion"] == release["benchmark_completion"]
    assert latest_pointer["benchmark_progress_matrix"] == release["benchmark_progress_matrix"]
    assert latest_pointer["weave_agents_completion"] == release["weave_agents_completion"]
    assert (
        latest_pointer["existing_results_formalization"]
        == release["existing_results_formalization"]
    )
    assert latest_pointer["paid_run_review_package"] == release["paid_run_review_package"]
    assert latest_pointer["wandb_completion_contract"] == release["wandb_completion_contract"]
    assert latest_pointer["nemoclaw_adoption"] == release["nemoclaw_adoption"]
    assert latest_pointer["operator_next_steps"] == release["operator_next_steps"]
    assert latest_pointer["operator_plan"] == release["operator_plan"]
    assert latest_pointer["external_action_checklist"] == release["external_action_checklist"]
    assert (
        latest_pointer["external_action_approval_packet"]
        == release["external_action_approval_packet"]
    )
    assert latest_pointer["bundle_file_count"] == release["bundle_file_count"]
    assert latest_pointer["bundle_missing_file_count"] == release["bundle_missing_file_count"]
    assert latest_pointer["checked_file_count"] == verification["checked_file_count"]
    assert isinstance(release["required_next_actions"], list)
    assert isinstance(release["remediation_plan"], list)
    assert isinstance(release["benchmark_completion"], list)
    assert isinstance(release["benchmark_progress_matrix"], list)
    assert isinstance(release["weave_agents_completion"], list)
    assert isinstance(release["existing_results_formalization"], dict)
    assert isinstance(release["paid_run_review_package"], dict)
    assert isinstance(release["wandb_completion_contract"], dict)
    assert isinstance(release["nemoclaw_adoption"], dict)
    assert isinstance(release["operator_next_steps"], dict)
    assert isinstance(release["external_action_checklist"], dict)
    assert isinstance(release["external_action_approval_packet"], dict)
    assert release["external_action_checklist"]["item_count"] == (
        release["operator_next_steps"]["step_count"]
    )
    assert release["external_action_approval_packet"]["json"] == (
        "external_action_approval_packet.json"
    )
    assert release["external_action_approval_packet"]["markdown"] == (
        "external_action_approval_packet.md"
    )
    assert release["external_action_approval_packet"]["status"] in {
        "pending_approval",
        "no_external_action_required",
    }
    assert isinstance(
        release["external_action_approval_packet"]["external_action_checklist_sha256"],
        str,
    )
    assert len(
        release["external_action_approval_packet"]["external_action_checklist_sha256"]
    ) == 64
    assert release["operator_plan"]["json"] == str(operator_json)
    assert release["operator_plan"]["markdown"] == str(operator_md)
    assert release["operator_plan"]["schema_version"] == 1
    assert operator_plan["schema_version"] == 1
    assert operator_plan["status"] == release["operator_next_steps"]["status"]
    assert operator_plan["operator_next_steps"] == release["operator_next_steps"]
    renderer = operator_plan["operator_execution_plan_renderer"]
    assert renderer["schema_version"] == 1
    assert renderer["script"] == "scripts/tools/render_taiwan_operator_execution_plan.py"
    assert renderer["required_before_external_action"] is True
    assert "--operator-plan-json" in renderer["review_command_template"]
    assert str(operator_json) in renderer["review_command_template"]
    assert renderer["approval_source_packet_json_template"].endswith(
        "external_action_approval_packet.json"
    )
    assert "--external-action-approval-source-packet-json" in renderer[
        "require_ready_command_template"
    ]
    assert renderer["approval_source_packet_json_template"] in renderer[
        "require_ready_command_template"
    ]
    assert "--require-ready" in renderer["require_ready_command_template"]
    assert operator_plan["benchmark_progress_matrix"] == release["benchmark_progress_matrix"]
    assert operator_plan["external_action_checklist"]["schema_version"] == 1
    assert operator_plan["external_action_checklist"]["item_count"] == (
        release["operator_next_steps"]["step_count"]
    )
    assert operator_plan["outputs"]["release_gate_json"] == str(release_json)
    assert operator_plan["outputs"]["latest_pointer_json"] == str(latest_pointer_json)
    assert operator_plan["outputs"]["latest_pointer_verification_json"] is None
    operator_markdown = operator_md.read_text(encoding="utf-8")
    assert "# Taiwan Release Operator Plan" in operator_markdown
    assert "## Requirement Counts" in operator_markdown
    assert "## Operator Execution Plan Renderer" in operator_markdown
    assert "render_taiwan_operator_execution_plan.py" in operator_markdown
    assert "Placeholder-ready shell" in operator_markdown
    assert "Command-template steps" in operator_markdown
    if release["operator_next_steps"].get("unresolved_placeholder_tokens"):
        assert "## Unresolved Placeholders" in operator_markdown
        assert "Commands: `" in operator_markdown
        assert "Evidence paths: `" in operator_markdown
        assert "Template commands:" in operator_markdown
        assert "Ready without placeholder edit:" in operator_markdown
    assert "## Benchmark Progress Matrix" in operator_markdown
    assert "## Steps" in operator_markdown
    assert "latest_pointer_json" in operator_markdown
    assert release["bundle"]["manifest"] == str(bundle_dir / "manifest.json")
    assert release["bundle"]["operator_plan"] == {
        "json": "operator_plan.json",
        "markdown": "operator_plan.md",
        "schema_version": 1,
        "status": release["operator_next_steps"]["status"],
    }
    assert release["bundle"]["external_action_approval_packet"] == (
        release["external_action_approval_packet"]
    )
    assert verification["integrity_ok"] is True
    assert verification["manifest"] == str(bundle_dir / "manifest.json")
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert (
        manifest["current_gate"]["benchmark_progress_matrix"]
        == release["benchmark_progress_matrix"]
    )
    summary_markdown = (bundle_dir / "summary.md").read_text(encoding="utf-8")
    assert "## Benchmark Progress Matrix" in summary_markdown
    roles = {
        role
        for record in manifest["files"]
        for role in record.get("roles", [])
    }
    assert "existing_results_audit" in roles
    assert "existing_results_audit_markdown" in roles
    assert "paid_run_review_check" in roles
    assert "paid_run_review_check_markdown" in roles
    assert "nemoclaw_adoption_check" in roles
    assert "nemoclaw_adoption_check_markdown" in roles


def test_release_gate_defaults_release_gate_json_and_latest_pointer(tmp_path):
    output_dir = tmp_path / "out"
    fake_post_install = write_fake_post_install(tmp_path / "fake_post_install.py")
    timestamp = "20260628T040000Z"
    release_json = output_dir / f"taiwan_release_gate_{timestamp}.json"
    latest_pointer_json = output_dir / "latest_taiwan_release_gate.json"
    latest_pointer_verification_json = (
        output_dir / f"latest_taiwan_release_gate_verify_{timestamp}.json"
    )

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--output-dir",
            str(output_dir),
            "--timestamp",
            timestamp,
            "--skip-nemoclaw-check",
            "--nemoclaw-post-install-verify-script",
            str(fake_post_install),
            "--quiet",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert release_json.exists()
    assert latest_pointer_json.exists()
    assert latest_pointer_verification_json.exists()

    release = json.loads(release_json.read_text(encoding="utf-8"))
    latest_pointer = json.loads(latest_pointer_json.read_text(encoding="utf-8"))
    pointer_verification = json.loads(
        latest_pointer_verification_json.read_text(encoding="utf-8")
    )

    assert release["release_gate_json"] == str(release_json)
    assert latest_pointer["release_gate_json"] == str(release_json)
    assert latest_pointer["latest_pointer_json"] == str(latest_pointer_json)
    assert latest_pointer["latest_pointer_verification_json"] == str(
        latest_pointer_verification_json
    )
    assert latest_pointer["latest_pointer_verification_ok"] is True
    assert latest_pointer["latest_pointer_verification_status"] == "passed"
    assert latest_pointer["latest_pointer_verification_issue_count"] == 0
    assert latest_pointer["timestamp"] == timestamp
    assert latest_pointer["bundle_manifest"] == release["bundle"]["manifest"]
    assert latest_pointer["operator_plan"] == release["operator_plan"]
    assert pointer_verification["ok"] is True
    assert pointer_verification["schema_version"] == 1
    assert pointer_verification["release_gate_json"] == str(release_json)
    assert release["latest_pointer_verification_ok"] is True
    assert release["latest_pointer_verification_json"] == str(
        latest_pointer_verification_json
    )


def test_release_gate_does_not_regress_latest_pointer_to_older_formal_timestamp(tmp_path):
    output_dir = tmp_path / "out"
    fake_post_install = write_fake_post_install(tmp_path / "fake_post_install.py")
    newer_timestamp = "20260628T050000Z"
    older_timestamp = "20260628T040000Z"
    newer_release_json = output_dir / f"taiwan_release_gate_{newer_timestamp}.json"
    older_release_json = output_dir / f"taiwan_release_gate_{older_timestamp}.json"
    latest_pointer_json = output_dir / "latest_taiwan_release_gate.json"
    older_pointer_verification_json = (
        output_dir / f"latest_taiwan_release_gate_verify_{older_timestamp}.json"
    )

    newer = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--output-dir",
            str(output_dir),
            "--timestamp",
            newer_timestamp,
            "--skip-nemoclaw-check",
            "--nemoclaw-post-install-verify-script",
            str(fake_post_install),
            "--quiet",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert newer.returncode == 0, newer.stderr
    assert newer_release_json.exists()
    latest_after_newer = json.loads(latest_pointer_json.read_text(encoding="utf-8"))
    assert latest_after_newer["timestamp"] == newer_timestamp
    assert latest_after_newer["release_gate_json"] == str(newer_release_json)

    older = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--output-dir",
            str(output_dir),
            "--timestamp",
            older_timestamp,
            "--skip-nemoclaw-check",
            "--nemoclaw-post-install-verify-script",
            str(fake_post_install),
            "--quiet",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert older.returncode == 0, older.stderr
    assert older_release_json.exists()
    assert not older_pointer_verification_json.exists()

    latest_after_older = json.loads(latest_pointer_json.read_text(encoding="utf-8"))
    older_release = json.loads(older_release_json.read_text(encoding="utf-8"))
    assert latest_after_older["timestamp"] == newer_timestamp
    assert latest_after_older["release_gate_json"] == str(newer_release_json)
    assert older_release["timestamp"] == older_timestamp
    assert older_release["latest_pointer_json"] == str(latest_pointer_json)
    assert older_release["latest_pointer_update"] == {
        "status": "skipped_older_timestamp",
        "existing_timestamp": newer_timestamp,
        "new_timestamp": older_timestamp,
        "latest_pointer_json": str(latest_pointer_json),
        "reason": "existing latest pointer has a newer formal release timestamp",
    }
    assert "latest_pointer_verification_json" not in older_release


def test_release_gate_require_ready_fails_for_not_ready_bundle(tmp_path):
    output_dir = tmp_path / "out"
    fake_post_install = write_fake_post_install(tmp_path / "fake_post_install.py")
    release_json = output_dir / "release_gate.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--output-dir",
            str(output_dir),
            "--timestamp",
            "REQUIRE_READY",
            "--release-gate-json",
            str(release_json),
            "--skip-nemoclaw-check",
            "--nemoclaw-post-install-verify-script",
            str(fake_post_install),
            "--require-ready",
            "--quiet",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    release = json.loads(release_json.read_text(encoding="utf-8"))
    assert release["ok"] is False
    assert release["status"] == "not_ready"
    assert release["release_ready"] is False
    assert release["readiness_status"] == "not_ready"
    assert release["blocker_count"] == len(release["blocking_gates"])
    assert release["blockers"] == release["blocking_gates"]
    assert isinstance(release["checked_file_count"], int)
    assert release["required_next_actions"]
    assert release["remediation_plan"]
    assert any(
        row["benchmark"] == "agentic_math"
        for row in release["benchmark_completion"]
    )
    assert isinstance(release["weave_agents_completion"], list)
    assert isinstance(release["existing_results_formalization"], dict)
    assert isinstance(release["paid_run_review_package"], dict)
    assert isinstance(release["wandb_completion_contract"], dict)
    assert isinstance(release["nemoclaw_adoption"], dict)
    assert isinstance(release["operator_next_steps"], dict)
    assert release["bundle_integrity_ok"] is True
    assert release["bundle_verification"]["readiness_ok"] is False
    assert "readiness_ok is false" in release["bundle_verification"]["errors"]


def test_release_gate_prints_operator_next_steps_summary(tmp_path):
    output_dir = tmp_path / "out"
    fake_post_install = write_fake_post_install(tmp_path / "fake_post_install.py")
    release_json = output_dir / "release_gate.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--output-dir",
            str(output_dir),
            "--timestamp",
            "OPERATOR_SUMMARY",
            "--release-gate-json",
            str(release_json),
            "--skip-nemoclaw-check",
            "--nemoclaw-post-install-verify-script",
            str(fake_post_install),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Operator next steps:" in result.stderr
    assert "Operator requirements:" in result.stderr
    assert "Operator command templates:" in result.stderr
    assert "Operator unresolved placeholders:" in result.stderr
    assert "Operator gates:" in result.stderr
    release = json.loads(release_json.read_text(encoding="utf-8"))
    assert release["operator_next_steps"]["status"] == "pending"
    assert release["operator_next_steps"]["step_count"] > 0


def test_release_gate_can_skip_latest_pointer(tmp_path):
    output_dir = tmp_path / "out"
    fake_post_install = write_fake_post_install(tmp_path / "fake_post_install.py")
    release_json = output_dir / "release_gate.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--output-dir",
            str(output_dir),
            "--timestamp",
            "NO_POINTER",
            "--release-gate-json",
            str(release_json),
            "--skip-nemoclaw-check",
            "--nemoclaw-post-install-verify-script",
            str(fake_post_install),
            "--no-latest-pointer",
            "--quiet",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert release_json.exists()
    assert not (output_dir / "latest_taiwan_release_gate.json").exists()
    release = json.loads(release_json.read_text(encoding="utf-8"))
    assert "latest_pointer_json" not in release


def test_release_gate_auto_verifies_latest_pointer_for_formal_gate(tmp_path):
    output_dir = tmp_path / "out"
    fake_post_install = write_fake_post_install(tmp_path / "fake_post_install.py")
    timestamp = "20260627T212500Z"
    release_json = output_dir / f"taiwan_release_gate_{timestamp}.json"
    latest_pointer_json = output_dir / "latest_taiwan_release_gate.json"
    latest_pointer_verification_json = (
        output_dir / f"latest_taiwan_release_gate_verify_{timestamp}.json"
    )
    operator_md = output_dir / f"taiwan_release_operator_plan_{timestamp}.md"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--output-dir",
            str(output_dir),
            "--timestamp",
            timestamp,
            "--release-gate-json",
            str(release_json),
            "--skip-nemoclaw-check",
            "--nemoclaw-post-install-verify-script",
            str(fake_post_install),
            "--quiet",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert release_json.exists()
    assert latest_pointer_json.exists()
    assert latest_pointer_verification_json.exists()
    assert operator_md.exists()

    release = json.loads(release_json.read_text(encoding="utf-8"))
    latest_pointer = json.loads(latest_pointer_json.read_text(encoding="utf-8"))
    verification = json.loads(
        latest_pointer_verification_json.read_text(encoding="utf-8")
    )

    assert release["latest_pointer_json"] == str(latest_pointer_json)
    assert release["latest_pointer_verification_json"] == str(
        latest_pointer_verification_json
    )
    assert release["latest_pointer_verification_ok"] is True
    assert release["latest_pointer_verification_status"] == "passed"
    assert release["latest_pointer_verification_issue_count"] == 0
    assert release["latest_pointer_verification"] == {
        "json": str(latest_pointer_verification_json),
        "ok": True,
        "status": "passed",
        "issue_count": 0,
    }
    assert latest_pointer["operator_plan"] == release["operator_plan"]
    assert latest_pointer["latest_pointer_json"] == str(latest_pointer_json)
    assert latest_pointer["latest_pointer_verification_json"] == str(
        latest_pointer_verification_json
    )
    assert latest_pointer["latest_pointer_verification_ok"] is True
    assert latest_pointer["latest_pointer_verification_status"] == "passed"
    assert latest_pointer["latest_pointer_verification_issue_count"] == 0
    assert latest_pointer["latest_pointer_verification"] == release[
        "latest_pointer_verification"
    ]
    assert verification["ok"] is True
    assert verification["schema_version"] == 1
    assert verification["issue_count"] == 0
    assert verification["release_gate_json"] == str(release_json)
    operator_markdown = operator_md.read_text(encoding="utf-8")
    assert "latest_pointer_json" in operator_markdown
    assert "latest_pointer_verification_json" in operator_markdown
    assert str(latest_pointer_verification_json) in operator_markdown
    bundle_dir = Path(release["bundle"]["output_dir"])
    bundle_manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert bundle_manifest["release_gate_pointer"]["release_gate_json"] == str(release_json)
    assert bundle_manifest["release_gate_pointer"]["latest_pointer_json"] == str(
        latest_pointer_json
    )
    assert bundle_manifest["release_gate_pointer"][
        "latest_pointer_verification_json"
    ] == str(latest_pointer_verification_json)
    assert bundle_manifest["release_gate_pointer"][
        "latest_pointer_verification_ok"
    ] is True
    pointer_roles_by_source = {
        record.get("source_path"): record.get("roles")
        for record in bundle_manifest.get("files", [])
        if isinstance(record, dict)
    }
    assert "release_gate_pointer:release_gate_json" in pointer_roles_by_source[
        str(release_json)
    ]
    assert "release_gate_pointer:latest_pointer_json" in pointer_roles_by_source[
        str(latest_pointer_json)
    ]
    assert (
        "release_gate_pointer:latest_pointer_verification_json"
        in pointer_roles_by_source[str(latest_pointer_verification_json)]
    )
    assert bundle_manifest["release_gate_pointer_proof"]["json"] == (
        "release_gate_pointer_proof.json"
    )
    assert bundle_manifest["release_gate_pointer_proof"]["schema_version"] == 1
    assert bundle_manifest["release_gate_pointer_proof"]["status"] == "passed"
    proof_record = next(
        record
        for record in bundle_manifest.get("files", [])
        if isinstance(record, dict)
        and record.get("bundle_path") == "release_gate_pointer_proof.json"
    )
    assert "release_gate_pointer:proof_json" in proof_record["roles"]
    pointer_proof = json.loads(
        (bundle_dir / "release_gate_pointer_proof.json").read_text(encoding="utf-8")
    )
    assert pointer_proof["schema_version"] == 1
    assert pointer_proof["kind"] == "release_gate_pointer_proof"
    assert pointer_proof["ok"] is True
    assert pointer_proof["status"] == "passed"
    assert pointer_proof["release_gate_pointer"] == bundle_manifest[
        "release_gate_pointer"
    ]
    assert pointer_proof["latest_pointer_verification"]["issue_count"] == 0
    bundled_operator_markdown = (bundle_dir / "operator_plan.md").read_text(
        encoding="utf-8"
    )
    bundled_summary = (bundle_dir / "summary.md").read_text(encoding="utf-8")
    assert "latest_pointer_json" in bundled_operator_markdown
    assert "latest_pointer_verification_json" in bundled_operator_markdown
    assert str(latest_pointer_verification_json) in bundled_operator_markdown
    assert "## Release Gate Pointer" in bundled_summary
    assert str(latest_pointer_verification_json) in bundled_summary
