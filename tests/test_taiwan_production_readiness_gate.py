import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tools" / "run_taiwan_production_readiness_gate.py"


def test_gate_refreshes_nemoclaw_setup_and_writes_report(tmp_path):
    fake_check = tmp_path / "fake_install_nemoclaw.sh"
    fake_check.write_text(
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
  "ok": false,
  "commands": {
    "nemoclaw": {"available": false},
    "openshell": {"available": false}
  }
}
JSON
exit 1
""",
        encoding="utf-8",
    )
    fake_check.chmod(0o755)
    output_dir = tmp_path / "out"
    report_json = output_dir / "report.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--output-dir",
            str(output_dir),
            "--timestamp",
            "TEST",
            "--report-json",
            str(report_json),
            "--nemoclaw-check-script",
            str(fake_check),
            "--quiet",
            "--no-require-metadata-readiness",
            "--no-require-weave-content-canary",
            "--no-require-nemoclaw",
            "--no-require-wandb-completion",
            "--no-require-one-model-canary",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    setup_json = output_dir / "nemoclaw_setup_check_TEST.json"
    assert setup_json.exists()
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["runner"]["nemoclaw_check"]["returncode"] == 1
    assert report["runner"]["nemoclaw_check"]["path"] == str(setup_json)
    audit_json = output_dir / "taiwan_existing_results_audit_TEST.json"
    assert audit_json.exists()
    assert report["runner"]["existing_results_audit"]["path"] == str(audit_json)
    adoption_draft_json = output_dir / "taiwan_wandb_adoption_draft_TEST.json"
    adoption_draft_md = output_dir / "taiwan_wandb_adoption_draft_TEST.md"
    assert adoption_draft_json.exists()
    assert adoption_draft_md.exists()
    assert report["runner"]["wandb_adoption_draft"]["path"] == str(adoption_draft_json)
    assert report["runner"]["wandb_adoption_draft"]["markdown_path"] == str(adoption_draft_md)
    paid_review_json = output_dir / "taiwan_paid_run_review_check_TEST.json"
    paid_review_md = output_dir / "taiwan_paid_run_review_check_TEST.md"
    assert paid_review_json.exists()
    assert paid_review_md.exists()
    assert report["runner"]["paid_run_review_check"]["path"] == str(paid_review_json)
    assert report["runner"]["paid_run_review_check"]["markdown_path"] == str(paid_review_md)
    adoption_json = output_dir / "taiwan_nemoclaw_adoption_check_TEST.json"
    adoption_md = output_dir / "taiwan_nemoclaw_adoption_check_TEST.md"
    assert adoption_json.exists()
    assert adoption_md.exists()
    assert report["runner"]["nemoclaw_adoption_check"]["path"] == str(adoption_json)
    assert report["runner"]["nemoclaw_adoption_check"]["markdown_path"] == str(adoption_md)
    post_install_json = output_dir / "nemoclaw_post_install_verification_TEST.json"
    post_install_md = output_dir / "nemoclaw_post_install_verification_TEST.md"
    assert post_install_json.exists()
    assert post_install_md.exists()
    assert report["runner"]["nemoclaw_post_install_verification"]["path"] == str(post_install_json)
    assert report["runner"]["nemoclaw_post_install_verification"]["markdown_path"] == str(post_install_md)
    assert any(gate["name"] == "existing_results_formalization" for gate in report["gates"])
    assert report["gates"][2]["latest_setup_report"]["path"] == str(setup_json)


def test_gate_includes_same_run_post_install_readiness_before_report(tmp_path):
    fake_post_install = tmp_path / "fake_post_install.py"
    fake_post_install.write_text(
        """
import json
import sys
from pathlib import Path

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

readiness = output_dir / f"nemoclaw_canary_readiness_{timestamp}.json"
readiness.parent.mkdir(parents=True, exist_ok=True)
readiness.write_text(json.dumps({
    "ok": True,
    "checks": [
        {"name": "manifest has exactly one canary", "ok": True},
        {"name": "NeMoClaw command is available", "ok": True},
        {"name": "OpenShell command is available", "ok": True},
        {"name": "NeMoClaw version command succeeds", "ok": True},
        {"name": "NeMoClaw sandbox status succeeds: nejumi-taiwan", "ok": True},
        {"name": "OpenClaw runs inside NeMoClaw sandbox: nejumi-taiwan", "ok": True},
        {"name": "NeMoClaw runtime network policies are allowlisted", "ok": True}
    ]
}), encoding="utf-8")
json_path.parent.mkdir(parents=True, exist_ok=True)
json_path.write_text(json.dumps({
    "ok": True,
    "status": "passed",
    "steps": [],
    "path": str(json_path),
    "markdown_path": str(markdown_path)
}), encoding="utf-8")
markdown_path.write_text("# passed\\n", encoding="utf-8")
""",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    timestamp = "TEST_READY"
    readiness_json = output_dir / f"nemoclaw_canary_readiness_{timestamp}.json"
    report_json = output_dir / "report.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--output-dir",
            str(output_dir),
            "--timestamp",
            timestamp,
            "--report-json",
            str(report_json),
            "--nemoclaw-post-install-verify-script",
            str(fake_post_install),
            "--skip-nemoclaw-check",
            "--skip-existing-results-audit",
            "--skip-wandb-adoption-draft",
            "--skip-paid-run-review-check",
            "--skip-nemoclaw-adoption-check",
            "--skip-nemoclaw-operator-docs-verification",
            "--quiet",
            "--no-require-metadata-readiness",
            "--no-require-weave-content-canary",
            "--no-require-nemoclaw-operator-docs",
            "--no-require-wandb-completion",
            "--no-require-one-model-canary",
            "--readiness-json",
            str(readiness_json),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(report_json.read_text(encoding="utf-8"))
    gate = next(gate for gate in report["gates"] if gate["name"] == "nemoclaw_readiness")
    assert gate["ok"] is True
    assert gate["evidence_paths"] == [str(readiness_json)]
    assert report["runner"]["nemoclaw_post_install_verification"]["path"] == str(
        output_dir / f"nemoclaw_post_install_verification_{timestamp}.json"
    )


def test_gate_fail_on_not_ready_returns_nonzero(tmp_path):
    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--output-dir",
            str(tmp_path),
            "--timestamp",
            "TEST_FAIL",
            "--skip-nemoclaw-check",
            "--fail-on-not-ready",
            "--quiet",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1


def test_gate_prints_human_summary_to_stderr_and_json_to_stdout(tmp_path):
    fake_check = tmp_path / "fake_install_nemoclaw.sh"
    fake_check.write_text(
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
  "ok": false,
  "commands": {
    "nemoclaw": {"available": false},
    "openshell": {"available": false}
  }
}
JSON
exit 1
""",
        encoding="utf-8",
    )
    fake_check.chmod(0o755)
    output_dir = tmp_path / "out"
    report_json = output_dir / "report.json"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--output-dir",
            str(output_dir),
            "--timestamp",
            "SUMMARY",
            "--report-json",
            str(report_json),
            "--nemoclaw-check-script",
            str(fake_check),
            "--no-require-metadata-readiness",
            "--no-require-weave-content-canary",
            "--no-require-nemoclaw",
            "--required-wandb-benchmark",
            "agentic_math",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["schema_version"] == 1
    assert report["runner"]["report_json"] == str(report_json)
    assert "Taiwan production readiness:" in result.stderr
    assert "Gate status:" in result.stderr
    assert "Benchmark W&B evidence:" in result.stderr
    assert "agentic_math" in result.stderr
    assert "standalone" in result.stderr
