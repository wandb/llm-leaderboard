#!/usr/bin/env python3
"""Verify that NeMoClaw operator docs match the release evidence contract.

This verifier is intentionally local-only. It reads README_nemoclaw.md and the
pinned installer lock JSON, then checks that human-facing setup instructions
include the same safety-critical command markers required by the release gate.
It does not download installers, install/onboard NeMoClaw, query W&B, or launch
model inference.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
DEFAULT_README = Path("docs") / "README_nemoclaw.md"
DEFAULT_LOCK_JSON = Path("scripts") / "setup" / "nemoclaw_installer_lock.json"
FORBIDDEN_DOC_MARKERS = ("openrouter", "OPENROUTER", "openrouter.ai")


REQUIRED_MARKER_GROUPS: dict[str, tuple[str, ...]] = {
    "check_only_command": (
        "scripts/setup/install_nemoclaw.sh",
        "--check-only",
        "--json temp/nemoclaw_setup_check.json",
    ),
    "installer_review_command": (
        "uv run python scripts/setup/review_nemoclaw_installer.py",
        "--url {installer_url}",
        "--install-ref {install_ref}",
        "--expected-sha256 {sha256}",
        "--lock-json {lock_json}",
        "--json temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json",
        "--markdown temp/nemoclaw_installer_review_YYYYMMDDTHHMM.md",
    ),
    "install_and_onboard_command": (
        "scripts/setup/install_nemoclaw.sh",
        "--install",
        "--onboard",
        "--install-ref {install_ref}",
        "--installer-lock-json {lock_json}",
        "--installer-sha256 REVIEWED_INSTALLER_SHA256",
        "--installer-review-json temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json",
        "--sandbox nejumi-taiwan",
        "--provider openai",
        "--policy-tier restricted",
        "--yes-i-accept-third-party-software",
        "--json temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.json",
    ),
    "install_operation_log_evidence": (
        "*.install.log",
        "*.onboard.log",
    ),
    "post_install_verification_command": (
        "uv run python scripts/setup/verify_nemoclaw_post_install.py",
        "--sandbox nejumi-taiwan",
        "--canary-manifest configs/taiwan_openai_canary_models.yaml",
        "--generated-full-dir configs/taiwan_full/generated_openai_canary",
        "--generated-nonagentic-dir configs/taiwan_full/generated_openai_canary_nonagentic",
        "--generated-agentic-dir configs/taiwan_full/generated_openai_canary_agentic_nemoclaw",
        "--generated-agentic-aggregate-dir configs/taiwan_full/generated_openai_canary_agentic_aggregate",
        "--json temp/nemoclaw_post_install_verification_TIMESTAMP.json",
        "--markdown temp/nemoclaw_post_install_verification_TIMESTAMP.md",
        "--fail-on-failed",
    ),
    "canary_readiness_command": (
        "uv run python scripts/tools/check_taiwan_canary_readiness.py",
        "--manifest configs/taiwan_openai_canary_models.yaml",
        "--generated-full-dir configs/taiwan_full/generated_openai_canary",
        "--generated-nonagentic-dir configs/taiwan_full/generated_openai_canary_nonagentic",
        "--generated-agentic-dir configs/taiwan_full/generated_openai_canary_agentic_nemoclaw",
        "--generated-agentic-aggregate-dir configs/taiwan_full/generated_openai_canary_agentic_aggregate",
        "--require-nemoclaw",
        "--nemoclaw-sandbox nejumi-taiwan",
        "--json outputs/taiwan_full_eval/openai_canary_readiness_nemoclaw.json",
    ),
    "adoption_check_command": (
        "uv run python scripts/tools/check_taiwan_nemoclaw_adoption.py",
        "--setup-json temp/nemoclaw_setup_check_YYYYMMDDTHHMM.json",
        "--readiness-json outputs/taiwan_full_eval/openai_canary_readiness_nemoclaw.json",
        "--sandbox nejumi-taiwan",
        "--agentic-config-glob configs/taiwan_full/generated_openai_canary_agentic_nemoclaw/*.yaml",
        "--json temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.json",
        "--markdown temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.md",
        "--fail-on-not-adoptable",
    ),
    "production_readiness_command": (
        "uv run python scripts/tools/run_taiwan_production_readiness_gate.py",
        "--report-json outputs/taiwan_full_eval/taiwan_production_readiness_report.json",
        "--fail-on-not-ready",
    ),
    "operator_docs_verifier_command": (
        "uv run python scripts/setup/verify_nemoclaw_operator_docs.py",
        "--json temp/nemoclaw_operator_docs_verification_YYYYMMDDTHHMM.json",
        "--markdown temp/nemoclaw_operator_docs_verification_YYYYMMDDTHHMM.md",
        "--fail-on-failed",
    ),
}


def utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def repo_path(path: Path | str) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    return REPO_ROOT / value


def path_display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def normalize_text(text: str) -> str:
    text = re.sub(r"\\\s*\n\s*", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def render_markers(lock: dict[str, Any], *, lock_path: Path) -> dict[str, tuple[str, ...]]:
    values = {
        "installer_url": str(lock.get("installer_url") or ""),
        "install_ref": str(lock.get("install_ref") or ""),
        "sha256": str(lock.get("sha256") or ""),
        "lock_json": path_display(lock_path),
    }
    rendered: dict[str, tuple[str, ...]] = {}
    for name, markers in REQUIRED_MARKER_GROUPS.items():
        rendered[name] = tuple(marker.format(**values) for marker in markers)
    return rendered


def lock_errors(lock: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if lock.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if lock.get("installer_url") != "https://www.nvidia.com/nemoclaw.sh":
        errors.append("installer_url must be https://www.nvidia.com/nemoclaw.sh")
    if lock.get("install_ref") != "lkg":
        errors.append("install_ref must be lkg")
    sha = str(lock.get("sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        errors.append("sha256 must be 64 lowercase hex characters")
    if not isinstance(lock.get("size_bytes"), int) or lock.get("size_bytes", 0) <= 0:
        errors.append("size_bytes must be a positive integer")
    for field in (
        "will_execute_installer",
        "will_install_or_onboard",
        "will_launch_model_inference",
        "will_query_wandb",
    ):
        if lock.get(field) is not False:
            errors.append(f"{field} must be false")
    return errors


def marker_check(
    *,
    name: str,
    normalized_readme: str,
    markers: tuple[str, ...],
) -> dict[str, Any]:
    missing = [marker for marker in markers if marker not in normalized_readme]
    return {
        "name": name,
        "ok": not missing,
        "missing_markers": missing,
        "required_markers": list(markers),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    readme_path = repo_path(args.readme)
    lock_path = repo_path(args.lock_json)
    checks: list[dict[str, Any]] = []
    missing_requirements: list[str] = []

    readme_text = ""
    if readme_path.exists():
        readme_text = readme_path.read_text(encoding="utf-8")
    checks.append(
        {
            "name": "readme_exists",
            "ok": readme_path.exists(),
            "path": path_display(readme_path),
        }
    )

    lock: dict[str, Any] = {}
    lock_read_ok = False
    lock_read_error = ""
    try:
        lock = read_json_object(lock_path)
        lock_read_ok = True
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        lock_read_error = str(exc)
    lock_validation_errors = lock_errors(lock) if lock_read_ok else [lock_read_error]
    checks.append(
        {
            "name": "installer_lock_json_valid",
            "ok": lock_read_ok and not lock_validation_errors,
            "path": path_display(lock_path),
            "errors": lock_validation_errors,
        }
    )

    normalized = normalize_text(readme_text)
    if lock_read_ok:
        for name, markers in render_markers(lock, lock_path=lock_path).items():
            checks.append(
                marker_check(
                    name=name,
                    normalized_readme=normalized,
                    markers=markers,
                )
            )

    forbidden_markers_found = [
        marker for marker in FORBIDDEN_DOC_MARKERS if marker.casefold() in readme_text.casefold()
    ]
    checks.append(
        {
            "name": "no_openrouter_markers",
            "ok": not forbidden_markers_found,
            "forbidden_markers": list(FORBIDDEN_DOC_MARKERS),
            "found": forbidden_markers_found,
        }
    )

    for check in checks:
        if check.get("ok"):
            continue
        name = str(check.get("name") or "unknown")
        missing_requirements.append(name)

    ok = not missing_requirements
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": ok,
        "status": "passed" if ok else "failed",
        "generated_at": utc_iso(),
        "readme_path": path_display(readme_path),
        "lock_json": path_display(lock_path),
        "will_execute_installer": False,
        "will_install_or_onboard": False,
        "will_launch_model_inference": False,
        "will_query_wandb": False,
        "missing_requirements": missing_requirements,
        "checks": checks,
        "lock_summary": {
            "installer_url": lock.get("installer_url"),
            "install_ref": lock.get("install_ref"),
            "sha256": lock.get("sha256"),
            "size_bytes": lock.get("size_bytes"),
            "reviewed_at": lock.get("reviewed_at"),
        }
        if lock_read_ok
        else {},
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# NeMoClaw Operator Docs Verification",
        "",
        f"- status: `{report.get('status')}`",
        f"- ok: `{report.get('ok')}`",
        f"- readme: `{report.get('readme_path')}`",
        f"- lock_json: `{report.get('lock_json')}`",
        f"- will_execute_installer: `{report.get('will_execute_installer')}`",
        f"- will_install_or_onboard: `{report.get('will_install_or_onboard')}`",
        f"- will_launch_model_inference: `{report.get('will_launch_model_inference')}`",
        f"- will_query_wandb: `{report.get('will_query_wandb')}`",
        "",
        "| Check | OK | Missing / Errors |",
        "|---|---:|---|",
    ]
    checks = report.get("checks")
    if isinstance(checks, list):
        for check in checks:
            if not isinstance(check, dict):
                continue
            missing = check.get("missing_markers") or check.get("errors") or check.get("found") or []
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(check.get("name") or ""),
                        str(bool(check.get("ok"))).lower(),
                        "`" + json.dumps(missing, ensure_ascii=False) + "`",
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    parser.add_argument("--lock-json", type=Path, default=DEFAULT_LOCK_JSON)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--fail-on-failed", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        report["path"] = str(repo_path(args.json))
    if args.markdown:
        report["markdown_path"] = str(repo_path(args.markdown))
    if args.json:
        write_json(repo_path(args.json), report)
    if args.markdown:
        path = repo_path(args.markdown)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_failed and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
