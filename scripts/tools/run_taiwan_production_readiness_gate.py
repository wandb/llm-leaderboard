#!/usr/bin/env python3
"""Refresh local evidence and build the Taiwan production-readiness report.

This wrapper is intentionally offline with respect to model providers and W&B:
it refreshes local NeMoClaw setup evidence, then aggregates existing JSON
evidence into the production-readiness report.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import build_taiwan_production_readiness_report as readiness
import audit_taiwan_existing_results as existing_results_audit
import check_taiwan_paid_run_review_package as paid_review_check
import check_taiwan_nemoclaw_adoption as nemoclaw_adoption
import draft_existing_wandb_adoption_review as wandb_adoption_draft


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path("temp")
DEFAULT_NEMOCLAW_CHECK_SCRIPT = REPO_ROOT / "scripts" / "setup" / "install_nemoclaw.sh"
DEFAULT_NEMOCLAW_POST_INSTALL_VERIFY_SCRIPT = (
    REPO_ROOT / "scripts" / "setup" / "verify_nemoclaw_post_install.py"
)
DEFAULT_NEMOCLAW_OPERATOR_DOCS_VERIFY_SCRIPT = (
    REPO_ROOT / "scripts" / "setup" / "verify_nemoclaw_operator_docs.py"
)
DEFAULT_NEMOCLAW_OPERATOR_DOCS_README = Path("docs") / "README_nemoclaw.md"
DEFAULT_NEMOCLAW_OPERATOR_DOCS_LOCK_JSON = Path("scripts") / "setup" / "nemoclaw_installer_lock.json"
DEFAULT_OPENAI_CANARY_MANIFEST = Path("configs") / "taiwan_openai_canary_models.yaml"
DEFAULT_OPENAI_CANARY_GENERATED_FULL_DIR = Path("configs") / "taiwan_full" / "generated_openai_canary"
DEFAULT_OPENAI_CANARY_GENERATED_NONAGENTIC_DIR = (
    Path("configs") / "taiwan_full" / "generated_openai_canary_nonagentic"
)
DEFAULT_OPENAI_CANARY_GENERATED_AGENTIC_DIR = (
    Path("configs") / "taiwan_full" / "generated_openai_canary_agentic_nemoclaw"
)
DEFAULT_OPENAI_CANARY_GENERATED_AGENTIC_AGGREGATE_DIR = (
    Path("configs") / "taiwan_full" / "generated_openai_canary_agentic_aggregate"
)
DEFAULT_EXISTING_RESULTS_OUTPUT_ROOT = Path("outputs") / "taiwan_full_eval"
DEFAULT_EXISTING_RESULTS_WANDB_COMPLETION_DIR = DEFAULT_EXISTING_RESULTS_OUTPUT_ROOT / "wandb_completion"
DEFAULT_EXISTING_RESULTS_ARCHIVE_MANIFEST = (
    DEFAULT_EXISTING_RESULTS_OUTPUT_ROOT / "existing_results_archive_manifest.json"
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def repo_path(path: Path | str) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    return REPO_ROOT / value


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def run_nemoclaw_check(args: argparse.Namespace, timestamp: str) -> dict[str, Any]:
    if args.skip_nemoclaw_check:
        return {
            "skipped": True,
            "path": None,
            "returncode": None,
            "command": [],
            "stdout_tail": "",
            "stderr_tail": "",
        }

    output_path = repo_path(
        args.nemoclaw_setup_json
        or args.output_dir / f"nemoclaw_setup_check_{timestamp}.json"
    )
    command = [
        str(repo_path(args.nemoclaw_check_script)),
        "--check-only",
        "--json",
        str(output_path),
    ]
    completed = run_command(command)
    return {
        "skipped": False,
        "path": str(output_path),
        "returncode": completed.returncode,
        "command": command,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def run_existing_results_audit(args: argparse.Namespace, timestamp: str) -> dict[str, Any]:
    if args.skip_existing_results_audit:
        return {
            "skipped": True,
            "path": None,
            "markdown_path": None,
            "ok": None,
            "status": "skipped",
        }

    output_path = repo_path(
        args.existing_results_audit_json
        or args.output_dir / f"taiwan_existing_results_audit_{timestamp}.json"
    )
    markdown_path = repo_path(
        args.existing_results_audit_markdown
        or args.output_dir / f"taiwan_existing_results_audit_{timestamp}.md"
    )
    audit = existing_results_audit.build_audit(
        output_root=repo_path(args.existing_results_output_root),
        completion_dir=repo_path(args.existing_results_wandb_completion_dir),
        archive_manifest=repo_path(args.existing_results_archive_manifest)
        if args.existing_results_archive_manifest
        else None,
    )
    existing_results_audit.write_json(output_path, audit)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        existing_results_audit.summary_markdown(audit),
        encoding="utf-8",
    )
    return {
        "skipped": False,
        "path": str(output_path),
        "markdown_path": str(markdown_path),
        "ok": bool(audit.get("ok")),
        "status": audit.get("status"),
        "summary": audit.get("summary"),
    }


def run_wandb_adoption_draft(
    args: argparse.Namespace,
    timestamp: str,
    existing_results_check: dict[str, Any],
) -> dict[str, Any]:
    if args.skip_wandb_adoption_draft or existing_results_check.get("skipped"):
        return {
            "skipped": True,
            "path": None,
            "markdown_path": None,
            "ok": None,
            "status": "skipped",
        }
    audit_path = existing_results_check.get("path")
    if not isinstance(audit_path, str) or not audit_path:
        return {
            "skipped": True,
            "path": None,
            "markdown_path": None,
            "ok": None,
            "status": "missing_existing_results_audit",
        }
    output_path = repo_path(
        args.wandb_adoption_draft_json
        or args.output_dir / f"taiwan_wandb_adoption_draft_{timestamp}.json"
    )
    markdown_path = repo_path(
        args.wandb_adoption_draft_markdown
        or args.output_dir / f"taiwan_wandb_adoption_draft_{timestamp}.md"
    )
    attestation_template_dir = repo_path(
        args.wandb_adoption_attestation_template_dir
        or args.output_dir / f"taiwan_wandb_adoption_attestations_{timestamp}"
    )
    draft = wandb_adoption_draft.build_draft(
        audit_json=repo_path(audit_path),
        review_json=args.wandb_adoption_draft_review_json,
    )
    attestation_templates = wandb_adoption_draft.write_scope_attestation_templates(
        draft=draft,
        output_dir=attestation_template_dir,
    )
    wandb_adoption_draft.attach_operator_handoffs(draft)
    draft["path"] = str(output_path)
    draft["markdown_path"] = str(markdown_path)
    wandb_adoption_draft.write_json(output_path, draft)
    wandb_adoption_draft.write_text(markdown_path, wandb_adoption_draft.markdown(draft))
    return {
        "skipped": False,
        "path": str(output_path),
        "markdown_path": str(markdown_path),
        "attestation_template_dir": str(attestation_template_dir),
        "attestation_templates": attestation_templates,
        "ok": bool(draft.get("ok")),
        "status": draft.get("status"),
        "summary": {
            "candidate_count": draft.get("candidate_count"),
            "requires_human_scope_confirmation": draft.get("requires_human_scope_confirmation"),
            "scope_attestation_template_count": draft.get("scope_attestation_template_count"),
        },
    }


def run_wandb_adoption_unconfirmed_checks(
    args: argparse.Namespace,
    timestamp: str,
    wandb_adoption_draft_check: dict[str, Any],
) -> dict[str, Any]:
    """Prove generated scope-attestation templates are not execution approval.

    These checks intentionally run the generated, unconfirmed templates through
    the offline preflight and sync dry-run tools. They are expected to fail
    before any paid-run review mutation happens. The output is release evidence
    that an existing W&B result cannot be silently adopted without concrete
    human scope, cost, and billing fields.
    """

    if wandb_adoption_draft_check.get("skipped"):
        return {
            "skipped": True,
            "ok": None,
            "status": "skipped",
            "record_count": 0,
            "records": [],
        }
    draft_path = wandb_adoption_draft_check.get("path")
    if not isinstance(draft_path, str) or not draft_path.strip():
        return {
            "skipped": True,
            "ok": None,
            "status": "missing_wandb_adoption_draft",
            "record_count": 0,
            "records": [],
        }
    draft_file = repo_path(draft_path)
    if not draft_file.exists():
        return {
            "skipped": True,
            "ok": None,
            "status": "missing_wandb_adoption_draft",
            "record_count": 0,
            "records": [],
        }
    try:
        draft = json.loads(draft_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "skipped": False,
            "ok": False,
            "status": "invalid_wandb_adoption_draft",
            "record_count": 0,
            "records": [],
            "error": str(exc),
        }
    if not isinstance(draft, dict):
        return {
            "skipped": False,
            "ok": False,
            "status": "invalid_wandb_adoption_draft",
            "record_count": 0,
            "records": [],
            "error": "draft JSON is not an object",
        }

    output_dir = repo_path(args.output_dir) / f"wandb_adoption_unconfirmed_checks_{timestamp}"
    candidates = draft.get("candidates")
    records: list[dict[str, Any]] = []
    if isinstance(candidates, list):
        for index, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, dict):
                continue
            if candidate.get("sync_ready") is not True:
                continue
            benchmark = str(candidate.get("benchmark") or f"candidate_{index}")
            run_id = str(candidate.get("wandb_run_id") or f"run_{index}")
            prefix = (
                wandb_adoption_draft.slug_part(benchmark)
                + "-"
                + wandb_adoption_draft.slug_part(run_id)
            )
            preflight_json = output_dir / f"{prefix}.unconfirmed_scope_preflight.validation_failed.json"
            sync_dry_run_json = output_dir / f"{prefix}.unconfirmed_sync_dry_run.validation_failed.json"
            review_json = str(candidate.get("target_review_json") or "")
            completion_json = str(candidate.get("wandb_completion_json") or "")
            scope_attestation_json = str(candidate.get("scope_attestation_template_json") or "")
            preflight_command = [
                sys.executable,
                str(repo_path("scripts/tools/verify_wandb_scope_attestation.py")),
                "--review-json",
                review_json,
                "--completion-json",
                completion_json,
                "--scope-attestation-json",
                scope_attestation_json,
                "--json",
                str(preflight_json),
            ]
            sync_command = [
                sys.executable,
                str(repo_path("scripts/tools/sync_wandb_completion_to_paid_review.py")),
                "--review-json",
                review_json,
                "--completion-json",
                completion_json,
                "--set-verify-wandb-completion",
                "--top-level",
                "--adopt-existing-result",
                "--scope-attestation-json",
                scope_attestation_json,
                "--report-json",
                str(sync_dry_run_json),
            ]
            preflight_completed = run_command(preflight_command)
            sync_completed = run_command(sync_command)
            preflight_payload: dict[str, Any] | None = None
            sync_payload: dict[str, Any] | None = None
            if preflight_json.exists():
                try:
                    parsed = json.loads(preflight_json.read_text(encoding="utf-8"))
                    preflight_payload = parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    preflight_payload = None
            if sync_dry_run_json.exists():
                try:
                    parsed = json.loads(sync_dry_run_json.read_text(encoding="utf-8"))
                    sync_payload = parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    sync_payload = None
            preflight_failed_as_expected = (
                preflight_completed.returncode != 0
                and isinstance(preflight_payload, dict)
                and preflight_payload.get("ok") is False
                and preflight_payload.get("status") == "validation_failed"
            )
            sync_failed_as_expected = (
                sync_completed.returncode != 0
                and isinstance(sync_payload, dict)
                and sync_payload.get("ok") is False
                and sync_payload.get("status") == "validation_failed"
                and sync_payload.get("dry_run") is True
                and sync_payload.get("in_place") is False
                and sync_payload.get("entry_count") == 0
                and sync_payload.get("change_count") == 0
            )
            records.append(
                {
                    "benchmark": benchmark,
                    "run_id": run_id,
                    "scope_attestation_template_json": scope_attestation_json,
                    "preflight_report_json": str(preflight_json),
                    "preflight_returncode": preflight_completed.returncode,
                    "preflight_failed_as_expected": preflight_failed_as_expected,
                    "preflight_status": (
                        preflight_payload.get("status")
                        if isinstance(preflight_payload, dict)
                        else "missing_or_invalid_output"
                    ),
                    "sync_dry_run_report_json": str(sync_dry_run_json),
                    "sync_dry_run_returncode": sync_completed.returncode,
                    "sync_dry_run_failed_as_expected": sync_failed_as_expected,
                    "sync_dry_run_status": (
                        sync_payload.get("status")
                        if isinstance(sync_payload, dict)
                        else "missing_or_invalid_output"
                    ),
                    "will_query_wandb": False,
                    "will_write_wandb": False,
                    "will_launch_model_inference": False,
                    "will_mutate_review_json": False,
                    "preflight_stdout_tail": preflight_completed.stdout[-4000:],
                    "preflight_stderr_tail": preflight_completed.stderr[-4000:],
                    "sync_dry_run_stdout_tail": sync_completed.stdout[-4000:],
                    "sync_dry_run_stderr_tail": sync_completed.stderr[-4000:],
                }
            )
    ok = all(
        record.get("preflight_failed_as_expected") is True
        and record.get("sync_dry_run_failed_as_expected") is True
        for record in records
    )
    return {
        "skipped": False,
        "ok": ok,
        "status": "passed" if ok else "failed",
        "record_count": len(records),
        "records": records,
        "output_dir": str(output_dir),
    }


def build_report_args(
    args: argparse.Namespace,
    report_args: list[str],
    *,
    report_json: Path,
    nemoclaw_check: dict[str, Any],
    existing_results_check: dict[str, Any],
    nemoclaw_operator_docs_check: dict[str, Any],
) -> argparse.Namespace:
    argv = list(report_args)
    if not any(item == "--json" or item.startswith("--json=") for item in argv):
        argv.extend(["--json", str(report_json)])
    if not nemoclaw_check.get("skipped") and nemoclaw_check.get("path"):
        argv.extend(["--nemoclaw-setup-json", str(nemoclaw_check["path"])])
    if not existing_results_check.get("skipped") and existing_results_check.get("path"):
        argv.extend(["--existing-results-audit-json", str(existing_results_check["path"])])
    if (
        not nemoclaw_operator_docs_check.get("skipped")
        and nemoclaw_operator_docs_check.get("path")
    ):
        argv.extend(
            [
                "--nemoclaw-operator-docs-json",
                str(nemoclaw_operator_docs_check["path"]),
            ]
        )
    return readiness.parse_args(argv)


def run_nemoclaw_operator_docs_verification(
    args: argparse.Namespace,
    timestamp: str,
) -> dict[str, Any]:
    if args.skip_nemoclaw_operator_docs_verification:
        return {
            "skipped": True,
            "path": None,
            "markdown_path": None,
            "ok": None,
            "status": "skipped",
            "returncode": None,
            "command": [],
            "stdout_tail": "",
            "stderr_tail": "",
        }

    output_path = repo_path(
        args.nemoclaw_operator_docs_verification_json
        or args.output_dir / f"nemoclaw_operator_docs_verification_{timestamp}.json"
    )
    markdown_path = repo_path(
        args.nemoclaw_operator_docs_verification_markdown
        or args.output_dir / f"nemoclaw_operator_docs_verification_{timestamp}.md"
    )
    command = [
        sys.executable,
        str(repo_path(args.nemoclaw_operator_docs_verify_script)),
        "--readme",
        str(repo_path(args.nemoclaw_operator_docs_readme)),
        "--lock-json",
        str(repo_path(args.nemoclaw_operator_docs_lock_json)),
        "--json",
        str(output_path),
        "--markdown",
        str(markdown_path),
        "--fail-on-failed",
    ]
    completed = run_command(command)
    payload: dict[str, Any] | None = None
    if output_path.exists():
        try:
            parsed = json.loads(output_path.read_text(encoding="utf-8"))
            payload = parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            payload = None
    return {
        "skipped": False,
        "path": str(output_path),
        "markdown_path": str(markdown_path),
        "ok": bool(payload.get("ok")) if payload else False,
        "status": payload.get("status") if payload else "missing_or_invalid_output",
        "summary": {
            "missing_requirements": payload.get("missing_requirements", [])
        }
        if payload
        else {},
        "returncode": completed.returncode,
        "command": command,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def run_paid_review_check(
    args: argparse.Namespace,
    timestamp: str,
    parsed_report_args: argparse.Namespace,
) -> dict[str, Any]:
    if args.skip_paid_run_review_check:
        return {
            "skipped": True,
            "path": None,
            "markdown_path": None,
            "ok": None,
            "status": "skipped",
        }

    output_path = repo_path(
        args.paid_run_review_check_json
        or args.output_dir / f"taiwan_paid_run_review_check_{timestamp}.json"
    )
    markdown_path = repo_path(
        args.paid_run_review_check_markdown
        or args.output_dir / f"taiwan_paid_run_review_check_{timestamp}.md"
    )
    required_benchmarks = (
        parsed_report_args.required_wandb_benchmark
        or list(readiness.DEFAULT_REQUIRED_WANDB_BENCHMARKS)
    )
    review_paths = parsed_report_args.batch_review_json
    if review_paths is None:
        review_paths = readiness.normalize_paths(
            None,
            (str(parsed_report_args.output_root / "*paid_run_review.json"),),
        )
    check_args = argparse.Namespace(
        review_json=review_paths,
        required_wandb_benchmark=required_benchmarks,
        required_wandb_run_id=parsed_report_args.required_wandb_run_id,
        required_wandb_run_id_all=parsed_report_args.required_wandb_run_id_all,
        require_one_model_canary=not parsed_report_args.no_require_one_model_canary,
        wandb_completion_max_age_seconds=parsed_report_args.wandb_completion_max_age_seconds,
        weave_agents_completion_max_age_seconds=parsed_report_args.wandb_completion_max_age_seconds,
    )
    report = paid_review_check.build_report(check_args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        paid_review_check.build_markdown(report),
        encoding="utf-8",
    )
    return {
        "skipped": False,
        "path": str(output_path),
        "markdown_path": str(markdown_path),
        "ok": bool(report.get("ok")),
        "status": report.get("status"),
        "summary": report.get("summary"),
    }


def run_nemoclaw_adoption_check(
    args: argparse.Namespace,
    timestamp: str,
    parsed_report_args: argparse.Namespace,
) -> dict[str, Any]:
    if args.skip_nemoclaw_adoption_check:
        return {
            "skipped": True,
            "path": None,
            "markdown_path": None,
            "ok": None,
            "status": "skipped",
        }

    output_path = repo_path(
        args.nemoclaw_adoption_check_json
        or args.output_dir / f"taiwan_nemoclaw_adoption_check_{timestamp}.json"
    )
    markdown_path = repo_path(
        args.nemoclaw_adoption_check_markdown
        or args.output_dir / f"taiwan_nemoclaw_adoption_check_{timestamp}.md"
    )
    setup_paths = readiness.normalize_paths(
        parsed_report_args.nemoclaw_setup_json,
        (readiness.DEFAULT_NEMOCLAW_SETUP_GLOB,),
        discover_defaults=not parsed_report_args.no_default_evidence_discovery,
    )
    readiness_paths = readiness.filter_canary_readiness_paths(
        readiness.normalize_paths(
            parsed_report_args.readiness_json,
            readiness.DEFAULT_READINESS_GLOBS,
            discover_defaults=not parsed_report_args.no_default_evidence_discovery,
        )
    )
    adoption_args = argparse.Namespace(
        setup_json=setup_paths,
        readiness_json=readiness_paths,
        agentic_config=args.nemoclaw_adoption_agentic_config,
        agentic_config_glob=args.nemoclaw_adoption_agentic_config_glob,
        sandbox=args.nemoclaw_adoption_sandbox,
        no_default_evidence_discovery=(
            parsed_report_args.no_default_evidence_discovery
        ),
    )
    report = nemoclaw_adoption.build_report(adoption_args)
    report["path"] = str(output_path)
    report["markdown_path"] = str(markdown_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        nemoclaw_adoption.markdown(report),
        encoding="utf-8",
    )
    return {
        "skipped": False,
        "path": str(output_path),
        "markdown_path": str(markdown_path),
        "ok": bool(report.get("ok")),
        "status": report.get("status"),
        "summary": report.get("summary"),
    }


def run_nemoclaw_post_install_verification(
    args: argparse.Namespace,
    timestamp: str,
) -> dict[str, Any]:
    if args.skip_nemoclaw_post_install_verification:
        return {
            "skipped": True,
            "path": None,
            "markdown_path": None,
            "ok": None,
            "status": "skipped",
            "returncode": None,
            "command": [],
            "stdout_tail": "",
            "stderr_tail": "",
        }

    output_path = repo_path(
        args.nemoclaw_post_install_verification_json
        or args.output_dir / f"nemoclaw_post_install_verification_{timestamp}.json"
    )
    markdown_path = repo_path(
        args.nemoclaw_post_install_verification_markdown
        or args.output_dir / f"nemoclaw_post_install_verification_{timestamp}.md"
    )
    command = [
        sys.executable,
        str(repo_path(args.nemoclaw_post_install_verify_script)),
        "--output-dir",
        str(repo_path(args.output_dir)),
        "--timestamp",
        timestamp,
        "--sandbox",
        args.nemoclaw_adoption_sandbox,
        "--nemoclaw-bin",
        args.nemoclaw_post_install_nemoclaw_bin,
        "--python-command",
        args.nemoclaw_post_install_python_command,
        "--step-timeout-seconds",
        str(args.nemoclaw_post_install_step_timeout_seconds),
        "--install-check-script",
        str(repo_path(args.nemoclaw_check_script)),
        "--setup-json",
        str(repo_path(args.output_dir) / f"nemoclaw_post_install_setup_check_{timestamp}.json"),
        "--adoption-json",
        str(repo_path(args.output_dir) / f"nemoclaw_post_install_adoption_check_{timestamp}.json"),
        "--adoption-markdown",
        str(repo_path(args.output_dir) / f"nemoclaw_post_install_adoption_check_{timestamp}.md"),
        "--json",
        str(output_path),
        "--markdown",
        str(markdown_path),
    ]
    for flag, value in (
        ("--canary-manifest", args.nemoclaw_post_install_canary_manifest),
        ("--generated-full-dir", args.nemoclaw_post_install_generated_full_dir),
        ("--generated-nonagentic-dir", args.nemoclaw_post_install_generated_nonagentic_dir),
        ("--generated-agentic-dir", args.nemoclaw_post_install_generated_agentic_dir),
        (
            "--generated-agentic-aggregate-dir",
            args.nemoclaw_post_install_generated_agentic_aggregate_dir,
        ),
    ):
        if value is not None:
            command.extend([flag, str(repo_path(value))])
    for flag, value in (
        ("--canary-slug", args.nemoclaw_post_install_canary_slug),
        ("--canary-openclaw-model", args.nemoclaw_post_install_canary_openclaw_model),
        (
            "--canary-expected-pretrained-model",
            args.nemoclaw_post_install_canary_expected_pretrained_model,
        ),
    ):
        if value:
            command.extend([flag, value])
    for value in args.nemoclaw_post_install_adoption_agentic_config or []:
        command.extend(["--adoption-agentic-config", str(repo_path(value))])
    for value in args.nemoclaw_post_install_adoption_agentic_config_glob or []:
        command.extend(["--adoption-agentic-config-glob", value])
    completed = run_command(command)
    payload: dict[str, Any] | None = None
    if output_path.exists():
        try:
            parsed = json.loads(output_path.read_text(encoding="utf-8"))
            payload = parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            payload = None
    return {
        "skipped": False,
        "path": str(output_path),
        "markdown_path": str(markdown_path),
        "ok": bool(payload.get("ok")) if payload else False,
        "status": payload.get("status") if payload else "missing_or_invalid_output",
        "summary": {
            "failed_steps": [
                step.get("name")
                for step in payload.get("steps", [])
                if isinstance(step, dict) and not step.get("ok")
            ]
        }
        if payload
        else {},
        "returncode": completed.returncode,
        "command": command,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _gate_statuses(report: dict[str, Any]) -> list[tuple[str, str, bool]]:
    gates = report.get("gates")
    if not isinstance(gates, list):
        return []
    statuses: list[tuple[str, str, bool]] = []
    for gate in gates:
        if not isinstance(gate, dict):
            continue
        name = str(gate.get("name") or "")
        status = str(gate.get("status") or "")
        ok = bool(gate.get("ok"))
        if name:
            statuses.append((name, status, ok))
    return statuses


def _benchmark_evidence_rows(report: dict[str, Any]) -> list[dict[str, str]]:
    summary = report.get("summary")
    if not isinstance(summary, dict):
        return []
    evidence = summary.get("benchmark_evidence")
    if not isinstance(evidence, list):
        return []
    rows: list[dict[str, str]] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        standalone = item.get("standalone_wandb_completion")
        review = item.get("review_wandb_completion")
        rows.append(
            {
                "benchmark": str(item.get("benchmark") or ""),
                "required": "yes" if item.get("required") is True else "no",
                "run_id": str(item.get("expected_run_id") or ""),
                "standalone": str(standalone.get("status") if isinstance(standalone, dict) else ""),
                "review": str(review.get("status") if isinstance(review, dict) else ""),
                "proven": "yes" if item.get("completion_proven") else "no",
            }
        )
    return rows


def _format_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [
        max(len(str(row[index])) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    rendered = ["  " + "  ".join(value.ljust(widths[index]) for index, value in enumerate(headers))]
    rendered.append("  " + "  ".join("-" * width for width in widths))
    rendered.extend(
        "  " + "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )
    return rendered


def format_summary(report: dict[str, Any]) -> str:
    lines = [
        f"Taiwan production readiness: {report.get('status', 'unknown')}",
    ]
    runner = report.get("runner")
    if isinstance(runner, dict) and runner.get("report_json"):
        lines.append(f"Report JSON: {runner['report_json']}")

    summary = report.get("summary")
    blockers = summary.get("blockers") if isinstance(summary, dict) else None
    if isinstance(blockers, list) and blockers:
        lines.append("Blocking gates: " + ", ".join(str(item) for item in blockers))
    else:
        lines.append("Blocking gates: none")

    gate_rows = [
        [name, status, "yes" if ok else "no"]
        for name, status, ok in _gate_statuses(report)
    ]
    if gate_rows:
        lines.append("")
        lines.append("Gate status:")
        lines.extend(_format_table(["gate", "status", "ok"], gate_rows))

    benchmark_rows = [
        [
            row["benchmark"],
            row["required"],
            row["run_id"],
            row["standalone"],
            row["review"],
            row["proven"],
        ]
        for row in _benchmark_evidence_rows(report)
    ]
    if benchmark_rows:
        lines.append("")
        lines.append("Benchmark W&B evidence:")
        lines.extend(
            _format_table(
                ["benchmark", "required", "expected_run_id", "standalone", "review", "proven"],
                benchmark_rows,
            )
        )

    remediation = report.get("remediation_plan")
    if isinstance(remediation, list) and remediation:
        lines.append("")
        lines.append("Next actions:")
        for item in remediation:
            if not isinstance(item, dict):
                continue
            gate = str(item.get("gate") or "")
            next_action = str(item.get("next_action") or "")
            if gate and next_action:
                lines.append(f"  - {gate}: {next_action}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Any unknown arguments are forwarded to "
            "build_taiwan_production_readiness_report.py. For example:\n"
            "  --no-require-nemoclaw\n"
            "  --required-wandb-benchmark agentic_math\n"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timestamp", default=utc_timestamp())
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--nemoclaw-setup-json", type=Path)
    parser.add_argument("--nemoclaw-check-script", type=Path, default=DEFAULT_NEMOCLAW_CHECK_SCRIPT)
    parser.add_argument("--skip-nemoclaw-check", action="store_true")
    parser.add_argument(
        "--nemoclaw-operator-docs-verify-script",
        type=Path,
        default=DEFAULT_NEMOCLAW_OPERATOR_DOCS_VERIFY_SCRIPT,
    )
    parser.add_argument("--nemoclaw-operator-docs-readme", type=Path, default=DEFAULT_NEMOCLAW_OPERATOR_DOCS_README)
    parser.add_argument("--nemoclaw-operator-docs-lock-json", type=Path, default=DEFAULT_NEMOCLAW_OPERATOR_DOCS_LOCK_JSON)
    parser.add_argument("--nemoclaw-operator-docs-verification-json", type=Path)
    parser.add_argument("--nemoclaw-operator-docs-verification-markdown", type=Path)
    parser.add_argument("--skip-nemoclaw-operator-docs-verification", action="store_true")
    parser.add_argument(
        "--nemoclaw-post-install-verify-script",
        type=Path,
        default=DEFAULT_NEMOCLAW_POST_INSTALL_VERIFY_SCRIPT,
    )
    parser.add_argument("--nemoclaw-post-install-verification-json", type=Path)
    parser.add_argument("--nemoclaw-post-install-verification-markdown", type=Path)
    parser.add_argument("--nemoclaw-post-install-python-command", default="uv run python")
    parser.add_argument("--nemoclaw-post-install-nemoclaw-bin", default="nemoclaw")
    parser.add_argument("--nemoclaw-post-install-step-timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--nemoclaw-post-install-canary-manifest",
        type=Path,
        default=DEFAULT_OPENAI_CANARY_MANIFEST,
    )
    parser.add_argument("--nemoclaw-post-install-canary-slug")
    parser.add_argument("--nemoclaw-post-install-canary-openclaw-model")
    parser.add_argument("--nemoclaw-post-install-canary-expected-pretrained-model")
    parser.add_argument(
        "--nemoclaw-post-install-generated-full-dir",
        type=Path,
        default=DEFAULT_OPENAI_CANARY_GENERATED_FULL_DIR,
    )
    parser.add_argument(
        "--nemoclaw-post-install-generated-nonagentic-dir",
        type=Path,
        default=DEFAULT_OPENAI_CANARY_GENERATED_NONAGENTIC_DIR,
    )
    parser.add_argument(
        "--nemoclaw-post-install-generated-agentic-dir",
        type=Path,
        default=DEFAULT_OPENAI_CANARY_GENERATED_AGENTIC_DIR,
    )
    parser.add_argument(
        "--nemoclaw-post-install-generated-agentic-aggregate-dir",
        type=Path,
        default=DEFAULT_OPENAI_CANARY_GENERATED_AGENTIC_AGGREGATE_DIR,
    )
    parser.add_argument("--nemoclaw-post-install-adoption-agentic-config", type=Path, action="append")
    parser.add_argument("--nemoclaw-post-install-adoption-agentic-config-glob", action="append")
    parser.add_argument("--skip-nemoclaw-post-install-verification", action="store_true")
    parser.add_argument("--existing-results-audit-json", type=Path)
    parser.add_argument("--existing-results-audit-markdown", type=Path)
    parser.add_argument("--existing-results-output-root", type=Path, default=DEFAULT_EXISTING_RESULTS_OUTPUT_ROOT)
    parser.add_argument(
        "--existing-results-wandb-completion-dir",
        type=Path,
        default=DEFAULT_EXISTING_RESULTS_WANDB_COMPLETION_DIR,
    )
    parser.add_argument(
        "--existing-results-archive-manifest",
        type=Path,
        default=DEFAULT_EXISTING_RESULTS_ARCHIVE_MANIFEST,
    )
    parser.add_argument("--skip-existing-results-audit", action="store_true")
    parser.add_argument("--wandb-adoption-draft-json", type=Path)
    parser.add_argument("--wandb-adoption-draft-markdown", type=Path)
    parser.add_argument("--wandb-adoption-draft-review-json", default=wandb_adoption_draft.DEFAULT_REVIEW_JSON)
    parser.add_argument("--wandb-adoption-attestation-template-dir", type=Path)
    parser.add_argument("--skip-wandb-adoption-draft", action="store_true")
    parser.add_argument("--paid-run-review-check-json", type=Path)
    parser.add_argument("--paid-run-review-check-markdown", type=Path)
    parser.add_argument("--skip-paid-run-review-check", action="store_true")
    parser.add_argument("--nemoclaw-adoption-check-json", type=Path)
    parser.add_argument("--nemoclaw-adoption-check-markdown", type=Path)
    parser.add_argument("--nemoclaw-adoption-sandbox", default="nejumi-taiwan")
    parser.add_argument("--nemoclaw-adoption-agentic-config", type=Path, action="append")
    parser.add_argument("--nemoclaw-adoption-agentic-config-glob", action="append")
    parser.add_argument("--skip-nemoclaw-adoption-check", action="store_true")
    parser.add_argument("--fail-on-not-ready", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not print the human-readable readiness summary to stderr.",
    )
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> None:
    args, report_args = parse_args(argv)
    report_json = repo_path(
        args.report_json
        or args.output_dir / f"taiwan_production_readiness_report_{args.timestamp}.json"
    )
    nemoclaw_check = run_nemoclaw_check(args, args.timestamp)
    existing_results_check = run_existing_results_audit(args, args.timestamp)
    wandb_adoption_draft_result = run_wandb_adoption_draft(
        args,
        args.timestamp,
        existing_results_check,
    )
    nemoclaw_operator_docs_check = run_nemoclaw_operator_docs_verification(
        args,
        args.timestamp,
    )
    nemoclaw_post_install_verification = run_nemoclaw_post_install_verification(
        args,
        args.timestamp,
    )
    parsed_report_args = build_report_args(
        args,
        report_args,
        report_json=report_json,
        nemoclaw_check=nemoclaw_check,
        existing_results_check=existing_results_check,
        nemoclaw_operator_docs_check=nemoclaw_operator_docs_check,
    )
    report = readiness.build_report(parsed_report_args)
    paid_review_check_result = run_paid_review_check(
        args,
        args.timestamp,
        parsed_report_args,
    )
    nemoclaw_adoption_check_result = run_nemoclaw_adoption_check(
        args,
        args.timestamp,
        parsed_report_args,
    )
    report["runner"] = {
        "name": "run_taiwan_production_readiness_gate.py",
        "generated_at": time.time(),
        "report_json": str(report_json),
        "nemoclaw_check": nemoclaw_check,
        "existing_results_audit": existing_results_check,
        "wandb_adoption_draft": wandb_adoption_draft_result,
        "nemoclaw_operator_docs_verification": nemoclaw_operator_docs_check,
        "paid_run_review_check": paid_review_check_result,
        "nemoclaw_adoption_check": nemoclaw_adoption_check_result,
        "nemoclaw_post_install_verification": nemoclaw_post_install_verification,
        "forwarded_report_args": report_args,
    }
    if parsed_report_args.json:
        write_report(repo_path(parsed_report_args.json), report)
    if not args.quiet:
        if not args.no_summary:
            print(format_summary(report), file=sys.stderr)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_not_ready and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
