#!/usr/bin/env python3
"""Build a readiness report for the Agentic SWE-Assorted DeepSWE High slice."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUBSET = "essential_anchored_high_8_glm52max_cap100_10m_lang_balanced"
LEGACY_ESSENTIAL3_SUBSET = "essential_anchored_high_8_essential3_cost_trimmed_lang_balanced"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "taiwan" / "deepswe" / "manifest.json"
DEFAULT_TASKS_ROOT = REPO_ROOT / "external" / "deep-swe" / "tasks"
DEFAULT_FINDINGS = (
    REPO_ROOT / "outputs" / "deepswe_subset_analysis" / "agentic_swe_assorted_high_empirical_findings.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "outputs" / "deepswe_subset_analysis" / "agentic_swe_assorted_high_readiness.json"
)
DEFAULT_OPENCLAW_RUNTIME_PATCH = REPO_ROOT / "scripts" / "setup" / "patch_openclaw_turn_budget_guard.py"
DEFAULT_NEMOCLAW_RUNTIME_PATCH = REPO_ROOT / "scripts" / "setup" / "patch_nemoclaw_openclaw_runtime.py"
SCOREABLE_PILOT_STATUSES = {
    "scoreable_success",
    "scoreable_pilot_completed",
    "scoreable_incorrect",
    "scoreable_model_side_budget_failure",
    "scoreable_path_verified_unresolved",
}
UNSTABLE_SCOREABLE_REASONS = {
    "llm_response_idle_timeout",
    "llm_response_idle_timeout_scored_as_incorrect",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def check_manifest(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    manifest = read_json(args.manifest)
    subset = manifest.get("subsets", {}).get(args.subset)
    if not isinstance(subset, dict):
        return {
            "ok": False,
            "checks": [
                {
                    "name": "manifest_subset",
                    "ok": False,
                    "detail": f"subset missing from manifest: {args.subset}",
                }
            ],
        }

    task_names_path = args.manifest.parent / subset["task_names_path"]
    metadata_path = args.manifest.parent / subset["metadata_jsonl_path"]
    selection_summary_path = (args.manifest.parent / subset["selection_summary_path"]).resolve()
    task_names = read_json(task_names_path)
    records = read_jsonl(metadata_path)
    summary = read_json(selection_summary_path)

    checks.append(
        {
            "name": "manifest_count",
            "ok": subset.get("count") == len(task_names) == len(records) == 8,
            "detail": {
                "manifest_count": subset.get("count"),
                "task_names": len(task_names),
                "records": len(records),
            },
        }
    )
    checks.append(
        {
            "name": "task_name_order",
            "ok": [record.get("task_name") for record in records] == task_names,
            "detail": task_names,
        }
    )
    language_distribution = {}
    for record in records:
        language = record.get("language")
        language_distribution[language] = language_distribution.get(language, 0) + 1
    expected_languages = {"go": 3, "python": 2, "typescript": 3}
    checks.append(
        {
            "name": "language_balance",
            "ok": language_distribution == expected_languages,
            "detail": language_distribution,
        }
    )

    metrics = summary.get("selected_subset", {}).get("metrics", {})
    checks.append(
        {
            "name": "correlation_threshold",
            "ok": metrics.get("pearson", 0) >= args.min_pearson
            and metrics.get("spearman", 0) >= args.min_spearman,
            "detail": {
                "pearson": metrics.get("pearson"),
                "spearman": metrics.get("spearman"),
                "min_pearson": args.min_pearson,
                "min_spearman": args.min_spearman,
            },
        }
    )
    constraints = summary.get("constraints", {})
    must_include = set(constraints.get("must_include", []))
    checks.append(
        {
            "name": "essential_anchor_tasks",
            "ok": must_include.issubset(task_names),
            "detail": {"must_include": sorted(must_include), "present": sorted(must_include & set(task_names))},
        }
    )

    public_costs = [
        record.get("selection_stats", {}).get("public_avg_cost_usd", 0.0) for record in records
    ]
    public_steps = [
        record.get("selection_stats", {}).get("public_avg_steps", 0.0) for record in records
    ]
    return {
        "ok": all(check["ok"] for check in checks),
        "checks": checks,
        "subset": {
            "name": args.subset,
            "display_name": subset.get("display_name"),
            "task_names": task_names,
            "language_distribution": language_distribution,
            "public_avg_cost_usd_sum": sum(public_costs),
            "public_avg_cost_usd_mean": sum(public_costs) / len(public_costs),
            "public_avg_steps_mean": sum(public_steps) / len(public_steps),
            "metrics": metrics,
            "leave_family_out_summary": summary.get("leave_family_out", {}).get("summary", {}),
        },
        "evidence": {
            "manifest": rel(args.manifest),
            "task_names": rel(task_names_path),
            "metadata_jsonl": rel(metadata_path),
            "selection_summary": rel(selection_summary_path),
        },
        "records": records,
    }


def task_docker_images(tasks_root: Path, task_names: list[str]) -> list[str]:
    images: list[str] = []
    seen = set()
    for task_name in task_names:
        task_toml = tasks_root / task_name / "task.toml"
        text = task_toml.read_text(encoding="utf-8")
        match = re.search(r"^docker_image\s*=\s*['\"]([^'\"]+)['\"]", text, re.M)
        if not match:
            continue
        image = match.group(1)
        if image not in seen:
            seen.add(image)
            images.append(image)
    return images


def run_command(command: list[str], timeout: int) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "elapsed_seconds": time.time() - started,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": None,
            "elapsed_seconds": time.time() - started,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "error": "timeout",
        }


def parse_command_json(result: dict[str, Any]) -> dict[str, Any] | None:
    stdout = result.get("stdout") or ""
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def runtime_patch_ok(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return False
    patch_payload = payload.get("patch")
    if isinstance(patch_payload, dict):
        return runtime_patch_ok(patch_payload)
    return bool(payload.get("verified_extended_thinking"))


def check_openclaw_runtime(args: argparse.Namespace) -> dict[str, Any]:
    if args.skip_openclaw_runtime_check:
        return {"ok": None, "skipped": True}

    checks: list[dict[str, Any]] = []

    host_command = [
        sys.executable,
        str(args.openclaw_runtime_patch_script),
        "--check",
        "--json",
    ]
    if args.openclaw_package_dir:
        host_command.extend(["--openclaw-package-dir", str(args.openclaw_package_dir)])
    host_result = run_command(host_command, timeout=args.openclaw_runtime_check_timeout)
    host_payload = parse_command_json(host_result)
    checks.append(
        {
            "name": "host_openclaw_runtime_patch",
            "ok": runtime_patch_ok(host_payload),
            "command": host_command,
            "returncode": host_result.get("returncode"),
            "payload": host_payload,
            "stderr": host_result.get("stderr", ""),
        }
    )

    if args.sandbox:
        sandbox_command = [
            sys.executable,
            str(args.nemoclaw_runtime_patch_script),
            "--sandbox",
            args.sandbox,
            "--nemoclaw-bin",
            args.nemoclaw_bin,
            "--openclaw-package-dir",
            args.nemoclaw_openclaw_package_dir,
            "--check-only",
            "--json",
        ]
        sandbox_result = run_command(
            sandbox_command,
            timeout=args.openclaw_runtime_check_timeout,
        )
        sandbox_payload = parse_command_json(sandbox_result)
        checks.append(
            {
                "name": "nemoclaw_gateway_openclaw_runtime_patch",
                "ok": runtime_patch_ok(sandbox_payload),
                "command": sandbox_command,
                "returncode": sandbox_result.get("returncode"),
                "payload": sandbox_payload,
                "stderr": sandbox_result.get("stderr", ""),
            }
        )

    remediation = (
        f"NEMOCLAW_SANDBOX={args.sandbox} "
        "scripts/setup/install_openclaw_budget_guard.sh"
        if args.sandbox
        else "scripts/setup/install_openclaw_budget_guard.sh"
    )
    return {
        "ok": all(check["ok"] for check in checks),
        "checks": checks,
        "remediation": remediation,
        "evidence": {
            "host_patcher": rel(args.openclaw_runtime_patch_script),
            "nemoclaw_runtime_patcher": rel(args.nemoclaw_runtime_patch_script),
        },
    }


def check_docker_images(tasks_root: Path, task_names: list[str]) -> dict[str, Any]:
    checks = []
    for image in task_docker_images(tasks_root, task_names):
        result = run_command(["docker", "image", "inspect", image], timeout=30)
        checks.append(
            {
                "image": image,
                "ok": result["ok"],
                "returncode": result["returncode"],
            }
        )
    return {"ok": all(check["ok"] for check in checks), "checks": checks}


def check_sandbox(args: argparse.Namespace, task_names_file: Path) -> dict[str, Any]:
    if args.skip_sandbox_check:
        return {"ok": None, "skipped": True}
    command = [
        str(REPO_ROOT / "scripts" / "setup" / "install_deepswe_sandbox_deps.sh"),
        "--sandbox",
        args.sandbox,
        "--nemoclaw-bin",
        args.nemoclaw_bin,
        "--tasks-root",
        str(args.tasks_root),
        "--task-names-file",
        str(task_names_file),
        "--check-only",
    ]
    return run_command(command, timeout=args.sandbox_check_timeout)


def check_provider_findings(args: argparse.Namespace) -> dict[str, Any]:
    if not args.findings.exists():
        return {
            "ok": False,
            "full_run_recommended": False,
            "checks": [
                {
                    "name": "pilot_findings_present",
                    "ok": False,
                    "detail": f"missing findings file: {rel(args.findings)}",
                }
            ],
        }
    findings = read_json(args.findings)
    entries = findings.get("findings", [])
    current_task_names = current_subset_task_names(args)
    current_subset_entries = [
        entry
        for entry in entries
        if is_current_subset_provider_entry(entry, args, current_task_names)
        and (not args.model or entry.get("model") == args.model)
    ]
    provider_blocked = [
        entry
        for entry in current_subset_entries
        if is_provider_blocker(entry)
    ]
    configuration_errors = [
        entry
        for entry in current_subset_entries
        if str(entry.get("reason", "")) == "unsupported_model_thinking"
        or str(entry.get("status", "")) == "invalid_pilot_configuration_error"
    ]
    scoreable_pilots = [
        entry
        for entry in current_subset_entries
        if is_stable_scoreable_pilot(entry)
    ]
    unstable_scoreable_pilots = [
        entry
        for entry in current_subset_entries
        if is_scoreable_pilot_status(entry) and not is_stable_scoreable_pilot(entry)
    ]
    readiness_ok = not provider_blocked and bool(scoreable_pilots)
    return {
        "ok": readiness_ok,
        "full_run_recommended": readiness_ok,
        "model": args.model,
        "checks": [
            {
                "name": "current_subset_scoreable_pilot",
                "ok": bool(scoreable_pilots),
                "detail": {
                    "scoreable_pilot_count": len(scoreable_pilots),
                    "runs": [
                        {
                            "task_name": entry.get("task_name"),
                            "run_dir": entry.get("run_dir"),
                            "status": entry.get("status"),
                            "deepswe_reward": entry.get("deepswe_reward"),
                            "openclaw_disqualified_reason": entry.get(
                                "openclaw_disqualified_reason"
                            ),
                            "estimated_cost_usd": entry.get("estimated_cost_usd"),
                            "weave_agents_ok": entry.get("weave_agents_ok"),
                        }
                        for entry in scoreable_pilots
                    ],
                },
            },
            {
                "name": "current_subset_provider_blocker",
                "ok": not provider_blocked,
                "detail": [
                    {
                        "task_name": entry.get("task_name"),
                        "reason": entry.get("reason"),
                        "run_dir": entry.get("run_dir"),
                        "conversation_url": entry.get("conversation_url"),
                    }
                    for entry in provider_blocked
                ],
            },
            {
                "name": "current_subset_unstable_scoreable_history",
                "ok": not unstable_scoreable_pilots,
                "detail": [
                    {
                        "task_name": entry.get("task_name"),
                        "reason": entry.get("reason"),
                        "status": entry.get("status"),
                        "scoreable_failure_reason": entry.get("scoreable_failure_reason"),
                        "openclaw_disqualified_reason": entry.get(
                            "openclaw_disqualified_reason"
                        ),
                        "weave_agents_ok": entry.get("weave_agents_ok"),
                        "run_dir": entry.get("run_dir"),
                        "conversation_url": entry.get("conversation_url"),
                    }
                    for entry in unstable_scoreable_pilots
                ],
            },
            {
                "name": "current_subset_configuration_error_history",
                "ok": True,
                "detail": [
                    {
                        "task_name": entry.get("task_name"),
                        "reason": entry.get("reason"),
                        "thinking": entry.get("thinking"),
                        "required_thinking": entry.get("required_thinking"),
                        "run_dir": entry.get("run_dir"),
                    }
                    for entry in configuration_errors
                ],
            },
        ],
        "evidence": {"findings": rel(args.findings)},
    }


def is_provider_blocker(entry: dict[str, Any]) -> bool:
    reason = str(entry.get("reason", ""))
    return (
        "provider_timeout" in reason
        or "upstream_idle_timeout" in reason
        or str(entry.get("openclaw_disqualified_reason", "")) == "provider_transient_exhausted"
        or str(entry.get("deepswe_non_scoreable_reason", "")) == "provider_timeout"
    )


def is_scoreable_pilot_status(entry: dict[str, Any]) -> bool:
    return entry.get("deepswe_reward") == 1.0 or entry.get("status") in SCOREABLE_PILOT_STATUSES


def is_stable_scoreable_pilot(entry: dict[str, Any]) -> bool:
    if not is_scoreable_pilot_status(entry):
        return False
    reason = str(entry.get("reason", ""))
    scoreable_failure_reason = str(entry.get("scoreable_failure_reason", ""))
    if reason in UNSTABLE_SCOREABLE_REASONS:
        return False
    if scoreable_failure_reason in UNSTABLE_SCOREABLE_REASONS:
        return False
    if "llm_response_idle_timeout" in reason or "llm_response_idle_timeout" in scoreable_failure_reason:
        return False
    if entry.get("weave_agents_ok") is False:
        return False
    if str(entry.get("openclaw_disqualified_reason", "")) == "provider_transient_exhausted":
        return False
    return True


def current_subset_task_names(args: argparse.Namespace) -> set[str]:
    try:
        manifest = read_json(args.manifest)
        subset = manifest.get("subsets", {}).get(args.subset, {})
        task_names_path = args.manifest.parent / subset["task_names_path"]
        return {str(item) for item in read_json(task_names_path)}
    except Exception:
        return set()


def is_current_subset_provider_entry(
    entry: dict[str, Any],
    args: argparse.Namespace,
    current_task_names: set[str],
) -> bool:
    explicit_subset = entry.get("subset") or entry.get("source_subset") or entry.get("deepswe_subset")
    if explicit_subset:
        return str(explicit_subset) == args.subset

    run_dir = str(entry.get("run_dir", ""))
    if args.subset in run_dir:
        return True

    if (
        args.subset == LEGACY_ESSENTIAL3_SUBSET
        and run_dir.startswith("outputs/agentic_swe_assorted_runs/high_essential3_")
    ):
        return True
    if run_dir.startswith("outputs/agentic_swe_assorted_runs/high_essential3_"):
        return False

    task_name = str(entry.get("task_name", ""))
    if current_task_names and task_name not in current_task_names:
        return False
    return run_dir.startswith("outputs/agentic_swe_assorted_runs/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tasks-root", type=Path, default=DEFAULT_TASKS_ROOT)
    parser.add_argument("--findings", type=Path, default=DEFAULT_FINDINGS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="openrouter-direct/z-ai/glm-5.2")
    parser.add_argument("--sandbox", default="nejumi-taiwan")
    parser.add_argument("--nemoclaw-bin", default="nemoclaw")
    parser.add_argument("--sandbox-check-timeout", type=int, default=180)
    parser.add_argument("--skip-sandbox-check", action="store_true")
    parser.add_argument("--skip-openclaw-runtime-check", action="store_true")
    parser.add_argument("--openclaw-runtime-check-timeout", type=int, default=120)
    parser.add_argument("--openclaw-runtime-patch-script", type=Path, default=DEFAULT_OPENCLAW_RUNTIME_PATCH)
    parser.add_argument("--nemoclaw-runtime-patch-script", type=Path, default=DEFAULT_NEMOCLAW_RUNTIME_PATCH)
    parser.add_argument("--openclaw-package-dir", type=Path, default=None)
    parser.add_argument("--nemoclaw-openclaw-package-dir", default="/usr/local/lib/node_modules/openclaw")
    parser.add_argument("--min-pearson", type=float, default=0.90)
    parser.add_argument("--min-spearman", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_report = check_manifest(args)
    task_names_file = Path(manifest_report["evidence"]["task_names"])
    if not task_names_file.is_absolute():
        task_names_file = REPO_ROOT / task_names_file
    task_names = manifest_report.get("subset", {}).get("task_names", [])
    docker_report = check_docker_images(args.tasks_root, task_names)
    sandbox_report = check_sandbox(args, task_names_file)
    openclaw_runtime_report = check_openclaw_runtime(args)
    provider_report = check_provider_findings(args)

    design_ready = bool(manifest_report["ok"])
    sandbox_ready = bool(docker_report["ok"]) and sandbox_report.get("ok") is not False
    runtime_ready = openclaw_runtime_report.get("ok") is not False
    provider_ready = bool(provider_report["ok"])
    full_run_recommended = design_ready and sandbox_ready and runtime_ready and provider_ready
    if not runtime_ready:
        recommendation = (
            "Do not start a paid full High run until the OpenClaw/NeMoClaw runtime patch "
            f"check passes. Remediation: {openclaw_runtime_report.get('remediation')}"
        )
    elif not provider_ready:
        recommendation = (
            "Do not start a paid full High run until a scoreable long-form pilot passes "
            "without provider timeout."
        )
    elif not sandbox_ready:
        recommendation = "Do not start a paid full High run until Docker images and sandbox toolchain checks pass."
    elif not design_ready:
        recommendation = "Do not start a paid full High run until the subset manifest checks pass."
    else:
        recommendation = "Full High run is allowed by current evidence."
    report = {
        "generated_at_unix": time.time(),
        "readiness": {
            "design_ready": design_ready,
            "sandbox_ready": sandbox_ready,
            "runtime_ready": runtime_ready,
            "provider_ready": provider_ready,
            "full_run_recommended": full_run_recommended,
            "status": "ready" if full_run_recommended else "not_ready",
        },
        "recommendation": recommendation,
        "manifest": manifest_report,
        "docker_images": docker_report,
        "sandbox_toolchain": sandbox_report,
        "openclaw_runtime": openclaw_runtime_report,
        "provider_pilots": provider_report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["readiness"], ensure_ascii=False, indent=2))
    if not full_run_recommended:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
