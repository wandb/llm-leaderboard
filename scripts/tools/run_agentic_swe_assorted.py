#!/usr/bin/env python3
"""Run Agentic SWE-Assorted: SWE-bench Lite Low/Middle + DeepSWE Essential High."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SWE_RUNNER = REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py"
LITE_EVAL_RUNNER = REPO_ROOT / "scripts" / "tools" / "evaluate_swebench_lite_patches.py"
DEEPSWE_RUNNER = REPO_ROOT / "scripts" / "tools" / "run_deepswe_openclaw.py"

DEFAULT_LOW_MIDDLE_JSONL = (
    REPO_ROOT / "data" / "taiwan" / "swebench_lite_assorted" / "subsets" / "low_middle_72.jsonl"
)
DEFAULT_LOW_MIDDLE_IDS = (
    REPO_ROOT
    / "data"
    / "taiwan"
    / "swebench_lite_assorted"
    / "subsets"
    / "low_middle_72_instance_ids.json"
)
DEFAULT_DEEPSWE_META = REPO_ROOT / "data" / "taiwan" / "deepswe" / "subsets" / "essential_8.jsonl"
DEFAULT_DEEPSWE_TASK_NAMES = (
    REPO_ROOT / "data" / "taiwan" / "deepswe" / "subsets" / "essential_8_task_names.json"
)
DEFAULT_OFFICIAL_SWEBENCH = REPO_ROOT / "external" / "SWE-bench"

OUTPUT_REQUIRED_COLUMNS = (
    "source_benchmark",
    "source_dataset",
    "source_subset",
    "source_instance_id",
    "agentic_swe_tier",
    "instance_id",
    "resolved",
    "score",
    "weave_agents_conversation_url",
    "openclaw_tool_call_count",
    "openclaw_usage",
)


def run_command(command: list[str], *, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    print("Running:", " ".join(shlex.quote(str(part)) for part in command), flush=True)
    proc = subprocess.Popen(
        [str(part) for part in command],
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        start_new_session=True,
    )
    stdout_parts: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        stdout_parts.append(line)
    returncode = proc.wait()
    stdout = "".join(stdout_parts)
    if returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {returncode}: {command}\n{stdout[-4000:]}"
        )
    return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr="")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def selected_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return rows
    return rows[: max(0, int(limit))]


def prepare_low_middle_inputs(args: argparse.Namespace) -> tuple[Path, Path, list[dict[str, Any]]]:
    rows = selected_rows(read_jsonl(args.low_middle_jsonl), args.low_middle_limit)
    jsonl_path = args.output_dir / "inputs" / "low_middle_selected.jsonl"
    ids_path = args.output_dir / "inputs" / "low_middle_selected_instance_ids.json"
    write_jsonl(jsonl_path, rows)
    write_json(ids_path, [str(row["instance_id"]) for row in rows])
    return jsonl_path, ids_path, rows


def prepare_high_inputs(args: argparse.Namespace) -> tuple[Path, list[dict[str, Any]]]:
    rows = selected_rows(read_jsonl(args.deepswe_metadata_jsonl), args.high_limit)
    task_names = [str(row["task_name"]) for row in rows]
    task_names_path = args.output_dir / "inputs" / "deepswe_high_task_names.json"
    write_json(task_names_path, task_names)
    return task_names_path, rows


def build_swe_command(args: argparse.Namespace, jsonl_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(SWE_RUNNER),
        "--dataset-jsonl",
        str(jsonl_path),
        "--output-dir",
        str(args.output_dir / "low_middle" / "openclaw"),
        "--checkout-root",
        str(args.checkout_root),
        "--prefix",
        args.prefix + "-lite",
        "--thinking",
        args.thinking,
        "--agent",
        args.agent,
        "--openclaw-timeout",
        str(args.swe_openclaw_timeout),
        "--openclaw-max-attempts",
        str(args.openclaw_max_attempts),
        "--openclaw-retry-base-seconds",
        str(args.openclaw_retry_base_seconds),
        "--openclaw-num-workers",
        str(args.swe_workers),
        "--openclaw-task-start-min-interval-seconds",
        str(args.swe_task_start_min_interval_seconds),
        "--max-input-tokens",
        str(args.max_input_tokens),
        "--max-cumulative-input-tokens",
        str(args.max_cumulative_input_tokens),
        "--max-cumulative-output-tokens",
        str(args.max_cumulative_output_tokens),
        "--max-tool-calls",
        str(args.max_tool_calls),
        "--max-agent-turns",
        str(args.max_agent_turns),
        "--max-tool-wall-seconds",
        str(args.max_tool_wall_seconds),
        "--model",
        args.model,
        "--nemoclaw-sandbox",
        args.nemoclaw_sandbox,
        "--nemoclaw-bin",
        args.nemoclaw_bin,
        "--nemoclaw-workdir",
        args.nemoclaw_workdir,
        "--nemoclaw-openclaw-config-path",
        args.nemoclaw_openclaw_config_path,
        "--nemoclaw-checkout-transfer-mode",
        args.nemoclaw_checkout_transfer_mode,
        "--nemoclaw-checkout-transfer-timeout",
        str(args.nemoclaw_checkout_transfer_timeout),
        "--openclaw-tool-profile",
        args.openclaw_tool_profile,
        "--task-agent-prefix",
        args.task_agent_prefix + "-lite",
        "--session-prefix",
        args.session_prefix + ":lite",
        "--no-local",
        "--no-weave-sidecar",
        "--weave-agents-entity",
        args.weave_agents_entity,
        "--weave-agents-project",
        args.weave_agents_project,
        "--weave-agents-agent-name",
        args.weave_agents_agent_name,
        "--weave-agents-limit",
        str(args.weave_agents_limit),
        "--weave-agents-verification-timeout",
        str(args.weave_agents_verification_timeout),
        "--weave-agents-poll-seconds",
        str(args.weave_agents_poll_seconds),
    ]
    if args.require_actual_token_usage:
        command.append("--require-actual-token-usage")
    if args.verify_weave_agents:
        command.append("--verify-weave-agents")
    if not args.use_task_agent:
        command.append("--no-use-task-agent")
    if args.dry_run:
        command.append("--dry-run")
    for denied_tool in args.deny_tool:
        command.extend(["--deny-tool", denied_tool])
    for pattern in args.deny_argument_pattern:
        command.extend(["--deny-argument-pattern", pattern])
    return command


def build_lite_eval_command(
    args: argparse.Namespace,
    *,
    patch_path: Path,
    ids_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(LITE_EVAL_RUNNER),
        "--official-repo",
        str(args.official_swebench_repo),
        "--patch-path",
        str(patch_path),
        "--instance-ids-json",
        str(ids_path),
        "--output-dir",
        str(args.output_dir / "low_middle" / "official_eval"),
        "--model-name",
        args.model,
        "--run-id",
        args.eval_run_id,
        "--max-workers",
        str(args.eval_workers),
        "--timeout",
        str(args.eval_timeout),
        "--namespace",
        args.swebench_namespace,
        "--cache-level",
        args.swebench_cache_level,
    ]
    if args.dry_run:
        command.append("--prepare-only")
    if args.no_docker_check:
        command.append("--no-docker-check")
    return command


def build_deepswe_command(args: argparse.Namespace, task_names_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(DEEPSWE_RUNNER),
        "--tasks-root",
        str(args.deepswe_tasks_root),
        "--task-names-file",
        str(task_names_path),
        "--output-dir",
        str(args.output_dir / "high" / "deepswe"),
        "--jobs-dir",
        str(args.output_dir / "high" / "pier_jobs"),
        "--job-name",
        args.prefix + "-deepswe",
        "--model",
        args.model,
        "--thinking",
        args.thinking,
        "--agent",
        args.agent,
        "--prefix",
        args.prefix + "-deepswe",
        "--n-concurrent",
        str(args.high_workers),
        "--openclaw-timeout",
        str(args.high_openclaw_timeout),
        "--openclaw-max-attempts",
        str(args.openclaw_max_attempts),
        "--openclaw-retry-base-seconds",
        str(args.openclaw_retry_base_seconds),
        "--max-input-tokens",
        str(args.max_input_tokens),
        "--max-cumulative-input-tokens",
        str(args.max_cumulative_input_tokens),
        "--max-cumulative-output-tokens",
        str(args.max_cumulative_output_tokens),
        "--max-tool-calls",
        str(args.max_tool_calls),
        "--max-agent-turns",
        str(args.max_agent_turns),
        "--max-tool-wall-seconds",
        str(args.max_tool_wall_seconds),
        "--nemoclaw-bin",
        args.nemoclaw_bin,
        "--nemoclaw-sandbox",
        args.nemoclaw_sandbox,
        "--nemoclaw-workdir",
        args.nemoclaw_workdir,
        "--nemoclaw-openclaw-config-path",
        args.nemoclaw_openclaw_config_path,
        "--nemoclaw-checkout-transfer-mode",
        args.nemoclaw_checkout_transfer_mode,
        "--nemoclaw-checkout-transfer-timeout",
        str(args.nemoclaw_checkout_transfer_timeout),
        "--openclaw-tool-profile",
        args.openclaw_tool_profile,
        "--task-agent-prefix",
        args.task_agent_prefix + "-deep",
        "--session-prefix",
        args.session_prefix + ":deep",
        "--weave-agents-entity",
        args.weave_agents_entity,
        "--weave-agents-project",
        args.weave_agents_project,
        "--weave-agents-agent-name",
        args.weave_agents_agent_name,
        "--weave-agents-limit",
        str(args.weave_agents_limit),
        "--weave-agents-verification-timeout",
        str(args.weave_agents_verification_timeout),
        "--weave-agents-poll-seconds",
        str(args.weave_agents_poll_seconds),
        "--no-local",
    ]
    command.append("--require-actual-token-usage" if args.require_actual_token_usage else "--no-require-actual-token-usage")
    command.append("--verify-weave-agents" if args.verify_weave_agents else "--no-verify-weave-agents")
    command.append("--use-task-agent" if args.use_task_agent else "--no-use-task-agent")
    if args.dry_run:
        command.append("--dry-run")
    for denied_tool in args.deny_tool:
        command.extend(["--deny-tool", denied_tool])
    for pattern in args.deny_argument_pattern:
        command.extend(["--deny-argument-pattern", pattern])
    return command


def numeric_value(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def usage_numbers(value: Any) -> dict[str, float]:
    totals = {"input_tokens": 0.0, "output_tokens": 0.0, "total_tokens": 0.0, "cost_usd": 0.0}

    def visit(item: Any, key_hint: str = "") -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                visit(child, str(key).lower())
            return
        if isinstance(item, list):
            for child in item:
                visit(child, key_hint)
            return
        number = numeric_value(item)
        if not number:
            return
        key = key_hint.replace("-", "_")
        if "cost" in key and ("usd" in key or key == "cost"):
            totals["cost_usd"] += number
        elif "input" in key or "prompt" in key:
            totals["input_tokens"] += number
        elif "output" in key or "completion" in key:
            totals["output_tokens"] += number
        elif "total" in key and "token" in key:
            totals["total_tokens"] += number

    visit(value)
    if not totals["total_tokens"]:
        totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
    return totals


def merge_usage(rows: list[dict[str, Any]]) -> dict[str, float]:
    totals = {"input_tokens": 0.0, "output_tokens": 0.0, "total_tokens": 0.0, "cost_usd": 0.0}
    for row in rows:
        usage = usage_numbers(row.get("openclaw_usage"))
        for key, value in usage.items():
            totals[key] += value
    return totals


def lite_rows(
    *,
    source_rows: list[dict[str, Any]],
    patch_rows: list[dict[str, Any]],
    eval_results: dict[str, bool],
) -> list[dict[str, Any]]:
    patch_by_id = {str(row.get("instance_id")): row for row in patch_rows}
    rows = []
    for source in source_rows:
        instance_id = str(source["instance_id"])
        patch = patch_by_id.get(instance_id) or {}
        resolved = bool(eval_results.get(instance_id, False))
        rows.append(
            {
                "benchmark_name": source.get("benchmark_name", "SWE-bench Lite"),
                "source_benchmark": source.get("source_benchmark", "SWE-bench Lite"),
                "source_dataset": source.get("source_dataset", "princeton-nlp/SWE-bench_Lite"),
                "source_subset": source.get("source_subset"),
                "source_instance_id": source.get("source_instance_id", instance_id),
                "agentic_swe_tier": source.get("agentic_swe_tier"),
                "instance_id": instance_id,
                "repo": source.get("repo"),
                "resolved": resolved,
                "score": 1.0 if resolved else 0.0,
                "patch_empty": not bool(patch.get("patch")),
                "has_patch_record": bool(patch),
                "openclaw_returncode": patch.get("openclaw_returncode"),
                "openclaw_disqualified_reason": patch.get("openclaw_disqualified_reason"),
                "openclaw_tool_call_count": patch.get("openclaw_tool_call_count"),
                "openclaw_tool_error_count": patch.get("openclaw_tool_error_count"),
                "openclaw_usage": patch.get("openclaw_usage"),
                "runtime_budget": patch.get("runtime_budget"),
                "weave_agents_ok": patch.get("weave_agents_ok"),
                "weave_agents_conversation_url": patch.get("weave_agents_conversation_url"),
                "weave_agents_conversation_link_html": patch.get(
                    "weave_agents_conversation_link_html"
                ),
                "weave_agents_trace_url": patch.get("weave_agents_trace_url"),
                "nemoclaw_session_audit_ok": patch.get("nemoclaw_session_audit_ok"),
                "openclaw_result_path": patch.get("openclaw_result_path"),
            }
        )
    return rows


def deepswe_rows(*, metadata_rows: list[dict[str, Any]], result_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task = {}
    for row in result_rows:
        task_name = str(row.get("task_name") or "")
        by_task[task_name.rsplit("/", 1)[-1]] = row
    rows = []
    for meta in metadata_rows:
        task_name = str(meta["task_name"])
        result = by_task.get(task_name) or {}
        resolved = bool(result.get("resolved"))
        score = result.get("score")
        rows.append(
            {
                "benchmark_name": "DeepSWE-Essential-8",
                "source_benchmark": "DeepSWE",
                "source_dataset": "DataCurve DeepSWE v1.1",
                "source_subset": str(meta.get("subset", "essential_8")),
                "source_instance_id": task_name,
                "agentic_swe_tier": "high",
                "instance_id": task_name,
                "repo": meta.get("repository"),
                "resolved": resolved,
                "score": score if isinstance(score, (int, float)) else (1.0 if resolved else 0.0),
                "patch_empty": None,
                "has_patch_record": bool(result),
                "openclaw_returncode": None,
                "openclaw_disqualified_reason": result.get("openclaw_disqualified_reason"),
                "openclaw_tool_call_count": result.get("openclaw_tool_call_count"),
                "openclaw_tool_error_count": None,
                "openclaw_usage": result.get("openclaw_usage"),
                "runtime_budget": None,
                "weave_agents_ok": result.get("weave_agents_ok"),
                "weave_agents_conversation_url": result.get("weave_agents_conversation_url"),
                "weave_agents_conversation_link_html": result.get(
                    "weave_agents_conversation_link_html"
                ),
                "weave_agents_trace_url": None,
                "nemoclaw_session_audit_ok": result.get("nemoclaw_session_audit_ok"),
                "openclaw_result_path": result.get("openclaw_result_path"),
                "public_avg_cost_usd": (meta.get("selection_stats") or {}).get("public_avg_cost_usd"),
            }
        )
    return rows


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    resolved = sum(1 for row in rows if row.get("resolved") is True)
    usage = merge_usage(rows)
    return {
        "total_instances": total,
        "resolved_instances": resolved,
        "unresolved_instances": total - resolved,
        "pass_at_1": resolved / total if total else 0.0,
        "empty_patches": sum(1 for row in rows if row.get("patch_empty") is True),
        "native_trace_ok": sum(1 for row in rows if row.get("weave_agents_ok") is True),
        "usage": usage,
    }


def build_summary(rows: list[dict[str, Any]], *, args: argparse.Namespace, elapsed: float) -> dict[str, Any]:
    by_tier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_tier[str(row.get("agentic_swe_tier"))].append(row)
        by_source[str(row.get("source_benchmark"))].append(row)
    return {
        "benchmark": "Agentic SWE-Assorted",
        "model": args.model,
        "elapsed_seconds": elapsed,
        "dry_run": bool(args.dry_run),
        "total": summarize_group(rows),
        "by_tier": {key: summarize_group(value) for key, value in sorted(by_tier.items())},
        "by_source": {key: summarize_group(value) for key, value in sorted(by_source.items())},
        "limits": {
            "max_input_tokens": args.max_input_tokens,
            "max_cumulative_input_tokens": args.max_cumulative_input_tokens,
            "max_cumulative_output_tokens": args.max_cumulative_output_tokens,
            "max_tool_calls": args.max_tool_calls,
            "max_agent_turns": args.max_agent_turns,
            "max_tool_wall_seconds": args.max_tool_wall_seconds,
            "swe_workers": args.swe_workers,
            "high_workers": args.high_workers,
        },
    }


def validate_output_rows(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows, start=1):
        missing = [column for column in OUTPUT_REQUIRED_COLUMNS if column not in row]
        if missing:
            raise ValueError(f"output row {index} is missing required columns: {missing}")


def json_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return value


def log_wandb(args: argparse.Namespace, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    import pandas as pd
    import wandb

    wandb.login()
    output_df = pd.DataFrame(rows)
    table_df = output_df.copy()
    for column in table_df.columns:
        if table_df[column].map(lambda value: isinstance(value, (dict, list, tuple))).any():
            table_df[column] = table_df[column].map(json_cell)
    leaderboard = pd.DataFrame(
        [
            {
                "model_name": args.model,
                "total_samples": summary["total"]["total_instances"],
                "issues_resolved": summary["total"]["resolved_instances"],
                "pass_at_1": summary["total"]["pass_at_1"],
                "low_pass_at_1": summary["by_tier"].get("low", {}).get("pass_at_1"),
                "middle_pass_at_1": summary["by_tier"].get("middle", {}).get("pass_at_1"),
                "high_pass_at_1": summary["by_tier"].get("high", {}).get("pass_at_1"),
            }
        ]
    )
    with wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        job_type="agentic-swe-assorted",
        name=args.wandb_run_name or f"agentic-swe-assorted/{args.model}",
        config=vars(args),
    ) as run:
        run.log(
            {
                "agentic_swe_leaderboard_table": wandb.Table(dataframe=leaderboard),
                "agentic_swe_output_table": wandb.Table(dataframe=table_df),
                "agentic_swe_results": summary,
                "agentic_swe/pass_at_1": float(summary["total"]["pass_at_1"]),
                "agentic_swe/resolved_instances": int(summary["total"]["resolved_instances"]),
                "agentic_swe/total_instances": int(summary["total"]["total_instances"]),
                "agentic_swe/low/pass_at_1": float(
                    summary["by_tier"].get("low", {}).get("pass_at_1") or 0.0
                ),
                "agentic_swe/middle/pass_at_1": float(
                    summary["by_tier"].get("middle", {}).get("pass_at_1") or 0.0
                ),
                "agentic_swe/high/pass_at_1": float(
                    summary["by_tier"].get("high", {}).get("pass_at_1") or 0.0
                ),
            }
        )
        artifact = wandb.Artifact(
            "agentic-swe-assorted-" + args.model.replace("/", "-").replace(":", "-"),
            type="evaluation-results",
            metadata={
                "model": args.model,
                "total_instances": summary["total"]["total_instances"],
                "resolved_instances": summary["total"]["resolved_instances"],
                "pass_at_1": summary["total"]["pass_at_1"],
            },
        )
        for filename in ("summary.json", "output_table.jsonl", "leaderboard_table.json"):
            path = args.output_dir / filename
            if path.exists():
                artifact.add_file(str(path), name=filename)
        run.log_artifact(artifact, aliases=["latest"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/agentic_swe_assorted"))
    parser.add_argument("--prefix", default="agentic-swe-assorted")
    parser.add_argument("--thinking", default="high")
    parser.add_argument("--agent", default="main")
    parser.add_argument("--low-middle-jsonl", type=Path, default=DEFAULT_LOW_MIDDLE_JSONL)
    parser.add_argument("--low-middle-instance-ids-json", type=Path, default=DEFAULT_LOW_MIDDLE_IDS)
    parser.add_argument("--low-middle-limit", type=int)
    parser.add_argument("--deepswe-metadata-jsonl", type=Path, default=DEFAULT_DEEPSWE_META)
    parser.add_argument("--deepswe-task-names-file", type=Path, default=DEFAULT_DEEPSWE_TASK_NAMES)
    parser.add_argument("--deepswe-tasks-root", type=Path, default=REPO_ROOT / "external" / "deep-swe" / "tasks")
    parser.add_argument("--high-limit", type=int)
    parser.add_argument("--official-swebench-repo", type=Path, default=DEFAULT_OFFICIAL_SWEBENCH)
    parser.add_argument("--checkout-root", type=Path, default=Path("outputs/swebench_lite_checkouts"))
    parser.add_argument("--skip-low-middle", action="store_true")
    parser.add_argument("--skip-high", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-docker-check", action="store_true")
    parser.add_argument("--swe-workers", type=int, default=4)
    parser.add_argument("--high-workers", type=int, default=2)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--swe-task-start-min-interval-seconds", type=float, default=5.0)
    parser.add_argument("--swe-openclaw-timeout", type=int, default=900)
    parser.add_argument("--high-openclaw-timeout", type=int, default=1800)
    parser.add_argument("--eval-timeout", type=int, default=1800)
    parser.add_argument("--eval-run-id", default=f"agentic-swe-assorted-{int(time.time())}")
    parser.add_argument("--openclaw-max-attempts", type=int, default=1)
    parser.add_argument("--openclaw-retry-base-seconds", type=float, default=15.0)
    parser.add_argument("--max-input-tokens", type=int, default=1_000_000)
    parser.add_argument("--max-cumulative-input-tokens", type=int, default=1_000_000)
    parser.add_argument("--max-cumulative-output-tokens", type=int, default=500_000)
    parser.add_argument("--max-tool-calls", type=int, default=40)
    parser.add_argument("--max-agent-turns", type=int, default=40)
    parser.add_argument("--max-tool-wall-seconds", type=int, default=120)
    parser.add_argument("--require-actual-token-usage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-task-agent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--verify-weave-agents", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nemoclaw-bin", default="nemoclaw")
    parser.add_argument("--nemoclaw-sandbox", default="nejumi-taiwan")
    parser.add_argument("--nemoclaw-workdir", default="/sandbox")
    parser.add_argument("--nemoclaw-openclaw-config-path", default="/sandbox/.openclaw/openclaw.json")
    parser.add_argument("--nemoclaw-checkout-transfer-mode", choices=["copy", "visible"], default="copy")
    parser.add_argument("--nemoclaw-checkout-transfer-timeout", type=int, default=600)
    parser.add_argument("--openclaw-tool-profile", default="coding")
    parser.add_argument("--task-agent-prefix", default="tw-swe-assorted")
    parser.add_argument("--session-prefix", default="agentic-swe-assorted")
    parser.add_argument("--weave-agents-entity", default=os.environ.get("WANDB_ENTITY", "llm-leaderboard"))
    parser.add_argument("--weave-agents-project", default=os.environ.get("WANDB_PROJECT", "tc-leaderboard"))
    parser.add_argument("--weave-agents-agent-name", default="nejumi-taiwan-openclaw")
    parser.add_argument("--weave-agents-limit", type=int, default=100)
    parser.add_argument("--weave-agents-verification-timeout", type=float, default=120.0)
    parser.add_argument("--weave-agents-poll-seconds", type=float, default=5.0)
    parser.add_argument("--swebench-namespace", default="swebench")
    parser.add_argument("--swebench-cache-level", choices=["none", "base", "env", "instance"], default="env")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "llm-leaderboard"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "tc-leaderboard"))
    parser.add_argument("--wandb-run-name")
    parser.add_argument(
        "--deny-tool",
        action="append",
        default=[
            "code_execution",
            "process",
            "process_*",
            "web_search",
            "web_fetch",
            "browser",
            "browser_*",
        ],
    )
    parser.add_argument(
        "--deny-argument-pattern",
        action="append",
        default=[
            r"https?://",
            r"\b(curl|wget)\b",
            r"\b(?:python(?:3)?\s+-m\s+)?pip(?:3)?\s+install\b",
            r"\b(requests|urllib|httpx)\.",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.checkout_root = args.checkout_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()

    output_rows: list[dict[str, Any]] = []
    low_middle_source_rows: list[dict[str, Any]] = []
    high_metadata_rows: list[dict[str, Any]] = []

    if not args.skip_low_middle:
        low_middle_jsonl, low_middle_ids, low_middle_source_rows = prepare_low_middle_inputs(args)
        run_command(build_swe_command(args, low_middle_jsonl))
        patch_path = args.output_dir / "low_middle" / "openclaw" / "patches.json"
        run_command(build_lite_eval_command(args, patch_path=patch_path, ids_path=low_middle_ids))
        patch_rows = read_json(patch_path)
        if args.dry_run:
            eval_results = {str(row["instance_id"]): False for row in low_middle_source_rows}
            write_json(args.output_dir / "low_middle" / "official_eval" / "eval_results.json", eval_results)
        else:
            eval_results = read_json(args.output_dir / "low_middle" / "official_eval" / "eval_results.json")
        output_rows.extend(
            lite_rows(
                source_rows=low_middle_source_rows,
                patch_rows=patch_rows,
                eval_results=eval_results,
            )
        )

    if not args.skip_high:
        high_task_names, high_metadata_rows = prepare_high_inputs(args)
        run_command(build_deepswe_command(args, high_task_names))
        high_results_path = args.output_dir / "high" / "deepswe" / "results.jsonl"
        high_result_rows = read_jsonl(high_results_path) if high_results_path.exists() else []
        output_rows.extend(deepswe_rows(metadata_rows=high_metadata_rows, result_rows=high_result_rows))

    validate_output_rows(output_rows)
    elapsed = time.time() - started_at
    summary = build_summary(output_rows, args=args, elapsed=elapsed)
    write_jsonl(args.output_dir / "output_table.jsonl", output_rows)
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        args.output_dir / "leaderboard_table.json",
        [
            {
                "model_name": args.model,
                "total_samples": summary["total"]["total_instances"],
                "issues_resolved": summary["total"]["resolved_instances"],
                "pass_at_1": summary["total"]["pass_at_1"],
                "low_pass_at_1": summary["by_tier"].get("low", {}).get("pass_at_1"),
                "middle_pass_at_1": summary["by_tier"].get("middle", {}).get("pass_at_1"),
                "high_pass_at_1": summary["by_tier"].get("high", {}).get("pass_at_1"),
            }
        ],
    )
    if args.wandb and not args.dry_run:
        log_wandb(args, output_rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
