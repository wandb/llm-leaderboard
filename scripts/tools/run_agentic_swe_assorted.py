#!/usr/bin/env python3
"""Run Agentic SWE-Assorted: SWE-bench Lite Low/Middle + DeepSWE Essential-anchored High."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SWE_RUNNER = REPO_ROOT / "scripts" / "tools" / "run_swebench_pro_openclaw.py"
LITE_EVAL_RUNNER = REPO_ROOT / "scripts" / "tools" / "evaluate_swebench_lite_patches.py"
DEEPSWE_RUNNER = REPO_ROOT / "scripts" / "tools" / "run_deepswe_openclaw.py"

DEFAULT_LOW_MIDDLE_JSONL = (
    REPO_ROOT
    / "data"
    / "taiwan"
    / "swebench_lite_assorted"
    / "subsets"
    / "low_middle_v2_72.jsonl"
)
DEFAULT_LOW_MIDDLE_IDS = (
    REPO_ROOT
    / "data"
    / "taiwan"
    / "swebench_lite_assorted"
    / "subsets"
    / "low_middle_v2_72_instance_ids.json"
)
DEFAULT_DEEPSWE_META = (
    REPO_ROOT
    / "data"
    / "taiwan"
    / "deepswe"
    / "subsets"
    / "essential_anchored_high_8_glm52max_cap100_10m_lang_balanced.jsonl"
)
DEFAULT_DEEPSWE_TASK_NAMES = (
    REPO_ROOT
    / "data"
    / "taiwan"
    / "deepswe"
    / "subsets"
    / "essential_anchored_high_8_glm52max_cap100_10m_lang_balanced_task_names.json"
)
DEFAULT_DEEPSWE_PUBLIC_TRIALS = (
    REPO_ROOT / "outputs" / "deepswe_subset_analysis" / "deepswe_v1_1_trials.json"
)
TIER_ORDER = ("low", "middle", "high")
DEFAULT_OFFICIAL_SWEBENCH = REPO_ROOT / "external" / "SWE-bench"
DEFAULT_TIER_WEIGHTS: dict[str, float] = {tier: 1.0 for tier in TIER_ORDER}
DEFAULT_MIN_FREE_DISK_GB = 30.0

PRICE_PER_MILLION: dict[str, dict[str, float]] = {
    # Accountability estimates only. Provider dashboards remain authoritative.
    "openai-direct/gpt-4.1-mini-2025-04-14": {
        "input": 0.40,
        "output": 1.60,
        "cacheRead": 0.10,
        "cacheWrite": 0.0,
    },
    "openai-direct/gpt-5.6-luna": {
        "input": 1.00,
        "output": 6.00,
        "cacheRead": 0.10,
        "cacheWrite": 1.25,
    },
    "openai-direct/gpt-5.6-terra": {
        "input": 2.00,
        "output": 12.00,
        "cacheRead": 0.20,
        "cacheWrite": 2.50,
    },
    "openai-direct/gpt-5.6-sol": {
        "input": 5.00,
        "output": 25.00,
        "cacheRead": 0.50,
        "cacheWrite": 5.00,
    },
    "openrouter-direct/z-ai/glm-5.2": {
        "input": 0.95,
        "output": 3.00,
        "cacheRead": 0.18,
        "cacheWrite": 0.0,
    },
    "openrouter-direct/google/gemini-3.1-pro-preview": {
        "input": 2.00,
        "output": 12.00,
        "cacheRead": 0.30,
        "cacheWrite": 0.0,
    },
    "openrouter-direct/qwen/qwen3.6-max-preview": {
        "input": 1.10,
        "output": 3.00,
        "cacheRead": 0.0,
        "cacheWrite": 0.0,
    },
    "openrouter-direct/anthropic/claude-sonnet-4.6": {
        "input": 3.00,
        "output": 15.00,
        "cacheRead": 0.30,
        "cacheWrite": 0.0,
    },
    "openrouter-direct/anthropic/claude-opus-4.7": {
        "input": 5.00,
        "output": 25.00,
        "cacheRead": 0.50,
        "cacheWrite": 0.0,
    },
}

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


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def selected_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return rows
    return rows[: max(0, int(limit))]


def interleave_row_groups(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    selected = []
    max_len = max((len(group) for group in groups), default=0)
    for index in range(max_len):
        for group in groups:
            if index < len(group):
                selected.append(group[index])
    return selected


def selected_low_middle_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if getattr(args, "low_limit", None) is None and getattr(args, "middle_limit", None) is None:
        return selected_rows(rows, args.low_middle_limit)

    low_limit = max(0, int(getattr(args, "low_limit", 0) or 0))
    middle_limit = max(0, int(getattr(args, "middle_limit", 0) or 0))
    low_rows = [row for row in rows if str(row.get("agentic_swe_tier")) == "low"]
    middle_rows = [row for row in rows if str(row.get("agentic_swe_tier")) == "middle"]
    return interleave_row_groups(
        [
            selected_rows(low_rows, low_limit),
            selected_rows(middle_rows, middle_limit),
        ]
    )


def prepare_low_middle_inputs(args: argparse.Namespace) -> tuple[Path, Path, list[dict[str, Any]]]:
    rows = selected_low_middle_rows(read_jsonl(args.low_middle_jsonl), args)
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


def _docker_root_dir() -> Path | None:
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.DockerRootDir}}"],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    return Path(text) if text else None


def _disk_free_gb(path: Path) -> float | None:
    try:
        return shutil.disk_usage(path).free / (1024 ** 3)
    except OSError:
        return None


def runtime_preflight(args: argparse.Namespace, *, low_middle_rows: list[dict[str, Any]], high_rows: list[dict[str, Any]]) -> dict[str, Any]:
    report_path = args.output_dir / "inputs" / "runtime_preflight.json"
    min_free_disk_gb = float(getattr(args, "min_free_disk_gb", DEFAULT_MIN_FREE_DISK_GB) or 0.0)
    checks: list[dict[str, Any]] = []
    blocking_reasons: list[str] = []

    def add_disk_check(name: str, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        free_gb = _disk_free_gb(path)
        ok = free_gb is None or min_free_disk_gb <= 0 or free_gb >= min_free_disk_gb
        check = {
            "name": name,
            "path": str(path),
            "free_gb": free_gb,
            "min_free_disk_gb": min_free_disk_gb,
            "ok": ok,
        }
        checks.append(check)
        if not ok:
            blocking_reasons.append(
                f"{name} has {free_gb:.1f}GB free, below required {min_free_disk_gb:.1f}GB"
            )

    if not args.skip_low_middle and not args.no_docker_check and not args.dry_run:
        add_disk_check("output_dir_filesystem", args.output_dir)
        docker_root = _docker_root_dir()
        if docker_root is not None:
            add_disk_check("docker_root_filesystem", docker_root)
        else:
            checks.append(
                {
                    "name": "docker_root_filesystem",
                    "path": None,
                    "free_gb": None,
                    "min_free_disk_gb": min_free_disk_gb,
                    "ok": True,
                    "warning": "docker root could not be detected; Docker reachability is checked by the official Lite eval runner",
                }
            )

    report = {
        "ok": not blocking_reasons,
        "blocking_reasons": blocking_reasons,
        "checks": checks,
        "selected_counts": {
            "low_middle": len(low_middle_rows),
            "high": len(high_rows),
            "low": sum(1 for row in low_middle_rows if row.get("agentic_swe_tier") == "low"),
            "middle": sum(1 for row in low_middle_rows if row.get("agentic_swe_tier") == "middle"),
        },
        "deny_tool": list(args.deny_tool),
        "deny_argument_pattern": list(args.deny_argument_pattern),
    }
    write_json(report_path, report)
    if blocking_reasons:
        raise RuntimeError(
            "Agentic SWE-Assorted runtime preflight failed before paid execution. "
            f"See {report_path}. First issue: {blocking_reasons[0]}"
        )
    return report


def normalize_deepswe_public_model(model: str | None) -> str | None:
    if not model:
        return None
    text = str(model).strip().lower()
    if not text:
        return None
    text = text.removeprefix("openrouter-direct/")
    text = text.removeprefix("openai-direct/")
    text = text.removeprefix("wandb-direct/")
    text = text.removeprefix("wandb-inference/")
    leaf = text.rsplit("/", 1)[-1]
    aliases = {
        "glm-5.2": "glm-5-2",
        "glm_5_2": "glm-5-2",
        "gpt-5.6-luna": "gpt-5-6-luna",
        "gpt-5.6-sol": "gpt-5-6-sol",
        "gpt-5.6-terra": "gpt-5-6-terra",
        "gemini-3.1-pro-preview": "gemini-3-1-pro-preview",
        "claude-opus-4.8": "claude-opus-4-8",
        "claude-sonnet-4.6": "claude-sonnet-4-6",
    }
    return aliases.get(leaf, leaf.replace(".", "-").replace("_", "-"))


def normalize_deepswe_public_effort(effort: str | None) -> str | None:
    if not effort:
        return None
    text = str(effort).strip().lower()
    if not text:
        return None
    aliases = {
        "reasoning=max": "max",
        "thinking=max": "max",
        "reasoning=xhigh": "xhigh",
        "thinking=xhigh": "xhigh",
        "reasoning=high": "high",
        "thinking=high": "high",
    }
    return aliases.get(text, text)


def _float_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.get(key)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "max": None,
        }
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 0.50),
        "p75": _percentile(values, 0.75),
        "p90": _percentile(values, 0.90),
        "max": max(values),
    }


def _stat_value(stats: dict[str, Any], key: str) -> float | None:
    value = stats.get(key)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def _high_limit_values(args: argparse.Namespace) -> dict[str, int | None]:
    high_max_agent_turns = (
        args.high_max_agent_turns
        if args.high_max_agent_turns is not None
        else args.max_agent_turns
    )
    high_max_cumulative_input_tokens = (
        args.high_max_cumulative_input_tokens
        if args.high_max_cumulative_input_tokens is not None
        else args.max_cumulative_input_tokens
    )
    return {
        "agent_turns": high_max_agent_turns,
        "cumulative_input_tokens": high_max_cumulative_input_tokens,
    }


def deepswe_budget_preflight(args: argparse.Namespace, metadata_rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_names = [str(row["task_name"]) for row in metadata_rows]
    report_path = args.output_dir / "inputs" / "deepswe_budget_preflight.json"
    policy = str(getattr(args, "deepswe_budget_preflight", "error") or "error")
    if policy == "off":
        report = {
            "ok": True,
            "policy": policy,
            "skipped": True,
            "task_names": task_names,
        }
        write_json(report_path, report)
        return report

    trials_path = Path(args.deepswe_public_trials_json)
    if not trials_path.exists():
        report = {
            "ok": False,
            "policy": policy,
            "error": f"DeepSWE public trials file not found: {trials_path}",
            "task_names": task_names,
        }
        write_json(report_path, report)
        if policy == "error" and not args.dry_run and not args.allow_deepswe_budget_mismatch:
            raise RuntimeError(report["error"])
        return report

    payload = read_json(trials_path)
    public_rows = [
        row
        for row in payload.get("rows", [])
        if row.get("source") == "deep-swe"
        and row.get("eval_scope") == "full"
        and bool(row.get("included_in_score"))
        and str(row.get("task_name")) in set(task_names)
    ]
    public_model = args.deepswe_public_model or normalize_deepswe_public_model(args.model)
    public_effort = args.deepswe_public_effort or normalize_deepswe_public_effort(args.thinking)
    model_rows = [
        row
        for row in public_rows
        if (not public_model or row.get("model") == public_model)
        and (not public_effort or row.get("reasoning_effort") == public_effort)
    ]
    selected_rows_for_budget = model_rows if model_rows else public_rows
    basis = "model_effort" if model_rows else "all_models_fallback"
    limits = _high_limit_values(args)
    hard_stat = str(getattr(args, "deepswe_preflight_hard_stat", "mean") or "mean")
    task_reports: list[dict[str, Any]] = []
    blocking_reasons: list[str] = []
    warning_reasons: list[str] = []

    for task_name in task_names:
        rows = [row for row in selected_rows_for_budget if str(row.get("task_name")) == task_name]
        steps = _summary_stats(_float_values(rows, "n_agent_steps"))
        input_tokens = _summary_stats(_float_values(rows, "n_input_tokens"))
        cost = _summary_stats(_float_values(rows, "cost_usd"))
        task_report = {
            "task_name": task_name,
            "n_trials": len(rows),
            "steps": steps,
            "input_tokens": input_tokens,
            "cost_usd": cost,
            "blocking_reasons": [],
            "warning_reasons": [],
        }
        if not rows:
            reason = f"{task_name}: no public DeepSWE rows for budget basis {basis}"
            task_report["blocking_reasons"].append(reason)
            blocking_reasons.append(reason)
        step_limit = limits["agent_turns"]
        step_value = _stat_value(steps, hard_stat)
        if step_limit is not None and step_value is not None and step_value > float(step_limit):
            reason = (
                f"{task_name}: public {hard_stat} steps {step_value:.1f} "
                f"exceeds configured high_max_agent_turns {step_limit}"
            )
            task_report["blocking_reasons"].append(reason)
            blocking_reasons.append(reason)
        step_p75 = _stat_value(steps, "p75")
        if step_limit is not None and step_p75 is not None and step_p75 > float(step_limit):
            reason = (
                f"{task_name}: public p75 steps {step_p75:.1f} "
                f"exceeds configured high_max_agent_turns {step_limit}"
            )
            task_report["warning_reasons"].append(reason)
            warning_reasons.append(reason)
        input_limit = limits["cumulative_input_tokens"]
        input_value = _stat_value(input_tokens, hard_stat)
        if input_limit is not None and input_value is not None and input_value > float(input_limit):
            reason = (
                f"{task_name}: public {hard_stat} input tokens {input_value:.0f} "
                f"exceeds configured high_max_cumulative_input_tokens {input_limit}"
            )
            task_report["blocking_reasons"].append(reason)
            blocking_reasons.append(reason)
        input_p75 = _stat_value(input_tokens, "p75")
        if input_limit is not None and input_p75 is not None and input_p75 > float(input_limit):
            reason = (
                f"{task_name}: public p75 input tokens {input_p75:.0f} "
                f"exceeds configured high_max_cumulative_input_tokens {input_limit}"
            )
            task_report["warning_reasons"].append(reason)
            warning_reasons.append(reason)
        task_reports.append(task_report)

    report = {
        "ok": not blocking_reasons,
        "policy": policy,
        "basis": basis,
        "public_model": public_model,
        "public_effort": public_effort,
        "hard_stat": hard_stat,
        "limits": limits,
        "trials_path": str(trials_path),
        "task_names": task_names,
        "blocking_reasons": blocking_reasons,
        "warning_reasons": warning_reasons,
        "tasks": task_reports,
    }
    write_json(report_path, report)
    if blocking_reasons and policy == "error" and not args.dry_run and not args.allow_deepswe_budget_mismatch:
        raise RuntimeError(
            "DeepSWE High budget preflight failed before paid execution. "
            f"See {report_path}. First issue: {blocking_reasons[0]}"
        )
    if warning_reasons:
        print(
            f"DeepSWE High budget preflight warnings: {len(warning_reasons)} "
            f"(details: {report_path})",
            flush=True,
        )
    return report


def parse_tier_weights(raw: str | None) -> dict[str, float]:
    if raw is None or not raw.strip():
        return dict(DEFAULT_TIER_WEIGHTS)

    text = raw.strip()
    if text.startswith("{"):
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("--tier-weights JSON must be an object")
        items = payload.items()
    else:
        items = []
        for part in text.split(","):
            if not part.strip():
                continue
            key, sep, value = part.partition("=")
            if sep != "=":
                raise ValueError(
                    "--tier-weights must be JSON or comma-separated tier=value pairs"
                )
            items.append((key.strip(), value.strip()))

    weights = dict(DEFAULT_TIER_WEIGHTS)
    for key, value in items:
        if key not in DEFAULT_TIER_WEIGHTS:
            raise ValueError(f"unknown tier weight key: {key}")
        weight = float(value)
        if weight < 0:
            raise ValueError(f"tier weight must be non-negative: {key}={weight}")
        weights[key] = weight
    if sum(weights.values()) <= 0:
        raise ValueError("at least one tier weight must be positive")
    return weights


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
        "--openclaw-model-params-json",
        getattr(args, "openclaw_model_params_json", None) or "{}",
        "--openclaw-model-overrides-json",
        getattr(args, "openclaw_model_overrides_json", None) or "{}",
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
    high_max_tool_calls = (
        args.high_max_tool_calls
        if args.high_max_tool_calls is not None
        else args.max_tool_calls
    )
    high_max_agent_turns = (
        args.high_max_agent_turns
        if args.high_max_agent_turns is not None
        else args.max_agent_turns
    )
    high_max_cumulative_input_tokens = (
        args.high_max_cumulative_input_tokens
        if args.high_max_cumulative_input_tokens is not None
        else args.max_cumulative_input_tokens
    )
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
        "--openclaw-model-params-json",
        getattr(args, "openclaw_model_params_json", None) or "{}",
        "--openclaw-model-overrides-json",
        getattr(args, "openclaw_model_overrides_json", None) or "{}",
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
        str(high_max_cumulative_input_tokens),
        "--max-cumulative-output-tokens",
        str(args.max_cumulative_output_tokens),
        "--max-tool-calls",
        str(high_max_tool_calls),
        "--max-agent-turns",
        str(high_max_agent_turns),
        "--max-tool-wall-seconds",
        str(args.max_tool_wall_seconds),
        "--llm-response-idle-timeout-seconds",
        str(args.llm_response_idle_timeout_seconds),
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


def deepswe_benchmark_name(source_subset: str) -> str:
    if source_subset == "essential_anchored_high_8_safe_lang_balanced":
        return "DeepSWE-Essential-Anchored-High-8-Safe-Lang-Balanced"
    if source_subset == "essential_anchored_high_8_safe_go_local_lang_balanced":
        return "DeepSWE-Essential-Anchored-High-8-Safe-Go-Local-Lang-Balanced"
    if source_subset == "essential_anchored_high_8_cost_trimmed_no_rust_lang_balanced":
        return "DeepSWE-Essential-Anchored-High-8-Cost-Trimmed-No-Rust-Lang-Balanced"
    if source_subset == "essential_anchored_high_8_cost_trimmed_cap_safe_lang_balanced":
        return "DeepSWE-Essential-Anchored-High-8-Cost-Trimmed-Cap-Safe-Lang-Balanced"
    if source_subset == "essential_anchored_high_8_essential3_cost_trimmed_lang_balanced":
        return "DeepSWE-Essential-Anchored-High-8-Essential3-Cost-Trimmed-Lang-Balanced"
    if source_subset == "essential_anchored_high_8_glm52max_cap100_10m_lang_balanced":
        return "DeepSWE-Essential-Anchored-High-8-GLM52Max-Cap100-10M-Lang-Balanced"
    if source_subset == "essential_anchored_high_8_lang_balanced":
        return "DeepSWE-Essential-Anchored-High-8-Lang-Balanced"
    if source_subset == "budgeted_high_8_cap50_lang_balanced":
        return "DeepSWE-Budgeted-High-8-Cap50-Lang-Balanced"
    if source_subset == "budgeted_high_8_lang_balanced":
        return "DeepSWE-Budgeted-High-8-Lang-Balanced"
    if source_subset == "budgeted_high_8":
        return "DeepSWE-Budgeted-High-8"
    return f"DeepSWE-{source_subset}"


def numeric_value(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def usage_numbers(value: Any) -> dict[str, float]:
    totals = {
        "input_tokens": 0.0,
        "output_tokens": 0.0,
        "cache_read_input_tokens": 0.0,
        "cache_write_input_tokens": 0.0,
        "total_tokens": 0.0,
        "cost_usd": 0.0,
    }

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
        elif "cache" in key and "read" in key:
            totals["cache_read_input_tokens"] += number
        elif "cache" in key and "write" in key:
            totals["cache_write_input_tokens"] += number
        elif "input" in key or "prompt" in key:
            totals["input_tokens"] += number
        elif "output" in key or "completion" in key:
            totals["output_tokens"] += number
        elif "total" in key and "token" in key:
            totals["total_tokens"] += number

    visit(value)
    if not totals["total_tokens"]:
        totals["total_tokens"] = (
            totals["input_tokens"]
            + totals["output_tokens"]
            + totals["cache_read_input_tokens"]
            + totals["cache_write_input_tokens"]
        )
    return totals


def estimate_cost_usd(model: str, usage: dict[str, float]) -> float | None:
    price = PRICE_PER_MILLION.get(model)
    if price is None:
        return None
    return (
        usage.get("input_tokens", 0.0) / 1_000_000 * price.get("input", 0.0)
        + usage.get("output_tokens", 0.0) / 1_000_000 * price.get("output", 0.0)
        + usage.get("cache_read_input_tokens", 0.0) / 1_000_000 * price.get("cacheRead", 0.0)
        + usage.get("cache_write_input_tokens", 0.0) / 1_000_000 * price.get("cacheWrite", 0.0)
    )


def merge_usage(rows: list[dict[str, Any]], *, model: str) -> dict[str, float]:
    totals = usage_numbers({})
    explicit_cost = 0.0
    estimated_cost = 0.0
    for row in rows:
        usage = usage_numbers(row.get("openclaw_usage"))
        for key, value in usage.items():
            if key == "cost_usd":
                explicit_cost += value
            else:
                totals[key] += value
        if usage["cost_usd"]:
            continue
        cost = estimate_cost_usd(model, usage)
        if cost is not None:
            estimated_cost += cost
    totals["cost_usd"] = explicit_cost + estimated_cost
    return totals


def usage_from_weave_agents_verifier(path_value: Any) -> dict[str, Any]:
    if not isinstance(path_value, str) or not path_value:
        return {}
    path = Path(path_value)
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    checks = payload.get("checks")
    if isinstance(checks, list):
        for check in checks:
            if (
                not isinstance(check, dict)
                or check.get("name") != "usage"
                or check.get("ok") is not True
            ):
                continue
            input_tokens = (
                numeric_value(check.get("agent_input_tokens"))
                + numeric_value(check.get("trace_input_tokens"))
            )
            output_tokens = (
                numeric_value(check.get("agent_output_tokens"))
                + numeric_value(check.get("trace_output_tokens"))
            )
            if input_tokens + output_tokens <= 0:
                return {}
            return {
                "inputTokens": int(input_tokens),
                "outputTokens": int(output_tokens),
                "cacheReadInputTokens": 0,
                "usageSource": "weave_agents_trace",
                "usageApproximate": True,
            }
    health = payload.get("content_capture_health")
    if isinstance(health, dict):
        input_tokens = numeric_value(health.get("trace_input_tokens"))
        output_tokens = numeric_value(health.get("trace_output_tokens"))
        if input_tokens + output_tokens <= 0:
            input_tokens = numeric_value(health.get("conversation_input_tokens"))
            output_tokens = numeric_value(health.get("conversation_output_tokens"))
        if input_tokens + output_tokens > 0:
            return {
                "inputTokens": int(input_tokens),
                "outputTokens": int(output_tokens),
                "cacheReadInputTokens": 0,
                "usageSource": "weave_agents_trace",
                "usageApproximate": True,
            }
    return {}


def runtime_budget_estimated_usage(openclaw: dict[str, Any]) -> dict[str, Any] | None:
    runtime_budget = openclaw.get("runtime_budget")
    if not isinstance(runtime_budget, dict):
        return None
    observed = runtime_budget.get("observed")
    live = runtime_budget.get("live")
    estimated_input_tokens = None
    if isinstance(observed, dict):
        estimated_input_tokens = observed.get("estimated_input_tokens")
    if not estimated_input_tokens and isinstance(live, dict):
        estimated_input_tokens = live.get("estimated_input_tokens")
    estimated_input_tokens = numeric_value(estimated_input_tokens)
    if not estimated_input_tokens:
        return None
    return {
        "inputTokens": int(estimated_input_tokens),
        "outputTokens": 0,
        "cacheReadInputTokens": 0,
        "cacheWriteInputTokens": 0,
        "usageSource": "runtime_budget_estimated_input_floor",
        "usageApproximate": True,
        "usageLowerBound": True,
    }


def openclaw_usage_with_trace_fallback(record: dict[str, Any]) -> dict[str, Any] | None:
    usage = record.get("openclaw_usage")
    if isinstance(usage, dict) and usage:
        return usage
    agent_result = record.get("agent_result") if isinstance(record.get("agent_result"), dict) else {}
    metadata = agent_result.get("metadata") if isinstance(agent_result.get("metadata"), dict) else {}
    openclaw = metadata.get("openclaw") if isinstance(metadata.get("openclaw"), dict) else {}
    usage = openclaw.get("openclaw_usage")
    if isinstance(usage, dict) and usage:
        return usage
    trace_usage = usage_from_weave_agents_verifier(
        record.get("weave_agents_verifier_json") or openclaw.get("weave_agents_verifier_json")
    )
    if trace_usage:
        return trace_usage
    return runtime_budget_estimated_usage(openclaw)


def runtime_budget_live_reason(openclaw: dict[str, Any]) -> str | None:
    runtime_budget = openclaw.get("runtime_budget")
    if not isinstance(runtime_budget, dict):
        return None
    live = runtime_budget.get("live")
    if not isinstance(live, dict):
        return None
    reason = live.get("reason") or live.get("interrupt_reason")
    return str(reason) if reason else None


def scoreable_failure_reason(openclaw: dict[str, Any]) -> str | None:
    reason = openclaw.get("scoreable_failure_reason") or openclaw.get(
        "deepswe_scoreable_failure_reason"
    )
    if reason:
        return str(reason)
    live_reason = runtime_budget_live_reason(openclaw)
    if live_reason in {"llm_response_idle_timeout"}:
        return live_reason
    return None


def nested_deepswe_openclaw_metadata(result: dict[str, Any]) -> dict[str, Any]:
    agent_result = result.get("agent_result") if isinstance(result.get("agent_result"), dict) else {}
    metadata = agent_result.get("metadata") if isinstance(agent_result.get("metadata"), dict) else {}
    openclaw = metadata.get("openclaw") if isinstance(metadata.get("openclaw"), dict) else {}
    return openclaw


def nested_deepswe_patch_empty(result: dict[str, Any], openclaw: dict[str, Any]) -> bool | None:
    patch_bytes = openclaw.get("deepswe_patch_bytes")
    if isinstance(patch_bytes, (int, float)):
        return int(patch_bytes) == 0
    agent_result = result.get("agent_result") if isinstance(result.get("agent_result"), dict) else {}
    metadata = agent_result.get("metadata") if isinstance(agent_result.get("metadata"), dict) else {}
    patch_apply = metadata.get("patch_apply") if isinstance(metadata.get("patch_apply"), dict) else {}
    if patch_apply.get("reason") == "empty_patch":
        return True
    return None


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
                "scoreable_failure_reason": patch.get("scoreable_failure_reason"),
                "openclaw_tool_call_count": patch.get("openclaw_tool_call_count"),
                "openclaw_tool_error_count": patch.get("openclaw_tool_error_count"),
                "openclaw_usage": openclaw_usage_with_trace_fallback(patch),
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
        openclaw = nested_deepswe_openclaw_metadata(result)
        resolved = bool(result.get("resolved"))
        score = result.get("score")
        source_subset = str(
            meta.get("subset", "essential_anchored_high_8_essential3_cost_trimmed_lang_balanced")
        )
        rows.append(
            {
                "benchmark_name": deepswe_benchmark_name(source_subset),
                "source_benchmark": "DeepSWE",
                "source_dataset": "DataCurve DeepSWE v1.1",
                "source_subset": source_subset,
                "source_instance_id": task_name,
                "agentic_swe_tier": "high",
                "instance_id": task_name,
                "repo": meta.get("repository"),
                "resolved": resolved,
                "score": score if isinstance(score, (int, float)) else (1.0 if resolved else 0.0),
                "patch_empty": nested_deepswe_patch_empty(result, openclaw),
                "has_patch_record": bool(result),
                "openclaw_returncode": openclaw.get("returncode"),
                "openclaw_disqualified_reason": result.get("openclaw_disqualified_reason")
                or openclaw.get("openclaw_disqualified_reason"),
                "scoreable_failure_reason": result.get("scoreable_failure_reason")
                or scoreable_failure_reason(openclaw),
                "openclaw_tool_call_count": result.get("openclaw_tool_call_count")
                or openclaw.get("openclaw_tool_call_count"),
                "openclaw_tool_error_count": openclaw.get("openclaw_tool_error_count"),
                "openclaw_usage": openclaw_usage_with_trace_fallback(result),
                "runtime_budget": openclaw.get("runtime_budget"),
                "weave_agents_ok": result.get("weave_agents_ok")
                if result.get("weave_agents_ok") is not None
                else openclaw.get("weave_agents_ok"),
                "weave_agents_conversation_url": result.get("weave_agents_conversation_url")
                or openclaw.get("weave_agents_conversation_url"),
                "weave_agents_conversation_link_html": result.get(
                    "weave_agents_conversation_link_html"
                )
                or openclaw.get("weave_agents_conversation_link_html"),
                "weave_agents_trace_url": openclaw.get("weave_agents_trace_url"),
                "nemoclaw_session_audit_ok": result.get("nemoclaw_session_audit_ok")
                if result.get("nemoclaw_session_audit_ok") is not None
                else openclaw.get("nemoclaw_session_audit_ok"),
                "openclaw_result_path": result.get("openclaw_result_path")
                or openclaw.get("openclaw_result_path"),
                "public_avg_cost_usd": (meta.get("selection_stats") or {}).get("public_avg_cost_usd"),
            }
        )
    return rows


def summarize_group(rows: list[dict[str, Any]], *, model: str) -> dict[str, Any]:
    total = len(rows)
    resolved = sum(1 for row in rows if row.get("resolved") is True)
    failure_reasons = Counter()
    for row in rows:
        if row.get("resolved") is True:
            continue
        reason = row.get("scoreable_failure_reason") or row.get("openclaw_disqualified_reason")
        if reason:
            failure_reasons[str(reason)] += 1
    usage = merge_usage(rows, model=model)
    return {
        "total_instances": total,
        "resolved_instances": resolved,
        "unresolved_instances": total - resolved,
        "pass_at_1": resolved / total if total else 0.0,
        "empty_patches": sum(1 for row in rows if row.get("patch_empty") is True),
        "native_trace_ok": sum(1 for row in rows if row.get("weave_agents_ok") is True),
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "usage": usage,
    }


def build_scoring_summary(by_tier: dict[str, list[dict[str, Any]]], *, args: argparse.Namespace) -> dict[str, Any]:
    weights = parse_tier_weights(getattr(args, "tier_weights", None))
    by_tier_summary = {
        tier: summarize_group(rows, model=args.model)
        for tier, rows in sorted(by_tier.items())
    }
    normalized_denominator = sum(weights.values())
    normalized_weights = {
        tier: weight / normalized_denominator
        for tier, weight in weights.items()
        if weight > 0
    }
    present_tiers = {
        tier
        for tier, summary in by_tier_summary.items()
        if summary.get("total_instances", 0) > 0
    }
    missing_tiers = [
        tier
        for tier, weight in weights.items()
        if weight > 0 and tier not in present_tiers
    ]

    present_weight_sum = sum(
        weight
        for tier, weight in weights.items()
        if weight > 0 and tier in present_tiers
    )
    weighted_present_pass_at_1 = None
    if present_weight_sum > 0:
        weighted_present_pass_at_1 = sum(
            by_tier_summary[tier]["pass_at_1"] * weight
            for tier, weight in weights.items()
            if weight > 0 and tier in present_tiers
        ) / present_weight_sum

    weighted_pass_at_1 = None
    if not missing_tiers and weighted_present_pass_at_1 is not None:
        weighted_pass_at_1 = weighted_present_pass_at_1

    return {
        "tier_weights": normalized_weights,
        "tier_weights_raw": weights,
        "weighted_pass_at_1": weighted_pass_at_1,
        "weighted_present_pass_at_1": weighted_present_pass_at_1,
        "weighted_pass_at_1_complete": not missing_tiers,
        "missing_weighted_tiers": missing_tiers,
        "score_definition": "equal-weight tier macro average by default: (low + middle + high) / 3",
    }


def build_summary(rows: list[dict[str, Any]], *, args: argparse.Namespace, elapsed: float) -> dict[str, Any]:
    by_tier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_tier[str(row.get("agentic_swe_tier"))].append(row)
        by_source[str(row.get("source_benchmark"))].append(row)
    by_tier_summary = {
        key: summarize_group(value, model=args.model)
        for key, value in sorted(by_tier.items())
    }
    total = summarize_group(rows, model=args.model)
    scoring = build_scoring_summary(by_tier, args=args)
    return {
        "benchmark": "Agentic SWE-Assorted",
        "model": args.model,
        "elapsed_seconds": elapsed,
        "dry_run": bool(args.dry_run),
        "total": {
            **total,
            "micro_pass_at_1": total["pass_at_1"],
            "weighted_pass_at_1": scoring["weighted_pass_at_1"],
            "weighted_present_pass_at_1": scoring["weighted_present_pass_at_1"],
        },
        "scoring": scoring,
        "by_tier": by_tier_summary,
        "by_source": {key: summarize_group(value, model=args.model) for key, value in sorted(by_source.items())},
        "limits": {
            "max_input_tokens": args.max_input_tokens,
            "max_cumulative_input_tokens": args.max_cumulative_input_tokens,
            "high_max_cumulative_input_tokens": getattr(
                args, "high_max_cumulative_input_tokens", None
            ),
            "max_cumulative_output_tokens": args.max_cumulative_output_tokens,
            "max_tool_calls": args.max_tool_calls,
            "high_max_tool_calls": getattr(args, "high_max_tool_calls", None),
            "max_agent_turns": args.max_agent_turns,
            "high_max_agent_turns": getattr(args, "high_max_agent_turns", None),
            "max_tool_wall_seconds": args.max_tool_wall_seconds,
            "llm_response_idle_timeout_seconds": args.llm_response_idle_timeout_seconds,
            "swe_workers": args.swe_workers,
            "high_workers": args.high_workers,
        },
    }


def _format_pct(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value * 100:.1f}%"


def _format_usd(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"${value:.2f}"


def write_markdown_report(path: Path, *, summary: dict[str, Any]) -> None:
    scoring = summary.get("scoring", {}) if isinstance(summary.get("scoring"), dict) else {}
    total = summary.get("total", {}) if isinstance(summary.get("total"), dict) else {}
    by_tier = summary.get("by_tier", {}) if isinstance(summary.get("by_tier"), dict) else {}
    limits = summary.get("limits", {}) if isinstance(summary.get("limits"), dict) else {}

    lines = [
        "# Agentic SWE-Assorted Report",
        "",
        f"- model: `{summary.get('model')}`",
        f"- elapsed_seconds: `{summary.get('elapsed_seconds'):.1f}`"
        if isinstance(summary.get("elapsed_seconds"), (int, float))
        else "- elapsed_seconds: `n/a`",
        f"- official_score: `{_format_pct(scoring.get('weighted_pass_at_1'))}`",
        f"- micro_pass_at_1: `{_format_pct(total.get('micro_pass_at_1'))}`",
        f"- score_complete: `{bool(scoring.get('weighted_pass_at_1_complete'))}`",
        f"- estimated_cost: `{_format_usd((total.get('usage') or {}).get('cost_usd'))}`",
        "",
        "## Tier Results",
        "",
        "| tier | resolved | total | pass@1 | estimated cost | empty patches | failure reasons |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for tier in TIER_ORDER:
        tier_summary = by_tier.get(tier, {}) if isinstance(by_tier.get(tier), dict) else {}
        usage = tier_summary.get("usage") if isinstance(tier_summary.get("usage"), dict) else {}
        failures = tier_summary.get("failure_reasons") if isinstance(tier_summary.get("failure_reasons"), dict) else {}
        failure_text = ", ".join(f"{key}:{value}" for key, value in sorted(failures.items())) or "-"
        lines.append(
            "| "
            + " | ".join(
                [
                    tier,
                    str(tier_summary.get("resolved_instances", 0)),
                    str(tier_summary.get("total_instances", 0)),
                    _format_pct(tier_summary.get("pass_at_1")),
                    _format_usd(usage.get("cost_usd")),
                    str(tier_summary.get("empty_patches", 0)),
                    failure_text,
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Scoring",
            "",
            f"- definition: {scoring.get('score_definition')}",
            f"- tier_weights: `{json.dumps(scoring.get('tier_weights'), ensure_ascii=False, sort_keys=True)}`",
            f"- missing_weighted_tiers: `{scoring.get('missing_weighted_tiers')}`",
            "",
            "## Runtime Limits",
            "",
            "| limit | value |",
            "|---|---:|",
        ]
    )
    for key in (
        "max_agent_turns",
        "high_max_agent_turns",
        "max_tool_calls",
        "high_max_tool_calls",
        "max_cumulative_input_tokens",
        "high_max_cumulative_input_tokens",
        "max_cumulative_output_tokens",
        "max_tool_wall_seconds",
        "llm_response_idle_timeout_seconds",
        "swe_workers",
        "high_workers",
    ):
        lines.append(f"| {key} | `{limits.get(key)}` |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
                "pass_at_1": summary["scoring"]["weighted_pass_at_1"],
                "micro_pass_at_1": summary["total"]["micro_pass_at_1"],
                "weighted_present_pass_at_1": summary["scoring"]["weighted_present_pass_at_1"],
                "weighted_pass_at_1_complete": summary["scoring"]["weighted_pass_at_1_complete"],
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
        metrics = {
            "agentic_swe_leaderboard_table": wandb.Table(dataframe=leaderboard),
            "agentic_swe_output_table": wandb.Table(dataframe=table_df),
            "agentic_swe_results": summary,
            "agentic_swe/micro_pass_at_1": float(summary["total"]["micro_pass_at_1"]),
            "agentic_swe/weighted_present_pass_at_1": float(
                summary["scoring"]["weighted_present_pass_at_1"] or 0.0
            ),
            "agentic_swe/weighted_pass_at_1_complete": int(
                bool(summary["scoring"]["weighted_pass_at_1_complete"])
            ),
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
        if summary["scoring"]["weighted_pass_at_1"] is not None:
            metrics["agentic_swe/pass_at_1"] = float(summary["scoring"]["weighted_pass_at_1"])
            metrics["agentic_swe/weighted_pass_at_1"] = float(
                summary["scoring"]["weighted_pass_at_1"]
            )
        run.log(metrics)
        artifact = wandb.Artifact(
            "agentic-swe-assorted-" + args.model.replace("/", "-").replace(":", "-"),
            type="evaluation-results",
            metadata={
                "model": args.model,
                "total_instances": summary["total"]["total_instances"],
                "resolved_instances": summary["total"]["resolved_instances"],
                "pass_at_1": summary["scoring"]["weighted_pass_at_1"],
                "micro_pass_at_1": summary["total"]["micro_pass_at_1"],
            },
        )
        for filename in ("summary.json", "output_table.jsonl", "leaderboard_table.json", "report.md"):
            path = args.output_dir / filename
            if path.exists():
                artifact.add_file(str(path), name=filename)
        run.log_artifact(artifact, aliases=["latest"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--openclaw-model-params-json",
        "--openclaw-extra-body-json",
        dest="openclaw_model_params_json",
        help=(
            "JSON object forwarded to Low/Middle and High OpenClaw runners and "
            "merged into the selected model entry's params."
        ),
    )
    parser.add_argument(
        "--openclaw-model-overrides-json",
        dest="openclaw_model_overrides_json",
        help=(
            "JSON object forwarded to Low/Middle and High OpenClaw runners and "
            "merged into the selected model entry itself, e.g. {\"maxTokens\":4096}."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/agentic_swe_assorted"))
    parser.add_argument("--prefix", default="agentic-swe-assorted")
    parser.add_argument("--thinking", default="high")
    parser.add_argument("--agent", default="main")
    parser.add_argument("--low-middle-jsonl", type=Path, default=DEFAULT_LOW_MIDDLE_JSONL)
    parser.add_argument("--low-middle-instance-ids-json", type=Path, default=DEFAULT_LOW_MIDDLE_IDS)
    parser.add_argument("--low-middle-limit", type=int)
    parser.add_argument("--low-limit", type=int)
    parser.add_argument("--middle-limit", type=int)
    parser.add_argument("--deepswe-metadata-jsonl", type=Path, default=DEFAULT_DEEPSWE_META)
    parser.add_argument("--deepswe-task-names-file", type=Path, default=DEFAULT_DEEPSWE_TASK_NAMES)
    parser.add_argument("--deepswe-tasks-root", type=Path, default=REPO_ROOT / "external" / "deep-swe" / "tasks")
    parser.add_argument("--high-limit", type=int)
    parser.add_argument(
        "--deepswe-public-trials-json",
        type=Path,
        default=DEFAULT_DEEPSWE_PUBLIC_TRIALS,
        help="Cached public DeepSWE v1.1 trials.json used for High budget preflight.",
    )
    parser.add_argument(
        "--deepswe-public-model",
        help=(
            "Model key in public DeepSWE trials for budget preflight. Defaults to "
            "a normalized form of --model, e.g. openrouter-direct/z-ai/glm-5.2 -> glm-5-2."
        ),
    )
    parser.add_argument(
        "--deepswe-public-effort",
        help="Reasoning effort in public DeepSWE trials for budget preflight. Defaults to --thinking.",
    )
    parser.add_argument(
        "--deepswe-budget-preflight",
        choices=["error", "warn", "off"],
        default="error",
        help=(
            "Before running paid DeepSWE High tasks, compare selected tasks with "
            "public DeepSWE step/input-token stats. error blocks known budget mismatches."
        ),
    )
    parser.add_argument(
        "--deepswe-preflight-hard-stat",
        choices=["mean", "p50", "p75", "p90", "max"],
        default="p75",
        help="Public statistic that must fit the configured High turn/input caps.",
    )
    parser.add_argument(
        "--allow-deepswe-budget-mismatch",
        action="store_true",
        help="Bypass DeepSWE budget preflight errors for intentional experiments.",
    )
    parser.add_argument(
        "--tier-weights",
        default="low=1,middle=1,high=1",
        help=(
            "Official score weights as JSON object or comma-separated tier=value pairs. "
            "Default makes Low, Middle, and High contribute one third each."
        ),
    )
    parser.add_argument("--official-swebench-repo", type=Path, default=DEFAULT_OFFICIAL_SWEBENCH)
    parser.add_argument("--checkout-root", type=Path, default=Path("outputs/swebench_lite_checkouts"))
    parser.add_argument("--skip-low-middle", action="store_true")
    parser.add_argument("--skip-high", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-docker-check", action="store_true")
    parser.add_argument(
        "--min-free-disk-gb",
        type=float,
        default=DEFAULT_MIN_FREE_DISK_GB,
        help=(
            "Minimum free disk required on the output and Docker filesystems before "
            "paid Low/Middle generation starts. Set 0 to disable."
        ),
    )
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
    parser.add_argument(
        "--high-max-cumulative-input-tokens",
        type=int,
        default=12_000_000,
        help=(
            "DeepSWE High cumulative input-token cap. Defaults to 12M because "
            "the default cap-aware High subset is selected from public GLM-5.2 "
            "max rollouts and must pass p75 input-token preflight."
        ),
    )
    parser.add_argument("--max-cumulative-output-tokens", type=int, default=500_000)
    parser.add_argument("--max-tool-calls", type=int, default=40)
    parser.add_argument(
        "--high-max-tool-calls",
        type=int,
        default=120,
        help=(
            "DeepSWE High tool-call cap. Defaults above Low/Middle and slightly above "
            "the default High-8 public rollout step profile."
        ),
    )
    parser.add_argument("--max-agent-turns", type=int, default=40)
    parser.add_argument(
        "--high-max-agent-turns",
        type=int,
        default=120,
        help=(
            "DeepSWE High turn cap. Low/Middle keep --max-agent-turns; High defaults "
            "to 120 because the default High-8 is selected to fit public GLM-5.2 "
            "max p75 step/input-token budgets before paid execution."
        ),
    )
    parser.add_argument("--max-tool-wall-seconds", type=int, default=120)
    parser.add_argument(
        "--llm-response-idle-timeout-seconds",
        type=float,
        default=900.0,
        help=(
            "DeepSWE High watchdog: if the live session waits after the initial "
            "user prompt or a tool result without a new assistant response for "
            "this many seconds, interrupt as a provider idle timeout. 0 disables it."
        ),
    )
    parser.add_argument("--require-actual-token-usage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-task-agent", action=argparse.BooleanOptionalAction, default=True)
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
            r"\b(?:ftp|sftp|ssh)://",
            r"\bgit\+",
            r"\bgit\s+(?:clone|fetch|pull|ls-remote)\b",
            r"\b(curl|wget)\b",
        ],
    )
    args = parser.parse_args()
    args.deny_tool = dedupe_preserve_order(args.deny_tool)
    args.deny_argument_pattern = dedupe_preserve_order(args.deny_argument_pattern)
    return args


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.checkout_root = args.checkout_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()

    output_rows: list[dict[str, Any]] = []
    low_middle_source_rows: list[dict[str, Any]] = []
    high_metadata_rows: list[dict[str, Any]] = []
    low_middle_jsonl = None
    low_middle_ids = None
    high_task_names = None

    if not args.skip_low_middle:
        low_middle_jsonl, low_middle_ids, low_middle_source_rows = prepare_low_middle_inputs(args)

    if not args.skip_high:
        high_task_names, high_metadata_rows = prepare_high_inputs(args)
        deepswe_budget_preflight(args, high_metadata_rows)

    runtime_preflight(args, low_middle_rows=low_middle_source_rows, high_rows=high_metadata_rows)

    if not args.skip_low_middle:
        assert low_middle_jsonl is not None
        assert low_middle_ids is not None
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
        assert high_task_names is not None
        run_command(build_deepswe_command(args, high_task_names))
        high_results_path = args.output_dir / "high" / "deepswe" / "results.jsonl"
        high_result_rows = read_jsonl(high_results_path) if high_results_path.exists() else []
        output_rows.extend(deepswe_rows(metadata_rows=high_metadata_rows, result_rows=high_result_rows))

    validate_output_rows(output_rows)
    elapsed = time.time() - started_at
    summary = build_summary(output_rows, args=args, elapsed=elapsed)
    write_jsonl(args.output_dir / "output_table.jsonl", output_rows)
    write_json(args.output_dir / "summary.json", summary)
    write_markdown_report(args.output_dir / "report.md", summary=summary)
    write_json(
        args.output_dir / "leaderboard_table.json",
        [
            {
                "model_name": args.model,
                "total_samples": summary["total"]["total_instances"],
                "issues_resolved": summary["total"]["resolved_instances"],
                "pass_at_1": summary["scoring"]["weighted_pass_at_1"],
                "micro_pass_at_1": summary["total"]["micro_pass_at_1"],
                "weighted_present_pass_at_1": summary["scoring"]["weighted_present_pass_at_1"],
                "weighted_pass_at_1_complete": summary["scoring"]["weighted_pass_at_1_complete"],
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
