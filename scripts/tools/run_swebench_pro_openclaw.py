#!/usr/bin/env python3
"""
Generate SWE-bench Pro patches with OpenClaw on real repository checkouts.

The runner creates or resets a checkout for each SWE-bench Pro instance at
`base_commit`, runs the Nejumi OpenClaw protocol in that checkout, then captures
`git diff --binary` plus untracked intent-to-add files as the model patch.
The output JSON is compatible with Scale's `swe_bench_pro_eval.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_RUNNER = REPO_ROOT / "scripts" / "tools" / "run_openclaw_agent_protocol.py"
RUNNER_VERSION = "swebench-pro-openclaw-2026-07-02-sandbox-live-budget-v1"
PATCH_CAPTURE_VERSION = "git-diff-with-untracked-excluding-selected-tests-v2"
DEFAULT_MAX_INPUT_TOKENS = 1_000_000
DEFAULT_MAX_TOOL_CALLS = 60
OPENCLAW_RUNTIME_DIR = ".nejumi_openclaw"
RUNTIME_EXCLUDED_PATHS = [OPENCLAW_RUNTIME_DIR]
NEMOCLAW_OPENCLAW_CONFIG_PATH = "/sandbox/.openclaw/openclaw.json"
NEMOCLAW_TRANSFER_CHUNK_BYTES = 512 * 1024
DEFAULT_DENIED_TOOLS = [
    "code_execution",
    "web_search",
    "web_fetch",
    "browser",
    "browser_*",
    "*search*",
]
DEFAULT_DENIED_ARGUMENT_PATTERNS = [
    r"https?://",
    r"\b(curl|wget)\b",
    r"\b(requests|urllib|httpx)\.",
]


def run_command(
    command: list[str],
    cwd: Path | None = None,
    timeout: int | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        result = subprocess.CompletedProcess(
            command,
            124,
            stdout=stdout,
            stderr=stderr + f"\nCommand timed out after {timeout} seconds",
        )
    if check and result.returncode != 0:
        raise RuntimeError(
            "Command failed\n"
            f"cmd: {' '.join(shlex.quote(part) for part in command)}\n"
            f"cwd: {cwd}\n"
            f"returncode: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe_id(instance_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in instance_id)


def safe_agent_id(instance_id: str, prefix: str) -> str:
    digest = hashlib.sha1(instance_id.encode("utf-8")).hexdigest()[:12]
    normalized_prefix = "".join(
        ch.lower() if ch.isalnum() else "-" for ch in prefix.strip()
    ).strip("-")
    return f"{normalized_prefix or 'nejumi-swe'}-{digest}"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def command_sha256(command: list[str]) -> str:
    payload = json.dumps(command, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_session_prefix(args: argparse.Namespace, default_prefix: str = "swebench-pro") -> str:
    configured = str(getattr(args, "session_prefix", "") or "").strip()
    prefix = configured or default_prefix
    wandb_run_id = os.environ.get("WANDB_RUN_ID", "").strip()
    if "{wandb_run_id}" in prefix:
        return prefix.replace("{wandb_run_id}", wandb_run_id or "no-wandb-run-id")
    if wandb_run_id and wandb_run_id not in prefix:
        return f"{wandb_run_id}:{prefix}"
    return prefix


def build_cache_key(row: dict[str, Any], prompt_text: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "runner_version": RUNNER_VERSION,
        "instance_id": str(row["instance_id"]),
        "prompt_hash": sha256_text(prompt_text),
        "model": args.model or "",
        "thinking": args.thinking,
        "deny_tools": effective_deny_tools(args),
        "deny_argument_patterns": effective_deny_argument_patterns(args),
        "openclaw_config_source": openclaw_config_cache_source(args),
        "max_input_tokens": int(getattr(args, "max_input_tokens", 0) or 0),
        "max_tool_calls": int(getattr(args, "max_tool_calls", 0) or 0),
        "nemoclaw_sandbox": str(getattr(args, "nemoclaw_sandbox", "") or ""),
        "nemoclaw_checkout_sandbox_root": str(
            getattr(args, "nemoclaw_checkout_sandbox_root", "") or ""
        ),
        "nemoclaw_checkout_transfer_mode": str(
            getattr(args, "nemoclaw_checkout_transfer_mode", "visible") or "visible"
        ),
        "session_prefix": resolve_session_prefix(args),
    }


def cache_key_matches(record: dict[str, Any], cache_key: dict[str, Any]) -> bool:
    return record.get("cache_key") == cache_key


def cache_requires_nemoclaw_session_audit(cache_key: dict[str, Any]) -> bool:
    return bool(cache_key.get("nemoclaw_sandbox"))


def sidecar_identity_matches_cache(sidecar: dict[str, Any], cache_key: dict[str, Any]) -> bool:
    metadata = sidecar.get("metadata") if isinstance(sidecar.get("metadata"), dict) else {}
    policy = sidecar.get("tool_policy") if isinstance(sidecar.get("tool_policy"), dict) else {}
    expected_config = cache_key.get("openclaw_config_source")
    observed_config = metadata.get("openclaw_config_source")
    return (
        metadata.get("task_id") == cache_key["instance_id"]
        and metadata.get("prompt_hash") == cache_key["prompt_hash"]
        and (not cache_key.get("model") or metadata.get("model_id") == cache_key.get("model"))
        and (not expected_config or observed_config == expected_config)
        and sorted(policy.get("deny_tools") or []) == sorted(cache_key.get("deny_tools") or [])
        and sorted(policy.get("deny_argument_patterns") or [])
        == sorted(cache_key.get("deny_argument_patterns") or [])
    )


def sidecar_nemoclaw_session_audit_matches_cache(sidecar: dict[str, Any], cache_key: dict[str, Any]) -> bool:
    audit = sidecar.get("nemoclaw_session_audit")
    if cache_requires_nemoclaw_session_audit(cache_key):
        return (
            isinstance(audit, dict)
            and audit.get("required") is True
            and audit.get("ok") is True
        )
    if isinstance(audit, dict) and audit.get("required") is True:
        return audit.get("ok") is True
    return True


def patch_record_nemoclaw_session_audit_matches_cache(record: dict[str, Any], cache_key: dict[str, Any]) -> bool:
    audit = record.get("nemoclaw_session_audit")
    if cache_requires_nemoclaw_session_audit(cache_key):
        return (
            isinstance(audit, dict)
            and audit.get("required") is True
            and audit.get("ok") is True
            and record.get("nemoclaw_session_audit_ok") is True
        )
    if isinstance(audit, dict) and audit.get("required") is True:
        return audit.get("ok") is True and record.get("nemoclaw_session_audit_ok") is True
    return True


def nemoclaw_session_copy_evidence(sidecar: dict[str, Any] | None) -> dict[str, Any]:
    audit = sidecar.get("nemoclaw_session_audit") if isinstance(sidecar, dict) else None
    copy_status = audit.get("copy") if isinstance(audit, dict) else None
    return {
        "nemoclaw_session_copy_source": (
            copy_status.get("source") if isinstance(copy_status, dict) else None
        ),
        "nemoclaw_session_copied_bytes": (
            audit.get("copied_session_bytes") if isinstance(audit, dict) else None
        ),
    }


def patch_record_conversation_order_allows_reuse(record: dict[str, Any]) -> bool:
    order = record.get("conversation_order")
    if record.get("conversation_order_ok") is False:
        return False
    if isinstance(order, dict) and order.get("ok") is False:
        return False
    return True


def patch_record_tool_policy_allows_reuse(record: dict[str, Any]) -> bool:
    if record.get("tool_policy_ok") is False:
        return False
    if record.get("tool_policy_violations"):
        return False
    return True


def patch_record_weave_sidecar_allows_reuse(record: dict[str, Any]) -> bool:
    sidecar = record.get("weave_sidecar")
    if record.get("weave_sidecar_ok") is False:
        return False
    if isinstance(sidecar, dict) and sidecar.get("ok") is False:
        return False
    return True


def patch_record_invocation_matches_cache(record: dict[str, Any], cache_key: dict[str, Any]) -> bool:
    if not cache_requires_nemoclaw_session_audit(cache_key):
        return True
    invocation_path_value = record.get("openclaw_invocation_path")
    invocation_sha = record.get("openclaw_invocation_sha256")
    command_sha = record.get("openclaw_command_sha256")
    if not isinstance(invocation_path_value, str) or not invocation_path_value:
        return False
    if not isinstance(invocation_sha, str) or len(invocation_sha) != 64:
        return False
    path = Path(invocation_path_value)
    if not path.exists():
        return False
    try:
        if sha256_file(path) != invocation_sha:
            return False
        invocation = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(invocation, dict):
        return False
    if invocation.get("cache_key") != cache_key:
        return False
    if isinstance(command_sha, str) and command_sha:
        if invocation.get("command_sha256") != command_sha:
            return False
    elif invocation.get("command"):
        return False
    expected_path = record.get("openclaw_result_path")
    if isinstance(expected_path, str) and expected_path:
        return invocation.get("expected_openclaw_result_path") == expected_path
    return True


def patch_mentions_paths(patch: str, paths: list[str]) -> bool:
    for path in paths:
        if f" a/{path} " in patch or f" b/{path}" in patch:
            return True
        if f"a/{path}\n" in patch or f"b/{path}\n" in patch:
            return True
    return False


def load_cached_patch_record(
    task_dir: Path,
    cache_key: dict[str, Any],
    prefix: str,
    excluded_paths: list[str] | None = None,
) -> dict[str, Any] | None:
    record_path = task_dir / "patch_record.json"
    if not record_path.exists():
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict) or not cache_key_matches(record, cache_key):
        return None
    if record.get("instance_id") != cache_key["instance_id"] or "patch" not in record:
        return None
    if not patch_record_nemoclaw_session_audit_matches_cache(record, cache_key):
        return None
    if not patch_record_conversation_order_allows_reuse(record):
        return None
    if not patch_record_tool_policy_allows_reuse(record):
        return None
    if not patch_record_weave_sidecar_allows_reuse(record):
        return None
    if not patch_record_invocation_matches_cache(record, cache_key):
        return None
    if not record.get("patch") and record.get("patch_capture_version") != PATCH_CAPTURE_VERSION:
        return None
    if (
        record.get("patch_capture_version") != PATCH_CAPTURE_VERSION
        and patch_mentions_paths(str(record.get("patch") or ""), excluded_paths or [])
    ):
        return None
    cached = dict(record)
    cached["prefix"] = prefix
    return cached


def task_sidecar_path(protocol_output_dir: Path, instance_id: str) -> Path:
    return protocol_output_dir / "agentic_swe" / instance_id / "openclaw_result.json"


def effective_deny_tools(args: argparse.Namespace) -> list[str]:
    return sorted(set(str(item) for item in (getattr(args, "deny_tool", None) or DEFAULT_DENIED_TOOLS)))


def effective_deny_argument_patterns(args: argparse.Namespace) -> list[str]:
    return sorted(
        set(str(item) for item in (getattr(args, "deny_argument_pattern", None) or DEFAULT_DENIED_ARGUMENT_PATTERNS))
    )


def run_nemoclaw_text_command(
    args: argparse.Namespace,
    command: list[str],
    *,
    input_text: str | None = None,
    timeout: int = 60,
    check: bool = True,
    workdir: str = "/sandbox",
) -> subprocess.CompletedProcess[str]:
    full_command = [
        args.nemoclaw_bin,
        "sandbox",
        "exec",
        args.nemoclaw_sandbox,
        "--workdir",
        workdir,
        "--no-tty",
        "--timeout",
        str(timeout),
        "--",
        *command,
    ]
    result = subprocess.run(
        full_command,
        text=True,
        input=input_text,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            "NeMoClaw command failed\n"
            f"cmd: {' '.join(shlex.quote(part) for part in full_command)}\n"
            f"returncode: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def run_nemoclaw_binary_command(
    args: argparse.Namespace,
    command: list[str],
    *,
    input_bytes: bytes | None = None,
    timeout: int = 60,
    check: bool = True,
    workdir: str = "/sandbox",
) -> subprocess.CompletedProcess[bytes]:
    full_command = [
        args.nemoclaw_bin,
        "sandbox",
        "exec",
        args.nemoclaw_sandbox,
        "--workdir",
        workdir,
        "--no-tty",
        "--timeout",
        str(timeout),
        "--",
        *command,
    ]
    result = subprocess.run(
        full_command,
        input=input_bytes,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            "NeMoClaw command failed\n"
            f"cmd: {' '.join(shlex.quote(part) for part in full_command)}\n"
            f"returncode: {result.returncode}\n"
            f"stdout:\n{result.stdout.decode('utf-8', errors='replace')}\n"
            f"stderr:\n{result.stderr.decode('utf-8', errors='replace')}"
        )
    return result


def read_openclaw_config_template(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.openclaw_config_template:
        template_path = args.openclaw_config_template.expanduser()
        return json.loads(template_path.read_text(encoding="utf-8")), str(template_path)
    if getattr(args, "nemoclaw_sandbox", None):
        sandbox_path = getattr(args, "nemoclaw_openclaw_config_path", NEMOCLAW_OPENCLAW_CONFIG_PATH)
        result = run_nemoclaw_text_command(args, ["cat", sandbox_path], timeout=60)
        return json.loads(result.stdout), sandbox_path
    template_path = default_openclaw_config_template()
    return json.loads(template_path.read_text(encoding="utf-8")), str(template_path)


def openclaw_config_cache_source(args: argparse.Namespace) -> str:
    template = getattr(args, "openclaw_config_template", None)
    if template:
        return str(Path(template).expanduser())
    if getattr(args, "nemoclaw_sandbox", None):
        return str(getattr(args, "nemoclaw_openclaw_config_path", NEMOCLAW_OPENCLAW_CONFIG_PATH))
    return str(default_openclaw_config_template())


def is_weave_sidecar_failure(sidecar: dict[str, Any] | None) -> bool:
    if not isinstance(sidecar, dict):
        return False
    weave_sidecar = sidecar.get("weave_sidecar")
    return (
        isinstance(weave_sidecar, dict)
        and weave_sidecar.get("ok") is False
        and sidecar.get("returncode") == 0
    )


def is_runtime_budget_exceeded(sidecar: dict[str, Any] | None) -> bool:
    if not isinstance(sidecar, dict):
        return False
    runtime_budget = sidecar.get("runtime_budget")
    return (
        isinstance(runtime_budget, dict)
        and runtime_budget.get("ok") is False
        and bool(runtime_budget.get("violations"))
    )


TRANSIENT_OPENCLAW_FAILURE_PATTERNS = [
    r"\bLLM request timed out\b",
    r"\bFailoverError:\s*LLM request timed out\b",
    r"\breason=timeout\b",
    r"\brawError=terminated\b",
    r"\bETIMEDOUT\b",
    r"\bECONNRESET\b",
    r"\bsocket hang up\b",
    r"\b429\b",
    r"\brate limit(?:ed)?\b",
    r"\brate[_ -]?limit(?:ed)?\b",
    r"\bJSON error injected into SSE stream\b",
    r"\btemporarily unavailable\b",
    r"\b503\b",
    r"\b504\b",
    r"\bgateway timeout\b",
]

NON_SCOREABLE_OPENCLAW_FAILURE_PATTERNS = [
    ("unsupported_thinking", r'Thinking level "[^"]+" is not supported'),
    ("insufficient_quota", r"\binsufficient_quota\b|exceeded your current quota"),
    ("authentication", r"\b401\b|unauthorized|invalid[_ -]?api[_ -]?key|incorrect api key|User not found"),
    ("workspace_vanished", r"\bWorkspaceVanishedError\b|workspace appears to have disappeared"),
    ("unknown_model", r"\bmodel .*not found\b|\bunknown model\b|\bNo provider\b"),
    ("conversation_order_violation", r"\bConversation order violation\b"),
    ("nemoclaw_session_audit_failed", r"\bNeMoClaw session audit failed\b"),
]
WORKSPACE_VANISHED_RE = re.compile(
    r"WorkspaceVanishedError:.*?: (?P<workspace>/[^\n]+?)\. Refusing.*?remove (?P<attestation>/[^\s]+\.attested)",
    flags=re.IGNORECASE | re.DOTALL,
)


def openclaw_failure_text(result: subprocess.CompletedProcess[str], sidecar: dict[str, Any] | None) -> str:
    parts = [result.stdout or "", result.stderr or ""]
    if isinstance(sidecar, dict):
        parts.extend([str(sidecar.get("stderr") or ""), str(sidecar.get("error") or "")])
    return "\n".join(part for part in parts if part)


def sidecar_error_text(sidecar: dict[str, Any] | None, fallback: str) -> str:
    if sidecar:
        stderr = str(sidecar.get("stderr") or "").strip()
        if stderr:
            return stderr[-4000:]
    return fallback[-4000:]


def is_transient_openclaw_failure(
    result: subprocess.CompletedProcess[str],
    sidecar: dict[str, Any] | None,
) -> bool:
    if result.returncode == 0:
        return False
    if isinstance(sidecar, dict) and (
        sidecar.get("tool_policy_ok") is False or sidecar.get("tool_policy_violations")
    ):
        return False
    text = openclaw_failure_text(result, sidecar)
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in TRANSIENT_OPENCLAW_FAILURE_PATTERNS)


def non_scoreable_openclaw_failure_reason(
    result: subprocess.CompletedProcess[str],
    sidecar: dict[str, Any] | None,
) -> str | None:
    """Return a setup/provider failure reason that must not become a benchmark patch."""
    if result.returncode == 0:
        return None
    text = openclaw_failure_text(result, sidecar)
    for reason, pattern in NON_SCOREABLE_OPENCLAW_FAILURE_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return reason
    return None


def append_transient_failure(task_dir: Path, payload: dict[str, Any]) -> None:
    with (task_dir / "openclaw_transient_failures.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def recover_workspace_vanished_failure(
    task_dir: Path,
    result: subprocess.CompletedProcess[str],
    sidecar: dict[str, Any] | None,
) -> dict[str, Any] | None:
    text = openclaw_failure_text(result, sidecar)
    match = WORKSPACE_VANISHED_RE.search(text)
    if not match:
        return None
    workspace = Path(match.group("workspace")).resolve()
    attestation = Path(match.group("attestation")).resolve()
    attestation_root = (Path.home() / ".openclaw" / "workspace-attestations").resolve()
    try:
        attestation.relative_to(attestation_root)
    except ValueError:
        return None
    if not workspace.exists() or not attestation.exists():
        return None
    recovery_dir = task_dir / "recovered_workspace_attestations"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    destination = recovery_dir / f"{int(time.time())}_{attestation.name}"
    attestation.replace(destination)
    return {
        "workspace": str(workspace),
        "attestation": str(attestation),
        "moved_to": str(destination),
    }


def sidecar_usage(sidecar: dict[str, Any]) -> dict[str, Any]:
    stdout_json = sidecar.get("stdout_json")
    if not isinstance(stdout_json, dict):
        return {}
    return stdout_json.get("meta", {}).get("agentMeta", {}).get("usage", {})


def repo_url(repo: str) -> str:
    if repo.startswith("http://") or repo.startswith("https://") or repo.endswith(".git"):
        return repo
    return f"https://github.com/{repo}.git"


def prepare_checkout(row: dict[str, Any], checkout_root: Path, reset: bool) -> Path:
    instance_id = str(row["instance_id"])
    checkout_dir = checkout_root / safe_id(instance_id)
    checkout_root.mkdir(parents=True, exist_ok=True)

    new_checkout = False
    if not checkout_dir.exists():
        run_command(["git", "clone", repo_url(str(row["repo"])), str(checkout_dir)])
        new_checkout = True

    if reset or new_checkout:
        run_command(["git", "fetch", "--all", "--tags", "--prune"], cwd=checkout_dir)
        run_command(["git", "reset", "--hard"], cwd=checkout_dir)
        run_command(["git", "clean", "-fdx"], cwd=checkout_dir)
    else:
        return checkout_dir

    base_commit = str(row["base_commit"])
    checkout = run_command(["git", "checkout", "--force", base_commit], cwd=checkout_dir, check=False)
    if checkout.returncode != 0:
        run_command(["git", "fetch", "origin", base_commit], cwd=checkout_dir)
        run_command(["git", "checkout", "--force", base_commit], cwd=checkout_dir)
    run_command(["git", "reset", "--hard", base_commit], cwd=checkout_dir)
    run_command(["git", "clean", "-fdx"], cwd=checkout_dir)
    return checkout_dir


def list_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(f"- {item}" for item in value)
    if isinstance(value, tuple):
        return "\n".join(f"- {item}" for item in value)
    return str(value)


def build_prompt(row: dict[str, Any]) -> str:
    issue_categories = list_text(row.get("issue_categories"))
    selected_tests = list_text(row.get("selected_test_files_to_run"))
    requirements = str(row.get("requirements") or "").strip()
    interface = str(row.get("interface") or "").strip()

    parts = [
        "# SWE-bench Pro Task",
        "",
        "You are running inside a checkout of the target repository at the base commit.",
        "Inspect the repository, edit files as needed, and leave the working tree with the minimal fix.",
        "Do not create commits. The harness will collect `git diff --binary` after you finish.",
        "",
        "## Metadata",
        "",
        f"- instance_id: {row.get('instance_id')}",
        f"- repo: {row.get('repo')}",
        f"- base_commit: {row.get('base_commit')}",
        f"- language: {row.get('repo_language')}",
        f"- issue_specificity: {row.get('issue_specificity')}",
        f"- docker image tag: {row.get('dockerhub_tag')}",
        "",
        "## Issue",
        "",
        str(row.get("problem_statement") or "").strip(),
    ]
    if requirements:
        parts.extend(["", "## Requirements", "", requirements])
    if interface:
        parts.extend(["", "## Interface", "", interface])
    if issue_categories:
        parts.extend(["", "## Issue Categories", "", issue_categories])
    if selected_tests:
        parts.extend(["", "## Selected Test Files", "", selected_tests])
    before_cmd = str(row.get("before_repo_set_cmd") or "").strip()
    if before_cmd:
        parts.extend(["", "## Evaluation Setup Hint", "", before_cmd])
    parts.extend(
        [
            "",
            "## Output Contract",
            "",
            "Finish with the repository files edited. The benchmark runner, not you, will collect the patch.",
            "Keep the change minimal and avoid unrelated formatting or dependency churn.",
            "Use local shell execution in the target checkout; do not use remote code interpreter tools.",
            "Do not use web search, browser, HTTP, or any external internet lookup.",
        ]
    )
    return "\n".join(parts) + "\n"


def selected_test_paths(row: dict[str, Any]) -> list[str]:
    raw = row.get("selected_test_files_to_run") or []
    if isinstance(raw, str):
        raw = [raw]
    paths: list[str] = []
    for item in raw:
        path = str(item).split("::", 1)[0].strip().lstrip("./")
        if path:
            paths.append(path)
    return sorted(set(paths))


def _is_excluded_path(path: str, excluded_paths: list[str]) -> bool:
    normalized = path.strip().lstrip("./")
    for excluded in excluded_paths:
        excluded = excluded.rstrip("/")
        if normalized == excluded or normalized.startswith(f"{excluded}/"):
            return True
    return False


def capture_patch(checkout_dir: Path, excluded_paths: list[str] | None = None) -> str:
    excluded_paths = sorted(set((excluded_paths or []) + RUNTIME_EXCLUDED_PATHS))
    if excluded_paths:
        for start in range(0, len(excluded_paths), 200):
            run_command(
                ["git", "reset", "-q", "--", *excluded_paths[start : start + 200]],
                cwd=checkout_dir,
                check=False,
            )
    untracked = run_command(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=checkout_dir,
    )
    paths = [
        path
        for path in untracked.stdout.split("\0")
        if path and not _is_excluded_path(path, excluded_paths)
    ]
    for start in range(0, len(paths), 200):
        run_command(["git", "add", "-N", "--", *paths[start : start + 200]], cwd=checkout_dir)
    diff_command = ["git", "diff", "--binary"]
    if excluded_paths:
        diff_command.extend(["--", ".", *[f":(exclude){path}" for path in excluded_paths]])
    result = run_command(diff_command, cwd=checkout_dir)
    return result.stdout


def _filter_untracked_paths(raw: str, excluded_paths: list[str]) -> list[str]:
    return [
        path
        for path in raw.split("\0")
        if path and not _is_excluded_path(path, excluded_paths)
    ]


def capture_patch_nemoclaw(
    checkout_dir: Path,
    args: argparse.Namespace,
    excluded_paths: list[str] | None = None,
) -> str:
    excluded_paths = sorted(set((excluded_paths or []) + RUNTIME_EXCLUDED_PATHS))
    sandbox_dir = str(sandbox_checkout_dir(checkout_dir, args))
    if excluded_paths:
        for start in range(0, len(excluded_paths), 200):
            run_nemoclaw_text_command(
                args,
                [
                    "bash",
                    "-lc",
                    'git reset -q -- "$@"',
                    "git-reset",
                    *excluded_paths[start : start + 200],
                ],
                workdir=sandbox_dir,
                check=False,
            )
    untracked = run_nemoclaw_text_command(
        args,
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        workdir=sandbox_dir,
    )
    paths = _filter_untracked_paths(untracked.stdout, excluded_paths)
    for start in range(0, len(paths), 200):
        run_nemoclaw_text_command(
            args,
            [
                "bash",
                "-lc",
                'git add -N -- "$@"',
                "git-add",
                *paths[start : start + 200],
            ],
            workdir=sandbox_dir,
        )
    diff_paths = ["."]
    if excluded_paths:
        diff_paths.extend([f":(exclude){path}" for path in excluded_paths])
    result = run_nemoclaw_text_command(
        args,
        ["bash", "-lc", 'git diff --binary -- "$@"', "git-diff", *diff_paths],
        workdir=sandbox_dir,
    )
    return result.stdout


def should_force_empty_patch(openclaw_metadata: dict[str, Any]) -> bool:
    return bool(openclaw_metadata.get("openclaw_disqualified_reason"))


def default_openclaw_config_template() -> Path:
    return Path(os.environ.get("OPENCLAW_CONFIG_PATH", "~/.openclaw/openclaw.json")).expanduser()


def sandbox_checkout_dir(checkout_dir: Path, args: argparse.Namespace) -> Path:
    sandbox_root = getattr(args, "nemoclaw_checkout_sandbox_root", None)
    if sandbox_root:
        return Path(str(sandbox_root)) / checkout_dir.name
    return checkout_dir.resolve()


def nemoclaw_checkout_visible(checkout_dir: Path, args: argparse.Namespace) -> bool:
    if not getattr(args, "nemoclaw_sandbox", None):
        return True
    sandbox_dir = str(sandbox_checkout_dir(checkout_dir, args))
    if "\n" in sandbox_dir or "\r" in sandbox_dir:
        raise RuntimeError(f"Invalid sandbox checkout path: {sandbox_dir!r}")
    result = run_nemoclaw_text_command(
        args,
        ["bash", "-lc", 'test -d "$1"', "check-checkout", sandbox_dir],
        timeout=30,
        check=False,
    )
    return result.returncode == 0


def assert_nemoclaw_checkout_visible(checkout_dir: Path, args: argparse.Namespace) -> None:
    if not getattr(args, "nemoclaw_sandbox", None):
        return
    if not nemoclaw_checkout_visible(checkout_dir, args):
        raise RuntimeError(
            "SWE-Bench Pro checkout is not visible inside the NeMoClaw sandbox: "
            f"{sandbox_checkout_dir(checkout_dir, args)}. Mount or prepare the checkout under the sandbox-visible root "
            "before running the agent."
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_checkout_archive(checkout_dir: Path, archive_path: Path) -> None:
    run_command(
        [
            "tar",
            "-C",
            str(checkout_dir),
            "--exclude",
            f"./{OPENCLAW_RUNTIME_DIR}",
            "-czf",
            str(archive_path),
            ".",
        ],
        cwd=REPO_ROOT,
    )


def upload_file_to_nemoclaw(
    local_path: Path,
    sandbox_path: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    host_sha = sha256_file(local_path)
    run_nemoclaw_text_command(
        args,
        ["bash", "-lc", 'mkdir -p "$(dirname "$1")" && : > "$1"', "init-upload", sandbox_path],
        timeout=30,
    )
    chunk_count = 0
    with local_path.open("rb") as f:
        while True:
            chunk = f.read(NEMOCLAW_TRANSFER_CHUNK_BYTES)
            if not chunk:
                break
            run_nemoclaw_binary_command(
                args,
                ["bash", "-lc", 'cat >> "$1"', "append-upload", sandbox_path],
                input_bytes=chunk,
                timeout=30,
            )
            chunk_count += 1
    result = run_nemoclaw_text_command(args, ["sha256sum", sandbox_path], timeout=60)
    sandbox_sha = result.stdout.strip().split()[0] if result.stdout.strip() else ""
    if sandbox_sha != host_sha:
        raise RuntimeError(
            "Transferred NeMoClaw checkout archive sha256 mismatch: "
            f"host={host_sha} sandbox={sandbox_sha}"
        )
    return {
        "archive_bytes": local_path.stat().st_size,
        "archive_sha256": host_sha,
        "chunk_bytes": NEMOCLAW_TRANSFER_CHUNK_BYTES,
        "chunk_count": chunk_count,
        "sandbox_archive": sandbox_path,
    }


def sync_checkout_to_nemoclaw_copy(
    checkout_dir: Path,
    task_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    sandbox_dir = str(sandbox_checkout_dir(checkout_dir, args))
    transfer_dir = task_dir / "nemoclaw_checkout_transfer"
    transfer_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=transfer_dir) as tmp_name:
        archive_path = Path(tmp_name) / "checkout.tgz"
        create_checkout_archive(checkout_dir, archive_path)
        sandbox_archive = f"/sandbox/tmp/nejumi_swe_checkout/{safe_id(checkout_dir.name)}.tgz"
        upload = upload_file_to_nemoclaw(archive_path, sandbox_archive, args)
    run_nemoclaw_text_command(
        args,
        [
            "bash",
            "-lc",
            'rm -rf "$1" && mkdir -p "$1" && tar -xzf "$2" -C "$1"',
            "extract-checkout",
            sandbox_dir,
            sandbox_archive,
        ],
        timeout=max(60, int(getattr(args, "nemoclaw_checkout_transfer_timeout", 300))),
    )
    return {
        "mode": "copy",
        "sandbox_checkout_dir": sandbox_dir,
        **upload,
    }


def ensure_nemoclaw_checkout_ready(
    checkout_dir: Path,
    task_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if not getattr(args, "nemoclaw_sandbox", None):
        return None
    if nemoclaw_checkout_visible(checkout_dir, args):
        return {"mode": "visible", "sandbox_checkout_dir": str(sandbox_checkout_dir(checkout_dir, args))}
    mode = str(getattr(args, "nemoclaw_checkout_transfer_mode", "visible") or "visible")
    if mode != "copy":
        assert_nemoclaw_checkout_visible(checkout_dir, args)
    return sync_checkout_to_nemoclaw_copy(checkout_dir, task_dir, args)


def task_openclaw_config_paths(
    checkout_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path]:
    host_runtime_dir = checkout_dir / OPENCLAW_RUNTIME_DIR
    host_config_path = host_runtime_dir / "openclaw_config.json"
    host_agent_dir = host_runtime_dir / "agent_state"
    if getattr(args, "nemoclaw_sandbox", None):
        sandbox_runtime_dir = sandbox_checkout_dir(checkout_dir, args) / OPENCLAW_RUNTIME_DIR
        sandbox_config_path = sandbox_runtime_dir / "openclaw_config.json"
        sandbox_agent_dir = sandbox_runtime_dir / "agent_state"
    else:
        sandbox_config_path = host_config_path
        sandbox_agent_dir = host_agent_dir
    return host_config_path, host_agent_dir, sandbox_config_path, sandbox_agent_dir


def task_live_session_dir(
    checkout_dir: Path,
    task_dir: Path,
    args: argparse.Namespace,
) -> Path | None:
    if not getattr(args, "use_task_agent", True):
        return None
    if getattr(args, "nemoclaw_sandbox", None):
        transfer_mode = str(
            getattr(args, "nemoclaw_checkout_transfer_mode", "visible") or "visible"
        )
        if transfer_mode != "visible":
            return None
        _, host_agent_dir, _, _ = task_openclaw_config_paths(checkout_dir, args)
        return host_agent_dir / "sessions"
    return task_dir / "openclaw_agent_state" / "sessions"


def task_live_sandbox_session_dir(checkout_dir: Path, args: argparse.Namespace) -> str | None:
    if not getattr(args, "use_task_agent", True):
        return None
    if not getattr(args, "nemoclaw_sandbox", None):
        return None
    transfer_mode = str(getattr(args, "nemoclaw_checkout_transfer_mode", "visible") or "visible")
    if transfer_mode == "visible":
        return None
    _, _, _, sandbox_agent_dir = task_openclaw_config_paths(checkout_dir, args)
    return str(sandbox_agent_dir / "sessions")


def write_task_openclaw_config(
    row: dict[str, Any],
    checkout_dir: Path,
    task_dir: Path,
    args: argparse.Namespace,
) -> tuple[str, Path | None]:
    if not args.use_task_agent:
        return args.agent, None
    if args.no_local and not getattr(args, "nemoclaw_sandbox", None):
        raise RuntimeError("--use-task-agent requires local OpenClaw execution; remove --no-local")

    config, template_path = read_openclaw_config_template(args)
    disable_remote_lookup_tools(config)
    agent_id = safe_agent_id(str(row["instance_id"]), args.task_agent_prefix)
    if getattr(args, "nemoclaw_sandbox", None):
        host_config_path, host_agent_dir, sandbox_config_path, sandbox_agent_dir = task_openclaw_config_paths(
            checkout_dir,
            args,
        )
        workspace = str(sandbox_checkout_dir(checkout_dir, args))
    else:
        host_config_path = task_dir / "openclaw_config.json"
        host_agent_dir = task_dir / "openclaw_agent_state"
        sandbox_config_path = host_config_path
        sandbox_agent_dir = host_agent_dir
        workspace = str(checkout_dir.resolve())
    host_config_path.parent.mkdir(parents=True, exist_ok=True)
    agent_entry = {
        "id": agent_id,
        "workspace": workspace,
        "agentDir": str(sandbox_agent_dir),
        "tools": {
            "profile": args.openclaw_tool_profile,
            "deny": effective_deny_tools(args),
        },
    }

    agents = config.setdefault("agents", {})
    entries = agents.setdefault("list", [])
    if not isinstance(entries, list):
        raise RuntimeError("OpenClaw config field agents.list must be a list")
    entries[:] = [entry for entry in entries if not (isinstance(entry, dict) and entry.get("id") == agent_id)]
    entries.append(agent_entry)

    config_text = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    host_config_path.write_text(config_text, encoding="utf-8")
    if getattr(args, "nemoclaw_sandbox", None) and str(
        getattr(args, "nemoclaw_checkout_transfer_mode", "visible") or "visible"
    ) == "copy":
        run_nemoclaw_text_command(
            args,
            [
                "python3",
                "-c",
                (
                    "import sys; from pathlib import Path; "
                    "path = Path(sys.argv[1]); path.parent.mkdir(parents=True, exist_ok=True); "
                    "path.write_text(sys.stdin.read(), encoding='utf-8')"
                ),
                str(sandbox_config_path),
            ],
            input_text=config_text,
            timeout=60,
        )
    (task_dir / "openclaw_task_agent.json").write_text(
        json.dumps(
            {
                "agent_id": agent_id,
                "workspace": agent_entry["workspace"],
                "agent_dir": agent_entry["agentDir"],
                "host_config_path": str(host_config_path),
                "host_agent_dir": str(host_agent_dir),
                "sandbox_config_path": str(sandbox_config_path),
                "sandbox_agent_dir": str(sandbox_agent_dir),
                "config_template": template_path,
                "config_path": str(sandbox_config_path),
                "nemoclaw_sandbox": getattr(args, "nemoclaw_sandbox", None),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return agent_id, sandbox_config_path


def disable_remote_lookup_tools(config: dict[str, Any]) -> None:
    tools = config.setdefault("tools", {})
    if not isinstance(tools, dict):
        return
    tools["toolSearch"] = False
    web = tools.setdefault("web", {})
    if isinstance(web, dict):
        fetch = web.setdefault("fetch", {})
        if isinstance(fetch, dict):
            fetch["enabled"] = False
    browser = tools.setdefault("browser", {})
    if isinstance(browser, dict):
        browser["enabled"] = False


def run_openclaw_for_task(
    row: dict[str, Any],
    checkout_dir: Path,
    task_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    task_dir.mkdir(parents=True, exist_ok=True)
    prompt_text = build_prompt(row)
    prompt_hash = sha256_text(prompt_text)
    cache_key = build_cache_key(row, prompt_text, args)
    prompt_file = task_dir / "prompt.md"
    prompt_file.write_text(prompt_text, encoding="utf-8")
    checkout_transfer = ensure_nemoclaw_checkout_ready(checkout_dir, task_dir, args)
    agent_id, openclaw_config_path = write_task_openclaw_config(row, checkout_dir, task_dir, args)

    invocation_dir = task_dir / "openclaw_invocations"
    invocation_dir.mkdir(parents=True, exist_ok=True)
    max_attempts = max(1, int(args.openclaw_max_attempts))
    metadata: dict[str, Any] | None = None
    sidecar_path: Path | None = None
    for attempt_number in range(1, max_attempts + 1):
        attempt_id = f"{int(time.time())}-{os.getpid()}-{attempt_number}"
        session_key = f"{resolve_session_prefix(args)}:{row['instance_id']}:{attempt_id}"
        attempt_output_dir = task_dir / "openclaw_attempts" / attempt_id
        sidecar_path = task_sidecar_path(attempt_output_dir, str(row["instance_id"]))
        invocation_path = invocation_dir / f"{attempt_id}.json"
        command = [
            sys.executable,
            str(PROTOCOL_RUNNER),
            "run",
            "--benchmark-id",
            "agentic_swe",
            "--task-id",
            str(row["instance_id"]),
            "--prompt-file",
            str(prompt_file),
            "--cwd",
            str(checkout_dir),
            "--output-dir",
            str(attempt_output_dir),
            "--agent",
            agent_id,
            "--session-key",
            session_key,
            "--timeout",
            str(args.openclaw_timeout),
            "--thinking",
            args.thinking,
            "--max-input-tokens",
            str(int(getattr(args, "max_input_tokens", 0) or 0)),
            "--max-tool-calls",
            str(int(getattr(args, "max_tool_calls", 0) or 0)),
        ]
        if openclaw_config_path:
            command.extend(["--openclaw-config-path", str(openclaw_config_path)])
        command.extend(["--openclaw-config-source", str(cache_key["openclaw_config_source"])])
        live_session_dir = task_live_session_dir(checkout_dir, task_dir, args)
        if live_session_dir is not None:
            command.extend(["--live-session-dir", str(live_session_dir)])
        live_sandbox_session_dir = task_live_sandbox_session_dir(checkout_dir, args)
        if live_sandbox_session_dir is not None:
            command.extend(["--live-sandbox-session-dir", live_sandbox_session_dir])
        if getattr(args, "nemoclaw_sandbox", None):
            command.extend(["--nemoclaw-bin", str(getattr(args, "nemoclaw_bin", "nemoclaw"))])
            command.extend(["--nemoclaw-sandbox", str(args.nemoclaw_sandbox)])
            nemoclaw_workdir = getattr(args, "nemoclaw_workdir", None)
            if not nemoclaw_workdir:
                nemoclaw_workdir = str(sandbox_checkout_dir(checkout_dir, args))
            command.extend(["--nemoclaw-workdir", str(nemoclaw_workdir)])
        if args.profile:
            command.extend(["--profile", args.profile])
        if args.model:
            command.extend(["--model", args.model])
        if args.no_local:
            command.append("--no-local")
        if args.allow_failed_preflight:
            command.append("--allow-failed-preflight")
        if args.weave_sidecar:
            command.append("--weave-sidecar")
        else:
            command.append("--no-weave-sidecar")
        if args.weave_sidecar and args.weave_sidecar_strict:
            command.append("--weave-sidecar-strict")
        for denied_tool in effective_deny_tools(args):
            command.extend(["--deny-tool", denied_tool])
        for pattern in effective_deny_argument_patterns(args):
            command.extend(["--deny-argument-pattern", pattern])
        if args.dry_run:
            command.append("--dry-run")

        started_at = time.time()
        result = run_command(command, cwd=REPO_ROOT, timeout=args.openclaw_timeout + 60, check=False)
        ended_at = time.time()
        metadata = {
            "attempt_id": attempt_id,
            "attempt_number": attempt_number,
            "max_attempts": max_attempts,
            "command": command,
            "command_sha256": command_sha256(command),
            "session_key": session_key,
            "runner_version": RUNNER_VERSION,
            "cache_key": cache_key,
            "prompt_hash": prompt_hash,
            "expected_openclaw_result_path": str(sidecar_path),
            "nemoclaw_sandbox": getattr(args, "nemoclaw_sandbox", None),
            "nemoclaw_workdir": getattr(args, "nemoclaw_workdir", None)
            or (
                str(sandbox_checkout_dir(checkout_dir, args))
                if getattr(args, "nemoclaw_sandbox", None)
                else None
            ),
            "nemoclaw_checkout_sandbox_root": getattr(args, "nemoclaw_checkout_sandbox_root", None),
            "nemoclaw_checkout_transfer": checkout_transfer,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "started_at": started_at,
            "ended_at": ended_at,
            "wall_clock_time": ended_at - started_at,
        }
        invocation_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        metadata["openclaw_invocation_path"] = str(invocation_path)
        metadata["openclaw_invocation_sha256"] = sha256_file(invocation_path)
        metadata["openclaw_command_sha256"] = metadata["command_sha256"]
        (task_dir / "openclaw_invocation.json").write_text(
            json.dumps(
                {
                    "latest_invocation_path": str(invocation_path),
                    "latest_attempt_id": attempt_id,
                    "latest_openclaw_result_path": str(sidecar_path),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if result.returncode == 0:
            break
        sidecar = None
        if sidecar_path.exists():
            try:
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                sidecar = None
        if is_weave_sidecar_failure(sidecar):
            raise RuntimeError(
                f"Diagnostic Weave sidecar logging failed for {row['instance_id']}"
            )
        if is_runtime_budget_exceeded(sidecar):
            metadata["openclaw_disqualified_reason"] = "runtime_budget_exceeded"
            metadata["runtime_budget"] = sidecar.get("runtime_budget")
            break
        if isinstance(sidecar, dict) and (
            sidecar.get("tool_policy_ok") is False or sidecar.get("tool_policy_violations")
        ):
            metadata["openclaw_disqualified_reason"] = "tool_policy_violation"
            break
        non_scoreable_reason = non_scoreable_openclaw_failure_reason(result, sidecar)
        if non_scoreable_reason:
            recovery = None
            if non_scoreable_reason == "workspace_vanished" and attempt_number < max_attempts:
                recovery = recover_workspace_vanished_failure(task_dir, result, sidecar)
            if recovery:
                append_transient_failure(
                    task_dir,
                    {
                        "instance_id": row["instance_id"],
                        "attempt_id": attempt_id,
                        "attempt_number": attempt_number,
                        "max_attempts": max_attempts,
                        "returncode": result.returncode,
                        "openclaw_result_path": str(sidecar_path),
                        "failure_text": sidecar_error_text(sidecar, openclaw_failure_text(result, sidecar)),
                        "recovered_non_scoreable_reason": non_scoreable_reason,
                        "recovery": recovery,
                    },
                )
                delay = max(0.0, float(args.openclaw_retry_base_seconds)) * attempt_number
                print(
                    f"OpenClaw recoverable workspace attestation failure for {row['instance_id']} on attempt "
                    f"{attempt_number}/{max_attempts}; retrying after {delay:.1f}s.",
                    flush=True,
                )
                if delay:
                    time.sleep(delay)
                continue
            raise RuntimeError(
                f"OpenClaw non-scoreable setup/provider failure for {row['instance_id']}: "
                f"{non_scoreable_reason}"
            )
        if is_transient_openclaw_failure(result, sidecar):
            append_transient_failure(
                task_dir,
                {
                    "instance_id": row["instance_id"],
                    "attempt_id": attempt_id,
                    "attempt_number": attempt_number,
                    "max_attempts": max_attempts,
                    "returncode": result.returncode,
                    "openclaw_result_path": str(sidecar_path),
                    "failure_text": sidecar_error_text(sidecar, openclaw_failure_text(result, sidecar)),
                    "exhausted": attempt_number >= max_attempts,
                },
            )
            if attempt_number < max_attempts:
                delay = max(0.0, float(args.openclaw_retry_base_seconds)) * attempt_number
                print(
                    f"OpenClaw transient failure for {row['instance_id']} on attempt "
                    f"{attempt_number}/{max_attempts}; retrying after {delay:.1f}s.",
                    flush=True,
                )
                if delay:
                    time.sleep(delay)
                continue
            metadata["openclaw_disqualified_reason"] = "provider_transient_exhausted"
            print(
                f"OpenClaw transient failure exhausted for {row['instance_id']} after "
                f"{attempt_number}/{max_attempts} attempts; recording an empty patch.",
                flush=True,
            )
            break
        raise RuntimeError(f"OpenClaw failed for {row['instance_id']} with code {result.returncode}")
    if metadata is None or sidecar_path is None:
        raise RuntimeError(f"OpenClaw did not run for {row['instance_id']}")
    if not sidecar_path.exists():
        raise RuntimeError(f"OpenClaw completed without sidecar result: {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not sidecar_identity_matches_cache(sidecar, cache_key):
        raise RuntimeError(
            f"OpenClaw sidecar metadata mismatch for {row['instance_id']}: "
            f"{sidecar_path}"
        )
    if not sidecar_nemoclaw_session_audit_matches_cache(sidecar, cache_key):
        raise RuntimeError(
            f"OpenClaw NeMoClaw session audit mismatch for {row['instance_id']}: "
            f"{sidecar_path}"
        )
    if is_weave_sidecar_failure(sidecar):
        raise RuntimeError(f"Diagnostic Weave sidecar logging failed for {row['instance_id']}")
    metadata.update(
        {
            "openclaw_result_path": str(sidecar_path),
            "openclaw_returncode": sidecar.get("returncode"),
            "openclaw_usage": sidecar_usage(sidecar),
            "openclaw_tool_call_count": sidecar.get("tool_call_count", 0),
            "openclaw_tool_error_count": sidecar.get("tool_error_count", 0),
            "tool_policy_ok": sidecar.get("tool_policy_ok"),
            "tool_policy_violations": sidecar.get("tool_policy_violations", []),
            "conversation_order_ok": (sidecar.get("conversation_order") or {}).get("ok"),
            "conversation_order": sidecar.get("conversation_order", {}),
            "nemoclaw_session_audit_ok": (sidecar.get("nemoclaw_session_audit") or {}).get("ok"),
            "nemoclaw_session_audit": sidecar.get("nemoclaw_session_audit", {}),
            **nemoclaw_session_copy_evidence(sidecar),
            "openclaw_config_source": cache_key.get("openclaw_config_source", ""),
            "runtime_budget": sidecar.get("runtime_budget", {}),
            "weave_sidecar": sidecar.get("weave_sidecar", {}),
            "weave_sidecar_ok": (sidecar.get("weave_sidecar") or {}).get("ok"),
            "openclaw_disqualified_reason": metadata.get("openclaw_disqualified_reason", ""),
        }
    )
    return metadata


def write_outputs(
    output_dir: Path,
    patches: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_path = output_dir / "patches.json"
    patch_path.write_text(json.dumps(patches, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    jsonl_path = output_dir / "patches.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for patch in patches:
            f.write(json.dumps(patch, ensure_ascii=False) + "\n")
    summary = {
        "total_requested": len(rows),
        "patches_written": len(patches),
        "empty_patches": sum(1 for patch in patches if not patch.get("patch")),
        "runner_version": RUNNER_VERSION,
        "runtime_budget": runtime_budget_summary(patches, args),
        "traced_patches": sum(1 for patch in patches if patch.get("openclaw_result_path")),
        "tool_called_patches": sum(1 for patch in patches if int(patch.get("openclaw_tool_call_count") or 0) > 0),
        "tool_error_patches": sum(1 for patch in patches if int(patch.get("openclaw_tool_error_count") or 0) > 0),
        "tool_policy_violation_patches": sum(1 for patch in patches if patch.get("tool_policy_violations")),
        "conversation_order_violation_patches": sum(
            1 for patch in patches if patch.get("conversation_order_ok") is False
        ),
        "nemoclaw_session_audit_required_patches": sum(
            1
            for patch in patches
            if isinstance(patch.get("nemoclaw_session_audit"), dict)
            and patch["nemoclaw_session_audit"].get("required") is True
        ),
        "nemoclaw_session_audit_passed_patches": sum(
            1
            for patch in patches
            if isinstance(patch.get("nemoclaw_session_audit"), dict)
            and patch["nemoclaw_session_audit"].get("required") is True
            and patch.get("nemoclaw_session_audit_ok") is True
        ),
        "nemoclaw_session_audit_failed_patches": sum(
            1
            for patch in patches
            if isinstance(patch.get("nemoclaw_session_audit"), dict)
            and patch["nemoclaw_session_audit"].get("required") is True
            and patch.get("nemoclaw_session_audit_ok") is False
        ),
        "runtime_budget_exceeded_patches": sum(
            1 for patch in patches if patch.get("openclaw_disqualified_reason") == "runtime_budget_exceeded"
        ),
        "disqualified_patches": sum(1 for patch in patches if patch.get("openclaw_disqualified_reason")),
        "patch_path": str(patch_path),
        "jsonl_path": str(jsonl_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def runtime_budget_summary(
    patches: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, int | None]:
    def observed_limit(name: str) -> int:
        return max(
            (
                int(((patch.get("runtime_budget") or {}).get("limits") or {}).get(name) or 0)
                for patch in patches
            ),
            default=0,
        )

    configured_input = int(getattr(args, "max_input_tokens", 0) or 0)
    configured_tools = int(getattr(args, "max_tool_calls", 0) or 0)
    max_input_tokens = configured_input or observed_limit("max_input_tokens")
    max_tool_calls = configured_tools or observed_limit("max_tool_calls")
    return {
        "max_input_tokens": max_input_tokens or None,
        "max_tool_calls": max_tool_calls or None,
    }


def filter_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.instance_id:
        wanted = set(args.instance_id)
        rows = [row for row in rows if str(row.get("instance_id")) in wanted]
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/swebench_pro_openclaw"))
    parser.add_argument("--checkout-root", type=Path, default=Path("outputs/swebench_pro_checkouts"))
    parser.add_argument("--prefix", default="openclaw")
    parser.add_argument("--model")
    parser.add_argument("--thinking", default="medium")
    parser.add_argument("--agent", default="main")
    parser.add_argument("--profile")
    parser.add_argument("--openclaw-config-template", type=Path)
    parser.add_argument("--openclaw-tool-profile", default="coding")
    parser.add_argument("--nemoclaw-bin", default="nemoclaw")
    parser.add_argument("--nemoclaw-sandbox")
    parser.add_argument("--nemoclaw-openclaw-config-path", default=NEMOCLAW_OPENCLAW_CONFIG_PATH)
    parser.add_argument(
        "--nemoclaw-workdir",
        help=(
            "Working directory inside the NeMoClaw sandbox. Defaults to the "
            "sandbox-visible checkout path."
        ),
    )
    parser.add_argument(
        "--nemoclaw-checkout-sandbox-root",
        help=(
            "Sandbox-visible root corresponding to --checkout-root. If omitted, "
            "the host checkout absolute path is assumed to be mounted unchanged."
        ),
    )
    parser.add_argument(
        "--nemoclaw-checkout-transfer-mode",
        choices=["visible", "copy"],
        default="visible",
        help=(
            "How to make host checkouts available to the NeMoClaw sandbox. "
            "'visible' requires a pre-mounted/shared checkout path. 'copy' "
            "uploads the checkout archive to the sandbox and captures git diff "
            "inside the sandbox."
        ),
    )
    parser.add_argument(
        "--nemoclaw-checkout-transfer-timeout",
        type=int,
        default=300,
        help="Timeout in seconds for extracting copied checkouts in the sandbox.",
    )
    parser.add_argument("--deny-tool", action="append")
    parser.add_argument("--deny-argument-pattern", action="append")
    parser.add_argument("--use-task-agent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--task-agent-prefix", default="nejumi-swe")
    parser.add_argument("--session-prefix", default="swebench-pro")
    parser.add_argument("--openclaw-timeout", type=int, default=3600)
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=DEFAULT_MAX_INPUT_TOKENS,
        help="SWE-Bench Pro per-task input-token budget. 0 disables the budget.",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=DEFAULT_MAX_TOOL_CALLS,
        help="SWE-Bench Pro per-task tool-call budget. 0 disables the budget.",
    )
    parser.add_argument(
        "--openclaw-max-attempts",
        type=int,
        default=3,
        help="Maximum attempts per instance for transient OpenClaw/provider failures.",
    )
    parser.add_argument(
        "--openclaw-retry-base-seconds",
        type=float,
        default=15.0,
        help="Linear backoff base seconds between transient OpenClaw retries.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--instance-id", action="append")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-local", action="store_true")
    parser.add_argument("--allow-failed-preflight", action="store_true")
    parser.add_argument("--no-reset", action="store_true")
    parser.add_argument("--redo", action="store_true", help="Regenerate patches even when cache keys match.")
    parser.add_argument(
        "--weave-sidecar",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Diagnostic fallback only; native weave-openclaw plugin is the production trace path.",
    )
    parser.add_argument(
        "--weave-sidecar-strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail if --weave-sidecar is enabled and the diagnostic sidecar trace cannot be logged.",
    )
    parser.add_argument(
        "--skip-agent",
        action="store_true",
        help="Only prepare prompts/checkouts and collect any existing diff.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = filter_rows(read_jsonl(args.dataset_jsonl), args)
    if not rows:
        raise SystemExit("No rows selected")

    patches: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, start=1):
        instance_id = str(row["instance_id"])
        print(f"[{index}/{len(rows)}] {instance_id}")
        task_dir = args.output_dir / safe_id(instance_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        prompt_text = build_prompt(row)
        cache_key = build_cache_key(row, prompt_text, args)
        if not args.redo:
            cached_patch = load_cached_patch_record(
                task_dir,
                cache_key,
                args.prefix,
                selected_test_paths(row),
            )
            if cached_patch is not None:
                print(f"Reusing existing SWE-bench Pro patch: {instance_id}", flush=True)
                patches.append(cached_patch)
                write_outputs(args.output_dir, patches, rows, args)
                continue
        openclaw_metadata: dict[str, Any] = {}
        if args.dry_run:
            (task_dir / "prompt.md").write_text(prompt_text, encoding="utf-8")
            patch = ""
        else:
            checkout_dir = prepare_checkout(row, args.checkout_root, reset=not args.no_reset)
            if not args.skip_agent:
                openclaw_metadata = run_openclaw_for_task(row, checkout_dir, task_dir, args)
            elif (
                getattr(args, "nemoclaw_sandbox", None)
                and str(getattr(args, "nemoclaw_checkout_transfer_mode", "visible") or "visible") == "copy"
            ):
                openclaw_metadata["nemoclaw_checkout_transfer"] = ensure_nemoclaw_checkout_ready(
                    checkout_dir,
                    task_dir,
                    args,
                )
            if should_force_empty_patch(openclaw_metadata):
                patch = ""
            elif (
                getattr(args, "nemoclaw_sandbox", None)
                and str(getattr(args, "nemoclaw_checkout_transfer_mode", "visible") or "visible") == "copy"
            ):
                patch = capture_patch_nemoclaw(checkout_dir, args, selected_test_paths(row))
            else:
                patch = capture_patch(checkout_dir, selected_test_paths(row))
            (task_dir / "patch.diff").write_text(patch, encoding="utf-8")
        patch_record = {
            "instance_id": instance_id,
            "patch": patch,
            "prefix": args.prefix,
            "cache_key": cache_key,
            "patch_capture_version": PATCH_CAPTURE_VERSION,
            **{
                key: value
                for key, value in openclaw_metadata.items()
                if key
                in {
                    "attempt_id",
                    "attempt_number",
                    "max_attempts",
                    "runner_version",
                    "prompt_hash",
                    "session_key",
                    "openclaw_invocation_path",
                    "openclaw_invocation_sha256",
                    "openclaw_command_sha256",
                    "openclaw_config_source",
                    "openclaw_result_path",
                    "openclaw_returncode",
                    "openclaw_usage",
                    "openclaw_tool_call_count",
                    "openclaw_tool_error_count",
                    "tool_policy_ok",
                    "tool_policy_violations",
                    "conversation_order_ok",
                    "conversation_order",
                    "nemoclaw_session_audit_ok",
                    "nemoclaw_session_audit",
                    "openclaw_disqualified_reason",
                    "nemoclaw_sandbox",
                    "nemoclaw_workdir",
                    "nemoclaw_checkout_sandbox_root",
                    "nemoclaw_checkout_transfer",
                    "runtime_budget",
                    "weave_sidecar",
                    "weave_sidecar_ok",
                }
            },
        }
        (task_dir / "patch_record.json").write_text(
            json.dumps(patch_record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        patches.append(patch_record)
        write_outputs(args.output_dir, patches, rows, args)


if __name__ == "__main__":
    main()
