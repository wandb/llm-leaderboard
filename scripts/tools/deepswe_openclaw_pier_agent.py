"""Pier agent adapter that runs Nejumi OpenClaw against DeepSWE tasks.

This adapter is intentionally thin: Pier owns task environment setup and
verification, while the existing Taiwan OpenClaw/NeMoClaw runner owns model
execution, native Weave Agents evidence, and runtime-budget enforcement.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib  # type: ignore
except ModuleNotFoundError:  # Pier currently runs on Python 3.13, tests may not.
    import tomli as tomllib  # type: ignore

from pier.agents.base import BaseAgent
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext


def _load_swe_runner() -> Any:
    path = Path(__file__).with_name("run_swebench_pro_openclaw.py")
    spec = importlib.util.spec_from_file_location("nejumi_swebench_pro_openclaw_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


swe_runner = _load_swe_runner()


_RUN_LOCK = threading.Lock()
RUNNER_VERSION = "deepswe-openclaw-pier-2026-07-23-v5"
DEEPSWE_SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
DEEPSWE_RECOVERABLE_COMPLETION_REASONS = {
    "runtime_budget_exceeded",
    "time_up",
    "model_output_truncated",
    "openclaw_no_response",
}
DEEPSWE_COMPLETION_REQUIREMENTS = "\n".join(
    [
        "Follow the task instruction. Keep the fix minimal.",
        "Do not stop after only describing a plan; implement the change in the working checkout.",
        "Before the final response, run `git diff HEAD --check` and inspect `git diff HEAD --stat`; staged changes are allowed.",
        "If `git diff HEAD --stat` is empty, continue editing instead of finishing unless the task is impossible; if it is impossible, state the concrete blocker.",
        "When the implementation and checks are complete, submit it by running this exact command as its own final shell action: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.",
        "Do not run more tools after that submit command. The harness scores all changes relative to HEAD.",
    ]
)


class NonScoreableOpenClawPolicyBlock(RuntimeError):
    """Raised when the benchmark policy, not model ability, prevents scoring."""


class NonScoreableOpenClawProviderTimeout(RuntimeError):
    """Raised when the provider fails before a scoreable model patch is produced."""


class NonScoreableOpenClawConfigurationError(RuntimeError):
    """Raised when the requested OpenClaw/model configuration cannot run."""


def _sanitize_deepswe_instruction(instruction: str) -> str:
    """Remove DeepSWE runner directives that conflict with this harness contract."""
    kept_lines: list[str] = []
    for line in str(instruction or "").splitlines():
        normalized = " ".join(line.strip().lower().split())
        if (
            "work on this in a new branch" in normalized
            and "commit" in normalized
        ) or (
            "create" in normalized
            and "new branch" in normalized
            and "commit" in normalized
        ):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines).strip()


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _read_task_toml(task_dir: Path) -> dict[str, Any]:
    with (task_dir / "task.toml").open("rb") as f:
        return tomllib.load(f)


def _deepswe_sandbox_checkout_root(task_id: str, output_root: Path) -> str:
    """Return a task/trial-scoped sandbox root for copied DeepSWE checkouts."""
    digest = hashlib.sha256(str(output_root.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"/sandbox/checkouts/deepswe/{swe_runner.safe_id(task_id)}-{digest}"


def _image_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _deepswe_python_runtime_paths(task_toml: dict[str, Any]) -> tuple[list[str], list[str]]:
    metadata = task_toml.get("metadata") if isinstance(task_toml.get("metadata"), dict) else {}
    language = str(metadata.get("language") or "").strip().lower()
    if language != "python":
        return [], []
    environment = (
        task_toml.get("environment") if isinstance(task_toml.get("environment"), dict) else {}
    )
    image = str(environment.get("docker_image") or "").strip()
    if not image:
        return [], []
    image_hash = _image_hash(image)
    return (
        [f"/sandbox/.deepswe-tools/python-bin/{image_hash}"],
        [f"/sandbox/.deepswe-tools/python-site/{image_hash}/site-packages"],
    )


def _deepswe_checkout_preflight_command(task_toml: dict[str, Any]) -> tuple[str, list[str]]:
    metadata = task_toml.get("metadata") if isinstance(task_toml.get("metadata"), dict) else {}
    language = str(metadata.get("language") or "").strip().lower()
    if language == "go":
        return language, [
            "bash",
            "-lc",
            "set -euo pipefail; test -f go.mod; "
            "export GOPROXY=off GOSUMDB=off; go test -run '^$' ./...",
        ]
    if language == "python":
        repository_url = str(metadata.get("repository_url") or "").rstrip("/")
        repository_name = repository_url.rsplit("/", 1)[-1].removesuffix(".git").lower()
        package_import, local_source = {
            "mnamer": ("mnamer", None),
            "langchain": ("langchain_core", "libs/core"),
        }.get(repository_name, (None, None))
        import_check = "import pytest, setuptools"
        if package_import:
            import_check += f", {package_import}"
        source_setup = ""
        if local_source:
            source_setup = (
                f"test -d {local_source}; "
                f"export PYTHONPATH=\"$PWD/{local_source}:${{PYTHONPATH:-}}\"; "
            )
        return language, [
            "bash",
            "-lc",
            f"set -euo pipefail; {source_setup}python3 -c '{import_check}'",
        ]
    if language in {"typescript", "javascript"}:
        return language, [
            "bash",
            "-lc",
            "set -euo pipefail; test -f package.json; test -d node_modules; "
            "node -e \"JSON.parse(require('fs').readFileSync('package.json', 'utf8'))\"",
        ]
    return language or "unknown", ["bash", "-lc", "git status --short >/dev/null"]


def _docker_image_identity(task_toml: dict[str, Any]) -> tuple[str | None, str | None]:
    environment = (
        task_toml.get("environment") if isinstance(task_toml.get("environment"), dict) else {}
    )
    image = str(environment.get("docker_image") or "").strip() or None
    if image is None:
        return None, None
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    image_id = result.stdout.strip() if result.returncode == 0 else None
    return image, image_id or None


def _preflight_deepswe_agent_checkout(
    task_id: str,
    task_toml: dict[str, Any],
    checkout_dir: Path,
    task_dir: Path,
    args: Any,
) -> dict[str, Any]:
    language, command = _deepswe_checkout_preflight_command(task_toml)
    exports: list[str] = []
    extra_path = [str(path) for path in (getattr(args, "nemoclaw_extra_path", None) or [])]
    extra_pythonpath = [
        str(path) for path in (getattr(args, "nemoclaw_extra_pythonpath", None) or [])
    ]
    if extra_path:
        exports.append(f"export PATH={':'.join(extra_path)!r}:\"$PATH\"")
    if extra_pythonpath:
        exports.append(
            f"export PYTHONPATH={':'.join(extra_pythonpath)!r}:\"${{PYTHONPATH:-}}\""
        )
    if exports and command[:2] == ["bash", "-lc"]:
        command = [*command[:2], "; ".join([*exports, command[2]])]
    image, image_id = _docker_image_identity(task_toml)
    sandbox_dir = swe_runner.sandbox_checkout_dir(checkout_dir, args)
    started_at = time.monotonic()
    result = swe_runner.run_nemoclaw_text_command(
        args,
        command,
        timeout=max(300, int(getattr(args, "max_tool_wall_seconds", 300) or 300)),
        check=False,
        workdir=str(sandbox_dir),
    )
    evidence = {
        "ok": result.returncode == 0,
        "task_id": task_id,
        "language": language,
        "environment_mode": "nemoclaw_dependency_overlay",
        "task_docker_image": image,
        "task_docker_image_id": image_id,
        "sandbox_checkout": str(sandbox_dir),
        "command": command,
        "extra_path": extra_path,
        "extra_pythonpath": extra_pythonpath,
        "returncode": result.returncode,
        "duration_seconds": round(time.monotonic() - started_at, 3),
        "stdout": (result.stdout or "")[-20_000:],
        "stderr": (result.stderr or "")[-20_000:],
    }
    evidence_path = task_dir / "deepswe_checkout_preflight.json"
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not evidence["ok"]:
        raise NonScoreableOpenClawConfigurationError(
            "non_scoreable_environment_error: copied DeepSWE checkout failed the "
            f"offline {language} preflight before model execution; task_id={task_id}; "
            f"evidence={evidence_path}"
        )
    return evidence


def _sidecar_usage(metadata: dict[str, Any]) -> dict[str, int]:
    usage = metadata.get("openclaw_usage")
    return usage if isinstance(usage, dict) else {}


def _metadata_has_non_scoreable_policy_block(metadata: dict[str, Any]) -> bool:
    # Denied tools/arguments are part of the benchmark interaction contract:
    # the model receives a tool error and may recover with an allowed approach.
    # They should be audited in metadata, but they must not make the trial
    # unscoreable. Provider timeouts and harness configuration failures remain
    # non-scoreable below.
    return False


def _metadata_has_non_scoreable_provider_timeout(metadata: dict[str, Any]) -> bool:
    if metadata.get("deepswe_non_scoreable_reason") == "provider_timeout":
        return True
    runtime_budget = metadata.get("runtime_budget")
    if not isinstance(runtime_budget, dict):
        return False
    live = runtime_budget.get("live")
    if not isinstance(live, dict):
        return False
    try:
        provider_timeout_count = int(live.get("live_provider_timeout_count") or 0)
    except (TypeError, ValueError):
        provider_timeout_count = 0
    if provider_timeout_count > 0:
        return True
    if live.get("reason") == "live_provider_timeout":
        return True
    if live.get("interrupt_reason") == "live_provider_timeout":
        return True
    exceeded_limits = live.get("exceeded_limits")
    if isinstance(exceeded_limits, list) and "live_provider_timeout" in exceeded_limits:
        return True
    return False


def _metadata_has_required_trace_failure(metadata: dict[str, Any]) -> bool:
    return (
        metadata.get("weave_agents_required") is True
        and metadata.get("weave_agents_ok") is False
    )


def _metadata_has_llm_response_idle_timeout(metadata: dict[str, Any]) -> bool:
    runtime_budget = metadata.get("runtime_budget")
    if not isinstance(runtime_budget, dict):
        return False
    live = runtime_budget.get("live")
    if not isinstance(live, dict):
        return False
    if live.get("reason") == "llm_response_idle_timeout":
        return True
    if live.get("interrupt_reason") == "llm_response_idle_timeout":
        return True
    exceeded_limits = live.get("exceeded_limits")
    if isinstance(exceeded_limits, list) and "llm_response_idle_timeout" in exceeded_limits:
        return True
    return False


def _metadata_has_scoreable_runtime_budget_failure(metadata: dict[str, Any]) -> bool:
    if metadata.get("openclaw_disqualified_reason") != "runtime_budget_exceeded":
        return False
    return not _metadata_has_non_scoreable_provider_timeout(metadata)


def _deepswe_should_force_empty_patch(metadata: dict[str, Any]) -> bool:
    if _metadata_has_non_scoreable_provider_timeout(metadata):
        return True
    if _metadata_has_scoreable_runtime_budget_failure(metadata):
        return False
    reason = str(metadata.get("openclaw_disqualified_reason") or "")
    if reason in DEEPSWE_RECOVERABLE_COMPLETION_REASONS:
        return False
    return swe_runner.should_force_empty_patch(metadata)


def _openclaw_sidecar(metadata: dict[str, Any]) -> dict[str, Any]:
    raw_path = str(metadata.get("openclaw_result_path") or "").strip()
    return _read_json(Path(raw_path)) if raw_path else {}


def _submission_trace(sidecar: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    timeline = sidecar.get("timeline_events")
    if not isinstance(timeline, list):
        timeline = []

    assistant_messages: list[str] = []
    reasoning_chunks: list[str] = []
    explicit_submit = False
    for event in timeline:
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        content = event.get("content")
        if event_type == "assistant_message" and isinstance(content, str) and content.strip():
            assistant_messages.append(content.strip())
        elif event_type == "assistant_reasoning" and isinstance(content, str) and content.strip():
            reasoning_chunks.append(content.strip())
        elif event_type == "tool_call":
            try:
                arguments = json.dumps(event.get("arguments"), ensure_ascii=False, sort_keys=True)
            except TypeError:
                arguments = str(event.get("arguments") or "")
            if DEEPSWE_SUBMIT_MARKER in arguments:
                explicit_submit = True

    usage = _sidecar_usage(metadata)
    try:
        reasoning_tokens = int(usage.get("reasoningTokens") or 0)
    except (TypeError, ValueError):
        reasoning_tokens = 0
    if reasoning_chunks:
        reasoning_source = "openclaw_session_jsonl"
    elif reasoning_tokens > 0:
        reasoning_source = "provider_usage_only"
    else:
        reasoning_source = "not_observed"
    final_assistant_text = assistant_messages[-1] if assistant_messages else ""
    return {
        "explicit_submit_marker_seen": explicit_submit,
        "final_assistant_text": final_assistant_text[-20_000:],
        "final_assistant_character_count": len(final_assistant_text),
        "assistant_message_count": len(assistant_messages),
        "reasoning_trace_present": bool(reasoning_chunks),
        "reasoning_trace_source": reasoning_source,
        "reasoning_event_count": len(reasoning_chunks),
        "reasoning_character_count": sum(len(chunk) for chunk in reasoning_chunks),
        "reasoning_tokens": reasoning_tokens,
    }


def _validate_captured_patch(checkout_dir: Path, patch: str) -> dict[str, Any]:
    if not patch.strip():
        return {
            "patch_nonempty": False,
            "patch_apply_check_ok": None,
            "patch_apply_check_returncode": None,
            "patch_stat": "",
            "patch_apply_check_stderr": "",
        }
    check = subprocess.run(
        ["git", "apply", "--check", "--whitespace=nowarn", "-"],
        cwd=checkout_dir,
        input=patch,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    stat = subprocess.run(
        ["git", "apply", "--stat", "-"],
        cwd=checkout_dir,
        input=patch,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    return {
        "patch_nonempty": True,
        "patch_apply_check_ok": check.returncode == 0,
        "patch_apply_check_returncode": check.returncode,
        "patch_stat": (stat.stdout or "")[-20_000:],
        "patch_apply_check_stderr": (check.stderr or "")[-20_000:],
    }


def _deepswe_submission_evidence(
    task_id: str,
    task_dir: Path,
    checkout_dir: Path,
    metadata: dict[str, Any],
    patch: str,
) -> dict[str, Any]:
    trace = _submission_trace(_openclaw_sidecar(metadata), metadata)
    validation = _validate_captured_patch(checkout_dir, patch)
    reason = str(metadata.get("openclaw_disqualified_reason") or "")
    if not validation["patch_nonempty"]:
        mode = "empty_submission"
    elif validation["patch_apply_check_ok"] is False:
        mode = "invalid_patch_submission"
    elif trace["explicit_submit_marker_seen"]:
        mode = "explicit_submit"
    elif reason in DEEPSWE_RECOVERABLE_COMPLETION_REASONS:
        mode = "interrupted_patch_recovery"
    else:
        mode = "implicit_final_submit_recovery"
    evidence = {
        "task_id": task_id,
        "submit_marker": DEEPSWE_SUBMIT_MARKER,
        "mode": mode,
        "accepted": bool(
            validation["patch_nonempty"] and validation["patch_apply_check_ok"]
        ),
        "completion_reason": reason or None,
        **trace,
        **validation,
    }
    path = task_dir / "deepswe_submission.json"
    path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    evidence["evidence_path"] = str(path)
    return evidence


def _metadata_has_unsupported_model_thinking(metadata: dict[str, Any]) -> bool:
    stderr = str(metadata.get("stderr") or "")
    return "Thinking level" in stderr and "is not supported for" in stderr


def _assert_gateway_task_agent_metadata(
    task_dir: Path,
    *,
    agent_id: str,
    canonical_config_path: str,
) -> None:
    metadata_path = task_dir / "openclaw_task_agent.json"
    metadata = _read_json(metadata_path)
    if not metadata:
        raise RuntimeError(f"Missing DeepSWE Gateway task-agent metadata: {metadata_path}")
    registration = metadata.get("gateway_registered")
    if (
        metadata.get("agent_id") != agent_id
        or metadata.get("config_path") != canonical_config_path
        or not isinstance(registration, dict)
        or registration.get("ok") is not True
    ):
        raise RuntimeError(
            "DeepSWE task-agent was not registered through the NeMoClaw Gateway. "
            f"agent_id={agent_id!r}, config_path={metadata.get('config_path')!r}, "
            f"expected_config_path={canonical_config_path!r}, "
            f"gateway_registered={registration!r}"
        )


class NejumiDeepSWEOpenClawAgent(BaseAgent):
    """DeepSWE Pier adapter for Nejumi Taiwan OpenClaw evaluation."""

    SUPPORTS_WINDOWS = False

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        repo_root: str | None = None,
        output_root: str | None = None,
        prefix: str = "deepswe-openclaw",
        openclaw_model: str | None = None,
        openclaw_model_params_json: str | None = None,
        openclaw_model_overrides_json: str | None = None,
        thinking: str = "high",
        agent: str = "main",
        openclaw_timeout: str | int = 3600,
        openclaw_max_attempts: str | int = 1,
        openclaw_retry_base_seconds: str | float = 15.0,
        provider_recovery_rounds: str | int = 2,
        provider_recovery_base_seconds: str | float = 60.0,
        native_trace_recovery_attempts: str | int = 1,
        native_trace_recovery_base_seconds: str | float = 15.0,
        max_input_tokens: str | int = 1_000_000,
        max_cumulative_input_tokens: str | int = 1_000_000,
        max_cumulative_output_tokens: str | int = 500_000,
        max_tool_calls: str | int = 40,
        max_agent_turns: str | int = 40,
        max_tool_wall_seconds: str | int = 300,
        final_assistant_idle_salvage_seconds: str | float = 60.0,
        final_assistant_shutdown_grace_seconds: str | float = 30.0,
        llm_response_idle_timeout_seconds: str | float = 900.0,
        require_actual_token_usage: str | bool = True,
        nemoclaw_bin: str = "nemoclaw",
        nemoclaw_sandbox: str = "nejumi-taiwan",
        nemoclaw_workdir: str = "/sandbox",
        nemoclaw_openclaw_config_path: str = "/sandbox/.openclaw/openclaw.json",
        nemoclaw_checkout_transfer_mode: str = "copy",
        nemoclaw_checkout_transfer_timeout: str | int = 600,
        openclaw_tool_profile: str = "coding",
        task_agent_prefix: str = "tw-deepswe",
        session_prefix: str = "deepswe",
        no_local: str | bool = True,
        use_task_agent: str | bool = True,
        restart_gateway_before_run: str | bool = True,
        verify_weave_agents: str | bool = True,
        weave_agents_entity: str = "llm-leaderboard",
        weave_agents_project: str = "tc-leaderboard",
        weave_agents_agent_name: str = "nejumi-taiwan-openclaw",
        weave_agents_limit: str | int = 50,
        weave_agents_verification_timeout: str | float = 120.0,
        weave_agents_poll_seconds: str | float = 5.0,
        fail_fast_trace_evidence: str | bool = True,
        deny_tool: str | None = None,
        deny_argument_pattern: str | None = None,
        allow_failed_preflight: str | bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.repo_root = Path(repo_root or os.getcwd()).resolve()
        self.output_root = Path(output_root or (self.logs_dir / "openclaw_runs")).resolve()
        self.prefix = prefix
        self.openclaw_model = openclaw_model or model_name
        self.openclaw_model_params_json = openclaw_model_params_json
        self.openclaw_model_overrides_json = openclaw_model_overrides_json
        self.thinking = thinking
        self.agent = agent
        self.openclaw_timeout = _int(openclaw_timeout, 3600)
        self.openclaw_max_attempts = _int(openclaw_max_attempts, 1)
        self.openclaw_retry_base_seconds = _float(openclaw_retry_base_seconds, 15.0)
        self.provider_recovery_rounds = max(0, _int(provider_recovery_rounds, 2))
        self.provider_recovery_base_seconds = max(
            0.0,
            _float(provider_recovery_base_seconds, 60.0),
        )
        self.native_trace_recovery_attempts = max(
            0,
            _int(native_trace_recovery_attempts, 1),
        )
        self.native_trace_recovery_base_seconds = max(
            0.0,
            _float(native_trace_recovery_base_seconds, 15.0),
        )
        self.max_input_tokens = _int(max_input_tokens, 1_000_000)
        self.max_cumulative_input_tokens = _int(max_cumulative_input_tokens, self.max_input_tokens)
        self.max_cumulative_output_tokens = _int(max_cumulative_output_tokens, 500_000)
        self.max_tool_calls = _int(max_tool_calls, 40)
        self.max_agent_turns = _int(max_agent_turns, 40)
        self.max_tool_wall_seconds = _int(max_tool_wall_seconds, 300)
        self.final_assistant_idle_salvage_seconds = _float(
            final_assistant_idle_salvage_seconds,
            60.0,
        )
        self.final_assistant_shutdown_grace_seconds = _float(
            final_assistant_shutdown_grace_seconds,
            30.0,
        )
        self.llm_response_idle_timeout_seconds = _float(
            llm_response_idle_timeout_seconds,
            900.0,
        )
        self.require_actual_token_usage = _truthy(require_actual_token_usage, True)
        self.nemoclaw_bin = nemoclaw_bin
        self.nemoclaw_sandbox = nemoclaw_sandbox
        self.nemoclaw_workdir = nemoclaw_workdir
        self.nemoclaw_openclaw_config_path = nemoclaw_openclaw_config_path
        self.nemoclaw_checkout_transfer_mode = nemoclaw_checkout_transfer_mode
        self.nemoclaw_checkout_transfer_timeout = _int(nemoclaw_checkout_transfer_timeout, 600)
        self.openclaw_tool_profile = openclaw_tool_profile
        self.task_agent_prefix = task_agent_prefix
        self.session_prefix = session_prefix
        self.no_local = _truthy(no_local, True)
        self.use_task_agent = _truthy(use_task_agent, True)
        self.restart_gateway_before_run = _truthy(restart_gateway_before_run, True)
        self.verify_weave_agents = _truthy(verify_weave_agents, True)
        self.weave_agents_entity = weave_agents_entity
        self.weave_agents_project = weave_agents_project
        self.weave_agents_agent_name = weave_agents_agent_name
        self.weave_agents_limit = _int(weave_agents_limit, 50)
        self.weave_agents_verification_timeout = _float(weave_agents_verification_timeout, 120.0)
        self.weave_agents_poll_seconds = _float(weave_agents_poll_seconds, 5.0)
        self.fail_fast_trace_evidence = _truthy(fail_fast_trace_evidence, True)
        self.deny_tool = [item for item in (deny_tool or "").split(",") if item]
        self.deny_argument_pattern = [
            item for item in (deny_argument_pattern or "").split("\n") if item
        ]
        self.allow_failed_preflight = _truthy(allow_failed_preflight, False)
        self._latest_metadata: dict[str, Any] = {}

    @staticmethod
    def name() -> str:
        return "nejumi-deepswe-openclaw"

    def version(self) -> str:
        return RUNNER_VERSION

    async def setup(self, environment: BaseEnvironment) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def _task_config(self) -> tuple[str, Path, dict[str, Any]]:
        config = _read_json(self.logs_dir.parent / "config.json")
        task_path = Path(config.get("task", {}).get("path") or "")
        if not task_path.is_absolute():
            task_path = (self.repo_root / task_path).resolve()
        task_id = task_path.name or self.logs_dir.parent.name
        task_toml = _read_task_toml(task_path)
        return task_id, task_path, task_toml

    def _args(self) -> Any:
        return type(
            "Args",
            (),
            {
                "prefix": self.prefix,
                "model": self.openclaw_model,
                "openclaw_model_params_json": self.openclaw_model_params_json,
                "openclaw_model_overrides_json": self.openclaw_model_overrides_json,
                "thinking": self.thinking,
                "agent": self.agent,
                "profile": None,
                "openclaw_config_template": None,
                "openclaw_tool_profile": self.openclaw_tool_profile,
                "nemoclaw_bin": self.nemoclaw_bin,
                "nemoclaw_sandbox": self.nemoclaw_sandbox,
                "nemoclaw_workdir": self.nemoclaw_workdir,
                "nemoclaw_openclaw_config_path": self.nemoclaw_openclaw_config_path,
                "nemoclaw_checkout_sandbox_root": None,
                "nemoclaw_checkout_transfer_mode": self.nemoclaw_checkout_transfer_mode,
                "nemoclaw_checkout_transfer_timeout": self.nemoclaw_checkout_transfer_timeout,
                "deny_tool": self.deny_tool or None,
                "deny_argument_pattern": self.deny_argument_pattern or None,
                "use_task_agent": self.use_task_agent,
                "task_agent_prefix": self.task_agent_prefix,
                "restart_gateway_after_task_agent_registration": self.restart_gateway_before_run,
                "session_prefix": self.session_prefix,
                "openclaw_timeout": self.openclaw_timeout,
                "openclaw_max_attempts": self.openclaw_max_attempts,
                "openclaw_retry_base_seconds": self.openclaw_retry_base_seconds,
                "max_input_tokens": self.max_input_tokens,
                "max_cumulative_input_tokens": self.max_cumulative_input_tokens,
                "max_cumulative_output_tokens": self.max_cumulative_output_tokens,
                "require_actual_token_usage": self.require_actual_token_usage,
                "max_tool_calls": self.max_tool_calls,
                "max_agent_turns": self.max_agent_turns,
                "max_tool_wall_seconds": self.max_tool_wall_seconds,
                "final_assistant_idle_salvage_seconds": (
                    self.final_assistant_idle_salvage_seconds
                ),
                "final_assistant_shutdown_grace_seconds": (
                    self.final_assistant_shutdown_grace_seconds
                ),
                "llm_response_idle_timeout_seconds": (
                    self.llm_response_idle_timeout_seconds
                ),
                "dry_run": False,
                "no_local": self.no_local,
                "allow_failed_preflight": self.allow_failed_preflight,
                "no_reset": False,
                "redo": False,
                "weave_sidecar": False,
                "weave_sidecar_strict": False,
                "verify_weave_agents": self.verify_weave_agents,
                "weave_agents_entity": self.weave_agents_entity,
                "weave_agents_project": self.weave_agents_project,
                "weave_agents_agent_name": self.weave_agents_agent_name,
                "weave_agents_env_file": self.repo_root / ".env",
                "weave_agents_limit": self.weave_agents_limit,
                "weave_agents_verification_timeout": self.weave_agents_verification_timeout,
                "weave_agents_poll_seconds": self.weave_agents_poll_seconds,
                "fail_fast_trace_evidence": self.fail_fast_trace_evidence,
                "skip_agent": False,
            },
        )()

    def _row(self, task_id: str, task_toml: dict[str, Any], instruction: str) -> dict[str, Any]:
        metadata = task_toml.get("metadata") if isinstance(task_toml.get("metadata"), dict) else {}
        environment = (
            task_toml.get("environment") if isinstance(task_toml.get("environment"), dict) else {}
        )
        return {
            "benchmark_id": "deepswe",
            "benchmark_name": "DeepSWE",
            "instance_id": task_id,
            "repo": metadata.get("repo_name") or metadata.get("repo_url"),
            "base_commit": metadata.get("base_commit"),
            "repo_language": metadata.get("language"),
            "dockerhub_tag": environment.get("docker_image"),
            "problem_statement": _sanitize_deepswe_instruction(instruction),
            "requirements": DEEPSWE_COMPLETION_REQUIREMENTS,
            "interface": "",
            "issue_specificity": "deep-swe",
            "issue_categories": [metadata.get("language")] if metadata.get("language") else [],
            "selected_test_files_to_run": [],
        }

    def _run_openclaw_host(self, task_id: str, task_toml: dict[str, Any], instruction: str, checkout_dir: Path) -> tuple[dict[str, Any], str]:
        args = self._args()
        if self.nemoclaw_sandbox and self.nemoclaw_checkout_transfer_mode == "copy":
            args.nemoclaw_checkout_sandbox_root = _deepswe_sandbox_checkout_root(
                task_id,
                self.output_root,
            )
        extra_path, extra_pythonpath = _deepswe_python_runtime_paths(task_toml)
        args.nemoclaw_extra_path = extra_path
        args.nemoclaw_extra_pythonpath = extra_pythonpath
        row = self._row(task_id, task_toml, instruction)
        task_dir = self.output_root / swe_runner.safe_id(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        initial_checkout_transfer: dict[str, Any] | None = None
        checkout_preflight: dict[str, Any] | None = None
        with _RUN_LOCK:
            swe_runner.ensure_nemoclaw_openclaw_permissions(args)
            initial_checkout_transfer = swe_runner.ensure_nemoclaw_checkout_ready(
                checkout_dir,
                task_dir,
                args,
            )
            gateway_task_agent = swe_runner.uses_nemoclaw_gateway_task_agent(args)
            if (
                self.verify_weave_agents
                and self.use_task_agent
                and self.nemoclaw_sandbox
                and not gateway_task_agent
            ):
                raise RuntimeError(
                    "DeepSWE native Weave Agents trace requires the NeMoClaw Gateway "
                    "task-agent path. Check no_local/use_task_agent/nemoclaw_sandbox wiring."
                )
            if self.restart_gateway_before_run and gateway_task_agent:
                agent_id, config_path = swe_runner.write_task_openclaw_config(
                    row, checkout_dir, task_dir, args
                )
                if config_path is not None:
                    raise RuntimeError(
                        "DeepSWE Gateway task-agent registration unexpectedly returned "
                        f"a checkout-local OpenClaw config path: {config_path}"
                    )
                _assert_gateway_task_agent_metadata(
                    task_dir,
                    agent_id=agent_id,
                    canonical_config_path=self.nemoclaw_openclaw_config_path,
                )
                swe_runner.restart_nemoclaw_gateway_after_task_agent_registration(args, label="DeepSWE")
            checkout_preflight = _preflight_deepswe_agent_checkout(
                task_id,
                task_toml,
                checkout_dir,
                task_dir,
                args,
            )
        provider_recovery_log: list[dict[str, Any]] = []
        metadata: dict[str, Any] = {}
        for recovery_round in range(self.provider_recovery_rounds + 1):
            if recovery_round > 0:
                cooldown = self.provider_recovery_base_seconds * (2 ** (recovery_round - 1))
                recovery_entry = {
                    "recovery_round": recovery_round,
                    "max_recovery_rounds": self.provider_recovery_rounds,
                    "cooldown_seconds": cooldown,
                    "started_at": time.time(),
                }
                provider_recovery_log.append(recovery_entry)
                (task_dir / "provider_recovery.json").write_text(
                    json.dumps(provider_recovery_log, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                if cooldown:
                    time.sleep(cooldown)
            current_metadata = swe_runner.run_openclaw_for_task(
                row,
                checkout_dir,
                task_dir,
                args,
            )
            metadata = swe_runner.merge_prior_billable_openclaw_usage(
                current_metadata,
                metadata or None,
            )
            if not _metadata_has_non_scoreable_provider_timeout(metadata):
                break
            if recovery_round < self.provider_recovery_rounds:
                print(
                    f"DeepSWE provider recovery deferred for {task_id}: round "
                    f"{recovery_round + 1}/{self.provider_recovery_rounds}",
                    flush=True,
                )
        if provider_recovery_log:
            metadata = dict(metadata)
            metadata["provider_recovery"] = provider_recovery_log
        native_trace_recovery_log: list[dict[str, Any]] = []
        for recovery_round in range(1, self.native_trace_recovery_attempts + 1):
            if not _metadata_has_required_trace_failure(metadata):
                break
            cooldown = self.native_trace_recovery_base_seconds
            recovery_entry = {
                "recovery_round": recovery_round,
                "max_recovery_attempts": self.native_trace_recovery_attempts,
                "cooldown_seconds": cooldown,
                "started_at": time.time(),
            }
            native_trace_recovery_log.append(recovery_entry)
            (task_dir / "native_trace_recovery.json").write_text(
                json.dumps(
                    native_trace_recovery_log,
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                f"DeepSWE isolated native trace recovery for {task_id}: "
                f"round {recovery_round}/{self.native_trace_recovery_attempts}",
                flush=True,
            )
            if cooldown:
                time.sleep(cooldown)
            current_metadata = swe_runner.run_openclaw_for_task(
                row,
                checkout_dir,
                task_dir,
                args,
            )
            metadata = swe_runner.merge_prior_billable_openclaw_usage(
                current_metadata,
                metadata or None,
            )
        if native_trace_recovery_log:
            metadata = dict(metadata)
            metadata["native_trace_recovery"] = native_trace_recovery_log
        if gateway_task_agent:
            _assert_gateway_task_agent_metadata(
                task_dir,
                agent_id=swe_runner.safe_agent_id(task_id, self.task_agent_prefix),
                canonical_config_path=self.nemoclaw_openclaw_config_path,
            )
        with _RUN_LOCK:
            if _metadata_has_non_scoreable_provider_timeout(metadata):
                patch_path = task_dir / "model.patch"
                patch_path.write_text("", encoding="utf-8")
                metadata = dict(metadata)
                metadata.update(
                    {
                        "runner_version": RUNNER_VERSION,
                        "deepswe_task_id": task_id,
                        "deepswe_initial_nemoclaw_checkout_transfer": initial_checkout_transfer,
                        "deepswe_checkout_preflight": checkout_preflight,
                        "deepswe_patch_path": str(patch_path),
                        "deepswe_patch_bytes": 0,
                        "deepswe_non_scoreable_reason": "provider_timeout",
                    }
                )
                (task_dir / "deepswe_openclaw_metadata.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                raise NonScoreableOpenClawProviderTimeout(
                    "non_scoreable_provider_timeout: OpenClaw provider timed out before "
                    f"a scoreable patch could be produced; task_id={task_id}"
                )
            if _metadata_has_llm_response_idle_timeout(metadata):
                metadata = dict(metadata)
                metadata.setdefault("openclaw_disqualified_reason", "runtime_budget_exceeded")
                metadata["deepswe_scoreable_failure_reason"] = "llm_response_idle_timeout"
            if _metadata_has_unsupported_model_thinking(metadata):
                patch_path = task_dir / "model.patch"
                patch_path.write_text("", encoding="utf-8")
                metadata = dict(metadata)
                metadata.update(
                    {
                        "runner_version": RUNNER_VERSION,
                        "deepswe_task_id": task_id,
                        "deepswe_initial_nemoclaw_checkout_transfer": initial_checkout_transfer,
                        "deepswe_checkout_preflight": checkout_preflight,
                        "deepswe_patch_path": str(patch_path),
                        "deepswe_patch_bytes": 0,
                        "deepswe_non_scoreable_reason": "unsupported_model_thinking",
                    }
                )
                (task_dir / "deepswe_openclaw_metadata.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                raise NonScoreableOpenClawConfigurationError(
                    "non_scoreable_configuration_error: requested OpenClaw thinking level "
                    f"is unsupported by the model; task_id={task_id}"
                )
            if _deepswe_should_force_empty_patch(metadata):
                patch = ""
            elif self.nemoclaw_sandbox and self.nemoclaw_checkout_transfer_mode == "copy":
                patch = swe_runner.capture_patch_nemoclaw(checkout_dir, args, [])
            else:
                patch = swe_runner.capture_patch(checkout_dir, [])
            submission = _deepswe_submission_evidence(
                task_id,
                task_dir,
                checkout_dir,
                metadata,
                patch,
            )
        patch_path = task_dir / "model.patch"
        patch_path.write_text(patch, encoding="utf-8")
        metadata = dict(metadata)
        metadata.update(
            {
                "runner_version": RUNNER_VERSION,
                "deepswe_task_id": task_id,
                "deepswe_initial_nemoclaw_checkout_transfer": initial_checkout_transfer,
                "deepswe_checkout_preflight": checkout_preflight,
                "deepswe_patch_path": str(patch_path),
                "deepswe_patch_bytes": len(patch.encode("utf-8")),
                "deepswe_submission": submission,
            }
        )
        (task_dir / "deepswe_openclaw_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return metadata, patch

    async def _apply_patch_and_commit(
        self,
        environment: BaseEnvironment,
        patch: str,
        task_id: str,
    ) -> dict[str, Any]:
        patch_file = self.logs_dir / "model.patch"
        patch_file.write_text(patch, encoding="utf-8")
        await environment.upload_file(patch_file, "/tmp/nejumi_deepswe_model.patch")
        command = r"""
set -euo pipefail
cd /app
git config --global --add safe.directory /app >/dev/null 2>&1 || true
git config user.email "nejumi-eval@example.invalid"
git config user.name "Nejumi Evaluation Harness"
if [ ! -s /tmp/nejumi_deepswe_model.patch ]; then
  echo '{"patch_applied": false, "reason": "empty_patch"}'
  exit 0
fi
apply_method="index"
if ! git apply --binary --index /tmp/nejumi_deepswe_model.patch 2>/tmp/nejumi_deepswe_git_apply_index.err; then
  index_error="$(cat /tmp/nejumi_deepswe_git_apply_index.err || true)"
  git reset --hard >/dev/null 2>&1 || true
  apply_method="worktree"
  if ! git apply --binary /tmp/nejumi_deepswe_model.patch 2>/tmp/nejumi_deepswe_git_apply_worktree.err; then
    echo "git apply --index failed:" >&2
    printf '%s\n' "$index_error" >&2
    echo "git apply worktree fallback failed:" >&2
    cat /tmp/nejumi_deepswe_git_apply_worktree.err >&2 || true
    exit 1
  fi
  git add -A
fi
git add -A
if git diff --cached --quiet; then
  echo "{\"patch_applied\": true, \"committed\": false, \"reason\": \"no_index_diff\", \"apply_method\": \"$apply_method\"}"
  exit 0
fi
git commit -m "DeepSWE model patch"
head_commit="$(git rev-parse HEAD)"
echo "{\"patch_applied\": true, \"committed\": true, \"head_commit\": \"$head_commit\", \"apply_method\": \"$apply_method\"}"
"""
        result = await environment.exec(command, timeout_sec=180)
        applied = {
            "return_code": result.return_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        if result.return_code != 0:
            raise RuntimeError(
                f"Failed to apply DeepSWE OpenClaw patch for {task_id}: {result.stdout}\n{result.stderr}"
            )
        try:
            applied.update(json.loads((result.stdout or "").strip().splitlines()[-1]))
        except (IndexError, json.JSONDecodeError):
            pass
        return applied

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        started_at = time.time()
        task_id, task_path, task_toml = self._task_config()
        checkout_dir = self.logs_dir / "openclaw_checkout"
        if checkout_dir.exists():
            shutil.rmtree(checkout_dir)
        checkout_dir.mkdir(parents=True, exist_ok=True)

        await environment.download_dir("/app", checkout_dir)
        metadata, patch = await asyncio.to_thread(
            self._run_openclaw_host,
            task_id,
            task_toml,
            instruction,
            checkout_dir,
        )
        apply_result = await self._apply_patch_and_commit(environment, patch, task_id)

        usage = _sidecar_usage(metadata)
        context.n_input_tokens = usage.get("inputTokens")
        context.n_cache_tokens = usage.get("cacheReadInputTokens")
        context.n_output_tokens = usage.get("outputTokens")
        context.peak_context_tokens = usage.get("inputTokens")
        context.n_agent_steps = metadata.get("openclaw_tool_call_count")
        context.metadata = {
            "runner_version": RUNNER_VERSION,
            "task_id": task_id,
            "task_path": str(task_path),
            "wall_clock_time": time.time() - started_at,
            "openclaw": metadata,
            "patch_apply": apply_result,
        }
        self._latest_metadata = context.metadata
        (self.logs_dir / "deepswe_openclaw_agent_context.json").write_text(
            context.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
