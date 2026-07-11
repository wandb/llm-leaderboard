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
RUNNER_VERSION = "deepswe-openclaw-pier-2026-07-11-v1"


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


def _deepswe_sandbox_checkout_root(task_id: str, logs_dir: Path) -> str:
    """Return a task/trial-scoped sandbox root for copied DeepSWE checkouts."""
    digest = hashlib.sha256(str(logs_dir.parent.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"/sandbox/checkouts/deepswe/{swe_runner.safe_id(task_id)}-{digest}"


def _sidecar_usage(metadata: dict[str, Any]) -> dict[str, int]:
    usage = metadata.get("openclaw_usage")
    return usage if isinstance(usage, dict) else {}


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
        thinking: str = "high",
        agent: str = "main",
        openclaw_timeout: str | int = 3600,
        openclaw_max_attempts: str | int = 1,
        openclaw_retry_base_seconds: str | float = 15.0,
        max_input_tokens: str | int = 1_000_000,
        max_cumulative_input_tokens: str | int = 1_000_000,
        max_cumulative_output_tokens: str | int = 500_000,
        max_tool_calls: str | int = 40,
        max_agent_turns: str | int = 40,
        max_tool_wall_seconds: str | int = 300,
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
        self.thinking = thinking
        self.agent = agent
        self.openclaw_timeout = _int(openclaw_timeout, 3600)
        self.openclaw_max_attempts = _int(openclaw_max_attempts, 1)
        self.openclaw_retry_base_seconds = _float(openclaw_retry_base_seconds, 15.0)
        self.max_input_tokens = _int(max_input_tokens, 1_000_000)
        self.max_cumulative_input_tokens = _int(max_cumulative_input_tokens, self.max_input_tokens)
        self.max_cumulative_output_tokens = _int(max_cumulative_output_tokens, 500_000)
        self.max_tool_calls = _int(max_tool_calls, 40)
        self.max_agent_turns = _int(max_agent_turns, 40)
        self.max_tool_wall_seconds = _int(max_tool_wall_seconds, 300)
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
            "problem_statement": instruction,
            "requirements": "Follow the task instruction. Keep the fix minimal.",
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
                self.logs_dir,
            )
        row = self._row(task_id, task_toml, instruction)
        task_dir = self.output_root / swe_runner.safe_id(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        with _RUN_LOCK:
            swe_runner.ensure_nemoclaw_openclaw_permissions(args)
            swe_runner.ensure_nemoclaw_checkout_ready(checkout_dir, task_dir, args)
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
            metadata = swe_runner.run_openclaw_for_task(row, checkout_dir, task_dir, args)
            if gateway_task_agent:
                _assert_gateway_task_agent_metadata(
                    task_dir,
                    agent_id=swe_runner.safe_agent_id(task_id, self.task_agent_prefix),
                    canonical_config_path=self.nemoclaw_openclaw_config_path,
                )
            if swe_runner.should_force_empty_patch(metadata):
                patch = ""
            elif self.nemoclaw_sandbox and self.nemoclaw_checkout_transfer_mode == "copy":
                patch = swe_runner.capture_patch_nemoclaw(checkout_dir, args, [])
            else:
                patch = swe_runner.capture_patch(checkout_dir, [])
        patch_path = task_dir / "model.patch"
        patch_path.write_text(patch, encoding="utf-8")
        metadata = dict(metadata)
        metadata.update(
            {
                "runner_version": RUNNER_VERSION,
                "deepswe_task_id": task_id,
                "deepswe_patch_path": str(patch_path),
                "deepswe_patch_bytes": len(patch.encode("utf-8")),
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
git apply --binary --index /tmp/nejumi_deepswe_model.patch
git add -A
if git diff --cached --quiet; then
  echo '{"patch_applied": true, "committed": false, "reason": "no_index_diff"}'
  exit 0
fi
git commit -m "DeepSWE model patch"
head_commit="$(git rev-parse HEAD)"
echo "{\"patch_applied\": true, \"committed\": true, \"head_commit\": \"$head_commit\"}"
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
