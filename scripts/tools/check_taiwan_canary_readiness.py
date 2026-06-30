#!/usr/bin/env python3
"""
Check Taiwan canary readiness without calling paid model APIs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
DEFAULT_OPENCLAW_CONFIG = Path.home() / ".openclaw" / "openclaw.json"
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "base_config_taiwan.yaml"
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "taiwan_openai_canary_models.yaml"
DEFAULT_CANARY_SLUG = "gpt-4_1-mini-openai-direct-canary"
DEFAULT_OPENCLAW_MODEL = "openai-direct/gpt-4.1-mini-2025-04-14"
DEFAULT_CANARY_CONFIG_DIRS = {
    "full": REPO_ROOT / "configs" / "taiwan_full" / "generated_openai_canary",
    "nonagentic": REPO_ROOT / "configs" / "taiwan_full" / "generated_openai_canary_nonagentic",
    "agentic": REPO_ROOT / "configs" / "taiwan_full" / "generated_openai_canary_agentic",
    "agentic_nemoclaw": REPO_ROOT / "configs" / "taiwan_full" / "generated_openai_canary_agentic_nemoclaw",
    "agentic_aggregate": REPO_ROOT / "configs" / "taiwan_full" / "generated_openai_canary_agentic_aggregate",
}
OPENCLAW_PROVIDER_ENV_KEYS = {
    "openai-direct": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
}
AGENTIC_REQUIRED_DENIED_TOOLS = {
    "*search*",
    "browser",
    "browser_*",
    "code_execution",
    "web_fetch",
    "web_search",
}
AGENTIC_REQUIRED_DENIED_ARGUMENT_PATTERNS = {
    r"\b(curl|wget)\b",
    r"\b(requests|urllib|httpx)\.",
    r"https?://",
}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def load_env(env_file: Path) -> dict[str, str]:
    env = dict(os.environ)
    if not env_file.exists():
        return env
    for raw in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in env:
            env[key] = value.strip().strip('"').strip("'")
    return env


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = OmegaConf.to_container(OmegaConf.load(manifest_path), resolve=True)
    models = manifest.get("models", []) if isinstance(manifest, dict) else []
    return [model for model in models if isinstance(model, dict)]


def canary_models(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [model for model in models if bool(model.get("canary"))]


def openclaw_provider_and_model(openclaw_model: str) -> tuple[str, str]:
    provider, sep, model_id = openclaw_model.partition("/")
    return provider, model_id if sep else ""


def infer_expected_pretrained_model(model: dict[str, Any]) -> str | None:
    source = model.get("source_config")
    if not isinstance(source, str) or not source:
        return None
    source_path = REPO_ROOT / "configs" / source
    if not source_path.exists():
        return None
    cfg = OmegaConf.load(source_path)
    value = cfg_get(cfg, "model.pretrained_model_name_or_path")
    return str(value) if value is not None else None


def artifact_name(artifact_path: str) -> str:
    leaf = artifact_path.rsplit("/", 1)[-1]
    return leaf.split(":", 1)[0]


def local_artifact_exists(artifacts_root: Path, artifact_path: str) -> bool:
    name = artifact_name(artifact_path)
    if not artifacts_root.exists():
        return False
    return any(path.is_dir() and path.name.startswith(f"{name}:") for path in artifacts_root.iterdir())


def cfg_get(cfg: Any, dotted: str) -> Any:
    current = cfg
    for part in dotted.split("."):
        if isinstance(current, DictConfig):
            if part not in current:
                return None
            current = current[part]
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def enabled_artifacts(cfg: DictConfig) -> dict[str, str]:
    artifact_keys = {
        "jaster": "jaster.artifacts_path",
        "mtbench_questions": "mtbench.question_artifacts_path",
        "mtbench_reference": "mtbench.referenceanswer_artifacts_path",
        "mtbench_judge_prompt": "mtbench.judge_prompt_artifacts_path",
        "bfcl": "bfcl.artifacts_path",
        "swebench_pro": "swebench_pro.artifacts_path",
        "agentic_math": "agentic_math.artifacts_path",
        "hle": "hle.artifact_path",
        "hallulens_zh_tw": "hallulens_zh_tw.artifacts_path",
        "ifeval_zh_tw": "ifeval_zh_tw.artifacts_path",
        "ts_bench": "ts_bench.artifacts_path",
    }
    run = cfg.get("run", {})
    result: dict[str, str] = {}
    for label, dotted in artifact_keys.items():
        task = label.split("_questions", 1)[0].split("_reference", 1)[0].split("_judge_prompt", 1)[0]
        task = "mtbench" if label.startswith("mtbench") else task
        if task in run and not run.get(task):
            continue
        value = cfg_get(cfg, dotted)
        if isinstance(value, str) and value:
            result[label] = value
    return result


def load_merged_config(base_config: Path, phase_config: Path) -> DictConfig:
    return OmegaConf.merge(OmegaConf.load(base_config), OmegaConf.load(phase_config))


def nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def string_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    try:
        items = list(value)
    except TypeError:
        return set()
    return {str(item) for item in items if str(item)}


def deny_policy_check(cfg: Any, section: str, field: str, required: set[str], label: str) -> Check:
    observed = string_set(cfg_get(cfg, f"{section}.{field}"))
    missing = sorted(required - observed)
    return Check(
        f"agentic {label} denies remote lookup via {field}",
        not missing,
        json.dumps(
            {
                "section": section,
                "missing": missing,
                "observed": sorted(observed),
            },
            ensure_ascii=False,
        ),
    )


def check_manifest(manifest_path: Path, *, expected_slug: str | None) -> list[Check]:
    checks: list[Check] = []
    models = load_manifest(manifest_path)
    canaries = canary_models(models)
    slug_ok = len(canaries) == 1 and (
        expected_slug is None or canaries[0].get("slug") == expected_slug
    )
    checks.append(
        Check(
            "manifest has exactly one canary",
            slug_ok,
            f"found {[model.get('slug') for model in canaries]}",
        )
    )
    opus = [
        model
        for model in models
        if isinstance(model, dict) and "opus" in str(model.get("slug", "")).lower()
    ]
    checks.append(
        Check(
            "Claude Opus is final_only",
            all(bool(model.get("final_only")) for model in opus),
            f"opus entries {[model.get('slug') for model in opus]}",
        )
    )
    return checks


def check_openclaw_config(path: Path, *, openclaw_model: str) -> list[Check]:
    if not path.exists():
        return [Check("OpenClaw config exists", False, str(path))]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [Check("OpenClaw config parses", False, str(exc))]
    provider_id, model_id = openclaw_provider_and_model(openclaw_model)
    provider = data.get("models", {}).get("providers", {}).get(provider_id, {})
    models = provider.get("models") if isinstance(provider, dict) else None
    model_ids = [model.get("id") for model in models or [] if isinstance(model, dict)]
    entries = data.get("plugins", {}).get("entries", {})
    weave = entries.get("weave") if isinstance(entries, dict) else {}
    return [
        Check(f"OpenClaw {provider_id} provider exists", bool(provider), str(path)),
        Check(f"OpenClaw model is registered: {openclaw_model}", model_id in model_ids, str(model_ids)),
        Check("OpenClaw Weave plugin is enabled", bool(weave and weave.get("enabled")), str(weave.get("enabled") if isinstance(weave, dict) else None)),
    ]


def check_env(env: dict[str, str], *, openclaw_model: str) -> list[Check]:
    provider_id, _ = openclaw_provider_and_model(openclaw_model)
    checks = [Check("WANDB_API_KEY is available", bool(env.get("WANDB_API_KEY")), "value hidden")]
    key_name = OPENCLAW_PROVIDER_ENV_KEYS.get(provider_id)
    if key_name:
        checks.append(
            Check(
                f"{key_name} is available for {provider_id}/OpenClaw",
                bool(env.get(key_name)),
                "value hidden",
            )
        )
    else:
        checks.append(
            Check(
                f"No known provider credential check for {provider_id}",
                True,
                "provider-specific credential check not configured",
            )
        )
    return checks


def _run_status(command: list[str], env: dict[str, str], timeout: int = 60) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            env=env,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    detail = (result.stdout + result.stderr).strip()
    if len(detail) > 500:
        detail = detail[:500] + "..."
    return result.returncode == 0, detail


def check_nemoclaw(
    env: dict[str, str],
    *,
    nemoclaw_bin: str,
    sandbox: str,
    require: bool,
) -> list[Check]:
    checks: list[Check] = []
    path = env.get("PATH")
    resolved_nemoclaw = shutil.which(nemoclaw_bin, path=path)
    resolved_openshell = shutil.which("openshell", path=path)

    checks.append(
        Check(
            "NeMoClaw command is available",
            bool(resolved_nemoclaw) or not require,
            resolved_nemoclaw or "missing; pass --require-nemoclaw to make this a hard gate",
        )
    )
    checks.append(
        Check(
            "OpenShell command is available",
            bool(resolved_openshell) or not require,
            resolved_openshell or "missing; pass --require-nemoclaw to make this a hard gate",
        )
    )
    if not resolved_nemoclaw:
        return checks

    version_ok, version_detail = _run_status([resolved_nemoclaw, "--version"], env)
    checks.append(
        Check(
            "NeMoClaw version command succeeds",
            version_ok or not require,
            version_detail or "no output",
        )
    )

    status_ok, status_detail = _run_status(
        [resolved_nemoclaw, "sandbox", "status", sandbox],
        env,
    )
    checks.append(
        Check(
            f"NeMoClaw sandbox status succeeds: {sandbox}",
            status_ok or not require,
            status_detail or "no output",
        )
    )

    openclaw_ok, openclaw_detail = _run_status(
        [
            resolved_nemoclaw,
            "sandbox",
            "exec",
            sandbox,
            "--no-tty",
            "--timeout",
            "30",
            "--",
            "openclaw",
            "--version",
        ],
        env,
        timeout=90,
    )
    checks.append(
        Check(
            f"OpenClaw runs inside NeMoClaw sandbox: {sandbox}",
            openclaw_ok or not require,
            openclaw_detail or "no output",
        )
    )
    return checks


def check_generated_configs(
    base_config: Path,
    config_paths: dict[str, Path],
    *,
    expected_pretrained_model: str | None,
    openclaw_model: str,
    require_nemoclaw: bool = False,
    nemoclaw_sandbox: str = "nejumi-taiwan",
) -> list[Check]:
    checks: list[Check] = []
    for phase, path in config_paths.items():
        checks.append(Check(f"{phase} generated config exists", path.exists(), str(path)))
        if not path.exists():
            continue
        cfg = load_merged_config(base_config, path)
        checks.append(
            Check(
                f"{phase} config targets expected canary model",
                expected_pretrained_model is not None
                and cfg.model.pretrained_model_name_or_path == expected_pretrained_model,
                str(cfg.model.pretrained_model_name_or_path),
            )
        )
        if phase in {"agentic", "full"}:
            checks.append(
                Check(
                    f"{phase} config uses expected OpenClaw model",
                    cfg.agentic_math.openclaw_model == openclaw_model
                    and cfg.swebench_pro.openclaw_model == openclaw_model,
                    f"{cfg.agentic_math.openclaw_model}, {cfg.swebench_pro.openclaw_model}",
                )
            )
        if phase == "agentic" and require_nemoclaw:
            math_sandbox = cfg_get(cfg, "agentic_math.nemoclaw_sandbox")
            swe_sandbox = cfg_get(cfg, "swebench_pro.nemoclaw_sandbox")
            task_agent_enabled = cfg_get(cfg, "agentic_math.use_task_agent") is not False
            swe_transfer_mode = cfg_get(cfg, "swebench_pro.nemoclaw_checkout_transfer_mode")
            swe_sandbox_root = cfg_get(cfg, "swebench_pro.nemoclaw_checkout_sandbox_root")
            checks.extend(
                [
                    Check(
                        "agentic config uses NeMoClaw sandbox",
                        math_sandbox == nemoclaw_sandbox and swe_sandbox == nemoclaw_sandbox,
                        f"agentic_math={math_sandbox!r}, swebench_pro={swe_sandbox!r}, "
                        f"expected={nemoclaw_sandbox!r}",
                    ),
                    Check(
                        "agentic config keeps task-agent enabled",
                        task_agent_enabled,
                        f"agentic_math.use_task_agent={cfg_get(cfg, 'agentic_math.use_task_agent')!r}",
                    ),
                    Check(
                        "agentic SWE checkout is sandbox-accessible",
                        swe_transfer_mode == "copy" or nonempty_string(swe_sandbox_root),
                        f"nemoclaw_checkout_transfer_mode={swe_transfer_mode!r}, "
                        f"nemoclaw_checkout_sandbox_root={swe_sandbox_root!r}",
                    ),
                    deny_policy_check(
                        cfg,
                        "agentic_math",
                        "deny_tool",
                        AGENTIC_REQUIRED_DENIED_TOOLS,
                        "Math",
                    ),
                    deny_policy_check(
                        cfg,
                        "swebench_pro",
                        "deny_tool",
                        AGENTIC_REQUIRED_DENIED_TOOLS,
                        "SWE",
                    ),
                    deny_policy_check(
                        cfg,
                        "agentic_math",
                        "deny_argument_pattern",
                        AGENTIC_REQUIRED_DENIED_ARGUMENT_PATTERNS,
                        "Math",
                    ),
                    deny_policy_check(
                        cfg,
                        "swebench_pro",
                        "deny_argument_pattern",
                        AGENTIC_REQUIRED_DENIED_ARGUMENT_PATTERNS,
                        "SWE",
                    ),
                ]
            )
    return checks


def check_execution_plans(output_root: Path, *, canary_slug: str) -> list[Check]:
    expected = {
        "canary_nonagentic_execution_plan.json": "nonagentic",
        "canary_agentic_execution_plan.json": "agentic",
        "canary_agentic_aggregate_execution_plan.json": "agentic_aggregate",
        "canary_full_execution_plan.json": "full",
    }
    checks: list[Check] = []
    for filename, phase in expected.items():
        path = output_root / filename
        checks.append(Check(f"{filename} exists", path.exists(), str(path)))
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        checks.append(
            Check(
                f"{filename} is canary {phase}",
                bool(data.get("canary"))
                and data.get("phase") == phase
                and data.get("model_count") == 1
                and any(canary_slug in item for item in data.get("configs", [])),
                json.dumps(
                    {
                        "phase": data.get("phase"),
                        "canary": data.get("canary"),
                        "model_count": data.get("model_count"),
                    },
                    ensure_ascii=False,
                ),
            )
        )
    return checks


def check_artifacts(
    base_config: Path,
    phase_configs: list[Path],
    artifacts_root: Path,
    *,
    allow_wandb_download: bool,
    verify_wandb_artifacts: bool,
) -> list[Check]:
    artifacts: dict[str, str] = {}
    for config in phase_configs:
        if config.exists():
            artifacts.update(enabled_artifacts(load_merged_config(base_config, config)))
    wandb_metadata: dict[str, str] = {}
    wandb_errors: dict[str, str] = {}
    if verify_wandb_artifacts and allow_wandb_download:
        try:
            import wandb

            api = wandb.Api(timeout=60)
            for label, path in sorted(artifacts.items()):
                try:
                    artifact = api.artifact(path)
                    wandb_metadata[label] = (
                        f"W&B metadata OK: {artifact.name} type={artifact.type} "
                        f"version={artifact.version}"
                    )
                except Exception as exc:  # noqa: BLE001 - report exact W&B lookup failure.
                    wandb_errors[label] = f"W&B metadata lookup failed: {exc!r}"
        except Exception as exc:  # noqa: BLE001 - readiness should report import/auth failures.
            for label in artifacts:
                wandb_errors[label] = f"W&B artifact verification unavailable: {exc!r}"
    checks: list[Check] = []
    for label, path in sorted(artifacts.items()):
        local_cache = local_artifact_exists(artifacts_root, path)
        if label in wandb_metadata:
            source = (
                f"local cache; {wandb_metadata[label]}"
                if local_cache
                else wandb_metadata[label]
            )
            ok = True
        elif label in wandb_errors:
            source = f"local cache; {wandb_errors[label]}" if local_cache else wandb_errors[label]
            ok = local_cache
        else:
            source = "local cache" if local_cache else "W&B download available" if allow_wandb_download else "missing"
            ok = local_cache or allow_wandb_download
        checks.append(
            Check(
                f"artifact source for {label}",
                ok,
                f"{source}: {path}",
            )
        )
    return checks


def check_docs(paths: list[Path]) -> list[Check]:
    return [Check(f"doc exists: {path.name}", path.exists(), str(path)) for path in paths]


def check_weave_content_canary_gate(path: Path | None, *, require: bool) -> list[Check]:
    name = "Weave Agents content canary gate passed"
    if path is None:
        return [
            Check(
                name,
                not require,
                "not configured; pass --weave-content-canary-gate to enforce a fresh content canary",
            )
        ]
    if not path.exists():
        return [Check(name, not require, f"missing: {path}")]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [Check(name, False, f"invalid JSON: {path}: {exc}")]
    ok = bool(data.get("ok")) and data.get("status") == "passed"
    return [
        Check(
            name,
            ok,
            json.dumps(
                {
                    "path": str(path),
                    "ok": data.get("ok"),
                    "status": data.get("status"),
                    "failure_kind": data.get("failure_kind"),
                },
                ensure_ascii=False,
            ),
        )
    ]


def default_generated_agentic_dir(*, require_nemoclaw: bool) -> Path:
    if require_nemoclaw:
        return DEFAULT_CANARY_CONFIG_DIRS["agentic_nemoclaw"]
    return DEFAULT_CANARY_CONFIG_DIRS["agentic"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Optional JSON report path.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--canary-slug", help="Expected canary slug. Defaults to the manifest canary.")
    parser.add_argument("--openclaw-model", help="Expected OpenClaw model. Defaults to the manifest canary model.")
    parser.add_argument(
        "--expected-pretrained-model",
        help="Expected generated cfg.model.pretrained_model_name_or_path. Defaults to the canary source config.",
    )
    parser.add_argument("--generated-full-dir", type=Path, default=DEFAULT_CANARY_CONFIG_DIRS["full"])
    parser.add_argument("--generated-nonagentic-dir", type=Path, default=DEFAULT_CANARY_CONFIG_DIRS["nonagentic"])
    parser.add_argument(
        "--generated-agentic-dir",
        type=Path,
        help=(
            "Generated agentic config directory. Defaults to the standard "
            "OpenAI canary dir, or to the NeMoClaw canary dir when "
            "--require-nemoclaw is set."
        ),
    )
    parser.add_argument(
        "--generated-agentic-aggregate-dir",
        type=Path,
        default=DEFAULT_CANARY_CONFIG_DIRS["agentic_aggregate"],
    )
    parser.add_argument("--openclaw-config", type=Path, default=DEFAULT_OPENCLAW_CONFIG)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs" / "taiwan_full_eval")
    parser.add_argument(
        "--verify-wandb-artifacts",
        action="store_true",
        help="Resolve artifact aliases through the W&B API without downloading files.",
    )
    parser.add_argument(
        "--require-nemoclaw",
        action="store_true",
        help="Fail readiness unless NeMoClaw, OpenShell, the sandbox, and sandbox OpenClaw preflight pass.",
    )
    parser.add_argument("--nemoclaw-bin", default="nemoclaw")
    parser.add_argument("--nemoclaw-sandbox", default="nejumi-taiwan")
    parser.add_argument(
        "--weave-content-canary-gate",
        type=Path,
        help="Gate JSON produced by verify_weave_agents_content_canary_result.py.",
    )
    parser.add_argument(
        "--require-weave-content-canary",
        action="store_true",
        help="Fail readiness unless the supplied Weave content canary gate passed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = load_manifest(args.manifest)
    manifest_canaries = canary_models(models)
    canary_model = manifest_canaries[0] if len(manifest_canaries) == 1 else {}
    canary_slug = str(args.canary_slug or canary_model.get("slug") or DEFAULT_CANARY_SLUG)
    openclaw_model = str(
        args.openclaw_model or canary_model.get("openclaw_model") or DEFAULT_OPENCLAW_MODEL
    )
    expected_pretrained_model = (
        args.expected_pretrained_model or infer_expected_pretrained_model(canary_model)
    )
    generated_agentic_dir = args.generated_agentic_dir or default_generated_agentic_dir(
        require_nemoclaw=bool(args.require_nemoclaw)
    )
    config_paths = {
        "full": args.generated_full_dir / f"config-taiwan-full-{canary_slug}.yaml",
        "nonagentic": args.generated_nonagentic_dir / f"config-taiwan-full-{canary_slug}.yaml",
        "agentic": generated_agentic_dir / f"config-taiwan-full-{canary_slug}.yaml",
        "agentic_aggregate": args.generated_agentic_aggregate_dir / f"config-taiwan-full-{canary_slug}.yaml",
    }
    checks: list[Check] = []
    env = load_env(args.env_file)
    if args.verify_wandb_artifacts and env.get("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_API_KEY", env["WANDB_API_KEY"])
    checks.extend(check_manifest(args.manifest, expected_slug=canary_slug))
    checks.extend(
        check_generated_configs(
            DEFAULT_BASE_CONFIG,
            config_paths,
            expected_pretrained_model=expected_pretrained_model,
            openclaw_model=openclaw_model,
            require_nemoclaw=bool(args.require_nemoclaw),
            nemoclaw_sandbox=args.nemoclaw_sandbox,
        )
    )
    checks.extend(check_execution_plans(args.output_root, canary_slug=canary_slug))
    checks.extend(check_openclaw_config(args.openclaw_config.expanduser(), openclaw_model=openclaw_model))
    checks.extend(
        check_nemoclaw(
            env,
            nemoclaw_bin=args.nemoclaw_bin,
            sandbox=args.nemoclaw_sandbox,
            require=bool(args.require_nemoclaw),
        )
    )
    checks.extend(
        check_weave_content_canary_gate(
            args.weave_content_canary_gate,
            require=bool(args.require_weave_content_canary),
        )
    )
    checks.extend(check_env(env, openclaw_model=openclaw_model))
    checks.extend(
        check_artifacts(
            DEFAULT_BASE_CONFIG,
            list(config_paths.values()),
            REPO_ROOT / "artifacts",
            allow_wandb_download=bool(env.get("WANDB_API_KEY")),
            verify_wandb_artifacts=bool(args.verify_wandb_artifacts),
        )
    )
    checks.extend(
        check_docs(
            [
                REPO_ROOT / "docs" / "taiwan_paid_run_review_template.md",
                REPO_ROOT / "docs" / "README_nemoclaw.md",
                REPO_ROOT / "docs" / "nejumi_agent_protocol_2026_04.md",
            ]
        )
    )

    report = {
        "ok": all(check.ok for check in checks),
        "manifest": str(args.manifest),
        "manifest_path": str(args.manifest),
        "canary_slug": canary_slug,
        "openclaw_model": openclaw_model,
        "expected_pretrained_model": expected_pretrained_model,
        "config_paths": [str(path) for path in config_paths.values()],
        "checks": [check.__dict__ for check in checks],
    }
    for check in checks:
        status = "OK" if check.ok else "FAIL"
        print(f"{status:4} {check.name}: {check.detail}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
