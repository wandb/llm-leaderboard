import os
if os.environ.get("NEJUMI_MAIN_STARTED") == "1":
    print("Duplicate invocation detected; skipping.")
    raise SystemExit(0)
os.environ["NEJUMI_MAIN_STARTED"] = "1"

import json
import hashlib
import signal
import time
from pathlib import Path
from argparse import ArgumentParser
from omegaconf import OmegaConf
import questionary
import importlib
import importlib.util

from utils import paginate_choices


def load_local_dotenv() -> None:
    """Load repo-local .env without overriding explicit process env values."""
    if os.environ.get("NEJUMI_DISABLE_DOTENV") == "1":
        return
    dotenv_path = Path.cwd() / ".env"
    if not dotenv_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(dotenv_path=dotenv_path, override=False)


class LazyEvaluatorModule:
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.module = None

    def __getattr__(self, item):
        if self.module is None:
            self.module = importlib.import_module(f"evaluator.{self.module_name}")
        return getattr(self.module, item)


jaster = LazyEvaluatorModule("jaster")
jbbq = LazyEvaluatorModule("jbbq")
mtbench = LazyEvaluatorModule("mtbench")
jaster_translation = LazyEvaluatorModule("jaster_translation")
toxicity = LazyEvaluatorModule("toxicity")
bfcl = LazyEvaluatorModule("bfcl")
jtruthfulqa = LazyEvaluatorModule("jtruthfulqa")
hle = LazyEvaluatorModule("hle")
hallulens = LazyEvaluatorModule("hallulens")
hallulens_zh_tw = LazyEvaluatorModule("hallulens_zh_tw")
m_ifeval = LazyEvaluatorModule("m_ifeval")
aggregate = LazyEvaluatorModule("aggregate")
aggregate_taiwan = LazyEvaluatorModule("aggregate_taiwan")
swe_bench = LazyEvaluatorModule("swe_bench")
swebench_pro = LazyEvaluatorModule("swebench_pro")
deepswe = LazyEvaluatorModule("deepswe")
agentic_swe_assorted = LazyEvaluatorModule("agentic_swe_assorted")
agentic_math = LazyEvaluatorModule("agentic_math")
arc_agi = LazyEvaluatorModule("arc_agi")
ifeval_zh_tw = LazyEvaluatorModule("ifeval_zh_tw")
ts_bench = LazyEvaluatorModule("ts_bench")
twbias = LazyEvaluatorModule("twbias")
tceval_v2 = LazyEvaluatorModule("tceval_v2")
script_adherence = LazyEvaluatorModule("script_adherence")

BENCHMARK_MAP = {
    'bfcl': 'bfcl',
    'agentic_math': 'agentic_math',
    'swebench': 'swebench',
    'swebench_pro': 'swebench_pro',
    'deepswe': 'deepswe',
    'agentic_swe_assorted': 'agentic_swe_assorted',
    'mtbench': 'mtbench',
    'script_adherence': 'script_adherence',
    'jbbq': 'jbbq',
    'toxicity': 'toxicity',
    'jtruthfulqa': 'jtruthfulqa',
    'hle': 'hle',
    'hallulens': 'hallulens',
    'hallulens_zh_tw': 'hallulens_zh_tw',
    'arc_agi': 'arc_agi',
    'm_ifeval': 'm_ifeval',
    'ifeval_zh_tw': 'ifeval_zh_tw',
    'ts_bench': 'ts_bench',
    'twbias': 'twbias',
    'tceval_v2': 'tceval_v2',
    'jaster': 'jaster',
    'aggregate': 'aggregate',
    'aggregate_taiwan': 'aggregate_taiwan',
}

AUXILIARY_RUN_FLAGS = {
    # Consumed inside jaster.evaluate() rather than dispatched as a standalone evaluator.
    "jmmlu_robustness",
    "tmmluplus_robustness",
}

KNOWN_RUN_FLAGS = set(BENCHMARK_MAP) | AUXILIARY_RUN_FLAGS


def load_validate_all_benchmarks():
    validation_helpers_path = (
        Path(__file__).resolve().parent
        / "evaluator"
        / "evaluate_utils"
        / "validation_helpers.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_nejumi_validation_helpers",
        validation_helpers_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load validation helper: {validation_helpers_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_all_benchmarks


validate_all_benchmarks = load_validate_all_benchmarks()


def _run_flags_dict(cfg) -> dict:
    run_cfg = OmegaConf.select(cfg, "run", default={})
    if run_cfg is None:
        return {}
    return OmegaConf.to_container(run_cfg, resolve=True) or {}


def is_run_flag_enabled(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def enabled_benchmarks_from_config(cfg):
    enabled = []
    for bench_key, bench_name in BENCHMARK_MAP.items():
        if is_run_flag_enabled(OmegaConf.select(cfg, f"run.{bench_key}", default=False)):
            enabled.append(bench_name)
    return enabled


def dispatch_validation_from_config(cfg):
    run_flags = _run_flags_dict(cfg)
    unsupported_truthy = sorted(
        key
        for key, value in run_flags.items()
        if is_run_flag_enabled(value) and key not in KNOWN_RUN_FLAGS
    )
    return {
        "ok": not unsupported_truthy,
        "known_run_flags": sorted(KNOWN_RUN_FLAGS),
        "dispatched_run_flags": [
            key
            for key in BENCHMARK_MAP
            if is_run_flag_enabled(run_flags.get(key, False))
        ],
        "auxiliary_run_flags": [
            key
            for key in sorted(AUXILIARY_RUN_FLAGS)
            if is_run_flag_enabled(run_flags.get(key, False))
        ],
        "unsupported_truthy_run_flags": unsupported_truthy,
    }


def summarize_wandb_scope_validation(cfg):
    entity = str(OmegaConf.select(cfg, "wandb.entity", default="") or "").strip()
    project = str(OmegaConf.select(cfg, "wandb.project", default="") or "").strip()
    run_name = str(OmegaConf.select(cfg, "wandb.run_name", default="") or "").strip()
    expected_entity = str(
        OmegaConf.select(cfg, "wandb.expected_entity", default="") or ""
    ).strip()
    expected_project = str(
        OmegaConf.select(cfg, "wandb.expected_project", default="") or ""
    ).strip()
    errors = []

    if run_name.startswith("taiwan/") and not (
        expected_entity and expected_project
    ):
        errors.append(
            "Taiwan W&B runs must declare wandb.expected_entity and "
            "wandb.expected_project. This prevents a Taiwan config from being "
            "merged with the Japanese/default base config."
        )
    if expected_entity and entity != expected_entity:
        errors.append(
            "W&B entity scope mismatch: "
            f"configured={entity!r}, expected={expected_entity!r}"
        )
    if expected_project and project != expected_project:
        errors.append(
            "W&B project scope mismatch: "
            f"configured={project!r}, expected={expected_project!r}"
        )

    return {
        "ok": not errors,
        "entity": entity,
        "project": project,
        "run_name": run_name,
        "expected_entity": expected_entity or None,
        "expected_project": expected_project or None,
        "errors": errors,
    }


def summarize_token_validation(cfg, enabled_benchmarks):
    validation_results = validate_all_benchmarks(cfg)
    rows = []
    has_warnings = False
    has_errors = False

    for benchmark, (is_valid, message) in validation_results.items():
        if benchmark not in enabled_benchmarks:
            continue
        severity = "ok"
        if not is_valid:
            if "❌" in message:
                has_errors = True
                severity = "error"
            else:
                has_warnings = True
                severity = "warning"
        rows.append(
            {
                "benchmark": benchmark,
                "ok": bool(is_valid),
                "severity": severity,
                "message": message,
            }
        )

    return {
        "ok": not has_errors,
        "has_errors": has_errors,
        "has_warnings": has_warnings,
        "results": rows,
    }


def summarize_runtime_validation(cfg, enabled_benchmarks):
    errors = []
    warnings = []
    credential_checks = []
    benchmark_preflights = []
    wandb_scope = summarize_wandb_scope_validation(cfg)
    errors.extend(wandb_scope["errors"])

    def require_number(
        path: str,
        *,
        minimum: float,
        inclusive: bool = False,
        default=None,
    ) -> None:
        raw = OmegaConf.select(cfg, path, default=default)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{path} must be numeric; got {raw!r}")
            return
        valid = value >= minimum if inclusive else value > minimum
        if not valid:
            operator = ">=" if inclusive else ">"
            errors.append(f"{path} must be {operator} {minimum}; got {value}")

    require_number("batch_size", minimum=0, default=256)
    require_number("network.retry.max_time_sec", minimum=0, default=1800)
    require_number("network.retry.max_tries", minimum=0, default=50)
    require_number(
        "network.progress_interval_sec",
        minimum=0,
        inclusive=True,
        default=60,
    )
    for timeout_name in ("connect", "read", "write", "pool"):
        require_number(
            f"network.http_timeout.{timeout_name}",
            minimum=0,
            default=300 if timeout_name in {"read", "write"} else 30,
        )
    require_number(
        "provider_rate_limit.min_request_interval_sec",
        minimum=0,
        inclusive=True,
        default=0,
    )
    require_number(
        "provider_rate_limit.request_jitter_sec",
        minimum=0,
        inclusive=True,
        default=0,
    )
    benchmark_timeout_paths = {
        "agentic_math": (
            ("agentic_math.benchmark_timeout_seconds", 21_600),
            ("agentic_math.preflight_timeout_seconds", 180),
        ),
        "agentic_swe_assorted": (
            ("agentic_swe_assorted.benchmark_timeout_seconds", 50_400),
            ("agentic_swe_assorted.preflight_timeout_seconds", 180),
        ),
        "swebench_pro": (
            ("swebench_pro.benchmark_timeout_seconds", 28_800),
            ("swebench_pro.grading_timeout_seconds", 14_400),
        ),
        "deepswe": (
            ("deepswe.benchmark_timeout_seconds", 129_600),
        ),
    }
    for benchmark_name, timeout_paths in benchmark_timeout_paths.items():
        if benchmark_name not in enabled_benchmarks:
            continue
        for path, default in timeout_paths:
            require_number(path, minimum=0, default=default)

    def env_is_present(name: str | None) -> bool:
        if not name:
            return False
        value = os.environ.get(name)
        if value is None:
            return False
        return value.strip() not in {"", "EMPTY", "empty", "NONE", "none", "null", "NULL"}

    def require_any_env(label: str, env_names: list[str]) -> None:
        env_names = [name for name in env_names if name]
        present = [name for name in env_names if env_is_present(name)]
        credential_checks.append(
            {
                "label": label,
                "required_any_of": env_names,
                "present_envs": present,
                "ok": bool(present),
            }
        )
        if not present:
            errors.append(
                f"Missing credential for {label}. Set one of: {', '.join(env_names)}. "
                "The evaluation is blocked before W&B/model execution to avoid invalid paid runs."
            )

    def configured_api_key_env() -> str | None:
        return (
            OmegaConf.select(cfg, "api_key_env", default=None)
            or OmegaConf.select(cfg, "openai_compatible.api_key_env", default=None)
        )

    def require_answer_model_credentials() -> None:
        api = str(OmegaConf.select(cfg, "api", default=""))
        custom_env = configured_api_key_env()
        if custom_env:
            require_any_env(f"answer model API key env {custom_env}", [str(custom_env)])
            return

        if api in {"openai", "openai_chat", "openai_responses"}:
            require_any_env("answer model OpenAI API", ["OPENAI_API_KEY"])
        elif api in {"xai", "xai_responses"}:
            require_any_env("answer model xAI API", ["XAI_API_KEY"])
        elif api == "deepseek":
            require_any_env("answer model DeepSeek API", ["DEEPSEEK_API_KEY", "OPENAI_COMPATIBLE_API_KEY"])
        elif api == "google":
            require_any_env("answer model Google API", ["GOOGLE_API_KEY"])
        elif api == "anthropic":
            require_any_env("answer model Anthropic API", ["ANTHROPIC_API_KEY"])
        elif api == "mistral":
            require_any_env("answer model Mistral API", ["MISTRAL_API_KEY"])
        elif api == "cohere":
            require_any_env("answer model Cohere API", ["COHERE_API_KEY"])
        elif api == "upstage":
            require_any_env("answer model Upstage API", ["UPSTAGE_API_KEY"])
        elif api == "azure-openai":
            require_any_env("answer model Azure OpenAI endpoint", ["AZURE_OPENAI_ENDPOINT"])
            require_any_env("answer model Azure OpenAI API", ["AZURE_OPENAI_API_KEY"])
        elif api == "openai-compatible":
            base_url = str(OmegaConf.select(cfg, "base_url", default="")).lower()
            if "openrouter.ai" in base_url:
                override_env = os.environ.get("NEJUMI_OPENROUTER_API_KEY_ENV")
                require_any_env(
                    "answer model OpenRouter API",
                    [override_env] if override_env else ["OPENROUTER_API_KEY", "OPENAI_COMPATIBLE_API_KEY"],
                )
            elif "api.x.ai" in base_url:
                require_any_env("answer model xAI-compatible API", ["XAI_API_KEY", "OPENAI_COMPATIBLE_API_KEY"])
            elif "api.inference.wandb.ai" in base_url:
                require_any_env(
                    "answer model W&B Inference API",
                    ["WANDB_API_KEY", "OPENAI_COMPATIBLE_API_KEY"],
                )
            elif any(host in base_url for host in ["localhost", "127.0.0.1", "0.0.0.0", "vllm"]):
                return
            else:
                require_any_env("answer model OpenAI-compatible API", ["OPENAI_COMPATIBLE_API_KEY", "VLLM_API_KEY"])

    def require_judge_credentials() -> None:
        judge_paths = {
            "mtbench": "mtbench.judge.model",
            "hle": "hle.judge.model",
            "hallulens": "hallulens.judge.model",
            "hallulens_zh_tw": "hallulens_zh_tw.judge.model",
            "toxicity": "toxicity.judge.model",
        }
        for benchmark, model_path in judge_paths.items():
            if benchmark not in enabled_benchmarks:
                continue
            model = str(OmegaConf.select(cfg, model_path, default="")).strip()
            if not model:
                continue
            label = f"{benchmark} judge model {model}"
            forced_provider = os.environ.get("NEJUMI_JUDGE_PROVIDER", "").strip().lower()
            if model.startswith("openrouter/") or forced_provider == "openrouter":
                override_env = os.environ.get("NEJUMI_OPENROUTER_API_KEY_ENV")
                require_any_env(
                    label,
                    [override_env] if override_env else ["OPENROUTER_API_KEY", "OPENAI_COMPATIBLE_API_KEY"],
                )
            elif model.startswith("deepseek/") or forced_provider == "deepseek":
                require_any_env(label, ["DEEPSEEK_API_KEY", "OPENAI_COMPATIBLE_API_KEY"])
            elif model.startswith("xai/") or forced_provider == "xai":
                require_any_env(label, ["XAI_API_KEY", "OPENAI_COMPATIBLE_API_KEY"])
            elif os.environ.get("OPENAI_API_TYPE", "openai") == "azure":
                require_any_env(f"{label} Azure OpenAI endpoint", ["AZURE_OPENAI_ENDPOINT"])
                require_any_env(label, ["AZURE_OPENAI_API_KEY"])
            else:
                require_any_env(label, ["OPENAI_API_KEY"])

    require_answer_model_credentials()
    require_judge_credentials()

    if "bfcl" in enabled_benchmarks:
        version = str(
            OmegaConf.select(cfg, "bfcl.version", default="v3")
        ).strip().lower()
        if version == "v4":
            raw_categories = OmegaConf.select(
                cfg, "bfcl.test_category", default=[]
            )
            categories = (
                raw_categories.split()
                if isinstance(raw_categories, str)
                else list(raw_categories or [])
            )
            uses_memory = any(
                category.startswith("memory_") for category in categories
            )
            uses_web = any(
                category.startswith("web_search_") for category in categories
            )
            backend_aliases = {
                "ddgs": "ddgs",
                "direct": "ddgs",
                "direct_search": "ddgs",
                "multi_engine": "ddgs",
                "duckduckgo": "duckduckgo_html",
                "duckduckgo_html": "duckduckgo_html",
                "ddg": "duckduckgo_html",
                "ddg_html": "duckduckgo_html",
                "serpapi": "serpapi",
                "serp_api": "serpapi",
            }
            raw_backend = str(
                OmegaConf.select(
                    cfg,
                    "bfcl.web_search.backend",
                    default="ddgs",
                )
            ).strip().lower().replace("-", "_")
            backend = backend_aliases.get(raw_backend)
            if uses_web and backend is None:
                errors.append(
                    "Unsupported BFCL v4 web search backend "
                    f"{raw_backend!r}; use ddgs, duckduckgo_html, or serpapi."
                )

            required_modules = {}
            if uses_memory:
                required_modules.update(
                    {
                        "rank_bm25": "rank-bm25",
                        "sentence_transformers": "sentence-transformers",
                        "faiss": "faiss-cpu",
                    }
                )
            if uses_web:
                required_modules["html2text"] = "html2text"
                if backend == "ddgs":
                    required_modules["ddgs"] = "ddgs"
                elif backend == "duckduckgo_html":
                    required_modules.update(
                        {
                            "bs4": "beautifulsoup4",
                            "requests": "requests",
                        }
                    )
                elif backend == "serpapi":
                    required_modules["serpapi"] = "google-search-results"
                    require_any_env(
                        "BFCL v4 SerpAPI web search",
                        ["SERPAPI_API_KEY"],
                    )

            missing_packages = sorted(
                package
                for module, package in required_modules.items()
                if importlib.util.find_spec(module) is None
            )
            if missing_packages:
                errors.append(
                    "BFCL v4 agentic runtime dependencies are missing: "
                    + ", ".join(missing_packages)
                    + ". Install the locked project environment before "
                    "starting the run."
                )

    if (
        "agentic_swe_assorted" in enabled_benchmarks
        and bool(
            OmegaConf.select(
                cfg,
                "agentic_swe_assorted.run_openclaw",
                default=True,
            )
        )
    ):
        configured_output = Path(
            str(
                OmegaConf.select(
                    cfg,
                    "agentic_swe_assorted.output_dir",
                    default="outputs/agentic_swe_assorted",
                )
            )
        )
        try:
            preflight_result = agentic_swe_assorted.preflight(
                cfg,
                configured_output / ".run_eval_preflight",
            )
        except Exception as exc:
            preflight_result = {
                "benchmark": "agentic_swe_assorted",
                "ok": False,
                "returncode": None,
                "report_path": None,
                "report": None,
                "error": f"{type(exc).__name__}: {exc}",
                "will_run_model": False,
                "will_run_gateway": False,
                "will_run_grading": False,
                "will_initialize_wandb": False,
            }
        benchmark_preflights.append(preflight_result)
        if not preflight_result["ok"]:
            detail = preflight_result.get("error") or "unknown static preflight error"
            errors.append(
                "Agentic SWE-Assorted static preflight failed before paid execution: "
                + detail
            )

    if "twbias" in enabled_benchmarks:
        backend = str(OmegaConf.select(cfg, "twbias.backend", default="hf_perplexity"))
        api = str(OmegaConf.select(cfg, "api", default=""))
        model_path = OmegaConf.select(cfg, "twbias.model_path", default=None)
        if backend != "hf_perplexity":
            errors.append(f"TWBias backend {backend!r} is not supported by scripts/evaluator/twbias.py.")
        elif api not in {"vllm", "vllm-docker", "vllm-local"} and not model_path:
            errors.append(
                "TWBias hf_perplexity requires direct HF/local model access. "
                f"api={api!r} with twbias.model_path unset would fail before scoring. "
                "Disable run.twbias or use a local/HF model config for TWBias internal tests."
            )
        if not bool(OmegaConf.select(cfg, "twbias.allow_unknown_license", default=False)):
            warnings.append(
                "TWBias source license is recorded as unknown in the prepared manifest; "
                "set twbias.allow_unknown_license=true only for internal verification."
            )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "wandb_scope": wandb_scope,
        "credential_checks": credential_checks,
        "benchmark_preflights": benchmark_preflights,
    }


def print_token_validation_summary(validation_summary):
    print("\n" + "=" * 80)
    print("🔍 GLOBAL TOKEN ALLOCATION VALIDATION")
    print("=" * 80)
    for row in validation_summary["results"]:
        print(row["message"])
    print("=" * 80)

    if validation_summary["has_errors"]:
        print("\n❌ CRITICAL: Some benchmarks have insufficient output tokens!")
        print("   This will likely cause empty responses and unfairly low scores.")
    elif validation_summary["has_warnings"]:
        print("\n⚠️  WARNING: Some benchmarks have suboptimal token allocation.")
    else:
        print("\n✅ All token allocations look good!")


def build_preflight_payload(custom_cfg_path, base_cfg_path, cfg, enabled_benchmarks):
    validation_summary = summarize_token_validation(cfg, enabled_benchmarks)
    dispatch_validation = dispatch_validation_from_config(cfg)
    runtime_validation = summarize_runtime_validation(cfg, enabled_benchmarks)
    ok = (
        bool(validation_summary["ok"])
        and bool(dispatch_validation["ok"])
        and bool(runtime_validation["ok"])
    )
    return {
        "schema_version": 1,
        "generated_at": time.time(),
        "status": "passed" if ok else "failed",
        "ok": ok,
        "config": str(custom_cfg_path),
        "base_config": str(base_cfg_path),
        "wandb": {
            "entity": OmegaConf.select(cfg, "wandb.entity"),
            "project": OmegaConf.select(cfg, "wandb.project"),
            "run_name": OmegaConf.select(cfg, "wandb.run_name"),
        },
        "api": OmegaConf.select(cfg, "api"),
        "model": OmegaConf.select(cfg, "model.pretrained_model_name_or_path"),
        "enabled_benchmarks": enabled_benchmarks,
        "scheduled_evaluators": enabled_benchmarks,
        "dispatch_validation": dispatch_validation,
        "will_initialize_wandb": False,
        "will_log_wandb_artifacts": False,
        "will_initialize_weave": False,
        "will_start_inference_engine": False,
        "will_run_evaluators": False,
        "token_validation": validation_summary,
        "runtime_validation": runtime_validation,
    }


def summarize_execution_readiness(cfg, run, enabled_benchmarks):
    """Run checks that require downloaded artifacts, before any model execution."""
    checks = []
    errors = []

    if "agentic_math" in enabled_benchmarks:
        configured_output = Path(
            str(
                OmegaConf.select(
                    cfg,
                    "agentic_math.output_dir",
                    default="outputs/agentic_math",
                )
            )
        )
        try:
            result = agentic_math.preflight(
                cfg,
                run,
                configured_output / ".execution_readiness",
            )
        except Exception as exc:
            result = {
                "benchmark": "agentic_math",
                "ok": False,
                "returncode": None,
                "report_path": None,
                "report": None,
                "error": f"{type(exc).__name__}: {exc}",
                "will_run_model": False,
                "will_run_gateway": False,
                "will_run_grading": False,
                "will_initialize_wandb": False,
            }
        checks.append(result)
        if not result["ok"]:
            errors.append(
                "Agentic Math execution-readiness check failed before model "
                f"execution: {result.get('error') or 'unknown error'}"
            )

    if (
        "agentic_swe_assorted" in enabled_benchmarks
        and bool(
            OmegaConf.select(
                cfg,
                "agentic_swe_assorted.run_openclaw",
                default=True,
            )
        )
    ):
        configured_output = Path(
            str(
                OmegaConf.select(
                    cfg,
                    "agentic_swe_assorted.output_dir",
                    default="outputs/agentic_swe_assorted",
                )
            )
        )
        try:
            result = agentic_swe_assorted.preflight(
                cfg,
                configured_output / ".execution_readiness",
            )
        except Exception as exc:
            result = {
                "benchmark": "agentic_swe_assorted",
                "ok": False,
                "returncode": None,
                "report_path": None,
                "report": None,
                "error": f"{type(exc).__name__}: {exc}",
                "will_run_model": False,
                "will_run_gateway": False,
                "will_run_grading": False,
                "will_initialize_wandb": False,
            }
        checks.append(result)
        if not result["ok"]:
            errors.append(
                "Agentic SWE-Assorted execution-readiness check failed before "
                f"model execution: {result.get('error') or 'unknown error'}"
            )

    return {
        "ok": not errors,
        "errors": errors,
        "checks": checks,
        "will_run_model": False,
    }

# Set config path
config_dir = Path("configs")
base_cfg_name = "base_config.yaml"
parser = ArgumentParser()
parser.add_argument("--config", "-c", type=str)
parser.add_argument("--select-config", "-s", action="store_true", default=False)
parser.add_argument("--base-config", type=str, default=base_cfg_name)
parser.add_argument("--yes", "-y", action="store_true", default=False)
parser.add_argument(
    "--allow-invalid-token-allocation",
    action="store_true",
    default=False,
    help=(
        "Explicitly bypass critical benchmark token-allocation errors. "
        "--yes alone never bypasses these errors."
    ),
)
parser.add_argument(
    "--preflight",
    action="store_true",
    default=False,
    help="Load and validate the merged config, then exit before W&B, Weave, model, or evaluator execution.",
)
parser.add_argument(
    "--preflight-json",
    type=str,
    default=None,
    help="Optional JSON output path for --preflight.",
)
args = parser.parse_args()
load_local_dotenv()

if args.select_config:
    choices = sorted([p.name for p in config_dir.iterdir() if p.suffix == ".yaml"])
    if len(choices) > 36:
        custom_cfg_name = paginate_choices(choices)
    else:
        custom_cfg_name = questionary.select(
            "Select config",
            choices=choices,
            use_shortcuts=True,
        ).ask()
    custom_cfg_path = config_dir / custom_cfg_name
elif args.config:
    custom_cfg_path = config_dir / args.config
else:
    raise ValueError("No arguments found. Please specify either --config or --select-config.")

if custom_cfg_path.suffix != ".yaml":
    custom_cfg_path = custom_cfg_path.with_suffix(".yaml")
assert custom_cfg_path.exists(), f"Config file {custom_cfg_path.resolve()} does not exist"

# Configuration loading
custom_cfg = OmegaConf.load(custom_cfg_path)
base_cfg_path = config_dir / args.base_config
base_cfg = OmegaConf.load(base_cfg_path)

# vLLM利用時にbase_urlが未指定の場合、Composeサービス名 'vllm' をデフォルト設定
if "api" in custom_cfg and custom_cfg.api in ["vllm", "vllm-docker"]:
    if "base_url" not in custom_cfg:
        print("INFO: api is vllm/vllm-docker and base_url is not set. Defaulting to http://vllm:8000/v1")
        custom_cfg.base_url = "http://vllm:8000/v1"

custom_cfg = OmegaConf.merge(base_cfg, custom_cfg)
cfg_dict = OmegaConf.to_container(custom_cfg, resolve=True)
assert isinstance(cfg_dict, dict), "instance.config must be a DictConfig"
enabled_benchmarks = enabled_benchmarks_from_config(custom_cfg)

if args.preflight:
    payload = build_preflight_payload(
        custom_cfg_path=custom_cfg_path,
        base_cfg_path=base_cfg_path,
        cfg=custom_cfg,
        enabled_benchmarks=enabled_benchmarks,
    )
    print_token_validation_summary(payload["token_validation"])
    print("\nPreflight summary:")
    print(f"  config: {payload['config']}")
    print(f"  base_config: {payload['base_config']}")
    print(f"  model: {payload['model']}")
    print(f"  enabled_benchmarks: {', '.join(enabled_benchmarks) if enabled_benchmarks else '(none)'}")
    print(
        "  wandb_target: "
        f"{payload['wandb']['entity']}/{payload['wandb']['project']}"
    )
    print(
        "  wandb_scope_contract: "
        f"{'passed' if payload['runtime_validation']['wandb_scope']['ok'] else 'failed'}"
    )
    print("  W&B/Weave/model/evaluator execution: skipped")
    if payload["runtime_validation"]["errors"]:
        print("  runtime_errors:")
        for error in payload["runtime_validation"]["errors"]:
            print(f"    - {error}")

    if args.preflight_json:
        preflight_json_path = Path(args.preflight_json)
        preflight_json_path.parent.mkdir(parents=True, exist_ok=True)
        preflight_json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"  preflight_json: {preflight_json_path}")

    raise SystemExit(0 if payload["ok"] else 2)

dispatch_validation = dispatch_validation_from_config(custom_cfg)
runtime_validation = summarize_runtime_validation(custom_cfg, enabled_benchmarks)

if not dispatch_validation["ok"]:
    raise SystemExit(
        "Unsupported truthy run flag(s): "
        + ", ".join(dispatch_validation["unsupported_truthy_run_flags"])
        + ". Add an evaluator dispatch or mark the flag as auxiliary before running."
    )

if not runtime_validation["ok"]:
    raise SystemExit(
        "Runtime validation failed before W&B/model execution:\n"
        + "\n".join(f"- {error}" for error in runtime_validation["errors"])
    )

import wandb
import weave
from blend_run import blend_run
from config_singleton import WandbConfigSingleton
from docker_vllm_manager import stop_vllm_container_if_needed, start_vllm_container_if_needed
from eval_output_scoping import apply_run_scoped_outputs
from llm_inference_adapter import get_llm_inference_engine
from vllm_server import shutdown_vllm_server

# プログレストラッカーをインポート
from evaluator.evaluate_utils.progress_tracker import (
    initialize_progress_tracker, start_benchmark_tracking,
    complete_benchmark_tracking, finish_progress_tracking
)
from evaluator.evaluate_utils.benchmark_checkpoint import BenchmarkCheckpointStore

# 環境変数からAPIキーを取得
def get_api_key_from_env(service_name):
    """環境変数からAPIキーを取得"""
    env_var_name = f"{service_name.upper()}_API_KEY"
    return os.getenv(env_var_name)

# W&B APIキーの設定
wandb_api_key = get_api_key_from_env("wandb")
if wandb_api_key:
    os.environ["WANDB_API_KEY"] = wandb_api_key
    print(f"Wandb API key loaded from environment variable")
else:
    print("Warning: WANDB_API_KEY not found in environment variables")

"""Safeguard W&B init in multi-run environments"""
wandb_run = os.environ.get("WANDB_RUN_ID")
allow_wandb_resume = os.environ.get("NEJUMI_ALLOW_WANDB_RESUME") == "1"
try:
    if os.environ.get("NEJUMI_WANDB_INIT_DONE") == "1":
        print("W&B already initialized; reusing existing run context.")
        run = wandb.run  # may be None if not active
    else:
        wandb.login()
        run = wandb.init(
            entity=cfg_dict["wandb"]["entity"],
            project=cfg_dict["wandb"]["project"],
            name=cfg_dict["wandb"]["run_name"],
            config=cfg_dict,
            job_type="evaluation",
            id=wandb_run if wandb_run else None,
            resume=("allow" if allow_wandb_resume else "never") if wandb_run else None,
            settings=wandb.Settings(init_timeout=300),
        )
        if run is not None and getattr(run, "id", None):
            os.environ["WANDB_RUN_ID"] = str(run.id)
            cfg_dict, output_scoping = apply_run_scoped_outputs(
                cfg_dict,
                run_id=str(run.id),
            )
            if output_scoping.get("applied"):
                run.config.update(cfg_dict, allow_val_change=True)
                print(
                    "Run-scoped output root: "
                    f"{output_scoping['run_root']} "
                    f"(wandb_run_id={output_scoping['run_id']})"
                )
        os.environ["NEJUMI_WANDB_INIT_DONE"] = "1"
except Exception as e:
    raise SystemExit(
        "Failed to initialize W&B. W&B is required for this evaluation because "
        "artifacts, run metadata, and result logging must all be tracked there.\n"
        f"Original error: {e}\n"
        "Check your WANDB_API_KEY / login state and verify the target entity/project."
    ) from e

_wandb_run_finished = False


def _finish_wandb_run(exit_code: int = 0) -> None:
    global _wandb_run_finished
    if run and not _wandb_run_finished:
        _wandb_run_finished = True
        run.finish(exit_code=exit_code)


def _finish_wandb_run_on_signal(signum, frame) -> None:
    print(
        f"Received signal {signum}; finishing W&B run before exit.",
        flush=True,
    )
    _finish_wandb_run(exit_code=128 + int(signum))
    raise SystemExit(128 + int(signum))


signal.signal(signal.SIGINT, _finish_wandb_run_on_signal)
signal.signal(signal.SIGTERM, _finish_wandb_run_on_signal)

# Initialize Weave separately so Weave failures don't disable W&B
if run:
    try:
        weave.init(cfg_dict["wandb"]["entity"]+"/"+cfg_dict["wandb"]["project"])
    except Exception as e:
        print(f"Warning: Failed to initialize Weave: {e}")
        print("Continuing without Weave...")

WandbConfigSingleton.initialize(run, llm=None, config_override=cfg_dict)
cfg = WandbConfigSingleton.get_instance().config

# Resolve remote datasets and validate mutable runtime prerequisites before the
# inference engine or any paid benchmark request starts.
execution_readiness = summarize_execution_readiness(
    cfg,
    run,
    enabled_benchmarks,
)
if run is not None:
    run.summary["execution_readiness_ok"] = bool(execution_readiness["ok"])
    run.summary["execution_readiness_check_count"] = len(
        execution_readiness["checks"]
    )
if not execution_readiness["ok"]:
    print("\nExecution readiness failed before model execution:", flush=True)
    for error in execution_readiness["errors"]:
        print(f"  - {error}", flush=True)
    _finish_wandb_run(exit_code=2)
    raise SystemExit(2)

# Save configuration as artifact
artifact = wandb.Artifact("config", type="config")
artifact.add_file(custom_cfg_path)
run.log_artifact(artifact)

# Inherit old runs
blend_run(run_chain=True)

# グローバルトークンバリデーション実行
try:
    validation_summary = summarize_token_validation(cfg, enabled_benchmarks)
    print_token_validation_summary(validation_summary)
except Exception as e:
    print(f"Token validation failed before model execution: {e}", flush=True)
    _finish_wandb_run(exit_code=2)
    raise SystemExit(2) from e

if validation_summary["has_errors"] and not args.allow_invalid_token_allocation:
    print(
        "Critical token-allocation errors block model execution. "
        "Fix the benchmark configuration or use "
        "--allow-invalid-token-allocation for an intentional experiment.",
        flush=True,
    )
    _finish_wandb_run(exit_code=2)
    raise SystemExit(2)
if validation_summary["has_errors"]:
    print(
        "WARNING: explicitly bypassing critical token-allocation errors.",
        flush=True,
    )
elif validation_summary["has_warnings"]:
    response = "y" if args.yes else input("\nContinue? (Y/n): ").strip().lower()
    if response in ['n', 'no']:
        print("Evaluation aborted by user.")
        _finish_wandb_run(exit_code=1)
        raise SystemExit(1)

# プログレストラッカーを初期化
tracker = initialize_progress_tracker(enabled_benchmarks)
tracker.start_tracking()
checkpoint_run_root = Path(
    str(
        OmegaConf.select(
            cfg,
            "output.resolved_run_root",
            default=f"outputs/taiwan_full_eval_runs/{getattr(run, 'id', 'local')}",
        )
    )
)


def _plain_config_value(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _benchmark_code_fingerprint(benchmark_name):
    scripts_root = Path(__file__).resolve().parent
    evaluator_root = scripts_root / "evaluator"
    paths = [
        evaluator_root / f"{benchmark_name}.py",
    ]
    direct_inference_benchmarks = set(BENCHMARK_MAP) - {
        "agentic_math",
        "swebench",
        "swebench_pro",
        "deepswe",
        "agentic_swe_assorted",
        "bfcl",
        "aggregate",
        "aggregate_taiwan",
    }
    if benchmark_name in direct_inference_benchmarks:
        paths.extend(
            [
                scripts_root / "llm_inference_adapter.py",
                evaluator_root / "evaluate_utils" / "llm_async_processor.py",
                evaluator_root / "evaluate_utils" / "llm_response_checkpoint.py",
                evaluator_root / "evaluate_utils" / "provider_rate_limiter.py",
            ]
        )
    if benchmark_name == "agentic_math":
        paths.extend(
            [
                scripts_root / "tools" / "run_agentic_math_openclaw.py",
                scripts_root / "tools" / "run_openclaw_agent_protocol.py",
            ]
        )
    if benchmark_name in {"swebench", "swebench_pro"}:
        paths.append(scripts_root / "tools" / "run_swebench_pro_openclaw.py")
    if benchmark_name == "deepswe":
        paths.append(scripts_root / "tools" / "run_deepswe_openclaw.py")
    if benchmark_name == "agentic_swe_assorted":
        paths.extend(
            [
                scripts_root / "tools" / "run_agentic_swe_assorted.py",
                scripts_root / "tools" / "run_swebench_pro_openclaw.py",
                scripts_root / "tools" / "run_deepswe_openclaw.py",
            ]
        )
    if benchmark_name == "jaster":
        paths.append(evaluator_root / "jaster_translation.py")
    if benchmark_name == "script_adherence":
        paths.append(evaluator_root / "mtbench.py")
    if benchmark_name == "bfcl":
        paths.extend(
            [
                evaluator_root / "bfcl.py",
                evaluator_root / "bfcl_v4.py",
                evaluator_root
                / "evaluate_utils"
                / "bfcl_pkg"
                / "bfcl"
                / "_llm_response_generation.py",
                evaluator_root
                / "evaluate_utils"
                / "bfcl_v4_pkg"
                / "bfcl_eval"
                / "_llm_response_generation.py",
            ]
        )
        paths.extend(
            sorted(
                path
                for package in ("bfcl_pkg", "bfcl_v4_pkg")
                for path in (evaluator_root / "evaluate_utils" / package).rglob("*.py")
            )
        )
    if benchmark_name == "aggregate_taiwan":
        paths.append(
            Path(__file__).resolve().parents[1]
            / "taxonomies"
            / "nejumi45_taiwan.yaml"
        )

    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(Path(__file__).resolve().parents[1])).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _benchmark_config_fingerprint(cfg, benchmark_name):
    common_keys = (
        "api",
        "model",
        "generator",
        "testmode",
        "batch_size",
        "inference_interval",
        "error_handling",
        "network",
        "provider_rate_limit",
        "num_few_shots",
    )
    payload = {
        "schema_version": 2,
        "benchmark": benchmark_name,
        "code": _benchmark_code_fingerprint(benchmark_name),
        "common": {
            key: _plain_config_value(
                OmegaConf.select(cfg, key, default=None)
            )
            for key in common_keys
        },
        "benchmark_config": _plain_config_value(
            OmegaConf.select(cfg, benchmark_name, default=None)
        ),
    }
    if benchmark_name == "jaster":
        payload["jaster_translation"] = _plain_config_value(
            OmegaConf.select(cfg, "jaster_translation", default=None)
        )
        payload["tmmluplus_robustness"] = bool(
            OmegaConf.select(cfg, "run.tmmluplus_robustness", default=False)
        )
    if benchmark_name in {"aggregate", "aggregate_taiwan"}:
        payload["run"] = _plain_config_value(
            OmegaConf.select(cfg, "run", default={})
        )
        payload["taiwan_aggregate"] = _plain_config_value(
            OmegaConf.select(cfg, "taiwan_aggregate", default=None)
        )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


benchmark_checkpoints = BenchmarkCheckpointStore(
    checkpoint_run_root / "benchmark_checkpoints",
    run_id=str(getattr(run, "id", "local")),
    resume_enabled=allow_wandb_resume,
    config_fingerprints={
        benchmark_name: _benchmark_config_fingerprint(cfg, benchmark_name)
        for benchmark_name in BENCHMARK_MAP
    },
)
allow_legacy_benchmark_checkpoint_migration = bool(
    OmegaConf.select(
        cfg,
        "output.allow_legacy_benchmark_checkpoint_migration",
        default=False,
    )
)
trusted_fingerprint_mismatch_benchmarks = set(
    OmegaConf.select(
        cfg,
        "output.trust_completed_checkpoint_fingerprint_mismatch",
        default=[],
    )
    or []
)


def _benchmark_is_resumable(benchmark_name):
    if not benchmark_checkpoints.is_completed(benchmark_name):
        raw_checkpoint = benchmark_checkpoints.load_raw(benchmark_name)
        remote_completed = bool(
            run is not None
            and run.summary.get(
                f"benchmark_completed_{benchmark_name}",
                False,
            )
        )
        legacy_completed = bool(
            raw_checkpoint
            and raw_checkpoint.get("status") == "completed"
            and "config_fingerprint" not in raw_checkpoint
        )
        if (
            allow_legacy_benchmark_checkpoint_migration
            and legacy_completed
            and remote_completed
        ):
            benchmark_checkpoints.mark_completed(benchmark_name)
            if run is not None:
                run.summary[
                    f"benchmark_checkpoint_migrated_{benchmark_name}"
                ] = True
            print(
                "Migrated legacy completion marker after matching local run ID "
                f"and W&B completion evidence: {benchmark_name}",
                flush=True,
            )
            return True
        fingerprint_mismatch_completed = bool(
            raw_checkpoint
            and raw_checkpoint.get("status") == "completed"
            and raw_checkpoint.get("config_fingerprint")
        )
        if (
            benchmark_name in trusted_fingerprint_mismatch_benchmarks
            and fingerprint_mismatch_completed
            and remote_completed
        ):
            benchmark_checkpoints.mark_completed(benchmark_name)
            if run is not None:
                run.summary[
                    f"benchmark_checkpoint_fingerprint_migrated_{benchmark_name}"
                ] = True
            print(
                "Migrated explicitly trusted completion marker after matching "
                "local run ID and W&B completion evidence: "
                f"{benchmark_name}",
                flush=True,
            )
            return True
        return False
    if run is None:
        return True
    remote_completed = bool(
        run.summary.get(f"benchmark_completed_{benchmark_name}", False)
    )
    if not remote_completed:
        print(
            "Local completion marker exists but W&B completion evidence is "
            f"missing for {benchmark_name}; re-running safely.",
            flush=True,
        )
    return remote_completed


def _execute_tracked_benchmark(benchmark_name, callback):
    if _benchmark_is_resumable(benchmark_name):
        if run is not None:
            run.summary[f"benchmark_error_{benchmark_name}"] = None
        print(
            f"Skipping completed benchmark on explicit resume: {benchmark_name}",
            flush=True,
        )
        complete_benchmark_tracking(benchmark_name, {"resumed": 1})
        return None

    start_benchmark_tracking(benchmark_name)
    benchmark_checkpoints.mark_started(benchmark_name)
    try:
        result = callback()
    except BaseException as exc:
        benchmark_checkpoints.mark_failed(benchmark_name, exc)
        if run is not None:
            run.summary[f"benchmark_completed_{benchmark_name}"] = False
            run.summary[f"benchmark_error_{benchmark_name}"] = (
                f"{type(exc).__name__}: {exc}"
            )
        raise
    if run is not None:
        run.summary[f"benchmark_completed_{benchmark_name}"] = True
        run.summary[f"benchmark_error_{benchmark_name}"] = None
    benchmark_checkpoints.mark_completed(benchmark_name)
    complete_benchmark_tracking(benchmark_name)
    return result

# vLLMコンテナの起動処理を追加
# Start vLLM container if needed (for vllm/vllm-docker API types)
# # Start vLLM container if needed
# if cfg.api in ["vllm-docker"]:
#     print(f"Starting vLLM container for model: {cfg.model.pretrained_model_name_or_path}")
#     
#     # 環境変数を設定（コンテナが正しいモデルを使用するために必要）
#     os.environ["EVAL_CONFIG_PATH"] = str(custom_cfg_path.name)
#     
#     # vLLMコンテナを起動（モデル名を明示的に渡す）
#     success = start_vllm_container_if_needed(model_name=cfg.model.pretrained_model_name_or_path)
#     if success is False:
#         print("Failed to start vLLM container, aborting evaluation")
#         if run:
#             wandb.finish()
#         exit(1)
#     elif success is True:
#         print("vLLM container started successfully")

# Start inference server
llm = get_llm_inference_engine()
if run:
    instance = WandbConfigSingleton.get_instance()
    instance.llm = llm

# BFCL
if cfg.run.bfcl:
    _execute_tracked_benchmark("bfcl", bfcl.evaluate)

# Agentic Math evaluation
if cfg.run.get('agentic_math', False):
    _execute_tracked_benchmark("agentic_math", agentic_math.evaluate)

# SWE-Bench Verified evaluation
if cfg.run.swebench:
    if cfg.swebench.background_eval:
        if _benchmark_is_resumable("swebench"):
            print("Skipping completed benchmark on explicit resume: swebench", flush=True)
            complete_benchmark_tracking("swebench", {"resumed": 1})
            swebench_postprocess = None
        else:
            start_benchmark_tracking("swebench")
            benchmark_checkpoints.mark_started("swebench")
            swebench_postprocess = swe_bench.evaluate()
    else:
        _execute_tracked_benchmark("swebench", swe_bench.evaluate)

# SWE-bench Pro agentic evaluation
if cfg.run.get('swebench_pro', False):
    _execute_tracked_benchmark("swebench_pro", swebench_pro.evaluate)

# DeepSWE agentic evaluation
if cfg.run.get('deepswe', False):
    _execute_tracked_benchmark("deepswe", deepswe.evaluate)

# Agentic SWE-Assorted evaluation
if cfg.run.get('agentic_swe_assorted', False):
    _execute_tracked_benchmark(
        "agentic_swe_assorted",
        agentic_swe_assorted.evaluate,
    )

# mt-bench evaluation
if is_run_flag_enabled(cfg.run.get("mtbench", False)):
    _execute_tracked_benchmark("mtbench", mtbench.evaluate)

# Traditional Chinese script adherence, derived from mtbench_output_table.
if is_run_flag_enabled(cfg.run.get("script_adherence", False)):
    _execute_tracked_benchmark("script_adherence", script_adherence.evaluate)

# jbbq
if cfg.run.jbbq:
    _execute_tracked_benchmark("jbbq", jbbq.evaluate)

# toxicity
if cfg.run.toxicity:
    _execute_tracked_benchmark("toxicity", toxicity.evaluate)

# JTruthfulQA
if cfg.run.jtruthfulqa:
    _execute_tracked_benchmark("jtruthfulqa", jtruthfulqa.evaluate)

# hle
if cfg.run.hle:
    _execute_tracked_benchmark("hle", hle.evaluate)

# HalluLens
if cfg.run.hallulens:
    _execute_tracked_benchmark("hallulens", hallulens.evaluate)

# HalluLens zh-TW
if is_run_flag_enabled(cfg.run.get("hallulens_zh_tw", False)):
    _execute_tracked_benchmark("hallulens_zh_tw", hallulens_zh_tw.evaluate)

# ARC-AGI
if cfg.run.arc_agi:
    _execute_tracked_benchmark("arc_agi", arc_agi.evaluate)

# M-IFEval
if cfg.run.m_ifeval:
    _execute_tracked_benchmark("m_ifeval", m_ifeval.evaluate)

# IFEval zh-TW
if is_run_flag_enabled(cfg.run.get("ifeval_zh_tw", False)):
    _execute_tracked_benchmark("ifeval_zh_tw", ifeval_zh_tw.evaluate)

# TS-Bench
if is_run_flag_enabled(cfg.run.get("ts_bench", False)):
    _execute_tracked_benchmark("ts_bench", ts_bench.evaluate)

# TWBias
if is_run_flag_enabled(cfg.run.get("twbias", False)):
    _execute_tracked_benchmark("twbias", twbias.evaluate)

# TCEval-v2 selected
if is_run_flag_enabled(cfg.run.get("tceval_v2", False)):
    _execute_tracked_benchmark("tceval_v2", tceval_v2.evaluate)

# Evaluation phase
def _evaluate_jaster_complete():
    jaster.evaluate()
    #### open weight model base evaluation
    # 1. evaluation for translation task in jaster with comet
    # APIタイプに応じてvLLMサーバー/コンテナをシャットダウン
    lifecycle_mode = cfg.vllm.get("lifecycle", "auto")
    
    # jaster_translation (COMET) や jtruthfulqa (RoBERTa) はGPUメモリを大きく消費するため、
    # vLLMを一時停止する必要がある。
    # lifecycle: 'always_on' が指定されている場合を除く
    if lifecycle_mode != 'always_on':
        if cfg.api == "vllm-local":
            shutdown_vllm_server()
        elif cfg.api in ["vllm", "vllm-docker"]:
            stop_vllm_container_if_needed()
    
    # COMET評価を実行
    jaster_translation.evaluate()
    
    # APIタイプに応じてvLLMサーバー/コンテナを再起動
    if lifecycle_mode != 'always_on':
        if cfg.api == "vllm-local":
            llm = get_llm_inference_engine()
            if run:
                instance.llm = llm
        elif cfg.api in ["vllm", "vllm-docker"]:
            start_vllm_container_if_needed(model_name=cfg.model.pretrained_model_name_or_path)
            # llmインスタンスは同じものを使い続ける
            pass


if cfg.run.jaster:
    _execute_tracked_benchmark("jaster", _evaluate_jaster_complete)

if cfg.run.swebench and cfg.swebench.background_eval:
    # SWE-Bench評価完了を待ってから集計・W&Bロギングを確実に実施
    if callable(swebench_postprocess):
        try:
            swebench_postprocess()
        except BaseException as exc:
            benchmark_checkpoints.mark_failed("swebench", exc)
            if run is not None:
                run.summary["benchmark_completed_swebench"] = False
                run.summary["benchmark_error_swebench"] = (
                    f"{type(exc).__name__}: {exc}"
                )
            raise
        benchmark_checkpoints.mark_completed("swebench")
        if run is not None:
            run.summary["benchmark_completed_swebench"] = True
            run.summary["benchmark_error_swebench"] = None
        complete_benchmark_tracking("swebench")
    elif not _benchmark_is_resumable("swebench"):
        raise RuntimeError(
            "SWE-Bench background evaluation returned no completion callback"
        )

# Aggregation
if cfg.run.aggregate:
    _execute_tracked_benchmark("aggregate", aggregate.evaluate)

# Taiwan leaderboard aggregation
if is_run_flag_enabled(cfg.run.get("aggregate_taiwan", False)):
    _execute_tracked_benchmark("aggregate_taiwan", aggregate_taiwan.evaluate)

# プログレストラッキング終了
finish_progress_tracking()

# 評価完了後、vLLMコンテナを停止
if cfg.api in ["vllm", "vllm-docker"]:
    print("Stopping vLLM container...")
    # stop_vllm_container_if_needed()
    pass

# Finish
if run:
    _finish_wandb_run(exit_code=0)
