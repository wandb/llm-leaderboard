"""Taiwan-leaderboard integration for the vendored BFCL v4 runtime."""

from __future__ import annotations

import json
import importlib.util
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pandas as pd
import wandb
from omegaconf import OmegaConf

from config_singleton import WandbConfigSingleton


V4_PACKAGE_ROOT = (
    Path(__file__).resolve().parent
    / "evaluate_utils"
    / "bfcl_v4_pkg"
)
CORE_CATEGORIES = (
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "irrelevance",
    "live_simple",
    "live_multiple",
    "live_irrelevance",
    "live_relevance",
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
)
AGENTIC_CATEGORIES = (
    "memory_kv",
    "memory_vector",
    "memory_rec_sum",
    "web_search_base",
    "web_search_no_snippet",
)
WEB_SEARCH_BACKENDS = {
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


class BFCLInfrastructureError(RuntimeError):
    """Raised when BFCL has non-model failures after bounded recovery."""


def _plain_dict(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        return OmegaConf.to_container(value, resolve=True) or {}
    except Exception:
        return {}


def _categories(value: Any) -> list[str]:
    if isinstance(value, str):
        return value.split()
    return list(value or [])


def _resolve_data_root(artifact_dir: str | Path) -> Path:
    root = Path(artifact_dir).resolve()
    candidates = [root]
    candidates.extend(path.parent for path in root.rglob("selection_metadata.json"))
    candidates.extend(path.parent for path in root.rglob("BFCL_v4_simple_python.json"))
    for candidate in candidates:
        if (candidate / "BFCL_v4_simple_python.json").is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate BFCL v4 data files under artifact: {root}"
    )


def _install_v4_runtime(data_root: Path, project_root: Path) -> None:
    package_root = str(V4_PACKAGE_ROOT)
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    os.environ["BFCL_DATA_ROOT"] = str(data_root)
    os.environ["BFCL_PROJECT_ROOT"] = str(project_root)
    os.environ["BFCL_IGNORE_MISSING_CATEGORIES"] = "1"


def _load_profile_metadata(
    data_root: Path,
    *,
    categories: list[str],
    expected_profile: str | None,
) -> dict[str, Any]:
    metadata_path = data_root / "selection_metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(
            "BFCL v4 artifact is missing selection_metadata.json"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("benchmark_version") != "v4":
        raise RuntimeError(
            "BFCL artifact benchmark_version must be v4, got "
            f"{metadata.get('benchmark_version')!r}"
        )
    if metadata.get("locale") != "zh-Hant-TW":
        raise RuntimeError(
            "BFCL v4 production artifact must use locale zh-Hant-TW, got "
            f"{metadata.get('locale')!r}"
        )
    profile = str(metadata.get("profile", ""))
    if expected_profile and profile != expected_profile:
        raise RuntimeError(
            f"BFCL v4 artifact profile is {profile!r}, expected "
            f"{expected_profile!r}"
        )
    artifact_categories = set(metadata.get("categories") or [])
    missing = sorted(set(categories) - artifact_categories)
    if missing:
        raise RuntimeError(
            "BFCL v4 artifact does not contain configured categories: "
            + ", ".join(missing)
        )
    if any("parallel" in category for category in artifact_categories):
        raise RuntimeError(
            "Taiwan BFCL v4 artifact unexpectedly contains parallel categories"
        )
    if int(metadata.get("max_multi_turn_turns", -1)) != 3:
        raise RuntimeError(
            "Taiwan BFCL v4 artifact must cap multi-turn cases at 3 turns"
        )
    if int(metadata.get("max_per_category", -1)) != 30:
        raise RuntimeError(
            "Taiwan BFCL v4 artifact must cap categories at 30 cases"
        )
    return metadata


def _validate_evaluation_profile(profile: str, categories: list[str]) -> None:
    expected = set(CORE_CATEGORIES)
    if profile == "full":
        expected.update(AGENTIC_CATEGORIES)
    elif profile != "core":
        raise ValueError(f"Unsupported BFCL v4 profile: {profile!r}")
    configured = set(categories)
    if configured != expected or len(categories) != len(configured):
        missing = sorted(expected - configured)
        unexpected = sorted(configured - expected)
        raise ValueError(
            f"BFCL v4 {profile!r} profile category mismatch; "
            f"missing={missing}, unexpected={unexpected}, "
            f"duplicates={len(categories) != len(configured)}"
        )


def _normalized_web_search_config(value: Any) -> dict[str, Any]:
    config = {
        "backend": "ddgs",
        "ddgs_backend": "auto",
        "endpoint": "https://html.duckduckgo.com/html/",
        "timeout_sec": 20.0,
        "max_attempts": 4,
        "retry_base_sec": 2.0,
        "min_request_interval_sec": 2.0,
        "request_jitter_sec": 0.25,
        "cache_path": "outputs/bfcl_v4/web_search_cache.sqlite3",
        "cache_ttl_sec": 0.0,
        "fetch_max_bytes": 2_000_000,
        "fetch_max_chars": 20_000,
        **_plain_dict(value),
    }
    raw_backend = str(config["backend"]).strip().lower().replace("-", "_")
    try:
        config["backend"] = WEB_SEARCH_BACKENDS[raw_backend]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported bfcl.web_search.backend {raw_backend!r}; "
            "expected ddgs, duckduckgo_html, or serpapi"
        ) from exc
    config["ddgs_backend"] = (
        str(config.get("ddgs_backend") or "auto").strip() or "auto"
    )
    try:
        config["timeout_sec"] = float(config["timeout_sec"])
        config["max_attempts"] = int(config["max_attempts"])
        config["retry_base_sec"] = float(config["retry_base_sec"])
        config["min_request_interval_sec"] = float(
            config["min_request_interval_sec"]
        )
        config["request_jitter_sec"] = float(
            config["request_jitter_sec"]
        )
        config["cache_ttl_sec"] = float(config["cache_ttl_sec"])
        config["fetch_max_bytes"] = int(config["fetch_max_bytes"])
        config["fetch_max_chars"] = int(config["fetch_max_chars"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "BFCL v4 web search timing and retry settings must be numeric"
        ) from exc
    if config["timeout_sec"] <= 0:
        raise ValueError("bfcl.web_search.timeout_sec must be positive")
    if config["max_attempts"] < 1:
        raise ValueError("bfcl.web_search.max_attempts must be at least 1")
    if config["fetch_max_bytes"] < 1:
        raise ValueError("bfcl.web_search.fetch_max_bytes must be positive")
    if config["fetch_max_chars"] < 1:
        raise ValueError("bfcl.web_search.fetch_max_chars must be positive")
    for key in (
        "retry_base_sec",
        "min_request_interval_sec",
        "request_jitter_sec",
        "cache_ttl_sec",
    ):
        if config[key] < 0:
            raise ValueError(f"bfcl.web_search.{key} must not be negative")
    endpoint = str(config.get("endpoint") or "").strip()
    if config["backend"] == "duckduckgo_html" and not endpoint.startswith(
        ("http://", "https://")
    ):
        raise ValueError(
            "bfcl.web_search.endpoint must be an HTTP(S) URL for "
            "duckduckgo_html"
        )
    config["endpoint"] = endpoint
    return config


def _configure_web_search(
    config: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    cache_value = str(config.get("cache_path") or "").strip()
    cache_path = (
        str(_absolute_path(cache_value, repo_root))
        if cache_value
        else ""
    )
    environment = {
        "BFCL_WEB_SEARCH_BACKEND": config["backend"],
        "BFCL_WEB_SEARCH_DDGS_BACKEND": config["ddgs_backend"],
        "BFCL_WEB_SEARCH_ENDPOINT": str(config["endpoint"]),
        "BFCL_WEB_SEARCH_TIMEOUT_SEC": str(config["timeout_sec"]),
        "BFCL_WEB_SEARCH_MAX_ATTEMPTS": str(config["max_attempts"]),
        "BFCL_WEB_SEARCH_RETRY_BASE_SEC": str(config["retry_base_sec"]),
        "BFCL_WEB_SEARCH_MIN_INTERVAL_SEC": str(
            config["min_request_interval_sec"]
        ),
        "BFCL_WEB_SEARCH_JITTER_SEC": str(config["request_jitter_sec"]),
        "BFCL_WEB_SEARCH_CACHE_PATH": cache_path,
        "BFCL_WEB_SEARCH_CACHE_TTL_SEC": str(config["cache_ttl_sec"]),
        "BFCL_WEB_FETCH_MAX_BYTES": str(config["fetch_max_bytes"]),
        "BFCL_WEB_FETCH_MAX_CHARS": str(config["fetch_max_chars"]),
    }
    os.environ.update(environment)
    return {
        **config,
        "cache_path": cache_path,
    }


def _validate_agentic_dependencies(
    categories: list[str],
    *,
    web_search_config: dict[str, Any],
) -> None:
    required_modules: dict[str, str] = {}
    if any(category.startswith("memory_") for category in categories):
        required_modules.update(
            {
                "rank_bm25": "rank-bm25",
                "sentence_transformers": "sentence-transformers",
                "faiss": "faiss-cpu",
            }
        )
    if any(category.startswith("web_search_") for category in categories):
        required_modules["html2text"] = "html2text"
        if web_search_config["backend"] == "ddgs":
            required_modules["ddgs"] = "ddgs"
        elif web_search_config["backend"] == "duckduckgo_html":
            required_modules.update(
                {
                    "bs4": "beautifulsoup4",
                    "requests": "requests",
                }
            )
        elif web_search_config["backend"] == "serpapi":
            required_modules["serpapi"] = "google-search-results"
            if not os.environ.get("SERPAPI_API_KEY"):
                raise RuntimeError(
                    "SERPAPI_API_KEY is required when "
                    "bfcl.web_search.backend=serpapi."
                )
    missing = [
        package
        for module, package in required_modules.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise RuntimeError(
            "BFCL v4 agentic dependencies are missing: "
            + ", ".join(sorted(missing))
            + ". Install the project environment before starting a paid run."
        )


def _absolute_path(value: Any, repo_root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (repo_root / path).resolve()


def _percent_to_float(value: Any) -> float:
    if value is None or pd.isna(value) or value in {"", "N/A"}:
        return 0.0
    text = str(value).strip()
    if text.endswith("%"):
        return float(text[:-1]) / 100.0
    return float(text)


def _flatten_numeric_sum(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return sum(_flatten_numeric_sum(item) for item in value)
    return 0.0


def _stable_table_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _prompt_text(entry: dict[str, Any]) -> str:
    prompt = entry.get("prompt", entry)
    question = prompt.get("question", []) if isinstance(prompt, dict) else []
    parts: list[str] = []
    for turn_index, turn in enumerate(question, start=1):
        if not isinstance(turn, list):
            continue
        contents = [
            str(message["content"])
            for message in turn
            if isinstance(message, dict) and message.get("content")
        ]
        if contents:
            prefix = f"Turn {turn_index}: " if len(question) > 1 else ""
            parts.append(prefix + " ".join(contents))
    return " | ".join(parts)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _collect_retryable_generation_failures(
    result_dir: Path,
    model_registry_name: str,
) -> list[dict[str, str]]:
    from bfcl_eval._llm_response_generation import _retryable_failed_result

    model_dir = result_dir / model_registry_name.replace("/", "_")
    failures: list[dict[str, str]] = []
    for path in model_dir.rglob("BFCL_v4_*_result.json"):
        for row in _read_jsonl(path):
            if _retryable_failed_result(row):
                failures.append(
                    {
                        "id": str(row.get("id") or ""),
                        "error": str(row.get("error") or "inference_error"),
                        "path": str(path),
                    }
                )
    return sorted(failures, key=lambda row: (row["id"], row["path"]))


def _run_generation_with_recovery(
    *,
    generation_main: Callable[[SimpleNamespace], Any],
    generation_args: SimpleNamespace,
    result_dir: Path,
    model_registry_name: str,
    recovery_rounds: int,
    recovery_base_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, str]]:
    generation_main(generation_args)
    failures = _collect_retryable_generation_failures(
        result_dir,
        model_registry_name,
    )
    for recovery_round in range(1, recovery_rounds + 1):
        if not failures:
            break
        delay = recovery_base_seconds * (2 ** (recovery_round - 1))
        ids = ", ".join(row["id"] for row in failures)
        print(
            "BFCL v4 infrastructure recovery "
            f"{recovery_round}/{recovery_rounds}: retrying "
            f"{len(failures)} failed cases after {delay:.1f}s: {ids}",
            flush=True,
        )
        if delay > 0:
            sleep(delay)
        generation_args.force_retry_ids = [
            row["id"] for row in failures
        ]
        generation_summary = generation_main(generation_args)
        generated_case_count = (
            generation_summary.get("generated_case_count")
            if isinstance(generation_summary, dict)
            else None
        )
        if generated_case_count == 0:
            raise BFCLInfrastructureError(
                "BFCL v4 recovery made no progress: the generation layer "
                "selected zero cases while retryable failures remained"
            )
        failures = _collect_retryable_generation_failures(
            result_dir,
            model_registry_name,
        )
    generation_args.force_retry_ids = []
    return failures


def _collect_output_rows(
    result_dir: Path,
    score_dir: Path,
    model_registry_name: str,
) -> tuple[list[dict[str, Any]], int, int, int]:
    from bfcl_eval._llm_response_generation import _retryable_failed_result
    from bfcl_eval.utils import extract_test_category

    model_dir_name = model_registry_name.replace("/", "_")
    result_model_dir = result_dir / model_dir_name
    score_model_dir = score_dir / model_dir_name

    result_by_id: dict[str, dict[str, Any]] = {}
    category_by_id: dict[str, str] = {}
    for path in result_model_dir.rglob("BFCL_v4_*_result.json"):
        category = extract_test_category(path)
        for row in _read_jsonl(path):
            result_by_id[str(row["id"])] = row
            category_by_id[str(row["id"])] = category

    output_rows: list[dict[str, Any]] = []
    scored_ids: set[str] = set()
    for path in score_model_dir.rglob("BFCL_v4_*_score.json"):
        category = extract_test_category(path)
        rows = _read_jsonl(path)
        for row in rows[1:]:
            entry_id = str(row.get("id", ""))
            scored_ids.add(entry_id)
            raw = result_by_id.get(entry_id, {})
            output_rows.append(
                {
                    "model": model_registry_name,
                    "id": entry_id,
                    "category": category,
                    "prompt": _prompt_text(row),
                    "output": json.dumps(
                        row.get(
                            "model_result_raw",
                            row.get("model_result", raw.get("result", "")),
                        ),
                        ensure_ascii=False,
                        default=str,
                    ),
                    "accuracy": int(bool(row.get("valid", False))),
                    "possible_answer": json.dumps(
                        row.get("possible_answer", ""),
                        ensure_ascii=False,
                        default=str,
                    ),
                    "reasoning_content": json.dumps(
                        raw.get("reasoning_content", ""),
                        ensure_ascii=False,
                        default=str,
                    ),
                    "input_token_count": _flatten_numeric_sum(
                        raw.get("input_token_count", 0)
                    ),
                    "output_token_count": _flatten_numeric_sum(
                        raw.get("output_token_count", 0)
                    ),
                    "timeout": bool(raw.get("timeout")),
                    "error": _stable_table_text(
                        raw.get("error", row.get("error", ""))
                    ),
                }
            )

    for entry_id, raw in result_by_id.items():
        if entry_id in scored_ids:
            continue
        output_rows.append(
            {
                "model": model_registry_name,
                "id": entry_id,
                "category": category_by_id.get(entry_id, ""),
                "prompt": "",
                "output": json.dumps(
                    raw.get("result", ""), ensure_ascii=False, default=str
                ),
                "accuracy": 0,
                "possible_answer": "",
                "reasoning_content": json.dumps(
                    raw.get("reasoning_content", ""),
                    ensure_ascii=False,
                    default=str,
                ),
                "input_token_count": _flatten_numeric_sum(
                    raw.get("input_token_count", 0)
                ),
                "output_token_count": _flatten_numeric_sum(
                    raw.get("output_token_count", 0)
                ),
                "timeout": bool(raw.get("timeout")),
                "error": _stable_table_text(raw.get("error", "")),
            }
        )

    timeout_count = sum(bool(row.get("timeout")) for row in result_by_id.values())
    inference_error_count = sum(
        _retryable_failed_result(row)
        for row in result_by_id.values()
    )
    model_error_count = sum(
        row.get("error") == "model_inference_error"
        for row in result_by_id.values()
    )
    return (
        output_rows,
        timeout_count,
        inference_error_count,
        model_error_count,
    )


def _validate_reused_generation_results(
    *,
    result_dir: Path,
    model_registry_name: str,
    categories: list[str],
    profile_metadata: dict[str, Any],
) -> dict[str, Any]:
    model_dir = result_dir / model_registry_name.replace("/", "_")
    if not model_dir.is_dir():
        raise FileNotFoundError(
            "BFCL v4 reused result directory does not contain the configured model: "
            f"{model_dir}"
        )

    rows_by_id: dict[str, dict[str, Any]] = {}
    files = []
    for category in categories:
        matches = list(model_dir.rglob(f"BFCL_v4_{category}_result.json"))
        if not matches:
            raise FileNotFoundError(
                f"BFCL v4 reused results are missing category {category!r} under "
                f"{model_dir}"
            )
        for path in matches:
            files.append(path)
            for row in _read_jsonl(path):
                entry_id = str(row.get("id") or "")
                if not entry_id:
                    raise ValueError(f"BFCL v4 reused result has no id: {path}")
                if entry_id in rows_by_id:
                    raise ValueError(
                        f"BFCL v4 reused results contain duplicate id {entry_id!r}"
                    )
                rows_by_id[entry_id] = row

    expected_count = sum(
        int(profile_metadata["category_counts"][category])
        for category in categories
    )
    if len(rows_by_id) != expected_count:
        raise ValueError(
            "BFCL v4 reused result coverage mismatch: "
            f"{len(rows_by_id)} unique cases found, {expected_count} expected"
        )
    return {
        "result_dir": str(result_dir),
        "model_dir": str(model_dir),
        "result_file_count": len(files),
        "unique_case_count": len(rows_by_id),
        "expected_case_count": expected_count,
    }


def evaluate_v4() -> None:
    print("BFCL v4 evaluation started")
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    bfcl_cfg = {
        "temperature": 0.01,
        "num_threads": 4,
        "include_input_log": False,
        "exclude_state_log": False,
        "allow_overwrite": True,
        "run_ids": False,
        "resume_existing_results": True,
        "retry_failed_cases": True,
        "run_generation": True,
        "case_timeout_sec": 600,
        "watchdog_interval_sec": 30,
        "stall_fail_fast_sec": 900,
        "consecutive_failure_fail_fast": 5,
        "infrastructure_recovery_rounds": 2,
        "infrastructure_recovery_base_seconds": 60,
        "profile": "full",
        "artifact_profile": "full",
        "web_search": {},
        **_plain_dict(cfg.get("bfcl", {})),
    }
    repo_root = Path(__file__).resolve().parents[2]
    model_registry_name = str(cfg.model.bfcl_model_id)
    categories = _categories(bfcl_cfg.get("test_category"))
    if not categories:
        raise ValueError("bfcl.test_category must list BFCL v4 categories")
    if any("parallel" in category for category in categories):
        raise ValueError("Taiwan BFCL v4 profile must not include parallel categories")
    profile = str(bfcl_cfg["profile"])
    _validate_evaluation_profile(profile, categories)
    web_search_config = _normalized_web_search_config(
        bfcl_cfg.get("web_search")
    )
    _validate_agentic_dependencies(
        categories,
        web_search_config=web_search_config,
    )
    web_search_config = _configure_web_search(
        web_search_config,
        repo_root=repo_root,
    )

    artifact = run.use_artifact(bfcl_cfg["artifacts_path"])
    data_root = _resolve_data_root(artifact.download())
    profile_metadata = _load_profile_metadata(
        data_root,
        categories=categories,
        expected_profile=str(bfcl_cfg.get("artifact_profile") or "") or None,
    )

    run_id = str(getattr(run, "id", "local"))
    default_output_root = repo_root / "outputs" / "bfcl_v4" / run_id
    bfcl_cfg.setdefault("result_dir", default_output_root / "result")
    bfcl_cfg.setdefault("score_dir", default_output_root / "score")
    result_dir = _absolute_path(bfcl_cfg["result_dir"], repo_root)
    score_dir = _absolute_path(bfcl_cfg["score_dir"], repo_root)
    result_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)
    _install_v4_runtime(data_root, default_output_root)

    from bfcl_eval._llm_response_generation import main as generation_main
    from bfcl_eval.eval_checker.eval_runner import main as evaluation_main
    if any(category.startswith("web_search_") for category in categories):
        from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.web_search_backends import (
            reset_search_stats,
        )

        reset_search_stats()

    reused_generation = None
    remaining_generation_failures: list[dict[str, str]] = []
    if bool(bfcl_cfg["run_generation"]):
        recovery_rounds = int(bfcl_cfg["infrastructure_recovery_rounds"])
        recovery_base_seconds = float(
            bfcl_cfg["infrastructure_recovery_base_seconds"]
        )
        if recovery_rounds < 0:
            raise ValueError("bfcl.infrastructure_recovery_rounds must be >= 0")
        if recovery_base_seconds < 0:
            raise ValueError(
                "bfcl.infrastructure_recovery_base_seconds must be >= 0"
            )
        generation_args = SimpleNamespace(
            model=model_registry_name,
            test_category=categories,
            temperature=float(bfcl_cfg["temperature"]),
            include_input_log=bool(bfcl_cfg["include_input_log"]),
            exclude_state_log=bool(bfcl_cfg["exclude_state_log"]),
            num_threads=int(bfcl_cfg["num_threads"]),
            num_gpus=int(bfcl_cfg.get("num_gpus") or 1),
            backend=str(bfcl_cfg.get("backend", "vllm")),
            gpu_memory_utilization=float(
                bfcl_cfg.get("gpu_memory_utilization", 0.9)
            ),
            result_dir=result_dir,
            run_ids=bool(bfcl_cfg["run_ids"]),
            allow_overwrite=bool(bfcl_cfg["allow_overwrite"]),
            resume_existing_results=bool(
                bfcl_cfg["resume_existing_results"]
            ),
            retry_failed_cases=bool(bfcl_cfg["retry_failed_cases"]),
            skip_server_setup=bool(bfcl_cfg.get("skip_server_setup", True)),
            local_model_path=bfcl_cfg.get("local_model_path"),
            lora_modules=None,
            enable_lora=False,
            max_lora_rank=None,
            case_timeout_sec=float(bfcl_cfg["case_timeout_sec"]),
            watchdog_interval_sec=float(bfcl_cfg["watchdog_interval_sec"]),
            stall_fail_fast_sec=float(bfcl_cfg["stall_fail_fast_sec"]),
            consecutive_failure_fail_fast=int(
                bfcl_cfg["consecutive_failure_fail_fast"]
            ),
            infrastructure_recovery_rounds=recovery_rounds,
            infrastructure_recovery_base_seconds=recovery_base_seconds,
            force_retry_ids=[],
            max_cases_per_category=(
                int(bfcl_cfg.get("testmode_cases_per_category", 2))
                if bool(getattr(cfg, "testmode", False))
                else None
            ),
        )
        remaining_generation_failures = _run_generation_with_recovery(
            generation_main=generation_main,
            generation_args=generation_args,
            result_dir=result_dir,
            model_registry_name=model_registry_name,
            recovery_rounds=recovery_rounds,
            recovery_base_seconds=recovery_base_seconds,
        )
    else:
        reused_generation = _validate_reused_generation_results(
            result_dir=result_dir,
            model_registry_name=model_registry_name,
            categories=categories,
            profile_metadata=profile_metadata,
        )
        print(
            "BFCL v4 generation reuse validated: "
            f"{reused_generation['unique_case_count']} cases from {result_dir}"
        )

    evaluation_main(
        model=[model_registry_name],
        test_categories=categories,
        result_dir=result_dir,
        score_dir=score_dir,
        partial_eval=bool(getattr(cfg, "testmode", False)),
    )

    overall_df = pd.read_csv(score_dir / "data_overall.csv")
    if overall_df.empty:
        raise RuntimeError("BFCL v4 produced an empty leaderboard")
    overall_df.rename(columns={"Model": "BFCL Model Name"}, inplace=True)
    overall_df["model_name"] = str(cfg.model.pretrained_model_name_or_path)
    (
        output_rows,
        timeout_count,
        inference_error_count,
        model_error_count,
    ) = _collect_output_rows(result_dir, score_dir, model_registry_name)
    web_search_stats: dict[str, Any] = {
        "backend": web_search_config["backend"],
        "queries": 0,
        "cache_hits": 0,
        "network_requests": 0,
        "retries": 0,
        "errors": 0,
        "fetches": 0,
        "fetch_truncations": 0,
        "fetch_errors": 0,
    }
    if any(category.startswith("web_search_") for category in categories):
        from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.web_search_backends import (
            get_search_stats,
        )

        web_search_stats = get_search_stats()
    if bool(getattr(cfg, "testmode", False)):
        testmode_count = int(bfcl_cfg.get("testmode_cases_per_category", 2))
        logical_case_count = sum(
            min(
                int(profile_metadata["category_counts"][category]),
                testmode_count,
            )
            for category in categories
        )
    else:
        logical_case_count = sum(
            int(profile_metadata["category_counts"][category])
            for category in categories
        )
    overall_df["timeout_count"] = timeout_count
    overall_df["inference_error_count"] = inference_error_count
    overall_df["model_error_count"] = model_error_count

    first = overall_df.iloc[0]
    radar_columns = (
        "Non-Live AST Acc",
        "Live Acc",
        "Multi Turn Acc",
        "Web Search Acc",
        "Memory Acc",
        "Irrelevance Detection",
        "Overall Acc",
    )
    radar_df = pd.DataFrame(
        [
            {"category": column, "score": _percent_to_float(first.get(column))}
            for column in radar_columns
        ]
    )

    run.log(
        {
            "bfcl_output_table": wandb.Table(
                dataframe=pd.DataFrame(output_rows)
            ),
            "bfcl_leaderboard_table": wandb.Table(dataframe=overall_df),
            "bfcl_radar_table": wandb.Table(dataframe=radar_df),
            "bfcl_timeout_count": timeout_count,
            "bfcl_inference_error_count": inference_error_count,
            "bfcl_model_error_count": model_error_count,
            "bfcl_timeout_policy": "score_as_incorrect",
            "bfcl_version": "v4",
            "bfcl_profile": profile,
            "bfcl_artifact_profile": profile_metadata["profile"],
            "bfcl_upstream_commit": profile_metadata.get("upstream_commit", ""),
            "bfcl_selection_seed": profile_metadata.get("selection_seed"),
            "bfcl_profile_case_count": logical_case_count,
            "bfcl_runtime_case_count": len(output_rows),
            "bfcl_generation_reused": not bool(bfcl_cfg["run_generation"]),
            "bfcl_reused_result_dir": (
                str(result_dir) if reused_generation is not None else ""
            ),
            "bfcl_web_search_backend": web_search_stats["backend"],
            "bfcl_web_search_queries": web_search_stats["queries"],
            "bfcl_web_search_cache_hits": web_search_stats["cache_hits"],
            "bfcl_web_search_network_requests": web_search_stats[
                "network_requests"
            ],
            "bfcl_web_search_retries": web_search_stats["retries"],
            "bfcl_web_search_errors": web_search_stats["errors"],
            "bfcl_web_fetches": web_search_stats["fetches"],
            "bfcl_web_fetch_truncations": web_search_stats[
                "fetch_truncations"
            ],
            "bfcl_web_fetch_errors": web_search_stats["fetch_errors"],
        }
    )
    print(
        "BFCL v4 evaluation completed: "
        f"{logical_case_count} logical scored cases, "
        f"{len(output_rows)} runtime rows, "
        f"{timeout_count} timeouts, "
        f"{inference_error_count} infrastructure errors, "
        f"{model_error_count} model response errors"
    )
    if inference_error_count:
        if not remaining_generation_failures:
            remaining_generation_failures = (
                _collect_retryable_generation_failures(
                    result_dir,
                    model_registry_name,
                )
            )
        failed_ids = ", ".join(
            row["id"] for row in remaining_generation_failures
        )
        details = f": {failed_ids}" if failed_ids else ""
        raise BFCLInfrastructureError(
            "BFCL v4 retained "
            f"{inference_error_count} non-timeout infrastructure errors "
            f"after bounded recovery{details}"
        )
