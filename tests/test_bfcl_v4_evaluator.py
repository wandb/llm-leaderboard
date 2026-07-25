import importlib.util
import json
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
BFCL_V4_PACKAGE_ROOT = (
    ROOT / "scripts" / "evaluator" / "evaluate_utils" / "bfcl_v4_pkg"
)
if str(BFCL_V4_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BFCL_V4_PACKAGE_ROOT))
MODULE_PATH = ROOT / "scripts" / "evaluator" / "bfcl_v4.py"
SPEC = importlib.util.spec_from_file_location("bfcl_v4", MODULE_PATH)
assert SPEC and SPEC.loader
bfcl_v4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bfcl_v4)


def _metadata(**overrides):
    value = {
        "benchmark_version": "v4",
        "locale": "zh-Hant-TW",
        "profile": "full",
        "categories": ["simple_python", "memory_kv"],
        "excluded_categories": ["parallel"],
        "max_multi_turn_turns": 3,
        "max_per_category": 30,
    }
    value.update(overrides)
    return value


def test_profile_metadata_accepts_expected_taiwan_contract(tmp_path):
    (tmp_path / "selection_metadata.json").write_text(
        json.dumps(_metadata()),
        encoding="utf-8",
    )

    result = bfcl_v4._load_profile_metadata(
        tmp_path,
        categories=["simple_python", "memory_kv"],
        expected_profile="full",
    )

    assert result["locale"] == "zh-Hant-TW"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"locale": "en"}, "zh-Hant-TW"),
        ({"profile": "core"}, "expected 'full'"),
        ({"max_multi_turn_turns": 4}, "3 turns"),
        ({"max_per_category": 31}, "30 cases"),
    ],
)
def test_profile_metadata_rejects_contract_mismatch(
    tmp_path, overrides, message
):
    (tmp_path / "selection_metadata.json").write_text(
        json.dumps(_metadata(**overrides)),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=message):
        bfcl_v4._load_profile_metadata(
            tmp_path,
            categories=["simple_python"],
            expected_profile="full",
        )


def test_core_profile_requires_exact_fixed_category_set():
    bfcl_v4._validate_evaluation_profile(
        "core",
        list(bfcl_v4.CORE_CATEGORIES),
    )

    with pytest.raises(ValueError, match="category mismatch"):
        bfcl_v4._validate_evaluation_profile(
            "core",
            list(bfcl_v4.CORE_CATEGORIES[:-1]),
        )


def test_full_profile_requires_core_and_agentic_categories():
    categories = list(
        bfcl_v4.CORE_CATEGORIES + bfcl_v4.AGENTIC_CATEGORIES
    )
    bfcl_v4._validate_evaluation_profile("full", categories)

    with pytest.raises(ValueError, match="memory_kv"):
        bfcl_v4._validate_evaluation_profile(
            "full",
            [item for item in categories if item != "memory_kv"],
        )


@pytest.mark.parametrize(
    "backend, expected",
    [
        ("ddgs", "ddgs"),
        ("direct-search", "ddgs"),
        ("duckduckgo", "duckduckgo_html"),
        ("ddg-html", "duckduckgo_html"),
        ("serp-api", "serpapi"),
    ],
)
def test_web_search_config_normalizes_backend_aliases(backend, expected):
    config = bfcl_v4._normalized_web_search_config(
        {"backend": backend}
    )
    assert config["backend"] == expected
    assert config["max_attempts"] == 4
    assert config["timeout_sec"] == 20.0
    assert config["fetch_max_bytes"] == 2_000_000
    assert config["fetch_max_chars"] == 20_000


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"backend": "unknown"}, "Unsupported"),
        ({"timeout_sec": 0}, "timeout_sec"),
        ({"max_attempts": 0}, "max_attempts"),
        ({"retry_base_sec": -1}, "retry_base_sec"),
        ({"fetch_max_bytes": 0}, "fetch_max_bytes"),
        ({"fetch_max_chars": 0}, "fetch_max_chars"),
        (
            {
                "backend": "duckduckgo_html",
                "endpoint": "not-a-url",
            },
            "endpoint",
        ),
    ],
)
def test_web_search_config_rejects_invalid_values(overrides, message):
    with pytest.raises(ValueError, match=message):
        bfcl_v4._normalized_web_search_config(overrides)


def test_web_search_config_sets_runtime_environment(tmp_path, monkeypatch):
    for name in (
        "BFCL_WEB_SEARCH_BACKEND",
        "BFCL_WEB_SEARCH_CACHE_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    config = bfcl_v4._normalized_web_search_config(
        {
            "backend": "duckduckgo",
            "cache_path": "cache/web.sqlite3",
        }
    )

    result = bfcl_v4._configure_web_search(config, repo_root=tmp_path)

    expected_cache = str((tmp_path / "cache/web.sqlite3").resolve())
    assert result["cache_path"] == expected_cache
    assert bfcl_v4.os.environ["BFCL_WEB_SEARCH_BACKEND"] == (
        "duckduckgo_html"
    )
    assert bfcl_v4.os.environ["BFCL_WEB_SEARCH_CACHE_PATH"] == (
        expected_cache
    )


def test_direct_web_backend_does_not_require_serpapi(monkeypatch):
    available = {"html2text", "ddgs"}
    monkeypatch.setattr(
        bfcl_v4.importlib.util,
        "find_spec",
        lambda module: object() if module in available else None,
    )
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)

    bfcl_v4._validate_agentic_dependencies(
        ["web_search_base"],
        web_search_config={"backend": "ddgs"},
    )


def test_serpapi_backend_requires_key(monkeypatch):
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="SERPAPI_API_KEY"):
        bfcl_v4._validate_agentic_dependencies(
            ["web_search_base"],
            web_search_config={"backend": "serpapi"},
        )


def test_runtime_uses_run_specific_project_root(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    project_root = tmp_path / "run"
    monkeypatch.delenv("BFCL_DATA_ROOT", raising=False)
    monkeypatch.delenv("BFCL_PROJECT_ROOT", raising=False)

    bfcl_v4._install_v4_runtime(data_root, project_root)

    assert bfcl_v4.os.environ["BFCL_DATA_ROOT"] == str(data_root)
    assert bfcl_v4.os.environ["BFCL_PROJECT_ROOT"] == str(project_root)


def test_testmode_uses_partial_evaluation_contract():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert 'partial_eval=bool(getattr(cfg, "testmode", False))' in source


def test_score_alignment_follows_parallel_result_completion_order():
    from bfcl_eval.eval_checker.eval_runner import _subset_entries_by_model_ids

    model_results = [{"id": "case-c"}, {"id": "case-a"}]
    prompts = [
        {"id": "case-a", "question": "A"},
        {"id": "case-b", "question": "B"},
        {"id": "case-c", "question": "C"},
    ]
    ground_truth = [
        {"id": "answer-a", "ground_truth": "A"},
        {"id": "answer-b", "ground_truth": "B"},
        {"id": "answer-c", "ground_truth": "C"},
    ]

    aligned_prompts, aligned_ground_truth = _subset_entries_by_model_ids(
        model_results,
        prompts,
        ground_truth,
        allow_missing=True,
    )

    assert [entry["id"] for entry in aligned_prompts] == ["case-c", "case-a"]
    assert [entry["id"] for entry in aligned_ground_truth] == [
        "answer-c",
        "answer-a",
    ]


def test_score_alignment_rejects_duplicate_result_ids():
    from bfcl_eval.eval_checker.eval_runner import _subset_entries_by_model_ids

    with pytest.raises(ValueError, match="duplicate"):
        _subset_entries_by_model_ids(
            [{"id": "case-a"}, {"id": "case-a"}],
            [{"id": "case-a"}],
            [{"ground_truth": "A"}],
            allow_missing=True,
        )


def test_wandb_table_values_flatten_multiturn_usage_and_errors():
    assert bfcl_v4._flatten_numeric_sum([[1, 2], [3, [4]]]) == 10
    assert bfcl_v4._stable_table_text(
        {"error_type": "multi_turn", "details": ["missing"]}
    ) == '{"details": ["missing"], "error_type": "multi_turn"}'


def test_reused_generation_results_require_complete_unique_coverage(tmp_path):
    result_dir = tmp_path / "result"
    model_dir = result_dir / "Configured_Model"
    model_dir.mkdir(parents=True)
    (model_dir / "BFCL_v4_simple_python_result.json").write_text(
        "\n".join(
            json.dumps({"id": entry_id, "result": "ok"})
            for entry_id in ("simple_1", "simple_2")
        )
        + "\n",
        encoding="utf-8",
    )
    (model_dir / "BFCL_v4_multiple_result.json").write_text(
        json.dumps({"id": "multiple_1", "result": "ok"}) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "category_counts": {
            "simple_python": 2,
            "multiple": 1,
        }
    }

    result = bfcl_v4._validate_reused_generation_results(
        result_dir=result_dir,
        model_registry_name="Configured/Model",
        categories=["simple_python", "multiple"],
        profile_metadata=metadata,
    )

    assert result["unique_case_count"] == 3
    assert result["expected_case_count"] == 3

    (model_dir / "BFCL_v4_multiple_result.json").unlink()
    with pytest.raises(FileNotFoundError, match="multiple"):
        bfcl_v4._validate_reused_generation_results(
            result_dir=result_dir,
            model_registry_name="Configured/Model",
            categories=["simple_python", "multiple"],
            profile_metadata=metadata,
        )


@pytest.mark.parametrize(
    "config_name",
    (
        "config-bfcl-v4-openai-fullrun.yaml",
        "config-bfcl-v4-anthropic-fullrun.yaml",
    ),
)
def test_paid_fullrun_configs_target_taiwan_wandb_project(config_name):
    config = OmegaConf.load(ROOT / "configs" / config_name)

    assert config.wandb.entity == "llm-leaderboard"
    assert config.wandb.project == "tc-leaderboard"
    assert config.bfcl.version == "v4"
    assert config.bfcl.profile == "full"
