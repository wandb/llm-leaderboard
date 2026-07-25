import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
for path in (
    ROOT / "scripts",
    ROOT / "scripts" / "evaluator" / "evaluate_utils" / "bfcl_v4_pkg",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

generation = importlib.import_module("bfcl_eval._llm_response_generation")


def test_limit_category_entries_keeps_only_requested_non_memory_cases():
    entries = [
        {"id": "simple_python_1"},
        {"id": "simple_python_2"},
        {"id": "simple_python_3"},
    ]

    limited = generation._limit_category_entries(entries, 2)

    assert [entry["id"] for entry in limited] == [
        "simple_python_1",
        "simple_python_2",
    ]


def test_limit_category_entries_keeps_memory_dependency_chain():
    entries = [
        {
            "id": "memory_kv_prereq_1-customer-0",
            "depends_on": [],
        },
        {
            "id": "memory_kv_prereq_1-customer-1",
            "depends_on": ["memory_kv_prereq_1-customer-0"],
        },
        {
            "id": "memory_kv_1",
            "depends_on": ["memory_kv_prereq_1-customer-1"],
        },
        {
            "id": "memory_kv_2",
            "depends_on": ["memory_kv_prereq_1-customer-1"],
        },
    ]

    limited = generation._limit_category_entries(entries, 1)

    assert [entry["id"] for entry in limited] == [
        "memory_kv_prereq_1-customer-0",
        "memory_kv_prereq_1-customer-1",
        "memory_kv_1",
    ]


def test_retryable_failure_includes_provider_and_web_backend_errors():
    assert generation._retryable_failed_result(
        {
            "id": "simple_python_0",
            "result": "Error during inference: provider timeout",
            "error": "inference_error",
        }
    )
    assert generation._retryable_failed_result(
        {
            "id": "web_search_base_0",
            "result": [],
            "inference_log": [
                {
                    "step_0": [
                        {
                            "role": "tool",
                            "content": json.dumps(
                                {
                                    "error": (
                                        "duckduckgo_html search failed after "
                                        "4 attempts: HTTP 403"
                                    )
                                }
                            ),
                        }
                    ]
                }
            ],
        }
    )
    assert not generation._retryable_failed_result(
        {"id": "simple_python_1", "result": []}
    )


def test_successful_inference_marks_web_backend_failure_as_infrastructure_error():
    class FakeHandler:
        def begin_case(self, _timeout):
            pass

        def end_case(self):
            pass

        def inference(self, *_args):
            return [], {
                "inference_log": [
                    {
                        "step_0": [
                            {
                                "role": "tool",
                                "content": (
                                    '{"error": "ddgs search failed after '
                                    '4 attempts: unavailable"}'
                                ),
                            }
                        ]
                    }
                ]
            }

    result = generation.multi_threaded_inference(
        FakeHandler(),
        {"id": "web_search_base_0", "function": []},
        False,
        False,
        30,
    )

    assert result["error"] == "web_search_backend_error"
    assert result["infrastructure_error"] is True


def test_collect_test_cases_resumes_successes_and_retries_failures(
    tmp_path, monkeypatch
):
    model_name = "fake/model"
    result_dir = tmp_path / "result"
    model_dir = result_dir / "fake_model"
    model_dir.mkdir(parents=True)
    result_file = model_dir / "simple_python_result.json"
    result_file.write_text(
        "\n".join(
            [
                json.dumps({"id": "simple_python_0", "result": []}),
                json.dumps(
                    {
                        "id": "simple_python_1",
                        "result": "Error during inference: interrupted",
                        "error": "inference_error",
                    }
                ),
                json.dumps(
                    {
                        "id": "simple_python_2",
                        "result": [],
                        "inference_log": [
                            {
                                "step_0": [
                                    {
                                        "role": "tool",
                                        "content": (
                                            '{"error": "ddgs search failed '
                                            'after 4 attempts: unavailable"}'
                                        ),
                                    }
                                ]
                            }
                        ],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        generation,
        "get_directory_structure_by_category",
        lambda _category: "",
    )
    monkeypatch.setattr(
        generation,
        "get_file_name_by_category",
        lambda category, is_result_file: f"{category}_result.json",
    )
    monkeypatch.setattr(generation, "is_memory", lambda _category: False)
    monkeypatch.setattr(
        generation,
        "clean_up_memory_prereq_entries",
        lambda entries: entries,
    )
    monkeypatch.setattr(
        generation,
        "populate_initial_settings_for_memory_test_cases",
        lambda entries, _model_dir: entries,
    )
    monkeypatch.setattr(
        generation,
        "populate_initial_settings_for_web_search_test_cases",
        lambda entries: entries,
    )
    monkeypatch.setattr(
        generation,
        "sort_key",
        lambda entry: entry["id"],
    )
    args = SimpleNamespace(
        result_dir=result_dir,
        allow_overwrite=True,
        run_ids=False,
        resume_existing_results=True,
        retry_failed_cases=True,
    )
    all_entries = [
        {"id": f"simple_python_{index}", "function": []}
        for index in range(4)
    ]

    selected = generation.collect_test_cases(
        args,
        model_name,
        ["simple_python"],
        all_entries,
    )

    assert [entry["id"] for entry in selected] == [
        "simple_python_1",
        "simple_python_2",
        "simple_python_3",
    ]
    assert result_file.exists()
