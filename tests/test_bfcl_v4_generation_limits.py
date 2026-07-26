import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx


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
    assert not generation._retryable_failed_result(
        {
            "id": "simple_python_2",
            "result": "Error during inference: case timeout",
            "error": "bfcl_case_timeout",
            "timeout": True,
        }
    )
    assert not generation._retryable_failed_result(
        {
            "id": "simple_python_3",
            "result": "Error during inference: malformed tool arguments",
            "error": "model_inference_error",
            "infrastructure_error": False,
        }
    )


def test_provider_transport_errors_are_infrastructure_failures():
    assert generation._is_provider_infrastructure_exception(
        httpx.ReadTimeout("provider stopped responding")
    )
    assert not generation._is_provider_infrastructure_exception(
        ValueError("model emitted malformed arguments")
    )


def test_retry_ids_expand_to_all_transitive_dependents():
    entries = [
        {"id": "prereq_0", "depends_on": []},
        {"id": "prereq_1", "depends_on": ["prereq_0"]},
        {"id": "prereq_2", "depends_on": ["prereq_1"]},
        {"id": "target", "depends_on": ["prereq_2"]},
        {"id": "independent", "depends_on": []},
    ]

    expanded = generation._expand_retry_ids_with_dependents(
        entries,
        {"prereq_1"},
    )

    assert expanded == {"prereq_1", "prereq_2", "target"}


def test_memory_retry_restores_prior_checkpoint_and_clears_descendants(
    tmp_path, monkeypatch
):
    model_result_dir = tmp_path / "Configured_Model"
    snapshot_root = (
        model_result_dir / "agentic/memory/kv" / "memory_snapshot"
    )
    checkpoint_dir = snapshot_root / "prereq_checkpoints"
    checkpoint_dir.mkdir(parents=True)
    prior = checkpoint_dir / "memory_kv_prereq_0-customer-0.json"
    failed = checkpoint_dir / "memory_kv_prereq_1-customer-1.json"
    later = checkpoint_dir / "memory_kv_prereq_2-customer-2.json"
    prior.write_text('{"state": "valid"}', encoding="utf-8")
    failed.write_text('{"state": "stale-failed"}', encoding="utf-8")
    later.write_text('{"state": "stale-later"}', encoding="utf-8")
    (snapshot_root / "customer_final.json").write_text(
        '{"state": "contaminated"}',
        encoding="utf-8",
    )

    entries = [
        {
            "id": "memory_kv_prereq_0-customer-0",
            "scenario": "customer",
            "depends_on": [],
        },
        {
            "id": "memory_kv_prereq_1-customer-1",
            "scenario": "customer",
            "depends_on": ["memory_kv_prereq_0-customer-0"],
        },
        {
            "id": "memory_kv_prereq_2-customer-2",
            "scenario": "customer",
            "depends_on": [
                "memory_kv_prereq_0-customer-0",
                "memory_kv_prereq_1-customer-1",
            ],
        },
        {
            "id": "memory_kv_1-customer-1",
            "scenario": "customer",
            "depends_on": [
                "memory_kv_prereq_0-customer-0",
                "memory_kv_prereq_1-customer-1",
                "memory_kv_prereq_2-customer-2",
            ],
        },
    ]
    selected = entries[1:]
    monkeypatch.setattr(
        generation,
        "get_directory_structure_by_id",
        lambda _entry_id: "agentic/memory/kv",
    )
    monkeypatch.setattr(
        generation,
        "is_memory_prereq",
        lambda entry_id: "_prereq_" in entry_id,
    )

    generation._restore_memory_snapshots_for_resume(
        selected,
        model_result_dir,
    )

    assert (snapshot_root / "customer_final.json").read_text(
        encoding="utf-8"
    ) == '{"state": "valid"}'
    assert not failed.exists()
    assert not later.exists()


def test_case_infrastructure_recovery_happens_before_result_is_returned(
    monkeypatch
):
    attempts = []
    sleeps = []

    def fake_inference(*_args, **_kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            return {
                "id": "simple_python_0",
                "result": "Error during inference: provider timeout",
                "error": "inference_error",
                "infrastructure_error": True,
            }
        return {"id": "simple_python_0", "result": []}

    monkeypatch.setattr(
        generation,
        "multi_threaded_inference",
        fake_inference,
    )

    result = generation._inference_with_recovery(
        object(),
        {"id": "simple_python_0"},
        False,
        False,
        600,
        2,
        3,
        sleep=sleeps.append,
    )

    assert result["result"] == []
    assert result["infrastructure_recovery_attempts"] == 1
    assert len(attempts) == 2
    assert sleeps == [3]


def test_generation_persists_infrastructure_failure_and_defers_dependents(
    monkeypatch
):
    written = []
    attempted = []

    class FakeHandler:
        def write(self, result, **_kwargs):
            written.append(result)

    def fake_inference(
        _handler,
        test_case,
        *_args,
        **_kwargs,
    ):
        attempted.append(test_case["id"])
        if test_case["id"] == "simple_0":
            return {
                "id": "simple_0",
                "result": "Error during inference: provider timeout",
                "error": "inference_error",
                "infrastructure_error": True,
            }
        return {"id": test_case["id"], "result": []}

    monkeypatch.setattr(generation, "build_handler", lambda *_args: FakeHandler())
    monkeypatch.setattr(generation, "_inference_with_recovery", fake_inference)
    args = SimpleNamespace(
        temperature=0.01,
        num_threads=1,
        case_timeout_sec=600,
        watchdog_interval_sec=30,
        stall_fail_fast_sec=900,
        consecutive_failure_fail_fast=5,
        infrastructure_recovery_rounds=0,
        infrastructure_recovery_base_seconds=0,
        include_input_log=False,
        exclude_state_log=False,
        result_dir=Path("."),
        run_ids=False,
        resume_existing_results=True,
    )
    cases = [
        {"id": "simple_0", "function": [], "depends_on": []},
        {
            "id": "simple_1",
            "function": [],
            "depends_on": ["simple_0"],
        },
        {"id": "simple_2", "function": [], "depends_on": []},
    ]

    summary = generation.generate_results(args, "Configured/Model", cases)

    assert attempted == ["simple_0", "simple_2"]
    assert [row["id"] for row in written] == ["simple_0", "simple_2"]
    assert summary["generated_case_count"] == 2
    assert summary["infrastructure_failure_ids"] == ["simple_0"]
    assert summary["stopped_early"] is False


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
