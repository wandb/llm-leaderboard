from __future__ import annotations

import gc
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


from evaluator.evaluate_utils import weave_compat


class FakeClient:
    _base_url = "https://api.openai.com/v1"
    _version = "1.90.0"


class FakeResponsesResource:
    def __init__(self) -> None:
        self._client = FakeClient()


def test_compatibility_patch_recognizes_responses_resources(monkeypatch):
    from weave.integrations.openai import openai_sdk

    def old_checker(obj):
        return (
            hasattr(obj, "messages")
            and hasattr(obj, "_client")
            and hasattr(obj._client, "_base_url")
            and hasattr(obj._client, "_version")
        )

    monkeypatch.setattr(openai_sdk, "completion_instance_check", old_checker)

    report = weave_compat.configure_openai_responses_input_sanitization()
    resource = FakeResponsesResource()

    assert report == {
        "available": True,
        "applied": True,
        "status": "compatibility_patch_applied",
    }
    assert openai_sdk.completion_instance_check(resource)
    assert openai_sdk.convert_completion_to_dict(resource) == {
        "client": {
            "base_url": "https://api.openai.com/v1",
            "version": "1.90.0",
        }
    }


def test_openai_input_handler_does_not_retain_responses_client(monkeypatch):
    from weave.integrations.openai import openai_sdk

    monkeypatch.setattr(
        openai_sdk,
        "completion_instance_check",
        lambda obj: hasattr(obj, "messages"),
    )
    weave_compat.configure_openai_responses_input_sanitization()

    resource = FakeResponsesResource()
    monkeypatch.setattr(
        openai_sdk,
        "_default_on_input_handler",
        lambda _func, _args, _kwargs: SimpleNamespace(
            inputs={"self": resource, "input": "test"}
        ),
    )

    processed = openai_sdk.openai_on_input_handler(
        SimpleNamespace(),
        (resource,),
        {"input": "test"},
    )

    assert processed is not None
    assert processed.inputs["self"] == {
        "client": {
            "base_url": "https://api.openai.com/v1",
            "version": "1.90.0",
        }
    }
    assert processed.inputs["self"] is not resource


def test_sanitized_trace_inputs_do_not_accumulate_responses_clients(monkeypatch):
    from weave.integrations.openai import openai_sdk

    monkeypatch.setattr(
        openai_sdk,
        "completion_instance_check",
        lambda obj: hasattr(obj, "messages"),
    )
    weave_compat.configure_openai_responses_input_sanitization()
    monkeypatch.setattr(
        openai_sdk,
        "_default_on_input_handler",
        lambda _func, args, _kwargs: SimpleNamespace(
            inputs={"self": args[0], "input": "test"}
        ),
    )

    trace_inputs = []
    client_refs = []
    for _ in range(2000):
        resource = FakeResponsesResource()
        client_refs.append(weakref.ref(resource._client))
        processed = openai_sdk.openai_on_input_handler(
            SimpleNamespace(),
            (resource,),
            {"input": "test"},
        )
        assert processed is not None
        trace_inputs.append(processed.inputs)

    del processed
    del resource
    gc.collect()

    assert all(ref() is None for ref in client_refs)
    assert all(
        entry["self"]["client"]["base_url"] == "https://api.openai.com/v1"
        for entry in trace_inputs
    )


def test_compatibility_patch_is_idempotent(monkeypatch):
    from weave.integrations.openai import openai_sdk

    monkeypatch.setattr(
        openai_sdk,
        "completion_instance_check",
        lambda obj: hasattr(obj, "messages"),
    )

    first = weave_compat.configure_openai_responses_input_sanitization()
    patched_checker = openai_sdk.completion_instance_check
    second = weave_compat.configure_openai_responses_input_sanitization()

    assert first["status"] == "compatibility_patch_applied"
    assert second["status"] == "compatibility_patch_already_applied"
    assert openai_sdk.completion_instance_check is patched_checker


def test_upstream_compatible_checker_is_not_replaced(monkeypatch):
    from weave.integrations.openai import openai_sdk

    def upstream_checker(obj):
        return hasattr(obj, "_client")

    monkeypatch.setattr(openai_sdk, "completion_instance_check", upstream_checker)

    report = weave_compat.configure_openai_responses_input_sanitization()

    assert report == {
        "available": True,
        "applied": False,
        "status": "upstream_behavior_compatible",
    }
    assert openai_sdk.completion_instance_check is upstream_checker
