import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tools" / "repair_agentic_swe_budget_stops.py"


def load_module():
    spec = importlib.util.spec_from_file_location(SCRIPT.stem, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def patch_record(instance_id: str, *, stopped: bool, input_tokens: int):
    reason = "runtime_budget_exceeded" if stopped else ""
    return {
        "instance_id": instance_id,
        "patch": f"diff --git a/{instance_id} b/{instance_id}\n",
        "cache_key": {"model": "anthropic/claude-sonnet-4-6"},
        "openclaw_disqualified_reason": reason,
        "weave_agents_ok": True,
        "nemoclaw_session_audit_ok": True,
        "billable_openclaw_usage": {"inputTokens": input_tokens},
        "billable_openclaw_attempt_count": 1,
        "billable_openclaw_attempts": [
            {
                "attempt_id": f"{instance_id}-attempt",
                "wall_clock_seconds": 10,
                "usage": {"inputTokens": input_tokens},
            }
        ],
        "billable_openclaw_wall_seconds": 10,
    }


def test_prepare_extracts_only_budget_stops(tmp_path):
    module = load_module()
    dataset = tmp_path / "dataset.jsonl"
    patches = tmp_path / "patches.json"
    repair_dataset = tmp_path / "repair.jsonl"
    write_jsonl(dataset, [{"instance_id": "one"}, {"instance_id": "two"}])
    write_json(
        patches,
        [
            patch_record("one", stopped=False, input_tokens=100),
            patch_record("two", stopped=True, input_tokens=200),
        ],
    )

    module.prepare(
        SimpleNamespace(
            dataset_jsonl=dataset,
            source_patches_json=patches,
            output_dataset_jsonl=repair_dataset,
            output_instance_ids_json=None,
            state_json=None,
        )
    )

    assert module.read_jsonl(repair_dataset) == [{"instance_id": "two"}]
    assert module.read_json(
        repair_dataset.with_name("repair_instance_ids.json")
    ) == ["two"]
    state = module.read_json(repair_dataset.with_suffix(".state.json"))
    assert state["repair_instance_ids"] == ["two"]
    assert state["output_instance_ids_json"].endswith(
        "repair_instance_ids.json"
    )


def test_merge_replaces_only_stops_and_preserves_all_billable_usage(tmp_path):
    module = load_module()
    dataset = tmp_path / "dataset.jsonl"
    source_path = tmp_path / "source.json"
    repair_path = tmp_path / "repair.json"
    output_path = tmp_path / "merged.json"
    write_jsonl(dataset, [{"instance_id": "one"}, {"instance_id": "two"}])
    source = [
        patch_record("one", stopped=False, input_tokens=100),
        patch_record("two", stopped=True, input_tokens=200),
    ]
    repair = patch_record("two", stopped=False, input_tokens=300)
    repair["patch"] = "diff --git a/two b/two\n+repaired\n"
    write_json(source_path, source)
    write_json(repair_path, [repair])

    module.merge(
        SimpleNamespace(
            dataset_jsonl=dataset,
            source_patches_json=source_path,
            repair_patches_json=repair_path,
            output_patches_json=output_path,
            state_json=None,
        )
    )

    merged = module.read_json(output_path)
    assert [row["instance_id"] for row in merged] == ["one", "two"]
    assert merged[0] == source[0]
    assert merged[1]["patch"].endswith("+repaired\n")
    assert merged[1]["openclaw_disqualified_reason"] == ""
    assert merged[1]["billable_openclaw_usage"]["inputTokens"] == 500
    assert merged[1]["billable_openclaw_attempt_count"] == 2
    assert merged[1]["budget_repair"]["source_reason"] == "runtime_budget_exceeded"


def test_merge_rejects_unscoreable_repair(tmp_path):
    module = load_module()
    dataset = tmp_path / "dataset.jsonl"
    source_path = tmp_path / "source.json"
    repair_path = tmp_path / "repair.json"
    write_jsonl(dataset, [{"instance_id": "one"}])
    write_json(source_path, [patch_record("one", stopped=True, input_tokens=100)])
    repair = patch_record("one", stopped=False, input_tokens=200)
    repair["openclaw_disqualified_reason"] = "provider_transient_exhausted"
    write_json(repair_path, [repair])

    with pytest.raises(ValueError, match="Repair remains unscoreable"):
        module.merge(
            SimpleNamespace(
                dataset_jsonl=dataset,
                source_patches_json=source_path,
                repair_patches_json=repair_path,
                output_patches_json=tmp_path / "merged.json",
                state_json=None,
            )
        )


def test_merge_rejects_unknown_repair_failure(tmp_path):
    module = load_module()
    source = patch_record("one", stopped=True, input_tokens=100)
    repair = patch_record("one", stopped=False, input_tokens=200)
    repair["openclaw_disqualified_reason"] = "unexpected_new_failure"

    with pytest.raises(ValueError, match="Repair remains unscoreable"):
        module.validate_repair_record(repair, source)
