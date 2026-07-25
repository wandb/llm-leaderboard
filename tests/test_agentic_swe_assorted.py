import importlib.util
import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tools" / "run_agentic_swe_assorted.py"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_agentic_swe_assorted_terminates_child_process_group(monkeypatch):
    module = load_module(SCRIPT)
    signals = []

    class FakeProcess:
        pid = 1234

        def __init__(self):
            self.wait_calls = 0

        def poll(self):
            return None

        def wait(self, timeout=None):
            self.wait_calls += 1
            return -signal.SIGKILL

    monkeypatch.setattr(module, "_descendant_pids", lambda pid: [2345])
    monkeypatch.setattr(module, "_process_groups", lambda pids: set(pids))
    monkeypatch.setattr(module, "_pid_exists", lambda pid: True)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    monotonic_values = iter([0.0, 0.0, 1.0])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(module.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    proc = FakeProcess()

    module.terminate_process_group(proc, grace_seconds=0.5)

    assert signals == [
        (1234, signal.SIGTERM),
        (2345, signal.SIGTERM),
        (1234, signal.SIGKILL),
        (2345, signal.SIGKILL),
    ]
    assert proc.wait_calls == 1


def test_agentic_swe_assorted_sigterm_handler_interrupts_child(monkeypatch):
    module = load_module(SCRIPT)
    terminated = []

    class FakeStdout:
        def __iter__(self):
            handler = signal.getsignal(signal.SIGTERM)
            handler(signal.SIGTERM, None)
            yield "unreachable"

    class FakeProcess:
        pid = 1234
        stdout = FakeStdout()

        def wait(self):
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(module, "terminate_process_group", lambda proc: terminated.append(proc.pid))

    with pytest.raises(KeyboardInterrupt, match="received SIGTERM"):
        module.run_command(["child"])

    assert terminated == [1234]


def test_agentic_swe_assorted_passes_sandbox_lease_to_child(monkeypatch):
    module = load_module(SCRIPT)
    captured = {}

    class FakeProcess:
        pid = 1234
        stdout = iter(())

        def wait(self):
            return 0

    def fake_popen(*args, **kwargs):
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)

    result = module.run_command(
        ["child", "--nemoclaw-sandbox", "nejumi-taiwan"]
    )

    assert result.returncode == 0
    assert captured["env"][module.NEMOCLAW_SANDBOX_LEASE_ENV] == "nejumi-taiwan"


def test_agentic_swe_assorted_keeps_tier_and_source_fields():
    module = load_module(SCRIPT)
    rows = module.lite_rows(
        source_rows=[
            {
                "instance_id": "django__django-1",
                "repo": "django/django",
                "agentic_swe_tier": "low",
                "source_benchmark": "SWE-bench Lite",
                "source_dataset": "princeton-nlp/SWE-bench_Lite",
                "source_subset": "low_36",
                "source_instance_id": "django__django-1",
            }
        ],
        patch_rows=[
            {
                "instance_id": "django__django-1",
                "patch": "diff --git a/a.py b/a.py\n",
                "openclaw_tool_call_count": 3,
                "openclaw_usage": {"inputTokens": 100, "outputTokens": 20},
            }
        ],
        eval_results={"django__django-1": True},
    )

    module.validate_output_rows(rows)
    [row] = rows
    assert row["agentic_swe_tier"] == "low"
    assert row["source_benchmark"] == "SWE-bench Lite"
    assert row["source_dataset"] == "princeton-nlp/SWE-bench_Lite"
    assert row["source_subset"] == "low_36"
    assert row["source_instance_id"] == "django__django-1"
    assert row["resolved"] is True
    assert row["score"] == 1.0
    assert row["official_score"] == 1.0


def test_lite_rows_backfills_flat_session_audit_evidence():
    module = load_module(SCRIPT)
    rows = module.lite_rows(
        source_rows=[
            {
                "instance_id": "django__django-1",
                "repo": "django/django",
                "agentic_swe_tier": "low",
            }
        ],
        patch_rows=[
            {
                "instance_id": "django__django-1",
                "patch": "",
                "nemoclaw_session_audit_ok": True,
                "nemoclaw_session_audit": {
                    "required": True,
                    "ok": True,
                    "copied_session_bytes": 2048,
                    "copy": {
                        "source": "stdout_agent_meta",
                        "bytes": 2048,
                    },
                },
            }
        ],
        eval_results={"django__django-1": False},
    )

    [row] = rows
    assert row["nemoclaw_session_audit_required"] is True
    assert row["nemoclaw_session_copy_source"] == "stdout_agent_meta"
    assert row["nemoclaw_session_copied_bytes"] == 2048


def test_lite_rows_resolved_result_overrides_missing_report_score():
    module = load_module(SCRIPT)
    rows = module.lite_rows(
        source_rows=[
            {
                "instance_id": "django__django-1",
                "repo": "django/django",
                "agentic_swe_tier": "low",
            }
        ],
        patch_rows=[
            {
                "instance_id": "django__django-1",
                "patch": "diff --git a/a.py b/a.py\n",
            }
        ],
        eval_results={"django__django-1": True},
        partial_eval_results={
            "django__django-1": {
                "score": 0.0,
                "official_score": 0.0,
                "official_partial_credit": 0.0,
                "official_score_version": (
                    "resolved-1-else-f2p-squared-p2p-cap0.3-v2"
                ),
                "binary_score": 0.0,
                "diagnostic_partial_credit": 0.0,
                "diagnostic_score_with_partial": 0.0,
            }
        },
    )

    [row] = rows
    assert row["resolved"] is True
    assert row["score"] == 1.0
    assert row["official_score"] == 1.0
    assert row["official_partial_credit"] == 0.0
    assert row["binary_score"] == 1.0
    assert row["diagnostic_score_with_partial"] == 1.0


def test_agentic_swe_assorted_defaults_to_frozen_20_20_10_subset():
    module = load_module(SCRIPT)

    assert module.DEFAULT_LOW_MIDDLE_JSONL.name == "low_middle_v3_40.jsonl"
    assert module.DEFAULT_LOW_MIDDLE_IDS.name == "low_middle_v3_40_instance_ids.json"
    assert (
        module.DEFAULT_DEEPSWE_META.name
        == "essential_anchored_high_10_model_fidelity_cost_balanced.jsonl"
    )
    assert (
        module.DEFAULT_DEEPSWE_TASK_NAMES.name
        == "essential_anchored_high_10_model_fidelity_cost_balanced_task_names.json"
    )
    assert len(module.FROZEN_DEEPSWE_HIGH_TASK_NAMES) == 10


def _reusable_high_result(task: str, *, model: str = "openrouter/z-ai/glm-5.2"):
    return {
        "task_name": f"datacurve/{task}",
        "score": 0,
        "resolved": False,
        "exception": None,
        "openclaw_disqualified_reason": "",
        "weave_agents_ok": True,
        "nemoclaw_session_audit_ok": True,
        "agent_result": {
            "metadata": {
                "openclaw": {
                    "cache_key": {"model": model},
                }
            }
        },
    }


def test_load_reused_high_results_accepts_complete_scoreable_split_files(tmp_path):
    module = load_module(SCRIPT)
    metadata = [{"task_name": "one"}, {"task_name": "two"}]
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(json.dumps(_reusable_high_result("one")) + "\n", encoding="utf-8")
    second.write_text(json.dumps(_reusable_high_result("two")) + "\n", encoding="utf-8")

    rows = module.load_reused_high_results(
        [first, second],
        metadata_rows=metadata,
        model="openrouter/z-ai/glm-5.2",
    )

    assert [row["task_name"] for row in rows] == ["datacurve/one", "datacurve/two"]


@pytest.mark.parametrize(
    "reason",
    ["runtime_budget_exceeded", "model_output_truncated", "time_up"],
)
def test_load_reused_high_results_accepts_scoreable_model_stop(tmp_path, reason):
    module = load_module(SCRIPT)
    result = _reusable_high_result("one")
    result["score"] = 0.0
    result["openclaw_disqualified_reason"] = reason
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(result) + "\n", encoding="utf-8")

    rows = module.load_reused_high_results(
        [path],
        metadata_rows=[{"task_name": "one"}],
        model="openrouter/z-ai/glm-5.2",
    )

    assert rows == [result]


def test_load_reused_high_results_rejects_infrastructure_failure(tmp_path):
    module = load_module(SCRIPT)
    result = _reusable_high_result("one")
    result["openclaw_disqualified_reason"] = "provider_transient_exhausted"
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(result) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unscoreable failure for one"):
        module.load_reused_high_results(
            [path],
            metadata_rows=[{"task_name": "one"}],
            model="openrouter/z-ai/glm-5.2",
        )


def test_load_reused_high_results_rejects_missing_or_wrong_model(tmp_path):
    module = load_module(SCRIPT)
    path = tmp_path / "results.jsonl"
    path.write_text(
        json.dumps(_reusable_high_result("one", model="different/model")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        module.load_reused_high_results(
            [path],
            metadata_rows=[{"task_name": "one"}, {"task_name": "two"}],
            model="openrouter/z-ai/glm-5.2",
        )

    message = str(exc_info.value)
    assert "model mismatch for one" in message
    assert "missing selected High tasks: two" in message


def test_load_reused_high_results_rejects_unscoreable_trace(tmp_path):
    module = load_module(SCRIPT)
    result = _reusable_high_result("one")
    result["score"] = None
    result["weave_agents_ok"] = False
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(result) + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        module.load_reused_high_results(
            [path],
            metadata_rows=[{"task_name": "one"}],
            model="openrouter/z-ai/glm-5.2",
        )

    message = str(exc_info.value)
    assert "non-numeric score for one" in message
    assert "native trace verification failed for one" in message


def test_load_checkpointable_high_results_reuses_valid_rows_and_leaves_failures_pending(
    tmp_path,
):
    module = load_module(SCRIPT)
    checkpoint = tmp_path / "checkpoint.jsonl"
    current = tmp_path / "current.jsonl"
    checkpoint.write_text(
        json.dumps(_reusable_high_result("one")) + "\n",
        encoding="utf-8",
    )
    failed = _reusable_high_result("two")
    failed["exception"] = {"exception_type": "RequiredWeaveAgentsTraceError"}
    failed["weave_agents_ok"] = None
    current.write_text(
        json.dumps(failed)
        + "\n"
        + json.dumps(_reusable_high_result("three"))
        + "\n",
        encoding="utf-8",
    )

    rows, report = module.load_checkpointable_high_results(
        [checkpoint, current],
        metadata_rows=[
            {"task_name": "one"},
            {"task_name": "two"},
            {"task_name": "three"},
            {"task_name": "four"},
        ],
        model="openrouter/z-ai/glm-5.2",
    )

    assert [row["task_name"] for row in rows] == [
        "datacurve/one",
        "datacurve/three",
    ]
    assert report["reused_task_names"] == ["one", "three"]
    assert report["pending_task_names"] == ["two", "four"]
    assert "exception recorded for two" in report["rejected"]["two"]
    assert "native trace verification failed for two" in report["rejected"]["two"]


def test_load_checkpointable_high_results_keeps_valid_checkpoint_over_new_invalid_row(
    tmp_path,
):
    module = load_module(SCRIPT)
    checkpoint = tmp_path / "checkpoint.jsonl"
    current = tmp_path / "current.jsonl"
    valid = _reusable_high_result("one")
    invalid = _reusable_high_result("one")
    invalid["exception"] = {"exception_type": "Interrupted"}
    checkpoint.write_text(json.dumps(valid) + "\n", encoding="utf-8")
    current.write_text(json.dumps(invalid) + "\n", encoding="utf-8")

    rows, report = module.load_checkpointable_high_results(
        [checkpoint, current],
        metadata_rows=[{"task_name": "one"}],
        model="openrouter/z-ai/glm-5.2",
    )

    assert rows == [valid]
    assert report["pending_task_names"] == []
    assert report["rejected"] == {}


def _reusable_lite_patch(
    instance_id: str,
    *,
    model: str = "anthropic/claude-sonnet-4-6",
    reason: str = "",
):
    return {
        "instance_id": instance_id,
        "patch": "diff --git a/a.py b/a.py\n",
        "cache_key": {
            "model": model,
            "nemoclaw_sandbox": "nejumi-taiwan",
        },
        "openclaw_disqualified_reason": reason,
        "weave_agents_ok": True,
        "nemoclaw_session_audit_ok": True,
        "nemoclaw_checkout_transfer": {
            "mode": "copy",
            "host_status": "clean",
            "host_head_tree": "tree-sha",
            "sandbox_baseline_tree": "tree-sha",
        },
    }


def test_load_reused_low_middle_patches_accepts_scoreable_budget_stop(tmp_path):
    module = load_module(SCRIPT)
    path = tmp_path / "patches.json"
    path.write_text(
        json.dumps(
            [
                _reusable_lite_patch("one"),
                _reusable_lite_patch("two", reason="runtime_budget_exceeded"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = module.load_reused_low_middle_patches(
        path,
        source_rows=[{"instance_id": "one"}, {"instance_id": "two"}],
        model="anthropic/claude-sonnet-4-6",
    )

    assert [row["instance_id"] for row in rows] == ["one", "two"]


def test_load_reused_low_middle_patches_rejects_unscoreable_or_incomplete_set(tmp_path):
    module = load_module(SCRIPT)
    path = tmp_path / "patches.json"
    path.write_text(
        json.dumps(
            [
                _reusable_lite_patch(
                    "one",
                    reason="provider_transient_exhausted",
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        module.load_reused_low_middle_patches(
            path,
            source_rows=[{"instance_id": "one"}, {"instance_id": "two"}],
            model="anthropic/claude-sonnet-4-6",
        )

    message = str(exc_info.value)
    assert "unscoreable failure for one: provider_transient_exhausted" in message
    assert "missing selected Low/Middle tasks: two" in message


def test_load_reused_low_middle_patches_rejects_legacy_unisolated_baseline(tmp_path):
    module = load_module(SCRIPT)
    patch = _reusable_lite_patch("one")
    patch["nemoclaw_checkout_transfer"] = {
        "mode": "visible",
        "sandbox_checkout_dir": "/sandbox/checkouts/one",
    }
    path = tmp_path / "patches.json"
    path.write_text(json.dumps([patch]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        module.load_reused_low_middle_patches(
            path,
            source_rows=[{"instance_id": "one"}],
            model="anthropic/claude-sonnet-4-6",
        )

    message = str(exc_info.value)
    assert "non-isolated sandbox baseline for one: visible" in message
    assert "sandbox baseline tree mismatch for one" in message


def test_agentic_swe_assorted_high_defaults_are_budgeted_but_above_low_middle(monkeypatch):
    module = load_module(SCRIPT)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_agentic_swe_assorted.py",
            "--model",
            "openrouter-direct/z-ai/glm-5.2",
        ],
    )

    args = module.parse_args()

    assert args.max_agent_turns == 40
    assert args.max_tool_calls == 40
    assert args.max_cumulative_input_tokens == 1_000_000
    assert args.high_max_agent_turns == 150
    assert args.high_max_tool_calls == 200
    assert args.high_max_cumulative_input_tokens == 14_000_000
    assert args.deepswe_budget_preflight == "error"
    assert args.deepswe_preflight_hard_stat == "p90"
    assert args.min_free_disk_gb == 30.0


def test_prepare_high_inputs_honors_requested_task_order_and_limit(tmp_path):
    module = load_module(SCRIPT)
    metadata_path = tmp_path / "metadata.jsonl"
    metadata_path.write_text(
        "".join(
            json.dumps({"task_name": task_name}) + "\n"
            for task_name in ("first", "second", "third")
        ),
        encoding="utf-8",
    )
    task_names_path = tmp_path / "task_names.json"
    task_names_path.write_text(json.dumps(["third", "first", "third"]), encoding="utf-8")
    args = SimpleNamespace(
        output_dir=tmp_path / "out",
        deepswe_metadata_jsonl=metadata_path,
        deepswe_task_names_file=task_names_path,
        high_limit=1,
    )

    selected_path, rows = module.prepare_high_inputs(args)

    assert rows == [{"task_name": "third"}]
    assert json.loads(selected_path.read_text(encoding="utf-8")) == ["third"]


def test_prepare_high_inputs_rejects_task_missing_from_metadata(tmp_path):
    module = load_module(SCRIPT)
    metadata_path = tmp_path / "metadata.jsonl"
    metadata_path.write_text(json.dumps({"task_name": "present"}) + "\n", encoding="utf-8")
    task_names_path = tmp_path / "task_names.json"
    task_names_path.write_text(json.dumps(["missing"]), encoding="utf-8")
    args = SimpleNamespace(
        output_dir=tmp_path / "out",
        deepswe_metadata_jsonl=metadata_path,
        deepswe_task_names_file=task_names_path,
        high_limit=None,
    )

    with pytest.raises(ValueError, match="missing from metadata: missing"):
        module.prepare_high_inputs(args)


def test_agentic_swe_assorted_dedupes_default_policy_flags(monkeypatch):
    module = load_module(SCRIPT)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_agentic_swe_assorted.py",
            "--model",
            "openrouter-direct/z-ai/glm-5.2",
            "--deny-tool",
            "web_search",
            "--deny-argument-pattern",
            r"https?://",
        ],
    )

    args = module.parse_args()

    assert args.deny_tool.count("web_search") == 1
    assert args.deny_argument_pattern.count(r"https?://") == 1


def test_agentic_swe_assorted_runtime_preflight_blocks_low_disk(tmp_path, monkeypatch):
    module = load_module(SCRIPT)

    monkeypatch.setattr(module, "_disk_free_gb", lambda path: 5.0)
    monkeypatch.setattr(module, "_docker_root_dir", lambda: None)
    args = SimpleNamespace(
        output_dir=tmp_path / "out",
        skip_low_middle=False,
        skip_high=False,
        no_docker_check=False,
        dry_run=False,
        min_free_disk_gb=30.0,
        deny_tool=[],
        deny_argument_pattern=[],
    )

    with pytest.raises(RuntimeError, match="runtime preflight failed"):
        module.runtime_preflight(
            args,
            low_middle_rows=[{"agentic_swe_tier": "low"}],
            high_rows=[{"task_name": "high"}],
        )

    report = json.loads((args.output_dir / "inputs" / "runtime_preflight.json").read_text())
    assert report["ok"] is False
    assert report["selected_counts"]["low"] == 1
    assert report["selected_counts"]["high"] == 1


def test_agentic_swe_assorted_runtime_preflight_skips_disk_when_disabled(tmp_path, monkeypatch):
    module = load_module(SCRIPT)

    monkeypatch.setattr(module, "_disk_free_gb", lambda path: 5.0)
    args = SimpleNamespace(
        output_dir=tmp_path / "out",
        skip_low_middle=False,
        skip_high=False,
        no_docker_check=False,
        dry_run=False,
        min_free_disk_gb=0.0,
        deny_tool=[],
        deny_argument_pattern=[],
    )

    report = module.runtime_preflight(
        args,
        low_middle_rows=[{"agentic_swe_tier": "middle"}],
        high_rows=[],
    )

    assert report["ok"] is True
    assert report["selected_counts"]["middle"] == 1


def test_deepswe_budget_preflight_blocks_known_over_budget_task(tmp_path):
    module = load_module(SCRIPT)
    trials_path = tmp_path / "trials.json"
    trials_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "source": "deep-swe",
                        "eval_scope": "full",
                        "included_in_score": True,
                        "task_name": "too-long",
                        "model": "glm-5-2",
                        "reasoning_effort": "max",
                        "n_agent_steps": 90,
                        "n_input_tokens": 4_000_000,
                        "cost_usd": 1.2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        output_dir=tmp_path / "out",
        deepswe_public_trials_json=trials_path,
        deepswe_public_model=None,
        deepswe_public_effort=None,
        deepswe_budget_preflight="error",
        deepswe_preflight_hard_stat="mean",
        allow_deepswe_budget_mismatch=False,
        dry_run=False,
        model="openrouter-direct/z-ai/glm-5.2",
        thinking="max",
        high_max_agent_turns=60,
        max_agent_turns=40,
        high_max_cumulative_input_tokens=3_000_000,
        max_cumulative_input_tokens=1_000_000,
    )

    with pytest.raises(RuntimeError, match="budget preflight failed"):
        module.deepswe_budget_preflight(args, [{"task_name": "too-long"}])

    report = json.loads((args.output_dir / "inputs" / "deepswe_budget_preflight.json").read_text())
    assert report["ok"] is False
    assert report["basis"] == "model_effort"
    assert report["public_model"] == "glm-5-2"
    assert report["public_effort"] == "max"
    assert any("steps" in reason for reason in report["blocking_reasons"])
    assert any("input tokens" in reason for reason in report["blocking_reasons"])


def test_deepswe_budget_preflight_dry_run_reports_without_raising(tmp_path):
    module = load_module(SCRIPT)
    trials_path = tmp_path / "trials.json"
    trials_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "source": "deep-swe",
                        "eval_scope": "full",
                        "included_in_score": True,
                        "task_name": "too-long",
                        "model": "glm-5-2",
                        "reasoning_effort": "max",
                        "n_agent_steps": 90,
                        "n_input_tokens": 4_000_000,
                        "cost_usd": 1.2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        output_dir=tmp_path / "out",
        deepswe_public_trials_json=trials_path,
        deepswe_public_model=None,
        deepswe_public_effort=None,
        deepswe_budget_preflight="error",
        deepswe_preflight_hard_stat="mean",
        allow_deepswe_budget_mismatch=False,
        dry_run=True,
        model="openrouter-direct/z-ai/glm-5.2",
        thinking="max",
        high_max_agent_turns=60,
        max_agent_turns=40,
        high_max_cumulative_input_tokens=3_000_000,
        max_cumulative_input_tokens=1_000_000,
    )

    report = module.deepswe_budget_preflight(args, [{"task_name": "too-long"}])

    assert report["ok"] is False
    assert (args.output_dir / "inputs" / "deepswe_budget_preflight.json").exists()


def test_default_deepswe_high_subset_passes_glm52max_budget_preflight(tmp_path):
    module = load_module(SCRIPT)
    rows = module.read_jsonl(module.DEFAULT_DEEPSWE_META)
    args = SimpleNamespace(
        output_dir=tmp_path / "out",
        deepswe_public_trials_json=module.DEFAULT_DEEPSWE_PUBLIC_TRIALS,
        deepswe_public_model=None,
        deepswe_public_effort=None,
        deepswe_budget_preflight="error",
        deepswe_preflight_hard_stat="p90",
        allow_deepswe_budget_mismatch=False,
        dry_run=False,
        model="openrouter-direct/z-ai/glm-5.2",
        thinking="max",
        high_max_agent_turns=140,
        max_agent_turns=40,
        high_max_cumulative_input_tokens=13_000_000,
        max_cumulative_input_tokens=1_000_000,
    )

    report = module.deepswe_budget_preflight(args, rows)

    assert report["ok"] is True
    assert report["basis"] == "model_effort"
    assert report["hard_stat"] == "p90"
    assert not report["blocking_reasons"]


def test_deepswe_command_can_use_high_specific_cumulative_input_cap(tmp_path):
    module = load_module(SCRIPT)
    params_json = json.dumps({"provider": {"only": ["z-ai/fp8"], "allow_fallbacks": False}})
    overrides_json = json.dumps({"maxTokens": 4096})
    args = SimpleNamespace(
        deepswe_tasks_root=tmp_path / "tasks",
        output_dir=tmp_path / "out",
        prefix="assorted",
        model="openrouter-direct/z-ai/glm-5.2",
        openclaw_model_params_json=params_json,
        openclaw_model_overrides_json=overrides_json,
        high_openclaw_model_params_json=json.dumps({"maxTokens": 65536}),
        high_openclaw_model_overrides_json=json.dumps({"maxTokens": 65536}),
        thinking="high",
        agent="main",
        high_workers=2,
        high_openclaw_timeout=1800,
        openclaw_max_attempts=1,
        openclaw_retry_base_seconds=15.0,
        max_input_tokens=1_000_000,
        max_cumulative_input_tokens=1_000_000,
        high_max_cumulative_input_tokens=3_000_000,
        max_cumulative_output_tokens=500_000,
        max_tool_calls=40,
        high_max_tool_calls=60,
        max_agent_turns=40,
        high_max_agent_turns=None,
        max_tool_wall_seconds=120,
        llm_response_idle_timeout_seconds=900.0,
        nemoclaw_bin="nemoclaw",
        nemoclaw_sandbox="nejumi-taiwan",
        nemoclaw_workdir="/sandbox",
        nemoclaw_openclaw_config_path="/sandbox/.openclaw/openclaw.json",
        nemoclaw_checkout_transfer_mode="copy",
        nemoclaw_checkout_transfer_timeout=600,
        openclaw_tool_profile="coding",
        task_agent_prefix="tw-swe-assorted",
        session_prefix="agentic-swe-assorted",
        weave_agents_entity="llm-leaderboard",
        weave_agents_project="tc-leaderboard",
        weave_agents_agent_name="nejumi-taiwan-openclaw",
        weave_agents_limit=100,
        weave_agents_verification_timeout=120.0,
        weave_agents_poll_seconds=5.0,
        require_actual_token_usage=True,
        verify_weave_agents=True,
        use_task_agent=True,
        dry_run=False,
        deny_tool=[],
        deny_argument_pattern=[],
    )

    task_names_path = tmp_path / "high_tasks.json"
    task_names_path.write_text(json.dumps(["task-a"]) + "\n", encoding="utf-8")
    command = module.build_deepswe_command(args, task_names_path)

    idx = command.index("--max-cumulative-input-tokens")
    assert command[idx + 1] == "3000000"
    idx = command.index("--openclaw-model-params-json")
    assert json.loads(command[idx + 1]) == {
        "provider": {"only": ["z-ai/fp8"], "allow_fallbacks": False},
        "maxTokens": 65536,
    }
    idx = command.index("--openclaw-model-overrides-json")
    assert json.loads(command[idx + 1]) == {"maxTokens": 65536}
    idx = command.index("--job-name")
    assert command[idx + 1].startswith("assorted-deepswe-")
    assert len(command[idx + 1].rsplit("-", 1)[-1]) == 12


def test_deepswe_job_name_is_stable_and_changes_with_pending_tasks(tmp_path):
    module = load_module(SCRIPT)
    args = SimpleNamespace(
        output_dir=tmp_path / "out",
        prefix="assorted",
        model="openai-direct/test",
        openclaw_model_params_json="{}",
        high_openclaw_model_params_json=None,
        openclaw_model_overrides_json="{}",
        high_openclaw_model_overrides_json=None,
        thinking="high",
        agent="main",
        high_workers=2,
        high_openclaw_timeout=1800,
        openclaw_max_attempts=1,
        openclaw_retry_base_seconds=15.0,
        provider_recovery_rounds=2,
        provider_recovery_base_seconds=60.0,
        native_trace_recovery_attempts=1,
        native_trace_recovery_base_seconds=15.0,
        max_input_tokens=1_000_000,
        max_cumulative_input_tokens=1_000_000,
        high_max_cumulative_input_tokens=3_000_000,
        max_cumulative_output_tokens=500_000,
        max_tool_calls=40,
        high_max_tool_calls=60,
        max_agent_turns=40,
        high_max_agent_turns=80,
        max_tool_wall_seconds=120,
        llm_response_idle_timeout_seconds=900.0,
        nemoclaw_bin="nemoclaw",
        nemoclaw_sandbox="nejumi-taiwan",
        nemoclaw_workdir="/sandbox",
        nemoclaw_openclaw_config_path="/sandbox/.openclaw/openclaw.json",
        nemoclaw_checkout_transfer_mode="copy",
        nemoclaw_checkout_transfer_timeout=600,
        openclaw_tool_profile="coding",
        task_agent_prefix="tw-swe-assorted",
        session_prefix="agentic-swe-assorted",
        weave_agents_entity="llm-leaderboard",
        weave_agents_project="tc-leaderboard",
        weave_agents_agent_name="nejumi-taiwan-openclaw",
        weave_agents_limit=100,
        weave_agents_verification_timeout=120.0,
        weave_agents_poll_seconds=5.0,
        require_actual_token_usage=True,
        verify_weave_agents=True,
        use_task_agent=True,
        dry_run=False,
        deny_tool=[],
        deny_argument_pattern=[],
        deepswe_tasks_root=tmp_path / "tasks",
    )
    task_names_path = tmp_path / "pending.json"
    task_names_path.write_text(json.dumps(["task-a"]) + "\n", encoding="utf-8")

    first = module.build_deepswe_command(args, task_names_path)
    second = module.build_deepswe_command(args, task_names_path)
    first_name = first[first.index("--job-name") + 1]
    second_name = second[second.index("--job-name") + 1]
    assert first_name == second_name

    task_names_path.write_text(
        json.dumps(["task-a", "task-b"]) + "\n",
        encoding="utf-8",
    )
    changed = module.build_deepswe_command(args, task_names_path)
    changed_name = changed[changed.index("--job-name") + 1]
    assert changed_name != first_name


def test_swe_command_forwards_openclaw_model_params_json(tmp_path):
    module = load_module(SCRIPT)
    params_json = json.dumps({"provider": {"only": ["z-ai/fp8"], "allow_fallbacks": False}})
    overrides_json = json.dumps({"maxTokens": 4096})
    args = SimpleNamespace(
        output_dir=tmp_path / "out",
        checkout_root=tmp_path / "checkouts",
        prefix="assorted",
        model="openrouter-direct/z-ai/glm-5.2",
        openclaw_model_params_json=params_json,
        openclaw_model_overrides_json=overrides_json,
        thinking="high",
        agent="main",
        swe_openclaw_timeout=900,
        openclaw_max_attempts=1,
        openclaw_retry_base_seconds=15.0,
        swe_workers=2,
        swe_task_start_min_interval_seconds=5.0,
        max_input_tokens=1_000_000,
        max_cumulative_input_tokens=1_000_000,
        max_cumulative_output_tokens=500_000,
        max_tool_calls=40,
        max_agent_turns=40,
        max_tool_wall_seconds=300,
        nemoclaw_sandbox="nejumi-taiwan",
        nemoclaw_bin="nemoclaw",
        nemoclaw_workdir="/sandbox",
        nemoclaw_openclaw_config_path="/sandbox/.openclaw/openclaw.json",
        nemoclaw_checkout_transfer_mode="copy",
        nemoclaw_checkout_transfer_timeout=600,
        openclaw_tool_profile="coding",
        task_agent_prefix="tw-swe-assorted",
        session_prefix="agentic-swe-assorted",
        weave_agents_entity="llm-leaderboard",
        weave_agents_project="tc-leaderboard",
        weave_agents_agent_name="nejumi-taiwan-openclaw",
        weave_agents_limit=100,
        weave_agents_verification_timeout=120.0,
        weave_agents_poll_seconds=5.0,
        require_actual_token_usage=True,
        verify_weave_agents=True,
        use_task_agent=True,
        dry_run=False,
        deny_tool=[],
        deny_argument_pattern=[],
    )

    command = module.build_swe_command(args, tmp_path / "low_middle.jsonl")

    idx = command.index("--openclaw-model-params-json")
    assert command[idx + 1] == params_json
    idx = command.index("--openclaw-model-overrides-json")
    assert command[idx + 1] == overrides_json


def test_agentic_swe_assorted_low_middle_limit_keeps_legacy_order():
    module = load_module(SCRIPT)
    rows = [
        {"instance_id": "low-1", "agentic_swe_tier": "low"},
        {"instance_id": "low-2", "agentic_swe_tier": "low"},
        {"instance_id": "middle-1", "agentic_swe_tier": "middle"},
    ]
    args = SimpleNamespace(low_middle_limit=2, low_limit=None, middle_limit=None)

    selected = module.selected_low_middle_rows(rows, args)

    assert [row["instance_id"] for row in selected] == ["low-1", "low-2"]


def test_agentic_swe_assorted_can_select_balanced_low_middle_subset():
    module = load_module(SCRIPT)
    rows = [
        {"instance_id": "low-1", "agentic_swe_tier": "low"},
        {"instance_id": "low-2", "agentic_swe_tier": "low"},
        {"instance_id": "low-3", "agentic_swe_tier": "low"},
        {"instance_id": "middle-1", "agentic_swe_tier": "middle"},
        {"instance_id": "middle-2", "agentic_swe_tier": "middle"},
        {"instance_id": "middle-3", "agentic_swe_tier": "middle"},
    ]
    args = SimpleNamespace(low_middle_limit=None, low_limit=2, middle_limit=2)

    selected = module.selected_low_middle_rows(rows, args)

    assert [row["instance_id"] for row in selected] == [
        "low-1",
        "middle-1",
        "low-2",
        "middle-2",
    ]


def test_agentic_swe_assorted_summary_groups_by_tier_and_source():
    module = load_module(SCRIPT)
    args = SimpleNamespace(
        model="dummy/local",
        dry_run=False,
        max_input_tokens=1_000_000,
        max_cumulative_input_tokens=1_000_000,
        high_max_cumulative_input_tokens=None,
        max_cumulative_output_tokens=500_000,
        max_tool_calls=40,
        high_max_tool_calls=60,
        max_agent_turns=40,
        high_max_agent_turns=None,
        max_tool_wall_seconds=120,
        swe_workers=4,
        high_workers=2,
        llm_response_idle_timeout_seconds=900.0,
        tier_weights="low=1,middle=1,high=1",
    )
    rows = [
        {
            "source_benchmark": "SWE-bench Lite",
            "source_dataset": "princeton-nlp/SWE-bench_Lite",
            "source_subset": "low_36",
            "source_instance_id": "a",
            "agentic_swe_tier": "low",
            "instance_id": "a",
            "resolved": True,
            "score": 1.0,
            "weave_agents_conversation_url": "https://example.com/a",
            "openclaw_tool_call_count": 2,
            "openclaw_usage": {"inputTokens": 100, "outputTokens": 50},
        },
        {
            "source_benchmark": "DeepSWE",
            "source_dataset": "DataCurve DeepSWE v1.1",
            "source_subset": "essential_8",
            "source_instance_id": "b",
            "agentic_swe_tier": "high",
            "instance_id": "b",
            "resolved": False,
            "score": 0.0,
            "weave_agents_conversation_url": "https://example.com/b",
            "openclaw_tool_call_count": 40,
            "openclaw_usage": {"inputTokens": 200, "outputTokens": 100},
        },
    ]

    summary = module.build_summary(rows, args=args, elapsed=12.5)

    assert summary["total"]["total_instances"] == 2
    assert summary["total"]["resolved_instances"] == 1
    assert summary["by_tier"]["low"]["pass_at_1"] == 1.0
    assert summary["by_tier"]["high"]["pass_at_1"] == 0.0
    assert summary["by_source"]["SWE-bench Lite"]["total_instances"] == 1
    assert summary["total"]["usage"]["input_tokens"] == 300
    assert summary["total"]["usage"]["output_tokens"] == 150
    assert summary["total"]["micro_pass_at_1"] == 0.5
    assert summary["scoring"]["weighted_pass_at_1"] is None
    assert summary["scoring"]["weighted_present_pass_at_1"] == 0.5
    assert summary["scoring"]["weighted_pass_at_1_complete"] is False
    assert summary["scoring"]["missing_weighted_tiers"] == ["middle"]
    assert summary["limits"]["llm_response_idle_timeout_seconds"] == 900.0


def test_agentic_swe_assorted_summary_reports_budget_stop_outcomes():
    module = load_module(SCRIPT)
    rows = [
        {
            "resolved": True,
            "openclaw_disqualified_reason": "runtime_budget_exceeded",
            "runtime_budget": {
                "ok": False,
                "violations": [
                    {"type": "max_tool_calls"},
                    {"type": "max_agent_turns"},
                ],
            },
        },
        {
            "resolved": False,
            "runtime_budget": {
                "ok": False,
                "violations": [{"type": "max_tool_calls"}],
            },
        },
        {"resolved": True, "runtime_budget": {"ok": True, "violations": []}},
        {"resolved": False, "runtime_budget": {"ok": True, "violations": []}},
    ]

    summary = module.summarize_group(rows, model="dummy/local")
    outcomes = summary["runtime_budget_outcomes"]

    assert outcomes["exceeded_instances"] == 2
    assert outcomes["resolved_after_budget_stop"] == 1
    assert outcomes["unresolved_after_budget_stop"] == 1
    assert outcomes["pass_rate_after_budget_stop"] == 0.5
    assert outcomes["within_budget_instances"] == 2
    assert outcomes["within_budget_resolved_instances"] == 1
    assert outcomes["within_budget_pass_rate"] == 0.5
    assert outcomes["by_violation"] == {
        "max_agent_turns": {"instances": 1, "resolved_instances": 1},
        "max_tool_calls": {"instances": 2, "resolved_instances": 1},
    }


def test_agentic_swe_assorted_summary_uses_tier_macro_score_when_complete():
    module = load_module(SCRIPT)
    args = SimpleNamespace(
        model="dummy/local",
        dry_run=False,
        max_input_tokens=1_000_000,
        max_cumulative_input_tokens=1_000_000,
        high_max_cumulative_input_tokens=None,
        max_cumulative_output_tokens=500_000,
        max_tool_calls=40,
        high_max_tool_calls=60,
        max_agent_turns=40,
        high_max_agent_turns=None,
        max_tool_wall_seconds=120,
        swe_workers=4,
        high_workers=2,
        llm_response_idle_timeout_seconds=900.0,
        tier_weights="low=1,middle=1,high=1",
    )
    rows = [
        {"source_benchmark": "SWE-bench Lite", "agentic_swe_tier": "low", "resolved": True},
        {"source_benchmark": "SWE-bench Lite", "agentic_swe_tier": "low", "resolved": True},
        {"source_benchmark": "SWE-bench Lite", "agentic_swe_tier": "middle", "resolved": False},
        {"source_benchmark": "DeepSWE", "agentic_swe_tier": "high", "resolved": True},
    ]

    summary = module.build_summary(rows, args=args, elapsed=1.0)

    assert summary["total"]["micro_pass_at_1"] == 0.75
    assert summary["scoring"]["weighted_pass_at_1"] == pytest.approx(2 / 3)
    assert summary["total"]["weighted_pass_at_1"] == pytest.approx(2 / 3)
    assert summary["scoring"]["weighted_pass_at_1_complete"] is True
    assert summary["scoring"]["missing_weighted_tiers"] == []
    assert summary["scoring"]["tier_weights"] == pytest.approx({
        "low": 1 / 3,
        "middle": 1 / 3,
        "high": 1 / 3,
    })


def test_agentic_swe_assorted_writes_human_readable_report(tmp_path):
    module = load_module(SCRIPT)
    args = SimpleNamespace(
        model="dummy/local",
        dry_run=False,
        max_input_tokens=1_000_000,
        max_cumulative_input_tokens=1_000_000,
        high_max_cumulative_input_tokens=13_000_000,
        max_cumulative_output_tokens=500_000,
        max_tool_calls=40,
        high_max_tool_calls=200,
        max_agent_turns=40,
        high_max_agent_turns=140,
        max_tool_wall_seconds=120,
        swe_workers=4,
        high_workers=2,
        llm_response_idle_timeout_seconds=900.0,
        tier_weights="low=1,middle=1,high=1",
    )
    summary = module.build_summary(
        [
            {"source_benchmark": "SWE-bench Lite", "agentic_swe_tier": "low", "resolved": True},
            {"source_benchmark": "SWE-bench Lite", "agentic_swe_tier": "middle", "resolved": False},
            {"source_benchmark": "DeepSWE", "agentic_swe_tier": "high", "resolved": True},
        ],
        args=args,
        elapsed=10.0,
    )

    report_path = tmp_path / "report.md"
    module.write_markdown_report(report_path, summary=summary)
    text = report_path.read_text(encoding="utf-8")

    assert "Agentic SWE-Assorted Report" in text
    assert "official_score" in text
    assert "| low |" in text
    assert "| high_max_agent_turns | `140` |" in text


def test_agentic_swe_assorted_summary_accepts_custom_tier_weights():
    module = load_module(SCRIPT)
    args = SimpleNamespace(
        model="dummy/local",
        dry_run=False,
        max_input_tokens=1_000_000,
        max_cumulative_input_tokens=1_000_000,
        high_max_cumulative_input_tokens=None,
        max_cumulative_output_tokens=500_000,
        max_tool_calls=40,
        high_max_tool_calls=60,
        max_agent_turns=40,
        high_max_agent_turns=None,
        max_tool_wall_seconds=120,
        swe_workers=4,
        high_workers=2,
        llm_response_idle_timeout_seconds=900.0,
        tier_weights="low=2,middle=1,high=1",
    )
    rows = [
        {"source_benchmark": "SWE-bench Lite", "agentic_swe_tier": "low", "resolved": True},
        {"source_benchmark": "SWE-bench Lite", "agentic_swe_tier": "middle", "resolved": False},
        {"source_benchmark": "DeepSWE", "agentic_swe_tier": "high", "resolved": True},
    ]

    summary = module.build_summary(rows, args=args, elapsed=1.0)

    assert summary["scoring"]["weighted_pass_at_1"] == pytest.approx(0.75)
    assert summary["scoring"]["tier_weights"] == pytest.approx({
        "low": 0.5,
        "middle": 0.25,
        "high": 0.25,
    })


def test_agentic_swe_assorted_usage_tracks_cache_tokens_separately():
    module = load_module(SCRIPT)

    usage = module.usage_numbers(
        {
            "inputTokens": 100,
            "outputTokens": 20,
            "cacheReadInputTokens": 300,
            "cacheWriteInputTokens": 40,
        }
    )

    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 20
    assert usage["cache_read_input_tokens"] == 300
    assert usage["cache_write_input_tokens"] == 40
    assert usage["total_tokens"] == 460


def test_agentic_swe_assorted_summary_estimates_model_cost():
    module = load_module(SCRIPT)
    args = SimpleNamespace(
        model="openai-direct/gpt-4.1-mini-2025-04-14",
        dry_run=False,
        max_input_tokens=1_000_000,
        max_cumulative_input_tokens=1_000_000,
        high_max_cumulative_input_tokens=None,
        max_cumulative_output_tokens=500_000,
        max_tool_calls=40,
        high_max_tool_calls=60,
        max_agent_turns=40,
        high_max_agent_turns=None,
        max_tool_wall_seconds=120,
        swe_workers=4,
        high_workers=2,
        llm_response_idle_timeout_seconds=900.0,
        tier_weights="low=1,middle=1,high=1",
    )
    rows = [
        {
            "source_benchmark": "SWE-bench Lite",
            "agentic_swe_tier": "low",
            "resolved": False,
            "patch_empty": True,
            "weave_agents_ok": True,
            "openclaw_usage": {
                "inputTokens": 1_000_000,
                "outputTokens": 100_000,
                "cacheReadInputTokens": 2_000_000,
            },
        }
    ]

    summary = module.build_summary(rows, args=args, elapsed=1.0)

    assert summary["total"]["usage"]["cost_usd"] == 0.76


def test_agentic_swe_assorted_summary_estimates_gpt56_luna_cost():
    module = load_module(SCRIPT)
    args = SimpleNamespace(
        model="openai-direct/gpt-5.6-luna",
        dry_run=False,
        max_input_tokens=1_000_000,
        max_cumulative_input_tokens=1_000_000,
        high_max_cumulative_input_tokens=5_000_000,
        max_cumulative_output_tokens=500_000,
        max_tool_calls=40,
        high_max_tool_calls=72,
        max_agent_turns=40,
        high_max_agent_turns=60,
        max_tool_wall_seconds=120,
        swe_workers=4,
        high_workers=1,
        llm_response_idle_timeout_seconds=900.0,
        tier_weights="low=1,middle=1,high=1",
    )
    rows = [
        {
            "source_benchmark": "DeepSWE",
            "agentic_swe_tier": "high",
            "resolved": False,
            "weave_agents_ok": True,
            "openclaw_usage": {
                "inputTokens": 32_008,
                "outputTokens": 7_319,
                "cacheReadInputTokens": 367_689,
            },
        }
    ]

    summary = module.build_summary(rows, args=args, elapsed=1.0)

    assert summary["total"]["usage"]["cost_usd"] == pytest.approx(0.1126909)


def test_agentic_swe_assorted_summary_estimates_wandb_inference_glm52_cost():
    module = load_module(SCRIPT)
    args = SimpleNamespace(
        model="wandb-inference/zai-org/GLM-5.2",
        dry_run=False,
        max_input_tokens=1_000_000,
        max_cumulative_input_tokens=1_000_000,
        high_max_cumulative_input_tokens=5_000_000,
        max_cumulative_output_tokens=500_000,
        max_tool_calls=40,
        high_max_tool_calls=72,
        max_agent_turns=40,
        high_max_agent_turns=60,
        max_tool_wall_seconds=120,
        swe_workers=4,
        high_workers=1,
        llm_response_idle_timeout_seconds=900.0,
        tier_weights="low=1,middle=1,high=1",
    )
    rows = [
        {
            "source_benchmark": "SWE-bench Lite",
            "agentic_swe_tier": "low",
            "resolved": True,
            "weave_agents_ok": True,
            "openclaw_usage": {
                "inputTokens": 55_884,
                "outputTokens": 2_274,
                "cacheReadInputTokens": 260_928,
            },
        }
    ]

    summary = module.build_summary(rows, args=args, elapsed=1.0)

    assert summary["total"]["usage"]["cost_usd"] == pytest.approx(0.15552564)


def test_agentic_swe_assorted_summary_estimates_canonical_openrouter_glm52_cost():
    module = load_module(SCRIPT)
    args = SimpleNamespace(
        model="openrouter/z-ai/glm-5.2",
        dry_run=False,
        max_input_tokens=1_000_000,
        max_cumulative_input_tokens=1_000_000,
        high_max_cumulative_input_tokens=13_000_000,
        max_cumulative_output_tokens=500_000,
        max_tool_calls=40,
        high_max_tool_calls=200,
        max_agent_turns=40,
        high_max_agent_turns=140,
        max_tool_wall_seconds=120,
        swe_workers=4,
        high_workers=1,
        llm_response_idle_timeout_seconds=900.0,
        tier_weights="low=1,middle=1,high=1",
    )
    rows = [
        {
            "source_benchmark": "DeepSWE",
            "agentic_swe_tier": "high",
            "resolved": False,
            "weave_agents_ok": True,
            "openclaw_usage": {
                "inputTokens": 82_346,
                "outputTokens": 75_171,
                "cacheReadInputTokens": 5_108_825,
            },
        }
    ]

    summary = module.build_summary(rows, args=args, elapsed=1.0)

    assert summary["total"]["usage"]["cost_usd"] == pytest.approx(1.7743313)


def test_deepswe_usage_falls_back_to_weave_agents_trace_usage(tmp_path):
    module = load_module(SCRIPT)
    verifier = tmp_path / "weave_agents.json"
    verifier.write_text(
        """{
  "checks": [
    {
      "name": "usage",
      "ok": true,
      "agent_input_tokens": 10,
      "agent_output_tokens": 2,
      "trace_input_tokens": 1000,
      "trace_output_tokens": 50
    }
  ]
}
""",
        encoding="utf-8",
    )
    rows = module.deepswe_rows(
        metadata_rows=[
            {
                "task_name": "example-task",
                "repository": "example/repo",
                "subset": "essential_8",
            }
        ],
        result_rows=[
            {
                "task_name": "datacurve/example-task",
                "resolved": False,
                "score": 0.0,
                "openclaw_usage": {},
                "weave_agents_verifier_json": str(verifier),
            }
        ],
    )

    assert rows[0]["openclaw_usage"] == {
        "inputTokens": 10,
        "outputTokens": 2,
        "cacheReadInputTokens": 0,
        "usageSource": "weave_agents_agent_summary",
        "usageApproximate": True,
    }


def test_deepswe_rows_reads_nested_openclaw_patch_and_runtime_metadata():
    module = load_module(SCRIPT)
    rows = module.deepswe_rows(
        metadata_rows=[
            {
                "task_name": "example-task",
                "repository": "example/repo",
                "subset": "essential_8",
            }
        ],
        result_rows=[
            {
                "task_name": "datacurve/example-task",
                "resolved": False,
                "score": 0.0,
                "agent_result": {
                    "metadata": {
                        "patch_apply": {"reason": "empty_patch"},
                        "openclaw": {
                            "returncode": 1,
                            "openclaw_disqualified_reason": "runtime_budget_exceeded",
                            "deepswe_scoreable_failure_reason": "llm_response_idle_timeout",
                            "deepswe_patch_bytes": 0,
                            "openclaw_tool_call_count": 28,
                            "runtime_budget": {
                                "ok": False,
                                "observed": {"estimated_input_tokens": 29137},
                                "live": {"reason": "llm_response_idle_timeout"},
                            },
                            "weave_agents_ok": False,
                            "weave_agents_conversation_url": "https://wandb.example/conversation",
                            "openclaw_result_path": "/tmp/openclaw_result.json",
                        },
                    }
                },
            }
        ],
    )

    assert rows[0]["patch_empty"] is True
    assert rows[0]["openclaw_returncode"] == 1
    assert rows[0]["openclaw_disqualified_reason"] == "runtime_budget_exceeded"
    assert rows[0]["scoreable_failure_reason"] == "llm_response_idle_timeout"
    assert rows[0]["openclaw_tool_call_count"] == 28
    assert rows[0]["openclaw_usage"] == {
        "inputTokens": 29137,
        "outputTokens": 0,
        "cacheReadInputTokens": 0,
        "cacheWriteInputTokens": 0,
        "usageSource": "runtime_budget_estimated_input_floor",
        "usageApproximate": True,
        "usageLowerBound": True,
    }
    assert rows[0]["runtime_budget"]["live"]["reason"] == "llm_response_idle_timeout"
    assert rows[0]["weave_agents_conversation_url"] == "https://wandb.example/conversation"


def test_deepswe_usage_falls_back_to_weave_agents_content_capture_health(tmp_path):
    module = load_module(SCRIPT)
    verifier = tmp_path / "weave_agents.json"
    verifier.write_text(
        """{
  "content_capture_health": {
    "trace_input_tokens": 4321501,
    "trace_output_tokens": 17563,
    "conversation_input_tokens": 4321501,
    "conversation_output_tokens": 17563
  }
}
""",
        encoding="utf-8",
    )
    rows = module.deepswe_rows(
        metadata_rows=[
            {
                "task_name": "psd-tools-blend-range-api",
                "repository": "psd-tools/psd-tools",
                "subset": "essential_8",
            }
        ],
        result_rows=[
            {
                "task_name": "datacurve/psd-tools-blend-range-api",
                "resolved": False,
                "score": 0.0,
                "openclaw_usage": {},
                "weave_agents_verifier_json": str(verifier),
            }
        ],
    )

    assert rows[0]["openclaw_usage"] == {
        "inputTokens": 4321501,
        "outputTokens": 17563,
        "cacheReadInputTokens": 0,
        "usageSource": "weave_agents_trace",
        "usageApproximate": True,
    }


def test_lite_usage_falls_back_to_weave_agents_trace_usage(tmp_path):
    module = load_module(SCRIPT)
    verifier = tmp_path / "weave_agents.json"
    verifier.write_text(
        """{
  "checks": [
    {
      "name": "usage",
      "ok": true,
      "trace_input_tokens": 1234,
      "trace_output_tokens": 56
    }
  ]
}
""",
        encoding="utf-8",
    )
    rows = module.lite_rows(
        source_rows=[
            {
                "instance_id": "django__django-1",
                "repo": "django/django",
                "agentic_swe_tier": "low",
            }
        ],
        patch_rows=[
            {
                "instance_id": "django__django-1",
                "patch": "",
                "openclaw_usage": {},
                "weave_agents_verifier_json": str(verifier),
            }
        ],
        eval_results={"django__django-1": False},
    )

    assert rows[0]["openclaw_usage"] == {
        "inputTokens": 1234,
        "outputTokens": 56,
        "cacheReadInputTokens": 0,
        "usageSource": "weave_agents_trace",
        "usageApproximate": True,
    }
