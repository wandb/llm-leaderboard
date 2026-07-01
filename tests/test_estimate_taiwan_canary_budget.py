import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = REPO_ROOT / "scripts" / "analysis" / "estimate_taiwan_canary_budget.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def load_usage_module():
    path = REPO_ROOT / "scripts" / "analysis" / "estimate_agentic_usage_costs.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def write_usage(path: Path, *, category: str, model: str, input_tokens: int, output_tokens: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model": model,
                "openclaw_result_path": f"{category}/{path.stem}.json",
                "openclaw_usage": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "cacheRead": 0,
                    "cacheWrite": 0,
                },
            }
        ),
        encoding="utf-8",
    )


def test_default_target_model_is_openai_direct_canary():
    module = load_module()

    assert module.DEFAULT_TARGET_MODEL == "openai-direct/gpt-4.1-mini-2025-04-14"
    assert "openrouter" not in module.DEFAULT_TARGET_MODEL.lower()
    assert module.PRICE_PER_MILLION[module.DEFAULT_TARGET_MODEL] == {
        "input": 0.40,
        "output": 1.60,
        "cacheRead": 0.10,
        "cacheWrite": 0.0,
    }


def test_main_estimates_openai_direct_canary_budget(tmp_path, monkeypatch, capsys):
    module = load_module()
    root = tmp_path / "outputs"
    write_usage(
        root / "agentic_math" / "row1.json",
        category="agentic_math",
        model="openai-direct/gpt-4.1-mini-2025-04-14",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    write_usage(
        root / "swebench_pro" / "row1.json",
        category="swebench_pro",
        model="openai-direct/gpt-4.1-mini-2025-04-14",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    output = tmp_path / "budget.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "estimate_taiwan_canary_budget.py",
            str(root),
            "--math-tasks",
            "1",
            "--swe-tasks",
            "1",
            "--nonagentic-buffer-usd",
            "0",
            "--output",
            str(output),
        ],
    )

    module.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["target_model"] == "openai-direct/gpt-4.1-mini-2025-04-14"
    assert payload["estimated_total_usd"] == {"low": 4.0, "mid": 4.0, "high": 4.0}
    assert json.loads(capsys.readouterr().out)["target_model"] == payload["target_model"]


def test_main_rejects_missing_agentic_budget_evidence(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "estimate_taiwan_canary_budget.py",
            str(tmp_path / "empty_outputs"),
        ],
    )

    with pytest.raises(SystemExit, match="agentic_math, swebench_pro"):
        module.main()


def test_main_rejects_partial_agentic_budget_evidence(tmp_path, monkeypatch):
    module = load_module()
    root = tmp_path / "outputs"
    write_usage(
        root / "agentic_math" / "row1.json",
        category="agentic_math",
        model="openai-direct/gpt-4.1-mini-2025-04-14",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "estimate_taiwan_canary_budget.py",
            str(root),
        ],
    )

    with pytest.raises(SystemExit, match="swebench_pro"):
        module.main()


def test_main_rejects_unknown_pricing_model(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "estimate_taiwan_canary_budget.py",
            str(tmp_path),
            "--target-model",
            "openai-direct/unknown",
        ],
    )

    with pytest.raises(SystemExit, match="unknown target model for pricing"):
        module.main()


def test_usage_scan_skips_swebench_checkouts(tmp_path):
    module = load_usage_module()
    model = "openai-direct/gpt-4.1-mini-2025-04-14"
    write_usage(
        tmp_path / "swebench_pro" / "row1.json",
        category="swebench_pro",
        model=model,
        input_tokens=1,
        output_tokens=1,
    )
    write_usage(
        tmp_path / "swebench_pro_checkouts" / "repo" / "package.json",
        category="swebench_pro",
        model=model,
        input_tokens=999_999,
        output_tokens=999_999,
    )

    rows = module.scan(tmp_path)

    assert len(rows) == 1
    assert rows[0]["source"].endswith("swebench_pro/row1.json")
