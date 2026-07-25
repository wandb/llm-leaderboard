import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path):
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pending_units_are_excluded_from_taiwan_means():
    module = load_module(REPO_ROOT / "scripts" / "evaluator" / "aggregate_taiwan.py")
    unit_df = pd.DataFrame(
        [
            {"category": "GLP", "score_0_to_100": 80.0, "score_included": True},
            {"category": "ALT", "score_0_to_100": 60.0, "score_included": True},
            {"category": "ALT", "score_0_to_100": np.nan, "score_included": False},
        ]
    )

    assert module._mean_for_included_units(unit_df, "GLP") == 80.0
    assert module._mean_for_included_units(unit_df, "ALT") == 60.0
    assert module._mean_for_included_units(unit_df) == 70.0
    assert module._weighted_overall({"GLP": 80.0, "ALT": 60.0}, {"GLP": 8.0, "ALT": 6.0}) == (
        (80.0 * 8.0 + 60.0 * 6.0) / 14.0
    )


def test_write_aggregate_summary_logs_display_and_namespaced_scores():
    module = load_module(REPO_ROOT / "scripts" / "evaluator" / "aggregate_taiwan.py")

    class FakeRun:
        summary = {}

    run = FakeRun()
    module._write_aggregate_summary(
        run,
        category_scores={"GLP": 34.5, "ALT": 67.0},
        overall=48.4285714286,
        missing_required_count=0,
        pending_count=1,
    )

    assert run.summary == {
        "GLP": 34.5,
        "taiwan_glp_score": 34.5,
        "ALT": 67.0,
        "taiwan_alt_score": 67.0,
        "Overall": 48.4285714286,
        "taiwan_overall_score": 48.4285714286,
        "taiwan_missing_required_count": 0,
        "taiwan_pending_count": 1,
    }


def test_missing_included_units_keep_taiwan_mean_incomplete():
    module = load_module(REPO_ROOT / "scripts" / "evaluator" / "aggregate_taiwan.py")
    unit_df = pd.DataFrame(
        [
            {"category": "GLP", "score_0_to_100": 80.0, "score_included": True},
            {"category": "GLP", "score_0_to_100": np.nan, "score_included": True},
        ]
    )

    assert np.isnan(module._mean_for_included_units(unit_df, "GLP"))
    missing = module._required_missing_rows(
        unit_df.assign(required=[True, True], unit_id=["ok", "missing"])
    )
    assert missing["unit_id"].tolist() == ["missing"]


def test_taiwan_taxonomy_rejects_auto_scale_and_required_pending():
    module = load_module(REPO_ROOT / "scripts" / "evaluator" / "aggregate_taiwan.py")
    taxonomy = {
        "categories": {"GLP": {"weight": 8}},
        "units": [
            {
                "id": "bad",
                "category": "GLP",
                "scale": "auto",
                "required": True,
                "pending": True,
            }
        ],
    }

    try:
        module._validate_taxonomy(taxonomy)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("taxonomy validation should fail")

    assert "scale must be" in message
    assert "required=true and pending=true" in message


def test_taiwan_taxonomy_file_is_valid():
    from omegaconf import OmegaConf

    module = load_module(REPO_ROOT / "scripts" / "evaluator" / "aggregate_taiwan.py")
    taxonomy = OmegaConf.to_container(
        OmegaConf.load(REPO_ROOT / "taxonomies" / "nejumi45_taiwan.yaml"),
        resolve=True,
    )

    module._validate_taxonomy(taxonomy)
    assert module._category_weights(taxonomy) == {"ALT": 6.0, "GLP": 8.0}


def test_fraction_score_accepts_percent_literal_without_double_scaling():
    module = load_module(REPO_ROOT / "scripts" / "evaluator" / "aggregate_taiwan.py")

    assert module._coerce_score_to_native_scale("38.72%", "fraction") == 0.3872
    assert module._normalize_score("38.72%", "fraction") == 38.72
    assert module._normalize_score(0.3872, "fraction") == 38.72


def test_score_normalization_rejects_empty_and_non_finite_values():
    module = load_module(REPO_ROOT / "scripts" / "evaluator" / "aggregate_taiwan.py")

    for value in ("", " ", "not-a-score", "nan", float("inf")):
        assert np.isnan(module._normalize_score(value, "fraction"))


def test_multi_source_scores_are_coerced_before_aggregation(monkeypatch):
    module = load_module(REPO_ROOT / "scripts" / "evaluator" / "aggregate_taiwan.py")
    values = iter([("50%", "first"), (0.25, "second")])
    monkeypatch.setattr(module, "_read_score_source", lambda *args, **kwargs: next(values))
    unit = {
        "scale": "fraction",
        "sources": [{"table_name": "first"}, {"table_name": "second"}],
        "aggregation": "mean",
    }

    raw_score, table_names = module._unit_raw_score(object(), {}, unit)

    assert raw_score == 0.375
    assert table_names == "first,second"
