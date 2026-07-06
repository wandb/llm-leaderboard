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


def test_missing_included_units_keep_taiwan_mean_incomplete():
    module = load_module(REPO_ROOT / "scripts" / "evaluator" / "aggregate_taiwan.py")
    unit_df = pd.DataFrame(
        [
            {"category": "GLP", "score_0_to_100": 80.0, "score_included": True},
            {"category": "GLP", "score_0_to_100": np.nan, "score_included": True},
        ]
    )

    assert np.isnan(module._mean_for_included_units(unit_df, "GLP"))
