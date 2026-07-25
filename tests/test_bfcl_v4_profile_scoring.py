import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
V4_PACKAGE_ROOT = (
    REPO_ROOT / "scripts" / "evaluator" / "evaluate_utils" / "bfcl_v4_pkg"
)
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (str(V4_PACKAGE_ROOT), str(SCRIPTS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

helper = importlib.import_module(
    "bfcl_eval.eval_checker.eval_runner_helper"
)


def _score(value, count=30, display=None):
    return {
        "accuracy": value,
        "total_count": count,
        "display_accuracy": value if display is None else display,
    }


def test_taiwan_profile_excludes_missing_parallel_categories(monkeypatch):
    monkeypatch.setenv("BFCL_IGNORE_MISSING_CATEGORIES", "1")
    values = [
        _score(0.8),
        _score(0.6),
        _score(0.0, count=0, display="N/A"),
        _score(0.0, count=0, display="N/A"),
    ]

    result = helper.calculate_unweighted_accuracy(
        values, display_na_if_category_missing=False
    )

    assert result["accuracy"] == 0.7
    assert result["total_count"] == 60


def test_taiwan_profile_renormalizes_domain_weights(monkeypatch):
    monkeypatch.setenv("BFCL_IGNORE_MISSING_CATEGORIES", "1")
    values = [
        _score(0.8),
        _score(0.6),
        _score(0.7),
        _score(0.5),
        _score(0.0, count=0, display="N/A"),
    ]

    result = helper.calculate_percentage_weighted_accuracy(
        values,
        [10, 10, 10, 30, 40],
        display_na_if_category_missing=False,
    )

    expected = (0.8 * 10 + 0.6 * 10 + 0.7 * 10 + 0.5 * 30) / 60
    assert result["accuracy"] == expected


def test_upstream_behavior_is_preserved_without_profile_flag(monkeypatch):
    monkeypatch.delenv("BFCL_IGNORE_MISSING_CATEGORIES", raising=False)
    values = [_score(1.0), _score(0.0, count=0, display="N/A")]

    result = helper.calculate_unweighted_accuracy(
        values, display_na_if_category_missing=False
    )

    assert result["accuracy"] == 0.5
