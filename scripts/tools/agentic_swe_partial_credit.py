#!/usr/bin/env python3
"""Official partial-credit helpers for Agentic SWE-Assorted.

Complete resolution receives 1 point. A scoreable unresolved patch can receive
at most 0.3 points from FAIL_TO_PASS and PASS_TO_PASS evidence.
"""

from __future__ import annotations

from typing import Any


OFFICIAL_SCORE_VERSION = "resolved-1-else-f2p-squared-p2p-cap0.3-v2"
OFFICIAL_PARTIAL_CREDIT_CAP = 0.3

# Compatibility aliases for already materialized result tables.
DIAGNOSTIC_PARTIAL_CREDIT_VERSION = OFFICIAL_SCORE_VERSION
DIAGNOSTIC_PARTIAL_CREDIT_CAP = OFFICIAL_PARTIAL_CREDIT_CAP


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return max(0, int(value))
    return 0


def _test_group_counts(group: Any) -> tuple[int, int]:
    if not isinstance(group, dict):
        return 0, 0
    successes = group.get("success")
    failures = group.get("failure")
    passed = len(successes) if isinstance(successes, list) else 0
    failed = len(failures) if isinstance(failures, list) else 0
    return passed, passed + failed


def verifier_rewards(verifier_result: Any) -> dict[str, Any]:
    """Return the DeepSWE reward/count mapping, if one is present."""
    if not isinstance(verifier_result, dict):
        return {}
    rewards = verifier_result.get("rewards")
    if isinstance(rewards, dict):
        return rewards
    metrics = verifier_result.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    return verifier_result


def build_diagnostic_partial_credit(
    *,
    resolved: bool,
    f2p_passed: Any,
    f2p_total: Any,
    p2p_passed: Any = 0,
    p2p_total: Any = 0,
    patch_applied: bool,
    scoreable: bool = True,
    evidence_source: str,
) -> dict[str, Any]:
    """Build the official and binary audit scores from F2P/P2P evidence.

    Resolved tasks always receive 1. Unresolved tasks can receive at most 0.3
    points, and only when a patch was
    applied and F2P evidence exists.  P2P has a neutral ratio of 1 when the
    task defines no P2P tests.
    """
    f2p_total_int = _nonnegative_int(f2p_total)
    f2p_passed_int = min(_nonnegative_int(f2p_passed), f2p_total_int)
    p2p_total_int = _nonnegative_int(p2p_total)
    p2p_passed_int = min(_nonnegative_int(p2p_passed), p2p_total_int)
    f2p_fraction = f2p_passed_int / f2p_total_int if f2p_total_int else 0.0
    p2p_fraction = p2p_passed_int / p2p_total_int if p2p_total_int else 1.0
    eligible = bool(scoreable and patch_applied and f2p_total_int > 0 and not resolved)
    partial = (
        OFFICIAL_PARTIAL_CREDIT_CAP * f2p_fraction**2 * p2p_fraction
        if eligible
        else 0.0
    )
    binary_score = 1.0 if resolved else 0.0
    official_score = binary_score if resolved else partial
    return {
        "score": official_score,
        "official_score": official_score,
        "official_partial_credit": partial,
        "official_score_version": OFFICIAL_SCORE_VERSION,
        "binary_score": binary_score,
        # Retained as aliases so historical analysis code can read both schema
        # generations without silently changing old artifacts.
        "diagnostic_partial_credit": partial,
        "diagnostic_score_with_partial": official_score,
        "diagnostic_partial_credit_eligible": eligible,
        "diagnostic_partial_credit_version": OFFICIAL_SCORE_VERSION,
        "diagnostic_evidence_source": evidence_source,
        "f2p_passed": f2p_passed_int,
        "f2p_total": f2p_total_int,
        "f2p_pass_fraction": f2p_fraction,
        "p2p_passed": p2p_passed_int,
        "p2p_total": p2p_total_int,
        "p2p_pass_fraction": p2p_fraction,
        "diagnostic_patch_applied": bool(patch_applied),
        "diagnostic_scoreable": bool(scoreable),
    }


def from_swebench_report(instance_id: str, detail: Any) -> dict[str, Any]:
    detail = detail if isinstance(detail, dict) else {}
    tests_status = detail.get("tests_status")
    tests_status = tests_status if isinstance(tests_status, dict) else {}
    f2p_passed, f2p_total = _test_group_counts(tests_status.get("FAIL_TO_PASS"))
    p2p_passed, p2p_total = _test_group_counts(tests_status.get("PASS_TO_PASS"))
    return build_diagnostic_partial_credit(
        resolved=bool(detail.get("resolved")),
        f2p_passed=f2p_passed,
        f2p_total=f2p_total,
        p2p_passed=p2p_passed,
        p2p_total=p2p_total,
        patch_applied=bool(detail.get("patch_successfully_applied")),
        scoreable=bool(detail),
        evidence_source=f"swebench_report:{instance_id}",
    )


def from_deepswe_verifier(
    verifier_result: Any,
    *,
    resolved: bool,
    patch_applied: bool,
    scoreable: bool = True,
) -> dict[str, Any]:
    rewards = verifier_rewards(verifier_result)
    return build_diagnostic_partial_credit(
        resolved=resolved,
        f2p_passed=rewards.get("f2p_passed"),
        f2p_total=rewards.get("f2p_total"),
        p2p_passed=rewards.get("p2p_passed"),
        p2p_total=rewards.get("p2p_total"),
        patch_applied=patch_applied,
        scoreable=scoreable and bool(rewards),
        evidence_source="deepswe_verifier",
    )
