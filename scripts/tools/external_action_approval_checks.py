#!/usr/bin/env python3
"""Shared validation helpers for source-bound external-action approvals."""

from __future__ import annotations

from typing import Any


def numeric_usd(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace("$", "").replace(",", "")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def approval_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("approval_results")
    if not isinstance(results, list):
        return []
    return [item for item in results if isinstance(item, dict)]


def approval_result(payload: dict[str, Any], requirement: str) -> dict[str, Any] | None:
    for item in approval_results(payload):
        if item.get("requirement") == requirement:
            return item
    return None


def extract_paid_api_approval(payload: dict[str, Any]) -> dict[str, Any]:
    result = {
        "present": False,
        "approved": False,
        "approved_budget_usd": None,
        "minimum_approved_budget_usd": None,
        "minimum_approved_budget_source": "",
        "approved_model_scope": "",
    }
    item = approval_result(payload, "paid_api")
    if not isinstance(item, dict):
        return result
    result["present"] = True
    result["approved"] = item.get("approved") is True
    result["approved_budget_usd"] = numeric_usd(item.get("approved_budget_usd"))
    result["minimum_approved_budget_usd"] = numeric_usd(
        item.get("minimum_approved_budget_usd")
    )
    result["minimum_approved_budget_source"] = (
        item.get("minimum_approved_budget_source")
        if isinstance(item.get("minimum_approved_budget_source"), str)
        else ""
    )
    result["approved_model_scope"] = (
        item.get("approved_model_scope")
        if isinstance(item.get("approved_model_scope"), str)
        else ""
    )
    return result


def validate_approval_results(payload: dict[str, Any]) -> dict[str, Any]:
    raw_results = payload.get("approval_results")
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    if not isinstance(raw_results, list):
        return {
            "valid": False,
            "checked_count": 0,
            "records": [],
            "errors": ["approval_results must be a list"],
        }

    required_count = payload.get("required_approval_count")
    if isinstance(required_count, bool) or not isinstance(required_count, int):
        required_count = None

    for raw_item in raw_results:
        if not isinstance(raw_item, dict):
            errors.append("approval_results contains a non-object item")
            continue
        requirement = str(raw_item.get("requirement") or "")
        item_errors = raw_item.get("errors")
        if item_errors is None:
            item_errors = []
        elif not isinstance(item_errors, list):
            item_errors = ["approval result errors must be a list"]
        required = raw_item.get("required") is True
        approved = raw_item.get("approved") is True
        record: dict[str, Any] = {
            "requirement": requirement,
            "required": required,
            "approved": approved,
            "errors": list(item_errors),
        }
        if required and not approved:
            record["errors"].append(
                f"{requirement or 'approval_result'} is required but not approved"
            )
        if requirement == "paid_api":
            budget = numeric_usd(raw_item.get("approved_budget_usd"))
            minimum_budget = numeric_usd(raw_item.get("minimum_approved_budget_usd"))
            record["approved_budget_usd"] = budget
            record["minimum_approved_budget_usd"] = minimum_budget
            record["minimum_approved_budget_source"] = (
                raw_item.get("minimum_approved_budget_source")
                if isinstance(raw_item.get("minimum_approved_budget_source"), str)
                else ""
            )
            if required:
                if budget is None or budget <= 0:
                    record["errors"].append(
                        "paid_api.approved_budget_usd must be a positive USD number"
                    )
                if minimum_budget is None:
                    record["errors"].append(
                        "paid_api.minimum_approved_budget_usd must be present"
                    )
                elif budget is not None and budget < minimum_budget:
                    record["errors"].append(
                        "paid_api.approved_budget_usd must be greater than or "
                        "equal to minimum_approved_budget_usd"
                    )
        if record["errors"]:
            errors.extend(
                f"{requirement or 'approval_result'}: {error}"
                for error in record["errors"]
            )
        records.append(record)

    required_record_count = sum(1 for record in records if record.get("required") is True)
    if required_count is not None and required_record_count != required_count:
        errors.append(
            "approval_results required=true row count must equal "
            "required_approval_count"
        )

    return {
        "valid": not errors,
        "checked_count": len(records),
        "records": records,
        "errors": errors,
    }
