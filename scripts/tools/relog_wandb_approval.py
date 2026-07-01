#!/usr/bin/env python3
"""Shared offline approval checks for W&B relog scripts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from external_action_approval_checks import validate_approval_results


HEX_CHARS = set("0123456789abcdef")


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_report_relative_path(path_value: str, *, report_path: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else report_path.parent / path


def is_lower_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in HEX_CHARS for char in value)
    )


def approval_result(payload: dict[str, Any], requirement: str) -> dict[str, Any] | None:
    results = payload.get("approval_results")
    if not isinstance(results, list):
        return None
    for item in results:
        if isinstance(item, dict) and item.get("requirement") == requirement:
            return item
    return None


def approval_requirement(packet: dict[str, Any], requirement: str) -> dict[str, Any] | None:
    requirements = packet.get("approval_requirements")
    if not isinstance(requirements, list):
        return None
    for item in requirements:
        if isinstance(item, dict) and item.get("requirement") == requirement:
            return item
    return None


def validate_external_action_approval_for_wandb_write(
    report_path: Path,
    *,
    entity: str,
    project: str,
    expected_source_packet_path: Path | None = None,
) -> None:
    report = read_json_object(report_path)
    errors: list[str] = []
    expected_source_sha: str | None = None
    if expected_source_packet_path is not None:
        try:
            expected_source_sha = sha256_file(expected_source_packet_path)
        except OSError as exc:
            errors.append(
                f"expected external action approval source packet is not readable: {exc}"
            )
    if report.get("schema_version") != 1:
        errors.append("external action approval report schema_version must be 1")
    if report.get("ok") is not True:
        errors.append("external action approval report ok must be true")
    if report.get("status") != "approved":
        errors.append("external action approval report status must be approved")
    if report.get("all_required_approvals_granted") is not True:
        errors.append("external action approval report must grant all required approvals")
    if report.get("granted_approval_count") != report.get("required_approval_count"):
        errors.append("external action approval granted count must equal required count")
    approval_results_validation = validate_approval_results(report)
    errors.extend(
        str(error)
        for error in approval_results_validation.get("errors", [])
        if isinstance(error, str)
    )

    source_binding = report.get("source_binding")
    if not isinstance(source_binding, dict) or source_binding.get("bound") is not True:
        errors.append("external action approval report must be source-bound")
    else:
        if source_binding.get("errors") not in ([], None):
            errors.append("external action approval source binding must have no errors")
        if source_binding.get("source_packet_readable") is not True:
            errors.append("external action approval source packet must be readable")
        source_packet_json = source_binding.get("source_packet_json")
        if not isinstance(source_packet_json, str) or not source_packet_json.strip():
            errors.append("external action approval source packet path is missing")
        source_sha = source_binding.get("source_approval_packet_sha256")
        if not is_lower_sha256(source_sha):
            errors.append(
                "external action approval source packet sha256 must be 64 lowercase hex characters"
            )
        elif isinstance(source_packet_json, str) and source_packet_json.strip():
            source_packet_path = resolve_report_relative_path(
                source_packet_json,
                report_path=report_path,
            )
            if expected_source_packet_path is not None:
                if source_packet_path.resolve() != expected_source_packet_path.resolve():
                    errors.append(
                        "external action approval source packet path does not match "
                        "expected source packet"
                    )
            try:
                observed_source_sha = sha256_file(source_packet_path)
            except OSError as exc:
                errors.append(f"external action approval source packet is not readable: {exc}")
            else:
                if observed_source_sha != source_sha:
                    errors.append(
                        "external action approval source packet sha256 does not match source packet"
                    )
                if expected_source_sha is not None and observed_source_sha != expected_source_sha:
                    errors.append(
                        "external action approval source packet sha256 does not match "
                        "expected source packet"
                    )

    for requirement in ("wandb_access", "wandb_write"):
        result = approval_result(report, requirement)
        if not isinstance(result, dict):
            errors.append(f"{requirement} approval result is missing")
        elif result.get("required") is True and result.get("approved") is not True:
            errors.append(f"{requirement} approval must be granted")

    packet_path_value = report.get("approval_packet_json")
    if not isinstance(packet_path_value, str) or not packet_path_value.strip():
        errors.append("external action approval report approval_packet_json is missing")
    else:
        packet_path = resolve_report_relative_path(packet_path_value, report_path=report_path)
        if isinstance(source_binding, dict):
            reviewed_packet_json = source_binding.get("reviewed_packet_json")
            if not isinstance(reviewed_packet_json, str) or not reviewed_packet_json.strip():
                errors.append("external action approval reviewed packet path is missing")
            else:
                reviewed_packet_path = resolve_report_relative_path(
                    reviewed_packet_json,
                    report_path=report_path,
                )
                if reviewed_packet_path.resolve() != packet_path.resolve():
                    errors.append(
                        "external action approval reviewed packet path does not match approval_packet_json"
                    )
        try:
            packet = read_json_object(packet_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"external action approval packet is not readable: {exc}")
        else:
            wandb_write = approval_requirement(packet, "wandb_write")
            if not isinstance(wandb_write, dict):
                errors.append("wandb_write approval requirement is missing from packet")
            else:
                approved_entity = wandb_write.get("approved_wandb_entity")
                approved_project = wandb_write.get("approved_wandb_project")
                if approved_entity != entity:
                    errors.append("wandb_write approved_wandb_entity does not match target entity")
                if approved_project != project:
                    errors.append("wandb_write approved_wandb_project does not match target project")

    if errors:
        raise ValueError("; ".join(errors))
