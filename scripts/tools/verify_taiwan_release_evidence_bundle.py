#!/usr/bin/env python3
"""Verify a Taiwan release evidence bundle.

The verifier checks that manifest.json is readable and that every bundled file
listed in the manifest exists, has the expected size, and matches its sha256.
It is offline and does not query W&B or model providers.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in stripped runtime envs.
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_GATE_JSON_NAME_RE = re.compile(r"^taiwan_release_gate_(\d{8}T\d{6}Z)\.json$")
LATEST_POINTER_JSON_NAME = "latest_taiwan_release_gate.json"
LATEST_POINTER_VERIFICATION_JSON_NAME_RE = re.compile(
    r"^latest_taiwan_release_gate_verify_(\d{8}T\d{6}Z)\.json$"
)
STANDALONE_OPERATOR_PLAN_JSON_NAME_RE = re.compile(
    r"^taiwan_release_operator_plan_(\d{8}T\d{6}Z)\.json$"
)
STANDALONE_OPERATOR_PLAN_MARKDOWN_NAME_RE = re.compile(
    r"^taiwan_release_operator_plan_(\d{8}T\d{6}Z)\.md$"
)
REQUIRED_WEAVE_AGENTS_CHECK_NAMES = {
    "request_model",
    "trace_timestamp_quality",
    "trace_order",
    "trace_user_message_order",
    "trace_final_answer_order",
}
AGENTIC_REQUIRED_WANDB_BENCHMARKS = {"agentic_math", "agentic_swe"}
WEAVE_AGENTS_CANARY_COMPLETION_PHASES = {"agentic", "full"}
WEAVE_AGENTS_QUERY_SOURCE_KIND = "wandb_agents_api"
WEAVE_AGENTS_API_BASE_URL = "https://trace.wandb.ai"
WEAVE_AGENTS_QUERY_ENDPOINT = "/agents/query"
WEAVE_AGENTS_SPANS_QUERY_ENDPOINT = "/agents/spans/query"
WANDB_COMPLETION_QUERY_SOURCE_KIND = "wandb_sdk"
WANDB_COMPLETION_SCHEMA_VERSION = 1
WANDB_COMPLETION_API_TIMEOUT_SECONDS = 60
COMMAND_SCRIPT_SUFFIXES = (".py", ".sh", ".mjs")
RELOG_WANDB_APPROVAL_HELPER_SCRIPT = "scripts/tools/relog_wandb_approval.py"
RELOG_COMMAND_SCRIPT_PATHS = {
    "scripts/tools/log_agentic_math_results_to_wandb.py",
    "scripts/tools/log_agentic_swe_results_to_wandb.py",
}
OPERATOR_EXECUTION_PLAN_RENDERER_SCRIPT = (
    "scripts/tools/render_taiwan_operator_execution_plan.py"
)
WEAVE_CONTENT_CANARY_GATE_CONTRACT_SCRIPT = (
    "scripts/tools/weave_content_canary_gate_contract.py"
)
NEMOCLAW_CANARY_READINESS_SCRIPT = "scripts/tools/check_taiwan_canary_readiness.py"
NEMOCLAW_ADOPTION_SCRIPT = "scripts/tools/check_taiwan_nemoclaw_adoption.py"
REQUIRED_NEMOCLAW_CANARY_REMOTE_LOOKUP_CHECK_NAMES = {
    "agentic Math denies remote lookup via deny_argument_pattern",
    "agentic Math denies remote lookup via deny_tool",
    "agentic SWE denies remote lookup via deny_argument_pattern",
    "agentic SWE denies remote lookup via deny_tool",
}
NEMOCLAW_AGENTIC_CONFIG_REQUIRED_DENIED_TOOLS = {
    "*search*",
    "browser",
    "browser_*",
    "code_execution",
    "web_fetch",
    "web_search",
}
NEMOCLAW_AGENTIC_CONFIG_REQUIRED_DENIED_ARGUMENT_PATTERNS = {
    r"\b(curl|wget)\b",
    r"\b(requests|urllib|httpx)\.",
    "https?://",
}
NEMOCLAW_AGENTIC_CONFIG_REQUIRED_ALLOWED_LOCAL_TOOLS = {"exec"}
NEMOCLAW_CANARY_READINESS_SCRIPT_SOURCE_TOKENS = (
    ("remote lookup deny tools constant", "AGENTIC_REQUIRED_DENIED_TOOLS"),
    ("local OpenClaw exec allow constant", "AGENTIC_REQUIRED_ALLOWED_LOCAL_TOOLS"),
    (
        "remote lookup deny argument patterns constant",
        "AGENTIC_REQUIRED_DENIED_ARGUMENT_PATTERNS",
    ),
    ("remote lookup deny policy checker", "def deny_policy_check("),
    ("local OpenClaw exec allow checker", "def local_tool_allow_check("),
    ("Math deny_tool check", '"agentic_math",\n                        "deny_tool"'),
    ("SWE deny_tool check", '"swebench_pro",\n                        "deny_tool"'),
    ("Math local exec allow check", "local_tool_allow_check(cfg, \"agentic_math\", \"Math\")"),
    ("SWE local exec allow check", "local_tool_allow_check(cfg, \"swebench_pro\", \"SWE\")"),
    ("Math deny_argument_pattern check", '"agentic_math",\n                        "deny_argument_pattern"'),
    ("SWE deny_argument_pattern check", '"swebench_pro",\n                        "deny_argument_pattern"'),
    ("NeMoClaw status JSON introspection", "def _run_json_status("),
    ("NeMoClaw detailed status policy parser", "def _network_policy_names_from_status_detail("),
    ("NeMoClaw sandbox policy detail parser", "def _sandbox_policy_detail("),
    (
        "NeMoClaw runtime network policy allowlist constant",
        "NEMOCLAW_ALLOWED_RUNTIME_NETWORK_POLICIES",
    ),
    ("NeMoClaw runtime policy check", "NeMoClaw sandbox runtime policy is introspectable"),
    ("NeMoClaw runtime policy count evidence", '"policy_count"'),
    ("NeMoClaw W&B/Weave runtime policy check", "NeMoClaw W&B/Weave runtime policy is present"),
    ("NeMoClaw W&B/Weave policy evidence", "wandb_weave_policy_present"),
    (
        "NeMoClaw runtime network allowlist check",
        "NeMoClaw runtime network policies are allowlisted",
    ),
    (
        "NeMoClaw unknown runtime network policies evidence",
        "unknown_runtime_network_policies",
    ),
    ("NeMoClaw runtime policy anti-cheat note", "OpenClaw deny_tool"),
)
NEMOCLAW_ADOPTION_SCRIPT_SOURCE_TOKENS = (
    ("W&B/Weave runtime policy criterion", "def runtime_wandb_weave_policy("),
    ("W&B/Weave runtime policy blocker", '"runtime_wandb_weave_policy"'),
    ("W&B/Weave canary check parser", "NeMoClaw W&B/Weave runtime policy is present"),
    ("W&B/Weave policy evidence field", "wandb_weave_policy_present"),
    (
        "runtime network policy allowlist criterion",
        "def runtime_network_policy_allowlist(",
    ),
    (
        "runtime network policy allowlist blocker",
        '"runtime_network_policy_allowlist"',
    ),
    (
        "runtime network allowlist canary check parser",
        "NeMoClaw runtime network policies are allowlisted",
    ),
    (
        "unknown runtime network policies evidence field",
        "unknown_runtime_network_policies",
    ),
)
OPERATOR_RENDERER_REQUIRED_SOURCE_TOKENS = (
    (
        "Weave content canary gate validator function",
        "def validate_weave_content_canary_gate_option",
    ),
    ("Weave content canary gate CLI option", "--weave-content-canary-gate"),
    ("Weave content canary required CLI option", "--require-weave-content-canary"),
    ("Weave content canary pass-state validation", "ok=true and status=passed"),
    (
        "Weave content canary freshness window",
        "DEFAULT_WEAVE_CONTENT_CANARY_MAX_AGE_SECONDS",
    ),
    (
        "native Weave content canary contract helper import",
        "from weave_content_canary_gate_contract import",
    ),
    (
        "native Weave content canary contract helper call",
        "weave_content_canary_gate_contract_issues(payload)",
    ),
    (
        "Weave content canary command-policy call",
        "validate_weave_content_canary_gate_option(",
    ),
    ("Weave content canary NeMoClaw command policy", "--nemoclaw-sandbox"),
    ("Weave Agents usage command policy", "--weave-agents-require-usage"),
    (
        "command approval source packet path validator",
        "def validate_external_action_source_packet_option(",
    ),
    (
        "command approval report path validator",
        "def validate_external_action_approval_report_option(",
    ),
    (
        "command approval source packet expectation",
        "expected_external_action_source_packet_path",
    ),
    (
        "command approval report expectation",
        "expected_external_action_approval_report_path",
    ),
)
AGENTIC_RUNNER_SCRIPT_CONTRACTS = {
    "scripts/evaluator/agentic_math.py": {
        "role": "agentic_runner:math_evaluator_script",
        "tokens": (
            ("Agentic Math runner path", "run_agentic_math_openclaw.py"),
            ("Agentic Math session-prefix config lookup", 'session_prefix = _cfg_get(cfg.agentic_math, "session_prefix")'),
            ("Agentic Math session-prefix pass-through", 'command.extend(["--session-prefix", str(session_prefix)])'),
            (
                "Agentic Math W&B audit metric logging",
                "agentic_math/nemoclaw_session_audit_required_instances",
            ),
        ),
    },
    "scripts/evaluator/swebench_pro.py": {
        "role": "agentic_runner:swe_evaluator_script",
        "tokens": (
            ("SWE-Bench Pro runner path", "run_swebench_pro_openclaw.py"),
            ("SWE-Bench Pro session-prefix config lookup", 'session_prefix = _cfg_get(cfg.swebench_pro, "session_prefix")'),
            ("SWE-Bench Pro session-prefix pass-through", 'command.extend(["--session-prefix", str(session_prefix)])'),
            ("SWE-Bench Pro patch audit row loading", "def _read_patch_rows("),
            (
                "SWE-Bench Pro W&B audit metric logging",
                "agentic_swe/nemoclaw_session_audit_required_patches",
            ),
        ),
    },
    "scripts/tools/run_openclaw_agent_protocol.py": {
        "role": "agentic_runner:protocol_script",
        "tokens": (
            ("conversation-order validator", "def conversation_order_status("),
            ("conversation-order sidecar field", '"conversation_order"'),
            ("conversation-order failure exit", "Conversation order violation"),
            ("tool-before-problem issue", "tool_before_or_at_first_user_message"),
            ("tool-after-answer issue", "tool_after_final_answer"),
            ("NeMoClaw session audit validator", "def nemoclaw_session_audit_status("),
            ("NeMoClaw session audit failure exit", "NeMoClaw session audit failed"),
            ("NeMoClaw session copy requirement", "missing_nemoclaw_session_copy_status"),
        ),
    },
    "scripts/tools/run_agentic_math_openclaw.py": {
        "role": "agentic_runner:math_script",
        "tokens": (
            ("W&B session-scope resolver", "def resolve_session_prefix("),
            ("W&B run id session binding", 'os.environ.get("WANDB_RUN_ID"'),
            ("session-prefix cache binding", '"session_prefix": resolve_session_prefix(args)'),
            ("session-key resolver call", 'session_key = f"{resolve_session_prefix(args)}'),
            ("remote lookup config mutator", "def disable_remote_lookup_tools("),
            ("remote lookup config call", "disable_remote_lookup_tools(config)"),
            ("OpenClaw tool search disabled", 'tools["toolSearch"] = False'),
            ("OpenClaw web fetch disabled", 'fetch["enabled"] = False'),
            ("OpenClaw browser disabled", 'browser["enabled"] = False'),
            ("task-agent deny policy", '"deny": effective_deny_tools(args)'),
            ("conversation-order non-scoreable reason", "conversation_order_violation"),
            ("NeMoClaw session-audit non-scoreable reason", "nemoclaw_session_audit_failed"),
            ("conversation-order result field", "conversation_order_ok"),
            ("conversation-order summary counter", "conversation_order_violation_instances"),
            ("NeMoClaw session-audit result field", "nemoclaw_session_audit_ok"),
            ("NeMoClaw session-audit summary counter", "nemoclaw_session_audit_required_instances"),
        ),
    },
    "scripts/tools/run_swebench_pro_openclaw.py": {
        "role": "agentic_runner:swe_script",
        "tokens": (
            ("W&B session-scope resolver", "def resolve_session_prefix("),
            ("W&B run id session binding", 'os.environ.get("WANDB_RUN_ID"'),
            ("session-prefix cache binding", '"session_prefix": resolve_session_prefix(args)'),
            ("session-key resolver call", 'session_key = f"{resolve_session_prefix(args)}'),
            ("remote lookup config mutator", "def disable_remote_lookup_tools("),
            ("remote lookup config call", "disable_remote_lookup_tools(config)"),
            ("OpenClaw tool search disabled", 'tools["toolSearch"] = False'),
            ("OpenClaw web fetch disabled", 'fetch["enabled"] = False'),
            ("OpenClaw browser disabled", 'browser["enabled"] = False'),
            ("task-agent deny policy", '"deny": effective_deny_tools(args)'),
            ("conversation-order non-scoreable reason", "conversation_order_violation"),
            ("NeMoClaw session-audit non-scoreable reason", "nemoclaw_session_audit_failed"),
            ("conversation-order result field", "conversation_order_ok"),
            ("conversation-order summary counter", "conversation_order_violation_patches"),
            ("NeMoClaw session-audit result field", "nemoclaw_session_audit_ok"),
            ("NeMoClaw session-audit summary counter", "nemoclaw_session_audit_required_patches"),
        ),
    },
    "scripts/tools/run_taiwan_full_eval_batch.py": {
        "role": "agentic_runner:full_batch_script",
        "tokens": (
            (
                "native Weave content canary contract helper import",
                "from weave_content_canary_gate_contract import",
            ),
            (
                "native Weave content canary contract helper call",
                "weave_content_canary_gate_contract_issues(payload)",
            ),
            ("native Weave contract blocking field", "native_weave_contract_ok"),
            ("hand-edited Weave gate rejection status", "weave_gate_contract_invalid"),
            (
                "NeMoClaw batch deny tools constant",
                "REQUIRED_NEMOCLAW_AGENTIC_DENIED_TOOLS",
            ),
            (
                "NeMoClaw batch deny argument patterns constant",
                "REQUIRED_NEMOCLAW_AGENTIC_DENIED_ARGUMENT_PATTERNS",
            ),
            (
                "NeMoClaw batch local exec allow constant",
                "REQUIRED_NEMOCLAW_AGENTIC_ALLOWED_LOCAL_TOOLS",
            ),
            (
                "NeMoClaw batch local exec allow guard",
                "def require_local_exec_allowed(",
            ),
            (
                "NeMoClaw W&B verifier config expectations constant",
                "BENCHMARK_NEMOCLAW_CONFIG_EXPECTATIONS",
            ),
            ("NeMoClaw batch Math deny_tool guard", '"agentic_math.deny_tool"'),
            ("NeMoClaw batch SWE deny_tool guard", '"swebench_pro.deny_tool"'),
            (
                "NeMoClaw batch Math deny_argument_pattern guard",
                '"agentic_math.deny_argument_pattern"',
            ),
            (
                "NeMoClaw batch SWE deny_argument_pattern guard",
                '"swebench_pro.deny_argument_pattern"',
            ),
            (
                "NeMoClaw W&B Math sandbox expectation",
                '"agentic_math.nemoclaw_sandbox"',
            ),
            (
                "NeMoClaw W&B Math task-agent expectation",
                '"agentic_math.use_task_agent"',
            ),
            (
                "NeMoClaw W&B SWE sandbox expectation",
                '"swebench_pro.nemoclaw_sandbox"',
            ),
            (
                "NeMoClaw W&B SWE checkout transfer expectation",
                '"swebench_pro.nemoclaw_checkout_transfer_mode"',
            ),
            (
                "Agentic production evidence requirements constant",
                "AGENTIC_PRODUCTION_EVIDENCE_REQUIREMENTS",
            ),
            (
                "Agentic production evidence guard function",
                "def build_agentic_production_evidence_guard(",
            ),
            (
                "Agentic production evidence missing-flags status",
                "agentic_production_evidence_required",
            ),
            (
                "Agentic production evidence requires W&B completion",
                "--verify-wandb-completion",
            ),
            (
                "Agentic production evidence requires Weave Agents",
                "--verify-weave-agents",
            ),
            (
                "Agentic production evidence requires Weave content canary",
                "--require-weave-content-canary",
            ),
            (
                "Agentic production evidence requires NeMoClaw config guard",
                "--require-nemoclaw-agentic-config",
            ),
            (
                "Agentic production evidence requires tool content",
                "--weave-agents-require-tool-content",
            ),
            (
                "Agentic production evidence requires usage",
                "--weave-agents-require-usage",
            ),
            (
                "Weave request-model alias resolver",
                "def weave_expected_request_models(",
            ),
            (
                "Weave request-model verifier option",
                "--expected-request-model",
            ),
            (
                "Weave request-model verifier pass-through",
                "expected_request_models=weave_expected_request_models(",
            ),
            (
                "W&B completion requires NeMoClaw session audit",
                "--require-nemoclaw-session-audit",
            ),
            (
                "W&B completion audit requirement flag",
                "require_nemoclaw_session_audit",
            ),
        ),
    },
    "scripts/tools/verify_taiwan_weave_agents.py": {
        "role": "agentic_runner:weave_agents_verifier_script",
        "tokens": (
            (
                "Weave request-model CLI option",
                "--expected-request-model",
            ),
            (
                "Weave request-model check name",
                '"request_model"',
            ),
            (
                "Weave expected request-model evidence",
                '"expected_request_models"',
            ),
            (
                "Weave observed request-model evidence",
                "observed_request_models",
            ),
        ),
    },
    "scripts/tools/weave_content_canary_gate_contract.py": {
        "role": "agentic_runner:weave_content_canary_gate_contract_script",
        "tokens": (
            (
                "native Weave content canary contract validator",
                "def weave_content_canary_gate_contract_issues(",
            ),
            ("native Weave content canary gate name", "WEAVE_CONTENT_CANARY_GATE_NAME"),
            (
                "native Weave verifier schema contract",
                "WEAVE_AGENTS_VERIFIER_SCHEMA_VERSION = 1",
            ),
            (
                "Agents diagnostic schema contract",
                "AGENTS_DIAGNOSTIC_SCHEMA_VERSION = 1",
            ),
            ("native Weave verifier status field", '"weave_verifier_ok"'),
            ("Agents diagnostic status field", '"agents_diagnostic_ok"'),
            ("Weave verifier artifact existence proof", '"verifier_json_exists"'),
            ("Agents diagnostic artifact existence proof", '"agents_diagnostic_json_exists"'),
            ("NeMoClaw content canary proof", '"nemoclaw"'),
            ("NeMoClaw sandbox proof", '"sandbox"'),
            ("Content canary expected request-model proof", '"expected_request_models"'),
            ("Content canary observed request-model proof", '"observed_request_models"'),
            ("Content canary request-model proven flag", '"request_model_proven"'),
        ),
    },
    "scripts/tools/log_agentic_math_results_to_wandb.py": {
        "role": "agentic_runner:math_relog_script",
        "tokens": (
            ("Agentic Math relog source validation", "validate_summary(summary, rows)"),
            ("Agentic Math output table", "agentic_math_output_table"),
            ("Agentic Math NeMoClaw audit field passthrough", "nemoclaw_session_audit_ok"),
            (
                "Agentic Math relog audit metric logging",
                "agentic_math/nemoclaw_session_audit_required_instances",
            ),
            (
                "Agentic Math relog verifier audit flag",
                "--require-nemoclaw-session-audit",
            ),
        ),
    },
    "scripts/tools/log_agentic_swe_results_to_wandb.py": {
        "role": "agentic_runner:swe_relog_script",
        "tokens": (
            ("SWE relog conversation-order field", "conversation_order_ok"),
            ("SWE relog NeMoClaw audit field", "nemoclaw_session_audit_ok"),
            ("SWE output table", "agentic_swe_output_table"),
            (
                "SWE relog audit metric logging",
                "agentic_swe/nemoclaw_session_audit_required_patches",
            ),
            (
                "SWE relog verifier audit flag",
                "--require-nemoclaw-session-audit",
            ),
        ),
    },
}
EXTERNAL_ACTION_APPROVAL_PACKET_VERIFIER_SCRIPT = (
    "scripts/tools/verify_external_action_approval_packet.py"
)
EXTERNAL_ACTION_APPROVAL_TEMPLATE_RENDERER_SCRIPT = (
    "scripts/tools/render_external_action_approval_template.py"
)
EXTERNAL_ACTION_REQUIREMENTS = (
    ("requires_paid_api", "paid_api"),
    ("requires_wandb_access", "wandb_access"),
    ("requires_wandb_write", "wandb_write"),
    ("requires_third_party_acceptance", "third_party_acceptance"),
    ("requires_nemoclaw_install", "nemoclaw_install"),
    ("requires_scope_confirmation", "scope_confirmation"),
)
EXTERNAL_ACTION_REQUIREMENT_LABELS = {
    "paid_api": "Paid API",
    "wandb_access": "W&B access",
    "wandb_write": "W&B write",
    "third_party_acceptance": "Third-party acceptance",
    "nemoclaw_install": "NeMoClaw install",
    "scope_confirmation": "Scope confirmation",
}
SCOPE_ATTESTATION_SCHEMA_VERSION = 1
MIN_SCOPE_CONFIRMATION_LENGTH = 20
NEMOCLAW_AUDIT_REQUIRED_BENCHMARKS = {"agentic_math", "agentic_swe"}
SCOPE_ATTESTATION_RENDER_SAFETY_FIELDS = (
    "executes_external_action",
    "queries_wandb",
    "writes_wandb",
    "installs_third_party",
    "launches_model_inference",
    "mutates_paid_review",
)
DEPRECATED_WANDB_ADOPTION_FLAGS = (
    "--scope-confirmed-by",
    "--scope-confirmed-at",
    "--scope-confirmation",
)
REQUIRED_NEMOCLAW_POLICY_TIER = "restricted"
ALLOWED_NEMOCLAW_POLICY_TIERS = ["restricted", "balanced", "open"]
REQUIRED_NEMOCLAW_OPERATOR_DOC_CHECK_NAMES = {
    "readme_exists",
    "installer_lock_json_valid",
    "check_only_command",
    "installer_review_command",
    "install_and_onboard_command",
    "post_install_verification_command",
    "canary_readiness_command",
    "adoption_check_command",
    "production_readiness_command",
    "operator_docs_verifier_command",
    "no_openrouter_markers",
}
NEMOCLAW_INSTALLER_PROVENANCE_NOTE = (
    "Installer integrity is not verified by this script; operator review is required before install/onboard."
)
FORBIDDEN_RELEASE_OPERATOR_COMMAND_MARKERS = (
    "openrouter",
    "OPENROUTER",
    "openrouter.ai",
)
FORBIDDEN_NEMOCLAW_POST_INSTALL_COMMAND_EXACT_TOKENS = {
    "--install",
    "--onboard",
    "--upload",
    "--wandb",
    "--yes-i-accept-third-party-software",
}
FORBIDDEN_NEMOCLAW_POST_INSTALL_COMMAND_PREFIXES = (
    "--upload",
    "--wandb",
    "ANTHROPIC_API_KEY=",
    "GEMINI_API_KEY=",
    "GOOGLE_API_KEY=",
    "OPENAI_API_KEY=",
    "OPENROUTER_",
    "OPENROUTER_API_KEY=",
    "WANDB_",
    "WEAVE_",
    "XAI_API_KEY=",
)
FORBIDDEN_NEMOCLAW_POST_INSTALL_COMMAND_MARKERS = (
    "openrouter",
    "wandb",
    "weave",
)
REQUIRED_NEMOCLAW_POST_INSTALL_COMMAND_TOKENS = {
    "setup_check": ("--check-only", "--json"),
    "protocol_preflight": ("preflight",),
    "canary_readiness": (
        "--require-nemoclaw",
        "--json",
    ),
    "adoption_check": (
        "--setup-json",
        "--readiness-json",
        "--json",
        "--markdown",
    ),
}
PLACEHOLDER_ACCOUNTING_VALUES = {
    "あとで",
    "仮",
    "仮置き",
    "不明",
    "未確認",
    "未定",
    "要確認",
    "actual or billing estimate",
    "actualorbillingestimate",
    "bill or dashboard reference",
    "billordashboardreference",
    "dummy",
    "fill me",
    "fill in",
    "n a",
    "na",
    "none",
    "pending",
    "placeholder",
    "tbd",
    "todo",
    "unknown",
}
PLACEHOLDER_ACCOUNTING_PREFIXES = (
    "あとで",
    "仮",
    "不明",
    "未確認",
    "未定",
    "要確認",
    "dummy",
    "fill me",
    "fill in",
    "pending",
    "placeholder",
    "tbd",
    "todo",
    "unknown",
)


def repo_path(path: Path | str) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    return REPO_ROOT / value


def path_display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def source_path_key(path_value: Any) -> str:
    if not isinstance(path_value, str) or not path_value.strip():
        return ""
    return path_display(repo_path(path_value))


def file_record_for_source(manifest: dict[str, Any], path_value: Any) -> dict[str, Any] | None:
    source_key = source_path_key(path_value)
    if not source_key:
        return None
    files = manifest.get("files")
    if not isinstance(files, list):
        return None
    for record in files:
        if not isinstance(record, dict):
            continue
        if source_path_key(record.get("source_path")) == source_key:
            return record
    return None


def file_record_for_bundle_path(
    manifest: dict[str, Any],
    bundle_path_value: Any,
) -> dict[str, Any] | None:
    if not isinstance(bundle_path_value, str) or not bundle_path_value.strip():
        return None
    files = manifest.get("files")
    if not isinstance(files, list):
        return None
    for record in files:
        if not isinstance(record, dict):
            continue
        if record.get("bundle_path") == bundle_path_value:
            return record
    return None


def require_bundled_json(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
    path_value: Any,
    label: str,
    required_role: str,
    errors: list[str],
) -> dict[str, Any] | None:
    record = file_record_for_source(manifest, path_value)
    if not isinstance(record, dict):
        errors.append(f"{label} is not listed in manifest files")
        return None
    roles = record.get("roles")
    if not isinstance(roles, list) or required_role not in roles:
        errors.append(f"{label} missing role {required_role}")
    bundle_path = record.get("bundle_path")
    if not isinstance(bundle_path, str) or not bundle_path:
        errors.append(f"{label} missing bundle_path")
        return None
    try:
        return read_json_object(bundle_dir / bundle_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"{label} is not readable from bundle: {exc}")
        return None


def accounting_value_placeholder(value: str) -> bool:
    normalized = value.strip().casefold()
    collapsed = re.sub(r"[\s_:/\\|.,;\-]+", " ", normalized).strip()
    compact = re.sub(r"[^0-9a-zぁ-んァ-ン一-龥]+", "", normalized).strip()
    return (
        collapsed in PLACEHOLDER_ACCOUNTING_VALUES
        or compact in PLACEHOLDER_ACCOUNTING_VALUES
        or any(
            collapsed == prefix or collapsed.startswith(prefix + " ")
            for prefix in PLACEHOLDER_ACCOUNTING_PREFIXES
        )
    )


def scope_confirmed_at_valid(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    text = value.strip()
    if any(marker in text for marker in ("YYYY", "MM", "DD", "HH", "SS")):
        return False
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def scope_confirmation_valid(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if len(stripped) < MIN_SCOPE_CONFIRMATION_LENGTH:
        return False
    return not accounting_value_placeholder(stripped)


def forbidden_nemoclaw_post_install_command_tokens(command_tokens: list[str]) -> list[str]:
    forbidden: list[str] = []
    for token in command_tokens:
        token_text = str(token)
        token_casefold = token_text.casefold()
        if token_text in FORBIDDEN_NEMOCLAW_POST_INSTALL_COMMAND_EXACT_TOKENS:
            forbidden.append(token_text)
            continue
        if any(
            token_text.startswith(prefix)
            for prefix in FORBIDDEN_NEMOCLAW_POST_INSTALL_COMMAND_PREFIXES
        ):
            forbidden.append(token_text)
            continue
        if any(marker in token_casefold for marker in FORBIDDEN_NEMOCLAW_POST_INSTALL_COMMAND_MARKERS):
            forbidden.append(token_text)
    return forbidden


def validate_string_list_superset(
    *,
    errors: list[str],
    payload: dict[str, Any],
    field: str,
    required_values: tuple[str, ...] | set[str],
    label: str,
    missing_noun: str,
) -> None:
    values = payload.get(field)
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        errors.append(f"{label} {field} is not a string list")
        return
    observed = set(values)
    for required_value in sorted(required_values):
        if required_value not in observed:
            errors.append(
                f"{label} {field} missing required {missing_noun} {required_value}"
            )


def validate_required_step_tokens(
    *,
    errors: list[str],
    payload: dict[str, Any],
    field: str,
    required_steps: dict[str, tuple[str, ...]],
    label: str,
) -> None:
    step_tokens = payload.get(field)
    if not isinstance(step_tokens, dict):
        errors.append(f"{label} {field} is not an object")
        return
    for step_name, required_tokens in sorted(required_steps.items()):
        values = step_tokens.get(step_name)
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            errors.append(f"{label} {field} {step_name} is not a string list")
            continue
        observed = set(values)
        for required_token in required_tokens:
            if required_token not in observed:
                errors.append(
                    f"{label} {field} {step_name} missing required token {required_token}"
                )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def read_yaml_object(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise ValueError("PyYAML is required to read bundled YAML evidence")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a YAML object")
    return payload


def manifest_path_from_args(args: argparse.Namespace) -> Path:
    if args.manifest:
        return repo_path(args.manifest)
    return repo_path(args.bundle_dir) / "manifest.json"


def validate_manifest_schema(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != 1:
        errors.append("manifest schema_version must be 1")
    if manifest.get("bundle_version") != 2:
        errors.append("manifest bundle_version must be 2")
    return errors


def verify_file_record(bundle_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    source_path = str(record.get("source_path") or "")
    bundle_path = str(record.get("bundle_path") or "")
    result = {
        "source_path": source_path,
        "bundle_path": bundle_path,
        "ok": False,
        "errors": [],
    }
    errors: list[str] = result["errors"]
    if not bundle_path:
        errors.append("missing bundle_path")
        return result
    path = bundle_dir / bundle_path
    if not path.exists():
        errors.append("bundle file is missing")
        return result
    if not path.is_file():
        errors.append("bundle path is not a file")
        return result

    expected_size = record.get("size_bytes")
    actual_size = path.stat().st_size
    result["size_bytes"] = actual_size
    if isinstance(expected_size, int) and expected_size != actual_size:
        errors.append(f"size mismatch: expected {expected_size}, got {actual_size}")

    expected_sha256 = record.get("sha256")
    actual_sha256 = sha256_file(path)
    result["sha256"] = actual_sha256
    if not isinstance(expected_sha256, str) or not expected_sha256:
        errors.append("missing expected sha256")
    elif expected_sha256 != actual_sha256:
        errors.append("sha256 mismatch")

    result["ok"] = not errors
    return result


def validate_current_gate(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    current_gate = manifest.get("current_gate")
    if not isinstance(current_gate, dict):
        return ["manifest current_gate is not an object"]
    for field in (
        "readiness_report_source",
        "readiness_report_schema_version",
        "status",
        "readiness_status",
        "readiness_ok",
        "blocking_gates",
        "required_next_actions",
        "benchmark_completion",
        "weave_agents_completion",
        "existing_results_formalization",
        "wandb_adoption_draft",
        "paid_run_review_package",
        "wandb_completion_contract",
        "nemoclaw_adoption",
        "operator_next_steps",
        "external_action_checklist",
        "remediation_plan",
        "runner_evidence",
    ):
        if field not in current_gate:
            errors.append(f"manifest current_gate missing {field}")
    if not isinstance(current_gate.get("blocking_gates"), list):
        errors.append("manifest current_gate blocking_gates is not a list")
    else:
        current_blocking_gates = current_gate.get("blocking_gates") or []
        if current_gate.get("blocker_count") != len(current_blocking_gates):
            errors.append(
                "manifest current_gate blocker_count does not match "
                "current_gate blocking_gates length"
            )
        if current_gate.get("readiness_ok") and current_blocking_gates:
            errors.append("manifest current_gate readiness_ok is true but blocking_gates is non-empty")
    top_blocking_gates = manifest.get("blocking_gates")
    if not isinstance(top_blocking_gates, list):
        errors.append("manifest blocking_gates is not a list")
    else:
        if manifest.get("blocker_count") != len(top_blocking_gates):
            errors.append("manifest blocker_count does not match blocking_gates length")
        if manifest.get("readiness_ok") and top_blocking_gates:
            errors.append("manifest readiness_ok is true but blocking_gates is non-empty")
    for field in (
        "readiness_report_source",
        "readiness_report_schema_version",
        "status",
        "readiness_status",
        "readiness_ok",
        "gate_count",
        "blocker_count",
        "blocking_gates",
    ):
        if manifest.get(field) != current_gate.get(field):
            errors.append(f"manifest {field} does not match current_gate {field}")
    if manifest.get("readiness_report_schema_version") != 1:
        errors.append("manifest readiness_report_schema_version must be 1")
    if current_gate.get("readiness_report_schema_version") != 1:
        errors.append("manifest current_gate readiness_report_schema_version must be 1")
    gates = manifest.get("gates")
    if isinstance(gates, list):
        if manifest.get("gate_count") != len(gates):
            errors.append("manifest gate_count does not match gates length")
        if current_gate.get("gate_count") != len(gates):
            errors.append("manifest current_gate gate_count does not match gates length")
    if not isinstance(current_gate.get("required_next_actions"), list):
        errors.append("manifest current_gate required_next_actions is not a list")
    if not isinstance(current_gate.get("benchmark_completion"), list):
        errors.append("manifest current_gate benchmark_completion is not a list")
    if not isinstance(current_gate.get("weave_agents_completion"), list):
        errors.append("manifest current_gate weave_agents_completion is not a list")
    if not isinstance(current_gate.get("existing_results_formalization"), dict):
        errors.append("manifest current_gate existing_results_formalization is not an object")
    if not isinstance(current_gate.get("wandb_adoption_draft"), dict):
        errors.append("manifest current_gate wandb_adoption_draft is not an object")
    if not isinstance(current_gate.get("paid_run_review_package"), dict):
        errors.append("manifest current_gate paid_run_review_package is not an object")
    if not isinstance(current_gate.get("wandb_completion_contract"), dict):
        errors.append("manifest current_gate wandb_completion_contract is not an object")
    if "benchmark_progress_matrix" in current_gate and not isinstance(
        current_gate.get("benchmark_progress_matrix"),
        list,
    ):
        errors.append("manifest current_gate benchmark_progress_matrix is not a list")
    if not isinstance(current_gate.get("nemoclaw_adoption"), dict):
        errors.append("manifest current_gate nemoclaw_adoption is not an object")
    if "nemoclaw_installer_review" in current_gate and not isinstance(
        current_gate.get("nemoclaw_installer_review"),
        dict,
    ):
        errors.append("manifest current_gate nemoclaw_installer_review is not an object")
    current_gate_installer_lock_json = None
    if isinstance(current_gate.get("nemoclaw_installer_review"), dict):
        lock_value = current_gate["nemoclaw_installer_review"].get("lock_json")
        if isinstance(lock_value, str) and lock_value.strip():
            current_gate_installer_lock_json = lock_value
    if not isinstance(current_gate.get("operator_next_steps"), dict):
        errors.append("manifest current_gate operator_next_steps is not an object")
    else:
        expected_checklist = expected_external_action_checklist(
            operator_steps=current_gate["operator_next_steps"],
            blocking_gates=current_gate.get("blocking_gates")
            if isinstance(current_gate.get("blocking_gates"), list)
            else [],
        )
        if current_gate.get("external_action_checklist") != expected_checklist:
            errors.append(
                "manifest current_gate external_action_checklist does not match operator_next_steps"
            )
    remediation_plan = current_gate.get("remediation_plan")
    if not isinstance(remediation_plan, list):
        errors.append("manifest current_gate remediation_plan is not a list")
    else:
        current_gate_has_nemoclaw_install = False
        current_gate_has_nemoclaw_installer_review = False
        current_gate_nemoclaw_review_outputs: set[str] = set()
        current_gate_nemoclaw_install_review_jsons: list[str] = []
        for row_index, row in enumerate(remediation_plan, start=1):
            if not isinstance(row, dict):
                errors.append(f"manifest current_gate remediation_plan row {row_index} is not an object")
                continue
            commands = row.get("commands")
            if commands is None:
                continue
            if not isinstance(commands, list):
                errors.append(
                    f"manifest current_gate remediation_plan row {row_index} commands is not a list"
                )
                continue
            for command_index, command in enumerate(commands, start=1):
                if not isinstance(command, str):
                    errors.append(
                        "manifest current_gate remediation_plan "
                        f"row {row_index} command {command_index} is not a string"
                    )
                    continue
                forbidden_markers = [
                    marker
                    for marker in FORBIDDEN_RELEASE_OPERATOR_COMMAND_MARKERS
                    if marker in command
                ]
                if forbidden_markers:
                    errors.append(
                        "manifest current_gate remediation_plan "
                        f"row {row_index} command {command_index} uses forbidden "
                        "release operator marker(s): "
                        + ", ".join(sorted(set(forbidden_markers)))
                        + "; use the approved OpenAI-direct canary path"
                    )
                try:
                    parts = shlex.split(command)
                except ValueError:
                    parts = command.split()
                validate_taiwan_full_batch_external_approval_command(
                    parts=parts,
                    label=(
                        "manifest current_gate remediation_plan "
                        f"row {row_index} command {command_index}"
                    ),
                    errors=errors,
                )
                validate_weave_content_canary_external_approval_command(
                    parts=parts,
                    label=(
                        "manifest current_gate remediation_plan "
                        f"row {row_index} command {command_index}"
                    ),
                    errors=errors,
                )
                validate_wandb_completion_sync_apply_command(
                    parts=parts,
                    label=(
                        "manifest current_gate remediation_plan "
                        f"row {row_index} command {command_index}"
                    ),
                    errors=errors,
                )
                validate_weave_agents_completion_sync_apply_command(
                    parts=parts,
                    label=(
                        "manifest current_gate remediation_plan "
                        f"row {row_index} command {command_index}"
                    ),
                    errors=errors,
                )
                if command_installs_nemoclaw(parts):
                    current_gate_has_nemoclaw_install = True
                if command_invokes_nemoclaw_installer_review(parts):
                    current_gate_has_nemoclaw_installer_review = True
                    review_json = validate_nemoclaw_installer_review_command(
                        parts=parts,
                        label=(
                            "manifest current_gate remediation_plan "
                            f"row {row_index} command {command_index}"
                        ),
                        errors=errors,
                    )
                    if review_json:
                        current_gate_nemoclaw_review_outputs.add(review_json)
                validate_nemoclaw_installer_sha_command(
                    parts=parts,
                    label=(
                        "manifest current_gate remediation_plan "
                        f"row {row_index} command {command_index}"
                    ),
                    errors=errors,
                )
                validate_nemoclaw_installer_lock_command(
                    parts=parts,
                    label=(
                        "manifest current_gate remediation_plan "
                        f"row {row_index} command {command_index}"
                    ),
                    errors=errors,
                    expected_lock_json=current_gate_installer_lock_json,
                )
                install_review_json = validate_nemoclaw_installer_review_json_command(
                    parts=parts,
                    label=(
                        "manifest current_gate remediation_plan "
                        f"row {row_index} command {command_index}"
                    ),
                    errors=errors,
                )
                if install_review_json:
                    current_gate_nemoclaw_install_review_jsons.append(install_review_json)
        if current_gate_has_nemoclaw_install and not current_gate_has_nemoclaw_installer_review:
            errors.append(
                "manifest current_gate remediation_plan installs NeMoClaw but has no "
                "review_nemoclaw_installer.py command"
            )
        for review_json in current_gate_nemoclaw_install_review_jsons:
            if review_json not in current_gate_nemoclaw_review_outputs:
                errors.append(
                    "manifest current_gate remediation_plan installs NeMoClaw with "
                    f"--installer-review-json {review_json} but no matching "
                    "review_nemoclaw_installer.py --json output"
                )
    if not isinstance(current_gate.get("runner_evidence"), dict):
        errors.append("manifest current_gate runner_evidence is not an object")
    return errors


def validate_production_readiness_report_evidence(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    files = manifest.get("files")
    file_records = files if isinstance(files, list) else []
    records = [
        record
        for record in file_records
        if isinstance(record, dict)
        and "production_readiness_report" in (record.get("roles") or [])
    ]
    if not records:
        return ["production readiness report is not listed in manifest files"]
    expected_source_key = source_path_key(manifest.get("readiness_report_source"))
    source_matched = False
    for record in records:
        if source_path_key(record.get("source_path")) == expected_source_key:
            source_matched = True
        bundle_path = record.get("bundle_path")
        if not isinstance(bundle_path, str) or not bundle_path:
            errors.append("production readiness report bundle_path is missing")
            continue
        try:
            payload = read_json_object(bundle_dir / bundle_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"production readiness report is not readable: {exc}")
            continue
        if payload.get("schema_version") != 1:
            errors.append("production readiness report schema_version must be 1")
        if payload.get("status") != manifest.get("readiness_status"):
            errors.append("production readiness report status does not match manifest")
        if bool(payload.get("ok")) != bool(manifest.get("readiness_ok")):
            errors.append("production readiness report ok does not match manifest")
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        for field in ("gate_count", "blocker_count"):
            if summary.get(field) != manifest.get(field):
                errors.append(f"production readiness report summary.{field} does not match manifest")
        blockers = summary.get("blockers")
        if isinstance(blockers, list) and blockers != manifest.get("blocking_gates"):
            errors.append("production readiness report summary.blockers does not match manifest")
        errors.extend(
            validate_one_model_canary_weave_agents_contract(
                payload=payload,
                manifest=manifest,
            )
        )
    if expected_source_key and not source_matched:
        errors.append("production readiness report source_path does not match manifest")
    return errors


def readiness_gate(payload: dict[str, Any], name: str) -> dict[str, Any] | None:
    gates = payload.get("gates")
    if not isinstance(gates, list):
        return None
    for gate in gates:
        if isinstance(gate, dict) and gate.get("name") == name:
            return gate
    return None


def manifest_required_wandb_benchmarks(manifest: dict[str, Any]) -> list[str]:
    current_gate = manifest.get("current_gate")
    if not isinstance(current_gate, dict):
        return []
    contract = current_gate.get("wandb_completion_contract")
    if not isinstance(contract, dict):
        return []
    values = contract.get("required_benchmarks")
    if not isinstance(values, list):
        return []
    return [value for value in values if isinstance(value, str) and value.strip()]


def one_model_weave_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    current_gate = manifest.get("current_gate")
    if not isinstance(current_gate, dict):
        return []
    rows = current_gate.get("weave_agents_completion")
    if not isinstance(rows, list):
        return []
    return [
        row
        for row in rows
        if isinstance(row, dict) and row.get("gate") == "one_model_full_canary"
    ]


def phase_has_completed_weave_entries(completed_phases: dict[str, Any], phase: str) -> bool:
    entries = completed_phases.get(phase)
    return isinstance(entries, list) and bool(entries)


def validate_one_model_canary_weave_agents_contract(
    *,
    payload: dict[str, Any],
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    gate = readiness_gate(payload, "one_model_full_canary")
    if not isinstance(gate, dict):
        return errors
    if gate.get("blocking") is False:
        return errors

    required_benchmarks = gate.get("required_wandb_benchmarks")
    if not isinstance(required_benchmarks, list):
        required_benchmarks = manifest_required_wandb_benchmarks(manifest)
    required_benchmark_names = [
        item for item in required_benchmarks if isinstance(item, str) and item.strip()
    ]
    agentic_required = bool(
        set(required_benchmark_names) & AGENTIC_REQUIRED_WANDB_BENCHMARKS
    )
    if not agentic_required:
        return errors

    label = "one_model_full_canary"
    if gate.get("weave_agents_completion_required") is not True:
        errors.append(
            f"{label} weave_agents_completion_required must be true when "
            "agentic W&B benchmarks are required"
        )

    completed_phases = gate.get("completed_weave_agents_completion_phases")
    if not isinstance(completed_phases, dict):
        errors.append(f"{label} completed_weave_agents_completion_phases must be an object")
        completed_phases = {}
    else:
        for phase in completed_phases:
            if phase not in WEAVE_AGENTS_CANARY_COMPLETION_PHASES:
                errors.append(
                    f"{label} completed_weave_agents_completion_phases has invalid phase {phase}"
                )

    missing_phases = gate.get("missing_required_weave_agents_completion_phases")
    if not isinstance(missing_phases, list) or not all(
        isinstance(item, str) and item.strip() for item in missing_phases
    ):
        errors.append(
            f"{label} missing_required_weave_agents_completion_phases must be a string list"
        )
        missing_phases = []

    phase_status = gate.get("phase_status")
    full_completed = (
        isinstance(phase_status, dict)
        and phase_status.get("full") == "completed"
    )
    phased_completed = (
        isinstance(phase_status, dict)
        and phase_status.get("nonagentic") == "completed"
        and phase_status.get("agentic") == "completed"
        and phase_status.get("agentic_aggregate") == "completed"
    )

    if gate.get("ok") is True or gate.get("status") == "passed":
        has_weave_phase = phase_has_completed_weave_entries(
            completed_phases,
            "full",
        ) or phase_has_completed_weave_entries(completed_phases, "agentic")
        if not has_weave_phase:
            errors.append(
                f"{label} passed but no completed full/agentic Weave Agents phase is recorded"
            )
        proven_rows = [
            row
            for row in one_model_weave_rows(manifest)
            if row.get("phase") in WEAVE_AGENTS_CANARY_COMPLETION_PHASES
            and row.get("required") is True
            and row.get("completion_proven") is True
            and isinstance(row.get("verified_count"), int)
            and row.get("verified_count") > 0
        ]
        if not proven_rows:
            errors.append(
                f"{label} passed but manifest current_gate.weave_agents_completion "
                "has no proven full/agentic row"
            )
    elif (
        gate.get("status") == "weave_agents_completion_not_proven"
        and not missing_phases
    ):
        errors.append(
            f"{label} weave_agents_completion_not_proven must list missing "
            "required Weave Agents phases"
        )

    if full_completed and not phase_has_completed_weave_entries(completed_phases, "full"):
        if "full" not in missing_phases and gate.get("status") != "incomplete":
            errors.append(f"{label} completed full phase is missing Weave Agents proof")
    if phased_completed and not phase_has_completed_weave_entries(completed_phases, "agentic"):
        if "agentic" not in missing_phases and gate.get("status") != "incomplete":
            errors.append(f"{label} completed agentic phase is missing Weave Agents proof")

    for row in one_model_weave_rows(manifest):
        phase = row.get("phase")
        if phase not in WEAVE_AGENTS_CANARY_COMPLETION_PHASES:
            continue
        if row.get("completion_proven") is True and not phase_has_completed_weave_entries(
            completed_phases,
            str(phase),
        ):
            errors.append(
                f"{label} manifest reports proven Weave Agents completion for {phase} "
                "but production readiness report does not"
            )
    return errors


def validate_operator_plan(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    operator_plan = manifest.get("operator_plan")
    if not isinstance(operator_plan, dict):
        return ["manifest operator_plan is not an object"]
    for field in ("json", "markdown", "schema_version", "status"):
        if field not in operator_plan:
            errors.append(f"manifest operator_plan missing {field}")
    for field in ("json", "markdown"):
        if not isinstance(operator_plan.get(field), str) or not operator_plan.get(field):
            errors.append(f"manifest operator_plan {field} is not a non-empty string")

    files = manifest.get("files")
    file_records = files if isinstance(files, list) else []
    bundle_paths = {
        record.get("bundle_path")
        for record in file_records
        if isinstance(record, dict)
    }
    for field in ("json", "markdown"):
        value = operator_plan.get(field)
        if isinstance(value, str) and value and value not in bundle_paths:
            errors.append(f"manifest operator_plan {field} is not listed in files")
    return errors


def validate_release_gate_pointer_summary(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    pointer = manifest.get("release_gate_pointer")
    if pointer is None:
        return []
    if not isinstance(pointer, dict):
        return ["manifest release_gate_pointer is not an object"]

    errors: list[str] = []
    for field in (
        "release_gate_json",
        "latest_pointer_json",
        "latest_pointer_verification_json",
    ):
        value = pointer.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(
                f"manifest release_gate_pointer {field} is not a non-empty string"
            )

    release_gate_timestamp = None
    release_gate_json = pointer.get("release_gate_json")
    release_gate_payload: dict[str, Any] | None = None
    if isinstance(release_gate_json, str) and release_gate_json.strip():
        match = RELEASE_GATE_JSON_NAME_RE.match(Path(release_gate_json).name)
        if match:
            release_gate_timestamp = match.group(1)
            release_gate_payload = require_bundled_json(
                bundle_dir=bundle_dir,
                manifest=manifest,
                path_value=release_gate_json,
                label="release gate pointer release gate JSON",
                required_role="release_gate_pointer:release_gate_json",
                errors=errors,
            )
        else:
            errors.append(
                "manifest release_gate_pointer release_gate_json must point to "
                "taiwan_release_gate_YYYYMMDDTHHMMSSZ.json"
            )

    latest_pointer_json = pointer.get("latest_pointer_json")
    latest_pointer_payload: dict[str, Any] | None = None
    if isinstance(latest_pointer_json, str) and latest_pointer_json.strip():
        if Path(latest_pointer_json).name != LATEST_POINTER_JSON_NAME:
            errors.append(
                "manifest release_gate_pointer latest_pointer_json must point to "
                f"{LATEST_POINTER_JSON_NAME}"
            )
        else:
            latest_pointer_payload = require_bundled_json(
                bundle_dir=bundle_dir,
                manifest=manifest,
                path_value=latest_pointer_json,
                label="release gate pointer latest pointer JSON",
                required_role="release_gate_pointer:latest_pointer_json",
                errors=errors,
            )

    verification_timestamp = None
    verification_json = pointer.get("latest_pointer_verification_json")
    verification_payload: dict[str, Any] | None = None
    if isinstance(verification_json, str) and verification_json.strip():
        match = LATEST_POINTER_VERIFICATION_JSON_NAME_RE.match(Path(verification_json).name)
        if match:
            verification_timestamp = match.group(1)
            verification_payload = require_bundled_json(
                bundle_dir=bundle_dir,
                manifest=manifest,
                path_value=verification_json,
                label="release gate pointer latest pointer verification JSON",
                required_role="release_gate_pointer:latest_pointer_verification_json",
                errors=errors,
            )
        else:
            errors.append(
                "manifest release_gate_pointer latest_pointer_verification_json must point to "
                "latest_taiwan_release_gate_verify_YYYYMMDDTHHMMSSZ.json"
            )

    if (
        release_gate_timestamp
        and verification_timestamp
        and release_gate_timestamp != verification_timestamp
    ):
        errors.append(
            "manifest release_gate_pointer release gate timestamp does not match "
            "latest pointer verification timestamp"
        )
    if pointer.get("latest_pointer_verification_ok") is not True:
        errors.append("manifest release_gate_pointer latest_pointer_verification_ok must be true")
    if pointer.get("latest_pointer_verification_status") != "passed":
        errors.append("manifest release_gate_pointer latest_pointer_verification_status must be passed")
    if pointer.get("latest_pointer_verification_issue_count") != 0:
        errors.append("manifest release_gate_pointer latest_pointer_verification_issue_count must be 0")
    operator_plan_summary = pointer.get("operator_plan")
    if not isinstance(operator_plan_summary, dict):
        errors.append("manifest release_gate_pointer operator_plan is not an object")
    else:
        operator_plan_json = operator_plan_summary.get("json")
        operator_plan_markdown = operator_plan_summary.get("markdown")
        for field, value in (
            ("json", operator_plan_json),
            ("markdown", operator_plan_markdown),
        ):
            if not isinstance(value, str) or not value.strip():
                errors.append(
                    f"manifest release_gate_pointer operator_plan {field} is not a non-empty string"
                )
        if operator_plan_summary.get("schema_version") != 1:
            errors.append(
                "manifest release_gate_pointer operator_plan schema_version must be 1"
            )
        if (
            not isinstance(operator_plan_summary.get("status"), str)
            or not operator_plan_summary.get("status")
        ):
            errors.append("manifest release_gate_pointer operator_plan status is required")
        if isinstance(operator_plan_json, str) and operator_plan_json.strip():
            match = STANDALONE_OPERATOR_PLAN_JSON_NAME_RE.match(
                Path(operator_plan_json).name
            )
            if not match:
                errors.append(
                    "manifest release_gate_pointer operator_plan json must point to "
                    "taiwan_release_operator_plan_YYYYMMDDTHHMMSSZ.json"
                )
            elif release_gate_timestamp and match.group(1) != release_gate_timestamp:
                errors.append(
                    "manifest release_gate_pointer operator_plan json timestamp does not "
                    "match release gate timestamp"
                )
        if isinstance(operator_plan_markdown, str) and operator_plan_markdown.strip():
            match = STANDALONE_OPERATOR_PLAN_MARKDOWN_NAME_RE.match(
                Path(operator_plan_markdown).name
            )
            if not match:
                errors.append(
                    "manifest release_gate_pointer operator_plan markdown must point to "
                    "taiwan_release_operator_plan_YYYYMMDDTHHMMSSZ.md"
                )
            elif release_gate_timestamp and match.group(1) != release_gate_timestamp:
                errors.append(
                    "manifest release_gate_pointer operator_plan markdown timestamp does not "
                    "match release gate timestamp"
                )
    if release_gate_payload is not None:
        if release_gate_payload.get("schema_version") != 1:
            errors.append("release gate pointer release gate JSON schema_version must be 1")
        if release_gate_payload.get("timestamp") != release_gate_timestamp:
            errors.append("release gate pointer release gate JSON timestamp mismatch")
        if source_path_key(release_gate_payload.get("release_gate_json")) != source_path_key(
            release_gate_json
        ):
            errors.append(
                "release gate pointer release gate JSON release_gate_json does not match "
                "manifest release_gate_pointer"
            )
        if source_path_key(release_gate_payload.get("latest_pointer_json")) != source_path_key(
            latest_pointer_json
        ):
            errors.append(
                "release gate pointer release gate JSON latest_pointer_json does not match "
                "manifest release_gate_pointer"
            )
        if source_path_key(
            release_gate_payload.get("latest_pointer_verification_json")
        ) != source_path_key(verification_json):
            errors.append(
                "release gate pointer release gate JSON latest_pointer_verification_json "
                "does not match manifest release_gate_pointer"
            )
        if isinstance(operator_plan_summary, dict) and release_gate_payload.get(
            "operator_plan"
        ) != operator_plan_summary:
            errors.append(
                "release gate pointer release gate JSON operator_plan does not match "
                "manifest release_gate_pointer"
            )
    if latest_pointer_payload is not None:
        if latest_pointer_payload.get("schema_version") != 1:
            errors.append("release gate pointer latest pointer JSON schema_version must be 1")
        if latest_pointer_payload.get("timestamp") != release_gate_timestamp:
            errors.append("release gate pointer latest pointer JSON timestamp mismatch")
        if source_path_key(latest_pointer_payload.get("release_gate_json")) != source_path_key(
            release_gate_json
        ):
            errors.append(
                "release gate pointer latest pointer JSON release_gate_json does not match "
                "manifest release_gate_pointer"
            )
        if source_path_key(
            latest_pointer_payload.get("latest_pointer_json")
        ) != source_path_key(latest_pointer_json):
            errors.append(
                "release gate pointer latest pointer JSON latest_pointer_json does not match "
                "manifest release_gate_pointer"
            )
        if source_path_key(
            latest_pointer_payload.get("latest_pointer_verification_json")
        ) != source_path_key(verification_json):
            errors.append(
                "release gate pointer latest pointer JSON latest_pointer_verification_json "
                "does not match manifest release_gate_pointer"
            )
        if latest_pointer_payload.get("latest_pointer_verification_ok") != pointer.get(
            "latest_pointer_verification_ok"
        ):
            errors.append(
                "release gate pointer latest pointer JSON latest_pointer_verification_ok "
                "does not match manifest release_gate_pointer"
            )
        if latest_pointer_payload.get("latest_pointer_verification_status") != pointer.get(
            "latest_pointer_verification_status"
        ):
            errors.append(
                "release gate pointer latest pointer JSON latest_pointer_verification_status "
                "does not match manifest release_gate_pointer"
            )
        if latest_pointer_payload.get("latest_pointer_verification_issue_count") != pointer.get(
            "latest_pointer_verification_issue_count"
        ):
            errors.append(
                "release gate pointer latest pointer JSON latest_pointer_verification_issue_count "
                "does not match manifest release_gate_pointer"
            )
        if isinstance(operator_plan_summary, dict) and latest_pointer_payload.get(
            "operator_plan"
        ) != operator_plan_summary:
            errors.append(
                "release gate pointer latest pointer JSON operator_plan does not match "
                "manifest release_gate_pointer"
            )
    if verification_payload is not None:
        if verification_payload.get("schema_version") != 1:
            errors.append("latest pointer verification JSON schema_version must be 1")
        if verification_payload.get("ok") is not True:
            errors.append("latest pointer verification JSON ok must be true")
        if verification_payload.get("status") != "passed":
            errors.append("latest pointer verification JSON status must be passed")
        if verification_payload.get("issue_count") != 0:
            errors.append("latest pointer verification JSON issue_count must be 0")
        if verification_payload.get("issues") not in ([], None):
            errors.append("latest pointer verification JSON issues must be empty")
        if source_path_key(verification_payload.get("release_gate_json")) != source_path_key(
            release_gate_json
        ):
            errors.append(
                "latest pointer verification JSON release_gate_json does not match "
                "manifest release_gate_pointer"
            )
        if source_path_key(verification_payload.get("pointer_json")) != source_path_key(
            latest_pointer_json
        ):
            errors.append(
                "latest pointer verification JSON pointer_json does not match "
                "manifest release_gate_pointer"
            )
    return errors


def validate_release_gate_pointer_proof(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    if manifest.get("release_gate_pointer") is None:
        return []
    errors: list[str] = []
    proof_ref = manifest.get("release_gate_pointer_proof")
    if not isinstance(proof_ref, dict):
        return ["manifest release_gate_pointer_proof is not an object"]

    proof_json = proof_ref.get("json")
    if not isinstance(proof_json, str) or not proof_json.strip():
        return ["manifest release_gate_pointer_proof json is not a non-empty string"]
    if Path(proof_json).name != "release_gate_pointer_proof.json":
        errors.append(
            "manifest release_gate_pointer_proof json must be release_gate_pointer_proof.json"
        )
    if proof_ref.get("schema_version") != 1:
        errors.append("manifest release_gate_pointer_proof schema_version must be 1")
    if proof_ref.get("status") != "passed":
        errors.append("manifest release_gate_pointer_proof status must be passed")

    record = file_record_for_bundle_path(manifest, proof_json)
    if not isinstance(record, dict):
        errors.append("manifest release_gate_pointer_proof json is not listed in files")
        return errors
    roles = record.get("roles")
    if not isinstance(roles, list) or "release_gate_pointer:proof_json" not in roles:
        errors.append(
            "manifest release_gate_pointer_proof json missing role "
            "release_gate_pointer:proof_json"
        )

    try:
        proof = read_json_object(bundle_dir / proof_json)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"release_gate_pointer_proof JSON is not readable from bundle: {exc}")
        return errors

    if proof.get("schema_version") != 1:
        errors.append("release_gate_pointer_proof JSON schema_version must be 1")
    if proof.get("kind") != "release_gate_pointer_proof":
        errors.append("release_gate_pointer_proof JSON kind must be release_gate_pointer_proof")
    if proof.get("status") != "passed":
        errors.append("release_gate_pointer_proof JSON status must be passed")
    if proof.get("ok") is not True:
        errors.append("release_gate_pointer_proof JSON ok must be true")
    if proof.get("release_gate_pointer") != manifest.get("release_gate_pointer"):
        errors.append("release_gate_pointer_proof JSON release_gate_pointer mismatch")

    latest_pointer_verification = proof.get("latest_pointer_verification")
    if not isinstance(latest_pointer_verification, dict):
        errors.append("release_gate_pointer_proof JSON latest_pointer_verification is not an object")
    else:
        if latest_pointer_verification.get("ok") is not True:
            errors.append("release_gate_pointer_proof JSON latest_pointer_verification ok must be true")
        if latest_pointer_verification.get("status") != "passed":
            errors.append(
                "release_gate_pointer_proof JSON latest_pointer_verification status must be passed"
            )
        if latest_pointer_verification.get("issue_count") != 0:
            errors.append(
                "release_gate_pointer_proof JSON latest_pointer_verification issue_count must be 0"
            )
        pointer = manifest.get("release_gate_pointer")
        if isinstance(pointer, dict):
            if source_path_key(latest_pointer_verification.get("json")) != source_path_key(
                pointer.get("latest_pointer_verification_json")
            ):
                errors.append(
                    "release_gate_pointer_proof JSON latest_pointer_verification json "
                    "does not match manifest release_gate_pointer"
                )

    return errors


def validate_operator_plan_payload(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    operator_plan = manifest.get("operator_plan")
    if not isinstance(operator_plan, dict):
        return []
    json_path_value = operator_plan.get("json")
    if not isinstance(json_path_value, str) or not json_path_value:
        return []

    plan_path = bundle_dir / json_path_value
    try:
        payload = read_json_object(plan_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return [f"operator_plan json is not readable: {exc}"]

    if payload.get("schema_version") != operator_plan.get("schema_version"):
        errors.append("operator_plan schema_version does not match manifest")
    if payload.get("status") != operator_plan.get("status"):
        errors.append("operator_plan status does not match manifest")

    payload_steps = payload.get("operator_next_steps")
    if not isinstance(payload_steps, dict):
        errors.append("operator_plan operator_next_steps is not an object")
    current_gate = manifest.get("current_gate")
    current_steps = (
        current_gate.get("operator_next_steps")
        if isinstance(current_gate, dict)
        else None
    )
    if isinstance(payload_steps, dict) and isinstance(current_steps, dict):
        if payload_steps != current_steps:
            errors.append("operator_plan operator_next_steps does not match manifest current_gate")
        expected_checklist = expected_external_action_checklist(
            operator_steps=payload_steps,
            blocking_gates=manifest.get("blocking_gates")
            if isinstance(manifest.get("blocking_gates"), list)
            else [],
        )
        if payload.get("external_action_checklist") != expected_checklist:
            errors.append(
                "operator_plan external_action_checklist does not match operator_next_steps"
            )
    if isinstance(current_gate, dict):
        if payload.get("timestamp") != manifest.get("timestamp"):
            errors.append("operator_plan timestamp does not match manifest")
        if payload.get("release_gate_status") != current_gate.get("status"):
            errors.append("operator_plan release_gate_status does not match manifest current_gate")
        if payload.get("readiness_status") != manifest.get("readiness_status"):
            errors.append("operator_plan readiness_status does not match manifest")
        if bool(payload.get("readiness_ok")) != bool(manifest.get("readiness_ok")):
            errors.append("operator_plan readiness_ok does not match manifest")
        if payload.get("readiness_report_source") != manifest.get("readiness_report_source"):
            errors.append("operator_plan readiness_report_source does not match manifest")
        if payload.get("blocking_gates") != manifest.get("blocking_gates"):
            errors.append("operator_plan blocking_gates does not match manifest")
        for field in (
            "wandb_completion_contract",
            "benchmark_progress_matrix",
            "wandb_adoption_draft",
            "paid_run_review_package",
            "nemoclaw_adoption",
        ):
            if field in current_gate and payload.get(field) != current_gate.get(field):
                errors.append(f"operator_plan {field} does not match manifest current_gate")
    release_gate_pointer = manifest.get("release_gate_pointer")
    if release_gate_pointer is not None:
        if not isinstance(release_gate_pointer, dict):
            errors.append("manifest release_gate_pointer is not an object")
        elif payload.get("release_gate_pointer") != release_gate_pointer:
            errors.append("operator_plan release_gate_pointer does not match manifest")
        elif isinstance(release_gate_pointer.get("release_gate_json"), str):
            expected_release_gate_json = release_gate_pointer.get("release_gate_json")
            if expected_release_gate_json:
                for field in ("source_release_gate_json", "release_gate_json"):
                    if source_path_key(payload.get(field)) != source_path_key(
                        expected_release_gate_json
                    ):
                        errors.append(
                            f"operator_plan {field} does not match manifest release_gate_pointer"
                        )

    markdown_text: str | None = None

    outputs = payload.get("outputs")
    if not isinstance(outputs, dict):
        errors.append("operator_plan outputs is not an object")
    else:
        if outputs.get("json") != operator_plan.get("json"):
            errors.append("operator_plan outputs.json does not match manifest")
        if outputs.get("markdown") != operator_plan.get("markdown"):
            errors.append("operator_plan outputs.markdown does not match manifest")
        if outputs.get("manifest") != "manifest.json":
            errors.append("operator_plan outputs.manifest is not manifest.json")
        if outputs.get("summary") != "summary.md":
            errors.append("operator_plan outputs.summary is not summary.md")
        if isinstance(release_gate_pointer, dict):
            for output_field, pointer_field in (
                ("release_gate_json", "release_gate_json"),
                ("latest_pointer_json", "latest_pointer_json"),
                (
                    "latest_pointer_verification_json",
                    "latest_pointer_verification_json",
                ),
            ):
                if outputs.get(output_field) != release_gate_pointer.get(pointer_field):
                    errors.append(
                        f"operator_plan outputs.{output_field} does not match manifest release_gate_pointer"
                    )
        markdown_path_value = outputs.get("markdown")
        if isinstance(markdown_path_value, str) and markdown_path_value:
            try:
                markdown_text = (bundle_dir / markdown_path_value).read_text(
                    encoding="utf-8"
                )
            except OSError as exc:
                errors.append(f"operator_plan markdown is not readable: {exc}")
            else:
                for snippet in (
                    "## External Action Checklist",
                    "External action items",
                    "| Gate | Status | Required external actions | Commands | Evidence paths |",
                ):
                    if snippet not in markdown_text:
                        errors.append(
                            f"operator_plan markdown missing external action checklist content: {snippet}"
                        )
    errors.extend(
        validate_operator_execution_plan_renderer(
            bundle_dir=bundle_dir,
            manifest=manifest,
            operator_payload=payload,
            operator_plan_json=operator_plan.get("json"),
            markdown_text=markdown_text,
        )
    )
    return errors


def validate_operator_execution_plan_renderer(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
    operator_payload: dict[str, Any],
    operator_plan_json: Any,
    markdown_text: str | None,
) -> list[str]:
    errors: list[str] = []
    renderer = operator_payload.get("operator_execution_plan_renderer")
    if not isinstance(renderer, dict):
        return ["operator_plan operator_execution_plan_renderer is not an object"]
    if renderer.get("schema_version") != 1:
        errors.append("operator_plan renderer schema_version must be 1")
    if renderer.get("status") != "available":
        errors.append("operator_plan renderer status must be available")
    if renderer.get("script") != OPERATOR_EXECUTION_PLAN_RENDERER_SCRIPT:
        errors.append("operator_plan renderer script mismatch")
    if renderer.get("required_before_external_action") is not True:
        errors.append("operator_plan renderer must be required before external action")
    safety = renderer.get("safety")
    if not isinstance(safety, dict):
        errors.append("operator_plan renderer safety is not an object")
    else:
        if safety.get("executes_external_action") is not False:
            errors.append("operator_plan renderer safety executes_external_action must be false")
        if safety.get("writes_shell_script_only_when_placeholders_resolved") is not True:
            errors.append(
                "operator_plan renderer safety must require resolved placeholders for shell script"
            )
        if safety.get("requires_valid_external_action_approval_for_shell_script") is not True:
            errors.append(
                "operator_plan renderer safety must require valid external-action approval for shell script"
            )
        if safety.get("requires_source_packet_match_for_shell_script") is not True:
            errors.append(
                "operator_plan renderer safety must require source packet match for shell script"
            )
        if (
            safety.get("requires_command_approval_paths_match_for_shell_script")
            is not True
        ):
            errors.append(
                "operator_plan renderer safety must require command approval path "
                "match for shell script"
            )
        if safety.get("requires_release_gate_match_for_shell_script") is not True:
            errors.append(
                "operator_plan renderer safety must require release gate match for shell script"
            )
        if safety.get("requires_command_policy_validation_for_shell_script") is not True:
            errors.append(
                "operator_plan renderer safety must require command policy validation for shell script"
            )
        if (
            safety.get("requires_weave_content_canary_gate_validation_for_shell_script")
            is not True
        ):
            errors.append(
                "operator_plan renderer safety must require Weave content canary gate validation for shell script"
            )
    operator_steps = operator_payload.get("operator_next_steps")
    expected_tokens = []
    if isinstance(operator_steps, dict):
        tokens = operator_steps.get("unresolved_placeholder_tokens")
        if isinstance(tokens, list):
            expected_tokens = [token for token in tokens if isinstance(token, str)]
    if renderer.get("placeholder_tokens") != expected_tokens:
        errors.append("operator_plan renderer placeholder_tokens do not match operator_next_steps")
    outputs = renderer.get("expected_outputs")
    if not isinstance(outputs, dict):
        errors.append("operator_plan renderer expected_outputs is not an object")
    else:
        for key in ("json", "markdown", "shell_script"):
            value = outputs.get(key)
            if not isinstance(value, str) or not value:
                errors.append(f"operator_plan renderer expected_outputs.{key} is missing")
    approval_report_template = renderer.get("approval_report_json_template")
    if not isinstance(approval_report_template, str) or not approval_report_template:
        errors.append("operator_plan renderer approval_report_json_template is missing")
    elif not approval_report_template.endswith(".verify.json"):
        errors.append("operator_plan renderer approval_report_json_template must end with .verify.json")
    approval_source_packet_template = renderer.get("approval_source_packet_json_template")
    if not isinstance(approval_source_packet_template, str) or not approval_source_packet_template:
        errors.append("operator_plan renderer approval_source_packet_json_template is missing")
    elif not approval_source_packet_template.endswith("external_action_approval_packet.json"):
        errors.append(
            "operator_plan renderer approval_source_packet_json_template must point to external_action_approval_packet.json"
        )
    release_gate_pointer = manifest.get("release_gate_pointer")
    expected_release_gate_json = (
        release_gate_pointer.get("release_gate_json")
        if isinstance(release_gate_pointer, dict)
        and isinstance(release_gate_pointer.get("release_gate_json"), str)
        else ""
    )
    release_gate_template = renderer.get("release_gate_json_template")
    if expected_release_gate_json:
        if not isinstance(release_gate_template, str) or not release_gate_template:
            errors.append("operator_plan renderer release_gate_json_template is missing")
        elif source_path_key(release_gate_template) != source_path_key(
            expected_release_gate_json
        ):
            errors.append(
                "operator_plan renderer release_gate_json_template does not match manifest release_gate_pointer"
            )
    review_command = renderer.get("review_command_template")
    ready_command = renderer.get("require_ready_command_template")
    for label, command in (
        ("review", review_command),
        ("require_ready", ready_command),
    ):
        if not isinstance(command, str) or not command.strip():
            errors.append(f"operator_plan renderer {label} command is missing")
            continue
        for required in (
            OPERATOR_EXECUTION_PLAN_RENDERER_SCRIPT,
            "--operator-plan-json",
            str(operator_plan_json),
            "--timestamp",
            "--output-json",
            "--markdown",
        ):
            if required not in command:
                errors.append(
                    f"operator_plan renderer {label} command missing {required}"
                )
        if expected_release_gate_json:
            for required in (
                "--release-gate-json",
                str(release_gate_template),
            ):
                if required not in command:
                    errors.append(
                        f"operator_plan renderer {label} command missing {required}"
                    )
        if label == "require_ready":
            for required in (
                "--external-action-approval-source-packet-json",
                str(approval_source_packet_template),
                "--external-action-approval-report-json",
                str(approval_report_template),
                "--shell-script",
                "--require-ready",
            ):
                if required not in command:
                    errors.append(
                        f"operator_plan renderer require_ready command missing {required}"
                    )
    records = file_records_by_source(manifest)
    record = validate_file_role(
        errors=errors,
        records=records,
        path_value=OPERATOR_EXECUTION_PLAN_RENDERER_SCRIPT,
        role="operator_execution_plan_renderer:script",
        label="operator execution plan renderer script",
    )
    if isinstance(record, dict):
        bundle_path = record.get("bundle_path")
        if not isinstance(bundle_path, str) or not bundle_path:
            errors.append("operator execution plan renderer script missing bundle_path")
        else:
            errors.extend(
                validate_operator_renderer_script_source(
                    bundle_dir=bundle_dir,
                    bundle_path=bundle_path,
                )
            )
    validate_file_role(
        errors=errors,
        records=records,
        path_value=WEAVE_CONTENT_CANARY_GATE_CONTRACT_SCRIPT,
        role="operator_execution_plan_renderer:dependency_script",
        label="operator execution plan renderer dependency script",
    )
    if isinstance(markdown_text, str):
        for snippet in (
            "## Operator Execution Plan Renderer",
            OPERATOR_EXECUTION_PLAN_RENDERER_SCRIPT,
            "Placeholder-ready shell",
            "External approval source packet",
        ):
            if snippet not in markdown_text:
                errors.append(f"operator_plan markdown missing renderer content: {snippet}")
    return errors


def validate_operator_renderer_script_source(
    *,
    bundle_dir: Path,
    bundle_path: str,
) -> list[str]:
    errors: list[str] = []
    script_path = bundle_dir / bundle_path
    try:
        text = script_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"operator execution plan renderer script is not readable: {exc}"]
    for label, token in OPERATOR_RENDERER_REQUIRED_SOURCE_TOKENS:
        if token not in text:
            errors.append(
                "operator execution plan renderer script missing source contract "
                f"{label}: {token}"
            )
    return errors


def validate_external_action_approval_packet_payload(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    current_gate = manifest.get("current_gate")
    checklist = (
        current_gate.get("external_action_checklist")
        if isinstance(current_gate, dict)
        and isinstance(current_gate.get("external_action_checklist"), dict)
        else None
    )
    if checklist is None:
        return ["manifest current_gate external_action_checklist is not an object"]

    packet_ref = manifest.get("external_action_approval_packet")
    if not isinstance(packet_ref, dict):
        return ["manifest external_action_approval_packet is not an object"]
    json_path_value = packet_ref.get("json")
    markdown_path_value = packet_ref.get("markdown")
    if not isinstance(json_path_value, str) or not json_path_value:
        errors.append("external_action_approval_packet json is missing")
        return errors
    if not isinstance(markdown_path_value, str) or not markdown_path_value:
        errors.append("external_action_approval_packet markdown is missing")

    packet_path = bundle_dir / json_path_value
    try:
        packet = read_json_object(packet_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return [f"external_action_approval_packet json is not readable: {exc}"]

    expected_hash = canonical_json_sha256(checklist)
    external_item_count = int(checklist.get("external_action_item_count") or 0)
    expected_status = (
        "pending_approval" if external_item_count else "no_external_action_required"
    )
    expected_requirements = expected_approval_requirements(checklist)
    expected_required_count = sum(1 for item in expected_requirements if item["required"])

    if packet.get("schema_version") != 1:
        errors.append("external_action_approval_packet schema_version must be 1")
    if packet_ref.get("schema_version") != packet.get("schema_version"):
        errors.append("external_action_approval_packet schema_version does not match manifest")
    if packet.get("status") != expected_status:
        errors.append("external_action_approval_packet status does not match checklist")
    if packet_ref.get("status") != packet.get("status"):
        errors.append("external_action_approval_packet status does not match manifest")
    if packet.get("readiness_status") != manifest.get("readiness_status"):
        errors.append("external_action_approval_packet readiness_status does not match manifest")
    if bool(packet.get("readiness_ok")) != bool(manifest.get("readiness_ok")):
        errors.append("external_action_approval_packet readiness_ok does not match manifest")
    if packet.get("blocking_gates") != manifest.get("blocking_gates"):
        errors.append("external_action_approval_packet blocking_gates does not match manifest")
    if packet.get("external_action_checklist") != checklist:
        errors.append("external_action_approval_packet checklist does not match current_gate")
    if packet.get("external_action_checklist_sha256") != expected_hash:
        errors.append("external_action_approval_packet checklist sha256 mismatch")
    if packet_ref.get("external_action_checklist_sha256") != expected_hash:
        errors.append("external_action_approval_packet manifest checklist sha256 mismatch")
    if packet.get("approval_requirements") != expected_requirements:
        errors.append("external_action_approval_packet approval_requirements do not match checklist")
    if packet.get("approval_requirement_count") != len(expected_requirements):
        errors.append("external_action_approval_packet approval_requirement_count mismatch")
    if packet.get("required_approval_count") != expected_required_count:
        errors.append("external_action_approval_packet required_approval_count mismatch")
    if packet_ref.get("required_approval_count") != expected_required_count:
        errors.append("external_action_approval_packet manifest required_approval_count mismatch")
    expected_all_granted = expected_required_count == 0
    if packet.get("all_required_approvals_granted") is not expected_all_granted:
        errors.append("external_action_approval_packet all_required_approvals_granted mismatch")
    if packet_ref.get("all_required_approvals_granted") is not expected_all_granted:
        errors.append("external_action_approval_packet manifest all_required_approvals_granted mismatch")
    verifier = packet.get("approval_verifier")
    manifest_verifier = packet_ref.get("approval_verifier")
    if not isinstance(verifier, dict):
        errors.append("external_action_approval_packet approval_verifier is not an object")
    else:
        if manifest_verifier != verifier:
            errors.append("external_action_approval_packet approval_verifier does not match manifest")
        if verifier.get("schema_version") != 1:
            errors.append("external_action_approval_packet approval_verifier schema_version must be 1")
        if verifier.get("status") != "available":
            errors.append("external_action_approval_packet approval_verifier status must be available")
        if verifier.get("script") != EXTERNAL_ACTION_APPROVAL_PACKET_VERIFIER_SCRIPT:
            errors.append("external_action_approval_packet approval_verifier script mismatch")
        if verifier.get("required_before_external_action") is not (external_item_count > 0):
            errors.append(
                "external_action_approval_packet approval_verifier required_before_external_action mismatch"
            )
        for key in (
            "source_packet_json",
            "reviewed_packet_json_template",
            "report_json_template",
            "command_template",
        ):
            value = verifier.get(key)
            if not isinstance(value, str) or not value:
                errors.append(f"external_action_approval_packet approval_verifier {key} is missing")
        if verifier.get("source_packet_json") != json_path_value:
            errors.append(
                "external_action_approval_packet approval_verifier source_packet_json mismatch"
            )
        command = verifier.get("command_template")
        if isinstance(command, str):
            for required in (
                EXTERNAL_ACTION_APPROVAL_PACKET_VERIFIER_SCRIPT,
                "--approval-packet-json",
                "--source-packet-json",
                "--require-approved",
                "--json",
            ):
                if required not in command:
                    errors.append(
                        "external_action_approval_packet approval_verifier command "
                        f"missing {required}"
                    )
        safety = verifier.get("safety")
        if not isinstance(safety, dict):
            errors.append("external_action_approval_packet approval_verifier safety is not an object")
        else:
            for field in (
                "executes_external_action",
                "queries_wandb",
                "writes_wandb",
                "installs_third_party",
                "launches_model_inference",
            ):
                if safety.get(field) is not False:
                    errors.append(
                        "external_action_approval_packet approval_verifier safety "
                        f"{field} must be false"
                    )

    renderer = packet.get("approval_template_renderer")
    manifest_renderer = packet_ref.get("approval_template_renderer")
    if not isinstance(renderer, dict):
        errors.append("external_action_approval_packet approval_template_renderer is not an object")
    else:
        if manifest_renderer != renderer:
            errors.append(
                "external_action_approval_packet approval_template_renderer does not match manifest"
            )
        if renderer.get("schema_version") != 1:
            errors.append(
                "external_action_approval_packet approval_template_renderer schema_version must be 1"
            )
        if renderer.get("status") != "available":
            errors.append(
                "external_action_approval_packet approval_template_renderer status must be available"
            )
        if renderer.get("script") != EXTERNAL_ACTION_APPROVAL_TEMPLATE_RENDERER_SCRIPT:
            errors.append(
                "external_action_approval_packet approval_template_renderer script mismatch"
            )
        if renderer.get("required_before_external_action") is not (external_item_count > 0):
            errors.append(
                "external_action_approval_packet approval_template_renderer "
                "required_before_external_action mismatch"
            )
        for key in (
            "reviewed_packet_json_template",
            "reviewed_packet_markdown_template",
            "command_template",
        ):
            value = renderer.get(key)
            if not isinstance(value, str) or not value:
                errors.append(
                    f"external_action_approval_packet approval_template_renderer {key} is missing"
                )
        command = renderer.get("command_template")
        if isinstance(command, str):
            for required in (
                EXTERNAL_ACTION_APPROVAL_TEMPLATE_RENDERER_SCRIPT,
                "--approval-packet-json",
                "--output-json",
                "--markdown",
            ):
                if required not in command:
                    errors.append(
                        "external_action_approval_packet approval_template_renderer command "
                        f"missing {required}"
                    )
        safety = renderer.get("safety")
        if not isinstance(safety, dict):
            errors.append(
                "external_action_approval_packet approval_template_renderer safety is not an object"
            )
        else:
            for field in (
                "executes_external_action",
                "queries_wandb",
                "writes_wandb",
                "installs_third_party",
                "launches_model_inference",
            ):
                if safety.get(field) is not False:
                    errors.append(
                        "external_action_approval_packet approval_template_renderer safety "
                        f"{field} must be false"
                    )

    source = packet.get("source")
    if not isinstance(source, dict):
        errors.append("external_action_approval_packet source is not an object")
    else:
        operator_plan = manifest.get("operator_plan") if isinstance(manifest.get("operator_plan"), dict) else {}
        if source.get("manifest") != "manifest.json":
            errors.append("external_action_approval_packet source.manifest must be manifest.json")
        if source.get("operator_plan_json") != operator_plan.get("json"):
            errors.append("external_action_approval_packet source.operator_plan_json mismatch")
        if source.get("operator_plan_markdown") != operator_plan.get("markdown"):
            errors.append("external_action_approval_packet source.operator_plan_markdown mismatch")
        if source.get("release_gate_pointer") != (manifest.get("release_gate_pointer") or {}):
            errors.append("external_action_approval_packet release_gate_pointer mismatch")

    policy = packet.get("execution_policy")
    if not isinstance(policy, dict):
        errors.append("external_action_approval_packet execution_policy is not an object")
    else:
        counts = checklist.get("requirement_counts")
        requirement_counts = counts if isinstance(counts, dict) else {}
        expected_policy = {
            "this_packet_launches_external_actions": False,
            "requires_human_review_before_paid_api": requirement_counts.get("paid_api", 0) > 0,
            "requires_human_review_before_wandb_write": requirement_counts.get("wandb_write", 0) > 0,
            "requires_human_review_before_third_party_install": (
                requirement_counts.get("third_party_acceptance", 0) > 0
                or requirement_counts.get("nemoclaw_install", 0) > 0
            ),
            "requires_scope_attestation_before_adopting_existing_results": (
                requirement_counts.get("scope_confirmation", 0) > 0
            ),
        }
        if policy != expected_policy:
            errors.append("external_action_approval_packet execution_policy mismatch")

    outputs = packet.get("outputs")
    if not isinstance(outputs, dict):
        errors.append("external_action_approval_packet outputs is not an object")
    else:
        if outputs.get("json") != packet_ref.get("json"):
            errors.append("external_action_approval_packet outputs.json does not match manifest")
        if outputs.get("markdown") != packet_ref.get("markdown"):
            errors.append("external_action_approval_packet outputs.markdown does not match manifest")

    if isinstance(markdown_path_value, str) and markdown_path_value:
        try:
            markdown_text = (bundle_dir / markdown_path_value).read_text(
                encoding="utf-8"
            )
        except OSError as exc:
            errors.append(f"external_action_approval_packet markdown is not readable: {exc}")
        else:
            for snippet in (
                "# Taiwan External Action Approval Packet",
                "## Approval Requirements",
                "## Approval Template Renderer",
                "## Approval Verifier",
                "## Execution Policy",
                "## External Action Checklist",
                str(expected_hash),
                EXTERNAL_ACTION_APPROVAL_TEMPLATE_RENDERER_SCRIPT,
                EXTERNAL_ACTION_APPROVAL_PACKET_VERIFIER_SCRIPT,
            ):
                if snippet not in markdown_text:
                    errors.append(
                        "external_action_approval_packet markdown missing content: "
                        f"{snippet}"
                    )

    file_records = manifest.get("files") if isinstance(manifest.get("files"), list) else []
    roles_by_bundle_path = {
        record.get("bundle_path"): record.get("roles")
        for record in file_records
        if isinstance(record, dict)
    }
    json_roles = roles_by_bundle_path.get(json_path_value)
    if not isinstance(json_roles, list) or "external_action_approval_packet_json" not in json_roles:
        errors.append("external_action_approval_packet json file record role is missing")
    if isinstance(markdown_path_value, str) and markdown_path_value:
        md_roles = roles_by_bundle_path.get(markdown_path_value)
        if not isinstance(md_roles, list) or "external_action_approval_packet_markdown" not in md_roles:
            errors.append("external_action_approval_packet markdown file record role is missing")
    records = file_records_by_source(manifest)
    record = validate_file_role(
        errors=errors,
        records=records,
        path_value=EXTERNAL_ACTION_APPROVAL_PACKET_VERIFIER_SCRIPT,
        role="external_action_approval_packet_verifier:script",
        label="external action approval packet verifier script",
    )
    if isinstance(record, dict) and not record.get("bundle_path"):
        errors.append("external action approval packet verifier script missing bundle_path")
    record = validate_file_role(
        errors=errors,
        records=records,
        path_value=EXTERNAL_ACTION_APPROVAL_TEMPLATE_RENDERER_SCRIPT,
        role="external_action_approval_template_renderer:script",
        label="external action approval template renderer script",
    )
    if isinstance(record, dict) and not record.get("bundle_path"):
        errors.append(
            "external action approval template renderer script missing bundle_path"
        )
    return errors


def validate_summary_markdown(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    files = manifest.get("files")
    file_records = files if isinstance(files, list) else []
    summary_record = next(
        (
            record
            for record in file_records
            if isinstance(record, dict) and record.get("bundle_path") == "summary.md"
        ),
        None,
    )
    if not isinstance(summary_record, dict):
        return ["summary.md is not listed in manifest files"]
    roles = summary_record.get("roles")
    if not isinstance(roles, list) or "release_summary_markdown" not in roles:
        errors.append("summary.md missing role release_summary_markdown")
    try:
        summary_text = (bundle_dir / "summary.md").read_text(encoding="utf-8")
    except OSError as exc:
        return [f"summary.md is not readable: {exc}"]

    for snippet in (
        "# Taiwan Release Evidence Bundle",
        "## Current Gate",
        "## Required Next Actions",
        "## Operator Next Steps",
        "## External Action Checklist",
        "## External Action Approval Packet",
        "## Benchmark W&B Completion",
        "## W&B Completion Contract",
        "## Existing W&B Adoption Draft",
        "## Existing W&B Adoption Unconfirmed-Template Checks",
        "## Weave Agents Completion",
        "## Paid Run Review Package",
        "## NeMoClaw Adoption",
        "## Gates",
        "## Files",
    ):
        if snippet not in summary_text:
            errors.append(f"summary.md missing required section: {snippet}")

    for snippet in (
        "External action items",
        "| Gate | Status | Required external actions | Commands | Evidence paths |",
    ):
        if snippet not in summary_text:
            errors.append(
                f"summary.md missing external action checklist content: {snippet}"
            )

    approval_packet = manifest.get("external_action_approval_packet")
    if isinstance(approval_packet, dict):
        for snippet in (
            "External Action Approval Packet",
            str(approval_packet.get("json") or ""),
            str(approval_packet.get("markdown") or ""),
            str(approval_packet.get("external_action_checklist_sha256") or ""),
            EXTERNAL_ACTION_APPROVAL_TEMPLATE_RENDERER_SCRIPT,
            EXTERNAL_ACTION_APPROVAL_PACKET_VERIFIER_SCRIPT,
        ):
            if snippet and snippet not in summary_text:
                errors.append(
                    f"summary.md missing external action approval packet content: {snippet}"
                )
    else:
        errors.append("manifest external_action_approval_packet is not an object")

    current_gate = manifest.get("current_gate")
    external_action_checklist = (
        current_gate.get("external_action_checklist")
        if isinstance(current_gate, dict)
        else None
    )
    if isinstance(external_action_checklist, dict):
        for snippet in expected_external_action_summary_snippets(
            external_action_checklist
        ):
            if snippet not in summary_text:
                errors.append(
                    "summary.md missing external action checklist content "
                    f"from current_gate: {snippet}"
                )
    elif isinstance(current_gate, dict):
        errors.append("summary.md cannot be checked because current_gate external_action_checklist is not an object")

    completion_contract = (
        current_gate.get("wandb_completion_contract")
        if isinstance(current_gate, dict)
        else None
    )
    if isinstance(completion_contract, dict):
        for snippet in expected_wandb_completion_contract_summary_snippets(
            completion_contract
        ):
            if snippet not in summary_text:
                errors.append(
                    "summary.md missing W&B completion contract content "
                    f"from current_gate: {snippet}"
                )
    elif isinstance(current_gate, dict):
        errors.append(
            "summary.md cannot be checked because current_gate wandb_completion_contract is not an object"
        )

    release_gate_pointer = manifest.get("release_gate_pointer")
    if isinstance(release_gate_pointer, dict):
        for snippet in (
            "## Release Gate Pointer",
            "Release gate JSON",
            "Latest pointer JSON",
            "Pointer verification JSON",
            str(release_gate_pointer.get("latest_pointer_verification_json") or ""),
        ):
            if snippet and snippet not in summary_text:
                errors.append(f"summary.md missing release pointer content: {snippet}")
    elif release_gate_pointer is not None:
        errors.append("manifest release_gate_pointer is not an object")

    runner = (
        current_gate.get("runner_evidence")
        if isinstance(current_gate, dict)
        else None
    )
    post_install = (
        runner.get("nemoclaw_post_install_verification")
        if isinstance(runner, dict)
        else None
    )
    operator_docs = (
        runner.get("nemoclaw_operator_docs_verification")
        if isinstance(runner, dict)
        else None
    )
    adoption_draft = (
        current_gate.get("wandb_adoption_draft")
        if isinstance(current_gate, dict)
        else None
    )
    if isinstance(adoption_draft, dict) and not adoption_draft.get("skipped"):
        for snippet in expected_wandb_adoption_draft_summary_snippets(
            adoption_draft
        ):
            if snippet not in summary_text:
                errors.append(
                    "summary.md missing W&B adoption draft content "
                    f"from current_gate: {snippet}"
                )
    if isinstance(operator_docs, dict) and operator_docs:
        for snippet in (
            "## NeMoClaw Operator Docs Verification",
            "Missing requirements",
        ):
            if snippet not in summary_text:
                errors.append(f"summary.md missing NeMoClaw operator docs content: {snippet}")
        path_value = operator_docs.get("path")
        if isinstance(path_value, str) and path_value and path_value not in summary_text:
            errors.append("summary.md missing NeMoClaw operator docs JSON path")
    if isinstance(post_install, dict) and post_install:
        for snippet in (
            "## NeMoClaw Post-Install Verification",
            "Will launch model inference",
            "Will query W&B",
            "Will install/onboard",
            "Command safety OK",
            "Forbidden token count",
            "Forbidden exact tokens",
            "Forbidden prefixes",
            "Forbidden markers",
            "Missing required token count",
            "Missing command count",
            "### NeMoClaw Post-Install Command Safety",
            "### NeMoClaw Post-Install Steps",
            "setup_check",
            "protocol_preflight",
            "canary_readiness",
            "adoption_check",
        ):
            if snippet not in summary_text:
                errors.append(f"summary.md missing NeMoClaw post-install content: {snippet}")
        path_value = post_install.get("path")
        if isinstance(path_value, str) and path_value and path_value not in summary_text:
            errors.append("summary.md missing NeMoClaw post-install JSON path")

    unconfirmed_checks = (
        runner.get("wandb_adoption_unconfirmed_checks")
        if isinstance(runner, dict)
        else None
    )
    if isinstance(unconfirmed_checks, dict) and not unconfirmed_checks.get("skipped"):
        for snippet in expected_wandb_adoption_unconfirmed_summary_snippets(
            unconfirmed_checks
        ):
            if snippet not in summary_text:
                errors.append(
                    "summary.md missing W&B adoption unconfirmed-template "
                    f"content from current_gate: {snippet}"
                )
    return errors


def command_script_paths(command: str) -> list[str]:
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    scripts: list[str] = []
    seen: set[str] = set()
    for part in parts:
        token = part.strip().strip("`'\".,;:()[]{}")
        if token.startswith("./scripts/"):
            token = token[2:]
        if not token.startswith("scripts/"):
            continue
        if not token.endswith(COMMAND_SCRIPT_SUFFIXES):
            continue
        if token in seen:
            continue
        seen.add(token)
        scripts.append(token)
    return scripts


def collect_operator_plan_command_scripts(payload: dict[str, Any]) -> list[str]:
    operator_steps = payload.get("operator_next_steps")
    if not isinstance(operator_steps, dict):
        return []
    steps = operator_steps.get("steps")
    if not isinstance(steps, list):
        return []
    scripts: list[str] = []
    seen: set[str] = set()
    for step in steps:
        if not isinstance(step, dict):
            continue
        commands = step.get("commands")
        if not isinstance(commands, list):
            continue
        for command in commands:
            if not isinstance(command, str) or not command.strip():
                continue
            for script in command_script_paths(command):
                if script in seen:
                    continue
                seen.add(script)
                scripts.append(script)
    return scripts


def collect_operator_plan_commands(payload: dict[str, Any]) -> list[str]:
    operator_steps = payload.get("operator_next_steps")
    if not isinstance(operator_steps, dict):
        return []
    steps = operator_steps.get("steps")
    if not isinstance(steps, list):
        return []
    commands: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        values = step.get("commands")
        if not isinstance(values, list):
            continue
        for command in values:
            if isinstance(command, str) and command.strip():
                commands.append(command)
    return commands


def collect_current_gate_remediation_command_scripts(manifest: dict[str, Any]) -> list[str]:
    current_gate = manifest.get("current_gate")
    if not isinstance(current_gate, dict):
        return []
    remediation_plan = current_gate.get("remediation_plan")
    if not isinstance(remediation_plan, list):
        return []
    scripts: list[str] = []
    seen: set[str] = set()
    for row in remediation_plan:
        if not isinstance(row, dict):
            continue
        commands = row.get("commands")
        if not isinstance(commands, list):
            continue
        for command in commands:
            if not isinstance(command, str) or not command.strip():
                continue
            for script in command_script_paths(command):
                if script in seen:
                    continue
                seen.add(script)
                scripts.append(script)
    return scripts


def collect_existing_results_relog_command_scripts(manifest: dict[str, Any]) -> list[str]:
    current_gate = manifest.get("current_gate")
    if not isinstance(current_gate, dict):
        return []
    existing = current_gate.get("existing_results_formalization")
    if not isinstance(existing, dict):
        return []
    scripts: list[str] = []
    seen: set[str] = set()
    for field in ("formalized_records", "unformalized_complete_records", "partial_records"):
        group = existing.get(field)
        if not isinstance(group, list):
            continue
        for record in group:
            if not isinstance(record, dict):
                continue
            for command_field in ("relog_dry_run_command", "relog_command"):
                command = record.get(command_field)
                if not isinstance(command, str) or not command.strip():
                    continue
                for script in command_script_paths(command):
                    if script in seen:
                        continue
                    seen.add(script)
                    scripts.append(script)
    return scripts


def unique_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def markdown_cell(value: Any) -> str:
    if value is None:
        text = ""
    elif isinstance(value, bool):
        text = str(value)
    elif isinstance(value, (list, tuple)):
        text = ", ".join(str(item) for item in value)
    else:
        text = str(value)
    return text.replace("\n", "<br>").replace("|", "\\|")


def adoption_draft_markdown_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    return text.replace("\n", " ").replace("|", "\\|")


def expected_external_action_checklist(
    *,
    operator_steps: dict[str, Any],
    blocking_gates: list[Any],
) -> dict[str, Any]:
    steps = operator_steps.get("steps")
    if not isinstance(steps, list):
        steps = []
    blocking_gate_names = [item for item in blocking_gates if isinstance(item, str)]
    requirement_counts = {name: 0 for _, name in EXTERNAL_ACTION_REQUIREMENTS}
    items: list[dict[str, Any]] = []
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        requirements = [
            name
            for field, name in EXTERNAL_ACTION_REQUIREMENTS
            if step.get(field) is True
        ]
        for name in requirements:
            requirement_counts[name] += 1
        commands = string_list(step.get("commands"))
        evidence_to_produce = string_list(step.get("evidence_to_produce"))
        warnings = string_list(step.get("warnings"))
        gate = step.get("gate")
        items.append(
            {
                "order": step.get("order") if isinstance(step.get("order"), int) else index,
                "gate": gate,
                "status": step.get("status"),
                "blocking": isinstance(gate, str) and gate in blocking_gate_names,
                "external_action_required": bool(requirements),
                "requirements": requirements,
                "command_count": len(commands),
                "evidence_path_count": len(evidence_to_produce),
                "next_action": step.get("next_action"),
                "commands": commands,
                "evidence_to_produce": evidence_to_produce,
                "warnings": warnings,
            }
        )
    return {
        "schema_version": 1,
        "status": operator_steps.get("status", "unknown"),
        "blocking_gate_count": len(blocking_gate_names),
        "item_count": len(items),
        "external_action_item_count": sum(
            1 for item in items if item.get("external_action_required")
        ),
        "requirement_counts": requirement_counts,
        "items": items,
    }


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def expected_approval_requirement(
    *,
    requirement: str,
    count: int,
    checklist: dict[str, Any],
) -> dict[str, Any]:
    gates = [
        item.get("gate")
        for item in checklist.get("items", [])
        if isinstance(item, dict)
        and requirement in (item.get("requirements") if isinstance(item.get("requirements"), list) else [])
    ]
    reviewer_fields = ["approved_by", "approved_at", "approval_reference"]
    if requirement == "paid_api":
        reviewer_fields.extend(["approved_budget_usd", "approved_model_scope"])
    elif requirement == "wandb_write":
        reviewer_fields.extend(["approved_wandb_entity", "approved_wandb_project"])
    elif requirement == "third_party_acceptance":
        reviewer_fields.append("third_party_terms_reviewed")
    elif requirement == "nemoclaw_install":
        reviewer_fields.extend(["installer_lock_json", "installer_sha256", "sandbox"])
    elif requirement == "scope_confirmation":
        reviewer_fields.append("scope_attestation_json")
    return {
        "requirement": requirement,
        "label": EXTERNAL_ACTION_REQUIREMENT_LABELS[requirement],
        "count": count,
        "required": count > 0,
        "approval_status": "not_granted" if count > 0 else "not_required",
        "required_before_gates": gates,
        "reviewer_fields": reviewer_fields if count > 0 else [],
    }


def expected_approval_requirements(checklist: dict[str, Any]) -> list[dict[str, Any]]:
    counts = checklist.get("requirement_counts")
    requirement_counts = counts if isinstance(counts, dict) else {}
    return [
        expected_approval_requirement(
            requirement=name,
            count=int(requirement_counts.get(name) or 0),
            checklist=checklist,
        )
        for _, name in EXTERNAL_ACTION_REQUIREMENTS
    ]


def expected_external_action_summary_snippets(checklist: dict[str, Any]) -> list[str]:
    snippets = [
        f"Status: `{checklist.get('status', 'unknown')}`",
        f"External action items: `{checklist.get('external_action_item_count', 0)}`",
    ]
    items = checklist.get("items")
    if not isinstance(items, list) or not items:
        snippets.append("| none |  | none | 0 | 0 |")
        return snippets
    for item in items:
        if not isinstance(item, dict):
            continue
        requirements = item.get("requirements")
        requirement_text = "none"
        if isinstance(requirements, list) and requirements:
            requirement_text = ", ".join(
                EXTERNAL_ACTION_REQUIREMENT_LABELS.get(str(name), str(name))
                for name in requirements
            )
        gate = item.get("gate") or "unknown"
        status = item.get("status") or "unknown"
        snippets.append(
            f"| `{markdown_cell(gate)}` | `{markdown_cell(status)}` | "
            f"{markdown_cell(requirement_text)} | "
            f"{markdown_cell(item.get('command_count'))} | "
            f"{markdown_cell(item.get('evidence_path_count'))} |"
        )
    return snippets


def expected_wandb_adoption_unconfirmed_summary_snippets(
    checks: dict[str, Any],
) -> list[str]:
    snippets = [
        "| Field | Value |",
        f"| Status | {markdown_cell(checks.get('status'))} |",
        f"| OK | {markdown_cell(checks.get('ok'))} |",
        f"| Record count | {markdown_cell(checks.get('record_count'))} |",
        f"| Output dir | `{markdown_cell(checks.get('output_dir'))}` |",
        (
            "| Benchmark | Run ID | Attestation template | Preflight report | "
            "Preflight failed as expected | Sync dry-run report | "
            "Sync dry-run failed as expected | Review mutation | W&B write | "
            "Model inference |"
        ),
    ]
    records = checks.get("records")
    if not isinstance(records, list) or not records:
        snippets.append("| none |  |  |  |  |  |  |  |  |  |")
        return snippets
    for record in records:
        if not isinstance(record, dict):
            continue
        snippets.append(
            "| "
            f"{markdown_cell(record.get('benchmark'))} | "
            f"{markdown_cell(record.get('run_id'))} | "
            f"{markdown_cell(record.get('scope_attestation_template_json'))} | "
            f"{markdown_cell(record.get('preflight_report_json'))} | "
            f"{markdown_cell(record.get('preflight_failed_as_expected'))} | "
            f"{markdown_cell(record.get('sync_dry_run_report_json'))} | "
            f"{markdown_cell(record.get('sync_dry_run_failed_as_expected'))} | "
            f"{markdown_cell(record.get('will_mutate_review_json'))} | "
            f"{markdown_cell(record.get('will_write_wandb'))} | "
            f"{markdown_cell(record.get('will_launch_model_inference'))} |"
        )
    return snippets


def expected_wandb_adoption_draft_summary_snippets(
    draft: dict[str, Any],
) -> list[str]:
    snippets = [
        "| Field | Value |",
        f"| Status | {markdown_cell(draft.get('status'))} |",
        f"| OK | {markdown_cell(draft.get('ok'))} |",
        f"| JSON | `{markdown_cell(draft.get('path'))}` |",
        f"| Markdown | `{markdown_cell(draft.get('markdown_path'))}` |",
        f"| Candidate count | {markdown_cell(draft.get('candidate_count'))} |",
        f"| Source audit JSON | `{markdown_cell(draft.get('source_audit_json'))}` |",
        f"| Source audit SHA-256 | {markdown_cell(draft.get('source_audit_sha256'))} |",
        (
            "| Attestation template count | "
            f"{markdown_cell(draft.get('scope_attestation_template_count'))} |"
        ),
        (
            "| Requires scope confirmation | "
            f"{markdown_cell(draft.get('requires_human_scope_confirmation'))} |"
        ),
        (
            "| Required human fields | "
            f"{markdown_cell(draft.get('required_human_fields'))} |"
        ),
        (
            "| Pending scope-confirmation candidates | "
            f"{markdown_cell(draft.get('pending_scope_confirmation_candidate_count'))} |"
        ),
        (
            "| Pending human field count | "
            f"{markdown_cell(draft.get('pending_human_field_count'))} |"
        ),
        (
            "| Pending human fields | "
            f"{markdown_cell(draft.get('pending_human_fields'))} |"
        ),
        (
            "| Handoff candidate count | "
            f"{markdown_cell(draft.get('operator_handoff_candidate_count'))} |"
        ),
        (
            "| Handoff available candidates | "
            f"{markdown_cell(draft.get('operator_handoff_available_candidate_count'))} |"
        ),
        (
            "| Handoff steps | "
            f"{markdown_cell(draft.get('operator_handoff_step_count'))} |"
        ),
        (
            "| Handoff commands | "
            f"{markdown_cell(draft.get('operator_handoff_command_count'))} |"
        ),
        (
            "| Handoff external-action steps | "
            f"{markdown_cell(draft.get('operator_handoff_external_action_step_count'))} |"
        ),
        (
            "| Handoff scope-confirmation steps | "
            f"{markdown_cell(draft.get('operator_handoff_scope_confirmation_step_count'))} |"
        ),
        (
            "| Handoff review-mutation steps | "
            f"{markdown_cell(draft.get('operator_handoff_review_mutation_step_count'))} |"
        ),
        (
            "| Handoff evidence paths | "
            f"{markdown_cell(draft.get('operator_handoff_evidence_path_count'))} |"
        ),
        (
            "| Handoff expected evidence paths | "
            f"{markdown_cell((draft.get('operator_handoff') or {}).get('expected_evidence_paths'))} |"
        ),
        (
            "| Benchmark | Model | Run ID | Source audit JSON | Source audit SHA-256 | "
            "Target review | Completion JSON | Attestation template | Render report | Preflight report | "
            "Dry-run report | Scope required | Handoff available | Handoff steps | Handoff commands | "
            "Handoff evidence paths | Handoff expected evidence paths | Render command | Preflight command | "
            "Dry-run command | Apply command |"
        ),
    ]
    candidates = draft.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        snippets.append("| none |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |")
        candidates = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        handoff = (
            candidate.get("operator_handoff")
            if isinstance(candidate.get("operator_handoff"), dict)
            else {}
        )
        snippets.append(
            "| "
            f"{markdown_cell(candidate.get('benchmark'))} | "
            f"{markdown_cell(candidate.get('model_slug'))} | "
            f"{markdown_cell(candidate.get('wandb_run_id'))} | "
            f"{markdown_cell(candidate.get('source_audit_json'))} | "
            f"{markdown_cell(candidate.get('source_audit_sha256'))} | "
            f"{markdown_cell(candidate.get('target_review_json'))} | "
            f"{markdown_cell(candidate.get('wandb_completion_json'))} | "
            f"{markdown_cell(candidate.get('scope_attestation_template_json'))} | "
            f"{markdown_cell(candidate.get('scope_attestation_render_report_json'))} | "
            f"{markdown_cell(candidate.get('scope_attestation_preflight_report_json'))} | "
            f"{markdown_cell(candidate.get('sync_dry_run_report_json'))} | "
            f"{markdown_cell(candidate.get('scope_attestation_required'))} | "
            f"{markdown_cell(handoff.get('available'))} | "
            f"{markdown_cell(handoff.get('step_count'))} | "
            f"{markdown_cell(handoff.get('command_count'))} | "
            f"{markdown_cell(handoff.get('evidence_path_count'))} | "
            f"{markdown_cell(handoff.get('expected_evidence_paths'))} | "
            f"{markdown_cell(candidate.get('scope_attestation_render_command'))} | "
            f"{markdown_cell(candidate.get('scope_attestation_preflight_command'))} | "
            f"{markdown_cell(candidate.get('sync_dry_run_command'))} | "
            f"{markdown_cell(candidate.get('sync_apply_command') or candidate.get('sync_command'))} |"
        )
    snippets.extend(
        [
            "### W&B Adoption Handoff Steps",
            (
                "| Benchmark | Model | Run ID | Step | Required | External action | "
                "Scope confirmation | Review mutation | Command | Expected evidence paths |"
            ),
        ]
    )
    step_row_count = 0
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        handoff = (
            candidate.get("operator_handoff")
            if isinstance(candidate.get("operator_handoff"), dict)
            else {}
        )
        steps = handoff.get("steps") if isinstance(handoff.get("steps"), list) else []
        for step in steps:
            if not isinstance(step, dict):
                continue
            step_row_count += 1
            snippets.append(
                "| "
                f"{markdown_cell(candidate.get('benchmark'))} | "
                f"{markdown_cell(candidate.get('model_slug'))} | "
                f"{markdown_cell(candidate.get('wandb_run_id'))} | "
                f"{markdown_cell(step.get('step'))} | "
                f"{markdown_cell(step.get('required'))} | "
                f"{markdown_cell(step.get('requires_external_action'))} | "
                f"{markdown_cell(step.get('requires_scope_confirmation'))} | "
                f"{markdown_cell(step.get('mutates_review_json'))} | "
                f"{markdown_cell(step.get('command'))} | "
                f"{markdown_cell(step.get('expected_evidence_paths'))} |"
            )
    if step_row_count == 0:
        snippets.append("| none |  |  |  |  |  |  |  |  |  |")
    return snippets


def expected_wandb_adoption_draft_markdown_snippets(
    draft: dict[str, Any],
) -> list[str]:
    snippets = [
        "# Existing W&B Adoption Review Draft",
        f"Status: `{draft.get('status')}`",
        f"Candidates: `{draft.get('candidate_count')}`",
        f"Source audit: `{draft.get('source_audit_json')}`",
        f"Source audit SHA256: `{draft.get('source_audit_sha256')}`",
        "## Required Human Fields",
        "## Candidates",
        "## Commands",
    ]
    fields = draft.get("required_human_fields")
    if isinstance(fields, list) and fields:
        snippets.extend(f"- `{field}`" for field in fields if isinstance(field, str))
    else:
        snippets.append("- none")
    candidates = draft.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        snippets.append("- none")
        return snippets
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            continue
        handoff = wandb_adoption_candidate_operator_handoff(candidate)
        snippets.extend(
            [
                f"### Candidate {index}",
                (
                    "- Operator handoff available: "
                    f"`{adoption_draft_markdown_cell(handoff.get('available'))}`"
                ),
                (
                    "- Handoff steps: "
                    f"`{adoption_draft_markdown_cell(handoff.get('step_count'))}`"
                ),
                (
                    "- Handoff commands: "
                    f"`{adoption_draft_markdown_cell(handoff.get('command_count'))}`"
                ),
                (
                    "- Scope confirmation steps: "
                    f"`{adoption_draft_markdown_cell(handoff.get('scope_confirmation_step_count'))}`"
                ),
                (
                    "- Evidence paths: "
                    f"`{adoption_draft_markdown_cell(handoff.get('evidence_path_count'))}`"
                ),
                f"#### Candidate {index} Handoff Steps",
                (
                    "| Step | Required | External action | Scope confirmation | "
                    "Review mutation | Command | Expected evidence paths |"
                ),
                "|---|---:|---:|---:|---:|---|---|",
            ]
        )
        steps = handoff.get("steps") if isinstance(handoff.get("steps"), list) else []
        if not steps:
            snippets.append("| none |  |  |  |  |  |  |")
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            snippets.append(
                "| "
                + " | ".join(
                    [
                        adoption_draft_markdown_cell(step.get("step")),
                        adoption_draft_markdown_cell(step.get("required")),
                        adoption_draft_markdown_cell(
                            step.get("requires_external_action")
                        ),
                        adoption_draft_markdown_cell(
                            step.get("requires_scope_confirmation")
                        ),
                        adoption_draft_markdown_cell(step.get("mutates_review_json")),
                        adoption_draft_markdown_cell(step.get("command")),
                        adoption_draft_markdown_cell(
                            step.get("expected_evidence_paths")
                        ),
                    ]
                )
                + " |"
            )
    return snippets


def expected_wandb_completion_contract_summary_snippets(
    contract: dict[str, Any],
) -> list[str]:
    snippets = [
        "| Field | Value |",
        f"| Status | {markdown_cell(contract.get('status'))} |",
        f"| Complete | {markdown_cell(contract.get('complete'))} |",
        f"| Required benchmarks | {markdown_cell(contract.get('required_benchmarks'))} |",
        f"| Required count | {markdown_cell(contract.get('required_count'))} |",
        (
            "| Release-proven count | "
            f"{markdown_cell(contract.get('release_completion_proven_count'))} |"
        ),
        (
            "| Standalone W&B OK count | "
            f"{markdown_cell(contract.get('standalone_completion_ok_count'))} |"
        ),
        (
            "| Existing formalized count | "
            f"{markdown_cell(contract.get('formalized_existing_result_count'))} |"
        ),
        (
            "| Missing release completion | "
            f"{markdown_cell(contract.get('missing_release_completion_benchmarks'))} |"
        ),
        f"| Next action count | {markdown_cell(contract.get('next_action_count'))} |",
        f"| Max age seconds | {markdown_cell(contract.get('max_age_seconds'))} |",
        (
            "| Benchmark | Required | Status | Release proven | Standalone | Review | "
            "Existing formalized | Run IDs | Attestation templates | "
            "Preflight reports | Dry-run reports | Missing reasons | "
            "Next actions | Commands |"
        ),
    ]
    rows = contract.get("benchmarks")
    if not isinstance(rows, list) or not rows:
        snippets.append("| none |  |  |  |  |  |  |  |  |  |  |  |  |  |")
        return snippets
    for row in rows:
        if not isinstance(row, dict):
            continue
        run_ids = unique_strings(
            [
                *(
                    row.get("review_run_ids")
                    if isinstance(row.get("review_run_ids"), list)
                    else []
                ),
                *(
                    row.get("standalone_run_ids")
                    if isinstance(row.get("standalone_run_ids"), list)
                    else []
                ),
                *(
                    row.get("formalized_existing_run_ids")
                    if isinstance(row.get("formalized_existing_run_ids"), list)
                    else []
                ),
            ]
        )
        snippets.append(
            "| "
            f"{markdown_cell(row.get('benchmark'))} | "
            f"{markdown_cell(row.get('required'))} | "
            f"{markdown_cell(row.get('status'))} | "
            f"{markdown_cell(row.get('release_completion_proven'))} | "
            f"{markdown_cell(row.get('standalone_status'))} | "
            f"{markdown_cell(row.get('review_status'))} | "
            f"{markdown_cell(row.get('formalized_existing_result'))} | "
            f"{markdown_cell(run_ids)} | "
            f"{markdown_cell(row.get('scope_attestation_template_paths'))} | "
            f"{markdown_cell(row.get('scope_attestation_preflight_report_paths'))} | "
            f"{markdown_cell(row.get('sync_dry_run_report_paths'))} | "
            f"{markdown_cell(row.get('missing_reasons'))} | "
            f"{markdown_cell(row.get('next_actions'))} | "
            f"{markdown_cell(row.get('recommended_commands'))} |"
        )
    return snippets


def operator_command_flag_value(
    parts: list[str], flag: str, default: str | None = None
) -> str | None:
    try:
        index = parts.index(flag)
    except ValueError:
        return default
    if index + 1 >= len(parts):
        return default
    return parts[index + 1]


def operator_command_flag_values(parts: list[str], flag: str) -> list[str]:
    values: list[str] = []
    for index, part in enumerate(parts[:-1]):
        if part == flag:
            values.append(parts[index + 1])
    return values


def command_invokes_taiwan_full_eval_batch(parts: list[str]) -> bool:
    return any(part.endswith("run_taiwan_full_eval_batch.py") for part in parts)


def command_invokes_weave_content_canary_execute(parts: list[str]) -> bool:
    return any(part.endswith("run_weave_agents_content_canary.py") for part in parts) and (
        "--execute" in parts
    )


def validate_source_bound_external_approval_command(
    *,
    parts: list[str],
    label: str,
    errors: list[str],
    command_description: str,
) -> None:
    approval_report = operator_command_flag_value(
        parts,
        "--external-action-approval-report-json",
    )
    approval_source_packet = operator_command_flag_value(
        parts,
        "--external-action-approval-source-packet-json",
    )
    if not isinstance(approval_source_packet, str) or not approval_source_packet.strip():
        errors.append(
            f"{label} runs {command_description} without "
            "--external-action-approval-source-packet-json"
        )
    elif not approval_source_packet.endswith("external_action_approval_packet.json"):
        errors.append(
            f"{label} --external-action-approval-source-packet-json must point to "
            "external_action_approval_packet.json"
        )
    if not isinstance(approval_report, str) or not approval_report.strip():
        errors.append(
            f"{label} runs {command_description} without "
            "--external-action-approval-report-json"
        )
        return
    if not approval_report.endswith(".verify.json"):
        errors.append(
            f"{label} --external-action-approval-report-json must point to a "
            ".verify.json approval verifier report"
        )


def validate_taiwan_full_batch_external_approval_command(
    *,
    parts: list[str],
    label: str,
    errors: list[str],
) -> None:
    if not command_invokes_taiwan_full_eval_batch(parts):
        return
    if "--prepare-only" in parts:
        return
    validate_source_bound_external_approval_command(
        parts=parts,
        label=label,
        errors=errors,
        command_description="run_taiwan_full_eval_batch.py",
    )


def validate_taiwan_full_batch_agentic_production_evidence_command(
    *,
    parts: list[str],
    label: str,
    errors: list[str],
) -> None:
    if not command_invokes_taiwan_full_eval_batch(parts):
        return
    if "--prepare-only" in parts:
        return
    phase = operator_command_flag_value(parts, "--phase", "full")
    if phase not in {"agentic", "full"}:
        return
    for option in (
        "--require-nemoclaw-agentic-config",
        "--weave-content-canary-gate",
        "--require-weave-content-canary",
        "--verify-wandb-completion",
        "--verify-weave-agents",
        "--wandb-run-id-prefix",
        "--weave-agents-require-tool-span",
        "--weave-agents-require-tool-content",
        "--weave-agents-require-usage",
    ):
        if option not in parts:
            errors.append(
                f"{label} runs paid {phase} run_taiwan_full_eval_batch.py "
                f"without {option}"
            )


def validate_weave_content_canary_external_approval_command(
    *,
    parts: list[str],
    label: str,
    errors: list[str],
) -> None:
    if not command_invokes_weave_content_canary_execute(parts):
        return
    validate_source_bound_external_approval_command(
        parts=parts,
        label=label,
        errors=errors,
        command_description="run_weave_agents_content_canary.py --execute",
    )
    if "--nemoclaw-sandbox" not in parts:
        errors.append(
            f"{label} runs run_weave_agents_content_canary.py --execute without "
            "--nemoclaw-sandbox"
        )


def command_invokes_wandb_completion_sync_apply(parts: list[str]) -> bool:
    return (
        any(part.endswith("sync_wandb_completion_to_paid_review.py") for part in parts)
        and "--in-place" in parts
    )


def validate_wandb_completion_sync_apply_command(
    *,
    parts: list[str],
    label: str,
    errors: list[str],
) -> None:
    if not command_invokes_wandb_completion_sync_apply(parts):
        return
    if "--validated-dry-run-report-json" not in parts:
        errors.append(
            f"{label} runs sync_wandb_completion_to_paid_review.py --in-place "
            "without --validated-dry-run-report-json"
        )


def command_invokes_weave_agents_completion_sync_apply(parts: list[str]) -> bool:
    return (
        any(part.endswith("sync_weave_agents_completion_to_paid_review.py") for part in parts)
        and "--in-place" in parts
    )


def validate_weave_agents_completion_sync_apply_command(
    *,
    parts: list[str],
    label: str,
    errors: list[str],
) -> None:
    if not command_invokes_weave_agents_completion_sync_apply(parts):
        return
    if "--validated-dry-run-report-json" not in parts:
        errors.append(
            f"{label} runs sync_weave_agents_completion_to_paid_review.py --in-place "
            "without --validated-dry-run-report-json"
        )


def command_invokes_openclaw_agents_check(parts: list[str]) -> bool:
    return any(part.endswith("run_openclaw_agent_protocol.py") for part in parts) and (
        "check-agents" in parts
    )


def validate_openclaw_agents_check_command(
    *,
    parts: list[str],
    label: str,
    errors: list[str],
) -> None:
    if not command_invokes_openclaw_agents_check(parts):
        return
    json_path = operator_command_flag_value(parts, "--json")
    if not isinstance(json_path, str) or not json_path.strip():
        errors.append(
            f"{label} runs run_openclaw_agent_protocol.py check-agents without --json"
        )
    elif not json_path.endswith(".agents.json"):
        errors.append(
            f"{label} check-agents --json output must end with .agents.json"
        )


def taiwan_full_batch_implicit_output_paths(parts: list[str]) -> list[str]:
    if not command_invokes_taiwan_full_eval_batch(parts):
        return []
    output_root = operator_command_flag_value(
        parts, "--output-root", "outputs/taiwan_full_eval"
    )
    phase = operator_command_flag_value(parts, "--phase", "full")
    if not output_root or not phase:
        return []
    plan_prefix = "canary_" if "--canary" in parts else ""
    root = Path(output_root)
    outputs = [
        str(root / f"{plan_prefix}{phase}_execution_plan.json"),
        str(root / f"{plan_prefix}{phase}_paid_run_review.json"),
    ]
    if "--prepare-only" not in parts:
        outputs.append(str(root / "batch_manifest.json"))
    if "--verify-wandb-completion" in parts:
        benchmarks = operator_command_flag_values(parts, "--wandb-verify-benchmark")
        if not benchmarks:
            benchmarks = {
                "full": ["taiwan_full"],
                "agentic": ["agentic_math", "agentic_swe"],
                "agentic_aggregate": ["taiwan_full"],
            }.get(phase, [])
        outputs.extend(
            str(root / "wandb_completion" / f"{phase}-MODEL_SLUG-{benchmark}.json")
            for benchmark in benchmarks
        )
    if "--verify-weave-agents" in parts:
        outputs.append(str(root / "weave_agents_completion" / f"{phase}-MODEL_SLUG.json"))
    return unique_strings(outputs)


def weave_content_canary_implicit_output_paths(parts: list[str]) -> list[str]:
    if not any(part.endswith("run_weave_agents_content_canary.py") for part in parts):
        return []
    output_dir = operator_command_flag_value(
        parts,
        "--output-dir",
        "outputs/weave_agents_content_canary",
    )
    canary_id = operator_command_flag_value(parts, "--canary-id", "CANARY_ID")
    if not output_dir or not canary_id:
        return []
    safe_id = "".join(char if char.isalnum() else "_" for char in canary_id)
    task_id = f"weave_agents_content_canary_{safe_id}"
    root = Path(output_dir)
    outputs = [
        str(root / "prompts" / f"{task_id}.md"),
        str(root / "plans" / f"{task_id}.json"),
        str(root / "plans" / f"{task_id}.gate.json"),
    ]
    if "--execute" in parts:
        outputs.extend(
            [
                str(root / "plans" / f"{task_id}.command_result.json"),
                str(root / "agentic_math" / task_id / "openclaw_result.json"),
                str(root / "verifier" / task_id / "attempt_*.json"),
                str(root / "agents_diagnostics" / f"{task_id}.agents.json"),
            ]
        )
    return unique_strings(outputs)


def operator_command_output_paths(command: str) -> list[str]:
    try:
        parts = shlex.split(command)
    except ValueError:
        return []
    output_flags = {
        "--json",
        "--markdown",
        "--report-json",
        "--release-gate-json",
        "--bundle-verification-json",
        "--readiness-report-json",
    }
    outputs: list[str] = []
    for index, part in enumerate(parts[:-1]):
        if part in output_flags:
            outputs.append(parts[index + 1])
    outputs.extend(taiwan_full_batch_implicit_output_paths(parts))
    outputs.extend(weave_content_canary_implicit_output_paths(parts))
    outputs.extend(nemoclaw_install_operation_log_paths(parts))
    return unique_strings(outputs)


def nemoclaw_install_operation_log_paths(parts: list[str]) -> list[str]:
    if not any(part.endswith("install_nemoclaw.sh") for part in parts):
        return []
    if "--install" not in parts and "--onboard" not in parts:
        return []
    output_json = ""
    for index, part in enumerate(parts[:-1]):
        if part == "--json":
            output_json = parts[index + 1]
    if not output_json:
        return []
    stem = output_json[:-5] if output_json.endswith(".json") else output_json
    outputs: list[str] = []
    if "--install" in parts:
        outputs.append(f"{stem}.install.log")
    if "--onboard" in parts:
        outputs.append(f"{stem}.onboard.log")
    return outputs


def operator_command_requires_paid_api(command: str) -> bool:
    if "--prepare-only" in command:
        return False
    return any(
        marker in command
        for marker in (
            "run_taiwan_full_eval_batch.py",
            "run_weave_agents_content_canary.py --execute",
            "run_agentic_math_openclaw.py",
            "run_swebench_pro_openclaw.py",
        )
    )


def operator_command_requires_wandb_access(command: str) -> bool:
    return any(
        marker in command
        for marker in (
            "verify_taiwan_wandb_completion.py",
            "run_taiwan_full_eval_batch.py",
            "run_weave_agents_content_canary.py",
            "run_openclaw_agent_protocol.py check-agents",
            "verify_taiwan_weave_agents.py",
            "sync_wandb_completion_to_paid_review.py",
            "sync_weave_agents_completion_to_paid_review.py",
        )
    )


def operator_command_requires_wandb_write(command: str) -> bool:
    if "--prepare-only" in command:
        return False
    return any(
        marker in command
        for marker in (
            "run_taiwan_full_eval_batch.py",
            "log_agentic_math_results_to_wandb.py",
            "log_agentic_swe_results_to_wandb.py",
            "run_weave_agents_content_canary.py --execute",
        )
    )


OPERATOR_PLACEHOLDER_TOKENS = (
    "PHASE",
    "RUN_ID",
    "MODEL_SLUG",
    "YYYYMMDD",
    "YYYYMMDDTHHMM",
    "YYYYMMDDTHHMMSS",
    "CONTENT_CANARY_YYYYMMDDTHHMM",
    "WEAVE_CONTENT_CANARY_GATE",
    "REQUIRED_SCOPE_ATTESTATION_JSON",
    "REQUIRED_SYNC_DRY_RUN_REPORT_JSON",
)


def unresolved_operator_placeholder_tokens(values: list[str]) -> list[str]:
    found: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        for token in OPERATOR_PLACEHOLDER_TOKENS:
            if token in value and token not in found:
                found.append(token)
    return found


def command_policy_tier(parts: list[str]) -> tuple[str | None, bool]:
    """Return (--policy-tier value, flag_present) for shell-split command parts."""
    for index, part in enumerate(parts):
        if part == "--policy-tier":
            if index + 1 < len(parts):
                return parts[index + 1], True
            return None, True
        if part.startswith("--policy-tier="):
            return part.split("=", 1)[1], True
    return None, False


def command_flag_value(parts: list[str], flag: str) -> tuple[str | None, bool]:
    for index, part in enumerate(parts):
        if part == flag:
            if index + 1 < len(parts):
                return parts[index + 1], True
            return None, True
        if part.startswith(flag + "="):
            return part.split("=", 1)[1], True
    return None, False


def command_invokes_nemoclaw_setup(parts: list[str]) -> bool:
    return any(str(part).endswith("install_nemoclaw.sh") for part in parts)


def command_invokes_nemoclaw_installer_review(parts: list[str]) -> bool:
    return any(str(part).endswith("review_nemoclaw_installer.py") for part in parts)


def command_invokes_nemoclaw_post_install(parts: list[str]) -> bool:
    return any(str(part).endswith("verify_nemoclaw_post_install.py") for part in parts)


def command_installs_nemoclaw(parts: list[str]) -> bool:
    return command_invokes_nemoclaw_setup(parts) and "--install" in parts


def validate_nemoclaw_installer_sha_command(
    *,
    parts: list[str],
    label: str,
    errors: list[str],
) -> None:
    if not command_installs_nemoclaw(parts):
        return
    installer_sha256, sha_flag_present = command_flag_value(parts, "--installer-sha256")
    if not sha_flag_present:
        errors.append(f"{label} installs NeMoClaw without --installer-sha256")
    elif not installer_sha256:
        errors.append(f"{label} has empty --installer-sha256")


def validate_nemoclaw_installer_lock_command(
    *,
    parts: list[str],
    label: str,
    errors: list[str],
    expected_lock_json: str | None = None,
) -> None:
    if not command_installs_nemoclaw(parts):
        return
    lock_json, lock_flag_present = command_flag_value(parts, "--installer-lock-json")
    if not lock_flag_present:
        errors.append(f"{label} installs NeMoClaw without --installer-lock-json")
    elif not lock_json:
        errors.append(f"{label} has empty --installer-lock-json")
    elif expected_lock_json and source_path_key(lock_json) != source_path_key(expected_lock_json):
        errors.append(
            f"{label} --installer-lock-json does not match bundled installer review summary"
        )


def validate_nemoclaw_installer_review_command(
    *,
    parts: list[str],
    label: str,
    errors: list[str],
) -> str | None:
    if not command_invokes_nemoclaw_installer_review(parts):
        return None
    review_json, review_flag_present = command_flag_value(parts, "--json")
    if not review_flag_present:
        errors.append(f"{label} invokes NeMoClaw installer review without --json")
        return None
    if not review_json:
        errors.append(f"{label} has empty NeMoClaw installer review --json")
        return None
    return review_json


def validate_nemoclaw_installer_review_json_command(
    *,
    parts: list[str],
    label: str,
    errors: list[str],
) -> str | None:
    if not command_installs_nemoclaw(parts):
        return None
    review_json, review_flag_present = command_flag_value(parts, "--installer-review-json")
    if not review_flag_present:
        errors.append(f"{label} installs NeMoClaw without --installer-review-json")
        return None
    if not review_json:
        errors.append(f"{label} has empty --installer-review-json")
        return None
    return review_json


def validate_nemoclaw_installer_review_payload(
    payload: dict[str, Any],
    *,
    summary: dict[str, Any],
    label: str,
) -> list[str]:
    errors: list[str] = []
    expected_sha = str(summary.get("sha256") or "").lower()
    if payload.get("schema_version") != 1:
        errors.append(f"{label} schema_version must be 1")
    if payload.get("ok") is not True:
        errors.append(f"{label} ok must be true")
    if payload.get("status") != "reviewed":
        errors.append(f"{label} status must be reviewed")
    sha = str(payload.get("sha256") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        errors.append(f"{label} sha256 is not 64 lowercase hex")
    if expected_sha and sha != expected_sha:
        errors.append(f"{label} sha256 does not match current_gate summary")
    for field in ("installer_url", "install_ref", "generated_at", "size_bytes"):
        if payload.get(field) != summary.get(field):
            errors.append(f"{label} {field} does not match current_gate summary")
    for field in (
        "will_execute_installer",
        "will_install_or_onboard",
        "will_launch_model_inference",
        "will_query_wandb",
    ):
        if payload.get(field) is not False:
            errors.append(f"{label} {field} must be false")
    for field in ("lock_json", "lock_verified", "expected_sha256"):
        if field in summary and summary.get(field) not in (None, "") and payload.get(field) != summary.get(field):
            errors.append(f"{label} {field} does not match current_gate summary")
    recommended_command = payload.get("recommended_install_command")
    if not isinstance(recommended_command, str) or not recommended_command.strip():
        errors.append(f"{label} recommended_install_command must be a non-empty string")
    else:
        recommended_parts = command_parts(recommended_command)
        recommended_label = f"{label} recommended_install_command"
        if not command_installs_nemoclaw(recommended_parts):
            errors.append(f"{recommended_label} must install NeMoClaw")
        if "--onboard" not in recommended_parts:
            errors.append(f"{recommended_label} must include --onboard")
        if "--yes-i-accept-third-party-software" not in recommended_parts:
            errors.append(
                f"{recommended_label} must include --yes-i-accept-third-party-software"
            )
        review_json, review_flag_present = command_flag_value(
            recommended_parts,
            "--installer-review-json",
        )
        if not review_flag_present:
            errors.append(f"{recommended_label} installs NeMoClaw without --installer-review-json")
        elif source_path_key(review_json) != source_path_key(summary.get("path")):
            errors.append(
                f"{recommended_label} --installer-review-json does not match current_gate summary"
            )
        output_json, output_flag_present = command_flag_value(recommended_parts, "--json")
        if not output_flag_present:
            errors.append(f"{recommended_label} installs NeMoClaw without --json")
        elif not output_json:
            errors.append(f"{recommended_label} has empty --json")
        policy_tier, policy_tier_present = command_flag_value(recommended_parts, "--policy-tier")
        if not policy_tier_present:
            errors.append(f"{recommended_label} installs NeMoClaw without --policy-tier")
        elif policy_tier != "restricted":
            errors.append(f"{recommended_label} --policy-tier must be restricted")
        validate_nemoclaw_install_command_matches_review_summary(
            parts=recommended_parts,
            summary=summary,
            label=recommended_label,
            errors=errors,
        )
    return errors


def validate_nemoclaw_installer_lock_payload(
    payload: dict[str, Any],
    *,
    summary: dict[str, Any],
    label: str,
) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append(f"{label} schema_version must be 1")
    for field in ("installer_url", "install_ref", "sha256", "size_bytes"):
        if payload.get(field) != summary.get(field):
            errors.append(f"{label} {field} does not match current_gate installer review summary")
    sha = str(payload.get("sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        errors.append(f"{label} sha256 is not 64 lowercase hex")
    for field in (
        "will_execute_installer",
        "will_install_or_onboard",
        "will_launch_model_inference",
        "will_query_wandb",
    ):
        if payload.get(field) is not False:
            errors.append(f"{label} {field} must be false")
    return errors


def validate_nemoclaw_installer_review_command_matches_summary(
    *,
    parts: list[str],
    summary: dict[str, Any],
    label: str,
    errors: list[str],
) -> None:
    if not command_invokes_nemoclaw_installer_review(parts):
        return
    expected_sha = str(summary.get("sha256") or "").lower()
    expected_url = summary.get("installer_url")
    expected_ref = summary.get("install_ref")
    expected_lock = summary.get("lock_json")
    url, url_present = command_flag_value(parts, "--url")
    install_ref, ref_present = command_flag_value(parts, "--install-ref")
    expected_sha_arg, expected_sha_present = command_flag_value(parts, "--expected-sha256")
    lock_json, lock_present = command_flag_value(parts, "--lock-json")
    if url_present and expected_url and url != expected_url:
        errors.append(f"{label} --url does not match bundled installer review summary")
    if ref_present and expected_ref and install_ref != expected_ref:
        errors.append(f"{label} --install-ref does not match bundled installer review summary")
    if expected_sha:
        if not expected_sha_present:
            errors.append(f"{label} invokes NeMoClaw installer review without --expected-sha256")
        elif str(expected_sha_arg).lower() != expected_sha:
            errors.append(f"{label} --expected-sha256 does not match bundled installer review summary")
    if expected_lock:
        if not lock_present:
            errors.append(f"{label} invokes NeMoClaw installer review without --lock-json")
        elif source_path_key(lock_json) != source_path_key(expected_lock):
            errors.append(f"{label} --lock-json does not match bundled installer review summary")


def validate_nemoclaw_install_command_matches_review_summary(
    *,
    parts: list[str],
    summary: dict[str, Any],
    label: str,
    errors: list[str],
) -> None:
    if not command_installs_nemoclaw(parts):
        return
    expected_sha = str(summary.get("sha256") or "").lower()
    expected_ref = summary.get("install_ref")
    expected_lock = summary.get("lock_json")
    installer_sha, sha_present = command_flag_value(parts, "--installer-sha256")
    install_ref, ref_present = command_flag_value(parts, "--install-ref")
    lock_json, lock_present = command_flag_value(parts, "--installer-lock-json")
    if expected_sha:
        if not sha_present:
            errors.append(f"{label} installs NeMoClaw without --installer-sha256")
        elif str(installer_sha).lower() != expected_sha:
            errors.append(f"{label} --installer-sha256 does not match bundled installer review summary")
    if expected_ref:
        if not ref_present:
            errors.append(f"{label} installs NeMoClaw without --install-ref")
        elif install_ref != expected_ref:
            errors.append(f"{label} --install-ref does not match bundled installer review summary")
    if expected_lock:
        if not lock_present:
            errors.append(f"{label} installs NeMoClaw without --installer-lock-json")
        elif source_path_key(lock_json) != source_path_key(expected_lock):
            errors.append(f"{label} --installer-lock-json does not match bundled installer review summary")


def validate_operator_next_steps_semantics(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    operator_plan = manifest.get("operator_plan")
    if not isinstance(operator_plan, dict):
        return errors
    json_path_value = operator_plan.get("json")
    if not isinstance(json_path_value, str) or not json_path_value:
        return errors
    expected_lock_json = None
    current_gate = manifest.get("current_gate")
    if isinstance(current_gate, dict) and isinstance(
        current_gate.get("nemoclaw_installer_review"),
        dict,
    ):
        lock_value = current_gate["nemoclaw_installer_review"].get("lock_json")
        if isinstance(lock_value, str) and lock_value.strip():
            expected_lock_json = lock_value
    try:
        payload = read_json_object(bundle_dir / json_path_value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return [f"operator_plan json is not readable for step semantic validation: {exc}"]
    operator_steps = payload.get("operator_next_steps")
    if not isinstance(operator_steps, dict):
        return errors
    steps = operator_steps.get("steps")
    if not isinstance(steps, list):
        errors.append("operator_plan operator_next_steps.steps is not a list")
        return errors
    if operator_steps.get("step_count") != len(steps):
        errors.append("operator_plan step_count does not match steps length")

    counter_fields = {
        "paid_api_step_count": "requires_paid_api",
        "wandb_access_step_count": "requires_wandb_access",
        "wandb_write_step_count": "requires_wandb_write",
        "third_party_acceptance_step_count": "requires_third_party_acceptance",
        "scope_confirmation_step_count": "requires_scope_confirmation",
    }
    for count_field, step_field in counter_fields.items():
        expected = sum(
            1 for step in steps if isinstance(step, dict) and step.get(step_field)
        )
        if operator_steps.get(count_field) != expected:
            errors.append(f"operator_plan {count_field} does not match step flags")

    expected_template_step_count = sum(
        1
        for step in steps
        if isinstance(step, dict)
        and isinstance(step.get("command_template_count"), int)
        and step.get("command_template_count", 0) > 0
    )
    if operator_steps.get("command_template_step_count") != expected_template_step_count:
        errors.append("operator_plan command_template_step_count does not match step templates")
    expected_operator_placeholders = unique_strings(
        [
            token
            for step in steps
            if isinstance(step, dict)
            for token in (
                step.get("unresolved_placeholder_tokens")
                if isinstance(step.get("unresolved_placeholder_tokens"), list)
                else []
            )
            if isinstance(token, str) and token
        ]
    )
    if operator_steps.get("unresolved_placeholder_tokens") != expected_operator_placeholders:
        errors.append("operator_plan unresolved_placeholder_tokens does not match steps")

    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            errors.append(f"operator_plan step {index} is not an object")
            continue
        gate = str(step.get("gate") or "")
        commands = [
            str(command)
            for command in step.get("commands", [])
            if isinstance(command, str) and command.strip()
        ]
        for field in (
            "requires_paid_api",
            "requires_wandb_access",
            "requires_wandb_write",
            "requires_third_party_acceptance",
            "requires_nemoclaw_install",
            "requires_scope_confirmation",
        ):
            if not isinstance(step.get(field), bool):
                errors.append(f"operator_plan step {index} {field} is not a bool")

        expected_flags = {
            "requires_paid_api": any(operator_command_requires_paid_api(command) for command in commands),
            "requires_wandb_access": any(operator_command_requires_wandb_access(command) for command in commands),
            "requires_wandb_write": any(operator_command_requires_wandb_write(command) for command in commands),
            "requires_third_party_acceptance": any(
                "--yes-i-accept-third-party-software" in command for command in commands
            ),
            "requires_nemoclaw_install": gate == "nemoclaw_readiness"
            or any(
                "install_nemoclaw.sh --install" in command
                or "install_nemoclaw.sh --install --onboard" in command
                for command in commands
            ),
            "requires_scope_confirmation": any(
                "--scope-attestation-json" in command for command in commands
            )
            or any(
                isinstance(row, dict)
                and row.get("scope_confirmation_required")
                and (
                    "sync_ready_adoption_candidate_count" not in row
                    or row.get("sync_ready_adoption_candidate_count", 0) > 0
                )
                for row in (
                    step.get("related_contract_benchmarks")
                    if isinstance(step.get("related_contract_benchmarks"), list)
                    else []
                )
            ),
        }
        for field, expected in expected_flags.items():
            if isinstance(step.get(field), bool) and step.get(field) != expected:
                errors.append(
                    f"operator_plan step {index} {field} does not match command semantics"
                )

        expected_outputs = unique_strings(
            [
                output
                for command in commands
                for output in operator_command_output_paths(command)
            ]
        )
        if step.get("command_count") != len(commands):
            errors.append(
                f"operator_plan step {index} command_count does not match commands"
            )
        evidence_to_produce = step.get("evidence_to_produce")
        if not isinstance(evidence_to_produce, list):
            errors.append(f"operator_plan step {index} evidence_to_produce is not a list")
        elif evidence_to_produce != expected_outputs:
            errors.append(
                f"operator_plan step {index} evidence_to_produce does not match command outputs"
            )
        if isinstance(evidence_to_produce, list) and step.get("evidence_path_count") != len(
            evidence_to_produce
        ):
            errors.append(
                f"operator_plan step {index} evidence_path_count does not match evidence_to_produce"
            )
        expected_placeholder_tokens = unresolved_operator_placeholder_tokens(
            [*commands, *(evidence_to_produce if isinstance(evidence_to_produce, list) else [])]
        )
        expected_command_template_count = sum(
            1 for command in commands if unresolved_operator_placeholder_tokens([command])
        )
        expected_evidence_template_count = sum(
            1
            for output in (evidence_to_produce if isinstance(evidence_to_produce, list) else [])
            if isinstance(output, str) and unresolved_operator_placeholder_tokens([output])
        )
        if step.get("unresolved_placeholder_tokens") != expected_placeholder_tokens:
            errors.append(
                f"operator_plan step {index} unresolved_placeholder_tokens does not match command/evidence placeholders"
            )
        if step.get("command_template_count") != expected_command_template_count:
            errors.append(
                f"operator_plan step {index} command_template_count does not match command placeholders"
            )
        if step.get("evidence_template_count") != expected_evidence_template_count:
            errors.append(
                f"operator_plan step {index} evidence_template_count does not match evidence placeholders"
            )
        expected_ready = not expected_placeholder_tokens
        if step.get("ready_to_execute_without_placeholder") is not expected_ready:
            errors.append(
                f"operator_plan step {index} ready_to_execute_without_placeholder does not match placeholders"
            )
    return errors


def validate_operator_plan_command_policy(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    operator_plan = manifest.get("operator_plan")
    if not isinstance(operator_plan, dict):
        return errors
    json_path_value = operator_plan.get("json")
    if not isinstance(json_path_value, str) or not json_path_value:
        return errors
    expected_lock_json = None
    current_gate = manifest.get("current_gate")
    if isinstance(current_gate, dict) and isinstance(
        current_gate.get("nemoclaw_installer_review"),
        dict,
    ):
        lock_value = current_gate["nemoclaw_installer_review"].get("lock_json")
        if isinstance(lock_value, str) and lock_value.strip():
            expected_lock_json = lock_value
    try:
        payload = read_json_object(bundle_dir / json_path_value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return [f"operator_plan json is not readable for command policy validation: {exc}"]

    commands = collect_operator_plan_commands(payload)
    has_nemoclaw_install = False
    has_nemoclaw_installer_review = False
    nemoclaw_review_outputs: set[str] = set()
    nemoclaw_install_review_jsons: list[str] = []
    for index, command in enumerate(commands, start=1):
        try:
            parts = shlex.split(command)
        except ValueError:
            parts = command.split()
        validate_taiwan_full_batch_external_approval_command(
            parts=parts,
            label=f"operator_plan command {index}",
            errors=errors,
        )
        validate_taiwan_full_batch_agentic_production_evidence_command(
            parts=parts,
            label=f"operator_plan command {index}",
            errors=errors,
        )
        validate_weave_content_canary_external_approval_command(
            parts=parts,
            label=f"operator_plan command {index}",
            errors=errors,
        )
        validate_wandb_completion_sync_apply_command(
            parts=parts,
            label=f"operator_plan command {index}",
            errors=errors,
        )
        validate_weave_agents_completion_sync_apply_command(
            parts=parts,
            label=f"operator_plan command {index}",
            errors=errors,
        )
        validate_openclaw_agents_check_command(
            parts=parts,
            label=f"operator_plan command {index}",
            errors=errors,
        )
        if command_installs_nemoclaw(parts):
            has_nemoclaw_install = True
        if command_invokes_nemoclaw_installer_review(parts):
            has_nemoclaw_installer_review = True
            review_json = validate_nemoclaw_installer_review_command(
                parts=parts,
                label=f"operator_plan command {index}",
                errors=errors,
            )
            if review_json:
                nemoclaw_review_outputs.add(review_json)
        forbidden_markers = [
            marker
            for marker in FORBIDDEN_RELEASE_OPERATOR_COMMAND_MARKERS
            if marker in command
        ]
        if forbidden_markers:
            errors.append(
                "operator_plan command "
                f"{index} uses forbidden release operator marker(s): "
                + ", ".join(sorted(set(forbidden_markers)))
                + "; use the approved OpenAI-direct canary path"
            )
        deprecated = [flag for flag in DEPRECATED_WANDB_ADOPTION_FLAGS if flag in parts]
        if deprecated:
            errors.append(
                "operator_plan command "
                f"{index} uses deprecated W&B adoption flag(s): "
                + ", ".join(deprecated)
                + "; use --scope-attestation-json"
            )
        if command_invokes_nemoclaw_setup(parts):
            validate_nemoclaw_installer_sha_command(
                parts=parts,
                label=f"operator_plan command {index}",
                errors=errors,
            )
            validate_nemoclaw_installer_lock_command(
                parts=parts,
                label=f"operator_plan command {index}",
                errors=errors,
                expected_lock_json=expected_lock_json,
            )
            install_review_json = validate_nemoclaw_installer_review_json_command(
                parts=parts,
                label=f"operator_plan command {index}",
                errors=errors,
            )
            if install_review_json:
                nemoclaw_install_review_jsons.append(install_review_json)
            policy_tier, policy_flag_present = command_policy_tier(parts)
            if policy_flag_present and policy_tier != REQUIRED_NEMOCLAW_POLICY_TIER:
                errors.append(
                    "operator_plan command "
                    f"{index} uses NeMoClaw --policy-tier {policy_tier or '<missing>'}; "
                    f"use --policy-tier {REQUIRED_NEMOCLAW_POLICY_TIER}"
                )
            if "--onboard" in parts and not policy_flag_present:
                errors.append(
                    "operator_plan command "
                    f"{index} onboards NeMoClaw without --policy-tier "
                    f"{REQUIRED_NEMOCLAW_POLICY_TIER}"
                )
    if has_nemoclaw_install and not has_nemoclaw_installer_review:
        errors.append(
            "operator_plan installs NeMoClaw but has no review_nemoclaw_installer.py command"
        )
    for review_json in nemoclaw_install_review_jsons:
        if review_json not in nemoclaw_review_outputs:
            errors.append(
                "operator_plan installs NeMoClaw with "
                f"--installer-review-json {review_json} but no matching "
                "review_nemoclaw_installer.py --json output"
            )
    return errors


def validate_operator_plan_command_scripts(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    operator_plan = manifest.get("operator_plan")
    if not isinstance(operator_plan, dict):
        return errors
    json_path_value = operator_plan.get("json")
    if not isinstance(json_path_value, str) or not json_path_value:
        return errors

    try:
        payload = read_json_object(bundle_dir / json_path_value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return [f"operator_plan json is not readable for command script validation: {exc}"]

    scripts = collect_operator_plan_command_scripts(payload)
    if not scripts:
        return errors
    records = file_records_by_source(manifest)
    for script in scripts:
        key = source_path_key(script)
        record = records.get(key)
        if not isinstance(record, dict):
            errors.append(f"operator_plan command script is not bundled: {script}")
            continue
        roles = record.get("roles")
        if not isinstance(roles, list) or "operator_plan:command_script" not in roles:
            errors.append(f"operator_plan command script missing role operator_plan:command_script: {script}")
        bundle_path = record.get("bundle_path")
        if not isinstance(bundle_path, str) or not bundle_path:
            errors.append(f"operator_plan command script missing bundle_path: {script}")
    return errors


def validate_current_gate_remediation_command_scripts(
    *,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    scripts = collect_current_gate_remediation_command_scripts(manifest)
    if not scripts:
        return errors
    records = file_records_by_source(manifest)
    for script in scripts:
        key = source_path_key(script)
        record = records.get(key)
        if not isinstance(record, dict):
            errors.append(
                f"current_gate remediation_plan command script is not bundled: {script}"
            )
            continue
        roles = record.get("roles")
        if not isinstance(roles, list) or (
            "current_gate:remediation_plan:command_script" not in roles
        ):
            errors.append(
                "current_gate remediation_plan command script missing role "
                f"current_gate:remediation_plan:command_script: {script}"
            )
        bundle_path = record.get("bundle_path")
        if not isinstance(bundle_path, str) or not bundle_path:
            errors.append(
                f"current_gate remediation_plan command script missing bundle_path: {script}"
            )
    return errors


def validate_existing_results_relog_command_scripts(
    *,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    scripts = collect_existing_results_relog_command_scripts(manifest)
    if not scripts:
        return errors
    records = file_records_by_source(manifest)
    for script in scripts:
        key = source_path_key(script)
        record = records.get(key)
        if not isinstance(record, dict):
            errors.append(f"existing_results relog command script is not bundled: {script}")
            continue
        roles = record.get("roles")
        if not isinstance(roles, list) or (
            "existing_results_audit:relog_command_script" not in roles
        ):
            errors.append(
                "existing_results relog command script missing role "
                f"existing_results_audit:relog_command_script: {script}"
            )
        bundle_path = record.get("bundle_path")
        if not isinstance(bundle_path, str) or not bundle_path:
            errors.append(f"existing_results relog command script missing bundle_path: {script}")

    if any(script in RELOG_COMMAND_SCRIPT_PATHS for script in scripts):
        helper_record = records.get(source_path_key(RELOG_WANDB_APPROVAL_HELPER_SCRIPT))
        if not isinstance(helper_record, dict):
            errors.append(
                "existing_results relog dependency script is not bundled: "
                f"{RELOG_WANDB_APPROVAL_HELPER_SCRIPT}"
            )
        else:
            roles = helper_record.get("roles")
            if not isinstance(roles, list) or (
                "existing_results_audit:relog_dependency_script" not in roles
            ):
                errors.append(
                    "existing_results relog dependency script missing role "
                    "existing_results_audit:relog_dependency_script: "
                    f"{RELOG_WANDB_APPROVAL_HELPER_SCRIPT}"
                )
            bundle_path = helper_record.get("bundle_path")
            if not isinstance(bundle_path, str) or not bundle_path:
                errors.append(
                    "existing_results relog dependency script missing bundle_path: "
                    f"{RELOG_WANDB_APPROVAL_HELPER_SCRIPT}"
                )
    return errors


def validate_agentic_runner_script_evidence(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    records = file_records_by_source(manifest)
    for script_path, contract in AGENTIC_RUNNER_SCRIPT_CONTRACTS.items():
        record = records.get(source_path_key(script_path))
        if not isinstance(record, dict):
            errors.append(f"agentic runner script is not bundled: {script_path}")
            continue
        roles = record.get("roles")
        required_role = str(contract["role"])
        if not isinstance(roles, list) or "agentic_runner:script" not in roles:
            errors.append(f"agentic runner script missing role agentic_runner:script: {script_path}")
        if not isinstance(roles, list) or required_role not in roles:
            errors.append(f"agentic runner script missing role {required_role}: {script_path}")
        bundle_path = record.get("bundle_path")
        if not isinstance(bundle_path, str) or not bundle_path:
            errors.append(f"agentic runner script missing bundle_path: {script_path}")
            continue
        script_file = bundle_dir / bundle_path
        try:
            text = script_file.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"agentic runner script is not readable: {script_path}: {exc}")
            continue
        for label, token in contract.get("tokens", ()):
            if token not in text:
                errors.append(
                    "agentic runner script missing source contract "
                    f"{label}: {script_path}: {token}"
                )
    return errors


def validate_nemoclaw_canary_readiness_script_source(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    records = file_records_by_source(manifest)
    record = records.get(source_path_key(NEMOCLAW_CANARY_READINESS_SCRIPT))
    if not isinstance(record, dict):
        return errors
    roles = record.get("roles")
    if not isinstance(roles, list) or not (
        "operator_plan:command_script" in roles
        or "current_gate:remediation_plan:command_script" in roles
    ):
        errors.append(
            "NeMoClaw canary readiness script missing command-script role: "
            f"{NEMOCLAW_CANARY_READINESS_SCRIPT}"
        )
    bundle_path = record.get("bundle_path")
    if not isinstance(bundle_path, str) or not bundle_path:
        errors.append(
            "NeMoClaw canary readiness script missing bundle_path: "
            f"{NEMOCLAW_CANARY_READINESS_SCRIPT}"
        )
        return errors
    script_file = bundle_dir / bundle_path
    try:
        text = script_file.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(
            "NeMoClaw canary readiness script is not readable: "
            f"{NEMOCLAW_CANARY_READINESS_SCRIPT}: {exc}"
        )
        return errors
    for label, token in NEMOCLAW_CANARY_READINESS_SCRIPT_SOURCE_TOKENS:
        if token not in text:
            errors.append(
                "NeMoClaw canary readiness script missing source contract "
                f"{label}: {NEMOCLAW_CANARY_READINESS_SCRIPT}: {token}"
            )
    return errors


def validate_nemoclaw_adoption_script_source(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    records = file_records_by_source(manifest)
    record = records.get(source_path_key(NEMOCLAW_ADOPTION_SCRIPT))
    if not isinstance(record, dict):
        return errors
    roles = record.get("roles")
    if not isinstance(roles, list) or not (
        "operator_plan:command_script" in roles
        or "current_gate:remediation_plan:command_script" in roles
    ):
        errors.append(
            "NeMoClaw adoption script missing command-script role: "
            f"{NEMOCLAW_ADOPTION_SCRIPT}"
        )
    bundle_path = record.get("bundle_path")
    if not isinstance(bundle_path, str) or not bundle_path:
        errors.append(
            "NeMoClaw adoption script missing bundle_path: "
            f"{NEMOCLAW_ADOPTION_SCRIPT}"
        )
        return errors
    script_file = bundle_dir / bundle_path
    try:
        text = script_file.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(
            "NeMoClaw adoption script is not readable: "
            f"{NEMOCLAW_ADOPTION_SCRIPT}: {exc}"
        )
        return errors
    for label, token in NEMOCLAW_ADOPTION_SCRIPT_SOURCE_TOKENS:
        if token not in text:
            errors.append(
                "NeMoClaw adoption script missing source contract "
                f"{label}: {NEMOCLAW_ADOPTION_SCRIPT}: {token}"
            )
    return errors


def validate_nemoclaw_canary_readiness_remote_lookup_checks(
    payload: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    checks = payload.get("checks")
    if not isinstance(checks, list):
        return ["NeMoClaw canary readiness checks is not a list"]
    checks_by_name = {
        check.get("name"): check
        for check in checks
        if isinstance(check, dict) and isinstance(check.get("name"), str)
    }
    for name in sorted(REQUIRED_NEMOCLAW_CANARY_REMOTE_LOOKUP_CHECK_NAMES):
        check = checks_by_name.get(name)
        if not isinstance(check, dict):
            errors.append(
                "NeMoClaw canary readiness missing required remote lookup check: "
                f"{name}"
            )
            continue
        if check.get("ok") is not True:
            errors.append(
                "NeMoClaw canary readiness remote lookup check is not ok: "
                f"{name}"
            )
        detail = check.get("detail")
        if isinstance(detail, str) and detail.strip():
            try:
                detail_payload = json.loads(detail)
            except json.JSONDecodeError:
                errors.append(
                    "NeMoClaw canary readiness remote lookup check detail is not JSON: "
                    f"{name}"
                )
            else:
                if detail_payload.get("missing") not in ([], None):
                    errors.append(
                        "NeMoClaw canary readiness remote lookup check has missing "
                        f"deny policy entries: {name}"
                    )
    return errors


def validate_nemoclaw_canary_readiness_runtime_policy_checks(
    payload: dict[str, Any],
) -> list[str]:
    if payload.get("ok") is not True:
        return []
    errors: list[str] = []
    checks = payload.get("checks")
    if not isinstance(checks, list):
        return ["NeMoClaw canary readiness checks is not a list"]
    checks_by_name = {
        check.get("name"): check
        for check in checks
        if isinstance(check, dict) and isinstance(check.get("name"), str)
    }
    required = (
        "NeMoClaw sandbox runtime policy is introspectable: nejumi-taiwan",
        "NeMoClaw W&B/Weave runtime policy is present: nejumi-taiwan",
        "NeMoClaw runtime network policies are allowlisted: nejumi-taiwan",
    )
    for name in required:
        check = checks_by_name.get(name)
        if not isinstance(check, dict):
            errors.append(f"NeMoClaw canary readiness missing runtime policy check: {name}")
            continue
        if check.get("ok") is not True:
            errors.append(f"NeMoClaw canary readiness runtime policy check is not ok: {name}")
        detail = check.get("detail")
        try:
            detail_payload = json.loads(detail) if isinstance(detail, str) else detail
        except json.JSONDecodeError:
            detail_payload = None
        if not isinstance(detail_payload, dict):
            errors.append(f"NeMoClaw canary readiness runtime policy detail is not JSON: {name}")
            continue
        if name.startswith("NeMoClaw W&B/Weave") and detail_payload.get(
            "wandb_weave_policy_present"
        ) is not True:
            errors.append("NeMoClaw canary readiness W&B/Weave policy evidence is not true")
        if name.startswith("NeMoClaw runtime network policies are allowlisted"):
            if detail_payload.get("runtime_network_policy_allowlist_ok") is not True:
                errors.append(
                    "NeMoClaw canary readiness runtime network policy allowlist "
                    "evidence is not true"
                )
            unknown_policies = detail_payload.get("unknown_runtime_network_policies")
            if not isinstance(unknown_policies, list):
                errors.append(
                    "NeMoClaw canary readiness unknown_runtime_network_policies "
                    "is not a list"
                )
            elif unknown_policies:
                errors.append(
                    "NeMoClaw canary readiness has unknown runtime network "
                    f"policies: {unknown_policies}"
                )
    return errors


def validate_nemoclaw_adoption_runtime_policy_criterion(
    payload: dict[str, Any],
) -> list[str]:
    if payload.get("ok") is not True:
        return []
    criteria = payload.get("criteria")
    if not isinstance(criteria, list):
        return ["NeMoClaw adoption JSON criteria is not a list"]
    by_name = {
        row.get("name"): row
        for row in criteria
        if isinstance(row, dict) and isinstance(row.get("name"), str)
    }
    wandb_policy = by_name.get("runtime_wandb_weave_policy")
    if not isinstance(wandb_policy, dict):
        return ["NeMoClaw adoption JSON missing runtime_wandb_weave_policy criterion"]
    if wandb_policy.get("ok") is not True:
        return ["NeMoClaw adoption JSON runtime_wandb_weave_policy criterion is not ok"]
    if wandb_policy.get("wandb_weave_policy_present") is not True:
        return ["NeMoClaw adoption JSON runtime_wandb_weave_policy evidence is not true"]

    allowlist_policy = by_name.get("runtime_network_policy_allowlist")
    if not isinstance(allowlist_policy, dict):
        return ["NeMoClaw adoption JSON missing runtime_network_policy_allowlist criterion"]
    if allowlist_policy.get("ok") is not True:
        return [
            "NeMoClaw adoption JSON runtime_network_policy_allowlist criterion is not ok"
        ]
    if allowlist_policy.get("runtime_network_policy_allowlist_ok") is not True:
        return [
            "NeMoClaw adoption JSON runtime_network_policy_allowlist evidence is not true"
        ]
    unknown_policies = allowlist_policy.get("unknown_runtime_network_policies")
    if not isinstance(unknown_policies, list):
        return [
            "NeMoClaw adoption JSON unknown_runtime_network_policies is not a list"
        ]
    if unknown_policies:
        return [
            "NeMoClaw adoption JSON has unknown runtime network policies: "
            f"{unknown_policies}"
        ]
    return []


def file_records_by_source(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    files = manifest.get("files")
    file_records = files if isinstance(files, list) else []
    result: dict[str, dict[str, Any]] = {}
    for record in file_records:
        if not isinstance(record, dict):
            continue
        source_path = record.get("source_path")
        if isinstance(source_path, str) and source_path:
            result[source_path] = record
    return result


def validate_file_role(
    *,
    errors: list[str],
    records: dict[str, dict[str, Any]],
    path_value: Any,
    role: str,
    label: str,
) -> dict[str, Any] | None:
    key = source_path_key(path_value)
    if not key:
        errors.append(f"{label} path is missing")
        return None
    record = records.get(key)
    if not isinstance(record, dict):
        errors.append(f"missing bundled evidence for {label}: {key}")
        return None
    roles = record.get("roles")
    if not isinstance(roles, list) or role not in roles:
        errors.append(f"bundled evidence for {label} missing role {role}: {key}")
    return record


def validate_wandb_adoption_sync_dry_run_report(
    payload: dict[str, Any],
    *,
    candidate: dict[str, Any],
    benchmark: str,
    scope_attestation_source: dict[str, Any] | None = None,
    records: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    errors: list[str] = []
    label = f"W&B adoption draft candidate {benchmark} sync dry-run report"
    if payload.get("ok") is not True:
        errors.append(f"{label} ok must be true")
    if payload.get("status") != "synced":
        errors.append(f"{label} status must be synced")
    generated_at = payload.get("generated_at")
    if (
        not isinstance(generated_at, (int, float))
        or not math.isfinite(float(generated_at))
        or generated_at <= 0
    ):
        errors.append(f"{label} generated_at must be a positive number")
    if payload.get("dry_run") is not True:
        errors.append(f"{label} dry_run must be true")
    if payload.get("in_place") is not False:
        errors.append(f"{label} in_place must be false")
    output_path = payload.get("output_path")
    if output_path not in ("", None):
        errors.append(f"{label} output_path must be empty for a dry run")
    if source_path_key(payload.get("review_path")) != source_path_key(
        candidate.get("target_review_json")
    ):
        errors.append(f"{label} review_path does not match candidate")
    source_review_sha256 = payload.get("source_review_sha256")
    if not isinstance(source_review_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}",
        source_review_sha256,
    ):
        errors.append(f"{label} source_review_sha256 must be a 64-character lowercase hex digest")
    elif isinstance(records, dict):
        target_review_key = source_path_key(candidate.get("target_review_json"))
        target_review_record = records.get(target_review_key) if target_review_key else None
        if isinstance(target_review_record, dict):
            target_review_sha256 = target_review_record.get("sha256")
            if source_review_sha256 != target_review_sha256:
                errors.append(f"{label} source_review_sha256 does not match bundled target review JSON")
    if payload.get("verify_wandb_completion") is not True:
        errors.append(f"{label} verify_wandb_completion must be true")
    if payload.get("unmatched_count") != 0:
        errors.append(f"{label} unmatched_count must be 0")
    before_status = payload.get("before_status")
    after_status = payload.get("after_status")
    if not isinstance(before_status, str) or not before_status.strip():
        errors.append(f"{label} before_status is missing")
    if not isinstance(after_status, str) or not after_status.strip():
        errors.append(f"{label} after_status is missing")
    if (
        isinstance(before_status, str)
        and before_status.strip()
        and isinstance(after_status, str)
        and after_status.strip()
        and before_status != after_status
    ):
        errors.append(f"{label} before_status and after_status must match")
    if not isinstance(payload.get("entry_count"), int) or payload.get("entry_count") < 1:
        errors.append(f"{label} entry_count must be >= 1")
    if (
        not isinstance(payload.get("adopted_existing_result_count"), int)
        or payload.get("adopted_existing_result_count") < 1
    ):
        errors.append(f"{label} adopted_existing_result_count must be >= 1")
    if not isinstance(payload.get("change_count"), int) or payload.get("change_count") < 1:
        errors.append(f"{label} change_count must be >= 1")

    entries = payload.get("entries")
    if not isinstance(entries, list):
        errors.append(f"{label} entries must be a list")
    elif payload.get("entry_count") != len(entries):
        errors.append(f"{label} entry_count must match entries length")
    else:
        matching_entries = [
            entry
            for entry in entries
            if isinstance(entry, dict)
            and entry.get("benchmark") == candidate.get("benchmark")
            and entry.get("entity") == candidate.get("wandb_entity")
            and entry.get("project") == candidate.get("wandb_project")
            and entry.get("run_id") == candidate.get("wandb_run_id")
            and source_path_key(entry.get("path"))
            == source_path_key(candidate.get("wandb_completion_json"))
        ]
        if not matching_entries:
            errors.append(f"{label} entries must include the candidate W&B completion")
        else:
            entry = matching_entries[0]
            if entry.get("ok") is not True:
                errors.append(f"{label} candidate entry ok must be true")
            if entry.get("adopted_existing_result") is not True:
                errors.append(
                    f"{label} candidate entry adopted_existing_result must be true"
                )
            candidate_sha = candidate.get("wandb_completion_sha256")
            if isinstance(candidate_sha, str) and candidate_sha:
                if entry.get("sha256") != candidate_sha:
                    errors.append(f"{label} candidate entry sha256 does not match candidate")
            scope = entry.get("scope_attestation")
            if not isinstance(scope, dict):
                errors.append(f"{label} candidate entry missing scope_attestation")
            else:
                source_attestation_json = scope.get("source_attestation_json")
                if not isinstance(source_attestation_json, str) or not source_attestation_json.strip():
                    errors.append(
                        f"{label} candidate entry scope_attestation source_attestation_json is missing"
                    )
                elif source_path_key(source_attestation_json) != source_path_key(
                    candidate.get("scope_attestation_template_json")
                ):
                    errors.append(
                        f"{label} candidate entry scope_attestation source_attestation_json "
                        "does not match candidate"
                    )
                source_attestation_sha256 = scope.get("source_attestation_sha256")
                if not isinstance(source_attestation_sha256, str) or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    source_attestation_sha256,
                ):
                    errors.append(
                        f"{label} candidate entry scope_attestation "
                        "source_attestation_sha256 must be a 64-character lowercase hex digest"
                    )
                elif isinstance(records, dict):
                    source_attestation_key = source_path_key(source_attestation_json)
                    source_attestation_record = (
                        records.get(source_attestation_key)
                        if source_attestation_key
                        else None
                    )
                    if isinstance(source_attestation_record, dict):
                        source_attestation_record_sha256 = source_attestation_record.get("sha256")
                        if source_attestation_sha256 != source_attestation_record_sha256:
                            errors.append(
                                f"{label} candidate entry scope_attestation "
                                "source_attestation_sha256 does not match bundled source attestation JSON"
                            )
                errors.extend(
                    validate_scope_attestation_payload(
                        scope,
                        label=f"{label} candidate entry scope_attestation",
                        entry=entry,
                        review_path=candidate.get("target_review_json"),
                    )
                )
                if isinstance(scope_attestation_source, dict):
                    errors.extend(
                        validate_scope_attestation_payload(
                            scope_attestation_source,
                            label=f"{label} source scope_attestation JSON",
                            entry=entry,
                            review_path=candidate.get("target_review_json"),
                        )
                    )
                    compared_fields = (
                        "schema_version",
                        "confirmed",
                        "confirmed_by",
                        "confirmed_at",
                        "confirmation",
                        "completion_sha256",
                        "source_audit_sha256",
                        "benchmark",
                        "entity",
                        "project",
                        "run_id",
                        "actual_cost_estimate",
                        "provider_bill_reference",
                    )
                    for field in compared_fields:
                        if scope_attestation_source.get(field) != scope.get(field):
                            errors.append(
                                f"{label} source scope_attestation JSON {field} "
                                "does not match dry-run entry"
                            )
                    for field in ("review_path", "completion_path", "source_audit_json"):
                        if source_path_key(scope_attestation_source.get(field)) != source_path_key(
                            scope.get(field)
                        ):
                            errors.append(
                                f"{label} source scope_attestation JSON {field} "
                                "does not match dry-run entry"
                            )
                    if (
                        isinstance(source_attestation_sha256, str)
                        and re.fullmatch(r"[0-9a-f]{64}", source_attestation_sha256)
                        and isinstance(records, dict)
                    ):
                        source_attestation_key = source_path_key(
                            scope.get("source_attestation_json")
                        )
                        source_attestation_record = (
                            records.get(source_attestation_key)
                            if source_attestation_key
                            else None
                        )
                        if isinstance(source_attestation_record, dict):
                            if source_attestation_sha256 != source_attestation_record.get("sha256"):
                                errors.append(
                                    f"{label} source scope_attestation JSON sha256 "
                                    "does not match dry-run entry source_attestation_sha256"
                                )

    changes = payload.get("changes")
    if not isinstance(changes, list):
        errors.append(f"{label} changes must be a list")
        return errors
    matching_changes = [
        change
        for change in changes
        if isinstance(change, dict)
        and change.get("target") == "top_level"
        and change.get("benchmark") == candidate.get("benchmark")
        and change.get("run_id") == candidate.get("wandb_run_id")
        and change.get("action") in {"added", "replaced", "kept_existing"}
    ]
    if not matching_changes:
        errors.append(
            f"{label} changes must include top_level action for candidate run"
        )
    return errors


def validate_wandb_adoption_scope_preflight_report(
    payload: dict[str, Any],
    *,
    candidate: dict[str, Any],
    benchmark: str,
    scope_attestation_source: dict[str, Any] | None = None,
    records: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    errors: list[str] = []
    label = f"W&B adoption draft candidate {benchmark} scope preflight report"
    if payload.get("ok") is not True:
        errors.append(f"{label} ok must be true")
    if payload.get("status") != "passed":
        errors.append(f"{label} status must be passed")
    generated_at = payload.get("generated_at")
    if (
        not isinstance(generated_at, (int, float))
        or not math.isfinite(float(generated_at))
        or generated_at <= 0
    ):
        errors.append(f"{label} generated_at must be a positive number")
    report_errors = payload.get("errors")
    if not isinstance(report_errors, list):
        errors.append(f"{label} errors must be a list")
    elif report_errors:
        errors.append(f"{label} errors must be empty")
    if source_path_key(payload.get("review_path")) != source_path_key(
        candidate.get("target_review_json")
    ):
        errors.append(f"{label} review_path does not match candidate")
    if source_path_key(payload.get("completion_json")) != source_path_key(
        candidate.get("wandb_completion_json")
    ):
        errors.append(f"{label} completion_json does not match candidate")
    if source_path_key(payload.get("scope_attestation_json")) != source_path_key(
        candidate.get("scope_attestation_template_json")
    ):
        errors.append(f"{label} scope_attestation_json does not match candidate")

    entry = payload.get("entry")
    scope = payload.get("scope_attestation")
    source_attestation_sha256 = (
        scope.get("source_attestation_sha256")
        if isinstance(scope, dict)
        else None
    )
    source_files = payload.get("source_files")
    if not isinstance(source_files, dict):
        errors.append(f"{label} source_files must be an object")
    else:
        errors.extend(
            validate_source_file_report_row(
                row=source_files.get("review_json"),
                label=f"{label} review_json",
                expected_path=payload.get("review_path"),
                records=records,
            )
        )
        errors.extend(
            validate_source_file_report_row(
                row=source_files.get("completion_json"),
                label=f"{label} completion_json",
                expected_path=payload.get("completion_json"),
                expected_sha256=(
                    entry.get("sha256")
                    if isinstance(entry, dict) and isinstance(entry.get("sha256"), str)
                    else candidate.get("wandb_completion_sha256")
                ),
                records=records,
            )
        )
        errors.extend(
            validate_source_file_report_row(
                row=source_files.get("scope_attestation_json"),
                label=f"{label} scope_attestation_json",
                expected_path=payload.get("scope_attestation_json"),
                expected_sha256=(
                    source_attestation_sha256
                    if isinstance(source_attestation_sha256, str)
                    else None
                ),
                records=records,
            )
        )

    if not isinstance(entry, dict):
        errors.append(f"{label} entry must be an object")
        entry = {}
    else:
        if entry.get("benchmark") != candidate.get("benchmark"):
            errors.append(f"{label} entry benchmark does not match candidate")
        if entry.get("entity") != candidate.get("wandb_entity"):
            errors.append(f"{label} entry entity does not match candidate")
        if entry.get("project") != candidate.get("wandb_project"):
            errors.append(f"{label} entry project does not match candidate")
        if entry.get("run_id") != candidate.get("wandb_run_id"):
            errors.append(f"{label} entry run_id does not match candidate")
        if source_path_key(entry.get("path")) != source_path_key(
            candidate.get("wandb_completion_json")
        ):
            errors.append(f"{label} entry path does not match candidate")
        if entry.get("ok") is not True:
            errors.append(f"{label} entry ok must be true")
        if entry.get("observed_evidence_valid") is not True:
            errors.append(f"{label} entry observed_evidence_valid must be true")
        if entry.get("run_metadata_valid") is not True:
            errors.append(f"{label} entry run_metadata_valid must be true")
        if entry.get("adopted_existing_result") is not False:
            errors.append(f"{label} entry adopted_existing_result must be false")
        if entry.get("query_source_kind") != WANDB_COMPLETION_QUERY_SOURCE_KIND:
            errors.append(
                f"{label} entry query_source_kind must be "
                f"{WANDB_COMPLETION_QUERY_SOURCE_KIND}"
            )
        candidate_sha = candidate.get("wandb_completion_sha256")
        if isinstance(candidate_sha, str) and candidate_sha:
            if entry.get("sha256") != candidate_sha:
                errors.append(f"{label} entry sha256 does not match candidate")

    if not isinstance(scope, dict):
        errors.append(f"{label} scope_attestation must be an object")
        return errors

    source_attestation_json = scope.get("source_attestation_json")
    if not isinstance(source_attestation_json, str) or not source_attestation_json.strip():
        errors.append(f"{label} scope_attestation source_attestation_json is missing")
    elif source_path_key(source_attestation_json) != source_path_key(
        candidate.get("scope_attestation_template_json")
    ):
        errors.append(
            f"{label} scope_attestation source_attestation_json does not match candidate"
        )
    if not isinstance(source_attestation_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}",
        source_attestation_sha256 or "",
    ):
        errors.append(
            f"{label} scope_attestation source_attestation_sha256 must be a "
            "64-character lowercase hex digest"
        )
    elif isinstance(records, dict):
        source_attestation_key = source_path_key(source_attestation_json)
        source_attestation_record = (
            records.get(source_attestation_key)
            if source_attestation_key
            else None
        )
        if isinstance(source_attestation_record, dict):
            if source_attestation_sha256 != source_attestation_record.get("sha256"):
                errors.append(
                    f"{label} scope_attestation source_attestation_sha256 "
                    "does not match bundled source attestation JSON"
                )

    errors.extend(
        validate_scope_attestation_payload(
            scope,
            label=f"{label} scope_attestation",
            entry=entry,
            review_path=candidate.get("target_review_json"),
        )
    )
    if isinstance(scope_attestation_source, dict):
        errors.extend(
            validate_scope_attestation_payload(
                scope_attestation_source,
                label=f"{label} source scope_attestation JSON",
                entry=entry,
                review_path=candidate.get("target_review_json"),
            )
        )
        compared_fields = (
            "schema_version",
            "confirmed",
            "confirmed_by",
            "confirmed_at",
            "confirmation",
            "completion_sha256",
            "source_audit_sha256",
            "benchmark",
            "entity",
            "project",
            "run_id",
            "actual_cost_estimate",
            "provider_bill_reference",
        )
        for field in compared_fields:
            if scope_attestation_source.get(field) != scope.get(field):
                errors.append(
                    f"{label} source scope_attestation JSON {field} "
                    "does not match preflight scope_attestation"
                )
        for field in ("review_path", "completion_path", "source_audit_json"):
            if source_path_key(scope_attestation_source.get(field)) != source_path_key(
                scope.get(field)
            ):
                errors.append(
                    f"{label} source scope_attestation JSON {field} "
                    "does not match preflight scope_attestation"
                )
    return errors


def validate_command_path_option(
    *,
    errors: list[str],
    label: str,
    parts: list[str],
    flag: str,
    expected_path: Any,
) -> None:
    value, present = command_flag_value(parts, flag)
    if not present:
        errors.append(f"{label} requires {flag}")
        return
    if source_path_key(value) != source_path_key(expected_path):
        errors.append(f"{label} {flag} does not match expected path")


def validate_wandb_adoption_scope_render_next_commands(
    next_commands: Any,
    *,
    candidate: dict[str, Any],
    payload: dict[str, Any],
    benchmark: str,
) -> list[str]:
    errors: list[str] = []
    label = f"W&B adoption draft candidate {benchmark} scope render report next_commands"
    if not isinstance(next_commands, dict):
        return [f"{label} must be an object"]

    preflight_command = next_commands.get("preflight")
    if not isinstance(preflight_command, str) or not preflight_command.strip():
        errors.append(f"{label}.preflight must be a non-empty command")
    else:
        preflight_parts = command_parts(preflight_command)
        if "scripts/tools/verify_wandb_scope_attestation.py" not in command_script_paths(
            preflight_command
        ):
            errors.append(
                f"{label}.preflight must invoke scripts/tools/verify_wandb_scope_attestation.py"
            )
        validate_command_path_option(
            errors=errors,
            label=f"{label}.preflight",
            parts=preflight_parts,
            flag="--review-json",
            expected_path=candidate.get("target_review_json"),
        )
        validate_command_path_option(
            errors=errors,
            label=f"{label}.preflight",
            parts=preflight_parts,
            flag="--completion-json",
            expected_path=candidate.get("wandb_completion_json"),
        )
        validate_command_path_option(
            errors=errors,
            label=f"{label}.preflight",
            parts=preflight_parts,
            flag="--scope-attestation-json",
            expected_path=payload.get("output_json"),
        )
        if (
            isinstance(candidate.get("scope_attestation_preflight_report_json"), str)
            and candidate.get("scope_attestation_preflight_report_json", "").strip()
        ):
            validate_command_path_option(
                errors=errors,
                label=f"{label}.preflight",
                parts=preflight_parts,
                flag="--json",
                expected_path=candidate.get("scope_attestation_preflight_report_json"),
            )

    sync_command = next_commands.get("sync_dry_run")
    if not isinstance(sync_command, str) or not sync_command.strip():
        errors.append(f"{label}.sync_dry_run must be a non-empty command")
    else:
        sync_parts = command_parts(sync_command)
        if "scripts/tools/sync_wandb_completion_to_paid_review.py" not in command_script_paths(
            sync_command
        ):
            errors.append(
                f"{label}.sync_dry_run must invoke "
                "scripts/tools/sync_wandb_completion_to_paid_review.py"
            )
        if "--in-place" in sync_parts:
            errors.append(f"{label}.sync_dry_run must not include --in-place")
        for flag in (
            "--set-verify-wandb-completion",
            "--top-level",
            "--adopt-existing-result",
        ):
            if flag not in sync_parts:
                errors.append(f"{label}.sync_dry_run requires {flag}")
        validate_command_path_option(
            errors=errors,
            label=f"{label}.sync_dry_run",
            parts=sync_parts,
            flag="--review-json",
            expected_path=candidate.get("target_review_json"),
        )
        validate_command_path_option(
            errors=errors,
            label=f"{label}.sync_dry_run",
            parts=sync_parts,
            flag="--completion-json",
            expected_path=candidate.get("wandb_completion_json"),
        )
        validate_command_path_option(
            errors=errors,
            label=f"{label}.sync_dry_run",
            parts=sync_parts,
            flag="--scope-attestation-json",
            expected_path=payload.get("output_json"),
        )
        if (
            isinstance(candidate.get("sync_dry_run_report_json"), str)
            and candidate.get("sync_dry_run_report_json", "").strip()
        ):
            validate_command_path_option(
                errors=errors,
                label=f"{label}.sync_dry_run",
                parts=sync_parts,
                flag="--report-json",
                expected_path=candidate.get("sync_dry_run_report_json"),
            )
    return errors


def validate_required_command_value(
    *,
    errors: list[str],
    label: str,
    parts: list[str],
    flag: str,
) -> None:
    value, present = command_flag_value(parts, flag)
    if not present:
        errors.append(f"{label} requires {flag}")
        return
    if value is None or not str(value).strip():
        errors.append(f"{label} {flag} requires a non-empty value")


def validate_wandb_adoption_scope_render_command(
    *,
    candidate: dict[str, Any],
    benchmark: str,
) -> list[str]:
    errors: list[str] = []
    label = f"W&B adoption draft candidate {benchmark} scope_attestation_render_command"
    command = candidate.get("scope_attestation_render_command")
    required = (
        candidate.get("sync_ready") is True
        or (
            isinstance(candidate.get("scope_attestation_render_report_json"), str)
            and candidate.get("scope_attestation_render_report_json", "").strip()
        )
        or (
            isinstance(candidate.get("scope_attestation_render_markdown"), str)
            and candidate.get("scope_attestation_render_markdown", "").strip()
        )
    )
    if not isinstance(command, str) or not command.strip():
        if required:
            errors.append(f"{label} must be a non-empty command")
        return errors

    parts = command_parts(command)
    if "scripts/tools/render_wandb_scope_attestation.py" not in command_script_paths(command):
        errors.append(f"{label} must invoke scripts/tools/render_wandb_scope_attestation.py")
    deprecated = [flag for flag in DEPRECATED_WANDB_ADOPTION_FLAGS if flag in parts]
    if deprecated:
        errors.append(
            f"{label} uses deprecated W&B adoption flag(s): "
            + ", ".join(deprecated)
            + "; use --scope-attestation-json in sync commands"
        )
    for flag in (
        "--confirmed-by",
        "--confirmed-at",
        "--confirmation",
        "--actual-cost-estimate",
        "--provider-bill-reference",
    ):
        validate_required_command_value(
            errors=errors,
            label=label,
            parts=parts,
            flag=flag,
        )
    validate_command_path_option(
        errors=errors,
        label=label,
        parts=parts,
        flag="--template-json",
        expected_path=candidate.get("scope_attestation_template_json"),
    )
    validate_command_path_option(
        errors=errors,
        label=label,
        parts=parts,
        flag="--output-json",
        expected_path=candidate.get("scope_attestation_template_json"),
    )
    if (
        isinstance(candidate.get("scope_attestation_render_report_json"), str)
        and candidate.get("scope_attestation_render_report_json", "").strip()
    ):
        validate_command_path_option(
            errors=errors,
            label=label,
            parts=parts,
            flag="--report-json",
            expected_path=candidate.get("scope_attestation_render_report_json"),
        )
    if (
        isinstance(candidate.get("scope_attestation_render_markdown"), str)
        and candidate.get("scope_attestation_render_markdown", "").strip()
    ):
        validate_command_path_option(
            errors=errors,
            label=label,
            parts=parts,
            flag="--markdown",
            expected_path=candidate.get("scope_attestation_render_markdown"),
        )
    if (
        isinstance(candidate.get("scope_attestation_preflight_report_json"), str)
        and candidate.get("scope_attestation_preflight_report_json", "").strip()
    ):
        validate_command_path_option(
            errors=errors,
            label=label,
            parts=parts,
            flag="--preflight-report-json",
            expected_path=candidate.get("scope_attestation_preflight_report_json"),
        )
    if (
        isinstance(candidate.get("sync_dry_run_report_json"), str)
        and candidate.get("sync_dry_run_report_json", "").strip()
    ):
        validate_command_path_option(
            errors=errors,
            label=label,
            parts=parts,
            flag="--sync-dry-run-report-json",
            expected_path=candidate.get("sync_dry_run_report_json"),
        )
    return errors


def validate_wandb_adoption_scope_render_report(
    payload: dict[str, Any],
    *,
    candidate: dict[str, Any],
    benchmark: str,
    scope_attestation_source: dict[str, Any] | None = None,
    records: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    errors: list[str] = []
    label = f"W&B adoption draft candidate {benchmark} scope render report"
    if payload.get("schema_version") != 1:
        errors.append(f"{label} schema_version must be 1")
    if payload.get("ok") is not True:
        errors.append(f"{label} ok must be true")
    if payload.get("status") != "rendered":
        errors.append(f"{label} status must be rendered")
    generated_at = payload.get("generated_at")
    if (
        not isinstance(generated_at, (int, float))
        or not math.isfinite(float(generated_at))
        or generated_at <= 0
    ):
        errors.append(f"{label} generated_at must be a positive number")
    report_errors = payload.get("errors")
    if not isinstance(report_errors, list):
        errors.append(f"{label} errors must be a list")
    elif report_errors:
        errors.append(f"{label} errors must be empty")
    if payload.get("will_execute_external_actions") is not False:
        errors.append(f"{label} will_execute_external_actions must be false")
    safety = payload.get("safety")
    if not isinstance(safety, dict):
        errors.append(f"{label} safety must be an object")
    else:
        for field in SCOPE_ATTESTATION_RENDER_SAFETY_FIELDS:
            if safety.get(field) is not False:
                errors.append(f"{label} safety {field} must be false")

    if source_path_key(payload.get("template_json")) != source_path_key(
        candidate.get("scope_attestation_template_json")
    ):
        errors.append(f"{label} template_json does not match candidate")
    if source_path_key(payload.get("output_json")) != source_path_key(
        candidate.get("scope_attestation_template_json")
    ):
        errors.append(f"{label} output_json does not match candidate")
    if source_path_key(payload.get("review_path")) != source_path_key(
        candidate.get("target_review_json")
    ):
        errors.append(f"{label} review_path does not match candidate")
    if source_path_key(payload.get("completion_path")) != source_path_key(
        candidate.get("wandb_completion_json")
    ):
        errors.append(f"{label} completion_path does not match candidate")
    if payload.get("benchmark") != candidate.get("benchmark"):
        errors.append(f"{label} benchmark does not match candidate")
    if payload.get("entity") != candidate.get("wandb_entity"):
        errors.append(f"{label} entity does not match candidate")
    if payload.get("project") != candidate.get("wandb_project"):
        errors.append(f"{label} project does not match candidate")
    if payload.get("run_id") != candidate.get("wandb_run_id"):
        errors.append(f"{label} run_id does not match candidate")
    candidate_completion_sha = candidate.get("wandb_completion_sha256")
    if isinstance(candidate_completion_sha, str) and candidate_completion_sha:
        if payload.get("completion_sha256") != candidate_completion_sha:
            errors.append(f"{label} completion_sha256 does not match candidate")
    if source_path_key(payload.get("source_audit_json")) != source_path_key(
        candidate.get("source_audit_json")
    ):
        errors.append(f"{label} source_audit_json does not match candidate")
    candidate_source_audit_sha = candidate.get("source_audit_sha256")
    if isinstance(candidate_source_audit_sha, str) and candidate_source_audit_sha:
        if payload.get("source_audit_sha256") != candidate_source_audit_sha:
            errors.append(f"{label} source_audit_sha256 does not match candidate")

    for field in ("template_sha256", "output_sha256"):
        value = payload.get(field)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value or ""):
            errors.append(f"{label} {field} must be a 64-character lowercase hex digest")

    if isinstance(records, dict):
        output_key = source_path_key(payload.get("output_json"))
        output_record = records.get(output_key) if output_key else None
        if isinstance(output_record, dict) and isinstance(payload.get("output_sha256"), str):
            if payload.get("output_sha256") != output_record.get("sha256"):
                errors.append(f"{label} output_sha256 does not match bundled output JSON")

    if isinstance(scope_attestation_source, dict):
        entry = {
            "benchmark": candidate.get("benchmark"),
            "entity": candidate.get("wandb_entity"),
            "project": candidate.get("wandb_project"),
            "run_id": candidate.get("wandb_run_id"),
            "path": candidate.get("wandb_completion_json"),
            "sha256": candidate.get("wandb_completion_sha256"),
        }
        errors.extend(
            validate_scope_attestation_payload(
                scope_attestation_source,
                label=f"{label} output scope_attestation JSON",
                entry=entry,
                review_path=candidate.get("target_review_json"),
            )
        )
        rendered = scope_attestation_source.get("rendered_scope_attestation")
        if not isinstance(rendered, dict):
            errors.append(f"{label} output rendered_scope_attestation must be an object")
        else:
            if rendered.get("schema_version") != 1:
                errors.append(f"{label} output rendered_scope_attestation schema_version must be 1")
            if source_path_key(rendered.get("source_template_json")) != source_path_key(
                payload.get("template_json")
            ):
                errors.append(
                    f"{label} output rendered_scope_attestation source_template_json "
                    "does not match report"
                )
            if rendered.get("source_template_sha256") != payload.get("template_sha256"):
                errors.append(
                    f"{label} output rendered_scope_attestation source_template_sha256 "
                    "does not match report"
                )
            if source_path_key(rendered.get("output_json")) != source_path_key(
                payload.get("output_json")
            ):
                errors.append(
                    f"{label} output rendered_scope_attestation output_json does not match report"
                )
            if rendered.get("will_execute_external_actions") is not False:
                errors.append(
                    f"{label} output rendered_scope_attestation "
                    "will_execute_external_actions must be false"
                )
            rendered_safety = rendered.get("safety")
            if not isinstance(rendered_safety, dict):
                errors.append(f"{label} output rendered_scope_attestation safety must be an object")
            else:
                for field in SCOPE_ATTESTATION_RENDER_SAFETY_FIELDS:
                    if rendered_safety.get(field) is not False:
                        errors.append(
                            f"{label} output rendered_scope_attestation safety {field} "
                            "must be false"
                        )
    errors.extend(
        validate_wandb_adoption_scope_render_next_commands(
            payload.get("next_commands"),
            candidate=candidate,
            payload=payload,
            benchmark=benchmark,
        )
    )
    return errors


def validate_wandb_adoption_scope_render_markdown(
    markdown: str,
    *,
    report_payload: dict[str, Any],
    benchmark: str,
) -> list[str]:
    errors: list[str] = []
    label = f"W&B adoption draft candidate {benchmark} scope render Markdown"
    if not isinstance(markdown, str) or not markdown.strip():
        return [f"{label} must be non-empty"]
    expected_snippets = [
        "# W&B Scope Attestation Render Report",
        f"Status: `{report_payload.get('status')}`",
        f"OK: `{report_payload.get('ok')}`",
        (
            "Will execute external actions: "
            f"`{report_payload.get('will_execute_external_actions')}`"
        ),
        "| Field | Value |",
        "|---|---|",
        "## Next Commands",
    ]
    for field in (
        "template_json",
        "template_sha256",
        "output_json",
        "output_sha256",
        "review_path",
        "completion_path",
        "completion_sha256",
        "source_audit_json",
        "source_audit_sha256",
        "benchmark",
        "entity",
        "project",
        "run_id",
    ):
        expected_snippets.append(
            f"| {field} | {markdown_cell(report_payload.get(field))} |"
        )
    next_commands = report_payload.get("next_commands")
    if not isinstance(next_commands, dict):
        errors.append(f"{label} source report next_commands must be an object")
    else:
        for command_name in ("preflight", "sync_dry_run"):
            command = next_commands.get(command_name)
            if not isinstance(command, str) or not command.strip():
                errors.append(f"{label} source report next_commands.{command_name} is missing")
                continue
            expected_snippets.extend(
                [
                    f"### {command_name}",
                    "```bash",
                    command,
                ]
            )
    for snippet in expected_snippets:
        if snippet not in markdown:
            errors.append(f"{label} missing expected snippet: {snippet}")
    return errors


def validate_source_file_report_row(
    *,
    row: Any,
    label: str,
    expected_path: Any,
    expected_sha256: str | None = None,
    records: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(row, dict):
        return [f"{label} source file row must be an object"]
    if source_path_key(row.get("path")) != source_path_key(expected_path):
        errors.append(f"{label} source file path does not match expected path")
    if row.get("readable") is not True:
        errors.append(f"{label} source file readable must be true")
    sha256 = row.get("sha256")
    if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
        errors.append(f"{label} source file sha256 must be a 64-character lowercase hex digest")
    else:
        if isinstance(expected_sha256, str) and expected_sha256 and sha256 != expected_sha256:
            errors.append(f"{label} source file sha256 does not match expected sha256")
        if isinstance(records, dict):
            expected_key = source_path_key(expected_path)
            expected_record = records.get(expected_key) if expected_key else None
            if isinstance(expected_record, dict) and sha256 != expected_record.get("sha256"):
                errors.append(f"{label} source file sha256 does not match bundled file")
    return errors


def validate_wandb_adoption_unconfirmed_report_payload(
    payload: dict[str, Any],
    *,
    candidate: dict[str, Any],
    record: dict[str, Any],
    kind: str,
    records: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    errors: list[str] = []
    benchmark = str(record.get("benchmark") or candidate.get("benchmark") or "unknown")
    run_id = str(record.get("run_id") or candidate.get("wandb_run_id") or "unknown")
    label = f"W&B adoption unconfirmed {kind} report {benchmark}/{run_id}"

    if payload.get("ok") is not False:
        errors.append(f"{label} ok must be false")
    schema_version = payload.get("schema_version")
    if schema_version is not None and schema_version != 1:
        errors.append(f"{label} schema_version must be 1")
    if payload.get("status") != "validation_failed":
        errors.append(f"{label} status must be validation_failed")
    generated_at = payload.get("generated_at")
    if (
        not isinstance(generated_at, (int, float))
        or not math.isfinite(float(generated_at))
        or generated_at <= 0
    ):
        errors.append(f"{label} generated_at must be a positive number")
    report_errors = payload.get("errors")
    if not isinstance(report_errors, list) or not report_errors:
        errors.append(f"{label} errors must be a non-empty list")
    if source_path_key(payload.get("review_path")) != source_path_key(
        candidate.get("target_review_json")
    ):
        errors.append(f"{label} review_path does not match candidate")

    template_path = record.get("scope_attestation_template_json")
    if kind == "preflight":
        if source_path_key(payload.get("completion_json")) != source_path_key(
            candidate.get("wandb_completion_json")
        ):
            errors.append(f"{label} completion_json does not match candidate")
        if source_path_key(payload.get("scope_attestation_json")) != source_path_key(
            template_path
        ):
            errors.append(f"{label} scope_attestation_json does not match record")
        entry = payload.get("entry")
        if not isinstance(entry, dict):
            errors.append(f"{label} entry must be an object")
        else:
            if entry.get("benchmark") != candidate.get("benchmark"):
                errors.append(f"{label} entry benchmark does not match candidate")
            expected_entity = candidate.get("wandb_entity") or candidate.get("entity")
            expected_project = candidate.get("wandb_project") or candidate.get("project")
            if isinstance(expected_entity, str) and entry.get("entity") != expected_entity:
                errors.append(f"{label} entry entity does not match candidate")
            if isinstance(expected_project, str) and entry.get("project") != expected_project:
                errors.append(f"{label} entry project does not match candidate")
            if entry.get("run_id") != candidate.get("wandb_run_id"):
                errors.append(f"{label} entry run_id does not match candidate")
            if source_path_key(entry.get("path")) != source_path_key(
                candidate.get("wandb_completion_json")
            ):
                errors.append(f"{label} entry path does not match candidate")
            if entry.get("adopted_existing_result") is not False:
                errors.append(f"{label} entry adopted_existing_result must be false")
            if entry.get("query_source_kind") != WANDB_COMPLETION_QUERY_SOURCE_KIND:
                errors.append(
                    f"{label} entry query_source_kind must be "
                    f"{WANDB_COMPLETION_QUERY_SOURCE_KIND}"
                )
            source_files = payload.get("source_files")
            if not isinstance(source_files, dict):
                errors.append(f"{label} source_files must be an object")
            else:
                errors.extend(
                    validate_source_file_report_row(
                        row=source_files.get("review_json"),
                        label=f"{label} review_json",
                        expected_path=payload.get("review_path"),
                        records=records,
                    )
                )
                errors.extend(
                    validate_source_file_report_row(
                        row=source_files.get("completion_json"),
                        label=f"{label} completion_json",
                        expected_path=payload.get("completion_json"),
                        expected_sha256=entry.get("sha256")
                        if isinstance(entry.get("sha256"), str)
                        else candidate.get("wandb_completion_sha256"),
                        records=records,
                    )
                )
                errors.extend(
                    validate_source_file_report_row(
                        row=source_files.get("scope_attestation_json"),
                        label=f"{label} scope_attestation_json",
                        expected_path=payload.get("scope_attestation_json"),
                        records=records,
                    )
                )
        if payload.get("scope_attestation") is not None:
            errors.append(f"{label} scope_attestation must be null for invalid template")
    elif kind == "sync dry-run":
        if payload.get("dry_run") is not True:
            errors.append(f"{label} dry_run must be true")
        if payload.get("in_place") is not False:
            errors.append(f"{label} in_place must be false")
        if payload.get("output_path") not in ("", None):
            errors.append(f"{label} output_path must be empty")
        if payload.get("entry_count") != 0:
            errors.append(f"{label} entry_count must be 0")
        if payload.get("adopted_existing_result_count") != 0:
            errors.append(f"{label} adopted_existing_result_count must be 0")
        if payload.get("change_count") != 0:
            errors.append(f"{label} change_count must be 0")
        if payload.get("unmatched_count") != 0:
            errors.append(f"{label} unmatched_count must be 0")
        for field in ("entries", "changes", "unmatched_entries"):
            value = payload.get(field)
            if not isinstance(value, list) or value:
                errors.append(f"{label} {field} must be an empty list")
        if payload.get("verify_wandb_completion") is not False:
            errors.append(f"{label} verify_wandb_completion must be false")
    else:
        errors.append(f"{label} has unknown report kind")

    return errors


def validate_wandb_adoption_unconfirmed_checks_evidence(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    current_gate = manifest.get("current_gate")
    if not isinstance(current_gate, dict):
        return errors
    runner_evidence = current_gate.get("runner_evidence")
    if not isinstance(runner_evidence, dict):
        return errors
    checks = runner_evidence.get("wandb_adoption_unconfirmed_checks")
    if not isinstance(checks, dict) or bool(checks.get("skipped")):
        return errors

    records = file_records_by_source(manifest)
    check_records = checks.get("records")
    if not isinstance(check_records, list):
        errors.append("W&B adoption unconfirmed checks records must be a list")
        return errors
    if checks.get("record_count") != len(check_records):
        errors.append("W&B adoption unconfirmed checks record_count must match records length")
    if check_records and checks.get("ok") is not True:
        errors.append("W&B adoption unconfirmed checks ok must be true when records exist")
    if check_records and checks.get("status") != "passed":
        errors.append("W&B adoption unconfirmed checks status must be passed when records exist")

    draft = current_gate.get("wandb_adoption_draft")
    candidates = draft.get("candidates") if isinstance(draft, dict) else []
    candidates_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            benchmark = candidate.get("benchmark")
            run_id = candidate.get("wandb_run_id")
            if isinstance(benchmark, str) and isinstance(run_id, str):
                candidates_by_key[(benchmark, run_id)] = candidate

    for index, record in enumerate(check_records, start=1):
        if not isinstance(record, dict):
            errors.append(f"W&B adoption unconfirmed check record #{index} must be an object")
            continue
        benchmark = record.get("benchmark")
        run_id = record.get("run_id")
        if not isinstance(benchmark, str) or not benchmark.strip():
            errors.append(f"W&B adoption unconfirmed check record #{index} benchmark is missing")
            benchmark = f"record_{index}"
        if not isinstance(run_id, str) or not run_id.strip():
            errors.append(f"W&B adoption unconfirmed check record #{index} run_id is missing")
            run_id = f"run_{index}"
        role_prefix = f"wandb_adoption_unconfirmed_checks:{benchmark}:{run_id}"
        candidate = candidates_by_key.get((str(benchmark), str(run_id)))
        if not isinstance(candidate, dict):
            errors.append(
                f"W&B adoption unconfirmed check {benchmark}/{run_id} "
                "does not match a W&B adoption draft candidate"
            )
            candidate = {}

        if record.get("preflight_failed_as_expected") is not True:
            errors.append(
                f"W&B adoption unconfirmed check {benchmark}/{run_id} "
                "preflight_failed_as_expected must be true"
            )
        if record.get("sync_dry_run_failed_as_expected") is not True:
            errors.append(
                f"W&B adoption unconfirmed check {benchmark}/{run_id} "
                "sync_dry_run_failed_as_expected must be true"
            )
        if not isinstance(record.get("preflight_returncode"), int) or record.get(
            "preflight_returncode"
        ) == 0:
            errors.append(
                f"W&B adoption unconfirmed check {benchmark}/{run_id} "
                "preflight_returncode must be non-zero"
            )
        if not isinstance(record.get("sync_dry_run_returncode"), int) or record.get(
            "sync_dry_run_returncode"
        ) == 0:
            errors.append(
                f"W&B adoption unconfirmed check {benchmark}/{run_id} "
                "sync_dry_run_returncode must be non-zero"
            )
        if record.get("preflight_status") != "validation_failed":
            errors.append(
                f"W&B adoption unconfirmed check {benchmark}/{run_id} "
                "preflight_status must be validation_failed"
            )
        if record.get("sync_dry_run_status") != "validation_failed":
            errors.append(
                f"W&B adoption unconfirmed check {benchmark}/{run_id} "
                "sync_dry_run_status must be validation_failed"
            )
        for field in (
            "will_query_wandb",
            "will_write_wandb",
            "will_launch_model_inference",
            "will_mutate_review_json",
        ):
            if record.get(field) is not False:
                errors.append(
                    f"W&B adoption unconfirmed check {benchmark}/{run_id} "
                    f"{field} must be false"
                )

        template_record = validate_file_role(
            errors=errors,
            records=records,
            path_value=record.get("scope_attestation_template_json"),
            role=f"{role_prefix}:scope_attestation_template_json",
            label=(
                f"W&B adoption unconfirmed check {benchmark}/{run_id} "
                "scope attestation template"
            ),
        )
        preflight_record = validate_file_role(
            errors=errors,
            records=records,
            path_value=record.get("preflight_report_json"),
            role=f"{role_prefix}:preflight_report_json",
            label=f"W&B adoption unconfirmed check {benchmark}/{run_id} preflight report",
        )
        sync_record = validate_file_role(
            errors=errors,
            records=records,
            path_value=record.get("sync_dry_run_report_json"),
            role=f"{role_prefix}:sync_dry_run_report_json",
            label=f"W&B adoption unconfirmed check {benchmark}/{run_id} sync dry-run report",
        )

        if candidate and source_path_key(
            record.get("scope_attestation_template_json")
        ) != source_path_key(candidate.get("scope_attestation_template_json")):
            errors.append(
                f"W&B adoption unconfirmed check {benchmark}/{run_id} "
                "scope_attestation_template_json does not match candidate"
            )

        if isinstance(template_record, dict):
            bundle_path = template_record.get("bundle_path")
            if isinstance(bundle_path, str) and bundle_path:
                try:
                    template_payload = read_json_object(bundle_dir / bundle_path)
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    errors.append(
                        "W&B adoption unconfirmed check "
                        f"{benchmark}/{run_id} scope attestation template is not readable: {exc}"
                    )
                    template_payload = {}
                if template_payload:
                    if template_payload.get("confirmed") is True:
                        errors.append(
                            f"W&B adoption unconfirmed check {benchmark}/{run_id} "
                            "template confirmed must not be true"
                        )
                    for field in ("actual_cost_estimate", "provider_bill_reference"):
                        value = template_payload.get(field)
                        if not isinstance(value, str) or not value.strip():
                            errors.append(
                                f"W&B adoption unconfirmed check {benchmark}/{run_id} "
                                f"template {field} is missing"
                            )
                        elif not accounting_value_placeholder(value):
                            errors.append(
                                f"W&B adoption unconfirmed check {benchmark}/{run_id} "
                                f"template {field} must still be a placeholder"
                            )

        for payload_record, kind in (
            (preflight_record, "preflight"),
            (sync_record, "sync dry-run"),
        ):
            if not isinstance(payload_record, dict):
                continue
            bundle_path = payload_record.get("bundle_path")
            if not isinstance(bundle_path, str) or not bundle_path:
                errors.append(
                    f"W&B adoption unconfirmed check {benchmark}/{run_id} "
                    f"{kind} report missing bundle_path"
                )
                continue
            try:
                payload = read_json_object(bundle_dir / bundle_path)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                errors.append(
                    f"W&B adoption unconfirmed check {benchmark}/{run_id} "
                    f"{kind} report is not readable: {exc}"
                )
                continue
            errors.extend(
                validate_wandb_adoption_unconfirmed_report_payload(
                    payload,
                    candidate=candidate,
                    record=record,
                    kind=kind,
                    records=records,
                )
            )

    return errors


def normalize_source_path_list(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return [source_path_key(item) for item in value]


def validate_weave_agents_adoption_failure_report_payload(
    payload: dict[str, Any],
    *,
    record: dict[str, Any],
    label: str,
) -> list[str]:
    errors: list[str] = []
    if payload.get("ok") is not False:
        errors.append(f"{label} ok must be false")
    if payload.get("status") != "validation_failed":
        errors.append(f"{label} status must be validation_failed")
    generated_at = payload.get("generated_at")
    if (
        not isinstance(generated_at, (int, float))
        or not math.isfinite(float(generated_at))
        or generated_at <= 0
    ):
        errors.append(f"{label} generated_at must be a positive number")
    if source_path_key(payload.get("review_path")) != source_path_key(record.get("review_path")):
        errors.append(f"{label} review_path does not match current_gate record")
    if payload.get("output_path") not in ("", None):
        errors.append(f"{label} output_path must be empty for a rejected dry-run")
    if payload.get("in_place") is not False:
        errors.append(f"{label} in_place must be false")
    if payload.get("dry_run") is not True:
        errors.append(f"{label} dry_run must be true")
    if payload.get("entry_count") != 0:
        errors.append(f"{label} entry_count must be 0")
    if payload.get("change_count") != 0:
        errors.append(f"{label} change_count must be 0")
    for field in ("entries", "changes", "unmatched_entries"):
        if payload.get(field) != []:
            errors.append(f"{label} {field} must be an empty list")

    payload_completion_paths = normalize_source_path_list(payload.get("completion_paths"))
    record_completion_paths = normalize_source_path_list(record.get("completion_paths"))
    if payload_completion_paths is None:
        errors.append(f"{label} completion_paths must be a string list")
    elif record_completion_paths is None:
        errors.append(f"{label} current_gate record completion_paths must be a string list")
    elif payload_completion_paths != record_completion_paths:
        errors.append(f"{label} completion_paths do not match current_gate record")

    payload_errors = payload.get("validation_errors")
    record_errors = record.get("validation_errors")
    if not isinstance(payload_errors, list) or not payload_errors:
        errors.append(f"{label} validation_errors must be a non-empty list")
    elif not all(isinstance(item, str) and item.strip() for item in payload_errors):
        errors.append(f"{label} validation_errors must contain non-empty strings")
    if not isinstance(record_errors, list) or not record_errors:
        errors.append(f"{label} current_gate record validation_errors must be a non-empty list")
    elif payload_errors != record_errors:
        errors.append(f"{label} validation_errors do not match current_gate record")
    if isinstance(payload_errors, list) and not any(
        "not a passing Weave Agents verifier JSON" in str(item)
        for item in payload_errors
    ):
        errors.append(
            f"{label} validation_errors must explain that the Weave Agents verifier was not passing"
        )
    return errors


def validate_weave_agents_rejected_completion_payload(
    payload: dict[str, Any],
    *,
    label: str,
) -> list[str]:
    errors: list[str] = []
    if payload.get("ok") is not False:
        errors.append(f"{label} ok must be false")
    schema_version = payload.get("schema_version")
    if schema_version is not None and schema_version != 1:
        errors.append(f"{label} schema_version must be 1")
    if payload.get("verification_schema_version") != 1:
        errors.append(f"{label} verification_schema_version must be 1")
    query_source = payload.get("query_source")
    if not isinstance(query_source, dict):
        errors.append(f"{label} query_source must be an object")
    else:
        if query_source.get("kind") != "wandb_agents_api":
            errors.append(f"{label} query_source.kind must be wandb_agents_api")
        matching_span_count = query_source.get("matching_span_count")
        if not isinstance(matching_span_count, int):
            errors.append(f"{label} query_source.matching_span_count must be an integer")
        elif matching_span_count != 0:
            errors.append(f"{label} query_source.matching_span_count must be 0")
    checks = payload.get("checks")
    if not isinstance(checks, list) or not checks:
        errors.append(f"{label} checks must be a non-empty list")
    elif not any(isinstance(check, dict) and check.get("ok") is False for check in checks):
        errors.append(f"{label} checks must contain a failing check")
    return errors


def validate_weave_agents_adoption_validation_failures_evidence(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    current_gate = manifest.get("current_gate")
    if not isinstance(current_gate, dict):
        return errors
    runner_evidence = current_gate.get("runner_evidence")
    if not isinstance(runner_evidence, dict):
        return errors
    failures = runner_evidence.get("weave_agents_adoption_validation_failures")
    if not isinstance(failures, dict):
        return errors
    records = failures.get("records")
    if records is None and failures.get("record_count") in (0, None):
        return errors
    if not isinstance(records, list):
        errors.append("Weave Agents adoption validation failures records must be a list")
        return errors
    if failures.get("record_count") != len(records):
        errors.append(
            "Weave Agents adoption validation failures record_count must match records length"
        )
    if records:
        if failures.get("ok") is not False:
            errors.append("Weave Agents adoption validation failures ok must be false")
        if failures.get("status") != "validation_failed_reports_present":
            errors.append(
                "Weave Agents adoption validation failures status must be "
                "validation_failed_reports_present"
            )

    file_records = file_records_by_source(manifest)
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            errors.append(f"Weave Agents adoption validation failure record #{index} must be an object")
            continue
        label = f"Weave Agents adoption validation failure #{index}"
        if record.get("ok") is not False:
            errors.append(f"{label} ok must be false")
        if record.get("status") != "validation_failed":
            errors.append(f"{label} status must be validation_failed")
        if not isinstance(record.get("review_path"), str) or not record.get("review_path"):
            errors.append(f"{label} review_path is missing")
        if record.get("dry_run") is not True:
            errors.append(f"{label} dry_run must be true")
        if record.get("in_place") is not False:
            errors.append(f"{label} in_place must be false")
        if record.get("entry_count") != 0:
            errors.append(f"{label} entry_count must be 0")
        if record.get("change_count") != 0:
            errors.append(f"{label} change_count must be 0")

        report_record = validate_file_role(
            errors=errors,
            records=file_records,
            path_value=record.get("path"),
            role=f"weave_agents_adoption_validation_failure:{index}:report_json",
            label=f"{label} report",
        )
        completion_paths = record.get("completion_paths")
        if not isinstance(completion_paths, list) or not completion_paths:
            errors.append(f"{label} completion_paths must be a non-empty list")
            completion_paths = []
        elif not all(isinstance(path, str) and path.strip() for path in completion_paths):
            errors.append(f"{label} completion_paths must contain non-empty strings")
            completion_paths = []
        validation_errors = record.get("validation_errors")
        if not isinstance(validation_errors, list) or not validation_errors:
            errors.append(f"{label} validation_errors must be a non-empty list")
        elif not all(isinstance(item, str) and item.strip() for item in validation_errors):
            errors.append(f"{label} validation_errors must contain non-empty strings")

        if isinstance(report_record, dict):
            bundle_path = report_record.get("bundle_path")
            if not isinstance(bundle_path, str) or not bundle_path:
                errors.append(f"{label} report missing bundle_path")
            else:
                try:
                    report_payload = read_json_object(bundle_dir / bundle_path)
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    errors.append(f"{label} report is not readable: {exc}")
                else:
                    errors.extend(
                        validate_weave_agents_adoption_failure_report_payload(
                            report_payload,
                            record=record,
                            label=f"{label} report",
                        )
                    )

        for completion_path in completion_paths:
            completion_record = validate_file_role(
                errors=errors,
                records=file_records,
                path_value=completion_path,
                role=f"weave_agents_adoption_validation_failure:{index}:completion_json",
                label=f"{label} rejected completion",
            )
            if not isinstance(completion_record, dict):
                continue
            bundle_path = completion_record.get("bundle_path")
            if not isinstance(bundle_path, str) or not bundle_path:
                errors.append(f"{label} rejected completion missing bundle_path")
                continue
            try:
                completion_payload = read_json_object(bundle_dir / bundle_path)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"{label} rejected completion is not readable: {exc}")
                continue
            errors.extend(
                validate_weave_agents_rejected_completion_payload(
                    completion_payload,
                    label=f"{label} rejected completion {completion_path}",
                )
            )
    return errors


def validate_any_file_role(
    *,
    errors: list[str],
    records: dict[str, dict[str, Any]],
    path_value: Any,
    role_suffix: str,
    label: str,
) -> dict[str, Any] | None:
    key = source_path_key(path_value)
    if not key:
        errors.append(f"{label} path is missing")
        return None
    record = records.get(key)
    if not isinstance(record, dict):
        errors.append(f"missing bundled evidence for {label}: {key}")
        return None
    roles = record.get("roles")
    if not isinstance(roles, list) or not any(str(role).endswith(role_suffix) for role in roles):
        errors.append(f"bundled evidence for {label} missing role suffix {role_suffix}: {key}")
    return record


def command_parts(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def validate_nemoclaw_installer_review_evidence(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    current_gate = manifest.get("current_gate")
    if not isinstance(current_gate, dict):
        return errors
    remediation_plan = current_gate.get("remediation_plan")
    has_install_command = False
    if isinstance(remediation_plan, list):
        for row in remediation_plan:
            if not isinstance(row, dict) or not isinstance(row.get("commands"), list):
                continue
            for command in row.get("commands", []):
                if not isinstance(command, str):
                    continue
                if command_installs_nemoclaw(command_parts(command)):
                    has_install_command = True
                    break
    summary = current_gate.get("nemoclaw_installer_review")
    if not isinstance(summary, dict) or not summary:
        if has_install_command:
            errors.append(
                "manifest current_gate installs NeMoClaw but has no nemoclaw_installer_review summary"
            )
        return errors
    records = file_records_by_source(manifest)
    json_record = validate_any_file_role(
        errors=errors,
        records=records,
        path_value=summary.get("path"),
        role_suffix="latest_installer_review_json",
        label="NeMoClaw installer review JSON",
    )
    if isinstance(json_record, dict):
        bundle_path = json_record.get("bundle_path")
        if not isinstance(bundle_path, str) or not bundle_path:
            errors.append("NeMoClaw installer review JSON missing bundle_path")
        else:
            try:
                payload = read_json_object(bundle_dir / bundle_path)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"NeMoClaw installer review JSON is not readable: {exc}")
            else:
                errors.extend(
                    validate_nemoclaw_installer_review_payload(
                        payload,
                        summary=summary,
                        label="NeMoClaw installer review JSON",
                    )
                )
    path_value = summary.get("path")
    if isinstance(path_value, str) and path_value.endswith(".json"):
        markdown_path = path_value[:-5] + ".md"
        markdown_record = validate_any_file_role(
            errors=errors,
            records=records,
            path_value=markdown_path,
            role_suffix="latest_installer_review_markdown",
            label="NeMoClaw installer review Markdown",
        )
        if isinstance(markdown_record, dict):
            bundle_path = markdown_record.get("bundle_path")
            if not isinstance(bundle_path, str) or not bundle_path:
                errors.append("NeMoClaw installer review Markdown missing bundle_path")
    lock_path = summary.get("lock_json")
    if isinstance(lock_path, str) and lock_path.strip():
        if summary.get("lock_verified") is not True:
            errors.append("NeMoClaw installer review lock_verified must be true when lock_json is present")
        lock_record = validate_any_file_role(
            errors=errors,
            records=records,
            path_value=lock_path,
            role_suffix="latest_installer_review_lock_json",
            label="NeMoClaw installer review lock JSON",
        )
        if isinstance(lock_record, dict):
            bundle_path = lock_record.get("bundle_path")
            if not isinstance(bundle_path, str) or not bundle_path:
                errors.append("NeMoClaw installer review lock JSON missing bundle_path")
            else:
                try:
                    payload = read_json_object(bundle_dir / bundle_path)
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    errors.append(f"NeMoClaw installer review lock JSON is not readable: {exc}")
                else:
                    errors.extend(
                        validate_nemoclaw_installer_lock_payload(
                            payload,
                            summary=summary,
                            label="NeMoClaw installer review lock JSON",
                        )
                    )
    for row_index, row in enumerate(remediation_plan or [], start=1):
        if not isinstance(row, dict) or not isinstance(row.get("commands"), list):
            continue
        for command_index, command in enumerate(row.get("commands", []), start=1):
            if not isinstance(command, str):
                continue
            parts = command_parts(command)
            label = f"manifest current_gate remediation_plan row {row_index} command {command_index}"
            validate_nemoclaw_installer_review_command_matches_summary(
                parts=parts,
                summary=summary,
                label=label,
                errors=errors,
            )
            validate_nemoclaw_install_command_matches_review_summary(
                parts=parts,
                summary=summary,
                label=label,
                errors=errors,
            )
    operator_plan = manifest.get("operator_plan")
    if isinstance(operator_plan, dict):
        json_path_value = operator_plan.get("json")
        if isinstance(json_path_value, str) and json_path_value:
            try:
                operator_payload = read_json_object(bundle_dir / json_path_value)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"operator_plan json is not readable for installer review validation: {exc}")
            else:
                for index, command in enumerate(collect_operator_plan_commands(operator_payload), start=1):
                    parts = command_parts(command)
                    label = f"operator_plan command {index}"
                    validate_nemoclaw_installer_review_command_matches_summary(
                        parts=parts,
                        summary=summary,
                        label=label,
                        errors=errors,
                    )
                    validate_nemoclaw_install_command_matches_review_summary(
                        parts=parts,
                        summary=summary,
                        label=label,
                        errors=errors,
                    )
    return errors


REQUIRED_NEMOCLAW_THIRD_PARTY_FIELDS = {
    "name": lambda value: isinstance(value, str) and bool(value),
    "vendor": lambda value: value == "NVIDIA",
    "repository_url": lambda value: isinstance(value, str) and bool(value),
    "documentation_url": lambda value: isinstance(value, str) and bool(value),
    "installer_url": lambda value: isinstance(value, str) and bool(value),
    "install_ref": lambda value: isinstance(value, str) and bool(value),
    "installer_sha256": lambda value: isinstance(value, str),
    "installer_signature": lambda value: isinstance(value, str),
    "installer_lock_json": lambda value: isinstance(value, str) and bool(value),
    "installer_review_json": lambda value: isinstance(value, str),
    "installer_review_verified": lambda value: isinstance(value, bool),
    "installer_integrity_verified": lambda value: isinstance(value, bool),
    "installer_provenance_locked": lambda value: isinstance(value, bool),
    "installer_provenance_note": lambda value: isinstance(value, str) and bool(value),
    "acceptance_required": lambda value: value is True,
    "acceptance_flag": lambda value: value == "--yes-i-accept-third-party-software",
    "accepted": lambda value: isinstance(value, bool),
    "install_or_onboard_requested": lambda value: isinstance(value, bool),
    "operator_review_required_before_install": lambda value: value is True,
}

REQUIRED_NEMOCLAW_ACCEPTANCE_LEDGER_FIELDS = [
    "accepted_third_party_software",
    "third_party_software.name",
    "third_party_software.vendor",
    "third_party_software.installer_url",
    "third_party_software.install_ref",
    "third_party_software.installer_sha256",
    "third_party_software.installer_signature",
    "third_party_software.installer_lock_json",
    "third_party_software.installer_review_json",
    "third_party_software.installer_review_verified",
    "third_party_software.installer_integrity_verified",
    "third_party_software.installer_provenance_locked",
    "third_party_software.installer_provenance_note",
    "third_party_software.acceptance_required",
    "third_party_software.acceptance_flag",
    "third_party_software.accepted",
    "policy_tier",
    "policy_tier_allowed_values",
    "policy_tier_valid",
    "setup_plan.policy_tier",
    "setup_plan.policy_tier_allowed_values",
    "setup_plan.policy_tier_valid",
    "setup_plan.installer_sha256",
    "setup_plan.installer_signature",
    "setup_plan.installer_lock_json",
    "setup_plan.installer_review_json",
    "setup_plan.installer_review_verified",
    "setup_plan.installer_integrity_verified",
    "setup_plan.installer_provenance_locked",
    "setup_plan.installer_provenance_note",
    "operation_results.install.log_path",
    "operation_results.onboard.log_path",
]

REQUIRED_NEMOCLAW_OPERATOR_SEQUENCE = [
    (
        "setup_check",
        "post_install_check_command",
        "temp/nemoclaw_setup_check_YYYYMMDDTHHMM.json",
        False,
    ),
    (
        "installer_review",
        "installer_review_command",
        "temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json",
        False,
    ),
    (
        "install_and_onboard",
        "install_and_onboard_command",
        "temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.json",
        True,
    ),
    (
        "post_install_verification",
        "post_install_verification_command",
        "temp/nemoclaw_post_install_verification_YYYYMMDDTHHMM.json",
        False,
    ),
    (
        "canary_readiness",
        "canary_readiness_command",
        "outputs/taiwan_full_eval/openai_canary_readiness_nemoclaw.json",
        False,
    ),
    (
        "adoption_check",
        "adoption_check_command",
        "temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.json",
        False,
    ),
    (
        "production_readiness",
        "production_readiness_command",
        "outputs/taiwan_full_eval/taiwan_production_readiness_report.json",
        False,
    ),
]

REQUIRED_NEMOCLAW_OPERATOR_EVIDENCE_PATHS = [
    "temp/nemoclaw_setup_check_YYYYMMDDTHHMM.json",
    "temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json",
    "temp/nemoclaw_installer_review_YYYYMMDDTHHMM.md",
    "temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.json",
    "temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.install.log",
    "temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.onboard.log",
    "temp/nemoclaw_post_install_verification_YYYYMMDDTHHMM.json",
    "temp/nemoclaw_post_install_verification_YYYYMMDDTHHMM.md",
    "outputs/taiwan_full_eval/openai_canary_readiness_nemoclaw.json",
    "temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.json",
    "temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.md",
    "outputs/taiwan_full_eval/taiwan_production_readiness_report.json",
]


def validate_nemoclaw_production_install_and_onboard_command(
    *,
    command: Any,
    label: str,
    expected_installer_lock_json: str | None,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(command, str) or not command.strip():
        return [f"{label} setup_plan.production_install_and_onboard_command is missing"]
    parts = command_parts(command)
    if not command_installs_nemoclaw(parts):
        errors.append(
            f"{label} setup_plan.production_install_and_onboard_command must install NeMoClaw"
        )
    if "--onboard" not in parts:
        errors.append(
            f"{label} setup_plan.production_install_and_onboard_command must include --onboard"
        )
    if "--yes-i-accept-third-party-software" not in parts:
        errors.append(
            f"{label} setup_plan.production_install_and_onboard_command must include --yes-i-accept-third-party-software"
        )
    for flag in (
        "--installer-sha256",
        "--installer-review-json",
        "--installer-lock-json",
        "--json",
    ):
        value, present = command_flag_value(parts, flag)
        if not present:
            errors.append(
                f"{label} setup_plan.production_install_and_onboard_command missing {flag}"
            )
        elif not value:
            errors.append(
                f"{label} setup_plan.production_install_and_onboard_command has empty {flag}"
            )
    policy_tier, policy_present = command_flag_value(parts, "--policy-tier")
    if not policy_present:
        errors.append(
            f"{label} setup_plan.production_install_and_onboard_command missing --policy-tier"
        )
    elif policy_tier != REQUIRED_NEMOCLAW_POLICY_TIER:
        errors.append(
            f"{label} setup_plan.production_install_and_onboard_command --policy-tier must be {REQUIRED_NEMOCLAW_POLICY_TIER}"
        )
    if expected_installer_lock_json:
        lock_json, lock_present = command_flag_value(parts, "--installer-lock-json")
        if lock_present and source_path_key(lock_json) != source_path_key(expected_installer_lock_json):
            errors.append(
            f"{label} setup_plan.production_install_and_onboard_command --installer-lock-json does not match current_gate.nemoclaw_installer_review.lock_json"
            )
    return errors


def validate_nemoclaw_setup_installer_review_command(
    *,
    command: Any,
    label: str,
    expected_installer_lock_json: str | None,
    expected_installer_sha256: str | None,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(command, str) or not command.strip():
        return [f"{label} setup_plan.installer_review_command is missing"]
    parts = command_parts(command)
    if not command_invokes_nemoclaw_installer_review(parts):
        errors.append(
            f"{label} setup_plan.installer_review_command must invoke review_nemoclaw_installer.py"
        )
    for flag in ("--url", "--install-ref", "--expected-sha256", "--lock-json", "--json", "--markdown"):
        value, present = command_flag_value(parts, flag)
        if not present:
            errors.append(f"{label} setup_plan.installer_review_command missing {flag}")
        elif not value:
            errors.append(f"{label} setup_plan.installer_review_command has empty {flag}")
    expected_sha_arg, expected_sha_present = command_flag_value(parts, "--expected-sha256")
    if expected_sha_present and expected_sha_arg:
        normalized = str(expected_sha_arg).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", normalized):
            errors.append(
                f"{label} setup_plan.installer_review_command --expected-sha256 is not 64 lowercase hex"
            )
        elif expected_installer_sha256 and normalized != expected_installer_sha256:
            errors.append(
                f"{label} setup_plan.installer_review_command --expected-sha256 does not match current_gate.nemoclaw_installer_review.sha256"
            )
    if expected_installer_lock_json:
        lock_json, lock_present = command_flag_value(parts, "--lock-json")
        if lock_present and source_path_key(lock_json) != source_path_key(expected_installer_lock_json):
            errors.append(
                f"{label} setup_plan.installer_review_command --lock-json does not match current_gate.nemoclaw_installer_review.lock_json"
            )
    return errors


def validate_nemoclaw_setup_post_install_command(
    *,
    command: Any,
    label: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(command, str) or not command.strip():
        return [f"{label} setup_plan.post_install_verification_command is missing"]
    parts = command_parts(command)
    if not command_invokes_nemoclaw_post_install(parts):
        errors.append(
            f"{label} setup_plan.post_install_verification_command must invoke verify_nemoclaw_post_install.py"
        )
    for flag in ("--json", "--markdown", "--fail-on-failed"):
        value, present = command_flag_value(parts, flag)
        if not present:
            errors.append(
                f"{label} setup_plan.post_install_verification_command missing {flag}"
            )
        elif flag != "--fail-on-failed" and not value:
            errors.append(
                f"{label} setup_plan.post_install_verification_command has empty {flag}"
            )
    return errors


def validate_nemoclaw_operator_sequence(plan: dict[str, Any], *, label: str) -> list[str]:
    errors: list[str] = []
    sequence = plan.get("operator_sequence")
    if not isinstance(sequence, list) or not sequence:
        errors.append(f"{label} setup_plan.operator_sequence is missing or invalid")
        sequence = []

    sequence_by_step = {
        row.get("step"): row
        for row in sequence
        if isinstance(row, dict) and isinstance(row.get("step"), str)
    }
    for step, command_field, expected_path, requires_external_action in (
        REQUIRED_NEMOCLAW_OPERATOR_SEQUENCE
    ):
        row = sequence_by_step.get(step)
        if not isinstance(row, dict):
            errors.append(f"{label} setup_plan.operator_sequence missing step {step}")
            continue
        if row.get("command") != plan.get(command_field):
            errors.append(
                f"{label} setup_plan.operator_sequence.{step}.command must match setup_plan.{command_field}"
            )
        if row.get("expected_evidence_path") != expected_path:
            errors.append(
                f"{label} setup_plan.operator_sequence.{step}.expected_evidence_path must be {expected_path}"
            )
        if row.get("requires_external_action") is not requires_external_action:
            errors.append(
                f"{label} setup_plan.operator_sequence.{step}.requires_external_action must be {str(requires_external_action).lower()}"
            )
        if row.get("required") is not True:
            errors.append(
                f"{label} setup_plan.operator_sequence.{step}.required must be true"
            )

    expected_paths = plan.get("expected_evidence_paths")
    if not isinstance(expected_paths, list) or not expected_paths:
        errors.append(f"{label} setup_plan.expected_evidence_paths is missing or invalid")
        expected_paths = []
    for expected_path in REQUIRED_NEMOCLAW_OPERATOR_EVIDENCE_PATHS:
        if expected_path not in expected_paths:
            errors.append(
                f"{label} setup_plan.expected_evidence_paths missing {expected_path}"
            )
    return errors


def nemoclaw_operator_handoff_from_setup_payload(
    payload: dict[str, Any],
    *,
    source_setup_json: str | None,
) -> dict[str, Any]:
    plan = payload.get("setup_plan") if isinstance(payload.get("setup_plan"), dict) else {}
    raw_steps = plan.get("operator_sequence")
    steps: list[dict[str, Any]] = []
    if isinstance(raw_steps, list):
        for row in raw_steps:
            if not isinstance(row, dict):
                continue
            steps.append(
                {
                    "step": row.get("step") if isinstance(row.get("step"), str) else None,
                    "command": (
                        row.get("command") if isinstance(row.get("command"), str) else None
                    ),
                    "expected_evidence_path": (
                        row.get("expected_evidence_path")
                        if isinstance(row.get("expected_evidence_path"), str)
                        else None
                    ),
                    "requires_external_action": row.get("requires_external_action") is True,
                    "required": row.get("required") is True,
                }
            )
    raw_evidence_paths = plan.get("expected_evidence_paths")
    expected_evidence_paths = (
        [path for path in raw_evidence_paths if isinstance(path, str) and path.strip()]
        if isinstance(raw_evidence_paths, list)
        else []
    )
    handoff: dict[str, Any] = {
        "available": bool(steps),
        "source_setup_json": source_setup_json,
        "step_count": len(steps),
        "required_step_count": sum(1 for row in steps if row.get("required") is True),
        "external_action_step_count": sum(
            1 for row in steps if row.get("requires_external_action") is True
        ),
        "evidence_path_count": len(expected_evidence_paths),
        "steps": steps,
        "expected_evidence_paths": expected_evidence_paths,
    }
    for field in (
        "post_install_check_command",
        "installer_review_command",
        "install_and_onboard_command",
        "production_install_and_onboard_command",
        "post_install_verification_command",
        "canary_readiness_command",
        "adoption_check_command",
        "production_readiness_command",
    ):
        value = plan.get(field)
        handoff[field] = value if isinstance(value, str) and value.strip() else None
    return handoff


def validate_nemoclaw_operator_handoff(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
    payload: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    handoff = payload.get("operator_handoff")
    if not isinstance(handoff, dict):
        return ["NeMoClaw adoption JSON operator_handoff is missing or invalid"]

    steps = handoff.get("steps") if isinstance(handoff.get("steps"), list) else []
    expected_evidence_paths = (
        handoff.get("expected_evidence_paths")
        if isinstance(handoff.get("expected_evidence_paths"), list)
        else []
    )
    if handoff.get("step_count") != len(steps):
        errors.append("NeMoClaw adoption JSON operator_handoff step_count is inconsistent")
    if handoff.get("required_step_count") != sum(
        1 for row in steps if isinstance(row, dict) and row.get("required") is True
    ):
        errors.append("NeMoClaw adoption JSON operator_handoff required_step_count is inconsistent")
    if handoff.get("external_action_step_count") != sum(
        1
        for row in steps
        if isinstance(row, dict) and row.get("requires_external_action") is True
    ):
        errors.append(
            "NeMoClaw adoption JSON operator_handoff external_action_step_count is inconsistent"
        )
    if handoff.get("evidence_path_count") != len(
        [path for path in expected_evidence_paths if isinstance(path, str) and path.strip()]
    ):
        errors.append(
            "NeMoClaw adoption JSON operator_handoff evidence_path_count is inconsistent"
        )

    source_setup_json = handoff.get("source_setup_json")
    source_key = source_path_key(source_setup_json)
    if not source_key:
        errors.append("NeMoClaw adoption JSON operator_handoff source_setup_json is missing")
        return errors
    records = file_records_by_source(manifest)
    setup_record = records.get(source_key)
    if not isinstance(setup_record, dict):
        errors.append(
            f"NeMoClaw adoption JSON operator_handoff setup evidence missing bundle_path: {source_key}"
        )
        return errors
    bundle_path = setup_record.get("bundle_path")
    if not isinstance(bundle_path, str) or not bundle_path:
        errors.append(
            f"NeMoClaw adoption JSON operator_handoff setup evidence missing bundle_path: {source_key}"
        )
        return errors
    try:
        setup_payload = read_json_object(bundle_dir / bundle_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"NeMoClaw adoption JSON operator_handoff setup evidence is not readable: {exc}")
        return errors
    expected = nemoclaw_operator_handoff_from_setup_payload(
        setup_payload,
        source_setup_json=source_path_key(source_setup_json),
    )
    if handoff != expected:
        errors.append("NeMoClaw adoption JSON operator_handoff does not match setup evidence")
    return errors


def validate_nemoclaw_setup_acceptance_payload(
    payload: dict[str, Any],
    *,
    label: str,
    expected_installer_lock_json: str | None = None,
    expected_installer_sha256: str | None = None,
) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append(f"{label} schema_version must be 1")
    third_party = (
        payload.get("third_party_software")
        if isinstance(payload.get("third_party_software"), dict)
        else {}
    )
    for field, predicate in REQUIRED_NEMOCLAW_THIRD_PARTY_FIELDS.items():
        if not predicate(third_party.get(field)):
            errors.append(f"{label} third_party_software missing/invalid {field}")

    plan = payload.get("setup_plan") if isinstance(payload.get("setup_plan"), dict) else {}
    if plan.get("third_party_software_name") != third_party.get("name"):
        errors.append(f"{label} setup_plan.third_party_software_name does not match third_party_software.name")
    if not isinstance(plan.get("installer_url"), str) or not plan.get("installer_url"):
        errors.append(f"{label} setup_plan.installer_url is missing")
    if not isinstance(plan.get("install_ref"), str) or not plan.get("install_ref"):
        errors.append(f"{label} setup_plan.install_ref is missing")
    errors.extend(validate_nemoclaw_operator_sequence(plan, label=label))
    errors.extend(
        validate_nemoclaw_setup_installer_review_command(
            command=plan.get("installer_review_command"),
            label=label,
            expected_installer_lock_json=expected_installer_lock_json,
            expected_installer_sha256=expected_installer_sha256,
        )
    )
    errors.extend(
        validate_nemoclaw_setup_post_install_command(
            command=plan.get("post_install_verification_command"),
            label=label,
        )
    )
    for field, predicate in (
        ("installer_sha256", lambda value: isinstance(value, str)),
        ("installer_signature", lambda value: isinstance(value, str)),
        ("installer_lock_json", lambda value: isinstance(value, str) and bool(value)),
        ("installer_review_json", lambda value: isinstance(value, str)),
        ("installer_review_verified", lambda value: isinstance(value, bool)),
        ("installer_integrity_verified", lambda value: isinstance(value, bool)),
        ("installer_provenance_locked", lambda value: isinstance(value, bool)),
        ("installer_provenance_note", lambda value: isinstance(value, str) and bool(value)),
    ):
        if not predicate(plan.get(field)):
            errors.append(f"{label} setup_plan.{field} is missing or invalid")
    errors.extend(
        validate_nemoclaw_production_install_and_onboard_command(
            command=plan.get("production_install_and_onboard_command"),
            label=label,
            expected_installer_lock_json=expected_installer_lock_json,
        )
    )
    if (
        isinstance(plan.get("production_install_and_onboard_command"), str)
        and isinstance(plan.get("install_and_onboard_command"), str)
        and plan.get("production_install_and_onboard_command")
        != plan.get("install_and_onboard_command")
    ):
        errors.append(
            f"{label} setup_plan.production_install_and_onboard_command must match setup_plan.install_and_onboard_command"
        )
    for field in (
        "installer_url",
        "install_ref",
        "installer_sha256",
        "installer_signature",
        "installer_lock_json",
        "installer_review_json",
        "installer_review_verified",
        "installer_integrity_verified",
        "installer_provenance_locked",
        "installer_provenance_note",
    ):
        if third_party.get(field) != plan.get(field):
            errors.append(
                f"{label} third_party_software.{field} does not match setup_plan.{field}"
            )
    if third_party.get("installer_provenance_note") != NEMOCLAW_INSTALLER_PROVENANCE_NOTE:
        errors.append(f"{label} third_party_software.installer_provenance_note has unexpected value")
    if plan.get("installer_provenance_note") != NEMOCLAW_INSTALLER_PROVENANCE_NOTE:
        errors.append(f"{label} setup_plan.installer_provenance_note has unexpected value")
    for source_name, source in (
        ("third_party_software", third_party),
        ("setup_plan", plan),
    ):
        if (
            expected_installer_lock_json
            and isinstance(source.get("installer_lock_json"), str)
            and source.get("installer_lock_json")
            and source.get("installer_lock_json") != expected_installer_lock_json
        ):
            errors.append(
                f"{label} {source_name}.installer_lock_json does not match current_gate.nemoclaw_installer_review.lock_json"
            )
        if source.get("installer_review_verified") is True and not source.get("installer_lock_json"):
            errors.append(
                f"{label} {source_name}.installer_review_verified=true requires non-empty installer_lock_json"
            )
        if source.get("installer_review_verified") is True and not source.get("installer_review_json"):
            errors.append(
                f"{label} {source_name}.installer_review_verified=true requires non-empty installer_review_json"
            )
        if source.get("installer_integrity_verified") is True and not source.get("installer_sha256"):
            errors.append(
                f"{label} {source_name}.installer_integrity_verified=true requires non-empty installer_sha256"
            )
        if source.get("installer_integrity_verified") is True and source.get("installer_review_verified") is not True:
            errors.append(
                f"{label} {source_name}.installer_integrity_verified=true requires installer_review_verified=true"
            )
        if source.get("installer_provenance_locked") is True and source.get("installer_integrity_verified") is not True:
            errors.append(
                f"{label} {source_name}.installer_provenance_locked=true requires installer_integrity_verified=true"
            )
        if source.get("installer_provenance_locked") is True and not source.get("installer_sha256"):
            errors.append(
                f"{label} {source_name}.installer_provenance_locked=true requires non-empty installer_sha256"
            )
        if source.get("installer_provenance_locked") is True and not source.get("installer_lock_json"):
            errors.append(
                f"{label} {source_name}.installer_provenance_locked=true requires non-empty installer_lock_json"
            )
    operation_results = (
        payload.get("operation_results")
        if isinstance(payload.get("operation_results"), dict)
        else {}
    )
    install_result = (
        operation_results.get("install")
        if isinstance(operation_results.get("install"), dict)
        else {}
    )
    onboard_result = (
        operation_results.get("onboard")
        if isinstance(operation_results.get("onboard"), dict)
        else {}
    )
    requested_or_attempted = any(
        value is True
        for value in (
            payload.get("install_requested"),
            payload.get("onboard_requested"),
            install_result.get("requested"),
            install_result.get("attempted"),
            onboard_result.get("requested"),
            onboard_result.get("attempted"),
        )
    )
    if requested_or_attempted:
        for source_name, source in (
            ("third_party_software", third_party),
            ("setup_plan", plan),
        ):
            if source.get("installer_review_verified") is not True:
                errors.append(
                    f"{label} install/onboard requested or attempted without {source_name}.installer_review_verified=true"
                )
            if not source.get("installer_lock_json"):
                errors.append(
                    f"{label} install/onboard requested or attempted without {source_name}.installer_lock_json"
                )
            if source.get("installer_integrity_verified") is not True:
                errors.append(
                    f"{label} install/onboard requested or attempted without {source_name}.installer_integrity_verified=true"
                )
            if source.get("installer_provenance_locked") is not True:
                errors.append(
                    f"{label} install/onboard requested or attempted without {source_name}.installer_provenance_locked=true"
                )
    if payload.get("policy_tier") != REQUIRED_NEMOCLAW_POLICY_TIER:
        errors.append(
            f"{label} policy_tier must be {REQUIRED_NEMOCLAW_POLICY_TIER}"
        )
    if payload.get("policy_tier_allowed_values") != ALLOWED_NEMOCLAW_POLICY_TIERS:
        errors.append(f"{label} policy_tier_allowed_values are missing or invalid")
    if payload.get("policy_tier_valid") is not True:
        errors.append(f"{label} policy_tier_valid must be true")
    if plan.get("policy_tier") != REQUIRED_NEMOCLAW_POLICY_TIER:
        errors.append(
            f"{label} setup_plan.policy_tier must be {REQUIRED_NEMOCLAW_POLICY_TIER}"
        )
    if plan.get("policy_tier_allowed_values") != ALLOWED_NEMOCLAW_POLICY_TIERS:
        errors.append(f"{label} setup_plan.policy_tier_allowed_values are missing or invalid")
    if plan.get("policy_tier_valid") is not True:
        errors.append(f"{label} setup_plan.policy_tier_valid must be true")
    ledger_fields = (
        plan.get("acceptance_ledger_fields")
        if isinstance(plan.get("acceptance_ledger_fields"), list)
        else []
    )
    for field in REQUIRED_NEMOCLAW_ACCEPTANCE_LEDGER_FIELDS:
        if field not in ledger_fields:
            errors.append(f"{label} setup_plan.acceptance_ledger_fields missing {field}")
    return errors


def validate_nemoclaw_operation_log_evidence(
    payload: dict[str, Any],
    *,
    records: dict[str, dict[str, Any]],
    label: str,
) -> list[str]:
    errors: list[str] = []
    operation_results = (
        payload.get("operation_results")
        if isinstance(payload.get("operation_results"), dict)
        else {}
    )
    for operation in ("install", "onboard"):
        result = (
            operation_results.get(operation)
            if isinstance(operation_results.get(operation), dict)
            else {}
        )
        if result.get("attempted") is not True:
            continue
        record = validate_any_file_role(
            errors=errors,
            records=records,
            path_value=result.get("log_path"),
            role_suffix=f"operation:{operation}:log",
            label=f"{label} operation_results.{operation}.log_path",
        )
        if not isinstance(record, dict):
            continue
        if not isinstance(record.get("bundle_path"), str) or not record.get("bundle_path"):
            errors.append(
                f"{label} operation_results.{operation}.log_path bundled record missing bundle_path"
            )
        size_bytes = record.get("size_bytes")
        if not isinstance(size_bytes, int) or size_bytes <= 0:
            errors.append(
                f"{label} operation_results.{operation}.log_path bundled log must be non-empty"
            )
    return errors


def validate_nemoclaw_setup_acceptance_evidence(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    current_gate = manifest.get("current_gate")
    if not isinstance(current_gate, dict):
        return errors
    adoption = current_gate.get("nemoclaw_adoption")
    if not isinstance(adoption, dict):
        return errors
    criteria = adoption.get("criteria")
    if not isinstance(criteria, list):
        return errors
    setup_criterion = next(
        (
            criterion
            for criterion in criteria
            if isinstance(criterion, dict) and criterion.get("name") == "setup_plan_safety"
        ),
        None,
    )
    if not isinstance(setup_criterion, dict) or setup_criterion.get("ok") is not True:
        return errors

    installer_review = current_gate.get("nemoclaw_installer_review")
    expected_installer_lock_json = (
        installer_review.get("lock_json")
        if isinstance(installer_review, dict)
        and isinstance(installer_review.get("lock_json"), str)
        and installer_review.get("lock_json")
        else None
    )
    expected_installer_sha256 = (
        str(installer_review.get("sha256") or "").lower()
        if isinstance(installer_review, dict)
        and re.fullmatch(r"[0-9a-f]{64}", str(installer_review.get("sha256") or "").lower())
        else None
    )

    evidence_paths = setup_criterion.get("evidence_paths")
    if not isinstance(evidence_paths, list) or not evidence_paths:
        return ["NeMoClaw setup_plan_safety passed but has no evidence_paths"]

    records = file_records_by_source(manifest)
    for evidence_path in evidence_paths:
        record = validate_any_file_role(
            errors=errors,
            records=records,
            path_value=evidence_path,
            role_suffix="setup_plan_safety:evidence",
            label="NeMoClaw setup_plan_safety evidence",
        )
        if not isinstance(record, dict):
            continue
        bundle_path = record.get("bundle_path")
        if not isinstance(bundle_path, str) or not bundle_path:
            errors.append(f"NeMoClaw setup_plan_safety evidence missing bundle_path: {source_path_key(evidence_path)}")
            continue
        try:
            payload = read_json_object(bundle_dir / bundle_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"NeMoClaw setup_plan_safety evidence is not readable: {exc}")
            continue
        errors.extend(
            validate_nemoclaw_setup_acceptance_payload(
                payload,
                label=f"NeMoClaw setup evidence {source_path_key(evidence_path)}",
                expected_installer_lock_json=expected_installer_lock_json,
                expected_installer_sha256=expected_installer_sha256,
            )
        )
        errors.extend(
            validate_nemoclaw_operation_log_evidence(
                payload,
                records=records,
                label=f"NeMoClaw setup evidence {source_path_key(evidence_path)}",
            )
        )
    return errors


def summarize_nemoclaw_adoption_criteria(criteria: Any) -> list[dict[str, Any]]:
    if not isinstance(criteria, list):
        return []
    result: list[dict[str, Any]] = []
    for row in criteria:
        if not isinstance(row, dict):
            continue
        criterion = {
            "name": row.get("name"),
            "ok": bool(row.get("ok")),
            "status": row.get("status"),
            "detail": row.get("detail"),
            "next_action": row.get("next_action"),
            "evidence_paths": (
                row.get("evidence_paths")
                if isinstance(row.get("evidence_paths"), list)
                else []
            ),
        }
        if "host_prerequisites_ok" in row:
            criterion["host_prerequisites_ok"] = row.get("host_prerequisites_ok")
        if "runtime_installed" in row:
            criterion["runtime_installed"] = row.get("runtime_installed")
        if isinstance(row.get("missing_required_commands"), list):
            criterion["missing_required_commands"] = row.get("missing_required_commands")
        if isinstance(row.get("missing_components"), list):
            criterion["missing_components"] = row.get("missing_components")
        for key in (
            "wandb_weave_policy_present",
            "runtime_network_policy_allowlist_ok",
            "unknown_runtime_network_policies",
            "allowed_runtime_network_policies",
            "detailed_status_network_policy_count",
            "detailed_status_network_policies",
            "non_wandb_network_policies",
        ):
            value = row.get(key)
            if isinstance(value, (bool, int, str)) or isinstance(value, list):
                criterion[key] = value
        result.append(criterion)
    return result


def read_bundled_nemoclaw_adoption_json(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
    errors: list[str],
) -> dict[str, Any] | None:
    current_gate = manifest.get("current_gate")
    if not isinstance(current_gate, dict):
        return None
    adoption = current_gate.get("nemoclaw_adoption")
    if not isinstance(adoption, dict) or not adoption:
        return None
    path_value = adoption.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        return None

    records = file_records_by_source(manifest)
    record = validate_file_role(
        errors=errors,
        records=records,
        path_value=path_value,
        role="nemoclaw_adoption_check",
        label="NeMoClaw adoption JSON",
    )
    if not isinstance(record, dict):
        return None
    bundle_path = record.get("bundle_path")
    if not isinstance(bundle_path, str) or not bundle_path:
        errors.append(f"NeMoClaw adoption JSON missing bundle_path: {source_path_key(path_value)}")
        return None
    try:
        payload = read_json_object(bundle_dir / bundle_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"NeMoClaw adoption JSON is not readable: {exc}")
        return None
    if payload.get("schema_version") != 1:
        errors.append("NeMoClaw adoption JSON schema_version must be 1")
    return payload


def validate_nemoclaw_adoption_payload_matches_current_gate(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    current_gate = manifest.get("current_gate")
    if not isinstance(current_gate, dict):
        return errors
    adoption = current_gate.get("nemoclaw_adoption")
    if not isinstance(adoption, dict) or not adoption:
        return errors
    payload = read_bundled_nemoclaw_adoption_json(
        bundle_dir=bundle_dir,
        manifest=manifest,
        errors=errors,
    )
    if not isinstance(payload, dict):
        return errors
    payload_path = payload.get("path")
    if isinstance(payload_path, str) and payload_path.strip():
        if source_path_key(adoption.get("path")) != source_path_key(payload_path):
            errors.append(
                "NeMoClaw adoption JSON path does not match manifest current_gate"
            )
    else:
        errors.append("NeMoClaw adoption JSON path is missing")
    payload_markdown_path = payload.get("markdown_path")
    if isinstance(payload_markdown_path, str) and payload_markdown_path.strip():
        if source_path_key(adoption.get("markdown_path")) != source_path_key(
            payload_markdown_path
        ):
            errors.append(
                "NeMoClaw adoption JSON markdown_path does not match manifest current_gate"
            )
    else:
        errors.append("NeMoClaw adoption JSON markdown_path is missing")
    adoption_decision = (
        payload.get("adoption_decision")
        if isinstance(payload.get("adoption_decision"), dict)
        else {}
    )
    adoption_summary = (
        payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    )
    setup_runtime = (
        payload.get("setup_runtime")
        if isinstance(payload.get("setup_runtime"), dict)
        else {}
    )
    summary_setup_runtime = (
        adoption_summary.get("setup_runtime")
        if isinstance(adoption_summary.get("setup_runtime"), dict)
        else {}
    )
    expected_blockers = (
        adoption_decision.get("blockers")
        if isinstance(adoption_decision.get("blockers"), list)
        else []
    )
    if payload.get("ready_for_use") != adoption_decision.get("ready_for_use"):
        errors.append(
            "NeMoClaw adoption JSON ready_for_use does not match adoption_decision"
        )
    if payload.get("adoption_recommendation") != adoption_decision.get("recommendation"):
        errors.append(
            "NeMoClaw adoption JSON adoption_recommendation does not match "
            "adoption_decision"
        )
    if payload.get("adoption_scope") != adoption_decision.get("scope"):
        errors.append(
            "NeMoClaw adoption JSON adoption_scope does not match adoption_decision"
        )
    if payload.get("design_ready") != adoption_decision.get("design_ready"):
        errors.append(
            "NeMoClaw adoption JSON design_ready does not match adoption_decision"
        )
    if payload.get("blockers") != expected_blockers:
        errors.append("NeMoClaw adoption JSON blockers does not match adoption_decision")
    for field in ("runtime_blockers", "design_blockers", "other_blockers"):
        expected_list = (
            adoption_decision.get(field)
            if isinstance(adoption_decision.get(field), list)
            else []
        )
        if payload.get(field) != expected_list:
            errors.append(
                f"NeMoClaw adoption JSON {field} does not match adoption_decision"
            )
    if setup_runtime != summary_setup_runtime:
        errors.append("NeMoClaw adoption JSON setup_runtime does not match summary")
    missing_required_commands = (
        setup_runtime.get("missing_required_commands")
        if isinstance(setup_runtime.get("missing_required_commands"), list)
        else []
    )
    if payload.get("missing_required_commands") != missing_required_commands:
        errors.append(
            "NeMoClaw adoption JSON missing_required_commands does not match "
            "setup_runtime"
        )
    missing_components = (
        setup_runtime.get("missing_components")
        if isinstance(setup_runtime.get("missing_components"), list)
        else []
    )
    if payload.get("missing_components") != missing_components:
        errors.append(
            "NeMoClaw adoption JSON missing_components does not match setup_runtime"
        )
    errors.extend(
        validate_nemoclaw_operator_handoff(
            bundle_dir=bundle_dir,
            manifest=manifest,
            payload=payload,
        )
    )
    operator_handoff = (
        payload.get("operator_handoff")
        if isinstance(payload.get("operator_handoff"), dict)
        else {}
    )

    expected = {
        "path": payload.get("path"),
        "markdown_path": payload.get("markdown_path"),
        "ok": bool(payload.get("ok")),
        "status": payload.get("status"),
        "generated_at": payload.get("generated_at"),
        "sandbox": payload.get("sandbox"),
        "adoption_decision": adoption_decision,
        "ready_for_use": payload.get("ready_for_use"),
        "adoption_recommendation": payload.get("adoption_recommendation"),
        "adoption_scope": payload.get("adoption_scope"),
        "design_ready": payload.get("design_ready"),
        "blockers": payload.get("blockers") if isinstance(payload.get("blockers"), list) else [],
        "runtime_blockers": (
            payload.get("runtime_blockers")
            if isinstance(payload.get("runtime_blockers"), list)
            else []
        ),
        "design_blockers": (
            payload.get("design_blockers")
            if isinstance(payload.get("design_blockers"), list)
            else []
        ),
        "other_blockers": (
            payload.get("other_blockers")
            if isinstance(payload.get("other_blockers"), list)
            else []
        ),
        "setup_runtime": setup_runtime,
        "missing_required_commands": (
            payload.get("missing_required_commands")
            if isinstance(payload.get("missing_required_commands"), list)
            else []
        ),
        "missing_components": (
            payload.get("missing_components")
            if isinstance(payload.get("missing_components"), list)
            else []
        ),
        "operator_handoff": operator_handoff,
        "operator_handoff_step_count": operator_handoff.get("step_count"),
        "operator_handoff_external_action_step_count": operator_handoff.get(
            "external_action_step_count"
        ),
        "operator_handoff_evidence_path_count": operator_handoff.get(
            "evidence_path_count"
        ),
        "summary": adoption_summary,
        "setup_paths": payload.get("setup_paths") if isinstance(payload.get("setup_paths"), list) else [],
        "readiness_paths": (
            payload.get("readiness_paths")
            if isinstance(payload.get("readiness_paths"), list)
            else []
        ),
        "agentic_config_paths": (
            payload.get("agentic_config_paths")
            if isinstance(payload.get("agentic_config_paths"), list)
            else []
        ),
        "criteria": summarize_nemoclaw_adoption_criteria(payload.get("criteria")),
    }
    for field, value in expected.items():
        if adoption.get(field) != value:
            errors.append(
                f"NeMoClaw adoption JSON {field} does not match manifest current_gate"
            )
    return errors


def nemoclaw_setup_runtime_summary(payload: dict[str, Any]) -> dict[str, Any]:
    commands = payload.get("commands") if isinstance(payload.get("commands"), dict) else {}
    docker = commands.get("docker") if isinstance(commands.get("docker"), dict) else {}
    nemoclaw = commands.get("nemoclaw") if isinstance(commands.get("nemoclaw"), dict) else {}
    openshell = commands.get("openshell") if isinstance(commands.get("openshell"), dict) else {}
    host_prerequisites_ok = payload.get("host_prerequisites_ok")
    if not isinstance(host_prerequisites_ok, bool):
        host_prerequisites_ok = docker.get("available") is True and docker.get("info_ok") is True
    runtime_installed = payload.get("runtime_installed")
    if not isinstance(runtime_installed, bool):
        runtime_installed = (
            nemoclaw.get("available") is True
            and openshell.get("available") is True
        )
    ok = (
        payload.get("ok") is True
        and docker.get("available") is True
        and docker.get("info_ok") is True
        and nemoclaw.get("available") is True
        and openshell.get("available") is True
    )
    missing_components = [
        name
        for name, command in (
            ("docker", docker),
            ("nemoclaw", nemoclaw),
            ("openshell", openshell),
        )
        if command.get("available") is not True
    ]
    if docker.get("available") is True and docker.get("info_ok") is not True:
        missing_components.append("docker_info")
    missing_required_commands = payload.get("missing_required_commands")
    if not isinstance(missing_required_commands, list):
        missing_required_commands = missing_components
    return {
        "ok": ok,
        "status": "passed" if ok else "missing_or_not_ready",
        "host_prerequisites_ok": host_prerequisites_ok,
        "runtime_installed": runtime_installed,
        "missing_required_commands": missing_required_commands,
        "missing_components": missing_components,
    }


def validate_nemoclaw_setup_installed_evidence(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    payload = read_bundled_nemoclaw_adoption_json(
        bundle_dir=bundle_dir,
        manifest=manifest,
        errors=errors,
    )
    if not isinstance(payload, dict):
        return errors
    criteria = payload.get("criteria")
    if not isinstance(criteria, list):
        errors.append("NeMoClaw adoption JSON criteria is not a list")
        return errors
    setup_criterion = next(
        (
            criterion
            for criterion in criteria
            if isinstance(criterion, dict) and criterion.get("name") == "setup_installed"
        ),
        None,
    )
    if not isinstance(setup_criterion, dict):
        errors.append("NeMoClaw adoption JSON missing setup_installed criterion")
        return errors
    evidence_paths = setup_criterion.get("evidence_paths")
    if not isinstance(evidence_paths, list) or not evidence_paths:
        errors.append("NeMoClaw setup_installed criterion has no evidence_paths")
        return errors

    records = file_records_by_source(manifest)
    for evidence_path in evidence_paths:
        record = validate_any_file_role(
            errors=errors,
            records=records,
            path_value=evidence_path,
            role_suffix="setup_installed:evidence",
            label="NeMoClaw setup_installed evidence",
        )
        if not isinstance(record, dict):
            continue
        bundle_path = record.get("bundle_path")
        if not isinstance(bundle_path, str) or not bundle_path:
            errors.append(
                f"NeMoClaw setup_installed evidence missing bundle_path: {source_path_key(evidence_path)}"
            )
            continue
        try:
            setup_payload = read_json_object(bundle_dir / bundle_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"NeMoClaw setup_installed evidence is not readable: {exc}")
            continue
        if setup_payload.get("schema_version") != 1:
            errors.append("NeMoClaw setup_installed evidence schema_version must be 1")
        expected = nemoclaw_setup_runtime_summary(setup_payload)
        for field in (
            "ok",
            "status",
            "host_prerequisites_ok",
            "runtime_installed",
            "missing_required_commands",
            "missing_components",
        ):
            if setup_criterion.get(field) != expected.get(field):
                errors.append(
                    f"NeMoClaw setup_installed {field} does not match setup evidence"
                )
    return errors


def validate_nemoclaw_swebench_non_adoption_evidence(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    current_gate = manifest.get("current_gate")
    if not isinstance(current_gate, dict):
        return errors
    adoption = current_gate.get("nemoclaw_adoption")
    if not isinstance(adoption, dict) or not adoption:
        return errors
    path_value = adoption.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        return errors

    payload = read_bundled_nemoclaw_adoption_json(
        bundle_dir=bundle_dir,
        manifest=manifest,
        errors=errors,
    )
    if not isinstance(payload, dict):
        return errors

    criteria = payload.get("criteria")
    if not isinstance(criteria, list):
        errors.append("NeMoClaw adoption JSON criteria is not a list")
        return errors
    swebench_criterion = next(
        (
            criterion
            for criterion in criteria
            if isinstance(criterion, dict)
            and criterion.get("name") == "swebench_pro_non_adoption_guard"
        ),
        None,
    )
    if not isinstance(swebench_criterion, dict):
        errors.append("NeMoClaw adoption JSON missing swebench_pro_non_adoption_guard criterion")
        return errors
    if swebench_criterion.get("ok") is not True:
        return errors

    adoption_decision = payload.get("adoption_decision")
    expected_sandbox = (
        adoption_decision.get("sandbox")
        if isinstance(adoption_decision, dict) and isinstance(adoption_decision.get("sandbox"), str)
        else None
    )
    offending = swebench_criterion.get("offending_records")
    if isinstance(offending, list) and offending:
        errors.append(
            "NeMoClaw SWE-Bench Pro migration guard passed but "
            f"offending_records is non-empty: {len(offending)}"
        )
    elif not isinstance(offending, list):
        errors.append(
            "NeMoClaw SWE-Bench Pro migration guard passed but "
            "offending_records is not a list"
        )

    evidence_paths = swebench_criterion.get("evidence_paths")
    if not isinstance(evidence_paths, list) or not evidence_paths:
        errors.append(
            "NeMoClaw SWE-Bench Pro migration guard passed but "
            "evidence_paths is empty"
        )
        evidence_paths = []
    evidence_path_keys = {source_path_key(path) for path in evidence_paths}
    evidence_path_keys.discard("")
    records_by_source = file_records_by_source(manifest)
    for evidence_path in evidence_paths:
        validate_any_file_role(
            errors=errors,
            records=records_by_source,
            path_value=evidence_path,
            role_suffix="swebench_pro_non_adoption_guard:evidence",
            label="NeMoClaw SWE-Bench Pro migration config evidence",
        )

    checked_records = swebench_criterion.get("records")
    if not isinstance(checked_records, list):
        errors.append(
            "NeMoClaw SWE-Bench Pro migration guard passed but records is not a list"
        )
        return errors
    for index, config_record in enumerate(checked_records, start=1):
        if not isinstance(config_record, dict):
            errors.append(f"NeMoClaw SWE-Bench Pro config record {index} is not an object")
            continue
        keys = config_record.get("swebench_pro_nemoclaw_keys")
        if not isinstance(keys, list):
            errors.append(
                "NeMoClaw SWE-Bench Pro config record "
                f"{index} has non-list swebench_pro_nemoclaw_keys"
            )
            continue
        if keys:
            sandbox = config_record.get("swebench_pro_nemoclaw_sandbox")
            matches_sandbox = config_record.get("swebench_pro_matches_sandbox")
            if not isinstance(sandbox, str) or not sandbox.strip():
                path = config_record.get("path") or f"record_{index}"
                errors.append(
                    "NeMoClaw SWE-Bench Pro migration guard passed but "
                    f"{path} contains NeMoClaw key(s) without swebench_pro_nemoclaw_sandbox"
                )
            if expected_sandbox and sandbox != expected_sandbox:
                path = config_record.get("path") or f"record_{index}"
                errors.append(
                    "NeMoClaw SWE-Bench Pro migration guard passed but "
                    f"{path} has swebench_pro_nemoclaw_sandbox={sandbox!r}, "
                    f"expected {expected_sandbox!r}"
                )
            if matches_sandbox is not True:
                path = config_record.get("path") or f"record_{index}"
                errors.append(
                    "NeMoClaw SWE-Bench Pro migration guard passed but "
                    f"{path} does not prove swebench_pro_matches_sandbox=true"
                )
        if not keys:
            record_path_key = source_path_key(config_record.get("path"))
            if record_path_key and record_path_key not in evidence_path_keys:
                errors.append(
                    "NeMoClaw SWE-Bench Pro migration guard record "
                    f"{config_record.get('path')} is not listed in evidence_paths"
                )
            continue
        record_path_key = source_path_key(config_record.get("path"))
        if record_path_key and record_path_key not in evidence_path_keys:
            path = config_record.get("path") or f"record_{index}"
            errors.append(
                "NeMoClaw SWE-Bench Pro migration guard record "
                f"{path} is not listed in evidence_paths"
            )
    return errors


def yaml_section(payload: dict[str, Any], section_name: str) -> dict[str, Any]:
    section = payload.get(section_name)
    return section if isinstance(section, dict) else {}


def validate_config_string_list_superset(
    *,
    errors: list[str],
    section: dict[str, Any],
    field: str,
    required_values: set[str],
    label: str,
) -> None:
    values = section.get(field)
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        errors.append(f"{label} {field} is not a string list")
        return
    observed = set(values)
    for required in sorted(required_values):
        if required not in observed:
            errors.append(f"{label} {field} missing required value {required}")


def tool_name_matches_pattern(tool_name: str, pattern: str) -> bool:
    tool_name_norm = tool_name.lower()
    pattern_norm = pattern.lower()
    if pattern_norm.startswith("re:"):
        return re.search(pattern_norm[3:], tool_name_norm) is not None
    if "*" in pattern_norm or "?" in pattern_norm:
        return fnmatch.fnmatch(tool_name_norm, pattern_norm)
    return tool_name_norm == pattern_norm


def validate_config_local_tools_allowed(
    *,
    errors: list[str],
    section: dict[str, Any],
    field: str,
    protected_tools: set[str],
    label: str,
) -> None:
    values = section.get(field)
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        return
    conflicts = sorted(
        pattern
        for pattern in values
        if any(tool_name_matches_pattern(tool_name, pattern) for tool_name in protected_tools)
    )
    for pattern in conflicts:
        errors.append(f"{label} {field} must not block local OpenClaw exec tool: {pattern}")


def validate_agentic_math_nemoclaw_config_yaml(
    *,
    errors: list[str],
    payload: dict[str, Any],
    expected_sandbox: str | None,
    label: str,
) -> None:
    run = yaml_section(payload, "run")
    agentic_math = yaml_section(payload, "agentic_math")
    if run.get("agentic_math") is not True:
        errors.append(f"{label} run.agentic_math must be true")
    if not agentic_math:
        errors.append(f"{label} agentic_math section is missing or invalid")
        return
    sandbox = agentic_math.get("nemoclaw_sandbox")
    if expected_sandbox and sandbox != expected_sandbox:
        errors.append(
            f"{label} agentic_math.nemoclaw_sandbox must be {expected_sandbox!r}"
        )
    elif not isinstance(sandbox, str) or not sandbox.strip():
        errors.append(f"{label} agentic_math.nemoclaw_sandbox is missing")
    if agentic_math.get("use_task_agent") is False:
        errors.append(f"{label} agentic_math.use_task_agent must not be false")
    validate_config_string_list_superset(
        errors=errors,
        section=agentic_math,
        field="deny_tool",
        required_values=NEMOCLAW_AGENTIC_CONFIG_REQUIRED_DENIED_TOOLS,
        label=f"{label} agentic_math",
    )
    validate_config_local_tools_allowed(
        errors=errors,
        section=agentic_math,
        field="deny_tool",
        protected_tools=NEMOCLAW_AGENTIC_CONFIG_REQUIRED_ALLOWED_LOCAL_TOOLS,
        label=f"{label} agentic_math",
    )
    validate_config_string_list_superset(
        errors=errors,
        section=agentic_math,
        field="deny_argument_pattern",
        required_values=NEMOCLAW_AGENTIC_CONFIG_REQUIRED_DENIED_ARGUMENT_PATTERNS,
        label=f"{label} agentic_math",
    )


def validate_swebench_nemoclaw_config_yaml(
    *,
    errors: list[str],
    payload: dict[str, Any],
    expected_sandbox: str | None,
    label: str,
) -> None:
    run = yaml_section(payload, "run")
    swebench_pro = yaml_section(payload, "swebench_pro")
    nemoclaw_keys = sorted(key for key in swebench_pro if str(key).startswith("nemoclaw"))
    if not nemoclaw_keys:
        return
    if run.get("swebench_pro") is not True:
        errors.append(f"{label} run.swebench_pro must be true")
    sandbox = swebench_pro.get("nemoclaw_sandbox")
    if expected_sandbox and sandbox != expected_sandbox:
        errors.append(
            f"{label} swebench_pro.nemoclaw_sandbox must be {expected_sandbox!r}"
        )
    elif not isinstance(sandbox, str) or not sandbox.strip():
        errors.append(f"{label} swebench_pro.nemoclaw_sandbox is missing")
    transfer_mode = swebench_pro.get("nemoclaw_checkout_transfer_mode")
    sandbox_root = swebench_pro.get("nemoclaw_checkout_sandbox_root")
    if transfer_mode != "copy" and not (
        isinstance(sandbox_root, str) and sandbox_root.strip()
    ):
        errors.append(
            f"{label} swebench_pro must set nemoclaw_checkout_transfer_mode='copy' "
            "or nemoclaw_checkout_sandbox_root"
        )
    validate_config_string_list_superset(
        errors=errors,
        section=swebench_pro,
        field="deny_tool",
        required_values=NEMOCLAW_AGENTIC_CONFIG_REQUIRED_DENIED_TOOLS,
        label=f"{label} swebench_pro",
    )
    validate_config_local_tools_allowed(
        errors=errors,
        section=swebench_pro,
        field="deny_tool",
        protected_tools=NEMOCLAW_AGENTIC_CONFIG_REQUIRED_ALLOWED_LOCAL_TOOLS,
        label=f"{label} swebench_pro",
    )
    validate_config_string_list_superset(
        errors=errors,
        section=swebench_pro,
        field="deny_argument_pattern",
        required_values=NEMOCLAW_AGENTIC_CONFIG_REQUIRED_DENIED_ARGUMENT_PATTERNS,
        label=f"{label} swebench_pro",
    )


def validate_bundled_nemoclaw_config_yaml(
    *,
    bundle_dir: Path,
    record: dict[str, Any],
    path_value: Any,
    errors: list[str],
    label: str,
) -> dict[str, Any] | None:
    bundle_path = record.get("bundle_path")
    if not isinstance(bundle_path, str) or not bundle_path:
        errors.append(f"{label} missing bundle_path: {source_path_key(path_value)}")
        return None
    try:
        return read_yaml_object(bundle_dir / bundle_path)
    except (OSError, ValueError) as exc:
        errors.append(f"{label} YAML is not readable: {source_path_key(path_value)}: {exc}")
        return None


def validate_nemoclaw_agentic_config_yaml_evidence(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    payload = read_bundled_nemoclaw_adoption_json(
        bundle_dir=bundle_dir,
        manifest=manifest,
        errors=errors,
    )
    if not isinstance(payload, dict):
        return errors
    adoption_decision = (
        payload.get("adoption_decision")
        if isinstance(payload.get("adoption_decision"), dict)
        else {}
    )
    expected_sandbox = (
        adoption_decision.get("sandbox")
        if isinstance(adoption_decision.get("sandbox"), str)
        else payload.get("sandbox")
        if isinstance(payload.get("sandbox"), str)
        else None
    )
    criteria = payload.get("criteria")
    if not isinstance(criteria, list):
        return errors
    records = file_records_by_source(manifest)
    for criterion in criteria:
        if not isinstance(criterion, dict) or criterion.get("ok") is not True:
            continue
        name = criterion.get("name")
        evidence_paths = (
            criterion.get("evidence_paths")
            if isinstance(criterion.get("evidence_paths"), list)
            else []
        )
        if name == "agentic_math_config":
            if not evidence_paths:
                errors.append(
                    "NeMoClaw agentic_math_config passed but evidence_paths is empty"
                )
            for evidence_path in evidence_paths:
                record = validate_any_file_role(
                    errors=errors,
                    records=records,
                    path_value=evidence_path,
                    role_suffix="agentic_math_config:evidence",
                    label="NeMoClaw Agentic Math config YAML evidence",
                )
                if not isinstance(record, dict):
                    continue
                config = validate_bundled_nemoclaw_config_yaml(
                    bundle_dir=bundle_dir,
                    record=record,
                    path_value=evidence_path,
                    errors=errors,
                    label="NeMoClaw Agentic Math config YAML evidence",
                )
                if isinstance(config, dict):
                    validate_agentic_math_nemoclaw_config_yaml(
                        errors=errors,
                        payload=config,
                        expected_sandbox=expected_sandbox,
                        label=f"NeMoClaw Agentic Math config {source_path_key(evidence_path)}",
                    )
        elif name == "swebench_pro_non_adoption_guard":
            for evidence_path in evidence_paths:
                record = validate_any_file_role(
                    errors=errors,
                    records=records,
                    path_value=evidence_path,
                    role_suffix="swebench_pro_non_adoption_guard:evidence",
                    label="NeMoClaw SWE-Bench Pro config YAML evidence",
                )
                if not isinstance(record, dict):
                    continue
                config = validate_bundled_nemoclaw_config_yaml(
                    bundle_dir=bundle_dir,
                    record=record,
                    path_value=evidence_path,
                    errors=errors,
                    label="NeMoClaw SWE-Bench Pro config YAML evidence",
                )
                if isinstance(config, dict):
                    validate_swebench_nemoclaw_config_yaml(
                        errors=errors,
                        payload=config,
                        expected_sandbox=expected_sandbox,
                        label=f"NeMoClaw SWE-Bench Pro config {source_path_key(evidence_path)}",
                    )
    return errors


def validate_nemoclaw_post_install_verification_evidence(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    current_gate = manifest.get("current_gate")
    if not isinstance(current_gate, dict):
        return errors
    runner = current_gate.get("runner_evidence")
    if not isinstance(runner, dict):
        return errors
    post_install = runner.get("nemoclaw_post_install_verification")
    if not isinstance(post_install, dict) or not post_install:
        return errors

    records = file_records_by_source(manifest)
    post_record = validate_file_role(
        errors=errors,
        records=records,
        path_value=post_install.get("path"),
        role="nemoclaw_post_install_verification",
        label="NeMoClaw post-install verification JSON",
    )
    markdown_path = post_install.get("markdown_path")
    if isinstance(markdown_path, str) and markdown_path.strip():
        validate_file_role(
            errors=errors,
            records=records,
            path_value=markdown_path,
            role="nemoclaw_post_install_verification_markdown",
            label="NeMoClaw post-install verification Markdown",
        )
    if not isinstance(post_record, dict):
        return errors
    bundle_path = post_record.get("bundle_path")
    if not isinstance(bundle_path, str) or not bundle_path:
        errors.append("NeMoClaw post-install verification JSON missing bundle_path")
        return errors
    try:
        payload = read_json_object(bundle_dir / bundle_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"NeMoClaw post-install verification JSON is not readable: {exc}")
        return errors

    if payload.get("schema_version") != 1:
        errors.append("NeMoClaw post-install verification schema_version must be 1")
    payload_path = payload.get("path")
    if isinstance(payload_path, str) and payload_path.strip():
        if source_path_key(post_install.get("path")) != source_path_key(payload_path):
            errors.append(
                "NeMoClaw post-install verification JSON path does not match "
                "runner_evidence"
            )
    else:
        errors.append("NeMoClaw post-install verification JSON path is missing")
    payload_markdown_path = payload.get("markdown_path")
    if isinstance(payload_markdown_path, str) and payload_markdown_path.strip():
        if source_path_key(markdown_path) != source_path_key(payload_markdown_path):
            errors.append(
                "NeMoClaw post-install verification JSON markdown_path does not "
                "match runner_evidence"
            )
    else:
        errors.append("NeMoClaw post-install verification JSON markdown_path is missing")
    for field in (
        "will_launch_model_inference",
        "will_query_wandb",
        "will_install_or_onboard",
    ):
        if payload.get(field) is not False:
            errors.append(f"NeMoClaw post-install verification {field} must be false")
    command_safety = payload.get("command_safety")
    if not isinstance(command_safety, dict):
        errors.append("NeMoClaw post-install verification command_safety is not an object")
    else:
        if command_safety.get("ok") is not True:
            errors.append("NeMoClaw post-install verification command_safety ok must be true")
        safety_label = "NeMoClaw post-install verification command_safety"
        validate_string_list_superset(
            errors=errors,
            payload=command_safety,
            field="forbidden_tokens",
            required_values=FORBIDDEN_NEMOCLAW_POST_INSTALL_COMMAND_EXACT_TOKENS,
            label=safety_label,
            missing_noun="token",
        )
        validate_string_list_superset(
            errors=errors,
            payload=command_safety,
            field="forbidden_prefixes",
            required_values=FORBIDDEN_NEMOCLAW_POST_INSTALL_COMMAND_PREFIXES,
            label=safety_label,
            missing_noun="prefix",
        )
        validate_string_list_superset(
            errors=errors,
            payload=command_safety,
            field="forbidden_markers",
            required_values=FORBIDDEN_NEMOCLAW_POST_INSTALL_COMMAND_MARKERS,
            label=safety_label,
            missing_noun="marker",
        )
        validate_required_step_tokens(
            errors=errors,
            payload=command_safety,
            field="required_step_tokens",
            required_steps=REQUIRED_NEMOCLAW_POST_INSTALL_COMMAND_TOKENS,
            label=safety_label,
        )
        for field in (
            "forbidden_token_count",
            "missing_required_token_count",
            "missing_command_count",
        ):
            if command_safety.get(field) != 0:
                errors.append(
                    "NeMoClaw post-install verification command_safety "
                    f"{field} must be 0"
                )
        safety_records = command_safety.get("records")
        if not isinstance(safety_records, list):
            errors.append("NeMoClaw post-install verification command_safety records is not a list")
        else:
            safety_by_name = {
                str(record.get("name")): record
                for record in safety_records
                if isinstance(record, dict) and record.get("name")
            }
            for required_step_name in (
                "setup_check",
                "protocol_preflight",
                "canary_readiness",
                "adoption_check",
            ):
                record = safety_by_name.get(required_step_name)
                if not isinstance(record, dict):
                    errors.append(
                        "NeMoClaw post-install verification command_safety "
                        f"missing record {required_step_name}"
                    )
                    continue
                if record.get("ok") is not True:
                    errors.append(
                        "NeMoClaw post-install verification command_safety "
                        f"record {required_step_name} ok must be true"
                    )
                if record.get("forbidden_tokens") not in ([], None):
                    errors.append(
                        "NeMoClaw post-install verification command_safety "
                        f"record {required_step_name} forbidden_tokens must be empty"
                    )
                if record.get("missing_required_tokens") not in ([], None):
                    errors.append(
                        "NeMoClaw post-install verification command_safety "
                        f"record {required_step_name} missing_required_tokens must be empty"
                    )
    if "ok" in post_install and post_install.get("ok") is not None:
        if bool(payload.get("ok")) != bool(post_install.get("ok")):
            errors.append("NeMoClaw post-install verification ok does not match runner_evidence")
    if post_install.get("status") is not None and payload.get("status") != post_install.get("status"):
        errors.append("NeMoClaw post-install verification status does not match runner_evidence")

    outputs = payload.get("outputs")
    required_outputs = {
        "setup_json",
        "preflight_json",
        "readiness_json",
        "adoption_json",
        "adoption_markdown",
    }
    if not isinstance(outputs, dict):
        errors.append("NeMoClaw post-install verification outputs is not an object")
        outputs = {}
    missing_outputs = sorted(
        name
        for name in required_outputs
        if not isinstance(outputs.get(name), str) or not outputs.get(name)
    )
    if missing_outputs:
        errors.append(
            "NeMoClaw post-install verification outputs missing: "
            + ", ".join(missing_outputs)
        )
    outputs_sha256 = payload.get("outputs_sha256")
    if not isinstance(outputs_sha256, dict):
        errors.append("NeMoClaw post-install verification outputs_sha256 is not an object")
        outputs_sha256 = {}
    missing_output_sha256 = sorted(
        name
        for name in required_outputs
        if not isinstance(outputs_sha256.get(name), str)
        or not re.fullmatch(r"[0-9a-f]{64}", str(outputs_sha256.get(name) or ""))
    )
    if missing_output_sha256:
        errors.append(
            "NeMoClaw post-install verification outputs_sha256 missing or invalid: "
            + ", ".join(missing_output_sha256)
        )
    output_records: dict[str, dict[str, Any]] = {}
    for name, path_value in outputs.items():
        if not isinstance(name, str) or not isinstance(path_value, str) or not path_value:
            continue
        record = validate_file_role(
            errors=errors,
            records=records,
            path_value=path_value,
            role=f"nemoclaw_post_install_verification:output:{name}",
            label=f"NeMoClaw post-install verification output {name}",
        )
        if isinstance(record, dict):
            output_records[name] = record
            bundle_path = record.get("bundle_path")
            expected_sha = outputs_sha256.get(name)
            if isinstance(bundle_path, str) and bundle_path and isinstance(expected_sha, str):
                try:
                    actual_sha = sha256_file(bundle_dir / bundle_path)
                except OSError as exc:
                    errors.append(
                        f"NeMoClaw post-install verification output {name} sha256 failed: {exc}"
                    )
                else:
                    if expected_sha and actual_sha != expected_sha:
                        errors.append(
                            f"NeMoClaw post-install verification output {name} "
                            "sha256 does not match outputs_sha256"
                        )

    output_payloads: dict[str, dict[str, Any]] = {}
    for name in ("setup_json", "preflight_json", "readiness_json", "adoption_json"):
        record = output_records.get(name)
        if not isinstance(record, dict):
            continue
        bundle_path = record.get("bundle_path")
        if not isinstance(bundle_path, str) or not bundle_path:
            errors.append(f"NeMoClaw post-install verification output {name} missing bundle_path")
            continue
        try:
            output_payload = read_json_object(bundle_dir / bundle_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(
                f"NeMoClaw post-install verification output {name} is not readable: {exc}"
            )
            continue
        if not isinstance(output_payload.get("ok"), bool):
            errors.append(f"NeMoClaw post-install verification output {name} ok is not a bool")
        output_payloads[name] = output_payload
    readiness_payload = output_payloads.get("readiness_json")
    if isinstance(readiness_payload, dict):
        errors.extend(validate_nemoclaw_canary_readiness_remote_lookup_checks(readiness_payload))
        errors.extend(validate_nemoclaw_canary_readiness_runtime_policy_checks(readiness_payload))
    adoption_payload = output_payloads.get("adoption_json")
    if isinstance(adoption_payload, dict):
        errors.extend(validate_nemoclaw_adoption_runtime_policy_criterion(adoption_payload))

    steps = payload.get("steps")
    required_steps = {
        "setup_check": "setup_json",
        "protocol_preflight": "preflight_json",
        "canary_readiness": "readiness_json",
        "adoption_check": "adoption_json",
    }
    if not isinstance(steps, list):
        errors.append("NeMoClaw post-install verification steps is not a list")
        return errors
    steps_by_name: dict[str, dict[str, Any]] = {
        str(step.get("name")): step
        for step in steps
        if isinstance(step, dict) and step.get("name")
    }
    required_command_tokens = {
        "setup_check": ("install_nemoclaw.sh", "--check-only", "--json"),
        "protocol_preflight": ("run_openclaw_agent_protocol.py", "preflight"),
        "canary_readiness": (
            "check_taiwan_canary_readiness.py",
            "--require-nemoclaw",
            "--json",
        ),
        "adoption_check": (
            "check_taiwan_nemoclaw_adoption.py",
            "--setup-json",
            "--readiness-json",
            "--json",
            "--markdown",
        ),
    }
    for step_name, output_name in required_steps.items():
        step = steps_by_name.get(step_name)
        if not isinstance(step, dict):
            errors.append(f"NeMoClaw post-install verification missing step {step_name}")
            continue
        command = step.get("command")
        if not isinstance(command, list) or not command:
            errors.append(f"NeMoClaw post-install verification step {step_name} command is missing")
            command_tokens: list[str] = []
        else:
            command_tokens = [str(token) for token in command]
            for token in forbidden_nemoclaw_post_install_command_tokens(command_tokens):
                errors.append(
                    f"NeMoClaw post-install verification step {step_name} "
                    f"contains forbidden command token {token}"
                )
            command_text = " ".join(command_tokens)
            for required_token in required_command_tokens.get(step_name, ()):
                if required_token not in command_text:
                    errors.append(
                        f"NeMoClaw post-install verification step {step_name} "
                        f"command missing required token {required_token}"
                    )
        if not isinstance(step.get("returncode"), int):
            errors.append(f"NeMoClaw post-install verification step {step_name} returncode is not an int")
            returncode_ok = None
        else:
            returncode_ok = step.get("returncode") == 0
        if not isinstance(step.get("returncode_ok"), bool):
            errors.append(f"NeMoClaw post-install verification step {step_name} returncode_ok is not a bool")
        elif returncode_ok is not None and step.get("returncode_ok") != returncode_ok:
            errors.append(
                f"NeMoClaw post-install verification step {step_name} returncode_ok "
                "does not match returncode"
            )
        if not isinstance(step.get("payload_ok"), bool):
            errors.append(f"NeMoClaw post-install verification step {step_name} payload_ok is not a bool")
        if not isinstance(step.get("payload_contract_ok"), bool):
            errors.append(
                f"NeMoClaw post-install verification step {step_name} "
                "payload_contract_ok is not a bool"
            )
        payload_contract_errors = step.get("payload_contract_errors")
        if not isinstance(payload_contract_errors, list) or not all(
            isinstance(item, str) for item in payload_contract_errors
        ):
            errors.append(
                f"NeMoClaw post-install verification step {step_name} "
                "payload_contract_errors is not a string list"
            )
        elif step.get("payload_contract_ok") is True and payload_contract_errors:
            errors.append(
                f"NeMoClaw post-install verification step {step_name} "
                "payload_contract_errors must be empty when payload_contract_ok=true"
            )
        elif step.get("payload_contract_ok") is False and not payload_contract_errors:
            errors.append(
                f"NeMoClaw post-install verification step {step_name} "
                "payload_contract_errors must be non-empty when payload_contract_ok=false"
            )
        if not isinstance(step.get("timed_out"), bool):
            errors.append(f"NeMoClaw post-install verification step {step_name} timed_out is not a bool")
        if not isinstance(step.get("ok"), bool):
            errors.append(f"NeMoClaw post-install verification step {step_name} ok is not a bool")
        if step.get("output_json") != outputs.get(output_name):
            errors.append(
                f"NeMoClaw post-install verification step {step_name} output_json "
                f"does not match outputs.{output_name}"
            )
        step_sha = step.get("output_json_sha256")
        expected_output_sha = outputs_sha256.get(output_name)
        if not isinstance(step_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", step_sha):
            errors.append(
                f"NeMoClaw post-install verification step {step_name} "
                "output_json_sha256 is missing or invalid"
            )
        elif isinstance(expected_output_sha, str) and step_sha != expected_output_sha:
            errors.append(
                f"NeMoClaw post-install verification step {step_name} "
                f"output_json_sha256 does not match outputs_sha256.{output_name}"
            )
        step_record = validate_file_role(
            errors=errors,
            records=records,
            path_value=step.get("output_json"),
            role=f"nemoclaw_post_install_verification:step:{step_name}",
            label=f"NeMoClaw post-install verification step {step_name} output",
        )
        output_record = output_records.get(output_name)
        if (
            isinstance(step_record, dict)
            and isinstance(output_record, dict)
            and step_record.get("bundle_path") != output_record.get("bundle_path")
        ):
            errors.append(
                f"NeMoClaw post-install verification step {step_name} output bundle_path "
                f"does not match outputs.{output_name}"
            )
        if isinstance(step_record, dict):
            step_bundle_path = step_record.get("bundle_path")
            if isinstance(step_bundle_path, str) and step_bundle_path and isinstance(step_sha, str):
                try:
                    actual_step_sha = sha256_file(bundle_dir / step_bundle_path)
                except OSError as exc:
                    errors.append(
                        f"NeMoClaw post-install verification step {step_name} "
                        f"output sha256 failed: {exc}"
                    )
                else:
                    if re.fullmatch(r"[0-9a-f]{64}", step_sha) and actual_step_sha != step_sha:
                        errors.append(
                            f"NeMoClaw post-install verification step {step_name} "
                            "output_json_sha256 does not match bundled output"
                        )
        output_payload = output_payloads.get(output_name)
        if isinstance(output_payload, dict):
            payload_ok = output_payload.get("ok")
            if isinstance(payload_ok, bool):
                if step.get("payload_ok") != payload_ok:
                    errors.append(
                        f"NeMoClaw post-install verification step {step_name} payload_ok "
                        f"does not match output {output_name}.ok"
                    )
                if (
                    isinstance(returncode_ok, bool)
                    and isinstance(step.get("ok"), bool)
                    and isinstance(step.get("payload_contract_ok"), bool)
                ):
                    expected_ok = returncode_ok and payload_ok and step["payload_contract_ok"]
                    if step.get("ok") != expected_ok:
                        errors.append(
                            f"NeMoClaw post-install verification step {step_name} ok "
                            "does not match returncode_ok, payload_ok, and payload_contract_ok"
                        )
            if output_payload.get("status") is not None and step.get("payload_status") != output_payload.get("status"):
                errors.append(
                    f"NeMoClaw post-install verification step {step_name} payload_status "
                    f"does not match output {output_name}.status"
                )
    return errors


def add_wandb_completion_path(
    paths: dict[str, str | None],
    *,
    path_value: Any,
    benchmark: Any = None,
) -> None:
    key = source_path_key(path_value)
    if not key:
        return
    paths[key] = str(benchmark) if isinstance(benchmark, str) and benchmark else paths.get(key)


def add_wandb_completion_proof(
    proofs: dict[str, dict[str, Any]],
    *,
    path_value: Any,
    benchmark: Any = None,
    source: str,
    require_run_metadata: bool,
) -> None:
    key = source_path_key(path_value)
    if not key:
        return
    proof = proofs.setdefault(
        key,
        {
            "benchmark": None,
            "sources": [],
            "require_run_metadata": False,
        },
    )
    if isinstance(benchmark, str) and benchmark:
        proof["benchmark"] = benchmark
    sources = proof.get("sources")
    if isinstance(sources, list) and source not in sources:
        sources.append(source)
    proof["require_run_metadata"] = bool(proof.get("require_run_metadata")) or require_run_metadata


def collect_wandb_completion_proofs(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    current_gate = manifest.get("current_gate")
    if not isinstance(current_gate, dict):
        return {}
    proofs: dict[str, dict[str, Any]] = {}

    benchmark_completion = current_gate.get("benchmark_completion")
    if isinstance(benchmark_completion, list):
        for row in benchmark_completion:
            if not isinstance(row, dict):
                continue
            benchmark = row.get("benchmark")
            for field, require_run_metadata in (
                ("standalone_records", False),
                ("review_entries", True),
            ):
                if field == "standalone_records" and row.get("standalone_ok") is not True:
                    continue
                if field == "review_entries" and row.get("review_ok") is not True:
                    continue
                records = row.get(field)
                if not isinstance(records, list):
                    continue
                for record in records:
                    if not isinstance(record, dict):
                        continue
                    if record.get("ok") is True or record.get("schema_valid") is True:
                        add_wandb_completion_proof(
                            proofs,
                            path_value=record.get("path"),
                            benchmark=benchmark,
                            source=f"benchmark_completion:{field}",
                            require_run_metadata=require_run_metadata,
                        )

    contract = current_gate.get("wandb_completion_contract")
    benchmarks = contract.get("benchmarks") if isinstance(contract, dict) else None
    if isinstance(benchmarks, list):
        for row in benchmarks:
            if not isinstance(row, dict):
                continue
            benchmark = row.get("benchmark")
            for field, require_run_metadata in (
                ("standalone_completion_paths", False),
                ("review_completion_paths", True),
                ("formalized_existing_completion_paths", False),
            ):
                if (
                    field == "standalone_completion_paths"
                    and row.get("standalone_completion_ok") is not True
                ):
                    continue
                if (
                    field == "review_completion_paths"
                    and row.get("review_completion_ok") is not True
                ):
                    continue
                if (
                    field == "formalized_existing_completion_paths"
                    and row.get("formalized_existing_result") is not True
                ):
                    continue
                values = row.get(field)
                if isinstance(values, list):
                    for value in values:
                        add_wandb_completion_proof(
                            proofs,
                            path_value=value,
                            benchmark=benchmark,
                            source=f"wandb_completion_contract:{field}",
                            require_run_metadata=require_run_metadata,
                        )

    adoption_draft = current_gate.get("wandb_adoption_draft")
    candidates = adoption_draft.get("candidates") if isinstance(adoption_draft, dict) else None
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            if candidate.get("sync_ready") is False:
                continue
            add_wandb_completion_proof(
                proofs,
                path_value=candidate.get("wandb_completion_json"),
                benchmark=candidate.get("benchmark"),
                source="wandb_adoption_draft:candidate",
                require_run_metadata=False,
            )

    paid_review_package = current_gate.get("paid_run_review_package")
    gates = paid_review_package.get("gates") if isinstance(paid_review_package, dict) else None
    if isinstance(gates, list):
        for gate in gates:
            if not isinstance(gate, dict):
                continue
            records = gate.get("records")
            if not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, dict):
                    continue
                entries = record.get("wandb_completion_entries")
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    if entry.get("verified") is True or entry.get("ok") is True:
                        add_wandb_completion_proof(
                            proofs,
                            path_value=entry.get("path"),
                            benchmark=entry.get("benchmark"),
                            source="paid_run_review_package:wandb_completion_entries",
                            require_run_metadata=True,
                        )

    return proofs


def collect_wandb_completion_proof_paths(manifest: dict[str, Any]) -> dict[str, str | None]:
    return {
        source_path: proof.get("benchmark")
        for source_path, proof in collect_wandb_completion_proofs(manifest).items()
    }


def nonempty_unique_strings(values: list[Any]) -> list[str]:
    return unique_strings(
        [
            str(value)
            for value in values
            if isinstance(value, str) and value.strip()
        ]
    )


def path_key_list(values: list[Any]) -> list[str]:
    return unique_strings(
        [
            source_path_key(value)
            for value in values
            if source_path_key(value)
        ]
    )


def validate_wandb_completion_contract_consistency(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    current_gate = manifest.get("current_gate")
    if not isinstance(current_gate, dict):
        return errors
    contract = current_gate.get("wandb_completion_contract")
    if not isinstance(contract, dict):
        return errors
    benchmarks = contract.get("benchmarks")
    if not isinstance(benchmarks, list):
        errors.append("wandb_completion_contract benchmarks is not a list")
        benchmarks = []
    required_benchmarks = contract.get("required_benchmarks")
    if not isinstance(required_benchmarks, list) or not all(
        isinstance(item, str) and item.strip() for item in required_benchmarks
    ):
        errors.append("wandb_completion_contract required_benchmarks is not a string list")
        required_benchmarks = []
    benchmark_completion = current_gate.get("benchmark_completion")
    if not isinstance(benchmark_completion, list):
        errors.append("manifest current_gate benchmark_completion is not a list")
        benchmark_completion = []
    benchmark_sources = {
        str(row.get("benchmark")): row
        for row in benchmark_completion
        if isinstance(row, dict) and isinstance(row.get("benchmark"), str) and row.get("benchmark")
    }
    for benchmark, source in benchmark_sources.items():
        if not isinstance(source.get("required"), bool):
            errors.append(f"benchmark_completion {benchmark} required is not boolean")
    existing_results = current_gate.get("existing_results_formalization")
    formalized_records = (
        existing_results.get("formalized_records")
        if isinstance(existing_results, dict)
        and isinstance(existing_results.get("formalized_records"), list)
        else []
    )
    formalized_by_benchmark: dict[str, list[dict[str, Any]]] = {}
    for record in formalized_records:
        if not isinstance(record, dict):
            continue
        benchmark = record.get("benchmark")
        if isinstance(benchmark, str) and benchmark:
            formalized_by_benchmark.setdefault(benchmark, []).append(record)

    benchmark_names: list[str] = []
    required_rows: list[dict[str, Any]] = []
    rows_by_benchmark: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(benchmarks, start=1):
        if not isinstance(row, dict):
            errors.append(f"wandb_completion_contract benchmark row {index} is not an object")
            continue
        benchmark = row.get("benchmark")
        if not isinstance(benchmark, str) or not benchmark.strip():
            errors.append(f"wandb_completion_contract benchmark row {index} benchmark is missing")
            continue
        if benchmark in benchmark_names:
            errors.append(f"wandb_completion_contract benchmark row is duplicated: {benchmark}")
        benchmark_names.append(benchmark)
        rows_by_benchmark[benchmark] = row
        if not isinstance(row.get("required"), bool):
            errors.append(f"wandb_completion_contract benchmark {benchmark} required is not boolean")
        if row.get("required") is True:
            required_rows.append(row)

    expected_benchmarks = unique_strings(
        [
            *[str(item) for item in required_benchmarks],
            *list(benchmark_sources.keys()),
            *list(formalized_by_benchmark.keys()),
        ]
    )
    if benchmark_names != expected_benchmarks:
        errors.append("wandb_completion_contract benchmarks do not match source benchmark set")

    required_row_names = [str(row.get("benchmark")) for row in required_rows]
    if required_row_names != required_benchmarks:
        errors.append(
            "wandb_completion_contract required_benchmarks does not match required benchmark rows"
        )

    missing_release = [
        str(row.get("benchmark"))
        for row in required_rows
        if row.get("release_completion_proven") is not True
    ]
    expected_complete = bool(required_rows) and not missing_release
    expected_status = (
        "passed"
        if expected_complete
        else ("no_required_benchmarks" if not required_rows else "incomplete")
    )
    expected_counts = {
        "required_count": len(required_rows),
        "release_completion_proven_count": sum(
            1 for row in required_rows if row.get("release_completion_proven") is True
        ),
        "standalone_completion_ok_count": sum(
            1 for row in required_rows if row.get("standalone_completion_ok") is True
        ),
        "formalized_existing_result_count": sum(
            1 for row in required_rows if row.get("formalized_existing_result") is True
        ),
        "next_action_count": sum(
            len(row.get("next_actions") or [])
            for row in required_rows
            if isinstance(row.get("next_actions") or [], list)
        ),
    }
    for field, expected in expected_counts.items():
        if contract.get(field) != expected:
            errors.append(f"wandb_completion_contract {field} does not match benchmarks")
    if contract.get("missing_release_completion_benchmarks") != missing_release:
        errors.append(
            "wandb_completion_contract missing_release_completion_benchmarks does not match benchmarks"
        )
    if contract.get("complete") is not expected_complete:
        errors.append("wandb_completion_contract complete does not match benchmarks")
    if contract.get("status") != expected_status:
        errors.append("wandb_completion_contract status does not match benchmarks")

    def _row_string_list(row: dict[str, Any], field: str) -> list[str]:
        value = row.get(field)
        if not isinstance(value, list):
            return []
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]

    adoption_draft = current_gate.get("wandb_adoption_draft")
    draft_sync_ready_by_benchmark: dict[str, list[dict[str, Any]]] = {}
    if isinstance(adoption_draft, dict) and adoption_draft.get("skipped") is not True:
        draft_candidates = adoption_draft.get("candidates")
        if isinstance(draft_candidates, list):
            for candidate in draft_candidates:
                if not isinstance(candidate, dict) or candidate.get("sync_ready") is not True:
                    continue
                candidate_benchmark = candidate.get("benchmark")
                if isinstance(candidate_benchmark, str) and candidate_benchmark.strip():
                    draft_sync_ready_by_benchmark.setdefault(
                        candidate_benchmark, []
                    ).append(candidate)

    for row in [row for row in benchmarks if isinstance(row, dict)]:
        benchmark = str(row.get("benchmark"))
        source = benchmark_sources.get(benchmark, {})
        source_present = benchmark in benchmark_sources
        standalone_records = (
            source.get("standalone_records")
            if isinstance(source, dict) and isinstance(source.get("standalone_records"), list)
            else []
        )
        review_entries = (
            source.get("review_entries")
            if isinstance(source, dict) and isinstance(source.get("review_entries"), list)
            else []
        )
        formalized = formalized_by_benchmark.get(benchmark, [])
        expected_source_fields: dict[str, Any] = {
            "release_completion_proven": bool(source.get("completion_proven"))
            if isinstance(source, dict)
            else False,
            "standalone_completion_ok": bool(source.get("standalone_ok"))
            if isinstance(source, dict)
            else False,
            "standalone_status": source.get("standalone_status")
            if isinstance(source, dict)
            else None,
            "review_completion_ok": bool(source.get("review_ok"))
            if isinstance(source, dict)
            else False,
            "review_status": source.get("review_status")
            if isinstance(source, dict)
            else None,
            "standalone_run_ids": nonempty_unique_strings(
                [
                    record.get("run_id")
                    for record in standalone_records
                    if isinstance(record, dict) and record.get("ok") is True
                ]
            ),
            "standalone_completion_paths": path_key_list(
                [
                    record.get("path")
                    for record in standalone_records
                    if isinstance(record, dict)
                ]
            ),
            "review_run_ids": nonempty_unique_strings(
                [
                    entry.get("run_id")
                    for entry in review_entries
                    if isinstance(entry, dict) and entry.get("ok") is True
                ]
            ),
            "review_completion_paths": path_key_list(
                [
                    entry.get("path")
                    for entry in review_entries
                    if isinstance(entry, dict)
                ]
            ),
            "formalized_existing_result": bool(formalized),
            "formalized_existing_run_ids": nonempty_unique_strings(
                [
                    (record.get("wandb_completion") or {}).get("run_id")
                    for record in formalized
                    if isinstance(record, dict)
                    and isinstance(record.get("wandb_completion"), dict)
                ]
            ),
            "formalized_existing_completion_paths": path_key_list(
                [
                    (record.get("wandb_completion") or {}).get("path")
                    for record in formalized
                    if isinstance(record, dict)
                    and isinstance(record.get("wandb_completion"), dict)
                ]
            ),
            "formalized_existing_count": len(formalized),
        }
        if source_present:
            expected_source_fields["required"] = bool(source.get("required"))
        for field, expected_value in expected_source_fields.items():
            actual_value = row.get(field)
            if field.endswith("_paths"):
                actual_value = path_key_list(actual_value if isinstance(actual_value, list) else [])
            if actual_value != expected_value:
                errors.append(
                    "wandb_completion_contract benchmark "
                    f"{benchmark} {field} does not match source evidence"
                )
        expected_row_status = (
            "release_complete"
            if row.get("release_completion_proven") is True
            else (
                "formalized_but_not_reviewed"
                if row.get("standalone_completion_ok") is True
                or row.get("formalized_existing_result") is True
                else (
                    "missing_release_completion"
                    if row.get("required") is True
                    else "not_required"
                )
            )
        )
        if row.get("status") != expected_row_status:
            errors.append(
                f"wandb_completion_contract benchmark {benchmark} status does not match evidence flags"
            )
        if row.get("release_completion_proven") is True:
            for field in ("standalone_completion_ok", "review_completion_ok"):
                if row.get(field) is not True:
                    errors.append(
                        "wandb_completion_contract benchmark "
                        f"{benchmark} release_completion_proven=true requires {field}=true"
                    )
        recommended_commands = _row_string_list(row, "recommended_commands")
        if benchmark in NEMOCLAW_AUDIT_REQUIRED_BENCHMARKS:
            expected_total = "100" if benchmark == "agentic_math" else "80"
            for command in recommended_commands:
                if "verify_taiwan_wandb_completion.py" not in command:
                    continue
                if "--require-nemoclaw-session-audit" not in command:
                    errors.append(
                        "wandb_completion_contract benchmark "
                        f"{benchmark} verify command missing --require-nemoclaw-session-audit"
                    )
                if command_option_value(command, "--expected-total") != expected_total:
                    errors.append(
                        "wandb_completion_contract benchmark "
                        f"{benchmark} verify command missing --expected-total {expected_total}"
                    )
        sync_ready_count = row.get("sync_ready_adoption_candidate_count")
        sync_ready = False
        if sync_ready_count is not None:
            if type(sync_ready_count) is not int or sync_ready_count < 0:
                errors.append(
                    "wandb_completion_contract benchmark "
                    f"{benchmark} sync_ready_adoption_candidate_count is not a non-negative int"
                )
            else:
                sync_ready = sync_ready_count > 0
        draft_sync_ready_candidates = draft_sync_ready_by_benchmark.get(benchmark, [])
        expected_sync_ready_count = len(draft_sync_ready_candidates)
        if sync_ready_count is not None and sync_ready_count != expected_sync_ready_count:
            errors.append(
                "wandb_completion_contract benchmark "
                f"{benchmark} sync_ready_adoption_candidate_count does not "
                "match W&B adoption draft"
            )
        if expected_sync_ready_count and sync_ready_count is None:
            errors.append(
                "wandb_completion_contract benchmark "
                f"{benchmark} sync_ready_adoption_candidate_count does not "
                "match W&B adoption draft"
            )
        if sync_ready:
            for row_field, candidate_field in (
                (
                    "scope_attestation_template_paths",
                    "scope_attestation_template_json",
                ),
                (
                    "scope_attestation_render_report_paths",
                    "scope_attestation_render_report_json",
                ),
                (
                    "scope_attestation_preflight_report_paths",
                    "scope_attestation_preflight_report_json",
                ),
                ("sync_dry_run_report_paths", "sync_dry_run_report_json"),
            ):
                expected_paths = path_key_list(
                    [candidate.get(candidate_field) for candidate in draft_sync_ready_candidates]
                )
                actual_paths = path_key_list(
                    row.get(row_field) if isinstance(row.get(row_field), list) else []
                )
                if actual_paths != expected_paths:
                    errors.append(
                        "wandb_completion_contract benchmark "
                        f"{benchmark} {row_field} does not match W&B adoption draft"
                    )
            candidate_run_ids = nonempty_unique_strings(
                [candidate.get("wandb_run_id") for candidate in draft_sync_ready_candidates]
            )
            contract_run_ids = nonempty_unique_strings(
                [
                    *_row_string_list(row, "standalone_run_ids"),
                    *_row_string_list(row, "formalized_existing_run_ids"),
                ]
            )
            missing_candidate_run_ids = [
                run_id for run_id in candidate_run_ids if run_id not in contract_run_ids
            ]
            if missing_candidate_run_ids:
                errors.append(
                    "wandb_completion_contract benchmark "
                    f"{benchmark} sync-ready adoption run_ids are not present "
                    "in completion evidence"
                )
            candidate_completion_paths = path_key_list(
                [candidate.get("wandb_completion_json") for candidate in draft_sync_ready_candidates]
            )
            contract_completion_paths = path_key_list(
                [
                    *_row_string_list(row, "standalone_completion_paths"),
                    *_row_string_list(row, "formalized_existing_completion_paths"),
                ]
            )
            missing_candidate_completion_paths = [
                path for path in candidate_completion_paths if path not in contract_completion_paths
            ]
            if missing_candidate_completion_paths:
                errors.append(
                    "wandb_completion_contract benchmark "
                    f"{benchmark} sync-ready adoption completion paths are not "
                    "present in completion evidence"
                )
            if row.get("scope_confirmation_required") is not True:
                errors.append(
                    "wandb_completion_contract benchmark "
                    f"{benchmark} sync-ready adoption requires scope_confirmation_required=true"
                )
            if not isinstance(row.get("scope_warning"), str) or not row.get("scope_warning", "").strip():
                errors.append(
                    "wandb_completion_contract benchmark "
                    f"{benchmark} sync-ready adoption requires a scope_warning"
                )
            required_command_checks = [
                (
                    "scope attestation render command",
                    lambda command: "render_wandb_scope_attestation.py" in command
                    and "--template-json" in command
                    and "--output-json" in command,
                ),
                (
                    "scope attestation preflight command",
                    lambda command: "verify_wandb_scope_attestation.py" in command
                    and "--scope-attestation-json" in command,
                ),
                (
                    "adopt-existing dry-run sync command",
                    lambda command: "sync_wandb_completion_to_paid_review.py" in command
                    and "--adopt-existing-result" in command
                    and "--scope-attestation-json" in command
                    and "--report-json" in command,
                ),
                (
                    "adopt-existing apply sync command",
                    lambda command: "sync_wandb_completion_to_paid_review.py" in command
                    and "--in-place" in command
                    and "--adopt-existing-result" in command
                    and "--scope-attestation-json" in command
                    and "--validated-dry-run-report-json" in command,
                ),
            ]
            for label, command_check in required_command_checks:
                if not any(command_check(command) for command in recommended_commands):
                    errors.append(
                        "wandb_completion_contract benchmark "
                        f"{benchmark} sync-ready adoption recommended_commands missing {label}"
                    )
            if row.get("standalone_completion_ok") is not True:
                refresh_commands = _row_string_list(row, "refresh_wandb_completion_commands")
                if not refresh_commands:
                    errors.append(
                        "wandb_completion_contract benchmark "
                        f"{benchmark} stale sync-ready adoption requires "
                        "refresh_wandb_completion_commands"
                    )
                for index, command in enumerate(refresh_commands, start=1):
                    if (
                        "verify_taiwan_wandb_completion.py" not in command
                        or "--run-id" not in command
                        or "--json" not in command
                        or "RUN_ID" in command
                    ):
                        errors.append(
                            "wandb_completion_contract benchmark "
                            f"{benchmark} stale sync-ready adoption refresh command "
                            f"{index} is not a concrete verify_taiwan_wandb_completion.py command"
                        )
                    if benchmark in NEMOCLAW_AUDIT_REQUIRED_BENCHMARKS:
                        expected_total = "100" if benchmark == "agentic_math" else "80"
                        if "--require-nemoclaw-session-audit" not in command:
                            errors.append(
                                "wandb_completion_contract benchmark "
                                f"{benchmark} stale sync-ready adoption refresh command "
                                f"{index} missing --require-nemoclaw-session-audit"
                            )
                        if command_option_value(command, "--expected-total") != expected_total:
                            errors.append(
                                "wandb_completion_contract benchmark "
                                f"{benchmark} stale sync-ready adoption refresh command "
                                f"{index} missing --expected-total {expected_total}"
                            )
                for candidate in draft_sync_ready_candidates:
                    candidate_run_id = candidate.get("wandb_run_id")
                    candidate_completion_path = source_path_key(
                        candidate.get("wandb_completion_json")
                    )
                    candidate_entity = candidate.get("wandb_entity")
                    candidate_project = candidate.get("wandb_project")
                    if not (
                        isinstance(candidate_run_id, str)
                        and candidate_run_id.strip()
                        and candidate_completion_path
                    ):
                        continue
                    matching_refresh_command = False
                    for command in refresh_commands:
                        if command_option_value(command, "--run-id") != candidate_run_id:
                            continue
                        if source_path_key(command_option_value(command, "--json")) != candidate_completion_path:
                            continue
                        if (
                            isinstance(candidate_entity, str)
                            and candidate_entity.strip()
                            and command_option_value(command, "--entity") != candidate_entity
                        ):
                            continue
                        if (
                            isinstance(candidate_project, str)
                            and candidate_project.strip()
                            and command_option_value(command, "--project") != candidate_project
                        ):
                            continue
                        matching_refresh_command = True
                        break
                    if not matching_refresh_command:
                        errors.append(
                            "wandb_completion_contract benchmark "
                            f"{benchmark} stale sync-ready adoption refresh commands "
                            "do not match W&B adoption draft candidate"
                        )
                if any(command not in recommended_commands for command in refresh_commands):
                    errors.append(
                        "wandb_completion_contract benchmark "
                        f"{benchmark} refresh_wandb_completion_commands are not included "
                        "in recommended_commands"
                    )
                next_actions = _row_string_list(row, "next_actions")
                if not any("refresh W&B completion verifier" in action for action in next_actions):
                    errors.append(
                        "wandb_completion_contract benchmark "
                        f"{benchmark} stale sync-ready adoption next_actions missing "
                        "refresh W&B completion verifier"
                    )
    return errors


def validate_wandb_completion_payload(
    payload: dict[str, Any],
    *,
    label: str,
    expected_benchmark: str | None,
) -> list[str]:
    errors: list[str] = []
    if payload.get("ok") is not True:
        errors.append(f"{label} ok is not true")
    schema_version = payload.get("schema_version")
    if schema_version is not None and schema_version != 1:
        errors.append(f"{label} schema_version is not 1")
    if payload.get("verification_schema_version") != 1:
        errors.append(f"{label} verification_schema_version is not 1")
    benchmark = payload.get("benchmark")
    if expected_benchmark and benchmark != expected_benchmark:
        errors.append(f"{label} benchmark mismatch: expected {expected_benchmark}, got {benchmark}")
    if not isinstance(payload.get("run_id"), str) or not payload.get("run_id"):
        errors.append(f"{label} run_id is missing")
    for field in ("entity", "project"):
        if not isinstance(payload.get(field), str) or not payload.get(field).strip():
            errors.append(f"{label} {field} is missing")
    if not isinstance(payload.get("generated_at"), (int, float)):
        errors.append(f"{label} generated_at is not numeric")
    errors.extend(validate_wandb_completion_query_source(payload, label=label))
    required_evidence = payload.get("required_evidence")
    if not isinstance(required_evidence, dict):
        errors.append(f"{label} required_evidence is not an object")
        required_evidence = {}
    observed = payload.get("observed_evidence")
    if not isinstance(observed, dict):
        errors.append(f"{label} observed_evidence is not an object")
        return errors
    if observed.get("run_state") != "finished":
        errors.append(f"{label} observed_evidence.run_state is not finished")
    errors.extend(
        validate_wandb_completion_nemoclaw_session_audit(
            payload=payload,
            required=required_evidence,
            observed=observed,
            label=label,
        )
    )
    errors.extend(
        validate_wandb_observed_evidence(
            required=required_evidence,
            observed=observed,
            label=label,
        )
    )
    errors.extend(
        validate_wandb_completion_checks(
            payload=payload,
            required=required_evidence,
            observed=observed,
            label=label,
        )
    )
    return errors


def validate_nemoclaw_operator_docs_evidence(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    current_gate = manifest.get("current_gate")
    if not isinstance(current_gate, dict):
        return errors
    runner = current_gate.get("runner_evidence")
    if not isinstance(runner, dict):
        return errors
    operator_docs = runner.get("nemoclaw_operator_docs_verification")
    if not isinstance(operator_docs, dict) or not operator_docs:
        return errors

    records = file_records_by_source(manifest)
    docs_record = validate_file_role(
        errors=errors,
        records=records,
        path_value=operator_docs.get("path"),
        role="nemoclaw_operator_docs_verification",
        label="NeMoClaw operator docs verification JSON",
    )
    markdown_path = operator_docs.get("markdown_path")
    if isinstance(markdown_path, str) and markdown_path.strip():
        validate_file_role(
            errors=errors,
            records=records,
            path_value=markdown_path,
            role="nemoclaw_operator_docs_verification_markdown",
            label="NeMoClaw operator docs verification Markdown",
        )
    if not isinstance(docs_record, dict):
        return errors
    bundle_path = docs_record.get("bundle_path")
    if not isinstance(bundle_path, str) or not bundle_path:
        errors.append("NeMoClaw operator docs verification JSON missing bundle_path")
        return errors
    try:
        payload = read_json_object(bundle_dir / bundle_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"NeMoClaw operator docs verification JSON is not readable: {exc}")
        return errors

    if payload.get("schema_version") != 1:
        errors.append("NeMoClaw operator docs verification schema_version must be 1")
    payload_path = payload.get("path")
    if isinstance(payload_path, str) and payload_path.strip():
        if source_path_key(operator_docs.get("path")) != source_path_key(payload_path):
            errors.append(
                "NeMoClaw operator docs verification JSON path does not match "
                "runner_evidence"
            )
    else:
        errors.append("NeMoClaw operator docs verification JSON path is missing")
    payload_markdown_path = payload.get("markdown_path")
    if isinstance(payload_markdown_path, str) and payload_markdown_path.strip():
        if source_path_key(markdown_path) != source_path_key(payload_markdown_path):
            errors.append(
                "NeMoClaw operator docs verification JSON markdown_path does not "
                "match runner_evidence"
            )
    else:
        errors.append("NeMoClaw operator docs verification JSON markdown_path is missing")
    for field in (
        "will_execute_installer",
        "will_install_or_onboard",
        "will_launch_model_inference",
        "will_query_wandb",
    ):
        if payload.get(field) is not False:
            errors.append(f"NeMoClaw operator docs verification {field} must be false")
    if "ok" in operator_docs and operator_docs.get("ok") is not None:
        if bool(payload.get("ok")) != bool(operator_docs.get("ok")):
            errors.append("NeMoClaw operator docs verification ok does not match runner_evidence")
    if operator_docs.get("status") is not None and payload.get("status") != operator_docs.get("status"):
        errors.append("NeMoClaw operator docs verification status does not match runner_evidence")
    checks = payload.get("checks")
    if not isinstance(checks, list) or not checks:
        errors.append("NeMoClaw operator docs verification checks must be a non-empty list")
    else:
        check_names: set[str] = set()
        for check in checks:
            if not isinstance(check, dict):
                errors.append("NeMoClaw operator docs verification contains a non-object check")
                continue
            check_name = str(check.get("name") or "")
            if check_name:
                check_names.add(check_name)
            if payload.get("ok") is True and check.get("ok") is not True:
                errors.append(
                    "NeMoClaw operator docs verification has ok=true but a check failed: "
                    f"{check.get('name')}"
                )
        for missing_name in sorted(REQUIRED_NEMOCLAW_OPERATOR_DOC_CHECK_NAMES - check_names):
            errors.append(
                "NeMoClaw operator docs verification missing required check: "
                f"{missing_name}"
            )
    if payload.get("ok") is True:
        if payload.get("status") != "passed":
            errors.append("NeMoClaw operator docs verification ok=true requires status=passed")
        if payload.get("missing_requirements") not in ([], None):
            errors.append(
                "NeMoClaw operator docs verification ok=true requires empty missing_requirements"
            )
    for field in ("readme_path", "lock_json"):
        if not isinstance(payload.get(field), str) or not payload.get(field):
            errors.append(f"NeMoClaw operator docs verification {field} must be a non-empty string")
    return errors


def validate_wandb_completion_query_source(
    payload: dict[str, Any],
    *,
    label: str,
    require_present: bool = False,
) -> list[str]:
    if "query_source" not in payload:
        if require_present:
            return [f"{label} query_source is not an object"]
        return []
    errors: list[str] = []
    query_source = payload.get("query_source")
    if not isinstance(query_source, dict):
        return [f"{label} query_source is not an object"]
    expected_values = {
        "kind": WANDB_COMPLETION_QUERY_SOURCE_KIND,
        "api": "wandb.Api",
        "entity": payload.get("entity"),
        "project": payload.get("project"),
        "run_id": payload.get("run_id"),
        "benchmark": payload.get("benchmark"),
        "summary_source": "run.summary_metrics",
        "artifact_source": "run.logged_artifacts",
        "history_scanned": False,
    }
    entity = payload.get("entity")
    project = payload.get("project")
    run_id = payload.get("run_id")
    if (
        isinstance(entity, str)
        and entity.strip()
        and isinstance(project, str)
        and project.strip()
        and isinstance(run_id, str)
        and run_id.strip()
    ):
        expected_values["run_path"] = f"{entity}/{project}/{run_id}"
    for field, expected in expected_values.items():
        if query_source.get(field) != expected:
            errors.append(
                f"{label} query_source.{field} mismatch: "
                f"expected {expected}, got {query_source.get(field)}"
            )
    if query_source.get("timeout_seconds") != WANDB_COMPLETION_API_TIMEOUT_SECONDS:
        errors.append(
            f"{label} query_source.timeout_seconds mismatch: "
            f"expected {WANDB_COMPLETION_API_TIMEOUT_SECONDS}, "
            f"got {query_source.get('timeout_seconds')}"
        )
    return errors


def validate_wandb_completion_nemoclaw_session_audit(
    *,
    payload: dict[str, Any],
    required: dict[str, Any],
    observed: dict[str, Any],
    label: str,
) -> list[str]:
    benchmark = payload.get("benchmark")
    if benchmark not in NEMOCLAW_AUDIT_REQUIRED_BENCHMARKS:
        return []
    errors: list[str] = []
    required_audit = required.get("nemoclaw_session_audit")
    if not isinstance(required_audit, dict):
        errors.append(f"{label} required_evidence.nemoclaw_session_audit is not an object")
    elif required_audit.get("required") is not True:
        errors.append(f"{label} required_evidence.nemoclaw_session_audit.required is not true")
    expected_total = _integer_value(required.get("expected_total"))
    if expected_total is None or expected_total <= 0:
        errors.append(f"{label} required_evidence.expected_total must be a positive integer")

    observed_audit = observed.get("nemoclaw_session_audit")
    if not isinstance(observed_audit, dict):
        errors.append(f"{label} observed_evidence.nemoclaw_session_audit is not an object")
        return errors
    if observed_audit.get("ok") is not True:
        errors.append(f"{label} observed_evidence.nemoclaw_session_audit.ok is not true")
    required_count = _integer_value(observed_audit.get("required"))
    passed_count = _integer_value(observed_audit.get("passed"))
    failed_count = _integer_value(observed_audit.get("failed"))
    for field, value in (
        ("required", required_count),
        ("passed", passed_count),
        ("failed", failed_count),
    ):
        if value is None:
            errors.append(f"{label} observed_evidence.nemoclaw_session_audit.{field} is not an integer")
    observed_expected_total = _integer_value(observed_audit.get("expected_total"))
    if expected_total is not None and observed_expected_total is not None and observed_expected_total != expected_total:
        errors.append(
            f"{label} observed_evidence.nemoclaw_session_audit.expected_total "
            "does not match required_evidence.expected_total"
        )
    if expected_total is not None and required_count is not None and required_count != expected_total:
        errors.append(f"{label} observed_evidence.nemoclaw_session_audit.required does not equal expected_total")
    if expected_total is not None and passed_count is not None and passed_count != expected_total:
        errors.append(f"{label} observed_evidence.nemoclaw_session_audit.passed does not equal expected_total")
    if required_count is not None and passed_count is not None and required_count != passed_count:
        errors.append(f"{label} observed_evidence.nemoclaw_session_audit.required does not equal passed")
    if failed_count is not None and failed_count != 0:
        errors.append(f"{label} observed_evidence.nemoclaw_session_audit.failed is not 0")
    return errors


def _positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _integer_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _finite_float_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _metric_row_value(row: Any) -> Any:
    if not isinstance(row, dict):
        return None
    return row.get("value")


def _metric_prefix(metric_name: str) -> str:
    if "/" not in metric_name:
        return ""
    return metric_name.rsplit("/", 1)[0]


def _observed_total_metric_value(required: dict[str, Any], observed: dict[str, Any]) -> int | None:
    required_metrics = required.get("summary_metrics")
    observed_metrics = observed.get("summary_metrics")
    if not isinstance(required_metrics, list) or not isinstance(observed_metrics, dict):
        return None
    for metric in required_metrics:
        if not isinstance(metric, str) or not metric.endswith("/total_instances"):
            continue
        value = _integer_value(_metric_row_value(observed_metrics.get(metric)))
        if value is not None:
            return value
    return None


def _observed_rows_by_name(rows: Any, *, key: str = "name") -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get(key)
        if isinstance(name, str) and name:
            result[name] = row
    return result


def required_wandb_completion_check_names(required: dict[str, Any]) -> set[str]:
    names = {"run_state"}
    required_metrics = required.get("summary_metrics")
    if isinstance(required_metrics, list):
        metric_suffixes = {
            metric.rsplit("/", 1)[1]
            for metric in required_metrics
            if isinstance(metric, str) and "/" in metric
        }
        if "total_instances" in metric_suffixes:
            names.add("total_metric")
        if "answered_instances" in metric_suffixes:
            names.add("answered_metric")
        if metric_suffixes.intersection({"correct_instances", "resolved_instances"}):
            names.add("correct_metric")
        if metric_suffixes.intersection({"accuracy", "pass_at_1"}):
            names.add("accuracy_metric")

    tables = required.get("tables")
    if isinstance(tables, list) and tables:
        names.update({"leaderboard_table", "output_table"})
    taxonomy_tables = required.get("taxonomy_tables")
    if isinstance(taxonomy_tables, list) and taxonomy_tables:
        names.add("taxonomy_table")
    aggregate_tables = required.get("aggregate_tables")
    if isinstance(aggregate_tables, list) and aggregate_tables:
        names.add("aggregate_table")
    artifacts = required.get("artifacts")
    if isinstance(artifacts, list) and artifacts:
        names.add("result_artifact")
    run_metadata = required.get("run_metadata")
    if isinstance(run_metadata, dict):
        if isinstance(run_metadata.get("config"), list) and run_metadata.get("config"):
            names.add("run_config")
        if isinstance(run_metadata.get("tags"), list) and run_metadata.get("tags"):
            names.add("run_tag")
        if isinstance(run_metadata.get("group"), str) and run_metadata.get("group"):
            names.add("run_group")
        if isinstance(run_metadata.get("job_type"), str) and run_metadata.get("job_type"):
            names.add("run_job_type")
    audit = required.get("nemoclaw_session_audit")
    if isinstance(audit, dict):
        names.add("nemoclaw_session_audit")
    return names


def validate_wandb_completion_checks(
    *,
    payload: dict[str, Any],
    required: dict[str, Any],
    observed: dict[str, Any],
    label: str,
) -> list[str]:
    errors: list[str] = []
    checks = payload.get("checks")
    if not isinstance(checks, list) or not checks:
        return [f"{label} checks is not a non-empty list"]

    allowed_names = required_wandb_completion_check_names(required)
    skipped_pending_units = required.get("skipped_pending_units")
    if isinstance(skipped_pending_units, list):
        allowed_names.add("taxonomy_unit_pending_skipped")
    singleton_names = {
        "run_state",
        "total_metric",
        "answered_metric",
        "correct_metric",
        "accuracy_metric",
        "leaderboard_table",
        "output_table",
        "result_artifact",
        "nemoclaw_session_audit",
        "run_group",
        "run_job_type",
    }
    name_counts: dict[str, int] = {}
    names: set[str] = set()
    for index, check in enumerate(checks, start=1):
        if not isinstance(check, dict):
            errors.append(f"{label} checks[{index}] is not an object")
            continue
        name = check.get("name")
        if not isinstance(name, str) or not name:
            errors.append(f"{label} checks[{index}].name is missing")
        else:
            names.add(name)
            name_counts[name] = name_counts.get(name, 0) + 1
            if name not in allowed_names:
                errors.append(f"{label} checks contains unknown check: {name}")
        if check.get("ok") is not True:
            errors.append(f"{label} checks contains failing check: {name or index}")

    for name in sorted(singleton_names.intersection(name_counts)):
        if name_counts[name] > 1:
            errors.append(f"{label} checks contains duplicate singleton check: {name}")
    for required_name in sorted(required_wandb_completion_check_names(required) - names):
        errors.append(f"{label} checks missing required check: {required_name}")
    errors.extend(
        validate_wandb_completion_checks_against_observed(
            checks=checks,
            required=required,
            observed=observed,
            label=label,
        )
    )
    return errors


def _checks_by_name(checks: list[Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for check in checks:
        if not isinstance(check, dict):
            continue
        name = check.get("name")
        if isinstance(name, str) and name:
            result.setdefault(name, []).append(check)
    return result


def _check_value_matches(left: Any, right: Any) -> bool:
    left_number = _finite_float_value(left)
    right_number = _finite_float_value(right)
    if left_number is not None and right_number is not None:
        return abs(left_number - right_number) <= 1e-12
    return left == right


def _first_check(checks_by_name: dict[str, list[dict[str, Any]]], name: str) -> dict[str, Any] | None:
    rows = checks_by_name.get(name)
    if not rows:
        return None
    return rows[0]


def _required_metric_check_name(metric: str) -> str | None:
    if "/" not in metric:
        return None
    suffix = metric.rsplit("/", 1)[1]
    if suffix == "total_instances":
        return "total_metric"
    if suffix == "answered_instances":
        return "answered_metric"
    if suffix in {"correct_instances", "resolved_instances"}:
        return "correct_metric"
    if suffix in {"accuracy", "pass_at_1"}:
        return "accuracy_metric"
    return None


def validate_wandb_completion_metric_checks_against_observed(
    *,
    checks_by_name: dict[str, list[dict[str, Any]]],
    required: dict[str, Any],
    observed: dict[str, Any],
    label: str,
) -> list[str]:
    errors: list[str] = []
    required_metrics = required.get("summary_metrics")
    observed_metrics = observed.get("summary_metrics")
    if not isinstance(required_metrics, list) or not isinstance(observed_metrics, dict):
        return errors
    for metric in required_metrics:
        if not isinstance(metric, str):
            continue
        check_name = _required_metric_check_name(metric)
        if not check_name:
            continue
        check = _first_check(checks_by_name, check_name)
        observed_row = observed_metrics.get(metric)
        if not isinstance(check, dict) or not isinstance(observed_row, dict):
            continue
        observed_value = observed_row.get("value")
        if "value" not in check:
            errors.append(f"{label} checks {check_name} missing value for {metric}")
            continue
        if not _check_value_matches(check.get("value"), observed_value):
            errors.append(
                f"{label} checks {check_name} value does not match "
                f"observed_evidence.summary_metrics {metric}: "
                f"expected {observed_value}, got {check.get('value')}"
            )
    return errors


def _validate_named_table_check(
    *,
    check: dict[str, Any] | None,
    observed_row: dict[str, Any] | None,
    check_name: str,
    table_name: str,
    label: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(check, dict) or not isinstance(observed_row, dict):
        return errors
    if "nrows" not in check:
        errors.append(f"{label} checks {check_name} missing nrows for {table_name}")
    elif check.get("nrows") != observed_row.get("nrows"):
        errors.append(
            f"{label} checks {check_name} nrows does not match "
            f"observed_evidence table {table_name}: "
            f"expected {observed_row.get('nrows')}, got {check.get('nrows')}"
        )
    return errors


def validate_wandb_completion_table_checks_against_observed(
    *,
    checks_by_name: dict[str, list[dict[str, Any]]],
    required: dict[str, Any],
    observed: dict[str, Any],
    label: str,
) -> list[str]:
    errors: list[str] = []
    observed_tables = _observed_rows_by_name(observed.get("tables"))
    required_tables = required.get("tables")
    if isinstance(required_tables, list):
        table_specs = [
            ("leaderboard_table", required_tables[0] if len(required_tables) > 0 else None),
            ("output_table", required_tables[1] if len(required_tables) > 1 else None),
        ]
        for check_name, table_spec in table_specs:
            if not isinstance(table_spec, dict):
                continue
            table_name = table_spec.get("name") or table_spec.get("table_name")
            if not isinstance(table_name, str) or not table_name:
                continue
            errors.extend(
                _validate_named_table_check(
                    check=_first_check(checks_by_name, check_name),
                    observed_row=observed_tables.get(table_name),
                    check_name=check_name,
                    table_name=table_name,
                    label=label,
                )
            )

    for check_name, required_field, observed_field, observed_key in (
        ("taxonomy_table", "taxonomy_tables", "taxonomy_tables", "table_name"),
        ("aggregate_table", "aggregate_tables", "aggregate_tables", "name"),
    ):
        required_rows = required.get(required_field)
        if not isinstance(required_rows, list) or not required_rows:
            continue
        observed_rows = _observed_rows_by_name(observed.get(observed_field), key=observed_key)
        checks = checks_by_name.get(check_name) or []
        checks_by_table = {
            check.get("table_name"): check
            for check in checks
            if isinstance(check, dict) and isinstance(check.get("table_name"), str)
        }
        for table_spec in required_rows:
            if not isinstance(table_spec, dict):
                continue
            table_name = table_spec.get("name") or table_spec.get("table_name")
            if not isinstance(table_name, str) or not table_name:
                continue
            errors.extend(
                _validate_named_table_check(
                    check=checks_by_table.get(table_name),
                    observed_row=observed_rows.get(table_name),
                    check_name=check_name,
                    table_name=table_name,
                    label=label,
                )
            )
    return errors


def validate_wandb_completion_artifact_checks_against_observed(
    *,
    checks_by_name: dict[str, list[dict[str, Any]]],
    required: dict[str, Any],
    label: str,
) -> list[str]:
    errors: list[str] = []
    required_artifacts = required.get("artifacts")
    if not isinstance(required_artifacts, list) or not required_artifacts:
        return errors
    check = _first_check(checks_by_name, "result_artifact")
    if not isinstance(check, dict):
        return errors
    check_artifacts = check.get("artifacts")
    if not isinstance(check_artifacts, list) or not check_artifacts:
        errors.append(f"{label} checks result_artifact artifacts is not a non-empty list")
        return errors
    for artifact in required_artifacts:
        if not isinstance(artifact, dict):
            continue
        artifact_type = artifact.get("type")
        if not isinstance(artifact_type, str) or not artifact_type:
            continue
        required_aliases = artifact.get("required_aliases")
        if not isinstance(required_aliases, list):
            required_aliases = []
        matching = [
            row
            for row in check_artifacts
            if isinstance(row, dict) and row.get("type") == artifact_type
        ]
        if not matching:
            errors.append(f"{label} checks result_artifact missing artifact type {artifact_type}")
            continue
        for alias in required_aliases:
            if not isinstance(alias, str) or not alias:
                continue
            if not any(
                isinstance(row.get("aliases"), list) and alias in row.get("aliases")
                for row in matching
            ):
                errors.append(
                    f"{label} checks result_artifact type {artifact_type} "
                    f"missing required alias {alias}"
                )
    return errors


def _observed_run_config_by_key(observed_metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = observed_metadata.get("config")
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("key")): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("key"), str) and row.get("key")
    }


def validate_wandb_completion_metadata_checks_against_observed(
    *,
    checks_by_name: dict[str, list[dict[str, Any]]],
    required: dict[str, Any],
    observed: dict[str, Any],
    label: str,
) -> list[str]:
    errors: list[str] = []
    required_metadata = required.get("run_metadata")
    if not isinstance(required_metadata, dict) or not required_metadata:
        return errors
    observed_metadata = observed.get("run_metadata")
    if not isinstance(observed_metadata, dict):
        return [f"{label} observed_evidence.run_metadata is not an object"]

    observed_config = _observed_run_config_by_key(observed_metadata)
    checks_by_config_key = {
        str(check.get("key")): check
        for check in checks_by_name.get("run_config", [])
        if isinstance(check, dict) and isinstance(check.get("key"), str) and check.get("key")
    }
    for row in required_metadata.get("config") or []:
        if not isinstance(row, dict):
            continue
        key = row.get("key")
        if not isinstance(key, str) or not key:
            continue
        expected = row.get("expected")
        observed_row = observed_config.get(key)
        if not isinstance(observed_row, dict):
            errors.append(f"{label} observed_evidence.run_metadata.config missing {key}")
            continue
        if observed_row.get("present") is not True:
            errors.append(f"{label} observed_evidence.run_metadata.config {key} is not present")
        if not _check_value_matches(observed_row.get("value"), expected):
            errors.append(f"{label} observed_evidence.run_metadata.config {key} value mismatch")
        check = checks_by_config_key.get(key)
        if not isinstance(check, dict):
            errors.append(f"{label} checks run_config missing key {key}")
            continue
        if check.get("present") is not True:
            errors.append(f"{label} checks run_config {key} is not present")
        if not _check_value_matches(check.get("value"), observed_row.get("value")):
            errors.append(f"{label} checks run_config {key} value does not match observed metadata")
        if not _check_value_matches(check.get("expected"), expected):
            errors.append(f"{label} checks run_config {key} expected value does not match required metadata")

    observed_tags = observed_metadata.get("tags")
    if not isinstance(observed_tags, list):
        observed_tags = []
    tag_checks = {
        check.get("tag")
        for check in checks_by_name.get("run_tag", [])
        if isinstance(check, dict)
    }
    for tag in required_metadata.get("tags") or []:
        if not isinstance(tag, str) or not tag:
            continue
        if tag not in observed_tags:
            errors.append(f"{label} observed_evidence.run_metadata.tags missing {tag}")
        if tag not in tag_checks:
            errors.append(f"{label} checks run_tag missing {tag}")

    for field, check_name in (("group", "run_group"), ("job_type", "run_job_type")):
        expected = required_metadata.get(field)
        if not isinstance(expected, str) or not expected:
            continue
        actual = observed_metadata.get(field)
        if actual != expected:
            errors.append(f"{label} observed_evidence.run_metadata.{field} mismatch")
        check = _first_check(checks_by_name, check_name)
        if not isinstance(check, dict):
            errors.append(f"{label} checks {check_name} missing")
        elif check.get("value") != actual:
            errors.append(f"{label} checks {check_name} value does not match observed metadata")
    return errors


def validate_wandb_completion_nemoclaw_check_against_observed(
    *,
    checks_by_name: dict[str, list[dict[str, Any]]],
    observed: dict[str, Any],
    label: str,
) -> list[str]:
    errors: list[str] = []
    observed_audit = observed.get("nemoclaw_session_audit")
    if not isinstance(observed_audit, dict):
        return errors
    check = _first_check(checks_by_name, "nemoclaw_session_audit")
    if not isinstance(check, dict):
        return errors
    for field in ("required", "passed", "failed", "expected_total"):
        if _integer_value(check.get(field)) != _integer_value(observed_audit.get(field)):
            errors.append(
                f"{label} checks nemoclaw_session_audit {field} does not match "
                f"observed_evidence.nemoclaw_session_audit"
            )
    return errors


def validate_wandb_completion_checks_against_observed(
    *,
    checks: list[Any],
    required: dict[str, Any],
    observed: dict[str, Any],
    label: str,
) -> list[str]:
    errors: list[str] = []
    checks_by_name = _checks_by_name(checks)
    run_state_check = _first_check(checks_by_name, "run_state")
    if isinstance(run_state_check, dict) and run_state_check.get("state") != observed.get("run_state"):
        errors.append(
            f"{label} checks run_state state does not match "
            f"observed_evidence.run_state: expected {observed.get('run_state')}, "
            f"got {run_state_check.get('state')}"
        )
    errors.extend(
        validate_wandb_completion_metric_checks_against_observed(
            checks_by_name=checks_by_name,
            required=required,
            observed=observed,
            label=label,
        )
    )
    errors.extend(
        validate_wandb_completion_table_checks_against_observed(
            checks_by_name=checks_by_name,
            required=required,
            observed=observed,
            label=label,
        )
    )
    errors.extend(
        validate_wandb_completion_artifact_checks_against_observed(
            checks_by_name=checks_by_name,
            required=required,
            label=label,
        )
    )
    errors.extend(
        validate_wandb_completion_metadata_checks_against_observed(
            checks_by_name=checks_by_name,
            required=required,
            observed=observed,
            label=label,
        )
    )
    errors.extend(
        validate_wandb_completion_nemoclaw_check_against_observed(
            checks_by_name=checks_by_name,
            observed=observed,
            label=label,
        )
    )
    return errors


def validate_wandb_observed_metrics(
    *,
    required: dict[str, Any],
    observed: dict[str, Any],
    label: str,
) -> list[str]:
    errors: list[str] = []
    required_metrics = required.get("summary_metrics")
    if not isinstance(required_metrics, list) or not required_metrics:
        return errors
    observed_metrics = observed.get("summary_metrics")
    if not isinstance(observed_metrics, dict):
        return [f"{label} observed_evidence.summary_metrics is not an object"]
    for metric in required_metrics:
        if not isinstance(metric, str) or not metric:
            errors.append(f"{label} required_evidence.summary_metrics contains an invalid metric name")
            continue
        row = observed_metrics.get(metric)
        if not isinstance(row, dict):
            errors.append(f"{label} observed_evidence.summary_metrics missing {metric}")
            continue
        if row.get("ok") is not True:
            errors.append(f"{label} observed_evidence.summary_metrics {metric} is not ok")
        if row.get("value") is None:
            errors.append(f"{label} observed_evidence.summary_metrics {metric} missing value")
    return errors


def validate_wandb_observed_metric_consistency(
    *,
    required: dict[str, Any],
    observed: dict[str, Any],
    label: str,
) -> list[str]:
    errors: list[str] = []
    required_metrics = required.get("summary_metrics")
    observed_metrics = observed.get("summary_metrics")
    if not isinstance(required_metrics, list) or not isinstance(observed_metrics, dict):
        return errors

    expected_total = _integer_value(required.get("expected_total"))
    observed_expected_total = observed.get("expected_total")
    if observed_expected_total is not None and expected_total is not None:
        observed_expected_total_int = _integer_value(observed_expected_total)
        if observed_expected_total_int != expected_total:
            errors.append(
                f"{label} observed_evidence.expected_total must equal "
                f"required_evidence.expected_total {expected_total}, got {observed_expected_total}"
            )

    totals: dict[str, int] = {}
    counts: dict[str, dict[str, int]] = {}
    ratios: dict[str, dict[str, float]] = {}
    for metric in required_metrics:
        if not isinstance(metric, str) or "/" not in metric:
            continue
        row = observed_metrics.get(metric)
        if not isinstance(row, dict) or row.get("ok") is not True:
            continue
        prefix = _metric_prefix(metric)
        suffix = metric.rsplit("/", 1)[1]
        if suffix == "total_instances":
            value = _integer_value(row.get("value"))
            if value is None or value <= 0:
                errors.append(f"{label} observed_evidence.summary_metrics {metric} must be a positive integer")
                continue
            totals[prefix] = value
            if expected_total is not None and value != expected_total:
                errors.append(
                    f"{label} observed_evidence.summary_metrics {metric} "
                    f"must equal expected_total {expected_total}, got {value}"
                )
        elif suffix in {"answered_instances", "correct_instances", "resolved_instances"}:
            value = _integer_value(row.get("value"))
            if value is None or value < 0:
                errors.append(f"{label} observed_evidence.summary_metrics {metric} must be a non-negative integer")
                continue
            counts.setdefault(prefix, {})[suffix] = value
        elif suffix in {"accuracy", "pass_at_1"}:
            value = _finite_float_value(row.get("value"))
            if value is None:
                errors.append(f"{label} observed_evidence.summary_metrics {metric} must be numeric")
                continue
            if value < 0.0 or value > 1.0:
                errors.append(f"{label} observed_evidence.summary_metrics {metric} must be in [0, 1]")
            ratios.setdefault(prefix, {})[suffix] = value

    for prefix, count_rows in counts.items():
        total = totals.get(prefix)
        if total is None:
            continue
        for suffix, value in count_rows.items():
            if value > total:
                errors.append(
                    f"{label} observed_evidence.summary_metrics {prefix}/{suffix} "
                    f"must be <= {prefix}/total_instances {total}, got {value}"
                )

    for prefix, ratio_rows in ratios.items():
        total = totals.get(prefix)
        if total is None:
            continue
        numerator = counts.get(prefix, {}).get("correct_instances")
        if numerator is None:
            numerator = counts.get(prefix, {}).get("resolved_instances")
        if numerator is None:
            continue
        expected_ratio = numerator / total if total else 0.0
        for suffix, value in ratio_rows.items():
            if abs(value - expected_ratio) > 1e-12:
                errors.append(
                    f"{label} observed_evidence.summary_metrics {prefix}/{suffix} "
                    f"must equal numerator/total {expected_ratio}, got {value}"
                )
    return errors


def validate_wandb_observed_tables(
    *,
    required: dict[str, Any],
    observed: dict[str, Any],
    label: str,
    required_field: str,
    observed_field: str,
    observed_name_key: str = "name",
) -> list[str]:
    errors: list[str] = []
    required_tables = required.get(required_field)
    if not isinstance(required_tables, list) or not required_tables:
        return errors
    observed_tables = observed.get(observed_field)
    if not isinstance(observed_tables, list):
        return [f"{label} observed_evidence.{observed_field} is not a list"]
    rows_by_name = _observed_rows_by_name(observed_tables, key=observed_name_key)
    expected_total = _observed_total_metric_value(required, observed)
    if expected_total is None:
        expected_total = _integer_value(required.get("expected_total"))
    for table in required_tables:
        if not isinstance(table, dict):
            errors.append(f"{label} required_evidence.{required_field} contains a non-object row")
            continue
        table_name = table.get("name") or table.get("table_name")
        if not isinstance(table_name, str) or not table_name:
            errors.append(f"{label} required_evidence.{required_field} row missing table name")
            continue
        row = rows_by_name.get(table_name)
        if not isinstance(row, dict):
            errors.append(f"{label} observed_evidence.{observed_field} missing {table_name}")
            continue
        if row.get("ok") is not True:
            errors.append(f"{label} observed_evidence.{observed_field} {table_name} is not ok")
        nrows = row.get("nrows")
        if not _positive_int(nrows):
            errors.append(f"{label} observed_evidence.{observed_field} {table_name} has invalid nrows")
            continue
        if (
            table.get("row_count") == "must equal total metric"
            and expected_total is not None
            and expected_total > 0
            and nrows != expected_total
        ):
            errors.append(
                f"{label} observed_evidence.{observed_field} {table_name} "
                f"nrows must equal expected_total {expected_total}, got {nrows}"
            )
    return errors


def validate_wandb_observed_artifacts(
    *,
    required: dict[str, Any],
    observed: dict[str, Any],
    label: str,
) -> list[str]:
    errors: list[str] = []
    required_artifacts = required.get("artifacts")
    if not isinstance(required_artifacts, list) or not required_artifacts:
        return errors
    observed_artifacts = observed.get("artifacts")
    if not isinstance(observed_artifacts, list):
        return [f"{label} observed_evidence.artifacts is not a list"]
    for artifact in required_artifacts:
        if not isinstance(artifact, dict):
            errors.append(f"{label} required_evidence.artifacts contains a non-object row")
            continue
        artifact_type = artifact.get("type")
        if not isinstance(artifact_type, str) or not artifact_type:
            errors.append(f"{label} required_evidence.artifacts row missing type")
            continue
        required_aliases = artifact.get("required_aliases")
        if not isinstance(required_aliases, list):
            required_aliases = []
        matching = [
            row
            for row in observed_artifacts
            if isinstance(row, dict) and row.get("type") == artifact_type
        ]
        if not matching:
            errors.append(f"{label} observed_evidence.artifacts missing type {artifact_type}")
            continue
        for alias in required_aliases:
            if not isinstance(alias, str) or not alias:
                continue
            if not any(
                isinstance(row.get("aliases"), list) and alias in row.get("aliases")
                for row in matching
            ):
                errors.append(
                    f"{label} observed_evidence.artifacts type {artifact_type} "
                    f"missing required alias {alias}"
                )
    return errors


def validate_wandb_observed_evidence(
    *,
    required: dict[str, Any],
    observed: dict[str, Any],
    label: str,
) -> list[str]:
    errors: list[str] = []
    errors.extend(
        validate_wandb_observed_metrics(
            required=required,
            observed=observed,
            label=label,
        )
    )
    errors.extend(
        validate_wandb_observed_metric_consistency(
            required=required,
            observed=observed,
            label=label,
        )
    )
    errors.extend(
        validate_wandb_observed_tables(
            required=required,
            observed=observed,
            label=label,
            required_field="tables",
            observed_field="tables",
        )
    )
    errors.extend(
        validate_wandb_observed_tables(
            required=required,
            observed=observed,
            label=label,
            required_field="taxonomy_tables",
            observed_field="taxonomy_tables",
            observed_name_key="table_name",
        )
    )
    errors.extend(
        validate_wandb_observed_tables(
            required=required,
            observed=observed,
            label=label,
            required_field="aggregate_tables",
            observed_field="aggregate_tables",
        )
    )
    errors.extend(
        validate_wandb_observed_artifacts(
            required=required,
            observed=observed,
            label=label,
        )
    )
    return errors


def validate_wandb_completion_required_run_metadata(
    payload: dict[str, Any],
    *,
    label: str,
    sources: list[Any],
) -> list[str]:
    source_text = ", ".join(str(source) for source in sources if isinstance(source, str) and source)
    if not source_text:
        source_text = "release proof"
    errors: list[str] = []
    required = payload.get("required_evidence")
    if not isinstance(required, dict):
        return [
            f"{label} required_evidence.run_metadata is required for paid-review/release proof "
            f"from {source_text}"
        ]
    required_metadata = required.get("run_metadata")
    if not isinstance(required_metadata, dict) or not required_metadata:
        errors.append(
            f"{label} required_evidence.run_metadata is required for paid-review/release proof "
            f"from {source_text}"
        )
        required_metadata = {}
    else:
        config = required_metadata.get("config")
        tags = required_metadata.get("tags")
        has_scope_requirement = False
        if isinstance(config, list):
            has_scope_requirement = any(
                isinstance(row, dict)
                and isinstance(row.get("key"), str)
                and row.get("key")
                and "expected" in row
                for row in config
            )
        if isinstance(tags, list) and any(isinstance(tag, str) and tag for tag in tags):
            has_scope_requirement = True
        if isinstance(required_metadata.get("group"), str) and required_metadata.get("group"):
            has_scope_requirement = True
        if isinstance(required_metadata.get("job_type"), str) and required_metadata.get("job_type"):
            has_scope_requirement = True
        if not has_scope_requirement:
            errors.append(
                f"{label} required_evidence.run_metadata must specify at least one config, tag, "
                f"group, or job_type requirement for paid-review/release proof from {source_text}"
            )

    observed = payload.get("observed_evidence")
    observed_metadata = observed.get("run_metadata") if isinstance(observed, dict) else None
    if not isinstance(observed_metadata, dict) or not observed_metadata:
        errors.append(
            f"{label} observed_evidence.run_metadata is required for paid-review/release proof "
            f"from {source_text}"
        )
    return errors


def validate_wandb_completion_proof_evidence(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    proofs = collect_wandb_completion_proofs(manifest)
    if not proofs:
        return errors
    records = file_records_by_source(manifest)
    for source_path, proof in sorted(proofs.items()):
        expected_benchmark = proof.get("benchmark") if isinstance(proof.get("benchmark"), str) else None
        record = records.get(source_path)
        if not isinstance(record, dict):
            errors.append(f"missing bundled W&B completion proof JSON: {source_path}")
            continue
        roles = record.get("roles")
        if not isinstance(roles, list) or not any("wandb_completion" in str(role) for role in roles):
            errors.append(f"bundled W&B completion proof JSON has no wandb_completion role: {source_path}")
        bundle_path = record.get("bundle_path")
        if not isinstance(bundle_path, str) or not bundle_path:
            errors.append(f"bundled W&B completion proof JSON missing bundle_path: {source_path}")
            continue
        try:
            payload = read_json_object(bundle_dir / bundle_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"W&B completion proof JSON is not readable: {source_path}: {exc}")
            continue
        errors.extend(
            validate_wandb_completion_payload(
                payload,
                label=f"W&B completion proof {source_path}",
                expected_benchmark=expected_benchmark,
            )
        )
        if proof.get("require_run_metadata") is True:
            errors.extend(
                validate_wandb_completion_required_run_metadata(
                    payload,
                    label=f"W&B completion proof {source_path}",
                    sources=proof.get("sources") if isinstance(proof.get("sources"), list) else [],
                )
            )
    return errors


def validate_scope_attestation_payload(
    payload: dict[str, Any],
    *,
    label: str,
    entry: dict[str, Any],
    review_path: Any,
) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != SCOPE_ATTESTATION_SCHEMA_VERSION:
        errors.append(
            f"{label} schema_version must be {SCOPE_ATTESTATION_SCHEMA_VERSION}"
        )
    if payload.get("confirmed") is not True:
        errors.append(f"{label} confirmed is not true")
    for field in (
        "confirmed_by",
        "confirmed_at",
        "confirmation",
        "actual_cost_estimate",
        "provider_bill_reference",
    ):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{label} {field} is missing")
    for field in ("actual_cost_estimate", "provider_bill_reference"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip() and accounting_value_placeholder(value):
            errors.append(f"{label} {field} must not be a placeholder")
    if not scope_confirmed_at_valid(payload.get("confirmed_at")):
        errors.append(f"{label} confirmed_at must be a timezone-aware ISO 8601 timestamp")
    if not scope_confirmation_valid(payload.get("confirmation")):
        errors.append(
            f"{label} confirmation must be a concrete non-placeholder sentence "
            f"with at least {MIN_SCOPE_CONFIRMATION_LENGTH} characters"
        )
    if payload.get("benchmark") != entry.get("benchmark"):
        errors.append(f"{label} benchmark does not match paid review entry")
    if payload.get("entity") != entry.get("entity"):
        errors.append(f"{label} entity does not match paid review entry")
    if payload.get("project") != entry.get("project"):
        errors.append(f"{label} project does not match paid review entry")
    if payload.get("run_id") != entry.get("run_id"):
        errors.append(f"{label} run_id does not match paid review entry")
    if source_path_key(payload.get("completion_path")) != source_path_key(entry.get("path")):
        errors.append(f"{label} completion_path does not match paid review entry")
    if not isinstance(payload.get("completion_sha256"), str) or not payload.get("completion_sha256"):
        errors.append(f"{label} completion_sha256 is missing")
    elif payload.get("completion_sha256") != entry.get("sha256"):
        errors.append(f"{label} completion_sha256 does not match paid review entry")
    if source_path_key(payload.get("review_path")) != source_path_key(review_path):
        errors.append(f"{label} review_path does not match paid review record")
    source_audit_json = payload.get("source_audit_json")
    source_audit_sha256 = payload.get("source_audit_sha256")
    if source_audit_json is not None or source_audit_sha256 is not None:
        if not isinstance(source_audit_json, str) or not source_audit_json.strip():
            errors.append(f"{label} source_audit_json is missing")
        if not isinstance(source_audit_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}",
            source_audit_sha256 or "",
        ):
            errors.append(
                f"{label} source_audit_sha256 must be a 64-character lowercase hex digest"
            )

    scope = entry.get("scope_attestation")
    if isinstance(scope, dict):
        for field in ("actual_cost_estimate", "provider_bill_reference"):
            value = scope.get(field)
            if isinstance(value, str) and value.strip() and accounting_value_placeholder(value):
                errors.append(f"{label} embedded scope_attestation {field} must not be a placeholder")
        if scope.get("actual_cost_estimate") and payload.get("actual_cost_estimate") != scope.get("actual_cost_estimate"):
            errors.append(f"{label} actual_cost_estimate does not match embedded scope_attestation")
        if scope.get("provider_bill_reference") and payload.get("provider_bill_reference") != scope.get("provider_bill_reference"):
            errors.append(f"{label} provider_bill_reference does not match embedded scope_attestation")
        if scope.get("completion_sha256") and payload.get("completion_sha256") != scope.get("completion_sha256"):
            errors.append(f"{label} completion_sha256 does not match embedded scope_attestation")
        if scope.get("source_audit_sha256") and payload.get("source_audit_sha256") != scope.get("source_audit_sha256"):
            errors.append(f"{label} source_audit_sha256 does not match embedded scope_attestation")
        if scope.get("source_audit_json") and source_path_key(payload.get("source_audit_json")) != source_path_key(scope.get("source_audit_json")):
            errors.append(f"{label} source_audit_json does not match embedded scope_attestation")
    return errors


def validate_paid_review_wandb_completion_entry_payload(
    *,
    bundle_dir: Path,
    records: dict[str, dict[str, Any]],
    entry: dict[str, Any],
    label: str,
) -> list[str]:
    errors: list[str] = []
    key = source_path_key(entry.get("path"))
    if not key:
        errors.append(f"{label} path is missing")
        return errors
    record = records.get(key)
    if not isinstance(record, dict):
        errors.append(f"missing bundled W&B completion JSON for {label}: {key}")
        return errors
    roles = record.get("roles")
    if not isinstance(roles, list) or not any("wandb_completion" in str(role) for role in roles):
        errors.append(f"bundled W&B completion JSON for {label} has no wandb_completion role: {key}")

    bundle_path = record.get("bundle_path")
    if not isinstance(bundle_path, str) or not bundle_path:
        errors.append(f"bundled W&B completion JSON for {label} missing bundle_path: {key}")
        return errors
    try:
        payload = read_json_object(bundle_dir / bundle_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"bundled W&B completion JSON for {label} is not readable: {exc}")
        return errors

    if entry.get("adopted_existing_result") is not True:
        query_source_errors = validate_wandb_completion_query_source(
            payload,
            label=label,
            require_present=True,
        )
        errors.extend(query_source_errors)
    for field in ("benchmark", "entity", "project", "run_id"):
        if payload.get(field) != entry.get(field):
            errors.append(f"{label} {field} does not match W&B completion payload")
    parent_field_map = {
        "parent_run_id": "run_id",
        "parent_entity": "entity",
        "parent_project": "project",
    }
    for parent_field, payload_field in parent_field_map.items():
        parent_value = entry.get(parent_field)
        if isinstance(parent_value, str) and parent_value:
            if payload.get(payload_field) != parent_value:
                errors.append(
                    f"{label} {payload_field} does not match parent review run"
                )
        match_field = f"{parent_field}_matches"
        if match_field in entry and entry.get(match_field) is not True:
            errors.append(f"{label} {match_field} is not true")
    expected_sha256 = record.get("sha256")
    entry_sha256 = entry.get("sha256")
    if not isinstance(entry_sha256, str) or not entry_sha256:
        errors.append(f"{label} sha256 is missing")
    elif entry_sha256 != expected_sha256:
        errors.append(f"{label} sha256 does not match bundled W&B completion JSON")
    return errors


def collect_review_wandb_completion_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    def append_rows(
        rows: Any,
        *,
        parent_run_id: str = "",
        parent_entity: str = "",
        parent_project: str = "",
    ) -> None:
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            benchmark = row.get("benchmark")
            if not isinstance(benchmark, str) or not benchmark:
                continue
            run_id = row.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                run_id = parent_run_id
            entries.append(
                {
                    "benchmark": benchmark,
                    "entity": row.get("entity") if isinstance(row.get("entity"), str) else "",
                    "project": row.get("project") if isinstance(row.get("project"), str) else "",
                    "run_id": run_id,
                    "path": row.get("path") if isinstance(row.get("path"), str) else "",
                    "sha256": row.get("sha256") if isinstance(row.get("sha256"), str) else "",
                    "parent_run_id": parent_run_id,
                    "parent_entity": parent_entity,
                    "parent_project": parent_project,
                    "adopted_existing_result": row.get("adopted_existing_result")
                    is True,
                    "scope_attestation": row.get("scope_attestation")
                    if isinstance(row.get("scope_attestation"), dict)
                    else None,
                }
            )

    append_rows(payload.get("wandb_completion"))
    runs = payload.get("runs")
    if isinstance(runs, list):
        for run in runs:
            if not isinstance(run, dict):
                continue
            parent_run_id = run.get("wandb_run_id")
            parent_entity = run.get("wandb_entity")
            parent_project = run.get("wandb_project")
            append_rows(
                run.get("wandb_completion"),
                parent_run_id=parent_run_id if isinstance(parent_run_id, str) else "",
                parent_entity=parent_entity if isinstance(parent_entity, str) else "",
                parent_project=parent_project if isinstance(parent_project, str) else "",
            )
    return entries


def review_wandb_completion_entries_match(
    *,
    entry: dict[str, Any],
    raw_entry: dict[str, Any],
) -> bool:
    for field in ("benchmark", "entity", "project", "run_id", "sha256"):
        if raw_entry.get(field) != entry.get(field):
            return False
    if source_path_key(raw_entry.get("path")) != source_path_key(entry.get("path")):
        return False
    for field in ("parent_run_id", "parent_entity", "parent_project"):
        value = entry.get(field)
        if isinstance(value, str) and value and raw_entry.get(field) != value:
            return False
    if raw_entry.get("adopted_existing_result") != (
        entry.get("adopted_existing_result") is True
    ):
        return False
    return True


def validate_raw_review_scope_attestation_match(
    *,
    entry: dict[str, Any],
    raw_entry: dict[str, Any],
    label: str,
) -> list[str]:
    errors: list[str] = []
    if entry.get("adopted_existing_result") is not True:
        return errors
    expected = entry.get("scope_attestation")
    actual = raw_entry.get("scope_attestation")
    if not isinstance(expected, dict):
        errors.append(f"{label} current_gate scope_attestation is missing")
        return errors
    if not isinstance(actual, dict):
        errors.append(
            f"{label} adopted existing result is missing scope_attestation "
            "in bundled paid review JSON"
        )
        return errors

    path_fields = {
        "review_path",
        "completion_path",
        "source_attestation_json",
        "source_audit_json",
    }
    compared_fields = (
        "schema_version",
        "confirmed",
        "confirmed_by",
        "confirmed_at",
        "confirmation",
        "review_path",
        "completion_path",
        "completion_sha256",
        "benchmark",
        "entity",
        "project",
        "run_id",
        "actual_cost_estimate",
        "provider_bill_reference",
        "source_attestation_json",
        "source_attestation_sha256",
        "source_audit_json",
        "source_audit_sha256",
    )
    for field in compared_fields:
        expected_value = expected.get(field)
        actual_value = actual.get(field)
        if field in path_fields:
            if source_path_key(actual_value) != source_path_key(expected_value):
                errors.append(
                    f"{label} raw paid review scope_attestation {field} "
                    "does not match current_gate"
                )
        elif actual_value != expected_value:
            errors.append(
                f"{label} raw paid review scope_attestation {field} "
                "does not match current_gate"
            )
    return errors


def validate_paid_review_wandb_completion_entry_review_source(
    *,
    bundle_dir: Path,
    records: dict[str, dict[str, Any]],
    record: dict[str, Any],
    entry: dict[str, Any],
    label: str,
) -> list[str]:
    errors: list[str] = []
    review_key = source_path_key(record.get("path"))
    if not review_key:
        errors.append(f"{label} paid review record path is missing")
        return errors
    review_record = records.get(review_key)
    if not isinstance(review_record, dict):
        errors.append(f"missing bundled paid review JSON for {label}: {review_key}")
        return errors
    bundle_path = review_record.get("bundle_path")
    if not isinstance(bundle_path, str) or not bundle_path:
        errors.append(f"bundled paid review JSON for {label} missing bundle_path: {review_key}")
        return errors
    try:
        review_payload = read_json_object(bundle_dir / bundle_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"bundled paid review JSON for {label} is not readable: {exc}")
        return errors

    for field in (
        "status",
        "phase",
        "canary",
        "model_count",
        "verify_wandb_completion",
        "verify_weave_agents",
    ):
        if field in review_payload and field in record and review_payload.get(field) != record.get(field):
            errors.append(f"{label} paid review {field} does not match bundled review JSON")

    raw_entries = collect_review_wandb_completion_entries(review_payload)
    matched_raw_entries = [
        raw_entry
        for raw_entry in raw_entries
        if review_wandb_completion_entries_match(entry=entry, raw_entry=raw_entry)
    ]
    if not matched_raw_entries:
        errors.append(f"{label} is not present in bundled paid review JSON")
    else:
        errors.extend(
            validate_raw_review_scope_attestation_match(
                entry=entry,
                raw_entry=matched_raw_entries[0],
                label=label,
            )
        )
    return errors


def validate_paid_review_wandb_completion_entry_evidence(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    current_gate = manifest.get("current_gate")
    if not isinstance(current_gate, dict):
        return errors
    paid_review_package = current_gate.get("paid_run_review_package")
    gates = paid_review_package.get("gates") if isinstance(paid_review_package, dict) else None
    if not isinstance(gates, list):
        return errors

    records = file_records_by_source(manifest)
    for gate in gates:
        if not isinstance(gate, dict):
            continue
        gate_name = str(gate.get("name") or "unknown_gate")
        review_records = gate.get("records")
        if not isinstance(review_records, list):
            continue
        for record_index, record in enumerate(review_records, start=1):
            if not isinstance(record, dict):
                continue
            entries = record.get("wandb_completion_entries")
            if not isinstance(entries, list):
                continue
            for entry_index, entry in enumerate(entries, start=1):
                if not isinstance(entry, dict):
                    continue
                benchmark = str(entry.get("benchmark") or "unknown_benchmark")
                label = (
                    f"paid review W&B completion entry {gate_name}#"
                    f"{record_index}.{entry_index} {benchmark}/{entry.get('run_id')}"
                )
                errors.extend(
                    validate_paid_review_wandb_completion_entry_payload(
                        bundle_dir=bundle_dir,
                        records=records,
                        entry=entry,
                        label=label,
                    )
                )
                errors.extend(
                    validate_paid_review_wandb_completion_entry_review_source(
                        bundle_dir=bundle_dir,
                        records=records,
                        record=record,
                        entry=entry,
                        label=label,
                    )
                )
    return errors


def nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def paid_review_record_summary_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    runs = payload.get("runs")
    wandb_entries = collect_review_wandb_completion_entries(payload)
    weave_entries = collect_review_weave_agents_completion_entries(payload)
    actual_cost = payload.get("actual_cost_estimate")
    provider_bill = payload.get("provider_bill_reference")
    return {
        "status": str(payload.get("status") or ""),
        "phase": payload.get("phase"),
        "canary": bool(payload.get("canary")),
        "model_count": payload.get("model_count"),
        "run_purpose_present": nonempty_string(payload.get("run_purpose")),
        "expected_cost_band_present": nonempty_string(payload.get("expected_cost_band")),
        "actual_cost_estimate_present": nonempty_string(actual_cost),
        "provider_bill_reference_present": nonempty_string(provider_bill),
        "actual_cost_estimate_placeholder": isinstance(actual_cost, str)
        and accounting_value_placeholder(actual_cost),
        "provider_bill_reference_placeholder": isinstance(provider_bill, str)
        and accounting_value_placeholder(provider_bill),
        "run_count": len(runs) if isinstance(runs, list) else 0,
        "verify_wandb_completion": bool(payload.get("verify_wandb_completion")),
        "wandb_completion_entry_count": len(wandb_entries),
        "verify_weave_agents": bool(payload.get("verify_weave_agents")),
        "weave_agents_completion_entry_count": len(weave_entries),
    }


def validate_paid_review_record_review_source_evidence(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    current_gate = manifest.get("current_gate")
    if not isinstance(current_gate, dict):
        return errors
    paid_review_package = current_gate.get("paid_run_review_package")
    gates = paid_review_package.get("gates") if isinstance(paid_review_package, dict) else None
    if not isinstance(gates, list):
        return errors

    records_by_source = file_records_by_source(manifest)
    compared_fields = (
        "status",
        "phase",
        "canary",
        "model_count",
        "run_purpose_present",
        "expected_cost_band_present",
        "actual_cost_estimate_present",
        "provider_bill_reference_present",
        "actual_cost_estimate_placeholder",
        "provider_bill_reference_placeholder",
        "run_count",
        "verify_wandb_completion",
        "wandb_completion_entry_count",
        "verify_weave_agents",
        "weave_agents_completion_entry_count",
    )
    for gate in gates:
        if not isinstance(gate, dict):
            continue
        gate_name = str(gate.get("name") or "unknown_gate")
        review_records = gate.get("records")
        if not isinstance(review_records, list):
            continue
        for record_index, record in enumerate(review_records, start=1):
            if not isinstance(record, dict):
                continue
            review_key = source_path_key(record.get("path"))
            label = f"paid review record {gate_name}#{record_index}"
            if not review_key:
                errors.append(f"{label} path is missing")
                continue
            review_record = records_by_source.get(review_key)
            if not isinstance(review_record, dict):
                errors.append(f"missing bundled paid review JSON for {label}: {review_key}")
                continue
            bundle_path = review_record.get("bundle_path")
            if not isinstance(bundle_path, str) or not bundle_path:
                errors.append(f"bundled paid review JSON for {label} missing bundle_path: {review_key}")
                continue
            try:
                review_payload = read_json_object(bundle_dir / bundle_path)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"bundled paid review JSON for {label} is not readable: {exc}")
                continue
            expected = paid_review_record_summary_from_payload(review_payload)
            for field in compared_fields:
                if field in record and record.get(field) != expected.get(field):
                    errors.append(f"{label} {field} does not match bundled review JSON")
    return errors


def validate_run_eval_preflight_payload(payload: dict[str, Any], *, label: str) -> list[str]:
    errors: list[str] = []
    if payload.get("ok") is not True:
        errors.append(f"{label} ok must be true")
    if payload.get("status") != "passed":
        errors.append(f"{label} status must be passed")
    for field in (
        "will_initialize_wandb",
        "will_log_wandb_artifacts",
        "will_initialize_weave",
        "will_start_inference_engine",
        "will_run_evaluators",
    ):
        if payload.get(field) is not False:
            errors.append(f"{label} {field} must be false")
    if not isinstance(payload.get("enabled_benchmarks"), list):
        errors.append(f"{label} enabled_benchmarks must be a list")
    return errors


def validate_paid_review_run_eval_preflight_evidence(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    current_gate = manifest.get("current_gate")
    if not isinstance(current_gate, dict):
        return errors
    paid_review_package = current_gate.get("paid_run_review_package")
    gates = paid_review_package.get("gates") if isinstance(paid_review_package, dict) else None
    if not isinstance(gates, list):
        return errors

    records_by_source = file_records_by_source(manifest)
    for gate in gates:
        if not isinstance(gate, dict):
            continue
        gate_name = str(gate.get("name") or "unknown_gate")
        review_records = gate.get("records")
        if not isinstance(review_records, list):
            continue
        for record_index, record in enumerate(review_records, start=1):
            if not isinstance(record, dict):
                continue
            if record.get("status") != "completed" and record.get("ok") is not True:
                continue
            review_key = source_path_key(record.get("path"))
            label = f"paid review run_eval preflight {gate_name}#{record_index}"
            review_record = records_by_source.get(review_key) if review_key else None
            if not isinstance(review_record, dict):
                errors.append(f"{label} missing bundled paid review JSON: {review_key}")
                continue
            review_bundle_path = review_record.get("bundle_path")
            if not isinstance(review_bundle_path, str) or not review_bundle_path:
                errors.append(f"{label} bundled paid review JSON missing bundle_path")
                continue
            try:
                review_payload = read_json_object(bundle_dir / review_bundle_path)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"{label} bundled paid review JSON is not readable: {exc}")
                continue
            if review_payload.get("status") != "completed":
                continue
            preflights = review_payload.get("run_eval_preflights")
            if not isinstance(preflights, list) or not preflights:
                errors.append(f"{label} bundled paid review JSON missing run_eval_preflights")
                continue
            top_outputs = {
                str(item.get("output_json"))
                for item in preflights
                if isinstance(item, dict) and nonempty_string(item.get("output_json"))
            }
            for preflight_index, preflight in enumerate(preflights, start=1):
                if not isinstance(preflight, dict):
                    errors.append(f"{label} run_eval_preflights {preflight_index} is not an object")
                    continue
                output_json = preflight.get("output_json")
                command = preflight.get("command")
                preflight_label = f"{label}.{preflight_index}"
                if preflight.get("required_before_run_eval") is not True:
                    errors.append(f"{preflight_label} required_before_run_eval must be true")
                if not isinstance(command, list) or not command:
                    errors.append(f"{preflight_label} command is missing")
                else:
                    command_parts = [str(part) for part in command]
                    if "scripts/run_eval.py" not in command_parts:
                        errors.append(f"{preflight_label} command must invoke scripts/run_eval.py")
                    if "--preflight" not in command_parts:
                        errors.append(f"{preflight_label} command missing --preflight")
                    if "--preflight-json" not in command_parts:
                        errors.append(f"{preflight_label} command missing --preflight-json")
                    elif nonempty_string(output_json):
                        try:
                            output_arg = command_parts[command_parts.index("--preflight-json") + 1]
                        except IndexError:
                            errors.append(f"{preflight_label} command --preflight-json missing value")
                        else:
                            if output_arg != output_json:
                                errors.append(
                                    f"{preflight_label} command --preflight-json does not match output_json"
                                )
                preflight_record = validate_any_file_role(
                    errors=errors,
                    records=records_by_source,
                    path_value=output_json,
                    role_suffix=":run_eval_preflight",
                    label=preflight_label,
                )
                if not isinstance(preflight_record, dict):
                    continue
                bundle_path = preflight_record.get("bundle_path")
                if not isinstance(bundle_path, str) or not bundle_path:
                    errors.append(f"{preflight_label} bundled preflight JSON missing bundle_path")
                    continue
                try:
                    payload = read_json_object(bundle_dir / bundle_path)
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    errors.append(f"{preflight_label} bundled preflight JSON is not readable: {exc}")
                    continue
                errors.extend(validate_run_eval_preflight_payload(payload, label=preflight_label))
            runs = review_payload.get("runs")
            if isinstance(runs, list):
                for run_index, run in enumerate(runs, start=1):
                    if not isinstance(run, dict):
                        continue
                    preflight_json = run.get("preflight_json")
                    run_label = f"{label}.run{run_index}"
                    if not nonempty_string(preflight_json):
                        errors.append(f"{run_label} missing preflight_json")
                        continue
                    if top_outputs and str(preflight_json) not in top_outputs:
                        errors.append(
                            f"{run_label} preflight_json does not match run_eval_preflights output_json"
                        )
                    if run.get("preflight_returncode") != 0:
                        errors.append(f"{run_label} preflight_returncode must be 0")
                    if run.get("preflight_ok") is not True:
                        errors.append(f"{run_label} preflight_ok must be true")
    return errors


def validate_paid_review_scope_attestation_evidence(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    current_gate = manifest.get("current_gate")
    if not isinstance(current_gate, dict):
        return errors
    paid_review_package = current_gate.get("paid_run_review_package")
    gates = paid_review_package.get("gates") if isinstance(paid_review_package, dict) else None
    if not isinstance(gates, list):
        return errors

    records = file_records_by_source(manifest)
    for gate in gates:
        if not isinstance(gate, dict):
            continue
        gate_name = str(gate.get("name") or "unknown_gate")
        review_records = gate.get("records")
        if not isinstance(review_records, list):
            continue
        for record in review_records:
            if not isinstance(record, dict):
                continue
            entries = record.get("wandb_completion_entries")
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict) or entry.get("adopted_existing_result") is not True:
                    continue
                benchmark = str(entry.get("benchmark") or "unknown_benchmark")
                scope = entry.get("scope_attestation")
                if not isinstance(scope, dict):
                    errors.append(
                        f"adopted existing W&B completion {benchmark}/{entry.get('run_id')} "
                        "missing scope_attestation"
                    )
                    continue
                source = scope.get("source_attestation_json")
                if not isinstance(source, str) or not source.strip():
                    errors.append(
                        f"adopted existing W&B completion {benchmark}/{entry.get('run_id')} "
                        "missing source_attestation_json"
                    )
                    continue
                role = (
                    f"paid_run_review_check:gate:{gate_name}:wandb_completion:"
                    f"{benchmark}:source_attestation_json"
                )
                source_record = validate_file_role(
                    errors=errors,
                    records=records,
                    path_value=source,
                    role=role,
                    label=f"paid review scope attestation source JSON for {benchmark}",
                )
                if not isinstance(source_record, dict):
                    continue
                bundle_path = source_record.get("bundle_path")
                if not isinstance(bundle_path, str) or not bundle_path:
                    errors.append(
                        f"paid review scope attestation source JSON for {benchmark} missing bundle_path"
                    )
                    continue
                try:
                    payload = read_json_object(bundle_dir / bundle_path)
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    errors.append(
                        f"paid review scope attestation source JSON for {benchmark} is not readable: {exc}"
                    )
                    continue
                errors.extend(
                    validate_scope_attestation_payload(
                        payload,
                        label=f"paid review scope attestation source JSON for {benchmark}",
                        entry=entry,
                        review_path=record.get("path"),
                    )
                )
                source_audit = scope.get("source_audit_json")
                source_audit_sha256 = scope.get("source_audit_sha256")
                if isinstance(source_audit, str) and source_audit.strip():
                    audit_role = (
                        f"paid_run_review_check:gate:{gate_name}:wandb_completion:"
                        f"{benchmark}:source_audit_json"
                    )
                    audit_record = validate_file_role(
                        errors=errors,
                        records=records,
                        path_value=source_audit,
                        role=audit_role,
                        label=f"paid review source audit JSON for {benchmark}",
                    )
                    if not isinstance(source_audit_sha256, str) or not re.fullmatch(
                        r"[0-9a-f]{64}",
                        source_audit_sha256 or "",
                    ):
                        errors.append(
                            f"paid review source audit JSON for {benchmark} "
                            "source_audit_sha256 must be a 64-character lowercase hex digest"
                        )
                    elif isinstance(audit_record, dict) and source_audit_sha256 != audit_record.get("sha256"):
                        errors.append(
                            f"paid review source audit JSON for {benchmark} "
                            "source_audit_sha256 does not match bundled source audit JSON"
                        )
    return errors


def nonempty_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def pre_run_budget_target_models(budget: dict[str, Any]) -> list[str]:
    values: list[str] = []
    target_model = budget.get("target_model")
    if isinstance(target_model, str) and target_model.strip():
        values.append(target_model.strip())
    values.extend(nonempty_string_list(budget.get("target_models")))
    return list(dict.fromkeys(values))


def pre_run_budget_has_selected_model_binding(budget: dict[str, Any]) -> bool:
    if nonempty_string_list(budget.get("selected_model_identifiers")):
        return True
    bindings = budget.get("selected_config_model_bindings")
    return isinstance(bindings, list) and bool(bindings)


def validate_pre_run_budget_model_binding(
    budget: dict[str, Any],
    *,
    label: str,
) -> list[str]:
    errors: list[str] = []
    if not pre_run_budget_has_selected_model_binding(budget):
        return errors

    target_models = pre_run_budget_target_models(budget)
    selected_identifiers = nonempty_string_list(budget.get("selected_model_identifiers"))
    if budget.get("target_model_matches_selected_config") is not True:
        errors.append(f"{label} target_model does not match selected config model identifiers")
    if not target_models:
        errors.append(f"{label} target_model or target_models must contain at least one model identifier")
    if not selected_identifiers:
        errors.append(f"{label} selected_model_identifiers is missing or empty")
    elif target_models and set(target_models).isdisjoint(selected_identifiers):
        errors.append(
            f"{label} target_model(s) do not intersect selected config model identifiers"
        )
    return errors


def validate_paid_review_pre_run_budget_evidence(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    current_gate = manifest.get("current_gate")
    if not isinstance(current_gate, dict):
        return errors
    paid_review_package = current_gate.get("paid_run_review_package")
    gates = paid_review_package.get("gates") if isinstance(paid_review_package, dict) else None
    if not isinstance(gates, list):
        return errors

    records_by_source = file_records_by_source(manifest)
    for gate in gates:
        if not isinstance(gate, dict):
            continue
        gate_name = str(gate.get("name") or "unknown_gate")
        review_records = gate.get("records")
        if not isinstance(review_records, list):
            continue
        for index, record in enumerate(review_records, start=1):
            if not isinstance(record, dict):
                continue
            budget = record.get("pre_run_budget_estimate")
            if not isinstance(budget, dict):
                continue
            label = f"paid review pre-run budget estimate {gate_name}#{index}"
            errors.extend(validate_pre_run_budget_model_binding(budget, label=label))
            budget_record = validate_any_file_role(
                errors=errors,
                records=records_by_source,
                path_value=budget.get("path"),
                role_suffix=":pre_run_budget_estimate",
                label=label,
            )
            if not isinstance(budget_record, dict):
                continue
            bundle_path = budget_record.get("bundle_path")
            if not isinstance(bundle_path, str) or not bundle_path:
                errors.append(f"{label} missing bundle_path")
                continue
            bundled_path = bundle_dir / bundle_path
            try:
                payload = read_json_object(bundled_path)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"{label} JSON is not readable: {exc}")
                continue
            if payload.get("target_model") != budget.get("target_model"):
                errors.append(f"{label} target_model does not match bundled budget JSON")
            expected_target_models = budget.get("target_models")
            if (
                isinstance(expected_target_models, list)
                and expected_target_models
                and isinstance(payload.get("target_models"), list)
                and payload.get("target_models") != expected_target_models
            ):
                errors.append(f"{label} target_models does not match bundled budget JSON")
            expected_totals = budget.get("estimated_total_usd")
            if isinstance(expected_totals, dict) and payload.get("estimated_total_usd") != expected_totals:
                errors.append(f"{label} estimated_total_usd does not match bundled budget JSON")
            expected_sha = budget.get("sha256")
            if isinstance(expected_sha, str) and expected_sha:
                actual_sha = sha256_file(bundled_path)
                if actual_sha != expected_sha:
                    errors.append(f"{label} sha256 does not match bundled budget JSON")
            review_key = source_path_key(record.get("path"))
            review_record = records_by_source.get(review_key) if review_key else None
            review_bundle_path = review_record.get("bundle_path") if isinstance(review_record, dict) else None
            if isinstance(review_bundle_path, str) and review_bundle_path:
                try:
                    review_payload = read_json_object(bundle_dir / review_bundle_path)
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    errors.append(f"{label} bundled paid review JSON is not readable: {exc}")
                    continue
                review_budget = review_payload.get("pre_run_budget_estimate")
                if isinstance(review_budget, dict):
                    errors.extend(
                        validate_pre_run_budget_model_binding(
                            review_budget,
                            label=f"{label} bundled paid review JSON",
                        )
                    )
                    for field in (
                        "sha256",
                        "target_model",
                        "target_models",
                        "selected_model_identifiers",
                        "target_model_matches_selected_config",
                        "estimated_total_usd",
                    ):
                        if field in budget and field in review_budget and budget.get(field) != review_budget.get(field):
                            errors.append(
                                f"{label} {field} does not match bundled paid review JSON"
                            )
    return errors


def validate_paid_review_external_action_approval_evidence(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    current_gate = manifest.get("current_gate")
    if not isinstance(current_gate, dict):
        return errors
    paid_review_package = current_gate.get("paid_run_review_package")
    gates = paid_review_package.get("gates") if isinstance(paid_review_package, dict) else None
    if not isinstance(gates, list):
        return errors

    records_by_source = file_records_by_source(manifest)
    for gate in gates:
        if not isinstance(gate, dict):
            continue
        gate_name = str(gate.get("name") or "unknown_gate")
        review_records = gate.get("records")
        if not isinstance(review_records, list):
            continue
        for index, record in enumerate(review_records, start=1):
            if not isinstance(record, dict):
                continue
            approval = record.get("external_action_approval")
            if not isinstance(approval, dict):
                continue
            required = bool(approval.get("required_before_external_action"))
            if not required and not approval.get("path"):
                continue
            label = f"paid review external-action approval report {gate_name}#{index}"
            approval_record = validate_any_file_role(
                errors=errors,
                records=records_by_source,
                path_value=approval.get("path"),
                role_suffix=":external_action_approval_report",
                label=label,
            )
            if not isinstance(approval_record, dict):
                continue
            bundle_path = approval_record.get("bundle_path")
            if not isinstance(bundle_path, str) or not bundle_path:
                errors.append(f"{label} missing bundle_path")
                continue
            bundled_path = bundle_dir / bundle_path
            try:
                payload = read_json_object(bundled_path)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"{label} JSON is not readable: {exc}")
                continue
            if payload.get("schema_version") != 1:
                errors.append(f"{label} schema_version must be 1")
            if payload.get("ok") is not True:
                errors.append(f"{label} ok must be true")
            if payload.get("status") != "approved":
                errors.append(f"{label} status must be approved")
            if payload.get("will_execute_external_actions") is not False:
                errors.append(f"{label} will_execute_external_actions must be false")
            source_binding = (
                payload.get("source_binding")
                if isinstance(payload.get("source_binding"), dict)
                else {}
            )
            if source_binding.get("bound") is not True:
                errors.append(f"{label} source_binding.bound must be true")
            if source_binding.get("errors") not in ([], None):
                errors.append(f"{label} source_binding.errors must be empty")
            if payload.get("required_approval_count") != approval.get("required_approval_count"):
                errors.append(
                    f"{label} required_approval_count does not match review record"
                )
            if payload.get("granted_approval_count") != approval.get("granted_approval_count"):
                errors.append(
                    f"{label} granted_approval_count does not match review record"
                )
            expected_sha = approval.get("sha256")
            if isinstance(expected_sha, str) and expected_sha:
                actual_sha = sha256_file(bundled_path)
                if actual_sha != expected_sha:
                    errors.append(
                        f"{label} sha256 does not match bundled approval report JSON"
                    )
    return errors


def validate_paid_review_package_accounting_evidence(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    current_gate = manifest.get("current_gate")
    if not isinstance(current_gate, dict):
        return errors
    paid_review_package = current_gate.get("paid_run_review_package")
    gates = paid_review_package.get("gates") if isinstance(paid_review_package, dict) else None
    if not isinstance(gates, list):
        return errors

    for gate in gates:
        if not isinstance(gate, dict):
            continue
        gate_name = str(gate.get("name") or "unknown_gate")
        records = gate.get("records")
        if not isinstance(records, list):
            continue
        for index, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                continue
            status = str(record.get("status") or "")
            record_label = f"paid review record {gate_name}#{index}"
            claims_completed = status == "completed" or record.get("ok") is True
            if not claims_completed:
                continue
            for field, flag in (
                ("actual_cost_estimate", "actual_cost_estimate_placeholder"),
                ("provider_bill_reference", "provider_bill_reference_placeholder"),
            ):
                if record.get(flag) is True:
                    errors.append(f"{record_label} {field} must not be a placeholder")
                if record.get(flag) not in (True, False):
                    errors.append(f"{record_label} missing boolean {flag}")
    return errors


def collect_review_weave_agents_completion_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    def append_entry(row: Any, *, parent_run_id: str = "") -> None:
        if not isinstance(row, dict):
            return
        path_value = row.get("path")
        agent_name = row.get("agent_name")
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            run_id = parent_run_id
        entries.append(
            {
                "path": path_value if isinstance(path_value, str) else "",
                "agent_name": agent_name if isinstance(agent_name, str) else "",
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "sync_dry_run_report_json": (
                    row.get("sync_dry_run_report_json")
                    if isinstance(row.get("sync_dry_run_report_json"), str)
                    else ""
                ),
                "sync_dry_run_source_review_json": (
                    row.get("sync_dry_run_source_review_json")
                    if isinstance(row.get("sync_dry_run_source_review_json"), str)
                    else ""
                ),
                "sync_dry_run_source_review_sha256": (
                    row.get("sync_dry_run_source_review_sha256")
                    if isinstance(row.get("sync_dry_run_source_review_sha256"), str)
                    else ""
                ),
            }
        )

    append_entry(payload.get("weave_agents_completion"))
    runs = payload.get("runs")
    if isinstance(runs, list):
        for run in runs:
            if not isinstance(run, dict):
                continue
            run_id = run.get("wandb_run_id")
            append_entry(
                run.get("weave_agents_completion"),
                parent_run_id=run_id if isinstance(run_id, str) else "",
            )
    return entries


def review_weave_agents_completion_entries_match(
    *,
    entry: dict[str, Any],
    raw_entry: dict[str, Any],
) -> bool:
    if source_path_key(raw_entry.get("path")) != source_path_key(entry.get("path")):
        return False
    for field in ("agent_name", "run_id"):
        value = entry.get(field)
        if isinstance(value, str) and value and raw_entry.get(field) != value:
            return False
    return True


def validate_weave_agents_sync_dry_run_report(
    payload: dict[str, Any],
    *,
    entry: dict[str, Any],
    review_path: Any,
    records: dict[str, dict[str, Any]] | None = None,
    label: str,
) -> list[str]:
    errors: list[str] = []
    if payload.get("ok") is not True:
        errors.append(f"{label} ok must be true")
    if payload.get("status") != "synced":
        errors.append(f"{label} status must be synced")
    generated_at = payload.get("generated_at")
    if (
        not isinstance(generated_at, (int, float))
        or not math.isfinite(float(generated_at))
        or generated_at <= 0
    ):
        errors.append(f"{label} generated_at must be a positive number")
    if payload.get("dry_run") is not True:
        errors.append(f"{label} dry_run must be true")
    if payload.get("in_place") is not False:
        errors.append(f"{label} in_place must be false")
    if payload.get("output_path") not in ("", None):
        errors.append(f"{label} output_path must be empty for a dry run")
    if source_path_key(payload.get("review_path")) != source_path_key(review_path):
        errors.append(f"{label} review_path does not match review entry")
    source_review_sha256 = entry.get("sync_dry_run_source_review_sha256")
    if not isinstance(source_review_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}",
        source_review_sha256,
    ):
        errors.append(f"{label} sync_dry_run_source_review_sha256 must be a 64-character lowercase hex digest")
    elif payload.get("source_review_sha256") != source_review_sha256:
        errors.append(f"{label} source_review_sha256 does not match review entry")
    elif isinstance(records, dict):
        source_review_key = source_path_key(review_path)
        source_review_record = records.get(source_review_key) if source_review_key else None
        if isinstance(source_review_record, dict):
            bundled_sha256 = source_review_record.get("sha256")
            if source_review_sha256 != bundled_sha256:
                errors.append(f"{label} source_review_sha256 does not match bundled source review JSON")
    if payload.get("verify_weave_agents") is not True:
        errors.append(f"{label} verify_weave_agents must be true")
    if payload.get("unmatched_count") != 0:
        errors.append(f"{label} unmatched_count must be 0")
    before_status = payload.get("before_status")
    after_status = payload.get("after_status")
    if not isinstance(before_status, str) or not before_status.strip():
        errors.append(f"{label} before_status is missing")
    if not isinstance(after_status, str) or not after_status.strip():
        errors.append(f"{label} after_status is missing")
    if (
        isinstance(before_status, str)
        and before_status.strip()
        and isinstance(after_status, str)
        and after_status.strip()
        and before_status != after_status
    ):
        errors.append(f"{label} before_status and after_status must match")

    entries = payload.get("entries")
    if not isinstance(entries, list):
        errors.append(f"{label} entries must be a list")
    elif payload.get("entry_count") != len(entries):
        errors.append(f"{label} entry_count must match entries length")
    else:
        matching_entries = [
            row
            for row in entries
            if isinstance(row, dict)
            and source_path_key(row.get("path")) == source_path_key(entry.get("path"))
            and row.get("run_id") == entry.get("run_id")
            and row.get("agent_name") == entry.get("agent_name")
        ]
        if not matching_entries:
            errors.append(f"{label} entries must include the claimed Weave completion")
        else:
            row = matching_entries[0]
            if row.get("ok") is not True:
                errors.append(f"{label} claimed entry ok must be true")
            if row.get("checks_valid") is not True:
                errors.append(f"{label} claimed entry checks_valid must be true")
            if row.get("trace_present") is not True:
                errors.append(f"{label} claimed entry trace_present must be true")
            if row.get("run_scope_proven") is not True:
                errors.append(f"{label} claimed entry run_scope_proven must be true")
            if row.get("request_model_proven") is not True:
                errors.append(f"{label} claimed entry request_model_proven must be true")
            latest_trace_id = entry.get("latest_trace_id")
            if isinstance(latest_trace_id, str) and latest_trace_id:
                if row.get("latest_trace_id") != latest_trace_id:
                    errors.append(f"{label} claimed entry latest_trace_id does not match")
            expected_query_fields = {
                "query_source_kind": WEAVE_AGENTS_QUERY_SOURCE_KIND,
                "query_source_api_base_url": WEAVE_AGENTS_API_BASE_URL,
                "query_source_agents_endpoint": WEAVE_AGENTS_QUERY_ENDPOINT,
                "query_source_spans_endpoint": WEAVE_AGENTS_SPANS_QUERY_ENDPOINT,
            }
            for field, expected in expected_query_fields.items():
                if row.get(field) != expected:
                    errors.append(f"{label} claimed entry {field} mismatch")

    changes = payload.get("changes")
    if not isinstance(changes, list):
        errors.append(f"{label} changes must be a list")
    elif payload.get("change_count") != len(changes):
        errors.append(f"{label} change_count must match changes length")
    else:
        matching_changes = [
            row
            for row in changes
            if isinstance(row, dict)
            and row.get("target") in {"run", "top_level"}
            and row.get("run_id") == entry.get("run_id")
            and row.get("agent_name") == entry.get("agent_name")
            and row.get("action") in {"added", "replaced", "kept_existing"}
        ]
        if not matching_changes:
            errors.append(f"{label} changes must include the claimed Weave completion")
    return errors


def validate_weave_agents_completion_review_source_evidence(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    current_gate = manifest.get("current_gate")
    if not isinstance(current_gate, dict):
        return errors
    rows = current_gate.get("weave_agents_completion")
    if not isinstance(rows, list):
        return errors

    records = file_records_by_source(manifest)
    for row_index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        entries = row.get("entries")
        if not isinstance(entries, list):
            continue
        claimed_entries = [
            entry
            for entry in entries
            if isinstance(entry, dict) and weave_agents_entry_is_claimed_proof(entry)
        ]
        if not claimed_entries:
            continue
        review_key = source_path_key(row.get("review_path"))
        row_label = f"Weave Agents completion row #{row_index}"
        if not review_key:
            errors.append(f"{row_label} review_path is missing")
            continue
        review_record = records.get(review_key)
        if not isinstance(review_record, dict):
            errors.append(f"missing bundled paid review JSON for {row_label}: {review_key}")
            continue
        bundle_path = review_record.get("bundle_path")
        if not isinstance(bundle_path, str) or not bundle_path:
            errors.append(f"bundled paid review JSON for {row_label} missing bundle_path: {review_key}")
            continue
        try:
            review_payload = read_json_object(bundle_dir / bundle_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"bundled paid review JSON for {row_label} is not readable: {exc}")
            continue

        if review_payload.get("status") != row.get("record_status"):
            errors.append(f"{row_label} paid review status does not match bundled review JSON")
        if review_payload.get("phase") != row.get("phase"):
            errors.append(f"{row_label} paid review phase does not match bundled review JSON")
        if review_payload.get("verify_weave_agents") != row.get("required"):
            errors.append(
                f"{row_label} paid review verify_weave_agents does not match bundled review JSON"
            )

        raw_entries = collect_review_weave_agents_completion_entries(review_payload)
        for entry_index, entry in enumerate(claimed_entries, start=1):
            label = f"Weave Agents completion entry row #{row_index}.{entry_index}"
            matching_raw_entries = [
                raw_entry
                for raw_entry in raw_entries
                if review_weave_agents_completion_entries_match(entry=entry, raw_entry=raw_entry)
            ]
            if not matching_raw_entries:
                errors.append(f"{label} is not present in bundled paid review JSON")
            dry_run_path = entry.get("sync_dry_run_report_json")
            if isinstance(dry_run_path, str) and dry_run_path.strip():
                if not any(
                    source_path_key(raw_entry.get("sync_dry_run_report_json"))
                    == source_path_key(dry_run_path)
                    for raw_entry in matching_raw_entries
                ):
                    errors.append(
                        f"{label} sync_dry_run_report_json is not present in bundled paid review JSON"
                    )
                source_review_path = entry.get("sync_dry_run_source_review_json")
                if isinstance(source_review_path, str) and source_review_path.strip():
                    if not any(
                        source_path_key(raw_entry.get("sync_dry_run_source_review_json"))
                        == source_path_key(source_review_path)
                        for raw_entry in matching_raw_entries
                    ):
                        errors.append(
                            f"{label} sync_dry_run_source_review_json is not present in bundled paid review JSON"
                        )
                source_review_sha256 = entry.get("sync_dry_run_source_review_sha256")
                if isinstance(source_review_sha256, str) and source_review_sha256.strip():
                    if not any(
                        raw_entry.get("sync_dry_run_source_review_sha256")
                        == source_review_sha256
                        for raw_entry in matching_raw_entries
                    ):
                        errors.append(
                            f"{label} sync_dry_run_source_review_sha256 is not present in bundled paid review JSON"
                        )
                dry_run_record = validate_file_role(
                    errors=errors,
                    records=records,
                    path_value=dry_run_path,
                    role=f"gate:{row.get('gate')}:weave_agents_completion_sync_dry_run",
                    label=f"{label} sync dry-run report",
                )
                if isinstance(dry_run_record, dict):
                    dry_run_bundle_path = dry_run_record.get("bundle_path")
                    if not isinstance(dry_run_bundle_path, str) or not dry_run_bundle_path:
                        errors.append(f"{label} sync dry-run report missing bundle_path")
                    else:
                        try:
                            dry_run_payload = read_json_object(bundle_dir / dry_run_bundle_path)
                        except (OSError, json.JSONDecodeError, ValueError) as exc:
                            errors.append(f"{label} sync dry-run report is not readable: {exc}")
                        else:
                            errors.extend(
                                validate_weave_agents_sync_dry_run_report(
                                    dry_run_payload,
                                    entry=entry,
                                    review_path=(
                                        entry.get("sync_dry_run_source_review_json")
                                        or row.get("review_path")
                                    ),
                                    records=records,
                                    label=f"{label} sync dry-run report",
                                )
                            )
    return errors


def add_weave_agents_completion_path(
    paths: dict[str, dict[str, str | None]],
    *,
    path_value: Any,
    agent_name: Any = None,
    latest_trace_id: Any = None,
    run_id: Any = None,
) -> None:
    key = source_path_key(path_value)
    if not key:
        return
    expected = paths.setdefault(
        key,
        {
            "agent_name": None,
            "latest_trace_id": None,
            "run_id": None,
        },
    )
    if isinstance(agent_name, str) and agent_name:
        expected["agent_name"] = agent_name
    if isinstance(latest_trace_id, str) and latest_trace_id:
        expected["latest_trace_id"] = latest_trace_id
    if isinstance(run_id, str) and run_id:
        expected["run_id"] = run_id


def weave_agents_entry_is_claimed_proof(entry: dict[str, Any]) -> bool:
    if entry.get("verified") is True:
        return True
    if (
        entry.get("entry_ok") is True
        and entry.get("schema_valid") is True
        and entry.get("checks_valid") is True
    ):
        return True
    return entry.get("ok") is True


def collect_weave_agents_completion_proof_paths(
    manifest: dict[str, Any],
) -> dict[str, dict[str, str | None]]:
    current_gate = manifest.get("current_gate")
    if not isinstance(current_gate, dict):
        return {}
    paths: dict[str, dict[str, str | None]] = {}

    rows = current_gate.get("weave_agents_completion")
    if not isinstance(rows, list):
        return paths
    for row in rows:
        if not isinstance(row, dict):
            continue
        entries = row.get("entries")
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if not weave_agents_entry_is_claimed_proof(entry):
                continue
            add_weave_agents_completion_path(
                paths,
                path_value=entry.get("path"),
                agent_name=entry.get("agent_name"),
                latest_trace_id=entry.get("latest_trace_id"),
                run_id=entry.get("run_id"),
            )
    return paths


def _numeric_minimum_check(
    payload: dict[str, Any],
    *,
    field: str,
    expected_min: Any,
    label: str,
    errors: list[str],
) -> None:
    if expected_min is None:
        return
    if not isinstance(expected_min, int):
        errors.append(f"{label} required_evidence.{field} is not an integer")
        return
    value = payload.get(field)
    if not isinstance(value, int):
        errors.append(f"{label} content_capture_health.{field} is not an integer")
        return
    if value < expected_min:
        errors.append(
            f"{label} content_capture_health.{field} below required minimum: "
            f"expected at least {expected_min}, got {value}"
        )


def _check_names(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {
        row.get("name")
        for row in value
        if isinstance(row, dict) and isinstance(row.get("name"), str)
    }


def _non_empty_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
    return result


def _check_by_name(value: Any, name: str) -> dict[str, Any] | None:
    if not isinstance(value, list):
        return None
    for row in value:
        if isinstance(row, dict) and row.get("name") == name:
            return row
    return None


def _validate_weave_agents_request_model_evidence(
    *,
    payload: dict[str, Any],
    required: dict[str, Any],
    health: dict[str, Any],
    label: str,
    errors: list[str],
) -> None:
    expected = _non_empty_string_list(required.get("expected_request_models"))
    if not expected:
        errors.append(
            f"{label} required_evidence.expected_request_models is not a non-empty list of strings"
        )

    check = _check_by_name(payload.get("checks"), "request_model")
    check_expected: list[str] = []
    check_observed: list[str] = []
    if check is None:
        errors.append(f"{label} checks missing required check: request_model")
    else:
        if check.get("ok") is not True:
            errors.append(f"{label} request_model check is not ok")
        check_expected = _non_empty_string_list(check.get("expected_request_models"))
        check_observed = _non_empty_string_list(check.get("observed_request_models"))
        if not check_expected:
            errors.append(
                f"{label} checks.request_model.expected_request_models is not a non-empty list of strings"
            )
        elif expected and set(check_expected) != set(expected):
            errors.append(
                f"{label} checks.request_model.expected_request_models does not match required_evidence"
            )
        if not check_observed:
            errors.append(
                f"{label} checks.request_model.observed_request_models is not a non-empty list of strings"
            )
        elif expected and set(expected).isdisjoint(check_observed):
            errors.append(
                f"{label} checks.request_model.observed_request_models does not include an expected model alias"
            )

    spans = payload.get("latest_trace_spans_chronological")
    span_rows = spans if isinstance(spans, list) else []
    span_models = sorted(
        {
            str(span.get("request_model")).strip()
            for span in span_rows
            if isinstance(span, dict)
            and isinstance(span.get("request_model"), str)
            and span.get("request_model").strip()
        }
    )
    if not span_models:
        errors.append(f"{label} latest_trace_spans_chronological does not expose request_model")
    elif expected and set(expected).isdisjoint(span_models):
        errors.append(
            f"{label} latest_trace_spans_chronological request_model values do not include an expected model alias"
        )
    if check_observed and span_models and set(check_observed) != set(span_models):
        errors.append(
            f"{label} checks.request_model.observed_request_models does not match latest_trace_spans_chronological"
        )

    request_model_count = health.get("request_model_count")
    if not isinstance(request_model_count, int) or request_model_count <= 0:
        errors.append(
            f"{label} content_capture_health.request_model_count is not a positive integer"
        )
    elif span_models and request_model_count != len(span_models):
        errors.append(
            f"{label} content_capture_health.request_model_count does not match unique request_model values"
        )


def _parse_weave_span_timestamp(value: Any) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _weave_span_timestamps(span: dict[str, Any]) -> tuple[float | None, float | None]:
    return (
        _parse_weave_span_timestamp(span.get("started_at")),
        _parse_weave_span_timestamp(span.get("ended_at")),
    )


def _weave_span_order_key(span: dict[str, Any]) -> tuple[float, float, str] | None:
    started_ts, ended_ts = _weave_span_timestamps(span)
    if started_ts is None or ended_ts is None:
        return None
    return (started_ts, ended_ts, str(span.get("span_id") or ""))


def _required_texts(value: Any, *, label: str, errors: list[str]) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        errors.append(f"{label} required_evidence.required_texts is not a list")
        return []
    result: list[str] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, str):
            errors.append(
                f"{label} required_evidence.required_texts[{index}] is not a string"
            )
            continue
        if item:
            result.append(item)
    return result


def validate_weave_agents_completion_payload(
    payload: dict[str, Any],
    *,
    label: str,
    expected_agent_name: str | None,
    expected_latest_trace_id: str | None,
    expected_run_id: str | None = None,
) -> list[str]:
    errors: list[str] = []
    if payload.get("ok") is not True:
        errors.append(f"{label} ok is not true")
    schema_version = payload.get("schema_version")
    if schema_version is not None and schema_version != 1:
        errors.append(f"{label} schema_version is not 1")
    if payload.get("verification_schema_version") != 1:
        errors.append(f"{label} verification_schema_version is not 1")
    if not isinstance(payload.get("generated_at"), (int, float)):
        errors.append(f"{label} generated_at is not numeric")
    if not isinstance(payload.get("project_id"), str) or not payload.get("project_id"):
        errors.append(f"{label} project_id is missing")
    if not isinstance(payload.get("agents_url"), str) or not payload.get("agents_url"):
        errors.append(f"{label} agents_url is missing")

    query_source = payload.get("query_source")
    if not isinstance(query_source, dict):
        errors.append(f"{label} query_source is not an object")
        query_source = {}
    else:
        expected_query_fields = {
            "kind": WEAVE_AGENTS_QUERY_SOURCE_KIND,
            "api_base_url": WEAVE_AGENTS_API_BASE_URL,
            "agents_endpoint": WEAVE_AGENTS_QUERY_ENDPOINT,
            "spans_endpoint": WEAVE_AGENTS_SPANS_QUERY_ENDPOINT,
            "project_id": payload.get("project_id"),
            "agent_name": payload.get("agent_name"),
        }
        for field, expected in expected_query_fields.items():
            if query_source.get(field) != expected:
                errors.append(
                    f"{label} query_source.{field} mismatch: "
                    f"expected {expected}, got {query_source.get(field)}"
                )
        for field in (
            "agents_count",
            "spans_count",
            "matching_span_count",
            "latest_trace_span_count",
        ):
            value = query_source.get(field)
            if not isinstance(value, int) or value < 0:
                errors.append(f"{label} query_source.{field} is not a non-negative integer")

    agent_name = payload.get("agent_name")
    if not isinstance(agent_name, str) or not agent_name:
        errors.append(f"{label} agent_name is missing")
    elif expected_agent_name and agent_name != expected_agent_name:
        errors.append(
            f"{label} agent_name mismatch: expected {expected_agent_name}, got {agent_name}"
        )

    latest_trace_id = payload.get("latest_trace_id")
    if not isinstance(latest_trace_id, str) or not latest_trace_id:
        errors.append(f"{label} latest_trace_id is missing")
    elif expected_latest_trace_id and latest_trace_id != expected_latest_trace_id:
        errors.append(
            f"{label} latest_trace_id mismatch: expected {expected_latest_trace_id}, got {latest_trace_id}"
        )

    required = payload.get("required_evidence")
    if not isinstance(required, dict):
        errors.append(f"{label} required_evidence is not an object")
        required = {}
    else:
        if required.get("input_message_required") is not True:
            errors.append(
                f"{label} required_evidence.input_message_required is not true"
            )
        if required.get("trace_timestamp_quality_required") is not True:
            errors.append(
                f"{label} required_evidence.trace_timestamp_quality_required is not true"
            )
        if required.get("trace_final_answer_order_required") is not True:
            errors.append(
                f"{label} required_evidence.trace_final_answer_order_required is not true"
            )
    required_texts = _required_texts(
        required.get("required_texts"),
        label=label,
        errors=errors,
    )
    if isinstance(query_source, dict):
        for field in ("conversation_id", "conversation_id_contains"):
            expected = required.get(field) if isinstance(required.get(field), str) else ""
            if query_source.get(field) != expected:
                errors.append(
                    f"{label} query_source.{field} does not match required_evidence"
                )
    if expected_run_id:
        conversation_id = required.get("conversation_id")
        if not isinstance(conversation_id, str):
            conversation_id = ""
        conversation_id_contains = required.get("conversation_id_contains")
        if not isinstance(conversation_id_contains, str):
            conversation_id_contains = ""
        if expected_run_id not in conversation_id and expected_run_id not in conversation_id_contains:
            errors.append(
                f"{label} required_evidence conversation scope does not include run_id "
                f"{expected_run_id}"
            )

    health = payload.get("content_capture_health")
    if not isinstance(health, dict):
        errors.append(f"{label} content_capture_health is not an object")
        health = {}

    checks = payload.get("checks")
    check_names = _check_names(checks)
    if not isinstance(checks, list) or not checks:
        errors.append(f"{label} checks is not a non-empty list")
    else:
        for index, check in enumerate(checks, start=1):
            if not isinstance(check, dict):
                errors.append(f"{label} checks[{index}] is not an object")
                continue
            if check.get("ok") is not True:
                name = check.get("name") or index
                errors.append(f"{label} checks contains failing check: {name}")
            if not isinstance(check.get("name"), str) or not check.get("name"):
                errors.append(f"{label} checks[{index}].name is missing")
        for required_check_name in sorted(REQUIRED_WEAVE_AGENTS_CHECK_NAMES - check_names):
            errors.append(f"{label} checks missing required check: {required_check_name}")
        if required_texts and "required_text_capture" not in check_names:
            errors.append(f"{label} checks missing required check: required_text_capture")

    _validate_weave_agents_request_model_evidence(
        payload=payload,
        required=required,
        health=health,
        label=label,
        errors=errors,
    )

    spans = payload.get("latest_trace_spans_chronological")
    valid_spans: list[dict[str, Any]] = []
    if not isinstance(spans, list) or not spans:
        errors.append(f"{label} latest_trace_spans_chronological is not a non-empty list")
        spans = []
    else:
        chronological_keys: list[tuple[str, str, str]] = []
        timestamp_order_keys: list[tuple[float, float, str]] = []
        for index, span in enumerate(spans, start=1):
            if not isinstance(span, dict):
                errors.append(f"{label} latest_trace_spans_chronological[{index}] is not an object")
                continue
            valid_spans.append(span)
            timestamp_key = _weave_span_order_key(span)
            if timestamp_key is None:
                errors.append(
                    f"{label} latest_trace_spans_chronological[{index}] has missing or invalid timestamps"
                )
            else:
                started_ts, ended_ts, _ = timestamp_key
                if ended_ts < started_ts:
                    errors.append(
                        f"{label} latest_trace_spans_chronological[{index}] ends before it starts"
                    )
                timestamp_order_keys.append(timestamp_key)
            chronological_keys.append(
                (
                    str(span.get("started_at") or ""),
                    str(span.get("ended_at") or ""),
                    str(span.get("span_id") or ""),
                )
            )
        if chronological_keys != sorted(chronological_keys):
            errors.append(f"{label} latest_trace_spans_chronological is not sorted by span time")
        if timestamp_order_keys != sorted(timestamp_order_keys):
            errors.append(f"{label} latest_trace_spans_chronological is not sorted by parsed span time")
        if isinstance(latest_trace_id, str) and latest_trace_id:
            mismatched_trace_spans = [
                span.get("span_id")
                for span in valid_spans
                if span.get("trace_id") != latest_trace_id
            ]
            if mismatched_trace_spans:
                errors.append(f"{label} latest_trace_spans_chronological contains spans from another trace")
        if isinstance(query_source, dict):
            latest_count = query_source.get("latest_trace_span_count")
            if isinstance(latest_count, int) and latest_count != len(valid_spans):
                errors.append(
                    f"{label} query_source.latest_trace_span_count does not match latest_trace_spans_chronological"
                )

    if valid_spans:
        message_spans = [
            span for span in valid_spans if span.get("operation_name") in {"chat", "invoke_agent"}
        ]
        tool_spans = [
            span for span in valid_spans if span.get("operation_name") == "execute_tool"
        ]
        input_message_spans = [
            span
            for span in message_spans
            if span.get("input_messages") or span.get("has_input_messages") is True
        ]
        final_answer_spans = []
        for span in message_spans:
            output_text = json.dumps(span.get("output_messages"), ensure_ascii=False)
            if any(
                marker in output_text
                for marker in (
                    "ANSWER:",
                    "FINAL ANSWER",
                    "Final answer",
                    "\\boxed",
                    "CANARY_RESULT",
                )
            ):
                final_answer_spans.append(span)
        if message_spans and tool_spans:
            message_starts = [
                _weave_span_timestamps(span)[0]
                for span in message_spans
                if _weave_span_timestamps(span)[0] is not None
            ]
            tool_starts = [
                _weave_span_timestamps(span)[0]
                for span in tool_spans
                if _weave_span_timestamps(span)[0] is not None
            ]
            if message_starts and tool_starts and min(tool_starts) <= min(message_starts):
                errors.append(f"{label} tool span starts before the first message span")
        if tool_spans:
            tool_starts = [
                _weave_span_timestamps(span)[0]
                for span in tool_spans
                if _weave_span_timestamps(span)[0] is not None
            ]
            if (
                isinstance(required, dict)
                and required.get("input_message_required") is True
                and not input_message_spans
            ):
                errors.append(
                    f"{label} tool span is present but no visible user/problem input span exists"
                )
            if input_message_spans:
                input_starts = [
                    _weave_span_timestamps(span)[0]
                    for span in input_message_spans
                    if _weave_span_timestamps(span)[0] is not None
                ]
                if input_starts and tool_starts and min(tool_starts) <= min(input_starts):
                    errors.append(
                        f"{label} tool span starts before the first visible user/problem input span"
                    )
        if final_answer_spans and tool_spans:
            final_answer_ends = [
                _weave_span_timestamps(span)[1]
                for span in final_answer_spans
                if _weave_span_timestamps(span)[1] is not None
            ]
            tool_starts = [
                _weave_span_timestamps(span)[0]
                for span in tool_spans
                if _weave_span_timestamps(span)[0] is not None
            ]
            if final_answer_ends and tool_starts and max(tool_starts) >= min(final_answer_ends):
                errors.append(f"{label} tool span starts after a final-answer message span")
        if isinstance(required, dict) and required.get("no_error_spans_required") is True:
            if any(span.get("error_type") for span in valid_spans):
                errors.append(f"{label} latest trace contains error spans")

    if isinstance(required, dict) and isinstance(health, dict):
        _numeric_minimum_check(
            health,
            field="span_count_checked",
            expected_min=required.get("min_trace_spans"),
            label=label,
            errors=errors,
        )
        _numeric_minimum_check(
            health,
            field="message_span_count",
            expected_min=required.get("min_message_spans"),
            label=label,
            errors=errors,
        )
        if required.get("content_required") is True:
            value = health.get("message_spans_with_content")
            if not isinstance(value, int) or value <= 0:
                errors.append(f"{label} required content is not visible in message spans")
        if required.get("input_message_required") is True:
            value = health.get("message_spans_with_input")
            if not isinstance(value, int) or value <= 0:
                errors.append(
                    f"{label} required user/problem input message is not visible"
                )
        if required.get("tool_span_required") is True:
            value = health.get("tool_span_count")
            if not isinstance(value, int) or value <= 0:
                errors.append(f"{label} required tool span is missing")
        if required.get("tool_content_required") is True:
            tool_count = health.get("tool_span_count")
            with_content = health.get("tool_spans_with_content")
            if not isinstance(tool_count, int) or tool_count <= 0:
                errors.append(f"{label} required tool content has no tool spans")
            elif not isinstance(with_content, int) or with_content < tool_count:
                errors.append(f"{label} required tool content is not visible for every tool span")
        if required_texts:
            required_text_count = health.get("required_text_count")
            if not isinstance(required_text_count, int) or required_text_count < len(required_texts):
                errors.append(
                    f"{label} content_capture_health.required_text_count below required minimum: "
                    f"expected at least {len(required_texts)}, got {required_text_count}"
                )
        if required.get("trace_timestamp_quality_required") is True:
            valid_count = health.get("spans_with_valid_timestamps")
            invalid_count = health.get("spans_with_invalid_timestamps")
            if not isinstance(valid_count, int):
                errors.append(
                    f"{label} content_capture_health.spans_with_valid_timestamps is not an integer"
                )
            elif valid_count != len(valid_spans):
                errors.append(
                    f"{label} content_capture_health.spans_with_valid_timestamps does not match span count"
                )
            if not isinstance(invalid_count, int):
                errors.append(
                    f"{label} content_capture_health.spans_with_invalid_timestamps is not an integer"
                )
            elif invalid_count != 0:
                errors.append(
                    f"{label} content_capture_health.spans_with_invalid_timestamps is not zero"
                )

    return errors


def validate_weave_agents_completion_proof_evidence(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    proof_paths = collect_weave_agents_completion_proof_paths(manifest)
    if not proof_paths:
        return errors
    records = file_records_by_source(manifest)
    for source_path, expected in sorted(proof_paths.items()):
        record = records.get(source_path)
        if not isinstance(record, dict):
            errors.append(f"missing bundled Weave Agents completion proof JSON: {source_path}")
            continue
        roles = record.get("roles")
        if not isinstance(roles, list) or not any("weave_agents_completion" in str(role) for role in roles):
            errors.append(f"bundled Weave Agents completion proof JSON has no weave_agents_completion role: {source_path}")
        bundle_path = record.get("bundle_path")
        if not isinstance(bundle_path, str) or not bundle_path:
            errors.append(f"bundled Weave Agents completion proof JSON missing bundle_path: {source_path}")
            continue
        try:
            payload = read_json_object(bundle_dir / bundle_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"Weave Agents completion proof JSON is not readable: {source_path}: {exc}")
            continue
        errors.extend(
            validate_weave_agents_completion_payload(
                payload,
                label=f"Weave Agents completion proof {source_path}",
                expected_agent_name=expected.get("agent_name"),
                expected_latest_trace_id=expected.get("latest_trace_id"),
                expected_run_id=expected.get("run_id"),
            )
        )
    return errors


def validate_weave_agents_diagnostic_payload(
    payload: dict[str, Any],
    *,
    label: str,
    expected_agent_name: str | None,
    expected_project_id: str | None,
    expected_task_id: str | None,
    expected_latest_trace_id: str | None = None,
) -> list[str]:
    errors: list[str] = []
    if payload.get("diagnostic_schema_version") != 1:
        errors.append(f"{label} diagnostic_schema_version is not 1")
    if not isinstance(payload.get("generated_at"), (int, float)):
        errors.append(f"{label} generated_at is not numeric")
    if expected_project_id and payload.get("project_id") != expected_project_id:
        errors.append(
            f"{label} project_id mismatch: expected {expected_project_id}, got {payload.get('project_id')}"
        )
    if expected_agent_name and payload.get("agent_name_filter") != expected_agent_name:
        errors.append(
            f"{label} agent_name_filter mismatch: expected {expected_agent_name}, "
            f"got {payload.get('agent_name_filter')}"
        )

    query_source = payload.get("query_source")
    if not isinstance(query_source, dict):
        errors.append(f"{label} query_source is not an object")
        query_source = {}
    expected_query_fields = {
        "kind": WEAVE_AGENTS_QUERY_SOURCE_KIND,
        "api_base_url": WEAVE_AGENTS_API_BASE_URL,
        "agents_endpoint": WEAVE_AGENTS_QUERY_ENDPOINT,
        "spans_endpoint": WEAVE_AGENTS_SPANS_QUERY_ENDPOINT,
    }
    if expected_project_id:
        expected_query_fields["project_id"] = expected_project_id
    if expected_agent_name:
        expected_query_fields["agent_name"] = expected_agent_name
    for field, expected in expected_query_fields.items():
        if query_source.get(field) != expected:
            errors.append(
                f"{label} query_source.{field} mismatch: "
                f"expected {expected}, got {query_source.get(field)}"
            )
    for field in (
        "agents_count",
        "spans_count",
        "matching_span_count",
        "latest_trace_span_count",
    ):
        value = query_source.get(field)
        if not isinstance(value, int) or value < 1:
            errors.append(f"{label} query_source.{field} is not a positive integer")
    if expected_task_id:
        scope_values = [
            value
            for value in (
                query_source.get("conversation_id"),
                query_source.get("conversation_id_contains"),
            )
            if isinstance(value, str) and value
        ]
        if not any(expected_task_id in value for value in scope_values):
            errors.append(
                f"{label} query_source conversation scope does not include task_id "
                f"{expected_task_id}"
            )

    latest_trace_id = payload.get("latest_trace_id")
    if not isinstance(latest_trace_id, str) or not latest_trace_id:
        errors.append(f"{label} latest_trace_id is missing")
    elif expected_latest_trace_id and latest_trace_id != expected_latest_trace_id:
        errors.append(
            f"{label} latest_trace_id mismatch: expected {expected_latest_trace_id}, "
            f"got {latest_trace_id}"
        )

    spans = payload.get("latest_trace_spans_chronological")
    valid_spans: list[dict[str, Any]] = []
    if not isinstance(spans, list) or not spans:
        errors.append(f"{label} latest_trace_spans_chronological is not a non-empty list")
        spans = []
    else:
        timestamp_order_keys: list[tuple[float, float, str]] = []
        for index, span in enumerate(spans, start=1):
            if not isinstance(span, dict):
                errors.append(f"{label} latest_trace_spans_chronological[{index}] is not an object")
                continue
            valid_spans.append(span)
            timestamp_key = _weave_span_order_key(span)
            if timestamp_key is None:
                errors.append(
                    f"{label} latest_trace_spans_chronological[{index}] has missing or invalid timestamps"
                )
            else:
                started_ts, ended_ts, _ = timestamp_key
                if ended_ts < started_ts:
                    errors.append(
                        f"{label} latest_trace_spans_chronological[{index}] ends before it starts"
                    )
                timestamp_order_keys.append(timestamp_key)
            if expected_agent_name and span.get("agent_name") != expected_agent_name:
                errors.append(
                    f"{label} latest_trace_spans_chronological[{index}].agent_name mismatch"
                )
            if isinstance(latest_trace_id, str) and latest_trace_id and span.get("trace_id") != latest_trace_id:
                errors.append(
                    f"{label} latest_trace_spans_chronological[{index}].trace_id does not match latest_trace_id"
                )
            conversation_id = span.get("conversation_id")
            if expected_task_id and (
                not isinstance(conversation_id, str) or expected_task_id not in conversation_id
            ):
                errors.append(
                    f"{label} latest_trace_spans_chronological[{index}].conversation_id "
                    f"does not include task_id {expected_task_id}"
                )
        if timestamp_order_keys != sorted(timestamp_order_keys):
            errors.append(f"{label} latest_trace_spans_chronological is not sorted by parsed span time")
        latest_count = query_source.get("latest_trace_span_count")
        if isinstance(latest_count, int) and latest_count != len(valid_spans):
            errors.append(
                f"{label} query_source.latest_trace_span_count does not match latest_trace_spans_chronological"
            )

    health = payload.get("content_capture_health")
    if not isinstance(health, dict):
        errors.append(f"{label} content_capture_health is not an object")
        health = {}
    for field in (
        "trace_timestamp_quality_ok",
        "trace_order_ok",
        "trace_user_message_order_ok",
        "trace_final_answer_order_ok",
    ):
        if health.get(field) is not True:
            errors.append(f"{label} content_capture_health.{field} is not true")
    for field in (
        "message_spans_with_input",
        "tool_span_count",
        "tool_spans_with_content",
        "final_answer_span_count",
    ):
        value = health.get(field)
        if not isinstance(value, int) or value < 1:
            errors.append(f"{label} content_capture_health.{field} is not a positive integer")
    if health.get("spans_with_invalid_timestamps") != 0:
        errors.append(f"{label} content_capture_health.spans_with_invalid_timestamps is not 0")

    trace_order = payload.get("trace_order_health")
    if not isinstance(trace_order, dict):
        errors.append(f"{label} trace_order_health is not an object")
        trace_order = {}
    for field in (
        "timestamp_quality_ok",
        "trace_order_ok",
        "trace_user_message_order_ok",
        "trace_final_answer_order_ok",
    ):
        if trace_order.get(field) is not True:
            errors.append(f"{label} trace_order_health.{field} is not true")
    order_issues = trace_order.get("order_issues")
    if not isinstance(order_issues, list):
        errors.append(f"{label} trace_order_health.order_issues is not a list")
    elif order_issues:
        errors.append(f"{label} trace_order_health.order_issues is not empty")
    return errors


def collect_weave_content_canary_gate_paths(manifest: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    gates = manifest.get("gates")
    if not isinstance(gates, list):
        return paths
    for gate in gates:
        if not isinstance(gate, dict) or gate.get("name") != "weave_content_canary":
            continue
        evidence_paths = gate.get("evidence_paths")
        if not isinstance(evidence_paths, list):
            continue
        for path_value in evidence_paths:
            key = source_path_key(path_value)
            if key and key not in paths:
                paths.append(key)
    return paths


def validate_weave_content_canary_gate_payload(
    payload: dict[str, Any],
    *,
    label: str,
) -> list[str]:
    errors: list[str] = []
    if payload.get("gate") != "weave_agents_content_canary":
        errors.append(f"{label} gate is not weave_agents_content_canary")
    if not isinstance(payload.get("ok"), bool):
        errors.append(f"{label} ok is not boolean")
    status = payload.get("status")
    if not isinstance(status, str) or not status:
        errors.append(f"{label} status is missing")
    if not isinstance(payload.get("generated_at"), (int, float)):
        errors.append(f"{label} generated_at is not numeric")
    paths = payload.get("paths")
    if not isinstance(paths, dict):
        errors.append(f"{label} paths is not an object")
        paths = {}

    claims_passed = payload.get("ok") is True or status == "passed"
    if not claims_passed:
        return errors

    if payload.get("ok") is not True:
        errors.append(f"{label} ok is not true for a passed gate")
    if status != "passed":
        errors.append(f"{label} status is not passed for an ok gate")
    if payload.get("weave_verifier_ok") is not True:
        errors.append(f"{label} weave_verifier_ok is not true")
    if payload.get("weave_verifier_schema_version") != 1:
        errors.append(f"{label} weave_verifier_schema_version is not 1")
    if not isinstance(payload.get("weave_verifier_latest_trace_id"), str) or not payload.get("weave_verifier_latest_trace_id"):
        errors.append(f"{label} weave_verifier_latest_trace_id is missing")
    if payload.get("agents_diagnostic_ok") is not True:
        errors.append(f"{label} agents_diagnostic_ok is not true")
    if payload.get("agents_diagnostic_schema_version") != 1:
        errors.append(f"{label} agents_diagnostic_schema_version is not 1")
    if not isinstance(payload.get("agents_diagnostic_latest_trace_id"), str) or not payload.get("agents_diagnostic_latest_trace_id"):
        errors.append(f"{label} agents_diagnostic_latest_trace_id is missing")
    elif payload.get("weave_verifier_latest_trace_id") and (
        payload.get("agents_diagnostic_latest_trace_id")
        != payload.get("weave_verifier_latest_trace_id")
    ):
        errors.append(f"{label} agents_diagnostic_latest_trace_id does not match weave_verifier_latest_trace_id")
    diagnostic_issues = payload.get("agents_diagnostic_validation_issues")
    if not isinstance(diagnostic_issues, list):
        errors.append(f"{label} agents_diagnostic_validation_issues is not a list")
    elif diagnostic_issues:
        errors.append(f"{label} agents_diagnostic_validation_issues is not empty")
    if not isinstance(payload.get("canary_id"), str) or not payload.get("canary_id"):
        errors.append(f"{label} canary_id is missing for a passed gate")
    if not isinstance(payload.get("task_id"), str) or not payload.get("task_id"):
        errors.append(f"{label} task_id is missing for a passed gate")
    expected_request_models = _non_empty_string_list(payload.get("expected_request_models"))
    observed_request_models = _non_empty_string_list(payload.get("observed_request_models"))
    span_request_models = _non_empty_string_list(payload.get("span_request_models"))
    if not expected_request_models:
        errors.append(f"{label} expected_request_models is not a non-empty list")
    if payload.get("request_model_proven") is not True:
        errors.append(f"{label} request_model_proven is not true")
    if not observed_request_models:
        errors.append(f"{label} observed_request_models is not a non-empty list")
    elif expected_request_models and set(expected_request_models).isdisjoint(
        observed_request_models
    ):
        errors.append(
            f"{label} observed_request_models does not include an expected model alias"
        )
    if not span_request_models:
        errors.append(f"{label} span_request_models is not a non-empty list")
    elif expected_request_models and set(expected_request_models).isdisjoint(
        span_request_models
    ):
        errors.append(
            f"{label} span_request_models does not include an expected model alias"
        )
    health = payload.get("content_capture_health")
    if not isinstance(health, dict):
        errors.append(f"{label} content_capture_health is not an object")
    else:
        request_model_count = health.get("request_model_count")
        if not isinstance(request_model_count, int) or request_model_count <= 0:
            errors.append(
                f"{label} content_capture_health.request_model_count must be positive"
            )
    nemoclaw = payload.get("nemoclaw")
    if not isinstance(nemoclaw, dict):
        errors.append(f"{label} nemoclaw is not an object for a passed gate")
    else:
        if nemoclaw.get("required") is not True:
            errors.append(f"{label} nemoclaw.required is not true")
        if nemoclaw.get("enabled") is not True:
            errors.append(f"{label} nemoclaw.enabled is not true")
        for field in ("sandbox", "bin", "workdir"):
            if not isinstance(nemoclaw.get(field), str) or not nemoclaw.get(field):
                errors.append(f"{label} nemoclaw.{field} is missing for a passed gate")
    issues = payload.get("weave_verifier_validation_issues")
    if not isinstance(issues, list):
        errors.append(f"{label} weave_verifier_validation_issues is not a list")
    elif issues:
        errors.append(f"{label} weave_verifier_validation_issues is not empty")
    for field in (
        "plan_file",
        "command_result_file",
        "verifier_json",
        "agents_diagnostic_json",
        "expected_sidecar",
        "prompt_file",
    ):
        if not isinstance(paths.get(field), str) or not paths.get(field):
            errors.append(f"{label} paths.{field} is missing for a passed gate")
    return errors


def _bundled_json_from_record(
    *,
    bundle_dir: Path,
    record: dict[str, Any] | None,
    label: str,
    errors: list[str],
) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    bundle_path = record.get("bundle_path")
    if not isinstance(bundle_path, str) or not bundle_path:
        errors.append(f"{label} missing bundle_path")
        return None
    try:
        return read_json_object(bundle_dir / bundle_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"{label} is not readable JSON: {exc}")
        return None


def _bundled_text_from_record(
    *,
    bundle_dir: Path,
    record: dict[str, Any] | None,
    label: str,
    errors: list[str],
) -> str | None:
    if not isinstance(record, dict):
        return None
    bundle_path = record.get("bundle_path")
    if not isinstance(bundle_path, str) or not bundle_path:
        errors.append(f"{label} missing bundle_path")
        return None
    try:
        return (bundle_dir / bundle_path).read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"{label} is not readable text: {exc}")
        return None


def _content_canary_conversation_values(payload: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for container_name in ("query_source", "required_evidence"):
        container = payload.get(container_name)
        if not isinstance(container, dict):
            continue
        for field in ("conversation_id", "conversation_id_contains"):
            value = container.get(field)
            if isinstance(value, str) and value:
                values.append(value)
    return values


def validate_weave_content_canary_support_evidence(
    *,
    gate_payload: dict[str, Any],
    plan_payload: dict[str, Any] | None,
    command_result_payload: dict[str, Any] | None,
    verifier_payload: dict[str, Any] | None,
    agents_diagnostic_payload: dict[str, Any] | None,
    sidecar_payload: dict[str, Any] | None,
    prompt_text: str | None,
    label: str,
) -> list[str]:
    errors: list[str] = []
    canary_id = gate_payload.get("canary_id")
    task_id = gate_payload.get("task_id")
    expected_final_text = (
        f"CANARY_RESULT {canary_id} 91" if isinstance(canary_id, str) and canary_id else ""
    )

    if isinstance(plan_payload, dict):
        expected_pairs = {
            "canary_id": canary_id,
            "task_id": task_id,
            "model": gate_payload.get("model"),
            "thinking": gate_payload.get("thinking"),
            "agent_name": gate_payload.get("agent_name"),
            "entity": gate_payload.get("entity"),
            "project": gate_payload.get("project"),
        }
        for field, expected in expected_pairs.items():
            if expected is not None and plan_payload.get(field) != expected:
                errors.append(
                    f"{label} plan.{field} mismatch: expected {expected}, "
                    f"got {plan_payload.get(field)}"
                )
        paths = gate_payload.get("paths") if isinstance(gate_payload.get("paths"), dict) else {}
        diagnostic_path = paths.get("agents_diagnostic_json")
        if isinstance(diagnostic_path, str) and diagnostic_path:
            plan_diagnostic_path = plan_payload.get("agents_diagnostic_file")
            if plan_diagnostic_path != diagnostic_path:
                errors.append(
                    f"{label} plan.agents_diagnostic_file mismatch: expected "
                    f"{diagnostic_path}, got {plan_diagnostic_path}"
                )
        if plan_payload.get("will_call_paid_model_api") is not True:
            errors.append(f"{label} plan.will_call_paid_model_api is not true")
        if plan_payload.get("will_execute_external_actions") is not True:
            errors.append(f"{label} plan.will_execute_external_actions is not true")
        requirements = plan_payload.get("verification_requirements")
        if not isinstance(requirements, dict):
            errors.append(f"{label} plan.verification_requirements is not an object")
        else:
            required_texts = requirements.get("required_texts")
            if not isinstance(required_texts, list):
                errors.append(f"{label} plan.verification_requirements.required_texts is not a list")
            else:
                for required_text in (canary_id, expected_final_text):
                    if required_text and required_text not in required_texts:
                        errors.append(
                            f"{label} plan.verification_requirements.required_texts "
                            f"does not include {required_text!r}"
                        )
            if requirements.get("require_content") is not True:
                errors.append(f"{label} plan.verification_requirements.require_content is not true")
            if requirements.get("require_tool_span") is not True:
                errors.append(f"{label} plan.verification_requirements.require_tool_span is not true")
            if requirements.get("require_tool_content") is not True:
                errors.append(f"{label} plan.verification_requirements.require_tool_content is not true")

    if isinstance(command_result_payload, dict):
        if command_result_payload.get("ok") is not True:
            errors.append(f"{label} command_result.ok is not true")
        if command_result_payload.get("returncode") != 0:
            errors.append(f"{label} command_result.returncode is not 0")

    if isinstance(verifier_payload, dict):
        project = gate_payload.get("project")
        entity = gate_payload.get("entity")
        if isinstance(entity, str) and entity and isinstance(project, str) and project:
            expected_project_id = f"{entity}/{project}"
            if verifier_payload.get("project_id") != expected_project_id:
                errors.append(
                    f"{label} verifier project_id mismatch: expected "
                    f"{expected_project_id}, got {verifier_payload.get('project_id')}"
                )
        if isinstance(task_id, str) and task_id:
            if not any(task_id in value for value in _content_canary_conversation_values(verifier_payload)):
                errors.append(
                    f"{label} verifier conversation scope does not include task_id {task_id}"
                )
            spans = verifier_payload.get("latest_trace_spans_chronological")
            if isinstance(spans, list):
                for index, span in enumerate(spans, start=1):
                    if not isinstance(span, dict):
                        continue
                    conversation_id = span.get("conversation_id")
                    if not isinstance(conversation_id, str) or task_id not in conversation_id:
                        errors.append(
                            f"{label} verifier latest_trace_spans_chronological[{index}]."
                            f"conversation_id does not include task_id {task_id}"
                        )
        required = verifier_payload.get("required_evidence")
        if not isinstance(required, dict):
            errors.append(f"{label} verifier.required_evidence is not an object")
        else:
            required_texts = required.get("required_texts")
            if not isinstance(required_texts, list):
                errors.append(f"{label} verifier.required_evidence.required_texts is not a list")
            else:
                for required_text in (canary_id, expected_final_text):
                    if required_text and required_text not in required_texts:
                        errors.append(
                            f"{label} verifier.required_evidence.required_texts "
                            f"does not include {required_text!r}"
                        )

    if not isinstance(agents_diagnostic_payload, dict):
        errors.append(f"{label} agents_diagnostic_json is not an object")

    if isinstance(sidecar_payload, dict):
        if sidecar_payload.get("ok") is False:
            errors.append(f"{label} expected_sidecar.ok is false")

    if prompt_text is not None:
        for required_text in (canary_id, expected_final_text):
            if required_text and required_text not in prompt_text:
                errors.append(f"{label} prompt_file does not include {required_text!r}")
        if "Python execution tool exactly once" not in prompt_text:
            errors.append(
                f"{label} prompt_file does not require exactly one Python execution tool use"
            )

    return errors


def validate_weave_content_canary_gate_evidence(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    proof_paths = collect_weave_content_canary_gate_paths(manifest)
    if not proof_paths:
        return errors
    records = file_records_by_source(manifest)
    for source_path in sorted(proof_paths):
        record = records.get(source_path)
        if not isinstance(record, dict):
            errors.append(f"missing bundled Weave content canary gate JSON: {source_path}")
            continue
        roles = record.get("roles")
        if not isinstance(roles, list) or "gate:weave_content_canary:evidence" not in roles:
            errors.append(f"bundled Weave content canary gate JSON has no gate:weave_content_canary:evidence role: {source_path}")
        bundle_path = record.get("bundle_path")
        if not isinstance(bundle_path, str) or not bundle_path:
            errors.append(f"bundled Weave content canary gate JSON missing bundle_path: {source_path}")
            continue
        try:
            payload = read_json_object(bundle_dir / bundle_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"Weave content canary gate JSON is not readable: {source_path}: {exc}")
            continue

        label = f"Weave content canary gate {source_path}"
        errors.extend(validate_weave_content_canary_gate_payload(payload, label=label))
        if not (payload.get("ok") is True or payload.get("status") == "passed"):
            continue

        paths = payload.get("paths") if isinstance(payload.get("paths"), dict) else {}
        support_payloads: dict[str, dict[str, Any] | None] = {}
        support_texts: dict[str, str | None] = {}
        support_roles = {
            "plan_file": "gate:weave_content_canary:evidence:plan_file",
            "command_result_file": "gate:weave_content_canary:evidence:command_result_file",
            "expected_sidecar": "gate:weave_content_canary:evidence:expected_sidecar",
            "prompt_file": "gate:weave_content_canary:evidence:prompt_file",
            "agents_diagnostic_json": "gate:weave_content_canary:evidence:agents_diagnostic_json",
        }
        for field, role in support_roles.items():
            support_record = validate_file_role(
                errors=errors,
                records=records,
                path_value=paths.get(field),
                role=role,
                label=f"{label} {field}",
            )
            if field == "prompt_file":
                support_texts[field] = _bundled_text_from_record(
                    bundle_dir=bundle_dir,
                    record=support_record,
                    label=f"{label} {field}",
                    errors=errors,
                )
            else:
                support_payloads[field] = _bundled_json_from_record(
                    bundle_dir=bundle_dir,
                    record=support_record,
                    label=f"{label} {field}",
                    errors=errors,
                )

        verifier_record = validate_file_role(
            errors=errors,
            records=records,
            path_value=paths.get("verifier_json"),
            role="gate:weave_content_canary:evidence:verifier_json",
            label=f"{label} verifier JSON",
        )
        verifier_payload = _bundled_json_from_record(
            bundle_dir=bundle_dir,
            record=verifier_record,
            label=f"{label} verifier JSON",
            errors=errors,
        )
        if isinstance(verifier_payload, dict):
            errors.extend(
                validate_weave_agents_completion_payload(
                    verifier_payload,
                    label=f"{label} verifier JSON",
                    expected_agent_name=payload.get("agent_name"),
                    expected_latest_trace_id=payload.get("weave_verifier_latest_trace_id"),
                )
            )
        diagnostic_payload = support_payloads.get("agents_diagnostic_json")
        if isinstance(diagnostic_payload, dict):
            expected_project_id = None
            entity = payload.get("entity")
            project = payload.get("project")
            if isinstance(entity, str) and entity and isinstance(project, str) and project:
                expected_project_id = f"{entity}/{project}"
            errors.extend(
                validate_weave_agents_diagnostic_payload(
                    diagnostic_payload,
                    label=f"{label} agents diagnostic JSON",
                    expected_agent_name=payload.get("agent_name")
                    if isinstance(payload.get("agent_name"), str)
                    else None,
                    expected_project_id=expected_project_id,
                    expected_task_id=payload.get("task_id")
                    if isinstance(payload.get("task_id"), str)
                    else None,
                    expected_latest_trace_id=payload.get("weave_verifier_latest_trace_id")
                    if isinstance(payload.get("weave_verifier_latest_trace_id"), str)
                    else None,
                )
            )
        errors.extend(
            validate_weave_content_canary_support_evidence(
                gate_payload=payload,
                plan_payload=support_payloads.get("plan_file"),
                command_result_payload=support_payloads.get("command_result_file"),
                verifier_payload=verifier_payload,
                agents_diagnostic_payload=diagnostic_payload,
                sidecar_payload=support_payloads.get("expected_sidecar"),
                prompt_text=support_texts.get("prompt_file"),
                label=label,
            )
        )
    return errors


def wandb_adoption_source_audit_record_matches_candidate(
    record: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    if record.get("benchmark") != candidate.get("benchmark"):
        return False
    completion = record.get("wandb_completion")
    if not isinstance(completion, dict):
        return False
    if source_path_key(completion.get("path")) != source_path_key(
        candidate.get("wandb_completion_json")
    ):
        return False
    comparisons = (
        ("entity", "wandb_entity"),
        ("project", "wandb_project"),
        ("run_id", "wandb_run_id"),
    )
    for completion_key, candidate_key in comparisons:
        completion_value = completion.get(completion_key)
        candidate_value = candidate.get(candidate_key)
        if (
            isinstance(completion_value, str)
            and completion_value.strip()
            and completion_value != candidate_value
        ):
            return False
    verification_schema_version = completion.get("verification_schema_version")
    candidate_schema_version = candidate.get("verification_schema_version")
    if (
        verification_schema_version is not None
        and candidate_schema_version is not None
        and verification_schema_version != candidate_schema_version
    ):
        return False
    return True


def wandb_adoption_source_audit_completion_record_matches_candidate(
    record: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    if record.get("benchmark") != candidate.get("benchmark"):
        return False
    if source_path_key(record.get("path")) != source_path_key(
        candidate.get("wandb_completion_json")
    ):
        return False
    comparisons = (
        ("entity", "wandb_entity"),
        ("project", "wandb_project"),
        ("run_id", "wandb_run_id"),
    )
    for record_key, candidate_key in comparisons:
        if record.get(record_key) != candidate.get(candidate_key):
            return False
    candidate_schema_version = candidate.get("verification_schema_version")
    if (
        candidate_schema_version is not None
        and record.get("verification_schema_version") != candidate_schema_version
    ):
        return False
    return True


def validate_wandb_adoption_source_audit_completion_record(
    record: dict[str, Any],
    *,
    benchmark: str,
) -> list[str]:
    errors: list[str] = []
    prefix = (
        "W&B adoption draft candidate "
        f"{benchmark} source_audit_json wandb_completion_records candidate"
    )
    if record.get("ok") is not True:
        errors.append(f"{prefix} ok is not true")
    if record.get("verification_schema_version") != WANDB_COMPLETION_SCHEMA_VERSION:
        errors.append(f"{prefix} verification_schema_version must be 1")
    if record.get("schema_current") is not True:
        errors.append(f"{prefix} schema_current is not true")
    schema_issues = record.get("schema_current_issues")
    if not isinstance(schema_issues, list):
        errors.append(f"{prefix} schema_current_issues must be a list")
    elif schema_issues:
        errors.append(f"{prefix} schema_current_issues must be empty")
    if record.get("observed_evidence_present") is not True:
        errors.append(f"{prefix} observed_evidence_present is not true")
    return errors


def validate_wandb_adoption_candidate_source_audit(
    *,
    candidate: dict[str, Any],
    benchmark: str,
    source_audit_payload: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    formalized_records = source_audit_payload.get("formalized_records")
    if not isinstance(formalized_records, list):
        errors.append(
            "W&B adoption draft source_audit_json formalized_records must be a list"
        )
        formalized_records = []
    if not any(
        isinstance(record, dict)
        and wandb_adoption_source_audit_record_matches_candidate(record, candidate)
        for record in formalized_records
    ):
        errors.append(
            "W&B adoption draft candidate "
            f"{benchmark} source_audit_json formalized_records does not "
            "include candidate W&B completion"
        )

    completion_records = source_audit_payload.get("wandb_completion_records")
    if not isinstance(completion_records, list):
        errors.append(
            "W&B adoption draft source_audit_json wandb_completion_records "
            "must be a list"
        )
        return errors
    matching_completion_records = [
        record
        for record in completion_records
        if isinstance(record, dict)
        and wandb_adoption_source_audit_completion_record_matches_candidate(
            record,
            candidate,
        )
    ]
    if not matching_completion_records:
        errors.append(
            "W&B adoption draft candidate "
            f"{benchmark} source_audit_json wandb_completion_records does not "
            "include candidate W&B completion"
        )
        return errors
    if len(matching_completion_records) > 1:
        errors.append(
            "W&B adoption draft candidate "
            f"{benchmark} source_audit_json wandb_completion_records includes "
            "multiple matching candidate W&B completions"
        )
    errors.extend(
        validate_wandb_adoption_source_audit_completion_record(
            matching_completion_records[0],
            benchmark=benchmark,
        )
    )
    return errors


def wandb_adoption_candidate_operator_handoff(candidate: dict[str, Any]) -> dict[str, Any]:
    required_human_fields = (
        candidate.get("required_human_fields")
        if isinstance(candidate.get("required_human_fields"), list)
        else []
    )
    if candidate.get("sync_ready") is not True:
        refresh_command = candidate.get("refresh_wandb_completion_command")
        steps: list[dict[str, Any]] = []
        if isinstance(refresh_command, str) and refresh_command.strip():
            evidence_paths = (
                [candidate.get("wandb_completion_json")]
                if isinstance(candidate.get("wandb_completion_json"), str)
                and candidate.get("wandb_completion_json")
                else []
            )
            steps.append(
                {
                    "step": "refresh_wandb_completion_verifier",
                    "command": refresh_command,
                    "expected_evidence_paths": evidence_paths,
                    "requires_external_action": False,
                    "requires_scope_confirmation": False,
                    "mutates_review_json": False,
                    "required": True,
                }
            )
        evidence_paths = [
            path
            for step in steps
            for path in step.get("expected_evidence_paths", [])
            if isinstance(path, str) and path.strip()
        ]
        return {
            "available": False,
            "blocked_reason": str(candidate.get("sync_command_blocked_reason") or ""),
            "requires_scope_confirmation": bool(candidate.get("scope_attestation_required")),
            "required_human_fields": required_human_fields,
            "pending_human_field_count": len(required_human_fields),
            "step_count": len(steps),
            "required_step_count": sum(1 for step in steps if step.get("required") is True),
            "command_count": sum(
                1
                for step in steps
                if isinstance(step.get("command"), str) and step.get("command")
            ),
            "external_action_step_count": sum(
                1 for step in steps if step.get("requires_external_action") is True
            ),
            "scope_confirmation_step_count": sum(
                1 for step in steps if step.get("requires_scope_confirmation") is True
            ),
            "review_mutation_step_count": sum(
                1 for step in steps if step.get("mutates_review_json") is True
            ),
            "evidence_path_count": len(evidence_paths),
            "expected_evidence_paths": evidence_paths,
            "steps": steps,
        }

    raw_steps = [
        {
            "step": "confirm_scope_attestation",
            "command": None,
            "expected_evidence_paths": [candidate.get("scope_attestation_template_json")],
            "requires_external_action": True,
            "requires_scope_confirmation": True,
            "mutates_review_json": False,
            "required": True,
        },
        {
            "step": "render_confirmed_attestation",
            "command": candidate.get("scope_attestation_render_command"),
            "expected_evidence_paths": [
                candidate.get("scope_attestation_template_json"),
                candidate.get("scope_attestation_render_report_json"),
                candidate.get("scope_attestation_render_markdown"),
            ],
            "requires_external_action": False,
            "requires_scope_confirmation": False,
            "mutates_review_json": False,
            "required": True,
        },
        {
            "step": "preflight_scope_attestation",
            "command": candidate.get("scope_attestation_preflight_command"),
            "expected_evidence_paths": [
                candidate.get("scope_attestation_preflight_report_json"),
            ],
            "requires_external_action": False,
            "requires_scope_confirmation": False,
            "mutates_review_json": False,
            "required": True,
        },
        {
            "step": "sync_paid_review_dry_run",
            "command": candidate.get("sync_dry_run_command"),
            "expected_evidence_paths": [candidate.get("sync_dry_run_report_json")],
            "requires_external_action": False,
            "requires_scope_confirmation": False,
            "mutates_review_json": False,
            "required": True,
        },
        {
            "step": "sync_paid_review_apply",
            "command": candidate.get("sync_apply_command") or candidate.get("sync_command"),
            "expected_evidence_paths": [candidate.get("target_review_json")],
            "requires_external_action": False,
            "requires_scope_confirmation": False,
            "mutates_review_json": True,
            "required": True,
        },
    ]
    steps: list[dict[str, Any]] = []
    for step in raw_steps:
        paths = [
            path
            for path in step.get("expected_evidence_paths", [])
            if isinstance(path, str) and path.strip()
        ]
        steps.append({**step, "expected_evidence_paths": paths})
    expected_evidence_paths: list[str] = []
    for step in steps:
        for path in step.get("expected_evidence_paths", []):
            if path not in expected_evidence_paths:
                expected_evidence_paths.append(path)
    return {
        "available": all(
            step.get("command") or step["step"] == "confirm_scope_attestation"
            for step in steps
        ),
        "blocked_reason": "",
        "requires_scope_confirmation": bool(candidate.get("scope_attestation_required")),
        "required_human_fields": required_human_fields,
        "pending_human_field_count": len(required_human_fields),
        "step_count": len(steps),
        "required_step_count": sum(1 for step in steps if step.get("required") is True),
        "command_count": sum(
            1 for step in steps if isinstance(step.get("command"), str) and step.get("command")
        ),
        "external_action_step_count": sum(
            1 for step in steps if step.get("requires_external_action") is True
        ),
        "scope_confirmation_step_count": sum(
            1 for step in steps if step.get("requires_scope_confirmation") is True
        ),
        "review_mutation_step_count": sum(
            1 for step in steps if step.get("mutates_review_json") is True
        ),
        "evidence_path_count": len(expected_evidence_paths),
        "expected_evidence_paths": expected_evidence_paths,
        "steps": steps,
    }


def wandb_adoption_operator_handoff_summary(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    aggregate = {
        "candidate_count": 0,
        "available_candidate_count": 0,
        "step_count": 0,
        "command_count": 0,
        "external_action_step_count": 0,
        "scope_confirmation_step_count": 0,
        "review_mutation_step_count": 0,
        "evidence_path_count": 0,
    }
    evidence_paths: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        handoff = wandb_adoption_candidate_operator_handoff(candidate)
        aggregate["candidate_count"] += 1
        if handoff.get("available") is True:
            aggregate["available_candidate_count"] += 1
        for field in (
            "step_count",
            "command_count",
            "external_action_step_count",
            "scope_confirmation_step_count",
            "review_mutation_step_count",
        ):
            value = handoff.get(field)
            if isinstance(value, int):
                aggregate[field] += value
        for path in handoff.get("expected_evidence_paths", []):
            if isinstance(path, str) and path.strip() and path not in evidence_paths:
                evidence_paths.append(path)
    aggregate["evidence_path_count"] = len(evidence_paths)
    aggregate["expected_evidence_paths"] = evidence_paths
    return aggregate


def validate_wandb_adoption_draft_evidence(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    current_gate = manifest.get("current_gate")
    if not isinstance(current_gate, dict):
        return errors
    draft = current_gate.get("wandb_adoption_draft")
    if not isinstance(draft, dict) or bool(draft.get("skipped")):
        return errors
    candidates = draft.get("candidates")
    candidate_count = draft.get("candidate_count")
    if (
        not isinstance(draft.get("path"), str)
        and not candidates
        and not candidate_count
    ):
        return errors

    records = file_records_by_source(manifest)
    draft_record = validate_file_role(
        errors=errors,
        records=records,
        path_value=draft.get("path"),
        role="wandb_adoption_draft",
        label="W&B adoption draft JSON",
    )
    markdown_path = draft.get("markdown_path")
    markdown_record: dict[str, Any] | None = None
    markdown_required = bool(candidates) or (
        isinstance(candidate_count, int) and candidate_count > 0
    )
    if markdown_required and (
        not isinstance(markdown_path, str) or not markdown_path.strip()
    ):
        errors.append(
            "manifest current_gate wandb_adoption_draft markdown_path is "
            "required when adoption candidates are present"
        )
    elif isinstance(markdown_path, str) and markdown_path.strip():
        markdown_record = validate_file_role(
            errors=errors,
            records=records,
            path_value=markdown_path,
            role="wandb_adoption_draft_markdown",
            label="W&B adoption draft Markdown",
        )

    payload: dict[str, Any] = {}
    if isinstance(draft_record, dict):
        bundle_path = draft_record.get("bundle_path")
        if isinstance(bundle_path, str) and bundle_path:
            try:
                payload = read_json_object(bundle_dir / bundle_path)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"W&B adoption draft JSON is not readable: {exc}")

    if payload:
        payload_path = payload.get("path")
        if isinstance(payload_path, str) and payload_path.strip():
            if source_path_key(draft.get("path")) != source_path_key(payload_path):
                errors.append(
                    "manifest current_gate wandb_adoption_draft path does not "
                    "match bundled draft JSON"
                )
        else:
            errors.append("W&B adoption draft JSON path is missing")
        payload_markdown_path = payload.get("markdown_path")
        if isinstance(payload_markdown_path, str) and payload_markdown_path.strip():
            if source_path_key(markdown_path) != source_path_key(payload_markdown_path):
                errors.append(
                    "manifest current_gate wandb_adoption_draft markdown_path "
                    "does not match bundled draft JSON"
                )
        elif markdown_required:
            errors.append("W&B adoption draft JSON markdown_path is missing")
        if isinstance(markdown_record, dict):
            bundle_path = markdown_record.get("bundle_path")
            if isinstance(bundle_path, str) and bundle_path:
                try:
                    markdown_text = (bundle_dir / bundle_path).read_text(
                        encoding="utf-8"
                    )
                except OSError as exc:
                    errors.append(f"W&B adoption draft Markdown is not readable: {exc}")
                    markdown_text = ""
                if markdown_text:
                    for snippet in expected_wandb_adoption_draft_markdown_snippets(
                        payload
                    ):
                        if snippet not in markdown_text:
                            errors.append(
                                "W&B adoption draft Markdown missing operator "
                                "handoff content from bundled draft JSON: "
                                f"{snippet}"
                            )
        source_audit = payload.get("source_audit_json")
        source_audit_record: dict[str, Any] | None = None
        source_audit_payload: dict[str, Any] | None = None
        source_audit_sha256 = payload.get("source_audit_sha256")
        if isinstance(source_audit, str) and source_audit.strip():
            if source_path_key(draft.get("source_audit_json")) != source_path_key(source_audit):
                errors.append(
                    "manifest current_gate wandb_adoption_draft source_audit_json "
                    "does not match bundled draft JSON"
                )
            if draft.get("source_audit_sha256") != source_audit_sha256:
                errors.append(
                    "manifest current_gate wandb_adoption_draft source_audit_sha256 "
                    "does not match bundled draft JSON"
                )
        if isinstance(source_audit, str) and source_audit.strip():
            source_audit_record = validate_file_role(
                errors=errors,
                records=records,
                path_value=source_audit,
                role="wandb_adoption_draft:source_audit_json",
                label="W&B adoption draft source_audit_json",
            )
            if not isinstance(source_audit_sha256, str) or not re.fullmatch(
                r"[0-9a-f]{64}",
                source_audit_sha256 or "",
            ):
                errors.append(
                    "W&B adoption draft source_audit_sha256 must be a "
                    "64-character lowercase hex digest"
                )
            elif isinstance(source_audit_record, dict) and source_audit_sha256 != source_audit_record.get("sha256"):
                errors.append(
                    "W&B adoption draft source_audit_sha256 does not match "
                    "bundled source_audit_json"
                )
            if isinstance(source_audit_record, dict):
                bundle_path = source_audit_record.get("bundle_path")
                if isinstance(bundle_path, str) and bundle_path:
                    try:
                        source_audit_payload = read_json_object(bundle_dir / bundle_path)
                    except (OSError, json.JSONDecodeError, ValueError) as exc:
                        errors.append(f"W&B adoption draft source_audit_json is not readable: {exc}")

        payload_candidates = payload.get("candidates")
        if isinstance(payload_candidates, list):
            summary_candidates = draft.get("candidates")
            if not isinstance(summary_candidates, list):
                summary_candidates = []
            draft_required_human_fields = (
                payload.get("required_human_fields")
                if isinstance(payload.get("required_human_fields"), list)
                else []
            )
            pending_scope_confirmation_candidate_count = 0
            pending_human_field_count = 0
            pending_human_fields: list[str] = []
            for candidate in payload_candidates:
                if not isinstance(candidate, dict) or not bool(
                    candidate.get("scope_attestation_required")
                ):
                    continue
                pending_scope_confirmation_candidate_count += 1
                candidate_required_human_fields = (
                    candidate.get("required_human_fields")
                    if isinstance(candidate.get("required_human_fields"), list)
                    else []
                )
                fields = (
                    candidate_required_human_fields
                    if candidate_required_human_fields
                    else draft_required_human_fields
                )
                normalized_fields = [
                    field
                    for field in fields
                    if isinstance(field, str) and field.strip()
                ]
                pending_human_field_count += len(normalized_fields)
                for field in normalized_fields:
                    if field not in pending_human_fields:
                        pending_human_fields.append(field)
            if pending_scope_confirmation_candidate_count:
                if draft_required_human_fields != pending_human_fields:
                    errors.append(
                        "W&B adoption draft JSON required_human_fields does "
                        "not match candidate required_human_fields"
                    )
                summary_required_human_fields = (
                    draft.get("required_human_fields")
                    if isinstance(draft.get("required_human_fields"), list)
                    else []
                )
                if summary_required_human_fields != pending_human_fields:
                    errors.append(
                        "manifest current_gate wandb_adoption_draft "
                        "required_human_fields does not match bundled draft JSON"
                    )
            if (
                draft.get("pending_scope_confirmation_candidate_count")
                != pending_scope_confirmation_candidate_count
            ):
                errors.append(
                    "manifest current_gate wandb_adoption_draft "
                    "pending_scope_confirmation_candidate_count does not match "
                    "bundled draft JSON"
                )
            if draft.get("pending_human_field_count") != pending_human_field_count:
                errors.append(
                    "manifest current_gate wandb_adoption_draft "
                    "pending_human_field_count does not match bundled draft JSON"
                )
            if draft.get("pending_human_fields") != pending_human_fields:
                errors.append(
                    "manifest current_gate wandb_adoption_draft "
                    "pending_human_fields does not match bundled draft JSON"
                )
            candidate_dicts = [
                candidate for candidate in payload_candidates if isinstance(candidate, dict)
            ]
            expected_operator_handoff = wandb_adoption_operator_handoff_summary(
                candidate_dicts
            )
            if payload.get("operator_handoff") != expected_operator_handoff:
                errors.append(
                    "W&B adoption draft JSON operator_handoff does not match candidates"
                )
            if draft.get("operator_handoff") != expected_operator_handoff:
                errors.append(
                    "manifest current_gate wandb_adoption_draft operator_handoff "
                    "does not match bundled draft JSON"
                )
            for field in (
                "operator_handoff_candidate_count",
                "operator_handoff_available_candidate_count",
                "operator_handoff_step_count",
                "operator_handoff_command_count",
                "operator_handoff_external_action_step_count",
                "operator_handoff_scope_confirmation_step_count",
                "operator_handoff_review_mutation_step_count",
                "operator_handoff_evidence_path_count",
            ):
                expected_key = field.removeprefix("operator_handoff_")
                if expected_key == "available_candidate_count":
                    expected_value = expected_operator_handoff.get(
                        "available_candidate_count"
                    )
                else:
                    expected_value = expected_operator_handoff.get(expected_key)
                if draft.get(field) != expected_value:
                    errors.append(
                        "manifest current_gate wandb_adoption_draft "
                        f"{field} does not match bundled draft JSON"
                    )
            for index, candidate in enumerate(payload_candidates, start=1):
                if not isinstance(candidate, dict):
                    continue
                benchmark = str(candidate.get("benchmark") or f"candidate_{index}")
                summary_candidate = next(
                    (
                        row
                        for row in summary_candidates
                        if isinstance(row, dict)
                        and row.get("benchmark") == candidate.get("benchmark")
                        and row.get("wandb_run_id") == candidate.get("wandb_run_id")
                        and source_path_key(row.get("wandb_completion_json"))
                        == source_path_key(candidate.get("wandb_completion_json"))
                    ),
                    None,
                )
                if isinstance(summary_candidate, dict) and isinstance(source_audit, str) and source_audit.strip():
                    if source_path_key(summary_candidate.get("source_audit_json")) != source_path_key(
                        candidate.get("source_audit_json")
                    ):
                        errors.append(
                            "manifest current_gate wandb_adoption_draft candidate "
                            f"{benchmark} source_audit_json does not match bundled draft JSON"
                        )
                    if summary_candidate.get("source_audit_sha256") != candidate.get("source_audit_sha256"):
                        errors.append(
                            "manifest current_gate wandb_adoption_draft candidate "
                            f"{benchmark} source_audit_sha256 does not match bundled draft JSON"
                        )
                if isinstance(summary_candidate, dict):
                    summary_required_human_fields = (
                        summary_candidate.get("required_human_fields")
                        if isinstance(summary_candidate.get("required_human_fields"), list)
                        else []
                    )
                    candidate_required_human_fields = (
                        candidate.get("required_human_fields")
                        if isinstance(candidate.get("required_human_fields"), list)
                        else []
                    )
                    if summary_required_human_fields != candidate_required_human_fields:
                        errors.append(
                            "manifest current_gate wandb_adoption_draft candidate "
                            f"{benchmark} required_human_fields does not match bundled draft JSON"
                        )
                    expected_pending_scope = bool(candidate.get("scope_attestation_required"))
                    if summary_candidate.get("pending_scope_confirmation") is not expected_pending_scope:
                        errors.append(
                            "manifest current_gate wandb_adoption_draft candidate "
                            f"{benchmark} pending_scope_confirmation does not match bundled draft JSON"
                        )
                    expected_handoff = wandb_adoption_candidate_operator_handoff(
                        candidate
                    )
                    if candidate.get("operator_handoff") != expected_handoff:
                        errors.append(
                            "W&B adoption draft candidate "
                            f"{benchmark} operator_handoff does not match candidate"
                        )
                    if summary_candidate.get("operator_handoff") != expected_handoff:
                        errors.append(
                            "manifest current_gate wandb_adoption_draft candidate "
                            f"{benchmark} operator_handoff does not match bundled draft JSON"
                        )
                completion_role = f"wandb_adoption_draft:candidate:{benchmark}:wandb_completion_json"
                validate_file_role(
                    errors=errors,
                    records=records,
                    path_value=candidate.get("wandb_completion_json"),
                    role=completion_role,
                    label=f"W&B adoption draft candidate {benchmark} completion JSON",
                )
                target_review = candidate.get("target_review_json")
                target_key = source_path_key(target_review)
                if target_key and target_key in records:
                    validate_file_role(
                        errors=errors,
                        records=records,
                        path_value=target_review,
                        role=f"wandb_adoption_draft:candidate:{benchmark}:target_review_json",
                        label=f"W&B adoption draft candidate {benchmark} target review JSON",
                    )
                scope_attestation_source: dict[str, Any] | None = None
                template_path = candidate.get("scope_attestation_template_json")
                if isinstance(source_audit, str) and source_audit.strip():
                    if source_path_key(candidate.get("source_audit_json")) != source_path_key(source_audit):
                        errors.append(
                            "W&B adoption draft candidate "
                            f"{benchmark} source_audit_json does not match draft"
                        )
                    candidate_source_audit_sha256 = candidate.get("source_audit_sha256")
                    if not isinstance(candidate_source_audit_sha256, str) or not re.fullmatch(
                        r"[0-9a-f]{64}",
                        candidate_source_audit_sha256 or "",
                    ):
                        errors.append(
                            "W&B adoption draft candidate "
                            f"{benchmark} source_audit_sha256 must be a "
                            "64-character lowercase hex digest"
                        )
                    elif (
                        isinstance(source_audit_sha256, str)
                        and re.fullmatch(r"[0-9a-f]{64}", source_audit_sha256)
                        and candidate_source_audit_sha256 != source_audit_sha256
                    ):
                        errors.append(
                            "W&B adoption draft candidate "
                            f"{benchmark} source_audit_sha256 does not match draft"
                        )
                    if source_audit_payload is not None:
                        errors.extend(
                            validate_wandb_adoption_candidate_source_audit(
                                candidate=candidate,
                                benchmark=benchmark,
                                source_audit_payload=source_audit_payload,
                            )
                        )
                if isinstance(template_path, str) and template_path.strip():
                    template_record = validate_file_role(
                        errors=errors,
                        records=records,
                        path_value=template_path,
                        role=f"wandb_adoption_draft:candidate:{benchmark}:scope_attestation_template_json",
                        label=f"W&B adoption draft candidate {benchmark} scope attestation template",
                    )
                    if isinstance(template_record, dict):
                        bundle_path = template_record.get("bundle_path")
                        if isinstance(bundle_path, str) and bundle_path:
                            try:
                                template = read_json_object(bundle_dir / bundle_path)
                            except (OSError, json.JSONDecodeError, ValueError) as exc:
                                errors.append(
                                    "W&B adoption draft scope attestation template "
                                    f"for {benchmark} is not readable: {exc}"
                                )
                                template = {}
                            if template:
                                scope_attestation_source = template
                                if template.get("schema_version") != SCOPE_ATTESTATION_SCHEMA_VERSION:
                                    errors.append(
                                        "W&B adoption draft scope attestation template "
                                        f"for {benchmark} schema_version must be "
                                        f"{SCOPE_ATTESTATION_SCHEMA_VERSION}"
                                    )
                                if template.get("benchmark") != candidate.get("benchmark"):
                                    errors.append(
                                        "W&B adoption draft scope attestation template "
                                        f"for {benchmark} benchmark does not match candidate"
                                    )
                                if template.get("entity") != candidate.get("wandb_entity"):
                                    errors.append(
                                        "W&B adoption draft scope attestation template "
                                        f"for {benchmark} entity does not match candidate"
                                    )
                                if template.get("project") != candidate.get("wandb_project"):
                                    errors.append(
                                        "W&B adoption draft scope attestation template "
                                        f"for {benchmark} project does not match candidate"
                                    )
                                if template.get("run_id") != candidate.get("wandb_run_id"):
                                    errors.append(
                                        "W&B adoption draft scope attestation template "
                                        f"for {benchmark} run_id does not match candidate"
                                    )
                                if source_path_key(template.get("completion_path")) != source_path_key(
                                    candidate.get("wandb_completion_json")
                                ):
                                    errors.append(
                                        "W&B adoption draft scope attestation template "
                                        f"for {benchmark} completion_path does not match candidate"
                                    )
                                if source_path_key(template.get("review_path")) != source_path_key(
                                    candidate.get("target_review_json")
                                ):
                                    errors.append(
                                        "W&B adoption draft scope attestation template "
                                        f"for {benchmark} review_path does not match candidate"
                                    )
                                if isinstance(source_audit, str) and source_audit.strip():
                                    if source_path_key(template.get("source_audit_json")) != source_path_key(source_audit):
                                        errors.append(
                                            "W&B adoption draft scope attestation template "
                                            f"for {benchmark} source_audit_json does not match draft"
                                        )
                                    template_source_audit_sha256 = template.get("source_audit_sha256")
                                    if not isinstance(template_source_audit_sha256, str) or not re.fullmatch(
                                        r"[0-9a-f]{64}",
                                        template_source_audit_sha256 or "",
                                    ):
                                        errors.append(
                                            "W&B adoption draft scope attestation template "
                                            f"for {benchmark} source_audit_sha256 must be a "
                                            "64-character lowercase hex digest"
                                        )
                                    elif (
                                        isinstance(source_audit_sha256, str)
                                        and re.fullmatch(r"[0-9a-f]{64}", source_audit_sha256)
                                        and template_source_audit_sha256 != source_audit_sha256
                                    ):
                                        errors.append(
                                            "W&B adoption draft scope attestation template "
                                            f"for {benchmark} source_audit_sha256 does not match draft"
                                        )
                errors.extend(
                    validate_wandb_adoption_scope_render_command(
                        candidate=candidate,
                        benchmark=benchmark,
                    )
                )
                render_report_payload: dict[str, Any] | None = None
                render_report_path = candidate.get(
                    "scope_attestation_render_report_json"
                )
                render_report_key = source_path_key(render_report_path)
                if render_report_key and render_report_key in records:
                    render_report_record = validate_file_role(
                        errors=errors,
                        records=records,
                        path_value=render_report_path,
                        role=(
                            "wandb_adoption_draft:candidate:"
                            f"{benchmark}:scope_attestation_render_report_json"
                        ),
                        label=(
                            f"W&B adoption draft candidate {benchmark} "
                            "scope render report"
                        ),
                    )
                    if isinstance(render_report_record, dict):
                        bundle_path = render_report_record.get("bundle_path")
                        if isinstance(bundle_path, str) and bundle_path:
                            try:
                                report_payload = read_json_object(bundle_dir / bundle_path)
                            except (OSError, json.JSONDecodeError, ValueError) as exc:
                                errors.append(
                                    "W&B adoption draft scope render report "
                                    f"for {benchmark} is not readable: {exc}"
                                )
                                report_payload = {}
                            if report_payload:
                                render_report_payload = report_payload
                                errors.extend(
                                    validate_wandb_adoption_scope_render_report(
                                        report_payload,
                                        candidate=candidate,
                                        benchmark=benchmark,
                                        scope_attestation_source=scope_attestation_source,
                                        records=records,
                                    )
                                )
                render_markdown_path = candidate.get(
                    "scope_attestation_render_markdown"
                )
                render_markdown_key = source_path_key(render_markdown_path)
                if render_markdown_key and render_markdown_key in records:
                    render_markdown_record = validate_file_role(
                        errors=errors,
                        records=records,
                        path_value=render_markdown_path,
                        role=(
                            "wandb_adoption_draft:candidate:"
                            f"{benchmark}:scope_attestation_render_markdown"
                        ),
                        label=(
                            f"W&B adoption draft candidate {benchmark} "
                            "scope render Markdown"
                        ),
                    )
                    if render_report_payload is None:
                        errors.append(
                            "W&B adoption draft candidate "
                            f"{benchmark} scope render Markdown requires bundled "
                            "scope render report JSON"
                        )
                    if isinstance(render_markdown_record, dict):
                        bundle_path = render_markdown_record.get("bundle_path")
                        if isinstance(bundle_path, str) and bundle_path:
                            try:
                                markdown_text = (bundle_dir / bundle_path).read_text(
                                    encoding="utf-8"
                                )
                            except OSError as exc:
                                errors.append(
                                    "W&B adoption draft scope render Markdown "
                                    f"for {benchmark} is not readable: {exc}"
                                )
                                markdown_text = ""
                            if markdown_text and isinstance(render_report_payload, dict):
                                errors.extend(
                                    validate_wandb_adoption_scope_render_markdown(
                                        markdown_text,
                                        report_payload=render_report_payload,
                                        benchmark=benchmark,
                                    )
                                )
                preflight_report_path = candidate.get(
                    "scope_attestation_preflight_report_json"
                )
                preflight_report_key = source_path_key(preflight_report_path)
                if preflight_report_key and preflight_report_key in records:
                    preflight_report_record = validate_file_role(
                        errors=errors,
                        records=records,
                        path_value=preflight_report_path,
                        role=(
                            "wandb_adoption_draft:candidate:"
                            f"{benchmark}:scope_attestation_preflight_report_json"
                        ),
                        label=(
                            f"W&B adoption draft candidate {benchmark} "
                            "scope preflight report"
                        ),
                    )
                    if isinstance(preflight_report_record, dict):
                        bundle_path = preflight_report_record.get("bundle_path")
                        if isinstance(bundle_path, str) and bundle_path:
                            try:
                                report_payload = read_json_object(bundle_dir / bundle_path)
                            except (OSError, json.JSONDecodeError, ValueError) as exc:
                                errors.append(
                                    "W&B adoption draft scope preflight report "
                                    f"for {benchmark} is not readable: {exc}"
                                )
                                report_payload = {}
                            if report_payload:
                                errors.extend(
                                    validate_wandb_adoption_scope_preflight_report(
                                        report_payload,
                                        candidate=candidate,
                                        benchmark=benchmark,
                                        scope_attestation_source=scope_attestation_source,
                                        records=records,
                                    )
                                )
                dry_run_report_path = candidate.get("sync_dry_run_report_json")
                dry_run_report_key = source_path_key(dry_run_report_path)
                if dry_run_report_key and dry_run_report_key in records:
                    dry_run_report_record = validate_file_role(
                        errors=errors,
                        records=records,
                        path_value=dry_run_report_path,
                        role=f"wandb_adoption_draft:candidate:{benchmark}:sync_dry_run_report_json",
                        label=f"W&B adoption draft candidate {benchmark} sync dry-run report",
                    )
                    if isinstance(dry_run_report_record, dict):
                        bundle_path = dry_run_report_record.get("bundle_path")
                        if isinstance(bundle_path, str) and bundle_path:
                            try:
                                report_payload = read_json_object(bundle_dir / bundle_path)
                            except (OSError, json.JSONDecodeError, ValueError) as exc:
                                errors.append(
                                    "W&B adoption draft sync dry-run report "
                                    f"for {benchmark} is not readable: {exc}"
                                )
                                report_payload = {}
                            if report_payload:
                                errors.extend(
                                    validate_wandb_adoption_sync_dry_run_report(
                                        report_payload,
                                        candidate=candidate,
                                        benchmark=benchmark,
                                        scope_attestation_source=scope_attestation_source,
                                        records=records,
                                    )
                                )
    elif isinstance(candidates, list):
        for index, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, dict):
                continue
            benchmark = str(candidate.get("benchmark") or f"candidate_{index}")
            completion_role = f"wandb_adoption_draft:candidate:{benchmark}:wandb_completion_json"
            validate_file_role(
                errors=errors,
                records=records,
                path_value=candidate.get("wandb_completion_json"),
                role=completion_role,
                label=f"W&B adoption draft candidate {benchmark} completion JSON",
            )

    return errors


def validate_existing_results_relog_dry_run_plan_payload(
    payload: dict[str, Any],
    *,
    record: dict[str, Any],
    label: str,
) -> list[str]:
    errors: list[str] = []
    benchmark = record.get("benchmark")
    if payload.get("schema_version") != 1:
        errors.append(f"{label} schema_version must be 1")
    if payload.get("will_write_wandb") is not False:
        errors.append(f"{label} will_write_wandb must be false")
    approval = payload.get("external_action_approval")
    if not isinstance(approval, dict):
        errors.append(f"{label} external_action_approval must be an object")
        approval = {}
    if approval.get("required_before_wandb_write") is not True:
        errors.append(f"{label} external_action_approval.required_before_wandb_write must be true")
    if approval.get("required_report_option") != "--external-action-approval-report-json":
        errors.append(
            f"{label} external_action_approval.required_report_option must be --external-action-approval-report-json"
        )
    if approval.get("required_source_packet_option") != "--external-action-approval-source-packet-json":
        errors.append(
            f"{label} external_action_approval.required_source_packet_option must be --external-action-approval-source-packet-json"
        )
    if approval.get("required_report_status") != "approved":
        errors.append(f"{label} external_action_approval.required_report_status must be approved")
    if approval.get("required_source_bound") is not True:
        errors.append(f"{label} external_action_approval.required_source_bound must be true")
    if approval.get("required_source_packet_sha256_match") is not True:
        errors.append(
            f"{label} external_action_approval.required_source_packet_sha256_match must be true"
        )
    required_requirements = approval.get("required_requirements")
    if not isinstance(required_requirements, list) or not {
        "wandb_access",
        "wandb_write",
    }.issubset(set(required_requirements)):
        errors.append(
            f"{label} external_action_approval.required_requirements must include wandb_access and wandb_write"
        )
    if approval.get("target_entity") != payload.get("entity"):
        errors.append(f"{label} external_action_approval.target_entity must match plan entity")
    if approval.get("target_project") != payload.get("project"):
        errors.append(f"{label} external_action_approval.target_project must match plan project")
    if payload.get("benchmark") != benchmark:
        errors.append(f"{label} benchmark does not match audit record")
    if not isinstance(payload.get("entity"), str) or not payload.get("entity"):
        errors.append(f"{label} entity is missing")
    if not isinstance(payload.get("project"), str) or not payload.get("project"):
        errors.append(f"{label} project is missing")

    source = payload.get("source")
    if not isinstance(source, dict):
        errors.append(f"{label} source must be an object")
        source = {}
    source_sha256 = source.get("source_sha256")
    if not isinstance(source_sha256, dict):
        errors.append(f"{label} source.source_sha256 must be an object")
        source_sha256 = {}

    def validate_source_sha256(source_key: str, source_path_value: Any, *, required: bool = True) -> None:
        digest = source_sha256.get(source_key)
        if digest is None:
            if required:
                errors.append(f"{label} source.source_sha256.{source_key} is missing")
            return
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            errors.append(f"{label} source.source_sha256.{source_key} must be 64 lowercase hex")
            return
        if not isinstance(source_path_value, str) or not source_path_value.strip():
            errors.append(f"{label} source path for {source_key} is missing")
            return
        source_path = repo_path(source_path_value)
        if source_path.exists():
            actual = sha256_file(source_path)
            if digest != actual:
                errors.append(f"{label} source.source_sha256.{source_key} does not match current source file")

    if benchmark == "agentic_math":
        if source_path_key(source.get("results_dir")) != source_path_key(record.get("result_dir")):
            errors.append(f"{label} source.results_dir does not match audit record")
        if source_path_key(source.get("summary_json")) != source_path_key(record.get("summary_path")):
            errors.append(f"{label} source.summary_json does not match audit record")
        if source_path_key(source.get("results_jsonl")) != source_path_key(record.get("results_path")):
            errors.append(f"{label} source.results_jsonl does not match audit record")
        validate_source_sha256("summary_json", source.get("summary_json"))
        validate_source_sha256("results_jsonl", source.get("results_jsonl"))
    elif benchmark == "agentic_swe":
        official_summary = source_path_key(record.get("official_summary_path"))
        official_eval_dir = str(Path(official_summary).parent) if official_summary else ""
        if source_path_key(source.get("official_eval_dir")) != official_eval_dir:
            errors.append(f"{label} source.official_eval_dir does not match audit record")
        if source_path_key(source.get("summary_json")) != official_summary:
            errors.append(f"{label} source.summary_json does not match audit record")
        if source_path_key(source.get("patch_path")) != source_path_key(record.get("patches_path")):
            errors.append(f"{label} source.patch_path does not match audit record")
        validate_source_sha256("summary_json", source.get("summary_json"))
        validate_source_sha256("patch_path", source.get("patch_path"))
        eval_results_path = source.get("eval_results_json")
        eval_results_required = (
            isinstance(eval_results_path, str)
            and bool(eval_results_path.strip())
            and repo_path(eval_results_path).exists()
        )
        validate_source_sha256(
            "eval_results_json",
            eval_results_path,
            required=eval_results_required,
        )

    if payload.get("ok") is True:
        config = payload.get("config")
        if not isinstance(config, dict):
            errors.append(f"{label} config must be an object for ok=true")
            config = {}
        relog_config = config.get("relog") if isinstance(config, dict) else {}
        if not isinstance(relog_config, dict):
            errors.append(f"{label} config.relog must be an object for ok=true")
            relog_config = {}
        if relog_config.get("source_sha256") != source_sha256:
            errors.append(f"{label} config.relog.source_sha256 must match source.source_sha256")
        would_log = payload.get("would_log")
        if not isinstance(would_log, dict):
            errors.append(f"{label} would_log must be an object for ok=true")
            would_log = {}
        artifact = would_log.get("artifact") if isinstance(would_log, dict) else {}
        aliases = artifact.get("aliases") if isinstance(artifact, dict) else None
        if not isinstance(aliases, list) or "production" not in aliases:
            errors.append(f"{label} artifact aliases must include production")
        verifier = payload.get("post_log_verifier_command_template")
        if not isinstance(verifier, str) or "verify_taiwan_wandb_completion.py" not in verifier:
            errors.append(f"{label} post_log_verifier_command_template is missing verifier")
        elif f"--benchmark {benchmark}" not in verifier:
            errors.append(f"{label} verifier command benchmark does not match plan")
        else:
            expected_total = "100" if benchmark == "agentic_math" else "80"
            if command_option_value(verifier, "--expected-total") != expected_total:
                errors.append(f"{label} verifier command missing --expected-total {expected_total}")
            if "--require-nemoclaw-session-audit" not in verifier:
                errors.append(f"{label} verifier command missing --require-nemoclaw-session-audit")
            if isinstance(source_sha256, dict):
                for source_key, digest in sorted(source_sha256.items()):
                    expected_arg = f"--expected-run-config relog.source_sha256.{source_key}={digest}"
                    if expected_arg not in verifier:
                        errors.append(
                            f"{label} verifier command missing relog source sha expected-run-config "
                            f"for {source_key}"
                        )
        tables = would_log.get("tables") if isinstance(would_log, dict) else {}
        if isinstance(tables, dict):
            output_table = (
                "agentic_math_output_table"
                if benchmark == "agentic_math"
                else "agentic_swe_output_table"
            )
            row_count = record.get("row_count")
            if isinstance(row_count, int) and tables.get(output_table) != row_count:
                errors.append(f"{label} output table row count does not match audit record")
        else:
            errors.append(f"{label} would_log.tables must be an object")
    else:
        if payload.get("status") != "validation_failed":
            errors.append(f"{label} non-ok plan status must be validation_failed")
        plan_errors = payload.get("errors")
        if not isinstance(plan_errors, list) or not plan_errors:
            errors.append(f"{label} validation_failed plan must include errors")
    return errors


def command_option_value(command: str, option: str) -> str:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return ""
    for index, token in enumerate(tokens):
        if token == option and index + 1 < len(tokens):
            return tokens[index + 1]
        prefix = option + "="
        if token.startswith(prefix):
            return token[len(prefix) :]
    return ""


def validate_existing_results_relog_command_contract(
    record: dict[str, Any],
    *,
    label: str,
) -> list[str]:
    errors: list[str] = []
    plan_path = record.get("relog_dry_run_plan_json")
    relog_command = record.get("relog_command")
    dry_run_command = record.get("relog_dry_run_command")
    if isinstance(dry_run_command, str) and "--validated-dry-run-plan-json" in dry_run_command:
        errors.append(f"{label} dry-run command must not require an existing validated plan")
    if (
        isinstance(dry_run_command, str)
        and "--external-action-approval-source-packet-json" in dry_run_command
    ):
        errors.append(
            f"{label} dry-run command must not require an external action approval source packet"
        )
    if isinstance(dry_run_command, str) and "--external-action-approval-report-json" in dry_run_command:
        errors.append(f"{label} dry-run command must not require an external action approval report")
    if not isinstance(relog_command, str) or not relog_command.strip():
        return errors
    if "log_agentic_math_results_to_wandb.py" not in relog_command and (
        "log_agentic_swe_results_to_wandb.py" not in relog_command
    ):
        return errors
    if not isinstance(plan_path, str) or not plan_path.strip():
        errors.append(f"{label} relog command requires relog_dry_run_plan_json")
    validated_plan = command_option_value(relog_command, "--validated-dry-run-plan-json")
    if not validated_plan:
        errors.append(f"{label} relog command must include --validated-dry-run-plan-json")
    elif source_path_key(validated_plan) != source_path_key(plan_path):
        errors.append(
            f"{label} relog command --validated-dry-run-plan-json does not match relog_dry_run_plan_json"
        )
    approval_report = command_option_value(relog_command, "--external-action-approval-report-json")
    if not approval_report:
        errors.append(f"{label} relog command must include --external-action-approval-report-json")
    approval_source_packet = command_option_value(
        relog_command,
        "--external-action-approval-source-packet-json",
    )
    if not approval_source_packet:
        errors.append(
            f"{label} relog command must include --external-action-approval-source-packet-json"
        )
    return errors


def validate_existing_results_relog_dry_run_plan_evidence(
    *,
    bundle_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    current_gate = manifest.get("current_gate")
    if not isinstance(current_gate, dict):
        return errors
    existing = current_gate.get("existing_results_formalization")
    if not isinstance(existing, dict):
        return errors
    records_by_source = file_records_by_source(manifest)
    record_groups = (
        existing.get("formalized_records"),
        existing.get("unformalized_complete_records"),
        existing.get("partial_records"),
    )
    for group in record_groups:
        if not isinstance(group, list):
            continue
        for index, record in enumerate(group, start=1):
            if not isinstance(record, dict):
                continue
            benchmark = str(record.get("benchmark") or f"record_{index}")
            model_slug = str(record.get("model_slug") or f"model_{index}")
            label = f"existing results relog dry-run plan {benchmark}/{model_slug}"
            errors.extend(
                validate_existing_results_relog_command_contract(
                    record,
                    label=label,
                )
            )
            plan_path = record.get("relog_dry_run_plan_json")
            key = source_path_key(plan_path)
            if not key or key not in records_by_source:
                continue
            role = (
                f"existing_results_audit:record:{benchmark}:{model_slug}:"
                "relog_dry_run_plan_json"
            )
            plan_record = validate_file_role(
                errors=errors,
                records=records_by_source,
                path_value=plan_path,
                role=role,
                label=label,
            )
            if not isinstance(plan_record, dict):
                continue
            bundle_path = plan_record.get("bundle_path")
            if not isinstance(bundle_path, str) or not bundle_path:
                errors.append(f"{label} missing bundle_path")
                continue
            try:
                payload = read_json_object(bundle_dir / bundle_path)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"{label} is not readable: {exc}")
                continue
            errors.extend(
                validate_existing_results_relog_dry_run_plan_payload(
                    payload,
                    record=record,
                    label=label,
                )
            )
    return errors


def verify_bundle(
    *,
    manifest_path: Path,
    require_ready: bool,
) -> dict[str, Any]:
    manifest = read_manifest(manifest_path)
    bundle_dir = manifest_path.parent
    files = manifest.get("files")
    file_records = files if isinstance(files, list) else []
    file_results: list[dict[str, Any]] = []
    errors: list[str] = []
    if not isinstance(files, list):
        errors.append("manifest files is not a list")
    errors.extend(validate_manifest_schema(manifest))
    errors.extend(validate_current_gate(manifest))
    errors.extend(validate_production_readiness_report_evidence(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_operator_plan(manifest))
    errors.extend(
        validate_release_gate_pointer_summary(
            bundle_dir=bundle_dir,
            manifest=manifest,
        )
    )
    errors.extend(
        validate_release_gate_pointer_proof(
            bundle_dir=bundle_dir,
            manifest=manifest,
        )
    )
    errors.extend(validate_operator_plan_payload(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_external_action_approval_packet_payload(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_operator_next_steps_semantics(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_summary_markdown(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_operator_plan_command_policy(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_operator_plan_command_scripts(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_current_gate_remediation_command_scripts(manifest=manifest))
    errors.extend(validate_existing_results_relog_command_scripts(manifest=manifest))
    errors.extend(validate_agentic_runner_script_evidence(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_nemoclaw_canary_readiness_script_source(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_nemoclaw_adoption_script_source(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_wandb_adoption_draft_evidence(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_wandb_adoption_unconfirmed_checks_evidence(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_weave_agents_adoption_validation_failures_evidence(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_existing_results_relog_dry_run_plan_evidence(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_nemoclaw_installer_review_evidence(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_nemoclaw_setup_acceptance_evidence(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_nemoclaw_adoption_payload_matches_current_gate(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_nemoclaw_setup_installed_evidence(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_nemoclaw_swebench_non_adoption_evidence(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_nemoclaw_agentic_config_yaml_evidence(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_nemoclaw_operator_docs_evidence(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_nemoclaw_post_install_verification_evidence(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_wandb_completion_contract_consistency(manifest))
    errors.extend(validate_wandb_completion_proof_evidence(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_paid_review_record_review_source_evidence(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_paid_review_run_eval_preflight_evidence(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_paid_review_wandb_completion_entry_evidence(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_paid_review_package_accounting_evidence(manifest))
    errors.extend(validate_paid_review_scope_attestation_evidence(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_paid_review_pre_run_budget_evidence(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_paid_review_external_action_approval_evidence(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_weave_agents_completion_proof_evidence(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_weave_agents_completion_review_source_evidence(bundle_dir=bundle_dir, manifest=manifest))
    errors.extend(validate_weave_content_canary_gate_evidence(bundle_dir=bundle_dir, manifest=manifest))
    for index, record in enumerate(file_records, start=1):
        if not isinstance(record, dict):
            errors.append(f"file record {index} is not an object")
            continue
        result = verify_file_record(bundle_dir, record)
        file_results.append(result)
        errors.extend(
            f"{result['source_path']}: {error}"
            for error in result["errors"]
        )

    integrity_ok = not errors
    readiness_ok = bool(manifest.get("readiness_ok"))
    ok = integrity_ok and (readiness_ok or not require_ready)
    if require_ready and not readiness_ok:
        errors.append("readiness_ok is false")
    status = "passed" if ok else "failed"
    return {
        "schema_version": 1,
        "ok": ok,
        "status": status,
        "integrity_ok": integrity_ok,
        "readiness_ok": readiness_ok,
        "require_ready": require_ready,
        "manifest": str(manifest_path),
        "bundle_dir": str(bundle_dir),
        "readiness_status": manifest.get("readiness_status"),
        "blocking_gates": manifest.get("blocking_gates", []),
        "file_count": len(file_records),
        "checked_file_count": len(file_results),
        "error_count": len(errors),
        "errors": errors,
        "files": file_results,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bundle-dir", type=Path)
    group.add_argument("--manifest", type=Path)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--json", type=Path, help="Optional path to write the verification JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest_path = manifest_path_from_args(args)
    result = verify_bundle(
        manifest_path=manifest_path,
        require_ready=bool(args.require_ready),
    )
    if args.json:
        write_json(repo_path(args.json), result)
    print(json.dumps(result, ensure_ascii=False))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
