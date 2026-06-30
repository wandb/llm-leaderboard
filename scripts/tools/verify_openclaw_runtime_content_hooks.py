#!/usr/bin/env python3
"""Verify OpenClaw runtime hooks needed for Weave Agents content capture.

The weave-openclaw plugin can only produce useful W&B Agents conversations if
OpenClaw emits typed hook events that carry prompt, assistant, and tool content.
This script is intentionally static and local-only: it inspects the installed
OpenClaw JavaScript bundle and fails when the expected hook/content contract is
not present. It does not run a model or send traces to W&B.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


WINDOW_CHARS = 5000


@dataclass(frozen=True)
class HookContract:
    hook: str
    required_terms: tuple[str, ...]


HOOK_CONTRACTS: tuple[HookContract, ...] = (
    HookContract(
        hook="model_call_started",
        required_terms=("runModelCallStarted", "callId", "provider", "model"),
    ),
    HookContract(
        hook="llm_input",
        required_terms=("runLlmInput", "systemPrompt", "prompt", "historyMessages", "tools"),
    ),
    HookContract(
        hook="llm_output",
        required_terms=("runLlmOutput", "assistantTexts", "lastAssistant", "usage"),
    ),
    HookContract(
        hook="before_tool_call",
        required_terms=("runBeforeToolCall", "toolName", "params", "toolCallId"),
    ),
    HookContract(
        hook="after_tool_call",
        required_terms=("runAfterToolCall", "toolName", "params", "result", "toolCallId"),
    ),
    HookContract(
        hook="before_message_write",
        required_terms=("runBeforeMessageWrite", "message"),
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--openclaw-package-dir",
        type=Path,
        default=None,
        help="Installed OpenClaw package directory. Defaults to OPENCLAW_PACKAGE_DIR, npm root -g/openclaw, or the openclaw binary target.",
    )
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=None,
        help="OpenClaw dist directory to inspect. Overrides --openclaw-package-dir.",
    )
    parser.add_argument(
        "--openclaw-bin",
        default="openclaw",
        help="OpenClaw executable used only for package-dir discovery.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    return parser.parse_args(argv)


def npm_global_openclaw_dir(env: dict[str, str]) -> Path | None:
    npm = shutil.which("npm", path=env.get("PATH"))
    if not npm:
        return None
    try:
        result = subprocess.run(
            [npm, "root", "-g"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    root = result.stdout.strip()
    if result.returncode != 0 or not root:
        return None
    return Path(root) / "openclaw"


def package_dir_from_binary(openclaw_bin: str, env: dict[str, str]) -> Path | None:
    binary = shutil.which(openclaw_bin, path=env.get("PATH"))
    if not binary:
        return None
    path = Path(binary).resolve()
    candidates = [path, *path.parents]
    for candidate in candidates:
        package_json = candidate / "package.json"
        if not package_json.exists():
            continue
        try:
            payload = json.loads(package_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("name") == "openclaw":
            return candidate
    for parent in path.parents:
        candidate = parent / "lib" / "node_modules" / "openclaw"
        if (candidate / "package.json").exists():
            return candidate
    return None


def resolve_dist_dir(args: argparse.Namespace, env: dict[str, str]) -> tuple[Path | None, Path]:
    if args.dist_dir is not None:
        dist_dir = args.dist_dir.resolve()
        return dist_dir.parent, dist_dir

    candidates: list[Path] = []
    if args.openclaw_package_dir is not None:
        candidates.append(args.openclaw_package_dir)
    if env.get("OPENCLAW_PACKAGE_DIR"):
        candidates.append(Path(env["OPENCLAW_PACKAGE_DIR"]))
    npm_candidate = npm_global_openclaw_dir(env)
    if npm_candidate is not None:
        candidates.append(npm_candidate)
    binary_candidate = package_dir_from_binary(args.openclaw_bin, env)
    if binary_candidate is not None:
        candidates.append(binary_candidate)

    seen: set[Path] = set()
    for candidate in candidates:
        package_dir = candidate.expanduser().resolve()
        if package_dir in seen:
            continue
        seen.add(package_dir)
        dist_dir = package_dir / "dist"
        if dist_dir.is_dir():
            return package_dir, dist_dir

    raise SystemExit(
        "Could not locate OpenClaw dist directory. Pass --openclaw-package-dir or --dist-dir."
    )


def iter_js_files(dist_dir: Path) -> Iterable[Path]:
    yield from sorted(path for path in dist_dir.rglob("*.js") if path.is_file())


def all_positions(text: str, needle: str) -> list[int]:
    positions: list[int] = []
    start = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            return positions
        positions.append(index)
        start = index + len(needle)


def line_offsets(text: str) -> list[int]:
    return [0, *[index + 1 for index, char in enumerate(text) if char == "\n"]]


def line_number(offsets: list[int], position: int) -> int:
    return bisect.bisect_right(offsets, position)


def snippet_around(text: str, position: int, radius: int = 160) -> str:
    start = max(0, position - radius)
    end = min(len(text), position + radius)
    return " ".join(text[start:end].split())


def inspect_contract(contract: HookContract, dist_dir: Path) -> dict[str, object]:
    quoted_hook = json.dumps(contract.hook)
    hook_windows: list[str] = []
    occurrences: list[dict[str, object]] = []
    matched_terms: dict[str, bool] = {term: False for term in contract.required_terms}

    for path in iter_js_files(dist_dir):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        positions = all_positions(text, quoted_hook)
        if not positions:
            continue
        offsets = line_offsets(text)
        relative = str(path.relative_to(dist_dir))
        for position in positions[:8]:
            start = max(0, position - WINDOW_CHARS)
            end = min(len(text), position + WINDOW_CHARS)
            hook_windows.append(text[start:end])
            occurrences.append(
                {
                    "file": relative,
                    "line": line_number(offsets, position),
                    "snippet": snippet_around(text, position),
                }
            )

    combined = "\n".join(hook_windows)
    for term in matched_terms:
        matched_terms[term] = term in combined

    missing_terms = [term for term, matched in matched_terms.items() if not matched]
    ok = bool(occurrences) and not missing_terms
    return {
        "hook": contract.hook,
        "ok": ok,
        "occurrences": len(occurrences),
        "matched_terms": matched_terms,
        "missing_terms": missing_terms,
        "evidence": occurrences[:5],
    }


def verify_runtime(dist_dir: Path, package_dir: Path | None = None) -> dict[str, object]:
    checks = [inspect_contract(contract, dist_dir) for contract in HOOK_CONTRACTS]
    ok = all(bool(check["ok"]) for check in checks)
    return {
        "ok": ok,
        "openclaw_package_dir": str(package_dir) if package_dir is not None else None,
        "dist_dir": str(dist_dir),
        "checks": checks,
    }


def print_human(payload: dict[str, object]) -> None:
    status = "OK" if payload["ok"] else "FAILED"
    print(f"OpenClaw runtime content-hook contract: {status}")
    print(f"dist_dir: {payload['dist_dir']}")
    for check in payload["checks"]:
        mark = "OK" if check["ok"] else "FAIL"
        missing = check["missing_terms"]
        detail = "" if not missing else f" missing={', '.join(missing)}"
        print(f"- {mark} {check['hook']} occurrences={check['occurrences']}{detail}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    package_dir, dist_dir = resolve_dist_dir(args, os.environ.copy())
    if not dist_dir.is_dir():
        raise SystemExit(f"OpenClaw dist directory does not exist: {dist_dir}")
    payload = verify_runtime(dist_dir, package_dir)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_human(payload)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
