#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/setup/install_agentic_math_sandbox_deps.sh [options]

Install or verify the Python math stack used by Agentic Math inside a NeMoClaw
sandbox. Python is allowed as a local, non-interactive tool for OlymMATH-HARD,
so the sandbox must provide the same basic scientific packages across machines.

Options:
  --sandbox NAME       NeMoClaw sandbox name. Default: nejumi-taiwan
  --nemoclaw-bin PATH  NeMoClaw executable. Default: nemoclaw
  --check-only         Verify imports only; do not install packages
  -h, --help           Show this help

Packages:
  numpy scipy sympy
USAGE
}

sandbox="nejumi-taiwan"
nemoclaw_bin="nemoclaw"
check_only=0
packages=(numpy scipy sympy)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sandbox)
      sandbox="${2:?--sandbox requires a value}"
      shift 2
      ;;
    --nemoclaw-bin)
      nemoclaw_bin="${2:?--nemoclaw-bin requires a value}"
      shift 2
      ;;
    --check-only)
      check_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v "$nemoclaw_bin" >/dev/null 2>&1; then
  echo "NeMoClaw executable not found: $nemoclaw_bin" >&2
  exit 127
fi

verify_code='import importlib, json
mods = ["numpy", "scipy", "sympy"]
versions = {}
missing = []
for name in mods:
    try:
        mod = importlib.import_module(name)
        versions[name] = getattr(mod, "__version__", "unknown")
    except Exception as exc:
        missing.append({"name": name, "error": repr(exc)})
print(json.dumps({"ok": not missing, "versions": versions, "missing": missing}, ensure_ascii=False, indent=2))
raise SystemExit(0 if not missing else 1)'
verify_code_b64="$(printf '%s' "$verify_code" | base64 | tr -d '\n')"
verify_command="import base64; exec(base64.b64decode('${verify_code_b64}'))"
sitecustomize_code='from __future__ import annotations

import sys
from pathlib import Path

version = f"{sys.version_info.major}.{sys.version_info.minor}"
for path in [Path("/tmp/.local/lib") / f"python{version}" / "site-packages"]:
    text = str(path)
    if path.exists() and text not in sys.path:
        sys.path.insert(0, text)'
sitecustomize_code_b64="$(printf '%s' "$sitecustomize_code" | base64 | tr -d '\n')"
sitecustomize_command="import base64; from pathlib import Path; Path('/sandbox/sitecustomize.py').write_text(base64.b64decode('${sitecustomize_code_b64}').decode('utf-8'), encoding='utf-8')"

if [[ "$check_only" -eq 1 ]]; then
  exec "$nemoclaw_bin" sandbox exec "$sandbox" \
    --workdir /sandbox \
    --no-tty \
    --timeout 120 \
    -- python3 -c "$verify_command"
fi

"$nemoclaw_bin" sandbox exec "$sandbox" \
  --workdir /sandbox \
  --no-tty \
  --timeout 120 \
  -- python3 -m ensurepip --upgrade >/dev/null 2>&1 || true

"$nemoclaw_bin" sandbox exec "$sandbox" \
  --workdir /sandbox \
  --no-tty \
  --timeout 900 \
  -- python3 -m pip install --upgrade --break-system-packages "${packages[@]}"

"$nemoclaw_bin" sandbox exec "$sandbox" \
  --workdir /sandbox \
  --no-tty \
  --timeout 120 \
  -- python3 -c "$sitecustomize_command"

"$nemoclaw_bin" sandbox exec "$sandbox" \
  --workdir /sandbox \
  --no-tty \
  --timeout 120 \
  -- python3 -c "$verify_command"
