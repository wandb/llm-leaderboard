#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/setup/run_nemoclaw_local_verification.sh [--include-release-gate]

Runs the offline/local NeMoClaw verification test slice used by the Taiwan
leaderboard work. The default path does not install or onboard NeMoClaw, query
W&B, call router providers, or run paid model inference.

Options:
  --include-release-gate  Also run the offline Taiwan release gate after tests.
  -h, --help              Show this help.
USAGE
}

include_release_gate=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --include-release-gate)
      include_release_gate=1
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

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

if command -v pytest >/dev/null 2>&1; then
  pytest_cmd=(pytest)
elif command -v python3 >/dev/null 2>&1 && python3 -c 'import pytest' >/dev/null 2>&1; then
  pytest_cmd=(python3 -m pytest)
elif command -v uv >/dev/null 2>&1; then
  pytest_cmd=(uv run pytest)
else
  echo "Could not find pytest, python3 -m pytest, or uv run pytest." >&2
  exit 127
fi

export PYTEST_DISABLE_PLUGIN_AUTOLOAD="${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}"

tests=(
  tests/test_run_nemoclaw_local_verification.py
  tests/test_install_nemoclaw_script.py
  tests/test_review_nemoclaw_installer.py
  tests/test_verify_nemoclaw_operator_docs.py
  tests/test_verify_nemoclaw_post_install.py
  tests/test_check_taiwan_nemoclaw_adoption.py
  tests/test_openclaw_agent_protocol.py
  tests/test_agentic_math.py
  tests/test_swebench_pro.py
  tests/test_taiwan_full_batch_runner.py
  tests/test_taiwan_production_readiness_gate.py
)

echo "Running offline NeMoClaw local verification tests..."
echo "PYTEST_DISABLE_PLUGIN_AUTOLOAD=${PYTEST_DISABLE_PLUGIN_AUTOLOAD}"
"${pytest_cmd[@]}" -q "${tests[@]}"

if [ "$include_release_gate" -eq 1 ]; then
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required for --include-release-gate." >&2
    exit 127
  fi
  echo "Running offline Taiwan release gate..."
  python3 scripts/tools/run_taiwan_release_gate.py --quiet
fi

echo "Offline NeMoClaw local verification passed."
