#!/usr/bin/env bash
set -euo pipefail

# Always run from repository root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

# Usage: ./scripts/evaluator/evaluate_utils/swebench_pkg/start_server.sh [--host 0.0.0.0] [--port 8000]
# Notes:
# - .env があれば読み込んで SWE_API_KEY を環境に流し込みます
# - ログは /tmp/swebench_server.out に出力します
# - 既存プロセスが居れば停止します

HOST=0.0.0.0
PORT=8000

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Load .env if exists
if [[ -f .env ]]; then
  # shellcheck disable=SC2046
  export $(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' .env | xargs -d'\n') || true
fi

# Ensure myenv python exists
if [[ ! -x ./myenv/bin/python ]]; then
  echo "[ERROR] ./myenv/bin/python が見つかりません。仮想環境を作成・依存をインストールしてください。" >&2
  exit 1
fi

# Kill existing server
pids=$(ps aux | grep -E "swebench_server\.py" | grep -v grep | awk '{print $2}' || true)
if [[ -n "${pids}" ]]; then
  echo "Killing existing server: ${pids}"
  # shellcheck disable=SC2086
  kill -9 ${pids} || true
fi

# Start server
LOG=/tmp/swebench_server.out
nohup ./myenv/bin/python scripts/server/swebench_server.py --host "$HOST" --port "$PORT" >"$LOG" 2>&1 & disown || true
sleep 2

echo "Server started on http://$HOST:$PORT"
echo "Logs: $LOG"