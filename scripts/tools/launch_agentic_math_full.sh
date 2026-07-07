#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/tools/launch_agentic_math_full.sh --slug SLUG --model OPENCLAW_MODEL --thinking LEVEL [options]

Options:
  --dataset-jsonl PATH       Dataset JSONL. Defaults to Taiwan OlymMATH-HARD leaderboard.
  --output-root PATH         Output root. Defaults to outputs/taiwan_full_eval/agentic_math.
  --env-file PATH            Environment file to source. Defaults to .env.
  --timeout SECONDS          Per OpenClaw task timeout. Defaults to 2400.
  --max-attempts N           Per-task retry attempts. Defaults to 3.
  --retry-base-seconds N     Linear retry base. Defaults to 15.
  --agent AGENT              Base OpenClaw agent. Defaults to main.
  --force                    Start even if the pidfile points at a live process.
EOF
}

slug=""
model=""
thinking=""
dataset_jsonl="data/taiwan/agentic_math_olymmath_hard_zh_tw/subsets/leaderboard.jsonl"
output_root="outputs/taiwan_full_eval/agentic_math"
env_file=".env"
timeout="2400"
max_attempts="3"
retry_base_seconds="15"
agent="main"
force="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --slug) slug="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --thinking) thinking="$2"; shift 2 ;;
    --dataset-jsonl) dataset_jsonl="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --env-file) env_file="$2"; shift 2 ;;
    --timeout) timeout="$2"; shift 2 ;;
    --max-attempts) max_attempts="$2"; shift 2 ;;
    --retry-base-seconds) retry_base_seconds="$2"; shift 2 ;;
    --agent) agent="$2"; shift 2 ;;
    --force) force="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$slug" || -z "$model" || -z "$thinking" ]]; then
  usage >&2
  exit 2
fi

if [[ ! -f "$dataset_jsonl" ]]; then
  echo "Dataset not found: $dataset_jsonl" >&2
  exit 1
fi

if [[ -f "$env_file" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
fi

base="$output_root/$slug"
log_dir="$base/logs"
output_dir="$base/openclaw"
pid_file="$log_dir/resume_full_100.pid"
logpath_file="$log_dir/resume_full_100.logpath"
mkdir -p "$log_dir" "$output_dir"

if [[ -f "$pid_file" && "$force" != "1" ]]; then
  old_pid="$(cat "$pid_file" || true)"
  if [[ -n "$old_pid" ]] && ps -p "$old_pid" >/dev/null 2>&1; then
    echo "Refusing to start: live process $old_pid already recorded in $pid_file" >&2
    exit 1
  fi
fi

timestamp="$(date +%Y%m%dT%H%M%S%z)"
log_path="$log_dir/resume_full_100_${timestamp}_setsid.log"
printf '%s\n' "$log_path" > "$logpath_file"

cmd=(
  ".venv/bin/python3"
  "scripts/tools/run_agentic_math_openclaw.py"
  "--dataset-jsonl" "$dataset_jsonl"
  "--output-dir" "$output_dir"
  "--prefix" "taiwan-math-$slug"
  "--thinking" "$thinking"
  "--agent" "$agent"
  "--openclaw-timeout" "$timeout"
  "--openclaw-max-attempts" "$max_attempts"
  "--openclaw-retry-base-seconds" "$retry_base_seconds"
  "--openclaw-tool-profile" "coding"
  "--task-agent-prefix" "tw-math-$slug"
  "--model" "$model"
  "--no-weave-sidecar"
  "--deny-tool" "code_execution"
  "--deny-tool" "web_search"
  "--deny-tool" "web_fetch"
  "--deny-tool" "browser"
  "--deny-tool" "browser_*"
  "--deny-argument-pattern" "https?://"
  "--deny-argument-pattern" "\\b(curl|wget)\\b"
  "--deny-argument-pattern" "\\b(requests|urllib|httpx)\\."
)

setsid env PYTHONUNBUFFERED=1 bash -c 'echo "START $(date -Is)"; exec "$@"' bash "${cmd[@]}" \
  > "$log_path" 2>&1 < /dev/null &
pid="$!"
printf '%s\n' "$pid" > "$pid_file"

echo "pid=$pid"
echo "log=$log_path"
ps -o pid,ppid,etime,pcpu,pmem,stat,cmd -p "$pid" || true
