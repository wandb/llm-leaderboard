#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/tools/launch_swebench_pro_full.sh --slug SLUG --model OPENCLAW_MODEL --thinking LEVEL [options]

Options:
  --dataset-jsonl PATH       Dataset JSONL. Defaults to local SWE-Bench Pro Compact80 artifact.
  --output-root PATH         Output root. Defaults to outputs/taiwan_full_eval/swebench_pro.
  --checkout-root PATH       Checkout root. Defaults to outputs/taiwan_full_eval/swebench_pro_checkouts/SLUG.
  --max-input-tokens N       Per-task input-token cap. Defaults to 1000000.
  --max-tool-calls N         Per-task tool-call cap. Defaults to 60.
  --nemoclaw-sandbox NAME    Run OpenClaw through NeMoClaw sandbox NAME.
  --nemoclaw-bin PATH        NeMoClaw executable. Defaults to nemoclaw.
  --nemoclaw-workdir PATH    Workdir inside sandbox. Defaults to sandbox-visible checkout.
  --nemoclaw-checkout-sandbox-root PATH
                            Sandbox-visible root corresponding to --checkout-root.
  --env-file PATH            Environment file to source. Defaults to .env.
  --timeout SECONDS          Per OpenClaw task timeout. Defaults to 3600.
  --max-attempts N           Per-task retry attempts. Defaults to 3.
  --retry-base-seconds N     Linear retry base. Defaults to 15.
  --agent AGENT              Base OpenClaw agent. Defaults to main.
  --redo                     Regenerate matching cached patches.
  --force                    Start even if the pidfile points at a live process.
EOF
}

slug=""
model=""
thinking=""
dataset_jsonl="data/taiwan/swebench_pro_public/subsets/leaderboard_compact_80.jsonl"
output_root="outputs/taiwan_full_eval/swebench_pro"
checkout_root=""
max_input_tokens="1000000"
max_tool_calls="60"
max_agent_turns="60"
nemoclaw_sandbox=""
nemoclaw_bin="nemoclaw"
nemoclaw_workdir=""
nemoclaw_checkout_sandbox_root=""
env_file=".env"
timeout="3600"
max_attempts="3"
retry_base_seconds="15"
agent="main"
redo="0"
force="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --slug) slug="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --thinking) thinking="$2"; shift 2 ;;
    --dataset-jsonl) dataset_jsonl="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --checkout-root) checkout_root="$2"; shift 2 ;;
    --max-input-tokens) max_input_tokens="$2"; shift 2 ;;
    --max-tool-calls) max_tool_calls="$2"; shift 2 ;;
    --max-agent-turns) max_agent_turns="$2"; shift 2 ;;
    --nemoclaw-sandbox) nemoclaw_sandbox="$2"; shift 2 ;;
    --nemoclaw-bin) nemoclaw_bin="$2"; shift 2 ;;
    --nemoclaw-workdir) nemoclaw_workdir="$2"; shift 2 ;;
    --nemoclaw-checkout-sandbox-root) nemoclaw_checkout_sandbox_root="$2"; shift 2 ;;
    --env-file) env_file="$2"; shift 2 ;;
    --timeout) timeout="$2"; shift 2 ;;
    --max-attempts) max_attempts="$2"; shift 2 ;;
    --retry-base-seconds) retry_base_seconds="$2"; shift 2 ;;
    --agent) agent="$2"; shift 2 ;;
    --redo) redo="1"; shift ;;
    --force) force="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$slug" || -z "$model" || -z "$thinking" ]]; then
  usage >&2
  exit 2
fi

if [[ -z "$checkout_root" ]]; then
  checkout_root="outputs/taiwan_full_eval/swebench_pro_checkouts/$slug"
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
pid_file="$log_dir/resume_full_80.pid"
logpath_file="$log_dir/resume_full_80.logpath"
mkdir -p "$log_dir" "$output_dir" "$checkout_root"

if [[ -f "$pid_file" && "$force" != "1" ]]; then
  old_pid="$(cat "$pid_file" || true)"
  if [[ -n "$old_pid" ]] && ps -p "$old_pid" >/dev/null 2>&1; then
    echo "Refusing to start: live process $old_pid already recorded in $pid_file" >&2
    exit 1
  fi
fi

timestamp="$(date +%Y%m%dT%H%M%S%z)"
log_path="$log_dir/resume_full_80_${timestamp}_setsid.log"
printf '%s\n' "$log_path" > "$logpath_file"

cmd=(
  ".venv/bin/python3"
  "scripts/tools/run_swebench_pro_openclaw.py"
  "--dataset-jsonl" "$dataset_jsonl"
  "--output-dir" "$output_dir"
  "--checkout-root" "$checkout_root"
  "--prefix" "taiwan-swe-$slug"
  "--thinking" "$thinking"
  "--agent" "$agent"
  "--openclaw-timeout" "$timeout"
  "--openclaw-max-attempts" "$max_attempts"
  "--openclaw-retry-base-seconds" "$retry_base_seconds"
  "--max-input-tokens" "$max_input_tokens"
  "--max-tool-calls" "$max_tool_calls"
  "--max-agent-turns" "$max_agent_turns"
  "--openclaw-tool-profile" "coding"
  "--task-agent-prefix" "tw-swe-$slug"
  "--model" "$model"
  "--no-weave-sidecar"
  "--deny-tool" "code_execution"
  "--deny-tool" "web_search"
  "--deny-tool" "web_fetch"
  "--deny-tool" "browser"
  "--deny-tool" "browser_*"
  "--deny-tool" "*search*"
  "--deny-argument-pattern" "https?://"
  "--deny-argument-pattern" "\\b(curl|wget)\\b"
  "--deny-argument-pattern" "\\b(requests|urllib|httpx)\\."
)

if [[ -n "$nemoclaw_sandbox" ]]; then
  cmd+=("--nemoclaw-sandbox" "$nemoclaw_sandbox")
  cmd+=("--nemoclaw-bin" "$nemoclaw_bin")
  if [[ -n "$nemoclaw_workdir" ]]; then
    cmd+=("--nemoclaw-workdir" "$nemoclaw_workdir")
  fi
  if [[ -n "$nemoclaw_checkout_sandbox_root" ]]; then
    cmd+=("--nemoclaw-checkout-sandbox-root" "$nemoclaw_checkout_sandbox_root")
  fi
fi

if [[ "$redo" == "1" ]]; then
  cmd+=("--redo")
fi

setsid env PYTHONUNBUFFERED=1 bash -c 'echo "START $(date -Is)"; exec "$@"' bash "${cmd[@]}" \
  > "$log_path" 2>&1 < /dev/null &
pid="$!"
printf '%s\n' "$pid" > "$pid_file"

echo "pid=$pid"
echo "log=$log_path"
ps -o pid,ppid,etime,pcpu,pmem,stat,cmd -p "$pid" || true
