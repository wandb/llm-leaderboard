#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Configure native weave-openclaw inside a NeMoClaw sandbox.

Usage:
  scripts/setup/configure_nemoclaw_weave.sh [options]

Options:
  --sandbox NAME              NeMoClaw sandbox name. Default: nejumi-taiwan.
  --nemoclaw-bin PATH         NeMoClaw executable. Default: nemoclaw.
  --env-file PATH             Env file to source before reading WANDB_API_KEY. Default: .env.
  --wandb-key-env NAME        Env var containing the W&B API key. Default: WANDB_API_KEY.
  --entity NAME               W&B entity. Default: llm-leaderboard.
  --project NAME              W&B project. Default: tc-leaderboard.
  --agent-name NAME           Weave agent name. Default: nejumi-taiwan-openclaw.
  --agent-version VALUE       Weave agent version. Default: nejumi-agent-protocol-2026.04.
  --service-name NAME         Weave service name. Default: openclaw-agent.
  --openai-key-env NAME       Env var containing the OpenAI API key. Default: OPENAI_API_KEY.
  --anthropic-key-env NAME    Env var containing the Anthropic API key. Default: ANTHROPIC_API_KEY.
  --wandb-inference-model-id ID
                              Also register a W&B Inference OpenClaw model. No default;
                              pass an ID confirmed by the W&B Inference /v1/models API.
  --wandb-inference-provider-id ID
                              OpenClaw provider ID for W&B Inference. Default: wandb-inference.
  --wandb-inference-base-url URL
                              W&B Inference base URL. Default: https://api.inference.wandb.ai/v1.
  --wandb-inference-max-tokens N
                              Model maxTokens field for OpenClaw. Default: 4096.
  --wandb-inference-context-window N
                              Optional model contextWindow field for OpenClaw.
  --wandb-inference-reasoning true|false
                              Model reasoning flag for OpenClaw. Default: true.
  --wandb-inference-model-params-json JSON
                              JSON object merged into the model params field.
  --policy-file PATH          NeMoClaw W&B egress policy YAML.
  --secret-file PATH          Sandbox secret JSON path. Default: /sandbox/.openclaw/nejumi_secrets.json.
  --openclaw-config PATH      Sandbox OpenClaw config path. Default: /sandbox/.openclaw/openclaw.json.
  --weave-plugin-source MODE  auto|local|npm. Default: auto.
  --local-weave-project PATH  Host weave-openclaw npm project for local install fallback.
  --force-plugin-install      Reinstall weave-openclaw even when the sandbox reports it as loaded.
  --check-only                Do not mutate; inspect current sandbox state only.
  --skip-policy               Do not add the W&B egress policy.
  --skip-plugin-install       Do not install weave-openclaw.
  --skip-openai-direct        Do not merge the OpenAI-direct canary provider.
  --skip-anthropic-direct     Do not merge the Anthropic-direct provider.
  --json PATH                 Write a machine-readable report.
  -h, --help                  Show this help.

The W&B key is sent to the sandbox over stdin and stored as a file SecretRef
source. The secret value is not included in command arguments or JSON reports.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SANDBOX="nejumi-taiwan"
NEMOCLAW_BIN="nemoclaw"
ENV_FILE="$REPO_ROOT/.env"
WANDB_KEY_ENV="WANDB_API_KEY"
OPENAI_KEY_ENV="OPENAI_API_KEY"
ANTHROPIC_KEY_ENV="ANTHROPIC_API_KEY"
WANDB_INFERENCE_MODEL_ID=""
WANDB_INFERENCE_PROVIDER_ID="wandb-inference"
WANDB_INFERENCE_BASE_URL="https://api.inference.wandb.ai/v1"
WANDB_INFERENCE_MAX_TOKENS="4096"
WANDB_INFERENCE_CONTEXT_WINDOW=""
WANDB_INFERENCE_REASONING="true"
WANDB_INFERENCE_MODEL_PARAMS_JSON="{}"
ENTITY="llm-leaderboard"
PROJECT="tc-leaderboard"
AGENT_NAME="nejumi-taiwan-openclaw"
AGENT_VERSION="nejumi-agent-protocol-2026.04"
SERVICE_NAME="openclaw-agent"
POLICY_FILE="$REPO_ROOT/configs/nemoclaw/policies/wandb_weave.yaml"
SECRET_FILE="/sandbox/.openclaw/nejumi_secrets.json"
OPENCLAW_CONFIG="/sandbox/.openclaw/openclaw.json"
WEAVE_PLUGIN_SOURCE="auto"
LOCAL_WEAVE_PROJECT="${OPENCLAW_WEAVE_PROJECT:-$HOME/.openclaw/npm/projects/weave-openclaw}"
SANDBOX_WEAVE_PROJECT="/sandbox/.openclaw/npm/projects/weave-openclaw"
CHECK_ONLY=0
SKIP_POLICY=0
SKIP_PLUGIN_INSTALL=0
SKIP_OPENAI_DIRECT=0
SKIP_ANTHROPIC_DIRECT=0
FORCE_PLUGIN_INSTALL=0
JSON_OUT=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --sandbox) SANDBOX="$2"; shift 2 ;;
    --nemoclaw-bin) NEMOCLAW_BIN="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --wandb-key-env) WANDB_KEY_ENV="$2"; shift 2 ;;
    --entity) ENTITY="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --agent-name) AGENT_NAME="$2"; shift 2 ;;
    --agent-version) AGENT_VERSION="$2"; shift 2 ;;
    --service-name) SERVICE_NAME="$2"; shift 2 ;;
    --openai-key-env) OPENAI_KEY_ENV="$2"; shift 2 ;;
    --anthropic-key-env) ANTHROPIC_KEY_ENV="$2"; shift 2 ;;
    --wandb-inference-model-id) WANDB_INFERENCE_MODEL_ID="$2"; shift 2 ;;
    --wandb-inference-provider-id) WANDB_INFERENCE_PROVIDER_ID="$2"; shift 2 ;;
    --wandb-inference-base-url) WANDB_INFERENCE_BASE_URL="$2"; shift 2 ;;
    --wandb-inference-max-tokens) WANDB_INFERENCE_MAX_TOKENS="$2"; shift 2 ;;
    --wandb-inference-context-window) WANDB_INFERENCE_CONTEXT_WINDOW="$2"; shift 2 ;;
    --wandb-inference-reasoning) WANDB_INFERENCE_REASONING="$2"; shift 2 ;;
    --wandb-inference-model-params-json) WANDB_INFERENCE_MODEL_PARAMS_JSON="$2"; shift 2 ;;
    --policy-file) POLICY_FILE="$2"; shift 2 ;;
    --secret-file) SECRET_FILE="$2"; shift 2 ;;
    --openclaw-config) OPENCLAW_CONFIG="$2"; shift 2 ;;
    --weave-plugin-source) WEAVE_PLUGIN_SOURCE="$2"; shift 2 ;;
    --local-weave-project) LOCAL_WEAVE_PROJECT="$2"; shift 2 ;;
    --force-plugin-install) FORCE_PLUGIN_INSTALL=1; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --skip-policy) SKIP_POLICY=1; shift ;;
    --skip-plugin-install) SKIP_PLUGIN_INSTALL=1; shift ;;
    --skip-openai-direct) SKIP_OPENAI_DIRECT=1; shift ;;
    --skip-anthropic-direct) SKIP_ANTHROPIC_DIRECT=1; shift ;;
    --json) JSON_OUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! "$WANDB_KEY_ENV" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
  echo "Invalid --wandb-key-env: $WANDB_KEY_ENV" >&2
  exit 2
fi
if [[ ! "$OPENAI_KEY_ENV" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
  echo "Invalid --openai-key-env: $OPENAI_KEY_ENV" >&2
  exit 2
fi
if [[ ! "$ANTHROPIC_KEY_ENV" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
  echo "Invalid --anthropic-key-env: $ANTHROPIC_KEY_ENV" >&2
  exit 2
fi
if [[ ! "$WANDB_INFERENCE_MAX_TOKENS" =~ ^[0-9]+$ ]] || [ "$WANDB_INFERENCE_MAX_TOKENS" -le 0 ]; then
  echo "Invalid --wandb-inference-max-tokens: $WANDB_INFERENCE_MAX_TOKENS" >&2
  exit 2
fi
if [ -n "$WANDB_INFERENCE_CONTEXT_WINDOW" ] && { [[ ! "$WANDB_INFERENCE_CONTEXT_WINDOW" =~ ^[0-9]+$ ]] || [ "$WANDB_INFERENCE_CONTEXT_WINDOW" -le 0 ]; }; then
  echo "Invalid --wandb-inference-context-window: $WANDB_INFERENCE_CONTEXT_WINDOW" >&2
  exit 2
fi
case "$WANDB_INFERENCE_REASONING" in
  true|false) ;;
  *) echo "Invalid --wandb-inference-reasoning: $WANDB_INFERENCE_REASONING" >&2; exit 2 ;;
esac
python3 - "$WANDB_INFERENCE_MODEL_PARAMS_JSON" <<'PY'
import json
import sys
try:
    value = json.loads(sys.argv[1])
except json.JSONDecodeError as exc:
    raise SystemExit(f"Invalid --wandb-inference-model-params-json: {exc}") from exc
if not isinstance(value, dict):
    raise SystemExit("--wandb-inference-model-params-json must be a JSON object")
PY

case "$WEAVE_PLUGIN_SOURCE" in
  auto|local|npm) ;;
  *) echo "Invalid --weave-plugin-source: $WEAVE_PLUGIN_SOURCE" >&2; exit 2 ;;
esac

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

if ! command -v "$NEMOCLAW_BIN" >/dev/null 2>&1; then
  echo "NeMoClaw executable not found: $NEMOCLAW_BIN" >&2
  exit 1
fi

if [ ! -f "$POLICY_FILE" ]; then
  echo "Policy file not found: $POLICY_FILE" >&2
  exit 1
fi

WANDb_KEY_VALUE="${!WANDB_KEY_ENV:-}"
credential_available=false
if [ -n "$WANDb_KEY_VALUE" ]; then
  credential_available=true
fi
OPENAI_KEY_VALUE="${!OPENAI_KEY_ENV:-}"
openai_credential_available=false
if [ -n "$OPENAI_KEY_VALUE" ]; then
  openai_credential_available=true
fi
ANTHROPIC_KEY_VALUE="${!ANTHROPIC_KEY_ENV:-}"
anthropic_credential_available=false
if [ -n "$ANTHROPIC_KEY_VALUE" ]; then
  anthropic_credential_available=true
fi

run_nemoclaw() {
  "$NEMOCLAW_BIN" sandbox exec "$SANDBOX" --workdir /sandbox --no-tty --timeout "${2:-120}" -- bash -lc "$1"
}

json_bool() {
  if [ "$1" = true ] || [ "$1" = 1 ]; then
    printf true
  else
    printf false
  fi
}

plugin_probe() {
  run_nemoclaw "openclaw plugins list --json | python3 -c 'import json,sys; raw=sys.stdin.read(); start=raw.find(\"{\"); end=raw.rfind(\"}\"); data=json.loads(raw[start:end+1]) if start >= 0 and end >= start else {}; print(str(any(p.get(\"id\")==\"weave\" and p.get(\"enabled\") for p in data.get(\"plugins\", []))).lower())'" 120 2>/dev/null || printf false
}

config_probe() {
  "$NEMOCLAW_BIN" sandbox exec "$SANDBOX" --workdir /sandbox --no-tty --timeout 60 -- python3 - "$OPENCLAW_CONFIG" <<'PY' 2>/dev/null || printf false
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("false")
    raise SystemExit
data = json.loads(path.read_text(encoding="utf-8"))
entry = data.get("plugins", {}).get("entries", {}).get("weave")
cfg = entry.get("config") if isinstance(entry, dict) else None
api_key = cfg.get("apiKey") if isinstance(cfg, dict) else None
print(str(bool(
    isinstance(entry, dict)
    and entry.get("enabled") is True
    and isinstance(api_key, dict)
    and api_key.get("source") == "file"
)).lower())
PY
}

policy_probe() {
  local status
  status="$("$NEMOCLAW_BIN" sandbox status "$SANDBOX" 2>/dev/null || true)"
  if ! grep -q "host: api.wandb.ai" <<<"$status"; then
    printf false
    return
  fi
  if [ -n "$WANDB_INFERENCE_MODEL_ID" ] && ! grep -q "host: api.inference.wandb.ai" <<<"$status"; then
    printf false
    return
  fi
  if [ "$SKIP_OPENAI_DIRECT" -eq 0 ] && ! grep -q "host: api.openai.com" <<<"$status"; then
    printf false
    return
  fi
  if [ "$SKIP_ANTHROPIC_DIRECT" -eq 0 ] && ! grep -q "host: api.anthropic.com" <<<"$status"; then
    printf false
    return
  fi
  printf true
}

install_weave_from_local_project() {
  local local_project="$1"
  local tmp_dir tar_path host_sha sandbox_sha chunk
  if [ ! -d "$local_project/node_modules/weave-openclaw" ]; then
    return 1
  fi
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "$tmp_dir"' RETURN
  tar_path="$tmp_dir/weave-openclaw-project.tgz"
  tar -C "$(dirname "$local_project")" -czf "$tar_path" "$(basename "$local_project")"
  host_sha="$(sha256sum "$tar_path" | awk '{print $1}')"
  "$NEMOCLAW_BIN" sandbox exec "$SANDBOX" --workdir /sandbox --no-tty --timeout 30 -- bash -lc \
    'mkdir -p /sandbox/tmp /sandbox/.openclaw/npm/projects && : > /sandbox/tmp/weave-openclaw-project.tgz'
  mkdir -p "$tmp_dir/chunks"
  split -b 512k "$tar_path" "$tmp_dir/chunks/part-"
  for chunk in "$tmp_dir"/chunks/part-*; do
    "$NEMOCLAW_BIN" sandbox exec "$SANDBOX" --workdir /sandbox --no-tty --timeout 30 -- bash -lc \
      'cat >> /sandbox/tmp/weave-openclaw-project.tgz' < "$chunk"
  done
  sandbox_sha="$("$NEMOCLAW_BIN" sandbox exec "$SANDBOX" --workdir /sandbox --no-tty --timeout 30 -- sha256sum /sandbox/tmp/weave-openclaw-project.tgz | awk '{print $1}')"
  if [ "$host_sha" != "$sandbox_sha" ]; then
    echo "Transferred weave-openclaw archive sha256 mismatch" >&2
    return 1
  fi
  "$NEMOCLAW_BIN" sandbox exec "$SANDBOX" --workdir /sandbox --no-tty --timeout 120 -- bash -lc \
    'rm -rf /sandbox/.openclaw/npm/projects/weave-openclaw && tar -xzf /sandbox/tmp/weave-openclaw-project.tgz -C /sandbox/.openclaw/npm/projects'
  "$NEMOCLAW_BIN" sandbox exec "$SANDBOX" --workdir /sandbox --no-tty --timeout 120 -- openclaw plugins install \
    "$SANDBOX_WEAVE_PROJECT/node_modules/weave-openclaw" --link
}

install_weave_from_npm() {
  "$NEMOCLAW_BIN" sandbox exec "$SANDBOX" --workdir /sandbox --no-tty --timeout 300 -- bash -lc \
    'openclaw plugins install weave-openclaw >/dev/null || openclaw plugins install wandb/weave-openclaw >/dev/null'
}

secret_probe() {
  "$NEMOCLAW_BIN" sandbox exec "$SANDBOX" --workdir /sandbox --no-tty --timeout 60 -- python3 - "$SECRET_FILE" <<'PY' 2>/dev/null || printf false
import json
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("false")
    raise SystemExit
mode = stat.S_IMODE(path.stat().st_mode)
data = json.loads(path.read_text(encoding="utf-8"))
print(str(bool(data.get("wandb", {}).get("apiKey")) and mode & 0o007 == 0).lower())
PY
}

openai_secret_probe() {
  if [ "$SKIP_OPENAI_DIRECT" -eq 1 ]; then
    printf true
    return 0
  fi
  "$NEMOCLAW_BIN" sandbox exec "$SANDBOX" --workdir /sandbox --no-tty --timeout 60 -- python3 - "$SECRET_FILE" <<'PY' 2>/dev/null || printf false
import json
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("false")
    raise SystemExit
mode = stat.S_IMODE(path.stat().st_mode)
data = json.loads(path.read_text(encoding="utf-8"))
print(str(bool(data.get("openai", {}).get("apiKey")) and mode & 0o007 == 0).lower())
PY
}

openai_direct_config_probe() {
  if [ "$SKIP_OPENAI_DIRECT" -eq 1 ]; then
    printf true
    return 0
  fi
  "$NEMOCLAW_BIN" sandbox exec "$SANDBOX" --workdir /sandbox --no-tty --timeout 60 -- python3 - "$OPENCLAW_CONFIG" <<'PY' 2>/dev/null || printf false
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("false")
    raise SystemExit
data = json.loads(path.read_text(encoding="utf-8"))
provider = data.get("models", {}).get("providers", {}).get("openai-direct")
api_key = provider.get("apiKey") if isinstance(provider, dict) else None
models = provider.get("models") if isinstance(provider, dict) else None
model_ids = {str(model.get("id")) for model in models or [] if isinstance(model, dict)}
print(str(bool(
    isinstance(provider, dict)
    and isinstance(api_key, dict)
    and api_key.get("source") == "file"
    and api_key.get("provider") == "nejumi-openai"
    and api_key.get("id") == "/openai/apiKey"
    and "gpt-4.1-nano-2025-04-14" in model_ids
    and "gpt-4.1-mini-2025-04-14" in model_ids
    and "gpt-5.6-luna" in model_ids
)).lower())
PY
}

anthropic_secret_probe() {
  if [ "$SKIP_ANTHROPIC_DIRECT" -eq 1 ]; then
    printf true
    return 0
  fi
  "$NEMOCLAW_BIN" sandbox exec "$SANDBOX" --workdir /sandbox --no-tty --timeout 60 -- python3 - "$SECRET_FILE" <<'PY' 2>/dev/null || printf false
import json
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("false")
    raise SystemExit
mode = stat.S_IMODE(path.stat().st_mode)
data = json.loads(path.read_text(encoding="utf-8"))
print(str(bool(data.get("anthropic", {}).get("apiKey")) and mode & 0o007 == 0).lower())
PY
}

anthropic_direct_config_probe() {
  if [ "$SKIP_ANTHROPIC_DIRECT" -eq 1 ]; then
    printf true
    return 0
  fi
  "$NEMOCLAW_BIN" sandbox exec "$SANDBOX" --workdir /sandbox --no-tty --timeout 60 -- python3 - "$OPENCLAW_CONFIG" <<'PY' 2>/dev/null || printf false
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("false")
    raise SystemExit
data = json.loads(path.read_text(encoding="utf-8"))
provider = data.get("models", {}).get("providers", {}).get("anthropic")
api_key = provider.get("apiKey") if isinstance(provider, dict) else None
models = provider.get("models") if isinstance(provider, dict) else None
model_ids = {str(model.get("id")) for model in models or [] if isinstance(model, dict)}
print(str(bool(
    isinstance(provider, dict)
    and provider.get("baseUrl") == "https://api.anthropic.com"
    and provider.get("api") == "anthropic-messages"
    and isinstance(api_key, dict)
    and api_key.get("source") == "file"
    and api_key.get("provider") == "nejumi-anthropic"
    and api_key.get("id") == "/anthropic/apiKey"
    and "claude-fable-5" in model_ids
)).lower())
PY
}

wandb_inference_config_probe() {
  if [ -z "$WANDB_INFERENCE_MODEL_ID" ]; then
    printf true
    return 0
  fi
  "$NEMOCLAW_BIN" sandbox exec "$SANDBOX" --workdir /sandbox --no-tty --timeout 60 -- python3 - \
    "$OPENCLAW_CONFIG" "$WANDB_INFERENCE_PROVIDER_ID" "$WANDB_INFERENCE_BASE_URL" "$WANDB_INFERENCE_MODEL_ID" "$WANDB_INFERENCE_MODEL_PARAMS_JSON" <<'PY' 2>/dev/null || printf false
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
provider_id, base_url, model_id, params_json = sys.argv[2:6]
expected_params = json.loads(params_json)
if not path.exists():
    print("false")
    raise SystemExit
data = json.loads(path.read_text(encoding="utf-8"))
provider = data.get("models", {}).get("providers", {}).get(provider_id)
api_key = provider.get("apiKey") if isinstance(provider, dict) else None
models = provider.get("models") if isinstance(provider, dict) else None
model = next(
    (
        item
        for item in models or []
        if isinstance(item, dict) and str(item.get("id")) == model_id
    ),
    None,
)
print(str(bool(
    isinstance(provider, dict)
    and provider.get("baseUrl") == base_url
    and isinstance(api_key, dict)
    and api_key.get("source") == "file"
    and api_key.get("provider") == "nejumi-wandb"
    and api_key.get("id") == "/wandb/apiKey"
    and isinstance(model, dict)
    and (model.get("params") or {}) == expected_params
)).lower())
PY
}

policy_added=false
plugin_install_attempted=false
plugin_install_method="none"
secret_written=false
openai_secret_written=false
anthropic_secret_written=false
config_written=false

if [ "$CHECK_ONLY" -eq 0 ]; then
  if [ "$SKIP_POLICY" -eq 0 ]; then
    "$NEMOCLAW_BIN" sandbox policy add "$SANDBOX" --from-file "$POLICY_FILE" --yes >/dev/null
    policy_added=true
  fi

  if [ "$SKIP_PLUGIN_INSTALL" -eq 0 ]; then
    if [ "$FORCE_PLUGIN_INSTALL" -eq 0 ] && [ "$(plugin_probe)" = true ]; then
      plugin_install_method="already_installed"
    elif [ "$WEAVE_PLUGIN_SOURCE" != "npm" ] && [ -d "$LOCAL_WEAVE_PROJECT/node_modules/weave-openclaw" ]; then
      plugin_install_attempted=true
      plugin_install_method="local_project"
      install_weave_from_local_project "$LOCAL_WEAVE_PROJECT"
    elif [ "$WEAVE_PLUGIN_SOURCE" = "local" ]; then
      echo "Local weave-openclaw project not found: $LOCAL_WEAVE_PROJECT" >&2
      exit 1
    else
      plugin_install_attempted=true
      plugin_install_method="npm"
      install_weave_from_npm
    fi
  fi

  if [ "$credential_available" = true ]; then
    printf '%s' "$WANDb_KEY_VALUE" | "$NEMOCLAW_BIN" sandbox exec "$SANDBOX" --workdir /sandbox --no-tty --timeout 60 -- python3 -c 'import json, os, sys; from pathlib import Path; path = Path(sys.argv[1]); secret = sys.stdin.read().strip(); path.parent.mkdir(parents=True, exist_ok=True); data = {}; data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}; data.setdefault("wandb", {})["apiKey"] = secret; tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); os.chmod(tmp, 0o600); tmp.replace(path); os.chmod(path, 0o600)' "$SECRET_FILE"
    secret_written=true
  fi
  if [ "$SKIP_OPENAI_DIRECT" -eq 0 ] && [ "$openai_credential_available" = true ]; then
    printf '%s' "$OPENAI_KEY_VALUE" | "$NEMOCLAW_BIN" sandbox exec "$SANDBOX" --workdir /sandbox --no-tty --timeout 60 -- python3 -c 'import json, os, sys; from pathlib import Path; path = Path(sys.argv[1]); secret = sys.stdin.read().strip(); path.parent.mkdir(parents=True, exist_ok=True); data = {}; data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}; data.setdefault("openai", {})["apiKey"] = secret; tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); os.chmod(tmp, 0o600); tmp.replace(path); os.chmod(path, 0o600)' "$SECRET_FILE"
    openai_secret_written=true
  fi
  if [ "$SKIP_ANTHROPIC_DIRECT" -eq 0 ] && [ "$anthropic_credential_available" = true ]; then
    printf '%s' "$ANTHROPIC_KEY_VALUE" | "$NEMOCLAW_BIN" sandbox exec "$SANDBOX" --workdir /sandbox --no-tty --timeout 60 -- python3 -c 'import json, os, sys; from pathlib import Path; path = Path(sys.argv[1]); secret = sys.stdin.read().strip(); path.parent.mkdir(parents=True, exist_ok=True); data = {}; data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}; data.setdefault("anthropic", {})["apiKey"] = secret; tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); os.chmod(tmp, 0o600); tmp.replace(path); os.chmod(path, 0o600)' "$SECRET_FILE"
    anthropic_secret_written=true
  fi

  "$NEMOCLAW_BIN" sandbox exec "$SANDBOX" --workdir /sandbox --no-tty --timeout 60 -- python3 - \
    "$OPENCLAW_CONFIG" "$SECRET_FILE" "$ENTITY" "$PROJECT" "$SERVICE_NAME" "$AGENT_NAME" "$AGENT_VERSION" "$SKIP_OPENAI_DIRECT" "$SKIP_ANTHROPIC_DIRECT" \
    "$WANDB_INFERENCE_PROVIDER_ID" "$WANDB_INFERENCE_BASE_URL" "$WANDB_INFERENCE_MODEL_ID" "$WANDB_INFERENCE_MAX_TOKENS" \
    "$WANDB_INFERENCE_CONTEXT_WINDOW" "$WANDB_INFERENCE_REASONING" "$WANDB_INFERENCE_MODEL_PARAMS_JSON" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
secret_file, entity, project, service_name, agent_name, agent_version, skip_openai_direct, skip_anthropic_direct = sys.argv[2:10]
(
    wandb_inference_provider_id,
    wandb_inference_base_url,
    wandb_inference_model_id,
    wandb_inference_max_tokens,
    wandb_inference_context_window,
    wandb_inference_reasoning,
    wandb_inference_model_params_json,
) = sys.argv[10:17]
wandb_inference_model_params = json.loads(wandb_inference_model_params_json)
if not isinstance(wandb_inference_model_params, dict):
    raise SystemExit("W&B Inference model params JSON must be an object")
data = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}


def merge_models(existing, additions):
    by_id = {}
    for model in existing or []:
        if isinstance(model, dict) and model.get("id"):
            by_id[str(model["id"])] = model
    for model in additions:
        merged = dict(by_id.get(model["id"], {}))
        merged.update(model)
        by_id[model["id"]] = merged
    return list(by_id.values())


plugins = data.setdefault("plugins", {})
allow = plugins.setdefault("allow", [])
if "weave" not in allow:
    allow.append("weave")
entries = plugins.setdefault("entries", {})
entries["weave"] = {
    "enabled": True,
    "config": {
        "entity": entity,
        "project": project,
        "apiKey": {"source": "file", "provider": "nejumi-wandb", "id": "/wandb/apiKey"},
        "serviceName": service_name,
        "agentName": agent_name,
        "agentVersion": agent_version,
        "agentDescription": "Nejumi 4.5 Taiwan agentic evaluation",
        "captureContent": True,
        "flushIntervalMs": 1000,
    },
    "hooks": {"allowConversationAccess": True},
}
secrets = data.setdefault("secrets", {})
providers = secrets.setdefault("providers", {})
providers["nejumi-wandb"] = {
    "source": "file",
    "path": secret_file,
    "mode": "json",
    "allowInsecurePath": True,
}
if skip_openai_direct != "1":
    providers["nejumi-openai"] = {
        "source": "file",
        "path": secret_file,
        "mode": "json",
        "allowInsecurePath": True,
    }
if skip_anthropic_direct != "1":
    providers["nejumi-anthropic"] = {
        "source": "file",
        "path": secret_file,
        "mode": "json",
        "allowInsecurePath": True,
    }
defaults = secrets.setdefault("defaults", {})
defaults.setdefault("file", "nejumi-wandb")
model_providers = data.setdefault("models", {}).setdefault("providers", {})
if wandb_inference_model_id:
    wandb_inference = model_providers.setdefault(wandb_inference_provider_id, {})
    wandb_inference.pop("agentRuntime", None)
    wandb_inference.update(
        {
            "baseUrl": wandb_inference_base_url,
            "apiKey": {
                "source": "file",
                "provider": "nejumi-wandb",
                "id": "/wandb/apiKey",
            },
            "auth": "api-key",
            "api": "openai-completions",
        }
    )
    model_entry = {
        "id": wandb_inference_model_id,
        "name": wandb_inference_model_id,
        "api": "openai-completions",
        "reasoning": wandb_inference_reasoning == "true",
        "input": ["text"],
        "maxTokens": int(wandb_inference_max_tokens),
    }
    if wandb_inference_context_window:
        model_entry["contextWindow"] = int(wandb_inference_context_window)
    if wandb_inference_model_params:
        model_entry["params"] = wandb_inference_model_params
    wandb_inference["models"] = merge_models(
        wandb_inference.get("models"),
        [model_entry],
    )
if skip_openai_direct != "1":
    openai_direct = model_providers.setdefault("openai-direct", {})
    openai_direct.pop("agentRuntime", None)
    openai_direct.update(
        {
            "baseUrl": "https://api.openai.com/v1",
            "apiKey": {
                "source": "file",
                "provider": "nejumi-openai",
                "id": "/openai/apiKey",
            },
            "auth": "api-key",
            "api": "openai-responses",
        }
    )
    openai_direct["models"] = merge_models(
        openai_direct.get("models"),
        [
            {
                "id": "gpt-4.1-nano-2025-04-14",
                "name": "gpt-4.1-nano-2025-04-14",
                "api": "openai-responses",
                "reasoning": False,
                "input": ["text"],
                "contextWindow": 1047576,
                "maxTokens": 32768,
            },
            {
                "id": "gpt-4.1-mini-2025-04-14",
                "name": "gpt-4.1-mini-2025-04-14",
                "api": "openai-responses",
                "reasoning": False,
                "input": ["text"],
                "contextWindow": 1047576,
                "maxTokens": 32768,
            },
            {
                "id": "gpt-5.6-luna",
                "name": "gpt-5.6-luna",
                "api": "openai-responses",
                "reasoning": True,
                "input": ["text", "image"],
                "contextWindow": 1000000,
                "maxTokens": 65536,
            },
        ],
    )
    for model in openai_direct.get("models") or []:
        if isinstance(model, dict):
            model.pop("agentRuntime", None)
if skip_anthropic_direct != "1":
    anthropic_direct = model_providers.setdefault("anthropic", {})
    anthropic_direct.pop("agentRuntime", None)
    anthropic_direct.update(
        {
            "baseUrl": "https://api.anthropic.com",
            "apiKey": {
                "source": "file",
                "provider": "nejumi-anthropic",
                "id": "/anthropic/apiKey",
            },
            "auth": "api-key",
            "api": "anthropic-messages",
        }
    )
    anthropic_direct["models"] = merge_models(
        anthropic_direct.get("models"),
        [
            {
                "id": "claude-sonnet-4-6",
                "name": "claude-sonnet-4-6",
                "api": "anthropic-messages",
                "reasoning": True,
                "input": ["text", "image"],
                "contextWindow": 1_000_000,
                "maxTokens": 64_000,
            },
            {
                "id": "claude-fable-5",
                "name": "claude-fable-5",
                "api": "anthropic-messages",
                "reasoning": True,
                "input": ["text", "image"],
                "contextWindow": 1000000,
                "maxTokens": 128000,
            },
        ],
    )
    for model in anthropic_direct.get("models") or []:
        if isinstance(model, dict):
            model.pop("agentRuntime", None)
tmp = config_path.with_suffix(config_path.suffix + ".tmp")
tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
tmp.replace(config_path)
PY
  config_written=true
fi

plugin_installed="$(plugin_probe)"
config_ok="$(config_probe)"
policy_ok="$(policy_probe)"
secret_ok="$(secret_probe)"
openai_direct_config_ok="$(openai_direct_config_probe)"
openai_secret_ok="$(openai_secret_probe)"
anthropic_direct_config_ok="$(anthropic_direct_config_probe)"
anthropic_secret_ok="$(anthropic_secret_probe)"
wandb_inference_config_ok="$(wandb_inference_config_probe)"
ok=false
if [ "$plugin_installed" = true ] \
  && [ "$config_ok" = true ] \
  && [ "$policy_ok" = true ] \
  && [ "$openai_direct_config_ok" = true ] \
  && [ "$wandb_inference_config_ok" = true ] \
  && [ "$openai_secret_ok" = true ] \
  && [ "$anthropic_direct_config_ok" = true ] \
  && [ "$anthropic_secret_ok" = true ] \
  && { [ "$credential_available" = false ] || [ "$secret_ok" = true ]; }; then
  ok=true
fi

report_json="$(python3 - "$SANDBOX" "$WANDB_KEY_ENV" "$OPENAI_KEY_ENV" "$ANTHROPIC_KEY_ENV" "$credential_available" "$openai_credential_available" "$anthropic_credential_available" "$policy_added" "$plugin_install_attempted" "$plugin_install_method" "$secret_written" "$openai_secret_written" "$anthropic_secret_written" "$config_written" "$plugin_installed" "$config_ok" "$policy_ok" "$secret_ok" "$openai_direct_config_ok" "$openai_secret_ok" "$anthropic_direct_config_ok" "$anthropic_secret_ok" "$wandb_inference_config_ok" "$ok" "$SECRET_FILE" "$OPENCLAW_CONFIG" "$POLICY_FILE" "$WEAVE_PLUGIN_SOURCE" "$LOCAL_WEAVE_PROJECT" "$SKIP_OPENAI_DIRECT" "$SKIP_ANTHROPIC_DIRECT" "$FORCE_PLUGIN_INSTALL" "$WANDB_INFERENCE_PROVIDER_ID" "$WANDB_INFERENCE_BASE_URL" "$WANDB_INFERENCE_MODEL_ID" "$WANDB_INFERENCE_MAX_TOKENS" "$WANDB_INFERENCE_CONTEXT_WINDOW" "$WANDB_INFERENCE_REASONING" "$WANDB_INFERENCE_MODEL_PARAMS_JSON" <<'PY'
import json
import sys

keys = [
    "sandbox",
    "wandb_key_env",
    "openai_key_env",
    "anthropic_key_env",
    "credential_available",
    "openai_credential_available",
    "anthropic_credential_available",
    "policy_added",
    "plugin_install_attempted",
    "plugin_install_method",
    "secret_written",
    "openai_secret_written",
    "anthropic_secret_written",
    "config_written",
    "plugin_installed",
    "config_ok",
    "policy_ok",
    "secret_ok",
    "openai_direct_config_ok",
    "openai_secret_ok",
    "anthropic_direct_config_ok",
    "anthropic_secret_ok",
    "wandb_inference_config_ok",
    "ok",
    "secret_file",
    "openclaw_config",
    "policy_file",
    "weave_plugin_source",
    "local_weave_project",
    "skip_openai_direct",
    "skip_anthropic_direct",
    "force_plugin_install",
    "wandb_inference_provider_id",
    "wandb_inference_base_url",
    "wandb_inference_model_id",
    "wandb_inference_max_tokens",
    "wandb_inference_context_window",
    "wandb_inference_reasoning",
    "wandb_inference_model_params_json",
]
payload = dict(zip(keys, sys.argv[1:]))
for key in [
    "credential_available",
    "openai_credential_available",
    "anthropic_credential_available",
    "policy_added",
    "plugin_install_attempted",
    "secret_written",
    "openai_secret_written",
    "anthropic_secret_written",
    "config_written",
    "plugin_installed",
    "config_ok",
    "policy_ok",
    "secret_ok",
    "openai_direct_config_ok",
    "openai_secret_ok",
    "anthropic_direct_config_ok",
    "anthropic_secret_ok",
    "wandb_inference_config_ok",
    "ok",
    "skip_openai_direct",
    "skip_anthropic_direct",
    "force_plugin_install",
]:
    payload[key] = str(payload[key]).lower() in {"1", "true", "yes", "on"}
payload["secret_value_in_report"] = False
payload["openai_secret_value_in_report"] = False
payload["anthropic_secret_value_in_report"] = False
payload["weave_agent_name"] = "nejumi-taiwan-openclaw"
payload["wandb_inference_enabled"] = bool(payload.get("wandb_inference_model_id"))
payload["wandb_inference_model_params"] = json.loads(
    payload.pop("wandb_inference_model_params_json")
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
)"

if [ -n "$JSON_OUT" ]; then
  mkdir -p "$(dirname "$JSON_OUT")"
  printf '%s\n' "$report_json" > "$JSON_OUT"
fi
printf '%s\n' "$report_json"

if [ "$ok" != true ]; then
  exit 1
fi
