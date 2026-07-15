#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Configure OpenRouter GLM-5.2 for OpenClaw inside a NeMoClaw sandbox.

Usage:
  scripts/setup/configure_nemoclaw_openrouter.sh [options]

Options:
  --sandbox NAME              NeMoClaw sandbox name. Default: nejumi-taiwan.
  --nemoclaw-bin PATH         NeMoClaw executable. Default: nemoclaw.
  --env-file PATH             Env file to source. Default: .env.
  --openrouter-key-env NAME   Env var containing the OpenRouter API key. Default: OPENROUTER_API_KEY.
  --policy-file PATH          NeMoClaw OpenRouter egress policy YAML.
  --secret-file PATH          Sandbox secret JSON path. Default: /sandbox/.openclaw/nejumi_secrets.json.
  --openclaw-config PATH      Sandbox OpenClaw config path. Default: /sandbox/.openclaw/openclaw.json.
  --openclaw-model-params-json JSON
                              JSON object written to the OpenClaw GLM model params.
                              Default: {}. Set provider routing from YAML/CLI when needed.
  --check-only                Do not mutate; inspect current sandbox state only.
  --skip-policy               Do not add the OpenRouter egress policy.
  --json PATH                 Write a machine-readable report.
  -h, --help                  Show this help.

The OpenRouter key is sent to the sandbox over stdin and stored as a file
SecretRef source. The secret value is not included in command arguments or JSON
reports.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SANDBOX="nejumi-taiwan"
NEMOCLAW_BIN="nemoclaw"
ENV_FILE="$REPO_ROOT/.env"
OPENROUTER_KEY_ENV="OPENROUTER_API_KEY"
POLICY_FILE="$REPO_ROOT/configs/nemoclaw/policies/openrouter_inference.yaml"
SECRET_FILE="/sandbox/.openclaw/nejumi_secrets.json"
OPENCLAW_CONFIG="/sandbox/.openclaw/openclaw.json"
OPENCLAW_MODEL_PARAMS_JSON='{}'
CHECK_ONLY=0
SKIP_POLICY=0
JSON_OUT=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --sandbox) SANDBOX="$2"; shift 2 ;;
    --nemoclaw-bin) NEMOCLAW_BIN="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --openrouter-key-env) OPENROUTER_KEY_ENV="$2"; shift 2 ;;
    --policy-file) POLICY_FILE="$2"; shift 2 ;;
    --secret-file) SECRET_FILE="$2"; shift 2 ;;
    --openclaw-config) OPENCLAW_CONFIG="$2"; shift 2 ;;
    --openclaw-model-params-json) OPENCLAW_MODEL_PARAMS_JSON="$2"; shift 2 ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --skip-policy) SKIP_POLICY=1; shift ;;
    --json) JSON_OUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! "$OPENROUTER_KEY_ENV" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
  echo "Invalid --openrouter-key-env: $OPENROUTER_KEY_ENV" >&2
  exit 2
fi

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

OPENROUTER_KEY_VALUE="${!OPENROUTER_KEY_ENV:-}"
credential_available=false
if [ -n "$OPENROUTER_KEY_VALUE" ]; then
  credential_available=true
fi

run_nemoclaw() {
  "$NEMOCLAW_BIN" sandbox exec "$SANDBOX" --workdir /sandbox --no-tty --timeout "${2:-120}" -- bash -lc "$1"
}

policy_probe() {
  local status
  status="$("$NEMOCLAW_BIN" sandbox status "$SANDBOX" 2>/dev/null || true)"
  if grep -q "host: openrouter.ai" <<<"$status"; then
    printf true
  else
    printf false
  fi
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
print(str(bool(data.get("openrouter", {}).get("apiKey")) and mode & 0o007 == 0).lower())
PY
}

config_probe() {
  "$NEMOCLAW_BIN" sandbox exec "$SANDBOX" --workdir /sandbox --no-tty --timeout 60 -- python3 - "$OPENCLAW_CONFIG" "$OPENCLAW_MODEL_PARAMS_JSON" <<'PY' 2>/dev/null || printf false
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_params = json.loads(sys.argv[2])
if not path.exists():
    print("false")
    raise SystemExit
data = json.loads(path.read_text(encoding="utf-8"))
provider = data.get("models", {}).get("providers", {}).get("openrouter-direct")
api_key = provider.get("apiKey") if isinstance(provider, dict) else None
models = provider.get("models") if isinstance(provider, dict) else None
glm = next(
    (
        model
        for model in models or []
        if isinstance(model, dict) and str(model.get("id")) == "z-ai/glm-5.2"
    ),
    None,
)
print(str(bool(
    isinstance(provider, dict)
    and provider.get("baseUrl") == "https://openrouter.ai/api/v1"
    and isinstance(api_key, dict)
    and api_key.get("source") == "file"
    and api_key.get("provider") == "nejumi-openrouter"
    and api_key.get("id") == "/openrouter/apiKey"
    and isinstance(glm, dict)
    and (glm.get("params") or {}) == expected_params
)).lower())
PY
}

policy_added=false
secret_written=false
config_written=false

if [ "$CHECK_ONLY" -eq 0 ]; then
  if [ "$SKIP_POLICY" -eq 0 ]; then
    "$NEMOCLAW_BIN" sandbox policy add "$SANDBOX" --from-file "$POLICY_FILE" --yes >/dev/null
    policy_added=true
  fi

  if [ "$credential_available" = true ]; then
    printf '%s' "$OPENROUTER_KEY_VALUE" | "$NEMOCLAW_BIN" sandbox exec "$SANDBOX" --workdir /sandbox --no-tty --timeout 60 -- python3 -c 'import json, os, sys; from pathlib import Path; path = Path(sys.argv[1]); secret = sys.stdin.read().strip(); path.parent.mkdir(parents=True, exist_ok=True); data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}; data.setdefault("openrouter", {})["apiKey"] = secret; tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); os.chmod(tmp, 0o660); tmp.replace(path); os.chmod(path, 0o660)' "$SECRET_FILE"
    secret_written=true
  fi

  "$NEMOCLAW_BIN" sandbox exec "$SANDBOX" --workdir /sandbox --no-tty --timeout 60 -- python3 - \
    "$OPENCLAW_CONFIG" "$SECRET_FILE" "$OPENCLAW_MODEL_PARAMS_JSON" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
secret_file = sys.argv[2]
model_params = json.loads(sys.argv[3])
if not isinstance(model_params, dict):
    raise SystemExit("OpenClaw model params JSON must be an object")
data = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

secrets = data.setdefault("secrets", {})
providers = secrets.setdefault("providers", {})
providers["nejumi-openrouter"] = {
    "source": "file",
    "path": secret_file,
    "mode": "json",
    "allowInsecurePath": True,
}

model_providers = data.setdefault("models", {}).setdefault("providers", {})
openrouter_direct = model_providers.setdefault("openrouter-direct", {})
openrouter_direct.pop("agentRuntime", None)
openrouter_direct.update(
    {
        "baseUrl": "https://openrouter.ai/api/v1",
        "apiKey": {
            "source": "file",
            "provider": "nejumi-openrouter",
            "id": "/openrouter/apiKey",
        },
        "auth": "api-key",
        "api": "openai-completions",
    }
)

models = []
seen = set()
for model in openrouter_direct.get("models") or []:
    if isinstance(model, dict) and model.get("id") and model["id"] != "z-ai/glm-5.2":
        if model["id"] not in seen:
            models.append(model)
            seen.add(model["id"])
glm = {
    "id": "z-ai/glm-5.2",
    "name": "Z.ai GLM 5.2 via OpenRouter",
    "reasoning": True,
    "input": ["text"],
    "contextWindow": 1048576,
    "maxTokens": 32768,
    "cost": {
        "input": 0.95,
        "output": 3,
        "cacheRead": 0.18,
        "cacheWrite": 0,
    },
}
if model_params:
    glm["params"] = model_params
models.append(glm)
openrouter_direct["models"] = models

tmp = config_path.with_suffix(config_path.suffix + ".tmp")
tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
tmp.replace(config_path)
PY
  config_written=true
fi

policy_ok="$(policy_probe)"
secret_ok="$(secret_probe)"
config_ok="$(config_probe)"
ok=false
if [ "$policy_ok" = true ] && [ "$config_ok" = true ] && { [ "$credential_available" = false ] || [ "$secret_ok" = true ]; }; then
  ok=true
fi

report_json="$(python3 - "$SANDBOX" "$OPENROUTER_KEY_ENV" "$credential_available" "$policy_added" "$secret_written" "$config_written" "$policy_ok" "$secret_ok" "$config_ok" "$ok" "$SECRET_FILE" "$OPENCLAW_CONFIG" "$POLICY_FILE" "$SKIP_POLICY" "$OPENCLAW_MODEL_PARAMS_JSON" <<'PY'
import json
import sys

keys = [
    "sandbox",
    "openrouter_key_env",
    "credential_available",
    "policy_added",
    "secret_written",
    "config_written",
    "policy_ok",
    "secret_ok",
    "config_ok",
    "ok",
    "secret_file",
    "openclaw_config",
    "policy_file",
    "skip_policy",
    "openclaw_model_params_json",
]
payload = dict(zip(keys, sys.argv[1:]))
for key in [
    "credential_available",
    "policy_added",
    "secret_written",
    "config_written",
    "policy_ok",
    "secret_ok",
    "config_ok",
    "ok",
    "skip_policy",
]:
    payload[key] = str(payload[key]).lower() in {"1", "true", "yes", "on"}
payload["secret_value_in_report"] = False
payload["registered_models"] = ["openrouter-direct/z-ai/glm-5.2"]
payload["openclaw_model_params"] = json.loads(payload.pop("openclaw_model_params_json"))
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
