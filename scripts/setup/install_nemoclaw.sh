#!/usr/bin/env bash
# Install or verify NVIDIA NemoClaw for the Taiwan leaderboard agentic harness.
#
# This script deliberately separates read-only checks from installation and
# onboarding. NVIDIA's installer requires explicit third-party software
# acceptance, so installation refuses to run unless the acceptance flag is
# provided here as well.

set -euo pipefail

NEMOCLAW_INSTALL_URL="${NEMOCLAW_INSTALL_URL:-https://www.nvidia.com/nemoclaw.sh}"
NEMOCLAW_INSTALL_REF="${NEMOCLAW_INSTALL_REF:-lkg}"
NEMOCLAW_INSTALLER_SHA256="${NEMOCLAW_INSTALLER_SHA256:-}"
NEMOCLAW_INSTALLER_SIGNATURE="${NEMOCLAW_INSTALLER_SIGNATURE:-}"
NEMOCLAW_INSTALLER_REVIEW_JSON="${NEMOCLAW_INSTALLER_REVIEW_JSON:-}"
NEMOCLAW_INSTALLER_LOCK_JSON="${NEMOCLAW_INSTALLER_LOCK_JSON:-scripts/setup/nemoclaw_installer_lock.json}"
NEMOCLAW_INSTALLER_PROVENANCE_NOTE="Installer integrity is not verified by this script; operator review is required before install/onboard."
SANDBOX_NAME="${NEMOCLAW_SANDBOX_NAME:-nejumi-taiwan}"
PROVIDER="${NEMOCLAW_PROVIDER:-openai}"
GATEWAY_PORT="${NEMOCLAW_GATEWAY_PORT:-}"
POLICY_TIER="${NEMOCLAW_POLICY_TIER:-restricted}"
POLICY_TIER_ALLOWED_VALUES=(restricted balanced open)
MODEL="${NEMOCLAW_MODEL:-}"
ENV_FILE="${NEMOCLAW_ENV_FILE:-}"
ENV_FILE_LOADED=0
ENV_FILE_LOAD_ERROR=""
PROVIDER_ENDPOINT_URL="${NEMOCLAW_ENDPOINT_URL:-}"
PROVIDER_KEY_ENV="${NEMOCLAW_PROVIDER_KEY_ENV:-}"
OPENAI_CANARY_MANIFEST="${OPENAI_CANARY_MANIFEST:-configs/taiwan_openai_canary_models.yaml}"
OPENAI_CANARY_FULL_DIR="${OPENAI_CANARY_FULL_DIR:-configs/taiwan_full/generated_openai_canary}"
OPENAI_CANARY_NONAGENTIC_DIR="${OPENAI_CANARY_NONAGENTIC_DIR:-configs/taiwan_full/generated_openai_canary_nonagentic}"
OPENAI_CANARY_AGENTIC_DIR="${OPENAI_CANARY_AGENTIC_DIR:-configs/taiwan_full/generated_openai_canary_agentic_nemoclaw}"
OPENAI_CANARY_AGENTIC_AGGREGATE_DIR="${OPENAI_CANARY_AGENTIC_AGGREGATE_DIR:-configs/taiwan_full/generated_openai_canary_agentic_aggregate}"
NEMOCLAW_OPENCLAW_CONFIG_PATH="${NEMOCLAW_OPENCLAW_CONFIG_PATH:-/sandbox/.openclaw/openclaw.json}"
CHECK_ONLY=0
INSTALL=0
ONBOARD=0
ACCEPT_THIRD_PARTY=0
FRESH=0
RECREATE_SANDBOX=0
INSTALL_ATTEMPTED=0
ONBOARD_ATTEMPTED=0
ONBOARD_SKIPPED=0
INSTALLER_INTEGRITY_VERIFIED=0
INSTALLER_PROVENANCE_LOCKED=0
INSTALLER_REVIEW_VERIFIED=0
INSTALL_RC=""
ONBOARD_RC=""
INSTALL_LOG_PATH=""
ONBOARD_LOG_PATH=""
JSON_PATH=""

usage() {
  cat <<'EOF'
Usage:
  scripts/setup/install_nemoclaw.sh --check-only
  scripts/setup/install_nemoclaw.sh --install --installer-lock-json PATH --installer-sha256 VALUE --installer-review-json PATH --yes-i-accept-third-party-software
  scripts/setup/install_nemoclaw.sh --install --onboard --installer-lock-json PATH --installer-sha256 VALUE --installer-review-json PATH --yes-i-accept-third-party-software [options]

Options:
  --check-only                         Run read-only prerequisite and status checks.
  --install                            Install/update NemoClaw through NVIDIA's installer.
  --onboard                            Run non-interactive `nemoclaw onboard` after install.
  --yes-i-accept-third-party-software  Required for install/onboard; forwarded to NemoClaw.
  --sandbox NAME                       Sandbox name. Default: nejumi-taiwan.
  --provider VALUE                     NemoClaw provider. Default: openai.
                                      Use build only when the inference endpoint
                                      should be NVIDIA hosted endpoints.
  --gateway-port PORT                  Host port for the OpenShell gateway.
                                      Useful when the default 8080 is occupied.
  --env-file PATH                       Load KEY=VALUE entries from a dotenv file
                                      before provider preflight/onboarding.
  --endpoint-url URL                    Endpoint URL for --provider custom.
                                      Exported to NEMOCLAW_ENDPOINT_URL for
                                      onboarding and provider preflight.
  --provider-key-env NAME               Env var to use as provider key for
                                      --provider custom. The secret value is
                                      not printed or written to JSON.
  --model VALUE                        NemoClaw model ID. Optional.
  --policy-tier VALUE                  restricted|balanced|open. Default: restricted.
  --install-ref VALUE                  Git ref/tag for NVIDIA/NemoClaw installer. Default: lkg.
  --installer-sha256 VALUE             Required for --install; verifies downloaded installer before execution.
  --installer-review-json PATH         Required for --install; reviewed installer evidence from review_nemoclaw_installer.py.
  --installer-lock-json PATH           Pinned installer lock used by the generated review command.
  --installer-signature VALUE          Optional signature/provenance reference recorded in setup JSON.
  --fresh                              Discard interrupted onboarding state.
  --recreate-sandbox                   Force sandbox recreation during onboarding.
  --json PATH                          Write a machine-readable setup/check report.
  -h, --help                           Show this help.

Notes:
  - No model inference is launched by this script.
  - `--check-only` never installs software and does not require acceptance.
  - Installation first verifies --installer-review-json, then downloads the installer,
    verifies --installer-sha256, and executes the verified file.
EOF
}

log() {
  printf '[nemoclaw-setup] %s\n' "$*"
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    log "missing: $cmd"
    return 1
  fi
  log "$cmd: $(command -v "$cmd")"
  return 0
}

json_bool() {
  if "$@" >/dev/null 2>&1; then
    printf 'true'
  else
    printf 'false'
  fi
}

json_string() {
  local value="$1"
  python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$value"
}

json_array() {
  python3 - "$@" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1:]))
PY
}

json_int_or_null() {
  local value="$1"
  if [ -n "$value" ]; then
    printf '%s' "$value"
  else
    printf 'null'
  fi
}

json_flag() {
  local value="$1"
  if [ "$value" -eq 1 ]; then
    printf 'true'
  else
    printf 'false'
  fi
}

operation_failure_json() {
  local path="$1"
  local returncode="$2"
  python3 - "$path" "$returncode" <<'PY'
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1]) if sys.argv[1] else None
returncode = sys.argv[2]
text = ""
if path and path.is_file():
    text = path.read_text(encoding="utf-8", errors="replace")
lower = text.lower()
kind = ""
detail = ""
if returncode and returncode != "0":
    if "http 429" in lower or "exceeded your current quota" in lower:
        kind = "provider_quota"
        detail = "provider validation returned HTTP 429/quota"
    elif "port " in lower and "not available" in lower:
        kind = "gateway_port_unavailable"
        match = re.search(r"port\\s+([0-9]+)\\s+is\\s+not\\s+available", lower)
        detail = f"gateway port {match.group(1)} unavailable" if match else "gateway port unavailable"
    elif "api_key" in lower and ("required" in lower or "missing" in lower):
        kind = "provider_auth"
        detail = "provider credential missing or rejected"
    elif "sandbox not found" in lower:
        kind = "sandbox_missing"
        detail = "sandbox not found"
    else:
        kind = "unknown"
        detail = "operation failed; inspect log_path"
print(json.dumps({"failure_kind": kind, "failure_detail": detail}, ensure_ascii=False))
PY
}

provider_preflight_json() {
  python3 - "$PROVIDER" "$MODEL" "$PROVIDER_KEY_ENV" <<'PY'
import json
import os
import re
import sys

provider = (sys.argv[1] or "").strip()
model = (sys.argv[2] or "").strip()
provider_key_env = (sys.argv[3] or "").strip()
normalized = provider.lower()
aliases = {
    "cloud": "build",
    "nvidia": "build",
    "compatible": "custom",
    "compatible-endpoint": "custom",
    "openai-api": "openai",
    "nim": "nim-local",
}
key = aliases.get(normalized, normalized)

provider_requirements = {
    "build": {
        "credential_envs": ["NVIDIA_API_KEY", "NEMOCLAW_PROVIDER_KEY"],
        "endpoint_envs": [],
        "model_required": False,
        "nvidia_hosted_endpoint": True,
        "label": "NVIDIA hosted endpoints",
    },
    "openai": {
        "credential_envs": ["OPENAI_API_KEY", "NEMOCLAW_PROVIDER_KEY"],
        "endpoint_envs": [],
        "model_required": False,
        "nvidia_hosted_endpoint": False,
        "label": "OpenAI hosted inference",
    },
    "custom": {
        "credential_envs": [
            "NEMOCLAW_PROVIDER_KEY",
            "COMPATIBLE_API_KEY",
            "OPENAI_COMPATIBLE_API_KEY",
            "VLLM_API_KEY",
            "LITELLM_MASTER_KEY",
            "LITELLM_API_KEY",
        ],
        "endpoint_envs": [
            "NEMOCLAW_ENDPOINT_URL",
            "OPENAI_COMPATIBLE_BASE_URL",
            "OPENAI_COMPATIBLE_API_BASE",
            "OPENAI_COMPATIBLE_ENDPOINT_URL",
            "OPENAI_COMPATIBLE_ENDPOINT",
            "VLLM_ENDPOINT_URL",
            "VLLM_BASE_URL",
        ],
        "model_required": True,
        "nvidia_hosted_endpoint": False,
        "label": "OpenAI-compatible endpoint",
    },
    "anthropic": {
        "credential_envs": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "NEMOCLAW_PROVIDER_KEY"],
        "endpoint_envs": [],
        "model_required": False,
        "nvidia_hosted_endpoint": False,
        "label": "Anthropic hosted inference",
    },
    "anthropiccompatible": {
        "credential_envs": ["COMPATIBLE_ANTHROPIC_API_KEY", "NEMOCLAW_PROVIDER_KEY"],
        "endpoint_envs": ["NEMOCLAW_ANTHROPIC_ENDPOINT_URL"],
        "model_required": True,
        "nvidia_hosted_endpoint": False,
        "label": "Anthropic-compatible endpoint",
    },
    "gemini": {
        "credential_envs": ["GEMINI_API_KEY", "GOOGLE_API_KEY", "NEMOCLAW_PROVIDER_KEY"],
        "endpoint_envs": [],
        "model_required": False,
        "nvidia_hosted_endpoint": False,
        "label": "Gemini hosted inference",
    },
    "routed": {
        "credential_envs": ["NVIDIA_API_KEY", "NEMOCLAW_PROVIDER_KEY"],
        "endpoint_envs": [],
        "model_required": False,
        "nvidia_hosted_endpoint": True,
        "label": "NemoClaw model router",
    },
}
local_providers = {
    "ollama",
    "install-ollama",
    "install-windows-ollama",
    "start-windows-ollama",
    "vllm",
    "install-vllm",
    "nim-local",
}
requirements = provider_requirements.get(key)
if requirements is None:
    requirements = {
        "credential_envs": [],
        "endpoint_envs": [],
        "model_required": False,
        "nvidia_hosted_endpoint": False,
        "label": "local or provider-specific inference",
    }
credential_envs = list(requirements["credential_envs"])
if provider_key_env and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", provider_key_env):
    credential_envs = [provider_key_env, *[name for name in credential_envs if name != provider_key_env]]
endpoint_envs = list(requirements["endpoint_envs"])
present_credential_envs = [name for name in credential_envs if os.environ.get(name)]
present_endpoint_envs = [name for name in endpoint_envs if os.environ.get(name)]
credential_required = bool(credential_envs) and key not in local_providers
endpoint_required = bool(endpoint_envs)
model_required = bool(requirements["model_required"])
missing = []
if credential_required and not present_credential_envs:
    missing.append("credential")
if endpoint_required and not present_endpoint_envs:
    missing.append("endpoint")
if model_required and not model:
    missing.append("model")
payload = {
    "provider": provider,
    "normalized_provider": key,
    "label": requirements["label"],
    "reads_dotenv_file": False,
    "will_launch_probe": False,
    "will_launch_benchmark_inference": False,
    "nvidia_hosted_endpoint": bool(requirements["nvidia_hosted_endpoint"]),
    "nvidia_api_key_required": bool(requirements["nvidia_hosted_endpoint"]),
    "credential_required": credential_required,
    "credential_envs": credential_envs,
    "provider_key_env": provider_key_env,
    "credential_available": bool(present_credential_envs) or not credential_required,
    "credential_present_envs": present_credential_envs,
    "endpoint_required": endpoint_required,
    "endpoint_envs": endpoint_envs,
    "endpoint_requirement_mode": "any" if endpoint_envs else "none",
    "endpoint_available": bool(present_endpoint_envs) or not endpoint_required,
    "endpoint_present_envs": present_endpoint_envs,
    "model_required": model_required,
    "model_configured": bool(model),
    "missing": missing,
    "ready_for_noninteractive_onboard_preflight": not missing,
    "notes": [
        "This preflight only checks local environment shape and never prints secret values.",
        "It does not prove provider quota or endpoint health; NemoClaw onboarding may still run a provider smoke check.",
    ],
}
print(json.dumps(payload, ensure_ascii=False))
PY
}

load_env_file() {
  [ -z "$ENV_FILE" ] && return 0
  if [ ! -f "$ENV_FILE" ]; then
    ENV_FILE_LOAD_ERROR="env file not found: $ENV_FILE"
    return 2
  fi
  local exports
  if ! exports="$(
    python3 - "$ENV_FILE" <<'PY'
import re
import shlex
import sys
from pathlib import Path

path = Path(sys.argv[1])
for line in path.read_text(encoding="utf-8").splitlines():
    raw = line.strip()
    if not raw or raw.startswith("#") or "=" not in raw:
        continue
    key, value = raw.split("=", 1)
    key = key.strip()
    if key.startswith("export "):
        key = key[len("export "):].strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        continue
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    print(f"export {key}={shlex.quote(value)}")
PY
  )"; then
    ENV_FILE_LOAD_ERROR="env file could not be parsed: $ENV_FILE"
    return 2
  fi
  eval "$exports"
  ENV_FILE_LOADED=1
  return 0
}

apply_cli_env_overrides() {
  if [ -n "$PROVIDER_ENDPOINT_URL" ]; then
    export NEMOCLAW_ENDPOINT_URL="$PROVIDER_ENDPOINT_URL"
  fi
  if [ -n "$PROVIDER_KEY_ENV" ]; then
    export NEMOCLAW_PROVIDER_KEY_ENV="$PROVIDER_KEY_ENV"
  fi
}

valid_provider_key_env() {
  [ -z "$PROVIDER_KEY_ENV" ] && return 0
  [[ "$PROVIDER_KEY_ENV" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]
}

installer_review_expected_sha256() {
  if [ -n "$NEMOCLAW_INSTALLER_SHA256" ]; then
    printf '%s' "$NEMOCLAW_INSTALLER_SHA256"
    return 0
  fi
  if [ -f "$NEMOCLAW_INSTALLER_LOCK_JSON" ]; then
    python3 - "$NEMOCLAW_INSTALLER_LOCK_JSON" <<'PY'
import json
import re
import sys

try:
    value = str(json.load(open(sys.argv[1], encoding="utf-8")).get("sha256") or "").strip().lower()
except Exception:
    value = ""
print(value if re.fullmatch(r"[0-9a-f]{64}", value) else "")
PY
    return 0
  fi
  printf ''
}

valid_policy_tier() {
  case "$POLICY_TIER" in
    restricted|balanced|open) return 0 ;;
    *) return 1 ;;
  esac
}

valid_gateway_port() {
  [ -z "$GATEWAY_PORT" ] && return 0
  [[ "$GATEWAY_PORT" =~ ^[0-9]+$ ]] || return 1
  [ "$GATEWAY_PORT" -ge 1 ] && [ "$GATEWAY_PORT" -le 65535 ]
}

command_path_or_empty() {
  local cmd="$1"
  command -v "$cmd" 2>/dev/null || true
}

nemoclaw_sandbox_configured() {
  local sandbox_name="$1"
  command -v nemoclaw >/dev/null 2>&1 || return 1
  local payload
  payload="$(nemoclaw list --json 2>/dev/null || true)"
  [ -n "$payload" ] || return 1
  python3 - "$sandbox_name" "$payload" <<'PY'
import json
import sys

sandbox = sys.argv[1]
try:
    payload = json.loads(sys.argv[2])
except Exception:
    raise SystemExit(1)

items = payload.get("sandboxes")
if not isinstance(items, list):
    raise SystemExit(1)
for item in items:
    if isinstance(item, str) and item == sandbox:
        raise SystemExit(0)
    if isinstance(item, dict) and item.get("name") == sandbox:
        raise SystemExit(0)
raise SystemExit(1)
PY
}

operation_log_path() {
  local name="$1"
  local dir base stem
  if [ -n "$JSON_PATH" ]; then
    dir="$(dirname "$JSON_PATH")"
    base="$(basename "$JSON_PATH")"
    stem="${base%.json}"
  else
    dir="temp"
    stem="nemoclaw_setup_$(date -u '+%Y%m%dT%H%M%SZ')"
  fi
  mkdir -p "$dir"
  printf '%s/%s.%s.log' "$dir" "$stem" "$name"
}

write_check_json() {
  local path="$1"
  local exit_code="$2"
  [ -z "$path" ] && return 0
  mkdir -p "$(dirname "$path")"
  local curl_path git_path docker_path node_path npm_path zstd_path nemoclaw_path openshell_path
  curl_path="$(command_path_or_empty curl)"
  git_path="$(command_path_or_empty git)"
  docker_path="$(command_path_or_empty docker)"
  node_path="$(command_path_or_empty node)"
  npm_path="$(command_path_or_empty npm)"
  zstd_path="$(command_path_or_empty zstd)"
  nemoclaw_path="$(command_path_or_empty nemoclaw)"
  openshell_path="$(command_path_or_empty openshell)"

  local docker_info_ok=false
  if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    docker_info_ok=true
  fi

  local host_prerequisites_ok=true
  local runtime_installed=true
  local sandbox_configured=false
  local missing_required_commands=()
  if [ -z "$curl_path" ]; then
    host_prerequisites_ok=false
    missing_required_commands+=("curl")
  fi
  if [ -z "$git_path" ]; then
    host_prerequisites_ok=false
    missing_required_commands+=("git")
  fi
  if [ -z "$docker_path" ]; then
    host_prerequisites_ok=false
    missing_required_commands+=("docker")
  elif [ "$docker_info_ok" != true ]; then
    host_prerequisites_ok=false
    missing_required_commands+=("docker:info")
  fi
  if [ -z "$node_path" ]; then
    host_prerequisites_ok=false
    missing_required_commands+=("node")
  fi
  if [ -z "$npm_path" ]; then
    host_prerequisites_ok=false
    missing_required_commands+=("npm")
  fi
  if [ -z "$nemoclaw_path" ]; then
    runtime_installed=false
    missing_required_commands+=("nemoclaw")
  fi
  if [ -z "$openshell_path" ]; then
    runtime_installed=false
    missing_required_commands+=("openshell")
  fi
  if [ "$runtime_installed" = true ] && nemoclaw_sandbox_configured "$SANDBOX_NAME"; then
    sandbox_configured=true
  fi

  local installer_review_command install_command onboard_command install_and_onboard_command production_install_and_onboard_command post_check_command post_install_verification_command canary_readiness_command adoption_check_command production_gate_command
  local installer_sha_arg installer_review_sha_arg gateway_flag env_file_flag endpoint_flag provider_key_env_flag
  installer_sha_arg="${NEMOCLAW_INSTALLER_SHA256:-REVIEWED_INSTALLER_SHA256}"
  installer_review_sha_arg="$(installer_review_expected_sha256)"
  installer_review_sha_arg="${installer_review_sha_arg:-REVIEWED_INSTALLER_SHA256}"
  gateway_flag=""
  if [ -n "$GATEWAY_PORT" ]; then
    gateway_flag=" --gateway-port $GATEWAY_PORT"
  fi
  env_file_flag=""
  if [ -n "$ENV_FILE" ]; then
    env_file_flag=" --env-file $ENV_FILE"
  fi
  endpoint_flag=""
  if [ -n "$PROVIDER_ENDPOINT_URL" ]; then
    endpoint_flag=" --endpoint-url $PROVIDER_ENDPOINT_URL"
  fi
  provider_key_env_flag=""
  if [ -n "$PROVIDER_KEY_ENV" ]; then
    provider_key_env_flag=" --provider-key-env $PROVIDER_KEY_ENV"
  fi
  installer_review_command="uv run python scripts/setup/review_nemoclaw_installer.py --url $NEMOCLAW_INSTALL_URL --install-ref $NEMOCLAW_INSTALL_REF --expected-sha256 $installer_review_sha_arg --lock-json $NEMOCLAW_INSTALLER_LOCK_JSON --json temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json --markdown temp/nemoclaw_installer_review_YYYYMMDDTHHMM.md"
  install_command="scripts/setup/install_nemoclaw.sh --install --install-ref $NEMOCLAW_INSTALL_REF --installer-lock-json $NEMOCLAW_INSTALLER_LOCK_JSON --installer-sha256 $installer_sha_arg --installer-review-json temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json$env_file_flag --yes-i-accept-third-party-software --json temp/nemoclaw_install_YYYYMMDDTHHMM.json"
  onboard_command="scripts/setup/install_nemoclaw.sh --onboard --sandbox $SANDBOX_NAME --provider $PROVIDER$gateway_flag$env_file_flag$endpoint_flag$provider_key_env_flag --policy-tier $POLICY_TIER --yes-i-accept-third-party-software --json temp/nemoclaw_onboard_YYYYMMDDTHHMM.json"
  install_and_onboard_command="scripts/setup/install_nemoclaw.sh --install --onboard --install-ref $NEMOCLAW_INSTALL_REF --installer-lock-json $NEMOCLAW_INSTALLER_LOCK_JSON --installer-sha256 $installer_sha_arg --installer-review-json temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json --sandbox $SANDBOX_NAME --provider $PROVIDER$gateway_flag$env_file_flag$endpoint_flag$provider_key_env_flag --policy-tier $POLICY_TIER --yes-i-accept-third-party-software --json temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.json"
  if [ -n "$NEMOCLAW_INSTALLER_SIGNATURE" ]; then
    install_command="$install_command --installer-signature $NEMOCLAW_INSTALLER_SIGNATURE"
    install_and_onboard_command="$install_and_onboard_command --installer-signature $NEMOCLAW_INSTALLER_SIGNATURE"
  fi
  if [ -n "$MODEL" ]; then
    onboard_command="$onboard_command --model $MODEL"
    install_and_onboard_command="$install_and_onboard_command --model $MODEL"
  fi
  production_install_and_onboard_command="$install_and_onboard_command"
  post_check_command="scripts/setup/install_nemoclaw.sh --check-only --sandbox $SANDBOX_NAME$gateway_flag$env_file_flag$endpoint_flag$provider_key_env_flag --json temp/nemoclaw_setup_check_YYYYMMDDTHHMM.json"
  post_install_verification_command="uv run python scripts/setup/verify_nemoclaw_post_install.py --sandbox $SANDBOX_NAME --nemoclaw-openclaw-config-path $NEMOCLAW_OPENCLAW_CONFIG_PATH --canary-manifest $OPENAI_CANARY_MANIFEST --generated-full-dir $OPENAI_CANARY_FULL_DIR --generated-nonagentic-dir $OPENAI_CANARY_NONAGENTIC_DIR --generated-agentic-dir $OPENAI_CANARY_AGENTIC_DIR --generated-agentic-aggregate-dir $OPENAI_CANARY_AGENTIC_AGGREGATE_DIR --json temp/nemoclaw_post_install_verification_YYYYMMDDTHHMM.json --markdown temp/nemoclaw_post_install_verification_YYYYMMDDTHHMM.md --fail-on-failed"
  canary_readiness_command="uv run python scripts/tools/check_taiwan_canary_readiness.py --manifest $OPENAI_CANARY_MANIFEST --generated-full-dir $OPENAI_CANARY_FULL_DIR --generated-nonagentic-dir $OPENAI_CANARY_NONAGENTIC_DIR --generated-agentic-dir $OPENAI_CANARY_AGENTIC_DIR --generated-agentic-aggregate-dir $OPENAI_CANARY_AGENTIC_AGGREGATE_DIR --require-nemoclaw --nemoclaw-sandbox $SANDBOX_NAME --json outputs/taiwan_full_eval/openai_canary_readiness_nemoclaw.json"
  adoption_check_command="uv run python scripts/tools/check_taiwan_nemoclaw_adoption.py --setup-json temp/nemoclaw_setup_check_YYYYMMDDTHHMM.json --readiness-json outputs/taiwan_full_eval/openai_canary_readiness_nemoclaw.json --sandbox $SANDBOX_NAME --agentic-config-glob $OPENAI_CANARY_AGENTIC_DIR/*.yaml --json temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.json --markdown temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.md --fail-on-not-adoptable"
  production_gate_command="uv run python scripts/tools/run_taiwan_production_readiness_gate.py --report-json outputs/taiwan_full_eval/taiwan_production_readiness_report.json --fail-on-not-ready"

  cat >"$path" <<EOF
{
  "schema_version": 1,
  "ok": $([ "$exit_code" -eq 0 ] && printf true || printf false),
  "exit_code": $exit_code,
  "generated_at": $(json_string "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"),
  "sandbox": $(json_string "$SANDBOX_NAME"),
  "provider": $(json_string "$PROVIDER"),
  "model": $(json_string "$MODEL"),
  "gateway_port": $([ -n "$GATEWAY_PORT" ] && json_string "$GATEWAY_PORT" || printf 'null'),
  "env_file": $(json_string "$ENV_FILE"),
  "env_file_loaded": $(json_flag "$ENV_FILE_LOADED"),
  "env_file_load_error": $(json_string "$ENV_FILE_LOAD_ERROR"),
  "provider_key_env": $(json_string "$PROVIDER_KEY_ENV"),
  "provider_preflight": $(provider_preflight_json),
  "sandbox_configured": $sandbox_configured,
  "policy_tier": $(json_string "$POLICY_TIER"),
  "policy_tier_allowed_values": $(json_array "${POLICY_TIER_ALLOWED_VALUES[@]}"),
  "policy_tier_valid": $(valid_policy_tier && printf true || printf false),
  "host_prerequisites_ok": $host_prerequisites_ok,
  "runtime_installed": $runtime_installed,
  "missing_required_commands": $(json_array "${missing_required_commands[@]}"),
  "install_requested": $([ "$INSTALL" -eq 1 ] && printf true || printf false),
  "onboard_requested": $([ "$ONBOARD" -eq 1 ] && printf true || printf false),
  "check_only": $([ "$CHECK_ONLY" -eq 1 ] && printf true || printf false),
  "accepted_third_party_software": $([ "$ACCEPT_THIRD_PARTY" -eq 1 ] && printf true || printf false),
  "third_party_software": {
    "name": "NVIDIA NemoClaw",
    "vendor": "NVIDIA",
    "repository_url": "https://github.com/NVIDIA/NemoClaw",
    "documentation_url": "https://docs.nvidia.com/nemoclaw/user-guide/openclaw/get-started/quickstart",
    "installer_url": $(json_string "$NEMOCLAW_INSTALL_URL"),
    "install_ref": $(json_string "$NEMOCLAW_INSTALL_REF"),
    "installer_sha256": $(json_string "$NEMOCLAW_INSTALLER_SHA256"),
    "installer_signature": $(json_string "$NEMOCLAW_INSTALLER_SIGNATURE"),
    "installer_lock_json": $(json_string "$NEMOCLAW_INSTALLER_LOCK_JSON"),
    "installer_review_json": $(json_string "$NEMOCLAW_INSTALLER_REVIEW_JSON"),
    "installer_review_verified": $(json_flag "$INSTALLER_REVIEW_VERIFIED"),
    "installer_integrity_verified": $(json_flag "$INSTALLER_INTEGRITY_VERIFIED"),
    "installer_provenance_locked": $(json_flag "$INSTALLER_PROVENANCE_LOCKED"),
    "installer_provenance_note": $(json_string "$NEMOCLAW_INSTALLER_PROVENANCE_NOTE"),
    "acceptance_required": true,
    "acceptance_flag": "--yes-i-accept-third-party-software",
    "accepted": $([ "$ACCEPT_THIRD_PARTY" -eq 1 ] && printf true || printf false),
    "install_or_onboard_requested": $([ "$INSTALL" -eq 1 ] || [ "$ONBOARD" -eq 1 ] && printf true || printf false),
    "operator_review_required_before_install": true
  },
  "operation_results": {
    "install": {
      "requested": $([ "$INSTALL" -eq 1 ] && printf true || printf false),
      "attempted": $([ "$INSTALL_ATTEMPTED" -eq 1 ] && printf true || printf false),
      "returncode": $(json_int_or_null "$INSTALL_RC"),
      "log_path": $(json_string "$INSTALL_LOG_PATH")
    },
    "onboard": {
      "requested": $([ "$ONBOARD" -eq 1 ] && printf true || printf false),
      "attempted": $([ "$ONBOARD_ATTEMPTED" -eq 1 ] && printf true || printf false),
      "skipped": $([ "$ONBOARD_SKIPPED" -eq 1 ] && printf true || printf false),
      "returncode": $(json_int_or_null "$ONBOARD_RC"),
      "log_path": $(json_string "$ONBOARD_LOG_PATH"),
      "failure": $(operation_failure_json "$ONBOARD_LOG_PATH" "$ONBOARD_RC")
    }
  },
  "commands": {
    "curl": {"available": $([ -n "$curl_path" ] && printf true || printf false), "path": $(json_string "$curl_path"), "required": true},
    "git": {"available": $([ -n "$git_path" ] && printf true || printf false), "path": $(json_string "$git_path"), "required": true},
    "docker": {"available": $([ -n "$docker_path" ] && printf true || printf false), "path": $(json_string "$docker_path"), "required": true, "info_ok": $docker_info_ok},
    "node": {"available": $([ -n "$node_path" ] && printf true || printf false), "path": $(json_string "$node_path"), "required": true},
    "npm": {"available": $([ -n "$npm_path" ] && printf true || printf false), "path": $(json_string "$npm_path"), "required": true},
    "zstd": {"available": $([ -n "$zstd_path" ] && printf true || printf false), "path": $(json_string "$zstd_path"), "required": false},
    "nemoclaw": {"available": $([ -n "$nemoclaw_path" ] && printf true || printf false), "path": $(json_string "$nemoclaw_path"), "required": true},
    "openshell": {"available": $([ -n "$openshell_path" ] && printf true || printf false), "path": $(json_string "$openshell_path"), "required": true}
  },
  "setup_plan": {
    "will_launch_model_inference": false,
    "will_launch_benchmark_inference": false,
    "onboard_may_validate_provider_endpoint": true,
    "sandbox_configured": $sandbox_configured,
    "sandbox_readiness_required": true,
    "provider_notes": {
      "openai": "Uses OPENAI_API_KEY for OpenAI-hosted inference.",
      "build": "Uses NVIDIA_API_KEY for NVIDIA hosted endpoints.",
      "custom": "Uses an OpenAI-compatible endpoint configured through NEMOCLAW_ENDPOINT_URL/endpoint aliases and COMPATIBLE_API_KEY/OpenAI-compatible key aliases."
    },
    "provider": $(json_string "$PROVIDER"),
    "model": $(json_string "$MODEL"),
    "gateway_port": $([ -n "$GATEWAY_PORT" ] && json_string "$GATEWAY_PORT" || printf 'null'),
    "env_file": $(json_string "$ENV_FILE"),
    "env_file_loaded": $(json_flag "$ENV_FILE_LOADED"),
    "env_file_load_error": $(json_string "$ENV_FILE_LOAD_ERROR"),
    "provider_key_env": $(json_string "$PROVIDER_KEY_ENV"),
    "provider_preflight": $(provider_preflight_json),
    "install_or_onboard_requires_explicit_acceptance": true,
    "acceptance_flag": "--yes-i-accept-third-party-software",
    "third_party_software_name": "NVIDIA NemoClaw",
    "installer_url": $(json_string "$NEMOCLAW_INSTALL_URL"),
    "install_ref": $(json_string "$NEMOCLAW_INSTALL_REF"),
    "installer_sha256": $(json_string "$NEMOCLAW_INSTALLER_SHA256"),
    "installer_signature": $(json_string "$NEMOCLAW_INSTALLER_SIGNATURE"),
    "installer_lock_json": $(json_string "$NEMOCLAW_INSTALLER_LOCK_JSON"),
    "installer_review_json": $(json_string "$NEMOCLAW_INSTALLER_REVIEW_JSON"),
    "installer_review_verified": $(json_flag "$INSTALLER_REVIEW_VERIFIED"),
    "installer_integrity_verified": $(json_flag "$INSTALLER_INTEGRITY_VERIFIED"),
    "installer_provenance_locked": $(json_flag "$INSTALLER_PROVENANCE_LOCKED"),
    "installer_provenance_note": $(json_string "$NEMOCLAW_INSTALLER_PROVENANCE_NOTE"),
    "acceptance_ledger_fields": [
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
      "operation_results.onboard.log_path"
    ],
    "policy_tier": $(json_string "$POLICY_TIER"),
    "policy_tier_allowed_values": $(json_array "${POLICY_TIER_ALLOWED_VALUES[@]}"),
    "policy_tier_valid": $(valid_policy_tier && printf true || printf false),
    "installer_review_command": $(json_string "$installer_review_command"),
    "install_command": $(json_string "$install_command"),
    "onboard_command": $(json_string "$onboard_command"),
    "install_and_onboard_command": $(json_string "$install_and_onboard_command"),
    "production_install_and_onboard_command": $(json_string "$production_install_and_onboard_command"),
    "post_install_check_command": $(json_string "$post_check_command"),
    "post_install_verification_command": $(json_string "$post_install_verification_command"),
    "canary_readiness_command": $(json_string "$canary_readiness_command"),
    "adoption_check_command": $(json_string "$adoption_check_command"),
    "production_readiness_command": $(json_string "$production_gate_command"),
    "operator_sequence": [
      {
        "step": "setup_check",
        "command": $(json_string "$post_check_command"),
        "expected_evidence_path": "temp/nemoclaw_setup_check_YYYYMMDDTHHMM.json",
        "requires_external_action": false,
        "required": true
      },
      {
        "step": "installer_review",
        "command": $(json_string "$installer_review_command"),
        "expected_evidence_path": "temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json",
        "requires_external_action": false,
        "required": true
      },
      {
        "step": "install_and_onboard",
        "command": $(json_string "$production_install_and_onboard_command"),
        "expected_evidence_path": "temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.json",
        "requires_external_action": true,
        "required": true
      },
      {
        "step": "post_install_verification",
        "command": $(json_string "$post_install_verification_command"),
        "expected_evidence_path": "temp/nemoclaw_post_install_verification_YYYYMMDDTHHMM.json",
        "requires_external_action": false,
        "required": true
      },
      {
        "step": "canary_readiness",
        "command": $(json_string "$canary_readiness_command"),
        "expected_evidence_path": "outputs/taiwan_full_eval/openai_canary_readiness_nemoclaw.json",
        "requires_external_action": false,
        "required": true
      },
      {
        "step": "adoption_check",
        "command": $(json_string "$adoption_check_command"),
        "expected_evidence_path": "temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.json",
        "requires_external_action": false,
        "required": true
      },
      {
        "step": "production_readiness",
        "command": $(json_string "$production_gate_command"),
        "expected_evidence_path": "outputs/taiwan_full_eval/taiwan_production_readiness_report.json",
        "requires_external_action": false,
        "required": true
      }
    ],
    "expected_evidence_paths": [
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
      "outputs/taiwan_full_eval/taiwan_production_readiness_report.json"
    ]
  },
  "next_action": $(json_string "$(
    if [ "$runtime_installed" != true ]; then
      printf 'Install/onboard NeMoClaw and OpenShell with explicit third-party acceptance, then rerun --check-only.'
    elif [ "$sandbox_configured" != true ]; then
      printf 'Run the install/onboard command with explicit third-party acceptance and a working provider, then rerun --check-only and post-install verification.'
    elif [ "$exit_code" -eq 0 ]; then
      printf 'Run the OpenAI-direct canary readiness command from setup_plan.canary_readiness_command to verify sandbox OpenClaw preflight.'
    else
      printf 'Fix failed host prerequisites, then rerun --check-only.'
    fi
  )")
}
EOF
}

print_system_status() {
  log "system"
  uname -a || true
  if [ -f /etc/os-release ]; then
    sed -n '1,8p' /etc/os-release
  fi
  log "disk"
  df -h . || true
  log "memory"
  free -h || true
}

check_prereqs() {
  local ok=0
  require_cmd curl || ok=1
  require_cmd git || ok=1
  require_cmd docker || ok=1
  require_cmd node || ok=1
  require_cmd npm || ok=1
  require_cmd zstd || true

  log "versions"
  node --version 2>/dev/null || true
  npm --version 2>/dev/null || true
  docker --version 2>/dev/null || true
  if ! docker info >/tmp/nejumi-nemoclaw-docker-info.txt 2>/tmp/nejumi-nemoclaw-docker-info.err; then
    log "docker info failed:"
    sed -n '1,80p' /tmp/nejumi-nemoclaw-docker-info.err || true
    ok=1
  else
    docker info --format '{{.ServerVersion}} {{.OSType}} {{.Architecture}}' || true
  fi

  if command -v nemoclaw >/dev/null 2>&1; then
    log "nemoclaw: $(command -v nemoclaw)"
    nemoclaw --version || true
    log "nemoclaw status --json"
    nemoclaw status --json >/tmp/nejumi-nemoclaw-status.json 2>/tmp/nejumi-nemoclaw-status.err || true
    if [ -s /tmp/nejumi-nemoclaw-status.json ]; then
      sed -n '1,120p' /tmp/nejumi-nemoclaw-status.json
    else
      sed -n '1,120p' /tmp/nejumi-nemoclaw-status.err || true
    fi
    log "nemoclaw list --json"
    nemoclaw list --json >/tmp/nejumi-nemoclaw-list.json 2>/tmp/nejumi-nemoclaw-list.err || true
    if [ -s /tmp/nejumi-nemoclaw-list.json ]; then
      sed -n '1,120p' /tmp/nejumi-nemoclaw-list.json
    else
      sed -n '1,120p' /tmp/nejumi-nemoclaw-list.err || true
    fi
  else
    log "nemoclaw: missing"
    ok=1
  fi

  if command -v openshell >/dev/null 2>&1; then
    log "openshell: $(command -v openshell)"
    openshell --version || true
    openshell status || true
  else
    log "openshell: missing"
    ok=1
  fi

  return "$ok"
}

validate_installer_review_json() {
  if [ -z "$NEMOCLAW_INSTALLER_REVIEW_JSON" ]; then
    cat >&2 <<'EOF'
Refusing to install NemoClaw without installer review evidence.
Rerun review first:
  uv run python scripts/setup/review_nemoclaw_installer.py --url https://www.nvidia.com/nemoclaw.sh --install-ref lkg --expected-sha256 REVIEWED_INSTALLER_SHA256 --lock-json scripts/setup/nemoclaw_installer_lock.json --json temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json --markdown temp/nemoclaw_installer_review_YYYYMMDDTHHMM.md
Then install with:
  scripts/setup/install_nemoclaw.sh --install --installer-lock-json scripts/setup/nemoclaw_installer_lock.json --installer-sha256 REVIEWED_INSTALLER_SHA256 --installer-review-json temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json --yes-i-accept-third-party-software
EOF
    return 2
  fi
  python3 - "$NEMOCLAW_INSTALLER_REVIEW_JSON" "$NEMOCLAW_INSTALLER_SHA256" "$NEMOCLAW_INSTALL_URL" "$NEMOCLAW_INSTALL_REF" "$NEMOCLAW_INSTALLER_LOCK_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_sha = sys.argv[2].strip().lower()
expected_url = sys.argv[3]
expected_ref = sys.argv[4]
expected_lock = sys.argv[5]

def path_key(value: str) -> str:
    if not value:
        return ""
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return str(candidate.resolve(strict=False))

try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception as exc:
    print(f"installer review JSON is not readable: {exc}", file=sys.stderr)
    raise SystemExit(2)

errors = []
if payload.get("schema_version") != 1:
    errors.append("schema_version must be 1")
if payload.get("ok") is not True:
    errors.append("ok must be true")
if payload.get("status") != "reviewed":
    errors.append("status must be reviewed")
if str(payload.get("sha256", "")).lower() != expected_sha:
    errors.append("sha256 does not match --installer-sha256")
if payload.get("installer_url") != expected_url:
    errors.append("installer_url does not match install URL")
if payload.get("install_ref") != expected_ref:
    errors.append("install_ref does not match --install-ref")
if expected_lock:
    if payload.get("lock_verified") is not True:
        errors.append("lock_verified must be true")
    if path_key(str(payload.get("lock_json") or "")) != path_key(expected_lock):
        errors.append("lock_json does not match --installer-lock-json")
for field in (
    "will_execute_installer",
    "will_install_or_onboard",
    "will_launch_model_inference",
    "will_query_wandb",
):
    if payload.get(field) is not False:
        errors.append(f"{field} must be false")
if errors:
    for error in errors:
        print(f"installer review JSON invalid: {error}", file=sys.stderr)
    raise SystemExit(2)
PY
}

install_nemoclaw() {
  if [ "$ACCEPT_THIRD_PARTY" -ne 1 ]; then
    cat >&2 <<'EOF'
Refusing to install NemoClaw without explicit third-party software acceptance.
Rerun with:
  uv run python scripts/setup/review_nemoclaw_installer.py --url https://www.nvidia.com/nemoclaw.sh --install-ref lkg --expected-sha256 REVIEWED_INSTALLER_SHA256 --lock-json scripts/setup/nemoclaw_installer_lock.json --json temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json --markdown temp/nemoclaw_installer_review_YYYYMMDDTHHMM.md
  scripts/setup/install_nemoclaw.sh --install --installer-lock-json scripts/setup/nemoclaw_installer_lock.json --installer-sha256 REVIEWED_INSTALLER_SHA256 --installer-review-json temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json --yes-i-accept-third-party-software
EOF
    return 2
  fi
  if [ -z "$NEMOCLAW_INSTALLER_SHA256" ]; then
    cat >&2 <<'EOF'
Refusing to install NemoClaw without a reviewed installer SHA-256.
Rerun with:
  uv run python scripts/setup/review_nemoclaw_installer.py --url https://www.nvidia.com/nemoclaw.sh --install-ref lkg --expected-sha256 REVIEWED_INSTALLER_SHA256 --lock-json scripts/setup/nemoclaw_installer_lock.json --json temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json --markdown temp/nemoclaw_installer_review_YYYYMMDDTHHMM.md
  scripts/setup/install_nemoclaw.sh --install --installer-lock-json scripts/setup/nemoclaw_installer_lock.json --installer-sha256 REVIEWED_INSTALLER_SHA256 --installer-review-json temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json --yes-i-accept-third-party-software
EOF
    return 2
  fi
  if ! python3 - "$NEMOCLAW_INSTALLER_SHA256" <<'PY'
import re
import sys
sys.exit(0 if re.fullmatch(r"[0-9a-fA-F]{64}", sys.argv[1]) else 1)
PY
	  then
	    echo "Invalid --installer-sha256: expected 64 hex characters" >&2
	    return 2
	  fi
	  if ! validate_installer_review_json; then
	    return 2
	  fi
	  INSTALLER_REVIEW_VERIFIED=1

	  local installer_path actual_sha expected_sha rc
  installer_path="$(mktemp)"
  log "downloading NemoClaw installer url=${NEMOCLAW_INSTALL_URL}"
  curl -fsSL "$NEMOCLAW_INSTALL_URL" -o "$installer_path"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    rm -f "$installer_path"
    return "$rc"
  fi
  actual_sha="$(
    python3 - "$installer_path" <<'PY'
import hashlib
import sys
path = sys.argv[1]
digest = hashlib.sha256()
with open(path, "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
  )"
  expected_sha="$(printf '%s' "$NEMOCLAW_INSTALLER_SHA256" | tr 'A-F' 'a-f')"
  if [ "$actual_sha" != "$expected_sha" ]; then
    echo "NemoClaw installer SHA-256 mismatch: expected ${expected_sha}, got ${actual_sha}" >&2
    rm -f "$installer_path"
    return 3
  fi
  INSTALLER_INTEGRITY_VERIFIED=1
  INSTALLER_PROVENANCE_LOCKED=1

  log "installing NemoClaw ref=${NEMOCLAW_INSTALL_REF} from verified installer sha256=${actual_sha}"
  NEMOCLAW_INSTALL_REF="$NEMOCLAW_INSTALL_REF" \
    NEMOCLAW_NON_INTERACTIVE=1 \
    NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1 \
    bash "$installer_path" --non-interactive --yes-i-accept-third-party-software
  rc=$?
  rm -f "$installer_path"
  return "$rc"
}

onboard_nemoclaw() {
  if [ "$ACCEPT_THIRD_PARTY" -ne 1 ]; then
    cat >&2 <<'EOF'
Refusing to onboard NemoClaw without explicit third-party software acceptance.
Rerun with --yes-i-accept-third-party-software.
EOF
    return 2
  fi
  if ! command -v nemoclaw >/dev/null 2>&1; then
    log "nemoclaw command is missing; run --install first"
    return 1
  fi

  local args=(onboard --non-interactive --yes-i-accept-third-party-software --name "$SANDBOX_NAME")
  [ "$FRESH" -eq 1 ] && args+=(--fresh)
  [ "$RECREATE_SANDBOX" -eq 1 ] && args+=(--recreate-sandbox)

  local env_args=(
    "NEMOCLAW_PROVIDER=$PROVIDER"
    "NEMOCLAW_SANDBOX_NAME=$SANDBOX_NAME"
    "NEMOCLAW_POLICY_TIER=$POLICY_TIER"
    "NEMOCLAW_POLICY_MODE=suggested"
    "NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1"
    "NEMOCLAW_MODEL=$MODEL"
  )
  if [ -n "$GATEWAY_PORT" ]; then
    env_args+=("NEMOCLAW_GATEWAY_PORT=$GATEWAY_PORT")
  fi

  local normalized_provider provider_key endpoint_url
  normalized_provider="$(printf '%s' "$PROVIDER" | tr 'A-Z' 'a-z')"
  case "$normalized_provider" in
    custom|compatible|compatible-endpoint)
      if [ -n "$PROVIDER_KEY_ENV" ]; then
        provider_key="${!PROVIDER_KEY_ENV:-}"
      else
        provider_key=""
      fi
      provider_key="${provider_key:-${NEMOCLAW_PROVIDER_KEY:-${COMPATIBLE_API_KEY:-${OPENAI_COMPATIBLE_API_KEY:-${VLLM_API_KEY:-${LITELLM_MASTER_KEY:-${LITELLM_API_KEY:-}}}}}}}"
      endpoint_url="${NEMOCLAW_ENDPOINT_URL:-${OPENAI_COMPATIBLE_BASE_URL:-${OPENAI_COMPATIBLE_API_BASE:-${OPENAI_COMPATIBLE_ENDPOINT_URL:-${OPENAI_COMPATIBLE_ENDPOINT:-${VLLM_ENDPOINT_URL:-${VLLM_BASE_URL:-}}}}}}}"
      if [ -n "$provider_key" ]; then
        env_args+=("NEMOCLAW_PROVIDER_KEY=$provider_key")
        env_args+=("COMPATIBLE_API_KEY=$provider_key")
      fi
      if [ -n "$endpoint_url" ]; then
        env_args+=("NEMOCLAW_ENDPOINT_URL=$endpoint_url")
      fi
      ;;
  esac

  log "onboarding sandbox=${SANDBOX_NAME} provider=${PROVIDER} policy_tier=${POLICY_TIER} gateway_port=${GATEWAY_PORT:-default}"
  env "${env_args[@]}" nemoclaw "${args[@]}"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --check-only) CHECK_ONLY=1 ;;
    --install) INSTALL=1 ;;
    --onboard) ONBOARD=1 ;;
    --yes-i-accept-third-party-software) ACCEPT_THIRD_PARTY=1 ;;
    --sandbox)
      SANDBOX_NAME="$2"
      shift
      ;;
    --provider)
      PROVIDER="$2"
      shift
      ;;
    --gateway-port)
      GATEWAY_PORT="$2"
      shift
      ;;
    --model)
      MODEL="$2"
      shift
      ;;
    --env-file)
      ENV_FILE="$2"
      shift
      ;;
    --endpoint-url)
      PROVIDER_ENDPOINT_URL="$2"
      shift
      ;;
    --provider-key-env)
      PROVIDER_KEY_ENV="$2"
      shift
      ;;
    --policy-tier)
      POLICY_TIER="$2"
      shift
      ;;
    --install-ref)
      NEMOCLAW_INSTALL_REF="$2"
      shift
      ;;
	    --installer-sha256)
	      NEMOCLAW_INSTALLER_SHA256="$2"
	      shift
	      ;;
	    --installer-review-json)
	      NEMOCLAW_INSTALLER_REVIEW_JSON="$2"
	      shift
	      ;;
	    --installer-lock-json)
	      NEMOCLAW_INSTALLER_LOCK_JSON="$2"
	      shift
	      ;;
	    --installer-signature)
	      NEMOCLAW_INSTALLER_SIGNATURE="$2"
	      shift
      ;;
    --fresh) FRESH=1 ;;
    --recreate-sandbox) RECREATE_SANDBOX=1 ;;
    --json)
      JSON_PATH="$2"
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
  shift
done

if [ "$CHECK_ONLY" -eq 0 ] && [ "$INSTALL" -eq 0 ] && [ "$ONBOARD" -eq 0 ]; then
  usage
  exit 2
fi

if ! load_env_file; then
  echo "$ENV_FILE_LOAD_ERROR" >&2
  write_check_json "$JSON_PATH" 2
  exit 2
fi
apply_cli_env_overrides

if ! valid_provider_key_env; then
  echo "Invalid --provider-key-env: $PROVIDER_KEY_ENV (expected shell environment variable name)" >&2
  write_check_json "$JSON_PATH" 2
  exit 2
fi

if ! valid_policy_tier; then
  echo "Invalid --policy-tier: $POLICY_TIER (allowed: ${POLICY_TIER_ALLOWED_VALUES[*]})" >&2
  write_check_json "$JSON_PATH" 2
  exit 2
fi

if ! valid_gateway_port; then
  echo "Invalid --gateway-port: $GATEWAY_PORT (allowed: 1-65535)" >&2
  write_check_json "$JSON_PATH" 2
  exit 2
fi

print_system_status
if [ "$CHECK_ONLY" -eq 1 ]; then
  if check_prereqs; then
    rc=0
  else
    rc=$?
  fi
  write_check_json "$JSON_PATH" "$rc"
  exit "$rc"
fi

check_prereqs || true
operation_rc=0
if [ "$INSTALL" -eq 1 ]; then
  INSTALL_ATTEMPTED=1
  INSTALL_LOG_PATH="$(operation_log_path install)"
  set +e
  install_nemoclaw >"$INSTALL_LOG_PATH" 2>&1
  INSTALL_RC=$?
  set -e
  log "install log: $INSTALL_LOG_PATH"
  if [ "$INSTALL_RC" -ne 0 ]; then
    operation_rc="$INSTALL_RC"
  fi
fi
if [ "$ONBOARD" -eq 1 ]; then
  if [ "$INSTALL" -eq 1 ] && [ -n "$INSTALL_RC" ] && [ "$INSTALL_RC" -ne 0 ]; then
    log "skipping onboard because install failed with return code ${INSTALL_RC}"
    ONBOARD_SKIPPED=1
  else
    ONBOARD_ATTEMPTED=1
    ONBOARD_LOG_PATH="$(operation_log_path onboard)"
    set +e
    onboard_nemoclaw >"$ONBOARD_LOG_PATH" 2>&1
    ONBOARD_RC=$?
    set -e
    log "onboard log: $ONBOARD_LOG_PATH"
    if [ "$ONBOARD_RC" -ne 0 ] && [ "$operation_rc" -eq 0 ]; then
      operation_rc="$ONBOARD_RC"
    fi
  fi
fi
if check_prereqs; then
  check_rc=0
else
  check_rc=$?
fi
if [ "$operation_rc" -ne 0 ]; then
  rc="$operation_rc"
else
  rc="$check_rc"
fi
write_check_json "$JSON_PATH" "$rc"
exit "$rc"
