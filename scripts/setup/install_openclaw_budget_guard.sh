#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PLUGIN_DIR="${ROOT_DIR}/openclaw-plugins/nejumi-budget-guard"
TURN_GUARD_PATCH="${ROOT_DIR}/scripts/setup/patch_openclaw_turn_budget_guard.py"
NEMOCLAW_RUNTIME_PATCH="${ROOT_DIR}/scripts/setup/patch_nemoclaw_openclaw_runtime.py"
SANDBOX="${NEMOCLAW_SANDBOX:-}"
PLUGIN_ID="nejumi-budget-guard"

patch_config() {
  local config_path="$1"
  python3 - "${config_path}" "${PLUGIN_ID}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]).expanduser()
plugin_id = sys.argv[2]
if not path.exists():
    raise SystemExit(f"OpenClaw config not found: {path}")
config = json.loads(path.read_text(encoding="utf-8"))
plugins = config.setdefault("plugins", {})
allow = plugins.setdefault("allow", [])
if isinstance(allow, list) and plugin_id not in allow:
    allow.append(plugin_id)
entries = plugins.setdefault("entries", {})
entry = entries.setdefault(plugin_id, {})
if not isinstance(entry, dict):
    raise SystemExit(f"plugins.entries.{plugin_id} must be an object in {path}")
entry["enabled"] = True
hooks = entry.setdefault("hooks", {})
if not isinstance(hooks, dict):
    raise SystemExit(f"plugins.entries.{plugin_id}.hooks must be an object in {path}")
hooks["allowConversationAccess"] = True
hooks.setdefault("timeoutMs", 1000)
path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Patched OpenClaw config for {plugin_id}: {path}")
PY
}

openclaw plugins install "${PLUGIN_DIR}" --force
patch_config "${OPENCLAW_CONFIG_PATH:-${HOME}/.openclaw/openclaw.json}"
python3 "${TURN_GUARD_PATCH}"

if [[ -n "${SANDBOX}" ]]; then
  ARCHIVE="$(mktemp -t nejumi-budget-guard.XXXXXX.tgz)"
  PATCH_ARCHIVE="$(mktemp -t nejumi-turn-budget-guard-patch.XXXXXX.py)"
  cp "${TURN_GUARD_PATCH}" "${PATCH_ARCHIVE}"
  tar -C "${PLUGIN_DIR}/.." -czf "${ARCHIVE}" "$(basename "${PLUGIN_DIR}")"
  REMOTE_ARCHIVE="/sandbox/tmp/nejumi-budget-guard.tgz"
  REMOTE_DIR="/sandbox/tmp/nejumi-budget-guard-plugin"
  REMOTE_PATCH="/sandbox/tmp/patch_openclaw_turn_budget_guard.py"
  if nemoclaw sandbox cp --help >/dev/null 2>&1; then
    nemoclaw sandbox cp "${ARCHIVE}" "${SANDBOX}:${REMOTE_ARCHIVE}"
    nemoclaw sandbox cp "${PATCH_ARCHIVE}" "${SANDBOX}:${REMOTE_PATCH}"
  else
    base64 "${ARCHIVE}" | nemoclaw sandbox exec "${SANDBOX}" --no-tty --timeout 60 -- bash -lc "mkdir -p /sandbox/tmp && base64 -d > ${REMOTE_ARCHIVE}"
    base64 "${PATCH_ARCHIVE}" | nemoclaw sandbox exec "${SANDBOX}" --no-tty --timeout 60 -- bash -lc "mkdir -p /sandbox/tmp && base64 -d > ${REMOTE_PATCH}"
  fi
  nemoclaw sandbox exec "${SANDBOX}" --no-tty --timeout 120 -- bash -lc "rm -rf ${REMOTE_DIR} && mkdir -p ${REMOTE_DIR} && tar -C ${REMOTE_DIR} -xzf ${REMOTE_ARCHIVE} && openclaw plugins install ${REMOTE_DIR}/nejumi-budget-guard --force"
  nemoclaw sandbox exec "${SANDBOX}" --no-tty --timeout 30 -- python3 -c 'import json, pathlib; p=pathlib.Path("/sandbox/.openclaw/openclaw.json"); c=json.loads(p.read_text(encoding="utf-8")); plugins=c.setdefault("plugins", {}); allow=plugins.setdefault("allow", []); pid="nejumi-budget-guard"; allow.append(pid) if isinstance(allow, list) and pid not in allow else None; entries=plugins.setdefault("entries", {}); entry=entries.setdefault(pid, {}); entry["enabled"]=True; hooks=entry.setdefault("hooks", {}); hooks["allowConversationAccess"]=True; hooks.setdefault("timeoutMs", 1000); p.write_text(json.dumps(c, ensure_ascii=False, indent=2)+"\n", encoding="utf-8"); print(f"Patched OpenClaw config for {pid}: {p}")'
  nemoclaw sandbox exec "${SANDBOX}" --no-tty --timeout 240 -- bash -lc "set -euo pipefail; mkdir -p /sandbox/.npm-global/lib/node_modules /sandbox/.npm-global/bin; if [ ! -f /sandbox/.npm-global/lib/node_modules/openclaw/package.json ]; then cp -a /usr/local/lib/node_modules/openclaw /sandbox/.npm-global/lib/node_modules/openclaw; fi; ln -sfn /sandbox/.npm-global/lib/node_modules/openclaw/openclaw.mjs /sandbox/.npm-global/bin/openclaw; python3 ${REMOTE_PATCH} --openclaw-package-dir /sandbox/.npm-global/lib/node_modules/openclaw"
  python3 "${NEMOCLAW_RUNTIME_PATCH}" --sandbox "${SANDBOX}" --json
  rm -f "${ARCHIVE}"
  rm -f "${PATCH_ARCHIVE}"
fi
