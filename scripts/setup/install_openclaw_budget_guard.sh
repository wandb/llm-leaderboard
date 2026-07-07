#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PLUGIN_DIR="${ROOT_DIR}/openclaw-plugins/nejumi-budget-guard"
SANDBOX="${NEMOCLAW_SANDBOX:-}"

openclaw plugins install "${PLUGIN_DIR}" --force

if [[ -n "${SANDBOX}" ]]; then
  ARCHIVE="$(mktemp -t nejumi-budget-guard.XXXXXX.tgz)"
  tar -C "${PLUGIN_DIR}/.." -czf "${ARCHIVE}" "$(basename "${PLUGIN_DIR}")"
  REMOTE_ARCHIVE="/sandbox/tmp/nejumi-budget-guard.tgz"
  REMOTE_DIR="/sandbox/tmp/nejumi-budget-guard-plugin"
  if nemoclaw sandbox cp --help >/dev/null 2>&1; then
    nemoclaw sandbox cp "${ARCHIVE}" "${SANDBOX}:${REMOTE_ARCHIVE}"
  else
    base64 "${ARCHIVE}" | nemoclaw sandbox exec "${SANDBOX}" --no-tty --timeout 60 -- bash -lc "mkdir -p /sandbox/tmp && base64 -d > ${REMOTE_ARCHIVE}"
  fi
  nemoclaw sandbox exec "${SANDBOX}" --no-tty --timeout 120 -- bash -lc "rm -rf ${REMOTE_DIR} && mkdir -p ${REMOTE_DIR} && tar -C ${REMOTE_DIR} -xzf ${REMOTE_ARCHIVE} && openclaw plugins install ${REMOTE_DIR}/nejumi-budget-guard --force"
  rm -f "${ARCHIVE}"
fi
