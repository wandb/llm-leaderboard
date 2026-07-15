#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/setup/install_deepswe_sandbox_deps.sh [options]

Install or verify the common toolchain expected by the Taiwan Agentic
SWE-Assorted DeepSWE High subset inside a NeMoClaw sandbox.

The script does not fetch Go from the public internet by default. It copies
/usr/local/go from one of the selected DeepSWE task Docker images into:

  /sandbox/.deepswe-tools/go

and exposes it through:

  /sandbox/.deepswe-tools/go/bin

It also links go/gofmt into /usr/local/bin inside the sandbox container when
Docker access to the OpenShell container is available. OpenClaw tool execution
does not reliably inherit the launcher PATH, so the standard PATH exposure is
required for model-visible test commands.

Options:
  --sandbox NAME          NeMoClaw sandbox name. Default: nejumi-taiwan
  --nemoclaw-bin PATH     NeMoClaw executable. Default: nemoclaw
  --tasks-root PATH       DeepSWE tasks root. Default: external/deep-swe/tasks
  --task-names-file PATH  JSON list of selected DeepSWE task names.
                          Default: data/taiwan/deepswe/subsets/essential_anchored_high_8_cost_trimmed_cap_safe_lang_balanced_task_names.json
  --go-image IMAGE        Explicit Docker image to use as the Go source.
  --no-pull               Do not docker pull if no selected Go image is present locally.
  --check-only            Verify tools only; do not install Go.
  -h, --help              Show this help.

Required tools for the current High subset:
  go node python3

For Python tasks, the script also copies Python package overlays from each
selected task Docker image into deterministic sandbox paths:

  /sandbox/.deepswe-tools/python-runtime/<image-hash>/usr/local
  /sandbox/.deepswe-tools/python-site/<image-hash>/site-packages
  /sandbox/.deepswe-tools/python-bin/<image-hash>

The DeepSWE OpenClaw runner injects these paths only for the matching task.
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
sandbox="nejumi-taiwan"
nemoclaw_bin="nemoclaw"
tasks_root="$repo_root/external/deep-swe/tasks"
task_names_file="$repo_root/data/taiwan/deepswe/subsets/essential_anchored_high_8_cost_trimmed_cap_safe_lang_balanced_task_names.json"
go_image=""
pull_image=1
check_only=0
required_tools=(go node python3)

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
    --tasks-root)
      tasks_root="${2:?--tasks-root requires a value}"
      shift 2
      ;;
    --task-names-file)
      task_names_file="${2:?--task-names-file requires a value}"
      shift 2
      ;;
    --go-image)
      go_image="${2:?--go-image requires a value}"
      shift 2
      ;;
    --no-pull)
      pull_image=0
      shift
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
if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required to copy Go from a DeepSWE task image" >&2
  exit 127
fi

sandbox_exec() {
  "$nemoclaw_bin" sandbox exec "$sandbox" \
    --workdir /sandbox \
    --no-tty \
    --timeout "${1:?timeout required}" \
    -- "${@:2}"
}

check_tools() {
  sandbox_exec 120 sh -lc 'ok=0; for cmd in "$@"; do if command -v "$cmd" >/dev/null 2>&1; then printf "%s\tOK\t%s\n" "$cmd" "$(command -v "$cmd")"; else printf "%s\tMISSING\n" "$cmd"; ok=1; fi; done; exit "$ok"' _ "${required_tools[@]}"
}

check_go_module_cache() {
  if [[ "${#selected_go_images[@]}" -eq 0 ]]; then
    echo "go-mod-cache	SKIPPED	no-selected-go-tasks"
    return 0
  fi
  sandbox_exec 120 sh -lc 'if find /sandbox/go/pkg/mod -type d -name "*@v*" 2>/dev/null | grep -v "/cache/download/" | head -1 | grep -q .; then echo "go-mod-cache	OK	/sandbox/go/pkg/mod"; else echo "go-mod-cache	MISSING"; exit 1; fi'
}

sandbox_container_id() {
  docker ps --format '{{.ID}}\t{{.Names}}' \
    | awk -v prefix="openshell-${sandbox}-" '$2 ~ "^" prefix { print $1; exit }'
}

install_standard_path_wrappers() {
  local container_id
  container_id="$(sandbox_container_id)"
  if [[ -z "$container_id" ]]; then
    echo "Could not find OpenShell Docker container for sandbox: $sandbox" >&2
    return 1
  fi
  docker exec --user root "$container_id" sh -lc \
    'ln -sfn /sandbox/.deepswe-tools/go/bin/go /usr/local/bin/go && ln -sfn /sandbox/.deepswe-tools/go/bin/gofmt /usr/local/bin/gofmt'
}

select_images_by_language() {
  local target_language="${1:?language required}"
  python3 - "$tasks_root" "$task_names_file" "$target_language" <<'PY'
import json
import re
import sys
from pathlib import Path

tasks_root = Path(sys.argv[1])
task_names_file = Path(sys.argv[2])
target_language = sys.argv[3].strip().lower()
names = json.loads(task_names_file.read_text(encoding="utf-8"))
images = []
seen = set()
for name in names:
    task_toml = tasks_root / name / "task.toml"
    text = task_toml.read_text(encoding="utf-8")
    lang_match = re.search(r"^language\s*=\s*['\"]([^'\"]+)['\"]", text, re.M)
    image_match = re.search(r"^docker_image\s*=\s*['\"]([^'\"]+)['\"]", text, re.M)
    if not lang_match or not image_match:
        continue
    if lang_match.group(1).strip().lower() == target_language:
        image = image_match.group(1).strip()
        if image not in seen:
            seen.add(image)
            images.append(image)
for image in images:
    print(image)
PY
}

if [[ -n "$go_image" ]]; then
  selected_go_images=("$go_image")
else
  mapfile -t selected_go_images < <(select_images_by_language go)
fi
mapfile -t selected_python_images < <(select_images_by_language python)

image_hash() {
  printf '%s' "${1:?image required}" | sha256sum | awk '{print substr($1, 1, 12)}'
}

check_python_overlays() {
  if [[ "${#selected_python_images[@]}" -eq 0 ]]; then
    echo "python-overlays	SKIPPED	no-selected-python-tasks"
    return 0
  fi
  local ok=0
  for selected_python_image in "${selected_python_images[@]}"; do
    local hash
    hash="$(image_hash "$selected_python_image")"
    local site_path="/sandbox/.deepswe-tools/python-site/${hash}/site-packages"
    local bin_path="/sandbox/.deepswe-tools/python-bin/${hash}"
    if ! sandbox_exec 120 sh -lc \
      "PYTHONPATH='${site_path}' PATH='${bin_path}:/sandbox/.deepswe-tools/go/bin:/sandbox/.npm-global/bin':\$PATH python3 -c 'import pytest, setuptools; print(\"python-overlay\tOK\")'" >/dev/null; then
      echo "python-overlay	MISSING	${selected_python_image}	${site_path}" >&2
      ok=1
    else
      echo "python-overlay	OK	${selected_python_image}	${site_path}"
    fi
  done
  return "$ok"
}

write_python_runtime_wrapper() {
  local wrapper_path="${1:?wrapper path required}"
  local hash="${2:?hash required}"
  cat >"$wrapper_path" <<'SH'
#!/usr/bin/env sh
set -eu
tool_name="$(basename "$0")"
runtime_root="__RUNTIME_ROOT__"
wrapper_dir="__WRAPPER_DIR__"
export PYTHONHOME="$runtime_root"
export LD_LIBRARY_PATH="$runtime_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$wrapper_dir:$runtime_root/bin:$PATH"
case "$tool_name" in
  python|python3|python3.*)
    exec "$runtime_root/bin/python3" "$@"
    ;;
  pip|pip3|pip3.*)
    exec "$runtime_root/bin/python3" -m pip "$@"
    ;;
esac
target="$runtime_root/bin/$tool_name"
if [ -f "$target" ] && head -n 1 "$target" 2>/dev/null | grep -qi 'python'; then
  exec "$runtime_root/bin/python3" "$target" "$@"
fi
exec "$target" "$@"
SH
  sed -i \
    -e "s#__RUNTIME_ROOT__#/sandbox/.deepswe-tools/python-runtime/${hash}/usr/local#g" \
    -e "s#__WRAPPER_DIR__#/sandbox/.deepswe-tools/python-bin/${hash}#g" \
    "$wrapper_path"
  chmod +x "$wrapper_path"
}

if [[ "$check_only" -eq 1 ]]; then
  check_status=0
  check_tools || check_status=1
  check_go_module_cache || check_status=1
  check_python_overlays || check_status=1
  exit "$check_status"
fi

tools_ok=0
module_cache_ok=0
python_overlays_ok=0
if check_tools; then
  tools_ok=1
fi
if check_go_module_cache; then
  module_cache_ok=1
fi
if check_python_overlays; then
  python_overlays_ok=1
fi
if [[ "$tools_ok" -eq 1 && "$module_cache_ok" -eq 1 && "$python_overlays_ok" -eq 1 && -z "$go_image" ]]; then
  echo "DeepSWE sandbox toolchain, Go module cache, and Python overlays already available."
  exit 0
fi

for selected_image in "${selected_go_images[@]}" "${selected_python_images[@]}"; do
  [[ -n "$selected_image" ]] || continue
  if ! docker image inspect "$selected_image" >/dev/null 2>&1; then
    if [[ "$pull_image" -ne 1 ]]; then
      echo "Selected DeepSWE source image is not present locally: $selected_image" >&2
      exit 2
    fi
    echo "Pulling DeepSWE source image: $selected_image" >&2
    docker pull "$selected_image"
  fi
done

tmpdir="$(mktemp -d)"
cache_upload_root="/sandbox/.deepswe-tools/go-mod-caches-$(date +%Y%m%d%H%M%S)-$$"
container_id=""
cleanup() {
  if [[ -n "$container_id" ]]; then
    docker rm -f "$container_id" >/dev/null 2>&1 || true
  fi
  chmod -R u+w "$tmpdir" >/dev/null 2>&1 || true
  rm -rf "$tmpdir"
}
trap cleanup EXIT

if [[ "${#selected_go_images[@]}" -gt 0 && ( "$tools_ok" -ne 1 || "$module_cache_ok" -ne 1 ) ]]; then
  container_id="$(docker create "${selected_go_images[0]}" /bin/sh -c true)"
  docker cp "$container_id:/usr/local/go" "$tmpdir/go"

  sandbox_exec 120 sh -lc "mkdir -p /sandbox/.deepswe-tools /sandbox/.npm-global/bin /sandbox/go/pkg/mod '$cache_upload_root' && rm -rf /sandbox/.deepswe-tools/go"
  "$nemoclaw_bin" sandbox upload "$sandbox" "$tmpdir/go" /sandbox/.deepswe-tools/
  sandbox_exec 120 sh -lc 'mkdir -p /sandbox/.npm-global/bin; ln -sfn /sandbox/.deepswe-tools/go/bin/go /sandbox/.npm-global/bin/go; ln -sfn /sandbox/.deepswe-tools/go/bin/gofmt /sandbox/.npm-global/bin/gofmt'
  install_standard_path_wrappers
fi

cache_index=0
if [[ "$module_cache_ok" -ne 1 ]]; then
  for selected_go_image in "${selected_go_images[@]}"; do
    if [[ -n "$container_id" ]]; then
      docker rm -f "$container_id" >/dev/null 2>&1 || true
      container_id=""
    fi
    cache_dir="$tmpdir/mod-${cache_index}"
    mkdir -p "$cache_dir"
    if docker run --rm "$selected_go_image" sh -lc 'cache="$(go env GOMODCACHE 2>/dev/null || true)"; cache="${cache:-/root/go/pkg/mod}"; test -d "$cache"; cd "$cache"; tar cf - .' | tar -xf - -C "$cache_dir"; then
      "$nemoclaw_bin" sandbox upload "$sandbox" "$cache_dir" "$cache_upload_root/"
      sandbox_exec 300 sh -lc "mkdir -p /sandbox/go/pkg/mod && chmod -R u+w /sandbox/go/pkg/mod >/dev/null 2>&1 || true; cp -a '$cache_upload_root/mod-${cache_index}/.' /sandbox/go/pkg/mod/"
      echo "Merged Go module cache from: $selected_go_image"
    else
      echo "No /root/go/pkg/mod cache found in image: $selected_go_image" >&2
    fi
    cache_index=$((cache_index + 1))
  done
fi

if [[ "${#selected_python_images[@]}" -gt 0 && "$python_overlays_ok" -ne 1 ]]; then
  sandbox_exec 120 sh -lc 'mkdir -p /sandbox/.deepswe-tools/python-runtime /sandbox/.deepswe-tools/python-site /sandbox/.deepswe-tools/python-bin'
fi

for selected_python_image in "${selected_python_images[@]}"; do
  hash="$(image_hash "$selected_python_image")"
  runtime_dir="$tmpdir/python-runtime/${hash}/usr/local"
  bin_dir="$tmpdir/python-bin/${hash}"
  mkdir -p "$runtime_dir/bin" "$runtime_dir/lib" "$bin_dir"

  mapfile -t python_site_paths < <(
    docker run --rm -i --entrypoint python3 "$selected_python_image" - <<'PY'
import os
import site
import sys

seen = set()
for path in list(site.getsitepackages()) + list(sys.path):
    if not isinstance(path, str) or not path:
        continue
    if path in seen:
        continue
    seen.add(path)
    if ("site-packages" in path or "dist-packages" in path) and os.path.isdir(path):
        print(path)
PY
  )
  if [[ "${#python_site_paths[@]}" -eq 0 ]]; then
    echo "No Python site-packages path found in image: $selected_python_image" >&2
    exit 3
  fi
  mapfile -t python_runtime_paths < <(
    docker run --rm -i --entrypoint python3 "$selected_python_image" - <<'PY'
import os
import sysconfig

for key in ("BINDIR", "LIBDIR"):
    value = sysconfig.get_config_var(key)
    if value and os.path.isdir(value):
        print(f"{key}\t{value}")
stdlib = sysconfig.get_path("stdlib")
if stdlib and os.path.isdir(stdlib):
    print(f"STDLIB\t{stdlib}")
PY
  )

  if [[ -n "$container_id" ]]; then
    docker rm -f "$container_id" >/dev/null 2>&1 || true
    container_id=""
  fi
  container_id="$(docker create "$selected_python_image" /bin/sh -c true)"
  for runtime_path_row in "${python_runtime_paths[@]}"; do
    runtime_kind="${runtime_path_row%%$'\t'*}"
    runtime_path="${runtime_path_row#*$'\t'}"
    case "$runtime_kind" in
      BINDIR)
        docker cp "$container_id:${runtime_path}/." "$runtime_dir/bin/" >/dev/null 2>&1 || true
        ;;
      STDLIB)
        docker cp "$container_id:${runtime_path}" "$runtime_dir/lib/" >/dev/null 2>&1 || true
        ;;
    esac
  done
  docker cp "$container_id:/usr/local/lib/." "$runtime_dir/lib-root" >/dev/null 2>&1 || true
  find "$runtime_dir/lib-root" -maxdepth 1 -type f -name 'libpython*' -exec cp -a {} "$runtime_dir/lib/" \; 2>/dev/null || true
  rm -rf "$runtime_dir/lib-root"
  if [[ ! -x "$runtime_dir/bin/python3" ]]; then
    echo "No runnable python3 copied from image: $selected_python_image" >&2
    exit 3
  fi
  find "$runtime_dir/bin" -maxdepth 1 -type f -perm -111 -printf '%f\n' | while IFS= read -r tool_name; do
    write_python_runtime_wrapper "$bin_dir/$tool_name" "$hash"
  done
  for standard_tool in python python3 pip pip3 pytest; do
    if [[ ! -e "$bin_dir/$standard_tool" ]]; then
      write_python_runtime_wrapper "$bin_dir/$standard_tool" "$hash"
    fi
  done
  chmod -R u+rwX "$runtime_dir" "$bin_dir" >/dev/null 2>&1 || true

  sandbox_exec 120 sh -lc "rm -rf '/sandbox/.deepswe-tools/python-runtime/${hash}' '/sandbox/.deepswe-tools/python-site/${hash}' '/sandbox/.deepswe-tools/python-bin/${hash}'"
  "$nemoclaw_bin" sandbox upload "$sandbox" "$tmpdir/python-runtime/${hash}" /sandbox/.deepswe-tools/python-runtime/
  "$nemoclaw_bin" sandbox upload "$sandbox" "$tmpdir/python-bin/${hash}" /sandbox/.deepswe-tools/python-bin/
  site_target=""
  for python_site_path in "${python_site_paths[@]}"; do
    if [[ "$python_site_path" == /usr/local/* ]]; then
      site_target="/sandbox/.deepswe-tools/python-runtime/${hash}/usr/local/${python_site_path#/usr/local/}"
      break
    fi
  done
  if [[ -z "$site_target" ]]; then
    echo "No /usr/local Python site-packages path found in image: $selected_python_image" >&2
    exit 3
  fi
  sandbox_exec 120 sh -lc "mkdir -p '/sandbox/.deepswe-tools/python-site/${hash}' && ln -sfn '${site_target}' '/sandbox/.deepswe-tools/python-site/${hash}/site-packages'"
  echo "Installed Python runtime overlay from: $selected_python_image -> hash=$hash"
done

check_tools
check_go_module_cache
check_python_overlays
check_go_module_cache
sandbox_exec 120 sh -lc 'export PATH=/sandbox/.deepswe-tools/go/bin:/sandbox/.npm-global/bin:$PATH; go version; node --version; python3 --version'
