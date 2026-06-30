#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OFFICIAL_REPO="${OFFICIAL_REPO:-external/SWE-bench_Pro-os}"
OFFICIAL_URL="${OFFICIAL_URL:-https://github.com/scaleapi/SWE-bench_Pro-os.git}"
MIN_DOCKER_SDK_VERSION="${MIN_DOCKER_SDK_VERSION:-7.1.0}"

INSTALL_PYTHON_DEPS=0
CLONE_OFFICIAL=0
CHECK_ONLY=0

usage() {
  cat <<'USAGE'
Usage:
  scripts/setup/install_swebench_pro_eval.sh --all
  scripts/setup/install_swebench_pro_eval.sh --check-only

Options:
  --all                   Install Python deps and clone the official evaluator if missing.
  --check-only            Validate local Docker, Python deps, and official evaluator files.
  --install-python-deps   Install local Docker evaluator Python dependencies.
  --clone-official        Clone Scale's SWE-bench Pro evaluator into external/SWE-bench_Pro-os.
  --official-repo PATH    Official evaluator checkout path. Default: external/SWE-bench_Pro-os.
  --python PATH           Python executable. Default: python3.
  -h, --help              Show this help.

Environment:
  PYTHON_BIN              Same as --python.
  OFFICIAL_REPO           Same as --official-repo.
  OFFICIAL_URL            Official evaluator git URL.
USAGE
}

log() {
  printf '[swebench-pro-setup] %s\n' "$*"
}

install_python_deps() {
  if [ "$INSTALL_PYTHON_DEPS" -eq 0 ]; then
    return 0
  fi
  log "Installing local Docker evaluator Python dependencies"
  "$PYTHON_BIN" -m pip install --user -U "docker>=$MIN_DOCKER_SDK_VERSION" pandas tqdm
}

clone_official_repo() {
  if [ "$CLONE_OFFICIAL" -eq 0 ]; then
    return 0
  fi
  if [ -d "$OFFICIAL_REPO/.git" ]; then
    log "Official evaluator already exists: $OFFICIAL_REPO"
    return 0
  fi
  mkdir -p "$(dirname "$OFFICIAL_REPO")"
  log "Cloning $OFFICIAL_URL into $OFFICIAL_REPO"
  git clone "$OFFICIAL_URL" "$OFFICIAL_REPO"
}

check_official_repo() {
  local missing=0
  for path in \
    "$OFFICIAL_REPO/swe_bench_pro_eval.py" \
    "$OFFICIAL_REPO/helper_code/image_uri.py" \
    "$OFFICIAL_REPO/run_scripts"
  do
    if [ ! -e "$path" ]; then
      log "Missing: $path"
      missing=1
    fi
  done
  return "$missing"
}

check_python_deps_and_docker() {
  "$PYTHON_BIN" - "$MIN_DOCKER_SDK_VERSION" <<'PY'
import sys

minimum = tuple(int(part) for part in sys.argv[1].split("."))

def parse(value):
    parts = []
    for part in value.split("."):
        digits = ""
        for char in part:
            if char.isdigit():
                digits += char
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)

try:
    import docker
except Exception as exc:
    print(f"docker import failed: {exc}", file=sys.stderr)
    sys.exit(1)

raw_version = getattr(docker, "__version__", "0")
if parse(raw_version) < minimum:
    print(f"docker SDK {raw_version} is older than {sys.argv[1]}", file=sys.stderr)
    sys.exit(1)

for package in ("pandas", "tqdm"):
    try:
        __import__(package)
    except Exception as exc:
        print(f"{package} import failed: {exc}", file=sys.stderr)
        sys.exit(1)

try:
    client = docker.from_env()
    client.ping()
finally:
    close = getattr(locals().get("client", None), "close", None)
    if callable(close):
        close()

print(f"Python deps OK; docker SDK {raw_version}; Docker ping OK")
PY
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --all)
      INSTALL_PYTHON_DEPS=1
      CLONE_OFFICIAL=1
      ;;
    --check-only)
      CHECK_ONLY=1
      ;;
    --install-python-deps)
      INSTALL_PYTHON_DEPS=1
      ;;
    --clone-official)
      CLONE_OFFICIAL=1
      ;;
    --official-repo)
      OFFICIAL_REPO="$2"
      shift
      ;;
    --python)
      PYTHON_BIN="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
  shift
done

if [ "$CHECK_ONLY" -eq 0 ]; then
  install_python_deps
  clone_official_repo
fi

check_official_repo
check_python_deps_and_docker
log "Done."
