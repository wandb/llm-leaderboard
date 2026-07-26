#!/usr/bin/env bash
#SBATCH --job-name=taiwan-eval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=36:00:00

set -euo pipefail

: "${TAIWAN_EVAL_JOB_MANIFEST:?TAIWAN_EVAL_JOB_MANIFEST is required}"
: "${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm job array}"

repo_root="${TAIWAN_LB_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
launcher_python="${TAIWAN_EVAL_LAUNCHER_PYTHON:-$repo_root/.venv/bin/python}"
cd "$repo_root"

exec "$launcher_python" \
  "$repo_root/scripts/tools/submit_taiwan_full_eval_array.py" \
  run-task \
  --job-manifest "$TAIWAN_EVAL_JOB_MANIFEST" \
  --index "$SLURM_ARRAY_TASK_ID"
