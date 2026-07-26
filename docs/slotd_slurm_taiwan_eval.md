# Taiwan evaluation arrays with slotd or Slurm

The Taiwan evaluation harness can submit one model per job-array task through
either local `slotd` or a regular Slurm cluster. The job script uses standard
`#SBATCH` directives and `SLURM_ARRAY_TASK_ID`; it has no slotd-only runtime
dependency.

Files:

- `scripts/slurm/taiwan_full_eval_array.sbatch`: portable array task
- `scripts/tools/submit_taiwan_full_eval_array.py`: bundle and submission tool

## Safety defaults

- Submission is a dry run unless `--submit` is present.
- Model API execution is disabled unless `--execute` is present.
- Without `--execute`, every array task invokes the existing batch runner with
  `--prepare-only`.
- External execution still requires the existing budget estimate and external
  action approval evidence.
- A config explicitly marked cash-cost-exempt can instead use
  `--allow-cash-cost-exempt-execution`; the batch runner verifies every selected
  config before allowing that path.
- Every model receives separate runner output and generated-config directories.
- The default array concurrency is 2. Each task reserves 16 CPUs and 64 GB RAM.
  This fits the local 40-CPU, 257-GB host while limiting provider bursts.
- The default wall limit is 36 hours. Override it to match the cluster
  partition and the selected benchmark phase.
- Each task writes a terminal `slurm_task.json` containing `completed` or
  `failed`, its return code, and end time.

The array layer is separate from benchmark-internal worker counts. Raising both
at once can multiply provider traffic, Docker activity, and checkout storage.

Agentic runners that name the same mutable NeMoClaw sandbox acquire an exclusive
lease. Non-Agentic phases can still overlap, while the Agentic phase waits
instead of corrupting another run's config. With the default shared
`nejumi-taiwan` sandbox, keep array concurrency at 2. For higher true Agentic
parallelism, provision distinct sandboxes and assign them per model in the
model manifest.

## Local slotd prepare-only check

```bash
.venv/bin/python scripts/tools/submit_taiwan_full_eval_array.py submit \
  --batch-id slotd-prepare-check \
  --model gpt-4_1-mini-openai-direct \
  --model glm-5_2-wandb-inference \
  --phase full \
  --max-parallel 2 \
  --submit

squeue
sacct
```

Logs are written under:

```text
outputs/slurm/taiwan_eval_jobs/slotd-prepare-check/logs/
```

Cancel the whole array with:

```bash
scancel <array-job-id>
```

## Paid or externally visible execution

Paid execution is deliberately verbose. Supply a fresh batch ID, model list,
budget estimate, approval report bound to its source packet, run purpose, cost
band, and W&B prefix:

```bash
.venv/bin/python scripts/tools/submit_taiwan_full_eval_array.py submit \
  --batch-id taiwan-release-wave-01 \
  --model gpt-4_1-mini-openai-direct \
  --model gpt-5_6-luna-openai-direct-high \
  --phase full \
  --execute \
  --run-purpose "Taiwan leaderboard release wave 1" \
  --expected-cost-band '$50-$100' \
  --pre-run-budget-estimate-json outputs/release/budget.json \
  --external-action-approval-report-json outputs/release/approval.report.json \
  --external-action-approval-source-packet-json outputs/release/approval.packet.json \
  --wandb-run-id-prefix tw-release-wave-01 \
  --runner-arg=--verify-wandb-completion \
  --runner-arg=--verify-weave-agents \
  --runner-arg=--require-nemoclaw-agentic-config \
  --runner-arg=--require-weave-content-canary \
  --runner-arg=--weave-content-canary-gate \
  --runner-arg=outputs/release/weave_content_canary_gate.json \
  --yes \
  --submit
```

Run the command once without `--submit` to inspect the generated manifest and
exact `sbatch` command.

## SUNk or another Slurm cluster

The same submission command works when `sbatch` is provided by Slurm. Override
resources for the target partition:

```bash
... submit \
  --partition cpu \
  --cpus-per-task 16 \
  --mem 64G \
  --time 36:00:00 \
  --max-parallel 2
```

`--python` selects both the array launcher interpreter and the interpreter used
by the existing batch runner. The submitted job receives it through Slurm's
standard `--export` mechanism, so no slotd-specific path lookup is required.
The job also prepends `$HOME/.local/bin` to `PATH`, which covers the default
NeMoClaw installation. Set `TAIWAN_EVAL_USER_BIN` through the scheduler export
when command-line tools are installed elsewhere.

The repository, `.venv`, evidence files, `.env`, and output root must be visible
on the execution node at the same absolute paths. If the cluster uses a shared
filesystem, place the repository and bundle root there before submission.

References:

- <https://zenn.dev/turing_motors/articles/0d528e31b9d8d7>
- <https://github.com/ymgaq/slotd>
