# Taiwan Evaluation Harness Reliability Review

Date: 2026-07-25

## Executive summary

The repeated stops were not one provider outage or one benchmark bug. The
harness had several independent recovery and validation gaps:

1. Long non-agentic batches kept results only in memory until the whole batch
   finished.
2. A local benchmark completion marker could be reused without proving that
   the matching W&B run had completed the same benchmark.
3. A benchmark process could hang indefinitely outside the model-level turn
   and token limits.
4. Aggregation could continue with missing required benchmark rows and make an
   incomplete run look usable.
5. HLE/JTruthfulQA/ARC-AGI contained paths that could turn dataset, scorer, or
   provider failures into apparent completion or model wrong answers.
6. Taiwan aggregation parsed table values before applying taxonomy-aware score
   normalization, so a valid percentage literal such as `38.72%` crashed the
   final benchmark.
7. Agentic SWE preserved complete NeMoClaw session audits inside a nested
   object but did not backfill the corresponding flat W&B columns for older
   Low/Middle rows.
8. Every benchmark fingerprint included unrelated shared files. A maintenance
   change to one evaluator could therefore invalidate already-completed,
   unrelated paid benchmarks.
9. Host process-group cleanup could not prove that a detached test worker
   inside the NeMoClaw sandbox had exited.

These paths are now addressed. The harness saves model and judge responses per
item, fingerprints checkpoints against configuration and code, requires both
local and W&B completion evidence before skipping work, enforces harness-level
wall-clock safety limits, terminates descendant processes on interruption, and
rejects incomplete Taiwan aggregation.

## Recovery contract

### Item-level recovery

Long API benchmarks write each completed response atomically. A checkpoint is
reused only when its request fingerprint matches the current request. Corrupt,
partial, or stale records are ignored and regenerated.

Covered paths include TCEval-v2, JASTER, IFEval zh-TW, TS-Bench, ARC-AGI,
M-IFEval, JTruthfulQA, JBBQ, toxicity, HLE, HalluLens, MT-Bench, and legacy
SWE-bench generation.

Configured dataset task lists are also treated as contracts. TCEval-v2 and
JASTER fail if a configured task/subset file is missing instead of silently
scoring a smaller dataset. Taiwan JASTER explicitly declares `tmmluplus`; the
legacy optional `jhumaneval` entry is not part of its W&B artifact.

### Benchmark-level recovery

Every benchmark writes `started`, `failed`, or `completed` state beneath the
run output root. A completed benchmark is skipped only on an explicitly
authorized W&B resume and only when:

- the local marker belongs to the same W&B run ID;
- its configuration and code fingerprint still match; and
- W&B summary contains the corresponding completion evidence.

Changing the model, benchmark configuration, common inference settings, or
relevant evaluator code invalidates the old marker.

Legacy markers without fingerprints are rejected by default. The one recovery
configuration for the in-progress July 25 run may migrate them only when the
same run ID and matching W&B benchmark completion evidence are both present.
That migration is controlled by the explicit
`output.trust_completed_checkpoint_fingerprint_mismatch` allowlist; it is not a
global fallback.

Code fingerprints are benchmark-scoped. Agentic SWE changes no longer
invalidate BFCL, Agentic Math, or direct-inference benchmarks. Shared inference
code is included only for benchmarks that actually call it, and BFCL includes
its own package tree.

### Failure propagation

Timeouts, infrastructure errors, missing grading phases, and incomplete
required output are failures. They are not silently converted into zero scores
or successful aggregate completion. Agentic task time limits remain benchmark
rules; the larger benchmark subprocess limits exist only to stop a stuck
harness.

Model responses that complete normally but violate an answer format remain
model errors. Provider failures, missing datasets, and scorer crashes remain
harness failures and are resumed from item checkpoints.

### Process cleanup

Agentic benchmark commands run in their own process groups. Parent interruption
or a global harness timeout sends `SIGTERM` to the whole group and escalates to
`SIGKILL` if required. This prevents abandoned OpenClaw, NeMoClaw, Docker, or
grader processes from colliding with a resume.

DeepSWE cleanup also reads the run's task-agent metadata, obtains only that
run's sandbox workspace roots, terminates processes whose command line belongs
to those roots, and verifies that none remain. It does not scan or terminate
unrelated project workspaces.

## Data normalization and relogging

Taiwan aggregation accepts native numeric cells, numeric strings, and `%`
literals. Values are normalized according to the taxonomy unit's declared
scale before averaging, so `38.72%` and native `38.72` on a 0-100 unit are
equivalent.

Agentic SWE result loading backfills flat observability fields from the nested
`nemoclaw_session_audit` object. This supports historical result artifacts
without weakening the strict W&B row verifier.

Maintenance relogging uses a dedicated configuration that enables exactly one
benchmark and sets `run_openclaw: false`. It can repair W&B table shape from
completed local results without launching model inference or unrelated
benchmarks.

## Safety limits

These are harness-stall ceilings, not per-task model budgets:

| Path | Ceiling |
| --- | ---: |
| Agentic Math benchmark | 6 hours |
| Agentic SWE-Assorted benchmark | 14 hours |
| SWE-bench Pro generation | 8 hours |
| SWE-bench Pro grading | 4 hours |
| DeepSWE benchmark | 36 hours |
| Agentic static preflight | 180 seconds |

The task-level turn, tool, token, idle, and wall-time limits remain separately
configured by each benchmark.

## Completion gate

A full run is release-usable only after all of the following pass:

1. The evaluation process exits with code 0.
2. Every enabled benchmark has matching local and W&B completion evidence.
3. Expected row counts and required W&B tables are present.
4. BFCL timeout count is present as a non-negative scored-outcome metric, and
   the infrastructure inference-error count is zero.
5. Agentic W&B/Weave evidence passes the configured trace checks.
6. Taiwan aggregation finds every required benchmark score.
7. Required Agentic session audit fields pass row-by-row; summary counts alone
   are insufficient.

W&B's top-level state may temporarily retain `crashed` while an existing run ID
is being resumed. It is not used alone as completion evidence.

## Operator procedure

1. Run the merged-config preflight before paid execution.
2. Use a fresh W&B run ID by default.
3. Resume only after explicit user authorization, with the same run ID and
   output root.
4. Do not delete per-item checkpoints or benchmark markers before a resume.
5. Do not launch a second process for the same run ID.
6. After process exit, run the W&B completion verifier before starting the next
   model.
7. Preserve batch logs. Each retry or resume is appended as a separate,
   timestamped attempt.

## July 25 recovery result

Run `tw-first-wave-20260725-gpt-4_1-mini-openai-direct` completed all 13 enabled
benchmark phases and is `finished` in W&B. Read-only completion verification
passes for:

- BFCL v4: 496 logical cases, 607 runtime rows, zero timeout rows, zero
  inference-error rows.
- Agentic Math: 50/50 result rows and required session evidence.
- Agentic SWE-Assorted: 50/50 required session audits passed, with tier counts
  20 Low / 20 Middle / 10 High.
- Taiwan full aggregation: every non-pending taxonomy unit and every aggregate
  table is present; missing required count is zero.

The Agentic SWE observability repair was a 2.1-second relog of existing results.
It did not rerun OpenClaw or incur additional model inference cost. The final
Taiwan overall score for this run is `47.53814970061933`.

Post-recovery validation:

- All four live W&B completion verifiers passed.
- A sandbox-local cleanup probe terminated a detached process under its exact
  run workspace and reported zero remaining processes.
- The repository test suite passed: `1657 passed` (one third-party Weave
  deprecation warning).
- `git diff --check` and Python compilation of the changed runtime paths
  passed.

## July 26 BFCL and process-level hardening

The next GLM-5.2 run exposed a BFCL v4 recovery defect after 496 logical cases
had been generated. One failed memory prerequisite was removed from the retry
set because its dependent main cases already existed. This made the recovery
attempt a no-op and left later customer-memory rows based on contaminated
state.

The corrected BFCL recovery contract is:

- a failed memory prerequisite invalidates its later prerequisites and all
  dependent main cases;
- memory state is restored from the last healthy prerequisite checkpoint;
- stale descendant snapshots are removed before regeneration;
- infrastructure failures are retried inline before dependencies are
  released;
- result writes are synchronous and atomic, and already-running cases are
  drained and persisted before failed rows are handed to the bounded outer
  recovery loop;
- a recovery round that generates zero rows while failures remain is rejected
  immediately.

The affected production-shaped result copy selected exactly 10 repair rows:
the failed prerequisite, three later prerequisites, and six dependent customer
cases. It restored the customer snapshot from prerequisite 5.

### Authorized process-level recovery

The batch runner still performs no automatic resume by default. A bounded
process-level resume is available only when all of these are true:

1. `--allow-wandb-resume` is supplied;
2. `--wandb-run-id-prefix` is non-empty;
3. the signed/reviewed resume JSON matches the run prefix and phase;
4. `--infrastructure-resume-attempts N` is between 1 and 3 and the resume JSON
   contains the same `infrastructure_recovery_attempts: N`;
5. the process wrote a new failed benchmark checkpoint during that attempt;
6. the checkpoint classifies the failure as known provider or gateway
   infrastructure.

Example resume authorization payload:

```json
{
  "allow_wandb_resume": true,
  "wandb_run_id_prefix": "reviewed-run-prefix",
  "phase": "full",
  "explicit_user_instruction": true,
  "purpose": "resume only reviewed transient infrastructure failures",
  "infrastructure_recovery_attempts": 2
}
```

Model/task timeouts, turn/tool/token budget exhaustion, operator interruption,
authentication, permission, billing, insufficient quota, deterministic code
errors, and unknown failures are never automatically resumed. Each process
attempt and its decision are written to the batch manifest.

### Persistence and resource bounds

- BFCL result JSONL, benchmark checkpoints, direct-LLM item checkpoints,
  Agentic Math partial results, and Agentic SWE result files use atomic replace;
  critical checkpoints are flushed with `fsync`.
- Long child-process output is streamed to the terminal while only a bounded
  tail is retained in parent memory.
- BFCL provider request timeout is 120 seconds; its case timeout remains an
  independent 600 seconds.

Post-change validation on July 26:

- production-shaped BFCL repair selection and snapshot restoration passed;
- focused BFCL, checkpoint, batch, Agentic Math, and Agentic SWE suites passed;
- the complete repository suite passed: `1732 passed` with seven third-party
  deprecation warnings;
- no paid model API was called during this hardening work.

## July 26 cross-boundary robustness audit

A second audit covered failure classification, checkpoint restoration,
parallel sandbox ownership, post-run verification, scheduler records, and
cash-cost-exempt W&B execution.

Corrections:

- A failed rerun can no longer restore a local `last_completed` snapshot unless
  the same W&B run still has matching remote completion evidence.
- Agentic inner preflight JSON is deleted before every preflight, so a stale
  passing report cannot authorize a failed current check.
- Processes sharing a mutable NeMoClaw sandbox wait on an exclusive,
  owner-recorded lease instead of racing config writes or failing immediately.
  Nested commands inherit the lease and do not deadlock themselves.
- BFCL malformed model output and case timeout remain scored model outcomes.
  Only provider/web-search infrastructure errors enter bounded recovery.
- BFCL now persists an exhausted infrastructure row, withholds its dependent
  cases, and returns control to the failed-ID recovery loop. The earlier
  implementation raised before that outer loop could run.
- Batch reviews, checkpoint files, result manifests, and scheduler task records
  use atomic replacement with flushed temporary files.
- W&B and Weave post-run verifiers have an independent 600-second timeout and
  write explicit timeout evidence instead of hanging after inference finishes.
- Slotd/Slurm tasks retain a terminal status, return code, and end time instead
  of remaining indistinguishable from an active task after process exit.
- Cash-cost-exempt execution is available only with an explicit launcher flag
  and only when every selected generated config carries both the exemption and
  its reason. Normal paid providers retain all budget and approval gates.

The focused recovery suites passed before the complete `1732`-test run.
