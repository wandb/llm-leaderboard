# Agentic SWE sandbox baseline leak (2026-07-24)

## Status

Fixed in the working tree. Legacy Low/Middle results without isolated baseline
evidence must not be used for model scoring or comparison.

This was runtime harness contamination, not evidence of model-training data
contamination.

## What happened

The generic SWE runner copied a clean host checkout to a stable sandbox path:

`/sandbox/checkouts/<task-id>`

On a later run, the runner treated an existing Git directory at that path as a
valid checkout and skipped synchronization. A patch left by an earlier model or
reference run could therefore become the next model's starting state.

## Confirmed evidence

The Sonnet Low/Middle run
`model_ranking_20260723_sonnet46_direct_high_current80` contains tasks where the
model reported that the requested fix was already present, made no repository
edit, and submitted a patch matching the gold patch. Confirmed examples include:

- `pytest-dev__pytest-5227`
- `pydata__xarray-5131`
- `sphinx-doc__sphinx-8595`
- `pytest-dev__pytest-11148`
- `pytest-dev__pytest-7168`

The run's reported 61/72 Low/Middle resolved result is not valid performance
evidence.

An artifact scan found 28 historical Agentic SWE run directories containing 785
legacy unscoped checkout records; 630 have billable usage recorded. This does
not prove every record was contaminated, but none has sufficient baseline
evidence to certify isolation.

The current model-ranking Low/Middle runs for GPT-4.1-mini, GPT-5.6 Luna,
Claude Sonnet 4.6, and W&B GLM-5.2 are therefore quarantined. DeepSWE High runs
already used task/run-hashed checkout roots and are assessed separately.

## Remediation

- Sandbox copy paths now include a deterministic hash of the output run.
- Copy mode always deletes and reconstructs the sandbox checkout; existence is
  never accepted as evidence of readiness.
- A dirty host checkout is rejected before archive creation.
- Host `HEAD` and tree SHA are recorded.
- The reconstructed sandbox baseline must be clean and have the same tree SHA.
- The runner version was bumped, invalidating legacy resume cache entries.
- Explicit Assorted-result reuse rejects missing, visible, dirty, or mismatched
  baseline evidence.

## Verification

- `tests/test_swebench_pro.py`: 94 passed
- `tests/test_deepswe_openclaw.py`: 68 passed
- `tests/test_agentic_swe_assorted.py`: 38 passed
- The actual legacy Sonnet Low/Middle artifact is rejected by the new reuse
  validation.

No paid rerun was started as part of this repair.
