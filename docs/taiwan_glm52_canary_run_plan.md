# GLM5.2 Taiwan Full Canary Run Plan

Status: prepared, metadata-verified, not executed.

## Purpose

Run one complete Taiwan leaderboard canary with `z-ai/glm-5.2` before expanding
to multiple models. The review should decide whether benchmark balance, W&B
logging, Weave/OpenClaw traces, and cost behavior are acceptable for broader
leaderboard execution.

## Scope

```text
Model: z-ai/glm-5.2
OpenClaw model: openrouter-direct/z-ai/glm-5.2
Generated config: configs/taiwan_full/generated_canary/config-taiwan-full-glm-5_2-openrouter-reasoning.yaml
Phase: recommended phased execution: nonagentic -> agentic -> agentic_aggregate
Benchmarks: all enabled Taiwan benchmarks in base_config_taiwan.yaml + generated full override
Expansion gate: no multi-model run until this canary is reviewed
Claude Opus: excluded from canary and default generation; final-only
```

## Budget Estimate

Local historical OpenClaw usage was converted to GLM5.2 pricing. This is a
planning estimate only; provider billing dashboards are authoritative.

```text
Estimate file: outputs/taiwan_full_eval/glm52_canary_budget_estimate.json
Agentic Math estimate: $19.54 based on completed DeepSeek 100-task baseline
SWE-Bench Pro estimate: $26.03 low / $49.05 mid / $184.97 high
Manual non-agentic + judge buffer: $75.00
Total planning band: about $120.57 low / $143.58 mid / $279.51 high
Recommended expected-cost-band for execution: $150-$300
```

## Prepared Commands

Generate or refresh the canary config without running paid APIs:

```bash
uv run python scripts/tools/run_taiwan_full_eval_batch.py \
  --canary \
  --prepare-only \
  --phase full \
  --generated-config-dir configs/taiwan_full/generated_canary \
  --output-root outputs/taiwan_full_eval \
  --run-purpose 'GLM5.2 one-model full canary for Taiwan leaderboard review' \
  --expected-cost-band 'TBD before execution'
```

Verify local config, OpenClaw registration, credentials, and W&B artifact aliases
without downloading benchmark files or running model inference:

```bash
uv run python scripts/tools/check_taiwan_canary_readiness.py \
  --verify-wandb-artifacts \
  --json outputs/taiwan_full_eval/glm52_canary_readiness.json
```

After NeMoClaw is installed and onboarded, enforce the sandbox gate before
agentic execution:

```bash
uv run python scripts/tools/check_taiwan_canary_readiness.py \
  --verify-wandb-artifacts \
  --require-nemoclaw \
  --nemoclaw-sandbox nejumi-taiwan \
  --json outputs/taiwan_full_eval/glm52_canary_readiness_nemoclaw.json
```

Before Phase 2 or any single-command `full` execution, require a successful
fresh Weave Agents content canary gate:

```bash
WEAVE_CONTENT_CANARY_GATE=outputs/weave_agents_content_canary/plans/weave_agents_content_canary_<CANARY_ID>.gate.json

uv run python scripts/tools/check_taiwan_canary_readiness.py \
  --weave-content-canary-gate "$WEAVE_CONTENT_CANARY_GATE" \
  --require-weave-content-canary \
  --json outputs/taiwan_full_eval/glm52_canary_readiness_weave_content.json
```

Recommended phased execution after review. Use the same W&B run id prefix across
all phases so outputs land on the same run.

Phase 1: non-agentic benchmarks.

```bash
uv run python scripts/tools/run_taiwan_full_eval_batch.py \
  --canary \
  --phase nonagentic \
  --generated-config-dir configs/taiwan_full/generated_canary_nonagentic \
  --output-root outputs/taiwan_full_eval \
  --wandb-run-id-prefix twcanary-glm52-20260627 \
  --yes \
  --run-purpose 'GLM5.2 one-model nonagentic phase before agentic canary expansion' \
  --expected-cost-band 'within $150-$300 full-canary planning band'
```

Phase 2: Agentic Math and SWE-Bench Pro generation.

```bash
uv run python scripts/tools/run_taiwan_full_eval_batch.py \
  --canary \
  --phase agentic \
  --generated-config-dir configs/taiwan_full/generated_canary_agentic \
  --output-root outputs/taiwan_full_eval \
  --wandb-run-id-prefix twcanary-glm52-20260627 \
  --verify-wandb-completion \
  --verify-weave-agents \
  --weave-agents-require-tool-span \
  --weave-agents-require-tool-content \
  --weave-content-canary-gate "$WEAVE_CONTENT_CANARY_GATE" \
  --require-weave-content-canary \
  --yes \
  --run-purpose 'GLM5.2 one-model agentic phase after nonagentic canary review' \
  --expected-cost-band 'within $150-$300 full-canary planning band; agentic estimate low/mid/high $45.57/$68.58/$204.51 before nonagentic buffer'
```

Phase 3: aggregate agentic outputs and Total Score.

```bash
uv run python scripts/tools/run_taiwan_full_eval_batch.py \
  --canary \
  --phase agentic_aggregate \
  --generated-config-dir configs/taiwan_full/generated_canary_agentic_aggregate \
  --output-root outputs/taiwan_full_eval \
  --wandb-run-id-prefix twcanary-glm52-20260627 \
  --verify-wandb-completion \
  --yes \
  --run-purpose 'GLM5.2 one-model aggregate phase after agentic outputs complete' \
  --expected-cost-band 'no new OpenClaw generation expected; W&B logging/aggregation only'
```

Single-command full execution is available but is not the recommended path for
cost visibility:

```bash
uv run python scripts/tools/run_taiwan_full_eval_batch.py \
  --canary \
  --phase full \
  --generated-config-dir configs/taiwan_full/generated_canary \
  --output-root outputs/taiwan_full_eval \
  --wandb-run-id-prefix twcanary-glm52-20260627 \
  --verify-wandb-completion \
  --verify-weave-agents \
  --weave-content-canary-gate "$WEAVE_CONTENT_CANARY_GATE" \
  --require-weave-content-canary \
  --yes \
  --run-purpose 'GLM5.2 one-model full canary for Taiwan leaderboard review' \
  --expected-cost-band '$150-$300 planning band based on local OpenClaw usage estimate'
```

Estimate local agentic OpenClaw usage cost after or during the run:

```bash
uv run python scripts/analysis/estimate_agentic_usage_costs.py \
  outputs/taiwan_full_eval \
  --csv outputs/taiwan_full_eval/glm52_canary_usage_cost_summary.csv
```

Refresh local setup evidence and build the offline production-readiness report:

```bash
uv run python scripts/tools/run_taiwan_production_readiness_gate.py \
  --report-json outputs/taiwan_full_eval/taiwan_production_readiness_report.json \
  --fail-on-not-ready
```

The Weave content-canary gate must be fresh. The default maximum age is 24
hours, computed from `generated_at` or file mtime. Use
`--weave-content-canary-max-age-seconds SECONDS` only when the review package
intentionally uses a different freshness window.
The batch runner applies the same check before paid agentic/full execution when
`--require-weave-content-canary` is enabled.

When the expected W&B run id is known, pin completion evidence to that run so
older verifier JSON cannot satisfy the gate accidentally:

```bash
uv run python scripts/tools/run_taiwan_production_readiness_gate.py \
  --report-json outputs/taiwan_full_eval/taiwan_production_readiness_report.json \
  --required-wandb-run-id-all RUN_ID \
  --fail-on-not-ready
```

Use per-benchmark `--required-wandb-run-id benchmark=RUN_ID` only when a
deliberate multi-run review package is being assembled.

The same pinned run id must also appear in completed one-model canary review
records through `wandb_run_id` or `runs[].wandb_run_id`. The production gate
does not treat W&B completion verifier JSON as a substitute for the paid-run
review package.
Completed review records must also include passing `runs[].wandb_completion[]`
entries for the required benchmarks, so W&B logging remains part of the canary
completion condition.
The production gate reads each entry's verifier JSON `path` and checks that
`ok`, `benchmark`, and `run_id` match the review entry.
Standalone W&B completion verifier JSONs and review-linked verifier JSONs must
also be fresh. The default maximum age is 24 hours, computed from
`generated_at`; file mtime is not accepted for W&B completion freshness. The
window can be overridden with
`--wandb-completion-max-age-seconds SECONDS` only by explicit review policy.

When the report is not ready, inspect `remediation_plan`. It lists the blocking
gate names and the exact follow-up commands needed to clear them. Commands that
install/onboard NeMoClaw still require explicit third-party acceptance and
should be reviewed before execution. The wrapper refreshes the local
`nemoclaw_setup_check*.json` first, then passes that evidence into the report.

## Required Review Artifacts

```text
W&B run URL:
Weave / Agents trace URL or status:
Execution plan JSON: outputs/taiwan_full_eval/canary_full_execution_plan.json
Phase review JSON: outputs/taiwan_full_eval/canary_nonagentic_paid_run_review.json, outputs/taiwan_full_eval/canary_agentic_paid_run_review.json, outputs/taiwan_full_eval/canary_agentic_aggregate_paid_run_review.json
Batch manifest: outputs/taiwan_full_eval/batch_manifest.json
W&B completion verifier JSON: outputs/taiwan_full_eval/wandb_completion/*.json
Weave Agents verifier JSON: outputs/taiwan_full_eval/weave_agents_completion/*.json
Production readiness report: outputs/taiwan_full_eval/taiwan_production_readiness_report.json
NeMoClaw readiness JSON: outputs/taiwan_full_eval/glm52_canary_readiness_nemoclaw.json
Total Score:
Category scores:
Per-benchmark completion table:
Agentic Math summary:
SWE-Bench Pro official evaluation summary:
Cost estimate CSV:
Known failures:
Reviewer decision:
```

## Acceptance Criteria

```text
All enabled benchmark outputs are present or explicitly marked non-scoreable.
W&B completion verifier passes for Agentic Math, Agentic SWE, and final Taiwan full aggregate.
Fresh Weave Agents content canary gate passes before paid agentic/full execution.
Weave Agents verifier passes for the agentic phase, including visible content and tool spans.
If NeMoClaw is selected for the canary, --require-nemoclaw readiness passes before agentic execution.
Taiwan aggregate logs Total Score and category scores to W&B.
Weave/Agents traces show ordered conversations and tool calls for agentic tasks.
SWE-Bench Pro uses real checkout-based agentic execution, not prompt-embedded repo source.
OlymMATH-HARD uses tracked Python tool calls through OpenClaw.
Cost is explainable from local usage records and provider dashboard.
No multi-model expansion starts before this review passes.
```
