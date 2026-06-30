# Taiwan Paid Run Review Template

Use this before any substantial paid evaluation run and update it after completion.

## Pre-Run

```text
Run purpose:
Decision this run should support:
Model(s):
Benchmark scope:
Canary or final expansion:
Expected cost band:
Expected runtime:
Known risks:
Stop condition:
Approver / requester:
```

## Completion Evidence

```text
W&B run(s):
W&B completion verifier result(s):
Paid-run review W&B completion entries:
Weave / Agents trace status:
Weave Agents verifier result(s):
Paid-run review Weave Agents completion entries:
Output directory:
Total Score status:
Per-benchmark completion:
Actual cost estimate:
Failures / retries:
Reusable artifacts:
Training-data usability:
Reviewer decision:
```

## Release Evidence Bundle

One-command local release gate:

```bash
python3 scripts/tools/run_taiwan_release_gate.py \
  --timestamp TIMESTAMP \
  --readiness-report-json temp/taiwan_release_gate_readiness_TIMESTAMP.json \
  --bundle-output-dir outputs/taiwan_release_evidence/bundle_TIMESTAMP \
  --bundle-verification-json temp/taiwan_release_gate_bundle_verify_TIMESTAMP.json \
  --release-gate-json temp/taiwan_release_gate_TIMESTAMP.json \
  --required-wandb-run-id-all WANDB_RUN_ID
```

Use `--require-ready` on the final release check to make the command exit
nonzero unless readiness, W&B completion evidence, paid-run review evidence,
Weave content evidence, and bundle integrity all pass.

Existing local result audit:

```bash
python3 scripts/tools/audit_taiwan_existing_results.py \
  --json temp/taiwan_existing_results_audit_TIMESTAMP.json \
  --markdown temp/taiwan_existing_results_audit_TIMESTAMP.md \
  --fail-on-unformalized
```

`run_taiwan_production_readiness_gate.py` and `run_taiwan_release_gate.py`
run this audit automatically unless `--skip-existing-results-audit` is used.
Complete local results without matching W&B completion verifier JSONs block
production readiness; partial/probe outputs are listed but not release evidence.
When the audit reports an unformalized complete SWE-Bench Pro result, relog it
with the audit-provided dry-run command first. Review the generated
`temp/wandb_relog_plans/*.plan.json`, then run the audit-provided re-log write
command with `--validated-dry-run-plan-json`. The relog writer refuses W&B
login/write if the reviewed plan no longer matches the current source files,
source SHA-256 values, `config.relog.source_sha256`, log tables, artifact
aliases, or post-log verifier command. The post-log verifier command should
also assert each `relog.source_sha256.*` run config key. After W&B logging, run
`scripts/tools/verify_taiwan_wandb_completion.py --benchmark agentic_swe`.

Current low-cost one-model canary path:

```bash
uv run python scripts/tools/run_taiwan_full_eval_batch.py \
  --manifest configs/taiwan_openai_canary_models.yaml \
  --canary \
  --prepare-only \
  --phase full \
  --generated-config-dir configs/taiwan_full/generated_openai_canary \
  --output-root outputs/taiwan_full_eval \
  --run-purpose 'Prepare OpenAI-direct gpt-4.1-mini one-model full canary metadata' \
  --expected-cost-band 'prepare-only; no model API calls'

uv run python scripts/tools/run_taiwan_full_eval_batch.py \
  --manifest configs/taiwan_openai_canary_models.yaml \
  --canary \
  --phase nonagentic \
  --generated-config-dir configs/taiwan_full/generated_openai_canary_nonagentic \
  --output-root outputs/taiwan_full_eval \
  --wandb-run-id-prefix twcanary-openai-mini-YYYYMMDD \
  --yes \
  --run-purpose 'OpenAI-direct gpt-4.1-mini one-model nonagentic phase' \
  --expected-cost-band 'low-cost OpenAI-direct canary; confirm cap before execution'

uv run python scripts/tools/run_taiwan_full_eval_batch.py \
  --manifest configs/taiwan_openai_canary_models.yaml \
  --canary \
  --phase agentic \
  --generated-config-dir configs/taiwan_full/generated_openai_canary_agentic \
  --output-root outputs/taiwan_full_eval \
  --wandb-run-id-prefix twcanary-openai-mini-YYYYMMDD \
  --verify-wandb-completion \
  --verify-weave-agents \
  --weave-agents-require-tool-span \
  --weave-agents-require-tool-content \
  --weave-agents-require-usage \
  --weave-content-canary-gate WEAVE_CONTENT_CANARY_GATE \
  --require-weave-content-canary \
  --yes \
  --run-purpose 'OpenAI-direct gpt-4.1-mini one-model agentic phase' \
  --expected-cost-band 'low-cost OpenAI-direct canary; confirm cap before execution'

uv run python scripts/tools/run_taiwan_full_eval_batch.py \
  --manifest configs/taiwan_openai_canary_models.yaml \
  --canary \
  --phase agentic_aggregate \
  --generated-config-dir configs/taiwan_full/generated_openai_canary_agentic_aggregate \
  --output-root outputs/taiwan_full_eval \
  --wandb-run-id-prefix twcanary-openai-mini-YYYYMMDD \
  --verify-wandb-completion \
  --yes \
  --run-purpose 'OpenAI-direct gpt-4.1-mini one-model aggregate phase' \
  --expected-cost-band 'no new OpenClaw generation expected'
```

Sync W&B completion verifier JSONs into a paid-run review by generating a
dry-run report first:

```bash
python3 scripts/tools/sync_wandb_completion_to_paid_review.py \
  --review-json outputs/taiwan_full_eval/canary_agentic_paid_run_review.json \
  --completion-json outputs/taiwan_full_eval/wandb_completion/BENCHMARK-RUN_ID.json \
  --set-verify-wandb-completion \
  --actual-cost-estimate '$ACTUAL_OR_BILLING_ESTIMATE' \
  --provider-bill-reference 'BILL_OR_DASHBOARD_REFERENCE' \
  --report-json temp/wandb_completion_BENCHMARK-RUN_ID.sync_dry_run.json
```

After reviewing the dry-run report, apply the same sync with the validated
dry-run report attached:

```bash
python3 scripts/tools/sync_wandb_completion_to_paid_review.py \
  --review-json outputs/taiwan_full_eval/canary_agentic_paid_run_review.json \
  --completion-json outputs/taiwan_full_eval/wandb_completion/BENCHMARK-RUN_ID.json \
  --in-place \
  --set-verify-wandb-completion \
  --actual-cost-estimate '$ACTUAL_OR_BILLING_ESTIMATE' \
  --provider-bill-reference 'BILL_OR_DASHBOARD_REFERENCE' \
  --validated-dry-run-report-json temp/wandb_completion_BENCHMARK-RUN_ID.sync_dry_run.json
```

Replace `$ACTUAL_OR_BILLING_ESTIMATE` and `BILL_OR_DASHBOARD_REFERENCE`
before running the command. The sync tool and paid-run review gate reject
these generated placeholders, as well as `TBD`, `TODO`, `pending`, and similar
placeholder text.

For the phased one-model canary, use these review JSONs:

```text
agentic_math / agentic_swe: outputs/taiwan_full_eval/canary_agentic_paid_run_review.json
taiwan_full: outputs/taiwan_full_eval/canary_agentic_aggregate_paid_run_review.json
```

When adopting an already-finished W&B run instead of a run created by the
current batch review, use a machine-readable scope-attestation JSON. Manual
`--scope-confirmed-*` flags are intentionally rejected by the sync tool because
they do not leave a portable review artifact.

The release gate drafts editable scope-attestation JSON templates for
formalized existing W&B results:

```bash
python3 scripts/tools/run_taiwan_release_gate.py \
  --timestamp TIMESTAMP \
  --release-gate-json temp/taiwan_release_gate_TIMESTAMP.json
```

For a candidate such as `agentic_math-f1veetyb`, review and edit:

```text
temp/taiwan_wandb_adoption_attestations_TIMESTAMP/agentic_math-f1veetyb.scope_attestation.json
```

Set `confirmed=true`, replace reviewer/time/cost/bill placeholders with
concrete values, and then generate a dry-run report with the
machine-readable attestation file:

```bash
python3 scripts/tools/sync_wandb_completion_to_paid_review.py \
  --review-json outputs/taiwan_full_eval/canary_agentic_paid_run_review.json \
  --completion-json outputs/taiwan_full_eval/wandb_completion/agentic_math-f1veetyb.json \
  --top-level \
  --set-verify-wandb-completion \
  --adopt-existing-result \
  --scope-attestation-json temp/taiwan_wandb_adoption_attestations_TIMESTAMP/agentic_math-f1veetyb.scope_attestation.json \
  --report-json temp/taiwan_wandb_adoption_attestations_TIMESTAMP/agentic_math-f1veetyb.sync_dry_run.json
```

After reviewing the dry-run report, apply the same sync with the validated
dry-run report attached:

```bash
python3 scripts/tools/sync_wandb_completion_to_paid_review.py \
  --review-json outputs/taiwan_full_eval/canary_agentic_paid_run_review.json \
  --completion-json outputs/taiwan_full_eval/wandb_completion/agentic_math-f1veetyb.json \
  --in-place \
  --top-level \
  --set-verify-wandb-completion \
  --adopt-existing-result \
  --scope-attestation-json temp/taiwan_wandb_adoption_attestations_TIMESTAMP/agentic_math-f1veetyb.scope_attestation.json \
  --validated-dry-run-report-json temp/taiwan_wandb_adoption_attestations_TIMESTAMP/agentic_math-f1veetyb.sync_dry_run.json
```

The JSON path is not a loose note. The sync command validates that its
`benchmark`, `run_id`, `completion_path`, and `review_path` match the verifier
JSON and review JSON before mutating the review. The apply step also validates
the dry-run report's source review SHA and scope-attestation source SHA, then
records `sync_dry_run_report_json`, `sync_dry_run_source_review_json`, and
`sync_dry_run_source_review_sha256` in the W&B completion entry. Existing-run
adoption should use `--top-level` unless the target paid-run review already has
a matching `runs[].wandb_run_id`; current generated canary review JSONs start
with empty `runs`, so top-level sync is the default safe path. The release gate
now generates the concrete command for each adoption candidate; prefer that
command over manually replacing `PHASE`.

The paid-review doctor and production-readiness gate require those sync fields
for adopted existing W&B results. A hand-edited `adopted_existing_result=true`
entry with only a scope attestation is rejected unless the dry-run report is
readable, its `source_review_sha256` matches `sync_dry_run_source_review_json`,
its entries/changes include the same W&B completion entry, and
`scope_attestation.source_attestation_sha256` matches the source attestation
JSON.

The sync command rejects legacy or incomplete verifier JSONs by default. The
completion JSON must have `ok=true`, `verification_schema_version=1`, numeric
`generated_at`, valid `observed_evidence.run_state=finished`, and no failed
verifier subprocess markers such as `returncode_ok=false` or `payload_ok=false`.
Existing-result adoption additionally requires `adopted_existing_result=true`
and a matching `scope_attestation` whose `source_attestation_json` is present,
readable, confirmed, and matches the paid-review entry for benchmark, run id,
completion path, review path, actual cost estimate, and provider bill
reference. The release gate fails adopted results whose attestation source is
missing or does not match the W&B completion entry. The release evidence bundle
includes the generated attestation template and, after sync, the confirmed
attestation source JSON.
Generated paid-run review JSONs include `completion_requirements`; use that
object as the phase-local checklist after execution. Completed review records
must include both `actual_cost_estimate` and `provider_bill_reference`, not only
the W&B run id and verifier paths. Placeholder values such as `TBD`, `TODO`,
`pending`, `$ACTUAL_OR_BILLING_ESTIMATE`, or `BILL_OR_DASHBOARD_REFERENCE` are
not valid completion evidence.

Sync Weave Agents verifier JSONs into the same paid-run review after a native
Agents verifier has passed. Generate a dry-run report first:

Before sync, operators can save a read-only Agents diagnostic for the same
agent name. This is not a substitute for the verifier JSON, but it makes
timestamp/content/order issues visible during review. The diagnostic JSON must
keep the `.agents.json` suffix; current output includes
`diagnostic_schema_version=1`, numeric `generated_at`, and W&B Agents API
`query_source` metadata:

```bash
python3 scripts/tools/run_openclaw_agent_protocol.py check-agents \
  --entity llm-leaderboard \
  --project tc-leaderboard \
  --agent-name nejumi-taiwan-openclaw \
  --limit 20 \
  --conversation-id-contains RUN_ID_OR_TASK_ID \
  --json temp/weave_agents_completion_PHASE_MODEL_SLUG.agents.json
```

```bash
python3 scripts/tools/sync_weave_agents_completion_to_paid_review.py \
  --review-json outputs/taiwan_full_eval/canary_agentic_paid_run_review.json \
  --completion-json outputs/taiwan_full_eval/weave_agents_completion/PHASE-MODEL_SLUG.json \
  --run-id RUN_ID \
  --set-verify-weave-agents \
  --report-json temp/weave_agents_completion_PHASE_MODEL_SLUG.sync_dry_run.json
```

After reviewing the dry-run report, apply the same sync with the validated
dry-run report attached:

```bash
python3 scripts/tools/sync_weave_agents_completion_to_paid_review.py \
  --review-json outputs/taiwan_full_eval/canary_agentic_paid_run_review.json \
  --completion-json outputs/taiwan_full_eval/weave_agents_completion/PHASE-MODEL_SLUG.json \
  --run-id RUN_ID \
  --in-place \
  --set-verify-weave-agents \
  --validated-dry-run-report-json temp/weave_agents_completion_PHASE_MODEL_SLUG.sync_dry_run.json
```

The Weave sync command rejects failed or structure-only verifier JSONs by
default. The verifier JSON must have `ok=true`, `verification_schema_version=1`,
numeric `generated_at`, a visible `latest_trace_id`, all checks passing, visible
required message/tool content, parseable non-reversed timestamps for every
latest-trace span, chronological trace spans with no tool span before visible
user/problem input, and no tool span starting after or at the same time as the
final-answer span end time. It must also prove run scope: the verifier JSON's
`required_evidence.conversation_id` or
`required_evidence.conversation_id_contains` must include the target W&B
`run_id`. `run_taiwan_full_eval_batch.py` uses the per-run `WANDB_RUN_ID` by
default; if you manually pass `--weave-agents-conversation-id-contains`, include
the actual run id or the `{wandb_run_id}` placeholder.

Check the paid-run review package before release gating:

```bash
python3 scripts/tools/check_taiwan_paid_run_review_package.py \
  --review-json outputs/taiwan_full_eval/canary_agentic_paid_run_review.json \
  --required-wandb-benchmark agentic_math \
  --required-wandb-benchmark agentic_swe \
  --required-wandb-benchmark taiwan_full \
  --require-one-model-canary \
  --json temp/taiwan_paid_run_review_check_TIMESTAMP.json \
  --markdown temp/taiwan_paid_run_review_check_TIMESTAMP.md \
  --fail-on-invalid
```

Lower-level bundle commands:

```bash
python3 scripts/tools/build_taiwan_release_evidence_bundle.py \
  --readiness-report temp/taiwan_production_readiness_gate_TIMESTAMP.json \
  --output-dir outputs/taiwan_release_evidence/bundle_TIMESTAMP

python3 scripts/tools/verify_taiwan_release_evidence_bundle.py \
  --bundle-dir outputs/taiwan_release_evidence/bundle_TIMESTAMP \
  --require-ready
```

## Policy

```text
Development/debug runs should be small, capped, and reviewable.
One complete model must pass review before expanding to multiple models.
Completed one-model review records must include passing W&B completion verifier entries for every required benchmark.
Each W&B completion entry must point to the verifier JSON used to prove the matching run id and benchmark.
W&B completion verifier JSONs must include generated_at and be fresh for the release review unless the reviewer explicitly accepts another freshness window.
New W&B completion verifier JSONs include `verification_schema_version`; production readiness reports legacy verifier JSONs under `legacy_schema_records` so they can be refreshed before final release.
New W&B completion verifier JSONs include `observed_evidence`; reviewers should confirm it shows the finished W&B run state, observed scalar metrics, W&B table row counts, and required production artifact aliases.
The production-readiness paid-run review gate verifies each referenced completion JSON, not just the presence of a path.
Paid-run review and one-model canary gates reject legacy verifier JSONs that do not have `verification_schema_version=1` and valid `observed_evidence`.
The W&B completion-to-review sync tool applies the same default policy before it mutates a review JSON.
The Weave Agents completion-to-review sync tool applies the same default policy before it mutates a review JSON, including trace order and required content checks; use a `--report-json` dry-run followed by apply with `--validated-dry-run-report-json` for reviewable paid-run mutation.
Production readiness blocks on paid_run_review_package until completed review JSONs include run evidence, W&B run ids for successful runs, completion status, actual_cost_estimate, W&B completion entries, and required Weave Agents completion entries.
Completed paid-run review JSONs must also include provider_bill_reference so the release evidence can tie the estimate back to a provider bill, export, or dashboard.
Claude Opus-class models are final-only unless explicitly requested.
Third-party API outputs are evaluation records, not training data.
```
