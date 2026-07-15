# Agentic SWE-Assorted V2 Selection

Agentic SWE-Assorted now uses `low_middle_v2_72.jsonl` as the default
SWE-bench Lite Low/Middle input.

## Inputs

- Source dataset: `princeton-nlp/SWE-bench_Lite`, split `test`
- Public prior: `SWE-bench/experiments`, `evaluation/lite/*/results/results.json`
- Public cache: `data/taiwan/swebench_lite_assorted/source/swebench_lite_public_results.json`
- Local pilot summary:
  `data/taiwan/swebench_lite_assorted/source/glm52_lm12m12_20260712_pilot_summary.json`

The public prior is used only to calibrate difficulty. It is not used as a model
score, and it is documented as contamination-prone public data.

## V2 Rules

Low:

- `public_lite_seen_count >= 20`
- `public_lite_resolve_rate >= 0.65`
- `static_difficulty_percentile <= 0.85`
- no local GLM-5.2 pilot unresolved signal
- no pilot cost-risk signal: empty patch, runtime budget exceeded, provider
  transient exhausted, or tool policy violation
- repository cap: 6

Middle:

- `public_lite_seen_count >= 20`
- `0.25 <= public_lite_resolve_rate <= 0.72`
- `static_difficulty_percentile <= 0.95`
- no pilot cost-risk signal
- repository cap: 6

Legacy static-only slices are retained as `low_36`, `middle_36`, and
`low_middle_72`.

## Generated Subsets

| subset | count | public resolve rate mean | median | min | max |
|---|---:|---:|---:|---:|---:|
| `low_v2_36` | 36 | 0.849 | 0.853 | 0.690 | 0.940 |
| `middle_v2_36` | 36 | 0.490 | 0.500 | 0.293 | 0.694 |
| `low_middle_v2_72` | 72 | 0.669 | 0.692 | 0.293 | 0.940 |

The current default runner input is:

```text
data/taiwan/swebench_lite_assorted/subsets/low_middle_v2_72.jsonl
```

The current default High input is:

```text
data/taiwan/deepswe/subsets/essential_anchored_high_8_glm52max_cap100_10m_lang_balanced.jsonl
```

This High-8 slice was selected from public DeepSWE v1.1 rollouts with explicit
GLM-5.2 max budget constraints:

- language balance: Go 3, Python 2, TypeScript 3
- public all-model average steps <= 75
- public all-model average input tokens <= 7M
- public GLM-5.2 max average steps <= 100
- public GLM-5.2 max average input tokens <= 10M

The runner uses a stricter paid-run gate before launching DeepSWE High:

- default High cap: `120` turns / `120` tool calls / `12M` cumulative input tokens
- preflight basis: public DeepSWE rows for the current model and `--thinking`
- hard statistic: public p75 by default
- policy: `--deepswe-budget-preflight error`

If the selected High tasks are known from public data to exceed the configured
caps, the runner writes `inputs/deepswe_budget_preflight.json` and exits before
paid OpenClaw execution. Use `--allow-deepswe-budget-mismatch` only for an
intentional experiment.

## Scoring

The default full set keeps the practical task mix at 36 Low, 36 Middle, and 8
High tasks, but the official Agentic SWE-Assorted score is a tier macro average:

```text
Score = (Low Pass@1 + Middle Pass@1 + High Pass@1) / 3
```

This keeps the Frontier/High tier visible even though it is intentionally small
for cost control. The runner exposes this as `weighted_pass_at_1` and records the
old instance-level micro average separately as `micro_pass_at_1`.

If a run omits any positive-weight tier, for example `--skip-high`, the formal
`weighted_pass_at_1` is left unset and `weighted_pass_at_1_complete` is false.
`weighted_present_pass_at_1` is logged only as a diagnostic for partial pilots.

## Validation

Implemented checks:

- Public per-instance prior is cached and recorded in each row.
- V2 rows retain full SWE-bench Lite provenance.
- V2 Low and Middle have no overlap.
- V2 Low has higher public solve-rate prior than V2 Middle.
- Runner defaults point to `low_middle_v2_72`.
- Runner scoring records complete official tier macro score only when all
  positive-weight tiers are present.

Command run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  tests/test_swebench_lite_assorted_selection.py \
  tests/test_agentic_swe_assorted.py
```

Result: `15 passed`.

Dry-run command:

```bash
python3 scripts/tools/run_agentic_swe_assorted.py \
  --model dummy/local \
  --output-dir outputs/agentic_swe_assorted_v2_dryrun \
  --low-limit 1 \
  --middle-limit 1 \
  --skip-high \
  --dry-run \
  --no-docker-check \
  --no-verify-weave-agents \
  --no-require-actual-token-usage
```

Result: v2 Low/Middle input preparation and SWE-bench Lite prepare-only path
completed.

## Runtime Setup

Agentic SWE-Assorted uses OpenClaw/NeMoClaw as the agent runtime. Before paid
runs, apply the repo-managed OpenClaw runtime patch instead of relying on a
local one-off edit:

```bash
NEMOCLAW_SANDBOX=nejumi-taiwan scripts/setup/install_openclaw_budget_guard.sh
```

This installs the Nejumi budget guard plugin and patches both the host OpenClaw
package and the NeMoClaw Gateway runtime package. The patch is required for:

- hard budget enforcement for turns, cumulative tokens, and tool wall time
- actual token usage exposure from OpenClaw diagnostic events
- passing `xhigh` and `max` thinking levels through OpenClaw validation when the
  provider/API supports them

Do not start a paid High or full Assorted run if this check fails:

```bash
python3 scripts/tools/check_agentic_swe_assorted_high_readiness.py \
  --model openai-direct/gpt-5.6-luna
```

The readiness report contains an `openclaw_runtime` section. It accepts either
the repo-managed patch or an upstream OpenClaw runtime that already exposes the
same capability. If either the host or NeMoClaw Gateway runtime is missing the
required capability, the report is `not_ready` and prints the setup command to
run.

## Next Paid Validation

Do not use paid runs to rediscover cap mismatches already visible in public
DeepSWE data. The order is:

1. regenerate or inspect the High subset from public trials;
2. run the Assorted runner in dry-run/preflight mode;
3. launch a small paid High sample only if `deepswe_budget_preflight.json` is
   clean;
4. expand to the full 36/36/8 suite after Low, Middle, and High all have clean
   harness behavior.
