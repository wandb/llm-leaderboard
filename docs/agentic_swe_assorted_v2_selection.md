# Agentic SWE-Assorted V2 Selection

> Historical note: this document describes the 36/36/8 v2 release. The active
> 20/20/10 release is documented in
> `docs/agentic_swe_assorted_20_20_10_selection.md`.

Agentic SWE-Assorted v2 used `low_middle_v2_72.jsonl` as the default
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
- no empirical exclusion from local W&B GLM validation
- repository cap: soft target 6; if candidate coverage is otherwise
  insufficient, the generator fills remaining slots from the best eligible rows

Middle:

- `public_lite_seen_count >= 20`
- `0.25 <= public_lite_resolve_rate <= 0.72`
- `static_difficulty_percentile <= 0.95`
- `PASS_TO_PASS <= 150`
- no pilot cost-risk signal
- no empirical exclusion from local W&B GLM validation
- repository cap: soft target 6; current generated Middle has one fallback
  `django/django` slot, so its max repo count is 7

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

## Selection Objective

The three tiers have different selection objectives. They are not all intended
to approximate their source benchmark.

- Low and Middle provide useful score range below the frontier tier. Public
  SWE-bench Lite outcomes are difficulty priors used to separate the tiers;
  Low and Middle are not optimized for correlation with the complete Lite
  leaderboard.
- High is the compact frontier slice. Its task selection is expected to retain
  useful out-of-family rank correlation with the complete DeepSWE result
  matrix while keeping runtime and token cost bounded.

Before changing Low or Middle membership, compare the current and proposed
slices on a small fixed reference-model panel. Prefer the proposed slice only
when it improves the benchmark's intended behavior without materially raising
cost:

1. aggregate pass rates follow `Low > Middle > High`;
2. Low and Middle preserve a sensible pairwise ordering of the reference
   models and avoid an all-zero floor;
3. stronger reference models generally score at least as well as weaker ones
   within Low and Middle, allowing for finite-sample variation;
4. Low is cheaper and shorter than Middle, and both are materially cheaper
   than High;
5. runtime-budget exhaustion is exceptional, with no systematic exhaustion in
   Low or Middle; and
6. repository concentration and environment failures do not dominate a tier.

These checks use results from this harness. They do not require running every
SWE-bench Lite task or reproducing the full Lite ranking. Provider incidents
are tracked separately and are not a reason to alter task membership.

## Runtime Budgets

Low and Middle use the same bounded runtime profile:

- 40 agent turns;
- 40 tool calls;
- 1,000,000 input tokens per model context;
- 1,000,000 cumulative input tokens across the task; and
- 500,000 cumulative output tokens.

The staged English warnings state that the current diff is submitted when a
limit is reached. A runtime stop is therefore scoreable, not an infrastructure
failure, and is never automatically retried.

This profile was challenged because 26/72 GPT-4.1 mini, 13/72 GPT-5.6 Luna
High, and 51/72 Claude Sonnet 4.6 rollouts reached a limit. A fixed Sonnet
pilot reran four Low and four Middle stopped tasks with 60 turns, 80 tools,
and 2,000,000 cumulative input tokens. Both the original and expanded diffs
passed all 8/8 official tests; the expansion added $9.02 and improved zero
binary outcomes. All expanded runs still reached a limit. This supports
retaining the 40/40/1M profile: the high Sonnet stop rate measures failure to
terminate efficiently, not lack of time to produce a correct patch.

Budget calibration uses one common tier profile across GPT-4.1 mini, GPT-5.6
Luna, Claude Sonnet 4.6, W&B GLM-5.2, and GPT-5.6 Sol. It considers official
grading of stopped diffs, not stop rate alone. Provider- or model-specific
budget advantages are not allowed; only the documented Low/Middle versus High
tier distinction changes the cap. Missing trace evidence is handled as an
observability repair and must not trigger a full model rerun.

The current default High input is:

```text
data/taiwan/deepswe/subsets/essential_anchored_high_8_wandb_glm52_cap100_10m_lang_balanced.jsonl
```

Status: frozen default as of 2026-07-15. Do not change task membership in this
subset in place. If a future High profile is needed, create a new subset name,
update the config/tests explicitly, and treat that as a separate approval item.

This High-8 slice was selected from public DeepSWE v1.1 rollouts with explicit
GLM-5.2 max budget constraints, then filtered to remove local W&B GLM-5.2
failure/cap-risk tasks observed in paid validation:

- language balance: Go 3, Python 2, TypeScript 3
- public all-model average steps <= 75
- public all-model average input tokens <= 7M
- public GLM-5.2 max average steps <= 100
- public GLM-5.2 max average input tokens <= 10M
- excluded task names:
  `termenv-preserve-ansi-resets`, `superjson-error-stack-serialization`,
  `ts-pattern-match-each`, `kcp-go-multiplexed-kcp-streams`

The selected High-8 tasks are:

| rank | task | language |
|---:|---|---|
| 1 | `go-genai-streamed-function-args` | Go |
| 2 | `etree-xml-diff-patch` | Go |
| 3 | `ytt-jsonpath-query-api` | Go |
| 4 | `mnamer-daemon-watch-lifecycle` | Python |
| 5 | `langchain-request-coalescing` | Python |
| 6 | `ofetch-per-origin-circuit-breaker` | TypeScript |
| 7 | `kea-atomic-signal-selectors` | TypeScript |
| 8 | `happy-dom-deterministic-intersectionobserver` | TypeScript |

Public DeepSWE approximation metrics for this High-8:

| metric | value |
|---|---:|
| Pearson | 0.917 |
| Spearman | 0.905 |
| Kendall | 0.762 |
| MAE | 0.088 |
| constrained leave-one-family-out reselection Pearson | 0.848 |
| constrained leave-one-family-out reselection Spearman | 0.800 |
| selector leave-one-family-out Pearson mean | 0.763 |
| selector leave-one-family-out Spearman mean | 0.758 |
| public mean cost | `$3.27` |
| public mean steps | 49.9 |
| public mean input tokens | 4.12M |

Scatter outputs:

```text
outputs/deepswe_subset_analysis/essential_anchored_high_8_wandb_glm52_cap100_10m_lang_balanced_scatter_in_sample.png
outputs/deepswe_subset_analysis/essential_anchored_high_8_wandb_glm52_cap100_10m_lang_balanced_scatter_leave_family_out.png
```

The runner uses a stricter paid-run gate before launching DeepSWE High:

- default High cap: `150` turns / `200` tool calls / `13M` cumulative input tokens
- preflight basis: public DeepSWE rows for the current model and `--thinking`
- hard statistic: public p90 by default
- policy: `--deepswe-budget-preflight error`

The tool-call cap is intentionally higher than the turn cap because one model
turn can issue multiple tool calls; DeepSWE's public `n_agent_steps` is therefore
not a valid one-to-one proxy for tool calls. The 150-turn and 13M-input limits
cover the largest per-task p90 values in the frozen High-8 GLM-5.2 max public
rollouts (134.2 steps and 12.307M input tokens) while still bounding the longest
roughly 10% tail.

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

### Diagnostic partial credit

The official score above remains binary and is the only primary leaderboard
score. A separate `diagnostic_score_with_partial` is emitted to improve the
resolution of the eight-task High tier and to help distinguish a nearly useful
patch from an empty or unrelated patch.

For an unresolved but scoreable task with an applied patch:

```text
partial = 0.3 * (F2P pass fraction * P2P pass fraction)^2
```

- A resolved task receives diagnostic score `1.0`, exactly like the official
  binary score.
- An unresolved task can receive at most `0.3` diagnostic points.
- `F2P` means tests that must change from failing to passing. Passing only
  pre-existing `P2P` regression tests cannot create any partial credit.
- `P2P` is a multiplicative regression penalty. If a task defines no P2P
  tests, its neutral fraction is `1.0`.
- Empty, unapplied, unscoreable, policy-disqualified, or missing-evidence
  results receive zero partial credit.
- Lite evidence is read from the official SWE-bench per-instance report.
  High evidence comes from DeepSWE verifier `f2p_*` and `p2p_*` counts.

The diagnostic tier score and full-set score use the same equal-tier macro
weights as the official score. Both are written to `summary.json`,
`leaderboard_table.json`, the Markdown report, and W&B. This diagnostic must
always be displayed with its full name; it is not a replacement for Pass@1.

## Validation

Implemented checks:

- Public per-instance prior is cached and recorded in each row.
- V2 rows retain full SWE-bench Lite provenance.
- V2 Low and Middle have no overlap.
- V2 Low has higher public solve-rate prior than V2 Middle.
- Runner defaults point to `low_middle_v2_72`.
- Runner defaults point to
  `essential_anchored_high_8_wandb_glm52_cap100_10m_lang_balanced` for High.
- Runner scoring records complete official tier macro score only when all
  positive-weight tiers are present.
- Official binary and diagnostic partial-credit scores are stored separately;
  the diagnostic formula cannot be inflated by P2P-only success.

Command run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  tests/test_swebench_lite_assorted_selection.py \
  tests/test_agentic_swe_assorted.py \
  tests/test_agentic_swe_assorted_high_readiness.py \
  tests/test_taiwan_full_config_generation.py \
  tests/test_deepswe_essential_subset.py
```

Result: `61 passed`.

Dry-run command:

```bash
python3 scripts/tools/run_agentic_swe_assorted.py \
  --model dummy/local \
  --output-dir outputs/agentic_swe_assorted_v2_wandb_glm52_dryrun \
  --low-limit 1 \
  --middle-limit 1 \
  --high-limit 1 \
  --dry-run \
  --no-docker-check \
  --no-verify-weave-agents \
  --no-require-actual-token-usage
```

Result: v2 Low/Middle input preparation, SWE-bench Lite prepare-only path,
DeepSWE High prepare-only path, and equal-tier weighted scoring completed.

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
