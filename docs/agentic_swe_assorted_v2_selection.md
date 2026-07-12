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

## Next Paid Validation

Do not re-run all 72 tasks for calibration. Use public prior plus a small paid
check:

- GLM-5.2: Low 8 + Middle 8, no High
- Optional: GPT-4.1-mini Low 8 only

The purpose is to confirm that V2 Low/Middle reduce empty patches and runtime
budget failures under the 40 tool-call cap.
