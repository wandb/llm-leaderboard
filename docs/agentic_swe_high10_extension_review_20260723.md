# Agentic SWE-Assorted High-10 Extension Review

Status: **approved and activated on 2026-07-24**

## Why this review exists

The frozen High-8 produced 7/8 passes for GPT-5.6 Luna high in the current
harness, while the same model/effort has a 15/32 (46.9%) public DeepSWE pass
rate over those eight tasks. A single run does not prove that the subset is
invalid, but 8 tasks also give only 12.5-point resolution and leave little
room above a 7/8 result.

This review preserves all existing tasks and adds two tasks. It does not
replace or reshuffle the frozen eight.

## Proposed extension

| Task | Language | Repository | Public pass rate | Public mean cost |
|---|---|---|---:|---:|
| `participle-grammar-conflict-analysis` | Go | `alecthomas/participle` | 21.9% | $2.50 |
| `bandit-incremental-cache-control` | Python | `PyCQA/bandit` | 50.6% | $4.25 |

The resulting language distribution is Go 4, Python 3, TypeScript 3. Every
task uses a distinct repository.

## Selection method

The reproducible selector is
`scripts/analysis/select_deepswe_high_extension.py`.

1. Keep the active High-8 as immutable anchors.
2. Add exactly one Go task and one Python task.
3. Exclude repositories already represented by the anchors and four tasks
   previously rejected after empirical runtime or environment failures.
4. Require each candidate to fit p90 150 agent turns and 13M cumulative input
   tokens for public GLM-5.2 max, GPT-5.6 Luna high, Claude Sonnet 4.6 high,
   and GPT-5.6 Sol max rollouts.
5. Require valid nonzero execution and cost evidence for Claude Fable 5 max;
   missing cost rows cannot make a candidate look artificially cheap.
6. Optimize full-DeepSWE Pearson/Spearman fidelity while penalizing MAE,
   public cost, an overly easy top quartile, and loss of upper-end headroom.
7. For selection-aware validation, freeze the original eight and reselect
   only the added pair after holding out each model family.

Model family means the public base-model identifier. Effort variants of one
base model stay in the same held-out family.

## Results

| Metric | Active High-8 | Proposed High-10 |
|---|---:|---:|
| Pearson vs all 113 tasks | 0.917 | 0.947 |
| Spearman vs all 113 tasks | 0.905 | 0.932 |
| Kendall tau | 0.762 | 0.801 |
| MAE | 0.088 | 0.060 |
| Highest public model/effort score | 87.5% | 80.0% |
| Score resolution | 12.5 points | 10 points |

Selection-aware leave-one-model-family-out validation for the two-task
extension gives Pearson 0.921, Spearman 0.897, Kendall 0.753, and MAE 0.077
over 40 public model/effort rows.

The public mean added cost is about $6.75 per evaluated model for both tasks
combined. The model-specific public estimates are $1.10 for Luna high, $4.04
for GLM max, $5.69 for Sonnet high, $11.22 for Sol max, and $25.39 for Fable
max. These are DeepSWE harness estimates, not provider quotes. The proposed
extension improves resolution but does not solve the separate problem that
Fable max is intrinsically expensive on long-horizon SWE tasks.

## Runtime evidence

The worst p90 among the four reference profiles is:

- `participle`: 91.6 turns for Sonnet high and 7.71M input tokens.
- `bandit`: 118.4 turns for GLM max and 7.78M input tokens.

Both fit the current High limits of 150 turns and 13M cumulative input tokens.
Their environment images install Go/Python dependencies at image-build time.

## Activation rule

Do not activate this proposal while a paid frozen-80 run is in progress.
Treat upper-end headroom as materially insufficient when either:

- at least two of the fixed Luna high, Sonnet high, GLM-5.2, and Sol max
  reference runs score 7/8 or better; or
- the High-8 produces a top-tier tie or rank reversal that conflicts with
  both the complete public DeepSWE matrix and the Low/Middle ordering.

A single 7/8 rollout is evidence for review, not enough by itself to change
the benchmark. After the current paid run finishes:

1. Run environment/preflight checks for only the two additions.
2. Run the two additions for already-completed reference models.
3. Compare observed runtime, budget-stop rate, and score ordering with public
   expectations.
4. Activate High-10 only if the headroom criterion is met, both additions are
   scoreable, traces are complete, and the observed ordering is no worse.

Review artifacts:

- `outputs/deepswe_subset_analysis/essential_anchored_high_10_extension_review_20260723_summary.json`
- `outputs/deepswe_subset_analysis/essential_anchored_high_10_extension_review_20260723_scatter_in_sample.png`
- `outputs/deepswe_subset_analysis/essential_anchored_high_10_extension_review_20260723_scatter_leave_family_out.png`
