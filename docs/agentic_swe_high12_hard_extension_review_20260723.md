# Agentic SWE-Assorted High-12 Hard Extension Review

Status: **review only; not active**

## Recommendation

Keep the active High-8 unchanged and add these four tasks:

| Task | Language | Public pass rate | Public mean cost |
|---|---|---:|---:|
| `participle-grammar-conflict-analysis` | Go | 21.9% | $2.50 |
| `dateutil-rfc5545-timezone-interop` | Python | 36.0% | $4.21 |
| `vulture-persistent-analysis-cache` | Python | 32.5% | $3.79 |
| `clack-async-autocomplete-options` | TypeScript | 27.2% | $4.03 |

The resulting High-12 has a balanced distribution of four Go, four Python,
and four TypeScript tasks, with every task drawn from a distinct repository.

## Why four tasks

The active High-8 has a 54.4% mean public task pass rate and a maximum public
model/effort score of 87.5%. Two additions improve score resolution to ten
points but leave the strongest public result at 80%. The proposed four-task
extension:

- has a 29.4% mean public pass rate;
- reduces the maximum public High score to 72.9%;
- improves score resolution from 12.5 to 8.3 points;
- preserves strong fidelity to the complete 113-task DeepSWE matrix.

## Selection constraints

The reproducible selector is
`scripts/analysis/select_deepswe_high12_hard_extension.py`.

1. Preserve all eight active tasks as immutable anchors.
2. Add one Go, two Python, and one TypeScript task.
3. Require every candidate task to have a public pass rate from 10% to 50%.
4. Require the four-task extension to average at most 30% public pass rate.
5. Require the extension's pass rate over the top public model quartile to be
   at most 45%.
6. Require the resulting High-12 maximum public model/effort score to be at
   most 75%.
7. Require p90 runtime to fit 150 steps and 13M cumulative input tokens for
   GLM-5.2 max, Luna high, Sonnet 4.6 high, and Sol max.
8. Require valid Fable max execution and cost evidence, so missing trials
   cannot make a candidate appear artificially cheap.
9. Among valid combinations, optimize Spearman and Pearson fidelity while
   penalizing MAE and public cost.

The same four tasks are selected in every leave-one-model-family-out fold.

## Fidelity and headroom

| Metric | Active High-8 | Proposed High-12 |
|---|---:|---:|
| Pearson vs all 113 tasks | 0.917 | 0.923 |
| Spearman vs all 113 tasks | 0.905 | 0.902 |
| Kendall tau | 0.762 | 0.760 |
| MAE | 0.088 | 0.057 |
| Highest public model/effort score | 87.5% | 72.9% |
| Score resolution | 12.5 points | 8.3 points |

Selection-aware leave-one-model-family-out metrics are identical here because
the selected extension is stable in every held-out-family fold.

Expected public High-12 scores include:

| Public model/effort | High-8 | High-12 | Full DeepSWE |
|---|---:|---:|---:|
| Luna high | 46.9% | 41.7% | 44.2% |
| Luna max | 62.5% | 60.4% | 67.2% |
| Sol max | 84.4% | 62.5% | 72.7% |
| Sonnet 4.6 high | 15.6% | 20.8% | 29.9% |
| Fable max | 74.0% | 66.0% | 69.7% |
| GLM-5.2 max | 46.9% | 41.7% | 43.8% |

## Cost

The four additions cost $14.53 per model on average across the public rollout
matrix. Public model-specific added costs are approximately:

- Luna high: $2.19
- Luna max: $8.55
- GLM-5.2 max: $9.12
- Sonnet 4.6 high: $11.38
- Sol max: $21.64
- Fable max: $56.38

These are DeepSWE harness measurements, not provider quotes.

## Runtime caveat

All four additions satisfy the current 13M p90 input constraint for the
selection's reference profiles. Luna max is not one of those reference
profiles: its public p90 cumulative input is 13.5M for `vulture`, 14.0M for
`dateutil`, and 15.5M for `clack`.

Before activation, choose one of these explicit policies:

1. Keep the 13M cap and accept that some Luna-max-like runs will submit at the
   budget boundary.
2. Raise only the High cumulative input cap to 16M, retaining the 150-turn,
   200-tool, and 1800-second hard limits.

The second policy is fairer to max-effort models and remains finite.

## Environment status

All four official task definitions are present. The `participle` image is
already local; the other three images were confirmed available from the
official public ECR registry. No model call was made during this review.

## Activation gate

Do not change the active manifest yet. First:

1. Pre-pull the three missing images and run task environment preflights.
2. Run only the four additions on already-completed reference models.
3. Confirm scoreability, complete traces, budget-stop rate, cost, and ordering.
4. Activate High-12 once, version it explicitly, and stop changing task
   membership for the provisional launch.

Artifacts:

- `outputs/deepswe_subset_analysis/essential_anchored_high_12_hard_extension_review_20260723_summary.json`
- `outputs/deepswe_subset_analysis/essential_anchored_high_12_hard_extension_review_20260723_task_names.json`
- `outputs/deepswe_subset_analysis/essential_anchored_high_12_hard_extension_review_20260723_selection_aware_oof.csv`
