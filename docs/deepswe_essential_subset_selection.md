# DeepSWE-Essential Subset Selection

DeepSWE-Essential is a budgeted DeepSWE subset family selected from the public
DeepSWE v1.1 rollout matrix. It is not a replacement for full DeepSWE. It is a
cost-control tier intended to preserve useful rank signal while reducing paid
agentic evaluation cost.

## Source Data

- Public rollout outcomes: `https://deepswe.datacurve.ai/artifacts/v1.1/trials.json`
- Public task metadata: `https://deepswe.datacurve.ai/artifacts/v1.1/tasks.json`
- Matrix used for selection: 40 model-efforts x 113 DeepSWE tasks after public
  `included_in_score` filtering.

## Method

For each model-effort, the target is its public full 113-task DeepSWE score.
Each task feature is the mean pass score for that model-effort on that task.

Selection uses supervised sparse greedy task selection:

- maximize Spearman correlation between the unweighted subset score and the full
  113-task score;
- penalize mean absolute error;
- weakly penalize high public average cost and duration;
- constrain language and repository concentration;
- validate with leave-one-model-family-out re-selection.

For Agentic SWE-Assorted High, a second budgeted selector is used:

- `scripts/analysis/select_deepswe_budgeted_high_subset.py`
- supports all-model caps such as `--max-avg-steps` and
  `--max-avg-input-tokens`
- supports model/effort-specific public caps such as
  `--budget-model glm-5-2 --budget-effort max --max-model-avg-steps 100`
- writes `public_budget_model_avg_steps` and
  `public_budget_model_avg_input_tokens` into each selected task record

This is intentionally stricter than pure correlation selection: if a task is
already known from public DeepSWE rollouts to exceed the intended runtime cap,
it should not be selected and then rediscovered via a paid failure.

L1/PCA-style methods are useful diagnostics, but the shipped score remains an
ordinary unweighted pass@1 over selected tasks. This keeps the leaderboard
auditable: public data is used only to choose tasks, not to weight the metric.

## Results

| subset | tasks | in-sample Spearman | LO-family mean Spearman | LO-family mean MAE |
|---|---:|---:|---:|---:|
| DeepSWE-Essential-8 | 8 | 0.983 | 0.778 | 0.057 |
| DeepSWE-Essential-10 | 10 | 0.987 | 0.697 | 0.051 |
| DeepSWE-Essential-16 | 16 | 0.992 | 0.803 | 0.038 |
| DeepSWE-Essential-20 | 20 | 0.996 | 0.842 | 0.042 |
| DeepSWE-Essential-30 | 30 | 0.996 | 0.925 | 0.037 |
| Assorted GLM52Max-cap High-8 | 8 | 0.942 | 0.773 | 0.080 |

Interpretation:

- `Essential-8` is a minimal frontier-signal tier for mixed-difficulty suites.
  It should not be interpreted as a precise DeepSWE rank estimate.
- `Essential-10` is a cheap screening tier. It is not stable enough to use as a
  final ranking signal.
- `Essential-16` is the current cost/performance compromise.
- `Essential-20` is easier to explain externally and has stronger rank
  correlation, but the expected cost remains high for expensive frontier models.
- `Essential-30` is the first tier that looks robust enough for stronger
  rank-order claims, but it is not budget-friendly.

## Generated Files

- `data/taiwan/deepswe/subsets/essential_10_task_names.json`
- `data/taiwan/deepswe/subsets/essential_8_task_names.json`
- `data/taiwan/deepswe/subsets/essential_16_task_names.json`
- `data/taiwan/deepswe/subsets/essential_20_task_names.json`
- `data/taiwan/deepswe/subsets/essential_30_task_names.json`
- matching metadata JSONL files under the same directory

The selection can be regenerated with:

```bash
python3 scripts/analysis/select_deepswe_essential_subset.py
```

The script writes detailed analysis CSV/JSON under
`outputs/deepswe_subset_analysis/`.

The current Agentic SWE-Assorted High-8 can be regenerated with:

```bash
python3 scripts/analysis/select_deepswe_budgeted_high_subset.py \
  --subset-name essential_anchored_high_8_glm52max_cap100_10m_lang_balanced \
  --no-default-must-include \
  --max-avg-cost-usd 5.0 \
  --max-avg-steps 75 \
  --max-avg-input-tokens 7000000 \
  --budget-model glm-5-2 \
  --budget-effort max \
  --max-model-avg-steps 100 \
  --max-model-avg-input-tokens 10000000 \
  --max-candidates-per-language 5
```
