# Agentic SWE-Assorted 20/20/10

Status: **frozen default as of 2026-07-24**

Agentic SWE-Assorted contains 50 repository-editing tasks:

| Tier | Source | Tasks | Role |
|---|---|---:|---|
| Low | SWE-bench Lite | 20 | Dynamic range for smaller and mid-tier models |
| Middle | SWE-bench Lite | 20 | Mainstream coding-agent discrimination |
| High | DeepSWE v1.1 | 10 | Frontier-model and long-horizon discrimination |

The task counts control cost. Each tier contributes exactly one third of the
benchmark score, so the larger Low/Middle pools do not drown out High.

## Official score

Each task receives one score:

- resolved: `1`
- unresolved, applied, scoreable patch:
  `0.3 * F2P_pass_fraction^2 * P2P_pass_fraction`
- no applied patch or no valid test evidence: `0`

The benchmark score is:

`(mean(Low task scores) + mean(Middle task scores) + mean(High task scores)) / 3`

The quadratic F2P term prevents a large number of easy passing tests from
inflating an incomplete solution. P2P is a linear regression penalty. Partial
credit is capped at 0.3, preserving the distinction between a useful partial
patch and a complete repair. Binary resolution is retained only as audit data.

## Low and Middle selection

Low-20 and Middle-20 are nested subsets of the reviewed v2 36-task pools.
Each source tier was sorted by public SWE-bench Lite resolution rate, divided
into five difficulty strata, and sampled at four tasks per stratum.

The selection objective preserved:

- public resolution-rate mean and spread;
- static task difficulty;
- observed binary and partial-score profiles for GPT-4.1 mini, GPT-5.6 Luna
  high, and Claude Sonnet 4.6 high;
- observed cost and wall-time profile.

Repository concentration was capped at four tasks. Against each original
36-task tier, the selected 20-task score differs by at most 2.2 percentage
points across the three reference runs.

| Tier | Public mean, source | Public mean, selected | Repositories | Max per repo |
|---|---:|---:|---:|---:|
| Low | 0.8490 | 0.8482 | 9 | 4 |
| Middle | 0.4756 | 0.4755 | 8 | 4 |

## High selection

High-10 preserves all eight tasks from the historical frozen High-8 and adds:

- `participle-grammar-conflict-analysis` (Go)
- `bandit-incremental-cache-control` (Python)

The final language distribution is Go 4, Python 3, TypeScript 3.

| Validation | Pearson | Spearman | Kendall |
|---|---:|---:|---:|
| Fixed High-10 vs all 113 DeepSWE tasks | 0.947 | 0.932 | 0.801 |
| Selection-aware leave-one-model-family-out | 0.921 | 0.897 | 0.753 |

Both additions fit the High p90 limits of 150 turns and 13M cumulative input
tokens in the public reference trajectories.

## Artifacts

- Low/Middle:
  `data/taiwan/swebench_lite_assorted/subsets/low_middle_v3_40.jsonl`
- High:
  `data/taiwan/deepswe/subsets/essential_anchored_high_10_model_fidelity_cost_balanced.jsonl`
- Selection audit:
  `outputs/agentic_swe_subset_analysis/assorted_20_20_10_selection_20260724.json`
- Materializer:
  `scripts/analysis/materialize_agentic_swe_assorted_20_20_10.py`

Old 36/36/8 artifacts remain immutable historical versions.
