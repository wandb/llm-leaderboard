# SWE-Bench Pro vs SWE-rebench Correlation

Generated with:

```bash
python3 scripts/analysis/build_swebench_pro_swerebench_correlation.py
```

Sources:

- SWE-rebench live leaderboard state: https://swe-rebench.com/
- SWE-Bench Pro current Hugging Face leaderboard: https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro
- SWE-Bench Pro release-time paper Table 1: https://arxiv.org/html/2509.16941

## Results

## Which Figure To Read First

For the staleness / contamination hypothesis, read these in this order:

1. `docs/figures/swebench_pro_release_vs_swerebench_sep_2025_scatter.png`

   This is the cleanest baseline. It compares SWE-Bench Pro's release-time
   paper Table 1 scores against SWE-rebench's September 2025 window. If the
   question is "did SWE-Bench Pro agree with the then-current live benchmark
   when it came out?", this is the main figure.

2. `docs/figures/swebench_pro_current_vs_swerebench_latest_default_scatter.png`

   This is the strict "current SWE-Bench Pro HF leaderboard vs latest
   SWE-rebench default window" view. The important conclusion is not the fitted
   line; it is that only three exact model labels overlap. Treat this as an
   overlap failure / freshness warning, not as a stable negative-correlation
   estimate.

3. `docs/figures/swebench_pro_current_vs_swerebench_sep_2025_scatter.png`

   This addresses the question: "if model weights are fixed and contamination
   itself does not progress, can we compare the current SWE-Bench Pro HF scores
   against SWE-rebench's September 2025 benchmark window?" The strict label
   overlap is still only four rows: Qwen3-Coder-480B-A35B-Instruct, Kimi K2
   Instruct 0905, GPT-OSS 120B, and GLM-4.6.

4. `docs/figures/swebench_pro_current_vs_swerebench_model_latest_scatter.png`

   This is a diagnostic fallback. It increases overlap by using each model's
   own latest active SWE-rebench window, but it is not a shared-time comparison.
   Use it only to inspect whether the broader rank order still looks plausible.

The post-release September window figure is only a sensitivity check.

```text
Comparison                                      n    Pearson r    Spearman rho    Kendall tau
Current HF Pro vs latest SWE-rebench default    3       -0.855          -1.000         -1.000
Current HF Pro vs Sep 2025 SWE-rebench          4       -0.075          -0.400         -0.333
Current HF Pro vs model-latest SWE-rebench     13        0.809           0.687          0.462
Release Table 1 vs Sep 2025 SWE-rebench         5        0.843           0.900          0.800
Release Table 1 vs post-release Sep window      5        0.671           0.900          0.800
```

The most decision-relevant comparison for the contamination/staleness question
is not the negative 3-point current-window number by itself. That comparison is
underpowered because the latest SWE-rebench default window, 2026-03-01 to
2026-05-15, only overlaps the current SWE-Bench Pro HF leaderboard on three
exact model labels: MiniMax M3, Kimi K2.6, and GLM-5.1.

The release-time comparison is cleaner. SWE-Bench Pro was published in
September 2025, and the paper reports public-set scores for models evaluated as
of 2025-09-18. Matching those to SWE-rebench's 2025-09-01 to 2025-10-01 window
gives five overlapping rows: Claude Sonnet 4.5, Claude Sonnet 4, GPT-5 high,
Kimi K2 Instruct, and GPT-OSS 120B. That view is strongly positive
(`Pearson r=0.843`, `Spearman rho=0.900`).

The model-latest comparison is included as a diagnostic only. It uses each
model's most recent active SWE-rebench task window, so it improves overlap to
13 rows but does not represent one shared time window.

## Artifacts

```text
data/analysis/swebench_pro_swerebench_correlation_summary.json
data/analysis/swebench_pro_swerebench_match_audit.csv
data/analysis/swebench_pro_current_vs_swerebench_latest_default_scores.csv
data/analysis/swebench_pro_current_vs_swerebench_sep_2025_scores.csv
data/analysis/swebench_pro_current_vs_swerebench_model_latest_scores.csv
data/analysis/swebench_pro_release_vs_swerebench_sep_2025_scores.csv
data/analysis/swebench_pro_release_vs_swerebench_post_release_scores.csv
docs/figures/swebench_pro_current_vs_swerebench_latest_default_scatter.png
docs/figures/swebench_pro_current_vs_swerebench_sep_2025_scatter.png
docs/figures/swebench_pro_current_vs_swerebench_model_latest_scatter.png
docs/figures/swebench_pro_release_vs_swerebench_sep_2025_scatter.png
docs/figures/swebench_pro_release_vs_swerebench_post_release_scatter.png
```
