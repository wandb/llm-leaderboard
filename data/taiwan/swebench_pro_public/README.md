# SWE-bench Pro Public for Nejumi Taiwan

This artifact contains the public `ScaleAI/SWE-bench_Pro` test split
materialized for Agentic SWE evaluation.

## Files

- `test.jsonl` / `test.csv`: full public split
- `subsets/smoke.*`: small harness check subset
- `subsets/leaderboard_80.*`: deterministic stratified leaderboard subset
- `subsets/leaderboard_compact_80.*`: deterministic cost-capped stratified subset
- `subsets/leaderboard_compact_40.*`: deterministic cost-capped stratified subset
- `subsets/full_public.*`: full 731-instance public split
- `manifest.json`: counts, hashes, source, and sampling metadata
- `repo_metadata.jsonl`: optional model-independent repo tree size metadata when compact subsets are built

JSONL and Hugging Face dataset rows also include:

- `agentic_prompt` / `text`: code-free task prompt for real checkout exploration

## Evaluation

Use `scripts/tools/run_swebench_pro_openclaw.py` to generate patches
and `scripts/tools/evaluate_swebench_pro_patches.py` to run the Scale
official evaluator with Docker or Modal.

Offline/on-prem support should mirror target repositories as separate
checkout or bare-repo artifacts instead of embedding source in prompts.

Compact subsets exclude high static-cost instances before stratified
sampling. They are separate benchmark variants and must not be reported
as the full SWE-bench Pro public split.

Taiwan leaderboard default recommendation:

- subset: `leaderboard_compact_80`
- runtime cap: 1,000,000 input tokens per task
- runtime cap: 60 tool calls per task
- runtime cap: 60 agent turns per task
- large-repo filtering is a static risk reducer; runtime caps are the primary cost guard

Rows: 731
