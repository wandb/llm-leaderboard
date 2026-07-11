# SWE-bench Lite Low/Middle Slices

This directory contains the Low and Middle tiers for Agentic SWE-Assorted.

- Source benchmark: SWE-bench Lite
- Source dataset: `princeton-nlp/SWE-bench_Lite`
- Source split: `test`
- Selection script: `scripts/analysis/prepare_swebench_lite_assorted.py`
- Default selection: `low_v2_36` + `middle_v2_36`

Two selection families are retained:

- Legacy static slices use gold patch size, gold patch file/hunk count, test
  patch size, issue length, and FAIL_TO_PASS/PASS_TO_PASS test counts only.
- V2 slices add public SWE-bench Lite per-instance resolution rates from
  `SWE-bench/experiments` as a difficulty prior, plus local GLM-5.2 pilot
  signals to remove cost-risk outliers. Public outcomes are used for subset
  calibration only, not for scoring any model.

This means V2 `Low` is intended to mean practical low-cost/easier Agentic SWE,
not merely small gold patch size.

Subsets:

- `subsets/low_36.jsonl`: lower difficulty band.
- `subsets/middle_36.jsonl`: middle difficulty band.
- `subsets/low_middle_72.jsonl`: legacy static Low+Middle concatenation.
- `subsets/low_v2_36.jsonl`: empirically calibrated low tier.
- `subsets/middle_v2_36.jsonl`: empirically calibrated middle tier.
- `subsets/low_middle_v2_72.jsonl`: default Low+Middle input for Agentic SWE-Assorted.

Each row records both tier and provenance:

- `agentic_swe_tier`: `low` or `middle`
- `agentic_swe_selection_version`: selection rule version
- `agentic_swe_selection_basis`: concise rule used for that row
- `source_benchmark`: `SWE-bench Lite`
- `source_dataset`: `princeton-nlp/SWE-bench_Lite`
- `source_config`: `default`
- `source_split`: `test`
- `source_subset`: `low_v2_36`, `middle_v2_36`, `low_36`, or `middle_36`
- `source_instance_id`: original SWE-bench Lite `instance_id`
- `public_lite_seen_count`, `public_lite_resolved_count`, `public_lite_resolve_rate`
- `pilot_glm52_lm12m12_*`: populated when a row was covered by the local GLM-5.2 pilot

Run the Assorted wrapper in dry-run mode:

```bash
python3 scripts/tools/run_agentic_swe_assorted.py \
  --model dummy/local \
  --output-dir outputs/agentic_swe_assorted_dryrun \
  --low-middle-limit 2 \
  --high-limit 1 \
  --dry-run
```
