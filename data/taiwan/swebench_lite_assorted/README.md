# SWE-bench Lite Low/Middle Slices

This directory contains the Low and Middle tiers for Agentic SWE-Assorted.

- Source benchmark: SWE-bench Lite
- Source dataset: `princeton-nlp/SWE-bench_Lite`
- Source split: `test`
- Selection script: `scripts/analysis/prepare_swebench_lite_assorted.py`
- Selection signal: static task features only. Model outcomes are not used.

The static difficulty score uses gold patch size, gold patch file/hunk count,
test patch size, issue length, and FAIL_TO_PASS/PASS_TO_PASS test counts. The
goal is not to predict exact model accuracy, but to create a low-cost tiered
assortment with auditable provenance.

Subsets:

- `subsets/low_36.jsonl`: lower difficulty band.
- `subsets/middle_36.jsonl`: middle difficulty band.
- `subsets/low_middle_72.jsonl`: concatenation used by Agentic SWE-Assorted.

Each row records both tier and provenance:

- `agentic_swe_tier`: `low` or `middle`
- `source_benchmark`: `SWE-bench Lite`
- `source_dataset`: `princeton-nlp/SWE-bench_Lite`
- `source_config`: `default`
- `source_split`: `test`
- `source_subset`: `low_36` or `middle_36`
- `source_instance_id`: original SWE-bench Lite `instance_id`

Run the Assorted wrapper in dry-run mode:

```bash
python3 scripts/tools/run_agentic_swe_assorted.py \
  --model dummy/local \
  --output-dir outputs/agentic_swe_assorted_dryrun \
  --low-middle-limit 2 \
  --high-limit 1 \
  --dry-run
```

