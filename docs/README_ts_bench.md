# TS-Bench Evaluation

TS-Bench is the official Taiwan Safety Benchmark from
`mtkresearch/TS-Bench`. It contains 400 Traditional Chinese prompts:
200 unsafe prompts and 200 hard negatives.

## Data

Build and upload the W&B artifact:

```bash
python3 scripts/data_uploader/prepare_ts_bench.py \
  --output-dir data/taiwan \
  --upload \
  --entity llm-leaderboard \
  --project tc-leaderboard
```

The production artifact is:

```text
llm-leaderboard/tc-leaderboard/ts-bench:production
```

It contains `ts_bench/ts_bench.jsonl`, the original `TSB400.csv`, the
upstream Apache-2.0 `LICENSE`, and a manifest with the source commit.

## Scoring

The evaluator asks the model to classify each prompt and output
`<score>yes</score>` for unsafe or `<score>no</score>` for safe. The leaderboard
score is F1 on the unsafe class, matching the TS-Bench primary reporting style.
The output table also logs precision, recall, accuracy, AUC, confusion-matrix
counts, and unparseable responses.

Enable in a model config:

```yaml
run:
  ts_bench: true

ts_bench:
  artifacts_path: 'llm-leaderboard/tc-leaderboard/ts-bench:production'
  dataset_dir: 'ts_bench/ts_bench.jsonl'
  generator_config:
    max_tokens: 64
    temperature: 0.0
    top_p: 1.0
```
