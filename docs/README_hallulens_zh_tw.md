# HalluLens zh-TW

HalluLens zh-TW is the Traditional Chinese version of the Nejumi4 HalluLens
hallucination-resistance benchmark.

## Artifact

Build and upload:

```bash
python3 scripts/data_uploader/prepare_hallulens_zh_tw.py \
  --output-dir data/taiwan \
  --model gpt-5.5 \
  --upload \
  --entity llm-leaderboard \
  --project tc-leaderboard
```

Production artifact:

```text
llm-leaderboard/tc-leaderboard/hallulens-zh-tw:production
```

The artifact contains:

```text
hallulens_zh_tw/dev/generation.jsonl
hallulens_zh_tw/test/generation.jsonl
hallulens_zh_tw/manifest.json
hallulens_zh_tw/README.md
```

Each active row has `place`, `type_`, `name`, and `prompt` in Traditional
Chinese / non-Japanese display form. Original Japanese fields are preserved as
`*_original` for auditability.

## Evaluation

Enable:

```yaml
run:
  hallulens_zh_tw: true
hallulens_zh_tw:
  artifacts_path: 'llm-leaderboard/tc-leaderboard/hallulens-zh-tw:production'
  dataset_dir: 'hallulens_zh_tw'
```

The evaluator logs:

```text
hallulens_zh_tw_output_table
hallulens_zh_tw_output_table_dev
hallulens_zh_tw_leaderboard_table.hallucination_resistance
```

The judge uses structured JSON output with `does_believe: true|false`; the
leaderboard score is the mean rate of `does_believe == false`.
