# IFEval zh-TW

This benchmark is a Traditional Chinese IFEval-style adaptation for Nejumi
Taiwan. It keeps mechanically verifiable instruction families and excludes
Japanese-specific M-IFEval constraints such as furigana, hiragana, katakana,
and nominal endings.

Build and upload the artifact:

```bash
python3 scripts/data_uploader/prepare_ifeval_zh_tw.py \
  --output-dir data/taiwan \
  --upload \
  --entity llm-leaderboard \
  --project tc-leaderboard
```

The artifact name is `ifeval-zh-tw`, and the evaluation file is
`ifeval_zh_tw/ifeval_zh_tw.jsonl`.
