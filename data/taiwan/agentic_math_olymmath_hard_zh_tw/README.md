# Agentic Math OlymMATH-HARD zh-TW for Nejumi Taiwan

This artifact contains the OlymMATH-HARD Chinese subset from
`RUC-AIBOX/OlymMATH`, converted from Simplified Chinese to
Traditional Chinese with OpenCC (`s2twp`/`s2tw` fallback).

AIME 2025 remains useful as an OpenClaw smoke test, but this
OlymMATH-HARD artifact is the stronger leaderboard candidate
because it has harder symbolic final-answer math problems.

## Files

- `test.jsonl`: full 100-problem OlymMATH-HARD zh-TW set
- `subsets/smoke.jsonl`: stratified harness check subset
- `subsets/leaderboard.jsonl`: full leaderboard subset
- `subsets/leaderboard_40.jsonl`: balanced lower-cost pilot subset
- `subsets/full.jsonl`: alias for the full set
- `manifest.json`: source, license, counts, and hashes

## Scoring

The OpenClaw runner extracts an `ANSWER:` line and scores it
with the official OlymMATH evaluator's Math-Verify direction:
`math_verify.parse` / `math_verify.verify` first, followed by
exact text normalization, SymPy, and conservative string
comparison fallbacks for API/OpenClaw output format differences.

Rows: 100
Smoke rows: 4
Leaderboard-40 rows: 40
