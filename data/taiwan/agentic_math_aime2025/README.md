# Agentic Math AIME 2025 for Nejumi Taiwan

This artifact contains AIME 2025 I and II from `opencompass/AIME2025`
for agentic mathematical reasoning evaluation.

## Files

- `test.jsonl`: full 30-problem set
- `subsets/smoke.jsonl`: small harness check subset
- `subsets/leaderboard.jsonl`: full leaderboard subset
- `manifest.json`: source, license, counts, and hashes

## Scoring

Each answer is scored by exact integer equality after extracting
`ANSWER: <integer>` from the OpenClaw response.

Rows: 30
Smoke rows: 3
