# BFCL Upgrade Status

Last checked: 2026-06-23

## Current Repository State

The current Nejumi harness integration is still BFCL v3 based.

- Dataset artifact: `llm-leaderboard/nejumi-leaderboard4/bfcl:production`
- Dataset files consumed by the evaluator: `BFCL_v3_*.json`
- Score files emitted by the evaluator: `BFCL_v3_*_score.json`
- Taiwan taxonomy display name: `BFCL v3`
- Taiwan BFCL v3 target artifact: `llm-leaderboard/tc-leaderboard/bfcl-zh-tw:production`

The Japanese BFCL v3 production artifact has alias `jp` and contains Japanese
prompt text. The Taiwan leaderboard should use a separate Traditional Chinese
artifact built from that source:

```bash
python scripts/data_uploader/taiwan_artifacts.py translate-bfcl-v3-ja \
  --source-artifact llm-leaderboard/nejumi-leaderboard4/bfcl:production \
  --output-dir data/taiwan \
  --upload \
  --entity llm-leaderboard \
  --project tc-leaderboard
```

This means BFCL has not yet been upgraded to BFCL v4 in this repository.

## Taiwan BFCL v3 Artifact

Last built: 2026-06-23

- Artifact: `llm-leaderboard/tc-leaderboard/bfcl-zh-tw:production`
- Version checked after upload: `v2`
- Aliases checked after upload: `latest`, `production`, `tc`, `zh-tw`
- Source artifact: `llm-leaderboard/nejumi-leaderboard4/bfcl:production` (`v19`, aliases `production`, `jp`)
- Translation target: prompt `question` messages and Japanese strings in `possible_answer`
- Preserved: function names, tool schemas, code-related fields, and English tool documentation
- Audit result: `BFCL_v3_*.json`, `possible_answer/BFCL_v3_*.json`, and `multi_turn_func_doc/*.json` contain `0` characters in the Japanese kana block `U+3040-U+30FF`

## What BFCL v4 Requires

The upstream Berkeley Function Calling Leaderboard now documents v4 categories around agentic web search, memory management, and format sensitivity. Moving this repository to v4 is not a model-string-only update. It requires:

- Fetching or building v4 datasets and deciding which subsets fit the Taiwan leaderboard.
- Translating or adapting prompts and tool descriptions when needed.
- Adding required external service configuration, such as web search credentials for web-search categories.
- Updating the vendored `bfcl_pkg` or replacing it with the maintained `bfcl-eval` package path.
- Creating new W&B artifacts and updating `configs/base_config_taiwan.yaml`.
- Revalidating leaderboard columns and taxonomy scoring.

## Recommendation

Keep the initial Taiwan leaderboard on the current BFCL v3 artifact until the v4 artifact and evaluator path are verified. Track the v4 migration as a separate task so the GLP taxonomy does not imply v4 coverage before the data and scorer actually support it.
