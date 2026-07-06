# Taiwan Benchmark Dataset Report

Last audited: 2026-07-07 JST

This report describes the datasets currently wired or intended for the Nejumi
4.5 Taiwan leaderboard. Sources checked:

- `configs/base_config_taiwan.yaml`
- `taxonomies/nejumi45_taiwan.yaml`
- evaluator implementations under `scripts/evaluator/`
- data builders under `scripts/data_uploader/`
- local materialized data under `data/taiwan/`
- W&B dataset artifacts under `llm-leaderboard/tc-leaderboard` and, for
  inherited ARC-AGI artifacts, `llm-leaderboard/nejumi-leaderboard4`

## High Priority Findings

1. SWE-Bench Pro production artifact mismatch:
   `configs/base_config_taiwan.yaml` uses subset `leaderboard_compact_80`, but
   W&B `llm-leaderboard/tc-leaderboard/swebench-pro-public:production` is
   currently `v2` and does not contain `leaderboard_compact_80`,
   `leaderboard_compact_40`, or `repo_metadata.jsonl`. The compact data exists
   locally under `data/taiwan/swebench_pro_public`, but the W&B production
   artifact must be updated or `swebench_pro.local_dataset_dir` must be set
   before a W&B-artifact-based full canary.
2. ARC-AGI artifact path is not present in `base_config_taiwan.yaml`.
   The evaluator requires `arc_agi.arc_agi_1_artifacts_path` and
   `arc_agi.arc_agi_2_artifacts_path`. These paths exist in `configs/base_config.yaml`
   and in the Taiwan GLP smoke config, but not in the Taiwan base config.
   Current inherited artifacts are in `llm-leaderboard/nejumi-leaderboard4`, not
   `tc-leaderboard`.
3. TWBias is intentionally pending and disabled. Local materialized data exists,
   but the source license is recorded as unknown and W&B
   `llm-leaderboard/tc-leaderboard/twbias:internal` is not present.

## Dataset Inventory

| Unit | Current Data Used | Source | Translation / Localization | Storage Status |
| --- | --- | --- | --- | --- |
| MT-Bench-TW | 80 questions; 41 reference-answer rows | `MediaTek-Research/TCEval-v2`, configs `mt_bench_tw-*` | Native Traditional Chinese from TCEval-v2. No Japanese translation. Judge prompt artifact is reused from `mtbench_ja_prompt`, but its prompt file is English and has no Japanese kana in the checked file. | W&B `mtbench_tw_question:production` v0 and `mtbench_tw_referenceanswer:production` v0. |
| ARC-AGI-2 | 50 public-eval tasks for ARC-AGI-2; ARC-AGI-1 is also evaluated by the current evaluator, but taxonomy uses ARC-AGI-2. | Existing ARC-AGI public-eval artifacts. | No natural-language translation of puzzle data. Taiwan config localizes the prompt template to Traditional Chinese. | W&B inherited artifacts: `nejumi-leaderboard4/arc-agi-1_public-eval_50:production` v2 and `nejumi-leaderboard4/arc-agi-2_public-eval_50:production` v6. Taiwan base config currently lacks their paths. |
| Agentic Math | OlymMATH-HARD zh-TW, 100 tasks; default subset `leaderboard`; optional `leaderboard_40`; smoke 4. | `RUC-AIBOX/OlymMATH`, config `zh-hard`, split `test`, MIT. | Source is Simplified Chinese hard split; builder converts to zh-TW with OpenCC `s2twp` with fallback to `s2tw/s2t`. English hard prompt is retained as audit metadata when available. | W&B `agentic-math-olymmath-hard-zh-tw:production` v0. Local copy under `data/taiwan/agentic_math_olymmath_hard_zh_tw`. |
| Agentic Math smoke only | AIME2025 I/II, 30 tasks. Not the main math leaderboard target. | `opencompass/AIME2025`, MIT. | English/math original; no zh-TW translation. Kept for OpenClaw/Weave smoke. | W&B `agentic-math-aime2025:production` v0. Local copy under `data/taiwan/agentic_math_aime2025`. |
| TMMLU+ | Test 20,118; dev 2,242; train 330. Current Jaster evaluator samples 100 test and 10 dev per run unless changed. | `ikala/tmmluplus`, materialized as `tmmluplus`. | Native Traditional Chinese benchmark. No translation step. | W&B `tmmluplus:production` v0. |
| TMMLU+ Robust | Same base sizes as TMMLU+, plus `IncorrectChoice` and `SymbolChoice` variants for each split. | Derived from `tc-leaderboard/tmmluplus:production` v0. | Original TMMLU+ preserved; robust variants are generated in Traditional Chinese by rewriting instructions/outputs and replacing choice labels for symbol-choice. | W&B `tmmluplus_robust:production` v1. |
| HLE zh-TW | dev 192, test 194. Evaluator skips test in `testmode`; full mode evaluates test and dev. | Japanese HLE artifact `nejumi-leaderboard4/hle-ja:production` v3. | `question` and `answer` translated to zh-Hant-TW with `gpt-5.4-mini-2026-03-17`; metadata records manual repairs and one intentional remaining hiragana case for the Japanese character `ろ`. System and judge prompts are localized in `base_config_taiwan.yaml`. | W&B `hle-zh-tw:production` v3. |
| TCEval-v2 selected | `drcd`: test 3,493/dev 5; `penguin_table`: test 144/dev 5. | `MediaTek-Research/TCEval-v2`. | Native Traditional Chinese. Converted to Jaster-style JSON: DRCd uses `char_f1`; penguin_table uses exact match. | W&B `tceval_v2_selected:production` v0. |
| Agentic SWE | Default intended subset is SWE-Bench Pro `leaderboard_compact_80`, 80 tasks. Local full public has 731 tasks; compact 40 also exists. | `ScaleAI/SWE-bench_Pro`; source code/eval repo `scaleapi/SWE-bench_Pro-os`. | No translation. Rows are code/issue metadata. Agentic prompt is code-free; repository source is inspected through checkout, not embedded into prompt. | Local compact data exists under `data/taiwan/swebench_pro_public`; W&B `swebench-pro-public:production` v2 currently lacks compact subsets and repo metadata. |
| BFCL v3 zh-TW | Artifact contains 17 prompt files / 615 prompt rows and 14 possible-answer files / 537 rows. Current config evaluates 12 categories totaling 429 prompt rows: `java`, `javascript`, `live_irrelevance`, `live_multiple`, `live_relevance`, `live_simple`, `multi_turn_base`, `multi_turn_miss_func`, `multi_turn_miss_param`, `simple`, `multiple`, `irrelevance`. | Japanese Nejumi4 BFCL artifact `nejumi-leaderboard4/bfcl:production` v19. | Prompt messages and possible answers translated to zh-Hant-TW with `gpt-5.4-mini-2026-03-17`. Metadata audit reports 0 Japanese kana chars after translation across prompt, possible_answer, and multi_turn_func_doc. | W&B `bfcl-zh-tw:production` v2 with aliases `production`, `zh-tw`, `tc`; local translation cache has 1,170 rows. |
| IFEval zh-TW | 24 handcrafted prompts. | Newly written Nejumi zh-TW adaptation inspired by IFEval/M-IFEval instruction families. | Native Traditional Chinese prompts; excludes Japanese-specific M-IFEval constraints such as hiragana/katakana/furigana. | W&B `ifeval-zh-tw:production` v0; local copy under `data/taiwan/ifeval_zh_tw`. |
| TS-Bench | 400 rows: 200 unsafe and 200 safe. | `mtkresearch/TS-Bench`, commit `b53fd7fabec6e053677cdfa68cdff4d7d1fba1b4`, Apache-2.0. | Native Traditional Chinese Taiwan safety messages. No translation. | W&B `ts-bench:production` v0; local copy under `data/taiwan/ts_bench`. |
| TWBias | Local gender/ethnicity bias data and prompts. Pending; not part of current Total Score. | `hsinmosyi/TWBias`, commit `d32e9ff0ee029a47898deac8ddbf80b0333e4341`. | Native Traditional Chinese dataset. No translation. | Local copy under `data/taiwan/twbias`; license recorded as unknown; W&B artifact not found; `run.twbias=false`; taxonomy marks pending and not required. |
| HalluLens zh-TW | dev 110 and test 110. | Japanese Nejumi4 HalluLens artifact or local source under `data/source_artifacts/hallulens_ja`. | Venue place/name/type translated to zh-TW; prompt template is `可以詳細介紹位於{place}的「{name}」這間{type_}嗎？`. Original Japanese fields are preserved as `*_original` audit fields. Translation cache is `data/taiwan/.cache/hallulens_zh_tw.jsonl`; metadata says translation model `gpt-5.5`. | W&B `hallulens-zh-tw:production` v0; local copy under `data/taiwan/hallulens_zh_tw`. |
| Traditional Chinese Script Adherence | No independent dataset. It reads `mtbench_output_table.answer`. | Derived from model outputs on MT-Bench-TW. | Measures Simplified-Chinese intrusion in free-form Traditional Chinese answers using OpenCC when available, otherwise a curated simplified-character set. Score is `1 - simplified_intrusion_rate`. | No W&B dataset artifact. Logs `traditional_chinese_script_adherence_*` tables to the same run. |

## Notes On Local Vs W&B Storage

- `data/taiwan/**` currently exists locally but is not tracked by git in this
  working tree. The intended canonical storage for release datasets is W&B
  dataset artifacts.
- For most units, W&B production artifacts are present and match the configured
  artifact names.
- The two concrete exceptions found in this audit are SWE-Bench Pro Compact
  and TWBias. SWE-Bench Pro needs a refreshed W&B artifact if the default
  `leaderboard_compact_80` path is used; TWBias remains pending because of
  license and artifact publication status.

## Current Run-Time Mapping

- `aggregate_taiwan` uses `taxonomies/nejumi45_taiwan.yaml`.
- `tmmluplus` and `tmmluplus_robust` are both produced by `scripts/evaluator/jaster.py`.
- `script_adherence` is a dependent metric and requires MT-Bench output to have
  been logged first.
- `arc_agi` evaluator logs both ARC-AGI-1 and ARC-AGI-2; the Taiwan taxonomy
  consumes only ARC-AGI-2.
- `twbias` is disabled in `base_config_taiwan.yaml` and marked pending in the
  taxonomy, so it is not required for the current Total Score.
