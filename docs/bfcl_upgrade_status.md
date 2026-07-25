# BFCL v4 Taiwan Status

Last checked: 2026-07-24 JST

## Release Decision

The Taiwan leaderboard uses a Traditional-Chinese BFCL v4 adaptation. It
retains the Japanese Nejumi selection policy:

- exclude parallel categories because not every evaluated model supports
  parallel tool calls;
- keep multi-turn cases with at most three user turns;
- deterministically select at most 30 cases per category;
- translate user-facing text to Traditional Chinese while preserving
  executable identifiers, schemas, enums, literals, and answers.

The release profile is `full`: 17 categories and 496 logical scored cases.
This includes the official v4 memory and web-search domains. The diagnostic
`core` profile is retained only for inexpensive provider/handler canaries; it
is not a leaderboard score.

| Profile | Categories | Logical cases | Use |
|---|---:|---:|---|
| `full` | 17 | 496 | Taiwan leaderboard default |
| `core` | 12 | 346 | Provider/handler diagnostic only |

The 496 cases are 16 categories with 30 selected cases plus all 16 eligible
`live_relevance` cases. Full adds `memory_kv`, `memory_vector`,
`memory_rec_sum`, `web_search_base`, and `web_search_no_snippet` to the 12
core categories.

Excluded categories are `parallel`, `parallel_multiple`, `live_parallel`,
`live_parallel_multiple`, `multi_turn_long_context`, and
`format_sensitivity`.

Official BFCL v4 weights are preserved after these exclusions: non-live 10%,
live 10%, irrelevance 10%, multi-turn 30%, and Agentic memory/web 40%.

## Web Search

Web search is backend-selectable in YAML:

```yaml
bfcl:
  web_search:
    backend: ddgs
    ddgs_backend: auto
    endpoint: https://html.duckduckgo.com/html/
    timeout_sec: 20
    max_attempts: 4
    retry_base_sec: 2
    min_request_interval_sec: 2
    request_jitter_sec: 0.25
    cache_path: outputs/bfcl_v4/web_search_cache.sqlite3
    cache_ttl_sec: 0
```

`ddgs` is the default and requires no paid search account. It uses the DDGS
multi-engine client and normalizes results to BFCL's
`title`/`href`/`body` contract, decodes redirect URLs, retries bounded
transient failures, paces requests, and caches successful results in SQLite.
`cache_ttl_sec: 0` means the successful result remains reusable, improving
repeatability and avoiding duplicate external requests.

Set `backend: duckduckgo_html` to use the direct DuckDuckGo HTML fallback.
Set `backend: serpapi` to use the official upstream-style SerpAPI path.
Only that backend requires the `google-search-results` package and
`SERPAPI_API_KEY`. Preflight enforces the distinction before W&B or model
execution.

Search trajectory is not exact-matched. The official checker evaluates the
final assistant answer against accepted answer text. Search results can change
over time, so the selected backend and request/cache counters are logged to
W&B.

## Source And Artifact

- Upstream: `ShishirPatil/gorilla`
- Pinned commit: `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`
- Builder: `scripts/data_uploader/bfcl_v4_zh_tw.py`
- Local data: `data/taiwan/bfcl-v4-zh-tw/bfcl`
- Translation cache:
  `data/taiwan/bfcl_v4_zh_tw_translation_cache.jsonl`
- W&B artifact:
  `llm-leaderboard/tc-leaderboard/bfcl-v4-zh-tw:production`
- Artifact version/digest:
  `v0` / `723977c468ce77ec5c746515eada3564`
- Selection seed: `20260724`
- Translation model: `gpt-5.4-mini-2026-03-17`

The translation audit covers 3,244 unique fields:

- 3,171 of 3,219 source fields containing letters changed;
- 3,167 translated fields contain CJK text;
- zero executable-schema mismatches;
- zero Japanese kana characters;
- 406 physical prompt rows and 496 logical scored cases.

## Runtime And Failure Policy

The evaluator is `scripts/evaluator/bfcl_v4.py`; the pinned official runtime is
vendored under `scripts/evaluator/evaluate_utils/bfcl_v4_pkg/bfcl_eval`.
`scripts/evaluator/bfcl.py` dispatches to v4 only when `bfcl.version: v4`, so
the Japanese v3 path remains intact.

Supported model paths are OpenAI Responses, Anthropic Messages, and
OpenAI-compatible chat completions. Provider settings, credentials,
`extra_body`, timeouts, retries, pacing, and sampling remain YAML-driven.

Requests have independent provider and case deadlines, bounded retries, a
30-second watchdog, and consecutive-failure fail-fast. Case timeouts are
scored incorrect. Provider/inference errors are counted separately and must be
zero for release completion. Runtime files are isolated under each run's
`outputs/bfcl_v4/<run-id>` directory.

Full-profile preflight checks memory packages, web packages, backend validity,
and backend-specific credentials. The vector-memory probe has successfully
loaded `all-MiniLM-L6-v2`, inserted embeddings into FAISS, and retrieved the
expected item.

## Validation Evidence

Provider canaries use one case from each of the 12 diagnostic core categories:

| Provider | W&B run | Result |
|---|---|---|
| OpenAI Direct, `gpt-4.1-mini-2025-04-14` | `iig4no3j` | 12/12 rows, 0 timeout, 0 inference error |
| Anthropic Direct, `claude-haiku-4-5-20251001` | `xiige94o` | 12/12 rows, 0 timeout, 0 inference error |

The OpenAI-compatible OSS path also passes a local HTTP integration test
through the actual SDK, adapter, BFCL handler, tool schema, usage, and
tool-call parser.

The direct web backend passed live non-model probes for English and
Traditional-Chinese queries with valid results and zero retries. Search
backend unit tests cover parsing, redirect decoding, snippets, retry
exhaustion, cache reuse, aliases, and invalid configuration. Full OpenAI
configuration preflight passed without executing W&B, a model, or an
evaluator.
The OpenAI Direct full-profile run `4reks3bj` completed all 496 logical
scored cases (607 runtime rows including memory prerequisites) with zero
timeouts, inference errors, or duplicate IDs. During review, the scorer was
found to align completion-ordered result rows with dataset-ordered prompts by
position. This made the initially reported 10.06% invalid. The scorer now
aligns prompts and ground truth by case ID, rejects duplicate and unknown
result IDs, and preserves the actual result order. A normal scorer-only rerun
produced a corrected Overall score of 33.50%; all checked score-row IDs match
their prompt IDs. The corrected W&B tables and alignment-fix metadata are
synchronized, and the completion verifier passes.

The Anthropic Direct full-profile run `2n6l2n4y` resumed from 140 saved
runtime rows and completed all 496 logical cases (607 runtime rows) without
duplicates, timeouts, or inference errors. Its Overall score is 50.11%, with
an estimated provider cost of $0.88 and a mean request latency of 7.50
seconds. Its W&B output and leaderboard tables pass completion verification.

The final repository regression completed with 1,588 tests passed and zero
failures.

## Commands

Preflight the official-like 496-case OpenAI configuration:

```bash
uv run python scripts/run_eval.py \
  --base-config base_config_taiwan.yaml \
  --config config-bfcl-v4-openai-fullrun.yaml \
  --preflight
```

Run it after preflight and operator review:

```bash
uv run python scripts/run_eval.py \
  --base-config base_config_taiwan.yaml \
  --config config-bfcl-v4-openai-fullrun.yaml \
  --yes
```

Verify a completed full BFCL run:

```bash
uv run python scripts/tools/verify_taiwan_wandb_completion.py \
  --entity llm-leaderboard \
  --project tc-leaderboard \
  --run-id RUN_ID \
  --benchmark bfcl \
  --expected-total 496 \
  --expected-run-config bfcl.version=v4 \
  --expected-run-config bfcl.profile=full
```

## Remaining Scope

The dataset, handlers, search backends, memory runtime, configuration, scorer,
preflight, resume path, W&B tables, and completion verifier are ready for
model-fleet evaluation. Full runs have passed on both OpenAI Direct and
Anthropic Direct.

Before a large fleet launch, reduce third-party DDGS/primp console noise and
record the selected DDGS engine/region more explicitly. The search wrapper
already bounds retries, records counters, and completed both full runs without
search errors, so this is an observability and operability improvement rather
than a correctness blocker. The 12-category canaries prove provider plumbing
only and must not be presented as the v4 leaderboard result.
