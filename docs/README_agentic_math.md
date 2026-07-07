# Agentic Math Evaluation

Agentic Math uses **OlymMATH-HARD zh-TW + OpenClaw** as the main Taiwan
leaderboard math benchmark.

AIME 2025 is still kept in the repository as an OpenClaw smoke test because it
is small and useful for checking Weave/Agents logging, but it is not the main
math leaderboard target.

## Main Data

Build and upload the OlymMATH-HARD zh-TW W&B artifact:

```bash
uv run python scripts/data_uploader/prepare_olymmath_zh_tw.py \
  --output-dir data/taiwan \
  --upload \
  --entity llm-leaderboard \
  --project tc-leaderboard
```

The main artifact name is `agentic-math-olymmath-hard-zh-tw`, with directory
`agentic_math_olymmath_hard_zh_tw`. Standard subsets:

```text
subsets/smoke.jsonl          stratified harness check subset
subsets/leaderboard.jsonl    full 100-problem OlymMATH-HARD zh-TW set
subsets/leaderboard_40.jsonl balanced lower-cost pilot subset
subsets/full.jsonl           alias for the full set
```

The source dataset is `RUC-AIBOX/OlymMATH` (`zh-hard`, MIT). The builder uses
OpenCC to convert Simplified Chinese prompts to Traditional Chinese and keeps
the English hard prompt as audit metadata when available.

## Smoke Data

The AIME 2025 artifact is still available for smoke testing:

```bash
uv run python scripts/data_uploader/prepare_agentic_math.py \
  --output-dir data/taiwan \
  --upload \
  --entity llm-leaderboard \
  --project tc-leaderboard
```

That artifact is `agentic-math-aime2025`, with directory
`agentic_math_aime2025`.

## OpenClaw Run

Run the main OlymMATH-HARD smoke subset:

```bash
uv run python scripts/tools/run_agentic_math_openclaw.py \
  --dataset-jsonl data/taiwan/agentic_math_olymmath_hard_zh_tw/subsets/smoke.jsonl \
  --output-dir outputs/agentic_math_openclaw/olymmath_smoke \
  --model MODEL_ID \
  --thinking high
```

The runner creates a task-scoped OpenClaw agent whose workspace is the task
scratch directory. It extracts an `ANSWER:` line and follows the official
OlymMATH local evaluator's answer-checking direction: `math_verify.parse` /
`math_verify.verify` is the primary symbolic judge, with exact normalization,
SymPy, and conservative string comparison fallbacks for API/OpenClaw output
format differences. The OlymMATH README recommends established MATH-format
evaluation tools and ships local evaluation code based on Math-Verify.

Python is intentionally allowed as an agent tool, following the same broad
direction as public AIMO3-style math evaluation notebooks: code may be used for
exact calculation, symbolic manipulation, small-case brute force, and numerical
verification. The prompt frames Python as support for mathematical reasoning
rather than a replacement for it.

Web search, browser tools, HTTP fetches, and external internet lookups are not
allowed for Agentic Math. The task-specific OpenClaw config denies search/browser
tools. Remote provider-backed code interpreter tools are also denied; Python is
allowed only as local shell execution inside the task workspace. The protocol
sidecar independently scans tool calls and tool arguments for policy violations
before scoring. A violation is logged to Weave and fails the task instead of
producing a leaderboard result.

Python usage must be non-interactive. Agents should run `python3 -c`, a heredoc
such as `python3 - <<'PY' ... PY`, or a short script file. Interactive shells,
Python REPLs, notebooks, background processes, and `exec` calls with `pty=true`
are disallowed because they can hang the harness and distort cost accounting.

The native `weave-openclaw` integration is the authoritative W&B Weave Agents
trace path. It records OpenClaw conversations and tool calls in execution order.
The production agent name is `nejumi-taiwan-openclaw`.
The local `openclaw_result.json` sidecar remains the harness audit artifact for
scoring, cache recovery, tool policy checks, and offline diagnosis. The
diagnostic `--weave-sidecar` path can be used for manual re-logging, but it is
not the production conversation source; diagnostic re-logs use
`nejumi-taiwan-sidecar-diagnostic`.

Each task result is keyed by runner version, prompt hash, task ID, model,
thinking level, and answer format. A stale `result.json` with a mismatched key is
archived under `stale_results/` and rerun, so prompt or policy changes cannot
silently mix with older benchmark outputs. OpenClaw sidecars and invocation logs
are written under attempt-specific directories:

```text
<task_id>/
  result.json
  stale_results/
  openclaw_invocations/<attempt_id>.json
  openclaw_attempts/<attempt_id>/agentic_math/<task_id>/openclaw_result.json
```

### NeMoClaw Backend

On this host, the `nejumi-taiwan` NeMoClaw sandbox is configured with native
`weave-openclaw`. The runner supports task-scoped OpenClaw agents inside the
sandbox by reading `/sandbox/.openclaw/openclaw.json` as the template and
writing the per-task config under the sandbox task workspace.

```bash
uv run python scripts/tools/run_agentic_math_openclaw.py \
  --dataset-jsonl data/taiwan/agentic_math_olymmath_hard_zh_tw/subsets/smoke.jsonl \
  --output-dir outputs/nemoclaw_agentic_math_smoke \
  --model inference/deepseek-v4-flash \
  --thinking high \
  --nemoclaw-sandbox nejumi-taiwan
```

Do not use `--no-use-task-agent` for the production NeMoClaw path unless a
separately reviewed sandbox agent already pins the correct task workspace.

For a no-inference wiring check, add `--dry-run --limit 1 --redo`. The dry-run
still exercises the task-agent/protocol path, writes the per-task OpenClaw
config, invocation JSON, and sidecar result, but does not call the model or log
to W&B.

## `run_eval.py`

Enable in a model config:

```yaml
run:
  agentic_math: true

agentic_math:
  artifacts_path: 'llm-leaderboard/tc-leaderboard/agentic-math-olymmath-hard-zh-tw:production'
  dataset_dir: 'agentic_math_olymmath_hard_zh_tw'
  subset: leaderboard
  openclaw_model: MODEL_ID
  thinking: high
  run_openclaw: true
```

## W&B Completion Gate

For leaderboard use, a local `summary.json` / `results.jsonl` pair is not
complete until the result has been logged to W&B. Verify a completed Agentic
Math run with:

```bash
uv run python scripts/tools/verify_taiwan_wandb_completion.py \
  --run-id RUN_ID \
  --benchmark agentic_math \
  --expected-total 100 \
  --env-file .env \
  --json outputs/taiwan_full_eval/wandb_completion/agentic_math-RUN_ID.json
```

The verifier checks that the run is finished, scalar metrics are present,
`agentic_math_leaderboard_table` has rows, `agentic_math_output_table` has one
row per problem, accuracy equals `correct/total`, and an
`evaluation-results` artifact with the `production` alias is logged.

If OpenClaw completed locally but the W&B run did not receive the production
tables/artifact, first generate and review a dry-run relog plan:

```bash
uv run python scripts/tools/log_agentic_math_results_to_wandb.py \
  --results-dir outputs/taiwan_full_eval/agentic_math/MODEL/openclaw \
  --model-name MODEL_ID \
  --dry-run \
  --plan-json temp/wandb_relog_plans/agentic-math-MODEL.plan.json
```

After reviewing that plan, relog the existing local result. The writer refuses
to call `wandb.login()` unless the validated plan still matches the current
inputs, including the SHA-256 of `summary.json` and `results.jsonl`. The dry-run
plan also stores the same hashes in `config.relog.source_sha256`, and its
post-log verifier command must assert those run config keys with
`--expected-run-config relog.source_sha256.*`:

```bash
uv run python scripts/tools/log_agentic_math_results_to_wandb.py \
  --results-dir outputs/taiwan_full_eval/agentic_math/MODEL/openclaw \
  --model-name MODEL_ID \
  --validated-dry-run-plan-json temp/wandb_relog_plans/agentic-math-MODEL.plan.json
```

Agentic Math also requires an inspectable W&B Weave Agents trace. Verify the
native `weave-openclaw` Agent record separately:

```bash
uv run python scripts/tools/verify_taiwan_weave_agents.py \
  --agent-name nejumi-taiwan-openclaw \
  --require-tool-span \
  --require-tool-content
```

This checks the live Agents API for the production agent, confirms a latest
trace exists, requires visible message content by default, and verifies that
tool spans do not precede the first message span.
For a specific canary or task, also pass `--conversation-id` or
`--conversation-id-contains` plus one or more `--require-text` values so the
verifier proves the intended conversation content is visible, not merely some
recent non-empty trace.
