# SWE-bench Pro Agentic Evaluation

SWE-bench Pro is the Agentic SWE candidate for Nejumi 4.5 Taiwan. It is kept
separate from the existing `swebench` evaluator, which is the legacy one-shot
unified-diff path for SWE-bench Verified.

## Data

Build the public split artifact locally:

```bash
python3 scripts/data_uploader/prepare_swebench_pro.py \
  --output-dir data/taiwan \
  --leaderboard-size 80 \
  --smoke-size 10
```

Build cost-capped compact subsets for lower-cost agentic evaluation:

```bash
python3 scripts/data_uploader/prepare_swebench_pro.py \
  --output-dir data/taiwan \
  --leaderboard-size 80 \
  --smoke-size 10 \
  --collect-repo-metadata \
  --compact-leaderboard-size 80 \
  --compact-pilot-size 40
```

The compact path clones/fetches bare Git mirrors under
`data/taiwan/swebench_pro_repo_mirrors`, computes model-independent static repo
tree metadata for each `instance_id` at `base_commit`, and writes
`repo_metadata.jsonl` into the artifact. To reuse already collected metadata:

```bash
python3 scripts/data_uploader/prepare_swebench_pro.py \
  --output-dir data/taiwan \
  --leaderboard-size 80 \
  --smoke-size 10 \
  --repo-metadata-jsonl data/taiwan/swebench_pro_public/repo_metadata.jsonl \
  --compact-leaderboard-size 80 \
  --compact-pilot-size 40
```

Upload to W&B after inspecting the output:

```bash
python3 scripts/data_uploader/prepare_swebench_pro.py \
  --output-dir data/taiwan \
  --leaderboard-size 80 \
  --smoke-size 10 \
  --upload \
  --entity llm-leaderboard \
  --project tc-leaderboard
```

The artifact name is `swebench-pro-public`, with directory
`swebench_pro_public`. The standard subsets are:

```text
subsets/smoke.jsonl             10 tasks for harness checks
subsets/leaderboard_80.jsonl    80 deterministic stratified tasks
subsets/leaderboard_compact_80.jsonl
                                80 deterministic cost-capped stratified tasks
subsets/leaderboard_compact_40.jsonl
                                40 lower-cost pilot tasks
subsets/full_public.jsonl       731 public tasks
```

`leaderboard_compact_*` is a named benchmark variant. It must be reported as
SWE-Bench Pro Compact / cost-capped, not as the full public split. It excludes
the highest static-cost instances before sampling, then applies the existing
language/specificity/category stratification plus a per-repository cap. The cost
proxy is based on repository tree metadata, not on post-hoc model token usage,
so the sample is not tuned to a specific model's tool behavior.

For the agentic OpenClaw path, JSONL rows include a code-free
`agentic_prompt` field, also mirrored as `text` for generic readers. Repository
source is not embedded in this prompt; the agent must inspect the checkout.

On-prem/offline support should be added as a separate repository snapshot
artifact, for example pre-cloned checkouts or bare mirrors keyed by
`instance_id` and `base_commit`. It should not embed source code into the prompt,
because that would turn this into a one-shot patch benchmark rather than an
agentic SWE benchmark.

## Patch Generation

The OpenClaw runner checks out each task repository at `base_commit`, gives the
agent only the issue and task metadata, lets the agent inspect/edit the real
checkout, and captures `git diff --binary`.

```bash
python3 scripts/tools/run_swebench_pro_openclaw.py \
  --dataset-jsonl data/taiwan/swebench_pro_public/subsets/smoke.jsonl \
  --output-dir outputs/swebench_pro_openclaw/smoke \
  --checkout-root outputs/swebench_pro_checkouts \
  --model MODEL_ID \
  --thinking medium
```

By default the runner generates a temporary OpenClaw config per task and adds a
task-scoped agent whose `workspace` is the target checkout. This avoids the
default OpenClaw `main` workspace (`~/.openclaw/workspace`) swallowing edits and
producing empty patches. Use `--no-use-task-agent` only when the selected agent
is already configured with the checkout as its workspace.

Production patch generation relies on the native `weave-openclaw` integration
for W&B Weave Agents conversations. The production agent name is
`nejumi-taiwan-openclaw`. The local OpenClaw sidecar is still written under an
attempt ID for scoring, tool-policy checks, cache recovery, and offline audit.
The diagnostic `--weave-sidecar` path can be used for manual re-logging, but it
is not the production conversation source; diagnostic re-logs use
`nejumi-taiwan-sidecar-diagnostic`.

```text
<instance_id>/
  patch.diff
  openclaw_invocations/<attempt_id>.json
  openclaw_attempts/<attempt_id>/agentic_swe/<instance_id>/openclaw_result.json
```

The generated `patches.json` includes the OpenClaw result path, runner version,
prompt hash, usage, and tool-call counts so the official Docker evaluation can
be audited back to the W&B Weave Agents trace.

Web search, browser tools, HTTP fetches, and external internet lookups are not
allowed during patch generation. The task-specific OpenClaw config denies
search/browser tools. Remote provider-backed code interpreter tools are also
denied; code execution must happen through local shell execution in the target
checkout. Local repository installs such as `pip install -e .` are allowed when
needed to run tests, but package installs from PyPI, git URLs, HTTP(S) URLs, or
other external sources are denied. The protocol sidecar independently scans tool
calls and tool arguments for policy violations before patch collection. A
violation is logged to Weave and fails the patch-generation step.

### NeMoClaw Backend

For Taiwan production, NeMoClaw is the preferred backend when the local
NeMoClaw/OpenShell sandbox is installed and the sandbox can see the checkout
tree. SWE-Bench Pro uses the mounted-checkout design: the host prepares and
scores the checkout, while OpenClaw runs inside NeMoClaw against the
sandbox-visible path for that same tree.

```bash
python3 scripts/tools/run_swebench_pro_openclaw.py \
  --dataset-jsonl data/taiwan/swebench_pro_public/subsets/leaderboard_compact_80.jsonl \
  --output-dir outputs/taiwan_full_eval/swebench_pro/MODEL/openclaw \
  --checkout-root outputs/taiwan_full_eval/swebench_pro_checkouts/MODEL \
  --model openai-direct/gpt-4.1-mini-2025-04-14 \
  --thinking off \
  --nemoclaw-sandbox nejumi-taiwan \
  --nemoclaw-checkout-sandbox-root /sandbox/checkouts \
  --max-input-tokens 1000000 \
  --max-tool-calls 40 \
  --max-agent-turns 40
```

If the sandbox mounts host paths unchanged, omit
`--nemoclaw-checkout-sandbox-root`. If it mounts the checkout root at a different
path, that flag must point to the sandbox-visible parent of the per-instance
checkout directories. The runner writes `.nejumi_openclaw/openclaw_config.json`
inside each checkout and passes the sandbox-visible config path through
`OPENCLAW_CONFIG_PATH`; `.nejumi_openclaw` is excluded from captured patches.

If the host checkout tree is not mounted into the sandbox, use copy mode:

```bash
uv run python scripts/tools/run_swebench_pro_openclaw.py \
  --dataset-jsonl data/taiwan/swebench_pro_public/subsets/leaderboard_compact_80.jsonl \
  --output-dir outputs/taiwan_full_eval/swebench_pro/MODEL/openclaw \
  --checkout-root outputs/taiwan_full_eval/swebench_pro_checkouts/MODEL \
  --model openai-direct/gpt-4.1-mini-2025-04-14 \
  --thinking off \
  --nemoclaw-sandbox nejumi-taiwan \
  --nemoclaw-checkout-transfer-mode copy \
  --max-input-tokens 1000000 \
  --max-tool-calls 40 \
  --max-agent-turns 40
```

Copy mode uploads the prepared git checkout, including `.git`, into the
sandbox and captures `git diff --binary` from the sandbox-side checkout. This
keeps the agentic path repo-based instead of prompt-embedding repository source.

The code paths are implemented and unit-tested. On this machine `nemoclaw`,
`openshell`, the `nejumi-taiwan` sandbox, and native `weave-openclaw` config are
present. A local no-inference probe has verified copy-mode checkout transfer and
sandbox-side diff capture on a minimal git repo. A completed production
SWE-Bench Pro result still has not been produced: the remaining runtime proof is
a real benchmark task run with W&B scalar/table/artifact logging, valid native
Weave Agents trace verification, and official SWE-Bench Pro evaluator results.

Use `--dry-run` only to generate prompts without cloning repos or running
OpenClaw. Dry-run output is not a leaderboard result.

## Official Evaluation

Clone Scale's official evaluator checkout:

```bash
scripts/setup/install_swebench_pro_eval.sh --all
scripts/setup/install_swebench_pro_eval.sh --check-only
```

Run local Docker evaluation:

```bash
python3 scripts/tools/evaluate_swebench_pro_patches.py \
  --official-repo external/SWE-bench_Pro-os \
  --raw-sample-path data/taiwan/swebench_pro_public/subsets/smoke.csv \
  --patch-path outputs/swebench_pro_openclaw/smoke/patches.json \
  --output-dir outputs/swebench_pro_eval/smoke \
  --use-local-docker \
  --num-workers 8
```

The wrapper writes `summary.json` and can log the production W&B payload with
`--wandb`: `agentic_swe_leaderboard_table`, `agentic_swe_output_table`,
`agentic_swe/*` scalar metrics, and an `evaluation-results` artifact containing
the official eval files and patch JSON.

If official evaluation completed locally but the W&B run did not receive the
tables/artifact, first generate and review a dry-run plan without touching W&B:

```bash
uv run python scripts/tools/log_agentic_swe_results_to_wandb.py \
  --official-eval-dir outputs/taiwan_full_eval/swebench_pro/MODEL/official_eval \
  --patch-path outputs/taiwan_full_eval/swebench_pro/MODEL/openclaw/patches.json \
  --model-name MODEL_ID \
  --expected-total 80 \
  --dry-run \
  --plan-json temp/wandb_relog_plans/agentic-swe-MODEL.plan.json
```

After reviewing that plan, relog without rerunning OpenClaw or Docker
evaluation. The writer refuses to call `wandb.login()` unless the validated
plan still matches the current inputs, including source SHA-256 values for
`summary.json`, `eval_results.json` when present, and `patches.json`. For
ok=true plans, the same hashes must appear in `config.relog.source_sha256` and
the post-log verifier command must assert them with
`--expected-run-config relog.source_sha256.*`:

```bash
uv run python scripts/tools/log_agentic_swe_results_to_wandb.py \
  --official-eval-dir outputs/taiwan_full_eval/swebench_pro/MODEL/official_eval \
  --patch-path outputs/taiwan_full_eval/swebench_pro/MODEL/openclaw/patches.json \
  --model-name MODEL_ID \
  --expected-total 80 \
  --validated-dry-run-plan-json temp/wandb_relog_plans/agentic-swe-MODEL.plan.json
```

For leaderboard use, verify completion in W&B:

```bash
uv run python scripts/tools/verify_taiwan_wandb_completion.py \
  --run-id RUN_ID \
  --benchmark agentic_swe \
  --expected-total 80 \
  --env-file .env \
  --json outputs/taiwan_full_eval/wandb_completion/agentic_swe-RUN_ID.json
```

The verifier checks that the W&B run is finished, output rows match the total
instance count, `pass_at_1` equals `resolved/total`, and a production result
artifact is logged.

Agentic SWE also requires an inspectable W&B Weave Agents trace:

```bash
uv run python scripts/tools/verify_taiwan_weave_agents.py \
  --agent-name nejumi-taiwan-openclaw \
  --require-tool-span \
  --require-tool-content
```

This is a live Agents API check. It must pass for production release: the
trace needs visible message content, tool arguments/results, and chronological
span structure where tool execution does not appear before the first message.
When checking a specific canary or SWE task, pin the conversation with
`--conversation-id` or `--conversation-id-contains` and add `--require-text`
for task-specific markers so an unrelated trace cannot satisfy the gate.

## `run_eval.py`

The Taiwan base config contains `run.swebench_pro: false` by default. Enable it
in a model config only when the OpenClaw and Docker/Modal environment is ready.

```yaml
run:
  swebench_pro: true

swebench_pro:
  subset: smoke
  thinking: medium
  openclaw_model: MODEL_ID
  agent: main
  use_task_agent: true
  task_agent_prefix: nejumi-swe
  run_openclaw: true
  evaluate: true
```

For a lower-cost production canary, use a compact subset after the compact
artifact has been built and uploaded:

```yaml
swebench_pro:
  subset: leaderboard_compact_40
  max_input_tokens: 1000000
  max_tool_calls: 40
  max_agent_turns: 40
```

For the main lower-cost Taiwan leaderboard candidate, use:

```yaml
swebench_pro:
  subset: leaderboard_compact_80
  max_input_tokens: 1000000
  max_tool_calls: 40
  max_agent_turns: 40
  nemoclaw_sandbox: nejumi-taiwan
  nemoclaw_checkout_sandbox_root: /sandbox/checkouts
```

`max_input_tokens`, `max_tool_calls`, and `max_agent_turns` are hard harness
budgets. A task that exceeds them is recorded as `runtime_budget_exceeded` and
receives an empty patch, so the run remains scoreable instead of becoming an
infrastructure failure. `max_tool_calls` and `max_agent_turns` are also enforced
live where the OpenClaw session JSONL is visible: the protocol wrapper monitors
the task agent's session file and terminates the OpenClaw subprocess after the
cap is exceeded. `max_input_tokens`
is checked from OpenClaw's completed usage metadata because OpenClaw 2026.6.9
does not expose live token usage to the CLI. If a future OpenClaw/NeMoClaw hook
supports native live token interruption, wire the same config field to that
hook.

## Notes

- This path is agentic: the model edits a real repository checkout instead of
  producing a one-shot patch from a packed prompt. Do not embed repository
  source code into the normal OpenClaw prompt.
- OpenClaw is the harness and Weave trace surface. It should not use Codex,
  Claude Code, Gemini CLI, or another native agent CLI as the primary runtime
  unless that runtime is explicitly being evaluated as a model/system.
- `weave-openclaw@0.1.1` may resolve incompatible OpenTelemetry dependencies
  under current npm/OpenClaw managed overrides. Run
  `scripts/setup/install_openclaw_weave.sh --all`; the script pins the
  plugin-local `@opentelemetry/core`, `@opentelemetry/resources`, and
  `@opentelemetry/sdk-trace-base` packages to the tested 1.x line, patches the
  Weave GenAI provider for OpenTelemetry 1.x/2.x processor compatibility, and
  checks both local OTLP export and the live W&B Agents API.
- SWE-bench Pro and DeepSWE final weighting is intentionally undecided. This
  implementation only makes the SWE-bench Pro path executable.
