# Nejumi Agent Protocol 2026.04

This protocol is the fixed agentic evaluation condition for Nejumi 4.5 Taiwan.

## Runtime

- Agent harness: OpenClaw
- Trace integration: `wandb/weave-openclaw`
- Required UI target: W&B Weave Agents view
- Minimum runtime requirements:
  - Node.js >= 22.19, or Node.js 24+
  - OpenClaw >= 2026.4.25
- Agent runtime: OpenClaw native runtime only
- External native agent CLIs such as Codex, Claude Code, and Gemini CLI are disabled as primary runtimes.

## Weave Requirements

Agentic SWE and Agentic Math runs must be visible as agent sessions in the Weave Agents view. The evaluation runner must persist the following metadata on each root agent run:

- `protocol_version`
- `benchmark_id`
- `task_id`
- `model_id`
- `agent_runtime`
- `openclaw_version`
- `weave_openclaw_version`
- `prompt_hash`
- `tool_policy_hash`
- `verifier_hash`
- `pass_fail`
- `cost`
- `tokens`
- `steps`
- `wall_clock_time`
- `trace_url`

If prompt/tool content capture is disabled for privacy, the run still needs structure, token totals, cost totals, and hashes.

## Runner

The repository entrypoint is:

```bash
scripts/setup/install_openclaw_weave.sh --check-only
scripts/setup/install_openclaw_weave.sh --all

python3 scripts/tools/run_openclaw_agent_protocol.py preflight
python3 scripts/tools/run_openclaw_agent_protocol.py check-agents \
  --limit 5 \
  --json temp/openclaw_agents_diagnostic.json
python3 scripts/tools/run_openclaw_agent_protocol.py write-weave-config \
  --output configs/openclaw_weave.example.json
python3 scripts/tools/run_openclaw_agent_protocol.py run \
  --benchmark-id agentic_math \
  --task-id TASK_ID \
  --prompt-file path/to/prompt.md \
  --model MODEL_ID
```

The generated Weave config is an example snippet. Inspect it and merge it into the OpenClaw gateway config used by the evaluation host. By default, native OpenClaw traces use the `nejumi-taiwan-openclaw` Weave Agent name. The runner prepends a Nejumi protocol metadata block to the message, invokes `openclaw agent --json --message`, and writes `openclaw_result.json` beside the generated message file.

`scripts/setup/install_openclaw_weave.sh --all` installs or updates Node through nvm when needed, installs OpenClaw with npm, installs the `weave-openclaw` plugin, pins the plugin-local OpenTelemetry dependencies to the tested 1.x line, patches the Weave GenAI provider for OpenTelemetry 1.x/2.x processor compatibility, merges the Weave plugin block into `~/.openclaw/openclaw.json`, merges Taiwan eval model provider entries, and writes the OpenClaw gateway systemd user `.env` drop-in. It keeps secrets out of the config by referencing environment variables. `--check-only` also verifies `.env` key discovery, local OTLP export with message/tool content markers, local `weave-openclaw` plugin event replay with message/tool content markers, the installed OpenClaw runtime hook/content contract, the gateway `.env` drop-in, and live W&B Agents API reachability.

The local OTLP content checks prove that the Weave GenAI exporter and the
installed `weave-openclaw` plugin handlers can emit message/tool content when
events are replayed locally. `scripts/tools/verify_openclaw_runtime_content_hooks.py`
adds a static no-inference guard that the installed OpenClaw bundle contains
the typed content-bearing hooks used by the plugin: `llm_input`, `llm_output`,
`before_tool_call`, `after_tool_call`, `before_message_write`, and
`model_call_started`. These checks do not replace the production trace gate.
For leaderboard runs, `scripts/tools/verify_taiwan_weave_agents.py` must pass
against the live W&B Agents API so the actual OpenClaw native runtime is known
to expose inspectable conversation and tool content.

If the W&B UI does not show the Agents records immediately, use `check-agents`.
It calls `https://trace.wandb.ai/agents/query` and `https://trace.wandb.ai/agents/spans/query`
with the configured entity/project/agent name, then prints the latest `trace_id`,
`span_id`, `conversation_id`, model, provider, and error fields without exposing
the API key. The diagnostic output also includes `content_capture_health` and
`trace_order_health`, including valid/invalid timestamp counts, visible
user-input span counts, final-answer marker span counts, and whether tool spans
start only after visible input and before the final-answer span end time. This
is an operator diagnostic; release proof still requires
`scripts/tools/verify_taiwan_weave_agents.py` to pass against the live W&B
Agents API. Use `--json` plus `--conversation-id` or
`--conversation-id-contains` during canary or release preparation so this
diagnostic is retained with the rest of the operator evidence and scoped to the
target task. The saved file should use a `.agents.json` suffix and includes
`diagnostic_schema_version=1`, `generated_at`, and W&B Agents API
`query_source` metadata. `run_weave_agents_content_canary.py --execute`
generates this diagnostic automatically after the formal verifier passes.

## Fresh Agents Content Canary

Use the dedicated canary before treating native OpenClaw/Weave tracing as
production-ready:

```bash
python3 scripts/tools/run_weave_agents_content_canary.py \
  --canary-id PREPARE_ONLY \
  --model openai-direct/gpt-4.1-nano-2025-04-14 \
  --thinking off
```

The command above is prepare-only. It writes the prompt and execution plan but
does not call a model. To run one paid live trace intentionally:

```bash
python3 scripts/tools/run_weave_agents_content_canary.py \
  --execute \
  --canary-id OPENAI_DIRECT_NANO_CONTENT_CANARY \
  --model openai-direct/gpt-4.1-nano-2025-04-14 \
  --thinking off \
  --timeout 180
```

The canary asks the agent to use Python once for `7 * 13`, then verifies the
fresh conversation through `scripts/tools/verify_taiwan_weave_agents.py` with
message content, tool content, usage, the canary id, and the exact expected
answer text `CANARY_RESULT <CANARY_ID> 91` required. This prevents a non-empty
but unrelated Agents trace from satisfying the content gate.

Every prepare-only or executed canary also writes a production gate summary:

```bash
python3 scripts/tools/verify_weave_agents_content_canary_result.py \
  --plan-file outputs/weave_agents_content_canary/plans/weave_agents_content_canary_CANARY_ID.json \
  --json outputs/weave_agents_content_canary/plans/weave_agents_content_canary_CANARY_ID.gate.json
```

The gate summary is offline and does not call W&B or any model provider. It
classifies the local run as `passed`, `not_run`, `provider_failure`,
`model_configuration_failure`, `trace_missing`, `content_missing`,
`tool_trace_missing`, `tool_content_missing`, `canary_text_missing`,
`usage_missing`, `trace_order_invalid`, or `trace_error`. A production content
canary is accepted only when this JSON has `"ok": true` and `"status":
"passed"`.

Production agentic/full batch execution should enforce that gate before any
paid model calls:

```bash
uv run python scripts/tools/run_taiwan_full_eval_batch.py \
  --phase agentic \
  --weave-content-canary-gate outputs/weave_agents_content_canary/plans/weave_agents_content_canary_<CANARY_ID>.gate.json \
  --require-weave-content-canary \
  ...
```

If the gate fails, the batch runner stops before `scripts/run_eval.py` and
writes `weave_content_canary_gate_failed` to the paid-run review JSON.

Use the production-readiness gate wrapper to refresh local setup evidence and
consolidate W&B completion evidence, Weave content-canary status, NeMoClaw
readiness, and one-model canary review status:

```bash
uv run python scripts/tools/run_taiwan_production_readiness_gate.py \
  --report-json outputs/taiwan_full_eval/taiwan_production_readiness_report.json \
  --fail-on-not-ready
```

Add `--required-wandb-run-id-all RUN_ID` when a specific one-model canary or
release run is under review. This pins every required W&B completion benchmark
to the same run and prevents older successful verifier JSONs from satisfying
the gate. Use per-benchmark `--required-wandb-run-id benchmark=RUN_ID` only for
deliberate multi-run review packages.

The Weave content-canary gate also enforces freshness. By default a passing
`*.gate.json` must be no older than 24 hours, using `generated_at` when present
and file mtime as a fallback. Override with
`--weave-content-canary-max-age-seconds SECONDS` only when the review policy
explicitly accepts a different freshness window.

`run_taiwan_full_eval_batch.py` enforces the same freshness window before paid
agentic/full execution when `--require-weave-content-canary` is used.

The same run-id pin is checked against completed one-model canary review
records. The report reads `wandb_run_id` and `runs[].wandb_run_id` from those
records, so a matching W&B completion verifier JSON alone does not prove the
one-model canary gate.

Completed one-model canary review records must also include passing
`runs[].wandb_completion[]` entries for the required W&B benchmarks. This keeps
W&B logging as a completion condition in both the standalone verifier JSONs and
the paid-run review package.
Each entry's `path` is opened and checked against the verifier JSON; `ok`,
`benchmark`, and `run_id` must match the review entry.
W&B completion verifier JSONs have the same default 24-hour freshness window,
using the verifier JSON `generated_at` field. File mtime is not accepted for
W&B completion freshness. Override with
`--wandb-completion-max-age-seconds SECONDS` only when the review policy
explicitly accepts a different window.

If the report returns `not_ready`, its top-level `remediation_plan` lists the
blocking gates and the exact commands for the next verification step. Treat
commands containing `--yes-i-accept-third-party-software` as explicit operator
actions, not automatic setup steps.

If the OpenClaw gateway is run as a systemd user service, remember that it does
not automatically read this repository's `.env` unless a drop-in is installed.
For persistent local setup, write a systemd user drop-in that references the
existing `.env` without copying or printing secret values:

```bash
scripts/setup/install_openclaw_weave.sh --all
scripts/setup/install_openclaw_weave.sh --restart-gateway

# Or use the lower-level helper directly:
python3 scripts/setup/configure_openclaw_gateway_env.py --check-only
python3 scripts/setup/configure_openclaw_gateway_env.py --write --restart
```

The script checks the `source: env` SecretRefs in `~/.openclaw/openclaw.json`
against `.env`, writes
`~/.config/systemd/user/openclaw-gateway.service.d/20-nejumi-env.conf`, runs
`systemctl --user daemon-reload`, and restarts `openclaw-gateway.service` only
when `--restart` is provided.

For a one-off local content canary, start the gateway from an environment that
has the same keys loaded:

```bash
set -a
source .env
set +a
openclaw gateway run --force --port 18789
```

This keeps secrets out of `~/.openclaw/openclaw.json`, but the live gateway
process still receives the `WANDB_API_KEY`, `OPENAI_API_KEY`, and any other
configured provider keys it needs to resolve SecretRefs.

## Tool Policy

Allowed for Agentic SWE:

- repository filesystem access scoped to the task checkout
- shell commands inside the task container
- code editing
- test execution
- final answer / patch submission

Allowed for Agentic Math:

- Python
- scratch file
- final answer

Disabled:

- web search
- external CAS or benchmark lookup
- provider-specific privileged agent runtimes
- model-specific extra retries

## Scoring

Only task success is included in the leaderboard score:

- Agentic SWE: Pass@1
- Agentic Math: correctness

Cost, token usage, steps, wall-clock time, failure type, and trace URL are detail metadata only.
