# Taiwan Agentic Runtime Readiness Audit (2026-07-23)

## Scope

This audit covers the Taiwan leaderboard execution path for Agentic Math and
Agentic SWE-Assorted, including its SWE-bench Lite and DeepSWE runners. It is a
local code and test audit. No paid model request, W&B API request, or shared
NeMoClaw sandbox operation was performed during this audit.

## Frozen Evaluation Shape

Agentic SWE-Assorted contains 80 tasks:

| Tier | Source | Tasks | Score weight | Tool / turn limits | Task timeout |
| --- | --- | ---: | ---: | --- | ---: |
| Low | SWE-bench Lite selected slice | 36 | 1/3 | 40 / 40 | 900s |
| Middle | SWE-bench Lite selected slice | 36 | 1/3 | 40 / 40 | 900s |
| High | DeepSWE frozen High-8 | 8 | 1/3 | 200 / 150 | 1800s |

Low/Middle use a 1M cumulative input-token cap. High uses 13M. Both use a
500k cumulative output-token cap and a 120-second per-tool wall limit. The
per-response output cap is model-specific and is supplied as an OpenClaw model
override in `configs/taiwan_full_eval_models.yaml`.

The official leaderboard score remains binary. A separate diagnostic score
adds at most 0.3 points for an unresolved but scoreable patch using
`0.3 * (F2P_fraction * P2P_fraction)^2`. It requires patch-application and test
evidence and cannot turn an unresolved task into an official pass.

## Runtime Outcome Contract

| Outcome | Classification | Retry | Patch / answer | Release effect |
| --- | --- | --- | --- | --- |
| Task wall time, tool/turn/token budget, response truncation, no model response | Model-side, scoreable | No | Preserve current patch; Math records incorrect | Valid model result |
| Denied external fetch or tool | Scoreable interaction | No automatic retry | Agent sees the tool error and may use an allowed approach | Valid model result |
| Provider 429/5xx/timeout | Infrastructure, non-scoreable | Bounded attempts, then deferred recovery rounds | Checkpoint healthy tasks | Run remains invalid if recovery is exhausted |
| Authentication, quota, unknown model, setup, sandbox, required trace failure | Infrastructure/configuration, non-scoreable | Only explicitly recoverable workspace repair | Cancel peer processes and checkpoint | Run fails before publication |
| SIGINT/SIGTERM or parent failure | Operator/runtime interruption | No | Terminate child process groups; retain checkpoints | Resume required |

Provider recovery now has the same semantics in Math, SWE-bench, and DeepSWE:
healthy tasks finish, only provider-failed tasks are retried, and exhausted
provider failures cannot be counted as model errors. Retry and recovery-round
token usage, cost, attempts, and wall time are accumulated rather than replaced
by the final attempt.

## Budget Communication

The English task prompt states every active tool, turn, token, response, and
wall-time limit. It also states that execution stops immediately at a hard
limit and the current answer or git diff is submitted without a guaranteed
cleanup turn.

The `nejumi-budget-guard` plugin emits one warning at 50%, 75%, 87.5%, and 95%
of both tool-call and assistant-turn budgets. Tool warnings are appended to the
tool result; turn warnings are appended to agent context. The final warning
states the remaining count and submission consequence.

## Isolation And Resume

- Evaluator subprocesses run in their own process groups and are terminated as
  groups on timeout, peer failure, SIGINT, or SIGTERM.
- A filesystem lease prevents two direct or evaluator-launched runs from
  mutating the same named NeMoClaw sandbox concurrently.
- `no_local: true` requires a non-empty named sandbox before runner launch.
- Parallel fatal failures cancel peer commands and retain ordered partial
  outputs. Model-side scoreable stops do not cancel healthy peers.
- Cache keys include runner version, prompt hash, model, runtime limits, policy,
  and OpenClaw configuration source. Changed contracts invalidate old results.
- Provider recovery state lists completed and pending task IDs so the same
  output directory can resume only unfinished infrastructure failures.

## Trace And Release Evidence

Native weave-openclaw trace verification records conversation identity, model,
usage, tool trace, and required prompt text per task. Manual Weave sidecars are
not accepted as production evidence. Model-side truncation remains a model
failure even if its final trace is incomplete; a general trace/configuration
failure remains non-scoreable.

Production readiness and NeMoClaw adoption tools now support strict evidence
mode (`--no-default-evidence-discovery`). Release tests and release automation
must use explicit evidence paths so stale historical reports cannot satisfy a
current gate accidentally.

## Verification And Remaining Work

Focused runtime, scoring, W&B completion, and release-gate tests pass locally.
The final repository suite completed with `1489 passed in 411.17s` using
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q`. Python compilation and
`git diff --check` also completed without errors.

The remaining validation is external, not a known code defect:

1. Use an isolated Taiwan-only NeMoClaw sandbox.
2. Run a small paid canary only after explicit approval, using direct OpenAI,
   direct Anthropic, or W&B Inference. Do not use OpenRouter while its shared
   credits are protected.
3. Exercise at least twice the configured worker count across Math and
   SWE-Assorted, then require zero non-scoreable failures and complete native
   trace evidence before a full paid run.

Until that canary passes, the harness is locally verified but not approved as
release-ready against live provider and sandbox infrastructure.
