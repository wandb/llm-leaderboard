# DeepSWE public result gap investigation

Date: 2026-07-17

## Executive conclusion

The gap is real enough that it should not be explained as ordinary sampling noise.
The two fresh Taiwan High-8 GLM-5.2 runs scored 3/16 (18.75%), while the public
DeepSWE GLM-5.2 max task-level rates imply 7.5/16 expected successes. Under those
public per-task probabilities, the exact probability of observing three or fewer
successes is 1.406%.

The most important finding is that these are not equivalent agent systems:

1. The public result runs mini-swe-agent inside each task's dedicated DeepSWE
   image. The Taiwan adapter downloads `/app` from that image, copies it into one
   generic NeMoClaw sandbox, lets OpenClaw work there, then applies only the patch
   back to the Pier task container for verification.
2. The generic sandbox approximates task dependencies with copied Go caches and
   Python overlays. Its Go readiness check accepts any non-empty module cache,
   rather than proving that every selected task's exact modules are present.
3. Public mini-swe-agent requires a bash tool call on every model response and an
   explicit submit command. OpenClaw can terminate on an ordinary final message.
   Seven of eight replication-2 trials ended normally before the 30-minute task
   timeout, including several near misses and one empty patch.
4. The public run used direct Z.AI with exact `reasoning_effort=max` and persistent
   thinking. The installed OpenClaw OpenAI-compatible adapter maps `max` to
   `xhigh` before forming an OpenRouter request. Provider routing was stable, but
   the model request was not identical to the public request.
5. Public DeepSWE permits 90 minutes and has no mini-swe step or cost cap. The
   Taiwan High profile permits 30 minutes, 140 turns, 200 tools and 13M cumulative
   input tokens. One local task exhausted these limits; some public successful
   trajectories also exceed them.

Therefore, the observed score is currently a valid result for a distinct
`GLM-5.2 + OpenClaw + NeMoClaw + Taiwan budget + generic sandbox` system. It is
not yet a controlled reproduction of the public `GLM-5.2 + mini-swe-agent +
task image + direct Z.AI` result. The task subset itself should stay frozen while
the execution contract is repaired and calibrated.

## Compared evidence

### Public DeepSWE GLM-5.2 max

The cached public v1.1 trial index contains the following exact configuration:

- model: `glm-5-2`
- provider: `zai`
- harness: `mini-swe-agent`
- config: `mini_swe_agent_glm_5_2_max`
- reasoning effort: `max`
- score: 197/450 included trials = 43.78%
- full-corpus mean: 129.1 steps, 12.61M input tokens, 78.2k output tokens,
  $3.92 and 43.8 minutes per included trial

For the frozen High-8, 30 of 32 raw public trials were included and 14 passed:

| Task | Public included passes | Public rate |
|---|---:|---:|
| `go-genai-streamed-function-args` | 3/4 | 75.0% |
| `etree-xml-diff-patch` | 2/4 | 50.0% |
| `ytt-jsonpath-query-api` | 2/4 | 50.0% |
| `mnamer-daemon-watch-lifecycle` | 1/3 | 33.3% |
| `langchain-request-coalescing` | 2/3 | 66.7% |
| `ofetch-per-origin-circuit-breaker` | 1/4 | 25.0% |
| `kea-atomic-signal-selectors` | 2/4 | 50.0% |
| `happy-dom-deterministic-intersectionobserver` | 1/4 | 25.0% |

The 30 included selected trials average 70.8 steps, 5.21M input tokens, 62.4k
output tokens, $1.82 and 46.3 minutes. The 14 successful selected trials average
90.9 steps, 6.97M input tokens, 76.7k output tokens, $2.40 and 36.5 minutes.
Selected successful trajectories reach 154 steps and 15.04M input tokens.

The public DeepSWE page reports the rounded full result as 44% +/- 2%, $3.92,
78k output tokens and 129 steps. The public repository states that all
leaderboard results use Pier, mini-swe-agent and isolated task environments.

### Taiwan OpenClaw replications

Both local runs used the same frozen eight tasks, Z.AI-first OpenRouter routing,
FP8 provider selection, max thinking at the CLI, 2 workers and fresh/no-resume
execution.

| Task | Replication 1 | Replication 2 | Public max rate | Replication-2 F2P / P2P |
|---|---:|---:|---:|---:|
| `go-genai-streamed-function-args` | fail | fail | 75.0% | 0/6, 0/62 |
| `etree-xml-diff-patch` | fail | fail | 50.0% | 0/52, 15/15 |
| `ytt-jsonpath-query-api` | pass | fail | 50.0% | 101/103, 1/1 |
| `mnamer-daemon-watch-lifecycle` | fail | fail | 33.3% | 34/51, 319/319 |
| `langchain-request-coalescing` | fail | fail | 66.7% | 49/50, 232/232 |
| `ofetch-per-origin-circuit-breaker` | pass | pass | 25.0% | 47/47, 13/13 |
| `kea-atomic-signal-selectors` | fail | fail | 50.0% | 0/12, 139/139 |
| `happy-dom-deterministic-intersectionobserver` | fail | fail | 25.0% | 13/14, 9/9 |

The strict agreement was 7/8. The combined local per-task rates correlate poorly
with the eight public rates (Pearson -0.425, Spearman -0.378), although n=8 and
the local rates are only 0, 0.5 or 1. This is not a stable estimator by itself,
but it demonstrates that public task difficulty is not transferring cleanly to
the current harness. It does not invalidate the subset's public-model OOF
correlation; it invalidates the assumption that the correlation automatically
survives a different agent and runtime contract.

Replication 2 produced valid Pier results and native traces for all eight tasks.
The exact OpenRouter audit covered 581/581 model responses. Z.AI served 577; four
Z.AI 504 responses were successfully retried by SiliconFlow. No task was lost to
an unrecovered provider error.

## Root causes by confidence

### P0: the agent-visible environment is not the task environment (confirmed)

DeepSWE defines a dedicated Docker image per task. The public agent works in that
image, and Pier later verifies its committed patch in a pristine verifier image.

The Taiwan Pier adapter instead:

1. downloads `/app` from the task environment to a host checkout;
2. copies that checkout into a shared generic NeMoClaw sandbox;
3. runs OpenClaw in the generic sandbox;
4. captures a patch from the generic sandbox; and
5. applies that patch back to the Pier task environment for grading.

This is visible in `deepswe_openclaw_pier_agent.py`: `/app` is downloaded at
lines 666-673, while the NeMoClaw copy path is prepared at lines 470-488 and its
patch is captured at lines 571-576. The adapter's opening comment says Pier owns
environment setup, but Pier's environment is not the environment in which the
model executes commands.

The dependency overlay cannot guarantee equivalence. In particular,
`check_go_module_cache()` reports success if it finds any `*@v*` directory under
the shared cache. Once that generic check succeeds, the installer exits early
and does not merge caches from the current selected images. A stale cache left by
an earlier subset can therefore pass readiness without the exact versions needed
by the current Go tasks.

This is not hypothetical: the replication-2 Go trial spent its final turns
diagnosing module versions absent from the generic cache and then exhausted the
140-turn/13M-input budget. The official task image is the canonical dependency
environment, so this failure should not have consumed model budget.

A no-network local probe confirms the mismatch. In the exact task image
`sha256:a945cc9c...adb5`, at base commit `87c0e5a...475`,
`GOPROXY=off go test -run '^$' ./...` compiles both packages successfully. The
same compile probe in the agent-visible shared sandbox fails before compilation,
attempting to download `cloud.google.com/go`, `github.com/gorilla/websocket`,
`github.com/google/go-cmp` and other modules while `GOPROXY=off`. The task image
has the needed cache; the environment that consumed the model's turns does not.

Current readiness is consequently too weak. It proves that task images exist,
that `go`, `node` and `python3` exist in the shared sandbox, that some Go module
cache exists, and that Python overlays import two generic packages. It does not
run each task's native dependency/test probe in the actual agent-visible
sandbox. Nevertheless, those booleans are combined into
`full_run_recommended=true`.

### P1: agent protocol and termination differ materially (confirmed)

The inspected public GLM trajectories used mini-swe-agent 2.4.2 with unlimited
step and cost limits. Its scaffold is deliberately narrow:

- one bash tool;
- every assistant response must contain at least one bash call;
- a linear history;
- completion only through the exact submit command
  `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`;
- after submission, the agent cannot continue.

All 30 inspected public selected trajectories had zero no-tool agent steps.

OpenClaw exposes a richer tool surface and treats ordinary final text as model
completion. In replication 2, seven tasks ended normally before the 30-minute
limit. Several asserted that implementation was complete despite one or two
held-out failures. `kea` is the clearest protocol failure: its last model call
used the full 65,536 output tokens, ended with `finish_reason=length`, produced no
tool call, and the run then stopped with an empty patch. A mini-swe-style format
guard would reject that response and continue or mark a format failure instead
of silently accepting completion.

This explains why simply increasing the wall timeout would not repair most
failures: seven trials did not reach the wall timeout. They ended under different
completion semantics.

The Taiwan wrapper also changes task instructions. It strips branch/commit
requirements and adds a generic completion contract. Removing harness-specific
commit requirements is defensible because the adapter captures a worktree diff,
but it means the prompt is not a reproduction of the public task/agent protocol.

### P1: exact model request differs (confirmed request mismatch, effect uncertain)

Public trajectories show direct Z.AI requests configured with:

```text
model=zai/glm-5.2
reasoning_effort=max
thinking.type=enabled
thinking.clear_thinking=false
```

The installed OpenClaw bundle computes:

```javascript
clampedReasoning === "max" ? "xhigh" : clampedReasoning
```

and sends OpenRouter `reasoning.effort` using that mapped value. No model-specific
thinking-level map restores `max` in the inspected run configuration. Thus the
UI/CLI label `max` did not mean the same API request as public DeepSWE max.

The local response audit also shows slightly less reasoning/output intensity:
about 60.0% of completion tokens were reasoning locally, versus about 67.2% in
the inspected public selected calls. This comparison is descriptive, not proof
that effort mapping accounts for a particular number of failed tasks.

### P2: budgets are lower than the public contract (confirmed, partially causal)

Every selected DeepSWE task allows 5,400 seconds for the agent. Public
mini-swe-agent trajectories set step and cost limits to zero. The Taiwan High
profile uses:

- 1,800 seconds per task;
- 140 agent turns;
- 200 tool calls;
- 13M cumulative input tokens;
- 500k cumulative output tokens;
- 900-second response-idle timeout.

The Go trial directly hit the local limits. Public selected successful trials can
reach 154 steps and 15.04M input tokens, so the current caps exclude at least one
known successful public trajectory. Across all 113 tasks, public included max
trajectories have much longer tails (up to 325 steps and 63.02M input tokens in
the cached rows).

Budgets are an intentional product choice, not inherently a bug. They do mean the
score should be named and calibrated as a budgeted agent score. They should not
be presented as an expected reproduction of the uncapped public score.

### P2: policy/prompt friction is secondary but real (confirmed)

The run observed policy blocks on ordinary library names in two tasks. A denied
operation should return a clear tool error so the agent can choose another path;
it should not automatically make the task non-scoreable or terminate the entire
evaluation. Textual URL/package restrictions also spend context explaining a
harness-specific constraint that the public agent does not see.

Network isolation should be enforced at the sandbox/network layer. Prompt text
should briefly state the available capabilities, while ordinary local uses of
`requests`, `urllib`, `httpx`, package managers and repository tooling should not
be rejected merely because their names occur in a command.

## Explanations ruled out or demoted

| Hypothesis | Finding |
|---|---|
| Wrong task IDs, base commits or verifier definitions | Ruled out. The frozen records map to the local official DeepSWE v1.1 task definitions. |
| Pier grading failure | Ruled out for these runs. All 16 trials were scoreable; replication 2 had zero Pier errors/retries. |
| Missing native traces | Ruled out. Native trace evidence was 8/8 in each run. |
| Unrecovered OpenRouter 504 | Ruled out as the direct score cause. Four events in each run recovered through configured fallback; no task-level provider failure remained. |
| Pure random variance | Unlikely as a complete explanation. Exact lower-tail probability for 3/16 is 1.406% under public task rates. |
| Wall timeout alone | Demoted. Only one replication-2 task hit the runtime budget; seven ended normally under OpenClaw completion semantics. |
| FP8 provider selection | Not supported as the main explanation. Nearly all audited calls used Z.AI FP8, as intended. Direct Z.AI versus OpenRouter and payload semantics remain uncontrolled differences. |

## Evaluation-design implications

DeepSWE evaluates a system, not an abstract model. A score depends on at least:

```text
model + provider payload + agent scaffold + tools + task environment
+ termination rule + budgets + verifier
```

There are two defensible targets:

1. **Public comparability target.** Use mini-swe-agent 2.4.2, the exact task image,
   direct Z.AI request semantics and public 90-minute/uncapped policy. This is a
   control condition, but it is expensive and is not the Taiwan product's desired
   OpenClaw/NeMoClaw experience.
2. **Taiwan governed-agent target.** Keep OpenClaw/NeMoClaw, explicit budgets,
   native W&B traces and policy controls. Name the tier as DeepSWE-derived under
   the Taiwan OpenClaw contract, and calibrate it using multiple reference models
   run through this exact harness. Public DeepSWE remains a task-selection prior,
   not a score-equivalence promise.

The second target is appropriate for Agentic SWE-Assorted. It still requires the
agent-visible environment to be faithful, because dependency failures measure
the harness rather than software-engineering ability.

## Recommended repair sequence

### 1. Make task-environment parity a release gate

Preferred design: OpenClaw's shell/file tools execute inside Pier's per-task
agent container, while the NeMoClaw gateway remains responsible for model access,
policy and tracing. The repository should not be copied into a generic runtime.

If NeMoClaw cannot delegate tools into an existing Pier container, create one
task-specific OpenShell/NeMoClaw sandbox from the exact DeepSWE task image. A
global union of caches is only a temporary compatibility path and should not be
considered release-grade.

The readiness report must record and verify, per task:

- expected task image and immutable image digest;
- image/digest actually visible to agent tools;
- repository HEAD/base commit;
- language/runtime versions;
- task-native dependency/test command succeeds offline before model invocation;
- no mutable shared cache from another task is required for success.

If overlays remain temporarily, cache markers must be keyed by task image digest
and must enumerate exact module/package artifacts. The current "any Go module is
present" predicate must be removed.

### 2. Add explicit, recoverable completion semantics

- Require a submit marker/tool for High tasks, analogous to mini-swe-agent.
- Before accepting submit, require a non-empty patch unless the model explicitly
  reports a concrete impossibility.
- Treat `finish_reason=length` without a tool/submit as recoverable; compact the
  state and continue within budget.
- Treat a plain final message as a request to submit, then run visible sanity
  checks and return failures to the agent if budget remains.
- Remove contradictory generic commit instructions. Either allow commits and
  extract them, or consistently score a worktree patch without telling the agent
  both behaviors.

### 3. Record the resolved model payload

- Preserve `max` when a provider/model supports it; do not silently map it to
  `xhigh`.
- Pass and record provider-specific `thinking` fields where supported.
- Store a redacted resolved request profile in run evidence: model revision,
  provider, precision, reasoning effort, thinking persistence, max output and
  fallback policy.
- Unit-test resolution without a paid call. Use one minimal paid request only
  after the resolved profile is correct and with explicit approval.

### 4. Revisit High budgets after environment/protocol repair

For public-like coverage of this frozen subset, current evidence suggests a
minimum diagnostic profile around 180 turns, 18M cumulative input and 60 minutes.
The exact public control is 90 minutes with no step/cost cap. This is not a
recommendation to raise every production limit immediately: first remove
environment and early-termination failures, then estimate the distribution from
successful Taiwan-harness trajectories.

High budget exceedance should be rare and visible. It remains a scoreable model
failure only after environment readiness has passed for that exact task.

### 5. Run a two-task discriminating pilot before another High-8

Do not pay for a third identical High-8 replication. After local/unit probes pass,
run only:

1. `go-genai-streamed-function-args` to test exact task-image dependency parity.
   It is public 3/4 for GLM-5.2 max and failed both local runs while visibly
   fighting the generic Go cache.
2. `langchain-request-coalescing` (or `ytt-jsonpath-query-api`) to test explicit
   submission and near-miss recovery. These reached 49/50 and 101/103 F2P in
   replication 2.

Acceptance criteria for each paid pilot:

- exact agent-visible task image/digest recorded;
- native dependency probe passes before model invocation;
- resolved reasoning setting is `max`, not silently `xhigh`;
- no policy false positive;
- no unrecovered provider error;
- no no-tool/truncated response is accepted as final without submit;
- verifier produces F2P/P2P and strict reward normally.

Only if both pilots satisfy the contract should another frozen High-8 run be
considered. A one-task official mini-swe-agent control would be useful to separate
model/provider effects from OpenClaw effects, but it is paid work and should be
run only with explicit approval.

## Evidence locations

- Cached public trials: `outputs/deepswe_subset_analysis/deepswe_v1_1_trials.json`
- Frozen subset metadata: `data/taiwan/deepswe/subsets/essential_anchored_high_8_wandb_glm52_cap100_10m_lang_balanced.jsonl`
- Replication 1 report: `outputs/agentic_swe_assorted_runs/glm52_zai_fp8_high8_combined_20260716/report.md`
- Replication 2 report: `outputs/agentic_swe_assorted_runs/glm52_zai_fp8_high8_rep2_20260717/replication_analysis.md`
- Replication 2 provider audit: `outputs/agentic_swe_assorted_runs/glm52_zai_fp8_high8_rep2_20260717/openrouter_provider_cost_audit.json`
- Pier adapter: `scripts/tools/deepswe_openclaw_pier_agent.py`
- Sandbox dependency installer: `scripts/setup/install_deepswe_sandbox_deps.sh`
- Readiness gate: `scripts/tools/check_agentic_swe_assorted_high_readiness.py`
- High budgets: `configs/base_config_taiwan.yaml`
- Official local repository: `external/deep-swe` at commit
  `3cda4081fed96103a6395de39c85e9b20275e307`
- Public leaderboard: <https://deepswe.datacurve.ai/>
- Public trial browser: <https://deepswe.datacurve.ai/data>
- Public repository: <https://github.com/datacurve-ai/deep-swe>
- mini-swe-agent: <https://github.com/SWE-agent/mini-swe-agent>

No additional paid model calls were made for this investigation.
