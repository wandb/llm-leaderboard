# Taiwan Leaderboard Launch Progress

Last updated: 2026-07-07 03:05 JST

## Current Decision

| Item | Status | Evidence / note |
|---|---|---|
| One-model full canary target | In progress | Target model is `openai-direct/gpt-4.1-mini-2025-04-14`. |
| Full paid run | Blocked before full execution | Not started because the live Weave Agents content gate still fails. |
| NeMoClaw runtime | In progress | `nejumi-taiwan` sandbox runs OpenClaw gateway and emits Agents spans. |
| Weave Agents span visibility | Partial | Agents API shows `invoke_agent`, `chat`, `execute_tool`, model, timestamps, and token usage. |
| Weave Agents content visibility | Blocked | Agents API returns empty `input_messages`, `output_messages`, `tool_call_arguments`, and `tool_call_result` even though plugin-side debug confirms content was captured before export. |
| Cost posture | Guarded | Only small content canaries and synthetic trace probes have been run in this step; full benchmark execution is held to avoid spending on non-reviewable evidence. |

## Benchmark-Level Progress

| Benchmark / unit | Type | Data / artifact status | Harness status | Current blocker |
|---|---|---|---|---|
| TMMLU+ | Non-agentic | Available in Taiwan config path | Wired through Jaster path | Needs full canary execution. |
| TMMLU+ robustness | Non-agentic ALT | Uses TMMLU+ with robustness perturbation path | Wired as Jaster auxiliary flag | Needs full canary execution. |
| MT-Bench TW | Non-agentic judge | Config present; judge uses Taiwan config | Wired | Needs full canary execution. |
| HLE zh-TW | Non-agentic judge | Artifact/config present; translation status tracked separately | Wired | Needs full canary execution and judge-cost review. |
| HalluLens zh-TW | Non-agentic judge | Evaluator/config present | Wired in `run_eval.py` after A690 | Needs full canary execution. |
| IFEval zh-TW | Non-agentic | Evaluator/config present | Wired in `run_eval.py` after A690 | Needs full canary execution. |
| TCEval-v2 | Non-agentic | Evaluator/config present | Wired in `run_eval.py` after A690 | Needs full canary execution. |
| TS-Bench | Non-agentic safety | Evaluator/config present | Wired in `run_eval.py` after A690 | Needs full canary execution. |
| Script adherence | ALT | Evaluator/config present | Wired in `run_eval.py` after A690 | Needs full canary execution. |
| BFCL v3 zh-TW | Tool/function calling | Current Taiwan path present; BFCL latest upgrade remains separate work | Existing path available | Needs full canary execution; BFCL update/sampling/translation remains later task. |
| Agentic Math | Agentic | OlymMATH-HARD/AIME paths exist in repo history; current canary uses OpenClaw protocol | NeMoClaw/OpenClaw path works, but Agents content proof fails | Weave Agents content gate. |
| SWE-Bench Pro compact | Agentic | `swebench-pro-public:v2`, `leaderboard_compact_80`, `max_input_tokens=1000000`, `max_tool_calls=60` | NeMoClaw copy-mode path and runtime caps implemented | Weave Agents content gate before paid full run. |
| TWBias | Pending | License not production-acceptable | Marked pending/non-required | License decision. |
| aggregate_taiwan | Aggregate | Taxonomy weighting fixed after A690 | Wired as final aggregate evaluator | Requires completed benchmark outputs. |

## Latest Live Checks

| Check | Result | Path / trace |
|---|---|---|
| Weave content canary r17 | Failed content gate | `outputs/weave_agents_content_canary/plans/weave_agents_content_canary_tw_openai_mini_full_canary_20260707_content_r17.gate.json` |
| r17 latest Agents trace | Partial success | Trace `fe65472d25cbe2496d80776c49772402`; spans and usage visible, content missing. |
| Weave SDK update probe | No fix | Sandbox Node `weave` updated from `0.15.1` to `0.16.1`; `weave-openclaw` remains `0.1.1`. |
| Weave content canary r18 | Failed content gate | `outputs/weave_agents_content_canary/plans/weave_agents_content_canary_tw_openai_mini_full_canary_20260707_content_r18_weave0161.gate.json`; trace `adf2a25ba399ad8e14695a340aa0897a`, usage visible, content still missing. |
| Plugin-side content debug | Captured before export | Sandbox debug shows `shapedInput=2`, `shapedOutput=1`, and tool args/result captured. |
| Synthetic no-model trace probe | Written, not usable yet | `nejumi-weave-event-content-smoke`; not returned by Agents spans query within retry window. |

## Next Step

| Priority | Action | Completion condition |
|---|---|---|
| P0 | Resolve or formally re-scope Weave Agents content proof | Either Agents API exposes message/tool content for live traces, or release criteria are changed with explicit rationale and tests. |
| P1 | Run `gpt-4.1-mini-2025-04-14` full canary | W&B completion, Weave/NeMoClaw proof, paid review, Total Score, and measured cost are all produced. |
| P2 | Review canary results | Compare score balance, failure modes, and measured cost before expanding to 5-10 models. |
