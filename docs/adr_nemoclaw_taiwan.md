# ADR: NeMoClaw For Taiwan Agentic Benchmarks

Status: Accepted with conditions

Date: 2026-06-27

## Decision

Use NeMoClaw as the preferred sandbox backend for Taiwan agentic benchmarks
after installation and onboarding pass the readiness gate. This is a business
priority for the Taiwan leaderboard because the NVIDIA partnership value is
part of the release rationale, not merely an implementation detail.

Agentic Math can adopt NeMoClaw first. SWE-Bench Pro also has a mounted-checkout
NeMoClaw code path, but it must not be reported as completed until the local
NeMoClaw runtime is installed, the sandbox sees the same checkout tree the host
will score, and W&B/Weave plus the official evaluator all pass for that run.

## Context

The Taiwan leaderboard needs agentic benchmarks with auditability, network
control, and reproducible execution. OpenClaw provides the agent runtime and
W&B Weave integration. NeMoClaw adds an NVIDIA OpenShell sandbox layer around
OpenClaw.

Official references reviewed on 2026-06-27:

- NVIDIA NemoClaw repository: https://github.com/NVIDIA/NemoClaw
- NVIDIA NemoClaw OpenClaw quickstart:
  https://docs.nvidia.com/nemoclaw/user-guide/openclaw/get-started/quickstart

The GitHub README describes NemoClaw as an open source reference stack for
running AI agents more safely inside OpenShell sandboxes, with OpenClaw as the
default supported agent. It also states that NemoClaw is an alpha project. That
means the integration should be gated and explicit rather than silently assumed
to be production-ready.

## Implemented Gates

The repository now includes these implementation pieces:

- `scripts/setup/install_nemoclaw.sh`
  - read-only `--check-only`
  - explicit `--yes-i-accept-third-party-software` required for install/onboard
  - `--installer-sha256` and `--installer-review-json` required for install
  - no model inference
  - machine-readable `setup_plan` in JSON output so operators can review the
    install/onboard/post-check commands before accepting third-party terms
  - machine-readable `third_party_software` metadata and
    `setup_plan.acceptance_ledger_fields` so the release evidence records
    exactly which installer URL/ref, vendor, acceptance flag, review JSON,
    SHA-256, and operation logs were reviewed
  - setup-plan install/onboard commands include `--json` paths, so actual
    install attempts produce operation-result JSON and log evidence
- `scripts/setup/review_nemoclaw_installer.py`
  - downloads the installer for review without executing it
  - records URL/ref, size, SHA-256, recommended install command, and
    no-external-action flags
  - can verify the installer against
    `scripts/setup/nemoclaw_installer_lock.json` via `--lock-json`
  - gives the operator the exact `--installer-sha256` and
    `--installer-review-json` values required by `install_nemoclaw.sh`
- `scripts/tools/run_openclaw_agent_protocol.py`
  - protocol wrapper for `nemoclaw sandbox exec <sandbox> -- openclaw ...`
  - passes sandbox-visible OpenClaw config paths through `OPENCLAW_CONFIG_PATH`
- `scripts/tools/run_agentic_math_openclaw.py`
  - forwards `--nemoclaw-sandbox`, `--nemoclaw-bin`, and `--nemoclaw-workdir`
  - requires `--no-use-task-agent` with NeMoClaw
- `scripts/evaluator/agentic_math.py`
  - passes NeMoClaw settings from YAML config to the Agentic Math runner
  - rejects `nemoclaw_sandbox` with `use_task_agent=true`
- `scripts/tools/prepare_taiwan_full_eval_configs.py`
  - can opt Agentic Math into NeMoClaw via `--agentic-math-nemoclaw-sandbox`
  - can opt SWE-Bench Pro into NeMoClaw via `--swebench-pro-nemoclaw-sandbox`
    and `--swebench-pro-nemoclaw-checkout-sandbox-root`
- `scripts/tools/run_swebench_pro_openclaw.py`
  - can wrap SWE-Bench Pro patch generation in NeMoClaw
  - writes the per-task OpenClaw config under each checkout as
    `.nejumi_openclaw/openclaw_config.json`
  - passes the sandbox-visible config path into OpenClaw
  - defaults NeMoClaw workdir to the sandbox-visible checkout path
  - excludes `.nejumi_openclaw` from captured patches
- `scripts/tools/check_taiwan_canary_readiness.py`
  - reports NeMoClaw/OpenShell/sandbox status
  - `--require-nemoclaw` makes the sandbox preflight a hard gate
- `scripts/tools/build_taiwan_production_readiness_report.py`
  - includes NeMoClaw setup JSON evidence in the production gate
  - records `latest_setup_report` and `stale_ready_reports` so fresh local
    missing-command evidence is not hidden by older readiness files
- `scripts/tools/check_taiwan_nemoclaw_adoption.py`
  - reports the ADR adoption criteria as JSON/Markdown
  - checks setup-plan safety, installed commands, sandbox readiness,
    Agentic Math NeMoClaw config readiness, and the SWE-Bench Pro migration
    guard
  - emits `adoption_decision` so operators can distinguish
    `conditional_adopt_for_agentic_math`,
    `conditional_adopt_for_agentic_benchmarks`, and
    `do_not_adopt_until_remediated`
  - is run by the production/release gate wrappers, and its outputs are copied
    into the release evidence bundle
- `scripts/setup/verify_nemoclaw_post_install.py`
  - runs the post-install local verification sequence in one command
  - writes setup check, protocol preflight, canary readiness, adoption check,
    and summary JSON/Markdown evidence
  - explicitly records that it does not install/onboard software, query W&B, or
    launch model inference
  - is run by the production/release gate wrappers, and the release evidence
    bundle copies its summary plus referenced per-step JSON outputs

## Acceptance Criteria

NeMoClaw may be used for Taiwan Agentic Math and SWE-Bench Pro when all of
these are true:

1. `scripts/setup/install_nemoclaw.sh --check-only --json temp/nemoclaw_setup_check.json` passes and records `nemoclaw` and `openshell` as available.
2. The same setup JSON records `third_party_software` metadata and
   `setup_plan.acceptance_ledger_fields` for the NVIDIA NemoClaw installer,
   explicit acceptance flag, restricted policy tier, installer lock JSON,
   installer review JSON, SHA-256, integrity/provenance fields, and
   install/onboard operation logs.
3. `uv run python scripts/tools/run_openclaw_agent_protocol.py preflight --nemoclaw-sandbox nejumi-taiwan` passes.
4. `uv run python scripts/tools/check_taiwan_canary_readiness.py --require-nemoclaw --nemoclaw-sandbox nejumi-taiwan` passes. With
   `--require-nemoclaw`, the checker defaults omitted `--generated-agentic-dir`
   to `configs/taiwan_full/generated_openai_canary_agentic_nemoclaw`; explicit
   values still override the default.
5. A no-paid-inference config check confirms that generated configs set
   `agentic_math.nemoclaw_sandbox`, `agentic_math.use_task_agent=false`, and, if
   SWE-Bench Pro is in scope, `swebench_pro.nemoclaw_sandbox` matching the target
   sandbox.
6. `uv run python scripts/tools/build_taiwan_production_readiness_report.py` shows
   `nemoclaw_readiness.status=passed` and no stale readiness override from a
   newer failed setup check.
7. A paid or approved canary run logs W&B tables/scalars/artifacts and Weave
   Agents traces with ordered conversation and tool-call content.
8. For SWE-Bench Pro, the sandbox-visible checkout path is proven to be the same
   tree from which the host captures `git diff --binary`, and the official
   SWE-Bench Pro evaluator passes on the resulting patch file.

If install/onboard is requested or attempted for production evidence, the setup
JSON must record `installer_review_verified=true`,
`installer_integrity_verified=true`, and `installer_provenance_locked=true` in
both `third_party_software` and `setup_plan`. The release evidence verifier also
requires the install command's `--installer-review-json` path to match the
preceding `review_nemoclaw_installer.py --json` output in the same operator
plan, and requires both the review command's `--lock-json` and the install
command's `--installer-lock-json` to match the bundled installer review
summary.
When `latest_installer_review.lock_json` is present, the release evidence bundle
must also contain that lock JSON and prove `lock_verified=true` against the
reviewed installer URL/ref/SHA/size.

For production Taiwan leaderboard runs, NeMoClaw adoption requires
`policy_tier=restricted`. The setup script can still render broader policy
commands for operator review, but the adoption doctor rejects them until a
specific benchmark exception is implemented and reviewed.

## SWE-Bench Pro Runtime Requirement

SWE-Bench Pro uses the mounted-checkout design. The runner creates per-instance
repository checkouts on the host, writes OpenClaw runtime config into the
checkout, runs OpenClaw through `nemoclaw sandbox exec`, and captures patches
from the same host tree. This is valid only if the sandbox sees that exact tree
at either the same path or at the configured
`swebench_pro.nemoclaw_checkout_sandbox_root` path.

The sandbox-local repository snapshot/export design remains a possible future
alternative, but it is not the current implementation.

## Current Local State

As of 2026-06-30:

- Docker, Node.js, npm, zstd, and OpenClaw are present.
- `nemoclaw` and `openshell` are not installed locally.
- The current expected adoption recommendation is
  `conditional_adopt_for_agentic_math` or
  `conditional_adopt_for_agentic_benchmarks`, depending on whether the checked
  configs opt SWE-Bench Pro into NeMoClaw. Runtime installation and sandbox
  readiness still block actual use.
- Optional readiness mode reports the missing commands without failing the
  broader local readiness report.
- `--require-nemoclaw` fails as expected until installation and onboarding are
  explicitly completed.

## Consequences

Positive:

- Agentic Math gets a clear path to NVIDIA/OpenShell sandbox execution.
- SWE-Bench Pro now has a mounted-checkout NeMoClaw runner path instead of a
  permanent exclusion.
- Missing NeMoClaw setup cannot silently pass a canary when the hard gate is
  enabled.
- Third-party acceptance remains explicit and auditable.
- SWE-Bench Pro is protected from a misleading partial NeMoClaw claim because
  config, runtime readiness, W&B/Weave evidence, and official evaluator evidence
  are separate gates.

Tradeoffs:

- NeMoClaw does not materially reduce LLM API cost by itself; token volume is
  still dominated by model, step, tool-output, and retry behavior.
- Agentic Math must use the sandbox's configured OpenClaw agent instead of
  host-generated per-task OpenClaw config files.
- SWE-Bench Pro still needs local NeMoClaw install/onboard plus mounted-checkout
  runtime evidence before it can satisfy the partnership-driven NeMoClaw
  priority in production.
