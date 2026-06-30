# NeMoClaw Integration Notes

NeMoClaw is NVIDIA's reference stack for running OpenClaw inside an OpenShell
sandbox. For the Taiwan leaderboard, the reason to use it is not cosmetic:
agentic benchmarks need stronger network and filesystem boundaries than a
plain host-side OpenClaw process can provide.

References:

- NVIDIA NeMoClaw docs: https://docs.nvidia.com/nemoclaw/user-guide/openclaw/get-started/quickstart
- NVIDIA NeMoClaw repository: https://github.com/NVIDIA/NemoClaw
- OpenClaw: https://openclaw.ai
- Taiwan adoption ADR: `docs/adr_nemoclaw_taiwan.md`

## Current Status

As of 2026-07-01 07:54 JST:

- Host prerequisites are present: Docker, Node.js, npm, zstd, and OpenClaw.
- `nemoclaw v0.0.55` and `openshell 0.0.44` are installed on this machine.
- The `nejumi-taiwan` sandbox is onboarded and is the default sandbox. A504
  onboarding used `--provider custom`, `--endpoint-url https://api.deepseek.com/v1`,
  `--provider-key-env DEEPSEEK_API_KEY`, `--model deepseek-v4-flash`, and
  `--gateway-port 18083`. The OpenShell gateway is connected at
  `http://127.0.0.1:18083`.
- `--provider build` is NVIDIA hosted endpoints and requires `NVIDIA_API_KEY`.
  That key is not required to use the NeMoClaw/OpenShell sandbox itself. Use
  `--provider openai` for OpenAI-hosted inference or `--provider custom` for an
  OpenAI-compatible/local endpoint. `provider=custom` accepts
  `COMPATIBLE_API_KEY`, `OPENAI_COMPATIBLE_API_KEY`, `VLLM_API_KEY`,
  `LITELLM_MASTER_KEY`, `LITELLM_API_KEY`, or `NEMOCLAW_PROVIDER_KEY` plus
  `NEMOCLAW_ENDPOINT_URL` or the supported endpoint aliases. Use
  `--provider-key-env NAME` when the effective proxy key is not the default
  alias. On this host, the LiteLLM proxy exposes `chat-model` at
  `http://127.0.0.1:8080/v1` when queried with its configured master key, but
  its upstream OpenAI key is currently rejected, so A504 used the DeepSeek
  OpenAI-compatible endpoint instead. A504 evidence confirms no configured API
  key value is written to setup/onboard/post-install JSON or operation logs.
- The Taiwan adoption decision is recorded in `docs/adr_nemoclaw_taiwan.md`:
  NeMoClaw is the preferred backend for Taiwan agentic benchmarks after hard
  readiness gates pass. Agentic Math now has a NeMoClaw task-agent path, and
  SWE-Bench Pro has a mounted-checkout path. A504 post-install verification
  passes for both Agentic Math and SWE-Bench Pro NeMoClaw config scope; the
  remaining SWE production proof is a completed run where the sandbox can see
  the same checkout tree later scored by the official evaluator.
- Native `weave-openclaw` is installed and enabled inside the `nejumi-taiwan`
  sandbox. `scripts/setup/configure_nemoclaw_weave.sh --check-only` reports
  plugin/config/policy/secret all OK and writes the latest local evidence to
  `temp/nemoclaw_weave_config_check_after_tests.json`.
- `configs/nemoclaw/policies/wandb_weave.yaml` is the W&B/Weave egress policy
  template for the sandbox. The latest host evidence
  `temp/nemoclaw_setup_check_20260701T075249.json` reports
  `policies: []` from `nemoclaw status --json`, so the current runtime sandbox
  policy should be treated as not observed/configured. Agentic benchmark
  anti-cheat still relies on generated OpenClaw `deny_tool` and
  `deny_argument_pattern` guards until a NeMoClaw runtime policy is visibly
  attached and re-verified. No provider-pricing/router egress policy is part of
  the current NeMoClaw sandbox policy.
- The repo now includes a reproducible setup/check script:
  `scripts/setup/install_nemoclaw.sh`.
- Setup plans and remediation commands include `--json` output paths for
  install/onboard actions, so third-party install attempts produce
  reviewable operation logs and JSON evidence instead of untracked terminal
  output.
- Setup plans now include `setup_plan.operator_sequence` and
  `setup_plan.expected_evidence_paths`. These fields define the production
  handoff order from check-only evidence through installer review,
  install+onboard, post-install verification, canary readiness, and production
  readiness. The adoption checker rejects setup JSONs that omit this sequence
  or whose sequence commands drift from the corresponding setup-plan commands.
- Setup/check JSON now includes a `third_party_software` section with the
  NVIDIA NemoClaw installer URL/ref, repository/docs URLs, acceptance flag,
  and acceptance state, plus `setup_plan.acceptance_ledger_fields`. The
  adoption checker treats older setup JSONs without this metadata as unsafe.
- Install/onboard setup evidence must link to a non-executing, lock-backed
  installer review JSON from `scripts/setup/review_nemoclaw_installer.py`.
  The install script verifies that the review JSON has `lock_verified=true`
  and that its `lock_json` matches `--installer-lock-json` before downloading
  the installer, then verifies the SHA-256 before executing the downloaded
  file.
- `scripts/setup/nemoclaw_installer_lock.json` pins the currently reviewed
  installer URL/ref/SHA/size. Operator review commands use `--lock-json` so a
  silent upstream installer byte change fails before install/onboard.
- `scripts/tools/run_openclaw_agent_protocol.py` can wrap OpenClaw calls with
  `nemoclaw sandbox exec <sandbox> -- openclaw agent ...`.
- `scripts/tools/run_agentic_math_openclaw.py` can forward
  `--nemoclaw-sandbox` to the protocol runner and build a per-task sandbox
  OpenClaw config from `/sandbox/.openclaw/openclaw.json`.
- `scripts/tools/run_swebench_pro_openclaw.py` can forward
  `--nemoclaw-sandbox` to the protocol runner, write the task OpenClaw config
  under the repository checkout as `.nejumi_openclaw/openclaw_config.json`, pass
  the sandbox-visible config path through `OPENCLAW_CONFIG_PATH`, and exclude the
  runtime directory from patch capture.
- `configs/base_config_taiwan.yaml` and
  `scripts/tools/prepare_taiwan_full_eval_configs.py` can carry Agentic Math and
  SWE-Bench Pro NeMoClaw settings into generated evaluation configs.
- `scripts/tools/check_taiwan_canary_readiness.py --require-nemoclaw` can make
  NeMoClaw/OpenShell/sandbox OpenClaw preflight a hard canary readiness gate.
- `scripts/tools/check_taiwan_nemoclaw_adoption.py` summarizes the ADR
  adoption criteria into JSON/Markdown: setup-plan safety, installed commands,
  sandbox readiness, Agentic Math config readiness, and SWE-Bench Pro
  migration guard.
- The adoption checker separates runtime readiness from the adoption decision.
  A local `status=not_installed` result can still emit
  `adoption_decision.recommendation=conditional_adopt_for_agentic_math` when
  the design and policy criteria pass but `nemoclaw`/`openshell` or sandbox
  readiness are not yet proven. This prevents "not installed yet" from being
  misread as "not suitable for this leaderboard."
- When an enabled SWE-Bench Pro config also sets the target
  `swebench_pro.nemoclaw_sandbox`, the adoption checker reports
  `scope=agentic_math_and_swebench_pro`; mismatched or partial SWE NeMoClaw
  config remains a design blocker.
- The adoption checker now treats `setup_plan` as valid only when it contains
  install, onboard, post-install check, post-install verification, canary
  readiness, production readiness commands, and install/onboard
  `operation_results`. Legacy setup-plan JSONs that omit the post-install,
  canary verification, or operation-result fields are not adoptable.
- `install_nemoclaw.sh` writes the requested `--json` setup evidence even when
  install or onboarding fails, including the return code for each attempted
  operation.
- `scripts/tools/build_taiwan_production_readiness_report.py` includes the
  latest `nemoclaw_setup_check*.json` as `latest_setup_report` and marks stale
  readiness evidence separately, so an older passing report cannot hide a newer
  missing local installation.

## Check Only

Read-only local check:

```bash
scripts/setup/install_nemoclaw.sh \
  --check-only \
  --json temp/nemoclaw_setup_check.json
```

The JSON file records command availability, Docker status, the selected
sandbox, provider, optional gateway port, the next action, and a `setup_plan`
containing install/onboard/post-check commands without accepting third-party
terms. `--check-only` does not run provider validation or benchmark inference.
By default it reads only exported environment variables. If credentials live in
the repo-local dotenv file, pass the file explicitly:

```bash
scripts/setup/install_nemoclaw.sh \
  --check-only \
  --gateway-port 18080 \
  --env-file .env \
  --sandbox nejumi-taiwan \
  --json temp/nemoclaw_setup_check.json
```

`provider_preflight` records only env variable names and booleans. It does not
print secret values, does not prove quota, and does not contact the provider.
For example, `provider=openai` expects `OPENAI_API_KEY` or
`NEMOCLAW_PROVIDER_KEY`; `provider=build` is NVIDIA hosted endpoints and
expects `NVIDIA_API_KEY` or `NEMOCLAW_PROVIDER_KEY`; `provider=custom` expects
one of `COMPATIBLE_API_KEY`, `OPENAI_COMPATIBLE_API_KEY`, `VLLM_API_KEY`, or
`NEMOCLAW_PROVIDER_KEY`, one endpoint value, and a model. `LITELLM_MASTER_KEY`
and `LITELLM_API_KEY` are also accepted for local LiteLLM proxy paths. For a LiteLLM proxy
that requires its own master key, set that value in an environment variable and
pass `--provider-key-env NAME`; the setup JSON records only the variable name.
Endpoint values can come from `--endpoint-url`, `NEMOCLAW_ENDPOINT_URL`,
`OPENAI_COMPATIBLE_BASE_URL`, `OPENAI_COMPATIBLE_API_BASE`,
`OPENAI_COMPATIBLE_ENDPOINT_URL`, `OPENAI_COMPATIBLE_ENDPOINT`,
`VLLM_ENDPOINT_URL`, or `VLLM_BASE_URL`.
For production adoption evidence, use
`setup_plan.production_install_and_onboard_command`; it is the same combined
install+onboard command as `setup_plan.install_and_onboard_command` and writes
one JSON carrying installer review, SHA-256, lock, third-party acceptance,
restricted policy, install result, and onboard result together.
`setup_plan.install_or_onboard_requires_explicit_acceptance=true` and
`setup_plan.acceptance_flag=--yes-i-accept-third-party-software` are part of
the evidence. The JSON also records `third_party_software`, including the
installer URL/ref and whether acceptance was supplied, and
`setup_plan.acceptance_ledger_fields`, so reviewers can see exactly which
fields prove third-party software acceptance and operation logging.
Review `setup_plan.operator_sequence` for the exact handoff order and
`setup_plan.expected_evidence_paths` for the files that must exist after a
production install/onboard attempt and post-install verification.
For production install/onboard, this includes the operation-log evidence
templates `temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.install.log` and
`temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.onboard.log`.
The JSON also records `operation_results.install` and
`operation_results.onboard`, including whether each action was requested,
attempted, skipped, its return code, and the operation log path. In check-only
mode the log path is empty because no install/onboard operation is attempted.
`setup_plan.post_install_verification_command` points to the one-command local
verifier described below.

Protocol-level preflight:

```bash
uv run python scripts/tools/run_openclaw_agent_protocol.py preflight \
  --nemoclaw-sandbox nejumi-taiwan
```

This reports `ok: false` until the sandbox exists and can run
`openclaw --version` inside the sandbox.

Canary readiness gate after installation/onboarding:

```bash
uv run python scripts/tools/check_taiwan_canary_readiness.py \
  --manifest configs/taiwan_openai_canary_models.yaml \
  --generated-full-dir configs/taiwan_full/generated_openai_canary \
  --generated-nonagentic-dir configs/taiwan_full/generated_openai_canary_nonagentic \
  --generated-agentic-dir configs/taiwan_full/generated_openai_canary_agentic_nemoclaw \
  --generated-agentic-aggregate-dir configs/taiwan_full/generated_openai_canary_agentic_aggregate \
  --require-nemoclaw \
  --nemoclaw-sandbox nejumi-taiwan \
  --json outputs/taiwan_full_eval/openai_canary_readiness_nemoclaw.json
```

Without `--require-nemoclaw`, the readiness checker reports NeMoClaw status but
does not fail the broader local readiness report just because the sandbox layer
has not been installed yet.
With `--require-nemoclaw`, omitting `--generated-agentic-dir` defaults the
agentic canary config directory to
`configs/taiwan_full/generated_openai_canary_agentic_nemoclaw`; explicit
`--generated-agentic-dir` values still take precedence.

Production readiness aggregation:

```bash
uv run python scripts/tools/run_taiwan_production_readiness_gate.py \
  --report-json outputs/taiwan_full_eval/taiwan_production_readiness_report.json \
  --fail-on-not-ready
```

The wrapper runs `install_nemoclaw.sh --check-only --json` first and passes the
fresh setup JSON into the generated report. Inspect
`nemoclaw_readiness.latest_setup_report`; that entry is the current host setup
evidence. If a newer setup check records missing `nemoclaw` or `openshell`,
older full-readiness evidence is listed under `stale_ready_reports` instead of
being accepted.

ADR adoption check:

```bash
uv run python scripts/tools/check_taiwan_nemoclaw_adoption.py \
  --setup-json temp/nemoclaw_setup_check_YYYYMMDDTHHMM.json \
  --readiness-json outputs/taiwan_full_eval/openai_canary_readiness_nemoclaw.json \
  --sandbox nejumi-taiwan \
  --agentic-config-glob configs/taiwan_full/generated_openai_canary_agentic_nemoclaw/*.yaml \
  --json temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.json \
  --markdown temp/taiwan_nemoclaw_adoption_check_YYYYMMDDTHHMM.md \
  --fail-on-not-adoptable
```

The production and release gate wrappers run this check automatically and copy
the JSON/Markdown outputs into the release evidence bundle. On a machine where
`nemoclaw` and `openshell` are still missing, the expected status is
`not_installed`; this is evidence that NeMoClaw cannot yet be claimed as the
active backend. Inspect `adoption_decision.recommendation` for the actual
adoption recommendation:

- `adopt_for_agentic_math`: ready to use for Agentic Math.
- `adopt_for_agentic_benchmarks`: ready to use for Agentic Math and SWE-Bench
  Pro.
- `conditional_adopt_for_agentic_math`: design/policy checks pass, but runtime
  install or sandbox readiness still blocks use.
- `conditional_adopt_for_agentic_benchmarks`: Agentic Math and SWE-Bench Pro
  configs are ready, but runtime install or sandbox readiness still blocks use.
- `do_not_adopt_until_remediated`: a design, policy, or config criterion must
  be fixed before adoption.
- `insufficient_evidence`: setup evidence is missing.

Post-install verification bundle:

```bash
uv run python scripts/setup/verify_nemoclaw_post_install.py \
  --sandbox nejumi-taiwan \
  --canary-manifest configs/taiwan_openai_canary_models.yaml \
  --generated-full-dir configs/taiwan_full/generated_openai_canary \
  --generated-nonagentic-dir configs/taiwan_full/generated_openai_canary_nonagentic \
  --generated-agentic-dir configs/taiwan_full/generated_openai_canary_agentic_nemoclaw \
  --generated-agentic-aggregate-dir configs/taiwan_full/generated_openai_canary_agentic_aggregate \
  --json temp/nemoclaw_post_install_verification_TIMESTAMP.json \
  --markdown temp/nemoclaw_post_install_verification_TIMESTAMP.md \
  --fail-on-failed
```

This command runs all required local post-install checks without installing
software, querying W&B, or launching model inference:

- `install_nemoclaw.sh --check-only`
- `run_openclaw_agent_protocol.py preflight --nemoclaw-sandbox`
- `check_taiwan_canary_readiness.py --require-nemoclaw`
- `check_taiwan_nemoclaw_adoption.py`

The post-install verifier defaults to the same OpenAI-direct canary manifest
and NeMoClaw agentic config directory shown above, so direct script execution
does not fall back to the non-NeMoClaw agentic canary config.

## SWE-Bench Pro

SWE-Bench Pro now has NeMoClaw runner paths for both mounted checkout trees and
explicit checkout copy. The runner still prepares host checkouts for the
official evaluator, but when `--nemoclaw-sandbox` is set it writes the per-task
OpenClaw config inside the checkout at `.nejumi_openclaw/openclaw_config.json`,
passes the sandbox-visible path to OpenClaw through `OPENCLAW_CONFIG_PATH`,
defaults the NeMoClaw workdir to the sandbox-visible checkout, and excludes
`.nejumi_openclaw` from captured patches.

The operator must ensure the sandbox sees the same checkout tree that the host
later scores. If host paths are mounted unchanged, omit
`--nemoclaw-checkout-sandbox-root`. If the sandbox sees checkout roots under a
different prefix, pass that prefix explicitly:

```bash
uv run python scripts/tools/run_swebench_pro_openclaw.py \
  --dataset-jsonl data/taiwan/swebench_pro_public/subsets/leaderboard_compact_80.jsonl \
  --output-dir outputs/taiwan_full_eval/swebench_pro/MODEL/openclaw \
  --checkout-root outputs/taiwan_full_eval/swebench_pro_checkouts/MODEL \
  --model openai-direct/gpt-4.1-mini-2025-04-14 \
  --thinking off \
  --nemoclaw-sandbox nejumi-taiwan \
  --nemoclaw-checkout-sandbox-root /sandbox/checkouts \
  --max-input-tokens 1000000 \
  --max-tool-calls 60
```

If the sandbox cannot see the host checkout root, use copy mode:

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
  --max-tool-calls 60
```

Copy mode uploads the prepared git checkout into the sandbox, runs OpenClaw
against that sandbox-side checkout, and captures `git diff --binary` in the
sandbox before returning the patch record to the host.

These code paths are covered by unit tests and config-generation tests.
`nemoclaw`, `openshell`, the `nejumi-taiwan` sandbox, and native
`weave-openclaw` config are present on this machine. A no-inference local git
repo probe has verified copy-mode transfer plus sandbox-side diff capture. Do
not mark a run as completed until a real NeMoClaw-backed benchmark run has W&B
logging, Weave Agents trace verification, and official SWE-Bench Pro evaluator
results.

The post-install verifier JSON has `will_launch_model_inference=false`,
`will_query_wandb=false`, and `will_install_or_onboard=false`, so it can be
attached to release evidence as an operator handoff artifact. The production and
release gate wrappers also run this verifier automatically. The release evidence
bundle copies the summary JSON/Markdown and the referenced setup, protocol
preflight, canary readiness, and adoption-check JSON files.

## Verify Operator Docs

Before handoff, verify this README still contains the command markers required
by the release evidence contract:

```bash
uv run python scripts/setup/verify_nemoclaw_operator_docs.py \
  --json temp/nemoclaw_operator_docs_verification_YYYYMMDDTHHMM.json \
  --markdown temp/nemoclaw_operator_docs_verification_YYYYMMDDTHHMM.md \
  --fail-on-failed
```

This command only reads `docs/README_nemoclaw.md` and
`scripts/setup/nemoclaw_installer_lock.json`. It does not download installers,
install or onboard NeMoClaw, query W&B, or launch model inference. The
production and release gate wrappers run it automatically and include the JSON
and Markdown output in release evidence.

## Review Installer

Before installing, generate review evidence without executing the installer:

```bash
uv run python scripts/setup/review_nemoclaw_installer.py \
  --url https://www.nvidia.com/nemoclaw.sh \
  --install-ref lkg \
  --expected-sha256 a4ebc5710dfd8b10035968fd25562773ad56ec4c73ecf9bfe5c78c77724000e7 \
  --lock-json scripts/setup/nemoclaw_installer_lock.json \
  --json temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json \
  --markdown temp/nemoclaw_installer_review_YYYYMMDDTHHMM.md
```

The review JSON records the installer URL/ref, byte size, SHA-256,
recommended install command, and explicit no-external-action flags. It does not
install or onboard software, query W&B, or launch model inference.
`scripts/setup/nemoclaw_installer_lock.json` pins the installer URL/ref,
expected SHA-256, size, and no-external-action flags. If NVIDIA changes the
installer bytes behind the same URL/ref, the review step fails before any
install command is run. The review command must pass the pinned SHA-256 through
`--expected-sha256`; the reviewed SHA-256 from the review JSON is the value to
pass as `--installer-sha256`; the JSON path itself must be passed as
`--installer-review-json`. Both the review command and the install command must
point to the same pinned lock path for release evidence.

## Install And Onboard

NVIDIA's installer requires explicit third-party software acceptance. The repo
script refuses to install without the acceptance flag, the reviewed SHA-256, and
the matching installer review JSON and installer lock JSON.

```bash
scripts/setup/install_nemoclaw.sh \
  --install \
  --onboard \
  --install-ref lkg \
  --installer-lock-json scripts/setup/nemoclaw_installer_lock.json \
  --installer-sha256 REVIEWED_INSTALLER_SHA256 \
  --installer-review-json temp/nemoclaw_installer_review_YYYYMMDDTHHMM.json \
  --sandbox nejumi-taiwan \
  --provider openai \
  --gateway-port 18080 \
  --env-file .env \
  --policy-tier restricted \
  --yes-i-accept-third-party-software \
  --json temp/nemoclaw_install_onboard_YYYYMMDDTHHMM.json
```

OpenAI-compatible/local endpoint example:

```bash
scripts/setup/install_nemoclaw.sh \
  --onboard \
  --sandbox nejumi-taiwan \
  --provider custom \
  --model chat-model \
  --endpoint-url http://127.0.0.1:8080/v1 \
  --provider-key-env LITELLM_MASTER_KEY \
  --env-file .env \
  --gateway-port 18081 \
  --policy-tier restricted \
  --yes-i-accept-third-party-software \
  --json temp/nemoclaw_onboard_custom_YYYYMMDDTHHMM.json
```

A504 verified endpoint example on this host:

```bash
scripts/setup/install_nemoclaw.sh \
  --onboard \
  --sandbox nejumi-taiwan \
  --provider custom \
  --model deepseek-v4-flash \
  --endpoint-url https://api.deepseek.com/v1 \
  --provider-key-env DEEPSEEK_API_KEY \
  --env-file .env \
  --gateway-port 18083 \
  --fresh \
  --yes-i-accept-third-party-software \
  --json temp/nemoclaw_onboard_A504_deepseek_port18083.json
```

Notes:

- `--provider openai` uses OpenAI-hosted inference via `OPENAI_API_KEY`.
- `--provider build` is NVIDIA hosted endpoints in the upstream quickstart
  flow and requires `NVIDIA_API_KEY`; use it only when the evaluation model
  endpoint itself should be NVIDIA-hosted.
- `--provider custom` is for OpenAI-compatible/local endpoints. The script maps
  `OPENAI_COMPATIBLE_API_KEY`, `VLLM_API_KEY`, `LITELLM_MASTER_KEY`, or
  `LITELLM_API_KEY` to the canonical
  NeMoClaw/OpenShell provider-key variables during onboarding without writing
  secret values to JSON or logs.
- `--provider-key-env NAME` takes precedence for `provider=custom`. This is the
  preferred path when a local proxy such as LiteLLM uses a key name that differs
  from the model-provider key. The script validates `NAME` as an environment
  variable name and records only `NAME`, never the value.
- `--endpoint-url` sets `NEMOCLAW_ENDPOINT_URL` for `provider=custom` and is
  recorded in setup-plan commands. The URL is not treated as a secret; API keys
  are still redacted by omission.
- `--gateway-port 18080` is only needed when the default OpenShell gateway port
  8080 is occupied.
- `--env-file .env` is optional but recommended when credentials are stored in
  the repo dotenv file rather than exported in the shell. The setup JSON records
  that the file was loaded but never records secret values.
- `--installer-review-json` must point to the JSON produced by
  `review_nemoclaw_installer.py` for the same URL/ref and SHA-256. The install
  script validates this before downloading the installer.
- `--installer-lock-json` must match the `lock_json` recorded by the reviewed
  JSON. The release bundle verifier also rejects operator/current-gate review
  or install commands whose lock path does not match the bundled installer
  review summary.
- Use `--policy-tier restricted` for production Taiwan benchmark runs. The
  adoption doctor rejects broader tiers until a benchmark-specific exception is
  implemented and reviewed.
- The script does not launch benchmark inference. `nemoclaw onboard` can still
  perform a small provider validation request while creating/configuring the
  sandbox, so treat onboarding as a paid/provider-touching action when the
  selected provider points at a paid API.
- Always keep `--json` on install/onboard commands. That JSON records
  `third_party_software`, `setup_plan.acceptance_ledger_fields`,
  installer review/integrity/provenance fields, `operation_results`, and points
  to the install/onboard stdout/stderr logs.
- Prefer the combined install+onboard command for production evidence. Separate
  install-only or onboard-only diagnostics can help local troubleshooting, but
  the adoption checker and release bundle verifier expect the production
  `setup_plan.production_install_and_onboard_command` evidence shape for final
  approval.
- If install or onboarding fails, the script still writes the requested
  `--json` file with `operation_results` before returning nonzero. Operation
  stdout/stderr is written next to the JSON as `*.install.log` /
  `*.onboard.log`, and the JSON records those paths. These files are
  release-bundle evidence when referenced by the setup JSON.
- Re-run `--check-only --json temp/nemoclaw_setup_check.json` after onboarding
  and require sandbox OpenClaw status to pass before using NeMoClaw for
  evaluation.

## Agentic Math

NeMoClaw can be used for Agentic Math after the sandbox is configured:

```bash
uv run python scripts/tools/run_agentic_math_openclaw.py \
  --dataset-jsonl data/taiwan/agentic_math_olymmath_hard_zh_tw/subsets/smoke.jsonl \
  --output-dir outputs/nemoclaw_agentic_math_smoke \
  --model inference/deepseek-v4-flash \
  --thinking high \
  --nemoclaw-sandbox nejumi-taiwan
```

The runner now keeps task-agent mode enabled for NeMoClaw by default. It reads
the sandbox OpenClaw config template from `/sandbox/.openclaw/openclaw.json`,
writes the per-task config under the sandbox task workspace, and points OpenClaw
at that sandbox-visible file.

For generated Taiwan full-evaluation configs, opt Agentic Math into NeMoClaw
first:

```bash
uv run python scripts/tools/prepare_taiwan_full_eval_configs.py \
  --manifest configs/taiwan_openai_canary_models.yaml \
  --canary \
  --phase agentic \
  --output-dir configs/taiwan_full/generated_openai_canary_agentic_nemoclaw \
  --output-root outputs/taiwan_full_eval \
  --agentic-math-nemoclaw-sandbox nejumi-taiwan
```

The generated Agentic Math config keeps `use_task_agent: true` by default when
a NeMoClaw sandbox is selected. Set `agentic_math_use_task_agent: false` in the
model manifest only when a separately reviewed sandbox agent already pins the
correct workspace.
The adoption doctor scans
`configs/taiwan_full/generated_openai_canary_agentic_nemoclaw/*.yaml` and the
older GLM canary glob by default, so this generated config is part of the local
release evidence without replacing the non-NeMoClaw canary config.

## SWE-Bench Pro Config Generation

Generate Agentic configs with both Agentic Math and SWE-Bench Pro pointed at the
same sandbox:

```bash
uv run python scripts/tools/prepare_taiwan_full_eval_configs.py \
  --manifest configs/taiwan_openai_canary_models.yaml \
  --canary \
  --phase agentic \
  --output-dir configs/taiwan_full/generated_openai_canary_agentic_nemoclaw \
  --output-root outputs/taiwan_full_eval \
  --agentic-math-nemoclaw-sandbox nejumi-taiwan \
  --swebench-pro-nemoclaw-sandbox nejumi-taiwan \
  --swebench-pro-nemoclaw-checkout-sandbox-root /sandbox/checkouts
```

Generated SWE-Bench Pro configs default to `leaderboard_compact_80`,
`max_input_tokens: 1000000`, and `max_tool_calls: 60`.
