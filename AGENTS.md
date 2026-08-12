# Repository Guidelines

## Project Structure & Module Organization
The repo is configuration‑driven: keep per‑model YAML files inside `configs/` (for example `config-gpt-4o-2024-11-20.yaml`) and override what differs from `configs/base_config.yaml`. Python orchestration lives in `scripts/`, with `scripts/run_eval.py` as the entry point and evaluation helpers under `scripts/evaluator/`. Chat templates consumed by vLLM sit in `chat_templates/`, while benchmark documentation and assets stay in `docs/`. Keep generated data inside `artifacts/`, `predictions/`, or `volumes/` and out of git.

## Build, Test, and Development Commands
- `uv venv .venv-uv && source .venv-uv/bin/activate` — create the pinned Python 3.11 virtual environment.
- `uv pip install -r requirements.txt && uv pip install -r scripts/evaluator/evaluate_utils/bfcl_pkg/requirements.txt` — install baseline and BFCL extras.
- `bash run_with_compose.sh config-o3-2025-04-16.yaml [-d]` — launch docker services, optional vLLM, and run `scripts/run_eval.py`.
- `python3 scripts/run_eval.py -c config-gpt-4o-2024-11-20.yaml` — run manually when datasets and env vars are already present.
- `python3 scripts/test_chat_template.py -m qwen3-max -c chat_templates/qwen3-max.jinja` — sanity-check a custom chat template.

## Coding Style & Naming Conventions
Target Python 3.11, four-space indents, and comprehensive type hints/docstrings. Use `snake_case` for functions, variables, and module names, `UpperCamelCase` for classes, and keep config filenames in the `config-<model>.yaml` pattern. Run `pylint path/to/module.py` locally; the shipped `pylintrc` enforces ordering, naming, and docstring rules. Shell scripts should be executable, start with `#!/usr/bin/env bash`, and use descriptive verbs (`run_with_compose.sh`, `generate_docker_override.sh`).

## Testing Guidelines
Use `./test.sh` for a smoke path: it provisions the uv venv, installs dependencies, and executes `scripts/run_eval.py -c config-gpt-4o-mini-2024-07-18.yaml`. Prefer targeted harnesses for faster feedback (`scripts/test_vllm_detailed.py` for launch debugging, `scripts/debug_jtruthfulqa.py` for scoring sanity checks). Store temporary fixtures under `artifacts/` or `scripts/evaluator/.../fixtures`, and record WANDB run IDs in PR descriptions when referencing external evaluations.

## Commit & Pull Request Guidelines
Match the repo history by prefixing commits with a type and scope (`feat(bfcl,qwen): add Qwen3 handler`, `fix(aggregate): include HalluLens`). For each PR, include a concise description, the config or docs touched, reproduction commands (`bash run_with_compose.sh <config>`), and screenshots or tables for leaderboard/UI changes. Rebase before opening, ensure tests/lints pass, and link issues or W&B reports so reviewers can trace the impact.

## Security & Configuration Tips
Never commit `.env`, API keys, or private datasets; copy from `env.example` and inject secrets as environment variables (`WANDB_API_KEY`, `HF_TOKEN`, etc.). Keep downloaded artifacts inside `volumes/` and restrict model configs that point to private endpoints to local overrides. When sharing configs, redact tokens and confirm that docker commands only expose ports required for evaluation.
