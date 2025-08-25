uv venv .venv-uv && source .venv-uv/bin/activate
uv pip install -r requirements.txt
uv pip install -r scripts/evaluator/evaluate_utils/bfcl_pkg/requirements.txt
python3 scripts/run_eval.py -c config-gpt-4o-mini-2024-07-18.yaml


