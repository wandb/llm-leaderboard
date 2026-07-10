# Response to Fable Review: Parallel Control

作成日: 2026-07-10

## 結論

Fable の結論は `B: 小規模 probe を1回挟むべき + full再開前に小修正2件`。
小修正2件は実装済み。paid probe は未実行。

## 対応済み

### F1: 非BFCLベンチの provider pacing 欠落

対応:

- `scripts/evaluator/evaluate_utils/llm_async_processor.py`
  - top-level `provider_rate_limit` config を読む shared provider limiter を追加。
  - provider call 直前、かつ semaphore 内で `wait_async()` する。
  - default は無効なので、日本語版など既存 config には影響しない。
- `scripts/tools/prepare_taiwan_full_eval_configs.py`
  - generated Taiwan full config で top-level limiter を有効化。
  - default:
    - `provider_rate_limit.enabled: true`
    - `provider_rate_limit.key: llm:{slug}`
    - `provider_rate_limit.min_request_interval_sec: 1.0`
    - `provider_rate_limit.request_jitter_sec: 0.25`
- `scripts/evaluator/evaluate_utils/bfcl_pkg/bfcl/model_handler/openai_compatible_handler.py`
  - BFCL は dedicated limiter を持つため、`LLMAsyncProcessor(..., provider_rate_limit_enabled=False)` にして二重 pacing を回避。

### F2: BFCL fail-fast が timeout のみ対象

対応:

- `scripts/evaluator/evaluate_utils/bfcl_pkg/bfcl/_llm_response_generation.py`
  - `bfcl_case_timeout` だけでなく、`Error during inference:` 系も連続 failure として数える。
  - `consecutive_failure_fail_fast` 到達時に `BFCLStalledError` で即停止。
  - watchdog 周期待ちだけでなく、result 書き込み直後にも fail-fast 判定。
- `scripts/evaluator/bfcl.py`
  - default config に `consecutive_failure_fail_fast: 5` を追加。
- `configs/base_config_taiwan.yaml`
  - `bfcl.consecutive_failure_fail_fast: 5` を追加。
- `scripts/tools/prepare_taiwan_full_eval_configs.py`
  - generated full config に `bfcl.consecutive_failure_fail_fast: 5` を明示。

## 検証

- `python3 -m py_compile`:
  - `scripts/evaluator/evaluate_utils/llm_async_processor.py`
  - `scripts/evaluator/evaluate_utils/bfcl_pkg/bfcl/model_handler/openai_compatible_handler.py`
  - `scripts/evaluator/evaluate_utils/bfcl_pkg/bfcl/_llm_response_generation.py`
  - `scripts/evaluator/bfcl.py`
  - `scripts/tools/prepare_taiwan_full_eval_configs.py`
  - passed
- Related tests:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_provider_rate_limiter.py tests/test_taiwan_full_config_generation.py tests/test_agentic_math.py tests/test_swebench_pro.py tests/test_bfcl_timeout_probe.py tests/test_bfcl_generation_resume.py tests/test_bfcl_wandb_table_normalization.py tests/test_openai_responses_bfcl_handler.py tests/test_bfcl_jsonschema_prompt.py tests/test_bfcl_artifact_utils.py`
  - result: `155 passed`
- Generated GLM-5.2 full config spot check:
  - `provider_rate_limit.enabled: true`
  - `provider_rate_limit.key: llm:glm-5_2-openrouter-reasoning`
  - `provider_rate_limit.min_request_interval_sec: 1.0`
  - `provider_rate_limit.request_jitter_sec: 0.25`
  - `bfcl.num_threads: 4`
  - `bfcl.provider_min_request_interval_sec: 2.0`
  - `bfcl.provider_request_jitter_sec: 0.5`
  - `bfcl.consecutive_failure_fail_fast: 5`
  - `agentic_math.num_workers: 8`
  - `swebench_pro.openclaw_num_workers: 8`

## まだ未実行

- OpenRouter GLM-5.2 probe は未実行。
- paid full run は未実行。

## 残る実運用リスク

- top-level limiter は process-local。複数 run/process を同時に動かすと横断制御はできない。
- Agentic Math/SWE の provider request は OpenClaw 内部から発行されるため、top-level `LLMAsyncProcessor` limiter は効かない。ここは Fable 指摘どおり probe で実測して判断する。
- SWE 8 workers はローカル disk/Docker/OpenClaw session 負荷が高い可能性がある。

## 次の推奨

1. OpenRouter GLM-5.2 の BFCL profile probe。
2. OpenRouter GLM-5.2 の agentic profile probe。
3. どちらも pass なら GLM-5.2 full run 再開を検討。
