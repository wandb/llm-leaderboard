# Fable Review Request: Taiwan Full Eval Parallel Control

作成日: 2026-07-10

## 背景

台湾リーダーボードの full 評価で、GLM-5.2/OpenRouter 実行時に次の問題が顕在化した。

- Agentic Math 50問のうち 38問時点で約7時間45分以上かかり、主因は timeout/retry による long tail だった。
- BFCL は `num_threads=2` でも provider 側 `limit_burst_rate` が発生した。理由は case 並列数ではなく、multi-turn case 内部で provider request が連続発行されるため。
- SWE-Bench Pro は80問想定で、2-4並列では実運用上長すぎる可能性が高い。
- ユーザー指示として、paid run を勝手に停止・resume しないこと、timeout は原則としてモデル側の未完了として採点し、同一 run 内で無制限に再課金しないことが重要。

## 今回実装した変更

### Agentic Math

- `scripts/tools/run_agentic_math_openclaw.py`
  - task-level parallel execution は既存実装を維持。
  - `--task-start-min-interval-seconds` を追加。
  - task worker が多い場合でも OpenClaw task 開始を stagger できるようにした。
  - `summary.json` に `num_workers` と `task_start_min_interval_seconds` を記録。
  - runner version を `agentic-math-openclaw-2026-07-10-timeup-parallel-v2` に更新。
- `scripts/evaluator/agentic_math.py`
  - config の `agentic_math.task_start_min_interval_seconds` を runner に渡す。
- `configs/base_config_taiwan.yaml`
  - `agentic_math.num_workers: 8`
  - `agentic_math.task_start_min_interval_seconds: 5.0`
  - `agentic_math.openclaw_timeout: 900`
  - `agentic_math.max_tool_calls: 40`
  - `agentic_math.max_agent_turns: 40`
  - `agentic_math.max_tool_wall_seconds: 120`
- `scripts/tools/prepare_taiwan_full_eval_configs.py`
  - generated full config の default を `math_num_workers=8`, `math_task_start_min_interval_seconds=5.0` に変更。

### SWE-Bench Pro

- `scripts/tools/run_swebench_pro_openclaw.py`
  - task-level parallel patch generation は既存実装を維持。
  - `--openclaw-task-start-min-interval-seconds` を追加。
  - OpenClaw task 開始を stagger できるようにした。
  - `summary.json` に `openclaw_num_workers` と `openclaw_task_start_min_interval_seconds` を記録。
  - runner version を `swebench-pro-openclaw-2026-07-10-timeup-parallel-v2` に更新。
- `scripts/evaluator/swebench_pro.py`
  - config の `swebench_pro.openclaw_task_start_min_interval_seconds` を runner に渡す。
- `configs/base_config_taiwan.yaml`
  - `swebench_pro.openclaw_num_workers: 8`
  - `swebench_pro.openclaw_task_start_min_interval_seconds: 15.0`
  - `swebench_pro.openclaw_timeout: 3600`
  - `swebench_pro.max_tool_calls: 40`
  - `swebench_pro.max_agent_turns: 40`
  - `swebench_pro.max_tool_wall_seconds: 300`
- `scripts/tools/prepare_taiwan_full_eval_configs.py`
  - generated full config の default を `swe_openclaw_num_workers=8`, `swe_openclaw_task_start_min_interval_seconds=15.0` に変更。
  - generated full config の SWE subset は現在のコスト制御方針に合わせて `leaderboard_compact_40`。

### BFCL

- `scripts/evaluator/evaluate_utils/provider_rate_limiter.py`
  - process-wide shared provider request limiter を新規追加。
  - key 単位で共有し、`min_interval_sec` と `jitter_sec` を制御。
  - sync/async 両方に対応。
- `openai_compatible_handler.py`
  - `_query_FC`, `_query_FC_async`, `_query_prompting`, `_query_prompting_async` の provider call 直前に shared limiter を差し込み。
- `openai_response.py`
  - Responses API 経路でも `generate_with_backoff` の provider call 直前に shared limiter を差し込み。
- `scripts/evaluator/bfcl.py`
  - `provider_min_request_interval_sec`, `provider_request_jitter_sec`, `provider_rate_limit_key` を config と `gen_args` に追加。
- `_llm_response_generation.py`
  - generation limits log に provider limiter 設定を出すようにした。
- `configs/base_config_taiwan.yaml`
  - `bfcl.num_threads: 4`
  - `bfcl.provider_min_request_interval_sec: 2.0`
  - `bfcl.provider_request_jitter_sec: 0.5`
  - `bfcl.case_timeout_retries: 0`
- `scripts/tools/prepare_taiwan_full_eval_configs.py`
  - generated full config の default を `bfcl_num_threads=4`, `bfcl_provider_min_request_interval_sec=2.0`, `bfcl_provider_request_jitter_sec=0.5` に変更。
  - `bfcl_provider_rate_limit_key` は default で `bfcl:{slug}`。

## 検証済み

- 構文チェック:
  - `python3 -m py_compile ...` 対象変更ファイルは成功。
- テスト:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_provider_rate_limiter.py tests/test_taiwan_full_config_generation.py tests/test_agentic_math.py tests/test_swebench_pro.py`
  - 結果: `136 passed`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_bfcl_timeout_probe.py tests/test_bfcl_generation_resume.py tests/test_bfcl_wandb_table_normalization.py tests/test_openai_responses_bfcl_handler.py tests/test_bfcl_jsonschema_prompt.py tests/test_bfcl_artifact_utils.py tests/test_provider_rate_limiter.py`
  - 結果: `19 passed`
- generated config spot check:
  - `glm-5_2-openrouter-reasoning`, phase `full`
  - `math_workers=8`
  - `math_start_interval=5.0`
  - `swe_workers=8`
  - `swe_start_interval=15.0`
  - `bfcl_threads=4`
  - `bfcl_provider_min_interval=2.0`
  - `bfcl_provider_jitter=0.5`
  - `math_limit=50`
  - `swe_subset=leaderboard_compact_40`

## 監査してほしい論点

1. **BFCL provider limiter の位置は十分か**
   - OpenAI-compatible handler と Responses handler には入れた。
   - 他の BFCL handler 経路で台湾 full eval に使われ得るものが残っていないか確認してほしい。

2. **BFCL limiter 設定値は妥当か**
   - 現状 default は `num_threads=4`, `provider_min_request_interval_sec=2.0`, `provider_request_jitter_sec=0.5`。
   - multi-turn を含む BFCL は約1000 provider calls 規模なので、理論下限は limiter だけで約34-42分。
   - burst 回避と所要時間のバランスとして妥当か確認してほしい。

3. **Agentic Math/SWE の worker 数と stagger は妥当か**
   - Math: 8 workers + 5s stagger。
   - SWE: 8 workers + 15s stagger。
   - provider quota、Weave Agents trace、NeMoClaw/OpenClaw session 起動負荷の観点から過剰でないか確認してほしい。

4. **timeout/retry 方針の整合性**
   - Agentic Math/SWE の outer OpenClaw timeout は `time_up` として scoreable failure にしており、transient retry 対象から外している。
   - BFCL は Taiwan base config で `case_timeout_retries=0` にした。
   - 一方、provider request level の backoff/retry は残している。
   - 「評価基盤エラーは retry、モデルが時間内に終わらない timeout は不正解」という方針に対して、実装上の境界が妥当か確認してほしい。

5. **W&B/Weave 証跡の追跡性**
   - runner summary と generation log に worker/limiter 設定を残すようにした。
   - output table には既存の Agents conversation URL が入る前提。
   - full run 後に「この run がどの並列度・制限値で実行されたか」を十分追えるか確認してほしい。

6. **paid run 前の追加 gate**
   - 今回は paid API full run は実行していない。
   - 8 workers/4 BFCL threads を本番投入する前に、OpenRouter GLM-5.2 で小規模 probe を行うべきか、あるいはすぐ full を再開してよいか判断材料がほしい。

## 既知の懸念・未解決

- provider limiter は process-local。複数プロセスで同じ provider/model を同時に回す場合、プロセス間では共有されない。
- provider limiter は request 開始間隔のみ制御する。token/sec や concurrent in-flight request 数の直接制御ではない。
- OpenRouter upstream provider が途中で切り替わる場合、`provider_rate_limit_key` は model slug 単位なので provider 実体単位ではない。
- SWE-Bench Pro 8 workers は repo checkout / Docker / disk I/O / OpenClaw session 負荷が高い可能性がある。DGX Station V100 のローカルリソース上限を要確認。
- Math の過去実測では timeout attempt が long tail の主因だった。`openclaw_timeout=900` が品質と時間のバランスとして適切かは full/near-full 再実行で再評価が必要。
- BFCL の `provider_min_request_interval_sec=2.0` は burst 回避優先の初期値。provider error がまだ出る場合は 3-5秒へ上げる、出ないが遅すぎる場合は 1秒へ下げる余地がある。

## Fable への依頼

上記変更について、特に「有料 full run 前にまだ危険な配線漏れ・制御漏れが残っていないか」を重点的に監査してください。
コード上の指摘は file:line 付きでお願いします。結論として、次に取るべき行動を以下のいずれかで明示してください。

- A. このまま GLM-5.2 full run を再開してよい。
- B. 小規模 probe を1回挟むべき。
- C. 追加修正が必要で、full/probe の前に止めるべき。
