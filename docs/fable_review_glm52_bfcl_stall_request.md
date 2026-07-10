# Claude Fable 監査依頼書: GLM-5.2 full 評価が BFCL で停止する問題

作成日: 2026-07-09 JST

## 依頼の目的

台湾版 LLM リーダーボードの full 評価で、GLM-5.2 (`openrouter-direct/z-ai/glm-5.2`) が BFCL の途中で繰り返し停止しています。

現状、W&B run は複数作られていますが、いずれも full 評価の証跡としては invalid です。理由は「評価品質が悪い」ではなく、BFCL が完走せず、Total Score まで到達していないためです。

Fable には、以下を監査してほしいです。

1. 現在の停止原因の主因をコード上で特定する。
2. 既に入れた BFCL timeout/concurrency 修正が十分か、不十分ならどこが不十分か指摘する。
3. 次に入れるべき修正を、再実行コストを無駄にしない順序で提案する。
4. full 評価を再開できる判定条件を明確にする。

## 背景

目的は、台湾版リーダーボードの正式リリース前に、1モデル full 評価を一度通して Total Score、W&B Table、Weave Agents trace、コスト実測を確認することです。

今回の対象モデルは GLM-5.2 です。OpenRouter 経由で使用しています。

- Model id in config: `z-ai/glm-5.2`
- Runtime model alias: `openrouter-direct/z-ai/glm-5.2`
- BFCL model handler: `OpenRouter-FC`
- Full run config: `configs/taiwan_full/generated_glm52_actualusage_20260709/config-taiwan-full-glm-5_2-openrouter-reasoning.yaml`
- BFCL default limits in `scripts/evaluator/bfcl.py`
  - `request_timeout_sec: 300`
  - `case_timeout_sec: 600`
  - `case_timeout_retries: 1`

## 直近の run 状況

以下はいずれも full 評価を意図した fresh no-resume run ですが、BFCL で停止したため production evidence としては invalid です。

| Run id | 状態 | 停止位置 | 主な観察 |
| --- | --- | --- | --- |
| `twfull-glm52-actualusage-20260709-2225-glm-5_2-openrouter-reasoning` | stopped/invalid | BFCL 約195/348 | OpenRouter async path が `num_threads` と `case_timeout_sec` を実質無視し、全件 `asyncio.gather` で投入していたことを確認。 |
| `twfull-glm52-actualusage-20260709-2240-glm-5_2-openrouter-reasoning` | stopped/invalid | BFCL 32/348 | bounded concurrency 修正後も、単一リクエスト待ちで数分進まない状況を確認。 |
| `twfull-glm52-actualusage-20260709-2250-glm-5_2-openrouter-reasoning` | stopped/invalid | BFCL 13/348 | `effective_case_timeout_sec=min(case_timeout_sec, request_timeout_sec)=300` 相当の修正後も停止。 |
| `twfull-glm52-actualusage-20260709-2300-glm-5_2-openrouter-reasoning` | stopped/invalid; W&B state may still appear `running` temporarily | BFCL 11/348 | ログ上は `BFCL async generation limits: max_concurrency=2, effective_case_timeout_sec=300.0, case_timeout_retries=1` と出たが、300秒を超えても次へ進まなかった。 |

注意: W&B UI では同名 run が複数あるように見えたが、実体は run id が異なる。旧設定の `wandb.run_name` がモデル名ベースで、時刻/run id を含まなかったため視認性が悪かった。ローカルプロセスは停止確認済み。

## 期待される挙動

BFCL の各 test case について、以下が成立してほしいです。

1. 同時実行数は `num_threads` で上限管理される。
2. 1 case が `effective_case_timeout_sec` を超えたら、その case を retry または timeout result として記録し、次の case に進む。
3. timeout した case は `bfcl_case_timeout` として W&B/log/local result に残る。
4. 一部 case の provider hang によって、348件全体が止まらない。
5. full run は BFCL 後の Agentic Math / SWE-Bench Pro / その他ベンチへ進む。

## 既に入れた修正

### 1. BFCL async path の bounded concurrency

対象ファイル:

- `scripts/evaluator/evaluate_utils/bfcl_pkg/bfcl/_llm_response_generation.py`

主な変更:

- 旧実装の `asyncio.gather(*all_tasks)` をやめた。
- `asyncio.Semaphore(max_concurrency)` を使い、`max_concurrency=max(1,args.num_threads)` にした。
- `asyncio.as_completed(tasks)` で完了した case から逐次 `handler.write(..., update_mode=True)` するようにした。

現在の該当コード:

- `async_generate_results`: lines 379-413
- `async_inference`: lines 283-377

### 2. BFCL async case timeout

対象ファイル:

- `scripts/evaluator/evaluate_utils/bfcl_pkg/bfcl/_llm_response_generation.py`

主な変更:

- `handler.inference_async(...)` を `asyncio.wait_for(..., timeout=case_timeout_sec)` で包んだ。
- timeout 時は `case_timeout_retries` 回だけ retry。
- retry を使い切ったら `bfcl_case_timeout` result を作って返す。
- `request_timeout_sec` と `case_timeout_sec` の両方がある場合、effective case timeout は短い方にした。

現在の該当コード:

- `_effective_case_timeout_sec`: lines 70-79
- `async_inference`: lines 283-377

### 3. async retry の blocking sleep 修正

対象ファイル:

- `scripts/evaluator/evaluate_utils/bfcl_pkg/bfcl/_llm_response_generation.py`

主な変更:

- async retry path の `time.sleep(RETRY_DELAY)` を `await asyncio.sleep(RETRY_DELAY)` に変更。

### 4. Regression test

対象ファイル:

- `tests/test_bfcl_generation_resume.py`

確認済み:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_bfcl_generation_resume.py
```

結果:

```text
6 passed
```

また以下の py_compile は通過済み:

```bash
python3 -m py_compile scripts/evaluator/evaluate_utils/bfcl_pkg/bfcl/_llm_response_generation.py
```

## それでも残っている症状

最新 run `2300` のログ抜粋:

```text
Generating results for OpenRouter-FC
Running full test cases for categories: ['irrelevance', 'java', 'javascript', 'live_irrelevance', 'live_multiple', 'live_relevance', 'live_simple', 'multi_turn_base', 'multi_turn_miss_func', 'multi_turn_miss_param', 'multiple', 'simple'].
BFCL async generation limits: max_concurrency=2, effective_case_timeout_sec=300.0, case_timeout_retries=1

Generating results for OpenRouter-FC:   0%|          | 0/348 [00:00<?, ?it/s]
Generating results for OpenRouter-FC:   0%|          | 1/348 [00:03<19:55,  3.44s/it]
Generating results for OpenRouter-FC:   1%|          | 2/348 [00:19<1:02:08, 10.78s/it]
...
Generating results for OpenRouter-FC:   3%|▎         | 11/348 [01:25<43:19,  7.71s/it]
```

この後、300秒を超えても `BFCL async case timed out` のログが出ず、進捗も止まった。手動停止したため full 評価として invalid。

## 現時点の仮説

### 仮説A: `asyncio.wait_for` の外側ではなく、内側の OpenAI/OpenRouter HTTP timeout が長すぎる

BFCL の OpenRouter handler は以下を使っている。

- `scripts/evaluator/evaluate_utils/bfcl_pkg/bfcl/model_handler/openai_compatible_handler.py`
- `_query_FC_async` lines 113-131
- `self.llm_ap.process_single_async(message, **kwargs)` に渡している。

しかし、`kwargs` には現在 `timeout` が入っていない。

一方、`scripts/llm_inference_adapter.py` の `OpenAIClient` は config から HTTP timeout を作るが、GLM config 側の `openai.http_timeout.read` が長く、実効 read timeout が 1800 秒になっている可能性がある。

さらに `OpenAIClient.allowed_params` に `timeout` が無い場合、仮に BFCL handler から `timeout` を渡してもフィルタで落ちる可能性がある。

確認してほしい点:

- `OpenAICompatibleHandler._query_FC_async` に `timeout=request_timeout_sec` を入れるべきか。
- `OpenAIClient.allowed_params` に `timeout` を追加すべきか。
- OpenAI Python client の chat completions create に per-request timeout が正しく渡るか。

### 仮説B: `LLMAsyncProcessor._ainvoke` の `backoff` が timeout を30分まで引き伸ばしている

対象ファイル:

- `scripts/evaluator/evaluate_utils/llm_async_processor.py`

該当:

```python
@backoff.on_exception(..., max_tries=50, max_time=1800, ...)
async def _ainvoke(...)
```

ここで `openai.APITimeoutError`, `TimeoutError`, `ConnectionError` などを retry 対象にしている。

仮に HTTP timeout を 300 秒にしても、`_ainvoke` が `APITimeoutError` を捕まえて最大 1800 秒 backoff retry し続けるなら、BFCL の case timeout と矛盾する。

確認してほしい点:

- `asyncio.wait_for` が `_ainvoke` の backoff 中でも確実に cancel できるか。
- `backoff` デコレータが `asyncio.CancelledError` を飲み込まないか。
- BFCL 経路だけ `max_time` / `max_tries` を短くする、または BFCL case timeout を最上位に置くべきか。

### 仮説C: 348 tasks を先に作っている設計が cancellation / visibility を悪くしている

現行修正では semaphore で active concurrency は制限しているが、`tasks = [asyncio.create_task(run_one(test_case)) for test_case in test_cases_total]` により、348個の asyncio task 自体は最初に作っている。

active HTTP request は semaphore で最大2のはずだが、監査観点では以下を確認してほしい。

- task を全件作る設計で問題ないか。
- producer/consumer 型にして「未投入 case は task 化しない」方が安全か。
- 停止時に pending task / HTTP request が残る可能性があるか。

### 仮説D: OpenRouter / GLM-5.2 特有の function-calling 応答形式が parser/backoff を誘発している

GLM-5.2 だけ BFCL で同じように詰まる。gpt-4.1-mini では少なくとも同じ形の停止は観測されていない、という比較観点がある。

確認してほしい点:

- GLM-5.2 の tool call response が `OpenAICompatibleHandler._parse_query_response_FC` と整合しているか。
- `content=None`, `tool_calls`あり、`reasoning`あり等のケースで parser が例外を出して backoff retry を誘発していないか。
- `reasoning_details` / `reasoning` を次 turn chat history に入れる処理が OpenRouter/GLM の tool-call continuity に悪影響を出していないか。

## Fable に特に見てほしいファイル

優先順:

1. `scripts/evaluator/evaluate_utils/bfcl_pkg/bfcl/_llm_response_generation.py`
2. `scripts/evaluator/evaluate_utils/bfcl_pkg/bfcl/model_handler/openai_compatible_handler.py`
3. `scripts/evaluator/evaluate_utils/llm_async_processor.py`
4. `scripts/llm_inference_adapter.py`
5. `scripts/evaluator/bfcl.py`
6. `tests/test_bfcl_generation_resume.py`
7. `docs/taiwan_launch_progress.md`

関連ログ:

1. `outputs/taiwan_full_eval_glm52_actualusage_20260709/full_run_glm52_20260709T2225.log`
2. `outputs/taiwan_full_eval_glm52_actualusage_20260709/full_run_glm52_20260709T2240.log`
3. `outputs/taiwan_full_eval_glm52_actualusage_20260709/full_run_glm52_20260709T2250.log`
4. `outputs/taiwan_full_eval_glm52_actualusage_20260709/full_run_glm52_20260709T2300.log`

## 監査への具体的な質問

1. `asyncio.wait_for(handler.inference_async(...), timeout=300)` が 300秒で戻らない原因として、最も疑わしいコードパスはどこか。
2. BFCL の `request_timeout_sec=300` は現状、本当に OpenRouter API request に効いているか。効いていないなら、どこで失われているか。
3. `LLMAsyncProcessor._ainvoke` の `backoff max_time=1800` は BFCL の per-case timeout と矛盾しているか。
4. OpenRouter-FC / GLM-5.2 に対して、async path をやめて threaded path に寄せる方が安全か。それとも async path を正しく直すべきか。
5. timeout した case を fail として記録し継続する設計は、BFCL の leaderboard quality として許容できるか。許容する場合、何件までを warning / hard fail にすべきか。
6. 次の fresh full run 前に必須とすべき regression test は何か。単なる mock test ではなく、実際の OpenRouter timeout/hang を近似できるテスト案がほしい。
7. W&B/Weave 上で「停止・invalid・stale running」が混ざらないようにするため、run lifecycle の後始末で追加すべき処理は何か。

## 現時点で考えている次の修正案

以下はまだ確定ではない。Fable に妥当性を見てほしい。

### 案1: BFCL OpenRouter handler から per-request timeout を明示的に渡す

`OpenAICompatibleHandler` で `cfg.bfcl.request_timeout_sec` を読み、`_query_FC` と `_query_FC_async` の `kwargs` に `timeout` を入れる。

あわせて `OpenAIClient.allowed_params` に `timeout` を追加し、OpenAI Python client まで落ちないようにする。

### 案2: BFCL 経路では `LLMAsyncProcessor` の backoff 上限を BFCL timeout と整合させる

例えば BFCL handler 生成時に `LLMAsyncProcessor(..., max_time=request_timeout_sec)` 相当を指定できるようにする、または BFCL 用の retry policy を分ける。

### 案3: async task を全件先に作らない

348件を `create_task` で一括作成せず、最大 `num_threads` 件だけ投入し、完了したら次を投入する producer/consumer 型に変える。

### 案4: timeout 実証テストを追加する

mock ではなく、以下のような実証テストを追加する。

- `OpenAIClient.ainvoke` に per-request `timeout` が渡ることを fake client で検証。
- `OpenAICompatibleHandler._query_FC_async` が `cfg.bfcl.request_timeout_sec` を `timeout` に変換することを検証。
- `LLMAsyncProcessor._ainvoke` が `asyncio.CancelledError` を飲み込まないことを検証。
- long-running fake async request が BFCL `effective_case_timeout_sec` で `bfcl_case_timeout` になることを検証。

## Fable への期待アウトプット

以下の形式で回答してほしいです。

1. 重大度順の findings
   - file/line 参照つき
   - なぜ full run 停止につながるか
2. 主因の判定
   - 確定 / 高確度 / 仮説 を区別
3. 推奨修正
   - 最小修正
   - より堅牢な修正
   - 避けるべき修正
4. 再実行前の確認項目
   - unit test
   - integration/probe
   - W&B run lifecycle
5. full run 再開可否
   - 今すぐ再開可 / 修正後に再開可 / 追加調査が必要
