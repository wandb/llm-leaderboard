# Fable 監査結果: GLM-5.2 full 評価 BFCL 停止問題 — 原因究明と Codex 向け修正指示

作成日: 2026-07-09 JST
監査者: Claude Fable 5
対象依頼書: `docs/fable_review_glm52_bfcl_stall_request.md`

## 結論サマリ

1. **デッドロックではない。** 現行の `wait_for` + backoff + openai SDK + semaphore のスタックは、ローカル実証テスト(応答しないHTTPサーバ / 429ストーム)で**指定秒ちょうどに cancel が完了する**ことを確認した。既に入れた修正1-3(bounded concurrency / case timeout / async sleep)は正しく機能しており、維持してよい。
2. **主因は「provider 側 stall × 実効タイムアウト構成の欠陥 × 可観測性ゼロ」の複合。** `request_timeout_sec=300` は HTTP リクエストに一切効いておらず、実効 read timeout は GLM config の `openai.http_timeout.read: 1800` 秒。provider(2225 ログで Novita / z-ai upstream rate-limit 429 を実証)が無応答化すると、1 case が最悪 600 秒(wait_for 300s × 2 attempts)拘束され、**その間ログが 1 行も出ない**。
3. **各 run は、最初の可視イベントが出る前に手動 kill された可能性が高い。** 2300 run の時系列: stall ≈22:55:30 → 最初の timeout print 期待時刻 ≈23:00:25(attempt 開始基準)→ 最初の pbar tick(case 記録)は ≈stall+600s ≈23:05。log mtime は 22:55 のまま。`PYTHONUNBUFFERED=1` は設定済み(`run_taiwan_full_eval_batch.py:2118`)なのでバッファリングでは説明できず、「発火前に kill」が最も整合する。断定はできないため、下記 watchdog を入れて次回確定させる。
4. 修正なしで再実行しても、provider stall 時のスループットは最悪 2 slot × 600s/case ≈ 12 case/時、残 337 case ≈ 28 時間の空焼きになる。**修正後に再開すべき(今すぐの再開は非推奨)。**

## 1. Findings(重大度順)

### F1 [Critical] `request_timeout_sec` が provider request に届いていない

- `scripts/evaluator/bfcl.py:46` の `request_timeout_sec: 300` は、`scripts/evaluator/evaluate_utils/bfcl_pkg/bfcl/_llm_response_generation.py:70-79` の `_effective_case_timeout_sec` で `min(600, 300)=300` として **case timeout に転用されるだけ**。HTTP 層へ渡す経路が存在しない。
- 実際の per-request timeout は `scripts/llm_inference_adapter.py:705-717` の `OpenAIClient.__init__` → `_resolve_http_timeout_from_cfg`(:443-485)→ GLM config `configs/taiwan_full/generated_glm52_actualusage_20260709/config-taiwan-full-glm-5_2-openrouter-reasoning.yaml:75-79` の **read: 1800 秒**。さらに `AsyncOpenAI(max_retries=3)` なので、無応答 request は SDK 単体で最悪 1800×4 秒エラーにならない。
- 帰結: 「1 リクエスト 300 秒で切る」という設計意図が、実際には「1 case 全体(multi-turn なら全 turn×全 step 合算)を 300 秒で cancel」になっている。`case_timeout_sec: 600` は **dead config**。
- なぜ full run 停止につながるか: provider が無応答化した瞬間、エラーも進捗も出ない 300-600 秒の沈黙窓が case ごとに発生する。

### F2 [Critical] Provider 側 stall(実証あり)+ 再実行のたびに悪化する構造

- `outputs/.../full_run_glm52_20260709T2225.log:312` に決定的証拠:
  `openai.RateLimitError: 429 ... 'z-ai/glm-5.2 is temporarily rate-limited upstream. Please retry shortly, or add your own key...' provider_name: Novita`
- 2225 run(旧 gather 版)の末尾では `multi_turn_base` 5 case が Turn 0, Step 1 の応答待ちのまま沈黙 = **エラーではなく無応答型の stall**。
- 停止位置の推移 195 → 32 → 13 → 11 は、**fresh run のたびに同じ先頭 case 群を再課金で再実行して quota を再消費**し、上流レート制限に早く到達する構造と整合する。GLM config は `bfcl.allow_overwrite: false` だが result_dir が run ごとに独立のため、run をまたいだ resume が効いていない。

### F3 [High] リトライ4層スタックの矛盾(economics 破壊)

- 層: openai SDK `max_retries=3`(:716)× `LLMAsyncProcessor._ainvoke` backoff `max_tries=50 / max_time=1800`(`llm_async_processor.py:73-91`、429/timeout/ConnectionError を retry)× BFCL case timeout retry(2 attempts)× BFCL 429 65 秒 sleep ループ(`_llm_response_generation.py:343-351`、**backoff が先に 429 を消費するため async では実質 dead code**)。
- 帰結: 429 ストーム下では各 case が wait_for に殺されるまで 300 秒間 silent に retry し続け、`bfcl_case_timeout`(=不正解)として記録される。**run は「完走」するが、スコアはゴミになる**(それはそれで release gate 的に危険)。

### F4 [High] 可観測性ゼロ + timeout 結果の下流ゲート欠如

- in-flight case の定期status出力がなく、timeout print(`_llm_response_generation.py:318-338`)にも timestamp がない。stall とデッドロックを 10 分間区別できず、**今回 4 run 分の課金がこの観測不能性だけで無駄になった**。
- `bfcl_case_timeout` は結果ファイルに最終結果として書かれ、`collect_test_cases`(:173-197)は id 一致だけで「生成済み」とみなすため **resume で修復されない**。また BFCL 集計・`verify_taiwan_wandb_completion.py` に timeout/error 件数のゲートがなく、timeout まみれの run が「valid な W&B completion」に見えてしまう。

### F5 [Medium] timeout retry が multi-turn case を先頭から全再実行する

- `async_inference`(:313-324)の retry は case 全体を Turn 0 からやり直す。multi-turn(最大 20 step × N turn、各 step が 32768 max_tokens + reasoning の API 呼び出し)では **完了済み step の token を全額再課金**した上で再び 300 秒で死ぬ公算が高い。

### F6 [Low] weave の CancelledError 未処理(call context リーク)

- `weave/trace/op.py:652-661`(`_call_async_func`)は `except Exception` と `except (SystemExit, KeyboardInterrupt)` のみで、`asyncio.CancelledError` 時に `finish()` も `call_context.pop_call` も走らない。**ブロックはしない**(cancel は素通りする)が、以後の trace が stale call の子にぶら下がり、Weave Agents 証跡の品質を汚す。F1 の per-request timeout 化で cancel 自体が稀になるため実害は縮小する。upstream issue 候補。

### 参考: 棄却した仮説

- **仮説A(HTTP timeout 1800s)**: 半分正解。read=1800 は事実だが、外側 wait_for の cancel は機能するため「停止」の直接原因ではなく F1/F3 の economics 問題。
- **仮説B(backoff が cancel を飲む)**: 棄却。実証テストで backoff 中の cancel も指定秒で完了。CancelledError は BaseException であり backoff/`error_handler`(Exception のみ)に捕まらない。
- **仮説C(348 task 事前生成)**: 実害なし。semaphore で in-flight は 2 に制限済み。producer/consumer 化は優先度低。
- **仮説D(GLM parser 不整合)**: 主因としては棄却(2225 で 195 case 完走)。ただし `_parse_query_response_FC` の bare `except:`(`openai_compatible_handler.py:160`)は握り潰しなので、ついでに `except Exception` に直すこと。

## 2. 主因判定

| 判定 | 内容 |
| --- | --- |
| 確定 | `request_timeout_sec` は HTTP request に効いていない(F1)。実効 read timeout 1800s + SDK 3 retries。`case_timeout_sec=600` は dead config |
| 確定 | OpenRouter 上流(Novita / z-ai)のレート制限・無応答 stall が発生していた(2225 ログの 429 + 無応答 multi_turn 5 件) |
| 高確度 | 「300 秒過ぎても timeout ログが出ない」は、最初の可視イベント(attempt 開始+300s の print、stall+600s の pbar tick)より前に kill したため。デッドロックの証拠はコード監査・実証テストとも皆無 |
| 仮説(要計測) | 本番のみで cancel が遅延する未知要因。watchdog(P1-4)導入後の次回 run で確定させる |

## 3. Codex への修正指示

### Phase 1: コード修正(再実行前に必須)

**P1-1. per-request timeout の配線(最小修正・最優先)**

- `scripts/llm_inference_adapter.py` `OpenAIClient.invoke/ainvoke`: kwargs の `timeout` を `chat.completions.create(..., timeout=...)` へ明示的に渡す(openai 1.99.1 で per-request `timeout` 対応確認済み)。`filter_params` で落とさないよう、`allowed_params` 追加ではなく `all_kwargs.pop("timeout", None)` → params へ直接入れる実装を推奨(他 client クラスへの波及が安全)。
- 同 kwargs の `request_max_retries` を `self.async_client.with_options(max_retries=...)` 経由で反映(create の kwarg では渡せない)。BFCL 経路は `max_retries=1` を推奨。
- `scripts/evaluator/evaluate_utils/bfcl_pkg/bfcl/model_handler/openai_compatible_handler.py` `__init__`: `cfg.bfcl.request_timeout_sec` を保持し、`_query_FC_async` / `_query_prompting_async`(および sync 版)の kwargs に `timeout=` を追加。

**P1-2. `LLMAsyncProcessor` の retry policy を per-instance 化**

- `scripts/evaluator/evaluate_utils/llm_async_processor.py`: クラス定義時デコレータをやめ、`__init__(..., backoff_max_time=None, backoff_max_tries=None)` を受けて `self._ainvoke = backoff.on_exception(...)( self._ainvoke_impl )` を構築。デフォルトは現行値(1800/50)を維持し**他ベンチの挙動を変えない**。
- `OpenAICompatibleHandler.__init__` は `LLMAsyncProcessor(llm, backoff_max_time=90, backoff_max_tries=4)` 程度で生成(request timeout 300s の内側で完結する値)。

**P1-3. timeout 階層の修正**

- `_llm_response_generation.py` `_effective_case_timeout_sec`: `min()` を廃止し、case timeout = `case_timeout_sec`(600)とする。per-request 300s は P1-1 が担う。single-turn は実質 request timeout 支配、multi-turn は 600s のケース上限になる。
- case timeout 時の全ケース再実行 retry は single-turn のみに限定し、multi-turn は即 `bfcl_case_timeout` 記録(F5 の再課金防止)。`case_timeout_retries` の意味は「single-turn の transient hang 救済」と README/コメントに明記。

**P1-4. watchdog + timestamp(観測性)**

- `async_generate_results` に 30 秒周期の status task を追加: `[BFCL watchdog 23:00:15] in-flight: irrelevance_12 (312s, attempt 2/2), java_3 (45s, attempt 1/2); done 11/348, timeouts 0` 形式で `print(..., flush=True)`。
- 既存の timeout/retry/429 print 全てに wall-clock timestamp を付ける。

**P1-5. stall/429 の経済的ガード(fail fast)**

- backoff give-up で `RateLimitError` が上がってきたら global cooldown(`asyncio.Event` で新規 case 投入を 60s→指数、上限 600s 停止。in-flight は継続)。
- フェーズ中断条件を追加: (a) 直近 15 分の完了 case 数 < 5、または (b) 連続 5 case が timeout/rate-limit で終了 → BFCL を例外 `BFCLStalledError` で即中断し、run 全体を明示的に fail させる(28 時間の空焼き・ゴミスコア完走の両方を防ぐ)。閾値は `bfcl` config に出す。

**P1-6. timeout/error の採点ゲートと resume 修復**

- BFCL 集計(`scripts/evaluator/bfcl.py`)の leaderboard/W&B table に `timeout_count` / `inference_error_count` 列を追加。
- `scripts/tools/verify_taiwan_wandb_completion.py` の bfcl 検証: `timeout_count > 0` で warning、`> 3`(≈1%)で hard fail。**release evidence 採用は 0 のみ**。
- `collect_test_cases`: 既存結果のうち `error == "bfcl_case_timeout"`(および `Error during inference:` 結果)のエントリを「未生成」として再実行対象にするフラグ `retry_failed_cases`(default true)を追加 → 完走後に timeout 分だけ resume で安く修復できる。

**P1-7. ついで修正**

- `openai_compatible_handler.py:160` の bare `except:` → `except Exception:`。
- weave の CancelledError リークは修正不能(サードパーティ)。`docs/taiwan_launch_progress.md` に既知制約として記録し、Weave trace 検証で「case timeout が発生した run の trace 階層は乱れ得る」ことを注記。

### Phase 2: provider 対策(再実行前に実施)

**P2-1. OpenRouter probe script(新規 `scripts/tools/probe_openrouter_model.py`)**

- 対象 model に tools 付き小 prompt を 12-20 件(multi_turn 相当 1 件含む)投げ、response の provider 名ごとに p50/p95 latency、429 率、60 秒超 hang 率を JSON 出力。総コスト目安 <$0.5。
- `run_taiwan_full_eval_batch.py` の paid full/agentic 実行前ゲートに probe report(閾値: 429 率 <10%、p95 <120s)を任意フラグで追加。

**P2-2. provider ピン止め / BYOK**

- probe 結果に基づき GLM config の `generator.extra_body.provider` に `{"ignore": [...], "allow_fallbacks": true}` 等を設定(Novita が再現的に劣化しているなら ignore)。
- OpenRouter に z-ai の BYOK キー追加(429 メッセージの推奨どおり)。これは外部アクション承認プロセスの対象として扱うこと。

**P2-3. デバッグ反復の課金防止**

- 修正検証用の run は result_dir を run 間で共有し resume(`allow_overwrite: false` を活かす)。**fresh no-resume は最終 evidence run のみ**。この方針を `docs/taiwan_launch_progress.md` の運用ルールに明記。

### 避けるべき修正

- **threaded path への全面移行**: 問題(HTTP timeout 欠如)は共通で、thread は cancel 不能なぶん悪化する。現行 threaded 実装の「executor ごと作り直し」(`_llm_response_generation.py:495-499`)はまさにその症状。async を直すのが正。
- **`asyncio.shield` や producer/consumer 全面改修**: cancel は実証上機能しており、根本原因ではない。
- **`MAX_TRIES`/`max_time` のグローバル短縮**: mtbench judge 等、他ベンチの長時間耐障害リトライを壊す。per-instance 化(P1-2)で解決する。

## 4. 再実行前の確認項目

### Unit tests(`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q` で確認)

1. `OpenAIClient.ainvoke` が `timeout` kwarg を fake AsyncOpenAI の `create` まで届ける。
2. `LLMAsyncProcessor(backoff_max_time=2)` + 常時 `APITimeoutError` の fake llm → 2-3 秒以内に give-up して例外が上がる。
3. `async_inference` + 永遠に pending の fake handler → `case_timeout_sec` で `bfcl_case_timeout` を記録し次 case へ進む(既存 `tests/test_bfcl_generation_resume.py` の拡張)。
4. 連続 RateLimitError fake → cooldown 発動、閾値超で `BFCLStalledError`。
5. `collect_test_cases(retry_failed_cases=True)` が timeout エントリを再生成対象に含める。

### Integration probe(mock でない実証)

6. 今回の監査で使った再現手法を恒久化: ローカル HTTP サーバ(①accept 後無応答 ②429 連発)+ 実 `OpenAIClient`→`LLMAsyncProcessor`→`async_inference` フルスタックで、`request_timeout_sec` 秒 +α で必ず先へ進むことを検証(`tests/test_bfcl_timeout_probe.py`。ネットワーク不要・数秒で完走)。監査時の実測: silent-hang / 429 とも wait_for 指定秒ちょうどで cancel 完了。
7. OpenRouter 実 probe(P2-1)を full run 直前に実行し、report を run ディレクトリに残す。

### W&B run lifecycle

8. `scripts/run_eval.py`: ベンチ実行部を try/finally 化し、例外・SIGINT/SIGTERM で `run.finish(exit_code=1)` を保証(現状 kill すると `running` が残る)。
9. `run_taiwan_full_eval_batch.py`: 子プロセス非 0 終了時に該当 run へ `invalid_stopped` tag + notes(停止理由・停止位置)を付ける後始末 step を追加。completion verifier が `finished` 以外を拒否することは既存確認済みなので、tag は人間の視認性向上目的。

## 5. full run 再開可否

**修正後に再開可(今すぐは非推奨)。**

再開手順: P1 unit tests(1-5)→ ローカル probe(6)→ OpenRouter probe(7、数十円)→ probe pass なら full run。probe fail(429 率・hang 率が閾値超)なら P2-2(BYOK / provider ピン止め)を先に解決する。full run 中は watchdog ログで in-flight を常時観測でき、stall しても P1-5 が 15 分以内に自動 fail するため、今回のような「見えない空焼き → 手動 kill → invalid run 蓄積」は再発しない。

## 依頼書の質問への直接回答

1. **300 秒で戻らない最有力コードパス**: 「戻っていなかった」証拠はない。attempt 開始 +300s の print・+600s の記録より先に kill された可能性が最も高い(時系列は結論サマリ 3)。実証テストでは全経路で cancel が機能。次回 watchdog で確定させる。
2. **`request_timeout_sec=300` は効いているか**: 効いていない(F1)。`_effective_case_timeout_sec` の min() に消費されるだけで、HTTP 層は config の read=1800s。
3. **backoff `max_time=1800` は矛盾するか**: する(F3)。ただし cancel は貫通するためデッドロックではなく、「300 秒間 silent に retry してから case を不正解で捨てる」という経済・品質問題。P1-2 で per-instance 化して整合させる。
4. **threaded に寄せるべきか**: 否。async を直す(「避けるべき修正」参照)。
5. **timeout=fail 記録の許容範囲**: 継続記録自体は正しい設計。ただし gate 必須 — canary: >0 warning / >3 hard fail、release evidence: 0 のみ許容。resume 修復(P1-6)で 0 に収束させてから採用する。
6. **必須 regression test**: 上記 4 節 1-7。特に 6(ローカル hang server による実 HTTP フルスタック検証)は mock でない実証として毎回 CI で回せる。
7. **run lifecycle の後始末**: 上記 4 節 8-9。
