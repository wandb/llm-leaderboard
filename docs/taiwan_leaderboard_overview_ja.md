# 台湾版リーダーボード 全体説明

最終更新: 2026-07-01 13:37 JST

## まず結論

台湾版リーダーボードは、まだ本番公開できる状態ではありません。

ただし、評価データ、評価runner、W&B/Weave検証、release gate、evidence bundleはかなり実装済みです。残っている中心課題は「1モデルで全ベンチを本番同等に完走し、その結果がW&B/Weave上で監査可能に残り、費用レビューとTotal Scoreまで揃うこと」です。

最新の機械判定は `temp/taiwan_release_gate_20260701T043635Z.json` です。状態は `not_ready`、bundle整合性はOK、checked filesは606、verification errorsは0です。つまり「証跡bundleそのものは壊れていないが、本番公開条件はまだ満たしていない」という状態です。現在の残ブロッカーは `weave_content_canary`、`wandb_completion`、`paid_run_review_package`、`one_model_full_canary` の4つです。

次に実行する1モデルcanaryはOpenAI-direct経路に固定しています。operator shell handoffは、`--canary` のbatch commandがOpenAI-direct canary manifest/generated config dirsから外れたり、openrouter/anthropic/claude/gemini/opus/sonnet系の文字列を含んだり、承認済み `paid_api.approved_model_scope` が `openai-direct/gpt-4.1-mini` を含まなかったりすると、shell生成前に拒否します。release evidence bundleのoperator renderer metadataとMarkdownにも、同じcanary approval-scope safety flagを出して検証するようにしました。

NeMoClawについては、ローカルruntime/sandbox/native weave readinessはrelease gate上で通っており、`nemoclaw_readiness` は現在のブロッカーから外れています。Agentic Math/SWE向けには、NeMoClaw config、task-agent実行経路、SWE-Bench Pro copy-mode checkout transfer、post-install/adoption/canary readiness、operator command policy、release bundle verifierまで実装済みです。さらに、OpenClaw session JSONLで「問題提示前のtool実行」や「最終回答後のtool実行」を検出してnon-scoreableにするガードを入れ、そのrunner source自体もrelease evidence bundleに含めてsource-contract検証するようにしました。

一方で、まだ本番公開はできません。実ベンチ問題でNeMoClaw + native Weave traceを作り、W&B completion、paid-run review、1モデルfull canary、Total Scoreまで揃える必要があります。既存run `f1veetyb` はAgentic Mathの古いW&B completion候補でしたが、現在のNeMoClaw session-audit契約を満たさないため正式採用しません。対応するローカルDeepSeek完走結果は、summary/results SHA-256付きの `outputs/taiwan_full_eval/existing_results_archive_manifest.json` により `archived_not_release_candidate` として保存し、release候補から外しています。

## 30秒版

このプロジェクトは「台湾市場向けLLMリーダーボード」を作る作業です。

単にモデルへ問題を投げて点数を出すだけではなく、公開後に説明責任を果たせるように、評価データ、モデル設定、回答、tool実行、採点、W&B run、Weave Agents trace、費用、請求参照、Total Scoreを一つながりの証跡として残すことを重視しています。

現状は、評価の部品と検証器はかなり揃っています。一方で、本番公開に必要な「安めの1モデルで全ベンチを最後まで実行し、W&B/Weaveに正しいログが残り、費用レビューとTotal Scoreまで揃った」実績がまだありません。

つまり、今の主戦場は新しいベンチを増やすことではなく、次の3点です。

```text
1. Agentic Math / Agentic SWEの実行過程がWeave Agents上で正しい順序で見えること
2. その結果がW&B completionとして正式run/artifactに残ること
3. 1モデルfull canaryを通して、費用、ログ品質、採点バランスをレビューできる状態にすること
```

もう少し具体的に言うと、現在は「評価を実行するコードがあるか」よりも「その実行結果を正式なリーダーボード証跡として採用できるか」を詰めています。W&Bに点数があるだけでは不十分です。Agentic評価では、問題提示、tool実行、tool結果、最終回答がWeave Agents上で正しい順序と内容で見え、そのrun_idとW&B completionとpaid-run reviewが同じ対象を指している必要があります。

直近の既存Agentic Math run `f1veetyb` は、現在のNeMoClaw session-audit契約では正式completionとして採用できません。native W&B Agents API照合でも `conversation_id_contains=f1veetyb` に一致するspanが0件だったため、Agents証跡としても採用できません。また、paid-run reviewへのscope/cost/bill同期も未実行です。したがって、今必要なのは既存runだけで全体を無理に通すことではなく、低コストの1モデルfull canaryを新しく通して、W&B completion、Weave Agents trace、paid-run review、Total Score、費用レビューを一式で揃えることです。

## 最低限の用語

```text
W&B
  Weights & Biases。評価run、metrics、artifact、tableを保存する場所。

Weave
  W&BのLLM trace基盤。モデル呼び出しやtool callを後から追える。

Agentsタブ
  Weave上で、Agentic実行の会話・tool実行・最終回答を見るUI。

Agentic評価
  モデルが1回回答するだけでなく、repoを読む、Pythonを使う、テストを走らせるなど、toolを使って解く評価。

OpenClaw
  Agentic実行を行うための基盤。Weave連携で会話やtool実行を記録したい対象。

NeMoClaw
  NVIDIA側のOpenClaw系ラッパーとして期待している実行基盤。sandboxやNVIDIA連携の説明力があるため、Agentic Mathで採用候補。

release gate
  台湾版リーダーボードを本番公開してよいかを機械的に判定するチェック一式。

evidence bundle
  release gateが判断に使った証跡ファイル一式。後から同じ判断を再検証できるようにまとめたもの。

one-model full canary
  まず1つのモデルだけで全ベンチを本番同等に回す試験。複数モデル展開前の品質確認。
```

## まず何をしているのか

このrepoで作っているのは、台湾向けLLMリーダーボードの「評価を実行し、結果を採点し、正式証跡として保存し、公開してよいかを判定する仕組み」です。

最終的に欲しいものは、単なるCSVやスコア表ではありません。次のことを第三者に説明できる状態です。

```text
どの評価データを使ったか
どのモデルを、どの設定で動かしたか
モデルの回答とtool実行がどのように記録されたか
採点結果がW&Bに正式runとして残っているか
費用がどれくらいかかり、どの請求と対応するか
複数ベンチの結果からTotal Scoreをどう計算したか
その結果を台湾版リーダーボードに載せてよいか
```

今やっている作業は、主に「評価を走らせる前後の証跡条件を本番レベルまで固めること」です。理由は、評価を回して高いスコアが出ても、W&B/Weaveのログ、費用レビュー、run identity、採用判断が曖昧だと、後で正式なリーダーボード結果として使えないためです。

最近の作業領域を平易に言うと、次の4つです。

```text
1. 評価そのもの
   台湾向けの知識、会話、数学、コード修正、関数呼び出し、繁体字品質を測る。

2. Agentic実行
   数学やSWE-Bench Proで、モデルがtoolを使いながら解く過程を正式に記録する。

3. W&B / Weave証跡
   点数だけでなく、回答、tool call、artifact、費用、run identityを後から監査できるようにする。

4. release gate
   本番リーダーボードへ載せてよい状態かを、手作業の雰囲気ではなく機械的に判定する。
```

## 全体の進め方

現在の進め方は次の順序です。

```text
1. 評価ハーネスとデータを用意する
2. W&BとWeaveに、正式結果として必要なログが残る条件を定義する
3. まず安めの1モデルで全ベンチを最後まで回す
4. その結果を見て、採点バランス、費用、ログ品質、市場感とのズレを確認する
5. 問題を直してから、5-10モデル程度に広げる
```

今は2から3へ進む直前です。コードや検証器はかなり揃っていますが、まだ「1モデルで全ベンチを完走し、W&B/Weave/費用レビュー/Total Scoreまで揃った」という状態ではありません。

2026-07-01 13:37 JST時点では、NeMoClawはローカル環境でAgentic Math/SWE評価のsandbox backendとして使える状態に到達しています。`nemoclaw_readiness` は最新release gateでpassedです。評価runner側も、Agentic Math task-agent経路、SWE-Bench Pro copy-mode checkout transfer、OpenClaw session JSONLの会話順序ガード、W&B/Weave証跡の検証入口、release bundleへのrunner source同梱まで実装済みです。ただし、実ベンチ問題をNeMoClaw経由で本番同等に完走し、native Weave trace、W&B completion、paid-run review、official evaluator結果まで揃えた証跡はまだありません。

既存W&B runを正式なpaid-run reviewへ採用するための手順も `operator_handoff` として機械検証可能になっています。ただし、`f1veetyb` は現在のNeMoClaw session-audit証跡を満たさないため、今は正式採用候補ではありません。今後、既存runを採用する場合の必要な流れは、scope attestationの人間確認、confirmed attestationのrender、preflight、dry-run、review JSONへのapplyです。`confirmed_by`、`confirmed_at`、`confirmation`、`completion_sha256`、`actual_cost_estimate`、`provider_bill_reference` など人間所有フィールドが未確定のままでは正式採用しません。最新のrelease evidenceでは、古いDeepSeek Agentic Mathローカル完走は `outputs/taiwan_full_eval/existing_results_archive_manifest.json` としてbundleに含まれ、採用ではなく非release候補として監査対象になっています。

## 現在の仮説と解決アプローチ

現在の仮説は次です。

```text
1. 台湾版リーダーボードの品質は、点数そのものだけでなく、評価runの再現性・監査性で決まる。
2. Agentic評価では、モデル回答、tool実行、最終回答の順序がWeave Agentsで自然に見えない限り、正式結果として採用しにくい。
3. NeMoClawはAgentic Mathのsandbox実行とNVIDIA連携の説明力を高める可能性があるが、まず導入・post-install・canaryの証跡を通す必要がある。
4. 既存W&B runを正式採用する場合も、run_idだけでは不十分で、モデル設定、費用、請求参照、採用scope、人間が確認したscope attestation、validated dry-runのsource review SHA、source audit SHA、source audit内の同一W&B completion recordを機械的に照合できる証跡が必要。
5. 5-10モデルへ広げる前に、安めの1モデルで全ベンチを最後まで通し、ログ品質、費用、Total Score、採点バランスを確認するべき。
```

そのため、解決アプローチは「いきなり大規模に回す」ではなく、次の順にしています。

```text
1. release gateが何を満たせば本番可とするかを機械判定できるようにする。
2. W&B completion、Weave Agents trace、paid-run review、NeMoClaw導入証跡をbundleに入れて再検証する。
3. 未承認・placeholder・別run混入・手編集だけのok=trueを検証器で拒否する。
4. その状態で1モデルfull canaryを実行する。
5. 結果を人間がレビューしてから、複数モデルへ拡張する。
```

今の優先順位は次です。

```text
1. Weave content canaryを、OpenAI-directの小さめモデルでscoreable traceまで通す。
2. NeMoClawをAgentic Math向けに導入・post-install検証し、Agentsタブで順序と内容が読めるか確認する。
3. Agentic SWE / Taiwan fullのW&B completionを作る。
4. paid-run reviewに、実コスト、provider bill参照、W&B completion、Weave Agents completionを同期する。
5. 1モデルfull canaryでTotal Scoreまで出し、スコア分布と市場感をレビューする。
```

実行を増やす前にこの順にしている理由は、途中生成物に高額API費用を使っても、W&B/Weave/費用レビューのどれかが欠けると正式結果として使えないためです。最終盤の本番候補評価では相応の費用を使う前提ですが、今のフェーズでは低コストでログ品質と採用条件を先に証明します。

また、非prepareの実行に進む場合は、外部アクション承認レポートも必要です。これは有料API、W&B write、NeMoClaw install、scope confirmationなどを始める前に、人間がレビュー済みpacketを作り、source-bound verifierで通したJSONです。batch runnerはこの verifier report を `--external-action-approval-report-json` で受け取り、承認packetそのものではなく「承認packetが現在のrelease bundleに紐づいて検証済みであること」を確認します。

## 目下の課題を一言でいうと

最大の課題は、Agentic評価の実行ログを本番で信用できる形にすることです。

Agentic評価では、モデルは単に1回回答するだけではありません。数学ならPythonを使う可能性があり、SWE-Bench Proならrepoを読み、編集し、テストを走らせます。したがって、点数だけでなく、次の順序がWeave Agents上で見える必要があります。

```text
問題が提示される
モデルが考えてtoolを使う
tool実行結果を読む
必要なら追加でtoolを使う
最後に回答する
```

この順序が崩れて見える、あるいはtool実行だけが先に見える、最終回答後にtoolが見える、問題文が見えない、といった状態では、評価結果として採用しにくいです。そのため、Weave Agentsのnative integration、OpenClaw/NeMoClaw実行、W&B completion verifier、release evidence bundle verifierを固めています。

直近では、Agentsログの順序検証をさらに厳しくしました。各spanの開始・終了時刻が解釈可能で、終了時刻が開始時刻より前になっておらず、toolが問題提示と同時刻またはそれ以前に見える曖昧な証跡を正式採用しません。また、最終回答後にtoolが走っていないかは、回答spanの開始時刻ではなく終了時刻を基準に判定します。これは、root spanがtool実行全体を包む場合に誤判定しないためです。

最新の機械判定は次です。

```text
release gate: not_ready
latest gate: temp/taiwan_release_gate_20260701T043635Z.json
latest pointer: temp/latest_taiwan_release_gate.json
latest pointer verification: temp/latest_taiwan_release_gate_verify_20260701T043635Z.json
evidence bundle: outputs/taiwan_release_evidence/bundle_20260701T043635Z
release gate pointer proof: outputs/taiwan_release_evidence/bundle_20260701T043635Z/release_gate_pointer_proof.json
bundle integrity: OK
checked files: 606
verification errors: 0
remaining blockers:
  weave_content_canary
  wandb_completion
  paid_run_review_package
  one_model_full_canary
```

最新の正式判定ファイルは、常に `temp/latest_taiwan_release_gate.json` から辿れます。2026-07-01 13:37 JST時点の実体は次です。

```text
release gate: temp/taiwan_release_gate_20260701T043635Z.json
latest pointer: temp/latest_taiwan_release_gate.json
latest pointer verification: temp/latest_taiwan_release_gate_verify_20260701T043635Z.json
operator plan: temp/taiwan_release_operator_plan_20260701T043635Z.md
evidence bundle: outputs/taiwan_release_evidence/bundle_20260701T043635Z
bundle verification: temp/taiwan_release_evidence_bundle_verify_20260701T043635Z.json
release gate pointer proof: outputs/taiwan_release_evidence/bundle_20260701T043635Z/release_gate_pointer_proof.json
status: not_ready
checked files: 606
verification errors: 0
remaining blockers:
  weave_content_canary
  wandb_completion
  paid_run_review_package
  one_model_full_canary
```

初見で読む順番と、必要に応じて見る詳細証跡は次です。

```text
1. このファイル
   全体像、前提知識、現在の課題を読む。

2. docs/taiwan_leaderboard_tasks.md
   省略なしの進捗表と最新証跡を見る。

3. temp/taiwan_release_operator_plan_20260701T043635Z.md
   次に人間が実行・承認すべき操作を見る。

4. outputs/taiwan_release_evidence/bundle_20260701T043635Z/summary.md
   release判断に使った証跡一式の人間向け要約を見る。

ここまでが通常の確認順序です。以下は詳細監査が必要なときに見る過去のnegative proofや補助証跡です。

5. temp/taiwan_operator_execution_plan_A411_invalid_approval.md
   operator planを実行前レビュー用に具体化したうえで、未承認の外部アクション承認レポートではshell生成・require-readyに進めないことを確認したA411のnegative proofを見る。
   A412では同じsource packet照合をbatch runner本体にも追加済み。
   A413では同じsource packet照合をlive content canary実行入口にも追加済み。
   A414ではW&B completionをpaid-run reviewへ直接反映する `--in-place` 同期にもvalidated dry-runを必須化済み。
   A415ではWeave Agents completionをpaid-run reviewへ直接反映する `--in-place` 同期にもvalidated dry-runを必須化済み。
   A416ではWeave Agents completion同期のdry-run元review JSONもSHA-256で固定し、後から差し替わったsource reviewでは正式証跡として通らないようにした。
   A417ではpassedなWeave content canaryについて、gate JSONだけでなくplan、command_result、prompt、OpenClaw sidecar、verifier JSONをbundle内で再読込し、task_id、canary text、command success、native Agents scopeが一致しない証跡を拒否するようにした。
   A418ではNeMoClaw post-install証跡について、setup、preflight、canary readiness、adoption、Markdown summaryの各出力をSHA-256で固定し、stepごとの出力JSON SHAとbundle内ファイルSHAが一致しない証跡を拒否するようにした。
   A419ではNeMoClaw導入ハンドオフのpost-install検証コマンドに `--fail-on-failed` を必須化し、失敗した検証を成功扱いで次工程へ進めないようにした。
   A420では同じ `--fail-on-failed` 条件をNeMoClaw adoption doctorの `setup_plan_safety` にも上げ、採用可否判断でも欠落を拒否するようにした。
   A421ではW&B必須ベンチかどうかを `benchmark_completion` source evidenceにも明示し、`wandb_completion_contract` との不一致をrelease bundle verifierで拒否するようにした。
   A422ではW&B adoption候補のsource audit JSON/SHAをrelease gate、operator plan、summary.mdに出し、実体draft JSONとの不一致をrelease bundle verifierで拒否するようにした。
   A423ではWeave Agents証跡に `trace_timestamp_quality` を必須化し、開始・終了時刻が欠けたspan、逆転したspan、同時刻で順序が証明できないtool実行、手編集されたtimestampカウンタを正式証跡として拒否するようにした。
   A424では `run_openclaw_agent_protocol.py check-agents` の診断出力にも `trace_order_health` を追加し、正式verifierを走らせる前のUI確認段階でtimestamp欠損、tool-before-input、final-answer後tool実行の疑いを見えるようにした。
   A425では `check-agents --json` を追加し、live content canary後のoperator planで診断JSONを `outputs/weave_agents_content_canary/agents_diagnostics/...agents.json` として残すようにした。
   A426ではこの診断JSONに `diagnostic_schema_version=1`、`generated_at`、W&B Agents APIの `query_source` を追加し、release bundle verifierがoperator plan内の `check-agents` コマンドについて `--json` と `.agents.json` 出力を要求するようにした。
   A427ではlive content canary実行自体が同じcanary task_idにscopeした `.agents.json` 診断を生成し、passed gateのsupport evidenceとしてbundleに含めて再検証するようにした。
   A428ではone-model full canaryについて、Agentic必須ベンチが含まれる場合はW&B completionだけでなくWeave Agents completionも合否条件にし、W&Bだけ揃ったcanaryをpassed扱いにしないようにした。
   A429ではこの条件をrelease bundle verifierにも上げ、production readiness reportのone_model_full_canaryとmanifestのcurrent_gate.weave_agents_completionを突き合わせ、passedなAgentic canaryに検証済みfull/agentic Weave rowがないbundleを拒否するようにした。
   A430ではW&B scope-attestation preflight reportにも、review JSON、completion JSON、scope-attestation JSONのpath/readable/sha256を `source_files` として入れ、bundle verifierが欠落・SHA不一致・path不一致を拒否するようにした。
   A431では採用済み候補のscope preflight reportが実ファイルとして存在する場合、bundleへ含め、passed reportのW&B identity、query_source、source_files、scope-attestation source JSON/SHA、bundle内SHA一致まで再検証するようにした。まだ生成されていない予定出力パスだけではbundleをinvalidにしない。
   A432では `render_wandb_scope_attestation.py` を追加し、reviewer、timestamp、実費、請求参照をCLI引数で入れたconfirmed scope-attestation JSONとrender report/Markdownを生成してからpreflight、dry-run、applyへ進む手順にした。
   A434ではbundle内にrender report実ファイルが存在する場合、candidate identity、output SHA、attestation JSON内のrender metadata、W&B writeなし・外部実行なしのsafety flagsまでrelease verifierが再検証するようにした。
   A435ではbundle内にrender Markdown実ファイルが存在する場合、reviewer向けMarkdownのstatus、table、preflight/sync commandがpaired JSON reportと一致することもrelease verifierが再検証するようにした。
   A436ではrender report内のnext_commands自体を契約化し、preflightが予定scope preflight reportへ `--json` を出すこと、sync dry-runが予定dry-run reportへ `--report-json` を出すこと、review/completion/scope pathがcandidateと一致することをrelease verifierで拒否条件にした。
   A437ではW&B adoption draftに載る `scope_attestation_render_command` 自体も検証対象にし、renderer script、reviewer/accounting flags、template/output/report/Markdown/preflight/dry-run path、deprecated flag不使用をbundle verifierで確認するようにした。
   A438ではNeMoClawのadoption checkをpost-install verifier内部の確認だけでなく、operator handoff上の明示ステップにした。setup_plan.operator_sequence、expected_evidence_paths、production readiness remediation、operator docs verifier、release bundle verifierが、standalone adoption JSON/Markdownとinstaller review Markdownを成果物として要求する。
   A439ではoperator_next_stepsに `command_count` と `evidence_path_count` を追加し、`evidence_template_count` と分けて表示・検証するようにした。これにより、NeMoClaw readinessは「6コマンド、9証跡、8テンプレート証跡」と読める。
   A440ではplaceholder解決後のoperator execution planにも、総コマンド数、実行可能コマンド数、証跡数をJSON/Markdown/shellコメントとして残すようにした。実行直前レビューでもrelease gateと同じ粒度で数を確認できる。
   A441ではoperator execution planに `source_operator_plan_sha256` を追加し、生成元operator_planのSHA-256をJSON、Markdown、shellコメントから確認できるようにした。最新bundleのoperator_plan.jsonに対するdry-runでSHA一致を確認済み。
   A442では既存run `f1veetyb` についてW&B Agents APIをread-onlyで確認し、`conversation_id_contains=f1veetyb` に一致するspanが0件であるためWeave Agents completionとして採用できないことをJSON化した。失敗verifierをpaid-review同期へ渡した場合もTracebackではなく `validation_failed` reportを残し、そのreportと元verifierをrelease evidence bundleに同梱する。
   A443では、その同梱済み失敗証跡をbundleに入れるだけでなく、release bundle verifierが中身を再読込して検証するようにした。失敗sync reportが `ok=false/status=validation_failed/dry_run=true/in_place=false/entry_count=0/change_count=0` のままか、参照するrejected verifierが `query_source.kind=wandb_agents_api` かつ `matching_span_count=0` のままかを確認し、改ざんやドリフトで成功扱いに見える場合はbundle integrityを落とす。
   A482では同じ `f1veetyb` のWeave Agents非採用証跡を再取得し、verifier自体も `schema_version=1/status=failed/ok=false` を持つ形式にした。native W&B Agents APIの結果は `matching_span_count=0/latest_trace_span_count=0` のままで、paired sync reportも `validation_failed/dry_run=true/in_place=false/entry_count=0/change_count=0` のため、release evidence bundleには成功証跡ではなく非採用証跡として入っている。
   A444ではNeMoClawのoperator handoffをさらに厳格化し、adoption checkは `--fail-on-not-adoptable`、production readinessは `--fail-on-not-ready` を必須にした。READMEだけでなく、operator docs verifier、production readiness、release bundle verifierも同じ必須check名を要求するため、古い `ok=true` JSONだけでは正式証跡として通らない。
   A445ではoperator planの各ステップに、template数だけでなく実際の合計command_countとevidence_path_countも表示するようにした。外部アクション実行前レビューで、例えばcontent canaryが「3コマンド、11証跡」なのに「2テンプレート、7テンプレート証跡」だけを見て誤解する状態を避ける。
   A446ではlatest pointerにもstandalone operator_planを含め、pointer verifierがformal gateとのoperator_plan一致を検査するようにした。bundle側はbundle-local operator_planを維持しつつ、top-level/pointerはstandalone operator planへ辿れる。
   A447ではNeMoClaw adoption要約にblockers、runtime_blockers、setup_runtime、missing_required_commandsをtop-levelで出すようにした。当時のlatest pointerだけを見ても、setup_installed/sandbox_readinessなどのreadiness理由と欠けているコマンドが分かる。
   A448ではW&B adoption候補ごとにrequired_human_fields、pending_scope_confirmation、pending_human_field_countをrelease gate / latest pointerへ出すようにした。既存runを正規ログへ採用する前に、scope確認、completion SHA、実費、請求参照など何が未記入かをstable pointerだけで確認できる。
   A472ではstandalone/bundled operator planに、どのrelease gate由来かを示すsource_release_gate_jsonとreadiness情報を持たせた。render_taiwan_operator_execution_plan.pyは --release-gate-json で期待gateと照合し、一致しない場合は --require-ready やshell生成に進めない。
   A473ではA472のrelease gate bindingをrelease bundle verifierでも再検証するようにした。source_release_gate_json、release_gate_json_template、--release-gate-json、requires_release_gate_match_for_shell_script が欠けたり、manifest.release_gate_pointerとずれたりするとbundle integrityが落ちる。
   A449ではW&B adoption draft全体にもpending_scope_confirmation_candidate_count、pending_human_field_count、pending_human_fieldsを出し、候補詳細を開かなくても人間確認待ちの件数とフィールド名が分かるようにした。release bundle verifierもこの集計を元draft JSONから再計算して照合する。
   A450ではNeMoClaw adoptionの元source JSON自体にもready_for_use、adoption_recommendation、blockers、runtime_blockers、setup_runtime、missing_required_commandsをtop-levelで出すようにした。latest pointerだけでなく、`temp/taiwan_nemoclaw_adoption_check_*.json` 単体を開いても採用可否とreadiness理由が読める。

6. outputs/taiwan_release_evidence/bundle_20260701T043635Z/external_action_approval_packet.md
   有料API、W&B write、NeMoClaw install、scope confirmationの承認要件を見る。

7. outputs/taiwan_release_evidence/bundle_20260701T043635Z/operator_plan.md
   non-prepare batch実行コマンドとlive content canary実行コマンドに `--external-action-approval-source-packet-json` と `--external-action-approval-report-json` の両方が入っていること、operator renderer safety flagsに `requires_canary_approval_scope_match_for_shell_script=true` があること、W&B completion同期とWeave Agents completion同期がdry-runとvalidated applyの2段になっていることを見る。

8. temp/taiwan_external_action_approval_REVIEWED_A411_TEMPLATE.verify.json
   元packetとのsource bindingは通るが、未承認テンプレートのままでは外部操作に進めないことを確認したnegative verifier reportを見る。

9. temp/wandb_relog_plans/agentic-math-deepseek-v4-pro-thinking-max.plan.json
   既存のAgentic Mathローカル完走結果をW&Bへre-logする前に、log予定のtable、summary metric、production artifact alias、post-log verifier commandを確認したdry-run planを見る。現在のbundle verifierはこのplanを同梱して再検証する。
   A394以降、実際にW&Bへre-logする `log_agentic_math_results_to_wandb.py` / `log_agentic_swe_results_to_wandb.py` は `--validated-dry-run-plan-json` がないと `wandb.login()` 前に停止する。
   そのplanのok、write flag、benchmark、entity/project、run名、tags、config、source path、log予定table/artifact、post-log verifier commandが現在の入力と一致しない場合もW&B writeへ進まない。
   A395ではさらに、dry-run plan内の `config.relog.source_sha256` が `source.source_sha256` と一致し、post-log verifier commandが `--expected-run-config relog.source_sha256.*` を要求することまでbundle verifierで確認する。
   A396ではNeMoClaw installer review commandにも、pinned lock JSONのSHAを `--expected-sha256` として明示することを必須にした。
   A397では既存W&B run採用templateに source audit JSON/SHA も入れ、draft、candidate、template、sync、release bundleで同じ既存結果auditに結び付いていることを検証する。
   A398ではrelease bundle verifierがsource audit JSONの中身も読み、formalized_records内にcandidateと同じW&B completion recordがあることまで検証する。
   A399ではproduction readinessとpaid-run review doctorも、adopted_existing_resultのscope attestationにsource audit JSON/SHAを必須化し、formalized_records内に同一W&B completion recordがなければ拒否する。
   A401では現行bundle向けに外部アクション承認テンプレートを再生成し、source-boundだが未承認のままではoperator execution planの--require-readyとshell生成が拒否されることを確認する。
   A402ではrelease bundle verifier自体もpre-run budgetのtarget_model / selected model bindingを検証し、manifest要約またはbundle内paid-review JSONが実行対象モデルと一致しない場合に拒否する。
   A403ではW&B re-log実writeにも `--external-action-approval-report-json` を必須化し、dry-run planにも成功/validation_failedを問わずW&B write承認要求を入れる。
   A404ではrelease evidence bundleが既存結果relog commandのscript本体とhelper依存も同梱・検証するようにし、dry-run planだけでは再現可能証跡として不十分な状態を拒否する。
   A405ではNeMoClaw setup/post-install/adoption JSONに `schema_version=1` を入れ、release bundle verifierがschemaなしのNeMoClaw証跡を拒否するようにした。
   A406ではrelease evidence bundleの `manifest.json` 自体にも `schema_version=1` を入れ、schemaなしの古いbundleを正式証跡として通さないようにした。
   A407では正式release gate JSONにも `schema_version=1` を入れ、latest pointer verifierがschemaなしの古いgateを指す状態を拒否するようにした。
   A408ではproduction readiness reportにも `schema_version=1` を入れ、release bundle verifierがschemaなしの古いreadiness reportを拒否するようにした。
   A409ではW&B re-log write前の外部承認レポートについて、source packetが読めること、source SHAが一致すること、reviewed packetがapproval_packet_jsonと同じ実体を指すことを追加で検証するようにした。
   A410ではW&B re-log write時に `--external-action-approval-source-packet-json` も必須化し、承認レポートのsource bindingがオペレーター指定のsource packetパスとSHAに一致しなければ `wandb.login()` 前に停止するようにした。
   A411ではoperator execution planのshell生成にも同じsource packet契約を入れ、`--external-action-approval-source-packet-json` と承認レポートのsource bindingが一致しなければ外部操作用shellを生成しないようにした。
   A412では `run_taiwan_full_eval_batch.py` 本体にも同じ契約を入れ、non-prepare実行では `--external-action-approval-source-packet-json` と `--external-action-approval-report-json` の両方を要求し、source packetのパスとSHAが承認レポートのsource bindingと一致することを実行入口で確認するようにした。
   A413では live content canary の実行入口 `run_weave_agents_content_canary.py --execute` にも同じ契約を入れ、OpenClawを起動する前にsource packetと承認レポートの一致を確認するようにした。
   A414では通常のW&B completion同期もdry-runなしでpaid-run review JSONを直接書き換えられないようにし、`--in-place` 実行には `--validated-dry-run-report-json` を必須化した。
   A415ではWeave Agents completion同期にも同じ制約を入れ、`sync_weave_agents_completion_to_paid_review.py --in-place` も検証済みdry-runなしではreview JSONを書き換えられないようにした。
   A416ではWeave Agents completion同期にも `source_review_sha256` を入れ、dry-run report、paid-review entry、bundle内source review JSONのSHA不一致をrelease verifierで拒否するようにした。
   A417ではWeave content canaryのpassed証跡にもsupport evidenceの完全性を要求し、verifier JSONだけ差し替えたbundleや、prompt/plan/command_resultが伴わないpassed gateをrelease verifierで拒否するようにした。
   A418ではNeMoClaw post-install verifierが `outputs_sha256` と各stepの `output_json_sha256` を出し、release verifierがbundle内ファイルを再ハッシュして一致確認するようにした。
   A419では `setup_plan.post_install_verification_command` 自体に `--fail-on-failed` を要求し、release verifierがこのフラグ欠落を拒否するようにした。
   A420ではadoption doctorも `missing_post_install_verification_command_markers` を出し、`--fail-on-failed` 欠落時は `setup_plan_safety` をinvalidにするようにした。
   A421では `benchmark_completion.required` をsource evidenceとして出し、W&B completion contract側のrequired判定だけが水増し・改ざんされる状態を拒否するようにした。
   A422では `wandb_adoption_draft.source_audit_json/source_audit_sha256` と各候補のsource audit情報をsummaryにも出し、stable pointerやoperator planだけを見ても既存結果auditへの紐付けが確認できるようにした。
   A423ではWeave Agents completionとcontent canaryの両方で `required_evidence.trace_timestamp_quality_required=true`、`trace_timestamp_quality` check、`spans_with_invalid_timestamps=0` を要求し、bundle verifierがspan本体を再パースして検証するようにした。
   A424では同じ考え方をoperator診断の `check-agents` にも広げ、`content_capture_health` でvalid/invalid timestamp数とfinal-answer marker数を、`trace_order_health` でtool/input/final-answerの時刻関係を出すようにした。
   A425ではこのoperator診断を標準出力だけでなくJSON evidenceとして保存できるようにし、content canaryの実行手順にも保存コマンドを入れた。
   A426では保存される診断JSONにschema/version/query provenanceを持たせ、operator planがJSON保存なしのAgents診断や `.agents.json` 以外の出力名を指示した場合はrelease bundle verificationで落とすようにした。
   A427では `check-agents` にconversation scopeを追加し、content canary runnerがformal verifier成功後に同じtask_idで診断JSONを作る。gate summarizerとrelease bundle verifierは、その診断JSONのschema、query_source、latest_trace_id、span時刻、content_capture_health、trace_order_healthを再検証する。
   A428では `one_model_full_canary` の合否にWeave Agents completionを加えた。`agentic_math` または `agentic_swe` が必須ベンチに含まれる場合、full phaseまたはagentic phaseの検証済みWeave Agents completionがなければ、W&B completionが揃っていてもcanaryは通らない。
   A429ではこのA428条件をportableなrelease bundleでも再検証する。`one_model_full_canary` がAgentic必須ベンチを含む場合、production readiness reportの `weave_agents_completion_required`、completed/missing phase fields、manifest `current_gate.weave_agents_completion` の検証済みrowが矛盾するとbundle integrityを落とす。
   A430では既存W&B run採用のpreflight段階も、source file hashつきの証跡にした。未確認テンプレートを検証して失敗する場合でも、どのreview/completion/attestationファイルを読んだ失敗なのかをSHAで追える。
   A431では確認済みscope-attestationでpreflightが通ったreportについても、bundle内に実ファイルがあればsource_filesとscope-attestation source SHAまで再検証する。予定パスだけでまだ生成されていないpreflight reportは、次に人間が作るべき証跡として扱い、bundle破損とは扱わない。
   A432ではscope-attestationの手編集を減らすため、draft templateからconfirmed版をレンダリングするoffline commandをoperator planに入れた。rendererはcompletion/review/source-auditのpathとSHAを確認し、W&B writeや外部実行をせずにrender reportを残す。
   A434では、そのrender reportがbundleに入った場合、reportだけを差し替えても通らないように、output_sha256、candidateのbenchmark/entity/project/run_id、review/completion/source-audit path、attestation JSON内のrender metadata、safety flagsをbundle verifierが読み直す。
   A435では、同じrender Markdownもpaired JSON reportから期待される表とコマンドを再構成して照合し、人間が見るMarkdownだけが古い・別run・危険な手順に差し替わる状態を拒否する。
   A436では、paired JSON/Markdownに載るpreflight/sync dry-run commandが、実際に期待されるpreflight reportとdry-run reportを作るコマンドであることまでbundle verifierが確認する。
   A437では、まだrender report実ファイルが未作成でも、sync-ready候補のdraft-level render commandが完全な証跡出力先を持つことを検証する。

10. temp/wandb_relog_plans/agentic-swe-deepseek-v4-pro-thinking-max.plan.json
   現在のDeepSeek SWE-Bench Pro証跡が1/80問のpartialであり、正式W&B completionとしてre-logできないことを確認したvalidation_failed planを見る。これもbundle verifierが同梱して再検証する。

11. temp/taiwan_existing_results_audit_20260701T043635Z.json
   既存DeepSeek Agentic Mathローカル完走が `archived_not_release_candidate` として記録され、W&B relog対象やNeMoClaw completion proofとして扱われていないことを見る。

12. outputs/taiwan_full_eval/existing_results_archive_manifest.json
   そのarchive判断がsummary/results/partial resultsのSHA-256に固定され、別内容の結果へ流用できないことを見る。このmanifest本体は最新release bundleにも `existing_results_audit:archive_manifest` として同梱される。

13. temp/taiwan_nemoclaw_adoption_check_20260701T043635Z.md
   NeMoClawがAgentic Math/SWE向けにadoptable_for_agentic_benchmarksであることを見る。
   temp/nemoclaw_setup_check_20260701T043635Z.json は runtime/sandbox状態の最新証跡を示す。
   setup、post-install、adoptionの各JSONは `schema_version=1` を持ち、release bundle verifierはこのschema markerが欠けた証跡を拒否する。
   同じJSONの setup_plan.installer_review_command は `--expected-sha256 a4ebc5710dfd8b10035968fd25562773ad56ec4c73ecf9bfe5c78c77724000e7` を含み、adoption doctorとrelease bundle verifierはこのSHAがないreview commandを拒否する。
   同じJSONの setup_plan.operator_sequence と setup_plan.expected_evidence_paths で、導入前確認からpost-install、canary、adoption check、production readinessまでのhandoff順序と必要証跡も確認できる。

14. temp/nemoclaw_post_install_verification_20260701T043635Z.json
   post-install検証が単なる `ok=true` JSONでは通らず、setup、protocol preflight、canary readiness、adoption checkの各payload contractを満たす必要があることを見る。
```

## 最初に読む用語説明

英語の名前は、スクリプトやJSONの機械判定名として残しています。意味は次の通りです。

```text
release gate
  本番公開してよいかを判定する最終チェック。

evidence bundle
  判定に使った証跡ファイル一式。W&B検証結果、Weave検証結果、費用レビュー、設定、summaryをまとめたもの。

W&B
  評価run、スコア、出力table、artifactを保存する場所。ここに正式結果が残っていない評価は、原則として完走扱いにしない。

Weave Agents
  Agentic評価の会話ログを見る場所。問題文、tool実行、最終回答の順序と中身を監査する。

OpenClaw
  評価対象モデルをAgentとして動かすための実行基盤。Codexはこのrepoを実装している開発エージェントで、OpenClawは評価を走らせる側。

NeMoClaw
  NVIDIA系のOpenClaw実行基盤候補。sandbox、ネットワーク制御、NVIDIA連携の説明力があるため、本番Agentic評価で使いたい。

canary
  いきなり全モデルを高額実行する前の代表試験。まず1モデルで全体が最後まで通るか確認する。

full canary
  1モデルで全ベンチを最後まで実行し、W&B、Weave、費用レビュー、Total Scoreまで揃える確認。

paid-run review
  有料APIを使った評価について、run、trace、費用、請求参照、採用可否を整理するレビュー。

Total Score
  各ベンチ結果を台湾版リーダーボードの重みに従って集計した総合点。
```

現在の4つのblockerは、日本語に直すと次です。

```text
weave_content_canary
  Weave Agentsに、問題文、tool実行、最終回答が正しい順序で見えることの確認。

wandb_completion
  必須ベンチの正式結果がW&B上で完走扱いになっていることの確認。

paid_run_review_package
  有料実行の費用、請求参照、W&B/Weave証跡が揃っていることの確認。

one_model_full_canary
  まず1モデルで全ベンチを最後まで通し、Total Scoreまで出すこと。
```

## このプロジェクトで作っているもの

目的は、台湾繁体字圏向けに、実運用で信頼できるLLM評価リーダーボードを作ることです。

ここで作っているものは、単なるスコア表ではありません。次をまとめた「評価パイプライン」です。

```text
1. 評価データ
2. モデル実行
3. 採点
4. W&Bへの結果記録
5. Weave Agentsへの会話・tool実行記録
6. 有料API実行の費用レビュー
7. Total Scoreの集計
8. 本番採用できるかを判定するrelease gate
```

重要なのは、後から第三者が見ても「このモデルを、このデータで、この条件で評価し、このログと請求根拠が残っている」と確認できることです。

## 前提知識

リーダーボードでは、複数モデルを同じ条件で評価して比較します。点数だけでなく、再現性、データの出所、採点方法、実行ログ、費用の説明責任が重要です。

W&Bは、評価run、入力データ、出力、スコア、artifactを記録する場所です。このプロジェクトでは、W&Bに正式な結果が残っていないものは原則として完走扱いにしません。

Weave Agentsは、Agentic評価の会話とtool callを見る場所です。たとえば数学でPythonを使う、SWE-Benchでrepoを編集する、という評価では、モデルが何を見て、いつtoolを使い、いつ最終回答したのかが見えないと本番結果として採用しにくいです。

OpenClawは、Agentic評価を動かすための実行基盤です。Codexはこのrepoを実装している開発エージェントであり、OpenClawは評価対象モデルを走らせる側の仕組みです。つまり、OpenClawを使うことは「Codexを評価対象にする」という意味ではありません。

NeMoClawは、NVIDIA側のOpenClaw系実行基盤候補です。sandbox、ネットワーク制御、NVIDIA連携の観点で本番評価の説明力が上がる可能性があります。現時点ではローカルruntime/sandbox/native weave readinessは通っており、台湾版Agentic Math/SWEの優先backendとして扱っています。ただし、実ベンチ問題でのnative Weave trace、W&B completion、paid-run review、official evaluator結果がまだないため、本番完了証跡としては未完了です。

## 何を測りたいのか

台湾版では、日本語版Nejumiをそのまま置き換えるだけでは不十分です。台湾繁体字圏の実利用に近い能力を測る必要があります。

主な評価領域は次です。

```text
知識・読解:
  TMMLU+, TCEval-v2, HLE zh-TW

対話品質:
  MT-Bench-TW

幻覚耐性:
  HalluLens zh-TW

指示遵守:
  IFEval zh-TW, Script adherence

関数呼び出し:
  BFCL-v3 zh-TW

数学・推論:
  OlymMATH-HARD zh-TW

Agentic coding:
  SWE-Bench Pro

堅牢性・ALT:
  TMMLU+ robustness, 簡体字混入率など

総合:
  Taiwan aggregate / Total Score
```

簡体字混入は、既存ベンチの点数から直接減点するのではなく、独立したALT指標として扱う方針です。

## 現在の作業仮説

現在は、次の仮説で進めています。

```text
仮説1
  まず1モデルだけで全ベンチを本番同等に完走できれば、
  データ、採点、W&B記録、Weave記録、費用、Total Scoreの問題点をまとめて洗い出せる。

仮説2
  Agentic MathやSWE-Bench Proは、モデルがtoolを使うこと自体が評価の一部なので、
  点数だけでなく、問題提示、tool実行、最終回答の順序が監査できる必要がある。

仮説3
  NeMoClawを使えるなら、sandboxとネットワーク制御を説明しやすくなり、
  Agentic評価を本番向けにしやすい。

仮説4
  高価なモデルを5-10個まとめて走らせる前に、
  低コストのOpenAI directモデルなどで1回full canaryを通す方が、
  使えない途中生成物と無駄な課金を減らせる。
```

## 目下の課題

残りの課題は5つです。

```text
1. Weave Agentsの記録品質
   問題文、tool実行、最終回答が正しい順序で見えるlive canaryを通す。

2. NeMoClaw導入確認
   ローカルでNeMoClawを導入し、restricted policyで安全に使えることを確認する。

3. W&B完走証跡
   Agentic Mathだけでなく、Agentic SWEとTaiwan full集計もW&B上で完走扱いにする。

4. 有料実行レビュー
   どのモデルにいくら使い、どの請求・run・traceに対応するかを記録する。

5. 1モデルfull canary
   低コストモデルで全ベンチを最後まで回し、Total Scoreまで出す。
```

## なぜ最近は検証ゲートを固めているのか

直近の作業は、点数を増やす実行ではなく「本番採用できる証跡条件」を固めています。

理由は単純です。評価を回しても、次の状態だとリーダーボードに載せられません。

```text
W&Bに正式な結果が残っていない
Weave Agentsで問題文、tool実行、最終回答の順序が見えない
実費や請求参照が記録されていない
どのrunを正式採用したのか曖昧
別runや別projectの証跡を混ぜても通ってしまう
```

そのため、release evidence bundle verifierを強化しています。これは、release bundleに入っているJSONやMarkdownを再読込し、見かけだけ `ok=true` の証跡を拒否するための仕組みです。

直近の追加では、paid-run reviewの要約だけでなく、bundle内の元JSONも読み直し、status、phase、canary、model_count、費用記入有無、W&B/Weave検証フラグ、completion entry数が一致することを確認するようにしました。これにより、prepared状態のone-model canaryでも、要約と実ファイルがずれていればrelease bundleが通りません。

さらに、release bundleの `manifest` top-levelと `current_gate` が同じstatus、readiness状態、gate数、blocker数、blocking gate一覧を主張していることも検証します。これにより、同じbundle内で人間向けの要約と機械判定用のcurrent gateが食い違う状態を拒否します。

W&B完走条件についても、`wandb_completion_contract` のrequired benchmark数、release-proven件数、standalone OK件数、formalized existing件数、missing benchmark一覧、`complete/status` を `benchmarks[]` から再計算して検証します。さらに各benchmark行を `benchmark_completion` と `existing_results_formalization.formalized_records` の元証跡に照合し、standalone/review/formalizedのflag、run_id、path、件数を水増ししたbundleを拒否します。A384以降は、人間向けの `summary.md` にもstatus、complete、件数、missing benchmark、run id、attestation template、preflight/dry-run report、missing reason、next action、commandが表示されていることを検証対象にしました。A421以降は、`benchmark_completion` 側にも `required` を明示し、source evidenceが存在する場合は `wandb_completion_contract.required` と一致しなければbundleを拒否します。A422以降は、W&B adoption候補のsource audit JSON/SHAもrelease gate要約、operator plan、summary.mdに保持し、要約だけが実体draft JSONと食い違うbundleを拒否します。A423以降は、Weave Agents証跡のtimestamp品質もW&B/Weave完走条件の一部として扱い、timestampが欠けた・壊れた・手編集された証跡をrelease bundle側でも拒否します。

新しく生成されるW&B completion verifier JSONには、`query_source` として `wandb.Api(timeout=60)`、entity、project、run_id、run_path、benchmark、summary/artifact取得元を記録します。通常の新規run同期ではこの `query_source` がない completion JSON を paid-run review に取り込めないようにし、paid-run review doctorとrelease bundle verifierも同じ条件を再検証します。既存runの採用だけは別途 scope-attestation JSON で人間の確認を必須にしています。release bundle verifierは、`manual_json` や別run pathへの改ざんだけでなく、手編集されたpaid-review JSONがadopted/non-adopted扱いをcurrent gateと食い違わせることも拒否します。

既存W&B結果を採用する場合は、いきなりpaid-run review JSONを書き換えません。現在の運用では、まずscope-attestation JSONにreviewer、timestamp、費用見積、請求参照、completion SHAを具体値で入れ、その後 `sync_wandb_completion_to_paid_review.py` のdry-runを `--report-json` 付きで実行します。dry-run reportを確認してから、同じ入力で `--in-place` と `--validated-dry-run-report-json` を付けたapply commandを実行する順序です。A414以降は既存結果採用に限らず、通常のW&B completion同期でも `--in-place` には `--validated-dry-run-report-json` が必須です。A415以降はWeave Agents completion同期でも同じく、`sync_weave_agents_completion_to_paid_review.py --in-place` には `--validated-dry-run-report-json` が必須です。A416以降はWeave Agents completion同期でもdry-run元review JSONのSHAを `source_review_sha256` として保存し、paid-review entryの `sync_dry_run_source_review_sha256`、dry-run payload、bundle内source review JSONのSHAが一致しなければ正式証跡として拒否します。A417以降はWeave content canaryのpassed証跡も、verifier JSONだけでなくplan、command_result、prompt、OpenClaw sidecarをbundle内で再検証します。A418以降はNeMoClaw post-install証跡も、各出力ファイルと各step output JSONのSHA-256をbundle内ファイルから再計算して確認します。A419以降はNeMoClaw導入ハンドオフのpost-install検証コマンドにも `--fail-on-failed` が必須です。A420以降は、adoption doctorの採用可否判断でも同じフラグ欠落を拒否します。A421以降は、W&B必須ベンチの扱いも `wandb_completion_contract` だけでなく `benchmark_completion.required` としてsource evidenceに残します。A422以降は、W&B adoption候補が参照する既存結果auditのJSON pathとSHAをstable pointer、operator plan、summary.mdから直接確認できます。A423以降は、Weave Agents completion同期とcontent canaryのどちらでもtimestamp品質を必須化し、`trace_timestamp_quality` checkや `spans_with_invalid_timestamps=0` が欠ける証跡、span本体の時刻が壊れている証跡、tool順序が同時刻で証明できない証跡を採用しません。apply側も、dry-run reportのentry、change、review path、review SHA、scope-attestation source SHA、verify flag、unmatched countが現在のsync内容と一致しない場合はJSONを書き換えません。applyに成功した場合、paid-review内のW&B completion entryには `sync_dry_run_report_json`、`sync_dry_run_source_review_json`、`sync_dry_run_source_review_sha256` も保存されます。さらにA390以降は、production readinessとpaid-review doctorもこのdry-run証跡を必須として扱い、手で `adopted_existing_result=true` を入れただけのentryを拒否します。A397では、scope-attestation JSONに `source_audit_json` と `source_audit_sha256` も入れ、draft、candidate、template、sync、release bundleの全段階で同じ既存結果auditに結び付いていることを検証します。A398ではrelease bundle verifierがsource audit JSONの `formalized_records` も読み、candidateと同じbenchmark/entity/project/run_id/completion pathを持つW&B completion recordがなければbundle integrityを落とします。A399では同じsource audit条件をproduction readinessとpaid-run review doctorにも上げ、review entryのscope-attestationに `source_audit_json` と `source_audit_sha256` があり、そのauditの `formalized_records` に同じW&B completion recordがなければpaid-run review自体を通さないようにしました。A488ではrelease bundle verifierもsource audit JSON内の `wandb_completion_records` を読み、同じcandidate行が `ok=true`、`schema_current=true`、`schema_current_issues=[]`、`verification_schema_version=1`、`observed_evidence_present=true` でなければbundle integrityを落とします。A588以降では、`f1veetyb` 由来のAgentic Math候補は現在のNeMoClaw session-audit契約を満たさないため正式採用しません。対応するローカルDeepSeek完走結果は `outputs/taiwan_full_eval/existing_results_archive_manifest.json` のSHA-256付きmanifestで `archived_not_release_candidate` として保存され、最新release gate `temp/taiwan_release_gate_20260701T043635Z.json` では既存結果formalization blockerから除外されています。A430以降はpreflight reportの `source_files` にreview JSON、completion JSON、scope-attestation JSONのpath/readable/sha256が必要で、release bundle verifierは欠落、path不一致、SHA不一致、未読込扱いを拒否します。A431以降は、確認済みscope-attestationでpreflightが通ったreportについても、bundle内に実ファイルがあればsource_files、candidate W&B identity、query_source、scope-attestation source JSON/SHA、bundle内SHA一致まで再検証します。A436以降は、render reportのnext_commandsが実際にそのpreflight reportとsync dry-run reportを作るコマンドであることも検証します。A437以降は、draft-levelの `scope_attestation_render_command` 自体も同じ予定出力先を持つことを確認します。A478以降は、sync-readyなW&B adoption contract行について、scope確認、scope-warning、render/preflight/dry-run/apply command、stale verifier refresh command、placeholderでないrun_idをrelease bundle verifierが独立に検査します。さらにrelease bundle verifierは、この2つのJSONの中身を読み直し、preflight/syncがどちらも `validation_failed` で、sync dry-runが `dry_run=true`、`in_place=false`、`entry_count=0`、`change_count=0` のままreview JSONを変えないことまで検証します。加えて、人間向けの `summary.md` にも採用候補draftのstatus、候補数、attestation template、人間が埋めるべきfield、preflight/dry-run/apply command、未承認テンプレートチェックのstatus、record count、output dir、各レコードのattestation/preflight/sync path、review mutation、W&B write、model inference列が表示されていることを検証するようにしました。A385では外部アクション承認packetのscope_confirmationも強化し、承認済みpacketが通るにはscope-attestation JSONが実在するcompletion JSON、review JSON、completion SHA、benchmark/entity/project/run_idに結び付いている必要があります。dry-run reportが後でbundleに含まれた場合、release verifierは `ok=true`、`status=synced`、`dry_run=true`、`in_place=false`、review path、unmatched count、change対象run id、source review SHA、source attestation SHA、source audit SHA、source audit formalized record対応などを再検証します。

Weave Agents証跡については、verifier JSONにW&B Agents API query provenanceを含めるようにし、release bundle verifierも `query_source.kind=wandb_agents_api`、API endpoint、project、agent、conversation scope、latest trace span数の整合性を検証します。さらにsync処理とpaid-run review doctorでも同じ `query_source` を必須化したので、手で整形しただけのWeave proof JSONはレビューJSONに取り込む段階、またはレビューpackageを合格させる段階で拒否されます。

また、`temp/latest_taiwan_release_gate.json` は単なるpath pointerではなく、正式なtimestamp付きrelease gateの決定サマリを持つようにしました。gate数、blocker数、required next actions、W&B completion contract、paid-run review、NeMoClaw adoption、operator next steps、bundle file count、checked file countをstable pointerだけで確認でき、pointer verifierが正式gateとの不一致を拒否します。

さらに、release gate、latest pointer、bundle current_gate、operator plan、bundle summaryには外部操作チェックリストを追加しました。これは各blockerの解除に必要な有料API、W&B access、W&B write、第三者承認、NeMoClaw install、scope confirmationを一覧化するものです。pointer verifierは正式gateとの一致を確認し、release bundle verifierはこのチェックリストを `operator_next_steps` から再計算して照合します。加えて、summary.mdに current_gate と同じstatus、外部操作件数、各ゲート行が表示されていることも検証するため、人間向けの実行前チェック表だけを手で軽く見せたり、レビュー用summaryから隠したり、placeholderの表でごまかしたりすることはできません。

## 現在できていること

```text
評価データ準備:
  多くは実装済み。未確定データやlicense確認が必要なものは別途gateで止める。

評価runner:
  非Agentic系、Agentic Math、SWE-Bench Pro、Taiwan aggregateの実装が進んでいる。

W&B検証:
  schema v1のcompletion verifierを作成し、run_state、metrics、tables、artifact aliasesを確認する。

Weave検証:
  Agents traceに問題文、tool、最終回答が正しい順序で見えるかを検証する仕組みがある。

NeMoClaw実行基盤:
  ローカルruntime/sandbox/native weave readinessは最新release gateでpassed。
  Agentic Math task-agent mode、SWE-Bench Pro copy-mode checkout transfer、runner source bundle contractまで実装済み。
  installer review、SHA lock、restricted policy、operator docs verifier、post-install verifierの設計とテストはある。
  READMEの導入手順はpinned installer lockとreview/install/post-install/canary/adoptionコマンド契約に照合され、release bundleでも検証される。
  本番採用証跡では `setup_plan.production_install_and_onboard_command` を使い、installとonboardを同じJSONに残す単一コマンドで検証する。

release gate / evidence bundle:
  最新bundleは605ファイルを検証し、integrityはOK。
  production readiness reportは schema_version=1 を持ち、release gate / latest pointer / bundle manifestは readiness_report_schema_version=1 を運ぶ。
  正式release gate JSONは schema_version=1 を持ち、latest pointer verifierはschemaなしの旧gateを拒否する。
  `manifest.json` は schema_version=1 / bundle_version=2 を持ち、verifierはschemaなしの旧bundleを拒否する。
  operator rendererのsource-contractとbundle metadataは、current one-model canaryがOpenAI-direct pathから外れたshell handoffにならないこと、承認済みpaid_api model scopeがOpenAI-direct canaryに一致すること、operator plan Markdownにcanary approval-scope safety flagが残ることを検証する。
```

## 現在できていないこと

```text
1モデルfull canary:
  まだ完走していない。

全体スコア:
  まだ正式には出せない。

NeMoClaw実ベンチ完走証跡:
  NeMoClaw readiness自体は通っているが、実ベンチ問題でのnative Weave trace、W&B completion、paid-run review、official evaluator結果がまだない。
  そのため、NeMoClawを使ったAgentic Math/SWEの本番完了とはまだ言えない。

Weave content canary:
  まだpassedなlive content canaryがない。次回はOpenRouterなしで、事前承認済みの低コストOpenAI-direct経路を使う。

複数モデル仮リーダーボード:
  まだ実行すべき段階ではない。まず1モデルfull canaryを通す。
```

## 次の実行順序

次に進めるなら、順序は次が妥当です。

```text
1. OpenAI directの低コストモデルを1つ選ぶ
2. operator planを render_taiwan_operator_execution_plan.py で具体化し、RUN_ID、MODEL_SLUG、Weave gate pathが残っていないか確認する
3. Weave content canaryを通す
4. NeMoClaw post-install / canary readinessを同じtimestampの証跡で再確認する
5. 同じ1モデルで全ベンチを実行する
6. W&B completion verifierとWeave Agents verifierを作る
7. paid-run reviewに実費と請求参照を入れる
8. Total Scoreを出す
9. ユーザーが結果、trace、費用をレビューする
10. 問題なければ5-10モデルへ展開する
```

この次手順では、有料API実行、W&B write、paid-run review mutationは明示承認後にだけ行います。OpenRouterは使いません。

## 英語用語の読み替え

```text
benchmark
  ベンチマーク。モデルに解かせる評価問題セット。

harness
  評価ハーネス。データ読込、モデル呼び出し、採点、記録を行う実行一式。

artifact
  評価データ、翻訳済みデータ、採点結果、設定ファイルなどの保存物。

trace
  証跡。モデルに何を見せ、何を返し、どのtoolをいつ使ったかの記録。

Weave Agents
  Agentic評価の会話とtool callを確認する画面。

OpenClaw
  Agentic評価を実行するための基盤。

NeMoClaw
  NVIDIA連携のAgentic実行基盤候補。

canary
  小さめ、または1モデルの代表実行。いきなり全モデルを走らせる前の本番同等確認。

release gate
  本番公開に必要な条件が揃っているかを機械的に判定するチェック。

evidence bundle
  release判断に必要なJSON、Markdown、設定、検証結果、操作スクリプトをまとめたフォルダ。

paid-run review
  有料API実行の目的、モデル、範囲、費用、請求参照、W&B/Weave証跡をまとめる記録。

Total Score
  個別ベンチの点数を集約したリーダーボード用の総合スコア。
```
