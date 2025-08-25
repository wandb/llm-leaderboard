# BFCL 評価ガイド（Nejumi LLM Leaderboard 4 版）

**TL;DR**

BFCL (Berkeley Function Calling Leaderboard) は、LLMの関数呼び出し能力を包括的に評価するベンチマークです。Nejumi Leaderboardでは**BFCL v3を日本語化**し、以下の特徴で運用しています：

🚀 **統一OSSハンドラー導入**: `unified-oss-fc`（FC対応）と`unified-oss-jsonschema`（FC未対応）により、OSSモデルの評価が大幅に簡素化

📊 **評価範囲**: シングルターンからマルチターン・マルチステップまでの関数呼び出しを評価（関数選択、並列実行、マルチステップ推論など）。ただし、Nejumi Leaderboardではparallelを除外

⚙️ **設定方法**: モデルconfigに`bfcl_model_id`を指定するだけ（詳細は[SUPPORTED_MODELS.md](../scripts/evaluator/evaluate_utils/bfcl_pkg/SUPPORTED_MODELS.md)参照）

---

## 1. BFCL とは

**BFCL (Berkeley Function Calling Leaderboard)** は、LLMの関数呼び出し能力を包括的に評価するベンチマークです。シングルターンからマルチターン、リアルワールドのシナリオまで、幅広い関数呼び出しスキルを測定します。

### BFCLのバージョン進化

| バージョン | 主要特徴 | 評価の焦点 |
|-----------|----------|-----------|
| **V1** | エキスパートが精選したシングルターン関数呼び出し | AST（抽象構文木）評価による正確性検証 |
| **V2** | リアルな関数ドキュメント・クエリを大量収集 | バイアス・データ汚染への耐性強化、リアルワールド多様性対応 |
| **V3** ⭐ | **マルチターン・マルチステップ**関数呼び出し導入 | 連続的な関数呼び出しと状態ベース検証 |
| **V4** | エージェント的能力評価への移行 | Web検索、メモリ管理、フォーマット頑健性 |


⭐ **現在使用**: Nejumi LeaderboardではBFCL v3を日本語化して利用

### 評価対象と評価軸

BFCLは以下の多面的なスキルセットを評価します：

#### 🎯 **基本的な関数呼び出し能力**
- **関数選択能力** (Multiple, Parallel Multiple): 複数の関数から適切なものを選択
- **並列呼び出し能力** (Parallel): 複数の関数を同時実行
- **構文正確性**: 正しい引数と型での関数呼び出し

#### 🧠 **高度な判断・推論能力**
- **関連性判断** (Relevance Detection): 不適切な関数を見極め、呼び出しを避ける判断力
- **マルチターン対応**: 連続対話での文脈維持と適切な関数選択
- **マルチステップ計画**: 複数ステップにわたる論理的な関数呼び出し計画

#### 🔄 **エージェント的能力**（V4で拡張。V3ではまだない）
- **多段階推論**: Web検索でのマルチホップ質問への対応
- **状態管理**: メモリへのアクセス・読み書き能力
- **フォーマット頑健性**: 関数・プロンプト・フォーマット変化への適応力

### 参考リンク

- 📊 **[BFCL Function-calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)**: 最新の評価結果と各モデルの性能比較
- 📝 **[BFCL V3 Blog](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html)**: マルチターン・マルチステップ機能の詳細解説



---


## 2. 日本語評価データセットの作り方
BFCL(v3)の問題には、いくつかの種類がある。詳細はBFCLの[ブログ](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html)を参照してください。

- **翻訳**
    - qwen/qwen3-235b-a22bを用いてベース翻訳。人手で修正も実施
    - llm-leaderboard/scripts/translation/bfcl_translation.pyを利用
    - **ルール**: 関数名、コード関連内容は翻訳対象外
- **抽出** 
    - max tokenが小さいモデルも評価できるようにturn数を3turn以下になるように抽出
        - llm-leaderboard/scripts/translation/bfcl_multi_turn_count.pyを用いて、Turn数を計算
        - llm-leaderboard/scripts/translation/sort_bfcl_file.pyを用いて並び替え
    - ランダムに30問を抽出
        - idが小さい順番に30問抽出すると似たような問題が多くなるため、ランダム化
        - 30問に満たない問題は全問
        - 保存するにあたり人手での翻訳確認の品質担保のため、50問以上あるカテゴリは、50問に絞ってW&B artifactsに保存[link](https://wandb.ai/llm-leaderboard/nejumi-leaderboard4/artifacts/dataset/bfcl)
    - parallelの削除
        - 並列処理に対応していないモデルが存在していたため、parallelの問題は削除
- **その他詳細な処理**
    - 問題文に英語の質問が含まれている場合、該当問題を削除
        - 対象: BFCL_v3_live_irrelevance.json, BFCL_v3_parallel_multiple.json, BFCL_v3_live_multiple.json （例: live_parallel_multiple_1-1-0, live_parallel_multiple_2-2-0, live_parallel_multiple_3-2-1）, BFCL_v3_multiple.json, BFCL_v3_simple.json, BFCL_v3_parallel_multiple.json
    - 各カテゴリごとにランダムに30問を抽出（30問未満の場合は全問を使用）
        - BFCL_v3_live_irrelevance.json（882問→30問）
        - BFCL_v3_irrelevance.json（240問→30問）
        - BFCL_v3_simple.json（400問→30問）
        - BFCL_v3_live_multiple.json（1,053問→30問）
        - BFCL_v3_live_simple.json（258問→30問）
        - BFCL_v3_multiple.json（200問→30問）
        - BFCL_v3_parallel.json（200問→30問）
        - BFCL_v3_parallel_multiple.json（200問→30問）
    - BFCL_v3_java.json, BFCL_v3_javascript.jsonは各30問をそのまま使用
    - BFCL_v3_live_parallel_multiple.jsonは24問から21問に削減（上記の英語以外の質問を含む問題を削除）
    - BFCL_v3_live_relevance.json: 18問
    - BFCL_v3_live_parallel.json: 16問
    - multi-turn系の問題は3turn以下に絞り込み
        - BFCL_v3_multi_turn_base.json: 200問
        - BFCL_v3_multi_turn_long_context.json: 200問
        - BFCL_v3_multi_turn_miss_func.json: 200問
        - BFCL_v3_multi_turn_miss_param.json: 200問
    - possible answerに日本語のオプションを追加
        - 対象: BFCL_v3_live_multiple.json, BFCL_v3_live_parallel.json, BFCL_v3_multiple.json, BFCL_v3_simple.json, BFCL_v3_parallel_multiple.json
    - 指示文の言語指定が必要な問題には「英語で回答して」という指示を追加
        - 対象: BFCL_v3_live_parallel.json, BFCL_v3_parallel_multiple.json



---

## 4. オリジナル BFCL からの拡張

このセクションでは、BFCLをNejumi Leaderboardに統合するために行った具体的な変更について詳細に説明します。

- **大きなUpdate**
    
    - 推論系
        - 課題: ツールユースの実装がモデルごとに異なり、BFCLではモデルごとにclassを定義してその処理を記載していた。モデルごとにBFCL_model_idを作成し、それを指定する形になっていたが、そのような実装はモデル追加際に労力が発生する。特にOpen Weightのモデルは、学習時のプロンプトを把握する必要がある上、Hugging Faceのモデルカードからもそれを推論することは難しい
        - 解決策: 
            - OpenAIの統一モデルIDとして、"OpenAIResponsesHandler-FC"を作成
            - OpenRouterに対応 "OpenRouter-FC"
            - OSSのモデルについては統一のハンドラーの追加
                - **UnifiedOSSFCHandler**: vLLMのtool call対応モデル用の統一ハンドラー
                    - tokenizerがtools引数をネイティブサポートするモデル向け
                    - vLLMのtool_call_parser機能を活用した関数呼び出し処理
                    - OpenAI互換APIでの関数呼び出しを統一処理
                    - single-turn/multi-turn両対応
                - **UnifiedOSSJsonSchemaHandler**: tool call未対応モデル用の統一ハンドラー
                    - tool callをネイティブサポートしないモデル向け
                    - プロンプトエンジニアリングによりtool call機能を実現
                    - vLLMの`guided_json`を使用したJSONSchema制約によるStructured Output
                    - Pydantic + Genericsによる型安全な関数名制限
                    - モデル固有要件への柔軟対応（Gemma-2のsystem role制限、Mistralのtool_call_id要件など）
                    - 実行結果の表示形式を詳細にカスタマイズ可能
            - その他
                - llm-leadrboardで起動されるLLMのclientを利用するように一部統一
    - STEP limitを20から10に変更 (in bfcl/constants/default_prompts.py) 
        - STEPとは? : 試行回数
        - max tokenが小さいモデルに対応するため。実際、STEP10を超えて解決できない問題はそれ以降繰り返しても解決できない場合は過半数
    
        
- **その他細かいUpdate**
    - API model handlerの追加
        - geminiをgemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-liteに対応
        - mistrallをmistral-small-2503, mistrall-medium-2505, mistrall-large-2411に対応
    - `scripts/run_eval.py`にBFCL評価を統合
    - bfclをpackageとしてdownloadしないように変更。bfcl_pkg内の絶対インポートを相対インポートに変換
        - 特に大きな変更
            - llm-leaderboard/scripts/evaluator/evaluate_utils/bfcl_pkg/bfcl/constants/eval_config.py内のpathを変更
            - llm-leaderboard/scripts/evaluator/evaluate_utils/bfcl_pkg/bfcl/eval_checker/multi_turn_eval/func_source_code内のlong_context.pyを実行時にpathの問題で利用できないファイルがあったので、該当ファイルにlong_context.py内のプロンプトを追加
            - クラス名ベースの比較への変更
                - 問題：type()比較が異なるモジュールオブジェクトで失敗(packageの方法を踏襲しなかったので問題になった)
                - 修正：__class__.__name__による比較に変更
                - 対象ファイル：multi_turn_checker.pyと各APIクラスファイル
    - BFCL依存関係に伴うuv.lockの更新とuvベースの依存関係管理への移行
    - `scripts/evaluator/bfcl.py`の作成
    - WandBConfigSingletonとの統合
    - 設定の動的マージ（デフォルト + ユーザー設定）
    - テストモード対応（サンプル数制限）
    - WandB Artifactからのデータセット取得
    - 評価結果のWandBテーブル生成
    - W&BのTableに詳細な結果を残すために、出力されるscore fileにより詳細な情報が追加されるように変更(成功・失敗両方のテストケースで詳細情報を包含)
    - base_configへの設定パラメータの追加
    - データの整合性: 問題ディレクトリとpossible_answerディレクトリの両方で同じ順番が保たれる(sortをfalseにするなど)
    - Leading Zerosエラーの修正
        - 問題：Python 3での8進数解釈によるTypeError
        - 修正：正規表現によるleading zerosの10進数変換
        - 対象ファイル：multi_turn_utils.py


---


## 5. 評価 E2E フロー

```
┌──────────────────────────┐
│ 設定マージ & 初期化          │
│  - デフォルト設定取得       │
│  - ユーザー設定とマージ     │
│  - testmode対応           │
└──────────┬───────────────┘
           │ 設定完了
           ▼
┌──────────────────────────┐
│ WandB Artifact取得        │
│  - データセット取得        │
│  - ローカルにダウンロード   │
└──────────┬───────────────┘
           │ BFCL_v3_*.json
           ▼
┌──────────────────────────┐
│ 推論実行 (generation)      │
│  - 統合ハンドラー選択      │
│  - FC/JsonSchema対応      │
│  - 関数呼び出し生成        │
└──────────┬───────────────┘
           │ 推論結果 (JSONL)
           ▼
┌──────────────────────────┐
│ 評価実行 (evaluation)      │
│  - AST解析 & 実行チェック  │
│  - カテゴリ別スコア計算    │
│  - Overall accuracy算出   │
└──────────┬───────────────┘
           │ スコアファイル
           ▼
┌──────────────────────────┐
│ 結果処理 & WandB記録       │
│  - リーダーボードテーブル   │
│  - レーダーチャート        │
│  - 詳細ログテーブル        │
└──────────────────────────┘
           ↓
      BFCL 総合スコア
   （Overall Accuracy）
```

### 主要ステップの詳細

1. **設定マージ & 初期化**
   - `get_default_config()`でBFCLデフォルト設定を取得
   - ユーザー設定（`cfg.bfcl`）とマージして最終設定を生成
   - testmode時は`samples_per_category=2`に制限
   - モデルごとのconfigに

2. **WandB Artifact取得**
   - 設定された`artifacts_path`からデータセットをダウンロード
   - 各カテゴリの`BFCL_v3_*.json`ファイルを取得

3. **推論実行 (generation)**
   - モデルに応じた統合ハンドラーを選択
     - **UnifiedOSSFCHandler**: tool call対応モデル
     - **UnifiedOSSJsonSchemaHandler**: tool call未対応モデル  
   - 各テストケースに対して関数呼び出し生成

4. **評価実行 (evaluation)**
   - **AST解析フェーズ**
     - 生成されたコードの構文解析（Pythonのast.parseを使用）
     - 関数名・引数の正確性チェック
     - 型エラー・構文エラーの検出
   - **実行チェックフェーズ**
     - 実際の関数実行による動作確認
     - 実行結果と正解（possible_answer）の比較
     - タイムアウト・例外処理（セキュアな実行環境）
   - **並列・マルチターン対応**
     - 複数関数の同時実行（parallel category）
     - 対話的なマルチターン実行（multi_turn category）
     - 実行結果の依存関係チェック
   - カテゴリ別精度計算（Non-Live, Live, Multi-Turn等）

5. **結果処理 & WandB記録**
   - リーダーボードテーブル（Overall Acc等）
   - レーダーチャート（各カテゴリ精度）
   - 詳細ログテーブル（個別テストケース結果）

### 出力ファイル構成

- **結果ファイル**: `{result_dir}/{model_name}/BFCL_v3_{category}.json`
  - 各テストケースの推論結果（JSONL形式）
- **スコアファイル**: `{score_dir}/{model_name}/BFCL_v3_{category}_score.json`
  - 評価結果とエラー詳細情報

## 6. コンフィグ主要項目

### モデル設定での`bfcl_model_id`指定

各モデルのconfigファイルには、`bfcl_model_id`を記載する必要があります。利用可能なモデルIDは[SUPPORTED_MODELS.md](../scripts/evaluator/evaluate_utils/bfcl_pkg/SUPPORTED_MODELS.md)から選択してください。

#### モデルタイプ別の推奨設定

| モデルタイプ | 推奨`bfcl_model_id` | 備考 |
|-------------|------------------|------|
| **OSSモデル（FC対応）** | `unified-oss-fc` | tokenizerがtools引数をネイティブサポート |
| **OSSモデル（FC未対応）** | `unified-oss-jsonschema` | プロンプトエンジニアリング + JSONSchema制約 |
| **OpenRouter** | `OpenRouter-FC` | OpenRouter経由での関数呼び出し |
| **各プロバイダーAPI** | 各プロバイダー専用ID | Gemini、Mistral、Claude、OpenAI等の専用ハンドラー（OpenAI例: `OpenAIResponsesHandler-FC`、⚠️ 最新モデルが常にカバーされるとは限りません） |

### 基本設定例

```yaml
bfcl:
  test_category: "java javascript live_irrelevance live_multiple live_relevance live_simple multi_turn_base multi_turn_miss_func multi_turn_miss_param simple multiple irrelevance"  # multi_turn_long_context以外のカテゴリを指定
  temperature: 0.01  # 推論共通化未対応のHandler用
  num_threads: 2
  artifacts_path: 'llm-leaderboard/nejumi-leaderboard4/bfcl:production'
  generator_config:
    max_tokens: 8096
    temperature: 0.01
    top_p: 1.0
  handler_config:
    unified_oss_jsonschema:
      # ツールの実行結果メッセージにtool callの文字列 tool(arg1=1, arg2=2) を含めるかどうか
      execution_result_include_call_str: true
      # ツールの実行結果メッセージにtool_call_idを含めるかどうか　
      # trueの場合 "tool_call_id"のfieldが追加される
      # falseの場合はcontentの先頭部分に [1]: のように実行番号が追加される
      execution_result_include_call_id: false
      # 複数のツールが並列実行されたとき、ツールの実行結果メッセージを1つに連結するかどうか
      execution_result_join_parallel_calls: true
      # ツールの実行結果のrole "tool"が許可されていない場合は"user"などに変更する
      execution_result_role: "tool"
```
---

## 7. FAQ

* **出力の文字列が途中できれてJSONデコードエラーになるのはなぜ？**
    このエラーは主に**トークン制限による生成の途中停止**が原因です。

    1. **max_tokens設定の制限**
    - BFCLでは`generator.max_tokens`（デフォルト128）が設定されている
    - 入力プロンプトが長い場合、残りトークン数が少なくなる

    2. **動的トークン計算**
    - `_estimate_leftover_tokens_count`関数で入力トークン数を計算
    - `min(self.max_tokens, self.max_context_length - input_token_count - 2)`で制限

    3. **LLMの生成プロセス**
    - トークン制限に達すると、LLMは生成を途中で停止
    - 関数呼び出しのJSONが不完全な状態で終了

    ### 解決策

    - **max_tokensの増加**: BFCL用に十分なトークン数を設定
    - **プロンプトの最適化**: 入力トークン数を削減
    - **エラーハンドリングの改善**: 不完全なJSONを修復する機能の強化

