
# Nejumi Leaderboardで行ったBFCLの変更と補足

## Nejumi Leaderboardのために行った変更
このセクションでは、BFCLをNejumi Leaderboardに統合するために行った具体的な変更について詳細に説明します。

- 評価データセットの日本語化とサンプリング
    - qwen/qwen3-235b-a22bを用いてベース翻訳。人手で修正も実施
    - llm-leaderboard/scripts/translation/bfcl_translation.pyを利用
        - **ルール**: 関数名、コード関連内容は翻訳対象外
    - llm-leaderboard/scripts/translation/bfcl_multi_turn_count.pyを用いて、Turn数を計算
    - llm-leaderboard/scripts/translation/sort_bfcl_file.pyを用いて並び替え
    - llm-leaderboard/scripts/data_uploader/upload_dataset.pyを用いてW&Bにupload
    - 詳細
        - BFCL_v3_live_irrelevance.json	882問 (ランダムに30問抽出)
            - 問題文に英語の質問が含む以下の問題を削除
        - BFCL_v3_irrelevance.json: 240問 (ランダムに30問抽出)
        - BFCL_v3_simple.json: 400問 (ランダムに30問抽出)
        - BFCL_v3_live_multiple.json 1,053問 (ランダムに30問抽出)
        - BFCL_v3_live_simple.json: 258問 (ランダムに30問抽出)
        - BFCL_v3_multiple.json: 200問 (ランダムに30問抽出)
        - BFCL_v3_parallel.json: 200問 (ランダムに30問抽出)
        - BFCL_v3_parallel_multiple.json: 200問 (ランダムに30問抽出)
            - 問題文に英語の質問が含む以下の問題を削除
        - BFCL_v3_java.json: 30問
        - BFCL_v3_javascript.json: 30問
        - BFCL_v3_live_parallel_multiple.json: 24問->21問
            - 問題文に英語以外の質問が含む以下の問題を削除
                - live_parallel_multiple_1-1-0
                - live_parallel_multiple_2-2-0
                - live_parallel_multiple_3-2-1
        - BFCL_v3_live_relevance.json: 18問
        - BFCL_v3_live_parallel.json: 16問
        - multi-turnの問題は3turn以下に絞り込み
            - BFCL_v3_multi_turn_base.json: 200問
            - BFCL_v3_multi_turn_long_context.json: 200問
            - BFCL_v3_multi_turn_miss_func.json: 200問
            - BFCL_v3_multi_turn_miss_param.json: 200問
    - データセットはWandBのartifactsに保存 [link](https://wandb.ai/llm-leaderboard/nejumi-leaderboard4/artifacts/dataset/bfcl)
    - Nejumi Leaderboardではサンプリングをして実装
        - 基本的に各カテゴリ30問を利用。30問に満たない問題は全問
        - live_parallel_multiple, live_multiple: 問題文に英語以外の質問が含む以下の問題を削除
            - live_parallel_multiple
                
            - parallel_multiple
                
        - 上記artifactsに保存するにあたり人手での翻訳確認の品質担保のため、以下の問題は50問に絞って保存
            - live_multiple, multiple, simple, parallel_multiple
        - possible answerに日本語のオプションを追加
            - live_multiple, live_parallel, multiple, simple, parallel_multiple
        - 指示文の言語指定をするべきと判断した問題に、英語で回答してという指示を追加
            - live_parallel, parallel_multiple
- `scripts/run_eval.py`にBFCL評価を統合
- BFCL依存関係に伴うuv.lockの更新とuvベースの依存関係管理への移行
- `scripts/evaluator/bfcl.py`の作成
  - WandBConfigSingletonとの統合
  - 設定の動的マージ（デフォルト + ユーザー設定）
  - テストモード対応（サンプル数制限）
  - WandB Artifactからのデータセット取得
  - 評価結果のWandBテーブル生成
- base_configへの設定パラメータの追加:
- bfclをpackageとしてdownloadしないように変更。bfcl_pkg内の絶対インポートを相対インポートに変換
- llm-leaderboard/scripts/evaluator/evaluate_utils/bfcl_pkg/bfcl/constants/eval_config.py内のpathを変更
- llm-leaderboard/scripts/evaluator/evaluate_utils/bfcl_pkg/bfcl/eval_checker/multi_turn_eval/func_source_code内のlong_context.pyを実行時にpathの問題で利用できないファイルがあったので、該当ファイルにlong_context.py内のプロンプトを追加
- W&Bへの結果表示
  - W&BのTableに詳細な結果を残すために、出力されるscore fileにより詳細な情報が追加されるように変更(成功・失敗両方のテストケースで詳細情報を包含)
- モデルごとのconfig fileにBFCLのmodel idを追加
- データの整合性: 問題ディレクトリとpossible_answerディレクトリの両方で同じ順番が保たれる(sortをfalseにするなど)
- クラス名ベースの比較への変更
    - 問題：type()比較が異なるモジュールオブジェクトで失敗(packageの方法を踏襲しなかったので問題になった)
    - 修正：__class__.__name__による比較に変更
    - 対象ファイル：multi_turn_checker.pyと各APIクラスファイル
- Leading Zerosエラーの修正
    - 問題：Python 3での8進数解釈によるTypeError
    - 修正：正規表現によるleading zerosの10進数変換
    - 対象ファイル：multi_turn_utils.py
- STEP limitを20から10に変更 (in bfcl/constants/default_prompts.py)
- llm-leadrboardで起動されるLLMのclientを利用するように変更


## モデルの選択方法


## 出力の文字列が途中できれてJSONデコードエラーになるのはなぜ？

このエラーは主に**トークン制限による生成の途中停止**が原因です。

### 主な原因

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

特に複雑な関数呼び出しや長い引数を持つ場合に発生しやすくなっています。
