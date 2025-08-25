# SWE-Bench Verified 評価ガイド

## 概要

このガイドでは、Nejumi LLMリーダーボードでSWE-Bench Verified評価を実行する方法について説明します。SWE-Bench Verifiedは、GitHubの実際のissueを解決するコード生成能力を評価するベンチマークです。

## 前提条件

### システム要件
- Docker Engine (20.10以上)
- Python 3.9以上
- 十分なディスク容量（各評価で数GB必要）
- 十分なメモリ（8GB以上推奨）

### 環境変数の設定
```bash
export WANDB_API_KEY=<your_wandb_api_key>
export OPENAI_API_KEY=<your_openai_api_key>  # APIモデルを使用する場合
```

### 依存関係のインストール
```bash
pip install -r requirements.txt
```

## セットアップ手順

### 1. データセットのアップロード

まず、SWE-Bench Verifiedデータセットをダウンロードして、WandBにアップロードします。

```bash
# データセットを確認（dry-run）
python scripts/data_uploader/upload_swebench_verified.py \
    --entity your-entity \
    --project your-project \
    --dry-run

# 実際にアップロード
python scripts/data_uploader/upload_swebench_verified.py \
    --entity your-entity \
    --project your-project
```

成功すると、以下のようなアーティファクトパスが表示されます：
```
your-entity/your-project/swebench_verified:latest
```

### 2. 設定ファイルの更新

`configs/base_config.yaml`で、アーティファクトパスを更新します：

```yaml
swebench:
  artifacts_path: "your-entity/your-project/swebench_verified:latest"
  dataset_dir: "swebench"
  max_samples: 500
  max_tokens: 4096
  max_workers: 8
```

## 評価の実行

### テスト実行

まず、少数のサンプルでテストを実行します：

```bash
python scripts/run_eval.py -c config-swebench-test.yaml
```

### フル評価

全500件の評価を実行します：

```bash
python scripts/run_eval.py -c config-swebench-full.yaml
```

## 評価プロセス

### 1. データ準備
- SWE-Bench Verifiedデータセット（500インスタンス）を読み込み
- 各インスタンスには以下が含まれます：
  - 問題文（GitHub issue）
  - リポジトリ情報
  - ベースコミット
  - テストケース（FAIL_TO_PASS, PASS_TO_PASS）

### 2. プロンプト生成
- 問題文をLLMが理解しやすい形式に整形
- リポジトリ情報とヒントを含める
- diff形式でのパッチ生成を指示

### 3. LLM推論
- 各問題に対してパッチを生成
- 設定されたmax_tokensまでの出力を許可

### 4. Docker評価
- 各インスタンスごとに独立したDockerコンテナを起動
- リポジトリをクローンし、ベースコミットにチェックアウト
- 生成されたパッチを適用
- FAIL_TO_PASSテストとPASS_TO_PASSテストを実行
- 両方のテストセットが通った場合のみ「解決済み」とみなす

### 5. 結果集計
- リーダーボードテーブル
- 詳細出力テーブル
- リポジトリ別統計

## 結果の解釈

### メトリクス

- **解決率 (Resolution Rate)**: 完全に解決された問題の割合
- **パッチ適用率 (Application Rate)**: パッチが正常に適用された問題の割合
- **FAIL_TO_PASS率**: FAIL_TO_PASSテストが通った問題の割合
- **PASS_TO_PASS率**: PASS_TO_PASSテストが通った問題の割合

### WandBでの結果確認

評価完了後、以下のテーブルがWandBにログされます：

1. **swebench_leaderboard_table**: 総合統計
2. **swebench_output_table**: 各問題の詳細結果
3. **swebench_repo_breakdown_table**: リポジトリ別統計

## 設定パラメータ

### 重要な設定項目

- `max_samples`: 評価するサンプル数（1-500）
- `max_tokens`: パッチ生成の最大トークン数
- `max_workers`: 並列実行数
- `temperature`: 生成の創造性（通常は0.1で決定論的）

### パフォーマンス調整

- **並列実行数**: `max_workers`を調整してCPU/メモリ使用量を制御
- **バッチサイズ**: APIモデルの場合、レート制限に応じて調整
- **推論間隔**: `inference_interval`でAPI呼び出し間隔を制御

## トラブルシューティング

### よくある問題

1. **Dockerエラー**
   ```bash
   # Dockerデーモンの確認
   docker info
   
   # 権限の確認
   sudo usermod -aG docker $USER
   ```

2. **メモリ不足**
   - `max_workers`を減らす
   - Dockerコンテナのメモリ制限を調整

3. **ディスク容量不足**
   - 一時ディレクトリの容量を確認
   - 古いDockerイメージを削除

4. **タイムアウトエラー**
   - 個別のテストタイムアウトを調整
   - ネットワーク接続を確認

### ログの確認

```bash
# 評価ログの確認
tail -f /var/log/swebench_eval.log

# Dockerログの確認
docker logs <container_id>
```

## ベストプラクティス

1. **段階的評価**: まずテスト設定で少数サンプルを評価
2. **リソース監視**: CPU、メモリ、ディスク使用量を監視
3. **結果保存**: WandBで結果を適切に保存・管理
4. **再現性**: 同じ設定での再実行で結果が一貫することを確認

## 参考情報

- [SWE-Bench論文](https://arxiv.org/abs/2310.06770)
- [SWE-Bench Verified データセット](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)
- [公式リポジトリ](https://github.com/princeton-nlp/SWE-bench)

## サポート

問題や質問がある場合は、以下のリソースを確認してください：

1. このドキュメントのトラブルシューティングセクション
2. GitHubリポジトリのissues
3. WandBコミュニティフォーラム 

## 評価方法の選択

Nejumi LLMリーダーボードでは、2つの評価方法を提供しています：

### 1. 公式SWE-benchパッケージ使用（推奨）

```yaml
swebench:
  evaluation_method: "official"
```

- 公式のSWE-benchパッケージを直接使用
- Princeton NLPチームが開発した公式評価ロジック
- 最も正確で一貫性のある結果
- 自動的にパッケージをインストール

### 2. 独自Docker実装

```yaml
swebench:
  evaluation_method: "docker" 
```

- カスタムDocker環境での評価
- より詳細なログとデバッグ情報
- リソース使用量の細かい制御が可能
- 実験的機能やカスタマイズに適している

**推奨**: 公式パッケージ (`evaluation_method: "official"`) を使用してください。これにより、公式のSWE-Bench Verifiedと完全に互換性のある結果が得られます。 