# 評価実行ガイド

## クイックスタート

### 1. 基本的な使い方

```bash
# モデル名を指定して実行
./evaluate.sh Meta-Llama-3-8B-Instruct

# 部分一致でも可能
./evaluate.sh Llama-3-8B

# 設定ファイル名でも可能
./evaluate.sh config-Meta-Llama-3-8B-Instruct.yaml
```

### 2. 便利なオプション

```bash
# 利用可能なモデルを表示
./evaluate.sh --list

# オフラインモード（Wandbを使用しない）
./evaluate.sh Meta-Llama-3-8B-Instruct --offline

# デバッグ情報を表示
./evaluate.sh Meta-Llama-3-8B-Instruct --debug

# ヘルプを表示
./evaluate.sh --help
```

## 特徴

1. **シンプル**: モデル名を指定するだけで実行可能
2. **重複なし**: YAMLファイルからモデル名を自動取得
3. **柔軟**: 部分一致や設定ファイル名でも指定可能
4. **自動管理**: Dockerコンテナの起動・停止を自動化

## 仕組み

1. 指定されたモデル名から設定ファイルを検索
2. YAMLファイルからモデル情報を読み取り
3. 必要な環境変数を自動設定
4. Dockerコンテナを起動して評価を実行

## トラブルシューティング

### コンテナが起動しない場合
```bash
# 既存のコンテナを削除
docker rm -f llm-leaderboard llm-stack-vllm-1

# 再実行
./evaluate.sh Meta-Llama-3-8B-Instruct
```

### Wandbエラーが発生する場合
```bash
# オフラインモードで実行
./evaluate.sh Meta-Llama-3-8B-Instruct --offline
``` 