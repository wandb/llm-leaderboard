# SWE-Bench Verified Docker環境での実行方法

## 概要

この文書では、nejumi4-devブランチのDocker環境でSWE-Bench Verified評価を実行する方法を説明します。

## Docker in Docker問題の解決

SWE-Bench評価はDockerコンテナを立ち上げて実行するため、Docker環境内で実行する場合は特別な設定が必要です。本実装では、Docker socketマウント方式を採用し、コンテナ内からホストのDockerデーモンを使用することで、Docker in Docker問題を回避しています。

## セットアップ

### 1. 環境変数の設定

`.env`ファイルに以下の環境変数を設定してください：

```bash
# 必須
WANDB_API_KEY=your_wandb_api_key
OPENAI_API_KEY=your_openai_api_key

# オプション（使用するAPIに応じて）
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key
# その他のAPI KEYs...
```

### 2. Docker権限の確認

ホストマシンでDockerが正常に動作することを確認してください：

```bash
docker ps
```

## 実行方法

### 1. Docker Composeを使用した実行

```bash
# 設定ファイルを指定して実行
docker-compose run --rm llm-leaderboard -c config-swebench-docker.yaml

# またはインタラクティブにコンフィグを選択
docker-compose run --rm llm-leaderboard -s
```

### 2. 独自の設定ファイルを作成

`configs/`ディレクトリに新しい設定ファイルを作成できます：

```yaml
# configs/config-swebench-custom.yaml
model:
  api_type: openai
  pretrained_model_name_or_path: gpt-4.1-2025-04-14
  max_seq_length: 32768

generator:
  temperature: 0.0
  top_p: 1.0
  max_tokens: 32768

wandb:
  entity: your-entity
  project: swebench-evaluation
  run_name: swebench-run-${timestamp}

run:
  swebench: true
  # 他の評価はfalseに設定

swebench:
  max_samples: 100  # 評価するサンプル数
  max_workers: 4    # 並列実行数
  evaluation_method: official  # 公式評価を使用
  artifacts_path: "your-entity/project/swebench-verified:latest"
  dataset_dir: swebench_verified
```

## 技術的な詳細

### Docker Socket Mount

`docker-compose.yaml`で以下の設定を行っています：

```yaml
volumes:
  - /var/run/docker.sock:/var/run/docker.sock
  - /etc/group:/etc/group:ro
  - /etc/passwd:/etc/passwd:ro

environment:
  - DOCKER_HOST=unix:///var/run/docker.sock
```

### 実行時権限設定

`docker-entrypoint.sh`が実行時に以下を行います：

1. Docker socketのGIDを取得
2. dockerグループを作成（必要な場合）
3. 実行ユーザーをdockerグループに追加

### パッチ適用の最適化

公式のパッチ適用コマンドに加えて、より高いfuzz値でのパッチ適用を試みます：

```python
extra_cmds = [
    "patch --batch --fuzz=10 -p1 -i",
    "patch --batch --fuzz=20 -p1 -i",
]
```

## トラブルシューティング

### Docker権限エラー

```
permission denied while trying to connect to the Docker daemon socket
```

このエラーが出る場合は、ホストのDockerグループにユーザーを追加してください：

```bash
sudo usermod -aG docker $USER
# ログアウト・ログインが必要
```

### メモリ不足

大規模な評価を実行する場合、Dockerのメモリ制限を増やしてください：

```yaml
# docker-compose.override.yml
services:
  llm-leaderboard:
    mem_limit: 32g
```

### タイムアウト

SWE-bench評価は時間がかかるため、必要に応じてタイムアウトを調整してください：

```yaml
swebench:
  timeout: 3600  # 秒単位
```

## ベストプラクティス

1. **小規模テストから開始**: `max_samples: 10`で動作確認してから本番実行
2. **並列数の調整**: システムリソースに応じて`max_workers`を調整
3. **ログの確認**: WandBダッシュボードで詳細な実行ログを確認
4. **定期的なクリーンアップ**: Dockerイメージとコンテナを定期的にクリーンアップ

```bash
# 未使用のコンテナを削除
docker container prune
# 未使用のイメージを削除
docker image prune
``` 