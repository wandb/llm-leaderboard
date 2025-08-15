### SWE-Bench 評価 API サーバーの起動と使い方

前提:
- Docker がインストール済みで `docker` コマンドが実行できること
- SWE-Bench Verified の評価用イメージがプル済み（または必要に応じて自動プル）
- Python 仮想環境 `myenv` を使用

#### 1) 依存関係のインストール
```bash
./myenv/bin/pip install fastapi "uvicorn[standard]" swebench
```

#### 2) サーバー起動
```bash
nohup ./myenv/bin/python scripts/server/swebench_server.py --host 0.0.0.0 --port 8000 \
  >/tmp/swebench_server.out 2>&1 & disown
```
ログ確認:
```bash
tail -f /tmp/swebench_server.out | sed -n '1,120p'
```

環境変数（任意）:
- `SWE_API_KEY`: API キー（設定すると `X-API-Key` ヘッダ必須）

#### 3) ジョブ送信例
```bash
PATCH_FILE=patch.diff
INSTANCE_ID=astropy__astropy-12907
curl -s -H "Content-Type: application/json" -H "X-API-Key: $SWE_API_KEY" \
  -d @<(jq -n --arg iid "$INSTANCE_ID" --arg patch "$(cat "$PATCH_FILE")" \
    '{instance_id:$iid, patch_diff:$patch, namespace:"swebench", tag:"latest", model_name_or_path:"nejumi-api"}') \
  http://127.0.0.1:8000/v1/jobs
```

#### 4) ステータス/ログ参照
```bash
JOB_ID=job_XXXXXXXXXXXX
curl -s -H "X-API-Key: $SWE_API_KEY" http://127.0.0.1:8000/v1/jobs/$JOB_ID | jq
curl -s -H "X-API-Key: $SWE_API_KEY" http://127.0.0.1:8000/v1/jobs/$JOB_ID/logs
```
- 作業ディレクトリ: レスポンス `result.work_dir`
- 予測ファイル: `<work_dir>/predictions.jsonl`
- 評価レポート: カレントディレクトリに `nejumi-api.api_<timestamp>.json`
- ハーネスログ: `logs/run_evaluation/<run_id>/nejumi-api/<instance_id>/run_instance.log`

#### 5) 再現性の要点
- パッチは基本そのまま適用。`diff --git` ヘッダ補完＋末尾改行のみを行い、過剰なハンク改変はしません。
- ローカルハーネスと同じ `GIT_APPLY_CMDS`（`git apply`→`patch --fuzz`）が使われます。
- `open_file_limit=4096` を設定。

#### 6) CSV リプレイ（WANDBエクスポートの再現）
```bash
./myenv/bin/python scripts/tools/replay_swebench_csv.py \
  --csv wandb_export_YYYY-mm-ddTHH_MM_SS.csv \
  --endpoint http://127.0.0.1:8000 \
  --api-key "$SWE_API_KEY" \
  --namespace swebench --tag latest \
  --max 0 --out replay_results.csv
```
出力列: `instance_id, orig_status, api_status, match, job_id, report_path`

#### 7) よくあるエラー
- `Unauthorized`: `SWE_API_KEY` を設定したときは `X-API-Key` ヘッダ必須。
- `permission denied /var/run/docker.sock`: `sudo usermod -aG docker $USER` 後 `newgrp docker`。
- `unexpected end of file in patch`: 入力パッチに末尾改行が無い場合が多いので付与、または`diff --git`ヘッダを補完。