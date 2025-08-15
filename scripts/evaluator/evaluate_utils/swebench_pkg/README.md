### SWE-Bench API サーバー運用メモ

- 起動: `./swebench_pkg/start_server.sh --host 0.0.0.0 --port 8000`
  - `.env` があれば読み込み、`SWE_API_KEY` などを環境へ流し込みます
  - ログ: `/tmp/swebench_server.out`
- クリーンアップ: `./swebench_pkg/cleanup_temp.sh`
  - `/tmp/swebench_job_*` や `logs/run_evaluation/*` の古いものを削除

- Cloudflare Tunnel（固定ドメイン）: `docs/swebench_api_server.md` を参照
- 外部からの確認:
  - `POST /v1/jobs`, `GET /v1/jobs/{id}`, `GET /v1/jobs/{id}/logs`, `GET /v1/jobs/{id}/report`
  - 認証: ヘッダ `X-API-Key: $SWE_API_KEY`