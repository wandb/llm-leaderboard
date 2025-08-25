# SWE-Bench 評価ガイド（Nejumi LLM Leaderboard 4 版）

**TL;DR**

* **何を測る？**
  実在 OSS の“本物のバグ修正タスク”に対し、モデルが生成した **統一 diff（unified diff）** を適用 → **公式テスト**が全部通るかで自動採点。
* **Verified サブセット**
  入力・環境・期待テストが整備済みの安定版。ここでは日本語化した `nejumi/swe-bench-verified-ja` から、**7,000トークン未満の 80 件**を抽出して使用。
* **最重要の出力制約**
  生成パッチの **@@ ハンクヘッダには必ず行番号**（`@@ -start,count +start,count @@`）を入れること。これが欠けると適用に失敗。

---

## 1. SWE-Bench とは

* **概要**：OSS リポジトリの **実バグ修正**を題材にした **エンドツーエンド評価ベンチマーク**。
  **入力**（Issue/PR 文脈、関連ファイル抜粋、再現テスト 等）→ **出力**（統一 diff 形式のパッチ）。
  パッチを当ててテストを回し、**全テスト合格なら Resolved** と判定。

* **Verified 版**：
  評価に必要な **入力・環境・期待テスト**が整理され、**再現性と採点の安定性**が高いサブセット。

---

## 2. この評価セット（80 件）の作り方

* **目的**：ローカル LLM（\~8k コンテキスト）でも扱えるよう、**入力 7,000 トークン未満**で揃えつつ、
  元データ（500 件）の **難易度分布**と **GPT-4.1 の解決率（resolved 率）** にできるだけ近い構成へ。

* **ベース**：`nejumi/swe-bench-verified-ja`（500 件、`problem_statement`と`hint_text`はQwen/Qwen3-235B-A22Bで日本語化）

* **フィルタ**：`hf-internal-testing/llama-tokenizer`（Llama 2 相当）で概算し、`num_tokens < 7000` のみ採用

* **サンプリング**：`(difficulty, status)` を層化キーに抽出

  * `difficulty`: 4 区分
  * `status`: resolved / not resolved

* **プロンプト整形**：
  `text` 内の「I need you to solve the provided issue」の直前に、**@@ ハンクヘッダの行番号必須**（CRITICAL 文）を埋め込み済み

* **配布形態**：W\&B アーティファクト（Arrow 形式）

  * 名前：`swebench_verified_official`（バージョンで管理）
  * 本評価では **80 件版** を使用

---

## 3. 分布の比較（元 500 件 vs 80 件）

| 指標              | 元(500) | 80件(<7k) |
| --------------- | -----: | -------: |
| <15 min fix     |   38.8 |     46.2 |
| 15 min – 1 hour |   52.2 |     50.0 |
| 1–4 hours       |    8.4 |      3.8 |
| >4 hours        |    0.6 |      0.0 |
| Resolved率       |   34.6 |     37.5 |

**所感**：短時間で直る課題（<15 min）がやや増え、1–4 hours と >4 hours は縮小。
**Resolved 率**は **34.6% → 37.5%** と近似しており、全体傾向の再現性は良好です。

---

## 4. 評価 E2E フロー（全体像）

```
┌────────────────┐
│ Dataset (80件) │
└──────┬─────────┘
       │ 入力（Issue/PR文脈、関連抜粋、再現/期待テスト、制約）
       ▼
┌───────────────────────────────┐
│ 生成 (LLM)                     │
│  - prompt 整形（CRITICAL文）   │
│  - fc_enabled: 統一diffを強制  │
└──────────┬────────────────────┘
           │ 統一diff（@@ 行番号必須）
           ▼
┌───────────────────────────────┐
│ 前処理（拡張 & 正規化）         │
│  - Minimal patch 抽出           │
│  - ハンクヘッダ拡張             │
│  - ファイル名正規化/重複マージ  │
└──────────┬────────────────────┘
           │ パッチ適用（git apply / patch --fuzz）
           ▼
┌───────────────────────────────┐
│ 評価ランナー                    │
│  - 公式 or Docker               │
│  - 事前ビルドイメージ使用可     │
│  - ユニットテスト実行           │
└──────────┬────────────────────┘
           │ 成否
           ▼
     Resolved / Not Resolved
             ↓
       SWE-Bench スコア
   （Resolved 率 = 合格率）
```

---

## 5. 入出力仕様（実例）

### 入力

* 課題説明（日本語化済み）、関連ファイル抜粋、再現/期待テスト、制約
* 評価環境メタ（`evaluation_method: official|docker`、`prebuild_images: true` 等）

### 出力（統一 diff）

```diff
--- a/astropy/modeling/separable.py
+++ b/astropy/modeling/separable.py
@@ -245,1 +245,1 @@
-        cright[-right.shape[0]:, -right.shape[1]:] = 1
+        cright[-right.shape[0]:, -right.shape[1]:] = right
```

⚠️ **CRITICAL**

* **@@ ハンクヘッダに行番号必須**
* `@@ -start,count +start,count @@` の形式を守ること

---

## 6. 実行方式と環境

* **評価環境**：固定化された Docker イメージ上で実行
* **推論制御**：`generator.*` で温度やトークン数を調整
* **fc\_enabled: true** で **統一 diff を強制**

---

## 7. クイックスタート

### 依存関係

```bash
./myenv/bin/pip install fastapi "uvicorn[standard]" swebench
```

### API サーバー起動

```bash
nohup ./myenv/bin/python scripts/server/swebench_server.py \
  --host 0.0.0.0 --port 8000 \
  >/tmp/swebench_server.out 2>&1 & disown
```

### ジョブ送信例

```bash
PATCH_FILE=patch.diff
INSTANCE_ID=astropy__astropy-12907
curl -s -H "Content-Type: application/json" \
     -H "X-API-Key: $SWE_API_KEY" \
     -d @<(jq -n --arg iid "$INSTANCE_ID" --arg patch "$(cat "$PATCH_FILE")" \
       '{instance_id:$iid, patch_diff:$patch, namespace:"swebench", tag:"latest", model_name_or_path:"nejumi-api"}') \
     http://127.0.0.1:8000/v1/jobs
```

---

## 8. コンフィグ主要項目

```yaml
swebench:
  artifacts_path: llm-leaderboard/nejumi-leaderboard4/swebench_verified_official:production
  dataset_dir: swebench_verified_official
  max_samples: 80
  max_tokens: 2048
  max_workers: 4
  evaluation_method: docker
  prebuild_images: true
  fc_enabled: true
  api_server:
    enabled: false
```

---

## 9. オリジナル SWE-Bench からの拡張

* **ハンクヘッダ拡張**

  * ハンクヘッダの `pre_len` / `post_len` を **ctx\*2（既定 ctx=5）だけ増加**
  * 実際に本文へ文脈行を差し込むわけではなく、**行数を増やすことで適用許容度を拡張**
  * さらに `git apply` 失敗時は **`patch --fuzz=10` / `--fuzz=20`** を段階的に試行

* **Minimal patch 抽出**

  * 余計な文や無関係な差分を除去し、ヘッダの行番号も再計算
  * テスト合格に必要最小限の変更に絞る

* **CRITICAL 注意文の付加**

  * @@ ハンクヘッダの行番号必須を明示し、出力の安定性を担保

---

## 10. ベストプラクティス

* **diff のみを出力**（説明テキスト混入禁止）
* **@@ ヘッダ行番号必須**
* **Minimal patch 原則**
* `fc_enabled: true` を有効化

---

## 11. FAQ

* **Q. Resolved 判定は？**
  → 生成パッチ適用後に **全テストが合格**すれば Resolved。

* **Q. なぜヘッダ拡張？**
  → 軽微なズレや空白差分でも `git apply` が通るようにするため。
  → 本文行の挿入ではなく **ヘッダの行数だけを増やす**。

* **Q. fuzz はいつ使う？**
  → `git apply` が失敗したときに `patch --fuzz=10,20` を順に試行。
