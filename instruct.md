# BFCL (Berkeley Function Calling Leaderboard) アップデート手順書

このドキュメントは、Gorilla BFCLリポジトリからの変更を現在のプロジェクトに反映させるための手順をまとめています。

## 📋 事前準備

### 1. 対象ディレクトリの確認
```
/home/olachinkeigpu/Project/llm-leaderboard/scripts/evaluator/evaluate_utils/bfcl_pkg
```

### 2. 主要ファイル構造
```
bfcl/
├── constants/
│   ├── model_config.py           # モデル設定
│   └── supported_models.py       # サポートモデル一覧
├── model_handler/
│   ├── api_inference/           # API推論ハンドラー
│   └── local_inference/         # ローカル推論ハンドラー
└── eval_checker/               # 評価機能
```

## 🔍 STEP 1: PR情報の収集

### 1.1 PRの特定
- GitHub URLからPR番号を取得
- https://github.com/ShishirPatil/gorilla/pull/1056/files


### 1.2 変更内容の確認
```bash
curl -s "https://api.github.com/repos/ShishirPatil/gorilla/pulls/PR番号/files" | head -100
```
### 1.3 変更ファイルの分類
- 📄 **新規追加**: APIハンドラー、モデル設定など
- 📝 **既存修正**: import文、設定追加など
- 🗑️ **削除**: 廃止ファイル、重複コードなど

## 🛠️ STEP 2: 変更の追加などの対応
### 注意点
- 元のパス: `from bfcl_eval.model_handler...`
- 現在のパス: `from ..model_handler...`
- **注意**: 相対importに変更すること。localの中のファイルを参考にして

## 🗂️ STEP 3: ファイルの整理
- 重複ファイルの削除などを行なってください
- 重複import文の削除
- 古いファイルからのimportを削除
- 新しいファイルからのimportが正しく設定されているか確認

## ✅ STEP 4: 検証
- 構文チェック
```bash
python3 -m py_compile bfcl/model_handler/api_inference/新しいファイル.py　
python3 -m py_compile bfcl/constants/model_config.py
python3 -m py_compile bfcl/constants/supported_models.py
```

## 

## 📚 注意事項
### ⚠️ 重要なポイント
1. **importパス**: 必ず相対パス（`..`）に変更
2. **ファイル命名**: 元のファイル名を維持

### 🚫 避けるべきこと
- import文の重複
- 古いファイルの放置
- テストを怠る

## 📝 作業ログテンプレート

```markdown
## [日付] BFCL更新: PR #番号

### 変更内容
- [ ] 新規APIハンドラー追加: `ファイル名.py`
- [ ] モデル設定追加: `モデル数`個
- [ ] サポートモデル更新
- [ ] 古いファイル削除: `ファイル名`

### 検証結果  
- [ ] 構文チェック: OK
- [ ] 設定確認: OK
- [ ] ファイル整理: OK

### 備考
特記事項があれば記載
```

---

このドキュメントに従って作業することで、今後のBFCLアップデートを効率的かつ安全に実行できます。

## 📋 作業履歴

### [2025-01-22] BFCL更新: PR #1063

#### 変更内容
- [x] モデル設定更新: DeepSeek-R1-0528、DeepSeek-V3-0324対応
- [x] サポートモデル更新: 3個のモデル追加
- [x] APIハンドラー更新: deepseek-reasoner対応
- [x] 重複import文削除: model_config.py
- [x] CHANGELOG更新: PR #1063エントリ追加

#### 検証結果  
- [x] 構文チェック: OK (model_config.py, supported_models.py, deepseek.py)
- [x] 設定確認: OK (DeepSeek-R1-0528: 2個, ModelConfig: 4個, deepseek-reasoner: 追加済み)
- [x] ファイル整理: OK

#### 備考
- 参照: https://github.com/ShishirPatil/gorilla/pull/1063/files
- DeepSeek-R1 → DeepSeek-R1-0528 への更新
- DeepSeek-V3-FC → DeepSeek-V3-0324-FC への更新
- 新規追加: DeepSeek-R1-0528-FC
- コミット: faa1d17

### [2025-01-22] BFCL更新: PR #1060

#### 変更内容
- [x] バグ修正: `_get_item()`メソッドで"."ディレクトリの処理を追加
- [x] ファイル修正: gorilla_file_system.py

#### 検証結果  
- [x] 構文チェック: OK (gorilla_file_system.py)
- [x] 変更確認: OK (_get_itemメソッドに"."処理追加)

#### 備考
- 参照: https://github.com/ShishirPatil/gorilla/pull/1060/files
- 修正内容: `if item_name == ".": return self`を追加
- 目的: "."ディレクトリ参照時の適切な処理を実現

### [2025-01-22] BFCL更新: PR #1056

#### 変更内容
- [x] 新規APIハンドラー追加: `ling.py`
- [x] モデル設定追加: `Ling/ling-lite-v1.5`
- [x] サポートモデル更新: 1個のモデル追加
- [x] SUPPORTED_MODELS.md更新: Ling-lite-v1.5エントリ追加
- [x] env.example更新: LING_API_KEY追加
- [x] import文追加: model_config.pyにLingAPIHandler

#### 検証結果  
- [x] 構文チェック: OK (ling.py, model_config.py, supported_models.py)
- [x] 設定確認: OK (Ling/ling-lite-v1.5: 追加済み, LingAPIHandler: 参照済み)
- [x] ファイル整理: OK

#### 備考
- 参照: https://github.com/ShishirPatil/gorilla/pull/1056/files
- 追加モデル: Ling-lite-v1.5 (Prompt)
- 特徴: Ant Group提供のLingモデル、プロンプトベース推論
- APIベース: https://bailingchat.alipay.com 経由での推論
- 継承関係: OpenAIHandler → LingAPIHandler
- 環境変数: LING_API_KEY (env.exampleに追加済み)

### [2025-01-22] BFCL更新: PR #1032

#### 変更内容
- [x] 新規APIハンドラー追加: `nemotron.py`
- [x] モデル設定追加: `nvidia/llama-3.1-nemotron-ultra-253b-v1`
- [x] サポートモデル更新: 1個のモデル追加
- [x] import文追加: model_config.pyにNemotronHandler

#### 検証結果  
- [x] 構文チェック: OK (nemotron.py, model_config.py, supported_models.py)
- [x] 設定確認: OK (nvidia/llama-3.1-nemotron-ultra-253b-v1: 追加済み, NemotronHandler: 参照済み)
- [x] ファイル整理: OK

#### 備考
- 参照: https://github.com/ShishirPatil/gorilla/pull/1032/files
- 追加モデル: Llama-3.1-Nemotron-Ultra-253B-v1 (Prompt)
- 特徴: XML形式の`<TOOLCALL>`タグを使用したツールコール処理
- APIベース: NVIDIA API経由での推論
- 継承関係: NvidiaHandler → NemotronHandler (正しい継承に修正済み)

### [2025-01-22] BFCL確認: PR #1061

#### 確認内容
- [x] Qwen3シリーズDashScope API推論サポート: 既に実装済み
- [x] ストリーミング機能: qwen.pyに実装済み
- [x] 推論コンテンツ処理: reasoning_content機能実装済み
- [x] QwqHandler統合: qwq.pyファイル削除済み、QwenAPIHandlerに統合
- [x] importパス修正: 絶対パス → 相対パス (bfcl_eval.model_handler... → ..model_handler...)

#### 検証結果  
- [x] 構文チェック: OK (qwen.py)
- [x] QwQ-32Bモデル設定: QwenAPIHandler使用に更新済み
- [x] ストリーミング機能: FC/Promptモード両対応
- [x] importパス: 相対パスに修正済み

#### 備考
- 参照: https://github.com/ShishirPatil/gorilla/pull/1061/files
- 対象: Qwen3シリーズとQwQシリーズのDashScope API統合
- 特徴: enable_thinking=True によるストリーミング推論処理
- 状況: 既に実装済み、importパスのみ修正