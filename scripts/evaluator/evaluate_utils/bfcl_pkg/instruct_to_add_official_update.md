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
https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/CHANGELOG.md

### 1.1 PRの特定
- GitHub URLからPR番号を取得
- https://github.com/ShishirPatil/gorilla/pull/1100/files
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
- 構文チェック（一気に実装して）
```bash
python3 -m py_compile bfcl/model_handler/api_inference/新しいファイル.py　
python3 -m py_compile bfcl/constants/model_config.py
python3 -m py_compile bfcl/constants/supported_models.py
```

## STEP5: 変更をgit addしてcommitして

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
