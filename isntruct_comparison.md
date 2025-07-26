

https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard/bfcl_eval/eval_checker
と
/home/olachinkeigpu/Project/llm-leaderboard/scripts/evaluator/evaluate_utils/bfcl_pkg/bfcl/eval_checker
を比較し、違いをリストアップして。
なお、import pathは気にしないで

# 📊 ファイル比較結果詳細

## 🏗️ 全体的な構造
- ディレクトリ構造は同一
- ファイル名も同一
- **ローカル版のほとんどのファイルがGitHub版より行数が多い**

## 📝 ファイル毎の詳細比較

### 1. `__init__.py`
- **GitHub**: 空ファイル
- **ローカル**: 空ファイル
- **差分**: なし

### 2. `eval_runner.py`
- **GitHub**: 547行
- **ローカル**: 644行 (+97行)
- **主要な違い**:
  - import pathの変更（相対import vs 絶対import）
  - `get_handler(model_id, model_name)` vs `get_handler(model_name)`の関数シグネチャ
  - 追加のエラーハンドリングフィールド（`success`, `input_token_count`, `output_token_count`, `status`）

### 3. `eval_runner_helper.py`
- **GitHub**: 510行
- **ローカル**: 527行 (+17行)
- **主要な違い**:
  - import pathの変更
  - 追加の機能やエラーハンドリング

### 4. `ast_eval/ast_checker.py`
- **GitHub**: 636行
- **ローカル**: 641行 (+5行)
- **主要な違い**:
  - import pathの変更
  - 軽微な機能追加

### 5. `ast_eval/type_convertor/java_type_converter.py`
- **GitHub**: 407行
- **ローカル**: 408行 (+1行)
- **主要な違い**: 軽微な修正

### 6. `ast_eval/type_convertor/js_type_converter.py`
- **GitHub**: 311行
- **ローカル**: 312行 (+1行)
- **主要な違い**: 軽微な修正

### 7. `multi_turn_eval/multi_turn_checker.py`
- **GitHub**: 314行
- **ローカル**: 321行 (+7行)
- **主要な違い**:
  - import pathの変更
  - 機能追加

### 8. `multi_turn_eval/multi_turn_utils.py`
- **GitHub**: 156行
- **ローカル**: 179行 (+23行)
- **主要な違い**:
  - import pathの変更
  - 追加の機能

### 9. `multi_turn_eval/func_source_code/gorilla_file_system.py` ⚠️
- **GitHub**: 814行
- **ローカル**: 1,225行 (+411行)
- **重要な違い**:
  - 大量の拡張データが直接埋め込まれている
  - GitHub版では`long_context`からimportしているデータがローカルで直接定義
  - 拡張データ：詳細なファイルシステム操作データ

### 10. `multi_turn_eval/func_source_code/trading_bot.py` ⚠️
- **GitHub**: 734行
- **ローカル**: 1,052行 (+318行)
- **重要な違い**:
  - 大量の拡張データが直接埋め込まれている
  - GitHub版では`long_context`からimportしているデータがローカルで直接定義
  - 拡張データ：
    - `WATCH_LIST_EXTENSION`: 約1000個のアイテム
    - `TRANSACTION_HISTORY_EXTENSION`: 取引履歴データ
    - `MA_5_EXTENSION`: 移動平均データ

## 🔍 主要パターン

### 1. Import パスの体系的変更
- **GitHub**: `from bfcl_eval.constants.xxx import yyy`
- **ローカル**: `from ..constants.xxx import yyy`

### 2. 機能拡張
- ローカル版はより詳細なエラーハンドリング
- 追加のトークンカウント機能
- ステータス管理機能

### 3. データの埋め込み
**特に重要**: `gorilla_file_system.py`と`trading_bot.py`では、GitHub版が外部モジュールからimportしているデータを、ローカル版では直接ファイル内に巨大な配列として埋め込んでいる。

### 4. ファイルサイズ増加の主因
- 主に`func_source_code`ディレクトリ内の2つのファイルが原因
- GitHub版: 簡潔で外部依存
- ローカル版: 自己完結型で大容量データ内蔵

## 🎯 結論
ローカル版は GitHub版の機能拡張版であり、特に評価用のテストデータを大量に内蔵する自己完結型のバージョンとなっている。