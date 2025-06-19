# BFCL JSONファイル並び替えスクリプト

このディレクトリには、BFCLで始まるJSONファイルをidフィールドで並び替えるためのPythonスクリプトが含まれています。

## ファイル説明

### 1. `sort_bfcl_files.py`
- **機能**: BFCL JSONファイルをidフィールドで並び替えて、新しいファイル（`_sorted`サフィックス付き）に保存
- **特徴**: 元のファイルは変更されません
- **出力**: `BFCL_v3_simple.json` → `BFCL_v3_simple_sorted.json`

### 2. `sort_bfcl_files_overwrite.py`
- **機能**: BFCL JSONファイルをidフィールドで並び替えて、元のファイルを上書き
- **特徴**: 元のファイルを直接変更します（バックアップ機能付き）
- **安全性**: 処理前にバックアップを作成し、エラー時は自動復元

## 使用方法

### 基本的な使用方法（元ファイルを保持）

```bash
cd /home/olachinkeigpu/Project/llm-leaderboard/scripts/evaluator/evaluate_utils/bfcl_pkg/data_jp
python3 sort_bfcl_files.py
```

### 元ファイルを上書きする場合

```bash
cd /home/olachinkeigpu/Project/llm-leaderboard/scripts/evaluator/evaluate_utils/bfcl_pkg/data_jp
python3 sort_bfcl_files_overwrite.py
```

## 並び替えルール

スクリプトは以下のルールでidフィールドを並び替えます：

1. **文字列IDの場合**: 
   - 数値部分を抽出して数値として比較
   - 例: `simple_186`, `simple_395` → `simple_186`が先
   - 例: `multiple_12`, `multiple_15` → `multiple_12`が先

2. **数値IDの場合**: 
   - 数値として直接比較

3. **プレフィックス付きIDの場合**:
   - プレフィックス（`simple_`, `multiple_`など）でグループ化
   - 各グループ内で数値部分で並び替え

## 処理対象ファイル

以下のファイルが処理対象となります：
- `BFCL_v3_simple.json`
- `BFCL_v3_multiple.json`
- `BFCL_v3_multi_turn_long_context.json`
- `BFCL_v3_multi_turn_miss_func.json`
- `BFCL_v3_multi_turn_miss_param.json`
- `BFCL_v3_live_relevance.json`
- `BFCL_v3_live_irrelevance.json`
- `BFCL_v3_live_simple.json`
- `BFCL_v3_live_parallel_multiple.json`
- `BFCL_v3_live_parallel.json`
- `BFCL_v3_live_multiple.json`
- `BFCL_v3_javascript.json`
- `BFCL_v3_multi_turn_base.json`
- `BFCL_v3_parallel.json`
- `BFCL_v3_irrelevance.json`
- `BFCL_v3_parallel_multiple.json`
- `BFCL_v3_java.json`

## 出力例

```
見つかったBFCLファイル: 18件
  - BFCL_v3_simple.json
  - BFCL_v3_multiple.json
  - BFCL_v3_multi_turn_long_context.json
  ...

並び替えを開始します...
✓ BFCL_v3_simple.json -> BFCL_v3_simple_sorted.json (401件)
✓ BFCL_v3_multiple.json -> BFCL_v3_multiple_sorted.json (201件)
...

完了: 18/18件のファイルを正常に処理しました。
すべてのファイルが正常に並び替えられました。
```

## 注意事項

1. **ファイル形式**: 各行が独立したJSONオブジェクトである必要があります
2. **エンコーディング**: UTF-8でエンコードされている必要があります
3. **idフィールド**: 各JSONオブジェクトに`id`フィールドが含まれている必要があります
4. **バックアップ**: 上書き版を使用する場合は、処理前にバックアップが作成されます

## トラブルシューティング

### JSON解析エラー
- ファイル内に不正なJSON行がある場合、警告が表示されます
- 該当行はスキップされ、処理は続行されます

### ファイルが見つからない
- スクリプトは`BFCL_*.json`パターンでファイルを検索します
- ファイル名が正しい形式であることを確認してください

### 権限エラー
- ファイルの読み書き権限を確認してください
- 必要に応じて`chmod`コマンドで権限を変更してください 