

# Task: BFCLにOSSモデルを簡単に追加できるようにしたい
## ✅ 完了状況

### ✅ handlerの作成
- `/home/olachinkeigpu/Project/llm-leaderboard/scripts/evaluator/evaluate_utils/bfcl_pkg/bfcl/model_handler/local_inference/unified_oss_handler.py` **完成**
- 全てのlocal_inferenceハンドラーを一つのファイルに統合
- 新しいOSSモデルは `bfcl_model_id: "oss_handler"` と設定するだけで使用可能

### ✅ デコード機能の自動対応
- 以下の9つの出力パターンに自動対応（モデル名での判断は不要）：

#### 1. 標準JSONパターン (Hammer系)
```json
[{"name": "func_name", "arguments": {"arg1": "val1"}}]
```

#### 2. Markdownコードブロック内JSON (DeepSeek系)
```
```json
[{"name": "func", "arguments": {"arg": "value"}}]
```
```

#### 3. XMLタグパターン (Hermes系)
```xml
<tool_call>
{"name": "func", "arguments": {"arg": "value"}}
</tool_call>
```

#### 4. 特殊タグパターン (Llama 3.1系)
```
<|python_tag|>{"name": "func", "arguments": {"arg": "val"}}; {"name": "func2", ...}
```

#### 5. 関数呼び出しタグパターン (Granite系)
```
<function_call> {"name": "func", "arguments": {"arg": "value"}}
```

#### 6. 複雑な思考タグパターン (MiniCPM系)
```
<|thought_start|>思考過程<|thought_end|>
<|tool_call_start|>
```python
func(arg=value)
```
<|tool_call_end|>
```

#### 7. 改行区切りパターン (GLM系)
```
func_name
{"arg1": "val1"}
```

#### 8. 単純なJSONオブジェクト
```json
{"name": "func", "arguments": {"arg": "value"}}
```

#### 9. デフォルトのAST解析 (最後の手段)

### ✅ 前処理のchat template自動取得
- config_singletonから自動取得
```python
from config_singleton import WandbConfigSingleton
instance = WandbConfigSingleton.get_instance()
cfg = instance.config
model_local_path = cfg.model.get("local_path", None)
chat_template_name = cfg.model.get("chat_template")
local_chat_template_path = Path(f"chat_templates/{chat_template_name}.jinja")
```

### ✅ 生の出力保存機能
- 全てのモデル出力を `scripts/evaluator/evaluate_utils/bfcl_pkg/result/{model_name}/raw_outputs_debug.txt` に保存
- デコードパターンの分析やデバッグに使用可能
- 次にどのような処理を組み込むべきかを理解できる

### ✅ model_config.pyの更新
- `"oss_handler"` エントリを追加
- `model_handler=UnifiedOSSHandler` を指定

### ✅ 使用方法の文書化

## 📋 新しいOSSモデルの追加方法（簡単版）

### 1. 設定ファイルの作成
```yaml
# configs/config-your-new-model.yaml
model:
  pretrained_model_name_or_path: your-org/your-model-name
  bfcl_model_id: "oss_handler"  # ★これだけでOK！★
  chat_template: your-org/your-model-name
```

### 2. Chat templateファイルの作成
```bash
# chat_templates/your-org_your-model-name.jinja を作成
```

### 3. 評価実行
```bash
python3 scripts/run_eval.py -c config-your-new-model.yaml
```

## 🔄 既存の複雑な手順との比較

### ❌ 従来の方法（回避された）
1. ✅ ~~`bfcl/model_handler/local_inference/base_oss_handler.py`を確認~~
2. ✅ ~~新しいモデルの新しいhandler classを作成~~
3. ✅ ~~`bfcl/constants/model_config.py`に新しいモデルの情報を追加~~
4. ✅ ~~modelごとのconfig内のbfcl_model_nameに追加したモデル名を記載~~

### ✅ 新しい方法（統合ハンドラー使用）
1. 設定ファイルで `bfcl_model_id: "oss_handler"` を指定
2. Chat templateファイルを作成
3. 完了！

## 🛠️ 技術的詳細

### モデル特徴の自動検出
- モデル名から自動的に特徴を推定
- DeepSeek、Llama、Qwen、Gemma、Phi、MiniCPM、GLM、Granite、Hermes等に対応
- 推論機能（reasoning）の有無も自動検出

### 前処理の自動適用
- DeepSeek系: システムプロンプトのユーザープロンプト変換
- Gemma: assistantロールのmodelロール置換
- FCモデル: 独自システムプロンプト使用

### 実行結果処理の自動選択
- Llama系: ipythonロール使用
- DeepSeek系: userロール使用（toolロール非対応）
- 標準: toolロール使用

## 📁 ファイル構成

```
scripts/evaluator/evaluate_utils/bfcl_pkg/bfcl/
├── model_handler/local_inference/
│   └── unified_oss_handler.py  # ✅ 統合OSSハンドラー
├── constants/
│   └── model_config.py  # ✅ "oss_handler"エントリ追加済み
└── result/
    └── {model_name}/
        └── raw_outputs_debug.txt  # ✅ 生の出力ログ
```

```
configs/
├── config-sample-new-oss-model.yaml  # ✅ サンプル設定
└── config-Meta-Llama-3-2-1B-Instruct.yaml  # ✅ 統合ハンドラー使用例
```

## 🚀 追加の利点

1. **メンテナンスの簡単さ**: 新しい出力パターンが見つかった場合、unified_oss_handler.pyの1箇所を更新するだけ
2. **デバッグ機能**: 生の出力が自動保存されるため、問題の原因分析が容易
3. **拡張性**: 新しいパターンの追加が簡単
4. **後方互換性**: 既存の個別ハンドラーも引き続き使用可能

## ✅ gitによる変更管理
- 全ての変更がgitで管理されており、いつでも戻ることが可能
- `.gitignore`に`result/`ディレクトリを追加済み

---

## ✅ **タスク完了**
BFCLにOSSモデルを簡単に追加できる統合ハンドラーシステムが完成しました！
