# Nejumi Leaderboard 4 分析スクリプト

このスクリプトは、Nejumi Leaderboard 4の結果を分析し、日本語モデルの特徴を可視化します。

## 概要

- **目的**: 日本語モデル（JA=1）の特徴分析
- **分析内容**: model size categoryごとの平均スコア比較
- **可視化**: 色分けされた棒グラフによる比較表示

## ファイル構成

```
scripts/analysis/
├── INSTRUCTION.md                # 分析要件
├── Nejumi4_result_20280826.csv   # データファイル
├── pyproject.toml                # 依存関係定義
├── analyze_overall_ja_oss.py     # メイン分析スクリプト
└── README.md                     # このファイル
```

## 環境構築

### 1. uvを使用した環境構築

```bash
# scripts/analysis ディレクトリに移動
cd scripts/analysis

# 仮想環境作成と依存関係インストール
uv sync

# 仮想環境をアクティベート
source .venv/bin/activate  # Linux/Mac
# または
.venv\Scripts\activate     # Windows
```

### 2. 手動インストール（uvがない場合）

```bash
pip install pandas matplotlib seaborn numpy
```

## 実行方法

```bash
# 分析スクリプト実行
python analyze_overall_ja_oss.py
```

## 出力

### 1. コンソール出力
- データセット概要（総モデル数、日本語モデル数など）
- モデルサイズカテゴリ別内訳
- 日本語モデル一覧
- 詳細統計（日本語モデル vs 全モデル平均）

### 2. 可視化チャート
- **ファイル名**: `nejumi4_analysis_chart.png`
- **内容**: model size categoryごとの性能比較チャート
- **バーの色分け**:
  - 🔷 **濃い青**: 日本語モデル平均
  - 🔹 **薄い青**: 全モデル平均
- **ラベルの色分け**:
  - 🟠 **オレンジ**: 日本語モデルが全モデル平均を上回る項目
  - ⚫ **黒**: 日本語モデルが全モデル平均を下回る項目

## 分析項目

### 主要メトリクス
- `TOTAL_SCORE`: 総合スコア
- `汎用的言語性能(GLP)_AVG`: GLP平均
- `アラインメント(ALT)_AVG`: ALT平均

### GLP系項目（17項目）
- GLP_応用的言語性能、GLP_推論能力、GLP_知識・質問応答
- GLP_基礎的言語性能、GLP_アプリケーション開発、GLP_表現
- GLP_翻訳、GLP_情報検索、GLP_抽象的推論
- GLP_論理的推論、GLP_数学的推論、GLP_一般的知識
- GLP_専門的知識、GLP_意味解析、GLP_構文解析
- GLP_コーディング、GLP_関数呼び出し

### ALT系項目（6項目）
- ALT_制御性、ALT_倫理・道徳、ALT_毒性
- ALT_バイアス、ALT_真実性、ALT_堅牢性

### ラベルの動的色分け
- **オレンジ色のラベル**: 日本語モデルが全モデル平均を上回る項目
- **黒色のラベル**: 日本語モデルが全モデル平均を下回る項目

## モデルサイズカテゴリ

- **Small (<10B)**: 100億パラメータ未満
- **Medium (10–30B)**: 100億〜300億パラメータ
- **Large (30B+)**: 300億パラメータ以上
- **Large (>30B)**: 300億パラメータ超
- **api**: APIモデル

## 日本語モデル（JA=1）

CSVの`JA`列が1のモデルが日本語モデルとして分析されます。

## 依存関係

- **pandas**: データ処理
- **matplotlib**: 基本的な可視化
- **seaborn**: 美しい統計グラフ
- **numpy**: 数値計算

## トラブルシューティング

### 日本語フォントが表示されない場合

```python
# 利用可能なフォントを確認
import matplotlib.font_manager as fm
fonts = [f.name for f in fm.fontManager.ttflist]
print([f for f in fonts if 'Gothic' in f or 'Sans' in f])
```

### データファイルが見つからない場合

`Nejumi4_result_20280826.csv`がscripts/analysisディレクトリに存在することを確認してください。

## 結果の解釈

- **薄い青バー**: 全モデル平均（全体のベンチマーク）
- **濃い青バー**: 日本語モデル平均（日本語特化モデルの性能）
- **オレンジ色ラベル**: 日本語モデルが全体平均を上回る項目
- **黒色ラベル**: 日本語モデルが全体平均を下回る項目
- **差分**: 正の値は日本語モデルが優秀、負の値は全体平均以下

## カスタマイズ

スクリプト内の以下の部分を修正することで、分析をカスタマイズできます：

- `get_metric_groups()`: 色分けグループの変更
- `create_comparison_chart()`: チャートの見た目調整
- `calculate_averages_by_category()`: 集計方法の変更
