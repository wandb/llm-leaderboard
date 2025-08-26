# 目的
Nejumi Leaderboard4の結果を分析したい

# 対象ファイル
llm-leaderboard/scripts/analysis/Nejumi4_result_20280826.csv

# 環境構築
分析に必要な環境をuvで作って

# 分析1 : 日本語モデルの特徴
python file名: analyze_overall_ja_oss.py
- 日本のモデルの特徴について理解をしたい。
- JAの列で1のフラグが立っているものが日本のモデルである
- 同じ"model size category"について、以下のカテゴリに対して平均を撮った棒グラフを作って並べて 
    -TOTAL_SCORE, 汎用的言語性能(GLP)_AVG, アラインメント(ALT)_AVG
    - GLP_応用的言語性能 GLP_推論能力 GLP_知識・質問応答 GLP_基礎的言語性能 GLP_アプリケーション開発 GLP_表現 GLP_翻訳 GLP_情報検索 GLP_抽象的推論 GLP_論理的推論 GLP_数学的推論 GLP_一般的知識 GLP_専門的知識 GLP_意味解析 GLP_構文解析 GLP_コーディング GLP_関数呼び出し
    - ALT_制御性 ALT_倫理・道徳 ALT_毒性 ALT_バイアス ALT_真実性 ALT_堅牢性
    - なお、その中で日本語のモデルの平均を同じグラフの中で横並びにして

上記を実現するpython scriptを作成して


# 分析2 : Swallow analysis
python file名: compare_swallow_llama.py
- tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4とmeta-llama/Llama-3.3-70B-Instructを比較したい
- 以下のカテゴリに対して棒グラフを作って並べて 
    - GLP_応用的言語性能 GLP_推論能力 GLP_知識・質問応答 GLP_基礎的言語性能 GLP_アプリケーション開発 GLP_表現 GLP_翻訳 GLP_情報検索 GLP_抽象的推論 GLP_論理的推論 GLP_数学的推論 GLP_一般的知識 GLP_専門的知識 GLP_意味解析 GLP_構文解析 GLP_コーディング GLP_関数呼び出し, ALT_制御性 ALT_倫理・道徳 ALT_毒性 ALT_バイアス ALT_真実性 ALT_堅牢性
- tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4は濃い青, meta-llama/Llama-3.3-70B-Instructは薄い青
- 全てのラベルは黒色で統一
- tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4が勝っている項目を左側に配置し、負けている項目を右側に配置
- 各グループ内で差分の大きい順に並べて


上記を実現するpython scriptを作成して