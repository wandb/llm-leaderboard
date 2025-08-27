#!/usr/bin/env python3
"""
Nejumi Leaderboard 4 結果分析スクリプト

日本語モデルの特徴を分析し、model size category ごとの平均スコアを可視化します。
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

def load_data(csv_path: str) -> pd.DataFrame:
    """CSVファイルを読み込む"""
    return pd.read_csv(csv_path)

def get_metric_groups() -> Dict[str, List[str]]:
    """メトリクスを色別にグループ化"""
    return {
        'orange': [
            'TOTAL_SCORE',
            '汎用的言語性能(GLP)_AVG',
            'アラインメント(ALT)_AVG'
        ],
        'blue': [
            'GLP_応用的言語性能', 'GLP_推論能力', 'GLP_知識・質問応答', 'GLP_基礎的言語性能',
            'GLP_アプリケーション開発', 'GLP_表現', 'GLP_翻訳', 'GLP_情報検索',
            'GLP_抽象的推論', 'GLP_論理的推論', 'GLP_数学的推論', 'GLP_一般的知識',
            'GLP_専門的知識', 'GLP_意味解析', 'GLP_構文解析', 'GLP_コーディング', 'GLP_関数呼び出し'
        ],
        'green': [
            'ALT_制御性', 'ALT_倫理・道徳', 'ALT_毒性', 'ALT_バイアス', 'ALT_真実性', 'ALT_堅牢性'
        ]
    }

def calculate_averages_by_category(df: pd.DataFrame, metrics: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    model_size_categoryごとに平均を計算
    
    Returns:
        all_models_avg: 全モデルの平均
        ja_models_avg: 日本語モデル（JA=1）の平均
    """
    # 欠損値を除外してグループ化
    df_clean = df.dropna(subset=['model_size_category'])
    
    # 全モデルの平均
    all_models_avg = df_clean.groupby('model_size_category')[metrics].mean()
    
    # 日本語モデル（JA=1）の平均
    ja_models = df_clean[df_clean['JA'] == 1]
    ja_models_avg = ja_models.groupby('model_size_category')[metrics].mean()
    
    return all_models_avg, ja_models_avg

def create_comparison_chart(all_avg: pd.DataFrame, ja_avg: pd.DataFrame, 
                          metric_groups: Dict[str, List[str]], output_path: str, 
                          df: pd.DataFrame):
    """比較チャートを作成"""
    
    # カテゴリの順序を定義
    size_categories = ['Small (<10B)', 'Medium (10–30B)', 'Large (30B+)', 'Large (>30B)', 'api']
    
    # 色の定義（青系で統一）
    colors = {
        'dark_blue': '#1E5799',   # 濃い青（日本語モデル用）
        'light_blue': '#7DB9E8'   # 薄い青（全モデル用）
    }
    
    # 全メトリクスをまとめる（色は青系で統一）
    all_metrics = []
    for metrics in metric_groups.values():
        all_metrics.extend(metrics)
    
    # データが存在するカテゴリのみを使用
    available_categories = [cat for cat in size_categories if cat in all_avg.index]
    
    # 図のサイズを調整
    fig, axes = plt.subplots(len(available_categories), 1, figsize=(24, 10 * len(available_categories)))
    if len(available_categories) == 1:
        axes = [axes]
    
    for idx, category in enumerate(available_categories):
        ax = axes[idx]
        
        # 各カテゴリのモデル数を計算
        df_clean = df.dropna(subset=['model_size_category'])
        total_models_in_category = len(df_clean[df_clean['model_size_category'] == category])
        ja_models_in_category = len(df_clean[(df_clean['model_size_category'] == category) & (df_clean['JA'] == 1)])
        
        # データを準備
        all_values = []
        ja_values = []
        labels = []
        label_colors = []  # ラベルの色を格納
        
        for metric in all_metrics:
            if metric in all_avg.columns:
                all_val = all_avg.loc[category, metric] if category in all_avg.index else np.nan
                ja_val = ja_avg.loc[category, metric] if category in ja_avg.index else np.nan
                
                if not (np.isnan(all_val) and np.isnan(ja_val)):
                    all_val_clean = all_val if not np.isnan(all_val) else 0
                    ja_val_clean = ja_val if not np.isnan(ja_val) else 0
                    
                    all_values.append(all_val_clean)
                    ja_values.append(ja_val_clean)
                    # prefixを残したラベル名
                    labels.append(metric)
                    
                    # 日本語モデルが上回っている場合はオレンジ色、そうでなければ黒色
                    if ja_val_clean > all_val_clean:
                        label_colors.append('#FF6600')  # オレンジ色
                    else:
                        label_colors.append('#000000')  # 黒色
        
        if not all_values:
            continue
            
        # バーの位置
        x = np.arange(len(labels))
        width = 0.35
        
        # バーチャート作成（青系で統一）
        bars1 = ax.bar(x - width/2, all_values, width, 
                      label=f'全モデル平均 (n={total_models_in_category})', 
                      color=colors['light_blue'])
        bars2 = ax.bar(x + width/2, ja_values, width, 
                      label=f'日本語モデル平均 (n={ja_models_in_category})', 
                      color=colors['dark_blue'])
        
        # グラフの設定
        ax.set_title(f'{category} - モデル性能比較\n'
                    f'全モデル: {total_models_in_category}個, 日本語モデル: {ja_models_in_category}個', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('評価項目', fontsize=14)
        ax.set_ylabel('スコア', fontsize=14)
        ax.set_xticks(x)
        # ラベルの色を個別に設定
        tick_labels = ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
        for i, (label, color) in enumerate(zip(tick_labels, label_colors)):
            label.set_color(color)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        # 統合された凡例を作成
        ax.legend(loc='upper right', fontsize=10)
        
        # 値をバーの上に表示
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            if all_values[i] > 0:
                ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.01, 
                       f'{all_values[i]:.3f}', ha='center', va='bottom', fontsize=9, color='gray')
            if ja_values[i] > 0:
                ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.01, 
                       f'{ja_values[i]:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_statistics(df: pd.DataFrame):
    """要約統計を表示"""
    print("="*80)
    print("データセット概要")
    print("="*80)
    
    total_models = len(df)
    ja_models = len(df[df['JA'] == 1])
    
    print(f"総モデル数: {total_models}")
    print(f"日本語モデル数: {ja_models}")
    print(f"日本語モデル比率: {ja_models/total_models:.1%}")
    
    print("\nモデルサイズカテゴリ別内訳:")
    category_counts = df['model_size_category'].value_counts()
    for category, count in category_counts.items():
        ja_count = len(df[(df['model_size_category'] == category) & (df['JA'] == 1)])
        print(f"  {category}: {count}モデル (日本語: {ja_count})")
    
    print("\n日本語モデル一覧:")
    ja_models_list = df[df['JA'] == 1]['model_name'].tolist()
    for model in ja_models_list:
        print(f"  - {model}")

def main():
    """メイン処理"""
    # ファイルパス設定
    script_dir = Path(__file__).parent
    csv_path = script_dir / "Nejumi4_result_20280826.csv"
    output_path = script_dir / "nejumi4_analysis_chart.png"
    
    # データ読み込み
    print("データを読み込み中...")
    df = load_data(csv_path)
    
    # 要約統計表示
    print_summary_statistics(df)
    
    # メトリクスグループ取得
    metric_groups = get_metric_groups()
    all_metrics = []
    for metrics in metric_groups.values():
        all_metrics.extend(metrics)
    
    # 平均値計算
    print("\n平均値を計算中...")
    all_avg, ja_avg = calculate_averages_by_category(df, all_metrics)
    
    print(f"\n利用可能なモデルサイズカテゴリ: {list(all_avg.index)}")
    
    # チャート作成
    print("チャートを作成中...")
    create_comparison_chart(all_avg, ja_avg, metric_groups, output_path, df)
    
    print(f"\n分析完了! チャートを {output_path} に保存しました。")
    
    # 詳細な統計情報を表示
    print("\n="*80)
    print("詳細統計 - 日本語モデル vs 全モデル平均")
    print("="*80)
    
    for category in all_avg.index:
        if category in ja_avg.index:
            print(f"\n【{category}】")
            print("メトリクス\t\t全モデル平均\t日本語モデル平均\t差分")
            print("-"*70)
            
            for metric in ['TOTAL_SCORE', '汎用的言語性能(GLP)_AVG', 'アラインメント(ALT)_AVG']:
                if metric in all_avg.columns:
                    all_val = all_avg.loc[category, metric]
                    ja_val = ja_avg.loc[category, metric] if not pd.isna(ja_avg.loc[category, metric]) else 0
                    diff = ja_val - all_val
                    print(f"{metric[:20]:<20}\t{all_val:.3f}\t\t{ja_val:.3f}\t\t{diff:+.3f}")

if __name__ == "__main__":
    main()
