#!/usr/bin/env python3
"""
Swallow vs Llama 比較分析スクリプト

tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4 と 
meta-llama/Llama-3.3-70B-Instruct を比較します。
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
    """メトリクスをグループ化"""
    return {
        'total': [
            # TOTAL_SCOREを削除
        ],
        'detailed': [
            'GLP_応用的言語性能', 'GLP_推論能力', 'GLP_知識・質問応答', 'GLP_基礎的言語性能',
            'GLP_アプリケーション開発', 'GLP_表現', 'GLP_翻訳', 'GLP_情報検索',
            'GLP_抽象的推論', 'GLP_論理的推論', 'GLP_数学的推論', 'GLP_一般的知識',
            'GLP_専門的知識', 'GLP_意味解析', 'GLP_構文解析', 'GLP_コーディング', 'GLP_関数呼び出し',
            'ALT_制御性', 'ALT_倫理・道徳', 'ALT_毒性', 'ALT_バイアス', 'ALT_真実性', 'ALT_堅牢性'
        ]
    }

def extract_model_data(df: pd.DataFrame, model_names: List[str]) -> Tuple[pd.Series, pd.Series]:
    """
    指定されたモデルのデータを抽出
    
    Returns:
        swallow_data: Swallowモデルのデータ
        llama_data: Llamaモデルのデータ
    """
    swallow_model = model_names[0]  # tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4
    llama_model = model_names[1]    # meta-llama/Llama-3.3-70B-Instruct
    
    swallow_data = df[df['model_name'] == swallow_model].iloc[0] if not df[df['model_name'] == swallow_model].empty else None
    llama_data = df[df['model_name'] == llama_model].iloc[0] if not df[df['model_name'] == llama_model].empty else None
    
    if swallow_data is None:
        raise ValueError(f"Swallowモデル '{swallow_model}' がデータに見つかりません")
    if llama_data is None:
        raise ValueError(f"Llamaモデル '{llama_model}' がデータに見つかりません")
    
    return swallow_data, llama_data

def create_comparison_chart(swallow_data: pd.Series, llama_data: pd.Series, 
                          metric_groups: Dict[str, List[str]], output_path: str):
    """比較チャートを作成"""
    
    # 色の定義
    colors = {
        'swallow': '#1E5799',    # 濃い青（Swallow用）
        'llama': '#7DB9E8'       # 薄い青（Llama用）
    }
    
    # TOTAL_SCOREを最初に配置
    total_metrics = metric_groups['total']
    detailed_metrics = metric_groups['detailed']
    
    # TOTAL_SCOREのデータを準備
    total_data = []
    for metric in total_metrics:
        if metric in swallow_data.index and metric in llama_data.index:
            swallow_val = float(swallow_data[metric]) if not pd.isna(swallow_data[metric]) else 0
            llama_val = float(llama_data[metric]) if not pd.isna(llama_data[metric]) else 0
            diff = swallow_val - llama_val
            
            total_data.append({
                'metric': metric,
                'swallow_val': swallow_val,
                'llama_val': llama_val,
                'diff': diff,
                'abs_diff': abs(diff)
            })
    
    # 詳細項目のデータを準備
    detailed_data = []
    for metric in detailed_metrics:
        if metric in swallow_data.index and metric in llama_data.index:
            swallow_val = float(swallow_data[metric]) if not pd.isna(swallow_data[metric]) else 0
            llama_val = float(llama_data[metric]) if not pd.isna(llama_data[metric]) else 0
            diff = swallow_val - llama_val
            
            detailed_data.append({
                'metric': metric,
                'swallow_val': swallow_val,
                'llama_val': llama_val,
                'diff': diff,
                'abs_diff': abs(diff)
            })
    
    # 詳細項目をSwallowが勝っている項目（プラス）とLlamaが勝っている項目（マイナス）に分ける
    positive_metrics = [data for data in detailed_data if data['diff'] > 0]
    negative_metrics = [data for data in detailed_data if data['diff'] <= 0]
    
    # それぞれを差分の大きい順でソート
    positive_metrics.sort(key=lambda x: x['diff'], reverse=True)  # プラスは大きい順
    negative_metrics.sort(key=lambda x: x['diff'], reverse=False)  # マイナスは小さい順（絶対値で大きい順）
    
    # TOTAL_SCORE + 詳細項目の順で配置
    if total_data:
        metric_data = total_data + positive_metrics + negative_metrics
    else:
        metric_data = positive_metrics + negative_metrics
    
    # ソートされたデータから配列を作成
    swallow_values = []
    llama_values = []
    labels = []
    label_colors = []  # ラベルの色を格納
    
    for data in metric_data:
        swallow_values.append(data['swallow_val'])
        llama_values.append(data['llama_val'])
        labels.append(data['metric'])
        
        # 全てのラベルを黒色に統一
        label_colors.append('#000000')  # 黒色
    
    if not swallow_values:
        print("比較可能なメトリクスが見つかりません")
        return
    
    # 図のサイズを調整
    fig, ax = plt.subplots(figsize=(24, 12))
    
    # バーの位置
    x = np.arange(len(labels))
    width = 0.35
    
    # バーチャート作成
    bars1 = ax.bar(x - width/2, llama_values, width, 
                  label='meta-llama/Llama-3.3-70B-Instruct', 
                  color=colors['llama'])
    bars2 = ax.bar(x + width/2, swallow_values, width, 
                  label='tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4', 
                  color=colors['swallow'])
    
    # グラフの設定
    ax.set_title('Swallow vs Llama 性能比較\n'
                'tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4 vs meta-llama/Llama-3.3-70B-Instruct', 
                fontsize=18, fontweight='bold')
    ax.set_xlabel('評価項目', fontsize=14)
    ax.set_ylabel('スコア', fontsize=14)
    ax.set_xticks(x)
    
    # ラベルの色を個別に設定
    tick_labels = ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=10)
    for i, (label, color) in enumerate(zip(tick_labels, label_colors)):
        label.set_color(color)
    
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # 統合された凡例を作成
    ax.legend(loc='upper right', fontsize=12)
    
    # 値をバーの上に表示
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        if llama_values[i] > 0:
            ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.01, 
                   f'{llama_values[i]:.3f}', ha='center', va='bottom', fontsize=9, color='gray')
        if swallow_values[i] > 0:
            ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.01, 
                   f'{swallow_values[i]:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

def print_comparison_statistics(swallow_data: pd.Series, llama_data: pd.Series):
    """比較統計を表示"""
    print("="*80)
    print("モデル比較分析")
    print("="*80)
    
    print(f"【Swallowモデル】: {swallow_data['model_name']}")
    print(f"  - JA フラグ: {swallow_data['JA']}")
    print(f"  - モデルサイズカテゴリ: {swallow_data['model_size_category']}")
    
    print(f"\n【Llamaモデル】: {llama_data['model_name']}")
    print(f"  - JA フラグ: {llama_data['JA']}")
    print(f"  - モデルサイズカテゴリ: {llama_data['model_size_category']}")
    
    print("\n="*80)
    print("詳細比較 - Swallow vs Llama")
    print("="*80)
    
    main_metrics = ['TOTAL_SCORE']
    
    print("メトリクス\t\t\tLlama\t\tSwallow\t\t差分\t\t勝者")
    print("-"*85)
    
    swallow_wins = 0
    llama_wins = 0
    
    for metric in main_metrics:
        if metric in swallow_data.index and metric in llama_data.index:
            swallow_val = float(swallow_data[metric]) if not pd.isna(swallow_data[metric]) else 0
            llama_val = float(llama_data[metric]) if not pd.isna(llama_data[metric]) else 0
            diff = swallow_val - llama_val
            winner = "Swallow" if diff > 0 else "Llama" if diff < 0 else "引き分け"
            
            if diff > 0:
                swallow_wins += 1
            elif diff < 0:
                llama_wins += 1
                
            print(f"{metric[:25]:<25}\t{llama_val:.3f}\t\t{swallow_val:.3f}\t\t{diff:+.3f}\t\t{winner}")
    
    print(f"\n【総合結果】")
    print(f"Swallow勝利: {swallow_wins}項目")
    print(f"Llama勝利: {llama_wins}項目")
    
    # 全項目での勝敗カウント
    metric_groups = get_metric_groups()
    all_metrics = metric_groups['total'] + metric_groups['detailed']
    
    total_swallow_wins = 0
    total_llama_wins = 0
    
    for metric in all_metrics:
        if metric in swallow_data.index and metric in llama_data.index:
            swallow_val = float(swallow_data[metric]) if not pd.isna(swallow_data[metric]) else 0
            llama_val = float(llama_data[metric]) if not pd.isna(llama_data[metric]) else 0
            diff = swallow_val - llama_val
            
            if diff > 0:
                total_swallow_wins += 1
            elif diff < 0:
                total_llama_wins += 1
    
    print(f"\n【全項目結果】（{len(all_metrics)}項目中）")
    print(f"Swallow勝利: {total_swallow_wins}項目 ({total_swallow_wins/len(all_metrics)*100:.1f}%)")
    print(f"Llama勝利: {total_llama_wins}項目 ({total_llama_wins/len(all_metrics)*100:.1f}%)")

def main():
    """メイン処理"""
    # ファイルパス設定
    script_dir = Path(__file__).parent
    csv_path = script_dir / "Nejumi4_result_20280826.csv"
    output_path = script_dir / "swallow_llama_comparison.png"
    
    # 比較対象モデル
    model_names = [
        "tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4",
        "meta-llama/Llama-3.3-70B-Instruct"
    ]
    
    # データ読み込み
    print("データを読み込み中...")
    df = load_data(csv_path)
    
    # モデルデータ抽出
    print("モデルデータを抽出中...")
    swallow_data, llama_data = extract_model_data(df, model_names)
    
    # 比較統計表示
    print_comparison_statistics(swallow_data, llama_data)
    
    # メトリクスグループ取得
    metric_groups = get_metric_groups()
    
    # チャート作成
    print("\nチャートを作成中...")
    create_comparison_chart(swallow_data, llama_data, metric_groups, output_path)
    
    print(f"\n分析完了! チャートを {output_path} に保存しました。")

if __name__ == "__main__":
    main()
