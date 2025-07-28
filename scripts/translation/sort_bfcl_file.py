#!/usr/bin/env python3
"""
BFCL JSONファイルをidフィールドで並び替えるスクリプト（turn数考慮版）

このスクリプトは、BFCLで始まるJSONファイルを読み込み、
各ファイル内のJSONオブジェクトを以下のルールで並び替えて、元のファイルを上書きします：
1. turn数が5未満のエントリを先頭に配置（ID順）
2. turn数が5以上のエントリを後半に配置（ID順）
"""

import json
import os
import glob
import pandas as pd
from pathlib import Path
import shutil

def load_turn_counts(csv_path):
    """
    turn数CSVファイルを読み込んで、IDとturn数のマッピングを作成する
    
    Args:
        csv_path (str): turn数CSVファイルのパス
    
    Returns:
        dict: IDをキー、turn数を値とする辞書
    """
    try:
        df = pd.read_csv(csv_path)
        turn_counts = {}
        for _, row in df.iterrows():
            turn_counts[row['id']] = row['turn_count']
        return turn_counts
    except Exception as e:
        print(f"警告: turn数CSVファイルの読み込みに失敗しました: {e}")
        return {}

def sort_json_file_by_id_and_turns_overwrite(input_file_path, turn_counts):
    """
    JSONファイル内のオブジェクトをturn数とidフィールドで並び替えて元のファイルを上書きする
    
    Args:
        input_file_path (str): 入力ファイルのパス
        turn_counts (dict): IDとturn数のマッピング
    
    Returns:
        bool: 成功した場合はTrue、失敗した場合はFalse
    """
    try:
        # バックアップファイルを作成
        backup_path = input_file_path + '.backup'
        shutil.copy2(input_file_path, backup_path)
        
        # ファイルを読み込み
        with open(input_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 各行をJSONオブジェクトとして解析
        json_objects = []
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if line:  # 空行をスキップ
                try:
                    obj = json.loads(line)
                    json_objects.append(obj)
                except json.JSONDecodeError as e:
                    print(f"警告: {input_file_path}の{line_num}行目でJSON解析エラー: {e}")
                    continue
        
        # turn数とidフィールドで並び替え
        def extract_sort_key(obj):
            id_value = obj.get('id', '')
            turn_count = turn_counts.get(id_value, 0)
            
            # turn数が5以上かどうかでグループ分け（0: 5未満, 1: 5以上）
            turn_group = 1 if turn_count >= 5 else 0
            
            if isinstance(id_value, str):
                # idが文字列の場合、数値部分を抽出して数値として比較
                import re
                numbers = re.findall(r'\d+', id_value)
                if numbers:
                    return (turn_group, id_value.split('_')[0] if '_' in id_value else '', int(numbers[0]))
                else:
                    return (turn_group, id_value, 0)
            else:
                return (turn_group, '', id_value)
        
        sorted_objects = sorted(json_objects, key=extract_sort_key)
        
        # 並び替えたオブジェクトを元のファイルに書き込み
        with open(input_file_path, 'w', encoding='utf-8') as f:
            for obj in sorted_objects:
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')
        
        # バックアップファイルを削除（成功した場合）
        os.remove(backup_path)
        
        # 統計情報を表示
        low_turn_count = sum(1 for obj in sorted_objects if turn_counts.get(obj.get('id', ''), 0) < 5)
        high_turn_count = len(sorted_objects) - low_turn_count
        
        print(f"✓ {input_file_path} を並び替えました ({len(sorted_objects)}件)")
        print(f"  - turn数5未満: {low_turn_count}件")
        print(f"  - turn数5以上: {high_turn_count}件")
        return True
        
    except Exception as e:
        print(f"✗ エラー: {input_file_path}の処理中にエラーが発生しました: {e}")
        # エラーが発生した場合はバックアップから復元
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, input_file_path)
            os.remove(backup_path)
            print(f"  バックアップから復元しました: {input_file_path}")
        return False

def main():
    """メイン関数"""
    # turn数CSVファイルのパス
    csv_path = "/home/olachinkeigpu/Project/llm-leaderboard/artifacts/bfcl:v5/bfcl/bfcl_turn_counts.csv"
    
    # turn数データを読み込み
    print("turn数データを読み込み中...")
    turn_counts = load_turn_counts(csv_path)
    if not turn_counts:
        print("turn数データの読み込みに失敗しました。処理を中止します。")
        return
    
    print(f"turn数データを読み込みました: {len(turn_counts)}件")
    
    # 処理対象のディレクトリリスト
    base_dir = "/home/olachinkeigpu/Project/llm-leaderboard/artifacts/bfcl:v5/bfcl"
    problem_dir = "/home/olachinkeigpu/Project/llm-leaderboard/artifacts/bfcl:v5/bfcl/possible_answer"  # 問題ディレクトリ
    possible_answer_dir = os.path.join(base_dir, "possible_answer")  # possible_answerディレクトリ
    
    target_directories = [
        problem_dir,  # 問題ディレクトリ
        possible_answer_dir  # possible_answerディレクトリ
    ]
    
    # 各ディレクトリでBFCLファイルを検索
    all_bfcl_files = []
    for directory in target_directories:
        if os.path.exists(directory):
            bfcl_files = glob.glob(os.path.join(directory, "BFCL_*.json"))
            all_bfcl_files.extend(bfcl_files)
            print(f"{directory}: {len(bfcl_files)}件のBFCLファイルを発見")
        else:
            print(f"警告: ディレクトリが存在しません: {directory}")
    
    if not all_bfcl_files:
        print("BFCLで始まるJSONファイルが見つかりませんでした。")
        return
    
    print(f"\n合計: {len(all_bfcl_files)}件のBFCLファイル")
    for file in all_bfcl_files:
        print(f"  - {Path(file).name}")
    
    print("\n⚠️  警告: このスクリプトは元のファイルを上書きします。")
    print("   処理を続行しますか？ (y/N): ", end="")
    
    response = input().strip().lower()
    if response not in ['y', 'yes']:
        print("処理をキャンセルしました。")
        return
    
    print("\n並び替えを開始します...")
    
    success_count = 0
    total_count = len(all_bfcl_files)
    
    for file_path in all_bfcl_files:
        if sort_json_file_by_id_and_turns_overwrite(file_path, turn_counts):
            success_count += 1
    
    print(f"\n完了: {success_count}/{total_count}件のファイルを正常に処理しました。")
    
    if success_count < total_count:
        print("一部のファイルでエラーが発生しました。")
    else:
        print("すべてのファイルが正常に並び替えられました。")

if __name__ == "__main__":
    main()