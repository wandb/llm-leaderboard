#!/usr/bin/env python3
"""
BFCL JSONファイルをidフィールドで並び替えるスクリプト

このスクリプトは、data_jpディレクトリ内のBFCLで始まるJSONファイルを読み込み、
各ファイル内のJSONオブジェクトをidフィールドで並び替えて、新しいファイルに保存します。
"""

import json
import os
import glob
from pathlib import Path

def sort_json_file_by_id(input_file_path, output_file_path=None):
    """
    JSONファイル内のオブジェクトをidフィールドで並び替える
    
    Args:
        input_file_path (str): 入力ファイルのパス
        output_file_path (str): 出力ファイルのパス（Noneの場合は元のファイルを上書き）
    
    Returns:
        bool: 成功した場合はTrue、失敗した場合はFalse
    """
    try:
        # 出力ファイルパスが指定されていない場合は、元のファイル名に_sortedを追加
        if output_file_path is None:
            file_path = Path(input_file_path)
            output_file_path = str(file_path.parent / f"{file_path.stem}_sorted{file_path.suffix}")
        
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
        
        # idフィールドで並び替え
        # idが数値の場合は数値として比較、文字列の場合は文字列として比較
        def extract_id_for_sorting(obj):
            id_value = obj.get('id', '')
            if isinstance(id_value, str):
                # idが文字列の場合、数値部分を抽出して数値として比較
                import re
                numbers = re.findall(r'\d+', id_value)
                if numbers:
                    return (id_value.split('_')[0] if '_' in id_value else '', int(numbers[0]))
                else:
                    return (id_value, 0)
            else:
                return ('', id_value)
        
        sorted_objects = sorted(json_objects, key=extract_id_for_sorting)
        
        # 並び替えたオブジェクトをファイルに書き込み
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for obj in sorted_objects:
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')
        
        print(f"✓ {input_file_path} -> {output_file_path} ({len(sorted_objects)}件)")
        return True
        
    except Exception as e:
        print(f"✗ エラー: {input_file_path}の処理中にエラーが発生しました: {e}")
        return False

def main():
    """メイン関数"""
    # 現在のディレクトリを取得
    current_dir = Path(__file__).parent
    
    # BFCLで始まるJSONファイルを検索
    bfcl_files = glob.glob(str(current_dir / "BFCL_*.json"))
    
    if not bfcl_files:
        print("BFCLで始まるJSONファイルが見つかりませんでした。")
        return
    
    print(f"見つかったBFCLファイル: {len(bfcl_files)}件")
    for file in bfcl_files:
        print(f"  - {Path(file).name}")
    
    print("\n並び替えを開始します...")
    
    success_count = 0
    total_count = len(bfcl_files)
    
    for file_path in bfcl_files:
        if sort_json_file_by_id(file_path):
            success_count += 1
    
    print(f"\n完了: {success_count}/{total_count}件のファイルを正常に処理しました。")
    
    if success_count < total_count:
        print("一部のファイルでエラーが発生しました。")
    else:
        print("すべてのファイルが正常に並び替えられました。")

if __name__ == "__main__":
    main() 