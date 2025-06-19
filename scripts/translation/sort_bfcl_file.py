#!/usr/bin/env python3
"""
BFCL JSONファイルをidフィールドで並び替えるスクリプト（元ファイル上書き版）

このスクリプトは、data_jpディレクトリ内のBFCLで始まるJSONファイルを読み込み、
各ファイル内のJSONオブジェクトをidフィールドで並び替えて、元のファイルを上書きします。
"""

import json
import os
import glob
from pathlib import Path
import shutil

def sort_json_file_by_id_overwrite(input_file_path):
    """
    JSONファイル内のオブジェクトをidフィールドで並び替えて元のファイルを上書きする
    
    Args:
        input_file_path (str): 入力ファイルのパス
    
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
        
        # idフィールドで並び替え
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
        
        # 並び替えたオブジェクトを元のファイルに書き込み
        with open(input_file_path, 'w', encoding='utf-8') as f:
            for obj in sorted_objects:
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')
        
        # バックアップファイルを削除（成功した場合）
        os.remove(backup_path)
        
        print(f"✓ {input_file_path} を並び替えました ({len(sorted_objects)}件)")
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
    
    print("\n⚠️  警告: このスクリプトは元のファイルを上書きします。")
    print("   処理を続行しますか？ (y/N): ", end="")
    
    response = input().strip().lower()
    if response not in ['y', 'yes']:
        print("処理をキャンセルしました。")
        return
    
    print("\n並び替えを開始します...")
    
    success_count = 0
    total_count = len(bfcl_files)
    
    for file_path in bfcl_files:
        if sort_json_file_by_id_overwrite(file_path):
            success_count += 1
    
    print(f"\n完了: {success_count}/{total_count}件のファイルを正常に処理しました。")
    
    if success_count < total_count:
        print("一部のファイルでエラーが発生しました。")
    else:
        print("すべてのファイルが正常に並び替えられました。")

if __name__ == "__main__":
    main()