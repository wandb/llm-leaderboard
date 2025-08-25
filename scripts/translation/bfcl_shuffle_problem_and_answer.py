#!/usr/bin/env python3
"""
Script to randomly shuffle problem files' ID order first, then shuffle possible_answer files to match.
This creates a truly randomized order for both problem and answer files.
For multi-turn files, filters to only include problems with 3 turns or less.
"""

"""
使用方法 (Usage):

このスクリプトは以下の機能を提供します：

1. BFCLファイルの処理 (process_bfcl_files):
   - en_new/bfclディレクトリのファイルを処理
   - 各ファイルから50問をランダムに抽出してシャッフル
   - 問題ファイルと回答ファイルの順序を一致させる
   
   実行方法:
   python bfcl_shuffle_problem_and_answer.py
   # スクリプト内で process_bfcl_files() を呼び出す

2. マルチターンファイルの処理 (main_multi_turn):
   - 3ターン以下の問題のみをフィルタリング
   - 問題ファイルと回答ファイルをシャッフル
   
   実行方法:
   python bfcl_shuffle_problem_and_answer.py
   # スクリプト内で main_multi_turn() を呼び出す

3. Java/JavaScriptファイルの処理 (main_java):
   - JavaとJavaScriptの問題ファイルをシャッフル
   
   実行方法:
   python bfcl_shuffle_problem_and_answer.py
   # スクリプト内で main_java() を呼び出す

4. ID順でのソート (sort_bfcl_files_by_id):
   - ファイルをID順でソート
   
   実行方法:
   python bfcl_shuffle_problem_and_answer.py
   # スクリプト内で sort_bfcl_files_by_id() を呼び出す

注意事項:
- 元のファイルは自動的に .backup 拡張子でバックアップされます
- ランダムシードは42に設定されているため、再現可能な結果が得られます
- ファイルが見つからない場合は警告が表示され、処理をスキップします

使用例:
# 特定の機能を実行するには、main()関数内で該当する関数をコメントアウトして実行
if __name__ == "__main__":
    # process_bfcl_files()      # BFCLファイルの処理
    # main_multi_turn()         # マルチターンファイルの処理
    # main_java()               # Java/JavaScriptファイルの処理
    sort_bfcl_files_by_id()    # ID順でのソート
"""

import json
import random
import os
import shutil
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Set

def load_json_file(file_path: str) -> List[Dict]:
    """Load JSON file and return list of objects."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def save_json_file(file_path: str, data: List[Dict]):
    """Save list of objects to JSON file, one object per line."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def load_turn_counts(csv_file: str) -> Dict[str, int]:
    """Load turn counts from CSV file and return mapping of ID to turn count."""
    turn_counts = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            turn_counts[row['id']] = int(row['turn_count'])
    return turn_counts

def filter_multi_turn_data(data: List[Dict], turn_counts: Dict[str, int], max_turns: int = 3) -> List[Dict]:
    """Filter multi-turn data to only include problems with max_turns or less."""
    filtered_data = []
    for item in data:
        item_id = item['id']
        if item_id in turn_counts and turn_counts[item_id] <= max_turns:
            filtered_data.append(item)
    return filtered_data

def shuffle_problem_file(problem_file: str, turn_counts: Dict[str, int] = None, max_turns: int = 3) -> Tuple[List[Dict], List[str]]:
    """Shuffle problem file and return shuffled data and new ID order."""
    data = load_json_file(problem_file)
    
    # Filter multi-turn data if turn_counts provided
    if turn_counts:
        data = filter_multi_turn_data(data, turn_counts, max_turns)
        print(f"Filtered to {len(data)} problems with {max_turns} turns or less")
    
    # Create a copy of the data to shuffle
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # Extract the new ID order
    new_id_order = [item['id'] for item in shuffled_data]
    
    return shuffled_data, new_id_order

def shuffle_possible_answer_file(possible_answer_file: str, target_id_order: List[str]) -> List[Dict]:
    """Shuffle possible_answer file to match target ID order."""
    data = load_json_file(possible_answer_file)
    
    # Create a mapping from ID to data item
    id_to_item = {item['id']: item for item in data}
    
    # Create new data list in target order
    shuffled_data = []
    for target_id in target_id_order:
        if target_id in id_to_item:
            shuffled_data.append(id_to_item[target_id])
        else:
            print(f"Warning: ID {target_id} not found in possible_answer file")
    
    return shuffled_data

def backup_file(file_path: str):
    """Create a backup of the original file."""
    backup_path = file_path + '.backup'
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"Created backup: {backup_path}")

def filter_parallel_multiple_data(data: List[Dict], ids_to_remove: List[str]) -> List[Dict]:
    """Filter out specific IDs from parallel_multiple data."""
    filtered_data = []
    for item in data:
        if item['id'] not in ids_to_remove:
            filtered_data.append(item)
    return filtered_data

def parse_id_for_sorting(id_str: str) -> tuple:
    """Parse ID string to extract numeric parts for proper sorting.
    Handles IDs like:
    - 'live_multiple_1038-265-0' -> (1038, 265, 0)
    - 'irrelevance_102' -> (102, 0, 0)
    - 'simple_13' -> (13, 0, 0)"""
    try:
        # Split by underscore and get the last part
        parts = id_str.split('_')
        if len(parts) >= 2:
            # Get the numeric part (last part)
            numeric_part = parts[-1]
            
            # Check if it contains dashes (like live_multiple_1038-265-0)
            if '-' in numeric_part:
                # Split by dash and convert to integers
                numbers = [int(x) for x in numeric_part.split('-')]
                # Pad with zeros if less than 3 numbers
                while len(numbers) < 3:
                    numbers.append(0)
                return tuple(numbers[:3])  # Ensure exactly 3 numbers
            else:
                # Single number format (like irrelevance_102, simple_13)
                number = int(numeric_part)
                return (number, 0, 0)
        else:
            # Fallback to string sorting if format is unexpected
            return (0, 0, 0)
    except (ValueError, IndexError):
        # Fallback to string sorting if parsing fails
        return (0, 0, 0)

def sort_file_by_id(file_path: str):
    """Sort a JSON file by the 'id' field in ascending order, prioritizing numeric parts."""
    print(f"Sorting {file_path} by ID...")
    
    # Load the file
    data = load_json_file(file_path)
    print(f"Loaded {len(data)} items from {file_path}")
    
    # Sort by ID using numeric parsing
    sorted_data = sorted(data, key=lambda x: parse_id_for_sorting(x['id']))
    
    # Create backup
    backup_file(file_path)
    
    # Save sorted data
    save_json_file(file_path, sorted_data)
    print(f"Saved {len(sorted_data)} sorted items to {file_path}")

def sort_directory_by_id(directory_path: str):
    """Sort all JSON files in a directory by ID."""
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return
    
    print(f"\nSorting all JSON files in {directory_path} by ID...")
    
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {directory_path}")
        return
    
    print(f"Found {len(json_files)} JSON files to sort:")
    for file_name in json_files:
        print(f"  - {file_name}")
    
    # Sort each file
    for file_name in json_files:
        file_path = os.path.join(directory_path, file_name)
        try:
            sort_file_by_id(file_path)
        except Exception as e:
            print(f"Error sorting {file_name}: {e}")
    
    print(f"Completed sorting all files in {directory_path}")

def process_bfcl_files():
    """Process BFCL files for en_new/bfcl directory with proper handling of file mismatches."""
    # Define file paths
    base_dir = "/home/olachinkeigpu/Project/llm-leaderboard/artifacts/en_new/bfcl"
    translated_dir = os.path.join(base_dir, "translated")
    possible_answer_dir = os.path.join(base_dir, "possible_answer")
    possible_answer_new_dir = os.path.join(base_dir, "possible_answer_new")
    translated_japanese_dir = os.path.join(base_dir, "translated_japanese")
    
    # Create directories if they don't exist
    os.makedirs(translated_dir, exist_ok=True)
    os.makedirs(possible_answer_new_dir, exist_ok=True)
    
    # Files to process with their expected counts and special handling
    files_to_process = [
        ("BFCL_v3_live_multiple.json", True),      # 1,053問 (but file has 62) - has possible_answer
        ("BFCL_v3_simple.json", True),             # 400問 - has possible_answer
        ("BFCL_v3_live_simple.json", True),        # 258問 - has possible_answer
        ("BFCL_v3_multiple.json", True),           # 200問 - has possible_answer
        ("BFCL_v3_parallel.json", True),           # 200問 - has possible_answer
        ("BFCL_v3_parallel_multiple.json", True),  # 200問 (special handling) - has possible_answer
        ("BFCL_v3_live_irrelevance.json", False),  # 882問 (ランダムに30問抽出) - no possible_answer
        ("BFCL_v3_irrelevance.json", False),       # 240問 (ランダムに30問抽出) - no possible_answer
    ]
    
    # IDs to remove from parallel_multiple
    ids_to_remove = [
        "live_multiple_2-1-0",
        "live_multiple_4-2-1",
        "live_multiple_6-3-1",
        "live_multiple_7-3-2",
        "live_multiple_10-4-2",
        "live_multiple_14-4-6",
        "live_multiple_16-4-8",
        "live_multiple_19-4-11",
        "live_multiple_20-4-12",
        "live_multiple_21-4-13",
        "live_multiple_22-4-14"
    ]
    
    # Set random seed for reproducible results
    random.seed(42)
    
    # Process each file
    for problem_file_name, has_possible_answer in files_to_process:
        problem_file = os.path.join(base_dir, problem_file_name)
        possible_answer_file = os.path.join(possible_answer_dir, problem_file_name)
        
        print(f"\nProcessing {problem_file_name}...")
        
        if not os.path.exists(problem_file):
            print(f"Problem file not found: {problem_file}")
            continue
        
        # Load problem data
        problem_data = load_json_file(problem_file)
        print(f"Loaded {len(problem_data)} problems from {problem_file_name}")
        
        # Special handling for parallel_multiple
        if problem_file_name == "BFCL_v3_parallel_multiple.json":
            print(f"Filtering out {len(ids_to_remove)} specified IDs...")
            problem_data = filter_parallel_multiple_data(problem_data, ids_to_remove)
            print(f"After filtering: {len(problem_data)} problems")
            
            # Shuffle and take first 50
            random.shuffle(problem_data)
            problem_data = problem_data[:50]
            print(f"Taking first 50 problems after shuffling")
        
        # Special handling for irrelevance files - take 30 random questions
        elif problem_file_name in ["BFCL_v3_live_irrelevance.json", "BFCL_v3_irrelevance.json"]:
            print(f"Taking 50 random questions from {len(problem_data)} total questions")
            random.shuffle(problem_data)
            problem_data = problem_data[:50]
            print(f"Selected 50 questions after shuffling")
        
        # For all other files, also limit to 50 questions
        else:
            print(f"Taking 50 random questions from {len(problem_data)} total questions")
            random.shuffle(problem_data)
            problem_data = problem_data[:50]
            print(f"Selected 50 questions after shuffling")
        
        # Shuffle the problem data
        shuffled_problem = problem_data.copy()
        random.shuffle(shuffled_problem)
        
        # Extract the new ID order from problem data
        new_id_order = [item['id'] for item in shuffled_problem]
        
        # Save shuffled problem data
        translated_file = os.path.join(translated_dir, problem_file_name)
        print(f"Saving shuffled problem data with {len(shuffled_problem)} items to {translated_file}")
        save_json_file(translated_file, shuffled_problem)
        
        # Handle possible_answer file if it exists
        if has_possible_answer:
            if not os.path.exists(possible_answer_file):
                print(f"Possible answer file not found: {possible_answer_file}")
                continue
            
            # Load possible answer data
            possible_answer_data = load_json_file(possible_answer_file)
            print(f"Loaded {len(possible_answer_data)} answers from possible_answer file")
            
            # Create mapping from ID to answer item
            id_to_answer = {item['id']: item for item in possible_answer_data}
            
            # Create shuffled answer data in same order as problem
            shuffled_answer = []
            missing_ids = []
            for target_id in new_id_order:
                if target_id in id_to_answer:
                    shuffled_answer.append(id_to_answer[target_id])
                else:
                    missing_ids.append(target_id)
                    print(f"Warning: ID {target_id} not found in possible_answer file")
            
            if missing_ids:
                print(f"Total missing IDs: {len(missing_ids)}")
            
            # Save shuffled answer data
            possible_answer_new_file = os.path.join(possible_answer_new_dir, problem_file_name)
            print(f"Saving shuffled answer data with {len(shuffled_answer)} items to {possible_answer_new_file}")
            save_json_file(possible_answer_new_file, shuffled_answer)
        else:
            print(f"No possible_answer file needed for {problem_file_name}")
        
        print(f"{problem_file_name} processed successfully!")
    
    print("\nAll files processed successfully!")

def sort_bfcl_files_by_id():
    """Sort BFCL files in possible_answer_new and translated_japanese directories by ID."""
    base_dir = "/home/olachinkeigpu/Project/llm-leaderboard/artifacts/en_new/bfcl"
    possible_answer_new_dir = os.path.join(base_dir, "possible_answer_new")
    translated_japanese_dir = os.path.join(base_dir, "translated_japanese")
    
    print("Sorting BFCL files by ID...")
    
    # Sort possible_answer_new directory
    if os.path.exists(possible_answer_new_dir):
        sort_directory_by_id(possible_answer_new_dir)
    else:
        print(f"Directory not found: {possible_answer_new_dir}")
    
    # Sort translated_japanese directory
    if os.path.exists(translated_japanese_dir):
        sort_directory_by_id(translated_japanese_dir)
    else:
        print(f"Directory not found: {translated_japanese_dir}")
    
    print("\nCompleted sorting all BFCL files by ID!")

def sort_bfcl_v14_multi_turn_files_by_id():
    """Sort BFCL v14 multi-turn files by ID in both problem and possible_answer directories."""
    base_dir = "/home/olachinkeigpu/Project/llm-leaderboard/artifacts/bfcl:v14/bfcl"
    possible_answer_dir = os.path.join(base_dir, "possible_answer")
    
    # Multi-turn files to sort
    multi_turn_files = [
        "BFCL_v3_multi_turn_base.json",
        "BFCL_v3_multi_turn_long_context.json", 
        "BFCL_v3_multi_turn_miss_func.json",
        "BFCL_v3_multi_turn_miss_param.json"
    ]
    
    print("Sorting BFCL v14 multi-turn files by ID...")
    
    # Sort problem files
    print(f"\nSorting problem files in {base_dir}:")
    for file_name in multi_turn_files:
        file_path = os.path.join(base_dir, file_name)
        if os.path.exists(file_path):
            try:
                sort_file_by_id(file_path)
            except Exception as e:
                print(f"Error sorting {file_name}: {e}")
        else:
            print(f"File not found: {file_name}")
    
    # Sort possible_answer files
    print(f"\nSorting possible_answer files in {possible_answer_dir}:")
    for file_name in multi_turn_files:
        file_path = os.path.join(possible_answer_dir, file_name)
        if os.path.exists(file_path):
            try:
                sort_file_by_id(file_path)
            except Exception as e:
                print(f"Error sorting {file_name}: {e}")
        else:
            print(f"File not found: {file_name}")
    
    print("\nCompleted sorting all BFCL v14 multi-turn files by ID!")

def main_multi_turn():
    # Define file paths
    base_dir = "/home/olachinkeigpu/Project/llm-leaderboard/artifacts/bfcl:v13/bfcl"
    turn_counts_file = "/home/olachinkeigpu/Project/llm-leaderboard/scripts/translation/bfcl_turn_counts.csv"
    
    # Multi-turn files
    multi_turn_files = [
        "BFCL_v3_multi_turn_base.json",
        "BFCL_v3_multi_turn_long_context.json", 
        "BFCL_v3_multi_turn_miss_func.json",
        "BFCL_v3_multi_turn_miss_param.json"
    ]
    
    # Set random seed for reproducible results
    random.seed(42)
    
    # Load turn counts
    print("Loading turn counts from CSV file...")
    turn_counts = load_turn_counts(turn_counts_file)
    print(f"Loaded turn counts for {len(turn_counts)} problems")
    
    # Process each multi-turn file
    for problem_file_name in multi_turn_files:
        problem_file = os.path.join(base_dir, problem_file_name)
        possible_answer_file = os.path.join(base_dir, "possible_answer", problem_file_name)
        
        print(f"\nProcessing {problem_file_name}...")
        
        if os.path.exists(problem_file) and os.path.exists(possible_answer_file):
            print(f"Shuffling problem file: {problem_file}")
            backup_file(problem_file)
            shuffled_problem, id_order = shuffle_problem_file(problem_file, turn_counts, max_turns=3)
            
            print(f"Saving shuffled problem data with {len(shuffled_problem)} items")
            save_json_file(problem_file, shuffled_problem)
            
            print(f"Shuffling possible_answer file: {possible_answer_file}")
            backup_file(possible_answer_file)
            shuffled_answer = shuffle_possible_answer_file(possible_answer_file, id_order)
            
            print(f"Saving shuffled answer data with {len(shuffled_answer)} items")
            save_json_file(possible_answer_file, shuffled_answer)
            print(f"{problem_file_name} processed successfully!")
        else:
            print(f"Files not found for {problem_file_name}, skipping...")
    
    print("\nMulti-turn shuffling completed!")

def main_java():
    # Define file paths
    base_dir = "/home/olachinkeigpu/Project/llm-leaderboard/artifacts/bfcl:v13/bfcl"
    
    # Java files
    java_problem_file = os.path.join(base_dir, "BFCL_v3_java.json")
    java_possible_answer_file = os.path.join(base_dir, "possible_answer", "BFCL_v3_java.json")
    
    # JavaScript files
    javascript_problem_file = os.path.join(base_dir, "BFCL_v3_javascript.json")
    javascript_possible_answer_file = os.path.join(base_dir, "possible_answer", "BFCL_v3_javascript.json")
    
    # Set random seed for reproducible results
    random.seed(42)
    
    # Process Java files
    print("Processing Java files...")
    if os.path.exists(java_problem_file) and os.path.exists(java_possible_answer_file):
        print(f"Shuffling problem file: {java_problem_file}")
        backup_file(java_problem_file)
        shuffled_java_problem, java_id_order = shuffle_problem_file(java_problem_file)
        
        print(f"Saving shuffled problem data with {len(shuffled_java_problem)} items")
        save_json_file(java_problem_file, shuffled_java_problem)
        
        print(f"Shuffling possible_answer file: {java_possible_answer_file}")
        backup_file(java_possible_answer_file)
        shuffled_java_answer = shuffle_possible_answer_file(java_possible_answer_file, java_id_order)
        
        print(f"Saving shuffled answer data with {len(shuffled_java_answer)} items")
        save_json_file(java_possible_answer_file, shuffled_java_answer)
        print("Java files processed successfully!")
    else:
        print("Java files not found, skipping...")
    
    # Process JavaScript files
    print("\nProcessing JavaScript files...")
    if os.path.exists(javascript_problem_file) and os.path.exists(javascript_possible_answer_file):
        print(f"Shuffling problem file: {javascript_problem_file}")
        backup_file(javascript_problem_file)
        shuffled_javascript_problem, javascript_id_order = shuffle_problem_file(javascript_problem_file)
        
        print(f"Saving shuffled problem data with {len(shuffled_javascript_problem)} items")
        save_json_file(javascript_problem_file, shuffled_javascript_problem)
        
        print(f"Shuffling possible_answer file: {javascript_possible_answer_file}")
        backup_file(javascript_possible_answer_file)
        shuffled_javascript_answer = shuffle_possible_answer_file(javascript_possible_answer_file, javascript_id_order)
        
        print(f"Saving shuffled answer data with {len(shuffled_javascript_answer)} items")
        save_json_file(javascript_possible_answer_file, shuffled_javascript_answer)
        print("JavaScript files processed successfully!")
    else:
        print("JavaScript files not found, skipping...")
    
    print("\nShuffling completed!")

if __name__ == "__main__":

    sort_bfcl_v14_multi_turn_files_by_id()
    



