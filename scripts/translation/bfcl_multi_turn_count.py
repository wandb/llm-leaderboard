import json
import pandas as pd
import os
from pathlib import Path

def count_turns_per_id(file_path):
    """
    Count the number of turns for each ID in a BFCL multi-turn file
    
    Args:
        file_path (str): Path to the BFCL JSON file
        
    Returns:
        list: List of dictionaries with 'id' and 'turn_count' for each entry
    """
    results = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read JSON lines format
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        entry = json.loads(line)
                        entry_id = entry.get('id', f'unknown_{line_num}')
                        
                        # Count turns in the question field
                        question_data = entry.get('question', [])
                        if isinstance(question_data, list):
                            turn_count = len(question_data)
                        else:
                            turn_count = 0
                        
                        results.append({
                            'id': entry_id,
                            'turn_count': turn_count,
                            'file_name': os.path.basename(file_path)
                        })
                        
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse JSON on line {line_num} in {file_path}: {e}")
                        continue
                        
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []
    
    return results

def process_bfcl_multi_turn_files(file_paths, output_csv_path):
    """
    Process multiple BFCL multi-turn files and create a CSV with turn counts
    
    Args:
        file_paths (list): List of file paths to process
        output_csv_path (str): Path where to save the CSV file
    """
    all_results = []
    
    for file_path in file_paths:
        print(f"Processing {file_path}...")
        file_results = count_turns_per_id(file_path)
        all_results.extend(file_results)
        print(f"  Found {len(file_results)} entries with {sum(r['turn_count'] for r in file_results)} total turns")
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    print(f"\nResults saved to: {output_csv_path}")
    print(f"Total entries processed: {len(df)}")
    print(f"Total turns across all files: {df['turn_count'].sum()}")
    
    # Print summary statistics
    print("\nSummary by file:")
    summary = df.groupby('file_name').agg({
        'id': 'count',
        'turn_count': ['sum', 'mean', 'min', 'max']
    }).round(2)
    print(summary)
    
    return df

def main():
    """
    Main function - replace these paths with your actual file paths
    """
    # TODO: Replace these paths with your actual file paths
    file_paths = [
        "/home/olachinkeigpu/Project/llm-leaderboard/artifacts/bfcl:v5/bfcl/BFCL_v3_multi_turn_base.json",
        "/home/olachinkeigpu/Project/llm-leaderboard/artifacts/bfcl:v5/bfcl/BFCL_v3_multi_turn_long_context.json", 
        "/home/olachinkeigpu/Project/llm-leaderboard/artifacts/bfcl:v5/bfcl/BFCL_v3_multi_turn_miss_func.json",
        "/home/olachinkeigpu/Project/llm-leaderboard/artifacts/bfcl:v5/bfcl/BFCL_v3_multi_turn_miss_param.json"
    ]
    
    # TODO: Replace with your desired output CSV path
    output_csv_path = "/home/olachinkeigpu/Project/llm-leaderboard/artifacts/bfcl:v5/bfcl/bfcl_turn_counts.csv"
    
    # Check if paths are set
    if not file_paths or file_paths[0].startswith("path/to/"):
        print("Please update the file_paths list with your actual BFCL file paths")
        return
    
    if output_csv_path.startswith("path/to/"):
        print("Please update the output_csv_path with your desired CSV output path")
        return
    
    # Process the files
    df = process_bfcl_multi_turn_files(file_paths, output_csv_path)
    
    # Display first few rows
    print("\nFirst 10 rows of the CSV:")
    print(df.head(10))

if __name__ == "__main__":
    main()
