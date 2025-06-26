#!/usr/bin/env python3

from datasets import load_from_disk
import json
from pathlib import Path

def debug_arrow_dataset():
    """Debug the Arrow dataset structure"""
    
    # データセットパスを指定
    dataset_path = "/home/yuya/nejumi-swebench/llm-leaderboard/artifacts/swebench_verified_official:v0/swebench_verified_official"
    
    print(f"Loading dataset from: {dataset_path}")
    
    # Arrow形式で読み込み
    try:
        hf_dataset = load_from_disk(dataset_path)
        print(f"Dataset type: {type(hf_dataset)}")
        print(f"Dataset length: {len(hf_dataset)}")
        
        # Convert to list and check structure
        task_data = list(hf_dataset)
        print(f"task_data type: {type(task_data)}")
        print(f"task_data length: {len(task_data)}")
        
        if task_data:
            first_sample = task_data[0]
            print(f"\nFirst sample type: {type(first_sample)}")
            print(f"First sample (first 500 chars): {str(first_sample)[:500]}")
            
            if isinstance(first_sample, dict):
                print(f"Sample keys: {list(first_sample.keys())}")
                if "instance_id" in first_sample:
                    print(f"instance_id: {first_sample['instance_id']}")
                    print(f"repo: {first_sample.get('repo', 'NOT_FOUND')}")
                else:
                    print("ERROR: instance_id not found in sample!")
                    
            elif isinstance(first_sample, str):
                print("Sample is a string, trying to parse as JSON...")
                try:
                    parsed = json.loads(first_sample)
                    print(f"Parsed successfully!")
                    print(f"Parsed keys: {list(parsed.keys())}")
                    if "instance_id" in parsed:
                        print(f"instance_id: {parsed['instance_id']}")
                except Exception as e:
                    print(f"Failed to parse as JSON: {e}")
            else:
                print(f"Unknown sample type: {type(first_sample)}")
                
        # Try alternative loading method
        print("\n=== Trying alternative method ===")
        for i, sample in enumerate(hf_dataset):
            print(f"Sample {i} type: {type(sample)}")
            print(f"Sample {i} content: {str(sample)[:200]}")
            if i >= 2:  # Just check first few samples
                break
                
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_arrow_dataset() 