#!/usr/bin/env python3

import json
from pathlib import Path

def check_predictions_file():
    """Find and check the latest predictions file"""
    
    # Check for recent prediction files
    log_dirs = list(Path("logs/run_evaluation").glob("nejumi_*/gpt-4.1-2025-04-14"))
    
    if not log_dirs:
        print("No evaluation logs found")
        return
    
    latest_dir = max(log_dirs, key=lambda p: p.stat().st_mtime)
    print(f"Latest evaluation: {latest_dir}")
    
    # Find predictions file from temp directory name
    temp_files = list(Path("/tmp").glob("swebench_official_*/predictions.jsonl"))
    
    for temp_file in temp_files:
        print(f"\n=== Checking predictions file: {temp_file} ===")
        try:
            with open(temp_file, 'r') as f:
                lines = f.readlines()
            
            print(f"Total predictions: {len(lines)}")
            
            for i, line in enumerate(lines[:2]):  # Check first 2
                pred = json.loads(line.strip())
                print(f"\nPrediction {i+1}:")
                print(f"  instance_id: {pred.get('instance_id')}")
                print(f"  patch length: {len(pred.get('patch', ''))}")
                print(f"  model_patch length: {len(pred.get('model_patch', ''))}")
                
                patch = pred.get('patch', '')
                if patch:
                    print(f"  patch preview (first 200 chars): {patch[:200]}")
                else:
                    print("  WARNING: Empty patch!")
                    
        except Exception as e:
            print(f"Error reading {temp_file}: {e}")

def check_generated_patches():
    """Check patch files in evaluation logs"""
    
    log_dirs = list(Path("logs/run_evaluation").glob("nejumi_*/gpt-4.1-2025-04-14/*"))
    
    for instance_dir in log_dirs[:2]:  # Check first 2
        patch_file = instance_dir / "patch.diff"
        if patch_file.exists():
            print(f"\n=== {instance_dir.name} ===")
            content = patch_file.read_text()
            print(f"Patch file size: {len(content)} chars")
            if content.strip():
                print(f"Content preview: {content[:200]}")
            else:
                print("EMPTY PATCH FILE!")

if __name__ == "__main__":
    print("=== SWE-bench Debug Tool ===")
    check_predictions_file()
    check_generated_patches() 