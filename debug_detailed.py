import json
import re
from datasets import load_dataset

def analyze_patch_format(patch: str):
    """パッチ形式を詳細に分析"""
    print("=== PATCH ANALYSIS ===")
    print(f"Total length: {len(patch)}")
    lines = patch.split('\n')
    print(f"Number of lines: {len(lines)}")
    print()
    
    for i, line in enumerate(lines, 1):
        prefix = f"{i:2d}: "
        if line.startswith('diff --git'):
            print(f"{prefix}[DIFF HEADER] {repr(line)}")
        elif line.startswith('---'):
            print(f"{prefix}[FILE OLD] {repr(line)}")
        elif line.startswith('+++'):
            print(f"{prefix}[FILE NEW] {repr(line)}")
        elif line.startswith('@@'):
            print(f"{prefix}[HUNK HEADER] {repr(line)}")
        elif line.startswith('+'):
            print(f"{prefix}[ADD] {repr(line)}")
        elif line.startswith('-'):
            print(f"{prefix}[DEL] {repr(line)}")
        elif line.startswith(' '):
            print(f"{prefix}[CTX] {repr(line)}")
        elif line.strip() == '':
            print(f"{prefix}[EMPTY] {repr(line)}")
        else:
            print(f"{prefix}[OTHER] {repr(line)}")

def main():
    print("Loading SWE-Bench Verified...")
    hf_dataset = load_dataset('princeton-nlp/SWE-bench_Verified', split='test')
    sample = hf_dataset[0]
    
    expected_patch = sample['patch']
    analyze_patch_format(expected_patch)
    
    print("\n=== VALIDATION TESTS ===")
    
    # 各行の検査
    lines = expected_patch.strip().split('\n')
    has_diff_header = False
    has_file_headers = False  
    has_hunk_header = False
    has_changes = False
    
    for line in lines:
        if line.startswith('diff --git'):
            has_diff_header = True
            print(f"✓ Found diff header: {repr(line)}")
        elif line.startswith('---') or line.startswith('+++'):
            has_file_headers = True
            print(f"✓ Found file header: {repr(line)}")
        elif line.startswith('@@') and '@@' in line[2:]:
            has_hunk_header = True
            print(f"✓ Found hunk header: {repr(line)}")
        elif len(line) > 0 and line[0] in ['+', '-', ' ']:
            has_changes = True
            print(f"✓ Found change line: {repr(line)}")
    
    print(f"\nSummary:")
    print(f"  Diff header: {has_diff_header}")
    print(f"  File headers: {has_file_headers}")
    print(f"  Hunk header: {has_hunk_header}")
    print(f"  Changes: {has_changes}")
    
    print(f"\nShould be valid: {has_hunk_header or (has_diff_header and has_changes)}")

if __name__ == "__main__":
    main() 