from datasets import load_dataset

def validate_diff_format(patch: str) -> bool:
    """diff形式の詳細なバリデーション（修正版）"""
    if not patch or not patch.strip():
        return False
    
    lines = patch.strip().split('\n')
    
    # 基本的なdiff構造をチェック
    has_diff_header = False
    has_hunk_header = False
    has_changes = False
    
    for line in lines:
        # diff --git ヘッダー
        if line.startswith('diff --git'):
            has_diff_header = True
        
        # ファイルヘッダー（--- +++）
        if line.startswith('---') or line.startswith('+++'):
            has_diff_header = True
        
        # ハンクヘッダー（@@）
        if line.startswith('@@') and '@@' in line[2:]:
            has_hunk_header = True
        
        # 変更行（+, -, または空白で始まる）
        if len(line) > 0 and line[0] in ['+', '-', ' ']:
            has_changes = True
    
    # 最低限の形式チェック：ハンクヘッダーがあれば有効
    if has_hunk_header:
        return True
    
    # または、diff --gitヘッダーがあれば有効
    if has_diff_header and has_changes:
        return True
    
    return False

def main():
    print("Loading SWE-Bench Verified...")
    hf_dataset = load_dataset('princeton-nlp/SWE-bench_Verified', split='test')
    
    # 最初のいくつかのサンプルをテスト
    for i in range(min(5, len(hf_dataset))):
        sample = hf_dataset[i]
        patch = sample['patch']
        instance_id = sample['instance_id']
        
        is_valid = validate_diff_format(patch)
        print(f"Instance {instance_id}: {'VALID' if is_valid else 'INVALID'}")
        
        if not is_valid:
            print(f"  Patch length: {len(patch)}")
            print(f"  First 200 chars: {repr(patch[:200])}")
            print()

if __name__ == "__main__":
    main() 