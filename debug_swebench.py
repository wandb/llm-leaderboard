import json
import re
from datasets import load_dataset
from typing import Dict

def format_problem_statement(instance: Dict) -> str:
    """改良されたプロンプト生成（より具体的な指示）"""
    problem_statement = instance["problem_statement"]
    hints_text = instance.get("hints_text", "")
    
    prompt = f"""You are a software engineer tasked with fixing a bug in a Python codebase. 

<issue>
{problem_statement}
</issue>

{f"<hints>{hints_text}</hints>" if hints_text.strip() else ""}

Please generate a patch to fix this issue. The patch must be in standard unified diff format.

IMPORTANT REQUIREMENTS:
1. Use proper unified diff format starting with "diff --git" or "--- filename" and "+++ filename"
2. Include proper hunk headers with @@ line numbers @@
3. Use exactly one space after +, -, or for unchanged context lines
4. Do not include any explanations, just the patch
5. Make minimal changes - only modify what's necessary to fix the issue
6. Ensure the patch can be applied with the standard `patch` command

Example format:
```diff
diff --git a/path/to/file.py b/path/to/file.py
index abc123..def456 100644
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -10,7 +10,7 @@
 def example_function():
     # existing code
-    old_line = "old value"
+    new_line = "new value"
     # more existing code
```

Generate the patch now:"""
    
    return prompt

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

def extract_patch_from_response(response: str) -> str:
    """レスポンスからパッチを抽出（改良版）"""
    if not response or not response.strip():
        return ""
    
    # コードブロック内のdiffを探す
    code_block_patterns = [
        r'```(?:diff|patch)\n(.*?)\n```',
        r'```\n(diff --git.*?)\n```',
        r'```\n(.*?@@.*?)\n```',
    ]
    
    for pattern in code_block_patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            patch = matches[0].strip()
            if validate_diff_format(patch):
                return patch
    
    # コードブロック外のdiffを探す
    diff_patterns = [
        r'(diff --git .*?)(?=\n\n|\ndiff --git|\Z)',
        r'(--- .*?\+\+\+ .*?(?:@@.*?)*?)(?=\n\n|--- |\Z)',
    ]
    
    for pattern in diff_patterns:
        matches = re.findall(pattern, response, re.DOTALL | re.MULTILINE)
        if matches:
            patch = matches[0].strip()
            if validate_diff_format(patch):
                return patch
    
    # 最後の手段：レスポンス全体を返す（ただし、diff形式であることを確認）
    if validate_diff_format(response):
        return response.strip()
    
    # diff形式でない場合は空文字を返す
    print("WARNING: No valid diff format found in response")
    return ""

def main():
    print("Loading SWE-Bench Verified...")
    hf_dataset = load_dataset('princeton-nlp/SWE-bench_Verified', split='test')
    sample = hf_dataset[0]
    
    print(f"=== Instance: {sample['instance_id']} ===")
    print(f"Repository: {sample['repo']}")
    print()
    
    print("=== PROBLEM STATEMENT ===")
    print(f"Length: {len(sample['problem_statement'])} chars")
    print(sample['problem_statement'])
    print()
    
    print("=== HINTS TEXT ===")
    print(f"Length: {len(sample['hints_text'])} chars")
    print(repr(sample['hints_text']))
    print()
    
    print("=== GENERATED PROMPT ===")
    prompt = format_problem_statement(sample)
    print(f"Prompt length: {len(prompt)} chars")
    print(prompt[:2000] + "..." if len(prompt) > 2000 else prompt)
    print()
    
    print("=== EXPECTED PATCH ===")
    expected_patch = sample['patch']
    print(f"Expected patch length: {len(expected_patch)} chars")
    print("Expected patch validation:", validate_diff_format(expected_patch))
    print("Expected patch (first 1000 chars):")
    print(expected_patch[:1000])
    print()
    
    # 実際のAPIレスポンスをシミュレート（期待されるパッチで）
    print("=== PATCH EXTRACTION TEST ===")
    # マークダウン形式でラップしてテスト
    wrapped_patch = f"```diff\n{expected_patch}\n```"
    extracted = extract_patch_from_response(wrapped_patch)
    print("Extraction successful:", bool(extracted))
    print("Extracted equals expected:", extracted.strip() == expected_patch.strip())
    if extracted != expected_patch.strip():
        print("DIFFERENCE FOUND!")
        print("Extracted length:", len(extracted))
        print("Expected length:", len(expected_patch.strip()))

if __name__ == "__main__":
    main() 