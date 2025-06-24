#!/usr/bin/env python3

import re

def extract_diff(response):
    """
    公式SWE-benchのextract_diff関数（完全複製）
    Extracts the diff from a response formatted in different ways
    """
    if response is None:
        return None
    diff_matches = []
    other_matches = []
    pattern = re.compile(r"\<([\w-]+)\>(.*?)\<\/\1\>", re.DOTALL)
    for code, match in pattern.findall(response):
        if code in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)
    pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
    for code, match in pattern.findall(response):
        if code in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)
    if diff_matches:
        return diff_matches[0]
    if other_matches:
        return other_matches[0]
    return response.split("</s>")[0]

def test_extract_diff():
    # テストケース1: ```diff ... ```
    response1 = """
Looking at this issue, I need to modify the file to fix the bug.

```diff
--- a/astropy/timeseries/core.py
+++ b/astropy/timeseries/core.py
@@ -77,7 +77,7 @@
     def example_function():
         # existing code
-        old_line = "old value"
+        new_line = "new value"
         # more existing code
```

This should fix the issue.
"""
    
    # テストケース2: <patch> ... </patch>
    response2 = """
Here's the fix:

<patch>
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -10,7 +10,7 @@
 def example_function():
     # existing code
-    old_line = "old value"
+    new_line = "new value"
     # more existing code
</patch>
"""

    # テストケース3: コードブロックなし（生テキスト）
    response3 = """
--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 def test():
-    return False
+    return True
</s>
More text after
"""

    print("=== 公式extract_diffテスト ===")
    
    test_cases = [
        ("```diff```形式", response1),
        ("<patch>形式", response2), 
        ("生テキスト", response3)
    ]
    
    for name, response in test_cases:
        print(f"\n{name}:")
        patch = extract_diff(response)
        print(f"  抽出結果: {len(patch) if patch else 0} 文字")
        if patch:
            print(f"  最初の数行:")
            for line in patch.split('\n')[:5]:
                print(f"    {repr(line)}")
        else:
            print("  抽出されたパッチ: None")

if __name__ == "__main__":
    test_extract_diff() 