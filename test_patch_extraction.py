#!/usr/bin/env python3

import sys
import os
sys.path.append('scripts/evaluator')
from swebench_official import extract_patch_from_response, is_valid_patch

def test_patch_extraction():
    # テストケース1: 正常なdiffパッチ
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

PATCH_END
"""
    
    # テストケース2: diff --gitフォーマット
    response2 = """
Here's the fix:

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

PATCH_END
"""

    # テストケース3: 無効なパッチ（ハンクヘッダなし）
    response3 = """
--- a/file.py
+++ b/file.py
-old line
+new line
"""

    print("=== パッチ抽出テスト ===")
    
    test_cases = [
        ("正常なdiffパッチ", response1),
        ("diff --gitフォーマット", response2), 
        ("無効なパッチ", response3)
    ]
    
    for name, response in test_cases:
        print(f"\n{name}:")
        patch = extract_patch_from_response(response)
        valid = is_valid_patch(patch)
        print(f"  抽出されたパッチ: {len(patch)} 文字")
        print(f"  有効性: {valid}")
        if patch:
            print(f"  最初の数行:")
            for line in patch.split('\n')[:5]:
                print(f"    {repr(line)}")
        if valid:
            print("  ✅ 有効なパッチ")
        else:
            print("  ❌ 無効なパッチ")

if __name__ == "__main__":
    test_patch_extraction() 