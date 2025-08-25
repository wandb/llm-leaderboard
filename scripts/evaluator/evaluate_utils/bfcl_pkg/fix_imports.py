#!/usr/bin/env python3
import os
import re
from pathlib import Path

def get_relative_path(from_file, to_module):
    """Calculate the relative import path from one file to another."""
    from_parts = Path(from_file).parent.parts
    to_parts = Path(to_module.replace('.', '/')).parts
    
    # Find common prefix
    common = 0
    for f, t in zip(from_parts, to_parts):
        if f == t:
            common += 1
        else:
            break
    
    # Calculate number of parent directories to traverse
    parents = len(from_parts) - common
    # Calculate the relative path
    relative = ['..'] * parents + list(to_parts[common:])
    return '.'.join(relative)

def fix_imports(file_path):
    """Fix import statements in a Python file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Get the relative path from this file to bfcl directory
    try:
        file_rel_path = Path(file_path).relative_to(Path(file_path).parent.parent.parent)
    except ValueError:
        # If the file is not in the expected structure, use absolute path
        file_rel_path = Path(file_path)
    
    def replace_import(match):
        import_path = match.group(1)
        # Remove bfcl_pkg.bfcl. prefix if present
        if import_path.startswith('bfcl_pkg.bfcl.'):
            import_path = import_path[len('bfcl_pkg.bfcl.'):]
        elif import_path.startswith('bfcl.'):
            import_path = import_path[len('bfcl.'):]
            
        relative_path = get_relative_path(file_rel_path, import_path)
        return f'from {relative_path} import'
    
    # First, handle any absolute imports (both bfcl_pkg.bfcl. and bfcl.)
    new_content = re.sub(r'from ((?:bfcl_pkg\.)?bfcl\.[^\s]+) import', replace_import, content)
    
    # Then handle relative imports with multiple dots
    new_content = re.sub(r'from \.+bfcl\.([^\s]+) import', r'from ..\1 import', new_content)
    
    # Clean up any duplicate dots
    while '...' in new_content:
        new_content = new_content.replace('...', '..')
    
    # Only write if changes were made
    if new_content != content:
        print(f"Fixing imports in {file_path}")
        with open(file_path, 'w') as f:
            f.write(new_content)
        return True
    return False

def main():
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    bfcl_dir = script_dir / 'bfcl'
    
    # Track modified files
    modified_files = []
    
    # Walk through all Python files in the bfcl directory
    for root, _, files in os.walk(bfcl_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                if fix_imports(file_path):
                    modified_files.append(file_path.relative_to(script_dir))
    
    # Print summary
    if modified_files:
        print("\nModified files:")
        for file in modified_files:
            print(f"- {file}")
        print("\nPlease check the changes and run your tests.")
    else:
        print("\nNo files needed modification.")

if __name__ == '__main__':
    main() 