#!/usr/bin/env python3
"""
BFCL Sanity Check Script

This script performs comprehensive validation of BFCL dataset files:
1. Line count matching between corresponding files
2. JSON format validation for each line
3. ID sequence matching between corresponding files
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
import argparse


class BFCLSanityChecker:
    def __init__(self, main_dir: str, possible_answer_dir: str):
        """
        Initialize the sanity checker.
        
        Args:
            main_dir: Path to main BFCL directory
            possible_answer_dir: Path to possible_answer directory
        """
        self.main_dir = Path(main_dir)
        self.possible_answer_dir = Path(possible_answer_dir)
        self.errors = []
        self.warnings = []
        
    def get_json_files(self, directory: Path) -> List[Path]:
        """Get all JSON files in a directory."""
        return list(directory.glob("*.json"))
    
    def load_json_lines(self, file_path: Path) -> List[Dict]:
        """
        Load JSON lines from a file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of parsed JSON objects
            
        Raises:
            ValueError: If JSON parsing fails
        """
        json_objects = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    json_objects.append(obj)
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSON parse error at line {line_num}: {e}")
        return json_objects
    
    def extract_ids(self, json_objects: List[Dict]) -> List[str]:
        """Extract IDs from JSON objects."""
        ids = []
        for obj in json_objects:
            if 'id' in obj:
                ids.append(obj['id'])
            else:
                ids.append(None)
        return ids
    
    def check_file_pair(self, main_file: Path, possible_answer_file: Path) -> Dict:
        """
        Check a pair of corresponding files.
        
        Args:
            main_file: Path to main file
            possible_answer_file: Path to corresponding possible_answer file
            
        Returns:
            Dictionary with check results
        """
        result = {
            'main_file': str(main_file),
            'possible_answer_file': str(possible_answer_file),
            'line_count_match': False,
            'json_format_valid': False,
            'id_sequence_match': False,
            'errors': []
        }
        
        try:
            # Load JSON objects from both files
            main_objects = self.load_json_lines(main_file)
            possible_answer_objects = self.load_json_lines(possible_answer_file)
            
            # Check line count matching
            main_line_count = len(main_objects)
            possible_answer_line_count = len(possible_answer_objects)
            result['line_count_match'] = main_line_count == possible_answer_line_count
            
            if not result['line_count_match']:
                result['errors'].append(
                    f"Line count mismatch: main={main_line_count}, possible_answer={possible_answer_line_count}"
                )
            
            # Check JSON format validity
            result['json_format_valid'] = True  # If we got here, JSON parsing succeeded
            
            # Check ID sequence matching
            main_ids = self.extract_ids(main_objects)
            possible_answer_ids = self.extract_ids(possible_answer_objects)
            
            # Check if IDs are in the same order
            result['id_sequence_match'] = main_ids == possible_answer_ids
            
            if not result['id_sequence_match']:
                result['errors'].append(
                    f"ID sequence mismatch: main_ids={main_ids}, possible_answer_ids={possible_answer_ids}"
                )
            
            # Additional checks for ID consistency
            main_id_set = set(id for id in main_ids if id is not None)
            possible_answer_id_set = set(id for id in possible_answer_ids if id is not None)
            
            if main_id_set != possible_answer_id_set:
                result['errors'].append(
                    f"ID set mismatch: main_ids={main_id_set}, possible_answer_ids={possible_answer_id_set}"
                )
            
        except Exception as e:
            result['errors'].append(f"Error processing files: {e}")
            result['json_format_valid'] = False
        
        return result
    
    def run_sanity_check(self) -> Dict:
        """
        Run the complete sanity check.
        
        Returns:
            Dictionary with overall results
        """
        print("Starting BFCL Sanity Check...")
        print(f"Main directory: {self.main_dir}")
        print(f"Possible answer directory: {self.possible_answer_dir}")
        print("-" * 80)
        
        # Get all JSON files
        main_files = self.get_json_files(self.main_dir)
        possible_answer_files = self.get_json_files(self.possible_answer_dir)
        
        # Create mapping of corresponding files
        file_pairs = []
        skipped_files = []
        for main_file in main_files:
            possible_answer_file = self.possible_answer_dir / main_file.name
            if possible_answer_file.exists():
                file_pairs.append((main_file, possible_answer_file))
            else:
                # Skip files that don't have corresponding possible_answer files
                # (like irrelevance and relevance files)
                skipped_files.append(main_file.name)
                print(f"Skipping {main_file.name} (no corresponding possible_answer file)")
        
        if skipped_files:
            print(f"\nSkipped files (no possible_answer): {', '.join(skipped_files)}")
            print("-" * 80)
        
        # Check each file pair
        results = []
        for main_file, possible_answer_file in file_pairs:
            print(f"Checking: {main_file.name}")
            result = self.check_file_pair(main_file, possible_answer_file)
            results.append(result)
            
            if result['errors']:
                print(f"  ❌ Errors found:")
                for error in result['errors']:
                    print(f"    - {error}")
            else:
                print(f"  ✅ All checks passed")
        
        # Summary
        print("\n" + "=" * 80)
        print("SANITY CHECK SUMMARY")
        print("=" * 80)
        
        total_files = len(results)
        passed_files = sum(1 for r in results if not r['errors'])
        failed_files = total_files - passed_files
        
        print(f"Total file pairs checked: {total_files}")
        print(f"Files passed all checks: {passed_files}")
        print(f"Files with errors: {failed_files}")
        print(f"Files skipped (no possible_answer): {len(skipped_files)}")
        
        if failed_files > 0:
            print("\nDetailed error report:")
            for result in results:
                if result['errors']:
                    print(f"\n{result['main_file']}:")
                    for error in result['errors']:
                        print(f"  - {error}")
        
        if skipped_files:
            print(f"\nSkipped files (no possible_answer): {', '.join(skipped_files)}")
        
        return {
            'total_files': total_files,
            'passed_files': passed_files,
            'failed_files': failed_files,
            'skipped_files': len(skipped_files),
            'results': results,
            'warnings': self.warnings,
            'skipped_file_names': skipped_files
        }


def main():
    parser = argparse.ArgumentParser(description='BFCL Sanity Check Tool')
    parser.add_argument(
        '--main-dir',
        default='/home/olachinkeigpu/Project/llm-leaderboard/artifacts/bfcl:v14/bfcl',
        help='Path to main BFCL directory'
    )
    parser.add_argument(
        '--possible-answer-dir',
        default='/home/olachinkeigpu/Project/llm-leaderboard/artifacts/bfcl:v14/bfcl/possible_answer',
        help='Path to possible_answer directory'
    )
    
    args = parser.parse_args()
    
    # Validate directories exist
    if not os.path.exists(args.main_dir):
        print(f"Error: Main directory does not exist: {args.main_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.possible_answer_dir):
        print(f"Error: Possible answer directory does not exist: {args.possible_answer_dir}")
        sys.exit(1)
    
    # Run sanity check
    checker = BFCLSanityChecker(args.main_dir, args.possible_answer_dir)
    results = checker.run_sanity_check()
    
    # Exit with appropriate code
    if results['failed_files'] > 0:
        print(f"\n❌ Sanity check failed with {results['failed_files']} file(s) having errors")
        sys.exit(1)
    else:
        print(f"\n✅ All sanity checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()