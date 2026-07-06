from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
BFCL_ROOT = ROOT / "scripts" / "evaluator" / "evaluate_utils" / "bfcl_pkg"
sys.path.insert(0, str(BFCL_ROOT))

from bfcl.utils import find_file_with_suffix


def test_find_file_with_suffix_ignores_metadata_json(tmp_path):
    metadata = tmp_path / "translation_metadata.json"
    metadata.write_text("{}", encoding="utf-8")
    category = tmp_path / "BFCL_v3_multiple.json"
    category.write_text("[]", encoding="utf-8")

    assert find_file_with_suffix(tmp_path, "multiple") == category
