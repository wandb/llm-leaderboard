import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    path = REPO_ROOT / "scripts" / "evaluator" / "agentic_swe_assorted.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_results_backfills_legacy_flat_session_audit_columns(tmp_path):
    module = load_module()
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "leaderboard_table.json").write_text("[]", encoding="utf-8")
    row = {
        "instance_id": "django__django-1",
        "nemoclaw_session_audit_ok": True,
        "nemoclaw_session_audit_required": None,
        "nemoclaw_session_copy_source": None,
        "nemoclaw_session_copied_bytes": None,
        "nemoclaw_session_audit": {
            "required": True,
            "ok": True,
            "copied_session_bytes": 4096,
            "copy": {
                "source": "stdout_agent_meta",
                "bytes": 4096,
            },
        },
    }
    (tmp_path / "output_table.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )

    _, output_df, _ = module._load_results(tmp_path)

    loaded = output_df.iloc[0].to_dict()
    assert loaded["nemoclaw_session_audit_required"] is True
    assert loaded["nemoclaw_session_copy_source"] == "stdout_agent_meta"
    assert loaded["nemoclaw_session_copied_bytes"] == 4096
