import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tools" / "evaluate_swebench_lite_patches.py"
OFFICIAL_REPO = REPO_ROOT / "external" / "SWE-bench"


@pytest.mark.skipif(
    not (OFFICIAL_REPO / "swebench" / "harness" / "run_evaluation.py").exists(),
    reason="official SWE-bench checkout is not installed",
)
def test_evaluate_swebench_lite_prepare_only_writes_official_predictions(tmp_path):
    patch_path = tmp_path / "patches.json"
    patch_path.write_text(
        json.dumps(
            [
                {
                    "instance_id": "django__django-13447",
                    "patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n",
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ids_path = tmp_path / "ids.json"
    ids_path.write_text(json.dumps(["django__django-13447"]) + "\n", encoding="utf-8")
    output_dir = tmp_path / "eval"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--official-repo",
            str(OFFICIAL_REPO),
            "--patch-path",
            str(patch_path),
            "--instance-ids-json",
            str(ids_path),
            "--output-dir",
            str(output_dir),
            "--model-name",
            "openai-direct/gpt-4.1-mini-2025-04-14",
            "--run-id",
            "prepare-test",
            "--prepare-only",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=True,
    )

    predictions = [
        json.loads(line)
        for line in (output_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    invocation = json.loads((output_dir / "official_invocation.json").read_text(encoding="utf-8"))

    assert result.stdout
    assert predictions == [
        {
            "instance_id": "django__django-13447",
            "model_patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n",
            "model_name_or_path": "openai-direct/gpt-4.1-mini-2025-04-14",
        }
    ]
    assert "--dataset_name" in invocation["command"]
    assert "princeton-nlp/SWE-bench_Lite" in invocation["command"]
    assert invocation["instance_ids"] == ["django__django-13447"]
