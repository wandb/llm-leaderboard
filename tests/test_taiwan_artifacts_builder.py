import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DATA_UPLOADER_DIR = ROOT / "scripts" / "data_uploader"
if str(DATA_UPLOADER_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_UPLOADER_DIR))

import taiwan_artifacts


def test_build_mtbench_tw_adds_supplemental_reference_for_tceval_missing_coding_123(
    tmp_path,
    monkeypatch,
):
    rows_by_config = {
        "mt_bench_tw-coding": [
            {
                "id": "123",
                "turns": [
                    "請寫一個簡單的網站以HTML撰寫。",
                    "如何使用 CSS 將笑話文字變為紅色？",
                ],
                "reference": None,
                "category": "coding",
            }
        ],
        "mt_bench_tw-reasoning": [
            {
                "id": "101",
                "turns": ["問題", "追問"],
                "reference": ["參考1", "參考2"],
                "category": "reasoning",
            }
        ],
    }

    def fake_hf_rows(dataset, config, split, request_interval):
        assert dataset == taiwan_artifacts.TCEVAL_V2_DATASET
        assert split == "test"
        return rows_by_config.get(config, [])

    monkeypatch.setattr(taiwan_artifacts, "hf_rows", fake_hf_rows)

    paths = taiwan_artifacts.build_mtbench_tw(tmp_path, request_interval=0.0)
    reference_rows = [
        json.loads(line)
        for line in paths["referenceanswer"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    reference_by_id = {int(row["question_id"]): row for row in reference_rows}
    assert set(reference_by_id) == {101, 123}
    supplemental = reference_by_id[123]
    assert supplemental["model_id"] == "tceval-v2-reference"
    assert supplemental["answer_id"] == "tceval-v2-ref-123-supplemental"
    assert supplemental["supplemental_reference"]["source_dataset"] == "FastChat MT-Bench"
    assert "color: red" in supplemental["choices"][0]["turns"][1]


def test_validate_mtbench_tw_reference_contract_rejects_missing_required_reference():
    question_rows = [
        {"question_id": 123, "category": "coding", "turns": ["q1", "q2"]},
        {"question_id": 90, "category": "writing", "turns": ["q1", "q2"]},
    ]
    reference_rows = [
        {
            "question_id": 90,
            "model_id": "tceval-v2-reference",
            "choices": [{"index": 0, "turns": ["r1", "r2"]}],
        }
    ]

    with pytest.raises(ValueError, match="question_id.*123"):
        taiwan_artifacts.validate_mtbench_tw_reference_contract(
            question_rows,
            reference_rows,
        )
