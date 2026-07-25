import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


from evaluator.mtbench import (
    Answer,
    Question,
    _answer_payload,
    _atomic_write_json,
    _load_answer_checkpoint,
    _load_judge_checkpoint,
    generate_model_answer,
)
from llm_inference_adapter import LLMResponse


def test_answer_checkpoint_round_trip(tmp_path):
    path = tmp_path / "answers" / "1.json"
    answer = Answer(
        question_id=1,
        model_id="model-run",
        choices=[{"turns": ["first", "second"]}],
        tstamp=123.0,
        answer_id="answer-1",
    )
    _atomic_write_json(path, _answer_payload(answer))

    loaded = _load_answer_checkpoint(
        path,
        question_id=1,
        model_id="model-run",
    )

    assert loaded == answer


def test_answer_checkpoint_rejects_another_run(tmp_path):
    path = tmp_path / "answers" / "1.json"
    _atomic_write_json(
        path,
        _answer_payload(
            Answer(
                question_id=1,
                model_id="old-run",
                choices=[{"turns": ["answer"]}],
            )
        ),
    )

    assert (
        _load_answer_checkpoint(path, question_id=1, model_id="new-run") is None
    )


def test_judge_checkpoint_requires_matching_identity(tmp_path):
    path = tmp_path / "judgments" / "q1-turn1-judge0.json"
    payload = {
        "question_id": 1,
        "turn": 1,
        "judge_index": 0,
        "score": 7,
    }
    _atomic_write_json(path, payload)

    assert _load_judge_checkpoint(
        path,
        question_id=1,
        turn=1,
        judge_index=0,
    ) == payload
    assert (
        _load_judge_checkpoint(
            path,
            question_id=1,
            turn=2,
            judge_index=0,
        )
        is None
    )


def test_generated_turn_is_checkpointed_immediately(tmp_path):
    class FakeProcessor:
        async def process_single_async(self, messages, **_kwargs):
            return LLMResponse(content=f"answer-{len(messages)}")

    answer = Answer(
        question_id=1,
        model_id="model-run",
        choices=[{"turns": []}],
    )
    path = tmp_path / "answers" / "1.json"

    result = asyncio.run(
        generate_model_answer(
            Question(
                question_id=1,
                category="reasoning",
                turns=["first", "second"],
            ),
            FakeProcessor(),
            answer,
            {"temperature": 0},
            {},
            checkpoint_path=path,
        )
    )

    assert result.choices[0]["turns"] == ["answer-1"]
    assert _load_answer_checkpoint(
        path,
        question_id=1,
        model_id="model-run",
    ).choices[0]["turns"] == ["answer-1"]
