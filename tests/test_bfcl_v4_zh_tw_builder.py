import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "data_uploader" / "bfcl_v4_zh_tw.py"
SPEC = importlib.util.spec_from_file_location("bfcl_v4_zh_tw", MODULE_PATH)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _row(category: str, index: int, turns: int = 1) -> dict:
    return {
        "id": f"{category}_{index}",
        "question": [
            [{"role": "user", "content": f"Question {index}, turn {turn}"}]
            for turn in range(turns)
        ],
        "function": [
            {
                "name": "lookup",
                "description": "Look up a value.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name to look up.",
                        }
                    },
                },
            }
        ],
    }


def test_stable_sample_is_order_independent_and_deterministic():
    rows = [_row("simple_python", index) for index in range(50)]
    selected = builder.stable_sample(
        rows,
        category="simple_python",
        seed=123,
        max_per_category=30,
    )
    repeated = builder.stable_sample(
        list(reversed(rows)),
        category="simple_python",
        seed=123,
        max_per_category=30,
    )

    assert [row["id"] for row in selected] == [
        row["id"] for row in repeated
    ]
    assert len(selected) == 30


def test_multi_turn_rows_are_filtered_before_sampling(tmp_path):
    source = tmp_path / "data"
    rows = [
        _row("multi_turn_base", index, turns)
        for index, turns in enumerate([1, 2, 3, 4, 5])
    ]
    _write_jsonl(source / "BFCL_v4_multi_turn_base.json", rows)

    loaded = builder.load_category_rows(source, "multi_turn_base")

    assert [len(row["question"]) for row in loaded] == [1, 2, 3]


def test_shared_agentic_sources_use_one_selection(tmp_path):
    source = tmp_path / "data"
    memory_rows = [_row("memory", index) for index in range(40)]
    web_rows = [_row("web_search", index) for index in range(40)]
    _write_jsonl(source / "BFCL_v4_memory.json", memory_rows)
    _write_jsonl(source / "BFCL_v4_web_search.json", web_rows)

    selected, selected_ids = builder.select_rows(
        source,
        builder.AGENTIC_CATEGORIES,
        seed=42,
        max_per_category=30,
    )

    assert selected_ids["memory_kv"] == selected_ids["memory_vector"]
    assert selected_ids["memory_kv"] == selected_ids["memory_rec_sum"]
    assert selected_ids["web_search_base"] == selected_ids["web_search_no_snippet"]
    assert len(builder.merge_rows_for_files(selected)["memory"]) == 30
    assert len(builder.merge_rows_for_files(selected)["web_search"]) == 30


def test_ground_truth_is_filtered_to_selected_ids(tmp_path):
    source = tmp_path / "data"
    selected_files = {
        "simple_python": [_row("simple_python", 2), _row("simple_python", 4)]
    }
    _write_jsonl(
        source / "possible_answer" / "BFCL_v4_simple_python.json",
        [
            {"id": f"simple_python_{index}", "ground_truth": [index]}
            for index in range(6)
        ],
    )

    result = builder.collect_ground_truth_rows(source, selected_files)

    assert [row["id"] for row in result["simple_python"]] == [
        "simple_python_2",
        "simple_python_4",
    ]


def test_translation_changes_descriptions_but_preserves_schema_literals():
    schema = {
        "name": "weather.lookup",
        "description": "Look up weather.",
        "parameters": {
            "type": "dict",
            "properties": {
                "unit": {
                    "type": "string",
                    "description": "Temperature unit.",
                    "enum": ["celsius", "fahrenheit"],
                }
            },
            "required": ["unit"],
        },
    }
    model = "translator"
    translations = {
        builder.translation_key(model, "Look up weather."): "查詢天氣。",
        builder.translation_key(model, "Temperature unit."): "溫度單位。",
    }

    translated = builder.translate_schema(schema, translations, model)

    assert translated["description"] == "查詢天氣。"
    assert (
        translated["parameters"]["properties"]["unit"]["description"]
        == "溫度單位。"
    )
    assert translated["name"] == "weather.lookup"
    assert translated["parameters"]["properties"]["unit"]["enum"] == [
        "celsius",
        "fahrenheit",
    ]
    assert translated["parameters"]["required"] == ["unit"]


def test_agentic_ground_truth_keeps_original_and_adds_translation():
    rows = [{"id": "memory_1", "ground_truth": ["Advanced Algorithms", "35"]}]
    model = "translator"
    translations = {
        builder.translation_key(model, "Advanced Algorithms"): "進階演算法",
        builder.translation_key(model, "35"): "35",
    }

    augmented = builder.augment_agentic_ground_truth(rows, translations, model)

    assert augmented[0]["ground_truth"] == [
        "Advanced Algorithms",
        "進階演算法",
        "35",
    ]
    assert rows[0]["ground_truth"] == ["Advanced Algorithms", "35"]


def test_translation_batches_respect_item_and_character_limits():
    missing = {
        "a": "1234",
        "b": "5678",
        "c": "90",
    }

    batches = builder.translation_batches(
        missing,
        max_items=2,
        max_chars=6,
    )

    assert batches == [{"a": "1234"}, {"b": "5678", "c": "90"}]


def test_translate_batch_rejects_reordered_response_ids():
    class Response:
        output_text = json.dumps(
            {
                "translations": [
                    {"id": "b", "text": "乙"},
                    {"id": "a", "text": "甲"},
                ]
            }
        )

    class Responses:
        def create(self, **kwargs):
            return Response()

    class Client:
        responses = Responses()

    try:
        builder.translate_batch(Client(), "translator", {"a": "A", "b": "B"})
    except ValueError as exc:
        assert "IDs/order" in str(exc)
    else:
        raise AssertionError("reordered translation IDs must be rejected")


def test_generated_output_audit_detects_executable_schema_changes(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    row = _row("simple_python", 1)
    _write_jsonl(
        source / "multi_turn_func_doc" / "simple_python.json",
        [],
    )
    translated = json.loads(json.dumps(row))
    translated["question"][0][0]["content"] = "問題"
    translated["function"][0]["description"] = "查詢值。"
    translated["function"][0]["name"] = "changed_name"
    _write_jsonl(output / "BFCL_v4_simple_python.json", [translated])
    _write_jsonl(
        output / "multi_turn_func_doc" / "simple_python.json",
        [],
    )

    try:
        builder.audit_generated_output(
            source_dir=source,
            output_dir=output,
            selected_files={"simple_python": [row]},
            translation_texts=set(),
            translations=None,
            model="translator",
            include_agentic=False,
        )
    except ValueError as exc:
        assert "executable schema fields changed" in str(exc)
    else:
        raise AssertionError("schema name mutation must fail the translation audit")
