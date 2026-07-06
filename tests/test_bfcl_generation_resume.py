from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_bfcl_generation_resume_branch_uses_model_name_and_returns_handler():
    source = (
        REPO_ROOT
        / "scripts"
        / "evaluator"
        / "evaluate_utils"
        / "bfcl_pkg"
        / "bfcl"
        / "_llm_response_generation.py"
    ).read_text(encoding="utf-8")

    assert "handler = build_handler(args.model_name, args.temperature)" in source
    assert "previously generated for {args.model_name}" in source
    assert "previously generated for {args.model}" not in source
    assert "return handler" in source
