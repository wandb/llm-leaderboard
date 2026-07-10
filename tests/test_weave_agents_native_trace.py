import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_model_aliases_include_openrouter_provider_stripped_model():
    module = load_module(REPO_ROOT / "scripts" / "tools" / "weave_agents_native_trace.py")

    assert module.model_aliases("openrouter-direct/z-ai/glm-5.2") == [
        "openrouter-direct/z-ai/glm-5.2",
        "glm-5.2",
        "z-ai/glm-5.2",
    ]
