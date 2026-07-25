import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


from evaluator.evaluate_utils import runtime_resources


def test_ensure_file_descriptor_capacity_raises_soft_limit(monkeypatch):
    limits = [1024, 65536]
    calls = []

    fake_resource = types.SimpleNamespace(
        RLIMIT_NOFILE=7,
        getrlimit=lambda _kind: tuple(limits),
    )

    def setrlimit(_kind, value):
        calls.append(value)
        limits[:] = value

    fake_resource.setrlimit = setrlimit
    monkeypatch.setitem(sys.modules, "resource", fake_resource)
    monkeypatch.setattr(
        runtime_resources,
        "_open_file_descriptor_count",
        lambda: 17,
    )

    snapshot = runtime_resources.ensure_file_descriptor_capacity(65536)

    assert calls == [(65536, 65536)]
    assert snapshot.open_count == 17
    assert snapshot.soft_limit == 65536
    assert snapshot.hard_limit == 65536


def test_ensure_file_descriptor_capacity_respects_hard_limit(monkeypatch):
    limits = [512, 4096]
    fake_resource = types.SimpleNamespace(
        RLIMIT_NOFILE=7,
        getrlimit=lambda _kind: tuple(limits),
    )

    def setrlimit(_kind, value):
        limits[:] = value

    fake_resource.setrlimit = setrlimit
    monkeypatch.setitem(sys.modules, "resource", fake_resource)
    monkeypatch.setattr(
        runtime_resources,
        "_open_file_descriptor_count",
        lambda: 9,
    )

    snapshot = runtime_resources.ensure_file_descriptor_capacity(65536)

    assert snapshot.soft_limit == 4096
    assert snapshot.hard_limit == 4096
    assert snapshot.format() == "open=9, soft_limit=4096, hard_limit=4096"
