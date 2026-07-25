import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "tools" / "nemoclaw_gateway_restart.py"
spec = importlib.util.spec_from_file_location("nemoclaw_gateway_restart", MODULE_PATH)
assert spec is not None and spec.loader is not None
gateway = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gateway)


def test_parse_forward_port_from_nemoclaw_output():
    output = "Forwarding port 18790 to sandbox nejumi-taiwan in the background\n"

    assert gateway.parse_forward_port(output) == 18790
    assert gateway.parse_forward_port("gateway restarted") is None


def test_remove_confirmed_stale_lock_only_under_nemoclaw_state(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway.Path, "home", classmethod(lambda cls: tmp_path))
    state_dir = tmp_path / ".nemoclaw" / "state"
    state_dir.mkdir(parents=True)
    lock = state_dir / "shields-transition-lock-nejumi-taiwan.json"
    lock.write_text("{}", encoding="utf-8")
    output = (
        f"shields transition lock '{lock}': recorded owner PID 99999999 "
        "is not running"
    )

    assert gateway.remove_confirmed_stale_lock(output) == str(lock)
    assert not lock.exists()


def test_remove_confirmed_stale_lock_rejects_path_outside_state(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway.Path, "home", classmethod(lambda cls: tmp_path))
    lock = tmp_path / "shields-transition-lock-nejumi-taiwan.json"
    lock.write_text("{}", encoding="utf-8")
    output = (
        f"shields transition lock '{lock}': recorded owner PID 99999999 "
        "is not running"
    )

    assert gateway.remove_confirmed_stale_lock(output) is None
    assert lock.exists()


def test_recover_background_forward_uses_nonblocking_openshell(monkeypatch):
    calls = []
    starts = []

    def fake_run(command, timeout_seconds):
        calls.append((command, timeout_seconds))
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(gateway, "_run", fake_run)
    monkeypatch.setattr(
        gateway,
        "_start_background_forward",
        lambda command: starts.append(command)
        or SimpleNamespace(pid=1234, poll=lambda: None),
    )
    monkeypatch.setattr(gateway, "_port_ready", lambda port, timeout: port == 18790)

    evidence = gateway.recover_background_forward(
        openshell_bin="openshell",
        sandbox="nejumi-taiwan",
        port=18790,
    )

    assert evidence["ready"] is True
    assert calls[0][0] == [
        "openshell",
        "forward",
        "stop",
        "18790",
        "nejumi-taiwan",
    ]
    assert starts[0] == [
        "openshell",
        "forward",
        "start",
        "18790",
        "nejumi-taiwan",
        "--background",
    ]


def test_recover_background_forward_rejects_unhealthy_forward(monkeypatch):
    monkeypatch.setattr(
        gateway,
        "_run",
        lambda command, timeout: subprocess.CompletedProcess(
            command, 0, stdout="ok", stderr=""
        ),
    )
    fake_process = SimpleNamespace(pid=1234, poll=lambda: None)
    monkeypatch.setattr(gateway, "_start_background_forward", lambda command: fake_process)
    monkeypatch.setattr(gateway, "_terminate_process_group", lambda proc: None)
    monkeypatch.setattr(gateway, "_port_ready", lambda port, timeout: False)

    try:
        gateway.recover_background_forward(
            openshell_bin="openshell",
            sandbox="nejumi-taiwan",
            port=18790,
        )
    except gateway.NeMoClawGatewayRestartError as exc:
        assert exc.evidence["ready"] is False
    else:
        raise AssertionError("unhealthy background forward was accepted")


def test_reload_gateway_uses_supervisor_process_without_lifecycle_restart(monkeypatch):
    calls = []

    def fake_run(command, timeout):
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='{"old_pid":10,"new_pid":11,"port":18790}\n',
            stderr="",
        )

    monkeypatch.setattr(gateway, "_run", fake_run)
    monkeypatch.setattr(gateway, "_port_ready", lambda port, timeout: True)

    evidence = gateway.reload_nemoclaw_gateway_process(
        sandbox="nejumi-taiwan",
        container_id="container-123",
    )

    assert evidence["ok"] is True
    assert evidence["mode"] == "supervisor_process_reload"
    assert calls[0][:5] == ["docker", "exec", "-u", "root", "container-123"]
    assert not any("nemoclaw" in part for part in calls[0])
