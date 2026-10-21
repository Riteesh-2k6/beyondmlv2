import pytest
from click.testing import CliRunner

from beyondml.cli import format_event, main


@pytest.fixture(autouse=True)
def _home(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("BEYONDML_HOME", str(tmp_path_factory.mktemp("bhome")))


def test_cli_init(tmp_path):
    result = CliRunner().invoke(main, ["init", str(tmp_path / "p"), "--goal", "predict x"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "p/.beyondml").is_dir() and "predict x" in (tmp_path / "p/PROJECT.md").read_text()


def test_cli_run_help():
    result = CliRunner().invoke(main, ["run", "--help"])
    assert result.exit_code == 0 and "--mode" in result.output and "autonomous" in result.output


def test_cli_has_tui_command():
    assert "tui" in main.commands


def test_format_event():
    assert format_event({"agent": "a", "kind": "message_delta", "data": {"text": "hi", "partial": True}}) is None
    line = format_event({"agent": "a", "kind": "tool_call", "data": {"name": "run_python", "args": {"script": "x.py"}}})
    assert line.startswith("[a] tool_call: run_python(")
    assert format_event({"agent": "e", "kind": "step_status", "data": {"id": "s", "status": "done"}}) == \
        "[e] step_status: s done"
    bad = format_event({"agent": "a", "kind": "tool_result", "data": {"name": "w", "response": {"ok": False, "error": "no"}}})
    assert bad == "[a] tool_result: w error: no"
