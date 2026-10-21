import pytest
from click.testing import CliRunner

import click

from beyondml.cli import _approve, format_event, main
from beyondml.harness.planner import PlanError
from beyondml.workspace.plan import Plan


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


def test_format_event_step_status_reason():
    ev = {"agent": "e", "kind": "step_status", "data": {"id": "s", "status": "failed", "reason": "boom"}}
    assert format_event(ev) == "[e] step_status: s failed — boom"


def _fake_run_goal(result=None, exc=None):
    async def fake(*args, **kwargs):
        if exc:
            raise exc
        return result
    return fake


def test_cli_run_exits_nonzero_when_a_step_fails(tmp_path, monkeypatch):
    plan = Plan.from_dict({"goal": "g", "metric": "f1", "budget": {},
                           "steps": [{"id": "a", "skill": "python-step", "goal": "a"}]})
    plan.step("a").status = "failed"
    monkeypatch.setattr("beyondml.harness.run.run_goal", _fake_run_goal(plan))
    result = CliRunner().invoke(main, ["run", "g", "--project", str(tmp_path)])
    assert result.exit_code == 1 and "a: failed" in result.output


def test_cli_run_reports_errors_in_one_line(tmp_path, monkeypatch):
    monkeypatch.setattr("beyondml.harness.run.run_goal", _fake_run_goal(exc=PlanError("planner gave up")))
    result = CliRunner().invoke(main, ["run", "g", "--project", str(tmp_path)])
    assert result.exit_code == 1 and "planner gave up" in result.output and "Traceback" not in result.output


def test_cli_run_missing_goal_is_usage_error(tmp_path, monkeypatch):
    from beyondml.harness.run import GoalRequired
    monkeypatch.setattr("beyondml.harness.run.run_goal", _fake_run_goal(exc=GoalRequired("a goal is required")))
    result = CliRunner().invoke(main, ["run", "--project", str(tmp_path)])
    assert result.exit_code == 2 and "goal is required" in result.output


async def test_approve_abort_means_no(monkeypatch):
    def abort(*args, **kwargs):
        raise click.Abort()
    monkeypatch.setattr("click.confirm", abort)
    assert await _approve({"kind": "budget", "agent": None, "reason": "x"}) is False
