from pathlib import Path

import pytest
from google.adk.runners import Runner
from google.adk.tools import FunctionTool

from beyondml.harness.adk import APP_NAME, confirmation_reply, ensure_session, run_turn, session_service, user_text
from beyondml.harness.factory import build_step_agent
from beyondml.harness.skills import SkillInfo
from beyondml.toolkit.env import StepOutcome
from beyondml.workspace.events import EventBus
from beyondml.workspace.plan import Plan, Step
from beyondml.workspace.project import Project
from scripted_llm import ScriptedLlm, fc, scripted_resolver, txt

HEADINGS = ["You are the 'b' agent", "## Your step", "## Upstream results", "## Project", "## Memory",
            "## How to do this step", "## Rules"]


@pytest.fixture(autouse=True)
def _home(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("BEYONDML_HOME", str(tmp_path_factory.mktemp("bhome")))


def _skill(name="demo-step", allowed=None, hint=None, body="Do the demo thing."):
    return SkillInfo(name, "demo", "step", allowed, hint, "any", body, Path("."), None)


async def _ask(q):
    return "ok"


def _setup(tmp_path, *, skill=None, step_model=None, last_failure="", resolver=None, mode="auto"):
    project = Project.init(tmp_path, goal="predict churn")
    bus = EventBus(project)
    skill = skill or _skill()
    a = Step(id="a", skill=skill.name, goal="clean", status="done", summary="made clean.csv")
    b = Step(id="b", skill=skill.name, goal="train", depends_on=["a"], outputs=["models/m.pkl"],
             model=step_model, last_failure=last_failure)
    plan = Plan(goal="predict churn", metric="f1", budget={}, steps=[a, b])
    resolver = resolver or (lambda name, model=None: ScriptedLlm(script=[[txt("hi")]], seen=[]))
    agent = build_step_agent(project, bus, plan, b, {skill.name: skill}, resolver=resolver, mode=mode,
                             asker=_ask, outcome=StepOutcome(), on_touch=lambda paths: None)
    return project, bus, agent


def test_instruction_written_in_order(tmp_path):
    project, _, _ = _setup(tmp_path)
    text = (tmp_path / "agents/b/instruction.md").read_text(encoding="utf-8")
    idx = [text.index(h) for h in HEADINGS]
    assert idx == sorted(idx)
    assert "Do the demo thing." in text and "predict churn" in text and "- a: made clean.csv" in text
    assert "## Previous attempt failed" not in text


def test_previous_failure_section(tmp_path):
    _setup(tmp_path, last_failure="boom")
    text = (tmp_path / "agents/b/instruction.md").read_text(encoding="utf-8")
    assert "## Previous attempt failed" in text and "boom" in text
    assert text.index("## Upstream results") < text.index("## Previous attempt failed") < text.index("## Project")


def test_tools_filtered_by_skill(tmp_path):
    _, _, agent = _setup(tmp_path, skill=_skill(allowed={"read_file", "finish_step"}))
    assert {t.name for t in agent.tools if isinstance(t, FunctionTool)} == {"read_file", "finish_step"}


def test_model_precedence(tmp_path):
    calls = []

    def spy(name, model=None):
        calls.append((name, model))
        return ScriptedLlm(script=[[txt("x")]], seen=[])

    _setup(tmp_path / "p1", skill=_skill(hint="m2"), step_model="m1", resolver=spy)
    _setup(tmp_path / "p2", skill=_skill(hint="m2"), resolver=spy)
    assert calls == [("b", "m1"), ("b", "m2")]


def test_agent_name_is_step_id(tmp_path):
    _, _, agent = _setup(tmp_path)
    assert agent.name == "b"


async def test_run_turn_emits_and_collects(tmp_path):
    resolver = scripted_resolver({"b": [[[fc("write_file", path="src/a.py", content="print(1)")], [txt("done writing")]]]})
    project, bus, agent = _setup(tmp_path, resolver=resolver)
    svc = session_service(project)
    runner = Runner(app_name=APP_NAME, agent=agent, session_service=svc)
    await ensure_session(svc, "s1")
    res = await run_turn(runner, "s1", user_text("go"), bus)
    kinds = [e["kind"] for e in bus.replay() if e["agent"] == "b"]
    assert "tool_call" in kinds and "tool_result" in kinds
    assert res.text == "done writing" and res.confirmations == []
    assert (tmp_path / "src/a.py").read_text() == "print(1)"


async def test_run_turn_confirmation(tmp_path):
    resolver = scripted_resolver({"b": [[[fc("run_python", script="src/a.py")], [txt("ran")]]]})
    project, bus, agent = _setup(tmp_path, resolver=resolver, mode="ask")
    project.write("src/a.py", "print('hi')\n")
    svc = session_service(project)
    runner = Runner(app_name=APP_NAME, agent=agent, session_service=svc)
    await ensure_session(svc, "s1")
    first = await run_turn(runner, "s1", user_text("go"), bus)
    assert first.confirmations[0]["tool"] == "run_python" and project.ledger() == []
    second = await run_turn(runner, "s1", confirmation_reply(first.confirmations[0]["call_id"], True), bus)
    assert len(project.ledger()) == 1 and second.text == "ran"
