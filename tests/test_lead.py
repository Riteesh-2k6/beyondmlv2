import json
from pathlib import Path

import pytest
from google.adk.tools import FunctionTool

from beyondml.harness.lead import chat, make_lead
from beyondml.harness.skills import SkillInfo
from beyondml.workspace.events import EventBus
from beyondml.workspace.plan import Plan, load_plan, save_plan
from beyondml.workspace.project import Project
from scripted_llm import fc, scripted_resolver, txt


@pytest.fixture(autouse=True)
def _home(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("BEYONDML_HOME", str(tmp_path_factory.mktemp("bhome")))


SKILLS = {"python-step": SkillInfo("python-step", "Generic step.", "step", None, None, "any", "", Path("."), None)}


def plan(*ids, skill="python-step"):
    return {"goal": "g", "metric": "f1", "budget": {},
            "steps": [{"id": i, "skill": skill, "goal": f"do {i}"} for i in ids]}


async def ask(q):
    return "ok"


def setup(tmp_path, script):
    project = Project.init(tmp_path, goal="g")
    save_plan(project, Plan.from_dict(plan("a", "b")))
    bus = EventBus(project)
    resolver = scripted_resolver({"lead": [script]})
    return project, bus, resolver, make_lead(project, bus, SKILLS, resolver=resolver, mode="auto", asker=ask)


async def test_lead_adds_step(tmp_path):
    project, bus, _, lead = setup(tmp_path, [[fc("update_plan", plan=plan("a", "b", "c"))], [txt("added c")]])
    assert await chat(project, bus, lead, "add a step c") == "added c"
    assert [s.id for s in load_plan(project).steps] == ["a", "b", "c"]
    assert [e["agent"] for e in bus.replay() if e["kind"] == "plan_updated"] == ["lead"]


async def test_lead_invalid_update(tmp_path):
    project, bus, resolver, lead = setup(tmp_path, [[fc("update_plan", plan=plan("a", skill="nope"))], [txt("sorry")]])
    await chat(project, bus, lead, "change it")
    assert [s.id for s in load_plan(project).steps] == ["a", "b"]
    second = resolver.models["lead"][0].seen[1]
    responses = [p.function_response.response for c in second.contents for p in c.parts or [] if p.function_response]
    assert any("unknown skill" in json.dumps(r) for r in responses)


async def test_lead_has_no_run_python(tmp_path):
    _, _, _, lead = setup(tmp_path, [[txt("hi")]])
    names = {t.name for t in lead.tools if isinstance(t, FunctionTool)}
    assert names == {"read_file", "list_files", "ask_user", "update_plan"}
