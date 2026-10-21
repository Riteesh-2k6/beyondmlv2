import json
from pathlib import Path

import pytest

from beyondml.harness.planner import PlanError, make_plan, replan
from beyondml.harness.skills import SkillInfo
from beyondml.workspace.events import EventBus
from beyondml.workspace.plan import Plan, Step, load_plan, save_plan
from beyondml.workspace.project import Project
from scripted_llm import fc, scripted_resolver, txt


@pytest.fixture(autouse=True)
def _home(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("BEYONDML_HOME", str(tmp_path_factory.mktemp("bhome")))


SKILLS = {"python-step": SkillInfo("python-step", "Generic step.", "step", None, None, "any", "", Path("."), None)}


def step(id, deps=(), goal="do it"):
    return {"id": id, "skill": "python-step", "goal": goal, "depends_on": list(deps)}


def plan(*steps):
    return {"goal": "predict churn", "metric": "f1", "budget": {"minutes": 10}, "steps": list(steps)}


GOOD = plan(step("clean"), step("train", ["clean"]))
CYCLE = plan(step("a", ["b"]), step("b", ["a"]))
UNKNOWN = plan({**step("a"), "skill": "nope"})


def _setup(tmp_path, scripts):
    project = Project.init(tmp_path, goal="predict churn")
    resolver = scripted_resolver({"planner": scripts})
    return project, EventBus(project), resolver


async def test_valid_plan_first_try(tmp_path):
    project, bus, resolver = _setup(tmp_path, [[[fc("submit_plan", plan=GOOD)], [txt("done")]]])
    result = await make_plan(project, bus, SKILLS, resolver=resolver, goal="predict churn")
    assert [s.id for s in result.steps] == ["clean", "train"]
    assert load_plan(project).to_dict() == result.to_dict()
    assert "| clean |" in project.read("PLAN.md")
    assert [e["data"] for e in bus.replay() if e["kind"] == "plan_updated"] == [{"steps": ["clean", "train"]}]
    assert resolver.models["planner"][0].calls <= 2


async def test_invalid_then_valid(tmp_path):
    script = [[fc("submit_plan", plan=CYCLE)], [txt("let me fix it")], [fc("submit_plan", plan=GOOD)], [txt("done")]]
    project, bus, resolver = _setup(tmp_path, [script])
    result = await make_plan(project, bus, SKILLS, resolver=resolver, goal="predict churn")
    assert [s.id for s in result.steps] == ["clean", "train"]
    third = resolver.models["planner"][0].seen[2]
    responses = [p.function_response.response for c in third.contents for p in c.parts or [] if p.function_response]
    assert any("cycle" in json.dumps(r) for r in responses)


async def test_three_invalid_plans_raise(tmp_path):
    project, bus, resolver = _setup(tmp_path, [[[fc("submit_plan", plan=UNKNOWN)]]])
    with pytest.raises(PlanError, match="unknown skill"):
        await make_plan(project, bus, SKILLS, resolver=resolver, goal="predict churn")
    assert load_plan(project) is None


async def test_no_tool_call_counts_attempt(tmp_path):
    project, bus, resolver = _setup(tmp_path, [[[txt("I am thinking")]]])
    with pytest.raises(PlanError):
        await make_plan(project, bus, SKILLS, resolver=resolver, goal="predict churn")
    assert resolver.models["planner"][0].calls == 3


async def test_replan_keeps_done(tmp_path):
    revision = plan(step("a", goal="other goal"), step("c", ["a"]))
    project, bus, resolver = _setup(tmp_path, [[[fc("submit_plan", plan=revision)], [txt("done")]]])
    old = Plan.from_dict(plan(step("a", goal="clean"), step("b", ["a"])))
    old.step("a").status, old.step("b").status = "done", "failed"
    save_plan(project, old)
    result = await replan(project, bus, SKILLS, resolver=resolver, plan=old,
                          failure={"step": "b", "reason": "boom", "summary": ""})
    assert result.step("a").status == "done" and result.step("a").goal == "clean"
    assert "c" in [s.id for s in result.steps] and load_plan(project).to_dict() == result.to_dict()


async def test_replan_relisted_failed_step_runs_again(tmp_path):
    old = Plan.from_dict(plan(step("a"), step("b", ["a"])))
    old.step("a").status = "done"
    old.step("b").status, old.step("b").attempts, old.step("b").last_failure = "failed", 2, "boom"
    revision = old.to_dict()  # the model re-submits the plan it was shown, failed step included
    project, bus, resolver = _setup(tmp_path, [[[fc("submit_plan", plan=revision)], [txt("done")]]])
    save_plan(project, old)
    result = await replan(project, bus, SKILLS, resolver=resolver, plan=old,
                          failure={"step": "b", "reason": "boom", "summary": ""})
    b = result.step("b")
    assert (b.status, b.attempts) == ("pending", 0) and not result.is_finished()
    assert result.step("a").status == "done"


async def test_model_plan_cannot_mark_steps_done(tmp_path):
    sneaky = plan({**step("clean"), "status": "done"}, step("train", ["clean"]))
    project, bus, resolver = _setup(tmp_path, [[[fc("submit_plan", plan=sneaky)], [txt("done")]]])
    result = await make_plan(project, bus, SKILLS, resolver=resolver, goal="predict churn")
    assert [s.status for s in result.steps] == ["pending", "pending"]
