import copy

import pytest

from beyondml.workspace.plan import Plan, load_plan, save_plan
from beyondml.workspace.project import Project

SKILLS = {"eda", "train"}


@pytest.fixture(autouse=True)
def _home(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("BEYONDML_HOME", str(tmp_path_factory.mktemp("bhome")))


def make(*steps) -> Plan:
    return Plan.from_dict({"goal": "g", "metric": "acc", "budget": {"minutes": 5}, "steps": list(steps)})


def s(id, deps=(), skill="eda", goal="do it", **kw):
    return {"id": id, "skill": skill, "goal": goal, "depends_on": list(deps), **kw}


def test_roundtrip(tmp_path):
    p = Project.init(tmp_path)
    plan = make(s("a"), s("b", ["a"], skill="train"))
    plan.step("b").status = "done"
    save_plan(p, plan)
    assert load_plan(p).to_dict() == plan.to_dict()
    md = p.read("PLAN.md")
    assert md.startswith("# Plan: g")
    assert "a" in md and "b" in md and "done" in md and "pending" in md


@pytest.mark.parametrize("mutation,needle", [
    ("dup_id", "duplicate"), ("bad_id", "identifier"), ("unknown_dep", "unknown dependency"),
    ("cycle", "cycle"), ("unknown_skill", "unknown skill"), ("empty_goal", "goal")])
def test_validate_errors(mutation, needle):
    steps = {
        "dup_id": [s("a"), s("a")],
        "bad_id": [s("Bad-Id")],
        "unknown_dep": [s("a", ["zzz"])],
        "cycle": [s("a", ["b"]), s("b", ["a"])],
        "unknown_skill": [s("a", skill="nope")],
        "empty_goal": [s("a", goal=" ")],
    }[mutation]
    errs = make(*steps).validate(SKILLS)
    assert any(needle in e for e in errs), errs
    if mutation == "cycle":
        assert any("a -> b -> a" in e for e in errs), errs


def test_validate_clean():
    assert make(s("a"), s("b", ["a"])).validate(SKILLS) == []


def test_ready_respects_deps():
    plan = make(s("a"), s("b", ["a"]), s("c"))
    assert [x.id for x in plan.ready()] == ["a", "c"]
    plan.step("a").status = "done"
    assert [x.id for x in plan.ready()] == ["b", "c"]
    plan.step("a").status = "skipped"
    assert [x.id for x in plan.ready()] == ["b", "c"]
    plan.step("c").status = "done"
    assert [x.id for x in plan.ready()] == ["b"]


def test_failed_dep_blocks():
    plan = make(s("a"), s("b", ["a"]))
    plan.step("a").status = "failed"
    assert plan.ready() == []
    assert [x.id for x in plan.blocked()] == ["b"]
    assert not plan.is_finished()
    plan.step("b").status = "skipped"
    assert plan.is_finished()


def test_merge_keeps_done_steps():
    old = make(s("a", goal="old goal"))
    old.step("a").status = "done"
    new = make(s("a", goal="new goal"), s("b", ["a"]))
    merged = old.merge_revision(new)
    assert merged.step("a").status == "done" and merged.step("a").goal == "old goal"
    assert merged.step("b").status == "pending"
    assert new.step("a").goal == "new goal"  # inputs untouched


def test_from_dict_rejects_garbage():
    with pytest.raises(ValueError):
        Plan.from_dict({"steps": "x"})
    with pytest.raises(ValueError):
        Plan.from_dict({"goal": "g", "metric": "m", "budget": {}, "steps": ["x"]})
    with pytest.raises(ValueError):
        Plan.from_dict({"goal": "g", "metric": "m", "budget": {}, "steps": [{"id": "a"}]})


def test_from_dict_ignores_unknown_step_keys():
    plan = make(s("a", extra="ignored"))
    assert plan.step("a").goal == "do it"
