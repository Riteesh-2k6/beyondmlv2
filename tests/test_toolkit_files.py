import json

import pytest

from beyondml.toolkit.control import control_tools
from beyondml.toolkit.env import AUTONOMOUS_ANSWER, StepOutcome, ToolEnv
from beyondml.toolkit.files import file_tools
from beyondml.workspace.events import EventBus
from beyondml.workspace.plan import load_plan
from beyondml.workspace.project import Project


@pytest.fixture(autouse=True)
def _home(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("BEYONDML_HOME", str(tmp_path_factory.mktemp("bhome")))


def mk(tmp_path, mode="auto", asker=None, touched=None):
    project = Project.init(tmp_path / "p")

    async def default(q):
        raise AssertionError("asker called")

    return ToolEnv(project, EventBus(project), "s1", mode, asker or default, skill_names={"eda"},
                   outcome=StepOutcome(), on_touch=(touched.extend if touched is not None else None))


def tools(funcs):
    return {f.__name__: f for f in funcs}


def kinds(env):
    lines = env.project.path("agents/s1/events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(x)["kind"] for x in lines]


async def test_file_tools_roundtrip(tmp_path):
    touched = []
    env = mk(tmp_path, touched=touched)
    t = tools(file_tools(env))
    assert t["write_file"]("src/a.py", "x=1")["ok"]
    assert t["read_file"]("src/a.py")["content"] == "x=1"
    assert "src/a.py" in t["list_files"]("src")["files"]
    assert "artifact" in kinds(env) and touched == ["src/a.py"]


async def test_file_tools_refuse_escapes(tmp_path):
    env = mk(tmp_path)
    t = tools(file_tools(env))
    for r in (t["write_file"]("../x", "1"), t["write_file"]("data/raw/a.csv", "1"),
              t["read_file"](str(tmp_path / "p" / "PLAN.md"))):
        assert r["ok"] is False and "error" in r
    assert not (tmp_path / "x").exists() and not (tmp_path / "p/data/raw/a.csv").exists()


async def test_read_truncates(tmp_path):
    env = mk(tmp_path)
    t = tools(file_tools(env))
    t["write_file"]("big.txt", "\n".join(str(i) for i in range(1000)))
    r = t["read_file"]("big.txt", max_lines=400)
    assert len(r["content"].splitlines()) == 400 and r["truncated"] is True


async def test_ask_user_autonomous(tmp_path):
    env = mk(tmp_path, mode="autonomous")
    r = await tools(control_tools(env))["ask_user"]("which?")
    assert r["answer"] == AUTONOMOUS_ANSWER and "question" in kinds(env)


async def test_ask_user_asks(tmp_path):
    async def yes(q):
        return "yes"

    env = mk(tmp_path, asker=yes)
    assert (await tools(control_tools(env))["ask_user"]("ok?"))["answer"] == "yes"


async def test_finish_step_sets_outcome(tmp_path):
    env = mk(tmp_path)
    tools(control_tools(env))["finish_step"]("done", ["a.txt"])
    assert env.outcome.finished and env.outcome.summary == "done" and env.outcome.outputs == ["a.txt"]


def plan(steps):
    return {"goal": "g", "metric": "m", "budget": {}, "steps": steps}


def step(id, deps=(), status="pending"):
    return {"id": id, "skill": "eda", "goal": "x", "depends_on": list(deps), "status": status}


async def test_update_plan_rejects_invalid(tmp_path):
    env = mk(tmp_path)
    up = tools(control_tools(env))["update_plan"]
    assert up(plan([step("a")]))["ok"]
    before = env.project.read("plan.json")
    r = up(plan([step("a", ["b"]), step("b", ["a"])]))
    assert r["ok"] is False and r["errors"] and env.project.read("plan.json") == before


async def test_update_plan_keeps_done(tmp_path):
    env = mk(tmp_path)
    up = tools(control_tools(env))["update_plan"]
    up(plan([step("a", status="done")]))
    assert up(plan([step("a")]))["ok"]
    assert load_plan(env.project).step("a").status == "done"
