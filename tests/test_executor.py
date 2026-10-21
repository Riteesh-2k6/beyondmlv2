import asyncio
import json
import subprocess


import pytest

from beyondml.harness.executor import Executor
from beyondml.harness.skills import discover
from beyondml.workspace.events import EventBus
from beyondml.workspace.plan import Plan, load_plan, save_plan
from beyondml.workspace.project import Project
from scripted_llm import fc, last_response, scripted_resolver, txt


@pytest.fixture(autouse=True)
def _home(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("BEYONDML_HOME", str(tmp_path_factory.mktemp("bhome")))


def step(id, deps=(), expects=(), **kw):
    return {"id": id, "skill": "python-step", "goal": f"do {id}", "depends_on": list(deps),
            "success_check": {"expects": list(expects)}, **kw}


def make(tmp_path, *steps):
    project = Project.init(tmp_path, goal="g")
    plan = Plan.from_dict({"goal": "g", "metric": "f1", "budget": {}, "steps": list(steps)})
    save_plan(project, plan)
    return project, EventBus(project), plan


def writer(id, out, sleep=0.0):
    code = f"import time; time.sleep({sleep}); open('{out}', 'w').write('1')\n"
    return [[fc("write_file", path=f"src/{id}.py", content=code)], [fc("run_python", script=f"src/{id}.py")],
            [fc("finish_step", summary=f"{id} done", outputs=[out])], [txt("finished")]]


async def never_asked(payload):
    raise AssertionError(f"approver called: {payload}")


async def no_answer(q):
    return "no"


def executor(project, bus, plan, scripts, *, mode="autonomous", approver=never_asked, **kw):
    return Executor(project, bus, plan, discover(project), resolver=scripted_resolver(scripts), mode=mode,
                    asker=no_answer, approver=approver, **kw)


def subjects(root):
    out = subprocess.run(["git", "log", "--reverse", "--format=%s"], cwd=root, capture_output=True, text=True)
    return out.stdout.splitlines()


async def test_two_steps_in_order(tmp_path):
    project, bus, plan = make(tmp_path, step("a", expects=["out_a.csv"]), step("b", ["a"], expects=["out_b.csv"]))
    result = await executor(project, bus, plan, {"a": [writer("a", "out_a.csv")], "b": [writer("b", "out_b.csv")]}).run()
    assert [s.status for s in result.steps] == ["done", "done"]
    assert len(project.ledger()) >= 2
    assert (tmp_path / "agents/a/events.jsonl").exists() and (tmp_path / "agents/b/events.jsonl").exists()
    log = [s for s in subjects(tmp_path) if s in ("start a", "done a", "start b", "done b")]
    assert log == ["start a", "done a", "start b", "done b"]


async def test_independent_steps_parallel(tmp_path):
    project, bus, plan = make(tmp_path, step("x", expects=["x.csv"]), step("y", expects=["y.csv"]))
    scripts = {"x": [writer("x", "x.csv", 1.5)], "y": [writer("y", "y.csv", 1.5)]}
    result = await executor(project, bus, plan, scripts).run()
    assert [s.status for s in result.steps] == ["done", "done"]
    runs = {r["agent"]: (r["ts"] - r["seconds"], r["ts"]) for r in project.ledger() if r["agent"] in ("x", "y")}
    (x0, x1), (y0, y1) = runs["x"], runs["y"]
    assert x0 < y1 and y0 < x1  # the two 1.5 s scripts ran at the same time


async def test_check_gates_done(tmp_path):
    project, bus, plan = make(tmp_path, step("a", expects=["missing.csv"]), step("b", ["a"]))
    claim = [[fc("finish_step", summary="claimed", outputs=["missing.csv"])], [txt("ok")]]
    result = await executor(project, bus, plan, {"a": [claim]}).run()
    a, b = result.step("a"), result.step("b")
    assert (a.attempts, a.status, b.status) == (2, "failed", "skipped")
    assert "missing.csv" in a.last_failure


async def test_retry_restores_touched(tmp_path):
    project, bus, plan = make(tmp_path, step("a", expects=["good.csv"]))
    seen = {}

    def check_listing(req):
        seen["junk"] = "junk.txt" in last_response(req).get("files", [])
        return [fc("write_file", path="good.csv", content="1")]

    first = [[fc("write_file", path="junk.txt", content="x")], [fc("finish_step", summary="", outputs=[])], [txt(".")]]
    second = [[fc("list_files", path=".")], check_listing, [fc("finish_step", summary="ok", outputs=["good.csv"])],
              [txt(".")]]
    result = await executor(project, bus, plan, {"a": [first, second]}).run()
    assert result.step("a").status == "done" and seen == {"junk": False}


async def test_replanner_after_two_failures(tmp_path):
    project, bus, plan = make(tmp_path, step("a", expects=["never.csv"]))
    calls = []

    async def replanner(old, failure):
        calls.append(failure)
        return Plan.from_dict({"goal": "g", "metric": "f1", "budget": {}, "steps": [step("a2")]})

    claim = [[fc("finish_step", summary="nope", outputs=[])], [txt(".")]]
    trivial = [[fc("finish_step", summary="ok", outputs=[])], [txt(".")]]
    result = await executor(project, bus, plan, {"a": [claim], "a2": [trivial]}, replanner=replanner).run()
    assert result.is_finished() and [s.id for s in result.steps] == ["a2"] and result.step("a2").status == "done"
    assert len(calls) == 1 and calls[0]["step"] == "a"


@pytest.mark.parametrize("approve,records", [(True, 1), (False, 0)])
async def test_ask_mode_approval(tmp_path, approve, records):
    project, bus, plan = make(tmp_path, step("a"))
    asked = []

    async def approver(payload):
        asked.append(payload)
        return approve

    script = [[fc("write_file", path="src/a.py", content="print(1)\n")], [fc("run_python", script="src/a.py")],
              [fc("finish_step", summary="ok", outputs=[])], [txt(".")]]
    await executor(project, bus, plan, {"a": [script]}, mode="ask", approver=approver).run()
    assert [(p["kind"], p["tool"]) for p in asked] == [("tool", "run_python")]
    assert len(project.ledger()) == records


async def test_autonomous_never_asks(tmp_path):
    project, bus, plan = make(tmp_path, step("a", expects=["out_a.csv"]))
    result = await executor(project, bus, plan, {"a": [writer("a", "out_a.csv")]}).run()
    assert result.step("a").status == "done"


async def test_tool_call_cap(tmp_path):
    project, bus, plan = make(tmp_path, step("a"))

    async def refuse(payload):
        return False

    loop = [[fc("list_files", path=".")]]
    result = await executor(project, bus, plan, {"a": [loop]}, approver=refuse, max_tool_calls=2).run()
    assert result.step("a").status == "failed"
    assert any(e["kind"] == "budget" for e in bus.replay())


async def test_three_tool_errors_fail(tmp_path):
    project, bus, plan = make(tmp_path, step("a"))
    bad = [[fc("write_file", path="../x", content="1")]]
    result = await executor(project, bus, plan, {"a": [bad]}).run()
    assert result.step("a").status == "failed" and result.step("a").last_failure == "three tool errors in a row"


async def test_resume_restores_only_touched_files(tmp_path):
    project, bus, plan = make(tmp_path, step("a", expects=["out_a.csv"]), step("b", ["a"], expects=["out_b.csv"]))
    (tmp_path / "data/raw/d.csv").write_text("x\n1\n")
    project.write("out_a.csv", "1")
    plan.step("a").status, plan.step("a").outputs = "done", ["out_a.csv"]
    save_plan(project, plan)
    project.checkpoint("a finished earlier")
    never = asyncio.Event()

    async def hang(req):
        await never.wait()
        return [txt("unreachable")]

    first = executor(project, bus, plan, {"b": [[[fc("write_file", path="partial.txt", content="p")], hang]]})
    task = asyncio.create_task(first.run())
    for _ in range(200):
        await asyncio.sleep(0.05)
        on_disk = load_plan(project).step("b")
        if on_disk.status == "running" and "partial.txt" in on_disk.touched:
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert load_plan(project).step("b").status == "running" and (tmp_path / "partial.txt").exists()

    seen = {}

    def check_listing(req):
        seen["partial"] = "partial.txt" in last_response(req).get("files", [])
        return [fc("write_file", path="out_b.csv", content="1")]

    second = [[fc("list_files", path=".")], check_listing, [fc("finish_step", summary="ok", outputs=["out_b.csv"])],
              [txt(".")]]
    result = await executor(project, bus, load_plan(project), {"b": [second]}).run()
    assert result.step("b").status == "done" and seen == {"partial": False}
    assert (tmp_path / "out_a.csv").exists() and (tmp_path / "data/raw/d.csv").exists()


async def test_events_shape(tmp_path):
    project, bus, plan = make(tmp_path, step("a", expects=["out_a.csv"]))
    await executor(project, bus, plan, {"a": [writer("a", "out_a.csv")]}).run()
    files = list((tmp_path / "agents").glob("*/events.jsonl"))
    assert files
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            assert set(json.loads(line)) == {"ts", "agent", "kind", "data"}
