import asyncio
import json

import pytest

from beyondml.gateway.session import Session
from beyondml.gateway.state import SessionState, plan_view
from beyondml.workspace.plan import load_plan
from scripted_llm import fc, last_response, scripted_resolver, txt


@pytest.fixture(autouse=True)
def _home(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("BEYONDML_HOME", str(tmp_path_factory.mktemp("bhome")))


PLAN = {"goal": "g", "metric": "accuracy", "budget": {}, "steps": [
    {"id": "a", "skill": "python-step", "goal": "do a"}]}
PLANNER = [[fc("submit_plan", plan=PLAN)], [txt("planned")]]
ASKER = [[fc("ask_user", question="which colour?")], [fc("finish_step", summary="a done", outputs=[])], [txt("ok")]]
METRIC_PY = 'import json, os\njson.dump({"accuracy": 0.91}, open(os.environ["BEYONDML_METRICS"], "w"))\n'
SCORER = [[fc("write_file", path="src/a.py", content=METRIC_PY)], [fc("run_python", script="src/a.py")],
          [fc("finish_step", summary="scored", outputs=[])], [txt("ok")]]


async def wait_for(pred, timeout=15.0):
    for _ in range(int(timeout / 0.02)):
        if pred():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("timed out waiting")


def first_pending(session):
    pid = next(iter(session.pending))
    return pid, session.pending[pid]


async def test_run_parks_plan_approval_and_question(tmp_path):
    resolver = scripted_resolver({"planner": [PLANNER], "a": [ASKER]})
    session = Session(tmp_path / "proj", mode="auto", resolver=resolver, goal="g")
    await session.start("g")
    assert session.running

    await wait_for(lambda: session.pending)
    pid, entry = first_pending(session)
    assert entry["kind"] == "approval_request" and entry["data"]["kind"] == "plan"
    assert entry["data"]["plan"]["steps"][0]["id"] == "a" and entry["agent"] == "planner"
    await session.answer(pid, True)

    await wait_for(lambda: any(e["kind"] == "question" for e in session.pending.values()))
    pid, entry = first_pending(session)
    assert entry["agent"] == "a" and entry["data"]["question"] == "which colour?"
    await session.answer(pid, "blue")

    await session.run_task
    assert not session.running and load_plan(session.project).step("a").status == "done"
    assert last_response(resolver.models["a"][0].seen[1]) == {"answer": "blue"}

    events = session.bus.replay()
    questions = [e for e in events if e["kind"] == "question" and "id" in e["data"]]
    assert [e["agent"] for e in questions] == ["a", "a"] and questions[1]["data"]["answer"] == "blue"
    approvals = [e for e in events if e["kind"] == "approval_request" and "id" in e["data"]]
    assert len(approvals) == 2 and approvals[1]["data"]["answer"] is True
    assert session.pending == {}


async def test_answer_unknown_id_raises(tmp_path):
    session = Session(tmp_path, goal="g")
    session.bind()
    with pytest.raises(KeyError):
        await session.answer("nope", True)


async def test_start_twice_raises_and_stop_cancels(tmp_path):
    resolver = scripted_resolver({"planner": [PLANNER], "a": [ASKER]})
    session = Session(tmp_path, mode="autonomous", resolver=resolver, goal="g")
    await session.start("g")
    with pytest.raises(RuntimeError):
        await session.start("g")
    # autonomous mode never asks, so the run finishes on its own
    await session.run_task
    assert session.stop() is False

    asker = scripted_resolver({"planner": [PLANNER], "a": [ASKER]})
    held = Session(tmp_path / "two", mode="ask", resolver=asker, goal="g")
    await held.start("g")
    await wait_for(lambda: held.pending)
    assert held.stop() is True
    await wait_for(lambda: not held.running)
    assert held.pending == {}
    assert any(e["kind"] == "error" and e["data"]["code"] == "stopped" for e in held.bus.replay())


async def test_chat_emits_user_line_and_returns_reply(tmp_path):
    session = Session(tmp_path, resolver=scripted_resolver({"lead": [[[txt("hello there")]]]}), goal="g")
    session.bind()
    assert await session.chat("hi") == "hello there"
    chat = [(e["agent"], e["data"]["text"]) for e in session.bus.replay() if e["kind"] == "message_delta"]
    assert chat == [("user", "hi"), ("lead", "hello there")]


async def test_summary_and_plan_view(tmp_path):
    resolver = scripted_resolver({"planner": [PLANNER], "a": [SCORER]})
    session = Session(tmp_path / "proj", mode="autonomous", resolver=resolver, goal="predict it")
    before = session.summary()
    assert before["name"] == "proj" and before["goal"] == "predict it" and before["running"] is False
    assert before["mode"] == "autonomous" and before["pending"] == []
    assert plan_view(session.project, SessionState()) is None

    await session.start("g")
    await session.run_task
    state = SessionState.fold(session.bus.replay())
    view = plan_view(session.project, state)
    step = view["steps"][0]
    assert step["status"] == "done" and step["finished"] >= step["started"] > 0
    assert step["metric"] == {"name": "accuracy", "value": 0.91}
    assert view["default_model"] == session.project.settings()["model"]["default"]


def test_fold_pending_resolved_and_series():
    events = [
        {"ts": 1.0, "agent": "executor", "kind": "step_status", "data": {"id": "a", "status": "running", "reason": ""}},
        {"ts": 2.0, "agent": "a", "kind": "question", "data": {"question": "raw, no id"}},
        {"ts": 2.1, "agent": "a", "kind": "question", "data": {"id": "q1", "question": "which?"}},
        {"ts": 2.2, "agent": "planner", "kind": "approval_request", "data": {"id": "p1", "kind": "plan", "plan": {}}},
        {"ts": 2.3, "agent": "a", "kind": "question", "data": {"id": "q1", "question": "which?", "answer": "x"}},
        {"ts": 3.0, "agent": "a", "kind": "generation", "data": {"gen": 1, "best_fitness": 0.8, "mean_fitness": 0.5,
                                                                   "diversity": 1.0, "evals": 5, "seconds": 1.0,
                                                                   "best_params": {"k": 1}}},
        {"ts": 3.1, "agent": "a", "kind": "chart", "data": {"title": "t", "kind": "line", "x": [1], "series": {"s": [1]}}},
        {"ts": 3.2, "agent": "a", "kind": "artifact", "data": {"path": "src/a.py"}},
        {"ts": 3.3, "agent": "user", "kind": "message_delta", "data": {"text": "hi", "partial": False}},
        {"ts": 3.4, "agent": "lead", "kind": "message_delta", "data": {"text": "he", "partial": True}},
        {"ts": 3.5, "agent": "lead", "kind": "message_delta", "data": {"text": "hello", "partial": False}},
        {"ts": 4.0, "agent": "executor", "kind": "step_status", "data": {"id": "a", "status": "done", "reason": ""}},
    ]
    state = SessionState.fold(events)
    assert state.steps["a"] == {"status": "done", "started": 1.0, "finished": 4.0, "reason": ""}
    assert list(state.pending) == ["p1"] and state.pending["p1"]["event"] == "approval_request"
    assert state.pending["p1"]["agent"] == "planner"
    assert [g["gen"] for g in state.generations["a"]] == [1]
    assert state.charts["a"][0]["title"] == "t" and state.artifacts["a"] == ["src/a.py"]
    assert [(m["agent"], m["text"]) for m in state.chat] == [("user", "hi"), ("lead", "hello")]
    assert state.last_ts == 4.0
    # incremental apply agrees with a one-shot fold
    inc = SessionState()
    for e in events:
        inc.apply(e)
    assert json.dumps(inc.to_dict(), sort_keys=True) == json.dumps(state.to_dict(), sort_keys=True)
