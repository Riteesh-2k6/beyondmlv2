import asyncio
import json

import pytest
from google.adk.events import Event, EventActions
from google.genai import types

from beyondml.workspace.events import EventBus, normalize
from beyondml.workspace.project import Project


@pytest.fixture(autouse=True)
def _home(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("BEYONDML_HOME", str(tmp_path_factory.mktemp("bhome")))


def ev(*parts, **kw):
    return Event(author="tool", invocation_id="i1", content=types.Content(role="model", parts=list(parts)), **kw)


def test_emit_writes_and_fans_out(tmp_path):
    bus = EventBus(Project.init(tmp_path))
    q = bus.subscribe()
    e = bus.emit("x", "budget", {"n": 1})
    assert q.get_nowait() == e
    lines = (tmp_path / "agents/x/events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and set(json.loads(lines[0])) == {"ts", "agent", "kind", "data"}
    bus.unsubscribe(q)
    bus.emit("x", "budget", {})
    assert q.empty()


def test_unknown_kind_rejected(tmp_path):
    with pytest.raises(ValueError):
        EventBus(Project.init(tmp_path)).emit("x", "nope", {})


def test_replay_merges_and_filters(tmp_path):
    bus = EventBus(Project.init(tmp_path))
    first = bus.emit("a", "budget", {})
    bus.emit("b", "budget", {})
    bus.emit("a", "budget", {})
    (tmp_path / "agents/a/events.jsonl").open("a").write("\nnot json\n")
    out = bus.replay()
    assert [e["ts"] for e in out] == sorted(e["ts"] for e in out) and len(out) == 3
    assert len(bus.replay(since=first["ts"])) == 2


def test_normalize_text_and_tools():
    out = normalize(ev(
        types.Part(text="hi"),
        types.Part(function_call=types.FunctionCall(id="c1", name="run_python", args={"script": "x.py"})),
        types.Part(function_response=types.FunctionResponse(id="c1", name="run_python", response={"ok": True})),
    ))
    assert [k for _, k, _ in out] == ["message_delta", "tool_call", "tool_result"]
    assert out[0] == ("tool", "message_delta", {"text": "hi", "partial": False})
    assert out[1][2] == {"id": "c1", "name": "run_python", "args": {"script": "x.py"}}


def test_normalize_confirmation():
    args = {"originalFunctionCall": {"id": "c1", "name": "run_python", "args": {"s": 1}},
            "toolConfirmation": {"hint": "ok?", "confirmed": False}}
    [(_, kind, data)] = normalize(ev(types.Part(function_call=types.FunctionCall(id="cf", name="adk_request_confirmation", args=args))))
    assert kind == "approval_request"
    assert data == {"call_id": "cf", "tool": "run_python", "args": {"s": 1}, "hint": "ok?"}


def test_normalize_bml_event():
    e = Event(author="tool", invocation_id="i1",
              actions=EventActions(state_delta={"bml_event": {"kind": "generation", "data": {"gen": 1}}}))
    assert normalize(e) == [("tool", "generation", {"gen": 1})]


def test_normalize_skips_thoughts():
    assert normalize(ev(types.Part(text="x", thought=True))) == []


def _bad(out):
    assert len(out) == 1 and out[0][1] == "error" and out[0][2]["code"] == "bad_event"


def test_normalize_malformed_confirmation():
    for args in ({}, {"originalFunctionCall": {"id": "c"}, "toolConfirmation": {}},
                 {"originalFunctionCall": "x", "toolConfirmation": {}},
                 {"originalFunctionCall": {"name": "n"}}):
        _bad(normalize(ev(types.Part(function_call=types.FunctionCall(id="cf", name="adk_request_confirmation", args=args)))))


def test_normalize_bad_bml_event():
    for v in ({"kind": "nope", "data": {}}, "str", {"data": {}}):
        _bad(normalize(Event(author="a", invocation_id="i", actions=EventActions(state_delta={"bml_event": v}))))


def test_normalize_error_code():
    e = Event(author="a", invocation_id="i", error_code="X", error_message="m")
    assert normalize(e) == [("a", "error", {"code": "X", "message": "m"})]


def test_normalize_partial_flag():
    [(_, _, d)] = normalize(ev(types.Part(text="x"), partial=True))
    assert d["partial"] is True


def test_emit_from_worker_thread_reaches_subscriber(tmp_path):
    bus = EventBus(Project.init(tmp_path))

    async def go():
        q = bus.subscribe()
        await asyncio.to_thread(bus.emit, "x", "budget", {"n": 1})
        return await asyncio.wait_for(q.get(), 2)

    assert asyncio.run(go())["data"] == {"n": 1}
