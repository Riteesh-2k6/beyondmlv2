import asyncio
import contextlib
import json

import httpx
import pytest
import uvicorn
from click.testing import CliRunner

from beyondml.cli import main
from beyondml.gateway.app import create_app
from beyondml.gateway.session import Session
from beyondml.workspace.plan import Plan, save_plan
from scripted_llm import fc, scripted_resolver, txt
from test_session import ASKER, PLAN, PLANNER, wait_for


@pytest.fixture(autouse=True)
def _home(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("BEYONDML_HOME", str(tmp_path_factory.mktemp("bhome")))


@contextlib.asynccontextmanager
async def served(app):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        if task.done():
            task.result()
        await asyncio.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=30) as client:
            yield client
    finally:
        server.should_exit = True
        await task


async def sse_events(lines, n):
    out = []
    async for line in lines:
        if line.startswith("data:"):
            out.append(json.loads(line[5:]))
            if len(out) == n:
                return out
    return out


async def test_projects_list_create_get(tmp_path):
    app = create_app({"proj": Session(tmp_path / "proj", goal="g")}, dist=None)
    async with served(app) as c:
        listed = (await c.get("/projects")).json()
        assert [p["name"] for p in listed] == ["proj"] and listed[0]["running"] is False
        r = await c.post("/projects", json={"path": str(tmp_path / "new"), "goal": "x"})
        assert r.status_code == 201 and r.json()["name"] == "new" and (tmp_path / "new/.beyondml").is_dir()
        assert (await c.get("/projects/new")).json()["goal"] == "x"
        assert (await c.get("/projects/nope")).status_code == 404
        assert (await c.post("/projects", json={"path": "relative/dir"})).status_code == 400
        assert len((await c.get("/projects")).json()) == 2


async def test_plan_and_agent_routes(tmp_path):
    session = Session(tmp_path, goal="g")
    app = create_app({"p": session}, dist=None)
    async with served(app) as c:
        assert (await c.get("/projects/p/plan")).status_code == 404
        save_plan(session.project, Plan.from_dict(PLAN))
        session.project.write("agents/a/instruction.md", "# do a\nexactly this")
        view = (await c.get("/projects/p/plan")).json()
        assert view["steps"][0]["id"] == "a" and view["steps"][0]["started"] is None and view["default_model"]
        agent = (await c.get("/projects/p/agents/a")).json()
        assert agent["instruction"] == "# do a\nexactly this" and agent["skill"] == "python-step"
        assert "run_python" in agent["tools"] and agent["status"] == "pending" and agent["model"]
        assert (await c.get("/projects/p/agents/zzz")).status_code == 404
        lead = (await c.get("/projects/p/agents/lead")).json()
        assert "update_plan" in lead["tools"] and lead["status"] == "n/a"


async def test_files_guard_and_listing(tmp_path):
    session = Session(tmp_path, goal="g")
    (tmp_path / "src/a.py").write_bytes(b"x = 1\n")  # bytes: write_text would give CRLF on Windows
    app = create_app({"p": session}, dist=None)
    async with served(app) as c:
        assert (await c.get("/projects/p/files/src/a.py")).text == "x = 1\n"
        assert (await c.get("/projects/p/files/src/missing.py")).status_code == 404
        assert (await c.get("/projects/p/files/..%2F..%2Fetc%2Fpasswd")).status_code == 400
        listing = (await c.get("/projects/p/files/src", params={"list": 1})).json()
        assert listing["entries"] == [{"name": "a.py", "dir": False, "size": 6}]
        root = (await c.get("/projects/p/files/", params={"list": 1})).json()
        names = [e["name"] for e in root["entries"]]
        assert "src" in names and ".git" not in names and root["entries"][0]["dir"] is True


async def test_events_replay_then_tail(tmp_path):
    session = Session(tmp_path, goal="g")
    e1 = session.bus.emit("x", "artifact", {"path": "one"})
    session.bus.emit("x", "artifact", {"path": "two"})
    app = create_app({"p": session}, dist=None)
    async with served(app) as c:
        async with c.stream("GET", "/projects/p/events", params={"since": e1["ts"]}) as r:
            assert r.headers["content-type"].startswith("text/event-stream")
            lines = r.aiter_lines()
            (first,) = await asyncio.wait_for(sse_events(lines, 1), 10)
            assert first["data"]["path"] == "two"
            session.bus.emit("x", "artifact", {"path": "three"})
            (live,) = await asyncio.wait_for(sse_events(lines, 1), 10)
            assert live["data"]["path"] == "three"
        async with c.stream("GET", "/projects/p/events", headers={"Last-Event-ID": repr(e1["ts"])}) as r:
            got = await asyncio.wait_for(sse_events(r.aiter_lines(), 2), 10)
            assert [e["data"]["path"] for e in got] == ["two", "three"]


async def test_run_question_answered_through_input(tmp_path):
    resolver = scripted_resolver({"planner": [PLANNER], "a": [ASKER]})
    session = Session(tmp_path / "proj", mode="auto", resolver=resolver, goal="g")
    app = create_app({"proj": session}, dist=None)
    async with served(app) as c:
        assert (await c.post("/projects/proj/run", json={"goal": "g"})).status_code == 202
        await wait_for(lambda: session.pending)
        assert (await c.post("/projects/proj/run", json={"goal": "g"})).status_code == 409
        summary = (await c.get("/projects/proj")).json()
        assert summary["running"] is True and summary["pending"][0]["kind"] == "plan"
        pid = summary["pending"][0]["id"]
        assert (await c.post("/projects/proj/input", json={"type": "answer", "id": pid, "answer": True})).status_code == 200
        await wait_for(lambda: any(e["kind"] == "question" for e in session.pending.values()))
        summary = (await c.get("/projects/proj")).json()
        q = summary["pending"][0]
        assert q["event"] == "question" and q["agent"] == "a" and q["question"] == "which colour?"
        assert (await c.post("/projects/proj/input", json={"type": "answer", "id": q["id"], "answer": "blue"})).status_code == 200
        await wait_for(lambda: not session.running)
        assert (await c.get("/projects/proj/plan")).json()["steps"][0]["status"] == "done"
        assert (await c.post("/projects/proj/stop")).json() == {"stopped": False}


async def test_stop_cancels_parked_run(tmp_path):
    resolver = scripted_resolver({"planner": [PLANNER], "a": [ASKER]})
    session = Session(tmp_path, mode="ask", resolver=resolver, goal="g")
    app = create_app({"p": session}, dist=None)
    async with served(app) as c:
        await c.post("/projects/p/run", json={"goal": "g"})
        await wait_for(lambda: session.pending)
        assert (await c.post("/projects/p/stop")).json() == {"stopped": True}
        await wait_for(lambda: not session.running)
        assert (await c.get("/projects/p")).json()["pending"] == []


async def test_chat_and_input_validation(tmp_path):
    session = Session(tmp_path, resolver=scripted_resolver({"lead": [[[txt("hello there")]]]}), goal="g")
    app = create_app({"p": session}, dist=None)
    async with served(app) as c:
        r = await c.post("/projects/p/input", json={"type": "chat", "text": "hi"})
        assert r.json() == {"reply": "hello there"}
        async with c.stream("GET", "/projects/p/events") as s:
            got = await asyncio.wait_for(sse_events(s.aiter_lines(), 2), 10)
        assert [(e["agent"], e["data"]["text"]) for e in got] == [("user", "hi"), ("lead", "hello there")]
        assert (await c.post("/projects/p/input", json={"type": "dance"})).status_code == 422
        assert (await c.post("/projects/p/input", json={"type": "chat"})).status_code == 400
        assert (await c.post("/projects/p/input", json={"type": "answer", "id": "nope", "answer": 1})).status_code == 404
        assert (await c.post("/projects/p/run", json={})).status_code == 400  # no goal, no plan to resume


async def test_upload_into_data_raw(tmp_path):
    session = Session(tmp_path, goal="g")
    app = create_app({"p": session}, dist=None)
    async with served(app) as c:
        r = await c.post("/projects/p/upload", files={"file": ("sub/dir/t.csv", b"a,b\n1,2\n", "text/csv")})
        assert r.json() == {"path": "data/raw/t.csv"} and (tmp_path / "data/raw/t.csv").read_bytes() == b"a,b\n1,2\n"
        assert (await c.post("/projects/p/upload", files={"file": ("..", b"x")})).status_code == 400


async def test_static_ui_fallback(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>ui</html>")
    (dist / "assets/a.js").write_text("js")
    app = create_app({"p": Session(tmp_path / "proj", goal="g")}, dist=dist)
    async with served(app) as c:
        assert (await c.get("/")).text == "<html>ui</html>"
        assert (await c.get("/assets/a.js")).text == "js"
        assert (await c.get("/p/proj/plan")).text == "<html>ui</html>"
        assert (await c.get("/projects")).json()[0]["name"] == "p"
        assert (await c.get("/projects/p/nothing-here")).status_code == 404


def test_cli_serve_help():
    assert "serve" in main.commands and "web" not in main.commands
    result = CliRunner().invoke(main, ["serve", "--help"])
    assert result.exit_code == 0 and "127.0.0.1" in result.output and "--port" in result.output


def test_cli_serve_outside_project(tmp_path):
    result = CliRunner().invoke(main, ["serve", "--project", str(tmp_path)])
    assert result.exit_code == 2 and "beyondml init" in result.output
