"""The HTTP gateway: REST for state, SSE for the live event stream, and the built web UI."""
from __future__ import annotations

import asyncio
import contextlib
import json
import mimetypes
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from beyondml.gateway.session import Session
from beyondml.gateway.state import SessionState, agent_view, plan_view
from beyondml.workspace.events import EventBus
from beyondml.workspace.plan import load_plan
from beyondml.workspace.project import PathError, ProjectError

DIST = Path(__file__).resolve().parent.parent / "web" / "ui" / "dist"
PING_SECONDS = 15.0


class ProjectCreate(BaseModel):
    path: str
    goal: str | None = None


class RunRequest(BaseModel):
    goal: str | None = None
    data: list[str] = []
    mode: Literal["ask", "auto", "autonomous"] | None = None
    model: str | None = None


class Input(BaseModel):
    type: Literal["chat", "answer"]
    text: str | None = None
    id: str | None = None
    answer: Any = None


def _sse(ev: dict) -> str:
    return f"id: {ev['ts']!r}\ndata: {json.dumps(ev, default=str)}\n\n"


async def event_stream(bus: EventBus, since: float | None) -> AsyncIterator[str]:
    """Replay from disk, then tail the bus. Subscribing first means nothing falls in the gap."""
    queue = bus.subscribe()
    try:
        last_ts, last_key = since or 0.0, None
        for ev in bus.replay(since):
            last_ts, last_key = ev["ts"], (ev["agent"], ev["kind"])
            yield _sse(ev)
        while True:
            try:
                ev = await asyncio.wait_for(queue.get(), PING_SECONDS)
            except asyncio.TimeoutError:
                yield ": ping\n\n"
                continue
            if ev["ts"] < last_ts or (ev["ts"] == last_ts and (ev["agent"], ev["kind"]) == last_key):
                continue  # already sent by the replay
            yield _sse(ev)
    finally:
        bus.unsubscribe(queue)


def create_app(registry: dict[str, Session], dist: Path | None = DIST) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(_app):
        for session in registry.values():
            session.bind()  # the harness runs on this loop, whichever thread serves
        yield

    app = FastAPI(title="BeyondML gateway", lifespan=lifespan)

    def get(name: str) -> Session:
        if name not in registry:
            raise HTTPException(404, f"unknown project {name!r}")
        return registry[name]

    def summary(name: str) -> dict:
        return {**registry[name].summary(), "name": name}  # the name you see is the name you address

    # ── projects ─────────────────────────────────────────────
    @app.get("/projects")
    def projects():
        return [{k: v for k, v in summary(name).items() if k in ("name", "root", "goal", "running")}
                for name in registry]

    @app.post("/projects", status_code=201)
    async def create_project(body: ProjectCreate):
        path = Path(body.path)
        if not path.is_absolute():
            raise HTTPException(400, "path must be absolute")
        name = path.resolve().name
        if name in registry:
            if registry[name].project.root.resolve() == path.resolve():
                return summary(name)
            raise HTTPException(409, f"a different project named {name!r} is already open")
        try:
            session = Session(path, goal=body.goal)
        except ProjectError as e:
            raise HTTPException(400, str(e))
        session.bind()
        registry[name] = session
        return summary(name)

    @app.get("/projects/{p}")
    def project(p: str):
        get(p)
        return summary(p)

    @app.get("/projects/{p}/plan")
    def plan(p: str):
        session = get(p)
        view = plan_view(session.project, SessionState.fold(session.bus.replay()))
        if view is None:
            raise HTTPException(404, "no plan yet")
        return view

    @app.get("/projects/{p}/agents/{agent_id}")
    def agent(p: str, agent_id: str):
        session = get(p)
        view = agent_view(session.project, SessionState.fold(session.bus.replay()), agent_id)
        if view is None:
            raise HTTPException(404, f"unknown agent {agent_id!r}")
        return view

    # ── files ────────────────────────────────────────────────
    @app.get("/projects/{p}/files/{path:path}")
    def files(p: str, path: str, list: bool = False):
        project = get(p).project
        try:
            target = project.path(path or ".")
        except PathError as e:
            raise HTTPException(400, str(e))
        if list or target.is_dir():
            if not target.is_dir():
                raise HTTPException(404, f"no such directory: {path}")
            entries = [{"name": c.name, "dir": c.is_dir(), "size": c.stat().st_size if c.is_file() else 0}
                       for c in target.iterdir() if c.name != ".git"]
            entries.sort(key=lambda e: (not e["dir"], e["name"].lower()))
            return {"path": path, "entries": entries}
        if not target.is_file():
            raise HTTPException(404, f"no such file: {path}")
        media = mimetypes.guess_type(target.name)[0] or "text/plain"
        return FileResponse(target, media_type=f"{media}; charset=utf-8" if media.startswith("text/") else media)

    @app.post("/projects/{p}/upload")
    async def upload(p: str, file: UploadFile):
        project = get(p).project
        name = Path(file.filename or "").name
        if not name or name in (".", ".."):
            raise HTTPException(400, "a file name is required")
        dest = project.path("data/raw") / name  # data/raw is read-only to agents, not to the user
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(await file.read())
        return {"path": f"data/raw/{name}"}

    # ── events ───────────────────────────────────────────────
    @app.get("/projects/{p}/events")
    async def events(p: str, request: Request, since: float | None = None):
        session = get(p)
        last_id = request.headers.get("last-event-id")
        if last_id:
            with contextlib.suppress(ValueError):
                since = float(last_id)
        return StreamingResponse(event_stream(session.bus, since), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ── input and run control ────────────────────────────────
    @app.post("/projects/{p}/input")
    async def post_input(p: str, body: Input):
        session = get(p)
        if body.type == "chat":
            if not body.text or not body.text.strip():
                raise HTTPException(400, "text is required")
            return {"reply": await session.call(session.chat(body.text))}
        if not body.id:
            raise HTTPException(400, "id is required")
        try:
            await session.call(session.answer(body.id, body.answer))
        except KeyError:
            raise HTTPException(404, f"no pending prompt {body.id!r}")
        return {"ok": True}

    @app.post("/projects/{p}/run", status_code=202)
    async def run(p: str, body: RunRequest):
        session = get(p)
        if session.running:
            raise HTTPException(409, "a run is already active")
        if not body.goal:
            try:
                current = load_plan(session.project)
            except ValueError:
                current = None
            if current is None or current.is_finished():
                raise HTTPException(400, "a goal is required to start a new plan")
        for d in body.data:
            if not Path(d).exists():
                raise HTTPException(400, f"data path not found: {d}")
        try:
            await session.call(session.start(body.goal, [Path(d) for d in body.data], mode=body.mode,
                                             model=body.model))
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        return {"ok": True}

    @app.post("/projects/{p}/stop")
    async def stop(p: str):
        session = get(p)

        async def do_stop():
            return session.stop()

        return {"stopped": await session.call(do_stop())}

    # ── the built web UI ─────────────────────────────────────
    if dist is not None and (dist / "index.html").is_file():
        if (dist / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=str(dist / "assets")), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def ui(path: str):
            if path.split("/", 1)[0] == "projects":
                raise HTTPException(404)
            candidate = (dist / path).resolve() if path else None
            if candidate and candidate.is_file() and candidate.is_relative_to(dist.resolve()):
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    return app
