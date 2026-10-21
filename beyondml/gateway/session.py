"""One project's live session: the harness run, parked prompts, and chat with the lead."""
from __future__ import annotations

import asyncio
import contextlib
import re
import uuid
from pathlib import Path
from typing import Sequence

from beyondml.harness.adk import ModelResolver, default_resolver
from beyondml.harness.lead import chat as lead_chat, make_lead
from beyondml.harness.run import run_goal
from beyondml.harness.skills import discover
from beyondml.workspace.events import EventBus
from beyondml.workspace.plan import load_plan
from beyondml.workspace.project import Project

GATEWAY = "gateway"
TRUE_WORDS = {"y", "yes", "true", "1", "ok", "approve"}


class SessionBus(EventBus):
    """Remembers which agent asked the last question, so the parked prompt is filed under it."""

    last_question_agent: str | None = None

    def emit(self, agent: str, kind: str, data: dict) -> dict:
        if kind == "question" and "id" not in data:
            self.last_question_agent = agent
        return super().emit(agent, kind, data)


class Session:
    def __init__(self, root: Path, *, mode: str | None = None, model: str | None = None,
                 resolver: ModelResolver | None = None, goal: str | None = None) -> None:
        root = Path(root)
        self.project = Project.open(root) if (root / ".beyondml").is_dir() else Project.init(root, goal)
        self.name = self.project.root.resolve().name
        self.bus = SessionBus(self.project)
        self.mode, self.model, self.resolver = mode, model, resolver
        self.pending: dict[str, dict] = {}  # id -> {agent, kind, data, future}
        self.run_task: asyncio.Task | None = None
        self.loop: asyncio.AbstractEventLoop | None = None

    # ── loop plumbing ────────────────────────────────────────
    def bind(self) -> None:
        self.loop = asyncio.get_running_loop()

    async def call(self, coro):
        """Await a session coroutine from any loop (the TUI lives on another thread than the gateway)."""
        if self.loop is None:
            self.bind()
        if self.loop is asyncio.get_running_loop():
            return await coro
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, self.loop))

    # ── the run ──────────────────────────────────────────────
    @property
    def running(self) -> bool:
        return self.run_task is not None and not self.run_task.done()

    async def start(self, goal: str | None, data: Sequence[Path] = (), *, mode: str | None = None,
                    model: str | None = None) -> None:
        if self.running:
            raise RuntimeError("a run is already active")
        if self.loop is None:
            self.bind()
        self.run_task = asyncio.create_task(self._run(goal, [Path(p) for p in data], mode or self.mode,
                                                      model or self.model))

    async def _run(self, goal, data, mode, model) -> None:
        try:
            await run_goal(self.project.root, goal, data, mode=mode, model=model, resolver=self.resolver,
                           asker=self._ask, approver=self._approve, bus=self.bus)
        except asyncio.CancelledError:
            self.bus.emit(GATEWAY, "error", {"code": "stopped", "message": "run stopped by the user"})
        except Exception as e:  # the session outlives a failed run; clients see why
            self.bus.emit(GATEWAY, "error", {"code": type(e).__name__, "message": str(e)})
        finally:
            for entry in list(self.pending.values()):
                entry["future"].cancel()
            self.pending.clear()

    def stop(self) -> bool:
        if not self.running:
            return False
        self.run_task.cancel()
        return True

    # ── prompts ──────────────────────────────────────────────
    async def _park(self, agent: str, kind: str, data: dict):
        pid = uuid.uuid4().hex[:8]
        future = asyncio.get_running_loop().create_future()
        data = {"id": pid, **data}
        self.pending[pid] = {"agent": agent, "kind": kind, "data": data, "future": future}
        self.bus.emit(agent, kind, data)
        try:
            return await future
        finally:
            self.pending.pop(pid, None)

    async def _ask(self, question: str) -> str:
        return await self._park(self.bus.last_question_agent or "lead", "question", {"question": question})

    async def _approve(self, payload: dict) -> bool:
        agent = payload.get("agent") or {"plan": "planner"}.get(payload.get("kind"), "executor")
        return await self._park(agent, "approval_request", dict(payload))

    async def answer(self, pid: str, answer) -> None:
        entry = self.pending[pid]
        if entry["kind"] == "approval_request":
            answer = answer.strip().lower() in TRUE_WORDS if isinstance(answer, str) else bool(answer)
        else:
            answer = str(answer)
        self.bus.emit(entry["agent"], entry["kind"], {**entry["data"], "answer": answer})
        if not entry["future"].done():
            entry["future"].set_result(answer)

    # ── chat ─────────────────────────────────────────────────
    def _resolver(self) -> ModelResolver:
        base = self.resolver or default_resolver(self.project)
        return (lambda name, requested=None: base(name, self.model)) if self.model else base

    async def chat(self, text: str) -> str:
        self.bus.emit("user", "message_delta", {"text": text, "partial": False})
        mode = self.mode or self.project.settings()["run"]["mode"]
        lead = make_lead(self.project, self.bus, discover(self.project), resolver=self._resolver(), mode=mode,
                         asker=self._ask)
        return await lead_chat(self.project, self.bus, lead, text)

    # ── summary ──────────────────────────────────────────────
    def goal(self) -> str:
        with contextlib.suppress(ValueError):
            plan = load_plan(self.project)
            if plan:
                return plan.goal
        m = re.search(r"^## Goal\s*\n(.+)$", self.project.read("PROJECT.md"), re.M)
        return m.group(1).strip() if m else ""

    def summary(self) -> dict:
        settings = self.project.settings()
        return {
            "name": self.name, "root": str(self.project.root.resolve()), "goal": self.goal(),
            "mode": self.mode or settings["run"]["mode"], "model": self.model or settings["model"]["default"],
            "running": self.running,
            "pending": [{"event": e["kind"], "agent": e["agent"], **e["data"]} for e in self.pending.values()],
        }
