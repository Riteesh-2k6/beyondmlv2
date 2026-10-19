"""Event bus: every agent's activity as {ts, agent, kind, data}, on disk and on subscriber queues."""
from __future__ import annotations

import asyncio
import json
import threading
import time

from google.adk.events import Event

from beyondml.workspace.project import Project

KINDS = ("message_delta", "tool_call", "tool_result", "plan_updated", "step_status", "generation", "trial",
         "chart", "artifact", "approval_request", "question", "budget", "error")


class EventBus:
    def __init__(self, project: Project) -> None:
        self.project = project
        self._subs: list[tuple[asyncio.Queue, asyncio.AbstractEventLoop | None]] = []
        self._lock = threading.Lock()

    def emit(self, agent: str, kind: str, data: dict) -> dict:
        if kind not in KINDS:
            raise ValueError(f"unknown event kind: {kind}")
        event = {"ts": time.time(), "agent": agent, "kind": kind, "data": data}
        p = self.project.path(f"agents/{agent}/events.jsonl", write=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")
        try:
            here = asyncio.get_running_loop()
        except RuntimeError:
            here = None
        for q, loop in list(self._subs):
            if loop is None or loop is here:
                q.put_nowait(event)
            else:
                loop.call_soon_threadsafe(q.put_nowait, event)
        return event

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        self._subs.append((q, loop))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs = [s for s in self._subs if s[0] is not q]

    def replay(self, since: float | None = None) -> list[dict]:
        out = []
        for f in self.project.path("agents").glob("*/events.jsonl"):
            for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if isinstance(e, dict) and isinstance(e.get("ts"), (int, float)) and (since is None or e["ts"] > since):
                    out.append(e)
        return sorted(out, key=lambda e: e["ts"])


def normalize(ev: Event) -> list[tuple[str, str, dict]]:
    out: list[tuple[str, str, dict]] = []
    try:
        _normalize(ev, out)
    except Exception as e:  # malformed model/tool output must never kill the run loop
        out.append((ev.author, "error", {"code": "bad_event", "message": f"{type(e).__name__}: {e}"}))
    return out


def _normalize(ev: Event, out: list) -> None:
    agent = ev.author
    for part in (ev.content.parts or []) if ev.content else []:
        if part.text and not part.thought:
            out.append((agent, "message_delta", {"text": part.text, "partial": bool(ev.partial)}))
        elif fc := part.function_call:
            args = fc.args or {}
            if fc.name == "adk_request_confirmation":
                orig, conf = args["originalFunctionCall"], args["toolConfirmation"]
                if not isinstance(orig, dict) or not isinstance(conf, dict) or not isinstance(orig["name"], str):
                    raise ValueError("malformed adk_request_confirmation")
                out.append((agent, "approval_request", {"call_id": fc.id, "tool": orig["name"],
                            "args": orig.get("args"), "hint": conf.get("hint")}))
            else:
                out.append((agent, "tool_call", {"id": fc.id, "name": fc.name, "args": args}))
        elif fr := part.function_response:
            out.append((agent, "tool_result", {"id": fr.id, "name": fr.name, "response": fr.response}))
    delta = (ev.actions.state_delta if ev.actions else None) or {}
    if "bml_event" in delta:
        if not isinstance(delta["bml_event"], dict) or "kind" not in delta["bml_event"]:
            raise ValueError("bml_event must be a dict with a kind")
        b = delta["bml_event"]
        if b["kind"] not in KINDS:
            raise ValueError(f"bml_event kind {b['kind']!r} unknown")
        out.append((agent, b["kind"], b.get("data", {})))
    if ev.error_code:
        out.append((agent, "error", {"code": ev.error_code, "message": ev.error_message}))
