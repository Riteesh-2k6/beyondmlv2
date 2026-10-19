"""Event bus: every agent's activity as {ts, agent, kind, data}, on disk and on subscriber queues."""
from __future__ import annotations

import asyncio
import json
import time

from google.adk.events import Event

from beyondml.workspace.project import Project

KINDS = ("message_delta", "tool_call", "tool_result", "plan_updated", "step_status", "generation", "trial",
         "chart", "artifact", "approval_request", "question", "budget", "error")


class EventBus:
    def __init__(self, project: Project) -> None:
        self.project = project
        self._subs: list[asyncio.Queue] = []

    def emit(self, agent: str, kind: str, data: dict) -> dict:
        if kind not in KINDS:
            raise ValueError(f"unknown event kind: {kind}")
        event = {"ts": time.time(), "agent": agent, "kind": kind, "data": data}
        p = self.project.path(f"agents/{agent}/events.jsonl", write=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")
        for q in list(self._subs):
            q.put_nowait(event)
        return event

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subs.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subs:
            self._subs.remove(q)

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
    agent, out = ev.author, []
    for part in (ev.content.parts or []) if ev.content else []:
        if part.text and not part.thought:
            out.append((agent, "message_delta", {"text": part.text, "partial": bool(ev.partial)}))
        elif fc := part.function_call:
            args = fc.args or {}
            if fc.name == "adk_request_confirmation":
                orig = args["originalFunctionCall"]
                out.append((agent, "approval_request", {"call_id": fc.id, "tool": orig["name"],
                            "args": orig.get("args"), "hint": args["toolConfirmation"].get("hint")}))
            else:
                out.append((agent, "tool_call", {"id": fc.id, "name": fc.name, "args": args}))
        elif fr := part.function_response:
            out.append((agent, "tool_result", {"id": fr.id, "name": fr.name, "response": fr.response}))
    delta = (ev.actions.state_delta if ev.actions else None) or {}
    if isinstance(delta.get("bml_event"), dict):
        b = delta["bml_event"]
        out.append((agent, b["kind"], b.get("data", {})))
    if ev.error_code:
        out.append((agent, "error", {"code": ev.error_code, "message": ev.error_message}))
    return out
