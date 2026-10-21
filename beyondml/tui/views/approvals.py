"""Approvals modal: one pending prompt at a time, answered with y / n / typed text."""
from __future__ import annotations

import json

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from beyondml.tui.theme import DIM, MUTED

TITLES = {"plan": "approve the plan", "tool": "allow a tool call", "budget": "budget reached", "question": "question"}


def describe(prompt: dict) -> Text:
    kind = prompt.get("kind") if prompt["event"] == "approval_request" else "question"
    if kind == "plan":
        steps = prompt.get("plan", {}).get("steps", [])
        lines = [Text.assemble((f"{s['id']}", "bold"), (f"  {s['skill']}", MUTED), f"  {s['goal']}") for s in steps]
        return Text("\n").join(lines) if lines else Text("(empty plan)", style=DIM)
    if kind == "tool":
        args = json.dumps(prompt.get("args", {}), default=str, indent=1)
        return Text.assemble((prompt.get("tool", "?"), "bold"), "\n", (args[:1200], MUTED),
                             ("\n" + prompt["hint"] if prompt.get("hint") else "", DIM))
    if kind == "budget":
        return Text(str(prompt.get("reason", "")))
    return Text(str(prompt.get("question", "")))


class ApprovalModal(ModalScreen):
    BINDINGS = [("y", "answer(True)", "Yes"), ("n", "answer(False)", "No"), ("escape", "later", "Later")]

    def __init__(self, prompt: dict) -> None:
        super().__init__()
        self.prompt = prompt

    @property
    def is_question(self) -> bool:
        return self.prompt["event"] == "question"

    def compose(self) -> ComposeResult:
        kind = "question" if self.is_question else self.prompt.get("kind", "tool")
        with Vertical(id="dialog"):
            yield Static(Text.assemble((self.prompt.get("agent", "?"), "bold #F97316"), ("  ·  ", DIM),
                                       (TITLES.get(kind, kind), "bold")), id="prompt-title")
            yield Static(describe(self.prompt), id="prompt-body")
            if self.is_question:
                yield Input(placeholder="your answer, then Enter", id="prompt-answer")
                yield Static(Text("esc  answer later", style=DIM))
            else:
                yield Static(Text.assemble(("y", "bold #22C55E"), ("  approve    ", MUTED), ("n", "bold #EF4444"),
                                           ("  deny    ", MUTED), ("esc", "bold"), ("  later", MUTED)))

    def on_mount(self) -> None:
        if self.is_question:
            self.query_one(Input).focus()

    def action_answer(self, approved: bool) -> None:
        if not self.is_question:
            self.dismiss(approved)

    def action_later(self) -> None:
        self.dismiss(None)

    def on_input_submitted(self, message: Input.Submitted) -> None:
        if message.value.strip():
            self.dismiss(message.value.strip())
