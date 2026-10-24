"""Approvals modal: one pending prompt at a time, answered with y / n, a numbered choice, or typed text."""
from __future__ import annotations

import json

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from beyondml.tui.theme import DIM, MUTED

TITLES = {"plan": "approve the plan", "tool": "allow a tool call", "budget": "budget reached", "question": "question"}


def options_of(prompt: dict) -> list[str]:
    if prompt["event"] != "question":
        return []
    raw = prompt.get("options") or []
    return [o for o in raw if isinstance(o, str) and o.strip()][:9]


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
    lines = [Text(str(prompt.get("question", "")))]
    options = options_of(prompt)
    if options:
        lines.append(Text(""))
        for n, option in enumerate(options, start=1):
            rec = option == prompt.get("recommended")
            lines.append(Text.assemble((f"{n}  ", "bold #F97316" if rec else "bold"), option,
                                       ("   recommended" if rec else "", "#F97316")))
    return Text("\n").join(lines)


class ApprovalModal(ModalScreen):
    BINDINGS = [("y", "answer(True)", "Yes"), ("n", "answer(False)", "No"), ("escape", "later", "Later")] + [
        Binding(str(n), f"pick({n})", "", show=False) for n in range(1, 10)]

    def __init__(self, prompt: dict) -> None:
        super().__init__()
        self.prompt = prompt
        self.options = options_of(prompt)

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
                yield Input(placeholder="or type your own answer, then Enter" if self.options else "your answer, then Enter",
                            id="prompt-answer")
                hint = ("1-9  pick a choice    tab  type your own    " if self.options else "") + "esc  answer later"
                yield Static(Text(hint, style=DIM))
            else:
                yield Static(Text.assemble(("y", "bold #22C55E"), ("  approve    ", MUTED), ("n", "bold #EF4444"),
                                           ("  deny    ", MUTED), ("esc", "bold"), ("  later", MUTED)))

    def on_mount(self) -> None:
        if self.is_question and not self.options:  # with choices the digit keys must reach the screen, not the input
            self.query_one(Input).focus()

    def action_answer(self, approved: bool) -> None:
        if not self.is_question:
            self.dismiss(approved)

    def action_pick(self, n: int) -> None:
        if 1 <= n <= len(self.options):
            self.dismiss(self.options[n - 1])

    def action_later(self) -> None:
        self.dismiss(None)

    def on_input_submitted(self, message: Input.Submitted) -> None:
        if message.value.strip():
            self.dismiss(message.value.strip())
