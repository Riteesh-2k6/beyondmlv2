"""Chat dock: the lead agent's transcript and an input, present on every screen."""
from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Input, RichLog, Static

from beyondml.tui.theme import MUTED

SPEAKER = {"user": ("you", "#22D3EE"), "lead": ("lead", "#F97316")}


class ChatDock(Vertical):
    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._lines: list[str] = []

    def compose(self) -> ComposeResult:
        yield Static(Text("lead agent", style=MUTED), id="chat-title")
        yield RichLog(id="chat-log", wrap=True, markup=False, highlight=False)
        yield Input(placeholder="message the lead agent  (c to focus, enter to send)", id="chat-input")

    def show_history(self, messages: list[dict]) -> None:
        """Called once by the app after it replays the event log (children mount before that replay)."""
        for m in messages:
            self._write(m["agent"], m["text"])

    def _write(self, agent: str, text: str) -> None:
        who, colour = SPEAKER.get(agent, (agent, MUTED))
        self._lines.append(f"{who} › {text}")
        self.query_one(RichLog).write(Text.assemble((f"{who} › ", f"bold {colour}"), text))

    def transcript(self) -> str:
        return "\n".join(self._lines)

    def set_title(self, pending: int, busy: bool) -> None:
        bits = [("lead agent", MUTED)]
        if busy:
            bits.append(("  thinking…", "#F97316"))
        if pending:
            bits.append((f"  {pending} waiting for you (a)", "#EAB308"))
        self.query_one("#chat-title", Static).update(Text.assemble(*bits))

    def on_bus_event(self, ev: dict) -> None:
        d = ev["data"]
        if ev["kind"] == "message_delta" and ev["agent"] in SPEAKER and not d.get("partial"):
            self._write(ev["agent"], d.get("text", ""))

    async def on_input_submitted(self, message: Input.Submitted) -> None:
        if message.input.id != "chat-input":
            return
        text = message.value.strip()
        message.input.value = ""
        if text:
            await self.app.send_chat(text)
