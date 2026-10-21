"""The BeyondML terminal app: six screens on one live session."""
from __future__ import annotations

import asyncio
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import ContentSwitcher, Footer, Input, Static

from beyondml.gateway.session import Session
from beyondml.gateway.state import SessionState
from beyondml.tui.theme import DIM, MUTED, OBSIDIAN
from beyondml.tui.views.agent import AgentView
from beyondml.tui.views.approvals import ApprovalModal
from beyondml.tui.views.chat import ChatDock
from beyondml.tui.views.experiments import ExperimentsView
from beyondml.tui.views.files import FilesView
from beyondml.tui.views.plan_board import PlanBoard

SCREENS = ("plan", "agent", "experiments", "files")


class BeyondMLApp(App):
    TITLE = "BeyondML"
    CSS = """
    Screen { background: $background; }
    #topbar { height: 1; background: $surface; padding: 0 1; }
    #main { height: 1fr; }
    PlanBoard { padding: 0 1; }
    #run-form { height: auto; padding: 1 0; }
    #run-form Input { margin: 0 0 1 0; }
    #board { height: 1fr; }
    #cards { width: 2fr; }
    DepGraph { width: 1fr; min-width: 28; border-left: solid $panel; padding: 0 1; }
    StepCard { height: auto; border: solid $panel; padding: 0 1; margin: 0 1 1 0; }
    StepCard:focus { border: solid $primary; }
    StepCard.running { border: solid $primary 60%; }
    StepCard.done { border: solid $success 50%; }
    StepCard.failed { border: solid $error; }
    ChatDock { height: 10; border-top: solid $panel; }
    #chat-title { height: 1; padding: 0 1; background: $surface; }
    #chat-log { height: 1fr; padding: 0 1; }
    Input { border: none; height: 1; background: $surface; padding: 0 1; }
    Input:focus { border: none; background: $panel; }
    ApprovalModal { align: center middle; }
    #dialog { width: 90; max-width: 100%; height: auto; max-height: 80%; border: solid $primary; background: $surface;
              padding: 1 2; }
    #prompt-title { height: 1; margin-bottom: 1; }
    #prompt-body { height: auto; max-height: 30; margin-bottom: 1; }
    """
    AUTO_FOCUS = ""  # an auto-focused Input would swallow the navigation keys
    BINDINGS = [
        ("1", "show('plan')", "Plan"), ("2", "show('agent')", "Agent"), ("3", "show('experiments')", "Experiments"),
        ("4", "show('files')", "Files"), ("a", "approvals", "Approvals"), ("c", "focus_chat", "Chat"),
        ("escape", "blur", "Leave input", ), ("q", "quit", "Quit"),
    ]

    def __init__(self, session: Session, *, gateway_url: str | None = None) -> None:
        super().__init__()
        self.session = session
        self.gateway_url = gateway_url
        self.state = SessionState()
        self.busy = False
        self._modal: ApprovalModal | None = None

    def compose(self) -> ComposeResult:
        yield Static(id="topbar")
        with ContentSwitcher(initial="plan", id="main"):
            yield PlanBoard(id="plan")
            yield AgentView(id="agent")
            yield ExperimentsView(id="experiments")
            yield FilesView(id="files")
        yield ChatDock(id="chat")
        yield Footer()

    # ── wiring ───────────────────────────────────────────────
    @property
    def views(self):
        return [self.query_one(PlanBoard), self.agent_view, self.query_one(ExperimentsView),
                self.query_one(FilesView), self.chat]

    @property
    def agent_view(self) -> AgentView:
        return self.query_one(AgentView)

    @property
    def chat(self) -> ChatDock:
        return self.query_one(ChatDock)

    def on_mount(self) -> None:
        self.register_theme(OBSIDIAN)
        self.theme = "obsidian"
        queue = self.session.bus.subscribe()  # subscribed from this loop, before the replay, so nothing is missed
        last = 0.0
        for ev in self.session.bus.replay():
            self.state.apply(ev)
            last = ev["ts"]
        self.query_one(PlanBoard).reload()
        self.run_worker(self._tail(queue, last), exclusive=True, name="bus")
        self.set_interval(1.0, self._tick)
        self._update_topbar()
        self._maybe_prompt()

    async def _tail(self, queue: asyncio.Queue, replayed_to: float) -> None:
        while True:
            ev = await queue.get()
            if ev["ts"] <= replayed_to:
                continue
            self._apply(ev)

    def _apply(self, ev: dict) -> None:
        self.state.apply(ev)
        for view in self.views:
            view.on_bus_event(ev)
        if ev["kind"] in ("question", "approval_request") and "id" in ev["data"]:
            if "answer" in ev["data"] and self._modal and self._modal.prompt["id"] == ev["data"]["id"]:
                self._modal.dismiss(None)  # answered elsewhere (the web, say)
            self._maybe_prompt()
        if ev["kind"] in ("step_status", "error", "plan_updated"):
            self._update_topbar()

    def _tick(self) -> None:
        self.query_one(PlanBoard).tick()
        self._update_topbar()

    def _update_topbar(self) -> None:
        s = self.session.summary()
        bits = [("BeyondML", "bold #F97316"), ("  ", ""), (s["name"], "bold"), ("  ", "")]
        if s["goal"]:
            bits.append((s["goal"][:60], MUTED))
        bits += [("   ", ""), (s["mode"], DIM), ("  ", ""), (s["model"], DIM), ("   ", "")]
        bits.append(("● RUNNING" if s["running"] else "○ idle", "bold #F97316" if s["running"] else DIM))
        if self.gateway_url:
            bits += [("   ", ""), (self.gateway_url, DIM)]
        self.query_one("#topbar", Static).update(Text.assemble(*bits))
        self.chat.set_title(len(self.state.pending), self.busy)

    # ── prompts ──────────────────────────────────────────────
    def _maybe_prompt(self) -> None:
        if self._modal is not None or not self.state.pending:
            return
        prompt = next(iter(self.state.pending.values()))
        self._modal = ApprovalModal(prompt)

        def done(result) -> None:
            self._modal = None
            if result is not None:
                self.run_worker(self.answer(prompt["id"], result), name="answer")
            elif prompt["id"] in self.state.pending:
                self.notify("press a to answer later", severity="information", timeout=3)
            self.call_later(self._maybe_prompt) if result is not None else None

        self.push_screen(self._modal, done)

    async def answer(self, pid: str, value) -> None:
        try:
            await self.session.call(self.session.answer(pid, value))
        except KeyError:
            self.notify("that prompt was already answered", severity="warning")

    def action_approvals(self) -> None:
        if self._modal is None and self.state.pending:
            self._maybe_prompt()
        elif self._modal is None:
            self.notify("nothing is waiting for you", timeout=2)

    # ── commands ─────────────────────────────────────────────
    async def send_chat(self, text: str) -> None:
        self.busy = True
        self._update_topbar()
        try:
            await self.session.call(self.session.chat(text))
        except Exception as e:  # the lead failing must not take the app down
            self.notify(f"lead error: {e}", severity="error")
        finally:
            self.busy = False
            self._update_topbar()

    async def start_run(self, goal: str | None, data: list[str]) -> None:
        try:
            await self.session.call(self.session.start(goal, [Path(d) for d in data]))
        except RuntimeError as e:
            self.notify(str(e), severity="warning")
        self._update_topbar()

    def open_agent(self, agent_id: str) -> None:
        self.agent_view.show_agent(agent_id)
        self.action_show("agent")

    def action_show(self, name: str) -> None:
        if name in SCREENS:
            self.query_one(ContentSwitcher).current = name

    def action_focus_chat(self) -> None:
        self.query_one("#chat-input", Input).focus()

    def action_blur(self) -> None:
        self.set_focus(None)
