"""Agent screen placeholder; Task 5 fills in the Instruction / Live / Outputs / Evolve tabs."""
from __future__ import annotations

from rich.text import Text
from textual.containers import Vertical
from textual.widgets import Static

from beyondml.tui.theme import DIM


class AgentView(Vertical):
    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.agent_id: str | None = None

    def compose(self):
        yield Static(Text("pick a step on the plan board", style=DIM), id="agent-empty")

    def show_agent(self, agent_id: str) -> None:
        self.agent_id = agent_id

    def on_bus_event(self, ev: dict) -> None:
        pass
