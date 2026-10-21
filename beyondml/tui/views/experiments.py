"""Experiments placeholder; Task 6 builds the ledger table."""
from __future__ import annotations

from rich.text import Text
from textual.containers import Vertical
from textual.widgets import Static

from beyondml.tui.theme import DIM


class ExperimentsView(Vertical):
    def compose(self):
        yield Static(Text("experiments", style=DIM))

    def on_bus_event(self, ev: dict) -> None:
        pass
