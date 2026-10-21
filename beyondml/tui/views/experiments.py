"""Experiments: the ledger (experiments.jsonl) as a table, newest first."""
from __future__ import annotations

import time

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Static

from beyondml.tui.theme import DIM, MUTED, fmt_num

COLUMNS = ("time", "agent", "script", "exit", "seconds", "metrics")


def row_for(record: dict) -> list[str]:
    when = time.strftime("%H:%M:%S", time.localtime(record.get("ts", 0)))
    script = " ".join([str(record.get("script", ""))] + [str(a) for a in record.get("args", [])])
    exit_code = "timeout" if record.get("timed_out") else str(record.get("exit_code", ""))
    metrics = record.get("metrics") or {}
    shown = ", ".join(f"{k}={fmt_num(v)}" for k, v in metrics.items() if not isinstance(v, (dict, list)))
    return [when, str(record.get("agent", "")), script, exit_code, f"{record.get('seconds', 0):.1f}", shown]


class ExperimentsView(Vertical):
    def compose(self) -> ComposeResult:
        yield Static(id="exp-title")
        yield DataTable(id="ledger", cursor_type="row", zebra_stripes=True)

    def on_mount(self) -> None:
        self.query_one(DataTable).add_columns(*COLUMNS)
        self.rows: list[list[str]] = []
        self.reload()

    def reload(self) -> None:
        records = self.app.session.project.ledger()
        self.rows = [row_for(r) for r in reversed(records)]
        table = self.query_one(DataTable)
        table.clear()
        for row in self.rows:
            table.add_row(*(Text(c, style=MUTED if i in (0, 2) else "") for i, c in enumerate(row)))
        title = Text.assemble(("experiments", "bold"), (f"  {len(records)} runs", MUTED),
                              ("  from experiments.jsonl" if records else "  none yet: every run_python lands here", DIM))
        self.query_one("#exp-title", Static).update(title)

    def on_bus_event(self, ev: dict) -> None:
        if (ev["kind"] == "tool_result" and ev["data"].get("name") == "run_python") or ev["kind"] == "step_status":
            self.reload()
