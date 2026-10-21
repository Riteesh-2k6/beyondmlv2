"""Plan board: one card per step, and the dependency graph as topological layers."""
from __future__ import annotations

import json

from rich.console import Group
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Input, Static

from beyondml.gateway.state import SessionState, plan_view
from beyondml.tui.theme import DIM, MUTED, STATUS_COLOR, STATUS_GLYPH, elapsed, fmt_num, status_text


class StepCard(Static):
    can_focus = True
    BINDINGS = [("enter", "open", "Open agent")]

    class Open(Message):
        def __init__(self, step_id: str) -> None:
            super().__init__()
            self.step_id = step_id

    def __init__(self, step: dict, default_model: str) -> None:
        super().__init__(id=f"card-{step['id']}")
        self.step, self.default_model = step, default_model

    @property
    def step_id(self) -> str:
        return self.step["id"]

    def update_step(self, step: dict) -> None:
        self.step = step
        self.set_classes(step["status"])
        self.refresh()

    def render(self) -> Text:
        s = self.step
        head = Text.assemble(status_text(s["status"]), "  ", (s["id"], "bold"), "  ", (s["skill"], MUTED))
        meta = Text.assemble((s.get("model") or self.default_model, MUTED), "  ",
                             (elapsed(s.get("started"), s.get("finished")), "bold" if s["status"] == "running" else MUTED))
        lines = [head, meta]
        if s.get("metric"):
            lines.append(Text.assemble((s["metric"]["name"], MUTED), " ", (fmt_num(s["metric"]["value"]), "bold")))
        if s["status"] == "failed" and s.get("last_failure"):
            lines.append(Text(s["last_failure"][:160], style=STATUS_COLOR["failed"]))
        elif s.get("summary"):
            lines.append(Text(s["summary"][:160], style=DIM))
        return Text("\n").join(lines)

    def action_open(self) -> None:
        self.post_message(self.Open(self.step_id))

    def on_click(self) -> None:
        self.focus()
        self.post_message(self.Open(self.step_id))


def layers(steps: list[dict]) -> list[list[dict]]:
    depth: dict[str, int] = {}
    by_id = {s["id"]: s for s in steps}

    def visit(sid: str, seen: frozenset) -> int:
        if sid in depth:
            return depth[sid]
        deps = [d for d in by_id[sid].get("depends_on", []) if d in by_id and d not in seen]
        depth[sid] = 1 + max((visit(d, seen | {sid}) for d in deps), default=-1)
        return depth[sid]

    for s in steps:
        visit(s["id"], frozenset())
    out: list[list[dict]] = [[] for _ in range(max(depth.values(), default=-1) + 1)]
    for s in steps:
        out[depth[s["id"]]].append(s)
    return out


class DepGraph(Static):
    def __init__(self) -> None:
        super().__init__(id="graph")
        self.steps: list[dict] = []

    def update_steps(self, steps: list[dict]) -> None:
        self.steps = steps
        self.refresh()

    def render(self) -> Text:
        if not self.steps:
            return Text("no plan", style=DIM)
        rows = [Text("dependency graph", style=MUTED)]
        for depth, layer in enumerate(layers(self.steps)):
            if depth:
                rows.append(Text("   │", style=DIM))
            for s in layer:
                line = Text.assemble((f"{STATUS_GLYPH.get(s['status'], '?')} ", STATUS_COLOR.get(s["status"], MUTED)),
                                     (s["id"], "bold"))
                if s.get("depends_on"):
                    line.append(f"  ← {', '.join(s['depends_on'])}", style=DIM)
                rows.append(line)
        return Text("\n").join(rows)


class RunForm(Vertical):
    """Shown when there is no plan: a goal and an optional data path start a run."""

    def compose(self) -> ComposeResult:
        yield Static(Text("No plan yet. Describe the goal and press Enter; data is optional.", style=MUTED))
        yield Input(placeholder="goal, e.g. predict churn from usage and billing", id="goal")
        yield Input(placeholder="data file or folder (optional)", id="data")


class PlanBoard(Vertical):
    def compose(self) -> ComposeResult:
        yield RunForm(id="run-form")
        with Horizontal(id="board"):
            yield VerticalScroll(id="cards")
            yield DepGraph()

    def on_mount(self) -> None:
        self.reload()

    @property
    def project(self):
        return self.app.session.project

    def reload(self) -> None:
        view = plan_view(self.project, self.app.state)
        form = self.query_one("#run-form")
        if view is None and not form.display:
            form.display = True
            self.query_one("#goal", Input).focus()
        form.display = view is None
        steps = view["steps"] if view else []
        self.query_one(DepGraph).update_steps(steps)
        cards = self.query_one("#cards")
        existing = {c.step_id: c for c in cards.query(StepCard)}
        if set(existing) != {s["id"] for s in steps}:
            cards.remove_children()
            for s in steps:
                cards.mount(StepCard(s, view["default_model"]))
            return
        for s in steps:
            existing[s["id"]].update_step(s)

    def on_bus_event(self, ev: dict) -> None:
        if ev["kind"] in ("step_status", "plan_updated") or (
                ev["kind"] == "tool_result" and ev["data"].get("name") == "run_python"):
            self.reload()

    def tick(self) -> None:
        for card in self.query(StepCard):
            if card.step["status"] == "running":
                card.refresh()

    def on_step_card_open(self, message: StepCard.Open) -> None:
        self.app.open_agent(message.step_id)

    async def on_input_submitted(self, message: Input.Submitted) -> None:
        if message.input.id not in ("goal", "data"):
            return
        goal = self.query_one("#goal", Input).value.strip()
        data = self.query_one("#data", Input).value.strip()
        if not goal:
            self.query_one("#goal", Input).focus()
            return
        await self.app.start_run(goal, [data] if data else [])
