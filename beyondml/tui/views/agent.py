"""Agent screen: exactly how the agent was instructed, what it is doing, and what it produced."""
from __future__ import annotations

import json

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Markdown, RichLog, Static, TabbedContent, TabPane

from beyondml.gateway.state import agent_view, latest_metrics
from beyondml.tui.plots import MEAN, PALETTE, bar_chart, chart_event, line_chart
from beyondml.tui.theme import DIM, MUTED, STATUS_COLOR, fmt_num, status_text

LIVE_KINDS = ("message_delta", "tool_call", "tool_result", "step_status", "error", "question", "approval_request",
              "generation", "trial", "budget")


def live_line(ev: dict) -> Text | None:
    kind, d = ev["kind"], ev["data"]
    if kind == "message_delta":
        return None if d.get("partial") else Text.assemble(("▌ ", DIM), d.get("text", ""))
    if kind == "tool_call":
        args = json.dumps(d.get("args", {}), default=str)
        return Text.assemble(("→ ", "#F97316"), (str(d.get("name")), "bold #F97316"), (f"({args[:160]})", MUTED))
    if kind == "tool_result":
        resp = d.get("response")
        failed = isinstance(resp, dict) and (resp.get("ok") is False or "error" in resp)
        if failed:
            detail = str(resp.get("error") or resp.get("errors"))[:200]
            return Text.assemble(("← ", "#EF4444"), (str(d.get("name")), "#EF4444"), (f"  {detail}", MUTED))
        brief = json.dumps(resp, default=str)[:160] if resp is not None else ""
        return Text.assemble(("← ", "#22C55E"), (str(d.get("name")), "#22C55E"), (f"  {brief}", DIM))
    if kind == "step_status":
        return Text.assemble(("■ ", STATUS_COLOR.get(d.get("status"), MUTED)), status_text(d.get("status", "?")),
                             (f"  {d.get('reason')}" if d.get("reason") else "", MUTED))
    if kind == "question":
        if "id" not in d:
            return None
        if d.get("cancelled"):
            return Text.assemble(("? ", "#F97316"), str(d.get("question", "")), ("  → cancelled", MUTED))
        return Text.assemble(("? ", "#F97316"), str(d.get("question", "")),
                             (f"  → {d['answer']}" if "answer" in d else "", "#22C55E"))
    if kind == "approval_request":
        if "id" not in d:
            return None
        what = d.get("tool") or d.get("kind")
        if d.get("cancelled"):
            return Text.assemble(("! ", "#F97316"), f"approval: {what}", ("  → cancelled", MUTED))
        verdict = "" if "answer" not in d else ("  → approved" if d["answer"] else "  → denied")
        return Text.assemble(("! ", "#F97316"), f"approval: {what}", (verdict, "#22C55E" if d.get("answer") else "#EF4444"))
    if kind == "generation":
        return Text(f"gen {d.get('gen')}  best {fmt_num(d.get('best_fitness'))}  mean {fmt_num(d.get('mean_fitness'))}"
                    f"  diversity {fmt_num(d.get('diversity'))}", style=DIM)
    if kind == "trial":
        return Text(f"trial {d.get('number')}  {str(d.get('state', '')).lower()}  value {fmt_num(d.get('value'))}", style=DIM)
    if kind == "budget":
        return Text.assemble(("$ ", "#EAB308"), f"budget: {d.get('reason')}")
    if kind == "error":
        return Text.assemble(("✕ ", "#EF4444"), (f"{d.get('code')}: {d.get('message')}", "#EF4444"))
    return None


class AgentView(Vertical):
    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.agent_id: str | None = None
        self.info: dict | None = None
        self.live_lines: list[str] = []
        self.outputs_text = ""
        self.evolve_text = ""

    def compose(self) -> ComposeResult:
        yield Static(Text("pick a step on the plan board (1), or press 2 after a run starts", style=DIM), id="agent-head")
        with TabbedContent(initial="instruction", id="agent-tabs"):
            with TabPane("Instruction", id="instruction"):
                with VerticalScroll():
                    yield Static(id="instr-meta")
                    yield Markdown("", id="instr-md")
            with TabPane("Live", id="live"):
                yield RichLog(id="live-log", wrap=True, markup=False, highlight=False)
            with TabPane("Outputs", id="outputs"):
                with VerticalScroll():
                    yield Static(id="outputs-body")
            with TabPane("Evolve", id="evolve"):
                with VerticalScroll():
                    yield Static(id="evolve-body")

    def on_mount(self) -> None:
        self.query_one(TabbedContent).hide_tab("evolve")

    @property
    def _width(self) -> int:
        return max(self.size.width - 6, 40)

    # ── loading ──────────────────────────────────────────────
    def show_agent(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.info = agent_view(self.app.session.project, self.app.state, agent_id)
        self._render_head()
        self._render_instruction()
        log = self.query_one(RichLog)
        log.clear()
        self.live_lines = []
        for ev in self.app.session.bus.replay():
            if ev["agent"] == agent_id:
                self._append_live(ev)
        self._render_outputs()
        self._render_evolve()

    def _render_head(self) -> None:
        if not self.info:
            self.query_one("#agent-head", Static).update(Text(f"{self.agent_id}: unknown agent", style=DIM))
            return
        i = self.info
        bits = [(i["id"], "bold"), ("  ", ""), status_text(i["status"]), ("  ", "")]
        if i["skill"]:
            bits += [(i["skill"], MUTED), ("  ", "")]
        bits += [(i["model"], MUTED)]
        if i["attempts"]:
            bits += [("  ", ""), (f"attempt {i['attempts']}", DIM)]
        self.query_one("#agent-head", Static).update(Text.assemble(*bits))

    def _render_instruction(self) -> None:
        i = self.info or {}
        meta = [Text.assemble(("skill   ", DIM), str(i.get("skill") or "—")),
                Text.assemble(("inputs  ", DIM), ", ".join(i.get("inputs") or []) or "—"),
                Text.assemble(("outputs ", DIM), ", ".join(i.get("outputs") or []) or "—"),
                Text.assemble(("tools   ", DIM), ", ".join(i.get("tools") or []) or "—"), Text("")]
        self.query_one("#instr-meta", Static).update(Text("\n").join(meta))
        self.query_one("#instr-md", Markdown).update(i.get("instruction") or "_no instruction written yet_")

    def _append_live(self, ev: dict) -> None:
        line = live_line(ev)
        if line is not None:
            self.live_lines.append(line.plain)
            self.query_one(RichLog).write(line)

    def _render_outputs(self) -> None:
        if not self.agent_id:
            return
        state, i = self.app.state, self.info or {}
        files = list(dict.fromkeys((i.get("outputs") or []) + state.artifacts.get(self.agent_id, [])))
        metrics = latest_metrics(self.app.session.project).get(self.agent_id, {})
        charts = state.charts.get(self.agent_id, [])
        parts = [Text("files", style=MUTED)]
        parts += [Text(f"  {f}") for f in files] or [Text("  none yet", style=DIM)]
        parts += [Text(""), Text("metrics", style=MUTED)]
        parts += [Text.assemble(f"  {k}  ", (fmt_num(v), "bold")) for k, v in metrics.items()] or [Text("  none yet", style=DIM)]
        parts += [Text(""), Text(f"charts ({len(charts)})", style=MUTED)]
        for c in charts:
            parts += [Text(""), chart_event(c, width=self._width, project=self.app.session.project)]
        body = Text("\n").join(parts)
        self.outputs_text = body.plain
        self.query_one("#outputs-body", Static).update(body)

    def _render_evolve(self) -> None:
        if not self.agent_id:
            return
        state = self.app.state
        gens, trials = state.generations.get(self.agent_id, []), state.trials.get(self.agent_id, [])
        tabs = self.query_one(TabbedContent)
        if not gens and not trials:
            tabs.hide_tab("evolve")
            self.evolve_text = ""
            return
        tabs.show_tab("evolve")
        parts: list[Text] = []
        if gens:
            last = gens[-1]
            parts.append(Text.assemble((f"generation {last.get('gen')}", "bold"), ("   evals ", DIM), str(last.get("evals")),
                                       ("   elapsed ", DIM), f"{fmt_num(last.get('seconds'))}s",
                                       ("   best ", DIM), (fmt_num(last.get("best_fitness")), "bold #F97316")))
            x = [g.get("gen", n + 1) for n, g in enumerate(gens)]
            parts.append(line_chart("fitness per generation", x, {"best": [g.get("best_fitness") for g in gens],
                                                                   "mean": [g.get("mean_fitness") for g in gens]},
                                    width=self._width, xlabel="generation",
                                    colors={"best": PALETTE[0], "mean": MEAN}))
            parts.append(bar_chart("population diversity", x, [g.get("diversity", 0) for g in gens], width=self._width))
            parts.append(Text("best genome", style=MUTED))
            best = last.get("best_params") or {}
            parts += [Text.assemble(f"  {k}  ", (fmt_num(v), "bold")) for k, v in best.items()] or [Text("  —", style=DIM)]
        if trials:
            complete = [t for t in trials if t.get("state") == "COMPLETE" and isinstance(t.get("value"), (int, float))]
            best = max(complete, key=lambda t: t["value"], default=None)
            pruned = sum(1 for t in trials if t.get("state") == "PRUNED")
            parts.append(Text.assemble((f"trials {len(trials)}", "bold"), ("   complete ", DIM), str(len(complete)),
                                       ("   pruned ", DIM), str(pruned), ("   max value ", DIM),
                                       (f"{fmt_num(best['value'])} (trial {best.get('number')})" if best else "—",
                                        "bold #F97316")))
            if complete:
                running, peak = [], float("-inf")
                for t in complete:
                    peak = max(peak, t["value"])
                    running.append(peak)
                x = [t.get("number", n + 1) for n, t in enumerate(complete)]
                parts.append(line_chart("objective per trial (complete)", x,
                                        {"value": [t["value"] for t in complete], "running max": running},
                                        width=self._width, xlabel="trial",
                                        colors={"value": MEAN, "running max": PALETTE[0]}))
            shown = best or trials[-1]
            parts.append(Text(f"parameters of trial {shown.get('number')}" + (" (max value)" if best else ""), style=MUTED))
            parts += [Text.assemble(f"  {k}  ", (fmt_num(v), "bold")) for k, v in (shown.get("params") or {}).items()] \
                or [Text("  —", style=DIM)]
        body = Text("\n").join(parts)
        self.evolve_text = body.plain
        self.query_one("#evolve-body", Static).update(body)

    # ── live updates ─────────────────────────────────────────
    def on_bus_event(self, ev: dict) -> None:
        if ev["agent"] != self.agent_id:
            return
        if ev["kind"] in LIVE_KINDS:
            self._append_live(ev)
        if ev["kind"] == "step_status":
            self.info = agent_view(self.app.session.project, self.app.state, self.agent_id)
            self._render_head()
            self._render_instruction()
        if ev["kind"] in ("artifact", "chart") or (ev["kind"] == "tool_result" and ev["data"].get("name") == "run_python"):
            self._render_outputs()
        if ev["kind"] in ("generation", "trial"):
            self._render_evolve()
