"""Pure fold of the event stream into what the screens show. Shared by the TUI and the gateway."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable

from beyondml.workspace.plan import load_plan
from beyondml.workspace.project import Project

CHAT_AGENTS = ("user", "lead")


@dataclass
class SessionState:
    steps: dict[str, dict] = field(default_factory=dict)        # id -> {status, started, finished, reason}
    pending: dict[str, dict] = field(default_factory=dict)      # prompt id -> {event, agent, **data}
    chat: list[dict] = field(default_factory=list)              # [{ts, agent, text}]
    generations: dict[str, list] = field(default_factory=dict)  # agent -> [generation data]
    trials: dict[str, list] = field(default_factory=dict)
    charts: dict[str, list] = field(default_factory=dict)
    artifacts: dict[str, list] = field(default_factory=dict)    # agent -> [paths]
    errors: list[dict] = field(default_factory=list)
    last_ts: float = 0.0

    @classmethod
    def fold(cls, events: Iterable[dict]) -> "SessionState":
        state = cls()
        for ev in events:
            state.apply(ev)
        return state

    def apply(self, ev: dict) -> None:
        agent, kind, data, ts = ev["agent"], ev["kind"], ev.get("data") or {}, ev.get("ts", 0.0)
        self.last_ts = max(self.last_ts, ts)
        if kind == "step_status" and isinstance(data.get("id"), str):
            step = self.steps.setdefault(data["id"], {"status": "pending", "started": None, "finished": None,
                                                       "reason": ""})
            step["status"], step["reason"] = data.get("status", "pending"), data.get("reason", "")
            if step["status"] == "running":
                step["started"], step["finished"] = ts, None
            elif step["status"] in ("done", "failed", "skipped"):
                step["finished"] = ts
        elif kind in ("question", "approval_request") and "id" in data:
            if "answer" in data:
                self.pending.pop(data["id"], None)
            else:
                self.pending[data["id"]] = {"event": kind, "agent": agent, **data}
        elif kind == "message_delta" and agent in CHAT_AGENTS and not data.get("partial"):
            self.chat.append({"ts": ts, "agent": agent, "text": data.get("text", "")})
        elif kind == "generation":
            self.generations.setdefault(agent, []).append(data)
        elif kind == "trial":
            self.trials.setdefault(agent, []).append(data)
        elif kind == "chart":
            self.charts.setdefault(agent, []).append(data)
        elif kind == "artifact" and isinstance(data.get("path"), str):
            paths = self.artifacts.setdefault(agent, [])
            if data["path"] not in paths:
                paths.append(data["path"])
        elif kind == "error":
            self.errors.append({"ts": ts, "agent": agent, **data})

    def to_dict(self) -> dict:
        return asdict(self)


def latest_metrics(project: Project) -> dict[str, dict]:
    """agent -> metrics of its most recent ledger record that reported any."""
    out: dict[str, dict] = {}
    for record in project.ledger():
        if isinstance(record.get("metrics"), dict) and record["metrics"]:
            out[record.get("agent", "")] = record["metrics"]
    return out


def key_metric(metrics: dict, preferred: str) -> dict | None:
    for name in ([preferred] if preferred in metrics else []) + list(metrics):
        if isinstance(metrics[name], (int, float)) and not isinstance(metrics[name], bool):
            return {"name": name, "value": metrics[name]}
    return None


def plan_view(project: Project, state: SessionState) -> dict | None:
    try:
        plan = load_plan(project)
    except ValueError:
        plan = None
    if plan is None:
        return None
    metrics = latest_metrics(project)
    view = plan.to_dict()
    for step in view["steps"]:
        folded = state.steps.get(step["id"], {})
        step["started"], step["finished"] = folded.get("started"), folded.get("finished")
        step["metric"] = key_metric(metrics.get(step["id"], {}), plan.metric)
    view["default_model"] = project.settings()["model"]["default"]
    return view
