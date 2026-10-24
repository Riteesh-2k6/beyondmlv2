"""plan.json: the single source of truth for a run (planner writes, executor schedules, update_plan edits)."""
from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass, field, fields

from beyondml.workspace.project import Project

STEP_ID = re.compile(r"^[a-z][a-z0-9_]*$")
STATUSES = ("pending", "running", "done", "failed", "skipped")
_LISTS = ("depends_on", "inputs", "outputs", "touched")


def _strs(v, what: str) -> None:
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise ValueError(f"{what} must be a list of strings")


@dataclass
class SuccessCheck:
    run: str | None = None
    expects: list[str] = field(default_factory=list)
    metric: str | None = None
    min: float | None = None


@dataclass
class Step:
    id: str
    skill: str
    goal: str
    depends_on: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    success_check: SuccessCheck = field(default_factory=SuccessCheck)
    status: str = "pending"  # pending | running | done | failed | skipped
    model: str | None = None
    attempts: int = 0
    checkpoint: str | None = None  # git sha taken when the current attempt started
    touched: list[str] = field(default_factory=list)  # project-relative paths this attempt wrote
    summary: str = ""
    last_failure: str = ""

    @classmethod
    def from_dict(cls, d) -> "Step":
        if not isinstance(d, dict):
            raise ValueError(f"step must be an object, got {type(d).__name__}")
        for k in ("id", "skill", "goal"):
            if not isinstance(d.get(k), str):
                raise ValueError(f"step {d.get('id', '?')!r}: missing or non-string '{k}'")
        for k in _LISTS:
            if not isinstance(d.get(k, []), list):
                raise ValueError(f"step {d['id']!r}: '{k}' must be a list")
        sid = d["id"]
        for k in ("depends_on", "inputs", "outputs"):
            _strs(d.get(k, []), f"step {sid!r}: '{k}'")
        if d.get("status", "pending") not in STATUSES:
            raise ValueError(f"step {sid!r}: invalid status {d['status']!r}")
        a = d.get("attempts", 0)
        if isinstance(a, bool) or not isinstance(a, int):
            raise ValueError(f"step {sid!r}: 'attempts' must be an integer")
        for k in ("model", "checkpoint"):
            if d.get(k) is not None and not isinstance(d[k], str):
                raise ValueError(f"step {sid!r}: '{k}' must be a string or null")
        for k in ("summary", "last_failure"):
            if not isinstance(d.get(k, ""), str):
                raise ValueError(f"step {sid!r}: '{k}' must be a string")
        _strs(d.get("touched", []), f"step {sid!r}: 'touched'")
        sc = d.get("success_check") or {}
        if not isinstance(sc, dict):
            raise ValueError(f"step {sid!r}: 'success_check' must be an object")
        _strs(sc.get("expects", []), f"step {sid!r}: success_check 'expects'")
        for k in ("run", "metric"):
            if sc.get(k) is not None and not isinstance(sc[k], str):
                raise ValueError(f"step {sid!r}: success_check '{k}' must be a string or null")
        m = sc.get("min")
        if m is not None and (isinstance(m, bool) or not isinstance(m, (int, float))):
            raise ValueError(f"step {sid!r}: success_check 'min' must be a number or null")
        known = {f.name for f in fields(cls)} - {"success_check"}
        check = SuccessCheck(**{k: v for k, v in sc.items() if k in {f.name for f in fields(SuccessCheck)}})
        return cls(success_check=check, **{k: v for k, v in d.items() if k in known})


@dataclass
class Plan:
    goal: str
    metric: str
    budget: dict
    steps: list[Step]

    @classmethod
    def from_dict(cls, d: dict) -> "Plan":
        if not isinstance(d, dict):
            raise ValueError("plan must be an object")
        for k, t in (("goal", str), ("metric", str), ("budget", dict), ("steps", list)):
            if not isinstance(d.get(k), t):
                raise ValueError(f"plan: '{k}' is missing or not a {t.__name__}")
        return cls(d["goal"], d["metric"], d["budget"], [Step.from_dict(x) for x in d["steps"]])

    @classmethod
    def from_model(cls, d: dict) -> "Plan":
        """A plan written by a model: its runtime fields are ignored, so it can't mark work done or carry stale state."""
        plan = cls.from_dict(d)
        for s in plan.steps:
            s.status, s.attempts, s.checkpoint, s.touched, s.summary, s.last_failure = "pending", 0, None, [], "", ""
        return plan

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self, known_skills: set[str]) -> list[str]:
        errs: list[str] = []
        seen: set[str] = set()
        for st in self.steps:
            if st.id in seen:
                errs.append(f"duplicate step id: {st.id}")
            seen.add(st.id)
            if not STEP_ID.match(st.id):
                errs.append(f"step id {st.id!r} is not a valid identifier (lowercase letters, digits, underscores)")
            if st.skill not in known_skills:
                errs.append(f"step {st.id}: unknown skill {st.skill!r}")
            if not st.goal.strip():
                errs.append(f"step {st.id}: empty goal")
            sc = st.success_check
            if not (sc.run or sc.expects or sc.metric):  # an empty check would pass on finish_step alone
                errs.append(f"step {st.id}: success_check needs run, expects or a metric")
            for dep in st.depends_on:
                if dep not in seen and dep not in {x.id for x in self.steps}:
                    errs.append(f"step {st.id}: unknown dependency {dep!r}")
        errs += self._cycles()
        return errs

    def _cycles(self) -> list[str]:
        deps = {st.id: [d for d in st.depends_on if d in {x.id for x in self.steps}] for st in self.steps}
        state: dict[str, int] = {}  # 1 = on path, 2 = done
        out: list[str] = []

        def visit(n: str, path: list[str]) -> None:
            state[n] = 1
            for m in deps[n]:
                if state.get(m) == 1:
                    cyc = path[path.index(m):] + [m]
                    out.append("dependency cycle: " + " -> ".join(cyc))
                elif m not in state:
                    visit(m, path + [m])
            state[n] = 2

        for n in deps:
            if n not in state:
                visit(n, [n])
        return out

    def _status(self, step_id: str) -> str | None:
        return next((s.status for s in self.steps if s.id == step_id), None)

    def ready(self) -> list[Step]:
        return [s for s in self.steps
                if s.status == "pending" and all(self._status(d) in ("done", "skipped") for d in s.depends_on)]

    def blocked(self) -> list[Step]:
        bad = {s.id for s in self.steps if s.status == "failed"}
        for _ in self.steps:  # fixed point; plan order is not guaranteed topological
            for s in self.steps:
                if s.status == "pending" and s.id not in bad and any(d in bad for d in s.depends_on):
                    bad.add(s.id)
        return [s for s in self.steps if s.status == "pending" and s.id in bad]

    def step(self, step_id: str) -> Step:
        for s in self.steps:
            if s.id == step_id:
                return s
        raise KeyError(step_id)

    def is_finished(self) -> bool:
        return all(s.status in ("done", "failed", "skipped") for s in self.steps)

    def merge_revision(self, new: "Plan") -> "Plan":
        keep = {s.id: s for s in self.steps if s.status in ("done", "running")}
        steps = [copy.deepcopy(keep.get(s.id, s)) for s in new.steps]
        listed = {s.id for s in new.steps}
        steps += [copy.deepcopy(s) for s in self.steps if s.id in keep and s.id not in listed]
        return Plan(new.goal, new.metric, copy.deepcopy(new.budget), steps)


def load_plan(project: Project) -> Plan | None:
    text = project.read("plan.json")
    return Plan.from_dict(json.loads(text)) if text.strip() else None


def save_plan(project: Project, plan: Plan) -> None:
    project.write("plan.json", json.dumps(plan.to_dict(), indent=2) + "\n")
    rows = ["| id | skill | status | depends_on | goal |", "|---|---|---|---|---|"]
    def cell(x: str) -> str:
        return " ".join(x.split()).replace("|", "\\|")

    rows += [f"| {cell(s.id)} | {cell(s.skill)} | {s.status} | {cell(', '.join(s.depends_on))} | {cell(s.goal)} |"
             for s in plan.steps]
    project.write("PLAN.md", f"# Plan: {plan.goal}\n\n" + "\n".join(rows) + "\n")
