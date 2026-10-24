"""One call from raw data to an executed plan: the entry point the CLI (and later the gateway) wraps."""
from __future__ import annotations

import asyncio
import contextlib
import re
from pathlib import Path
from typing import Callable, Sequence

from beyondml.harness.adk import ModelResolver, default_resolver
from beyondml.harness.executor import Approver, Executor
from beyondml.harness.intake import run_intake
from beyondml.harness.planner import make_plan, replan
from beyondml.harness.skills import discover
from beyondml.toolkit.env import Asker
from beyondml.workspace.events import EventBus
from beyondml.workspace.plan import Plan, load_plan
from beyondml.workspace.project import Project


async def _pump(queue: asyncio.Queue, on_event: Callable[[dict], None]) -> None:
    while True:
        on_event(await queue.get())


class GoalRequired(ValueError): ...


def _set_goal(project: Project, goal: str) -> None:
    text = project.read("PROJECT.md")
    if "## Goal" in text:
        text = re.sub(r"## Goal\n.*?(?=\n## |\Z)", lambda m: f"## Goal\n{goal}\n", text, count=1, flags=re.S)
    else:
        text = text.rstrip() + f"\n\n## Goal\n{goal}\n"
    project.write("PROJECT.md", text)


async def run_goal(root: Path, goal: str | None, data: Sequence[Path] = (), *, mode: str | None = None,
                   model: str | None = None, resolver: ModelResolver | None = None,
                   asker: Asker, approver: Approver, on_event: Callable[[dict], None] | None = None,
                   bus: EventBus | None = None) -> Plan:
    root = Path(root)
    project = Project.open(root) if (root / ".beyondml").is_dir() else Project.init(root, goal)
    missing = []
    for path in data:
        try:
            project.add_data(Path(path))
        except OSError as e:  # a bad input is reported in DATA.md, it doesn't stop the run
            missing.append((str(path), f"{type(e).__name__}: {e}"))
    if len(missing) < len(data):
        project.checkpoint("add data")
    skills = discover(project)
    settings = project.settings()
    mode = mode or settings["run"]["mode"]
    base = resolver or default_resolver(project)
    resolve = (lambda name, requested=None: base(name, model)) if model else base
    bus = bus or EventBus(project)
    pump = asyncio.create_task(_pump(bus.subscribe(), on_event)) if on_event else None
    try:
        plan = load_plan(project)
        if plan is None or plan.is_finished():
            if not goal:
                raise GoalRequired("a goal is required to start a new plan")
            _set_goal(project, goal)  # PROJECT.md is what every agent reads as the goal
            run_intake(project, bus, missing)
            plan = await make_plan(project, bus, skills, resolver=resolve, goal=goal)
            if mode != "autonomous" and not await approver({"kind": "plan", "plan": plan.to_dict()}):
                return plan

        async def replanner(current: Plan, failure: dict) -> Plan:
            return await replan(project, bus, skills, resolver=resolve, plan=current, failure=failure)

        run_cfg = settings["run"]
        return await Executor(project, bus, plan, skills, resolver=resolve, mode=mode, asker=asker,
                              approver=approver, replanner=replanner, max_tool_calls=run_cfg["max_tool_calls"],
                              budget_minutes=run_cfg["budget_minutes"], budget_usd=run_cfg["budget_usd"]).run()
    finally:
        if pump:
            await asyncio.sleep(0)  # let queued events reach the printer
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump
