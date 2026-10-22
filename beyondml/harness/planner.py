"""The planner agent: turns the goal and DATA.md into a validated plan.json."""
from __future__ import annotations

import json
from typing import Callable

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.tools import ToolContext

from beyondml.harness.adk import APP_NAME, USER_ID, ModelResolver, ensure_session, run_turn, session_service, user_text
from beyondml.harness.skills import SkillInfo, step_skills
from beyondml.workspace.events import EventBus
from beyondml.workspace.plan import Plan, save_plan
from beyondml.workspace.project import Project

EXAMPLE = {
    "goal": "predict churn", "metric": "f1", "budget": {"minutes": 60, "usd": 2.0},
    "steps": [{
        "id": "features", "skill": "python-step", "depends_on": ["eda"],
        "goal": "derive features from usage and billing columns",
        "inputs": ["data/processed/clean.parquet"], "outputs": ["src/features.py"],
        "success_check": {"run": "src/features.py", "expects": ["data/processed/features.parquet"]},
    }],
}
RULES = """- Step ids are snake_case identifiers (letters, digits, underscores; start with a letter).
- Every step names a skill from the catalogue; use python-step when nothing else fits.
- Success checks name concrete files (expects) or a metric with a minimum.
- When a skill's description names a Check, use it as that step's success_check (you choose the minimum).
- No step reads an input marked unusable in DATA.md."""


class PlanError(Exception): ...


def _instruction(project: Project, skills: dict[str, SkillInfo], goal: str, extra: str = "") -> str:
    catalogue = "\n".join(f"- {s.name} ({s.modality or 'any'}): {s.description}" for s in step_skills(skills).values())
    return (
        "You are the planner for a machine-learning project built by BeyondML. Write the plan for the goal below "
        "and submit it with submit_plan. Fix every error it reports and submit again.\n\n"
        f"## Goal\n{goal}\n\n## Project\n{project.read('PROJECT.md')}\n\n## Data\n{project.read('DATA.md')}\n\n"
        f"## Step skills\n{catalogue}\n\n## Plan shape\n```json\n{json.dumps(EXAMPLE, indent=2)}\n```\n\n"
        f"## Rules\n{RULES}\n{extra}"
    )


async def _plan_loop(project: Project, bus: EventBus, skills: dict[str, SkillInfo], resolver: ModelResolver,
                     instruction: str, finish: Callable[[Plan], Plan], max_attempts: int) -> Plan:
    known = set(step_skills(skills))
    state = {"calls": 0, "accepted": None, "errors": []}

    def submit_plan(plan: dict, tool_context: ToolContext) -> dict:
        """Submit the full plan. Returns ok, or the errors to fix before submitting again."""
        state["calls"] += 1
        try:
            candidate = finish(Plan.from_model(plan))
            errors = candidate.validate(known)
        except ValueError as e:
            errors = [str(e)]
        if not errors:
            state["accepted"] = candidate
        else:
            state["errors"] = errors
        if not errors or state["calls"] >= max_attempts:
            tool_context.actions.skip_summarization = True  # end the turn here
        return {"ok": not errors, "errors": errors}

    agent = LlmAgent(name="planner", model=resolver("planner", None), instruction=instruction, tools=[submit_plan])
    svc = session_service(project)
    existing = await svc.list_sessions(app_name=APP_NAME, user_id=USER_ID)
    session_id = f"planner-{sum(s.id.startswith('planner-') for s in existing.sessions) + 1}"
    await ensure_session(svc, session_id)
    runner = Runner(app_name=APP_NAME, agent=agent, session_service=svc)

    silent_turns = 0
    message = user_text("Write the plan and call submit_plan.")
    while True:
        before = state["calls"]
        await run_turn(runner, session_id, message, bus)
        if state["accepted"] is not None:
            plan = state["accepted"]
            save_plan(project, plan)
            bus.emit("planner", "plan_updated", {"steps": [s.id for s in plan.steps]})
            return plan
        silent_turns += state["calls"] == before
        if state["calls"] + silent_turns >= max_attempts:
            raise PlanError("planner gave up after %d attempts; last errors: %s"
                            % (max_attempts, "; ".join(state["errors"]) or "no plan submitted"))
        message = user_text("Call submit_plan with a corrected plan.")


async def make_plan(project: Project, bus: EventBus, skills: dict[str, SkillInfo], *, resolver: ModelResolver,
                    goal: str, max_attempts: int = 3) -> Plan:
    return await _plan_loop(project, bus, skills, resolver, _instruction(project, skills, goal),
                            lambda p: p, max_attempts)


async def replan(project: Project, bus: EventBus, skills: dict[str, SkillInfo], *, resolver: ModelResolver,
                 plan: Plan, failure: dict, max_attempts: int = 3) -> Plan:
    extra = (f"\n## Current plan\n```json\n{json.dumps(plan.to_dict(), indent=2)}\n```\n\n"
             f"## What failed\n```json\n{json.dumps(failure, indent=2)}\n```\n"
             "Submit a revised full plan. Steps already done are kept as they are, so you may omit them.")
    return await _plan_loop(project, bus, skills, resolver, _instruction(project, skills, plan.goal, extra),
                            plan.merge_revision, max_attempts)
