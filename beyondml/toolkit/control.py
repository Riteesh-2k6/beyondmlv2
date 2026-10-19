"""Control tools: ask the human, finish the step, revise the plan."""
from __future__ import annotations

from typing import Callable

from beyondml.toolkit.env import AUTONOMOUS_ANSWER, StepOutcome, ToolEnv
from beyondml.workspace.plan import Plan, load_plan, save_plan


def control_tools(env: ToolEnv) -> list[Callable]:
    async def ask_user(question: str) -> dict:
        """Ask the human a question and wait for the answer.

        Args:
            question: The question text.
        """
        env.bus.emit(env.agent, "question", {"question": question})
        if env.mode == "autonomous":
            return {"answer": AUTONOMOUS_ANSWER}
        return {"answer": await env.asker(question)}

    def finish_step(summary: str, outputs: list[str]) -> dict:
        """Declare this step finished.

        Args:
            summary: What was done, including any assumptions.
            outputs: Project-relative paths of the files this step produced.
        """
        if env.outcome is None:
            env.outcome = StepOutcome()
        env.outcome.finished = True
        env.outcome.summary = summary
        env.outcome.outputs = list(outputs)
        return {"ok": True}

    def update_plan(plan: dict) -> dict:
        """Replace the project plan with a revised one. Done and running steps are kept as they are.

        Args:
            plan: Object with goal, metric, budget and steps.
        """
        try:
            new = Plan.from_dict(plan)
            errs = new.validate(env.skill_names)
            if errs:
                return {"ok": False, "errors": errs}
            old = load_plan(env.project)
            merged = old.merge_revision(new) if old else new
            save_plan(env.project, merged)
        except ValueError as e:
            return {"ok": False, "errors": [str(e)]}
        env.bus.emit(env.agent, "plan_updated", {"steps": [s.id for s in merged.steps]})
        return {"ok": True}

    return [ask_user, finish_step, update_plan]
