"""Control tools: ask the human, finish the step, revise the plan."""
from __future__ import annotations

from typing import Callable

from google.adk.tools import ToolContext

from beyondml.toolkit.env import AUTONOMOUS_ANSWER, StepOutcome, ToolEnv
from beyondml.workspace.plan import Plan, load_plan, save_plan


def control_tools(env: ToolEnv) -> list[Callable]:
    async def ask_user(question: str, options: list[str] = [], recommended: str = "") -> dict:
        """Ask the human a question and wait for the answer. In autonomous mode your recommendation is taken.

        Args:
            question: The question text.
            options: Choices to offer, if the answer is one of a few.
            recommended: The choice you would make yourself.
        """
        env.bus.emit(env.agent, "question", {"question": question, "options": list(options),
                                             "recommended": recommended})
        if env.mode == "autonomous":
            return {"answer": recommended or AUTONOMOUS_ANSWER}
        text = question
        if options:
            text += "\nOptions: " + " | ".join(options) + (f" (recommended: {recommended})" if recommended else "")
        return {"answer": await env.asker(text)}

    def finish_step(summary: str, outputs: list[str], tool_context: ToolContext) -> dict:
        """Declare this step finished. Your turn ends here.

        Args:
            summary: What was done, including any assumptions.
            outputs: Project-relative paths of the files this step produced.
        """
        if env.outcome is None:
            env.outcome = StepOutcome()
        env.outcome.finished = True
        env.outcome.summary = summary
        env.outcome.outputs = list(outputs)
        tool_context.actions.skip_summarization = True  # end the turn; small models otherwise call it again and again
        return {"ok": True}

    def update_plan(plan: dict) -> dict:
        """Replace the project plan with a revised one. Done and running steps are kept as they are.

        Args:
            plan: Object with goal, metric, budget and steps.
        """
        try:
            new = Plan.from_model(plan)
            old = load_plan(env.project)
            merged = old.merge_revision(new) if old else new  # validate the merge: a revision may omit done steps
            errs = merged.validate(env.skill_names)
            if errs:
                return {"ok": False, "errors": errs}
            save_plan(env.project, merged)
        except ValueError as e:
            return {"ok": False, "errors": [str(e)]}
        env.bus.emit(env.agent, "plan_updated", {"steps": [s.id for s in merged.steps]})
        return {"ok": True}

    return [ask_user, finish_step, update_plan]
