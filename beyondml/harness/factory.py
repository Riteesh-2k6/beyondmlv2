"""Builds one dedicated ADK agent per plan step, and saves exactly what it was told."""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Callable

from google.adk.agents import LlmAgent

from beyondml.harness.adk import ModelResolver
from beyondml.harness.skills import SkillInfo, knowledge_toolset, step_skills
from beyondml.toolkit.control import control_tools
from beyondml.toolkit.env import Asker, StepOutcome, ToolEnv
from beyondml.toolkit.files import file_tools
from beyondml.toolkit.ml import ml_tools
from beyondml.toolkit.python import python_tools
from beyondml.toolkit.registry import build_tools
from beyondml.workspace.events import EventBus
from beyondml.workspace.plan import Plan, Step
from beyondml.workspace.project import Project

ROLE = ("You are the '{step_id}' agent in a machine-learning project built by BeyondML. "
        "You own exactly one plan step. Work only toward its goal and outputs.")
RULES = ("- Call finish_step(summary, outputs) when the outputs exist.\n- Never write under data/raw/.\n"
         "- Write key numbers to os.environ.get('BEYONDML_METRICS', 'metrics.json').\n"
         "- If you are blocked, call finish_step explaining what blocks you.")


def _instruction(project: Project, plan: Plan, step: Step, skill: SkillInfo) -> str:
    detail = {"inputs": step.inputs, "outputs": step.outputs, "success_check": asdict(step.success_check)}
    upstream = [f"- {d}: {plan.step(d).summary}" for d in step.depends_on if plan.step(d).status == "done"]
    parts = [
        ROLE.format(step_id=step.id),
        f"## Your step\n{step.goal}\n\n```json\n{json.dumps(detail, indent=2)}\n```",
        "## Upstream results\n" + ("\n".join(upstream) or "- none"),
    ]
    if step.last_failure:
        parts.append(f"## Previous attempt failed\n{step.last_failure}")
    parts += [
        f"## Project\n{project.read('PROJECT.md')}",
        f"## Data\n{project.read('DATA.md')}",  # the real column names; small models otherwise guess famous ones
        f"## Memory\n{project.read('MEMORY.md')}",
        f"## How to do this step\n{skill.body}",
        f"## Rules\n{RULES}",
    ]
    return "\n\n".join(parts) + "\n"


def build_step_agent(project: Project, bus: EventBus, plan: Plan, step: Step, skills: dict[str, SkillInfo], *,
                     resolver: ModelResolver, mode: str, asker: Asker, outcome: StepOutcome,
                     on_touch: Callable[[list[str]], None]) -> LlmAgent:
    skill = skills[step.skill]
    instruction = _instruction(project, plan, step, skill)
    project.write(f"agents/{step.id}/instruction.md", instruction)
    env = ToolEnv(project, bus, agent=step.id, mode=mode, asker=asker, skill_names=set(step_skills(skills)),
                  outcome=outcome, on_touch=on_touch)
    tools = build_tools(env, file_tools(env) + python_tools(env) + control_tools(env) + ml_tools(env),
                        allowed=skill.allowed_tools)
    knowledge = knowledge_toolset(skills)
    return LlmAgent(name=step.id, model=resolver(step.id, step.model or skill.model_hint),
                    instruction=instruction, tools=tools + ([knowledge] if knowledge else []))
