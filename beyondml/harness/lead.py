"""The lead agent: the one the user talks to. It explains and edits the plan; it never runs code."""
from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.runners import Runner

from beyondml.harness.adk import APP_NAME, ModelResolver, ensure_session, run_turn, session_service, user_text
from beyondml.harness.skills import SkillInfo, step_skills
from beyondml.toolkit.control import control_tools
from beyondml.toolkit.env import Asker, ToolEnv
from beyondml.toolkit.files import file_tools
from beyondml.toolkit.registry import build_tools
from beyondml.workspace.events import EventBus
from beyondml.workspace.project import Project

ROLE = ("You are the lead ML engineer for this project. Answer the user, explain decisions, and change the plan with "
        "update_plan when asked; you never run code yourself.")
LEAD_TOOLS = {"read_file", "list_files", "ask_user", "update_plan"}


def make_lead(project: Project, bus: EventBus, skills: dict[str, SkillInfo], *, resolver: ModelResolver, mode: str,
              asker: Asker) -> LlmAgent:
    env = ToolEnv(project, bus, agent="lead", mode=mode, asker=asker, skill_names=set(step_skills(skills)))
    instruction = (f"{ROLE}\n\n## Project\n{project.read('PROJECT.md')}\n\n## Memory\n{project.read('MEMORY.md')}\n\n"
                   f"## Plan\n{project.read('PLAN.md')}\n")
    tools = build_tools(env, file_tools(env) + control_tools(env), allowed=LEAD_TOOLS)
    return LlmAgent(name="lead", model=resolver("lead", None), instruction=instruction, tools=tools)


async def chat(project: Project, bus: EventBus, lead: LlmAgent, text: str, *, session_id: str = "lead") -> str:
    sessions = session_service(project)
    await ensure_session(sessions, session_id)
    runner = Runner(app_name=APP_NAME, agent=lead, session_service=sessions)
    return (await run_turn(runner, session_id, user_text(text), bus)).text
