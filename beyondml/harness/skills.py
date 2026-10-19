"""Three-level skill discovery: bundled, BEYONDML_HOME, project."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from google.adk.skills import Skill, load_skill_from_dir
from google.adk.tools.skill_toolset import SkillToolset

from beyondml.workspace.project import Project, beyondml_home

BUNDLED = Path(__file__).resolve().parent.parent / "skills"
KNOWLEDGE_TOOLS = ["list_skills", "load_skill", "load_skill_resource"]


@dataclass
class SkillInfo:
    name: str
    description: str
    kind: str  # "step" | "knowledge"
    allowed_tools: set[str] | None
    model_hint: str | None
    modality: str | None
    body: str
    path: Path
    skill: Skill


def _info(skill: Skill, path: Path) -> SkillInfo:
    fm = skill.frontmatter
    meta = (fm.metadata or {}).get("beyondml")
    meta = meta if isinstance(meta, dict) else {}
    tools = fm.allowed_tools
    return SkillInfo(
        name=fm.name, description=fm.description,
        kind="step" if meta.get("kind") == "step" else "knowledge",
        allowed_tools=set(tools.split()) if tools else None,
        model_hint=meta.get("model_hint"), modality=meta.get("modality"),
        body=skill.instructions, path=path, skill=skill,
    )


def discover(project: Project | None) -> dict[str, SkillInfo]:
    levels = [BUNDLED, beyondml_home() / "skills"]
    if project is not None:
        levels.append(Path(project.root) / ".beyondml" / "skills")
    found: dict[str, SkillInfo] = {}
    for level in levels:
        if not level.is_dir():
            continue
        for d in sorted(p for p in level.iterdir() if p.is_dir()):
            try:
                info = _info(load_skill_from_dir(d), d)
            except Exception as e:  # ADK raises ValueError on bad/missing frontmatter
                logging.warning("Skipping skill folder %s: %s", d, e)
                continue
            found[info.name] = info
    return found


def step_skills(skills: dict[str, SkillInfo]) -> dict[str, SkillInfo]:
    return {n: s for n, s in skills.items() if s.kind == "step"}


def knowledge_toolset(skills: dict[str, SkillInfo]) -> SkillToolset | None:
    ks = [s.skill for s in skills.values() if s.kind == "knowledge"]
    return SkillToolset(ks, tool_filter=KNOWLEDGE_TOOLS) if ks else None
