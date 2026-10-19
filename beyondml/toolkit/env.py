"""Shared context for every ADK tool a step agent gets."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable

from beyondml.workspace.events import EventBus
from beyondml.workspace.project import Project

Asker = Callable[[str], Awaitable[str]]
AUTONOMOUS_ANSWER = ("No human is available (autonomous mode). Choose the most reasonable option yourself "
                     "and state the assumption in one line of your finish_step summary.")


class StepAborted(Exception): ...


@dataclass
class StepOutcome:
    finished: bool = False
    summary: str = ""
    outputs: list[str] = field(default_factory=list)
    touched: set[str] = field(default_factory=set)
    consecutive_errors: int = 0


@dataclass
class ToolEnv:
    project: Project
    bus: EventBus
    agent: str
    mode: str  # ask | auto | autonomous
    asker: Asker
    skill_names: set[str] = field(default_factory=set)
    outcome: StepOutcome | None = None
    on_touch: Callable[[list[str]], None] | None = None  # called with project-relative paths after every write
