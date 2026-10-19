"""Permission modes and wrapping tool functions into ADK FunctionTools."""
from __future__ import annotations

import functools
import inspect
import re
from typing import Callable

from google.adk.tools import FunctionTool

from beyondml.toolkit.env import StepAborted, ToolEnv
from beyondml.workspace.project import PathError

NETWORK = re.compile(r"\b(requests|urllib|httpx|aiohttp|socket|http\.client|ftplib|smtplib)\b|https?://")


def needs_confirmation(env: ToolEnv, tool: str, args: dict) -> bool:
    if env.mode == "autonomous":
        return False
    if env.mode == "ask":
        return tool in {"run_python", "install_package"}
    if tool == "install_package":
        return True
    if tool == "run_python":
        # ponytail: regex network check, upgrade to the docker sandbox with network off when it matters
        try:
            text = env.project.path(args.get("script", "")).read_text(encoding="utf-8", errors="replace")
        except (PathError, OSError):
            return False  # unreadable: run_python rejects it anyway
        return bool(NETWORK.search(text))
    return False


def _failed(result) -> bool:
    return isinstance(result, dict) and (result.get("ok") is False or "error" in result)


def _count(env: ToolEnv, result) -> None:
    if env.outcome is None:  # R4
        return
    env.outcome.consecutive_errors = env.outcome.consecutive_errors + 1 if _failed(result) else 0
    if env.outcome.consecutive_errors >= 3:
        raise StepAborted(f"{env.agent}: 3 consecutive tool errors")


def _wrap(env: ToolEnv, fn: Callable) -> Callable:
    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def wrapped(*a, **kw):
            result = await fn(*a, **kw)
            _count(env, result)
            return result
    else:
        @functools.wraps(fn)
        def wrapped(*a, **kw):
            result = fn(*a, **kw)
            _count(env, result)
            return result
    return wrapped


def build_tools(env: ToolEnv, funcs: list[Callable], allowed: set[str] | None = None) -> list[FunctionTool]:
    tools = []
    for fn in funcs:
        name = fn.__name__
        if allowed is not None and name not in allowed:
            continue
        if env.mode == "autonomous":
            tools.append(FunctionTool(_wrap(env, fn)))
            continue

        # ADK invokes this as confirm(**tool_args), filtered to the tool's own parameters
        def confirm(_n=name, **args) -> bool:
            return needs_confirmation(env, _n, args)

        tools.append(FunctionTool(_wrap(env, fn), require_confirmation=confirm))
    return tools
