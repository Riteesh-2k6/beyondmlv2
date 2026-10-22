"""Thin glue between BeyondML and Google ADK: models, sessions, and one agent turn."""
from __future__ import annotations

import asyncio
import functools
import re
from dataclasses import dataclass, field
from typing import Callable

from google.adk.agents.run_config import RunConfig
from google.adk.flows.llm_flows.functions import REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
from google.adk.models.base_llm import BaseLlm
from google.adk.runners import Runner
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.genai import types

from beyondml.workspace.events import EventBus, normalize
from beyondml.workspace.project import Project

APP_NAME = "beyondml"
USER_ID = "local"
ModelResolver = Callable[[str, "str | None"], "str | BaseLlm"]


def default_resolver(project: Project) -> ModelResolver:
    cfg = project.settings()["model"]

    def resolve(agent_name: str, requested: str | None = None):
        name = requested or cfg["default"]
        if name.startswith("gemini"):
            return name
        if name.startswith("ollama"):
            return _llm_class()(model=name, api_base=cfg["api_base"], num_retries=3)
        return _llm_class()(model=name, num_retries=3)

    return resolve


RATE_LIMIT_WAITS = 10  # ponytail: fixed count; a daily-quota 429 still fails after ~10 min of waiting


def _retry_after(error) -> float:
    """Seconds the provider asked us to wait ("try again in 1m2.5s"), else 20."""
    match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(error))
    return 60 * int(match[1] or 0) + float(match[2]) if match else 20.0


@functools.cache
def _llm_class():
    from google.adk.models.lite_llm import LiteLlm  # heavy import, only when a model is built
    from litellm.exceptions import RateLimitError

    class BeyondLlm(LiteLlm):
        """LiteLlm that waits out rate limits (litellm's own retries do not wait) and suits Groq's history rules."""

        async def generate_content_async(self, llm_request, stream=False):
            if self.model.startswith("groq/"):  # Groq rejects `reasoning_content`, which ADK replays from thoughts
                contents = []
                for content in llm_request.contents:
                    parts = [p for p in content.parts or [] if not p.thought]
                    if parts:
                        contents.append(content.model_copy(update={"parts": parts}))
                llm_request = llm_request.model_copy(update={"contents": contents})
            for attempt in range(RATE_LIMIT_WAITS + 1):
                started = False
                try:
                    async for response in super().generate_content_async(llm_request, stream):
                        started = True
                        yield response
                    return
                except RateLimitError as error:
                    if started or attempt == RATE_LIMIT_WAITS:
                        raise
                    await asyncio.sleep(_retry_after(error))

    return BeyondLlm


def session_service(project: Project) -> SqliteSessionService:
    return SqliteSessionService(db_path=str(project.root / ".beyondml" / "sessions.sqlite"))


async def ensure_session(service, session_id: str) -> None:
    if await service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id) is None:
        await service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)


def user_text(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=text)])


def confirmation_reply(call_id: str, approved: bool) -> types.Content:
    return types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
        id=call_id, name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME, response={"confirmed": approved}))])


@dataclass
class TurnResult:
    confirmations: list[dict] = field(default_factory=list)
    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


async def run_turn(runner: Runner, session_id: str, message: types.Content, bus: EventBus,
                   *, max_llm_calls: int = 500) -> TurnResult:
    result = TurnResult()
    me = runner.agent.name
    async for ev in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=message,
                                     run_config=RunConfig(max_llm_calls=max_llm_calls)):
        texts = []
        for agent, kind, data in normalize(ev):
            bus.emit(agent, kind, data)
            if kind == "approval_request":
                result.confirmations.append(data)
            elif kind == "message_delta" and agent == me and not data["partial"]:
                texts.append(data["text"])
        if texts:
            result.text = "".join(texts)
        usage = ev.usage_metadata
        if usage is not None:
            result.prompt_tokens += usage.prompt_token_count or 0
            result.completion_tokens += usage.candidates_token_count or 0
    return result
