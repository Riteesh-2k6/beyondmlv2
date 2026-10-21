"""Deterministic stand-in for an LLM, so ADK agents can be driven in tests."""
import inspect
from collections import defaultdict
from typing import Callable

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types


def fc(name: str, **args) -> types.Part:
    return types.Part(function_call=types.FunctionCall(name=name, args=args))


def txt(text: str) -> types.Part:
    return types.Part(text=text)


def last_response(req: LlmRequest) -> dict:
    for content in reversed(req.contents):
        for part in reversed(content.parts or []):
            if part.function_response is not None:
                return part.function_response.response or {}
    return {}


class ScriptedLlm(BaseLlm):
    model: str = "scripted"
    script: list = []  # list[Part] or Callable[[LlmRequest], list[Part]] per call
    calls: int = 0
    seen: list = []

    async def generate_content_async(self, llm_request, stream=False):
        entry = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        self.seen.append(llm_request)
        parts = entry(llm_request) if callable(entry) else entry
        if inspect.isawaitable(parts):  # async callables let a test hold the model mid-step
            parts = await parts
        yield LlmResponse(content=types.Content(role="model", parts=list(parts)))


def scripted_resolver(scripts: dict[str, list[list]]) -> Callable[[str, str | None], ScriptedLlm]:
    counts: dict[str, int] = defaultdict(int)

    def resolve(name: str, model: str | None = None) -> ScriptedLlm:
        if name not in scripts:
            raise KeyError(f"no script for {name}")
        attempts = scripts[name]
        llm = ScriptedLlm(script=attempts[min(counts[name], len(attempts) - 1)], seen=[])
        counts[name] += 1
        resolve.models[name].append(llm)
        return llm

    resolve.models = defaultdict(list)
    return resolve
