import json
from pathlib import Path

import pytest
from google.adk.tools import FunctionTool

from beyondml.harness.executor import Executor
from beyondml.harness.factory import build_step_agent
from beyondml.harness.skills import SkillInfo
from beyondml.toolkit.control import control_tools
from beyondml.toolkit.env import StepOutcome, ToolEnv
from beyondml.workspace.events import EventBus
from beyondml.workspace.plan import Plan
from beyondml.workspace.project import Project
from scripted_llm import ScriptedLlm, txt


@pytest.fixture(autouse=True)
def _home(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("BEYONDML_HOME", str(tmp_path_factory.mktemp("bhome")))


def _ask_tool(env):
    return {f.__name__: f for f in control_tools(env)}["ask_user"]


def _questions(env):
    lines = env.project.path("agents/s1/events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(x)["data"] for x in lines if json.loads(x)["kind"] == "question"]


async def test_ask_user_options_autonomous(tmp_path):
    project = Project.init(tmp_path)

    async def never(q):
        raise AssertionError("asker called")

    env = ToolEnv(project, EventBus(project), "s1", "autonomous", never)
    answer = await _ask_tool(env)("Which column is the target?", options=["a", "b"], recommended="a")
    assert answer == {"answer": "a"}
    assert _questions(env) == [{"question": "Which column is the target?", "options": ["a", "b"], "recommended": "a"}]


async def test_ask_user_options_prompt(tmp_path):
    project = Project.init(tmp_path)
    seen = []

    async def asker(q):
        seen.append(q)
        return "b"

    env = ToolEnv(project, EventBus(project), "s1", "auto", asker)
    assert await _ask_tool(env)("Pick one", options=["a", "b"], recommended="a") == {"answer": "b"}
    assert "Options: a | b (recommended: a)" in seen[0]


def test_step_agent_gets_ml_tools(tmp_path):
    project = Project.init(tmp_path)
    skill = SkillInfo("search", "demo", "step", {"evolve", "finish_step"}, None, "tabular", "Search.", Path("."), None)
    plan = Plan.from_dict({"goal": "g", "metric": "f1", "budget": {},
                           "steps": [{"id": "s", "skill": "search", "goal": "search"}]})

    async def ask(q):
        return "ok"

    agent = build_step_agent(project, EventBus(project), plan, plan.step("s"), {"search": skill},
                             resolver=lambda n, m=None: ScriptedLlm(script=[[txt("x")]], seen=[]), mode="auto",
                             asker=ask, outcome=StepOutcome(), on_touch=lambda p: None)
    assert {t.name for t in agent.tools if isinstance(t, FunctionTool)} == {"evolve", "finish_step"}


def test_metric_lookup_by_key(tmp_path):
    project = Project.init(tmp_path)
    plan = Plan.from_dict({"goal": "g", "metric": "f1", "budget": {}, "steps": [
        {"id": "evaluate", "skill": "python-step", "goal": "e", "success_check": {"metric": "health", "min": 1}}]})
    project.ledger_append({"agent": "evaluate", "tool": "evaluate", "metrics": {"health": 1, "accuracy": 0.8}})
    project.ledger_append({"agent": "evaluate", "tool": "explain", "metrics": {"top_share": 0.3}})

    async def no(payload):
        return False

    executor = Executor(project, EventBus(project), plan, {}, resolver=None, mode="autonomous", asker=None,
                        approver=no)
    assert executor.check(plan.step("evaluate")) == (True, "")


def test_default_model_is_qwen3(tmp_path):
    assert Project.init(tmp_path).settings()["model"]["default"] == "ollama_chat/qwen3:8b"


def test_litellm_tool_calls_can_import_mcp():
    # litellm>=1.106 imports its MCP gateway handler on every tool-carrying call but does not declare `mcp`;
    # without it every real agent turn fails with APIConnectionError("No module named 'mcp'").
    import litellm.responses.mcp.chat_completions_handler  # noqa: F401


async def test_groq_drops_replayed_reasoning(tmp_path, monkeypatch):
    # Groq rejects `reasoning_content` on assistant history turns, which ADK builds from a reasoning model's thought
    # parts (gpt-oss-120b failed its first retry with BadRequestError "property 'reasoning_content' is unsupported").
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.models.llm_request import LlmRequest
    from google.genai import types

    from beyondml.harness.adk import default_resolver

    sent = []

    async def capture(self, llm_request, stream=False):
        sent.append(llm_request.contents)
        yield None

    monkeypatch.setattr(LiteLlm, "generate_content_async", capture)
    turn = types.Content(role="model", parts=[types.Part(text="thinking", thought=True),
                                              types.Part(function_call=types.FunctionCall(name="f", args={}))])
    only_thought = types.Content(role="model", parts=[types.Part(text="hmm", thought=True)])
    request = LlmRequest(contents=[types.Content(role="user", parts=[types.Part(text="hi")]), turn, only_thought])
    model = default_resolver(Project.init(tmp_path))("planner", "groq/openai/gpt-oss-120b")
    [_ async for _ in model.generate_content_async(request)]
    assert [[p.thought for p in c.parts] for c in sent[0]] == [[None], [None]]
    assert turn.parts[0].thought  # the session's own event is not mutated
