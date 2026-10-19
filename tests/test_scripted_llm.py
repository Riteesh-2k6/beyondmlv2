from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from scripted_llm import ScriptedLlm, fc, scripted_resolver, txt


def add(a: int, b: int) -> dict:
    """Add."""
    return {"sum": a + b}


async def test_scripted_agent_calls_tool():
    model = ScriptedLlm(script=[[fc("add", a=2, b=3)], [txt("done")]])
    agent = LlmAgent(name="a", model=model, tools=[add])
    ss = InMemorySessionService()
    runner = Runner(app_name="t", agent=agent, session_service=ss)
    await ss.create_session(app_name="t", user_id="u", session_id="s")
    events = [
        ev
        async for ev in runner.run_async(
            user_id="u", session_id="s",
            new_message=types.Content(role="user", parts=[types.Part(text="hi")]),
        )
    ]
    resps = [r.response for ev in events for r in ev.get_function_responses()]
    assert {"sum": 5} in resps
    final = [ev for ev in events if ev.is_final_response()][-1]
    assert final.content.parts[0].text == "done"
    assert model.calls == 2


def test_resolver_attempts_and_unknown():
    r = scripted_resolver({"x": [[[txt("a")]], [[txt("b")]]]})
    m1, m2, m3 = r("x", None), r("x", None), r("x", None)
    assert m1.script == [[txt("a")]] and m2.script == m3.script == [[txt("b")]]
    assert r.models["x"] == [m1, m2, m3]
    try:
        r("y", None)
        assert False
    except KeyError as e:
        assert "no script for y" in str(e)
