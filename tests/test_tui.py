import pytest
from textual.widgets import ContentSwitcher, Input

from beyondml.gateway.session import Session
from beyondml.tui.app import BeyondMLApp
from beyondml.tui.views.approvals import ApprovalModal
from beyondml.tui.views.plan_board import DepGraph, StepCard
from beyondml.workspace.plan import Plan, load_plan, save_plan
from scripted_llm import scripted_resolver, txt
from test_session import ASKER, PLAN, PLANNER


@pytest.fixture(autouse=True)
def _home(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("BEYONDML_HOME", str(tmp_path_factory.mktemp("bhome")))


TWO = {"goal": "g", "metric": "accuracy", "budget": {}, "steps": [
    {"id": "prepare", "skill": "python-step", "goal": "clean"},
    {"id": "train", "skill": "python-step", "goal": "fit", "depends_on": ["prepare"]}]}
SIZE = (120, 40)


async def until(pilot, pred, timeout=15.0):
    for _ in range(int(timeout / 0.05)):
        if pred():
            return
        await pilot.pause(0.05)
    raise AssertionError("timed out waiting for the TUI")


def with_plan(tmp_path, plan=TWO, **kw):
    session = Session(tmp_path, goal="g", **kw)
    session.bind()
    save_plan(session.project, Plan.from_dict(plan))
    return session


async def test_plan_board_cards_graph_and_live_status(tmp_path):
    session = with_plan(tmp_path)
    app = BeyondMLApp(session)
    async with app.run_test(size=SIZE) as pilot:
        await until(pilot, lambda: len(app.query(StepCard)) == 2)
        cards = {c.step_id: c for c in app.query(StepCard)}
        assert cards["train"].step["status"] == "pending"
        graph = str(app.query_one(DepGraph).render())
        assert "prepare" in graph and "train" in graph and "← prepare" in graph
        plan = load_plan(session.project)  # the executor saves plan.json, then emits
        plan.step("train").status = "running"
        save_plan(session.project, plan)
        session.bus.emit("executor", "step_status", {"id": "train", "status": "running", "reason": ""})
        await until(pilot, lambda: {c.step_id: c for c in app.query(StepCard)}["train"].step["status"] == "running")
        assert cards["train"].step["started"] > 0
        assert "◐ train" in str(app.query_one(DepGraph).render())
        assert "idle" in str(app.query_one("#topbar").render())  # no run task in this session


async def test_enter_on_card_opens_agent_screen(tmp_path):
    session = with_plan(tmp_path)
    session.project.write("agents/prepare/instruction.md", "# prepare\nexactly this text")
    app = BeyondMLApp(session)
    async with app.run_test(size=SIZE) as pilot:
        await until(pilot, lambda: len(app.query(StepCard)) == 2)
        app.query(StepCard).first().focus()
        await pilot.press("enter")
        await until(pilot, lambda: app.query_one(ContentSwitcher).current == "agent")
        assert app.agent_view.agent_id == "prepare"


async def test_keys_switch_screens(tmp_path):
    app = BeyondMLApp(with_plan(tmp_path))
    async with app.run_test(size=SIZE) as pilot:
        switcher = app.query_one(ContentSwitcher)
        for key, name in (("3", "experiments"), ("4", "files"), ("2", "agent"), ("1", "plan")):
            await pilot.press(key)
            await until(pilot, lambda: switcher.current == name)


async def test_chat_dock_sends_to_lead(tmp_path):
    session = with_plan(tmp_path, resolver=scripted_resolver({"lead": [[[txt("hello there")]]]}))
    app = BeyondMLApp(session)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("c")
        await until(pilot, lambda: isinstance(app.focused, Input))
        await pilot.press(*"hi", "enter")
        await until(pilot, lambda: "hello there" in app.chat.transcript())
        assert "hi" in app.chat.transcript()


async def test_prompts_open_modal_and_answers_reach_the_run(tmp_path):
    resolver = scripted_resolver({"planner": [PLANNER], "a": [ASKER]})
    session = Session(tmp_path, mode="auto", resolver=resolver, goal="g")
    session.bind()
    app = BeyondMLApp(session)
    async with app.run_test(size=SIZE) as pilot:
        await session.start("g")
        await until(pilot, lambda: isinstance(app.screen, ApprovalModal))
        assert app.screen.prompt["kind"] == "plan" and "a" in str(app.screen.query_one("#prompt-body").render())
        await pilot.press("y")
        await until(pilot, lambda: isinstance(app.screen, ApprovalModal) and app.screen.prompt["event"] == "question")
        assert "which colour?" in str(app.screen.query_one("#prompt-body").render())
        await pilot.press(*"blue", "enter")
        await until(pilot, lambda: not session.running)
        assert load_plan(session.project).step("a").status == "done"
        await until(pilot, lambda: not isinstance(app.screen, ApprovalModal))


async def test_run_form_starts_a_run_when_there_is_no_plan(tmp_path):
    resolver = scripted_resolver({"planner": [PLANNER], "a": [ASKER]})
    session = Session(tmp_path, mode="autonomous", resolver=resolver, goal="g")
    session.bind()
    app = BeyondMLApp(session)
    async with app.run_test(size=SIZE) as pilot:
        await until(pilot, lambda: app.query_one("#goal").display)
        app.query_one("#goal", Input).focus()
        await pilot.press(*"predict", "enter")
        await until(pilot, lambda: session.running or session.run_task is not None)
        await until(pilot, lambda: session.run_task.done())
        await until(pilot, lambda: len(app.query(StepCard)) == 1)
        assert load_plan(session.project).step("a").status == "done"
