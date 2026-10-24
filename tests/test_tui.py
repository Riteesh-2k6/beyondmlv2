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


ANY = {"expects": ["PROJECT.md"]}  # plans need a success check; these tests are not about checks
TWO = {"goal": "g", "metric": "accuracy", "budget": {}, "steps": [
    {"id": "prepare", "skill": "python-step", "goal": "clean", "success_check": ANY},
    {"id": "train", "skill": "python-step", "goal": "fit", "depends_on": ["prepare"], "success_check": ANY}]}
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


def gen(n, best, mean, div):
    return {"gen": n, "best_fitness": best, "mean_fitness": mean, "diversity": div, "evals": 10 * n,
            "seconds": 2.5 * n, "best_params": {"model": "rf", "n_estimators": 100 + n}}


async def test_agent_screen_tabs(tmp_path):
    session = with_plan(tmp_path)
    session.project.write("agents/prepare/instruction.md", "# prepare\nexactly this text")
    bus = session.bus
    bus.emit("prepare", "tool_call", {"id": "1", "name": "write_file", "args": {"path": "src/prepare.py"}})
    bus.emit("prepare", "tool_result", {"id": "1", "name": "write_file", "response": {"ok": True}})
    bus.emit("prepare", "artifact", {"path": "src/prepare.py"})
    bus.emit("prepare", "message_delta", {"text": "wrote the script", "partial": False})
    bus.emit("prepare", "chart", {"title": "missing by column", "kind": "bar", "x": [1, 2, 3],
                                  "series": {"missing": [0.1, 0.0, 0.4]}})
    bus.emit("train", "tool_call", {"id": "9", "name": "run_python", "args": {"script": "src/train.py"}})
    app = BeyondMLApp(session)
    async with app.run_test(size=SIZE) as pilot:
        await until(pilot, lambda: len(app.query(StepCard)) == 2)
        app.open_agent("prepare")
        await until(pilot, lambda: app.agent_view.info is not None)
        view = app.agent_view
        assert view.info["instruction"] == "# prepare\nexactly this text" and "run_python" in view.info["tools"]
        assert any("write_file" in line for line in view.live_lines) and not any("run_python" in l for l in view.live_lines)
        assert "wrote the script" in "\n".join(view.live_lines)
        assert "src/prepare.py" in view.outputs_text and "missing by column" in view.outputs_text
        assert view.evolve_text == ""
        bus.emit("prepare", "tool_call", {"id": "2", "name": "run_python", "args": {"script": "src/prepare.py"}})
        await until(pilot, lambda: any("run_python" in line for line in view.live_lines))
        for n, (b, m, d) in enumerate([(0.5, 0.3, 1.0), (0.7, 0.5, 0.8), (0.82, 0.6, 0.6)], start=1):
            bus.emit("prepare", "generation", gen(n, b, m, d))
        await until(pilot, lambda: "generation 3" in view.evolve_text)
        assert "0.82" in view.evolve_text and "n_estimators  103" in view.evolve_text
        assert "fitness per generation" in view.evolve_text and "diversity" in view.evolve_text


async def test_experiments_table_newest_first_and_live(tmp_path):
    session = with_plan(tmp_path)
    project = session.project
    project.ledger_append({"run_id": "1", "agent": "prepare", "script": "src/prepare.py", "args": [], "exit_code": 0,
                           "timed_out": False, "seconds": 1.2, "metrics": {}})
    project.ledger_append({"run_id": "2", "agent": "train", "script": "src/train.py", "args": ["--fast"],
                           "exit_code": 0, "timed_out": False, "seconds": 3.4, "metrics": {"accuracy": 0.9123}})
    app = BeyondMLApp(session)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("3")
        view = app.query_one("#experiments")
        await until(pilot, lambda: len(view.rows) == 2)
        assert view.rows[0][1:] == ["train", "src/train.py --fast", "0", "3.4", "accuracy=0.9123"]
        project.ledger_append({"run_id": "3", "agent": "train", "script": "src/train.py", "args": [], "exit_code": 1,
                               "timed_out": True, "seconds": 600.0, "metrics": {}})
        session.bus.emit("train", "tool_result", {"id": "x", "name": "run_python", "response": {"ok": False}})
        await until(pilot, lambda: len(view.rows) == 3)
        assert view.rows[0][3] == "timeout"


async def test_files_tree_hides_internals_and_previews(tmp_path):
    session = with_plan(tmp_path)
    (tmp_path / "src/a.py").write_bytes(b"x = 1\n")
    (tmp_path / "data/processed/t.csv").write_bytes(b"a,b\n1,2\n3,4\n")
    app = BeyondMLApp(session)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("4")
        view = app.query_one("#files")
        await until(pilot, lambda: view.tree.root.children)
        names = {str(c.label) for c in view.tree.root.children}
        assert "src" in names and ".git" not in names and ".beyondml" not in names
        view.preview(tmp_path / "src/a.py")
        assert "x = 1" in view.preview_text
        view.preview(tmp_path / "data/processed/t.csv")
        assert view.preview_title == "data/processed/t.csv" and "3,4" in view.preview_text
        view.preview(tmp_path.parent)  # outside the project: refused, nothing crashes
        assert "x = 1" not in view.preview_text


def test_cli_no_command_outside_project(tmp_path):
    from click.testing import CliRunner
    from beyondml.cli import main

    result = CliRunner().invoke(main, ["--project", str(tmp_path)])
    assert result.exit_code == 2 and "beyondml init" in result.output
    assert "tui" in main.commands
    assert "--no-serve" in CliRunner().invoke(main, ["tui", "--help"]).output


async def test_question_with_options_is_picked_by_digit(tmp_path):
    import asyncio

    session = with_plan(tmp_path)
    session.bus.emit("train", "question", {"question": "which model?", "options": ["rf", "xgb"], "recommended": "xgb"})
    asked = asyncio.ensure_future(session._ask("which model?"))
    app = BeyondMLApp(session)
    async with app.run_test(size=SIZE) as pilot:
        await until(pilot, lambda: isinstance(app.screen, ApprovalModal))
        body = str(app.screen.query_one("#prompt-body").render())
        assert "1  rf" in body and "2  xgb" in body and "recommended" in body
        assert not isinstance(app.focused, Input)  # digits must reach the modal, not an input
        await pilot.press("2")
        await until(pilot, lambda: asked.done())
        assert asked.result() == "xgb"


async def test_agent_tune_and_figure_file(tmp_path):
    import json

    session = with_plan(tmp_path)
    bus = session.bus
    for n, (v, state) in enumerate([(0.71, "COMPLETE"), (None, "PRUNED"), (0.78, "COMPLETE"), (0.74, "COMPLETE")], start=1):
        bus.emit("train", "trial", {"number": n, "value": v, "params": {"lr": 0.1 * n}, "state": state})
    session.project.write("reports/charts/folds.json", json.dumps({
        "data": [{"type": "scatter", "mode": "lines", "name": "valid", "x": [1, 2, 3], "y": [0.8, 0.79, 0.82]},
                 {"type": "bar", "name": "missing", "x": ["age", "fare"], "y": [19.9, 0.1]}],
        "layout": {"xaxis": {"title": {"text": "fold"}}}}))
    bus.emit("train", "chart", {"title": "accuracy by fold", "path": "reports/charts/folds.json"})
    bus.emit("train", "chart", {"title": "ghost", "path": "reports/charts/missing.json"})
    app = BeyondMLApp(session)
    async with app.run_test(size=SIZE) as pilot:
        app.open_agent("train")
        await until(pilot, lambda: app.agent_view.info is not None)
        view = app.agent_view
        assert "trials 4" in view.evolve_text and "complete 3" in view.evolve_text and "pruned 1" in view.evolve_text
        assert "max value 0.78 (trial 3)" in view.evolve_text and "lr  0.3" in view.evolve_text
        assert "accuracy by fold" in view.outputs_text and "fold" in view.outputs_text
        assert "ghost: reports/charts/missing.json could not be read" in view.outputs_text


async def test_chat_history_replays_on_open(tmp_path):
    session = with_plan(tmp_path)
    session.bus.emit("user", "message_delta", {"text": "what did the plan do?", "partial": False})
    session.bus.emit("lead", "message_delta", {"text": "it trained a model", "partial": False})
    app = BeyondMLApp(session)
    async with app.run_test(size=SIZE) as pilot:
        await until(pilot, lambda: "it trained a model" in app.chat.transcript())
        assert "what did the plan do?" in app.chat.transcript()
