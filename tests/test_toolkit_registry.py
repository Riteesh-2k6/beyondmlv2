import pytest

from beyondml.toolkit.env import StepAborted, StepOutcome, ToolEnv
from beyondml.toolkit.files import file_tools
from beyondml.toolkit.registry import build_tools, needs_confirmation
from beyondml.workspace.events import EventBus
from beyondml.workspace.project import Project


@pytest.fixture(autouse=True)
def _home(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("BEYONDML_HOME", str(tmp_path_factory.mktemp("bhome")))


def mk(tmp_path, mode="auto"):
    project = Project.init(tmp_path / "p")

    async def asker(q):
        return ""

    return ToolEnv(project, EventBus(project), "s1", mode, asker, outcome=StepOutcome())


@pytest.mark.parametrize("mode,tool,script,expected", [
    ("autonomous", "run_python", "import requests", False), ("ask", "run_python", "print(1)", True),
    ("auto", "run_python", "print(1)", False), ("auto", "run_python", "import requests", True),
    ("auto", "install_package", None, True), ("ask", "read_file", None, False)])
def test_needs_confirmation(mode, tool, script, expected, tmp_path):
    env = mk(tmp_path, mode)
    args = {}
    if script is not None:
        env.project.write("s.py", script)
        args = {"script": "s.py"}
    assert needs_confirmation(env, tool, args) is expected


def test_build_tools_filters_allowed(tmp_path):
    env = mk(tmp_path)
    funcs = file_tools(env)
    assert [t.name for t in build_tools(env, funcs, {"read_file"})] == ["read_file"]
    assert [t.name for t in build_tools(env, funcs)] == ["read_file", "write_file", "list_files"]


async def test_three_errors_abort(tmp_path):
    env = mk(tmp_path)
    write = build_tools(env, file_tools(env), {"write_file"})[0].func
    assert [write("../x", "1")["ok"] for _ in range(2)] == [False, False]
    with pytest.raises(StepAborted):
        write("../x", "1")


async def test_confirmation_callable_matches_adk(tmp_path):
    env = mk(tmp_path, "auto")
    env.project.write("s.py", "import requests")

    def run_python(script: str) -> dict:
        """Run a script."""
        return {"ok": True}

    t = build_tools(env, [run_python])[0]
    assert await t.check_require_confirmation({"script": "s.py"}, None) is True


async def test_outcome_none_not_counted(tmp_path):
    env = mk(tmp_path)
    env.outcome = None
    write = build_tools(env, file_tools(env), {"write_file"})[0].func
    for _ in range(3):
        assert write("../x", "1")["ok"] is False


async def test_autonomous_tools_have_no_confirmation(tmp_path):
    def run_python(script: str) -> dict:
        """Run a script."""
        return {"ok": True}

    for mode, expected in (("autonomous", False), ("ask", True)):
        env = mk(tmp_path / mode, mode)
        t = build_tools(env, [run_python])[0]
        assert await t.check_require_confirmation({"script": "s.py"}, None) is expected
