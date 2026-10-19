import subprocess
import sys
import time

import pytest

from beyondml.toolkit.env import StepOutcome, ToolEnv
from beyondml.toolkit.python import python_tools
from beyondml.workspace.events import EventBus
from beyondml.workspace.project import Project


@pytest.fixture(autouse=True)
def _home(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("BEYONDML_HOME", str(tmp_path_factory.mktemp("bhome")))


def mk(tmp_path, name="p", touched=None):
    project = Project.init(tmp_path / name)

    async def asker(q):
        raise AssertionError("asker called")

    env = ToolEnv(project, EventBus(project), "s1", "auto", asker, outcome=StepOutcome(),
                  on_touch=(touched.extend if touched is not None else None))
    return env, {f.__name__: f for f in python_tools(env)}


def put(env, rel, text):
    p = env.project.root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


async def test_runs_and_captures(tmp_path):
    env, t = mk(tmp_path)
    put(env, "src/a.py", "print('hi')")
    r = await t["run_python"]("src/a.py")
    assert r["ok"] and r["exit_code"] == 0 and "hi" in r["stdout_tail"]
    assert len(env.project.ledger()) == 1


async def test_metrics_collected(tmp_path):
    env, t = mk(tmp_path)
    put(env, "src/a.py", "import os, json\njson.dump({'acc': 0.9}, open(os.environ['BEYONDML_METRICS'], 'w'))")
    r = await t["run_python"]("src/a.py")
    assert r["metrics"] == {"acc": 0.9}
    assert env.project.ledger()[0]["metrics"] == {"acc": 0.9}


async def test_env_stripped(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sekret")
    env, t = mk(tmp_path)
    put(env, "src/a.py", "import os\nprint(os.environ.get('OPENAI_API_KEY'))\nprint('PATH' in os.environ or 'Path' in os.environ)")
    r = await t["run_python"]("src/a.py")
    assert "None" in r["stdout_tail"] and "sekret" not in r["stdout_tail"] and "True" in r["stdout_tail"]


def _alive(pid):
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    import os
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


async def test_timeout_kills_tree(tmp_path):
    env, t = mk(tmp_path)
    put(env, "src/a.py", "import subprocess, sys, time\n"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "print(c.pid, flush=True)\ntime.sleep(60)")
    r = await t["run_python"]("src/a.py", timeout=3)
    assert r["timed_out"] and not r["ok"]
    pid = int(r["stdout_tail"].split()[0])
    deadline = time.time() + 5
    while _alive(pid) and time.time() < deadline:
        time.sleep(0.2)
    assert not _alive(pid)


async def test_files_changed(tmp_path):
    touched = []
    env, t = mk(tmp_path, touched=touched)
    put(env, "src/a.py", "import os\nos.makedirs('data/processed', exist_ok=True)\nopen('data/processed/x.csv','w').write('a')")
    r = await t["run_python"]("src/a.py")
    assert "data/processed/x.csv" in r["files_changed"]
    assert "data/processed/x.csv" in touched


async def test_rejects_outside_script(tmp_path):
    env, t = mk(tmp_path)
    assert (await t["run_python"]("../x.py"))["ok"] is False


async def test_project_path_with_spaces(tmp_path):
    env, t = mk(tmp_path, name="my project")
    put(env, "src/a.py", "print('ok')")
    r = await t["run_python"]("src/a.py")
    assert r["exit_code"] == 0


async def test_tail_200_lines(tmp_path):
    env, t = mk(tmp_path)
    put(env, "src/a.py", "for i in range(500): print(i)")
    r = await t["run_python"]("src/a.py")
    lines = r["stdout_tail"].splitlines()
    assert len(lines) == 200 and lines[-1] == "499"


async def test_install_package_validates(monkeypatch, tmp_path):
    env, t = mk(tmp_path)
    assert (await t["install_package"]("--index-url=x"))["ok"] is False
    assert (await t["install_package"]("a; b"))["ok"] is False
    calls = []

    def fake(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=b"done", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake)
    r = await t["install_package"]("pandas")
    assert r["ok"] and calls == [[sys.executable, "-m", "pip", "install", "pandas"]]
