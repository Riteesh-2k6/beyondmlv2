import logging
import shutil
from pathlib import Path

import pytest

from beyondml.harness.skills import discover, knowledge_toolset, step_skills
from beyondml.workspace.project import Project, beyondml_home

DATA = Path(__file__).parent / "data" / "skills"


@pytest.fixture(autouse=True)
def _home(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("BEYONDML_HOME", str(tmp_path_factory.mktemp("bhome")))


def test_bundled_python_step_found():
    s = discover(None)["python-step"]
    assert s.kind == "step" and "run_python" in s.allowed_tools and "finish_step" in s.body


def test_project_overrides_bundled(tmp_path):
    p = Project.init(tmp_path / "proj")
    p.write(".beyondml/skills/python-step/SKILL.md",
            "---\nname: python-step\ndescription: custom\n---\nbody\n")
    assert discover(p)["python-step"].description == "custom"


def test_home_level_loaded():
    shutil.copytree(DATA / "leakage-demo", beyondml_home() / "skills" / "leakage-demo")
    s = discover(None)["leakage-demo"]
    assert s.kind == "knowledge" and s.allowed_tools is None


def test_step_fixture_metadata(tmp_path):
    p = Project.init(tmp_path / "proj")
    shutil.copytree(DATA / "tabular-demo", Path(p.root) / ".beyondml/skills/tabular-demo")
    found = discover(p)
    t = step_skills(found)["tabular-demo"]
    assert t.allowed_tools == {"read_file", "run_python", "finish_step"}
    assert (t.modality, t.model_hint) == ("tabular", "small")


def test_bad_skill_skipped(tmp_path, caplog):
    p = Project.init(tmp_path / "proj")
    shutil.copytree(DATA / "BadName", Path(p.root) / ".beyondml/skills/BadName")
    with caplog.at_level(logging.WARNING):
        found = discover(p)
    assert "bad-name" not in found and "BadName" in caplog.text
    assert "python-step" in found


async def test_knowledge_toolset_tools():
    shutil.copytree(DATA / "leakage-demo", beyondml_home() / "skills" / "leakage-demo")
    ts = knowledge_toolset(discover(None))
    assert {t.name for t in await ts.get_tools()} == {"list_skills", "load_skill", "load_skill_resource"}


def test_no_knowledge_gives_none():
    assert knowledge_toolset({}) is None
