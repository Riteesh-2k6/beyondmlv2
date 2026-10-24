import pytest

from beyondml.harness.skills import discover
from beyondml.toolkit.control import control_tools
from beyondml.toolkit.env import StepOutcome, ToolEnv
from beyondml.toolkit.files import file_tools
from beyondml.toolkit.ml import ml_tools
from beyondml.toolkit.python import python_tools
from beyondml.workspace.events import EventBus
from beyondml.workspace.project import Project

STEP = {"tabular-audit", "tabular-clean", "tabular-eda", "tabular-features", "evolve-search", "tabular-ensemble",
        "tabular-export", "evaluate-explain", "tabular-cluster", "drift-check"}
KNOWLEDGE = {"leakage-checklist", "choose-metric", "class-imbalance", "train-test-split"}


@pytest.fixture(autouse=True)
def _home(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("BEYONDML_HOME", str(tmp_path_factory.mktemp("bhome")))


@pytest.fixture(scope="module")
def skills():
    return discover(None)


def test_all_bundled_skills_load(skills):
    assert STEP | KNOWLEDGE <= set(skills)
    assert all(skills[n].kind == "step" and skills[n].modality == "tabular" for n in STEP)
    assert all(skills[n].kind == "knowledge" for n in KNOWLEDGE)


def test_allowed_tools_exist(skills, tmp_path):
    project = Project.init(tmp_path)
    env = ToolEnv(project, EventBus(project), "s", "auto", None, outcome=StepOutcome())
    known = {f.__name__ for f in file_tools(env) + python_tools(env) + control_tools(env) + ml_tools(env)}
    for name in STEP:
        assert skills[name].allowed_tools and skills[name].allowed_tools <= known, name


def test_checkpoints_present(skills):
    for name in ("tabular-audit", "tabular-clean", "tabular-features"):
        assert "ask_user(" in skills[name].body and "recommended" in skills[name].body, name


def test_descriptions_state_order(skills):
    for name in STEP:
        assert "Usually" in skills[name].description or "Only when" in skills[name].description, name


def test_descriptions_name_the_standard_check(skills):
    # the planner sees only descriptions; qwen3:8b invented outputs (models/best_model.pkl, a root train.py) and an
    # export metric train.py never writes (f1) until each step skill named its standard check (acceptance 10-22)
    for name in STEP:
        assert "Check:" in skills[name].description, name
    export = skills["tabular-export"].description
    assert "src/train.py" in export and " train.py" not in export and "accuracy" in export and "r2" in export


def test_write_steps_name_write_file(skills):
    # qwen3:8b passed the report to finish_step as `content` instead of writing it (acceptance 10-24)
    for name in STEP:
        for line in skills[name].body.splitlines():
            if line.split(". ", 1)[-1].startswith("Write "):
                assert "write_file" in line, (name, line)
