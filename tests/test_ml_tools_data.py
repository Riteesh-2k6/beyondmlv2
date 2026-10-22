import json

import numpy as np
import pytest

from beyondml.toolkit.env import StepOutcome, ToolEnv
from beyondml.toolkit.ml import ml_tools
from beyondml.workspace.events import EventBus
from beyondml.workspace.project import Project


@pytest.fixture(autouse=True)
def _home(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("BEYONDML_HOME", str(tmp_path_factory.mktemp("bhome")))


async def _no_asker(q):
    raise AssertionError("asker called")


@pytest.fixture
def env(tmp_path, classification_df):
    project = Project.init(tmp_path / "p")
    classification_df.to_csv(project.root / "data" / "raw" / "t.csv", index=False)
    return ToolEnv(project, EventBus(project), "s1", "auto", _no_asker, outcome=StepOutcome())


def tools(env):
    return {f.__name__: f for f in ml_tools(env)}


def events(env, kind):
    lines = env.project.path("agents/s1/events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(x)["data"] for x in lines if json.loads(x)["kind"] == kind]


async def test_profile_tool(env):
    result = tools(env)["profile"]("data/raw/t.csv", target="target")
    assert result["ok"] and result["rows"] == 200 and result["target"] == "target"
    assert result["target_analysis"]["target_type"] == "classification"
    json.dumps(result)  # JSON-friendly all the way down


async def test_scan_leakage_tool(env, classification_df):
    leaky = classification_df.assign(leak=classification_df["target"] * 3 + 1)
    leaky.to_csv(env.project.root / "data" / "raw" / "leaky.csv", index=False)
    result = tools(env)["scan_leakage"]("data/raw/leaky.csv", "target")
    assert result["ok"] and "leak" in [s["column"] for s in result["suspects"]]


async def test_outliers_tool(env):
    t = tools(env)
    assert t["detect_outliers"]("data/raw/t.csv")["ok"]
    assert t["detect_outliers"]("data/raw/t.csv", method="magic")["ok"] is False


async def test_drift_tool(env, classification_df):
    shifted = classification_df.assign(income=classification_df["income"] + 40_000)
    shifted.to_csv(env.project.root / "data" / "raw" / "new.csv", index=False)
    result = tools(env)["drift"]("data/raw/t.csv", "data/raw/new.csv")
    assert result["ok"] and "income" in result["drifted"]


async def test_chart_tool(env):
    result = tools(env)["chart"]("hist", "data/raw/t.csv", "age", title="Age spread")
    assert result["ok"]
    figure = json.loads(env.project.path(result["path"]).read_text(encoding="utf-8"))
    assert "data" in figure and result["path"].startswith("reports/charts/")
    assert events(env, "chart") == [{"title": "Age spread", "path": result["path"]}]


async def test_tools_refuse_escapes(env):
    t = tools(env)
    assert t["profile"]("../x.csv")["ok"] is False
    assert t["chart"]("bar", "data/raw/t.csv", "no_such_column")["ok"] is False


async def test_tools_write_ledger(env):
    t = tools(env)
    t["profile"]("data/raw/t.csv")
    t["scan_leakage"]("data/raw/t.csv", "target")
    t["detect_outliers"]("data/raw/t.csv")
    t["chart"]("hist", "data/raw/t.csv", "age")
    records = env.project.ledger()
    assert [r["tool"] for r in records] == ["profile", "scan_leakage", "detect_outliers", "chart"]
    assert all(r["agent"] == "s1" for r in records)
