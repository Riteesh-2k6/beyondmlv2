import subprocess
import sys
from pathlib import Path

import pytest

from beyondml.harness.run import run_goal
from beyondml.workspace.plan import Plan, load_plan, save_plan
from beyondml.workspace.project import Project
from scripted_llm import fc, scripted_resolver, txt

TITANIC = Path(__file__).resolve().parent.parent / "data" / "titanic.csv"

PREPARE_PY = """import os
import pandas as pd
df = pd.read_csv("data/raw/titanic.csv")
df = df.fillna(df.median(numeric_only=True))
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/clean.csv", index=False)
print("rows", len(df))
"""

TRAIN_PY = """import json, os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
df = pd.read_csv("data/processed/clean.csv")
X = df.select_dtypes("number").drop(columns=["target"])
X_train, X_test, y_train, y_test = train_test_split(X, df["target"], test_size=0.2, random_state=0)
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
acc = float(model.score(X_test, y_test))
with open(os.environ.get("BEYONDML_METRICS", "metrics.json"), "w") as f:
    json.dump({"accuracy": acc}, f)
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")
print("accuracy", acc)
"""

PLAN = {"goal": "predict survival", "metric": "accuracy", "budget": {"minutes": 10}, "steps": [
    {"id": "prepare", "skill": "python-step", "goal": "clean the data",
     "success_check": {"expects": ["data/processed/clean.csv"]}},
    {"id": "train", "skill": "python-step", "goal": "train a model", "depends_on": ["prepare"],
     "success_check": {"run": "src/train.py", "expects": ["models/model.joblib"], "metric": "accuracy", "min": 0.6}},
]}


def worker(id, code, outputs):
    return [[fc("write_file", path=f"src/{id}.py", content=code)], [fc("run_python", script=f"src/{id}.py")],
            [fc("finish_step", summary=f"{id} done", outputs=outputs)], [txt("finished")]]


@pytest.fixture(autouse=True)
def _home(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("BEYONDML_HOME", str(tmp_path_factory.mktemp("bhome")))


async def ask(q):
    return "ok"


async def approve(payload):
    return True


async def test_e2e_titanic_scripted(tmp_path):
    proj = tmp_path / "proj"
    resolver = scripted_resolver({
        "planner": [[[fc("submit_plan", plan=PLAN)], [txt("planned")]]],
        "prepare": [worker("prepare", PREPARE_PY, ["data/processed/clean.csv"])],
        "train": [worker("train", TRAIN_PY, ["models/model.joblib"])],
    })
    plan = await run_goal(proj, "predict survival", [TITANIC], mode="autonomous", resolver=resolver,
                          asker=ask, approver=approve)
    assert plan.is_finished() and [s.status for s in plan.steps] == ["done", "done"]
    assert subprocess.run([sys.executable, "src/train.py"], cwd=proj, capture_output=True).returncode == 0
    project = Project.open(proj)
    assert len([r for r in project.ledger() if r["agent"] in ("prepare", "train")]) >= 2
    assert (proj / "agents/train/instruction.md").exists()
    for agent in ("intake", "planner", "prepare", "train"):
        assert (proj / f"agents/{agent}/events.jsonl").exists(), agent
    assert "titanic.csv" in (proj / "DATA.md").read_text(encoding="utf-8")


async def test_run_goal_resumes_existing_plan(tmp_path):
    project = Project.init(tmp_path, goal="g")
    plan = Plan.from_dict({"goal": "g", "metric": "f1", "budget": {}, "steps": [
        {"id": "a", "skill": "python-step", "goal": "a"},
        {"id": "b", "skill": "python-step", "goal": "b", "depends_on": ["a"]}]})
    plan.step("a").status = "done"
    save_plan(project, plan)
    resolver = scripted_resolver({"b": [[[fc("finish_step", summary="b done", outputs=[])], [txt(".")]]]})
    result = await run_goal(tmp_path, None, mode="autonomous", resolver=resolver, asker=ask, approver=approve)
    assert result.step("b").status == "done"


async def test_plan_rejected_runs_nothing(tmp_path):
    asked = []

    async def reject(payload):
        asked.append(payload["kind"])
        return False

    resolver = scripted_resolver({"planner": [[[fc("submit_plan", plan=PLAN)], [txt("planned")]]]})
    plan = await run_goal(tmp_path / "proj", "predict survival", [TITANIC], mode="auto", resolver=resolver,
                          asker=ask, approver=reject)
    assert asked == ["plan"] and [s.status for s in plan.steps] == ["pending", "pending"]
    assert Project.open(tmp_path / "proj").ledger() == [] and load_plan(Project.open(tmp_path / "proj"))


async def reject(payload):
    return False


async def test_missing_data_path_reported(tmp_path):
    resolver = scripted_resolver({"planner": [[[fc("submit_plan", plan=PLAN)], [txt("planned")]]]})
    await run_goal(tmp_path / "proj", "predict survival", [TITANIC, tmp_path / "nope.csv"], mode="auto",
                   resolver=resolver, asker=ask, approver=reject)
    text = (tmp_path / "proj/DATA.md").read_text(encoding="utf-8")
    assert "nope.csv" in text and "unusable" in text and "titanic.csv" in text


async def test_new_goal_updates_project_md(tmp_path):
    project = Project.init(tmp_path, goal="old goal")
    done = Plan.from_dict({"goal": "old goal", "metric": "f1", "budget": {},
                           "steps": [{"id": "a", "skill": "python-step", "goal": "a"}]})
    done.step("a").status = "done"
    save_plan(project, done)
    resolver = scripted_resolver({"planner": [[[fc("submit_plan", plan=PLAN)], [txt("planned")]]]})
    await run_goal(tmp_path, "new goal", mode="auto", resolver=resolver, asker=ask, approver=reject)
    text = project.read("PROJECT.md")
    assert "new goal" in text and "old goal" not in text
