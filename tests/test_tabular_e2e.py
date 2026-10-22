"""Scripted end-to-end runs of the tabular plan shapes on the real tools (spec §6)."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from beyondml.harness.run import run_goal
from scripted_llm import fc, last_response, scripted_resolver, txt

ROOT = Path(__file__).resolve().parent.parent
TITANIC = ROOT / "data" / "titanic.csv"
BASELINE = json.loads((ROOT / "tests" / "data" / "titanic_v1_baseline.json").read_text())["holdout_accuracy"]
BAR = BASELINE - 0.01  # owner decision 2026-10-22: the v1-vs-v2 tolerance approved for Plan A
GUARD = ("import sys, runpy; sys.modules['beyondml'] = None; script = sys.argv[1]; sys.argv = sys.argv[1:]; "
         "runpy.run_path(script, run_name='__main__')")

CLEAN_PY = '''import os

import pandas as pd


def clean(df):
    df = df.copy()
    for column in df.select_dtypes("number").columns:
        df[column] = df[column].fillna(df[column].median())
    return df


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    raw = [f for f in os.listdir("data/raw") if f.endswith(".csv")][0]
    clean(pd.read_csv(os.path.join("data/raw", raw))).to_csv("data/processed/clean.csv", index=False)
'''

FEATURES_PY = '''import pandas as pd


def make_features(df):
    return df.assign(family_size=df["sibsp"] + df["parch"] + 1)


if __name__ == "__main__":
    make_features(pd.read_csv("data/processed/clean.csv")).to_csv("data/processed/features.csv", index=False)
'''


def step(id, skill, deps=(), **check):
    return {"id": id, "skill": skill, "goal": f"{id} the data", "depends_on": list(deps), "success_check": check}


def finish(summary, outputs):
    return [[fc("finish_step", summary=summary, outputs=outputs)], [txt("done")]]


def write_and_run(path, code, summary, outputs):
    return [[fc("write_file", path=path, content=code)], [fc("run_python", script=path)]] + finish(summary, outputs)


def search_then_finish(req):
    run_id = last_response(req).get("run_id", "")
    return [fc("finish_step", summary=f"run_id={run_id}", outputs=[])]


async def ask(q):
    return "ok"


async def approve(payload):
    return True


def guarded(root, *args):
    return subprocess.run([sys.executable, "-c", GUARD, *args], cwd=root, capture_output=True, text=True)


@pytest.fixture(autouse=True)
def _home(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("BEYONDML_HOME", str(tmp_path_factory.mktemp("bhome")))


async def test_titanic_full_shape(tmp_path):
    plan = {"goal": "predict survival", "metric": "accuracy", "budget": {}, "steps": [
        step("audit", "tabular-audit", expects=["reports/audit.md"]),
        step("clean", "tabular-clean", ["audit"], run="src/clean.py", expects=["data/processed/clean.csv"]),
        step("eda", "tabular-eda", ["clean"], expects=["reports/eda.md"]),
        step("features", "tabular-features", ["eda"], run="src/features.py",
             expects=["data/processed/features.csv"]),
        step("search", "evolve-search", ["features"], metric="mu_cv", min=0.6),
        step("export", "tabular-export", ["search"], run="src/train.py", expects=["models/model.joblib"],
             metric="accuracy", min=BAR),
        step("evaluate", "evaluate-explain", ["export"], metric="health", min=1),
    ]}
    features = "data/processed/features.csv"
    scripts = {
        "planner": [[[fc("submit_plan", plan=plan)], [txt("planned")]]],
        "audit": [[[fc("profile", path="data/raw/titanic.csv")],
                   [fc("ask_user", question="Which column is the target?", options=["target"], recommended="target")],
                   [fc("scan_leakage", path="data/raw/titanic.csv", target="target")],
                   [fc("write_file", path="reports/audit.md", content="target: target\nproblem: classification\n")]]
                  + finish("target=target, classification, drop none", ["reports/audit.md"])],
        "clean": [write_and_run("src/clean.py", CLEAN_PY, "median imputation", ["data/processed/clean.csv"])],
        "eda": [[[fc("chart", kind="hist", data="data/processed/clean.csv", x="age")],
                 [fc("write_file", path="reports/eda.md", content="- class and gender separate survival\n")]]
                + finish("eda written", ["reports/eda.md"])],
        "features": [write_and_run("src/features.py", FEATURES_PY, "family_size added", [features])],
        "search": [[[fc("evolve", data=features, target="target", pop_size=10, generations=5, seed=0)],
                    search_then_finish, [txt("done")]]],
        "export": [[[fc("export_pipeline")], [fc("run_python", script="src/train.py")]]
                   + finish("trained", ["src/pipeline.py", "src/train.py", "models/model.joblib"])],
        "evaluate": [[[fc("explain", data=features, target="target")], [fc("evaluate", data=features, target="target")],
                      [fc("write_file", path="reports/evaluation.md", content="evaluated\n")]]
                     + finish("evaluated", ["reports/evaluation.md"])],
    }
    proj = tmp_path / "proj"
    result = await run_goal(proj, "predict survival", [TITANIC], mode="autonomous", resolver=scripted_resolver(scripts),
                            asker=ask, approver=approve)
    statuses = {s.id: (s.status, s.last_failure) for s in result.steps}
    assert all(status == "done" for status, _ in statuses.values()), statuses
    trained = guarded(proj, "src/train.py")
    assert trained.returncode == 0, trained.stderr
    assert json.loads((proj / "metrics.json").read_text())["accuracy"] >= BAR
    pd.read_csv(TITANIC).drop(columns=["target"]).head(5).to_csv(proj / "new.csv", index=False)
    predicted = guarded(proj, "src/predict.py", "new.csv", "out.csv")
    assert predicted.returncode == 0, predicted.stderr
    assert len(pd.read_csv(proj / "out.csv")["prediction"]) == 5
    generations = [json.loads(line) for line in (proj / "agents/search/events.jsonl").read_text().splitlines()]
    assert sum(e["kind"] == "generation" for e in generations) >= 5


async def test_no_target_cluster_shape(tmp_path):
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=0)
    data = tmp_path / "blobs.csv"
    pd.DataFrame(X, columns=["x0", "x1"]).to_csv(data, index=False)
    plan = {"goal": "group the rows", "metric": "silhouette", "budget": {}, "steps": [
        step("audit", "tabular-audit", expects=["reports/audit.md"]),
        step("clean", "tabular-clean", ["audit"], run="src/clean.py", expects=["data/processed/clean.csv"]),
        step("cluster", "tabular-cluster", ["clean"], expects=["data/processed/clusters.csv"]),
        step("export", "tabular-export", ["cluster"], run="src/train.py", metric="silhouette", min=0.3),
    ]}
    scripts = {
        "planner": [[[fc("submit_plan", plan=plan)], [txt("planned")]]],
        "audit": [[[fc("write_file", path="reports/audit.md", content="no target: clustering\n")]]
                  + finish("no target", ["reports/audit.md"])],
        "clean": [write_and_run("src/clean.py", CLEAN_PY, "median imputation", ["data/processed/clean.csv"])],
        "cluster": [[[fc("cluster", data="data/processed/clean.csv", algorithms=["kmeans"], pop_size=4,
                         generations=2)]] + finish("clustered", ["data/processed/clusters.csv"])],
        "export": [[[fc("export_pipeline")], [fc("run_python", script="src/train.py")]]
                   + finish("exported", ["src/pipeline.py", "src/train.py"])],
    }
    result = await run_goal(tmp_path / "proj", "group the rows", [data], mode="autonomous",
                            resolver=scripted_resolver(scripts), asker=ask, approver=approve)
    statuses = {s.id: (s.status, s.last_failure) for s in result.steps}
    assert all(status == "done" for status, _ in statuses.values()), statuses
    assert (tmp_path / "proj" / "data" / "processed" / "clusters.csv").exists()
