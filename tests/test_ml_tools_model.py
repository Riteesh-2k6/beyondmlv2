import builtins
import json
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from beyondml.toolkit.env import StepOutcome, ToolEnv
from beyondml.toolkit.ml import ml_tools
from beyondml.toolkit.python import run_script
from beyondml.workspace.events import EventBus
from beyondml.workspace.project import Project

GENERATION_KEYS = {"gen", "best_fitness", "mean_fitness", "diversity", "evals", "seconds", "best_params"}


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


async def test_evolve_streams_generations(env):
    result = await tools(env)["evolve"]("data/raw/t.csv", "target", pop_size=3, generations=2)
    assert result["ok"], result
    gens = events(env, "generation")
    assert len(gens) >= 2 and all(set(g) == GENERATION_KEYS for g in gens)
    assert env.project.path(f"reports/evolve-{result['run_id']}.json").exists()
    assert env.project.ledger()[-1]["metrics"]["mu_cv"] == result["mu_cv"]


async def test_evolve_string_target_and_nan_rows(env, classification_df):
    df = classification_df.assign(target=np.where(classification_df["target"] == 1, "yes", "no").astype(object))
    df.loc[:4, "target"] = None
    df.to_csv(env.project.root / "data" / "raw" / "s.csv", index=False)
    result = await tools(env)["evolve"]("data/raw/s.csv", "target", pop_size=3, generations=1)
    assert result["ok"] and result["dropped_rows"] == 5


async def test_evolve_bad_seed_is_error(env):
    result = await tools(env)["evolve"]("data/raw/t.csv", "target", pop_size=3, generations=1, seeds=[{"bogus": 1}])
    assert result["ok"] is False and "bogus" in result["error"]


async def test_tune_or_install_hint(env, monkeypatch):
    result = await tools(env)["tune"]("data/raw/t.csv", "target", trials=4)
    assert result["ok"], result
    trials = events(env, "trial")
    assert trials and set(trials[0]) == {"number", "value", "params", "state"}
    real_import = builtins.__import__

    def no_optuna(name, *args, **kwargs):
        if name == "optuna" or name.startswith("optuna."):
            raise ImportError("No module named 'optuna'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_optuna)
    monkeypatch.delitem(__import__("sys").modules, "beyondml.engine.tune", raising=False)
    hint = await tools(env)["tune"]("data/raw/t.csv", "target", trials=2)
    assert hint["ok"] is False and "pip install" in hint["error"]


async def test_export_latest_and_train_runs(env):
    t = tools(env)
    assert (await t["evolve"]("data/raw/t.csv", "target", pop_size=3, generations=1))["ok"]
    exported = t["export_pipeline"]()
    assert exported["ok"] and "src/pipeline.py" in exported["files"]
    trained = run_script(env.project, "src/train.py", [], 300, agent="s1")
    assert trained["ok"], trained["stderr_tail"]


async def test_evaluate_reports_health(env):
    t = tools(env)
    await t["evolve"]("data/raw/t.csv", "target", pop_size=3, generations=1)
    t["export_pipeline"]()
    result = await t["evaluate"]("data/raw/t.csv", "target")
    assert result["ok"] and result["health"] in (0, 1)
    record = env.project.ledger()[-1]
    assert record["tool"] == "evaluate" and record["agent"] == "s1" and "health" in record["metrics"]


async def test_explain_writes_importance(env):
    t = tools(env)
    await t["evolve"]("data/raw/t.csv", "target", pop_size=3, generations=1)
    t["export_pipeline"]()
    result = await t["explain"]("data/raw/t.csv", "target", top_k=3)
    assert result["ok"] and env.project.path("reports/importance.json").exists()
    values = [row["value"] for row in result["top"]]
    assert values == sorted(values, reverse=True)


async def test_ensemble_from_latest(env):
    t = tools(env)
    await t["evolve"]("data/raw/t.csv", "target", pop_size=6, generations=2)
    result = await t["ensemble"](top_n=3)
    assert result["ok"], result
    assert len(result["members"]) >= 2
    assert t["export_pipeline"](run_id=result["run_id"])["ok"]


async def test_cluster_tool(env):
    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.6, random_state=0)
    pd.DataFrame(X, columns=["x0", "x1"]).to_csv(env.project.root / "data" / "raw" / "b.csv", index=False)
    result = await tools(env)["cluster"]("data/raw/b.csv", algorithms=["kmeans"], pop_size=4, generations=2)
    assert result["ok"], result
    assert env.project.path("data/processed/clusters.csv").exists() and events(env, "generation")


async def test_single_row_class_gives_a_clear_error(env):
    # review 10-24: 29 "no" + 1 "yes" made every candidate fail behind a misleading pool-shutdown error
    pd.DataFrame({"x": np.arange(30.0), "y": ["no"] * 29 + ["yes"]}).to_csv(
        env.project.root / "data" / "raw" / "rare.csv", index=False)
    result = await tools(env)["evolve"]("data/raw/rare.csv", "y", pop_size=3, generations=1)
    assert result["ok"] is False and "two classes" in result["error"]


async def test_single_row_class_is_dropped_and_counted(env, classification_df):
    df = classification_df.copy()
    df.loc[0, "target"] = 7  # a third class with one row
    df.to_csv(env.project.root / "data" / "raw" / "three.csv", index=False)
    t = tools(env)
    result = await t["evolve"]("data/raw/three.csv", "target", pop_size=3, generations=1)
    assert result["ok"] and result["dropped_rows"] == 1
    exported = t["export_pipeline"](raw="data/raw/three.csv")
    assert exported["ok"], exported
    for seed in range(3):
        evaluated = await t["evaluate"]("data/raw/three.csv", "target", seed=seed)
        assert evaluated["ok"], evaluated


async def test_evaluate_runs_model_code_out_of_process(env, monkeypatch):
    # review 10-24: evaluate imported the agent-writable pipeline into the harness, secrets and all
    monkeypatch.setenv("BML_PROBE_SECRET_KEY", "hush")
    t = tools(env)
    await t["evolve"]("data/raw/t.csv", "target", pop_size=3, generations=1)
    t["export_pipeline"]()
    probe = ("\nimport json as _j, os as _o\nopen('probe.json', 'w').write(_j.dumps("
             "{'pid': _o.getpid(), 'secret': _o.environ.get('BML_PROBE_SECRET_KEY')}))\n")
    pipeline = env.project.path("src/pipeline.py")
    pipeline.write_text(pipeline.read_text(encoding="utf-8") + probe, encoding="utf-8")
    assert (await t["evaluate"]("data/raw/t.csv", "target"))["ok"]
    seen = json.loads(env.project.path("probe.json").read_text())
    assert seen["pid"] != os.getpid() and seen["secret"] is None
