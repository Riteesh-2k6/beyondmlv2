import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from beyondml.engine.evolve import Budget, Fidelity
from beyondml.engine.export import model_spec, render_pipeline, write_project_code
from beyondml.engine.fitness import TabularFitness, evolve_tabular, warm_seeds
from beyondml.engine.pipeline import column_kinds
from beyondml.engine.profiler import DatasetProfiler

TITANIC = Path(__file__).resolve().parent.parent / "data" / "titanic.csv"
GUARD = ("import sys, runpy; sys.modules['beyondml'] = None; script = sys.argv[1]; sys.argv = sys.argv[1:]; "
         "runpy.run_path(script, run_name='__main__')")


def run(root: Path, script: str, *args: str) -> subprocess.CompletedProcess:
    """Run a shipped script with beyondml made unimportable, from the project root."""
    return subprocess.run([sys.executable, "-c", GUARD, script, *args], cwd=root, capture_output=True, text=True)


def load(path: Path):
    spec = importlib.util.spec_from_file_location(f"pipeline_{abs(hash(path))}", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def project(tmp_path: Path, df: pd.DataFrame, params: dict, target: str = "target") -> tuple[Path, list[str]]:
    root = tmp_path / "proj"
    (root / "data" / "raw").mkdir(parents=True)
    df.to_csv(root / "data" / "raw" / "d.csv", index=False)
    X = df.drop(columns=[target])
    names, kinds = column_kinds(X)
    problem = "classification"
    spec = model_spec(params, names, kinds, problem, None, target, "data/raw/d.csv")
    return root, write_project_code(root, spec)


def equivalence(df, profile, params, tmp_path):
    X, y = df.drop(columns=["target"]), df["target"]
    names, kinds = column_kinds(X)
    spec = model_spec(params, names, kinds, "classification", None, "target", "data/raw/d.csv")
    (tmp_path / "pipeline.py").write_text(render_pipeline(spec), encoding="utf-8")
    module = load(tmp_path / "pipeline.py")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    shipped = float(np.mean(cross_validate(module.build_pipeline(), X, y, cv=cv, scoring="accuracy")["test_score"]))
    scored = TabularFitness(X, y, "classification", profile)(params, Fidelity())[1]["mu_cv"]
    assert abs(shipped - scored) < 1e-9


def test_export_equivalence(classification_df, classification_profile, tmp_path):
    best = evolve_tabular(classification_df, "target", classification_profile, pop_size=4,
                          budget=Budget(max_generations=2), seed=0).best.params
    equivalence(classification_df, classification_profile, best, tmp_path)


def test_export_equivalence_all_categorical(tmp_path):
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"colour": rng.choice(["r", "g", "b"], 120), "size": rng.choice(["s", "m", "l"], 120)})
    df["target"] = (df["colour"] == "r").astype(int)
    profile = DatasetProfiler(df, target_column="target").run()
    params = warm_seeds("classification", 2, models=("LogisticRegression",))[0]
    equivalence(df, profile, params, tmp_path)


def test_train_and_predict_run_without_beyondml(tmp_path):
    df = pd.read_csv(TITANIC)
    root, files = project(tmp_path, df, warm_seeds("classification", df.shape[1] - 1, models=("RandomForest",))[0])
    assert sorted(files) == ["src/pipeline.py", "src/predict.py", "src/train.py"]
    trained = run(root, "src/train.py")
    assert trained.returncode == 0, trained.stderr
    assert (root / "models" / "model.joblib").exists() and "accuracy" in (root / "metrics.json").read_text()
    df.drop(columns=["target"]).head(5).to_csv(root / "new.csv", index=False)
    predicted = run(root, "src/predict.py", "new.csv", "out.csv")
    assert predicted.returncode == 0, predicted.stderr
    assert len(pd.read_csv(root / "out.csv")["prediction"]) == 5


def test_train_applies_clean_and_features(tmp_path):
    df = pd.read_csv(TITANIC)
    feats = df.assign(family_size=df["sibsp"] + df["parch"] + 1)
    root = tmp_path / "proj"
    (root / "data" / "raw").mkdir(parents=True)
    (root / "src").mkdir()
    df.to_csv(root / "data" / "raw" / "d.csv", index=False)
    (root / "src" / "features.py").write_text(
        "def make_features(df):\n    return df.assign(family_size=df['sibsp'] + df['parch'] + 1)\n", encoding="utf-8")
    names, kinds = column_kinds(feats.drop(columns=["target"]))
    params = warm_seeds("classification", len(names), models=("RandomForest",))[0]
    write_project_code(root, model_spec(params, names, kinds, "classification", None, "target", "data/raw/d.csv"))
    assert "family_size" in (root / "src" / "pipeline.py").read_text()
    result = run(root, "src/train.py")
    assert result.returncode == 0, result.stderr


def test_predict_unseen_category_and_missing_column(tmp_path):
    rng = np.random.RandomState(1)
    df = pd.DataFrame({"colour": rng.choice(["r", "g"], 80), "x": rng.normal(size=80)})
    df["target"] = (df["x"] > 0).astype(int)
    root, _ = project(tmp_path, df, warm_seeds("classification", 2, models=("LogisticRegression",))[0])
    assert run(root, "src/train.py").returncode == 0
    pd.DataFrame({"colour": ["purple"], "x": [0.3]}).to_csv(root / "unseen.csv", index=False)
    assert run(root, "src/predict.py", "unseen.csv", "o.csv").returncode == 0
    pd.DataFrame({"colour": ["r"]}).to_csv(root / "short.csv", index=False)
    short = run(root, "src/predict.py", "short.csv", "o2.csv")
    assert short.returncode == 2 and "missing columns: x" in short.stderr


def test_mlp_export_copies_module(tmp_path):
    rng = np.random.RandomState(2)
    df = pd.DataFrame({"a": rng.normal(size=120), "b": rng.normal(size=120)})
    df["target"] = (df["a"] + df["b"] > 0).astype(int)
    params = {"model": "MLP", "hidden": (16,), "lr": 1e-2, "epochs": 5, "dropout": 0.0, "batch_size": 32,
              "features": (1, 1)}
    root, files = project(tmp_path, df, params)
    assert "src/mlp.py" in files and (root / "src" / "mlp.py").exists()
    result = run(root, "src/train.py")
    assert result.returncode == 0, result.stderr


def test_rendered_code_never_imports_beyondml(tmp_path):
    df = pd.read_csv(TITANIC).head(200)
    root, files = project(tmp_path, df, warm_seeds("classification", df.shape[1] - 1, models=("RandomForest",))[0])
    for rel in files:
        text = (root / rel).read_text(encoding="utf-8")
        assert "import beyondml" not in text and "from beyondml" not in text


def test_export_equivalence_high_cardinality(tmp_path):
    # review 10-24: a 400-value column was one-hot encoded without limit; the cap must ship in pipeline.py too
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"ticket": [f"t{i}" for i in rng.randint(0, 400, 600)], "x": rng.normal(size=600)})
    df["target"] = (df["x"] > 0).astype(int)
    from beyondml.engine.pipeline import MAX_CATEGORIES, preprocessor
    assert preprocessor(["x"], ["ticket"]).fit_transform(df).shape[1] <= MAX_CATEGORIES + 1
    profile = DatasetProfiler(df, target_column="target").run()
    equivalence(df, profile, warm_seeds("classification", 2, models=("LogisticRegression",))[0], tmp_path)


def test_train_drops_single_row_classes(tmp_path):
    # review 10-24: a one-row class can be neither stratified nor cross-validated
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"x": rng.normal(size=60)})
    df["target"] = np.where(df["x"] > 0, "a", "b")
    df.loc[0, "target"] = "rare"
    root, _ = project(tmp_path, df, warm_seeds("classification", 1, models=("LogisticRegression",))[0])
    trained = run(root, "src/train.py")
    assert trained.returncode == 0, trained.stderr
    metrics = json.loads((root / "metrics.json").read_text())
    assert metrics["n_train"] + metrics["n_test"] == 59


def test_train_single_class_left_exits_2(tmp_path):
    df = pd.DataFrame({"x": np.arange(30.0), "target": ["a"] * 29 + ["b"]})
    root, _ = project(tmp_path, df, warm_seeds("classification", 1, models=("LogisticRegression",))[0])
    trained = run(root, "src/train.py")
    assert trained.returncode == 2 and "two classes" in trained.stderr


def test_predict_bad_input_exits_2(tmp_path):
    # review 10-24: a column clean/features need, or a non-numeric value, gave a traceback instead of exit 2
    rng = np.random.RandomState(1)
    df = pd.DataFrame({"colour": rng.choice(["r", "g"], 80), "x": rng.normal(size=80)})
    df["target"] = (df["x"] > 0).astype(int)
    root, _ = project(tmp_path, df, warm_seeds("classification", 2, models=("LogisticRegression",))[0])
    (root / "src" / "features.py").write_text("def make_features(df):\n    return df.assign(x2=df['x'] * 2)\n",
                                              encoding="utf-8")
    assert run(root, "src/train.py").returncode == 0
    pd.DataFrame({"colour": ["r"]}).to_csv(root / "no_x.csv", index=False)
    short = run(root, "src/predict.py", "no_x.csv", "o.csv")
    assert short.returncode == 2 and "x" in short.stderr and "Traceback" not in short.stderr
    pd.DataFrame({"colour": ["r"], "x": ["abc"]}).to_csv(root / "bad.csv", index=False)
    bad = run(root, "src/predict.py", "bad.csv", "o2.csv")
    assert bad.returncode == 2 and "abc" in bad.stderr and "Traceback" not in bad.stderr
