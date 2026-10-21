import importlib.util
import sys

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_score

from beyondml.engine.export import ensemble_spec, render_pipeline
from beyondml.engine.fitness import warm_seeds
from beyondml.engine.pipeline import build_ensemble, column_kinds, distinct_top


def _members(n_features):
    rf, lr = warm_seeds("classification", n_features, models=("RandomForest", "LogisticRegression"))
    deeper = dict(rf, max_depth=5, n_estimators=50)
    return [rf, lr, deeper]


def test_distinct_top_dedupes():
    a = {"model": "LogisticRegression", "C": 1.0, "features": (1, 1)}
    b = {"model": "LogisticRegression", "C": 2.0, "features": (1, 1)}
    c = {"model": "LogisticRegression", "C": 1.0, "features": (1, 0)}
    d = {"model": "KNN", "n_neighbors": 5, "features": (1, 1)}
    population = [{"params": p, "fitness": f} for p, f in [(a, 0.9), (dict(a), 0.9), (b, 0.8), (c, 0.7), (d, 0.6)]]
    assert distinct_top(population, 10) == [a, b, c, d]
    assert distinct_top(population, 2) == [a, b]


def test_stacking_builds_and_scores(classification_df):
    X, y = classification_df.drop(columns=["target"]), classification_df["target"]
    names, kinds = column_kinds(X)
    model = build_ensemble(_members(len(names)), names, kinds, True)
    score = cross_val_score(model, X, y, cv=3).mean()
    assert np.isfinite(score) and 0 <= score <= 1


def test_ensemble_rejects_one_member(classification_df):
    names, kinds = column_kinds(classification_df.drop(columns=["target"]))
    with pytest.raises(ValueError):
        build_ensemble(_members(len(names))[:1], names, kinds, True)
    with pytest.raises(ValueError):
        build_ensemble(_members(len(names)), names, kinds, True, kind="bagging")


@pytest.mark.parametrize("kind", ["stacking", "voting"])
def test_ensemble_export_equivalence(tmp_path, classification_df, kind):
    X, y = classification_df.drop(columns=["target"]), classification_df["target"]
    names, kinds = column_kinds(X)
    members = _members(len(names))
    spec = ensemble_spec(members, names, kinds, "classification", None, "target", "data/raw/d.csv", kind)
    path = tmp_path / "pipeline.py"
    path.write_text(render_pipeline(spec), encoding="utf-8")
    module_spec = importlib.util.spec_from_file_location(f"ens_{kind}", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    shipped = cross_val_score(module.build_pipeline(), X, y, cv=cv).mean()
    built = cross_val_score(build_ensemble(members, names, kinds, True, kind=kind), X, y, cv=cv).mean()
    assert abs(shipped - built) < 1e-9
