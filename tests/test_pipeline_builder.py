import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from beyondml.engine.evolve import Fidelity
from beyondml.engine.fitness import TabularFitness, warm_seeds
from beyondml.engine.pipeline import build_pipeline, column_kinds


def _split(df):
    return df.drop(columns=["target"]), df["target"]


def test_build_pipeline_matches_fitness_cv(classification_df, classification_profile):
    X, y = _split(classification_df)
    names, kinds = column_kinds(X)
    params = warm_seeds("classification", X.shape[1], models=("RandomForest",))[0]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    ours = cross_validate(build_pipeline(params, names, kinds, True), X, y, cv=cv, scoring="accuracy")
    fitness = TabularFitness(X, y, "classification", classification_profile)(params, Fidelity())[1]
    assert abs(float(np.mean(ours["test_score"])) - fitness["mu_cv"]) < 1e-9


def test_build_pipeline_drops_unselected_columns():
    X = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": ["x", "y", "x", "y"], "c": [0.5, 0.1, 0.2, 0.9]})
    y = pd.Series([0, 1, 0, 1])
    names, kinds = column_kinds(X)
    params = {"model": "LogisticRegression", "C": 1.0, "max_iter": 1000, "features": (1, 0, 1)}
    pipe = build_pipeline(params, names, kinds, True).fit(X, y)
    used = [cols for name, _, cols in pipe[0].transformers_ if name != "remainder"]
    assert sorted(c for cols in used for c in cols) == ["a", "c"]


def test_column_kinds():
    X = pd.DataFrame({"n": [1, 2], "s": ["a", "b"], "f": [0.1, 0.2], "b": [True, False]})
    names, kinds = column_kinds(X)
    assert names == ["n", "s", "f", "b"] and kinds == [True, False, True, True]
