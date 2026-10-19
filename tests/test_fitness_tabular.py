"""Tests for the tabular fitness, search space, warm seeds and evolve_tabular (evolution engine v2).

Fitness calls are made directly (no pool), so thread/process state never matters here.
"""

import numpy as np
import pandas as pd
import pytest

import beyondml.engine.fitness as fitness_mod
from beyondml.engine.evolve import Budget, Fidelity
from beyondml.engine.fitness import (
    DEFAULT_MODELS, MODELS, TabularFitness, default_ladder, evolve_tabular, tabular_space, warm_seeds,
)
from beyondml.engine.profiler import DatasetProfiler

METRIC_KEYS = {"mu_cv", "sigma_cv", "gap", "complexity", "n_rows", "folds"}


def _split(df):
    return df.drop(columns=["target"]), df["target"]


def _score_warm_seed(df, profile, problem_type, model):
    X, y = _split(df)
    fit = TabularFitness(X, y, problem_type, profile)
    seed = warm_seeds(problem_type, X.shape[1], models=(model,))[0]
    return fit(seed, Fidelity(1.0, 3))


@pytest.mark.parametrize("model", ["RandomForest", "LogisticRegression", "SVM", "DecisionTree", "KNN", "GradientBoosting"])
def test_every_classifier_scores(model, classification_df, classification_profile):
    score, metrics = _score_warm_seed(classification_df, classification_profile, "classification", model)
    assert score > -1
    assert set(metrics) == METRIC_KEYS


@pytest.mark.parametrize("model", ["RandomForest", "LinearRegression", "SVM", "DecisionTree", "KNN", "GradientBoosting"])
def test_every_regressor_scores(model, regression_df, regression_profile):
    score, metrics = _score_warm_seed(regression_df, regression_profile, "regression", model)
    assert score > -1
    assert set(metrics) == METRIC_KEYS


def test_formula_matches_v1_terms(classification_df, classification_profile):
    score, m = _score_warm_seed(classification_df, classification_profile, "classification", "RandomForest")
    scale = 1 + classification_profile["overfitting_risk_index"]["score"]
    l1, l2, l3 = 0.02 * scale, 0.10 * scale, 0.50 * scale
    assert score == pytest.approx(m["mu_cv"] - l1 * m["complexity"] - l2 * m["sigma_cv"] - l3 * m["gap"], abs=1e-9)
    # v1's C(P): all features on, 100 trees, max_depth None counts as 20
    assert m["complexity"] == pytest.approx(0.4 * 1.0 + 0.6 * ((100 / 250) * 0.5 + (20 / 20) * 0.5))
    assert m["gap"] >= 0 and all(isinstance(v, (int, float)) for v in m.values())


def test_low_fidelity_uses_fewer_rows(classification_df, classification_profile):
    X, y = _split(classification_df)
    fit = TabularFitness(X, y, "classification", classification_profile)
    seed = warm_seeds("classification", X.shape[1], models=("DecisionTree",))[0]
    assert fit(seed, Fidelity(rows=0.5))[1]["n_rows"] == 100
    assert fit(seed, Fidelity(rows=1.0))[1]["n_rows"] == 200


def _binary_profile(imbalance=False):
    return {"target_analysis": {"target_type": "classification", "imbalance_risk": imbalance},
            "overfitting_risk_index": {"score": 0.5}}


def test_rare_class_reduces_folds():
    rng = np.random.RandomState(0)
    y = np.array([0] * 57 + [1] * 3)
    df = pd.DataFrame({"x1": rng.normal(size=60), "x2": rng.normal(size=60), "target": y})
    X, y = _split(df)
    fit = TabularFitness(X, y, "classification", _binary_profile(imbalance=True))
    seed = warm_seeds("classification", 2, models=("RandomForest",))[0]
    assert fit(seed, Fidelity(1.0, 5))[1]["folds"] == 3
    assert fit(seed, Fidelity(1.0, 2))[1]["folds"] == 2   # never more folds than asked for


def test_singleton_class_falls_back_to_plain_kfold():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"x1": rng.normal(size=40), "target": [0] * 39 + [1]})
    X, y = _split(df)
    fit = TabularFitness(X, y, "classification", _binary_profile())
    seed = warm_seeds("classification", 1, models=("DecisionTree",))[0]
    assert fit(seed, Fidelity(1.0, 5))[1]["folds"] == 2   # StratifiedKFold would raise on a 1-row class


def test_single_feature_frame():
    rng = np.random.RandomState(1)
    x = rng.normal(size=80)
    df = pd.DataFrame({"x": x, "target": (x > 0).astype(int)})
    profile = DatasetProfiler(df, target_column="target").run()
    r = evolve_tabular(df, "target", profile, pop_size=4, budget=Budget(max_generations=2), workers=1)
    assert r.best.params["features"] == (1,)


def test_all_categorical_frame():
    rng = np.random.RandomState(2)
    n = 90
    a = rng.choice(["u", "v", "w"], n)
    df = pd.DataFrame({"a": a, "b": rng.choice(["p", "q"], n), "target": (a == "u").astype(int)})
    profile = DatasetProfiler(df, target_column="target").run()
    r = evolve_tabular(df, "target", profile, pop_size=4, budget=Budget(max_generations=2), workers=1)
    assert r.best.fitness > -1


def test_warm_seed_values():
    seeds = {s["model"]: s for s in warm_seeds("classification", 4)}
    assert set(seeds) == {"RandomForest", "LogisticRegression"}
    rf, lr = seeds["RandomForest"], seeds["LogisticRegression"]
    assert rf["n_estimators"] == 100 and rf["max_depth"] is None
    assert lr["C"] == 1.0 and lr["max_iter"] == 1000
    assert rf["features"] == (1, 1, 1, 1)


def test_warm_seed_evaluated_first_gen(classification_df, classification_profile):
    r = evolve_tabular(classification_df, "target", classification_profile, models=("RandomForest",),
                       pop_size=1, budget=Budget(max_generations=1), workers=1)
    assert r.best.params["n_estimators"] == 100


def test_executor_by_rows(monkeypatch):
    seen = []
    monkeypatch.setattr(fitness_mod, "evolve", lambda *a, **kw: seen.append(kw))
    profile = _binary_profile()
    for n in (200, 20_000):
        df = pd.DataFrame({"x": np.arange(n, dtype=float), "target": np.arange(n) % 2})
        evolve_tabular(df, "target", profile)
    assert [kw["executor"] for kw in seen] == ["threads", "processes"]
    assert seen[0]["fidelity"] == default_ladder(200) and seen[1]["fidelity"] == default_ladder(20_000)


def test_space_rejects_wrong_problem_type():
    with pytest.raises(ValueError):
        tabular_space("regression", 3, models=("LogisticRegression",))


def test_models_filter_keeps_given_order_and_fitting_models():
    assert list(tabular_space("regression", 3, models=("KNN", "LogisticRegression", "RandomForest"))["model"].options) \
        == ["KNN", "RandomForest"]
    assert [s["model"] for s in warm_seeds("classification", 2, models=("SVM", "LinearRegression", "KNN"))] \
        == ["SVM", "KNN"]
    assert list(tabular_space("classification", 2)["model"].options) == list(DEFAULT_MODELS["classification"])
    assert set(DEFAULT_MODELS["regression"]) <= set(MODELS)
    for bad in (dict(problem_type="clustering", models=None), dict(problem_type="classification", models=("Nope",))):
        with pytest.raises(ValueError):
            tabular_space(bad["problem_type"], 3, models=bad["models"])
        with pytest.raises(ValueError):
            warm_seeds(bad["problem_type"], 3, models=bad["models"])


def test_default_ladder():
    assert default_ladder(4_999) == (Fidelity(1.0, 3), Fidelity(1.0, 5))
    assert default_ladder(5_000) == (Fidelity(0.25, 3), Fidelity(1.0, 5))


def test_fitness_lets_errors_propagate(classification_df, classification_profile):
    X, y = _split(classification_df)
    fit = TabularFitness(X, y, "classification", classification_profile)
    with pytest.raises(ValueError):
        fit({"model": "NoSuchModel", "features": (1,) * X.shape[1]}, Fidelity(1.0, 3))
    with pytest.raises(ValueError):   # a model that cannot fit this problem type must not score -1.0 quietly
        fit({"model": "LinearRegression", "features": (1,) * X.shape[1]}, Fidelity(1.0, 3))
