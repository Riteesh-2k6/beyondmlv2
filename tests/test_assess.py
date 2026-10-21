import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from beyondml.engine.assess import assess, importance


def _frame(n=400, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({k: rng.normal(size=n) for k in "abcd"})
    return X, rng


def test_assess_healthy():
    X, rng = _frame()
    y = (X.sum(axis=1) + rng.normal(scale=0.5, size=len(X)) > 0).astype(int)   # signal spread over four features
    result = assess(lambda: LogisticRegression(max_iter=1000), X, y, classification=True)
    assert result["health"] == 1 and result["health_reasons"] == []
    assert result["score_name"] == "accuracy" and len(result["confusion"]) == 2


def test_assess_flags_overfit():
    X, rng = _frame()
    y = pd.Series(rng.randint(0, 2, len(X)))
    result = assess(lambda: DecisionTreeClassifier(random_state=0), X, y, classification=True)
    assert result["gap"] > 0.10 and result["health"] == 0 and "gap" in result["health_reasons"]


def test_assess_flags_dominance():
    X, rng = _frame()
    y = (X["a"] > 0).astype(int)
    X = X.assign(a=X["a"] + rng.normal(scale=0.3, size=len(X)))
    result = assess(lambda: LogisticRegression(max_iter=1000), X, y, classification=True)
    assert result["top_share"] > 0.5 and "top_share" in result["health_reasons"]


def test_assess_flags_perfect_score():
    X, rng = _frame()
    y = (X["b"] > 0).astype(int)
    X = X.assign(leak=y)
    result = assess(lambda: RandomForestClassifier(random_state=0), X, y, classification=True)
    assert result["score"] >= 0.999 and "score" in result["health_reasons"]


def test_evaluate_tiny_imbalanced():
    X, rng = _frame(n=30)
    y = pd.Series([0] * 29 + [1])
    result = assess(lambda: LogisticRegression(max_iter=1000), X, y, classification=True)
    assert result["score_name"] == "accuracy" and 0.0 <= result["score"] <= 1.0


def test_importance_permutation_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "shap", None)   # import shap now raises ImportError
    X, rng = _frame()
    y = (2 * X["a"] + X["b"] > 0).astype(int)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    result = importance(model, X, y, top_k=3)
    values = [t["value"] for t in result["top"]]
    assert result["method"] == "permutation" and values == sorted(values, reverse=True) and len(values) == 3
    assert result["top"][0]["feature"] == "a" and 0 < result["top_share"] <= 1
