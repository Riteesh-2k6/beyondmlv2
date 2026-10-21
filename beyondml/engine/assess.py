"""Holdout assessment with the old reflection agent's health rules, and feature importance."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

HEALTH = {"max_gap": 0.10, "max_top_share": 0.5, "max_score": 0.999}
SHAP_ROWS = 100


def _shap_values(estimator, X: pd.DataFrame) -> np.ndarray:
    """Mean |SHAP| per raw column through a model-agnostic explainer (works on whole pipelines)."""
    import shap  # optional extra; ImportError sends the caller to permutation importance

    sample = X.sample(min(SHAP_ROWS, len(X)), random_state=0)
    predict = estimator.predict_proba if hasattr(estimator, "predict_proba") else estimator.predict
    values = shap.Explainer(predict, sample)(sample).values
    if values.ndim == 3:  # (rows, features, classes)
        values = np.abs(values).mean(axis=2)
    return np.abs(values).mean(axis=0)


def importance(estimator, X: pd.DataFrame, y: pd.Series, top_k: int = 10, seed: int = 0) -> dict:
    try:
        values, method = _shap_values(estimator, X), "shap"
    except ImportError:
        values = permutation_importance(estimator, X, y, n_repeats=5, random_state=seed).importances_mean
        method = "permutation"
    values = np.abs(np.asarray(values, dtype=float))
    order = np.argsort(-values)
    total = float(values.sum())
    top = [{"feature": str(X.columns[i]), "value": float(values[i])} for i in order[:top_k]]
    return {"method": method, "top": top, "top_share": float(values[order[0]] / total) if total > 0 else 0.0}


def _split(X: pd.DataFrame, y: pd.Series, classification: bool, test_size: float, seed: int):
    if classification and y.value_counts().min() >= 2:
        try:
            return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
        except ValueError:  # too few rows for every class on both sides
            pass
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def assess(estimator_factory: Callable[[], Any], X: pd.DataFrame, y: pd.Series, classification: bool,
           test_size: float = 0.2, seed: int = 0) -> dict:
    X_train, X_test, y_train, y_test = _split(X, y, classification, test_size, seed)
    model = estimator_factory().fit(X_train, y_train)
    train_score, score = float(model.score(X_train, y_train)), float(model.score(X_test, y_test))
    gap = max(0.0, train_score - score)
    imp = importance(model, X_test, y_test, seed=seed)
    reasons = []
    if gap > HEALTH["max_gap"]:
        reasons.append("gap")
    if imp["top_share"] > HEALTH["max_top_share"]:
        reasons.append("top_share")
    if score >= HEALTH["max_score"]:
        reasons.append("score")
    confusion = []
    if classification:
        labels = sorted(pd.unique(pd.concat([y_train, y_test])), key=str)
        confusion = confusion_matrix(y_test, model.predict(X_test), labels=labels).tolist()
    return {"score_name": "accuracy" if classification else "r2", "score": score, "train_score": train_score,
            "gap": gap, "top_share": imp["top_share"], "confusion": confusion, "importance": imp,
            "health": 0 if reasons else 1, "health_reasons": reasons}
