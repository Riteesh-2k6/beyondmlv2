"""Holdout assessment with the old reflection agent's health rules, and feature importance."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
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


def usable_rows(df: pd.DataFrame, target: str, classification: bool) -> tuple[pd.DataFrame, int]:
    """Drop rows with no target and, for classification, rows of a class too rare to split (one row).
    Returns the kept rows and how many were dropped; raises when fewer than two classes are left."""
    keep = df[target].notna()
    if classification:
        counts = df.loc[keep, target].value_counts()
        keep &= df[target].isin(counts[counts >= 2].index)
        if df.loc[keep, target].nunique() < 2:
            raise ValueError(f"target {target!r} needs at least two classes with 2+ rows each; "
                             f"class counts are {counts.to_dict()}")
    kept = df[keep].reset_index(drop=True)
    return kept, len(df) - len(kept)


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
    # Owner decision 2026-10-22: only a near-perfect score (likely leakage) fails health. A large train/test gap or
    # one dominant feature are warnings to explain: a strong legitimate feature (titanic's gender) must not be cut.
    reasons = ["score"] if score >= HEALTH["max_score"] else []
    warnings = []
    if gap > HEALTH["max_gap"]:
        warnings.append("gap")
    if imp["top_share"] > HEALTH["max_top_share"]:
        warnings.append("top_share")
    confusion = []
    if classification:
        labels = sorted(pd.unique(pd.concat([y_train, y_test])), key=str)
        confusion = confusion_matrix(y_test, model.predict(X_test), labels=labels).tolist()
    return {"score_name": "accuracy" if classification else "r2", "score": score, "train_score": train_score,
            "gap": gap, "top_share": imp["top_share"], "confusion": confusion, "importance": imp,
            "health": 0 if reasons else 1, "health_reasons": reasons, "warnings": warnings}


def _load_pipeline(path: Path):
    spec = importlib.util.spec_from_file_location("project_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))  # src/mlp.py sits next to pipeline.py
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    """Run a project's model code outside the harness process (evaluate and explain call this through run_module):
    python -m beyondml.engine.assess {evaluate|explain} PIPELINE DATA TARGET OUT [--test-size F] [--seed N] [--top-k N]"""
    parser = argparse.ArgumentParser(prog="beyondml.engine.assess")
    parser.add_argument("kind", choices=["evaluate", "explain"])
    for name in ("pipeline", "data", "target", "out"):
        parser.add_argument(name)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)
    module = _load_pipeline(Path(args.pipeline).resolve())
    data = Path(args.data)
    df = pd.read_parquet(data) if data.suffix.lower() == ".parquet" else pd.read_csv(data)
    classification = module.PROBLEM == "classification"
    df, dropped = usable_rows(df, args.target, classification)
    X, y = df[module.FEATURES], df[args.target]
    if args.kind == "evaluate":
        result = assess(module.build_pipeline, X, y, classification, args.test_size, args.seed)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        result = importance(module.build_pipeline().fit(X_train, y_train), X_test, y_test, args.top_k)
    result["dropped_rows"] = dropped
    Path(args.out).write_text(json.dumps(result, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                              encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
