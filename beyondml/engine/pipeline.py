"""The one place a tabular sklearn pipeline is built: the GA scores it and export ships it, so they never drift."""
from __future__ import annotations

import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
    StackingClassifier, StackingRegressor, VotingClassifier, VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

MODEL_NAMES = ("RandomForest", "LogisticRegression", "LinearRegression", "SVM", "DecisionTree", "KNN", "MLP",
               "GradientBoosting")


def make_model(name: str, classification: bool, hp: dict, class_weight: str | None):
    if name == "RandomForest":
        return (RandomForestClassifier(**hp, class_weight=class_weight, n_jobs=1, random_state=42) if classification
                else RandomForestRegressor(**hp, n_jobs=1, random_state=42))
    if name == "LogisticRegression":
        return LogisticRegression(**hp, class_weight=class_weight)
    if name == "SVM":
        return SVC(**hp, class_weight=class_weight) if classification else SVR(**hp)
    if name == "DecisionTree":
        return (DecisionTreeClassifier(**hp, class_weight=class_weight, random_state=42) if classification
                else DecisionTreeRegressor(**hp, random_state=42))
    if name == "KNN":
        return KNeighborsClassifier(**hp, n_jobs=1) if classification else KNeighborsRegressor(**hp, n_jobs=1)
    if name == "GradientBoosting":
        return (GradientBoostingClassifier(**hp, random_state=42) if classification
                else GradientBoostingRegressor(**hp, random_state=42))
    if name == "LinearRegression":
        return LinearRegression()
    if name == "MLP":  # torch loads only when an MLP is actually built
        from .mlp import TorchMLPClassifier, TorchMLPRegressor
        return TorchMLPClassifier(**hp) if classification else TorchMLPRegressor(**hp)
    raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")


def column_kinds(X: pd.DataFrame) -> tuple[list[str], list[bool]]:
    names = X.columns.tolist()
    return names, [bool(pd.api.types.is_numeric_dtype(X[c])) for c in names]


MAX_CATEGORIES = 50  # per categorical column; rarer values share one "infrequent" column (a ticket or zip code
                    # would otherwise one-hot into thousands of columns). export.PREPROCESS ships the same number.


def preprocessor(numeric: list[str], categorical: list[str], dense: bool = False) -> ColumnTransformer:
    """dense=True for clusterers (PCA and silhouette want dense input); supervised models keep sklearn's default."""
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                              OneHotEncoder(handle_unknown="ignore", max_categories=MAX_CATEGORIES)), categorical),
    ], sparse_threshold=0.0 if dense else 0.3)


def build_pipeline(params: dict, names: list[str], is_numeric: list[bool], classification: bool,
                   class_weight: str | None = None) -> Pipeline:
    """Preprocessing over the columns params["features"] selects, then the model. Takes the full feature frame."""
    picked = [i for i, bit in enumerate(params["features"]) if bit]
    numeric = [names[i] for i in picked if is_numeric[i]]
    categorical = [names[i] for i in picked if not is_numeric[i]]
    hp = {k: v for k, v in params.items() if k not in ("model", "features")}
    return make_pipeline(preprocessor(numeric, categorical), make_model(params["model"], classification, hp, class_weight))


def _key(params: dict) -> str:
    return json.dumps(params, sort_keys=True, default=list)


def distinct_top(population: list[dict], top_n: int) -> list[dict]:
    """The first `top_n` distinct params of a best-first [{"params", "fitness"}] population."""
    seen, out = set(), []
    for entry in population:
        key = _key(entry["params"])
        if key not in seen:
            seen.add(key)
            out.append(entry["params"])
            if len(out) == top_n:
                break
    return out


def final_estimator(classification: bool):
    return LogisticRegression(max_iter=1000) if classification else RidgeCV()


def build_ensemble(members: list[dict], names: list[str], is_numeric: list[bool], classification: bool,
                   class_weight: str | None = None, kind: str = "stacking"):
    if kind not in ("stacking", "voting"):
        raise ValueError(f"unknown ensemble kind {kind!r}; expected stacking or voting")
    if len(members) < 2:
        raise ValueError("an ensemble needs at least 2 distinct members")
    estimators = [(f"m{i}", build_pipeline(p, names, is_numeric, classification, class_weight))
                  for i, p in enumerate(members)]
    if kind == "stacking":
        cls = StackingClassifier if classification else StackingRegressor
        return cls(estimators=estimators, final_estimator=final_estimator(classification), cv=5)
    if classification:
        soft = all(hasattr(est, "predict_proba") for _, est in estimators)
        return VotingClassifier(estimators=estimators, voting="soft" if soft else "hard")
    return VotingRegressor(estimators=estimators)
