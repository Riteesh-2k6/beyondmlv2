"""The one place a tabular sklearn pipeline is built: the GA scores it and export ships it, so they never drift."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

MODEL_NAMES = ("RandomForest", "LogisticRegression", "LinearRegression", "SVM", "DecisionTree", "KNN",
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
    raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")


def column_kinds(X: pd.DataFrame) -> tuple[list[str], list[bool]]:
    names = X.columns.tolist()
    return names, [bool(pd.api.types.is_numeric_dtype(X[c])) for c in names]


def preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                              OneHotEncoder(handle_unknown="ignore")), categorical),
    ])


def build_pipeline(params: dict, names: list[str], is_numeric: list[bool], classification: bool,
                   class_weight: str | None = None) -> Pipeline:
    """Preprocessing over the columns params["features"] selects, then the model. Takes the full feature frame."""
    picked = [i for i, bit in enumerate(params["features"]) if bit]
    numeric = [names[i] for i in picked if is_numeric[i]]
    categorical = [names[i] for i in picked if not is_numeric[i]]
    hp = {k: v for k, v in params.items() if k not in ("model", "features")}
    return make_pipeline(preprocessor(numeric, categorical), make_model(params["model"], classification, hp, class_weight))
