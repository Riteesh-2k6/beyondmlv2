"""Evolution engine v2, tabular side: sklearn fitness, search space, warm seeds and evolve_tabular().

The fitness is v1's penalised CV score (engine/genetic.py `_evaluate_genome_worker`):
    mu_cv - l1 * C(P) - l2 * sigma_cv - l3 * gap,  (l1, l2, l3) = (0.02, 0.10, 0.50) * (1 + ORI)
It never swallows an exception: evolve() turns a raising candidate into -inf with the error in its metrics,
and raises EvolveError when every first-generation candidate fails.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split

from .evolve import Budget, Choice, EvolveResult, Fidelity, Float, Mask, Space, evolve
from .pipeline import build_pipeline

_BOTH = ("classification", "regression")

# name -> (problem types it serves, v1's random-init sub-space, v1's warm-start hyperparameters);
# Choice((v,)) is a fixed value
_MODEL_TABLE: dict[str, tuple[tuple[str, ...], dict, dict]] = {
    "RandomForest": (_BOTH, {
        "n_estimators": Choice((50, 100, 150, 200, 250)),
        "max_depth": Choice((None, 5, 10, 15, 20, 30)),
        "min_samples_split": Choice((2, 5, 10)),
        "min_samples_leaf": Choice((1, 2, 4)),
        "max_features": Choice(("sqrt", "log2", 0.5, 0.7)),
    }, {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1,
        "max_features": "sqrt"}),
    "LogisticRegression": (("classification",), {"C": Float(0.01, 10.0), "max_iter": Choice((1000,))},
                           {"C": 1.0, "max_iter": 1000}),
    "LinearRegression": (("regression",), {}, {}),
    "SVM": (_BOTH, {"C": Float(0.1, 10.0), "kernel": Choice(("rbf", "linear", "poly"))},
            {"C": 1.0, "kernel": "rbf"}),
    "DecisionTree": (_BOTH, {
        "max_depth": Choice((None, 5, 10, 15, 20, 30)),
        "min_samples_split": Choice((2, 5, 10)),
    }, {"max_depth": None, "min_samples_split": 2}),
    "KNN": (_BOTH, {"n_neighbors": Choice((3, 5, 7, 9, 11, 15))}, {"n_neighbors": 5}),
    "GradientBoosting": (_BOTH, {
        "n_estimators": Choice((50, 100, 150, 200, 300)),
        "learning_rate": Choice((0.01, 0.05, 0.1, 0.2)),
        "max_depth": Choice((3, 5, 7, 9)),
    }, {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}),
    "MLP": (_BOTH, {   # torch, engine/mlp.py; opt-in only (not in DEFAULT_MODELS)
        "hidden": Choice(((32,), (64,), (64, 32), (128, 64))),
        "lr": Float(1e-4, 1e-2, log=True),
        "epochs": Choice((20, 40, 80)),
        "dropout": Choice((0.0, 0.1, 0.3)),
        "batch_size": Choice((32, 64, 128)),
    }, {"hidden": (64, 32), "lr": 1e-3, "epochs": 40, "dropout": 0.1, "batch_size": 64}),
}

MODELS = tuple(_MODEL_TABLE)
DEFAULT_MODELS = {"classification": ("RandomForest", "LogisticRegression"),
                  "regression": ("RandomForest", "LinearRegression")}   # the same pair v1 picked from


def _usable(problem_type: str, models: Sequence[str] | None) -> tuple[str, ...]:
    """The models of `models` (default: v1's pair) that serve `problem_type`, in the given order."""
    if problem_type not in DEFAULT_MODELS:
        raise ValueError(f"unknown problem type {problem_type!r}; expected one of {sorted(DEFAULT_MODELS)}")
    names = DEFAULT_MODELS[problem_type] if models is None else models
    usable = tuple(n for n in dict.fromkeys(names) if n in _MODEL_TABLE and problem_type in _MODEL_TABLE[n][0])
    if not usable:
        raise ValueError(f"none of {tuple(names)!r} fits a {problem_type} problem; known models: {MODELS}")
    return usable


def tabular_space(problem_type: str, n_features: int, models: Sequence[str] | None = None) -> Space:
    return {"model": Choice({n: _MODEL_TABLE[n][1] for n in _usable(problem_type, models)}),
            "features": Mask(n_features)}


def warm_seeds(problem_type: str, n_features: int, models: Sequence[str] | None = None) -> list[dict]:
    """v1's warm-start defaults, one seed per usable model, every feature on."""
    return [{"model": n, **_MODEL_TABLE[n][2], "features": (1,) * n_features} for n in _usable(problem_type, models)]


# Folds of one CV run on joblib's loky process pool, which outlives the call (v1 leaned on the same pool).
# The size is FIXED: loky kills and respawns its workers whenever n_jobs changes between calls, which
# measured slower than serial. Five is the top rung's fold count.
CV_POOL = 5
SCREEN_EFFORT = 0.2   # the screening rung fits ensembles with this fraction of their trees


def default_ladder(n_rows: int) -> tuple[Fidelity, ...]:
    return ((Fidelity(1.0, 3, SCREEN_EFFORT), Fidelity(1.0, 5)) if n_rows < 5000
            else (Fidelity(0.25, 3, SCREEN_EFFORT), Fidelity(1.0, 5)))


class TabularFitness:
    """v1's penalised CV score for one params dict at one Fidelity. Picklable; safe to call from threads
    (it only reads shared state and builds a fresh pipeline per call)."""

    def __init__(self, X: pd.DataFrame, y: pd.Series, problem_type: str, profile: dict,
                 cv_jobs: int = CV_POOL) -> None:
        self.X, self.y = X, y
        self.cv_jobs = cv_jobs
        self.classification = problem_type == "classification"
        counts = y.value_counts() if self.classification else None
        self.n_classes, self.min_class = (len(counts), int(counts.min())) if self.classification else (0, 0)
        self.names = X.columns.tolist()
        self.is_numeric = [pd.api.types.is_numeric_dtype(X[c]) for c in self.names]
        imbalance = profile.get("target_analysis", {}).get("imbalance_risk", False)
        self.class_weight = "balanced" if imbalance and self.classification else None
        self.scale = 1 + profile.get("overfitting_risk_index", {}).get("score", 0.5)   # 1 + beta * ORI, beta = 1

    def _rows(self, fidelity: Fidelity) -> tuple[np.ndarray | None, pd.Series]:
        """Row positions of the (sub)sample at this fidelity (None = every row) and its target."""
        total = len(self.y)
        if fidelity.rows >= 1:
            return None, self.y
        n = max(round(fidelity.rows * total), 2, self.n_classes)
        if n >= total:
            return None, self.y
        # a stratified split needs >= 2 rows per class overall and at least one per class on each side
        stratify = self.y if self.classification and self.min_class >= 2 and total - n >= self.n_classes else None
        idx, _ = train_test_split(np.arange(total), train_size=n, stratify=stratify, random_state=0)
        return idx, self.y.iloc[idx]

    def __call__(self, params: dict, fidelity: Fidelity) -> tuple[float, dict]:
        hp = {k: v for k, v in params.items() if k not in ("model", "features")}
        name = params["model"]
        fit_params = params
        if fidelity.effort < 1 and "n_estimators" in hp:   # a low rung screens with fewer trees
            fit_params = dict(params, n_estimators=max(10, round(hp["n_estimators"] * fidelity.effort)))
        pipe = build_pipeline(fit_params, self.names, self.is_numeric, self.classification, self.class_weight)
        selected = [self.names[i] for i, bit in enumerate(params["features"]) if bit]

        idx, y = self._rows(fidelity)
        X = self.X if idx is None else self.X.iloc[idx]
        n_rows = len(y)

        smallest = (int(y.value_counts().min()) if idx is not None else self.min_class) if self.classification else n_rows
        n_splits = min(max(2, min(fidelity.folds, smallest)), n_rows)   # never more folds than the rarest class
        cv_cls = StratifiedKFold if self.classification and smallest >= 2 else KFold
        cv = cv_cls(n_splits=n_splits, shuffle=True, random_state=0)

        res = cross_validate(pipe, X, y, cv=cv, scoring="accuracy" if self.classification else "r2",
                             return_train_score=True, error_score="raise", n_jobs=self.cv_jobs)
        mu_cv = float(np.mean(res["test_score"]))
        sigma_cv = float(np.std(res["test_score"]))
        gap = float(max(0.0, np.mean(res["train_score"]) - mu_cv))

        # Complexity penalty C(P)
        model_penalty = 0.0
        if name in ("RandomForest", "GradientBoosting"):
            depth = hp.get("max_depth", 10) or 20
            model_penalty = (hp.get("n_estimators", 100) / 250) * 0.5 + (max(2, depth) / 20) * 0.5
        elif name == "DecisionTree":
            model_penalty = max(2, hp.get("max_depth", 10) or 20) / 20
        c_p = (len(selected) / len(self.names)) * 0.4 + model_penalty * 0.6

        fitness = mu_cv - 0.02 * self.scale * c_p - 0.10 * self.scale * sigma_cv - 0.50 * self.scale * gap
        return fitness, {"mu_cv": mu_cv, "sigma_cv": sigma_cv, "gap": gap, "complexity": float(c_p),
                         "n_rows": int(n_rows), "folds": int(n_splits)}


def evolve_tabular(df: pd.DataFrame, target: str, profile: dict, *, models: Sequence[str] | None = None,
                   pop_size: int = 10, budget: Budget = Budget(), workers: int | None = None,
                   seeds: Sequence[dict] = (), on_generation=None, seed: int = 0) -> EvolveResult:
    problem_type = profile["target_analysis"]["target_type"]
    X, y = df.drop(columns=[target]), df[target]
    big = len(df) >= 20_000   # candidates then run in processes, where each worker would start its own loky pool
    return evolve(tabular_space(problem_type, X.shape[1], models),
                  TabularFitness(X, y, problem_type, profile, cv_jobs=1 if big else CV_POOL),
                  pop_size=pop_size, budget=budget, workers=workers,
                  executor="processes" if big else "threads", fidelity=default_ladder(len(df)),
                  seeds=list(seeds) + warm_seeds(problem_type, X.shape[1], models),
                  on_generation=on_generation, seed=seed)
