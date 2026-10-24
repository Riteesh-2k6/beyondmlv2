"""The no-target path: evolve KMeans / DBSCAN (optionally after PCA) for the best silhouette."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline, make_pipeline

from .evolve import Budget, Choice, EvolveResult, Fidelity, Float, Int, Space, evolve
from .pipeline import column_kinds, preprocessor

ALGORITHMS = {"kmeans": {"n_clusters": Int(2, 10)},
              "dbscan": {"eps": Float(0.1, 5.0, log=True), "min_samples": Int(3, 20)}}
MAX_NOISE = 0.5
SILHOUETTE_ROWS = 2000


def cluster_space(algorithms: Sequence[str] = ("kmeans", "dbscan"), pca: bool = True) -> Space:
    unknown = set(algorithms) - set(ALGORITHMS)
    if unknown or not algorithms:
        raise ValueError(f"unknown clustering algorithms {sorted(unknown)}; expected {sorted(ALGORITHMS)}")
    return {"algo": Choice({a: ALGORITHMS[a] for a in algorithms}), "pca": Choice((0, 2, 5) if pca else (0,))}


def make_clusterer(params: dict):
    if params["algo"] == "kmeans":
        return KMeans(n_clusters=params["n_clusters"], n_init=10, random_state=0)
    return DBSCAN(eps=params["eps"], min_samples=params["min_samples"])


def build_clusterer(params: dict, names: list[str], is_numeric: list[bool]) -> Pipeline:
    numeric = [n for n, num in zip(names, is_numeric) if num]
    categorical = [n for n, num in zip(names, is_numeric) if not num]
    steps = [preprocessor(numeric, categorical, dense=True)]
    if params.get("pca"):
        steps.append(PCA(n_components=min(params["pca"], len(names)), random_state=0))
    return make_pipeline(*steps, make_clusterer(params))


def silhouette(Z: np.ndarray, labels: np.ndarray) -> tuple[float, int, float]:
    """(silhouette over non-noise points, cluster count, noise share); -1.0 when the clustering is degenerate."""
    noise = float(np.mean(labels == -1))
    n_clusters = len(set(labels.tolist()) - {-1})
    keep = labels != -1
    if n_clusters < 2 or noise > MAX_NOISE or keep.sum() <= n_clusters:
        return -1.0, n_clusters, noise
    sample = min(SILHOUETTE_ROWS, int(keep.sum()))
    return float(silhouette_score(Z[keep], labels[keep], sample_size=sample, random_state=0)), n_clusters, noise


class SilhouetteFitness:
    def __init__(self, X: pd.DataFrame) -> None:
        self.X = X
        self.names, self.is_numeric = column_kinds(X)

    def __call__(self, params: dict, fidelity: Fidelity) -> tuple[float, dict]:
        X = self.X if fidelity.rows >= 1 else self.X.sample(frac=fidelity.rows, random_state=0)
        pipe = build_clusterer(params, self.names, self.is_numeric)
        labels = np.asarray(pipe.fit_predict(X))
        score, n_clusters, noise = silhouette(np.asarray(pipe[:-1].transform(X)), labels)
        return score, {"silhouette": score, "n_clusters": n_clusters, "noise_pct": noise}


def evolve_clusters(df: pd.DataFrame, *, algorithms: Sequence[str] = ("kmeans", "dbscan"), pca: bool = True,
                    pop_size: int = 10, generations: int = 8, seed: int = 0, on_generation=None) -> EvolveResult:
    seeds = [{"algo": "kmeans", "n_clusters": 3, "pca": 0}] if "kmeans" in algorithms else []
    return evolve(cluster_space(algorithms, pca), SilhouetteFitness(df), pop_size=pop_size,
                  budget=Budget(max_generations=generations), seeds=seeds, on_generation=on_generation, seed=seed)


def cluster_spec(params: dict, names: list[str], is_numeric: list[bool], raw: str) -> dict:
    return {"kind": "cluster", "params": dict(params), "names": list(names), "is_numeric": list(is_numeric),
            "problem_type": "clustering", "class_weight": None, "target": None, "raw": raw}
