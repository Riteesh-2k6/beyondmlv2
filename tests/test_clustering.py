import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from beyondml.engine.clustering import SilhouetteFitness, cluster_spec, evolve_clusters
from beyondml.engine.evolve import Fidelity
from beyondml.engine.export import write_project_code
from beyondml.engine.pipeline import column_kinds
from test_export import run


def _blobs(n=300):
    X, _ = make_blobs(n_samples=n, centers=3, cluster_std=0.6, random_state=0)
    return pd.DataFrame(X, columns=["x0", "x1"])


def test_kmeans_finds_blobs():
    result = evolve_clusters(_blobs(), algorithms=("kmeans",), pop_size=6, generations=4, seed=0)
    assert result.best.params["algo"] == "kmeans" and result.best.metrics["n_clusters"] == 3
    assert result.best.metrics["silhouette"] > 0.6


def test_dbscan_noise_penalised():
    params = {"algo": "dbscan", "eps": 1e-4, "min_samples": 5, "pca": 0}
    score, metrics = SilhouetteFitness(_blobs())(params, Fidelity())
    assert score == -1.0 and metrics["noise_pct"] > 0.5


@pytest.mark.parametrize("params", [{"algo": "kmeans", "n_clusters": 3, "pca": 0},
                                    {"algo": "dbscan", "eps": 0.8, "min_samples": 5, "pca": 2}])
def test_cluster_export_runs(tmp_path, params):
    df = _blobs()
    root = tmp_path / "proj"
    (root / "data" / "raw").mkdir(parents=True)
    df.to_csv(root / "data" / "raw" / "b.csv", index=False)
    names, kinds = column_kinds(df)
    write_project_code(root, cluster_spec(params, names, kinds, "data/raw/b.csv"))
    trained = run(root, "src/train.py")
    assert trained.returncode == 0, trained.stderr
    clusters = pd.read_csv(root / "data" / "processed" / "clusters.csv")
    assert "cluster" in clusters.columns and len(clusters) == len(df)
    df.head(5).to_csv(root / "new.csv", index=False)
    predicted = run(root, "src/predict.py", "new.csv", "out.csv")
    assert predicted.returncode == 0, predicted.stderr
    assert len(pd.read_csv(root / "out.csv")["cluster"]) == 5


def test_cluster_predict_bad_input_exits_2(tmp_path):
    # review 10-24: same gap as the supervised predict.py; clean/features column or a text value gave a traceback
    df = _blobs()
    root = tmp_path / "proj"
    (root / "data" / "raw").mkdir(parents=True)
    df.to_csv(root / "data" / "raw" / "b.csv", index=False)
    names, kinds = column_kinds(df)
    write_project_code(root, cluster_spec({"algo": "kmeans", "n_clusters": 3, "pca": 0}, names, kinds, "data/raw/b.csv"))
    (root / "src" / "features.py").write_text("def make_features(df):\n    return df.assign(s=df['x0'] + df['x1'])\n",
                                              encoding="utf-8")
    assert run(root, "src/train.py").returncode == 0
    df[["x1"]].head(2).to_csv(root / "no_x0.csv", index=False)
    short = run(root, "src/predict.py", "no_x0.csv", "o.csv")
    assert short.returncode == 2 and "x0" in short.stderr and "Traceback" not in short.stderr
    pd.DataFrame({"x0": ["abc"], "x1": [1.0]}).to_csv(root / "bad.csv", index=False)
    bad = run(root, "src/predict.py", "bad.csv", "o2.csv")
    assert bad.returncode == 2 and "Traceback" not in bad.stderr
