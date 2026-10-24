"""PMLB benchmark runner and the synthetic frames the evolution benchmarks use."""

import numpy as np
import pandas as pd
from typing import List
from pmlb import fetch_data

from beyondml.engine.evolve import Budget
from beyondml.engine.fitness import evolve_tabular
from beyondml.engine.profiler import DatasetProfiler


def latency_frame() -> pd.DataFrame:
    """Synthetic 300-row frame used by the GA speed benchmark.

    Copy of make_df(300) from the retired data/latency_profile.py (seed 42, 20 NaNs in income).
    """
    n = 300
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "age":      rng.randint(18, 70, n),
        "income":   rng.normal(50_000, 15_000, n).round(2),
        "score":    rng.uniform(0, 100, n).round(2),
        "category": rng.choice(["A", "B", "C"], n),
        "flag":     rng.choice([0, 1], n),
        "target":   rng.choice([0, 1], n),
    })
    df.loc[rng.choice(n, 20, replace=False), "income"] = np.nan
    return df


def quality_frame() -> pd.DataFrame:
    """Learnable 300-row frame (6 features, 4 informative) used by the GA quality benchmark."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=300, n_features=6, n_informative=4, n_redundant=0, random_state=0
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["target"] = y
    return df


class PMLBRunner:
    """Runs the genetic search on PMLB datasets and writes benchmark_results.md."""

    def __init__(self, datasets: List[str] = None):
        self.datasets = datasets or ["titanic", "breast_cancer", "iris", "wine_recognition", "diabetes"]
        self.results = []

    def run_benchmark(self):
        print("Starting BeyondML Standardized Benchmark Suite...")
        for name in self.datasets:
            print(f"\n[Benchmark: {name}] Fetching data...")
            try:
                X, y = fetch_data(name, return_X_y=True, local_cache_dir="./data/pmlb_cache")
                df = pd.DataFrame(X)
                df.columns = [str(c) for c in df.columns]
                df["target"] = y
                print(f"  Shape: {df.shape}")
                profile = DatasetProfiler(df, target_column="target").run()
                best = evolve_tabular(df, "target", profile, pop_size=5, budget=Budget(max_generations=3)).best
                res = {"dataset": name, "path": profile["target_analysis"]["target_type"],
                       "model": best.params.get("model", "N/A"), "mu_cv": best.metrics.get("mu_cv", best.fitness)}
                self.results.append(res)
                print(f"  Result: {res['mu_cv']:.4f}")
            except Exception as e:
                print(f"  Error on {name}: {e}")
        self._export_results()

    def _export_results(self):
        if not self.results:
            print("No results to export.")
            return

        df_res = pd.DataFrame(self.results)
        print("\nBENCHMARK SUMMARY")
        print(df_res.to_string())
        
        report_path = "benchmark_results.md"
        with open(report_path, "w") as f:
            f.write("# BeyondML Benchmark Report (PMLB)\n\n")
            f.write("| Dataset | Path | Model | Mu_CV / Score |\n")
            f.write("|---|---|---|---|\n")
            for r in self.results:
                f.write(f"| {r['dataset']} | {r['path']} | {r['model']} | {r['mu_cv']:.4f} |\n")
        
        print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    PMLBRunner().run_benchmark()
