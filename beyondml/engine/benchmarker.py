"""
PMLB Benchmarking Suite for BeyondML.
Runs the autonomous pipeline on standardized datasets to measure effectiveness.
"""

import pandas as pd
import numpy as np
import asyncio
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from pmlb import fetch_data, dataset_names

# Load API keys
load_dotenv()
from beyondml.engine.profiler import DatasetProfiler, TargetIdentifier
from beyondml.engine.genetic import GeneticModelOptimizer
from beyondml.agents.orchestrator import OrchestratorAgent
from beyondml.agents.ga_trainer import GATrainerAgent
from beyondml.llm import get_llm_provider

def latency_frame() -> pd.DataFrame:
    """Synthetic 300-row frame used by the GA speed benchmark.

    Exact copy of make_df(300) in data/latency_profile.py (seed 42, 20 NaNs in income).
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
    """Benchmark runner for BeyondML using PMLB datasets."""

    def __init__(self, datasets: List[str] = None):
        if datasets is None:
            # A mix of classification and regression
            self.datasets = ["titanic", "breast_cancer", "iris", "wine_recognition", "diabetes"] 
        else:
            self.datasets = datasets
        self.results = []

    async def run_benchmark(self):
        print("Starting BeyondML Standardized Benchmark Suite...")
        llm = get_llm_provider()
        orchestrator = OrchestratorAgent(llm)
        
        for name in self.datasets:
            print(f"\n[Benchmark: {name}] Fetching data...")
            try:
                X, y = fetch_data(name, return_X_y=True, local_cache_dir='./data/pmlb_cache')
                df = pd.DataFrame(X)
                df.columns = [str(c) for c in df.columns]
                df['target'] = y
                
                print(f"  Shape: {df.shape}")
                
                async def mock_log(msg): print(f"    {msg}")
                
                # 1. Orchestrate
                df_summary = f"Rows: {len(df)}, Columns: {len(df.columns)}"
                target_id = TargetIdentifier(df)
                target_info = target_id.identify()
                
                decision = await orchestrator.run(
                    df_summary=df_summary,
                    description=f"PMLB Dataset: {name}",
                    target_info=target_info,
                    user_path_choice="auto",
                    log=mock_log
                )
                
                path = decision["path"]
                target = decision.get("suggested_target", "target")
                model_recs = decision.get("model_recommendations", ["RandomForest"])
                raw_model = model_recs[0] if model_recs else "RandomForest"
                
                # Normalize LLM model names → GA-compatible names
                MODEL_NORMALIZE = {
                    "logistic regression": "LogisticRegression",
                    "logisticregression": "LogisticRegression",
                    "random forest": "RandomForest",
                    "randomforest": "RandomForest",
                    "gradient boosting": "GradientBoosting",
                    "gradientboosting": "GradientBoosting",
                    "decision tree": "DecisionTree",
                    "decisiontree": "DecisionTree",
                    "linear regression": "LinearRegression",
                    "linearregression": "LinearRegression",
                    "svm": "SVM",
                    "support vector machine": "SVM",
                    "knn": "KNN",
                    "k-nearest neighbors": "KNN",
                }
                model_choice = MODEL_NORMALIZE.get(raw_model.lower().strip(), raw_model)
                print(f"  Orchestrator chose: {path} | Model: {model_choice}")

                # 2. Execute Branch
                if path == "supervised":
                    optimizer = GeneticModelOptimizer(
                        df=df,
                        target_column=target,
                        profile=DatasetProfiler(df, target_column=target).run(),
                        pop_size=5,
                        generations=3,
                        model_choice=model_choice
                    )
                    history, best = optimizer.evolve()
                    res = {
                        "dataset": name,
                        "path": path,
                        "best_fitness": best.fitness,
                        "model": best.model_choice,
                        "mu_cv": best.metrics.get("mu_cv", 0)
                    }
                elif path == "deep_learning":
                    from beyondml.agents.dl_agent import DeepLearningAgent
                    dl_agent = DeepLearningAgent(llm)
                    dl_res = await dl_agent.run(df, target, problem_type="classification", log=mock_log, epochs=3)
                    res = {
                        "dataset": name,
                        "path": path,
                        "best_fitness": dl_res.get("test_score", 0),
                        "model": "SimpleMLP",
                        "mu_cv": dl_res.get("test_score", 0)
                    }
                else:
                    res = {"dataset": name, "path": path, "best_fitness": 0, "model": "N/A", "mu_cv": 0}

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
    runner = PMLBRunner()
    asyncio.run(runner.run_benchmark())
