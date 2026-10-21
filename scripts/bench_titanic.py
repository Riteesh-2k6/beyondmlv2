"""Titanic holdout baseline for the v1 GA (sub-project 3's acceptance bar).

v1's optimiser no longer exists on main, so it is loaded from git history (commit 01aa1fb) at runtime.

    python scripts/bench_titanic.py --engine v1 [--record tests/data/titanic_v1_baseline.json]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from beyondml.engine.profiler import DatasetProfiler  # noqa: E402

V1_COMMIT = "01aa1fb"
POP_SIZE, GENERATIONS, SEED = 10, 5, 0


def load_v1():
    source = subprocess.run(["git", "show", f"{V1_COMMIT}:beyondml/engine/genetic.py"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout
    source = source.replace("from .metrics import", "from beyondml.engine.metrics import")
    path = Path(tempfile.mkdtemp()) / "genetic_v1.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("genetic_v1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def v1_pipeline(genome, X: pd.DataFrame, imbalance: bool):
    """The pipeline v1's evaluation worker builds for a genome (v1's default families are RF and LR)."""
    selected = [c for c, bit in zip(X.columns, genome.feature_mask) if bit]
    numeric = [c for c in selected if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in selected if c not in numeric]
    weight = "balanced" if imbalance else None
    if genome.model_choice == "RandomForest":
        model = RandomForestClassifier(**genome.hparams, class_weight=weight, random_state=42)
    elif genome.model_choice == "LogisticRegression":
        model = LogisticRegression(**genome.hparams, class_weight=weight)
    else:
        raise SystemExit(f"unexpected v1 model {genome.model_choice}")
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                              OneHotEncoder(handle_unknown="ignore")), categorical),
    ])
    return make_pipeline(pre, model), selected


def run_v1() -> dict:
    df = pd.read_csv(ROOT / "data" / "titanic.csv")
    train, holdout = train_test_split(df, test_size=0.2, random_state=0, stratify=df["target"])
    train, holdout = train.reset_index(drop=True), holdout.reset_index(drop=True)
    profile = DatasetProfiler(train, target_column="target").run()
    random.seed(SEED)
    np.random.seed(SEED)
    _, best = load_v1().GeneticModelOptimizer(train, "target", profile, pop_size=POP_SIZE,
                                              generations=GENERATIONS).evolve()
    imbalance = profile.get("target_analysis", {}).get("imbalance_risk", False)
    X = train.drop(columns=["target"])
    pipe, selected = v1_pipeline(best, X, imbalance)
    pipe.fit(X, train["target"])
    accuracy = float(pipe.score(holdout.drop(columns=["target"]), holdout["target"]))
    return {"engine": "v1", "seed": SEED, "holdout_accuracy": round(accuracy, 6), "best_model": best.model_choice,
            "pop_size": POP_SIZE, "generations": GENERATIONS, "features": selected}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", choices=["v1"], required=True)
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    result = run_v1()
    text = json.dumps(result, indent=2)
    print(text)
    if args.record:
        args.record.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
