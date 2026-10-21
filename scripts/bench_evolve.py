"""Record genetic-optimiser speed and quality numbers as JSON.

Usage (from the repo root):
    python scripts/bench_evolve.py --engine v1|v2 [--record PATH] [--dataset latency|quality|adult]

Sections run per dataset (sections not run are null in the JSON):
    latency -> speed
    quality -> speed + quality   (default)
    adult   -> adult
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SPEED_POP, SPEED_GENS = 5, 3
QUALITY_POP, QUALITY_GENS, QUALITY_SEEDS = 10, 5, [0, 1, 2]
ADULT_POP, ADULT_GENS = 5, 3


def _profile(df):
    from beyondml.engine.profiler import DatasetProfiler

    return DatasetProfiler(df, target_column="target").run()


V1_GONE = (
    "bench_evolve: engine v1 was replaced by an adapter over v2 (Task 6), so it can no longer be measured here.\n"
    "Its recorded numbers are tests/data/evolve_v1_baseline.json and data/evolve_benchmark_v1_adult.json;\n"
    "reproduce them from commit 8553b86."
)


def _run_v2(df, profile, pop_size, generations, model_choice, seed):
    """One seeded v2 run. Times evolve_tabular() only; time-to-best is the GenStats.seconds of the first
    generation whose best_fitness equals the final best.

    Returns (seconds, time_to_best_s, best).
    """
    from beyondml.engine.evolve import Budget
    from beyondml.engine.fitness import evolve_tabular

    t0 = time.perf_counter()
    result = evolve_tabular(
        df, "target", profile, models=(model_choice,) if model_choice else None, pop_size=pop_size,
        budget=Budget(max_generations=generations), seed=seed,
    )
    seconds = time.perf_counter() - t0
    time_to_best = next(h.seconds for h in result.history if h.best_fitness == result.best.fitness)
    return seconds, time_to_best, result.best


def _speed(runner, df, profile):
    """Two consecutive identical seeded runs: cold (first in process), then warm."""
    cold, _, _ = runner(df, profile, SPEED_POP, SPEED_GENS, "RandomForest", 0)
    warm, _, _ = runner(df, profile, SPEED_POP, SPEED_GENS, "RandomForest", 0)
    return {"cold_s": round(cold, 3), "warm_s": round(warm, 3)}


def _quality(runner, df, profile):
    mus = []
    for seed in QUALITY_SEEDS:
        _, _, best = runner(df, profile, QUALITY_POP, QUALITY_GENS, None, seed)
        mus.append(round(float(best.metrics["mu_cv"]), 6))
    return {"seeds": QUALITY_SEEDS, "mu_cv": mus, "mean_mu_cv": round(float(np.mean(mus)), 6)}


def _adult(runner):
    df = pd.read_csv(ROOT / "data" / "adult.csv")
    profile = _profile(df)
    seconds, time_to_best, best = runner(df, profile, ADULT_POP, ADULT_GENS, "RandomForest", 0)
    return {
        "seconds": round(seconds, 3),
        "time_to_best_s": round(time_to_best, 3),
        "best_mu_cv": round(float(best.metrics["mu_cv"]), 6),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["v1", "v2"], required=True)
    ap.add_argument("--record", metavar="PATH", help="also write the JSON to PATH")
    ap.add_argument("--dataset", choices=["latency", "quality", "adult"], default="quality")
    args = ap.parse_args()

    if args.engine == "v1":
        sys.exit(V1_GONE)
    runner = _run_v2

    from beyondml.engine.benchmarker import latency_frame, quality_frame

    out = {"engine": args.engine, "cpu_count": os.cpu_count(), "speed": None, "quality": None, "adult": None}

    if args.dataset in ("latency", "quality"):
        df = latency_frame()
        out["speed"] = _speed(runner, df, _profile(df))
    if args.dataset == "quality":
        df = quality_frame()
        out["quality"] = _quality(runner, df, _profile(df))
    if args.dataset == "adult":
        out["adult"] = _adult(runner)

    text = json.dumps(out, indent=2)
    print(text)
    if args.record:
        path = Path(args.record)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
