"""Evolution engine v2 targets (spec section 7, re-stated 2026-10-21): on the latency frame with RandomForest,
pop 5 x 3 generations, the warm run takes <= 5 s and the cold run no longer than v1's recorded cold run;
over seeds 0-2 on the quality frame the mean best mu_cv is >= v1's mean - 0.01.
"""

import json
import subprocess
import sys
from pathlib import Path

from beyondml.engine.benchmarker import quality_frame
from beyondml.engine.evolve import Budget
from beyondml.engine.fitness import evolve_tabular
from beyondml.engine.profiler import DatasetProfiler

ROOT = Path(__file__).resolve().parents[1]
BASELINE = json.loads((ROOT / "tests" / "data" / "evolve_v1_baseline.json").read_text())


def test_latency_frame_speed():
    # a fresh interpreter, so "cold" really pays the import and the fold pool's start
    run = subprocess.run([sys.executable, "scripts/bench_evolve.py", "--engine", "v2", "--dataset", "latency"],
                         cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]
    speed = json.loads(run.stdout)["speed"]
    assert speed["warm_s"] <= 5.0, speed
    assert speed["cold_s"] <= BASELINE["speed"]["cold_s"], speed


def test_quality_not_worse_than_v1():
    df = quality_frame()
    profile = DatasetProfiler(df, target_column="target").run()
    mu = [evolve_tabular(df, "target", profile, pop_size=10, budget=Budget(max_generations=5), seed=s)
          .best.metrics["mu_cv"] for s in (0, 1, 2)]
    assert sum(mu) / 3 >= BASELINE["quality"]["mean_mu_cv"] - 0.01, mu


def test_pmlb_runner_evolves_without_an_llm(monkeypatch, tmp_path):
    # the runner used the legacy LLM orchestrator and the GA adapter; it now runs evolve_tabular directly
    from beyondml.engine import benchmarker

    df = quality_frame()
    monkeypatch.setattr(benchmarker, "fetch_data", lambda name, **kw: (df.drop(columns=["target"]).values,
                                                                          df["target"].values))
    monkeypatch.chdir(tmp_path)
    runner = benchmarker.PMLBRunner(datasets=["tiny"])
    runner.run_benchmark()
    [row] = runner.results
    assert row["dataset"] == "tiny" and row["model"] and 0.5 < row["mu_cv"] <= 1.0
    assert "tiny" in (tmp_path / "benchmark_results.md").read_text()
