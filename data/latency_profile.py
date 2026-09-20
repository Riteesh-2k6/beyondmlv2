"""
BeyondML Pipeline Latency Profiler
-----------------------------------
Instruments every agent stage with wall-clock timing.
Runs a full supervised pipeline on a small synthetic dataset,
records per-agent latency, and writes a JSON report.

Usage:
    PYTHONPATH=. venv/Scripts/python data/latency_profile.py

Requires: Ollama running locally with OLLAMA_MODEL set (or defaulting to qwen3:8b)
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── make beyondml importable ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from beyondml.engine.profiler import DatasetProfiler, TargetIdentifier
from beyondml.agents.orchestrator import OrchestratorAgent
from beyondml.agents.sanity_agent import SanityAgent
from beyondml.agents.imputation_agent import ImputationAgent
from beyondml.agents.leakage_agent import LeakageAgent
from beyondml.agents.eda_agent import EDAAgent
from beyondml.agents.outlier_agent import OutlierAgent
from beyondml.agents.feature_agent import FeatureAgent
from beyondml.agents.ga_trainer import GATrainerAgent
from beyondml.agents.evaluator_agent import EvaluatorAgent
from beyondml.agents.reflection_agent import ReflectionAgent

# Load .env
_env = Path(__file__).parent.parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


# ── Instrumentation helpers ──────────────────────────────────────────

class Timer:
    def __init__(self, name: str, results: list):
        self.name = name
        self.results = results
        self._t0 = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        elapsed = time.perf_counter() - self._t0
        self.results.append({"stage": self.name, "seconds": round(elapsed, 3)})
        print(f"  [{self.name}] {elapsed:.3f}s")


logs_collected = []
async def noop_log(msg: str):
    logs_collected.append(msg)

async def noop_input(prompt: str) -> str:
    return ""  # always accept defaults


# ── Synthetic dataset ────────────────────────────────────────────────

def make_df(n=300):
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "age":      rng.randint(18, 70, n),
        "income":   rng.normal(50_000, 15_000, n).round(2),
        "score":    rng.uniform(0, 100, n).round(2),
        "category": rng.choice(["A", "B", "C"], n),
        "flag":     rng.choice([0, 1], n),
        "target":   rng.choice([0, 1], n),
    })
    # Introduce some NaNs so ImputationAgent has work to do
    df.loc[rng.choice(n, 20, replace=False), "income"] = np.nan
    return df


# ── Main profiler ─────────────────────────────────────────────────────

async def run_profile():
    timings = []
    pipeline_start = time.perf_counter()

    df = make_df()
    print(f"\nDataset: {df.shape[0]} rows × {df.shape[1]} cols")

    # Detect LLM provider
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    print(f"LLM provider: {provider}\n")
    if provider == "groq":
        from beyondml.llm.groq_provider import GroqProvider
        llm = GroqProvider()
    else:
        from beyondml.llm.ollama_provider import OllamaProvider
        llm = OllamaProvider()

    # ── Stage: Profile ─────────────────────────────────────────────
    with Timer("profiler", timings):
        profiler = DatasetProfiler(df)
        profile = profiler.run()
        target_id = TargetIdentifier(df)
        target_info = target_id.identify()
        df_summary = (
            f"Shape: {df.shape}\nColumns: {list(df.columns)}\n"
            f"Sample:\n{df.head(3).to_string()}"
        )

    # ── Stage: Orchestrator ────────────────────────────────────────
    with Timer("orchestrator", timings):
        orch = OrchestratorAgent(llm)
        decision = await orch.run(
            df_summary=df_summary,
            description="Synthetic binary classification dataset for latency profiling.",
            user_target=target_info.get("suggested_target"),
            user_path_choice="supervised",
            log=noop_log,
        )
    print(f"    path={decision.get('path')}  target={decision.get('suggested_target')}")

    # ── Stage: Sanity Agent ────────────────────────────────────────
    with Timer("sanity_agent", timings):
        col_stats = profile.get("numerical_summary", {})
        sanity = SanityAgent(llm)
        await sanity.run(df_summary, col_stats, noop_log)

    # ── Stage: Imputation Agent ────────────────────────────────────
    missing_pct = profile.get("missing_analysis", {}).get("missing_percentage", {})
    missing_report = {k: v for k, v in missing_pct.items() if v > 0}
    with Timer("imputation_agent", timings):
        imp = ImputationAgent(llm)
        imp_result = await imp.run(df_summary, missing_report, noop_log)

    # Simple imputation application (fill NaN with median)
    for s in imp_result.get("strategies", []):
        col = s.get("column")
        strategy = s.get("strategy")
        if col in df.columns:
            if strategy == "median":
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == "mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == "mode":
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif strategy == "drop":
                df.drop(columns=[col], inplace=True)

    # ── Stage: Leakage Agent ───────────────────────────────────────
    target = decision.get("suggested_target") or "target"
    with Timer("leakage_agent", timings):
        corrs = {
            col: float(df[col].corr(df[target]))
            for col in df.select_dtypes("number").columns
            if col != target
        }
        col_descs = "\n".join(f"{c}: {df[c].dtype}" for c in df.columns)
        leakage = LeakageAgent(llm)
        await leakage.run(target, col_descs, corrs, noop_log)

    # ── Stage: EDA Agent ───────────────────────────────────────────
    with Timer("eda_agent", timings):
        profiler2 = DatasetProfiler(df, target_column=target)
        profile2 = profiler2.run()
        eda = EDAAgent(llm)
        eda_result = await eda.run(df, profile2, target_info, "", noop_log)

    # ── Stage: Outlier Agent ───────────────────────────────────────
    with Timer("outlier_agent", timings):
        outlier_summary = profile2.get("outlier_summary", {})
        outlier = OutlierAgent(llm)
        outlier_result = await outlier.run(df, outlier_summary, profile2, noop_log, noop_input)
    df = outlier_result["df"]

    # ── Stage: Feature Agent ───────────────────────────────────────
    with Timer("feature_agent", timings):
        profiler3 = DatasetProfiler(df, target_column=target)
        profile3 = profiler3.run()
        feat = FeatureAgent(llm)
        feat_result = await feat.run(df, profile3, eda_result.get("eda_insights", []), noop_log)
    df = feat_result["df"]

    # ── Stage: GA Trainer ──────────────────────────────────────────
    with Timer("ga_trainer", timings):
        profiler4 = DatasetProfiler(df, target_column=target)
        profile4 = profiler4.run()
        ga = GATrainerAgent(llm)
        ga_result = await ga.run(
            df=df,
            target_column=target,
            profile=profile4,
            model_choice="RandomForest",
            log=noop_log,
            get_user_input=noop_input,
            pop_size=5,     # small for profiling speed
            generations=3,
        )
    print(f"    GA best_cv_score={ga_result['best_cv_score']:.4f}")

    # ── Stage: Evaluator ──────────────────────────────────────────
    with Timer("evaluator_agent", timings):
        evaluator = EvaluatorAgent(llm)
        eval_result = await evaluator.run(
            df=df,
            target_column=target,
            profile=profile4,
            best_params=ga_result["best_params"],
            model_type=ga_result["model_type"],
            problem_type=profile4["target_analysis"]["target_type"],
            log=noop_log,
        )
    print(f"    test_score={eval_result.get('test_score', 0):.4f}")

    # ── Stage: Reflection Agent ────────────────────────────────────
    with Timer("reflection_agent", timings):
        reflect = ReflectionAgent(llm)
        await reflect.run(eval_result, iteration=1, max_iterations=3, log=noop_log)

    # ── Summary ────────────────────────────────────────────────────
    total = time.perf_counter() - pipeline_start

    print("\n" + "=" * 52)
    print("  LATENCY REPORT")
    print("=" * 52)
    print(f"  {'Stage':<28} {'Time (s)':>8}  {'% total':>7}")
    print("  " + "-" * 46)
    for t in timings:
        pct = t["seconds"] / total * 100
        flag = " <-- BOTTLENECK" if pct > 25 else ""
        print(f"  {t['stage']:<28} {t['seconds']:>8.3f}  {pct:>6.1f}%{flag}")
    print("  " + "-" * 46)
    print(f"  {'TOTAL':<28} {total:>8.3f}  100.0%")
    print("=" * 52)

    # ── Derive inter-agent gaps ────────────────────────────────────
    gaps = []
    for i in range(1, len(timings)):
        # Gap is implicitly zero (stages run back-to-back in runner.py)
        # but serial ordering means each LLM call blocks the next stage.
        gaps.append({
            "from": timings[i-1]["stage"],
            "to":   timings[i]["stage"],
            "blocking_seconds": timings[i-1]["seconds"],
        })

    report = {
        "provider": provider,
        "dataset_shape": list(df.shape),
        "total_seconds": round(total, 3),
        "stages": timings,
        "serial_bottlenecks": [t for t in timings if t["seconds"] / total > 0.25],
        "parallelisable_pairs": [
            ["sanity_agent", "imputation_agent"],
            ["eda_agent", "outlier_agent"],
            ["leakage_agent", "sanity_agent"],
        ],
    }

    out = Path(__file__).parent / "latency_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n  Report saved → {out}")
    return report


if __name__ == "__main__":
    asyncio.run(run_profile())
