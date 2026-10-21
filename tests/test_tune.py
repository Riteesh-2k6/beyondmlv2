"""Tests for the optional Optuna tune() backend; skipped when optuna is not installed.

Tests that count calls pass the default workers (one Optuna thread), so call order is deterministic.
"""

import math
import sys

import pytest

optuna = pytest.importorskip("optuna")

from beyondml.engine.evolve import (  # noqa: E402
    Budget, Candidate, Choice, EvolveError, Fidelity, Float, GenStats, Int, Mask, active_keys,
)
from beyondml.engine.tune import tune  # noqa: E402

SPACE = {"x": Float(-5, 5), "y": Float(-5, 5)}
COND = {
    "model": Choice({
        "rf": {"n": Int(10, 200), "depth": Choice({"shallow": {"leaf": Int(1, 5)}, "deep": {}})},
        "lr": {"c": Float(0.01, 10.0, log=True)},
    }),
    "cols": Mask(3),
    "scale": Choice((True, False)),
}


def quad(params, fid):
    return -(params["x"] - 1) ** 2 - (params["y"] + 2) ** 2, {}


def cond_fit(params, fid):
    return -((params.get("n", 100) - 100) ** 2) / 1e4 - params.get("c", 1.0) - params.get("leaf", 0), {}


def test_tpe_finds_optimum():
    r = tune(SPACE, quad, n_trials=60, seed=0)
    assert abs(r.best.params["x"] - 1) < 0.5 and abs(r.best.params["y"] + 2) < 0.5
    assert isinstance(r.best, Candidate) and r.stop_reason == "trials" and r.pareto == []


def test_conditional_params_clean():
    r = tune(COND, cond_fit, n_trials=30, seed=0)
    assert r.population
    for c in [r.best, *r.population]:
        assert set(c.params) == active_keys(COND, c.params)
        assert not any(":" in k or "[" in k for k in c.params)
        assert isinstance(c.params["cols"], tuple) and any(c.params["cols"])
    assert {c.params["model"] for c in r.population} == {"rf", "lr"}


def test_mask_never_empty():
    r = tune({"m": Mask(2)}, lambda p, f: (-float(sum(p["m"])), {}), n_trials=30, seed=0)
    assert all(any(c.params["m"]) for c in r.population)


def test_pruning_cuts_top_rung_calls():
    rows = []

    def fitness(params, fid):
        rows.append(fid.rows)
        return quad(params, fid)

    r = tune(SPACE, fitness, n_trials=30, fidelity=(Fidelity(0.5, 3), Fidelity()), seed=0)
    assert rows.count(0.5) == 30 and 0 < rows.count(1.0) < 30
    assert len(r.history) == rows.count(1.0) and r.evals == len(rows)
    assert r.best.rung == 1


def test_nsga2_returns_pareto():
    r = tune(SPACE, lambda p, f: (quad(p, f)[0], {"cost": p["x"] ** 2}), strategy="nsga2",
             objectives=("fitness:max", "cost:min"), n_trials=40, seed=0)
    assert len(r.pareto) >= 2 and all(isinstance(c, Candidate) for c in r.pareto)
    assert r.best.fitness == max(c.fitness for c in r.pareto)   # highest first objective


def test_cmaes_runs():
    r = tune(SPACE, quad, strategy="cmaes", n_trials=20, seed=0)
    assert r.best.fitness > float("-inf")


def test_seed_enqueued_first():
    r = tune(SPACE, quad, n_trials=5, seeds=[{"x": 1.0, "y": -2.0}], seed=0)
    assert r.history[0].best_fitness == 0.0


def test_conditional_seed_round_trips():
    calls = []
    seed = {"model": "rf", "n": 50, "depth": "shallow", "leaf": 2, "cols": (1, 0, 1), "scale": False}
    tune(COND, lambda p, f: calls.append(p) or cond_fit(p, f), n_trials=3, seeds=[seed], seed=0)
    assert calls[0] == seed


def test_invalid_seed_rejected_before_any_call():
    calls = []
    with pytest.raises(ValueError, match="zzz"):
        tune(SPACE, lambda p, f: calls.append(p) or quad(p, f), seeds=[{"zzz": 1}])
    assert calls == []


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError, match="grid"):
        tune(SPACE, quad, strategy="grid")


def test_all_failing_fitness_raises():
    def broken(params, fid):
        raise KeyError("col")

    with pytest.raises(EvolveError, match="KeyError"):
        tune(SPACE, broken, n_trials=5, seed=0)


def test_partial_failures_tolerated():
    def picky(params, fid):
        if params["x"] < 0:
            raise ValueError("negative x")
        return quad(params, fid)

    r = tune(SPACE, picky, n_trials=20, seed=0)
    assert math.isfinite(r.best.fitness) and r.best.params["x"] >= 0


def test_history_and_callback():
    seen = []
    r = tune(SPACE, quad, n_trials=12, on_trial=seen.append, seed=0)
    assert [h.gen for h in r.history] == list(range(1, 13)) and all(isinstance(h, GenStats) for h in r.history)
    assert len(seen) == 12 and set(seen[0]) >= set(GenStats.__dataclass_fields__)
    best = [h.best_fitness for h in r.history]
    assert best == sorted(best) and best[-1] == r.best.fitness
    assert r.history[-1].evals == r.evals == 12 and r.history[-1].diversity == 1.0


def test_workers_run_trials_in_threads():
    r = tune(SPACE, quad, n_trials=20, workers=2, seed=0)
    assert len(r.history) == 20 and r.evals == 20


def test_max_seconds_stops():
    import time

    def slow(params, fid):
        time.sleep(0.05)
        return quad(params, fid)

    r = tune(SPACE, slow, n_trials=1000, budget=Budget(max_seconds=0.3), seed=0)
    assert r.stop_reason == "seconds" and 0 < len(r.history) < 1000


def test_spent_time_budget_still_returns_best():
    r = tune(SPACE, quad, n_trials=50, budget=Budget(max_seconds=0), seed=0)
    assert r.stop_reason == "seconds" and r.best.fitness > float("-inf")


def test_single_objective_min():
    r = tune(SPACE, lambda p, f: (quad(p, f)[0], {"cost": p["x"] ** 2}), objectives=("cost:min",),
             n_trials=30, seed=0)
    assert r.best.metrics["cost"] == min(c.metrics["cost"] for c in r.population) and r.pareto == []


@pytest.mark.parametrize("bad", [("fitness",), ("fitness:max", "cost:down")])
def test_bad_objective_rejected(bad):
    with pytest.raises(ValueError, match="max"):
        tune(SPACE, quad, objectives=bad)


def test_missing_objective_metric_raises():
    with pytest.raises(EvolveError, match="nope"):
        tune(SPACE, quad, objectives=("nope:min",), n_trials=3)


def test_missing_optuna_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "optuna", None)
    with pytest.raises(ImportError, match=r"beyondml\[optuna\]"):
        tune(SPACE, quad)
