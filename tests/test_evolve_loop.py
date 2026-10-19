"""Tests for the evolution engine v2 GA loop (serial)."""

import math
import time

import pytest

from beyondml.engine.evolve import (
    Budget, Choice, EvolveError, Fidelity, Float, evolve,
)

SPACE = {"x": Float(-5, 5), "y": Float(-5, 5)}


def quad(params, fid):
    return -(params["x"] - 1) ** 2 - (params["y"] + 2) ** 2, {}


def test_converges_on_quadratic():
    r = evolve(SPACE, quad, pop_size=20, budget=Budget(max_generations=15, plateau_generations=99), seed=0)
    assert abs(r.best.params["x"] - 1) < 0.5 and abs(r.best.params["y"] + 2) < 0.5


def test_same_seed_same_history():
    runs = [evolve(SPACE, quad, pop_size=8, seed=3) for _ in range(2)]
    assert [h.best_fitness for h in runs[0].history] == [h.best_fitness for h in runs[1].history]
    assert runs[0].best.params == runs[1].best.params


def test_cache_avoids_reevaluation():
    calls = []

    def fitness(params, fid):
        calls.append(params["k"])
        return float(params["k"]), {}

    r = evolve({"k": Choice((1, 2, 3))}, fitness, pop_size=6,
               budget=Budget(max_generations=5, plateau_generations=99), seed=0)
    assert len(r.history) == 5
    assert len(calls) <= 3 and r.evals == len(calls)


def test_fidelity_promotes_top_fraction():
    rows = []

    def fitness(params, fid):
        rows.append(fid.rows)
        return quad(params, fid)

    evolve(SPACE, fitness, pop_size=10, budget=Budget(max_generations=1),
           fidelity=(Fidelity(0.5, 3), Fidelity()), seed=0)
    assert rows.count(0.5) == 10 and rows.count(1.0) == 3


def test_best_is_top_rung():
    r = evolve(SPACE, quad, pop_size=10, budget=Budget(max_generations=3),
               fidelity=(Fidelity(0.5, 3), Fidelity()), seed=0)
    assert r.best.rung == 1


def test_plateau_stops():
    r = evolve(SPACE, lambda p, f: (1.0, {}), pop_size=6, seed=0)
    assert r.stop_reason == "plateau" and len(r.history) == 4


def test_max_evals_stops():
    r = evolve(SPACE, quad, pop_size=10, budget=Budget(max_evals=5), seed=0)
    assert r.stop_reason == "evals" and len(r.history) == 1
    assert r.best.fitness > float("-inf")


def test_max_seconds_stops():
    def slow(params, fid):
        time.sleep(0.05)
        return quad(params, fid)

    r = evolve(SPACE, slow, pop_size=4, workers=1,
               budget=Budget(max_seconds=0.2, plateau_generations=99), seed=0)
    assert r.stop_reason == "seconds" and r.best.fitness > float("-inf")


def test_all_failing_fitness_raises():
    def broken(params, fid):
        raise KeyError("col")

    with pytest.raises(EvolveError, match="KeyError"):
        evolve(SPACE, broken, pop_size=5, seed=0)


def test_partial_failures_tolerated():
    def picky(params, fid):
        if params["x"] < 0:
            raise ValueError("negative x")
        return quad(params, fid)

    r = evolve(SPACE, picky, pop_size=10, seed=0)
    assert r.best.params["x"] >= 0 and r.best.fitness > float("-inf")


def test_seed_enters_population():
    r = evolve(SPACE, quad, pop_size=5, seeds=[{"x": 1.0, "y": -2.0}], seed=0)
    assert r.history[0].best_fitness == 0.0


def test_on_generation_each_gen():
    seen = []
    r = evolve(SPACE, quad, pop_size=6, on_generation=seen.append, seed=0)
    assert len(seen) == len(r.history) and seen == r.history
    assert all(0 < s.diversity <= 1 for s in seen)


# --- additional coverage from the controller rulings ---------------------------------


def test_invalid_inputs_raise_value_error():
    for bad in ({"pop_size": 0}, {"fidelity": ()}, {"promote": 0.0}, {"promote": 1.5},
                {"budget": Budget(max_generations=0)}):
        with pytest.raises(ValueError):
            evolve(SPACE, quad, **bad)


def test_bad_seed_raises_before_any_evaluation():
    calls = []

    def fitness(params, fid):
        calls.append(params)
        return quad(params, fid)

    with pytest.raises(ValueError, match="'bogus'"):
        evolve(SPACE, fitness, seeds=[{"bogus": 1}])
    assert calls == []


def test_mean_fitness_ignores_failed_candidates():
    def picky(params, fid):
        if params["x"] < 0:
            raise ValueError("negative x")
        return quad(params, fid)

    # one generation: a random population, about half of it failing
    r = evolve(SPACE, picky, pop_size=10, budget=Budget(max_generations=1), seed=0)
    failed = [c for c in r.population if "error" in c.metrics]
    finite = [c.fitness for c in r.population if math.isfinite(c.fitness)]
    assert failed and finite                                 # failures are recorded, not hidden
    assert all(c.fitness == float("-inf") for c in failed) and "ValueError" in failed[0].metrics["error"]
    assert r.history[0].mean_fitness == pytest.approx(sum(finite) / len(finite))


def test_promotion_takes_only_finite_candidates():
    ok0, top = set(), set()   # x of candidates that scored at rung 0 / were promoted to rung 1

    def fitness(params, fid):
        if fid.rows == 1.0:
            top.add(params["x"])
        elif params["x"] < 3.5:                  # most of the population fails at rung 0
            raise ValueError("low fidelity failure")
        else:
            ok0.add(params["x"])
        return quad(params, fid)

    evolve(SPACE, fitness, pop_size=20, budget=Budget(max_generations=1),
           fidelity=(Fidelity(0.5, 3), Fidelity()), seed=0)
    assert 0 < len(ok0) < math.ceil(0.3 * 20)    # fewer finite candidates than promotion slots
    assert top == ok0                            # every finite one was promoted, no failed one was


def test_population_is_ranked_best_first_with_failures_last():
    state = {"top_calls": 0}

    def fitness(params, fid):
        if fid.rows == 1.0:
            state["top_calls"] += 1
            if state["top_calls"] > 1:           # only the first promoted candidate survives the top rung
                raise RuntimeError("top rung failure")
        return quad(params, fid)

    r = evolve(SPACE, fitness, pop_size=10, budget=Budget(max_generations=1),
               fidelity=(Fidelity(0.5, 3), Fidelity()), seed=0)
    finite = [math.isfinite(c.fitness) for c in r.population]
    assert finite == sorted(finite, reverse=True) and not all(finite)
    assert r.population[0].rung == 1 and r.population[0].params == r.best.params
    healthy = [c for c in r.population if math.isfinite(c.fitness)]
    assert [(c.rung, c.fitness) for c in healthy] == sorted(
        [(c.rung, c.fitness) for c in healthy], reverse=True)


def test_top_rung_failing_everywhere_raises():
    def fitness(params, fid):
        if fid.rows == 1.0:
            raise RuntimeError("top rung down")
        return quad(params, fid)

    with pytest.raises(EvolveError, match="RuntimeError"):
        evolve(SPACE, fitness, pop_size=10, fidelity=(Fidelity(0.5, 3), Fidelity()), seed=0)


def test_result_bookkeeping():
    calls = []

    def fitness(params, fid):
        calls.append(1)
        return quad(params, fid)

    r = evolve(SPACE, fitness, pop_size=8, budget=Budget(max_generations=4, plateau_generations=99), seed=1)
    assert r.stop_reason == "generations" and [h.gen for h in r.history] == [1, 2, 3, 4]
    assert r.evals == len(calls) == r.history[-1].evals          # GenStats.evals is cumulative
    assert [h.evals for h in r.history] == sorted(h.evals for h in r.history)
    assert [h.seconds for h in r.history] == sorted(h.seconds for h in r.history)
    assert r.history[-1].seconds <= r.seconds
    assert len(r.population) == 8 and r.best.rung == 0
    assert r.history[-1].best_fitness == r.best.fitness and r.history[-1].best_params == r.best.params
