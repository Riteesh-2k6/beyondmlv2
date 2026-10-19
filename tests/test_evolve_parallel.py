"""Tests for the evolution engine v2 persistent worker pool (threads and processes)."""

import os
import pickle
import time

import pytest

import beyondml.engine.evolve as evolve_module
from beyondml.engine.evolve import Budget, EvolveError, Fidelity, Float, evolve

SPACE = {"x": Float(-5, 5), "y": Float(-5, 5)}


def quad(params, fid):
    return -(params["x"] - 1) ** 2 - (params["y"] + 2) ** 2, {}


class RungQuad:
    """Module-level, so a spawned worker can import and unpickle it.

    The score depends on the rung (rows), the metrics echo it, and negative x fails, so a process
    run only matches the serial one if the ladder, the results and the errors all travel intact.
    """

    def __call__(self, params, fid):
        if params["x"] < 0:
            raise ValueError("negative x")
        penalty = (1.0 - fid.rows) * 0.01 * params["x"] ** 2
        return quad(params, fid)[0] - penalty, {"rows": fid.rows}


def sig(r):
    """Everything that must be bit-identical for any worker count (wall time excluded)."""
    history = [(h.gen, h.best_fitness, h.mean_fitness, h.diversity, h.evals, h.best_params) for h in r.history]
    population = [(c.params, c.fitness, c.rung, c.metrics) for c in r.population]
    return history, population, r.best.params, r.best.fitness, r.evals, r.stop_reason


def spy_pool(monkeypatch):
    """Patch a counting ThreadPoolExecutor into the evolve module; return what it saw."""
    log = {"made": [], "closed": 0}

    class Counting(evolve_module.ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            log["made"].append(kwargs.get("max_workers", args[0] if args else None))
            super().__init__(*args, **kwargs)

        def shutdown(self, *args, **kwargs):
            log["closed"] += 1
            super().shutdown(*args, **kwargs)

    monkeypatch.setattr(evolve_module, "ThreadPoolExecutor", Counting)
    return log


def test_threads_match_serial():
    kw = dict(pop_size=8, budget=Budget(max_generations=5, plateau_generations=99), seed=5)
    serial = evolve(SPACE, quad, workers=1, **kw)
    threaded = evolve(SPACE, quad, workers=4, executor="threads", **kw)
    assert sig(threaded) == sig(serial)


@pytest.mark.parametrize("generations", [1, 3])
def test_processes_match_serial(generations):
    kw = dict(pop_size=8, budget=Budget(max_generations=generations, plateau_generations=99),
              fidelity=(Fidelity(0.5, 3), Fidelity()), seed=1)
    serial = evolve(SPACE, RungQuad(), workers=1, **kw)
    procs = evolve(SPACE, RungQuad(), workers=2, executor="processes", **kw)
    assert sig(procs) == sig(serial)
    assert procs.best.rung == 1 and procs.best.metrics == {"rows": 1.0}   # the ladder reached the workers
    if generations == 1:   # a random population: failures cross the process boundary as plain data
        assert any("ValueError('negative x')" == c.metrics.get("error") for c in procs.population)


def test_pool_created_once(monkeypatch):
    log = spy_pool(monkeypatch)
    r = evolve(SPACE, quad, pop_size=6, workers=4, budget=Budget(max_generations=4, plateau_generations=99), seed=0)
    assert len(r.history) == 4
    assert len(log["made"]) == 1 and log["closed"] == 1


def test_pool_created_once_for_a_single_worker(monkeypatch):
    log = spy_pool(monkeypatch)
    evolve(SPACE, quad, pop_size=6, workers=1, budget=Budget(max_generations=3, plateau_generations=99), seed=0)
    assert log["made"] == [1] and log["closed"] == 1


def test_pool_closed_when_the_run_raises(monkeypatch):
    log = spy_pool(monkeypatch)

    def broken(params, fid):
        raise KeyError("col")

    with pytest.raises(EvolveError, match="KeyError"):
        evolve(SPACE, broken, pop_size=4, seed=0)
    assert len(log["made"]) == 1 and log["closed"] == 1


def test_pool_size_follows_workers_and_defaults_to_cpu_count_minus_one(monkeypatch):
    log = spy_pool(monkeypatch)
    one_gen = dict(pop_size=4, budget=Budget(max_generations=1), seed=0)
    evolve(SPACE, quad, workers=3, **one_gen)
    evolve(SPACE, quad, **one_gen)
    assert log["made"] == [3, max(1, os.cpu_count() - 1)]


def test_parallel_is_faster():
    def slow(params, fid):
        time.sleep(0.1)
        return quad(params, fid)

    t0 = time.perf_counter()
    r = evolve(SPACE, slow, pop_size=8, workers=8, executor="threads", budget=Budget(max_generations=1), seed=0)
    assert r.evals == 8 and time.perf_counter() - t0 < 0.5   # serial would take 0.8 s


@pytest.mark.parametrize("workers", [1, 4])
def test_fitness_mutating_its_params_does_not_corrupt_the_loop(workers):
    def eats_params(params, fid):
        x = params.pop("x")   # the loop must hand the fitness a copy
        return -(x - 1) ** 2 - (params["y"] + 2) ** 2, {}

    kw = dict(pop_size=8, budget=Budget(max_generations=3, plateau_generations=99), seed=2)
    r = evolve(SPACE, eats_params, workers=workers, **kw)
    assert sig(r) == sig(evolve(SPACE, quad, workers=1, **kw))
    assert set(r.best.params) == {"x", "y"} and all(set(c.params) == {"x", "y"} for c in r.population)


def test_unpicklable_fitness_in_process_mode_raises_instead_of_hanging():
    with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
        evolve(SPACE, lambda p, f: quad(p, f), pop_size=4, workers=2, executor="processes", seed=0)


def test_promotion_count_keeps_the_float_epsilon():
    rows = []

    def fitness(params, fid):
        rows.append(fid.rows)
        return quad(params, fid)

    # 0.28 * 25 is 7.000000000000001 in floating point; a bare ceil() would promote 8
    evolve(SPACE, fitness, pop_size=25, workers=1, promote=0.28, budget=Budget(max_generations=1),
           fidelity=(Fidelity(0.5, 3), Fidelity()), seed=0)
    assert rows.count(0.5) == 25 and rows.count(1.0) == 7
