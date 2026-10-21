"""Optional Optuna backend behind the evolve() interface: tune() returns an EvolveResult.

Optuna is an extra (`pip install beyondml[optuna]`), imported only when tune() is called.
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict

from beyondml.engine.evolve import (
    Budget, Candidate, Choice, EvolveError, EvolveResult, Fidelity, Fitness, Float, GenStats, Int, Mask, Space,
    _evaluate, _opts, _sub, complete,
)

_REDUCTION = 3   # successive halving keeps ~1/3 per rung, close to evolve()'s promote=0.3


def _suggest(trial, space: Space, prefix: str, out: dict) -> None:
    """Draw every active key of `space` from `trial` into `out` (clean keys, no Optuna prefixes)."""
    for k, spec in space.items():
        name = prefix + k
        if isinstance(spec, Int):
            v = trial.suggest_int(name, spec.low, spec.high, log=spec.log)
        elif isinstance(spec, Float):
            v = trial.suggest_float(name, spec.low, spec.high, log=spec.log)
        elif isinstance(spec, Mask):
            bits = [trial.suggest_categorical(f"{name}[{i}]", [0, 1]) for i in range(spec.n)]
            if not any(bits):
                bits[0] = 1   # an all-zero mask selects nothing: set bit 0
            v = tuple(bits)
        else:
            v = trial.suggest_categorical(name, list(_opts(spec)))
        out[k] = v
        if isinstance(spec, Choice):
            _suggest(trial, _sub(spec, v), f"{prefix}{v}:", out)   # option sub-keys: "{option}:{key}"


def _flatten(space: Space, params: dict, prefix: str, out: dict) -> None:
    """Inverse of _suggest: a complete params dict -> {optuna name: value} for enqueue_trial."""
    for k, spec in space.items():
        v = params[k]
        if isinstance(spec, Mask):
            out.update({f"{prefix}{k}[{i}]": b for i, b in enumerate(v)})
        else:
            out[prefix + k] = v
        if isinstance(spec, Choice):
            _flatten(_sub(spec, v), params, f"{prefix}{v}:", out)


def _parse_objectives(objectives: Sequence[str]) -> list[tuple[str, int]]:
    """["name:max", "name:min"] -> [(name, +1 | -1)]; no entries means maximise the fitness score."""
    out = []
    for o in objectives:
        name, _, direction = o.rpartition(":")
        if not name or direction not in ("max", "min"):
            raise ValueError(f"objective {o!r} must look like 'name:max' or 'name:min'")
        out.append((name, 1 if direction == "max" else -1))
    return out or [("fitness", 1)]


def tune(space: Space, fitness: Fitness, *, strategy: str = "tpe", n_trials: int = 50, budget: Budget = Budget(),
         workers: int | None = None, fidelity: Sequence[Fidelity] = (Fidelity(),), seeds: Sequence[dict] = (),
         objectives: Sequence[str] = (), on_trial: Callable[[dict], None] | None = None,
         seed: int = 0) -> EvolveResult:
    """Optimise `space` with Optuna ("tpe", "nsga2" or "cmaes") and return the same EvolveResult as evolve().

    `n_trials` and `budget.max_seconds` bound the run (the other Budget fields are GA-only). A ladder of
    several `fidelity` rungs prunes with successive halving; a fitness that raises is a failed trial.
    `history` has one GenStats per finished (not pruned) trial. With two or more `objectives`
    ("name:max" / "name:min"; "fitness" is the score, any other name reads `metrics`) the study is
    multi-objective on the last rung only, and `pareto` lists every Pareto trial. `workers` is a thread count.
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("pip install beyondml[optuna]") from None
    if strategy not in ("tpe", "nsga2", "cmaes"):
        raise ValueError(f"strategy must be 'tpe', 'nsga2' or 'cmaes', got {strategy!r}")
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials!r}")
    if len(fidelity) < 1:
        raise ValueError("fidelity must hold at least one Fidelity")
    if workers is not None and workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers!r}")
    objs = _parse_objectives(objectives)
    multi = len(objs) > 1

    t0 = time.perf_counter()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    top = len(fidelity) - 1
    rungs = [top] if multi else range(top + 1)
    sampler = {"tpe": lambda: optuna.samplers.TPESampler(seed=seed),
               "nsga2": lambda: optuna.samplers.NSGAIISampler(seed=seed),
               "cmaes": lambda: optuna.samplers.CmaEsSampler(seed=seed, warn_independent_sampling=False)}[strategy]()
    prune = top > 0 and not multi
    pruner = (optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=_REDUCTION) if prune
              else optuna.pruners.NopPruner())
    dirs = ["maximize" if sign > 0 else "minimize" for _, sign in objs]
    study = optuna.create_study(sampler=sampler, pruner=pruner,
                                **({"direction": dirs[0]} if not multi else {"directions": dirs}))

    flat_seeds = []
    for s in seeds:   # validates every seed (ValueError naming the key) before any trial runs
        flat: dict = {}
        _flatten(space, complete(space, s, random.Random(seed)), "", flat)
        flat_seeds.append(flat)
    for flat in flat_seeds[:n_trials]:
        study.enqueue_trial(flat)   # enqueued trials run first, in order

    lock = threading.Lock()
    records: dict[int, tuple[Candidate, list[float]]] = {}   # trial number -> (candidate, objective values)
    history: list[GenStats] = []
    calls = 0
    run_best: tuple[Candidate, float] | None = None   # best finished trial by the first objective

    def values_of(score: float, metrics: dict) -> list[float]:
        if not math.isfinite(score):
            return [-sign * math.inf for _, sign in objs]   # the worst value in each direction
        out = []
        for name, sign in objs:
            try:
                v = float(score if name == "fitness" else metrics[name])
            except KeyError:
                raise EvolveError(f"fitness returned no metric {name!r} for objective {name!r}") from None
            out.append(-sign * math.inf if math.isnan(v) else v)
        return out

    def objective(trial):
        nonlocal calls
        params: dict = {}
        _suggest(trial, space, "", params)
        for r in rungs:
            score, metrics = _evaluate(fitness, params, fidelity[r])
            with lock:
                calls += 1
            values = values_of(score, metrics)
            if not math.isfinite(score):
                break   # a failed trial finishes with the worst value, not as a crash
            if r < top and prune:
                trial.report(values[0], _REDUCTION ** r)   # step = rung's promotion step (min_resource=1)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        with lock:
            records[trial.number] = (Candidate(params, score, top, metrics), values)
        return tuple(values) if multi else values[0]

    def finished(study, ft):
        nonlocal run_best
        if ft.state != optuna.trial.TrialState.COMPLETE:
            return   # pruned trials are not history entries
        with lock:
            cand, values = records[ft.number]
            key = objs[0][1] * values[0]
            if run_best is None or key > run_best[1]:
                run_best = (cand, key)
            finite = [c.fitness for c, _ in records.values() if math.isfinite(c.fitness)]
            stats = GenStats(ft.number + 1, run_best[0].fitness, sum(finite) / len(finite) if finite else -math.inf,
                             dict(run_best[0].params), calls, time.perf_counter() - t0, 1.0)
            history.append(stats)
            if on_trial is not None:
                on_trial(asdict(stats))

    n_jobs = workers or 1
    study.optimize(objective, n_trials=n_trials, timeout=budget.max_seconds, n_jobs=n_jobs, callbacks=[finished])
    if not history:   # the time budget was spent before the first trial: still evaluate one
        study.optimize(objective, n_trials=1, callbacks=[finished])
    elapsed = time.perf_counter() - t0

    if not any(math.isfinite(c.fitness) for c, _ in records.values()):
        first = next((records[n][0].metrics["error"] for n in sorted(records) if "error" in records[n][0].metrics),
                     None)
        raise EvolveError(f"every trial failed; first error: {first}")

    if multi:
        front = study.best_trials
        pareto = [records[t.number][0] for t in front]
        best = records[max(front, key=lambda t: objs[0][1] * t.values[0]).number][0]
    else:
        pareto = []
        best = records[study.best_trial.number][0]
    population = [c for c, _ in sorted(records.values(), key=lambda r: objs[0][1] * r[1][0], reverse=True)]
    n_done = sum(t.state.is_finished() for t in study.trials)
    timed_out = budget.max_seconds is not None and (elapsed >= budget.max_seconds or n_done < n_trials)
    return EvolveResult(best, history, population, calls, elapsed, "seconds" if timed_out else "trials", pareto)
