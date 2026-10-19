"""Evolution engine v2: model-agnostic search space and GA loop.

Space section: Choice/Int/Float/Mask specs, and sample/mutate/crossover/complete/freeze
over flat params dicts. A conditional Choice (dict options) activates its option's
sub-space; only active keys are present in a params dict.

Loop section: evolve() runs a GA over a Space with a score cache, a fidelity ladder
(successive promotion of the best candidates to costlier evaluations) and budgets.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Choice:
    options: tuple | dict   # a dict maps option -> sub-space, active only when that option is chosen


@dataclass(frozen=True)
class Int:
    low: int
    high: int
    log: bool = False


@dataclass(frozen=True)
class Float:
    low: float
    high: float
    log: bool = False


@dataclass(frozen=True)
class Mask:
    n: int   # value: tuple of n ints in {0, 1} with at least one 1


Space = dict[str, "Choice | Int | Float | Mask"]


def _opts(spec: Choice) -> tuple:
    return tuple(spec.options)   # a dict yields its keys


def _sub(spec, value) -> dict:
    """Sub-space activated when `spec` holds `value` ({} for non-conditional or invalid)."""
    if isinstance(spec, Choice) and isinstance(spec.options, dict):
        try:
            return spec.options.get(value, {})
        except TypeError:   # unhashable value cannot be an option
            return {}
    return {}


def _specs(space: Space, params: dict) -> dict:
    """key -> spec for every key active given the choices in `params`."""
    out = {}
    for k, spec in space.items():
        out[k] = spec
        if k in params:
            out.update(_specs(_sub(spec, params[k]), params))
    return out


def _nonempty(bits, rng: random.Random) -> tuple:
    bits = list(bits)
    if not any(bits):
        bits[rng.randrange(len(bits))] = 1
    return tuple(bits)


def _fwd(spec, x):
    return math.log(x) if spec.log else x


def _inv(spec, z):
    x = math.exp(z) if spec.log else z
    x = min(max(x, spec.low), spec.high)   # exp/log round-trips can overshoot by an ulp
    return round(x) if isinstance(spec, Int) else float(x)   # float(): clipping to int bounds yields an int


def _draw(spec, rng: random.Random):
    if isinstance(spec, Choice):
        return rng.choice(_opts(spec))
    if isinstance(spec, Mask):
        return _nonempty([rng.randint(0, 1) for _ in range(spec.n)], rng)
    if isinstance(spec, Int) and not spec.log:
        return rng.randint(spec.low, spec.high)
    return _inv(spec, rng.uniform(_fwd(spec, spec.low), _fwd(spec, spec.high)))


def _perturb(spec, v, rng: random.Random, rate: float):
    if isinstance(spec, Choice):
        others = [o for o in _opts(spec) if o != v]
        return rng.choice(others) if others else v   # a single option is a fixed value
    if isinstance(spec, Mask):
        return _nonempty([b ^ (rng.random() < rate) for b in v], rng)
    sigma = 0.2 * (_fwd(spec, spec.high) - _fwd(spec, spec.low))
    return _inv(spec, _fwd(spec, v) + rng.gauss(0, sigma))


def _check(key: str, spec, v):
    """Return `v` normalised for storage, or raise ValueError naming `key`."""
    if isinstance(spec, Choice):
        ok = v in _opts(spec)
    elif isinstance(spec, Mask):
        ok = (isinstance(v, (tuple, list)) and len(v) == spec.n
              and all(b in (0, 1) for b in v) and any(v))
        v = tuple(int(b) for b in v) if ok else v
    else:
        ok = isinstance(v, (int, float)) and not isinstance(v, bool) and spec.low <= v <= spec.high
        if ok and isinstance(spec, Int):
            ok = v == int(v)
        if ok:
            v = int(v) if isinstance(spec, Int) else float(v)
    if not ok:
        raise ValueError(f"invalid value for {key!r}: {v!r}")
    return v


def _fill(space: Space, given: dict, rng: random.Random, out: dict) -> None:
    for k, spec in space.items():
        out[k] = given[k] if k in given else _draw(spec, rng)
        _fill(_sub(spec, out[k]), given, rng, out)


def _mutate(space: Space, params: dict, rng: random.Random, rate: float, out: dict) -> None:
    for k, spec in space.items():
        old = params[k]
        # a Mask applies the rate per bit inside _perturb, so it is not gated per key as well
        new = _perturb(spec, old, rng, rate) if isinstance(spec, Mask) or rng.random() < rate else old
        out[k] = new
        if new != old:   # option switched: drop the old sub-keys, sample the new sub-space
            _fill(_sub(spec, new), {}, rng, out)
        else:
            _mutate(_sub(spec, new), params, rng, rate, out)


def _cross(space: Space, a: dict, b: dict, rng: random.Random, c1: dict, c2: dict) -> None:
    for k, spec in space.items():
        if isinstance(spec, Mask):
            x, y = a[k], b[k]
            if spec.n > 1:   # one-point crossover; a 1-bit mask has no cut point
                p = rng.randint(1, spec.n - 1)
                x, y = a[k][:p] + b[k][p:], b[k][:p] + a[k][p:]
            c1[k], c2[k] = _nonempty(x, rng), _nonempty(y, rng)   # a cut can empty a child
            continue
        s1, s2 = (b, a) if rng.random() < 0.5 else (a, b)   # parent each child takes this key from
        c1[k], c2[k] = s1[k], s2[k]
        if a[k] == b[k]:
            _cross(_sub(spec, a[k]), a, b, rng, c1, c2)   # shared sub-space: swap per key
        else:   # a child keeps the sub-keys of the parent whose option it holds
            for child, src in ((c1, s1), (c2, s2)):
                for sk in _specs(_sub(spec, src[k]), src):
                    child[sk] = src[sk]


def sample(space: Space, rng: random.Random) -> dict:
    return complete(space, {}, rng)


def active_keys(space: Space, params: dict) -> set[str]:
    return set(_specs(space, params))


def mutate(space: Space, params: dict, rng: random.Random, rate: float = 0.15) -> dict:
    out: dict = {}
    _mutate(space, params, rng, rate, out)
    return out


def crossover(space: Space, a: dict, b: dict, rng: random.Random) -> tuple[dict, dict]:
    c1: dict = {}
    c2: dict = {}
    _cross(space, a, b, rng, c1, c2)
    return c1, c2


def complete(space: Space, partial: dict, rng: random.Random) -> dict:
    """Validate a seed (ValueError naming the key), then sample every missing active key."""
    active = _specs(space, partial)
    given = {}
    for k, v in partial.items():
        if k not in active:
            raise ValueError(f"unknown or inactive key {k!r}")
        given[k] = _check(k, active[k], v)
    out: dict = {}
    _fill(space, given, rng, out)
    return out


def freeze(params: dict) -> tuple:
    return tuple(sorted(params.items()))   # keys are unique, so values are never compared


# --- GA loop ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fidelity:
    rows: float = 1.0
    folds: int = 5


@dataclass(frozen=True)
class Budget:
    max_generations: int = 10
    max_evals: int | None = None
    max_seconds: float | None = None
    plateau_generations: int = 3


@dataclass
class Candidate:
    params: dict
    fitness: float = float("-inf")
    rung: int = -1
    metrics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GenStats:
    gen: int
    best_fitness: float          # the global best's fitness so far
    mean_fitness: float          # over the finite fitness values in the population
    best_params: dict
    evals: int                   # cumulative real fitness calls since evolve() started
    seconds: float               # wall seconds since evolve() started
    diversity: float             # distinct params / pop_size


@dataclass
class EvolveResult:
    best: Candidate
    history: list[GenStats]
    population: list[Candidate]  # the final generation, best first
    evals: int
    seconds: float
    stop_reason: str             # "generations" | "evals" | "seconds" | "plateau" | "trials"
    pareto: list[Candidate] = field(default_factory=list)   # filled only by tune() with several objectives


class EvolveError(RuntimeError):
    pass


Fitness = Callable[[dict, Fidelity], tuple[float, dict]]   # (score, metrics); a higher score is better

_ELITE_FRACTION = 0.1
_PLATEAU_DELTA = 1e-4


def _evaluate(fitness: Fitness, params: dict, fid: Fidelity) -> tuple[float, dict]:
    """One real fitness call. A raising fitness (or a NaN score) is a failed candidate, not a crash."""
    try:
        score, metrics = fitness(params, fid)
        score = float(score)
        if math.isnan(score):
            raise ValueError("fitness returned NaN")
        return score, dict(metrics)
    except Exception as exc:
        return float("-inf"), {"error": repr(exc)}


def _rank(c: Candidate) -> tuple:
    # a failure at a promoted rung must not outrank healthy candidates that stayed at a lower rung
    return (math.isfinite(c.fitness), c.rung, c.fitness)


def evolve(space: Space, fitness: Fitness, *, pop_size: int = 10, budget: Budget = Budget(),
           workers: int | None = None, executor: str = "threads", fidelity: Sequence[Fidelity] = (Fidelity(),),
           promote: float = 0.3, seeds: Sequence[dict] = (), on_generation: Callable[[GenStats], None] | None = None,
           seed: int = 0) -> EvolveResult:
    """Run the GA. `workers` and `executor` are accepted but the work runs serially (Task 4)."""
    if pop_size < 1:
        raise ValueError(f"pop_size must be >= 1, got {pop_size!r}")
    if len(fidelity) < 1:
        raise ValueError("fidelity must hold at least one Fidelity")
    if not 0 < promote <= 1:
        raise ValueError(f"promote must be in (0, 1], got {promote!r}")
    if budget.max_generations < 1:
        raise ValueError(f"budget.max_generations must be >= 1, got {budget.max_generations!r}")

    t0 = time.perf_counter()
    rng = random.Random(seed)
    cache: dict = {}   # (freeze(params), rung) -> (score, metrics), failures included
    top = len(fidelity) - 1
    n_promote = max(1, math.ceil(promote * pop_size - 1e-9))   # the epsilon keeps 0.7 * 10 at 7, not 8
    n_elite = max(1, int(_ELITE_FRACTION * pop_size))
    tournament = max(2, min(3, pop_size // 5))

    population = [complete(space, s, rng) for s in seeds][:pop_size]   # validates every seed up front
    while len(population) < pop_size:
        population.append(sample(space, rng))

    def score(batch: list[Candidate], rung: int) -> int:
        """Set each candidate's fitness/rung/metrics from its score at `rung`; return real calls made."""
        pending: dict = {}   # uncached unique (freeze(params), rung) keys, in population order
        for c in batch:
            key = (freeze(c.params), rung)
            if key not in cache and key not in pending:
                pending[key] = c.params
        results = [_evaluate(fitness, p, fidelity[rung]) for p in pending.values()]   # Task 4: pool.map here
        cache.update(zip(pending, results))
        for c in batch:
            c.fitness, c.metrics = cache[(freeze(c.params), rung)]
            c.rung = rung
        return len(pending)

    best: Candidate | None = None   # highest fitness ever scored at the top rung
    history: list[GenStats] = []
    evals = 0
    stalled = 0
    prev_best = float("-inf")
    for gen in range(1, budget.max_generations + 1):
        cands = [Candidate(p) for p in population]
        batch = cands
        for rung in range(top + 1):
            if rung:   # promote the best finite scorers of the previous rung
                batch = sorted((c for c in batch if math.isfinite(c.fitness)),
                               key=lambda c: c.fitness, reverse=True)[:n_promote]
            evals += score(batch, rung)
        ranked = sorted(cands, key=_rank, reverse=True)

        for c in cands:
            if c.rung == top and (best is None or c.fitness > best.fitness):
                best = Candidate(c.params, c.fitness, c.rung, c.metrics)
        if gen == 1 and (best is None or not math.isfinite(best.fitness)):
            first = next((c.metrics["error"] for c in cands if "error" in c.metrics), None)
            raise EvolveError(f"every candidate failed; first error: {first}")

        finite = [c.fitness for c in cands if math.isfinite(c.fitness)]
        elapsed = time.perf_counter() - t0
        stats = GenStats(gen, best.fitness, sum(finite) / len(finite) if finite else float("-inf"),
                         best.params, evals, elapsed, len({freeze(c.params) for c in cands}) / pop_size)
        history.append(stats)
        if on_generation is not None:
            on_generation(stats)

        stalled = 0 if best.fitness - prev_best >= _PLATEAU_DELTA else stalled + 1
        prev_best = best.fitness
        if budget.max_evals is not None and evals >= budget.max_evals:
            reason = "evals"
        elif budget.max_seconds is not None and elapsed >= budget.max_seconds:
            reason = "seconds"
        elif stalled >= budget.plateau_generations:
            reason = "plateau"
        elif gen == budget.max_generations:
            reason = "generations"
        else:
            reason = None
        if reason:
            return EvolveResult(best, history, ranked, evals, time.perf_counter() - t0, reason)

        population = [c.params for c in ranked[:n_elite]]   # elites carry over; next generation they are cache hits
        while len(population) < pop_size:
            pa, pb = (max(rng.sample(ranked, min(tournament, len(ranked))), key=_rank) for _ in range(2))
            for child in crossover(space, pa.params, pb.params, rng):
                if len(population) < pop_size:
                    population.append(mutate(space, child, rng))
