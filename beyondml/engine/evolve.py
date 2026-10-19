"""Evolution engine v2: model-agnostic search space and GA loop.

Space section: Choice/Int/Float/Mask specs, and sample/mutate/crossover/complete/freeze
over flat params dicts. A conditional Choice (dict options) activates its option's
sub-space; only active keys are present in a params dict.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


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
    return round(x) if isinstance(spec, Int) else x


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
        new = _perturb(spec, old, rng, rate) if rng.random() < rate else old
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
