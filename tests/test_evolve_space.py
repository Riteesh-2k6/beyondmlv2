"""Tests for the evolution engine v2 search-space primitives."""

import random

import pytest

from beyondml.engine.evolve import (
    Choice, Float, Int, Mask,
    active_keys, complete, crossover, freeze, mutate, sample,
)

SPACE = {"n": Int(1, 10), "lr": Float(1e-4, 1e-1, log=True), "m": Mask(5)}
COND = {"model": Choice({"rf": {"trees": Int(10, 50)}, "lr": {"C": Float(0.01, 10.0, log=True)}})}

# Shaped like the Task 5 tabular space: None/str/float options, an empty sub-space,
# a fixed single-option value, and sub-key names shared across options.
TAB = {
    "model": Choice({
        "RF": {"n_estimators": Choice((50, 100, 150)), "max_depth": Choice((None, 5, 10))},
        "LR": {"C": Float(0.01, 10.0), "max_iter": Choice((1000,))},
        "Linear": {},
        "GB": {"n_estimators": Choice((50, 100, 300)), "max_depth": Choice((3, 5, 7))},
    }),
    "features": Mask(3),
}


def test_sample_respects_bounds():
    rng = random.Random(0)
    for _ in range(200):
        p = sample(SPACE, rng)
        assert 1 <= p["n"] <= 10 and isinstance(p["n"], int)
        assert 1e-4 <= p["lr"] <= 1e-1
        assert len(p["m"]) == 5 and sum(p["m"]) >= 1


def test_conditional_only_active_keys():
    rng = random.Random(1)
    seen = set()
    for _ in range(100):
        p = sample(COND, rng)
        seen.add(p["model"])
        assert set(p) == ({"model", "trees"} if p["model"] == "rf" else {"model", "C"})
    assert seen == {"rf", "lr"}


def test_mutate_switch_drops_old_subkeys():
    rng = random.Random(2)
    p = {"model": "rf", "trees": 20}
    for _ in range(50):
        p = mutate(COND, p, rng, rate=1.0)
        assert set(p) == ({"model", "trees"} if p["model"] == "rf" else {"model", "C"})
        if p["model"] == "lr":
            assert "trees" not in p and "C" in p
            return
    pytest.fail("model never switched to lr")


def test_mask_never_empty():
    rng = random.Random(3)
    for _ in range(100):
        assert mutate({"m": Mask(1)}, {"m": (1,)}, rng, rate=1.0) == {"m": (1,)}
    for _ in range(200):
        assert sum(mutate({"m": Mask(4)}, {"m": (1, 0, 0, 0)}, rng, rate=1.0)["m"]) >= 1


def test_crossover_keeps_choice_consistency():
    rng = random.Random(4)
    a, b = {"model": "rf", "trees": 20}, {"model": "lr", "C": 1.0}
    for _ in range(50):
        c1, c2 = crossover(COND, a, b, rng)
        assert set(c1) == active_keys(COND, c1)
        assert set(c2) == active_keys(COND, c2)


def test_complete_rejects_unknown_key():
    with pytest.raises(ValueError, match=r"'bogus'"):
        complete(SPACE, {"bogus": 1}, random.Random(0))


def test_complete_rejects_out_of_range():
    with pytest.raises(ValueError, match=r"'n'"):
        complete(SPACE, {"n": 99}, random.Random(0))


def test_complete_fills_missing():
    p = complete(SPACE, {"n": 3}, random.Random(0))
    assert p["n"] == 3 and "lr" in p and "m" in p


def test_seeded_rng_is_deterministic():
    assert sample(SPACE, random.Random(7)) == sample(SPACE, random.Random(7))


def test_freeze_order_independent():
    f = freeze({"a": 1, "b": (1, 0)})
    assert f == freeze({"b": (1, 0), "a": 1})
    assert isinstance(hash(f), int)


# --- additional coverage from the controller rulings ---------------------------------


def test_single_option_choice_is_never_mutated():
    rng = random.Random(5)
    for _ in range(50):
        assert mutate({"k": Choice((1000,))}, {"k": 1000}, rng, rate=1.0) == {"k": 1000}


def test_single_option_conditional_still_mutates_subkeys():
    space = {"model": Choice({"only": {"x": Float(0.0, 1.0)}})}
    rng = random.Random(6)
    out = {mutate(space, {"model": "only", "x": 0.5}, rng, rate=1.0)["x"] for _ in range(20)}
    assert 0.5 not in out and all(0.0 <= x <= 1.0 for x in out)


def test_mask_bits_flip_at_rate_not_rate_squared():
    # v1 flipped each feature bit at 0.15; gating the whole mask on the rate as well would give ~0.2 flips
    rng = random.Random(16)
    m = (1,) * 10
    flips = sum(sum(a != b for a, b in zip(m, mutate({"m": Mask(10)}, {"m": m}, rng, rate=0.15)["m"]))
                for _ in range(2000)) / 2000
    assert 1.3 < flips < 1.7


def test_mutate_float_with_int_bounds_returns_float():
    rng = random.Random(17)
    v = 0.0   # half the gaussian steps fall below the lower bound and are clipped
    for _ in range(200):
        v = mutate({"x": Float(0, 1)}, {"x": v}, rng, rate=1.0)["x"]
        assert isinstance(v, float) and 0.0 <= v <= 1.0
    assert isinstance(sample({"x": Float(0, 1)}, rng)["x"], float)


def test_mutate_rate_zero_is_identity():
    rng = random.Random(8)
    p = sample(TAB, rng)
    assert mutate(TAB, p, rng, rate=0.0) == p


def test_mutate_numeric_stays_in_bounds():
    rng = random.Random(9)
    p = {"n": 1, "lr": 1e-4, "m": (1, 1, 1, 1, 1)}
    for _ in range(300):
        p = mutate(SPACE, p, rng, rate=1.0)
        assert 1 <= p["n"] <= 10 and isinstance(p["n"], int)
        assert 1e-4 <= p["lr"] <= 1e-1


def test_crossover_same_option_swaps_subkeys_per_key():
    a = {"model": "RF", "n_estimators": 50, "max_depth": None, "features": (1, 1, 1)}
    b = {"model": "RF", "n_estimators": 150, "max_depth": 10, "features": (1, 1, 1)}
    rng = random.Random(10)
    mixed = False
    for _ in range(100):
        c1, c2 = crossover(TAB, a, b, rng)
        assert c1["model"] == c2["model"] == "RF"
        assert {c1["n_estimators"], c2["n_estimators"]} == {50, 150}
        assert {c1["max_depth"], c2["max_depth"]} == {None, 10}
        mixed |= (c1["n_estimators"] == 50) != (c1["max_depth"] is None)
    assert mixed  # n_estimators and max_depth swap independently


def test_crossover_differing_option_never_moves_colliding_subkeys():
    a = {"model": "RF", "n_estimators": 50, "max_depth": None, "features": (1, 1, 0)}
    b = {"model": "GB", "n_estimators": 300, "max_depth": 7, "features": (0, 1, 1)}
    rng = random.Random(11)
    for _ in range(100):
        for child in crossover(TAB, a, b, rng):
            src = a if child["model"] == "RF" else b
            assert {k: v for k, v in child.items() if k != "features"} == {
                k: v for k, v in src.items() if k != "features"}
            assert set(child) == active_keys(TAB, child)


def test_crossover_outputs_are_valid_and_masks_non_empty():
    rng = random.Random(12)
    for _ in range(300):
        a, b = sample(TAB, rng), sample(TAB, rng)
        for child in crossover(TAB, a, b, rng):
            assert complete(TAB, child, rng) == child
    # two complementary masks at the only cut point would give an empty child without repair
    for _ in range(100):
        for child in crossover({"m": Mask(2)}, {"m": (1, 0)}, {"m": (0, 1)}, rng):
            assert sum(child["m"]) >= 1


def test_crossover_mask_of_one_bit():
    rng = random.Random(13)
    assert crossover({"m": Mask(1)}, {"m": (1,)}, {"m": (1,)}, rng) == ({"m": (1,)}, {"m": (1,)})


def test_sample_and_mutate_always_valid_on_tabular_space():
    rng = random.Random(14)
    p = sample(TAB, rng)
    for _ in range(300):
        p = mutate(TAB, p, rng, rate=0.5)
        assert set(p) == active_keys(TAB, p)
        assert complete(TAB, p, rng) == p


def test_empty_subspace_option():
    assert active_keys(TAB, {"model": "Linear", "features": (1, 0, 1)}) == {"model", "features"}


def test_complete_warm_start_seed_with_none_value():
    seed = {"model": "RF", "n_estimators": 100, "max_depth": None, "features": [1, 1, 1]}
    p = complete(TAB, seed, random.Random(0))
    assert p == {"model": "RF", "n_estimators": 100, "max_depth": None, "features": (1, 1, 1)}


def test_complete_samples_missing_subkeys_of_given_choice():
    p = complete(TAB, {"model": "LR"}, random.Random(0))
    assert set(p) == {"model", "C", "max_iter", "features"} and p["max_iter"] == 1000


@pytest.mark.parametrize("partial, key", [
    ({"model": "RF", "C": 1.0}, "C"),                         # key not active under the chosen option
    ({"C": 1.0}, "C"),                                        # sub-key without its choice
    ({"model": "SVM"}, "model"),                              # not an option
    ({"model": "LR", "C": 11.0}, "C"),                        # float above range
    ({"model": "LR", "C": 0.0}, "C"),                         # float below range
    ({"model": "RF", "n_estimators": 75}, "n_estimators"),    # not among options
    ({"features": (1, 1)}, "features"),                       # wrong length
    ({"features": (0, 0, 0)}, "features"),                    # all zeros
    ({"features": (1, 2, 0)}, "features"),                    # bits outside {0, 1}
    ({"features": "111"}, "features"),                        # wrong type
])
def test_complete_rejects_invalid_tabular_seeds(partial, key):
    with pytest.raises(ValueError, match=f"'{key}'"):
        complete(TAB, partial, random.Random(0))


@pytest.mark.parametrize("partial, key", [
    ({"n": 0}, "n"), ({"n": 2.5}, "n"), ({"n": "3"}, "n"), ({"n": True}, "n"),
    ({"lr": 1.0}, "lr"), ({"lr": float("nan")}, "lr"),
])
def test_complete_rejects_bad_numbers(partial, key):
    with pytest.raises(ValueError, match=f"'{key}'"):
        complete(SPACE, partial, random.Random(0))


def test_complete_accepts_integral_float_for_int_and_int_for_float():
    p = complete({"n": Int(1, 10), "x": Float(0.0, 5.0)}, {"n": 4.0, "x": 1}, random.Random(0))
    assert p == {"n": 4, "x": 1.0} and isinstance(p["n"], int) and isinstance(p["x"], float)


def test_complete_validates_before_sampling():
    rng = random.Random(15)
    state = rng.getstate()
    with pytest.raises(ValueError, match=r"'n'"):
        complete(SPACE, {"n": 99}, rng)
    assert rng.getstate() == state


def test_complete_stores_mask_list_as_tuple():
    assert complete(SPACE, {"m": [1, 0, 1, 0, 1]}, random.Random(0))["m"] == (1, 0, 1, 0, 1)


def test_freeze_handles_mixed_value_types_and_is_hashable():
    p = {"model": "RF", "max_depth": None, "C": 0.5, "n": 3, "features": (1, 0, 1)}
    f = freeze(p)
    assert f == freeze(dict(reversed(list(p.items()))))
    assert {f: 1}[freeze(dict(p))] == 1
    assert freeze({"a": 1}) != freeze({"a": 2})
