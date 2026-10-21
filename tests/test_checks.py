import numpy as np
import pandas as pd
import pytest

from beyondml.engine.checks import detect_outliers, drift, scan_leakage


def _reasons(suspects, column):
    return {s["reason"] for s in suspects if s["column"] == column}


def _frame(n=200, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.uniform(size=n), "c": rng.choice(["x", "y", "z"], n)})
    df["target"] = (df["a"] + rng.normal(scale=1.0, size=n) > 0).astype(int)
    return df


def test_scan_flags_target_copy():
    df = _frame()
    df["leak"] = df["target"] * 2 + np.random.RandomState(1).normal(scale=0.01, size=len(df))
    assert "target_correlation" in _reasons(scan_leakage(df, "target", "classification"), "leak")


def test_scan_flags_id_like():
    df = _frame()
    df["name"] = [f"passenger {i}" for i in range(len(df))]
    assert "id_like" in _reasons(scan_leakage(df, "target", "classification"), "name")


def test_scan_flags_single_feature():
    df = _frame()
    df["label_copy"] = np.where(df["target"] == 1, "yes", "no")
    assert "single_feature_score" in _reasons(scan_leakage(df, "target", "classification"), "label_copy")


def test_scan_clean_frame_is_quiet(classification_df):
    suspects = scan_leakage(classification_df, "target", "classification")
    assert all(s["reason"] == "post_outcome_name" for s in suspects)


def test_outliers_iqr_counts():
    values = list(np.random.RandomState(0).normal(size=100)) + [1000.0, -1000.0]
    result = detect_outliers(pd.DataFrame({"v": values}), method="iqr")
    assert result["v"]["count"] == 2 and set(result["v"]["sample_rows"]) == {100, 101}


def test_outliers_unknown_method():
    with pytest.raises(ValueError):
        detect_outliers(pd.DataFrame({"v": [1.0, 2.0]}), method="magic")


def test_drift_detects_shift():
    rng = np.random.RandomState(0)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 500), "k": rng.choice(["a", "b"], 500, p=[0.5, 0.5])})
    shifted = pd.DataFrame({"x": rng.normal(2, 1, 500), "k": rng.choice(["a", "b"], 500, p=[0.9, 0.1])})
    same = pd.DataFrame({"x": rng.normal(0, 1, 500), "k": rng.choice(["a", "b"], 500, p=[0.5, 0.5])})
    moved = drift(ref, shifted)
    assert moved["x"]["drifted"] and moved["k"]["drifted"] and moved["k"]["test"] == "chi2"
    assert not drift(ref, same)["x"]["drifted"]
