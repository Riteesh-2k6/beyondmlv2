"""
Shared fixtures for BeyondML tests.

All fixtures use synthetic data — no network, no LLM, no external files.
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def classification_df():
    """200-row binary classification dataset with mixed types."""
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame({
        "age": rng.randint(18, 70, n),
        "income": rng.normal(50000, 15000, n).round(2),
        "score": rng.uniform(0, 100, n).round(2),
        "category": rng.choice(["A", "B", "C"], n),
        "flag": rng.choice([0, 1], n),
        "target": rng.choice([0, 1], n),
    })
    return df


@pytest.fixture
def regression_df():
    """200-row regression dataset."""
    rng = np.random.RandomState(42)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    noise = rng.normal(0, 0.5, n)
    df = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "x3": rng.uniform(-1, 1, n),
        "target": (3 * x1 + 2 * x2 + noise).round(4),
    })
    return df


@pytest.fixture
def classification_profile(classification_df):
    """Pre-built profile dict for the classification dataset."""
    from beyondml.engine.profiler import DatasetProfiler
    profiler = DatasetProfiler(classification_df, target_column="target")
    return profiler.run()


@pytest.fixture
def regression_profile(regression_df):
    """Pre-built profile dict for the regression dataset."""
    from beyondml.engine.profiler import DatasetProfiler
    profiler = DatasetProfiler(regression_df, target_column="target")
    return profiler.run()

