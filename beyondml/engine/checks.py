"""Deterministic data checks the step agents lean on: leakage suspects, outliers, and drift between two batches."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

POST_OUTCOME = re.compile(r"(after|post|outcome|result|final|refund|cancel)", re.I)  # weak, name-only hint
CORR_LIMIT, SINGLE_LIMIT, UNIQUE_LIMIT = 0.95, 0.98, 0.95
SAMPLE_ROWS = 5000  # single-feature probes subsample big frames


def _codes(s: pd.Series) -> pd.Series:
    """A numeric view of any column (categories become integer codes, missing becomes -1)."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float).fillna(s.median() if s.notna().any() else 0.0)
    return pd.Series(pd.factorize(s.astype("string"))[0], index=s.index).astype(float)


def _id_like(s: pd.Series) -> bool:
    values = s.dropna()
    if len(values) < 10 or values.nunique() / len(values) < UNIQUE_LIMIT:
        return False
    if not pd.api.types.is_numeric_dtype(s):
        return True  # names, ids, free text
    # an integer counter (row ids), not a continuous measurement that happens to be unique
    return pd.api.types.is_integer_dtype(s) and (values.is_monotonic_increasing or values.is_monotonic_decreasing)


def _single_feature_score(x: pd.Series, y: pd.Series, problem_type: str) -> float:
    X = _codes(x).to_frame()
    if problem_type == "classification":
        binary = y.nunique() == 2
        model, scoring = DecisionTreeClassifier(max_depth=3, random_state=0), "roc_auc" if binary else "accuracy"
        y = pd.Series(pd.factorize(y.astype("string"))[0], index=y.index)
        folds = max(2, min(3, int(y.value_counts().min())))
    else:
        model, scoring, folds = DecisionTreeRegressor(max_depth=3, random_state=0), "r2", 3
    return float(np.mean(cross_val_score(model, X, y, cv=folds, scoring=scoring)))


def scan_leakage(df: pd.DataFrame, target: str, problem_type: str) -> list[dict]:
    """Columns that look like they leak the target: [{column, reason, evidence}]."""
    data = df[df[target].notna()]
    if len(data) > SAMPLE_ROWS:
        data = data.sample(SAMPLE_ROWS, random_state=0)
    y = data[target]
    y_num = _codes(y)
    suspects = []
    for column in data.columns:
        if column == target:
            continue
        s = data[column]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 1 and y_num.nunique() > 1:
            corr = abs(float(np.corrcoef(_codes(s), y_num)[0, 1]))
            if corr >= CORR_LIMIT:
                suspects.append({"column": column, "reason": "target_correlation", "evidence": f"|corr| = {corr:.3f}"})
        if _id_like(s):
            share = s.nunique() / max(1, s.notna().sum())
            suspects.append({"column": column, "reason": "id_like", "evidence": f"{share:.0%} unique values"})
        if s.nunique() > 1 and y.nunique() > 1:
            try:
                score = _single_feature_score(s, y, problem_type)
            except ValueError:
                score = float("nan")
            if score >= SINGLE_LIMIT:
                suspects.append({"column": column, "reason": "single_feature_score",
                                 "evidence": f"alone it scores {score:.3f}"})
        if POST_OUTCOME.search(str(column)):
            suspects.append({"column": column, "reason": "post_outcome_name",
                             "evidence": "the name suggests it is recorded after the outcome"})
    return suspects


def _report(mask: pd.Series) -> dict:
    rows = mask[mask].index.tolist()
    return {"count": int(mask.sum()), "pct": round(float(mask.mean()) * 100, 2), "sample_rows": rows[:5]}


def detect_outliers(df: pd.DataFrame, columns: list[str] | None = None, method: str = "iqr") -> dict:
    """Per numeric column {count, pct, sample_rows}. isolation_forest judges the columns jointly and reports
    under the key "_joint"."""
    if method not in ("iqr", "zscore", "isolation_forest"):
        raise ValueError(f"unknown method {method!r}; expected iqr, zscore or isolation_forest")
    numeric = [c for c in (columns or df.columns) if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if method == "isolation_forest":
        if not numeric:
            return {}
        X = df[numeric].fillna(df[numeric].median())
        flags = IsolationForest(random_state=0).fit_predict(X) == -1
        return {"_joint": _report(pd.Series(flags, index=df.index))}
    out = {}
    for column in numeric:
        s = df[column]
        if method == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            mask = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
        else:
            std = s.std()
            mask = ((s - s.mean()).abs() / std > 3) if std and std > 0 else pd.Series(False, index=s.index)
        out[column] = _report(mask.fillna(False))
    return out


def drift(reference: pd.DataFrame, current: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Per shared column: KS test for numeric, chi-square on aligned value counts for categorical."""
    out = {}
    for column in [c for c in reference.columns if c in current.columns]:
        ref, cur = reference[column].dropna(), current[column].dropna()
        if len(ref) == 0 or len(cur) == 0:
            continue
        if pd.api.types.is_numeric_dtype(ref) and pd.api.types.is_numeric_dtype(cur):
            statistic, p_value = stats.ks_2samp(ref, cur)
            test = "ks"
        else:
            counts = pd.concat([ref.astype(str).value_counts(), cur.astype(str).value_counts()], axis=1).fillna(0)
            if len(counts) < 2:
                statistic, p_value = 0.0, 1.0
            else:
                statistic, p_value, _, _ = stats.chi2_contingency(counts.T.to_numpy())
            test = "chi2"
        out[column] = {"test": test, "statistic": float(statistic), "p_value": float(p_value),
                       "drifted": bool(p_value < alpha)}
    return out
