"""ML tools for step agents: thin wrappers that put the engine behind ADK-callable functions.

Every tool takes project-relative paths, returns a JSON-friendly dict, never raises to the model ({"ok": False,
"error": ...} instead), and appends one ledger record {"agent", "tool", "metrics", ...} so the executor's success
checks can read what it measured.
"""
from __future__ import annotations

import functools
import json
import re
from typing import Callable

import pandas as pd

from beyondml.engine.checks import detect_outliers as _outliers, drift as _drift, scan_leakage as _leakage
from beyondml.engine.profiler import DatasetProfiler
from beyondml.toolkit.env import ToolEnv

PROFILE_KEYS = ("feature_types", "missing_analysis", "target_analysis", "correlation_summary", "outlier_summary",
                "overfitting_risk_index")
CHART_KINDS = ("hist", "bar", "scatter", "box", "corr", "line")


def _jsonable(value):
    return json.loads(json.dumps(value, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def _read(env: ToolEnv, path: str) -> pd.DataFrame:
    p = env.project.path(path)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def _record(env: ToolEnv, tool: str, metrics: dict | None = None, **extra) -> None:
    env.project.ledger_append({"agent": env.agent, "tool": tool, "metrics": metrics or {}, **extra})


def _safe(fn: Callable) -> Callable:
    """Tools never raise to the model: any failure becomes {"ok": False, "error": ...}."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return wrapped


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "chart"


def _figure(kind: str, df: pd.DataFrame, x: str, y: str, title: str):
    import plotly.express as px

    if kind not in CHART_KINDS:
        raise ValueError(f"unknown chart kind {kind!r}; expected one of {CHART_KINDS}")
    for column in (x, y):
        if column and column not in df.columns:
            raise KeyError(f"no column {column!r}")
    if kind == "corr":
        return px.imshow(df.select_dtypes("number").corr(), title=title, text_auto=".2f")
    if kind == "hist":
        return px.histogram(df, x=x, color=y or None, title=title)
    if kind == "bar":
        counts = df[x].value_counts().reset_index() if not y else df.groupby(x, as_index=False)[y].mean()
        return px.bar(counts, x=x, y=y or "count", title=title)
    return {"scatter": px.scatter, "box": px.box, "line": px.line}[kind](df, x=x, y=y or None, title=title)


def ml_tools(env: ToolEnv) -> list[Callable]:
    @_safe
    def profile(path: str, target: str = "") -> dict:
        """Profile a table: column types, missing values, target analysis, correlations, outliers, overfitting risk.

        Args:
            path: Project-relative CSV or Parquet file.
            target: Target column, if known.
        """
        df = _read(env, path)
        report = DatasetProfiler(df, target_column=target or None).run()
        result = {"ok": True, "path": path, "target": target or None, "rows": int(len(df)),
                  "cols": int(df.shape[1]), **{k: report.get(k) for k in PROFILE_KEYS}}
        _record(env, "profile", {"rows": int(len(df)), "cols": int(df.shape[1])}, path=path)
        return _jsonable(result)

    @_safe
    def scan_leakage(path: str, target: str) -> dict:
        """Find columns that may leak the target (near-copies, ID-like columns, after-the-fact names).

        Args:
            path: Project-relative CSV or Parquet file.
            target: Target column.
        """
        df = _read(env, path)
        problem = DatasetProfiler(df, target_column=target).run()["target_analysis"]["target_type"]
        suspects = _leakage(df, target, problem)
        _record(env, "scan_leakage", {"suspects": len(suspects)}, path=path)
        return _jsonable({"ok": True, "problem_type": problem, "suspects": suspects})

    @_safe
    def detect_outliers(path: str, columns: list[str] = [], method: str = "iqr") -> dict:
        """Count outliers per numeric column.

        Args:
            path: Project-relative CSV or Parquet file.
            columns: Columns to check (default: every numeric column).
            method: iqr, zscore or isolation_forest (isolation_forest reports one joint verdict under "_joint").
        """
        result = _outliers(_read(env, path), columns or None, method)
        _record(env, "detect_outliers", {"flagged": sum(v["count"] for v in result.values())}, path=path)
        return _jsonable({"ok": True, "method": method, "columns": result})

    @_safe
    def drift(reference: str, current: str, alpha: float = 0.05) -> dict:
        """Compare a new batch with the reference data, column by column (KS / chi-square tests).

        Args:
            reference: Project-relative file the model was built on.
            current: Project-relative file with the new batch.
            alpha: Significance level.
        """
        result = _drift(_read(env, reference), _read(env, current), alpha)
        drifted = [c for c, r in result.items() if r["drifted"]]
        _record(env, "drift", {"drifted": len(drifted)}, reference=reference, current=current)
        return _jsonable({"ok": True, "columns": result, "drifted": drifted})

    @_safe
    def chart(kind: str, data: str, x: str, y: str = "", title: str = "") -> dict:
        """Draw a chart into reports/charts/ and show it on the agent's screen.

        Args:
            kind: hist, bar, scatter, box, corr or line.
            data: Project-relative CSV or Parquet file.
            x: Column for the x axis (ignored for corr).
            y: Optional column for the y axis (or the colour, for hist).
            title: Chart title.
        """
        title = title or " ".join(p for p in (kind, x, y) if p)
        figure = _figure(kind, _read(env, data), x, y, title)
        rel = f"reports/charts/{_slug(title)}.json"
        env.project.write(rel, figure.to_json())
        env.bus.emit(env.agent, "chart", {"title": title, "path": rel})
        _record(env, "chart", None, path=rel)
        return {"ok": True, "path": rel}

    return [profile, scan_leakage, detect_outliers, drift, chart]
