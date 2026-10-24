"""ML tools for step agents: thin wrappers that put the engine behind ADK-callable functions.

Every tool takes project-relative paths, returns a JSON-friendly dict, never raises to the model ({"ok": False,
"error": ...} instead), and appends one ledger record {"agent", "tool", "metrics", ...} so the executor's success
checks can read what it measured.
"""
from __future__ import annotations

import asyncio
import functools
import json
import re
import uuid
from dataclasses import asdict
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from beyondml.engine.assess import usable_rows
from beyondml.engine.checks import detect_outliers as _outliers, drift as _drift, scan_leakage as _leakage
from beyondml.engine.clustering import build_clusterer, cluster_spec, evolve_clusters
from beyondml.engine.evolve import Budget
from beyondml.engine.export import ensemble_spec, model_spec, write_project_code
from beyondml.engine.fitness import TabularFitness, default_ladder, evolve_tabular, tabular_space
from beyondml.engine.pipeline import build_ensemble, column_kinds, distinct_top
from beyondml.engine.profiler import DatasetProfiler
from beyondml.toolkit.env import ToolEnv
from beyondml.toolkit.python import run_module

SEARCH_KINDS = ("evolve", "tune")

PROFILE_KEYS = ("feature_types", "missing_analysis", "target_analysis", "correlation_summary", "outlier_summary",
                "overfitting_risk_index")
CHART_KINDS = ("hist", "bar", "scatter", "box", "corr", "line")


def _jsonable(value):
    return json.loads(json.dumps(value, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def _read(env: ToolEnv, path: str) -> pd.DataFrame:
    p = env.project.path(path)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def _need(df: pd.DataFrame, *columns: str) -> None:
    """Name the real columns when one is missing, so the model can correct a guessed name."""
    for column in columns:
        if column and column not in df.columns:
            raise KeyError(f"no column {column!r}; columns are {list(df.columns)}")


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


def _safe_async(fn: Callable) -> Callable:
    @functools.wraps(fn)
    async def wrapped(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return wrapped


def _tuples(value):
    """JSON arrives with lists where the search space holds tuples (feature masks, MLP layer sizes)."""
    if isinstance(value, list):
        return tuple(_tuples(v) for v in value)
    if isinstance(value, dict):
        return {k: _tuples(v) for k, v in value.items()}
    return value


def _labelled(env: ToolEnv, data: str, target: str) -> tuple[pd.DataFrame, int]:
    df = _read(env, data)
    if target not in df.columns:
        raise KeyError(f"no target column {target!r} in {data}")
    kept = df[df[target].notna()].reset_index(drop=True)
    return kept, len(df) - len(kept)


def _class_weight(profile: dict, classification: bool) -> str | None:
    """The same rule TabularFitness applies, recorded so export reproduces it."""
    return "balanced" if classification and profile.get("target_analysis", {}).get("imbalance_risk", False) else None


def _write_report(env: ToolEnv, kind: str, report: dict) -> str:
    run_id = uuid.uuid4().hex[:8]
    body = _jsonable({"kind": kind, "run_id": run_id, **report})
    env.project.write(f"reports/{kind}-{run_id}.json", json.dumps(body, indent=1))
    return run_id


def _load_report(env: ToolEnv, run_id: str, kinds: tuple) -> dict:
    reports = env.project.root / "reports"
    if run_id:
        found = sorted(reports.glob(f"*-{run_id}.json"))
        if not found:
            raise FileNotFoundError(f"no report for run_id {run_id!r}")
        return json.loads(found[0].read_text(encoding="utf-8"))
    found = [p for k in kinds for p in reports.glob(f"{k}-*.json")]
    if not found:
        raise FileNotFoundError("no " + "/".join(kinds) + " report yet; run one of those tools first")
    return json.loads(max(found, key=lambda p: p.stat().st_mtime_ns).read_text(encoding="utf-8"))


def _raw(env: ToolEnv, raw: str) -> str:
    if raw:
        env.project.path(raw)
        return raw
    files = [p for p in (env.project.root / "data" / "raw").iterdir() if p.is_file()]
    if len(files) != 1:
        raise ValueError(f"data/raw holds {len(files)} files; pass raw=<the one train.py should read>")
    return f"data/raw/{files[0].name}"


ASSESS_TIMEOUT = 900


async def _out_of_process(env: ToolEnv, kind: str, pipeline: str, data: str, target: str, out: str, *extra) -> dict:
    """evaluate and explain run the project's model code (which agents may edit) in a subprocess, as run_python does:
    no harness secrets in its environment, a timeout, and nothing it does can take the gateway down."""
    for rel in (pipeline, data):
        env.project.path(rel)  # refuse paths outside the project before spawning anything
    r = await asyncio.to_thread(run_module, env.project, "beyondml.engine.assess",
                                [kind, pipeline, data, target, out, *extra], ASSESS_TIMEOUT, agent=env.agent, bus=env.bus)
    if not r["ok"]:
        lines = (r.get("stderr_tail") or "").strip().splitlines()
        reason = r.get("error") or ("timed out" if r.get("timed_out") else lines[-1] if lines else f"exit {r['exit_code']}")
        raise RuntimeError(f"{kind} failed: {reason}")
    return _jsonable(json.loads(env.project.read(out)))


GENERATION_FIELDS = ("gen", "best_fitness", "mean_fitness", "diversity", "evals", "seconds", "best_params")


def _generation(env: ToolEnv) -> Callable:
    def emit(stats) -> None:
        data = asdict(stats)
        env.bus.emit(env.agent, "generation", _jsonable({k: data[k] for k in GENERATION_FIELDS}))
    return emit


def _search_report(df: pd.DataFrame, data: str, target: str, problem: str, class_weight, result) -> dict:
    names, is_numeric = column_kinds(df.drop(columns=[target]))
    best = result.best
    return {"target": target, "data": data, "problem_type": problem, "names": names, "is_numeric": is_numeric,
            "class_weight": class_weight, "best_params": best.params, "best_fitness": best.fitness,
            "mu_cv": best.metrics.get("mu_cv"),
            "population": [{"params": c.params, "fitness": c.fitness} for c in result.population],
            "history": [asdict(h) for h in result.history]}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "chart"


def _figure(kind: str, df: pd.DataFrame, x: str, y: str, title: str):
    import plotly.express as px

    if kind not in CHART_KINDS:
        raise ValueError(f"unknown chart kind {kind!r}; expected one of {CHART_KINDS}")
    _need(df, x, y)
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
        _need(df, target)
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
    def chart(kind: str, data: str, x: str = "", y: str = "", title: str = "") -> dict:
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

    @_safe_async
    async def evolve(data: str, target: str, models: list[str] = [], pop_size: int = 10, generations: int = 10,
                     minutes: float = 0.0, seeds: list[dict] = [], seed: int = 0) -> dict:
        """Search models and hyperparameters with the genetic algorithm. Streams one event per generation.

        Args:
            data: Project-relative CSV/Parquet with the features and the target (usually data/processed/features.csv).
            target: Target column.
            models: Model families to search (RandomForest, LogisticRegression, LinearRegression, SVM, DecisionTree,
                KNN, GradientBoosting, MLP); empty means the default pair.
            pop_size: Population size.
            generations: Maximum generations.
            minutes: Time budget in minutes (0 = no limit).
            seeds: Starting configurations, e.g. [{"model": "RandomForest", "n_estimators": 200}].
            seed: Random seed.
        """
        df, dropped = _labelled(env, data, target)
        profile = DatasetProfiler(df, target_column=target).run()
        problem = profile["target_analysis"]["target_type"]
        df, rare = usable_rows(df, target, problem == "classification")
        dropped += rare
        budget = Budget(max_generations=generations, max_seconds=minutes * 60 if minutes > 0 else None)
        result = await asyncio.to_thread(evolve_tabular, df, target, profile, models=models or None,
                                         pop_size=pop_size, budget=budget, seeds=[_tuples(s) for s in seeds],
                                         on_generation=_generation(env), seed=seed)
        weight = _class_weight(profile, problem == "classification")
        report = _search_report(df, data, target, problem, weight, result)
        run_id = _write_report(env, "evolve", report)
        _record(env, "evolve", {"mu_cv": report["mu_cv"], "best_fitness": report["best_fitness"]}, run_id=run_id)
        return _jsonable({"ok": True, "run_id": run_id, "best_params": report["best_params"],
                          "best_fitness": report["best_fitness"], "mu_cv": report["mu_cv"], "evals": result.evals,
                          "seconds": result.seconds, "stop_reason": result.stop_reason, "dropped_rows": dropped,
                          "problem_type": problem})

    @_safe_async
    async def tune(data: str, target: str, strategy: str = "tpe", trials: int = 50, objectives: list[str] = [],
                   seed: int = 0) -> dict:
        """Search with Optuna (tpe, nsga2 or cmaes) instead of the genetic algorithm. Streams one event per trial.

        Args:
            data: Project-relative CSV/Parquet with the features and the target.
            target: Target column.
            strategy: tpe, nsga2 or cmaes.
            trials: Number of trials.
            objectives: Optional multi-objective list such as ["fitness:max", "complexity:min"].
            seed: Random seed.
        """
        try:
            import optuna  # noqa: F401
            from beyondml.engine.tune import tune as run_tune
        except ImportError:
            return {"ok": False, "error": "Optuna is not installed: pip install 'beyondml[optuna]'"}
        df, dropped = _labelled(env, data, target)
        profile = DatasetProfiler(df, target_column=target).run()
        problem = profile["target_analysis"]["target_type"]
        df, rare = usable_rows(df, target, problem == "classification")
        dropped += rare
        X, y = df.drop(columns=[target]), df[target]

        def on_trial(stats: dict) -> None:
            env.bus.emit(env.agent, "trial", _jsonable({"number": stats["gen"], "value": stats["best_fitness"],
                                                        "params": stats["best_params"], "state": "complete"}))

        result = await asyncio.to_thread(run_tune, tabular_space(problem, X.shape[1]),
                                         TabularFitness(X, y, problem, profile), strategy=strategy, n_trials=trials,
                                         fidelity=default_ladder(len(df)), objectives=objectives, on_trial=on_trial,
                                         seed=seed)
        weight = _class_weight(profile, problem == "classification")
        report = _search_report(df, data, target, problem, weight, result)
        report["pareto"] = [{"params": c.params, "fitness": c.fitness} for c in result.pareto]
        run_id = _write_report(env, "tune", report)
        _record(env, "tune", {"mu_cv": report["mu_cv"], "best_fitness": report["best_fitness"]}, run_id=run_id)
        return _jsonable({"ok": True, "run_id": run_id, "best_params": report["best_params"],
                          "best_fitness": report["best_fitness"], "mu_cv": report["mu_cv"],
                          "pareto": report["pareto"], "dropped_rows": dropped})

    @_safe_async
    async def ensemble(run_id: str = "", top_n: int = 5, kind: str = "stacking") -> dict:
        """Combine the best distinct models of a search into a stacking or voting ensemble and score it.

        Args:
            run_id: The evolve/tune run to draw from; empty means the latest.
            top_n: How many distinct top models to combine.
            kind: stacking or voting.
        """
        report = _load_report(env, run_id, SEARCH_KINDS)
        classification = report["problem_type"] == "classification"
        df, _ = usable_rows(_labelled(env, report["data"], report["target"])[0], report["target"], classification)
        members = [_tuples(p) for p in distinct_top(report["population"], top_n)]
        model = build_ensemble(members, report["names"], report["is_numeric"], classification,
                               report["class_weight"], kind)
        cv = (StratifiedKFold if classification else KFold)(n_splits=5, shuffle=True, random_state=0)
        X, y = df[report["names"]], df[report["target"]]
        cv_score = float(np.mean(await asyncio.to_thread(cross_val_score, model, X, y, cv=cv)))
        shared = {k: report[k] for k in ("target", "data", "problem_type", "names", "is_numeric", "class_weight")}
        new_id = _write_report(env, "ensemble", {**shared, "method": kind, "members": members, "cv_score": cv_score,
                                                 "best_single": report["mu_cv"], "source": report["run_id"]})
        _record(env, "ensemble", {"cv_score": cv_score, "best_single": report["mu_cv"]}, run_id=new_id)
        return _jsonable({"ok": True, "run_id": new_id, "cv_score": cv_score, "best_single": report["mu_cv"],
                          "better": cv_score > report["mu_cv"], "members": members})

    @_safe_async
    async def cluster(data: str, algorithms: list[str] = ["kmeans", "dbscan"], pca: bool = True, pop_size: int = 10,
                      generations: int = 8) -> dict:
        """Find clusters when there is no target: evolve KMeans/DBSCAN (optionally after PCA) for the best
        silhouette. Writes data/processed/clusters.csv and streams one event per generation.

        Args:
            data: Project-relative CSV/Parquet.
            algorithms: kmeans and/or dbscan.
            pca: Allow a PCA step before clustering.
            pop_size: Population size.
            generations: Maximum generations.
        """
        df = _read(env, data)
        result = await asyncio.to_thread(evolve_clusters, df, algorithms=tuple(algorithms), pca=pca,
                                         pop_size=pop_size, generations=generations, on_generation=_generation(env))
        names, is_numeric = column_kinds(df)
        labels = build_clusterer(result.best.params, names, is_numeric).fit_predict(df)
        env.project.write("data/processed/clusters.csv", df.assign(cluster=labels).to_csv(index=False))
        metrics = result.best.metrics
        run_id = _write_report(env, "cluster", {"data": data, "names": names, "is_numeric": is_numeric,
                                                "best_params": result.best.params, "silhouette": metrics["silhouette"],
                                                "n_clusters": metrics["n_clusters"]})
        _record(env, "cluster", {"silhouette": metrics["silhouette"]}, run_id=run_id)
        return _jsonable({"ok": True, "run_id": run_id, "best_params": result.best.params, **metrics,
                          "labels": "data/processed/clusters.csv"})

    @_safe
    def export_pipeline(run_id: str = "", raw: str = "") -> dict:
        """Write src/pipeline.py (the exact pipeline the search scored), src/train.py and src/predict.py.
        Edit them afterwards if needed; train.py re-scores whatever pipeline.py builds.

        Args:
            run_id: The evolve/tune/ensemble/cluster run to export; empty means the latest.
            raw: The raw data file train.py starts from; empty means the only file in data/raw.
        """
        report = _load_report(env, run_id, SEARCH_KINDS + ("ensemble", "cluster"))
        raw = _raw(env, raw)
        if report["kind"] == "cluster":
            spec = cluster_spec(_tuples(report["best_params"]), report["names"], report["is_numeric"], raw)
        elif report["kind"] == "ensemble":
            spec = ensemble_spec([_tuples(m) for m in report["members"]], report["names"], report["is_numeric"],
                                 report["problem_type"], report["class_weight"], report["target"], raw,
                                 report["method"])
        else:
            spec = model_spec(_tuples(report["best_params"]), report["names"], report["is_numeric"],
                              report["problem_type"], report["class_weight"], report["target"], raw)
        files = write_project_code(env.project.root, spec)
        for rel in files:
            env.bus.emit(env.agent, "artifact", {"path": rel})
        if env.on_touch:
            env.on_touch(files)
        _record(env, "export_pipeline", None, run_id=report["run_id"], files=files)
        return {"ok": True, "kind": report["kind"], "run_id": report["run_id"], "files": files, "raw": raw}

    @_safe_async
    async def evaluate(data: str, target: str, pipeline: str = "src/pipeline.py", test_size: float = 0.2,
                       seed: int = 0) -> dict:
        """Hold out a test split, fit the pipeline, and report score, train/test gap, importance and health.
        health is 0 only for a near-perfect score (>= 0.999, likely leakage); a gap > 0.10 or one feature carrying
        > 50% of the importance come back as warnings to explain.

        Args:
            data: Project-relative CSV/Parquet with the features and the target.
            target: Target column.
            pipeline: The pipeline module to evaluate.
            test_size: Holdout share.
            seed: Split seed.
        """
        result = await _out_of_process(env, "evaluate", pipeline, data, target, "reports/eval.json",
                                       "--test-size", str(test_size), "--seed", str(seed))
        _record(env, "evaluate", {result["score_name"]: result["score"], "train_score": result["train_score"],
                                  "gap": result["gap"], "top_share": result["top_share"], "health": result["health"]})
        return {"ok": True, **result}

    @_safe_async
    async def explain(data: str, target: str, pipeline: str = "src/pipeline.py", top_k: int = 10) -> dict:
        """Which columns drive the model: SHAP when installed, otherwise permutation importance. Draws a chart.

        Args:
            data: Project-relative CSV/Parquet with the features and the target.
            target: Target column.
            pipeline: The pipeline module to explain.
            top_k: How many columns to report.
        """
        import plotly.express as px

        result = await _out_of_process(env, "explain", pipeline, data, target, "reports/importance.json",
                                       "--top-k", str(top_k))
        rel = "reports/charts/importance.json"
        figure = px.bar(pd.DataFrame(result["top"]), x="value", y="feature", orientation="h",
                        title="Feature importance")
        env.project.write(rel, figure.to_json())
        env.bus.emit(env.agent, "chart", {"title": "Feature importance", "path": rel})
        _record(env, "explain", {"top_share": result["top_share"]})
        return {"ok": True, **result}

    return [profile, scan_leakage, detect_outliers, drift, chart, evolve, tune, ensemble, cluster, export_pipeline,
            evaluate, explain]
