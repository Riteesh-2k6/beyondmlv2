"""Render the code a BeyondML project ships: src/pipeline.py (the exact pipeline that was scored), train.py, predict.py.

The rendered files import only the stdlib, pandas, scikit-learn and joblib (plus torch via src/mlp.py when the MLP
wins), never beyondml, so the project keeps working after BeyondML is uninstalled.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .pipeline import build_ensemble, make_model

HEADER = '"""The pipeline BeyondML\'s search scored, written out with literals. Edit freely: train.py re-scores it."""\n'


def model_spec(params: dict, names: list[str], is_numeric: list[bool], problem_type: str,
               class_weight: str | None, target: str, raw: str) -> dict:
    return {"kind": "model", "params": dict(params), "names": list(names), "is_numeric": list(is_numeric),
            "problem_type": problem_type, "class_weight": class_weight, "target": target, "raw": raw}


def ensemble_spec(members: list[dict], names: list[str], is_numeric: list[bool], problem_type: str,
                  class_weight: str | None, target: str, raw: str, kind: str) -> dict:
    return {"kind": "ensemble", "method": kind, "members": [dict(m) for m in members], "names": list(names),
            "is_numeric": list(is_numeric), "problem_type": problem_type, "class_weight": class_weight,
            "target": target, "raw": raw}


# ── rendering ──────────────────────────────────────────────────────────────

def _module(cls) -> str:
    if cls.__module__.split(".")[-1] == "mlp":
        return "mlp"  # shipped next to pipeline.py as src/mlp.py
    public = []
    for part in cls.__module__.split("."):
        if part.startswith("_"):
            break
        public.append(part)
    return ".".join(public)


def _call(est) -> tuple[str, str]:
    """(import line, constructor expression) that rebuilds `est`: only parameters that differ from the defaults."""
    cls = type(est)
    defaults = cls().get_params(deep=False)
    kwargs = {k: v for k, v in est.get_params(deep=False).items() if repr(defaults.get(k)) != repr(v)}
    args = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    return f"from {_module(cls)} import {cls.__name__}", f"{cls.__name__}({args})"


def _selected(params: dict, names: list[str], is_numeric: list[bool]) -> tuple[list[str], list[str]]:
    picked = [i for i, bit in enumerate(params["features"]) if bit]
    return [names[i] for i in picked if is_numeric[i]], [names[i] for i in picked if not is_numeric[i]]


PREPROCESS = '''def preprocessor(numeric, categorical, dense=False):
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                              OneHotEncoder(handle_unknown="ignore")), categorical),
    ], sparse_threshold=0.0 if dense else 0.3)
'''
BASE_IMPORTS = ["from sklearn.compose import ColumnTransformer", "from sklearn.impute import SimpleImputer",
                "from sklearn.pipeline import make_pipeline",
                "from sklearn.preprocessing import OneHotEncoder, StandardScaler"]


def _member(params: dict, spec: dict) -> tuple[set[str], str, list[str]]:
    """(imports, `make_pipeline(preprocessor(...), Model(...))` expression, used columns) for one params dict."""
    classification = spec["problem_type"] == "classification"
    hp = {k: v for k, v in params.items() if k not in ("model", "features")}
    model_import, model_expr = _call(make_model(params["model"], classification, hp, spec["class_weight"]))
    numeric, categorical = _selected(params, spec["names"], spec["is_numeric"])
    return {model_import}, f"make_pipeline(preprocessor({numeric!r}, {categorical!r}), {model_expr})", numeric + categorical


def _ensemble(spec: dict) -> tuple[set[str], str, list[str]]:
    classification = spec["problem_type"] == "classification"
    parts = [_member(p, spec) for p in spec["members"]]
    built = build_ensemble(spec["members"], spec["names"], spec["is_numeric"], classification,
                           spec["class_weight"], spec["method"])
    name = type(built).__name__
    imports = set().union(*(p[0] for p in parts)) | {f"from sklearn.ensemble import {name}"}
    members = ",\n        ".join(f'("m{i}", {p[1]})' for i, p in enumerate(parts))
    if spec["method"] == "stacking":
        final_import, final_expr = _call(built.final_estimator)
        imports.add(final_import)
        tail = f", final_estimator={final_expr}, cv=5"
    else:
        tail = f", voting={built.voting!r}" if classification else ""
    used = [c for c in spec["names"] if any(c in p[2] for p in parts)]
    return imports, f"{name}(estimators=[\n        {members}]{tail})", used


def _module_text(spec: dict, imports: set[str], expr: str, used: list[str], extra: str = "") -> str:
    return (f"{HEADER}" + "\n".join(BASE_IMPORTS + sorted(imports - set(BASE_IMPORTS))) + "\n\n"
            f"DATA = {spec['raw']!r}\nTARGET = {spec.get('target')!r}\nPROBLEM = {spec['problem_type']!r}\n"
            f"FEATURES = {spec['names']!r}  # the feature frame build_pipeline() expects\n"
            f"REQUIRED = {used!r}  # the columns the model actually uses\n{extra}\n\n"
            f"{PREPROCESS}\n\n"
            f"def build_pipeline():\n"
            f"    return {expr}\n")


def _cluster(spec: dict) -> tuple[set[str], str, list[str]]:
    from .clustering import make_clusterer

    params, names = spec["params"], spec["names"]
    numeric = [n for n, num in zip(names, spec["is_numeric"]) if num]
    categorical = [n for n, num in zip(names, spec["is_numeric"]) if not num]
    imports, steps = set(), [f"preprocessor({numeric!r}, {categorical!r}, dense=True)"]
    if params.get("pca"):
        imports.add("from sklearn.decomposition import PCA")
        steps.append(f"PCA(n_components={min(params['pca'], len(names))!r}, random_state=0)")
    algo_import, algo_expr = _call(make_clusterer(params))
    imports.add(algo_import)
    return imports, "make_pipeline(" + ", ".join(steps + [algo_expr]) + ")", list(names)


def render_pipeline(spec: dict) -> str:
    if spec["kind"] == "model":
        imports, expr, used = _member(spec["params"], spec)
    elif spec["kind"] == "ensemble":
        imports, expr, used = _ensemble(spec)
    elif spec["kind"] == "cluster":
        imports, expr, used = _cluster(spec)
        return _module_text(spec, imports, expr, used, "ALGO = " + repr(spec["params"]["algo"]) + "\n")
    else:
        raise ValueError(f"unknown spec kind {spec['kind']!r}")
    return _module_text(spec, imports, expr, used)


TRAIN_PY = '''"""Train the model: python src/train.py (from the project root). Written by BeyondML; needs only pandas,
scikit-learn and joblib."""
import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
from pipeline import DATA, FEATURES, PROBLEM, TARGET, build_pipeline  # noqa: E402


def prepare(df):
    """The same clean -> features transforms for training and prediction."""
    if (SRC / "clean.py").exists():
        from clean import clean
        df = clean(df)
    if (SRC / "features.py").exists():
        from features import make_features
        df = make_features(df)
    return df


def main():
    df = prepare(pd.read_csv(ROOT / DATA))
    df = df[df[TARGET].notna()]
    X, y = df[FEATURES], df[TARGET]
    classification = PROBLEM == "classification"
    stratify = y if classification and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=stratify)
    score = float(build_pipeline().fit(X_train, y_train).score(X_test, y_test))
    model = build_pipeline().fit(X, y)
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models" / "model.joblib")
    metrics = {"accuracy" if classification else "r2": score, "n_train": len(X_train), "n_test": len(X_test)}
    Path(os.environ.get("BEYONDML_METRICS", "metrics.json")).write_text(json.dumps(metrics))
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
'''

PREDICT_PY = '''"""Predict with the trained model: python src/predict.py INPUT.csv [OUTPUT.csv]. Written by BeyondML."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
from pipeline import FEATURES, REQUIRED  # noqa: E402
from train import prepare  # noqa: E402


def main(argv):
    if not argv:
        print("usage: python src/predict.py INPUT.csv [OUTPUT.csv]", file=sys.stderr)
        return 2
    df = prepare(pd.read_csv(argv[0]))
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        print("missing columns: " + ", ".join(missing), file=sys.stderr)
        return 2
    for column in FEATURES:  # columns the model never uses may be absent
        if column not in df.columns:
            df[column] = np.nan
    model = joblib.load(ROOT / "models" / "model.joblib")
    out = df.copy()
    out["prediction"] = model.predict(df[FEATURES])
    target = argv[1] if len(argv) > 1 else "predictions.csv"
    out.to_csv(target, index=False)
    print(f"wrote {len(out)} predictions to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
'''


CLUSTER_TRAIN_PY = '''"""Fit the clustering: python src/train.py (from the project root). Written by BeyondML."""
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
from pipeline import ALGO, DATA, FEATURES, build_pipeline  # noqa: E402


def prepare(df):
    if (SRC / "clean.py").exists():
        from clean import clean
        df = clean(df)
    if (SRC / "features.py").exists():
        from features import make_features
        df = make_features(df)
    return df


def main():
    df = prepare(pd.read_csv(ROOT / DATA))
    X = df[FEATURES]
    model = build_pipeline()
    labels = np.asarray(model.fit_predict(X))
    Z = np.asarray(model[:-1].transform(X))
    keep = labels != -1
    n_clusters = len(set(labels.tolist()) - {-1})
    assigner = None
    if ALGO == "dbscan" and keep.any():  # DBSCAN cannot predict: new rows join their nearest core point's cluster
        assigner = KNeighborsClassifier(n_neighbors=1).fit(Z[keep], labels[keep])
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"pipeline": model, "assigner": assigner}, ROOT / "models" / "model.joblib")
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    df.assign(cluster=labels).to_csv(ROOT / "data" / "processed" / "clusters.csv", index=False)
    score = -1.0
    if n_clusters >= 2 and keep.sum() > n_clusters:
        score = float(silhouette_score(Z[keep], labels[keep], sample_size=min(2000, int(keep.sum())), random_state=0))
    metrics = {"silhouette": score, "n_clusters": n_clusters, "noise_pct": float(np.mean(~keep))}
    Path(os.environ.get("BEYONDML_METRICS", "metrics.json")).write_text(json.dumps(metrics))
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
'''

CLUSTER_PREDICT_PY = '''"""Assign clusters to new rows: python src/predict.py INPUT.csv [OUTPUT.csv]. Written by BeyondML."""
import sys
from pathlib import Path

import joblib
import pandas as pd

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
from pipeline import FEATURES, REQUIRED  # noqa: E402
from train import prepare  # noqa: E402


def main(argv):
    if not argv:
        print("usage: python src/predict.py INPUT.csv [OUTPUT.csv]", file=sys.stderr)
        return 2
    df = prepare(pd.read_csv(argv[0]))
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        print("missing columns: " + ", ".join(missing), file=sys.stderr)
        return 2
    bundle = joblib.load(ROOT / "models" / "model.joblib")
    model, assigner = bundle["pipeline"], bundle["assigner"]
    X = df[FEATURES]
    labels = assigner.predict(model[:-1].transform(X)) if assigner is not None else model.predict(X)
    target = argv[1] if len(argv) > 1 else "clusters_new.csv"
    df.assign(cluster=labels).to_csv(target, index=False)
    print(f"wrote {len(df)} cluster labels to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
'''


def write_project_code(root: Path, spec: dict) -> list[str]:
    src = Path(root) / "src"
    src.mkdir(parents=True, exist_ok=True)
    cluster = spec["kind"] == "cluster"
    files = {"src/pipeline.py": render_pipeline(spec), "src/train.py": CLUSTER_TRAIN_PY if cluster else TRAIN_PY,
             "src/predict.py": CLUSTER_PREDICT_PY if cluster else PREDICT_PY}
    for rel, text in files.items():
        (Path(root) / rel).write_text(text, encoding="utf-8")
    written = list(files)
    models = ([spec["params"]["model"]] if spec["kind"] == "model"
              else [m["model"] for m in spec.get("members", [])])
    if "MLP" in models:
        shutil.copyfile(Path(__file__).with_name("mlp.py"), src / "mlp.py")
        written.append("src/mlp.py")
    return written
