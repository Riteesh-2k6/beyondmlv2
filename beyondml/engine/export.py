"""Render the code a BeyondML project ships: src/pipeline.py (the exact pipeline that was scored), train.py, predict.py.

The rendered files import only the stdlib, pandas, scikit-learn and joblib (plus torch via src/mlp.py when the MLP
wins), never beyondml, so the project keeps working after BeyondML is uninstalled.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .pipeline import make_model

HEADER = '"""The pipeline BeyondML\'s search scored, written out with literals. Edit freely: train.py re-scores it."""\n'


def model_spec(params: dict, names: list[str], is_numeric: list[bool], problem_type: str,
               class_weight: str | None, target: str, raw: str) -> dict:
    return {"kind": "model", "params": dict(params), "names": list(names), "is_numeric": list(is_numeric),
            "problem_type": problem_type, "class_weight": class_weight, "target": target, "raw": raw}


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


PREPROCESS = '''def preprocessor(numeric, categorical):
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                              OneHotEncoder(handle_unknown="ignore")), categorical),
    ])
'''
BASE_IMPORTS = ["from sklearn.compose import ColumnTransformer", "from sklearn.impute import SimpleImputer",
                "from sklearn.pipeline import make_pipeline",
                "from sklearn.preprocessing import OneHotEncoder, StandardScaler"]


def render_pipeline(spec: dict) -> str:
    if spec["kind"] != "model":
        raise ValueError(f"unknown spec kind {spec['kind']!r}")
    classification = spec["problem_type"] == "classification"
    params = spec["params"]
    hp = {k: v for k, v in params.items() if k not in ("model", "features")}
    model_import, model_expr = _call(make_model(params["model"], classification, hp, spec["class_weight"]))
    numeric, categorical = _selected(params, spec["names"], spec["is_numeric"])
    imports = "\n".join(BASE_IMPORTS + [model_import])
    return (f"{HEADER}{imports}\n\n"
            f"DATA = {spec['raw']!r}\nTARGET = {spec['target']!r}\nPROBLEM = {spec['problem_type']!r}\n"
            f"FEATURES = {spec['names']!r}  # the feature frame build_pipeline() expects\n"
            f"REQUIRED = {numeric + categorical!r}  # the columns the model actually uses\n\n\n"
            f"{PREPROCESS}\n\n"
            f"def build_pipeline():\n"
            f"    return make_pipeline(preprocessor({numeric!r}, {categorical!r}),\n"
            f"                         {model_expr})\n")


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


def write_project_code(root: Path, spec: dict) -> list[str]:
    src = Path(root) / "src"
    src.mkdir(parents=True, exist_ok=True)
    files = {"src/pipeline.py": render_pipeline(spec), "src/train.py": TRAIN_PY, "src/predict.py": PREDICT_PY}
    for rel, text in files.items():
        (Path(root) / rel).write_text(text, encoding="utf-8")
    written = list(files)
    if spec.get("params", {}).get("model") == "MLP":
        shutil.copyfile(Path(__file__).with_name("mlp.py"), src / "mlp.py")
        written.append("src/mlp.py")
    return written
