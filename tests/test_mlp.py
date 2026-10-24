import ast
import pickle
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification, make_regression

from beyondml.engine import mlp
from beyondml.engine.evolve import Budget
from beyondml.engine.fitness import evolve_tabular
from beyondml.engine.mlp import TorchMLPClassifier, TorchMLPRegressor


def test_mlp_classifier_learns_separable():
    X, y = make_classification(300, 6, n_informative=4, class_sep=2.0, random_state=0)
    labels = np.where(y == 1, "yes", "no")
    model = TorchMLPClassifier(epochs=60).fit(X, labels)
    assert (model.predict(X) == labels).mean() > 0.9
    assert set(model.predict(X)) <= {"yes", "no"} and model.predict_proba(X).shape == (300, 2)


def test_mlp_regressor_fits():
    X, y = make_regression(300, 5, noise=1.0, random_state=0)
    model = TorchMLPRegressor(epochs=200, lr=1e-2).fit(X, y)
    pred = model.predict(X)
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert r2 > 0.8


def test_mlp_is_deterministic_and_picklable():
    X, y = make_classification(200, 5, random_state=1)
    a = TorchMLPClassifier(epochs=10, seed=3).fit(X, y).predict_proba(X)
    fitted = TorchMLPClassifier(epochs=10, seed=3).fit(X, y)
    assert np.allclose(a, fitted.predict_proba(X))
    again = pickle.loads(pickle.dumps(fitted))
    assert np.allclose(again.predict_proba(X), a)


def test_mlp_imports_only_allowed():
    tree = ast.parse(Path(mlp.__file__).read_text(encoding="utf-8"))
    roots = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            roots.add((node.module or "").split(".")[0])
    assert roots <= {"numpy", "torch", "sklearn", "__future__"}


def test_evolve_tabular_with_mlp(classification_df, classification_profile):
    result = evolve_tabular(classification_df, "target", classification_profile, models=("MLP",), pop_size=3,
                            budget=Budget(max_generations=1), seed=0)
    assert result.best.params["model"] == "MLP" and result.best.fitness > -1
