"""Small scikit-learn-compatible torch MLPs, searchable by the GA.

Self-contained on purpose: it imports only numpy, torch and scikit-learn, so export copies this file verbatim into a
shipped project as src/mlp.py.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


def _dense(X) -> np.ndarray:
    X = X.toarray() if hasattr(X, "toarray") else X
    return np.asarray(X, dtype=np.float32)


class _TorchMLP(BaseEstimator):
    def __init__(self, hidden: tuple = (64, 32), lr: float = 1e-3, epochs: int = 30, dropout: float = 0.1,
                 batch_size: int = 64, seed: int = 0):
        self.hidden, self.lr, self.epochs = hidden, lr, epochs
        self.dropout, self.batch_size, self.seed = dropout, batch_size, seed

    def _build(self, n_in: int, n_out: int) -> torch.nn.Module:
        layers, width = [], n_in
        for size in self.hidden:
            layers += [torch.nn.Linear(width, size), torch.nn.ReLU(), torch.nn.Dropout(self.dropout)]
            width = size
        return torch.nn.Sequential(*layers, torch.nn.Linear(width, n_out))

    def _train(self, X: np.ndarray, target: torch.Tensor, n_out: int, loss_fn) -> None:
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)  # the GA parallelises across candidates, not inside one
        net = self._build(X.shape[1], n_out)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        data = torch.from_numpy(X)
        gen = torch.Generator().manual_seed(self.seed)
        net.train()
        for _ in range(self.epochs):
            for idx in torch.randperm(len(data), generator=gen).split(self.batch_size):
                opt.zero_grad()
                loss_fn(net(data[idx]), target[idx]).backward()
                opt.step()
        net.eval()
        self.net_ = net

    def _forward(self, X) -> torch.Tensor:
        with torch.no_grad():
            return self.net_(torch.from_numpy(_dense(X)))


class TorchMLPClassifier(ClassifierMixin, _TorchMLP):
    def fit(self, X, y):
        self.classes_, codes = np.unique(np.asarray(y), return_inverse=True)
        self._train(_dense(X), torch.from_numpy(codes.astype(np.int64)), len(self.classes_),
                    torch.nn.CrossEntropyLoss())
        return self

    def predict_proba(self, X) -> np.ndarray:
        return torch.softmax(self._forward(X), dim=1).numpy()

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class TorchMLPRegressor(RegressorMixin, _TorchMLP):
    def fit(self, X, y):
        y = np.asarray(y, dtype=np.float32)
        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std() or 1.0)  # train on a standardised target
        scaled = torch.from_numpy(((y - self.y_mean_) / self.y_std_).reshape(-1, 1))
        self._train(_dense(X), scaled, 1, torch.nn.MSELoss())
        return self

    def predict(self, X) -> np.ndarray:
        return self._forward(X).numpy().ravel() * self.y_std_ + self.y_mean_
