"""v1's GeneticModelOptimizer API as a thin adapter over evolution engine v2.

The GA itself lives in engine/evolve.py and the sklearn fitness in engine/fitness.py; this module keeps the
surface that agents/ga_trainer.py and engine/benchmarker.py call until sub-project 3 retires them.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from .evolve import Budget, Candidate, GenStats
from .fitness import _usable, evolve_tabular

log = logging.getLogger(__name__)

_RESERVED = ("model", "features")


def _hparams(params: dict) -> dict:
    return {k: v for k, v in params.items() if k not in _RESERVED}


@dataclass
class Genome:
    """A scored configuration; a result record, no longer a GA operator."""

    model_choice: str
    hparams: dict
    feature_mask: List[int]
    fitness: float
    metrics: dict = field(default_factory=dict)
    rung: int = -1   # fidelity rung the score came from; scores from different rungs are not comparable

    @classmethod
    def from_candidate(cls, c: Candidate) -> "Genome":
        p = c.params
        return cls(p["model"], _hparams(p), list(p["features"]), c.fitness, dict(c.metrics), c.rung)


def _round4(x: float) -> float:
    return round(x, 4) if math.isfinite(x) else x   # round(-inf) raises


class GeneticModelOptimizer:
    """GA optimizer with callback-based progress reporting for TUI integration."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str,
        profile: Dict[str, Any],
        pop_size: int = 10,
        generations: int = 5,
        model_choice: Optional[str] = None,
        gen_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.df = df
        self.target_column = target_column
        self.profile = profile
        self.pop_size = pop_size
        self.generations = generations
        self.model_choice = model_choice
        self.gen_callback = gen_callback
        self.problem_type = profile["target_analysis"]["target_type"]
        self.population: List[Genome] = []   # empty until evolve(), then sorted best-first

    def _models(self) -> Optional[Tuple[str, ...]]:
        """(model_choice,) when it fits the problem, else None (the default pair) with a warning.
        Callers pass raw LLM model names; v1 scored a bad one -1.0 in silence."""
        if self.model_choice is None:
            return None
        try:
            return _usable(self.problem_type, (self.model_choice,))
        except ValueError as exc:
            log.warning("model_choice %r unusable (%s); using the default models", self.model_choice, exc)
            return None

    def evolve(self) -> Tuple[List[Dict[str, Any]], Genome]:
        history: List[Dict[str, Any]] = []

        def on_generation(s: GenStats) -> None:
            p = s.best_params
            summary = {
                "gen": s.gen,
                "best_fitness": _round4(s.best_fitness),
                "avg_fitness": _round4(s.mean_fitness),
                "best_model": p["model"],
                "num_features": sum(p["features"]),
                "best_hparams": _hparams(p),
            }
            history.append(summary)
            if self.gen_callback:
                self.gen_callback(summary)

        result = evolve_tabular(
            self.df, self.target_column, self.profile, models=self._models(), pop_size=self.pop_size,
            budget=Budget(max_generations=self.generations), on_generation=on_generation,
        )
        self.population = [Genome.from_candidate(c) for c in result.population]
        return history, Genome.from_candidate(result.best)
