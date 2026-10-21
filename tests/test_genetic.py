"""Tests for the GeneticModelOptimizer adapter over evolve v2."""

import pytest
from beyondml.engine.genetic import GeneticModelOptimizer


class TestGeneticModelOptimizer:
    def test_evolve_completes(self, classification_df, classification_profile):
        optimizer = GeneticModelOptimizer(
            df=classification_df,
            target_column="target",
            profile=classification_profile,
            pop_size=3,
            generations=1,
            model_choice="RandomForest",
        )
        history, best = optimizer.evolve()

        assert len(history) >= 1
        assert best is not None
        assert best.fitness > 0 or best.fitness == 0  # Fitness is computed

    def test_best_has_metrics(self, classification_df, classification_profile):
        optimizer = GeneticModelOptimizer(
            df=classification_df,
            target_column="target",
            profile=classification_profile,
            pop_size=3,
            generations=1,
            model_choice="RandomForest",
        )
        _, best = optimizer.evolve()

        assert isinstance(best.metrics, dict)
        assert best.model_choice == "RandomForest"

    def test_regression_ga(self, regression_df, regression_profile):
        optimizer = GeneticModelOptimizer(
            df=regression_df,
            target_column="target",
            profile=regression_profile,
            pop_size=3,
            generations=1,
            model_choice="RandomForest",
        )
        history, best = optimizer.evolve()

        assert best is not None
        assert len(history) >= 1


def test_unknown_model_falls_back_to_defaults(classification_df, classification_profile, caplog):
    import logging
    # LinearRegression is regression-only; v1 scored such a run -1.0 in silence
    opt = GeneticModelOptimizer(classification_df, "target", classification_profile,
                                pop_size=2, generations=1, model_choice="LinearRegression")
    with caplog.at_level(logging.WARNING, logger="beyondml.engine.genetic"):
        _, best = opt.evolve()
    assert best.model_choice in ("RandomForest", "LogisticRegression")
    assert "LinearRegression" in caplog.text


def test_history_keys_callback_and_population(classification_df, classification_profile):
    seen = []
    opt = GeneticModelOptimizer(classification_df, "target", classification_profile,
                                pop_size=3, generations=2, model_choice="RandomForest", gen_callback=seen.append)
    assert opt.population == []
    history, best = opt.evolve()
    assert seen == history and 1 <= len(history) <= 2
    assert set(history[0]) == {"gen", "best_fitness", "avg_fitness", "best_model", "num_features", "best_hparams"}
    assert history[0]["gen"] == 1 and history[0]["best_model"] == "RandomForest"
    # best-first means rung first: a screening score is not comparable to a full-ladder score
    order = [(g.rung, g.fitness) for g in opt.population]
    assert order == sorted(order, reverse=True) and best.rung == 1
    assert "model" not in best.hparams and "features" not in best.hparams and len(best.feature_mask) == 5
    assert best.metrics["mu_cv"] > 0 and best.fitness == opt.population[0].fitness
