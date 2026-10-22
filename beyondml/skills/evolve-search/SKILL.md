---
name: evolve-search
description: "Searches models and hyperparameters with the genetic algorithm (or Optuna for expensive models). Usually right after tabular-features."
allowed-tools: read_file list_files profile evolve tune finish_step
metadata:
  beyondml:
    kind: step
    modality: tabular
---
You find the best model with BeyondML's search.

1. Call profile(path="data/processed/features.csv", target=<the target>) to see rows, columns and the problem type.
2. Choose the search: evolve (the genetic algorithm) by default; tune(strategy="tpe") only when a single model fit is expensive and you can afford fewer than 50 evaluations.
3. Choose model families: the defaults are fine for small data; add "GradientBoosting"; add "MLP" when there are at least 2,000 rows.
4. Seed the search with 2-4 sensible starting configurations, e.g. [{"model": "RandomForest", "n_estimators": 200, "max_depth": 10}, {"model": "LogisticRegression", "C": 1.0, "max_iter": 1000}].
5. Call evolve(data="data/processed/features.csv", target=<the target>, models=[...], seeds=[...]). Use minutes=<a budget> on large data.
6. Call finish_step with a summary that includes "run_id=<the run_id>" and mu_cv, outputs=[].
