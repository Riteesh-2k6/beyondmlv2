# [TICKET-L07] EvaluatorAgent re-fits from scratch — reuse GA best genome's model
parent: MAP-002
labels: wayfinder:task
status: open

## Finding

EvaluatorAgent (1.643s) independently rebuilds the preprocessing pipeline and fits
a fresh model from scratch using `best_params`. The GA already fitted a pipeline on
the same data. This is a redundant full fit + train/test split evaluation.

## Fix

1. Have `GATrainerAgent.run()` return a `best_fitted_pipeline` (the already-fitted
   sklearn Pipeline from the winning genome's final cross_val run).
2. Pass it to `EvaluatorAgent` as `prebuilt_model`. The `prebuilt_model` path already
   exists in `EvaluatorAgent.run()` — it just short-circuits the model building step.
3. EvaluatorAgent still runs the train/test evaluation (metrics, charts) — only the
   fit is skipped.

Expected speedup: saves ~1.2s (the model build + fit portion of the 1.643s).

## Verified 2026-10-19
- GATrainerAgent.run() in beyondml/agents/ga_trainer.py returns best_params but no best_fitted_pipeline.
- Runner never passes a GA-fitted model to EvaluatorAgent as prebuilt_model (only the ensemble path sets it); the prebuilt_model parameter itself exists in evaluator_agent.py.
