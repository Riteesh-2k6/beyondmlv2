# [TICKET-L02] Profiler called 4× serially — redundant computation
parent: MAP-002
labels: wayfinder:task
status: superseded

## Finding

`DatasetProfiler.run()` is called **4 separate times** in the pipeline runner:
1. Initial profile (before orchestrator)
2. After imputation (for EDA/outlier)
3. After outlier handling (for feature engineering)
4. After feature engineering (for GA trainer)

Each call re-computes: missing analysis, correlation matrix (O(n·f²)), outlier summary
(z-score scan), mutual information proxy, and categorical cardinality. On the 300-row
dataset this is 0.064s × 4 = ~0.25s, but on wide datasets (100+ cols) it compounds.
The ORI calculation inside `_calculate_ori` re-calls `_target_analysis`, `_outlier_summary`,
and `_correlation_summary` — doubling the work on each call.

## Fix

1. Add a `ProfileCache` in `runner.py`: store the last profile keyed by `(df.shape, df.columns.tolist(), df._fingerprint)`. Invalidate when df shape/columns change (after imputation, outlier, feature steps).
2. Partial re-profile: after outlier handling, only re-compute `outlier_summary` and `numerical_summary` (columns haven't changed). After feature engineering, re-compute everything (new columns added).
3. In `_calculate_ori`: avoid re-calling `_target_analysis()` and `_outlier_summary()` if they've already been computed in `run()` — accept them as arguments instead.

Expected speedup: eliminates ~0.2s on small datasets; 1-3s on wide datasets (50+ cols).

## Verified 2026-10-19
- No ProfileCache in beyondml/web/runner.py; DatasetProfiler is re-instantiated and run() each time.
- No partial re-profile.
- beyondml/engine/profiler.py _calculate_ori still calls _target_analysis() and _outlier_summary() itself instead of taking them as arguments.

Superseded 2026-10-21 by docs/superpowers/specs/2026-10-19-beyondml-ml-harness-design.md (core harness, sub-project 1).
