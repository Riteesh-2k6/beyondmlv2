# [TICKET-L01] GA: 5-fold CV is the entire bottleneck (85.3% of total time)
parent: MAP-002
labels: wayfinder:task
status: open

## Finding

With pop=5, gen=3, the GA trainer consumed **35.86s of a 42.04s total run**.
Root cause: every genome evaluation runs `cross_val_score(..., cv=5)` plus a full
re-fit on the training set — that is **6 sklearn model fits per genome per generation**,
running via `joblib.Parallel(n_jobs=-1)` (process-based).

On a 300-row dataset with RandomForest(n_estimators default):
  - pop=5, gen=3 → 15 genome evaluations minimum  
  - 15 × 6 fits × ~0.4s/fit ≈ 36s wall-clock (single-threaded equivalent)

joblib spawns processes on Windows which adds ~1-2s fork overhead per generation.

## Fix

1. Reduce default CV folds from 5 → 3 for pop<20, keep 5 for pop≥20.
2. Adaptive early-stopping fitness cache: skip re-evaluating elite genomes (fitness != 0
   AND gen > 0) — the condition `genome.fitness == 0 or gen == 0` already exists but is
   incomplete; elites that survive crossover get reset to 0 unnecessarily.
3. Replace `Parallel(n_jobs=-1)` (multiprocessing) with `Parallel(n_jobs=-1, prefer="threads")`
   for sklearn estimators that release the GIL (most do via numpy/BLAS), eliminating
   Windows process-spawn overhead.
4. Reduce default n_estimators warm-start from 100 → 50 for the first gen; allow mutation
   to increase it. Cap at 150 for automated runs.

Expected speedup: 3-4× on the GA stage (36s → ~9-12s) for default params.
