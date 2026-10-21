# [TICKET-L06] GA: adaptive CV folds + n_estimators cap + fitness cache
parent: MAP-002
labels: wayfinder:task
status: superseded
superseded-by: evolution engine v2 (docs/superpowers/specs/2026-10-19-beyondml-ml-harness-design.md, section 7)
blocked-by: TICKET-L01

## Implementation detail for TICKET-L01

### CV fold reduction
```python
# In _evaluate_genome_worker:
cv_folds = 3 if len(X_sub) < 500 else 5
cv_scores = cross_val_score(pipe, X_sub, y, cv=cv_folds, ...)
```

### n_estimators cap
```python
# In Genome.__init__ for RandomForest:
"n_estimators": random.choice([10, 25, 50]) if warm_start else random.choice([10, 25, 50, 75])
# Never initialise above 75 — mutation can raise it
```

### prefer="threads" for joblib
```python
results = Parallel(n_jobs=-1, prefer="threads")(...)
```

### Fitness cache — elites preserve fitness across generations
Current condition `genome.fitness == 0 or gen == 0` resets elite fitness back to 0
after crossover. Fix: only reset fitness on genomes that were mutated or are new offspring.
```python
# Track which genomes are new this generation
new_genomes = set(id(g) for g in new_pop if g not in elite_set)
results = Parallel(n_jobs=-1, prefer="threads")(
    delayed(_evaluate_genome_worker)(genome, ...)
    for genome in self.population
    if id(genome) in new_genomes or gen == 0
)
```

## Verified 2026-10-19
- Adaptive cv_folds (3 if <500 rows else 5) missing in beyondml/engine/genetic.py _evaluate_genome_worker (cv=5 hardcoded).
- n_estimators init cap of 75 / choice([10,25,50]) missing (current choices 50..300).
- Parallel(prefer='threads') missing.
- Fitness cache for elites (evaluate only new genomes) missing.
