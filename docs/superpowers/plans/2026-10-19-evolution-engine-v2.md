# Evolution Engine v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the v1 genetic optimiser with a generic, faster `evolve()`, plus an optional Optuna-backed `tune()`. Today's callers keep working through a thin adapter.

**Architecture:**
- `engine/evolve.py` is model-agnostic. It holds the search-space vocabulary, a seeded GA loop with a fitness cache and a fidelity ladder, and one persistent worker pool per run.
- `engine/fitness.py` holds the tabular sklearn fitness (v1's penalised CV score), the tabular search space and the warm-start seeds.
- `engine/genetic.py` shrinks to an adapter that exposes the v1 `GeneticModelOptimizer` API on top of `evolve_tabular()`. `agents/ga_trainer.py` and `engine/benchmarker.py` stay untouched until sub-project 3 retires them.

**Tech Stack:** Python, numpy, pandas, scikit-learn, stdlib `concurrent.futures` and `random`. Optuna ≥4 and cmaes ≥0.10 come only through the `beyondml[optuna]` extra.

**Spec:** `docs/superpowers/specs/2026-10-19-beyondml-ml-harness-design.md`, §7 (sub-project 2 in §12)

**Deviations from the spec:**
1. **Fitness return type.** `fitness` returns `(score, metrics)`, not a bare float. The adapter and the screens need `mu_cv`, `sigma_cv` and `gap`.
2. **Sharing data with workers.** Data reaches process workers once, through the pool initializer, rather than through memmap. It is stdlib-only and has the same once-per-worker cost.
3. **LLM seeding.** LLM-seeded populations come from the `evolve` skill in sub-project 3. The engine only takes `seeds`.

## Global Constraints

- **Interpreter:** run everything from the repo root with `venv/Scripts/python.exe` (Windows). The `python` on PATH is a different interpreter.
- **Prerequisite:** `main` contains the commit `chore: health pass and harness spec`.
- **Branch and commits:** work on branch `harness/evolve-v2`, commit after every task, never push.
- **Dependencies:** add no new required dependency. Optuna and cmaes go only under `[project.optional-dependencies] optuna`.
- **Parallelism:** fitness models are built with `n_jobs=1`. Work runs in parallel in the population pool and, for the tabular fitness, in a fixed-size joblib (loky) pool over the folds (`CV_POOL = 5`); inside the process-pool path the folds run serially so pools never nest.
- **Spec values:**
  - promotion fraction 0.3
  - `plateau_generations` default 3
  - `workers` default `max(1, os.cpu_count() - 1)`
  - threads below 20,000 rows, processes at or above
  - elitism `max(1, int(0.1 * pop_size))`
  - tournament size `max(2, min(3, pop_size // 5))`
  - mutation rate 0.15
- **Speed target** (re-stated 2026-10-21 with the owner): on the latency frame with RandomForest only, `pop_size=5` and 3 generations, the warm run (second in the same process) takes ≤5.0 s and the cold run no longer than v1's recorded cold time. v1 measured 15.6 s cold and 9.4 s warm on the 16-core dev machine (`tests/data/evolve_v1_baseline.json`).
- **Quality target:** over seeds 0, 1 and 2 on the quality frame, mean best `mu_cv` is ≥ v1's recorded mean − 0.01.

## Review Focus

1. **A fitness that raises for every candidate** (missing import, wrong column) raises `EvolveError` naming the first exception. It must not return a silent −1.0 "best"; v1 hid a `NameError` for DecisionTree and KNN exactly this way. → Task 3, `test_all_failing_fitness_raises`
2. **Seeds with an unknown key or an out-of-range value** raise `ValueError` naming the key, before any evaluation runs. → Task 2, `test_complete_rejects_*`
3. **A budget smaller than one generation** (`max_evals < pop_size`, or `max_seconds` already spent) still returns the best candidate evaluated, with `stop_reason` set. It never returns an empty `best`. → Task 3, `test_max_evals_stops`, `test_max_seconds_stops`
4. **A one-feature or all-categorical frame:** the mask never empties and the sklearn pipeline still builds. → Task 5, `test_single_feature_frame`, `test_all_categorical_frame`
5. **A class rarer than the fold count** at a low-fidelity rung: the folds drop to the smallest class count (minimum 2) instead of crashing sklearn. → Task 5, `test_rare_class_reduces_folds`

---

### Task 1: Record the v1 baseline

**Files:**
- Modify: `beyondml/engine/benchmarker.py` (add two frame builders at module level)
- Create: `scripts/bench_evolve.py`
- Create: `tests/data/evolve_v1_baseline.json` (generated)

**Interfaces:**
- **Produces:** `latency_frame() -> pd.DataFrame` in `beyondml/engine/benchmarker.py`. It is an exact copy of `make_df(300)` in `data/latency_profile.py`: seed 42, columns age/income/score/category/flag/target, 20 NaNs in income.
- **Produces:** `quality_frame() -> pd.DataFrame` in the same module: `sklearn.datasets.make_classification(n_samples=300, n_features=6, n_informative=4, n_redundant=0, random_state=0)`, with columns `f0..f5` plus `target`.
- **Produces:** the CLI `venv/Scripts/python.exe scripts/bench_evolve.py --engine v1|v2 [--record PATH] [--dataset latency|quality|adult]`. It prints JSON; `--record` also writes it to PATH. The JSON shape is:

  ```json
  {"engine": "v1", "cpu_count": 16,
   "speed": {"cold_s": 0.0, "warm_s": 0.0},
   "quality": {"seeds": [0, 1, 2], "mu_cv": [0.0, 0.0, 0.0], "mean_mu_cv": 0.0},
   "adult": {"seconds": 0.0, "time_to_best_s": 0.0, "best_mu_cv": 0.0}}
  ```

  `adult` is filled only with `--dataset adult`.
- **Run shapes:**
  - Speed: the latency frame, RandomForest only, pop 5, 3 generations, two consecutive runs (cold, then warm).
  - Quality: the quality frame, default models, pop 10, 5 generations, seeds 0, 1 and 2.
  - Adult: `data/adult.csv`, RandomForest, pop 5, 3 generations.
  - v1 is seeded with `random.seed(s); np.random.seed(s)` before each run.
  - v1's time-to-best is the moment `gen_callback` first reports the final best fitness. For v2 it is the `GenStats.seconds` of the first generation whose `best_fitness` equals the final best.

- [ ] **Step 1: Add `latency_frame()` and `quality_frame()` to `benchmarker.py`. Write `scripts/bench_evolve.py` with v1 support only. `--engine v2` exits with "not implemented yet" until Task 7.**
- [ ] **Step 2: Record the baseline.**
  Run: `venv/Scripts/python.exe scripts/bench_evolve.py --engine v1 --dataset quality --record tests/data/evolve_v1_baseline.json`
  Expected: the JSON prints, the file exists, and `speed.cold_s` is between 5 and 60 (on the dev machine, roughly 16 cold and 10 warm).
- [ ] **Step 3: Record v1 on adult in the background (it may take minutes).**
  Run: `venv/Scripts/python.exe scripts/bench_evolve.py --engine v1 --dataset adult --record data/evolve_benchmark_v1_adult.json`
  Expected: the file exists with `adult.seconds` > 0.
- [ ] **Step 4: Commit**

  ```bash
  git add beyondml/engine/benchmarker.py scripts/bench_evolve.py tests/data/evolve_v1_baseline.json data/evolve_benchmark_v1_adult.json
  git commit -m "bench: record v1 GA baseline for evolution engine v2"
  ```

---

### Task 2: Search-space primitives

**Files:**
- Create: `beyondml/engine/evolve.py` (space section only)
- Test: `tests/test_evolve_space.py`

**Interfaces:**
- **Produces**, in `beyondml/engine/evolve.py`:

  ```python
  @dataclass(frozen=True)
  class Choice:
      options: tuple | dict   # a dict maps option -> sub-space; that sub-space is active only when the option is chosen

  @dataclass(frozen=True)
  class Int:
      low: int
      high: int
      log: bool = False

  @dataclass(frozen=True)
  class Float:
      low: float
      high: float
      log: bool = False

  @dataclass(frozen=True)
  class Mask:
      n: int   # value: tuple of n ints in {0, 1} with at least one 1

  Space = dict[str, "Choice | Int | Float | Mask"]

  def sample(space: Space, rng: random.Random) -> dict
  def active_keys(space: Space, params: dict) -> set[str]
  def mutate(space: Space, params: dict, rng: random.Random, rate: float = 0.15) -> dict
  def crossover(space: Space, a: dict, b: dict, rng: random.Random) -> tuple[dict, dict]
  def complete(space: Space, partial: dict, rng: random.Random) -> dict   # validates a seed and samples any missing active keys
  def freeze(params: dict) -> tuple                                       # hashable and independent of key order
  ```

- **Params are one flat dict.** Keys from a conditional sub-space sit at the top level, and only active keys are present.
- **Mutation:** each active key mutates with probability `rate`.
  - `Int` and `Float` take a Gaussian step with σ = 0.2 × range, in log space when `log=True`, then are clipped. `Int` values are rounded.
  - A `Choice` resamples a different option. For a conditional `Choice`, the old sub-keys are dropped and the new sub-space is sampled.
  - A `Mask` flips each bit with probability `rate`, then sets one random bit if all bits are 0.
- **Crossover:** keys present in both parents are swapped per key with probability 0.5. `Mask` values use one-point crossover. When a conditional `Choice` differs between parents, each child keeps its own parent's choice and that choice's sub-keys; only the shared keys are swapped.

- [ ] **Step 1: Write the failing tests**

  ```python
  SPACE = {"n": Int(1, 10), "lr": Float(1e-4, 1e-1, log=True), "m": Mask(5)}
  COND = {"model": Choice({"rf": {"trees": Int(10, 50)}, "lr": {"C": Float(0.01, 10.0, log=True)}})}

  def test_sample_respects_bounds():          # 200 samples: 1<=n<=10, 1e-4<=lr<=1e-1, len(m)==5, sum(m)>=1
  def test_conditional_only_active_keys():    # 100 samples of COND: keys == {"model","trees"} if rf else {"model","C"}
  def test_mutate_switch_drops_old_subkeys(): # mutate({"model":"rf","trees":20}, rate=1.0) until model=="lr": "trees" absent, "C" present
  def test_mask_never_empty():                # Mask(1) mutated 100x at rate 1.0 is always (1,)
  def test_crossover_keeps_choice_consistency():  # parents rf/lr: each child's keys == active_keys(COND, child)
  def test_complete_rejects_unknown_key():    # complete(SPACE, {"bogus": 1}) raises ValueError matching "bogus"
  def test_complete_rejects_out_of_range():   # complete(SPACE, {"n": 99}) raises ValueError matching "n"
  def test_complete_fills_missing():          # complete(SPACE, {"n": 3}) keeps n==3 and has lr and m
  def test_seeded_rng_is_deterministic():     # sample(SPACE, Random(7)) twice gives equal dicts
  def test_freeze_order_independent():        # freeze({"a":1,"b":(1,0)}) == freeze({"b":(1,0),"a":1}), and it is hashable
  ```

- [ ] **Step 2: Run them and confirm they fail**
  Run: `venv/Scripts/python.exe -m pytest tests/test_evolve_space.py -q -p no:cacheprovider`
  Expected: FAIL with `ModuleNotFoundError: beyondml.engine.evolve`
- [ ] **Step 3: Implement the space section of `beyondml/engine/evolve.py` to the interface above**
- [ ] **Step 4: Run the tests and confirm they pass**
  Run: the same command. Expected: 10 passed.
- [ ] **Step 5: Commit** `feat(engine): search-space primitives for evolution v2`

---

### Task 3: The `evolve()` loop (serial)

**Files:**
- Modify: `beyondml/engine/evolve.py`
- Test: `tests/test_evolve_loop.py`

**Interfaces:**
- **Consumes:** Task 2's `sample`, `mutate`, `crossover`, `complete` and `freeze`.
- **Produces:**

  ```python
  @dataclass(frozen=True)
  class Fidelity:
      rows: float = 1.0
      folds: int = 5

  @dataclass(frozen=True)
  class Budget:
      max_generations: int = 10
      max_evals: int | None = None
      max_seconds: float | None = None
      plateau_generations: int = 3

  @dataclass
  class Candidate:
      params: dict
      fitness: float = float("-inf")
      rung: int = -1
      metrics: dict = field(default_factory=dict)

  @dataclass(frozen=True)
  class GenStats:
      gen: int
      best_fitness: float
      mean_fitness: float
      best_params: dict
      evals: int
      seconds: float
      diversity: float

  @dataclass
  class EvolveResult:
      best: Candidate
      history: list[GenStats]
      population: list[Candidate]
      evals: int
      seconds: float
      stop_reason: str                                     # "generations" | "evals" | "seconds" | "plateau" | "trials"
      pareto: list[Candidate] = field(default_factory=list)   # filled only by tune() with several objectives

  class EvolveError(RuntimeError): ...

  Fitness = Callable[[dict, Fidelity], tuple[float, dict]]   # (score, metrics); a higher score is better

  def evolve(space: Space, fitness: Fitness, *, pop_size: int = 10, budget: Budget = Budget(),
             workers: int | None = None, executor: str = "threads", fidelity: Sequence[Fidelity] = (Fidelity(),),
             promote: float = 0.3, seeds: Sequence[dict] = (), on_generation: Callable[[GenStats], None] | None = None,
             seed: int = 0) -> EvolveResult
  ```

  In this task, `workers` and `executor` are accepted but the work runs serially. Task 4 makes them real.

**Algorithm** (the signature doesn't determine it, so it is fixed here):

```
rng = random.Random(seed); cache = {}                  # key: (freeze(params), rung) -> (score, metrics)
population = [complete(space, s, rng) for s in seeds][:pop_size], then fill to pop_size with sample(space, rng)
for gen in 1..budget.max_generations:
    score every candidate at rung 0 (cache hits are free and not counted in evals)
    for r in 1..len(fidelity)-1:
        take the top ceil(promote * pop_size) candidates by their rung r-1 score and score them at rung r
    each candidate's rung = the highest rung it was scored at; its fitness = its score at that rung
    rank the population by (rung, fitness), descending
    global best = the highest-fitness candidate ever scored at the top rung, len(fidelity)-1
    diversity = len({freeze(c.params) for c in population}) / pop_size
    emit GenStats (best_fitness is the global best's fitness)
    stop if evals >= max_evals ("evals"); else if elapsed >= max_seconds ("seconds");
         else if the global best has improved by < 1e-4 for plateau_generations generations in a row ("plateau");
         else if gen == max_generations ("generations")
    next population = the top elites by rank,
        then children: tournament-select two parents -> crossover -> mutate each, until pop_size is reached
```

- **Errors:** if `fitness` raises, that candidate gets fitness `-inf` and `metrics={"error": repr(exc)}`. If every candidate in generation 1 failed, raise `EvolveError(f"every candidate failed; first error: {first!r}")`.
- **Overshoot:** budgets are checked between generations, so `max_evals` and `max_seconds` can overshoot by at most one generation.

- [ ] **Step 1: Write the failing tests.** Use the quadratic fitness `-(x-1)**2 - (y+2)**2` over `{"x": Float(-5, 5), "y": Float(-5, 5)}` unless a test says otherwise.

  ```python
  def test_converges_on_quadratic():        # pop 20, Budget(max_generations=15, plateau_generations=99), seed 0: |x-1|<0.5, |y+2|<0.5
  def test_same_seed_same_history():        # two runs with seed 3: identical [h.best_fitness for h in history] and identical best.params
  def test_cache_avoids_reevaluation():     # space {"k": Choice((1,2,3))}, pop 6, 5 generations: fitness call count <= 3
  def test_fidelity_promotes_top_fraction():# ladder (Fidelity(0.5,3), Fidelity()), pop 10, 1 generation, unique params: 10 calls at rung 0, 3 at rung 1
  def test_best_is_top_rung():              # with that ladder: result.best.rung == 1
  def test_plateau_stops():                 # constant fitness: stop_reason == "plateau", len(history) == 4
  def test_max_evals_stops():               # Budget(max_evals=5), pop 10: stop_reason == "evals", len(history) == 1, best.fitness > -inf
  def test_max_seconds_stops():             # fitness sleeps 0.05 s, Budget(max_seconds=0.2), pop 4: stop_reason == "seconds", best.fitness > -inf
  def test_all_failing_fitness_raises():    # fitness raises KeyError("col"): pytest.raises(EvolveError, match="KeyError")
  def test_partial_failures_tolerated():    # fitness raises when x < 0: completes, best.params["x"] >= 0
  def test_seed_enters_population():        # seeds=[{"x": 1.0, "y": -2.0}]: history[0].best_fitness == 0.0
  def test_on_generation_each_gen():        # collected GenStats: len == len(history), 0 < diversity <= 1
  ```

- [ ] **Step 2: Run them and confirm they fail**
  Run: `venv/Scripts/python.exe -m pytest tests/test_evolve_loop.py -q -p no:cacheprovider`
  Expected: FAIL with `ImportError: cannot import name 'evolve'`
- [ ] **Step 3: Implement the dataclasses, `EvolveError` and the serial `evolve()` following the algorithm above**
- [ ] **Step 4: Run Task 2's and Task 3's tests and confirm they pass**
  Run: `venv/Scripts/python.exe -m pytest tests/test_evolve_space.py tests/test_evolve_loop.py -q -p no:cacheprovider`
  Expected: 22 passed.
- [ ] **Step 5: Commit** `feat(engine): evolve() loop with cache, fidelity ladder and budgets`

---

### Task 4: A persistent parallel pool

**Files:**
- Modify: `beyondml/engine/evolve.py`
- Test: `tests/test_evolve_parallel.py`

**Interfaces:**
- **Consumes:** Task 3's `evolve()`.
- **Produces:** `workers` and `executor` now take effect.
  - **Pool lifetime:** each `evolve()` call creates exactly one `ThreadPoolExecutor` or `ProcessPoolExecutor` with `max_workers=workers or max(1, os.cpu_count() - 1)` and closes it when the call returns.
  - **Process mode:** the pool gets `initializer=_install_fitness, initargs=(fitness,)`, so the fitness and its data are pickled once per worker. Tasks send only `(params, rung_index)`, and the ladder travels inside the initializer args.
  - **Determinism:** results are collected in submission order, so the outcome does not depend on `workers`.

- [ ] **Step 1: Write the failing tests**

  ```python
  def test_threads_match_serial():       # seed 5: workers=4 executor="threads" gives the same history and best.params as workers=1
  def test_processes_match_serial():     # executor="processes", workers=2, with a module-level picklable fitness class: same as serial
  def test_pool_created_once(monkeypatch):  # a counting ThreadPoolExecutor subclass is patched into the evolve module:
                                         #   constructed exactly once for a 4-generation run
  def test_parallel_is_faster():         # fitness sleeps 0.1 s, pop 8, 1 generation, workers=8 threads: wall < 0.5 s
  ```

- [ ] **Step 2: Run them and confirm they fail.** Run: `venv/Scripts/python.exe -m pytest tests/test_evolve_parallel.py -q -p no:cacheprovider`. Expected: `test_pool_created_once` and `test_parallel_is_faster` FAIL.
- [ ] **Step 3: Implement the pool as described in Interfaces**
- [ ] **Step 4: Run all evolve tests and confirm they pass.** Run: `venv/Scripts/python.exe -m pytest tests/test_evolve_space.py tests/test_evolve_loop.py tests/test_evolve_parallel.py -q -p no:cacheprovider`. Expected: 26 passed.
- [ ] **Step 5: Commit** `perf(engine): one persistent worker pool per evolve run`

---

### Task 5: Tabular fitness, space, seeds and `evolve_tabular`

**Files:**
- Create: `beyondml/engine/fitness.py`
- Test: `tests/test_fitness_tabular.py`

**Interfaces:**
- **Consumes:** Task 3's `evolve`, `Fidelity`, `Budget`, `EvolveResult`, `Choice`, `Int`, `Float` and `Mask`.
- **Produces:**

  ```python
  MODELS = ("RandomForest", "LogisticRegression", "LinearRegression", "SVM", "DecisionTree", "KNN", "GradientBoosting")
  DEFAULT_MODELS = {"classification": ("RandomForest", "LogisticRegression"),
                    "regression": ("RandomForest", "LinearRegression")}   # the same pair v1 picked from

  def tabular_space(problem_type: str, n_features: int, models: Sequence[str] | None = None) -> Space
      # {"model": Choice({name: subspace}), "features": Mask(n_features)}.
      # The subspaces use v1's random-init ranges from Genome._init_hparams.
      # LogisticRegression is classification-only and LinearRegression is regression-only; the others serve both.
      # ValueError if no model in `models` fits the problem type.
  def warm_seeds(problem_type: str, n_features: int, models: Sequence[str] | None = None) -> list[dict]
      # v1's warm-start defaults for each model, with every feature on
  def default_ladder(n_rows: int) -> tuple[Fidelity, ...]
      # n_rows < 5000: (Fidelity(1.0, 3, effort=0.2), Fidelity(1.0, 5)); otherwise (Fidelity(0.25, 3, effort=0.2), Fidelity(1.0, 5))
      # effort < 1 fits ensembles with that fraction of their trees (at least 10); the penalty still charges the full count

  class TabularFitness:   # picklable
      def __init__(self, X: pd.DataFrame, y: pd.Series, problem_type: str, profile: dict) -> None
      def __call__(self, params: dict, fidelity: Fidelity) -> tuple[float, dict]
          # metrics keys: mu_cv, sigma_cv, gap, complexity, n_rows, folds

  def evolve_tabular(df: pd.DataFrame, target: str, profile: dict, *, models: Sequence[str] | None = None,
                     pop_size: int = 10, budget: Budget = Budget(), workers: int | None = None,
                     seeds: Sequence[dict] = (), on_generation=None, seed: int = 0) -> EvolveResult
      # executor = "processes" if len(df) >= 20_000 else "threads"; fidelity = default_ladder(len(df));
      # seeds = list(seeds) + warm_seeds(...); problem_type = profile["target_analysis"]["target_type"]
  ```

- **The fitness formula is v1's, copied term for term** from `engine/genetic.py` `_evaluate_genome_worker`: the preprocessing pipeline, `class_weight` when there is imbalance risk, the complexity penalty `c_p`, and the lambdas `l1 = 0.02`, `l2 = 0.10`, `l3 = 0.50`, each scaled by `(1 + ORI)`. Four changes:
  1. **Train scores** come from `sklearn.model_selection.cross_validate(..., return_train_score=True)`, with `gap = max(0, mean(train) - mu_cv)`. This removes v1's extra full fit.
  2. **Row subsampling:** `fidelity.rows < 1` subsamples with `random_state=0`, stratified for classification.
  3. **Folds:** `StratifiedKFold` or `KFold` with `shuffle=True` and `random_state=0`. For classification, `n_splits = max(2, min(fidelity.folds, smallest class count))`. If the smallest class has fewer than 2 rows, fall back to `KFold`.
  4. **Imports:** import `DecisionTreeClassifier`/`DecisionTreeRegressor` and `KNeighborsClassifier`/`KNeighborsRegressor`. v1 never imported them.

- [ ] **Step 1: Write the failing tests.** They use the `classification_df`, `regression_df` and matching `*_profile` fixtures from `tests/conftest.py`.

  ```python
  @pytest.mark.parametrize("model", ["RandomForest","LogisticRegression","SVM","DecisionTree","KNN","GradientBoosting"])
  def test_every_classifier_scores(model, ...):   # TabularFitness(...)(warm seed for model, Fidelity(1.0, 3)): fitness > -1, metrics keys as listed
  @pytest.mark.parametrize("model", ["RandomForest","LinearRegression","SVM","DecisionTree","KNN","GradientBoosting"])
  def test_every_regressor_scores(model, ...):
  def test_formula_matches_v1_terms(...):         # RF warm seed: fitness == approx(mu_cv - l1*complexity - l2*sigma_cv - l3*gap, abs=1e-9),
                                                  #   with the lambdas recomputed from profile["overfitting_risk_index"]["score"]
  def test_low_fidelity_uses_fewer_rows(...):     # Fidelity(rows=0.5) on 200 rows: metrics["n_rows"] == 100
  def test_rare_class_reduces_folds():            # 60 rows where one class has 3 rows, Fidelity(folds=5): metrics["folds"] == 3
  def test_single_feature_frame():                # columns x, target; pop_size=4, Budget(max_generations=2): best.params["features"] == (1,)
  def test_all_categorical_frame():               # two string columns plus target: best.fitness > -1
  def test_warm_seed_values():                    # RF n_estimators == 100, max_depth is None; LR C == 1.0, max_iter == 1000
  def test_warm_seed_evaluated_first_gen():       # pop_size=1, models=("RandomForest",), Budget(max_generations=1): best.params["n_estimators"] == 100
  def test_executor_by_rows(monkeypatch):         # evolve patched to capture its kwargs: "threads" at 200 rows, "processes" at 20_000 rows
  def test_space_rejects_wrong_problem_type():    # tabular_space("regression", 3, models=("LogisticRegression",)) raises ValueError
  ```

- [ ] **Step 2: Run them and confirm they fail.** Run: `venv/Scripts/python.exe -m pytest tests/test_fitness_tabular.py -q -p no:cacheprovider`. Expected: FAIL with `ModuleNotFoundError: beyondml.engine.fitness`.
- [ ] **Step 3: Implement `beyondml/engine/fitness.py` to the interface and formula above**
- [ ] **Step 4: Run the tests and confirm they pass.** Expected: 21 passed.
- [ ] **Step 5: Commit** `feat(engine): tabular fitness, search space and evolve_tabular`

---

### Task 6: v1 adapter

**Files:**
- Modify: `beyondml/engine/genetic.py` (replace the whole body)
- Modify: `tests/test_genetic.py` (delete `class TestGenome`)
- Delete: `tests/test_warm_start.py` (Task 5's `test_warm_seed_*` covers it)

**Interfaces:**
- **Consumes:** Task 5's `evolve_tabular` and Task 3's `Budget` and `GenStats`.
- **Produces:** the v1 surface that `agents/ga_trainer.py` and `engine/benchmarker.py` use:

  ```python
  @dataclass
  class Genome:                # now only a result record, no longer a GA operator
      model_choice: str
      hparams: dict            # params minus "model" and "features"
      feature_mask: list[int]
      fitness: float
      metrics: dict

  class GeneticModelOptimizer:
      def __init__(self, df, target_column, profile, pop_size=10, generations=5, model_choice=None, gen_callback=None): ...
      population: list[Genome]                  # empty before evolve(), sorted best-first after it
      def evolve(self) -> tuple[list[dict], Genome]
          # history dicts keep v1's keys: gen, best_fitness, avg_fitness, best_model, num_features, best_hparams
          # gen_callback receives each such dict
          # model_choice=None -> models=None (DEFAULT_MODELS); otherwise models=(model_choice,)
          # budget = Budget(max_generations=generations)
  ```

- [ ] **Step 1: Delete `TestGenome` and `tests/test_warm_start.py`. Run the remaining GA tests to confirm they still pass against v1 before the rewrite.** Run: `venv/Scripts/python.exe -m pytest tests/test_genetic.py -q -p no:cacheprovider`. Expected: 3 passed.
- [ ] **Step 2: Replace `genetic.py` with the adapter**
- [ ] **Step 3: Run the full suite.** Run: `venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider`. Expected: 0 failed. `test_genetic.py`, `test_agents.py` (GA trainer paths) and `test_pipeline_integration.py` all pass.
- [ ] **Step 4: Commit** `refactor(engine): GeneticModelOptimizer becomes an adapter over evolve v2`

---

### Task 7: Speed and quality targets as tests

**Files:**
- Create: `tests/test_evolve_bench.py`
- Modify: `scripts/bench_evolve.py` (add `--engine v2`)
- Create: `data/evolve_benchmark_v2_adult.json` (generated)
- Modify: `.scratch/TICKET-L01-ga-cv-cost.md`, `.scratch/TICKET-L06-ga-cv-adaptive.md` (status line only)

**Interfaces:**
- **Consumes:** `latency_frame` and `quality_frame` (Task 1), `evolve_tabular` and `Budget` (Tasks 3 and 5), and `tests/data/evolve_v1_baseline.json`.

- [ ] **Step 1: Write the tests**

  ```python
  def test_latency_frame_speed():
      # run `scripts/bench_evolve.py --engine v2 --dataset latency` in a fresh subprocess (so "cold" is cold)
      # and read its JSON; baseline = tests/data/evolve_v1_baseline.json
      assert speed["warm_s"] <= 5.0 and speed["cold_s"] <= baseline["speed"]["cold_s"]

  def test_quality_not_worse_than_v1():
      # baseline = json.load(open("tests/data/evolve_v1_baseline.json"))
      # mu = [evolve_tabular(quality_frame(), "target", profile, pop_size=10,
      #                      budget=Budget(max_generations=5), seed=s).best.metrics["mu_cv"] for s in (0, 1, 2)]
      assert sum(mu) / 3 >= baseline["quality"]["mean_mu_cv"] - 0.01
  ```

- [ ] **Step 2: Run them.** Run: `venv/Scripts/python.exe -m pytest tests/test_evolve_bench.py -q -p no:cacheprovider`. Expected: 2 passed. If the speed test fails, profile with `python -X importtime` and `cProfile` before changing the targets. The targets come from the spec and may only change with the owner's agreement.
- [ ] **Step 3: Add `--engine v2` to the bench script and run it on adult in the background.** Run: `venv/Scripts/python.exe scripts/bench_evolve.py --engine v2 --dataset adult --record data/evolve_benchmark_v2_adult.json`. Expected: the file exists. Put v1's and v2's `adult.seconds` and `time_to_best_s` in the commit message.
- [ ] **Step 4: Set `status: superseded` in the L01 and L06 tickets, with one line pointing to the spec, §7.**
- [ ] **Step 5: Commit** `test(engine): evolution v2 speed and quality targets; adult benchmark`

---

### Task 8: Optuna `tune()`

**Files:**
- Create: `beyondml/engine/tune.py`
- Modify: `pyproject.toml` (`[project.optional-dependencies] optuna = ["optuna>=4.0", "cmaes>=0.10"]`)
- Test: `tests/test_tune.py` (the module starts with `optuna = pytest.importorskip("optuna")`)

**Interfaces:**
- **Consumes:** Task 3's `Space`, `Fitness`, `Fidelity`, `Budget`, `Candidate`, `GenStats`, `EvolveResult`, `active_keys` and `complete`.
- **Produces:**

  ```python
  def tune(space: Space, fitness: Fitness, *, strategy: str = "tpe", n_trials: int = 50, budget: Budget = Budget(),
           workers: int | None = None, fidelity: Sequence[Fidelity] = (Fidelity(),), seeds: Sequence[dict] = (),
           objectives: Sequence[str] = (), on_trial: Callable[[dict], None] | None = None, seed: int = 0) -> EvolveResult
  ```

- **Strategies:** `"tpe"` is `TPESampler(seed=seed)`, `"nsga2"` is `NSGAIISampler(seed=seed)`, and `"cmaes"` is `CmaEsSampler(seed=seed, warn_independent_sampling=False)`. Any other value raises `ValueError`.
- **Space mapping:**
  - `Int` → `suggest_int(log=)` and `Float` → `suggest_float(log=)`.
  - A tuple `Choice` → `suggest_categorical`.
  - A conditional `Choice` suggests the option first, then the option's sub-space under Optuna names `"{option}:{key}"`. These prefixes are stripped back off in `Candidate.params`.
  - `Mask(n)` → n `suggest_categorical(f"{key}[{i}]", [0, 1])`. If all of them come back 0, bit 0 is set in params.
  - `seeds` are flattened to Optuna names and added with `study.enqueue_trial`.
- **Fidelity:** for a ladder longer than 1, use `SuccessiveHalvingPruner()`. Each rung calls `trial.report(score, r)` and raises `optuna.TrialPruned` when `should_prune()` says so. Otherwise use `NopPruner`.
- **Objectives:**
  - `objectives=()` means a single objective: maximise fitness.
  - Otherwise each entry is `"name:max"` or `"name:min"`, where `"fitness"` is the score and any other name is read from `metrics`.
  - With several objectives, the study runs multi-objective without pruning, using only the last rung. `best` is the Pareto trial with the highest first objective, and `pareto` lists every Pareto trial.
- **Running:**
  - `study.optimize(n_trials=n_trials, timeout=budget.max_seconds, n_jobs=workers or 1)`.
  - `history` has one `GenStats` per finished trial, with `gen` set to the trial number + 1, the running best, and `diversity` set to 1.0.
  - `stop_reason` is `"trials"` or `"seconds"`.
  - Optuna logging is set to WARNING.

- [ ] **Step 1: Install the extra.** Run: `venv/Scripts/python.exe -m pip install "optuna>=4.0" "cmaes>=0.10"`. Expected: both install successfully.
- [ ] **Step 2: Write the failing tests** (quadratic fitness, as in Task 3)

  ```python
  def test_tpe_finds_optimum():                # n_trials 60, seed 0: |x-1| < 0.5 and |y+2| < 0.5
  def test_conditional_params_clean():         # COND space: set(best.params) == active_keys(COND, best.params), no ":" in any key
  def test_pruning_cuts_top_rung_calls():      # ladder of 2, n_trials 30: rung-1 call count < 30
  def test_nsga2_returns_pareto():             # objectives ("fitness:max", "cost:min"), metrics {"cost": x**2}, n_trials 40: len(pareto) >= 2
  def test_cmaes_runs():                       # n_trials 20: best.fitness > -inf
  def test_seed_enqueued_first():              # seeds=[{"x": 1.0, "y": -2.0}]: history[0].best_fitness == 0.0
  def test_unknown_strategy_rejected():        # strategy="grid" raises ValueError
  ```

- [ ] **Step 3: Run them and confirm they fail** (`ModuleNotFoundError: beyondml.engine.tune`)
- [ ] **Step 4: Implement `beyondml/engine/tune.py`**
- [ ] **Step 5: Run the full suite.** Run: `venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider`. Expected: 0 failed.
- [ ] **Step 6: Commit** `feat(engine): optional Optuna tune() behind the evolve interface`
