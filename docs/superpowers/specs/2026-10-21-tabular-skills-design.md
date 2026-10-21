# Sub-project 3: tabular skills — design

- **Date:** 2026-10-21
- **Status:** draft, awaiting owner review
- **Parent spec:** `2026-10-19-beyondml-ml-harness-design.md` (§5 skills, §6 tools, §7 evolution, §12 sub-project 3)
- **Builds on:** core harness (`harness/core`, with its final-review fixes) and evolution engine v2 (`claude/evolve-v2-ee9c45`), both merged into `main` first

## 1. Intent

Make the harness actually do tabular machine learning. The old app's 14 agents are retired. Their judgement becomes skills, their machinery becomes typed tools, and the evolution engine becomes a tool the agents call. A run on a CSV ends with a project folder whose `src/train.py` and `src/predict.py` reproduce the chosen model with plain scikit-learn (or torch, for the MLP), with no BeyondML installed.

**Decided with the owner:**

| Topic | Decision |
|---|---|
| Retraining the shipped project | **Freeze the winner.** `evolve`/`tune` run while BeyondML builds the project. The shipped code hard-codes the winning pipeline. Re-tuning means running BeyondML again |
| From GA winner to code | **Template plus agent edits.** `export_pipeline` writes the exact pipeline that was scored. The agent may edit around it (for example, wiring in its feature transforms), and the success check re-scores the edited version |
| Human check-ins | **Classic checkpoints.** In `ask` and `auto` modes the agents confirm the target column, the outlier strategy and the accepted features. In `autonomous` mode they take their own recommendation and log it |
| Scope beyond the core | Ensembles, the unsupervised path, an MLP as a GA model family, and a drift check, all included |
| Structure | **Skills for judgement, typed tools for machinery.** Agents write glue code (cleaning, feature transforms) with `run_python` |
| Acceptance model | **Both** local `ollama_chat/qwen3:8b` and Groq. The default model setting changes from `qwen2.5:14b`, which is not pulled on the dev machine, to `qwen3:8b` |

## 2. Typed tools (`beyondml/toolkit/ml.py`)

All tools follow the core harness rules. They take project-relative paths, return JSON-friendly dicts, never raise to the model, and append a ledger record with `agent` set to the calling step, plus `metrics` where they have any. That last part matters: the executor's metric checks read the step's latest ledger metrics.

| Tool | Signature | Notes |
|---|---|---|
| `profile` | `(path: str, target: str = "") -> dict` | Wraps `engine/profiler.py`: shape, types, missing values, target analysis and ORI |
| `scan_leakage` | `(path: str, target: str) -> dict` | Returns `{"suspects": [{column, reason, evidence}]}`. A column is a suspect if its \|corr\| with the target is ≥ 0.95, if it scores ≥ 0.98 alone (cross-validated AUC or R²), if it is ID-like (≥ 95% unique), or if its name suggests it was recorded after the outcome |
| `detect_outliers` | `(path: str, columns: list[str] = [], method: str = "iqr") -> dict` | Methods: `iqr`, `zscore`, `isolation_forest`. Returns a count, a percentage and up to 5 sample row indices per column |
| `evolve` | `(data: str, target: str, models: list[str] = [], pop_size: int = 10, generations: int = 10, minutes: float = 0, seeds: list[dict] = [], seed: int = 0)` | Calls `engine.fitness.evolve_tabular` in a worker thread. An async-generator tool: it yields one `generation` event per generation, then returns `{ok, run_id, best_params, best_fitness, mu_cv, evals, seconds, history}`. `history` is `reports/evolve-<run_id>.json` |
| `tune` | `(data: str, target: str, strategy: str = "tpe", trials: int = 50, objectives: list[str] = [], seed: int = 0)` | Calls `engine.tune.tune` and yields `trial` events. Same return shape as `evolve`, plus `pareto` |
| `export_pipeline` | `(params: dict, data: str, target: str) -> dict` | Writes `src/pipeline.py`, `src/train.py` and `src/predict.py`, plus `src/mlp.py` when the MLP family wins. Accepts `evolve`/`tune` winners, `ensemble` results and `cluster` results |
| `evaluate` | `(data: str, target: str, pipeline: str = "src/pipeline.py", test_size: float = 0.2, seed: int = 0) -> dict` | Holdout metrics, confusion matrix and the train/test gap. Also computes `health` (§4). Writes `reports/eval.json` |
| `explain` | `(data: str, target: str, pipeline: str = "src/pipeline.py", top_k: int = 10) -> dict` | Uses SHAP when installed, otherwise permutation importance. Reports `top_share` (the largest feature's share of total importance). Writes `reports/importance.json` and a chart |
| `ensemble` | `(run_id: str, top_n: int = 5, kind: str = "stacking") -> dict` | Builds from the distinct top genomes in that `evolve` history. Returns `{cv_score, best_single, members}` |
| `cluster` | `(data: str, algorithms: list[str] = ["kmeans", "dbscan"], pca: bool = True, pop_size: int = 10, generations: int = 8) -> dict` | Runs `evolve` over clustering parameters with silhouette fitness and streams `generation` events. Writes `data/processed/clusters.csv` |
| `drift` | `(reference: str, current: str, alpha: float = 0.05) -> dict` | KS test for numeric columns, chi-square for categorical. Returns per-column `{statistic, p_value, drifted}` |
| `chart` | `(kind: str, data: str, x: str, y: str = "", title: str = "") -> dict` | Kinds: `hist`, `bar`, `scatter`, `box`, `corr`, `line`. Writes Plotly JSON to `reports/charts/<slug>.json` and emits a `chart` event `{title, path}` |

**Event contracts**, shared with sub-project 4 (screens):
- `generation` data: `{gen, best_fitness, mean_fitness, diversity, evals, seconds, best_params}`. These mirror `GenStats`.
- `trial` data: `{number, value, params, state}`.
- Long tools emit these as `Event(state={"bml_event": {"kind", "data"}})`. The core harness's `normalize()` already routes that shape.

**MLP model family.** `beyondml/engine/mlp.py` adds `TorchMLPClassifier` and `TorchMLPRegressor`, scikit-learn-compatible estimators:
- **Search space:** hidden sizes, learning rate (log scale), epochs, dropout and batch size, joining `tabular_space` as model `"MLP"`.
- **Constraints:** seeded, CPU, and picklable.
- **Self-contained:** the file imports only torch, numpy and scikit-learn, so `export_pipeline` copies it verbatim into `src/mlp.py`.

**The export guarantee.** `engine/fitness.py` gains `build_pipeline(params, columns, problem_type) -> sklearn.Pipeline`. The fitness function and `export_pipeline` both use it. `src/pipeline.py` is rendered so that its `build_pipeline()` constructs the same pipeline from literals, with the same feature mask, preprocessing and model. A test proves that cross-validating the exported pipeline reproduces the GA's `mu_cv` for the same parameters, to within 1e-9.

## 3. Shipped-code conventions

Agents write these and the templates rely on them:
- **`src/clean.py`** defines `clean(df) -> df` and still runs as a script (`data/raw` in, `data/processed/clean.csv` out).
- **`src/features.py`** defines `make_features(df) -> df` the same way (`clean.csv` in, `features.csv` out).
- **`src/train.py`** (exported) reads the raw data, applies `clean` and then `make_features` when those modules exist, fits `build_pipeline()`, and writes `models/model.joblib` plus metrics to `BEYONDML_METRICS` (falling back to `metrics.json`).
- **`src/predict.py`** (exported) loads the model, applies the same `clean` and `make_features` to new rows, and writes predictions. New data therefore goes through exactly the training transforms.

## 4. Skills (`beyondml/skills/`, bundled)

**Step skills.** Each has `kind: step, modality: tabular` and lists only the tools it needs:

| Skill | Job | Checkpoint (`ask`/`auto` only) |
|---|---|---|
| `tabular-audit` | Runs `profile`, `scan_leakage` and the old sanity rules (impossible values, wrong types). Fixes the target and problem type, lists the columns to drop. Writes `reports/audit.md` and records decisions in `MEMORY.md` | **Target**: options are the ranked candidates, recommended is the top one |
| `tabular-clean` | The old imputation and outlier rules, using `detect_outliers`. Writes `src/clean.py` | **Outlier strategy**: options are cap, remove, keep, transform |
| `tabular-eda` | `profile` plus `chart`. Insights go to `reports/eda.md` for the feature step | — |
| `tabular-features` | The old feature agent's rules: vectorised pandas expressions, no target use, no leakage. Writes `src/features.py` | **Accept features**: lists the proposals, recommends all non-suspect ones |
| `evolve-search` | Chooses between `evolve` and `tune` (TPE when a single evaluation is expensive and the budget is under 50 evaluations). Seeds 2–4 sensible configurations. Picks model families, adding MLP when there are ≥ 2,000 rows | — |
| `tabular-ensemble` | Planned only when the top genomes are within 1% of each other. Keeps the ensemble only if its CV score beats the best single model | — |
| `tabular-export` | Runs `export_pipeline` on the winner, wires in `clean`/`make_features`, runs `src/train.py` | — |
| `evaluate-explain` | Runs `evaluate` and `explain`, writes `reports/evaluation.md`. Its success check is `{"metric": "health", "min": 1}` | — |
| `tabular-cluster` | The no-target path: `cluster` and charts, then `export_pipeline` | — |
| `drift-check` | Planned only when the inputs include a reference set and a newer batch | — |

**`health`.** `evaluate` reports `health = 0` when any of the old reflection rules fires: the train/test gap exceeds 0.10, `top_share` exceeds 0.5, or the holdout score is ≥ 0.999 (suspected leakage). Otherwise `health = 1`, and the reasons are listed in either case. A red flag therefore fails the step's check, and the executor's retry and re-plan handle the fix. No separate reflection agent is needed.

**Knowledge skills.** These are loaded on demand: `leakage-checklist`, `choose-metric`, `class-imbalance`, `train-test-split`.

**Checkpoints.** `ask_user` gains `options: list[str] = []` and `recommended: str = ""`:
- The `question` event carries both, so the screens render buttons.
- In `autonomous` mode the call returns `recommended` (or the existing autonomous answer when there is none), and the agent logs the choice to `MEMORY.md`.

**Plan shapes.** Skill descriptions state where each step usually goes, so the planner is guided toward these, but it may adapt:
- **Supervised:** audit → clean → eda → features → evolve-search → (ensemble) → export → evaluate-explain.
- **No target:** audit → clean → eda → cluster → export → evaluate.

## 5. Retirement and sequencing

**Removed in the last task of this sub-project:**
- `beyondml/agents/` (all 14 agents, `orchestrator.py`, `ga_trainer.py`, `__init__.py`)
- `beyondml/llm/`
- `beyondml/state.py`
- `data/latency_profile.py`
- the `engine/genetic.py` adapter. `engine/benchmarker.py` moves to `evolve_tabular`.
- the `groq` dependency
- tests of the old agent pipeline: `test_agents`, `test_pipeline_integration`, `test_llm`, `verify_dl_agent`, and the agent-level parts of the drift, explainability, ensemble and deep-learning tests. Their engine-level checks move into the new tool tests.

**Sequencing:**
- `tui_app.py` and `beyondml/web/runner.py` import the old agents, and sub-project 4 retires both. So the removal task runs only after the screens branch has merged into `main`. Every other task in this sub-project only adds code.
- Branch order: `harness/core` (once green) and `claude/evolve-v2-ee9c45` merge into `main`, locally, as approved by the owner on 2026-10-21. This sub-project then branches from `main`, and the screens branch merges `main` before it finishes.

## 6. Testing and acceptance

**Unit tests.** One per tool, on small synthetic frames. Plus:
- the export-equivalence test (§2)
- `health` red-flag cases
- `ask_user` options in autonomous mode
- MLP fit, predict and pickle
- the `generation` event shape produced by `evolve`

**Baseline (measured first).** v1's GA (`GeneticModelOptimizer` defaults, seed 0) on `data/titanic.csv`, using an 80/20 holdout with `random_state=0`. Its holdout accuracy is recorded in `tests/data/titanic_v1_baseline.json` before anything is removed.

**Scripted end-to-end runs** (deterministic, in the suite):
1. **Titanic, full supervised shape.** Scripted agents call the real tools. In the resulting folder, `python src/train.py` and `python src/predict.py` both run with BeyondML absent from the subprocess path, and holdout accuracy is ≥ the v1 baseline.
2. **A frame with no target:** the cluster shape, end to end.

**Real-model acceptance** (manual; results recorded in the plan's final report):
- `beyondml run "predict survival" --data data/titanic.csv --mode autonomous`, once with `--model ollama_chat/qwen3:8b` and once with a Groq model through LiteLlm (`GROQ_API_KEY` from `.env`).
- **Pass** means the plan finishes and holdout accuracy is ≥ the v1 baseline.
- **If a model fails,** the failing step's events and the fix to the skill instructions go in the report.

## 7. Risks

- **Small local models may not follow multi-tool skills.** That is why the risky maths sits in tools, and why `qwen3:8b` is part of acceptance.
- **The GA on the feature frame inherits Plan A's speed profile.** The `evolve-search` skill defaults to `minutes` budgets on large data.
- **SHAP is not installed,** so `explain` falls back to permutation importance. SHAP stays an optional extra.
