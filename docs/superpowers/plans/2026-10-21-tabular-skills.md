# Tabular Skills Implementation Plan (sub-project 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the harness do tabular ML end to end. The engine is exposed as typed tools, the old agents' judgement becomes skills, and a run ends in a project whose `src/train.py` and `src/predict.py` reproduce the GA winner without BeyondML.

**Architecture:**
- **New engine modules** (pure functions, unit-testable):
  - `pipeline`: one pipeline builder shared by GA scoring and export.
  - `export`: renders `src/*.py`.
  - `mlp`: a torch estimator.
  - `checks`: leakage, outliers, drift.
  - `assess`: holdout and health, importance.
  - `clustering`.
- **Tools:** `beyondml/toolkit/ml.py` wraps them as ADK tools bound to a step's `ToolEnv`.
- **Skills:** `SKILL.md` files under `beyondml/skills/` teach the step agents when and how to use the tools.
- **Retirement:** the old agents are deleted last, once the screens branch has landed.

**Tech Stack:** Python ≥3.11, pandas, scikit-learn, torch (already a dependency), plotly (already installed), scipy, google-adk 2.12.0 (via the harness), Optuna (optional extra).

**Spec:** `docs/superpowers/specs/2026-10-21-tabular-skills-design.md` (parent: `docs/superpowers/specs/2026-10-19-beyondml-ml-harness-design.md`)

**Deviations from the spec, decided while planning:**
1. **Progress events.** `evolve`, `tune` and `cluster` publish `generation` and `trial` events with `env.bus.emit(env.agent, ...)` from the worker thread; the bus is thread-safe. They do not yield ADK `Event(state=bml_event)`. ADK attributes generator-tool events to the tool's name, so they would land in `agents/evolve/` instead of the step agent's screen.
2. **Export by run id.** `export_pipeline` takes a `run_id`, the id that `evolve`, `tune`, `ensemble` or `cluster` returned, with `""` meaning the latest. It does not take a params dict. Agents (small local models especially) then never copy large parameter dicts.
3. **Engine layout.** The engine is split by responsibility into the six modules listed under Architecture, plus `toolkit/ml.py` for the thin tool wrappers. The spec said only "toolkit/ml.py".
4. **Metric lookup.** The executor's metric lookup searches the step's ledger records for the requested key, instead of taking the last record that has any metrics. `explain` and `evaluate` both write metrics in one step.

## Global Constraints

- **Interpreter:** `C:\Users\Riteesh\Programming\BeyondML\venv\Scripts\python.exe`, run from the repo or worktree root with `-m`.
- **Branch:** `tabular/skills`, cut from `main` at or after `b8b7eb2`. Commit per task with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never push, never merge.
- **Dependencies:** no new required ones. SHAP stays optional; when it is not installed, `explain` falls back to permutation importance.
- **Shipped code** (everything rendered into a project's `src/`) imports only the stdlib, numpy, pandas, scikit-learn and joblib, plus torch when the MLP wins. It **never imports beyondml**.
- **Tool rules** (as in the core harness): type hints and a docstring, project-relative paths through `env.project.path`, JSON-friendly dict results, and failures returned as `{"ok": False, "error": ...}`, never raised. Each tool appends one ledger record `{"agent": env.agent, "tool": <name>, "metrics": {...}, ...}`.
- **Event contracts:**
  - `generation`: `{gen, best_fitness, mean_fitness, diversity, evals, seconds, best_params}`
  - `trial`: `{number, value, params, state}`
  - `chart`: `{title, path}`
  - `question`: `{question, options, recommended}`
- **Thresholds:**
  - **Health:** health is 0 when the train/test gap > 0.10, top_share > 0.5, or the holdout score ≥ 0.999.
  - **Leakage:** a column is a suspect at \|corr\| ≥ 0.95, a single-feature score ≥ 0.98, or ≥ 95% unique values (ID-like).
  - **Ensemble:** used when the top genomes are within 1%.
  - **MLP:** a GA family when rows ≥ 2,000.
  - **TPE:** chosen when one evaluation is expensive and the budget is under 50 evaluations.
  - **Export equivalence:** tolerance 1e-9.
- **Default model:** `ollama_chat/qwen3:8b`.
- **Full suite:** `python -m pytest tests -q -p no:cacheprovider`. It must be green at the end of every task.

## Review Focus

1. **String or partly missing target.** A target of string labels ("yes"/"no"), or with NaN rows, works in `evolve`, `evaluate` and the exported train.py. Rows with a NaN target are dropped, and the count is reported. → Task 10, `test_evolve_string_target_and_nan_rows`
2. **ID-like high-cardinality column.** A column such as a passenger name is flagged by `scan_leakage` and never silently one-hot-encoded into thousands of columns by the audit's recommended drop list. → Task 5, `test_scan_flags_id_like`
3. **Bad new data at predict time.** predict.py on new rows with an unseen category still predicts. When a required column is missing, it exits 2 and names the column, rather than raising a traceback. → Task 4, `test_predict_unseen_category_and_missing_column`
4. **Tiny or extremely imbalanced data.** Fewer than 50 rows, or a single member in a class: `evaluate` falls back from a stratified to a plain split and does not crash. → Task 6, `test_evaluate_tiny_imbalanced`
5. **All-categorical frame.** Export equivalence holds when there are no numeric columns. → Task 4, `test_export_equivalence_all_categorical`

---

### Task 1: Titanic v1 baseline

**Files:**
- Create: `scripts/bench_titanic.py`
- Create: `tests/data/titanic_v1_baseline.json` (generated)

**Interfaces:**
- **Produces:** `tests/data/titanic_v1_baseline.json`:

  ```json
  {"engine": "v1", "seed": 0, "holdout_accuracy": 0.0, "best_model": "", "pop_size": 10, "generations": 5}
  ```

- **Protocol** (also used by Task 13):
  - Split `data/titanic.csv` into train and holdout with `train_test_split(test_size=0.2, random_state=0, stratify=target)`.
  - Profile the train part with `DatasetProfiler(train, target_column="target").run()`.
  - Run v1's `GeneticModelOptimizer(train, "target", profile, pop_size=10, generations=5)`, after `random.seed(0)` and `np.random.seed(0)`.
  - Rebuild v1's pipeline for the best genome (selected features, the median+scale and constant+onehot preprocessing, `_make_model`-equivalent v1 model), fit it on train, and score accuracy on the holdout.
- **Loading v1:** v1 no longer exists on `main`, so the script loads it from git: `git show 01aa1fb:beyondml/engine/genetic.py`. It writes that into a temp folder as `genetic_v1.py`, rewrites `from .metrics import` to `from beyondml.engine.metrics import`, and imports it with importlib.

- [ ] **Step 1: Write `scripts/bench_titanic.py --engine v1 [--record PATH]`.** It prints the JSON.
- [ ] **Step 2: Record the baseline.** Run: `python scripts/bench_titanic.py --engine v1 --record tests/data/titanic_v1_baseline.json`. Expected: JSON with `0.6 < holdout_accuracy < 0.95`.
- [ ] **Step 3: Commit** `bench: titanic v1 holdout baseline for sub-project 3`.

---

### Task 2: One pipeline builder for scoring and export

**Files:**
- Create: `beyondml/engine/pipeline.py`
- Modify: `beyondml/engine/fitness.py`. Move `_make_model` and `_MODEL_TABLE`'s model construction out of it, and make `TabularFitness.__call__` call `build_pipeline`.
- Test: `tests/test_pipeline_builder.py`

**Interfaces:**
- **Produces** in `beyondml/engine/pipeline.py`:

  ```python
  def make_model(name: str, classification: bool, hp: dict, class_weight: str | None):  # moved verbatim from fitness._make_model
  def column_kinds(X: pd.DataFrame) -> tuple[list[str], list[bool]]                    # (names, is_numeric) in column order
  def build_pipeline(params: dict, names: list[str], is_numeric: list[bool], classification: bool,
                     class_weight: str | None = None) -> sklearn.pipeline.Pipeline
      # ColumnTransformer(num: SimpleImputer(median)+StandardScaler on the selected numeric names,
      #                   cat: SimpleImputer(constant "missing")+OneHotEncoder(handle_unknown="ignore") on the selected categorical names)
      # -> make_model(params["model"], ...). The selection is params["features"] (mask) applied to names.
      # The pipeline takes the FULL feature frame; unselected columns are dropped by the ColumnTransformer.
  ```

- **Behaviour must not change.** All existing `tests/test_fitness_tabular.py`, `tests/test_evolve_*.py` and `tests/test_genetic.py` stay green.

- [ ] **Step 1: Write the failing tests.**

  ```python
  def test_build_pipeline_matches_fitness_cv(classification_df, classification_profile):
      # params = warm RF seed; cross_validate(build_pipeline(params, names, kinds, True), X, y, same StratifiedKFold(5, shuffle, 0))
      # mean == TabularFitness(X, y, "classification", profile)(params, Fidelity())[1]["mu_cv"]   (abs 1e-9)
  def test_build_pipeline_drops_unselected_columns():   # features mask (1,0,1): the fitted ColumnTransformer only uses columns 0 and 2
  def test_column_kinds():                              # mixed frame -> names in order, is_numeric right
  ```

- [ ] **Step 2: Run them and confirm they fail.** Expected: `ModuleNotFoundError: beyondml.engine.pipeline`.
- [ ] **Step 3: Implement `pipeline.py` and refactor `fitness.py`** to use it. `fitness.py` re-exports `make_model` under its old private name only if a test imports it.
- [ ] **Step 4: Run the tests plus the engine tests.** Run: `python -m pytest tests/test_pipeline_builder.py tests/test_fitness_tabular.py tests/test_evolve_loop.py tests/test_genetic.py -q -p no:cacheprovider`. Expected: all pass.
- [ ] **Step 5: Commit** `refactor(engine): one pipeline builder shared by scoring and export`.

---

### Task 3: MLP as a GA model family

**Files:**
- Create: `beyondml/engine/mlp.py`
- Modify: `beyondml/engine/fitness.py` (`_MODEL_TABLE` gains `"MLP"`) and `beyondml/engine/pipeline.py` (`make_model` builds it)
- Test: `tests/test_mlp.py`

**Interfaces:**
- **Produces:**

  ```python
  class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
      def __init__(self, hidden: tuple[int, ...] = (64, 32), lr: float = 1e-3, epochs: int = 30,
                   dropout: float = 0.1, batch_size: int = 64, seed: int = 0): ...
      def fit(self, X, y): ...      # classes_ from y (any label type); torch.manual_seed(seed); torch.set_num_threads(1); CPU
      def predict(self, X): ...
      def predict_proba(self, X): ...
  class TorchMLPRegressor(BaseEstimator, RegressorMixin): ...   # same params; fit/predict
  ```

  - X arrives as a dense or sparse matrix from the ColumnTransformer. Convert with `.toarray()` when it is sparse.
  - **The module imports only numpy, torch and sklearn,** so Task 4 can copy it verbatim into `src/mlp.py`.
- **`_MODEL_TABLE["MLP"]`:**
  - Problem types: both.
  - Search space: `{"hidden": Choice(((32,), (64,), (64, 32), (128, 64))), "lr": Float(1e-4, 1e-2, log=True), "epochs": Choice((20, 40, 80)), "dropout": Choice((0.0, 0.1, 0.3)), "batch_size": Choice((32, 64, 128))}`.
  - Warm seed: `{"hidden": (64, 32), "lr": 1e-3, "epochs": 40, "dropout": 0.1, "batch_size": 64}`.
  - Not in `DEFAULT_MODELS`; it is only used when asked for.
  - The complexity penalty for MLP is 0.0, like the linear models.

- [ ] **Step 1: Write the failing tests.**

  ```python
  def test_mlp_classifier_learns_separable():   # make_classification(300, 6, class_sep=2, random_state=0): train accuracy > 0.9; string labels round-trip
  def test_mlp_regressor_fits():                # make_regression(300, 5, noise=1, random_state=0): r2 > 0.8
  def test_mlp_is_deterministic_and_picklable():# same seed -> identical predictions; pickle.loads(pickle.dumps(fitted)) predicts the same
  def test_mlp_imports_only_allowed():          # ast-parse mlp.py: top-level imports subset of {numpy, torch, sklearn, __future__}
  def test_evolve_tabular_with_mlp(classification_df, classification_profile):
      # evolve_tabular(..., models=("MLP",), pop_size=3, budget=Budget(max_generations=1)): best.params["model"] == "MLP", fitness > -1
  ```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement `mlp.py` and the table entry.**
- [ ] **Step 4: Run them and confirm they pass.** Also run tests/test_fitness_tabular.py, which must stay green.
- [ ] **Step 5: Commit** `feat(engine): torch MLP estimators as a GA model family`.

---

### Task 4: Render the shipped code

**Files:**
- Create: `beyondml/engine/export.py`
- Test: `tests/test_export.py`

**Interfaces:**
- **Consumes:** `build_pipeline` and `column_kinds` (Task 2), and `mlp.py` (Task 3).
- **Produces:**

  ```python
  def model_spec(params: dict, names: list[str], is_numeric: list[bool], problem_type: str,
                 class_weight: str | None, target: str, raw: str) -> dict
      # {"kind": "model", "params", "names", "is_numeric", "problem_type", "class_weight", "target", "raw"}
  def render_pipeline(spec: dict) -> str
      # source of src/pipeline.py: module constants FEATURES (the full feature-frame column list), TARGET, PROBLEM, DATA (=raw);
      # def build_pipeline(): returns the same estimator build_pipeline/build_ensemble/build_clusterer would, written with literals
  TRAIN_PY: str     # the src/train.py template (fixed text, no placeholders: it reads constants from pipeline.py)
  PREDICT_PY: str   # the src/predict.py template
  def write_project_code(root: Path, spec: dict) -> list[str]
      # writes src/pipeline.py, src/train.py, src/predict.py (+ src/mlp.py copied from engine/mlp.py when the model is MLP);
      # returns project-relative paths
  ```

- **`src/train.py` behaviour:**
  1. Read `DATA`.
  2. If `src/clean.py` exists, `from clean import clean` and apply it. Then do the same with `src/features.py` / `make_features`.
  3. Drop rows with a NaN `TARGET`.
  4. Split 80/20 (`random_state=0`, stratified when every class has ≥ 2 rows).
  5. Fit `build_pipeline()`, then score the holdout (accuracy, or r2 for regression).
  6. Refit on all rows and `joblib.dump` to `models/model.joblib`.
  7. Write `{<score name>: score, "n_train", "n_test"}` to `os.environ.get("BEYONDML_METRICS", "metrics.json")`, and print it.

  Modules are imported via `sys.path.insert(0, <src dir>)`.
- **`src/predict.py` behaviour:** `python src/predict.py INPUT [OUTPUT]`.
  1. Load the model.
  2. Apply `clean` and `make_features` when they exist.
  3. If any `FEATURES` column is missing, print `missing columns: a, b` to stderr and exit 2.
  4. Otherwise write INPUT's rows plus a `prediction` column to OUTPUT (default `predictions.csv`).

- [ ] **Step 1: Write the failing tests.** They render into `tmp_path` and run the scripts with `subprocess` plus `sys.executable`. A guard makes `import beyondml` fail inside the child: `-c "import sys, runpy; sys.modules['beyondml'] = None; sys.argv = [...]; runpy.run_path('src/train.py', run_name='__main__')"`.

  ```python
  def test_export_equivalence(classification_df, classification_profile):
      # params = evolve_tabular(..., pop_size=4, Budget(max_generations=2), seed=0).best.params; render + import src/pipeline.py from tmp
      # cross_validate(module.build_pipeline(), X, y, StratifiedKFold(5, shuffle, 0)).mean == fitness mu_cv (abs 1e-9)
  def test_export_equivalence_all_categorical():  # frame with only string columns + target: same equality
  def test_train_and_predict_run_without_beyondml(tmp_path):
      # titanic copy in data/raw, spec from the RF warm seed: train.py exits 0, writes models/model.joblib and metrics.json with "accuracy";
      # predict.py on 5 rows exits 0 and the output has 5 predictions
  def test_train_applies_clean_and_features(tmp_path):  # src/features.py adds family_size; FEATURES includes it; train exits 0
  def test_predict_unseen_category_and_missing_column(tmp_path):
      # unseen category value -> exit 0; a FEATURES column dropped from input -> exit 2 and stderr names it
  def test_mlp_export_copies_module(tmp_path):    # MLP spec -> src/mlp.py exists; train.py exits 0
  def test_rendered_code_never_imports_beyondml(tmp_path):  # "beyondml" not in any rendered src/*.py text
  ```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement `export.py`.** Render the model with literals via `repr()` of the hyperparameters, and match the class and constructor `make_model` uses (random_state=42, n_jobs=1, class_weight).
- [ ] **Step 4: Run them and confirm they pass.**
- [ ] **Step 5: Commit** `feat(engine): render shipped train/predict/pipeline code from a GA winner`.

---

### Task 5: Data checks (leakage, outliers, drift)

**Files:**
- Create: `beyondml/engine/checks.py`
- Test: `tests/test_checks.py`

**Interfaces:**
- **Produces:**

  ```python
  POST_OUTCOME = re.compile(r"(after|post|outcome|result|final|refund|cancel)", re.I)   # weak, name-only hint
  def scan_leakage(df: pd.DataFrame, target: str, problem_type: str) -> list[dict]
      # [{column, reason, evidence}], reason in {"target_correlation", "single_feature_score", "id_like", "post_outcome_name"};
      # thresholds from Global Constraints; single_feature_score = 3-fold CV of DecisionTree(max_depth=3) on that column alone
      # (ROC AUC for binary, accuracy for multiclass, r2 for regression); categorical columns ordinal-encoded for it
  def detect_outliers(df: pd.DataFrame, columns: list[str] | None = None, method: str = "iqr") -> dict
      # numeric columns only; {col: {"count", "pct", "sample_rows": [<=5 row indices]}};
      # iqr: outside [Q1-1.5IQR, Q3+1.5IQR]; zscore: |z| > 3; isolation_forest: IsolationForest(random_state=0) on the columns jointly
      # ValueError on an unknown method
  def drift(reference: pd.DataFrame, current: pd.DataFrame, alpha: float = 0.05) -> dict
      # shared columns; numeric: scipy.stats.ks_2samp; categorical: chi2_contingency on the aligned value counts;
      # {col: {"test", "statistic", "p_value", "drifted": p_value < alpha}}
  ```

- [ ] **Step 1: Write the failing tests.**

  ```python
  def test_scan_flags_target_copy():        # column = target * 2 + noise(0.01): "target_correlation"
  def test_scan_flags_id_like():            # a unique string "name" column: "id_like"
  def test_scan_flags_single_feature():     # a column equal to the target's label: "single_feature_score"
  def test_scan_clean_frame_is_quiet():     # classification_df: no suspects except possibly name hints
  def test_outliers_iqr_counts():           # 100 normal values + [1000, -1000]: count == 2, sample_rows hold their indices
  def test_outliers_unknown_method():       # pytest.raises(ValueError)
  def test_drift_detects_shift():           # N(0,1) vs N(2,1), 500 rows each: drifted True; same distribution: False; a categorical column too
  ```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement `checks.py`.** Port the deterministic parts of `agents/drift_agent.py` and `agents/outlier_agent.py`, and leave the LLM calls behind.
- [ ] **Step 4: Run them and confirm they pass.**
- [ ] **Step 5: Commit** `feat(engine): leakage, outlier and drift checks`.

---

### Task 6: Holdout assessment and importance

**Files:**
- Create: `beyondml/engine/assess.py`
- Test: `tests/test_assess.py`

**Interfaces:**
- **Produces:**

  ```python
  HEALTH = {"max_gap": 0.10, "max_top_share": 0.5, "max_score": 0.999}
  def importance(estimator, X: pd.DataFrame, y: pd.Series, top_k: int = 10, seed: int = 0) -> dict
      # fitted estimator; shap when importable (TreeExplainer/LinearExplainer, else KernelExplainer on <=100 rows), else
      # sklearn.inspection.permutation_importance(n_repeats=5, random_state=seed) on the raw columns;
      # {"method": "shap"|"permutation", "top": [{"feature", "value"}] (desc), "top_share": largest / sum(abs)}
  def assess(estimator_factory: Callable[[], Any], X: pd.DataFrame, y: pd.Series, classification: bool,
             test_size: float = 0.2, seed: int = 0) -> dict
      # split (stratified when every class has >= 2 rows, else plain), fit a fresh estimator, score train and holdout
      # (accuracy / r2), confusion matrix for classification, importance(top_k=10) on the holdout;
      # {"score_name", "score", "train_score", "gap", "top_share", "confusion", "importance", "health", "health_reasons"}
      # health = 0 if gap > max_gap or top_share > max_top_share or score >= max_score, else 1; reasons name each trigger
  ```

- [ ] **Step 1: Write the failing tests.**

  ```python
  def test_assess_healthy():               # RF on classification_df with signal: health 1, reasons []
  def test_assess_flags_overfit():         # DecisionTree(max_depth=None) on a noise target: gap > 0.10 -> health 0, "gap" in reasons
  def test_assess_flags_dominance():       # one feature equal to the label (+ noise columns): top_share > 0.5 -> health 0
  def test_assess_flags_perfect_score():   # a leak column: score >= 0.999 -> health 0, "score" in reasons
  def test_evaluate_tiny_imbalanced():     # 30 rows, one class with a single member: no exception, plain split used
  def test_importance_permutation_fallback(monkeypatch):  # shap import blocked: method == "permutation", top sorted desc
  ```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement `assess.py`.**
- [ ] **Step 4: Run them and confirm they pass.**
- [ ] **Step 5: Commit** `feat(engine): holdout assessment with health rules and importance`.

---

### Task 7: Ensembles

**Files:**
- Modify: `beyondml/engine/pipeline.py` (add `build_ensemble`) and `beyondml/engine/export.py` (the `"ensemble"` kind)
- Test: `tests/test_ensemble_v2.py`

**Interfaces:**
- **Produces:**

  ```python
  def distinct_top(population: list[dict], top_n: int) -> list[dict]
      # population is a list of {"params", "fitness"} best-first; dedupe by (model, hyperparameters, features); first top_n
  def build_ensemble(members: list[dict], names, is_numeric, classification, class_weight=None, kind: str = "stacking")
      # members: params dicts. stacking -> StackingClassifier/Regressor(estimators=[(f"m{i}", build_pipeline(p, ...))],
      #   final_estimator=LogisticRegression(max_iter=1000)/RidgeCV(), cv=5); voting -> VotingClassifier(voting="soft" when all
      #   members have predict_proba else "hard")/VotingRegressor. ValueError for another kind or < 2 members.
  def ensemble_spec(members, names, is_numeric, problem_type, class_weight, target, raw, kind) -> dict   # {"kind": "ensemble", ...}
  ```

  `render_pipeline` renders an ensemble spec into the same `build_pipeline()` function.

- [ ] **Step 1: Write the failing tests.**

  ```python
  def test_distinct_top_dedupes():          # 5 entries, 2 identical: 4 distinct, best-first order kept
  def test_stacking_builds_and_scores(classification_df):   # 3 distinct members: cross_val mean is finite and in [0, 1]
  def test_ensemble_rejects_one_member():   # pytest.raises(ValueError)
  def test_ensemble_export_equivalence(tmp_path, classification_df):  # rendered build_pipeline() CV == build_ensemble CV (abs 1e-9)
  ```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them and confirm they pass.**
- [ ] **Step 5: Commit** `feat(engine): stacking and voting ensembles of GA winners`.

---

### Task 8: Clustering

**Files:**
- Create: `beyondml/engine/clustering.py`
- Modify: `beyondml/engine/export.py` (the `"cluster"` kind)
- Test: `tests/test_clustering.py`

**Interfaces:**
- **Produces:**

  ```python
  def cluster_space(algorithms: Sequence[str] = ("kmeans", "dbscan"), pca: bool = True) -> Space
      # {"algo": Choice({"kmeans": {"n_clusters": Int(2, 10)}, "dbscan": {"eps": Float(0.1, 5.0, log=True),
      #  "min_samples": Int(3, 20)}} restricted to `algorithms`), "pca": Choice((0, 2, 5)) if pca else Choice((0,))}
  def build_clusterer(params: dict, names, is_numeric) -> Pipeline
      # the same preprocessing as build_pipeline (all columns selected) -> PCA(n_components=min(pca, n_features), random_state=0)
      # when pca > 0 -> KMeans(n_clusters, n_init=10, random_state=0) or DBSCAN(eps, min_samples)
  class SilhouetteFitness:     # picklable; __call__(params, fidelity) -> (score, {"silhouette", "n_clusters", "noise_pct"})
      # rows subsampled by fidelity.rows (random_state=0); fewer than 2 clusters, or > 50% DBSCAN noise -> (-1.0, ...)
  def evolve_clusters(df: pd.DataFrame, *, algorithms=("kmeans", "dbscan"), pca=True, pop_size=10, generations=8,
                      seed=0, on_generation=None) -> EvolveResult
  def cluster_spec(params, names, is_numeric, raw) -> dict   # {"kind": "cluster", ...}
  ```

- **Exported code for the cluster kind:**
  - **train.py** fits, writes `data/processed/clusters.csv` (the input rows plus `cluster`), and puts silhouette in the metrics. It saves `{"pipeline", "assigner"}`, where `assigner` is a `KNeighborsClassifier(n_neighbors=1)` fitted on the transformed non-noise points when the algorithm is DBSCAN (KMeans: `None`).
  - **predict.py** uses `pipeline.predict` for KMeans and `assigner` for DBSCAN.

- [ ] **Step 1: Write the failing tests.**

  ```python
  def test_kmeans_finds_blobs():            # make_blobs(300, centers=3, random_state=0): evolve_clusters(..., algorithms=("kmeans",)) best n_clusters == 3
  def test_dbscan_noise_penalised():        # params with eps tiny -> fitness -1.0
  def test_cluster_export_runs(tmp_path):   # train.py exits 0 and writes clusters.csv; predict.py labels 5 new rows (both kmeans and dbscan specs)
  ```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them and confirm they pass.**
- [ ] **Step 5: Commit** `feat(engine): evolved clustering and its exported code`.

---

### Task 9: Data tools

**Files:**
- Create: `beyondml/toolkit/ml.py` (the data half)
- Test: `tests/test_ml_tools_data.py`

**Interfaces:**
- **Consumes:** `ToolEnv` (`beyondml/toolkit/env.py`), `DatasetProfiler`, and `checks.py` (Task 5).
- **Produces:** `def ml_tools(env: ToolEnv) -> list[Callable]`. This task adds:

  ```python
  def profile(path: str, target: str = "") -> dict
      # {ok, rows, cols, feature_types, missing, target_analysis, correlation_summary, outlier_summary, overfitting_risk_index}
  def scan_leakage(path: str, target: str) -> dict          # {ok, problem_type, suspects}
  def detect_outliers(path: str, columns: list[str] = [], method: str = "iqr") -> dict   # {ok, columns}
  def drift(reference: str, current: str, alpha: float = 0.05) -> dict                   # {ok, columns, drifted: [names]}
  def chart(kind: str, data: str, x: str, y: str = "", title: str = "") -> dict
      # kinds hist|bar|scatter|box|corr|line via plotly.express; writes reports/charts/<slug(title or kind-x-y)>.json
      # (fig.to_json()); emits chart {title, path}; returns {ok, path}
  ```

  - Every tool reads CSVs through `env.project.path` (Parquet when the suffix is .parquet).
  - Every tool appends a ledger record (`agent`, `tool`, plus `metrics` where meaningful, for example `{"suspects": n}`).
  - Errors (a missing file, an unknown column or method) come back as `ok False`.

- [ ] **Step 1: Write the failing tests** (a temp project with `data/raw/t.csv` from `classification_df`, plus a fake ToolEnv as in tests/test_toolkit_files.py):

  ```python
  async def test_profile_tool():           # ok, rows == 200, target_analysis present with target "target"
  async def test_scan_leakage_tool():      # a leak column added: its name in [s["column"] for s in suspects]
  async def test_outliers_tool():          # ok; unknown method -> ok False
  async def test_drift_tool():             # shifted copy -> drifted non-empty
  async def test_chart_tool():             # writes reports/charts/*.json (valid JSON with "data"); chart event emitted with path
  async def test_tools_refuse_escapes():   # path "../x.csv" -> ok False
  async def test_tools_write_ledger():     # every call above appended a record with agent == env.agent and tool name
  ```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement the data half of `ml.py`.**
- [ ] **Step 4: Run them and confirm they pass.**
- [ ] **Step 5: Commit** `feat(toolkit): data tools (profile, leakage, outliers, drift, chart)`.

---

### Task 10: Model tools

**Files:**
- Modify: `beyondml/toolkit/ml.py` (the model half)
- Test: `tests/test_ml_tools_model.py`

**Interfaces:**
- **Consumes:**
  - `evolve_tabular`, `tabular_space`, `TabularFitness` and `default_ladder` (`engine/fitness.py`)
  - `tune` (`engine/tune.py`, which is optional: `ok False` with an install hint when Optuna is missing)
  - Tasks 2–8
- **Produces**, added to `ml_tools(env)`:

  ```python
  async def evolve(data: str, target: str, models: list[str] = [], pop_size: int = 10, generations: int = 10,
                   minutes: float = 0.0, seeds: list[dict] = [], seed: int = 0) -> dict
  async def tune(data: str, target: str, strategy: str = "tpe", trials: int = 50, objectives: list[str] = [],
                 seed: int = 0) -> dict
  async def ensemble(run_id: str = "", top_n: int = 5, kind: str = "stacking") -> dict
  async def cluster(data: str, algorithms: list[str] = ["kmeans", "dbscan"], pca: bool = True, pop_size: int = 10,
                    generations: int = 8) -> dict
  def export_pipeline(run_id: str = "", raw: str = "") -> dict          # "" -> the latest search report; raw "" -> the only file in data/raw
  async def evaluate(data: str, target: str, pipeline: str = "src/pipeline.py", test_size: float = 0.2, seed: int = 0) -> dict
  async def explain(data: str, target: str, pipeline: str = "src/pipeline.py", top_k: int = 10) -> dict
  ```

- **Search runs:**
  - The search tools run the engine in `asyncio.to_thread`. Each `on_generation` / `on_trial` callback calls `env.bus.emit(env.agent, "generation" | "trial", {...})` with the contract fields; `best_params` is JSON-safe (tuples become lists).
  - Rows with a NaN target are dropped first, and the result reports `dropped_rows`.
  - Each search writes `reports/<kind>-<run_id>.json` with `{kind, run_id, target, data, raw, problem_type, names, is_numeric, class_weight, best_params, best_fitness, mu_cv, population: [{params, fitness}], history}`. `run_id` is `uuid4().hex[:8]`.
  - `export_pipeline` rebuilds the spec from that report.
  - The search tools' ledger metrics are `{"mu_cv", "best_fitness"}`; for cluster, `{"silhouette"}`.
- **`evaluate`** imports the project's `src/pipeline.py` with `importlib.util.spec_from_file_location` (a fresh module name per call) and calls `assess(module.build_pipeline, X, y, ...)`. It writes `reports/eval.json`, and its ledger metrics are `{score_name, "train_score", "gap", "top_share", "health"}`.
- **`explain`** fits `build_pipeline()` on 80% of the rows and calls `importance` on the rest. It writes `reports/importance.json` plus a bar chart (through `chart`'s writer), with ledger metrics `{"top_share"}`.

- [ ] **Step 1: Write the failing tests.**

  ```python
  async def test_evolve_streams_generations(tmp_path):   # pop_size=3, generations=2: ok; >= 2 "generation" events under env.agent
                                                          #   with exactly the contract keys; reports/evolve-<id>.json exists
  async def test_evolve_string_target_and_nan_rows(tmp_path):  # target "yes"/"no" with 5 NaN rows: ok, dropped_rows == 5
  async def test_evolve_bad_seed_is_error(tmp_path):     # seeds=[{"bogus": 1}] -> ok False, error names "bogus"
  async def test_tune_or_install_hint(tmp_path):         # optuna present -> ok with trial events; patched missing -> ok False, "pip install"
  async def test_export_latest_and_train_runs(tmp_path): # evolve, then export_pipeline(): files listed; run_script("src/train.py") ok
  async def test_evaluate_reports_health(tmp_path):      # after export: ok, metrics has health in {0, 1}; ledger has it under env.agent
  async def test_explain_writes_importance(tmp_path):    # ok; reports/importance.json; top sorted desc
  async def test_ensemble_from_latest(tmp_path):         # after evolve pop 6: ok, members >= 2, new run_id exportable
  async def test_cluster_tool(tmp_path):                 # blobs frame: ok, data/processed/clusters.csv, generation events
  ```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement the model half of `ml.py`.**
- [ ] **Step 4: Run them and confirm they pass.**
- [ ] **Step 5: Commit** `feat(toolkit): model tools (evolve, tune, ensemble, cluster, export, evaluate, explain)`.

---

### Task 11: Harness wiring

**Files:**
- Modify: `beyondml/toolkit/control.py` (`ask_user`), `beyondml/harness/factory.py` (step agents get `ml_tools`), `beyondml/harness/executor.py` (metric lookup by key) and `beyondml/workspace/project.py` (default model)
- Test: `tests/test_harness_ml_wiring.py`

**Interfaces:**
- **`ask_user(question: str, options: list[str] = [], recommended: str = "") -> dict`**:
  - It emits `question {question, options, recommended}`.
  - In autonomous mode it returns `{"answer": recommended}` when `recommended` is set, else the existing `AUTONOMOUS_ANSWER`.
  - Otherwise it calls `env.asker` with the question text, plus `"\nOptions: a | b | c (recommended: a)"` when there are options.
- **Factory:** step agents' tool list is `file_tools + python_tools + control_tools + ml_tools`, filtered by the skill's `allowed-tools`.
- **Executor:** `_last_metrics(step_id, key)` returns the newest ledger record for that step whose `metrics` contains `key`, and `check()` passes `sc.metric`.
- **Settings:** `DEFAULT_SETTINGS["model"]["default"] = "ollama_chat/qwen3:8b"`, and the same in `SETTINGS_TOML`.

- [ ] **Step 1: Write the failing tests.**

  ```python
  async def test_ask_user_options_autonomous():   # recommended "a" -> answer "a"; event data has options and recommended
  async def test_ask_user_options_prompt():       # mode auto: the asker received text containing "Options: a | b"
  def test_step_agent_gets_ml_tools(tmp_path):    # skill allowed {"evolve", "finish_step"} -> tool names == that set
  def test_metric_lookup_by_key(tmp_path):        # ledger: evaluate {"health": 1} then explain {"top_share": 0.3}; check metric health min 1 passes
  def test_default_model_is_qwen3(tmp_path):      # Project.init(...).settings()["model"]["default"] == "ollama_chat/qwen3:8b"
  ```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run these tests and the full suite.** Expected: 0 failed.
- [ ] **Step 5: Commit** `feat(harness): ask_user options, ML tools for step agents, metric lookup by key`.

---

### Task 12: The skills

**Files:**
- Create: `beyondml/skills/<name>/SKILL.md` for the ten step skills and four knowledge skills below.
- Test: `tests/test_tabular_skills.py`

**Content rules:**
- **Each step skill** has frontmatter `name`, a `description` that says what the step does and where it usually sits (for example, "Usually right after tabular-clean"), `allowed-tools`, and `metadata.beyondml: {kind: step, modality: tabular}`.
- **Bodies** are short imperative lines, ported from the old agents' prompts and rules. Read each `beyondml/agents/<x>_agent.py` for its SYSTEM_PROMPT and decision rules. Bodies must use the Task 9 and 10 tool names, the shipped-code conventions (`clean(df)` in `src/clean.py`, `make_features(df)` in `src/features.py`, both runnable as scripts), and each checkpoint's exact `ask_user` call with `options` and `recommended`.

| Skill | allowed-tools | Ported from | Must say |
|---|---|---|---|
| `tabular-audit` | read_file list_files write_file run_python profile scan_leakage chart ask_user finish_step | sanity, leakage, orchestrator target logic | Target checkpoint: `ask_user("Which column is the target?", options=<ranked candidates>, recommended=<top>)`. Write reports/audit.md with the drop list. Record decisions in the finish_step summary |
| `tabular-clean` | read_file list_files write_file run_python profile detect_outliers ask_user finish_step | imputation, outlier | Outlier checkpoint, options `["cap", "remove", "keep", "transform"]`. `src/clean.py` defines `clean(df)` |
| `tabular-eda` | read_file list_files write_file run_python profile chart finish_step | eda | Writes reports/eda.md with insights for the features step |
| `tabular-features` | read_file list_files write_file run_python profile scan_leakage ask_user finish_step | feature | Features checkpoint (options = the proposals, recommended = all non-suspects). Vectorised pandas only, never use the target. `src/features.py` defines `make_features(df)` |
| `evolve-search` | read_file list_files profile evolve tune finish_step | ga_trainer, dl | The TPE vs GA rule; 2–4 seeds; add "MLP" to `models` when rows ≥ 2000; put the run_id in the finish_step summary |
| `tabular-ensemble` | read_file ensemble finish_step | ensemble | Only keep it if cv_score > best_single |
| `tabular-export` | read_file list_files write_file run_python export_pipeline finish_step | codegen | `export_pipeline()`, then run src/train.py, fix it if needed |
| `evaluate-explain` | read_file list_files write_file evaluate explain chart finish_step | evaluator, explainability, reflection | Write reports/evaluation.md. On health 0, finish with the reasons so the re-plan can fix them |
| `tabular-cluster` | read_file list_files write_file run_python profile cluster chart export_pipeline finish_step | unsupervised engine | — |
| `drift-check` | read_file list_files write_file drift chart finish_step | drift | — |

**Knowledge skills** (`kind: knowledge`, no allowed-tools):
- `leakage-checklist` (from leakage_agent's prompt)
- `choose-metric`
- `class-imbalance`
- `train-test-split`

- [ ] **Step 1: Write the failing tests.**

  ```python
  def test_all_bundled_skills_load():           # discover(None) has the 10 step + 4 knowledge names; kinds right; modality "tabular" on steps
  def test_allowed_tools_exist():               # every allowed tool name is in the names of file/python/control/ml tools (built on a fake env)
  def test_checkpoints_present():               # audit, clean and features bodies contain "ask_user(" and "recommended"
  def test_descriptions_state_order():          # every step description contains "Usually" or "Only when"
  ```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Write the skills.**
- [ ] **Step 4: Run them and confirm they pass.**
- [ ] **Step 5: Commit** `feat(skills): tabular step and knowledge skills`.

---

### Task 13: Scripted end-to-end runs

**Files:**
- Test: `tests/test_tabular_e2e.py`

**Interfaces:**
- **Consumes:** everything above, `run_goal` and `scripted_resolver`.
- **Scripted planner plan (titanic):**
  1. `audit` (expects reports/audit.md)
  2. `clean` (run src/clean.py, expects data/processed/clean.csv)
  3. `eda` (expects reports/eda.md)
  4. `features` (run src/features.py, expects data/processed/features.csv)
  5. `search` (skill evolve-search, metric mu_cv min 0.6)
  6. `export` (run src/train.py, expects models/model.joblib, metric accuracy min = the Task 1 baseline)
  7. `evaluate` (skill evaluate-explain, metric health min 1)
- **Scripted step agents** call the real tools in the order their skill says:
  - The audit step's `ask_user` gets `recommended="target"`.
  - The search step runs `evolve` with pop_size 10, generations 5 and seed 0, matching the baseline budget, on `data/processed/features.csv`.
  - The feature step adds `family_size = sibsp + parch + 1`.

- [ ] **Step 1: Write the failing tests.**

  ```python
  async def test_titanic_full_shape(tmp_path):
      # run_goal(..., mode="autonomous", resolver=scripted): all 7 steps done
      # in the project: `sys.modules['beyondml'] = None` guard; runpy src/train.py exits 0; predict.py on 5 rows writes 5 predictions
      # metrics.json accuracy >= json.load(tests/data/titanic_v1_baseline.json)["holdout_accuracy"]
      # agents/search/events.jsonl has >= 5 "generation" events
  async def test_no_target_cluster_shape(tmp_path):
      # make_blobs frame without a target, plan audit -> clean -> cluster -> export (run src/train.py, metric silhouette min 0.3):
      # all done; data/processed/clusters.csv exists. (The supervised evaluate tool needs a target, so this shape ends at export.)
  ```

- [ ] **Step 2: Run them and confirm they fail.** They pass only once the earlier tasks are wired, so the first run on the branch must fail if any wiring is missing.
- [ ] **Step 3: Fix whatever the run exposes.** Each fix gets its own focused test in the owning task's test file.
- [ ] **Step 4: Run them, then the full suite.** Expected: 0 failed.
- [ ] **Step 5: Commit** `test: scripted titanic and clustering end-to-end runs on the real tools`.

---

### Task 14: Real-model acceptance

**Files:**
- Create: `docs/superpowers/reports/2026-10-21-tabular-acceptance.md`
- Modify: `beyondml/skills/*/SKILL.md` only if a run exposes unclear instructions. At most 3 rounds of skill fixes per model.

- [ ] **Step 1: Make sure Ollama is up.**
  - Run `curl -s http://127.0.0.1:11434/api/tags`.
  - If that fails, start `ollama serve` in the background. `C:\Users\Riteesh\AppData\Local\Programs\Ollama\ollama.exe` is installed, and `qwen3:8b` is already pulled.
- [ ] **Step 2: Run the local model.** In a scratch folder: `python -m beyondml.cli run "predict survival" --data data/titanic.csv --project <scratch>/qwen --mode autonomous --model ollama_chat/qwen3:8b`. Record the exit code, each step's status, the holdout accuracy, the time taken, and any failure reasons.
- [ ] **Step 3: Run Groq.**
  - Export `GROQ_API_KEY` from the repo's `.env` into the shell. Print only that it is set, never the value.
  - Run the same command with `--project <scratch>/groq --model groq/llama-3.3-70b-versatile`.
- [ ] **Step 4: Fix and retry.** Where a model fails a step, read `agents/<step>/events.jsonl`, fix the skill text, add a regression line to that skill's test if one applies, and rerun.
- [ ] **Step 5: Write the report.**
  - One table per model: step, status, attempts, notes.
  - A pass or fail per model, where pass means the plan finished and holdout accuracy ≥ the baseline.
  - The skill fixes made.
- [ ] **Step 6: Commit** `docs: tabular acceptance on qwen3:8b and Groq`.

---

### Task 15: Retire the old agents

**Gate:** run this task only when `git merge-base --is-ancestor <screens branch head> main` succeeds, meaning sub-project 4 has retired `tui_app.py` and `web/runner.py` into `main`. Merge `main` into this branch first. If the gate fails, stop and report; every other task is already done.

**Files to delete:**
- `beyondml/agents/`
- `beyondml/llm/`
- `beyondml/state.py`
- `data/latency_profile.py`
- `beyondml/engine/genetic.py`, after moving `engine/benchmarker.py` onto `evolve_tabular`
- `beyondml/engine/ensemble.py`, superseded by Task 7
- `tests/test_agents.py`, `tests/test_pipeline_integration.py`, `tests/test_llm.py` and `tests/verify_dl_agent.py`
- the agent-level tests in `tests/test_drift.py`, `test_explainability.py`, `test_ensemble.py` and `test_deep_learning.py`. Their engine-level assertions move into Tasks 5, 6 and 7's test files when they aren't already covered.

**Also:** remove `groq` from `pyproject.toml`.

- [ ] **Step 1: Prove nothing still imports the deleted modules.** Run `grep -rn "beyondml.agents\|beyondml.llm\|beyondml.state\|engine.genetic\|engine.ensemble" beyondml tests scripts` after the deletions. Expected: no matches.
- [ ] **Step 2: Run the full suite.** Expected: 0 failed.
- [ ] **Step 3: Commit** `chore: retire the legacy agents, LLM providers and state now that the harness replaces them`.
