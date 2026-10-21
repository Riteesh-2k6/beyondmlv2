# BeyondML as an ML-engineering agent harness — design

- **Date:** 2026-10-19
- **Status:** draft, awaiting owner review
- **Replaces:** the fixed Orchestrator → EDA → … → Reflection pipeline (`tui_app.py`, `beyondml/web/runner.py`)

## 1. Intent

**What the owner asked for.** Restructure BeyondML into "OpenClaw, but for machine learning projects": an end-to-end harness that does what an ML engineer does, on its own. It accepts any data (tabular, text, time series, images, audio), analyses it, writes a concrete plan, and builds a dedicated agent for each part of the plan. Every agent has a screen that shows how it was instructed and what it is doing. Evolutionary hyperparameter tuning stays the core idea and must get faster. Optional tooling takes the result to production.

**Decisions made while brainstorming:**

| Topic | Decision |
|---|---|
| Audience | Open-source project: strangers install it and point it at their own data |
| What "production" means | BeyondML writes a standalone project repo (train, predict, serve, Dockerfile, tests, model card) that runs without BeyondML installed |
| Data | Any modality. The core is modality-agnostic; per-modality know-how lives in skills. Tabular ships first |
| Models | Any endpoint through ADK `LiteLlm` (Ollama, vLLM, LM Studio, Groq, OpenRouter, OpenAI, Anthropic) or native Gemini. Local models are supported, with tighter guard-rails |
| Agent framework | Google ADK 2.x (v2.12.0 at time of writing), version pinned |
| Agent model | Intake → planner → one dedicated agent per plan step, plus a lead agent the user chats with |
| Interaction | Chat-first. The agent plans, runs, and checks in at decision points |
| Clients | An interactive Textual TUI that works like an app, plus the React web UI. Both show the same live session |
| Tuning | Evolutionary search is the default and gets faster. Optuna (TPE, NSGA-II, CMA-ES, pruners) is an optional extra behind the same interface |
| Execution | The agent writes code as files in the workspace and runs them by subprocess. The workspace is the deliverable repo |

**Assumed, not stated by the owner:** skills use the open `SKILL.md` format (shared by OpenClaw, Claude Code and ADK's SkillToolset), so community skills can move between them.

**Success means:**
- `beyondml run "predict survival" --data titanic.csv --mode autonomous` produces a project folder.
- In that folder, `python src/train.py` reproduces the chosen model and the repo's own tests pass.
- Every step's agent, its exact instruction and its live activity can be inspected in the TUI and the web UI.

## 2. Architecture

```
beyondml/
  agents/       intake.py, planner.py, lead.py      ADK LlmAgents with fixed roles (the old *_agent.py files here are deleted)
                factory.py                          builds one LlmAgent per plan step
                executor.py                         runs the plan graph, checks success, retries, re-plans
  toolkit/      FunctionTools: workspace (read/write/list files, run_python), profile, evolve, tune,
                evaluate, explain, drift, chart, ask_user, install_package, update_plan, finish_step
  workspace/    project.py (init/open, PROJECT.md, MEMORY.md, ledger, git checkpoints), events.py (bus)
  engine/       kept: profiler, genetic (→ v2, section 7), metrics, supervised, unsupervised, ensemble, deep_learning
  skills/       bundled SKILL.md skills (section 5)
  gateway/      ADK get_fast_api_app + project/plan/agent/file/event routes; serves the web UI
  tui/          Textual app
  cli.py        beyondml | init | serve | run | benchmark
```

**Flow:**
1. **Intake** detects the modality of each input and runs the matching profiler, then writes `DATA.md`.
2. **Planner** writes `plan.json`. The user approves it, except in autonomous mode.
3. **Executor** builds step agents through the **factory** as their dependencies finish. Independent steps run in parallel.
4. **Lead** is the agent the user talks to. It explains, delegates, and amends the plan with `update_plan`.

**Retired:**
- `beyondml/agents/*_agent.py` and `orchestrator.py`. Their prompts move into skills; their deterministic code moves into `toolkit/`.
- `beyondml/llm/`, replaced by ADK models.
- `beyondml/state.py`, replaced by the workspace.
- `beyondml/web/runner.py`, `web/markup.py` and `tui_app.py`, the two hard-coded copies of the pipeline.
- Tests for these modules retire with them. Engine tests stay.

## 3. Workspace and memory

A project is a folder, and it is the repo that ships:

```
churn-model/
  PROJECT.md          goal, target, metric, constraints (user and agents edit it)
  DATA.md             what intake found, per input
  PLAN.md             rendered from plan.json for humans
  MEMORY.md           decisions and lessons; the lead agent curates it; capped at 200 lines
  plan.json           the plan (section 4)
  data/raw/           user data; read-only to every tool; not committed
  data/processed/
  src/                agent-written code: features.py, train.py, predict.py, ...
  models/  reports/
  experiments.jsonl   one record per run_python / evolve / tune / evaluate call
  agents/<step>/      instruction.md (exact prompt sent) and events.jsonl (that agent's events)
  .beyondml/          sessions.sqlite (ADK DatabaseSessionService), settings.toml, skills/
```

- **Context injected every turn:** `PROJECT.md`, `MEMORY.md` and the step's own goal and inputs. Everything else is read on demand.
- **The harness writes the ledger, not the agent.** Scripts only drop a `metrics.json`, so the shipped repo never imports BeyondML.
- **The workspace is a git repo.** The harness commits a checkpoint at each step's start and after each tool call that changed files. Rewind is `git reset` to a checkpoint. `data/raw/` and large artifacts are gitignored.
- **Config and skills come in three levels:** bundled, then `~/.beyondml/`, then `.beyondml/`. The nearest level wins. Endpoints and keys come from `settings.toml` plus environment variables.

## 4. Agents and the plan

`plan.json`:

```json
{
  "goal": "predict churn", "metric": "f1", "budget": {"minutes": 60, "usd": 2.0},
  "steps": [
    {
      "id": "features", "skill": "tabular-features", "depends_on": ["eda"],
      "goal": "derive features from usage and billing columns",
      "inputs": ["data/processed/clean.parquet"], "outputs": ["src/features.py"],
      "success_check": {"run": "src/features.py", "expects": ["data/processed/features.parquet"]},
      "status": "pending", "model": null, "attempts": 0
    }
  ]
}
```

- **Status values:** `pending | running | done | failed | skipped`.
- **`success_check`:** an optional script to run and the files it must produce, plus an optional `{"metric": "f1", "min": 0.7}` read from that run's `metrics.json`. **The executor runs the check. An agent's own claim never completes a step.**
- **Factory:** the instruction is the role preamble + the step goal and inputs + the skill body + `PROJECT.md` + summaries of upstream steps. The tools are the skill's `allowed-tools`. The model is `step.model`, then the skill's `model_hint`, then the project default. The instruction is saved to `agents/<id>/instruction.md` before the agent runs.
- **Ending a step:** the agent calls `finish_step(summary, outputs)`, then the executor runs `success_check`.
- **Failure:** retry once with the failure output. If that also fails, the planner gets the summary and returns a revised plan. In `ask` mode the revision needs confirmation; in `auto` and `autonomous` mode it is shown without prompting.
- **Graph runtime:** an ADK 2.x workflow built from `plan.json` at runtime, if ADK supports building one dynamically. Otherwise a small topological runner over ADK `Runner`s (see section 11).

## 5. Skills

- **Format:** `SKILL.md` with frontmatter `name` (≤64 chars), `description` (≤1024), `allowed-tools`, and `metadata.beyondml: {kind: step|knowledge, step, modality, model_hint}`. Optional `scripts/`, `references/` and `assets/`. They are loaded through ADK `SkillToolset` (list → load → resource). Skill scripts run through `run_python`, so they reach the ledger.
- **Step skills** say how to do one plan step for one modality. The first set is tabular, ported from today's agent prompts: `tabular-audit`, `tabular-clean`, `tabular-eda`, `tabular-features`, `tabular-baseline`, `evolve`, `evaluate-explain`, `package`.
- **Knowledge skills** are loaded on demand: `leakage-checklist`, `choose-metric`, `class-imbalance`, `train-test-split`.
- **The planner builds plans from skills.** It may only reference step skills that exist. When none fits, it uses a generic `python-step` skill.

## 6. Tools and safety

| Tool | Notes |
|---|---|
| `read_file`, `write_file`, `list_files` | Paths resolve inside the project root; `data/raw/` is read-only |
| `run_python(script, args, timeout)` | Subprocess with cwd = project and API keys stripped from the environment. Returns the exit code, the last 200 lines of stdout and stderr, the files changed and `metrics.json`. Appends a ledger record. On timeout, kills the whole process tree |
| `profile(path)` | Dispatches by modality. Tabular uses `engine/profiler.py`; other modalities register profilers through skill scripts |
| `evolve`, `tune` | Long-running (ADK `LongRunningFunctionTool`). Emit one `generation` / `trial` event each |
| `evaluate`, `explain`, `drift`, `chart` | Wrap the existing engine and agent code |
| `ask_user`, `install_package`, `update_plan`, `finish_step` | Control. `update_plan` is for the lead and planner only |

**Permission modes,** implemented with ADK `require_confirmation`:
- `ask`: confirm every `run_python` and install.
- `auto` (the default): confirm installs, network access and writes outside the project.
- `autonomous`: no prompts.

**Stated limits:**
- A subprocess is not a security boundary. An optional `sandbox = "docker"` setting runs code through ADK's container executor.
- Scripts use BeyondML's own Python environment by default, with an optional per-project venv. The `package` skill pins only what `src/` imports.

**Budgets:**
- Each agent: at most 40 tool calls.
- Each plan: minutes, plus USD for cloud models.

Reaching a budget pauses the run and asks; it never fails silently.

## 7. Evolution engine v2

**Measured today** (`data/latency_report.json`, 300×6 dataset): the GA trainer takes 28.0 s of a 33.5 s run (84%).
- 10 genomes × 5 generations, each doing 5-fold CV plus one full fit.
- `Parallel(n_jobs=-1)` builds a fresh process pool and copies the data into it every generation.
- There is no cache, so genomes that come out identical are scored again.

**Interface:**

```python
evolve(space, fitness, *, budget, workers=None, fidelity=None, seeds=(), on_generation=None, seed=0) -> EvolveResult
```

- **`space`:** `Choice`, `Int`, `Float(log=...)`, `Mask(n)`, and conditional sub-spaces keyed by a `Choice` value (model → its hyperparameters). This replaces the hard-coded sklearn `Genome`.
- **`fitness(params, fidelity) -> float`:** supplied by the caller. The toolkit provides the sklearn-pipeline CV fitness used today, including its overfitting-gap penalty. Torch skills supply their own.
- **`budget`:** `max_evals`, `max_seconds`, and `plateau_generations` (default 3).
- **`fidelity`:** a ladder such as `[{"rows": 0.25, "folds": 3, "effort": 0.2}, {"rows": 1.0, "folds": 5}]` with a promotion fraction of 0.3; `effort` is the fraction of the model's iterative budget (trees, boosting rounds, epochs). Each generation scores everyone at the lowest rung and promotes the top fraction upward.
- **Speed:**
  - one persistent worker pool for the whole run, using threads below 20k rows and processes above; the tabular fitness runs its folds on a fixed-size joblib (loky) pool that outlives the run, because sklearn on small frames is GIL-bound;
  - the data is shared with workers once (memmap), not once per generation;
  - a fitness cache keyed by `(params, fidelity)`.
- **`seeds`:** starting genomes. The `evolve` skill asks the LLM for 2–4 sensible configurations and adds today's warm-start defaults.
- **`workers`** defaults to cores − 1, and the user can cap it.

**`tune(strategy="tpe"|"nsga2"|"cmaes", ...)`** has the same shape, backed by Optuna (`pip install beyondml[optuna]`). NSGA-II offers multi-objective search, such as F1 against latency. Hyperband/ASHA pruners handle epoch-based training. The `evolve` skill picks evolution by default and suggests TPE when a single evaluation is expensive and the budget allows fewer than 50 evaluations.

**Targets:**
- On the 300×6 latency benchmark, the tuning stage takes ≤5 s warm (a second run in the same process) and no longer than v1 cold; the first run pays the fold pool's start, 6–8 s on Windows. Over 3 seeds, the mean best CV score is no more than 0.01 below v1's. (Re-stated 2026-10-21: the original ≤5 s including pool start is not reachable with sklearn RandomForest on this OS.)
- On PMLB `adult`, time-to-best-score is reported for v1 and v2.
- With a fixed seed and `workers=1`, results are bit-identical between runs.

## 8. Gateway, events, screens, CLI

**Event bus,** one per project:
- Shape: `{ts, agent, kind, data}`.
- Kinds: `message_delta`, `tool_call`, `tool_result`, `plan_updated`, `step_status`, `generation`, `trial`, `chart`, `artifact`, `approval_request`, `question`, `budget`, `error`.
- ADK events are normalised into this shape, using ADK's `author` as `agent`.
- Every event is appended to `agents/<agent>/events.jsonl` and fanned out to subscribers. A late subscriber replays from disk and then tails.

**Gateway:** ADK `get_fast_api_app` plus:
- `GET /projects`, `POST /projects`
- `GET /projects/{p}/plan`
- `GET /projects/{p}/agents/{id}` (instruction, status)
- `GET /projects/{p}/files/{path}`
- `GET /projects/{p}/events` (SSE, with `?since=`)
- `POST /projects/{p}/input` (chat, approval and answer payloads)

SSE replaces the current WebSocket.

**Screens,** the same set in the TUI and the web:
1. **Plan board:** step cards (status, model, elapsed, key metric) and the dependency graph.
2. **Agent screen:**
   - Instruction: the exact `instruction.md`, the skill, inputs and granted tools.
   - Live: messages and tool calls.
   - Outputs: files, metrics, charts.
   - Plus the step's own view. For `evolve` that's best and mean fitness per generation, population diversity and the best genome.
3. **Chat dock:** talk to the lead agent from any screen.
4. **Approvals:** a modal.
5. **Experiments:** a ledger table.
6. **Files.**

**CLI:**
- `beyondml` starts the gateway in-process and opens the TUI on the current project, so the web can attach to the same session.
- `beyondml init`.
- `beyondml serve` runs headless.
- `beyondml run "<goal>" --data <path> [--mode ask|auto|autonomous]` runs headless and prints events.
- `beyondml benchmark`.

## 9. Error handling

- **LLM errors:** LiteLLM retries with backoff. Malformed tool arguments go through ADK's tolerant parsing. Three consecutive tool errors in one step fail that step, which then follows the retry → re-plan path.
- **Crash or restart:** `plan.json` statuses, ADK sessions in `sessions.sqlite`, and the checkpoint taken at each step's start let `beyondml` resume. A step that was `running` resets to its start checkpoint and runs again.
- **Budget reached:** a `budget` event is emitted and the run pauses until the user raises the budget or stops.
- **Missing data, unreadable file, or unknown modality:** intake reports it in `DATA.md`. The planner does not plan steps that need it.

## 10. Testing

- **Engine:** the existing engine tests stay. Evolution v2 adds a benchmark test asserting the ≤5 s target on the 300×6 set, a determinism test, and fidelity-promotion and cache tests.
- **Toolkit:** path guard, read-only `data/raw`, env stripping, timeout kill, ledger records.
- **Executor,** with fake step agents: dependency order, parallel steps, `success_check` gating, retry, re-plan, resume after a simulated crash.
- **End to end with a scripted model:** a custom ADK `BaseLlm` returns scripted tool calls. Deterministic, runs in CI.
  - In sub-project 1 it covers titanic CSV → plan → steps → `src/` files that run, with ledger and events on disk.
  - Sub-project 5 extends it to a packaged repo whose own tests pass.
- **Optional, slow:** ADK `AgentEvaluator` eval sets against a real local model. Not part of the default run.

## 11. Verify before planning

1. Can an ADK 2.x workflow graph be built at runtime from `plan.json`, with parallel nodes and pauses for approval? If not, use the topological runner from section 4.
2. Does `SkillToolset` accept `allowed-tools` and `metadata`, and can it restrict tools per agent? If not, the factory filters tools itself.
3. Tool-calling quality through LiteLLM with one local model (Qwen 2.5/3 14B on Ollama), on a 5-step scripted task.
4. ADK's minimum Python version against `requires-python = ">=3.10"`.

**Resolved 2026-10-19** by probing ADK 2.12.0:
1. A workflow can be built at runtime, but its resume is driven by ADK session events and loses retry counts. So the executor is the topological runner.
2. `allowed-tools` is advisory only, so the factory filters tools itself.
3. Not tested: Ollama was not running. The core-harness plan's final task runs this smoke test when a local model is available.
4. ADK needs Python ≥3.11, so the floor goes up.

Details are in `docs/superpowers/plans/2026-10-19-core-harness.md`.

## 12. Sub-projects and order

| # | Sub-project | Depends on | Done when |
|---|---|---|---|
| 1 | Core harness, headless: workspace, toolkit basics, intake/planner/lead/factory/executor, event bus, `init`/`run` | — | The scripted-model e2e passes; events land in `events.jsonl`; resume after a kill works |
| 2 | Evolution engine v2 + `tune` (Optuna extra) | — (engine only) | The section 7 targets pass; v1 `GeneticModelOptimizer` callers move to v2 |
| 3 | Tabular skills: port the old agents to skills and toolkit tools, delete `beyondml/agents/*_agent.py` | 1, 2 | A real-model autonomous run on titanic scores ≥ v1's test accuracy |
| 4 | Screens: Textual TUI app + web UI rework and polish | 1 (event schema) | All six screens run on live events in both clients |
| 5 | Production: `package` skill (train/predict/serve, Dockerfile, tests, model card); optional tracking and CI skills | 3 | The generated repo's tests pass and `docker build` succeeds |
| 6 | More modalities: time series, text, images, audio (one skill pack each) | 3 | Each pack has a scripted e2e and one real-model run |

1 and 2 run in parallel. This spec details 1 and 2. Sub-projects 3–6 get their own short spec when they start.

## 13. Current state (2026-10-19)

- The health pass is not committed:
  - The 2 failing integration tests are fixed; they passed `target_info=` to `OrchestratorAgent.run`.
  - 32 `__pycache__`/`.pkl` files are untracked.
  - `debug_shap.py` and `direct_shap.py` are removed.
- Commit `8d7be8a`, "perf: 5.4x pipeline speedup", contains no engine, agent or LLM code, only tickets and a profiling script. The performance work it describes does not exist. Sub-project 2 replaces it.
- All `.scratch` tickets stay open, and the W/L tickets carry `## Verified 2026-10-19` notes. Once this spec is approved, W01–W08 are superseded by sub-project 4, L01–L07 by sub-projects 1 and 2, and MAP-001 (TICKET-001–008, removing user loops) by the permission modes.
