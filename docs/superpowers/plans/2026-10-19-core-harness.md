# Core Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A headless `beyondml run "<goal>" --data <path>` takes a project folder from raw data to an executed, checkpointed plan. It runs one ADK agent per plan step, writes every event to disk, and resumes after a crash.

**Architecture:**
- `beyondml/workspace/` (project folder, `plan.json`, event bus) and `beyondml/toolkit/` (ADK FunctionTools bound to a project) are plain Python and testable without a model.
- `beyondml/harness/` holds the agents: a deterministic intake, the planner, the step-agent factory, the lead, and a topological executor over ADK `Runner`s.
- `plan.json` is the single source of truth.
- Every test drives the agents with a scripted `BaseLlm`, so the whole flow runs offline.

**Tech Stack:** Python ≥3.11, google-adk 2.12.0 (LlmAgent, Runner, FunctionTool, SkillToolset, SqliteSessionService, LiteLlm), litellm, click, git CLI. Existing deps: pandas and scikit-learn.

**Spec:** `docs/superpowers/specs/2026-10-19-beyondml-ml-harness-design.md`, §§2–6, 8 (event bus and CLI only), 9, 10 (sub-project 1 in §12)

**Deviations from the spec, decided in planning from the ADK 2.12.0 probes** (probe scripts were in the session scratchpad):
1. **Package name.** The package is `beyondml/harness/`, not `beyondml/agents/`. `agents/__init__.py` eagerly imports all 14 legacy agents, torch included, until sub-project 3 deletes them.
2. **Executor.** A custom topological runner over `Runner`s, not an ADK `Workflow`. Workflow resume is driven by ADK session events, and its retry count is lost on resume, so `plan.json` could not stay the single source of truth.
3. **Skill tools.** ADK treats `allowed-tools` as advisory, so the factory filters tools itself. Knowledge skills come through `SkillToolset` with `tool_filter=["list_skills", "load_skill", "load_skill_resource"]`. Skill scripts run through our `run_python` so they reach the ledger. ADK's `run_skill_script` runs in a temp dir outside it.
4. **Intake.** Intake is deterministic code that emits events as agent `intake`. Detection and profiling need no LLM.
5. **Sessions.** `SqliteSessionService` (`google.adk.sessions.sqlite_session_service`) needs no SQLAlchemy.
6. **Gateway.** The HTTP gateway routes move to sub-project 4, alongside the web client that needs them. This plan exposes `run_goal`, `EventBus` and `chat` for those routes to wrap.
7. **Tool progress.** Long tools stream progress as async-generator FunctionTools that yield `Event(state={"bml_event": {"kind": ..., "data": ...}})`. `LongRunningFunctionTool` cannot stream. Sub-project 3 uses this for `evolve`; this plan only normalises it.
8. **CLI transition.** The legacy TUI moves from `beyondml run` to `beyondml tui`. Running `beyondml` with no subcommand still opens it until sub-project 4. `beyondml serve` arrives with the gateway in sub-project 4.
9. **ML tools.** The toolkit's ML tools (`profile`, `evolve`, `tune`, `evaluate`, `explain`, `drift`, `chart`) arrive with sub-project 3, which owns the tabular skills that call them. This plan ships the workspace, Python and control tools.
10. **Docker sandbox.** The optional `sandbox = "docker"` mode is deferred. This plan runs code by subprocess only, as the spec's stated default.

## Global Constraints

- **Interpreter:** run everything from the repo root with `venv/Scripts/python.exe` (Windows).
- **Prerequisite:** `main` contains the commit `chore: health pass and harness spec`.
- **Branch and commits:** work on branch `harness/core`, commit after every task, never push.
- **Python floor:** `requires-python = ">=3.11"`, because ADK 2.12.0 requires it.
- **New dependencies:** `google-adk==2.12.0` (pinned, because Skills are experimental) and `litellm>=1.106`. Nothing else.
- **ADK names:** agent names must be Python identifiers, so step ids match `^[a-z][a-z0-9_]*$`.
- **Shared ADK constants:** `APP_NAME = "beyondml"`, `USER_ID = "local"`.
- **Modes:** `ask | auto | autonomous`, default `auto`.
- **Caps:** at most 40 tool calls per agent (`RunConfig(max_llm_calls=41)`); `MEMORY.md` keeps its last 200 lines; `run_python` returns the last 200 lines of stdout and stderr.
- **Events:** each event is `{ts, agent, kind, data}`, appended to `agents/<agent>/events.jsonl`. The allowed kinds are `message_delta, tool_call, tool_result, plan_updated, step_status, generation, trial, chart, artifact, approval_request, question, budget, error`.
- **Per-step files:** the exact instruction goes to `agents/<step_id>/instruction.md`. Run metrics go to the path in env var `BEYONDML_METRICS`, which scripts fall back from to `metrics.json`.
- **Gitignore for a project:** `data/raw/`, `data/processed/`, `.beyondml/sessions.sqlite*`, `.beyondml/runs/`, `*.pkl`, `*.joblib`, `*.pt`.
- **Tool functions** have type hints and a docstring, because ADK builds the model-facing declarations from them. They return dicts and never raise to the model. Failures come back as `{"ok": False, "error": "..."}`.
- **Noise to ignore:** the ADK warnings about the experimental `JSON_SCHEMA_FOR_FUNC_DECL` and missing token usage metadata. Filter them in the pytest config.

## Review Focus

1. **A data path that is a missing file, an empty file or a binary CSV** is recorded in `DATA.md` as unusable, with the reason. The planner is never offered steps on it, and the run continues. → Task 8, `test_unusable_inputs_reported`
2. **A model that submits an invalid plan** (a cycle, an unknown skill, a missing dependency, a non-identifier id) gets the errors back from `submit_plan`. After 3 rejected submissions, `PlanError` lists them. → Task 10, `test_three_invalid_plans_raise`
3. **Path escapes:** `../x`, an absolute path, a symlink that leaves the project, or a write under `data/raw/`. The tool returns an error dict and writes nothing. → Task 5, `test_file_tools_refuse_escapes`
4. **A script that never exits and spawns children:** the timeout kills the whole process tree, the result says `timed_out: True`, and a ledger record exists. This holds even when the project path contains spaces. → Task 6, `test_timeout_kills_tree`, `test_project_path_with_spaces`
5. **A crash in the middle of a step:** on resume, only the files that step touched are restored to its start checkpoint. Other steps' finished outputs and `data/raw/` are untouched. → Task 11, `test_resume_restores_only_touched_files`

---

### Task 1: Dependencies and the scripted test model

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/scripted_llm.py`
- Test: `tests/test_scripted_llm.py`

**Interfaces:**
- **`pyproject.toml` changes:**
  - `requires-python = ">=3.11"`, with the 3.10 classifier removed.
  - Add `google-adk==2.12.0` and `litellm>=1.106` to `dependencies`.
  - Add `[tool.setuptools.package-data] beyondml = ["skills/**/*"]`.
  - Under `[tool.pytest.ini_options]`, add `filterwarnings = ["ignore:.*JSON_SCHEMA_FOR_FUNC_DECL.*"]`.
- **Produces `tests/scripted_llm.py`:**

  ```python
  def fc(name: str, **args) -> types.Part            # a function_call part
  def txt(text: str) -> types.Part
  def last_response(req: LlmRequest) -> dict         # .response of the last function_response part in req.contents ({} if none)

  class ScriptedLlm(BaseLlm):                        # pydantic: fields must be declared
      model: str = "scripted"
      script: list = []                              # one entry per model call: list[Part] or Callable[[LlmRequest], list[Part]]
      calls: int = 0
      seen: list = []                                # the LlmRequests it received
      async def generate_content_async(self, llm_request, stream=False): ...   # yields one LlmResponse(content=Content(role="model", parts=...)) per call
                                                                                 # past the end of the script, repeats the last entry

  def scripted_resolver(scripts: dict[str, list[list]]) -> Callable[[str, str | None], ScriptedLlm]
      # scripts[agent_name] is a list of scripts, one per resolver call for that agent (attempt 1, attempt 2, ...),
      # and the last one is reused. KeyError("no script for <name>") if the agent has none.
      # The returned resolver exposes .models: dict[str, list[ScriptedLlm]] for assertions.
  ```

- [ ] **Step 1: Install.** Run: `venv/Scripts/python.exe -m pip install -e ".[dev]"`. Expected: `google-adk-2.12.0` installs successfully.
- [ ] **Step 2: Write the test**

  ```python
  async def test_scripted_agent_calls_tool():
      # def add(a: int, b: int) -> dict: "Add."  returns {"sum": a + b}
      # LlmAgent(name="a", model=ScriptedLlm(script=[[fc("add", a=2, b=3)], [txt("done")]]), tools=[add])
      # run with Runner + InMemorySessionService
      # assert: some event has a function_response with response == {"sum": 5}; the final text == "done"; model.calls == 2
  ```

- [ ] **Step 3: Run it.** Run: `venv/Scripts/python.exe -m pytest tests/test_scripted_llm.py -q -p no:cacheprovider`. Expected: it passes once `tests/scripted_llm.py` exists. Write the helper first if the run fails on import.
- [ ] **Step 4: Run the full suite** to confirm nothing regressed. Expected: 0 failed.
- [ ] **Step 5: Commit** `build: python>=3.11, pin google-adk 2.12.0, scripted test model`

---

### Task 2: The project workspace

**Files:**
- Create: `beyondml/workspace/__init__.py` (empty), `beyondml/workspace/project.py`
- Test: `tests/test_workspace_project.py`

**Interfaces:**
- **Produces:**

  ```python
  class ProjectError(Exception): ...
  class PathError(ProjectError): ...

  def beyondml_home() -> Path       # $BEYONDML_HOME, else ~/.beyondml

  class Project:
      root: Path
      @classmethod
      def init(cls, root: Path, goal: str | None = None) -> "Project"
          # creates data/raw, data/processed, src, models, reports, agents, .beyondml/skills;
          # writes PROJECT.md ("# Project\n\n## Goal\n<goal, or '(not set yet)'>\n"), MEMORY.md ("# Memory\n"), the .gitignore from Global Constraints,
          # .beyondml/settings.toml (defaults below); git init + commit "beyondml: init". Idempotent: never overwrites existing files.
          # ProjectError("git is required ...") if git is not on PATH.
      @classmethod
      def open(cls, root: Path) -> "Project"      # ProjectError if root/.beyondml is missing
      def path(self, rel: str, *, write: bool = False) -> Path
          # resolved path (symlinks followed) must stay inside root, otherwise PathError;
          # write=True under data/raw also raises PathError; absolute input paths raise PathError
      def read(self, rel: str) -> str               # "" if the file is missing
      def write(self, rel: str, text: str) -> None  # via path(write=True); creates parent dirs; utf-8
      def append_memory(self, line: str) -> None    # appends; keeps the header plus the last 200 lines
      def ledger_append(self, record: dict) -> None # adds "ts" (epoch float); one JSON line in experiments.jsonl
      def ledger(self) -> list[dict]
      def checkpoint(self, message: str) -> str     # git add -A; commit only if there are changes; returns the HEAD sha
      def restore(self, sha: str, paths: Iterable[str]) -> None
          # each path returns to its state at sha: git checkout sha -- path if it existed there, else delete it
      def settings(self) -> dict                    # defaults <- beyondml_home()/settings.toml <- .beyondml/settings.toml (tomllib), deep-merged
      def add_data(self, src: Path) -> Path         # copies a file or directory into data/raw/<name>, returns the destination
  ```

- **Default settings:**

  ```toml
  [model]
  default = "ollama_chat/qwen2.5:14b"
  api_base = "http://127.0.0.1:11434"

  [run]
  mode = "auto"
  max_tool_calls = 40
  budget_minutes = 60
  budget_usd = 2.0
  ```

- **Git operations** (`checkpoint` and `restore`) hold one `threading.Lock` per `Project` instance, because parallel steps share one repo.

- [ ] **Step 1: Write the failing tests**

  ```python
  def test_init_creates_layout(tmp_path):         # dirs and files exist; `git log --oneline` has 1 line; ".gitignore" contains "data/raw/"
  def test_init_is_idempotent(tmp_path):          # init twice, PROJECT.md edited in between: the edit survives, still 1 commit
  def test_open_rejects_plain_folder(tmp_path):   # raises ProjectError
  def test_path_guard(tmp_path):                  # "../x", str(Path(tmp_path).anchor) + "Windows", and path("data/raw/a.csv", write=True) raise PathError;
                                                  #   reading data/raw/a.csv is allowed; a symlink to outside raises PathError
                                                  #   (pytest.skip if os.symlink raises OSError on Windows)
  def test_checkpoint_only_on_change(tmp_path):   # two checkpoints with no change in between return the same sha
  def test_restore_only_given_paths(tmp_path):    # sha = checkpoint; edit a.txt, create b.txt and c.txt; restore(sha, ["a.txt", "b.txt"]):
                                                  #   a.txt back to its old content, b.txt gone, c.txt still present
  def test_memory_capped(tmp_path):               # append 250 lines: the file holds the header plus the last 200; line "249" present, line "0" absent
  def test_settings_merge(tmp_path, monkeypatch): # BEYONDML_HOME sets model.default="x"; the project sets run.mode="ask":
                                                  #   both apply; max_tool_calls is still 40
  def test_add_data_not_committed(tmp_path):      # add_data(csv); checkpoint: `git ls-files` does not list data/raw/...
  def test_ledger_roundtrip(tmp_path):            # ledger_append({"a": 1}): ledger()[0]["a"] == 1 and it has "ts"
  ```

- [ ] **Step 2: Run them and confirm they fail.** Run: `venv/Scripts/python.exe -m pytest tests/test_workspace_project.py -q -p no:cacheprovider`. Expected: `ModuleNotFoundError`.
- [ ] **Step 3: Implement `project.py`**, calling git via `subprocess.run([...], cwd=root, check=True, capture_output=True)`. Pass `-c user.name=beyondml -c user.email=beyondml@localhost` on every commit so it works without a configured git identity.
- [ ] **Step 4: Run the tests and confirm they pass.** Expected: 10 passed.
- [ ] **Step 5: Commit** `feat(workspace): project folder with path guard, ledger, git checkpoints`

---

### Task 3: The plan model

**Files:**
- Create: `beyondml/workspace/plan.py`
- Test: `tests/test_plan.py`

**Interfaces:**
- **Consumes:** Task 2's `Project`.
- **Produces:**

  ```python
  STEP_ID = re.compile(r"^[a-z][a-z0-9_]*$")

  @dataclass
  class SuccessCheck:
      run: str | None = None
      expects: list[str] = field(default_factory=list)
      metric: str | None = None
      min: float | None = None

  @dataclass
  class Step:
      id: str; skill: str; goal: str
      depends_on: list[str] = field(default_factory=list)
      inputs: list[str] = field(default_factory=list)
      outputs: list[str] = field(default_factory=list)
      success_check: SuccessCheck = field(default_factory=SuccessCheck)
      status: str = "pending"                 # pending | running | done | failed | skipped
      model: str | None = None
      attempts: int = 0
      checkpoint: str | None = None           # git sha taken when the current attempt started
      touched: list[str] = field(default_factory=list)   # project-relative paths this attempt wrote
      summary: str = ""
      last_failure: str = ""

  @dataclass
  class Plan:
      goal: str
      metric: str
      budget: dict
      steps: list[Step]
      @classmethod
      def from_dict(cls, d: dict) -> "Plan"   # ValueError with a readable message on wrong shape
      def to_dict(self) -> dict
      def validate(self, known_skills: set[str]) -> list[str]
          # error strings: duplicate id, id not matching STEP_ID, unknown depends_on, dependency cycle (names the cycle),
          # unknown skill, empty goal
      def ready(self) -> list[Step]           # pending steps whose depends_on are all done or skipped, in plan order
      def blocked(self) -> list[Step]         # pending steps with a failed dependency
      def step(self, step_id: str) -> Step
      def is_finished(self) -> bool           # every step is done, failed or skipped
      def merge_revision(self, new: "Plan") -> "Plan"   # takes new, but every step done in self is kept verbatim (by id)

  def load_plan(project: Project) -> Plan | None
  def save_plan(project: Project, plan: Plan) -> None
      # writes plan.json (indent 2) and PLAN.md: "# Plan: <goal>", then a table of id | skill | status | depends_on | goal
  ```

- [ ] **Step 1: Write the failing tests**

  ```python
  def test_roundtrip(tmp_path):                 # save_plan then load_plan: to_dict() is equal; PLAN.md contains every step id and status
  @pytest.mark.parametrize("mutation,needle", [
      ("dup_id", "duplicate"), ("bad_id", "identifier"), ("unknown_dep", "unknown dependency"),
      ("cycle", "cycle"), ("unknown_skill", "unknown skill"), ("empty_goal", "goal")])
  def test_validate_errors(mutation, needle):   # the errors list contains a message containing the needle
  def test_ready_respects_deps():               # a -> b, c independent: ready() == [a, c]; after a done: [b]
  def test_failed_dep_blocks():                 # a failed: ready() excludes b, blocked() == [b]
  def test_merge_keeps_done_steps():            # self has a done; new has a pending with another goal: merged a is done with the old goal
  def test_from_dict_rejects_garbage():         # from_dict({"steps": "x"}) raises ValueError
  ```

- [ ] **Step 2: Run them and confirm they fail.** Run: `venv/Scripts/python.exe -m pytest tests/test_plan.py -q -p no:cacheprovider`.
- [ ] **Step 3: Implement `plan.py`**
- [ ] **Step 4: Run the tests and confirm they pass.** Expected: 11 passed.
- [ ] **Step 5: Commit** `feat(workspace): plan.json model with validation and merge`

---

### Task 4: The event bus and ADK event normalisation

**Files:**
- Create: `beyondml/workspace/events.py`
- Test: `tests/test_events.py`

**Interfaces:**
- **Consumes:** Task 2's `Project`.
- **Produces:**

  ```python
  KINDS = ("message_delta", "tool_call", "tool_result", "plan_updated", "step_status", "generation", "trial",
           "chart", "artifact", "approval_request", "question", "budget", "error")

  class EventBus:
      def __init__(self, project: Project) -> None
      def emit(self, agent: str, kind: str, data: dict) -> dict
          # ValueError for an unknown kind; event = {"ts": time.time(), "agent", "kind", "data"};
          # appends one JSON line to agents/<agent>/events.jsonl (default=str for non-JSON values); puts it on every subscriber queue
      def subscribe(self) -> asyncio.Queue
      def unsubscribe(self, q: asyncio.Queue) -> None
      def replay(self, since: float | None = None) -> list[dict]   # all agents' files, merged and sorted by ts, filtered ts > since

  def normalize(ev: google.adk.events.Event) -> list[tuple[str, str, dict]]   # (agent, kind, data), in part order
  ```

- **Normalisation rules** (agent = `ev.author`):
  - A text part that is not a thought → `message_delta {"text", "partial": bool(ev.partial)}`.
  - A function_call named `adk_request_confirmation` → `approval_request {"call_id": fc.id, "tool": args["originalFunctionCall"]["name"], "args": args["originalFunctionCall"]["args"], "hint": args["toolConfirmation"]["hint"]}`.
  - Any other function_call → `tool_call {"id", "name", "args"}`.
  - A function_response → `tool_result {"id", "name", "response"}`.
  - `ev.actions.state_delta["bml_event"] == {"kind", "data"}` → that kind and data.
  - `ev.error_code` → `error {"code", "message"}`.

- [ ] **Step 1: Write the failing tests.** Build ADK `Event`s by hand with `google.genai.types` parts.

  ```python
  def test_emit_writes_and_fans_out(tmp_path):   # subscriber q gets the event; agents/x/events.jsonl has 1 line with keys ts, agent, kind, data
  def test_unknown_kind_rejected(tmp_path):      # pytest.raises(ValueError)
  def test_replay_merges_and_filters(tmp_path):  # emits for agents a, b, a: replay() is sorted by ts and has 3 events; replay(since=first ts) has 2
  def test_normalize_text_and_tools():           # one event with text + function_call + function_response: kinds in order
  def test_normalize_confirmation():             # adk_request_confirmation call: approval_request with tool "run_python" and hint
  def test_normalize_bml_event():                # state_delta {"bml_event": {"kind": "generation", "data": {"gen": 1}}}: ("tool", "generation", {"gen": 1})
  def test_normalize_skips_thoughts():           # Part(text="x", thought=True): []
  ```

- [ ] **Step 2: Run them and confirm they fail.** Run: `venv/Scripts/python.exe -m pytest tests/test_events.py -q -p no:cacheprovider`.
- [ ] **Step 3: Implement `events.py`**
- [ ] **Step 4: Run the tests and confirm they pass.** Expected: 7 passed.
- [ ] **Step 5: Commit** `feat(workspace): event bus with on-disk replay and ADK normalisation`

---

### Task 5: Toolkit file and control tools, and permission modes

**Files:**
- Create: `beyondml/toolkit/__init__.py` (empty), `beyondml/toolkit/env.py`, `beyondml/toolkit/files.py`, `beyondml/toolkit/control.py`, `beyondml/toolkit/registry.py`
- Test: `tests/test_toolkit_files.py`, `tests/test_toolkit_registry.py`

**Interfaces:**
- **Consumes:** `Project` and `PathError` (Task 2), `Plan`, `load_plan`, `save_plan` and `merge_revision` (Task 3), and `EventBus` (Task 4).
- **Produces in `env.py`:**

  ```python
  Asker = Callable[[str], Awaitable[str]]
  AUTONOMOUS_ANSWER = ("No human is available (autonomous mode). Choose the most reasonable option yourself "
                       "and state the assumption in one line of your finish_step summary.")

  class StepAborted(Exception): ...

  @dataclass
  class StepOutcome:
      finished: bool = False
      summary: str = ""
      outputs: list[str] = field(default_factory=list)
      touched: set[str] = field(default_factory=set)
      consecutive_errors: int = 0

  @dataclass
  class ToolEnv:
      project: Project
      bus: EventBus
      agent: str
      mode: str                                  # ask | auto | autonomous
      asker: Asker
      skill_names: set[str] = field(default_factory=set)
      outcome: StepOutcome | None = None
      on_touch: Callable[[list[str]], None] | None = None   # called with project-relative paths after every write
  ```

- **Produces in `files.py`:** `file_tools(env: ToolEnv) -> list[Callable]`. The tools are:
  - `read_file(path: str, max_lines: int = 400) -> dict`, returning `{"ok", "content", "truncated"}`.
  - `write_file(path: str, content: str) -> dict`. On success it emits `artifact {"path"}` and calls `on_touch`.
  - `list_files(path: str = ".", pattern: str = "*") -> dict`, returning `{"ok", "files": [...]}`. It is non-recursive unless the pattern contains `**`, and it skips `.git` and `.beyondml`.
- **Produces in `control.py`:** `control_tools(env: ToolEnv) -> list[Callable]`. The tools are:
  - `ask_user(question: str) -> dict`, returning `{"answer"}`. It emits `question`. In autonomous mode it returns `AUTONOMOUS_ANSWER` without calling the asker; otherwise it awaits `env.asker`.
  - `finish_step(summary: str, outputs: list[str]) -> dict`. It sets `env.outcome` to finished with that summary and outputs.
  - `update_plan(plan: dict) -> dict`. It runs `from_dict`, then `validate(env.skill_names)`, then `merge_revision` against `load_plan`, then `save_plan`, and emits `plan_updated`. It returns `{"ok": False, "errors": [...]}` without saving when invalid.
- **Produces in `registry.py`:**

  ```python
  NETWORK = re.compile(r"\b(requests|urllib|httpx|aiohttp|socket|http\.client|ftplib|smtplib)\b|https?://")
  def needs_confirmation(env: ToolEnv, tool: str, args: dict) -> bool
      # autonomous: never.
      # ask: tool in {"run_python", "install_package"}.
      # auto: install_package always; run_python only when NETWORK matches the script's text (read via env.project).
      # ponytail: regex network check, upgrade to the docker sandbox with network off when it matters
  def build_tools(env: ToolEnv, funcs: list[Callable], allowed: set[str] | None = None) -> list[FunctionTool]
      # keeps funcs whose __name__ is in allowed (None keeps all). Wraps each with functools.wraps (async stays async):
      # a result dict with ok False or an "error" key increments env.outcome.consecutive_errors, anything else resets it;
      # reaching 3 raises StepAborted.
      # FunctionTool(wrapped, require_confirmation=<callable(args) -> needs_confirmation(env, name, args)>),
      # with require_confirmation omitted in autonomous mode
  ```

- [ ] **Step 1: Write the failing tests**

  ```python
  async def test_file_tools_roundtrip(tmp_path):   # write_file("src/a.py", "x=1"): read_file gives "x=1"; list_files("src") includes "src/a.py";
                                                   #   artifact event emitted; on_touch received ["src/a.py"]
  async def test_file_tools_refuse_escapes(tmp_path):  # write_file("../x"), write_file("data/raw/a.csv"), read_file absolute path:
                                                   #   ok False with "error", and nothing created outside the project
  async def test_read_truncates(tmp_path):         # a 1000-line file with max_lines=400: 400 lines, truncated True
  async def test_ask_user_autonomous(tmp_path):    # mode autonomous: answer == AUTONOMOUS_ANSWER, asker not called, question event emitted
  async def test_ask_user_asks(tmp_path):          # mode auto with asker returning "yes": answer == "yes"
  async def test_finish_step_sets_outcome(tmp_path):
  async def test_update_plan_rejects_invalid(tmp_path):  # a plan with a cycle: ok False, plan.json unchanged
  async def test_update_plan_keeps_done(tmp_path):       # step a done on disk, the update marks it pending: saved plan still has a done
  @pytest.mark.parametrize("mode,tool,script,expected", [
      ("autonomous", "run_python", "import requests", False), ("ask", "run_python", "print(1)", True),
      ("auto", "run_python", "print(1)", False), ("auto", "run_python", "import requests", True),
      ("auto", "install_package", None, True), ("ask", "read_file", None, False)])
  def test_needs_confirmation(mode, tool, script, expected, tmp_path):
  def test_build_tools_filters_allowed(tmp_path):  # allowed {"read_file", "finish_step"}: [t.name for t in tools] == ["read_file", "finish_step"]
  async def test_three_errors_abort(tmp_path):     # call the wrapped write_file("../x") 3 times: the third raises StepAborted
  ```

- [ ] **Step 2: Run them and confirm they fail.** Run: `venv/Scripts/python.exe -m pytest tests/test_toolkit_files.py tests/test_toolkit_registry.py -q -p no:cacheprovider`.
- [ ] **Step 3: Implement `env.py`, `files.py`, `control.py` and `registry.py`**
- [ ] **Step 4: Run the tests and confirm they pass.** Expected: 17 passed.
- [ ] **Step 5: Commit** `feat(toolkit): file and control tools, permission modes`

---

### Task 6: `run_python` and `install_package`

**Files:**
- Create: `beyondml/toolkit/python.py`
- Test: `tests/test_toolkit_python.py`

**Interfaces:**
- **Consumes:** `ToolEnv` (Task 5), and `Project.path`, `ledger_append` and `checkpoint` (Task 2).
- **Produces:**

  ```python
  SECRET = re.compile(r"KEY|TOKEN|SECRET|PASSWORD", re.I)
  def clean_env(environ: Mapping[str, str]) -> dict[str, str]    # drops names matching SECRET; adds PYTHONIOENCODING=utf-8
  def run_script(project: Project, script: str, args: list[str], timeout: int, *, agent: str,
                 bus: EventBus | None = None, python: str = sys.executable) -> dict
  def python_tools(env: ToolEnv, python: str = sys.executable) -> list[Callable]
      # run_python(script: str, args: list[str] = [], timeout: int = 600) -> dict   (calls run_script, then on_touch(files_changed))
      # install_package(name: str) -> dict
  ```

- **`run_script` returns** `{"ok", "exit_code", "timed_out", "stdout_tail", "stderr_tail", "files_changed", "metrics", "seconds", "run_id"}`.
- **How it runs:**
  - The script resolves through `project.path`. It must exist and end in `.py`; otherwise the result is ok False.
  - It runs in a subprocess with `cwd=root` and `env=clean_env(os.environ)`, plus `BEYONDML_METRICS=<root>/.beyondml/runs/<run_id>/metrics.json`. `run_id` is `uuid4().hex[:8]`.
  - On Windows it starts with `CREATE_NEW_PROCESS_GROUP`, and a timeout runs `taskkill /T /F /PID`. On POSIX it uses `start_new_session=True`, and a timeout runs `os.killpg(SIGKILL)`.
  - `files_changed` compares (mtime_ns, size) snapshots taken before and after the run, excluding `.git`, `.beyondml` and `agents`.
  - `metrics` is read from `BEYONDML_METRICS` if that file exists.
  - The ledger record is `{"run_id", "agent", "script", "args", "exit_code", "timed_out", "seconds", "files_changed", "metrics", "data_hash"}`. `data_hash` is the sha256 of the sorted `(relpath, size, mtime_ns)` of files under `data/raw`.
  - It emits `artifact {"path"}` for each changed file.
- **`install_package` rules:**
  - `name` must match `^[A-Za-z0-9_.\[\],<>=!~-]+$` and must not start with `-`; otherwise ok False.
  - It runs `[python, "-m", "pip", "install", name]` with a 600 s timeout and returns `{"ok", "exit_code", "output_tail"}`.

- [ ] **Step 1: Write the failing tests**

  ```python
  async def test_runs_and_captures(tmp_path):        # src/a.py prints "hi": ok, exit_code 0, "hi" in stdout_tail, ledger has 1 record
  async def test_metrics_collected(tmp_path):        # the script writes {"acc": 0.9} to os.environ["BEYONDML_METRICS"]: metrics == {"acc": 0.9}, also in the ledger
  async def test_env_stripped(tmp_path, monkeypatch):# OPENAI_API_KEY set: the script prints os.environ.get("OPENAI_API_KEY") and gets "None"; PATH is still present
  async def test_timeout_kills_tree(tmp_path):       # the script spawns `python -c "import time; time.sleep(60)"`, prints the child pid, sleeps;
                                                     #   timeout=3: timed_out True, and within 5 s the child pid no longer exists (psutil-free check:
                                                     #   `tasklist /FI "PID eq <pid>"` on Windows, os.kill(pid, 0) on POSIX)
  async def test_files_changed(tmp_path):            # the script writes data/processed/x.csv: it appears in files_changed, and on_touch received it
  async def test_rejects_outside_script(tmp_path):   # run_python("../x.py"): ok False
  async def test_project_path_with_spaces(tmp_path): # project at tmp_path / "my project": runs with exit_code 0
  async def test_tail_200_lines(tmp_path):           # prints 500 lines: stdout_tail has 200 lines and ends with "499"
  async def test_install_package_validates(monkeypatch, tmp_path):  # "--index-url=x" and "a; b" give ok False;
                                                     #   "pandas" calls subprocess.run with [python, "-m", "pip", "install", "pandas"] (monkeypatched)
  ```

- [ ] **Step 2: Run them and confirm they fail.** Run: `venv/Scripts/python.exe -m pytest tests/test_toolkit_python.py -q -p no:cacheprovider`.
- [ ] **Step 3: Implement `python.py`**
- [ ] **Step 4: Run the tests and confirm they pass.** Expected: 9 passed.
- [ ] **Step 5: Commit** `feat(toolkit): run_python with ledger, env stripping, tree kill`

---

### Task 7: Skill discovery and the bundled `python_step` skill

**Files:**
- Create: `beyondml/harness/__init__.py` (empty), `beyondml/harness/skills.py`, `beyondml/skills/python-step/SKILL.md`
- Create fixtures: `tests/data/skills/tabular-demo/SKILL.md` (step), `tests/data/skills/leakage-demo/SKILL.md` (knowledge), `tests/data/skills/BadName/SKILL.md` (whose name doesn't match its folder)
- Test: `tests/test_skills.py`

**Interfaces:**
- **Consumes:** Task 2's `Project` and `beyondml_home`.
- **Produces:**

  ```python
  @dataclass
  class SkillInfo:
      name: str; description: str
      kind: str                      # "step" | "knowledge"; skills without metadata.beyondml are "knowledge"
      allowed_tools: set[str] | None # parsed from the space-delimited allowed-tools; None if absent
      model_hint: str | None; modality: str | None
      body: str; path: Path
      skill: "google.adk.skills.Skill"

  def discover(project: Project | None) -> dict[str, SkillInfo]
      # bundled beyondml/skills, then beyondml_home()/skills, then project/.beyondml/skills; a later level overrides by name.
      # A folder that fails to load (ADK raises ValueError) is skipped with logging.warning naming it.
  def step_skills(skills: dict[str, SkillInfo]) -> dict[str, SkillInfo]
  def knowledge_toolset(skills: dict[str, SkillInfo]) -> SkillToolset | None
      # SkillToolset([s.skill for knowledge skills], tool_filter=["list_skills", "load_skill", "load_skill_resource"]);
      # None if there are no knowledge skills
  ```

  Load each level with `google.adk.skills.load_skills_from_dir(dir)`. Read metadata from `skill.frontmatter`, where `metadata.beyondml` is a dict with keys `kind`, `step`, `modality` and `model_hint`.
- **The bundled `python-step/SKILL.md` frontmatter:**
  - `name: python-step`
  - `description`: "Generic plan step: write a standalone Python script under src/ that turns the step's inputs into its outputs, run it, fix it until it succeeds, then call finish_step."
  - `allowed-tools: read_file write_file list_files run_python ask_user finish_step`
  - `metadata: {beyondml: {kind: step, modality: any}}`
- **Its body** gives these rules, one short line each:
  - Put scripts in `src/<step_id>.py`, runnable as `python src/<step_id>.py` from the project root.
  - Read only `data/raw/` and upstream outputs; never write under `data/raw/`.
  - Write key numbers as JSON to `os.environ.get("BEYONDML_METRICS", "metrics.json")`.
  - Print a short summary.
  - After a failure, read `stderr_tail` and fix the script rather than starting over.
  - Finish with `finish_step(summary, outputs)` listing every output path.
- **Plan reference:** the plan references the skill as `python-step`. Step ids stay snake_case; skill names are kebab-case, as ADK requires.

- [ ] **Step 1: Write the failing tests**

  ```python
  def test_bundled_python_step_found():            # discover(None)["python-step"]: kind == "step", "run_python" in allowed_tools
  def test_project_overrides_bundled(tmp_path):    # project/.beyondml/skills/python-step with another description wins
  def test_home_level_loaded(tmp_path, monkeypatch):  # BEYONDML_HOME/skills/leakage-demo is discovered as kind "knowledge"
  def test_bad_skill_skipped(tmp_path, caplog):    # copy BadName into the project: not in the result, and the warning mentions "BadName"
  async def test_knowledge_toolset_tools(tmp_path):# tool names == {"list_skills", "load_skill", "load_skill_resource"} (via await ts.get_tools())
  def test_no_knowledge_gives_none():              # knowledge_toolset({}) is None
  ```

- [ ] **Step 2: Run them and confirm they fail.** Run: `venv/Scripts/python.exe -m pytest tests/test_skills.py -q -p no:cacheprovider`.
- [ ] **Step 3: Implement `skills.py`, write the bundled `SKILL.md` and the fixtures**
- [ ] **Step 4: Run the tests and confirm they pass.** Expected: 6 passed.
- [ ] **Step 5: Commit** `feat(harness): three-level skill discovery and python-step skill`

---

### Task 8: Intake

**Files:**
- Create: `beyondml/harness/intake.py`
- Test: `tests/test_intake.py`

**Interfaces:**
- **Consumes:** `Project` (Task 2), `EventBus` (Task 4), and the existing `beyondml.engine.profiler.DatasetProfiler` and `TargetIdentifier`.
- **Produces:**

  ```python
  @dataclass
  class InputReport:
      path: str                # relative to the project root
      modality: str            # tabular | timeseries | text | image | audio | unknown
      usable: bool
      summary: dict
      problem: str | None

  def detect_modality(path: Path) -> str
      # files: .csv .tsv .parquet .xlsx -> tabular, or timeseries when some column parses with pd.to_datetime for >= 90% of rows
      #   and is monotonic increasing; .txt .md .jsonl -> text; .wav .mp3 .flac -> audio; .jpg .jpeg .png .bmp .webp -> image.
      # directories: majority extension among their files decides; else unknown
  def run_intake(project: Project, bus: EventBus) -> list[InputReport]
      # every top-level entry in data/raw. Tabular/timeseries: read (csv/tsv via pandas) and give
      #   summary {rows, cols, dtypes, missing_pct, suggested_target (TargetIdentifier), target_confidence}.
      #   Other modalities: {files, total_bytes}.
      # Any exception or an empty file -> usable False, problem = the message.
      # Writes DATA.md: one "## <path>" section per input with modality, usable, summary as a bullet list, problem.
      # Emits step_status running then done as agent "intake".
  ```

- [ ] **Step 1: Write the failing tests**

  ```python
  def test_tabular_titanic(tmp_path):              # copy data/titanic.csv in: modality "tabular", usable, rows == 2099, suggested_target == "target"
  def test_timeseries_detected(tmp_path):          # csv with date column pd.date_range("2024-01-01", periods=50) and value: "timeseries"
  def test_image_dir(tmp_path):                    # data/raw/imgs with 3 files holding PNG magic bytes: "image", summary files == 3
  def test_unusable_inputs_reported(tmp_path):     # empty.csv and garbage.csv (os.urandom(256)): usable False with a problem;
                                                   #   DATA.md mentions both and contains the word "unusable"
  def test_intake_events(tmp_path):                # agents/intake/events.jsonl has step_status running and done
  ```

- [ ] **Step 2: Run them and confirm they fail.** Run: `venv/Scripts/python.exe -m pytest tests/test_intake.py -q -p no:cacheprovider`.
- [ ] **Step 3: Implement `intake.py`**
- [ ] **Step 4: Run the tests and confirm they pass.** Expected: 5 passed.
- [ ] **Step 5: Commit** `feat(harness): deterministic intake writes DATA.md`

---

### Task 9: ADK runtime helpers and the step-agent factory

**Files:**
- Create: `beyondml/harness/adk.py`, `beyondml/harness/factory.py`
- Test: `tests/test_factory.py`

**Interfaces:**
- **Consumes:**
  - `Project` and `settings` (Task 2), `Plan` and `Step` (Task 3), `EventBus` and `normalize` (Task 4).
  - `ToolEnv`, `StepOutcome`, `file_tools`, `control_tools` and `build_tools` (Task 5), and `python_tools` (Task 6).
  - `SkillInfo`, `step_skills` and `knowledge_toolset` (Task 7).
- **Produces in `adk.py`:**

  ```python
  APP_NAME = "beyondml"; USER_ID = "local"
  ModelResolver = Callable[[str, str | None], "str | BaseLlm"]       # (agent_name, requested model) -> model
  def default_resolver(project: Project) -> ModelResolver
      # requested or settings["model"]["default"]; "gemini-*" -> the string itself;
      # else LiteLlm(model=name, api_base=settings["model"]["api_base"], num_retries=3) for "ollama*" names,
      # LiteLlm(model=name, num_retries=3) otherwise (spec §9: LLM call failures retry with backoff)
  def session_service(project: Project) -> SqliteSessionService     # db_path=.beyondml/sessions.sqlite
  async def ensure_session(service, session_id: str) -> None         # create it if get_session returns None
  def user_text(text: str) -> types.Content
  def confirmation_reply(call_id: str, approved: bool) -> types.Content
      # function_response(id=call_id, name="adk_request_confirmation", response={"confirmed": approved}), role "user"

  @dataclass
  class TurnResult:
      confirmations: list[dict]        # approval_request data dicts (see Task 4)
      text: str                        # final non-partial text authored by runner.agent.name
      prompt_tokens: int
      completion_tokens: int

  async def run_turn(runner: Runner, session_id: str, message: types.Content, bus: EventBus,
                     *, max_llm_calls: int = 500) -> TurnResult
      # iterates runner.run_async(user_id=USER_ID, session_id=..., new_message=message, run_config=RunConfig(max_llm_calls=...));
      # emits every normalize(ev) tuple on the bus; sums usage_metadata prompt_token_count / candidates_token_count (None counts as 0).
      # LlmCallsLimitExceededError propagates.
  ```

- **Produces in `factory.py`:**

  ```python
  ROLE = ("You are the '{step_id}' agent in a machine-learning project built by BeyondML. "
          "You own exactly one plan step. Work only toward its goal and outputs.")
  RULES = ("- Call finish_step(summary, outputs) when the outputs exist.\n- Never write under data/raw/.\n"
           "- Write key numbers to os.environ.get('BEYONDML_METRICS', 'metrics.json').\n"
           "- If you are blocked, call finish_step explaining what blocks you.")
  def build_step_agent(project, bus, plan: Plan, step: Step, skills: dict[str, SkillInfo], *, resolver: ModelResolver,
                       mode: str, asker: Asker, outcome: StepOutcome, on_touch: Callable[[list[str]], None]) -> LlmAgent
  ```

- **The instruction** is built in this exact order, with these headings:
  1. `ROLE`
  2. `## Your step`: goal, then inputs, outputs and success_check as JSON
  3. `## Upstream results`: one bullet per done dependency, `id: summary`
  4. `## Previous attempt failed`: only when `step.last_failure` is set
  5. `## Project`: PROJECT.md
  6. `## Memory`: MEMORY.md
  7. `## How to do this step`: the skill body
  8. `## Rules`: RULES

  It is written to `agents/<step.id>/instruction.md` before the agent is returned.
- **Tools:** `env = ToolEnv(project, bus, agent=step.id, mode, asker, skill_names=set(step_skills(skills)), outcome=outcome, on_touch=on_touch)`. Then `build_tools(env, file_tools(env) + python_tools(env) + control_tools(env), allowed=skill.allowed_tools)`, plus `knowledge_toolset(skills)` when it is not None.
- **Model:** `resolver(step.id, step.model or skill.model_hint)`.
- **Agent name:** `step.id`.

- [ ] **Step 1: Write the failing tests** (they use `scripted_resolver` from `tests/scripted_llm.py`)

  ```python
  def test_instruction_written_in_order(tmp_path):  # instruction.md exists; the heading indexes ascend in the order listed above;
                                                    #   the skill body and PROJECT.md text are present
  def test_previous_failure_section(tmp_path):      # step.last_failure="boom": the heading and "boom" are present
  def test_tools_filtered_by_skill(tmp_path):       # skill allowed {"read_file", "finish_step"}: the agent's FunctionTool names == {"read_file", "finish_step"}
  def test_model_precedence(tmp_path):              # step.model "m1", skill hint "m2": resolver called with ("<id>", "m1"); with step.model None: "m2"
  def test_agent_name_is_step_id(tmp_path):
  async def test_run_turn_emits_and_collects(tmp_path):  # scripted agent: write_file("src/a.py", ...), then a text reply (mode auto):
                                                    #   the bus has tool_call and tool_result events; TurnResult.text == that reply; confirmations == []
  async def test_run_turn_confirmation(tmp_path):   # ask mode, scripted run_python call: TurnResult.confirmations[0]["tool"] == "run_python";
                                                    #   confirmation_reply(call_id, True) on the same session runs it (ledger has 1 record)
  ```

- [ ] **Step 2: Run them and confirm they fail.** Run: `venv/Scripts/python.exe -m pytest tests/test_factory.py -q -p no:cacheprovider`.
- [ ] **Step 3: Implement `adk.py` and `factory.py`**
- [ ] **Step 4: Run the tests and confirm they pass.** Expected: 7 passed.
- [ ] **Step 5: Commit** `feat(harness): ADK runtime helpers and step-agent factory`

---

### Task 10: The planner

**Files:**
- Create: `beyondml/harness/planner.py`
- Test: `tests/test_planner.py`

**Interfaces:**
- **Consumes:** `Plan`, `save_plan` and `merge_revision` (Task 3), `EventBus` (Task 4), `SkillInfo` and `step_skills` (Task 7), and `ModelResolver`, `session_service`, `ensure_session`, `run_turn` and `user_text` (Task 9).
- **Produces:**

  ```python
  class PlanError(Exception): ...
  async def make_plan(project, bus, skills, *, resolver: ModelResolver, goal: str, max_attempts: int = 3) -> Plan
  async def replan(project, bus, skills, *, resolver: ModelResolver, plan: Plan, failure: dict, max_attempts: int = 3) -> Plan
      # failure = {"step": id, "reason": str, "summary": str}
  ```

- **The agent:** an `LlmAgent` named `planner`, with the single tool `submit_plan(plan: dict) -> dict`.
  - On `Plan.from_dict` + `validate(set(step_skills(skills)))` errors, the tool returns `{"ok": False, "errors": [...]}`.
  - Otherwise it stores the plan and returns `{"ok": True}`.
- **Its instruction contains:**
  - the goal
  - PROJECT.md and DATA.md
  - the step-skill catalogue, one line each: `- <name> (<modality>): <description>`
  - the plan JSON shape, using the spec's example
  - these rules: step ids are snake_case identifiers; every step names a skill from the catalogue (use `python-step` when nothing fits); success checks name concrete files or metrics; no step reads an input marked unusable in DATA.md
- **The loop:**
  1. Session `planner-<n>`, where n is the count of existing planner sessions + 1.
  2. Send `user_text("Write the plan and call submit_plan.")`.
  3. After each turn, an accepted plan ends the loop. Otherwise send `"Call submit_plan with a corrected plan."`.
  4. Every `submit_plan` call and every turn without a call counts as one attempt. Exceeding `max_attempts` raises `PlanError` with the last errors.
- **On success:** `save_plan`, then emit `plan_updated {"steps": [ids]}` as agent `planner`.
- **`replan`** works the same way, with the current plan and the failure added to the instruction. The result is `old.merge_revision(new)`.

- [ ] **Step 1: Write the failing tests** (scripted planner)

  ```python
  async def test_valid_plan_first_try(tmp_path):   # plan.json and PLAN.md saved; plan_updated emitted; the planner model was called <= 2 times
  async def test_invalid_then_valid(tmp_path):     # 1st submit has a cycle, 2nd is valid: success; the model's 3rd request has a function_response containing "cycle"
  async def test_three_invalid_plans_raise(tmp_path):  # always unknown skill: pytest.raises(PlanError, match="unknown skill")
  async def test_no_tool_call_counts_attempt(tmp_path):  # the model only replies with text: PlanError after 3 turns
  async def test_replan_keeps_done(tmp_path):      # old plan a done, b failed; the model returns a pending (other goal) + c:
                                                   #   result a done with the old goal, c present
  ```

- [ ] **Step 2: Run them and confirm they fail.** Run: `venv/Scripts/python.exe -m pytest tests/test_planner.py -q -p no:cacheprovider`.
- [ ] **Step 3: Implement `planner.py`**
- [ ] **Step 4: Run the tests and confirm they pass.** Expected: 5 passed.
- [ ] **Step 5: Commit** `feat(harness): planner agent with validated submit_plan`

---

### Task 11: The executor

**Files:**
- Create: `beyondml/harness/executor.py`
- Test: `tests/test_executor.py`

**Interfaces:**
- **Consumes:**
  - `Project` (checkpoint, restore) from Task 2, and `Plan`, `Step`, `save_plan` and `merge_revision` from Task 3.
  - `EventBus` (Task 4), and `StepOutcome`, `StepAborted` and `Asker` (Task 5).
  - `run_script` (Task 6), `build_step_agent` (Task 9), and `session_service`, `ensure_session`, `run_turn`, `user_text` and `confirmation_reply` (Task 9).
- **Produces:**

  ```python
  Approver = Callable[[dict], Awaitable[bool]]
      # payloads: {"kind": "tool", "agent", "tool", "args", "hint"} | {"kind": "plan", "plan": dict}
      #         | {"kind": "budget", "agent": str | None, "reason": str}
  Replanner = Callable[[Plan, dict], Awaitable[Plan]]

  class Executor:
      def __init__(self, project, bus, plan: Plan, skills, *, resolver, mode: str, asker: Asker, approver: Approver,
                   replanner: Replanner | None = None, max_tool_calls: int = 40,
                   budget_minutes: float | None = None, budget_usd: float | None = None) -> None
      async def run(self) -> Plan
      def check(self, step: Step) -> tuple[bool, str]
          # success_check: if run -> run_script(..., agent="executor") must exit 0; every expects path exists;
          # metric/min -> that run's metrics[metric] >= min. Returns (ok, reason). An empty check passes once outcome.finished.
  ```

**Algorithm** (fixed here, because it is the core):

```
recover: for every step with status "running": project.restore(step.checkpoint, step.touched); status "pending"; touched []; save_plan
loop:
    if plan.is_finished(): break
    for step in plan.blocked(): status "skipped", summary f"blocked by failed dependency"; emit step_status
    launch every step in plan.ready() not already running, each as an asyncio task:
        attempts += 1; checkpoint = project.checkpoint(f"start {id}"); touched = []; status "running"; save_plan; emit step_status
    if no tasks are running: break
    await asyncio.wait(running, return_when=FIRST_COMPLETED); for each finished task -> (step, outcome, error):
        ok, reason = (False, error) if error else self.check(step) if outcome.finished else (False, "agent ended without finish_step")
        if ok:   status "done", summary = outcome.summary; project.checkpoint(f"done {id}")
        elif attempts < 2:   project.restore(step.checkpoint, step.touched); last_failure = reason; status "pending"
        elif replanner:   plan = plan.merge_revision(await replanner(plan, {"step": id, "reason": reason, "summary": outcome.summary}));
                          emit plan_updated
        else:   status "failed", last_failure = reason
        save_plan; emit step_status {"id", "status", "reason"}
    budget: minutes since run() started > budget_minutes, or cost > budget_usd ->
            emit budget; if not await approver({"kind": "budget", "agent": None, "reason": ...}): break (pending steps stay pending)
return plan
```

- **One step attempt** (`_attempt`):
  1. Build the agent with `on_touch` appending to `step.touched` and saving the plan, under an `asyncio.Lock`.
  2. Create a `Runner(app_name=APP_NAME, agent=agent, session_service=...)` on session `f"{id}-{attempts}"`.
  3. Send the first message `user_text("Start your step.")`.
  4. Loop on `run_turn(..., max_llm_calls=max_tool_calls + 1)`:
     - **Confirmations requested:** for each, `approved = await approver({"kind": "tool", ...})`, then send `confirmation_reply`. Only in modes other than autonomous.
     - **Turn ended without `finish_step`:** nudge once with `"Call finish_step when your outputs exist, or explain what blocks you."` On the second miss, stop.
     - **`LlmCallsLimitExceededError`:** emit budget. If `await approver({"kind": "budget", "agent": id, "reason": "tool-call cap"})`, continue with `"Continue."`; otherwise the attempt fails.
     - **`StepAborted`:** the attempt fails with reason `"three tool errors in a row"`.
- **Cost:** after each turn, add `litellm.cost_per_token(model=<model name>, prompt_tokens=p, completion_tokens=c)`. Any exception counts as 0.
- `# ponytail: path-scoped restore assumes parallel steps write disjoint files; a shared file written by two running steps can be restored over. Add per-step output ownership if that happens.`
- `# ponytail: plan edits from the lead during a run are not merged live. Sub-project 4 adds a reload-and-merge per scheduling round.`

- [ ] **Step 1: Write the failing tests.** They use scripted step agents that call the real tools, in a temp project with `python-step`.

  ```python
  async def test_two_steps_in_order(tmp_path):       # b depends on a, each writes and runs src/<id>.py, then finish_step: both done;
                                                     #   ledger has >= 2 records; agents/a and agents/b events exist;
                                                     #   git log has "start a", "done a", "start b", "done b" in that order
  async def test_independent_steps_parallel(tmp_path):  # two independent steps whose scripts sleep 1.5 s: total wall < 2.6 s
  async def test_check_gates_done(tmp_path):         # finish_step but the expected output is never created: attempts == 2, status failed;
                                                     #   its dependent is skipped
  async def test_retry_restores_touched(tmp_path):   # attempt 1 writes junk.txt then finishes without its output; attempt 2's script
                                                     #   asserts (callable reading last_response of list_files) junk.txt is gone, then succeeds: done
  async def test_replanner_after_two_failures(tmp_path):  # the fake replanner swaps the failing step for a trivial one: plan finished, replanner called once
  async def test_ask_mode_approval(tmp_path):        # mode ask: approver gets kind "tool" with tool "run_python"; approve -> ledger 1 record;
                                                     #   deny (fresh project) -> ledger 0 records
  async def test_autonomous_never_asks(tmp_path):    # an approver that raises if called: the run completes
  async def test_tool_call_cap(tmp_path):            # max_tool_calls=2, the model calls list_files forever, approver returns False: failed, budget event emitted
  async def test_three_tool_errors_fail(tmp_path):   # write_file("../x") three times: the attempt fails with reason "three tool errors in a row"
  async def test_resume_restores_only_touched_files(tmp_path):
      # step a done (out_a.csv committed); step b's tool awaits an asyncio.Event after writing partial.txt; cancel run() while b is running:
      # plan.json shows b running. A new Executor(...).run() with b scripted to succeed: partial.txt removed before b restarts,
      # out_a.csv and data/raw intact, b done
  async def test_events_shape(tmp_path):             # every line of every agents/*/events.jsonl has exactly the keys {"ts", "agent", "kind", "data"}
  ```

- [ ] **Step 2: Run them and confirm they fail.** Run: `venv/Scripts/python.exe -m pytest tests/test_executor.py -q -p no:cacheprovider`.
- [ ] **Step 3: Implement `executor.py` following the algorithm above**
- [ ] **Step 4: Run the tests and confirm they pass.** Expected: 11 passed.
- [ ] **Step 5: Commit** `feat(harness): topological executor with checks, retry, replan, resume`

---

### Task 12: The lead agent

**Files:**
- Create: `beyondml/harness/lead.py`
- Test: `tests/test_lead.py`

**Interfaces:**
- **Consumes:** `ToolEnv`, `file_tools`, `control_tools` and `build_tools` (Task 5), `SkillInfo` (Task 7), and `ModelResolver`, `session_service`, `ensure_session`, `run_turn` and `user_text` (Task 9).
- **Produces:**

  ```python
  def make_lead(project, bus, skills, *, resolver: ModelResolver, mode: str, asker: Asker) -> LlmAgent
      # name "lead"; tools build_tools(env, file_tools(env) + control_tools(env), allowed={"read_file", "list_files", "ask_user", "update_plan"});
      # instruction: "You are the lead ML engineer for this project. Answer the user, explain decisions, and change the plan with
      # update_plan when asked; you never run code yourself." + PROJECT.md + MEMORY.md + PLAN.md
  async def chat(project, bus, lead: LlmAgent, text: str, *, session_id: str = "lead") -> str   # one turn; returns TurnResult.text
  ```

- [ ] **Step 1: Write the failing tests**

  ```python
  async def test_lead_adds_step(tmp_path):         # scripted update_plan adds step c: plan.json has c; plan_updated emitted by agent "lead"
  async def test_lead_invalid_update(tmp_path):    # update with an unknown skill: plan.json unchanged; the model's next request has the error
  async def test_lead_has_no_run_python(tmp_path): # the tool names exclude run_python and write_file
  ```

- [ ] **Step 2: Run them and confirm they fail.** Run: `venv/Scripts/python.exe -m pytest tests/test_lead.py -q -p no:cacheprovider`.
- [ ] **Step 3: Implement `lead.py`**
- [ ] **Step 4: Run the tests and confirm they pass.** Expected: 3 passed.
- [ ] **Step 5: Commit** `feat(harness): lead agent for chat and plan edits`

---

### Task 13: `run_goal`, the CLI and the end-to-end test

**Files:**
- Create: `beyondml/harness/run.py`
- Modify: `beyondml/cli.py`
- Test: `tests/test_harness_e2e.py`, `tests/test_cli.py`
- Modify: `.scratch/MAP-001-constrained-decoding.md`, `.scratch/TICKET-001..008-*.md`, and `.scratch/TICKET-L02..L05-*.md` (status line only)

**Interfaces:**
- **Consumes:** everything above.
- **Produces:**

  ```python
  async def run_goal(root: Path, goal: str | None, data: Sequence[Path] = (), *, mode: str | None = None,
                     model: str | None = None, resolver: ModelResolver | None = None,
                     asker: Asker, approver: Approver) -> Plan
  ```

  **`run_goal` steps:**
  1. Open the project, or `Project.init(root, goal)` if it is missing.
  2. `add_data` each path, then `checkpoint("add data")`.
  3. `skills = discover(project)`. Set `mode = mode or settings["run"]["mode"]` and `resolver = resolver or default_resolver(project)`. When `model` is given, a wrapper forces it as the requested model.
  4. If `load_plan` returns a plan that isn't finished, resume it and skip intake and planning.
  5. Otherwise, with `goal` required (`click.UsageError` from the CLI layer, `ValueError` here): `run_intake`, then `make_plan`.
  6. Unless mode is autonomous, `await approver({"kind": "plan", ...})`. On False, return the plan with nothing run.
  7. `Executor(..., replanner=lambda p, f: replan(...), max_tool_calls / budget_minutes / budget_usd from settings).run()`.

- **The CLI** is a `click` group with `invoke_without_command=True`. With no subcommand it opens the legacy TUI, just like the old `run` body. Its commands:
  - `beyondml tui`: the legacy TUI.
  - `beyondml init [DIR] --goal TEXT`.
  - `beyondml run [GOAL] --data PATH (multiple) --project DIR (default ".") --mode [ask|auto|autonomous] --model TEXT`. It prints each bus event via `format_event`, asks questions with `click.prompt`, and handles approvals with `click.confirm`.
  - `benchmark` and `web` are unchanged.

  ```python
  def format_event(ev: dict) -> str | None
      # "[agent] kind: short": message_delta (non-partial) -> text; tool_call -> "name(args≤80 chars)";
      # tool_result -> "name ok" or "name error: ..."; step_status -> "id status"; other kinds -> json.dumps(data)[:120].
      # Returns None for partial deltas.
  ```

- **The e2e scripts** live in the test module.
  - `PREPARE_PY`: reads `data/raw/titanic.csv`, fills numeric NaNs with the median, writes `data/processed/clean.csv`.
  - `TRAIN_PY`: reads `clean.csv`, does `train_test_split(test_size=0.2, random_state=0)`, fits `LogisticRegression(max_iter=1000)` on the numeric columns except `target`, writes `{"accuracy": acc}` to `BEYONDML_METRICS`, and dumps `models/model.joblib`.
  - The planner submits `prepare` (python-step, expects `data/processed/clean.csv`), then `train` (python-step, depends on `prepare`, `success_check {"run": "src/train.py", "expects": ["models/model.joblib"], "metric": "accuracy", "min": 0.6}`).

- [ ] **Step 1: Write the failing tests**

  ```python
  async def test_e2e_titanic_scripted(tmp_path):
      # run_goal(tmp_path / "proj", "predict survival", [Path("data/titanic.csv")], mode="autonomous", resolver=scripted, ...)
      # assert plan.is_finished() and all steps are done
      # assert subprocess.run([sys.executable, "src/train.py"], cwd=proj).returncode == 0     # the repo runs without BeyondML
      # assert len([r for r in project.ledger() if r["agent"] in ("prepare", "train")]) >= 2
      # assert (proj / "agents/train/instruction.md").exists(); events exist for intake, planner, prepare and train
      # assert "titanic.csv" in (proj / "DATA.md").read_text()
  async def test_run_goal_resumes_existing_plan(tmp_path):  # plan.json with a done and b pending; the resolver raises for "planner";
                                                            #   run_goal(goal=None) finishes b
  async def test_plan_rejected_runs_nothing(tmp_path):      # mode auto, approver returns False for kind "plan": every step pending, ledger empty
  def test_cli_init(tmp_path):                              # CliRunner: init <dir> --goal x creates .beyondml and PROJECT.md contains "x"
  def test_cli_run_help():                                  # "--mode" and "autonomous" appear in the help output
  def test_cli_has_tui_command():                           # "tui" is in main.commands
  def test_format_event():                                  # a partial delta -> None; tool_call -> "[a] tool_call: run_python(...)"
  ```

- [ ] **Step 2: Run them and confirm they fail.** Run: `venv/Scripts/python.exe -m pytest tests/test_harness_e2e.py tests/test_cli.py -q -p no:cacheprovider`.
- [ ] **Step 3: Implement `run.py` and the CLI changes**
- [ ] **Step 4: Run the full suite.** Run: `venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider`. Expected: 0 failed.
- [ ] **Step 5: Run a smoke test against a real local model, if one is available.** Check that Ollama is running (`curl -s http://127.0.0.1:11434/api/tags`). If so, run: `venv/Scripts/python.exe -m beyondml.cli run "predict survival" --data data/titanic.csv --project <scratch dir> --mode autonomous --model ollama_chat/qwen2.5:14b`. Record in the commit message whether the plan finished. If Ollama isn't running, write "real-model smoke not run: Ollama unavailable" instead. This answers spec §11 item 3.
- [ ] **Step 6: Set `status: superseded`**, with one line pointing to the spec, in MAP-001, TICKET-001..008 and L02–L05.
- [ ] **Step 7: Commit** `feat: beyondml run/init on the ADK harness; scripted titanic e2e`
