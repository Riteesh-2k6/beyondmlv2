# Screens (sub-project 4): report

- **Branch:** `claude/screens-b92631` (worktree `.claude/worktrees/screens-b92631`), cut from `harness/core` at `3ab1b5c`, with `refs/heads/harness/core` (`a7ac96c`, the final-review fixes) merged in. Nothing pushed, nothing merged into `main` or `harness/core`.
- **Spec:** `docs/superpowers/specs/2026-10-21-screens-design.md`. **Plan and rulings:** `docs/superpowers/plans/2026-10-21-screens.md`.

## What works

**Gateway** (`beyondml/gateway/`, `beyondml serve [--project DIR] [--host 127.0.0.1] [--port 7777] [--mode] [--model]`)
- `Session` wraps `run_goal` (which gained `bus=`): one bus per project, `asker`/`approver` park on futures, answers arrive as `{type: answer, id, answer}`; chat rebuilds the lead per turn; `stop()` cancels the run and the executor leaves the steps `running` on disk for the next resume.
- Routes per spec §4: `GET/POST /projects`, `GET /projects/{p}` (summary with pending prompts), `/plan` (plan.json plus started/finished/metric per step), `/agents/{id}` (instruction, skill, inputs, outputs, tools, model, status), `/files/{path}` (path-guarded, `?list=1`), `/events` (SSE: replay by `since` or `Last-Event-ID`, then tail, 15 s pings), `POST /input`, `POST /run`, `POST /stop`, `POST /upload`, and the built web UI with an `index.html` fallback.
- `beyondml/gateway/state.py` is the one fold both clients use (`SessionState.apply`, `plan_view`, `agent_view`); `src/session.ts` mirrors it.

**TUI** (`beyondml/tui/`, `beyondml [--project DIR] [--port 7777] [--no-serve]`, `beyondml tui` as alias)
- Plan board (step cards with status, model, elapsed, key metric, failure; layered dependency graph; a run form when there is no plan), agent screen (Instruction = the exact `instruction.md` with skill/inputs/outputs/tools; Live; Outputs with files, metrics and plotext charts; Evolve/Tune tab from `generation`/`trial` events), chat dock on every screen, approvals modal (y/n, numbered choices with the recommended one marked, free text), experiments table, files tree with preview.
- The gateway runs in a daemon thread under the TUI, so a browser on `http://127.0.0.1:7777` follows the same session. Keys: `1-4` screens, `a` prompts, `c` chat, `esc` leave an input, `q` quit.

**Web UI** (`beyondml/web/ui/`, React 19 + Vite 8 + TypeScript + Tailwind + Plotly basic dist, lazy-loaded)
- The same six screens over SSE + REST, a projects page, keyboard shortcuts, a collapsible chat dock, prompts as a modal with a "later" path and a waiting badge.
- Polished with impeccable: established world inherited from PRODUCT.md (obsidian, one orange, Geist/Geist Mono), drawn SVG status marks, charts in orange plus neutrals, contrast floor met (`muted` raised to `#85858F`). Finish review ran as a fresh read-only subagent from the skill's degraded reference: `disposition: fix` with seven material fixes, all applied and recaptured; the verdict is recorded below. `DESIGN.md` and `.impeccable/design.json` were written by the documenter from the built code.
- `npm run build` passes; `dist/` is gitignored (ruling 21) and served by the gateway when present.

**Contracts from sub-project 3** (spec §3): `question {question, options, recommended}`, `trial {number, value, params, state}`, `chart {title, path}` (Plotly figure JSON under `reports/charts/`) are rendered by both clients; the web fetches the figure through `/files`, the TUI approximates its traces with plotext.

**Retired:** `tui_app.py`, `beyondml/web/{server,models,runner,markup}.py`, `tests/test_web_server.py`, `beyondml web`. `pyproject.toml` now declares `fastapi`, `uvicorn`, `python-multipart` (the old server used them undeclared) and `httpx` for tests.

**Tests:** `tests/test_session.py` (7), `tests/test_gateway.py` (13, real uvicorn on a random port for SSE), `tests/test_tui.py` (12, Textual pilot), plus the CLI additions. Full suite after the merge: see the verification line below.

## How to try it

```bash
# from the worktree root, with the shared venv
C:\Users\Riteesh\Programming\BeyondML\venv\Scripts\python.exe -m beyondml.cli init C:\tmp\churn --goal "predict survival"
cd C:\tmp\churn
C:\Users\Riteesh\Programming\BeyondML\venv\Scripts\python.exe -m beyondml.cli            # TUI + gateway on :7777
C:\Users\Riteesh\Programming\BeyondML\venv\Scripts\python.exe -m beyondml.cli serve      # headless gateway only
```

Build the web UI once with `cd beyondml/web/ui && npm install && npm run build`; the gateway then serves it at `/`. A real model (Ollama by default, `--model` to override) is needed for an actual run; the tests and the demo script in the session scratchpad drive everything with `tests/scripted_llm.py`.

## What's left

- **No real-model run was done from the screens** (Ollama was not exercised in this session); every live path was proven with scripted agents. The first real `beyondml` run will be the true end-to-end check of approvals and chat latency.
- **`chart` and `trial` rendering is tested with synthetic events** until sub-project 3 lands the tools; the Plotly-figure path reads `{data, layout}` and tolerates missing files.
- **The web has no JS unit tests** (ruling 17): tsc strict + the build + the browser round are its verification. If the fold grows, add vitest for `session.ts`.
- **Code preview is untinted** in the web files view (the reviewer's ceiling note); a small tokenizer would add syntax colour.
- **Projects registry is in memory** (ruling 10); projects opened from the web are forgotten on restart.
- Impeccable reports `PRODUCT.md` as a legacy schema (`route: init`); not acted on, as instructed by the skill.

## Rulings

See the numbered list in `docs/superpowers/plans/2026-10-21-screens.md` (23 rulings).
