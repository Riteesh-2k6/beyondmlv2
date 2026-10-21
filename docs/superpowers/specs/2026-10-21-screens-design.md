# Screens: gateway, Textual TUI, web UI — design

- **Date:** 2026-10-21
- **Status:** approved in advance by the owner; decisions made in-session are recorded as rulings in the plan
- **Parent:** `2026-10-19-beyondml-ml-harness-design.md` §8 (sub-project 4)
- **Depends on:** sub-project 1 (branch `harness/core`): `run_goal`, `EventBus`, `make_lead`/`chat`, `plan.json`, `agents/<id>/instruction.md`, `experiments.jsonl`

## 1. Intent

Every agent gets a screen that shows exactly how it was instructed and what it is doing. Two clients, one live session: an app-like Textual TUI and a polished React web UI, both fed by the same event stream. The gateway is the one process that owns the session; the TUI runs it in-process, the web attaches over HTTP.

## 2. Architecture

```
beyondml/gateway/
  session.py    Session: one project, its EventBus, the running harness, pending prompts, the lead
  state.py      fold(events) -> SessionState   pure; used by the TUI and by GET /projects/{p}
  app.py        FastAPI app: REST + SSE + static web UI
  serve.py      run the app in a background thread (TUI) or foreground (beyondml serve)
beyondml/tui/
  app.py        BeyondMLApp: ContentSwitcher over five views + ChatDock + ApprovalModal
  views/        plan_board.py, agent.py, experiments.py, files.py, chat.py, approvals.py
beyondml/web/ui/  React + Vite + TS + Tailwind + Plotly, built to dist/, served by the gateway
```

- **One `Session` per project.** It opens (or inits) the `Project`, owns the `EventBus`, starts `run_goal` as a task, parks `asker`/`approver` calls on futures until an answer arrives, and builds the lead for chat. `run_goal` gains an optional `bus=` parameter so the session's bus is the only bus.
- **The session lives on the gateway's event loop.** In `beyondml` (TUI mode) that loop runs in a daemon thread so git and subprocess calls in the harness never stall the terminal. The TUI subscribes to the bus from its own loop (the bus delivers cross-thread) and submits commands with `run_coroutine_threadsafe`. In tests the session binds to the test loop and the same code path awaits directly.
- **The web attaches over SSE and REST only**, so a browser can follow a TUI-started run.

## 3. Event contract additions

The core kinds are unchanged. This sub-project fixes the data shapes clients render:

| Kind | Author | Data | Emitted by |
|---|---|---|---|
| `question` | the asking agent | `{id, question}` pending; `{id, question, answer}` resolved | Session asker (the id-less `question` from `ask_user` is ignored by clients) |
| `approval_request` | `payload.agent` / `planner` / `executor` | `{id, kind: tool\|plan\|budget, ...payload}` pending; same `+ answer: bool` resolved | Session approver (ADK's id-less `approval_request` is ignored by clients) |
| `message_delta` | `user` | `{text, partial: false}` | Session chat, so the transcript replays |
| `generation` | the evolve step | `{gen, best_fitness, mean_fitness, diversity, evals, seconds, best_params}` | sub-project 3's `evolve` tool (mirrors `GenStats`) |
| `trial` | the tune step | `{trial, value, best_value, params, seconds}` | sub-project 3's `tune` tool |
| `chart` | any | `{title, kind: line\|bar\|scatter, x: [...], series: {name: [...]}, path?}` | sub-project 3's `chart` tool |

A **prompt** is a `question` or `approval_request` event carrying an `id`. The same `id` with an `answer` key resolves it. Pending prompts are the ids seen without a resolution; both clients fold them this way, so reconnecting shows what is still waiting.

## 4. Gateway

FastAPI, bound to `127.0.0.1:7777` by default. No ADK `get_fast_api_app`: it would add ADK's own session routes and auth model for no benefit here.

| Route | Returns |
|---|---|
| `GET /projects` | `[{name, root, goal, running}]` |
| `POST /projects {path, goal?}` | inits or opens a project at an absolute path, registers it |
| `GET /projects/{p}` | `{name, root, goal, mode, model, running, pending: [prompt data...]}` |
| `GET /projects/{p}/plan` | `plan.json` plus per-step `started`, `finished`, `metric` folded from events and the ledger; 404 before planning |
| `GET /projects/{p}/agents/{id}` | `{id, instruction, status, skill, inputs, outputs, tools, model}` |
| `GET /projects/{p}/files/{path}` | the file through `Project.path`; 404 when missing, 400 when the path escapes; `?list=1` lists a directory |
| `GET /projects/{p}/events?since=` | SSE: replay (`ts > since`, or `Last-Event-ID`), then tail; `id:` is the event `ts`; `: ping` every 15 s |
| `POST /projects/{p}/input` | `{type: chat, text}` → `{reply}`; `{type: answer, id, answer}` → `{ok}` (404 for an unknown id) |
| `POST /projects/{p}/run {goal?, data?: [path], mode?, model?}` | starts `run_goal`; 409 while a run is active |
| `POST /projects/{p}/stop` | cancels the run; steps stay `running` on disk and resume next time |
| `POST /projects/{p}/upload` (multipart) | saves into `data/raw/<name>`, returns `{path}` |
| `GET /{anything else}` | the built web UI (`index.html` fallback for client routes) when `web/ui/dist` exists |

The registry of projects is in memory, seeded with the project the CLI was started on.

## 5. Screens (both clients)

1. **Plan board.** One card per step: status glyph, id, skill, model (step model or project default), elapsed (from `step_status` timestamps, ticking while running), key metric (the plan's metric from this step's latest ledger record, else the first metric it reported), last failure. Beside it the dependency graph as topological layers (TUI: boxes per layer with arrows between layers; web: SVG nodes with edges). Click or Enter opens the agent.
2. **Agent screen.** Header: id, status, skill, model, attempts. Tabs:
   - *Instruction*: the exact `agents/<id>/instruction.md`, with skill, inputs and the granted tools above it.
   - *Live*: messages, tool calls (name and arguments) and tool results (ok/error), in order, from this agent's events.
   - *Outputs*: files (`artifact` events and the step's declared outputs), metrics (the agent's latest ledger metrics), charts (`chart` events, rendered with plotext in the TUI and Plotly in the web).
   - *Evolve*: shown once a `generation` event exists. Best and mean fitness per generation, diversity per generation, the best genome as a key/value table, and the running totals (evals, seconds). `trial` events get the same tab under the name *Tune*.
3. **Chat dock.** Always visible, bottom of the layout. The transcript is the `message_delta` events of `user` and `lead`; Enter sends to the lead.
4. **Approvals modal.** Opens when a prompt becomes pending. Plan approvals list the steps; tool approvals show the tool and arguments; budget approvals the reason; questions take free text. `y`/`n`/Enter answer it; Esc hides it and `a` reopens.
5. **Experiments.** A table from `experiments.jsonl`: time, agent, script, exit, seconds, metrics; newest first; reloads on each `tool_result`.
6. **Files.** A tree of the project (hiding `.git` and `.beyondml`) and a preview pane: text as text, CSV as the first rows.

**Visual tone (PRODUCT.md):** obsidian `#09090B` with `#18181B` surfaces and `#27272A` hairlines, one orange accent `#F97316`, Geist Mono for data. Status colours: pending muted, running orange, done green `#22C55E`, failed red `#EF4444`, skipped dim. The same tokens drive the Textual theme and the Tailwind config.

## 6. CLI

- `beyondml [--project DIR] [--port 7777] [--no-serve]` opens the TUI; the gateway runs in-process so a browser at `http://127.0.0.1:7777` sees the same session. Outside a project it prints the `beyondml init` hint and exits 2.
- `beyondml serve [--project DIR] [--host 127.0.0.1] [--port 7777]` runs the gateway headless.
- `beyondml tui` is an alias for the TUI; the legacy `tui_app.py` and `beyondml web` are removed with `web/runner.py`, `web/markup.py`, `web/server.py`, `web/models.py` and `tests/test_web_server.py`.

## 7. Testing

- **Session and gateway** (`tests/test_gateway.py`): `httpx.AsyncClient` over `ASGITransport`, scripted agents from `tests/scripted_llm.py`. Covers the REST shapes, the path guard, SSE replay then tail, a run that asks a question and is answered through `POST /input`, a plan approval through the same route, and chat to a scripted lead.
- **TUI** (`tests/test_tui.py`): Textual pilot on a temp project with a plan and events on disk; synthetic `generation` and `chart` events; a scripted run that raises an approval modal; keyboard navigation across all six screens.
- **Web**: `npm run build` (tsc strict + Vite) must pass; the design detector runs over the changed UI; one browser round against the gateway on a project with real and synthetic events.

## 8. Out of scope

Authentication, remote (non-loopback) deployment, a persistent project registry, multi-user sessions, and the `evolve`/`tune`/`chart` tools themselves (sub-project 3).
