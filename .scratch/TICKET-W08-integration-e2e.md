# [TICKET-W08] End-to-end integration: launch script, static serving, full pipeline run
parent: MAP-002
labels: wayfinder:task
status: open
blocked-by: TICKET-W04, TICKET-W05, TICKET-W06, TICKET-W07
blocks: none

## Question

Does the full stack run end-to-end: `beyondml web` → browser → SetupPage → file upload → PipelineScreen live → completion?

Deliverables:
1. `beyondml web` CLI subcommand added to `beyondml/cli.py`. Launches uvicorn on port 7777 and opens `http://localhost:7777` in the default browser.
2. Vite prod build (`npm run build`) output lands in `beyondml/web/ui/dist/`. FastAPI mounts it at `/`.
3. Dev mode: `npm run dev` serves React on 5173, proxy `/api` and `/ws` to uvicorn on 7777 via vite.config.ts.
4. Smoke test: upload `data/test_beyondml.py`'s CSV fixture, run autonomous path, confirm all WebSocket event types are received, pipeline completes without exception, at least one chart renders inline.
5. `PYTHONPATH=. pytest tests/test_web_server.py -v` passes: covers upload endpoint, WS message schema, input_request/response round-trip, markup parser.

Also: confirm existing TUI (`beyondml run`) still works — `beyondml/cli.py` change must not break the `run` subcommand.

## Verified 2026-10-19
- No smoke test (item 4): nothing uploads the CSV fixture, runs autonomous path and asserts all WS event types and an inline chart.
- TUI 'beyondml run' still working not verified by any test.
- Done: 'beyondml web' in cli.py, dist/ build present, StaticFiles mount, vite proxy, tests/test_web_server.py (78 pass). cli binds 0.0.0.0, not localhost only.
