# [TICKET-W02] FastAPI backend: project structure, WebSocket protocol, agent runner
parent: MAP-002
labels: wayfinder:task
status: open
blocked-by: TICKET-W01
blocks: TICKET-W04, TICKET-W05, TICKET-W06, TICKET-W07

## Question

What is the exact structure of `beyondml/web/` and the WebSocket protocol the backend uses to stream pipeline events to the browser?

Design decisions:
1. File layout: `beyondml/web/server.py` (FastAPI app), `beyondml/web/runner.py` (pipeline worker), `beyondml/web/models.py` (Pydantic schemas).
2. WebSocket message schema: `{type: "log"|"stat"|"chart"|"input_request"|"done"|"error", payload: {...}}`. Each agent `_log()` call emits a `log` event. Stat updates emit `stat`. Chart data emits `chart` with Plotly JSON.
3. How `get_user_input` is bridged: the backend sends `input_request` and awaits a `user_response` message from the frontend — an async queue per connection.
4. File upload: multipart POST `/api/upload` saves to a temp dir; `/api/run` starts the pipeline WebSocket with the saved path.
5. Static serving: `fastapi.staticfiles.StaticFiles` mounts the Vite build output.
6. Entry point: `beyondml web` CLI subcommand launches uvicorn.

Test cases:
- `test_ws_log_event`: mock agent emits a log line → WebSocket message is `{type:"log", payload:{level:"info", content:"..."}}`
- `test_stat_event`: stat update → `{type:"stat", payload:{key:"shape", value:"200×6"}}`
- `test_input_request_roundtrip`: backend sends `input_request`, client sends `user_response` → `get_user_input()` resolves
- `test_chart_event`: Plotly JSON delivered as `{type:"chart", payload:{plotly_json:{...}}}`
