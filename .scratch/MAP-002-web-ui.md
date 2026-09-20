# [MAP-002] BeyondML Web UI — Wayfinder Map
labels: wayfinder:map
status: open

## Destination

A production-ready web interface (`beyondml/web/`) that faithfully maps the TUI's 3-panel layout and full pipeline orchestration to the browser, with inline Plotly/Chart.js graph rendering replacing the TUI's file:// chart pop-outs. The web UI replaces `tui_app.py` as the primary interactive surface without removing the CLI. FastAPI serves agents via WebSocket; React (Vite + Tailwind) renders the UI. The design is an obsidian-terminal aesthetic — not a generic dashboard, not a generic dev tool.

## Notes

- TUI screens to map: ConfigScreen (setup modal) → WelcomeScreen (file upload + params) → PipelineScreen (3-panel: pipeline tree / live log + charts / stats + data preview) → CompletionModal.
- Agent pipeline: all agent imports stay in `beyondml/agents/`; web backend re-uses them verbatim via an async background worker.
- Graph rendering: interactive charts currently open in browser tabs via `webbrowser.open(file://)`. In the web UI they render inline in the center panel via Plotly.js (Plotly figures already generated via `beyondml/interactive_charts.py`).
- Skills every session should consult: `impeccable`, `react-frontend-development`, `writing-plans`.
- Tech stack: FastAPI 0.115+, WebSocket for streaming log lines, React 18 + Vite + Tailwind CSS 3, `shadcn/ui` for primitive components, Plotly.js for charts, `react-router-dom` for /setup and /pipeline routes.
- Port: web UI runs on 7777 (FastAPI) + 5173 (Vite dev). Production: FastAPI serves built static files.
- No emojis in agent log output (existing agents already comply). UI controls use icon SVGs, not emoji.
- Obsidian palette: `#09090B` bg, `#18181B` surface, `#27272A` border, `#E4E4E7` foreground, `#F97316` accent (orange, matching TUI `$warning`/`$accent`).

## Decisions so far

<!-- empty -->

## Not yet specified

- Whether the WebSocket streams raw Rich markup from agents or the backend strips markup to plain text + metadata before sending. (Likely: backend parses markup tags into structured events `{type, level, content}` so the frontend can style them).
- Whether chart rendering is Plotly JSON sent over WebSocket or pre-generated HTML iframes embedded inline.
- Mobile breakpoint treatment (pipeline screen is inherently dense; tablet min-width is likely the floor).
- Whether file upload uses multipart (drag-and-drop in browser) or the user still types a server-side path as in the TUI WelcomeScreen.

## Out of scope

- User authentication / multi-user sessions.
- Cloud deployment / Docker networking beyond local dev.
- Replacing the TUI `tui_app.py` (it stays alongside the web UI).
- The constrained-decoding integration (MAP-001 — separate effort).
- Jupyter notebook export UI (the export action fires the existing CodeGenAgent; no new export screen needed).
