# [TICKET-W07] Inline chart rendering: Plotly JSON → React components
parent: MAP-002
labels: wayfinder:task
status: open
blocked-by: TICKET-W02, TICKET-W03
blocks: TICKET-W05, TICKET-W08

## Question

How do the interactive Plotly charts (currently opened as `file://` HTML pages) render inline in the center log panel?

Current source: `beyondml/interactive_charts.py` generates Plotly HTML files, saved to `workspace/charts/`. TUI opens them with `webbrowser.open(file://)`.

Web approach:
1. Backend intercepts `action_view_charts` calls — whenever an agent creates a chart, instead of writing HTML, it calls a hook that extracts the Plotly `fig` JSON and emits a `{type:"chart", payload:{title, plotly_json}}` WebSocket event.
2. `interactive_charts.py` needs a new function `fig_to_json(fig) -> dict` that returns `fig.to_json()` parsed as dict. Backend calls this instead of `fig.write_html()` (or in addition to, for compatibility).
3. Frontend `<PlotlyChart>` component: uses `react-plotly.js` (wraps Plotly.js). Receives `plotly_json`, renders with obsidian dark theme override (`paper_bgcolor="#09090B"`, `plot_bgcolor="#18181B"`, `font.color="#E4E4E7"`).
4. Charts appear inline in the log flow at the insertion point of the WebSocket event. They do not interrupt the log — they appear between log lines at the moment the agent emits them.
5. "View all charts" button opens a full-width chart gallery modal (grid of all charts seen so far, each with a title).

Test cases:
- `fig_to_json(go.Figure())` → valid dict with `data` and `layout` keys
- Frontend: PlotlyChart renders without crashing on a minimal `{data:[], layout:{}}` JSON
- Dark theme is applied: bg colors match obsidian tokens

## Verified 2026-10-19
- No fig_to_json(fig) -> dict in beyondml/interactive_charts.py (conversion is inline in PipelineRunner.chart in beyondml/web/runner.py).
- No test for fig_to_json, no backend test of chart event, no frontend PlotlyChart test (component is inline ChartEntry in components/LogPane.tsx, no standalone PlotlyChart).
- Existing agents still write HTML via interactive_charts; no hook intercepts action_view_charts.
