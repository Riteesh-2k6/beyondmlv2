# [TICKET-W05] PipelineScreen: 3-panel layout, pipeline tree, stats column
parent: MAP-002
labels: wayfinder:task
status: open
blocked-by: TICKET-W02, TICKET-W03
blocks: TICKET-W08

## Question

How does PipelineScreen's 3-column CSS grid (`1fr 3fr 1fr`) map to the web?

Left panel — Pipeline tree + Agent Reasoning:
- Pipeline tree: collapsible `<ul>` tree. Each node has a status dot (waiting / running / done / error) that animates: `waiting` = dim gray, `running` = orange pulse, `done` = green static, `error` = red static. Clicking a node scrolls the reasoning section to that agent's last reasoning entry.
- Agent Reasoning: scrollable monospace log below the tree, auto-scrolls to latest.

Center panel — Live log + inline charts:
- `RichLog` maps to a `<div class="log-pane">` with auto-scroll. Each line rendered as a `<p>` with class derived from Rich markup tag: `bold-magenta → class="agent-header"`, `bold-green → class="success"`, `dim → class="muted"`, `red → class="error"` etc.
- Charts render inline below/between log lines. When backend emits `{type:"chart"}`, a `<PlotlyChart>` component is inserted at the current scroll position — not in a separate tab.
- GA fitness sparkline: Recharts `<AreaChart>` in center panel below active log area.

Right panel — Stats + Data preview:
- DATASET section: shape, target, task, classes, split, new features — styled as `key: value` monospace rows.
- GA PROGRESS: mini sparkline (Recharts `<SparklineBar>`).
- DATA PREVIEW: `<table>` with first 15 rows / 20 cols, scrollable horizontally. Zebra stripes on `#18181B` / `#09090B`.

Bottom bar — Input:
- Persists across the pipeline. When backend sends `input_request`, the input field becomes active and shows the prompt. Otherwise it shows dim keybind hints.
- Keybinds: s=save, e=export, c=view all charts (opens chart gallery modal).
