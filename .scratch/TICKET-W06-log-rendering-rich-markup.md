# [TICKET-W06] Rich markup → HTML rendering: log line parser
parent: MAP-002
labels: wayfinder:task
status: open
blocked-by: TICKET-W02, TICKET-W03
blocks: TICKET-W05, TICKET-W08

## Question

How does the backend convert Rich markup tags from agent log calls into structured frontend events?

Current agent log calls use Rich markup like:
- `[bold magenta]● Orchestrator[/bold magenta]  Analyzing dataset...`
- `[bold green]✓ Model performance is satisfactory![/bold green]`
- `[dim]─── Feature Engineer ────────[/dim]`
- `[red]⚠ Leakage Risk[/red]`

Two options:
**Option A (recommended):** Strip markup server-side into structured events. Backend has a `parse_rich_markup(text: str) -> List[Span]` function that tokenises Rich tags into `[{text, style}]`. Spans assemble into a `LogLine` object serialised to `{type:"log", payload:{spans:[{text, cls}]}}`. Frontend renders `<span class={cls}>` per span. CSS classes: `agent-header`, `success`, `muted`, `error-text`, `warn`, `info`, `cyan`, `code`.

**Option B:** Let frontend parse raw markup strings. Simpler backend, more frontend logic.

Decision needed: Option A is architecturally cleaner (backend owns formatting knowledge), but adds ~50 lines of parsing code. Confirm option and document the complete CSS class map so frontend and backend are aligned.

Test cases:
- `parse_rich_markup("[bold magenta]● Orch[/bold magenta]  msg")` → `[{text:"● Orch", cls:"agent-header"}, {text:"  msg", cls:""}]`
- Nested tags: `[bold][red]text[/red][/bold]` → combined class
- Unknown tag: pass through as plain text, no crash
