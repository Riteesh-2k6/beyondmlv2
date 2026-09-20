# [TICKET-W01] Lock visual direction and DESIGN.md
parent: MAP-002
labels: wayfinder:grilling
status: open
blocks: TICKET-W02, TICKET-W03, TICKET-W04, TICKET-W05, TICKET-W06, TICKET-W07

## Question

What is the exact visual world this Web UI inhabits — translated from the TUI into CSS/HTML terms?

The TUI uses Textual's built-in theming: `$accent` (orange3), `$warning` (amber), `$surface` (#202020-ish), dark `Screen`. The web UI must feel like the same product opened in a browser tab, not a generic dashboard. Specific decisions to make:

1. Typography: confirm Geist Mono for all monospace agent log text; what is the display face for panel titles and metric labels?
2. Color tokens: map all TUI `$` variables to exact hex values in Tailwind config.
3. Panel borders: TUI uses `border: solid $accent` (orange) on center, `border: solid $accent` on left and right. Does the web preserve this or use subtler borders?
4. Agent log markup: TUI uses Rich markup `[bold magenta]`, `[bold green]`, `[dim]` etc. How are these rendered in the browser (CSS classes? inline spans?)?
5. Chart panel: TUI has charts as ANSI art + separate browser tabs. Web has inline Plotly. What is the panel split — does the chart area sit below the log in the center panel, or side-by-side?
6. GA fitness sparkline: TUI uses Textual's Sparkline widget. Web replacement: Recharts / Chart.js inline sparkline.
