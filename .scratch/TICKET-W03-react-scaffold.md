# [TICKET-W03] React/Vite scaffold: routing, layout shell, Tailwind config, design tokens
parent: MAP-002
labels: wayfinder:task
status: open
blocked-by: TICKET-W01
blocks: TICKET-W04, TICKET-W05, TICKET-W06, TICKET-W07

## Question

What is the exact frontend directory structure and how are design tokens wired into Tailwind?

Decisions:
1. `beyondml/web/ui/` — Vite project root. `src/` has `pages/`, `components/`, `hooks/`, `lib/`.
2. Routes: `/` → SetupPage, `/pipeline` → PipelinePage. React Router v6.
3. Tailwind config: custom colors (`obsidian`, `surface`, `border-subtle`, `accent-orange`, `foreground`) mapped to exact hex from TUI palette. Custom font: `Geist Mono` loaded from Fontsource.
4. Global CSS: custom scrollbar styling (thin, `#27272A`), text selection (`#F97316`/white), caret color, focus-ring override. These are the craft-floor browser-surface details.
5. shadcn/ui installed for: `Button`, `Input`, `Dialog`, `RadioGroup`, `Progress`. All components re-skinned to obsidian tokens — no default slate/gray.
6. `useWebSocket` custom hook (wraps native WebSocket, auto-reconnect, message queue).
