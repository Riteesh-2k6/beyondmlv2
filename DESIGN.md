---
name: BeyondML
description: An operator's console for watching agents work. Obsidian ground, one orange, Geist for labels and Geist Mono for everything that is data.
colors:
  obsidian: "#09090B"
  surface: "#18181B"
  hairline: "#27272A"
  rule-strong: "#3F3F46"
  fg: "#E4E4E7"
  muted: "#85858F"
  orange: "#F97316"
  success: "#22C55E"
  danger: "#EF4444"
  mark-pending: "#71717A"
  mark-dim: "#52525B"
  chart-ink: "#A1A1AA"
  chart-pale: "#D4D4D8"
typography:
  headline:
    fontFamily: "Geist, ui-sans-serif, system-ui, sans-serif"
    fontSize: "16px"
    fontWeight: 500
    lineHeight: "24px"
  title:
    fontFamily: "Geist, ui-sans-serif, system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 500
    lineHeight: "20px"
  body:
    fontFamily: "Geist, ui-sans-serif, system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: "20px"
  label:
    fontFamily: "Geist, ui-sans-serif, system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 400
    lineHeight: "16px"
  mono:
    fontFamily: "Geist Mono, ui-monospace, SFMono-Regular, monospace"
    fontSize: "12px"
    fontWeight: 400
    lineHeight: "16px"
    fontFeature: "tabular-nums"
  mono-micro:
    fontFamily: "Geist Mono, ui-monospace, SFMono-Regular, monospace"
    fontSize: "11px"
    fontWeight: 400
    lineHeight: "16px"
    fontFeature: "tabular-nums"
rounded:
  none: "0"
  sm: "2px"
spacing:
  hair: "2px"
  xs: "4px"
  sm: "6px"
  md: "8px"
  ml: "10px"
  lg: "12px"
  xl: "16px"
components:
  button-primary:
    backgroundColor: "{colors.orange}"
    textColor: "{colors.obsidian}"
    typography: "{typography.label}"
    rounded: "{rounded.sm}"
    padding: "0 10px"
    height: "28px"
  button-primary-hover:
    backgroundColor: "rgba(249,115,22,0.9)"
  button-ghost:
    backgroundColor: "transparent"
    textColor: "{colors.fg}"
    typography: "{typography.label}"
    rounded: "{rounded.sm}"
    padding: "0 10px"
    height: "28px"
  button-ghost-hover:
    backgroundColor: "{colors.surface}"
  button-danger:
    backgroundColor: "transparent"
    textColor: "{colors.danger}"
    typography: "{typography.label}"
    rounded: "{rounded.sm}"
    padding: "0 10px"
    height: "28px"
  button-danger-hover:
    backgroundColor: "rgba(239,68,68,0.1)"
  input:
    backgroundColor: "{colors.obsidian}"
    textColor: "{colors.fg}"
    typography: "{typography.label}"
    rounded: "{rounded.sm}"
    padding: "0 8px"
    height: "28px"
  chip:
    backgroundColor: "transparent"
    textColor: "{colors.muted}"
    typography: "{typography.mono-micro}"
    rounded: "{rounded.sm}"
    padding: "1px 6px"
  chip-accent:
    textColor: "{colors.orange}"
  chip-danger:
    textColor: "{colors.danger}"
  key:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.muted}"
    typography: "{typography.mono-micro}"
    rounded: "{rounded.sm}"
    padding: "0 4px"
  nav-tab:
    backgroundColor: "transparent"
    textColor: "{colors.muted}"
    typography: "{typography.label}"
    rounded: "{rounded.sm}"
    padding: "0 8px"
    height: "28px"
  nav-tab-active:
    backgroundColor: "{colors.obsidian}"
    textColor: "{colors.orange}"
  top-bar:
    backgroundColor: "{colors.surface}"
    padding: "0 12px"
    height: "40px"
  panel-title:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.fg}"
    typography: "{typography.label}"
    padding: "0 12px"
    height: "32px"
  step-row:
    backgroundColor: "transparent"
    padding: "8px 12px"
  step-row-active:
    backgroundColor: "{colors.surface}"
  graph-node:
    backgroundColor: "{colors.surface}"
    typography: "{typography.mono}"
    rounded: "{rounded.sm}"
    width: "150px"
    height: "34px"
  chat-dock:
    backgroundColor: "{colors.obsidian}"
    height: "208px"
  modal:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.none}"
    width: "672px"
---

# Design System: BeyondML

The web console in `beyondml/web/ui` is the surface recorded here. The same Obsidian values drive the terminal client through `beyondml/tui/theme.py`; the CSS custom properties in `src/index.css` and the `theme.extend` block in `tailwind.config.js` are the web's token source.

## Overview

**Creative North Star: "The Operator's Console"**

BeyondML's console is a dark instrument panel for watching autonomous agents do work, not a dashboard for summarising it. The plan graph is the page. Every agent opens to the literal instruction it was given, then to its live stream of tool calls, then to what it wrote. The visual system exists to make that legible at a glance in a dark room: an obsidian ground, panels one tone lighter, 1px hairlines between regions, and a single orange that means "this is happening now" or "this is what you have selected". PRODUCT.md's one durable constraint on the visual layer holds throughout: every visible element carries data or enables action.

Density comes from type, not chrome. Nearly everything is set at 12px; identifiers, numbers, timestamps, paths, events and code are in Geist Mono with tabular figures, so a row of steps or a ledger table reads like a well-formatted terminal. Status is never a coloured pill; it is a small drawn mark (ring, half-disc, disc, cross, dotted ring) in a fixed six-colour vocabulary that the terminal client shares glyph-for-glyph, and the mark alone carries the colour; the word beside it stays muted. Green and red exist only inside those readings; every cue that asks the operator to act (a waiting prompt, a retry, a question or approval marker, a budget line) is orange. The restraint is the point: when something turns orange, the operator looks.

The system is flat. Depth is tonal (obsidian under surface), separation is a hairline, and the only shadow in the product sits under the approval modal, the one moment the console asks the operator to decide. Motion is likewise rationed to one authored gesture, the running mark breathing, plus skeleton shimmer and a 180ms modal entrance; everything else is a 150ms colour transition that `prefers-reduced-motion` switches off.

**Key Characteristics:**
- Obsidian ground (#09090B), surface panels (#18181B), hairline rules (#27272A); no gradients, no shadows outside the modal
- One accent (#F97316) carrying running, selected, focus, caret, text selection, the primary button and the lead agent's voice
- Green and red live only in drawn status marks and verdict readings (tool results, exit codes, errors); every cue that needs the operator is orange; no yellow, cyan or violet
- Geist at 12–14px for labels and prose; Geist Mono for every id, number, path, event and code block, tabular figures on
- A flat type ramp (16 / 14 / 12 / 11) with weight 500 as the only emphasis
- 2px corners on every control, 1px borders everywhere, 28px controls, 32px panel titles, 40px top bar
- Regions split by hairlines into fixed-height chrome and a scrolling body; the lead-agent dock is pinned under every project screen

## Colors

Zinc neutrals from near-black to pale grey, one orange for attention and intervention, and green and red that appear only as status readings. Charts draw from the same orange and neutrals.

### Primary
- **Signal Orange** (`{colors.orange}`): the one accent, meaning "now" or "needs you". It marks the running status mark and half-disc, the running graph node's stroke and live edges, the active navigation tab, the selected file, the primary button's fill, the focus ring (`1.5px solid`, 1px offset), the text caret, text selection (orange ground, obsidian text), the lead agent's name in the chat transcript, the "new run / resume" link, tool-call names in the live log, the "N new" scroll-back pill, and every intervention cue: the "N waiting" badge in the top bar, the dock's "prompts waiting for you" link, the `?` and `!` markers on question and approval events, budget lines, retrying text, and the recommended option's border in the modal. It is never a panel tint.

### Neutral
- **Obsidian** (`{colors.obsidian}`): the page ground, the chat dock, every input and `pre` block, the sticky table header, and the active tab inside a surface header.
- **Surface** (`{colors.surface}`): the top bar, panel titles, agent header, graph nodes, kbd chips, the modal, and the hover/selected state of rows on the obsidian ground.
- **Hairline** (`{colors.hairline}`): every 1px rule — region borders, panel-title underlines, row dividers (often at 50–60% opacity inside tables), chip and ghost-button borders, input borders, chart grid lines.
- **Rule Strong** (`{colors.rule-strong}`): the second line weight, used only where a hairline would vanish against obsidian: dependency-graph edges at rest, chart axis and tick lines, the scrollbar thumb, hover-label borders.
- **Foreground** (`{colors.fg}`): primary text, step ids, graph labels, chart titles, the fourth chart series. Prose body usually sits at 90% opacity and descriptive goal lines at 80%.
- **Muted** (`{colors.muted}`): secondary text — skill names, form labels, timestamps, table headers, placeholders, metadata. Holds at least 4.5:1 on both obsidian and surface; it is the floor for readable text.
- **Mark Pending** (`{colors.mark-pending}`): the pending status ring, the third chart series and the "mean" chart line.
- **Mark Dim** (`{colors.mark-dim}`): the skipped (dotted ring) and n/a (dot) marks, and the sixth chart series.
- **Chart Ink** (`{colors.chart-ink}`): Plotly axis text and the second chart series; at 35% (`rgba(161,161,170,0.35)`, `NEUTRAL_FILL`) it is the fill of every bar.
- **Chart Pale** (`{colors.chart-pale}`): the fifth chart series only.

### Status
- **Success** (`{colors.success}`): the done disc, "ok" tool results, approved answers, a zero exit code.
- **Danger** (`{colors.danger}`): the failed cross, failed tool results, error lines, non-zero exit codes, denied answers, the last-failure line, the Stop and Deny buttons.

### Named Rules
**The One Orange Rule.** Orange is the only accent and it means "now", "yours" or "needs you": running, selected, focused, primary, and every prompt, retry or budget cue that waits on the operator. A second accent, an orange panel tint, or orange decoration is out of the system.

**The Status-Only Rule.** Green and red appear only as a status reading: a drawn mark, a tool-result verdict, an exit code, an answer, an error line, or the danger button that would stop or deny. The status word itself stays muted; the mark carries the colour. There is no yellow, cyan or violet in the system.

**The Series Rule.** Charts take colours from the fixed series in order — orange, then zinc neutrals (#A1A1AA, #71717A, #E4E4E7, #D4D4D8, #52525B) — on a transparent paper with a hairline y grid, no x grid, and rule-strong axes. Bars fill with chart ink at 35% under a 1px chart-ink line.

**The Half-Tone Border Rule.** A coloured border on a chip or secondary button is the colour at 40–60% opacity (`border-orange/40` on a chip, `border-orange/60` on the waiting badge and the recommended option, `border-danger/50–60`). Full-strength coloured borders are reserved for the modal frame, the primary button, the running or selected graph node, and the scroll-back pill.

## Typography

**Body Font:** Geist (with ui-sans-serif, system-ui, sans-serif), loaded from `@fontsource/geist`
**Label/Mono Font:** Geist Mono (with ui-monospace, SFMono-Regular, monospace), loaded from `@fontsource/geist-mono`

**Character:** A terminal translated to the web. Geist carries the few words of UI and prose; Geist Mono carries everything that is data. Weight 500 is the only emphasis; there is no display type, no uppercase tracking, no italics. `tabular-nums` is on for every mono, code, kbd and table context so columns of numbers align.

### Hierarchy
- **Headline** (500, 16px, 24px): the project index heading only; the ceiling of the ramp.
- **Title** (500, 14px, 20px): panel and modal titles, the agent header's id (mono), step ids in rows (mono).
- **Body** (400, 14px, 20px): chat transcript, modal body, step goals in the plan approval, the goal in the top bar.
- **Label** (400, 12px, 16px): the working size. Navigation tabs, buttons, form labels, panel titles (at 500), step skill and goal lines, empty-state copy, log messages.
- **Mono** (400, 12px, 16px; 20px in `pre` blocks): ids, model names, paths, events, tool names, the instruction, file previews, table cells, parameter tables.
- **Mono Micro** (400, 11px, 16px): chips, kbd keys, timestamps, model names in step rows, file sizes, the top-bar run state.

### Named Rules
**The Mono-for-Data Rule.** Anything an operator might copy, compare or count — an id, a number, a timestamp, a path, a model name, an event, a tool call, code — is set in Geist Mono with tabular figures. Geist is for labels and sentences.

**The Twelve-Pixel Rule.** The console's working size is 12px. Sizes go to 14px for the thing the panel is about and 16px exactly once. Hierarchy comes from mono versus sans, 500 versus 400, and foreground versus muted, not from scale.

## Layout

The shell is a full-height column: a 40px top bar (surface, bottom hairline, 12px side padding) holding project, goal, the four screen tabs, mode and model, run state and Stop; a `main` that fills the rest and clips; and the lead-agent dock pinned at the bottom (obsidian, top hairline, 36px collapsed, 208px expanded). Every screen inside `main` is built the same way: a 32px panel title (surface, bottom hairline, 12px padding, 12px/500 heading left, mono metadata right) above a body that scrolls independently.

Regions divide by hairlines, never by gaps or cards. The plan board is a row at `lg` (1024px) and above: step rows fill the left, a 38%-wide dependency graph sits right behind a left hairline; below `lg` the graph stacks under the steps at a 288px cap. The agent and files screens put a 208px / 256px sidebar left of the content at `md` (768px); below `md` the sidebar becomes a top strip or a select. The top bar hides the wordmark and key hints below `sm` (640px), the goal below `md`, and mode·model below `lg`; when the screen tabs overflow they scroll horizontally with the scrollbar hidden (`.no-scrollbar`). These are Tailwind's default breakpoints; nothing else is defined.

Content has one container width: 672px (`max-w-2xl`) for the project index, the run form and the modal; empty states cap at `max-w-prose`. Rows and cells use 12px horizontal padding and 6–8px vertical; chips and kbd use 4–6px; controls are 28px tall with 8–10px side padding. Lists are flush: rows carry their own bottom hairline and padding, with no gutters between them.

**The Hairline Rule.** Regions, rows and panel titles separate by a 1px `{colors.hairline}` rule. No gaps, no cards, no shadows between them.

**The Fixed-Chrome Rule.** Chrome has fixed heights — 40px top bar, 32px panel title, 36px dock header, 28px for every control — and content scrolls beneath it inside `min-h-0 flex-1 overflow-auto`.

## Elevation & Depth

Flat, with tonal layering. Obsidian is the ground; surface sits one tone above it for chrome and for anything hovered or selected on the ground. The relationship inverts inside chrome: an active tab in a surface header drops to obsidian, and inputs and `pre` blocks sink to obsidian inside surface. There are no gradients and no hairline glows.

### Shadow Vocabulary
- **Modal** (`box-shadow: 0 12px 40px -8px rgba(0,0,0,0.8)`): the approval modal only, over a 70% obsidian scrim (`bg-obsidian/70`), with a full-strength orange frame.

### Named Rules
**The One Shadow Rule.** The only shadow in the product is the approval modal's. Every other surface is flat; depth is obsidian under surface and a hairline between.

**The Inverse-Layer Rule.** The selected or active thing takes the other tone from its container: a row on the obsidian ground rises to surface; a tab or field inside surface chrome drops to obsidian.

## Shapes

Square with the corners knocked off. Every control, chip, kbd, tab, graph node and focus ring has a 2px radius (`rounded-[2px]`, SVG `rx=2`); the modal has none. Borders are 1px everywhere, with 1.5px reserved for the running or selected graph node and for strokes inside status marks. The scrollbar thumb (4px radius, 2px obsidian inset) is the one softer corner. Status marks are drawn circles at 1.5px stroke: ring for pending, half-disc for running, filled disc for done, cross in a ring for failed, dotted ring for skipped, a dot for n/a. The only other icons are inline-SVG disclosure chevrons at 1.5px stroke: 10px in the dock header and file tree, 8px on tool rows in the live log, rotating 90° when open.

**The 2px Rule.** Corners are 2px or square. No pills, no 4px+ radii on controls, no circles except the drawn status marks.

## Components

### Buttons
- **Shape:** 2px corners, 1px border, 28px tall, 10px side padding, 6px gap, 12px Geist, 150ms colour transition.
- **Primary:** orange fill and border, obsidian text (`{components.button-primary}`); hover dims the fill to 90%. One per view: Send, Approve, Plan and run, Open project.
- **Ghost (default):** transparent, hairline border, foreground text; hover lifts to surface with a muted border.
- **Danger:** transparent, danger border at 60%, danger text; hover tints 10% danger. Stop, Deny.
- **Disabled:** 40% opacity, `not-allowed` cursor.
- **Focus:** the global ring, 1.5px orange with 1px offset and 2px radius.
- **Shortcut:** a button that has a key shows it as a kbd chip after the label (`Approve y`, `Later esc`).

### Chips
- **Style:** transparent, 1px border, 2px corners, 1px/6px padding, 11px Geist Mono, 16px line.
- **Tones:** muted (hairline border, muted text) for dependencies, inputs, outputs, metrics and `attempt N`; accent (orange at 40%); danger (red at 50%). A `k=v` metric chip puts the value in foreground.
- **Key (kbd):** surface fill, hairline border, 4px side padding, 11px mono muted; sits beside the label it triggers.

### Panels and rows
- **Panel title:** 32px, surface, bottom hairline, 12px padding; 12px/500 heading left, 11px muted metadata and a text link right.
- **Step row:** full-width button, bottom hairline, 8px/12px padding; status mark (12px) · mono id (14px/500) · muted skill · chips · right-aligned mono metric and elapsed (orange while running). Second line indented 20px: goal at muted (foreground on hover), model in 11px mono. Third line: `← dep` chips and the outcome (muted summary, red failure, orange retry). Hover and active lift to surface.
- **List rows** (agents, files, projects, ledger): 12px mono, 6–10px vertical padding, hover and selected to surface; the selected file goes orange; `aria-current` marks the selection.
- **Empty state:** dashed hairline box, 16px padding, 16px margin, `max-w-prose`; 500 title and muted body that says what will appear here and why.
- **Skeleton:** surface-to-#1F1F23 shimmer (1.4s), 2px radius, sized to the rows it stands in for.

### Inputs / Fields
- **Style:** obsidian fill, hairline border, 2px corners, 28px tall, 8px padding, 12px text, muted placeholder (`{components.input}`); textareas keep the same border and drop the fixed height.
- **Focus:** border turns orange, outline removed; caret is orange.
- **Select:** same box with a 4px drawn chevron in muted at the right; the native marker is hidden.
- **Labels:** 12px muted, 4px above the field, lower-case.

### Navigation
- **Top-bar tabs:** 28px, 8px padding, 2px corners, 12px Geist, muted; hover to foreground; active drops to obsidian with orange text. Each shows its number key as a kbd chip. The agent screen's Instruction / Live / Outputs / Evolve tabs use the same treatment with event counts after a middle dot.
- **Run state:** a 10px status mark and a mono word (`running` in orange, `idle` or `reconnecting` in muted).
- **Waiting badge:** when prompts are pending, a 28px button in orange text with an orange border at 60% and a 10% orange hover, counting them with the `a` key beside.
- **Sidebar lists:** 208px (agents) and 256px (files) at `md`; a mono select replaces them below.

### Status marks (signature)
A 12px SVG (10px in dense chrome, 9px in the graph) at 1.5px stroke, one per status, in `STATUS_COLOR`: pending ring (#71717A), running half-disc (orange, breathing 1.6s), done disc (green), failed cross-in-ring (red), skipped dotted ring (#52525B), n/a dot (#52525B). `StatusWord` sets the mark beside the status word in 12px muted mono; the mark alone carries the colour. The terminal client renders the same six as glyphs.

### Dependency graph (signature)
Steps laid in topological columns: 150×34px surface nodes with a 2px radius, hairline stroke (orange at 60% while running, full orange and 1.5px when selected), a 4.5px status mark at x=14 and the id in 11.5px mono. Edges are cubic curves in rule-strong at 1px, orange at 1.5px when they feed a running step, 50% when the target is skipped. Columns are 52px apart, rows 12px.

### Live log (signature)
One line per event, 2px vertical padding, a 64px mono timestamp column in 11px muted. Messages in 90% foreground; tool calls and results are collapsible `details` in 12px mono behind an 8px muted chevron that turns 90° when open, the call name in orange and the result in green or red with its verdict in muted; expanded JSON sits in an obsidian `pre` with a hairline border. Questions and approvals open with an orange `?` or `!` and close with their answer in green or red; budget lines are orange; errors red. Generations, trials, artifacts and charts are single muted mono lines. When the operator scrolls up, a surface pill with an orange border counts unseen events at bottom right.

### Approval modal (signature)
A 70% obsidian scrim; a 672px surface dialog with a full orange frame, the one shadow, square corners, and a 180ms rise-and-fade entrance. Header: agent in orange mono · title in 14px/500 · "N more after this" in muted. Body scrolls at 50vh: a plan list, a tool name over an obsidian `pre` of its args, a budget sentence, or a question with an input. A question with choices lists them as full-width 2px-radius rows with a hairline border that lifts to a muted border on obsidian on hover; the recommended one is framed in orange at 60% with a 10% orange hover and a `recommended` tag in 11px orange mono, and each carries its number key as a kbd chip. Footer: Approve `y` (primary), Deny `n` (danger), Later `esc` (ghost, right).

### Charts
Plotly through one `Plot` wrapper: transparent paper and plot, Geist Mono 11px in chart ink, a hairline y grid and zero lines with the x grid off, rule-strong axis and tick lines, horizontal legend above the plot at 10px, surface hover labels with rule-strong borders, no modebar; an x-axis title gets an 8px standoff and a 48px bottom margin. Titles are 12px foreground, left-aligned. Series follow The Series Rule: the headline trace is orange, comparisons are zinc, bars fill with `NEUTRAL_FILL`; a figure JSON from `reports/charts/` keeps the colours it sets and takes the series in order for the rest, and integer axes get whole-number ticks. A skeleton of the same height stands in while Plotly or the figure loads.

## Do's and Don'ts

### Do:
- **Do** draw status with the six-mark vocabulary at 1.5px stroke and `STATUS_COLOR`; the mark carries the colour and the word beside it stays muted mono.
- **Do** set every id, number, timestamp, path, model, event and code block in Geist Mono with `tabular-nums`.
- **Do** separate regions and rows with a 1px `#27272A` hairline, and use `#3F3F46` only for graph edges, chart axes and scrollbars.
- **Do** keep readable text at `#85858F` or lighter; drop prose to 90% or 80% foreground, never to a darker grey.
- **Do** use the 28px control, 32px panel title and 40px top bar heights, with 12px side padding and 2px corners.
- **Do** show a keyboard shortcut as a kbd chip beside the control it fires.
- **Do** write empty states as a dashed hairline box that says what will appear and when.
- **Do** render charts through `Plot` so the palette, fonts, grid and hidden modebar carry.
- **Do** keep the lead-agent dock pinned beneath every project screen.

### Don't:
- **Don't** introduce a second accent or tint a panel orange; orange is text, a 1px border, a mark, the caret, the selection, or the one filled button.
- **Don't** cast a shadow anywhere but the approval modal.
- **Don't** use a radius other than 2px (or square for the modal); no pills.
- **Don't** use green or red for anything but a status reading, and don't bring in yellow, cyan or violet; a cue that needs the operator is orange.
- **Don't** build metric tiles or KPI cards; metrics are mono table rows or inline `label value` pairs.
- **Don't** add display sizes, uppercase tracking, or weights other than 400 and 500.
- **Don't** add animation beyond the breathing running mark, the skeleton shimmer, the 180ms modal entrance and 150ms colour transitions; respect `prefers-reduced-motion`.
- **Don't** use icon fonts or decorative icons; the only icons are the status marks and 8–10px inline-SVG disclosure chevrons.
