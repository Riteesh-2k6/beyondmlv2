# [TICKET-W04] SetupPage: maps ConfigScreen + WelcomeScreen to web
parent: MAP-002
labels: wayfinder:task
status: open
blocked-by: TICKET-W02, TICKET-W03
blocks: TICKET-W08

## Question

How do ConfigScreen and WelcomeScreen map to a single web SetupPage?

TUI flow: ConfigScreen modal (LLM provider, API key, model) → WelcomeScreen (file path, description, path choice, GA params, ensemble toggle, test size) → PipelineScreen.

Web mapping:
1. Single `/` route with two logical steps: Step 1 = LLM config (inline form, not a modal, since there is no prior page to overlay), Step 2 = dataset config. A horizontal step indicator at top.
2. File input: `<input type="file">` drag-drop zone for CSV/XLSX/JSON. Displays filename + preview row count after selection. Also accepts a server-side path field for power users.
3. Path choice: 6 cards (Autonomous / Explore / Supervised / Unsupervised / Dimensionality Reduction / Deep Learning) replacing the RadioSet. Selected card gets orange border.
4. LLM selection: Groq / Ollama radio with conditional API key + model name inputs.
5. GA params: pop size, generations, test split — compact inline numeric inputs.
6. Start button: disabled until file is loaded and LLM is configured. Shows a progress spinner while the backend uploads and begins the pipeline WebSocket connection.
7. ASCII banner: BEYOND ML banner rendered as a `<pre>` element in Geist Mono with orange text gradient — matches the TUI BANNER string character-for-character.

Visual: full-width single-column form, max-width 680px centered, bg `#09090B`, orange accent on focus/selected. No card grid scaffolding (craft-floor ban).
