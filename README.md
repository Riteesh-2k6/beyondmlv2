# 🚀 BeyondML

**An ML-engineering agent harness: give it data and a goal, and it plans the project, runs one agent per plan step, and ships a repo you can train and predict with.**

BeyondML is built on Google ADK. A planner turns your goal and data into a plan; each step (audit, clean, EDA, features, model search, export, evaluation) gets its own agent, instructed by a skill and limited to typed tools. Hyperparameter search is a genetic algorithm (or Optuna). Every agent's instructions and live activity are visible in the terminal UI and the web UI.

---

## 🎯 Problem

Building machine learning pipelines is traditionally a tedious, manually iterative process requiring deep expertise in EDA, feature engineering, and hyperparameter tuning. While automation tools exist, they are often opaque "black boxes" that don't provide visibility into their intermediate decision-making or data transformations, making them hard to trust and debug.

## 💡 Approach

BeyondML addresses this by combining the reasoning capabilities of LLMs with a highly observable, terminal-native UI. A planner splits the goal into steps and one agent per step (audit, cleaning, EDA, features, model search, evaluation) works through a shared project folder, so every decision and file is on disk. By pairing LLM-driven feature engineering with rigorous, non-LLM Genetic Algorithms for optimization, the platform balances intelligent heuristics with mathematical precision—all completely transparent to the user via the TUI.

## 🔄 Iterations

1. **V1 (Core ML Pipeline):** Built the foundational pipeline architecture supporting standard supervised ML models and a basic TUI layout.
2. **V2 (Genetic Optimization & Orchestration):** Introduced native Genetic Algorithm optimization for hyperparameters and structured the engine layout.
3. **V3 (Autonomous Agents):** Integrated Groq and local Ollama. Added LLM-driven EDA, Outlier, and Feature Engineering agents for autonomous data preparation.
4. **V4 (Unsupervised & Visualizations):** Added support for PCA, DBSCAN, KMeans, and interactive Plotly charts to supplement terminal-native rendering.
5. **V5 (Performance & Workflow):** Parallelized agent execution using `asyncio` to drastically reduce processing bottlenecks and exposed customizable settings like dynamic train/test splits.
6. **V6 (Agent harness):** Rebuilt on Google ADK: a planner, one agent per plan step driven by SKILL.md skills, evolution engine v2, standalone exported code, and a gateway serving the Textual TUI and the web UI. The fixed V1–V5 agent pipeline is retired.

## 📐 Key Design Choices

- **Terminal-Native TUI:** Built using `Textual` and `plotext` rather than a standard web interface to offer developers immediate, lightweight accessibility without managing frontend/backend server states.
- **One agent per plan step:** each step's agent gets its goal, a skill (SKILL.md) and only the tools that skill allows; independent steps run in parallel, and a step passes only when its success check (files, a script run, or a metric) holds.
- **Hybrid Optimization:** Leveraged LLMs strictly for qualitative decisions (feature generation, sanity checks) and evolutionary search (Genetic Algorithms) for exact, quantitative tasks (hyperparameter tuning).

## ⏱️ Daily Time Commitment

The design and implementation of BeyondML evolved over several weeks, involving a consistent daily time commitment of approximately **2 to 4 hours**. Time was roughly split between core ML architecture, asynchronous agent orchestration, LLM prompt engineering, and UI state management.

---

## ⚡ Quick Start

The fastest way to install BeyondML globally (like a standalone CLI app) is using `pipx` or `pip`:

```bash
# Recommended 1-Click Install (isolated environment):
pipx install git+https://github.com/Riteesh-2k6/beyondml.git

# Alternative (standard pip):
pip install git+https://github.com/Riteesh-2k6/beyondml.git

# Create a project and open the TUI on it
beyondml init my-project && cd my-project && beyondml
```

If you want to contribute or modify the code locally instead:
```bash
git clone https://github.com/Riteesh-2k6/beyondml.git
cd beyondml
pip install -e ".[dev]"
```

---

## 🏗️ Architecture

```
beyondml/
├── cli.py              ← beyondml init / run / serve / tui / benchmark
├── harness/            ← ADK glue (models, rate limits), intake, planner, lead, step-agent factory, executor
├── toolkit/            ← typed agent tools: files, run_python, control (ask_user, finish_step), ML tools
├── skills/             ← SKILL.md step skills (tabular-audit … tabular-export, evaluate-explain) and knowledge skills
├── engine/             ← evolve (GA) + tune (Optuna), fitness, pipeline builder, export, checks, assess, clustering, MLP
├── workspace/          ← project folder, plan.json, event bus, ledger
├── gateway/            ← FastAPI + SSE over a run session (`beyondml serve`)
├── tui/                ← Textual app: plan board, agent screens, chat, approvals, experiments, files
└── web/                ← web UI over the gateway
```

### How a run flows

```
data + goal → intake (DATA.md) → planner (plan.json) → one agent per step, in dependency order
           → each step's success check → src/pipeline.py, train.py, predict.py, models/model.joblib
```

The exported `src/` code never imports BeyondML.

---

## 🛠️ Usage

### Docker (Recommended for Reproducibility)

Run the completely isolated and reproduced BeyondML environment via Docker:

```bash
# Build and run the TUI natively in Docker
docker-compose run --rm beyondml beyondml run
```

_Note: The `workspace` and `data` directories are mounted automatically, so your files are saved locally. You only need to set your keys in `.env`._

**Using Local Ollama with Docker:**
The `docker-compose.yml` is pre-configured to point `OLLAMA_HOST` to `http://host.docker.internal:11434`. Ensure Ollama is running on your host machine before starting the Docker container.

### TUI (Interactive)

```bash
beyondml init my-project
cd my-project
beyondml
```

Opens the TUI on the project (and the gateway for the web UI on port 7777).

### Headless

```bash
beyondml run "predict survival" --data data/titanic.csv --project my-project --mode autonomous
```

`--mode` is `ask` (confirm every script run), `auto` (confirm installs and network) or `autonomous` (never ask). `beyondml serve` runs the gateway and web UI without the TUI.

### Benchmark (PMLB)

```bash
# Default datasets (titanic, car)
beyondml benchmark

# Custom datasets
beyondml benchmark -d adult -d iris
```

### Environment Configuration

Create a `.env` file:

Any LiteLLM model string works with `--model` or the project's `[model] default` (default: `ollama_chat/qwen3:8b` on a local Ollama). Hosted providers read their usual key from the environment, for example:

```env
GROQ_API_KEY=your-api-key-here   # for --model groq/openai/gpt-oss-120b
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=beyondml --cov-report=term-missing
```

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` / `numpy` | Data manipulation |
| `scikit-learn` | ML models & preprocessing |
| `google-adk` / `litellm` | Agents and any LLM provider (Ollama, Groq, OpenAI, Gemini, …) |
| `textual` / `fastapi` | Terminal UI and the gateway for the web UI |
| `torch` | MLP model family |
| `pyarrow` | Parquet data |
| `pmlb` | Penn ML Benchmark datasets |
| `click` | CLI framework |

---

## 📄 License

MIT
