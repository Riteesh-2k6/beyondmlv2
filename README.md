# 🚀 BeyondML

**AI Agent Orchestration Platform — Terminal-native AutoML with LLM-powered agents and Genetic Algorithm optimization.**

BeyondML is an autonomous machine learning pipeline that uses multiple specialized AI agents (EDA, Feature Engineering, GA Optimization, Reflection) orchestrated through an LLM to profile, preprocess, train, and evaluate models — all from a beautiful terminal UI.

---

## 🎯 Problem

Building machine learning pipelines is traditionally a tedious, manually iterative process requiring deep expertise in EDA, feature engineering, and hyperparameter tuning. While automation tools exist, they are often opaque "black boxes" that don't provide visibility into their intermediate decision-making or data transformations, making them hard to trust and debug.

## 💡 Approach

BeyondML addresses this by combining the reasoning capabilities of LLMs with a highly observable, terminal-native UI. We use a multi-agent orchestration architecture where specialized AI agents (EDA, Feature Engineering, Outlier Handling, and Reflection) communicate to systematically process data. By pairing LLM-driven feature engineering with rigorous, non-LLM Genetic Algorithms for optimization, the platform balances intelligent heuristics with mathematical precision—all completely transparent to the user via the TUI.

## 🔄 Iterations

1. **V1 (Core ML Pipeline):** Built the foundational pipeline architecture supporting standard supervised ML models and a basic TUI layout.
2. **V2 (Genetic Optimization & Orchestration):** Introduced native Genetic Algorithm optimization for hyperparameters and structured the engine layout.
3. **V3 (Autonomous Agents):** Integrated Groq and local Ollama. Added LLM-driven EDA, Outlier, and Feature Engineering agents for autonomous data preparation.
4. **V4 (Unsupervised & Visualizations):** Added support for PCA, DBSCAN, KMeans, and interactive Plotly charts to supplement terminal-native rendering.
5. **V5 (Performance & Workflow):** Parallelized agent execution using `asyncio` to drastically reduce processing bottlenecks and exposed customizable settings like dynamic train/test splits.

## 📐 Key Design Choices

- **Terminal-Native TUI:** Built using `Textual` and `plotext` rather than a standard web interface to offer developers immediate, lightweight accessibility without managing frontend/backend server states.
- **Micro-Agent Architecture:** Separated the LLM capabilities into modular, single-responsibility agents (EDA, Leakage, Sanity, Feature, Reflection). This allows parallel execution, improves context utilization, and makes the pipeline's logic highly debuggable.
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

# Launch the TUI immediately
beyondml run
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
beyondml/tui/                ← Textual TUI (plan board, agent screens, chat, approvals, experiments, files)
beyondml/
├── cli.py                   ← CLI entry point (beyondml run / benchmark)
├── state.py                 ← MLState — shared pipeline state
├── charts.py                ← Plotext chart rendering
├── llm/
│   ├── base.py              ← Abstract LLMProvider
│   ├── groq_provider.py     ← Groq cloud LLM
│   └── ollama_provider.py   ← Ollama local LLM
├── engine/
│   ├── profiler.py          ← DatasetProfiler + TargetIdentifier + ORI
│   ├── supervised.py        ← SupervisedPipeline (baselines + final)
│   ├── unsupervised.py      ← UnsupervisedPipeline (KMeans, DBSCAN, PCA)
│   ├── genetic.py           ← Genome + GeneticModelOptimizer
│   ├── metrics.py           ← Classification & regression metrics
│   └── benchmarker.py       ← PMLB benchmark runner
└── agents/
    ├── orchestrator.py      ← LLM-powered path router
    ├── eda_agent.py         ← Exploratory Data Analysis
    ├── outlier_agent.py     ← Outlier detection & handling
    ├── feature_agent.py     ← LLM feature engineering
    ├── ga_trainer.py        ← GA evolution + TUI progress
    ├── evaluator_agent.py   ← Final model evaluation + narration
    └── reflection_agent.py  ← Iterative improvement loop
```

### Agent Pipeline Flow

```
Dataset → Orchestrator → EDA Agent → Outlier Agent → Feature Agent
       → GA Trainer → Evaluator → Reflection Agent → (loop or finish)
```

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
beyondml run
```

Launches the full interactive pipeline with dataset selection, agent orchestration, and real-time progress.

### Benchmark (PMLB)

```bash
# Default datasets (titanic, car)
beyondml benchmark

# Custom datasets
beyondml benchmark -d adult -d iris
```

### Environment Configuration

Create a `.env` file:

```env
# LLM Provider: "ollama" (default) or "groq"
LLM_PROVIDER=ollama

# Required only if using Groq
GROQ_API_KEY=your-api-key-here
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
| `textual` | Terminal UI framework |
| `plotext` | Terminal-native charts |
| `groq` / `requests` | LLM providers (Groq / Ollama) |
| `pmlb` | Penn ML Benchmark datasets |
| `click` | CLI framework |

---

## 📄 License

MIT
