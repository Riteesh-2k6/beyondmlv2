# BeyondML — Product Truth

## What it is
BeyondML is a local-first, fully autonomous AutoML platform. It orchestrates a multi-agent pipeline (Orchestrator → EDA → Outlier → Feature Engineer → GA Trainer → Evaluator → Reflection) that runs entirely on the user's machine via Ollama or Groq, with no mandatory cloud dependency.

## Who uses it
ML engineers and data scientists who want a transparent, observable, zero-cost AutoML loop — people who distrust black boxes and want to see exactly what each agent decided and why. Primary user: Riteesh Thiruveedhula (student/indie researcher, Hyderabad). Usage scene: dark room, laptop, local inference.

## What success looks like
User uploads a CSV, optionally picks a path and LLM, and watches each agent run in real time. They can intercept decisions (outlier strategy, feature acceptance, target column). At the end they see a final model score, feature importances, charts, and can export a Jupyter notebook.

## Product constraints
- All inference stays local (Ollama) unless user opts into Groq
- Zero paid cloud for the core pipeline
- CLI entry point `beyondml run` keeps working; web UI is an alternative, not a replacement
- Python package structure under `beyondml/`; FastAPI backend speaks to the same agent code

## Tone
Obsidian dark. Terminal-native aesthetic translated to the web. No decoration for its own sake. Every visible element carries data or enables action. Dense, not cluttered. Precision over warmth.
