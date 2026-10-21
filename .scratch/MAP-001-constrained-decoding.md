# [MAP-001] Constrained Decoding for BeyondML — Wayfinder Map
labels: wayfinder:map
status: superseded

## Destination

A fully local, in-house constrained-decoding layer (`beyondml/llm/constrained.py`) built on top of the existing `LLMProvider`/`OllamaProvider`, replacing free-text LLM calls in `LeakageAgent`, `SanityAgent`, `OutlierAgent`, `FeatureAgent`, and `ReflectionAgent` with logprob-gated decisions (Noul/Choice/Score question types). Low-confidence decisions escalate to the TUI rather than silently defaulting. No new cloud dependencies. Pipeline remains fully local/free.

## Notes

- Domain: BeyondML `beyondml/` Python package — agents in `beyondml/agents/`, LLM providers in `beyondml/llm/`, tests in `tests/`.
- Skills every session should consult: `writing-plans`, `test-driven-development`.
- Standing preferences: TDD (tests first), zero emojis in CLI/agent output, obsidian dark theme, PYTHONPATH=. pytest tests/, venv at `venv/`.
- Constrained-decoding pattern: call Ollama's `/api/generate` with `logprobs: true` and a fixed token candidate set (yes/no for Noul, enumerated options for Choice, 1–N for Score). Extract log-probabilities over only those tokens, convert to a calibrated probability, return `(answer, confidence)`. If confidence < threshold, surface a TUI prompt for human override.
- Do NOT adopt Laya/Jev/any external model — build the pattern ourselves on top of OllamaProvider.
- Evaluation report (full text) is the authoritative source for what sites to integrate: TargetIdentifier (Choice), leakage/sanity checks (Noul), outlier acceptance (Noul/Score), feature acceptance (Noul), ReflectionAgent satisfied / needs_improvement (Noul).
- OllamaProvider already has `chat()` (JSON mode) and `chat_stream()`. New `logprob_decision()` method needed.

## Decisions so far

<!-- empty — no tickets closed yet -->

## Not yet specified

- Whether the Ollama `/api/generate` endpoint with `logprobs` is available in the installed Ollama version (needs runtime check; fallback to softmax-over-chat if not).
- Whether TUI escalation modal blocks the async pipeline or uses a background prompt queue.
- Whether `FeatureAgent` proposals need per-feature Noul gating or a batch Choice across candidates.

## Out of scope

- Adopting any third-party constrained-decoding library (Laya, OpenJev, NanoJev, von).
- Integrating TypeSafe's hosted Jev API.
- Replacing the OrchestratorAgent's path routing (already a single-call LLM decision with hardcoded fallback — not performance-critical enough to change).
- DL agent or ensemble agent integration (not listed in the evaluation report as integration sites).

Superseded 2026-10-21 by docs/superpowers/specs/2026-10-19-beyondml-ml-harness-design.md (core harness, sub-project 1).
