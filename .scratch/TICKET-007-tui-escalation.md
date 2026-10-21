# [TICKET-007] TUI escalation modal — surface low-confidence decisions to user
parent: MAP-001
labels: wayfinder:task
status: superseded
blocked-by: TICKET-002
blocks: TICKET-008

## Question

How is a `LowConfidenceDecision` exception surfaced in the TUI without blocking the async pipeline?

**Current TUI pattern** (from `OutlierAgent`): `get_user_input(prompt: str) -> str` is an async callback injected at agent construction time. This already handles human-in-the-loop for outlier strategy.

**Target behavior**: Re-use the same `get_user_input` async callback pattern across all agents. When a `LowConfidenceDecision` is caught, format a prompt showing:
- The question asked
- The model's tentative answer and confidence score  
- The candidate list (for Choice) or Yes/No (for Noul)
- The user's option to accept the tentative answer or override

Decisions:
1. Confirm that `get_user_input` is the right vehicle (vs. a new dedicated event queue). It is — keeps the pattern consistent, no new TUI architecture needed.
2. What does the prompt string look like? Define a `format_escalation_prompt(question, tentative_answer, confidence, candidates) -> str` utility function.
3. Where does `format_escalation_prompt` live? Propose `beyondml/llm/constrained.py` alongside the decision engine.
4. Agents that don't yet receive `get_user_input` (LeakageAgent, SanityAgent, FeatureAgent, ReflectionAgent) must add it as an optional kwarg with a default that auto-accepts the tentative answer (safe for non-interactive / benchmark runs).
5. Test cases:
   - `format_escalation_prompt` produces valid string with all fields
   - default `get_user_input=None` auto-accepts tentative answer
   - explicit override input routes to correct answer string

Superseded 2026-10-21 by docs/superpowers/specs/2026-10-19-beyondml-ml-harness-design.md (core harness, sub-project 1).
