# [TICKET-004] Wire constrained decisions into OutlierAgent
parent: MAP-001
labels: wayfinder:task
status: superseded
blocked-by: TICKET-002
blocks: TICKET-008

## Question

How does `OutlierAgent` integrate constrained decisions?

**Current flow**: LLM free call → recommend strategy → `get_user_input` prompt (human picks 1-5).  
**Target behavior**: Replace the LLM strategy recommendation with a Choice decision over `["remove", "cap", "log", "flag", "keep"]`, returning `(strategy, confidence)`. If confidence >= threshold, skip the `get_user_input` prompt and auto-apply. If confidence < threshold, surface the choice to the user as before — but pre-fill the recommendation label from the constrained answer.

Additionally: before applying any strategy, fire a Noul — "Do these outliers likely represent real signal rather than noise?" — to decide whether to default to `keep` without asking.

Decisions:
1. Exact threshold values for auto-apply (propose 0.80 for Choice strategy auto-apply).
2. Whether the Noul "are they real signal?" gate comes before or after the strategy Choice.
3. Test cases:
   - high-confidence Choice → auto-applies without TUI prompt
   - low-confidence Choice → falls through to `get_user_input` with pre-filled recommendation
   - Noul "real signal" high-confidence yes → forces `keep` regardless of strategy Choice

Superseded 2026-10-21 by docs/superpowers/specs/2026-10-19-beyondml-ml-harness-design.md (core harness, sub-project 1).
