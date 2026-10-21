# [TICKET-005] Wire constrained decisions into FeatureAgent
parent: MAP-001
labels: wayfinder:task
status: superseded
blocked-by: TICKET-002
blocks: TICKET-008

## Question

How does `FeatureAgent` integrate constrained decisions?

**Current flow**: LLM proposes 2-4 features in JSON → `eval()` applies each → all accepted if they produce a valid Series.  
**Target behavior**: After LLM proposes a feature and before `eval()` applies it, fire a Noul decision — "Does feature `{name}` (formula: `{expr}`) add genuine predictive signal for this dataset?" — to gate application. High-confidence "yes" → apply. High-confidence "no" → skip and log as rejected. Low-confidence → escalate to `get_user_input`.

Decisions:
1. The Noul is fired per-feature, not batch. (Parallel firing is possible but deferred to a later optimisation — out of scope for this map.)
2. The Noul prompt must include the column names and the expr text so the model can reason about it.
3. Features that fail `eval()` are still rejected regardless of Noul result (safety net unchanged).
4. Test cases:
   - high-confidence Noul "yes" → feature applied to df
   - high-confidence Noul "no" → feature in `features_rejected`, not in `df`
   - low-confidence Noul → routes to `get_user_input` for human call
   - eval() failure still rejects regardless of confidence

Superseded 2026-10-21 by docs/superpowers/specs/2026-10-19-beyondml-ml-harness-design.md (core harness, sub-project 1).
