# [TICKET-003] Wire constrained decisions into LeakageAgent and SanityAgent
parent: MAP-001
labels: wayfinder:task
status: superseded
blocked-by: TICKET-002
blocks: TICKET-008

## Question

How do `LeakageAgent` and `SanityAgent` integrate constrained decisions?

**LeakageAgent** currently: free LLM call → JSON parse → `leakage_findings` list.  
**Target behavior**: For each candidate column, fire a Noul decision — "Is `{col}` leaking information about `{target}`?" — with confidence gating. High-risk findings (confidence > threshold) auto-flag; low-confidence ones escalate to TUI.

**SanityAgent** currently: free LLM call → JSON parse → `issues` list.  
**Target behavior**: For each column with suspicious values, fire a Noul — "Is value `{v}` in column `{col}` a logical impossibility for this domain?" — with confidence gating.

Decisions:
1. Keep the existing free LLM call for the initial discovery pass (generates candidates). Replace only the per-column **acceptance** decision with constrained Noul.
2. New signature additions to both agents — must not break existing `run()` interface.
3. TUI escalation: when `LowConfidenceDecision` is raised, how is it surfaced in the pipeline? (Delegate to `get_user_input` callback already present in `OutlierAgent` — replicate that pattern.)
4. What tests cover the new Noul calls? Test that:
   - high-confidence "yes" auto-accepts the finding
   - high-confidence "no" auto-rejects it (finding dropped)
   - low-confidence raises and routes to `get_user_input`

Superseded 2026-10-21 by docs/superpowers/specs/2026-10-19-beyondml-ml-harness-design.md (core harness, sub-project 1).
