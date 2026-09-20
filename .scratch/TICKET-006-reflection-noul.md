# [TICKET-006] Wire constrained decisions into ReflectionAgent
parent: MAP-001
labels: wayfinder:task
status: open
blocked-by: TICKET-002
blocks: TICKET-008

## Question

How does `ReflectionAgent` integrate constrained decisions?

**Current flow**: Rule-based checks (overfitting/dominance/underperforming/leaking) short-circuit to `satisfied` or call LLM for modifications.  
**Target behavior**: The final "satisfied vs needs_improvement" classification becomes a Noul decision — "Given test_score={score}, train_score={train}, is this model performance satisfactory?" — rather than the hardcoded thresholds (>0.85 test score, <0.10 overfit gap). The hardcoded rules remain as a fast-path pre-check; the Noul adds a calibrated probability layer on top.

Additionally: each proposed feature drop (from LLM modifications) is confirmed with a Noul — "Should column `{col}` be dropped to reduce overfitting/dominance?" — before being executed.

Decisions:
1. Hardcoded rules still fire first (they are instant). Noul is called only when the rule-based verdict is borderline (e.g., test_score between 0.80–0.90, or overfit gap 0.05–0.15).
2. "Borderline zone" thresholds must be defined concretely before implementation.
3. Feature-drop Noul fires per column in `features_to_drop`; low-confidence escalates to `get_user_input`.
4. Test cases:
   - score 0.95 → rule-based "satisfied", no Noul called
   - score 0.70, gap 0.20 → rule-based "needs_improvement", Noul also fires and confirms
   - score 0.83 (borderline) → rule-based skips, Noul decides
   - low-confidence drop decision → get_user_input escalation
