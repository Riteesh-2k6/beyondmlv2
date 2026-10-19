# [TICKET-L03] Sanity + Imputation + Leakage run serially — 3 independent LLM calls
parent: MAP-002
labels: wayfinder:task
status: open

## Finding

Measured wall-clock times (all independent, no data dependency between them):
  - sanity_agent:      0.061s  (Groq — fast cloud)
  - imputation_agent:  0.073s  (Groq — fast cloud)
  - leakage_agent:     0.090s  (Groq — fast cloud)

On Ollama (local), each LLM call takes 3-15s. Three serial independent calls = 9-45s
of waiting for context that could all be fired together.

The runner.py currently runs all three sequentially even though:
  - SanityAgent needs: df_summary, col_stats → no dependency on Imputation or Leakage
  - ImputationAgent needs: df_summary, missing_report → no dependency on Sanity or Leakage
  - LeakageAgent needs: target, col_descs, correlations → no dependency on Sanity or Imputation

## Fix

Replace the three sequential awaits with a single `asyncio.gather()`:
```python
sanity_result, imp_result, leakage_result = await asyncio.gather(
    sanity_agent.run(df_summary, col_stats, self.log),
    imp_agent.run(df_summary, missing_report, self.log),
    leakage_agent.run(target, col_descs, corrs, self.log),
)
```
This works because all three use `self.llm.chat()` which is synchronous but runs
inside an event loop — wrapping with `asyncio.to_thread` needed for the LLM call
within each agent to truly parallelize.

Expected speedup: 3× reduction on these 3 agents (on Ollama: saves ~20-30s).

## Verified 2026-10-19
- asyncio.gather exists in beyondml/web/runner.py (sanity, imputation, leakage), but agents call synchronous self.llm.chat inside async run, so calls are not actually concurrent; no asyncio.to_thread/achat (depends on L05).
