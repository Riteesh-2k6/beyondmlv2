# [TICKET-L04] EDA + Outlier run serially — 2 independent agents (6.2% of total)
parent: MAP-002
labels: wayfinder:task
status: open

## Finding

  - eda_agent:     2.538s  (LLM call + 4 chart renders)
  - outlier_agent: 0.078s  (LLM call — Groq fast)

On Ollama, EDA takes 5-20s (large prompt) and OutlierAgent takes 3-8s.
They are fully independent given the same input `df` and `profile`.

The runner.py already has `asyncio.gather()` placeholder for these two in comments
but the actual implementation is sequential.

## Fix

```python
eda_result, outlier_result = await asyncio.gather(
    eda_agent.run(df, profile, target_info, description, self.log),
    outlier_agent.run(df, outlier_summary, profile, self.log, self.get_user_input),
)
df = outlier_result["df"]
```

Note: `get_user_input` in outlier_agent must not conflict with concurrent log writes.
The `self.log` function is already safe (appends to a queue); `get_user_input` is
already serialized via an asyncio.Queue in PipelineRunner.

Expected speedup: 2× on EDA+Outlier block (on Ollama: saves 3-18s).
