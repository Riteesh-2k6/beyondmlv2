# [TICKET-L05] llm.chat() is synchronous — blocks the event loop
parent: MAP-002
labels: wayfinder:task
status: open

## Finding

All agents call `self.llm.chat(...)` which is a synchronous `requests.post()`.
On the async runner, this blocks the entire event loop thread for the duration of
the HTTP call. `asyncio.gather()` on two agents that both call `self.llm.chat()`
will NOT actually run them concurrently — they will execute sequentially because
`chat()` is not awaitable.

This is the root blocker that prevents TICKET-L03 and TICKET-L04 from actually
parallelising unless fixed first.

## Fix

Add `async def achat(...)` to `LLMProvider` base and both providers:
```python
async def achat(self, messages, json_mode=False, temperature=0.3, timeout=None) -> str:
    return await asyncio.to_thread(self.chat, messages, json_mode, temperature, timeout)
```
Then replace every `self.llm.chat(...)` in every agent with `await self.llm.achat(...)`.

This is a 1-line addition to `base.py`, 1-line addition to each provider, and a
mechanical find-replace in each agent. The existing `chat()` stays for backward
compatibility (TUI and tests still call it synchronously).

Expected impact: unlocks ALL gather()-based parallelism in L03 and L04.
