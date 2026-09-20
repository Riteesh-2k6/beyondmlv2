# [TICKET-001] Verify Ollama logprob API availability and design fallback
parent: MAP-001
labels: wayfinder:research
status: open
blocks: TICKET-002

## Question

Does the installed Ollama version expose `logprobs` via `/api/generate` or `/api/chat`? What token-level probability data is available? If not, what softmax-over-chat fallback is usable to derive calibrated probabilities from a constrained candidate set?

Specifically:
1. Hit `GET /api/version` — what version is running?
2. POST `/api/generate` with `{"logprobs": true}` on a tiny prompt — what fields come back?
3. If logprobs aren't available, test the fallback: POST `/api/generate` with `{"options": {"temperature": 0.01}}` and parse the first token of the response as the decision (works because greedy decode at low temperature is deterministic, but gives no probability).
4. Research alternative: use `/api/generate` with `num_predict: 1` and multiple requests, one per candidate, measuring the `eval_duration` as a proxy — NOT calibrated, discard this.
5. Determine the actual viable path and record the exact API shape needed for TICKET-002.

## Resolution

<!-- filled on close -->
