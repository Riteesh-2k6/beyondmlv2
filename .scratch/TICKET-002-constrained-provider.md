# [TICKET-002] Implement `constrained.py` — the core logprob decision engine
parent: MAP-001
labels: wayfinder:task
status: open
blocked-by: TICKET-001
blocks: TICKET-003, TICKET-004, TICKET-005, TICKET-006, TICKET-007

## Question

What is the exact implementation of `beyondml/llm/constrained.py`?

This module extends `OllamaProvider` with a `logprob_decision()` method that:
- Accepts: `messages: List[Dict]`, `question_type: Literal["noul", "choice", "score"]`, `candidates: List[str]`, `confidence_threshold: float = 0.65`
- Calls Ollama with constrained decoding (logprobs over candidates only — TICKET-001 determines exact API shape)
- Returns: `ConstrainedResult(answer: str, confidence: float, raw_logprobs: dict)`
- Raises `LowConfidenceDecision(answer, confidence, candidates)` when confidence < threshold

Key design decisions locked here:
1. `ConstrainedResult` dataclass shape
2. `LowConfidenceDecision` exception class for TUI escalation
3. `ConstrainedOllamaProvider` subclassing `OllamaProvider` or composing it
4. The logprob→softmax→confidence conversion math (or the chat fallback from TICKET-001)
5. The `candidates` prompt injection strategy (list options inline so model output is constrained)

TDD: write `tests/test_constrained.py` first with MockLLM, covering:
- noul returns `(yes|no, float)`  
- choice returns one of the candidate strings  
- score returns a string that is one of the score labels  
- low confidence raises `LowConfidenceDecision`  
- fallback path works when logprobs unavailable  
