# [TICKET-008] Integration test: full pipeline with constrained decisions
parent: MAP-001
labels: wayfinder:task
status: open
blocked-by: TICKET-003, TICKET-004, TICKET-005, TICKET-006, TICKET-007

## Question

Does the full constrained-decoding integration work end-to-end in the supervised pipeline?

Write `tests/test_constrained_integration.py` that:
1. Constructs a `ConstrainedOllamaProvider` backed by `MockConstrainedLLM` (extends `MockLLM` to return `(answer, logprob_dict)` pairs).
2. Runs `LeakageAgent`, `SanityAgent`, `OutlierAgent`, `FeatureAgent`, `ReflectionAgent` in sequence on `classification_df`.
3. Asserts: no uncaught exceptions, all agents return their expected dict shapes, confidence is a float between 0 and 1, TUI escalation fires exactly for low-confidence cases.
4. Tests the `get_user_input=None` auto-accept path (benchmark / non-interactive mode).
5. Tests no regression: existing `tests/test_agents.py` still passes unchanged.

Also: run `PYTHONPATH=. pytest tests/ -v` and confirm all existing tests still green before merging.

Decisions:
1. `MockConstrainedLLM` fixture shape — confirm it lives in `tests/conftest.py`.
2. Which confidence values to inject per agent to exercise both auto-apply and escalation paths.
3. Assert that `ConstrainedResult.confidence` is in `[0, 1]` for all calls.
