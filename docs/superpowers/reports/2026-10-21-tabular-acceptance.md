# Tabular acceptance on real models (sub-project 3, Task 14)

Runs from 2026-10-22 to 2026-10-24 on branch `tabular/skills`. Every run used the same command, changing only `--model` and `--project`:

```
python -m beyondml.cli run "predict survival" --data data/titanic.csv --project <scratch>/<name> --mode autonomous --model <model>
```

## Verdict

| Model | Result | Plan | Holdout accuracy |
|---|---|---|---|
| `ollama_chat/qwen3:8b` (local) | **Pass** against the owner's bar | 6 of 6 steps done | 0.7976 |
| Groq `openai/gpt-oss-120b` | **Not completed** | stopped at eda | n/a |

- **The bar:** the plan finishes, and holdout accuracy is at least the v1 baseline minus 0.01. The owner set that tolerance on 2026-10-22. With a v1 baseline of 0.8024, the bar is 0.7924.
- **qwen3:8b against the bar:** it clears it. It is 2 holdout rows (of 420) short of the v1 baseline itself.
- **Shipped code:** the qwen project's `src/train.py` and `src/predict.py` run with `beyondml` blocked from import. No file in `src/` mentions `beyondml`.
- **Groq:** `llama-3.3-70b-versatile`, the model named in the plan, is retired on Groq, so the runs used `openai/gpt-oss-120b`. The free tier caps that model at 200,000 tokens a day. One run used the whole day's quota and reached eda. A full run needs 2–3 days of quota or a paid tier.

## qwen3:8b, run 4 (the accepted run)

The run took 47 minutes. The planner submitted twice. Its first plan named the skill `tab,ular-features`, and validation rejected it.

| Step | Skill | Status | Attempts | Minutes | Notes |
|---|---|---|---|---|---|
| audit | tabular-audit | done | 2 | 19.5 | Attempt 1 passed the report to `finish_step` as `content` and never wrote `reports/audit.md`; the check caught it. |
| clean | tabular-clean | done | 2 | 17.0 | Attempt 1 printed a made-up tool result as text instead of calling `write_file`. |
| eda | tabular-eda | done | 1 | 18.5 | One `chart(kind="heatmap")` was rejected with the valid kinds listed; the agent used `corr`. |
| features | tabular-features | done | 1 | 3.6 | Added `family_size`, `is_alone`, `fare_per_person` and `log_age`. |
| evolve | evolve-search | done | 1 | 3.1 | 8 generations streamed. The winner was LogisticRegression with mu_cv 0.783. |
| export | tabular-export | done | 1 | 3.1 | `src/train.py` reported accuracy 0.7976. |

Audit and clean ran in parallel, because the planner did not make clean depend on audit. The plan also had no evaluate-explain step.

## What each run exposed

| Run | Model | Outcome | Fix |
|---|---|---|---|
| qwen-0 | qwen3:8b | Every agent call failed: "No module named 'mcp'". | `6bfd219` |
| qwen-0b | qwen3:8b | Ollama: "does not support chat". The weights blob was missing on disk; it was re-pulled. | none needed in code |
| groq-0 | llama-3.3-70b | The model is retired on Groq. | switched to gpt-oss-120b |
| groq-0b | gpt-oss-120b | Groq rejected `reasoning_content` on history turns. | `15a029c` |
| groq-0c | gpt-oss-120b | One 429 failed the step and then its replan, 46 s in. | `c45eee8` |
| groq-1 | gpt-oss-120b | Server-side schema check rejected `chart(kind="corr")` without `x`. Every Parquet write failed. Then a silent 27-minute wait on a quota limit. | `4469f50`, `6fab424`, `d8817cd` |
| groq-2 | gpt-oss-120b | Daily quota exhausted. It failed in 43 s with the provider's message. | none needed; fail-fast worked |
| qwen-1 | qwen3:8b | The planner invented outputs (`models/best_model.pkl`, a root `train.py`) and an export metric `train.py` never writes (`f1`). | `cb11303` |
| qwen-2 | qwen3:8b | Empty checks let four steps pass on `finish_step` alone. Clean wrote `clean.py` instead of `clean.csv`. Evolve called `finish_step` 20 times in one turn. | `ec00083` |
| qwen-3 | qwen3:8b | Step agents guessed Kaggle's column names (`Survived`, `Pclass`), and eda aborted on three chart errors. `profile` accepted a missing target. | `5edb649` |
| qwen-4 | qwen3:8b | Passed (above). Exposed the `finish_step(content=...)` habit. | `0ab2752` |

## Fixes made

Each fix has a regression test that failed first and now passes.

- `6bfd219`: declare `mcp`. litellm 1.106 imports it on every call that carries tools, but doesn't declare it.
- `15a029c`: drop replayed thought parts from Groq history turns.
- `c45eee8`: wait out provider rate limits. litellm's `num_retries` retries immediately.
- `d8817cd`: log each rate-limit wait. A wait longer than 120 s is a quota, so the call fails at once.
- `4469f50`: `chart`'s `x` is optional.
- `6fab424`: declare `pyarrow`.
- `17be9ca`: the console shows a failed script's last stderr line, not "error: None".
- `cb11303`: each step skill's description names its standard check, and the planner uses it as the step's `success_check`.
- `ec00083`: plans need a real success check (`run`, `expects` or a metric), and `finish_step` ends the agent's turn.
- `5edb649`: step agents see `DATA.md`. `profile` and `chart` reject a missing column and list the real ones.
- `0ab2752` (skill text, qwen round 1 of 3): every "Write <file>" instruction names `write_file`.

## Not verified, or still open

- **`0ab2752`:** it landed after run 4 started, so no run has exercised it yet.
- **Groq acceptance:** open. It needs a paid tier or several days of free quota.
- **The bar was checked by hand:** run 4's export check had no minimum.
- **Planner quality on qwen3:8b:**
  - Run 4 left out evaluate-explain.
  - It missed the clean → audit dependency.
  - It left export's check without a minimum.
- **evaluate-explain and its health gate** have not run on a real model.
- **Parallel tool errors:** "three tool errors in a row" counts parallel calls from one model response separately. That aborted eda in qwen-3, where one wrong assumption produced three errors at once.
