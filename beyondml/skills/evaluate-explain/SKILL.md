---
name: evaluate-explain
description: "Holds out data, checks for overfitting, dominance and leakage (health), and explains the model. Usually the last step of a supervised plan."
allowed-tools: read_file list_files write_file evaluate explain chart finish_step
metadata:
  beyondml:
    kind: step
    modality: tabular
---
You judge whether the model is trustworthy.

1. Call explain(data="data/processed/features.csv", target=<the target>).
2. Call evaluate(data="data/processed/features.csv", target=<the target>). health is 0 when the train/test gap is over 0.10, one feature carries over half the importance, or the score is 0.999 or more (likely leakage).
3. Write reports/evaluation.md: score, gap, top features, health and its reasons, and what you would change.
4. If health is 0, call finish_step with the reasons stated plainly so the re-plan can fix them (drop the dominant or leaking feature, constrain the model). Otherwise summarise the result.
5. outputs=["reports/evaluation.md"].
