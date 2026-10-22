---
name: evaluate-explain
description: "Holds out data, checks for leakage (health) and warns on overfitting and dominance, and explains the model. Usually the last step of a supervised plan. Check: metric health, min 1."
allowed-tools: read_file list_files write_file evaluate explain chart finish_step
metadata:
  beyondml:
    kind: step
    modality: tabular
---
You judge whether the model is trustworthy.

1. Call explain(data="data/processed/features.csv", target=<the target>).
2. Call evaluate(data="data/processed/features.csv", target=<the target>). health is 0 only when the score is 0.999 or more (likely leakage). warnings lists a train/test gap over 0.10 ("gap") or one feature carrying over half the importance ("top_share").
3. Write reports/evaluation.md: score, gap, top features, health, and one line per warning explaining it. A dominant feature can be legitimate (gender on titanic); say whether you think it is, and never drop a legitimate feature just to spread importance.
4. If health is 0, call finish_step with the reason stated plainly so the re-plan can remove the leak. Otherwise summarise the result and the warnings.
5. outputs=["reports/evaluation.md"].
