---
name: choose-metric
description: "How to pick the evaluation metric for a tabular problem."
metadata:
  beyondml:
    kind: knowledge
---
- Balanced classification: accuracy is fine; report the confusion matrix too.
- Imbalanced classification: prefer F1 (or recall when missing positives is costly) and look at the minority class.
- Ranking or probability quality: ROC AUC.
- Regression: R2 for explained variance; MAE when outliers should not dominate; RMSE when large errors matter most.
- Always compare against a trivial baseline (majority class, or the mean).
