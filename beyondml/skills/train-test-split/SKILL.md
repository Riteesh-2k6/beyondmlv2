---
name: train-test-split
description: "How to split data so the score is honest."
metadata:
  beyondml:
    kind: knowledge
---
- Split before any fitting; fit imputers, scalers and encoders on the training part only (pipelines do this).
- Stratify classification splits.
- Time-ordered data: split by time, never at random.
- Grouped data (several rows per customer): keep each group on one side.
- Use the same random_state everywhere so results are reproducible.
