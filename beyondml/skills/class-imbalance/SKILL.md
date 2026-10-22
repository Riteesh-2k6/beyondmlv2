---
name: class-imbalance
description: "What to do when one class is rare."
metadata:
  beyondml:
    kind: knowledge
---
- Below about 20% minority share, accuracy misleads; use F1 or recall.
- Prefer class_weight="balanced" (the search applies it when the profile flags imbalance) over resampling.
- Stratify every split so the rare class appears in train and test.
- Do not oversample before splitting; that leaks copies of test rows into training.
