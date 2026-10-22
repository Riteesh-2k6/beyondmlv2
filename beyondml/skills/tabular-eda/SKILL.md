---
name: tabular-eda
description: "Explores the cleaned table and writes insights for the feature step. Usually right after tabular-clean. Check: expects reports/eda.md."
allowed-tools: read_file list_files write_file run_python profile chart finish_step
metadata:
  beyondml:
    kind: step
    modality: tabular
---
You explore the cleaned data and leave findings the feature step can act on.

1. Call profile(path="data/processed/clean.csv", target=<the target>).
2. Draw 2-4 charts with chart(...): the target distribution, the strongest relationships with the target, a corr heatmap for numeric columns.
3. Write reports/eda.md with 3-5 insights. Each insight states what you saw and what it implies for modelling (skew, imbalance, interactions worth a feature, columns that look redundant).
4. Call finish_step with a two-line summary, outputs=["reports/eda.md"].
