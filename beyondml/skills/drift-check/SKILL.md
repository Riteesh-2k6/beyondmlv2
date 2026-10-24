---
name: drift-check
description: "Compares a new batch with the data the model was built on. Only when the inputs include a reference set and a newer batch. Check: expects reports/drift.md."
allowed-tools: read_file list_files write_file drift chart finish_step
metadata:
  beyondml:
    kind: step
    modality: tabular
---
You check whether new data still looks like the training data.

1. Call drift(reference=<the training file>, current=<the new batch>).
2. For the drifted columns, draw a histogram of each with chart("hist", ...).
3. Write reports/drift.md with write_file: drifted columns with p-values, and whether retraining is advisable.
4. Call finish_step with the drifted columns, outputs=["reports/drift.md"].
