---
name: tabular-ensemble
description: "Combines the search's top models into a stacking ensemble. Only when the top genomes are within about 1% of each other. Check: metric cv_score with a minimum."
allowed-tools: read_file ensemble finish_step
metadata:
  beyondml:
    kind: step
    modality: tabular
---
You try an ensemble of the best distinct models from the latest search.

1. Call ensemble(top_n=5, kind="stacking").
2. Keep it only if cv_score > best_single. If it is not better, say so; the export step will use the single best model.
3. Call finish_step with a summary that includes "run_id=<the ensemble run_id>" when kept, and the two scores, outputs=[].
