---
name: tabular-export
description: "Writes the shipped code (src/pipeline.py, src/train.py, src/predict.py) from the winning run and trains it. Usually right after evolve-search or tabular-ensemble. Check: run src/train.py, expects models/model.joblib, metric accuracy (classification) or r2 (regression) with a minimum."
allowed-tools: read_file list_files write_file run_python export_pipeline finish_step
metadata:
  beyondml:
    kind: step
    modality: tabular
---
You turn the winning run into code that works without BeyondML.

1. Call export_pipeline(run_id=<the run_id from the upstream summary, or "" for the latest>).
2. src/train.py already applies clean(df) and make_features(df) when src/clean.py and src/features.py exist; do not duplicate them.
3. Run run_python("src/train.py"). If it fails, read stderr_tail and fix the code (usually a column mismatch between features.py and the exported FEATURES list), then run it again.
4. Call finish_step with the reported score, outputs=["src/pipeline.py", "src/train.py", "src/predict.py", "models/model.joblib"].
