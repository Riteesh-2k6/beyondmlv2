---
name: tabular-features
description: "Writes src/features.py with derived features (ratios, logs, interactions) and no target use. Usually right after tabular-eda. Check: run src/features.py, expects data/processed/features.csv."
allowed-tools: read_file list_files write_file run_python profile scan_leakage ask_user finish_step
metadata:
  beyondml:
    kind: step
    modality: tabular
---
You add 2-4 derived features that give the model more signal.

1. Read reports/eda.md and reports/audit.md.
2. Propose features as vectorised pandas expressions only: arithmetic, ratios, np.log1p, sums of counts. Reference existing columns exactly. Never use the target, never import beyond pandas/numpy, never read files inside make_features.
3. Confirm them. Call ask_user("Which proposed features should be kept?", options=<the proposal names>, recommended=<all proposals that are not leakage suspects, comma-separated>).
4. Write src/features.py defining make_features(df) -> df that adds the accepted features. It also runs as a script: read data/processed/clean.csv, write data/processed/features.csv (keeping the target column).
5. Run it with run_python("src/features.py"), then call scan_leakage("data/processed/features.csv", target) and drop any new feature it flags.
6. Call finish_step listing the features kept, outputs=["src/features.py", "data/processed/features.csv"].
