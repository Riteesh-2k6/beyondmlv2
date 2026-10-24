---
name: tabular-clean
description: "Writes src/clean.py: drops audited columns, fixes impossible values, imputes and handles outliers. Usually right after tabular-audit. Check: run src/clean.py, expects data/processed/clean.csv."
allowed-tools: read_file list_files write_file run_python profile detect_outliers ask_user finish_step
metadata:
  beyondml:
    kind: step
    modality: tabular
---
You write the cleaning code the shipped project reuses for training and prediction.

1. Read reports/audit.md for the target, the drop list and the impossible values.
2. Call detect_outliers(path=<the raw file>) to see which numeric columns have outliers.
3. Choose the outlier strategy. Call ask_user("How should outliers be handled?", options=["cap", "remove", "keep", "transform"], recommended=<your choice>). cap clips at the IQR bounds; transform applies np.log1p to skewed positive columns; keep when outliers are informative.
4. Imputation per column: median for skewed numbers, mean for roughly normal ones, the most frequent value for categories, a constant such as "Unknown" when missing means something, drop the column when over 70% is missing.
5. Write src/clean.py with write_file. It defines clean(df) -> df. It must not require the target column (predict.py calls it on new rows), must not drop rows unless the strategy is remove, and must also run as a script: read the raw file from data/raw, write data/processed/clean.csv.
6. Run it with run_python("src/clean.py") and fix it until it succeeds.
7. Call finish_step with the strategies you chose, outputs=["src/clean.py", "data/processed/clean.csv"].
