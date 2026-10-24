---
name: tabular-audit
description: "First step for a table: profile it, fix the target and problem type, find leaking and impossible columns. Usually the first step of a tabular plan. Check: expects reports/audit.md."
allowed-tools: read_file list_files write_file run_python profile scan_leakage chart ask_user finish_step
metadata:
  beyondml:
    kind: step
    modality: tabular
---
You audit the raw table before anything is built on it.

1. Call profile(path=<the raw file>) and read the target analysis and the ranked target candidates.
2. Confirm the target. Call ask_user("Which column is the target?", options=<the ranked candidates>, recommended=<the top candidate>) and use the answer.
3. Call profile again with target=<the target> and note the problem type (classification or regression).
4. Call scan_leakage(path, target). Treat target_correlation and single_feature_score suspects as leaks; treat id_like columns as drop candidates; name hints need your judgement.
5. Look for impossible values (sentinels such as age 999 or zip 00000, negative counts, future years). Only flag values that are clearly impossible, not merely rare.
6. Write reports/audit.md with write_file: target, problem type, the drop list with one reason per column, impossible values per column.
7. Call finish_step with a summary that states the target, the problem type and the drop list, and outputs=["reports/audit.md"].
