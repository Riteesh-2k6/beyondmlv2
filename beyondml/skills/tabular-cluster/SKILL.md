---
name: tabular-cluster
description: "Groups rows when there is no target: evolves KMeans/DBSCAN for the best silhouette and exports the code. Only when the data has no target."
allowed-tools: read_file list_files write_file run_python profile cluster chart export_pipeline finish_step
metadata:
  beyondml:
    kind: step
    modality: tabular
---
You find natural groups in the data.

1. Call profile(path=<the cleaned file, or the raw file if there is no clean step>).
2. Call cluster(data=<that file>). Use algorithms=["kmeans"] for large data.
3. Draw a scatter of two informative columns coloured by cluster with chart("scatter", "data/processed/clusters.csv", x, y).
4. Call export_pipeline() and run run_python("src/train.py").
5. Call finish_step with the silhouette and the cluster count, outputs=["data/processed/clusters.csv", "src/pipeline.py", "src/train.py"].
