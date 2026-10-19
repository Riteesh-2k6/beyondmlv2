---
name: python-step
description: "Generic plan step: write a standalone Python script under src/ that turns the step's inputs into its outputs, run it, fix it until it succeeds, then call finish_step."
allowed-tools: read_file write_file list_files run_python ask_user finish_step
metadata:
  beyondml:
    kind: step
    modality: any
---
# Python step

- Put the script in `src/<step_id>.py`, runnable as `python src/<step_id>.py` from the project root.
- Read only `data/raw/` and upstream outputs; never write under `data/raw/`.
- Write key numbers as JSON to `os.environ.get("BEYONDML_METRICS", "metrics.json")`.
- Print a short summary.
- After a failure, read `stderr_tail` and fix the script rather than starting over.
- Finish with `finish_step(summary, outputs)` listing every output path.
