# Final test results

This directory contains the configurations, predictions, and evaluation
outputs for the four systems evaluated on the held-out mixed test set.

## Contents

- `configs/`: the eight exact JSON configurations: one selected system
  and one LLM-only baseline for each model.
- `predictions/`: every final-test prediction, for seeds 42, 43, and 44
  (24 JSONL files).
- `evaluations/`: the corresponding mixed-test, GuiaSalud-only, and
  CasiMedicos-only evaluation outputs (72 JSON files).
- [`results.md`](results.md): the consolidated held-out test results in
  Markdown.
- `write_test_results_table.py`: regenerates both the Markdown and LaTeX
  test-results tables from the per-seed evaluation JSONs.

The selected-system table identifiers are `15a`, `6b`, `5c`, and `1d`, paired
respectively with retrieval-free baselines `0a`, `0b`, `0c`, and `0d`.
Configurations were selected exclusively on the development set. Test results
were not used for configuration selection.

## Recreating the table

From the repository root, run:

```bash
python reproducibility/final_test/write_test_results_table.py
```

The table writer reads the JSON files directly from `evaluations/` and writes
`results.md` and `table_test_results.tex` in this directory.

## Scope and data access

The source splits and retrieval indexes are not stored in this directory.
GuiaSalud construction and its fixed splits are maintained in the separate
[`guiasalud`](https://github.com/iker-gutierrez/guiasalud) repository.
CasiMedicos-Exp is obtained from its upstream source. Model weights and
retrieval indexes must be downloaded or rebuilt locally.
