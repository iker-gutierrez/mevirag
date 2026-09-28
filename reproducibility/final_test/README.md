# Final test-set artefacts

This directory contains the complete, publication-facing test-set record for
the four evaluated models. It is deliberately separate from
`experiments/runs/` and `reports/metrics/`, which retain the much larger
working archive of intermediate runs.

## Contents

- `configs/`: the eight exact JSON configurations: one selected RAG system
  and one LLM-only baseline for each model.
- `predictions/`: every final-test prediction, for seeds 42, 43, and 44
  (24 JSONL files).
- `evaluations/`: the corresponding mixed-test, GuiaSalud-only, and
  CasiMedicos-only evaluation outputs (72 JSON files).
- `write_test_results_table.py`: regenerates the manuscript test-results
  table from the per-seed evaluation JSONs.

The configuration IDs used in the manuscript are `15a`, `6b`, `5c`, and
`1d` for the selected RAG systems, paired respectively with baselines `0a`,
`0b`, `0c`, and `0d`. No final-test result was used to select a
configuration.

## Recreating the table

From the repository root, run:

```bash
python reproducibility/final_test/write_test_results_table.py
```

The table writer reads the bundled JSON files directly from `evaluations/` and
writes `table_test_results.tex` in this directory. The full development tables
and the MST decisions used to choose configurations are documented in the
accompanying manuscript; the obsolete GuiaSalud-only development summaries
are intentionally not included in this final mixed-dataset bundle.

## Scope and data access

The source splits and retrieval indexes are not duplicated here. GuiaSalud
construction and the fixed split are maintained in the separate
[`guiasalud`](https://github.com/iker-gutierrez/guiasalud) repository;
CasiMedicos-Exp is obtained from its upstream source. Model weights and
retrieval indexes must be rebuilt locally and are intentionally excluded.
