#!/usr/bin/env python
"""Smoke check: does evaluate_predictions.py's reference-enrichment actually
find non-empty gold text for a real predictions file, or does it silently
score against an empty/missing reference?

Motivated by a real investigation (2026-08-17): a metrics summary with a
near-zero ROUGE-L was initially (wrongly) suspected to be caused by a broken
reference lookup, because the raw predictions.jsonl itself has no
`reference_justification` key -- which is EXPECTED (that field is only
added by enrich_records_with_references() at scoring time, never written
back to the predictions file), not a bug. This check exists so that
diagnosis takes one command instead of a multi-step manual trace next time,
and so a GENUINE regression (e.g. the input dataset path resolving to a
stale/wrong file, or the gold field actually being empty) is caught loudly
instead of silently deflating a MeanQ score.

Checks, for a given predictions.jsonl:
  1. Its reference path (resolved via predictions.meta.json's own `input`
     field, exactly like evaluate_predictions.py does) exists on disk.
  2. reference_sections() returns non-empty short_answer AND evidence text
     for at least N records (default: every record that has a resolvable
     reference at all -- a record legitimately missing gold data, e.g. a
     malformed source row, should not silently pass as "fine").
  3. Prints the actual resolved reference path and a live example, so a
     human can eyeball it rather than trust a boolan alone.

Usage:
    python scripts/check_reference_enrichment.py \\
        experiments/runs/<run>/predictions.jsonl
    python scripts/check_reference_enrichment.py \\
        experiments/runs/5000_llama31_8b_no_rag_extractive_guiasalud_dev_seed42/predictions.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from evaluate_predictions import enrich_records_with_references, reference_path_from_metadata  # noqa: E402
from medical_rag_thesis.data_io import read_jsonl  # noqa: E402
from medical_rag_thesis.evaluation import reference_sections  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("predictions", help="Path to a predictions.jsonl file.")
    args = ap.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        print(f"FAIL: predictions file does not exist: {pred_path}")
        raise SystemExit(1)

    ref_path = reference_path_from_metadata(pred_path)
    print(f"predictions: {pred_path}")
    print(f"resolved reference path: {ref_path}")
    if ref_path is None or not ref_path.exists():
        print("FAIL: reference path missing or does not exist -- every record will silently "
              "score against an EMPTY gold reference (enrich_records_with_references() is a "
              "no-op when references_path is None/missing, per its own early-return).")
        raise SystemExit(1)

    records = read_jsonl(pred_path)
    enriched = enrich_records_with_references(records, ref_path)

    n_total = len(enriched)
    n_empty_short_answer = 0
    n_empty_evidence = 0
    example_shown = False
    for record in enriched:
        sections = reference_sections(record)
        if not sections["short_answer"].strip():
            n_empty_short_answer += 1
        if not sections["evidence"].strip():
            n_empty_evidence += 1
        if not example_shown and sections["short_answer"].strip() and sections["evidence"].strip():
            print(f"\nlive example ({record.get('id')}):")
            print(f"  reference short_answer: {sections['short_answer'][:120]!r}")
            print(f"  reference evidence:     {sections['evidence'][:120]!r}")
            example_shown = True

    print(f"\n{n_total} records checked")
    print(f"  empty reference short_answer: {n_empty_short_answer}/{n_total}")
    print(f"  empty reference evidence:     {n_empty_evidence}/{n_total}")

    # Some legitimate records (e.g. a source row with a genuinely blank
    # field) can be empty, but if MOST records come back empty, references
    # are not resolving and every downstream MeanQ number is deflated.
    threshold = 0.5
    if n_empty_short_answer / n_total > threshold or n_empty_evidence / n_total > threshold:
        print(f"\nFAIL: more than {threshold:.0%} of records have an empty reference field. "
              "Reference enrichment is very likely broken (wrong path, empty source dataset "
              "field, or a dataset rebuild mid-flight) -- do not trust this run's metrics.")
        raise SystemExit(1)

    print("\nPASS: reference enrichment resolves real gold text for the large majority of records.")


if __name__ == "__main__":
    main()
