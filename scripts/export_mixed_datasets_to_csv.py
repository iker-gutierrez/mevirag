#!/usr/bin/env python
"""Export the mixed GuiaSalud+CasiMedicos-Exp dataset JSONL files
(guiasalud_casimedicos, guiasalud_casimedicos_eu) to CSV, for visual
inspection only -- JSONL remains the format every script in the pipeline
(retrieval indexing, generation, evaluation) actually reads, since query/
justification now contain embedded newlines (dash-bulleted sub-fields) that
JSON represents unambiguously but CSV would need careful quoting for, and a
spreadsheet tool re-saving the file could silently corrupt that structure.
This export is read-only output: nothing downstream consumes the CSVs.

Usage:
    python scripts/export_mixed_datasets_to_csv.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

TARGET_DIRS = ["guiasalud_casimedicos", "guiasalud_casimedicos_eu"]
SPLITS = ["train", "dev", "test"]


def export_file(jsonl_path: Path, csv_path: Path) -> int:
    if not jsonl_path.exists():
        return 0
    records = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not records:
        return 0
    # Union of every record's keys, in first-seen order: CasiMedicos-Exp
    # records carry an extra `options` field GuiaSalud records don't, so the
    # first record alone may not cover every column (e.g. dev.jsonl's own
    # 63 GuiaSalud records are listed before its 63 CasiMedicos-Exp ones).
    fieldnames: list[str] = []
    seen = set()
    for record in records:
        for key in record:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
                   for k, v in record.items()}
            writer.writerow(row)
    return len(records)


def main() -> None:
    for dir_name in TARGET_DIRS:
        base = PROCESSED / dir_name
        for split in SPLITS:
            jsonl_path = base / f"{split}.jsonl"
            csv_path = base / f"{split}.csv"
            n = export_file(jsonl_path, csv_path)
            print(f"{dir_name}/{split}.csv: {n} records")


if __name__ == "__main__":
    main()
