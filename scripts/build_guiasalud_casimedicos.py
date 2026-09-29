#!/usr/bin/env python
"""Build the Spanish mixed dataset (guiasalud_casimedicos), the Spanish
counterpart of scripts/build_guiasalud_casimedicos_eu.py (Basque). Same
schema, same per-split concatenation pattern: guiasalud records keep
`guidebook` and get an empty `specialty`, casimedicos records keep
`specialty` (read from `topic`, casimedicos's own on-disk name for the MIR
exam-section label -- see main.tex's sec:translation-qe for why it's called
`type` in the original CasiMedicos-Exp dataset and `specialty` in the mixed
schema) and get an empty `guidebook`. `query`/`justification`/`short_answer`
are already present in both source files, so this script only selects and
relabels fields, it does not build or rebuild any content.

Usage:
    python scripts/build_guiasalud_casimedicos.py
    python scripts/build_guiasalud_casimedicos.py --all-only
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT_DIR = PROCESSED / "guiasalud_casimedicos"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all-only",
        action="store_true",
        help="Rebuild all.jsonl from the existing mixed split files only.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(records: list[dict], path: Path) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )


def mixed_record(record: dict, *, is_guiasalud: bool) -> dict:
    out = {
        "id": record["id"],
        "guidebook": record.get("guidebook", "") if is_guiasalud else "",
        "specialty": "" if is_guiasalud else record.get("topic", ""),
        "query": record.get("query", ""),
        "short_answer": record.get("short_answer", ""),
        "justification": record.get("justification", ""),
    }
    if not is_guiasalud:
        out["options"] = record.get("options", {})
    return out


def rebuild_all_from_splits() -> None:
    """Write the audit-only aggregate without touching any split or index.

    ``all.jsonl`` carries ``split`` explicitly, while the individual mixed
    files imply it through their filenames.  Keeping this reconstruction here
    prevents the aggregate from becoming stale after a curated source refresh.
    """
    records: list[dict] = []
    for split in ("train", "dev", "test"):
        split_path = OUT_DIR / f"{split}.jsonl"
        if not split_path.exists():
            raise FileNotFoundError(f"missing mixed split: {split_path}")
        records.extend({**record, "split": split} for record in load_jsonl(split_path))
    write_jsonl(records, OUT_DIR / "all.jsonl")
    print(f"all: {len(records)} records from existing mixed splits -> {OUT_DIR / 'all.jsonl'}")


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.all_only:
        rebuild_all_from_splits()
        return
    for split in ("train", "dev", "test"):
        gs_path = PROCESSED / "guiasalud" / f"{split}.jsonl"
        cm_path = PROCESSED / "casimedicos" / f"{split}.jsonl"
        gs_records = [mixed_record(r, is_guiasalud=True) for r in load_jsonl(gs_path)]
        cm_records = [mixed_record(r, is_guiasalud=False) for r in load_jsonl(cm_path)]
        combined = gs_records + cm_records
        out_path = OUT_DIR / f"{split}.jsonl"
        write_jsonl(combined, out_path)
        print(f"{split}: {len(gs_records)} guiasalud + {len(cm_records)} casimedicos = {len(combined)} -> {out_path}")
    rebuild_all_from_splits()


if __name__ == "__main__":
    main()
