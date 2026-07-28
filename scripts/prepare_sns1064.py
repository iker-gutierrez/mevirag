#!/usr/bin/env python
"""Loads the pre-split SNS1064 train/dev/test.csv from the SNS1064-dataset
repository (https://github.com/iker-gutierrez/SNS1064-dataset) and writes
them out in this project's own JSONL/CSV layout.

This no longer re-derives a train/dev/test split from a flat, unsplit
dataset.csv: the split used for every experiment in this thesis was fixed
once, is now the canonical split published in the dataset repository itself
(train_df.csv/dev_df.csv/test_df.csv there), and re-splitting here would
silently produce a DIFFERENT split (verified: re-running the old version of
this script against dataset.csv with its own default args reproduces
neither the split that was actually used for the reported experiments, nor
any single canonical split -- there is no "the" split to derive on demand,
only the one already fixed and published).

Usage:
  python scripts/prepare_sns1064.py \\
      --train-df /path/to/train_df.csv \\
      --dev-df   /path/to/dev_df.csv \\
      --test-df  /path/to/test_df.csv \\
      --output-dir data/processed/sns1064
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag_thesis.data_io import (  # noqa: E402
    clean_text,
    compact_record,
    ensure_id,
    find_column,
    read_table,
    records_to_dataframe,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load the pre-split SNS1064 train/dev/test CSVs (from the "
        "SNS1064-dataset GitHub repository) into this project's JSONL layout."
    )
    parser.add_argument("--train-df", required=True, help="train_df.csv from the SNS1064-dataset repo.")
    parser.add_argument("--dev-df", required=True, help="dev_df.csv from the SNS1064-dataset repo.")
    parser.add_argument("--test-df", required=True, help="test_df.csv from the SNS1064-dataset repo.")
    parser.add_argument("--output-dir", required=True, help="Directory for all/train/dev/test files.")
    parser.add_argument("--source", default="SNS1064")
    return parser.parse_args()


def normalize_split(input_path: str, source: str, split: str, id_offset: int) -> list[dict[str, str]]:
    """Normalize one pre-split CSV (train_df.csv/dev_df.csv/test_df.csv) into
    this project's record schema. Handles both the current schema (id,
    guidebook, topic, question, subquestion, judgement, evidence, split --
    evidence already includes considerations, merged) and, defensively, the
    dataset repo's older 7-column raw schema (guidebook, topic, question,
    subquestion, judgement, evidence, considerations, no id/split), in case
    a future export reverts to it.
    """
    df = read_table(input_path)
    id_col = find_column(df, ["id", "sample_id", "question_id"], required=False)
    guidebook_col = find_column(df, ["guidebook", "guide_book", "guia", "guía"], required=False)
    topic_col = find_column(df, ["Topic", "topic", "tema"], required=False)
    question_col = find_column(df, ["Question", "question", "pregunta"])
    subquestion_col = find_column(
        df,
        ["subquestion", "sub_question", "subpregunta", "sub pregunta"],
        required=False,
    )
    short_answer_col = find_column(
        df,
        ["Short answer", "short_answer", "answer", "judgement", "juicio", "respuesta corta", "respuesta"],
    )
    evidence_col = find_column(df, ["evidence"], required=False)
    additional_col = find_column(df, ["considerations"], required=False)
    split_col = find_column(df, ["split"], required=False)

    records = []
    for index, row in df.iterrows():
        question = clean_text(row[question_col])
        subquestion = clean_text(row[subquestion_col]) if subquestion_col else ""
        evidence_parts = []
        evidence = clean_text(row[evidence_col]) if evidence_col else ""
        additional = clean_text(row[additional_col]) if additional_col else ""
        if evidence:
            evidence_parts.append(evidence)
        if additional:
            evidence_parts.append(additional)
        row_split = clean_text(row[split_col]) if split_col else split
        record = compact_record(
            {
                "id": ensure_id("sns1064", id_offset + int(index), row[id_col] if id_col else None),
                "source": source,
                "guidebook": clean_text(row[guidebook_col]) if guidebook_col else "",
                "topic": clean_text(row[topic_col]) if topic_col else "",
                "question": question,
                "subquestion": subquestion,
                "short_answer": clean_text(row[short_answer_col]),
                "evidence": "\n\n".join(evidence_parts),
                "split": row_split or split,
            }
        )
        if record.get("question") and record.get("short_answer"):
            records.append(record)
    if not records:
        raise ValueError(f"No usable SNS1064 rows found in {input_path}.")
    return records


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = normalize_split(args.train_df, args.source, "train", id_offset=0)
    dev_records = normalize_split(args.dev_df, args.source, "dev", id_offset=len(train_records))
    test_records = normalize_split(
        args.test_df, args.source, "test", id_offset=len(train_records) + len(dev_records)
    )
    records = [*train_records, *dev_records, *test_records]

    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate ids across train/dev/test after normalization.")

    write_jsonl(records, output_dir / "all.jsonl")
    records_to_dataframe(records).to_csv(output_dir / "all.csv", index=False)

    summary = {
        "inputs": {"train_df": args.train_df, "dev_df": args.dev_df, "test_df": args.test_df},
        "num_records": len(records),
        "splits": dict(Counter(record["split"] for record in records)),
        "topics": dict(Counter(record.get("topic", "") for record in records if record.get("topic"))),
    }
    for split in ["train", "dev", "test"]:
        split_records = [record for record in records if record["split"] == split]
        write_jsonl(split_records, output_dir / f"{split}.jsonl")
        records_to_dataframe(split_records).to_csv(output_dir / f"{split}.csv", index=False)

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
