#!/usr/bin/env python
"""Loads the pre-split GuiaSalud train/dev/test.csv from the guiasalud
repository (https://github.com/iker-gutierrez/guiasalud) and writes
them out in this project's own JSONL/CSV layout.

This no longer re-derives a train/dev/test split from a flat, unsplit
dataset.csv: the split used for every experiment in this thesis was fixed
once, is now the canonical split published in the dataset repository itself
(train_df.csv/dev_df.csv/test_df.csv there), and re-splitting here would
silently produce a DIFFERENT split (verified: re-running the old version of
this script against dataset.csv with its own default args reproduces
neither the split that was actually used for the reported experiments, nor
any single canonical split, there is no "the" split to derive on demand,
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
from medical_rag_thesis.prompts import format_question  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load the pre-split GuiaSalud train/dev/test CSVs (from the "
        "guiasalud GitHub repository) into this project's JSONL layout."
    )
    parser.add_argument("--train-df", required=True, help="train_df.csv from the guiasalud repo.")
    parser.add_argument("--dev-df", required=True, help="dev_df.csv from the guiasalud repo.")
    parser.add_argument("--test-df", required=True, help="test_df.csv from the guiasalud repo.")
    parser.add_argument("--output-dir", required=True, help="Directory for all/train/dev/test files.")
    parser.add_argument("--source", default="GuiaSalud")
    return parser.parse_args()


def build_justification(evidence: str, considerations: str) -> str:
    """Compose the `justification` field the same way prompts.format_question
    composes `query`: one dash-bulleted labeled line per non-empty part
    (matching format_question's own bullet convention, so the model sees the
    same "each line is a component of the block above it" cue in both
    fields), skipping any blank field entirely (considerations is empty in
    ~18% of GuiaSalud records per data/interim/guiasalud/postcuration_log.txt,
    so this must not emit a bare "- Consideraciones adicionales:" line for
    those)."""
    lines = []
    if evidence:
        lines.append(f"- Evidencia procedente de la investigación: {evidence}")
    if considerations:
        lines.append(f"- Consideraciones adicionales: {considerations}")
    return "\n".join(lines)


def normalize_split(input_path: str, source: str, split: str, id_offset: int) -> list[dict[str, str]]:
    """Normalize one pre-split CSV (train.csv/dev.csv/test.csv) into this
    project's record schema: id, guidebook, topic, subtopic, question,
    focus, judgement, evidence, considerations, split. `subquestion` no
    longer exists as a single column, it was split into `subtopic` (the
    more specific subdivided question) and `focus` (a population subgroup
    or comparison-arm label); `evidence`/`considerations` are kept as their
    own separate fields, not merged, matching datasetting_precuration.py's
    current 7-field schema (topic/subtopic/question/focus/judgement/
    evidence/considerations). `short_answer` is still read as a fallback
    label for `judgement` (used downstream in scripts/reports that expect a
    `short_answer` column), defensively, in case an older export or a
    different upstream source still uses that name instead of `judgement`.
    """
    df = read_table(input_path)
    id_col = find_column(df, ["id", "sample_id", "question_id"], required=False)
    guidebook_col = find_column(df, ["guidebook", "guide_book", "guia", "guía"], required=False)
    topic_col = find_column(df, ["Topic", "topic", "tema"], required=False)
    subtopic_col = find_column(df, ["subtopic", "sub_topic", "subtema"], required=False)
    question_col = find_column(df, ["Question", "question", "pregunta"])
    focus_col = find_column(
        df,
        ["focus", "foco", "subquestion", "sub_question", "subpregunta", "sub pregunta"],
        required=False,
    )
    judgement_col = find_column(
        df,
        ["judgement", "juicio", "Short answer", "short_answer", "answer", "respuesta corta", "respuesta"],
        required=False,
    )
    evidence_col = find_column(df, ["evidence"], required=False)
    considerations_col = find_column(df, ["considerations"], required=False)
    split_col = find_column(df, ["split"], required=False)

    records = []
    for index, row in df.iterrows():
        question = clean_text(row[question_col])
        subtopic = clean_text(row[subtopic_col]) if subtopic_col else ""
        focus = clean_text(row[focus_col]) if focus_col else ""
        judgement = clean_text(row[judgement_col]) if judgement_col else ""
        evidence = clean_text(row[evidence_col]) if evidence_col else ""
        considerations = clean_text(row[considerations_col]) if considerations_col else ""
        row_split = clean_text(row[split_col]) if split_col else split
        topic = clean_text(row[topic_col]) if topic_col else ""
        query = format_question(
            {"topic": topic, "subtopic": subtopic, "question": question, "focus": focus}
        )
        justification = build_justification(evidence, considerations)
        record = compact_record(
            {
                "id": ensure_id("guiasalud", id_offset + int(index), row[id_col] if id_col else None),
                "source": source,
                "guidebook": clean_text(row[guidebook_col]) if guidebook_col else "",
                "topic": topic,
                "subtopic": subtopic,
                "question": question,
                "focus": focus,
                "judgement": judgement,
                "short_answer": judgement,
                "evidence": evidence,
                "considerations": considerations,
                "query": query,
                "justification": justification,
                "split": row_split or split,
            }
        )
        if record.get("question") and record.get("judgement"):
            records.append(record)
    if not records:
        raise ValueError(f"No usable GuiaSalud rows found in {input_path}.")
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
