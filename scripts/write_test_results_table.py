#!/usr/bin/env python
"""Generates the test-set results table (manuscript/table_test_results.tex):
each model's own best frozen dev config, run against test.jsonl instead of
dev.jsonl, 3 seeds each (slurm/test_set_inference.sh, slurm/eval_test_set.sh).

Reuses write_result_tables.py's own fmt()/esc()/mean_std()/value_or_none()
so this table follows the same row-format and MeanQ-aggregation conventions
as the other MeviRAG result tables, but reads metrics
directly by stem (test-set run dirs are named "{stem}_seed{N}", not the
"{prefix}_{base}_seed{N}" pattern collect() assumes) rather than reusing
collect() itself.

Columns are: #, Model, Config, SF, Quality
(ROUGE-L, BERT-F1, MC-acc, MeanQ), Cost (sec, tok).

Usage:
  python scripts/write_test_results_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from write_result_tables import (  # noqa: E402
    METRICS_DIR, SEEDS, RAW_QUALITY_FIELDS,
    esc, fmt, mean_std, value_or_none, load_summary, values,
)

OUT_PATH = ROOT / "manuscript" / "table_test_results.tex"

QUALITY = [
    ("rouge_l_f1", "ROUGE-L"),
    ("bertscore_f1", "BERT-F1"),
    ("mc_accuracy", "MC-acc"),
    ("meanq", "MeanQ"),
]

# (row_id, model_label, config_label, stem, use_sf): row_id is the SAME
# unique row id (number + model letter, e.g. "15a", "6b", "5c", "1d") that this
# exact configuration already carries in the table where it was originally
# introduced. It is distinct from the server-side configuration number
# (1530/1280/1042/1053), which is used for experiment tracking. Qwen
# no-think's row is a reasoning-pipeline
# row (table_reasoning_es.tex's "15a", MA-RAG, the
# only pipeline that beat its own single-pass baseline, see
# sec:results-reasoning). The other three are each model's own single-pass
# ablation winner (row "6b" for Qwen think, "5c" for Llama, and "1d" for
# Latxa), matching write_result_tables.py's global MODEL_LETTERS
# scheme (a=Qwen no-think, b=Qwen think, c=Llama, d=Latxa) so this table's ids
# are traceable back to the exact dev-set row the test run was frozen from.
#
# use_sf is each model's own dev-set MeanQ-selected SF state at this config
# (all four current winners are their noSF state), NOT a blanket True for
# every self-feedback-capable config,
# self_feedback=True in a config JSON only means the pipeline COMPUTES both
# passes, not that the SF pass is the one that actually wins in the dev
# ablation.
ROWS = [
    ("0a", "Qwen no-think", "Baseline LLM only",
     "17100_qwen35_9b_no_rag_no_think_extractive_guiasalud_llm_only_final_test", False),
    ("15a", "Qwen no-think", "MA-RAG",
     "17000_qwen35_9b_no_think_marag_e5_topk5_extractive_guiasalud_final_test_costaware", False),
    ("0b", "Qwen think", "Baseline LLM only",
     "17101_qwen35_9b_no_rag_think_extractive_guiasalud_llm_only_final_test", False),
    ("6b", "Qwen think", "rerank top 5",
     "17001_qwen35_9b_rag_e5_rerank5_think_extractive_guiasalud_final_test", False),
    ("0c", "Llama", "Baseline LLM only",
     "17102_llama31_8b_no_rag_extractive_guiasalud_llm_only_final_test", False),
    ("5c", "Llama", "rerank top 3",
     "17002_llama31_8b_rag_e5_rerank3_extractive_guiasalud_final_test", False),
    ("0d", "Latxa", "Baseline LLM only",
     "17103_latxa_llama31_8b_no_rag_extractive_guiasalud_llm_only_final_test", False),
    ("1d", "Latxa", "e5 top 1",
     "17003_latxa_llama31_8b_rag_e5_topk1_extractive_guiasalud_final_test", False),
]

# Shown directly below each model's selected RAG row.  Each delta is paired by
# seed and defined as RAG minus that model's LLM-only baseline.
DELTA_PAIRS = {
    "15a": "0a",
    "6b": "0b",
    "5c": "0c",
    "1d": "0d",
}


def _nested_mean(block: dict, name: str) -> float:
    value = (block.get(name) or {}).get("mean")
    return float(value) if value is not None else 0.0


def cost(summaries: list[dict], token: bool, *, use_sf: bool) -> float | None:
    vals = []
    for summary in summaries:
        block = summary.get("cost")
        if not block:
            continue
        timing = block.get("timing") or {}
        tokens = block.get("token_counts") or {}
        if token:
            value = (
                _nested_mean(tokens, "total_tokens") if use_sf else
                _nested_mean(tokens, "input_tokens") + _nested_mean(tokens, "initial_output_tokens")
            )
        else:
            value = (
                _nested_mean(timing, "example_seconds") if use_sf else
                _nested_mean(timing, "retrieval_seconds")
                + _nested_mean(timing, "rerank_seconds")
                + _nested_mean(timing, "few_shot_seconds")
                + _nested_mean(timing, "prompt_seconds")
                + _nested_mean(timing, "generation_seconds")
            )
        vals.append(value)
    return sum(vals) / len(vals) if vals else None


def collect_by_stem(stem: str, use_sf: bool) -> dict | None:
    summaries = [s for s in (load_summary(f"{stem}_seed{sd}", "") for sd in SEEDS) if s]
    if not summaries:
        return None
    row: dict = {"n": len(summaries)}
    for metric in RAW_QUALITY_FIELDS:
        row[metric] = mean_std(values(summaries, metric, use_sf=use_sf))

    mc_summaries = [load_summary(f"{stem}_seed{sd}", "_casimedicos") for sd in SEEDS]
    row["mc_accuracy"] = mean_std(values([s for s in mc_summaries if s], "mc_accuracy", use_sf=use_sf))

    row["sec"] = cost(summaries, token=False, use_sf=use_sf)
    row["tok"] = cost(summaries, token=True, use_sf=use_sf)

    per_seed_meanq = []
    mixed_summaries = [load_summary(f"{stem}_seed{sd}", "") for sd in SEEDS]
    for i in range(len(SEEDS)):
        rouge = value_or_none(mixed_summaries[i], "rouge_l_f1", use_sf=use_sf)
        bert = value_or_none(mixed_summaries[i], "bertscore_f1", use_sf=use_sf)
        mc_seed = value_or_none(mc_summaries[i], "mc_accuracy", use_sf=use_sf)
        parts = [v for v in (rouge, bert, mc_seed) if v is not None]
        if parts:
            per_seed_meanq.append(sum(parts) / len(parts))
    row["meanq"] = mean_std(per_seed_meanq)
    return row


def _cost_one(summary: dict, token: bool, *, use_sf: bool) -> float | None:
    return cost([summary], token=token, use_sf=use_sf)


def fmt_delta(mean: float | None, std: float | None) -> str:
    """Format a paired difference with an explicit leading sign."""
    if mean is None:
        return "---"
    return f"{mean:+.2f}{{\\tiny$\\pm${std:.2f}}}" if std else f"{mean:+.2f}"


def collect_delta(rag_stem: str, rag_sf: bool, base_stem: str, base_sf: bool) -> dict | None:
    """Paired seed-wise RAG-minus-baseline differences for the test table."""
    rag = [load_summary(f"{rag_stem}_seed{sd}", "") for sd in SEEDS]
    base = [load_summary(f"{base_stem}_seed{sd}", "") for sd in SEEDS]
    rag_mc = [load_summary(f"{rag_stem}_seed{sd}", "_casimedicos") for sd in SEEDS]
    base_mc = [load_summary(f"{base_stem}_seed{sd}", "_casimedicos") for sd in SEEDS]
    if not all(rag) or not all(base) or not all(rag_mc) or not all(base_mc):
        return None

    row: dict = {"n": len(SEEDS)}
    for metric in RAW_QUALITY_FIELDS:
        diffs = [
            value_or_none(r, metric, use_sf=rag_sf) - value_or_none(b, metric, use_sf=base_sf)
            for r, b in zip(rag, base)
        ]
        row[metric] = mean_std(diffs)
    mc_diffs = [
        value_or_none(r, "mc_accuracy", use_sf=rag_sf) - value_or_none(b, "mc_accuracy", use_sf=base_sf)
        for r, b in zip(rag_mc, base_mc)
    ]
    row["mc_accuracy"] = mean_std(mc_diffs)
    meanq_diffs = []
    for r, b, rmc, bmc in zip(rag, base, rag_mc, base_mc):
        rag_parts = [value_or_none(r, m, use_sf=rag_sf) for m in ("rouge_l_f1", "bertscore_f1")] + [value_or_none(rmc, "mc_accuracy", use_sf=rag_sf)]
        base_parts = [value_or_none(b, m, use_sf=base_sf) for m in ("rouge_l_f1", "bertscore_f1")] + [value_or_none(bmc, "mc_accuracy", use_sf=base_sf)]
        meanq_diffs.append(sum(rag_parts) / 3 - sum(base_parts) / 3)
    row["meanq"] = mean_std(meanq_diffs)
    for key, token in (("sec", False), ("tok", True)):
        diffs = [_cost_one(r, token, use_sf=rag_sf) - _cost_one(b, token, use_sf=base_sf) for r, b in zip(rag, base)]
        row[key] = sum(diffs) / len(diffs)
    return row


def main() -> None:
    gathered = []
    source_rows = {}
    for row_id, model_label, config_label, stem, use_sf in ROWS:
        row = collect_by_stem(stem, use_sf)
        if row is None:
            print(f"WARNING: no metrics for {stem}", file=sys.stderr)
            continue
        source_rows[row_id] = (model_label, config_label, stem, use_sf, row)
        gathered.append((row_id, model_label, config_label, use_sf, row, False))
        baseline_id = DELTA_PAIRS.get(row_id)
        if baseline_id:
            _, _, baseline_stem, baseline_sf, _ = source_rows[baseline_id]
            delta = collect_delta(stem, use_sf, baseline_stem, baseline_sf)
            if delta is not None:
                gathered.append((rf"$\Delta${row_id[-1]}", model_label, "", False, delta, True))

    best_meanq = max((g[4]["meanq"][0] for g in gathered if not g[5] and g[4]["meanq"][0] is not None), default=None)
    best_of_col: dict[str, float] = {}
    for metric, _ in QUALITY:
        if metric == "meanq":
            continue
        vals = [g[4][metric][0] for g in gathered if not g[5] and g[4][metric][0] is not None]
        if vals:
            best_of_col[metric] = max(vals)

    # The blue row is the better tested system within each model's direct
    # LLM-only-versus-RAG comparison, determined by MeanQ (not hardcoded).
    winner_by_model: dict[str, str] = {}
    for row_id, model_label, _, _, row, is_delta in gathered:
        if is_delta or row["meanq"][0] is None:
            continue
        previous_id = winner_by_model.get(model_label)
        if previous_id is None:
            winner_by_model[model_label] = row_id
            continue
        previous = next(g for g in gathered if g[0] == previous_id)
        if row["meanq"][0] > previous[4]["meanq"][0]:
            winner_by_model[model_label] = row_id

    lines: list[str] = []
    lines.append(r"\begin{scriptsize}")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\setlength{\LTcapwidth}{\linewidth}")
    lines.append(r"\begin{longtable}{r l >{\raggedright\arraybackslash}p{3.8cm} c c c c c c c}")
    lines.append(
        r"\caption[Test results]{"
        r"Held-out test-set results (250-example mixed test set, "
        r"\autoref{sec:datasets-top}), comparing each model's LLM-only baseline "
        r"with its own best dev-set RAG configuration (\autoref{sec:results}), "
        r"held frozen; no hyperparameter differs from the corresponding dev run "
        r"except the input file. The higher-MeanQ configuration per model is "
        r"blue-highlighted; paired $\Delta$ MeanQ values are in bold.} "
        r"\label{tab:test-results} \\"
    )
    header = (
        r"\# & Model & Config & SF & \multicolumn{4}{c}{Quality $\uparrow$} & "
        r"\multicolumn{2}{c}{Cost $\downarrow$} \\"
        "\n" + r"\cmidrule(lr){5-8}\cmidrule(lr){9-10}"
        "\n" + r" &  &  &  & ROUGE-L & BERT-F1 & MC-acc & MeanQ & sec & tok \\"
    )
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{10}{r}{\textit{continued on next page}} \\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    for item_index, (row_id, model_label, config_label, use_sf, row, is_delta) in enumerate(gathered):
        cells = [row_id, esc(model_label), config_label, r"\checkmark" if use_sf else ""]
        for metric, _ in QUALITY:
            mean, std = row[metric]
            cell = fmt_delta(mean, std) if is_delta else fmt(mean, std)
            if is_delta and metric == "meanq":
                cell = r"\textbf{%s}" % cell
            cells.append(cell)
        cells += [
            f"{row['sec']:+.2f}" if is_delta and row["sec"] is not None else (f"{row['sec']:.2f}" if row["sec"] is not None else "---"),
            f"{row['tok']:+.0f}" if is_delta and row["tok"] is not None else (f"{row['tok']:.0f}" if row["tok"] is not None else "---"),
        ]
        row_prefix = r"\rowcolor{pinnedrow}" if winner_by_model.get(model_label) == row_id else ""
        lines.append(row_prefix + " & ".join(cells) + r" \\")
        if is_delta and item_index < len(gathered) - 1:
            lines.append(r"\midrule")

    lines.append(r"\end{longtable}")
    lines.append(r"\end{scriptsize}")

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
