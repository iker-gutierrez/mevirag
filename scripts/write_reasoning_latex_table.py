#!/usr/bin/env python
"""LaTeX tables for the reasoning pipelines, in the same format as the ablation
tables (scripts/write_result_tables.py): longtable, a two-level header (Quality
spanning ROUGE-L/BERT-F1/MC-acc/MeanQ, Cost spanning sec/tok/calls), and the
best value per metric column in bold.

Each pipeline is compared against the single-pass RAG baseline it was built on,
with retrieval held fixed, so any difference is attributable to the reasoning
procedure rather than to a change in the evidence supplied.

The baseline shows only its better-MeanQ SF state (noSF wins for both languages
currently), not both, matching the ablation tables' reference-row convention,
so there is no per-row SF split left to show, and no SF column: the baseline's
own row label ("<model>, <config>") names the frozen configuration directly
rather than requiring the caption or a footnote to spell it out.

`calls` (mean LLM generations per answer) is always a real column, in both
languages, not folded into a footnote: a three-round agentic loop and a
single-pass baseline both emit exactly one final answer, so cost reported
without it would make them look equally expensive.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[1]
METRICS = REPO / "reports" / "metrics"
RUNS = REPO / "experiments" / "runs"
OUT = REPO / "manuscript"
SEEDS = [42, 43, 44]

QUALITY = [
    ("rouge_l_f1", "ROUGE-L"),
    ("bertscore_f1", "BERT-F1"),
    ("mc_accuracy", "MC-acc"),
    ("meanq", "MeanQ"),
]

# (display label, run stem, has self-feedback). ES uses Qwen3.5-9B no-think +
# rerank5 (think mode's MeanQ edge was 0.01, inside noise, at ~3x the cost). EU
# uses Latxa + retrieve top-3, Latxa's own ablation winner on the 2026-07-21
# staged rerun (rebuilt CasiMedicos-Exp/SNS-1064 splits + rebuilt retrieval
# indices, best_by_meanq_robust's variance/cost-aware pick: see
# scripts/write_result_tables.py's FORCED_REFERENCES comment for the full
# numbers and the cost-tiebreak rationale), and Llama's reasoning-pipeline base
# is ALSO retrieve top-3 on this rerun (Llama's own outright MeanQ winner, no
# tiebreak needed, same comment). Both EU models now share one retrieval
# depth, so the EU pipeline rows (all built on Llama configs 1500-1503) are
# directly comparable to the single Latxa-based baseline row on retrieval
# depth, even though the baseline itself is only run once, on Latxa.
#
# The baseline's row label names its model + config directly and compactly
# ("<model>, <config>") rather than a generic "Single-pass RAG (baseline)",
# a longer, more explicit inline label ("...(baseline: Qwen3.5-9B (no-think),
# rerank top 5)") was tried first and overflowed the page width, clipping the
# rightmost column. This compact form fits within the same column width the
# other rows use. The row's own number (carried forward from its ablation
# table, e.g. "6a") plus the dashed rule separating it from the pipeline rows
# already mark it as the frozen reference, so the label no longer needs to say
# "Single-pass RAG" either.
ES_BASELINE_DESC = "Qwen no-think, rerank top 5"
EU_BASELINE_DESC = "Latxa, e5 top 1"
ES_ROWS = [
    (ES_BASELINE_DESC, "1134_qwen35_9b_rag_e5_rerank5_no_think_extractive_mixed_dev", True, ("6a", "6a'")),
    # Structured CoT is reported in two retrieval variants: with our own frozen
    # best RAG config (same evidence as the baseline and the other three
    # pipelines, isolates the effect of the four-stage causal reasoning
    # alone), and with MedCoT-RAG's own causal-aware retrieval scoring
    # (sec:reasoning-pipelines, src/medical_rag_thesis/causal_scoring.py),
    # faithful to the full original method, retrieval and generation together.
    ("MedCoT-RAG (our best retrieval)", "1530_qwen35_9b_no_think_structured_cot_meanq_best_extractive_mixed_dev", False, None),
    ("MedCoT-RAG (their top-5 retrieval)", "1341_qwen35_9b_structured_cot_causal_no_think_extractive_mixed_dev", False, None),
    # thought_rag (RAR2 Parallel Scaling) and marag (multi-query retrieval
    # agent, confidence-sorted full-candidate history) were revised for closer
    # faithfulness to their sources, see src/medical_rag_thesis/reasoning.py's
    # module docstring. Configs 1601/1600 are the current, in-code pipelines,
    # the previous configs (1531/1533) used superseded pre-revision logic and
    # are not part of the publication-facing repository.
    ("RAR$^2$ (parallel scaling)", "1601_qwen35_9b_no_think_thought_rag_meanq_best_extractive_mixed_dev", False, None),
    ("RAR$^2$ (iterative scaling)", "1532_qwen35_9b_no_think_thought_rag_iter_meanq_best_extractive_mixed_dev", False, None),
    ("MA-RAG", "1600_qwen35_9b_no_think_marag_meanq_best_extractive_mixed_dev", False, None),
]

# Qwen think's own reasoning-pipeline family: same rerank-top-5 base as
# ES_ROWS above (its own stage-A winner too, see FORCED_REFERENCES["ES"] in
# write_result_tables.py), reported as its own table (tab:reasoning-es-think)
# rather than appended into ES_ROWS, for the same reason Llama's EU rows are
# separate from Latxa's: each table's baseline row and per-column bolding are
# computed within that table alone.
#
# The MedCoT-RAG (causal-scoring) row (config 1340) was regenerated against
# the full-corpus retrieval index (2026-07-21 rebuild, the July 19 predictions
# were stale) and evaluated with evaluate_predictions_by_source.py + the
# patch_mc_accuracy.py follow-up, matching every other mixed-dev row's
# convention, it was previously missing from this table because its
# original evaluation never produced the _casimedicos split MC-acc/MeanQ
# need, not because the run didn't exist.
ES_THINK_ROWS = [
    # Row 1 is the TRUE base the four pipeline rows below are actually built
    # on: config 1280 (3-shot + rerank top 5, MeanQ 73.51, row 8's winner),
    # not plain rerank top 5 (1276, 72.13/72.24), configs 1554-1557 were
    # explicitly chained onto 1280 (see their own rag_base_source field), so
    # showing 1276 here as "the frozen baseline" was comparing the pipelines
    # against a config they were never built on top of, understating the gap.
    ("Qwen think, 3-shot + rerank top 5", "1280_qwen35_9b_rag_3shot_e5_rerank5_think_extractive_mixed_dev", True, ("8b", "8b'")),
    ("MedCoT-RAG (our best retrieval)", "1554_qwen35_9b_think_structured_cot_fewshot_rerank5_extractive_mixed_dev", False, None),
    ("MedCoT-RAG (their top-5 retrieval)", "1340_qwen35_9b_structured_cot_causal_extractive_mixed_dev", False, None),
    ("RAR$^2$ (parallel scaling)", "1611_qwen35_9b_think_thought_rag_meanq_best_extractive_mixed_dev", False, None),
    ("RAR$^2$ (iterative scaling)", "1556_qwen35_9b_think_thought_rag_iter_fewshot_rerank5_extractive_mixed_dev", False, None),
    ("MA-RAG", "1610_qwen35_9b_think_marag_meanq_best_extractive_mixed_dev", False, None),
]

# Current Basque 11000-series chain.  The baseline rows are exactly the two
# configurations selected by the staged decision rule: Latxa E5 top-1 (the
# initial/noSF reading, 1d) and Llama rerank top-5 (the SF reading, 6c').
# The pipelines below are the completed 13000/13200-series runs written by
# finalize_basque_ablation_and_write_reasoning_configs.py.
EU_ROWS = [
    (EU_BASELINE_DESC, "11012_latxa_llama31_8b_rag_e5_topk1_extractive_guiasalud_dev", True, ("1d", "1d'")),
    ("MedCoT-RAG (our best retrieval)", "13200_latxa_llama31_8b_structured_cot_e5_topk1_extractive_guiasalud_dev", False, None),
    ("MedCoT-RAG (their top-5 retrieval)", "13204_latxa_llama31_8b_structured_cot_causal_extractive_guiasalud_dev", False, None),
    ("RAR$^2$ (parallel scaling)", "13201_latxa_llama31_8b_thought_rag_e5_topk1_extractive_guiasalud_dev", False, None),
    ("RAR$^2$ (iterative scaling)", "13202_latxa_llama31_8b_thought_rag_iter_e5_topk1_extractive_guiasalud_dev", False, None),
    ("MA-RAG", "13203_latxa_llama31_8b_marag_e5_topk1_extractive_guiasalud_dev", False, None),
]

# Llama's own reasoning-pipeline family: same retrieve-top-3 base and no
# self-feedback as EU_ROWS above (its stage-A winner too, see
# FORCED_REFERENCES["EU"] in write_result_tables.py), reported as its own
# table (tab:reasoning-eu-llama) rather than appended into EU_ROWS, since the
# table's baseline/highlight logic assumes one frozen reference row per table.
EU_LLAMA_ROWS = [
    ("Llama, rerank top 3", "11005_llama31_8b_rag_e5_rerank3_extractive_guiasalud_dev", True, ("5c", "5c'")),
    ("MedCoT-RAG (our best retrieval)", "16200_llama31_8b_structured_cot_e5_rerank3_extractive_guiasalud_dev_costaware", False, None),
    ("MedCoT-RAG (their top-5 retrieval)", "16204_llama31_8b_structured_cot_causal_extractive_guiasalud_dev_costaware", False, None),
    ("RAR$^2$ (parallel scaling)", "16201_llama31_8b_thought_rag_e5_rerank3_extractive_guiasalud_dev_costaware", False, None),
    ("RAR$^2$ (iterative scaling)", "16202_llama31_8b_thought_rag_iter_e5_rerank3_extractive_guiasalud_dev_costaware", False, None),
    ("MA-RAG", "16203_llama31_8b_marag_e5_rerank3_extractive_guiasalud_dev_costaware", False, None),
]


def summaries_for(stem: str, suffix: str) -> list[dict[str, Any]]:
    out = []
    for seed in SEEDS:
        path = METRICS / f"{stem}_seed{seed}{suffix}.json"
        if path.exists():
            out.append(json.loads(path.read_text()).get("summary", {}))
    return out


def mean_std(vals: list[float]) -> tuple[Optional[float], Optional[float]]:
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var)


def _value(summary: dict, section: str, name: str, use_sf: bool) -> Optional[float]:
    block = "after_feedback" if use_sf else "before_feedback"
    value = (summary.get(block) or {}).get(section, {}).get(name)
    if value is None and not use_sf:
        value = (summary.get(section) or {}).get(name)
    return float(value) if value is not None else None


def metric(summaries: list[dict], name: str, use_sf: bool) -> tuple[Optional[float], Optional[float]]:
    vals = [v for s in summaries if (v := _value(s, "overall", name, use_sf)) is not None]
    return mean_std(vals)


def meanq_per_seed(summaries: list[dict], mc_summaries: list[dict], use_sf: bool) -> tuple[Optional[float], Optional[float]]:
    """MeanQ per seed (pairing ROUGE-L/BERT-F1/MC-acc from the same seed by index --
    summaries and mc_summaries must be built from the same seed-ordered stem list),
    then averaged. Same method as scripts/metric_tables.py's meanq_per_seed, so this
    is directly comparable to the ablation decision tables and MeanQ column."""
    per_seed = []
    for i in range(len(summaries)):
        rouge = _value(summaries[i], "overall", "rouge_l_f1", use_sf)
        bert = _value(summaries[i], "overall", "bertscore_f1", use_sf)
        mc = _value(mc_summaries[i], "overall", "mc_accuracy", use_sf) if i < len(mc_summaries) else None
        parts = [v for v in (rouge, bert, mc) if v is not None]
        if parts:
            per_seed.append(sum(parts) / len(parts))
    return mean_std(per_seed)


def _nested_mean(block: dict, name: str) -> float:
    value = (block.get(name) or {}).get("mean")
    return float(value) if value is not None else 0.0


def cost(summaries: list[dict], token: bool, *, use_sf: bool) -> Optional[float]:
    """noSF and SF cost are not the same number: the raw metric JSON stores one
    pipeline-wide total (`example_seconds`, `total_tokens`) that already includes
    the self-feedback pass, that total IS the SF row's cost. The noSF row's cost
    is the pre-feedback components only, matching scripts/summarize_metrics.py's
    cost_rows() split (also mirrored in scripts/write_result_tables.py's cost()).
    """
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


def llm_calls(stem: str) -> Optional[float]:
    vals = []
    for seed in SEEDS:
        path = RUNS / f"{stem}_seed{seed}" / "predictions.meta.json"
        if path.exists():
            value = json.loads(path.read_text()).get("mean_llm_calls_per_record")
            if value is not None:
                vals.append(float(value))
    return sum(vals) / len(vals) if vals else None


def fmt(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None:
        return "---"
    return f"{mean:.2f}{{\\tiny$\\pm${std:.2f}}}" if std else f"{mean:.2f}"


def esc(text: str) -> str:
    return (str(text).replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#"))


# Labels highlighted with the same blue-bar treatment as the baseline row, per
# language, for a reason other than "this is the frozen config everything
# else is compared against" (that's is_baseline). Currently: "MedCoT-RAG (our
# best retrieval)" (config/row 2) in Spanish, the reasoning pipeline with the
# actual highest MeanQ on fresh Qwen no-think numbers (71.11, vs MedCoT-RAG's
# their-top-5-retrieval variant's 67.69 and the single-pass baseline's 69.15),
# worth calling out visually.
EXTRA_HIGHLIGHT: dict[str, set[str]] = {
    "ES": {"MedCoT-RAG (our best retrieval)"},
}

# Languages where the baseline row's blue bar is suppressed even though
# is_baseline is True, the row still keeps its dashed-rule separator and
# bolded MeanQ (it's still the frozen reference the other rows are compared
# against), just not the highlight, since the highlight in this table is
# reserved for calling out MedCoT-RAG's win rather than marking the reference.
SUPPRESS_BASELINE_HIGHLIGHT: set[str] = {"ES"}


def build(rows, lang: str, suffix: str, dev: str, *, label_slug: Optional[str] = None,
          caption_model: Optional[str] = None, model_letter: str = "") -> str:
    # MC-acc (and hence MeanQ) on the mixed table comes from the CasiMedicos
    # subset, matching scripts/meanq.py, undefined on an open-answer-only suffix.
    mc_suffix = "_casimedicos" if suffix == "" else (suffix if suffix == "_casimedicos" else None)

    gathered = []
    for label, stem, has_sf, source_row in rows:
        summaries = summaries_for(stem, suffix)
        if not summaries:
            continue
        mc_summaries = summaries_for(stem, mc_suffix) if mc_suffix else []
        sf_states = (False, True) if has_sf else (False,)
        rows_this_label = []
        for use_sf in sf_states:
            row = {m: metric(summaries, m, use_sf) for m, _ in QUALITY if m != "meanq"}
            row["mc_accuracy"] = metric(mc_summaries, "mc_accuracy", use_sf) if mc_summaries else (None, None)
            row["meanq"] = meanq_per_seed(summaries, mc_summaries, use_sf)
            rows_this_label.append({
                "label": label,
                "sf": use_sf,
                "is_baseline": has_sf,
                "quality": row,
                "sec": cost(summaries, False, use_sf=use_sf),
                "tok": cost(summaries, True, use_sf=use_sf),
                "calls": llm_calls(stem),
                # Row number the baseline carries over from its own ablation
                # table (e.g. "6a"), not a sequential index, source_row is
                # (nosf_label, sf_label), None for every non-baseline row,
                # which is numbered sequentially starting at 11 instead.
                "source_row": (source_row[1] if use_sf else source_row[0]) if source_row else None,
            })
        if has_sf and len(rows_this_label) == 2:
            # The baseline is the frozen RAG config this whole table holds
            # fixed, like a carried-forward reference row in the ablation
            # tables, only its better-MeanQ SF state is shown, not both. This
            # is also why there is no SF column: with every row (baseline
            # included) collapsed to a single state, and the reasoning
            # pipelines never having an SF pass at all, the column would be
            # blank everywhere it isn't simply redundant.
            nosf_mean = rows_this_label[0]["quality"]["meanq"][0]
            sf_mean = rows_this_label[1]["quality"]["meanq"][0]
            keep_sf = sf_mean is not None and (nosf_mean is None or sf_mean > nosf_mean)
            rows_this_label = [rows_this_label[1] if keep_sf else rows_this_label[0]]
        gathered.extend(rows_this_label)

    best = {}
    for m, _ in QUALITY:
        vals = [g["quality"][m][0] for g in gathered if g["quality"][m][0] is not None]
        if vals:
            best[m] = max(vals)

    # The baseline's own row id (e.g. "6a", "8b", "2c", "2d'"), gathered[0]
    # is always the baseline (every *_ROWS list's first entry has has_sf=True
    # and a source_row), used in the caption below so it names the actual row
    # printed in the table rather than a stale literal "row 1" left over from
    # before every table gained per-model row ids.
    baseline_row_id = gathered[0]["source_row"] if gathered and gathered[0]["is_baseline"] else "?"

    # 3 label columns (#, Pipeline, SF) + quality + cost (sec, tok, calls,
    # always shown: a three-round agentic loop and a single-pass baseline both
    # emit exactly one final answer, so cost without it would make them look
    # equally expensive and conceal an order-of-magnitude compute difference).
    # SF is a checkmark, same convention as the ablation tables
    # (write_result_tables.py): every row here already shows exactly one
    # state (the baseline shows only its better-MeanQ state, pipeline rows
    # never run self-feedback at all), so the column is a per-row indicator
    # of which state that is, not a filter, checked for the baseline when
    # SF won, blank on every pipeline row.
    n_cost = 3
    ncol = 3 + len(QUALITY) + n_cost
    colspec = r"r l c " + "c " * len(QUALITY) + "c " * n_cost
    header = (
        r"\# & Pipeline & SF & \multicolumn{%d}{c}{Quality $\uparrow$} & "
        r"\multicolumn{%d}{c}{Cost $\downarrow$} \\" % (len(QUALITY), n_cost)
        + "\n" + r"\cmidrule(lr){4-%d}\cmidrule(lr){%d-%d}" % (
            3 + len(QUALITY), 4 + len(QUALITY), ncol)
        + "\n" + r" &  &  & " + " & ".join(n for _, n in QUALITY) + r" & sec & tok & calls \\"
    )
    lines = [
        r"\begin{scriptsize}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\setlength{\LTcapwidth}{\linewidth}",
        r"\begin{longtable}{" + colspec + r"}",
        r"\caption[Reasoning pipelines (%s)]{Reasoning pipelines on the frozen best "
        r"RAG configuration (%s, %s dev), row %s. Retrieval is held fixed, so every "
        r"difference is attributable to the reasoning procedure. Best value per "
        r"metric column in bold.} \label{tab:reasoning-%s} \\" % (
            caption_model or lang, lang, dev, baseline_row_id,
            label_slug or lang.lower()),
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{%d}{r}{\textit{continued on next page}} \\" % ncol,
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    extra_highlight = EXTRA_HIGHLIGHT.get(lang, set())
    # The baseline row keeps the number it carries over from its own ablation
    # table (e.g. "6a"), so it is traceable back to that table rather than
    # colliding with another table's row 1. The actual reasoning-pipeline rows,
    # new to this table, not carried from anywhere, are numbered
    # sequentially starting at 11, so no reasoning-pipeline row number is ever
    # reused across the four reasoning tables or collides with an ablation
    # table's own row numbers (which run 0-10 there). Each also carries the
    # same model letter as the ablation tables (write_result_tables.py's
    # MODEL_LETTERS: a=Qwen no-think, b=Qwen think, c=Llama, d=Latxa), since
    # every row in a given reasoning table is that one model's own pipeline,
    # the baseline row already has its letter as part of source_row, so only
    # the newly-numbered pipeline rows need it appended here.
    next_pipeline_row = 11
    for g in gathered:
        if g["is_baseline"] and g["source_row"]:
            row_num = g["source_row"]
        else:
            row_num = f"{next_pipeline_row}{model_letter}"
            next_pipeline_row += 1
        # The baseline is the frozen RAG config every pipeline below it is
        # compared against, highlighted the same way a carried-forward
        # reference row is in the ablation tables (scripts/write_result_tables.py):
        # a translucent blue row background, with its MeanQ bolded too.
        baseline_highlighted = g["is_baseline"] and lang not in SUPPRESS_BASELINE_HIGHLIGHT
        is_highlighted = baseline_highlighted or g["label"] in extra_highlight
        row_prefix = r"\rowcolor{pinnedrow}" if is_highlighted else ""
        cells = [row_num, esc(g["label"]), r"\checkmark" if g["sf"] else ""]
        for m, _ in QUALITY:
            mean, std = g["quality"][m]
            cell = fmt(mean, std)
            is_col_best = mean is not None and m in best and mean == best[m]
            is_baseline_meanq = m == "meanq" and g["is_baseline"]
            if is_col_best or is_baseline_meanq:
                cell = r"\textbf{%s}" % cell
            cells.append(cell)
        cells += [
            f"{g['sec']:.2f}" if g["sec"] is not None else "---",
            f"{g['tok']:.0f}" if g["tok"] is not None else "---",
            f"{g['calls']:.1f}" if g["calls"] is not None else "1.0",
        ]
        lines.append(row_prefix + " & ".join(cells) + r" \\")
        # Dashed rule after the baseline row, splitting "the frozen config
        # everything else is compared against" from "the pipelines being
        # compared", same convention as the ablation tables' reference/
        # new-comparison split, with matching spacing on both sides.
        if g["is_baseline"]:
            lines.append(r"\addlinespace[4pt]")
            lines.append(r"\cdashline{1-%d}" % ncol)
            lines.append(r"\addlinespace[4pt]")

    lines += [
        r"\end{longtable}",
        r"\end{scriptsize}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    (OUT / "table_reasoning_es.tex").write_text(
        build(ES_ROWS, "ES", "", "mixed", caption_model="ES, Qwen no-think", model_letter="a")
    )
    (OUT / "table_reasoning_es_think.tex").write_text(
        build(ES_THINK_ROWS, "ES, Qwen think", "", "mixed", label_slug="es-think", model_letter="b")
    )
    (OUT / "table_reasoning_eu.tex").write_text(
        build(EU_ROWS, "EU", "", "mixed", caption_model="EU, Latxa", model_letter="d")
    )
    (OUT / "table_reasoning_eu_llama.tex").write_text(
        build(EU_LLAMA_ROWS, "EU, Llama", "", "mixed", label_slug="eu-llama", model_letter="c")
    )
    print("  wrote table_reasoning_es.tex, table_reasoning_es_think.tex, "
          "table_reasoning_eu.tex, table_reasoning_eu_llama.tex")


if __name__ == "__main__":
    main()
