#!/usr/bin/env python
"""MeanQ: the quality score used to choose the best RAG configuration.

MeanQ = mean(ROUGE-L, BERT-F1, MC-accuracy), each on its raw 0-100 scale.
It is computed per seed before the mean and standard deviation across seeds.

Rationale for the choice of metrics: ROUGE-L captures lexical fidelity, BERT-F1
captures semantic fidelity, and MC-accuracy captures decision correctness. Averaging
the three rewards configurations that perform well across all three axes rather than
on only one, which a single metric such as BERTScore can hide.

MC-accuracy is defined only for CasiMedicos-Exp (multiple-choice) records. On the
mixed dev set it is therefore read from the CasiMedicos subset (`_casimedicos.json`),
matching how the results tables report it. On an open-answer-only set (GuiaSalud)
MC-accuracy does not exist and MeanQ is the mean of the two overlap metrics.

This module is pure computation over the metric JSONs, it loads no models and needs
no GPU. It is the single source of truth for "which RAG config is best" and is used
both to report the decision and to wire the dependent configs (few-shot, domain
restriction) to the correct base.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "metrics"
SEEDS = [42, 43, 44]

QUALITY_METRICS = ("rouge_l_f1", "bertscore_f1", "mc_accuracy")


def _overall(summary: dict, use_sf: bool) -> dict:
    block = "after_feedback" if use_sf else "before_feedback"
    return (summary.get(block) or {}).get("overall") or summary.get("overall") or {}


def _load(run: str, suffix: str = "") -> Optional[dict]:
    p = METRICS / f"{run}{suffix}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("summary")


def metric_over_seeds(prefix: str, base: str, metric: str, *, suffix: str = "",
                      use_sf: bool = False) -> Optional[float]:
    vals = []
    for seed in SEEDS:
        summ = _load(f"{prefix}_{base}_seed{seed}", suffix)
        if not summ:
            continue
        v = _overall(summ, use_sf).get(metric)
        if v is not None:
            vals.append(float(v))
    return statistics.mean(vals) if vals else None


def metric_per_seed(prefix: str, base: str, metric: str, *, suffix: str = "",
                    use_sf: bool = False) -> list[Optional[float]]:
    """Per-seed values (None where a seed is missing the metric), seed order preserved.

    Unlike metric_over_seeds this keeps one entry per seed so callers can compute a
    std or pair the value with another metric from the same seed.
    """
    out: list[Optional[float]] = []
    for seed in SEEDS:
        summ = _load(f"{prefix}_{base}_seed{seed}", suffix)
        v = _overall(summ, use_sf).get(metric) if summ else None
        out.append(float(v) if v is not None else None)
    return out


def meanq_per_seed(prefix: str, base: str, *, use_sf: bool = False) -> list[Optional[float]]:
    """Per-seed MeanQ = mean(ROUGE-L, BERT-F1, MC-acc) within each seed.

    ROUGE-L/BERT-F1 come from the mixed metric file, MC-acc from that same seed's
    CasiMedicos subset (undefined on the open-answer half). A seed's MeanQ is the
    mean of whichever of its three components exist, None if the seed has none.
    Keeping it per-seed lets the summary/decision tables report a real ±std on MeanQ.
    """
    rouge = metric_per_seed(prefix, base, "rouge_l_f1", use_sf=use_sf)
    bert = metric_per_seed(prefix, base, "bertscore_f1", use_sf=use_sf)
    mc = metric_per_seed(prefix, base, "mc_accuracy", suffix="_casimedicos", use_sf=use_sf)
    out: list[Optional[float]] = []
    for r, b, m in zip(rouge, bert, mc):
        present = [v for v in (r, b, m) if v is not None]
        out.append(statistics.mean(present) if present else None)
    return out


def meanq(prefix: str, base: str, *, use_sf: bool = False) -> tuple[Optional[float], dict]:
    """MeanQ on the MIXED dev set. Returns (MeanQ, per-metric components).

    ROUGE-L and BERT-F1 come from the mixed metric file, MC-accuracy from the
    CasiMedicos subset (it is undefined on the open-answer half). MeanQ is the mean
    of whichever of the three are available.
    """
    rouge = metric_over_seeds(prefix, base, "rouge_l_f1", use_sf=use_sf)
    bert = metric_over_seeds(prefix, base, "bertscore_f1", use_sf=use_sf)
    mc = metric_over_seeds(prefix, base, "mc_accuracy", suffix="_casimedicos", use_sf=use_sf)
    components = {"rouge_l_f1": rouge, "bertscore_f1": bert, "mc_accuracy": mc}
    present = [v for v in components.values() if v is not None]
    return (statistics.mean(present) if present else None), components


def best_by_meanq(candidates: dict[str, tuple[str, str]], *, use_sf: bool = False
                  ) -> tuple[Optional[str], dict[str, float]]:
    """Given {label: (id_prefix, base)}, return (winning label, {label: MeanQ}).

    A configuration with no metrics yet is skipped, so this is safe to call before
    every run has been evaluated, it just picks the best of what exists.
    """
    scores: dict[str, float] = {}
    for label, (prefix, base) in candidates.items():
        mq, _ = meanq(prefix, base, use_sf=use_sf)
        if mq is not None:
            scores[label] = mq
    if not scores:
        return None, {}
    winner = max(scores, key=scores.get)
    return winner, scores


def token_cost(prefix: str, base: str, *, use_sf: bool = False) -> Optional[float]:
    """Mean LLM tokens consumed per answer across the available seeds.

    This is the cost quantity used only to break a near-tie in MeanQ.  It is
    deliberately the measured token total, not a hand-built retrieval proxy:
    self-feedback genuinely makes a second LLM call and must therefore cost
    more than its initial-answer reading.  Conversely, dense retrieval and
    CPU cross-encoder reranking do not themselves consume LLM tokens, so they
    are not converted into fictitious tokens here.  Their end-to-end runtime
    remains reported separately in the experiment tables.

    Metric summaries record the whole run's ``total_tokens`` for the revised
    (SF) reading.  The initial-answer reading is reconstructed from its input
    and initial-output components, matching the table writers.
    """
    values: list[float] = []
    for seed in SEEDS:
        summary = _load(f"{prefix}_{base}_seed{seed}")
        if not summary:
            continue
        tokens = ((summary.get("cost") or {}).get("token_counts") or {})

        def mean_of(name: str) -> Optional[float]:
            value = (tokens.get(name) or {}).get("mean")
            return float(value) if value is not None else None

        if use_sf:
            value = mean_of("total_tokens")
        else:
            input_tokens = mean_of("input_tokens")
            initial_output = mean_of("initial_output_tokens")
            value = (input_tokens + initial_output
                     if input_tokens is not None and initial_output is not None
                     else None)
        if value is not None:
            values.append(value)
    return statistics.mean(values) if values else None


def incomplete_candidates(
    candidates: "dict[str, tuple[str, str] | tuple[str, str, bool]]", *, use_sf: bool = False,
    required_seeds: int = len(SEEDS),
) -> "dict[str, int]":
    """{label: n_seeds_present} for every candidate with fewer than
    `required_seeds` seeds of metrics on disk, empty if the pool is complete.

    A stage-A/B/C generation array can finish with some tasks failed (GPU
    contention, a truncation hard-fail not yet retried, etc.); the
    evaluation script that writes metrics files skips any run with no
    predictions.jsonl rather than crashing (see slurm/*_ablation_evaluation_
    stage*.sh's own "Skipping ${run}: no predictions" line), and
    meanq_per_seed/metric_over_seeds silently drop missing seeds rather than
    erroring. best_by_meanq_robust therefore CAN select a winner from a
    candidate pool where one or more rows only have 1 or 2 of the normal 3
    seeds -- a real, silent risk: eval firing on a generation array with
    partial failures (necessary, since eval depends on the array via
    afterany, not afterok, or a single failed task in a 42-task array would
    permanently deadlock the whole chain) must not be allowed to quietly
    carry a weaker-sampled candidate into the stage-to-stage selection.

    Callers should call this immediately before best_by_meanq_robust on the
    SAME candidates dict and refuse to proceed (hard error, not a warning) if
    it returns anything non-empty, exactly the same "verified complete or
    fail loudly" principle already used for truncation
    (fail_on_remaining_truncation)."""
    incomplete = {}
    for label, spec in candidates.items():
        prefix, base = spec[0], spec[1]
        candidate_use_sf = spec[2] if len(spec) > 2 else use_sf
        per_seed = meanq_per_seed(prefix, base, use_sf=candidate_use_sf)
        n_present = sum(1 for v in per_seed if v is not None)
        if n_present < required_seeds:
            incomplete[label] = n_present
    return incomplete


def best_by_meanq_robust(
    candidates: "dict[str, tuple[str, str] | tuple[str, str, bool]]", *, use_sf: bool = False,
    margin: float = 0.5, std_threshold: float = 0.5, token_threshold: float = 1000.0,
) -> tuple[Optional[str], dict[str, dict]]:
    """Select a configuration with the MeanQ--Stability--Token rule.

    Reduces to the plain highest-mean-MeanQ winner whenever the leader's mean
    is at least `margin` points clear of every other candidate. Otherwise, the
    leader is compared pairwise against each candidate within `margin` points:
    standard deviation awards one point to whichever side has a std at least
    `std_threshold` MeanQ points lower (an ABSOLUTE difference, not a fraction
    of the larger std, a relative/percentage threshold makes it trivially
    easy to "meaningfully" beat a candidate whose own std is already small,
    since a tiny absolute gap can still clear a large percentage of a tiny
    denominator. ``std_threshold`` uses the same absolute scale as ``margin``.
    Cost awards its own point the
    same way whenever its measured LLM-token cost is at least
    `token_threshold` tokens/sample lower.  An absolute token threshold avoids
    treating a small difference as meaningful merely because a cheap baseline
    makes it a large percentage. A criterion that
    doesn't clear its own threshold awards no point to either side. Whichever
    side has more points after both criteria is preferred. If the two split
    one point each, or neither criterion is decisive, the pairwise winner
    falls back to whichever of the two has the higher mean MeanQ (even though,
    by construction, that difference is itself under `margin`), a real, if
    marginal, quality edge is never discarded once stability/cost are
    themselves inconclusive. The leader is replaced by the pairwise winner and
    the process repeats against the next candidate, so the final winner has
    beaten every other candidate under this rule.

    std_threshold=0.5 was checked against every candidate set this function is
    actually called on in this codebase (both ablation stages, all four
    current models) and changes no current winner relative to the old
    std_ratio=0.4 relative rule, across every threshold from 0.05 to 2.00
    swept in 0.05 steps. The std criterion is exercised (both candidates
    within `margin` of each other) in some of those sets, but the outcome is
    always settled by cost or the final mean tie-break either way.

    Each candidate value is normally (prefix, base), scored with the call-level
    `use_sf`. A candidate may instead be (prefix, base, candidate_use_sf) to
    override `use_sf` just for that one entry. This allows a row's noSF and SF
    variants to compete in the same candidate pool.

    Returns (winning label, {label: {"mean": ..., "std": ..., "cost": ..., "n": ...}}).
    """
    stats: dict[str, dict] = {}
    for label, spec in candidates.items():
        prefix, base = spec[0], spec[1]
        candidate_use_sf = spec[2] if len(spec) > 2 else use_sf
        per_seed = meanq_per_seed(prefix, base, use_sf=candidate_use_sf)
        values = [v for v in per_seed if v is not None]
        if not values:
            continue
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        stats[label] = {
            "mean": mean, "std": std, "n": len(values),
            "cost": token_cost(prefix, base, use_sf=candidate_use_sf),
        }
    if not stats:
        return None, {}

    def pairwise_winner(a: str, b: str) -> str:
        """Which of a, b wins under the point-scored rule, a vs b."""
        mean_a, mean_b = stats[a]["mean"], stats[b]["mean"]
        if abs(mean_a - mean_b) >= margin:
            return a if mean_a > mean_b else b

        std_a, std_b = stats[a]["std"], stats[b]["std"]
        cost_a, cost_b = stats[a]["cost"], stats[b]["cost"]
        points = {a: 0, b: 0}

        if std_b - std_a > std_threshold:
            points[a] += 1
        elif std_a - std_b > std_threshold:
            points[b] += 1

        if cost_a is not None and cost_b is not None:
            if cost_b - cost_a > token_threshold:
                points[a] += 1
            elif cost_a - cost_b > token_threshold:
                points[b] += 1

        if points[a] != points[b]:
            return a if points[a] > points[b] else b
        return a if mean_a >= mean_b else b

    labels = list(stats)
    winner = labels[0]
    for label in labels[1:]:
        winner = pairwise_winner(winner, label)
    return winner, stats


def decision_table(candidates: "dict[str, tuple[str, str] | tuple[str, str, bool]]", *, title: str,
                   use_sf: bool = False, top: Optional[int] = None) -> str:
    """Render a MeanQ-ranked markdown decision table for one model+language.

    `candidates` maps a display label -> (id_prefix, base), or (id_prefix,
    base, candidate_use_sf) to override `use_sf` for just that entry (see
    best_by_meanq_robust's own docstring -- same convention, used the same
    way by the staged selectors to list a row's noSF and SF variants
    side by side). Rows are sorted by MeanQ descending, the top row is bolded
    as the chosen configuration. This is the single source of the decision
    prose, so the .md reports never drift from what the staged ablation
    actually selected (see best_by_meanq).
    """
    rows = []
    for label, spec in candidates.items():
        prefix, base = spec[0], spec[1]
        candidate_use_sf = spec[2] if len(spec) > 2 else use_sf
        mq, comp = meanq(prefix, base, use_sf=candidate_use_sf)
        if mq is None:
            continue
        rows.append((label, mq, comp))
    rows.sort(key=lambda r: -r[1])
    if top is not None:
        rows = rows[:top]

    def cell(v: Optional[float]) -> str:
        return f"{v:.2f}" if v is not None else "--"

    out = [f"\n### {title}\n",
           "| Rank | Experiment | ROUGE-L | BERT-F1 | MC-acc | **MeanQ** |",
           "|---:|---|---:|---:|---:|---:|"]
    for i, (label, mq, comp) in enumerate(rows):
        r, b, m = comp["rouge_l_f1"], comp["bertscore_f1"], comp["mc_accuracy"]
        line = (f"| {i+1} | {label} | {cell(r)} | {cell(b)} | {cell(m)} | "
                f"{cell(mq)} |")
        if i == 0:
            line = (f"| **{i+1}** | **{label}** | {cell(r)} | {cell(b)} | "
                    f"{cell(m)} | **{cell(mq)}** |")
        out.append(line)
    if rows:
        best_label, best_mq, _ = rows[0]
        prediction_note = (
            "the no-self-feedback prediction" if not any(len(c) > 2 for c in candidates.values())
            else "either the no-self-feedback or self-feedback prediction, whichever scores "
            "higher for that row (candidates labeled \"(SF)\" are the self-feedback variant)"
        )
        out.append(
            f"\n**Chosen config: {best_label}** (MeanQ {best_mq:.2f}), the highest "
            f"MeanQ = mean(ROUGE-L, BERT-F1, MC-acc) over 3 seeds on {prediction_note}. "
            "MeanQ is used instead of any single metric so a config is only "
            "chosen if it does well on lexical, semantic, and decision correctness "
            "together.\n")
    return "\n".join(out)


if __name__ == "__main__":
    # Smoke: print MeanQ for the Llama retrieval configs.
    cand = {
        "retrieve top1": ("1041", "llama31_8b_rag_e5_topk1_extractive_mixed_eu_dev"),
        "retrieve top3": ("1042", "llama31_8b_rag_e5_topk3_extractive_mixed_eu_dev"),
        "retrieve top5": ("1043", "llama31_8b_rag_e5_topk5_extractive_mixed_eu_dev"),
        "rerank1": ("1044", "llama31_8b_rag_e5_rerank1_extractive_mixed_eu_dev"),
        "rerank3": ("1045", "llama31_8b_rag_e5_rerank3_extractive_mixed_eu_dev"),
        "rerank5": ("1046", "llama31_8b_rag_e5_rerank5_extractive_mixed_eu_dev"),
    }
    winner, scores = best_by_meanq(cand)
    for label, mq in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {label:10} MeanQ={mq:.2f}")
    print(f"  best: {winner}")
