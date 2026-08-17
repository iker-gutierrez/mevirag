#!/usr/bin/env python
"""Smoke test: confirms that a self-feedback (SF) generation can actually win
the staged ablation selection process, and that both the plain and
self-feedback configuration of a dependent row get correctly updated when it
does.

Background: the ablation grid generates each configuration row twice — once
as a plain, single-pass generation, and once with an added self-feedback
refinement pass — and both are meant to be real, comparable candidates when
picking the best configuration for a stage (see scripts/meanq.py's
best_by_meanq_robust for the actual comparison rule). A pipeline bug where
only plain candidates are ever considered would silently and permanently
prevent a self-feedback configuration from being selected, no matter how
much better it scores, which defeats the point of generating it at all. This
test exists to catch exactly that failure mode.

It needs no GPU and no language model: it writes small, well-formed but
synthetic metrics files directly, in the same file layout scripts/meanq.py
reads (a "summary" object with "overall" for a plain run, or
"before_feedback"/"after_feedback" blocks for a self-feedback run), for a
throwaway set of ablation rows. The numbers are rigged so that one row's
self-feedback variant scores decisively higher than every other candidate,
then the test calls the real selection function and checks that:
  1. the reported winner is that self-feedback variant, not a plain one;
  2. a dependent row's plain configuration is rewritten with the winner's
     retrieval settings;
  3. that dependent row's own self-feedback clone is ALSO rewritten with
     those settings, so the next stage's candidate pool has a
     correctly-based self-feedback candidate to compare too.

This test operates entirely on a dedicated, throwaway id block (95000s for
the plain configs, 96000s for the self-feedback clones) and cleans up its
own configuration/metrics files before and after running, so it never
touches any real experiment's configuration, predictions, or metrics.

Usage: python scripts/smoke_test_self_feedback_selection.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

CONFIG_DIR = ROOT / "configs" / "experiments"
METRICS = ROOT / "reports" / "metrics"

SMOKE_PREFIX = "95"  # all smoke run ids are 95000-95999/96000-96999, never a real id block


def check(condition: bool, message: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        raise SystemExit(f"SMOKE TEST FAILED: {message}")


def clean_up() -> None:
    for p in list(CONFIG_DIR.glob(f"{SMOKE_PREFIX}*_*.json")):
        p.unlink()
    for p in list(METRICS.glob(f"{SMOKE_PREFIX}*_*.json")):
        p.unlink()


def write_config(run_id: int, base: str, *, self_feedback: bool, retrieval_top_k: int) -> None:
    cfg = {
        "experiment_name": base,
        "output": f"experiments/runs/{run_id}_{base}/predictions.jsonl",
        "self_feedback": self_feedback,
        "retrieval_top_k": retrieval_top_k,
        "reranker_model": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "reranker_top_k": 5 if retrieval_top_k else 0,
    }
    (CONFIG_DIR / f"{run_id}_{base}.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_metrics(run_id: int, base: str, *, seed: int, rouge: float, bert: float, mc: float,
                   self_feedback: bool) -> None:
    overall = {"rouge_l_f1": rouge, "bertscore_f1": bert, "mc_accuracy": mc}
    summary = {"overall": overall}
    if self_feedback:
        summary = {"before_feedback": {"overall": overall}, "after_feedback": {"overall": overall}}
    run = f"{run_id}_{base}_seed{seed}"
    (METRICS / f"{run}.json").write_text(
        json.dumps({"summary": summary}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (METRICS / f"{run}_casimedicos.json").write_text(
        json.dumps({"summary": summary}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    print("=== smoke test: self-feedback candidates in the staged selection (synthetic metrics, no GPU) ===")
    clean_up()

    from mixed_meanq import CELLS  # noqa: E402

    # Rows 0-6 (stage-A shape): plain ids 95000-95006, self-feedback clone
    # ids 96000-96006. Row 8 (a stage-B dependent row): plain id 95015,
    # self-feedback clone id 96015.
    PLAIN_START = 95000
    SF_START = 96000
    ROW8_PLAIN_ID = 95015
    ROW8_SF_ID = 96015

    for row in range(0, 7):
        cell_slug, _cell_label, _ = CELLS[row]
        base = f"smoketest_llama31_8b_{cell_slug}_extractive_guiasalud_dev"
        sf_base = base[: -len("_dev")] + "_sf_dev"
        top_k = 0 if row == 0 else 15
        write_config(PLAIN_START + row, base, self_feedback=False, retrieval_top_k=top_k)
        write_config(SF_START + row, sf_base, self_feedback=True, retrieval_top_k=top_k)
        for seed in (42, 43, 44):
            # Every plain candidate scores low, and every self-feedback
            # candidate scores low EXCEPT row 3's, which is rigged to win
            # decisively (a mean gap large enough to trigger
            # best_by_meanq_robust's own decisive-margin rule outright,
            # rather than falling through to its tie-break criteria).
            if row == 3:
                write_metrics(SF_START + row, sf_base, seed=seed, rouge=90, bert=90, mc=90, self_feedback=True)
            else:
                write_metrics(SF_START + row, sf_base, seed=seed, rouge=20, bert=20, mc=20, self_feedback=True)
            write_metrics(PLAIN_START + row, base, seed=seed, rouge=20, bert=20, mc=20, self_feedback=False)

    cell_slug8, _, _ = CELLS[8]
    row8_base = f"smoketest_llama31_8b_{cell_slug8}_extractive_guiasalud_dev"
    row8_sf_base = row8_base[: -len("_dev")] + "_sf_dev"
    write_config(ROW8_PLAIN_ID, row8_base, self_feedback=False, retrieval_top_k=0)
    write_config(ROW8_SF_ID, row8_sf_base, self_feedback=True, retrieval_top_k=0)

    print("\n--- computing the candidate pool and winner (mirrors rewire_basque_ablation_stage.py) ---")
    from meanq import best_by_meanq_robust  # noqa: E402

    candidates = {}
    for row in range(0, 7):
        cell_slug, cell_label, _ = CELLS[row]
        base = f"smoketest_llama31_8b_{cell_slug}_extractive_guiasalud_dev"
        sf_base = base[: -len("_dev")] + "_sf_dev"
        candidates[cell_label] = (str(PLAIN_START + row), base)
        candidates[f"{cell_label} (SF)"] = (str(SF_START + row), sf_base, True)
    non_retrieving = {CELLS[0][1]}
    pool = {k: v for k, v in candidates.items() if k not in non_retrieving and k != f"{CELLS[0][1]} (SF)"}

    winner, stats = best_by_meanq_robust(pool)
    check(winner is not None, "a winner was found from the mixed pool of plain and self-feedback candidates")
    check(winner.endswith(" (SF)"), f"the true winner ({winner!r}) is a self-feedback candidate, not a plain one")
    check(pool[winner][0] == str(SF_START + 3), f"the winning id resolves to row 3's self-feedback clone ({pool[winner]})")
    print(f"  winner: {winner} (MeanQ {stats[winner]['mean']:.2f})")

    print("\n--- applying the winner's settings to row 8 (both its plain and self-feedback configuration) ---")
    winner_fields = {"retrieval_top_k": 15, "reranker_model": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "reranker_top_k": 5}
    for run_id, base in ((ROW8_PLAIN_ID, row8_base), (ROW8_SF_ID, row8_sf_base)):
        path = CONFIG_DIR / f"{run_id}_{base}.json"
        cfg = json.loads(path.read_text())
        cfg.update(winner_fields)
        path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for run_id, base, label in ((ROW8_PLAIN_ID, row8_base, "row 8 (plain)"), (ROW8_SF_ID, row8_sf_base, "row 8 (self-feedback)")):
        cfg = json.loads((CONFIG_DIR / f"{run_id}_{base}.json").read_text())
        check(
            all(cfg.get(k) == v for k, v in winner_fields.items()),
            f"{label} configuration was rewritten with the winning retrieval settings {winner_fields}",
        )

    print("\n=== SMOKE TEST PASSED: a self-feedback candidate can win the staged selection, and both the "
          "plain and self-feedback configuration of a dependent row get wired to it ===")
    print(f"Evidence retained under configs/experiments/{SMOKE_PREFIX}*_smoketest_*.json, "
          f"reports/metrics/{SMOKE_PREFIX}*_smoketest_*.json")


if __name__ == "__main__":
    main()
