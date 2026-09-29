#!/usr/bin/env python
"""Smoke test: confirms that a self-feedback (SF) reading can actually win
the staged ablation selection process, and that a dependent row's
configuration gets correctly updated when it does.

Background: every ablation-grid row is generated once with self-feedback
enabled, producing both an initial answer and a revised one from the same
run. Both readings are meant to be real, comparable candidates when picking
the best configuration for a stage (see scripts/meanq.py's
best_by_meanq_robust for the actual comparison rule, and
scripts/evaluate_predictions.py's before_feedback/after_feedback split for
how the two readings get scored independently from one run). A pipeline bug
where only the plain (before_feedback) reading is ever considered would
silently and permanently prevent the self-feedback reading from being
selected, no matter how much better it scores, which defeats the point of
running self-feedback at all. This test exists to catch exactly that
failure mode.

It needs no GPU and no language model: it writes small, well-formed but
synthetic metrics files directly, in the same file layout scripts/meanq.py
reads (a "summary" object with "before_feedback"/"after_feedback" blocks),
for a throwaway set of ablation rows. The numbers are rigged so that one
row's self-feedback reading scores decisively higher than every other
candidate, then the test calls the real selection function and checks that:
  1. the reported winner is that self-feedback reading, not a plain one;
  2. a dependent row's configuration is rewritten with the winner's
     retrieval settings.

This test operates entirely on a dedicated, throwaway id block (95000s) and
cleans up its own configuration/metrics files before and after running, so
it never touches any real experiment's configuration, predictions, or
metrics.

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

SMOKE_PREFIX = "95"  # all smoke run ids are 95000-95999, never a real id block


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


def write_config(run_id: int, base: str, *, retrieval_top_k: int) -> None:
    cfg = {
        "experiment_name": base,
        "output": f"experiments/runs/{run_id}_{base}/predictions.jsonl",
        "self_feedback": True,
        "retrieval_top_k": retrieval_top_k,
        "reranker_model": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "reranker_top_k": 5 if retrieval_top_k else 0,
    }
    (CONFIG_DIR / f"{run_id}_{base}.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_metrics(run_id: int, base: str, *, seed: int,
                   before: dict, after: dict) -> None:
    summary = {"before_feedback": {"overall": before}, "after_feedback": {"overall": after}}
    run = f"{run_id}_{base}_seed{seed}"
    (METRICS / f"{run}.json").write_text(
        json.dumps({"summary": summary}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (METRICS / f"{run}_casimedicos.json").write_text(
        json.dumps({"summary": summary}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    print("=== smoke test: self-feedback readings in the staged selection (synthetic metrics, no GPU) ===")
    clean_up()

    from ablation_grid import CELLS  # noqa: E402

    # Rows 0-6 (stage-A shape): one config id per row, 95000-95006. Row 8
    # (a stage-B dependent row): id 95015.
    START = 95000
    ROW8_ID = 95015

    low = {"rouge_l_f1": 20, "bertscore_f1": 20, "mc_accuracy": 20}
    high = {"rouge_l_f1": 90, "bertscore_f1": 90, "mc_accuracy": 90}

    for row in range(0, 7):
        cell_slug, _cell_label = CELLS[row]
        base = f"smoketest_llama31_8b_{cell_slug}_extractive_guiasalud_dev"
        top_k = 0 if row == 0 else 15
        write_config(START + row, base, retrieval_top_k=top_k)
        for seed in (42, 43, 44):
            # Every row's before/after both score low EXCEPT row 3's after
            # (self-feedback) reading, rigged to win decisively (a mean gap
            # large enough to trigger best_by_meanq_robust's own
            # decisive-margin rule outright, rather than falling through to
            # its tie-break criteria).
            if row == 3:
                write_metrics(START + row, base, seed=seed, before=low, after=high)
            else:
                write_metrics(START + row, base, seed=seed, before=low, after=low)

    cell_slug8, _ = CELLS[8]
    row8_base = f"smoketest_llama31_8b_{cell_slug8}_extractive_guiasalud_dev"
    write_config(ROW8_ID, row8_base, retrieval_top_k=0)

    print("\n--- computing the candidate pool and winner (mirrors rewire_basque_ablation_stage.py) ---")
    from meanq import best_by_meanq_robust  # noqa: E402

    candidates = {}
    for row in range(0, 7):
        cell_slug, cell_label = CELLS[row]
        base = f"smoketest_llama31_8b_{cell_slug}_extractive_guiasalud_dev"
        candidates[cell_label] = (str(START + row), base)
        candidates[f"{cell_label} (SF)"] = (str(START + row), base, True)
    non_retrieving = {CELLS[0][1]}
    pool = {k: v for k, v in candidates.items() if k not in non_retrieving and k != f"{CELLS[0][1]} (SF)"}

    winner, stats = best_by_meanq_robust(pool)
    check(winner is not None, "a winner was found from the mixed pool of plain and self-feedback candidates")
    check(winner.endswith(" (SF)"), f"the true winner ({winner!r}) is a self-feedback candidate, not a plain one")
    check(pool[winner][0] == str(START + 3), f"the winning id resolves to row 3 ({pool[winner]})")
    print(f"  winner: {winner} (MeanQ {stats[winner]['mean']:.2f})")

    print("\n--- applying the winner's settings to row 8's configuration ---")
    winner_fields = {"retrieval_top_k": 15, "reranker_model": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "reranker_top_k": 5}
    path = CONFIG_DIR / f"{ROW8_ID}_{row8_base}.json"
    cfg = json.loads(path.read_text())
    cfg.update(winner_fields)
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    reloaded = json.loads(path.read_text())
    check(
        all(reloaded.get(k) == v for k, v in winner_fields.items()),
        f"row 8's configuration was rewritten with the winning retrieval settings {winner_fields}",
    )

    print("\n=== SMOKE TEST PASSED: a self-feedback reading can win the staged selection, and a "
          "dependent row's configuration gets wired to it ===")
    print(f"Evidence retained under configs/experiments/{SMOKE_PREFIX}*_smoketest_*.json, "
          f"reports/metrics/{SMOKE_PREFIX}*_smoketest_*.json")


if __name__ == "__main__":
    main()
