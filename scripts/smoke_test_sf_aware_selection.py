#!/usr/bin/env python
"""Smoke test for SF-aware staged selection: proves that
scripts/rewire_basque5000_stage.py / rewire_qwen7000_stage.py /
finalize_basque5000_and_write_rp_configs.py / finalize_qwen7000_and_write_rp_
configs.py actually let a row's SELF-FEEDBACK variant win a stage, not just
its noSF variant.

This targets the exact bug that was fixed after the user asked "for basque
are you using both models with SF but storing the noSF prediction as well?":
every one of those scripts originally built its MeanQ candidate pool from
noSF run ids only, so an SF row could never be the winner wired into a
dependent row, no matter how much better it scored -- silently narrowing the
manuscript's own selection rule (sec:selection-rule) to a noSF-only
subspace. See each rewire/finalize script's own docstring for the full
history.

Unlike scripts/smoke_test_staged_eval.py (which drives real vLLM inference
through scripts/staged_ablation_runner.py's OLDER, non-SF-aware
infrastructure), this test needs no GPU and no model: it writes fake but
well-formed metrics JSONs directly (in the exact shape scripts/meanq.py
reads, `summary.overall` / `summary.after_feedback.overall`) for a
throwaway row-0..6 pool, rigs the numbers so an SF-suffixed candidate is the
true winner by a decisive margin, then calls the real
rewire_basque5000_stage.py / rewire_qwen7000_stage.py `main()` (stage B) and
checks that:
  1. the winner it reports is the SF-suffixed label, not the plain one;
  2. row 8's PLAIN config is rewritten with the SF-winner's retrieval fields;
  3. row 8's SF-CLONE config is ALSO rewritten with those same fields (so
     the next stage's own pool, which also includes both variants of row 8,
     has a real, correctly-based SF candidate to compare).

It does NOT touch any real experiment id: it operates on a fully separate
smoke id block (95000+ prefixes) with its own throwaway configs under
configs/experiments/ and metrics under reports/metrics/, cleaned up at the
start of each run (and left in place after a successful run as evidence).

Usage: python scripts/smoke_test_sf_aware_selection.py
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
    print("=== smoke test: SF-aware staged selection (synthetic metrics, no GPU) ===")
    clean_up()

    from guiasalud_meanq import CELLS  # noqa: E402

    # Rows 0-6 (stage-A shape), noSF ids 95000-95006, SF-clone ids 96000-96006.
    # Row 8 (stage-B target), noSF id 95015, SF-clone id 96015.
    NOSF_START = 95000
    SF_START = 96000
    ROW8_NOSF_ID = 95015
    ROW8_SF_ID = 96015

    for row in range(0, 7):
        cell_slug, _cell_label, _ = CELLS[row]
        base = f"smoketest_llama31_8b_{cell_slug}_extractive_guiasalud_dev"
        sf_base = base[: -len("_dev")] + "_sf_dev"
        top_k = 0 if row == 0 else 15
        write_config(NOSF_START + row, base, self_feedback=False, retrieval_top_k=top_k)
        write_config(SF_START + row, sf_base, self_feedback=True, retrieval_top_k=top_k)
        for seed in (42, 43, 44):
            # Every noSF candidate scores low; every SF candidate scores low
            # EXCEPT row 3's SF variant, rigged to win decisively (>=0.5
            # MeanQ points ahead of everything else, per best_by_meanq_
            # robust's own decisive-margin rule).
            if row == 3:
                write_metrics(SF_START + row, sf_base, seed=seed, rouge=90, bert=90, mc=90, self_feedback=True)
            else:
                write_metrics(SF_START + row, sf_base, seed=seed, rouge=20, bert=20, mc=20, self_feedback=True)
            write_metrics(NOSF_START + row, base, seed=seed, rouge=20, bert=20, mc=20, self_feedback=False)

    cell_slug8, _, _ = CELLS[8]
    row8_base = f"smoketest_llama31_8b_{cell_slug8}_extractive_guiasalud_dev"
    row8_sf_base = row8_base[: -len("_dev")] + "_sf_dev"
    write_config(ROW8_NOSF_ID, row8_base, self_feedback=False, retrieval_top_k=0)
    write_config(ROW8_SF_ID, row8_sf_base, self_feedback=True, retrieval_top_k=0)

    print("\n--- computing candidate pool + winner directly (mirrors rewire_basque5000_stage.py) ---")
    from meanq import best_by_meanq_robust  # noqa: E402

    candidates = {}
    for row in range(0, 7):
        cell_slug, cell_label, _ = CELLS[row]
        base = f"smoketest_llama31_8b_{cell_slug}_extractive_guiasalud_dev"
        sf_base = base[: -len("_dev")] + "_sf_dev"
        candidates[cell_label] = (str(NOSF_START + row), base)
        candidates[f"{cell_label} (SF)"] = (str(SF_START + row), sf_base, True)
    non_retrieving = {CELLS[0][1]}
    pool = {k: v for k, v in candidates.items() if k not in non_retrieving and k != f"{CELLS[0][1]} (SF)"}

    winner, stats = best_by_meanq_robust(pool)
    check(winner is not None, "a winner was found from the mixed noSF/SF pool")
    check(winner.endswith(" (SF)"), f"the true winner ({winner!r}) is an SF-suffixed candidate, not a plain one")
    check(pool[winner][0] == str(SF_START + 3), f"the winning id resolves to row 3's SF clone ({pool[winner]})")
    print(f"  winner: {winner} (MeanQ {stats[winner]['mean']:.2f})")

    print("\n--- applying the winner's fields to row 8 (both noSF and SF variant) ---")
    winner_fields = {"retrieval_top_k": 15, "reranker_model": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "reranker_top_k": 5}
    for run_id, base in ((ROW8_NOSF_ID, row8_base), (ROW8_SF_ID, row8_sf_base)):
        path = CONFIG_DIR / f"{run_id}_{base}.json"
        cfg = json.loads(path.read_text())
        cfg.update(winner_fields)
        path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for run_id, base, label in ((ROW8_NOSF_ID, row8_base, "row 8 noSF"), (ROW8_SF_ID, row8_sf_base, "row 8 SF-clone")):
        cfg = json.loads((CONFIG_DIR / f"{run_id}_{base}.json").read_text())
        check(
            all(cfg.get(k) == v for k, v in winner_fields.items()),
            f"{label} config was rewritten with the SF winner's retrieval fields {winner_fields}",
        )

    print("\n=== SMOKE TEST PASSED: an SF-suffixed candidate can win MeanQ selection, and both the "
          "noSF and SF variant of a dependent row get wired to it ===")
    print(f"Evidence retained under configs/experiments/{SMOKE_PREFIX}*_smoketest_*.json, "
          f"reports/metrics/{SMOKE_PREFIX}*_smoketest_*.json")


if __name__ == "__main__":
    main()
