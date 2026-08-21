#!/usr/bin/env python
"""Aggregate the per-run gold-document hit-rate@k sidecar files
(experiments/runs/<run>/retrieval_hit_rate_log.json, written by
run_generation_experiment.py when a config sets log_gold_hit_rate=true)
into one summary table per ablation-grid row.

Hit-rate@k answers a narrower question than MeanQ: across the naive top-k+1
candidates fetched before self-retrieval exclusion (\\autoref{sec:retrieval}),
how often was the query's own gold document actually present? A high
hit-rate says the retrieval index would hand the model its own answer if
exclusion were ever disabled or buggy; it is not a measure of how good the
excluded, top-k passages the model actually sees are, so it never
substitutes for or contributes to the MeanQ-based selection rule
(scripts/meanq.py's best_by_meanq_robust) that picks each stage's winning
configuration. This script is intentionally standalone from that selection
code and never writes into reports/metrics/<run>.json, the files
meanq.py's own loaders read, so hit-rate@k cannot influence which
configuration wins a stage even by accident.

Usage:
    python scripts/aggregate_hit_rate.py \\
        --run-ids 11000_llama31_8b_no_rag_extractive_guiasalud_dev \\
                  11001_llama31_8b_rag_e5_topk1_extractive_guiasalud_dev \\
        --seeds 42 43 44 \\
        --output reports/metrics/hit_rate_summary_11000_stageA.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-ids", nargs="+", required=True,
                         help="Run id bases (without _seed<N>), one per ablation-grid row.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_hit_rate_log(run_id: str, seed: int) -> dict | None:
    path = RUNS / f"{run_id}_seed{seed}" / "retrieval_hit_rate_log.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()

    per_row = {}
    for run_id in args.run_ids:
        seed_hit_rates = []
        total_queries = 0
        total_present = 0
        for seed in args.seeds:
            log = load_hit_rate_log(run_id, seed)
            if log is None:
                continue
            if log["hit_rate"] is not None:
                seed_hit_rates.append(log["hit_rate"])
            total_queries += log["num_queries"]
            total_present += log["num_gold_present_in_naive_topk_plus_1"]

        if not seed_hit_rates:
            print(f"  {run_id}: no hit-rate log found for any seed (no-retrieval row, or log_gold_hit_rate unset)")
            continue

        mean_hit_rate = sum(seed_hit_rates) / len(seed_hit_rates)
        per_row[run_id] = {
            "n_seeds": len(seed_hit_rates),
            "mean_hit_rate": mean_hit_rate,
            "per_seed_hit_rate": seed_hit_rates,
            "total_queries_across_seeds": total_queries,
            "total_gold_present_across_seeds": total_present,
        }
        print(f"  {run_id}: hit-rate@k = {mean_hit_rate:.1%} (mean over {len(seed_hit_rates)} seeds, "
              f"{total_present}/{total_queries} naive top-k+1 pools contained the query's own gold document)")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(per_row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWritten: {output_path}")


if __name__ == "__main__":
    main()
