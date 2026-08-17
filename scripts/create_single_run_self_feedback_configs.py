#!/usr/bin/env python
"""Create the single-run self-feedback ablation-grid configs (Basque
11000-11021, Spanish 12000-12021), cloned from the existing plain-run
configs (Basque 5000-5021, Spanish 7000-7021) with self_feedback=true added.

Supersedes the earlier two-separate-generation-runs design (create_self_
feedback_ablation_configs.py, ids 9000-9021/10000-10021): the manuscript's
own description of self-feedback (sec:sf, and Figures fig:pipeline-base-rag/
fig:pipeline-base-noretrieval) is a SINGLE generation run per row that
produces both the initial and the revised answer in one pass ("Both the
initial and the revised answers are stored for every run, so every
configuration can be read in two ways: noSF, using the first answer, and
SF, using the revised one"), which run_generation_experiment.py's own
--self-feedback flag already implements directly -- no separate clone
config or second generation run is needed. Each reading is still evaluated
completely independently (scripts/evaluate_predictions.py scores
initial_prediction_text and prediction_text as two separate metric blocks,
before_feedback and after_feedback, from the same run).

The two-run design's configs (5000-5021/7000-7021, all self_feedback=false,
plus their now-obsolete 9000-9021/10000-10021 clones) are left completely
untouched: real predictions and metrics already exist for a substantial
number of them, so this writes an entirely fresh id block rather than
editing those configs in place, per the standing rule that a config file
must never claim a generation setting different from how its existing
predictions were actually produced.

Usage:
    python scripts/create_single_run_self_feedback_configs.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "configs" / "experiments"

FEEDBACK_MAX_NEW_TOKENS = 512

# (new start id, [source (plain, self_feedback=false) config ids in row
# order 0-10]). Each family spans 11 consecutive new ids (one per row), so
# the second family in each language starts 11 ids after the first.
BASQUE_SOURCE_IDS = {
    "llama31_8b": (11000, [5000, 5001, 5002, 5003, 5004, 5005, 5006, 5014, 5015, 5018, 5019]),
    "latxa_llama31_8b": (11011, [5007, 5008, 5009, 5010, 5011, 5012, 5013, 5016, 5017, 5020, 5021]),
}
SPANISH_SOURCE_IDS = {
    "qwen35_9b_no_think": (12000, [7000, 7001, 7002, 7003, 7004, 7005, 7006, 7014, 7015, 7018, 7019]),
    "qwen35_9b_think": (12011, [7007, 7008, 7009, 7010, 7011, 7012, 7013, 7016, 7017, 7020, 7021]),
}


def clone_family(family: str, new_start: int, source_ids: list[int]) -> list[tuple[int, str]]:
    written = []
    new_id = new_start
    for source_id in source_ids:
        matches = sorted(CFG_DIR.glob(f"{source_id}_*.json"))
        assert len(matches) == 1, f"Expected 1 config for id {source_id}, found {matches}"
        src = json.loads(matches[0].read_text(encoding="utf-8"))
        assert src.get("self_feedback") is False, (
            f"source config {source_id} already has self_feedback={src.get('self_feedback')!r}; "
            f"expected a plain (false) config to clone from"
        )

        name = src["experiment_name"]
        new_output = f"experiments/runs/{new_id}_{name}_dev/predictions.jsonl"

        cfg = dict(src)
        cfg["output"] = new_output
        cfg["self_feedback"] = True
        cfg["feedback_max_new_tokens"] = src["max_new_tokens"] if src.get("think") else FEEDBACK_MAX_NEW_TOKENS

        out_path = CFG_DIR / f"{new_id}_{name}_dev.json"
        out_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append((new_id, out_path.name))
        print(f"  {new_id}: {out_path.name}  (from {source_id})")
        new_id += 1
    return written


def main() -> None:
    print("=== Basque single-run self-feedback configs (11000-11021) ===")
    for family, (new_start, source_ids) in BASQUE_SOURCE_IDS.items():
        print(f"-- {family} --")
        clone_family(family, new_start, source_ids)

    print("\n=== Spanish single-run self-feedback configs (12000-12021) ===")
    for family, (new_start, source_ids) in SPANISH_SOURCE_IDS.items():
        print(f"-- {family} --")
        clone_family(family, new_start, source_ids)


if __name__ == "__main__":
    main()
