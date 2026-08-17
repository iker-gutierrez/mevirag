#!/usr/bin/env python
"""Generate self-feedback (self_feedback=true) clones of the Basque and
Spanish ablation-grid configuration files.

A "self-feedback" clone is identical to its source configuration in every
retrieval and sampling setting, except that generation runs an extra
refinement pass: the model produces an initial answer, critiques it, and
revises it, with the revised answer scored separately from the initial one.
Because a row's self-feedback variant can outperform its plain variant (or
vice versa), both need to exist as real, separately generated and scored
runs for the staged selection process to compare them fairly — see
scripts/mixed_meanq.py's family_candidates() docstring, and
scripts/rewire_basque_ablation_stage.py / rewire_spanish_ablation_stage.py,
which build the candidate pool from both variants of every row.

Each clone is written to a new configuration id, with its own output path,
so generating it never overwrites the plain configuration or its
predictions. New ids: Basque self-feedback clones start at 9000, Spanish
self-feedback clones start at 10000, both in the same row order (0 through
10) as their plain source configurations.

Usage:
    python scripts/create_self_feedback_ablation_configs.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "configs" / "experiments"

FEEDBACK_MAX_NEW_TOKENS = 512

# (self-feedback start id, [source config ids in row order 0-10])
BASQUE_SOURCE_IDS = {
    "llama31_8b": (9000, [5000, 5001, 5002, 5003, 5004, 5005, 5006, 5014, 5015, 5018, 5019]),
    "latxa_llama31_8b": (9011, [5007, 5008, 5009, 5010, 5011, 5012, 5013, 5016, 5017, 5020, 5021]),
}
SPANISH_SOURCE_IDS = {
    "qwen35_9b_no_think": (10000, [7000, 7001, 7002, 7003, 7004, 7005, 7006, 7014, 7015, 7018, 7019]),
    "qwen35_9b_think": (10011, [7007, 7008, 7009, 7010, 7011, 7012, 7013, 7016, 7017, 7020, 7021]),
}


def clone_family(family: str, sf_start: int, source_ids: list[int]) -> list[tuple[int, str]]:
    written = []
    sf_id = sf_start
    for nosf_id in source_ids:
        matches = sorted(CFG_DIR.glob(f"{nosf_id}_*.json"))
        assert len(matches) == 1, f"Expected 1 config for id {nosf_id}, found {matches}"
        src = json.loads(matches[0].read_text(encoding="utf-8"))

        old_name = src["experiment_name"]
        new_name = old_name + "_sf"
        new_output = f"experiments/runs/{sf_id}_{new_name}_dev/predictions.jsonl"

        cfg = dict(src)
        cfg["experiment_name"] = new_name
        cfg["output"] = new_output
        cfg["self_feedback"] = True
        cfg["feedback_max_new_tokens"] = src["max_new_tokens"] if src.get("think") else FEEDBACK_MAX_NEW_TOKENS

        out_path = CFG_DIR / f"{sf_id}_{new_name}_dev.json"
        out_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append((sf_id, out_path.name))
        print(f"  {sf_id}: {out_path.name}  (from {nosf_id})")
        sf_id += 1
    return written


def main() -> None:
    print("=== Basque self-feedback clones (ids 9000+) ===")
    for family, (sf_start, source_ids) in BASQUE_SOURCE_IDS.items():
        print(f"-- {family} --")
        clone_family(family, sf_start, source_ids)

    print("\n=== Spanish self-feedback clones (ids 10000+) ===")
    for family, (sf_start, source_ids) in SPANISH_SOURCE_IDS.items():
        print(f"-- {family} --")
        clone_family(family, sf_start, source_ids)


if __name__ == "__main__":
    main()
