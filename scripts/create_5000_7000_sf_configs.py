#!/usr/bin/env python
"""Generate self_feedback=true clones of the 5000-series (Basque) and
7000-series (Spanish) GuiaSalud ablation-grid configs, mirroring
create_guiasalud_ablation_sf_configs.py's own cloning pattern for the
original 3xxx grid (source -> self_feedback=true clone at a fresh id, same
retrieval/sampling fields, new output path).

Fills a real gap in the 5000/7000-series rerun: every one of those configs
was cloned with self_feedback=false and no SF counterpart was ever created,
so the staged selection was choosing a winner from a noSF-only candidate
pool -- a real, meaningful narrowing relative to the manuscript's own stated
rule (self-feedback is applied only where it is a row's own dev-set
MeanQ-winning state, so a genuine comparison needs both variants in the
pool, see scripts/mixed_meanq.py's family_candidates() docstring).

New ids: Basque SF clones at 9000+, Spanish SF clones at 10000+, both clear
of every id used so far. Existing 5000-5021/7000-7021 configs and their
predictions are left completely untouched -- this only ADDS new configs.

Usage:
    python scripts/create_5000_7000_sf_configs.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "configs" / "experiments"

FEEDBACK_MAX_NEW_TOKENS = 512

# (sf_start_id, [source ids in row order 0-10])
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
    print("=== Basque (9000-series SF clones) ===")
    for family, (sf_start, source_ids) in BASQUE_SOURCE_IDS.items():
        print(f"-- {family} --")
        clone_family(family, sf_start, source_ids)

    print("\n=== Spanish (10000-series SF clones) ===")
    for family, (sf_start, source_ids) in SPANISH_SOURCE_IDS.items():
        print(f"-- {family} --")
        clone_family(family, sf_start, source_ids)


if __name__ == "__main__":
    main()
