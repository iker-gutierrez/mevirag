#!/usr/bin/env python3
"""Clone RP-stage winners onto the curated held-out test splits.

The winning source config is read from ``rp_stage_selection.json`` rather
than being duplicated here.  This changes only input/output identifiers; all
generation and retrieval hyperparameters remain inherited from the selected
dev config.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "experiments"
TASKS = ROOT / "experiments" / "final_test_seeded_tasks.txt"
SELECTION = ROOT / "reports" / "metrics" / "rp_stage_selection.json"
FIRST_TEST_ID = 17000
# Presentation/submission order only; source configurations are still read
# from the finalized selection artifact below.
MODEL_ORDER = (
    "qwen35_9b_no_think",
    "qwen35_9b_think",
    "llama31_8b",
    "latxa_llama31_8b",
)


def main() -> None:
    if not SELECTION.exists():
        raise FileNotFoundError(f"Run scripts/select_rp_stage_winners.py first: {SELECTION}")
    selections = json.loads(SELECTION.read_text())
    tasks: list[str] = []
    if set(selections) != set(MODEL_ORDER):
        raise ValueError(f"unexpected model set in {SELECTION}: {sorted(selections)}")
    # Test run IDs are output labels only. The selected source configuration
    # comes exclusively from the RP-stage artifact, never from this order.
    for offset, family in enumerate(MODEL_ORDER):
        selected = selections[family]
        ident = str(FIRST_TEST_ID + offset)
        source_path = ROOT / selected["winner_config_path"]
        source = json.loads(source_path.read_text())
        # RP configs include ``_guiasalud_dev`` in their experiment name,
        # whereas single-pass configs keep ``_dev`` only in their run id.
        # Normalize both to one unambiguous held-out-test name.
        name = source["experiment_name"].replace("_guiasalud_dev", "_guiasalud_final_test")
        if name == source["experiment_name"]:
            name = name.replace("_guiasalud", "_guiasalud_final_test")
        stem = f"{ident}_{name}"
        source["experiment_name"] = name
        source["input"] = selected["test_input"]
        source["output"] = f"experiments/runs/{stem}/predictions.jsonl"
        source["num_runs"] = 3
        (CONFIGS / f"{stem}.json").write_text(json.dumps(source, indent=2, ensure_ascii=False) + "\n")
        tasks.extend(f"{stem} {selected['driver']} {seed}" for seed in (42, 43, 44))
    TASKS.write_text("\n".join(tasks) + "\n")
    print(f"wrote {len(selections)} configs and {len(tasks)} seeded tasks from {SELECTION.name}")


if __name__ == "__main__":
    main()
