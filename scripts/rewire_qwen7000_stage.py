#!/usr/bin/env python
"""Stage B/C rewiring for the fresh 7000-series Qwen3.5-9B GuiaSalud rerun
(configs/experiments/7000-7021 noSF, 10000-10021 SF clones), scoped
standalone rather than editing scripts/guiasalud_meanq.py's own FAMILIES
(start_id 3281/3290, hardcoded and used by the Basque/Llama/Latxa path too,
which must not be disturbed).

Reruns the exact staged procedure of scripts/rewire_guiasalud_stage.py against
the fresh ids only, after (1) adding dash bullets to format_question()'s
query composite and prepare_sns1064.py's build_justification() composite, and
(2) fixing the guiasalud_272 "levopremazina" -> "levomepromazina"
data-quality bug (a genuine content change to one gold record's evidence/
justification, not just formatting). The original 3281-3320 configs and
their predictions are left untouched.

Includes self-feedback candidates in the pool (each row contributes both a
plain candidate and a "(SF)"-suffixed one, read from the SF-clone config's
own after_feedback block), mirroring scripts/guiasalud_meanq.py's
family_candidates(include_sf=True): omitting SF would let a noSF row win by
construction whenever it happens to be evaluated first, even if that row's
own SF variant scores higher, contradicting the manuscript's own stated rule
that self-feedback is applied only where it is a row's own dev-set MeanQ-
winning state (sec:results-test).

id layout (name_template.format(cell=...) + row offset):
  noSF: qwen35_9b_no_think row 0-6 -> 7000-7006, row 7-8 -> 7014-7015, row 9-10 -> 7018-7019 (explicit)
        qwen35_9b_think    row 0-6 -> 7007-7013, row 7-8 -> 7016-7017, row 9-10 -> 7020-7021 (explicit)
  SF:   qwen35_9b_no_think rows 0-10 -> 10000-10010 (row-ordered, one clone per noSF row above)
        qwen35_9b_think    rows 0-10 -> 10011-10021 (row-ordered)

Usage:
    python scripts/rewire_qwen7000_stage.py --stage B
    python scripts/rewire_qwen7000_stage.py --stage C
    python scripts/rewire_qwen7000_stage.py --stage C --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from guiasalud_meanq import CELLS  # noqa: E402
from meanq import best_by_meanq_robust  # noqa: E402

CONFIG_DIR = ROOT / "configs" / "experiments"

BASE_FIELDS = ("retrieval_top_k", "reranker_model", "reranker_top_k")
NON_RETRIEVING_ROWS = {0, 7}

# (family key, display label, name template, start id [rows 0-6 use
# start_id+row; rows 7-8/9-10 use SEPARATE explicit ids, since 7000-7013 is
# already rows 0-6 for both variants back to back])
FAMILIES_7K = [
    ("qwen35_9b_no_think", "Qwen3.5-9B (no-think)",
     "qwen35_9b_{cell}_no_think_extractive_guiasalud_dev", 7000),
    ("qwen35_9b_think", "Qwen3.5-9B (think)",
     "qwen35_9b_{cell}_think_extractive_guiasalud_dev", 7007),
]

# Row 7-8 explicit ids (can't extend the row-0-6 blocks: 7000-7006/7007-7013
# are already full, stage-A rows 0-6 only).
ROW_7_8_IDS = {
    "qwen35_9b_no_think": {7: 7014, 8: 7015},
    "qwen35_9b_think": {7: 7016, 8: 7017},
}

# Row 9-10 explicit ids (domain restriction).
ROW_9_10_IDS = {
    "qwen35_9b_no_think": {9: 7018, 10: 7019},
    "qwen35_9b_think": {9: 7020, 10: 7021},
}

# SF-clone ids, row-ordered (row 0 first, row 10 last), one block per family
# -- see scripts/create_5000_7000_sf_configs.py, which wrote these in this
# exact order from the noSF ids above.
SF_ROW_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SF_START_ID = {
    "qwen35_9b_no_think": 10000,
    "qwen35_9b_think": 10011,
}

STAGES = {
    "B": {
        "pool_rows": range(0, 7),
        "pool_label": "stage A (rows 0-6)",
        "target_rows": [8],
        "target_label": "row 8 (3-shot + best RAG)",
    },
    "C": {
        "pool_rows": range(0, 9),
        "pool_label": "stages A+B (rows 0-8)",
        "target_rows": [9, 10],
        "target_label": "rows 9-10 (domain restriction)",
    },
}


def config_path(prefix: str, base: str) -> Path:
    return CONFIG_DIR / f"{prefix}_{base}.json"


def sf_run_id(family_key: str, row: int) -> int:
    return SF_START_ID[family_key] + SF_ROW_ORDER.index(row)


def candidates_7k(family_key: str, name_template: str, start_id: int, rows) -> dict:
    candidates = {}
    for row in rows:
        cell_slug, cell_label, _ = CELLS[row]
        if row <= 6:
            run_id = start_id + row
        elif row in (7, 8):
            run_id = ROW_7_8_IDS[family_key][row]
        elif row in (9, 10):
            run_id = ROW_9_10_IDS[family_key][row]
        else:
            raise ValueError(f"unexpected row {row}")
        base = name_template.format(cell=cell_slug)
        candidates[cell_label] = (str(run_id), base)

        sf_id = sf_run_id(family_key, row)
        assert base.endswith("_dev"), f"expected base to end in _dev, got {base!r}"
        sf_base = base[: -len("_dev")] + "_sf_dev"
        candidates[f"{cell_label} (SF)"] = (str(sf_id), sf_base, True)
    return candidates


def retrieving_pool(candidates: dict, pool_rows) -> dict:
    excluded = {CELLS[row][1] for row in pool_rows if row in NON_RETRIEVING_ROWS}

    def base_label(label: str) -> str:
        return label[: -len(" (SF)")] if label.endswith(" (SF)") else label

    return {k: v for k, v in candidates.items() if base_label(k) not in excluded}


def base_fields_of(prefix: str, base: str) -> dict:
    cfg = json.loads(config_path(prefix, base).read_text(encoding="utf-8"))
    return {field: cfg.get(field) for field in BASE_FIELDS}


def apply_base(prefix: str, base: str, fields: dict, *, dry_run: bool) -> bool:
    path = config_path(prefix, base)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    changed = any(cfg.get(k) != v for k, v in fields.items())
    if not changed:
        print(f"    {path.name}: already matches, no change")
        return False
    print(f"    {path.name}: {({k: cfg.get(k) for k in fields})} -> {fields}")
    if not dry_run:
        cfg.update(fields)
        path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stage = STAGES[args.stage]
    any_changed = False

    for family_key, family_label, name_template, start_id in FAMILIES_7K:
        print(f"=== {family_label} ({stage['pool_label']} -> {stage['target_label']}) ===")
        candidates = candidates_7k(family_key, name_template, start_id, stage["pool_rows"])
        pool = retrieving_pool(candidates, stage["pool_rows"])

        winner, stats = best_by_meanq_robust(pool)
        if winner is None:
            print(f"  SKIP {family_label}: no metrics for any retrieving candidate yet")
            continue

        winner_spec = pool[winner]
        winner_prefix, winner_base = winner_spec[0], winner_spec[1]
        print(f"  winner: {winner} (id {winner_prefix}, MeanQ {stats[winner]['mean']:.2f})")
        fields = base_fields_of(winner_prefix, winner_base)

        for target_row in stage["target_rows"]:
            cell_slug, cell_label, _ = CELLS[target_row]
            if target_row in (7, 8):
                target_id = ROW_7_8_IDS[family_key][target_row]
            else:
                target_id = ROW_9_10_IDS[family_key][target_row]
            target_base = name_template.format(cell=cell_slug)
            changed = apply_base(str(target_id), target_base, fields, dry_run=args.dry_run)
            any_changed = any_changed or changed

            target_sf_id = sf_run_id(family_key, target_row)
            target_sf_base = target_base[: -len("_dev")] + "_sf_dev"
            changed_sf = apply_base(str(target_sf_id), target_sf_base, fields, dry_run=args.dry_run)
            any_changed = any_changed or changed_sf
        print()

    if args.dry_run:
        print("Dry run: no files written.")
    elif not any_changed:
        print("No configs needed rewiring.")


if __name__ == "__main__":
    main()
