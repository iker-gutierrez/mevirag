#!/usr/bin/env python3
"""Select each model's final dev winner from its RAG baseline and RP runs.

The resulting ``reports/metrics/rp_stage_selection.json`` is the only input
used to prepare held-out test configurations.  This prevents a manually
copied winner name from drifting away from the RP-stage decision rule.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from meanq import best_by_meanq_robust, incomplete_candidates  # noqa: E402

METRICS = ROOT / "reports" / "metrics"
OUTPUT = METRICS / "rp_stage_selection.json"

# These define the model families and their reasoning-pipeline candidate
# manifests. They do not prescribe a winner: the winner is
# always calculated from the complete baseline + RP metric pool below.
FAMILIES: dict[str, dict[str, str]] = {
    "qwen35_9b_no_think": {
        "baseline_selection": "costaware_rerun_selection_qwen35_9b_no_think.json",
        "baseline_key": "qwen35_9b_no_think",
        "rp_manifest": "costaware_reasoning_configs_manifest_qwen35_9b_no_think.txt",
        "test_input": "data/processed/guiasalud_casimedicos/test.jsonl",
    },
    "qwen35_9b_think": {
        "baseline_selection": "costaware_rerun_selection_qwen35_9b_think.json",
        "baseline_key": "qwen35_9b_think",
        "rp_manifest": "costaware_reasoning_configs_manifest_qwen35_9b_think.txt",
        "test_input": "data/processed/guiasalud_casimedicos/test.jsonl",
    },
    "llama31_8b": {
        "baseline_selection": "costaware_rerun_selection_llama31_8b.json",
        "baseline_key": "llama31_8b",
        "rp_manifest": "costaware_reasoning_configs_manifest_llama31_8b.txt",
        "test_input": "data/processed/guiasalud_casimedicos_eu/test.jsonl",
    },
    "latxa_llama31_8b": {
        "baseline_selection": "mixed_meanq_selection_11000.json",
        "baseline_key": "latxa_llama31_8b",
        "rp_manifest": "guiasalud_reasoning_configs_manifest_11000.txt",
        "rp_stem_prefix": "132",
        "test_input": "data/processed/guiasalud_casimedicos_eu/test.jsonl",
    },
}


def config_info(config_path: Path) -> dict[str, Any]:
    payload = json.loads(config_path.read_text())
    stem = config_path.stem
    prefix, _, base = stem.partition("_")
    if not prefix or not base:
        raise ValueError(f"Config name must start with numeric run id: {config_path}")
    return {
        "config_path": str(config_path.relative_to(ROOT)),
        "run_id": stem,
        "prefix": prefix,
        "base": base,
        "pipeline": payload.get("pipeline"),
        "experiment_name": payload["experiment_name"],
    }


def main() -> None:
    selections: dict[str, Any] = {}
    for family, spec in FAMILIES.items():
        baseline_payload = json.loads((METRICS / spec["baseline_selection"]).read_text())
        baseline = config_info(ROOT / baseline_payload[spec["baseline_key"]]["config_path"])

        candidates: dict[str, tuple[str, str]] = {"RAG baseline": (baseline["prefix"], baseline["base"])}
        candidate_info: dict[str, dict[str, Any]] = {"RAG baseline": baseline}
        for line in (METRICS / spec["rp_manifest"]).read_text().splitlines():
            if not line.strip():
                continue
            info = config_info(ROOT / line.strip())
            if spec.get("rp_stem_prefix") and not info["prefix"].startswith(spec["rp_stem_prefix"]):
                continue
            if not info["pipeline"]:
                raise ValueError(f"RP manifest contains a non-RP config: {info['config_path']}")
            label = f"RP: {info['pipeline']} ({info['run_id']})"
            candidates[label] = (info["prefix"], info["base"])
            candidate_info[label] = info

        incomplete = incomplete_candidates(candidates)
        if incomplete:
            raise RuntimeError(f"{family}: incomplete candidate metrics: {incomplete}")
        winner, scores = best_by_meanq_robust(candidates)
        if winner is None:
            raise RuntimeError(f"{family}: no candidate had MeanQ metrics")

        chosen = candidate_info[winner]
        selections[family] = {
            "model_label": family,
            "selection_stage": "RP stage",
            "decision_rule": {
                "function": "best_by_meanq_robust",
                "meanq_margin": 0.5,
                "std_threshold": 0.5,
                "token_threshold": 1000.0,
            },
            "test_input": spec["test_input"],
            "winner_label": winner,
            "winner_run_id": chosen["run_id"],
            "winner_config_path": chosen["config_path"],
            "driver": "reasoning" if chosen["pipeline"] else "generation",
            "pipeline": chosen["pipeline"],
            "winner_stats": scores[winner],
            "candidates": {
                label: {**candidate_info[label], "stats": scores[label]}
                for label in candidates
            },
        }

    OUTPUT.write_text(json.dumps(selections, indent=2, ensure_ascii=False) + "\n")
    for family, selection in selections.items():
        print(f"{family}: {selection['winner_label']} -> {selection['winner_run_id']}")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
