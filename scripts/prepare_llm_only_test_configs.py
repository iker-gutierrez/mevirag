#!/usr/bin/env python3
"""Clone the four pre-registered dev LLM-only baselines onto test.jsonl."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "experiments"
TASKS = ROOT / "experiments" / "llm_only_test_seeded_tasks.txt"

# These are the original no-retrieval dev baseline cells (0a--0d), not
# configurations selected using held-out results.
BASELINES = (
    ("17100", "12000_qwen35_9b_no_rag_no_think_extractive_guiasalud_dev", "data/processed/guiasalud_casimedicos/test.jsonl"),
    ("17101", "12011_qwen35_9b_no_rag_think_extractive_guiasalud_dev", "data/processed/guiasalud_casimedicos/test.jsonl"),
    ("17102", "11000_llama31_8b_no_rag_extractive_guiasalud_dev", "data/processed/guiasalud_casimedicos_eu/test.jsonl"),
    ("17103", "11011_latxa_llama31_8b_no_rag_extractive_guiasalud_dev", "data/processed/guiasalud_casimedicos_eu/test.jsonl"),
)


def main() -> None:
    tasks: list[str] = []
    for ident, source_stem, test_input in BASELINES:
        payload = json.loads((CONFIGS / f"{source_stem}.json").read_text())
        name = payload["experiment_name"].replace("_guiasalud", "_guiasalud_llm_only_final_test")
        stem = f"{ident}_{name}"
        payload["experiment_name"] = name
        payload["input"] = test_input
        payload["output"] = f"experiments/runs/{stem}/predictions.jsonl"
        payload["num_runs"] = 3
        (CONFIGS / f"{stem}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        tasks.extend(f"{stem} {seed}" for seed in (42, 43, 44))
    TASKS.write_text("\n".join(tasks) + "\n")
    print(f"wrote {len(BASELINES)} configs and {len(tasks)} seeded tasks")


if __name__ == "__main__":
    main()
