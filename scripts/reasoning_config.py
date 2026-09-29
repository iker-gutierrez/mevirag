"""Shared configuration builders for the five reasoning-pipeline variants."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

QWEN_ENGINE_FIELDS = {
    "max_model_len": 31744,
    "thinking_token_budget": 20700,
    "max_new_tokens": 22450,
    "thought_max_new_tokens": 22450,
    "repetition_detection_max_pattern": 20,
    "repetition_detection_min_pattern": 1,
    "repetition_detection_min_count": 8,
    "max_truncation_retries": 5,
    "truncation_retry_presence_penalty": 0.1,
}

LLAMA_ENGINE_FIELDS = {
    "max_model_len": 24576,
    "max_new_tokens": 7000,
    "thought_max_new_tokens": 7000,
    "repetition_detection_max_pattern": 32,
    "repetition_detection_min_pattern": 1,
    "repetition_detection_min_count": 8,
    "max_truncation_retries": 5,
    "truncation_retry_presence_penalty": 0.1,
}

# (ID offset, pipeline, filename suffix, overrides, drop shared reranker)
PIPELINES = [
    (0, "structured_cot", "structured_cot", {}, False),
    (1, "thought_rag", "thought_rag", {"thought_query_mode": "question_plus_thought"}, False),
    (2, "thought_rag_iter", "thought_rag_iter", {"rounds": 2, "max_context_docs": 8}, False),
    (
        3,
        "marag",
        "marag",
        {
            "rounds": 3,
            "num_candidates": 3,
            "conflict_threshold": 0.15,
            "max_context_docs": 10,
            "query_max_new_tokens": 128,
            "ranking_max_new_tokens": 32,
        },
        False,
    ),
    (
        4,
        "structured_cot",
        "structured_cot",
        {
            "retrieval_top_k": 15,
            "causal_scoring": True,
            "causal_alpha": 1.0,
            "causal_beta": 1.0,
            "causal_pool_size": 15,
            "causal_top_k": 5,
        },
        True,
    ),
]


def base_config_for(winner: dict, engine_fields: dict) -> dict:
    """Build a reasoning config from a development-selected ablation config."""
    config_path = ROOT / winner["config_path"]
    winning_config = json.loads(config_path.read_text(encoding="utf-8"))
    base = {
        "model": winning_config["model"],
        "think": winning_config["think"],
        "prompt_style": winning_config["prompt_style"],
        "trust_remote_code": winning_config.get("trust_remote_code", False),
        "temperature": winning_config["temperature"],
        "top_p": winning_config["top_p"],
        "input": winning_config["input"],
        "retrieval_index": winner.get("retrieval_index", ""),
        "retrieval_top_k": winner.get("retrieval_top_k", 0),
        "reranker_model": winner.get("reranker_model", ""),
        "reranker_top_k": winner.get("reranker_top_k", 0),
        "few_shot_file": winning_config.get("few_shot_file"),
        "few_shot_k": winner.get("few_shot_k", 0),
    }
    if winning_config.get("language"):
        base["language"] = winning_config["language"]
    if winning_config.get("top_k"):
        base["top_k"] = winning_config["top_k"]
    if winning_config.get("min_p") is not None:
        base["min_p"] = winning_config["min_p"]
    if winning_config.get("presence_penalty") is not None:
        base["presence_penalty"] = winning_config["presence_penalty"]
    base.update(engine_fields)
    if winning_config.get("think"):
        base["reasoning_parser"] = "qwen3"
    else:
        base.pop("thinking_token_budget", None)
        base.pop("reasoning_parser", None)
    return base


def retrieval_tag_for(winner: dict) -> str:
    """Return the filename tag for a selected retrieval configuration."""
    tags = {
        "Baseline LLM only": "no_rag",
        "e5 top 1": "e5_topk1",
        "e5 top 3": "e5_topk3",
        "e5 top 5": "e5_topk5",
        "rerank top 1": "e5_rerank1",
        "rerank top 3": "e5_rerank3",
        "rerank top 5": "e5_rerank5",
        "3-shot, no RAG": "3shot_no_rag",
        "3-shot + rerank top 5": "3shot_e5_rerank5",
        "domain: GuiaSalud only": "domain_guiasalud",
        "domain: CasiMedicos only": "domain_casimedicos",
    }
    return tags.get(winner["winning_cell"], "custom")
