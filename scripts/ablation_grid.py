"""Shared definitions for the eleven-condition MeviRAG ablation grid."""

from __future__ import annotations


# (configuration slug, display label)
CELLS = [
    ("no_rag", "Baseline LLM only"),
    ("rag_e5_topk1", "e5 top 1"),
    ("rag_e5_topk3", "e5 top 3"),
    ("rag_e5_topk5", "e5 top 5"),
    ("rag_e5_rerank1", "rerank top 1"),
    ("rag_e5_rerank3", "rerank top 3"),
    ("rag_e5_rerank5", "rerank top 5"),
    ("3shot_no_rag", "3-shot, no RAG"),
    ("rag_3shot_e5_rerank5", "3-shot + rerank top 5"),
    ("rag_domain_guiasalud_e5_rerank5", "domain: GuiaSalud only"),
    ("rag_domain_casimedicos_e5_rerank5", "domain: CasiMedicos only"),
]
