#!/usr/bin/env python
"""Run a reasoning pipeline on top of the frozen best RAG configuration.

Pipelines: structured_cot | thought_rag | thought_rag_iter | marag
(see src/mevirag/reasoning.py for what each one is and where it
comes from).

thought_rag is RAR2's Parallel Scaling strategy (xuEtAl2025 Fig. 4): sample
`parallel_thoughts` independent thoughts, concatenate, retrieve once, answer
once. thought_rag_iter is the separate Iterative Scaling strategy (re-think
with retrieved evidence, re-retrieve, repeat for `rounds`). marag samples
`num_candidates` per round, measures conflict, turns conflict into up to four
retrieval queries, and carries ALL candidates (re-sorted by confidence) into
the next round, matching the actual NJU-RL/MA-RAG code, not just its paper.
Predictions are written in exactly the schema produced by
run_generation_experiment.py, so scripts/evaluate_predictions.py and every
downstream summary script work on these runs unchanged.

Usage:
    python scripts/run_reasoning_pipeline.py --config configs/experiments/16000_*.json
    python scripts/run_reasoning_pipeline.py --config ... --seed 43
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_generation_experiment import sample_random_examples  # noqa: E402
from truncation_safety import remove_failed_output  # noqa: E402

from mevirag.data_io import read_jsonl, write_jsonl  # noqa: E402
from mevirag.generation import (  # noqa: E402
    build_chat_prompt,
    load_vllm_model,
    strip_thinking_text,
)
from mevirag.prompts import (  # noqa: E402
    SYSTEM_PROMPT_ES,
    SYSTEM_PROMPTS,
    query_text,
)
from mevirag.reasoning import (  # noqa: E402
    PIPELINES,
    build_conflict_query_prompt,
    build_consensus_prompt,
    build_solver_prompt,
    build_structured_cot_prompt,
    build_thought_answer_prompt,
    build_thought_prompt,
    concatenate_thoughts,
    conflict_score,
    has_answer_label,
    majority_candidate,
    merge_documents,
    parse_conflict_queries,
    parse_pipeline_answer,
    sort_by_confidence,
    token_confidence,
    trim_leaked_thinking_prefix,
)
from mevirag.causal_scoring import causal_score  # noqa: E402
from mevirag.retrieval import EmbeddingRetriever  # noqa: E402
from mevirag.run_logging import run_with_logs  # noqa: E402

INCOMPLETE_FINISH_REASONS = ("length", "repetition")


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a reasoning pipeline from a JSON config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None, help="Override config seed, output goes to a _seedK dir.")
    parser.add_argument("--limit", type=int, default=None, help="Override config limit (debug).")
    parser.add_argument("--save-prompts", action="store_true")
    return parser.parse_args()


class Config:
    """Thin typed view over the experiment JSON, with the same key names the
    single-pass configs already use so the two families stay comparable."""

    def __init__(self, payload: Mapping[str, Any]):
        self.raw = dict(payload)
        self.experiment_name: str = payload["experiment_name"]
        self.pipeline: str = payload["pipeline"]
        if self.pipeline not in PIPELINES:
            raise ValueError(f"pipeline must be one of {PIPELINES}, got {self.pipeline!r}")
        self.input: str = payload["input"]
        self.output: str = payload["output"]
        self.model: str = payload["model"]
        self.language: str = payload.get("language", "es")
        self.think: bool = bool(payload.get("think", False))
        self.trust_remote_code: bool = bool(payload.get("trust_remote_code", False))
        self.limit: Optional[int] = payload.get("limit")
        self.seed: int = int(payload.get("seed", 42))

        # sampling
        self.max_new_tokens: int = int(payload.get("max_new_tokens", 2048))
        self.thought_max_new_tokens: int = int(payload.get("thought_max_new_tokens", self.max_new_tokens))
        self.query_max_new_tokens: int = int(payload.get("query_max_new_tokens", 128))
        self.temperature: float = float(payload.get("temperature", 0.7))
        self.top_p: float = float(payload.get("top_p", 0.95))
        self.top_k: int = int(payload.get("top_k", 0))
        self.min_p: float = float(payload.get("min_p", 0.0))
        self.presence_penalty: float = float(payload.get("presence_penalty", 0.0))
        self.max_truncation_retries: int = int(payload.get("max_truncation_retries", 5))
        self.truncation_retry_presence_penalty: float = float(
            payload.get("truncation_retry_presence_penalty", 0.1)
        )

        # vLLM
        self.dtype: str = payload.get("dtype", "auto")
        self.max_model_len: int = int(payload.get("max_model_len", 16384))
        self.gpu_memory_utilization: float = float(payload.get("gpu_memory_utilization", 0.90))
        self.tensor_parallel_size: int = int(payload.get("tensor_parallel_size", 1))
        self.reasoning_parser: Optional[str] = payload.get("reasoning_parser")
        self.thinking_token_budget: Optional[int] = payload.get("thinking_token_budget")
        self.repetition_detection: Optional[dict] = (
            {
                "max_pattern_size": int(payload["repetition_detection_max_pattern"]),
                "min_pattern_size": int(payload.get("repetition_detection_min_pattern", 1)),
                "min_count": int(payload.get("repetition_detection_min_count", 8)),
            }
            if payload.get("repetition_detection_max_pattern")
            else None
        )

        # retrieval (the frozen best config). `or default` rather than
        # `.get(key, default)`: domain-restriction / causal-scoring winners
        # The staged selection scripts can write these
        # keys as an EXPLICIT null (no reranker stage, not merely "unset"),
        # and .get()'s own default only fires when the key is absent, not
        # when it's present-but-None -- confirmed as a live bug: config 3320
        # (frozen to a domain-restricted, no-reranker winner) crashed every
        # array task with "int() argument ... not 'NoneType'" until this fix.
        self.retrieval_index: str = payload["retrieval_index"]
        self.retrieval_top_k: int = int(payload.get("retrieval_top_k") or 15)
        self.reranker_model: str = payload.get("reranker_model") or ""
        self.reranker_top_k: int = int(payload.get("reranker_top_k") or 5)
        self.reranker_device: str = payload.get("reranker_device") or "cpu"
        # Query encoder shares the GPU with vLLM, so it defaults to CPU (see Retriever).
        self.retriever_device: str = payload.get("retriever_device", "cpu")

        # few-shot demonstrations (optional, row 8's "3-shot + <base>" ablation
        # row on top of a pipeline instead of the plain single-pass base). Only
        # the pipeline's final, answer-emitting call(s) get the examples,
        # see reasoning._few_shot_section. Absent by default so every existing
        # config (no few_shot_file) behaves exactly as before.
        self.few_shot_file: Optional[str] = payload.get("few_shot_file")
        self.few_shot_k: int = int(payload.get("few_shot_k", 0))
        self.few_shot_mode: str = payload.get("few_shot_mode", "random")
        if self.few_shot_k > 0 and self.few_shot_mode != "random":
            raise ValueError(
                f"few_shot_mode={self.few_shot_mode!r} not supported here, "
                "only 'random' (config 1280's mode) is wired into the "
                "reasoning-pipeline runner, 'retrieval' mode would need a "
                "live per-record retriever call this driver doesn't make."
            )

        # pipeline knobs
        self.rounds: int = int(payload.get("rounds", 2))
        self.num_candidates: int = int(payload.get("num_candidates", 3))
        # RAR2 Parallel Scaling: m independent thoughts sampled per record,
        # concatenated, retrieved once. Distinct from `rounds`
        # (thought_rag_iter's re-retrieve budget). Parallel Scaling never
        # re-retrieves.
        self.parallel_thoughts: int = int(payload.get("parallel_thoughts", 3))
        # Two thresholds, because the two conflict signals live on different scales.
        # Multiple choice is discrete: any candidate not backing the plurality option
        # is a real disagreement, so the bar is 0 (MA-RAG's own criterion). Semantic
        # disagreement between open answers never reaches 0, two correct paraphrases
        # of a short answer sit around 0.15 cosine distance, so a shared threshold
        # would treat every open record as conflicted and make the signal useless.
        legacy = payload.get("conflict_threshold")
        self.conflict_threshold_mc: float = float(payload.get("conflict_threshold_mc", 0.0))
        self.conflict_threshold_open: float = float(
            payload.get("conflict_threshold_open", legacy if legacy is not None else 0.25)
        )
        self.max_context_docs: int = int(payload.get("max_context_docs", 10))
        self.thought_query_mode: str = payload.get("thought_query_mode", "question_plus_thought")
        self.thought_query_max_chars: int = int(payload.get("thought_query_max_chars", 1200))

        # structured_cot only: MedCoT-RAG's causal-aware retrieval scoring,
        # s(d,q) = causal_alpha*sim(q,d) + causal_beta*psi(d) (sec:reasoning-pipelines,
        # src/mevirag/causal_scoring.py). Off by default so the other three
        # pipelines, and any existing structured_cot config that doesn't set this,
        # keep using the shared dense-retrieval-plus-reranking stage unchanged.
        self.causal_scoring: bool = bool(payload.get("causal_scoring", False))
        self.causal_alpha: float = float(payload.get("causal_alpha", 1.0))
        self.causal_beta: float = float(payload.get("causal_beta", 1.0))
        self.causal_pool_size: int = int(payload.get("causal_pool_size", 15))
        # The paper retrieves top-5 directly by s(d,q), no reranking stage at
        # all, so this is deliberately its own field, not reranker_top_k.
        self.causal_top_k: int = int(payload.get("causal_top_k", 5))


# --------------------------------------------------------------------------
# vLLM helpers
# --------------------------------------------------------------------------


class Generator:
    """One vLLM engine, several sampling profiles. Every call is batched across
    the whole dev set: a pipeline stage costs one engine call, not 126."""

    def __init__(self, config: Config):
        from transformers import AutoTokenizer

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model, trust_remote_code=config.trust_remote_code, fix_mistral_regex=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        load_started = time.perf_counter()
        self.llm = load_vllm_model(
            config.model,
            dtype=config.dtype,
            trust_remote_code=config.trust_remote_code,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            tensor_parallel_size=config.tensor_parallel_size,
            reasoning_parser=config.reasoning_parser,
        )
        self.model_load_seconds = time.perf_counter() - load_started
        self.native_thinking = "qwen3" in config.model.lower()
        self.enable_thinking = config.think if self.native_thinking else None
        self.total_generation_seconds = 0.0
        self._call_index = 0

    def chat(self, user_prompts: Sequence[str]) -> list[str]:
        system_prompt = SYSTEM_PROMPTS.get(self.config.language, SYSTEM_PROMPT_ES)
        return [
            build_chat_prompt(
                self.tokenizer,
                system_prompt,
                user_prompt,
                language=self.config.language,
                enable_thinking=self.enable_thinking,
            )
            for user_prompt in user_prompts
        ]

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenizer(text)["input_ids"])

    def _truncate(self, chat_prompts: Sequence[str], max_new_tokens: int) -> list[str]:
        allowed = self.config.max_model_len - max_new_tokens
        if allowed <= 0:
            raise ValueError(
                f"max_new_tokens ({max_new_tokens}) leaves no room in max_model_len ({self.config.max_model_len})"
            )
        truncated = []
        for prompt in chat_prompts:
            ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if len(ids) > allowed:
                # Keep the tail: the question and the output contract live at the end.
                ids = ids[-allowed:]
                prompt = self.tokenizer.decode(ids, skip_special_tokens=False)
            truncated.append(prompt)
        return truncated

    def _sampling_kwargs(self, *, n: int, max_new_tokens: int, temperature: Optional[float],
                          logprobs: Optional[int] = None) -> dict[str, Any]:
        from vllm.sampling_params import RepetitionDetectionParams

        config = self.config
        # The seed MUST reach SamplingParams. Plumbing it only as far as the config
        # (which is what the single-pass runs do) leaves vLLM on its own fixed engine
        # seed, so every "seed" re-runs the identical sample and the resulting +/-std
        # across seeds is exactly 0.00, a number that looks like measured stability
        # and is really the same run averaged with itself.
        #
        # `seed` is offset per call so the stages of a pipeline don't all draw the same
        # stream, and (critically for MA-RAG) so the N solver candidates stay
        # independent draws rather than N copies. self._call_index advances every call.
        sp_kwargs: dict[str, Any] = dict(
            n=n,
            seed=config.seed * 1000 + self._call_index,
            max_tokens=max_new_tokens,
            temperature=config.temperature if temperature is None else temperature,
            top_p=config.top_p,
            top_k=config.top_k if config.top_k > 0 else -1,
            min_p=config.min_p,
            presence_penalty=config.presence_penalty,
        )
        self._call_index += 1
        if config.thinking_token_budget is not None:
            sp_kwargs["thinking_token_budget"] = config.thinking_token_budget
        if config.repetition_detection is not None:
            sp_kwargs["repetition_detection"] = RepetitionDetectionParams(**config.repetition_detection)
        if logprobs is not None:
            sp_kwargs["logprobs"] = logprobs
        return sp_kwargs

    def generate(
        self,
        user_prompts: Sequence[str],
        *,
        max_new_tokens: int,
        n: int = 1,
        temperature: Optional[float] = None,
    ) -> tuple[list[list[str]], list[list[Optional[str]]], list[str]]:
        """Returns (texts[record][sample], finish_reasons[record][sample], chat_prompts).

        `n > 1` asks vLLM for n independent samples per prompt in a single pass --
        this is the Solver agent's parallel sampling, and it shares the prompt's
        KV cache across the samples instead of re-encoding it n times.

        Every prompt is retried (all its n samples together, fresh seed each
        attempt) until every sample finishes cleanly (finish_reason not in
        INCOMPLETE_FINISH_REASONS), up to config.max_truncation_retries extra
        attempts. Repetition-detection alone cannot catch every degenerate
        pattern (a run can drift into long, non-repeating word-salad that
        never repeats a short pattern and so never trips repetition_detection,
        confirmed on real GuiaSalud output -- 99.98% unique words), so once a
        prompt has used half its retry budget without succeeding, remaining
        attempts fall back to config.truncation_retry_presence_penalty (a
        conservative, previously-validated value) instead of repeating the
        configured value that may itself be the cause.
        """
        from vllm import SamplingParams

        config = self.config
        chat_prompts = self.chat(user_prompts)
        truncated = self._truncate(chat_prompts, max_new_tokens)

        n_prompts = len(truncated)
        texts: list[list[Optional[str]]] = [None] * n_prompts  # type: ignore[list-item]
        reasons: list[list[Optional[str]]] = [None] * n_prompts  # type: ignore[list-item]
        pending = list(range(n_prompts))

        started = time.perf_counter()
        for attempt in range(config.max_truncation_retries + 1):
            if not pending:
                break
            presence_penalty_override = (
                config.truncation_retry_presence_penalty
                if attempt > max(1, config.max_truncation_retries // 2)
                else None
            )
            sp_kwargs = self._sampling_kwargs(n=n, max_new_tokens=max_new_tokens, temperature=temperature)
            if presence_penalty_override is not None:
                sp_kwargs["presence_penalty"] = presence_penalty_override
            outputs = self.llm.generate([truncated[i] for i in pending], SamplingParams(**sp_kwargs))

            still_pending = []
            for i, output in zip(pending, outputs):
                sample_texts = [completion.text.strip() for completion in output.outputs]
                sample_reasons = [completion.finish_reason for completion in output.outputs]
                texts[i] = sample_texts
                reasons[i] = sample_reasons
                if attempt < config.max_truncation_retries and any(
                    r in INCOMPLETE_FINISH_REASONS for r in sample_reasons
                ):
                    still_pending.append(i)
            pending = still_pending
        self.total_generation_seconds += time.perf_counter() - started

        return texts, reasons, chat_prompts  # type: ignore[return-value]

    def generate_one(
        self,
        user_prompts: Sequence[str],
        *,
        max_new_tokens: int,
        temperature: Optional[float] = None,
    ) -> tuple[list[str], list[Optional[str]], list[str]]:
        texts, reasons, prompts = self.generate(
            user_prompts, max_new_tokens=max_new_tokens, n=1, temperature=temperature
        )
        return [t[0] for t in texts], [r[0] for r in reasons], prompts

    def generate_with_confidence(
        self,
        user_prompts: Sequence[str],
        *,
        max_new_tokens: int,
        n: int,
    ) -> tuple[list[list[str]], list[list[Optional[str]]], list[list[float]], list[str]]:
        """Like generate(), plus per-sample mean sampled-token logprob (see
        reasoning.token_confidence for what this stands in for and why).
        MA-RAG's Solver call goes through this so its history can be sorted
        by confidence, matching the original NJU-RL/MA-RAG code. Returns
        (texts, finish_reasons, confidences, chat_prompts).

        Same per-prompt truncation-retry loop as generate() (see its
        docstring): retried prompts also get fresh confidences from their
        successful attempt, never mixed with a discarded truncated one."""
        from vllm import SamplingParams

        config = self.config
        chat_prompts = self.chat(user_prompts)
        truncated = self._truncate(chat_prompts, max_new_tokens)

        n_prompts = len(truncated)
        texts: list[list[Optional[str]]] = [None] * n_prompts  # type: ignore[list-item]
        reasons: list[list[Optional[str]]] = [None] * n_prompts  # type: ignore[list-item]
        confidences: list[list[Optional[float]]] = [None] * n_prompts  # type: ignore[list-item]
        pending = list(range(n_prompts))

        started = time.perf_counter()
        for attempt in range(config.max_truncation_retries + 1):
            if not pending:
                break
            presence_penalty_override = (
                config.truncation_retry_presence_penalty
                if attempt > max(1, config.max_truncation_retries // 2)
                else None
            )
            # logprobs=0: return only the sampled token's own logprob per
            # position (no top-k alternatives), the minimum needed for
            # token_confidence's mean-logprob proxy without the extra cost of
            # requesting a full top-k distribution per token.
            sp_kwargs = self._sampling_kwargs(n=n, max_new_tokens=max_new_tokens, temperature=None, logprobs=0)
            if presence_penalty_override is not None:
                sp_kwargs["presence_penalty"] = presence_penalty_override
            outputs = self.llm.generate([truncated[i] for i in pending], SamplingParams(**sp_kwargs))

            still_pending = []
            for i, output in zip(pending, outputs):
                sample_texts = [completion.text.strip() for completion in output.outputs]
                sample_reasons = [completion.finish_reason for completion in output.outputs]
                sample_confidences = []
                for completion in output.outputs:
                    token_logprobs = []
                    for position_index, position in enumerate(completion.logprobs or []):
                        # Each position maps {token_id: Logprob}, the sampled
                        # token's own entry is what was actually emitted. Indexed
                        # by the loop position, not by len(token_logprobs), a
                        # missing entry at any position must not shift every
                        # later lookup out of alignment with completion.token_ids.
                        if position is None or position_index >= len(completion.token_ids):
                            continue
                        entry = position.get(completion.token_ids[position_index])
                        if entry is not None:
                            token_logprobs.append(entry.logprob)
                    sample_confidences.append(token_confidence(token_logprobs))
                texts[i] = sample_texts
                reasons[i] = sample_reasons
                confidences[i] = sample_confidences
                if attempt < config.max_truncation_retries and any(
                    r in INCOMPLETE_FINISH_REASONS for r in sample_reasons
                ):
                    still_pending.append(i)
            pending = still_pending
        self.total_generation_seconds += time.perf_counter() - started

        return texts, reasons, confidences, chat_prompts  # type: ignore[return-value]

    def visible(self, text: str) -> str:
        """Drop the thinking block for models that emit one, so downstream stages
        (and the answer parser) only ever see the model's visible output."""
        return strip_thinking_text(text) if self.native_thinking else text


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


class Retriever:
    def __init__(self, config: Config):
        self.config = config
        # Build the query encoder ON the target device, not on CUDA-then-move.
        # multilingual-e5-large is ~2.1 GiB. Constructing it on CUDA and calling
        # .to("cpu") afterwards still allocates those 2.1 GiB on the GPU, and
        # PyTorch's caching allocator keeps the block RESERVED after the tensors
        # move away, so vLLM, which then claims gpu_memory_utilization (0.90) of
        # what it can see, starts life 2 GiB short. That is what produced the
        # intermittent OOM: it struck whichever pipeline happened to retrieve again
        # after generation, and only on some seeds, because it was a race for the
        # remaining memory rather than a deterministic overflow.
        self.retriever = EmbeddingRetriever(
            config.retrieval_index, device=config.retriever_device
        )
        self.reranker = None
        if config.reranker_model and config.reranker_top_k > 0:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(config.reranker_model, device=config.reranker_device)
        self.total_seconds = 0.0

    @property
    def embedder(self) -> Any:
        """The retriever's own multilingual-e5-large, reused to score semantic
        conflict between MA-RAG candidates, no second embedding model loaded."""
        return getattr(self.retriever, "model", None)

    @property
    def query_prefix(self) -> str:
        return self.retriever.config.get("query_prefix", "")

    def search(self, query: str, exclude_id: Optional[str]) -> list[dict[str, Any]]:
        started = time.perf_counter()
        docs = self.retriever.query(
            query, top_k=self.config.retrieval_top_k, exclude_id=exclude_id
        )
        if self.reranker is not None and docs:
            pairs = [(query, str(doc.get("text") or "")) for doc in docs]
            scores = self.reranker.predict(pairs)
            ranked = sorted(zip(docs, scores), key=lambda item: float(item[1]), reverse=True)
            docs = []
            for rank, (doc, score) in enumerate(ranked[: self.config.reranker_top_k], start=1):
                item = dict(doc)
                item["pre_rerank_rank"] = item.get("rank")
                item["pre_rerank_score"] = item.get("score")
                item["reranker_score"] = float(score)
                item["rank"] = rank
                docs.append(item)
        self.total_seconds += time.perf_counter() - started
        return docs

    def search_causal(self, query: str, exclude_id: Optional[str], language: str) -> list[dict[str, Any]]:
        """MedCoT-RAG's causal-aware retrieval (sec:reasoning-pipelines): this is
        the paper's ONLY retrieval step, there is no separate cross-encoder
        reranking stage. We dense-retrieve a larger candidate pool (so psi(d) has
        genuinely different documents to select among, not just the same top-k
        the dense score would already return) and select the final top-k directly
        by the composite score s(d,q) = alpha*sim(q,d) + beta*psi(d). The paper
        retrieves "the top five documents ... using FAISS indexing" (Sec. III-A),
        no other value given, so causal_top_k defaults to 5 for every model and
        language, unlike the shared single-pass pipeline's per-model retrieval
        depth. structured_cot-only."""
        started = time.perf_counter()
        pool = self.retriever.query(
            query, top_k=self.config.causal_pool_size, exclude_id=exclude_id
        )
        docs = causal_score(
            query,
            pool,
            language=language,
            alpha=self.config.causal_alpha,
            beta=self.config.causal_beta,
            top_k=self.config.causal_top_k,
        )
        self.total_seconds += time.perf_counter() - started
        return docs


def thought_query(record: Mapping[str, Any], thought: str, config: Config) -> str:
    """RAR2 retrieves with the *thought*, not the surface question.

    `thought_only` is the paper's setting. We default to `question_plus_thought`
    because our questions carry the clinical topic, which a thought that wanders
    off can lose entirely, the mode is a config knob so the deviation is testable.
    The truncation matters: e5 encodes ~512 tokens, so an unbounded thought would
    silently drop its own tail out of the query.
    """
    base = query_text(record, language=config.language)
    thought = " ".join((thought or "").split())[: config.thought_query_max_chars]
    if config.thought_query_mode == "thought_only" and thought:
        return thought
    return f"{base}\n{thought}".strip()


# --------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------


def blank_stage_stats() -> dict[str, Any]:
    return {
        "generation_seconds": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "num_llm_calls": 0,
        "incomplete": 0,
    }


class Accumulator:
    """Per-record cost + trace ledger. Every LLM call in every pipeline funnels
    through `add`, so the reported tokens/sample covers the whole pipeline rather
    than only its last stage. This permits a valid cost comparison between a
    three-round agentic loop and a single-pass baseline."""

    def __init__(self, num_records: int):
        self.stats = [blank_stage_stats() for _ in range(num_records)]
        self.trace: list[dict[str, Any]] = [{} for _ in range(num_records)]

    def add(
        self,
        index: int,
        *,
        generator: Generator,
        prompt: str,
        outputs: Sequence[str],
        finish_reasons: Sequence[Optional[str]],
        seconds: float,
    ) -> None:
        entry = self.stats[index]
        entry["generation_seconds"] += seconds
        entry["input_tokens"] += generator.count_tokens(prompt)
        entry["output_tokens"] += sum(generator.count_tokens(text) for text in outputs)
        entry["num_llm_calls"] += len(outputs)
        entry["incomplete"] += sum(1 for reason in finish_reasons if reason in INCOMPLETE_FINISH_REASONS)


def run_structured_cot(
    records: list[dict[str, Any]],
    generator: Generator,
    retriever: Retriever,
    config: Config,
    acc: Accumulator,
    *,
    examples_per_record: Optional[list[list[dict[str, Any]]]] = None,
) -> tuple[list[str], list[Optional[str]], list[list[dict[str, Any]]], list[list[str]]]:
    docs_per_record = []
    candidate_ids = []
    for record in records:
        query = query_text(record, language=config.language)
        if config.causal_scoring:
            docs = retriever.search_causal(query, record.get("id"), config.language)
        else:
            docs = retriever.search(query, record.get("id"))
        docs_per_record.append(docs)
        candidate_ids.append([str(doc.get("doc_id") or "") for doc in docs])

    examples_per_record = examples_per_record or [[] for _ in records]
    prompts = [
        build_structured_cot_prompt(record, docs, language=config.language, examples=examples)
        for record, docs, examples in zip(records, docs_per_record, examples_per_record)
    ]
    started = time.perf_counter()
    answers, reasons, chat_prompts = generator.generate_one(
        prompts, max_new_tokens=config.max_new_tokens
    )
    per_record_seconds = (time.perf_counter() - started) / max(len(records), 1)
    for i in range(len(records)):
        acc.add(
            i,
            generator=generator,
            prompt=chat_prompts[i],
            outputs=[answers[i]],
            finish_reasons=[reasons[i]],
            seconds=per_record_seconds,
        )
        acc.trace[i] = {
            "pipeline": "structured_cot",
            "causal_scoring": config.causal_scoring,
        }
    return answers, reasons, docs_per_record, candidate_ids


def run_thought_rag(
    records: list[dict[str, Any]],
    generator: Generator,
    retriever: Retriever,
    config: Config,
    acc: Accumulator,
    *,
    examples_per_record: Optional[list[list[dict[str, Any]]]] = None,
) -> tuple[list[str], list[Optional[str]], list[list[dict[str, Any]]], list[list[str]]]:
    """RAR2 Parallel Scaling (xuEtAl2025 Fig. 4), tuning-free: sample
    `config.parallel_thoughts` independent thought processes per record (one
    vLLM call, n=parallel_thoughts, sharing the prompt's KV cache, same
    mechanism MA-RAG's Solver uses for its own parallel sampling), concatenate
    them into a single extended thought, retrieve ONCE with the
    concatenation, answer ONCE. Never re-retrieves or re-thinks across rounds,
    that is thought_rag_iter's job, not this pipeline's."""
    n = len(records)
    examples_per_record = examples_per_record or [[] for _ in range(n)]

    prompts = [build_thought_prompt(record, language=config.language) for record in records]
    started = time.perf_counter()
    sampled, reasons, chat_prompts = generator.generate(
        prompts, max_new_tokens=config.thought_max_new_tokens, n=config.parallel_thoughts
    )
    per_record_seconds = (time.perf_counter() - started) / max(n, 1)

    combined_thoughts: list[str] = []
    for i in range(n):
        visible = [generator.visible(text) for text in sampled[i]]
        acc.add(
            i,
            generator=generator,
            prompt=chat_prompts[i],
            outputs=sampled[i],
            finish_reasons=reasons[i],
            seconds=per_record_seconds,
        )
        combined_thoughts.append(concatenate_thoughts(visible))

    docs_per_record: list[list[dict[str, Any]]] = []
    candidate_ids: list[list[str]] = []
    for i, record in enumerate(records):
        docs = retriever.search(thought_query(record, combined_thoughts[i], config), record.get("id"))
        docs_per_record.append(docs)
        candidate_ids.append([str(doc.get("doc_id") or "") for doc in docs])

    prompts = [
        build_thought_answer_prompt(
            record, docs_per_record[i], combined_thoughts[i], language=config.language,
            examples=examples_per_record[i],
        )
        for i, record in enumerate(records)
    ]
    started = time.perf_counter()
    answers, reasons, chat_prompts = generator.generate_one(prompts, max_new_tokens=config.max_new_tokens)
    per_record_seconds = (time.perf_counter() - started) / max(n, 1)
    for i in range(n):
        acc.add(
            i,
            generator=generator,
            prompt=chat_prompts[i],
            outputs=[answers[i]],
            finish_reasons=[reasons[i]],
            seconds=per_record_seconds,
        )
        acc.trace[i] = {
            "pipeline": "thought_rag",
            "parallel_thoughts": config.parallel_thoughts,
            "combined_thought": combined_thoughts[i],
        }
    return answers, reasons, docs_per_record, candidate_ids


def run_thought_rag_iter(
    records: list[dict[str, Any]],
    generator: Generator,
    retriever: Retriever,
    config: Config,
    acc: Accumulator,
    *,
    rounds: int,
    examples_per_record: Optional[list[list[dict[str, Any]]]] = None,
) -> tuple[list[str], list[Optional[str]], list[list[dict[str, Any]]], list[list[str]]]:
    """RAR2 Iterative Scaling: thought -> retrieve -> re-think *with* what was
    retrieved -> retrieve again, for `rounds` rounds, then answer over the
    union of every round's evidence."""
    n = len(records)
    examples_per_record = examples_per_record or [[] for _ in range(n)]
    docs_per_record: list[list[dict[str, Any]]] = [[] for _ in range(n)]
    candidate_ids: list[list[str]] = [[] for _ in range(n)]
    thoughts: list[str] = ["" for _ in range(n)]

    for round_index in range(rounds):
        prompts = [
            build_thought_prompt(
                record,
                language=config.language,
                previous_thought=thoughts[i] if round_index > 0 else "",
                documents=docs_per_record[i] if round_index > 0 else None,
            )
            for i, record in enumerate(records)
        ]
        started = time.perf_counter()
        raw_thoughts, reasons, chat_prompts = generator.generate_one(
            prompts, max_new_tokens=config.thought_max_new_tokens
        )
        per_record_seconds = (time.perf_counter() - started) / max(n, 1)
        for i in range(n):
            acc.add(
                i,
                generator=generator,
                prompt=chat_prompts[i],
                outputs=[raw_thoughts[i]],
                finish_reasons=[reasons[i]],
                seconds=per_record_seconds,
            )
            thoughts[i] = generator.visible(raw_thoughts[i]).strip()

        for i, record in enumerate(records):
            new_docs = retriever.search(thought_query(record, thoughts[i], config), record.get("id"))
            docs_per_record[i] = merge_documents(
                docs_per_record[i] if round_index > 0 else [],
                new_docs,
                max_docs=config.max_context_docs,
            )
            for doc in new_docs:
                doc_id = str(doc.get("doc_id") or "")
                if doc_id and doc_id not in candidate_ids[i]:
                    candidate_ids[i].append(doc_id)
        print(f"  [thought_rag_iter] round {round_index + 1}/{rounds} retrieved", flush=True)

    prompts = [
        build_thought_answer_prompt(
            record, docs_per_record[i], thoughts[i], language=config.language,
            examples=examples_per_record[i],
        )
        for i, record in enumerate(records)
    ]
    started = time.perf_counter()
    answers, reasons, chat_prompts = generator.generate_one(
        prompts, max_new_tokens=config.max_new_tokens
    )
    per_record_seconds = (time.perf_counter() - started) / max(n, 1)
    for i in range(n):
        acc.add(
            i,
            generator=generator,
            prompt=chat_prompts[i],
            outputs=[answers[i]],
            finish_reasons=[reasons[i]],
            seconds=per_record_seconds,
        )
        acc.trace[i] = {"pipeline": "thought_rag_iter", "rounds": rounds, "final_thought": thoughts[i]}
    return answers, reasons, docs_per_record, candidate_ids


def run_marag(
    records: list[dict[str, Any]],
    generator: Generator,
    retriever: Retriever,
    config: Config,
    acc: Accumulator,
    *,
    examples_per_record: Optional[list[list[dict[str, Any]]]] = None,
) -> tuple[list[str], list[Optional[str]], list[list[dict[str, Any]]], list[list[str]]]:
    """MA-RAG, matching NJU-RL/MA-RAG's ma_rag_entropy.py.

    Per round: Solver samples `num_candidates` answers (with logprobs) ->
    conflict is measured -> records already unanimous are frozen and stop
    consuming compute -> the rest turn their conflict into UP TO FOUR
    retrieval queries (the original's own multi-query retrieval agent), pull
    new evidence from all of them, and carry ALL candidates (re-sorted
    ascending by confidence) into the next round's Solver prompt as history,
    there is no ranking/pruning agent in the original. A final synthesis
    pass (an adaptation for MeviRAG; the original instead takes the
    last round's plurality vote, which has no equivalent on the open-answer
    half of this dev set) resolves any record that never reaches unanimity.
    """
    n = len(records)
    examples_per_record = examples_per_record or [[] for _ in range(n)]
    docs_per_record: list[list[dict[str, Any]]] = []
    candidate_ids: list[list[str]] = []
    for record in records:
        docs = retriever.search(query_text(record, language=config.language), record.get("id"))
        docs_per_record.append(docs)
        candidate_ids.append([str(doc.get("doc_id") or "") for doc in docs])

    # history[i] = [(candidate_text, confidence), ...] for ALL candidates of
    # the most recent round, ascending by confidence, fed back whole into
    # the next round's Solver prompt (build_solver_prompt).
    history: list[list[tuple[str, float]]] = [[] for _ in range(n)]
    final_candidates: list[list[str]] = [[] for _ in range(n)]
    settled: list[bool] = [False for _ in range(n)]
    round_log: list[list[dict[str, Any]]] = [[] for _ in range(n)]
    final_answers: list[str] = ["" for _ in range(n)]
    final_reasons: list[Optional[str]] = [None for _ in range(n)]

    for round_index in range(config.rounds):
        active = [i for i in range(n) if not settled[i]]
        if not active:
            print(f"  [marag] all records reached consensus before round {round_index + 1}", flush=True)
            break

        prompts = [
            build_solver_prompt(
                records[i], docs_per_record[i], language=config.language, history=history[i],
                examples=examples_per_record[i],
            )
            for i in active
        ]
        started = time.perf_counter()
        sampled, reasons, confidences, chat_prompts = generator.generate_with_confidence(
            prompts, max_new_tokens=config.max_new_tokens, n=config.num_candidates
        )
        per_record_seconds = (time.perf_counter() - started) / max(len(active), 1)

        conflicted: list[int] = []
        for slot, i in enumerate(active):
            visible = [generator.visible(text) for text in sampled[slot]]
            acc.add(
                i,
                generator=generator,
                prompt=chat_prompts[slot],
                outputs=sampled[slot],
                finish_reasons=reasons[slot],
                seconds=per_record_seconds,
            )
            final_candidates[i] = visible
            score, mode = conflict_score(
                visible,
                record=records[i],
                embedder=retriever.embedder,
                query_prefix=retriever.query_prefix,
            )
            threshold = (
                config.conflict_threshold_mc
                if mode == "option_disagreement"
                else config.conflict_threshold_open
            )
            round_log[i].append(
                {
                    "round": round_index + 1,
                    "conflict": round(score, 4),
                    "conflict_mode": mode,
                    "threshold": threshold,
                    "num_candidates": len(visible),
                    "num_docs": len(docs_per_record[i]),
                }
            )
            if score <= threshold:
                # Consensus reached. This is plain self-consistency: the candidates
                # agree, so their agreed answer *is* the output, no synthesis pass,
                # and the record stops consuming compute in later rounds.
                choice = majority_candidate(visible, records[i])
                settled[i] = True
                final_answers[i] = visible[choice]
                final_reasons[i] = reasons[slot][choice]
            else:
                conflicted.append(i)
                # Full history for the next round's Solver prompt: ALL
                # candidates, ascending by confidence, no ranking agent.
                history[i] = sort_by_confidence(visible, confidences[slot])

        print(
            f"  [marag] round {round_index + 1}/{config.rounds}: "
            f"{len(active) - len(conflicted)}/{len(active)} reached consensus",
            flush=True,
        )
        if not conflicted:
            break
        if round_index == config.rounds - 1:
            # Rounds exhausted with these records still in conflict: retrieving more
            # evidence nobody will read would just burn compute. They fall through to
            # the consensus pass below.
            break

        # --- Retrieval agent: conflict -> up to 4 queries -> new evidence ---
        query_prompts = [
            build_conflict_query_prompt(records[i], final_candidates[i], language=config.language)
            for i in conflicted
        ]
        started = time.perf_counter()
        raw_queries, query_reasons, chat_prompts = generator.generate_one(
            query_prompts, max_new_tokens=config.query_max_new_tokens
        )
        per_record_seconds = (time.perf_counter() - started) / max(len(conflicted), 1)
        for slot, i in enumerate(conflicted):
            acc.add(
                i,
                generator=generator,
                prompt=chat_prompts[slot],
                outputs=[raw_queries[slot]],
                finish_reasons=[query_reasons[slot]],
                seconds=per_record_seconds,
            )
            fallback = query_text(records[i], language=config.language)
            queries = parse_conflict_queries(generator.visible(raw_queries[slot]), fallback=fallback)
            new_docs: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            for query in queries:
                for doc in retriever.search(query, records[i].get("id")):
                    doc_id = str(doc.get("doc_id") or "")
                    if doc_id and doc_id in seen_ids:
                        continue
                    seen_ids.add(doc_id)
                    new_docs.append(doc)
            docs_per_record[i] = merge_documents(docs_per_record[i], new_docs, max_docs=config.max_context_docs)
            for doc in new_docs:
                doc_id = str(doc.get("doc_id") or "")
                if doc_id and doc_id not in candidate_ids[i]:
                    candidate_ids[i].append(doc_id)
            round_log[i][-1]["conflict_queries"] = queries

    # --- Final synthesis, only for records that never reached unanimity ---
    unresolved = [i for i in range(n) if not settled[i]]
    if unresolved:
        prompts = [
            build_consensus_prompt(
                records[i],
                docs_per_record[i],
                # final_candidates[i] is set every round a record stays
                # active, so it is populated whenever the record reaches this
                # point, the history fallback only matters if a future edit
                # adds a code path that leaves a record unresolved without
                # ever sampling a candidate for it.
                final_candidates[i] or [text for text, _ in history[i][-1:]],
                language=config.language,
                examples=examples_per_record[i],
            )
            for i in unresolved
        ]
        started = time.perf_counter()
        synthesised, synth_reasons, chat_prompts = generator.generate_one(
            prompts, max_new_tokens=config.max_new_tokens
        )
        per_record_seconds = (time.perf_counter() - started) / max(len(unresolved), 1)
        for slot, i in enumerate(unresolved):
            acc.add(
                i,
                generator=generator,
                prompt=chat_prompts[slot],
                outputs=[synthesised[slot]],
                finish_reasons=[synth_reasons[slot]],
                seconds=per_record_seconds,
            )
            final_answers[i] = generator.visible(synthesised[slot])
            final_reasons[i] = synth_reasons[slot]
    print(
        f"  [marag] {n - len(unresolved)}/{n} records answered by consensus, "
        f"{len(unresolved)}/{n} by final synthesis",
        flush=True,
    )

    for i in range(n):
        acc.trace[i] = {
            "pipeline": "marag",
            "rounds_run": len(round_log[i]),
            "reached_consensus": settled[i],
            "resolution": "consensus" if settled[i] else "synthesis",
            "round_log": round_log[i],
            "candidates_final_round": final_candidates[i],
        }
    return final_answers, final_reasons, docs_per_record, candidate_ids


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def resolve_output(config: Config, seed_override: Optional[int]) -> Path:
    output = Path(config.output)
    if seed_override is not None:
        output = output.parent.parent / f"{output.parent.name}_seed{seed_override}" / output.name
    if not output.is_absolute():
        output = ROOT / output
    return output


def run(args: argparse.Namespace) -> None:
    run_started = time.perf_counter()
    payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config = Config(payload)
    if args.seed is not None:
        config.seed = args.seed
    if args.limit is not None:
        config.limit = args.limit
    output_path = resolve_output(config, args.seed)

    input_path = Path(config.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    records = read_jsonl(input_path)
    if config.limit:
        records = records[: config.limit]
    print(f"{config.experiment_name}: {config.pipeline} on {len(records)} records (seed {config.seed})", flush=True)

    examples_per_record: list[list[dict[str, Any]]] = [[] for _ in records]
    if config.few_shot_k > 0:
        few_shot_path = Path(config.few_shot_file)
        if not few_shot_path.is_absolute():
            few_shot_path = ROOT / few_shot_path
        few_shot_pool = read_jsonl(few_shot_path)
        rng = random.Random(config.seed)
        examples_per_record = [
            sample_random_examples(rng, few_shot_pool, record, config.few_shot_k)
            for record in records
        ]

    retriever = Retriever(config)
    generator = Generator(config)
    acc = Accumulator(len(records))

    if config.pipeline == "structured_cot":
        answers, reasons, docs_per_record, candidate_ids = run_structured_cot(
            records, generator, retriever, config, acc, examples_per_record=examples_per_record
        )
    elif config.pipeline == "thought_rag":
        answers, reasons, docs_per_record, candidate_ids = run_thought_rag(
            records, generator, retriever, config, acc, examples_per_record=examples_per_record
        )
    elif config.pipeline == "thought_rag_iter":
        answers, reasons, docs_per_record, candidate_ids = run_thought_rag_iter(
            records, generator, retriever, config, acc, rounds=max(2, config.rounds),
            examples_per_record=examples_per_record,
        )
    elif config.pipeline == "marag":
        answers, reasons, docs_per_record, candidate_ids = run_marag(
            records, generator, retriever, config, acc, examples_per_record=examples_per_record
        )
    else:  # unreachable: Config validates
        raise ValueError(config.pipeline)

    retrieval_seconds_per = retriever.total_seconds / max(len(records), 1)

    outputs = []
    num_answer_truncated = 0
    num_missing_answer_label = 0
    num_leaked_cot_trimmed = 0
    for i, record in enumerate(records):
        stats = acc.stats[i]
        raw_answer_text = generator.visible(answers[i])
        answer_text, leaked_cot_trimmed = trim_leaked_thinking_prefix(raw_answer_text)
        if leaked_cot_trimmed:
            num_leaked_cot_trimmed += 1
        parsed = parse_pipeline_answer(answer_text) if answer_text else {}
        truncated = reasons[i] in INCOMPLETE_FINISH_REASONS
        if truncated:
            num_answer_truncated += 1
        # A pipeline that reasons out loud can forget to emit the answer block at
        # all, in which case the whole chain of thought lands in short_answer and
        # the metrics for that record are meaningless. Count it rather than let it
        # quietly depress the row.
        if answer_text and not has_answer_label(answer_text):
            num_missing_answer_label += 1

        example_seconds = stats["generation_seconds"] + retrieval_seconds_per
        outputs.append(
            {
                "id": record.get("id"),
                "experiment_name": config.experiment_name,
                "model": config.model,
                "prompt_style": "extractive",
                "pipeline": config.pipeline,
                "language": config.language,
                "rag_condition": "rag",
                "reasoning_condition": config.pipeline,
                "native_thinking": generator.native_thinking,
                "source": record.get("source"),
                "topic": record.get("topic"),
                "subtopic": record.get("subtopic", ""),
                "question": record.get("question"),
                "focus": record.get("focus", ""),
                "reference_short_answer": record.get("short_answer", ""),
                "reference_evidence": record.get("evidence", ""),
                # No self-feedback stage here: the pipeline *is* the refinement
                # mechanism. initial == final so the SF columns of the shared
                # summary tooling stay well-defined (and show a zero delta).
                "initial_prediction_text": answer_text,
                "parsed_initial_prediction": parsed,
                "prediction_text": answer_text,
                "parsed_prediction": parsed,
                # Keep a complete audit trail whenever an unmistakable CoT
                # prefix was removed before scoring.  If no safe boundary was
                # found, prediction_text remains the unmodified model output.
                "raw_prediction_text": raw_answer_text if leaked_cot_trimmed else None,
                "format_cleanup": {
                    "leaked_cot_trimmed": leaked_cot_trimmed,
                    "fallback_evaluated_as_is": bool(raw_answer_text) and not leaked_cot_trimmed,
                },
                "few_shot_ids": [
                    str(example.get("id") or "") for example in examples_per_record[i]
                ],
                "retrieval_docs": docs_per_record[i],
                "retrieval_candidate_ids": candidate_ids[i],
                "reasoning_trace": acc.trace[i],
                "timing": {
                    "retrieval_seconds": retrieval_seconds_per,
                    "rerank_seconds": 0.0,
                    "few_shot_seconds": 0.0,
                    "prompt_seconds": 0.0,
                    "generation_seconds": stats["generation_seconds"],
                    "feedback_generation_seconds": 0.0,
                    "example_seconds": example_seconds,
                },
                "truncation": {
                    "initial_finish_reason": reasons[i],
                    "initial_thinking_truncated": False,
                    "initial_answer_truncated": truncated,
                    "feedback_finish_reason": None,
                    "feedback_thinking_truncated": False,
                    "feedback_answer_truncated": False,
                    "num_incomplete_stage_generations": stats["incomplete"],
                },
                "token_counts": {
                    "input_tokens": stats["input_tokens"],
                    "feedback_input_tokens": None,
                    "initial_output_tokens": stats["output_tokens"],
                    "output_tokens": stats["output_tokens"],
                },
                "pipeline_cost": {
                    "num_llm_calls": stats["num_llm_calls"],
                    "num_incomplete_generations": stats["incomplete"],
                },
            }
        )
        print(f"[{i + 1}/{len(records)}] {record.get('id', '')}", flush=True)

    still_truncated = [
        r["id"] for r in outputs if r["truncation"]["initial_finish_reason"] in INCOMPLETE_FINISH_REASONS
    ]
    if still_truncated:
        # Keep afterany RP evaluation from accepting a partial artifact when
        # one pipeline task fails: failed runs must leave no predictions file
        # to discover and score.
        remove_failed_output(output_path)
        raise RuntimeError(
            f"{len(still_truncated)} record(s) still truncated after "
            f"{config.max_truncation_retries} retries (ids: {still_truncated}). "
            "Zero truncated output is required -- raise max_truncation_retries "
            "or inspect these records manually rather than accepting partial answers."
        )

    write_jsonl(outputs, output_path)

    total_run_seconds = time.perf_counter() - run_started
    mean_calls = sum(s["num_llm_calls"] for s in acc.stats) / max(len(records), 1)
    metadata = {
        "experiment_name": config.experiment_name,
        "pipeline": config.pipeline,
        "model": config.model,
        "prompt_style": "extractive",
        "rag_condition": "rag",
        "reasoning_condition": config.pipeline,
        "native_thinking": generator.native_thinking,
        "language": config.language,
        "input": str(input_path.relative_to(ROOT) if input_path.is_relative_to(ROOT) else input_path),
        "output": str(output_path.relative_to(ROOT) if output_path.is_relative_to(ROOT) else output_path),
        "num_records": len(outputs),
        "dry_run": False,
        "seed": config.seed,
        "model_load_seconds": generator.model_load_seconds,
        "total_run_seconds": total_run_seconds,
        "mean_generation_seconds": sum(s["generation_seconds"] for s in acc.stats) / max(len(records), 1),
        "mean_example_seconds": sum(r["timing"]["example_seconds"] for r in outputs) / max(len(outputs), 1),
        "mean_llm_calls_per_record": mean_calls,
        "retrieval_index": config.retrieval_index,
        "retrieval_top_k": config.retrieval_top_k,
        "reranker_model": config.reranker_model,
        "reranker_top_k": config.reranker_top_k,
        "reranker_device": config.reranker_device,
        "self_feedback": False,
        "few_shot_file": config.few_shot_file,
        "few_shot_k": config.few_shot_k,
        "few_shot_mode": config.few_shot_mode,
        "thinking_token_budget": config.thinking_token_budget,
        "pipeline_params": {
            "rounds": config.rounds,
            "num_candidates": config.num_candidates,
            "parallel_thoughts": config.parallel_thoughts,
            "conflict_threshold_mc": config.conflict_threshold_mc,
            "conflict_threshold_open": config.conflict_threshold_open,
            "max_context_docs": config.max_context_docs,
            "thought_query_mode": config.thought_query_mode,
        },
        "truncation_counts": {
            "num_initial_thinking_truncated": 0,
            "num_initial_answer_truncated": num_answer_truncated,
            "num_feedback_thinking_truncated": 0,
            "num_feedback_answer_truncated": 0,
            "num_records": len(outputs),
        },
        "format_compliance": {
            "num_missing_answer_label": num_missing_answer_label,
            "num_leaked_cot_trimmed": num_leaked_cot_trimmed,
            "num_records": len(outputs),
        },
    }
    if config.pipeline == "marag":
        consensus = [r["reasoning_trace"].get("reached_consensus") for r in outputs]
        metadata["marag_stats"] = {
            "num_reached_consensus": sum(1 for value in consensus if value),
            "mean_rounds_run": sum(r["reasoning_trace"].get("rounds_run", 0) for r in outputs)
            / max(len(outputs), 1),
        }
    output_path.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output_path = resolve_output(Config(payload), args.seed)
    run_dir = output_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    run_with_logs(run_dir / "run.log", run_dir / "run.err", lambda: run(args))


if __name__ == "__main__":
    main()
