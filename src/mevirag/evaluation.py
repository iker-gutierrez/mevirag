from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Optional, Union

import numpy as np


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^0-9a-záéíóúüñ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_prf(prediction: str, reference: str) -> dict[str, float]:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    if overlap == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall)}


def token_f1(prediction: str, reference: str) -> float:
    return token_prf(prediction, reference)["f1"]


def lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for idx, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[idx - 1] + 1)
            else:
                current.append(max(previous[idx], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def cosine_similarity_score(
    predictions: list[str],
    references: list[str],
    model_name: str,
    batch_size: int = 32,
) -> list[float]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    pred_embeddings = model.encode(
        predictions,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    ref_embeddings = model.encode(
        references,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    scores = np.sum(pred_embeddings * ref_embeddings, axis=1)
    return [float(score) for score in scores]


def score_text_pairs_with_bert(
    predictions: list[str],
    references: list[str],
    model_name: str,
    batch_size: int = 16,
    lang: str = "es",
    device: Optional[str] = None,
) -> list[float]:
    from bert_score import score

    scores: list[Optional[float]] = [None] * len(predictions)
    valid_indices = [
        idx for idx, (prediction, reference) in enumerate(zip(predictions, references)) if prediction and reference
    ]
    if valid_indices:
        valid_predictions = [predictions[idx] for idx in valid_indices]
        valid_references = [references[idx] for idx in valid_indices]
        _, _, f1 = score(
            valid_predictions,
            valid_references,
            model_type=model_name,
            lang=lang,
            batch_size=batch_size,
            rescale_with_baseline=False,
            verbose=False,
            device=device,
        )
        for idx, value in zip(valid_indices, f1.tolist()):
            scores[idx] = float(value)
    return [score if score is not None else 0.0 for score in scores]


def bertscore_f1(
    predictions: list[str],
    references: list[str],
    model_name: str,
    batch_size: int = 16,
    lang: str = "es",
    device: Optional[str] = None,
) -> list[float]:
    return score_text_pairs_with_bert(
        predictions,
        references,
        model_name=model_name,
        batch_size=batch_size,
        lang=lang,
        device=device,
    )


def context_text(record: Mapping[str, Any]) -> str:
    docs = record.get("retrieval_docs") or []
    texts = []
    if isinstance(docs, list):
        for doc in docs:
            if isinstance(doc, Mapping):
                texts.append(str(doc.get("text") or doc.get("evidence") or ""))
            else:
                texts.append(str(doc))
    return "\n".join(text for text in texts if text)


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [value for value in values if value is not None]
    return mean(clean) if clean else None


def percent(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return value * 100


def number_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_numeric(values: Iterable[Any]) -> dict[str, Optional[float]]:
    clean = [number_or_none(value) for value in values]
    clean = [value for value in clean if value is not None]
    if not clean:
        return {"total": None, "mean": None, "median": None, "min": None, "max": None}
    return {
        "total": sum(clean),
        "mean": mean(clean),
        "median": median(clean),
        "min": min(clean),
        "max": max(clean),
    }


def summarize_tokens(values: Iterable[Any]) -> dict[str, Optional[float]]:
    summary = summarize_numeric(values)
    if summary["total"] is not None:
        summary["total"] = int(summary["total"])
    return summary


def sample_cost(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "timing": dict(record.get("timing") or {}),
        "token_counts": dict(record.get("token_counts") or {}),
    }


def summarize_cost(records: list[Mapping[str, Any]], run_metadata: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    run_metadata = run_metadata or {}
    timing_fields = (
        "retrieval_seconds",
        "rerank_seconds",
        "few_shot_seconds",
        "prompt_seconds",
        "generation_seconds",
        "feedback_generation_seconds",
        "example_seconds",
    )
    token_fields = (
        "input_tokens",
        "feedback_input_tokens",
        "initial_output_tokens",
        "output_tokens",
    )

    timing = {
        field: summarize_numeric((record.get("timing") or {}).get(field) for record in records)
        for field in timing_fields
    }
    token_counts = {
        field: summarize_tokens((record.get("token_counts") or {}).get(field) for record in records)
        for field in token_fields
    }
    total_token_values = []
    for record in records:
        counts = record.get("token_counts") or {}
        input_tokens = number_or_none(counts.get("input_tokens")) or 0.0
        output_tokens = number_or_none(counts.get("output_tokens")) or 0.0
        feedback_input_tokens = number_or_none(counts.get("feedback_input_tokens")) or 0.0
        total_token_values.append(input_tokens + output_tokens + feedback_input_tokens)
    token_counts["total_tokens"] = summarize_tokens(total_token_values)

    total_run_seconds = number_or_none(run_metadata.get("total_run_seconds"))
    if total_run_seconds is None:
        total_run_seconds = timing["example_seconds"]["total"]
    model_load_seconds = number_or_none(run_metadata.get("model_load_seconds"))
    gpu_count = int(run_metadata.get("gpu_count") or 1)
    gpu_hours = total_run_seconds / 3600 * gpu_count if total_run_seconds is not None else None

    return {
        "total_run_seconds": total_run_seconds,
        "model_load_seconds": model_load_seconds,
        "gpu_count": gpu_count,
        "gpu_hours": gpu_hours,
        "timing": timing,
        "token_counts": token_counts,
    }


def prediction_text(
    record: Mapping[str, Any],
    *,
    parsed_key: str = "parsed_prediction",
    text_key: str = "prediction_text",
) -> str:
    parsed = record.get(parsed_key) or {}
    if isinstance(parsed, dict) and parsed.get("short_answer"):
        return str(parsed["short_answer"])
    return str(record.get(text_key, ""))


def parsed_prediction_sections(
    record: Mapping[str, Any],
    *,
    parsed_key: str = "parsed_prediction",
    text_key: str = "prediction_text",
) -> dict[str, str]:
    # Predictions are never instructed to produce GuiaSalud's own
    # "Evidencia procedente de la investigación:"/"Consideraciones
    # adicionales:" section labels, but the prompt DOES tell the model it
    # "puede copiar frases exactas" from retrieved context, and for the
    # mixed corpus's trimmed fallback (format_context_document), that
    # context is the raw justification field WITH those labels still
    # attached to the start of the text (no boundary marker separates
    # label from content). A minority of predictions (confirmed by
    # inspection, ~1-2% of records in spot-checked runs) do copy the label
    # verbatim as part of an otherwise-legitimate exact-phrase copy. Since
    # reference_sections() strips the same labels from the gold side (the
    # model is never ASKED to reproduce them, so leaving them unstripped
    # only on the gold side would deflate every prediction's score by
    # unmatchable boilerplate), the two sides must be stripped
    # symmetrically here too, or the rare predictions that DID copy a
    # label would be penalized for following the copy-exact-phrases
    # instruction correctly.
    parsed = record.get(parsed_key) or {}
    if isinstance(parsed, Mapping):
        return {
            "short_answer": str(parsed.get("short_answer") or ""),
            "evidence": strip_justification_labels(str(parsed.get("evidence") or "")),
        }
    return {
        "short_answer": prediction_text(record, parsed_key=parsed_key, text_key=text_key),
        "evidence": "",
    }


# Matches reasoning.py's OPTION_RE: the answer contract asks multiple-choice
# records to open the short answer with the chosen option number ("3. ..."),
# optionally bolded/parenthesized ("**3.**", "(3)").
OPTION_ANSWER_RE = re.compile(r"^\s*\**\s*\(?([1-9][0-9]*)\s*[\.\)\-:]")


def extract_option_number(short_answer: str) -> Optional[str]:
    match = OPTION_ANSWER_RE.match(short_answer or "")
    return match.group(1) if match else None


def resolve_predicted_option(prediction: str, options: Mapping[str, Any]) -> Optional[str]:
    """The answer contract asks for a leading option number ("3. ..."), which
    most predictions provide. Use it directly when present. Some generators
    (observed for Mistral-7B) instead answer in free text with no leading
    digit. For those, fall back to whichever option's text the prediction
    overlaps with most by token F1 (reusing the same token_prf scorer already
    used for ROUGE/BERT-adjacent quality metrics elsewhere in this module).
    The digit-first rule handles the expected output format, while the content
    fallback supports otherwise valid answers that omit the leading number."""
    predicted = extract_option_number(prediction)
    if predicted is not None and predicted in options:
        return predicted
    if not prediction.strip():
        return None
    return max(options.keys(), key=lambda k: token_f1(prediction, str(options[k])))


def mc_accuracy(
    prediction: str,
    options: Mapping[str, Any],
    correct_option: Any,
) -> Optional[float]:
    """1.0 if the predicted option (see resolve_predicted_option) matches the
    gold option, 0.0 if it names a different one, None if no option could be
    resolved at all (a genuine format failure, an empty prediction) or no
    gold option is available to compare against."""
    if not options or correct_option is None:
        return None
    predicted = resolve_predicted_option(prediction, options)
    if predicted is None:
        return None
    return 1.0 if str(predicted) == str(correct_option) else 0.0


# GuiaSalud's justification field is a labeled composite of its own two
# source components (evidence + considerations), built by joining them with
# these fixed section labels (scripts/prepare_sns1064.py for Spanish,
# scripts/translate_to_basque.py for Basque, where both labels are
# machine-translated ONCE up front and reused for every record, so the
# Basque phrasing below is stable, not per-record variable -- confirmed by
# scanning the full guiasalud_casimedicos_eu dev split for GuiaSalud-only
# records, only these two phrases ever appear). The model is never asked to
# reproduce these labels (its own "Evidencia:"/"Ebidentzia:" placeholder is
# generic free text), so leaving them in the gold reference text scored
# against the model's prediction would deflate every overlap-based metric
# (ROUGE-L, token F1) by a fixed amount of unmatchable boilerplate,
# independent of the prediction's actual content quality. Stripped here,
# not at construction time, so the raw justification field (used elsewhere,
# e.g. the schema-normalization figure/retrieval index) keeps its labels.
_JUSTIFICATION_LABEL_PATTERNS = [
    re.compile(r"(?:^|(?<=\n))-?[ \t]*Evidencia procedente de la investigación:[ \t]*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"(?:^|(?<=\n))-?[ \t]*Consideraciones adicionales:[ \t]*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"(?:^|(?<=\n))-?[ \t]*Ikerketatik datorren ebidentzia:[ \t]*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"(?:^|(?<=\n))-?[ \t]*Gogoeta gehigarriak:[ \t]*", re.IGNORECASE | re.MULTILINE),
]


def strip_justification_labels(text: str) -> str:
    for pattern in _JUSTIFICATION_LABEL_PATTERNS:
        text = pattern.sub("", text)
    return text.strip()


def reference_sections(record: Mapping[str, Any]) -> dict[str, str]:
    # The "evidence" section name is kept as-is (see SECTION_NAMES): it's tied
    # to the model's own prompt-output label "Evidencia:" (prompts.py's
    # build_extractive_user_prompt), which is unchanged. Only the SOURCE of
    # the gold reference value for that section changes here: justification
    # (GuiaSalud's "Evidencia procedente de la investigación: .../
    # Consideraciones adicionales: ..." composite, CasiMedicos's full_answer)
    # is now the scored reference, falling back through reference_justification
    # (run-time snapshot) -> justification (dataset field) -> reference_evidence
    # / evidence (older runs/datasets without justification) for backward
    # compatibility, mirroring reference_short_answer's existing fallback style.
    # GuiaSalud's own section labels are stripped (see
    # strip_justification_labels) since the model is never asked to
    # reproduce them; CasiMedicos's justification (doctors' free-text
    # explanation) has no such labels, so stripping is a no-op there.
    return {
        "short_answer": str(
            record.get("reference_short_answer")
            or record.get("short_answer")
            or ""
        ),
        "evidence": strip_justification_labels(str(
            record.get("reference_justification")
            or record.get("justification")
            or record.get("reference_evidence")
            or record.get("evidence")
            or ""
        )),
    }


def reference_text(record: Mapping[str, Any]) -> str:
    return str(
        record.get("reference_short_answer")
        or record.get("short_answer")
        or ""
    )


def reference_options_and_correct(record: Mapping[str, Any]) -> tuple[dict, Any]:
    """(options, correct_option) for a multiple-choice record, or ({}, None)
    for an open-answer one. Mirrors reference_sections' fallback style:
    evaluate_predictions.py's enrich_records_with_references attaches
    reference_options/reference_correct_option freshly at evaluation time
    (the trimmed mixed dataset, data/processed/guiasalud_casimedicos/*.jsonl,
    no longer carries a standalone correct_option field, it is folded into
    short_answer as "<option key>. <option text>" instead, see that
    function's own comment), falling back to a record that already carries
    options/correct_option directly (e.g. scripts/patch_mc_accuracy.py's own
    older ref_index-merge convention, or a dataset predating this schema)."""
    options = record.get("reference_options") or record.get("options") or {}
    correct_option = record.get("reference_correct_option")
    if correct_option is None:
        correct_option = record.get("correct_option") or record.get("correct_answer")
    return options, correct_option


SECTION_NAMES = ("short_answer", "evidence")
SUMMARY_NAMES = (*SECTION_NAMES, "overall")


def mean_metric(values: Iterable[Optional[float]]) -> Optional[float]:
    return mean_or_none(values)


def overall_section_score(section_metrics: Mapping[str, Mapping[str, Optional[float]]], metric: str) -> Optional[float]:
    return mean_metric(section_metrics[name].get(metric) for name in SECTION_NAMES)


def metric_delta(
    after: Mapping[str, Mapping[str, Optional[float]]],
    before: Mapping[str, Mapping[str, Optional[float]]],
    metric_names: Iterable[str],
) -> dict[str, dict[str, Optional[float]]]:
    delta: dict[str, dict[str, Optional[float]]] = {}
    for section in SUMMARY_NAMES:
        delta[section] = {}
        for metric in metric_names:
            after_value = after.get(section, {}).get(metric)
            before_value = before.get(section, {}).get(metric)
            if after_value is None or before_value is None:
                delta[section][metric] = None
            else:
                delta[section][metric] = after_value - before_value
    return delta


def evaluate_records(
    records: list[Mapping[str, Any]],
    *,
    run_metadata: Optional[Mapping[str, Any]] = None,
    semantic_model: Optional[str] = None,
    semantic_batch_size: int = 32,
    bertscore_model: Optional[str] = None,
    bertscore_batch_size: int = 16,
    bertscore_lang: str = "es",
    bertscore_device: Optional[str] = None,
    enable_ragas: bool = False,
) -> dict[str, Any]:
    reference_section_values = [reference_sections(record) for record in records]
    warnings = []

    references_by_section = {
        name: [section.get(name, "") for section in reference_section_values] for name in SECTION_NAMES
    }

    if enable_ragas:
        warnings.append(
            "RAGAS skipped: ragas is not installed/configured in this environment. "
            "Use RAGAS with an evaluator LLM and embeddings for faithfulness, answer relevancy, "
            "context precision, and context recall."
        )

    metric_names = (
        "token_precision",
        "token_recall",
        "token_overlap_f1",
        "rouge_l_f1",
        "cosine_similarity",
        "answer_context_token_f1",
        "gold_context_token_f1",
        "bertscore_f1",
    )

    def score_prediction_sections(
        prediction_sections: list[Mapping[str, str]],
        *,
        warning_prefix: str = "",
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        predictions_by_section = {
            name: [section.get(name, "") for section in prediction_sections] for name in SECTION_NAMES
        }

        cosine_scores_by_section: dict[str, list[Optional[float]]] = {
            name: [None] * len(records) for name in SECTION_NAMES
        }
        if semantic_model:
            try:
                for name in SECTION_NAMES:
                    cosine_scores_by_section[name] = cosine_similarity_score(
                        predictions_by_section[name],
                        references_by_section[name],
                        model_name=semantic_model,
                        batch_size=semantic_batch_size,
                    )
            except Exception as exc:  # pragma: no cover - depends on local model cache/network.
                prefix = f"{warning_prefix} " if warning_prefix else ""
                warnings.append(f"{prefix}cosine_similarity skipped: {exc}")

        bert_scores_by_section: dict[str, list[Optional[float]]] = {
            name: [None] * len(records) for name in SECTION_NAMES
        }
        if bertscore_model:
            try:
                for name in SECTION_NAMES:
                    bert_scores_by_section[name] = bertscore_f1(
                        predictions_by_section[name],
                        references_by_section[name],
                        model_name=bertscore_model,
                        batch_size=bertscore_batch_size,
                        lang=bertscore_lang,
                        device=bertscore_device,
                    )
            except Exception as exc:  # pragma: no cover - optional dependency/model.
                prefix = f"{warning_prefix} " if warning_prefix else ""
                warnings.append(f"{prefix}bertscore_f1 skipped: {exc}")

        scored_rows = []
        mc_scores: list[Optional[float]] = []
        for idx, record in enumerate(records):
            context = context_text(record)
            row: dict[str, Any] = {"id": record.get("id")}
            section_metrics: dict[str, dict[str, Optional[float]]] = {}
            for name in SECTION_NAMES:
                prediction = prediction_sections[idx].get(name, "")
                reference = reference_section_values[idx].get(name, "")
                token_scores = token_prf(prediction, reference)
                metrics = {
                    "token_precision": percent(token_scores["precision"]),
                    "token_recall": percent(token_scores["recall"]),
                    "token_overlap_f1": percent(token_scores["f1"]),
                    "rouge_l_f1": percent(rouge_l_f1(prediction, reference)),
                    "cosine_similarity": percent(cosine_scores_by_section[name][idx]),
                    "answer_context_token_f1": percent(token_f1(prediction, context)) if context else None,
                    "gold_context_token_f1": percent(token_f1(reference, context)) if context else None,
                }
                if bertscore_model:
                    metrics["bertscore_f1"] = percent(bert_scores_by_section[name][idx])
                else:
                    metrics["bertscore_f1"] = None
                section_metrics[name] = metrics
                row[name] = metrics
            # mc_accuracy is only meaningful for the short_answer section (an
            # option choice, not free text) and only defined for records that
            # carry options (CasiMedicos, via reference_options_and_correct's
            # own fallback chain); it is None for open-answer (GuiaSalud)
            # records, the same "undefined, not 0" convention every other
            # per-record metric here already uses when a section is empty.
            options, correct_option = reference_options_and_correct(record)
            row_mc = mc_accuracy(prediction_sections[idx].get("short_answer", ""), options, correct_option)
            row["short_answer"]["mc_accuracy"] = percent(row_mc) if row_mc is not None else None
            row["evidence"]["mc_accuracy"] = None
            mc_scores.append(row_mc)
            row["overall"] = {
                metric: overall_section_score(section_metrics, metric) for metric in metric_names
            }
            row["overall"]["mc_accuracy"] = percent(row_mc) if row_mc is not None else None
            scored_rows.append(row)

        scored_summary: dict[str, Any] = {"num_examples": len(records)}
        for name in SUMMARY_NAMES:
            scored_summary[name] = {
                metric: mean_or_none(row[name][metric] for row in scored_rows) for metric in metric_names
            }
        mc_summary_value = percent(mean_or_none(mc_scores)) if any(v is not None for v in mc_scores) else None
        scored_summary["short_answer"]["mc_accuracy"] = mc_summary_value
        scored_summary["evidence"]["mc_accuracy"] = None
        scored_summary["overall"]["mc_accuracy"] = mc_summary_value
        return scored_summary, scored_rows

    final_prediction_sections = [parsed_prediction_sections(record) for record in records]
    summary, rows = score_prediction_sections(final_prediction_sections)
    for row, record in zip(rows, records):
        row["cost"] = sample_cost(record)
    summary["cost"] = summarize_cost(records, run_metadata)

    has_self_feedback = bool(run_metadata.get("self_feedback")) or any(
        record.get("parsed_initial_prediction") or record.get("initial_prediction_text") for record in records
    )
    if has_self_feedback:
        initial_prediction_sections = [
            parsed_prediction_sections(
                record,
                parsed_key="parsed_initial_prediction",
                text_key="initial_prediction_text",
            )
            for record in records
        ]
        before_summary, before_rows = score_prediction_sections(
            initial_prediction_sections,
            warning_prefix="before_feedback",
        )
        after_summary = {
            "num_examples": summary["num_examples"],
            **{name: summary[name] for name in SUMMARY_NAMES},
        }
        delta_summary = metric_delta(after_summary, before_summary, metric_names)
        summary["before_feedback"] = before_summary
        summary["after_feedback"] = after_summary
        summary["self_feedback_delta"] = delta_summary
        for row, before_row in zip(rows, before_rows):
            before_metrics = {name: before_row[name] for name in SUMMARY_NAMES}
            after_metrics = {name: row[name] for name in SUMMARY_NAMES}
            row["before_feedback"] = before_metrics
            row["self_feedback_delta"] = metric_delta(after_metrics, before_metrics, metric_names)

    return {"summary": summary, "rows": rows, "warnings": warnings}


def write_metrics(metrics: Mapping[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
