"""Reasoning pipelines layered on top of the frozen best RAG configuration.

Four pipelines, all inference-only (no fine-tuning), all reusing the same
retriever + reranker as the winning single-pass RAG row of the dev ablation:

    structured_cot   MedCoT-RAG-style structured clinical chain of thought
                     (wangEtAl2025). Retrieval is unchanged, only the prompt
                     imposes a diagnosis-shaped reasoning scaffold.

    thought_rag      RAR2 (xuEtAl2025) Parallel Scaling, tuning-free (Fig. 4):
                     sample `parallel_thoughts` independent thought processes,
                     concatenate them into one extended thought, retrieve ONCE
                     with the concatenation, answer ONCE.

    thought_rag_iter RAR2 iterative test-time scaling: thought -> retrieve ->
                     re-think with the retrieved evidence -> retrieve again,
                     for `rounds` rounds, then answer over the union of docs.

    marag            MA-RAG (wuEtAl2026), re-implemented against the actual
                     NJU-RL/MA-RAG code (ma_rag_entropy.py), not just the
                     paper's prose: a Solver samples several candidates each
                     round, conflict among them is measured, conflicted
                     records turn that conflict into up to four retrieval
                     queries (the original's own multi-query retrieval
                     agent), all candidates, re-sorted by confidence, are
                     carried into the next round's Solver prompt as history
                     (the original has no ranking/pruning agent, an earlier
                     draft of this pipeline added one, which was not a
                     faithful reproduction of the original and has been
                     removed), a final synthesis pass resolves any record
                     that never reaches unanimity once the round budget is
                     exhausted (the original instead takes the last round's
                     plurality vote, which has no equivalent on this thesis's
                     open-answer half, so synthesis is kept as the
                     open-answer-compatible resolution mechanism).

thought_rag_iter is the one pipeline still without a from-code faithfulness
pass: RAR2's Iterative Scaling strategy has no published code (the paper
gives no repository), so it is implemented directly from Section 4.3's own
description, and no further deviation was found there worth revising.

The MA-RAG adaptation is the one real deviation from a published method and is
deliberate: the paper detects conflict as exact disagreement between candidate
letters on multiple-choice benchmarks. Half of our dev set (SNS1064) is
open-answer, where that signal does not exist. `conflict_score` therefore falls
back to semantic disagreement (1 - mean pairwise cosine over the candidates'
short answers, embedded with the same multilingual-e5-large already loaded for
retrieval) whenever a record carries no answer options.
"""
from __future__ import annotations

import re
from itertools import combinations
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from medical_rag_thesis.prompts import (
    format_context_text,
    format_examples,
    format_options,
    format_question,
    parse_answer_sections,
)

PIPELINES = ("structured_cot", "thought_rag", "thought_rag_iter", "marag")


# --------------------------------------------------------------------------
# Language-specific surface strings
# --------------------------------------------------------------------------

LABELS = {
    "es": {
        "question": "Pregunta",
        "options": "Opciones",
        "context": "Contexto extraído",
        "answer": "Respuesta corta",
        "evidence": "Evidencia",
        "thought": "Razonamiento",
        "reply_in": "Responde en español.",
    },
    "eu": {
        "question": "Galdera",
        "options": "Aukerak",
        "context": "Erauzitako testuingurua",
        "answer": "Erantzun laburra",
        "evidence": "Ebidentzia",
        "thought": "Arrazoiketa",
        "reply_in": "Erantzun euskaraz.",
    },
}


def labels(language: str) -> Mapping[str, str]:
    return LABELS.get(language, LABELS["es"])


def question_of(record: Mapping[str, Any], language: str) -> str:
    lab = labels(language)
    return format_question(record, language=language).removeprefix(f"{lab['question']}: ").strip()


def output_format_block(record: Mapping[str, Any], language: str) -> str:
    """The exact two-field output contract used by every run in this thesis.

    Kept byte-identical in spirit to prompts.build_extractive_user_prompt so the
    parser (parse_answer_sections) and therefore the metrics stay comparable
    with the single-pass RAG rows.
    """
    lab = labels(language)
    has_options = bool(format_options(record))
    if language == "eu":
        answer_placeholder = "aukeraren zenbakia. erantzun laburra" if has_options else "erantzun laburra"
        return (
            "Erantzun honako formatuan, testurik gehitu gabe adierazitako eremuetatik kanpo:\n\n"
            f"{lab['answer']}:\n<{answer_placeholder}>\n\n"
            f"{lab['evidence']}:\n<ebidentzia>\n\n"
            "< > sinboloek leku-markak mugatzen dituzte. Ordezkatu leku-markak eta haien "
            "muga-sinboloak espero den edukiarekin."
        )
    answer_placeholder = "número de opción. respuesta breve" if has_options else "respuesta breve"
    return (
        "Responde en este formato, sin incluir texto fuera de los campos indicados:\n\n"
        f"{lab['answer']}:\n<{answer_placeholder}>\n\n"
        f"{lab['evidence']}:\n<evidencia>\n\n"
        "Los símbolos < > delimitan marcadores de posición. Sustituye esos marcadores y sus "
        "símbolos delimitadores por el contenido esperado."
    )


def option_rule(record: Mapping[str, Any], language: str) -> str:
    if not format_options(record):
        return ""
    if language == "eu":
        return (
            '- Erantzun-aukerak badaude, "Erantzun laburra" atalean adierazi bakarrik '
            'aukeratutako zenbakia (adib. "3.") eta aukera horren testua.\n'
        )
    return (
        '- Si hay opciones de respuesta, en el apartado de "Respuesta corta" incluye solo el '
        'número de la opción elegida (ej. "3.") y el texto de esa opción.\n'
    )


def _question_and_options(record: Mapping[str, Any], language: str) -> list[str]:
    lab = labels(language)
    sections = [f"{lab['question']}:\n" + question_of(record, language)]
    options = format_options(record)
    if options:
        sections.append(f"{lab['options']}:\n" + options)
    return sections


def _context_section(documents: Sequence[Mapping[str, Any]], language: str) -> list[str]:
    lab = labels(language)
    context_text = format_context_text(documents or [], language=language)
    if not context_text:
        return []
    return [f"{lab['context']}:\n" + context_text]


FEW_SHOT_INTRO = {
    "es": "Usa estos ejemplos solo para aprender el formato de salida:",
    "eu": "Erabili adibide hauek soilik irteera-formatua ikasteko:",
}


def _few_shot_section(
    examples: Optional[Sequence[Mapping[str, Any]]], language: str
) -> list[str]:
    """Same demonstrations, same intro line, as prompts.build_extractive_user_prompt
    (row 8's "3-shot + <base>" ablation row), only for the pipeline's final,
    answer-emitting call, since the demonstrations teach output *format*, and the
    intermediate reasoning/ranking/query calls in these pipelines don't emit that
    format at all."""
    if not examples:
        return []
    intro = FEW_SHOT_INTRO.get(language, FEW_SHOT_INTRO["es"])
    return [intro + "\n\n" + format_examples(examples, language=language)]


# --------------------------------------------------------------------------
# 1. Structured clinical chain of thought (MedCoT-RAG)
# --------------------------------------------------------------------------

# The step names deliberately avoid the words the answer parser keys on
# ("Evidencia"/"Ebidentzia", "Respuesta corta"/"Erantzun laburra"). An earlier
# draft used "Evidencia" as the name of step 3 and the model duly emitted it as a
# heading, so parse_answer_sections captured a *reasoning step* as the evidence
# field and the whole chain of thought as the short answer. The reasoning is now
# fenced under its own "Razonamiento"/"Arrazoiketa" heading, which cannot collide.
STRUCTURED_COT_STEPS = {
    "es": (
        "Antes de responder, razona siguiendo estos pasos clínicos, bajo el "
        "encabezado \"Razonamiento\":\n"
        "1. Síntomas: identifica los hallazgos clave para responder a la pregunta.\n"
        "2. Mecanismo: explica la fisiopatología causal.\n"
        "3. Diferencial: compara diagnósticos alternativos.\n"
        "4. Justificación: cita documentos extraídos que sirvan para responder a la pregunta y concluye.\n"
        "Después del razonamiento, escribe SIEMPRE los campos finales \"Respuesta corta\" y "
        "\"Evidencia\", aunque el razonamiento ya los anticipe. No uses esos dos nombres como "
        "títulos dentro del razonamiento."
    ),
    "eu": (
        "Erantzun aurretik, arrazoitu urrats kliniko hauei jarraituz, "
        "\"Arrazoiketa\" izenburupean:\n"
        "1. Sintomak: identifikatu galderari erantzuteko funtsezko aurkikuntzak.\n"
        "2. Mekanismoa: azaldu fisiopatologia kausala.\n"
        "3. Diferentziala: konparatu diagnostiko alternatiboak.\n"
        "4. Justifikazioa: aipatu galderari erantzuteko balio duten erauzitako dokumentuak eta ondorioztatu.\n"
        "Arrazoiketaren ondoren, idatzi BETI \"Erantzun laburra\" eta \"Ebidentzia\" azken eremuak, "
        "arrazoiketak jada aurreratzen baditu ere. Ez erabili bi izen horiek arrazoiketaren barruko "
        "izenburu gisa."
    ),
}

STRUCTURED_COT_FORMAT = {
    "es": (
        "Responde exactamente con esta estructura:\n\n"
        "Razonamiento:\n<los cuatro pasos>\n\n"
        "Respuesta corta:\n<{answer}>\n\n"
        "Evidencia:\n<evidencia>"
    ),
    "eu": (
        "Erantzun zehazki egitura honekin:\n\n"
        "Arrazoiketa:\n<lau urratsak>\n\n"
        "Erantzun laburra:\n<{answer}>\n\n"
        "Ebidentzia:\n<ebidentzia>"
    ),
}


def build_structured_cot_prompt(
    record: Mapping[str, Any],
    documents: Sequence[Mapping[str, Any]],
    language: str = "es",
    *,
    examples: Optional[Sequence[Mapping[str, Any]]] = None,
) -> str:
    lab = labels(language)
    if language == "eu":
        header = "Zure ataza galdera medikoari modu justifikatuan erantzutea da."
        rules = (
            "Arauak:\n"
            "- Oinarritu erantzuna erauzitako testuinguruko informazioan. Testuingurua nahikoa ez bada, "
            "konbinatu zure medikuntza-ezagutzarekin.\n"
            "- EZ asmatu testuinguruak edo zure medikuntza-ezagutzak babesten ez duen daturik.\n"
            f"{option_rule(record, language)}"
            f"- {lab['reply_in']}"
        )
    else:
        header = "Tu tarea es responder a la pregunta médica de forma justificada."
        rules = (
            "Reglas:\n"
            "- Basa tu respuesta en la información del contexto extraído. Si el contexto es insuficiente, "
            "combínalo con tu conocimiento médico.\n"
            "- NO inventes datos que no estén respaldados por el contexto extraído o por tu conocimiento médico.\n"
            f"{option_rule(record, language)}"
            f"- {lab['reply_in']}"
        )
    sections = [header, rules, STRUCTURED_COT_STEPS.get(language, STRUCTURED_COT_STEPS["es"])]
    sections += _few_shot_section(examples, language)
    sections += _context_section(documents, language)
    sections += _question_and_options(record, language)
    has_options = bool(format_options(record))
    if language == "eu":
        answer_placeholder = "aukeraren zenbakia. erantzun laburra" if has_options else "erantzun laburra"
    else:
        answer_placeholder = "número de opción. respuesta breve" if has_options else "respuesta breve"
    template = STRUCTURED_COT_FORMAT.get(language, STRUCTURED_COT_FORMAT["es"])
    sections.append(template.format(answer=answer_placeholder))
    return "\n\n".join(sections)


# --------------------------------------------------------------------------
# 2/3. RAR2: reasoning-augmented retrieval
# --------------------------------------------------------------------------

THOUGHT_INSTRUCTION = {
    "es": (
        "Razona paso a paso para identificar el conocimiento médico que puede estar implicado en la "
        "pregunta: qué entidades clínicas, mecanismos, pruebas o tratamientos habría que consultar para "
        "responderla. NO des todavía la respuesta final; escribe solo ese razonamiento."
    ),
    "eu": (
        "Arrazoitu urratsez urrats galderan inplikatuta egon daitekeen medikuntza-ezagutza identifikatzeko: "
        "zein entitate kliniko, mekanismo, proba edo tratamendu kontsultatu beharko liratekeen erantzuteko. "
        "EZ eman oraindik azken erantzuna; idatzi arrazoiketa hori soilik."
    ),
}

THOUGHT_REFINE_INSTRUCTION = {
    "es": (
        "Este es tu razonamiento previo y la evidencia recuperada con él. Revísalo: corrige lo que la "
        "evidencia contradiga y señala qué conocimiento médico sigue faltando para responder. NO des "
        "todavía la respuesta final; escribe solo el razonamiento actualizado."
    ),
    "eu": (
        "Hau da zure aurreko arrazoiketa eta harekin berreskuratutako ebidentzia. Berrikusi ezazu: zuzendu "
        "ebidentziak kontraesaten duena eta adierazi zein medikuntza-ezagutza falta den oraindik erantzuteko. "
        "EZ eman oraindik azken erantzuna; idatzi arrazoiketa eguneratua soilik."
    ),
}


def build_thought_prompt(
    record: Mapping[str, Any],
    language: str = "es",
    *,
    previous_thought: str = "",
    documents: Optional[Sequence[Mapping[str, Any]]] = None,
) -> str:
    """A thought process, generated *before* any retrieval, shared by both
    thought_rag (Parallel Scaling: m independent, zero-shot samples of this
    same prompt) and thought_rag_iter's first round.

    With `previous_thought` and `documents` this becomes thought_rag_iter's
    re-think step (RAR2's "Iterative Scaling"), thought_rag never calls it
    that way, since Parallel Scaling never re-thinks with retrieved evidence.
    """
    lab = labels(language)
    refining = bool(previous_thought)
    instruction = (THOUGHT_REFINE_INSTRUCTION if refining else THOUGHT_INSTRUCTION)[
        language if language in THOUGHT_INSTRUCTION else "es"
    ]
    sections = [instruction]
    sections += _question_and_options(record, language)
    if refining:
        sections += _context_section(documents or [], language)
        sections.append(f"{lab['thought']} previo:\n" + previous_thought if language == "es"
                        else f"Aurreko {lab['thought'].lower()}:\n" + previous_thought)
    sections.append(f"{lab['thought']}:")
    return "\n\n".join(sections)


def build_thought_answer_prompt(
    record: Mapping[str, Any],
    documents: Sequence[Mapping[str, Any]],
    thought: str,
    language: str = "es",
    *,
    examples: Optional[Sequence[Mapping[str, Any]]] = None,
) -> str:
    """Answer given a thought process (thought_rag's concatenation of m
    parallel thoughts, or thought_rag_iter's final round's thought) *and* the
    evidence retrieved with it."""
    lab = labels(language)
    if language == "eu":
        header = "Zure ataza galdera medikoari modu justifikatuan erantzutea da."
        rules = (
            "Arauak:\n"
            "- Erabili zure aurreko arrazoiketa gida gisa, baina oinarritu erantzuna erauzitako testuinguruan.\n"
            "- Arrazoiketak testuinguruak babesten ez duen zerbait badio, lehenetsi testuingurua.\n"
            "- EZ asmatu daturik.\n"
            f"{option_rule(record, language)}"
            f"- {lab['reply_in']}"
        )
    else:
        header = "Tu tarea es responder a la pregunta médica de forma justificada."
        rules = (
            "Reglas:\n"
            "- Usa tu razonamiento previo como guía, pero basa la respuesta en el contexto extraído.\n"
            "- Si el razonamiento afirma algo que el contexto no respalda, prioriza el contexto.\n"
            "- NO inventes datos.\n"
            f"{option_rule(record, language)}"
            f"- {lab['reply_in']}"
        )
    sections = [header, rules]
    sections += _few_shot_section(examples, language)
    sections += _context_section(documents, language)
    sections += _question_and_options(record, language)
    sections.append(f"{lab['thought']}:\n" + thought.strip())
    sections.append(output_format_block(record, language))
    return "\n\n".join(sections)


def concatenate_thoughts(thoughts: Sequence[str]) -> str:
    """RAR2 Parallel Scaling (xuEtAl2025 Fig. 4): join m independently-sampled
    thought processes into the single extended thought thought_rag retrieves
    with, the paper's "+" ("Thought 1 + Thought 2 + ... + Thought m")."""
    return "\n\n".join(t.strip() for t in thoughts if t and t.strip())


# --------------------------------------------------------------------------
# 4. MA-RAG: conflict -> retrieval -> ranking-free consensus
# --------------------------------------------------------------------------

# Matches the original's system_prompt_query (ma_rag_entropy.py /
# ma_rag_evaluator.py), which asks for "1-4 precise queries", each on its own
# "[Query N] ..." line, parsed back out with a regex over that exact tag,
# translated into Spanish and Basque (the original is English-only,
# mirroring MA-RAG's own MedCorp/BM25 setup, which has no Spanish/Basque
# equivalent here either).
CONFLICT_QUERY_INSTRUCTION = {
    "es": (
        "Eres un experto médico. Te voy a dar varias respuestas distintas a la misma pregunta médica, "
        "que pueden ser incorrectas. Tu tarea es identificar contradicciones, ambigüedades y los puntos "
        "concretos de desacuerdo entre las respuestas, y extraer consultas de conceptos clave para "
        "buscarlas y verificarlas.\n\n"
        "Pasos:\n"
        "1. Analiza las respuestas distintas y resume en qué difieren.\n"
        "2. Extrae palabras clave de esas diferencias para la búsqueda.\n"
        "3. Genera de 1 a 4 consultas precisas para la búsqueda.\n\n"
        "Formato de salida (SOLO las consultas, sin explicaciones):\n"
        "[Consulta 1] xxx\n"
        "[Consulta 2] xxx\n"
        "(más consultas si hacen falta)"
    ),
    "eu": (
        "Aditu medikoa zara. Galdera mediko beraren hainbat erantzun ezberdin emango dizkizut, okerrak "
        "izan daitezkeenak. Zure zeregina erantzunen arteko kontraesanak, anbiguotasunak eta desadostasun "
        "puntu zehatzak identifikatzea da, eta kontzeptu gakoen kontsultak erauztea haiek bilatu eta "
        "egiaztatzeko.\n\n"
        "Urratsak:\n"
        "1. Aztertu erantzun ezberdinak eta laburtu zertan desberdintzen diren.\n"
        "2. Erauzi desberdintasun horietako gako-hitzak bilaketarako.\n"
        "3. Sortu 1 eta 4 kontsulta zehatz artean bilaketarako.\n\n"
        "Irteera-formatua (SOILIK kontsultak, azalpenik gabe):\n"
        "[Kontsulta 1] xxx\n"
        "[Kontsulta 2] xxx\n"
        "(kontsulta gehiago behar izanez gero)"
    ),
}

QUERY_TAG_RE = re.compile(r"\[(?:Consulta|Kontsulta|Query)\s*\d+\]\s*(.*?)\s*$", re.MULTILINE | re.IGNORECASE)


def build_conflict_query_prompt(
    record: Mapping[str, Any],
    candidates: Sequence[str],
    language: str = "es",
) -> str:
    """Retrieval agent. Turns *conflict between candidates* into up to four
    actionable queries, matches the original's system_prompt_query +
    user_prompt_query (ma_rag_entropy.py)."""
    instruction = CONFLICT_QUERY_INSTRUCTION.get(language, CONFLICT_QUERY_INSTRUCTION["es"])
    candidate_label = "Hautagaia" if language == "eu" else "Candidata"
    sections = [instruction]
    sections += _question_and_options(record, language)
    for idx, candidate in enumerate(candidates, start=1):
        sections.append(f"{candidate_label} {idx}:\n{candidate.strip()}")
    return "\n\n".join(sections)


def parse_conflict_queries(text: str, *, fallback: str) -> list[str]:
    """Up to 4 deduplicated queries from the [Consulta N]/[Kontsulta N]/[Query
    N] tags, mirroring the original's `re.findall(r"\\[Query .*?\\](.*?)$", ...)`
    + `set()`-dedup. Falls back to the surface question if the model emits no
    tagged line at all (the original has no such fallback, it would run
    retrieval with an empty query list, i.e. skip retrieval that round, but
    silently skipping evidence retrieval because of a formatting slip is worse
    for this thesis's purposes than falling back to the plain question)."""
    matches = [m.strip() for m in QUERY_TAG_RE.findall(text or "") if m.strip()]
    queries = list(dict.fromkeys(matches))  # de-dup, order-preserving
    return queries[:4] if queries else [fallback]


# The original's user_prompt_round (ma_rag_entropy.py) feeds back every one of
# the N previous candidates, each labelled with its confidence, sorted from
# lowest confidence (highest entropy) to highest, "sorted by their
# confidence (low entropy means high confidence)" per the prompt's own text,
# and the loop `for i, (answer, answer_entropy) in enumerate(zip(...), start=1)`
# in the source confirms the iteration order is the sort order, ascending
# confidence. There is no ranking/pruning agent in the original at all.
SOLVER_HISTORY_LABEL = {
    "es": "Estas son las respuestas candidatas de la ronda anterior (ordenadas de menor a mayor confianza; "
          "la confianza no garantiza que la respuesta sea correcta):",
    "eu": "Hauek dira aurreko txandako erantzun hautagaiak (konfiantza gutxienetik gehienera ordenatuta; "
          "konfiantzak ez du bermatzen erantzuna zuzena denik):",
}


def build_solver_prompt(
    record: Mapping[str, Any],
    documents: Sequence[Mapping[str, Any]],
    language: str = "es",
    *,
    history: Sequence[tuple[str, float]] = (),
    examples: Optional[Sequence[Mapping[str, Any]]] = None,
) -> str:
    """Solver agent. `history` is [(candidate_text, confidence), ...] for ALL
    of the previous round's candidates, already sorted ascending by
    confidence (matching the original's own iteration order), not a single
    pruned "best" trace."""
    lab = labels(language)
    if language == "eu":
        header = "Zure ataza galdera medikoari modu justifikatuan erantzutea da."
        rules = (
            "Arauak:\n"
            "- Oinarritu erantzuna erauzitako testuinguruan; nahikoa ez bada, konbinatu zure "
            "medikuntza-ezagutzarekin.\n"
            "- EZ asmatu daturik.\n"
            f"{option_rule(record, language)}"
            f"- {lab['reply_in']}"
        )
    else:
        header = "Tu tarea es responder a la pregunta médica de forma justificada."
        rules = (
            "Reglas:\n"
            "- Basa la respuesta en el contexto extraído; si es insuficiente, combínalo con tu "
            "conocimiento médico.\n"
            "- NO inventes datos.\n"
            f"{option_rule(record, language)}"
            f"- {lab['reply_in']}"
        )
    sections = [header, rules]
    sections += _few_shot_section(examples, language)
    sections += _context_section(documents, language)
    sections += _question_and_options(record, language)
    if history:
        label = SOLVER_HISTORY_LABEL.get(language, SOLVER_HISTORY_LABEL["es"])
        entries = [
            f"{idx}. ({'confianza' if language != 'eu' else 'konfiantza'} {confidence:.2f}) {text.strip()}"
            for idx, (text, confidence) in enumerate(history, start=1)
        ]
        sections.append(label + "\n" + "\n\n".join(entries))
    sections.append(output_format_block(record, language))
    return "\n\n".join(sections)


def token_confidence(token_logprobs: Sequence[float]) -> float:
    """Confidence proxy standing in for the original's per-token entropy over
    the top-20 logprobs (ma_rag_entropy.py's process_response/token_entropies):
    this thesis's Generator requests only the sampled token's own logprob from
    vLLM, not a full top-k distribution, so full categorical entropy is not
    available. Mean per-token log probability of the SAMPLED token is used
    instead, both are monotone proxies for "how sure was the model of what
    it actually said", and higher values mean higher confidence in both."""
    if not token_logprobs:
        return 0.0
    return float(np.mean(token_logprobs))


def sort_by_confidence(
    candidates: Sequence[str], confidences: Sequence[float]
) -> list[tuple[str, float]]:
    """Ascending by confidence (least to most confident), matching the
    original's own iteration order in user_prompt_round's construction."""
    paired = list(zip(candidates, confidences))
    paired.sort(key=lambda item: item[1])
    return paired


CONSENSUS_INSTRUCTION = {
    "es": (
        "Estas son las mejores respuestas candidatas a la pregunta, junto con toda la evidencia recuperada. "
        "Escribe la respuesta final de consenso: conserva lo que la evidencia respalda, descarta lo que "
        "contradiga y no inventes datos nuevos."
    ),
    "eu": (
        "Hauek dira galderaren erantzun hautagai onenak, berreskuratutako ebidentzia guztiarekin batera. "
        "Idatzi adostasunezko azken erantzuna: gorde ebidentziak babesten duena, baztertu kontraesaten duena "
        "eta ez asmatu datu berririk."
    ),
}


def build_consensus_prompt(
    record: Mapping[str, Any],
    documents: Sequence[Mapping[str, Any]],
    candidates: Sequence[str],
    language: str = "es",
    *,
    examples: Optional[Sequence[Mapping[str, Any]]] = None,
) -> str:
    """Final synthesis pass, only for records that never reach unanimity once
    the round budget is exhausted. This thesis's open-answer-compatible
    resolution mechanism (the original instead just takes the last round's
    plurality vote, which has no equivalent for open-answer records)."""
    lab = labels(language)
    instruction = CONSENSUS_INSTRUCTION.get(language, CONSENSUS_INSTRUCTION["es"])
    candidate_label = "Hautagaia" if language == "eu" else "Candidata"
    rules = f"{option_rule(record, language)}- {lab['reply_in']}"
    sections = [instruction, ("Arauak:\n" if language == "eu" else "Reglas:\n") + rules]
    sections += _few_shot_section(examples, language)
    sections += _context_section(documents, language)
    sections += _question_and_options(record, language)
    for idx, candidate in enumerate(candidates, start=1):
        sections.append(f"{candidate_label} {idx}:\n{candidate.strip()}")
    sections.append(output_format_block(record, language))
    return "\n\n".join(sections)


# --------------------------------------------------------------------------
# Conflict measurement
# --------------------------------------------------------------------------

OPTION_RE = re.compile(r"^\s*\**\s*\(?([1-9])\s*[\.\)\-:]")


def extract_option(short_answer: str) -> Optional[str]:
    """The answer contract asks multiple-choice records to open with the option
    number ("3. ..."), so a leading digit is the model's choice."""
    match = OPTION_RE.match(short_answer or "")
    return match.group(1) if match else None


# Instruction-tuned models routinely emit "**Respuesta corta:**" or "### Evidencia:"
# rather than the bare "Respuesta corta:" the parser anchors on, which silently
# drops the field. Strip the markdown around label lines before parsing.
_LABEL_WORDS = r"respuesta corta|erantzun laburra|juicio|evidencia|ebidentzia|consideraciones"
MARKDOWN_LABEL_RE = re.compile(
    # Trailing [*_]* after the colon matters: "**Evidencia:** texto" closes its
    # bold *after* the colon, and without this the "**" leaks into the field value.
    rf"(?im)^[ \t]*[*_#>\-]*[ \t]*({_LABEL_WORDS})[ \t]*[*_]*[ \t]*[:：][ \t]*[*_]*[ \t]*"
)
ANSWER_LABEL_RE = re.compile(r"(?im)^[ \t]*(?:respuesta corta|erantzun laburra|juicio)[ \t]*[:：]")


def normalize_answer_labels(text: str) -> str:
    return MARKDOWN_LABEL_RE.sub(lambda m: f"{m.group(1)}:\n", text or "")


def parse_pipeline_answer(text: str) -> dict[str, str]:
    """Parse a pipeline's final answer into the two graded fields.

    Two hardening steps over the shared parser, both needed because these
    pipelines make the model *reason out loud before answering*:

      1. Markdown around the labels is stripped (see MARKDOWN_LABEL_RE).
      2. Parsing starts at the LAST answer label. A chain of thought that
         rehearses "Respuesta corta: ..." mid-reasoning would otherwise be
         captured as the answer, and the real answer block appended to it.

    If the model never emits an answer label at all, we fall back to the shared
    parser (whose own fallback puts the whole text in short_answer), that is a
    genuine format failure and is counted, not hidden.
    """
    normalized = normalize_answer_labels(text)
    matches = list(ANSWER_LABEL_RE.finditer(normalized))
    if matches:
        normalized = normalized[matches[-1].start():]
    return parse_answer_sections(normalized)


def trim_leaked_thinking_prefix(text: str) -> tuple[str, bool]:
    """Return a clean final answer block when a leaked CoT prefix is unambiguous.

    This is deliberately a narrow output-cleanup step, not a repair of a
    model's answer.  Some think-mode pipeline calls emit a long visible
    ``Thinking Process``/``Razonamiento`` preamble and only then provide a
    normal extractive answer block.  In that specific case, the last answer
    label marks the boundary between leaked internal reasoning and the answer
    the model ultimately chose.  Preserve the raw text otherwise: an output
    with no answer label, or without a reasoning marker before the final
    answer block, remains available to evaluation as produced.

    The caller records both the cleanup decision and the original text for
    auditability.  This must never silently invent or regenerate content.
    """
    normalized = normalize_answer_labels(text)
    matches = list(ANSWER_LABEL_RE.finditer(normalized))
    if not matches:
        return text, False
    start = matches[-1].start()
    prefix = normalized[:start]
    leaked_markers = (
        "thinking process",
        "razonamiento:",
        "reasoning:",
        "<think>",
        "</think>",
        "[think]",
        "[/think]",
    )
    if any(marker in prefix.lower() for marker in leaked_markers):
        return normalized[start:].strip(), True
    return text, False


def has_answer_label(text: str) -> bool:
    return bool(ANSWER_LABEL_RE.search(normalize_answer_labels(text)))


def short_answer_of(text: str) -> str:
    parsed = parse_pipeline_answer(text or "")
    return str(parsed.get("short_answer") or "").strip()


def conflict_score(
    candidates: Sequence[str],
    *,
    record: Mapping[str, Any],
    embedder: Any = None,
    query_prefix: str = "",
) -> tuple[float, str]:
    """How much do the sampled candidates disagree? Returns (score in [0,1], mode).

    Multiple-choice records get MA-RAG's own signal: the fraction of candidates
    that do not back the plurality option (0.0 = unanimous). Open-answer records
    have no such discrete signal, so we fall back to semantic disagreement,
    1 - mean pairwise cosine over the candidates' short answers.
    """
    answers = [short_answer_of(text) for text in candidates]
    answers = [answer for answer in answers if answer]
    if len(answers) < 2:
        return 0.0, "singleton"

    if format_options(record):
        options = [extract_option(answer) for answer in answers]
        if all(option is not None for option in options):
            counts: dict[str, int] = {}
            for option in options:
                counts[option] = counts.get(option, 0) + 1
            plurality = max(counts.values())
            return 1.0 - plurality / len(options), "option_disagreement"

    if embedder is None:
        # No embedder available: fall back to exact-string agreement.
        unique = len(set(answers))
        return 0.0 if unique == 1 else 1.0, "exact_match_fallback"

    embeddings = embedder.encode(
        [f"{query_prefix}{answer}" for answer in answers],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    sims = [float(np.dot(embeddings[i], embeddings[j])) for i, j in combinations(range(len(answers)), 2)]
    mean_sim = float(np.mean(sims)) if sims else 1.0
    return max(0.0, min(1.0, 1.0 - mean_sim)), "semantic_disagreement"


def majority_candidate(candidates: Sequence[str], record: Mapping[str, Any]) -> int:
    """Index of the candidate holding the plurality option, else 0. Used only as
    the consensus shortcut when conflict is already below threshold."""
    if not format_options(record):
        return 0
    options = [extract_option(short_answer_of(text)) for text in candidates]
    counts: dict[str, int] = {}
    for option in options:
        if option:
            counts[option] = counts.get(option, 0) + 1
    if not counts:
        return 0
    winner = max(counts.items(), key=lambda item: item[1])[0]
    for idx, option in enumerate(options):
        if option == winner:
            return idx
    return 0


def merge_documents(
    existing: Sequence[Mapping[str, Any]],
    new: Sequence[Mapping[str, Any]],
    *,
    max_docs: int,
) -> list[dict[str, Any]]:
    """Union of retrieved evidence across rounds, de-duplicated by doc_id and
    capped so the growing context cannot silently blow past the window."""
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for document in list(existing) + list(new):
        doc_id = str(document.get("doc_id") or document.get("id") or "")
        key = doc_id or str(document.get("text") or "")[:200]
        if key in seen:
            continue
        seen.add(key)
        merged.append(dict(document))
        if len(merged) >= max_docs:
            break
    for rank, document in enumerate(merged, start=1):
        document["rank"] = rank
    return merged
