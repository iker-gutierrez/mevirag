import unittest

from mevirag.prompts import (
    build_extractive_self_feedback_prompt,
    build_extractive_user_prompt,
    format_context_document,
)


def _query_record() -> dict:
    return {
        "id": "guiasalud_query",
        "query": "- Tema: Tratamiento del TAG\n- Pregunta: ¿Es factible?",
    }


class ContextSerializationTests(unittest.TestCase):
    def test_context_prefers_exact_indexed_passage(self) -> None:
        indexed_text = (
            "Guía de práctica clínica: Guía TAG\n"
            "- Tema: Tratamiento del TAG\n"
            "- Pregunta: ¿Es aceptable?\n"
            "Respuesta corta: Sí, es aceptable.\n"
            "- Evidencia: Los pacientes la aceptan."
        )
        document = {
            "guidebook_title": "Guía TAG",
            "query": "this metadata must not replace the indexed passage",
            "short_answer": "this metadata must not replace the indexed passage",
            "justification": "this metadata must not replace the indexed passage",
            "text": indexed_text,
        }

        self.assertEqual(format_context_document(document), indexed_text)

    def test_context_fallback_keeps_current_answer_bearing_fields(self) -> None:
        document = {
            "guidebook_title": "Guía TAG",
            "query": "- Tema: TAG\n- Pregunta: ¿Es aceptable?",
            "short_answer": "Sí, es aceptable.",
            "justification": "- Evidencia: Los pacientes la aceptan.",
        }

        rendered = format_context_document(document)

        self.assertIn("Guía de práctica clínica: Guía TAG", rendered)
        self.assertIn("- Pregunta: ¿Es aceptable?", rendered)
        self.assertIn("Respuesta corta: Sí, es aceptable.", rendered)
        self.assertIn("- Evidencia: Los pacientes la aceptan.", rendered)

    def test_initial_and_feedback_prompts_receive_full_indexed_passage(self) -> None:
        passage = (
            "Guía de práctica clínica: Guía TAG\n"
            "- Pregunta: ¿Es aceptable?\n"
            "Respuesta corta: Sí, es aceptable.\n"
            "- Evidencia: Los pacientes la aceptan."
        )
        documents = [{"text": passage, "guidebook_title": "Guía TAG"}]

        initial = build_extractive_user_prompt(_query_record(), documents=documents)
        feedback = build_extractive_self_feedback_prompt(
            _query_record(), answer="Respuesta corta: No.\n\nEvidencia: Ninguna.", documents=documents
        )

        self.assertIn(passage, initial)
        self.assertIn(passage, feedback)


if __name__ == "__main__":
    unittest.main()
