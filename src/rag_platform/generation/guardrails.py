"""
Guardrails: verify a generated answer is actually grounded in the retrieved
chunks before it's returned to a user.

Two checks:
1. Citation presence — does the answer cite at least one source excerpt?
   An ungrounded answer with zero citations is an immediate red flag.
2. Lexical faithfulness — do the answer's claims share enough vocabulary
   with the cited excerpts to plausibly be derived from them? This is a
   cheap, fast, dependency-light proxy for the same idea RAGAS's
   `faithfulness` metric measures more rigorously offline (see
   evaluation/ragas_eval.py) — this one runs synchronously, in the request
   path, to decide whether to flag an answer before it ships.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rag_platform.config import settings
from rag_platform.text_utils import meaningful_tokens

CITATION_PATTERN = re.compile(r"\[(\d+)\]")
REFUSAL_PHRASE = "i don't have enough information"


@dataclass
class GuardrailResult:
    is_grounded: bool
    cited_indices: list[int]
    faithfulness_score: float
    is_refusal: bool
    reason: str
    unsupported_sentences: list[str]


def check_answer(answer: str, chunks: list[dict]) -> GuardrailResult:
    is_refusal = REFUSAL_PHRASE in answer.lower()
    cited_indices = sorted({int(m) for m in CITATION_PATTERN.findall(answer)})

    if is_refusal:
        return GuardrailResult(
            is_grounded=True,  # an honest refusal is a correctly grounded response
            cited_indices=[],
            faithfulness_score=1.0,
            is_refusal=True,
            reason="model declined to answer due to insufficient context",
            unsupported_sentences=[],
        )

    if not cited_indices:
        return GuardrailResult(
            is_grounded=False,
            cited_indices=[],
            faithfulness_score=0.0,
            is_refusal=False,
            reason="answer contains no citations to retrieved sources",
            unsupported_sentences=[],
        )

    cited_text = " ".join(
        chunks[i - 1]["payload"]["text"] for i in cited_indices if 0 < i <= len(chunks)
    )
    source_tokens = meaningful_tokens(cited_text)

    # Whole-answer overlap (kept for backward compatibility / the headline score).
    answer_tokens = meaningful_tokens(answer)
    faithfulness_score = (
        len(answer_tokens & source_tokens) / len(answer_tokens) if answer_tokens else 0.0
    )

    # Per-sentence check — this is the part that actually catches hallucination.
    # A long, mostly-grounded answer can average out one or two fabricated
    # sentences under whole-answer bag-of-words overlap (a specific, invented
    # claim can still share enough generic domain vocabulary — "refund",
    # "days", "order" — with the real source to look fine on average). Scoring
    # each sentence separately catches a sentence that's floating almost
    # entirely on invented specifics, even inside an otherwise-grounded answer.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", answer) if s.strip()]
    unsupported_sentences: list[str] = []
    for sentence in sentences:
        s_tokens = meaningful_tokens(sentence)
        if len(s_tokens) < 3:
            continue  # headers/short fragments — not enough signal either way
        s_overlap = len(s_tokens & source_tokens) / len(s_tokens)
        if s_overlap < settings.sentence_faithfulness_floor:
            unsupported_sentences.append(sentence)

    is_grounded = (
        faithfulness_score >= settings.faithfulness_threshold and not unsupported_sentences
    )
    if unsupported_sentences:
        reason = "one or more sentences contain claims not supported by the cited sources — possible hallucination"
    elif is_grounded:
        reason = "answer sufficiently overlaps cited sources"
    else:
        reason = "answer's vocabulary diverges too far from cited sources — possible hallucination"

    return GuardrailResult(
        is_grounded=is_grounded,
        cited_indices=cited_indices,
        faithfulness_score=round(faithfulness_score, 3),
        is_refusal=False,
        reason=reason,
        unsupported_sentences=unsupported_sentences,
    )
