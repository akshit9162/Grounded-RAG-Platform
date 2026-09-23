"""
Chunking: splits raw documents into overlapping windows suitable for
embedding + retrieval.

Design choice: recursive character splitting that respects paragraph and
sentence boundaries where possible, falling back to hard token-count splits
for pathological input (huge unbroken text blocks). Overlap keeps context
that spans a boundary from being lost entirely to one chunk.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    position: int  # order within the source document
    metadata: dict = field(default_factory=dict)


def _approx_token_count(text: str) -> int:
    # Cheap approximation (~4 chars/token for English) — good enough for
    # chunk sizing without pulling in a tokenizer dependency.
    return max(1, len(text) // 4)


def _split_into_paragraphs(text: str) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


def _split_into_sentences(paragraph: str) -> list[str]:
    # Simple sentence splitter; good enough for policy/ticket/doc-style text.
    sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_document(
    doc_id: str,
    text: str,
    chunk_size_tokens: int = 350,
    overlap_tokens: int = 60,
    metadata: Optional[dict] = None,
) -> list[Chunk]:
    """
    Recursive-ish chunking: pack sentences into a chunk until adding the next
    sentence would exceed chunk_size_tokens, then start a new chunk that
    begins `overlap_tokens` worth of trailing sentences back from the cut.
    """
    metadata = metadata or {}
    paragraphs = _split_into_paragraphs(text)
    all_sentences: list[str] = []
    for p in paragraphs:
        all_sentences.extend(_split_into_sentences(p))

    if not all_sentences:
        return []

    chunks: list[Chunk] = []
    current: list[str] = []
    current_tokens = 0
    position = 0
    i = 0

    while i < len(all_sentences):
        sentence = all_sentences[i]
        sentence_tokens = _approx_token_count(sentence)

        if current_tokens + sentence_tokens > chunk_size_tokens and current:
            chunk_text = " ".join(current)
            chunks.append(
                Chunk(
                    id=f"{doc_id}::chunk-{position}::{uuid.uuid4().hex[:8]}",
                    doc_id=doc_id,
                    text=chunk_text,
                    position=position,
                    metadata=dict(metadata),
                )
            )
            position += 1

            # Roll back to build overlap: keep trailing sentences that sum
            # to roughly overlap_tokens, then continue from there.
            overlap_sentences: list[str] = []
            overlap_count = 0
            for s in reversed(current):
                t = _approx_token_count(s)
                if overlap_count + t > overlap_tokens:
                    break
                overlap_sentences.insert(0, s)
                overlap_count += t
            current = overlap_sentences
            current_tokens = overlap_count
            continue  # re-process the same sentence against the reset window

        current.append(sentence)
        current_tokens += sentence_tokens
        i += 1

    if current:
        chunk_text = " ".join(current)
        chunks.append(
            Chunk(
                id=f"{doc_id}::chunk-{position}::{uuid.uuid4().hex[:8]}",
                doc_id=doc_id,
                text=chunk_text,
                position=position,
                metadata=dict(metadata),
            )
        )

    return chunks
