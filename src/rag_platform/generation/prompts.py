"""
Prompt templates. The core idea: the model is only allowed to answer from
the supplied chunks, must cite which chunk(s) support each claim, and must
say so explicitly when the chunks don't contain an answer — that last
instruction is what turns "hallucinate confidently" into "say I don't know,"
which the guardrail layer then double-checks rather than trusting blindly.
"""
from __future__ import annotations

SYSTEM_PROMPT = """You are a retrieval-grounded assistant. You will be given
a user question and a set of numbered source excerpts. Rules:

1. Answer ONLY using information found in the excerpts below. Do not use
   outside knowledge, even if you believe it to be true.
2. After every factual claim, cite the excerpt number(s) it came from, like
   this: [1] or [2][3].
3. If the excerpts do not contain enough information to answer the
   question, say exactly: "I don't have enough information in the
   available sources to answer that." Do not guess.
4. Be concise. Do not repeat the excerpts back verbatim at length."""


def build_user_prompt(question: str, chunks: list[dict]) -> str:
    excerpt_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        text = chunk["payload"]["text"]
        source = chunk["payload"].get("doc_id", "unknown source")
        excerpt_blocks.append(f"[{i}] (source: {source})\n{text}")

    excerpts_section = "\n\n".join(excerpt_blocks)
    return (
        f"Question: {question}\n\n"
        f"Available excerpts:\n\n{excerpts_section}\n\n"
        f"Answer the question using only the excerpts above, with citations."
    )
