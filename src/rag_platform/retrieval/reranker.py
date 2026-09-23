"""
Reranker: re-scores the merged hybrid candidate set with a model that reads
the query and each candidate chunk *together*, which is far more precise
than the independent-similarity scores hybrid retrieval produces. This is
usually the single biggest lever on RAG answer quality, and it's the step
most beginner RAG projects skip entirely.

Two implementations:
- CrossEncoderReranker: real cross-encoder (e.g. BAAI/bge-reranker-base).
  Use this in production.
- LexicalOverlapReranker: dependency-light fallback (token-overlap +
  position prior) so the pipeline is runnable without downloading a model.
"""
from __future__ import annotations

from typing import Protocol

from rag_platform.text_utils import meaningful_tokens


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder  # optional dep

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        pairs = [(query, c["payload"]["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        ranked = sorted(candidates, key=lambda c: -c["rerank_score"])
        return ranked[:top_k]


class LexicalOverlapReranker:
    """
    Fallback reranker: scores each candidate by token overlap with the
    query (weighted toward rarer, longer tokens) plus a small boost for
    candidates ranked highly by more than one retrieval channel. This is
    intentionally simple — it exists so the pipeline runs without a
    downloaded model, not as a claim that it matches cross-encoder quality.
    """

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        query_tokens = meaningful_tokens(query)
        scored = []
        for c in candidates:
            text_tokens = meaningful_tokens(c["payload"]["text"])
            overlap = query_tokens & text_tokens
            overlap_score = sum(len(t) for t in overlap)  # longer shared tokens weigh more
            c["rerank_score"] = overlap_score + 0.1 * c.get("hybrid_score", 0.0)
            scored.append(c)
        # A candidate with literally zero lexical overlap with the query has
        # no business being called "relevant" just because it happened to
        # rank in the fused hybrid list — drop it rather than let a near-zero
        # score still make the cut and get passed to generation.
        relevant = [c for c in scored if c["rerank_score"] > 0.15]
        ranked = sorted(relevant, key=lambda c: -c["rerank_score"])
        return ranked[:top_k]


_reranker_singleton: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker_singleton
    if _reranker_singleton is not None:
        return _reranker_singleton

    from rag_platform.config import settings

    if settings.reranker_model:
        _reranker_singleton = CrossEncoderReranker(settings.reranker_model)
    else:
        _reranker_singleton = LexicalOverlapReranker()
    return _reranker_singleton
