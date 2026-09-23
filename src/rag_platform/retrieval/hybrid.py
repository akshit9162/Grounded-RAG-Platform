"""
Hybrid retrieval: runs dense (vector) search and BM25 (keyword) search in
parallel, merges the two candidate sets with reciprocal rank fusion, then
hands the merged set to the reranker for final scoring.

Reciprocal rank fusion (RRF) is used instead of a raw weighted-score blend
because dense cosine scores and BM25 scores live on completely different
scales — RRF sidesteps that by fusing on *rank position* instead of raw
score, which is the standard, defensible way to combine heterogeneous
retrievers.
"""
from __future__ import annotations

from rag_platform.config import settings
from rag_platform.retrieval.bm25 import BM25Index
from rag_platform.retrieval.embeddings import get_embedder
from rag_platform.retrieval.reranker import get_reranker
from rag_platform.retrieval.vector_store import get_vector_store


def _reciprocal_rank_fusion(
    ranked_lists: list[list[dict]], k: int = 60
) -> dict[str, dict]:
    """RRF score for id `d` = sum over lists of 1 / (k + rank_in_list)."""
    fused: dict[str, dict] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            item_id = item["id"]
            if item_id not in fused:
                fused[item_id] = {"id": item_id, "payload": item["payload"], "hybrid_score": 0.0}
            fused[item_id]["hybrid_score"] += 1.0 / (k + rank + 1)
    return fused


class HybridRetriever:
    def __init__(self, bm25_index: BM25Index):
        self.bm25_index = bm25_index
        self.embedder = get_embedder()
        self.vector_store = get_vector_store()
        self.reranker = get_reranker()

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or settings.rerank_top_k

        query_vector = self.embedder.embed_query(query)
        dense_hits = self.vector_store.search(query_vector, settings.dense_top_k)
        bm25_hits = self.bm25_index.search(query, settings.bm25_top_k)

        fused = _reciprocal_rank_fusion([dense_hits, bm25_hits])
        candidates = list(fused.values())

        if not candidates:
            return []

        reranked = self.reranker.rerank(query, candidates, top_k)
        return reranked
