"""
BM25 keyword index — the lexical half of hybrid retrieval.

Dense embeddings are great at "what does this mean" but routinely miss
exact identifiers, product codes, names, or rare terms that don't cluster
semantically. BM25 catches those. Combining both (see hybrid.py) is what
separates hybrid retrieval from naive top-k cosine search.
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(self):
        self._ids: list[str] = []
        self._payloads: list[dict] = []
        self._corpus_tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def add_documents(self, ids: list[str], texts: list[str], payloads: list[dict]) -> None:
        self._ids.extend(ids)
        self._payloads.extend(payloads)
        self._corpus_tokens.extend(_tokenize(t) for t in texts)
        # BM25Okapi has no incremental API; rebuilding is cheap relative to
        # embedding cost and fine at the corpus sizes this project targets.
        # At real production scale, swap for Elasticsearch/OpenSearch, which
        # this class's interface is deliberately narrow enough to be replaced by.
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None

    def search(self, query: str, top_k: int) -> list[dict]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [
            {"id": self._ids[i], "score": float(scores[i]), "payload": self._payloads[i]}
            for i in ranked
            if scores[i] > 0
        ]

    def count(self) -> int:
        return len(self._ids)
