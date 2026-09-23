"""
Vector store abstraction. Two backends:

- QdrantVectorStore: real vector DB for production (see docker-compose.yml).
- InMemoryVectorStore: numpy-based cosine-similarity store, used by default
  so the demo/tests run with zero external services.

Swapping backends is a one-line env var change (VECTOR_STORE_BACKEND);
nothing above this layer (ingestion, retrieval) knows or cares which one
is active.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class VectorStore(Protocol):
    def upsert(self, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None: ...
    def search(self, query_vector: np.ndarray, top_k: int) -> list[dict]: ...
    def count(self) -> int: ...


class InMemoryVectorStore:
    def __init__(self):
        self._ids: list[str] = []
        self._vectors: np.ndarray | None = None
        self._payloads: list[dict] = []

    def upsert(self, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None:
        if self._vectors is None:
            self._vectors = vectors.copy()
        else:
            self._vectors = np.vstack([self._vectors, vectors])
        self._ids.extend(ids)
        self._payloads.extend(payloads)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[dict]:
        if self._vectors is None or len(self._ids) == 0:
            return []
        norms = np.linalg.norm(self._vectors, axis=1) * (np.linalg.norm(query_vector) + 1e-9)
        norms = np.where(norms == 0, 1e-9, norms)
        sims = (self._vectors @ query_vector) / norms
        top_idx = np.argsort(-sims)[:top_k]
        return [
            {"id": self._ids[i], "score": float(sims[i]), "payload": self._payloads[i]}
            for i in top_idx
        ]

    def count(self) -> int:
        return len(self._ids)


class QdrantVectorStore:
    """
    Production backend. Requires `qdrant-client` and a running Qdrant
    instance (see docker/docker-compose.yml). Uses cosine distance to match
    the in-memory backend's semantics exactly, so eval numbers are
    comparable across environments.
    """

    def __init__(self, url: str, collection: str, dim: int):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.client = QdrantClient(url=url)
        self.collection = collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(id=i, vector=vectors[i].tolist(), payload=payloads[i])
            for i in range(len(ids))
        ]
        # Qdrant point IDs must be int/UUID; map string chunk ids via payload.
        for p, cid in zip(points, ids):
            p.payload["chunk_id"] = cid
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[dict]:
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector.tolist(),
            limit=top_k,
        )
        return [
            {"id": r.payload.get("chunk_id", r.id), "score": r.score, "payload": r.payload}
            for r in results
        ]

    def count(self) -> int:
        return self.client.count(collection_name=self.collection).count


_store_singleton: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store_singleton
    if _store_singleton is not None:
        return _store_singleton

    from rag_platform.config import settings
    from rag_platform.retrieval.embeddings import get_embedder

    if settings.vector_store_backend == "qdrant":
        dim = get_embedder().dim
        _store_singleton = QdrantVectorStore(settings.qdrant_url, settings.qdrant_collection, dim)
    else:
        _store_singleton = InMemoryVectorStore()
    return _store_singleton
