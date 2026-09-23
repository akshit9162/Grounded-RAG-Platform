"""
Embedding layer. Two implementations behind one interface:

- SentenceTransformerEmbedder: real dense embeddings (e.g. BAAI/bge-large-en-v1.5).
  This is what you should use in production — swap it in by setting
  EMBEDDING_MODEL in config.
- TfidfEmbedder: a local, zero-download fallback used when no model is
  configured (e.g. running this demo without internet/HuggingFace access).
  It is NOT a substitute for real dense embeddings in production — it can't
  capture semantic similarity the way a trained encoder can — but it lets
  the whole pipeline run end-to-end with zero external dependencies, which
  is what makes the demo script in this repo actually runnable anywhere.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # optional dep

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class TfidfEmbedder:
    """
    Deterministic, dependency-light fallback embedder based on hashed
    char n-gram TF-IDF, L2-normalized so cosine similarity works the same
    way it would with real dense embeddings. Fit incrementally as documents
    arrive (no separate "training corpus" needed).
    """

    def __init__(self, dim: int = 512):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim
        self._vectorizer = HashingVectorizer(
            n_features=dim,
            analyzer="char_wb",
            ngram_range=(3, 5),
            norm="l2",
            alternate_sign=False,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = self._vectorizer.transform(texts)
        return np.asarray(matrix.todense())

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


_embedder_singleton: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder_singleton
    if _embedder_singleton is not None:
        return _embedder_singleton

    from rag_platform.config import settings

    if settings.embedding_model:
        _embedder_singleton = SentenceTransformerEmbedder(settings.embedding_model)
    else:
        _embedder_singleton = TfidfEmbedder()
    return _embedder_singleton
