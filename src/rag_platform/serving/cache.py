"""
Semantic query cache. Unlike a plain string-keyed cache, this caches by
*embedding similarity* — so "What's the refund policy?" and "How do refunds
work?" hit the same cache entry instead of both paying full pipeline cost.
This is what turns caching into an actual latency/cost lever instead of a
no-op for a Q&A workload where the same question is rarely asked verbatim
twice.
"""
from __future__ import annotations

import json
import time
from typing import Optional, Protocol

import numpy as np

from rag_platform.config import settings


class SemanticCache(Protocol):
    def get(self, query_vector: np.ndarray) -> Optional[dict]: ...
    def set(self, query: str, query_vector: np.ndarray, response: dict) -> None: ...


class InMemorySemanticCache:
    def __init__(self):
        self._entries: list[dict] = []  # [{vector, query, response, expires_at}]

    def get(self, query_vector: np.ndarray) -> Optional[dict]:
        now = time.time()
        self._entries = [e for e in self._entries if e["expires_at"] > now]
        best, best_score = None, -1.0
        for e in self._entries:
            denom = (np.linalg.norm(e["vector"]) * np.linalg.norm(query_vector)) + 1e-9
            score = float(np.dot(e["vector"], query_vector) / denom)
            if score > best_score:
                best, best_score = e, score
        if best and best_score >= settings.cache_similarity_threshold:
            return {**best["response"], "cache_hit": True, "cache_similarity": round(best_score, 4)}
        return None

    def set(self, query: str, query_vector: np.ndarray, response: dict) -> None:
        self._entries.append(
            {
                "query": query,
                "vector": query_vector,
                "response": response,
                "expires_at": time.time() + settings.cache_ttl_seconds,
            }
        )


class RedisSemanticCache:
    """
    Production backend. Stores (vector, response) pairs in Redis with a
    TTL; similarity search is done client-side over a bounded recent set
    (or via RediSearch's vector index if the module is available). Requires
    `redis` and a running Redis instance (see docker-compose.yml).
    """

    def __init__(self, url: str):
        import redis

        self.client = redis.from_url(url)
        self.index_key = "rag:semantic_cache:index"

    def get(self, query_vector: np.ndarray) -> Optional[dict]:
        entry_keys = self.client.lrange(self.index_key, 0, 199)  # bounded scan window
        best, best_score = None, -1.0
        for key in entry_keys:
            raw = self.client.get(key)
            if raw is None:
                continue
            entry = json.loads(raw)
            vec = np.array(entry["vector"])
            denom = (np.linalg.norm(vec) * np.linalg.norm(query_vector)) + 1e-9
            score = float(np.dot(vec, query_vector) / denom)
            if score > best_score:
                best, best_score = entry, score
        if best and best_score >= settings.cache_similarity_threshold:
            return {**best["response"], "cache_hit": True, "cache_similarity": round(best_score, 4)}
        return None

    def set(self, query: str, query_vector: np.ndarray, response: dict) -> None:
        key = f"rag:semantic_cache:{abs(hash(query))}"
        payload = json.dumps({"vector": query_vector.tolist(), "response": response})
        self.client.setex(key, settings.cache_ttl_seconds, payload)
        self.client.lpush(self.index_key, key)
        self.client.ltrim(self.index_key, 0, 999)  # cap index size


_cache_singleton: SemanticCache | None = None


def get_cache() -> SemanticCache:
    global _cache_singleton
    if _cache_singleton is not None:
        return _cache_singleton

    if settings.cache_backend == "redis":
        _cache_singleton = RedisSemanticCache(settings.redis_url)
    else:
        _cache_singleton = InMemorySemanticCache()
    return _cache_singleton
