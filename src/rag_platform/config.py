"""
Central configuration for the RAG platform.

Everything that differs between "quick local demo" and "real production
deployment" is controlled from here via environment variables, so the same
codebase runs in both modes without code changes.
"""
import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()  # loads .env from the current working directory, if present
except ImportError:
    pass  # python-dotenv not installed — fall back to real environment variables only


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    # --- Chunking ---
    chunk_size_tokens: int = int(os.getenv("CHUNK_SIZE_TOKENS", "350"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "60"))

    # --- Embeddings ---
    # Set to a real sentence-transformers model name (e.g. "BAAI/bge-large-en-v1.5")
    # to use real dense embeddings. If unset, falls back to a local TF-IDF
    # embedder so the platform runs with zero external downloads.
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "")

    # --- Vector store ---
    # "memory" (default, zero setup) or "qdrant"
    vector_store_backend: str = os.getenv("VECTOR_STORE_BACKEND", "memory")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "rag_chunks")

    # --- Retrieval ---
    dense_top_k: int = int(os.getenv("DENSE_TOP_K", "20"))
    bm25_top_k: int = int(os.getenv("BM25_TOP_K", "20"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "3"))

    # --- Reranker ---
    # Set to a real cross-encoder model name (e.g. "BAAI/bge-reranker-base")
    # to use a real reranker. If unset, falls back to a lexical-overlap scorer.
    reranker_model: str = os.getenv("RERANKER_MODEL", "")

    # --- Generation ---
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic")
    llm_model: str = os.getenv("LLM_MODEL", "claude-sonnet-5")
    max_answer_tokens: int = int(os.getenv("MAX_ANSWER_TOKENS", "800"))
    faithfulness_threshold: float = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.5"))
    sentence_faithfulness_floor: float = float(os.getenv("SENTENCE_FAITHFULNESS_FLOOR", "0.2"))

    # --- Ingestion transport ---
    # "local" (in-process queue, default) or "kafka"
    ingestion_transport: str = os.getenv("INGESTION_TRANSPORT", "local")
    kafka_bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    kafka_topic: str = os.getenv("KAFKA_TOPIC", "documents")

    # --- Cache ---
    # "memory" (default) or "redis"
    cache_backend: str = os.getenv("CACHE_BACKEND", "memory")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    cache_similarity_threshold: float = float(os.getenv("CACHE_SIM_THRESHOLD", "0.95"))
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))

    # --- Observability ---
    tracing_enabled: bool = _bool("TRACING_ENABLED", True)
    trace_log_path: str = os.getenv("TRACE_LOG_PATH", "./data/traces.jsonl")

    # --- Serving ---
    api_key: str = os.getenv("RAG_API_KEY", "dev-only-key-change-me")
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))


settings = Settings()
