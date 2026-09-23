"""
FastAPI serving layer. This is the piece that turns the pipeline into a
callable service: auth, semantic caching, request tracing, and the actual
retrieve -> rerank -> generate -> guardrail-check flow.

Run locally with:
    uvicorn rag_platform.serving.app:app --reload
"""
from __future__ import annotations

import uuid

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from rag_platform.config import settings
from rag_platform.generation.guardrails import check_answer
from rag_platform.generation.llm import get_llm_client
from rag_platform.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from rag_platform.ingestion.pipeline import IngestionPipeline
from rag_platform.observability.tracing import RequestTrace
from rag_platform.retrieval.embeddings import get_embedder
from rag_platform.retrieval.hybrid import HybridRetriever
from rag_platform.serving.auth import verify_api_key
from rag_platform.serving.cache import get_cache

app = FastAPI(title="RAG Platform", version="1.0.0")

# Shared pipeline singleton (ingestion pipeline owns the bm25 index that
# retrieval reads from, keeping ingest + retrieve in lockstep).
_pipeline = IngestionPipeline()
_retriever = HybridRetriever(bm25_index=_pipeline.bm25_index)


class IngestRequest(BaseModel):
    doc_id: str
    text: str
    metadata: dict = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    citations: list[dict]
    is_grounded: bool
    faithfulness_score: float
    unsupported_sentences: list[str]
    cache_hit: bool
    latency_ms: float


@app.get("/health")
def health():
    return {
        "status": "ok",
        "indexed_chunks": _pipeline.vector_store.count(),
        "bm25_documents": _pipeline.bm25_index.count(),
    }


@app.post("/ingest")
def ingest(request: IngestRequest, _: str = Depends(verify_api_key)):
    _pipeline.submit_document(request.doc_id, request.text, request.metadata)
    n_chunks = _pipeline.process_pending()
    return {"doc_id": request.doc_id, "chunks_indexed": n_chunks}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, _: str = Depends(verify_api_key)):
    request_id = str(uuid.uuid4())
    trace = RequestTrace(request_id=request_id, query=request.question)
    embedder = get_embedder()
    cache = get_cache()

    with trace.span("embed_query"):
        query_vector = embedder.embed_query(request.question)

    with trace.span("cache_lookup"):
        cached = cache.get(query_vector)

    if cached:
        trace.flush()
        return QueryResponse(
            request_id=request_id,
            answer=cached["answer"],
            citations=cached["citations"],
            is_grounded=cached["is_grounded"],
            faithfulness_score=cached["faithfulness_score"],
            unsupported_sentences=cached.get("unsupported_sentences", []),
            cache_hit=True,
            latency_ms=trace.total_duration_ms(),
        )

    with trace.span("retrieve") as span:
        chunks = _retriever.retrieve(request.question)
        span.metadata["retrieved_ids"] = [c["id"] for c in chunks]
        span.metadata["rerank_scores"] = [round(c.get("rerank_score", 0), 4) for c in chunks]

    with trace.span("generate"):
        llm = get_llm_client()
        user_prompt = build_user_prompt(request.question, chunks)
        answer = llm.generate(SYSTEM_PROMPT, user_prompt, settings.max_answer_tokens)

    with trace.span("guardrail_check") as span:
        guardrail_result = check_answer(answer, chunks)
        span.metadata["is_grounded"] = guardrail_result.is_grounded
        span.metadata["faithfulness_score"] = guardrail_result.faithfulness_score

    citations = [
        {
            "index": i,
            "doc_id": chunk["payload"].get("doc_id"),
            "text_snippet": chunk["payload"]["text"][:200],
        }
        for i, chunk in enumerate(chunks, start=1)
        if i in guardrail_result.cited_indices
    ]

    response_payload = {
        "answer": answer,
        "citations": citations,
        "is_grounded": guardrail_result.is_grounded,
        "faithfulness_score": guardrail_result.faithfulness_score,
        "unsupported_sentences": guardrail_result.unsupported_sentences,
    }
    cache.set(request.question, query_vector, response_payload)
    trace.flush()

    return QueryResponse(
        request_id=request_id,
        cache_hit=False,
        latency_ms=trace.total_duration_ms(),
        **response_payload,
    )
