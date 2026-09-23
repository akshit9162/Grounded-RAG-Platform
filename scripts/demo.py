"""
End-to-end smoke test: ingests the sample corpus, runs a few questions
through the full pipeline (hybrid retrieval -> rerank -> generate ->
guardrail check), and prints the result of each stage.

Runs with ZERO external services or API keys (uses the TF-IDF embedder,
in-memory vector store, and the mock extractive generator by default).
Set EMBEDDING_MODEL / RERANKER_MODEL / ANTHROPIC_API_KEY to swap in the
real components — see README.md.

    python scripts/demo.py
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rag_platform.generation.guardrails import check_answer  # noqa: E402
from rag_platform.generation.llm import get_llm_client  # noqa: E402
from rag_platform.generation.prompts import SYSTEM_PROMPT, build_user_prompt  # noqa: E402
from rag_platform.observability.tracing import RequestTrace  # noqa: E402
from rag_platform.retrieval.hybrid import HybridRetriever  # noqa: E402
from scripts.ingest_corpus import main as ingest_corpus  # noqa: E402

DEMO_QUESTIONS = [
    "How many days do I have to return an order for a refund?",
    "Can I still get a refund on a gift card I already redeemed?",
    "When is two-factor authentication mandatory on my account?",
    "What is the capital of France?",  # deliberately off-corpus: should trigger a refusal
]


def answer_question(retriever: HybridRetriever, question: str) -> dict:
    trace = RequestTrace(request_id="demo", query=question)
    llm = get_llm_client()

    with trace.span("retrieve"):
        chunks = retriever.retrieve(question)

    with trace.span("generate"):
        prompt = build_user_prompt(question, chunks)
        answer = llm.generate(SYSTEM_PROMPT, prompt, max_tokens=400)

    with trace.span("guardrail_check"):
        result = check_answer(answer, chunks)

    return {
        "question": question,
        "retrieved_chunks": len(chunks),
        "retrieved_sources": sorted({c["payload"].get("doc_id") for c in chunks}),
        "answer": answer,
        "is_grounded": result.is_grounded,
        "faithfulness_score": result.faithfulness_score,
        "is_refusal": result.is_refusal,
        "latency_ms": trace.total_duration_ms(),
    }


def main():
    print("=" * 70)
    print("STEP 1: Ingesting sample corpus")
    print("=" * 70)
    pipeline = ingest_corpus()

    print()
    print("=" * 70)
    print("STEP 2: Running questions through the full pipeline")
    print("=" * 70)
    retriever = HybridRetriever(bm25_index=pipeline.bm25_index)

    results = []
    for question in DEMO_QUESTIONS:
        result = answer_question(retriever, question)
        results.append(result)
        print(f"\nQ: {result['question']}")
        print(f"   Retrieved from: {result['retrieved_sources']}")
        print(f"   Answer: {result['answer']}")
        print(f"   Grounded: {result['is_grounded']} | Faithfulness: {result['faithfulness_score']} | Latency: {result['latency_ms']}ms")

    out_path = pathlib.Path(__file__).resolve().parents[1] / "data" / "demo_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
