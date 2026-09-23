import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("RAG_API_KEY", "test-key")

from rag_platform.ingestion.chunker import chunk_document
from rag_platform.ingestion.pipeline import IngestionPipeline
from rag_platform.retrieval.hybrid import HybridRetriever
from rag_platform.generation.guardrails import check_answer


def test_chunker_respects_overlap():
    text = " ".join([f"Sentence number {i} about refunds and policy details." for i in range(40)])
    chunks = chunk_document("doc1", text, chunk_size_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    # consecutive chunks should share some trailing/leading content (overlap)
    assert any(
        chunks[i].text.split()[-3:] == chunks[i + 1].text.split()[: len(chunks[i].text.split()[-3:])]
        or set(chunks[i].text.split()[-5:]) & set(chunks[i + 1].text.split()[:5])
        for i in range(len(chunks) - 1)
    )


def test_chunker_empty_text_returns_no_chunks():
    assert chunk_document("doc1", "") == []


def test_ingestion_and_retrieval_end_to_end():
    pipeline = IngestionPipeline()
    pipeline.submit_document(
        "policy_a",
        "Refunds are issued within 30 days of purchase for unused items. "
        "Shipping costs are non-refundable under any circumstance.",
    )
    pipeline.submit_document(
        "policy_b",
        "Two-factor authentication is mandatory for accounts with high balances. "
        "Passwords must be at least ten characters long.",
    )
    n_chunks = pipeline.process_pending()
    assert n_chunks >= 2

    retriever = HybridRetriever(bm25_index=pipeline.bm25_index)
    results = retriever.retrieve("How many days until I can get a refund?")
    assert len(results) > 0
    assert any(r["payload"]["doc_id"] == "policy_a" for r in results)


def test_guardrail_flags_answer_with_no_citations():
    result = check_answer("Refunds take 30 days.", chunks=[{"payload": {"text": "irrelevant", "doc_id": "x"}}])
    assert result.is_grounded is False
    assert "no citations" in result.reason


def test_guardrail_accepts_honest_refusal():
    result = check_answer(
        "I don't have enough information in the available sources to answer that.",
        chunks=[{"payload": {"text": "irrelevant", "doc_id": "x"}}],
    )
    assert result.is_grounded is True
    assert result.is_refusal is True


def test_guardrail_flags_low_faithfulness():
    chunks = [{"payload": {"text": "Refunds are processed within 30 days.", "doc_id": "policy_a"}}]
    # Answer cites [1] but its actual content has nothing to do with the source text.
    unfaithful_answer = "Elephants migrate across continents every winter season. [1]"
    result = check_answer(unfaithful_answer, chunks)
    assert result.is_grounded is False


def test_guardrail_catches_hallucination_hidden_inside_a_mostly_grounded_answer():
    # Regression test for a real bug found in manual testing: a long answer
    # that is MOSTLY grounded can still pass whole-answer bag-of-words
    # overlap even when it contains a fabricated sentence, because the
    # fabricated sentence shares enough generic domain vocabulary ("refund",
    # "days", "order") with the real source to not drag the average down
    # far enough. Per-sentence checking should catch the fabricated sentence
    # even though the answer as a whole looks fine on average.
    chunks = [
        {
            "payload": {
                "text": "Standard orders are eligible for a full refund within 30 days of "
                "delivery, provided the item is returned in its original condition.",
                "doc_id": "refund_policy",
            }
        }
    ]
    answer = (
        "Standard orders are eligible for a full refund within 30 days of delivery, "
        "provided the item is returned in its original condition. [1] "
        "Digital goods purchased in error may be refunded within 14 days if the "
        "activation code has not been redeemed. [1]"
    )
    result = check_answer(answer, chunks)
    assert result.is_grounded is False
    assert any("digital" in s.lower() for s in result.unsupported_sentences)


def test_api_rejects_missing_api_key():
    from fastapi.testclient import TestClient
    from rag_platform.serving.app import app

    client = TestClient(app)
    response = client.post("/query", json={"question": "test"})
    assert response.status_code == 422  # missing required header


def test_api_health_endpoint():
    from fastapi.testclient import TestClient
    from rag_platform.serving.app import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert "indexed_chunks" in response.json()
