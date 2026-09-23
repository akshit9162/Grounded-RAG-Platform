"""
Shows exactly what the retrieval + reranking layer returns for a question,
BEFORE it reaches generation. Use this to verify retrieval quality in
isolation — useful because the mock generator has its own separate
(cruder) sentence-selection logic layered on top, so a bad final answer
doesn't necessarily mean retrieval got it wrong; this script lets you
check that specific layer directly.

Usage:
    python scripts/debug_retrieval.py "How many days do I have to return an order?"
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rag_platform.ingestion.pipeline import IngestionPipeline  # noqa: E402
from rag_platform.retrieval.embeddings import get_embedder  # noqa: E402
from rag_platform.retrieval.hybrid import HybridRetriever  # noqa: E402
from rag_platform.retrieval.reranker import get_reranker  # noqa: E402

CORPUS_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "corpus"


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "How many days do I have to return an order?"

    print(f"Embedder in use:  {type(get_embedder()).__name__}")
    print(f"Reranker in use:  {type(get_reranker()).__name__}\n")

    pipeline = IngestionPipeline()
    for path in sorted(CORPUS_DIR.glob("*.txt")):
        pipeline.submit_document(path.stem, path.read_text(), metadata={"source_file": path.name})
        pipeline.process_pending()

    retriever = HybridRetriever(bm25_index=pipeline.bm25_index)

    print(f"Question: {question!r}\n")
    print("Retrieved + reranked chunks, in the order generation will see them:\n")

    results = retriever.retrieve(question)
    if not results:
        print("  (no candidates returned at all)")
        return

    for rank, r in enumerate(results, start=1):
        doc_id = r["payload"].get("doc_id", "?")
        text = r["payload"]["text"][:140].replace("\n", " ")
        score = r.get("rerank_score", r.get("hybrid_score", 0))
        print(f"  #{rank}  doc={doc_id:<24} score={score:.4f}")
        print(f"        {text}...\n")

    top_doc = results[0]["payload"].get("doc_id", "")
    if top_doc.startswith("refund"):
        print("Top result is from refund_policy — retrieval correctly identified the right document.")
    else:
        print(f"Top result is from '{top_doc}', not refund_policy — retrieval itself got it wrong here, not just the mock generator.")


if __name__ == "__main__":
    main()
