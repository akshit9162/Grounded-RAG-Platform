"""
Ingests every .txt file in data/corpus/ into the pipeline. Run this once
before querying or evaluating.

    python scripts/ingest_corpus.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rag_platform.ingestion.pipeline import IngestionPipeline  # noqa: E402

CORPUS_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "corpus"


def main() -> IngestionPipeline:
    pipeline = IngestionPipeline()
    total = 0
    for path in sorted(CORPUS_DIR.glob("*.txt")):
        doc_id = path.stem
        text = path.read_text()
        pipeline.submit_document(doc_id, text, metadata={"source_file": path.name})
        n = pipeline.process_pending()
        print(f"  ingested {doc_id}: {n} chunks")
        total += n
    print(f"\nTotal chunks indexed: {total}")
    print(f"Vector store count: {pipeline.vector_store.count()}")
    print(f"BM25 index count:   {pipeline.bm25_index.count()}")
    return pipeline


if __name__ == "__main__":
    main()
