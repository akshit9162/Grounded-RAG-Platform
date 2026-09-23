"""
Ingestion pipeline: watches for new/updated documents, chunks them, embeds
them, and upserts them into the vector store + BM25 index.

Transport is pluggable:
  - INGESTION_TRANSPORT=local  -> an in-process queue (zero setup, used by
    the demo script and tests)
  - INGESTION_TRANSPORT=kafka  -> a real Kafka topic (used in production;
    lets ingestion scale independently and survive restarts)

This is the piece that answers "why Kafka" in an interview: documents don't
arrive all at once, ingestion is CPU/GPU-bound (embedding) and should scale
independently of the API layer, and a topic gives you replay + backpressure
for free instead of hand-rolling a job queue.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from queue import Queue
from typing import Iterable, Optional

from rag_platform.config import settings
from rag_platform.ingestion.chunker import chunk_document
from rag_platform.retrieval.embeddings import get_embedder
from rag_platform.retrieval.vector_store import get_vector_store
from rag_platform.retrieval.bm25 import BM25Index

logger = logging.getLogger(__name__)


class LocalQueueTransport:
    """In-process stand-in for Kafka. Same interface, no external services."""

    def __init__(self) -> None:
        self._queue: Queue = Queue()

    def send(self, document: dict) -> None:
        self._queue.put(document)

    def poll(self) -> Iterable[dict]:
        while not self._queue.empty():
            yield self._queue.get()


class KafkaTransport:
    """
    Real Kafka transport. Requires `kafka-python` and a running broker
    (see docker/docker-compose.yml). Not used by the in-sandbox demo, but
    this is the code path production deployments should use.
    """

    def __init__(self, bootstrap_servers: str, topic: str) -> None:
        from kafka import KafkaProducer, KafkaConsumer  # local import: optional dep

        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            group_id="rag-ingestion",
        )

    def send(self, document: dict) -> None:
        self.producer.send(self.topic, value=document)
        self.producer.flush()

    def poll(self) -> Iterable[dict]:
        # Non-blocking poll batch; production code would run this in a
        # long-lived consumer loop instead of draining once.
        records = self.consumer.poll(timeout_ms=1000)
        for _, batch in records.items():
            for record in batch:
                yield record.value


def get_transport():
    if settings.ingestion_transport == "kafka":
        return KafkaTransport(settings.kafka_bootstrap_servers, settings.kafka_topic)
    return LocalQueueTransport()


class IngestionPipeline:
    """
    Consumes documents from the transport, chunks + embeds them, and writes
    them into both the vector store (dense) and the BM25 index (keyword).
    """

    def __init__(self, transport=None):
        self.transport = transport or get_transport()
        self.embedder = get_embedder()
        self.vector_store = get_vector_store()
        self.bm25_index = BM25Index()

    def submit_document(self, doc_id: str, text: str, metadata: Optional[dict] = None) -> None:
        self.transport.send({"doc_id": doc_id, "text": text, "metadata": metadata or {}})

    def process_pending(self) -> int:
        """Drain the transport, chunk + embed + index each document. Returns
        the number of chunks written."""
        total_chunks = 0
        for doc in self.transport.poll():
            chunks = chunk_document(
                doc_id=doc["doc_id"],
                text=doc["text"],
                chunk_size_tokens=settings.chunk_size_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
                metadata=doc.get("metadata", {}),
            )
            if not chunks:
                continue

            texts = [c.text for c in chunks]
            vectors = self.embedder.embed(texts)

            self.vector_store.upsert(
                ids=[c.id for c in chunks],
                vectors=vectors,
                payloads=[
                    {"doc_id": c.doc_id, "text": c.text, "position": c.position, **c.metadata}
                    for c in chunks
                ],
            )
            self.bm25_index.add_documents(
                ids=[c.id for c in chunks],
                texts=texts,
                payloads=[asdict(c) for c in chunks],
            )
            total_chunks += len(chunks)
            logger.info("Ingested %s: %d chunks", doc["doc_id"], len(chunks))

        return total_chunks
