"""
Observability: records structured, timed traces of every request through
the pipeline (retrieval latency, which chunks were retrieved, rerank
scores, generation latency, guardrail verdict).

Written as JSONL to disk by default so the demo needs no external service.
In production, point OTEL_EXPORTER_OTLP_ENDPOINT at Arize Phoenix
(self-hosted, open-source, OpenTelemetry-native) and swap this module for
a real OTEL exporter — the trace *shape* defined here (spans per stage,
timing, retrieved-chunk ids) maps directly onto OTEL spans, so the schema
doesn't need to change, only where it's sent.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

from rag_platform.config import settings


@dataclass
class Span:
    name: str
    start_time: float
    end_time: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return round((self.end_time - self.start_time) * 1000, 2)


class RequestTrace:
    def __init__(self, request_id: str, query: str):
        self.request_id = request_id
        self.query = query
        self.spans: list[Span] = []
        self.created_at = time.time()

    @contextmanager
    def span(self, name: str, **metadata):
        s = Span(name=name, start_time=time.time(), metadata=metadata)
        try:
            yield s
        finally:
            s.end_time = time.time()
            self.spans.append(s)

    def total_duration_ms(self) -> float:
        return round(sum(s.duration_ms for s in self.spans), 2)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "query": self.query,
            "created_at": self.created_at,
            "total_duration_ms": self.total_duration_ms(),
            "spans": [
                {"name": s.name, "duration_ms": s.duration_ms, "metadata": s.metadata}
                for s in self.spans
            ],
        }

    def flush(self) -> None:
        if not settings.tracing_enabled:
            return
        with open(settings.trace_log_path, "a") as f:
            f.write(json.dumps(self.to_dict()) + "\n")
