FROM python:3.12-slim

WORKDIR /app

# Only the core (zero-external-service) dependencies by default.
# Add sentence-transformers / qdrant-client / redis / kafka-python / ragas
# to this list (or `pip install -r requirements.txt` for everything) once
# you're wiring up the real backends.
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    "uvicorn[standard]==0.30.6" \
    pydantic==2.9.2 \
    numpy==1.26.4 \
    scikit-learn==1.5.2 \
    rank-bm25==0.2.2 \
    anthropic==0.34.2 \
    python-dotenv==1.0.1

COPY src/ ./src/
COPY data/ ./data/

ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["uvicorn", "rag_platform.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
