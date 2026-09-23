"""
Loads every .txt file in data/corpus/ into a RUNNING server via HTTP.
Use this instead of ingest_corpus.py when you want to test through
uvicorn/the /docs UI, since ingest_corpus.py uses its own separate,
in-process pipeline that never touches the live server.

Usage (with the server already running in another terminal):
    python scripts/ingest_via_api.py
"""
import json
import pathlib
import urllib.request

API_URL = "http://localhost:8000/ingest"
API_KEY = "dev-only-key-change-me"  # match RAG_API_KEY / the default in .env
CORPUS_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "corpus"

for path in sorted(CORPUS_DIR.glob("*.txt")):
    payload = {
        "doc_id": path.stem,
        "text": path.read_text(),
        "metadata": {"source_file": path.name},
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "x-api-key": API_KEY},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        print(path.stem, "->", resp.read().decode())

print("\nDone. The running server now has the sample corpus loaded.")
