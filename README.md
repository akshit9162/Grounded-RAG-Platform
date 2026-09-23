# Grounded RAG Platform

**[→ Visual, end-to-end walkthrough of the pipeline](https://akshit9162.github.io/Grounded-RAG-Platform/)** — every stage explained with a diagram, including the actual Claude API call, the guardrail catching a fabricated sentence, and the eval scorecard.

This project is an end-to-end RAG platform that lets users ask questions about a document corpus. It retrieves relevant evidence using semantic and keyword search, reranks it, sends the evidence to Claude to generate a cited answer, and checks whether the response is grounded in the source documents.

A production-shaped Retrieval-Augmented Generation system: hybrid (dense +
BM25) retrieval, cross-encoder reranking, citation-grounded generation with
a faithfulness guardrail, a FastAPI serving layer with semantic caching, and
a RAGAS-based evaluation harness that scores retrieval and generation
*separately* instead of one black-box score.

It runs end to end with **zero external services and zero API keys** using
local fallbacks (TF-IDF embeddings, lexical reranking, an in-memory vector
store, an extractive stand-in for the LLM). Every fallback is a one-line
config swap away from the real production component (sentence-transformers
embeddings, a cross-encoder reranker, Qdrant, Redis, Kafka, Claude/an
OpenAI-compatible model). This is deliberate: you should be able to clone
this, run it in five minutes, and only pull in heavier infra once you're
ready to test against it.

## Why it's built this way

Most student RAG projects are "embed docs, cosine-similarity search, call
an LLM" with no way to know if it's actually working. This one adds the
three things that separate a demo from a system:

1. **Hybrid retrieval + reranking** — dense search alone misses exact
   keywords/IDs; BM25 alone misses paraphrases. Combining both, then
   reranking the merged candidates with a cross-encoder, is what makes
   retrieval quality defensible rather than "seems fine on my three test
   questions."
2. **A faithfulness guardrail** — the generation step is required to cite
   which retrieved chunk supports each claim, and an answer that isn't
   traceable back to a chunk is flagged as ungrounded rather than shipped
   silently. Run `python scripts/demo.py` and ask it something outside the
   corpus ("what's the capital of France?") — it refuses instead of
   hallucinating. Note what this guardrail does and doesn't do: it
   *detects* an ungrounded answer after generation completes, exposing it
   as `is_grounded: false` with the specific unsupported sentence named —
   it does not stop the model from generating that answer, or stop the
   API from returning it. A caller has to actually check `is_grounded` and
   decide what to do (retry, refuse to show it, log it) — the guardrail
   reports, it doesn't gate.
3. **An evaluation harness that scores components separately** —
   `context_precision` / `context_recall` (did retrieval find the right
   evidence?) vs `faithfulness` / `answer_relevancy` (did generation use it
   correctly?). This is what lets you say "faithfulness went from X% to Y%
   after adding reranking" instead of "it seems to work."

## Architecture

```
Document sources
      |
      v
Ingestion pipeline  (chunk -> embed; local queue or Kafka transport)
      |
      v
Vector store  (dense embeddings)  +  BM25 index  (keyword)
      \_______________  ________________/
                       v
                Reranker (cross-encoder)
                       |
                       v
           LLM generation (forced citations)
                       |
                       v
             Faithfulness guardrail
                       |
                       v
        FastAPI serving layer (auth, semantic cache)
                       |
                       v
                   Client
                       ^
                       |
     Eval & observability (RAGAS + request tracing)
     feeds back into retrieval/reranking tuning
```

## Project layout

```
src/rag_platform/
  config.py              # every env-driven setting; local vs production mode lives here
  ingestion/
    chunker.py            # token-aware chunking with overlap
    pipeline.py            # ingest -> chunk -> embed -> index; local or Kafka transport
  retrieval/
    embeddings.py          # TF-IDF fallback / sentence-transformers production embedder
    vector_store.py        # in-memory fallback / Qdrant production store
    bm25.py                # keyword index
    hybrid.py               # merges dense + BM25, calls the reranker
    reranker.py             # lexical-overlap fallback / cross-encoder production reranker
  generation/
    prompts.py              # forces citation format
    llm.py                   # Anthropic client / vLLM OpenAI-compatible client / extractive mock
    guardrails.py            # faithfulness scoring, grounded/ungrounded decision
  serving/
    app.py                   # FastAPI: /ingest, /query, /health
    auth.py                  # API key check
    cache.py                 # semantic cache (embedding-similarity, not exact match)
  observability/
    tracing.py               # per-request span timing -> data/traces.jsonl
  evaluation/
    ragas_eval.py             # real RAGAS run + a lightweight offline proxy
    eval_dataset.json         # sample labeled eval set

scripts/
  ingest_corpus.py    # loads data/corpus/*.txt into an in-process pipeline (for demo.py/run_eval.py)
  ingest_via_api.py   # loads data/corpus/*.txt into a RUNNING server, over HTTP
  debug_retrieval.py  # shows raw retrieval+reranking output, isolated from generation
  demo.py             # end-to-end: ingest sample corpus, run sample questions, print results
  run_eval.py         # runs the evaluation harness and prints a scorecard

docker/               # Dockerfile + docker-compose (api, qdrant, redis, kafka)
terraform/            # AWS EC2 provisioning (Docker Compose deploy via user_data)
tests/                # pytest suite, all passing on the zero-dependency fallback path
```

## Running it

```bash
pip install -r requirements.txt   # core deps only; see comments in requirements.txt
                                    # for the production swap-ins (sentence-transformers,
                                    # qdrant-client, redis, kafka-python, ragas)

PYTHONPATH=src python scripts/demo.py       # ingest sample corpus, ask it questions
PYTHONPATH=src python scripts/run_eval.py   # lightweight eval scorecard
PYTHONPATH=src pytest tests/ -v             # full test suite

PYTHONPATH=src uvicorn rag_platform.serving.app:app --reload   # run the API
curl -X POST localhost:8000/query \
  -H "x-api-key: dev-only-key-change-me" -H "Content-Type: application/json" \
  -d '{"question": "your question here"}'
```

## Going to "real" production

Everything below is a config change in `.env`, not a code change:

| Component   | Local default (zero setup)     | Production                                  |
|-------------|---------------------------------|----------------------------------------------|
| Embeddings  | TF-IDF (`retrieval/embeddings.py`) | `EMBEDDING_MODEL=BAAI/bge-large-en-v1.5` (sentence-transformers) |
| Reranker    | Lexical overlap                 | `RERANKER_MODEL=BAAI/bge-reranker-base`       |
| Vector store| In-memory                       | `VECTOR_STORE_BACKEND=qdrant`                 |
| Cache       | In-memory                       | `CACHE_BACKEND=redis`                         |
| Ingestion   | In-process queue                | `INGESTION_TRANSPORT=kafka`                   |
| Generation  | Extractive mock (no API key)    | Set `ANTHROPIC_API_KEY` — auto-switches to Claude |
| Eval        | Lexical proxy metrics           | `run_ragas_eval()` — real RAGAS + LLM judge   |
| Deployment  | `uvicorn --reload`              | `docker/docker-compose.yml`, or `terraform apply` in `terraform/` for AWS EC2 |

To adapt this to a different domain, only `data/corpus/` and
`evaluation/eval_dataset.json` need to change — the pipeline is
domain-agnostic (support tickets, internal engineering docs, compliance
policies, whatever you point it at).

## A real bug I found and fixed while building this

Swapping in real embeddings (`sentence-transformers/all-MiniLM-L6-v2`) fixed
retrieval, but a follow-up test still cited the wrong document. The cause
wasn't retrieval — it was `MockExtractiveLLMClient`, which does its own
independent, cruder keyword-overlap scoring to pick which sentences to
quote, on top of whatever the (now-correct) reranker hands it. Its
tokenizer did exact string matching with no stemming, so the question
"return an **order**" (singular) failed to match a retrieved sentence
about "**orders**" (plural) — losing a point of overlap and getting
edged out by unrelated chunks that happened to contain "order" verbatim.

Fixed in `text_utils.py` with a small, explicitly-labeled crude
pluralization stripper (not a real stemmer/lemmatizer — see the docstring
for what it does and doesn't handle). `scripts/debug_retrieval.py` was
added specifically to make this kind of thing checkable going forward: it
prints what retrieval+reranking returns *before* generation touches it, so
a bad final answer can be isolated to the right layer instead of guessed
at.

This is a concrete, specific example of debugging a multi-stage ML system by
isolating layers — checking what retrieval returned independently of what
generation did with it — rather than only testing end to end and guessing
which stage is at fault.

## A second finding: context_precision dropped after the retrieval upgrade — and that's not a regression

After swapping in real embeddings, `context_precision` in the eval
scorecard went *down* (0.5 → 0.333), which looks alarming at first glance —
like the "upgrade" made retrieval worse. It didn't. The metric itself was
the problem, interacting with a tiny demo corpus.

`context_precision` here is document-level and rank-blind: it just checks
whether the retrieved chunks' documents match the ground-truth document,
regardless of what order they came back in. With only 3 documents and 6
chunks total, `RERANK_TOP_K=5` meant retrieval returned 5 of 6 chunks every
time — structurally touching almost every document regardless of ranking
quality, capping precision near 1/3 no matter how good the reranker is.

The old fallback reranker (`LexicalOverlapReranker`) happened to score
higher on this metric for an unrelated reason: it has a hardcoded relevance
floor that drops low-scoring chunks, which occasionally excluded whole
documents by accident. The real `CrossEncoderReranker` doesn't gatekeep
like that — it ranks everything and returns the top K, which is the more
correct design, but it made the metric's flaw visible.

Fix: lowered `RERANK_TOP_K` from 5 to 3, which is a better match for a
corpus this size. Verified with `RERANK_TOP_K` swept across 5/3/2 that
precision responds exactly as this explanation predicts (2 → 0.7, 3 →
0.567, 5 → 0.333) — confirming the drop was a windowing artifact, not a
retrieval regression. `debug_retrieval.py`'s rank-ordered output remains
the more trustworthy signal for actual retrieval quality on a corpus this
small; the lightweight `context_precision` proxy is naturally noisy here
and would need a much larger corpus (or a rank-aware metric like
precision@1 / MRR) to be fully reliable.

## A correction: what looked like a hallucination wasn't one

Worth documenting honestly, because it's a real lesson: after wiring in
real Claude generation, a refund-policy query returned a long, detailed
answer covering Standard Orders, Digital Goods, Perishable Goods,
Subscriptions, and International Orders. It was assumed — incorrectly —
that the corpus only contained Standard Orders and International Orders
policy, making the rest look fabricated, since the API's citation preview
only ever shows a ~200-character truncated snippet of each retrieved
chunk. Debugging proceeded on that assumption for several rounds (checking
for stale server processes, cache issues, file-sync problems) before
actually re-reading the full, untruncated corpus file directly.

**The corpus was always more detailed than the truncated previews
suggested** — `data/corpus/refund_policy.txt` genuinely contains full
Digital Goods, Perishable Goods, and Subscriptions policy, all within the
same two retrieved chunks. Claude's answer was accurate and correctly
cited the whole time. `is_grounded: true` was the right call on every
single request, including before any guardrail changes were made.

**The actual lesson:** when verifying whether a RAG answer is grounded,
check it against the *full* source chunk text, not a UI's truncated
preview of it — a citation preview is for human skimming, not for
deciding whether content is fabricated. This is a more useful thing to
have learned than a fake bug would have been.

A smaller, real, and still-valid improvement came out of this process
regardless: a per-sentence faithfulness check was added alongside the
original whole-answer overlap score (see
`generation/guardrails.py::check_answer`), which now also returns an
`unsupported_sentences` field naming any specific sentence that doesn't
overlap enough with the cited source — useful defense-in-depth for a
genuine future hallucination, even though it wasn't needed to explain this
particular case. It was verified against the real grounded answer above to
confirm it does *not* incorrectly flag real, correctly-cited content — a
guardrail that's stricter but also wrong under real content isn't an
improvement. A stopword list gap (`within` wasn't excluded, quietly
inflating overlap between any two policy-ish sentences) was also fixed
along the way and remains a legitimate small fix.

## Final results (real embeddings + real reranker + real Claude generation)

Everything below is running for real, not the fallback path — this is the
actual scorecard from `scripts/run_eval.py` with
`sentence-transformers/all-MiniLM-L6-v2` embeddings,
`cross-encoder/ms-marco-MiniLM-L6-v2` reranking, and real
`claude-sonnet-5` generation:

| Metric | Mock generator | Real Claude |
|---|---|---|
| context_precision | 0.567 | 0.5 |
| context_recall | 1.0 | 1.0 |
| faithfulness | 0.877 | 0.799 |
| answer_relevancy | 0.766 | **0.947** |

`answer_relevancy` improved substantially — real Claude directly reasons
about and addresses each question, rather than the mock's blunt
sentence-extraction. `context_precision`/`context_recall` are unchanged,
exactly as expected, since the generator doesn't touch retrieval at all.

`faithfulness` dropped slightly, and it's worth understanding rather than
worrying about: this project's faithfulness metric is a *lexical overlap*
proxy, not a real entailment check. The mock generator copies source
sentences verbatim, which trivially maximizes word overlap by
construction. Real Claude paraphrases — restating the same facts in its
own words — which is normal, desirable behavior, but it naturally lowers
raw word-overlap against the source even when the content is equally or
more accurate. A lexical proxy can't distinguish "paraphrased correctly"
from "drifted off-topic"; a real RAGAS setup would use an LLM judge for
this instead of word overlap, specifically because it handles paraphrasing
correctly where bag-of-words can't. This is a genuine, known limitation of
the lightweight proxy used here, not a regression in answer quality.

## Design FAQ

- **"Why hybrid retrieval instead of just dense?"** — dense embeddings miss
  exact terms (order IDs, product names, error codes); BM25 misses
  paraphrased questions. `hybrid.py` merges both result sets with
  reciprocal rank fusion (RRF) before reranking — dense cosine scores and
  BM25 scores live on incompatible scales, so the two ranked lists are
  fused by rank position (`score(d) = Σ 1 / (60 + rank + 1)`) rather than
  a weighted blend of raw scores.
- **"How do you know it's not hallucinating?"** — every generated claim
  must cite a chunk index; `guardrails.py` checks the citations are
  actually present in the retrieved chunks and computes a faithfulness
  score. Anything below `FAITHFULNESS_THRESHOLD` is flagged ungrounded.
- **"How do you measure quality?"** — `evaluation/ragas_eval.py` scores
  retrieval (context precision/recall) and generation (faithfulness/answer
  relevancy) as separate numbers, so a regression can be traced to the
  right component instead of "the answers got worse somehow."
- **"What would you change at real scale?"** — swap the in-memory vector
  store for Qdrant (already wired, config-only change), add a background
  worker pool consuming from Kafka instead of processing ingestion inline,
  and put the reranker behind a small dedicated inference service so it
  can be scaled independently of the API layer.
