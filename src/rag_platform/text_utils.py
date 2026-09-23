"""
Shared tokenization for every lexical-overlap heuristic in the codebase
(the fallback reranker, the mock extractive generator, the faithfulness
guardrail). Centralized so all three agree on what counts as a
"meaningful" token — without this, common words like "the" or "and" match
everything and silently make every relevance/faithfulness score look
inflated, which is exactly the kind of bug that's invisible until you
actually run an off-topic question through the pipeline and notice it
didn't refuse.
"""
from __future__ import annotations

import re

# Deliberately small and English-specific — good enough for a demo corpus.
# Swap for a real stopword list (e.g. via nltk or spaCy) in production.
STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "was",
    "has", "have", "had", "this", "that", "with", "from", "your", "will",
    "may", "must", "should", "could", "would", "does", "did", "been",
    "being", "into", "onto", "than", "then", "them", "their", "there",
    "these", "those", "what", "when", "where", "which", "who", "how",
    "any", "each", "such", "only", "own", "same", "few", "more", "most",
    "other", "some", "per", "via", "our", "its", "within", "upon",
}


def _singularize(token: str) -> str:
    """
    Deliberately crude plural stripping — NOT a real stemmer (no lemmatization,
    no irregular plurals like 'policies'->'policy'). It exists only to stop
    the most common failure mode of exact-string lexical matching: "order"
    vs "orders" being treated as unrelated words. A production system should
    use real embeddings for this instead of trying to patch lexical matching
    further — this fix narrows one specific gap, it doesn't close the class
    of problem.
    """
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es") and token[-3] in "sxzh":
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def meaningful_tokens(text: str) -> set[str]:
    raw = re.findall(r"[a-z0-9]{3,}", text.lower())
    return {_singularize(t) for t in raw if t not in STOPWORDS}
