"""
BM25 keyword search, and Reciprocal Rank Fusion (RRF) to combine it with vector search.

Why both: embeddings capture meaning ("stop overlapping runs" ≈ "concurrencyPolicy: Forbid")
but are weak on exact identifiers — field names, commands, error codes. BM25 is the opposite:
exact terms, no notion of meaning. Hybrid search runs both and fuses the two rankings.

BM25 score of a chunk d for query q (Okapi BM25):

    Σ_{t ∈ q}  idf(t) · tf(t,d) · (k1 + 1) / (tf(t,d) + k1 · (1 − b + b · |d| / avgdl))

  idf(t) = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))   rare terms count more
  k1     = term-frequency saturation (the 10th "kubectl" adds little over the 3rd)
  b      = length normalisation (long chunks don't win just by being long)

RRF merges ranked lists using ranks only, so the two very different score scales
(cosine similarity vs. BM25) never have to be calibrated against each other:

    rrf(d) = Σ_{lists} 1 / (k + rank_in_list(d)),   k = 60

The index is built in memory from the Qdrant collection on first use (~5k chunks builds in
about a second). At larger scale this would move to Qdrant sparse vectors or a search engine.
"""

import math
import re
from collections import Counter, defaultdict
from threading import Lock

import logfire

K1 = 1.5
B = 0.75
RRF_K = 60

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.\-]*[a-z0-9]|[a-z0-9]")
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i if in into is it its my of on or so "
    "that the their them then there these this to was what when where which who why will with "
    "you your".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase terms; keeps identifiers like 'kubectl', 'max_concurrent_runs', 'next.js' whole."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def _key(doc: dict) -> tuple:
    return (doc["source"], doc.get("chunk_index"))


class BM25Index:
    def __init__(self, docs: list[dict]):
        self.docs = docs
        self.term_freqs = [Counter(tokenize(d["content"])) for d in docs]
        self.lengths = [sum(tf.values()) for tf in self.term_freqs]
        self.avgdl = (sum(self.lengths) / len(docs)) if docs else 0.0
        self.postings: dict[str, list[int]] = defaultdict(list)  # term -> ids of chunks containing it
        for i, tf in enumerate(self.term_freqs):
            for term in tf:
                self.postings[term].append(i)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - len(ids) + 0.5) / (len(ids) + 0.5)) for t, ids in self.postings.items()}

    def search(self, query: str, limit: int = 15, source: str | None = None) -> list[dict]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in self.postings[term]:
                tf = self.term_freqs[i][term]
                norm = K1 * (1 - B + B * self.lengths[i] / self.avgdl)
                scores[i] += idf * tf * (K1 + 1) / (tf + norm)
        hits = [i for i in scores if source is None or self.docs[i]["source"] == source]
        hits.sort(key=lambda i: -scores[i])
        return [{**self.docs[i], "score": scores[i]} for i in hits[:limit]]


_index: BM25Index | None = None
_lock = Lock()


def get_index() -> BM25Index:
    global _index
    with _lock:
        if _index is None:
            from app.services.retrieval.qdrant_service import iter_chunks

            with logfire.span("Build BM25 index"):
                _index = BM25Index(list(iter_chunks()))
                logfire.info(f"BM25 index built over {len(_index.docs)} chunks, {len(_index.idf)} terms")
    return _index


def reset_index() -> None:
    """Drop the in-memory index (e.g. after re-ingestion); it is rebuilt on the next search."""
    global _index
    with _lock:
        _index = None


def keyword_search(query: str, limit: int = 15, source: str | None = None) -> list[dict]:
    return get_index().search(query, limit=limit, source=source)


def rrf_fuse(ranked_lists: dict[str, list[dict]], limit: int, k: int = RRF_K) -> list[dict]:
    """
    Fuse ranked lists of chunks with Reciprocal Rank Fusion. Each returned chunk keeps the
    first list's fields, gets `score` = its RRF score, and records its score and rank in every
    list it appeared in (e.g. "vector_score", "bm25_rank") — later used as ranking features.
    """
    fused: dict[tuple, dict] = {}
    for name, docs in ranked_lists.items():
        for rank, doc in enumerate(docs, 1):
            entry = fused.setdefault(_key(doc), {**doc, "score": 0.0})
            entry["score"] += 1 / (k + rank)
            entry[f"{name}_score"] = doc["score"]
            entry[f"{name}_rank"] = rank
    return sorted(fused.values(), key=lambda d: -d["score"])[:limit]
