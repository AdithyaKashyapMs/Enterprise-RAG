import os
import tempfile
import time

import logfire
from flashrank import Ranker, RerankRequest

_ranker: Ranker | None = None


def _get_ranker() -> Ranker:
    """
    Lazily load FlashRank's default cross-encoder (ms-marco-TinyBERT-L-2-v2, ONNX).
    A cross-encoder scores (query, passage) pairs jointly, which is much more precise
    than cosine similarity on independently computed embeddings.
    """
    global _ranker
    if _ranker is None:
        cache_dir = os.path.join(tempfile.gettempdir(), "flashrank")
        logfire.info(f"Loading FlashRank model (cache: {cache_dir})")
        _ranker = Ranker(cache_dir=cache_dir)
    return _ranker


def rerank_documents(query: str, documents: list[dict], top_n: int = 5) -> list[dict]:
    """
    Rerank retrieved docs ({content, source, chunk_index, score}) against the query.
    Returns the top_n docs with `score` replaced by the reranker's score.
    Falls back to the original vector-search order if reranking fails.
    """
    if not documents:
        return []

    start = time.perf_counter()
    try:
        passages = [{"id": i, "text": d["content"]} for i, d in enumerate(documents)]
        results = _get_ranker().rerank(RerankRequest(query=query, passages=passages))
        # Keep every original field (source, chunk_index, ...) and swap in the rerank score.
        reranked = [{**documents[r["id"]], "score": float(r["score"])} for r in results[:top_n]]
        logfire.info(
            f"Reranked {len(documents)} docs in {time.perf_counter() - start:.2f}s; "
            f"top score {reranked[0]['score']:.3f}"
        )
        return reranked
    except Exception as e:
        logfire.error(f"Reranking failed, using vector-search order: {e}")
        return documents[:top_n]
