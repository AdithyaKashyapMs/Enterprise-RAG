"""
The documentation tools — one implementation shared by the LangGraph agent
(app/agents/tools.py) and the MCP server (app/mcp_server.py), so both always
expose identical behaviour.

Every chunk carries a citation id "<source>#<chunk_index>" (e.g. "cronjobs.docx#12")
that stays stable across searches and re-ingestion.
"""

import re

import logfire

from app.config import settings
from app.services import personalise
from app.services.retrieval.diversity import mmr
from app.services.retrieval.keyword_index import keyword_search, rrf_fuse
from app.services.retrieval.qdrant_service import (
    RetrievalError,
    list_sources,
    read_chunks,
    search_enterprise_knowledge,
)
from app.services.retrieval.ranking_service import rerank_documents

SEARCH_CANDIDATES = 15
MAX_TOP_K = 10
MAX_READ_CHUNKS = 6

_CITATION_RE = re.compile(r"\[([^\[\]]+#\d+)\]")


def citation_id(doc: dict) -> str:
    idx = doc.get("chunk_index")
    return f"{doc['source']}#{idx if idx is not None else '?'}"


def cited_ids(answer: str, available_ids: list[str]) -> list[str]:
    """Citation ids the answer actually uses, e.g. "[cronjobs.docx#12]", limited to known passages."""
    known = set(available_ids)
    return list(dict.fromkeys(c for c in _CITATION_RE.findall(answer) if c in known))


def _with_ids(docs: list[dict]) -> list[dict]:
    return [{"id": citation_id(d), **d} for d in docs]


def generate_candidates(
    query: str, limit: int = SEARCH_CANDIDATES, source: str | None = None, hybrid: bool | None = None
) -> list[dict]:
    """
    Stage 1, candidate generation: cheap, broad recall over the whole index.
    Hybrid mode runs vector search and BM25 keyword search and fuses them with RRF;
    if the keyword index is unavailable it falls back to vector search alone.
    """
    vector = search_enterprise_knowledge(query, limit=limit, source=source)
    if not (settings.HYBRID_SEARCH if hybrid is None else hybrid):
        return [{**d, "vector_score": d["score"], "vector_rank": r} for r, d in enumerate(vector, 1)]
    try:
        keyword = keyword_search(query, limit=limit, source=source)
    except (RetrievalError, Exception) as e:
        logfire.warning(f"Keyword search unavailable, using vector search only: {e}")
        return [{**d, "vector_score": d["score"], "vector_rank": r} for r, d in enumerate(vector, 1)]
    return rrf_fuse({"vector": vector, "bm25": keyword}, limit=limit)


def rank_candidates(
    query: str,
    candidates: list[dict],
    top_k: int,
    source: str | None = None,
    use_feedback: bool = True,
    ranker=None,
) -> list[dict]:
    """
    Stages 2-3, applied to the candidates:
      2. ranking:     cross-encoder rerank of every candidate
      3. re-ranking:  personalisation (learned ranker or capped 👍/👎 boost, plus a small
                      exploration boost for new documents), then MMR diversity with max
                      2 chunks per file (no cap when the search is restricted to one file)
    use_feedback=False skips personalisation (evals); `ranker` overrides the saved model.
    """
    ranked = _with_ids(rerank_documents(query, candidates, top_n=len(candidates)))
    ranked = [{**d, "rerank_score": d["score"]} for d in ranked]
    ranked = personalise.apply(ranked, ranker=ranker) if use_feedback else personalise.describe(ranked)
    return mmr(ranked, top_k, max_per_source=None if source else 2)


def search_docs(query: str, top_k: int = 5, source: str | None = None) -> list[dict]:
    """
    Search shaped like a recommender: candidate generation (vector + BM25, fused) ->
    ranking (cross-encoder) -> re-ranking (personalisation + diversity).
    Returns chunks with citation ids.
    """
    top_k = max(1, min(top_k, MAX_TOP_K))
    candidates = generate_candidates(query, limit=max(SEARCH_CANDIDATES, top_k), source=source)
    return rank_candidates(query, candidates, top_k, source)


def list_documents() -> list[dict]:
    """Every indexed source file with its chunk count."""
    return list_sources()


def read_document(source: str, start_chunk: int = 0, num_chunks: int = 3) -> list[dict]:
    """Read consecutive chunks of one file — e.g. the chunks around a search hit."""
    num_chunks = max(1, min(num_chunks, MAX_READ_CHUNKS))
    return _with_ids(read_chunks(source, max(0, start_chunk), num_chunks))


def format_chunks(docs: list[dict], empty_message: str) -> str:
    """Render chunks as text for an LLM, each headed by its citation id."""
    if not docs:
        return empty_message
    return "\n\n".join(f"[{d['id']}] (score {d['score']:.2f})\n{d['content']}" for d in docs)
