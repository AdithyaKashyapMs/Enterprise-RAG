"""
The documentation tools — one implementation shared by the LangGraph agent
(app/agents/tools.py) and the MCP server (app/mcp_server.py), so both always
expose identical behaviour.

Every chunk carries a citation id "<source>#<chunk_index>" (e.g. "cronjobs.docx#12")
that stays stable across searches and re-ingestion.
"""

from app.services.retrieval.qdrant_service import list_sources, read_chunks, search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents

SEARCH_CANDIDATES = 15
MAX_TOP_K = 10
MAX_READ_CHUNKS = 6


def citation_id(doc: dict) -> str:
    idx = doc.get("chunk_index")
    return f"{doc['source']}#{idx if idx is not None else '?'}"


def _with_ids(docs: list[dict]) -> list[dict]:
    return [{"id": citation_id(d), **d} for d in docs]


def search_docs(query: str, top_k: int = 5, source: str | None = None) -> list[dict]:
    """Semantic search (vector recall + cross-encoder rerank). Returns chunks with citation ids."""
    top_k = max(1, min(top_k, MAX_TOP_K))
    candidates = search_enterprise_knowledge(query, limit=max(SEARCH_CANDIDATES, top_k), source=source)
    return _with_ids(rerank_documents(query, candidates, top_n=top_k))


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
