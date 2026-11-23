"""
LangChain tool wrappers around app/tools/docs.py for the LangGraph agent.

Each tool returns (text for the LLM, artifact). The artifact — the raw chunks —
rides on the ToolMessage, so the API can return structured sources without
re-parsing the model-facing text.
"""

from langchain_core.tools import tool

from app.services.retrieval.qdrant_service import RetrievalError
from app.tools import docs


@tool(response_format="content_and_artifact")
def search_docs(query: str, top_k: int = 5, source: str | None = None) -> tuple[str, list[dict]]:
    """Search the internal technical documentation (Kubernetes, Jobs/CronJobs, autoscaling,
    Databricks jobs, cluster architecture) and return the most relevant passages.

    Args:
        query: A specific, standalone search query. Rewrite follow-ups ("that", "it") into full terms.
        top_k: Number of passages to return (1-10).
        source: Optional exact file name (from list_documents) to search within only that document.
    """
    try:
        results = docs.search_docs(query, top_k=top_k, source=source)
    except RetrievalError as e:
        return f"Search unavailable: {e}", []
    return docs.format_chunks(results, "No matching passages found."), results


@tool(response_format="content_and_artifact")
def list_documents() -> tuple[str, list[dict]]:
    """List every document in the knowledge base with its number of chunks.
    Use this when the user asks what documentation exists, or to find a file name for `source`."""
    try:
        sources = docs.list_documents()
    except RetrievalError as e:
        return f"Listing unavailable: {e}", []
    if not sources:
        return "The knowledge base is empty.", []
    return "\n".join(f"- {s['source']} ({s['chunks']} chunks)" for s in sources), []


@tool(response_format="content_and_artifact")
def read_document(source: str, start_chunk: int = 0, num_chunks: int = 3) -> tuple[str, list[dict]]:
    """Read consecutive chunks of one document, e.g. to see the full YAML or steps around a
    search hit. A hit with id "cronjobs.docx#12" can be expanded with source="cronjobs.docx",
    start_chunk=11, num_chunks=3.

    Args:
        source: Exact file name, e.g. "cronjobs.docx".
        start_chunk: Index of the first chunk to read.
        num_chunks: How many consecutive chunks to read (1-4). Usually 2-3.
    """
    try:
        chunks = docs.read_document(source, start_chunk=start_chunk, num_chunks=num_chunks)
    except RetrievalError as e:
        return f"Read unavailable: {e}", []
    return docs.format_chunks(chunks, f"No chunks found for '{source}' at that position."), chunks


TOOLS = [search_docs, list_documents, read_document]
