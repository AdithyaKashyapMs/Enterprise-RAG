"""
MCP server exposing the documentation knowledge base to any MCP client
(Claude Desktop, Claude Code, Cursor, ...).

Tools
  search_docs      semantic search + rerank, returns passages with citation ids   (read-only)
  list_documents   every indexed file and its chunk count                          (read-only)
  read_document    consecutive chunks of one file, e.g. around a search hit        (read-only)
  ask_assistant    the full agent: guardrails + tool-calling LLM + memory           (calls an LLM)

The first three need only Qdrant + the embedding model — the client's own model does
the reasoning. ask_assistant runs this project's agent end-to-end.

Run:
    python -m app.mcp_server                                  # stdio (Claude Desktop / Code, Cursor)
    python -m app.mcp_server --transport streamable-http --port 8765   # http://127.0.0.1:8765/mcp
"""

import argparse

import logfire

from app.config import settings

# Never write logs to stdout: on the stdio transport stdout carries the protocol.
logfire.configure(
    service_name="enterprise-rag-mcp",
    token=settings.LOGFIRE_TOKEN,
    send_to_logfire="if-token-present",
    console=False,
)

import anyio  # noqa: E402
from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402
from mcp.types import ToolAnnotations  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app.services.retrieval.qdrant_service import RetrievalError  # noqa: E402
from app.tools import docs  # noqa: E402

mcp = MCPServer(
    name="enterprise-docs",
    title="Enterprise Docs (RAG)",
    version="1.0.0",
    log_level="WARNING",
    instructions=(
        "Internal technical documentation: Kubernetes architecture, Jobs/CronJobs, pod "
        "autoscaling (HPA/VPA), job monitoring, and Databricks job management. "
        "Use search_docs for questions, read_document to expand around a hit, and cite "
        "passages by their id (e.g. cronjobs.docx#12). Use list_documents to see what exists."
    ),
)

READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)


class Passage(BaseModel):
    id: str = Field(description="Citation id: <source>#<chunk_index>")
    source: str
    chunk_index: int | None
    score: float = Field(description="Relevance score (rerank score for search, 0 for reads)")
    content: str


class SearchResult(BaseModel):
    query: str
    passages: list[Passage]


class DocumentInfo(BaseModel):
    source: str
    source_type: str | None
    chunks: int


class DocumentList(BaseModel):
    documents: list[DocumentInfo]


class DocumentSlice(BaseModel):
    source: str
    passages: list[Passage]


class Answer(BaseModel):
    answer: str
    thread_id: str = Field(description="Pass back to continue the same conversation")
    steps: list[str]
    sources: list[Passage]


async def _run(fn, *args, **kwargs):
    """Run blocking retrieval code off the event loop and map errors to MCP tool errors."""
    try:
        return await anyio.to_thread.run_sync(lambda: fn(*args, **kwargs))
    except RetrievalError as e:
        raise ToolError(str(e)) from e


@mcp.tool(annotations=READ_ONLY)
async def search_docs(query: str, top_k: int = 5, source: str | None = None) -> SearchResult:
    """Search the internal documentation and return the most relevant passages.

    Args:
        query: A specific, standalone search query.
        top_k: Number of passages to return (1-10).
        source: Optional exact file name (see list_documents) to search only that document.
    """
    results = await _run(docs.search_docs, query, top_k=top_k, source=source)
    return SearchResult(query=query, passages=[Passage(**r) for r in results])


@mcp.tool(annotations=READ_ONLY)
async def list_documents() -> DocumentList:
    """List every document in the knowledge base with its number of chunks."""
    sources = await _run(docs.list_documents)
    return DocumentList(documents=[DocumentInfo(**s) for s in sources])


@mcp.tool(annotations=READ_ONLY)
async def read_document(source: str, start_chunk: int = 0, num_chunks: int = 3) -> DocumentSlice:
    """Read consecutive chunks of one document, e.g. the full manifest around a search hit.
    A hit with id "cronjobs.docx#12" can be expanded with source="cronjobs.docx", start_chunk=11.

    Args:
        source: Exact file name, e.g. "cronjobs.docx".
        start_chunk: Index of the first chunk to read.
        num_chunks: Number of consecutive chunks (1-6).
    """
    chunks = await _run(docs.read_document, source, start_chunk=start_chunk, num_chunks=num_chunks)
    if not chunks:
        raise ToolError(f"No chunks found for '{source}' starting at {start_chunk}. Check list_documents.")
    return DocumentSlice(source=source, passages=[Passage(**c) for c in chunks])


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True))
async def ask_assistant(question: str, thread_id: str | None = None) -> Answer:
    """Ask this project's own RAG agent (guardrails + tool-calling LLM + conversation memory)
    and get a cited answer. Prefer search_docs if you want to reason over raw passages yourself.

    Args:
        question: The question to answer.
        thread_id: Optional id from a previous answer to continue that conversation.
    """
    from app.agents import service  # imported lazily: loads the LLM stack only when used

    try:
        result = await anyio.to_thread.run_sync(lambda: service.ask(question, thread_id))
    except Exception as e:
        logfire.exception(f"ask_assistant failed: {e}")
        raise ToolError(f"The assistant failed: {e}") from e
    return Answer(
        answer=result["answer"],
        thread_id=result["thread_id"],
        steps=result["thought_process"],
        sources=[Passage(**s) for s in result["sources"]],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Enterprise docs MCP server")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run("stdio")
    else:
        mcp.run("streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
