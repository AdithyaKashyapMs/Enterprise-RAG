"""MCP server tests through a real MCP client session (in-process transport)."""

import anyio
import pytest
from mcp import Client

from app import mcp_server
from app.agents import service
from app.tools import docs as docs_mod
from tests.fakes import fake_list_sources, fake_read_chunks, fake_search


@pytest.fixture(autouse=True)
def fake_backend(monkeypatch):
    monkeypatch.setattr(docs_mod, "search_enterprise_knowledge", fake_search)
    monkeypatch.setattr(docs_mod, "rerank_documents", lambda q, d, top_n=5: d[:top_n])
    monkeypatch.setattr(docs_mod, "list_sources", fake_list_sources)
    monkeypatch.setattr(docs_mod, "read_chunks", fake_read_chunks)


def call(name, args=None):
    async def go():
        async with Client(mcp_server.mcp) as client:
            return await client.call_tool(name, args or {})

    return anyio.run(go)


def test_lists_tools_with_schemas_and_annotations():
    async def go():
        async with Client(mcp_server.mcp) as client:
            return await client.list_tools()

    tools = {t.name: t for t in anyio.run(go).tools}
    assert set(tools) == {"search_docs", "list_documents", "read_document", "ask_assistant"}
    assert tools["search_docs"].annotations.read_only_hint is True
    assert "query" in tools["search_docs"].input_schema["required"]
    assert tools["search_docs"].output_schema is not None


def test_search_docs_returns_structured_passages():
    result = call("search_docs", {"query": "concurrencyPolicy Forbid overlapping", "top_k": 2})
    assert not result.is_error
    passages = result.structured_content["passages"]
    assert passages[0]["id"] == "cronjobs.docx#3"
    assert len(passages) == 2


def test_search_docs_source_filter():
    result = call("search_docs", {"query": "logs", "source": "monitor_job.docx"})
    assert {p["source"] for p in result.structured_content["passages"]} == {"monitor_job.docx"}


def test_list_documents():
    docs = call("list_documents").structured_content["documents"]
    assert {d["source"]: d["chunks"] for d in docs}["cronjobs.docx"] == 2


def test_read_document_and_missing_document_error():
    ok = call("read_document", {"source": "cronjobs.docx", "start_chunk": 3, "num_chunks": 2})
    assert [p["id"] for p in ok.structured_content["passages"]] == ["cronjobs.docx#3", "cronjobs.docx#4"]

    missing = call("read_document", {"source": "nope.pdf"})
    assert missing.is_error


def test_ask_assistant_wraps_agent(monkeypatch):
    monkeypatch.setattr(
        service,
        "ask",
        lambda q, t=None: {
            "question": q, "answer": "Use Forbid [cronjobs.docx#3].", "thread_id": "t-1",
            "thought_process": ["Tool: search_docs(query='x')"], "status": "ok",
            "sources": [dict(fake_search("Forbid")[0], id="cronjobs.docx#3")],
            "structured": {
                "answer": "Use Forbid [cronjobs.docx#3].", "citations": ["cronjobs.docx#3"],
                "answered_from_docs": True, "confidence": "high",
                "follow_up_questions": ["What does Replace do?"], "invalid_citations": [], "source": "structured",
            },
        },
    )
    out = call("ask_assistant", {"question": "overlapping runs?"}).structured_content
    assert out["thread_id"] == "t-1" and out["sources"][0]["id"] == "cronjobs.docx#3"
    assert out["confidence"] == "high" and out["citations"] == ["cronjobs.docx#3"]
    assert out["follow_up_questions"] == ["What does Replace do?"]
