"""Tool-calling agent + API tests, with the LLM and Qdrant replaced by fakes."""

import uuid

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from app.agents import graph as graph_mod
from app.agents import service
from app.agents.nodes import agent as agent_mod
from app.tools import docs as docs_mod
from tests.fakes import ScriptedLLM, fake_list_sources, fake_read_chunks, fake_search


@pytest.fixture
def llm(monkeypatch):
    fake = ScriptedLLM()
    monkeypatch.setattr(agent_mod, "_tool_llm", fake)
    monkeypatch.setattr(agent_mod, "_plain_llm", fake)
    monkeypatch.setattr(docs_mod, "search_enterprise_knowledge", fake_search)
    monkeypatch.setattr(docs_mod, "rerank_documents", lambda q, d, top_n=5: d[:top_n])
    monkeypatch.setattr(docs_mod, "list_sources", fake_list_sources)
    monkeypatch.setattr(docs_mod, "read_chunks", fake_read_chunks)
    monkeypatch.setattr(service, "guard", lambda q: (False, None))
    return fake


def test_agent_calls_search_and_cites_sources(llm):
    result = service.ask("How do I stop overlapping cronjob runs with concurrencyPolicy?")

    assert result["thought_process"][0].startswith("Tool: search_docs(query=")
    assert result["sources"][0]["id"] == "cronjobs.docx#3"
    assert "[cronjobs.docx#3]" in result["answer"]
    # The model saw the system prompt first and the tool result last.
    final_prompt = llm.prompts[-1]
    assert isinstance(final_prompt[0], SystemMessage)
    assert isinstance(final_prompt[-1], ToolMessage)


def test_memory_question_uses_no_tools(llm):
    thread = str(uuid.uuid4())
    service.ask("How do I stop overlapping cronjob runs?", thread)
    result = service.ask("what did I just ask you?", thread)

    assert result["thought_process"] == ["Answered from memory (no tools)"]
    assert result["sources"] == []
    assert "overlapping cronjob runs" in result["answer"]


def test_previous_turn_tool_traffic_is_compacted(llm):
    thread = str(uuid.uuid4())
    service.ask("How do I stop overlapping cronjob runs?", thread)
    service.ask("How do I view job pod logs?", thread)

    second_turn_first_prompt = next(p for p in llm.prompts if p[-1].content == "How do I view job pod logs?")
    # Earlier turn appears only as question + final answer, not tool calls/results.
    assert not any(isinstance(m, ToolMessage) for m in second_turn_first_prompt)
    assert not any(isinstance(m, AIMessage) and m.tool_calls for m in second_turn_first_prompt)


def test_tool_budget_forces_final_answer(llm, monkeypatch):
    class AlwaysSearch(ScriptedLLM):
        def invoke(self, messages):
            self.prompts.append(messages)
            if isinstance(messages[-1], SystemMessage):  # the "answer now" nudge
                return AIMessage(content="Best effort answer.")
            return AIMessage(content="", tool_calls=[{"name": "search_docs", "args": {"query": "x"}, "id": str(uuid.uuid4())}])

    greedy = AlwaysSearch()
    monkeypatch.setattr(agent_mod, "_tool_llm", greedy)
    monkeypatch.setattr(agent_mod, "_plain_llm", greedy)
    monkeypatch.setattr(agent_mod.settings, "MAX_TOOL_ROUNDS", 2)

    result = service.ask("loop forever on pods")
    assert result["answer"] == "Best effort answer."
    assert sum(s.startswith("Tool:") for s in result["thought_process"]) == 2


def test_multi_tool_plan_filter_and_read(llm, monkeypatch):
    planner = ScriptedLLM(
        tool_plan=[
            {"name": "search_docs", "args": {"query": "history limit", "source": "cronjobs.docx"}},
            {"name": "read_document", "args": {"source": "cronjobs.docx", "start_chunk": 3, "num_chunks": 2}},
        ]
    )
    monkeypatch.setattr(agent_mod, "_tool_llm", planner)
    monkeypatch.setattr(agent_mod, "_plain_llm", planner)

    result = service.ask("How many finished cronjob runs are kept?")
    ids = {s["id"] for s in result["sources"]}
    assert ids == {"cronjobs.docx#3", "cronjobs.docx#4"}  # de-duplicated across both tools
    assert all(s["source"] == "cronjobs.docx" for s in result["sources"])


def test_degrades_to_single_search_when_tool_call_request_fails(llm, monkeypatch):
    class BrokenTools(ScriptedLLM):
        def invoke(self, messages):
            raise RuntimeError("tool_use_failed")

    monkeypatch.setattr(agent_mod, "_tool_llm", BrokenTools())
    plain = ScriptedLLM()
    plain.invoke = lambda messages: AIMessage(content="Fallback answer [cronjobs.docx#3].")
    monkeypatch.setattr(agent_mod, "_plain_llm", plain)

    result = service.ask("overlapping cronjob runs concurrencyPolicy")
    assert result["answer"].startswith("Fallback answer")
    assert result["sources"][0]["id"] == "cronjobs.docx#3"


def test_guardrail_turns_are_recorded_in_memory(monkeypatch):
    monkeypatch.setattr(service, "guard", lambda q: (True, "Hello! I'm your Enterprise IT Assistant."))
    thread = str(uuid.uuid4())
    result = service.ask("hello", thread)
    assert result["status"] == "Handled by guardrails."
    state = graph_mod.rag_agent.get_state({"configurable": {"thread_id": thread}})
    assert [m.type for m in state.values["messages"]] == ["human", "ai"]


def test_api_flow(llm):
    from app import main

    client = TestClient(main.app)  # not a context manager -> lifespan (NeMo) not started
    r = client.post("/query", json={"q": "How do I stop overlapping cronjob runs?"})
    assert r.status_code == 200
    body = r.json()
    assert body["thread_id"] and body["sources"][0]["id"] == "cronjobs.docx#3"

    r2 = client.post("/query", json={"q": "How do I stop overlapping cronjob runs?"})
    assert r2.json()["thread_id"] != body["thread_id"]
    assert client.post("/query", json={"q": ""}).status_code == 422


def test_api_returns_500_on_failure(monkeypatch):
    from app import main

    def boom(q, t):
        raise RuntimeError("boom")

    monkeypatch.setattr(main.service, "ask", boom)
    assert TestClient(main.app).post("/query", json={"q": "pods?"}).status_code == 500


def test_api_returns_503_when_llm_rate_limited(monkeypatch):
    from app import main

    class RateLimited(Exception):
        status_code = 429

    def limited(q, t=None):
        raise RateLimited("tokens per day")

    monkeypatch.setattr(main.service, "ask", limited)
    r = TestClient(main.app).post("/query", json={"q": "pods?"})
    assert r.status_code == 503 and "rate limit" in r.json()["detail"]


def test_model_fallback_chain_order(monkeypatch):
    from app.gateway import client

    monkeypatch.setattr(client.settings, "GROQ_API_KEY", "k1")
    monkeypatch.setattr(client.settings, "GROQ_FALLBACK_API_KEY", None)
    monkeypatch.setattr(client.settings, "GROQ_MODEL", "big")
    monkeypatch.setattr(client.settings, "GROQ_FALLBACK_MODELS", ["small", "big", "other"])
    monkeypatch.setattr(client.settings, "PORTKEY_API_KEY", None)
    assert [m.model_name for m in client._base_models("agent", 0.0)] == ["big", "small", "other"]
