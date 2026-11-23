import itertools

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents import graph, service
from app.agents.nodes import agent as agent_mod
from app.agents.structured import SUBMIT_ANSWER, SUBMIT_ANSWER_TOOL, from_text, validate
from app.services import feedback
from app.tools import docs as docs_mod
from tests.fakes import fake_list_sources, fake_read_chunks, fake_search

_ids = itertools.count()


class SubmittingLLM:
    """Searches once, then finishes by calling submit_answer with the given arguments."""

    def __init__(self, submit_args):
        self.submit_args = submit_args

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        last = messages[-1]
        if isinstance(last, HumanMessage):
            return AIMessage(content="", tool_calls=[{"name": "search_docs", "args": {"query": last.content}, "id": f"s{next(_ids)}"}])
        return AIMessage(content="", tool_calls=[{"name": SUBMIT_ANSWER, "args": self.submit_args, "id": f"a{next(_ids)}"}])


@pytest.fixture
def wired(monkeypatch):
    feedback.reset(":memory:")
    monkeypatch.setattr(docs_mod, "search_enterprise_knowledge", fake_search)
    monkeypatch.setattr(docs_mod, "rerank_documents", lambda q, d, top_n=5: d[:top_n])
    monkeypatch.setattr(docs_mod, "list_sources", fake_list_sources)
    monkeypatch.setattr(docs_mod, "read_chunks", fake_read_chunks)
    monkeypatch.setattr(service, "guard", lambda q: (False, None))

    def use(llm):
        monkeypatch.setattr(agent_mod, "_tool_llm", llm)
        monkeypatch.setattr(agent_mod, "_plain_llm", llm)

    yield use
    feedback.reset()


# --- validation (pure) ---------------------------------------------------------------------

def test_schema_is_exposed_as_submit_answer_tool():
    fn = SUBMIT_ANSWER_TOOL["function"]
    assert fn["name"] == "submit_answer"
    assert {"answer", "answered_from_docs", "confidence"} <= set(fn["parameters"]["required"])
    assert fn["parameters"]["properties"]["confidence"]["enum"] == ["high", "medium", "low"]


def test_validate_keeps_only_retrieved_citations():
    rec = validate(
        {"answer": "Use Forbid [a#1]. Also see [made-up#9].", "citations": ["a#1", "ghost#2"],
         "answered_from_docs": True, "confidence": "high", "follow_up_questions": ["q1", " ", "q2", "q3", "q4"]},
        retrieved_ids=["a#1", "b#2"],
    )
    assert rec["citations"] == ["a#1"]
    assert sorted(rec["invalid_citations"]) == ["ghost#2", "made-up#9"]
    assert "[made-up#9]" not in rec["answer"] and "[a#1]" in rec["answer"]
    assert rec["confidence"] == "high" and rec["source"] == "structured"
    assert rec["follow_up_questions"] == ["q1", "q2", "q3"]  # blanks dropped, trimmed to 3, not rejected


def test_answer_without_valid_citation_is_downgraded():
    rec = validate({"answer": "Trust me [ghost#1].", "citations": ["ghost#1"], "answered_from_docs": True,
                    "confidence": "high"}, retrieved_ids=["a#1"])
    assert rec["confidence"] == "low" and rec["answered_from_docs"] is False and rec["citations"] == []


def test_malformed_arguments_fall_back_to_text():
    rec = validate({"answer": "Use Forbid [a#1].", "confidence": "certain"}, retrieved_ids=["a#1"])
    assert rec["source"] == "text" and rec["citations"] == ["a#1"]


def test_from_text_detects_not_found():
    rec = from_text("I couldn't find this in the documentation.", ["a#1"])
    assert rec["answered_from_docs"] is False and rec["confidence"] == "low"


# --- graph ---------------------------------------------------------------------------------

def test_routing():
    def state(msg):
        return {"messages": [HumanMessage("q"), msg]}

    assert graph.route_after_agent(state(AIMessage(content="text"))) == "__end__"
    assert graph.route_after_agent(state(AIMessage(content="", tool_calls=[{"name": "search_docs", "args": {}, "id": "1"}]))) == "tools"
    assert graph.route_after_agent(state(AIMessage(content="", tool_calls=[{"name": SUBMIT_ANSWER, "args": {}, "id": "2"}]))) == "finalize"


def test_finalize_acknowledges_every_tool_call():
    calls = [{"name": SUBMIT_ANSWER, "args": {"answer": "x", "answered_from_docs": False, "confidence": "low"}, "id": "c1"},
             {"name": "search_docs", "args": {"query": "y"}, "id": "c2"}]
    out = graph.finalize_node({"messages": [HumanMessage("q"), AIMessage(content="", tool_calls=calls)]})["messages"]
    assert [m.tool_call_id for m in out if isinstance(m, ToolMessage)] == ["c1", "c2"]
    assert out[-1].content == "x" and out[-1].additional_kwargs["structured"]["source"] == "structured"


def test_end_to_end_structured_answer(wired):
    wired(SubmittingLLM({
        "answer": "Set concurrencyPolicy: Forbid [cronjobs.docx#3]. See also [nowhere.pdf#1].",
        "citations": ["cronjobs.docx#3"], "answered_from_docs": True, "confidence": "high",
        "follow_up_questions": ["What does Replace do?"],
    }))
    result = service.ask("How do I stop overlapping CronJob runs?")
    s = result["structured"]
    assert s["source"] == "structured" and s["citations"] == ["cronjobs.docx#3"]
    assert s["invalid_citations"] == ["nowhere.pdf#1"] and "nowhere.pdf" not in result["answer"]
    assert s["follow_up_questions"] == ["What does Replace do?"]
    assert not any("submit_answer" in step for step in result["thought_process"])

    # The next turn's prompt sees the answer as normal history (no dangling tool calls).
    follow = SubmittingLLM({"answer": "ok", "answered_from_docs": False, "confidence": "low"})
    wired(follow)
    service.ask("thanks", thread_id=result["thread_id"])


def test_api_returns_structured_fields(wired):
    from app import main

    wired(SubmittingLLM({"answer": "Use Forbid [cronjobs.docx#3].", "citations": ["cronjobs.docx#3"],
                         "answered_from_docs": True, "confidence": "medium"}))
    body = TestClient(main.app).post("/query", json={"q": "How do I stop overlapping CronJob runs?"}).json()
    assert body["structured"]["confidence"] == "medium"
    assert body["structured"]["citations"] == ["cronjobs.docx#3"]


def test_budget_path_forces_structured_answer(wired, monkeypatch):
    class AlwaysSearches(SubmittingLLM):
        def invoke(self, messages):
            return AIMessage(content="", tool_calls=[{"name": "search_docs", "args": {"query": "again"}, "id": f"s{next(_ids)}"}])

    class JsonMode:  # JSON mode returns the schema as a JSON string, no tool call
        def invoke(self, messages):
            assert "JSON" in messages[-1].content
            return AIMessage(content='{"answer": "Forbid [cronjobs.docx#3].", "citations": ["cronjobs.docx#3"], '
                                     '"answered_from_docs": true, "confidence": "medium"}')

    wired(AlwaysSearches({}))
    monkeypatch.setattr(agent_mod, "_final_llm", JsonMode())
    monkeypatch.setattr(agent_mod.settings, "MAX_TOOL_ROUNDS", 2)
    result = service.ask("How do I stop overlapping CronJob runs?")
    assert result["structured"]["source"] == "structured_text"
    assert result["answer"] == "Forbid [cronjobs.docx#3]."
    assert result["structured"]["citations"] == ["cronjobs.docx#3"]
    assert sum(s.startswith("Tool: search_docs") for s in result["thought_process"]) == 2


def test_schema_written_as_json_text_is_parsed():
    text = '{"answer": "Use Forbid [a#1].", "citations": ["a#1"], "answered_from_docs": true, "confidence": "high"}'
    rec = from_text(text, ["a#1"])
    assert rec["source"] == "structured_text" and rec["answer"] == "Use Forbid [a#1]." and rec["citations"] == ["a#1"]
    fenced = from_text("```json\n" + text + "\n```", ["a#1"])
    assert fenced["source"] == "structured_text"
    partial = from_text('{"answer": "Sorry, I can only help with the docs."}', [])
    assert partial["source"] == "text" and partial["answer"] == "Sorry, I can only help with the docs."
