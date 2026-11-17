"""
One function to run a user question end-to-end (guardrails -> agent graph), shared by
the FastAPI endpoint and the MCP `ask_assistant` tool.
"""

import uuid
from typing import TypedDict

import logfire
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents.graph import rag_agent, record_turn
from app.agents.state import split_turn
from app.guardrails import guard
from app.tools.docs import citation_id


class SourceOut(TypedDict):
    id: str
    source: str
    chunk_index: int | None
    content: str
    score: float


class AgentResult(TypedDict):
    question: str
    answer: str
    thread_id: str
    thought_process: list[str]
    status: str
    sources: list[SourceOut]


def _describe_call(call: dict) -> str:
    args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
    return f"Tool: {call['name']}({args})"


def summarise_turn(messages: list) -> tuple[str, list[str], list[SourceOut]]:
    """Extract (answer, step log, de-duplicated sources) from the current turn's messages."""
    _, turn = split_turn(messages)
    steps: list[str] = []
    sources: dict[str, SourceOut] = {}

    for m in turn:
        if isinstance(m, AIMessage) and m.tool_calls:
            steps.extend(_describe_call(c) for c in m.tool_calls)
        elif isinstance(m, ToolMessage):
            for doc in m.artifact or []:
                sid = doc.get("id") or citation_id(doc)
                if sid not in sources or doc["score"] > sources[sid]["score"]:
                    sources[sid] = {
                        "id": sid,
                        "source": doc["source"],
                        "chunk_index": doc.get("chunk_index"),
                        "content": doc["content"],
                        "score": doc["score"],
                    }

    final = next((m for m in reversed(turn) if isinstance(m, AIMessage) and not m.tool_calls), None)
    answer = final.content if final and isinstance(final.content, str) else ""
    steps.append("Answered from memory (no tools)" if not steps else f"Answered after {len(steps)} tool call(s)")
    return answer, steps, sorted(sources.values(), key=lambda s: -s["score"])


def ask(question: str, thread_id: str | None = None) -> AgentResult:
    question = question.strip()
    thread_id = thread_id or str(uuid.uuid4())

    rail_fired, rail_response = guard(question)
    if rail_fired:
        record_turn(thread_id, question, rail_response)
        return AgentResult(
            question=question,
            answer=rail_response,
            thread_id=thread_id,
            thought_process=["Guardrails fired", "Tools: skipped"],
            status="Handled by guardrails.",
            sources=[],
        )

    with logfire.span("Agent run", thread_id=thread_id):
        final = rag_agent.invoke(
            {"messages": [HumanMessage(question)]},
            config={"configurable": {"thread_id": thread_id}, "recursion_limit": 25},
        )

    answer, steps, sources = summarise_turn(final["messages"])
    return AgentResult(
        question=question,
        answer=answer,
        thread_id=thread_id,
        thought_process=steps,
        status="Response generated.",
        sources=sources,
    )
