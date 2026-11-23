import os
import sqlite3

import logfire
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.agents.nodes.agent import agent_node
from app.agents.state import AgentState, split_turn
from app.agents.structured import SUBMIT_ANSWER, validate
from app.agents.tools import TOOLS


def retrieved_ids(turn: list) -> list[str]:
    """Ids of every passage a tool returned in this turn: the only valid citations."""
    return [doc["id"] for m in turn if isinstance(m, ToolMessage) for doc in (m.artifact or []) if doc.get("id")]


def route_after_agent(state: AgentState) -> str:
    """Search/read tool calls -> tools; submit_answer -> finalize; plain text -> done."""
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []
    if any(c["name"] == SUBMIT_ANSWER for c in calls):
        return "finalize"
    return "tools" if calls else END


def finalize_node(state: AgentState):
    """
    Turn the submit_answer call into the final answer: validate it against the passages
    retrieved this turn, acknowledge every tool call (so the history stays well-formed for
    the LLM API), and emit the answer as a normal assistant message carrying the structure.
    """
    last = state["messages"][-1]
    _, turn = split_turn(state["messages"])
    call = next(c for c in last.tool_calls if c["name"] == SUBMIT_ANSWER)
    record = validate(call["args"], retrieved_ids(turn))
    if record["invalid_citations"]:
        logfire.warning(f"Dropped citations that were never retrieved: {record['invalid_citations']}")
    acks = [ToolMessage(content="Answer submitted.", tool_call_id=c["id"], name=c["name"]) for c in last.tool_calls]
    return {"messages": [*acks, AIMessage(content=record["answer"], additional_kwargs={"structured": record})]}


def build_workflow() -> StateGraph:
    """
    START -> agent --(search / read tools)--> tools -> agent -> ...
                   --(submit_answer)--------> finalize -> END   (structured, validated answer)
                   --(plain text)-----------> END               (budget / fallback paths)

    The agent node caps the number of tool rounds per question (MAX_TOOL_ROUNDS).
    """
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    # handle_tool_errors: a failing tool returns an error message to the model instead of crashing.
    workflow.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
    workflow.add_node("finalize", finalize_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "finalize": "finalize", END: END}
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("finalize", END)
    return workflow


def build_checkpointer():
    """
    Conversation memory keyed by thread_id.
    SQLite (survives restarts) when langgraph-checkpoint-sqlite is installed;
    in-memory otherwise. Set CHECKPOINT_DB=":memory:" to force in-memory.
    """
    db_path = os.getenv("CHECKPOINT_DB", "checkpoints.sqlite")
    if db_path != ":memory:":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            logfire.info(f"Using SQLite conversation memory at {db_path}")
            return SqliteSaver(sqlite3.connect(db_path, check_same_thread=False))
        except ImportError:
            logfire.warning("langgraph-checkpoint-sqlite not installed — memory is in-process only.")

    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver()


rag_agent = build_workflow().compile(checkpointer=build_checkpointer())


def record_turn(thread_id: str, user_msg: str, assistant_msg: str) -> None:
    """Append a turn handled outside the graph (e.g. by guardrails) to the thread's memory."""
    rag_agent.update_state(
        {"configurable": {"thread_id": thread_id}},
        {"messages": [HumanMessage(user_msg), AIMessage(assistant_msg)]},
        as_node="agent",
    )
