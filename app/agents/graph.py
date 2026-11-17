import os
import sqlite3

import logfire
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agents.nodes.agent import agent_node
from app.agents.state import AgentState
from app.agents.tools import TOOLS


def build_workflow() -> StateGraph:
    """
    START -> agent --(tool calls?)--> tools -> agent -> ... -> END

    The loop ends when the agent replies without tool calls; the agent node caps
    the number of tool rounds per question (MAX_TOOL_ROUNDS).
    """
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    # handle_tool_errors: a failing tool returns an error message to the model instead of crashing.
    workflow.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
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
