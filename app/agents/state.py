from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # add_messages appends to (and de-duplicates by id) the checkpointed history.
    messages: Annotated[list[AnyMessage], add_messages]


# Earlier turns are compressed to user questions + final answers, and only the most
# recent ones are kept, so prompts stay bounded however long a thread runs.
MAX_HISTORY_MESSAGES = 10


def split_turn(messages: list[AnyMessage]) -> tuple[list[AnyMessage], list[AnyMessage]]:
    """Split into (earlier turns, current turn). The current turn starts at the last user message."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[:i], messages[i:]
    return [], list(messages)


def compact_history(previous: list[AnyMessage]) -> list[AnyMessage]:
    """Keep only user messages and final assistant answers (drop tool traffic) from earlier turns."""
    kept = [
        m
        for m in previous
        if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls and m.content)
    ]
    kept = kept[-MAX_HISTORY_MESSAGES:]
    while kept and not isinstance(kept[0], HumanMessage):
        kept = kept[1:]
    return kept
