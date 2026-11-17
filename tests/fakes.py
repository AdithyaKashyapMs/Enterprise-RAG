"""Shared fakes: a scripted tool-calling LLM and an in-memory corpus."""

import itertools

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

CORPUS = [
    {"content": "Set concurrencyPolicy: Forbid to skip overlapping CronJob runs.", "source": "cronjobs.docx", "chunk_index": 3},
    {"content": "successfulJobsHistoryLimit: 3 keeps three finished Jobs.", "source": "cronjobs.docx", "chunk_index": 4},
    {"content": "kubectl logs <pod-name> shows a Job pod's output.", "source": "monitor_job.docx", "chunk_index": 7},
    {"content": "HPA scales replicas between minReplicas and maxReplicas.", "source": "pods_autoscale.html", "chunk_index": 2},
]


def fake_search(query, limit=8, source=None):
    words = set(query.lower().split())
    docs = [d for d in CORPUS if source in (None, d["source"])]
    scored = [dict(d, score=float(len(words & set(d["content"].lower().split())))) for d in docs]
    return sorted(scored, key=lambda d: -d["score"])[:limit]


def fake_list_sources():
    counts = {}
    for d in CORPUS:
        counts.setdefault(d["source"], {"source": d["source"], "source_type": "true", "chunks": 0})["chunks"] += 1
    return sorted(counts.values(), key=lambda d: d["source"])


def fake_read_chunks(source, start_chunk=0, num_chunks=3):
    return [
        dict(d, score=0.0)
        for d in CORPUS
        if d["source"] == source and start_chunk <= d["chunk_index"] < start_chunk + num_chunks
    ]


_ids = itertools.count()


class ScriptedLLM:
    """
    Behaves like a tool-calling chat model:
      - on a new technical question: calls search_docs with the question
      - after a tool result: answers citing the first passage id it saw
      - on "what did I ask": answers from history without tools
    `prompts` records every message list it was invoked with.
    """

    def __init__(self, tool_plan=None):
        self.prompts = []
        self.tool_plan = tool_plan  # optional fixed list of tool calls for the first step

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        self.prompts.append(messages)
        last = messages[-1]
        if isinstance(last, ToolMessage):
            cited = last.content.split("]")[0].lstrip("[") if last.content.startswith("[") else None
            return AIMessage(content=f"Use concurrencyPolicy: Forbid [{cited}]." if cited else "I couldn't find this in the documentation.")
        if isinstance(last, HumanMessage):
            if "what did i" in last.content.lower():
                earlier = [m.content for m in messages[1:-1] if isinstance(m, HumanMessage)]
                return AIMessage(content=f"You asked: {earlier[-1] if earlier else 'nothing yet'}")
            calls = self.tool_plan or [{"name": "search_docs", "args": {"query": last.content}}]
            return AIMessage(content="", tool_calls=[dict(c, id=f"call_{next(_ids)}") for c in calls])
        return AIMessage(content="Final answer.")
