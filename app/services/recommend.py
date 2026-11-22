"""
Recommendations shown next to an answer, and starter questions for new users.

Related documents combine two classic approaches:
  content-based   documents whose chunks are closest (in embedding space) to the passages the
                  answer cited. Needs only the documents themselves, so it works from day one
                  and for brand-new documents (item cold start).
  collaborative   item-to-item co-occurrence: documents cited in *other* conversations that
                  also cited these passages ("people who read this also read"). Captures
                  connections embeddings miss, but needs interaction history — empty at first.
A collaborative suggestion needs support from MIN_SUPPORT conversations, to avoid
recommending from a single coincidence.

User cold start: a new user has no history, so starter questions come from what other users
asked most (popular, with a cited answer and no net 👎). With no traffic yet, curated
defaults fill the gaps.
"""

from typing import TypedDict

import logfire

from app.services import feedback
from app.services.retrieval.qdrant_service import similar_chunks

MIN_SUPPORT = 2
MAX_SEED_PASSAGES = 3   # nearest-neighbour lookups per answer (one Qdrant query each)

DEFAULT_STARTERS = [
    "How do I stop a CronJob from starting a new run while the previous one is still running?",
    "How do I configure a HorizontalPodAutoscaler based on CPU usage?",
    "How do I view the logs of a pod created by a Kubernetes Job?",
    "How do I create a Databricks job from a JSON file with the CLI?",
]


class RelatedDocument(TypedDict):
    source: str
    id: str          # best matching passage, e.g. "pods_autoscale.html#4"
    reason: str      # "similar content" | "read together in N conversations"
    snippet: str


def _passage(doc_id: str) -> tuple[str, int] | None:
    source, _, idx = doc_id.rpartition("#")
    return (source, int(idx)) if source and idx.isdigit() else None


def related_documents(sources: list[dict], cited: list[str], thread_id: str | None = None, limit: int = 3) -> list[RelatedDocument]:
    """Documents to read next. `sources` are the passages shown; `cited` the ids the answer used."""
    seeds = cited or [s["id"] for s in sources]
    if not seeds:
        return []
    shown_sources = sorted({s["source"] for s in sources})
    results: dict[str, RelatedDocument] = {}

    try:
        for source, n in feedback.co_cited_sources(seeds, shown_sources, thread_id):
            if n >= MIN_SUPPORT and len(results) < limit:
                results[source] = RelatedDocument(
                    source=source, id=source, reason=f"read together in {n} conversations", snippet=""
                )
    except Exception as e:
        logfire.warning(f"Collaborative recommendations unavailable: {e}")

    best: dict[str, dict] = {}
    try:
        for seed in seeds[:MAX_SEED_PASSAGES]:
            parsed = _passage(seed)
            if not parsed:
                continue
            for hit in similar_chunks(*parsed, exclude_sources=shown_sources, limit=8):
                if hit["source"] not in best or hit["score"] > best[hit["source"]]["score"]:
                    best[hit["source"]] = hit
    except Exception as e:
        logfire.warning(f"Content-based recommendations unavailable: {e}")

    for source, hit in sorted(best.items(), key=lambda kv: -kv[1]["score"]):
        if len(results) >= limit:
            break
        if source in results:
            results[source]["id"] = f"{source}#{hit['chunk_index']}"
            results[source]["snippet"] = hit["content"][:200]
            continue
        results[source] = RelatedDocument(
            source=source,
            id=f"{source}#{hit['chunk_index']}",
            reason="similar content",
            snippet=hit["content"][:200],
        )
    return list(results.values())[:limit]


def related_questions(cited: list[str], thread_id: str | None, question: str, limit: int = 3) -> list[str]:
    """"People who asked about this also asked" — collaborative, from other conversations."""
    if not cited:
        return []
    try:
        return [q for q, _ in feedback.co_asked_questions(cited, thread_id, question)[:limit]]
    except Exception as e:
        logfire.warning(f"Related questions unavailable: {e}")
        return []


def starter_questions(limit: int = 4) -> list[dict]:
    """Popular questions from other users, topped up with curated defaults (user cold start)."""
    try:
        popular = [{"question": q, "reason": f"asked in {n} conversations"} for q, n in feedback.popular_questions(limit)]
    except Exception as e:
        logfire.warning(f"Popular questions unavailable: {e}")
        popular = []
    seen = {p["question"].strip().lower() for p in popular}
    defaults = [{"question": q, "reason": "suggested"} for q in DEFAULT_STARTERS if q.strip().lower() not in seen]
    return (popular + defaults)[:limit]
