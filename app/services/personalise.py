"""
Stage 3 of ranking (before MMR diversity): adjust relevance using what users did.

  score = learned ranker's P(👍)            if a model has been trained (app/services/ltr.py)
        = rerank score + capped vote boost  otherwise
        + exploration boost for new documents (item cold start)

Item cold start: a document ingested recently has no votes and has rarely been shown, so
anything driven by feedback ranks it low and it never gets the chance to collect feedback.
A small boost that fades as the document reaches more conversations gives it that chance
(a simple form of explore/exploit).

Every doc leaves with its ranking `features`, which are logged with the impression and become
training data for the learned ranker.
"""

from datetime import datetime, timedelta, timezone

import logfire

from app.services import feedback, ltr

NEW_ITEM_DAYS = 14         # a document counts as new for this long after ingestion
EXPLORE_BOOST = 0.05       # max boost for a brand-new document
EXPLORE_UNTIL_REACH = 10   # ... fading to zero once it has been shown in this many conversations


def exploration_boost(ingested_at: str | None, reach: int, now: datetime | None = None) -> float:
    if not ingested_at or reach >= EXPLORE_UNTIL_REACH:
        return 0.0
    try:
        when = datetime.fromisoformat(ingested_at)
    except ValueError:
        return 0.0
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    if (now or datetime.now(timezone.utc)) - when > timedelta(days=NEW_ITEM_DAYS):
        return 0.0
    return EXPLORE_BOOST * (1 - reach / EXPLORE_UNTIL_REACH)


def describe(docs: list[dict]) -> list[dict]:
    """Attach features without changing scores (used when personalisation is off, e.g. evals)."""
    return [{**d, "features": ltr.features(d)} for d in docs]


def apply(docs: list[dict], ranker: "ltr.RankerModel | None" = None) -> list[dict]:
    """Re-score docs (they need "id", "source", "rerank_score"). Never fails search."""
    try:
        votes = feedback.net_votes([d["id"] for d in docs])
        reach = feedback.source_reach(list({d["source"] for d in docs}))
    except Exception as e:
        logfire.warning(f"Interaction store unavailable, ranking without it: {e}")
        votes, reach = {}, {}

    model = ranker or ltr.load()
    out = []
    for d in docs:
        net = votes.get(d["id"], 0)
        feats = ltr.features(d, net)
        score = model.score(feats) if model else feats["rerank"] + feedback.boost(net)
        score += exploration_boost(d.get("ingested_at"), reach.get(d["source"], 0))
        out.append({**d, "score": score, "features": feats})
    return out
