"""
Result diversification with Maximal Marginal Relevance (MMR).

After reranking, the top passages are often near-duplicates (overlapping chunks of
the same file). MMR picks results one at a time, each time taking the candidate with
the best trade-off between relevance and novelty:

    mmr(d) = λ · relevance(d) − (1 − λ) · max_{s ∈ selected} similarity(d, s)

The first pick is always the most relevant passage; later picks are penalised for
repeating what is already selected. A per-file cap adds a hard business rule on top.

Similarity is token-set Jaccard on the passage text: cheap, dependency-free and good
at catching the overlapping-chunk duplicates that chunking with overlap produces.
"""

import re

DEFAULT_LAMBDA = 0.7       # 1.0 = pure relevance, 0.0 = pure novelty
DEFAULT_MAX_PER_SOURCE = 2

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall(text.lower()))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _normalise(scores: list[float]) -> list[float]:
    """
    Scale relevance to [0, 1] (like similarity) by dividing by the top score. This keeps
    proportions: 0.90 vs 0.95 stays a small gap. Min-max scaling would instead stretch
    the weakest candidate down to 0, so MMR would favour a near-duplicate over a
    different but almost-as-relevant passage. Min-max is only used if no score is positive.
    """
    hi = max(scores)
    if hi > 0:
        return [max(s, 0.0) / hi for s in scores]
    lo = min(scores)
    if hi - lo < 1e-9:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def mmr(
    docs: list[dict],
    top_k: int,
    lam: float = DEFAULT_LAMBDA,
    max_per_source: int | None = DEFAULT_MAX_PER_SOURCE,
) -> list[dict]:
    """
    Select top_k docs from `docs` (dicts with "content", "source", "score"), most relevant
    first, trading relevance against redundancy. `max_per_source=None` disables the cap.
    If the cap leaves fewer than top_k results, the remaining slots are filled by relevance.
    """
    if not docs:
        return []

    relevance = _normalise([d["score"] for d in docs])
    tokens = [_tokens(d["content"]) for d in docs]
    remaining = list(range(len(docs)))
    selected: list[int] = []
    per_source: dict[str, int] = {}

    while remaining and len(selected) < top_k:
        allowed = [
            i for i in remaining
            if max_per_source is None or per_source.get(docs[i]["source"], 0) < max_per_source
        ]
        if not allowed:
            break

        def score(i: int) -> float:
            redundancy = max((jaccard(tokens[i], tokens[j]) for j in selected), default=0.0)
            return lam * relevance[i] - (1 - lam) * redundancy

        best = max(allowed, key=score)
        selected.append(best)
        remaining.remove(best)
        per_source[docs[best]["source"]] = per_source.get(docs[best]["source"], 0) + 1

    # The cap must never make results disappear: top up by relevance if it ran out.
    for i in sorted(remaining, key=lambda i: -relevance[i]):
        if len(selected) >= top_k:
            break
        selected.append(i)

    return [docs[i] for i in selected]
