"""
Learning to rank from 👍/👎 feedback: a small logistic regression, trained with
inverse propensity weighting (IPW) to correct for position bias.

Features (per shown passage; all computed at ranking time, so no position):
  rerank  cross-encoder score                          (query-passage relevance)
  vector  cosine similarity from vector search, 0 if the passage came only from BM25
  bm25    1 / BM25 rank, 0 if BM25 didn't retrieve it  (exact keyword match strength)
  votes   net 👍 minus 👎 so far, scaled to [-1, 1]

Position bias: people look at (and vote on) the top results far more than lower ones, so a
passage at position 1 collects votes partly *because* it was at position 1. Trained naively,
the model learns "whatever the current ranker likes is good" and just copies the old order.

IPW fix (position-based click model): P(vote) = P(examined | position) · P(relevant), with
examination falling off as  p(k) = (1/k)^η. How it is applied depends on the data:

  explicit votes (👍/👎; we know the user looked):  the labels are fine, but the *sample* is
      skewed toward top positions. Weight each example by 1/p(k), so a vote at position 4
      counts 4× one at position 1 (η = 1): the training set then looks like every position
      had been examined equally.
  implicit clicks (no click ≠ "bad", maybe just never seen):  unbiased pointwise estimator
      (Saito et al., 2020): train toward the target  click / p(k)  instead of the raw click.
      A click at position 4 counts 4×; a non-click is still 0 but no longer up-weighted.
      (Weighting non-clicks by 1/p(k) too, the naive "IPW", makes it worse: those are
      exactly the rows where the user probably never looked.)

Both assume η is known; production systems estimate it by occasionally swapping results
(RandPair) and comparing click rates.

Position is used only for weighting, never as a feature: at ranking time there is no
position yet.

Train:  python -m app.services.ltr train            (reads the feedback store)
"""

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np

from app.config import settings

FEATURES = ["rerank", "vector", "bm25", "votes"]
MIN_EXAMPLES = 30


def features(doc: dict, net_votes: int = 0) -> dict[str, float]:
    bm25_rank = doc.get("bm25_rank")
    return {
        "rerank": float(doc.get("rerank_score", doc.get("score", 0.0))),
        "vector": float(doc.get("vector_score") or 0.0),
        "bm25": 1.0 / bm25_rank if bm25_rank else 0.0,
        "votes": max(-1.0, min(1.0, net_votes / 5)),
    }


def propensity(position: int, eta: float = 1.0) -> float:
    """Estimated probability that a user examines the result at this position."""
    return (1.0 / max(position, 1)) ** eta


@dataclass
class RankerModel:
    weights: list[float]
    bias: float
    mean: list[float]
    std: list[float]
    meta: dict = field(default_factory=dict)

    def score(self, feats: dict[str, float]) -> float:
        """Predicted probability that a user would 👍 this passage."""
        x = (np.array([feats[f] for f in FEATURES]) - self.mean) / self.std
        z = float(np.dot(self.weights, x) + self.bias)
        return 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load_file(cls, path: str) -> "RankerModel":
        with open(path) as f:
            return cls(**json.load(f))


def train(
    rows: list[tuple[dict, int, int]],
    ipw: bool = True,
    implicit: bool = False,
    eta: float = 1.0,
    l2: float = 0.01,
    epochs: int = 2000,
    lr: float = 0.5,
    max_weight: float = 10.0,
) -> RankerModel:
    """
    Logistic regression by full-batch gradient descent.
    rows: (features, position, label). `implicit=True` for click-style data (see module doc).
    Inverse propensities are clipped at `max_weight`: unclipped, a handful of votes at low
    positions would dominate (high variance).
    """
    labels = [y for _, _, y in rows]
    if len(rows) < MIN_EXAMPLES or len(set(labels)) < 2:
        raise ValueError(f"Need at least {MIN_EXAMPLES} voted impressions with both 👍 and 👎 (have {len(rows)}).")

    X = np.array([[f[name] for name in FEATURES] for f, _, _ in rows], dtype=float)
    y = np.array(labels, dtype=float)
    inv_p = np.array([min(1 / propensity(p, eta), max_weight) for _, p, _ in rows])
    if ipw and implicit:
        target, w_ex = y * inv_p, np.ones(len(y))     # unbiased target, unit weights
    elif ipw:
        target, w_ex = y, inv_p / inv_p.mean()         # reweight the sample
    else:
        target, w_ex = y, np.ones(len(y))

    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std < 1e-9] = 1.0
    Xs = (X - mean) / std

    w, b = np.zeros(len(FEATURES)), 0.0
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-(Xs @ w + b)))
        err = w_ex * (p - target)
        w -= lr * (Xs.T @ err / len(y) + l2 * w)
        b -= lr * err.mean()

    return RankerModel(
        weights=w.round(6).tolist(),
        bias=round(float(b), 6),
        mean=mean.round(6).tolist(),
        std=std.round(6).tolist(),
        meta={
            "features": FEATURES,
            "examples": len(rows),
            "positives": int(y.sum()),
            "ipw": ipw,
            "implicit": implicit,
            "eta": eta,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )


_cached: tuple[float, RankerModel] | None = None


def load() -> RankerModel | None:
    """The saved model, reloaded when the file changes; None if no model has been trained."""
    global _cached
    path = settings.LTR_MODEL_PATH
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    if _cached is None or _cached[0] != mtime:
        _cached = (mtime, RankerModel.load_file(path))
    return _cached[1]


if __name__ == "__main__":
    from app.services import feedback

    parser = argparse.ArgumentParser(description="Train the feedback ranker.")
    parser.add_argument("command", choices=["train"])
    parser.add_argument("--no-ipw", action="store_true", help="ignore position bias (for comparison)")
    parser.add_argument("--eta", type=float, default=1.0, help="position-bias strength")
    parser.add_argument("--out", default=settings.LTR_MODEL_PATH)
    args = parser.parse_args()

    model = train(feedback.training_rows(), ipw=not args.no_ipw, eta=args.eta)
    model.save(args.out)
    print(f"Saved {args.out}: {dict(zip(FEATURES, model.weights))}  ({model.meta['examples']} examples)")
