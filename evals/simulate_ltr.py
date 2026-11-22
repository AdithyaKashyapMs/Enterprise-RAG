"""
Does the learned ranker (app/services/ltr.py) work, and does position-bias correction matter?

There are no real users yet, so feedback is *simulated* on the real index and golden set:

  1. For each golden question, take the 15 hybrid candidates (vector + BM25) and their real
     ranking features (cross-encoder, vector, BM25). A passage is "relevant" if it comes from
     the question's expected file.
  2. Simulated users see the production top 5 (cross-encoder order; in 25% of sessions two
     neighbouring results are swapped, a light randomisation). They examine position k with
     probability 1/k, and vote 👍 on a relevant passage with probability 0.9 (0.1 if irrelevant).
  3. Two kinds of feedback data:
       explicit  only passages the user voted on, 👍 = 1 / 👎 = 0
       implicit  every shown passage, 👍 = 1, anything else = 0   (like click logs)
  4. Train the ranker naively and with IPW, 3-fold cross-validated over questions, and score
     each by ranking all 15 candidates of the held-out questions.

The simulated users follow exactly the click model IPW assumes (examination = 1/k), so this
shows the mechanism works, not how large the gain would be with real people.

Usage: python -m evals.simulate_ltr
"""

import json
import os
import random
from datetime import date

HERE = os.path.dirname(__file__)
SESSIONS_PER_QUESTION = 200
FOLDS = 3
TOP_SHOWN = 5


def question_data():
    from app.services import ltr
    from app.services.retrieval.keyword_index import keyword_search, rrf_fuse
    from app.services.retrieval.qdrant_service import search_enterprise_knowledge
    from app.services.retrieval.ranking_service import rerank_documents

    data = []
    for case in json.load(open(os.path.join(HERE, "golden_set.json"))):
        q = case["question"]
        cands = rrf_fuse({"vector": search_enterprise_knowledge(q, limit=15), "bm25": keyword_search(q, limit=15)}, limit=15)
        ranked = rerank_documents(q, cands, top_n=len(cands))  # production order: cross-encoder
        items = [
            {"features": ltr.features({**d, "rerank_score": d["score"]}), "relevant": d["source"] in case["expected_sources"]}
            for d in ranked
        ]
        data.append({"question": q, "items": items})
    return data


def simulate(questions, kind: str, rng: random.Random):
    from app.services.ltr import propensity

    rows = []
    for qd in questions:
        for _ in range(SESSIONS_PER_QUESTION):
            shown = list(range(min(TOP_SHOWN, len(qd["items"]))))
            if rng.random() < 0.25 and len(shown) > 1:
                i = rng.randrange(len(shown) - 1)
                shown[i], shown[i + 1] = shown[i + 1], shown[i]
            for pos, idx in enumerate(shown, 1):
                item = qd["items"][idx]
                examined = rng.random() < propensity(pos)
                liked = examined and rng.random() < (0.9 if item["relevant"] else 0.1)
                if kind == "explicit":
                    if examined:
                        rows.append((item["features"], pos, int(liked)))
                else:
                    rows.append((item["features"], pos, int(liked)))
    return rows


def evaluate(questions, score_fn):
    hit1, rr = [], []
    for qd in questions:
        order = sorted(range(len(qd["items"])), key=lambda i: -score_fn(qd["items"][i], i))
        rank = next((r for r, i in enumerate(order, 1) if qd["items"][i]["relevant"]), None)
        hit1.append(float(rank == 1))
        rr.append(1 / rank if rank else 0.0)
    return hit1, rr


def main():
    from app.services import ltr

    rng = random.Random(7)
    data = question_data()
    order = list(range(len(data)))
    rng.shuffle(order)
    folds = [order[f::FOLDS] for f in range(FOLDS)]

    results = {}
    for kind in ("explicit", "implicit"):
        scores = {"cross-encoder only": ([], []), "learned (naive)": ([], []), "learned (IPW)": ([], [])}
        weights = {"learned (naive)": [], "learned (IPW)": []}
        for f in range(FOLDS):
            test = [data[i] for i in folds[f]]
            train_q = [data[i] for g in range(FOLDS) if g != f for i in folds[g]]
            rows = simulate(train_q, kind, rng)
            implicit = kind == "implicit"
            models = {
                "learned (naive)": ltr.train(rows, ipw=False, implicit=implicit),
                "learned (IPW)": ltr.train(rows, ipw=True, implicit=implicit),
            }

            for name, (h, r) in scores.items():
                if name == "cross-encoder only":
                    fn = lambda item, i: -i  # noqa: E731  (production order)
                else:
                    model = models[name]
                    fn = lambda item, i, m=model: m.score(item["features"])  # noqa: E731
                    weights[name].append(dict(zip(ltr.FEATURES, m.weights)) if (m := model) else None)
                hh, rrr = evaluate(test, fn)
                h.extend(hh)
                r.extend(rrr)

        results[kind] = {
            name: {"hit@1": round(sum(h) / len(h), 3), "mrr": round(sum(r) / len(r), 3), "n": len(h)}
            for name, (h, r) in scores.items()
        }
        results[kind]["mean_weights"] = {
            name: {feat: round(sum(w[feat] for w in ws) / len(ws), 3) for feat in ltr.FEATURES}
            for name, ws in weights.items()
        }

    for kind, res in results.items():
        print(f"\n{kind} feedback")
        print(f"  {'ranker':<20} {'hit@1':>6} {'mrr':>6}")
        for name, m in res.items():
            if name != "mean_weights":
                print(f"  {name:<20} {m['hit@1']:>6.3f} {m['mrr']:>6.3f}")
        print("  weights:", res["mean_weights"])

    out = os.path.join(HERE, "results", "ltr_simulation.json")
    with open(out, "w") as f:
        json.dump({"date": date.today().isoformat(), "sessions_per_question": SESSIONS_PER_QUESTION,
                   "folds": FOLDS, "results": results}, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
