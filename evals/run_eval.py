"""
Evaluate retrieval (and optionally the agent) against evals/golden_set.json.

Every question is scored twice from the same 15 vector-search candidates:
  - vector   : Qdrant cosine order, top 5            (no reranker)
  - reranked : FlashRank cross-encoder order, top 5  (what search_docs returns)
so the reranker's contribution is measured, not assumed.

Retrieval metrics, per mode and per difficulty ("direct" / "paraphrased"):
  - hit@1    : the top passage comes from an expected source file
  - hit@5    : an expected source appears in the top 5
  - mrr      : mean reciprocal rank of the first expected source
  - noise@5  : share of top-5 passages from files outside DATA/true_data (distractors)

Agent metrics (--answers; calls the LLM):
  - mention_rate  : answer contains every `must_mention` term
  - citation_rate : answer cites a passage id like [cronjobs.docx#12]
  - avg_tool_calls

Usage:
    python -m evals.run_eval --label distractors          # writes evals/results/distractors.json
    python -m evals.run_eval --answers --min-hit 0.9
"""

import argparse
import json
import os
import re
import sys
from datetime import date

HERE = os.path.dirname(__file__)
GOLDEN = os.path.join(HERE, "golden_set.json")
TOP_K = 5
MODES = ("vector", "reranked")


def _score(sources: list[str], expected: list[str], true_sources: set[str]) -> dict:
    rank = next((i for i, s in enumerate(sources, 1) if s in expected), None)
    return {
        "rank": rank,
        "hit@1": float(rank == 1),
        f"hit@{TOP_K}": float(rank is not None and rank <= TOP_K),
        "mrr": 1 / rank if rank else 0.0,
        f"noise@{TOP_K}": sum(s not in true_sources for s in sources) / max(len(sources), 1),
    }


def _aggregate(rows: list[dict]) -> dict:
    keys = ["hit@1", f"hit@{TOP_K}", "mrr", f"noise@{TOP_K}"]
    return {k: round(sum(r[k] for r in rows) / len(rows), 3) for k in keys} | {"n": len(rows)}


def evaluate(with_answers: bool) -> dict:
    from app.services.retrieval.qdrant_service import list_sources, search_enterprise_knowledge
    from app.services.retrieval.ranking_service import rerank_documents
    from app.tools.docs import SEARCH_CANDIDATES

    true_sources = set(os.listdir(os.path.join("DATA", "true_data")))
    indexed = list_sources()
    corpus = {
        "documents": len(indexed),
        "chunks": sum(s["chunks"] for s in indexed),
        "distractor_chunks": sum(s["chunks"] for s in indexed if s["source"] not in true_sources),
    }
    print(f"Index: {corpus['documents']} documents, {corpus['chunks']} chunks "
          f"({corpus['distractor_chunks']} from distractors)\n")

    cases = json.load(open(GOLDEN))
    per_case = []
    agent_rows = []

    for case in cases:
        q, expected = case["question"], case["expected_sources"]
        candidates = search_enterprise_knowledge(q, limit=SEARCH_CANDIDATES)
        ranked = {
            "vector": [d["source"] for d in candidates[:TOP_K]],
            "reranked": [d["source"] for d in rerank_documents(q, candidates, top_n=TOP_K)],
        }
        scores = {m: _score(ranked[m], expected, true_sources) for m in MODES}
        per_case.append({"difficulty": case["difficulty"], "question": q, "top_sources": ranked, **{m: scores[m] for m in MODES}})

        v, r = scores["vector"]["rank"], scores["reranked"]["rank"]
        mark = "✓" if r == 1 else ("~" if r else "✗")
        line = f"{mark} [{case['difficulty'][:4]}] vec={v or '-':<2} rr={r or '-':<2} {q[:72]:<72} -> {ranked['reranked'][:2]}"

        if with_answers:
            from app.agents import service

            result = service.ask(q)
            answer = result["answer"]
            row = {
                "mention": all(t.lower() in answer.lower() for t in case["must_mention"]),
                "cited": bool(re.search(r"\[[^\]]+#\d+\]", answer)),
                "tool_calls": sum(s.startswith("Tool:") for s in result["thought_process"]),
            }
            agent_rows.append(row)
            line += f"  mention={'✓' if row['mention'] else '✗'} cited={'✓' if row['cited'] else '✗'} tools={row['tool_calls']}"
        print(line)

    report = {"date": date.today().isoformat(), "corpus": corpus, "top_k": TOP_K, "metrics": {}}
    for mode in MODES:
        report["metrics"][mode] = {"all": _aggregate([c[mode] for c in per_case])}
        for diff in sorted({c["difficulty"] for c in per_case}):
            report["metrics"][mode][diff] = _aggregate([c[mode] for c in per_case if c["difficulty"] == diff])
    if agent_rows:
        n = len(agent_rows)
        report["agent"] = {
            "mention_rate": round(sum(r["mention"] for r in agent_rows) / n, 3),
            "citation_rate": round(sum(r["cited"] for r in agent_rows) / n, 3),
            "avg_tool_calls": round(sum(r["tool_calls"] for r in agent_rows) / n, 2),
        }
    report["cases"] = per_case
    return report


def print_summary(report: dict) -> None:
    print(f"\n{'mode':<10} {'subset':<13} {'n':>3} {'hit@1':>7} {'hit@5':>7} {'mrr':>7} {'noise@5':>8}")
    for mode, subsets in report["metrics"].items():
        for subset, m in subsets.items():
            print(f"{mode:<10} {subset:<13} {m['n']:>3} {m['hit@1']:>7.3f} {m[f'hit@{TOP_K}']:>7.3f} "
                  f"{m['mrr']:>7.3f} {m[f'noise@{TOP_K}']:>8.3f}")
    if "agent" in report:
        print("\nagent:", report["agent"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", action="store_true", help="also run the agent and score answers")
    parser.add_argument("--label", help="save the report to evals/results/<label>.json")
    parser.add_argument("--min-hit", type=float, default=0.0, help="exit non-zero if reranked hit@5 is below this")
    args = parser.parse_args()

    os.environ.setdefault("CHECKPOINT_DB", ":memory:")
    report = evaluate(args.answers)
    print_summary(report)

    if args.label:
        out_dir = os.path.join(HERE, "results")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{args.label}.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved {path}")

    sys.exit(1 if report["metrics"]["reranked"]["all"][f"hit@{TOP_K}"] < args.min_hit else 0)
