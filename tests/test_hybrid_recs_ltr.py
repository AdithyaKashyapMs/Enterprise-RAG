from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.services import feedback, ltr, personalise, recommend
from app.services.retrieval.keyword_index import BM25Index, rrf_fuse, tokenize
from app.tools import docs as docs_mod

CHUNKS = [
    {"source": "cron.docx", "chunk_index": 0, "content": "Set concurrencyPolicy: Forbid to skip overlapping runs."},
    {"source": "cron.docx", "chunk_index": 1, "content": "A CronJob creates Jobs on a repeating schedule."},
    {"source": "jobs.html", "chunk_index": 0, "content": "max_concurrent_runs limits parallel Databricks job runs."},
    {"source": "hpa.html", "chunk_index": 0, "content": "The autoscaler scales replicas between minReplicas and maxReplicas."},
]


@pytest.fixture(autouse=True)
def fresh_store():
    feedback.reset(":memory:")
    yield
    feedback.reset()


# --- #1 Hybrid search ------------------------------------------------------------------------

def test_tokenize_keeps_identifiers_whole():
    assert "max_concurrent_runs" in tokenize("What is max_concurrent_runs?")
    assert "next.js" in tokenize("Built with Next.js")
    assert "the" not in tokenize("the job")


def test_bm25_finds_exact_identifier():
    hits = BM25Index(CHUNKS).search("max_concurrent_runs", limit=2)
    assert hits[0]["source"] == "jobs.html" and len(hits) == 1


def test_bm25_rare_terms_outrank_common_ones():
    hits = BM25Index(CHUNKS).search("cronjob concurrencypolicy forbid")
    assert hits[0]["chunk_index"] == 0  # three matching terms beat one


def test_bm25_source_filter():
    hits = BM25Index(CHUNKS).search("job runs", source="cron.docx")
    assert {h["source"] for h in hits} == {"cron.docx"}


def test_rrf_rewards_agreement_and_records_ranks():
    a, b, c = ({"source": s, "chunk_index": 0, "content": s, "score": 1.0} for s in "abc")
    fused = rrf_fuse({"vector": [a, b], "bm25": [c, b]}, limit=3)
    assert fused[0]["source"] == "b"  # ranked 2nd by both beats 1st by only one
    assert fused[0]["vector_rank"] == 2 and fused[0]["bm25_rank"] == 2
    assert "bm25_rank" not in next(d for d in fused if d["source"] == "a")


def test_hybrid_candidates_add_keyword_hits(monkeypatch):
    vector_only = [{**CHUNKS[3], "score": 0.4}]
    monkeypatch.setattr(docs_mod, "search_enterprise_knowledge", lambda q, limit, source=None: vector_only)
    monkeypatch.setattr(docs_mod, "keyword_search", lambda q, limit, source=None: BM25Index(CHUNKS).search(q, limit, source))
    out = docs_mod.generate_candidates("max_concurrent_runs", hybrid=True)
    assert {d["source"] for d in out} == {"hpa.html", "jobs.html"}


def test_hybrid_falls_back_to_vector_when_keyword_index_fails(monkeypatch):
    monkeypatch.setattr(docs_mod, "search_enterprise_knowledge", lambda q, limit, source=None: [{**CHUNKS[0], "score": 0.9}])
    monkeypatch.setattr(docs_mod, "keyword_search", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no index")))
    out = docs_mod.generate_candidates("anything", hybrid=True)
    assert out[0]["vector_score"] == 0.9 and out[0]["vector_rank"] == 1


# --- #2 Recommendations and cold start --------------------------------------------------------

def shown(*ids):
    return [{"id": i, "source": i.split("#")[0], "score": 1.0} for i in ids]


def test_collaborative_needs_support_and_skips_own_thread(monkeypatch):
    monkeypatch.setattr(recommend, "similar_chunks", lambda *a, **k: [])
    for t in ("t1", "t2"):
        feedback.log_impressions(t, "q", shown("cron.docx#0", "jobs.html#0"), ["cron.docx#0", "jobs.html#0"])
    feedback.log_impressions("t3", "q", shown("cron.docx#0", "hpa.html#0"), ["cron.docx#0", "hpa.html#0"])

    recs = recommend.related_documents(shown("cron.docx#0"), ["cron.docx#0"], thread_id="me")
    assert [r["source"] for r in recs] == ["jobs.html"]  # hpa.html has support 1 < MIN_SUPPORT
    assert "2 conversations" in recs[0]["reason"]


def test_content_based_works_with_no_history(monkeypatch):
    calls = []

    def fake_similar(source, idx, exclude_sources, limit):
        calls.append((source, idx, exclude_sources))
        return [{"source": "hpa.html", "chunk_index": 4, "content": "autoscaling text", "score": 0.7}]

    monkeypatch.setattr(recommend, "similar_chunks", fake_similar)
    recs = recommend.related_documents(shown("cron.docx#0"), ["cron.docx#0"])
    assert recs == [{"source": "hpa.html", "id": "hpa.html#4", "reason": "similar content", "snippet": "autoscaling text"}]
    assert calls == [("cron.docx", 0, ["cron.docx"])]  # never recommends what was already shown


def test_related_documents_survive_qdrant_failure(monkeypatch):
    monkeypatch.setattr(recommend, "similar_chunks", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    assert recommend.related_documents(shown("cron.docx#0"), ["cron.docx#0"]) == []


def test_people_also_asked():
    feedback.log_impressions("t1", "How do I stop overlapping runs?", shown("cron.docx#0"), ["cron.docx#0"])
    feedback.log_impressions("t1", "What does Forbid do?", shown("cron.docx#0"), ["cron.docx#0"])
    feedback.log_impressions("t2", "Unrelated?", shown("hpa.html#0"), ["hpa.html#0"])
    qs = recommend.related_questions(["cron.docx#0"], "me", "How do I stop overlapping runs?")
    assert qs == ["What does Forbid do?"]


def test_starters_cold_start_then_popular():
    assert [s["reason"] for s in recommend.starter_questions(4)] == ["suggested"] * 4  # empty store
    for t in ("t1", "t2", "t3"):
        feedback.log_impressions(t, "What is a CronJob?", shown("cron.docx#1"), ["cron.docx#1"])
    starters = recommend.starter_questions(4)
    assert starters[0] == {"question": "What is a CronJob?", "reason": "asked in 3 conversations"}
    assert len(starters) == 4


def test_downvoted_questions_are_not_starters():
    feedback.log_impressions("t1", "Bad question?", shown("cron.docx#1"), ["cron.docx#1"])
    feedback.record("t1", "Bad question?", ["cron.docx#1"], -1)
    assert all(s["question"] != "Bad question?" for s in recommend.starter_questions(4))


def test_exploration_boost_fades_and_expires():
    now = datetime.now(timezone.utc)
    fresh = now.isoformat()
    assert personalise.exploration_boost(fresh, reach=0, now=now) == personalise.EXPLORE_BOOST
    assert 0 < personalise.exploration_boost(fresh, reach=5, now=now) < personalise.EXPLORE_BOOST
    assert personalise.exploration_boost(fresh, reach=10, now=now) == 0
    old = (now - timedelta(days=30)).isoformat()
    assert personalise.exploration_boost(old, reach=0, now=now) == 0
    assert personalise.exploration_boost(None, reach=0, now=now) == 0


# --- #3 Learning to rank --------------------------------------------------------------------

def test_ipw_weights_lower_positions_more():
    assert ltr.propensity(1) == 1.0 and ltr.propensity(4) == 0.25


def synthetic_rows(n=400):
    """Votes depend only on the rerank score; position is irrelevant to relevance."""
    import random

    rng = random.Random(0)
    rows = []
    for _ in range(n):
        rerank = rng.random()
        feats = {"rerank": rerank, "vector": rng.random(), "bm25": 0.0, "votes": 0.0}
        rows.append((feats, rng.randint(1, 5), int(rng.random() < rerank)))
    return rows


def test_trained_model_learns_the_signal():
    model = ltr.train(synthetic_rows())
    weights = dict(zip(ltr.FEATURES, model.weights))
    assert weights["rerank"] > 0.5 and abs(weights["vector"]) < weights["rerank"] / 3
    assert model.score({"rerank": 0.9, "vector": 0.5, "bm25": 0, "votes": 0}) > model.score(
        {"rerank": 0.1, "vector": 0.5, "bm25": 0, "votes": 0}
    )


def test_train_refuses_too_little_or_one_sided_data():
    with pytest.raises(ValueError):
        ltr.train(synthetic_rows(10))
    with pytest.raises(ValueError):
        ltr.train([(f, p, 1) for f, p, _ in synthetic_rows()])


def test_model_round_trip(tmp_path):
    model = ltr.train(synthetic_rows())
    path = tmp_path / "m.json"
    model.save(str(path))
    assert ltr.RankerModel.load_file(str(path)).score({"rerank": 0.5, "vector": 0, "bm25": 0, "votes": 0}) == pytest.approx(
        model.score({"rerank": 0.5, "vector": 0, "bm25": 0, "votes": 0})
    )


def test_learned_ranker_replaces_fixed_boost(monkeypatch):
    class PrefersBM25:
        def score(self, feats):
            return feats["bm25"]

    monkeypatch.setattr(docs_mod, "rerank_documents", lambda q, d, top_n=5: d[:top_n])
    candidates = [
        {"source": "a.docx", "chunk_index": 1, "content": "alpha", "score": 0.9},
        {"source": "b.docx", "chunk_index": 2, "content": "beta", "score": 0.5, "bm25_rank": 1},
    ]
    assert docs_mod.rank_candidates("q", candidates, 2)[0]["source"] == "a.docx"
    assert docs_mod.rank_candidates("q", candidates, 2, ranker=PrefersBM25())[0]["source"] == "b.docx"


def test_training_rows_join_impressions_and_votes():
    sources = [{"id": "a#1", "source": "a", "features": {"rerank": 0.9, "vector": 0.5, "bm25": 1.0, "votes": 0.0}},
               {"id": "b#1", "source": "b", "features": {"rerank": 0.2, "vector": 0.1, "bm25": 0.0, "votes": 0.0}}]
    feedback.log_impressions("t1", "q", sources, ["a#1"])
    feedback.record("t1", "q", ["b#1"], -1)
    assert feedback.training_rows() == [(sources[1]["features"], 2, 0)]  # only voted rows, with position


# --- API -------------------------------------------------------------------------------------

def test_starters_endpoint():
    from app import main

    r = TestClient(main.app).get("/starters?limit=2")
    assert r.status_code == 200 and len(r.json()) == 2 and r.json()[0]["reason"] == "suggested"


def test_implicit_ipw_upweights_clicks_not_skips():
    # A click at position 5 should pull the model 5x harder than one at position 1;
    # skips get no extra weight.
    rows = [({"rerank": 1.0, "vector": 0, "bm25": 0, "votes": 0}, 5, 1)] * 20 + \
           [({"rerank": 0.0, "vector": 0, "bm25": 0, "votes": 0}, 1, 0)] * 20
    naive = ltr.train(rows, ipw=False, implicit=True)
    ipw = ltr.train(rows, ipw=True, implicit=True)
    hi = {"rerank": 1.0, "vector": 0, "bm25": 0, "votes": 0}
    assert ipw.score(hi) > naive.score(hi)
