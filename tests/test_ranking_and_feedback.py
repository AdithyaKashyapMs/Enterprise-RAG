import pytest
from fastapi.testclient import TestClient

from app.services import feedback
from app.services.retrieval.diversity import jaccard, mmr
from app.tools import docs as docs_mod


def doc(source, idx, content, score):
    return {"id": f"{source}#{idx}", "source": source, "chunk_index": idx, "content": content, "score": score}


@pytest.fixture(autouse=True)
def fresh_feedback_db():
    feedback.reset(":memory:")
    yield
    feedback.reset()


# --- MMR diversity -------------------------------------------------------------------------

def test_mmr_first_pick_is_most_relevant():
    docs = [doc("a.docx", 1, "alpha beta", 0.2), doc("b.docx", 1, "gamma delta", 0.9)]
    assert mmr(docs, top_k=2)[0]["id"] == "b.docx#1"


def test_mmr_caps_chunks_per_file():
    docs = [doc("cron.docx", i, f"cronjob text part {i}", 0.9 - i * 0.01) for i in range(5)]
    docs.append(doc("hpa.html", 0, "autoscaler replicas", 0.1))
    picked = mmr(docs, top_k=3, max_per_source=2)
    assert [d["source"] for d in picked].count("cron.docx") == 2
    assert "hpa.html" in {d["source"] for d in picked}


def test_mmr_skips_near_duplicates():
    text = "set concurrencyPolicy Forbid to skip overlapping cronjob runs"
    docs = [
        doc("a.docx", 1, text, 1.0),
        doc("a.docx", 2, text + " again", 0.95),  # overlapping chunk, almost identical
        doc("b.docx", 7, "kubectl logs shows the output of a job pod", 0.9),
    ]
    picked = mmr(docs, top_k=2, max_per_source=None)
    assert [d["id"] for d in picked] == ["a.docx#1", "b.docx#7"]


def test_mmr_tops_up_when_cap_leaves_too_few():
    docs = [doc("only.docx", i, f"chunk {i}", 1 - i * 0.1) for i in range(4)]
    assert len(mmr(docs, top_k=4, max_per_source=2)) == 4


def test_jaccard():
    assert jaccard(frozenset("ab"), frozenset("ab")) == 1.0
    assert jaccard(frozenset("ab"), frozenset("cd")) == 0.0


# --- Feedback ------------------------------------------------------------------------------

def test_votes_are_one_per_thread_and_changeable():
    feedback.record("t1", "q", ["a#1", "a#1"], 1)  # duplicate id in one request counts once
    feedback.record("t1", "q", ["a#1"], 1)         # same thread voting again doesn't add up
    feedback.record("t2", "q", ["a#1"], 1)
    assert feedback.net_votes(["a#1"]) == {"a#1": 2}
    feedback.record("t2", "q", ["a#1"], -1)        # changing a vote replaces it
    assert feedback.net_votes(["a#1"]) == {"a#1": 0}


def test_boost_is_capped():
    assert feedback.boost(1000) == feedback.MAX_BOOST
    assert feedback.boost(-1000) == -feedback.MAX_BOOST
    assert 0 < feedback.boost(1) < feedback.MAX_BOOST


def test_invalid_vote_rejected():
    with pytest.raises(ValueError):
        feedback.record("t", "q", ["a#1"], 0)


def test_feedback_reorders_close_results(monkeypatch):
    candidates = [
        {"source": "a.docx", "chunk_index": 1, "content": "alpha text", "score": 0.80},
        {"source": "b.docx", "chunk_index": 2, "content": "beta words", "score": 0.78},
    ]
    monkeypatch.setattr(docs_mod, "rerank_documents", lambda q, d, top_n=5: d[:top_n])
    assert docs_mod.rank_candidates("q", candidates, 2)[0]["id"] == "a.docx#1"

    for thread in ("t1", "t2", "t3"):
        feedback.record(thread, "q", ["b.docx#2"], 1)
    assert docs_mod.rank_candidates("q", candidates, 2)[0]["id"] == "b.docx#2"
    # ...but votes are switched off for evaluation.
    assert docs_mod.rank_candidates("q", candidates, 2, use_feedback=False)[0]["id"] == "a.docx#1"


def test_votes_cannot_override_a_much_more_relevant_result(monkeypatch):
    candidates = [
        {"source": "a.docx", "chunk_index": 1, "content": "alpha text", "score": 0.95},
        {"source": "b.docx", "chunk_index": 2, "content": "beta words", "score": 0.30},
    ]
    monkeypatch.setattr(docs_mod, "rerank_documents", lambda q, d, top_n=5: d[:top_n])
    for thread in range(50):
        feedback.record(f"t{thread}", "q", ["b.docx#2"], 1)
    assert docs_mod.rank_candidates("q", candidates, 2)[0]["id"] == "a.docx#1"


def test_search_survives_feedback_store_failure(monkeypatch):
    from app.services import personalise

    monkeypatch.setattr(feedback, "net_votes", lambda ids: (_ for _ in ()).throw(RuntimeError("db gone")))
    out = personalise.apply([doc("a.docx", 1, "x", 0.5)])
    assert out[0]["score"] == 0.5 and out[0]["id"] == "a.docx#1"


# --- API -----------------------------------------------------------------------------------

def test_feedback_endpoint():
    from app import main

    client = TestClient(main.app)
    ok = client.post("/feedback", json={"thread_id": "t1", "question": "q", "chunk_ids": ["a#1", "b#2"], "vote": 1})
    assert ok.status_code == 200 and ok.json() == {"stored": 2}
    assert feedback.net_votes(["a#1"]) == {"a#1": 1}

    assert client.post("/feedback", json={"thread_id": "t1", "question": "q", "chunk_ids": ["a#1"], "vote": 0}).status_code == 422
    assert client.post("/feedback", json={"thread_id": "t1", "question": "q", "chunk_ids": [], "vote": 1}).status_code == 422
