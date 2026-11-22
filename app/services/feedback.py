"""
Interaction store (SQLite): what users were shown, and how they voted.

Two tables:
  impressions  every passage shown with an answer: its position (1 = top), whether the
               answer cited it, and its ranking features at the time. Implicit signal.
  feedback     👍/👎 per passage. One vote per conversation per passage; voting again
               replaces the vote, so one user can't inflate a passage by clicking repeatedly.

Everything built on top reads from here:
  - fixed vote boost / learned ranker training data   (app/services/personalise.py, ltr.py)
  - collaborative filtering and cold-start starters   (app/services/recommend.py)
"""

import json
import os
import sqlite3
import uuid
from threading import Lock

import logfire

MAX_BOOST = 0.1            # on the 0-1 relevance scale used for ranking
VOTES_FOR_MAX_BOOST = 5    # net votes needed to reach the full boost

_conn: sqlite3.Connection | None = None
_lock = Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    thread_id  TEXT    NOT NULL,
    chunk_id   TEXT    NOT NULL,
    question   TEXT    NOT NULL,
    vote       INTEGER NOT NULL CHECK (vote IN (-1, 1)),
    created_at TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (thread_id, chunk_id)
);
CREATE INDEX IF NOT EXISTS feedback_chunk ON feedback (chunk_id);

CREATE TABLE IF NOT EXISTS impressions (
    turn_id    TEXT    NOT NULL,
    thread_id  TEXT    NOT NULL,
    question   TEXT    NOT NULL,
    chunk_id   TEXT    NOT NULL,
    source     TEXT    NOT NULL,
    position   INTEGER NOT NULL,
    cited      INTEGER NOT NULL,
    features   TEXT,
    created_at TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (turn_id, chunk_id)
);
CREATE INDEX IF NOT EXISTS impressions_chunk ON impressions (chunk_id);
CREATE INDEX IF NOT EXISTS impressions_thread ON impressions (thread_id);
CREATE INDEX IF NOT EXISTS impressions_source ON impressions (source);
"""


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        path = os.getenv("FEEDBACK_DB", "feedback.sqlite")
        _conn = sqlite3.connect(path, check_same_thread=False)
        _conn.executescript(_SCHEMA)
    return _conn


def _query(sql: str, params: list | tuple = ()) -> list[tuple]:
    with _lock:
        return _db().execute(sql, params).fetchall()


def _marks(n: int) -> str:
    return ",".join("?" * n)


def reset(path: str | None = None) -> None:
    """Close the connection (tests use this to start from an empty in-memory DB)."""
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
        _conn = None
        if path is not None:
            os.environ["FEEDBACK_DB"] = path


# --- Votes ---------------------------------------------------------------------------------

def record(thread_id: str, question: str, chunk_ids: list[str], vote: int) -> int:
    """Store a vote (+1 / -1) for each passage behind an answer. Returns how many were stored."""
    if vote not in (-1, 1):
        raise ValueError("vote must be 1 or -1")
    unique_ids = list(dict.fromkeys(chunk_ids))
    with _lock:
        db = _db()
        db.executemany(
            """
            INSERT INTO feedback (thread_id, chunk_id, question, vote) VALUES (?, ?, ?, ?)
            ON CONFLICT (thread_id, chunk_id)
            DO UPDATE SET vote = excluded.vote, question = excluded.question, created_at = CURRENT_TIMESTAMP
            """,
            [(thread_id, cid, question, vote) for cid in unique_ids],
        )
        db.commit()
    logfire.info(f"Feedback {vote:+d} on {len(unique_ids)} passage(s)", thread_id=thread_id)
    return len(unique_ids)


def net_votes(chunk_ids: list[str]) -> dict[str, int]:
    """Net votes (ups minus downs, across all conversations) for the given passages."""
    if not chunk_ids:
        return {}
    rows = _query(
        f"SELECT chunk_id, SUM(vote) FROM feedback WHERE chunk_id IN ({_marks(len(chunk_ids))}) GROUP BY chunk_id",
        list(chunk_ids),
    )
    return {cid: int(total) for cid, total in rows}


def boost(net: int) -> float:
    """
    Fixed, capped boost used until a learned ranker exists. The cap prevents a popularity
    feedback loop (boosted passages get shown more, so collect more votes, so get boosted more)
    and stops votes from overriding a clearly more relevant passage.
    """
    return MAX_BOOST * max(-1.0, min(1.0, net / VOTES_FOR_MAX_BOOST))


# --- Impressions ---------------------------------------------------------------------------

def log_impressions(thread_id: str, question: str, sources: list[dict], cited: list[str]) -> str:
    """Record the passages shown with one answer, in display order. Returns the turn id."""
    turn_id = str(uuid.uuid4())
    cited_set = set(cited)
    rows = [
        (turn_id, thread_id, question, s["id"], s["source"], position, int(s["id"] in cited_set),
         json.dumps(s["features"]) if s.get("features") else None)
        for position, s in enumerate(sources, 1)
    ]
    with _lock:
        db = _db()
        db.executemany(
            "INSERT OR IGNORE INTO impressions "
            "(turn_id, thread_id, question, chunk_id, source, position, cited, features) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        db.commit()
    return turn_id


def source_reach(sources: list[str]) -> dict[str, int]:
    """In how many conversations each document has been shown (used for the new-item boost)."""
    if not sources:
        return {}
    rows = _query(
        f"SELECT source, COUNT(DISTINCT thread_id) FROM impressions WHERE source IN ({_marks(len(sources))}) "
        "GROUP BY source",
        list(sources),
    )
    return dict(rows)


def _threads_citing(chunk_ids: list[str], exclude_thread: str | None) -> list[str]:
    if not chunk_ids:
        return []
    rows = _query(
        f"SELECT DISTINCT thread_id FROM impressions WHERE cited = 1 AND chunk_id IN ({_marks(len(chunk_ids))}) "
        "AND thread_id != ?",
        [*chunk_ids, exclude_thread or ""],
    )
    return [r[0] for r in rows]


def co_cited_sources(chunk_ids: list[str], exclude_sources: list[str], exclude_thread: str | None) -> list[tuple[str, int]]:
    """
    Item-to-item collaborative filtering: documents cited in *other* conversations that also
    cited these passages, with how many such conversations ("support").
    """
    threads = _threads_citing(chunk_ids, exclude_thread)
    if not threads:
        return []
    params = [*threads, *exclude_sources]
    exclude = f"AND source NOT IN ({_marks(len(exclude_sources))})" if exclude_sources else ""
    return _query(
        f"SELECT source, COUNT(DISTINCT thread_id) AS n FROM impressions WHERE cited = 1 "
        f"AND thread_id IN ({_marks(len(threads))}) {exclude} GROUP BY source ORDER BY n DESC, source",
        params,
    )


def co_asked_questions(chunk_ids: list[str], exclude_thread: str | None, exclude_question: str) -> list[tuple[str, int]]:
    """"People who asked about this also asked": questions from other conversations citing these passages."""
    threads = _threads_citing(chunk_ids, exclude_thread)
    if not threads:
        return []
    return _query(
        f"SELECT MIN(question), COUNT(DISTINCT thread_id) AS n FROM impressions "
        f"WHERE thread_id IN ({_marks(len(threads))}) AND lower(trim(question)) != lower(trim(?)) "
        "GROUP BY lower(trim(question)) ORDER BY n DESC, MIN(question)",
        [*threads, exclude_question],
    )


def popular_questions(limit: int) -> list[tuple[str, int]]:
    """Questions asked in the most conversations that got a cited answer and no net 👎."""
    return _query(
        """
        SELECT MIN(i.question), COUNT(DISTINCT i.thread_id) AS n
        FROM impressions i
        WHERE i.cited = 1
          AND COALESCE((SELECT SUM(f.vote) FROM feedback f
                        WHERE f.thread_id = i.thread_id AND f.chunk_id = i.chunk_id), 0) >= 0
        GROUP BY lower(trim(i.question))
        ORDER BY n DESC, MIN(i.question)
        LIMIT ?
        """,
        [limit],
    )


def training_rows() -> list[tuple[dict, int, int]]:
    """(features, position, label) for every shown passage that received a vote. 👍 = 1, 👎 = 0."""
    rows = _query(
        """
        SELECT i.features, i.position, f.vote
        FROM impressions i JOIN feedback f ON f.thread_id = i.thread_id AND f.chunk_id = i.chunk_id
        WHERE i.features IS NOT NULL
        """
    )
    return [(json.loads(feats), position, int(vote == 1)) for feats, position, vote in rows]
